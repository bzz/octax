"""Audit and report the frozen two-seed current-rollout pixel-RND pilot."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

from flax import serialization
import jax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from round_card import write_round_card
from summarize_credit_ablation import compare
from summarize_v2_baseline import wilson

ROOT = Path(__file__).resolve().parent
SERIES = ROOT/'runs/tetris/pixel_novelty_64k'


def load(path):
    return json.loads(path.read_text())


def metrics(run):
    with (run/'metrics.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(r['step']) for r in rows] == list(range(2048, 65537, 2048))
    assert all(np.isfinite(float(v)) for row in rows for v in row.values())
    return rows


def training(run):
    rows = metrics(run)
    return dict(summary=load(run/'summary.json'),
        reward_events=sum(int(r['positive_reward_steps']) for r in rows),
        completed_games=sum(int(r['done_count']) for r in rows),
        reward_event_rollout_ends=[int(r['step']) for r in rows if int(r['positive_reward_steps'])],
        final_entropy=float(rows[-1]['entropy']), final_action_distribution=[float(rows[-1][f'action_{i}_fraction']) for i in range(5)])


def exact(a, b):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(x, y)


def main():
    status = load(SERIES/'run_status.json')
    assert status['phase'] == 'complete' and status['total_new_training_steps'] == 131072
    assert status['separate_smoke_training_steps'] == 16384
    for name, digest in load(SERIES/'source_manifest.json').items():
        assert hashlib.sha256((SERIES/'source_snapshot'/name).read_bytes()).hexdigest() == digest
    records = {}; shared_random = None
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), constrained_layout=True)
    for column, seed in enumerate([1, 2]):
        run = ROOT/f'runs/tetris/pixel_novelty_seed{seed}'
        control = ROOT/f'runs/tetris/v2_credit_horizon_seed{seed}'
        cfg = OmegaConf.to_container(OmegaConf.load(run/'config.yaml'))
        old = OmegaConf.to_container(OmegaConf.load(control/'config.yaml'))
        allowed = {'run_dir', 'method', 'purpose', 'novelty', 'normalization_warmup_decisions'}
        differences = {k: [old.get(k), cfg.get(k)] for k in cfg.keys()|old.keys() if cfg.get(k) != old.get(k)}
        assert set(differences) == allowed
        assert cfg['novelty'] == dict(coefficient=.01, learning_rate=.0001, gamma=.99,
            variance_floor=.0001, features=128, intrinsic_value_coefficient=.5)
        assert cfg['purpose'] == 'discovery_pilot' and cfg['normalization_warmup_decisions'] == 0
        assert load(run/'run_status.json')['phase'] == 'complete'
        for name, digest in load(run/'source_manifest.json').items():
            assert hashlib.sha256((run/'source_snapshot'/name).read_bytes()).hexdigest() == digest
        full = serialization.msgpack_restore((run/'checkpoint_full.msgpack').read_bytes())
        exported = serialization.msgpack_restore((run/'checkpoint.msgpack').read_bytes())
        assert full['format_version'] == 2 and exported['purpose'] == 'inference_only'
        state = full['state']; ppo = state['ppo']
        assert int(ppo['global_step']) == int(exported['train_state']['global_step']) == 65536
        assert int(ppo['agent_ts']['step']) == int(state['intrinsic_value']['step']) == int(state['novelty']['predictor']['step']) == 2048
        assert int(state['rollouts']) == int(state['novelty']['rollouts']) == 32 and not state['pending_update']
        exact(ppo['agent_ts']['params'], exported['train_state']['agent_ts']['params'])
        assert full['config'] == exported['config'] == cfg
        initial = load(ROOT/f'runs/tetris/v2_replication_64k/initial_seed{seed}.json')
        preflight = load(run/'preflight.json')
        assert preflight['initial_parameter_sha256'] == initial['initial_parameter_sha256']
        assert preflight['unchanged_environment_and_base_ppo_sources']
        assert preflight['config_differences'] == differences
        target_hash = hashlib.sha256(serialization.to_bytes(jax.tree.map(np.asarray, state['novelty']['target_params']))).hexdigest()
        assert target_hash == preflight['target_sha256']
        evaluations = load(run/'evaluation.json')
        assert [e['step'] for e in evaluations] == [0, 65536]
        assert all(e['checkpoint_roundtrip_exact'] for e in evaluations)
        assert evaluations[0]['episodes'] == load(control/'evaluation.json')[0]['episodes']
        final, old_final = load(run/'heldout/comparison.json'), load(control/'heldout/comparison.json')
        assert final['seed_start'] == old_final['seed_start'] == initial['seed_start'] == 50000
        assert final['episodes_per_policy'] == old_final['episodes_per_policy'] == initial['episodes_per_policy'] == 128
        for mode in ['random', 'noop']:
            assert final[mode]['episodes'] == old_final[mode]['episodes']
        if shared_random is not None: assert final['random']['episodes'] == shared_random['episodes']
        shared_random = final['random']
        policies = dict(novelty=final['policy'], control=old_final['policy'],
            untrained=initial['policy'], random=final['random'])
        counts = {name: dict(scoring_games=sum(r > 0 for r in policy['episodes']['returns']),
            scoring_rate_ci95_wilson=wilson(sum(r > 0 for r in policy['episodes']['returns']), 128),
            **policy['summary']) for name, policy in policies.items()}
        comparisons = {name: compare(final['policy'], policy) for name, policy in policies.items() if name != 'novelty'}
        passes = all(comparisons[name]['mean_return_difference'] > 0 for name in ['random', 'untrained'])
        rows = metrics(run); train = training(run)
        assert train['summary']['target_unchanged'] and train['summary']['optimizer_updates'] == 2048
        assert train['reward_events'] == train['summary']['training_reward_events']
        assert train['completed_games'] == train['summary']['training_completed_games']
        assert all(float(r['clipped_grad_norm']) <= .500001 for r in rows)
        records[str(seed)] = dict(training=train, control_training=training(control),
            random_training=load(ROOT/f'runs/tetris/v2_baseline_seed{seed}/random_training_budget.json'),
            policies=counts, raw_policies=policies, paired_comparisons=comparisons,
            candidate_for_confirmation=passes, config_differences=differences)
        steps = [int(r['step']) for r in rows]
        for name, path, color in [('Pixel novelty', run, '#205c4f'), ('Score-only PPO', control, '#8066b3')]:
            current = metrics(path)
            axes[0, column].step([0, *steps], [0, *np.cumsum([int(r['positive_reward_steps']) for r in current])], where='post', label=name, color=color)
            axes[1, column].plot(steps, [float(r['entropy']) for r in current], label=name, color=color)
        axes[0, column].set_title(f'Seed {seed}: original-score events')
        axes[1, column].set_title(f'Seed {seed}: policy entropy')
        axes[1, column].set_ylim(0, np.log(5)*1.03)
        for key, label in [('extrinsic_advantage_std', 'Score advantage'), ('weighted_intrinsic_advantage_std', '0.01 × novelty advantage')]:
            axes[2, column].plot(steps, [float(r[key]) for r in rows], label=label)
        axes[2, column].set_title(f'Seed {seed}: advantages before mixing/normalization')
        axes[2, column].set_yscale('symlog', linthresh=.001)
        write_round_card(run, f'On-policy pixel novelty · seed {seed}',
            'Original-score exploratory comparison; see the two-seed report for interpretation.',
            [('Novelty / score-only scoring', f"{counts['novelty']['scoring_games']} / {counts['control']['scoring_games']} of 128"),
             ('Untrained / random scoring', f"{counts['untrained']['scoring_games']} / {counts['random']['scoring_games']} of 128"),
             ('Original training rewards', train['reward_events'])],
            [('Two-seed report', '../pixel_novelty_64k/RESULT.md'), ('Protocol', '../../../PIXEL_EXPLORATION_PROPOSAL.md'), ('Implementation log', '../../../IMPLEMENTATION_LOG.md')])
    for ax in axes.flat: ax.set_xlabel('Training decisions'); ax.set_xlim(0, 65536)
    axes[0, 0].legend(); axes[2, 0].legend(fontsize=8)
    fig.suptitle('Current-rollout PPO · pixel novelty versus original score only')
    fig.savefig(SERIES/'comparison_curves.png', dpi=150); plt.close(fig)
    passes = all(r['candidate_for_confirmation'] for r in records.values())
    conclusion = ('Both seeds exceed random and their untrained networks on the exploratory bank; unused-seed confirmation is still required.' if passes else
        'Pixel novelty does not pass the two-seed original-score screen; useful on-policy learning remains unestablished.')
    analysis = dict(conclusion=conclusion, candidate_for_confirmation=passes, records=records,
        separate_smoke_training_decisions=16384, pilot_training_decisions=131072,
        limitations=['Only two training seeds; reused evaluation bank is exploratory.',
            'Random controls are shared, not independent repeated evidence.',
            'Paired tests are unadjusted; a screen pass is not confirmation.',
            'This tests the complete auxiliary reward/value/predictor adaptation, not an isolated scalar coefficient.',
            'The predictor is trained on current-rollout pixels only; learned weights and normalization statistics persist online.'])
    (SERIES/'analysis.json').write_text(json.dumps(analysis, indent=2)+'\n')
    table = '\n'.join(f"| {seed} | {r['training']['reward_events']} / {r['control_training']['reward_events']} / {r['random_training']['positive_reward_events']} | {r['policies']['novelty']['scoring_games']}/128 | {r['policies']['control']['scoring_games']}/128 | {r['policies']['untrained']['scoring_games']}/128 | {r['policies']['random']['scoring_games']}/128 |" for seed, r in records.items())
    (SERIES/'RESULT.md').write_text(f'''# Pixel-only novelty · online, on-policy PPO at 64K

{conclusion}

Fresh seeds 1 and 2 each received 65,536 training decisions, 32 rollouts and 2,048 PPO updates. Actor observations remain four monochrome frames; no geometry, demonstrations or historical replay enters training. RND uses full newest-frame pixels, frozen bonuses per rollout, a fixed target and a predictor trained only on that current rollout. Original game score is the evaluation objective. The added value head and predictor each receive 2,048 optimizer updates; the shared CNN is also influenced by the intrinsic value loss. This is a complete exploration-method comparison, not a single-coefficient ablation.

| Seed | Training rewards: novelty / score-only / random | Novelty scoring | Score-only scoring | Untrained scoring | Shared random scoring |
| --- | ---: | ---: | ---: | ---: | ---: |
{table}

Final scoring uses the reused 128-game bank, seeds 50000–50127. Confidence intervals, paired exploratory comparisons, survival/action statistics, reward-discovery timing and exact control arrays are in `analysis.json`. Random is one shared control. The separate 16,384 smoke-training decisions and smoke evaluation decisions are recorded in `smoke_validation.json`; they are excluded from the 131,072 pilot-training decisions. Each run's summary records its monitoring evaluation decisions separately. Final comparison and illustration games are additional evaluation work, never policy updates. No normalization warm-up was used.

Initial parameter hashes and initial development arrays matched controls. Frozen source hashes, complete environment/update budgets, original-score logs, full-checkpoint restoration, fixed-target hashes, inference-export equality and repeated random/no-op arrays passed. No novelty or predictor metric by itself is learning evidence. The fresh confirmation bank remains unused.

![Original score, entropy and advantage diagnostics](comparison_curves.png)

[Actual-policy card](result_card.html) · [Protocol](../../../PIXEL_EXPLORATION_PROPOSAL.md) · [Implementation log](../../../IMPLEMENTATION_LOG.md)
''')
    # The aggregate illustration was preselected as training seed 1/game 30016.
    source = ROOT/'runs/tetris/pixel_novelty_seed1/visualizations/policy_trajectory.npz'
    (SERIES/'visualizations').mkdir(exist_ok=True)
    shutil.copyfile(source, SERIES/'visualizations/policy_trajectory.npz')
    write_round_card(SERIES, 'On-policy pixel novelty · two seeds', conclusion,
        [('Novelty scoring · seeds 1 / 2', ' / '.join(str(records[str(s)]['policies']['novelty']['scoring_games']) for s in [1, 2])+' of 128'),
         ('Score-only scoring · seeds 1 / 2', ' / '.join(str(records[str(s)]['policies']['control']['scoring_games']) for s in [1, 2])+' of 128'),
         ('Shared random scoring', f"{records['1']['policies']['random']['scoring_games']} of 128"),
         ('Training decisions per seed', '65,536')],
        [('Full report', 'RESULT.md'), ('Protocol', '../../../PIXEL_EXPLORATION_PROPOSAL.md'), ('Implementation log', '../../../IMPLEMENTATION_LOG.md')])
    log = ROOT/'IMPLEMENTATION_LOG.md'
    heading = '## 2026-11-23 — Two-seed on-policy pixel-novelty pilot completed'
    if heading not in log.read_text():
        with log.open('a') as stream:
            stream.write(f"\n{heading}\n\n{conclusion} Original-score scoring games by seed: "
                f"{ {s: r['policies']['novelty']['scoring_games'] for s, r in records.items()} }. "
                'Exact initialization, frozen sources, complete budgets, full checkpoints, unchanged target and original-score controls passed. '
                'This controlled shaping experiment retains pixel-only inputs and strictly current-rollout updates. '
                'Full evidence, limitations and actual-policy cards: `runs/tetris/pixel_novelty_64k/RESULT.md`.\n')
    print(json.dumps(dict(conclusion=conclusion, scoring={s: r['policies']['novelty']['scoring_games'] for s, r in records.items()}), indent=2))


if __name__ == '__main__':
    main()
