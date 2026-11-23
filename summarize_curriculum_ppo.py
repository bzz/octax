"""Compare completed fresh PPO against each exact offline initialization."""
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

from experiment_utils import write_json
from policy_initialization import read_policy_weights
from round_card import write_round_card
from summarize_credit_ablation import training, compare

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def scoring(result):
    return sum(x > 0 for x in result['episodes']['returns'])


def check_sources(directory):
    for name, digest in read(directory/'source_manifest.json').items():
        assert hashlib.sha256((directory/'source_snapshot'/name).read_bytes()).hexdigest() == digest


def main():
    series = ROOT/'runs/tetris/human_bc_100epoch_ppo_64k'
    assert read(series/'run_status.json')['phase'] == 'complete'
    check_sources(series)
    results = {}
    for seed in [1, 2]:
        run = ROOT/f'runs/tetris/human_bc_100epoch_ppo_seed{seed}'
        parent = ROOT/f'runs/tetris/human_bc_100epoch_seed{seed}/pretrain'
        assert read(run/'run_status.json')['phase'] == 'complete'
        check_sources(run)
        audit = read(run/'handoff_audit.json')
        assert audit['initial_episode_arrays_exact'] and audit['training_decisions'] == 65536 and audit['optimizer_updates'] == 2048
        assert read(run/'evaluation.json')[0]['episodes'] == read(parent/'pretrained_evaluation.json')['episodes']
        assert all(e['checkpoint_roundtrip_exact'] for e in read(run/'evaluation.json'))
        full = serialization.msgpack_restore((run/'checkpoint.msgpack').read_bytes())
        assert int(full['train_state']['global_step']) == 65536 and int(full['train_state']['agent_ts']['step']) == 2048
        weights, provenance = read_policy_weights(run/'initialization_weights.msgpack', audit['parent_weights_sha256'])
        assert hashlib.sha256((parent/'policy_weights.msgpack').read_bytes()).hexdigest() == audit['parent_weights_sha256']
        after = full['train_state']['agent_ts']['params']['params']
        policy_after = {'params': {name: after[name] for name in ['features', 'actor']}}
        assert jax.tree.structure(weights) == jax.tree.structure(policy_after)
        deltas = [np.asarray(a, dtype=np.float64)-np.asarray(b, dtype=np.float64)
            for a, b in zip(jax.tree.leaves(policy_after), jax.tree.leaves(weights))]
        delta = float(np.sqrt(sum(np.sum(x*x) for x in deltas)))
        train = training(run)
        assert train['steps'] == 65536 and train['rollouts'] == 32
        assert train['max_clipped_grad_norm'] <= .500001
        if train['positive_reward_events'] == 0:
            assert delta == 0, 'Entropy-zero reward-free policy should remain unchanged'
        current = read(run/'heldout/comparison.json')
        before = read(parent/'heldout/comparison.json')
        scratch = read(ROOT/f'runs/tetris/v2_entropy_zero_seed{seed}/heldout/comparison.json')
        credit = read(ROOT/f'runs/tetris/v2_credit_horizon_seed{seed}/heldout/comparison.json')
        initial = read(ROOT/f'runs/tetris/v2_replication_64k/initial_seed{seed}.json')
        for other in [current, before, scratch, credit, initial]:
            assert other['seed_start'] == 50000 and other['episodes_per_policy'] == 128
            for mode in ['random', 'noop']:
                if mode in other:
                    assert other[mode]['episodes'] == current[mode]['episodes']
        policies = dict(ppo=current['policy'], pretrained=before['policy'], scratch_entropy_zero=scratch['policy'],
            scratch_entropy_001=credit['policy'], untrained=initial['policy'], random=current['random'], noop=current['noop'])
        scores = {name: scoring(value) for name, value in policies.items()}
        candidate = scores['ppo'] > scores['pretrained'] and scores['ppo'] > scores['random']
        results[seed] = dict(scores=scores, candidate_screen=candidate, training=train,
            encoder_actor_parameter_change_l2=delta, parent=provenance, handoff_audit=audit,
            checkpoint_sha256=hashlib.sha256((run/'checkpoint.msgpack').read_bytes()).hexdigest(),
            policies={name: value['summary'] for name, value in policies.items()},
            paired_comparisons={name: compare(current['policy'], value) for name, value in
                [('pretrained', before['policy']), ('random', current['random']), ('scratch_entropy_zero', scratch['policy'])]})
    candidate = all(r['candidate_screen'] for r in results.values())
    conclusion = ('Both PPO seeds improve on their own offline policy and random in the exploratory scoring screen; independent confirmation is still required.' if candidate else
        'Fresh PPO does not establish a replicated scoring gain beyond offline initialization and random play.')
    analysis = dict(conclusion=conclusion, candidate_screen=candidate, seeds=results,
        total_fresh_ppo_decisions=131072, total_ppo_optimizer_updates=4096,
        offline_updates_including_reference_fits=4320, offline_example_presentations=537600,
        original_human_decisions=2240, shared_human_prefix=True, shared_game_bank=True,
        reserved_confirmation_bank_used=False, offline_fit_data_replayed_during_ppo=False,
        limitations=['Two initialization seeds; both share one incomplete human prefix and the same exploratory game bank.',
            'Do not pool columns into 256 independent game seeds or attribute offline gains to PPO.',
            'Paired exact-test p-values are unadjusted exploratory diagnostics after multiple investigations.',
            'Raw human recording stays local; published weights suffice for PPO and gameplay reproduction.'])
    write_json(series/'analysis.json', analysis)
    table = '\n'.join(f'| {name} | {results[1]["scores"][name]}/128 | {results[2]["scores"][name]}/128 |'
        for name in ['ppo', 'pretrained', 'scratch_entropy_zero', 'scratch_entropy_001', 'untrained', 'random', 'noop'])
    discoveries = '\n'.join(f'| {seed} | {r["training"]["positive_reward_events"]} | {r["training"]["completed_episodes"]} | '
        f'{r["encoder_actor_parameter_change_l2"]:.4f} |'
        for seed, r in results.items())
    (series/'RESULT.md').write_text(f'''# Fresh on-policy PPO after offline cloning: two seeds at 64K

**{conclusion}**

| Policy | Training seed 1 | Training seed 2 |
| --- | ---: | ---: |
{table}

The offline advantage failed to replicate (7/128 and 1/128 versus shared random
3/128). The bounded PPO protocol was declared before seed 2's gameplay result,
kept both fixed 100-epoch initializations, and did not select a better duration
per seed. Initial eight-game arrays exactly match each offline policy. The
comparison above separates the starting-policy benefit from changes due to RL.

## Actual online training

Each independent process reads only the weights artifact, constructs fresh
optimizer/critic/RNG/environment state and collects 65,536 current-policy
decisions. There is no demonstration loader, replay, auxiliary loss or intrinsic
reward in PPO. Entropy regularization is zero, but collection/evaluation still
sample actions. The matched scratch entropy-zero control has identical settings
apart from initialization/provenance/output directory. Full source, configuration,
checkpoint, counter, initial-evaluation and random/no-op control audits pass.

| Seed | Positive training reward events | Completed training games | Encoder/actor L2 change |
| --- | ---: | ---: | ---: |
{discoveries}

Reward timing, first-reward loss/advantage context, action distributions, finite
gradient checks, full metrics and paired exploratory comparisons are retained
in [analysis.json](analysis.json). Parameter movement or discovered rewards alone
are not proof of better play. Both seeds complete 32 rollouts and 2,048 optimizer
updates; this is 131,072 fresh decisions and 4,096 PPO updates combined.

The offline experiments additionally used 4,320 updates and 537,600 presentations,
including both 20-epoch references and fresh 100-epoch fits. They reuse only 2,240
human decisions from one incomplete prefix. Validation replays and evaluation
steps are separately recorded, not counted as fresh training experience. Raw
human data remains local, but all PPO inputs are published weights/configs.

## Limits and next decision

The two columns share the same 128 exploratory game seeds; they are not 256
independent games. Paired tests are unadjusted diagnostics after many investigations,
not confirmatory evidence. No checkpoint was chosen by its score and the reserved
60000–60511 confirmation bank remains unused. The candidate screen requires both
PPO seeds to exceed their own initialization and random; its outcome is
`{candidate}`. Do not mark the overall goal complete based on this screen alone.

The animation is the preselected seed-1/game-30016 final PPO trajectory.
[Card](result_card.html) · [Protocol](../../../CURRICULUM_PPO_EXPERIMENT.md) ·
[Offline comparison](../human_bc_duration_100epoch/RESULT.md) ·
[Implementation log](../../../IMPLEMENTATION_LOG.md).
''')
    fig, ax = plt.subplots(figsize=(7, 3.5), layout='constrained')
    for seed, result in results.items():
        events = result['training']['reward_event_rollouts']
        points = [0]+[e['rollout_end_step'] for e in events]+[65536]
        rewards = [0]+list(np.cumsum([e['reward_sum'] for e in events]))
        rewards.append(rewards[-1])
        ax.step(points, rewards, where='post', label=f'Seed {seed}')
    ax.set(xlabel='Fresh on-policy decisions', ylabel='Cumulative original game reward', title='Reward discovery during PPO')
    ax.legend(); fig.savefig(series/'reward_discovery.png', dpi=160); plt.close(fig)
    (series/'visualizations').mkdir(exist_ok=True)
    shutil.copyfile(ROOT/'runs/tetris/human_bc_100epoch_ppo_seed1/visualizations/policy_trajectory.npz', series/'visualizations/policy_trajectory.npz')
    write_round_card(series, 'Offline initialization → on-policy PPO', conclusion,
        [('Seed 1: PPO / pretrained', f'{results[1]["scores"]["ppo"]} / {results[1]["scores"]["pretrained"]} scoring games'),
         ('Seed 2: PPO / pretrained', f'{results[2]["scores"]["ppo"]} / {results[2]["scores"]["pretrained"]} scoring games'),
         ('Shared random', '3/128 games'), ('Fresh PPO decisions', '2 × 65,536'),
         ('Replay during PPO', 'None')],
        [('Report', 'RESULT.md'), ('Evidence', 'analysis.json'),
         ('Protocol', '../../../CURRICULUM_PPO_EXPERIMENT.md'), ('Implementation log', '../../../IMPLEMENTATION_LOG.md')])
    manifest = {}
    for name in ['summarize_curriculum_ppo.py', 'summarize_credit_ablation.py', 'summarize_v2_baseline.py',
        'round_card.py', 'experiment_utils.py', 'policy_initialization.py']:
        data = (ROOT/name).read_bytes(); target = series/'report_source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
        manifest[name] = hashlib.sha256(data).hexdigest()
    write_json(series/'report_source_manifest.json', manifest)
    print(conclusion, {seed: r['scores'] for seed, r in results.items()})


if __name__ == '__main__':
    main()
