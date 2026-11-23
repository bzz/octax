"""Training-seed-aware comparison of the six matched 64K Tetris runs."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from omegaconf import OmegaConf
from summarize_credit_ablation import training, compare
from round_card import write_round_card

ROOT = Path(__file__).resolve().parent
SERIES = ROOT / 'runs/tetris/v2_replication_64k'
CONDITIONS = ['baseline', 'credit_horizon']
SEEDS = [0, 1, 2]


def load(path):
    return json.loads(path.read_text())


def run_path(condition, seed):
    return ROOT / f'runs/tetris/v2_{condition}_seed{seed}'


def main():
    status = load(SERIES/'run_status.json')
    assert status['phase'] == 'complete' and status['total_new_training_steps'] == 262144
    original = load(run_path('baseline', 0)/'heldout/comparison.json')
    controls = {mode: original[mode] for mode in ['random', 'noop']}
    random_mean = controls['random']['summary']['mean_return']
    random_games = sum(x > 0 for x in controls['random']['episodes']['returns'])
    records = []
    initial = {seed: load(SERIES/f'initial_seed{seed}.json') for seed in SEEDS}
    random_budget = {seed: load(run_path('baseline', seed)/'random_training_budget.json') for seed in SEEDS}
    for seed in SEEDS:
        assert initial[seed]['training_seed'] == seed
        assert initial[seed]['initial_development_arrays_exact']
        assert initial[seed]['seed_start'] == 50000 and initial[seed]['episodes_per_policy'] == 128
        assert random_budget[seed]['total_steps'] == 65536 and random_budget[seed]['timer_mode'] == 'frame'
        initial_dev = [load(run_path(condition, seed)/'evaluation.json')[0]['episodes'] for condition in CONDITIONS]
        assert initial_dev[0] == initial_dev[1], f'Initial policy mismatch for seed {seed}'
        for condition in CONDITIONS:
            run = run_path(condition, seed)
            cfg = OmegaConf.to_container(OmegaConf.load(run/'config.yaml'))
            reference = OmegaConf.to_container(OmegaConf.load(run_path(condition, 0)/'config.yaml'))
            assert all(cfg[k] == v for k, v in reference.items() if k not in ['seed', 'run_dir'])
            assert cfg['seed'] == seed and cfg['total_timesteps'] == 65536
            assert cfg['resume'] is None and cfg['encoder_checkpoint'] is None
            evaluation = load(run/'heldout/comparison.json')
            assert evaluation['seed_start'] == 50000 and evaluation['episodes_per_policy'] == 128
            for mode in controls:
                assert evaluation[mode]['episodes'] == controls[mode]['episodes'], (condition, seed, mode)
            train = training(run)
            assert train['rollouts'] == (128 if condition == 'baseline' else 32)
            assert all(e['checkpoint_roundtrip_exact'] for e in load(run/'evaluation.json'))
            policy = evaluation['policy']; returns = policy['episodes']['returns']
            untrained = initial[seed]['policy']; start = untrained['episodes']['returns']
            records.append(dict(condition=condition, seed=seed, training=train,
                scoring_games=sum(x > 0 for x in returns), mean_return=float(np.mean(returns)),
                initial_scoring_games=sum(x > 0 for x in start), initial_mean_return=float(np.mean(start)),
                return_gain_over_initial=float(np.mean(returns)-np.mean(start)),
                mean_length=policy['summary']['mean_length'], truncated=policy['summary']['truncated'],
                paired_final_vs_initial=compare(policy, untrained),
                paired_final_vs_random=compare(policy, controls['random']),
                returns=returns, action_distribution=policy['summary']['action_distribution']))
    by_condition = {}
    for condition in CONDITIONS:
        rows = sorted([r for r in records if r['condition'] == condition], key=lambda r: r['seed'])
        means = [r['mean_return'] for r in rows]
        gain = [r['return_gain_over_initial'] for r in rows]
        reward_seeds = sum(r['training']['positive_reward_events'] > 0 for r in rows)
        gate = reward_seeds >= 2 and np.mean(means) > random_mean and np.mean(gain) > 0 and sum(x > 0 for x in gain) >= 2
        by_condition[condition] = dict(training_seeds=3, seeds_discovering_rewards=reward_seeds,
            training_reward_events=[r['training']['positive_reward_events'] for r in rows],
            scoring_games=[r['scoring_games'] for r in rows],
            mean_return_across_training_seeds=float(np.mean(means)),
            std_return_across_training_seeds=float(np.std(means, ddof=1)),
            mean_return_gain_over_initial=float(np.mean(gain)),
            scoring_gain_over_initial=[r['scoring_games']-r['initial_scoring_games'] for r in rows],
            screening_gate_passed=bool(gate))
    paired = []
    for seed in SEEDS:
        b = next(r for r in records if r['seed'] == seed and r['condition'] == 'baseline')
        c = next(r for r in records if r['seed'] == seed and r['condition'] == 'credit_horizon')
        paired.append(dict(seed=seed, credit_minus_baseline_mean_return=c['mean_return']-b['mean_return'],
                           credit_minus_baseline_reward_events=c['training']['positive_reward_events']-b['training']['positive_reward_events']))
    candidates = [name for name, summary in by_condition.items() if summary['screening_gate_passed']]
    conclusion = ('A configuration passes the exploratory screening gate; freeze it and test on new evaluation seeds before claiming a learning setup.'
                  if candidates else 'Neither configuration passes the exploratory learning gate at 64K steps; useful scoring improvement is not established.')
    analysis = dict(records=records, conditions=by_condition, paired_training_seed_differences=paired,
                    random_training_budgets=random_budget, random_evaluation=controls['random'],
                    noop_evaluation=controls['noop'], initial_policies=initial, candidates=candidates,
                    conclusion=conclusion, evaluation_seed_start=50000, episodes_per_policy=128,
                    total_training_steps_per_condition=196608, total_new_training_steps=262144,
                    limitations=['Three independent training seeds per condition, not 384 independent trained agents.',
                                 'All policies use the same previously inspected evaluation seeds.',
                                 'Repeated uniform-random evaluations are one shared 128-game control, not six independent samples.',
                                 'The screening gate is a decision rule, not a significance test or goal-completion criterion.',
                                 'Rollout, gamma and lambda differ together; causes are not isolated.'])
    (SERIES/'analysis.json').write_text(json.dumps(analysis, indent=2)+'\n')
    names = {'baseline': 'v2 baseline', 'credit_horizon': 'Longer credit'}
    table = '\n'.join(f"| {names[r['condition']]} | {r['seed']} | {r['training']['positive_reward_events']} | {r['training']['reward_event_rollouts'][0]['rollout_end_step'] if r['training']['reward_event_rollouts'] else '—'} | {r['scoring_games']}/128 | {r['initial_scoring_games']}/128 | {r['mean_return']:.5f} | {r['training']['final_entropy']:.4f} |" for r in records)
    summary_table = '\n'.join(f"| {names[name]} | {s['seeds_discovering_rewards']}/3 | {s['mean_return_across_training_seeds']:.5f} | {s['std_return_across_training_seeds']:.5f} | {s['mean_return_gain_over_initial']:+.5f} |" for name, s in by_condition.items())
    text = f'''# Three-seed Tetris replication · 64K per run

{conclusion}

## Fixed protocol

Both configurations were trained from fresh seeds 0, 1 and 2 for exactly 65,536 environment decisions each. Seeds 1 and 2 are the four new runs; seed 0 results are reused. The model remains a four-frame monochrome CNN with original game-score rewards and corrected per-frame timers. No board heuristics, shaping, recurrence or transfer were introduced.

Baseline: rollout 32, gamma 0.99, lambda 0.95, four minibatches. Longer credit: rollout 128, gamma 0.995, lambda 0.98, sixteen minibatches. Both use 16 environments, four epochs, minibatch size 128 and 2,048 optimizer steps per run. Training sources/ROM are snapshotted; the sequential runner rejects source drift.

## Reward discovery and scores by training seed

The first-event column is the **end of the rollout containing the first reward**, not its exact action index. Initial-network and final-policy scoring use the same 128 evaluation seeds, 50000–50127.

| Configuration | Training seed | Training reward events | First event rollout | Final scored | Untrained scored | Final mean score | Final entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

Uniform random scored in **{random_games}/128** games (mean {random_mean:.5f}); no-op scored in 0/128. The random control is identical across all evaluations and must not be multiplied into a larger sample. Equal-budget random reward-event counts for training seeds 0/1/2: {[random_budget[s]['positive_reward_events'] for s in SEEDS]}.

| Configuration | Training seeds finding rewards | Mean score across seeds | Between-seed SD | Mean score gain over untrained |
| --- | ---: | ---: | ---: | ---: |
{summary_table}

The three training seeds, rather than individual game episodes from a shared policy, are the replication units. Episode-paired tests are retained per seed in `analysis.json` as exploratory descriptions; they are not treated as proof across a population of trained networks. With only three training seeds and previously inspected game seeds, a small difference is weak evidence.

## Learning decision

The screening rule requires rewards in at least two training seeds, a condition's average score above random and above its own untrained networks, and positive score gains in at least two seeds. This rule chooses whether a frozen configuration warrants fresh-seed evaluation; it is not proof of useful learning. Candidates: {', '.join(candidates) if candidates else 'none'}.

The project goal remains a saved pixel-based network with a convincing scoring advantage over random, not merely changed weights, nonzero loss or a completed run. If a candidate passes screening, evaluate all three trained policies of that configuration on a new fixed 512-game bank starting at seed 60000. If neither passes, investigate the shortage of useful experience before extending the same setup indiscriminately.

## Validation and artifacts

All six configurations match their arm's settings except training seed and output path. Initial development arrays match between arms within each seed. Reconstructed initial networks reproduce their stored development evaluations exactly. All repeated random/no-op per-game arrays match; checkpoint roundtrips and finite diagnostics passed. Full per-run metrics, final returns, action distributions and termination/truncation flags remain in the run directories.

[Analysis](analysis.json) · [Result card](result_card.html) · [Implementation log](../../../IMPLEMENTATION_LOG.md) · [Protocol](../../../EXPERIMENT.md) · [Seed-0 ablation](../v2_credit_horizon_seed0/RESULT.md)

The card shows the **preselected longer-credit training seed 1**, gameplay seed 30016, rather than a seed selected for its final score. It is illustrative and does not replace the aggregate table.
'''
    (SERIES/'RESULT.md').write_text(text)
    (SERIES/'visualizations').mkdir(exist_ok=True)
    shutil.copy2(run_path('credit_horizon', 1)/'visualizations/policy_trajectory.npz', SERIES/'visualizations/policy_trajectory.npz')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    links = [('Full result', 'RESULT.md'), ('Implementation log', '../../../IMPLEMENTATION_LOG.md'),
             ('Code changes', f'https://github.com/bzz/octax/compare/7b0445da131cd16f2270d06089d70b7b93c3358d...{commit}')]
    write_round_card(SERIES, 'Tetris · three seeds · 64K each', conclusion,
                     [(names[name]+' scored (seeds 0/1/2)', ' / '.join(map(str,s['scoring_games']))+' of 128 each') for name,s in by_condition.items()]+[('Uniform random scored',f'{random_games}/128')], links)
    heading = '## 2026-11-23 — Three-seed 64K replication completed'
    entry = f'''\n{heading}\n\nCompleted the four new seed-1/2 runs at 65,536 steps each, with unchanged pixel-only observations, original score reward and corrected timing. Including seed 0, baseline training reward events were {by_condition['baseline']['training_reward_events']}; longer credit events were {by_condition['credit_horizon']['training_reward_events']}. Final scoring games (out of 128 per policy) were baseline {by_condition['baseline']['scoring_games']} and longer credit {by_condition['credit_horizon']['scoring_games']}, versus {random_games}/128 for the shared random control.\n\nCompared each final policy with its exactly reconstructed untrained network on the same game seeds. Mean score gain over initialization: baseline {by_condition['baseline']['mean_return_gain_over_initial']:+.5f}, longer credit {by_condition['credit_horizon']['mean_return_gain_over_initial']:+.5f}. {conclusion} Three training seeds and reused evaluation seeds do not establish broad generalization; duplicated random evaluations are not independent control samples.\n\nConfiguration, initial-policy, repeated-control, finite-metric and checkpoint checks passed. The run directory `runs/tetris/v2_replication_64k/` contains the full comparison and a compact actual-gameplay card using the preselected longer-credit seed-1 policy. The goal of a demonstrably useful learning setup remains subject to fresh evidence; the screening gate is not a completion criterion.\n'''
    for name in ['IMPLEMENTATION_LOG.md','RESULTS.md']:
        p = ROOT/name
        if heading not in p.read_text():
            p.open('a').write(entry)
    sources = {}
    for name in ['summarize_replication_64k.py','evaluate_initial_policy.py','round_card.py','summarize_credit_ablation.py','summarize_v2_baseline.py']:
        data = (ROOT/name).read_bytes(); sources[name] = hashlib.sha256(data).hexdigest()
        dest = SERIES/'report_source_snapshot'/name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(data)
    (SERIES/'report_source_manifest.json').write_text(json.dumps(sources,indent=2)+'\n')
    print(json.dumps({'conditions':by_condition,'candidates':candidates,'conclusion':conclusion},indent=2))


if __name__ == '__main__':
    main()
