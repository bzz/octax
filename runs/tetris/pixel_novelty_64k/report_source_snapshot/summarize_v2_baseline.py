"""Write an evidence-backed summary of the completed, fixed-budget v2 baseline."""
import csv
import hashlib
import json
import math
from pathlib import Path
from scipy.stats import binomtest

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/tetris/v2_baseline_seed0'


def wilson(k,n):
    z=1.959963984540054
    d=1+z*z/n
    center=(k/n+z*z/(2*n))/d
    half=z*math.sqrt((k/n)*(1-k/n)/n+z*z/(4*n*n))/d
    return [max(0.,center-half),min(1.,center+half)]


def main():
    state=json.loads((RUN/'run_status.json').read_text())
    if state['phase']!='complete':
        raise RuntimeError('Wait for training, held-out evaluation and rendering to finish.')
    summary=json.loads((RUN/'summary.json').read_text())
    evaluation=json.loads((RUN/'heldout/comparison.json').read_text())
    random_budget=json.loads((RUN/'random_training_budget.json').read_text())
    with (RUN/'metrics.csv').open() as f: rows=list(csv.DictReader(f))
    assert int(rows[-1]['step'])==summary['step']==65536
    assert len(rows)==128
    rewards=sum(int(row['positive_reward_steps']) for row in rows)
    reward_sum=sum(float(row['reward_sum']) for row in rows)
    ended=sum(int(row['done_count']) for row in rows)
    events=[{'rollout_end_step':int(row['step']),'positive_reward_events':int(row['positive_reward_steps']),
             'reward_sum':float(row['reward_sum'])} for row in rows if int(row['positive_reward_steps'])]
    policies={}
    for key in ['policy','random','noop']:
        data=evaluation[key]
        returns=data['episodes']['returns'];n=len(returns);scoring=sum(x>0 for x in returns)
        assert n==128
        policies[key]={'episodes':n,'scoring_games':scoring,'mean_return':sum(returns)/n,
                       'scoring_rate_ci95_wilson':wilson(scoring,n),**data['summary']}
    a=evaluation['policy']['episodes']['returns'];b=evaluation['random']['episodes']['returns']
    wins=sum(x>0 and y<=0 for x,y in zip(a,b))
    losses=sum(y>0 and x<=0 for x,y in zip(a,b))
    discordant=wins+losses
    p=float(binomtest(wins,discordant,.5).pvalue) if discordant else 1.
    analysis=dict(training_steps=65536,positive_training_events=rewards,training_reward_sum=reward_sum,
                  completed_training_episodes=ended,reward_event_rollouts=events,random_equal_budget=random_budget,
                  evaluation_seed_start=evaluation['seed_start'],policies=policies,
                  paired_scoring_comparison={'ppo_only':wins,'random_only':losses,'exact_mcnemar_p':p},
                  limitation='One training seed; paired test concerns scoring probability, not generalization across training seeds.')
    (RUN/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
    table='\n'.join(f"| {name} | {policies[key]['scoring_games']}/128 | {policies[key]['mean_return']:.5f} | {policies[key]['mean_length']:.1f} | {policies[key]['truncated']} |" for name,key in [('PPO','policy'),('Random','random'),('No-op','noop')])
    if p < .05 and wins>losses:
        conclusion='This fixed policy shows a detectable scoring-rate advantage on this evaluation set. It remains a one-training-seed result and needs replication.'
    else:
        conclusion='This evaluation does not establish a scoring advantage over random play.'
    first=f"in the rollout ending at step {events[0]['rollout_end_step']:,}" if events else 'not observed'
    text=f'''# Fresh v2 Tetris baseline: 65,536 steps

The run is complete. {conclusion}

## Reward discovery during training

- PPO: **{rewards} positive reward events**, total original game reward {reward_sum:g}, over 65,536 steps and {ended} completed episodes. First event: {first}.
- Uniform-random control: **{random_budget['positive_reward_events']} positive reward events**, total reward {random_budget['reward_sum']:g}, over the same 65,536-step budget and {random_budget['completed_episodes']} completed episodes.
- The random control matches initial reset states, seed, parallelism and step budget; action draws and episode boundaries are not identical. This is one realization per condition, not a replicated estimate of event frequency.
- The value of the fixes is evaluated separately from proving learned skill. Correct timer behavior and stable reward-free updates are verified engineering results; improved Tetris performance requires score evidence.

## Corrected-timing final comparison

All three policies use `timer_mode=frame`, the same action space, the same raw score reward and 128 matched seeds 50000–50127. These evaluation seeds were selected before observing this run and were unused in the earlier experiments.

| Policy | Games scoring | Mean score | Mean episode decisions | Time-limit truncations |
| --- | ---: | ---: | ---: | ---: |
{table}

There were {wins} PPO-only and {losses} random-only scoring seed pairs. The exact paired test for scoring probability gives p={p:.4g}. This test does not establish robustness across independently trained agents, nor test a difference in mean score magnitude. Per-policy Wilson intervals and all raw counts are retained in `analysis.json`; per-game results are in `heldout/comparison.json`.

## Training checks and timing

- Fresh seed-0 initialization; no resumed weights or transferred encoder.
- 16 environments × 32 steps, four minibatches/epochs, unchanged gamma=0.99 and lambda=0.95.
- Pixel observations and original score-only reward; no board features, shaping or action-repeat change.
- Corrected timers, zero value output initialization, advantage std floor 0.01 and global-norm clip 0.5.
- Final entropy: {float(rows[-1]['entropy']):.6f}; final approximate KL: {float(rows[-1]['approx_kl']):.6g}.
- All recorded diagnostics/parameters were checked finite during training. Saved checkpoint state and inference reloaded exactly.
- Observed training throughput: {summary['training_steps_per_second']:.1f} steps/s. Training plus final development evaluation/export: {summary['wall_training_and_periodic_eval_seconds']/60:.1f} minutes, excluding initial evaluation and the final 128-game comparison. The equal-budget random control briefly shared the CPU, so this is practical throughput rather than a hardware benchmark.

## Outputs and limits

[Learning curves](visualizations/learning_curves.png) · [Gameplay](visualizations/gameplay.gif) · [Checkpoint](checkpoint.msgpack) · [Raw comparison](heldout/comparison.json) · [Analysis](analysis.json)

The curves use the eight-game corrected-timing development baseline. The table above is the separate final 128-game comparison. One displayed gameplay seed is illustrative, not a substitute for aggregate evaluation. Historical v1 results used different timer dynamics and must not be pooled with this experiment.

No extension, additional training seed or credit-horizon ablation was launched. The prepared longer-credit profile remains a separate next experiment.
'''
    (RUN/'RESULT.md').write_text(text)
    entry=f'''\n## 2026-11-23 — Fresh 64K v2 baseline completed\n\nCompleted 65,536 CPU environment steps from fresh seed 0 at {summary['training_steps_per_second']:.1f} observed steps/s. PPO encountered {rewards} positive reward events (total reward {reward_sum:g}) in {ended} completed episodes; the equal-budget uniform-random control encountered {random_budget['positive_reward_events']} in {random_budget['completed_episodes']} episodes. First PPO reward: {first}. The control matches initial reset state and experience budget, not exact action RNG paths/episode boundaries.\n\nFinal evaluation used 128 matched, previously unused seeds 50000–50127 under corrected frame timing. PPO scored in {policies['policy']['scoring_games']}/128 games (mean {policies['policy']['mean_return']:.5f}); random scored in {policies['random']['scoring_games']}/128 (mean {policies['random']['mean_return']:.5f}); no-op scored in {policies['noop']['scoring_games']}/128. Exact paired scoring-rate test p={p:.4g}. {conclusion} This is one training seed and does not establish reproducibility across trained agents.\n\nAll training checks and exact checkpoint reload passed. Final policy entropy was {float(rows[-1]['entropy']):.6f}. The runner exported matched-timing learning curves and MinAtar-style gameplay. The run folder contains raw episode arrays, reward-event rollout indices, scoring-rate intervals, source snapshots and `RESULT.md`. No further training or horizon ablation was launched.\n'''
    log=ROOT/'IMPLEMENTATION_LOG.md'
    if '## 2026-11-23 — Fresh 64K v2 baseline completed' not in log.read_text():
        with log.open('a') as f:f.write(entry)
    results=ROOT/'RESULTS.md'
    if '## Fresh 64K v2 baseline result' not in results.read_text():
        with results.open('a') as f:
            f.write(f"\n## Fresh 64K v2 baseline result\n\nCompleted 65,536 fresh v2 steps. Training positive rewards: {rewards}; equal-budget random: {random_budget['positive_reward_events']}. Final matched corrected-timing evaluation: PPO {policies['policy']['scoring_games']}/128 scoring games, random {policies['random']['scoring_games']}/128. {conclusion}\n\n[Full v2 result](runs/tetris/v2_baseline_seed0/RESULT.md).\n")
    manifest=json.loads((RUN/'source_manifest.json').read_text())
    for name in ['compare_random_budget.py','summarize_v2_baseline.py']:
        data=(ROOT/name).read_bytes();manifest[name]=hashlib.sha256(data).hexdigest()
        (RUN/'source_snapshot'/name).write_bytes(data)
    (RUN/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(analysis,indent=2))


if __name__=='__main__':
    main()
