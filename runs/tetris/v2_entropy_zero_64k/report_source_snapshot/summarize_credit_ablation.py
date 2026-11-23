"""Compare the completed credit-horizon ablation with saved v2 controls."""
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy.stats import binomtest
from round_card import write_round_card
from summarize_v2_baseline import wilson

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/tetris/v2_credit_horizon_seed0'
BASE = ROOT / 'runs/tetris/v2_baseline_seed0'


def load(path):
    return json.loads(path.read_text())


def training(run):
    with (run / 'metrics.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert int(rows[-1]['step']) == 65536
    assert all(np.isfinite(float(v)) for row in rows for v in row.values())
    positive = [dict(rollout_end_step=int(r['step']), positive_events=int(r['positive_reward_steps']),
                     reward_sum=float(r['reward_sum'])) for r in rows if int(r['positive_reward_steps'])]
    first_event_index = next((i for i, r in enumerate(rows) if int(r['positive_reward_steps'])), len(rows))
    before = rows[:first_event_index]
    event_context = [{k: float(r[k]) for k in ['step', 'reward_sum', 'positive_reward_steps', 'entropy', 'advantage_std', 'value_loss', 'approx_kl']}
                     for r in rows[max(0, first_event_index-1):first_event_index+3]] if positive else []
    return dict(first_reward_context=event_context,
                zero_advantage_and_value_loss_before_first_reward=all(float(r['advantage_std']) == float(r['value_loss']) == 0 for r in before),
                steps=65536, rollouts=len(rows), reward_sum=sum(float(r['reward_sum']) for r in rows),
                positive_reward_events=sum(int(r['positive_reward_steps']) for r in rows),
                completed_episodes=sum(int(r['done_count']) for r in rows), reward_event_rollouts=positive,
                final_entropy=float(rows[-1]['entropy']), final_approx_kl=float(rows[-1]['approx_kl']),
                max_clipped_grad_norm=max(float(r['clipped_grad_norm']) for r in rows),
                max_advantage_std=max(float(r['advantage_std']) for r in rows), **load(run / 'summary.json'))


def compare(a, b):
    a = np.array(a['episodes']['returns']); b = np.array(b['episodes']['returns'])
    assert a.shape == b.shape == (128,)
    wins, losses = int(((a > 0) & (b <= 0)).sum()), int(((b > 0) & (a <= 0)).sum())
    return dict(ablation_only_scoring=wins, control_only_scoring=losses,
                exact_p=float(binomtest(wins, wins + losses, .5).pvalue) if wins + losses else 1.,
                mean_return_difference=float((a - b).mean()))


def main():
    if load(RUN / 'run_status.json')['phase'] != 'complete':
        raise RuntimeError('Training, final evaluation and visualization must finish first.')
    cfg = OmegaConf.to_container(OmegaConf.load(RUN / 'config.yaml'))
    base_cfg = OmegaConf.to_container(OmegaConf.load(BASE / 'config.yaml'))
    differences = {k: [base_cfg.get(k), cfg.get(k)] for k in base_cfg.keys() | cfg.keys() if cfg.get(k) != base_cfg.get(k)}
    assert set(differences) == {'num_steps', 'num_minibatches', 'gamma', 'gae_lambda', 'run_dir'}
    current, baseline = load(RUN / 'heldout/comparison.json'), load(BASE / 'heldout/comparison.json')
    assert current['seed_start'] == baseline['seed_start'] == 50000
    assert current['episodes_per_policy'] == baseline['episodes_per_policy'] == 128
    # Check the repeated controls and initial policy against saved evidence.
    for mode in ['random', 'noop']:
        assert current[mode]['episodes'] == baseline[mode]['episodes'], mode
    assert load(RUN / 'evaluation.json')[0]['episodes'] == load(BASE / 'evaluation.json')[0]['episodes']
    assert all(e['checkpoint_roundtrip_exact'] for e in load(RUN / 'evaluation.json'))
    policies = {'longer_credit': current['policy'], 'baseline': baseline['policy'], 'random': current['random'], 'noop': current['noop']}
    counts = {}
    for name, result in policies.items():
        n = len(result['episodes']['returns']); k = sum(x > 0 for x in result['episodes']['returns'])
        counts[name] = dict(games=n, scoring_games=k, scoring_rate_ci95_wilson=wilson(k, n), **result['summary'])
    a, b = training(RUN), training(BASE)
    assert a['rollouts'] == 32 and b['rollouts'] == 128
    comparisons = {name: compare(current['policy'], other) for name, other in [('baseline', baseline['policy']), ('random', current['random'])]}
    random_budget = load(BASE / 'random_training_budget.json')
    conclusion = ('No scoring improvement is established by this one-seed exploratory comparison.'
                  if any(x['exact_p'] >= .025 or x['mean_return_difference'] <= 0 for x in comparisons.values())
                  else 'This policy scored more often in both exploratory comparisons; fresh evaluation seeds and independent training seeds are needed to confirm it.')
    analysis = dict(config_differences=differences, ablation_training=a, baseline_training=b,
                    reused_random_budget=random_budget, policies=counts, paired_comparisons=comparisons,
                    controls_reproduced_exactly=True, initial_policy_evaluation_matched=True,
                    conclusion=conclusion, limitations=['One training seed per configuration.',
                    'Evaluation seeds 50000–50127 were already inspected in the baseline round.',
                    'Two paired scoring-probability tests; reported p-values are unadjusted exploratory values.',
                    'Rollout, gamma and lambda change together; their individual effects are not identified.',
                    'The random training-budget control is reused, not an independent replicate.'])
    (RUN / 'analysis.json').write_text(json.dumps(analysis, indent=2) + '\n')
    names = [('Longer credit', 'longer_credit'), ('v2 baseline', 'baseline'), ('Uniform random', 'random'), ('No-op', 'noop')]
    table = '\n'.join(f"| {label} | {counts[key]['scoring_games']}/128 | {counts[key]['mean_return']:.5f} | {counts[key]['mean_length']:.1f} | {counts[key]['truncated']} |" for label, key in names)
    event_word = 'event' if a['positive_reward_events'] == 1 else 'events'
    first = a['reward_event_rollouts'][0]['rollout_end_step'] if a['reward_event_rollouts'] else 'not observed'
    learning = ('Both PPO configurations encountered zero training rewards. Longer credit assignment had no observed score signal to propagate; final near-uniform action selection is not evidence of learned placement skill.'
                if a['positive_reward_events'] == b['positive_reward_events'] == 0 else
                'Training reward events show whether useful targets became available, but sparse evaluation scores and a single trained seed cannot establish robust policy improvement.')
    if a['positive_reward_events'] > 0 and b['positive_reward_events'] == 0:
        learning += (' The ablation encountered feedback in a run where the baseline did not. This is evidence that the corrected pipeline can update from a real game score, not proof that longer credit caused the discovery. '
                     'Before the first reward, logged advantage variation and value loss were zero; changing gamma/lambda cannot propagate a score that has not been encountered. Different rollout/update scheduling also changes subsequent random draws and trajectories. Independent training seeds are needed to distinguish a repeatable benefit from a lucky discovery.')
    text = f'''# Longer-credit Tetris ablation · 65,536 steps

{conclusion}

## What changed

Fresh seed 0; rollout length 32→128, gamma 0.99→0.995 and GAE lambda 0.95→0.98. Minibatches increase 4→16 to keep 128 samples per minibatch. Both arms process 65,536 decisions, four epochs per rollout and 2,048 optimizer steps. The ablation has 32 policy refreshes rather than 128. The approximate GAE decay scale 1/(1−gamma×lambda), ignoring terminals/truncation, increases from 16.8 to 40.2 decisions.

Pixel observations, CNN, corrected frame timing, original score reward, learning rate, entropy/value coefficients, zero value-output initialization, advantage floor and global gradient clipping are unchanged. Initial eight-game evaluation arrays match exactly. This bundled comparison cannot isolate the effect of each changed setting.

## Reward discovery and final scores

Longer-credit training: **{a['positive_reward_events']} positive reward {event_word}**, reward sum {a['reward_sum']:g}, {a['completed_episodes']} completed episodes. First reward rollout end: {first}. Baseline: {b['positive_reward_events']} events, {b['completed_episodes']} completed episodes. The previously recorded equal-budget random rollout had {random_budget['positive_reward_events']} events; it is reused, not a new replicate.

Final comparisons use the same 128 seeds **50000–50127** and corrected timing. These seeds were already inspected, so this is exploratory evaluation, not new held-out confirmation. Repeated random/no-op per-game arrays reproduced the baseline controls exactly.

| Policy | Games scoring | Mean score | Mean decisions | Truncations |
| --- | ---: | ---: | ---: | ---: |
{table}

Exact paired scoring-probability tests: longer credit versus baseline p={comparisons['baseline']['exact_p']:.4g}; versus random p={comparisons['random']['exact_p']:.4g}. These two p-values are unadjusted and do not test score magnitude or robustness across training seeds. Raw per-game arrays and Wilson intervals are in the linked evidence.

## What we learned

{learning}

Maximum logged raw advantage standard deviation: {a['max_advantage_std']:.6g}. Final entropy: {a['final_entropy']:.6f} (log(5)=1.609438); final approximate KL: {a['final_approx_kl']:.6g}. All logged quantities and parameters remained finite, and full-state/inference checkpoint reload checks passed. Maximum logged clipped gradient norm: {a['max_clipped_grad_norm']:.6g}. Training throughput was {a['training_steps_per_second']:.1f} decisions/s, versus {b['training_steps_per_second']:.1f} in the earlier baseline; these sequential desktop runs are not isolated hardware benchmarks.

The next hypothesis to test is reward discovery through pixel-based exploration or a larger replicated experience budget. This run alone does not show that credit length is irrelevant once rewards are encountered. No additional training or exploration objective was launched.

## Evidence and replay

[Result card](result_card.html) · [Policy animation](policy_card.gif) · [Learning curves](visualizations/learning_curves.png) · [Checkpoint](checkpoint.msgpack) · [Analysis](analysis.json) · [Per-game results](heldout/comparison.json) · [Baseline](../v2_baseline_seed0/RESULT.md) · [Implementation log](../../../IMPLEMENTATION_LOG.md) · [Configuration](../../../conf/credit_horizon.yaml)

Gameplay seed 30016 is the same illustrative seed used for the baseline, selected independently of this round's final evaluation. It is not a highlight selected for scoring. Exact training sources and ROM are saved with hashes; report/card-generation sources are recorded separately after the run.
'''
    (RUN / 'RESULT.md').write_text(text)
    report_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    changes_url = f'https://github.com/bzz/octax/compare/5f7efc494d50d9534791426efbb636c48ebfd1d0...{report_commit}'
    links = [('Full result', 'RESULT.md'), ('Implementation log', '../../../IMPLEMENTATION_LOG.md'),
             ('Changed settings', '../../../conf/credit_horizon.yaml'), ('Code changes', changes_url)]
    write_round_card(RUN, 'Longer credit · 65,536 steps', conclusion,
                     [('Training reward events', a['positive_reward_events'])] + [(label + ' scored', f"{counts[key]['scoring_games']}/128") for label, key in names[:3]], links)
    heading = '## 2026-11-23 — Longer-credit 64K ablation completed'
    entry = f'''\n{heading}\n\nCompleted the fresh seed-0 65,536-step run with 128-step rollouts, gamma 0.995 and lambda 0.98. Training encountered {a['positive_reward_events']} positive reward {event_word} (reward sum {a['reward_sum']:g}) in {a['completed_episodes']} completed games. The baseline encountered {b['positive_reward_events']} events; the reused equal-budget random rollout encountered {random_budget['positive_reward_events']}.\n\nOn the same previously inspected 128 seeds, longer credit scored in {counts['longer_credit']['scoring_games']}/128 games, baseline in {counts['baseline']['scoring_games']}/128 and random in {counts['random']['scoring_games']}/128. Unadjusted exact paired scoring tests: p={comparisons['baseline']['exact_p']:.4g} versus baseline, p={comparisons['random']['exact_p']:.4g} versus random. {conclusion} {learning}\n\nInitial policy evaluation and repeated random/no-op controls matched their baseline arrays exactly. Finite-state checks and exact checkpoint roundtrip passed; final entropy {a['final_entropy']:.6f}, throughput {a['training_steps_per_second']:.1f} decisions/s. This is a single-seed, bundled, exploratory comparison; it does not isolate rollout/gamma/lambda effects or establish performance across seeds. No further run was launched.\n\nSaved full evidence in `runs/tetris/v2_credit_horizon_seed0/RESULT.md`, plus a small reusable result card with actual policy gameplay and links to the settings and log. Rendering continues to use Matplotlib/Seaborn primitives from the existing MinAtar-style workflow.\n'''
    p = ROOT / 'IMPLEMENTATION_LOG.md'
    if heading not in p.read_text():
        p.open('a').write(entry)
    p = ROOT / 'RESULTS.md'
    if heading not in p.read_text():
        p.open('a').write(entry)
    manifest = {}
    for name in ['summarize_credit_ablation.py', 'round_card.py', 'summarize_v2_baseline.py']:
        data = (ROOT / name).read_bytes()
        manifest[name] = hashlib.sha256(data).hexdigest()
        dest = RUN / 'report_source_snapshot' / name
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(data)
    (RUN / 'report_source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'training_events': a['positive_reward_events'], 'scoring_games': {k: v['scoring_games'] for k,v in counts.items()}, 'comparisons': comparisons, 'conclusion': conclusion}, indent=2))


if __name__ == '__main__':
    main()
