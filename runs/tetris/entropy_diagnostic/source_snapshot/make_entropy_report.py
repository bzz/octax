from pathlib import Path
import hashlib,json,shutil,sys
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
from round_card import write_round_card
run=ROOT/'runs/tetris/entropy_diagnostic'
gradients=json.loads((run/'gradients.json').read_text());timing=json.loads((run/'reward_timing.json').read_text())
assert len(gradients)==2 and all(x['validation']['reconstructed_update_matches_ppo'] for x in gradients)
assert [x['checkpoint_step'] for x in gradients]==[65536,524288]
ratios=[x['gradient_summary']['entropy_to_policy_norm']['mean'] for x in gradients]
rows='\n'.join(f"| {x['checkpoint_step']:,} | {x['diagnostic_positive_events']} | {x['gradient_summary']['entropy_to_policy_norm']['mean']:.3f} | {x['mean_entropy_before']:.6f} | {x['single_minibatch_counterfactual']['configured_entropy']['mean_entropy_after_one_minibatch']:.6f} | {x['single_minibatch_counterfactual']['zero_entropy']['mean_entropy_after_one_minibatch']:.6f} |" for x in gradients)
credit='\n'.join(f"| {x['episode_transitions_through_reward']} | {x['current_rollout_credit_transitions']} | {x['hypothetical_1024_credit_transitions']} |" for x in timing['pre_replay_rewards'])
text=f'''# On-policy PPO: gradient and credit-timing diagnostic

The local gradient audit does not show entropy dominating PPO's policy gradient. The first recorded reward had only 11 current-rollout transitions available for direct credit. The next experiment keeps entropy unchanged and tests longer on-policy traces; it is a hypothesis test, not a learning result.

## Gradient audit

Collect 2,048 fresh decisions from each saved PPO state, then inspect all 16 minibatches at the unchanged parameter state. Both rollouts were reward-free. Compare one actual saved-Adam update with and without entropy; discard every diagnostic update. The component loss and reconstructed configured update match the actual PPO implementation, with maximum state-element difference 1.49e-8.

| Checkpoint step | Diagnostic rewards | Mean entropy/policy gradient norm | Entropy before | After configured update | After zero-entropy update |
| --- | ---: | ---: | ---: | ---: | ---: |
{rows}

These are local measurements, not a four-epoch learning trajectory or a long-run causal experiment. Raw gradient norms do not include Adam scaling; the separate one-minibatch counterfactual does. Entropy still changes update pressure, but these measurements do not establish it as the dominant cause of failed learning. Full per-minibatch values and checkpoint hashes are in `gradients.json`.

The diagnostic was run twice from the same saved states while validating the decomposition: 8,192 executed diagnostic decisions, representing 4,096 distinct deterministic transitions. None enter a learning run or its environment budget. They are not independent replications.

## Credit timing

The historical trajectory prefixes below precede the first self-imitation update and match the earlier PPO prefix diagnostics. They are inspected only for timing and never supplied as training data. Direct credit cannot cross a termination boundary or reach a rollout already consumed before a reward arrives.

| Episode transitions through score | Available in its 128-step rollout | Hypothetical 1024-step segmentation |
| ---: | ---: | ---: |
{credit}

The longer-rollout column holds the recorded trajectory fixed and caps at episode boundaries. It is not a prediction: changing rollout length also changes update timing, RNG progression and subsequent experience. Source episode hashes and limitations are in `reward_timing.json`.

## Next controlled test

Fresh on-policy PPO, seed 1, 524,288 decisions, rollout 1024, gamma 0.999 and lambda 1.0. Keep the same pixels, CNN, original rewards, entropy coefficient 0.01, minibatch size 128 and 16,384 optimizer updates. This changes three related credit settings together. Initial parameters match seed 1 exactly. See [protocol](../../../ON_POLICY_CREDIT.md) and [implementation log](../../../IMPLEMENTATION_LOG.md).

The animation is a reference from the **existing 512K on-policy PPO checkpoint**, game seed 30016, score 0. It is not a newly learned diagnostic policy. [Reference policy result](../v2_credit_horizon_512k_seed1/RESULT.md).

To reproduce the numerical diagnostics, run `diagnose_ppo_entropy.py` on the two PPO run folders and `diagnose_credit_timing.py` on the recorded v3 seed-1 folder, using the arguments shown in the implementation sources. Source snapshots include the report recipe; run it from the repository root to rebuild this card from saved JSON.
'''
(run/'RESULT.md').write_text(text)
visual=run/'visualizations';visual.mkdir(exist_ok=True)
shutil.copy2(ROOT/'runs/tetris/v2_credit_horizon_512k_seed1/visualizations/policy_trajectory.npz',visual/'policy_trajectory.npz')
write_round_card(run,'On-policy PPO · credit diagnostic','Gradient checks do not show entropy dominance. The first score had only 11 rollout transitions available. Animation: existing 512K PPO reference.',[('Entropy / policy gradient norm at 64K',f'{ratios[0]:.1%}'),('Entropy / policy gradient norm at 512K',f'{ratios[1]:.1%}'),('First score: available episode history','11 of 907 transitions')],[('Full diagnostic','RESULT.md'),('Next on-policy protocol','../../../ON_POLICY_CREDIT.md'),('Implementation log','../../../IMPLEMENTATION_LOG.md')])
names=['diagnose_ppo_entropy.py','diagnose_credit_timing.py','round_card.py','evaluate_initial_policy.py','octax/agents/ppo.py','octax/wrappers.py','octax/env.py','octax/checkpoints.py','experiment_utils.py','conf/config.yaml','conf/credit_horizon.yaml','conf/long_trace.yaml','requirements-macos.lock.txt']
manifest={}
for name in names:
 data=(ROOT/name).read_bytes();p=run/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);manifest[name]=hashlib.sha256(data).hexdigest()
data=Path(__file__).read_bytes();(run/'source_snapshot/make_entropy_report.py').write_bytes(data);manifest['make_entropy_report.py']=hashlib.sha256(data).hexdigest()
(run/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Diagnostic report, source snapshots and reference-policy card written.')
