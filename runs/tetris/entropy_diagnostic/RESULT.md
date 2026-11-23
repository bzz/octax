# On-policy PPO: gradient and credit-timing diagnostic

The local gradient audit does not show entropy dominating PPO's policy gradient. The first recorded reward had only 11 current-rollout transitions available for direct credit. The next experiment keeps entropy unchanged and tests longer on-policy traces; it is a hypothesis test, not a learning result.

## Gradient audit

Collect 2,048 fresh decisions from each saved PPO state, then inspect all 16 minibatches at the unchanged parameter state. Both rollouts were reward-free. Compare one actual saved-Adam update with and without entropy; discard every diagnostic update. The component loss and reconstructed configured update match the actual PPO implementation, with maximum state-element difference 1.49e-8.

| Checkpoint step | Diagnostic rewards | Mean entropy/policy gradient norm | Entropy before | After configured update | After zero-entropy update |
| --- | ---: | ---: | ---: | ---: | ---: |
| 65,536 | 0 | 0.114 | 1.525222 | 1.524373 | 1.523979 |
| 524,288 | 0 | 0.139 | 1.602169 | 1.602143 | 1.602118 |

These are local measurements, not a four-epoch learning trajectory or a long-run causal experiment. Raw gradient norms do not include Adam scaling; the separate one-minibatch counterfactual does. Entropy still changes update pressure, but these measurements do not establish it as the dominant cause of failed learning. Full per-minibatch values and checkpoint hashes are in `gradients.json`.

The diagnostic was run twice from the same saved states while validating the decomposition: 8,192 executed diagnostic decisions, representing 4,096 distinct deterministic transitions. None enter a learning run or its environment budget. They are not independent replications.

## Credit timing

The historical trajectory prefixes below precede the first self-imitation update and match the earlier PPO prefix diagnostics. They are inspected only for timing and never supplied as training data. Direct credit cannot cross a termination boundary or reach a rollout already consumed before a reward arrives.

| Episode transitions through score | Available in its 128-step rollout | Hypothetical 1024-step segmentation |
| ---: | ---: | ---: |
| 907 | 11 | 907 |
| 533 | 118 | 533 |

The longer-rollout column holds the recorded trajectory fixed and caps at episode boundaries. It is not a prediction: changing rollout length also changes update timing, RNG progression and subsequent experience. Source episode hashes and limitations are in `reward_timing.json`.

## Next controlled test

Fresh on-policy PPO, seed 1, 524,288 decisions, rollout 1024, gamma 0.999 and lambda 1.0. Keep the same pixels, CNN, original rewards, entropy coefficient 0.01, minibatch size 128 and 16,384 optimizer updates. This changes three related credit settings together. Initial parameters match seed 1 exactly. See [protocol](../../../ON_POLICY_CREDIT.md) and [implementation log](../../../IMPLEMENTATION_LOG.md).

The animation is a reference from the **existing 512K on-policy PPO checkpoint**, game seed 30016, score 0. It is not a newly learned diagnostic policy. [Reference policy result](../v2_credit_horizon_512k_seed1/RESULT.md).

To reproduce the numerical diagnostics, run `diagnose_ppo_entropy.py` on the two PPO run folders and `diagnose_credit_timing.py` on the recorded v3 seed-1 folder, using the arguments shown in the implementation sources. Source snapshots include the report recipe; run it from the repository root to rebuild this card from saved JSON.
