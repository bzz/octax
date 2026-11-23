# Longer-credit Tetris ablation · 65,536 steps

No scoring improvement is established by this one-seed exploratory comparison.

## What changed

Fresh seed 0; rollout length 32→128, gamma 0.99→0.995 and GAE lambda 0.95→0.98. Minibatches increase 4→16 to keep 128 samples per minibatch. Both arms process 65,536 decisions, four epochs per rollout and 2,048 optimizer steps. The ablation has 32 policy refreshes rather than 128. The approximate GAE decay scale 1/(1−gamma×lambda), ignoring terminals/truncation, increases from 16.8 to 40.2 decisions.

Pixel observations, CNN, corrected frame timing, original score reward, learning rate, entropy/value coefficients, zero value-output initialization, advantage floor and global gradient clipping are unchanged. Initial eight-game evaluation arrays match exactly. This bundled comparison cannot isolate the effect of each changed setting.

## Reward discovery and final scores

Longer-credit training: **1 positive reward event**, reward sum 1, 42 completed episodes. First reward rollout end: 34816. Baseline: 0 events, 44 completed episodes. The previously recorded equal-budget random rollout had 0 events; it is reused, not a new replicate.

Final comparisons use the same 128 seeds **50000–50127** and corrected timing. These seeds were already inspected, so this is exploratory evaluation, not new held-out confirmation. Repeated random/no-op per-game arrays reproduced the baseline controls exactly.

| Policy | Games scoring | Mean score | Mean decisions | Truncations |
| --- | ---: | ---: | ---: | ---: |
| Longer credit | 2/128 | 0.01562 | 1375.8 | 0 |
| v2 baseline | 1/128 | 0.00781 | 1308.7 | 0 |
| Uniform random | 3/128 | 0.02344 | 1273.9 | 0 |
| No-op | 0/128 | 0.00000 | 777.7 | 0 |

Exact paired scoring-probability tests: longer credit versus baseline p=1; versus random p=1. These two p-values are unadjusted and do not test score magnitude or robustness across training seeds. Raw per-game arrays and Wilson intervals are in the linked evidence.

## What we learned

Training reward events show whether useful targets became available, but sparse evaluation scores and a single trained seed cannot establish robust policy improvement. The ablation encountered feedback in a run where the baseline did not. This is evidence that the corrected pipeline can update from a real game score, not proof that longer credit caused the discovery. Before the first reward, logged advantage variation and value loss were zero; changing gamma/lambda cannot propagate a score that has not been encountered. Different rollout/update scheduling also changes subsequent random draws and trajectories. Independent training seeds are needed to distinguish a repeatable benefit from a lucky discovery.

Maximum logged raw advantage standard deviation: 0.0870864. Final entropy: 1.578472 (log(5)=1.609438); final approximate KL: 0.00807059. All logged quantities and parameters remained finite, and full-state/inference checkpoint reload checks passed. Maximum logged clipped gradient norm: 0.111259. Training throughput was 139.5 decisions/s, versus 114.1 in the earlier baseline; these sequential desktop runs are not isolated hardware benchmarks.

The next hypothesis to test is reward discovery through pixel-based exploration or a larger replicated experience budget. This run alone does not show that credit length is irrelevant once rewards are encountered. No additional training or exploration objective was launched.

## Evidence and replay

[Result card](result_card.html) · [Policy animation](policy_card.gif) · [Learning curves](visualizations/learning_curves.png) · [Checkpoint](checkpoint.msgpack) · [Analysis](analysis.json) · [Per-game results](heldout/comparison.json) · [Baseline](../v2_baseline_seed0/RESULT.md) · [Implementation log](../../../IMPLEMENTATION_LOG.md) · [Configuration](../../../conf/credit_horizon.yaml)

Gameplay seed 30016 is the same illustrative seed used for the baseline, selected independently of this round's final evaluation. It is not a highlight selected for scoring. Exact training sources and ROM are saved with hashes; report/card-generation sources are recorded separately after the run.
