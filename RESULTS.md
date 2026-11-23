# Tetris CPU experiment result

The training pipeline works on this Mac's CPU. The 65,536-step smoke test completed, saved a usable checkpoint, and passed exact reload/inference checks. **A convincing learning improvement has not been established.**

## Measured result

| Measure | Result |
| --- | --- |
| Environment steps | 65,536 |
| Parallel environments / rollout length | 16 / 32 |
| PPO iterations / optimizer minibatch updates | 128 / 2,048 |
| Backend | JAX CPU, macOS arm64 |
| Training throughput | 123.4 steps/s |
| Steady throughput (first iteration excluded) | 123.7 steps/s |
| Rollout collection time | 35.1 s |
| PPO update time | 496.0 s |
| Training + final evaluation/export wall time | 556.6 s |
| Positive training reward events | 0 |
| Completed training episodes | 15 |
| Gradient norm range | 0.137–0.616 |
| Parameter change (L2) | 16.460 |
| Regression tests | 72 passed |

The wall-time figure excludes setup and the initial evaluation. Timed collection/update throughput excludes most host bookkeeping and evaluation; occasional validation work shared the CPU, so treat this as practical observed throughput rather than an isolated hardware benchmark. PPO updates consumed 93.4% of timed training.

At this throughput a 499,712-step pilot projects to about 67 minutes per seed before evaluation, or about 3.4 hours for three seeds. These longer runs have **not** been started, per the user's choice.

## Evaluation and its limits

On the eight development evaluation seeds (10000–10007), PPO's final mean return was 0; random was 0.125 and no-op 0. Three of PPO's eight games reached the 4,500-step cap.

The final checkpoint was then independently reloaded by `evaluate_checkpoint.py` and compared on 16 held-out seeds (30000–30015):

| Policy | Mean score | Episodes scoring | Mean length | Time-limit truncations |
| --- | ---: | ---: | ---: | ---: |
| Random | 0.0000 | 0/16 | 2395.1 | 0/16 |
| No-op | 0.0000 | 0/16 | 2724.4 | 0/16 |
| PPO | 0.0625 | 1/16 | 3348.8 | 7/16 |

PPO cleared one line in one held-out episode. A one-event difference across 16 games and one training seed is insufficient to establish superiority over random play. The longer PPO episode lengths are partly time-limit truncations, and are not evidence of successful play. No positive rewards were encountered during training; the sparse-reward bottleneck remains the main finding.

The initial environment sanity check separately proved that all controls work, games terminate, later pieces vary across RNG keys, and a naturally reached +1 reward matches the V[10] score change. See `runs/tetris/sanity/reward_validation.json` and `positive_reward.png`.

## Visual results

- [Learning curves](runs/tetris/ppo_seed0_smoke/visualizations/learning_curves.png): development evaluation scores, reward events, PPO losses, entropy, action distribution and throughput.
- [Gameplay comparison](runs/tetris/ppo_seed0_smoke/visualizations/gameplay.gif): full random/PPO episodes on seed 30016, sampled every eight actions at approximately 4× speed. Both score 0 in this illustrative episode; PPO ends at step 881 and random at step 2140. This sample is not the aggregate evaluation.
- [Gameplay still](runs/tetris/ppo_seed0_smoke/visualizations/gameplay.png).
- [Checkpoint](runs/tetris/ppo_seed0_smoke/checkpoint.msgpack), [metrics](runs/tetris/ppo_seed0_smoke/metrics.csv), [held-out results](runs/tetris/ppo_seed0_smoke/heldout/comparison.json).

Gameplay reuses the Matplotlib/Seaborn rendering primitives used by [MinAtar's display_state](https://github.com/kenjyoung/MinAtar/blob/master/minatar/environment.py): discrete colors, cubehelix palette and uninterpolated pixels. The underlying environment remains Octax Tetris; its monochrome observations are unchanged.

## What is ready next

The [experiment guide](EXPERIMENT.md) contains environment recreation, evaluation, visualization, resume and three-seed pilot commands. `./run_stages.sh pilot` runs the planned 499,712-step seeds sequentially. A shorter next diagnostic is to inspect reward-discovery frequency and policy action bias in a single continued pilot before scaling further. Keep the original score for evaluation if reward shaping is later introduced as a separate experiment.

Brix/Pong transfer commands are prepared but unexecuted and unvalidated on those target games; they are explicitly a future experiment. The [implementation log](IMPLEMENTATION_LOG.md) records major decisions and deviations, including the additional convolution fix, Gymnax pin, clipping setting, host-driven timing/logging and the visualization clarification.

## Follow-up: 499,712-step pilot completed

The longer continuation is complete; it ran at 142.1 steps/s and took 54.2 minutes for training plus periodic evaluation. It encountered only one positive training reward across the 434,176 additional steps. All development checkpoint evaluations scored zero.

| Final policy, 64 held-out games each | Scoring games | Mean game score |
| --- | ---: | ---: |
| PPO | 1/64 | 0.015625 |
| Random | 0/64 | 0 |
| No-op | 0/64 | 0 |

This is not convincing evidence that PPO beats random play. The reward-discovery problem persists. Nine PPO games reached the time cap, so longer average survival is not a substitute for score.

[Longer-run curves](runs/tetris/continued_seed0/visualizations/learning_curves.png) · [Gameplay](runs/tetris/continued_seed0/visualizations/gameplay.gif) · [Raw held-out evaluation](runs/tetris/continued_seed0/heldout/comparison.json)

## Follow-up: timing and sparse-reward stability fixes

A later investigation found unsigned timer underflow, a once-per-action rather than once-per-frame timer tick, and amplification of tiny reward-free advantages by normalization. New v2 runs correct timing, zero-initialize the value output, floor advantage normalization and clip by global norm. Historical scores above are legacy-environment results and must not be pooled with v2. A fresh 8,192-step validation passed, but no learning superiority is claimed. See [REWARD_DIAGNOSIS.md](REWARD_DIAGNOSIS.md) and [IMPLEMENTATION_LOG.md](IMPLEMENTATION_LOG.md) for the controlled probes and next experiment.

## Fresh 64K v2 baseline result

Completed 65,536 fresh v2 steps. Training positive rewards: 0; equal-budget random: 0. Final matched corrected-timing evaluation: PPO 1/128 scoring games, random 3/128. This evaluation does not establish a scoring advantage over random play.

[Full v2 result](runs/tetris/v2_baseline_seed0/RESULT.md).

## 2026-11-23 — Longer-credit 64K ablation completed

Completed the fresh seed-0 65,536-step run with 128-step rollouts, gamma 0.995 and lambda 0.98. Training encountered 1 positive reward event (reward sum 1) in 42 completed games. The baseline encountered 0 events; the reused equal-budget random rollout encountered 0.

On the same previously inspected 128 seeds, longer credit scored in 2/128 games, baseline in 1/128 and random in 3/128. Unadjusted exact paired scoring tests: p=1 versus baseline, p=1 versus random. No scoring improvement is established by this one-seed exploratory comparison. Training reward events show whether useful targets became available, but sparse evaluation scores and a single trained seed cannot establish robust policy improvement. The ablation encountered feedback in a run where the baseline did not. This is evidence that the corrected pipeline can update from a real game score, not proof that longer credit caused the discovery. Before the first reward, logged advantage variation and value loss were zero; changing gamma/lambda cannot propagate a score that has not been encountered. Different rollout/update scheduling also changes subsequent random draws and trajectories. Independent training seeds are needed to distinguish a repeatable benefit from a lucky discovery.

Initial policy evaluation and repeated random/no-op controls matched their baseline arrays exactly. Finite-state checks and exact checkpoint roundtrip passed; final entropy 1.578472, throughput 139.5 decisions/s. This is a single-seed, bundled, exploratory comparison; it does not isolate rollout/gamma/lambda effects or establish performance across seeds. No further run was launched.

Saved full evidence in `runs/tetris/v2_credit_horizon_seed0/RESULT.md`, plus a small reusable result card with actual policy gameplay and links to the settings and log. Rendering continues to use Matplotlib/Seaborn primitives from the existing MinAtar-style workflow.

## 2026-11-23 — Three-seed 64K replication completed

Completed the four new seed-1/2 runs at 65,536 steps each, with unchanged pixel-only observations, original score reward and corrected timing. Including seed 0, baseline training reward events were [0, 0, 0]; longer credit events were [1, 2, 0]. Final scoring games (out of 128 per policy) were baseline [1, 1, 1] and longer credit [2, 1, 1], versus 3/128 for the shared random control.

Compared each final policy with its exactly reconstructed untrained network on the same game seeds. Mean score gain over initialization: baseline -0.01042, longer credit -0.00781. Neither configuration passes the exploratory learning gate at 64K steps; useful scoring improvement is not established. Three training seeds and reused evaluation seeds do not establish broad generalization; duplicated random evaluations are not independent control samples.

Configuration, initial-policy, repeated-control, finite-metric and checkpoint checks passed. The run directory `runs/tetris/v2_replication_64k/` contains the full comparison and a compact actual-gameplay card using the preselected longer-credit seed-1 policy. The goal of a demonstrably useful learning setup remains subject to fresh evidence; the screening gate is not a completion criterion.

## 64K self-imitation mechanism test

Replay does not beat all scoring controls in this exploratory sample; useful learning remains unestablished. See [full evidence and animation](runs/tetris/v3_self_imitation_seed1/RESULT.md). This selected seed-1 test retains original pixels and game rewards; it adds replay compute.

## Corrected PPO and score replay at 512K

The on-policy PPO arm does not exceed both random and its untrained network; the learning objective remains unmet. See [full cumulative comparison and animation](runs/tetris/learning_512k_seed1/RESULT.md). Both arms continue seed 1 from 64K; these are not independent replications.

## Strictly on-policy long-trace PPO at 512K

The long-trace policy does not exceed both random and its initialization; useful on-policy learning remains unestablished. See [results and actual-policy animation](runs/tetris/v2_long_trace_512k_seed1/RESULT.md).

## Pixel response and greedy inference

The trained networks remain responsive to pixels, but greedy inference does not establish a scoring gain. Long trace scores 1/128, prior PPO 0/128 and the initial network 0/128, versus random 3/128. These are frozen-policy diagnostics, with no added training updates. [Evidence and greedy-policy animation](runs/tetris/pixel_response_diagnostic/RESULT.md).

## Two-seed PPO entropy ablation at 64K

Removing entropy regularization does not pass the two-seed scoring screen; useful on-policy learning remains unestablished. Both entropy-zero seeds discovered one reward and scored 0/128, compared with matched controls 1/128 each and shared random 3/128. [Results and animations](runs/tetris/v2_entropy_zero_64k/RESULT.md).

## Pixel-novelty smoke update

Matched, strictly on-policy pixel-novelty smoke runs passed correctness gates at 8,192 decisions each. Neither found original game rewards. Warmed throughput was 157.29 decisions/s with the bonus disabled and 73.88 enabled. This is implementation/runtime evidence, not evidence of better play. See the [smoke report and actual-policy card](runs/tetris/pixel_novelty_smoke/SMOKE_REPORT.md). A separate frozen two-seed 64K pilot is in progress; its scores are not yet available.
