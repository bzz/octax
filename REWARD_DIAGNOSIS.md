# Why rewards are sparse, and what changes before the next run

The 499,712-step run is not a sound basis for simply buying more training time. We found an emulator timing defect and a PPO normalization problem. Both are now corrected for new runs. Line-clear rewards also remain intrinsically rare under random keypress exploration; the fixes do not establish that PPO will learn Tetris.

The agent remains strictly pixel-based: four monochrome 32×64 frames, the same CNN and actor/value heads, and five keypress actions. No board geometry, holes, piece locations or emulator registers are supplied to the policy. No reward shaping was added.

## 1. The score signal works, but timing was wrong

The ROM increments V[10] at instruction address `0x3AC` after its line-removal routine. Octax correctly converts that score to floating point and returns its difference between steps. The earlier scoring replay demonstrated a naturally reached +1. The diagnostic random-policy runs below also reach positive rewards with corrected timing. There is no evidence that the score-difference calculation is dropping line-clear rewards.

Two problems existed in `OctaxEnv.step`:

- `jnp.maximum(uint8_timer - 1, 0)` wraps zero to 255 **before** maximum is applied. Tetris's down-key code explicitly sets its delay timer to zero at `0x24C–0x24E`. A resulting bogus delay can therefore suppress natural falling until another keypress or the countdown releases it.
- Four 60 Hz frames are executed per action, but timers were decremented only once per action. Even without underflow, that clock ran four times too slowly relative to the stated frame rate.

New `timer_mode=frame` runs execute instructions and tick saturating timers once per simulated frame. Subtraction widens to a signed type first, then casts back to uint8. There are still four observation frames and the same amount of CHIP-8 instruction work per action. Frame timer mode explicitly requires 60 FPS; instruction frequency still uses upstream integer division (700//60 = 11 instructions/frame, effectively 660 instructions/s).

For reproducible comparisons, `legacy` preserves old behavior and `step_saturating` isolates the underflow repair. These diagnostic modes are not the recommended new baseline. This timing change changes the environment dynamics: old and new returns must not be combined into a single learning curve.

### Matched-seed timing/action probes

64 complete games per condition, seeds 40000–40063, no learning. The register and timer counters below are diagnostic instrumentation only.

| Timing | Policy | Action repeat | Mean episode decisions | Games scoring | Steps with delay >16 |
| --- | --- | ---: | ---: | ---: | ---: |
| Legacy | Random | 1 | 2,288.5 | 1/64 | 29.65% |
| Saturating, once per action | Random | 1 | 2,037.1 | 1/64 | 0% |
| Correct, once per frame | Random | 1 | 1,249.3 | 2/64 | 0% |
| Correct, once per frame | Random | 4 | 650.4 | 0/64 | 0% |
| Legacy | No-op | 1 | 2,805.9 | 0/64 | 0% |
| Correct, once per frame | No-op | 1 | 777.2 | 0/64 | 0% |

The strongest evidence is the timer counter and timing correction, not the difference between one and two scoring games. The corrected random condition yielded only two reward events in 79,954 steps: scoring remains sparse. Repeating a key four decisions at a time reduced survival and did not help score in this probe, so action repetition is not enabled.

Source data: `runs/tetris/reward_diagnosis/timing_action_ablation.json`. Reproduce with `python diagnose_rewards.py` in the environment documented in EXPERIMENT.md.

## 2. PPO was amplifying critic noise into policy updates

At the end of the old run, value loss was almost zero, yet gradients remained substantial. On one fixed, reward-free rollout from the 499,712-step checkpoint:

| Quantity | Measured value |
| --- | ---: |
| Reward sum | 0 |
| Raw advantage standard deviation | 0.00003431 |
| Policy-gradient norm with ordinary normalization | 0.37467 |
| Entropy-gradient norm, coefficient 0.01 | 0.00830 |
| Policy-gradient norm with std floor 0.001 | 0.01286 |
| Policy-gradient norm with std floor 0.01 | 0.00129 |

Normalizing arbitrarily small advantages to unit variance made the policy gradient about 45 times the entropy gradient on this rollout. Those advantages came from value predictions/bootstrap errors, not an observed reward. This explains how substantial policy drift can occur without useful game feedback; it does not prove the share of historical drift caused by each mechanism.

New-run changes:

1. Initialize the final value-output layer to zero. The model structure and parameter names are unchanged. Before the first reward, value predictions and resulting reward-free advantages remain zero rather than starting from random critic output.
2. Normalize using `max(advantage_std + 1e-8, 0.01)`. This suppresses amplification of tiny errors. Ordinary normalization is unchanged when advantage variation exceeds the floor. The floor is a tunable stabilization choice supported by the probe, not a proven optimal hyperparameter.
3. Use `optax.clip_by_global_norm(0.5)`. The previous `optax.clip(0.5)` bounded each coordinate, despite the setting being named `max_grad_norm`; it did not bound the full gradient norm.
4. Log raw and normalized advantage standard deviations, approximate KL, clipping fraction, and gradient norms before and after clipping. A nonzero gradient alone is no longer treated as evidence of a useful learning signal.

Source data: `runs/tetris/reward_diagnosis/ppo_noise.json`; script: `diagnose_ppo.py`.

## 3. Credit assignment is a separate next hypothesis

The original rollout has 32 decisions. With gamma=0.99 and lambda=0.95, the direct GAE coefficient for a reward 100 decisions later would be `(0.99×0.95)^100 ≈ 0.00217`; at a rollout boundary it must instead be propagated through the learned bootstrap value. Very rare rewards give that bootstrap little training. This is a plausible limitation, not an isolated empirical finding from this investigation.

The default v2 configuration retains the original rollout/horizon settings to avoid attributing a change in learning to several untested hyperparameters at once. A prepared, **unlaunched** `credit_horizon` configuration uses 128 steps, 16 minibatches, gamma=0.995 and lambda=0.98. It keeps 16 environments, minibatch size 128, learning rate, network, reward, and update epochs fixed. Its direct 100-step GAE coefficient is about 0.0803. Compare it against the corrected baseline at the same environment-step budget and on the same evaluation seeds.

## 4. Validation, compatibility and next decision

- Timer regressions cover zero/one/255 boundaries and one tick per frame using a looping test ROM.
- PPO regressions cover tiny-noise suppression, preserving score-scale normalization, zero initial value output, and actual global-norm clipping.
- Resume validation rejects changes to timer/PPO semantics before creating a new run folder.
- Old checkpoint evaluation automatically selects legacy timers when the original config has no timer settings. The saved smoke checkpoint reproduces **all** prior per-episode scores, lengths, action counts and terminal flags exactly.
- A short 8,192-step fresh v2 run validates the whole changed pipeline. Its measured completion is appended below once checks finish. It is not a learning-performance experiment.

Do **not** continue the old optimizer/environment state into v2. Start fresh. Keep original game score as the evaluation metric. Use the corrected-timing random/no-op baselines; do not compare a v2 policy to a legacy-timing baseline. The plotting command now rejects mismatched timer modes.

Recommended next experiment: a fresh 65,536-step corrected baseline. Inspect positive-reward count, development score, entropy, advantage scale and KL before increasing the budget. Then run the matched credit-horizon ablation if reward discovery/credit remains inadequate. Neither a new long run nor the horizon ablation has been launched here.

If both arms still rarely encounter rewards, a separately labeled pixel-only exploration experiment (for example, image-based novelty) is more informative than increasing model size. That would change the training objective and needs separate ablation; it is not implemented or enabled here. No handcrafted Tetris reward features are proposed.

## Reference context

The ROM bytecode, local JAX reproduction and controlled rollouts are the evidence for these findings. External references provide context, not a substitute for the measurements:

- [Timendus CHIP-8 timer test source](https://github.com/Timendus/chip8-test-suite/blob/main/src/tests/7-beep.8o) explicitly waits for the delay timer to reach zero.
- [PPO paper](https://arxiv.org/abs/1707.06347) describes the clipped surrogate and repeated minibatch updates; the normalization floor and zero value initialization here are experimental implementation choices.

### Completed short validation

The fresh 8,192-step run completed with exact checkpoint reload and finite diagnostics. It observed no positive training reward, so it cannot establish learning. Raw advantages and value loss stayed zero; final entropy was 1.609437 (uniform five-action entropy is log(5)=1.609438), final gradient norm about 8.6e-7. The eight final development games all ended naturally with score zero and mean length 1,160.5; no time-limit truncations occurred. This supports removal of the reward-free policy-drift mechanism in this short run.

85 regression tests passed. Original checkpoint episode arrays reproduced exactly under legacy timing. The validation artifacts are in `runs/tetris/v2_validation_8192/`; corrected-timing sanity games are in `runs/tetris/sanity_frame_v2/`. The short run's 86.6 steps/s was measured while other diagnostics shared the CPU and is not a clean throughput benchmark.
