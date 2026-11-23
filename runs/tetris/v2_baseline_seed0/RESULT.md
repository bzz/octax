# Fresh v2 Tetris baseline: 65,536 steps

The run is complete. This evaluation does not establish a scoring advantage over random play.

## Reward discovery during training

- PPO: **0 positive reward events**, total original game reward 0, over 65,536 steps and 44 completed episodes. First event: not observed.
- Uniform-random control: **0 positive reward events**, total reward 0, over the same 65,536-step budget and 42 completed episodes.
- The random control matches initial reset states, seed, parallelism and step budget; action draws and episode boundaries are not identical. This is one realization per condition, not a replicated estimate of event frequency.
- The value of the fixes is evaluated separately from proving learned skill. Correct timer behavior and stable reward-free updates are verified engineering results; improved Tetris performance requires score evidence.

## Corrected-timing final comparison

All three policies use `timer_mode=frame`, the same action space, the same raw score reward and 128 matched seeds 50000–50127. These evaluation seeds were selected before observing this run and were unused in the earlier experiments.

| Policy | Games scoring | Mean score | Mean episode decisions | Time-limit truncations |
| --- | ---: | ---: | ---: | ---: |
| PPO | 1/128 | 0.00781 | 1308.7 | 0 |
| Random | 3/128 | 0.02344 | 1273.9 | 0 |
| No-op | 0/128 | 0.00000 | 777.7 | 0 |

There were 1 PPO-only and 3 random-only scoring seed pairs. The exact paired test for scoring probability gives p=0.625. This test does not establish robustness across independently trained agents, nor test a difference in mean score magnitude. Per-policy Wilson intervals and all raw counts are retained in `analysis.json`; per-game results are in `heldout/comparison.json`.

## Training checks and timing

- Fresh seed-0 initialization; no resumed weights or transferred encoder.
- 16 environments × 32 steps, four minibatches/epochs, unchanged gamma=0.99 and lambda=0.95.
- Pixel observations and original score-only reward; no board features, shaping or action-repeat change.
- Corrected timers, zero value output initialization, advantage std floor 0.01 and global-norm clip 0.5.
- Final entropy: 1.609438; final approximate KL: 0.
- All recorded diagnostics/parameters were checked finite during training. Saved checkpoint state and inference reloaded exactly.
- Observed training throughput: 114.1 steps/s. Training plus final development evaluation/export: 9.8 minutes, excluding initial evaluation and the final 128-game comparison. The equal-budget random control briefly shared the CPU, so this is practical throughput rather than a hardware benchmark.

## Outputs and limits

[Learning curves](visualizations/learning_curves.png) · [Gameplay](visualizations/gameplay.gif) · [Checkpoint](checkpoint.msgpack) · [Raw comparison](heldout/comparison.json) · [Analysis](analysis.json)

The curves use the eight-game corrected-timing development baseline. The table above is the separate final 128-game comparison. One displayed gameplay seed is illustrative, not a substitute for aggregate evaluation. Historical v1 results used different timer dynamics and must not be pooled with this experiment.

No extension, additional training seed or credit-horizon ablation was launched. The prepared longer-credit profile remains a separate next experiment.
