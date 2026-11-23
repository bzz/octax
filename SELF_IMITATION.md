# Score replay: the next pixel-only learning test

The three-seed 64K comparison did not establish learning: baseline reward events were [0, 0, 0], longer-credit events [1, 2, 0], and final scoring was below the shared random control. More feedback in the longer-credit arm did not produce a consistent gain over its untrained networks. This experiment tests whether retaining successful experience helps; it does not assume that replay solves Tetris.

## Method and constraints

Keep the same four-frame monochrome CNN, actor/value heads, corrected game timing, keypress actions, original score reward, and longer-credit PPO settings. No board features, demonstrations, new reset distribution, intrinsic reward or recurrent state is added. The auxiliary updates consume only stored pixel observations, the agent's own actions, and discounted original game rewards from completed episodes.

The replay objective follows the positive-return-advantage idea in [Self-Imitation Learning, Oh et al. (ICML 2018)](https://proceedings.mlr.press/v80/oh18b/oh18b.pdf): reinforce a recorded action only when its observed return exceeds the current value estimate, and fit the value toward that better return. Stop gradients through the advantage in the actor term. This is a local PPO+SIL variant, not a reproduction of the paper's Atari or MuJoCo experiments; its Tetris efficacy is unproven.

For an observation/action/return triple, let `adv = max(return - value, 0)`. The actor loss is `-log_probability * stop_gradient(adv)`; the value loss is `0.5 * adv**2`. Both use replay sampling weights. The auxiliary loss scale is 0.1 and value coefficient 0.01. Sampling uses advantage-based priorities, exponent 0.6 and partial bias correction 0.1. These are starting choices informed by the [paper's supplement](https://proceedings.mlr.press/v80/oh18b/oh18b-supp.pdf), not demonstrated optimal settings for this task.

Store at most 50,000 completed-episode transitions. Lossless bit packing stores binary images without adding features. Complete episodes are accumulated independently per environment; returns cannot cross a reset. Use ten replay minibatches of 128 per PPO rollout after activation. A batch with no positive advantage leaves both weights and optimizer state unchanged.

An explicit activation rule waits until a scoring episode completes. Before that, replay cannot change optimizer or collection RNG state. Replay sampling uses a separate persisted NumPy RNG. This preserves the initial v2 training path and makes the first retained success a useful mechanism checkpoint.

## First controlled run

Fresh seed 1, 65,536 environment decisions, no transferred/resumed weights. Compare against the already completed longer-credit seed-1 run: same rollout 128, gamma 0.995, lambda 0.98, 16 environments, four PPO epochs and minibatch size 128. Seed 1 is chosen because the control encountered feedback early; this is a mechanism test, not an independent training-seed replication or a claim that the seed generalizes.

All extra replay updates and their time are logged. The environment-step budget is matched, but compute/optimizer-update counts increase. Evaluate original game score on the same exploratory 128-game bank, seeds 50000–50127, and compare to the exact untrained seed-1 network and uniform random. Gameplay seed 30016 remains illustrative. New seeds starting at 60000 remain reserved for a frozen candidate's confirmation.

`run_self_imitation.py` records exact sources and ROM, runs fresh training, evaluates, and renders gameplay. It refuses an existing run directory and requires the fixed replication to be complete. For a separate reproduction, use a fresh output path:

```sh
JAX_PLATFORMS=cpu .venv/bin/python train_self_imitation.py run_dir=runs/tetris/sil_reproduction_seed1
```

## Checkpoint and validation contract

Save the full model/optimizer/RNG/emulator state and replay buffer, including pending episodes and replay RNG. A hash manifest binds model and replay at the same step; resume rejects mismatched/corrupt pairs or changed replay settings. Every evaluation checkpoint verifies exact model-state/inference and replay restoration. Successful episodes are also exported as packed pixel/action/reward/return arrays for inspection, and completed game returns are logged separately from individual reward events.

Six focused tests cover terminal/environment boundaries, loss-gradient direction, stopping the actor's value shortcut, exact replay continuation, inactive replay preserving nonzero Adam momentum, and positive replay updating the actual CNN actor/value heads. The complete suite passes 93 tests. A separate 2,048-step run in each trainer produced identical complete model, optimizer, RNG and emulator state before replay activation; checkpoint-pair integrity also passed. Those 4,096 validation decisions are not part of the matched learning budget. GIF rendering was stubbed during that state-equivalence check; the live investigation uses the normal renderer.

The experiment is useful only if actual scoring improves. Replaying a successful trajectory can overfit its pixels/actions, and extra optimization alone is not evidence of skill. The goal remains a reloadable pixel policy with convincing score evidence against random on unused evaluation seeds.
