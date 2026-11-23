# Pixel-only Octax PPO on macOS

This checkout implements the staged experiment in the supplied brief. It keeps Octax's PPO, GAE, pixel observations and keypress action space. The host runner calls the existing rollout and update methods separately so their timing is measurable.

## Reproduce the environment

Run commands from this repository directory. The tested machine is macOS arm64; CPU execution is intentional.

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-macos.lock.txt -e '.[training]'
export JAX_PLATFORMS=cpu
export MPLCONFIGDIR="$PWD/.matplotlib"
.venv/bin/python -m pytest -q
```

The lock file captures the installed versions. Gymnax is pinned to 0.0.9 because Octax/Rejax expect the five-value step API. Gymnax 1.1.0 has a six-value API. OpenCV-headless and Pillow are needed because Octax imports rendering utilities even without a GUI. Existing emulator code emits JAX dtype-cast deprecation warnings; these did not prevent the tests or runs.

Upstream: https://github.com/riiswa/octax, commit `3aa53b516152e97f2ed91eae6e33b6ee9a97596b`.

## Changes

- Convert `[batch, frames, height, width]` to `[batch, height, width, frames]` in a shared encoder method used by all actor/critic entry points.
- Use explicit two-dimensional kernel and stride tuples for all three convolution layers. Scalar Flax kernel sizes select a one-dimensional convolution.
- Preserve the shared 32/64/64 CNN and separate actor/critic 256-unit heads. Input is four monochrome 32×64 frames. There are no board features, shaping rewards, recurrence or transformers.
- Save actual model parameters, optimizer moments and update count, RNG, current emulator states, last observations and global step in a versioned MessagePack checkpoint. Saves are atomic. Reloaded states and policy outputs are checked exactly.
- Log PPO losses, entropy, gradient norm, reward events, episode completions, action proportions and separate collection/update times on every update.
- Run seeds sequentially, avoid silently overshooting the requested budget, and refuse to overwrite existing run folders.

`max_grad_norm=0.5` is explicitly configured. Octax's existing implementation uses elementwise `optax.clip`, despite the inherited setting name; this experiment preserves that behavior. CPU timings exclude evaluation and most host bookkeeping; summary.json also records wall time including periodic evaluation. The first rollout/update timing includes JIT compilation; the steady estimate excludes that pair.

## Stage A: controls and baseline games

```sh
.venv/bin/python sanity.py
.venv/bin/python inspect_reward.py
```

`runs/tetris/sanity/` contains eight random and eight no-op episodes, GIFs, control probes and a positive-reward replay. The controls are indices 0–4: rotate (CHIP-8 key 4), left (5), right (6), down (7), no-op. Raw observations are `(4,64,32)`; the Gymnax wrapper converts them to `(4,32,64)` before the CNN conversion.

Resets use a cached startup snapshot: the initial image and first piece are identical. The supplied RNG changes subsequent pieces; full no-op episode lengths vary across keys. The short 120-step probe alone does not establish reset diversity. This preserves upstream reset semantics.

The random policy with evaluation seed 10002 obtains +1 at step 2330. `reward_validation.json` verifies every step's reward equals the score difference and that score equals V[10]. `positive_reward.png` captures the associated frames. This establishes that positive reward is reachable without Tetris heuristics.

## Stage B: smoke test

```sh
.venv/bin/python train.py run_dir=runs/tetris/my_smoke
```

Defaults: seed 0, 16 environments, 32 steps/rollout, four minibatches, four epochs, learning rate 0.0005, 65,536 steps, eight evaluation episodes. Evaluation seeds 10000–10007 are distinct from the training initialization. GIFs use seed 10008. Stochastic policy sampling is used consistently for baseline and learned-policy assessment.

Each run contains:

- `config.yaml`, `system.json`
- `checkpoint.msgpack` (latest evaluated state, including optimizer and environment)
- `metrics.csv`, `evaluation.json`, `summary.json`
- `eval.gif` and `eval_STEP.gif` for each evaluated state

The final weights are not evidence of successful learning by themselves. Compare raw game score against controls, not survival alone.

## Stage C: longer runs, prepared but not launched

```sh
./run_stages.sh pilot
```

This trains three independent seeds (0, 1, 2), sequentially, for 499,712 steps each: the closest complete 512-step rollout below 500K. Evaluations occur every 65,536 steps and at the final step. Each final policy is then compared with random and no-op on 64 held-out seeds (30000–30063).

For one pilot only:

```sh
.venv/bin/python train.py total_timesteps=499712 run_dir=runs/tetris/pilot_one
.venv/bin/python evaluate_checkpoint.py runs/tetris/pilot_one/checkpoint.msgpack --out runs/tetris/pilot_one/heldout
```

To continue the existing smoke checkpoint with its optimizer and RNG intact:

```sh
.venv/bin/python train.py total_timesteps=499712 resume=runs/tetris/ppo_seed0_smoke/checkpoint.msgpack run_dir=runs/tetris/continued_seed0
```

Total timesteps refers to the cumulative step target when resuming. Resume requires the same environment, seed, parallelism and optimizer/PPO settings. To test 32–64 environments, start a separate run and keep rollout/minibatch divisibility valid. Compare throughput as well as learning; larger batches are not guaranteed to be faster.

Do not interpret eight evaluation episodes or one training seed as statistically convincing improvement. Use all three independent training seeds, preserve per-episode returns, and examine uncertainty of the difference from random. Extend toward 2M–5M only if rewards and evaluation trends justify it. At a sparse-reward plateau, inspect controls and reward events before scaling model capacity or changing rewards.

## Stage D: matched transfer experiment, prepared but not launched

```sh
./run_stages.sh transfer runs/tetris/pilot/seed0/checkpoint.msgpack
```

For Brix and Pong, this runs scratch and Tetris-encoder initialization for target seeds 0, 1 and 2. Only encoder weights transfer; both action/value heads and optimizer moments start fresh. Target seeds, step budgets, optimizer settings and evaluation seeds match within each pair. Each condition uses 499,712 steps and a final 64-episode held-out comparison.

Compare evaluation learning curves at matched target-environment steps. Report pretraining cost separately. This design uses one fixed source checkpoint; it measures transfer from that checkpoint, not variability across Tetris pretraining seeds.

## Tests

The upstream emulator tests are complemented by tests for all CNN entry points and kernel shapes, terminal-aware GAE, and exact optimizer/RNG checkpoint restoration. The training runner checks finite diagnostics/weights, and checkpoint reload equality at each evaluation. The transfer experiment is prepared but has not been executed or validated on target games.

## Visualize gameplay and results with MinAtar's plotting tools

MinAtar's `Environment.display_state()` uses Matplotlib `imshow`, a discrete colormap and Seaborn's cubehelix colors. `visualize_run.py` uses those same tools for the exact Octax pixels (one foreground channel), with no feature extraction or change to the training observations.

```sh
.venv/bin/python visualize_run.py --run runs/tetris/ppo_seed0_smoke
```

This produces learning curves (PNG/PDF), a side-by-side random/PPO gameplay GIF, a still image, and compressed raw trajectory files. The gameplay uses held-out seed 30016, records the complete episode through termination/truncation, and shows eight-action-spaced frames at approximately 4× simulated game speed. Policy sampling remains stochastic. The original short monochrome `eval.gif` is also retained.

Reference: https://github.com/kenjyoung/MinAtar/blob/master/minatar/environment.py

For a local live Matplotlib window, replay either saved trajectory:

```sh
.venv/bin/python replay.py runs/tetris/ppo_seed0_smoke/visualizations/policy_trajectory.npz --speed 4
```

This uses Matplotlib's native animation window. Exported GIFs/PNGs do not require a GUI backend.
