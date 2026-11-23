# Pixel-only Octax PPO: current v2 experiment guide

The current baseline fixes emulator timing and sparse-reward PPO stability. Read [REWARD_DIAGNOSIS.md](REWARD_DIAGNOSIS.md) for measurements and [IMPLEMENTATION_LOG.md](IMPLEMENTATION_LOG.md) for history. The previous 65K/500K results remain intact, but used legacy timing and PPO settings. They do not establish learning superiority.

## Recreate the CPU environment

Run from this repository directory:

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-macos.lock.txt -e '.[training]'
export JAX_PLATFORMS=cpu
export MPLCONFIGDIR="$PWD/.matplotlib"
.venv/bin/python -m pytest -q
```

The lockfile captures installed versions. Gymnax 0.0.9 matches the five-value step API expected by Octax/Rejax. Headless OpenCV and Pillow are included because Octax imports rendering utilities even for CPU training. Some upstream emulator/dependency dtype deprecation warnings remain.

Upstream commit: `3aa53b516152e97f2ed91eae6e33b6ee9a97596b` from https://github.com/riiswa/octax.

## New baseline

```sh
.venv/bin/python sanity.py
.venv/bin/python train.py
```

The default output is `runs/tetris/ppo_seed0_v2_smoke/`. Run folders cannot be overwritten. Defaults: CPU, seed 0, 65,536 environment decisions, 16 environments, 32 decisions/rollout, four minibatches/epochs, learning rate 0.0005, gamma 0.99, lambda 0.95, eight development evaluation games.

The model consumes only four monochrome 32×64 frames and chooses rotate/left/right/down/no-op. Shared three-layer 2D CNN (32/64/64) and separate 256-unit actor/value heads. No shaping, board features, recurrence or transformers.

v2 adds per-frame saturating timers, zero initial value-output weights, an advantage standard-deviation floor of 0.01 and global-norm gradient clipping at 0.5. `experiment_version: 2` and `env_kwargs.timer_mode: frame` are recorded in each configuration.

Each run contains a complete atomic checkpoint (weights, optimizer, RNG and emulator state), exact reload checks, configuration/system metadata, per-update metrics, per-checkpoint evaluation arrays/GIFs and a final summary. Extra diagnostics include approximate KL, clipping fraction, advantage scales and clipped gradient norm.

## Evaluate and visualize

```sh
.venv/bin/python evaluate_checkpoint.py runs/tetris/ppo_seed0_v2_smoke/checkpoint.msgpack --episodes 64 --out runs/tetris/ppo_seed0_v2_smoke/heldout
.venv/bin/python visualize_run.py --run runs/tetris/ppo_seed0_v2_smoke
```

The evaluator uses the checkpoint's environment configuration. Old checkpoint configs select legacy timing automatically. Held-out seeds start at 30000, separate from development seeds 10000–10007. The 64-game set includes the 16 seeds previously inspected in v1; it should not be described as wholly untouched validation after repeated experiment selection. For a final confirmatory comparison, choose a new seed range (for example `--seed-start 50000`) and include all independent training seeds.

Visualization uses Matplotlib/Seaborn primitives used by MinAtar: discrete cubehelix color and uninterpolated pixels. It exports learning-curve PNG/PDF, random-versus-policy gameplay GIF/PNG and raw NPZ trajectories. It selects the baseline matching timer mode and rejects mismatches. Corrected baseline games live in `runs/tetris/sanity_frame_v2/`; legacy ones remain in `runs/tetris/sanity/`.

Local Matplotlib replay:

```sh
.venv/bin/python replay.py runs/tetris/ppo_seed0_v2_smoke/visualizations/policy_trajectory.npz --speed 4
```

## Next comparison — ablation not launched

The fresh corrected baseline is complete: see [the result](runs/tetris/v2_baseline_seed0/RESULT.md). For a new independent reproduction and the prepared longer-credit ablation, each at 65,536 decisions:

```sh
.venv/bin/python train.py run_dir=runs/tetris/v2_baseline_reproduction
.venv/bin/python train.py --config-name credit_horizon
```

The ablation changes rollout length to 128, minibatch count to 16 (same minibatch size), gamma to 0.995 and lambda to 0.98. Other settings match. It is a hypothesis to test after the corrected baseline, not a demonstrated improvement.

For the three-seed nominal 500K pilot, only once evidence justifies it:

```sh
./run_stages.sh pilot
```

This starts fresh seeds 0–2 sequentially, each for 499,712 steps, followed by 64-game evaluation. 499,712 is divisible by both baseline and ablation rollout sizes. Nothing starts merely by preparing this command.

To continue an existing **v2** baseline to a cumulative target:

```sh
.venv/bin/python train.py total_timesteps=499712 resume=runs/tetris/ppo_seed0_v2_smoke/checkpoint.msgpack run_dir=runs/tetris/v2_continued_seed0
```

Resume requires identical timer semantics, experiment version, initialization/normalization/clipping settings, seed, parallelism and PPO/optimizer configuration. Do not resume a v1 checkpoint into v2; start fresh. The historic `run_pilot.py` is explicitly pinned to legacy settings and its existing output folder; it is retained as the record of the earlier continuation.

## Reproduce the investigation

```sh
.venv/bin/python diagnose_rewards.py
.venv/bin/python diagnose_ppo.py
```

These write to `runs/tetris/reward_diagnosis/` and rerun the documented diagnostic conditions. `inspect_reward.py` deliberately replays the original legacy-timing scoring episode. It is not a corrected-timing baseline.

Transfer to Brix/Pong remains aspirational. The prepared transfer path has not been validated on those games and is outside the current Tetris investigation.

## Published baseline: inspect or reproduce

The repository includes the saved results, so use a new output directory when rerunning. To evaluate the published checkpoint without training:

```sh
.venv/bin/python evaluate_checkpoint.py runs/tetris/v2_baseline_seed0/checkpoint.msgpack --episodes 128 --seed-start 50000 --out runs/tetris/v2_recheck
```

To reproduce the fresh run and final comparison:

```sh
.venv/bin/python train.py total_timesteps=65536 run_dir=runs/tetris/v2_reproduction_seed0 resume=null encoder_checkpoint=null
.venv/bin/python evaluate_checkpoint.py runs/tetris/v2_reproduction_seed0/checkpoint.msgpack --episodes 128 --seed-start 50000 --out runs/tetris/v2_reproduction_seed0/heldout
.venv/bin/python visualize_run.py --run runs/tetris/v2_reproduction_seed0
```

`compare_random_budget.py` records the original equal-budget uniform-random control; `summarize_v2_baseline.py` rebuilds the published baseline analysis from stored episode arrays and metrics. These two scripts and `run_v2_baseline.py` target the original baseline directory. The launcher deliberately refuses to overwrite it. Use the direct commands above for a new run, and do not interpret the archived process-status files as live jobs.

The exact v2 source snapshot and manifest live inside the published run. Older experiments have more limited source provenance, detailed in [FORK_CHANGELOG.md](FORK_CHANGELOG.md). The dependency lock records the observed macOS environment; bitwise reproducibility across different JAX versions/backends/hardware is not asserted. New runs are ignored by Git and should be explicitly reviewed before publishing.
