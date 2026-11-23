# Pixel-action pretraining, then independent on-policy PPO

This recipe separates reusable stage implementations from experiment settings and run artifacts. The current fixed CNN is a control choice; future attention/architecture experiments are within scope. See [the curriculum rationale](../../CURRICULUM.md).

## Stage 1: offline human-action cloning

The raw human recording is excluded from the public repository. Reproducing
this stage and its training-prefix diagnostics requires a separately supplied
copy matching the checksum below. The published weights suffice for stage 2
and standalone gameplay evaluation.

```sh
JAX_PLATFORMS=cpu .venv/bin/python pretrain_human_policy.py \
  --seed 1 --dataset PATH_TO_PINNED_HUMAN_EPISODE \
  --run runs/tetris/human_bc_seed1/pretrain
```

The script requires the recorded archive with SHA-256 `7a5a60ca8632cd751a23a2553d21762254b8aa208a94f658d835f6abc3c05cec`. It copies that version, independently replays it for validation, and trains on `observations[:-1]` and actions only. It preserves all no-op timing. The prefix has 2,240 decisions, 1,323 unique stacks and 1,715 no-op labels (76.56%); majority-action accuracy alone would be misleading.

Fixed settings: the original CNN/actor initialization for seed 1 or 2, Adam learning rate .0005, global gradient norm limit .5, batch size 128, 20 shuffled epochs, 360 updates and 44,800 example presentations. No critic is optimized. These settings are declared before real pretraining; do not tune them against game evaluation results. With one trajectory, fit loss is a training diagnostic, not held-out generalization. Require final training cross-entropy below both its initial value and the empirical constant-action predictor before advancing this particular experiment to a full PPO pilot. If it fails, diagnose underfitting/data problems first; do not describe a failed fit gate as successful pretraining.

Outputs include the dataset/validation manifest, frozen sources, training metrics, a resumable **cloning-only** checkpoint, initial/pretrained original-score monitoring evaluations and `policy_weights.msgpack`. The weights file's checksum is `policy_weights_sha256` in `summary.json`.

## Stage 2: a new PPO process

```sh
JAX_PLATFORMS=cpu .venv/bin/python train_pretrained_ppo.py \
  --seed 1 --weights runs/tetris/human_bc_seed1/pretrain/policy_weights.msgpack \
  --weights-sha256 SHA256_FROM_PRETRAINING_SUMMARY \
  --run runs/tetris/human_bc_seed1/ppo
```

This program imports the ordinary PPO runner and the weights-only initialization module. It does not load demonstrations, cloning code or its optimizer. It creates the target seed's fresh optimizer, critic, RNG and environments before loading only CNN/actor tensors from disk. PPO trains solely on fresh rollouts with original game rewards. The first process can have exited and the dataset can be unavailable to the PPO process.

Use `--smoke` with a separate output directory for a separately counted 2,048-decision handoff check before the full pilot. Otherwise the budget is 65,536 fresh PPO decisions, using the matched 128-step credit profile and 2,048 PPO optimizer updates. Repeat with seed 2; the human dataset is shared evidence, not two independent demonstrations. Keep pretrained-only and post-PPO evaluations so the contribution of RL is visible. Include random and scratch-PPO controls; the reused 128-game bank is exploratory and the fresh confirmation bank is reserved.

The 2,240 original demonstration decisions, 2,241 diagnostic replay transitions per validation, offline optimizer work, smoke decisions and actual fresh PPO decisions are different quantities and must be reported separately. No curriculum run has yet established a scoring improvement. Do not use the old concurrent self-imitation runner for this recipe.

## Completed first investigation

The frozen 20-epoch seed-1 fit passed the training-loss gate but scored 0/128 versus random 3/128. Both independent 2,048-step handoff smokes reproduced the initial policy exactly. Entropy .01 erased its preferences in the absence of task feedback; `--entropy-coef 0` preserved weights exactly. This is a diagnostic comparison, not a full 64K result or a claim that zero entropy solves learning. No full curriculum PPO run has started. See the [result and actual-policy card](../../runs/tetris/human_bc_seed1/RESULT.md) before launching a longer run.

Reproduce the read-only audit after those runs exist:

```sh
JAX_PLATFORMS=cpu .venv/bin/python diagnose_cloning_handoff.py \
  --pretrain runs/tetris/human_bc_seed1/pretrain \
  --entropy-smoke runs/tetris/human_bc_handoff_smoke_seed1 \
  --zero-smoke runs/tetris/human_bc_handoff_entropy_zero_seed1 \
  --out runs/tetris/human_bc_seed1/handoff_diagnostic_reproduced
MPLCONFIGDIR=../../work/mpl .venv/bin/python summarize_cloning_handoff.py
```

All fit/confusion metrics use the same training prefix; none is independent action validation. The audit reads completed files, runs no environments and performs no optimization. The next offline-fit experiment must have its own settings, output directory and evaluation, preserving this completed baseline.

## Later fixed experiments

The [100-epoch offline comparison](../../runs/tetris/human_bc_duration_100epoch/RESULT.md)
scored [7,1]/128 across seeds, versus shared random 3/128. The original-rate
[fresh 64K PPO pilot](../../runs/tetris/human_bc_100epoch_ppo_64k/RESULT.md)
then scored [0,0]/128, with a destructive first rewarded update in each seed.
These are completed later experiments; the first investigation above describes
the earlier 20-epoch handoff stage.

The [smaller-step experiment](../../CURRICULUM_STABILITY_EXPERIMENT.md) freezes
learning rate .000005, target KL .015 and entropy zero. Its orchestration script
launches a separate smoke and independent fresh PPO processes for seeds 1 and 2:

```sh
JAX_PLATFORMS=cpu .venv/bin/python run_curriculum_stability.py
```

The runner refuses existing output directories. Reproduce a single fresh stage
in a new directory with the published 100-epoch weights and their recorded hash:

```sh
JAX_PLATFORMS=cpu .venv/bin/python train_pretrained_ppo.py \
  --seed 1 --entropy-coef 0 --learning-rate .000005 --target-kl .015 \
  --weights runs/tetris/human_bc_100epoch_seed1/pretrain/policy_weights.msgpack \
  --weights-sha256 8dde4760ef6ff3512af083ce6f5b3cea04e7fdb90f526e0985c8da9f4bca1d20 \
  --run runs/tetris/reproduced_stability_seed1
```

With stopping enabled, the 65,536-decision budget remains fixed but actual
optimizer updates can be fewer than 2,048. The threshold is a pre-update
minibatch stopping rule, not rollback or a hard full-rollout divergence cap.
The PPO process still consumes weights only and performs no offline replay.

Completed smaller-step results: [3,3]/128 scoring games, equal to shared random, versus starting policies [7,1]/128. Training rewards [3,2] and smaller updates do not establish useful learning. [Report and actual-policy card](../../runs/tetris/human_bc_stability_ppo_64k/RESULT.md).
