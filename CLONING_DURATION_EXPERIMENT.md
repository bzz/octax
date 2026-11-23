# Offline cloning duration: 20 versus 100 epochs

Declared before the 100-epoch run. The completed 20-epoch seed-1 clone reaches
training CE 0.4201 but never predicts rotation as its most likely action on the
47 rotation labels. Its stochastic original-score evaluation is 0/128, versus
random 3/128. The same-input empirical CE floor is 0.12386; that bound does not
prove the current CNN can reach it or that improved fit produces useful play.

## Fixed comparison

Run seed 1 from the identical initialization for exactly 100 epochs. Preserve
all 2,240 pixel/action pairs, shuffled batches of 128, Adam .0005, global clip
.5, network, preprocessing and action timing. This is 1,800 optimizer updates
and 224,000 example presentations. Do not rebalance actions, select checkpoints
by game score, add reconstruction, or change architecture in this experiment.
At epoch 20 require the entire cloning state (parameters, optimizer, RNG and
counters) to equal the archived 20-epoch state exactly. Fail rather than silently
continue an unmatched comparison. Offline checkpoint reuse is validation only;
the run starts fresh. No PPO training is included in this offline experiment.

The original human archive remains local and excluded from publication. Its
SHA-256 is 7a5a60ca8632cd751a23a2553d21762254b8aa208a94f658d835f6abc3c05cec.
It is a validated prefix of one game, not an independent validation set.

```sh
JAX_PLATFORMS=cpu MPLCONFIGDIR=../../work/mpl .venv/bin/python pretrain_human_policy.py \
  --seed 1 --epochs 100 \
  --dataset runs/tetris/human_bc_seed1/pretrain/dataset.npz \
  --reference-20-epoch runs/tetris/human_bc_seed1/pretrain/cloning_checkpoint.msgpack \
  --run runs/tetris/human_bc_100epoch_seed1/pretrain
```

Evaluate the final weights only, with the existing actor-only evaluator: verify
the eight monitoring episodes, then 128 original-score games 50000–50127 with
matched random/no-op controls, plus the preselected illustration seed 30016.
Report training CE/entropy, per-action recall and confusion, unchanged dataset,
matched initialization, source hashes, exact 20-epoch audit and offline budget.
All fit diagnostics are in-sample. Compare scores with the 20-epoch clone,
untrained seed-1 network and matched scratch PPO, not only the training loss.

The reused game bank is exploratory. A final policy scoring in more than 3/128
games (the shared random result) and above the 20-epoch clone is a candidate for
seed-2 replication and a separately declared on-policy PPO experiment. That
screen is not a significance test or confirmation. If stronger fit still fails
this screen, diagnose data coverage/observation ambiguity rather than declare
success or automatically extend PPO. Keep seeds 60000–60511 unused.

If a PPO follow-up is justified, it remains a separate process reading only the
weights file, with fresh critic/optimizer/environment state and fresh rollouts.
Entropy-zero preserves the initialization in a reward-free smoke, but that is
not proof it supplies sufficient exploration. Account for offline gains and
subsequent RL gains separately. Future architecture experiments are allowed.

## Seed-2 replication triggered

Seed 1 completed with final fit CE 0.1703 and 7/128 scoring games versus the
20-epoch clone 0/128 and random 3/128. This passes the exploratory screen, not a
significance test. The fixed seed-2 replication uses the same data and settings.
First generate its 20-epoch reference, then start a fresh 100-epoch process and
require an exact epoch-20 full-state match. Evaluate both final policies on the
same declared bank. The two fits cost 2,160 updates and 268,800 presentations;
their duplicated first 20 epochs must not be hidden in budget accounting.
This is an independent initialization, not an independent human demonstration
or fresh confirmation bank. Run `run_cloning_duration_replication.py`; no PPO
process is started by that launcher. The raw dataset remains local.
