# Fresh on-policy PPO after fixed offline cloning

Prepared before any full curriculum PPO run. Seed 1's fixed 100-epoch offline
clone passes the exploratory screen (7/128 scoring games versus random 3/128
and the 20-epoch clone 0/128). Seed 2's offline replication is still in progress
when this protocol is declared. The offline screen is not confirmation, and
supervised improvement does not establish an on-policy learning result.

## Question and admission

Does 65,536 decisions of fresh, original-score PPO improve each 100-epoch
initialization? Complete the seed-2 offline comparison first and report both
initializations, even if their scoring advantage differs. The seed-1 exploratory
lead motivates this bounded two-seed PPO pilot; do not drop a weaker seed or
select different pretraining durations per seed. Admission requires both fixed
100-epoch artifacts to pass their fit, checkpoint, data, source, exact epoch-20
state and standalone-evaluation checks. If the offline advantage does not
replicate, label that result explicitly; this pilot then tests PPO's response to
different-quality initializations, not a confirmed offline method. No 512K
extension is authorized by this protocol alone.

## Independent process and fixed optimization

For training seeds 1 and 2, launch `train_pretrained_ppo.py` in a new process with
the corresponding checksummed `policy_weights.msgpack`, `--entropy-coef 0`, and
a new output directory `runs/tetris/human_bc_100epoch_ppo_seed{seed}`. The trainer
initializes its own critic, optimizer, RNG and environment state and imports no
dataset/cloning code. No observations, actions, returns, replay buffer, optimizer
or environment state are transferred from offline training. All PPO updates use
fresh current-policy rollouts. The actor remains stochastic during collection
and evaluation; entropy coefficient zero does not mean greedy action selection.

Use the existing matched longer-credit profile: 16 environments × 128 decisions,
32 rollouts, 65,536 decisions and 2,048 PPO optimizer updates per seed. Adam
learning rate .0005, four epochs, 16 minibatches, gamma .995, GAE lambda .98,
clip .2, value coefficient .5, global gradient clipping .5, advantage standard
deviation floor .01, fresh zero-output critic, original CNN and corrected frame
timing. The only intentional difference from the saved scratch entropy-zero
control is initialization and its provenance. Offline work is additional cost,
not included in the PPO decision budget.

Entropy is disabled because the earlier matched handoff smoke isolated entropy-
driven erasure of the clone before task feedback. This preserves a meaningful
initialization but is not proof of adequate exploration. Do not add an entropy
schedule, replay, novelty reward, shaping or auxiliary loss during this pilot.

## Required comparisons and audit

Before judging training, require the PPO step-zero eight-game monitoring arrays
to exactly equal the corresponding pretrained-only arrays. Audit the copied
weights hash, complete source snapshots, fresh counters and final full-state
checkpoint; require all loss/gradient metrics finite and the full fixed budget.
The existing six component tests and matched independent-process smokes validate
the loading boundary; actual-run initial arrays verify this invocation.

Evaluate the final checkpoint on the same 128 exploratory game seeds 50000–50127
using original scores. Compare each seed separately with its exact pretrained
policy, the scratch entropy-zero 64K control, the original entropy-.01 credit
control, its untrained network, random and no-op. Require shared random/no-op
episode arrays to match. The game bank is shared, so do not pool columns as
independent games. Report reward-discovery counts/timing, completed games,
original returns, score changes, action distributions and parameter movement.
Use a preselected seed-1/game-30016 actual-policy illustration, not a selected win.

No checkpoint selection by evaluation, no automatic restart with changed
hyperparameters and no fresh confirmation bank during this pilot. A candidate
for subsequent confirmation must score above both its own pretrained-only
policy and random in both training seeds. This exploratory screen is not a
significance test. If PPO preserves or harms the offline policy, record that
explicitly: initialization gains must not be credited to online RL. The overall
goal remains incomplete until a useful scoring improvement is established.
