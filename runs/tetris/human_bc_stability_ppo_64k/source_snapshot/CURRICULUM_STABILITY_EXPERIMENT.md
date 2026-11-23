# Smaller PPO steps after offline cloning

Declared before fresh training. The completed original-rate pilot scored 0/128
for both training seeds, versus their own fixed 100-epoch initializations at
7/128 and 1/128 and shared random at 3/128. A matched first-reward diagnostic
identified a destructive update. On that fixed batch, learning rate .000005
with a .015 categorical-KL stopping threshold accepted all 64 updates and ended
at exact full-batch KL .00403. That is local optimization evidence, not gameplay
success; the guard was inactive and its added benefit has not been established.

## Fixed candidate and stage boundary

Run `run_curriculum_stability.py` after component regressions pass. It first
launches a separately counted 2,048-decision seed-1 smoke, then independent fresh
PPO processes for training seeds 1 and 2, each at 65,536 new decisions. Read only
the original 100-epoch offline encoder/actor weights from disk; do not initialize
from the failed pilot, diagnostic branches, the smoke, or offline training state.
The PPO process creates its own optimizer, critic, RNG and environments. No human
dataset, replay, auxiliary objective or intrinsic reward is used by PPO.

Freeze learning rate .000005 (100 times below the failed pilot), target_kl .015,
and entropy coefficient zero. Keep the CNN, pixels, keypress actions, corrected
frame timing, original game reward, 16 environments × 128 steps, four epochs,
16 minibatches, gamma .995, GAE lambda .98, clip .2, value coefficient .5,
global gradient clipping .5, advantage standard-deviation floor .01 and fresh
zero-output critic. Sampling remains stochastic. This is a bundled stability
candidate, not an isolated attribution of learning rate versus the KL stop.

## KL semantics and accounting

Before each proposed update, compare the rollout-start and current full
categorical action distributions on the current shuffled minibatch. If their
mean old-to-current KL exceeds .015, stop the shared optimizer for the remainder
of that rollout, including actor, critic and shared encoder. Skip gradient
calculations after the stop. Preserve the ordinary epoch RNG split schedule.
The guard resets for each new rollout. It does not roll back an accepted update
and is not a hard full-rollout KL bound. Log final full-rollout KL separately.

Record accepted optimizer updates, gradient evaluations, minibatches checked,
skipped updates and stop frequency. The environment budget remains fixed; actual
optimizer updates can be below 2,048 per seed. Ordinary reward-free PPO updates
still execute and advance Adam normally; the diagnostic's analytical shortcut
is not used in training.

## Admission, audits and comparison

The seed-1 smoke must retain encoder/actor weights exactly with zero rewards and
gradients, execute 64 updates, pass full-checkpoint roundtrip, and reproduce the
first two archived initial monitoring episodes. It is separate from the pilot.
Both full runs must reproduce all eight archived initial monitoring episodes,
keep frozen source/config/weight hashes, complete 32 rollouts, and reconcile
accepted-update metrics with the final optimizer clock. Check all metrics and
parameters for finiteness. Seed 1 should reproduce its known reward-free prefix
and first reward at rollout end 28,672; compare its actual first-reward KL with
the local probe as an integration check, not a scoring claim.

Evaluate each final checkpoint once on the shared exploratory seeds 50000–50127.
Compare with that seed's exact pretrained policy, the failed original-rate PPO,
scratch entropy-zero PPO, scratch longer-credit entropy-.01 PPO, untrained
network, random and no-op. Require random/no-op episode arrays to match archived
controls. Report original returns, scoring games, reward discovery/timing, action
distributions, parameter movement and actual compute counts. Do not pool columns
as 256 independent games. No checkpoint selection by score or automatic tuning.

The exploratory screen requires both seeds to exceed their own pretrained
policy and random in scoring-game count. Passing it would justify independent
confirmation, not goal completion. The reserved 60000–60511 bank stays unused
during this pilot. Retention, small KL or successful tests are not useful learning.
Use the preselected final seed-1/game-30016 trajectory for the actual-policy card.

## Outputs and reproducibility

- Smoke: `runs/tetris/human_bc_stability_smoke_seed1`.
- Fresh runs: `runs/tetris/human_bc_stability_ppo_seed{1,2}`.
- Combined evidence/card: `runs/tetris/human_bc_stability_ppo_64k`.
- Reporter: `summarize_curriculum_stability.py`.

Archive exact sources and checksummed weights, final checkpoints, configs,
metrics, episode arrays and audits. Keep raw human archives local. Published
weights suffice for these independent PPO runs; recreating offline fitting still
requires the separately supplied human recording. Offline, smoke, training,
diagnostic replay, evaluation and illustration costs remain distinct.
