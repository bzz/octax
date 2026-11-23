# Smaller-step on-policy PPO: two seeds at 64K

**Smaller-step PPO does not establish a replicated scoring gain above its starting policies and random play.**

| Policy | Training seed 1 | Training seed 2 |
| --- | ---: | ---: |
| Smaller-step PPO | 3/128 | 3/128 |
| Pretrained starting policy | 7/128 | 1/128 |
| Previous PPO rate | 0/128 | 0/128 |
| Scratch PPO, entropy 0 | 0/128 | 0/128 |
| Scratch longer-credit PPO, entropy .01 | 1/128 | 1/128 |
| Untrained network | 2/128 | 0/128 |
| Uniform random | 3/128 | 3/128 |
| No-op | 0/128 | 0/128 |

Each policy was evaluated on the same 128 exploratory game seeds. The columns
are not 256 independent games. Original returns, episode lengths, action
distributions, paired comparisons and all audit results are in
[analysis.json](analysis.json). Offline gains must not be attributed to PPO.

## Fixed change and actual work

Two independent fresh PPO processes read the original offline weights from disk
and construct their own optimizer, critic, RNG and environments. Learning rate
is .000005 (100× below the failed pilot), with categorical-KL stopping at .015.
All other training settings remain fixed, including entropy zero, pixel CNN,
original rewards and stochastic actions. No demonstration replay or auxiliary
loss is used. This bundled candidate does not isolate the two changes' effects.

| Seed | Training reward events | Accepted optimizer updates | Rollouts stopped | Maximum final full-batch KL |
| --- | ---: | ---: | ---: | ---: |
| 1 | 3 | 2033 | 1 | 0.00894 |
| 2 | 2 | 2048 | 0 | 0.00391 |

Each seed completes 32 fresh rollouts and 65,536 decisions. Actual accepted
updates total 4,081; gradient evaluations equal accepted updates. No
gradients or shared-optimizer changes occur after stopping within a rollout.
The stop checks the current minibatch before the next update: it cannot undo
overshoot and does not impose a hard full-rollout KL cap.

The separate seed-1 smoke used 2,048 decisions and 64 optimizer updates. It
retained encoder/actor weights and initial episode arrays exactly with zero
reward and gradients. Full pilots reproduce all eight initial monitoring
episodes, pass source/config/checkpoint/count audits, and reproduce random/no-op
controls. Seed 1's known pre-reward rollout summaries and first reward at 28,672
match the previous pilot; its first-reward KL matches the diagnostic probe.

Offline work remains separate: 4,320 updates and 537,600 example presentations,
including reference fits, from only 2,240 human decisions. Diagnostic replays,
evaluation and illustrations are not fresh training experience. Raw human
recordings remain local; published weight artifacts suffice for PPO reproduction.
Seed-2 throughput varied substantially during high host load. Raw timings remain
in the metrics, with a [runtime observation](runtime_observation.json); they are
not an isolated CPU benchmark.

## Interpretation

Training reward events are [3, 2]
across seeds, compared with [1, 1] in the previous pilot. Its final policies
scored [0, 0]/128; the current final policies score
[3, 3]/128. These comparisons
distinguish continued reward discovery from held-out scoring gains. The earlier
seed-1 first-reward diagnostic ended at exact full-batch KL 15.7814; the actual
new update ends at 0.00403.
This validates control of that previously destructive update, not successful RL.

The exploratory screen requires both seeds to score above their own pretrained
policy and random; outcome: `False`. Small KL, retained preferences and
parameter movement do not establish useful learning. Paired test results are
unadjusted diagnostics after multiple investigations. No checkpoint was selected
by evaluation, and confirmation seeds 60000–60511 remain unused.

The card uses the preselected final seed-1/game-30016 trajectory, whether it wins
or loses. [Actual-policy card](result_card.html) ·
[Protocol](../../../CURRICULUM_STABILITY_EXPERIMENT.md) ·
[Previous pilot](../human_bc_100epoch_ppo_64k/RESULT.md) ·
[Implementation log](../../../IMPLEMENTATION_LOG.md).
