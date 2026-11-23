# Fresh on-policy PPO after offline cloning: two seeds at 64K

**Fresh PPO does not establish a replicated scoring gain beyond offline initialization and random play.**

| Policy | Training seed 1 | Training seed 2 |
| --- | ---: | ---: |
| ppo | 0/128 | 0/128 |
| pretrained | 7/128 | 1/128 |
| scratch_entropy_zero | 0/128 | 0/128 |
| scratch_entropy_001 | 1/128 | 1/128 |
| untrained | 2/128 | 0/128 |
| random | 3/128 | 3/128 |
| noop | 0/128 | 0/128 |

The offline advantage failed to replicate (7/128 and 1/128 versus shared random
3/128). The bounded PPO protocol was declared before seed 2's gameplay result,
kept both fixed 100-epoch initializations, and did not select a better duration
per seed. Initial eight-game arrays exactly match each offline policy. The
comparison above separates the starting-policy benefit from changes due to RL.

## Actual online training

Each independent process reads only the weights artifact, constructs fresh
optimizer/critic/RNG/environment state and collects 65,536 current-policy
decisions. There is no demonstration loader, replay, auxiliary loss or intrinsic
reward in PPO. Entropy regularization is zero, but collection/evaluation still
sample actions. The matched scratch entropy-zero control has identical settings
apart from initialization/provenance/output directory. Full source, configuration,
checkpoint, counter, initial-evaluation and random/no-op control audits pass.

| Seed | Positive training reward events | Completed training games | Encoder/actor L2 change |
| --- | ---: | ---: | ---: |
| 1 | 1 | 84 | 9.0463 |
| 2 | 1 | 50 | 9.4647 |

Reward timing, first-reward loss/advantage context, action distributions, finite
gradient checks, full metrics and paired exploratory comparisons are retained
in [analysis.json](analysis.json). Parameter movement or discovered rewards alone
are not proof of better play. Both seeds complete 32 rollouts and 2,048 optimizer
updates; this is 131,072 fresh decisions and 4,096 PPO updates combined.

The offline experiments additionally used 4,320 updates and 537,600 presentations,
including both 20-epoch references and fresh 100-epoch fits. They reuse only 2,240
human decisions from one incomplete prefix. Validation replays and evaluation
steps are separately recorded, not counted as fresh training experience. Raw
human data remains local, but all PPO inputs are published weights/configs.

## Limits and next decision

The two columns share the same 128 exploratory game seeds; they are not 256
independent games. Paired tests are unadjusted diagnostics after many investigations,
not confirmatory evidence. No checkpoint was chosen by its score and the reserved
60000–60511 confirmation bank remains unused. The candidate screen requires both
PPO seeds to exceed their own initialization and random; its outcome is
`False`. Do not mark the overall goal complete based on this screen alone.

The animation is the preselected seed-1/game-30016 final PPO trajectory.
[Card](result_card.html) · [Protocol](../../../CURRICULUM_PPO_EXPERIMENT.md) ·
[Offline comparison](../human_bc_duration_100epoch/RESULT.md) ·
[Implementation log](../../../IMPLEMENTATION_LOG.md).
