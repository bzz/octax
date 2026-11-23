# Smaller PPO steps with KL stopping

This follow-up examines the same first-reward batch after the original diagnostic
showed that one .0005 step could already overshoot before an early-stop check.
It exports no candidate policy and performs no gameplay evaluation.

| Update profile | Accepted updates | Final exact full-batch KL | Encoder/actor L2 change |
| --- | ---: | ---: | ---: |
| Original .0005, no guard | 64 | 15.781416 | 8.279867 |
| .00005 + exact minibatch KL stop .015 | 1 | 0.017228 | 0.059839 |
| .000005 + exact minibatch KL stop .015 | 64 | 0.004025 | 0.111791 |

The .000005 profile is the next proposed **fresh-run** stability experiment:
it accepts all 64 updates with controlled divergence on this batch, while the
.00005 profile stops after one update and slightly exceeds the nominal threshold.
The guard did not activate in the .000005 probe, so that result does not establish
an additional benefit from the guard; it remains protection to test on later
batches. This choice is based on local update diagnostics, not a demonstrated
scoring gain. A full run must compare against its exact offline initialization,
the failed original-rate PPO pilot and random play. No diagnostic state is used
to initialize that run.

The current-minibatch pre-update stopping check is not a hard full-rollout
constraint and cannot undo an already accepted overshoot. The follow-up retains
this limitation explicitly. The production implementation must stop gradient
work after the guard fires; these instrumented probes still calculate rejected
proposals for inspection. Report actual accepted updates separately from the
fixed environment-decision budget.

## Faster reconstruction, validated against the ordinary updater

For the known reward-free prefix only, verify zero rewards/values, zero entropy
coefficient, finite action log probabilities and exactly zero Adam moments.
Advancing the sole optimizer clock, optimizer-step count and four epoch RNG
splits is then equivalent to 64 zero-gradient updates. The first such advance is
checked against the ordinary updater's entire resulting state with zero error.
Every reconstructed rollout still matches recorded rewards/actions/metrics;
the rewarded update and all original-branch results exactly match the earlier
ordinary reconstruction. This shortcut is diagnostic-only and is not added to
the PPO trainer.

This follow-up re-executes another 28,672 known-prefix decisions, not new
independent experience. It advances 832 virtual zero-gradient optimizer steps,
uses 64 ordinary updates for the equivalence proof, 64 rewarded oracle updates,
and 192 counterfactual gradient calculations. Count these separately from the
first diagnostic and from the unchanged online pilot. The raw human archive is
not read; the starting artifact is the saved fresh-PPO initial checkpoint.

[Evidence](analysis.json) · [Original diagnostic](../curriculum_first_reward_diagnostic/RESULT.md) ·
[Completed PPO pilot](../human_bc_100epoch_ppo_64k/RESULT.md) ·
[Implementation log](../../../IMPLEMENTATION_LOG.md).
