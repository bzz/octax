# Replay retention investigation — shelved side experiment

At the saved 65,536-step replay checkpoint, 1,440 of the 50,000 replay transitions had a positive discounted return. At 131,072 steps, that number was **zero**, despite the cumulative counter still showing two completed scoring episodes. The read-only, hash-verified audit is recorded in `runs/tetris/learning_512k_seed1/replay_buffer_audit.jsonl`.

The FIFO implementation behaved as configured: new completed experience replaced old experience. It therefore did not retain the original successes throughout the longer run. This is an experiment-design limitation, not an upstream Octax bug. It does not by itself prove that eviction caused a scoring failure: the policy could already have learned from those transitions, and the full comparison is still running. The 131K development evaluation scored 1/8; that tiny repeated sample is not a learning claim.

A zero-return replay transition can still trigger an update when its current value estimate is negative. Thus nonzero auxiliary-update counts are not evidence that the buffer still contains scoring experience. The final report distinguishes cumulative scoring episodes, positive-return transitions currently stored, and sampling mass assigned to those transitions.

## Prepared storage ablation

`ProtectedEpisodeReplay` in `self_imitation_retention.py` changes only replacement: a new nonpositive-return transition cannot evict an existing positive-return transition. New positive-return transitions can replace old entries, including other positive entries, so capacity stays bounded even when every slot has a positive return. When a full buffer contains only positive targets, incoming nonpositive targets are discarded.

Array indices, priority calculation, sampling, replay loss, auxiliary-update budget, pixel observations, original rewards and PPO remain the same. No board features, demonstrations, shaped rewards or additional model are introduced. This variant may overfit old successes; preservation alone is not evidence of useful skill.

The module supports an explicit fork from a FIFO checkpoint that preserves all replay arrays, indices, pending episodes and RNG. Subsequent checkpoints carry a retention-mode marker and reject an implicit mode change. Four focused tests pass: survival under a flood of nonpositive targets, bounded replacement when full of positive targets, exact FIFO/sample equivalence before the first protected eviction, and explicit-fork/checkpoint continuation with unchanged parent bytes.

## Historical activation requirements (not scheduled)

The user clarified that the main investigation should remain online and on-policy PPO. This off-policy auxiliary variant is therefore **shelved** and is not connected to either trainer. The following describes the previously prepared protocol, not a scheduled run: start the ablation from the exact saved 64K replay parent and retain the same 524,288-decision target. Record a distinct method-fork checkpoint/configuration and original parent hashes; do not edit the published parent or call this an unchanged resume. Verify model/optimizer/emulator/collection-RNG identity and replay/sample identity at the fork, then verify trainer integration before launching.

The common trainer requires a small replay-factory hook after the frozen run finishes. Fresh-game confirmation and independent training-seed replication remain necessary before claiming a dependable learning setup. The original scoring objective has not been replaced by a retention test.
