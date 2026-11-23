# Offline cloning duration: 20 versus 100 epochs

**Longer offline fitting passes the exploratory scoring screen; seed replication and fresh on-policy RL remain necessary.**

The [predeclared protocol](../../../CLONING_DURATION_EXPERIMENT.md) changes only
offline duration. Both runs use seed 1, the same human prefix, CNN, four pixel
frames, action space, optimizer and shuffled minibatches. The 100-epoch run
starts fresh; its entire state at epoch 20 exactly matches the prior archived
cloning checkpoint, including parameters, Adam moments, RNG and counters.
Initial network hashes, first-20-epoch metrics and source snapshots also match
their declared checks. No PPO updates occur in this experiment.

## Original-score evaluation

| Policy/control | Scoring games |
| --- | ---: |
| clone_100_epochs | 7/128 |
| clone_20_epochs | 0/128 |
| scratch_ppo_64k | 1/128 |
| untrained_seed1 | 2/128 |
| random | 3/128 |
| noop | 0/128 |

All policies use the same 128 exploratory game seeds 50000–50127 and corrected
timing. Random and no-op episode arrays match the earlier controls exactly.
The weights-only evaluator exactly reproduces all eight monitoring episodes.
No checkpoint was selected by evaluation score; these are final-epoch weights.
The candidate screen is not a significance test. The fresh confirmation bank
60000–60511 remains unused. There is only one training seed and one human prefix.

## Offline fit

| Diagnostic | 20 epochs | 100 epochs |
| --- | ---: | ---: |
| Cross-entropy | 0.4201 | 0.1703 |
| Entropy (nats) | 0.5341 | 0.1865 |
| Argmax accuracy | 0.8607 | 0.9223 |

| True action | 20-epoch recall | 100-epoch recall |
| --- | ---: | ---: |
| Rotate | 0.00% | 82.98% |
| Left | 33.62% | 71.55% |
| Right | 70.16% | 78.01% |
| Down | 69.01% | 77.78% |
| No-op | 95.45% | 96.91% |

All fit and recall values are in-sample. The prefix has 76.56% no-op labels and
only 47 rotation labels. Conflicting actions on identical inputs impose an
empirical training CE lower bound of 0.12386
and argmax accuracy upper bound of 92.95%;
these do not predict gameplay performance or prove this CNN can reach them.
The curve shows loss averaged during each epoch; black points are the separately
measured final training losses, which are different quantities.

![Training fit and per-action recall](fit_comparison.png)

## Budget and next action

The new run uses 1,800 offline Adam updates and 224,000 presentations of the
same 2,240 demonstration decisions. It repeats the first 20 epochs for a strict
audit; it is not an 80-epoch continuation. Its recorded training time is
390.10 seconds, excluding validation and gameplay
evaluation. It executes 2,241 diagnostic replay transitions for dataset checking.
These are repeated known actions, not new experience. Evaluation episode lengths
are retained separately in the monitoring/comparison files. The offline report
itself collects zero environment decisions and performs zero updates.

Replicate the fixed offline setup with seed 2 before declaring a learning setup. Any PPO follow-up must be separately declared and preserve the independent weights-only handoff.

The human prefix is incomplete (`complete=false`, original score 11); it is not
held-out data or a completed expert episode. Its raw archive remains local and
excluded from GitHub. Reproducing offline training/fit requires the separately
supplied checksum-matched recording. Published weights, sources and config permit
standalone gameplay evaluation and an independent fresh PPO process.

The actual-policy animation uses preselected seed 30016, without selecting a
successful game. See [the card](result_card.html), [full evidence](analysis.json),
[20-epoch result](../human_bc_seed1/RESULT.md), and
[implementation log](../../../IMPLEMENTATION_LOG.md).
