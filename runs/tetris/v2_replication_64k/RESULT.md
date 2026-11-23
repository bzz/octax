# Three-seed Tetris replication · 64K per run

Neither configuration passes the exploratory learning gate at 64K steps; useful scoring improvement is not established.

## Fixed protocol

Both configurations were trained from fresh seeds 0, 1 and 2 for exactly 65,536 environment decisions each. Seeds 1 and 2 are the four new runs; seed 0 results are reused. The model remains a four-frame monochrome CNN with original game-score rewards and corrected per-frame timers. No board heuristics, shaping, recurrence or transfer were introduced.

Baseline: rollout 32, gamma 0.99, lambda 0.95, four minibatches. Longer credit: rollout 128, gamma 0.995, lambda 0.98, sixteen minibatches. Both use 16 environments, four epochs, minibatch size 128 and 2,048 optimizer steps per run. Training sources/ROM are snapshotted; the sequential runner rejects source drift.

## Reward discovery and scores by training seed

The first-event column is the **end of the rollout containing the first reward**, not its exact action index. Initial-network and final-policy scoring use the same 128 evaluation seeds, 50000–50127.

| Configuration | Training seed | Training reward events | First event rollout | Final scored | Untrained scored | Final mean score | Final entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v2 baseline | 0 | 0 | — | 1/128 | 5/128 | 0.00781 | 1.6094 |
| Longer credit | 0 | 1 | 34816 | 2/128 | 5/128 | 0.01562 | 1.5785 |
| v2 baseline | 1 | 0 | — | 1/128 | 2/128 | 0.00781 | 1.6094 |
| Longer credit | 1 | 2 | 16384 | 1/128 | 2/128 | 0.00781 | 1.5329 |
| v2 baseline | 2 | 0 | — | 1/128 | 0/128 | 0.00781 | 1.6094 |
| Longer credit | 2 | 0 | — | 1/128 | 0/128 | 0.00781 | 1.6094 |

Uniform random scored in **3/128** games (mean 0.02344); no-op scored in 0/128. The random control is identical across all evaluations and must not be multiplied into a larger sample. Equal-budget random reward-event counts for training seeds 0/1/2: [0, 2, 0].

| Configuration | Training seeds finding rewards | Mean score across seeds | Between-seed SD | Mean score gain over untrained |
| --- | ---: | ---: | ---: | ---: |
| v2 baseline | 0/3 | 0.00781 | 0.00000 | -0.01042 |
| Longer credit | 2/3 | 0.01042 | 0.00451 | -0.00781 |

The three training seeds, rather than individual game episodes from a shared policy, are the replication units. Episode-paired tests are retained per seed in `analysis.json` as exploratory descriptions; they are not treated as proof across a population of trained networks. With only three training seeds and previously inspected game seeds, a small difference is weak evidence.

## Learning decision

The screening rule requires rewards in at least two training seeds, a condition's average score above random and above its own untrained networks, and positive score gains in at least two seeds. This rule chooses whether a frozen configuration warrants fresh-seed evaluation; it is not proof of useful learning. Candidates: none.

The project goal remains a saved pixel-based network with a convincing scoring advantage over random, not merely changed weights, nonzero loss or a completed run. If a candidate passes screening, evaluate all three trained policies of that configuration on a new fixed 512-game bank starting at seed 60000. If neither passes, investigate the shortage of useful experience before extending the same setup indiscriminately.

## Validation and artifacts

All six configurations match their arm's settings except training seed and output path. Initial development arrays match between arms within each seed. Reconstructed initial networks reproduce their stored development evaluations exactly. All repeated random/no-op per-game arrays match; checkpoint roundtrips and finite diagnostics passed. Full per-run metrics, final returns, action distributions and termination/truncation flags remain in the run directories.

[Analysis](analysis.json) · [Result card](result_card.html) · [Implementation log](../../../IMPLEMENTATION_LOG.md) · [Protocol](../../../EXPERIMENT.md) · [Seed-0 ablation](../v2_credit_horizon_seed0/RESULT.md)

The card shows the **preselected longer-credit training seed 1**, gameplay seed 30016, rather than a seed selected for its final score. It is illustrative and does not replace the aggregate table.
