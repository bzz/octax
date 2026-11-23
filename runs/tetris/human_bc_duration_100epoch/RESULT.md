# Two-seed offline cloning duration comparison

**The exploratory offline scoring advantage does not replicate across both initializations; useful learning remains unestablished.**

| Policy | Training seed 1 | Training seed 2 |
| --- | ---: | ---: |
| 20-epoch clone | 0/128 | 1/128 |
| 100-epoch clone | 7/128 | 1/128 |
| Scratch longer-credit PPO, 64K | 1/128 | 1/128 |
| Random (shared) | 3/128 | 3/128 |

Seed 2 was triggered by seed 1 passing the declared screen, so this is a
sequential exploratory investigation. Both initializations share one human
prefix and the same 128 evaluation game seeds. Do not pool the columns as 256
independent games or treat the random control as two independent baselines.
The screen is not a significance test. No unused confirmation bank was consumed.

| Training seed | Training CE, 20 → 100 epochs | Training rotation recall, 20 → 100 |
| --- | ---: | ---: |
| 1 | 0.4201 → 0.1703 | 0.0% → 83.0% |
| 2 | 0.4222 → 0.1713 | 0.0% → 72.3% |

In both seeds, the fresh 100-epoch run exactly reproduces its archived complete
20-epoch state, including optimizer/RNG/counters, before continuing. Inputs,
data order, architecture, optimizer and timing are fixed. All action-fit metrics
use training examples. Final-only weights are evaluated; there is no selection
of the best checkpoint by game score. Original-score controls and source audits
pass. Full evidence: [seed 1](../human_bc_100epoch_seed1/RESULT.md) and
[seed 2](../human_bc_100epoch_seed2/RESULT.md).

Across both seeds, the 20-epoch reference plus fresh 100-epoch run costs 4,320
offline updates and 537,600 example presentations, including seed 1's previously
completed 20-epoch run. There are only 2,240 unique human decisions from one
incomplete, score-11 prefix. Validation replays known actions; it is not new
experience. **No PPO training occurs in this comparison.** The raw archive stays
local; published weights/configurations support evaluation and fresh PPO, while
offline reproduction requires the separately supplied checksum-matched archive.

Do not interpret seed-1 selection as a replicated initialization. Inspect the seed difference and action coverage before a long PPO commitment; further experiments must remain separately declared and preserve the failed replication result.

The card retains the preselected seed-1 policy/game-30016 illustration, score
zero, and does not choose a successful game. [Card](result_card.html) ·
[Machine-readable evidence](analysis.json) ·
[Protocol](../../../CLONING_DURATION_EXPERIMENT.md) ·
[Implementation log](../../../IMPLEMENTATION_LOG.md).
