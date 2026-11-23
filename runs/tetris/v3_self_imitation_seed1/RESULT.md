# Pixel-only score replay · fresh seed 1 · 64K

Replay does not beat all scoring controls in this exploratory sample; useful learning remains unestablished.

## Matched experiment

The same longer-credit CNN/PPO configuration receives completed-episode self-imitation updates after its first scoring episode completes. Four monochrome frames, keypress actions, corrected frame timing and original score reward are unchanged. No board features, demonstrations or shaping are used. Seed 1 was selected because it previously discovered feedback; this is a targeted mechanism test. Both runs use 65,536 environment decisions and 2,048 PPO optimizer updates. Replay adds 170 applied auxiliary updates and 5.4 seconds of replay processing, so compute is not matched. See [method](../../../SELF_IMITATION.md).

## Reward discovery and retention

Replay encountered 2 positive reward events, versus 2 in the control. It completed 40 games, including 2 scoring games. First auxiliary update rollout: 32768. Final entropy: 1.417150. Successful completed trajectories are retained as packed pixels/actions/rewards/returns; a reward event and a completed scoring game are different units.

Replay waits for episode completion, so it can start substantially after a reward is first seen. The exported episode audit in `analysis.json` records the first reward index and this delay for each success. At this small budget, the number of active replay rollouts matters when interpreting the result; the trial does not test unlimited reuse of those successes.

All non-timing PPO diagnostics and action/reward counts matched the control for 16 rollouts through the first activation rollout. This extends the separately verified 2,048-step complete-state equality check; diagnostic equality alone is not proof of complete-state identity at every later step. Replay/model checkpoint hashes and roundtrips passed, as did independent discounted-return reconstruction of every exported scoring episode.

## Original game scores

| Policy | Scoring games | Mean score | Mean decisions per game |
| --- | ---: | ---: | ---: |
| replay | 0/128 | 0.00000 | 1299.1 |
| longer credit | 1/128 | 0.00781 | 1400.7 |
| untrained | 2/128 | 0.01562 | 1312.2 |
| random | 3/128 | 0.02344 | 1273.9 |

The 128 evaluation seeds, 50000–50127, have been inspected in earlier rounds. Random is the same shared control, not a new independent sample. Paired comparisons against the original longer-credit policy, its exact untrained network and random are in `analysis.json`; they are exploratory. One selected training seed and these reused games cannot establish generalization. Seeds 60000 onward remain reserved for a frozen candidate's confirmation.

The result card uses the preselected gameplay seed 30016, not a selected scoring game. Extra optimizer activity, memorized experience and changed weights alone do not meet the goal: a saved pixel policy must demonstrate scoring improvement against random.
