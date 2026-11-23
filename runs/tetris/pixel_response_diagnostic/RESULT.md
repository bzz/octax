# Pixel response and greedy inference

Neither trained policy establishes a scoring gain with greedy inference. Pixel response remains present; a completely inactive network is not the explanation on this sample.

## What was checked

The same 2,048 observed four-frame pixel stacks were supplied to the exact untrained seed-1 network and the two completed 512K PPO networks. They contain 1164 unique stacks. Collection repeats the first 2,048 decisions of the preceding long-trace gradient diagnostic, with zero rewards; these are additional executed diagnostic interactions, not independent new experience. No network is trained on this data.

Instrumented action probabilities and values match the actual PPO network for all five actions on the checked minibatch. All convolutional channels activate somewhere in the sample. The long-trace actor has more hidden units that never activate on this sample, but it still changes its probabilities with the observed pixels. These measurements do not show complete representation collapse, establish useful features or identify why learning failed.

## Same-bank inference comparison

| Network | Mean KL to its mean action distribution | Actor units never active on sample | Stochastic scoring | Greedy scoring |
| --- | ---: | ---: | ---: | ---: |
| untrained | 0.000216 | 28/256 | 2/128 | 0/128 |
| long trace | 0.000264 | 171/256 | 2/128 | 1/128 |
| credit control | 0.006244 | 110/256 | 1/128 | 0/128 |

KL is in nats and measures state-dependent action variation on these particular observations. It is not a game-skill metric. Greedy inference selects the largest logit (ties choose the lowest action index); stochastic inference samples the learned probabilities. The greedy adapter uses the existing environment evaluator, and its argmax matches the actual policy log probabilities. The saved stochastic monitoring evaluation reproduces exactly before changing action selection.

All final comparisons reuse seeds 50000–50127. The shared random control scores 3/128. Paired comparisons in `comparison.json` are exploratory, unadjusted tests of scoring probability. The same counts need not represent the same scoring games. Current evidence does not justify using the unused confirmation bank or claiming a policy has learned Tetris.

The animation is the **long-trace final checkpoint with greedy inference**, preselected illustrative seed 30016. It is newly captured from actual gameplay; it does not represent an aggregate evaluation score or a newly trained checkpoint.

## Reproduction

From the repository root with the saved run artifacts installed:

```sh
JAX_PLATFORMS=cpu .venv/bin/python diagnose_pixel_response.py
JAX_PLATFORMS=cpu .venv/bin/python evaluate_greedy_diagnostic.py
JAX_PLATFORMS=cpu .venv/bin/python summarize_pixel_response.py
```

Saved checkpoints remain byte-identical. Observation bytes, model hashes and the exact diagnostic sources accompany the report. [Completed long-trace trial](../v2_long_trace_512k_seed1/RESULT.md) · [Implementation log](../../../IMPLEMENTATION_LOG.md).
