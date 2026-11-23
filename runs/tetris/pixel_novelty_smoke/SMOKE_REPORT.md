# Pixel-novelty CPU smoke comparison

Both matched seed-101 runs completed 8,192 decisions and 256 PPO updates. All states/metrics were finite, the fixed target stayed unchanged and full checkpoints round-tripped exactly. The disabled arm had zero auxiliary optimizer updates; the enabled arm had 256 predictor and intrinsic-head updates. Initial policy hashes and evaluation arrays matched exactly. The only resolved differences were output path and novelty coefficient 0 versus .01. No replay or demonstrations were used.

| Diagnostic | Disabled | Enabled |
| --- | ---: | ---: |
| Original training rewards | 0 | 0 |
| All-rollout decisions/s | 154.43 | 73.17 |
| Warmed decisions/s (three rollouts) | 157.29 | 73.88 |
| Monitoring evaluation decisions | 4820 | 4834 |

These are separately counted smoke diagnostics, not pilot-training experience or evidence of improvement. The two-game initial/final evaluation checks have no useful power to assess learning. The disabled control still executes the extended collector's auxiliary-value inference, so this is not a benchmark of the untouched baseline runner. Throughput covers collection and updates, excluding checkpoint I/O and evaluation. Zero normalization warm-up decisions were used. The illustration adds 1020 evaluation decisions on preselected seed 30016; it is not used for updates or policy selection.

The next frozen experiment is fresh seeds 1 and 2 at 65,536 decisions each, compared on original game score against the saved score-only PPO, exact untrained networks and random play. Coefficient .01 and other settings remain fixed. No scoring improvement is claimed by this smoke test.

[Actual smoke-policy card](result_card.html) · [Protocol](../../../PIXEL_EXPLORATION_PROPOSAL.md) · [Implementation log](../../../IMPLEMENTATION_LOG.md)
