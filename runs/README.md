# Recorded Tetris experiments

All training policies observe only four monochrome pixel frames and receive the ROM score difference. Checkpoints contain model, optimizer, RNG and emulator state. The four checkpoints are about 26 MB each and are stored directly in Git. New runs remain ignored; this is an explicitly published snapshot.

| Directory | Timing / purpose | Recorded budget |
| --- | --- | --- |
| [sanity](tetris/sanity) | Legacy random/no-op controls and reward verification | Eight development games per policy |
| [ppo_seed0_smoke](tetris/ppo_seed0_smoke) | v1 seed-0 smoke, legacy timing | 65,536 training steps; 16 final games per policy |
| [continued_seed0](tetris/continued_seed0) | Continuation of the same v1 agent | 499,712 cumulative steps; 64 final games per policy |
| [reward_diagnosis](tetris/reward_diagnosis) | Timer/action and PPO-gradient probes, before-fix source | 64 games per timing/action condition |
| [sanity_frame_v2](tetris/sanity_frame_v2) | Corrected-timing random/no-op controls | Eight development games per policy |
| [v2_validation_8192](tetris/v2_validation_8192) | Fresh stability check | 8,192 steps |
| [v2_baseline_seed0](tetris/v2_baseline_seed0/RESULT.md) | Fresh corrected-timing baseline | 65,536 steps; 128 final games per policy |

The latest PPO agent saw zero training rewards and scored in 1/128 final evaluation games, versus 3/128 for random. The equal-budget random rollout also saw zero rewards. This does not demonstrate learned scoring skill. The full [result](tetris/v2_baseline_seed0/RESULT.md) includes uncertainty and comparison limits.

Use each run's configuration and timer semantics; do not combine v1 and v2 as one learning curve. Held-out results contain per-game return, length, actions, termination and truncation arrays. Development-evaluation curves are separate from the final comparisons. A displayed gameplay seed is illustrative only.

See [EXPERIMENT.md](../EXPERIMENT.md) for commands and [FORK_CHANGELOG.md](../FORK_CHANGELOG.md) for source-provenance limits. Historical status JSON describes completed local jobs, not current processes. Local checkout prefixes in text files have been made relative; checkpoint bytes, numeric results and source snapshots are unchanged.

[ARTIFACT_SHA256.json](ARTIFACT_SHA256.json) records sizes and SHA-256 hashes of all published artifacts in this directory except the manifest itself.
