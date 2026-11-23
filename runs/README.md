# Recorded Tetris experiments

All actors observe only four monochrome pixel frames. Score-only PPO uses the ROM score difference; explicitly labeled auxiliary-method experiments document their additional training objectives. Full checkpoints contain model, optimizer, RNG and emulator state; inference-only policy exports are labeled separately. Checkpoints are stored directly in Git. New runs remain ignored; this is an explicitly published snapshot.

| Directory | Timing / purpose | Recorded budget |
| --- | --- | --- |
| [sanity](tetris/sanity) | Legacy random/no-op controls and reward verification | Eight development games per policy |
| [ppo_seed0_smoke](tetris/ppo_seed0_smoke) | v1 seed-0 smoke, legacy timing | 65,536 training steps; 16 final games per policy |
| [continued_seed0](tetris/continued_seed0) | Continuation of the same v1 agent | 499,712 cumulative steps; 64 final games per policy |
| [reward_diagnosis](tetris/reward_diagnosis) | Timer/action and PPO-gradient probes, before-fix source | 64 games per timing/action condition |
| [sanity_frame_v2](tetris/sanity_frame_v2) | Corrected-timing random/no-op controls | Eight development games per policy |
| [v2_validation_8192](tetris/v2_validation_8192) | Fresh stability check | 8,192 steps |
| [v2_baseline_seed0](tetris/v2_baseline_seed0/RESULT.md) | Fresh corrected-timing baseline | 65,536 steps; 128 final games per policy |
| [v2_credit_horizon_seed0](tetris/v2_credit_horizon_seed0/RESULT.md) | Fresh longer-credit ablation, same budget | 65,536 steps; same 128 evaluation seeds |
| [v2_replication_64k](tetris/v2_replication_64k/RESULT.md) | Baseline/longer-credit seeds 0, 1 and 2; initialization controls | Four new 65,536-step runs; 128 shared game seeds |

The three-seed replication found baseline reward events [0, 0, 0] and longer-credit events [1, 2, 0]. Final scoring games were baseline [1, 1, 1] and longer credit [2, 1, 1] out of 128 per policy; the shared random control scored in 3/128. See the [latest result](tetris/v2_replication_64k/RESULT.md) for initialization comparisons, uncertainty and the learning decision.

Use each run's configuration and timer semantics; do not combine v1 and v2 as one learning curve. Held-out results contain per-game return, length, actions, termination and truncation arrays. Development-evaluation curves are separate from the final comparisons. A displayed gameplay seed is illustrative only.

See [EXPERIMENT.md](../EXPERIMENT.md) for commands and [FORK_CHANGELOG.md](../FORK_CHANGELOG.md) for source-provenance limits. Historical status JSON describes completed local jobs, not current processes. Local checkout prefixes in text files have been made relative; checkpoint bytes, numeric results and source snapshots are unchanged.

[ARTIFACT_SHA256.json](ARTIFACT_SHA256.json) records sizes and SHA-256 hashes of all published artifacts in this directory except the manifest itself.

[Self-imitation seed 1](tetris/v3_self_imitation_seed1/RESULT.md): fresh 65,536-step score-replay test, full model/replay checkpoints, exported successful episodes, validation evidence and actual-gameplay card.

[Longer-credit PPO at 512K](tetris/v2_credit_horizon_512k_seed1/RESULT.md): completed seed-1 continuation, 1/128 games scoring. The completed matched comparison is linked below.

[Matched 512K continuation](tetris/learning_512k_seed1/RESULT.md): corrected longer-credit PPO versus self-imitation replay, each resumed from 64K to 524,288 decisions. Full checkpoints, parent hashes, cumulative metrics and actual-gameplay card are included.

[Pixel-novelty CPU smoke](tetris/pixel_novelty_smoke/SMOKE_REPORT.md): strictly current-rollout PPO, matched coefficient-zero/enabled seed-101 runs at 8,192 decisions each. Correctness gates passed; neither found original game rewards. Includes full and inference-only checkpoints, frozen source snapshots, runtime measurements and an actual-policy animation. This is smoke evidence, not a learning result.

[Two-seed pixel-novelty pilot at 64K](tetris/pixel_novelty_64k/RESULT.md): fresh on-policy seeds 1 and 2, each 65,536 decisions. Both scored 0/128 versus random 3/128 and found no original training rewards. Full/inference checkpoints, frozen sources, separate smoke accounting, source/control audits, curves and actual-policy cards are included. This version failed the exploratory screen; the reserved confirmation bank remains unused.


[Independent cloning / PPO handoff](tetris/human_bc_seed1/RESULT.md): human-prefix provenance and validation, offline checkpoint and weights-only artifact, actor-only original-score evaluation, two separate fresh PPO smokes, read-only confusion/retention audit and actual-policy card. The raw human archive is excluded; offline reproduction requires a separately supplied copy, while published weights suffice for evaluation and PPO. The clone scores 0/128 versus random 3/128; no full 64K curriculum run or useful learning is claimed.
