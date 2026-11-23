# Implementation log

## 2026-11-23 — Scope and hardware

The requested deliverable is working Tetris training on the available Apple Silicon CPU, with results and gameplay visualization. GPU is intentionally unused. The user chose to finish the 65,536-step smoke test and prepare longer runs rather than spend several hours on three 500K-step seeds. Transfer is aspirational and remains a prepared future experiment.

The public Octax repository was cloned at commit `3aa53b516152e97f2ed91eae6e33b6ee9a97596b`. An isolated Python 3.11 environment was created in this checkout. No global Python packages were changed.

## Environment and dependency checks

- Confirmed CPU JAX execution and raw `(4,64,32)` observations with five actions.
- Found that `.[training]` lacks OpenCV/Pillow even though Octax imports its renderer unconditionally. Added headless OpenCV and Pillow to training dependencies.
- The initial dependency resolution installed Gymnax 1.1.0, whose six-value step API breaks the existing Octax/Rejax rollout. Pinned Gymnax 0.0.9, matching the original five-value API. Captured all installed versions in `requirements-macos.lock.txt`.
- The failed compatibility attempt was moved to the task's scratch directory and is excluded from successful experiment results.

## CNN corrections

The proposed layout bug was confirmed: Gymnax supplies NCHW and Flax expects NHWC. Added one shared encoder conversion used by `__call__`, `call_actor`, and `call_critic`.

An additional issue was found: scalar second/third kernel sizes produce one-dimensional Flax convolutions. Changed those kernel/stride arguments to explicit 2D tuples. The model remains a shared three-layer 32/64/64 CNN with separate 256-unit actor and critic heads. No Tetris geometry, hand-built features, shaping, recurrence or transformers were added.

## Training and checkpointing

Retained Octax's PPO loss, rollout collector and GAE implementation. The host runner invokes collection and updates separately to log their runtime and writes diagnostics each iteration. This replaces the original all-in-one JIT training entry point; it is not a new RL algorithm.

The original script saved `params=[]`. The new version saves learned parameters, optimizer moments/update count, RNG, emulator states, observations and cumulative environment step. It reloads each saved checkpoint and checks exact state and inference equality. Atomic writes and refusal to overwrite run directories protect completed results.

Explicitly configured a clip threshold of 0.5. Octax uses elementwise `optax.clip`, not global-norm clipping; that existing algorithm behavior is retained and documented rather than silently changed during the run.

Budgets must be divisible by the 512-step rollout size. The prepared nominal 500K pilot uses 499,712 steps, avoiding silent overshoot. Independent seeds run sequentially.

## Sanity evidence

Eight random and eight no-op games terminated naturally. Random mean return was 0.125 (one scoring episode); no-op mean return was 0. Control trajectories visibly confirm rotation, left/right movement and accelerated dropping.

Reset pixels and the first piece are identical because upstream caches startup. Full no-op episode lengths vary across keys, demonstrating later RNG variation. We preserved this behavior and explicitly label the 120-step probe's limited horizon.

Replayed random evaluation seed 10002: +1 reward at step 2330, V[10] increases to 1, and every reward equals the score difference. Saved surrounding frames as `positive_reward.png`.

## Validation and run scope

The 69 upstream tests passed. Added regression checks for spatial CNN kernel dimensions, all actor/critic input paths, terminal-aware GAE and exact optimizer/RNG checkpoint continuation. The live smoke run checks finite losses and weights and logs nonzero gradient norms.

Initial measured throughput was roughly 110–125 environment steps/s; PPO updates dominate collection time. This projects roughly 70–75 minutes per 500K seed, before evaluations. On that evidence, the user explicitly selected finishing the smoke test and preparing longer runs.

## Visualization direction

The user clarified that “MinAtar style” means the rendering tools MinAtar actually uses, rather than an inline visualization skill. The implementation will follow the upstream MinAtar renderer's plotting approach while preserving the Octax ROM's exact pixel observations. Training still consumes the unmodified monochrome frames.

## MinAtar rendering choice confirmed

Inspected upstream `minatar/environment.py`: `display_state()` uses Matplotlib `imshow`, `ListedColormap`/`BoundaryNorm` and Seaborn's cubehelix palette. Reused those libraries and primitives in `visualize_run.py`, keeping the exact 64×32 Octax display as a single foreground channel. No MinAtar game, 10×10 observation conversion or object segmentation was introduced.

The renderer captures complete deterministic-seed, stochastic-policy episodes and retains the terminal frame. Its side-by-side random/PPO GIF samples every eight actions at approximately 4× playback. The plotted learning diagnostics come directly from saved CSV/JSON, and raw captured trajectories are retained as NPZ files. This replaces the abandoned inline-visualization approach after the user's clarification.

Added `replay.py` for local Matplotlib animation of saved trajectories, using the same palette/rendering primitives. Export remains headless and does not require a desktop plotting backend. The interactive window is a prepared convenience; visual verification focuses on the generated GIF and PNG artifacts.

Final regression run: 72 tests passed (69 upstream + 3 new), with only dependency/emulator deprecation warnings. `git diff --check`, shell syntax and Python compilation checks passed.

## Completed smoke test and evaluation

Completed 65,536 CPU environment steps in 531.1 timed training seconds (123.4 steps/s). All 128 logged iterations have finite losses and nonzero gradients. Parameters changed by L2 16.460; checkpoint reload preserves full state and inference exactly. Training saw zero positive reward events.

Final development evaluation: zero score in eight games, including three time-limit truncations. Held-out evaluation: one scoring PPO episode out of 16 (mean 0.0625), versus zero out of 16 for random and no-op. This is insufficient evidence of learning superiority. Full-episode gameplay and learning curves were generated with Matplotlib/Seaborn; both the raw trajectories and results are retained. Longer training and transfer were not launched.

Added a source/ROM SHA-256 manifest and copied the actual training and test logs into the run folder. Visual review checked the exported curve layout and gameplay; removed an inherited chart grid from gameplay so it does not obscure the original pixels.

## 2026-11-23 — Longer CPU pilot launched

The user asked to start longer training. Started one continuation of seed 0 from the verified 65,536-step smoke checkpoint to 499,712 cumulative environment steps (434,176 additional steps). Kept the same CPU backend, model, 16 environments, rollout size, optimizer, PPO settings and evaluation seeds; restored optimizer and RNG/environment state rather than restarting.

The run writes to `runs/tetris/continued_seed0/`, evaluates and saves checkpoints at 65,536-step intervals and at the final target, and preserves the smoke results. A detached local runner (`run_pilot.py`) records its current phase or failure in `run_status.json`, with output in `job.log`. After training, it automatically runs the prepared 64-episode held-out comparison and exports the Matplotlib/Seaborn learning curves and gameplay. Three-seed training and transfer remain unlaunched. Estimated remaining training is about 59 minutes at measured smoke throughput, plus evaluation/rendering overhead.

Launch verification: the restored 65,536-step checkpoint reproduced the original evaluation exactly, and resumed PPO updates advanced past step 66,048 with finite loss and nonzero gradients. The job is running in the background.

## 2026-11-23 — Longer pilot completed

The resumed CPU run finished at 499,712 total environment steps. Additional training plus periodic evaluation took 3,253.9 seconds (54.2 minutes); observed collection/update throughput was 142.1 steps/s. The continuation encountered one positive reward event in 434,176 additional steps, with 141 completed episodes. All eight development evaluations (including the resumed starting checkpoint) returned zero mean score.

The final 64-episode held-out evaluation recorded one scoring PPO episode (mean score 0.015625) and zero scoring episodes for random and no-op. This one-event difference is insufficient evidence of improvement. PPO's mean episode length was 3,222 steps, with nine time-limit truncations; longer survival must not be interpreted as better scoring. Termination and truncation flags can both be true on the same final transition. The runner completed evaluation and MinAtar-style Matplotlib gameplay/curve export successfully. No further training was started.

## 2026-11-23 — Sparse-reward investigation and v2 changes

The user authorized investigation and changes before the next run, with a strict pixel-only policy and an implementation record. No new long training run was launched. Full evidence, settings, limitations and next commands are in `REWARD_DIAGNOSIS.md` and the revised `EXPERIMENT.md`.

### Finding: emulator timing was not sound

Reproduced a uint8 underflow: `jnp.maximum(jnp.uint8(0)-1,0)` returns 255. The Tetris ROM sets delay to zero for down input; the old step logic could replace that zero with a long artificial delay. Also, the old environment executed four simulated frames but decremented timers only once. Earlier control/reward sanity tests did not check this arithmetic or clock rate; their successful results were insufficient to rule it out.

Implemented per-frame, saturating 60 Hz timer updates without changing the raw frame shape, number of instructions per frame, actions or reward. Signed intermediate subtraction prevents underflow for both delay and sound timers. Frame timing explicitly requires 60 FPS. `legacy` and `step_saturating` modes support reproducibility and a controlled timing ablation. Before-change environment/PPO source snapshots are saved in the diagnosis run directory.

Matched-seed probes (64 games/condition, seeds 40000–40063): legacy random games averaged 2,288.5 decisions and spent 29.65% of steps at delay >16; saturating-only averaged 2,037.1; per-frame timing averaged 1,249.3 and never exceeded delay 16. Random scoring was 1/64, 1/64 and 2/64 respectively. These small score counts do not prove an improvement. No-op length fell from 2,805.9 to 777.2 decisions. Repeating each action four times under corrected timing produced 0/64 scoring games, so action repetition is not enabled. The reward itself remains the ROM line-score difference, and is reachable under corrected timing.

### Finding: normalization amplified tiny reward-free critic errors

Probed a fixed rollout from the old 499,712-step policy: reward sum 0, raw advantage std 0.00003431. Ordinary normalization yielded policy-gradient norm 0.37467, versus entropy-gradient norm 0.00830 (about 45×). An advantage std floor of 0.01 reduced that policy-gradient norm to 0.00129. This supports a noise-amplification mechanism; it does not quantify its entire historical contribution.

New v2 defaults initialize the final value-output weights/bias to zero, use an advantage std floor of 0.01, and replace coordinate-wise clipping with actual global-norm clipping at 0.5. The model architecture and parameter names stay the same. The floor is an experimentally motivated stabilization choice, not a universal PPO requirement. New metrics record raw/normalized advantage scale, approximate KL, clip fraction and raw/clipped gradient norm. Nonzero gradients alone are not evidence of useful reward learning.

### Validation and compatibility

All 85 tests pass, including eight timer/legacy-routing cases and five sparse-PPO/resume cases. A fresh 8,192-step v2 validation completed and reloaded its checkpoint exactly. No positive training rewards occurred. Final entropy was 1.609437 (near log(5)); reward-free value loss/advantages stayed zero, and the final gradient norm was about 8.6e-7. Final development evaluation scored 0 in eight games, all naturally terminated (no truncations), mean length 1,160.5. This verifies stability and control flow, not successful learning. The short run observed 86.6 training steps/s while diagnostics shared the CPU; do not use it as an isolated performance comparison.

Checkpoint evaluation now reconstructs timer settings from saved config. Configs predating the timer fix explicitly select legacy timing. The prior smoke checkpoint reproduced every stored evaluation array exactly (scores, lengths, action counts, terminated/truncated flags). New `experiment_version=2` configs cannot silently resume old optimizer/environment state. Plotting chooses a matching-timing baseline and rejects mismatches. Historical outputs and checkpoints are preserved. The old continuation launcher is explicitly pinned to legacy settings; new pilot paths use `v2_pilot`.

### Next experiment, not launched

Start a fresh corrected 65,536-step baseline and inspect reward events, game scores, policy entropy, advantage scale and KL. Keep original gamma=0.99/lambda=0.95/32-step rollout initially to isolate the implemented correctness/stability changes. A separate prepared `credit_horizon` profile uses 128-step rollouts, 16 minibatches (same minibatch size), gamma=0.995 and lambda=0.98, at the same step budget. It tests longer credit assignment; no effectiveness claim is made.

No board features, geometry-derived rewards, piece-placement rules, reward shaping, recurrence, transformers or intrinsic reward were added. If corrected baseline and credit-horizon arms still rarely discover scores, a separately labeled pixel-only exploration objective would be the next controlled experiment. Transfer remains deferred.

Final validation also checks that a zero-initialized value head learns from a synthetic +1 target, that score-scale advantages remain normalized, and that the update respects the 0.5 gradient-norm bound. The complete suite passed 85 tests. Updated v2 curves/gameplay were exported with corrected-timing baselines; entropy uses its natural 0-to-log(5) scale to avoid visually magnifying tiny changes.

## 2026-11-23 — Fresh 64K v2 baseline started

The user authorized the full fresh v2 baseline and a comparison against random under corrected timing. Started seed 0 from newly initialized weights, optimizer, RNG and emulator state, targeting 65,536 CPU environment decisions. No checkpoint or encoder weights are reused. The default v2 configuration is unchanged: four monochrome frames, corrected per-frame timers, zero value-output initialization, advantage std floor 0.01, global-norm clip 0.5, 16 environments, 32-step rollout, four minibatches/epochs, gamma 0.99 and lambda 0.95. Original line-clear score remains the only reward.

The final saved policy, uniform-random and no-op controls will each be evaluated on 128 matched, previously unused environment seeds 50000–50127, using corrected timing. This expands the final comparison sample because score events are rare; it does not supply additional training data or change hyperparameters. Development evaluation remains eight games on the existing seeds. The run writes to `runs/tetris/v2_baseline_seed0/`, with an exact source snapshot, source hashes, phase/status JSON and job log. The runner also exports learning curves and MinAtar-style gameplay after evaluation. Results will be appended after completion; no horizon ablation or further training is scheduled.

## 2026-11-23 — Fresh 64K v2 baseline completed

Completed 65,536 CPU environment steps from fresh seed 0 at 114.1 observed steps/s. PPO encountered 0 positive reward events (total reward 0) in 44 completed episodes; the equal-budget uniform-random control encountered 0 in 42 episodes. First PPO reward: not observed. The control matches initial reset state and experience budget, not exact action RNG paths/episode boundaries.

Final evaluation used 128 matched, previously unused seeds 50000–50127 under corrected frame timing. PPO scored in 1/128 games (mean 0.00781); random scored in 3/128 (mean 0.02344); no-op scored in 0/128. Exact paired scoring-rate test p=0.625. This evaluation does not establish a scoring advantage over random play. This is one training seed and does not establish reproducibility across trained agents.

All training checks and exact checkpoint reload passed. Final policy entropy was 1.609438. The runner exported matched-timing learning curves and MinAtar-style gameplay. The run folder contains raw episode arrays, reward-event rollout indices, scoring-rate intervals, source snapshots and `RESULT.md`. No further training or horizon ablation was launched.

## 2026-11-23 — Reproducible fork publication

Prepared publication to `bzz/octax` with five narrow, tested upstream-candidate commits: CNN layout/spatial kernels, saturating timer arithmetic, per-frame timer cadence, actual learned weights/full-state checkpoints, and global gradient-norm clipping. Each has its own rationale and regression tests. `upstream-fixes` stops at that series; the pixel-only experiment workflow and recorded results follow separately on `main`. `FORK_CHANGELOG.md` is the central index and documents dependencies and the 60-FPS cadence limitation.

Validated the final isolated upstream-fix tree with 78 passing tests and the experiment tree with 87. Packaging moved identical checkpoint helpers into a reusable module; experimental zero-value initialization, advantage flooring and historical replay settings remain in the experiment layer. Added reproduction instructions using fresh output paths and separated the future source commit/dirty flag from the fixed upstream base.

Published evidence includes original configurations, metrics, evaluations, four full-state checkpoints, MinAtar-style plots/gameplay, and source manifests/snapshots where available. Text-only checkout-path normalization preserves numeric measurements and model bytes. The final v2 baseline has an exact source snapshot; earlier runs' source-hash records do not reconstruct every historical uncommitted file. Artifact hashes support download integrity checks. No further training was launched during packaging.

Publication recheck: the packaged evaluator reloaded the final v2 checkpoint and reproduced all per-episode arrays from its eight-game development evaluation exactly (including returns, lengths, action counts, termination and truncation). Source snapshot and artifact integrity hashes were verified.

## 2026-11-23 — Longer-credit 64K ablation started

The user authorized a fresh seed-0 longer-credit run at the same 65,536 environment-decision budget as the v2 baseline. The policy remains pixel-only, with the same corrected frame timing and original score reward. The prepared configuration changes rollout length 32→128, gamma 0.99→0.995 and GAE lambda 0.95→0.98; minibatches increase 4→16 to retain 128 samples per minibatch. Both arms use four epochs and 2,048 optimizer updates in total, but the ablation collects 32 larger rollouts rather than 128 smaller ones. This is a bundled credit-assignment ablation, not an isolated test of one hyperparameter.

Training starts from fresh weights/optimizer/RNG, with no checkpoint transfer. Final evaluation will reuse the baseline's 128 environment seeds 50000–50127 for paired comparisons with the v2 baseline and uniform random play. These seeds have already been inspected: this round is an exploratory matched comparison, not fresh confirmatory validation. The earlier equal-budget random rollout is reused as a recorded control, not counted as a new independent sample. Exact source/ROM snapshots and hashes are recorded before launch.

Each investigation round should end with a compact in-chat result card: actual policy gameplay rendered with the existing MinAtar-style Matplotlib tools, concise evidence, and links to the implementation changes and full result.

## 2026-11-23 — Longer-credit 64K ablation completed

Completed the fresh seed-0 65,536-step run with 128-step rollouts, gamma 0.995 and lambda 0.98. Training encountered 1 positive reward event (reward sum 1) in 42 completed games. The baseline encountered 0 events; the reused equal-budget random rollout encountered 0.

On the same previously inspected 128 seeds, longer credit scored in 2/128 games, baseline in 1/128 and random in 3/128. Unadjusted exact paired scoring tests: p=1 versus baseline, p=1 versus random. No scoring improvement is established by this one-seed exploratory comparison. Training reward events show whether useful targets became available, but sparse evaluation scores and a single trained seed cannot establish robust policy improvement. The ablation encountered feedback in a run where the baseline did not. This is evidence that the corrected pipeline can update from a real game score, not proof that longer credit caused the discovery. Before the first reward, logged advantage variation and value loss were zero; changing gamma/lambda cannot propagate a score that has not been encountered. Different rollout/update scheduling also changes subsequent random draws and trajectories. Independent training seeds are needed to distinguish a repeatable benefit from a lucky discovery.

Initial policy evaluation and repeated random/no-op controls matched their baseline arrays exactly. Finite-state checks and exact checkpoint roundtrip passed; final entropy 1.578472, throughput 139.5 decisions/s. This is a single-seed, bundled, exploratory comparison; it does not isolate rollout/gamma/lambda effects or establish performance across seeds. No further run was launched.

Saved full evidence in `runs/tetris/v2_credit_horizon_seed0/RESULT.md`, plus a small reusable result card with actual policy gameplay and links to the settings and log. Rendering continues to use Matplotlib/Seaborn primitives from the existing MinAtar-style workflow.

## 2026-11-23 — Three-seed replication started

The active goal is to repeat baseline and longer-credit configurations with fresh seeds 1 and 2, each at 65,536 decisions, then compare them with seed 0. The broader objective remains a transferable pixel-only network that measurably improves game score over random; nonzero gradients or a completed run do not satisfy that objective.

Four new runs are scheduled sequentially: baseline seed 1, longer-credit seed 1, baseline seed 2, longer-credit seed 2 (262,144 new training decisions). Settings remain identical to the previous two arms, including corrected timing, original score-only reward, architecture and optimization. The runner snapshots sources and ROM at launch and rejects source drift or existing output directories. No board features, heuristics, recurrence, reward shaping or architecture change is introduced.

Each final policy uses the same 128 previously inspected evaluation seeds 50000–50127 and the preselected illustrative gameplay seed 30016. Random/no-op evaluation is repeated for integrity, not counted as independent control replications. Equal-budget uniform-random rollouts are added for training seeds 1 and 2. Reports will compare individual training seeds, reward discovery and actual scoring; pooling repeated controls as independent samples would overstate evidence. Any proposed next configuration must be justified by these outcomes and the project constraints.

## 2026-11-23 — Three-seed 64K replication completed

Completed the four new seed-1/2 runs at 65,536 steps each, with unchanged pixel-only observations, original score reward and corrected timing. Including seed 0, baseline training reward events were [0, 0, 0]; longer credit events were [1, 2, 0]. Final scoring games (out of 128 per policy) were baseline [1, 1, 1] and longer credit [2, 1, 1], versus 3/128 for the shared random control.

Compared each final policy with its exactly reconstructed untrained network on the same game seeds. Mean score gain over initialization: baseline -0.01042, longer credit -0.00781. Neither configuration passes the exploratory learning gate at 64K steps; useful scoring improvement is not established. Three training seeds and reused evaluation seeds do not establish broad generalization; duplicated random evaluations are not independent control samples.

Configuration, initial-policy, repeated-control, finite-metric and checkpoint checks passed. The run directory `runs/tetris/v2_replication_64k/` contains the full comparison and a compact actual-gameplay card using the preselected longer-credit seed-1 policy. The goal of a demonstrably useful learning setup remains subject to fresh evidence; the screening gate is not a completion criterion.

## 2026-11-23 — Score-replay follow-up prepared

The completed three-seed comparison found no useful scoring gain at 64K. The next mechanism test adds self-imitation replay of the policy's own completed episodes to the longer-credit arm. The CNN still receives only four monochrome frames; actions, corrected timing and original game reward are unchanged. Replay reinforces recorded actions only where their discounted return exceeds the current value estimate. No demonstrations, board features or shaped rewards are introduced.

Use a fresh seed-1 run at 65,536 decisions. Seed 1 was selected because the control encountered rewards early; this is a targeted mechanism test, not independent evidence across training seeds. Ten auxiliary minibatches per rollout activate only after a scoring episode completes. Environment budget is matched, but extra optimizer updates and replay time are logged explicitly. The same 128-game bank remains exploratory; seeds 60000 onward remain unused for confirmation of a frozen candidate. Details and source references are in `SELF_IMITATION.md`.

Validation: 93 tests passed, including six replay/loss/checkpoint tests. A 2,048-step run through each trainer produced bit-identical complete model, optimizer, collection RNG and emulator state before replay activation. Paired model/replay checkpoint integrity and restoration passed. These 4,096 validation decisions are separate from the learning budget. The candidate has no measured learning result yet; source snapshots, full replay checkpoints and actual-gameplay cards will accompany the run.

## 2026-11-23 — 64K score-replay mechanism test completed

Replay does not beat all scoring controls in this exploratory sample; useful learning remains unestablished. Training reward events: 2 (control 2); completed scoring episodes retained: 2; auxiliary updates applied: 170. Final scored games: replay 0/128, longer credit 1/128, untrained 2/128, random 3/128. Evaluation remains exploratory on reused seeds. Full checks and evidence: `runs/tetris/v3_self_imitation_seed1/RESULT.md`. The broader learning goal remains dependent on stronger scoring evidence.

## 2026-11-23 — Matched 512K learning-budget continuation prepared

The 64K replay result scored 0/128, compared with longer-credit PPO 1/128, untrained seed 1 at 2/128 and random 3/128. Correctness checks passed, but useful learning remains unestablished. Completed-episode replay started only at step 32,768 and consumed two scoring episodes. The original Stage C budget was approximately 500K; corrected timing has not yet been tested at that scale.

Continue both seed-1 longer-credit PPO and PPO+replay from their exact 65,536-step checkpoints to 524,288 total decisions each (458,752 added per arm). This keeps source code, architecture, original reward and optimizer settings fixed while increasing experience; no new mechanism is added. Full model/optimizer/RNG/emulator state is retained, plus replay state for the replay arm. Both receive the same environment budget, with auxiliary compute reported separately. CPU estimate is approximately 51 minutes training per arm at recent throughput, plus evaluation.

The fixed launcher checks resolved configuration, parent source/checkpoint integrity, source drift, restored initial evaluations and target steps. Periodic evaluation uses eight development games every 64K; final 128-game comparisons remain exploratory. Freeze a promising checkpoint before unused-seed confirmation and independent training-seed replication. Details, limitations and commands are in `LEARNING_512K.md`. Run completion alone will not satisfy the goal.

### 512K run in progress: equal-budget discovery control

At the 196,608-step checkpoint, longer-credit PPO had no additional training rewards beyond its original two and scored 0/8 on the repeated development games. These small monitoring evaluations do not establish final performance. Added a uniform-random rollout at the same 524,288-decision budget, using the existing corrected-timing control script and seed-1 initialization. Its source is snapshotted separately. This diagnostic leaves both learning runs unchanged and complements the 128-game final skill comparison. The report verifies the first 64K random reward-event records against the earlier control; control overlap with PPO is disclosed when interpreting observed throughput.
