# Fork changes and upstream candidates

Fork of [riiswa/octax](https://github.com/riiswa/octax), starting at `3aa53b516152e97f2ed91eae6e33b6ee9a97596b`.

## Upstream candidate series

The [`upstream-fixes` branch](https://github.com/bzz/octax/tree/upstream-fixes) ends at `98374b72ad5d7b0b5c6824918917029d7297b54e`. It contains only the five fixes below, their focused regression tests and rationale. It does not include experiment runners, Tetris hyperparameters, legacy replay modes, dependency pins or result artifacts. These are candidates for review, not already accepted upstream changes.

| Order | Commit | Rationale and tests |
| --- | --- | --- |
| 1 | [`f00b8c5`](https://github.com/bzz/octax/commit/f00b8c515a003ce08cd186c660c5c3e0f044e4a7) | [CNN layout and spatial kernels](docs/fixes/01_cnn.md) |
| 2 | [`fd03b07`](https://github.com/bzz/octax/commit/fd03b07c663f0cdbcbd9f2ec5daf357b7b7e1e6f) | [Saturating CHIP-8 timer arithmetic](docs/fixes/02_saturation.md) |
| 3 | [`7e4a057`](https://github.com/bzz/octax/commit/7e4a05740ffd7203bbcf5ea5459ca0fa6e2c8f4a) | [Timer cadence across four simulated frames](docs/fixes/03_cadence.md) |
| 4 | [`ecaf5a3`](https://github.com/bzz/octax/commit/ecaf5a31bc4e53d72d18a447a30ed807977c2bff) | [Actual weights and resumable checkpoints](docs/fixes/04_checkpoint.md) |
| 5 | [`98374b7`](https://github.com/bzz/octax/commit/98374b72ad5d7b0b5c6824918917029d7297b54e) | [Full gradient-norm clipping](docs/fixes/05_gradient_clip.md) |

Each commit is independently reviewable. Apply the series in order: cadence uses the saturation helper, and later PPO changes share context with the CNN fix. Other fixes can be cherry-picked separately if needed, resolving context against the target upstream revision. The checkpoint helper is generic; its trainer integration preserves the original training flow and pickle output while adding a full-state MessagePack artifact.

The cadence fix currently requires 60 FPS to keep frame and CHIP-8 clocks aligned. Supporting arbitrary FPS needs a fractional timer scheduler; this series does not claim to implement one. The clipping fix bounds the pre-Adam gradient norm, not the resulting parameter-update norm.

Validation on macOS arm64, Python 3.11 and CPU JAX: focused tests passed at each of the first four commit snapshots; the final isolated five-fix tree passes **78 tests**. The complete fork passes **87 tests**. The packaged evaluator also reproduced every saved per-episode array for the final v2 checkpoint’s eight-game development evaluation exactly. See the per-fix notes for commands. The isolated tests used the recorded experiment dependency environment, including Gymnax 0.0.9, OpenCV-headless and Pillow; this is not a claim of compatibility with every allowed dependency version. Existing dtype/deprecation warnings remain.

To export only the upstream candidates from this fork:

```sh
git format-patch 3aa53b516152e97f2ed91eae6e33b6ee9a97596b..upstream-fixes
```

## Experiment layer on main

Commits after `upstream-fixes` add the pixel-only Tetris PPO workflow and then the recorded results. [EXPERIMENT.md](EXPERIMENT.md) gives setup and reproduction commands; [IMPLEMENTATION_LOG.md](IMPLEMENTATION_LOG.md) records the investigation and decisions; [runs/README.md](runs/README.md) indexes evidence.

The experiment layer includes four-frame monochrome observations, training/evaluation/checkpoint tools, CPU dependency pins, MinAtar-style Matplotlib/Seaborn exports, source manifests, and explicit historical-timing replay. Zero-initialized value output and the 0.01 advantage standard-deviation floor are **experimental stabilization choices**, not universal PPO bug fixes. The main configuration selects global-norm clipping; the class also permits historical elementwise clipping so old checkpoints can be replayed faithfully.

No features derived from board geometry, reward shaping, recurrence or transformers are used. The longer-credit profile has now been tested at the same 65,536-step budget; [its result](runs/tetris/v2_credit_horizon_seed0/RESULT.md) records the exploratory comparison. Neither the historical v1 experiments nor the fresh 65,536-step v2 baseline demonstrate superiority to random play. v1 and v2 use different timing semantics and must not be pooled.

## Publication and provenance

The commit series was organized after the experiments ran. Historical `system.json` files record the base HEAD in a field named `upstream_commit`; they do not identify the then-uncommitted working tree. The final v2 baseline includes exact source snapshots with SHA-256 hashes. Earlier runs retain hashes, configurations, checkpoints and the before-fix environment/PPO snapshots, but not every historical source file; exact reconstruction of every earlier working tree is not claimed.

Current metadata records the fixed upstream base, current source commit and dirty state separately. Packaging extracted identical checkpoint functions into `octax/checkpoints.py` and updated metadata/cache paths; it did not retrain or modify saved model state. Known local checkout prefixes in published text logs and checkpoint-location metadata are normalized to repository-relative paths. Original numeric measurements, checkpoint bytes, source snapshots and their hashes are preserved. No credential files, virtual environment or local caches are included.

The longer-credit investigation adds only experiment/reporting work after the original publication: a fresh matched-budget runner, paired comparison checks and a reusable compact gameplay card. The five-commit `upstream-fixes` branch remains unchanged.

The three-seed replication adds fresh baseline/longer-credit seeds 1 and 2, exact untrained-network comparisons and an aggregate report that treats training seeds as replication units. [Results](runs/tetris/v2_replication_64k/RESULT.md) retain the limitations of reused game seeds and shared random controls. All additions are experiment-layer work; `upstream-fixes` remains unchanged.

### Pixel-only self-imitation candidate

`self_imitation.py`, `train_self_imitation.py` and `conf/self_imitation.yaml` add an experimental completed-episode replay objective after the negative three-seed comparison. This is experiment-only work, excluded from `upstream-fixes`. See `SELF_IMITATION.md` for rationale, constraints, validation and the matched-budget protocol.

### Matched 512K continuation

The fixed seed-1 PPO/replay continuation and cumulative audit are experiment-only work. Parent-checkpoint hashes and source snapshots preserve provenance across the resumed budget; `upstream-fixes` is unchanged. See [results](runs/tetris/learning_512k_seed1/RESULT.md).

## Strictly on-policy long-trace PPO at 512K

The fresh long-trace PPO trial and matched-control report are experiment-only work. No emulator or upstream-fix changes enter this run. [Evidence](runs/tetris/v2_long_trace_512k_seed1/RESULT.md).

## Two-seed PPO entropy ablation at 64K

The two-seed entropy-coefficient ablation is experiment-only work. Each fresh 64K run changes only coefficient 0.01 to 0.0. Frozen sources, exact initialization and original-score comparisons are recorded; upstream-fixes is unchanged. [Results and animations](runs/tetris/v2_entropy_zero_64k/RESULT.md).

### Inactive pixel-novelty component

The optional RND component, terminal-observation wrapper and opt-in on-policy integration are experiment preparation, not upstream emulator fixes. Twelve targeted tests validate the component and integration, including exact disabled-bonus baseline equivalence and complete checkpoint continuation. Existing PPO trainers do not import them. No novelty learning pilot has run; learning evidence remains outstanding. See [scope and checks](PIXEL_EXPLORATION_PROPOSAL.md).
