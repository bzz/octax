# Human demonstrations for pixel-based Tetris

The local recorder is ready. Behavior-cloning training is not implemented or run yet: the next input is real human play. The intended experiment is supervised pixel-to-action initialization followed by online, on-policy PPO. This is a separate initialization experiment from PPO trained from scratch.

## Play and record

From this repository, with the existing virtual environment:

```sh
JAX_PLATFORMS=cpu .venv/bin/python record_human.py
```

The first launch compiles the emulator, then opens `http://127.0.0.1:8765/` in your browser. Keep the terminal running. No additional dependencies are needed beyond the project's existing environment. Use `--no-open` to open the printed URL yourself, or `--port 8766` if the default port is occupied.

Click **Start game**, then use:

| Key | Action |
| --- | --- |
| Left / Right | Move left / right |
| Up | Rotate |
| Down | Down |
| No key | No-op |
| Space | Pause / resume |
| Escape | Save and end this game |
| Enter | Start the next game |

Hold a key to repeat. The most recently pressed held key wins; short taps are buffered for the next decision. Leaving the window pauses the game. **Slower · ½×** gives more thinking time without changing the simulated timing, action repeat, or game rules. This is keyboard play, not a placement planner.

Start with 5–10 games or about ten minutes as a pilot. Try to include line clears, but imperfect games are useful for checking coverage and controls. That amount is a starting dataset, not a promise of a strong policy. Click **Save & end game** before leaving; terminal/game-over saves automatically, and Ctrl+C in the terminal saves the current game before stopping the server.

## What gets saved

Each launch creates a fresh `runs/tetris/human_demos/<UTC timestamp>_<id>/` directory. `--out <fresh-directory>` overrides it. Existing episode archives are never overwritten by a new recorder. During play, atomic autosaves occur every 64 decisions (about 4.27 seconds of simulated game time); a hard crash can lose actions since the last autosave. Empty games produce no episode archive.

Each `episode_*.npz` contains:

- `observations`: `(T+1, 4, 32, 64)` boolean pixels, in the same axis order as PPO. **Training pairs are `observations[:-1]` and `actions`.**
- `actions`: `(T,)` integer keypress choices: rotate, left, right, down, no-op.
- Rewards, termination/truncation flags, wall-clock decision times and selected playback pace.
- Full `state__...` emulator snapshots at each observation, for replay/debugging only. **These are not policy inputs.**
- Metadata recording seed, environment settings, ROM hash, source revision and source hashes. The session includes a source snapshot; sidecar JSON files include archive checksums.

Completed games are marked `complete=true`. A manually ended game or an in-progress autosave is a valid prefix marked `complete=false`. Automated test recordings are explicitly labeled `source=automated_test` and must be excluded from human training.

The recorder uses the same corrected Octax environment as PPO: frame-mode timers at 60 Hz, four frames per decision, 700 instructions/second and the 4,500-decision time limit. It does not use the older `play.py` debug loop. The browser displays the four simulated frames in order; slow computation or autosave can slow wall-clock playback but never skips simulated decisions.

Validate a session with:

```sh
JAX_PLATFORMS=cpu .venv/bin/python validate_human_demos.py runs/tetris/human_demos/<session>
```

This replays every action and compares pixels, rewards, done flags and all emulator-state fields exactly. It also checks an independently restored mid-game snapshot. Validation uses the current checkout; the recorded source snapshot helps reproduce older sessions after code changes.

## Next experiment once human data exists

1. Validate recordings, inspect scores, action balance and successful line clears. Preserve no-op timing; blindly dropping waiting frames changes the task.
2. Split by whole games, not neighboring frames, to avoid near-identical observations leaking into validation.
3. Fit the same CNN and actor with supervised cross-entropy on human actions from four-frame pixels. Keep emulator state and reward-derived features out of the input.
4. Evaluate actual game scores on fresh seeds against random play and the original initialization. Held-out action accuracy alone does not establish useful play; mistakes can lead the model into unfamiliar states.
5. Warm-start PPO with the learned **encoder and actor**, initializing the critic and optimizer separately. The current `encoder_checkpoint` option resets the actor and is therefore insufficient for this full behavior-cloning warm start; implement explicit actor transfer before attempting it.

The supervised stage uses a fixed offline demonstration dataset. Subsequent PPO can remain strictly online and on-policy, without continuing to replay human data in its updates. Report the demonstration budget separately from PPO environment steps.

## Verification

Four recorder tests pass: pre-action alignment and exact replay, stale/foreign/invalid request rejection, a real terminal transition, and detection of corrupted pixels. Browser smoke testing exercised arrow keys, pause/resume and Save & end. Its 12-decision automated recording passed exact full-state replay and independent snapshot restoration. No human demonstrations or behavior-cloning results are claimed by these tests.
