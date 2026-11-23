# Matched 512K learning-budget test

The corrected 64K experiments established reward discovery and checkpoint correctness, but no scoring advantage over random. Three baseline seeds found zero rewards; longer-credit seeds found 1/2/0. The selected seed-1 self-imitation follow-up retained two scoring episodes and applied 170 auxiliary updates, but scored 0/128 against longer credit 1/128, its untrained network 2/128 and random 3/128. The learning objective remains unmet.

The original staged experiment proposed approximately 500K steps for the first learning experiment. The only previous long run used the old incorrect timing. Rather than making another method change after roughly fifty games, this round tests more experience under corrected timing.

## Fixed comparison

Continue both saved seed-1 longer-credit PPO and PPO+self-imitation states from 65,536 to **524,288 total decisions** (512 × 1024; eight times the pilot budget). Each arm adds 458,752 decisions, for 917,504 new decisions in the series. Run sequentially on CPU with 16 environments. At recent measured throughput around 150 decisions/s, training alone projects about 51 minutes per arm, plus evaluation and rendering.

Restore the full model, optimizer, collection RNG and emulator state. Replay also restores pending/completed experience, priorities and its own RNG. These are continuations, not new independent seeds. Keep all architecture, timing, reward, credit and optimizer settings fixed. PPO uses a constant Adam learning rate, so increasing the target step does not change a learning-rate schedule. Keep the same four monochrome frames and original score; no board information or demonstrations are added.

Both arms reach 16,384 cumulative PPO optimizer updates. Replay adds auxiliary updates and processing time, which must be reported separately. The comparison matches environment decisions, not compute. Use eight development games at each additional 64K checkpoint; inspect score, reward discovery, entropy, action frequencies and finite/checkpoint checks. Stop on a genuine numerical or execution failure. A run finishing without rewards does not satisfy the goal.

## Evidence and decision

The launcher records parent-checkpoint hashes, exact source/ROM snapshots, continuation provenance, status and logs. It refuses existing outputs, detects source drift and verifies that each restored initial development evaluation exactly matches its parent's final evaluation. The final 128-game bank (50000–50127) remains exploratory. Compare each final policy with its own 64K parent, the exact untrained seed-1 network and the shared random control. Count training events cumulatively across parent and continuation; do not mistake a segment counter for a fresh full run.

The illustration remains game seed 30016. Each completed round gets the actual-policy animation card, quantitative comparison and implementation-log entry. If a candidate scores above random and its untrained network on the exploratory bank, freeze it before evaluating unused seeds 60000–60511. Replicate promising settings across independent training seeds before claiming a dependable setup. More elapsed time, gradients or parameter changes alone are insufficient.

Run the fixed series with `JAX_PLATFORMS=cpu .venv/bin/python run_learning_512k.py`. This launcher intentionally refuses duplicate starts. To reproduce independently, resume the published parent with `train.py --config-name credit_horizon` or `train_self_imitation.py --config-name self_imitation`, set seed 1, a fresh output directory and `total_timesteps=524288`; the parent and settings must match.
