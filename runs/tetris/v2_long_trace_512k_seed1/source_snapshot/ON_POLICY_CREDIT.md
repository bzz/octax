# Strictly on-policy PPO: longer rollout and credit trace

The matched 512K round did not meet the learning goal: on-policy PPO scored 1/128, its untrained network 2/128 and random 3/128. The replay side experiment also scored 1/128 and is now archived; its protected-retention follow-up is shelved after the user's on-policy clarification.

## Evidence for the next test

Entropy drift toward a uniform policy suggested excessive regularization as one possibility. The gradient audit does not show entropy dominating the local updates: on fresh reward-free rollouts from the 64K and 512K PPO checkpoints, mean entropy-to-policy gradient-norm ratios were about 0.11 and 0.14. These are local measurements, not a causal test of long-run forgetting. The reconstructed first-minibatch update matches the actual PPO implementation to a maximum state-element difference of approximately 1.5e-8. Keep the entropy coefficient at 0.01 in this round.

Credit timing gives a more direct limitation. The first recorded score occurred at per-environment decision 907, just 11 transitions into a 128-step rollout. Only those 11 transitions were still available in that rollout for direct GAE credit from this reward; earlier rollouts had already been updated. The second score had 118 transitions available. Those records precede the first replay update and are used only for diagnosis. They are not training inputs or demonstrations.

## Fixed on-policy ablation

Fresh seed 1, 524,288 decisions, the same four-frame monochrome CNN and original game-score reward. Use the unchanged `train.py` PPO implementation. Each rollout is collected by the current policy, used for four PPO epochs and then discarded. There are no historical replay updates, auxiliary losses, shaped rewards, board features, recurrence or transferred weights.

Change rollout length 128→1024, gamma 0.995→0.999 and GAE lambda 0.98→1.0. Increase minibatches 16→128 to retain batch size 128. This is a **bundled credit-assignment ablation**, not an isolated causal test of one hyperparameter. All other settings, including learning rate and entropy coefficient, stay fixed. Both this arm and the previous 512K PPO arm receive 16,384 optimizer updates; this arm collects 32 larger rollouts rather than 256 smaller ones.

Lambda 1 removes GAE's additional decay within the available rollout. It does not remove terminal boundaries or guarantee a complete-episode Monte Carlo target: unfinished rollouts still bootstrap from the value function. Longer rollouts change update timing and RNG progression, so the earlier reward discoveries are not guaranteed to recur. Initial-network inference must still reproduce seed 1 exactly.

## Measurement and reproduction

Evaluate eight development games every 65,536 decisions. Final evaluation reuses the exploratory 128-game bank, seeds 50000–50127, against saved PPO, untrained and random controls. Report training reward discovery, score, entropy, action frequencies, checkpoint restoration and throughput. Compare the full 512K budgets; do not treat development games as fresh confirmation. Game seed 30016 remains the illustrative animation.

Only an on-policy candidate showing an exploratory gain can proceed to a frozen-checkpoint test on unused seeds 60000–60511 and independent training-seed replication. The objective remains actual game-score improvement, not larger gradients or longer traces.

Run the recorded trial with `JAX_PLATFORMS=cpu .venv/bin/python run_long_trace.py`. For a separate reproduction, use a fresh path:

```sh
JAX_PLATFORMS=cpu .venv/bin/python train.py --config-name long_trace \
  run_dir=runs/tetris/long_trace_reproduction_seed1
```

The diagnostics collect 2,048 new decisions per saved PPO policy solely for gradient inspection; no diagnostic updates are saved or used to initialize this run. `diagnose_ppo_entropy.py` and `diagnose_credit_timing.py` reproduce those measurements. Their limitations and source hashes accompany the results.
