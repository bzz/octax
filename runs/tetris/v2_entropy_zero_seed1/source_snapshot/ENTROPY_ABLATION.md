# On-policy PPO: remove entropy regularization

The completed long-trace trial scored 2/128, equal to its untrained initialization and below random's 3/128. Longer credit did not provide a useful policy. Frozen-policy checks found pixel-dependent outputs but no rescue from greedy action selection. Local entropy-to-policy gradient ratios of roughly 0.11–0.22 do not show dominant entropy gradients, but also do not isolate the cumulative effect of entropy regularization during long reward-free periods.

Test that effect directly within the existing score-only, on-policy setup. Run fresh seeds 1 and 2 for 65,536 decisions each, using the previously measured 128-step credit-horizon configuration. Change only `ent_coef: 0.01 → 0.0`; all other resolved settings must match the corresponding saved control except the output path. This is not a continuation of the unsuccessful 1024-step trial.

Both conditions use 16 environments, 128-step rollouts, four epochs, 16 minibatches, gamma 0.995, lambda 0.98, the same CNN and four-frame pixels, original score reward, learning rate 0.0005, advantage floor 0.01 and global norm clipping. Each seed has 32 rollouts and 2,048 optimizer updates. Fresh initialization must match its saved control exactly. Every rollout is collected by the current policy, updated with PPO and discarded; there is no replay, demonstration, novelty bonus or auxiliary loss.

With zero-initialized values and no entropy bonus, reward-free experience before the first positive reward should yield zero policy/value gradients. This is an expected diagnostic consequence, not successful learning. Removing entropy regularization can also reduce exploration; there is no presumption it will help.

## Measurement and decision

The two seeds are fixed before launch, including seed 2 whose prior control discovered no reward. Evaluate on the same eight monitoring games and reused 128-game exploratory bank (50000–50127). Compare each seed with its matched entropy-0.01 policy, exact untrained network and shared random control. Report reward discovery, score, survival, entropy, action frequencies, parameter change, gradient behavior and compute. Do not pool repeated random arrays as independent controls.

A candidate should show a gain over its initialization and random play in both training seeds before spending unused confirmation games or extending the budget. A one-seed increase may guide investigation but cannot establish robust learning. Selection, reused seeds and multiple exploratory comparisons remain limitations. The objective is a policy that improves original game score, not zero gradients, finite losses or stronger action preferences.

Actual-gameplay cards use preselected seed 30016. All training sources, resolved settings, checkpoint and evaluation records accompany the results. Upstream-fix commits remain separate and unchanged. The question about allowing a pixel-only novelty reward in a future experiment remains separate; this ablation changes no game reward.

## Reproduction

`JAX_PLATFORMS=cpu .venv/bin/python run_entropy_ablation.py` runs the fixed two-seed series once and refuses overwriting existing runs. For a separate reproduction:

```sh
JAX_PLATFORMS=cpu .venv/bin/python train.py --config-name entropy_zero \
  seed=1 run_dir=runs/tetris/entropy_zero_reproduction_seed1
```
