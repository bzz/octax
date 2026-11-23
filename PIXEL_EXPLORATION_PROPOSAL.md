# Pixel-only exploration with on-policy PPO — proposed, not enabled

Status: the reward-scope preference remains unanswered. `pixel_novelty.py` and the opt-in `on_policy_novelty.py` integration are implemented and tested, but neither is imported by the existing PPO trainers. Tiny test-only optimizer updates validate the integration; no novelty learning pilot or training smoke benchmark has run. The completed score-only experiments and their original checkpoints remain the evidence for the current learning result.

## What the source review establishes

Random network distillation (RND) rewards prediction error against a fixed random network on an observation. Its published PPO method separates intrinsic and extrinsic value estimates and trains on each newly collected rollout. This permits online, on-policy policy updates without replaying old successful episodes. The paper also normalizes the auxiliary inputs and rewards. These facts support method compatibility; they do not establish that it will solve Octax Tetris. [Original paper, method and Algorithm 1](https://arxiv.org/html/1810.12894v1).

The reference launcher defaults to a recurrent policy but explicitly supports a CNN option. Its Atari, TensorFlow and MPI setup is not our JAX/CPU configuration. Copying its defaults would violate this project's no-recurrence constraint. [Reference launcher](https://raw.githubusercontent.com/openai/random-network-distillation/master/run_atari.py). The feed-forward implementation has separate score/novelty value outputs and predictor/target networks; it uses the newest image for the auxiliary prediction task. [CNN implementation](https://raw.githubusercontent.com/openai/random-network-distillation/master/policies/cnn_policy_param_matched.py).

The paper's common experiment used 30,000 rollouts × 128 decisions × 128 environments: 491,520,000 decisions, about 938 times our 524,288-decision trial. This arithmetic is a scale comparison, not a proposed budget. A small Octax pilot would test this adaptation, not reproduce the paper's performance. [Paper, Sections 3 and A.4](https://arxiv.org/html/1810.12894v1).

## Proposed Octax adaptation

These are project-specific choices to freeze before a pilot, not validated settings from the paper:

- Keep the existing feed-forward actor, four monochrome frames, five keypress actions, corrected timing and original game score. Add a second value output for novelty; no board features, demonstrations, recurrent state or historical replay.
- Use two separate small feed-forward image networks for RND. The target remains fixed; the predictor learns from the current rollout only. A proposed size is three convolutions with 16/32/32 channels and 128 output features, with an extra 256-unit predictor layer. The actor and shared CNN dimensions remain unchanged. These sizes need a CPU smoke benchmark before a longer run.
- Use the newest full 32×64 frame, already binary, for both auxiliary networks. Do not divide the pixels by 255 or crop game-specific regions. Freeze preprocessing and bonus values for each PPO batch; keep the target bit-identical throughout training.
- Retain the 128-step credit configuration and entropy coefficient 0.01. Compute separate GAE streams, initially using gamma 0.995 for score and 0.99 for novelty, lambda 0.98 for both. Compute novelty GAE from normalized intrinsic rewards, then combine score advantage with a proposed 0.01 times novelty advantage. Retain the existing normalization of the combined PPO advantage. Log both contributions explicitly; this coefficient is a declared pilot setting, not an inferred optimum. The reference implementation likewise keeps separate returns and combines advantages. [PPO reward/advantage handling](https://raw.githubusercontent.com/openai/random-network-distillation/master/ppo_agent.py).
- Initially terminate both return streams at game-over. This intentionally differs from the paper's continuing intrinsic-return setting and must be labeled an adaptation. It keeps episode boundaries explicit in the first test. Novelty statistics and predictor weights persist online across episodes, but previous transitions are not replayed.

## Integration finding: terminal pixels must be explicit

The installed Gymnax `Environment.step` replaces the returned observation with a reset observation when `done` is true. The current `OctaxGymnaxWrapper.step_env` has the actual successor pixels before that replacement. A novelty collector must retain those transition pixels separately from the reset pixels used for the next action. Otherwise a terminal action could be rewarded for the next game's initial display. This is an integration requirement found by reading the local code, not an established bug in the existing score-only PPO runs.

Use experiment-specific observation plumbing and verify it against raw environment transitions, preserving baseline timing and RNG behavior. Do not expose emulator registers, board geometry or score-derived features to either the actor or novelty networks.

## Required evidence before a learning claim

Before any novelty-driven policy run, implement and check: frozen target hashes; correct terminal pixels; fixed bonus/preprocessing within PPO epochs; no retained transition buffer across rollouts; predictor fitting on repeated diagnostic inputs; and full restoration of predictor, target, optimizer, normalizer, policy, RNG and environment state. With the intrinsic coefficient disabled, the extended path must reproduce the original PPO actor/critic updates and trajectories; additional-head gradients must also be disabled for that check.

Then run a separately counted short smoke test and measure CPU overhead. A proposed first discovery pilot uses fresh seeds 1 and 2 at 65,536 training decisions each, matched to the saved score-only controls. Report any normalization warm-up interactions separately and include them in total environment-use accounting. Freeze settings before either pilot starts; do not tune the coefficient against the 128-game evaluation bank.

Success remains improved original game score, evaluated with the existing ordinary game resets and no novelty reward in the score. Monitor raw score events, novelty scale, return/advantage balance, episode lengths, predictor error, action probabilities and actual gameplay. New or unpredictable images could be attractive without producing line clears; that is a project-specific failure possibility, not a result already observed for RND here. Additional novelty reward or unique frames alone would not justify declaring the goal complete.

Use the reused 128-game bank only for exploratory screening. A frozen candidate needs unused game seeds and independent training-seed evidence. Do not consume the reserved 60000–60511 bank for a policy that fails the scoring screen. Preserve the implementation log, source/checkpoint provenance and actual-policy animation cards.

## Inactive component validation

`pixel_novelty.py` now contains the small target/predictor CNNs, observation and discounted-return moments, frozen per-rollout rewards/targets, predictor updates, stale-batch rejection and an experiment-specific terminal-observation wrapper. Persistent auxiliary state has no experience buffer. The existing trainer, actor/critic and environment wrapper are unchanged.

Seven targeted CPU tests passed (7.73 seconds). They check the full newest-frame input, unchanged target arrays and canonical hashes, predictor fitting, episode-boundary normalization, zero-variance inputs, stale-batch rejection, complete auxiliary checkpoint restoration and an identical next update. A natural terminal no-op episode verifies that transition pixels are retained while the original wrapper's reset observation, state, reward and done flag remain exact. Synthetic predictor fitting and this environment validation are not PPO training or learning evidence.

The initial integrity assertions were corrected to canonicalize dictionary order across JIT and compare serialized optimizer values rather than newly created Python callback identities. Exact array comparisons and restored next-update checks remain in place. Three existing emulator dtype warnings remain.

Reproduce the component checks with `JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q tests/test_pixel_novelty.py`.

## Opt-in on-policy integration validation

`on_policy_novelty.py` composes the existing PPO state with a separate intrinsic-value head and the auxiliary RND state. It does not register another Rejax initializer: doing so would alter the order of RNG splitting and invalidate matched initializations. Tagged independent initialization keys preserve the original policy parameters and collection RNG. The extra value head consumes only the original shared CNN's pixel features. Both return streams stop at game-over, with separate discounts and unchanged original-score rewards. PPO gradients from both value heads and the actor are clipped jointly before their Adam updates; the separate predictor is fit afterward on the current rollout only.

The host lifecycle requires collection followed by consumption of that rollout before further collection, and rejects stale generation/step/RNG markers. Frozen bonus values are computed once before PPO epochs. These guards prevent accidental reuse in the runner; they are not a security boundary against deliberately modifying state. Persistent state contains no historical transition buffer. Format-2 checkpoints contain the entire PPO, auxiliary-head, predictor, fixed-target, normalizer, environment and RNG state. Format-1 policy exports are explicitly inference-only and cannot restore either full trainer.

The combined targeted suite passed **12 tests in 25.68 seconds** on CPU. Five integration tests verify:

- Exact original initialization, trajectory, full PPO state and metrics with novelty disabled across two updates, including validation-only injected reward/terminal flags to exercise nonzero targets. Auxiliary parameters and optimizers stay fixed.
- Separate score/novelty GAE against an independent NumPy recurrence with internal terminals and nonzero rollout-end bootstrap values.
- Finite enabled updates, changed policy/intrinsic-head/predictor weights, fixed target weights, bounded reported clipped gradient norm and no additional environment steps during updates.
- Rejection of duplicate, stale and pending rollouts/checkpoints.
- Full save/restore into a fresh template followed by an exactly identical next collection, PPO update, predictor update and metrics; configuration mismatch rejection; inference-only exports rejected for resume.

Reproduce with `JAX_PLATFORMS=cpu MPLCONFIGDIR=.matplotlib .venv/bin/python -m pytest -q tests/test_on_policy_novelty.py tests/test_pixel_novelty.py`. Existing emulator dtype warnings and dependency deprecations remain. These tests execute tiny diagnostic rollouts and optimizer updates; they do not measure learning or constitute a pilot. A reproducible run harness, CPU smoke benchmark, source freeze and actual original-score comparison remain outstanding.

## Decision needed

The pending question is whether to permit this pixel-derived exploration reward while retaining on-policy PPO and original-score evaluation, or to keep the training reward strictly game score. The request is about the experimental objective; no skill file or automatic approval rejection requires it. Without that answer, this proposal remains inactive.

[Latest score-only result](runs/tetris/v2_entropy_zero_64k/RESULT.md) · [Implementation log](IMPLEMENTATION_LOG.md).
