# Honor the full gradient-norm limit

`max_grad_norm` previously configured `optax.clip`, which limits individual coordinates. A vector of individually small coordinates can still have a very large norm. Use `optax.clip_by_global_norm` to honor the advertised full-gradient limit.

`tests/test_global_gradient_clip.py` initializes the actual PPO optimizer, substitutes identity for Adam to isolate its preceding clip transform, and verifies that a multi-parameter gradient is bounded by the configured norm. No sparse-reward normalization, initialization choice or Tetris-specific hyperparameter changes are included.

The experiment layer later makes the clipping mode explicit so historical checkpoints can retain elementwise behavior; v2 selects global-norm clipping.
