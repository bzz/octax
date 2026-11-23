# Persist learned weights and complete training state

The original trainer wrote params=[] and lost the trained model. Keep actual weights in the existing results payload and additionally save a versioned atomic MessagePack checkpoint containing model/optimizer state, RNG, emulator state, global step and the complete pre-mutation configuration. The reusable helper restores against a caller-provided state template.

## Regression coverage

`tests/test_checkpoint.py` verifies the behavior independently of the Tetris experiment runner.

Validation uses Python 3.11, CPU JAX, and the installed training dependencies with Gymnax 0.0.9, OpenCV-headless and Pillow. A fresh upstream checkout can prepare these with:

```sh
python -m pip install -e ".[training]" gymnax==0.0.9 opencv-python-headless pillow pytest
JAX_PLATFORMS=cpu python -m pytest -q tests/test_checkpoint.py
```
