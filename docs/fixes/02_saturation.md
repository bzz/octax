# Prevent unsigned timer underflow

Subtracting one from uint8 zero wraps to 255 before maximum is applied. Widen to signed arithmetic, saturate, and restore uint8 for delay and sound timers. This commit intentionally preserves the old once-per-action cadence to isolate arithmetic correctness.

## Regression coverage

`tests/test_timer_saturation.py` verifies the behavior independently of the Tetris experiment runner.

Validation uses Python 3.11, CPU JAX, and the installed training dependencies with Gymnax 0.0.9, OpenCV-headless and Pillow. A fresh upstream checkout can prepare these with:

```sh
python -m pip install -e ".[training]" gymnax==0.0.9 opencv-python-headless pillow pytest
JAX_PLATFORMS=cpu python -m pytest -q tests/test_timer_saturation.py
```
