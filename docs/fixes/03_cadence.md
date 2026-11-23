# Tick timers once per emulated frame

Four frames are executed per action, so decrementing only once per action makes timers four times too slow relative to the documented 60 Hz frame clock. Tick after each frame while keeping frame-stack shape and instruction budget unchanged. This patch explicitly requires fps=60; a general fractional-rate scheduler is outside this candidate and should be considered before supporting other FPS values.

## Regression coverage

`tests/test_timer_cadence.py` verifies the behavior independently of the Tetris experiment runner.

Validation uses Python 3.11, CPU JAX, and the installed training dependencies with Gymnax 0.0.9, OpenCV-headless and Pillow. A fresh upstream checkout can prepare these with:

```sh
python -m pip install -e ".[training]" gymnax==0.0.9 opencv-python-headless pillow pytest
JAX_PLATFORMS=cpu python -m pytest -q tests/test_timer_cadence.py
```
