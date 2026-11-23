# Fix CNN channel layout and spatial kernels

CNN input was NCHW although Flax Conv expects NHWC. Scalar later kernel sizes selected 1D convolutions. Convert once in a shared encoder used by actor and critic, and specify all spatial dimensions explicitly.

## Regression coverage

`tests/test_cnn_layout.py` verifies the behavior independently of the Tetris experiment runner.

Validation uses Python 3.11, CPU JAX, and the installed training dependencies with Gymnax 0.0.9, OpenCV-headless and Pillow. A fresh upstream checkout can prepare these with:

```sh
python -m pip install -e ".[training]" gymnax==0.0.9 opencv-python-headless pillow pytest
JAX_PLATFORMS=cpu python -m pytest -q tests/test_cnn_layout.py
```
