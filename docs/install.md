# Installation

```bash
pip install temgym-core
```

Python 3.11 or later, JAX 0.10.x. Optional extras:

| Extra | Adds |
|---|---|
| `plot` | Matplotlib figures |
| `widgets` | interactive browsers (`ipywidgets`, `ipympl`) |
| `notebooks` | everything needed to run the example notebooks |
| `symbolic` | SymPy, for symbolic ray-transfer matrices |
| `gpu` / `gpu-cuda12` | CUDA 13 / CUDA 12 JAX wheels (Linux, NVIDIA) |

The Gaussian renderer uses custom GPU kernels when a supported NVIDIA GPU and the
`gpu` extra are present and otherwise falls back to a portable CPU implementation.

## 64-bit arithmetic

Electron wavelengths are picometres and optical paths are metres, so single precision
is not enough. Enable 64-bit JAX before calling the library:

```python
import jax
jax.config.update("jax_enable_x64", True)
```
