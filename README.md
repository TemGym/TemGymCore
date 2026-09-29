# temgym-core

**Differentiable electron optics in JAX.** Describe a microscope column once and compute with
it in three ways: as **rays** (geometry, ABCD transfer matrices), as **Gaussian beamlets**
(coherent images from $10^5$–$10^6$ beamlets, on the GPU), and, planned, as **sampled
wavefronts**. Every result is a JAX array, so gradients with respect to lens strengths,
positions or sample parameters are one `jax.grad` away.

[Documentation](https://temgym-core.readthedocs.io) ·
[Changelog](https://github.com/TemGym/TemGymCore/blob/main/CHANGELOG.md) ·
[Architecture](https://github.com/TemGym/TemGymCore/blob/main/ARCHITECTURE.md)

> Status: alpha. The API is stabilising towards 0.1.0.

## Install

```bash
pip install temgym-core              # CPU
pip install "temgym-core[gpu]"       # NVIDIA GPU on Linux (CUDA 13); gpu-cuda12 for CUDA 12
pip install "temgym-core[plot]"      # Matplotlib figures
```

Python 3.11 or later. Electron wavelengths are picometres, so 64-bit arithmetic is required:
call `temgym_core.enable_x64()` once, before creating arrays. Functions that need it raise a
clear error instead of quietly computing in single precision.

## A ray through a lens

```python
import temgym_core

temgym_core.enable_x64()

from temgym_core.components import Detector, Lens
from temgym_core.ray import Ray
from temgym_core.run import column_matrix, run_to_end

ray = Ray(x=0.1, y=0.2, dx=0.3, dy=0.4, z=0.0, pathlength=0.0)
lens = Lens(z=0.5, focal_length=1.0)
detector = Detector(z=1.0, pixel_size=(0.01, 0.01), shape=(128, 128))
model = (lens, detector)

ray_out = run_to_end(ray, model)
print(ray_out)  # Ray(x=0.275, y=0.4, dx=0.05, dy=0.0, z=1.0, pathlength=0.88875, ...)
```

Coordinates are metres and radians. A *model* is a sequence of components ordered along $z$;
`run_to_end` carries the ray from plane to plane.

## The same column as one matrix

`column_matrix` differentiates the native ray propagation, so the ABCD matrix of any column is
obtained, never derived by hand. It acts on $[x, y, \theta_x, \theta_y, 1]$; the trailing 1
carries offsets.

```python
matrix = column_matrix(model)
print(matrix)
# [[ 0.5  0.   0.75 0.   0.  ]
#  [ 0.   0.5  0.   0.75 0.  ]
#  [-1.   0.   0.5  0.   0.  ]
#  [ 0.  -1.   0.   0.5  0.  ]
#  [ 0.   0.   0.   0.   1.  ]]
```

## Gradients

```python
import jax

def output_x(focal_length):
    return run_to_end(ray, (Lens(z=0.5, focal_length=focal_length), detector)).x

gradient = jax.grad(output_x)(1.0)
print(gradient)  # 0.125: how far the ray moves per unit change of focal length
```

## Gaussian beamlets and a coherent image

A wide beam is represented by many small Gaussians. They ride through the same components, and
their complex fields are summed on the detector.

```python
import jax.numpy as jnp

from temgym_core.evaluate import evaluate_gaussians
from temgym_core.source import rectangular_input_wave

# A 40 um square lit by 22 x 22 beamlets (4 um waist) at 200 kV, imaged 1:1 by a lens.
beam = rectangular_input_wave(
    aperture_width=40e-6, aperture_height=40e-6, waist=4e-6,
    voltage=200e3, overlap_factor=2.0, z0=0.0,
)
imaging = (
    Lens(z=0.4, focal_length=0.2),
    Detector(z=0.8, pixel_size=(1e-6, 1e-6), shape=(64, 64)),
)
at_detector = run_to_end(beam, imaging)
image = jnp.abs(evaluate_gaussians(at_detector, imaging[-1])) ** 2
print(float(image[32, 32]))  # about 0.68: the flat top of the imaged square
```

On a supported NVIDIA GPU `evaluate_gaussians` uses custom kernels; elsewhere it falls back to
a portable implementation with the same result.

## Going further

- [`examples/microscope_models/microscope_calibration_pipeline.ipynb`](https://github.com/TemGym/TemGymCore/blob/main/examples/microscope_models/microscope_calibration_pipeline.ipynb):
  invert a microscope from beam images.
- [`examples/sem_reflection/`](https://github.com/TemGym/TemGymCore/tree/main/examples/sem_reflection):
  a scanning-electron column with reflection diffraction.
- [`examples/holography/gaussian_biprism_hologram.ipynb`](https://github.com/TemGym/TemGymCore/blob/main/examples/holography/gaussian_biprism_hologram.ipynb):
  an off-axis hologram from summed Gaussian beamlets.
- [Contributing](https://github.com/TemGym/TemGymCore/blob/main/CONTRIBUTING.md).

## Citing

Landers, D., Clancy, I., Weber, D., Dunin-Borkowski, R. E. & Stewart, A. (2023).
*TEMGYM Basic: transmission electron microscopy simulation software for teaching and training of microscope operation.*
*J. Appl. Cryst.* **56**, 1267-1276, [doi:10.1107/S1600576723005174](https://doi.org/10.1107/S1600576723005174).

## License

GPL-3.0. See `LICENSE.md`.
