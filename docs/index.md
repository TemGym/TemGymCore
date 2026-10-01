---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# temgym-core

Differentiable electron optics in [JAX](https://docs.jax.dev): one description of a
microscope column, three ways to compute with it.

| Representation | Object | Use it for |
|---|---|---|
| **Rays** | `Ray` | geometry, ABCD transfer matrices, Taylor maps of nonlinear optics |
| **Gaussian beamlets** | `GaussianBeam` | coherent images from $10^5$ to $10^6$ beamlets on the GPU, with gradients |
| **Sampled wavefronts** | `Wavefront` (available, experimental) | Collins-integral propagation between planes, imported arrays, independent checks |

Every representation runs through the same components, so a lens, an aperture or a
biprism is defined once.

## A first ray

```{code-cell} python
import jax
jax.config.update("jax_enable_x64", True)   # required: electron wavelengths are picometres

from temgym_core.ray import Ray
from temgym_core.components import Detector, Lens
from temgym_core.run import column_matrix, run_to_end

ray = Ray(x=0.1, y=0.2, dx=0.3, dy=0.4, z=0.0, pathlength=0.0)
model = (Lens(z=0.5, focal_length=1.0), Detector(z=1.0, pixel_size=(0.01, 0.01), shape=(128, 128)))
run_to_end(ray, model)
```

The same column as a single ray-transfer matrix in $[x, y, \theta_x, \theta_y, 1]$ order
(the drift, lens and second drift compose to $A=\tfrac12,\ B=\tfrac34,\ C=-1,\ D=\tfrac12$):

```{code-cell} python
column_matrix(model)
```

```{toctree}
:maxdepth: 2
:hidden:

install
concepts
tutorials/gaussian_imaging
tutorials/skip_planes
api
changelog
```
