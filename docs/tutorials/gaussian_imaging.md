---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Gaussian beamlets: from a wide beam to a coherent image

A wide coherent beam is written as a sum of many small Gaussian beamlets. Each beamlet carries a
position, a slope, a complex amplitude, the optical path it has accumulated and a complex
curvature matrix `Q_inv`. `run_to_end` moves the whole bundle through the same components a ray
uses, and `evaluate_gaussians` adds the beamlets' complex fields on a detector.

## 1. A source

```{code-cell} python
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import temgym_core

temgym_core.enable_x64()

from temgym_core.components import Detector, Lens
from temgym_core.evaluate import evaluate_gaussians
from temgym_core.run import run_to_end
from temgym_core.source import rectangular_input_wave

beam = rectangular_input_wave(
    aperture_width=40e-6, aperture_height=40e-6, waist=4e-6,
    voltage=200e3, overlap_factor=2.0, z0=0.0,
)
print(beam.x.shape[0], "beamlets, each with a 4 um waist")
```

`overlap_factor` is the waist divided by the lattice spacing. With 2, neighbouring beamlets
overlap strongly, so their sum is a smooth, nearly uniform field over the 40 um square.

## 2. Through the column

A lens with focal length 0.2 m, 0.4 m from the object and 0.4 m from the detector, images the
square one to one:

```{code-cell} python
detector = Detector(z=0.8, pixel_size=(1e-6, 1e-6), shape=(64, 64))

def image(focal_length, beam=beam, method="scan"):
    column = (Lens(z=0.4, focal_length=focal_length), detector)
    field = evaluate_gaussians(run_to_end(beam, column), detector, method=method)
    return jnp.abs(field) ** 2

intensity = image(0.2)
fig, ax = plt.subplots(figsize=(3.6, 3.4), constrained_layout=True)
ax.imshow(np.asarray(intensity), cmap="gray", extent=(-32, 32, -32, 32))
ax.set(xlabel="x (um)", ylabel="y (um)", title="Imaged 40 um square")
plt.show()
```

## 3. Check it against optics

A perfect 1:1 image of a 40 um square is 40 um wide. Measure the full width at half maximum
along the middle row:

```{code-cell} python
def fwhm_um(intensity):
    row = np.asarray(intensity[32])
    return float((row > row.max() / 2).sum())   # 1 um pixels

print(f"width at half maximum: {fwhm_um(intensity):.0f} um")
```

The beamlet lattice is only a numerical basis, so the answer must not depend on it. Refine it:

```{code-cell} python
for overlap in (1.5, 2.0, 3.0):
    finer = rectangular_input_wave(
        aperture_width=40e-6, aperture_height=40e-6, waist=4e-6,
        voltage=200e3, overlap_factor=overlap, z0=0.0,
    )
    print(f"overlap {overlap}: {finer.x.shape[0]:4d} beamlets, "
          f"width {fwhm_um(image(0.2, beam=finer)):.0f} um")
```

## 4. Gradients

Everything above is JAX code, so the image can be differentiated with respect to a component
parameter. Here the derivative of the central intensity with respect to the focal length, checked
against a finite difference:

```{code-cell} python
import jax

centre = lambda f: image(f)[32, 32]
f0, step = 0.2, 1e-6
automatic = jax.grad(centre)(f0)
finite = (centre(f0 + step) - centre(f0 - step)) / (2 * step)
print(f"jax.grad: {float(automatic):.4f}   finite difference: {float(finite):.4f}  (per metre)")
```

```{note}
Gradients go through `evaluate_gaussians`, which is plain JAX: see *Gradients through the
images* in the concepts page.
```
