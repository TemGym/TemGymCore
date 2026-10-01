---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Skipping planes: sampled wavefronts

```{warning}
`temgym_core.wave` is **experimental**: it works and is tested against independent references, but its
API may still change.
```

A Gaussian beamlet carries a whole wave in a handful of numbers. Sometimes you want the wave itself,
sampled on an array: an exit wave that you loaded from a file, an aperture whose edge is not a Gaussian,
or an independent check on the beamlet sum. Propagating an array plane by plane through a microscope
column would need every intermediate plane sampled finely enough for every lens chirp. The linear
canonical (Collins) transform avoids that: every *linear* stretch of the column (drifts, lenses,
stigmators, deflectors, rotations, solenoids) is one integral built from its $5\times5$ ray matrix
$\begin{pmatrix}A&B\\C&D\end{pmatrix}$, so all planes in between are skipped. Thin elements that are not
linear, such as an aperture, multiply the field where they sit.

## 1. A Gaussian through a lens by one transform

The column is a lens 3 mm behind the source and a plane 5 mm behind the lens:

```{code-cell} python
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import temgym_core

temgym_core.enable_x64()

from temgym_core import wave
from temgym_core.components import Lens, Plane
from temgym_core.constants import energy2wavelength

voltage = 200e3
wavelength = float(energy2wavelength(voltage))
column = [Lens(z=3e-3, focal_length=2e-3), Plane(z=8e-3)]

transform = wave.column_transform(column, source_z=0.0, voltage=voltage)
A, B, C, D = (np.asarray(transform.matrix)[i:i + 2, j:j + 2] for i in (0, 2) for j in (0, 2))
print("A =", A[0, 0], " B =", B[0, 0], "m/rad  C =", C[0, 0], "rad/m  D =", D[0, 0])
print("constant phase of the span:", float(transform.phase0), "rad")
```

`column_transform` differentiates the same ray code the rest of the package uses, so the matrix is never
derived by hand. It also propagates a Gaussian through the same components: the matrix fixes the
transform only up to a sign (the Gouy phase), and that chief Gaussian supplies the physical one, so you
never see it.

The wave lives on a *lattice*: sample positions $\mathbf r = \mathbf c + P\,\boldsymbol\mu$ with a
full 2 x 2 pitch matrix $P$, so a lattice can be rotated, scaled or sheared. A Gaussian of 100 nm waist on
192 x 192 samples of 10 nm:

```{code-cell} python
lattice = wave.Lattice.regular((192, 192), 10e-9)
waist = 100e-9
Q0 = 1j * wavelength / (np.pi * waist**2)                     # complex curvature 1/q of the beam
source = wave.Wavefront(
    wave.gaussian_field(lattice.positions(), 1.0, np.zeros(2), np.zeros(2), Q0 * np.eye(2), wavelength),
    lattice, z=0.0, voltage=voltage,
)
result, reports = wave.propagate(source, column, return_reports=True)
print(reports[0].kind, "branch; output pitch", float(result.lattice.pitch[0, 0]), "m; z =", result.z, "m")
```

The engine chose the branch and the output lattice: near an image plane the lattice is only rescaled,
far from it a matrix Fourier transform resamples. Its sampling planner checks the field's support and
bandwidth against the rules for each branch and raises `SamplingError` for a span that would alias,
listing the inequality that failed.

Two independent answers. The complex-$q$ law of a round Gaussian (drift $d$: $Q\to Q/(1+dQ)$ with
amplitude $1/(1+dQ)$; lens: $Q\to Q-1/f$) needs no Collins algebra:

```{code-cell} python
Q, amp = Q0, 1.0
for kind, value in (("drift", 3e-3), ("lens", 2e-3), ("drift", 5e-3)):
    if kind == "drift":
        amp, Q = amp / (1 + value * Q), Q / (1 + value * Q)
    else:
        Q = Q - 1 / value
expected = wave.gaussian_field(result.lattice.positions(), amp, np.zeros(2), np.zeros(2), Q * np.eye(2), wavelength)
print("relative difference to the complex-q law:", float(jnp.linalg.norm(result.data - expected) / jnp.linalg.norm(expected)))
```

The second is the package's own `FresnelPropagator`, which handles free space only and needs every plane
of the column sampled. A drift followed by a thin lens is the simplest column it can check: the FFT
propagator drifts the samples, and the lens then multiplies them by its phase. The Collins transform of
the whole column lands on the very same lattice, because the lens is the last element:

```{code-cell} python
from temgym_core.propagator import FresnelPropagator

drift = 0.5e-3
drift_then_lens = [Lens(z=drift, focal_length=2e-3)]
one_transform = wave.propagate(source, drift_then_lens)
fresnel = FresnelPropagator(source.data, 192 * 10e-9, wavelength, drift)
fresnel = fresnel * jnp.exp(-2j * np.pi * drift / wavelength)      # the propagator includes exp(ikz), the engine does not
x, y = lattice.coords()
fresnel = fresnel * jnp.exp(-1j * (2 * np.pi / wavelength) * (x**2 + y**2) / (2 * 2e-3))
print("same lattice:", bool(jnp.allclose(one_transform.lattice.pitch, lattice.pitch)))
print("Collins transform vs FresnelPropagator then lens:",
      float(jnp.linalg.norm(one_transform.data - fresnel) / jnp.linalg.norm(fresnel)))
```

The engine excludes the axial carrier $e^{ik(z-z_0)}$, as Gaussian beamlets do, so phases stay small even
over a metre.

## 2. A small object, against the beamlet sum

An aperture is not a linear element: it multiplies the field where it sits. In `propagate` it is a
*mask*, the exact action that a beamlet's Gaussian update expands to second order. Put a soft aperture
(300 nm radius, 80 nm edge) at the object plane of a lens that images it, and illuminate it with a
bundle of Gaussian beamlets, whose coherent sum is a flat beam. The same column then runs through both
representations:

```{code-cell} python
import types

from temgym_core.components import Detector, SigmoidAperture
from temgym_core.evaluate import evaluate_gaussians
from temgym_core.run import run_to_end
from temgym_core.source import rectangular_input_wave


def image_object(waist):
    detector = Detector(z=9e-3, pixel_size=(12e-9, 12e-9), shape=(192, 192))
    obj = [SigmoidAperture(z=0.0, radius=300e-9, edge_width=80e-9), Lens(z=3e-3, focal_length=2e-3), detector]
    beams = rectangular_input_wave(aperture_width=800e-9, aperture_height=800e-9, waist=waist,
                                   voltage=voltage, overlap_factor=2.0, z0=0.0)
    beamlet_sum = evaluate_gaussians(run_to_end(beams, obj), detector, method="scan")
    grid = wave.Lattice.from_grid(detector)
    lattice_in = wave.input_lattice(obj, grid, source_z=0.0, voltage=voltage)     # where the source must be sampled
    samples = types.SimpleNamespace(coords=lattice_in.positions().reshape(-1, 2), shape=lattice_in.shape)
    start = wave.Wavefront(evaluate_gaussians(beams, samples, method="scan"), lattice_in, 0.0, voltage)
    return wave.propagate(start, obj, grid, tol=1e-10), beamlet_sum


sampled, beamlets = image_object(waist=60e-9)
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2), constrained_layout=True)
for ax, image, title in zip(axes, (jnp.abs(sampled.data) ** 2, jnp.abs(beamlets) ** 2,
                                   jnp.abs(jnp.abs(sampled.data) ** 2 - jnp.abs(beamlets) ** 2)),
                            ("wave (Collins + mask)", "Gaussian beamlet sum", "difference")):
    ax.imshow(np.asarray(image), cmap="gray")
    ax.set(title=title, xticks=[], yticks=[])
plt.show()
```

The source was generated on `wave.input_lattice(...)`: the lattice that lands exactly on the detector
grid, adapted backwards from it through the column, so that nothing is interpolated. The sampled wave is
the reference here (an exact transform and an exact mask); the beamlet sum approximates the aperture
edge and improves as the beamlets shrink against it:

```{code-cell} python
for waist in (90e-9, 60e-9, 40e-9):
    sampled, beamlets = image_object(waist)
    difference = jnp.linalg.norm(jnp.abs(sampled.data) ** 2 - jnp.abs(beamlets) ** 2) / jnp.linalg.norm(jnp.abs(sampled.data) ** 2)
    print(f"beamlet waist {waist * 1e9:3.0f} nm: relative difference of the intensity {float(difference):.1e}")
```

## 3. An imported array

An exit wave that arrives as a Cartesian array is not on the engine's lattice, which is rotated by a
magnetic lens or a `Rotator` and scaled by the optics. `wave.resample_field` interpolates a field between
lattices with the exact band-limited (Fourier) interpolant, so a compact, resolved field survives the
trip to rounding. Here an elliptical object, "read from a file" as a plain array, goes through a
rotation, a lens and a drift to a plane near the back focal plane:

```{code-cell} python
from temgym_core.components import Rotator
from temgym_core.gaussian import make_gaussian

column = [Rotator(z=0.0, angle=25.0), Lens(z=3e-3, focal_length=2e-3), Plane(z=3e-3 + 1.03 * 2e-3)]
cartesian = wave.Lattice.regular((192, 192), 10e-9)
wx, wy = 100e-9, 60e-9
positions = cartesian.positions()
array = np.asarray(jnp.exp(-(positions[..., 0] / wx) ** 2 - (positions[..., 1] / wy) ** 2))    # the "file"

detector = Detector(z=0.0, pixel_size=(3e-9, 3e-9), shape=(128, 128))
sampled = wave.propagate(wave.Wavefront(array, cartesian, 0.0, voltage), column, tol=1e-14)
on_detector = wave.resample_field(sampled.data, sampled.lattice, detector)
print("engine lattice pitch:", np.asarray(sampled.lattice.pitch).round(12).tolist())
```

The propagated field lives on a tilted lattice (see the off-diagonal pitch). `resample_field` returns it on the
detector's Cartesian grid. The beamlet representation gives an independent check: one elliptical
Gaussian through the same column (its curvature matrix is rotated by the same components):

```{code-cell} python
beam = make_gaussian(z=0.0, voltage=voltage, waist_x=wx, waist_y=wy)
beamlet = evaluate_gaussians(run_to_end(beam, column), detector, method="scan")
print("relative difference between the resampled wave and the Gaussian:",
      float(jnp.linalg.norm(on_detector - beamlet) / jnp.linalg.norm(beamlet)))

fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2), constrained_layout=True)
x, y = sampled.lattice.coords()
axes[0].imshow(array, cmap="gray", extent=(-960, 960, -960, 960))
axes[0].set(title="imported array (nm)")
axes[1].pcolormesh(np.asarray(x) * 1e9, np.asarray(y) * 1e9, np.asarray(jnp.abs(sampled.data) ** 2), cmap="gray", shading="auto")
axes[1].set(title="engine lattice (nm)", aspect="equal")
axes[2].imshow(np.asarray(jnp.abs(on_detector) ** 2), cmap="gray")
axes[2].set(title="resampled to the detector", xticks=[], yticks=[])
plt.show()
```

Going the other way, `wave.input_lattice(column, detector, ...)` gives the lattice that lands on the
detector grid, and `wave.resample_field(array, cartesian, that_lattice)` puts the array on it, so only the
input is interpolated.

## 4. Differentiating

`column_transform`, `plan_lct` and `apply_lct` are ordinary JAX functions. The derivative of a
focused intensity with respect to the focal length, through the matrix, the plan and the transform,
checked against a finite difference:

```{code-cell} python
n = 64
out = wave.Lattice.regular((n, n), 12e-9)
distance = 4.6e-3
target = jnp.exp(-(out.positions() ** 2).sum(-1) / (2 * (0.1e-6) ** 2))


def loss(focal_length):
    transform = wave.column_transform([Lens(z=3e-3, focal_length=focal_length, x0=60e-9), Plane(z=3e-3 + distance)],
                                      0.0, voltage)
    lattice_in = wave.adapted_input_lattice(transform.matrix[:2, 2:4], wavelength, out, (1.0 / n, 1.0 / n), (n, n))
    field = wave.gaussian_field(lattice_in.positions(), 1.0, jnp.zeros(2), jnp.zeros(2), Q0 * jnp.eye(2), wavelength)
    plan = wave.plan_lct(transform.matrix, wavelength, lattice_in, out, method="mft", check="off")
    return jnp.real(jnp.sum(target * wave.apply_lct(plan, field, transform.phase0)))


f0, h = 2e-3, 1e-9
print("jax.grad:", float(jax.grad(loss)(f0)), "  finite difference:", float((loss(f0 + h) - loss(f0 - h)) / (2 * h)))
```

`plan_lct` takes an explicit `method` here because the branch is a static choice: under `jax.grad` and
`jax.jit` the matrix is a tracer, so the engine cannot look at its values.

```{note}
The engine runs on the GPU unchanged: a 4096 x 4096 transform takes 0.3 to 1.5 ms for the
non-matrix branches (`benchmarks/wave.py`). Energy spread is a `jax.vmap` over the voltage of
`column_transform`, the adapted lattice and `plan_lct`, one chief phase per wavelength.
```
