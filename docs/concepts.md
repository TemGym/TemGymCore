# Concepts and conventions

## One column, three representations

A **column** is a sequence of components ordered along $z$. The same components serve three
representations of a beam:

| Representation | Object | Use it for | Status |
|---|---|---|---|
| Ray | `Ray` | geometry, ABCD matrices, checking a column quickly | available |
| Gaussian beamlets | `GaussianBeam` | coherent images from $10^5$ to $10^6$ beamlets | available, with first-order gradients through the GPU renderers |
| Sampled wavefront | complex array on a grid | Collins-integral propagation between planes, independent checks | planned |

`run_to_end` carries a ray or a bundle of beamlets from its current plane to each component in
turn. Between components it applies free-space propagation, and a finite component such as a
solenoid handles its own body.

## Rays and ABCD matrices

A `Ray` carries position $(x, y)$, slope $(\theta_x, \theta_y)$, axial position $z$, excess
optical path and accelerating voltage. `run.column_matrix` differentiates the native ray
propagation to give the $5\times5$ matrix of any span of components, acting on
$[x, y, \theta_x, \theta_y, 1]$; the trailing 1 lets it carry offsets. The matrix is exact for
linear optics and a local linearisation otherwise, so it is never derived by hand.

## Gaussian beamlets

A `GaussianBeam` adds a complex amplitude $A$ and a complex curvature matrix `Q_inv` to the
ray's centre, slope and path. Its field at a point is

$$
E(\mathbf r) = A\,\exp\!\Big\{ i k \big[\text{path} + \mathbf d\cdot\boldsymbol\theta
+ \tfrac12\,\mathbf d^{\mathsf T} Q^{-1}\mathbf d\big]\Big\},\qquad \mathbf d = \mathbf r - \mathbf r_0,
$$

with $k = 2\pi/\lambda$ for the beamlet's voltage. The imaginary part of `Q_inv` sets the width:
the intensity has spatial covariance $(2k\,\mathrm{Im}\,Q^{-1})^{-1}$. Thin elements (an aperture,
a lens, a biprism, a specimen) are `GaussianActionComponent`s: an optical path and a log amplitude
as functions of position, expanded to second order around each beamlet's centre.
`evaluate_gaussians` adds the beamlets' complex fields on a detector. Sum fields, never
intensities, unless the beamlets are mutually incoherent.

## Units and signs

- SI units: metres, radians, volts. Gaussian bundles can be expressed in another length unit
  through `wavelength_unit`; then keep every length in that unit.
- `pathlength` and `phase_shift` are **optical paths** in metres, not phases. The wave phase is
  $k\cdot\text{path}$.
- `pathlength` is the **excess** optical path: what a ray or beamlet accumulates beyond its
  straight-line axial travel from the launch plane. Free space adds $\tfrac12 d\,\theta^2$ (the
  same for rays and Gaussian beamlets), a thin element adds its optical path and kicks the slope
  by that path's gradient, so a ray parallel to the axis in free space accumulates none. The common
  carrier $k\,(z-z_0)$ is never accumulated: it is the same for every beamlet of one wavelength
  and drops out of every intensity.
- Images use (row, column) = (y, x) order. On a `Detector`, positive $x$ points right and
  positive $y$ points up.

## Gradients through the renderer

`evaluate_gaussians` has three renderers. `method="scan"` is plain JAX and supports every JAX
transformation. `method="sparse"` (the default on a GPU; alias `"spatial"`) and `method="dense"`
(alias `"gpu_kernel"`) are Pallas kernels with hand-written derivative rules. They are
differentiable to **first order** with `jit`, `grad`, `jvp`, `jacfwd`, `jacrev`, `vmap` and
`checkpoint`, with respect to every field of the beamlets (centre, slope, amplitude, path
length, `Q_inv`, voltage) and, by default, the detector's rotation, flip, pixel size and centre.
Second derivatives raise `NotImplementedError`; use `scan` for Hessians.

```python
from temgym_core.evaluate import evaluate_gaussians
from temgym_core.gaussian import make_gaussian

def loss(waist):                          # one waist per beamlet
    beam = make_gaussian(x=x, y=y, waist_x=waist, waist_y=waist, voltage=200e3)
    image = jnp.abs(evaluate_gaussians(beam, detector, method="sparse")) ** 2
    return jnp.sum((image - target) ** 2)

gradient = jax.grad(loss)(waist)          # d loss / d waist for every beamlet
```

**The gradient is that of the truncated sum.** `sparse` adds, for each beamlet, only the pixels
where its envelope $e^{-\delta}$ is at least the `cutoff` (default $10^{-12}$, set with
`RenderConfig(cutoff=...)`; $\delta$ is the quadratic form of $\mathrm{Im}\,Q^{-1}$), so the image
differs from the exact sum by at most $\text{cutoff}\times\sum|A|$. One test decides which pairs
are kept, and the forward, tangent and adjoint kernels all use it: the derivative is exactly that
of the truncated image, with the set of kept pixels held fixed, and it differs from the exact
sum's derivative by terms of the order of the cutoff. `dense` sums every pair and has no cutoff.

**Two coordinate modes.** The default `coordinates="index"` maps the beamlets into the
detector's pixel-index frame before rendering, so derivatives with respect to the detector pose
are ordinary automatic differentiation. The map re-rounds each packed parameter, which matters
only for extreme phases: for $10^{10}$ rad it moves the image by about $10^{-7}$ of the peak.
`RenderConfig(coordinates="metre")` renders with the metre-space parameters and pixel
coordinates exactly as the historical dense kernel did, accurate to the inputs; differentiating
with respect to the detector pose then raises `NotImplementedError` and points back to the
index mode.

**Beamlets that are not well behaved.** A beamlet with any non-finite field contributes zero and
receives a zero gradient, and never disturbs the others. A beamlet whose footprint contains no
pixel centre is never visited. Beamlets are binned on the device by the radius of their cutoff
footprint, so a few very broad beamlets among many narrow ones do not slow the rest; those
wider than the detector, or whose `Im Q_inv` is not positive definite, are summed in full by every
pixel. `describe_plan(beam, detector)` reports how a bundle is classified.

`RenderConfig` collects the options (cutoff, coordinate mode, radius classes, launch shapes) and
documents every field; pass it as `evaluate_gaussians(..., config=RenderConfig(...))`.

## Numerical precision

Electron wavelengths are picometres, so even the excess optical path of a column gives
$k\cdot\text{path}$ of order $10^{9}$ rad, and the phases inside the renderer's beamlet sums reach
$10^{10}$ rad. Everything therefore runs in 64-bit arithmetic (`temgym_core.enable_x64()`), which
resolves such phases to about $10^{-7}$ rad: on the reference hologram the CPU `scan` renderer and
the GPU kernels agree to about $10^{-8}$ of the peak field. Compare fields after removing the
global phase, or compare intensities.

## Known limitations

These are tracked in the changelog and will change in a future release.

- The GPU renderers are differentiable to first order only; Hessians need `method="scan"`.
