# Concepts and conventions

## One column, three representations

A **column** is a sequence of components ordered along $z$. The same components serve three
representations of a beam:

| Representation | Object | Use it for | Status |
|---|---|---|---|
| Ray | `Ray` | geometry, ABCD matrices, checking a column quickly | available |
| Gaussian beamlets | `GaussianBeam` | coherent images from $10^5$ to $10^6$ beamlets | available; gradients on the GPU kernels are planned |
| Sampled wavefront | `Wavefront`: complex array on a `Lattice` | Collins-integral propagation between planes, imported arrays, independent checks | available; experimental |

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

## Sampled wavefronts

A `Wavefront` (`temgym_core.wave`, experimental) is a complex field sampled on a `Lattice`: sample
positions $\mathbf r = \mathbf c + P\,\boldsymbol\mu$ with a full 2 x 2 pitch matrix $P$, so a lattice can be
rotated, scaled or sheared. `wave.propagate` runs it through the same column. Every linear stretch of
the column is one linear canonical (Collins) transform built from its $5\times5$ ray matrix, so the planes in
between are skipped. Two implementations of one operator cover the regimes: a matrix Fourier transform
away from an image plane, and a drift, lattice map and chirp near one (the lattice is only rescaled or
rotated, never interpolated). A thin element that is not linear, such as an aperture or a biprism,
is a multiplicative mask: exactly the action its Gaussian branch applies, without the second-order
expansion. Elements whose action depends on the ray slope (`SeidelLens`) are refused.

Three things follow from sampling. The engine **owns the lattices**: rotation and astigmatism change the
output lattice, so a source is generated on `wave.input_lattice(...)`, the lattice adapted backwards from
the detector grid, and an array that arrives on a Cartesian grid goes through `wave.resample_field`. A
**sampling planner** checks each span against the field's support and bandwidth (a span that would alias
raises `SamplingError` naming the inequality that failed; `check="off"` is the expert override). And the
**phase** of a span, which its matrix fixes only up to a sign, comes from a chief Gaussian propagated
through the same components, so the Gouy phase is continuous across a whole column. The field excludes
the axial carrier, like a beamlet's. The sampled wave is the reference for the beamlet sum, which
approximates thin elements to second order about each beamlet's centre. See the
[tutorial](tutorials/skip_planes.md).

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

## Numerical precision

Electron wavelengths are picometres, so even the excess optical path of a column gives
$k\cdot\text{path}$ of order $10^{9}$ rad, and the phases inside the renderer's beamlet sums reach
$10^{10}$ rad. Everything therefore runs in 64-bit arithmetic (`temgym_core.enable_x64()`), which
resolves such phases to about $10^{-7}$ rad: on the reference hologram the CPU `scan` renderer and
the GPU kernels agree to about $10^{-8}$ of the peak field. Compare fields after removing the
global phase, or compare intensities.

## Known limitations

These are tracked in the changelog and will change in a future release.

- The GPU renderers are forward-only. Differentiate with `method="scan"` for small problems.
