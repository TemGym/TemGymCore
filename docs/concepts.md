# Concepts and conventions

## One column, three representations

A **column** is a sequence of components ordered along $z$. The same components serve three
representations of a beam:

| Representation | Object | Use it for | Status |
|---|---|---|---|
| Ray | `Ray` | geometry, ABCD matrices, checking a column quickly | available |
| Gaussian beamlets | `GaussianBeam` | coherent images from $10^5$ to $10^6$ beamlets | available; gradients on the GPU kernels are planned |
| Sampled wavefront | complex array on a grid | Collins-integral propagation between planes, independent checks | planned |

`run_to_end` carries a ray or a bundle of beamlets from its current plane to each component in
turn. Between components it applies free-space propagation, and a finite component such as a
solenoid handles its own body.

## Rays and ABCD matrices

A `Ray` carries position $(x, y)$, slope $(\theta_x, \theta_y)$, axial position $z$, accumulated
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
- Images use (row, column) = (y, x) order. On a `Detector`, positive $x$ points right and
  positive $y$ points up.

## Numerical precision

Electron wavelengths are picometres and optical paths are metres, so $k\cdot\text{path}$ reaches
$10^{12}$ rad. Everything therefore runs in 64-bit arithmetic (`temgym_core.enable_x64()`).
Even so, float64 resolves such a phase only to about $2\times10^{-4}$ rad per beamlet, so the CPU
`scan` renderer and the GPU kernels agree to $10^{-5}$–$10^{-4}$ of the peak field, not to
machine precision. Compare intensities, or fields after removing the global phase.

## Known limitations

These are tracked in the changelog and will change in a future release.

- The GPU renderers are forward-only. Differentiate with `method="scan"` for small problems.
- In free space a ray's `pathlength` grows by the axial distance, while Gaussian propagation also
  adds the quadratic term $\tfrac12 d\,\theta^2$. Compare phases within one representation.
- The absolute intensity of `rectangular_input_wave` depends on the beamlet lattice (a few
  percent at hologram scale, up to tens of percent for coarse lattices). Relative structure such as
  fringes and edges is unaffected.
