# Concepts and conventions

## One column, three representations

A **column** is a sequence of components ordered along $z$. A `Ray` carries position,
slope, axial position $z$, accumulated optical path and accelerating voltage. A
`GaussianBeam` additionally carries a complex amplitude and the complex curvature matrix
`Q_inv`. Each component knows how to transform both, and `run_to_end` moves a ray or
beamlet bundle from its current plane to each component in turn.

## Units and signs

- SI units: metres, radians, volts. Gaussian bundles can be expressed in another
  length unit through `wavelength_unit`; then keep every length in that unit.
- Ray coordinates are ordered $[x, y, \theta_x, \theta_y, 1]$; the trailing 1 lets
  matrices carry offsets (`run.column_matrix`).
- `pathlength` and `phase_shift` are **optical paths** in metres, not phases. The wave
  phase is $k\cdot\text{path}$ with $k = 2\pi/\lambda$ for the ray's voltage.
- Images use (row, column) = (y, x) order; on a `Detector` positive $x$ points right and
  positive $y$ points up.
