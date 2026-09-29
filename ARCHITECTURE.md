# Architecture

This page is for contributors. It says what the package is made of and which rules keep it
understandable. Items marked **planned** are on the road to 0.1.0 and are not yet in the code.

## One column, three representations

A microscope column is a sequence of `Component`s. The same components serve three
representations of a beam:

| Representation | Object | Role | Status |
|---|---|---|---|
| Ray | `Ray` | geometry, ABCD via `run.column_matrix`, Taylor maps | available |
| Gaussian beamlets | `GaussianBeam` | $10^5$–$10^6$ coherent beamlets, images via `evaluate_gaussians` | available; gradients through the fast GPU renderers are **planned** |
| Sampled wavefront | complex array + `Grid` + wavelength | Collins-integral propagation between planes, independent reference | **planned** |

Rules that tie them together:

1. A thin element is a `GaussianActionComponent`: an optical path and a log amplitude as
   functions of position. The same action updates a Gaussian and, in the sampled picture,
   multiplies the field.
2. A linear span of components is one 5×5 matrix (`run.column_matrix`), obtained by
   differentiating native ray propagation, never re-derived by hand.
3. Consistency is tested, not assumed: the chief ray of a Gaussian follows the ray, and a sum
   of Gaussians reproduces the sampled wave on the same column.

## Module map

| Responsibility | Modules |
|---|---|
| Rays, propagation | `ray`, `run`, `propagator`, `transfer_matrices`, `coordinate_transforms` |
| Components | `components` (planes, lenses, steering, actions, specimens) |
| Gaussian beamlets | `gaussian`, `source`, `evaluate`, `grid` |
| Microscope | `microscope` (named components, current tables, imaging condition) |
| Measurements and fitting | `moments`, `optimization`, `calibration/`, `reflection_calibration` |
| Plots | `plotting` (to be replaced by one `plot()` entry point) |
| Constants and aberrations | `constants`, `aberrations` |
| Experimental | `experimental/` |

## Conventions

See *Concepts and conventions* in the documentation: SI units, $[x, y, \theta_x, \theta_y, 1]$
ray vectors, `pathlength` and `phase_shift` as optical paths, (y, x) image order.

## Numerical policy

- Everything runs in float64. `jax_enable_x64` must be enabled by the caller.
- GPU renderers are checked against the portable `scan` renderer and against frozen baselines.
- Nothing global is changed on import.
