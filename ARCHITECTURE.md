# Architecture

This page is for contributors. It says what the package is made of and which rules keep it
understandable. Items marked **planned** are on the road to 0.1.0 and are not yet in the code.

## One column, two representations

A microscope column is a sequence of `Component`s. The same components serve two
representations of a beam:

| Representation | Object | Role | Status |
|---|---|---|---|
| Ray | `Ray` | geometry, ABCD via `run.column_matrix`, Taylor maps | available |
| Gaussian beamlets | `GaussianBeam` | coherent beamlets, images via `evaluate_gaussians` | available; plain JAX, every derivative |

Rules that tie them together:

1. A thin element is a `GaussianActionComponent`: an optical path and a log amplitude as
   functions of position. The same action updates a Gaussian and weights a ray.
2. A linear span of components is one 5×5 matrix (`run.column_matrix`), obtained by
   differentiating native ray propagation, never re-derived by hand.
3. Consistency is tested, not assumed: the chief ray of a Gaussian follows the ray.

## Module map

| Responsibility | Modules |
|---|---|
| Rays, propagation | `ray`, `run`, `propagator`, `transfer_matrices`, `coordinate_transforms` |
| Components | `components/` (`planes`, `lenses`, `deflection`, `actions`, `specimens`, `interpolated`) |
| Gaussian beamlets | `gaussian`, `source`, `evaluate`, `grid` |
| Simulated microscope behind PyJEM | `pyjem_simulator/` (takes any column of components as its hidden optics; nothing in the package imports it) |
| Plots | `plotting` (to be replaced by one `plot()` entry point) |
| Constants and aberrations | `constants`, `aberrations` |

## Conventions

See *Concepts and conventions* in the documentation: SI units, $[x, y, \theta_x, \theta_y, 1]$
ray vectors, `pathlength` (the excess optical path) and `phase_shift` as optical paths, (y, x) image order.

## Numerical policy

- Everything runs in float64. `jax_enable_x64` must be enabled by the caller.
- Nothing global is changed on import.
