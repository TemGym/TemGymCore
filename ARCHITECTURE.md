# Architecture

This page is for contributors. It says what the package is made of and which rules keep it
understandable. Items marked **planned** are on the road to 0.1.0 and are not yet in the code.

## One column, three representations

A microscope column is a sequence of `Component`s. The same components serve three
representations of a beam:

| Representation | Object | Role | Status |
|---|---|---|---|
| Ray | `Ray` | geometry, ABCD via `run.column_matrix`, Taylor maps | available |
| Gaussian beamlets | `GaussianBeam` | $10^5$–$10^6$ coherent beamlets, images via `evaluate_gaussians` | available; first-order gradients through the GPU renderers |
| Sampled wavefront | `Wavefront`: complex array + `Lattice` + voltage | Collins-integral propagation between planes, imported arrays, independent reference | available; **experimental** |

Rules that tie them together:

1. A thin element is a `GaussianActionComponent`: an optical path and a log amplitude as
   functions of position. The same action updates a Gaussian and, in the sampled picture,
   multiplies the field.
2. A linear span of components is one 5×5 matrix (`run.column_matrix`), obtained by
   differentiating native ray propagation, never re-derived by hand.
3. Consistency is tested, not assumed: the chief ray of a Gaussian follows the ray, and a sum
   of Gaussians reproduces the sampled wave on the same column.
4. In the sampled picture a linear span is one Collins transform planned from its ray matrix, a thin
   non-linear element is a mask, and the physical phase of a span comes from a chief Gaussian
   propagated through the same components. The engine owns the lattices; arrays are resampled onto
   them.

## Module map

| Responsibility | Modules |
|---|---|
| Rays, propagation | `ray`, `run`, `propagator`, `transfer_matrices`, `coordinate_transforms` |
| Components | `components/` (`planes`, `lenses`, `deflection`, `actions`, `specimens`, `interpolated`) |
| Gaussian beamlets | `gaussian`, `source`, `evaluate` (public dispatcher), `render/` (GPU renderer), `grid` |
| Sampled wavefronts (experimental) | `wave/`: `lattice` (affine sample lattices, backward lattice chain), `lct` (plan and apply the Collins transform), `planner` (sampling rules), `masks` (thin elements as multipliers), `column` (`propagate`, `column_transform`), `resample_field`, `reference` (analytic Gaussians, chief phase) |
| Microscope | `microscope` (named components, current tables, imaging condition) |
| Measurements and fitting | `moments`, `optimization`, `calibration/`, `reflection_calibration` |
| Virtual PyJEM-driven microscope | `virtual/` (may import `calibration/`; never the reverse) |
| Plots | `plotting` (to be replaced by one `plot()` entry point) |
| Constants and aberrations | `constants`, `aberrations` |
| Experimental | `experimental/` |

## The GPU renderer

`evaluate.evaluate_gaussians` is the dispatcher (`auto`, `scan`, `sparse`/`spatial`,
`dense`/`gpu_kernel`, `gpu_kernel_required`, `loop`); `render/` is what it dispatches to.

| Module | Role |
|---|---|
| `render/config.py` | `RenderConfig` (user options), `Tiles`, `KernelSpec` (static primitive parameter) |
| `render/packing.py` | the packed `(N, 12)` float64 bundle, invalid-beamlet policy, pixel-index map |
| `render/predicate.py` | the one pair function and truncation predicate every kernel uses |
| `render/coefficients.py` | chain rule from packed tangents to polynomial coefficients and back |
| `render/plan.py` | device-side radius classes, bins, windows and adjoint work items |
| `render/primitives.py` | forward, tangent and adjoint primitives, their JAX rules, `evaluate_packed` |
| `render/kernels/` | `reference` (pure JAX), `dense` and `sparse` (Pallas/Triton), shared `common` |

Rules that keep the renderer correct and differentiable:

1. **One predicate.** The renderer is the truncated sum over pairs with `delta <= -ln(cutoff)`.
   Forward, tangent and adjoint kernels evaluate it through `predicate.pair_terms`; bins, windows
   and tiles only enumerate supersets. Derivatives are exactly those of the truncated sum.
2. **Two linear kernels, each the transpose of the other.** `tangent` (`K`) and `adjoint` (`K^T`)
   are primitives linear in their last operand; JAX derives `jvp`, `vjp`, `jacfwd`, `jacrev`,
   `vmap` and `checkpoint` from that, and the elementwise chain rule around them is ordinary JAX.
3. **Plan on the device, static shapes.** Classification, binning and work items are jittable and
   never synchronise with the host; a bundle's off-detector beamlets cost only the sort.
4. **Deterministic.** No atomics: every kernel is bitwise repeatable for a fixed configuration.
5. **First order only.** Second derivatives raise a clear error; `scan` is the Hessian path.

## Conventions

See *Concepts and conventions* in the documentation: SI units, $[x, y, \theta_x, \theta_y, 1]$
ray vectors, `pathlength` (the excess optical path) and `phase_shift` as optical paths, (y, x) image order.

## Numerical policy

- Everything runs in float64. `jax_enable_x64` must be enabled by the caller.
- GPU renderers are checked against the portable `scan` renderer, against a frozen copy of the
  historical dense kernel and against frozen baselines; their gradients are checked against
  automatic differentiation of `scan` and by `jax.test_util.check_grads`.
- Nothing global is changed on import.
