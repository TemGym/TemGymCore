# Reading TemGymCore

The microscope work starts at
[examples/microscope_models](../../examples/microscope_models/README.md): the stage-1 projector
calibration (a PyJEM session script, its analysis and fit) rehearsed on the
[virtual microscope](virtual/README.md). Both use the TemGym projector and condenser models in
[calibration/](calibration/README.md).

| Responsibility | Code |
|---|---|
| Optical elements, including lenses, solenoids and detector | `components/` |
| Rays and Gaussian beams through those elements; affine 5×5 matrix | `ray.py`, `gaussian.py`, `source.py`, `propagator.py`, `run.py` |
| Gaussian fields and integrated detector images | `evaluate.py`, `grid.py` |
| Image measurements and their matrix predictions | `moments.py` |
| Bounded JAX least squares and fits with shared parameters | `optimization.py` |
| Interpolated and Fourier Bragg specimens | `components/specimens.py`, `components/interpolated.py` |
| Analytical matrices and transport through supplied matrices | `transfer_matrices.py` |
| Experimental atomic potentials and `AtomicPotential` component | `experimental/potential.py` |
| Microscope configuration, operating modes and plots | `microscope.py`, `plotting.py` |
| TEM column models for calibration (projector, condenser, ray-traced setting) | `calibration/` |
| Virtual PyJEM-driven microscope | `virtual/` |

`microscope.py` holds a named collection of native components, current tables in
amperes, and the explicit `find_imaging_condition` design solve.
The solve differentiates native ray propagation and uses the shared bounded
JAX least-squares solver. `run.column_matrix` returns the local Jacobian and
affine offset; it is also differentiable with respect to component parameters.
It replaces `microscope_model.py` and `imaging_modes.py`; no compatibility layer
or second component configuration hierarchy remains. The calibration workflow
uses its own parameter-to-component builders and does not depend on this container.

`components.BraggSample.from_unit_cell` represents an infinite periodic thin
specimen by selected Fourier orders of its complex transmission. Each incident
Gaussian creates one child per order, preserving the phase at its centre. It
uses the ordinary `run_to_end` and detector renderer; pass a whole bundle
directly rather than wrapping it in another `vmap`. A Bragg example (a sinusoidal
phase grating with analytic intensity and flux checks) was removed in the 2026-11-06 cleanup; Git revision `b2d873e` retains it. Unit-cell samples use
image (y, x) order without repeated endpoints, and all lengths use the beam's
coordinate unit. Refine the input grid and retained orders separately; omitted
orders are not renormalized. The transmission must be supplied for the chosen
beam energy. This component does not calculate dynamical crystal scattering.

The remaining files supply constants, coordinate transforms, aberrations and
array/tree utilities. The [calibration code map](calibration/README.md)
explains the experiment-specific code.

The former `utils.py` helpers now live with their callers: source sampling in
`source.py`, detector accumulation and reshaping in `grid.py`, Ray Jacobian
conversion in `run.py`, and field phase normalization in `evaluate.py`.
The two 2D Fresnel entry points share one kernel in `propagator.py`, retaining
their existing NumPy/JAX interfaces and axial phase conventions. Angular-spectrum
propagation lives there too. These sampled-field routines also provide independent
checks of Gaussian propagation.

The magnetic cube recipe and its projection helper (`holography_distortion/magnetic_cube.py`)
were removed in the 2026-11-06 cleanup; Git revision `b2d873e` retains it.
The unused 1D Fresnel and action-centre helpers were removed; the circular
comparison mask and thin-lens FFT reference live in `tests/test_gaussians.py`.
This cleanup passed 108 targeted tests with two existing skips. Before/after
checks matched exactly for sampled coordinates, weights, detector accumulation,
Ray Jacobians, tested FFT fields and derivatives, and the magnetic cube outputs.
Notebook imports were checked; full notebook simulations were not repeated.

`transfer_matrices.py` defines each matrix once, with a small conversion helper
for SymPy, NumPy and JAX. The former `transfer.py` helpers live here too.
Cumulative maps follow travel order: `M0`, `M1 @ M0`, `M2 @ M1 @ M0`.
This corrects the old helper's reversed order; the calibration workflow uses
`run.column_matrix` and did not use that helper. The unused duplicate in
`tests/transfer_matrices.py` has been removed.

Atomic-scattering work is now imported explicitly from
`temgym_core.experimental.potential`. Its Gaussian component still uses the
regular TemGym interface, but importing ordinary components does not load it.
The notebook that called it (`glauber_ray_example.ipynb`) was removed in the 2026-11-06 cleanup; Git revision `b2d873e` retains it.

## Reading components

`components/` is one module per kind of element: `_base.py` (dispatch and the shared
local action), `planes.py`, `lenses.py`, `deflection.py`, `actions.py` (apertures and
prescribed phase/amplitude), `specimens.py` and `interpolated.py`. Everything is
re-exported from `temgym_core.components`. Read `Component` for dispatch and
`Lens` for the simplest physical example. `run_to_end` handles travel to each
component; finite elements such as `HardEdgeSolenoid` handle their own body.

Each component supplies a ray update and/or a Gaussian update. Where the
calculation is identical, both use the same method. Gaussian-only elements
inherit the base class's error for unsupported ray input. `ElectromagneticLens`
shares one thin/thick focusing sequence, retaining the separate ray and
Gaussian propagators. `GaussianActionComponent` and `InterpolatedSample2D`
share one absorption and curvature update; they differ in how they obtain
the action derivatives.

`phase_shift` is a historical name for **optical path**, not phase in radians.
The phase is `k * path`; `log_transmission` describes field amplitude.
`ABCDTransfer` transports coordinates and curvature only. Physical wrappers
must supply optical path and coherent amplitude, as `HardEdgeSolenoid` does.

The audit also identified existing behaviour needing separate physics work:

| Component | Current limitation |
|---|---|
| `ElectromagneticLens` | The round-ray branch ignores x0/y0; Gaussian and stigmator branches use them. |
| `InterpolatedFields3D` | Fields can be interpolated, but calling the component leaves the beam unchanged. |

The readability changes preserve those behaviours. They also preserve the
separately edited Seidel and distorted-lens formulas.
132 targeted tests passed for both the working version and an isolated cleanup
snapshot. All 77 before/after comparisons matched exactly, including the tested
action, specimen and electromagnetic-lens derivatives. Full notebook fits were
not repeated.

## On-axis ray aberration jets (not yet implemented)

Two pieces described in earlier notes are **not in the code** and are tracked in the
roadmap: a polynomial `KrivanekLensCartesian` (its tests in
`tests/test_lens_ray_jets.py` are skipped) and ray support for `SeidelLens` at singular
rays (`SeidelLens` raises `NotImplementedError` for ray input; those tests are
expected failures). Until they land, use `KrivanekLens` for Gaussian beams and the
native Gaussian action of `SeidelLens`.

The Cartesian helper functions `W_krivanek_cartesian` and `grad_W_krivanek_cartesian`
in `aberrations.py` exist and agree with the polar forms.

## Images and measurements

`evaluate.gaussian_image(beam, detector, ...)` uses the regular `Detector` from
`components/planes.py`. Supply beams already propagated to its plane with `run_to_end`.
It sums complex Gaussian fields before forming intensity and integrating pixels.
There is no separate coherent detector or pixel-fitting workflow.

All rendering backends now live in `evaluate.py`. Use the same
`evaluate_gaussians(beam, grid, method=...)` entry point:

| Method | Purpose |
|---|---|
| `scan` | Portable coherent sum with JAX derivatives |
| `gpu_kernel` | Full coherent sum on GPU; scan fallback on CPU |
| `spatial` | GPU neighbour bins with a field-tail cutoff; dense fallback for broad beams |
| `loop` | Small independent loop for numerical checks |

`auto` selects the dense GPU kernel or CPU scan. `gpu_kernel_required` requires
a verified GPU backend. The GPU paths are for forward rendering; choose `scan`
for differentiation. Spatial bins are rebuilt on every call, and interference
is retained. The removed Numba/CuPy renderer was replaced in the two diffraction
examples by the standard entry point. Public dense-kernel wrappers used by the
other examples remain. Gaussian batching is supplied by `GaussianBeam.to_vector()`.

`source.decompose_gaussian(source, num_across=768, through=upstream)` returns
589,824 coherent beamlets at the original source plane. The optional lossless
round column ends just before an aperture: it chooses a basis narrow there,
without changing the physical source as the count increases. Propagate that
bundle through the actual aperture and remaining lenses, then sum its complex
field with `evaluate_gaussians(..., method="spatial")`. Refine the sampling to
check the aperture approximation separately from the renderer's tail cutoff.

The renderer consolidation passed 37 CPU tests and 24 explicitly enabled GPU
tests. Before/after complex fields matched exactly for scan, loop, dense GPU
and spatial GPU evaluation on the comparison bundle. The complete 40,000×
MAGICAL notebook also passed on GPU with unchanged image-error results.

`moments.py` replaces `image.py`. Read its first two functions together:

- `measure_detector_moments(images=..., background_counts=..., read_noise_counts=...)`
  returns five moments per image—centre x/y, variance x, covariance xy,
  variance y—plus total signal and their joint measurement uncertainty. Coordinates come from the native
  `Detector`: positive x points rightward and positive y upward.
- `predict_detector_moments(matrix, mean, covariance, pixel_pitch_m=..., psf_sigma_pixels=...)`
  predicts those same quantities from a 5×5 matrix and the specimen's 4D beam state.

`SampledTransmission` integrates a transmission component to get that beam state,
including phase and amplitude gradients. Its `from_component` constructor accepts
quadrature points and weights; the beam-calibration fitter uses it for the native
sigmoid aperture. The fit compares these predictions with measurements without
rendering pixels at every iteration. Pixel-sampling helpers remain available for
other measurements.

Single-beam moments belong to `GaussianBeam.moments()`; mixture propagation
lives in `run.py`. Aperture transmission stays in `calibration/apertures.py`.
The generic intensity renderer remains in `evaluate.py`; calibration uses
coherent Gaussian images only. Exposure/background fitting lives in
`optimization.py`. There is no calibration-specific intensity renderer.

`InterpolatedSample2D.from_action_maps(...)` constructs a regular component
from optical path in metres, log field amplitude, and x/y pixel-centre arrays.
Maps use image (y, x) order. The C2 cubic spline, spatial derivatives, specimen
pose, periodic boundaries and Gaussian update are retained from `SampleView`.
The existing complex-array constructor, `from_array`, keeps its original behaviour.

Gaussian approximation diagnostics now live only in `tests/gaussian_sampling.py`.
One helper compares explicit optical-path and log-amplitude functions with their
local quadratic expansions at eight points around each beam. Tests cover smooth
lenses, aperture edges and unresolved sample structure. It does not modify beams,
guess component arguments or prescribe new widths. The 40k notebook's Gaussian
refinement remains the check of resulting image error. The former production
`gaussian_corrections.py` module and sample `sampling_diagnostic()` method have
been removed; Git revision `822cb5c` retains the old API. The simplified test
helper is 69 lines; 29 targeted sampling, sample and coherent-image tests passed.

## Why keep local optimization code?

`optimization.least_squares` handles finite bounds, parameter scaling and
supplied JAX Jacobians. `fit_shared_least_squares` fits per-image beam parameters
alongside a shared projector calibration, retaining their correlations without
building one large dense Jacobian. The active beam-calibration procedure uses `least_squares` directly;
the shared-fit helper remains available for other applications.

[Optimistix](https://docs.kidger.site/optimistix/api/least_squares/) was tested
with Levenberg–Marquardt and sigmoid bounds on the recorded 200/195 kV Gaussian
experiment. The current bounded solver is retained because the replacement
lost useful solutions from the existing starting guesses.

| Recorded starting guess | Current solver χ² | Optimistix + sigmoid χ² |
|---|---:|---:|
| Projector 1 | 766.2734 | 387,426.7 |
| Condenser 1 | 638.3346 | 638.3346 |
| Condenser 4 | 638.3346 | 4.325 × 10⁹ |

The successful condenser comparison differed by at most 1.6e-6 of each
parameter's allowed interval. The failed projector solve reported package
convergence, but its physical scaled gradient was 0.239 (target 1e-5), so it
was rejected. Several starts lie within 1e-9 of a bound in normalized
coordinates; sigmoid saturation can suppress their physical sensitivity.
Capping transform scaling and using interval-based scaling also failed the
difficult condenser start. Moving both solvers' starts 0.1% inside the bounds
changed the current solver's basin too, so this is not a reliable simple fix.

These are local fits to existing measurements, using Optimistix 0.1.0 and
JAX 0.10.2 on CPU with float64. The first projector prototype guarded sigmoid
fractions at 1e-6 from the endpoints. Additional projector trials were stopped
and are not used as evidence. Images were not regenerated, the full joint
refinement was not repeated, and this is not a GPU performance comparison.
A package replacement still needs reliable bound handling or a separately
validated geometry parameterization. The exploratory code stays outside the
maintained source tree.

Older APIs and experiments remain in Git: `fc992bc` precedes the first core
cleanup and `31adfc3` precedes this consolidation. The moment-based beam-image calibration
(workflow, inference, checks) was removed in the 2026-11-06 cleanup; Git revision `b2d873e` retains it.

At the sample-cleanup checkpoint, core and calibration contained 34 Python files.
That cleanup removed about 2,200 source lines. At both 200 and 195 kV, the fixed specimen maps, spline
coefficients, sampled action values/gradients/Hessians and Gaussian updates match
the previous implementation exactly. The calibration intensity template, sampled
transmission and predicted moments also match exactly.

115 targeted tests passed. The complete 256 × 256, 40,000× MAGICAL notebook ran
on the GPU: refining the Gaussian waist from 2 nm to 1 nm reduced total image
error from 0.920% to 0.256%, with propagation error below 0.0003%. The calibration
notebook's imports and public call signatures were checked; its full acquisition
and multiple-start fits were not rerun. Source fingerprints will invalidate old
workflow caches on the next run, as they do for other core changes.

### Reflection geometry and angular Gaussian bases

`coordinate_transforms.PlaneFrame` describes a surface or detector by a 3D origin and orthonormal in-plane axes. It provides local/world transforms and differentiable line-plane intersection. Native column components continue to use their usual axial coordinates; a change to a reflected propagation frame is explicit.

`source.decompose_gaussian_angular` reconstructs a coherent Gaussian with wider, tilted packets. Refining the angular quadrature leaves packet widths fixed. The optional `through` column chooses the sampling plane while returning a source-plane bundle for native propagation.

`plotting.plot_reflection_column` combines native Gaussian basis traces with caller-supplied physical reflected paths and detector endpoints. See [the SEM reflection notebook](../../examples/sem_reflection/sem_reflection.ipynb) for the experimental kinematic scattering adapter, its assumptions and verification.
