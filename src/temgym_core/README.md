# Reading TemGymCore

Start with the [calibration notebook](../../examples/microscope_models/microscope_calibration_pipeline.ipynb).
Its public functions live in [calibration/workflow.py](calibration/workflow.py):
**generate images → measure images → fit components → check predictions**.

| Responsibility | Code |
|---|---|
| Optical elements, including lenses, solenoids and detector | `components.py` |
| Rays and Gaussian beams through those elements; affine 5×5 matrix | `ray.py`, `gaussian.py`, `source.py`, `propagator.py`, `run.py` |
| Gaussian fields and integrated detector images | `evaluate.py`, `grid.py` |
| Image measurements and their matrix predictions | `moments.py` |
| Bounded JAX least squares and fits with shared parameters | `optimization.py` |
| Interpolated specimen component; nominal MAGICAL recipe | `components.py`; `calibration/gaussian.py` |
| Analytical matrices and transport through supplied matrices | `transfer_matrices.py` |
| Experimental atomic potentials and `AtomicPotential` component | `experimental/potential.py` |
| Microscope configuration, operating modes and plots | `microscope_model.py`, `imaging_modes.py`, `plotting.py` |

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

The magnetic cube recipe and its projection helper live beside their notebooks in
[holography_distortion/magnetic_cube.py](../../examples/holography_distortion/magnetic_cube.py).
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
The notebook that calls it moved to
[examples/experimental/glauber_ray_example.ipynb](../../examples/experimental/glauber_ray_example.ipynb).

## Images and measurements

`evaluate.gaussian_image(beam, detector, ...)` uses the regular `Detector` from
`components.py`. Supply beams already propagated to its plane with `run_to_end`.
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

The renderer consolidation passed 37 CPU tests and 24 explicitly enabled GPU
tests. Before/after complex fields matched exactly for scan, loop, dense GPU
and spatial GPU evaluation on the comparison bundle. The complete 40,000×
MAGICAL notebook also passed on GPU with unchanged image-error results.

`moments.py` replaces `image.py`. Read its first two functions together:

- `measure_detector_moments(images=..., background_counts=..., read_noise_counts=...)`
  returns five measurements per image—centre x/y, variance x, covariance xy,
  variance y—and their measurement uncertainty.
- `predict_detector_moments(matrix, mean, covariance, pixel_pitch_m=..., psf_sigma_pixels=...)`
  predicts those same quantities from a 5×5 matrix and the specimen's 4D beam state.

`SampledTransmission` integrates the known specimen to get that beam state,
including phase and amplitude gradients. The last two functions handle the
pixel-sampling correction used for narrow condenser images. The fit compares
these predictions with measurements without rendering pixels at every iteration.

Single-beam moments belong to `GaussianBeam.moments()`; mixture propagation
lives in `run.py`. Aperture transmission stays in `calibration/apertures.py`.
The optional intensity renderer is in `evaluate.py`, its initialization formula
in `calibration/condenser_images.py`, and exposure/background fitting in
`optimization.py`. There is no extra image module or Gaussian-spot sample class.

This cleanup passed 36 targeted tests. Before/after checks matched exactly for
measurements, specimen moments, rendering initialization, Gaussian propagation,
aperture flux and the tested specimen, pixel-correction and aperture Jacobians.
The notebook's nine public calls were checked; full acquisition and fitting
were not repeated.

`InterpolatedSample2D.from_action_maps(...)` constructs a regular component
from optical path in metres, log field amplitude, and x/y pixel-centre arrays.
Maps use image (y, x) order. The C2 cubic spline, spatial derivatives, specimen
pose, periodic boundaries and Gaussian update are retained from `SampleView`.
The existing complex-array constructor, `from_array`, keeps its original behaviour.

`calibration.gaussian.make_magical_sample()` returns that component, known
feature dimensions and a calibration centre. Both the calibration and the 40k
MAGICAL imaging notebook use this one synthetic recipe. It makes the same fixed
specimen map as before, without building unused regional maps or pyramids.
The old atlas, certificates and automatic field selection have been removed;
Git revision `5aa2ae7` retains them. This fixed map is an overview target, not an
atomic-resolution model. Refine specimen sampling and the Gaussian approximation
separately when changing the scene.

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
building one large dense Jacobian. These are numerical helpers within the
same notebook procedure.

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
cleanup and `31adfc3` precedes this consolidation. The current calibration
notebook's public calls are unchanged.

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
