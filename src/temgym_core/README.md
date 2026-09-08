# Reading TemGymCore

Start with the [calibration notebook](../../examples/microscope_models/microscope_calibration_pipeline.ipynb).
Its public functions live in [calibration/workflow.py](calibration/workflow.py):
**generate images → measure images → fit components → check predictions**.

| Responsibility | Code |
|---|---|
| Optical elements, including lenses, solenoids and detector | `components.py` |
| Rays and Gaussian beams through those elements; affine 5×5 matrix | `ray.py`, `gaussian.py`, `source.py`, `propagator.py`, `run.py` |
| Gaussian fields and integrated detector images | `evaluate.py`, `spatial_evaluate.py`, `grid.py` |
| Image measurements, their matrix predictions and intensity initialization | `image.py` |
| Bounded JAX least squares and fits with shared parameters | `optimization.py` |
| Interpolated specimen component; nominal MAGICAL recipe | `components.py`; `calibration/gaussian.py` |
| Atomic-scattering physics used by `AtomicPotential` | `potential.py` |
| Microscope configuration, operating modes and plots | `microscope_model.py`, `imaging_modes.py`, `plotting.py` |

The remaining files supply constants, coordinate transforms, transfer matrices,
aberrations and array/tree utilities. The [calibration code map](calibration/README.md)
explains the experiment-specific code.

## Images and measurements

`evaluate.gaussian_image(beam, detector, ...)` uses the regular `Detector` from
`components.py`. Supply beams already propagated to its plane with `run_to_end`.
It sums complex Gaussian fields before forming intensity and integrating pixels.
There is no separate coherent detector or pixel-fitting workflow.

`image.py` measures image centroids and widths and predicts them from the
specimen wave and a transfer matrix. Its phase-gradient and amplitude-gradient
terms retain the wave contribution to those measurements. The fitter can
therefore compare predicted and measured quantities without rendering pixels
at every iteration. Its intensity approximation is used for initialization
and the optional faster synthetic acquisition.

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
separately when changing the scene. The component's `sampling_diagnostic` still
flags beams for which the local quadratic sample approximation is inadequate.

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

The core and calibration now contain 36 Python files. The sample cleanup removes
about 2,200 source lines. At both 200 and 195 kV, the fixed specimen maps, spline
coefficients, sampled action values/gradients/Hessians and Gaussian updates match
the previous implementation exactly. The calibration intensity template, sampled
transmission and predicted moments also match exactly.

115 targeted tests passed. The complete 256 × 256, 40,000× MAGICAL notebook ran
on the GPU: refining the Gaussian waist from 2 nm to 1 nm reduced total image
error from 0.920% to 0.256%, with propagation error below 0.0003%. The calibration
notebook's imports and public call signatures were checked; its full acquisition
and multiple-start fits were not rerun. Source fingerprints will invalidate old
workflow caches on the next run, as they do for other core changes.
