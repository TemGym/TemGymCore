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
| Known MAGICAL specimen and cubic interpolation | `multiscale_sample.py` |
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

The multiscale sample is used: it creates the MAGICAL transmission and its
spatial derivatives. The calibration uses a fixed sampled specimen; it does
not choose resolution levels during optimization. The separate MAGICAL imaging
example also uses the atlas at different magnifications. The old NEOARM TOML
loader has been removed; use `nominal` or `from_certificate` directly.

## Why keep local optimization code?

`optimization.least_squares` handles finite bounds, parameter scaling and
supplied JAX Jacobians. `fit_shared_least_squares` fits per-image beam parameters
alongside a shared projector calibration, retaining their correlations without
building one large dense Jacobian. These are numerical helpers within the
same notebook procedure.

[JAXopt](https://github.com/google/jaxopt) is no longer maintained.
[Optimistix's standard least-squares API](https://docs.kidger.site/optimistix/api/least_squares/)
does not directly expose our box bounds and supplied block-scheduled Jacobian.
Replacing these routines remains possible, but requires adapting and validating
the solver behavior. This cleanup preserves the algorithms and consolidates
them into one module; it does not claim a package replacement was tested.

Older APIs and experiments remain in Git: `fc992bc` precedes the first core
cleanup and `31adfc3` precedes this consolidation. The current calibration
notebook's public calls are unchanged.

This consolidation reduces maintained Python from 41 to 37 files (core plus
calibration). 105 distinct targeted tests pass. Gaussian images and derivatives
match the previous detector implementation to floating-point precision; MAGICAL
sample values and gradients are unchanged. The wheel imports and renders outside
the repository. Full acquisition and multiple-start fitting were not repeated.
