# Reading TemGymCore

The microscope work starts at
[examples/microscope_inversion](../../examples/microscope_inversion/stage1_projection_response.ipynb): the projection-system
calibration (a PyJEM session script, and a notebook with the lens model, the fit and the figures)
rehearsed on the [simulated microscope](pyjem_simulator/README.md). The lens model is part of that
experiment, built from TemGym components. [ARCHITECTURE.md](../../ARCHITECTURE.md) has the rules
that keep the package understandable.

| Responsibility | Code |
|---|---|
| Optical elements: lenses, solenoids, deflectors, apertures, specimens, detector | `components/` |
| Rays and Gaussian beams through those elements; the 5×5 ray map and A, B (`run.position_and_angle_response`) | `ray.py`, `gaussian.py`, `source.py`, `propagator.py`, `run.py` |
| Gaussian fields and camera images | `evaluate.py`, `grid.py` |
| Analytical matrices, for tests and checks | `transfer_matrices.py` |
| Simulated microscope behind PyJEM (takes any TemGym column as its hidden optics) | `pyjem_simulator/` |
| Plots | `plotting.py` |
| Constants, coordinate transforms, aberrations, configuration, tree utilities | `constants.py`, `coordinate_transforms.py`, `aberrations.py`, `config.py`, `tree_utils.py` |

## Reading components

`components/` is one module per kind of element: `_base.py` (dispatch and the shared
local action), `planes.py`, `lenses.py`, `deflection.py`, `actions.py` (apertures and
prescribed phase/amplitude), `specimens.py` and `interpolated.py`. Everything is
re-exported from `temgym_core.components`. Read `Component` for dispatch and
`Lens` for the simplest physical example. `run_to_end` handles travel to each
component; finite elements such as `HardEdgeSolenoid` handle their own body.

Each component supplies a ray update and/or a Gaussian update. Where the calculation is
identical, both use the same method. `phase_shift` is a historical name for **optical path**,
not phase in radians: the phase is `k * path`, and `log_transmission` describes field
amplitude. `ABCDTransfer` transports coordinates and curvature only.

Known limitations: `ElectromagneticLens`'s round-ray branch ignores x0/y0;
`InterpolatedFields3D` interpolates fields but leaves the beam unchanged; `SeidelLens` has no
ray update, and the polynomial `KrivanekLensCartesian` is not implemented (its tests are
skipped).

## Images

`evaluate.evaluate_gaussians(beam, grid)` sums a bundle's complex fields on a grid: a `lax.scan`
over beamlets by default; `method="gpu"`, a dense Pallas kernel (fast on a GPU, interpreted
elsewhere, no gradients); `method="loop"`, a plain reference loop. `evaluate.gaussian_image(beam,
detector, ...)` integrates the coherent intensity over a `Detector`'s pixels, and
`gaussian_intensity_image` is the fast incoherent alternative (independent round spots). Supply
beams already propagated to the detector plane with `run_to_end`. Apart from the Pallas kernel, all
of it is plain JAX: CPU or GPU, every derivative.

`evaluate.ray_image(rays, components, detector, ...)` is the ray counterpart: rays (for example
from `source.sample_rays(mean, covariance, ...)`) traced through the components, weighted by
the transmission of apertures and specimens, counted in their pixels on a finer sub-pixel grid
and blurred by the detector's PSF.

## Parked

Sampled wavefronts (`wave/`), the custom GPU Gaussian renderer (`render/`, also on branch
`park/gpu-gaussian-renderer`), the SEM reflection work, the holography example and
`microscope.py`, the image moments, the atomic potentials and the condenser model were
removed from this branch in the 2026-11-09 cleanup; branch `park/before-cleanup-2026-11-09`
keeps them.
