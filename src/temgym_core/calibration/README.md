# Reading the calibration code

Start with `workflow.py`, which implements the notebook's public calls:
**generate images → measure images → fit the microscope → check predictions**.
Each stage receives explicit inputs and saves its outputs. The workflow also
checks input contents and source changes before reusing saved results.

| File | Responsibility |
|---|---|
| `workflow.py` | Public steps, defaults and cache validation |
| `acquisition.py` | Condenser controls and synthetic acquisition |
| `gaussian.py` | Known specimen, Gaussian image generation and convergence checks |
| `projector_images.py`, `condenser_images.py` | Image records and intensity-based initialization |
| `projector_model.py`, `condenser_model.py` | Physical parameters mapped to native components |
| `projector.py` | Projector measurements and fitting |
| `apertures.py` | Transmission, current uncertainty and aperture centres |
| `inference.py` | Shared uncertainty, multiple starts and joint microscope fit |
| `checks.py` | New predictions, geometry comparison and display |
| `presets/` | Demonstration controls, independent guesses and bounds |

Reusable physics stays in the parent package: `components.py`, `run.py`,
`source.py`, `evaluate.py`, `image.py` and `multiscale_sample.py`. Numerical
fitting uses `optimization.py`. See the [core code map](../README.md).

Image analysis receives the known specimen and detector calibration. Fitting
receives the measured quantities and uncertainties. Generating parameters are
reserved for scoring the recovered geometry. The final refinement returns to
the original image moments; it does not double-count the intermediate fits.

Coherent sampling, detector integration and correlated measurements need their
existing numerical checks. The saved Gaussian solution still has two acceptable
second-aperture centres; successful predictions and local rank do not prove
that the physical geometry is globally unique.

The saved reference result is
[full_gaussian_recovery.json](../../../examples/microscope_models/calibration_results/full_gaussian_recovery.json).
At checkpoint `fc992bc`, its residual was independently re-evaluated at
chi-squared 1650.0543239, with JAX directional derivatives agreeing with finite
differences to 2.3e-6 relative or better. The later structural cleanups preserve
the numerical algorithms; they do not repeat the full multiple-start search.
