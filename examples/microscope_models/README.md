# Microscope calibration

Start with [microscope_calibration_pipeline.ipynb](microscope_calibration_pipeline.ipynb).
It uses one procedure for both halves of the microscope:

**Gaussian images → image measurements → matrix fitting → new-image checks.**

The notebook imports `from temgym_core import calibration`. No example scripts
or `sys.path` changes are needed. Install the notebook dependencies with
`pip install -e '.[inversion]'`; on a supported CUDA system use `.[inversion,gpu]`.
Enable JAX 64-bit arithmetic as shown in the notebook.

| Call | Supply | Receive |
|---|---|---|
| `simulate_experiment` | Voltages, aperture diameters, output folder | Recorded images, currents and controls |
| `measure_projector` | Experiment folder | Saved image measurements and covariance |
| `fit_projector` | Measurements; optional starting guesses and bounds | Projector parameters and uncertainty |
| `measure_condenser` | Experiment and fitted projector | Specimen-beam estimates with shared uncertainty |
| `fit_microscope` | Saved measurements; optional condenser guesses | Both fitted halves and joint covariance |
| `check_full_microscope` | Experiment and fitted microscope | New-image, current and geometry checks |

The default [presets](../../src/temgym_core/calibration/presets) contain the
acquisition controls, independent guesses and engineering bounds. They are
included in the installed package. Pass your own JSON files to the fitting
calls to replace the guesses or bounds.

To refit an existing acquisition in a separate folder, use
`copy_recorded_experiment(source_directory=..., output_directory=...)`, then
start at `measure_projector`. This copies known specimen data and recorded
controls, without previous measurements or fitted answers. Simulated truth is
copied for the final geometry score; fitting does not read it.

The full demonstration uses 200/195 kV with objective current fixed: 198
projector images, 144 condenser images and 234 current readings. A separate
set supplies 72 prediction images and 48 current readings. Images are 256×256.
The specimen's complex transmission, detector response and calibrated specimen
tilts/shifts are known. Apertures have known 4/10 µm diameters and an assumed
0.2 µm centering accuracy. These assumptions matter when interpreting recovery.

The Gaussian model includes interference. The matrix fitter predicts the
measured statistics, including specimen phase and diffraction, without rendering
pixels on every iteration. `image_model="intensity"` remains a faster option
within the same procedure; it also supplies initial beam estimates.

See the [code map and audit](../../src/temgym_core/calibration/README.md),
[results and limits](COHERENT_INTEGRATION.md), and
[saved reference result](calibration_results/full_gaussian_recovery.json).
Earlier calibration experiments are preserved in Git checkpoint `61457c3`.
The other notebooks here concern microscope design and visualization, not
alternative fitting procedures.
