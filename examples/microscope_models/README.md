# Microscope calibration

Use [microscope_calibration_pipeline.ipynb](microscope_calibration_pipeline.ipynb).
It contains one procedure for fitting both halves of the microscope at two voltages.

1. **Generate images.** TemGym propagates Gaussian beams through its lens components and sums complex fields, including interference.
2. **Measure the images.** Extract centroids and second moments with their covariance; measure aperture-current ratios separately.
3. **Fit the microscope.** Estimate the projector and specimen beams, initialize the condenser from independent guesses, then jointly fit the original measurements with TemGym matrices and JAX least squares.
4. **Check predictions.** Freeze the recovered geometry, render new Gaussian images, and compare them with a separate acquisition. Open the generating parameters only for scoring.

The acquisition uses 200/195 kV with objective current fixed, 198 projector
calibration images, 144 condenser calibration images and 234 current readings.
The prediction set contains 72 images and 48 current readings. Camera images
are 256×256; the display crops around the illuminated region.

The 4/10 µm apertures have independent centres and a declared 0.2 µm centring
uncertainty per axis. They are retracted for imaging. The specimen's complex
transmission, detector response, voltages and specimen-plane tilt/shift changes
are known. Lens positions, effective field lengths and strengths, source states
at each voltage, and aperture geometry are fitted. The model is coaxial and
paraxial. Engineering bounds and the stated centring accuracy are used, without additional
lens-position or thickness penalties.

`full_microscope_workflow.py` supplies the notebook's staged API.
`coherent_microscope_workflow.py` handles Gaussian acquisition and numerical
checks. Fitting uses TemGym components, moment propagation and its JAX solvers;
it does not render Gaussian images inside the optimization loop. The final
joint fit avoids treating weak, nonlinear beam estimates as Gaussian data.

On a supported GPU, rendering uses the nearby-Gaussian grouping code and its
dense fallback. The CPU scan path remains available. FFT propagation is not
part of the active pipeline. Saved stages are reused when their inputs match;
logs and numerical checks live beside the generated data.

The full synthetic fit recovers lens positions within 0.0101 mm; new-image
RMS in the illuminated region is 0.9819 noise units. The second
aperture centre retains two acceptable solutions. See the
[full Gaussian recovery](calibration_results/full_gaussian_recovery.json) and
[results and limits](COHERENT_INTEGRATION.md). The earlier
[intensity recovery](calibration_results/intensity_jax_recovery.json) and
[Gaussian GPU benchmarks](calibration_results/gaussian_gpu_256.json) remain
separate reference results.
