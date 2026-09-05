# Adaptive projector calibration

The [companion notebook](projector_adaptive_calibration.ipynb) continues the
MAGICAL/tilt/beam-shift experiment. It keeps the specimen visible and uses
calibrated beam displacements and tilts to measure the positional response of
the column. The [numerical summary](calibration_results/adaptive_summary.json)
records all 19 parameter errors, controls, seeds and independent checks.

## What changed

The first 82 images used single-lens projector wobbles. The objective stayed
fixed, and calibrated tilt/shift measurements were concentrated at IL1-wobble
settings. New designs cover mixed projector changes, compensated objective
changes, and wider compensated configurations. Every selected setting adds
one ordinary image, four tilt images and four beam-shift images.

Candidate selection uses the current fitted model, measured noise estimates,
and the weak directions of its Jacobian. It does not read generating truth.
Predictions are saved before new instrument images are generated. Image
analysis refits image shape, pose and brightness before the microscope fit
receives measurements and covariance. The final objective uses every measured
quantity once, with the same priors as the original notebook.

| Added measurements | Total calibration images | New-data RMS before refitting | After refitting | Largest parameter error |
|---|---:|---:|---:|---:|
| Starting experiment | 82 | — | — | 24.2% |
| Eight mixed-projector settings | 154 | 15.38 | 0.93 | 1.65% |
| Eight compensated objective settings | 226 | 9.38 | 0.88 | 2.04% |
| Twelve wider compensated settings | 334 | 5.13 | 0.87 | 0.108% |

Errors are in measurement-noise units; parameter errors refer to this synthetic
realization. Individual errors need not improve monotonically as noisy data
are added.

## Why the fitting order matters

For round lenses, calibrated position and angle changes give signed image
motion through `A @ position_change + B @ angle_change`. These measurements do
not require known specimen rotation, illumination curvature or angular spread.
Fitting the 16 optical parameters to these displacements first provides a
useful starting geometry. The final joint fit then estimates all 19 parameters
from the complete data, with unchanged priors.

Direct joint fitting reached the good solution from 1 of 20 dispersed starts.
The displacement-first procedure reached the same solution from all 20.
Optimizers can report convergence at very poor fits, so the workflow also
checks the residual magnitude. A separate rotation-only estimate recovered
lens strengths accurately but did not improve joint convergence by itself.

The paper's two-dimensional lens sweeps and separate objective calibration
motivated the new measurement design (Pfeifer et al., supplied PDF
`../lens_inversion/paper/2607.29411v1.pdf`, pp. 3–5). This implementation fits the
finite-lens matrix model directly rather than applying thin-lens critical-point
formulas to image diameters.

## Checks

- A frozen final model predicts 72 independent images at new settings with
  RMS 0.949 noise units. None of these images is used in the reported fit.
- Two further independent realizations regenerate all 334 images. Their
  maximum errors across all 19 parameters are 0.103% and 0.215%.
- Removing soft priors changes detector position by about 0.0063 mm. Hard
  geometric and parameter bounds remain in place.
- The smallest data-Jacobian singular value, in the same declared parameter
  units, rises from 0.000349 to 46.9. The final local sensitivity has no
  comparably weak direction. This is not a global uniqueness theorem.
- Detector profiles refit the other 18 parameters at every point. The final
  detector error is about 0.207 mm, versus 10.29 mm in the starting calibration.
- Using only calibrated displacements recovers all 16 optical parameters
  within 0.387%, without using common illumination parameters in the optical fit.

## Scope and reproduction

The broadest settings include IL1/IL2 currents between three and four times
reference excitation. The linear control law and finite-solenoid model are
assumed valid throughout. Input calibrations are exact; independent frame drift
has standard deviation 0.003 detector pixels. The specimen intensity template
is known. Coherent contrast, hysteresis, saturation, hardware limits, and real
illumination changes during objective adjustment are not validated here.

Run the original calibration notebook first, then the companion notebook.
`reuse_completed=True` opens the recorded experiment. Set it to `False` to
regenerate the candidate designs, images, fits, prediction checks and noise
repeats. Helpers in `projector_adaptive_experiments.py` keep those steps separate.
The generating microscope is read only by image simulation and final scoring.
