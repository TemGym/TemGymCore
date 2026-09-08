# Full microscope calibration

[The notebook](microscope_calibration_pipeline.ipynb) now runs the full two-voltage
Gaussian-image calibration. The [portable result](calibration_results/full_gaussian_recovery.json)
contains the measurements' provenance, all fitting stages, numerical checks,
geometry errors and prediction results.

## Procedure

TemGym generates coherent images from Gaussian beams propagated through its
lens components. Complex fields are summed before source averaging and detector
integration. The GPU renderer uses nearby-Gaussian grouping where it helps,
with its dense fallback at broad crossovers. FFT propagation is absent from
the active pipeline.

Image analysis measures centroids and second moments with their covariance.
The projector fit uses ordinary images and calibrated tilt/shift pairs. Three
projector views then estimate each condenser beam. These beam estimates
initialize the condenser fit; the final fit uses the **original measured image
moments and current ratios**, with both halves and their uncertainty shared.

The expensive specimen integrals are updated between inexpensive nonlinear
matrix fits. Each accepted update must decrease the original measurement
objective. The final Jacobian and covariance are evaluated at the returned
model. This avoids a Gaussian likelihood approximation for weak logarithmic
angular-spread estimates, which failed the earlier goodness-of-fit check.
Gaussian images are rendered for acquisition and independent prediction checks;
optimization uses TemGym matrices, wave moments and JAX least squares.

## Acquisition and assumptions

- 200/195 kV, with objective current fixed throughout.
- 198 projector and 144 condenser calibration images; 234 current readings.
- 72 independent prediction images and 48 prediction current readings.
- 256×256 cameras, with calibrated effective pixel pitches of 110 µm for the projector calibration and 220 µm for condenser observations.
- Known 4/10 µm aperture diameters, separate unknown centres and an assumed 0.2 µm centring accuracy per axis; apertures are retracted for imaging.
- Known complex specimen transmission, detector response, voltages and calibrated specimen-plane tilts/shifts. Projector calibration uses a known circular illumination profile, with fitted curvature and angular spread at each voltage.

There are 47 microscope/source/aperture parameters and five projector-illumination
nuisance parameters. Source states are independent between voltages. Engineering
bounds and the declared centring constraint are used, without additional
lens-position or thickness penalties. The coaxial, paraxial lens model uses
effective uniform-field lengths; the source model is Gaussian.

## Result

The selected joint fit has RMS **1.000016 noise units**, rank
**52 of 52**, and no active bounds. Initial searches use four
projector guesses and eight condenser guesses. A targeted current-only search
also proposes alternative aperture centres, which undergo full joint refinement.
The selected minimum is chosen using calibration data, before prediction checks.

| Check | RMS in noise units |
|---|---:|
| Calibration image moments | 1.010973 |
| New Gaussian images, illuminated regions | 0.981934 |
| New Gaussian images, full camera | 0.962125 |
| New projector measurements | 0.887935 |
| New aperture-current ratios | 1.198698 |

The new-image check renders 36 condenser images with geometry fixed. Only
per-frame dose and background are fitted. The illuminated-region check prevents
the large dark background from dominating the image score. Current and projector
prediction scores include propagated parameter uncertainty.

In this realization, the largest lens-position error is **0.01013 mm**,
the largest relative field-length error is **0.403%**, and
the largest fitted source-plane position error is **0.02517 mm**.
These recovery errors describe one simulated acquisition. Repeated-noise
coverage remains untested.

**The second aperture centre has two acceptable solutions.** Their separation
is about 0.26 µm in x. The notebook shows their range as well as the local error
bars; the latter describe only one minimum. Agreement between initial guesses,
a full-rank Jacobian and good new-image predictions do not prove global uniqueness.

## Validation and reproducibility

All 180 condenser acquisition/prediction images pass the independent noiseless
image-moment audit: discrepancy RMS 0.015049 measurement-noise units.
Gaussian-basis, pixel-integration and known-specimen quadrature checks are separate.
The image basis uses 0.5 nm spacing; moment integrals use 0.25 nm spacing with a
0.0625 nm local grid for very small beams. Adaptive pixel refinement and its
history remain beside the recorded images. This audit tests the fitted image
quantities; earlier detailed forward convergence checks are preserved in checkpoint `61457c3`.

Full-residual directional finite differences agree with JAX derivatives to
2.2e-06 relative error; an additional weak-direction check agrees to
1.8e-6. Re-evaluating the covariance reproduces the saved result. These checks
forbid access to generating parameters and calls to the Gaussian renderer during
fitting. Unit tests also check shared covariance, nonlinear refinement, aperture
alternatives and numerical moment propagation.

The recorded dataset was assembled during numerical validation and then adopted
into the notebook cache after its physics, residuals and covariance were checked
against the final code. The portable report includes raw-data hashes and this
provenance. The executed notebook reuses those verified stages. The cleanup moved this procedure into `temgym_core.calibration`; source changes
invalidate stage caches. Setting `reuse_completed=False` regenerates or refits
the relevant stage. The portable result records the original validated run.

This validates the stated ideal model. Real field profiles, aberrations, specimen
uncertainty, voltage readback errors and repeated-noise confidence coverage remain
outside this demonstration. The fast spatial renderer is forward-only. An older,
separate direct-pixel GPU derivative test retains one tolerance failure; that path
is not used by this matrix/moment fitter.
