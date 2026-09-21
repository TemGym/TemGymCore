# Current calibration checkpoint

The Y-tilt / smaller-projector-perturbation comparison is complete. The maintained
notebook selects a 30% projector cap and the 50 µm projector aperture.

- Normal settings: 40,000×/45,000× at 200/195 kV; near-parallel illumination.
  Objective current fixed; previous condenser design-curve settings unchanged.
- Steering: signed X shifts, X tilts and Y tilts, at 36 projector settings.
- Selected procedure: 216 projector images, 128 condenser images (both 20/50 µm
  apertures), 12 prediction images = 356 used images. Exposure unchanged.
- The notebook retains 572 simulated images for aperture comparison. The selected
  fit ignores the 20 µm projector images in both initialization and joint fitting.
- New optional API: `fit_microscope(..., projector_aperture_um=50)`; `None` uses both.
  Dose stays unknown, and unused residual rows do not count as observations.
- Both 20% and 30% image sets pass numerical checks (worst pair 0.05993 noise units).
  Both produce accepted full-rank fits. 20% gives much weaker projector geometry.
- At 30%, source-z uncertainty is 0.298 mm with both apertures and 0.337 mm with
  50 µm only; PL1 uncertainty is 0.337 versus 0.358 mm. One aperture saves 216 images.
- Selected fit: source-z error +0.01049 mm, local uncertainty 0.33690 mm;
  fit RMS 0.88065, prediction RMS 1.08765, rank 47/47. Several projector positions
  still miss 0.1 mm. This is not a global uniqueness or real-microscope accuracy proof.
- All six notebook code cells executed without errors; all 21 workflow tests pass.

Notebook: `microscope_calibration_pipeline.ipynb`.
Report: `../generated/microscope_calibration_tilt_y/README.md`.
Saved comparison: `calibration_results/tilt_y_30_ray_acquisition/validation.json`.
The previous 40% X-only and earlier 50/150 µm acquisitions remain preserved.
No commit was made; the workspace includes earlier uncommitted work.


## Software validation (10 September 2026)

Camera export/conversion and actual microscope pilot collection are explicitly
**deferred by the user**. Do not ask for these again in this task.

- `calibration.import_images` accepts calibrated electron-count arrays plus frame
  records. It needs only the selected 356 images and creates no truth file.
- Measurement, fitting, prediction checks and browser support missing 20 µm
  projector frames. Without truth, geometry truth/error columns are blank.
- Replaying seed 262510 preserves selected pixels, measurement values, covariance
  and regions exactly. Refitted geometry changes by at most 0.0014 local standard
  deviations from the earlier dense-input result.
- All six preselected seeds 262510–262515 produced accepted rank-47 fits at fixed
  exposure. Source-z empirical SD: 0.3071 mm; mean local SD: 0.3374 mm; RMS error:
  0.3249 mm; largest error: 0.6631 mm. Mean signed error was −0.1643 mm.
- All 24 projector starts passed; 15/24 condenser initializers passed (eight
  failed/stalled, one converged to a bad residual and was rejected). All 15 joint
  starts passed and agreed within 0.021 local standard deviations per dataset.
- Prediction RMS across repeats: 0.933–1.118 noise units. This is encouraging
  synthetic repeatability, not global uniqueness or real-microscope validation.
  Several projector positions still do not have 0.1 mm precision.
- All 25 workflow tests pass. The imported-data notebook branch executed all six
  code cells without errors or captured widget tracebacks, on the GPU.
- Full synthetic notebook rerun also passed: all six code cells, fresh ray images,
  measurement extraction, four-start fitting, predictions and the interactive browser.
  Both notebook paths execute without errors or captured widget tracebacks.

Maintained notebook: `microscope_calibration_pipeline.ipynb`. Its
`recorded_directory=None` keeps the synthetic workflow; an imported folder uses
exactly the same remaining notebook cells. The array input contract and short
pilot outline are in `../../src/temgym_core/calibration/README.md`.

Permanent repeat summary:
`calibration_results/tilt_y_30_ray_acquisition/repeat_validation.json`.
Detailed report and reproducible scripts:
`../generated/calibration_repeat_validation/README.md` (`run.py`, `summarize.py`).
Every run is retained, including individual failed starts. Truth is used only in
separate post-fit synthetic scoring. The original noiseless ray signals are
reused for fresh Poisson/read-noise realizations; no new exposure is assumed.
No commit was made; earlier uncommitted work is preserved.
