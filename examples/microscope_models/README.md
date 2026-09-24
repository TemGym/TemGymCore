# Microscope fitting

Start with [microscope_calibration_pipeline.ipynb](microscope_calibration_pipeline.ipynb).

**TemGym truth model → ray images → measurements from pixels → independent
matrix fit → geometry check.** The browser shows saved images beside native
ray diagrams, including the source-to-condenser-mini section.

The selected procedure uses 356 images at 200/195 kV: 216 projector images through
the 50 µm aperture, 128 condenser images through 20/50 µm apertures and 12 checks.
Normal illumination is nearly parallel, with focused 40,000×/45,000× baselines.
Projector controls vary within ±30%; objective current remains fixed. Projector
calibration keeps one illumination per voltage and uses signed X shifts, X tilts
and Y tilts. The previous condenser design-curve settings are unchanged.

For comparison, the notebook retains 572 synthetic images with both apertures.
`fit_microscope(..., projector_aperture_um=50)` excludes 20 µm projector images
from both initialization and joint fitting. Setting this argument to `None`
compares both apertures using the same saved pixels. Condenser images always use
both apertures; incident dose remains unknown.

The 256×256 images use 4,194,304 native rays and 32 internal detector samples per
pixel axis. Exposure stays at 10⁸ incident electrons/image. Image generation and
fitting are cached. `%matplotlib widget` requires `ipympl` in the notebook kernel.

Both the 20% and 30% experiments pass numerical checks and produce full-rank fits;
20% has much weaker projector-position precision. The selected 30%/50 µm fit gives
0.0105 mm source-z error but 0.337 mm local uncertainty, with prediction RMS 1.088.
Several projector positions still miss the 0.1 mm target. These are synthetic
ray tests, not a uniqueness or real-microscope accuracy proof. Diffraction is off.

See the [comparison](../generated/microscope_calibration_tilt_y/README.md) and
[saved validation](calibration_results/tilt_y_30_ray_acquisition/validation.json).
The previous [40%/X-only 20/50 µm trial](../generated/microscope_calibration_parallel_20_50/README.md)
and [50/150 µm trial](../generated/microscope_calibration_crossover_search/README.md)
are preserved. [condenser_design_curve.ipynb](condenser_design_curve.ipynb) retains
the separate CL1/CL3 design exercise. See also [implementation notes](../../src/temgym_core/calibration/README.md).

## Interactive Gaussian smiley

[smiley_gaussian_defocus.ipynb](smiley_gaussian_defocus.ipynb) uses the saved fitted
synthetic microscope from the calibration pipeline, including its finite
solenoids and virtual-source waist, position and mean tilt. The portable
[parameter snapshot](smiley_calibration.json) records its provenance.
This is a coherent member of the fitted source; additional incoherent tilt
spread is not averaged into this coherent-wave demonstration.

After updating the lens code, restart the notebook kernel and Run All. Reloading
`smiley_defocus` alone retains previously imported core classes; the helper now
checks for stale transport methods before creating a renderer. A failed render
reports its projector, illumination, sampling and aberration settings.

The default is 200 kV, 40,000×, spot 3, aperture out, 500,000 specimen Gaussians.
Controls cover specimen height, count (10k–1M), solved condenser spots (1–8),
focused magnification (40k–600k), sigmoid aperture radii (10, 25, 80, 150 µm),
objective aberrations and signed PL1 distortion. The condenser follows the
calibration's source-image design curve. Magnification solves native IL1–PL1
controls, keeping the objective fixed and the reference rotation unchanged.
The first source crossover lies between CL1 and CL3.

**View plane → IL1 crossover · diffraction** evaluates the coherent field at the
nominal specimen Fourier plane between IL1 and IL2. Its position is recalculated
for each magnification, and marked on the ray diagram. Physical screen coordinates
are in µm; its field spans nominal frequencies ±0.12 nm⁻¹. PL1 is downstream and
cannot affect that screen. Switch back to **Detector image** for imaging.

Apertures use the fitted plane and 0.15 µm sigmoid edge. The 10/25 µm radii use
their fitted centres; the larger two radii are illustrative extensions. Inserting
an aperture activates a resolved source-to-aperture calculation and 2D complex
field recomposition at the specimen (512×512 illumination and 256×256 interpolation grids).
The full selected specimen bundle then traverses the aberrated downstream column.
Without an aperture, it propagates continuously from the fitted source.

The ray diagram shows sampled Gaussian centres, not beam widths or interference.
After recomposition its upstream trace is an illumination guide. The detector
is point-sampled; quantitative results require specimen/aperture convergence.
Run all cells with the notebook extras, preferably on a JAX GPU.

[Historical basis checks](smiley_source_basis_checks.json) and their radial
comparison script concern the previous illustrative source/TOML microscope;
they are not validation of this fitted model. Current tests check the native
source and crossover, imaging and Fourier conditions, coherent vacuum propagation,
and control/aberration behaviour.

[Current fitted-model checks](smiley_calibrated_checks.json) record the source,
spot radii, Fourier-plane locations, focus/aperture convergence and slider sweep.
