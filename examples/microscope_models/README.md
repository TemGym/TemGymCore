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

[smiley_gaussian_defocus.ipynb](smiley_gaussian_defocus.ipynb) sends exactly
500,000 coherent Gaussian beamlets by default through a smoothed smiley and the full
40,000× microscope imaging column onto a 128×128 detector. The full condenser
supplies its coherent illumination. A sample-height slider (±10 µm) produces
defocus with fixed lens currents; a second slider adjusts display contrast.
An optional objective-aberration panel adds twofold astigmatism, axial coma,
and spherical aberration through a native `KrivanekLens` objective, retaining
the original objective focal length and magnetic rotation. Its large,
explicitly labelled coefficient ranges make the effects visible at overview scale.
Run all cells with the notebook extras and a JAX GPU for interactive updates.
The notebook includes the calibration ray diagram beside the live image, a saved
full-column ray overview, image previews, measured timings, and a focus check.
The companion [smiley_defocus.py](smiley_defocus.py) contains the forward model.

The **Gaussians** selector switches between six counts (10,000–1,000,000).
Each count has its own compiled renderer and image cache; the focused reference
and live image update together. Lower counts trade numerical accuracy for speed.

The live ray diagram follows a subset of the actual Gaussian beam centres through
the specimen and aberrated Krivanek objective. Aberration sliders update the
downstream trajectories; a dashed guide shows the zero-aberration comparison.
