# Microscope calibration

Start with [the notebook](../../../examples/microscope_models/microscope_calibration_pipeline.ipynb),
then read `workflow.py`. The active notebook uses:

**Native rays → saved detector pixels → measured moments → independent fit → geometry check.**

- `simulate_ray_experiment` supplies the saved acquisition and generating column
  to native ray transport, aperture transmission and detector binning. The ray
  count controls numerical sampling; exposure stays fixed. It saves `images.npz`.
- `measure_images` reads those pixels and detector calibration, measures the beam
  centre, shape and total intensity, and estimates their covariance. It does not
  read generating microscope parameters.
- `fit_microscope` predicts statistics with native matrices and aperture integrals;
  it fits from independent guesses, without rendering pixels each iteration.
- `check_microscope(..., render_images=False)` compares unused measured statistics
  and reports local geometry uncertainty. If synthetic truth is present, it also
  scores geometry against truth. `browse_images` shows the saved images.

The geometric image experiment sets `aperture_diffraction=False` in its metadata;
`make_predictions` uses that setting to omit diffraction added by the aperture.
The source distribution is unchanged. Wave-image experiments default to including
aperture diffraction. Geometric rays cannot validate diffraction rings.

The active ray notebook adds Poisson counting noise and calibrated detector read
noise. It uses 4,194,304 rays/image, 32 internal detector samples per pixel
axis and a 16-pixel measurement margin. Numerical ray sampling is separate from
exposure. Uncertainties come from the recorded pixels. Image fits use ordinary
convergence tolerances, even when detector noise is disabled.

`simulate_measurements` remains available for exact direct-moment solver tests.
The earlier noiseless test recovered source z within 0.000002 mm after fixing
premature stopping; it did not test image extraction. Its results are preserved
in `examples/generated/microscope_calibration_moments/`.

The Gaussian workflow remains available through `simulate_experiment`,
`measure_images`, and `check_microscope(..., render_images=True)`. The archived
executed notebook and reference measurements are in
`examples/generated/microscope_calibration_40k_45k/`. The following acquisition
and physics notes apply to both workflows.

The length scale is assumed calibrated. Images show the beam through a large
vacuum hole, with a condenser aperture inserted throughout. The current notebook
loads its selected acquisition from
`examples/microscope_models/calibration_results/tilt_y_30_ray_acquisition/`:
200/195 kV, 20/50 µm apertures, 432 simulated projector images, 128 condenser
images and 12 prediction-check images. The selected fit uses only the 216
projector images through 50 µm, giving 356 images including checks. Exposure remains 10⁸ incident electrons per image.
Objective current stays fixed. One illumination setting is kept per voltage
during projector calibration. Known X shifts and X/Y tilts measure the round
projector's scale and rotation response.

The two normal focused baselines are 40,000×/45,000×, with nearly parallel
specimen illumination. Combined projector-current perturbations reach 30%;
these calibration views can be defocused and very small. The finer detector
grid avoids rounding their centroids before applying the detector PSF.
`design_condenser_curve` solves CL1/CL3 source imaging at a fixed plane before
mini. The selected acquisition explores independent condenser changes around
this normal-operation curve. The fit does not receive its image plane or
magnifications as observations.

The first source crossover must stay in the free CL1–CL3 gap, between fractions
0.01 and 0.99 on the first excitation branch. Feasibility penalties guide the
solve; accepted fits must have violation below 1e-6. Penalty rows are excluded
from data chi-squared, rank and local covariance. This operating restriction is
an assumption to check on the real instrument, not a measured axial distance.

Both 20% and 30% acquisitions pass numerical image checks (worst pair 0.060
noise units), and both give accepted rank-47 fits. The selected 30%/50 µm procedure
gives source-z error +0.0105 mm, local uncertainty 0.337 mm and prediction RMS
1.088 noise units. Several projector positions still miss the 0.1 mm target.
See the [comparison](../../../examples/generated/microscope_calibration_tilt_y/README.md).
The earlier 40% X-only and 50/150 µm acquisitions remain preserved.

| File | Responsibility |
|---|---|
| `workflow.py` | Fast synthetic measurements, image workflow, logs and saved-result validation |
| `viewer.py` | Browse saved images or geometric ray previews beside ray diagrams; display fit results |
| `recorded.py` | Validate calibrated image arrays and their acquisition records |
| `acquisition.py` | Recorded settings and native geometric ray rendering |
| `gaussian.py` | Interfering Gaussian beam images and numerical accuracy checks |
| `projector_model.py`, `condenser_model.py` | Physical parameters mapped to native components |
| `apertures.py` | Integrate native aperture transmission to predict beam moments and flux |
| `inference.py` | Image measurements, independent starting guesses and joint least squares |
| `checks.py` | Predictions at unused settings and recovered-geometry comparison |
| `presets/` | Demonstration controls, independent guesses and engineering bounds |

`presets/beam_measurement_protocol.json` supplies the baseline acquisition recipe.
The active notebook instead loads the selected recorded settings above. Each
projector entry means two signed kicks through both apertures (four images).
Each condenser entry means one image through each aperture (two images).
Targets and seed currents set up the synthetic instrument; fitting receives
only the recorded commands. These are not hardware currents for a real TEM.

Reusable physics lives in `components.py`, `run.py`, `source.py`, `evaluate.py`
and `moments.py`. Both image generation and fitting use the same native
solenoid and sigmoid-aperture components. The bounded solver is the existing
JAX solver in `optimization.py`; matrix derivatives come from JAX. GPU image
summation groups nearby Gaussians and retains interference. The MAGICAL specimen
recipe remains in `gaussian.py` for other imaging notebooks; calibration does
not use it.

## What is measured and fitted?

Each image supplies centre x/y, variance x, covariance xy, variance y and total
intensity, with their full measurement covariance. Coordinates match the native
`Detector`: x rightward, y upward. For wave-image datasets, aperture amplitude gradients contribute diffraction
to the predicted moments; the ray experiment disables that term. Fitting these quantities avoids rendering
pixels at every optimizer step.

The projector is initialized from centroid differences between positive and
negative calibrated shifts or tilts. Each pair directly measures the matrix
acting on its known input change; a zero-kick frame and all four kick axes are
not required at every lens setting. The condenser is then initialized with
that projector fixed, retaining its existing log/unit coordinate scales and
using the native solver’s geodesic steps. A final
joint fit returns to the original measurements, so the initialization estimates
are not counted twice. The fit has 47 physical coordinates: 16 for the projector
and 31 for the condenser, source and aperture positions. Source position, size,
angle and incoherent angular spread can differ between voltages; the native
Gaussian diffraction floor is always included. Engineering bounds are used,
without soft geometry priors.

Each pair of aperture sizes shares an unknown incident dose, eliminated
analytically during fitting. There is **no aperture-out reference**. This requires
stable incident dose and equal exposure within each pair, or an independently
measured correction. Calibrated shifts and tilts specify the response at the specimen. The forward
model realizes them with the existing `DoubleDeflector` in the free gap after
the aperture and before the mini lens, leaving aperture illumination unchanged.
Newton–Raphson aligns its four balance ratios: shift produces no specimen tilt,
and tilt produces no specimen displacement. The alignment uses JAX derivatives
of the native ray response. Its squared error is quadratic in the ratios, so
one Newton step suffices for this linear column. A final linear solve converts
the requested specimen displacement/angle to the aligned component's commands.
The viewer reports these model ratios; they are not PyJEM balance readbacks.
The two physical positions are illustrative fractions of that gap, not measured
or fitted. The
known specimen response is enough for the present moment fit. Recovering the
deflector positions additionally needs the individual X/Y coil drives, or a
known mapping from controls to those drives, and their calibrated specimen
responses. That hardware mapping is not established for this dataset.

For the current synthetic geometry, the two effective kick planes are at
−63.95 and −42.55 mm relative to the specimen: **21.4 mm apart**. This is an
engineering starting guess, roughly 20–25 mm, with 10–40 mm alternatives worth
checking inside the model's 53.5 mm aperture-to-mini gap. These are not measured
NEOARM dimensions or fitting priors. [JEOL describes the general location](https://www.jeol.com/words/emterms/20121023.075958.php)
but gives no spacing there. A different electron-optical design uses 67 and
76 mm separations ([published design](https://patents.google.com/patent/US20240355577A1/en));
its dimensions support considering a centimetre scale, not narrowing the
NEOARM spacing to our illustrative value. The distances refer to effective
deflection planes, not clearances between finite coil bodies.

Analysis and fitting read recorded images, commands and calibration information.
Generating parameters are used by the dataset browser, numerical forward-model
checks and the final synthetic geometry score. Noise agreement, multiple starts, local rank
and new-image predictions are checks of the model; none proves global uniqueness
or establishes that real lens fields are hard-edge solenoids.

Saved results are reused only when their inputs and numerical source fingerprints
match. Viewer and plotting edits do not invalidate the numerical stages.
Generated arrays, fit records and logs live in the notebook's output directory.
The earlier specimen-image/current-reference experiment is documented separately
in [the historical results](../../../archived/examples/microscope_models/COHERENT_INTEGRATION.md).

## Recorded data

`images.npz` contains `images[pair, aperture, row, column]` in detector counts.
An aperture pair has identical voltage, lens commands, calibrated shift/tilt and
exposure; only the inserted aperture changes. `experiment.json` records the two
voltages, aperture diameters, detector pixel pitch, PSF, dark level and read noise.

`settings.npz` has one row per pair:

| Array | Meaning |
|---|---|
| `voltage_index` | Index into the two recorded voltages |
| `condenser_controls` | Relative currents for CL1, CL3 and the mini lens |
| `projector_controls` | Five relative lens currents, in `projector_model.LENSES` order |
| `specimen_kick` | Known x/y displacement in metres and x/y tilt in radians |
| `purpose` | `projector`, `condenser` or `check` |
| `response_group` | Groups the calibrated shifts/tilts at one projector setting; −1 otherwise |

A real acquisition can supply these files and start at `measure_images`.
`check_microscope(..., render_images=False)` also works without
`scoring_truth.json`; truth/error entries are then blank. Geometry scoring is
only available for synthetic data.

## PyJEM control names

The [PyJEM 1.4 Def3 reference](https://pyjem.github.io/PyJEM/1.4.0.3782/interface/PyJEM.TEM3.__deflector.html)
provides separate readbacks for the following settings. Their documented units
are integer I/O values from 0 to 65535, not metres, radians or calibrated coil
current percentages.

| Readback | Documented name |
|---|---|
| `GetCLA1()`, `GetCLA2()` | CLAlignment1 and CLAlignment2 |
| `GetShifBal()` | ShiftBalance |
| `GetTiltBal()` | TiltBalance |
| `GetAngBal()` | AngleBalance |

Balance adjusts how the upper and lower deflection coils work together. Tilt
balance keeps the beam position fixed during tilt; shift balance keeps its angle
fixed during shift. These functions are described in the JEOL JEM-2100 manual,
sections 5.7.3a–b ([manual copy](https://www.scribd.com/document/359396730/TEM-manual-JEOL-2100-pdf)).
This describes the alignment principle, not a verified NEOARM control mapping.

In particular, [Instamatic's JEOL interface](https://github.com/instamatic-dev/instamatic/blob/main/src/instamatic/microscope/interface/jeol_microscope.py)
uses CLA1 for beam shift and CLA2 for beam tilt. Do not assume these are unmixed
upper/lower coil drives. Save both CLA readbacks and all three balance readbacks,
along with the voltage, lens settings and TEM/STEM mode, when calibrating the
specimen response. Keep the balance settings fixed within a calibration series.
The PyJEM reference does not specify their conversion to physical coil mixing;
it also describes a STEM image-shift contribution superimposed on CLA.

The present fit only needs the measured specimen displacement and angle.
Estimating physical deflector positions requires resolving that additional
control-to-coil mapping. A measured response to four control channels can still
calibrate steering without uniquely locating the coils. TemGym's
`DoubleDeflector` balance coefficients are model ratios, and must not be filled
directly with the PyJEM integer balance values.


`fit_microscope(..., projector_aperture_um=50)` selects one recorded projector
aperture. `None` keeps both. The response initializer selects the same aperture;
the joint likelihood profiles dose using only selected images and removes unused
residual rows from rank, covariance and degrees-of-freedom calculations. Condenser
and prediction-check images retain both apertures. Both projector apertures stay
in the synthetic notebook's image files to support direct comparison, but only
one is required by the selected measurement procedure.


## Recorded image arrays

The [six-noise-repeat report](../../../examples/generated/calibration_repeat_validation/README.md)
uses the selected 356 frames through this input boundary, with no truth file
available during fitting. All six fits passed; source-z scatter was 0.307 mm
versus 0.337 mm predicted local uncertainty. Individual failed starts are retained.

The input boundary accepts calibrated arrays, independent of camera file format:

```python
images = calibration.import_images(
    images=electron_count_images,  # shape: (frames, rows, columns)
    records=frame_records,         # one dictionary per frame; fields below
    specification=detector_and_model_assumptions,
    output_directory="recorded_calibration",
)
measurements = calibration.measure_images(experiment_directory=images)
microscope = calibration.fit_microscope(
    measurements_directory=measurements,
    number_of_starts=4,
    projector_aperture_um=50,
)
checks = calibration.check_microscope(
    experiment_directory=images, microscope=microscope, render_images=False,
)
```

Each record supplies `measurement_id`, `aperture_um`, `voltage_v`, `exposure_s`,
`purpose`, `response_group`, `projector_controls`, `condenser_controls` and
`specimen_kick`. Optional `nominal_magnification` labels the browser.
Two aperture frames at the same setting share a measurement ID. For projector
settings, record only the 50 µm frame; condenser and check settings use both.
An equal-and-opposite, single-axis steering pair shares a nonnegative response
group; other settings use −1. Lens controls are five projector and three
condenser **current/reference-current ratios**, not DAC values. Kicks are
specimen-plane [x, y, theta_x, theta_y] in metres/radians, with y upward.

The specification must explicitly contain `voltages_v`, `aperture_diameters_um`,
`pixel_pitch_m`, `psf_sigma_pixels`, `background_counts`, `read_noise_counts`,
`aperture_edge_width_m` and `aperture_diffraction`. Set `image_units="electrons"`,
`image_axes="row_down_column_right"` and
`control_units="current/reference_current"`. The active ray test uses voltages
[200000, 195000], apertures [20, 50] µm and `aperture_diffraction=False`.
These are model assumptions to validate before fitting real data. The optional
`moment_margin_pixels` controls the measurement region; the current test uses 16.
Set `crossover_fraction_bounds=[0.01, 0.99]` to use the same first-crossover
restriction as the demonstration. Optional `saturation_counts` rejects saturated
frames when the detector limit is supplied.

The importer requires an empty output folder and validates pairing, fixed
objective current and one projector illumination per voltage. Missing aperture
slots are masked, never counted as zero-valued observations. Equal exposure and
stable incident dose within each aperture pair are still required. The fit
profiles an unknown dose per measurement; exposure alone does not calibrate it.
No source/lens truth file is created or read. Camera-format conversion, detector
calibration and the real acquisition are deferred at the user's request.

## Deferred microscope pilot

When returning to the microscope, first test a short signed shift/X-tilt/Y-tilt
sequence at fixed voltage, objective current and illumination, with an aperture
inserted. Repeated zero-steering images and a dark image would test drift,
return-to-zero and background subtraction. A mildly defocused projector view
makes the tilt response visible; a focused image with B near zero cannot verify
the tilt amplitude by centroid motion alone. This is a proposed later check,
not collected data. Synthetic lens multipliers are not a hardware DAC recipe.
