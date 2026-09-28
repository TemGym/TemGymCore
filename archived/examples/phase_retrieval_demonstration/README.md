# Phase retrieval from calibrated IL1 changes

**Transport-of-intensity version:** start with [transport_of_intensity.ipynb](transport_of_intensity.ipynb). It reconstructs phase from three calibrated IL1 images using a linear TIE solve. See [TIE_README.md](TIE_README.md) for results and assumptions. The earlier L-BFGS demonstration is retained below.

**Sample and sampling clarification:** the current retrieval uses a phase-remapped version of the smiley, not the original function’s complex transmission. It uses 2,304 Gaussians for both synthetic data and inversion. The reported reconstruction error is not a Gaussian-basis convergence test. The original smiley notebook’s circular-wave settings give 246,301 Gaussians. `figures/smiley_function_amplitude_phase_256.png` shows the unmodified function.

Start with **`figures/00_slide_overview.png`** for a 16:9 slide. Every figure is
also exported as SVG (editable labels) and PDF. Open
`phase_retrieval_demonstration.ipynb` to inspect the saved figures and dataset.

- `01_detector_defocus_series`: four intensity images on one fixed physical camera.
- `01b_detector_phase_contrast`: flat-field-normalized contrast, with one shared
  contrast scale; the overview slide uses these panels.
- `02_phase_retrieval`: true/retrieved exit-wave phase, a deliberately incorrect
  IL1 calibration, and the calibrated phase error.
- `03_native_ray_diagrams`: the existing TemGym `plot_model` ray renderer and
  `legacy_beam_plot_params`, with one panel per IL1 setting.
- `03b_lens_image_planes`: every lens conjugate, with real, virtual and intercepted
  image planes distinguished and the focused reference shown.
- `04_calibration_and_convergence`: the calibrated optical changes and inverse-fit residual.

## Reproduce

From the repository root, using its installed numerical environment:

```bash
.venv/bin/python examples/phase_retrieval_demonstration/demonstration.py
```

The run writes `dataset.npz`, `calibration.json`, and the figures here. JAX uses an
available GPU; the dense Gaussian response matrices also run on CPU, but take
more time and several GB of memory. Use `--figures-only` to re-export layouts from the saved reconstruction.
No core package or source notebook is changed.

## What is being simulated?

The ideal 200 kV microscope comes from
`../microscope_models/microscope.toml`, the companion configuration to the
microscope calibration examples. `find_imaging_condition` establishes its
40,000× focused reference. Only IL1 changes between the four acquisitions:
−28%, −20%, −10%, and +8% relative to the focused reference current. The sample,
objective, IL2, IL3, projector and detector positions/settings stay fixed.
This assumes that the ideal microscope's optical parameters are already known;
it does **not** claim the earlier noisy calibration recovered them exactly.
Currents are simulation amperes, not hardware DAC commands.

The smiley function in `smiley.py` is copied unchanged from
`../gaussians/microscope_sample_objects.ipynb`. That notebook sets its phase to
approximately zero. Here its dark features are instead encoded as a smooth
0–1.2 rad phase pattern, 155 nm across, under a prescribed Gaussian illumination
at the specimen. This deliberate phase version of the sample makes the phase
retrieval visible. Its additive sinusoidal signature texture is removed before mapping the dark
features to phase, so the face remains legible on the slide. The vertical flip makes it upright with `origin='lower'`.

The exit wave is a coherent sum of 48×48 overlapping native Gaussian beamlets,
on a 4 nm lattice with 4 nm waist. Native Gaussian cardinal interpolation fits its coefficients to the illumination
and phase pattern at lattice nodes. The *actual summed exit wave* is the ground truth shown and scored,
not the unsmoothed phase design. Finite interpolation leaves some between-node amplitude variation, even when
the intended specimen transmission is phase-only.
The condenser is represented by this prescribed illumination at the specimen;
propagation starts there. This is not a full source-to-detector coherence model.

Each beamlet propagates through the native objective, IL1, IL2, IL3, PL1 and
drifts. Coherent summation gives the measured intensities. The camera samples a
256×256 grid at 55 µm pitch (14.08 mm full width). These are noiseless point-sampled intensities, without
pixel integration, detector blur, shot noise, aberrations or partial coherence.
The stored intensities include a fixed baseline 40,000² factor. Displayed raw
intensities are additionally corrected for each calibrated magnification (M²
area factor), on a shared linear scale. Panels crop around the beam with explicit
physical detector axes; the camera and saved full image arrays do not change.
The contrast panels show `I / I(vacuum) − 1`, on a shared −0.25 to +0.25 scale,
masking locations below 3% of the vacuum image peak. This display reference is
calculated from the known illumination and optics; it is not an extra exposure
or an extra constraint supplied to the retrieval.

## How calibration enters

`column_matrix` differentiates the actual native ray propagation. In lab-frame
ordering `(x,y,theta_x,theta_y)`, write its blocks as `A, B, C, D`. For these round
ideal lenses,

\[
A^{-1}B=z_{\rm eff} I,\qquad
M=\sqrt{\det A},\qquad
\theta=\operatorname{atan2}(A_{yx},A_{xx}).
\]

`z_eff` denotes signed **specimen-equivalent free propagation** before the
magnifying optical transform. It is not the physical displacement of the detector
or a universal microscope underfocus convention. The code asserts the scalar
form of `A^{-1}B`. The full matrices, currents, magnifications and rotations are
saved in `calibration.json`. The inverse calculation uses the full propagated
Gaussian response, including the phase action; it does not merely rescale a
nominal defocus value or register the images by hand.

The ray plots are geometric illustrations in the co-rotating frame. Their rays
are specimen positions and angles spanning a bundle, not electron intensity or
Gaussian diffraction envelopes. The illustrated launch positions are −65, 0 and
+65 nm, each with −1, 0 and +1 mrad launch angles so conjugate crossings are visible. The top row shows the objective and its image plane; the middle row zooms into
IL1–PL1; the bottom row shows the final detector drift and final image plane.
Each row retains common physical scales across the four settings. Dashed planes
mark real specimen images, found by solving `B + d D = 0` for the transfer matrix
through each lens. The objective image remains fixed when IL1 changes. Arrows
show the signed detector-minus-final-image distance in millimetres, distinct
from the specimen-equivalent defocus in micrometres. The companion conjugate
plot shows all image-plane positions and the focused reference. Virtual images
(backward extrapolations) and forward conjugates intercepted by a later lens
are explicitly distinguished from real crossings.

## Retrieval and checks

For every setting, the propagated native Gaussian fields form a linear operator
`H_j`. The measured data are `I_j = |H_j c|²`. L-BFGS fits **both real and imaginary
parts** of all coefficients by minimizing

\[
\frac{\sum_j\|\sqrt{|H_j c|^2+\epsilon}-\sqrt{I_j+\epsilon}\|^2}
     {\sum_j\|\sqrt{I_j+\epsilon}\|^2}.
\]

The solver starts from the known smooth illumination with zero phase. It receives
only the intensities, optical operators and initial coefficients. It receives no
smiley phase or amplitude map, and imposes no known-phase, phase-only or shape
constraint. Its Gaussian basis and finite spatial support are known. This is a
finite-dimensional synthetic inverse problem using the same basis for generation
and inversion, not a proof of uniqueness for arbitrary objects or real data.

A separate inverse fit assumes 80% of each actual IL1 current step. It uses the
same data and initial wave, without refitting optics. Both fits have the same 3,000-iteration maximum.
This controlled comparison
illustrates sensitivity to a particular calibration error, rather than estimating
experimental calibration uncertainty.

Phase scores remove the unobservable constant global phase, using the truth only
for scoring. RMS errors are measured inside a 77 nm radius; full square fields
are plotted. The script checks the cached Gaussian operator against the public
`evaluate_gaussians` renderer (relative intensity error < 2×10⁻⁴), requires phase
RMS < 0.04 rad and requires the correctly calibrated fit to explain the data
better. The renderer builds optics in float64 and stores inverse operators in
complex64. Optimization status and residuals are recorded without suppressing
iteration limits.

## Saved 256 × 256 run

The four detector intensity arrays have shape **(4, 256, 256)** at 55 µm pitch.
The calibrated fit converged after 1,590 iterations.
Its piston-aligned phase RMS is **0.00684 rad** inside the
77 nm scoring radius; complex exit-wave relative error is
**0.613%**. The 20% IL1-step-error
comparison reached its 3,000-iteration limit and has phase RMS
**1.803 rad**. This comparison is the
reported local optimization result, not a global lower bound on miscalibrated fits.

An independent fifth setting (IL1 −15%, excluded from fitting) is predicted on
the same 256 × 256 grid with relative intensity error
**0.00227%**, using the public native Gaussian
renderer. Run `verify.py` to repeat this check; `validation.json` stores the
result. Validation also checks the saved detector dimensions and native rays
converging at each marked lens conjugate. The presentation notebook embeds all
regenerated figures.

## Context

Focal-series exit-wave reconstruction recovers a complex wave from intensity
measurements under known propagation conditions; see Allen et al.,
[Exit wave reconstruction at atomic resolution (2004)](https://doi.org/10.1016/j.ultramic.2004.01.012).
The exact microscope, Gaussian representation, current steps and numerical
results here are this repository's synthetic example. The original smiley
notebook attributes its recipe to [doi:10.1093/micmic/ozad021](https://doi.org/10.1093/micmic/ozad021).

## Objective conjugate plane

`figures/03c_objective_image_plane.png` shows the specimen (object plane) and
its real objective image, with three launch angles per specimen point. The
image is 188.193669 mm after the specimen, 185.893669 mm after the objective,
and 25.330287 mm before IL1, at magnification 80.8233×. Native propagation
verifies B = 0 there. Changing the downstream IL1 current leaves it fixed.
Axis labels now read `x (µm)` or `x (mm)`; the plotted coordinate convention
is unchanged. `objective_image_plane.json` records the numerical check.

## IL1 defocus shown upstream of IL1

`figures/03d_IL1_object_plane_defocus.png` overlays the fixed objective rays
with rays back-projected from the fixed detector, for the focused reference
and all four exposures. The coloured crossing is the plane imaged onto the
detector by IL1 and the remaining downstream optics. Its displacement from
the fixed objective image (purple dotted line) is defocus in this gap.
The shifts are −50.43, −25.09, −9.02 and +4.74 mm for IL1 changes of
−28%, −20%, −10% and +8%. These are intermediate-space distances, not
specimen-equivalent defocus distances. Positive means towards IL1.

`plot_il1_object_plane.py` reproduces the plot. For the downstream matrix
from the IL1 entrance to the detector, the upstream conjugate satisfies
`z_conjugate = z_IL1 + trace(inv(A) @ B)/2`. An independent native column
trace verifies B = 0 from each calculated plane to the fixed detector.
Results are saved in `IL1_object_plane_defocus.json`.
