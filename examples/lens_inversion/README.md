# NEOARM calibration, one section at a time

Start with [the section-wise notebook](neoarm_full_column_inversion.ipynb).
It reads the recorded free-lens table and explains the next experiment; it
does not execute the historical recovery jobs or claim a new microscope fit.
The original [DAC investigation](dac_lens_inversion.ipynb) and
[geometry notebook](../microscope_models/measure_microscope_geometry.ipynb)
remain unchanged.

## What the paper gives us

Pfeifer, Lupini, and Hoglund explicitly start their projector rays at the
sample plane to avoid propagating condenser errors into the projector model
(PDF p. 6). They calibrate the condenser separately, including transmission
through an aperture between C1 and C2 (p. 4), and combine focusing information
with rotation to separate effective lens strength and length (p. 5).
The reference is [arXiv:2607.29411v1](https://arxiv.org/abs/2607.29411), with
the supplied [local PDF](paper/2607.29411v1.pdf).

The experiments demonstrate useful prediction and adjustment of microscope
states. They do not establish global uniqueness of every internal parameter
or uniform sub-percent accuracy. Table II reports projector camera-length
errors of 6.3%, 16%, and 28%, while Table I includes a final condenser-lens
adjustment to recover focus. Our original 1% physical-parameter target is
therefore a stronger requirement than the paper demonstrates.

Three parts of their method transfer directly to our experiment design:

- Pairwise lens sweeps, or a scan of one lens with a small wobble of another,
  locate crossover/conjugacy regimes. The rest of the active section stays
  in the forward model because a one-lens beam-diameter curve depends on the
  other lenses.
- Rotation supplies information complementary to focusing. Their preferred
  projector measurement registers defocused specimen texture; ellipse
  orientation is an alternative, with orientation and crossover ambiguities.
- Aperture transmission constrains where the beam passes through a known
  plane. Detector clipping and source-current drift must be separated from
  genuine aperture loss.

Their critical-line equations describe simplified lens/crossover conditions.
For finite lenses and finite-emittance beams, fit the actual matrix/covariance
response around a candidate line. A vanishing diameter derivative at one
point alone does not establish an exact conjugate plane.

## Two calibrations with one sample-plane interface

The proposed division is

```text
source → CL1 → CL3 → C-mini → objective prefield
                             ↓
               sample-plane illumination
                             ↓
                    physical specimen
                             ↓
objective postfield → IL1 → IL2 → IL3 → PL1 → detector
```

The upstream calibration predicts the illumination at the specimen; the
downstream calibration predicts how a specified specimen-plane input reaches
the detector. The objective controls remain fixed and the corrector is off.
The objective prefield and postfield share one field hypothesis, sample
position, voltage, and coordinate convention. Fixing its controls does not
make the objective geometry or field profile exactly known.

For grouped mechanical coordinates, write

\[
\mathbf r_s=(x_s,y_s,\alpha_{x,s},\alpha_{y,s}),\qquad
\mathbf x_d=A\mathbf x_s+B\boldsymbol\alpha_s+\mathbf b.
\]

The internal library also uses interleaved coordinates in some APIs; adapters
must explicitly permute them and convert mechanical angles to canonical
momenta where the objective field is nonzero. The same conversion belongs
on both sides of the sample boundary.

At an image plane, the certified specimen largely measures the position block
\(A\). A calibrated angle response or indexed diffraction measurement adds
information about \(B\). A focused specimen image alone cannot determine
every internal lens length and position. Absolute geometry anchors and
measurements away from exact conjugacy are also needed to test physical
length-scale ambiguities.

Once calibrated, the downstream system serves as a measuring instrument for
the condenser. Carry its calibration uncertainty into the condenser fit.
For an optional joint refinement, combine the original independent data
blocks once, with shared objective and detector parameters. Do not add a
projector posterior as a prior and also reuse its original images.

## First experiment: the downstream section

1. **Read the actual operating trajectory.** The table contains 13 nominal
   settings from 40k to 600k; OLf, OLc, and PL1 are constant in this range.
   These are control values and magnification labels, not measured currents,
   exact magnifications, or evidence of exactly constant image rotation.
   PL1's response to changing its command needs an additional perturbation.
   The table contains no condenser settings.
2. **Begin at one or two accessible magnifications**, initially 40k and 80k.
   Record the registered specimen region, all raw lens controls, approach
   direction, exposure, detector corrections, and repeated reference frames.
   Measure scale, rotation, distortion, and their repeatability from the same
   physical multiscale specimen. Use asymmetric registered texture to resolve
   the orientation ambiguities of a perfectly periodic or square pattern.
3. **Perturb IL1, IL2, IL3, and PL1 individually in both directions**, keeping
   illumination and objective controls fixed. Add small paired scans where
   the measured responses cannot distinguish competing models. Initial scan
   sizes come from operator-approved safe ranges and detector capture, not
   from the normalized synthetic settings in the old research scripts.
   Continuous sequences help identify crossover branches; repeat the
   approach direction to measure hysteresis.
4. **Measure an angular channel.** Use calibrated specimen-plane position
   and tilt inputs when available, with their covariance. Indexed diffraction
   vectors on the same specimen provide camera-length information; they do
   not automatically supply a calibrated commanded beam tilt. Objective
   wobbling, used in parts of the paper, is outside our fixed-objective plan.
5. **Fit measured responses, then release physical parameters.** Start with
   approximate fixed geometry and one excitation gain per lens. Test a
   zero-offset proportional raw-control law first, then an affine law if
   residuals require it. Fit finite, rotating lens maps so focusing and
   rotation constrain the same excitation. Release selected lengths and
   positions only after their sensitivity can be distinguished from the
   control law, objective, and detector calibration.
6. **Predict withheld states.** Reserve mixed-lens settings and a second
   magnification before choosing the fit. Compare predicted magnification,
   rotation, centroid, and diffraction vectors with fresh measurements.
   A competing branch rejected using a frame makes that frame training data;
   obtain new held-out data for the selected model.

For an effective hard-edge lens, use integrated excitation
\(q_i(D_i)=a_iD_i\) initially and the exact paraxial finite-lens matrix.
Its Larmor rotation is \(-q_i\), while the weak-lens power is approximately
\(q_i^2/L_i\). In that limit \(1/f\), not \(f\), is proportional to current
squared. The fitted gain absorbs DAC-to-current and current-to-field
conversion, so current readback is not required. Proportionality, zero offset,
and negligible hysteresis are hypotheses to test, not instrument facts.

## Second experiment: the condenser section

Hold the calibrated projector settings fixed during each condenser sweep.
Acquire the same condenser setting through several calibrated downstream
analyser states so beam size, angular spread, and position–angle correlations
can be distinguished. For a single transverse axis, at least three independent
combinations of \(A^2,2AB,B^2\) are needed to recover its three covariance
entries; full two-dimensional covariance can retain unobservable components
under round optics, so assess the actual measurement rank.

Use a reproducible astigmatic illumination state and fit its entrance
covariance as a nuisance quantity. Its ellipse angle is an observed quantity,
not automatically the sum of condenser Larmor rotations. The stigmator's
location relative to the swept lenses determines that interpretation and
still needs confirmation. No calibrated upstream beam kick is assumed.

Complement these beam images with unnormalised aperture-transmission counts
as CL1, CL3, and C-mini are varied. A stationary aperture can already provide
an intensity-versus-lens curve; reproducible aperture translation would add
edge information, but is not required for that first test. Confirm the
aperture plane, size, and which lenses lie upstream of it. The paper's
C1–VOA–C2–C3 ordering cannot be assigned to NEOARM without checking.

Use the actual illumination integral through the aperture. In the supplied
PDF, Eq. (8) prints a transmission above unity for an overfilled aperture;
for a centred uniform disk the fraction is instead
\(\min[1,(r_{aperture}/r_{beam})^2]\). Gaussian illumination requires its
own integral. A small specimen perforation or a clipped detector does not
measure total condenser-aperture transmission. Record a dose reference and
verify collection efficiency across the sweep.

## Where the Gaussian specimen images enter

The specimen and GPU renderer remain part of the project. They test whether
scale, rotation, focus, and intensity can actually be extracted at the selected
magnification, dose, detector sampling, and specimen contrast. Millions of
Gaussian beamlets are allowed; forward renders use the explicit
`gpu_kernel_required` path, with no silent CPU fallback.

The first lens fit uses these measured observables and their covariance, so
it need not render millions of Gaussians at every optimizer step. Full-image
fitting becomes useful when it adds information beyond those observables.
Sample interpolation, Gaussian expansion, detector integration, and physical
model errors receive separate checks. In particular, several historical
condenser states failed a nonparaxial sensitivity check despite a successful
first-order statistical fit. Lower-angle measurements or a better physical
model must resolve that discrepancy before instrument-level claims.

The original numerical targets remain useful: detector discretization bias
below 0.2 experimental noise standard deviations RMS, fitted-observable bias
below 0.2 of its uncertainty, and identifiable parameter shifts below 0.1
standard deviations over two refinements. Fresh held-out targets are 0.5%
magnification, 0.5° rotation, 0.5 pixel centroid, 2% beam diameter, and one
pixel diffraction-vector error. A 1% physical recovery claim additionally
requires uncertainty and alternative-solution analysis; a favourable error
in one synthetic fit is insufficient.

The existing ray plotting APIs are retained for true-z section overviews,
height/angle basis rays, Gaussian envelopes, and SVG/PDF/300-dpi PNG export.
Every new fitted state should show both calibrated inputs and the Gaussian
basis, with its actual specimen region and identical comparison extents.

## Current implementation boundary

The simulator now has a separately verified
[40k–600k focused imaging mode](../microscope_models/IMAGING_MODES.md): ten
stored states and refinement between them satisfy specimen conjugacy,
magnification, and zero net rotation in the existing thin-round-lens model.
This supplies a consistent simulation baseline; it is not a fit to measured
NEOARM optical responses or a calibration of the recorded raw controls.

The section-wise notebook executes the recorded-control audit. The single
[projector fitting notebook](../microscope_models/projector_calibration_pipeline.ipynb)
runs a separate synthetic experiment: make MAGICAL images, measure their
responses, fit an unknown microscope, and check predictions at new settings.
It works without running another notebook.

The current protocol uses eleven lens settings: 99 calibration images, with
one ordinary image, four ±3 mrad tilt images and four ±10 nm beam-shift images
per setting. It retains a 100 nm illumination radius and the known nominal
MAGICAL intensity pattern. The specimen stays fixed while the illumination
and its curved wavefront move together. Paired detector displacements measure
`A @ position_change + B @ angle_change`.

One fitting procedure estimates 16 optical parameters from displacements,
then all 19 parameters jointly. Both stages retain hard engineering bounds
and remove every soft penalty. Image analysis receives no lens parameters;
the fit receives measured responses, covariances and independent starting
guesses. The notebook compares 20 starts and predicts 72 separate images.

The [reduction study](../microscope_models/DATASET_REDUCTION_NOTES.md) records
how the settings were chosen, uncertainty, failures and noise checks. The
[older adaptive study](../microscope_models/ADAPTIVE_CALIBRATION_NOTES.md)
records the original 334-image result. The
[lens comparison](../microscope_models/LENS_MODEL_COMPARISON.md) explains why
this experiment keeps the same finite-solenoid model. Some selected controls
still reach 3–4× reference excitation. This is a synthetic calibration test,
not evidence that these settings or this control law are valid on a real
NEOARM. Coherent specimen contrast is not simulated here.

The general cubic sample, GPU field evaluation,
finite-lens optics, detector likelihoods, covariance tools, and ray plotting
remain in `src/temgym_core`, with their regression tests. The historical
synthetic column is now explicitly a test fixture in
`tests/fixtures/neoarm_synthetic_column.toml`.

See [research recovery and evidence](archive/README.md) for the checkpoint,
historical results, and known test-suite limitations.
