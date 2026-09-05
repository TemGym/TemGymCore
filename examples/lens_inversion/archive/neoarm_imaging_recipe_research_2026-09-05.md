# Toward a short NEOARM calibration experiment

5 September 2026. The goal remains recovery of the projector and condenser
systems from available NEOARM controls, with Gaussian-rendered specimen images
and held-out experiments. The current numerical evidence supports an
imaging-first design investigation; it does not establish a completed
instrument inversion.

## What the latest checks change

GPU specimen images and their seven-parameter fits now include pixel-area
integration and a detector-blur halo. They pass at 40k and 80k, using 10.5 and
2.42 million Gaussian beams. This establishes conditional image metrology,
not recovery of individual lenses.

The last-lens experiment now uses signed responses extracted from 207 noisy
detector images. At one million incident electrons per frame, PL1 length/gap
uncertainties are approximately 8–9%. A 64-start weighted-QR search finds the
same best solution without initializing the upstream coefficients at truth;
the remaining work is measurement design and physical-family discrimination,
not simply increasing the start count.

A new joint calculation shares the objective field between its upstream and
downstream halves and releases all 34 optical parameters, source covariance,
and calibration nuisances. Wider PL1 excitation and a roughly 2 nm calibration
probe improve geometry information, but several positions and thicknesses
remain weak. Selected-frame numerical refinement passes; extending the
truncated objective tails still changes rotation by about 3.75°. Field extent
and overlap therefore remain physical-model questions.

Both initializers read the approximate geometry notebook as data, without
modifying it. The manual annotations place CL1, CL3, and the condenser aperture
about 510.4, 429.2, and 420.8 mm before the sample, rather than the archived
fit's 125, 85, and 75 mm. Connected 40k/80k states have been constructed on this
geometry with fixed objective controls; lens thicknesses and fields remain
provisional. Historical calculations below are explicitly separated from it.

The earlier 0.24% condenser fit used an unsupported independent angle law,
`offset - sum(q)`, while releasing a general entrance covariance. Focusing can
change that covariance's ellipse orientation in addition to Larmor rotation.
Its numerical success is not physical condenser recovery.

The stigmator location and aperture translation capability remain unconfirmed
by the user. The calculation consequently treats them as experimental choices
to establish, rather than available calibrated measurements.

## The imaging-first candidate

Use **40k for calibration and 80k for a second-scale check**, initially using
the specimen's multilayer region and vacuum opening. At 55 µm detector pitch,
the nominal 23 nm repeat projects to 16.7 and 33.5 pixels; the specimen fields
are 352 and 176 nm wide. Both scales resolve several intervals without relying
on lattice-image contrast. Actual certificate distances and registered region
coordinates replace these nominal values.

1. **Establish geometry and control coordinates.** Rebuild the entire column
   from the measured diagram, record real free-lens values at the two imaging
   settings, and declare safe local changes and approach directions. Keep the
   objective controls fixed and corrector disabled. The source reference plane
   is a convention unless independently located; a transported source
   covariance can exactly compensate its displacement.
2. **Measure the last-lens response first.** Begin with roughly nine connected
   PL1 settings and all four signed sample-position/angle responses. Measure
   the response uncertainty before enlarging the scan. The PL1 tomography
   derivation below separates the last lens from an unknown upstream map, but
   requires demanding precision in the tested relay. Once resolved, PL1 can
   analyse IL3, then earlier lenses. Known specimen distances anchor scale;
   calibrated input responses retain their covariance. Record intermediate
   observations, since endpoint approach metadata do not resolve a branch.
3. **Test condenser illumination in three achievable imaging states.** For each
   CL1/CL3/C-mini setting, keep the entrance illumination and aperture condition
   unchanged while recording nominal focus and two calibrated projector states.
   The measured-geometry initializer supplies only about −0.067/+0.064 µm
   `B/A` within ±0.05 synthetic IL1/PL1 controls, or −0.287/+0.244 µm within
   ±0.2 controls at 40k. The earlier ±100 µm target belonged to a different
   synthetic column. These are verified paths, not global bounds on achievable
   diversity. Fit the transported image
   covariance or full intensity, including detector truncation, instead of an
   independent total-Larmor tag. These are optical projection changes, not
   assumed stage motion or a known DAC-to-defocus scale.
4. **Supply an intermediate-plane length reference.** The current diagnostic
   tests three centred aperture radii with independent aperture-out dose
   references, without assuming translated scans. Radius calibration, aperture
   location/centring, availability of those apertures, and full transmitted-flux
   capture still require confirmation. If reproducible edge traversal becomes
   available, it is an alternative measurement with its own translation gain,
   direction and cross-coupling nuisances; it is not a confirmed input now.
5. **Choose the smallest validated dataset.** Start computationally with the
   necessary measurements, then remove states and repeats while preserving
   blind recovery and held-out prediction across perturbed truths, calibration
   errors, and noise realizations. The retained 80k states and multivariate lens
   perturbations stay outside fitting and experiment selection. Select extra
   measurements from the remaining weak directions, not from a predetermined
   large frame budget.

Diffraction is a candidate additional angular measurement if imaging diversity
cannot be calibrated accurately enough. In the tested model it does not
replace the intermediate-plane geometry reference. A 40 nm opening in the
current synthetic atlas is also not a vacuum window for a 60–90 nm RMS beam;
the real opening must contain the beam or its clipping must enter the forward
model. Even with open illumination, finite-detector inference is necessary:
at 40k/80k, six-RMS full-beam containment permits only about 59/29 nm RMS.

The digital-twin paper motivates lens sweeps, rotation, and aperture throughput,
but its detailed projector examples and current-based calibration do not
establish the NEOARM condenser measurement contract above.
[Reference paper](https://arxiv.org/html/2607.29411v1)

## Historical information audit in the archived geometry

All rows below retain unknown entrance/source/objective-prefield quantities.
They are **local, optimistic covariance-summary calculations in the archived
synthetic geometry**, not fits to condenser detector images. Standard
uncertainties use the archived physical parameter scales; they are not
unqualified relative errors on every named coordinate.

| Measurements | Local condenser target rank | Worst scaled 1σ |
|---|---:|---:|
| Translated aperture + in-focus spatial ellipses | 11/12 | Unbounded |
| Same, plus calibrated ±100 µm imaging diversity | 12/12 | 0.01476 |
| Same, plus ±100 and ±200 µm | 12/12 | 0.01453 |
| Translated aperture + spatial/angular ellipses | 12/12 | 0.01012 |
| Imaging diversity or spatial/angular data without aperture reference | 10/12 | Unbounded |
| Translated aperture + imaging diversity, shared translation gain fitted without its prior | 12/12 | 0.02696 |
| Centered aperture total counts at three known radii + imaging diversity | 12/12 | 8.14 |

The last row is **8.14, not 0.00814**. In the tested excitation grid, centered
total intensity contains far less useful information than an edge traversal;
earlier optimistic centered-aperture results additionally assumed an image of
the beam shape at the aperture plane. A different excitation grid or a separate
geometry measurement still merits testing, so this is not a universal
impossibility claim.

The JAX-only control solver constructed actual IL2/IL3 changes within
±0.05 of the archived synthetic controls at both 40k and 80k, preserving
magnification and reaching `B/A = −100, 0, +100 µm`. Maximum dimensionless
forward-constraint error is below 3e−13. It obtains the transfer matrix by
differentiating a single ray, then differentiates that matrix with respect to
controls. This proves control feasibility for that synthetic column; it
supplies neither acquired branch-continuation images nor a NEOARM command list.
The geometry-corrected control results below supersede its ±100 µm acquisition
target for the current first-branch initialization.

## Measured-geometry initialization and a detector-image model

`neoarm_measured_initialization.py` constructs a chain of real intermediate
image planes in the diagram's drift gaps, solving each finite-lens conjugacy
condition on its first positive phase root. One shared relay coordinate gives
40k and 80k with every downstream absolute lens phase below π. The full matrix,
obtained by differentiating a single ray, reaches the requested magnifications
within 6e−13 relative error and `|B/A| < 2e−18 m`. A nine-state connection keeps
the objective fixed; it is a computed trajectory, with no acquired branch data.

| 40k IL1/PL1 control half-width | Verified B/A endpoints, µm |
|---|---:|
| 0.05, dimensionless synthetic coordinates | −0.06715, +0.06449 |
| 0.20, dimensionless synthetic coordinates | −0.28730, +0.24423 |

The finite-pixel information calculation below now uses this smaller diversity
and a physical source. Successful ±100 µm control design in the old geometry
cannot justify that measurement in the new model.

The new `gaussian_image_likelihood.py` predicts pixel-area-integrated counts
from a transported Gaussian phase-space distribution. It includes the detector
affine and an illumination halo for detector blur, retains electrons lost
outside the detector, and fits six image coordinates jointly with their full
local covariance. It compares two pixel-quadrature orders before admitting a
fit. Ellipse angle is encoded by the covariance, not counted as extra data.
Three isotropic projection planes contain nine independent second-moment
combinations; the antisymmetric position/angle cross-covariance remains
unobserved by these projections alone.

An independent GPU calculation propagates explicit coherent Gaussian modes
and sums their intensities to represent one partially coherent source. At
2,401 modes the worst detector discrepancy across three engine-test states is
1.59e−6 noise σ RMS, and the largest actual six-parameter image-fit bias is
1.42e−6 estimated σ. Independent two-dimensional pixel integration agrees with
the conditional-normal integration. These tests validate Gaussian illumination
images, with no specimen or inserted aperture; their synthetic ±100 µm maps
are engine tests, not achieved NEOARM projection states. Independent Poisson
and read noise after the expected-image blur remain a detector assumption.

New ray diagrams use the approximate measured positions, first-branch optics,
true-z overview, three physical zooms, both ray bases, and a Larmor strip.
The illustrated 60 nm Gaussian waist is defined at the specimen and propagated
backwards for the condenser panel; it is not the independently inferred mixed
source. The corrector is labelled disabled without inventing its axial extent.

## Fresh specimen images in the measured geometry

The same invariant atlas is propagated through the newly initialized column,
with fresh region/crop selections at the new image rotations. The independent
conjugate-image template fits magnification, rotation, specimen x/y position,
dose, illumination width, and attenuation contrast jointly. Each fit begins
with all seven coordinates perturbed from truth.

| State | Final coherent beamlets | Truth discrepancy, noise σ RMS | Noisy magnification error | Noisy rotation error |
|---|---:|---:|---:|---:|
| 40k | 10,504,081 | 0.0001128 | +0.00833% | +0.00995° |
| 80k | 2,424,249 | 0.00003922 | +0.04030% | +0.01165° |

Both modes pass two successive refinements with all seven fitted coordinates
changing by less than 0.1 estimated σ. Their final deterministic biases are at
most 0.00356σ and 0.00224σ, and noisy residuals are 1.00076σ and 0.99765σ RMS.
The final GPU re-expansion and rendering took about 22.4 seconds at 40k and
9.1 seconds at 80k in the recorded runs, excluding atlas construction and fits.
The notebook displays these exposures, fitted images, and noise-scaled
differences alongside the measured-geometry ray diagrams.

Concrete Gaussian grids and inverse-mapped detector edges lie inside the
selected views. All beams in each expansion share one input Q matrix; direct
full-matrix propagation omits only a common axial phase. A separate 80k
comparison with 64 lens slices changes intensity by 0.0000278 noise σ RMS,
but this single comparison is not a slice-refinement series. That saved
refinement series uses pixel-centre sampling and the older edge-blur
convention; the finite-pixel test below separately replaces both approximations.
Detector/certificate calibration, beam centre, and coherent
illumination are conditional inputs, while view coverage over the complete
fitting range and real-specimen contrast remain untested.

`measured_specimen.json` binds controls, matrices, source files, noise seeds,
and every refinement array to checksums. It supersedes the archived-column
images below for the current 40k/80k demonstration. Image-scale accuracy is
distinct from physical lens-parameter recovery.

### Pixel-integrated specimen images and fits

`neoarm_finite_pixel_specimen.py` evaluates the actual propagated coherent
Gaussian fields on detector-area quadrature points, sums their amplitudes,
and integrates intensity over each pixel. A physical illumination halo is
blurred before cropping to the detector. The regular-grid accelerator checks
Gaussian support and compares its retained neighbours with exhaustive GPU
evaluation; broad or irregular bundles require the general renderer.

The independent pixel-integrated cubic predictor and Gaussian images differ
by less than 0.0064 in total whitened image L2 norm at both magnifications.
`neoarm_finite_pixel_metrology.py` then fits all seven image coordinates from
perturbed initial values, using Gaussian pixel-order-16 images and a separate
pixel-order-8 cubic predictor.

| State | Largest deterministic parameter bias | Noisy residual, σ RMS | Magnification error | Rotation error |
|---|---:|---:|---:|---:|
| 40k | 0.00302σ | 0.99874 | −0.01359% | −0.01487° |
| 80k | 0.00221σ | 1.00377 | +0.02992% | +0.01742° |

The largest fitted pixel-order-8/16 parameter change is 0.000106σ. These tests
admit terminal-detector intensities and conditional image metrology. A stricter
complex-field comparison between evaluation backends fails its 1e−9 relative
threshold: detector carrier phases reach approximately 3.4e8 rad, with field
differences up to 1.1e−8 while corresponding sampled intensity differences
remain below 2.5e−10 noise σ. Further propagation of those complex arrays is
not certified. The new reports retain that failed diagnostic explicitly;
neighbour-radius refinement is separately converged. Full fitting-range view
coverage and real detector/specimen calibration remain outstanding.

## The earlier independent-prefield information test

The measured-geometry audit fits out all ten entrance-covariance coordinates,
source and aperture centres, the source reference plane, and the independent
objective-prefield parameters. Its image information comes from derivatives
of expected 256 × 256 pixel counts, including clipping and blur, with per-image
centre and dose fitted as nuisance quantities. No independent ellipse-angle
tag is added to those pixels.

That design uses 28 condenser states, three achievable imaging
projections, and centred total-aperture counts at three hypothetical known
radii. Independent aperture-out dose references bring the count to 252 frames
before calibration or held-out measurements. Aperture throughput assumes total
flux capture; the corresponding inserted-aperture wave images remain untested.
The aperture plane, 0.2% projector-scale/diversity calibration, and 0.5% radius
priors are conditional inputs, not established instrument capabilities.

| Conditional measurements or knowledge | C-mini axial-position 1σ |
|---|---:|
| Imaging and aperture counts; prefield unknown | 56.9 mm |
| Same, plus an idealized calibrated angular-image plane | 42.9 mm |
| Imaging and aperture counts; only prefield position known exactly | 2.68 mm |
| Imaging and aperture counts; entire prefield known exactly | 0.241 mm |

The angular plane is an upper-bound measurement calculation, not a designed
NEOARM diffraction state. The last two rows do not include it. Knowing the
entire prefield reduces the largest condenser relative thickness uncertainty
to 0.82%; it identifies the dominant coupling, but supplies no justification
for fixing an unknown field. With prefield and aperture position free, common
axial displacement remains unobservable. Even when numerical rank is full,
the large position uncertainties prevent a useful physical calibration.

The shared-objective model below joins the prefield and postfield into one
continuous magnetic field. JEOL describes the condenser-objective as one
field peak, with the specimen inside it; its upstream and downstream sections
perform different optical functions. [JEOL condenser-objective description](https://www.jeol.com/words/emterms/20121023.034459.php)
The draft's independently adjustable halves, including their separate signs,
are therefore a surrogate that needs comparison with shared-field models.
Downstream measurements may then constrain the prefield through common field
parameters. This is a hypothesis to test with possible axial asymmetry and
uncertain geometry released, not an assumption that the field is symmetric or
known. Calibrated mechanical tilts at the immersed specimen also require an
explicit conversion to canonical ray coordinates.

More frames or diffraction alone do not resolve that tested weak family well.
Its uncertainties are conditional on the downstream projection calibration;
the new joint calculation releases the downstream optics too.

## Shared immersed objective: implementation and current limits

`immersed_objective.py` supplies one globally normalized field with fitted
strength, peak position, width, and bounded asymmetry. `neoarm_shared_objective.py`
combines it with the seven controlled lenses and obtains each transfer matrix
by differentiating a single input ray. Objective controls remain fixed across
connected 40k/80k states; the two objective halves share four parameters.

Physical specimen tilts are converted to canonical momenta using the current
fitted field. With signed electron charge Q and k=QB/(2p), the normalized
momenta are πx=αx−ky and πy=αy+kx; the conversions cancel when the complete
column is composed. Gaussian phases and curvatures remain canonical, while
calibrated position/tilt observations retain their physical coordinates and
full response matrices. [Canonical momenta in a solenoid, Eq. 2](https://arxiv.org/html/2006.12948v2)

The new `neoarm_shared_objective_audit.py` jointly releases all 34 optical
parameters, ten source-covariance coordinates, alignment, dose, detector-scale,
sample-state calibration, and aperture nuisances. It uses actual transported
Gaussian pixel information and recorded synthetic lens controls, with no
independent ellipse-angle tag. The source-reference-plane/covariance gauge
remains. A fixed astigmatic entrance distribution is a synthetic source
assumption, not evidence locating the instrument's stigmator.

The selected design uses a separate calibration probe with 1.825/2.190 nm RMS
radii and nine wide, same-sign first-branch PL1 excitations at each nominal
magnification. Removing the wide PL1 states from the same information matrix
increases the largest projector-position uncertainty from 40.6 to 479.7 mm;
with conditional 1 mm detector/aperture-plane metrology, from 4.36 to 51.8 mm.
This is evidence of insufficient measurement diversity, not an exact projector
null. With that metrology prior and optimistic aperture total-flux data:

| Parameters | Approximate local standard deviation |
|---|---:|
| Projector centres | 2.4–4.4 mm |
| Projector effective lengths | 1.0–3.4% |
| Condenser centres | 1.5–9.3 mm |
| Condenser effective lengths | 4.5–11.8% |
| Objective peak position / width | 23 µm / 3.7% |

These are local information estimates, not noisy recovery or all-parameter 1%
acceptance. The broad multilayer exposure and narrow probe calibration serve
different measurements and may use different condenser settings on the same
specimen. The diagnostic now contains 222 aperture-out images, 42 aperture-in
measurements and 42 independent dose references before repeats or held-out
data. It is not yet a compact recipe. Input calibration to 0.2%, aperture-radius
calibration to 0.5%, stable alignment within each basis group, full transmitted
flux capture, and an exposure scheme supporting the simulated 20-million-electron
images remain explicit experimental conditions, not confirmed capabilities.

The field integration also needs its own noise-relative checks. In
`shared_objective_convergence.json`, 256 uniform slices per side change a fitted
image coordinate by 35.4σ relative to the fine-grid reference, whereas 4096→8192
and 8192→16384 pass the tested nominal-probe image/observable gates. For the
complete selected wide-sweep/narrow-probe frame set, a fixed tangent mesh needs
8192→16384→32768 slices per side: the largest image-coordinate Mahalanobis
changes are 0.07139σ and 0.01785σ. Pixel quadrature 32→64→128 gives total
whitened pixel discrepancies of 0.06940σ and less than 9e−10σ. These checks pass
for this design at truth, not for the complete fitted parameter box.

Field truncation remains a distinct model error. Extending the initial
±50 mm window to [−75,+150] mm while preserving the physical profile amplitude
changes rotation by 3.75°, magnification by −0.262%, and detector images by up
to 0.852 noise σ RMS. No refitting concealed that change. An inference claiming
physical field parameters must account for those tails and any overlap with
other lens fields, or establish an adequate effective-field model on held-out
measurements. Successful discretization of the truncated model does not supply
that evidence.

### Whole-column field integration

`axial_field_column.py` and `neoarm_objective_tail_overlap.py` implement a
continuous-column alternative. The signed objective and controlled-lens fields
are summed before the focusing term is squared, preserving the overlap
cross-terms. Fourth-order commutator-free propagation uses explicit moving
lens-body/gap boundaries, so JAX differentiates physical edge motion without a
discontinuous field-membership mask. Mechanical/canonical conversions use the
actual field at the source, specimen, and detector.

Across 55 probe frames, 64→128→256→512 steps per segment give maximum joint
image-coordinate changes of 0.02672σ, 0.001769σ, and 0.0001123σ; 32 steps fail
the 0.2σ gate. The full 34-parameter GPU Jacobian is finite. With the original
parameters and controls fixed, continuing the assumed objective profile across
the whole column changes rotation by +5.25°, magnification by −0.695%/−0.609%
at 40k/80k, and nominal 80k beam capture to approximately 90%. These are model
sensitivity results without refitting. The continued profile is not a measured
NEOARM far field, and the narrow calibration-probe states and full fitting
parameter range need their own integration tests.

## A smaller downstream experiment: last-lens tomography

The new `neoarm_last_lens_tomography.py` tests a way to separate PL1 from the
unknown optics before it. Measure the full signed 2×4 position response H to
calibrated specimen inputs (x,y,αx,αy), in physical units, while sweeping only
PL1. For an effective hard-edge lens of length L followed by a detector gap d,
write s=d/L. With sinc(q)=sin(q)/q, the derotated response is

\[
R(q)H(q)=\cos q\,W_0+q\sin q\,W_1+\operatorname{sinc}(q)W_2,
\qquad W_0=-W_1/s+sW_2.
\]

The four-dimensional transfer-volume identity then gives

\[
d=\sqrt{\det\begin{pmatrix}W_1\\W_2\end{pmatrix}},\qquad L=d/s.
\]

This uses both transverse planes. An unknown magnetic shear at the immersed
sample has determinant one and does not invalidate the identity; neither does
a constant unknown transverse frame rotation. Pixel and input-step scaling
must be removed before taking the determinant. No known upstream map or known
sample field is supplied to the fit.

The saved example releases 19 coordinates: two arbitrary upstream coefficient
blocks, s, and excitation bias/slope. Its 25-point same-sign first-branch scan
uses 5 nm position and 1 mrad angle inputs. One seeded noisy local fit gives:

| Quantity | Fitted error | Conditional standard error |
|---|---:|---:|
| PL1 effective length | +0.168% | 1.034% |
| PL1-to-detector gap | +0.272% | 1.002% |

The assumed response uncertainty is **0.0001 detector pixel**. At 0.001 pixel,
the corresponding geometry uncertainties are about 10%. The largest tilt
response in this relay is only 0.00325 pixel per 1 mrad input. These precisions
are assumptions on already measured responses, not results from rendered noisy
detector images. The fit starts near the generating branch, and opposite
polarity/adjacent branches have not been excluded.

Start with roughly nine PL1 settings and signed responses, then assess
repeatability and conditioning before extending the scan. Once PL1 is
calibrated, two sufficiently independent PL1 analyser settings can reconstruct
the full upstream map for each IL3 state; a test verifies that algebraic step.
Noisy multi-lens reconstruction and its accumulated calibration uncertainty
remain to be demonstrated. This route concerns downstream lenses only:
CL1/CL3/C-mini still require separate condenser measurements.

### Image-derived precision and initialization

`neoarm_last_lens_images.py` renders reference and ±4 calibrated-input frames
for 25 candidate PL1 excitations. The specimen input has 2 nm × 1.5 nm RMS
intensity radii, and the detector receives one million incident electrons per
frame with Poisson statistics, 0.6-electron read noise, finite pixels, and a
0.65-pixel PSF. Two subpixel crossover states fail the declared width gate;
all 207 frames at the other 23 settings pass six-parameter image fitting.
Actual Gaussian-beam propagation and field evaluation independently reproduce
the narrowest and broadest admitted images, with noise-whitened L2 differences
below 2e−8 and pixel-quadrature differences below 7e−6.

Central differences remove the common reference from the signed response;
their full covariance uses the marginalized mean blocks of each image fit.
Response uncertainties range from 0.000410 to 0.001975 pixel, with essentially
complete detector capture in this experiment. The local fit gives length/gap
errors of +3.55%/+7.22%, versus expected uncertainties of 8.70%/8.30%; adding
four shared 0.2% input-scale priors changes the latter to 8.71%/8.30%. The
source preparation, exact synthetic pixel-noise variance, and within-group
alignment stability remain conditional. The readout model describes accumulated
counts, not a validated Medipix rate, counter-depth, or exposure schedule.

`neoarm_last_lens_branch_search.py` profiles the sixteen linear upstream
coefficients by QR in that same covariance metric. Its 64 initializations span
eight phase-bias intervals, both excitation-slope signs, and two length/gap
ratios, with no generating upstream map supplied to initialization. All starts
converge in the current run, and the best fit matches the local image result.
Standard errors evaluated at that fitted solution are 8.17%/7.72%.

The nearest distinct relaxed-model minimum has the same excitation law and
only Δχ²=1.70, but a 1.58 m effective lens and 2.24 m detector gap. That solution
exceeds the approximate 0.719 m sample–detector span; its rejection alone does
not exclude a nearby, geometry-constrained minimum. Moreover, the nineteen-
coordinate response model retains arbitrary volume-preserving upstream maps,
whereas physical magnetic maps obey a stronger symplectic condition in
canonical coordinates. Geometry-constrained profiles and physical-map fits
are required before calling a relaxed competitor a physical ambiguity.

Dose-only scaling predicts roughly 76 million/69 million incident electrons
per frame for 1% length/gap precision, assuming unchanged shot-noise-limited
conditions. This is an information estimate, not a hardware recommendation.
The more direct design test changes the fixed IL1/IL2/IL3 relay to improve the
calibrated tilt information before repeating the PL1 scan.

### Relay conditioning at the same dose

The first small-matrix search, followed by exact finite-image information,
finds a relay with 2.38%/2.24% expected length/gap uncertainty. Its narrowest
selected beam is only 0.250071 pixel RMS, close to the 0.25-pixel gate, so it is
retained as a boundary-sensitive candidate in `last_lens_design.json`.

The separate `last_lens_robust_design.json` uses nominal widths of at least
0.35 pixel to select eighteen PL1 states, then keeps those states fixed.
Its IL1/IL2/IL3 phases are [2.3036457, 2.7378026, 0.9596225] rad; these are
synthetic control coordinates, not NEOARM commands. Exact image information
predicts 3.43%/3.15% nominal uncertainty with the same one-million-electron dose,
versus 8.70%/8.30% for the original relay. Nine scenarios—nominal, six signed
0.02-rad single-relay perturbations, and ±10% quantum-valid source-size changes—
give 2.94–4.03% length and 2.70–3.73% gap uncertainty.

All 24 combined relay-corner/source-size width and capture checks retain the
same eighteen states; their smallest width is 0.251355 pixel. These finite
scenario checks are not a certificate for the continuous parameter box, and
the redesigned scan has not undergone noisy recovery or held-out testing.

## Collecting the condenser aperture signal

The direct aperture-in/out ratio in the original 40k/80k off-band settings is
not the total transmitted fraction. Geometric finite-camera integration finds
only 0.76–1.82% aperture-out capture at 40k and 0.19–0.46% at 80k in the affected
states. Those detected rays already pass the smallest aperture, so the measured
ratio is nearly one while true aperture transmission is approximately 0.46–0.51.
The earlier total-flux Fisher calculation therefore remains an optimistic
measurement bound, not an executable acquisition for those settings.

`neoarm_aperture_capture.py` constructs a separate aperture-imaging collection
state with fixed objective parameters. Its aperture magnification is 120 and
the aperture-to-detector B-block norm is at most 1.14e−12 m, placing the nominal
10/20/30 µm radii at 1.2/2.4/3.6 mm on the detector. Across fifteen illumination
families, computed aperture-out capture is 0.999998916–1.0. C-mini and projector
controls change, while the aperture-plane incident maps remain unchanged
because these lenses are downstream of that plane.

This collection state has approximately 2002× specimen magnification and
0.349–0.721 µm RMS specimen beam radii. Six-RMS containment requires a clear
specimen radius of at least 4.328 µm plus pose/alignment margins; the largest
aperture image at the specimen has radius 1.825 µm. The current 40 nm-radius
synthetic perforation would pass only 0.171–0.590% of the aperture-out beam.
The collector therefore requires a larger confirmed vacuum path on the same
physical specimen, without changing specimen identity between acquisitions.
The existing atlas has not been altered to conceal this requirement.

Aperture-image radii could also supply a measured aperture-to-detector scale,
without a translated aperture or an independent ellipse-angle tag. Full wave
convergence, bores/other-aperture clearance, control repeatability, and fitting
these actual images remain requirements; the saved collection result is a
first-order capture and conjugacy calculation in the truncated shared-field
model, not acceptance of the condenser experiment on the instrument.

## Verified specimen-image calibration in the archived column

These earlier images use a different column. The finite-envelope harmonic
estimator gave 3.08/2.62σ frequency bias at 40k/80k because the registered crops
contain nonperiodic structure; those harmonic measurement blocks remain
excluded. The exact cubic-object template includes that structure and gave
the conditional calibration results below.

The template fits magnification, rotation, specimen x/y position, dose, beam
waist, and attenuation contrast jointly, using the saved or freshly refined
Gaussian images as data. Its initial seven parameters are deliberately
perturbed from the nominal configuration; this is a local image-calibration
test, not blind lens recovery. The independent conjugate-image formula uses
the same cubic object without Gaussian re-expansion.

| State | Selected Gaussian beams | Truth image discrepancy, noise σ RMS | Noisy magnification error | Noisy rotation error |
|---|---:|---:|---:|---:|
| 40k | 10,504,081, freshly rendered on GPU | 0.0001269 | −0.0157% | −0.0419° |
| 80k | 606,841, archived GPU render | 0.0005347 | −0.0441% | −0.0134° |

The 40k grid keeps the cubic level and physical extent unchanged while
increasing the number of Gaussian beams. Its final two refinement transitions
change all seven image-calibration parameters by at most 0.05933 and 0.009716
estimated standard deviations, satisfying the 0.1σ criterion. Maximum final
deterministic bias across all seven parameters is 0.002917σ at 40k and
0.002663σ at 80k; noisy residuals are 1.00694σ and 1.00160σ RMS. The final
10.5-million-beam GPU re-expansion/propagation/render run took approximately
32 seconds in the measured run; atlas construction and fitting are additional.

The notebook displays the noisy exposures, fitted images, and noise-scaled
difference maps, exported as SVG, PDF, and 300-dpi PNG. The earlier failing
40k image is preserved in the report alongside its accepted refinements.
Certificate, detector, and input-centre calibration are exact in this test;
pixel-centre detector sampling and coherent illumination remain modelling
assumptions. Full fitting-box view coverage, pixel-area integration, partial
coherence, and calibration uncertainty still need their own tests before
real-instrument use.

## Computational handoff and remaining proof

- `neoarm_imaging_recipe.py` provides the single-ray Jacobian, JAX-native
  bounded control design, specimen-scale table, and measured-geometry import.
- `neoarm_imaging_information_audit.py` reproduces the measurement ablations,
  nuisance profiling, corrected orientation tests, and aperture alternatives.
- `neoarm_periodic_render_audit.py` checks harmonic specimen metrology against
  actual saved Gaussian images; failure is retained as evidence.
- `neoarm_conjugate_sample_metrology.py` fits the exact cubic object and
  reproduces the successful GPU-refined seven-parameter image calibration.
- `neoarm_measured_initialization.py` rebuilds conjugate 40k/80k states and
  bounded imaging-diversity paths on the diagram geometry.
- `neoarm_mixed_gaussian_validation.py` checks the new finite-pixel condenser
  image model against propagated incoherent Gaussian modes.
- `neoarm_measured_ray_review.py` exports the measured-geometry ray diagnostics.
- `neoarm_measured_specimen.py` renders and refines both current imaging states
  and refits the seven image-calibration parameters on the GPU.
- `neoarm_measured_information_audit.py` evaluates the actual finite-pixel
  condenser information and the objective-prefield/aperture ablations.
- `neoarm_finite_pixel_specimen.py` and `neoarm_finite_pixel_metrology.py`
  integrate the GPU Gaussian specimen images over finite pixels and fit the
  independent cubic-object image model.
- `neoarm_shared_objective.py` initializes the shared immersed-field column;
  `neoarm_shared_objective_convergence.py` tests discretization and field tails.
- `neoarm_shared_objective_audit.py` releases all optical, source and calibration
  parameters in a coupled finite-pixel information calculation.
- `neoarm_last_lens_tomography.py` tests the last-lens response identities and
  a noisy local fit with unknown upstream coefficients and excitation law.
- `neoarm_last_lens_images.py` generates and fits the finite-detector PL1
  calibration images; `neoarm_last_lens_branch_search.py` tests initialization
  and alternatives with their image-derived response covariance.
- `neoarm_objective_tail_overlap.py` tests whole-column summed fields with
  the differentiable fourth-order integrator in `axial_field_column.py`.
- `neoarm_last_lens_design.py` and `neoarm_last_lens_robust_design.py` compare
  fixed upstream relays using profiled image information and explicit scenarios.
- `neoarm_aperture_capture.py` checks finite-camera aperture throughput and
  constructs the separate aperture-imaging collection state.

Run the control and information checks explicitly; opening the review
notebook does not start them or launch a bootstrap:

```bash
OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
  examples/lens_inversion/neoarm_measured_initialization.py \
  --output examples/generated/neoarm_inversion/imaging_recipe/measured_initialization.json

OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
  examples/lens_inversion/neoarm_measured_information_audit.py \
  --information --pixel-information --off-band-q-steps .015 .05 --angular-information \
  --output examples/generated/neoarm_inversion/imaging_recipe/measured_off_band_pixel_information_audit.json

OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
  examples/lens_inversion/neoarm_measured_specimen.py \
  --output examples/generated/neoarm_inversion/imaging_recipe/measured_specimen.json

OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
  examples/lens_inversion/neoarm_mixed_gaussian_validation.py \
  --output examples/generated/neoarm_inversion/imaging_recipe/mixed_gaussian_validation.json

OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
  examples/lens_inversion/neoarm_finite_pixel_specimen.py \
  --output examples/generated/neoarm_inversion/imaging_recipe/finite_pixel_specimen.json

OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
  examples/lens_inversion/neoarm_finite_pixel_metrology.py \
  --output examples/generated/neoarm_inversion/imaging_recipe/finite_pixel_metrology.json

OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
  examples/lens_inversion/neoarm_shared_objective_audit.py \
  --grid-points 33 --state-count 8 --quadrature-order 32 \
  --angle-step-rad .001 --projector-wobble .2 \
  --field-mesh tangent --slices-per-side 8192 \
  --wide-pl1-scan --check-forward --calibration-probe-rms-m 2e-9 \
  --information-output examples/generated/neoarm_inversion/imaging_recipe/shared_objective_information_narrow_probe.npz \
  --output examples/generated/neoarm_inversion/imaging_recipe/shared_objective_information_narrow_probe.json

OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
  .venv/bin/python examples/lens_inversion/neoarm_last_lens_tomography.py \
  --output examples/generated/neoarm_inversion/imaging_recipe/last_lens_tomography.json

OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
  .venv/bin/python examples/lens_inversion/neoarm_last_lens_images.py \
  --state-count 25 --incident-electrons 1e6 \
  --output examples/generated/neoarm_inversion/imaging_recipe/last_lens_images.json

OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
  .venv/bin/python examples/lens_inversion/neoarm_last_lens_branch_search.py \
  --input examples/generated/neoarm_inversion/imaging_recipe/last_lens_tomography.json \
  --image-report examples/generated/neoarm_inversion/imaging_recipe/last_lens_images.json \
  --output examples/generated/neoarm_inversion/imaging_recipe/last_lens_image_branches.json
```

The commands initialize conjugate states, calculate image information, validate
GPU specimen/mixed-source images, fit finite-pixel specimen images, and test the
shared-field and PL1 recipes. The finite-pixel metrology command consumes the
preceding render's checksummed arrays. The shared-field command records its
explicit mesh, quadrature and acquisition settings; the coarse default
initialization is not admitted as a converged inversion model.
Fresh Gaussian detector rendering continues to require
the GPU, with no fixed beam-count ceiling. The new least-squares routine uses
JAX derivatives and augmented-SVD damped steps, retaining physical bounds and
all parameter directions without SciPy optimization.

Completion still requires validation of the shared objective's physical field
extent and recovery around its measured-geometry initialization; image-based
projector and condenser likelihoods with calibration uncertainty; Gaussian and
independent-grid convergence on their complete parameter envelope; blind,
multistart coupled recovery; and held-out magnification, rotation, centroid,
diameter, and specimen-image predictions. Alternative excitation branches need
measured discrimination or equivalent-solution profiles. None of those
requirements is replaced by a local rank or optimizer-success flag.
