# NEOARM inversion: implementation audit

Reviewed 5 September 2026 against the multiresolution cubic-sample plan.

Historical follow-up: the [truth audit and recovery report](NEOARM_RECOVERY.md) records a
successful **conditional probe/condenser fit**, including the separate
astigmatic-beam/aperture data. A subsequent [image-contract and geometry
audit](NEOARM_IMAGING_RECIPE.md) found that the fitted astigmatic-angle law is
not valid for the released general entrance covariance; this remains a
stipulated-summary benchmark, not demonstrated physical condenser recovery.
The newer image-derived conditional fit below does not use that angle law.
The original full specimen-feature likelihood fails at synthetic truth and
remains excluded. Historical results below retain their original assumptions.

## Bottom line

The GPU specimen-image calculation works, and the last-lens fit now enforces
physical magnetic optics using responses extracted from noisy detector images.
A fresh IL3–PL1 experiment separates the tested competing branch; successive
IL2 and IL1 batches now give a joint fit of all four projector lenses and
untouched held-out response checks. Length/gap uncertainties range from
0.9% to 5.5%, with IL1 weakest. The first stationary conditional condenser fit
now also exists, with CL1/CL3/C-mini length errors of +1.36%/+1.16%/+0.78%.
Its fresh beam-diameter predictions are within 0.47%, but their raw residuals
exceed image noise and need fitted-parameter uncertainty to explain them.
The newest fit also releases the six downstream endpoint-map coordinates,
calibrated with 26 fresh GPU images in the condenser's own column. All 29
coordinates converge at data χ²=110.45/118, and independent signed-input
checks give χ²=8.87/8. The older projector experiment uses a different objective
field, so its full calibration posterior cannot be combined with this fit.
The same joint minimum is now reached from one genuinely displaced cold
initialization without a saved fitted geometry; the other two prescribed
starts remain unresolved.
Full-column inversion still requires robust initialization, nonlinear profiles,
joint objective/projector/condenser calibration, and an instrument-valid
acquisition contract.
Synthetic controls remain computational inputs, not instrument commands.

The [finite-pixel image test](NEOARM_IMAGING_RECIPE.md#pixel-integrated-specimen-images-and-fits)
uses 10,504,081 and 2,424,249 Gaussian beams, integrated over detector pixels on
the GPU. Seven image parameters are fitted together. Noisy magnification errors
are −0.0136% and +0.0299%; rotation errors are −0.0149° and +0.0174°. These are
conditional image-calibration results, not individual-lens recovery. Detector,
certificate and beam-centre calibration are exact in this test. The images use
the approximate diagram geometry and the earlier independent-half objective
surrogate, not the new shared-field model.

The [shared-objective information test](NEOARM_IMAGING_RECIPE.md#shared-immersed-objective-implementation-and-current-limits)
uses transported finite-pixel beam images, a free source covariance, and
calibration nuisances. A wider PL1 sweep and separate roughly 2 nm calibration
probe improve geometry information substantially. Even granting 1 mm detector
and aperture-plane metrology and optimistic aperture total-flux measurements,
projector thickness uncertainties are about 1.0–3.4% and condenser thickness
uncertainties 4.5–11.8%. Source-reference position has an exact covariance
transport ambiguity and cannot be uniquely recovered from these data.

The selected-frame field and pixel discretizations pass two successive
noise-relative refinements. A new whole-column integrator now sums objective
tails and controlled-lens fields before propagating, retaining their focusing
cross-terms. Continuing the assumed profile across the entire column changes
rotation by about 5.25° and nominal 80k beam capture to about 90%, with all
controls and field amplitude fixed. Its integration is converged, but its
unmeasured far-field profile remains a model hypothesis. Refocusing and
held-out field-model comparison are required before interpreting its parameters.

PL1 response tomography and the coupled four-projector fit now use signed
position/tilt images independently of a known upstream source covariance.
The next projector test targets IL1 precision and a smaller acquisition.
Its practical gate is measured response
precision, not the number of optimizer starts. Condenser recovery uses
its separate beam/aperture measurements. Stigmator location and reproducible
aperture motion are unconfirmed; the approximate model does not supply them.

The earlier [algebraic PL1 example](neoarm_last_lens_tomography.py) assumed
0.0001-pixel response uncertainty. The new [image-level test](neoarm_last_lens_images.py)
instead generates and fits 207 noisy images at one million incident electrons
per frame, admitting 23 of 25 settings and retaining two unresolved crossover
settings as exclusions. Signed response uncertainties are 0.000410–0.001975
pixel; fitted PL1 length/gap errors are +3.55%/+7.22%, consistent with their
approximately 8–9% uncertainties, but outside the 1% target.

The [64-start response search](neoarm_last_lens_branch_search.py) eliminates
the sixteen unknown upstream coefficients by weighted QR at every trial,
reaching the same best image-derived solution without truth-based upstream
initialization. A second, same-excitation minimum is only 1.70 χ² above the best
fit, although its 1.58 m lens and 2.24 m detector gap exceed the approximate
column dimensions. Refitting inside a conditional 0.9 m span limit raises the
tested rival to Δχ²=59.42; one of 56 constrained/profile fits remains unresolved.
These local minima are not global bounds on alternative branches.
The free upstream coefficient model also imposes volume preservation rather
than the full magnetic-symplectic constraint, so its alternative solutions are
relaxed-model competitors, not demonstrated physical microscope families.

A separate relay-design search led to 162 new training images and 27 separate
held-out images at the same dose. Its nine-coordinate physical fit releases
the upstream round symplectic map and magnetic sample shear, recovering PL1
length/gap with errors −0.0456%/+0.4596% and uncertainties 3.36%/3.10%.
Held-out response χ² is 20.17/24 (predictive χ²19.20); 48/64 physical starts
converge, sixteen remain unresolved. The tested rival has Δχ²≈868 at a conditional
0.9 m span bound. Sub-1% fitted errors do not establish sub-1% parameter precision.

Two held-out analyser states reconstruct the upstream four-row ray map with
χ²10.67/16 against generating truth. The new tomography component propagates
shared analyser covariance across settings and handles response/calibration
cross-covariance when data are reused. A further ninety-image paired IL3/PL1
acquisition passes image fitting and two Gaussian pixel refinements, with
54 training images and 36 held-out endpoint images for the coupled optical test.

The coupled thirteen-coordinate IL3–PL1 fit now uses those original responses
and the standalone PL1 calibration images exactly once. Training χ² is
176.60/192 and held-out predictive χ² is 51.31/56 (raw χ²82.84). Length/gap
errors are −2.62%/−1.50% for PL1 and −3.32%/−2.54% for IL3, consistent with
approximately 3% local uncertainties. Sixteen of 24 starts converge. Releasing
the higher-excitation candidate's original phase boundary gives an interior
alternative with IL3 length 33.69 mm rather than 17.40 mm and training Δχ²=2.56.
Its saved held-out χ² is 135.59/56; those images favour the best fit but cannot
also remain untouched validation if used for branch selection. Fresh
discriminating settings and profile tests are required.
IL2/IL1 and the coupled condenser were outside that two-projector fit.

A subsequent eight-setting challenge generates 72 fresh GPU images, with
positive IL3 offsets chosen before noise generation and 36 frames withheld.
All image fits and the selected narrow/wide Gaussian pixel refinements pass. Refitting all thirteen
coordinates to the original plus new training responses gives χ²=200.40/224;
the earlier interior rival remains a local minimum but is worse by Δχ²=108.47.
Fresh held-out χ² is 32.89/32 (predictive 31.14), evaluated only after training
selection. Thirteen of 22 starts converge; nine remain unresolved.
PL1 length/gap errors are −0.27%/+0.76%, and IL3 errors are −0.86%/−0.53%,
while their marginal uncertainties remain 2.4–2.8%. This is tested branch
separation and favourable recovery, not 1% precision or global uniqueness.
The report is `projector_branch_recovery.json`.
Its accepted two-projector evidence uses 351 probe frames across 39
acquisitions, including independently repeated controls. A five-input ablation
now reduces this to 195 frames, retaining both detector coordinates and all
thirteen optics. The tested branch separation remains Δχ²=40.93, while geometry
uncertainty grows to 3.48–4.06%. Six predeclared orthogonal checks add 24 frames
and give χ²=22.05/24. These checks reuse historical data, and the fit starts
from the nine-input result; the 219-frame candidate still needs an independent
cold-start acquisition. It covers only two projectors, excluding material and
condenser images, so the compact full-column target remains unmet.

A separate full image-summary likelihood profiles the unknown quantum source
without duplicating centroid information. The old training shapes give
conditional χ²=627.73 versus 627.32 for the two fixed optical candidates,
providing little additional discrimination. Their joint optics/source fits
remain unconverged and are marked unaccepted; the fresh scan, not a presumed
source covariance, supplies the demonstrated branch separation.

The IL2 design holds PL1 fixed, toggles two calibrated IL3 analyser settings,
and scans four IL2 values. All 72 new GPU frames pass image fitting and the
selected narrow/wide two-step Gaussian refinements. Noisy relative maps
initialize IL2 without generating parameters. The final seventeen-coordinate
fit uses the original 224 and new 48 response coordinates once, with no added
posterior prior. Training χ² is 253.52/272 (270.49 at truth); untouched +0.375
IL2 heldouts give χ²=13.54/16, or 11.57 with prediction uncertainty.
Fourteen of 24 starts converge; the best is interior with rank seventeen.
PL1 length/gap errors are −0.73%/+0.04%, IL3 −1.25%/−0.93%, and IL2
−2.25%/−1.55%, with marginal uncertainties of 1.37–2.77%. Ten unresolved
starts retain the global-uniqueness limitation. The three-projector evidence
now totals 423 probe frames. See `three_projector_recovery.json`.

A further 72-image IL1 experiment releases all twenty-one projector/prefix
coordinates. Original 272 and new 48 response coordinates enter once, with
training χ²=283.68/320 (312.72 at truth) and fresh held-out χ²=11.97/16,
or 11.52 after parameter uncertainty. The best solution is interior with
rank twenty-one; nine of 24 starts converge, and a tested interior rival is
worse by Δχ²=68.29. Fifteen unresolved starts retain the uniqueness limitation.
PL1 length/gap uncertainties are 1.34%/0.88%, IL3 1.65%/1.48%, IL2 2.40%/2.01%,
and IL1 5.50%/3.35%; this is statistically consistent recovery, not 1% precision.
The four-projector evidence totals 495 probe frames. Both new Gaussian pixel
refinements and all image fits pass. See `four_projector_recovery.json`.

The shared calibration audit verifies three exact scale families in that
three-projector response. Data alone have rank 17/20 after releasing sample
position, tilt, and detector-pitch gains; one external prior makes rank 20.
Illustrative independent 0.2% calibration uncertainties impose a common
0.49% length floor and raise marginal uncertainties to 1.45–2.81%. This
uncertainty is shared across images and survives repeat averaging; the actual
instrument covariance remains to be measured. Analytic and full-Jacobian
covariance agree to 7×10⁻¹⁵ relative. See `projector_calibration_audit.json`.

The condenser capture test invalidates direct throughput ratios in several
original settings: only about 0.2–1.8% of aperture-out flux reaches the camera,
and clipping makes aperture-in/out ratios almost one despite roughly half the
incident beam passing the aperture. A fixed-objective aperture-imaging setting
instead places all three nominal aperture images inside the camera, with
computed aperture-out capture above 99.9998%. It requires a **clear specimen
radius of at least 4.33 µm plus alignment margins**, and a separate approximately
2002× specimen-magnification setting. The existing 40 nm-radius synthetic hole
blocks almost all of that beam; the atlas was not silently changed. Actual
clearance through the specimen, other apertures, and lens bores remains a
required instrument check.

Twelve exact-conjugate aperture images now fit beam covariance and aperture
magnification, with independent doses and unknown aperture centres. All three
families converge from measured-image initialization; statistical scale
uncertainties are 0.0087–0.0157%, while a 0.5% common radius error sets a 0.5%
scale floor. Two complete quadrature refits pass. Circular apertures do not
supply absolute rotation, and masked finite-defocus propagation remains
uncertified; these spatial fits are not the coupled condenser-lens fit.

The subsequent physical condenser checkpoint combines 22 new GPU beam images
with those aperture covariances. Its likelihood passes at synthetic truth
(χ²=59.50/57 training and 14.18/18 beam-only heldout), with noiseless closure
below 5×10⁻¹⁰σ. The optimizer nevertheless reaches its 1,500-evaluation limit;
its held-out χ²=471.82/18 fails, so recovery remains explicitly unaccepted.
All twelve profiled optical directions have local rank at that unconverged
iterate. The source covariance and collection rotation are free, but radius,
objective, and downstream calibration are held exact, and the diagram seed
coincides with generating geometry. Family 14's aperture data enter training,
so it is not a fully held-out optical state. Future fits also need moving
geometry to keep the aperture outside CL3 or an appropriate partial-lens map.

The revised `condenser_covariance_fit.json` profiles the invisible source
coordinate over its quantum-admissible interval, enforces aperture clearance,
adds one common 0.5% radius uncertainty, and withholds family 14 in both
modalities. Truth is consistent at χ²=57.68/54 training and 16.00/21 held out.
The bounded 100-iteration fit remains unaccepted, with normalized gradient
5.13×10⁻⁴ and held-out χ²=8072.79/21 despite training χ²=34.61/54.
At truth, the normalized condition number is about two million; a weak
CL1/CL3 centre-and-length direction predicts a 15.17σ combined held-out shift
for a one-linear-σ training change. Independent, captured CL1/CL3 excitation
states and informative narrow-beam training data are the next design need,
with separate new validation states. Objective/downstream calibration and
unchanged upstream astigmatic preparation remain conditional.

The stronger follow-up now supplies 22 actual GPU Gaussian-mode images at
eight training and three reserved settings, each with two downstream analysers.
All finite-pixel image fits pass; selected narrow/wide mode and pixel
refinements remain below 0.00124 noise σ RMS and 0.00227σ joint observable
bias. Refocusing C-mini permits well-captured independent CL1/CL3 excitation,
but those larger synthetic control ranges are not cleared for the instrument.
These are clear-path vacuum images, not transmission through the existing
40 nm-radius hole.

The old 54 and new 48 training coordinates pass at truth (χ²=122.80/102).
The geodesic descent bug is repaired, and the remaining optical/source
curvature is handled by profiling six excitation-law and eleven source/
calibration coordinates inside a six-coordinate geometry fit. This preserves
the likelihood and releases all twelve optical parameters, converging in
seventeen outer steps at data χ²=92.2120/102 plus a single radius-prior
contribution of 0.3977. The original-coordinate gradient is 8.85×10⁻⁹ and the
full observed Hessian is positive definite. The first stationary fit is in
`condenser_geometry_profile_diagnostic.json`.

Its selection was frozen before fresh held-out observations were loaded.
Raw χ²=166.54/18 exceeds image noise, while including local fitted-parameter
prediction covariance gives 22.41/18; principal beam-diameter errors remain
below 0.468%. CL1/CL3/C-mini length errors are +1.36%/+1.16%/+0.78%, with local
conditional uncertainties 1.04%/0.88%/0.35%, and centre errors are
−3.85/−2.83/−0.21 mm. All twelve errors are below 2.24 local standard errors.
These are local diagnostics, pending nonlinear profiles, not established
confidence coverage. Independent Gaussian-mode propagation at all six held-out
maps also passes mode/pixel refinements, with worst image discrepancy 0.000997
noise σ RMS and joint observable bias 0.00123σ. See
`condenser_stationary_validation.json` and `condenser_fitted_gaussian_validation.json`.

The stationary solution continues from a near-generating optical seed;
independent offset starts remain unsuccessful. Its source is highly chirped,
with a physically admissible Gaussian representative and an explicit invisible
covariance family. A small coordinate-dependent Hermitian margin here reflects
conditioning, not a measured proximity to the quantum limit. Blind
initialization, nonlinear profiles, and the coupled column fit remain open.

A first nonlinear CL1 profile moves its centre by +2.667 mm and refits all
22 other parameters, reaching a stationary fit with Δχ²=0.8933. The negative
one-sigma target has a feasible candidate at Δχ²=0.9513 but unresolved final
profile stationarity; both two-sigma targets remain unresolved. Thus the
near-equivalent physical family is demonstrated, while a complete confidence
interval and global branch exclusion are not. Enlarging the solver's
noise-unit step allowance helped the cold profile seeds without changing
relative likelihood weights. See `condenser_scaled_cl1_profile.json`.

The unknown internal-stigmator model now allows all ten source moments. With
independent CL1/CL3/C-mini scans, both tested stigmator gaps retain ideal rank
25, including all twelve optical coordinates. Holding lenses before the
stigmator fixed leaves exact source/stigmator/upstream-optics families; a round
source also retains discrete upstream-polarity alternatives. Using the two
actual analyser maps preserves formal rank but admits no detector frames from
the unrefocused Cartesian scan, with at most 0.14% power capture. A refocused
beam-only experiment needed refocusing; the subsequent autofocus design is
described below. See
`condenser_general_source_contract.json` and `condenser_actual_analyser_contract.json`.

### Calibration in the same synthetic column

The four-projector and condenser examples share controlled-lens lengths/gaps,
but their objective/sample fields differ. An affine control bridge matches
all four controlled-lens excitations to 4.4×10⁻¹⁶ rad while leaving detector
responses different by up to 1.234 pixels for the declared 5 nm/1 mrad inputs.
Their sample magnetic-shear invariants are +800 and −872.94 m⁻¹. Thus the
full four-projector posterior is incompatible with the condenser's conditional
calibration; see `column_calibration_compatibility.json`.

The new same-column calibration generates 26 GPU Gaussian images at the
condenser's two near-40k analyser settings. Eighteen images supply 16 signed
response coordinates; eight independent diagonal-input images remain reserved.
All image fits pass using image-derived initialization and fitted-Poisson noise
weights. Gaussian/reference differences are below 6.4×10⁻⁹ noise σ in full-image
Euclidean norm, and 8→16→32 detector quadrature refinements pass. These are
single pure calibration probes, separate from the upstream condenser source.

The joint fit releases all 23 condenser/source/aperture coordinates and six
canonical endpoint-row coordinates. The original 102 condenser observations,
16 new calibration coordinates, and one radius prior occur once, with no
posterior appended. Four outer updates reach data χ²=110.4473/118 plus a radius
contribution of 1.1963; the original normalized gradient is 4.58×10⁻⁹, the
full-information gradient is 1.16×10⁻⁷, and rank is 29. The training selection is
frozen before loading the new checks, whose χ² is 8.87/8 (local predictive 6.71).
Previously inspected condenser checks have raw χ²=146.53/18 and local predictive
χ²=21.29/18; these remain reused diagnostics, not fresh confirmation.

Conditional CL1/CL3/C-mini length uncertainties are 1.036%/0.874%/0.384%, and
centre uncertainties are 2.71/1.94/0.182 mm. This joint endpoint fit includes
statistical downstream-map uncertainty, while the objective prefield/shear,
external input/pixel calibration, aperture geometry, and source preparation
remain conditional. It does not identify individual projector lenses.
See `condenser_endpoint_calibration.json` and `condenser_endpoint_recovery.json`.

Independent postfit reconstruction reproduces the gradient and a positive full
observed Hessian, with information-whitened eigenvalues 0.9524–1.0518. The
generating-truth likelihood is χ²=145.13/118; all twelve condenser errors are
within 2.44 local standard errors. Length errors are +1.36%/+1.19%/+0.94%,
while centre errors are −4.33/−3.30/−0.367 mm. At identical fitted parameters,
releasing endpoint calibration inflates C-mini centre uncertainty by 1.79×
and length uncertainty by 1.09×. `condenser_endpoint_validation.json` separates
this calibration effect from the movement of the optimum.

The first independent three-start cold benchmark reconstructs its seeds from the
diagram and training-only covariance tomography. Both displaced seeds reach
the source constraint boundary before a geometry step; the nominal seed
profiles its nuisance parameters but its enlarged outer step is inadmissible.
The nominal geometry's coincidence with the generator is explicitly disclosed.
`condenser_cold_start.json` preserves these failures.

A subsequent moment-variable-projection initializer succeeds from the negative
displaced seed. At each geometry/excitation trial, a column-scaled QR solve fits
all nine visible source-covariance coordinates to measured second moments; the
temporary guide permits an unphysical covariance while moving geometry. Only a
physical guide result enters the original log-observable likelihood, which
then converges with all 23 parameters free at χ²=92.60968788, normalized
gradient 5.80×10⁻⁹, and full-information gradient 9.53×10⁻⁸. The starting lengths
differ by −8%/+6%/−10% in log coordinates and centres by −1/+0.3/−0.15 mm.
No saved fitted geometry, numerical generating truth, or held-out observations
enter this route. Independent recomputation agrees with the earlier minimum
within 8.3×10⁻⁸ local optical standard errors.

The opposite seed's guide ends at the aperture-gap boundary with an unphysical
source, while the nominal guide leads to an unresolved exact-fit stationarity
check. The negative-offset and nominal moment guides hit their iteration
budgets, while the positive-offset guide fails its line search. They serve only
as initializers, and the one accepted original-likelihood fit supplies the
recovery evidence. Thus one genuinely displaced start succeeds, not a uniform
basin or global-branch guarantee. See `condenser_moment_initialization.json`.

That cold result now also seeds the full 29-coordinate joint endpoint fit.
The endpoint coordinates are initialized exclusively from the measured signed
responses; no warm recovery supplies initialization, and no held-out arrays are loaded. It converges
at χ²=111.64361809 with original/full-information gradients 3.79×10⁻⁹ and
3.53×10⁻⁸, all positive geometry gaps, and a physical source. Its parameters
agree with the independently validated warm solution within 9.1×10⁻⁸ local
standard errors. This is a complete demonstrated cold-moment → physical
condenser → joint endpoint-calibration route under the existing conditional
field/preparation assumptions. See `condenser_cold_joint_endpoint.json`.

A five-input endpoint ablation retains reference/±x/±αx at each analyser,
using ten calibration images; eight orthogonal-input images supply checks.
All 29 parameters remain free at data χ²=100.36/110, and the fit converges.
Relative to the larger calibration set at identical parameters, C-mini centre
and length uncertainties increase by 26% and 7%. Reused orthogonal checks give
raw χ²=16.75/8 and predictive χ²=10.26/8. The 18-image calibration candidate still
needs a fresh independent acquisition; this ablation uses already-inspected
images and the previous full-calibration fit as its warm start. See
`condenser_endpoint_ablation.json`.

### Unknown internal stigmator: information after refocusing

A common-focus patch now captures at least 99.97% of the beam across both
tested stigmator locations and source/strength/position perturbations, but
the admitted control range provides insufficient optical information. Its
compact scan predicts lens-length errors of hundreds of percent; pixel
quadrature is converged, so further rendering cannot repair that design.
`condenser_internal_beam_design.json` retains the failed common-patch test.

Preparation-specific autofocus opens a broader route when the stigmator is
after CL1. Twelve training settings at two analysers retain all 25 optical,
source, and stigmator coordinates, with finite-pixel predicted lens-length
uncertainties 1.225%/0.507%/0.197% at 2 million electrons per frame. A new
4-million-electron preflight predicts 0.864%/0.358%/0.139% length uncertainties,
rank 25, and at least 99.94% capture for 24 training plus eight held-out frames.
Actual Gaussian rendering stops before generating frames or noise because
26 of 32 composed canonical maps fail the propagator's symplectic guard.
The guard remains unchanged; accumulated roundoff in ill-conditioned map
composition is a hypothesis to test, not an established cause. These are
conditional design calculations, not recovered optics or a measured autofocus
controller. For a later stigmator, informative admitted vertices lie outside
the largest sampled connected component; finer path continuation remains
necessary before judging that preparation. See `condenser_autofocus_routes.json`.
The frozen preflight is `condenser_internal_beam_acquisition_design.json`;
the [failure audit](../generated/neoarm_inversion/imaging_recipe/condenser_internal_beam_images_failure.json)
records zero rendered images and no noise draws. It also finds a sampled
C-mini mechanical angular RMS of 36.07 mrad, with a six-RMS envelope of
0.216 rad. These are covariance-envelope diagnostics, not maximum ray angles;
only nine planes per condenser were sampled, excluding objective/projector
interiors. Paraxial accuracy and intermediate aperture clearance therefore
need checking alongside canonical propagation accuracy.

### Same-column specimen integration

A third near-40k state, with zero IL1/PL1 offsets, is conjugate to numerical
precision in the condenser's shared-objective column. Both analyser states
converge to this target with the objective unchanged. Its residual propagation
phase bound is 6.0×10⁻¹¹ rad, and the existing cubic-sample renderer predicts a
nine-neighbour-radius evaluation using approximately 10.5 million beamlets.
`shared_material_design.json` records the GPU map/root check, not fresh sample
images. The next material render must verify the registered tile/halo coverage
and image inference in this actual column. An independently prescribed sample
waist remains a conditional illumination model; transporting the fitted mixed
condenser source through the specimen is a further integration step.

The two defocused 40k analyser states have sample-referred propagation distances
near −0.241 and +0.205 µm. Their highest retained sample frequencies violate
the conjugate predictor's phase gate, and predicted Gaussian-neighbour radii
35/30 exceed the existing default cap of 24. They need expanded-support
Gaussian checks and a GPU factored-Fresnel reference, not silent reuse of the
conjugate image estimator.

## What changed in this cleanup

- Replaced the 58-cell, roughly 2,445-code-line notebook with a reading path
  now containing 50 cells, including 31 code cells of at most 30 lines each.
  The original is preserved, unchanged, in
  [archive/neoarm_full_column_research.ipynb](archive/neoarm_full_column_research.ipynb).
- Separated physical detector images, numerical convergence, conditional
  one-parameter recovery, and the unsuccessful full-column fit. None is presented
  as a substitute for another.
- Extracted checked artifact loading, count-scaled image figures, a GPU sample
  render, and dense physical ray tracing into
  [neoarm_workflow.py](neoarm_workflow.py). The notebook does not execute hidden
  cells from the archive or launch a global fit when opened.
- Made magnification and GPU reproduction explicit controls. Review mode uses
  saved GPU arrays; fresh propagation requires the GPU. Bootstrap is not run.
- Kept all existing core APIs, simulation checkpoints and alternative experiments.
  The microscope configuration and user's geometry file were not changed.

This is a cleanup of the entry point and its evidence, not a claim that the
large archived orchestration has been refactored into a production application.

## Historical evidence inspected

The review loader uses the existing checkpoint validator to check the manifest,
frame requests, detector geometry, storage policy and every saved array checksum.
It also compares the configuration checksum. Missing or mismatched artifacts
produce an error; they are not silently replaced with preview images.

| Evidence | Result | Interpretation |
|---|---|---|
| Native imaging v5 GPU checkpoint | 224 detector arrays | Real saved Gaussian-rendered synthetic intensities, not just a manifest |
| Native diffraction v3 GPU checkpoint | 6 detector arrays | Native-column synthetic diffraction; not the older ideal-camera shortcut |
| Chosen 150k v11 GPU checkpoint | 9 frames, including sample-plane fields for material frames | Reproducible local forward demonstration |
| Physical validation ledger v2 | 145/145 signatures recorded accepted | Forward-reference evidence; its requirements explicitly set `require_parameter_stability=false` |
| 150k production refinement v2 | 480,249 → 751,689 → 1,172,889 beamlets | A local convergence experiment, not all-state/global parameter convergence |
| 150k independent near-conjugate reference | Detector difference 0.000152 noise σ RMS; largest recorded observable difference 0.006 σ | Strong numerical agreement in this example |
| Completed physical global-fit report v2 | 0 converged starts; `fit_success=false` | Full-column recovery failed |
| Same fit: training / held-out residual | 68,446 / 18,329 σ RMS | Far outside an acceptable statistical fit |
| Same fit: selected route / instrument acceptance | Both false | No calibrated full-column model to hand over |

The convergence numbers are reports from earlier runs, not all recomputed by
opening the review notebook. The checksum checks establish artifact integrity,
not scientific correctness. Historical checkpoints record package versions and
the configuration hash, but not a complete source-code fingerprint: changed
source cannot be certified merely by loading them.

During this cleanup, a fresh 150k sample/projector render was also run on the
GH200: 480,249 beamlets, about 9.9 seconds for re-expansion/propagation/rendering
(atlas construction about 29 seconds separately). Its relative image L2
difference from the archived render was 1.06e-7. A newly evaluated independent
near-conjugate grid reference differed by 0.0001521 noise σ RMS. The images and
metadata are in `review/fresh_gpu_reference.*`. This checks the current forward
path, not the global inverse fit or the quantitative realism of the specimen.

Sources: files under `examples/generated/neoarm_inversion/`, especially
`physical_checkpoint_native_imaging_v5_gpu/manifest.json`,
`physical_checkpoint_native_diffraction_v3_gpu/manifest.json`,
`neoarm_physical_150k_production_refinement_v2_gpu.metadata.json`,
`neoarm_physical_validation_ledger_v2.json`, and
`neoarm_physical_identifiable_subspace_v2.metadata.json`.

## Comparison with the plan

| Planned deliverable | Current state | Remaining work |
|---|---|---|
| C2 cubic action field, log amplitude/pathlength, explicit derivatives, pose and boundaries | Implemented in `multiscale_sample.py`, with extensive targeted tests; interpax pinned to 0.3.14 | Validate the actual specimen contrast and boundaries; retain compatibility tests |
| One registered multiscale atlas and automatic 40k–600k routing | Implemented and demonstrated on synthetic geometry | Replace nominal certificate and specimen layout; audit regional/global coarse-graining consistency |
| Local Gaussian action versus grid multiplication/re-expansion | Both implemented; production material images use re-expansion | Demonstrate robust automatic routing over the eventual measured parameter envelope |
| Large GPU simulations and streamed detector evaluation | Implemented; fresh 40k image uses 10.5 million beamlets, with source and array hashes | Keep topology frozen across compared optical states; cover the complete fitting envelope |
| Independent grid-wave reference | Implemented, including nonzero-B propagation and near-conjugate checks | Rerun relevant references after model/sample changes; do not confuse saved-reference coverage with fit acceptance |
| Convergence of all identifiable fitted directions over two refinements | Seven image parameters pass beam-count refinements; new finite-pixel specimen fits pass truth-bias and quadrature checks | Combine these gates over the full optical fitting envelope and calibration uncertainties |
| 30 settings / approximately 230 exposures | Historical downstream dataset exists; current four-projector evidence uses 495 probe images before specimen/condenser exposures | Demonstrate a smaller cold-start acquisition; bind real control units, branch-connected settings, safe limits and detector operation |
| Full condenser recovery without current readback | One displaced cold start reaches the conditional 29-coordinate condenser/source/aperture/endpoint fit using measured downstream calibration; a displaced CL1 profile retains a near-equivalent geometry | Make initialization robust, confirm illumination/aperture geometry, and fit the shared physical projector/objective model; the internal-stigmator alternative still needs valid propagation and images |
| Staged initialization, 64-start physical fit, profiles and equivalent solutions | Four-projector and conditional condenser fits demonstrated separately; tested projector branches separated; exact source-plane gauge and nonlinear CL1 alternative retained | Full joint-column multistart recovery, unresolved branch/profile searches and held-out validation; separate successful fits are not the complete 64-start column fit |
| Eight-setting adaptive experimental batches | One synthetic condenser cycle now connects eight new training settings plus three reserved checks to GPU images and a stationary conditional fit | Repeat from uncertain initial models; verify practical information gain and compactness with the instrument-valid measurement contract |
| Thin / hard-edge / Glaser / symplectic hierarchy | Thin/hard-edge/Glaser utilities and shared asymmetric immersed-objective field implemented | Resolve finite-support and overlapping-field error; held-out model selection on the same data |
| Publication-quality two-ray figures | Six full overlapping-field figure sets now add true-z/zoom panels, both bases, a continuous Larmor strip, verified events/envelope, common-scale IL3 wobbles, and source/array/figure hashes | Common-scale truth/initial/accepted-fit series require an accepted full-column fit |
| Identifiable recovery within 1%, held-out targets | Conditional held-out probe targets pass; not every physical parameter is recovered within 1% | Specimen/diffraction estimator acceptance, broader recovery/refinement tests, and instrument validation remain |

## Scientific issues hidden by the old notebook

### 1. The specimen is still a numerical target

`_render_certified_scene` includes synthetic Gabor markers, nominal multilayer
contrast, and two orthogonal sinusoidal lattice carriers. A 0.31356 nm period and
an hkl label do not turn that field into a quantitative Si diffraction model.
There is no demonstrated treatment of specimen thickness, dynamical scattering,
partial coherence or the real instrument's aberrations sufficient to predict
experimental crystalline intensities.

That does not invalidate layer-distance or lattice-vector metrology. It means
we should initially fit robust geometric observables and treat intensities only
where the illumination, specimen and detector models are justified. The atlas
also synthesizes regional finest grids separately and includes atomic detail
only on resolving grids. Within-pyramid antialiasing is implemented; exact
cross-region coarse-graining of one authoritative field needs a separate audit.

At 600k, adequate lattice sampling does not imply usable contrast. The recorded
in-focus phase CTF is about 8.2e-7; the pipeline excludes the corresponding
real-space Fourier blocks. Signed object defocus near ±19.6 nm was demonstrated
separately. This must enter the acquisition, not be hidden by image normalization.

### 2. The 230 frames do not identify the condenser

The declared calibrated sample-plane input fixes what the downstream optics see.
In the synthetic forward sequence, source beams are back-propagated from those
sample conditions for each state. Those exposures therefore cannot identify
CL1, CL3 and C-mini simply because upstream lenses appear in the software model.

The current candidate uses fixed astigmatic illumination, known downstream
projection states, and an intermediate aperture reference. Entrance covariance
is fitted jointly; ellipse orientation is computed from the transported image,
because focusing can rotate a general ellipse as well as Larmor rotation.
The shared-field diagnostic adds a narrow calibration probe and a wider PL1
sweep. It contains 222 aperture-out images, 42 aperture-in measurements and 42
independent dose references before repeats and held-out data. This expanded
diagnostic is not the compact final recipe. The user has not confirmed the
stigmator's position or reproducible aperture translation; neither is assumed
to be an established hardware input.

The earlier astigmatic proposal budgeted approximately 232 additional frames,
or 184 with joint image-summary covariance, giving about 462 or 414 combined
with the downstream set. Those are historical budgets, not the current minimal
recipe. The corrected local information audit favours imaging diversity and
aperture edge traversal; detector-image fits and blind recovery still need to
establish how many settings suffice.

### 3. Synthetic control branches are not instrument settings

The configuration explicitly uses unscaled control coordinates. Its nominal
magnification search reaches high-order hard-edge excitations, and small changes
in individual model coordinates can yield huge changes in the optical response.
A parameter vector close in an arbitrary scaled Euclidean norm need not be a
useful optical initialization.

Hard-edge transport is periodic in excitation phase. When effective length and
position can move too, several physical parameterizations can predict similar
endpoint maps. More random starts alone do not establish uniqueness.

Use connected one-lens scans, signed responses and rotation to track the branch;
use independently credible geometry or a verified weak-excitation reference to
anchor it. Unknown entrance-ellipse orientation must remain a nuisance, not an
unacknowledged phase anchor. Existing `dac_ramp_steps=8` metadata describe the
approach to an endpoint: there are no observations at those intermediate points.
The new continuation utilities are not yet an executed branch-discrimination
experiment on this instrument.

### 4. A stable subspace at a bad fit is not successful inversion

The historical completed report contains a stable terminal subspace and some passing
individual flags despite enormous residuals and zero converged starts. These
flags cannot establish recovery. The later archived fitter contains further
admission checks and optimizer conditioning changes. The subsequent
[conditional recovery](NEOARM_RECOVERY.md) succeeds with a consistent probe
likelihood and sensitivity-scaled bounded fitting; the historical full material
likelihood still fails at truth.

The new driver separates truth checks, image-derived initialization, optimization,
rank diagnostics, and branch challenges. It retains failed-stage statuses and
reports held-out measurements independently of optimizer convergence.

## Recommended sequence from here

1. **Improve the projector recipe.** Target the weak IL1 directions and test
   a cold-start five-input acquisition across all four lenses, with reserved
   orthogonal checks. Measure the achievable precision, include the external
   calibration covariance, and retain unresolved excitation branches.
2. **Make condenser recovery reproducible.** Extend the successful cold
   moment-initialization route to the other starts and excitation branches,
   and continue nonlinear profiles from nearby feasible solutions. For the
   unknown internal stigmator, validate canonical propagation and intermediate
   angles/clearance before rendering the refocused scan; its information
   forecast alone cannot replace the conditional preparation model.
3. **Validate the coupled physical likelihood.** Resolve objective-field tails
   and overlap, integrate the accepted cubic specimen estimator and condenser
   images, and retain source/calibration nuisances. Require likelihood closure
   at truth, noisy recovery, two optical-parameter refinements, and weak-family
   profiles before promising a physical fit. Aperture transmission needs a
   verified detector-capture model, not assumed collection of all transmitted flux.
4. **Instrument reconnaissance.** Confirm actual lens-control limits, fixed
   objectives/corrector state, aperture z/radius/translation, stigmator location,
   detector response, and specimen certificate. Translation is optional for
   the aperture-image recipe; insertion/selection and focus must be repeatable.
   Obtain small connected
   one-lens scans in both approach directions. No normalized synthetic controls
   should be sent directly to hardware.
5. **First real predictive fit.** Fit a limited downstream range plus the
   confirmed condenser experiment; validate held-out magnification, rotation,
   centroid, diameter and diffraction vectors against the plan's tolerances.
   Expand toward 40k–600k only as coverage and model agreement justify it.

We can reasonably aim to calibrate a predictive NEOARM optical model under these
constraints. We cannot yet promise a unique physical reconstruction of every
lens position, thickness and excitation law. Those are distinct outcomes, and
the notebook now keeps them distinct.

## Running the cleaned notebook

Choose the environment containing the installed checkout (`TemGymCore (GPU)`
in this workspace). Run the notebook top-to-bottom. Saved artifacts are required
for review mode; a missing checkpoint gives an explicit error. Set
`RERENDER_150K_ON_GPU=True` for the fresh sample/projector render.

Figures belong under `examples/generated/neoarm_inversion/review/`, with the
new full-field ray exports under `imaging_recipe/review/shared_field/` and the
preserved surrogate figures under `imaging_recipe/rays/`. The current executed copy is
`imaging_recipe/neoarm_physical_response_review.executed.ipynb` beneath the same
generated root. The source notebook keeps outputs cleared so that old results
cannot be mistaken for the next execution.

## Verification

The current 50-cell notebook executed all 31 code cells without errors, with
identical source/executed cell text and cleared source outputs. It now shows
the four-projector fit, fresh IL1 and Gaussian-mode condenser detector images,
the five-input ablation, shared calibration uncertainty, the stationary
conditional condenser fit, its near-equivalent CL1 geometry, and the same-column
joint endpoint calibration. The new image figures were visually inspected and
exported in SVG, PDF, and 300-dpi PNG. Code cells remain at most 30 lines.

The latest targeted run passed **214 tests in 79.54 seconds**, covering
four-projector design/acquisition/recovery, condenser Gaussian rendering,
training-data separation, the internal-stigmator and general-source contracts,
guarded/profiled least squares, reference-source coordinates, stationary-fit
validation, nonlinear CL1 profiles, the cold moment-to-joint route, same-column
calibration and ablation, shared material-state design, autofocus, acquisition
failure diagnostics, and ray/notebook dependencies. The production image and
recovery reports independently record GPU execution. The final audit of the
13 new stable reports passed **1,103 declared checksum comparisons covering
183 distinct target files**, including 16 NPZ archives and 51 JSON files,
with no stale or missing targets. The failed internal-stigmator acquisition
correctly declares that it created no public inference arrays.
These counts include repeated references to the same input; they are integrity
checks, not independent scientific tests. The accepted cold condenser fit was
independently recomputed in the original likelihood, and the cold joint result
agrees with the independently validated warm minimum within 9.1×10⁻⁸ local
standard errors. The same-column audit reconstructs the calibration responses,
likelihood, positive local Hessian, and truth likelihood. Earlier independent
checks cover the four-projector assembly, fitted likelihood, held-out prediction,
and core-map closure. These checks support the reported conditional results,
not full-column acceptance.

The preceding run passed **152 tests in 64.62 seconds**; its condenser audit
passed 1,284 repeated checksum comparisons across 24 reports and 111 distinct
target files. This earlier report set differs from the 13 new reports above.

The preceding targeted run passed **80 tests in 60.86 seconds**, covering
calibration families/covariance, the five-input likelihood, IL2 acquisition and
three-projector recovery, condenser source/geometry guards, and earlier image,
aperture, and ray dependencies. Its seven reports passed 405 integrity checks.

The preceding 45-cell notebook executed all 28 code cells without errors, showing
pixel-integrated specimen images, fits and residuals, specimen footprints,
PL1 calibration images, fresh branch-challenge images, accepted IL3/PL1 fits,
aperture images, the unaccepted condenser checkpoint, and both full-field ray
bases. Code cells remain at most 30 lines. New ray and acquisition figures were
visually inspected and exported in SVG, PDF, and 300-dpi PNG. That
targeted run passed **83 tests in 79.22 seconds**, covering full-field ray
profiles/figures, new branch acquisition/recovery, the source-shape likelihood,
condenser diagnostics, IL2 design, and their coupled calibration dependencies.
Across seven reports, 337 source/report/figure hash checks and the declared
array checksums passed; its source and executed notebook had identical
cell text and cleared source outputs.

The preceding targeted run passed **156 tests in 117.22 seconds**, covering the
immersed objective, shared-column initialization, field convergence, coupled
information, finite-pixel specimen rendering, algebraic PL1 tomography,
image-derived PL1 fitting, branch/profile searches, whole-column integration,
relay/aperture design, physical and coupled fitting, inference-only manifests,
shared-covariance reconstruction, and aperture-image initialization/refinement.
Two additional shifted-boundary tests pass separately. This is focused
coverage, not full-repository or
physical-inversion acceptance.

The current recipe report was shortened to about one third of its previous
length; its complete
pre-consolidation text is preserved in
[the research record](archive/neoarm_imaging_recipe_research_2026-09-05.md).

The aperture audit propagates 64 incoherent Hermite–Gaussian modes on the GPU
for all thirty original settings and three aperture radii. Its fifty dependency
hashes and input/source digests match the saved reports. A difficult off-band
refinement changes detected aperture power by 1.88e−7 absolute; this selected
power check does not certify full-image convergence over the fitting bounds.

The shared-field audit was rerun on the GPU with 49 repository/input hashes
verified before and after calculation, an NPZ checksum, invocation arguments
and version metadata. Declared specimen/metrology source and array hashes were
independently checked; the image-comparison figure was visually inspected.
The convergence report's generated-image hashes have no companion saved pixel
arrays and require rerendering for direct verification; its source and field
mesh hashes match. The preceding 82-test run and historical counts below are
separate runs, not additive coverage.

### Historical cleanup checks

- The review notebook executed top-to-bottom, including a second execution
  with fresh GPU rendering and the independent reference enabled.
- 249 distinct targeted tests passed: cubic sample, grid reference, plotting,
  phase continuation, calibration, inversion, nuisance-profiled fitting and
  the new review helpers. This is not a full-repository test run and does not
  replace the scientific end-to-end acceptance criteria.
- Generated image and ray figures were visually inspected; the Gaussian ray
  basis now uses the selected exposure's actual specimen illumination waist.
- The original inversion configuration, inversion model and user's
  `microscope.toml` retained their pre-cleanup checksums.

The subsequent recovery work passed 161 distinct targeted tests (including the
new finite-detector moments, signed-noise covariance, constrained optimizer, and
branch-seed tests). The updated reading notebook also executed top-to-bottom;
its executed copy is under `examples/generated/neoarm_inversion/recovery/`.
The [recovery report](NEOARM_RECOVERY.md) gives the scientific scope of those tests.
