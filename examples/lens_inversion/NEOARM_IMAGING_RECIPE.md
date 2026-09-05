# NEOARM: the current imaging-first recipe

5 September 2026. GPU specimen imaging, image-derived four-projector fitting,
and a stationary conditional condenser fit work under the stated synthetic
assumptions. A same-column fit now jointly releases condenser parameters and
six downstream endpoint-map coordinates, using 26 new signed-input calibration
images. A moment-projection initializer now reaches the same joint minimum
from one genuinely displaced geometry without a saved fitted seed. Two other
prescribed starts remain unresolved, and the beam/aperture geometry is still
conditional; this is not a hardware-ready recipe.
Recovery of the complete projector
and condenser systems, with instrument-valid controls and held-out predictions,
remains unfinished. The [notebook](neoarm_full_column_inversion.ipynb) is the
reading entry point; [NEOARM_STATUS.md](NEOARM_STATUS.md) tracks the full plan.
The [detailed research record](archive/neoarm_imaging_recipe_research_2026-09-05.md)
preserves earlier models, failed assumptions, numerical comparisons, and commands.

## The imaging-first candidate

Use the same registered specimen throughout, with certified multilayers for
40k calibration and an 80k check. With 55 µm detector pixels, a nominal 23 nm
repeat occupies 16.7/33.5 pixels and the fields of view are 352/176 nm.
Actual certificate distances replace the nominal Si/SiGe widths.

1. **Record the control/geometry contract.** Read the approximate diagram,
   retain real free-lens values and approach direction, and establish permitted
   changes. Objective controls stay fixed; the corrector is disabled. Synthetic
   DAC coordinates in these examples are not instrument commands.
2. **Calibrate PL1 with signed inputs.** At each selected PL1 state, record a
   reference and ±x, ±y, ±αx, and ±αy images. The tested inputs are 5 nm and
   1 mrad. Fit image means, covariance, and dose, then use their full covariance
   in the lens fit. A separate narrow probe is used for this measurement.
3. **Work upstream through the projectors.** Two sufficiently independent
   calibrated PL1 analyser settings can reconstruct the upstream map for each
   IL3 state; repeat for earlier lenses. Noisy joint IL3–PL1 recovery now passes
   fresh held-out response checks, and a wider connected IL3 scan separates the
   tested competing branch. A further 72-image IL2 experiment now passes joint
   seventeen-coordinate recovery and fresh held-out prediction. A further IL1
   batch now passes the twenty-one-coordinate four-projector test; its length
   precision remains the weakest at approximately 5.5%.
4. **Measure the condenser separately.** Scan CL1/CL3/C-mini with a stable
   astigmatic illumination and fit physically transported beam images. Add
   aperture images/counts in the collection state below, with independent dose
   references. Neither calibrated upstream deflections nor aperture translation
   is assumed available.
5. **Keep independent checks.** Reserve optical settings and the second specimen
   scale for held-out prediction, including opposite approach directions.
   Choose additional measurements from the remaining weak directions.

Aperture collection introduces a separate approximately 2002× specimen setting;
the material calibration remains at 40k/80k. This is an imaging-mode candidate,
not a promise that every exposure stays at those two specimen magnifications.
Its clear-vacuum and other hardware requirements must be confirmed before
finalizing a compact frame budget.

## Pixel-integrated specimen images and fits

The cubic specimen is propagated with 10,504,081 Gaussian beams at 40k and
2,424,249 at 80k. GPU evaluation integrates coherent intensity over detector
pixels and includes the external illumination halo for blur. An independent
cubic-object predictor fits seven image parameters together.

| State | Noisy residual, σ RMS | Magnification error | Rotation error |
|---|---:|---:|---:|
| 40k | 0.99874 | −0.01359% | −0.01487° |
| 80k | 1.00377 | +0.02992% | +0.01742° |

Terminal intensity and image-metrology checks pass; the maximum deterministic
parameter bias is 0.00302σ. Earlier two-step beam-count refinements and the
new pixel-quadrature checks are separate gates. A stricter complex-field
backend comparison fails its 1e−9 threshold because of very large carrier
phases; further propagation of those sampled complex arrays is not certified.

These images use the approximate diagram geometry and independent effective
objective halves. They are not shared-field lens-recovery evidence. Detector,
certificate, and beam-centre calibration are exact in this example; real
contrast, partial coherence, and full fitting-range view coverage remain open.
Reports: `finite_pixel_specimen.json` and `finite_pixel_metrology.json`.

A GPU design check now finds a conjugate 40k state in the condenser's own
shared-objective column, at zero IL1/PL1 offsets. Its objective stays unchanged
from the two analyser states; `shared_material_design.json` verifies the map
and phase bound. This third setting keeps the same magnification and admits
the conjugate cubic-sample reference, but fresh material images and footprint/
halo checks remain to be run. The two defocused analyser states require a
factored-Fresnel GPU reference and wider Gaussian support. Prescribing a sample
waist is separate from propagating the fitted condenser's mixed illumination
through the specimen.

## A smaller downstream experiment: last-lens tomography

Measure the full signed 2×4 position response H to specimen inputs
(x,y,αx,αy). For a final hard-edge lens of length L and detector gap d, let
s=d/L and sinc(q)=sin(q)/q. Derotation gives

\[
R(q)H(q)=\cos q\,W_0+q\sin q\,W_1+\operatorname{sinc}(q)W_2,
\quad W_0=-W_1/s+sW_2,
\quad d=\sqrt{\det\begin{pmatrix}W_1\\W_2\end{pmatrix}},\quad L=d/s.
\]

Both transverse planes and SI input/pixel calibration are needed. Unknown
sample magnetic shear and a constant frame rotation preserve this volume
identity. The initial model releases sixteen upstream coefficients plus s,
excitation bias, and excitation slope.

The GPU image test admits 23 of 25 states; two unresolved subpixel crossover
settings are retained as exclusions. All 207 noisy images fit six image
coordinates. At one million incident electrons/frame, extracted response
uncertainties are 0.000410–0.001975 pixel. Independent Gaussian propagation
checks the narrowest and broadest admitted images below 2e−8 in whitened L2.

| Physical fit | PL1 length | Detector gap |
|---|---:|---:|
| Baseline fitted error | +0.201% | +3.013% |
| Baseline local standard uncertainty | 8.96% | 8.50% |
| Redesigned fitted error | −0.0456% | +0.4596% |
| Redesigned local standard uncertainty | 3.36% | 3.10% |

The redesigned relay keeps eighteen PL1 states fixed. Its 162 training and
27 separately generated held-out images now pass image fitting and two
successive Gaussian pixel-quadrature refinements. Nine information scenarios
and 24 width/capture corner checks also cover stated relay/source perturbations;
the finite checks do not certify a continuous parameter box.

### Initialization, branches, and physical families

The earlier nineteen-coordinate volume-only fit is now an initializer and
comparison. Training-only weighted QR seeds a nine-coordinate physical model
with round symplectic upstream optics, unknown sample magnetic shear, and
four free PL1 parameters. No generating optics or held-out response initializes
the physical fit; all measured response covariances remain in the likelihood.

The baseline and redesigned fits converge from 47/64 and 48/64 starts, leaving
17 and 16 unresolved. Redesigned training χ² is 125.16 for 144 coordinates;
the three held-out states give χ²=20.17/24, or 19.20 after including prediction
uncertainty. A tested same-phase competitor has Δχ²≈868 at the conditional
L+d≤0.9 m bound. These finite local searches do not establish global uniqueness.
The small fitted errors in the table coexist with several-percent uncertainty.

Two redesigned held-out analyser states reconstruct the unknown upstream map,
with χ²=10.67 for its sixteen coordinates against generating truth. The
reconstruction carries PL1's full marginal covariance and preserves correlations
between upstream settings. Eight tests include the exact calibration-scale
family L,d ∝ position gain × tilt gain / pixel gain², so improved counting
statistics cannot replace external physical calibration.

A separate ninety-image IL3/PL1 acquisition supplies three training and two
held-out IL3 states, each at two analyser settings. The thirteen-coordinate
joint fit uses the original responses and counts the standalone PL1 calibration
images once. Training χ² is 176.60/192; held-out χ² is 82.84/56, or 51.31 after
joint parameter uncertainty. PL1 length/gap errors are −2.62%/−1.50%; IL3
length/gap errors are −3.32%/−2.54%, with all four local uncertainties about 3%.
Sixteen of 24 starts converge. Releasing the higher-excitation candidate's old
phase boundary gives an interior alternative with IL3 length 33.69 mm instead
of 17.40 mm and training Δχ²=2.56. Its specific held-out χ² is worse at 135.59/56;
one shifted-boundary start remains unresolved. Fresh measurements must validate
a branch chosen using these held-out images.

The fresh eight-setting challenge supplies those measurements. Four positive
IL3 offsets (0.40, 0.45, 0.60, 0.75 in synthetic native units) are recorded at
each of two unchanged PL1 analysers. Offsets 0.40/0.75 enter training; 0.45/0.60
are withheld until every training fit finishes. All 72 GPU images pass their
six-coordinate image fits; selected narrow/wide states pass two successive
independent Gaussian pixel refinements.
The selected path stays resolved under both previously fitted hypotheses;
an alternative negative scan crossed a subpixel waist and was excluded.

The joint likelihood contains the old 192 and fresh 32 response coordinates
once, with all thirteen optical coordinates free. Training χ² is 200.40/224,
compared with 213.54 at synthetic truth. The previous rival refits to an
interior minimum at Δχ²=108.47; a further tested minimum is Δχ²=295.11.
Fresh held-out χ² is 32.89/32, or 31.14 after parameter uncertainty. Thirteen
of 22 starts converge, and nine remain unresolved. The best solution has no
active bounds, although the conditional span bound and inherited negative
PL1 branch still define the search.

PL1 length/gap errors are −0.27%/+0.76% and IL3 errors −0.86%/−0.53%, while
their marginal uncertainties remain 2.4–2.8%. A favourable sub-1% noise draw
therefore establishes neither 1% precision nor global uniqueness. These results
are recorded in `projector_branch_recovery.json`; old heldouts are separately
labelled reused model-selection evidence.

The accepted two-projector evidence currently comprises 351 probe frames:
28 training acquisitions and eleven held-out acquisitions, each with nine
input images. Two training controls are repeated with independent noise.
The five-input ablation now retains the reference, ±x position, and ±x tilt
images, with both detector centroid coordinates and full per-image covariance.
Its 195 frames preserve the tested branch separation at Δχ²=40.93, compared
with 108.44 for the matched nine-input centroid likelihood. Geometry uncertainty
increases by about 46%, to 3.48–4.06%. Six predeclared orthogonal-input checks
add 24 frames and give χ²=22.05/24; these previously analysed data support a
219-frame candidate, not an independent compact acquisition. Warm starts came
from the nine-input fit. Cold initialization, three-projector extension, and
departures from round optics still require testing. This budget excludes
material images and the separate condenser experiment.

Using the old beam shapes instead was also tested. The five retained image-fit
coordinates use their full covariance, with free dose and per-setting alignment
profiled; no centroid-response term is appended. For each fixed optical
candidate, all ten quantum-admissible source-covariance directions are free,
including the exact invisible antisymmetric position–momentum covariance.
The conditional shape χ² values are 627.73 and 627.32, so those training shapes
do not separate the rival. The joint optics/source fits still reach their
evaluation limits and are explicitly unaccepted in `projector_shape_likelihood.json`.
The wider lens scan supplies the demonstrated discrimination.

The generator uses a pure 2 nm × 1.5 nm source, while the response fit requires
no known source covariance or purity. Input calibration, detector response,
and within-group alignment stability remain conditional. Counts describe synthetic accumulated
dose, not a validated Medipix exposure/rate/counter-depth schedule.

The IL2 batch specified in `il2_extension_design.json` is now acquired. Hold PL1
at normalized control +0.8333, alternate the already calibrated IL3 offsets
+0.125/+0.75, and use IL2 offsets −0.75, 0, +0.375, +0.75. The +0.375 state is
held out, giving eight settings and 72 fresh GPU frames. All six-coordinate
image fits pass; selected narrow/wide Gaussian images pass both independent
8→16→32 pixel refinements. Noisy relative maps initialize IL2 without its
generating parameters, and the final fit uses the original 224 and new 48
response coordinates once, releasing all seventeen optical coordinates.

Training χ² is 253.52/272, compared with 270.49 at truth. Fresh held-out χ² is
13.54/16, or 11.57 with parameter uncertainty. Fourteen of 24 starts converge;
the best is interior with rank seventeen. PL1 length/gap errors are
−0.73%/+0.04%, IL3 −1.25%/−0.93%, and IL2 −2.25%/−1.55%, with uncertainties
1.71%/1.37%, 1.96%/1.82%, and 2.77%/2.34%, respectively. The nearest converged
alternative is far worse, but ten unresolved starts and the inherited branch
and span bounds prevent a global uniqueness claim. The three-projector
evidence now comprises 423 probe frames. Reports are `il2_extension_images.json`
and `three_projector_recovery.json`; hardware-safe control intervals remain
unconfirmed.

The next 72-image IL1 acquisition now passes the same image-level tests and
extends the joint fit through all four projectors. Noisy relative maps provide
IL1 initialization; the final twenty-one-coordinate likelihood contains the
previous 272 and new 48 response coordinates once. Training χ²=283.68/320
(312.72 at truth), and untouched IL1 +0.375 heldouts give χ²=11.97/16,
or 11.52 with prediction uncertainty. Nine of 24 starts converge, with an
interior rank-twenty-one best solution. A tested interior rival is worse by
Δχ²=68.29; fifteen unresolved starts prevent a global uniqueness claim.

| Lens | Length uncertainty | Following-gap uncertainty |
|---|---:|---:|
| PL1 | 1.34% | 0.88% |
| IL3 | 1.65% | 1.48% |
| IL2 | 2.40% | 2.01% |
| IL1 | 5.50% | 3.35% |

The accumulated four-projector evidence comprises 495 probe frames. The
report `four_projector_recovery.json` retains the raw held-out checks,
initialization, alternative solutions, and conditional calibration assumptions.
IL1 precision and the full-column frame budget remain design targets.

### Shared physical calibration

Position, tilt, and pixel-pitch uncertainty enters as one shared calibration
factor, not independent noise on every response. The round-lens model has an
exact family: when actual calibration gains are ignored, every lens length
and following gap scales as position gain × tilt gain / pixel gain². For fixed
observations, a trial gain is compensated by the inverse geometry scaling.
Excitation laws and dimensionless geometry ratios stay unchanged.

The three-projector calibration audit verifies data rank 17/20 when all three
gains are released, and rank 20 after including their external prior once.
Illustrative independent 0.2% log-gain uncertainties give a shared 0.49% length
floor and marginal length/gap uncertainties of 1.45–2.81%. Analytic and
full-Jacobian marginal covariance agree to 7×10⁻¹⁵ relative; the correlated
same-input prediction covariance agrees to 7×10⁻¹². Repeated images reduce
statistical error but not that common floor. These are assumed calibration
uncertainties, not measured NEOARM values. `projector_calibration_audit.json`
records the exact families and conditional repeat forecasts.

## Downstream calibration in the condenser column

The separate four-projector example uses a different objective/sample field.
Even after matching all four lens excitations, the sample-to-detector responses
differ by up to 1.234 pixels for the stated calibration inputs, so its full
posterior cannot supply the condenser's downstream calibration.

The new experiment calibrates the condenser column's own two near-40k analyser
settings. A reference and ±x, ±y, ±αx, and ±αy at each setting give 18 training
images; two additional diagonal positive/negative pairs at each setting give
eight independent checks. Inputs are 5 nm and 1 mrad, with 2 million electrons
per image. All 26 GPU Gaussian images pass fitting with data-derived noise
weights and the selected two successive pixel refinements.

The original 102 condenser observations and 16 new signed responses enter one
29-coordinate fit with a single aperture-radius prior. Six canonical endpoint
row coordinates describe the downstream responses; they are neither fixed nor
constrained by a reused posterior. The fit converges at data χ²=110.45/118 and
new check χ²=8.87/8 (local predictive χ²=6.71), with conditional lens-length
uncertainties 1.036%, 0.874%, and 0.384% for CL1, CL3, and C-mini. Objective
prefield/shear, calibrated input/pixel scales, aperture geometry, and unchanged
upstream preparation remain conditional. The endpoint maps are measured
responses, not an additional physical four-projector reconstruction.

The independent audit reproduces a positive full local Hessian and generating
truth χ²=145.13/118; all twelve condenser errors are within 2.44 local standard
errors. Comparing at identical fitted parameters, endpoint uncertainty increases
C-mini centre uncertainty by 1.79× and length uncertainty by 1.09×. Its previously
inspected beam-diameter predictions remain within 0.442%.

Reports are `column_calibration_compatibility.json`,
`condenser_endpoint_calibration.json`, `condenser_endpoint_recovery.json`, and
`condenser_endpoint_validation.json`.
The previous condenser solution initializes this test. A five-input ablation
retains reference/±x/±αx at each analyser, giving ten calibration images plus
eight orthogonal checks; all 29 parameters still converge. C-mini centre/length
uncertainties increase by 26%/7% at identical parameters. This 18-image candidate
reuses inspected data and needs an independent acquisition. See
`condenser_endpoint_ablation.json`.

Cold initialization now has one successful displaced start. The moment-based
guide profiles all nine visible source coefficients by weighted QR while
geometry moves; only its physical output enters the original log-observable
likelihood. That final 23-coordinate fit converges to χ²=92.60968788 from
log-length offsets −0.08/+0.06/−0.10 and centre offsets −1/+0.3/−0.15 mm,
without a saved accepted geometry, generating values, or held-out data. The
opposite-offset and nominal starts remain unresolved. The moment objective is
initialization only, with separate convergence recorded for the final original
likelihood. See `condenser_moment_initialization.json`; the earlier direct-profile
failures remain in `condenser_cold_start.json`.

The cold result also seeds the full 29-coordinate endpoint/condenser fit,
with measured-response endpoint initialization and no warm recovery artifact.
The original joint likelihood converges at χ²=111.64361809, agreeing with the
validated warm solution within 9.1×10⁻⁸ local standard errors. No held-out data
are loaded by this continuation. `condenser_cold_joint_endpoint.json` records
the complete conditional cold-to-joint route, with all previous geometry,
source-preparation, and field assumptions retained.

## Collecting the condenser aperture signal

The original high-magnification throughput assumption fails. In affected
off-band states, only 0.76–1.82% of aperture-out flux reaches the 40k camera,
or 0.19–0.46% at 80k. The detected rays already pass the small aperture,
so their in/out ratio is nearly one while actual transmission is about
0.46–0.51. The previous total-flux information calculation is therefore an
optimistic bound, not an executable measurement at those settings.

GPU wave propagation of 64 incoherent Hermite–Gaussian modes confirms finite
camera capture for all thirty original states and three aperture radii.
Even a focused 80k image collects only 86.64% of transmitted power in one
10 µm-aperture case. Diffraction redistribution can make the detected in/out
ratio exceed one; it must not be interpreted as total transmission.

A separate fixed-objective collection solution has aperture magnification
120 and aperture-to-detector B-block norm ≤1.14e−12 m. The nominal
10/20/30 µm aperture radii image to 1.2/2.4/3.6 mm on the detector. Computed
aperture-out capture exceeds 99.9998% across fifteen illumination families.
C-mini and projector controls change without changing illumination upstream
of the aperture.

The specimen beam RMS is 0.349–0.721 µm, requiring a **clear vacuum radius
of at least 4.33 µm plus alignment margins**. The current 40 nm-radius synthetic
hole passes only 0.171–0.590% of that beam; the atlas has not been altered to
hide this failure. A sufficiently large registered vacuum region on the same
physical specimen is a requirement. Other apertures and lens bores also need
clearance checks.

Aperture-image fitting now recovers spatial beam covariance and geometric scale
from twelve GPU images across three beam families. Each sixteen-coordinate fit
releases aperture centres and independent exposure doses, starting from measured
moments and a common 100× seed for the generating 120× collection state.
Magnification errors are +0.0108%, −0.0277%, and +0.00422%; statistical standard
uncertainties are 0.0087–0.0157%. Two complete quadrature refits move parameters
by less than 0.000292σ. A separate 31-coordinate fit also releases per-frame beam
shape and retains scale information in the tested family.

The fitted aperture-space covariance is detector-aligned; circular apertures
leave absolute rotation unmeasured, and a shared 0.5% radius uncertainty gives
a 0.5% scale floor. Independent doses mean raw count ratios alone still cannot
measure throughput. These image fits use exact conjugacy in the truncated
shared-field model. The masked finite-defocus error and allowable focus range
remain uncertified, requiring an aperture-edge focus sweep or finite-B image
fit. Stigmator location, certified radii, and instrument controls remain unconfirmed.

### Condenser recovery checkpoint

The first image-level physical condenser fit generates 22 GPU beam images at
two downstream analysers and combines their fitted covariance measurements
with the aperture-image covariances. Twelve condenser coordinates, ten source
covariance coordinates, and one common aperture-frame angle are free. The
expected invisible antisymmetric covariance direction is retained explicitly.

The likelihood passes at synthetic truth: training χ²=59.50/57 and beam-only
held-out χ²=14.18/18. Noiseless beam/aperture closure is below 5×10⁻¹⁰ noise σ,
including a comparison with the independently traced full column. Optimization
still exhausts 1,500 evaluations, giving training χ²=40.25 and held-out
χ²=471.82/18. `condenser_image_recovery.json` therefore marks recovery unaccepted;
its rank-12 profiled optical information is a local diagnostic at an
unconverged iterate.

The objective/downstream maps and radii are exact conditional calibrations,
and the diagram seed coincides with generating condenser geometry. Family 14
is a beam-only holdout because its aperture covariance enters training.
The next accepted fit needs source/optics reparameterization or stronger
optimization, a fully withheld beam/aperture family, and common calibration
uncertainty. Moving geometry also needs a clearance constraint or a partial
lens map: current broad trial bounds can place the aperture inside CL3,
whereas the fast map assumes the entire lens precedes it. The saved final
iterate has positive clearance, so this is an unresolved fitting-domain
requirement, not an explanation of the reported failed solution.

The revised `condenser_covariance_fit.json` removes the invisible source
coordinate from optimization while retaining its complete quantum-admissible
interval, constrains the aperture to remain outside CL3, and adds one common
0.5% radius uncertainty. Family 14 is now withheld in both measurement types.
The revised truth likelihood passes at χ²=57.68/54 training and 16.00/21 held
out. An SVD trust-region fit with JAX curvature corrections stops after a
bounded 100 iterations, with normalized gradient 5.13×10⁻⁴ and training
χ²=34.61/54; held-out χ²=8072.79/21 fails. Recovery remains unaccepted.

The truth Jacobian has a normalized condition number near two million. Its
weakest linear direction combines CL1/CL3 centres and lengths and predicts a
15.17σ combined held-out shift for a one-linear-σ training change. Large linear
uncertainties may leave the allowed geometry domain, so this diagnostic is not
a nonlinear profile. It identifies the next design target: independently
excited, captured CL1/CL3 states, including informative narrow-beam states in
training and separate new validation states. The already used heldout must not
silently become training data. A larger iteration budget alone does not add
information along that family.

The approximate geometry notebook supplies a source reference, lens centres,
and a condenser-aperture anchor after CL3, but no stigmator position, calibrated
upstream kick, or aperture translation. Its source line is an effective
reference below the accelerating tube, not a measured virtual-source plane.
The unchanged astigmatic input upstream of CL1 remains an experimental
preparation to establish, while aperture translation is optional for the
known-radius imaging method.

### Stronger condenser measurements and the stigmator contract

The new eight-setting training batch varies CL1/CL3 independently over a larger
synthetic excitation range and refocuses C-mini. Three additional settings are
reserved geometrically before noise generation. All 22 exposures use 49
independently propagated Gaussian modes, finite-pixel integration, and the
unchanged partially coherent source. They collect at least 99.9997% of the
incident power; all six-coordinate image fits pass with data-derived noise
weights. Selected narrow/wide checks pass both 9→25→49 mode and 8→16→32 pixel
refinements. Their largest image discrepancy is 0.00124 noise σ RMS and largest
joint observable bias 0.00227σ. These are vacuum images, with a clear path
assumed; the existing aperture metrology remains a separate measurement.

The enlarged likelihood contains the old 54 and new 48 training coordinates
once, including the shared radius prior once. It passes at synthetic truth
(χ²=122.80/102). The first optimizer attempts failed; a diagnosed geodesic-step
bug could turn a downhill Gauss–Newton direction uphill. The new solver
requires descent and positive predicted reduction and retries the original
Gauss–Newton step before reporting failure. All attempts and their separate
source/initialization assumptions remain in the recovery reports. The guarded
run leaves three starts with very large residuals. A fourth, using rough
geometry and a source estimated only from training data, reaches total
χ²=93.84935. Twenty bounded exact-Hessian steps lower this to 93.84910
(data-only χ²=93.82884/102), but the normalized gradient remains 0.00188.
Neither fit is stationary. Fresh held-out predictions are not evaluated.
Because the rough optical seed coincides with the generating design, this
nearby fit is not evidence of robust initialization or blind recovery.
The next bounded test should investigate source/optics parameterization and
curvature before spending more frames or optimizer iterations. Independent
Hessian-vector products and directional-gradient differences confirm the
large smooth optical/source curvature, rather than a quantum-completion or
aperture-boundary derivative. A source-profiled fit should retain the original
log-width/correlation likelihood and its complete covariance; a linear raw
moment fit would be an initializer, not the same statistical objective. See
`condenser_guarded_recovery.json` and `condenser_newton_refinement.json`.

The source-profile experiment separates the optimization difficulty. With
optics fixed, all eleven source/calibration nuisances reach stationarity in one
Newton step. Referencing the source covariance to one fixed acquisition
preserves the residuals, although fitting all twelve optical coordinates in
the outer problem still stops unconverged after thirty steps. Profiling the
six excitation-law coordinates alongside the source and calibration nuisances
then leaves six geometry coordinates in the outer problem. All twelve optical
parameters remain free in the combined fit, which converges in seventeen outer
steps at data χ²=92.2120/102, plus 0.3977 from the single radius prior; the
original-coordinate normalized gradient is 8.85×10⁻⁹. The full observed Hessian
is positive definite. See `condenser_geometry_profile_diagnostic.json`.

The selected training fit was written to an immutable selection record before
loading the fresh held-out observations. Their raw χ² is 166.54/18, while
including the local fitted-parameter prediction covariance gives 22.41/18;
principal beam-diameter errors are below 0.468%. The point prediction therefore
misses the very small image-statistical errors, although it meets the plan's
2% diameter tolerance. The prediction covariance still requires nonlinear
profile checks. CL1/CL3/C-mini length errors are +1.36%, +1.16%, and +0.78%,
with conditional local uncertainties of 1.04%, 0.88%, and 0.35%. Centre errors
are −3.85, −2.83, and −0.21 mm. Every optical error is below 2.24 local standard
errors, but those local errors are not nonlinear confidence intervals.
See `condenser_stationary_validation.json`.

At that frozen fitted source and optics, all six held-out propagation maps
also pass independent 9→25→49 Gaussian-mode and 8→16→32 pixel refinements.
The largest detector discrepancy is 0.000997 experimental noise σ RMS and
largest joint image-observable bias is 0.00123σ, so the point-prediction
residuals above are not explained by Gaussian discretization error. This
calculation uses an admissible source representative, retaining the unmeasured
covariance family. See `condenser_fitted_gaussian_validation.json`.

This is the first stationary conditional condenser fit. It continues from a
training solution near the generating optical seed; the independent offset
starts remain unsuccessful. The source is highly chirped at its chosen
reference plane and has an admissible Gaussian-mode decomposition. Its small
coordinate-dependent Hermitian margin does not imply proximity to the quantum
limit; an admissible representative has invariant eigen-emittances 1.253 and
1.353 times the minimum. The invisible source-covariance family remains free.
Hardware preparation, initialization, nonlinear profiles, and the joint
objective/projector/condenser fit remain open.

The first nonlinear CL1-centre check fixes that coordinate and refits the other
22 parameters. Moving CL1 by +2.667 mm gives a stationary profile with
Δχ²=0.8933; the −2.667 mm target has a feasible candidate at Δχ²=0.9513,
although its final profile gradient remains unresolved at numerical cost
resolution. Both prescribed two-sigma targets remain unresolved from their
cold starts. These results establish a near-equivalent physical family around
the fitted model, while leaving the two-sigma interval and global branches
open. The small one-sigma cost changes support the local CL1 uncertainty in
this neighborhood, not confidence coverage for every parameter.

The initial profile solver restricted steps to one experimental-noise unit,
which prevented far conditional seeds from progressing within thirty steps.
Uniformly scaling the entire residual and its stopping tolerance by 10⁻³
enlarges that numerical step allowance without changing relative data/prior
weights or the likelihood minimum. Every reported residual and gradient is
converted back to original units. See `condenser_cl1_profile.json` and
`condenser_scaled_cl1_profile.json`.

The paper uses an upstream elliptical aperture or defocused specimen pattern
to measure subsequent rotation; its aperture sequence is C1/VOA/C2/C3, unlike
the NEOARM diagram. It does not locate a NEOARM stigmator before CL1.
[Reference paper, §§II.2.1–II.2.2](https://arxiv.org/html/2607.29411v1)

An internal stigmator can instead be included explicitly as an unknown
quadrupole Q at an unknown plane:

\[
\Sigma_d=M_{\mathrm{down}} Q M_{\mathrm{up}}\Sigma_0
M_{\mathrm{up}}^T Q^T M_{\mathrm{down}}^T.
\]

The ideal covariance-only test retains local rank 19/19 when Q is after CL1
or CL3, releasing twelve lens coordinates, four rotationally invariant source
moments, and three quadrupole coordinates. Thus a later stigmator need not
remove upstream focusing information. For isolated hard-edge round lenses and
that round source, however, reversing both excitation bias and slope of a lens
before Q leaves the measurements unchanged, to below 4×10⁻¹⁵ in these tests.
These discrete polarity families remain even with local full rank. The current
nine-coordinate round-transport source quotient cannot be reused unchanged
once Q exposes its previously invisible coordinate.
See `condenser_stigmator_contract.json`.

Releasing all ten source moments retains ideal local rank 25, including twelve
optical and thirteen source/quadrupole coordinates, for either tested Q gap.
Holding the lenses before Q fixed reduces the profiled optical rank to eight
or four, because a changed upstream map F can be absorbed into a physical
source covariance through
\(\Sigma'_0=F'^{-1}F\Sigma_0F^TF'^{-T}\).
Finite simultaneous source, quadrupole, and upstream-lens changes reproduce
all downstream-only covariances to below 4×10⁻¹⁵; independent upstream scans
break those particular families. See `condenser_general_source_contract.json`.

The same formal rank survives with the two actual conditional analyser maps,
but none of the 54 frames per Q hypothesis is captured adequately on the
256×256 detector. At most 0.14% of the power reaches it, and column-normalized
conditioning is approximately 4×10⁸. Thus the algebra does not require aperture
measurements in every generic case, while this unrefocused scan supplies no
practical beam-only replacement. A refocused design with realistic pixel noise
and source stability is required. See `condenser_actual_analyser_contract.json`.

A common refocused patch repairs capture across both tested stigmator locations
but supplies too little optical information. Preparation-specific autofocus
finds a broader after-CL1 route; its 4-million-electron preflight predicts
CL1/CL3/C-mini length uncertainties 0.864%/0.358%/0.139%, rank 25, and at least
99.94% capture for 32 frames. The attempted Gaussian acquisition stops before
rendering or drawing noise because its composed matrices fail a symplectic
guard. Matrix-composition roundoff and transport accuracy need separate checks;
the preflight is not an acquired or fitted internal-stigmator dataset.
The [failure audit](../generated/neoarm_inversion/imaging_recipe/condenser_internal_beam_images_failure.json)
also finds a sampled C-mini mechanical angular RMS of 36.07 mrad and a
six-RMS envelope of 0.216 rad. Those values require a paraxial-accuracy and
intermediate-clearance check; nine sampled planes per condenser do not
certify continuous-column maxima or objective/projector interior angles.

For the later-stigmator preparation, informative settings appear outside the
largest connected region of the sampled grid. That grid does not rule out a
curved autofocus path connecting them. Reports are
`condenser_internal_beam_design.json`, `condenser_autofocus_routes.json`, and
`condenser_internal_beam_acquisition_design.json`.

## Shared immersed objective: implementation and current limits

The column now supports one objective field with strength, peak, width, and
asymmetry shared across the specimen. Mechanical tilts are converted to
canonical momenta using k=QB/(2p): πx=αx−ky and πy=αy+kx, with signed charge Q.
[Canonical momenta in a solenoid, Eq. 2](https://arxiv.org/html/2006.12948v2)

The shared-field information test releases all 34 optical parameters plus
source/calibration nuisances. Its numerical checks pass on the selected
frames, but its optimistic aperture total-flux block is not an established
measurement. The source-reference-plane/covariance transport gauge is exact;
those data cannot uniquely locate the physical source plane.

The whole-column alternative sums objective tails and controlled fields
before computing focusing, retaining overlap cross-terms. Fourth-order
symplectic propagation includes moving lens boundaries and physical endpoint
conversions, with a finite GPU Jacobian for all 34 parameters.

Across 55 probe frames, successive 64→128→256→512 steps/segment give maximum
image-coordinate shifts of 0.02672σ, 0.001769σ, and 0.0001123σ. With all controls
and the profile amplitude fixed, full-tail continuation changes rotation by
about 5.25°, magnification by −0.695%/−0.609%, and nominal 80k capture to about
90%. This is sensitivity to an unmeasured field hypothesis, not evidence that
the NEOARM has those long tails. Refocusing, held-out model comparison, and
fitting-range convergence remain necessary.

The notebook's ray figures now use that full overlapping field. The existing
renderer supplies true-z overviews, linear condenser/objective/projector zooms,
finite lens glyphs, and common-scale IL3 wobble panels. The smooth objective
glyph explicitly marks its FWHM; the cumulative Larmor strip integrates the
whole summed signed field. A circular Gaussian is specified by its canonical
60 nm sample reference radius and flat radial wavefront. Its actual radius
minima are located independently; a flat wavefront inside the objective need
not be a waist minimum. Mechanical 5 nm / 1 mrad basis rays include the
sample-field coordinate conversion and retain their orthogonal response.

Detector basis errors are below 3×10⁻¹⁴ relative to independent single-ray
Jacobians. The two-ray envelope agrees with covariance propagation to better
than 1.1×10⁻⁹; successive 128→256→512-step refinements change the nominal rays
by at most 4.02×10⁻⁶ and 2.55×10⁻⁷ of the local radius. The six SVG/PDF/300-dpi
PNG figure sets pass label-overlap checks and save source, array, and figure
checksums in `review/shared_field/ray_review.json`. They illustrate a specified
field hypothesis, not fitted full-column geometry.

## Computational handoff and remaining proof

All report names above resolve under
`examples/generated/neoarm_inversion/imaging_recipe/`. The notebook checks
source/array hashes, displays images and calibrated ray figures, and does not
launch fitting or bootstrap on opening.

| Task | Executable example |
|---|---|
| Cubic specimen rendering / image calibration | `neoarm_finite_pixel_specimen.py` / `neoarm_finite_pixel_metrology.py` |
| Physical PL1 / coupled IL3–PL1 fits | `neoarm_last_lens_physical_fit.py` / `neoarm_coupled_projector_fit.py` |
| Fresh branch images / joint refit | `neoarm_projector_branch_experiment.py` / `neoarm_projector_branch_recovery.py` |
| IL2 design / images / three-projector recovery | `neoarm_il2_extension_design.py` / `neoarm_il2_extension_images.py` / `neoarm_three_projector_recovery.py` |
| IL1 design / images / four-projector recovery | `neoarm_il1_extension_design.py` / `neoarm_il1_extension_images.py` / `neoarm_four_projector_recovery.py` |
| Five-input experiment / shared calibration audit | `neoarm_projector_five_input_ablation.py` / `neoarm_projector_calibration_audit.py` |
| Upstream map reconstruction with shared covariance | `neoarm_projector_peeling.py` |
| Relay design with finite-scenario checks | `neoarm_last_lens_robust_design.py` |
| Aperture capture / image fitting | `neoarm_aperture_capture.py` / `neoarm_aperture_image_fit.py` |
| Condenser images / revised unaccepted source–optics fit | `neoarm_condenser_image_recovery.py` / `neoarm_condenser_covariance_fit.py` |
| Fresh condenser design / Gaussian images / guarded fit | `neoarm_condenser_extension_design.py` / `neoarm_condenser_extension_images.py` / `neoarm_condenser_guarded_recovery.py` |
| Stationary condenser fit / frozen-selection checks | `neoarm_condenser_geometry_profile_diagnostic.py` / `neoarm_condenser_stationary_validation.py` |
| Same-column calibration images / joint endpoint fit / independent checks | `neoarm_condenser_endpoint_calibration.py` / `neoarm_condenser_endpoint_recovery.py` / `neoarm_condenser_endpoint_validation.py` |
| Objective/control compatibility audit | `neoarm_column_calibration_compatibility.py` |
| Displaced-geometry cold initialization | `neoarm_condenser_cold_start.py` |
| Moment-projected source / original-likelihood recovery | `neoarm_condenser_moment_initialization.py` |
| Cold moment-to-joint endpoint/condenser pipeline | `neoarm_condenser_cold_joint_endpoint.py` |
| Five-input endpoint calibration ablation | `neoarm_condenser_endpoint_ablation.py` |
| Fixed-CL1 nonlinear profiles / larger-step retry | `neoarm_condenser_cl1_profile.py` / `neoarm_condenser_scaled_cl1_profile.py` |
| Gaussian propagation at fitted source and optics | `neoarm_condenser_fitted_gaussian_validation.py` |
| Unknown internal-stigmator / polarity contract | `neoarm_condenser_stigmator_contract.py` |
| General source / actual-analyser contract | `neoarm_condenser_general_source_contract.py` / `neoarm_condenser_actual_analyser_contract.py` |
| Refocused unknown-stigmator design | `neoarm_condenser_internal_beam_design.py` |
| Preparation-specific autofocus routes | `neoarm_condenser_autofocus_routes.py` |
| Attempted internal-stigmator acquisition / prerequisite failure audit | `neoarm_condenser_internal_beam_images.py` / `neoarm_condenser_internal_beam_failure.py` |
| Shared objective / whole-column tails | `neoarm_shared_objective.py` / `neoarm_objective_tail_overlap.py` |
| Verified full-field ray figures | `neoarm_shared_field_ray_review.py` |
| Same-column 40k conjugate material-state design | `neoarm_shared_material_design.py` |

For example, reproduce the four-projector fit from the verified image artifacts
at the repository root:

```bash
OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
  .venv/bin/python examples/lens_inversion/neoarm_four_projector_recovery.py \
  --output examples/generated/neoarm_inversion/imaging_recipe/four_projector_recovery.recheck.json
```

The conditional condenser continuation uses the saved three-start moment
benchmark and measured endpoint calibration. It reruns the 29-coordinate
original-likelihood fit without a saved warm-fit seed:

```bash
OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
  .venv/bin/python examples/lens_inversion/neoarm_condenser_cold_joint_endpoint.py \
  --output examples/generated/neoarm_inversion/imaging_recipe/condenser_cold_joint_endpoint.recheck.json
```

These separate output paths preserve the frozen evidence. Repeating the cold
initialization itself uses `neoarm_condenser_moment_initialization.py`; its
three starts include the two unresolved outcomes described above.

The new image generators are `neoarm_last_lens_redesigned_images.py` and
`neoarm_pl1_il3_images.py`; each takes an `--output` report path. The
[research record](archive/neoarm_imaging_recipe_research_2026-09-05.md) retains
the earlier prerequisite and branch-search commands. Regenerating an input makes
dependent reports stale until rerun; a matching checksum is integrity evidence,
not proof of scientific correctness.

The next projector tests improve the weak IL1 directions and reduce frame count
with shared calibration uncertainty, unresolved-branch checks, and independent
held-out exposures. The tested IL3 rival is now separated by fresh measurements.
The conditional condenser likelihood now has a stationary cold-to-joint fit
from one of three starts; robustness across the remaining starts, nonlinear
profiles, and confirmed beam/collection geometry remain open. Completion still requires all requested
identifiable parameters, equivalent-solution profiles, two successive
optical-parameter refinements, and held-out image/optical predictions.

The corrected [reference paper](https://arxiv.org/abs/2607.29411) motivates
lens sweeps, rotation, and throughput measurements; its current-based examples
do not establish this NEOARM measurement contract. The earlier conditional
condenser fit used an unsupported independent ellipse-angle law and remains
historical evidence, as documented in [NEOARM_RECOVERY.md](NEOARM_RECOVERY.md).
