# Objective-current experiment

**Current full-microscope workflow:** [one two-voltage notebook](microscope_calibration_pipeline.ipynb). The results and standalone procedures below are historical.

The aim is to recover the physical parameters of the finite-solenoid condenser
model. The objective current changes both its pre-field and its post-field.
They are never actuated independently in this experiment.

## What a new image can establish

A prediction image can reject a numerical fit that happens to match the original
measurements. It cannot distinguish an exact family that predicts identical
observations under every available control setting. The fixed-objective
experiment has such a continuous family, even with new CL1/CL3/mini settings.

Changing the objective current is a new control axis. A single source and lens
geometry must explain the response across currents. The compensation that
preserved the fixed-objective images need not preserve these new images.

## Shared objective control, uncertain post-field response

Write the recorded relative objective current as $u=I/I_0$, with $u=1$ at the
previous projector calibration. The pre-field uses

$$q_{\mathrm{pre}}(u)=u q_{\mathrm{pre},0}.$$

The post-field reference geometry and downstream lenses retain that calibration.
We allow its extrapolation to new currents to have three unknown slopes:

$$
q_{\mathrm{post}}(u)=q_{\mathrm{post},0}[1+s_q(u-1)],\qquad
L_{\mathrm{post}}(u)=L_{\mathrm{post},0}e^{s_L(u-1)},\qquad
\Delta\phi(u)=s_\phi(u-1).
$$

The effective post-field centre stays fixed when its length changes. All three
departures vanish at the reference current. The nominal response is
$(s_q,s_L,s_\phi)=(1,0,0)$. The hidden generator uses $(1.08,0.20,0.03)$ to test a
modest error in that extrapolation. The independent fit starts at the nominal
response and does not read the hidden values.

These slopes represent uncertainty in the post-field response. This is not a
fit with an arbitrary unknown transfer matrix at each current, nor does it
release uncertainty in the entire reference projector calibration.

## Images and their independent analysis

The existing data contain 120 MAGICAL images at fixed objective current and 100
aperture-current readings. The current sweep establishes that the aperture is
after CL3 within the two tested placements. We retain those data and that ordering.

The new acquisition has six condenser settings at 98% and 102% objective current,
with five downstream views per state: 60 images. Three different condenser
settings at 97% and 103% supply 30 prediction images. The simulated setup selects
states with visible illumination. Selection belongs to generation; inference
receives the recorded commands, not source parameters or beam moments.

For each objective current, downstream lens commands are adjusted using the
nominal projector model to retain the intended focused and defocused views.
These planned commands are not claimed to satisfy hardware limits on a real
microscope. The true post-field differs from the nominal model, so the actual
maps need not equal the planned maps.

The analysis measures the pattern freely instead of assuming the planned maps
are exact. Let $P_{\mathrm{nom},j}(u)$ be a planned projector map for view $j$.
All rays use $(x,y,\theta_x,\theta_y,1)^T$. For the shared post-field change,

$$P_{\mathrm{actual},j}(u)=P_{\mathrm{nom},j}(u)E(u).$$

The known MAGICAL specimen is represented by a positive Gaussian mixture.
Multiplying it by Gaussian illumination and transporting by the round map $E$
still produces a Gaussian mixture. Fourteen quantities describe it:

- illumination centre and width: three quantities determining component weights;
- two scale/rotation maps: four quantities determining how component positions
  map to output position and angle;
- output position/angle offsets: four quantities;
- a round position/angle covariance: three quantities.

The analysis estimates these quantities and exposure/background values directly
from the five images. It receives only pixels, the known specimen/detector,
nominal maps and baseline image measurements for initialization. It does not
receive the true condenser, actual post-field response or actual changed maps.
The microscope fit predicts the fourteen measured quantities, using their full
marginal covariance.

There is no phase retrieval in this pipeline, and illumination is not assumed
parallel. Images use Poisson counts plus read noise. Aperture transmission retains
the existing TemGym Gaussian `SigmoidAperture` calculation.

## Fitting and checks

The main fit releases all 22 condenser/source/aperture parameters plus the three
post-field slopes. There are no soft priors and no fixed pre-field thickness.
Hard bounds retain positive lengths, the declared strength branch and the
aperture beside CL3. The source remains shared across objective settings.

Several starts span the old exact family and perturb other optical parameters.
A comparison fit holds the nominal post-field response fixed, which is an
intentionally imperfect extrapolation for this generated dataset.

Prediction checks freeze each fit before comparing with the independently
analysed 97%/103% images. A targeted ambiguity check changes pre-field strength
along the old equivalent family and refits the three post-field slopes. This
checks that particular family; it is not a profile over all other geometry
parameters or an exhaustive count of minima.

Local sensitivity rank and local covariance describe information near the fitted
solution. They do not prove global uniqueness. The final geometry comparison
reads the hidden parameters only after fitting and prediction checking.

## Saved result

The 25-parameter fit has full local sensitivity rank. Three of six starts
converge to the same best solution; two fail badly and one reaches the evaluation
limit about 7.8 chi-squared units above the best. All starts remain in the report.
This is not a demonstration of globally reliable optimization.

| Quantity | Result |
|---|---:|
| Joint calibration residual RMS | 1.039 |
| New measured-pattern RMS, measurement noise only | 1.252 |
| New measured-pattern RMS, including local fitted uncertainty | 1.136 |
| Frozen-model prediction RMS over the 30 new images' pixels | 0.983 |
| Largest relative error among four lens lengths | 0.316% |
| Pre-field length: truth / fit | 1.0000 / 0.9991 mm |
| Source z: truth / fit | −325.000 / −325.197 mm |

Local one-standard-error estimates are 0.0069 mm for pre-field length and 1.21 mm
for source z. The observed errors in this one noise realization are smaller than
those uncertainties. The lateral aperture centre remains poorly measured:
the fit is (0.085, −0.111) µm, versus (0.200, −0.150) µm in the generator; local
standard errors are about (0.981, 1.561) µm. Full local rank is not the same as
precise recovery of every parameter.

The recovered post-field slopes are approximately (1.0801, 0.2002, 0.0265).
Holding the imperfect nominal response fixed produces calibration RMS 63.97
and new measured-pattern RMS 249.6. Reference calibration alone therefore cannot
be extrapolated blindly in this generated example.

Old-family alternatives with pre-field lengths 0.5415 and 1.5285 mm acquire
extra chi-squared values about 215,065 and 162,200, even after refitting their
post-field slopes. Those alternatives were indistinguishable at fixed objective
current. The new axis rejects them strongly within the assumed response model.

The pixel prediction check freezes all condenser and post-field parameters.
Only dose/background estimates from independent image analysis accompany the
predicted patterns. The uncertainty calculation retains correlations between
prediction states through the fitted parameter covariance.

The executed [archived condenser notebook](old/condenser_calibration_pipeline.ipynb)
contains the pipeline and images. Full numerical results are saved in
[condenser_objective_summary.json](calibration_results/condenser_objective_summary.json).
