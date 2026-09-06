# A smaller calibration experiment

**Current full-microscope workflow:** [one two-voltage notebook](microscope_calibration_pipeline.ipynb). The results and standalone procedures below are historical.

The [archived standalone fitting notebook](old/projector_calibration_pipeline.ipynb)
runs independently: generate images, measure them, fit from independent guesses,
and check new predictions. The earlier two notebooks are preserved in commit
`985f7e4`; their history is summarized in [the adaptive notes](ADAPTIVE_CALIBRATION_NOTES.md).
The [numerical results](calibration_results/dataset_reduction_summary.json)
include failed starts, all parameter errors and the selected controls.

## What we tested

The successful earlier experiment used 334 images. It contained 28 complete
lens settings, each with one ordinary image, four calibrated tilt images and
four calibrated beam-shift images. We reduced these complete nine-image groups,
without duplicating exposures or adding the discarded measurements to a fit.

Selection ranked the measured-data Jacobian at the previous fitted model.
At each step it removed the setting causing the least loss of local information.
It did not use generating parameters. This is retrospective selection after
an exploratory calibration, not a procedure that discovers the required
controls from an entirely unknown instrument.

Every fit used the same 20 dispersed starting guesses. It first fitted the
16 optical parameters from signed displacements, then all 19 parameters from
all supplied responses. Both stages removed all soft penalties; the original
hard bounds remained. The new explicit `use_soft_priors=False` switch removes
position penalties too. The older regularization-weight option alone does not.

The initial acceptance rule required agreement of all 20 fits, no active
bounds, measurement RMS below 1.5, and maximum local relative standard error
below 1%. Agreement means costs within 0.01 chi-squared and named physical
parameters within 0.1%. This is a repeatability test, not a proof of uniqueness.

## Fewer images and fewer reliable starts are different problems

| Selection | Settings | Images | Starts reaching the same good fit | Largest local standard error |
|---|---:|---:|---:|---:|
| Original full dataset | — | 334 | 20/20 | 0.13% |
| Information only | 28 | 252 | 20/20 | 0.13% |
| Information only | 20 | 180 | 18/20 | 0.15% |
| Information only | 16 | 144 | 19/20 | 0.19% |
| Information only | 12 | 108 | 18/20 | 0.29% |
| Information only | 10 | 90 | 17/20 | 0.61% |
| Information only | 8 | 72 | 18/20 | 1.80% |
| Information only | 6 | 54 | 20/20 | 7.71% |
| Keep moderate settings | 10 | 90 | 20/20 | 0.82% |
| **Keep moderate settings** | **11** | **99** | **20/20** | **0.28%** |
| Keep moderate settings | 12 | 108 | 19/20 | 0.26% |
| Keep moderate settings | 13 | 117 | 18/20 | 0.22% |
| Keep moderate settings | 14 | 126 | 18/20 | 0.20% |
| Keep moderate settings | 16 | 144 | 18/20 | 0.19% |

Failed starts in these larger subsets were rejected during displacement
initialization, with errors around 1,000 noise units. They were clearly poor
fits, not competing solutions with similar image error. Four settings gave a
rank-15 data Jacobian for 19 parameters: small residuals could not determine all
parameters. Six settings were repeatable but imprecise and predicted omitted
settings poorly.

The local information score does not describe the whole optimization surface.
To retain useful moderate-excitation observations, a second elimination kept
at least four mixed-projector settings and two compensated-objective settings.
The 90-image version then recovered from 20/20 guesses in the original and
both pre-existing noise repeats. Its largest parameter error across those
repeats was 0.75%, but predictions could differ by 2.28 measurement-noise units.

We chose **eleven settings** for the final procedure. Adding one setting to
that 90-image protocol reduced the largest local standard error from 0.82% to
0.28%, while preserving 20/20 recovery. It gives a useful margin below 0.5%
local standard error. It is a tested compromise, not the smallest theoretically
possible dataset. Other selections or a different optimizer may do better.

## The fixed measurement procedure

The [protocol file](projector_measurement_protocol.json) specifies four moderate
mixed-projector settings, two compensated-objective settings and five wider
compensated settings. At each setting record:

1. One ordinary MAGICAL image.
2. Images with −3 and +3 mrad input tilts along each axis: four images.
3. Images with −10 and +10 nm beam shifts along each axis: four images.

This is **99 calibration images**, about **70% fewer than 334**. The specimen
stays fixed and visible; the illumination and its curved wavefront move together.
Both the known texture and the circular illumination boundary are used by
image analysis. The code retains the same exposure rule and 0.003-pixel frame
drift. Known inputs include the full specimen intensity template, pixel pitch,
voltage and exact input shift/tilt calibration.

## Fresh validation after choosing the protocol

We froze the eleven settings before generating these new images. The notebook
uses noise seed 7301, an independent repeat uses 8301, and eight separate
prediction settings use seed 7401. Each simulation uses seed +1 for its tilt
images and seed +2 for its beam-shift images. No old optical fit or true lens
parameters initialize image analysis or the microscope fit. Image registration
uses nominal image scales and a rotation search on the pixels.

| Check | Notebook run | Independent repeat |
|---|---:|---:|
| Calibration images | 99 | 99 |
| Starts reaching the same fit | 20/20 | 20/20 |
| Largest error across all 19 parameters | 0.425% | 0.258% |
| Error predicting 72 new images, in measurement-noise units | 1.477 | 1.575 |
| Prediction error including local fit uncertainty | 0.996 | 0.954 |

The parameter errors compare with truth; the local standard errors use data
sensitivity. They are different quantities. For the final row we propagate the
calibration covariance into the new predictions. If $r$ is the prediction error
already whitened by measurement noise, $J_v$ its prediction Jacobian in fitting
coordinates, and $\Sigma_p$ the local calibration covariance, the covariance of
$r$ is approximated by

$$\Sigma_r=I+J_v\Sigma_pJ_v^T.$$

We whiten by this full matrix, retaining correlations between predictions.
The fitted microscope is unchanged; this calculation only changes how its
uncertainty is reported. The approximation is local and linear. Both the raw
measurement-noise error and this uncertainty-aware error are shown.

The two fresh runs share the same 72 prediction images, which are independent
of both calibration runs and never enter a refit. The repeatability checks
cover a finite set of starts, not every possible model.

## Reproduce and interpret

Run [the notebook](old/projector_calibration_pipeline.ipynb) from top to bottom.
`reuse_completed=True` reopens completed results; `False` regenerates and
remeasures the images, then refits. The [workflow helper](projector_calibration_workflow.py)
keeps these steps separate. A separate noise repeat uses the same functions
with calibration seed 8301. No earlier notebook or generated dataset is needed.

The [reduction helper](projector_dataset_reduction.py) preserves the exploratory
selection and multiple-start comparison. It can use the original generated
measurements from the historical checkpoint. The versioned summary includes
all tested sizes; exploratory image arrays remain in `examples/generated/`.

This study leaves the lens model unchanged. It demonstrates a smaller fitting
procedure for that model, with somewhat less precision than the 334-image fit.
Some controls still require 3–4× reference excitation. Their real hardware
feasibility, control-law accuracy, coherent contrast and calibration errors
remain outside this experiment. Removing soft priors does not remove the hard
geometric bounds or these modelling assumptions.
