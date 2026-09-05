# NEOARM recovery: truth audit and coupled fit

4 September 2026. This report follows the [implementation audit](NEOARM_STATUS.md).

## Outcome

**Subsequent image-contract audit:** the condenser result below is a fit of
stipulated synthetic summaries. Its independent astigmatic-angle law is
inconsistent with the general entrance covariance released in that fit, and
the synthetic geometry differs substantially from the user's measured diagram.
The 0.24% result therefore cannot establish physically realizable condenser
recovery. [The imaging-recipe investigation](NEOARM_IMAGING_RECIPE.md) records
the corrected information tests and the next experiment. Historical numeric
results and acquisition arrays below are retained unchanged.

The historical **conditional probe/condenser benchmark produced a successful
coupled synthetic fit**. It fits 36 optical parameters and 27 condenser/source/calibration
nuisance parameters, using the GPU for the Gaussian forward calculation.
All parameter directions remain available in the final fit, and the original
physical bounds are retained.

The original full specimen-feature likelihood remains inconsistent at synthetic
truth. Its material/Fourier/layer/Bragg measurement families are excluded from
the historical benchmark as complete families, rather than selectively removing
noisy rows. The result establishes a downstream probe fit and a fit of the
stipulated condenser summaries; physically realizable coupled recovery remains
to be demonstrated with the corrected image likelihood.

## 1. Likelihood consistency

Three distinct problems emerged.

**Finite detector.** The original centroid predictor transported the centre of
an unbounded beam. A defocused beam can extend beyond the Medipix, so its measured
centroid differs from that prediction. Across all 126 saved vacuum/basis images,
the unbounded prediction was wrong by 4.81 pixels RMS. The finite-detector Gaussian
model, including affine geometry, pixel centres, and the existing detector blur,
agrees with the saved GPU images to 6.14e-9 pixel RMS, with a maximum error of
2.06e-8 pixel. These are deterministic image checks, separate from noise fitting.

**Noise clipping.** The old moment extractor clipped negative dark-corrected
pixels before calculating centroids. Read noise then contributes a positive
background. The new probe likelihood retains signed pixel values and propagates
Poisson/read-noise variance analytically, including the covariance between the
five raw moments. It reproduces the original synthetic noise seeds without
regenerating a bootstrap.

**Specimen peak assignments.** The archived 246-row training likelihood has
2888.49 σ RMS residual at its generating optical truth; correcting the finite
centroids alone leaves 2888.25 σ. The largest residual is a registered specimen
Fourier coordinate at 27,270 σ. In particular, the legacy extractor zeros the DC
disk before finding maxima, creating artificial rim peaks, then assigns the
nearest candidate without a rejection distance. A smooth Gaussian with no grating
produces reported vectors near (3.498, 1.497) Fourier pixels, matching the spurious
family in several defocused frames.

The legacy material estimator is preserved with the archived pipeline and is
**not readmitted**. Peak rejection alone also cannot establish an unbiased
frequency/layer/Bragg estimator. The recovery driver checks truth closure before
optimization and refuses to fit the inconsistent full dataset. The remaining
specimen-estimator work is therefore visible, not absorbed into fitted lenses
or an inflated covariance.

## 2. Initialization and coupled recovery

The new initialization reconstructs each finite-detector Gaussian from its
measured moments, then estimates signed sample-position and sample-angle
responses by generalized least squares. The propagated centre uncertainties
matter because a heavily clipped beam gives a weak estimate of its unbounded
centre. Held-out states are excluded from this optical initialization.

Excitation parameters are initialized before downstream geometry. A bounded
least-squares stage is followed by sensitivity-scaled minimization of the raw
joint chi-square. The scaling retains every direction, caps amplification of
weak directions, and transforms physical bounds into exact linear constraints;
it does not freeze a practical-rank subspace at the rough starting point.

The default starting perturbation is explicitly recorded in the report. Although
small in physical parameter coordinates, it produces 38,735 σ RMS downstream
and 11.26 σ condenser residuals. The larger historical perturbation still fails
in the tested local initialization; this is not an arbitrary-start global solver.

| Measurement block | At generating truth, σ RMS | After fitting, σ RMS |
|---|---:|---:|
| Downstream probe training | 1.018 | 0.996 |
| Downstream probe held out | 1.058 | 1.090 |
| Condenser training | 0.971 | 0.926 |
| Condenser held out | 0.876 | 0.837 |

There are 126 probe images, comprising 108 training and 18 held-out exposures.
The condenser contributes 407 training residuals and 21 held-out summary
residuals; its training block includes aperture counts, astigmatic summaries,
and four calibration-prior residuals. The final joint cost is 442.57 and the
final sensitivity-scaled optimizer converges. Intermediate evaluation-limit
terminations are retained in the report.

The condenser requires its **separate astigmatic-beam/aperture measurements**.
The downstream images are conditioned on calibrated specimen-plane inputs and
cannot identify CL1, CL3, or C-mini by themselves. Their physical parameters
share the column vector in the combined fit, with unknown upstream entrance
moments and calibration nuisance parameters released.

### Recovery and held-out prediction

All twelve CL1/CL3/C-mini lens parameters lie within 0.24% of their synthetic
values in this realization. The maximum held-out probe-state prediction errors
are:

| Quantity | Error | Planned limit |
|---|---:|---:|
| Magnification | 0.496% | 0.5% |
| Rotation | 0.000011° | 0.5° |
| Detector centroid | 0.00333 pixel | 0.5 pixel |
| Detector beam diameter | 0.0129% | 2% |

These limits apply to the demonstrated probe states. Diffraction-vector and
specimen-image acceptance remain separate. The magnification result is close
to its limit and needs additional noise realizations and calibration-uncertainty
tests before being treated as a robust experimental specification.

The fit does **not** recover every physical parameter uniquely or within 1%.
For example, the objective-postfield thickness differs by about 4%, while some
downstream axial positions reach their allowed bounds. The local,
nuisance-profiled Jacobian has 30 directions with singular value above one in
the declared parameter scales; a numerical-rank calculation gives 32 of 36.
These are scale- and noise-dependent local diagnostics, not counts of uniquely
determined named lens properties.

An explicit equivalent family shifts the source plane by ±1 mm and transports
the fitted entrance covariance by the corresponding free-space drift. The
predicted residual vector changes by less than 4e-10 σ. Source-plane position
therefore requires an external convention or measurement; more copies of the
same optical data cannot identify it.

## Excitation branches

Eight adjacent one-lens phase hypotheses were tested: ±π at the reference
excitation of IL1, IL2, IL3, and PL1, with the corresponding hard-edge thickness
change. Seeds are checked against physical bounds without clipping, and every
training state is fitted rather than relying on a single anchor matrix.

None reached a competing noise-consistent solution; their residuals remained
436–2357 σ RMS. **All eight optimizations hit their evaluation limit**, so these
are unresolved hypotheses, not exclusions or proof of a unique branch. The
software distinguishes a selected noise-consistent local solution from global
branch certification. A connected measured scan, stronger independent geometry,
or more effective branch profiling is still needed. Endpoint approach metadata
provide no measurements at intermediate DAC values.

## Reproduction and remaining work

The readable [notebook](neoarm_full_column_inversion.ipynb) displays the saved
result without launching a fit. From the repository:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python examples/lens_inversion/neoarm_recovery.py \
  --problem examples/generated/neoarm_inversion/recovery/problem.json \
  --output examples/generated/neoarm_inversion/recovery/recovery_report.json \
  --probes-only --fit
```

The one-thread BLAS setting avoids oversubscribing CPUs for the small optimizer
linear algebra; the Gaussian forward calculation remains on the GPU.
`--initial-parameters FILE` supplies an explicit JSON starting vector.
`--branch-audit-output FILE` adds the adjacent-branch challenge. Omitting
`--probes-only` checks the original full likelihood and stops at failed closure.

The numeric snapshot is checksum-verified and reconstructs whitelisted public
dataclasses, not executable notebook cells or pickles. Saved evidence is under
`examples/generated/neoarm_inversion/recovery/`: `probe_closure.json`,
`finite_fit.json`, `recovery_report.json`, and `branch_audit.json`. The original
acquisition arrays, inversion configuration, and instrument geometry are unchanged.

The subsequent [imaging-recipe investigation](NEOARM_IMAGING_RECIPE.md) supplies
a passing conditional cubic-object image estimator at 40k/80k. Before using
the coupled fit on the NEOARM, replace the independent-angle law with transported
condenser images, rebuild the measured geometry, anchor the branch, propagate
real input/probe calibration uncertainty, and bind safe hardware controls.
The present experiment assumes exact calibrated sample-plane states and physical
probe covariance; it does not yet propagate their uncertainty through the joint
fit. Condenser aperture/scale priors are included. Full-image realism, global
physical uniqueness, all-parameter refinement stability, and real-instrument
acceptance remain outstanding.

## Verification

161 distinct targeted tests passed, covering the finite-detector values and
derivatives, signed-moment noise covariance, numerical snapshot checksums,
bounded optimization, phase seeds, existing physical observables, condenser
likelihood/adapter, and the calibration workflow. The updated notebook executed
top-to-bottom in review mode. These software checks supplement the numerical
recovery results; they do not certify real-instrument accuracy.
