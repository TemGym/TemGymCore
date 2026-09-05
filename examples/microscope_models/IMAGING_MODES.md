# Focused imaging modes, 40k–600k

The `magnification` mode in `microscope.toml` now forms a specimen-to-detector
image with **A = magnification × identity and B = 0**, including zero net
rotation, in the 200 kV thin-round-lens model. The ten saved magnifications
are 40k, 60k, 80k, 100k, 150k, 200k, 250k, 300k, 400k, and 600k.

Use the existing API, including the combined illumination/imaging mode:

```python
import jax
from temgym_core.microscope_model import MicroscopeModel

jax.config.update("jax_enable_x64", True)
model = MicroscopeModel.from_toml("examples/microscope_models/microscope.toml")
components = model.build_components({"parallel": 3.0, "magnification": 40_000})
# Intermediate values are also solved, for example 50_000 or 500_000.
```

`OperatingMode.interpolate_currents` remains a table interpolation primitive.
For this mode, `build_components` uses that interpolation as an initial guess
and refines IL1–IL3 to restore focus, magnification, and rotation. Requests
outside 40k–600k raise an error. Modes without the `conjugate_imaging` option
retain their previous behavior. The constraint currently supports this
five-lens, thin, round downstream topology; adding lens thickness or
astigmatism requires a corresponding forward-model design routine.

## What was solved

All physical z positions, focal-power coefficients, turns, and the objective
current were retained. PL1 is fixed at a model current of −3.5 A throughout
the imaging range. The three intermediate currents solve three constraints:
relay magnification, relay B = 0, and zero total signed ampere-turn excitation
through the downstream lenses. The latter gives zero rotation in the
existing `ElectromagneticLens` model. The nonlinear solve operates between
the fixed conjugate planes to avoid the cancellation in a high-magnification
full-column residual.

This PL1 setting was selected after comparing candidate branches with the
recorded control trends. It keeps all intermediate currents positive, avoids
an almost inactive IL3 at 600k, and gives a fixed final-projector magnification
of approximately 135×. The curve comparison is therefore a design diagnostic,
not independent validation or a measured current calibration.

The sample is at z = 325 mm and the detector at z = 1043.75 mm. The stages are:

| Stage | Object plane z (mm) | Image plane z (mm) | Signed radial magnification |
| --- | ---: | ---: | ---: |
| Objective postfield | 325.000 | 513.194 | −80.8233 |
| IL1–IL3 relay | 513.194 | 730.680 | Mag / (80.8233 × 134.9233) |
| PL1 | 730.680 | 1043.750 | −134.9233 |

Each stage is conjugate. Individual stages rotate their images; the total
sample-to-detector rotation cancels. The objective current remains
−2.472521284171875 model A. The final-projector current has changed from the
previous imaging table; the diffraction table is not redesigned here.

## Effective focal lengths

All entries below are in **mm**, using the model's
`1/f = Gc × (turns × current)^2` law at 200 kV. The objective postfield is
fixed at **2.271891 mm**, and PL1 at **2.286337 mm**.

| Magnification | IL1 | IL2 | IL3 |
| ---: | ---: | ---: | ---: |
| 40,000 | 18.326755 | 18.748942 | 32.858555 |
| 60,000 | 18.115995 | 16.107059 | 38.320834 |
| 80,000 | 18.123582 | 14.360330 | 43.036966 |
| 100,000 | 18.167883 | 13.096851 | 47.548504 |
| 150,000 | 18.285136 | 10.990164 | 59.058375 |
| 200,000 | 18.377913 | 9.628008 | 71.865995 |
| 250,000 | 18.449578 | 8.639717 | 86.825603 |
| 300,000 | 18.506464 | 7.873488 | 104.839630 |
| 400,000 | 18.591756 | 6.734753 | 155.261598 |
| 600,000 | 18.701441 | 5.258956 | 408.848764 |

| Lens | z relative to sample (mm) | Model current at 40k (A) | Model current at 600k (A) |
| --- | ---: | ---: | ---: |
| Objective postfield | 2.300 | −2.472521 | −2.472521 |
| IL1 | 213.524 | 2.205678 | 2.183471 |
| IL2 | 290.480 | 1.762284 | 3.327473 |
| IL3 | 352.980 | 2.137981 | 0.606103 |
| PL1 | 407.983 | −3.500000 | −3.500000 |

These are effective simulation parameters. In particular, no physical lens
thicknesses or axial field profiles were recovered: the active lenses here
have `Tc = 0`. The model currents and assumed turns are not measured NEOARM
electrical parameters. A large focal length for IL3 means weak focusing; it
does not require the physical lens or its neighboring drift to be that long.

## Comparison with the recorded NEOARM table

Both series are divided by their own value at 40k, without fitting a scale or
offset. This is a shape comparison under a provisional proportional-control
interpretation, not proof that current is proportional to DAC.

| Lens | Designed 600k / 40k | Recorded DAC 600k / 40k | Interpretation |
| --- | ---: | ---: | --- |
| IL1 | 0.98993 | 1.00981 | Both peak at low magnification; the model peak is smaller and its high-magnification value is lower. |
| IL2 | 1.88816 | 2.01589 | Both increase; the model requires less relative change. |
| IL3 | 0.28349 | 0.28407 | Close agreement across the curve; normalized RMS difference is 0.00957. |
| PL1 | 1.00000 | 1.00000 | Fixed across the range in both cases. |

The sequence and division of magnification are consistent with the general
JEOL description of an [intermediate lens system](https://www.jeol.com/words/emterms/20121023.062458.php)
and a [fixed final projector near 150×](https://www.jeol.com/words/emterms/20121023.072558.php).
That supports plausibility of the architecture, not the numerical field
parameters of the NEOARM. Measured focus and rotation sweeps are still needed
to calibrate the instrument. These are thin-lens results and do not validate
finite-field overlap, aberrations, or image contrast.

## Verification and reproduction

On 5 September 2026, direct differentiation of `run_to_end` at all ten saved
rows (with refinement disabled) and nine refined intermediate settings gave:

- Maximum elementwise error in `A / Mag` relative to identity: **2.48e−14**.
- Maximum operator norm of B: **8.35e−14 m**.
- The separate objective, relay, and final-projector stages also passed
  direct basis-ray conjugacy and composition tests at 40k, 200k, and 600k.
- **61 tests passed** across the new imaging tests and `test_microscope_model`.

Plain interpolation of the ten current rows, without refinement, gave up to
2.29% magnification error at the tested midpoints. This is why refinement is
part of `build_components` for the opted-in mode.

```bash
OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu .venv/bin/python examples/microscope_models/design_imaging_modes.py
OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q tests/test_conjugate_imaging_modes.py tests/test_microscope_model.py
```

Add `--write-table` to regenerate the ten TOML rows. The driver writes
`examples/generated/microscope_imaging_modes/report.json` and three figures
in SVG, PDF, and 300-dpi PNG: `neoarm_curve_comparison`, `lens_parameters`,
and `conjugate_rays`. The JSON includes all focal lengths, model currents,
full matrices, intermediate-value checks, and normalized recorded curves.
The older `microscope_design_curves.ipynb` contains a historical design with
its own geometry assumptions; use this driver to regenerate these modes.

## From focused modes to an unknown-geometry fit

The [projector calibration notebook](projector_calibration_pipeline.ipynb)
now follows one short experiment: synthetic MAGICAL images at 40k and 150k,
with ±2.5% lens wobbles, followed by image measurements and one microscope fit.
All 18 images are used. The fit receives the controls, measurements, known
specimen/detector values, named starting guesses, and
[editable position priors](projector_position_priors.json). It returns named
positions, lengths, strengths, and beam parameters, which are compared with
the generating model. The prior ranges are engineering assumptions.
Its finite-lens reference controls are generated separately from the
thin-lens table above.
