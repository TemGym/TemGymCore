# Configuring and focusing a microscope

`Microscope` holds named native TemGym components. Current tables contain
amperes. Loading a file or selecting a table does not run an optimisation.

```python
import jax
from temgym_core.microscope import Microscope, find_imaging_condition

jax.config.update("jax_enable_x64", True)
microscope = Microscope.from_toml("examples/microscope_models/microscope.toml")
initial = microscope.with_settings(parallel=3.0, magnification=40_000)
solution = find_imaging_condition(
    initial,
    magnification=40_000,
    vary=["IL1", "IL2", "IL3"],
    bounds=(0.01, 5.0),
    rotation=0.0,
)
focused = solution.microscope
matrix = solution.matrix  # 5×5, ordered x, y, theta_x, theta_y, 1
print(matrix[:2, :2])     # A = 40,000 × identity
print(matrix[:2, 2:4])    # B ≈ 0
```

The solve changes only the selected component parameters. A bare lens name
means its `current`; use names such as `lens.focal_length`, `solenoid.q`, or
`steer.def_x` for other native controls. Bounds use the selected parameters'
units, with one lower and upper bound per parameter when necessary.

`column_matrix` uses `jax.jacfwd` on native ray propagation. Our existing
bounded JAX least-squares solver then differentiates the imaging residuals
with respect to the selected parameters. We still need least squares:
derivatives describe the response; the solver finds settings that meet the target.
There is no separate thin-lens chain or SciPy solve here.

The residual contains the eight entries of the A/B mismatch. They are scaled
in coordinates at the first adjustable component, using the differentiated
fixed upstream optics. This avoids nearly dependent magnification/focus errors
at high magnification. Failed solves raise an error; `solution.residual` stores
these scaled errors and `solution.matrix` is the physical laboratory-frame map.

For a nonzero input ray, set `reference=(x, y, theta_x, theta_y)`. The matrix is
its local linearisation, with offset `F(reference) - J @ reference`. Optionally
set `image_position=(x, y)` to constrain where that ray lands, adding two
position residuals scaled by 1 mm. This lets steering controls centre the image.

For an experiment, wobble a current without automatically refocusing:

```python
wobbled = focused.with_currents(IL1=1.05 * focused.components["IL1"].current)
```

`with_components(IL1=replace(lens, x0=...))` accepts native lens parameters.
It also adds apertures or replaces the sample. Each update returns a new
microscope. `microscope.column` is the ordered component tuple for `run_to_end`.
The loaded sample is a `Plane`; source and aperture choices remain explicit.
Their original TOML specifications are accessible in `microscope.config`.

`microscope.tables["magnification"].at(50_000)` returns named interpolated
currents. Refocus explicitly to achieve an intermediate magnification.
Out-of-range interpolation and conflicting tables raise errors. No mode aliases,
normalised-current wrappers, largest-current merging or NPZ model bundles remain.

Native lenses handle the beam voltage. TOML Gc is converted once at the file's
reference voltage; its units differ from native Gc. To design a focused setting
at another voltage, pass `voltage=...` to `find_imaging_condition`. To simulate
only a voltage wobble, change the beam voltage and keep the components fixed.

There is no fixed lens count or thin-lens restriction. The solve uses each
component's implemented ray behaviour, including finite lenses, solenoids,
stigmators and deflectors. Component entrance positions stay fixed. For finite
bodies, supply bounds that preserve a physically valid column geometry.

For nonlinear optics, meeting the local matrix target does not establish
image quality away from the reference ray. Aperture transmission, interference
and finite-field aberrations need separate forward-image checks. The selected
controls may not be sufficient to meet the target, and a local solution is not
a uniqueness proof. The calibration notebook keeps its existing fitting workflow.

## Saved 200 kV settings

The ten stored rows span 40k–600k and satisfy A = magnification × identity,
B = 0 and zero total rotation. All are effective simulation parameters, not
measured NEOARM currents, field profiles or physical lens thicknesses.

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


Reproduce the settings and matrix checks from the repository root:

```sh
JAX_PLATFORMS=cpu .venv/bin/python examples/microscope_models/design_imaging_modes.py
```

Add `--write-table` to update the TOML. The script now only designs and checks
these settings; the old curve-comparison and report-generation experiment was removed.

The older diffraction table is unchanged. Its verification notebook still fails
the strict specimen-plane `A = 0` assertion; this was reproduced with both the
original and simplified implementations.
