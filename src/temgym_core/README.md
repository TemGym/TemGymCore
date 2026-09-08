# Reading TemGymCore

Start the microscope fitting procedure in the
[calibration notebook](../../examples/microscope_models/microscope_calibration_pipeline.ipynb).
Its steps are **Gaussian images → image measurements → matrix fit → prediction checks**.
The [calibration code map](calibration/README.md) explains the functions behind those steps.

The reusable physics lives here:

| Modules | Purpose |
|---|---|
| `components.py` | Optical components, including `ThickLens`, `HardEdgeSolenoid` and apertures |
| `run.py` | Propagation through components and the affine 5×5 `column_matrix` |
| `ray.py`, `gaussian.py`, `source.py`, `propagator.py` | Rays, Gaussian beams, source constructors and free propagation |
| `evaluate.py`, `spatial_evaluate.py`, `grid.py` | Detector grids and coherent Gaussian summation, including spatial grouping |
| `coherent_image.py`, `coherent_moments.py` | Detector integration, image moments and Gaussian covariance transport |
| `intensity.py` | The explicit faster intensity approximation |
| `multiscale_sample.py` | The known MAGICAL specimen and its sampled representation |
| `jax_least_squares.py`, `shared_least_squares.py` | JAX fitting and shared measurement uncertainty |
| `microscope_model.py`, `imaging_modes.py`, `plotting.py` | Microscope configuration, imaging settings and display |

The remaining modules provide constants, coordinate transforms, transfer
matrices, aberrations, potentials and small array/tree utilities.

A solenoid uses the same component interface as the other lenses:

```python
from temgym_core.components import HardEdgeSolenoid, Plane
from temgym_core.run import column_matrix

column = (
    HardEdgeSolenoid(z=0.01, length=0.003, q=0.5),
    Plane(z=0.1),
)
transfer = column_matrix(column, source_z=0.0)
# transfer acts on [x, y, theta_x, theta_y, 1].
```

`q` is the signed integrated Larmor angle in radians; positions and lengths
are in metres. The same column propagates rays or Gaussian beams through
`run_to_end`. The component's own `matrix` is its linear 4×4 map; `column_matrix`
includes the free-space gaps and offsets in the 5×5 map.

## Core cleanup

The older research stack and NEOARM notebook are recoverable from Git at
`fc992bc`. Removing their unused APIs and tests, and moving the few shared
helpers into the modules above, reduced top-level Python from **57 to 28 files**
and approximately **49,900 to 16,600 lines**. The calibration package is counted
separately. Package convenience exports now load on demand and report import
errors instead of silently swallowing them.

Solenoid matrices and JAX derivatives match pre-cleanup snapshots to floating
point precision. The propagated Gaussian source mean is unchanged and its
covariance agrees to numerical precision. Across the targeted runs, 295 distinct tests pass, covering the current
calibration, core components, Gaussian sources, plotting and imaging modes.
One pre-existing deflector test used obsolete argument names; its call now
matches the unchanged analytic matrix API.
The built wheel includes all seven calibration presets and no retired modules.
Full Gaussian acquisition and multiple-start fitting were not repeated for
this structural cleanup.
