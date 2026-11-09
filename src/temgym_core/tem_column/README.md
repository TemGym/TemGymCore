# TEM column models

TemGym models of a TEM column, used in two places:

- as the **hidden optics** of the [virtual microscope](../virtual/README.md);
- as the **fit model** of the projector calibration
  ([`examples/microscope_models/model.py`](../../../examples/microscope_models/model.py)).

| Module | What it holds |
|---|---|
| `projector.py` | The projector (objective, IL1, IL2, IL3, PL1) as finite solenoids from 16 parameters (6 log gaps, 5 log lengths, objective strength, 4 log gains); `matrix5` gives its 5×5 ray map; `at_voltage` rescales it to another HT; `initial_guess` and `REFERENCE_40K_CONTROLS`, the engineering start and the 40k-row strengths it was made for. |
| `condenser.py` | The condenser (CL1, CL3, mini lens), its Gaussian source moments and the post-aperture double deflector. |
| `apertures.py` | Effective condenser parameters per HT and the per-hole aperture centres. |
| `high_tension.py` | `magnetic_strength_ratio`: how a lens's strength scales with the HT. |
| `render.py` | One setting as a native column (`setting_column`) and its ray-traced camera image (`render_rays`). |

The moment-based beam-image calibration that used to live here (as `temgym_core.calibration`)
was removed in the 2026-11-06 cleanup; Git revision `b2d873e` retains it.
