# Calibration models

TemGym models of a TEM column, used in two places:

- as the **hidden optics** of the [virtual microscope](../virtual/README.md);
- as the **fit model** of the stage-1 projector calibration
  ([`examples/microscope_models/projector_calibration.py`](../../../examples/microscope_models/projector_calibration.py)).

| Module | What it holds |
|---|---|
| `projector_model.py` | The projector (objective, IL1, IL2, IL3, PL1) as finite solenoids from 16 parameters (6 log gaps, 5 log lengths, objective strength, 4 log gains); `matrix5` gives its 5×5 ray map; `at_voltage` rescales it to another HT. |
| `condenser_model.py` | The condenser (CL1, CL3, mini lens), its Gaussian source moments, the post-aperture double deflector, and `magnetic_strength_ratio` for HT changes. |
| `apertures.py` | Effective condenser parameters per HT and the per-hole aperture centres. |
| `acquisition.py` | One setting as a native column (`setting_column`) and its ray-traced detector image (`render_rays`). |
| `presets/beam_measurement_protocol.json` | The reference (gauge) projector controls of the 40k row. |

The moment-based beam-image calibration that used to live here (`workflow`, `inference`,
`checks`, `gaussian`, `viewer`, `recorded`, `protocol`) was removed in the 2026-11-06 cleanup;
Git revision `b2d873e` retains it.
