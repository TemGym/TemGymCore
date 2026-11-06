# Microscope models: stage-1 projector calibration and the virtual NEOARM

Stage 1 measures, at one HT and without a specimen, how far the beam moves on the camera for a
beam shift (CLA1) and a beam tilt (CLA2) at nine lens settings. It then fits a model that predicts
those moves for any IL1, IL2, IL3, PL1 DAC values in the explored range.

| File | Role |
|---|---|
| [stage1_at_the_microscope.md](stage1_at_the_microscope.md) | The session sheet: what to do at the microscope, by hand or with PyJEM. |
| [stage1_acquire.py](stage1_acquire.py) | The session script. numpy + PyJEM only; copy it to the microscope PC. |
| [stage1_analysis.py](stage1_analysis.py) | Saved frames → beam moves → the predictive model (`fit_session`, `predict`, `compare`). |
| [projector_calibration.py](projector_calibration.py) | The fit: TemGym projector model, deflector scales and mixing, weighted least squares; also anchors and stage 2. |
| [projector_figures.py](projector_figures.py) | Paper-style figures: the column with ring lenses, frames, fits. |
| [stage1_virtual_session.ipynb](stage1_virtual_session.ipynb) | The whole session rehearsed on the virtual NEOARM through PyJEM calls. |
| [virtual_neoarm.py](virtual_neoarm.py) | Builds the virtual NEOARM (MAG 40k/45k rows, DIFF 15 cm – 1 m). |
| [projector_calibration_outline.md](projector_calibration_outline.md) | The mathematics: ray maps, gauges, anchors, the HT step. |
| [neoarm_flc_table.json](neoarm_flc_table.json), [neoarm_flc.py](neoarm_flc.py) | The recorded NEOARM free-lens table (MAG rows) and its reader. |
| [microscope.toml](microscope.toml) | The example microscope used by the core tests and the holography notebook. |
| `calibration_results/tilt_y_30_ray_acquisition/` | The virtual NEOARM's hidden truth and reference settings. |

Run the rehearsal on CPU in about a minute:

```bash
JAX_PLATFORMS=cpu jupyter nbconvert --to notebook --execute --inplace stage1_virtual_session.ipynb
```

The earlier exploration (moment-based inversion, condenser design, smiley/MAGICAL imaging,
projector-inversion figures) was removed in the 2026-11-06 cleanup; Git revision `b2d873e`
retains it.
