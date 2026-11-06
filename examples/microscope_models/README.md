# Microscope models: projector calibration and the virtual NEOARM

**Stage 1** measures, at one HT and without a specimen, how far the beam moves on the camera for a
beam shift (CLA1) and a beam tilt (CLA2) at nine lens settings. It then fits a model that predicts
those moves for any IL1, IL2, IL3, PL1 DAC values in the explored range. **Stage 2** (draft) adds two
specimen frames and a second HT to make the model physical: real units and lens positions.

```python
import acquire, analyse, figures, virtual_neoarm

microscope = virtual_neoarm.build(ht_v=60e3)      # as left at the console: 60 kV, MAG 40k, spot 3, ...
TEM3, detector = microscope.pyjem()               # at the microscope: from PyJEM import TEM3, detector

folder = acquire.stage1(TEM3, detector, "stage1", camera_length_cm=40)
model = analyse.stage1(folder)                    # shown in a notebook as its summary table
check = acquire.check(TEM3, detector, "check", analyse.unseen_settings(folder))
figures.prediction_check(analyse.compare(model, check))
```

| File | Role |
|---|---|
| [stage1_session_sheet.md](stage1_session_sheet.md) | What to do at the microscope, by hand or with PyJEM. |
| [stage1_rehearsal.ipynb](stage1_rehearsal.ipynb) | The stage-1 session rehearsed on the virtual NEOARM at 60 kV. |
| [acquire.py](acquire.py) | Drives the microscope: `stage1`, `check`, `take_frame`, and the draft `stage2`. numpy + PyJEM only; the one file to copy to the microscope PC. |
| [analyse.py](analyse.py) | Saved frames → the model: `stage1`, `predict`, `compare`, and the draft `stage2`. |
| [model.py](model.py) | The TemGym projector model and the least-squares fit; the specimen measurements (grating lattice, gold rings). |
| [figures.py](figures.py) | Tables and figures in the paper style: session frames, the prediction check, the column with ring lenses. |
| [virtual_neoarm.py](virtual_neoarm.py) | The virtual NEOARM, aligned at 60 or 200 kV (MAG 40k/45k rows, DIFF 15 cm – 1 m), behind PyJEM's names. |
| [projector_calibration_outline.md](projector_calibration_outline.md) | The mathematics: ray maps, gauges, the two stages. |
| [neoarm_flc_table.json](neoarm_flc_table.json), [neoarm_flc.py](neoarm_flc.py) | The recorded NEOARM free-lens table (MAG rows) and its reader. |
| [microscope.toml](microscope.toml) | The example microscope used by the core tests and the holography notebook. |
| `calibration_results/tilt_y_30_ray_acquisition/` | The virtual NEOARM's hidden truth and reference settings. |

Run the rehearsal on CPU in about a minute:

```bash
JAX_PLATFORMS=cpu jupyter nbconvert --to notebook --execute --inplace stage1_rehearsal.ipynb
```

The earlier exploration (moment-based inversion, condenser design, smiley/MAGICAL imaging,
projector-inversion figures) was removed in the 2026-11-06 cleanup; Git revision `b2d873e`
retains it.
