# Microscope models: projector calibration and the virtual NEOARM

**Stage 1** records the projection system's response: at one HT and without a specimen, how far
the beam moves on the camera for a beam shift (CLA1) and a beam tilt (CLA2) at 16 lens settings, image-like and
diffraction-like evenly (around the 40k row and the DIFF row).
It then fits a model, unscaled, that predicts those moves for any IL1, IL2, IL3, PL1 DAC values in
the explored range. **Stage 2** (draft) adds two specimen frames and a second HT to make the model
physical: real units and lens positions. Every function takes named arguments only, and the
recording runs one step at a time.

```python
import acquire, analyse, figures, virtual_neoarm

microscope = virtual_neoarm.build(ht_v=60e3)      # as left at the console: 60 kV, MAG 40k, spot 3, ...
TEM3, detector = microscope.pyjem()               # at the microscope: from PyJEM import TEM3, detector

acquire.check_connection(TEM3=TEM3, detector=detector)      # harmless: reads, one frame, small kicks

recording = acquire.Recording(TEM3=TEM3, detector=detector, folder="projection_response")
recording.read_diff_row(camera_length_cm=40)
recording.take_over_lenses()
recording.plan_settings(beam_move_px=50, test_kick_units=300, max_kick_units=20000)
recording.dark_frame()
recording.measure_remaining()
recording.restore()
model = analyse.fit_projection_response(folder=recording.folder)   # shown in a notebook as its table

new = acquire.Recording(TEM3=TEM3, detector=detector, folder="new_settings")
new.take_over_lenses()
new.plan_new_settings(settings=analyse.new_settings(folder=recording.folder),
                      beam_move_px=50, test_kick_units=300, max_kick_units=20000)
new.dark_frame()
new.measure_remaining()
new.restore()
figures.prediction_test(model=model, folder=new.folder)
```

| File | Role |
|---|---|
| [stage1_session_sheet.md](stage1_session_sheet.md) | What to do at the microscope, by hand or with PyJEM. |
| [stage1_rehearsal.ipynb](stage1_rehearsal.ipynb) | The stage-1 session rehearsed on the virtual NEOARM at 60 kV. |
| [acquire.py](acquire.py) | Drives the microscope: `check_connection`, `Recording` (one step at a time), `take_frame`, and the draft `record_second_ht`. numpy + PyJEM only; the one file to copy to the microscope PC. |
| [analyse.py](analyse.py) | Saved frames → the model: `fit_projection_response`, `predict`, `compare`, `new_settings`, and the draft `fit_physical_projection`. |
| [model.py](model.py) | The TemGym projector model and the least-squares fit; the specimen measurements (grating lattice, gold rings). |
| [figures.py](figures.py) | Tables and figures in the paper style: recorded frames, the prediction test, the column with ring lenses. |
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
