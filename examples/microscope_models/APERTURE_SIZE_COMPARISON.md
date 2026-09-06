# Two or three aligned aperture sizes

Use the [single full-microscope notebook](microscope_calibration_pipeline.ipynb). Its default is **4 and 10 µm**, with **0.2 µm assumed alignment uncertainty per axis**. A 20 µm hole is optional in the same function call.

Each hole is aligned once to the same beam at 200 kV and reference condenser commands `[1.75, 0.75, 1.0]`. We generate different residual offsets for the holes, then fit each centre independently. The centres remain fixed across lens sweeps and both voltages; the holes share one unknown axial plane. No movement scan, calibrated displacement or sub-nanometre centering observation is used. The finite alignment constraint is an experimental assumption, not a measured scan uncertainty.

The comparison reuses exactly the same open-aperture images and their full, correlated projector calibration uncertainty. It regenerates transmitted/reference counts for 39 condenser settings at each voltage. Two- and three-hole cases share the first two holes' alignment errors and count noise. Different holes have independent count noise; the covariance from their shared open reference is retained. Eight engineering/broad starts are attempted in every case.

The table gives **local 95% half-widths**, not guaranteed accuracy:

| Acquisition | Calibration current readings | Acceptable starts | Source z (mm) | Aperture z (mm) | Pre-field length |
|---|---:|---:|---:|---:|---:|
| 4, 10 µm | 234 | 6/8 | 0.790 | 0.0887 | 8.26% |
| 4, 10, 20 µm | 312 | 5/8 | 0.789 | 0.0886 | 8.26% |
| Three sizes; no alignment constraint | 312 | 6/8 | 0.790 | 0.0886 | 8.26% |

Adding 20 µm gives less than 0.2% improvement in these axial uncertainty estimates while increasing current readings from 234 to 312. Both primary fits predict the new images at RMS 0.983 and the new aperture currents at RMS about 0.94–0.95 in uncertainty units. The extra diagnostic removes the alignment constraint from the three-size fit; its axial precision remains similar. This checks that the axial result is not supplied by a tight centering assumption.

**Transverse aperture-centre directions remain ambiguous.** Some acceptable fits put the 10 µm hole's x coordinate near −0.085 µm, others near +0.154 µm. The latter is near the generating value, despite having slightly higher cost. A local interval around the lowest-cost fit can therefore miss the actual centre. The notebook and [portable comparison](calibration_results/aperture_sizes_summary.json) retain the spread across acceptable starts. We do not interpret this as unique recovery of every physical coordinate.

The axial and lens-length recovery is conditional on the round-solenoid/source model, known sizes/voltages/input calibrations and the selected hard bounds. This single acquisition does not establish repeated-noise coverage or real-microscope accuracy. In particular, the pre-field's roughly ±8.3% local 95% interval still exceeds the earlier 5% length-precision target if that target is interpreted at 95% confidence.

To repeat the comparison after running the notebook:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 .venv/bin/python examples/microscope_models/aperture_size_comparison.py
```

All fits, including rejected or unfinished attempts, are saved under `examples/generated/full_microscope_aperture_sizes/`. This comparison is an optional backend experiment, not a second fitting notebook.
