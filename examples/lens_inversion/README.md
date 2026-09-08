# Recorded lens controls and reference papers

Use the [microscope calibration notebook](../microscope_models/microscope_calibration_pipeline.ipynb)
for the maintained fitting procedure: generate Gaussian images, measure them,
and fit the microscope with TemGym components and JAX least squares.

This directory retains:

- `dac_lens_inversion.ipynb`: the original NEOARM free-lens control table.
- `neoarm_flc.py`: the small table reader used by the imaging-mode design tools.
- `paper/`: the supplied reference papers.

The old `neoarm_full_column_inversion.ipynb` and its research modules were
removed. Their last retained version is available in Git at `fc992bc`.
The recorded controls are inputs to simulation design, not a measured
calibration of the microscope geometry.
