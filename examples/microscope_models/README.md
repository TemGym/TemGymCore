# Microscope calibration

Start with [microscope_calibration_pipeline.ipynb](microscope_calibration_pipeline.ipynb). This is the single current fitting notebook; run it from top to bottom using the project's Python environment with the `inversion` dependencies installed.

It demonstrates four steps:

1. Generate MAGICAL images and aperture-current measurements at **200 and 195 kV**, with objective current fixed.
2. Measure image scale, rotation, blur and calibrated shift/tilt responses; fit one shared projector geometry.
3. Measure specimen beams through that uncertain projector; fit the condenser, independent source states and aperture geometry, then update the complete microscope.
4. Freeze the fit, predict new measurements and compare geometry with the generating model.

The notebook has 11 cells, saved outputs, a 5×5 ray matrix and explicit named function inputs. Detailed optimization logs and generated arrays live in `../generated/full_microscope_aligned_apertures/`; successful steps are cached. The [portable result](calibration_results/full_microscope_summary.json) records predictions, geometry errors, local uncertainty and all starting-attempt outcomes. The [condenser starting model](microscope_condenser_starting_guess.json) contains literal engineering guesses, not a previous fit.

The default acquisition uses **4 and 10 µm aperture diameters**, aligned once with 0.2 µm assumed residual uncertainty per axis. Every hole has its own fitted transverse centre and shares an unknown axial plane. There is no calibrated movement scan. It uses 342 calibration images and 234 current readings; another 72 images and 48 readings test predictions. The [two-versus-three-diameter comparison](APERTURE_SIZE_COMPARISON.md) found little axial-precision benefit from adding 20 µm, and documents the remaining transverse-centre alternatives.

Condenser beam images use a retracted aperture; current measurements use each hole and an open reference. The projector images use 110 µm detector pixels; condenser images use 2× binning (220 µm pixels). Recorded commands, calibrations and noise assumptions are saved alongside the measurements.

This is a model-recovery demonstration, not a proof of unique real-microscope geometry. It assumes known voltages, calibrated specimen-plane shifts/tilts at both voltages, round coaxial solenoids, a Gaussian source waist, known lens order/polarity, an aperture between CL3 and the mini, and repeatable lens/aperture controls. No soft geometry penalties are used. The stated alignment accuracy supplies a finite-uncertainty constraint on each hole centre; it is an assumption to verify experimentally. The uncertainty handoff is locally Gaussian and retains correlations between both halves; original-pixel predictions check the approximation. Effective field lengths do not directly measure metal thickness.

The former standalone [projector](old/projector_calibration_pipeline.ipynb) and [condenser](old/condenser_calibration_pipeline.ipynb) notebooks are archived. Design notebooks and historical study notes remain available for context; they are not prerequisites.
