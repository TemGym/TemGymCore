# Simulated microscope (PyJEM-compatible)

A JEOL-like TEM you drive with the same calls as PyJEM 1.4, for rehearsing microscope sessions
before the real instrument. Behind those calls sit hidden TemGym optics, hidden electronics and a
camera. It knows nothing about a particular microscope or experiment: the optics, the lens tables
and the electronics are given to it.

```python
from temgym_core.pyjem_simulator import SimulatedMicroscope

microscope = SimulatedMicroscope.from_file("virtual_neoarm.json", ht_v=60e3)
TEM3, detector = microscope.pyjem()       # used exactly like PyJEM's
TEM3.EOS3().SelectFunctionMode(4)         # DIFF, at the camera length the console starts on
```

`from_file` reads a microscope described in plain physics: each lens's place, length and strength
per DAC unit (TemGym `HardEdgeSolenoid`s), the camera's place, the deflectors' shift and tilt per
DAC unit, the beam at the specimen and the console's lens table. Optionally also PLA (a TemGym `Deflector` at
the plane of PL1) and the objective's Seidel field aberrations, acting by where the beam is at the specimen (a ray version of TemGym's `SeidelLens` at the specimen plane: a centred beam stays round).
`diffraction_camera_length_m=` chooses the camera length the DIFF selector starts on, as a console left on it. Any other hidden column can be
given directly to `SimulatedMicroscope(OpticalTruth(column=...), ...)`.

or, for an unmodified script that does `from PyJEM import TEM3, detector`:

```python
from temgym_core.pyjem_simulator import install

with install(microscope):
    import my_pyjem_script
```

The virtual NEOARM's file, [`examples/virtual_neoarm/virtual_neoarm.json`](../../../examples/virtual_neoarm/virtual_neoarm.json),
is designed, not fitted, by [`design_projector.py`](../../../examples/virtual_neoarm/design_projector.py): a plausible objective, IL1-IL3 and PL1 with a lens table solved so every MAG row is in focus at its labelled magnification and every DIFF row has its labelled camera length; lenses a micron off the axis and a small hysteresis; the real NEOARM's CLA1/CLA2 response and camera. It is used by the scripts in
[`examples/microscope_inversion/at_the_microscope/`](../../../examples/microscope_inversion/at_the_microscope/) (run with `--simulated`).
Its TEM-mode condenser, [`examples/virtual_neoarm/condenser.json`](../../../examples/virtual_neoarm/condenser.json), is designed (not fitted)
by [`design_condenser.py`](../../../examples/virtual_neoarm/design_condenser.py) and loaded with `from_file(..., condenser_path=...)`.

## What is modelled

| Part | Model | Code |
|---|---|---|
| Condenser (hidden, optional) | TEM mode, from the electron source: source → CL1 (spot size) → CL3 (brightness) → condenser aperture (holes 1–4, Apt3 position) → CLA1/CLA2 as two `Deflector` planes, their kicks solved to give the specimen shift and tilt the electronics ask for → the probe corrector's round optics only (a transfer doublet of two thin `Lens`es: magnification −1, multipoles not modelled) → CM → the objective's front field (the objective mirrored) → specimen. Lenses are `ElectromagneticLens`; spot size sets CL1 and CL3 from a spot table; the current is the source's, cut by the aperture. | `condenser.Condenser` |
| Optics (hidden) | Any TemGym column after the specimen, given as a function of the relative lens strengths (objective, IL1, IL2, IL3, PL1) and the HT (`OpticalTruth.column`). Without a condenser the beam starts at the specimen plane: a round Gaussian beam of chosen width and spread of angles, centred where CLA1/CLA2 put it. Frames are rays traced through the column. Optional parts of it: a PLA `Deflector` at the plane of PL1 (`OpticalTruth.pla_z_m`, `pla_kick`) and an aberration plate at the objective. | `microscope.OpticalTruth`, `microscope.SpecimenBeam`; images by the core `source.sample_rays` and `evaluate.ray_image` |
| Electronics (hidden) | Linear 16-bit DAC laws for IL1–IL3 and PL1 (DAC value → relative strength) and CLA1/CLA2 (DAC values → specimen shift and tilt, scaled as 1/p with HT). | `electronics.py` |
| Modes | TEM MAG (selector rows) and TEM DIFF (camera-length rows), from the tables given. | `microscope.SimulatedMicroscope` |
| Specimens | None (vacuum) by default; any TemGym component at the specimen plane, set by hand (PyJEM has no call for it). | `microscope.SimulatedMicroscope.specimen` |
| Camera | Binned sensor, PSF, Poisson + background + read noise, gain/offset, signed 16- or 32-bit output, frame integration. | `camera.py` |
| Interface | A subset of PyJEM 1.4: `Lens3` (including free-lens control), `Def3`, `EOS3`, `HT3`, `Apt3`, `Stage3`, `detector.Detector`, and `TEMError` with `code = kind*1024 + cause`. Any other name raises `NotEmulatedError`. | `pyjem.py` |
| Strictness | A state the optics cannot render raises `UnmodelledStateError` listing every reason, for example STEM or LowMAG mode, a moved IS1, an inserted objective or SA aperture, an HT it was not given or a changed objective. | `microscope.SimulatedMicroscope.physical_setting` |

`microscope.truth.scoring_truth()` is for scoring only; nothing reached through the PyJEM calls exposes
positions or other z distances.

**Detector sampling.** Rays are binned on a grid `detector_samples_per_pixel` times finer than the
output pixels before the PSF. With 1, a spot of about a pixel locks to pixel centres; use 8 when
measuring small spots.

## What is left out

| Missing | Here | Next step |
|---|---|---|
| **Lens alignment** | A projection lens can sit off the axis (`offset_m`): its field acts about its own axis, so the beam lands off-centre, differently at each magnification. No tilt of a lens. | Tilted lenses. |
| **Lens DAC law** | Linear through zero. Optional hysteresis (`electronics.LensHysteresis`, the file's `hysteresis`): a lens's field lags behind its DAC value by the order of the values it had (a sum of backlash elements; no dependence on time). No offset, nonlinearity, normalisation or settling time. | Time effects; hysteresis measured on the real lenses. |
| **Free-lens control** | Switch-on keeps a stored value if there is one, else the present value; selector, mode and spot changes leave free-lens lenses alone. Not verified on an instrument. | Check on the instrument. |
| **Deflectors** | CLA1 = pure shift, CLA2 = pure tilt, linear. PLA, if the optics give it a plane, is a pure slope kick at the plane of PL1, plus a fixed off-centre kick in DIFF; every other deflector must stay neutral. | Mixing as a switch; PLA's own response from the lens fields. |
| **Camera** | Ideal by default: 32-bit output, never saturates. Its image is mirrored relative to the column's x + iy. | `CameraModel(data_bits=16, …)`. |
| **Illumination** | With a condenser: TEM mode only (alpha 0, TEM probe mode); CL2 (not on the NEOARM) must keep its value; spot sizes only from the spot table; no gun, gun lens or corrector; the condenser lenses' positions and the source are designed, not measured. Without one: given at the specimen plane (optionally sized by CL3). Objective and SA apertures must stay open. | Probe modes and alpha from their lens tables; fit the condenser to a condenser recording. |
| **Stage** | Tracked, no optical effect. | Stage position selects the specimen. |
| **Modes** | TEM MAG and DIFF. | Focused probe, STEM. |
| **Timing** | Every change is instant; no drift. | Time-stamped drift. |
