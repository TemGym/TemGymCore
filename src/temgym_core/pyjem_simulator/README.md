# Simulated microscope (PyJEM-compatible)

A JEOL-like TEM you drive with the same calls as PyJEM 1.4, for rehearsing microscope sessions
before the real instrument. Behind those calls sit hidden TemGym optics, hidden electronics and a
camera. It knows nothing about a particular microscope or experiment: the optics, the lens tables
and the electronics are given to it.

```python
from temgym_core.pyjem_simulator import SimulatedMicroscope

microscope = SimulatedMicroscope.from_file("virtual_neoarm.json", ht_v=60e3)
TEM3, detector = microscope.pyjem()       # used exactly like PyJEM's
TEM3.EOS3().SelectFunctionMode(4)         # DIFF
```

`from_file` reads a microscope described in plain physics: each lens's place, length and strength
per DAC unit (TemGym `HardEdgeSolenoid`s), the camera's place, the deflectors' shift and tilt per
DAC unit, the beam at the specimen and the console's lens table. Any other hidden column can be
given directly to `SimulatedMicroscope(OpticalTruth(column=...), ...)`.

or, for an unmodified script that does `from PyJEM import TEM3, detector`:

```python
from temgym_core.pyjem_simulator import install

with install(microscope):
    import my_pyjem_script
```

The virtual NEOARM's file, [`examples/virtual_neoarm/virtual_neoarm.json`](../../../examples/virtual_neoarm/virtual_neoarm.json),
is worked out in [`design.ipynb`](../../../examples/virtual_neoarm/design.ipynb) and used by
[`examples/microscope_inversion/stage1_rehearsal.ipynb`](../../../examples/microscope_inversion/stage1_rehearsal.ipynb).

## What is modelled

| Part | Model | Code |
|---|---|---|
| Optics (hidden) | Any TemGym column after the specimen, given as a function of the relative lens strengths (objective, IL1, IL2, IL3, PL1) and the HT (`OpticalTruth.column`). The beam starts at the specimen plane: a round Gaussian beam of chosen width and spread of angles, centred where CLA1/CLA2 put it. Frames are rays traced through the column. | `microscope.OpticalTruth`, `microscope.SpecimenBeam`; images by the core `source.sample_rays` and `evaluate.ray_image` |
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
| **Lens DAC law** | Linear through zero; no offset, nonlinearity, hysteresis, normalisation or settling time. | Offsets and hysteresis as switches. |
| **Free-lens control** | Switch-on keeps a stored value if there is one, else the present value; selector, mode and spot changes leave free-lens lenses alone. Not verified on an instrument. | Check on the instrument. |
| **Deflectors** | CLA1 = pure shift, CLA2 = pure tilt, linear. | Mixing as a switch. |
| **Camera** | Ideal by default: 32-bit output, never saturates. Its image is mirrored relative to the column's x + iy. | `CameraModel(data_bits=16, …)`. |
| **Illumination** | Given at the specimen plane; the condenser lenses, spot size and aperture are accepted but change nothing. Objective and SA apertures must stay open. | A condenser model. |
| **Stage** | Tracked, no optical effect. | Stage position selects the specimen. |
| **Modes** | TEM MAG and DIFF. | Focused probe, STEM. |
| **Timing** | Every change is instant; no drift. | Time-stamped drift. |
