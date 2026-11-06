# Virtual microscope (PyJEM-compatible)

A virtual JEOL-like TEM column for rehearsing microscope sessions before the real instrument. You
drive it with the same calls as PyJEM 1.4. Behind those calls sit hidden TemGym optics, hidden
electronics and a camera:

```python
from temgym_core.virtual.pyjem import facade

TEM3, detector = facade(scope)            # scope: a VirtualMicroscope
TEM3.EOS3().SelectFunctionMode(4)         # DIFF
```

or, for an unmodified script that does `from PyJEM import TEM3, detector`:

```python
from temgym_core.virtual import install

with install(scope):
    import my_pyjem_script
```

[`examples/microscope_models/virtual_neoarm.py`](../../../examples/microscope_models/virtual_neoarm.py)
builds the virtual NEOARM, and
[`stage1_virtual_session.ipynb`](../../../examples/microscope_models/stage1_virtual_session.ipynb)
runs the stage-1 projector calibration on it through PyJEM calls only.

## What is modelled

| Part | Model | Code |
|---|---|---|
| Optics (hidden) | TemGym finite solenoid lenses (objective, IL1–IL3, PL1), one Gaussian source per high tension, sigmoid condenser aperture, post-aperture double deflector. Frames come from native ray tracing. | `scope.OpticalTruth`, `calibration.acquisition.setting_column` / `render_rays` |
| Electronics (hidden) | Linear 16-bit DAC laws for each lens (DAC value → relative control) and CLA1/CLA2 (DAC values → specimen shift and tilt, scaled as 1/p with HT). | `electronics.py` |
| Modes | TEM MAG (selector rows) and TEM DIFF (camera-length rows; in the virtual NEOARM designed so that A = 0 and \|B\| = camera length, with IL1 weakened). | `scope.VirtualMicroscope`, `virtual_neoarm.design_diffraction_table` |
| Specimens | None (vacuum) by default. `scope.specimen = CrossGrating(...)`: bars that absorb; rays are weighted by the transmission (in-focus bright-field image). `scope.specimen = PolycrystallineGold(...)`: the direct beam by rays plus one TemGym Gaussian beamlet per lit grain at its Bragg angle (111, 200, 220 rings). Set by hand: PyJEM has no call for it. | `specimens.py` |
| Camera | Binned sensor, PSF, Poisson + background + read noise, gain/offset, signed 16- or 32-bit output, frame integration. | `camera.py` |
| Interface | A subset of PyJEM 1.4: `Lens3` (including free-lens control), `Def3`, `EOS3`, `HT3`, `Apt3`, `Stage3`, `detector.Detector`, and `TEMError` with `code = kind*1024 + cause`. Any other name raises `NotEmulatedError`. | `pyjem.py` |
| Strictness | A state the optics cannot render raises `UnmodelledStateError` listing every reason, for example STEM or LowMAG mode, a moved IS1, an open aperture, an untabulated HT or a changed objective. | `scope.VirtualMicroscope.physical_setting` |

`scope.truth.scoring_truth()` is for scoring only; nothing reached through the PyJEM calls exposes
positions or other z distances.

**Detector sampling.** Rays are binned on a grid `detector_samples_per_pixel` times finer than the
output pixels before the PSF. With 1, a spot of about a pixel locks to pixel centres; use 8 when
measuring small spots (the stage-1 rehearsal does).

## Realism ledger: what is still left out

| Missing realism | Here | Real NEOARM | Next step |
|---|---|---|---|
| **Lens DAC law** | Linear through zero; no offset, nonlinearity, hysteresis, normalisation or settling time. | Unknown. | Hidden offsets and hysteresis as switches. |
| **Lens DAC range** | Relaxed: the 40k row is not the NEOARM's (PL1 24575 here, 64000 on the NEOARM). | PL1 sits near the top of its range. | NEOARM-anchored drives. |
| **DIFF table** | Designed on the hidden optics (15 cm – 1 m). | The console's own table; read it in DIFF mode. | `build_virtual_neoarm(diffraction_table=...)` with recorded values. |
| **Free-lens control** | Switch-on keeps a stored value if there is one, else the present value; selector, mode and spot changes leave free-lens lenses alone. | Not verified. `stage1_acquire.run` writes the values before switching on, so either behaviour is safe. | Check on the instrument. |
| **Deflectors** | CLA1 = pure shift, CLA2 = pure tilt, linear. | Mixing, balances and hysteresis unknown; the stage-1 analysis can fit a mix. | Mixing as a switch. |
| **Camera** | Ideal by default: 32-bit output, never saturates. Its image is mirrored relative to the column's x + iy (the stage-1 analysis detects this). | 16-bit or 12-bit words; orientation unknown. | `CameraModel(data_bits=16, …)`. |
| **High tension** | 200 and 195 kV only (one source per HT); lens DAC values stay fixed when HT changes. | Settling and drift. | Drift model. |
| **Objective** | Fixed at (OLc, OLf) = (45818, 39178). | Kept fixed in stage 1. | — |
| **Apertures** | Condenser holes 1 = 50 µm, 2 = 20 µm; all others open. | Hole numbering and the ARM aperture API unverified. | Map the real strip. |
| **Stage / specimen** | Stage tracked, no optical effect; the specimen is set by hand. The grating is amplitude-only (no phase contrast, no defocus fringes); gold grains scatter independently (no interference, no dynamical effects). | Real grids: alloy decoration, carbon support, specimen height. | Stage position selects grating, gold or a hole. |
| **Modes** | TEM MAG and DIFF. | LowMAG, STEM and a focused probe exist. | Focused probe. |
| **Model mismatch** | The truth uses the fit's own model family: hard-edge solenoids, geometric rays. | Real fields and wave optics differ. | Truths from field maps or wave rendering. |
| **Timing** | Every change is instant; no drift. | Moves and HT changes take time. | Time-stamped drift. |
