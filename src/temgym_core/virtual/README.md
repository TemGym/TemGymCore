# Virtual microscope (PyJEM-compatible)

A virtual JEOL-like TEM column for testing calibration protocols and the
microscope inversion before the real instrument. You drive it with the same
calls as PyJEM 1.4. Behind those calls sit a hidden optical truth, hidden
electronics and a camera:

```python
from temgym_core.virtual import install
from temgym_core.calibration.protocol import acquire_calibration

with install(scope):                      # scope: a VirtualMicroscope
    from PyJEM import TEM3, detector      # resolves to the virtual microscope
    folder = acquire_calibration(TEM3, detector, ...)
```

[`examples/microscope_models/virtual_neoarm.py`](../../../examples/microscope_models/virtual_neoarm.py)
builds the virtual NEOARM from the selected synthetic acquisition. It then
runs the whole loop:

1. acquire the calibration through PyJEM calls only;
2. import the frames;
3. measure them;
4. fit;
5. score the fit against the hidden truth.

## What is modelled

| Part | Model | Code |
|---|---|---|
| Optics (hidden) | The calibration model itself: finite solenoid lenses, one Gaussian source per high tension, sigmoid condenser aperture, post-aperture double deflector. Images come from native ray tracing; the renderer is bitwise identical to `render_ray_image`. | `scope.OpticalTruth`, `calibration.acquisition.setting_column` / `render_rays` |
| Electronics (hidden) | Linear 16-bit DAC laws for each lens (word → relative control) and CLA1/CLA2 (words → specimen shift and tilt, scaled as 1/p with HT). | `electronics.py` |
| Camera | Binned sensor, PSF, Poisson + background + read noise, gain/offset, signed 16- or 32-bit output, frame integration. | `camera.py` |
| Interface | A subset of PyJEM 1.4: `Lens3`, `Def3`, `EOS3`, `HT3`, `Apt3`, `Stage3`, `detector.Detector`, and `TEMError` with `code = kind*1024 + cause`. Any other name raises `NotEmulatedError`. | `pyjem.py` |
| Strictness | A state the optics cannot render raises `UnmodelledStateError` listing every reason, for example DIFF/STEM mode, a moved IS1, an open aperture, an untabulated HT or a changed objective. | `scope.VirtualMicroscope.physical_setting` |

The protocol and the inversion see only the PyJEM calls and a
`VendorCalibration` (deflector kick per word, camera gain, offset, noise,
pixel pitch, PSF). They never see positions or other z distances.

## Realism ledger: what the first version leaves out

Each row below should become a switch on the virtual microscope or a sweep in
the end-to-end example, not a hidden assumption. Evidence comes from the
moment-level screen
`virtual_neoarm_screen.py`, run on 2026-11-03 (since removed; archived in
`~/TemGymCore-archive-2026-11-05-projector-exploration.tar.gz`).

In that screen, source-z SD is the shot-noise local standard deviation. It
gives 0.38 mm for the selected plan, against 0.34 mm from the full pixel fit.

| Missing realism | First version | Real NEOARM / evidence | What it costs | Next step |
|---|---|---|---|---|
| **Lens DAC range (headroom)** | Relaxed: each lens's largest planned control maps to 60,000 words. | Recorded free-lens table: PL1 = 64,000 of 65,535 from 30k to 2M (at 25k it is 65,535). With the 40k row anchored there, the selected plan needs PL1 up to 84,019 words at 200 kV (43/142 rows impossible) and 156,253 at 195 kV (all 144 rows). | Every quick word-feasible variant keeps rank 47 but loses 5–15× in source-z SD: 2.0 mm when PL1 increases are mirrored and words are fixed at 195 kV; 6–11 mm for 197–199 kV; 5.7 mm when refocused at 20–25k at 195 kV. | Design the acquisition with the word limits as hard constraints (constrained optimal design), then switch to NEOARM-anchored drives. |
| **Focus at the second HT** | Any planned word is applied. | With the objective and PL1 at their 200 kV words, no focused setting above \|M\| = 30k exists within 16-bit words at 195 kV (closest about 25k). At 200 kV, 40k sits at the edge of the reachable range. | The original 195 kV projector baselines cannot be set; most 195 kV kicks leave the camera. | The constrained design must use reachable conditions per HT. |
| **NEOARM-like tables** | Selector rows only at 40k and 45k (the plan's baselines); one spot row. | NEOARM has no 45k row. A model-consistent 50k row (PL1 fixed, IL1 nearest NEOARM) lands 5–8% from the recorded IL words. | — | Build the full model-consistent table (plan milestone M5). |
| **Lens DAC law** | Linear through zero; no offset, nonlinearity, polarity switch, hysteresis, normalisation or settling time. | Unknown. The protocol's gauge (`word = control · w_ref / r_ref`) assumes linear-through-zero supplies. | Unknown. | Hidden offsets and hysteresis as switches; sweep the bias they cause. |
| **Camera saturation** | Ideal: 32-bit output, unit gain, no offset, never saturates. | A JEOL camera reads 16-bit words (the PyJEM example shows `EffectBits: 12`). At 10⁸ e⁻/image, 78% of reference frames peak above 3×10⁴ e⁻/px (median 5×10⁵, max 1.2×10⁷). | Single int16 frames: source-z SD ×7.5 (12-bit ×21). Integrating 32 int16 frames (256 for 12-bit) recovers it to ×1.7. | `CameraModel(data_bits=16, …)` plus exposure control and `frames` in the protocol (`frameIntegration`). |
| **Deflectors** | CLA1 = shift and CLA2 = tilt, diagonal and linear; the vendor calibration is exact unless biased (`vendor_calibration(kick_scale=…, kick_rotation_rad=…)`). | Real coil mixing, balances (`ShifBal`, `TiltBal`, `AngBal`), cross-talk and hysteresis are unknown. Kicks do not depend on the condenser state. | Word rounding: 0.01 nm and 0.2 µrad per word. | Bias sweeps; cross-talk; balances; kicks that depend on CM and HT. |
| **High tension** | Only 200 and 195 kV (the truth has one source per HT); the change is instant. | You confirmed that lens words stay fixed when HT changes; this is modelled. | — | Settling and drift after an HT change. |
| **Objective** | Fixed at (OLc, OLf) = (45818, 39178); any change is unmodelled. | The inversion keeps the objective fixed. | — | Objective-current axis (see the `explore/condenser-ht-pipeline` notes). |
| **Free lens control semantics** | Switch-on keeps the present word unless a value was stored; values stored while off apply at switch-on; selector changes leave FLC lenses alone; direct setters are cleared by selector or spot changes. | Not verified on hardware. | — | Check on the instrument; make the behaviour switchable. |
| **Apertures** | Condenser aperture holes 1 = 50 µm and 2 = 20 µm at fixed positions; every other aperture open. The ARM extension kind 0 maps to the condenser aperture. | Real hole numbering, which API the ARM200F uses (`SelectExpKind`/`SetExpSize`) and position units are unverified. | — | Map the real strip; model aperture positions. |
| **Stage** | Tracked; no optical effect (vacuum hole). | Stage z moves the specimen through the objective field (Larmor rotation). | — | Needed once a specimen is modelled. |
| **Modes** | TEM MAG only. | DIFF, LowMAG, STEM and a focused probe exist. | — | Focused probe (STEM) is the next mode. |
| **Dose** | Constant beam current; equal dose within each aperture pair. | Gun drift; the inversion requires equal dose per pair. | — | Dose drift as a switch. |
| **Model mismatch** | The truth uses the fit's own model family: hard-edge solenoids, geometric rays, no aperture diffraction. | Real fields and wave optics differ. | Results here are best-case. | Truths from TOML thin lenses, Gaussian/wave rendering, field maps. |
| **Camera geometry** | Rows down, columns right; no mirror, no camera rotation. | Unknown on the instrument. A mirrored camera cannot be absorbed by the round-lens fit. | — | Mirror/rotation switch. |
| **Timing** | Every change is instant; no drift. | Moves and HT changes take time; things drift. | — | Time-stamped drift model. |
| **PyJEM coverage** | Only the calls listed in `pyjem.py`. `snapshot(ext)` with encoded images is not emulated. | — | — | Add calls as protocols need them. |
