"""Stage 2 analysis: lens and camera positions, and the deflectors in physical units.

**Draft** (2026-11-06): written for 60 -> 80 kV, not yet run there (the virtual NEOARM covers
200 and 195 kV only).

Everything is fitted together in one weighted least squares through the TemGym projector model:
the stage-1 beam moves (one HT), the stage-2 beam moves (both HTs), and the two specimen
measurements -- the magnification |A| from the grating image at the 40k row and the camera length
|B| from the gold rings at the diffraction setting.

    import stage2_analysis as A2
    result = A2.fit(stage1_folder, stage2_folder, grating_frame, gold_frame, period_m=230e-9)

The handedness, deflector type (pure or mixed), wobble and DAC units come from the stage-1 model
(``model.json``, written by ``stage1_analysis.fit_session``).
"""

import json
from pathlib import Path

import numpy as np

import projector_calibration as P
import stage1_analysis as A


def fit(stage1_folder, stage2_folder, grating_frame, gold_frame, *, period_m, period_sigma=0.002,
        ring_sigma=0.001, dark=0.0):
    """Fit everything. ``period_m`` is the grating's certified period and ``period_sigma`` its
    relative uncertainty; ``ring_sigma`` is the relative uncertainty of the camera length from the
    rings. Returns the lens and camera positions (mm, from the specimen) with 1-sigma
    uncertainties, the beam shift (m) and tilt (rad) per deflector unit at the stage-1 HT, the two
    specimen measurements, chi2 and whether the fit is at the noise level."""
    model = json.loads((Path(stage1_folder) / "model.json").read_text())
    plan2 = json.loads((Path(stage2_folder) / "plan.json").read_text())
    parts = [A.load_session(folder)[0] for folder in (stage1_folder, stage2_folder)]
    data = {k: np.concatenate([d[k] for d in parts]) for k in ("dac", "voltage", "kind", "command", "measured")}
    units, scales = np.asarray(model["units"]), model["deflector_scales"]
    data["dac"] = data["dac"] * units
    command = data["command"] * np.where(data["kind"] == 0, scales[0], scales[1])
    data["command"] = np.conj(command) if model["y_flipped"] else command
    data["measured"] = np.conj(data["measured"]) if model["mirrored"] else data["measured"]
    data["wobble_px"] = model["wobble_px"]
    setup = P.Setup(v0=model["ht_v"], v1=plan2["ht_v1"], base_dac=np.asarray(model["base_dac"]) * units,
                    diffraction_setting=np.asarray(model["diffraction"]) * units,
                    camera=P.Camera(n=model["frame_width"], pitch=model["pitch_m"]),
                    wobble_px=model["wobble_px"], mix_unit=scales[1] / scales[0])

    measured = A.magnification_and_camera_length(grating_frame, gold_frame, period_m=period_m, pitch_m=model["pitch_m"],
                         ht_v=model["ht_v"], dark=dark)
    specimen = P.specimen_scales(a_dac=setup.base_dac, magnification=measured["magnification"],
                             magnification_sigma=period_sigma, b_dac=setup.diffraction_setting,
                             b_abs=measured["camera_length_m"], b_sigma=ring_sigma)
    free = P.TWO_HT_FREE + [P.LOG_KS, P.LOG_KT] + (P.MIX if model["deflectors"] == "mixed" else [])
    result = P.fit(data, setup, free=free, specimen=specimen)
    positions, sigma = P.position_errors(result)
    theta = result["theta"]
    return dict(names=P.POSITION_NAMES, positions_mm=positions, sigma_mm=sigma,
                shift_m_per_unit=scales[0] * float(np.exp(theta[P.LOG_KS])),
                tilt_rad_per_unit=scales[1] * float(np.exp(theta[P.LOG_KT])),
                magnification=measured["magnification"], camera_length_m=measured["camera_length_m"],
                chi2=result["chi2"], dof=result["dof"], accepted=bool(result["accepted"]),
                beam_moves=len(data["kind"]), theta=theta, units=units, setup=setup)
