"""Turn the saved frames into the model of the projection system. Every function takes named
arguments only.

    import analyse

    model = analyse.fit_projection_response(folder=recording.folder)   # shown in a notebook as its table
    analyse.predict(model=model, dac=[...], shift=(dx, dy))      # beam move on the camera, px
    analyse.compare(model=model, folder=new.folder)                     # measured against predicted

**Stage 1** gives a predictor: where the beam moves on the camera for any lens setting of the
projection system (IL1, IL2, IL3, PL1 DAC values, within the explored range) and any change of the
beam-shift (CLA1) or beam-tilt (CLA2) deflector, in deflector units. Steps: dark-subtracted
centres of every frame; for each +/- pair, half the difference (the beam's starting position
cancels); the wobble from the three measurements of the 40k row; then the fit of ``model.py`` at one HT,
trying both camera handednesses (a camera image can be mirrored relative to the column) and
keeping the one at the noise level. Its lens positions mean nothing yet (it is unscaled); its
predictions do.

Stage 2 (draft) is in ``stage2/``.
"""

import json
from pathlib import Path

import numpy as np

import model as M
from acquire import measure_spot


#: A frame with more than this fraction of its signal within 3 px of the edge has its beam cut off.
EDGE_LIMIT = 1e-3


def load_session(*, folder):
    """Read a recording folder. Returns the measured half-differences as fit data (commands in
    deflector units, complex x + iy), the plan, and one row per pair for inspection."""
    folder = Path(folder)
    log = json.loads((folder / "log.json").read_text())
    plan = json.loads((folder / "plan.json").read_text())
    dark = [e for e in log if e["role"] == "dark"]
    background = float(np.median(np.load(folder / dark[0]["file"]))) if dark else 0.0
    centres = {e["index"]: measure_spot(frame=np.load(folder / e["file"]), background=background) for e in log}
    rows = []
    for number in sorted({e["measurement"] for e in log if e["measurement"] >= 1}):
        for kind in ("shift", "tilt"):
            plus = [e for e in log if e["measurement"] == number and e["kind"] == kind and e["role"] == "plus"]
            minus = [e for e in log if e["measurement"] == number and e["kind"] == kind and e["role"] == "minus"]
            if not plus or not minus:
                continue
            p, m = plus[-1], minus[-1]
            half = (centres[p["index"]][0] - centres[m["index"]][0]) / 2
            command = complex(*p["offset"])
            rows.append(dict(measurement=number, setting=p["setting"], kind=kind, dac=p["dac"], ht_v=p["ht_v"],
                             command=command, half=half, rms=centres[p["index"]][1],
                             edge=max(centres[p["index"]][2], centres[m["index"]][2])))
    cut = [r for r in rows if r["edge"] > EDGE_LIMIT]
    if cut:
        print("  dropped (beam cut by the frame edge): " + ", ".join(f"measurement {r['measurement']} {r['kind']}" for r in cut),
              flush=True)
    rows = [r for r in rows if r["edge"] <= EDGE_LIMIT]
    data = dict(dac=np.array([r["dac"] for r in rows], float), voltage=np.array([r["ht_v"] for r in rows], float),
                kind=np.array([0 if r["kind"] == "shift" else 1 for r in rows]),
                command=np.array([r["command"] for r in rows]), measured=np.array([r["half"] for r in rows]))
    return data, plan, rows


def _units(base_dac, ht_v):
    """Per-lens DAC unit: this microscope's 40k row is mapped onto the 40k row the lens model's
    engineering start expects at this HT. Only a choice of unit (the fitted gains absorb it), but
    it makes the start -- and so the fit -- work for any lens table."""
    expected = M.NEOARM_40K_200 / M.strength_ratio(ht_v, 200e3)
    return expected / np.asarray(base_dac, float)


def _setup(data, plan, pitch_m, shape, units):
    v0 = float(np.median(data["voltage"]))
    return M.Setup(v0=v0, base_dac=np.asarray(plan["base_dac"], float) * units,
                   camera=M.Camera(n=int(shape), pitch=pitch_m))


def _nominal_scales(data, setup):
    """Deflector units -> nominal metres / radians, chosen so the engineering lens model's
    predicted displacements match the measured ones in size (median over the clear responses).
    Only a starting scale: at one HT the fit absorbs any error in it."""
    start = M._engineering_start(setup)
    a, b = M.responses(start[:16], data["dac"], data["voltage"], setup.v0)
    clear = np.abs(data["measured"]) > 10 * setup.wobble_px
    scales = []
    for kind, response in ((0, a), (1, b)):
        sel = clear & (data["kind"] == kind)
        scales.append(float(np.median(np.abs(data["measured"][sel]) * setup.camera.pitch
                                      / (np.abs(response[sel]) * np.abs(data["command"][sel])))))
    return scales


class Model(dict):
    """The fitted projection-system response (stage 1, unscaled): a dict (as saved in
    ``model.json``) that a notebook shows as its summary."""

    def _repr_markdown_(self):
        return ("| projection-system response (unscaled) | |\n|---|---|\n"
                + "\n".join(f"| {name} | {value} |" for name, value in summary(model=self)))


def fit_projection_response(*, folder, pitch_m=None, save=True, verbose=False):
    """Fit the projection-system response (stage 1, unscaled) to a recording folder from
    an :class:`acquire.Recording`. Returns the :class:`Model` (a dict, also saved as
    ``model.json``): the fitted parameters, the deflector scales, the camera handedness, the wobble,
    chi2, whether the fit reached the noise level and the three worst-fitting measurements.
    ``verbose`` prints each handedness trial and the mixing test."""
    folder = Path(folder)
    data, plan, rows = load_session(folder=folder)
    state = json.loads((folder / "state_before.json").read_text())
    info = state["camera"].get("OutputImageInformation", {})
    pitch = pitch_m or (1.0 / info["PixelsPerMeter"] if info.get("PixelsPerMeter") else 110e-6)
    units = _units(plan["base_dac"], float(np.median(data["voltage"])))
    data["dac"] = data["dac"] * units
    setup = _setup(data, plan, pitch, state["camera"]["ImagingArea"]["Width"], units)

    # wobble from the x-kick measurements of the 40k row (three of them)
    at0 = np.array([r["setting"] == "40k row" for r in rows]) & (np.imag(data["command"]) == 0)    # x kicks, 40k row
    squares, dof = 0.0, 0
    for kind in (0, 1):                     # kick sizes differ a little per measurement: compare per deflector unit
        sel = at0 & (data["kind"] == kind)
        m, c = data["measured"][sel], data["command"][sel]
        squares, dof = squares + np.sum(np.abs(m - c * np.mean(m / c)) ** 2), dof + 2 * (len(m) - 1)
    wobble = float(np.sqrt(squares / dof) * np.sqrt(2))
    setup.wobble_px = wobble
    data["wobble_px"] = wobble

    scales = _nominal_scales(data, setup)
    setup.mix_unit = scales[1] / scales[0]          # both deflectors take the same raw units
    fit_data = dict(data, command=data["command"] * np.where(data["kind"] == 0, scales[0], scales[1]))
    # Handedness: the camera image may be mirrored relative to the column, and the deflector's y
    # axis may be mirrored relative to its x axis. Both are tried; the fit at the noise level wins.
    best = None
    for mirrored in (False, True):
        for y_flipped in (False, True):
            command = np.conj(fit_data["command"]) if y_flipped else fit_data["command"]
            measured = np.conj(fit_data["measured"]) if mirrored else fit_data["measured"]
            result = M.fit(dict(fit_data, command=command, measured=measured), setup, free=M.ONE_HT_FREE + M.MIX)
            if verbose:
                print(f"  camera mirrored {mirrored!s:5s} y deflector flipped {y_flipped!s:5s}: "
                      f"chi2/dof {result['chi2'] / result['dof']:.3g}", flush=True)
            if best is None or result["chi2"] < best[2]["chi2"]:
                best = (mirrored, y_flipped, result)
    mirrored, y_flipped, mixed = best
    # The smallest model at the noise level: pure deflectors unless the mixing lowers chi2 by more
    # than five extra parameters would by chance (99 %), on the scale of the observed noise.
    command = np.conj(fit_data["command"]) if y_flipped else fit_data["command"]
    measured = np.conj(fit_data["measured"]) if mirrored else fit_data["measured"]
    pure = M.fit(dict(fit_data, command=command, measured=measured), setup, free=M.ONE_HT_FREE)
    from scipy.stats import chi2 as chi2_dist
    gain = (pure["chi2"] - mixed["chi2"]) / max(mixed["chi2"] / mixed["dof"], 1.0)
    use_mixing = gain > chi2_dist.ppf(0.99, len(M.MIX))
    result = mixed if use_mixing else pure
    if verbose:
        print(f"  deflectors: {'mixed' if use_mixing else 'pure'} (chi2 drop from fitting the mixing: {gain:.1f}, "
              f"chance level {chi2_dist.ppf(0.99, len(M.MIX)):.1f})", flush=True)
    n = len(rows)
    misfit = np.abs(result["residuals"][:n] + 1j * result["residuals"][n:2 * n]) * wobble / np.sqrt(2)
    worst = [f"measurement {rows[k]['measurement']} ({rows[k]['setting']}) {rows[k]['kind']}: {misfit[k]:.3f} px"
             for k in np.argsort(misfit)[::-1][:3]]
    theta = result["theta"]
    mixing = [complex(theta[M.MIX[0]], theta[M.MIX[1]]), complex(theta[M.MIX[2]], theta[M.MIX[3]])]
    model = dict(theta=theta.tolist(), mirrored=mirrored, y_flipped=y_flipped, deflector_scales=scales,
                 deflectors="mixed" if use_mixing else "pure",
                 mixing_shift_deflector_tilt=[mixing[0].real, mixing[0].imag],
                 mixing_tilt_deflector_shift=[mixing[1].real, mixing[1].imag],
                 tilt_axis_angle_rad=float(theta[M.MIX[4]]), wobble_px=wobble,
                 wobble_dof=dof, chi2=result["chi2"], dof=result["dof"], accepted=bool(result["accepted"]),
                 ht_v=setup.v0, units=units.tolist(), base_dac=list(plan["base_dac"]),
                 pitch_m=pitch, frame_width=setup.camera.n, residuals=result["residuals"].tolist(), worst=worst)
    if save:
        (folder / "model.json").write_text(json.dumps(model, indent=1))
    return Model(model)


def summary(*, model):
    """Rows (name, value) describing a fitted model, for a table."""
    return [["camera image mirrored", model["mirrored"]],
            ["deflector y axis flipped", model["y_flipped"]],
            ["deflectors", model["deflectors"]],
            ["beam-position wobble (from the repeats)", f"{model['wobble_px']:.3f} px"],
            ["χ²/dof (≈ 1: fits to the wobble)", f"{model['chi2'] / model['dof']:.2f}"],
            ["worst-fitting measurements", "; ".join(model["worst"])]]


def new_settings(*, folder, n=8, seed=7):
    """``n`` lens settings the recording never used, each a random blend of three of its
    settings (so inside the explored range): rows of IL1, IL2, IL3, PL1 DAC values."""
    settings = np.array(list(json.loads((Path(folder) / "plan.json").read_text())["settings"].values()))
    rng = np.random.default_rng(seed)
    return np.round([rng.dirichlet(np.ones(3)) @ settings[rng.choice(len(settings), 3, replace=False)]
                     for _ in range(n)])


def predict(*, model, dac, shift=(0.0, 0.0), tilt=(0.0, 0.0)):
    """Predicted beam displacement on the camera (complex px, x + iy, camera axes) for a change
    of the shift (CLA1) and tilt (CLA2) deflectors from their centre values, in deflector units,
    at lens ``dac`` (IL1, IL2, IL3, PL1) and the model's HT."""
    units = np.asarray(model["units"])
    setup = M.Setup(v0=model["ht_v"], base_dac=np.asarray(model["base_dac"]) * units,
                    camera=M.Camera(n=model["frame_width"], pitch=model["pitch_m"]),
                    mix_unit=model["deflector_scales"][1] / model["deflector_scales"][0])
    theta = np.asarray(model["theta"])
    dac = np.atleast_2d(np.asarray(dac, float)) * units
    out = 0
    for kind, kick, scale in ((0, shift, model["deflector_scales"][0]), (1, tilt, model["deflector_scales"][1])):
        command = complex(*kick) * scale
        command = np.conj(command) if model.get("y_flipped") else command
        if command == 0:
            continue
        rows = dict(dac=np.vstack([dac, dac]), voltage=np.full(2 * len(dac), model["ht_v"]),
                    kind=np.full(2 * len(dac), kind), command=np.full(2 * len(dac), command))
        out = out + np.asarray(M.predict(theta, rows, setup))[:len(dac)]
    out = np.asarray(out, complex)
    return np.conj(out) if model["mirrored"] else out


def compare(*, model, folder):
    """Measured vs predicted beam displacements for a recording at new settings
    (:meth:`acquire.Recording.plan_new_settings`).
    Returns rows of (setting, kind, measured px, predicted px, error px)."""
    data, _, rows = load_session(folder=folder)
    out = []
    for r, dac, kind, command, measured in zip(rows, data["dac"], data["kind"], data["command"], data["measured"]):
        kick = (command.real, command.imag)
        pred = predict(model=model, dac=dac, shift=kick if kind == 0 else (0, 0), tilt=kick if kind == 1 else (0, 0))[0]
        out.append(dict(setting=r["setting"], kind=r["kind"], measured=measured, predicted=pred,
                        error=abs(measured - pred)))
    return out


