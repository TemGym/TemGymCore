"""Stage 2's copy of the stage-1 fit (the stage-1 notebook has the version in use). Fit the projection-system response: from the saved frames to a model that predicts where the
beam moves on the camera. Every public step takes named arguments only.

    import fit

    model = fit.projection_response(folder=recording.folder)   # shown in a notebook as its table
    fit.predict(model=model, dac=[...], shift=(dx, dy))            # beam move on the camera, px
    fit.compare(model=model, folder=new.folder)                    # measured against predicted

**What is fitted.** The projection system's lenses (objective, IL1, IL2, IL3, PL1) are five TemGym
finite solenoids (:mod:`projection_lenses`). For each lens setting their ray map gives A, how far the
beam moves on the camera per metre of shift at the specimen, and B, per radian of tilt. Deflector
units enter through two unknown scales (and optionally a mix of shift and tilt), the camera through
an unknown rotation. Lens DAC values map to model controls by a fixed unit (:func:`controls`); the
fitted gains absorb any per-lens scale.

**From frames.** Dark-subtracted beam centres of every frame; for each +/- pair, half the
difference (the beam's starting position cancels); the wobble from the three measurements of the
40k row; then weighted least squares at that wobble, trying both camera handednesses (a camera image
can be mirrored relative to the column) and keeping the one at the noise level.

**Uses TemGym** in one place: for each setting, :func:`lens_responses` builds the lenses
(``lenses.column``) and asks TemGym for A and B (``temgym_core.run.position_and_angle_response``,
read off the ray map, a ``jax.jacfwd`` of the ray trace). Beam centres come from
``record.find_beam`` (from the shared ``examples/pyjem_session.py``). Everything else here
is numpy, scipy and JAX.

**Stage 1** (one HT, no specimen) fixes the predictions; the lens positions are not unique yet (the
model is unscaled). **Stage 2** (draft, ``stage2/``) adds two specimen frames and a second HT to make
it physical.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_dist

from stage2 import neoarm
from stage2 import projection_lenses as lenses
from stage2.stage1_record import find_beam


def measure_spot(*, frame, background=0.0):
    """Stage 2's short form of :func:`record.find_beam`: centre, rms radius and 0 if usable (1 if not)."""
    beam = find_beam(frame=frame, background=background)
    return beam.centre, beam.rms, 0.0 if beam.usable else 1.0
from temgym_core.constants import magnetic_strength_ratio
from temgym_core.run import position_and_angle_response

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------- conventions
#: The DAC -> control unit: the NEOARM 40k row at 200 kV maps onto the 40k-row strengths the
#: engineering start was made for (a choice of unit only; the fitted gains absorb any scale).
NEOARM_40K_200KV = np.array([neoarm.row_40k()[name] for name in neoarm.PROJECTOR_LENSES], float)
CONTROL_PER_DAC = lenses.REFERENCE_40K_CONTROLS[1:] / NEOARM_40K_200KV

# ---------------------------------------------------------------------------- the fitted numbers
# theta, 24 numbers:
#   theta[LENS]             16  the projection lenses' p (see projection_lenses.py)
#   theta[CAMERA_ROTATION]   1  rotation of the camera relative to the column (rad)
#   theta[LOG_SHIFT_SCALE]   1  log of the beam shift per deflector unit, relative to the nominal scale
#   theta[LOG_TILT_SCALE]    1  the same for the beam tilt
#   theta[MIX]               5  how much each deflector also does the other's job (below)
LENS = slice(0, lenses.N_PARAMETERS)
CAMERA_ROTATION, LOG_SHIFT_SCALE, LOG_TILT_SCALE = 16, 17, 18
#: Deflector mixing (complex, as real and imaginary parts). Per deflector unit, the shift
#: deflector (CLA1) also makes e1 times the tilt the tilt deflector (CLA2) makes, and CLA2 also
#: makes e2 times the shift CLA1 makes. ``Setup.mix_unit`` converts between the two commands'
#: nominal units (rad of tilt command per m of shift command for the same deflector units).
#: Index 23 is the angle (rad) between the tilt deflector's x axis and the shift deflector's.
#: All zero for pure, aligned deflectors; fitted when that is not known.
MIX = [19, 20, 21, 22, 23]
N_THETA = 24
#: Fitted at one HT: the lens numbers except the last gap. That gap, the camera rotation and the
#: two deflector scales each trade off exactly against the others at one HT, so they stay at
#: their start (stage 2 frees them).
CAMERA_GAP = lenses.GAPS.stop - 1
ONE_HT_FREE = [i for i in range(lenses.N_PARAMETERS) if i != CAMERA_GAP]


def strength_ratio(voltage, reference):
    """p(reference) / p(voltage): how much stronger a fixed magnetic field acts at ``voltage``."""
    return float(magnetic_strength_ratio(voltage, reference))


def controls(dac):
    """Model controls (objective first, fixed at 1) for IL1, IL2, IL3, PL1 DAC values."""
    return np.r_[1.0, np.asarray(dac, float) * CONTROL_PER_DAC]


# ---------------------------------------------------------------------------- the lens model
def _response(p, u, voltage, reference):
    """A and B (complex x + iy) for one setting, from TemGym (its ray map of the lenses)."""
    return position_and_angle_response(lenses.column(lenses.at_voltage(p, voltage, reference), u))


#: A and B (complex) for a batch of model controls and HTs, in JAX (usable inside a fit).
lens_responses = jax.jit(jax.vmap(_response, in_axes=(None, 0, 0, None)))


def responses(p, dac, voltages, reference):
    """A (detector m per specimen m) and B (detector m per rad), complex, for each setting.

    ``p`` are the 16 projector coordinates defined at HT ``reference``. Pads batches of one
    (a batch of one hits a JAX CPU bug).
    """
    dac = np.atleast_2d(dac)
    voltages = np.broadcast_to(np.atleast_1d(np.asarray(voltages, float)), (len(dac),))
    us = np.array([controls(w) for w in dac])
    if len(us) == 1:
        a, b = lens_responses(jnp.asarray(p), jnp.asarray(np.repeat(us, 2, 0)), jnp.asarray(np.repeat(voltages, 2)),
                              jnp.asarray(float(reference)))
        return np.asarray(a[:1]), np.asarray(b[:1])
    a, b = lens_responses(jnp.asarray(p), jnp.asarray(us), jnp.asarray(voltages), jnp.asarray(float(reference)))
    return np.asarray(a), np.asarray(b)


@dataclass
class Camera:
    """Camera at the binning used: ``n`` x ``n`` pixels of ``pitch`` metres."""
    n: int = 256
    pitch: float = 110e-6

    @property
    def coordinates(self):
        return np.arange(self.n) - (self.n - 1) / 2


# ---------------------------------------------------------------------------- the experiment
@dataclass
class Setup:
    """What the experimenter knows: the HT, the 40k row (DAC values read at the console), the
    camera, the beam-position wobble and the deflector unit ratio."""
    v0: float
    base_dac: np.ndarray                      # 40k-row DAC values at v0, read from the console
    camera: Camera = field(default_factory=Camera)
    wobble_px: float = 0.01                     # beam-position wobble, from the stage-1 repeats
    mix_unit: float = 4e4                       # rad of tilt command per m of shift command, per deflector unit


# ---------------------------------------------------------------------------- the fit
def half_differences(theta, data, setup):
    """Predicted half-differences (complex px) for every row of ``data`` under ``theta``."""
    us = jnp.asarray(np.array([controls(w) for w in data["dac"]]))
    a, b = lens_responses(theta[LENS], us, jnp.asarray(data["voltage"]), jnp.asarray(setup.v0))
    ratio = magnetic_strength_ratio(jnp.asarray(data["voltage"]), setup.v0)
    kind = jnp.asarray(data["kind"])
    ks, kt = jnp.exp(theta[LOG_SHIFT_SCALE]), jnp.exp(theta[LOG_TILT_SCALE])
    if theta.shape[0] > MIX[0]:                                   # each deflector a mix of shift and tilt
        e1, e2 = theta[MIX[0]] + 1j * theta[MIX[1]], theta[MIX[2]] + 1j * theta[MIX[3]]
        r = setup.mix_unit
        response = jnp.where(kind == 0, a * ks + e1 * b * kt * r,
                             jnp.exp(1j * theta[MIX[4]]) * (b * kt + e2 * a * ks / r))
    else:
        response = jnp.where(kind == 0, a * ks, b * kt)
    return jnp.exp(1j * theta[CAMERA_ROTATION]) * response * jnp.asarray(data["command"]) * ratio / setup.camera.pitch


def engineering_start(setup):
    """Engineering guess carried to v0 (objective re-excited), with neutral camera and kicks."""
    p = np.array(lenses.at_voltage(jnp.asarray(lenses.starting_guess()), setup.v0, 200e3))
    p[lenses.OBJECTIVE] = lenses.starting_guess()[lenses.OBJECTIVE]
    return np.r_[p, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def rotation_start(data, setup):
    """Closed-form start for the gains (and with two HTs the objective q) from the rotations:
    arg(response) = const - r(V) (q0 + sum s_i g_i u_i), modulo pi."""
    sigma = data["wobble_px"] / np.sqrt(2)
    sel = np.abs(data["measured"]) > 5 * sigma
    m = data["measured"][sel] * np.exp(-1j * np.angle(data["command"][sel]))    # x-kick equivalent
    u = np.array([controls(w)[1:] for w in data["dac"][sel]])
    r = np.array([strength_ratio(v, setup.v0) for v in data["voltage"][sel]])
    two_ht = len(np.unique(data["voltage"])) > 1
    design = np.c_[np.ones(len(u)), r, r[:, None] * u] if two_ht else np.c_[np.ones(len(u)), u]
    weight = np.abs(m) / sigma
    ref = np.argmax(np.where(data["kind"][sel] == 0, np.abs(m), 0))
    angle = np.angle((m / m[ref]) ** 2) / 2 + np.angle(m[ref])
    coef = np.zeros(design.shape[1])
    for _ in range(6):
        if np.any(coef):
            angle = angle + np.pi * np.round((design @ coef - angle) / np.pi)
        coef, *_ = np.linalg.lstsq(design * weight[:, None], angle * weight, rcond=None)
    theta = engineering_start(setup)
    theta[lenses.GAINS] = np.log(np.clip(-coef[-4:] / np.asarray(lenses.GAIN_SIGNS), 1e-3, None))
    if two_ht:
        theta[lenses.OBJECTIVE] = -coef[1]
    ref_row = np.argmax(np.where(data["kind"] == 0, np.abs(data["measured"]), 0))
    one = {k: data[k][[ref_row, ref_row]] for k in ("dac", "voltage", "kind", "command")}
    theta[CAMERA_ROTATION] = float(np.angle(data["measured"][ref_row] / complex(half_differences(jnp.asarray(theta), one, setup)[0])))
    return theta


def least_squares_fit(data, setup, free, start, extra=None):
    """Least squares of the half-differences over the ``free`` parameters from ``start``.
    ``extra(theta)``, if given, adds residuals (already divided by their uncertainties)."""
    free = np.asarray(free)
    sigma = data["wobble_px"] / np.sqrt(2)
    meas = jnp.asarray(data["measured"])

    def residual(x):
        theta = jnp.asarray(start).at[free].set(x)
        r = (half_differences(theta, data, setup) - meas) / sigma
        r = jnp.concatenate([jnp.real(r), jnp.imag(r)])
        return r if extra is None else jnp.concatenate([r, extra(theta)])

    f, j = jax.jit(residual), jax.jit(jax.jacfwd(residual))
    out = least_squares(lambda x: np.asarray(f(x)), start[free], jac=lambda x: np.asarray(j(x)),
                        method="lm", xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=2000)
    theta = np.array(start, float)
    theta[free] = out.x
    jac = np.asarray(j(out.x))
    chi2 = float(out.fun @ out.fun)
    dof = len(out.fun) - len(free)
    finite = np.isfinite(chi2) and np.all(np.isfinite(jac))
    chi2 = chi2 if finite else np.inf                     # a diverged fit loses every comparison
    covariance = np.linalg.pinv(jac.T @ jac) if finite else np.full((len(free), len(free)), np.nan)
    return dict(theta=theta, free=[int(i) for i in free], chi2=chi2, dof=dof, jacobian=jac,
                covariance=covariance, accepted=chi2 <= chi2_dist.ppf(0.99, dof),
                residuals=np.asarray(out.fun))


def fit_half_differences(data, setup, *, free, start=None):
    """Weighted least squares of the half-differences, from a data-driven start.

    Without a ``start``: the gains (and with two HTs the objective q) come from the closed-form
    rotation regression and the camera rotation from the strongest shift. That rotation is known
    only modulo pi (the start model may invert the image), so both are fitted and the better is
    kept. Returns theta, chi2, dof, the covariance of the free parameters and the acceptance at
    the 99 % point of chi2.
    """
    free = list(free)
    if start is not None:
        return least_squares_fit(data, setup, free, np.asarray(start, float))
    best = None
    for flip in (0.0, np.pi):
        start = rotation_start(data, setup)
        start[CAMERA_ROTATION] += flip
        result = least_squares_fit(data, setup, free, start)
        if best is None or result["chi2"] < best["chi2"]:
            best = result
    return best


# ---------------------------------------------------------------------------- from frames
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
    expected = NEOARM_40K_200KV / strength_ratio(ht_v, 200e3)
    return expected / np.asarray(base_dac, float)


def _setup(data, plan, pitch_m, shape, units):
    v0 = float(np.median(data["voltage"]))
    return Setup(v0=v0, base_dac=np.asarray(plan["base_dac"], float) * units,
                   camera=Camera(n=int(shape), pitch=pitch_m))


def _nominal_scales(data, setup):
    """Deflector units -> nominal metres / radians, chosen so the engineering lens model's
    predicted displacements match the measured ones in size (median over the clear responses).
    Only a starting scale: at one HT the fit absorbs any error in it."""
    start = engineering_start(setup)
    a, b = responses(start[LENS], data["dac"], data["voltage"], setup.v0)
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


def projection_response(*, folder, pitch_m=None, save=True, verbose=False):
    """Fit the projection-system response (stage 1, unscaled) to a recording folder from
    an :class:`record.Recording`. Returns the :class:`Model` (a dict, also saved as
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
            result = fit_half_differences(dict(fit_data, command=command, measured=measured), setup, free=ONE_HT_FREE + MIX)
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
    pure = fit_half_differences(dict(fit_data, command=command, measured=measured), setup, free=ONE_HT_FREE)
    gain = (pure["chi2"] - mixed["chi2"]) / max(mixed["chi2"] / mixed["dof"], 1.0)
    use_mixing = gain > chi2_dist.ppf(0.99, len(MIX))
    result = mixed if use_mixing else pure
    if verbose:
        print(f"  deflectors: {'mixed' if use_mixing else 'pure'} (chi2 drop from fitting the mixing: {gain:.1f}, "
              f"chance level {chi2_dist.ppf(0.99, len(MIX)):.1f})", flush=True)
    n = len(rows)
    misfit = np.abs(result["residuals"][:n] + 1j * result["residuals"][n:2 * n]) * wobble / np.sqrt(2)
    worst = [f"measurement {rows[k]['measurement']} ({rows[k]['setting']}) {rows[k]['kind']}: {misfit[k]:.3f} px"
             for k in np.argsort(misfit)[::-1][:3]]
    theta = result["theta"]
    mixing = [complex(theta[MIX[0]], theta[MIX[1]]), complex(theta[MIX[2]], theta[MIX[3]])]
    model = dict(theta=theta.tolist(), mirrored=mirrored, y_flipped=y_flipped, deflector_scales=scales,
                 deflectors="mixed" if use_mixing else "pure",
                 mixing_shift_deflector_tilt=[mixing[0].real, mixing[0].imag],
                 mixing_tilt_deflector_shift=[mixing[1].real, mixing[1].imag],
                 tilt_axis_angle_rad=float(theta[MIX[4]]), wobble_px=wobble,
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
    setup = Setup(v0=model["ht_v"], base_dac=np.asarray(model["base_dac"]) * units,
                    camera=Camera(n=model["frame_width"], pitch=model["pitch_m"]),
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
        out = out + np.asarray(half_differences(theta, rows, setup))[:len(dac)]
    out = np.asarray(out, complex)
    return np.conj(out) if model["mirrored"] else out


def compare(*, model, folder):
    """Measured vs predicted beam displacements for a recording at new settings
    (:meth:`record.Recording.plan_new_settings`).
    Returns rows of (setting, kind, measured px, predicted px, error px)."""
    data, _, rows = load_session(folder=folder)
    out = []
    for r, dac, kind, command, measured in zip(rows, data["dac"], data["kind"], data["command"], data["measured"]):
        kick = (command.real, command.imag)
        pred = predict(model=model, dac=dac, shift=kick if kind == 0 else (0, 0), tilt=kick if kind == 1 else (0, 0))[0]
        out.append(dict(setting=r["setting"], kind=r["kind"], measured=measured, predicted=pred,
                        error=abs(measured - pred)))
    return out


