"""Stage 2 (draft): the stage-1 and second-HT recordings and the two specimen measurements fitted
together -- the lens and camera positions, and the deflectors in metres and radians."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import model as M
from analyse import load_session
from stage2.specimen_frames import magnification_and_camera_length


POSITION_NAMES = ("Objective", "IL1", "IL2", "IL3", "PL1", "Detector")


TWO_HT_FREE = list(range(17))


def positions_mm(theta):
    """Centres of the objective, IL1, IL2, IL3, PL1 and the detector, from the specimen (mm)."""
    gaps, lengths = np.exp(np.asarray(theta[:6])), np.exp(np.asarray(theta[6:11]))
    entrances = np.cumsum(gaps[:-1]) + np.r_[0.0, np.cumsum(lengths[:-1])]
    return 1e3 * np.r_[entrances + lengths / 2, gaps.sum() + lengths.sum()]


def _positions_jnp(theta):
    gaps, lengths = jnp.exp(theta[:6]), jnp.exp(theta[6:11])
    entrances = jnp.cumsum(gaps[:-1]) + jnp.concatenate([jnp.zeros(1), jnp.cumsum(lengths[:-1])])
    return 1e3 * jnp.concatenate([entrances + lengths / 2, (gaps.sum() + lengths.sum())[None]])


def fit_with_specimens(data, setup, *, free, a_dac, magnification, magnification_sigma, b_dac, b_abs, b_sigma):
    """:func:`model.fit` with the two specimen measurements as two more residuals: |A| at ``a_dac``
    and |B| (m/rad) at ``b_dac`` (both at HT v0), with relative uncertainties. The geometry is
    fitted first, the kick scales are then read off the two measurements, and everything is
    freed. Both camera-rotation starts are tried."""
    us = jnp.asarray(np.array([M.controls(w) for w in (a_dac, b_dac)]))

    def specimen_residuals(theta):
        a, b = M._responses(theta[:16], us, jnp.asarray([setup.v0, setup.v0]), jnp.asarray(setup.v0))
        return jnp.array([(jnp.abs(a[0]) / magnification - 1) / magnification_sigma,
                          (jnp.abs(b[1]) / b_abs - 1) / b_sigma])

    geometry = [f for f in free if f not in (M.LOG_KS, M.LOG_KT)]
    best = None
    for flip in (0.0, np.pi):
        start = M._rotation_start(data, setup)
        start[M.PHI] += flip
        theta = M._least_squares(data, setup, geometry, start)["theta"]
        a, _ = M.responses(theta[:16], a_dac, setup.v0, setup.v0)
        _, b = M.responses(theta[:16], b_dac, setup.v0, setup.v0)
        theta[M.LOG_KS] += np.log(abs(a[0]) / magnification)
        theta[M.LOG_KT] += np.log(abs(b[0]) / b_abs)
        theta = M._least_squares(data, setup, geometry, theta)["theta"]
        result = M._least_squares(data, setup, list(free), theta, extra=specimen_residuals)
        if best is None or result["chi2"] < best["chi2"]:
            best = result
    return best


def position_errors(result):
    """Lens and detector positions (mm) with 1-sigma uncertainties from the fit covariance."""
    free = np.asarray(result["free"])
    theta = jnp.asarray(result["theta"])
    g = np.asarray(jax.jacfwd(lambda x: _positions_jnp(theta.at[free].set(x)))(theta[free]))
    cov = g @ result["covariance"] @ g.T
    return positions_mm(result["theta"]), np.sqrt(np.diag(cov))


def fit_physical_projection(*, first_ht_folder, second_ht_folder, grating_frame, gold_frame, period_m,
                            period_sigma=0.002, ring_sigma=0.001, dark=0.0):
    """**Draft.** Stage 2: fit everything together -- the recordings at both HTs and the two
    specimen measurements, the magnification |A| from the grating image at the 40k row and the
    camera length |B| from the gold rings at the DIFF row. ``period_m`` is the grating's certified
    period and ``period_sigma`` its relative uncertainty; ``ring_sigma`` is the relative uncertainty of the camera length from the
    rings. Returns the lens and camera positions (mm, from the specimen) with 1-sigma
    uncertainties, the beam shift (m) and tilt (rad) per deflector unit at the first HT, the two
    specimen measurements, chi2 and whether the fit is at the noise level."""
    model = json.loads((Path(first_ht_folder) / "model.json").read_text())
    plan2 = json.loads((Path(second_ht_folder) / "plan.json").read_text())
    parts = [load_session(folder=folder)[0] for folder in (first_ht_folder, second_ht_folder)]
    data = {k: np.concatenate([d[k] for d in parts]) for k in ("dac", "voltage", "kind", "command", "measured")}
    units, scales = np.asarray(model["units"]), model["deflector_scales"]
    data["dac"] = data["dac"] * units
    command = data["command"] * np.where(data["kind"] == 0, scales[0], scales[1])
    data["command"] = np.conj(command) if model["y_flipped"] else command
    data["measured"] = np.conj(data["measured"]) if model["mirrored"] else data["measured"]
    data["wobble_px"] = model["wobble_px"]
    diff_row = np.asarray(plan2["diffraction"]) * units
    setup = M.Setup(v0=model["ht_v"], base_dac=np.asarray(model["base_dac"]) * units,
                    camera=M.Camera(n=model["frame_width"], pitch=model["pitch_m"]),
                    wobble_px=model["wobble_px"], mix_unit=scales[1] / scales[0])

    measured = magnification_and_camera_length(grating_frame=grating_frame, gold_frame=gold_frame,
                                               period_m=period_m, pitch_m=model["pitch_m"], ht_v=model["ht_v"],
                                               dark=dark)
    free = TWO_HT_FREE + [M.LOG_KS, M.LOG_KT] + (M.MIX if model["deflectors"] == "mixed" else [])
    result = fit_with_specimens(data, setup, free=free, a_dac=setup.base_dac,
                                magnification=measured["magnification"], magnification_sigma=period_sigma,
                                b_dac=diff_row, b_abs=measured["camera_length_m"], b_sigma=ring_sigma)
    positions, sigma = position_errors(result)
    theta = result["theta"]
    return dict(names=POSITION_NAMES, positions_mm=positions, sigma_mm=sigma,
                shift_m_per_unit=scales[0] * float(np.exp(theta[M.LOG_KS])),
                tilt_rad_per_unit=scales[1] * float(np.exp(theta[M.LOG_KT])),
                magnification=measured["magnification"], camera_length_m=measured["camera_length_m"],
                chi2=result["chi2"], dof=result["dof"], accepted=bool(result["accepted"]),
                measurements=len(data["kind"]), theta=theta, units=units, setup=setup)
