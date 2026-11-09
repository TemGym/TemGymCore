"""Stage 2 (draft): the stage-1 and second-HT recordings and the two specimen measurements fitted
together -- the lens and camera positions, and the deflectors in metres and radians.
Uses TemGym only through :mod:`fit` (the lens model's ray maps)."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from stage2 import stage1_fit as F
from stage2 import projection_lenses as lenses
from stage2.specimen_frames import magnification_and_camera_length


POSITION_NAMES = ("Objective", "IL1", "IL2", "IL3", "PL1", "Detector")


TWO_HT_FREE = list(range(F.LENS.stop)) + [F.CAMERA_ROTATION]   # all lens numbers and the camera rotation


def _positions(theta):
    _, _, centres, total = lenses.geometry(theta[F.LENS])
    return 1e3 * jnp.concatenate([centres, total[None]])


def positions_mm(theta):
    """Centres of the objective, IL1, IL2, IL3, PL1 and the detector, from the specimen (mm)."""
    return np.asarray(_positions(jnp.asarray(theta)))


def fit_with_specimens(data, setup, *, free, a_dac, magnification, magnification_sigma, b_dac, b_abs, b_sigma):
    """:func:`fit.fit_half_differences` with the two specimen measurements as two more residuals: |A| at ``a_dac``
    and |B| (m/rad) at ``b_dac`` (both at HT v0), with relative uncertainties. The geometry is
    fitted first, the kick scales are then read off the two measurements, and everything is
    freed. Both camera-rotation starts are tried."""
    us = jnp.asarray(np.array([F.controls(w) for w in (a_dac, b_dac)]))

    def specimen_residuals(theta):
        a, b = F.lens_responses(theta[F.LENS], us, jnp.asarray([setup.v0, setup.v0]), jnp.asarray(setup.v0))
        return jnp.array([(jnp.abs(a[0]) / magnification - 1) / magnification_sigma,
                          (jnp.abs(b[1]) / b_abs - 1) / b_sigma])

    geometry = [f for f in free if f not in (F.LOG_SHIFT_SCALE, F.LOG_TILT_SCALE)]
    best = None
    for flip in (0.0, np.pi):
        start = F.rotation_start(data, setup)
        start[F.CAMERA_ROTATION] += flip
        theta = F.least_squares_fit(data, setup, geometry, start)["theta"]
        a, _ = F.responses(theta[F.LENS], a_dac, setup.v0, setup.v0)
        _, b = F.responses(theta[F.LENS], b_dac, setup.v0, setup.v0)
        theta[F.LOG_SHIFT_SCALE] += np.log(abs(a[0]) / magnification)
        theta[F.LOG_TILT_SCALE] += np.log(abs(b[0]) / b_abs)
        theta = F.least_squares_fit(data, setup, geometry, theta)["theta"]
        result = F.least_squares_fit(data, setup, list(free), theta, extra=specimen_residuals)
        if best is None or result["chi2"] < best["chi2"]:
            best = result
    return best


def position_errors(result):
    """Lens and detector positions (mm) with 1-sigma uncertainties from the fit covariance."""
    free = np.asarray(result["free"])
    theta = jnp.asarray(result["theta"])
    g = np.asarray(jax.jacfwd(lambda x: _positions(theta.at[free].set(x)))(theta[free]))
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
    parts = [F.load_session(folder=folder)[0] for folder in (first_ht_folder, second_ht_folder)]
    data = {k: np.concatenate([d[k] for d in parts]) for k in ("dac", "voltage", "kind", "command", "measured")}
    units, scales = np.asarray(model["units"]), model["deflector_scales"]
    data["dac"] = data["dac"] * units
    command = data["command"] * np.where(data["kind"] == 0, scales[0], scales[1])
    data["command"] = np.conj(command) if model["y_flipped"] else command
    data["measured"] = np.conj(data["measured"]) if model["mirrored"] else data["measured"]
    data["wobble_px"] = model["wobble_px"]
    diff_row = np.asarray(plan2["diffraction"]) * units
    setup = F.Setup(v0=model["ht_v"], base_dac=np.asarray(model["base_dac"]) * units,
                    camera=F.Camera(n=model["frame_width"], pitch=model["pitch_m"]),
                    wobble_px=model["wobble_px"], mix_unit=scales[1] / scales[0])

    measured = magnification_and_camera_length(grating_frame=grating_frame, gold_frame=gold_frame,
                                               period_m=period_m, pitch_m=model["pitch_m"], ht_v=model["ht_v"],
                                               dark=dark)
    free = TWO_HT_FREE + [F.LOG_SHIFT_SCALE, F.LOG_TILT_SCALE] + (F.MIX if model["deflectors"] == "mixed" else [])
    result = fit_with_specimens(data, setup, free=free, a_dac=setup.base_dac,
                                magnification=measured["magnification"], magnification_sigma=period_sigma,
                                b_dac=diff_row, b_abs=measured["camera_length_m"], b_sigma=ring_sigma)
    positions, sigma = position_errors(result)
    theta = result["theta"]
    return dict(names=POSITION_NAMES, positions_mm=positions, sigma_mm=sigma,
                shift_m_per_unit=scales[0] * float(np.exp(theta[F.LOG_SHIFT_SCALE])),
                tilt_rad_per_unit=scales[1] * float(np.exp(theta[F.LOG_TILT_SCALE])),
                magnification=measured["magnification"], camera_length_m=measured["camera_length_m"],
                chi2=result["chi2"], dof=result["dof"], accepted=bool(result["accepted"]),
                measurements=len(data["kind"]), theta=theta, units=units, setup=setup)
