"""Calibration of the projection system from beam shifts and tilts: the model and the fit.

The model is TemGym's projector (objective, IL1, IL2, IL3, PL1 as finite solenoids,
:mod:`temgym_core.tem_column.projector`): for each lens setting its ray map gives A, how
far the beam moves on the camera per metre of shift at the specimen, and B, per radian of tilt.
Deflector commands enter through unknown scales (and optionally a mix of shift and tilt), and the
camera through an unknown rotation. The fit is weighted least squares to the measured
half-differences of +/- kicks, at the beam-position wobble.

* **Stage 1** (one HT, no specimen; ``acquire.Recording`` /
  ``analyse.fit_projection_response``) fixes the predictions; lens positions are not unique yet
  (the model is unscaled).
* **Stage 2** (draft, in ``stage2/``) makes it physical: two specimen frames put the kick scales
  in metres and radians, and a second HT fixes the lens and camera positions.

See ``projector_calibration_outline.md`` for the mathematics. Lens DAC values map to model controls
by a fixed unit convention (:func:`controls`); the fitted gains absorb any per-lens scale.
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_dist

from temgym_core.tem_column.high_tension import magnetic_strength_ratio
from temgym_core.tem_column.projector import REFERENCE_40K_CONTROLS, SIGNS, at_voltage, initial_guess, matrix5

jax.config.update("jax_enable_x64", True)

LENSES = ("IL1", "IL2", "IL3", "PL1")
DAC_MAX = 65535

# ---------------------------------------------------------------------------- conventions
#: The DAC -> control unit: the NEOARM 40k row at 200 kV maps onto the 40k-row strengths the
#: engineering start was made for (a choice of unit only; the fitted gains absorb any scale).
NEOARM_40K_200 = np.array([41782.0, 24862.0, 35544.0, 64000.0])
CONTROL_PER_DAC = REFERENCE_40K_CONTROLS[1:] / NEOARM_40K_200

# Parameter vector theta (19): 16 projector coordinates (6 log gaps, 5 log lengths, objective q,
# 4 log gains), camera rotation, log shift scale, log tilt scale.
PHI, LOG_KS, LOG_KT = 16, 17, 18
#: Deflector mixing (complex, as real and imaginary parts). Per deflector unit, the shift
#: deflector (CLA1) also makes e1 times the tilt the tilt deflector (CLA2) makes, and CLA2 also
#: makes e2 times the shift CLA1 makes. ``Setup.mix_unit`` converts between the two commands'
#: nominal units (rad of tilt command per m of shift command for the same deflector units).
#: Index 23 is the angle (rad) between the tilt deflector's x axis and the shift deflector's.
#: All zero for pure, aligned deflectors; fitted when that is not known.
MIX = [19, 20, 21, 22, 23]
N_THETA = 24
ONE_HT_FREE = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]   # one HT: detector gap, phi are gauges


def strength_ratio(voltage, reference):
    """p(reference) / p(voltage): how much stronger a fixed magnetic field acts at ``voltage``."""
    return float(magnetic_strength_ratio(voltage, reference))


def controls(dac):
    """Model controls (objective first, fixed at 1) for IL1, IL2, IL3, PL1 DAC values."""
    return np.r_[1.0, np.asarray(dac, float) * CONTROL_PER_DAC]


# ---------------------------------------------------------------------------- the lens model
def _response(p, u, voltage, reference):
    m = matrix5(at_voltage(p, voltage, reference), u)
    return m[0, 0] + 1j * m[1, 0], m[0, 2] + 1j * m[1, 2]


_responses = jax.jit(jax.vmap(_response, in_axes=(None, 0, 0, None)))


def responses(p, dac, voltages, reference):
    """A (detector m per specimen m) and B (detector m per rad), complex, for each setting.

    ``p`` are the 16 projector coordinates defined at HT ``reference``. Pads batches of one
    (a batch of one hits a JAX CPU bug).
    """
    dac = np.atleast_2d(dac)
    voltages = np.broadcast_to(np.atleast_1d(np.asarray(voltages, float)), (len(dac),))
    us = np.array([controls(w) for w in dac])
    if len(us) == 1:
        a, b = _responses(jnp.asarray(p), jnp.asarray(np.repeat(us, 2, 0)), jnp.asarray(np.repeat(voltages, 2)),
                          jnp.asarray(float(reference)))
        return np.asarray(a[:1]), np.asarray(b[:1])
    a, b = _responses(jnp.asarray(p), jnp.asarray(us), jnp.asarray(voltages), jnp.asarray(float(reference)))
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
def predict(theta, data, setup):
    """Predicted half-differences (complex px) for every row of ``data`` under ``theta``."""
    us = jnp.asarray(np.array([controls(w) for w in data["dac"]]))
    a, b = _responses(theta[:16], us, jnp.asarray(data["voltage"]), jnp.asarray(setup.v0))
    ratio = magnetic_strength_ratio(jnp.asarray(data["voltage"]), setup.v0)
    kind = jnp.asarray(data["kind"])
    ks, kt = jnp.exp(theta[LOG_KS]), jnp.exp(theta[LOG_KT])
    if theta.shape[0] > MIX[0]:                                   # each deflector a mix of shift and tilt
        e1, e2 = theta[MIX[0]] + 1j * theta[MIX[1]], theta[MIX[2]] + 1j * theta[MIX[3]]
        r = setup.mix_unit
        response = jnp.where(kind == 0, a * ks + e1 * b * kt * r,
                             jnp.exp(1j * theta[MIX[4]]) * (b * kt + e2 * a * ks / r))
    else:
        response = jnp.where(kind == 0, a * ks, b * kt)
    return jnp.exp(1j * theta[PHI]) * response * jnp.asarray(data["command"]) * ratio / setup.camera.pitch


def _engineering_start(setup):
    """Engineering guess carried to v0 (objective re-excited), with neutral camera and kicks."""
    p = np.array(at_voltage(jnp.asarray(initial_guess()), setup.v0, 200e3))
    p[11] = initial_guess()[11]
    return np.r_[p, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _rotation_start(data, setup):
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
    theta = _engineering_start(setup)
    theta[12:16] = np.log(np.clip(-coef[-4:] / np.asarray(SIGNS), 1e-3, None))
    if two_ht:
        theta[11] = -coef[1]
    ref_row = np.argmax(np.where(data["kind"] == 0, np.abs(data["measured"]), 0))
    one = {k: data[k][[ref_row, ref_row]] for k in ("dac", "voltage", "kind", "command")}
    theta[PHI] = float(np.angle(data["measured"][ref_row] / complex(predict(jnp.asarray(theta), one, setup)[0])))
    return theta


def _least_squares(data, setup, free, start, extra=None):
    """Least squares of the half-differences over the ``free`` parameters from ``start``.
    ``extra(theta)``, if given, adds residuals (already divided by their uncertainties)."""
    free = np.asarray(free)
    sigma = data["wobble_px"] / np.sqrt(2)
    meas = jnp.asarray(data["measured"])

    def residual(x):
        theta = jnp.asarray(start).at[free].set(x)
        r = (predict(theta, data, setup) - meas) / sigma
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


def fit(data, setup, *, free, start=None):
    """Weighted least squares of the half-differences, from a data-driven start.

    Without a ``start``: the gains (and with two HTs the objective q) come from the closed-form
    rotation regression and the camera rotation from the strongest shift. That rotation is known
    only modulo pi (the start model may invert the image), so both are fitted and the better is
    kept. Returns theta, chi2, dof, the covariance of the free parameters and the acceptance at
    the 99 % point of chi2.
    """
    free = list(free)
    if start is not None:
        return _least_squares(data, setup, free, np.asarray(start, float))
    best = None
    for flip in (0.0, np.pi):
        start = _rotation_start(data, setup)
        start[PHI] += flip
        result = _least_squares(data, setup, free, start)
        if best is None or result["chi2"] < best["chi2"]:
            best = result
    return best


