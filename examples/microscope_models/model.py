"""Calibration of the projection system from beam shifts and tilts: the model and the fit.

The model is TemGym's projector (objective, IL1, IL2, IL3, PL1 as finite solenoids,
:mod:`temgym_core.calibration.projector_model`): for each lens setting its ray map gives A, how
far the beam moves on the camera per metre of shift at the specimen, and B, per radian of tilt.
Deflector commands enter through unknown scales (and optionally a mix of shift and tilt), and the
camera through an unknown rotation. The fit is weighted least squares to the measured
half-differences of +/- kicks, at the beam-position wobble.

* **Stage 1** (one HT, no specimen; ``acquire.Recording`` /
  ``analyse.fit_projection_response``) fixes the predictions; lens positions are not unique yet
  (the model is unscaled).
* **Stage 2** (``acquire.record_second_ht`` / ``analyse.fit_physical_projection``, draft) makes it
  physical. Two specimen
  frames: a cross-grating image at setting 0 gives |A| (:func:`grating_period_px`) and a gold ring
  pattern at the diffraction setting gives |B| (:func:`ring_radii_px`), so the kick scales are in
  metres and radians (:func:`specimen_scales`); and the beam moves at a second HT fix the lens and
  camera positions.

See ``projector_calibration_outline.md`` for the mathematics. Lens DAC values map to model controls
by a fixed unit convention (:func:`controls`); the fitted gains absorb any per-lens scale.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_dist

from temgym_core.calibration.condenser_model import magnetic_strength_ratio
from temgym_core.calibration.projector_model import SIGNS, at_voltage, initial_guess, matrix5

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).parent
_PRESETS = HERE.parents[1] / "src" / "temgym_core" / "calibration" / "presets" / "beam_measurement_protocol.json"

LENSES = ("IL1", "IL2", "IL3", "PL1")
POSITION_NAMES = ("Objective", "IL1", "IL2", "IL3", "PL1", "Detector")
DAC_MAX = 65535
AU_LATTICE = 0.40782e-9                                    # gold lattice constant (m)
AU_RINGS = {"111": AU_LATTICE / np.sqrt(3), "200": AU_LATTICE / 2, "220": AU_LATTICE / np.sqrt(8)}

# ---------------------------------------------------------------------------- conventions
#: The NEOARM 40k row at 200 kV and the 200 kV reference controls define the DAC -> control unit.
NEOARM_40K_200 = np.array([41782.0, 24862.0, 35544.0, 64000.0])
_REFERENCE = np.asarray(json.loads(_PRESETS.read_text())["projector_reference_controls"])
CONTROL_PER_DAC = _REFERENCE[1:] / NEOARM_40K_200

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
TWO_HT_FREE = list(range(17))


def strength_ratio(voltage, reference):
    """p(reference) / p(voltage): how much stronger a fixed magnetic field acts at ``voltage``."""
    return float(magnetic_strength_ratio(voltage, reference))


def controls(dac):
    """Model controls (objective first, fixed at 1) for IL1, IL2, IL3, PL1 DAC values."""
    return np.r_[1.0, np.asarray(dac, float) * CONTROL_PER_DAC]


def positions_mm(theta):
    """Centres of the objective, IL1, IL2, IL3, PL1 and the detector, from the specimen (mm)."""
    gaps, lengths = np.exp(np.asarray(theta[:6])), np.exp(np.asarray(theta[6:11]))
    entrances = np.cumsum(gaps[:-1]) + np.r_[0.0, np.cumsum(lengths[:-1])]
    return 1e3 * np.r_[entrances + lengths / 2, gaps.sum() + lengths.sum()]


def _positions_jnp(theta):
    gaps, lengths = jnp.exp(theta[:6]), jnp.exp(theta[6:11])
    entrances = jnp.cumsum(gaps[:-1]) + jnp.concatenate([jnp.zeros(1), jnp.cumsum(lengths[:-1])])
    return 1e3 * jnp.concatenate([entrances + lengths / 2, (gaps.sum() + lengths.sum())[None]])


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
    """What the experimenter knows: the HTs, the 40k row and (stage 2) the diffraction setting (DAC
    values read at the console), the camera, the beam-position wobble and the deflector unit ratio."""
    v0: float
    v1: float
    base_dac: np.ndarray                      # 40k-row DAC values at v0, read from the console
    diffraction_setting: np.ndarray = None      # stage 2: DAC values read in DIFF mode (A = 0)
    camera: Camera = field(default_factory=Camera)
    wobble_px: float = 0.01                     # beam-position wobble, from the stage-1 repeats
    mix_unit: float = 4e4                       # rad of tilt command per m of shift command, per deflector unit


# ---------------------------------------------------------------------------- the specimen frames
def _fft_lattice(image, min_cycles=1.5, pad=16):
    """Two perpendicular fundamental wave vectors (cycles per px) from the zero-padded FFT."""
    n = image.shape[0]
    window = np.outer(np.hanning(n), np.hanning(n))
    f = np.abs(np.fft.fftshift(np.fft.fft2((image - image.mean()) * window, s=(n * pad, n * pad))))
    k = (np.arange(n * pad) - n * pad // 2) / (n * pad)
    KX, KY = np.meshgrid(k, k)
    f[np.hypot(KX, KY) < min_cycles / n] = 0
    f[KY < 0] = 0                                                # half plane (Hermitian)
    vectors = []
    for _ in range(2):
        i, j = np.unravel_index(np.argmax(f), f.shape)
        vectors.append(np.array([k[j], k[i]]))
        angle = np.abs(np.angle(np.exp(1j * (np.arctan2(KY, KX) - np.arctan2(k[i], k[j])))))
        f[(angle < np.pi / 4) | (angle > 3 * np.pi / 4)] = 0     # keep only the perpendicular one
    return np.array(vectors)


def grating_period_px(image):
    """Lattice period (px) of a cross-grating image, from a fit of the whole image.

    Model of every pixel: ``exp(P(r)) * (b + (1 - b) * hole(s) * hole(t))``. Here ``(s, t)`` are
    lattice coordinates, ``r = r0 + s a1 + t a2``; ``hole(f)`` is 1 inside a square hole of side
    ``h`` periods and falls to 0 over a soft edge ``e``; ``b`` is the bar level; ``P`` is a quadratic
    polynomial, the beam profile. The profile is fitted, so an uneven beam does not bias the
    lattice, and squares cut by the frame edge still count. Start: lattice vectors from the FFT
    peaks, the origin from a scan over one cell. Returns the mean period, the two lattice vectors
    (px), and the fitted parameters with the fraction of the image the fit explains. Raises
    ``ValueError`` if that fraction is below 0.9 (too few squares lit).
    """
    from scipy.ndimage import gaussian_filter
    n = image.shape[0]
    x = np.arange(n) - (n - 1) / 2
    X, Y = np.meshgrid(x, x)
    data = np.asarray(image, float) / np.max(image)
    xs, ys, d = jnp.asarray(X.ravel()), jnp.asarray(Y.ravel()), jnp.asarray(data.ravel())

    def model(q):
        lattice = jnp.array([[q[0], q[2]], [q[1], q[3]]])            # columns a1, a2
        s, t = jnp.linalg.solve(lattice, jnp.stack([xs - q[4], ys - q[5]]))
        h, e, b = q[6], q[7], q[8]
        hole = (jax.nn.sigmoid((h / 2 - jnp.abs(s - jnp.round(s))) / e)
                * jax.nn.sigmoid((h / 2 - jnp.abs(t - jnp.round(t))) / e))
        u, v = xs / n, ys / n
        profile = jnp.exp(q[9] + q[10] * u + q[11] * v + q[12] * u * u + q[13] * u * v + q[14] * v * v)
        return profile * (b + (1 - b) * hole)

    residual = jax.jit(lambda q: model(q) - d)
    jacobian = jax.jit(jax.jacfwd(residual))
    # start: lattice vectors from the FFT, the beam profile from a heavy blur, levels from percentiles
    k = _fft_lattice(image)
    lattice = np.linalg.inv(k)                                     # columns: real-space lattice vectors
    blur = np.log(np.maximum(gaussian_filter(data, 0.15 * n), 1e-6))
    u, v = X.ravel() / n, Y.ravel() / n
    design = np.c_[np.ones_like(u), u, v, u * u, u * v, v * v]
    profile, *_ = np.linalg.lstsq(design, blur.ravel(), rcond=None)
    flat = data / np.exp((design @ profile).reshape(n, n))
    b0 = float(np.clip(np.percentile(flat, 10) / np.percentile(flat, 90), 0.02, 0.9))
    best = None
    for fx in np.arange(8) / 8:                                    # origin: scan one cell
        for fy in np.arange(8) / 8:
            r0 = lattice @ np.array([fx, fy])
            q = np.r_[lattice[:, 0], lattice[:, 1], r0, 0.6, 0.03, b0, profile]
            cost = float(np.sum(np.asarray(residual(jnp.asarray(q))) ** 2))
            if best is None or cost < best[0]:
                best = (cost, q)
    out = least_squares(lambda q: np.asarray(residual(jnp.asarray(q))), best[1],
                        jac=lambda q: np.asarray(jacobian(jnp.asarray(q))), method="lm", xtol=1e-12, ftol=1e-12)
    a1, a2 = out.x[0:2], out.x[2:4]
    explained = 1 - np.sum(out.fun ** 2) / np.sum((data - data.mean()) ** 2)
    if explained < 0.9:                       # a good fit explains >= 96 % of the image
        raise ValueError(f"the lattice fit explains only {explained:.0%} of the image: too few squares lit, "
                         "or not a cross grating. Spread the beam over the frame and try again.")
    return float((np.hypot(*a1) + np.hypot(*a2)) / 2), (a1, a2), dict(parameters=out.x, explained=float(explained))


def ring_radii_px(image, camera, rings=("111", "200"), half_width_px=5.0, direct_mask_px=8.0):
    """Ring radii (px) of a gold pattern.

    The centre is the centroid of the direct beam. The radial intensity profile (direct beam
    masked) gives a first guess of the 111 radius; the other rings are placed by their known
    d-spacing ratios. Each radius is then the intensity-weighted mean radius within
    +/- ``half_width_px`` of its guess (biased outward by about s^2 / 2R for spots of rms s:
    0.03 % for 2 px spots at 80 px). Returns the radii and the profile.
    """
    x = camera.coordinates
    X, Y = np.meshgrid(x, x)
    j = np.unravel_index(np.argmax(image), image.shape)
    near = np.hypot(X - x[j[1]], Y - x[j[0]]) < direct_mask_px
    centre = complex((image * X)[near].sum() / image[near].sum(), (image * Y)[near].sum() / image[near].sum())
    r = np.hypot(X - centre.real, Y - centre.imag)
    rest = np.where(r > direct_mask_px, image, 0.0)
    bins = np.arange(0, camera.n / 2, 0.5)
    profile = np.bincount(np.digitize(r.ravel(), bins), weights=rest.ravel(), minlength=len(bins) + 1)[1:len(bins) + 1]
    profile = profile / np.maximum(np.bincount(np.digitize(r.ravel(), bins), minlength=len(bins) + 1)[1:len(bins) + 1], 1)
    ratio = {k: AU_RINGS["111"] / AU_RINGS[k] for k in rings}
    # the 111 guess that puts the most intensity on all the rings
    candidates = bins[(bins > direct_mask_px + half_width_px)]
    score = [sum(np.interp(g * ratio[k], bins, profile) for k in rings) for g in candidates]
    guess = candidates[int(np.argmax(score))]
    radii = {}
    for k in rings:
        window = np.abs(r - guess * ratio[k]) < half_width_px
        radii[k] = float((rest * r)[window].sum() / rest[window].sum())
    return radii, dict(centre=centre, bins=bins, profile=profile)


def specimen_scales(*, a_dac, magnification, magnification_sigma, b_dac, b_abs, b_sigma):
    """The two specimen measurements: |A| at ``a_dac`` and |B| (m/rad) at ``b_dac`` (HT v0), with
    relative uncertainties."""
    return dict(a_dac=np.asarray(a_dac), magnification=magnification, magnification_sigma=magnification_sigma,
                b_dac=np.asarray(b_dac), b_abs=b_abs, b_sigma=b_sigma)


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


def _least_squares(data, setup, free, specimen, start):
    free = np.asarray(free)
    sigma = data["wobble_px"] / np.sqrt(2)
    meas = jnp.asarray(data["measured"])
    if specimen is not None:
        specimen_dac = np.vstack([specimen["a_dac"], specimen["b_dac"]])
        specimen_us = jnp.asarray(np.array([controls(w) for w in specimen_dac]))

    def residual(x):
        theta = jnp.asarray(start).at[free].set(x)
        r = (predict(theta, data, setup) - meas) / sigma
        r = jnp.concatenate([jnp.real(r), jnp.imag(r)])
        if specimen is not None:
            a, b = _responses(theta[:16], specimen_us, jnp.asarray([setup.v0, setup.v0]), jnp.asarray(setup.v0))
            r = jnp.concatenate([r, jnp.array([
                (jnp.abs(a[0]) / specimen["magnification"] - 1) / specimen["magnification_sigma"],
                (jnp.abs(b[1]) / specimen["b_abs"] - 1) / specimen["b_sigma"]])])
        return r

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


def fit(data, setup, *, free, specimen=None, start=None):
    """Weighted least squares of the half-differences (and the two specimen measurements), from a data-driven start.

    Without a ``start``: the gains (and with two HTs the objective q) come from the closed-form
    rotation regression and the camera rotation from the strongest shift. That rotation is known
    only modulo pi (the start model may invert the image), so both are fitted and the better is
    kept. With ``specimen`` the kick scales are first read off the specimen measurements (continuation), then
    everything is freed. Returns theta, chi2, dof, the covariance of the free parameters and the
    acceptance at the 99 % point of chi2.
    """
    free = list(free)
    if start is not None:
        return _least_squares(data, setup, free, specimen, np.asarray(start, float))
    geometry = [f for f in free if f not in (LOG_KS, LOG_KT)]
    best = None
    for flip in (0.0, np.pi):
        start = _rotation_start(data, setup)
        start[PHI] += flip
        result = _least_squares(data, setup, geometry, None, start)
        if specimen is not None:
            theta = result["theta"]
            a, _ = responses(theta[:16], specimen["a_dac"], setup.v0, setup.v0)
            _, b = responses(theta[:16], specimen["b_dac"], setup.v0, setup.v0)
            theta[LOG_KS] += np.log(abs(a[0]) / specimen["magnification"])
            theta[LOG_KT] += np.log(abs(b[0]) / specimen["b_abs"])
            theta = _least_squares(data, setup, geometry, None, theta)["theta"]
            result = _least_squares(data, setup, free, specimen, theta)
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
