"""Projector calibration from beam centroids: the experiment, a simulated microscope, and the fit.

The experiment (see ``projector_calibration_outline.md``):

* **stage 1** -- no specimen, HT ``v0``: at 9 lens settings, a +/- beam shift and a +/- beam tilt.
  Setting 0 is measured three times (start, middle, end), which gives the beam-position wobble, and
  once more with y kicks, which checks that the deflectors and lenses are round. Fit: a model that
  turns lens DAC values and deflector commands into beam positions (kick units).
* **anchors** -- a cross-grating image at setting 0 (gives |A|) and a gold ring pattern at the
  diffraction setting (gives |B|). Fit: everything in metres and radians.
* **stage 2** -- one block, specimen out: at HT ``v1`` the 9 stage-1 settings and 30 settings from a
  fixed list, then the 30 again at ``v0``. Every setting is known before the session starts.
  Fit: absolute lens and camera positions.

The module keeps two worlds apart:

* :class:`SimulatedMicroscope` holds the hidden column (lens geometry, kick scales, camera rotation,
  beam offsets, beam-position wobble). Its only experimental output is camera frames.
* Everything else -- :class:`Setup`, the acquisition, the measurements and the fit -- sees only
  lens DAC values, HTs, deflector commands and frames, as on the real microscope.

Simulation conventions: "infinite dose" (no shot noise, no saturation); the beam-position wobble
(drift, lens return, ...) is a random beam displacement of ``wobble_px`` per frame and component;
paraxial optics; linear deflectors. Lens DAC values map to model controls by a fixed unit convention
(:func:`controls`); the fitted gains absorb any per-lens scale.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import c, e, h, m_e
from scipy.optimize import brentq, least_squares
from scipy.special import erf
from scipy.stats import chi2 as chi2_dist

from temgym_core.calibration.condenser_model import magnetic_strength_ratio
from temgym_core.calibration.projector_model import SIGNS, at_voltage, initial_guess, matrix5

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).parent
_TRUTH_FILE = HERE / "calibration_results" / "tilt_y_30_ray_acquisition" / "scoring_truth.json"
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

#: The 8 stage-1 settings as multiples of the 40k-row DAC values (IL1, IL2, IL3, PL1). Set 0 is the row.
SET_PATTERN = np.array([
    [41782, 24862, 35544, 64000],
    [25000, 24862, 35544, 64000],
    [41782, 24862, 25000, 64000],
    [41782, 24862, 35544, 38400],
    [41782, 24862, 36000, 32000],
    [41782, 40000, 35544, 38400],
    [41782, 40000, 36000, 64000],
    [41782, 40000, 35544, 32000],
]) / NEOARM_40K_200

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


def wavelength(voltage):
    """Relativistic electron wavelength (m)."""
    p = np.sqrt(2 * m_e * e * voltage * (1 + e * voltage / (2 * m_e * c ** 2)))
    return h / p


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


# ---------------------------------------------------------------------------- the camera
@dataclass(frozen=True)
class Camera:
    """Camera at the binning used: ``n`` x ``n`` pixels of ``pitch`` metres, Gaussian PSF (px)."""
    n: int = 256
    pitch: float = 110e-6
    psf_px: float = 1.0

    @property
    def coordinates(self):
        return np.arange(self.n) - (self.n - 1) / 2


def render_spot(camera, centre_px, rms_px):
    """Pixel-integrated round Gaussian spot of unit flux (plus the PSF), centre as complex px."""
    s = np.sqrt(rms_px ** 2 + camera.psf_px ** 2) * np.sqrt(2)
    edges = np.arange(camera.n + 1) - camera.n / 2
    gx = np.diff(0.5 * erf((edges - centre_px.real) / s))
    gy = np.diff(0.5 * erf((edges - centre_px.imag) / s))
    return np.outer(gy, gx)


def centroid(frame, camera):
    """Intensity-weighted centre of a frame, as complex px from the frame centre."""
    x = camera.coordinates
    total = frame.sum()
    return complex((frame.sum(0) @ x) / total, (frame.sum(1) @ x) / total)


def spot_rms(frame, camera):
    """Rms radius per axis of a frame (px). Used only to keep the beam on the camera."""
    x = camera.coordinates
    total = frame.sum()
    c0 = centroid(frame, camera)
    vx = (frame.sum(0) @ (x - c0.real) ** 2) / total
    vy = (frame.sum(1) @ (x - c0.imag) ** 2) / total
    return float(np.sqrt((vx + vy) / 2))


# ---------------------------------------------------------------------------- the simulated microscope
@dataclass
class SimulatedMicroscope:
    """The hidden truth. The experiment may only call :meth:`frame`, :meth:`grating_image` and
    :meth:`ring_pattern`, which return camera frames; :meth:`truth` is for scoring only.

    The column is the synthetic projector of ``calibration_results/tilt_y_30_ray_acquisition``,
    carried to HT ``v0`` the way the console would do it: the objective re-excited to the same
    strength, and the free-lens DAC values scaled by p(v0)/p(200 kV) (see :meth:`base_dac`).
    """
    v0: float = 60e3
    kick_errors: tuple = (0.07, -0.06)          # true kick scales are 1 + error (nominal units)
    mixing: tuple = (0j, 0j)                    # see MIX: shift deflector's tilt, tilt deflector's shift
    mix_unit: float = 4e4                       # rad of tilt command per m of shift command, per deflector unit
    tilt_axis_angle: float = 0.0                # angle of the tilt deflector's axes against the shift deflector's
    camera_rotation: float = 0.6                # rad
    wobble_px: float = 0.01                      # beam-position wobble per frame and component (px)
    beam_offset_rms_px: float = 5.0             # where the unkicked beam lands, per setting
    illumination: tuple = (30e-9, 15e-6)        # condensed beam at the specimen: rms size (m), rms angle (rad)
    camera: Camera = field(default_factory=Camera)
    seed: int = 0

    def __post_init__(self):
        p200 = np.asarray(json.loads(_TRUTH_FILE.read_text())["projector_parameters"])
        p = np.array(at_voltage(jnp.asarray(p200), self.v0, 200e3))
        p[11] = p200[11]                        # objective re-excited for the same optics
        self._p = p
        self._rng = np.random.default_rng(self.seed)
        self._offsets = {}

    # -- console facts (what the operator reads off the microscope) -------------------------
    def base_dac(self):
        """The 40k-row free-lens DAC values at ``v0``, as GetFLCInfo would report them."""
        return np.round(NEOARM_40K_200 / strength_ratio(self.v0, 200e3), 1)

    def diff_mode_setting(self, ring_px=80.0, ring="111"):
        """The free-lens DAC values the console uses in DIFF mode, at the camera length that puts
        the gold ``ring`` at about ``ring_px``: a lens setting with A = 0.

        On the microscope these are read with PyJEM after switching to DIFF mode. Here the console's
        DIFF table is played by the hidden column: IL2 (camera length) and IL3 are chosen for A = 0.
        """
        base = self.base_dac()
        lam = wavelength(self.v0)

        def signed_a(il2, il3):
            """Magnification with the lens rotation removed: changes sign where A passes 0."""
            il3 = np.atleast_1d(il3)
            dac = np.c_[np.full(len(il3), base[0]), np.full(len(il3), il2), il3, np.full(len(il3), base[3])]
            a, _ = responses(self._p, dac, np.full(len(il3), self.v0), self.v0)
            u = np.array([controls(w) for w in dac])
            angle = self._p[11] * u[:, 0] + u[:, 1:] @ (np.asarray(SIGNS) * np.exp(self._p[12:16]))
            return np.real(a * np.exp(1j * angle))

        best = None
        for il2 in np.linspace(0.6, 1.2, 31) * base[1] * 1.68:       # camera lengths around the gold rings
            grid = np.linspace(0.1, 2.0, 200) * base[2]
            values = signed_a(il2, grid)
            for i in np.flatnonzero(np.sign(values[:-1]) != np.sign(values[1:])):
                il3 = brentq(lambda w: float(signed_a(il2, w)[0]), grid[i], grid[i + 1])
                dac = np.r_[base[0], il2, il3, base[3]]
                a, b = responses(self._p, dac, self.v0, self.v0)
                if abs(a[0]) > 1.0:
                    continue                                      # a rotation wrap, not A = 0
                radius = abs(b[0]) * lam / AU_RINGS[ring] / self.camera.pitch
                if best is None or abs(radius - ring_px) < abs(best[1] - ring_px):
                    best = (np.round(dac, 1), radius)
        return best[0]

    # -- frames ------------------------------------------------------------------------------
    def _beam(self, dac, voltage, shift, tilt):
        a, b = responses(self._p, dac, voltage, self.v0)
        a, b = complex(a[0]), complex(b[0])
        ratio = strength_ratio(voltage, self.v0)                  # deflectors act more strongly at lower HT
        ks, kt = 1 + self.kick_errors[0], 1 + self.kick_errors[1]
        key = (tuple(np.round(dac, 3)), float(voltage))
        if key not in self._offsets:
            self._offsets[key] = complex(*self._rng.normal(0, self.beam_offset_rms_px, 2))
        jitter = complex(*self._rng.normal(0, self.wobble_px, 2))
        e1, e2 = self.mixing
        r = self.mix_unit
        pos = np.exp(1j * self.camera_rotation) * ratio * (
            shift * (a * ks + e1 * b * kt * r)
            + tilt * np.exp(1j * self.tilt_axis_angle) * (b * kt + e2 * a * ks / r)) / self.camera.pitch
        sx, st = self.illumination
        rms = np.hypot(abs(a) * sx, abs(b) * st) / self.camera.pitch
        return self._offsets[key] + pos + jitter, rms

    def frame(self, dac, voltage, shift=0.0, tilt=0.0):
        """A camera frame (unit flux) with the beam shifted / tilted by the deflector commands
        ``shift`` and ``tilt`` (nominal units, complex: x + iy)."""
        centre, rms = self._beam(np.asarray(dac, float), voltage, shift, tilt)
        return render_spot(self.camera, centre, rms)

    def _coherent_image(self, dac, orders, beam_rms, subsamples=2):
        """Coherent image at ``v0`` of a specimen given as diffraction ``orders`` (list of
        (slope_x, slope_y, amplitude)) under a parallel beam of ``beam_rms``. Each order is a
        Gaussian beam through the true ray matrix (TemGym's closed-form Collins transform); the
        orders add in amplitude. The field is sampled ``subsamples`` times per pixel per axis."""
        from temgym_core import source
        from temgym_core.wave.reference import collins_gaussian, gaussian_field
        lam = wavelength(self.v0)
        matrix = np.asarray(matrix5(jnp.asarray(self._p), jnp.asarray(controls(dac))))
        n = self.camera.n * subsamples
        x = (np.arange(n) - (n - 1) / 2) * self.camera.pitch / subsamples
        rot = np.exp(-1j * self.camera_rotation)                  # camera frame -> column frame
        X, Y = np.meshgrid(x, x)
        w = (X + 1j * Y) * rot
        pos = np.stack([w.real, w.imag], -1)
        q0 = np.asarray(source.gaussian_waist(z=0.0, voltage=self.v0, rms_size=beam_rms).Q_inv)
        field_sum = 0
        for sx, sy, amp in orders:
            o = collins_gaussian(matrix, lam, amp, np.zeros(2), np.array([sx, sy]), q0)
            field_sum = field_sum + np.asarray(gaussian_field(pos, o[0], o[1], o[2], o[3], lam, o[4]))
        image = np.abs(field_sum) ** 2
        image = image.reshape(self.camera.n, subsamples, self.camera.n, subsamples).sum((1, 3))
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(image, self.camera.psf_px)

    def grating_image(self, dac, period, *, square_fraction=0.6, contrast=0.5, beam_rms=5e-6, max_order=5):
        """Cross grating: square holes (``square_fraction`` of the period) in a film of amplitude
        ``1 - contrast``; image at lens DAC values ``dac`` (HT ``v0``) with a parallel beam of ``beam_rms``."""
        lam = wavelength(self.v0)
        n = np.arange(-max_order, max_order + 1)
        cn = square_fraction * np.sinc(n * square_fraction)      # Fourier series of the square wave
        orders = []
        for i, nx in enumerate(n):
            for j, ny in enumerate(n):
                amp = contrast * cn[i] * cn[j] + (1 - contrast) * (nx == 0 and ny == 0)
                if abs(amp) > 1e-6:
                    orders.append((lam * nx / period, lam * ny / period, amp))
        return self._coherent_image(dac, orders, beam_rms)

    def ring_pattern(self, dac, *, grains=600, grain_rms=4e-9, beam_rms=1e-6, grain_amplitude=0.15, seed=1):
        """Polycrystalline gold at lens DAC values ``dac`` (HT ``v0``): the direct beam plus one reflection
        from each of ``grains`` randomly oriented grains (111, 200, 220 by multiplicity).

        Every beam is a Gaussian through the true ray matrix (Collins transform); they add in
        intensity, and each is integrated over the pixels analytically because at the diffraction setting the spots
        are smaller than a pixel.
        """
        from temgym_core import source
        from temgym_core.wave.reference import collins_gaussian
        rng = np.random.default_rng(seed)
        lam = wavelength(self.v0)
        k = 2 * np.pi / lam
        matrix = np.asarray(matrix5(jnp.asarray(self._p), jnp.asarray(controls(dac))))
        names = list(AU_RINGS)
        weights = np.array([8, 6, 12]) * np.array([1.0, 0.8, 0.5])
        beams = [(0.0, 1.0, beam_rms)]
        for kk in rng.choice(3, grains, p=weights / weights.sum()):
            g = lam / AU_RINGS[names[kk]] * np.exp(1j * rng.uniform(0, 2 * np.pi))
            beams.append((g, grain_amplitude, grain_rms))
        image = 0
        for g, amp, waist in beams:
            q0 = np.asarray(source.gaussian_waist(z=0.0, voltage=self.v0, rms_size=waist).Q_inv)
            o = collins_gaussian(matrix, lam, amp, np.zeros(2), np.array([g.real, g.imag]), q0)
            centre = complex(*np.asarray(o[1])) * np.exp(1j * self.camera_rotation) / self.camera.pitch
            rms = np.sqrt(1 / (2 * k * np.asarray(o[3]).imag[0, 0])) / self.camera.pitch
            image = image + abs(amp) ** 2 * render_spot(self.camera, centre, rms)
        return image

    # -- scoring only ------------------------------------------------------------------------
    def truth(self):
        """The hidden column (scoring only): projector coordinates at ``v0``, camera rotation,
        log kick scales and deflector mixing, as a 24-vector comparable with a fitted theta."""
        e1, e2 = self.mixing
        return np.r_[self._p, self.camera_rotation, np.log1p(self.kick_errors[0]), np.log1p(self.kick_errors[1]),
                     e1.real, e1.imag, e2.real, e2.imag, self.tilt_axis_angle]


# ---------------------------------------------------------------------------- the experiment
@dataclass
class Setup:
    """What the experimenter knows and chooses."""
    v0: float
    v1: float
    base_dac: np.ndarray                      # 40k-row DAC values at v0, read from the console
    diffraction_setting: np.ndarray             # DAC values read in DIFF mode (A = 0)
    camera: Camera = field(default_factory=Camera)
    wobble_px: float = 0.01                     # beam-position wobble, from the stage-1 repeats
    target_px: float = 50.0                     # auto-scaled kicks aim for this displacement
    test_kick: tuple = (5e-9, 0.2e-3)           # test shift (m) and tilt (rad), nominal units
    kick_caps: tuple = (1e-6, 10e-3)            # deflector range (nominal units)
    edge_margin_px: float = 3.0
    n_stage2: int = 30                          # settings in the fixed stage-2 list
    mix_unit: float = 4e4                       # rad of tilt command per m of shift command, per deflector unit

    @property
    def sets(self):
        """The 9 stage-1 settings: 8 relative to the 40k row, then the diffraction setting."""
        return np.vstack([np.round(SET_PATTERN * self.base_dac, 1), self.diffraction_setting])

    @property
    def stage1_plan(self):
        """Stage-1 order: (setting index, kick direction). Setting 0 at the start, middle and end
        (the repeats give the wobble and include lens return), and once with y kicks."""
        order = [0, 1, 2, 3, 4, 0, 5, 6, 7, 8, 0, 0]
        directions = [1.0] * 11 + [1j]
        return order, directions

    @property
    def stage2_range(self):
        """DAC range of the stage-2 settings: 0.1-2x the 40k-row values, within the DAC."""
        return 0.1 * self.base_dac, np.minimum(2 * self.base_dac, DAC_MAX)

    @property
    def stage2_settings(self):
        """The fixed stage-2 list: ``n_stage2`` settings spread evenly over :attr:`stage2_range`
        (a Halton sequence), known before the session. Settings whose beam leaves the camera are
        skipped live."""
        from scipy.stats import qmc
        lo, hi = self.stage2_range
        return np.round(lo + (hi - lo) * qmc.Halton(d=4, scramble=False).random(self.n_stage2 + 1)[1:], 1)

    def room_px(self, centre, rms):
        """Free distance (px) from the beam's 4.5-rms edge to the frame edge."""
        return self.camera.n / 2 - self.edge_margin_px - abs(centre) - 4.5 * rms


def measure_setting(scope, setup, dac, voltage, direction=1.0, keep_frames=False):
    """The per-setting protocol: base frame, a test shift and tilt, then auto-scaled +/- pairs.

    Kicks are along ``direction`` (1 for x, 1j for y). The base frame's spot size is used only to
    keep the beam on the camera. Each kick aims at ``setup.target_px`` within the room left on the
    camera; a pair that clips is halved and retaken. Returns rows ``(kind, command,
    half_difference_px)`` (kind 0 shift, 1 tilt; the command is complex, carrying the direction),
    or an empty list if the beam does not fit, and optionally the frames.
    """
    base = scope.frame(dac, voltage)
    c0, rms = centroid(base, setup.camera), spot_rms(base, setup.camera)
    room = setup.room_px(c0, rms)
    frames = {"base": base}
    if room < 5:
        return ([], frames) if keep_frames else []
    rows = []
    for kind, test in enumerate(setup.test_kick):
        name = ("shift", "tilt")[kind]
        per_unit = abs(centroid(scope.frame(dac, voltage, **{name: test * direction}), setup.camera) - c0) / test
        size = min(setup.kick_caps[kind], min(setup.target_px, room) / max(per_unit, 1e-30))
        for _ in range(10):
            command = size * direction
            plus = scope.frame(dac, voltage, **{name: command})
            minus = scope.frame(dac, voltage, **{name: -command})
            if min(plus.sum(), minus.sum()) > (1 - 1e-6) * base.sum():
                break
            size /= 2                                             # clipped: halve and retake
        half = (centroid(plus, setup.camera) - centroid(minus, setup.camera)) / 2
        rows.append((kind, command, half))
        frames[name] = (minus, plus)
    return (rows, frames) if keep_frames else rows


def acquire(scope, setup, settings, voltages, directions=None, frames_for=()):
    """Run :func:`measure_setting` at every setting (in order) and HT. Returns the data (one row
    per setting, HT and kick type) and the frames of the settings whose index is in ``frames_for``."""
    settings = np.atleast_2d(settings)
    directions = [1.0] * len(settings) if directions is None else directions
    rows, kept = [], {}
    for v in voltages:
        for i, (dac, direction) in enumerate(zip(settings, directions)):
            keep = i in frames_for
            out = measure_setting(scope, setup, dac, v, direction=direction, keep_frames=keep)
            if keep:
                out, kept[(i, v)] = out
            for kind, command, half in out:
                rows.append((dac, v, kind, command, half))
    w, v, k, c_, m = zip(*rows)
    data = dict(dac=np.array(w), voltage=np.array(v, float), kind=np.array(k), command=np.array(c_),
                measured=np.array(m), wobble_px=setup.wobble_px)
    return data, kept


def acquire_stage1(scope, setup, frames_for=()):
    """Stage 1 at ``v0`` in the order of :attr:`Setup.stage1_plan`. ``frames_for`` indexes that order."""
    order, directions = setup.stage1_plan
    return acquire(scope, setup, setup.sets[order], (setup.v0,), directions=directions, frames_for=frames_for)


def acquire_stage2(scope, setup):
    """Stage 2 as one block: at ``v1`` the 9 stage-1 settings and the fixed list, then the fixed
    list again at ``v0``."""
    at_v1, _ = acquire(scope, setup, np.vstack([setup.sets, setup.stage2_settings]), (setup.v1,))
    at_v0, _ = acquire(scope, setup, setup.stage2_settings, (setup.v0,))
    return merge(at_v1, at_v0)


def wobble_from_repeats(data, setup):
    """Beam-position wobble (px per frame and axis) from the repeated x kicks at setting 0: the
    scatter of the half-differences about their mean, pooled over shift/tilt and x/y, times sqrt 2
    (a half-difference of two frames has 1/sqrt 2 of the per-frame wobble). Returns the estimate
    and its degrees of freedom."""
    at0 = (data["dac"] == setup.sets[0]).all(1) & (np.abs(np.imag(data["command"])) == 0) & (data["voltage"] == setup.v0)
    squares, dof = 0.0, 0
    for kind in (0, 1):                     # kick sizes differ a little per visit: compare per deflector unit
        sel = at0 & (data["kind"] == kind)
        m, c = data["measured"][sel], data["command"][sel]
        squares += np.sum(np.abs(m - c * np.mean(m / c)) ** 2)
        dof += 2 * (len(m) - 1)
    return float(np.sqrt(squares / dof) * np.sqrt(2)), dof


def kick_symmetry(data, setup):
    """For round lenses and deflectors a y kick moves the beam exactly as an x kick turned by 90 deg.
    Returns, for shift and tilt at setting 0, the measured (y response) / (i x response) - 1:
    zero when round, its size and angle the departure."""
    at0 = (data["dac"] == setup.sets[0]).all(1) & (data["voltage"] == setup.v0)
    out = {}
    for kind, name in ((0, "shift"), (1, "tilt")):
        sel = at0 & (data["kind"] == kind)
        per_unit = data["measured"][sel] / data["command"][sel]
        x = per_unit[np.imag(data["command"][sel]) == 0].mean()
        y = per_unit[np.imag(data["command"][sel]) != 0].mean()
        out[name] = complex(y / x - 1)
    return out


def merge(*datasets):
    """Concatenate data taken with the same wobble."""
    out = {k: np.concatenate([d[k] for d in datasets]) for k in ("dac", "voltage", "kind", "command", "measured")}
    out["wobble_px"] = datasets[0]["wobble_px"]
    return out


# ---------------------------------------------------------------------------- anchor measurements
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
    """Lattice period (px) of a cross-grating image from the centres of the squares.

    The squares (bright holes) are segmented at the mid level, those touching the frame edge are
    dropped, and each centre is the centroid of the intensity above the mid level. A lattice
    ``origin + i a1 + j a2`` (start from the FFT peaks) is fitted to the centres by least squares.
    Returns the mean period, the two lattice vectors and the centres.
    """
    from scipy.ndimage import label
    n = image.shape[0]
    x = np.arange(n) - (n - 1) / 2
    X, Y = np.meshgrid(x, x)
    level = 0.5 * (np.percentile(image, 10) + np.percentile(image, 90))
    labels, count = label(image > level)
    centres = []
    for k in range(1, count + 1):
        mask = labels == k
        rows, cols = np.nonzero(mask)
        if rows.min() == 0 or cols.min() == 0 or rows.max() == n - 1 or cols.max() == n - 1:
            continue                                              # cut by the frame edge
        w = (image - level)[mask]
        centres.append(complex((w * X[mask]).sum() / w.sum(), (w * Y[mask]).sum() / w.sum()))
    centres = np.array(centres)
    k = _fft_lattice(image)
    lattice = np.linalg.inv(k)                                    # columns: real-space lattice vectors
    rel = np.stack([centres.real - centres[0].real, centres.imag - centres[0].imag])
    index = np.round(np.linalg.solve(lattice, rel))
    design = np.c_[np.ones(len(centres)), index.T]
    coef_x, *_ = np.linalg.lstsq(design, centres.real, rcond=None)
    coef_y, *_ = np.linalg.lstsq(design, centres.imag, rcond=None)
    a1, a2 = np.array([coef_x[1], coef_y[1]]), np.array([coef_x[2], coef_y[2]])
    return float((np.hypot(*a1) + np.hypot(*a2)) / 2), (a1, a2), centres


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


def make_anchors(*, a_dac, magnification, magnification_sigma, b_dac, b_abs, b_sigma):
    """The two anchors: measured |A| at ``a_dac`` and |B| (m/rad) at ``b_dac`` (HT v0), with
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


def _least_squares(data, setup, free, anchors, start):
    free = np.asarray(free)
    sigma = data["wobble_px"] / np.sqrt(2)
    meas = jnp.asarray(data["measured"])
    if anchors is not None:
        anchor_dac = np.vstack([anchors["a_dac"], anchors["b_dac"]])
        anchor_us = jnp.asarray(np.array([controls(w) for w in anchor_dac]))

    def residual(x):
        theta = jnp.asarray(start).at[free].set(x)
        r = (predict(theta, data, setup) - meas) / sigma
        r = jnp.concatenate([jnp.real(r), jnp.imag(r)])
        if anchors is not None:
            a, b = _responses(theta[:16], anchor_us, jnp.asarray([setup.v0, setup.v0]), jnp.asarray(setup.v0))
            r = jnp.concatenate([r, jnp.array([
                (jnp.abs(a[0]) / anchors["magnification"] - 1) / anchors["magnification_sigma"],
                (jnp.abs(b[1]) / anchors["b_abs"] - 1) / anchors["b_sigma"]])])
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


def fit(data, setup, *, free, anchors=None, start=None):
    """Weighted least squares of the half-differences (and the anchors), from a data-driven start.

    Without a ``start``: the gains (and with two HTs the objective q) come from the closed-form
    rotation regression and the camera rotation from the strongest shift. That rotation is known
    only modulo pi (the start model may invert the image), so both are fitted and the better is
    kept. With ``anchors`` the kick scales are first read off the anchors (continuation), then
    everything is freed. Returns theta, chi2, dof, the covariance of the free parameters and the
    acceptance at the 99 % point of chi2.
    """
    free = list(free)
    if start is not None:
        return _least_squares(data, setup, free, anchors, np.asarray(start, float))
    geometry = [f for f in free if f not in (LOG_KS, LOG_KT)]
    best = None
    for flip in (0.0, np.pi):
        start = _rotation_start(data, setup)
        start[PHI] += flip
        result = _least_squares(data, setup, geometry, None, start)
        if anchors is not None:
            theta = result["theta"]
            a, _ = responses(theta[:16], anchors["a_dac"], setup.v0, setup.v0)
            _, b = responses(theta[:16], anchors["b_dac"], setup.v0, setup.v0)
            theta[LOG_KS] += np.log(abs(a[0]) / anchors["magnification"])
            theta[LOG_KT] += np.log(abs(b[0]) / anchors["b_abs"])
            theta = _least_squares(data, setup, geometry, None, theta)["theta"]
            result = _least_squares(data, setup, free, anchors, theta)
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


# ---------------------------------------------------------------------------- scoring (simulation only)
def held_out_settings(setup, n=60, seed=1):
    """Random settings inside the range of the 9 stage-1 settings, for scoring."""
    rng = np.random.default_rng(seed)
    lo, hi = setup.sets.min(0), setup.sets.max(0)
    return lo + (hi - lo) * rng.random((n, 4))


def beam_position_error_px(result, scope, setup, dac, kick_px=50.0):
    """Error (px) of the predicted beam displacement for kicks that move the beam ``kick_px``,
    at ``dac`` (HT v0), shift and tilt, against the hidden truth."""
    truth = scope.truth()
    data = dict(dac=np.vstack([dac, dac]), voltage=np.full(2 * len(dac), setup.v0),
                kind=np.r_[np.zeros(len(dac)), np.ones(len(dac))], command=np.ones(2 * len(dac)))
    true = np.asarray(predict(jnp.asarray(truth), data, setup))
    fitted = np.asarray(predict(jnp.asarray(result["theta"]), data, setup))
    err = kick_px * np.abs(fitted / true - 1)
    return err[:len(dac)], err[len(dac):]


def map_errors(result, scope, setup, dac, absolute):
    """Errors of |A|, |B| and the rotation at ``dac`` (HT v0) against the truth. Relative maps
    (``absolute=False``) are normalised at set 0 (|A|) and the diffraction setting (|B|)."""
    sets = np.vstack([setup.sets[[0, -1]], dac])
    at, bt = responses(scope.truth()[:16], sets, setup.v0, setup.v0)
    af, bf = responses(result["theta"][:16], sets, setup.v0, setup.v0)
    if absolute:
        mag, b = np.abs(af[2:]) / np.abs(at[2:]) - 1, np.abs(bf[2:]) / np.abs(bt[2:]) - 1
    else:
        mag = (np.abs(af[2:]) / np.abs(af[0])) / (np.abs(at[2:]) / np.abs(at[0])) - 1
        b = (np.abs(bf[2:]) / np.abs(bf[1])) / (np.abs(bt[2:]) / np.abs(bt[1])) - 1
    rotation = np.angle((af[2:] / af[0]) / (at[2:] / at[0]))
    return dict(magnification=mag, b=b, rotation=rotation)


# ---------------------------------------------------------------------------- the whole calibration
def calibrate(scope, setup, anchors):
    """The whole calibration (for Monte Carlo): stage 1, the anchored fit, and the final fit with
    the one-block stage 2."""
    d1, _ = acquire_stage1(scope, setup)
    stage1 = fit(d1, setup, free=ONE_HT_FREE + [LOG_KS, LOG_KT], anchors=anchors)
    d2 = acquire_stage2(scope, setup)
    final = fit(merge(d1, d2), setup, free=TWO_HT_FREE + [LOG_KS, LOG_KT], anchors=anchors)
    return dict(stage1=stage1, final=final, frames=(2 * len(d1["kind"]), 2 * len(d2["kind"])))
