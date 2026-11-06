"""Build the virtual NEOARM: a PyJEM-driven microscope with hidden TemGym optics.

The hidden optics are a synthetic truth
(``calibration_results/tilt_y_30_ray_acquisition/scoring_truth.json``). The hidden
electronics map 16-bit DAC values to lens controls and deflector kicks, and the camera
is ideal by default. TEM MAG has the 40k and 45k rows; TEM DIFF has camera lengths
0.15-1 m designed on the hidden optics (:func:`design_diffraction_table`).

Drive it through PyJEM's own names::

    from temgym_core.virtual.pyjem import facade
    scope, _, _ = build_virtual_neoarm(ray_sample_power=17, detector_samples_per_pixel=8)
    TEM3, detector = facade(scope)

``stage1_virtual_session.ipynb`` runs the stage-1 session this way.
"""

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
from scipy.constants import elementary_charge  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

from temgym_core.calibration.projector_model import at_voltage, matrix5  # noqa: E402
from temgym_core.virtual import (  # noqa: E402
    CameraModel,
    DeflectorDrive,
    Electronics,
    LensDrive,
    OpticalTruth,
    VirtualMicroscope,
)

HERE = Path(__file__).parent
ACQUISITION = HERE / "calibration_results" / "tilt_y_30_ray_acquisition"
PRESETS = HERE.parents[1] / "src" / "temgym_core" / "calibration" / "presets"
PROJECTOR = ("IL1", "IL2", "IL3", "PL1")
CONDENSER = ("CL1", "CL3", "CM")
REFERENCE_SPOT = 2                        # PyJEM index of "spot 3"
APERTURE_HOLES = {50.0: 1, 20.0: 2}       # condenser-aperture hole numbers
DOSE_PER_IMAGE = 1e8                      # incident electrons, as in the synthetic acquisition
EXPOSURE_S = 0.01
#: DIFF selector rows: camera length (m) at this camera, as JEOL's nominal steps.
CAMERA_LENGTHS_M = (0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)


def load_plan():
    """The selected acquisition's settings and its reference (gauge) controls."""
    plan = {k: np.asarray(v) for k, v in np.load(ACQUISITION / "settings.npz").items()}
    spec = json.loads((ACQUISITION / "experiment.json").read_text())
    protocol = json.loads((PRESETS / "beam_measurement_protocol.json").read_text())
    baseline45 = plan["projector_controls"][(plan["purpose"] == "check") & (plan["voltage_index"] == 0)
                                            & (plan["nominal_magnification"] == 45000.0)][0]
    return plan, spec, {40000.0: np.asarray(protocol["projector_reference_controls"])[1:],
                        45000.0: baseline45[1:]}, np.asarray(spec["condenser_reference_controls"])


def design_diffraction_table(truth, drives, mag_words, camera_lengths_m=CAMERA_LENGTHS_M, *, max_word=63000):
    """TEM DIFF rows for the hidden optics: words that put the back focal plane on the camera.

    For each camera length L: PL1 stays at its MAG value; IL1, IL2 and IL3 are chosen so that
    A = 0 (a shift at the specimen does not move the beam) and |B| = L (m/rad, at 200 kV).
    That leaves one free direction, taken closest to the MAG row in IL2 and IL3, so IL1 does
    the switching, as in a JEOL DIFF mode. Rows are solved in increasing L, each starting
    from the last, and kept within ``max_word``.

    Parameters
    ----------
    truth : OpticalTruth
    drives : dict
        Lens name -> :class:`LensDrive`.
    mag_words : dict
        The MAG row the DIFF rows start from (words for IL1, IL2, IL3, PL1).
    camera_lengths_m : sequence of float
    max_word : int

    Returns
    -------
    dict
        Camera length (m) -> words for IL1, IL2, IL3, PL1.
    """
    pp = at_voltage(np.asarray(truth.projector_parameters), truth.voltages_v[0], truth.voltages_v[0])
    ray_map = jax.jit(lambda c: matrix5(pp, c))
    base = np.array([drives[n].control(mag_words[n]) for n in PROJECTOR])
    per_unit = np.array([drives[n].words_per_unit for n in PROJECTOR[:3]])

    def a_b(controls):
        m = np.asarray(ray_map(np.r_[1.0, controls]))
        return complex(m[0, 0], -m[0, 1]), complex(m[0, 2], -m[0, 3])

    def residuals(log_ratio, length):
        a, b = a_b(np.r_[base[:3] * np.exp(log_ratio), base[3]])
        signed_a = (a * np.conj(b)).real / abs(b)      # A and B share the Larmor rotation
        return np.r_[signed_a / 100, 50 * np.log(abs(b) / length), log_ratio[1:], 0.1 * log_ratio[:1]]

    bounds = np.log(np.array([2000.0, max_word])[:, None] / (per_unit * base[:3]))
    x, table = np.log([0.6, 1.0, 1.0]), {}
    for length in sorted(camera_lengths_m):
        x = least_squares(residuals, np.clip(x, *(bounds + [[1e-3], [-1e-3]])), args=(length,),
                          x_scale=0.1, bounds=bounds).x
        controls = np.r_[base[:3] * np.exp(x), base[3]]
        rounded = np.array([drives[n].word(c) for n, c in zip(PROJECTOR, controls)])
        # whole DAC values: the neighbour (within 2 of each of IL1-IL3) with the smallest |A|
        steps = np.stack(np.meshgrid(*[np.arange(-2, 3)] * 3, indexing="ij"), -1).reshape(-1, 3)
        candidates = rounded + np.c_[steps, np.zeros(len(steps), int)]
        a_values = [abs(a_b(np.array([drives[n].control(w) for n, w in zip(PROJECTOR, c)]))[0]) for c in candidates]
        words = dict(zip(PROJECTOR, (int(w) for w in candidates[int(np.argmin(a_values))])))
        a, b = a_b(np.array([drives[n].control(words[n]) for n in PROJECTOR]))
        if abs(a) > 1.0 or abs(np.log(abs(b) / length)) > 1e-3:
            raise ValueError(f"no DIFF row for camera length {length} m within the DAC range (|A| {abs(a):.3g}, "
                             f"|B| {abs(b):.4g} m/rad)")
        table[float(length)] = words
    return table


def build_virtual_neoarm(*, headroom_word=60000, ray_sample_power=18, detector_samples_per_pixel=1,
                         camera=None, seed=0, diffraction_table=None):
    """Virtual NEOARM whose relaxed lens ranges fit the selected plan.

    ``diffraction_table`` (camera length in m -> IL1, IL2, IL3, PL1 words) replaces the
    designed TEM DIFF rows, e.g. with recorded DIFF values mapped onto this lens table.

    Returns
    -------
    scope : VirtualMicroscope
    plan : dict
        The selected acquisition's settings.
    gauge : dict
        Reference magnification/spot and the relative controls assigned to them.
    """
    plan, spec, baselines, condenser_reference = load_plan()
    truth = OpticalTruth.from_files(ACQUISITION / "experiment.json", ACQUISITION / "scoring_truth.json")
    largest = np.r_[plan["projector_controls"][:, 1:].max(axis=0), plan["condenser_controls"].max(axis=0)]
    drives = {name: LensDrive(headroom_word / top) for name, top in zip(PROJECTOR + CONDENSER, largest)}
    # 0.01 nm and 0.2 µrad per CLA word: 20 nm and 3 mrad are 2000 and 15000 words.
    deflector = DeflectorDrive(np.diag([1e-11, 1e-11, 2e-7, 2e-7]))
    electronics = Electronics(drives, deflector)
    magnification_table = {m: {n: drives[n].word(c) for n, c in zip(PROJECTOR, controls)}
                           for m, controls in baselines.items()}
    if diffraction_table is None:
        diffraction_table = design_diffraction_table(truth, drives, magnification_table[40000.0])
    spot_table = {REFERENCE_SPOT: {n: drives[n].word(c) for n, c in zip(CONDENSER, condenser_reference)}}
    scope = VirtualMicroscope(
        truth, electronics, magnification_table=magnification_table, spot_table=spot_table,
        diffraction_table=diffraction_table,
        camera=camera or CameraModel(), aperture_holes={hole: d for d, hole in APERTURE_HOLES.items()},
        beam_current_a=DOSE_PER_IMAGE * elementary_charge / EXPOSURE_S, ray_sample_power=ray_sample_power,
        detector_samples_per_pixel=detector_samples_per_pixel, seed=seed)
    gauge = dict(reference_magnification=40000.0, reference_spot=REFERENCE_SPOT,
                 projector_reference_ratios=baselines[40000.0], condenser_reference_ratios=condenser_reference)
    return scope, plan, gauge


def true_magnification_and_camera_length(scope):
    """The hidden |A| and |B| (m/rad) at the scope's present lens setting -- for scoring a
    rehearsal only; nothing the session sees."""
    s, t = scope.physical_setting(), scope.truth
    m = np.asarray(matrix5(at_voltage(np.asarray(t.projector_parameters), s.voltage_v, t.voltages_v[0]),
                           np.asarray(s.projector_controls)))
    return float(np.hypot(m[0, 0], m[0, 1])), float(np.hypot(m[0, 2], m[0, 3]))


def true_projector(scope, units):
    """The hidden projector as the 16 fit parameters, in the fit's DAC-unit convention
    (``stage1_analysis`` maps DAC values by ``units`` and ``projector_calibration.controls``) --
    for scoring and drawing a rehearsal only."""
    from projector_calibration import CONTROL_PER_DAC
    p = np.array(scope.truth.projector_parameters, float)
    per_unit = np.array([scope.electronics.lenses[n].words_per_unit for n in PROJECTOR])
    p[12:16] += np.log((1 / per_unit) / (np.asarray(units) * CONTROL_PER_DAC))
    return p


def true_deflector_calibration(scope):
    """The hidden beam shift (m) and tilt (rad) per deflector unit at the reference HT -- for
    scoring a rehearsal only."""
    k = np.asarray(scope.electronics.deflector.kick_per_word)
    return float(k[0, 0]), float(k[2, 2])
