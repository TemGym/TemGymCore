"""Build the virtual NEOARM: a PyJEM-driven microscope with hidden TemGym optics.

The hidden optics are a synthetic truth
(``calibration_results/tilt_y_30_ray_acquisition/scoring_truth.json``). The hidden
electronics map 16-bit DAC values to lens controls and deflector kicks, and the camera
is ideal by default. TEM MAG has the 40k and 45k rows; TEM DIFF has camera lengths
0.15-1 m designed on the hidden optics (:func:`design_diffraction_table`). It is aligned at
200 kV or 60 kV (``ht_v``): the console tables scale with p(HT), so the optics are the same.

Use it as you would the real microscope::

    import virtual_neoarm
    microscope = virtual_neoarm.build(ht_v=60e3)      # set up as left at the console
    TEM3, detector = microscope.pyjem()               # at the microscope: from PyJEM import TEM3, detector

``stage1_rehearsal.ipynb`` runs the stage-1 session this way. :func:`build_virtual_neoarm` is the
lower-level builder (the bare scope, at its power-on state).
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
from temgym_core.virtual.pyjem import pyjem_interface  # noqa: E402

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


def design_diffraction_table(*, truth, drives, mag_words, camera_lengths_m=CAMERA_LENGTHS_M, max_word=63000):
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
    pp = at_voltage(np.asarray(truth.projector_parameters), truth.reference_voltage_v, truth.reference_voltage_v)
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


#: The second HT that goes with each aligned HT (the stage-2 step), by default.
SECOND_HT = {200e3: 195e3, 60e3: 80e3}


def build_virtual_neoarm(*, ht_v=200e3, second_ht_v=None, headroom_word=60000, ray_sample_power=18,
                         detector_samples_per_pixel=1, camera=None, seed=0, diffraction_table=None):
    """Virtual NEOARM, aligned at ``ht_v`` (200 or 60 kV), whose relaxed lens ranges fit the selected plan.

    The console tables (MAG rows, DIFF rows, spot, objective) are the 200 kV alignment with every DAC
    value scaled by p(ht_v) / p(200 kV): a magnetic lens acts through B / p, so the optics at ``ht_v``
    are the same as at 200 kV. The microscope starts at ``ht_v``; ``second_ht_v`` (default 195 kV
    with 200 kV, 80 kV with 60 kV) is the other HT it can run at, with the DAC values held.

    ``diffraction_table`` (camera length in m -> IL1, IL2, IL3, PL1 DAC values) replaces the
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
    from temgym_core.calibration.condenser_model import magnetic_strength_ratio

    second_ht_v = second_ht_v or SECOND_HT[float(ht_v)]
    truth = OpticalTruth.from_files(ACQUISITION / "experiment.json", ACQUISITION / "scoring_truth.json",
                                    voltages_v=(float(ht_v), float(second_ht_v)))
    scale = 1 / float(magnetic_strength_ratio(ht_v, truth.reference_voltage_v))      # p(ht_v) / p(200 kV)
    largest = np.r_[plan["projector_controls"][:, 1:].max(axis=0), plan["condenser_controls"].max(axis=0)]
    drives = {name: LensDrive(headroom_word / top) for name, top in zip(PROJECTOR + CONDENSER, largest)}
    # 0.01 nm and 0.2 µrad per CLA word: 20 nm and 3 mrad are 2000 and 15000 words.
    deflector = DeflectorDrive(np.diag([1e-11, 1e-11, 2e-7, 2e-7]))
    objective = Electronics(drives, deflector).objective_words              # the 200 kV objective
    electronics = Electronics(drives, deflector, objective_words=tuple(round(w * scale) for w in objective),
                              objective_control=scale)
    aligned = {m: {n: drives[n].word(c) for n, c in zip(PROJECTOR, controls)} for m, controls in baselines.items()}
    if diffraction_table is None:
        diffraction_table = {length: {n: round(w * scale) for n, w in row.items()} for length, row in
                             design_diffraction_table(truth=truth, drives=drives, mag_words=aligned[40000.0]).items()}
    magnification_table = {m: {n: round(w * scale) for n, w in row.items()} for m, row in aligned.items()}
    spot_table = {REFERENCE_SPOT: {n: drives[n].word(c * scale) for n, c in zip(CONDENSER, condenser_reference)}}
    scope = VirtualMicroscope(
        truth, electronics, magnification_table=magnification_table, spot_table=spot_table,
        diffraction_table=diffraction_table,
        camera=camera or CameraModel(), aperture_holes={hole: d for d, hole in APERTURE_HOLES.items()},
        beam_current_a=DOSE_PER_IMAGE * elementary_charge / EXPOSURE_S, ray_sample_power=ray_sample_power,
        detector_samples_per_pixel=detector_samples_per_pixel, seed=seed)
    gauge = dict(reference_magnification=40000.0, reference_spot=REFERENCE_SPOT,
                 projector_reference_ratios=baselines[40000.0], condenser_reference_ratios=condenser_reference)
    return scope, plan, gauge


def true_magnification_and_camera_length(*, scope):
    """The hidden |A| and |B| (m/rad) at the scope's present lens setting -- for scoring a
    rehearsal only; nothing the session sees."""
    s, t = scope.physical_setting(), scope.truth
    m = np.asarray(matrix5(at_voltage(np.asarray(t.projector_parameters), s.voltage_v, t.reference_voltage_v),
                           np.asarray(s.projector_controls)))
    return float(np.hypot(m[0, 0], m[0, 1])), float(np.hypot(m[0, 2], m[0, 3]))


def true_projector(*, scope, units):
    """The hidden projector as the 16 fit parameters, in the fit's DAC-unit convention
    (``analyse`` maps DAC values by ``units`` and ``model.controls``) --
    for scoring and drawing a rehearsal only."""
    from model import CONTROL_PER_DAC
    p = np.array(scope.truth.projector_parameters, float)
    per_unit = np.array([scope.electronics.lenses[n].words_per_unit for n in PROJECTOR])
    p[12:16] += np.log((1 / per_unit) / (np.asarray(units) * CONTROL_PER_DAC))
    return p


def true_deflector_calibration(*, scope):
    """The hidden beam shift (m) and tilt (rad) per deflector unit at the reference HT -- for
    scoring a rehearsal only."""
    k = np.asarray(scope.electronics.deflector.kick_per_word)
    return float(k[0, 0]), float(k[2, 2])


class VirtualNEOARM:
    """The virtual NEOARM, set up as an operator leaves it at the console before a session.

    ``pyjem()`` returns its ``TEM3`` and ``detector``, used exactly like PyJEM's. ``specimen`` puts a
    specimen in (a :class:`~temgym_core.virtual.CrossGrating` or
    :class:`~temgym_core.virtual.PolycrystallineGold`) or takes it out (``None``); on the real
    microscope that is done by hand. ``scope`` is the underlying
    :class:`~temgym_core.virtual.VirtualMicroscope`, with the hidden truth for scoring.
    """

    def __init__(self, scope):
        self.scope = scope
        self._pyjem = pyjem_interface(scope)

    def pyjem(self):
        """``TEM3, detector``: the virtual microscope behind PyJEM's names."""
        return self._pyjem

    @property
    def specimen(self):
        """The specimen in the beam (``None``: vacuum)."""
        return self.scope.specimen

    @specimen.setter
    def specimen(self, value):
        self.scope.specimen = value


def build(*, ht_v=60e3, magnification=40000.0, binning=8, exposure_s=0.01, ray_sample_power=22,
          detector_samples_per_pixel=8, **options):
    """The virtual NEOARM aligned at ``ht_v`` and set up as left at the console for a session: MAG at
    the ``magnification`` row, spot 3, condenser-aperture hole 1, the camera at ``binning`` and
    ``exposure_s``, specimen out. 8 x 8 detector sub-samples keep sub-pixel spot positions (with 1,
    a spot of about a pixel locks to pixel centres); 2**22 rays per frame keep the beam smooth (2**17
    leaves about 10 % grain). ``options`` go to :func:`build_virtual_neoarm`."""
    from acquire import select_magnification

    scope, _, _ = build_virtual_neoarm(ht_v=ht_v, ray_sample_power=ray_sample_power,
                                       detector_samples_per_pixel=detector_samples_per_pixel, **options)
    microscope = VirtualNEOARM(scope)
    TEM3, detector = microscope.pyjem()
    TEM3.HT3().SetHtValue(ht_v)
    eos = TEM3.EOS3()
    eos.SelectFunctionMode(0)
    select_magnification(eos=eos, magnification=magnification)
    eos.SelectSpotSize(REFERENCE_SPOT)
    aperture = TEM3.Apt3()
    aperture.SelectKind(1)
    aperture.SetSize(APERTURE_HOLES[50.0])
    detector.Detector(detector.get_attached_detector()[0]).set_detectorsetting(
        {"BinningSize": binning, "ExposureTimeValue": exposure_s * 1e6})
    return microscope
