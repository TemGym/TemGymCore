"""Virtual microscope: electronics, camera, strictness and identity with the ray pipeline."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from temgym_core.virtual import (
    CameraModel,
    DeflectorDrive,
    LensDrive,
    PhysicalSetting,
    UnmodelledStateError,
    WordRangeError,
)

ROOT = Path(__file__).parents[1]
REFERENCE = Path(__file__).parent / "reference"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


neoarm = load("virtual_neoarm_example", ROOT / "examples/microscope_models/virtual_neoarm.py")
ray_images = load("make_ray_images", REFERENCE / "make_ray_images.py")

#: 512 sensor pixels of 55 um binned by 8: the reference scenario's 64x64 frame at 440 um, PSF 0.25 px.
SMALL_CAMERA = CameraModel(sensor_shape=(512, 512), sensor_pixel_m=55e-6, psf_sigma_sensor_pixels=2.0,
                           binnings=(8,))


@pytest.fixture
def scope():
    built, _, _ = neoarm.build_virtual_neoarm(ray_sample_power=10, detector_samples_per_pixel=2,
                                              camera=SMALL_CAMERA)
    return built


def test_lens_drive_round_trip_and_range():
    drive = LensDrive(words_per_unit=30000.0, offset_word=100.0)
    assert drive.word(1.5) == 45100
    assert drive.control(45100) == pytest.approx(1.5)
    with pytest.raises(WordRangeError):
        drive.word(3.0)
    with pytest.raises(WordRangeError):
        drive.control(70000)


def test_deflector_words_round_trip_and_momentum_scaling():
    drive = DeflectorDrive(np.diag([1e-11, 1e-11, 2e-7, 2e-7]))
    kick = np.array([20e-9, 0.0, 0.0, 3e-3])
    words = drive.words(kick, 200e3)
    assert words == (32768 + 2000, 32768, 32768, 32768 + 15000)
    np.testing.assert_allclose(drive.kick(words, 200e3), kick, rtol=1e-12)
    # The same words deflect more at a lower high tension (fixed field, smaller momentum).
    assert drive.kick(words, 195e3)[3] > kick[3]


def test_camera_noise_statistics_and_digitisation():
    rng = np.random.default_rng(1)
    camera = CameraModel(sensor_shape=(256, 256), binnings=(1,), read_noise_electrons=2.0, background_electrons=3.0)
    frame = camera.expose(np.full((256, 256), 100.0), rng).astype(float)
    assert frame.mean() == pytest.approx(103.0, rel=0.01)
    assert frame.var() == pytest.approx(103.0 + 4.0 + 1 / 12, rel=0.03)
    clipped = CameraModel(sensor_shape=(8, 8), binnings=(1,), data_bits=16).expose(np.full((8, 8), 1e6), rng)
    assert clipped.dtype == np.dtype("<i2") and clipped.max() == 32767
    summed = camera.expose(np.full((256, 256), 100.0), rng, frames=4).astype(float)
    assert summed.mean() == pytest.approx(100.0 + 4 * 3.0, rel=0.01)


def test_ray_images_match_the_saved_reference_images():
    saved = np.load(REFERENCE / "ray_images.npz")
    np.testing.assert_allclose(ray_images.render(*ray_images.scenario()), saved["images"], rtol=1e-9, atol=0)


def test_virtual_microscope_renders_exactly_like_the_ray_pipeline(scope):
    spec, settings, truth = ray_images.scenario()
    for row in ray_images.ROWS:
        vi = int(settings["voltage_index"][row])
        for aperture in range(2):
            setting = PhysicalSetting(spec["voltages_v"][vi], vi, tuple(settings["condenser_controls"][row]),
                                      tuple(settings["projector_controls"][row]),
                                      tuple(settings["specimen_kick"][row]), aperture)
            expected = scope.expected_image(setting, 8) * spec["nominal_incident_electrons_per_image"]
            assert np.array_equal(expected, ray_images.render_row(spec, settings, truth, row, aperture))


def test_words_map_to_the_planned_controls(scope):
    plan, _, _, _ = neoarm.load_plan()
    setting = scope.physical_setting()          # power-on: 40k row, spot 3, 50 um hole, 200 kV
    np.testing.assert_allclose(setting.projector_controls, plan["projector_controls"][0] * [1, 1, 1, 1, 1]
                               / plan["projector_controls"][0] * neoarm.load_plan()[2][40000.0].tolist()[0:1]
                               * 0 + np.r_[1.0, neoarm.load_plan()[2][40000.0]], rtol=3e-5)
    assert setting.specimen_kick == (0.0, 0.0, 0.0, 0.0)
    assert setting.aperture_index == 1 and setting.voltage_v == 200e3


@pytest.mark.parametrize("change, reason", [
    (lambda s: s.apertures[1].update(size=0), "condenser aperture hole 0"),
    (lambda s: setattr(s, "function_mode", 2), "LowMAG"),
    (lambda s: s.deflectors.update(IS1=[30000, 32768]), "IS1"),
    (lambda s: setattr(s, "ht_v", 120e3), "high tension"),
    (lambda s: s.direct.update(OLf=39000), "objective"),
    (lambda s: s.direct.update(PL2=40000), "PL2"),
])
def test_unmodelled_states_are_refused_with_reasons(scope, change, reason):
    change(scope)
    with pytest.raises(UnmodelledStateError) as error:
        scope.acquire()
    assert any(reason in r for r in error.value.reasons)


def test_blanked_beam_gives_a_dark_frame_even_in_unmodelled_states(scope):
    scope.function_mode = 2
    scope.beam_blank = 1
    frame = scope.acquire().astype(float)
    assert frame.shape == (64, 64)
    assert frame.mean() == pytest.approx(scope.camera.background_electrons, abs=0.5)


def test_diff_mode_loads_its_own_rows_and_puts_the_back_focal_plane_on_the_camera(scope):
    from temgym_core.virtual.pyjem import pyjem_interface
    tem3, _ = pyjem_interface(scope)
    eos, lens = tem3.EOS3(), tem3.Lens3()
    mag = [lens.GetIL1(), lens.GetIL2(), lens.GetIL3(), lens.GetPL1()]
    eos.SelectFunctionMode(4)
    eos.SetSelector(7)
    assert eos.GetMagValue() == [80, "cm", "80cm"]
    words = scope.diffraction_table[7][1]
    assert [lens.GetIL1(), lens.GetIL2(), lens.GetIL3(), lens.GetPL1()] == [words[n] for n in neoarm.PROJECTOR]
    setting = scope.physical_setting()                     # DIFF is modelled: no refusal
    m = np.asarray(neoarm.matrix5(neoarm.at_voltage(np.asarray(scope.truth.projector_parameters), 200e3),
                                  np.asarray(setting.projector_controls)))
    assert np.hypot(m[0, 0], m[0, 1]) < 1.0                 # A = 0 (40 000 at the 40k row)
    assert np.hypot(m[0, 2], m[0, 3]) == pytest.approx(0.8, rel=1e-3)   # |B| = camera length
    eos.SelectFunctionMode(0)
    assert [lens.GetIL1(), lens.GetIL2(), lens.GetIL3(), lens.GetPL1()] == mag


def test_truth_is_available_for_scoring_only(scope):
    truth = json.loads((neoarm.ACQUISITION / "scoring_truth.json").read_text())
    assert scope.truth.scoring_truth() == truth


def _anchor_scope():
    from temgym_core.virtual.pyjem import pyjem_interface
    built, _, _ = neoarm.build_virtual_neoarm(ray_sample_power=17, detector_samples_per_pixel=4)   # 256 x 256, 110 um
    tem3, detector = pyjem_interface(built)
    camera = detector.Detector(detector.get_attached_detector()[0])
    camera.set_detectorsetting({"BinningSize": 8, "ExposureTimeValue": 0.01e6})
    return built, tem3, camera


def _true_ab(scope):
    s = scope.physical_setting()
    m = np.asarray(neoarm.matrix5(neoarm.at_voltage(np.asarray(scope.truth.projector_parameters), s.voltage_v),
                                  np.asarray(s.projector_controls)))
    return np.hypot(m[0, 0], m[0, 1]), np.hypot(m[0, 2], m[0, 3])


def _frame(camera):
    return np.frombuffer(camera.snapshot_rawdata(), "<i4").reshape(256, 256).astype(float)


def test_cross_grating_image_gives_the_true_magnification():
    """Rays weighted by the grating's transmission; the whole-image lattice fit gives |A|."""
    calibration = load("model", ROOT / "examples/microscope_models/model.py")
    from temgym_core.virtual import CrossGrating
    scope, tem3, camera = _anchor_scope()
    lens = tem3.Lens3()
    lens.SetFLCSw(2, 1)
    lens.SetFLCAbs(2, int(0.7 * lens.GetCL3()))          # spread the beam over the frame
    scope.specimen = CrossGrating(period=230e-9)
    period_px, _, info = calibration.grating_period_px(_frame(camera))
    assert info["explained"] > 0.9
    assert period_px * 110e-6 / 230e-9 == pytest.approx(_true_ab(scope)[0], rel=2e-3)


def test_gold_rings_give_the_true_camera_length():
    """Gaussian beamlets, one per lit grain, through the projector; the ring radius gives |B|."""
    calibration = load("model", ROOT / "examples/microscope_models/model.py")
    from temgym_core.constants import energy2wavelength
    from temgym_core.virtual import PolycrystallineGold
    scope, tem3, camera = _anchor_scope()
    eos = tem3.EOS3()
    eos.SelectFunctionMode(4)
    eos.SetSelector(7)                                   # 80 cm
    scope.specimen = PolycrystallineGold()
    radii, _ = calibration.ring_radii_px(_frame(camera), calibration.Camera(n=256, pitch=110e-6))
    b = radii["111"] * 110e-6 / (energy2wavelength(200e3) / calibration.AU_RINGS["111"])
    assert b == pytest.approx(_true_ab(scope)[1], rel=1e-3)


def test_the_beam_starts_at_the_specimen_plane(monkeypatch):
    """``build`` leaves the condenser out: the spot on the camera is the beam's width times |A|
    at the 40k row and its spread of angles times |B| at the DIFF row, and a condenser change
    does not move or resize it."""
    monkeypatch.syspath_prepend(str(ROOT / "examples/microscope_models"))
    from acquire import measure_spot, select_camera_length
    microscope = neoarm.build(ht_v=60e3, ray_sample_power=15, detector_samples_per_pixel=4,
                              beam_width_nm=40.0, convergence_mrad=1.1)
    tem3, detector = microscope.pyjem()
    camera = detector.Detector(detector.get_attached_detector()[0])
    centre, rms_40k, _ = measure_spot(frame=_frame(camera), background=3.0)
    a = neoarm.true_magnification_and_camera_length(scope=microscope.scope)[0]
    lens = tem3.Lens3()
    lens.SetFLCAbs(2, int(0.9 * lens.GetCL3()))
    lens.SetFLCSw(2, 1)
    moved, rms_cl3, _ = measure_spot(frame=_frame(camera), background=3.0)
    assert abs(moved - centre) < 0.05 and rms_cl3 == pytest.approx(rms_40k, rel=0.02)
    select_camera_length(eos=tem3.EOS3(), camera_length_cm=40)
    rms_diff = measure_spot(frame=_frame(camera), background=3.0)[1]
    b = neoarm.true_magnification_and_camera_length(scope=microscope.scope)[1]
    fwhm = 2.3548 * 110e-6                                       # px rms -> m full width on the camera
    assert rms_40k * fwhm == pytest.approx(40e-9 * a, rel=0.1)
    assert rms_diff * fwhm == pytest.approx(1.1e-3 * b, rel=0.2)
