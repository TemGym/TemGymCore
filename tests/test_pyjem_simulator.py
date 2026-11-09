"""Simulated microscope: electronics, camera, strictness, modes and the specimen-plane beam, on a small
column of its own (five solenoids imaging at 40 000x)."""
import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.components import HardEdgeSolenoid, Plane
from temgym_core.pyjem_simulator import (
    CameraModel,
    DACRangeError,
    DeflectorDrive,
    Electronics,
    LensDrive,
    OpticalTruth,
    SimulatedMicroscope,
    SpecimenBeam,
    UnmodelledStateError,
)
from temgym_core.run import position_and_angle_response

LENSES = ("IL1", "IL2", "IL3", "PL1")
#: Five solenoids (objective, IL1, IL2, IL3, PL1): gaps and lengths (m), and strengths per unit control.
GAPS = np.exp([-6.166, -1.568, -2.632, -2.856, -2.956, -1.170])
LENGTHS = np.exp([-7.824, -5.298, -5.298, -5.298, -6.908])
GAINS = np.array([-0.4257, 0.5500, 0.6500, 0.4500, -0.7500])
#: Relative strengths that image at 40 000x, and the DAC values that give them.
CONTROLS_40K = np.array([1.0, 0.6157, 0.5675, 1.7562, 0.8388])
MAG_ROW = {"IL1": 41782, "IL2": 24862, "IL3": 35544, "PL1": 64000}
DIFF_ROW = {"IL1": 19064, "IL2": 20701, "IL3": 39162, "PL1": 64000}
#: 512 sensor pixels of 55 um binned by 8: a 64x64 frame at 440 um, PSF 0.25 px.
SMALL_CAMERA = CameraModel(sensor_shape=(512, 512), sensor_pixel_m=55e-6, psf_sigma_sensor_pixels=2.0,
                           binnings=(8,))


ENTRANCES = np.cumsum(GAPS[:5]) + np.r_[0.0, np.cumsum(LENGTHS[:4])]
CAMERA_Z = GAPS.sum() + LENGTHS.sum()


def column(controls, voltage_v):
    q = jnp.asarray(GAINS) * jnp.asarray(controls)
    return [*(HardEdgeSolenoid(z=z, length=length, q=q[i]) for i, (z, length) in enumerate(zip(ENTRANCES, LENGTHS))),
            Plane(z=CAMERA_Z)]


def microscope(ray_sample_power=12, beam=SpecimenBeam(rms_size_m=17e-9, rms_angle_rad=0.47e-3)):
    drives = {n: LensDrive(MAG_ROW[n] / u) for n, u in zip(LENSES, CONTROLS_40K[1:])}
    electronics = Electronics(drives, DeflectorDrive(np.diag([1e-11, 1e-11, 2e-7, 2e-7])),
                              objective_dac=(45818, 39178), objective_control=1.0)
    truth = OpticalTruth(column=column, voltages_v=(200e3, 195e3), parameters={"made by": "the test"})
    return SimulatedMicroscope(truth, electronics, magnification_table={40000.0: MAG_ROW},
                               diffraction_table={0.4: DIFF_ROW}, incident_beam=beam, camera=SMALL_CAMERA,
                               beam_current_a=1.6e-9, ray_sample_power=ray_sample_power, detector_samples_per_pixel=4)


@pytest.fixture
def scope():
    return microscope()


def spot(frame):
    """Centre (x + iy, px) and rms radius (px) of the beam, background subtracted."""
    f = np.clip(frame - np.median(frame), 0, None)
    y, x = np.mgrid[:f.shape[0], :f.shape[1]]
    cx, cy = (f * x).sum() / f.sum(), (f * y).sum() / f.sum()
    return complex(cx, cy), float(np.sqrt((f * ((x - cx) ** 2 + (y - cy) ** 2)).sum() / f.sum() / 2))


def test_lens_drive_round_trip_and_range():
    drive = LensDrive(dac_per_unit=30000.0, dac_offset=100.0)
    assert drive.dac(1.5) == 45100
    assert drive.control(45100) == pytest.approx(1.5)
    with pytest.raises(DACRangeError):
        drive.dac(3.0)
    with pytest.raises(DACRangeError):
        drive.control(70000)


def test_deflector_dac_round_trip_and_momentum_scaling():
    drive = DeflectorDrive(np.diag([1e-11, 1e-11, 2e-7, 2e-7]))
    kick = np.array([20e-9, 0.0, 0.0, 3e-3])
    dac_values = drive.dac_values(kick, 200e3)
    assert dac_values == (32768 + 2000, 32768, 32768, 32768 + 15000)
    np.testing.assert_allclose(drive.kick(dac_values, 200e3), kick, rtol=1e-12)
    # The same DAC values deflect more at a lower high tension (fixed field, smaller momentum).
    assert drive.kick(dac_values, 195e3)[3] > kick[3]


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


def test_the_mag_row_gives_its_strengths_and_images_at_40000x(scope):
    setting = scope.physical_setting()
    np.testing.assert_allclose(setting.projector_controls, CONTROLS_40K, rtol=1e-4)
    assert setting.specimen_kick == (0.0, 0.0, 0.0, 0.0) and setting.voltage_v == 200e3
    a, b = position_and_angle_response(column(setting.projector_controls, setting.voltage_v))
    assert abs(a) == pytest.approx(40000, rel=0.01) and abs(b) < 1e-3 * abs(a)


@pytest.mark.parametrize("change, reason", [
    (lambda s: s.apertures[2].update(size=1), "OLA"),
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


def test_diff_mode_loads_its_own_rows_and_mag_brings_them_back(scope):
    tem3, _ = scope.pyjem()
    eos, lens = tem3.EOS3(), tem3.Lens3()
    read = lambda: [lens.GetIL1(), lens.GetIL2(), lens.GetIL3(), lens.GetPL1()]
    eos.SelectFunctionMode(4)
    assert eos.GetMagValue() == [40, "cm", "40cm"]
    assert read() == [DIFF_ROW[n] for n in LENSES]
    scope.physical_setting()                               # DIFF is modelled: no refusal
    eos.SelectFunctionMode(0)
    assert read() == [MAG_ROW[n] for n in LENSES]


def test_truth_is_available_for_scoring_only(scope):
    assert scope.truth.scoring_truth() == {"made by": "the test"}


def test_the_beam_starts_at_the_specimen_plane_and_moves_with_the_shift_deflector():
    """The spot is the beam's width times |A|; a condenser change neither moves nor resizes it; a
    CLA1 shift moves it by |A| times the shift."""
    scope = microscope(ray_sample_power=15)
    tem3, detector = scope.pyjem()
    camera = detector.Detector(detector.get_attached_detector()[0])
    frame = lambda: np.frombuffer(camera.snapshot_rawdata(), "<i4").reshape(64, 64).astype(float)
    pitch = 440e-6
    centre, rms = spot(frame())
    a = abs(position_and_angle_response(column(CONTROLS_40K, 200e3))[0])
    assert rms * pitch == pytest.approx(17e-9 * a, rel=0.1)
    lens = tem3.Lens3()
    lens.SetFLCAbs(2, int(0.9 * lens.GetCL3()))
    lens.SetFLCSw(2, 1)
    moved, rms_cl3 = spot(frame())
    assert abs(moved - centre) < 0.05 and rms_cl3 == pytest.approx(rms, rel=0.02)
    x, y = tem3.Def3().GetCLA1()
    tem3.Def3().SetCLA1(x + 2000, y)                       # 20 nm at the specimen
    shifted, _ = spot(frame())
    assert abs(shifted - centre) * pitch == pytest.approx(20e-9 * a, rel=0.02)


def test_a_microscope_from_a_ground_truth_file_is_the_same_microscope(tmp_path):
    """The file states each lens's place, length and strength per DAC unit; at another HT the console
    scales its DAC values with the beam's momentum, so the optics are the same."""
    import json
    names = ("objective", *LENSES)
    per_dac = GAINS[1:] * CONTROLS_40K[1:] / np.array([MAG_ROW[n] for n in LENSES])
    lenses = {n: {"z_m": z, "length_m": length} for n, z, length in zip(names, ENTRANCES, LENGTHS)}
    lenses["objective"]["q"] = GAINS[0]
    for n, k in zip(LENSES, per_dac):
        lenses[n]["q_per_dac"] = k
    path = tmp_path / "microscope.json"
    path.write_text(json.dumps(dict(
        reference_voltage_v=200e3, lenses=lenses, camera_z_m=CAMERA_Z,
        deflectors=dict(shift_m_per_dac=1e-11, tilt_rad_per_dac=2e-7),
        beam=dict(rms_size_m=17e-9, rms_angle_rad=0.47e-3, current_a=1.6e-9),
        objective_dac=dict(OLc=45818, OLf=39178), lens_table=dict(MAG={"40000": MAG_ROW}, DIFF={"0.4": DIFF_ROW})),
        default=float))
    expected = position_and_angle_response(column(CONTROLS_40K, 200e3))
    for ht_v in (200e3, 60e3):
        scope = SimulatedMicroscope.from_file(path, ht_v=ht_v)
        setting = scope.physical_setting()
        actual = position_and_angle_response(scope.truth.column(setting.projector_controls, setting.voltage_v))
        # the same up to the whole DAC values the console holds at another HT
        np.testing.assert_allclose(complex(actual[0]), complex(expected[0]), rtol=2e-4)       # A
        np.testing.assert_allclose(complex(actual[1]), complex(expected[1]), atol=1e-4)       # B, near 0 in focus
