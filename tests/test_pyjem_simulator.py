"""Simulated microscope: electronics, camera, strictness, modes and the specimen-plane beam, on a small
column of its own (five solenoids imaging at 40 000x)."""
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.aberrations import SeidelCoeffs
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
from temgym_core.pyjem_simulator.seidel_plate import SeidelPlate
from temgym_core.ray import Ray
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
    (lambda s: s.deflectors.update(PLA=[30000, 32768]), "PLA"),
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


def microscope_with(pla=False, diffraction_camera_length_m=None, ray_sample_power=12):
    """The test microscope with two DIFF rows (0.2 m and 0.4 m), and optionally a PLA deflector at the plane of PL1
    that kicks 2e-5 rad per DAC unit."""
    drives = {n: LensDrive(MAG_ROW[n] / u) for n, u in zip(LENSES, CONTROLS_40K[1:])}
    electronics = Electronics(drives, DeflectorDrive(np.diag([1e-11, 1e-11, 2e-7, 2e-7])),
                              objective_dac=(45818, 39178), objective_control=1.0)
    extra = {}
    if pla:
        extra = dict(pla_z_m=ENTRANCES[4],
                     pla_kick=lambda x, y, voltage_v, diffraction: np.array([2e-5 * (x - 32768), 2e-5 * (y - 32768)]))
    truth = OpticalTruth(column=column, voltages_v=(200e3, 195e3), parameters={"made by": "the test"}, **extra)
    return SimulatedMicroscope(truth, electronics, magnification_table={40000.0: MAG_ROW},
                               diffraction_table={0.2: DIFF_ROW, 0.4: DIFF_ROW}, incident_beam=SpecimenBeam(
                                   rms_size_m=17e-9, rms_angle_rad=0.47e-3),
                               diffraction_camera_length_m=diffraction_camera_length_m, camera=SMALL_CAMERA,
                               beam_current_a=1.6e-9, ray_sample_power=ray_sample_power, detector_samples_per_pixel=4)


def test_pla_moves_the_pattern_linearly_when_the_optics_give_it_a_plane():
    """PLA is a deflector at the plane of PL1: it moves the beam on the camera, equally far either way."""
    scope = microscope_with(pla=True, ray_sample_power=15)
    tem3, detector = scope.pyjem()
    camera = detector.Detector(detector.get_attached_detector()[0])
    frame = lambda: np.frombuffer(camera.snapshot_rawdata(), "<i4").reshape(64, 64).astype(float)
    start, _ = spot(frame())
    moved = {}
    for sign in (+1, -1):
        tem3.Def3().SetPLA(32768 + sign * 1000, 32768)
        centre, _ = spot(frame())
        moved[sign] = centre - start
    assert abs(moved[+1]) > 3                                           # px: 0.02 rad over about 0.25 m
    assert abs(moved[+1] + moved[-1]) < 0.05 * abs(moved[+1])


def test_the_diffraction_selector_starts_on_the_chosen_camera_length():
    """A console left on 40 cm gives 40 cm when switched to DIFF; by default the first (shortest) row."""
    chosen = microscope_with(diffraction_camera_length_m=0.4).pyjem()[0].EOS3()
    chosen.SelectFunctionMode(4)
    assert chosen.GetMagValue() == [40, "cm", "40cm"]
    default = microscope_with().pyjem()[0].EOS3()
    default.SelectFunctionMode(4)
    assert default.GetMagValue() == [20, "cm", "20cm"]


def test_a_seidel_plate_has_field_terms_only_so_a_centred_beam_is_untouched():
    """Distortion E kicks a ray from object height h by -E h^3 (the plate adds -W to the path); a ray from the axis is not kicked, and A and B are
    unchanged. (Spherical aberration B, which acts on a ray by its aperture position alone, is left out.)"""
    plate_distance = 5e-4
    plate = SeidelPlate(z=plate_distance, object_plane_dist=plate_distance, coeffs=SeidelCoeffs(E=2.75e17))

    def kick(x, y=0.0, dx=0.0, dy=0.0):
        after = plate(Ray(x=x, y=y, dx=dx, dy=dy, z=plate_distance, pathlength=0.0))
        return float(after.dx - dx), float(after.dy - dy)

    kick_x, kick_y = kick(2e-7)                                          # a ray from 0.2 um off the axis, parallel
    assert kick_x == pytest.approx(-2.75e17 * (2e-7) ** 3, rel=1e-6)     # -E h^3 = -2.2 mrad
    assert abs(kick_y) < 1e-12 * abs(kick_x)
    assert kick(4e-7)[0] == pytest.approx(8 * kick_x, rel=1e-6)          # twice as far: eight times the kick
    assert abs(kick(0.0)[0]) < 1e-30                                     # from the axis: none
    # a ray from the axis that crosses the plate off-axis (tilted by 4e-4 rad) is not kicked either
    assert abs(kick(plate_distance * 4e-4, dx=4e-4)[0]) < 1e-12 * abs(kick_x)
    with_plate = position_and_angle_response((plate, *column(CONTROLS_40K, 200e3)))
    without = position_and_angle_response(column(CONTROLS_40K, 200e3))
    np.testing.assert_allclose([complex(v) for v in with_plate], [complex(v) for v in without], rtol=1e-6, atol=1e-9)


def test_cl3_sets_the_beam_size_and_an_off_axis_lens_moves_the_beam(tmp_path):
    """With ``beam_size`` in the file, CL3 sets the beam's rms size at the specimen (seen on the camera times |A|);
    with ``offset_m`` on a lens, the beam lands off-centre by what that lens, shifted, gives a ray from the axis."""
    import json
    from temgym_core.ray import Ray
    from temgym_core.run import run_to_end
    names = ("objective", *LENSES)
    per_dac = GAINS[1:] * CONTROLS_40K[1:] / np.array([MAG_ROW[n] for n in LENSES])
    lenses = {n: {"z_m": z, "length_m": length} for n, z, length in zip(names, ENTRANCES, LENGTHS)}
    lenses["objective"]["q"] = GAINS[0]
    for n, k in zip(LENSES, per_dac):
        lenses[n]["q_per_dac"] = k
    lenses["IL2"]["offset_m"] = [2e-6, 0.0]
    path = tmp_path / "microscope.json"
    path.write_text(json.dumps(dict(
        reference_voltage_v=200e3, lenses=lenses, camera_z_m=CAMERA_Z,
        deflectors=dict(shift_m_per_dac=1e-11, tilt_rad_per_dac=2e-7),
        beam=dict(rms_size_m=17e-9, rms_angle_rad=0.47e-3, current_a=1.6e-9),
        beam_size=dict(CL3_smallest_dac=32000, smallest_rms_m=3e-9, rms_m_per_dac=4e-11),
        objective_dac=dict(OLc=45818, OLf=39178), lens_table=dict(MAG={"40000": MAG_ROW}, DIFF={"0.4": DIFF_ROW})),
        default=float))
    scope = SimulatedMicroscope.from_file(path, ht_v=200e3, ray_sample_power=15, detector_samples_per_pixel=4)
    scope.camera = SMALL_CAMERA
    tem3, detector = scope.pyjem()
    camera = detector.Detector(detector.get_attached_detector()[0])
    frame = lambda: np.frombuffer(camera.snapshot_rawdata(), "<i4").reshape(64, 64).astype(float)
    pitch = 440e-6
    setting = scope.physical_setting()
    column = scope.truth.column(np.asarray(setting.projector_controls), setting.voltage_v)
    a = abs(complex(position_and_angle_response(column)[0]))
    end = run_to_end(Ray(x=0.0, y=0.0, dx=0.0, dy=0.0, z=0.0, pathlength=0.0), list(column))
    landing = complex(float(end.x), float(end.y)) / pitch                    # px from the frame centre
    assert abs(landing) > 2                                                  # the offset moves the beam measurably
    lens = tem3.Lens3()
    lens.SetFLCSw(2, 1)
    for cl3 in (33000, 34000):
        lens.SetFLCAbs(2, cl3)
        centre, rms = spot(frame())
        expected = np.hypot(3e-9, 4e-11 * (cl3 - 32000)) * a / pitch
        assert rms == pytest.approx(expected, rel=0.05)
        assert abs(centre - (31.5 + 31.5j) - np.conj(landing)) < 0.3          # the camera's image is mirrored


def test_a_file_of_electromagnetic_lenses_shows_a_beam_shift_where_temgym_puts_it(tmp_path):
    """Lenses as TemGym ElectromagneticLenses, a deflector matrix, a turned and unmirrored camera and the console's
    condenser values, all from the file: a CLA1 kick displaces the beam on the frame by exactly what TemGym's column
    gives, x along the columns and y up the rows."""
    import json
    from temgym_core.components import ElectromagneticLens, Rotator
    lenses = {"objective": dict(z_m=0.002, ampere_turns=1.0, Gc=1.6e8)}
    for n, z, turns, gc in zip(LENSES, (0.4, 0.47, 0.58, 0.69), (-0.08, -0.08, -0.08, 0.08), (2.0, 2.0, 0.3, 9.0)):
        lenses[n] = dict(z_m=z, ampere_turns_per_dac=turns, Gc=gc, Tc=1e-6)
    shift = 1e-11
    camera = dict(sensor_shape=[512, 512], sensor_pixel_m=55e-6, psf_sigma_sensor_pixels=2.0, binnings=[8], mirrored=False)
    path = tmp_path / "microscope.json"
    path.write_text(json.dumps(dict(
        reference_voltage_v=200e3, lenses=lenses, camera_z_m=1.03, camera_rotation_rad=0.6, camera=camera,
        deflectors=dict(kick_per_dac=[[shift, 0, 0, 0], [0, -shift, 0, 0], [0, 0, 1e-7, 0], [0, 0, 0, 1e-7]]),
        beam=dict(rms_size_m=2e-9, rms_angle_rad=1e-5, current_a=1.6e-9), condenser_dac=dict(CL3=33866),
        objective_dac=dict(OLc=45818, OLf=38938), lens_table=dict(MAG={"40000": MAG_ROW}, DIFF={"0.4": DIFF_ROW}))))
    scope = SimulatedMicroscope.from_file(path, ht_v=200e3, ray_sample_power=14, detector_samples_per_pixel=4)
    tem3, detector = scope.pyjem()
    assert tem3.Lens3().GetCL3() == 33866
    camera = detector.Detector(detector.get_attached_detector()[0])
    frame = lambda: np.frombuffer(camera.snapshot_rawdata(), "<i4").reshape(64, 64).astype(float)
    column = [ElectromagneticLens(z=0.002, turns=1.0, current=1.0, Gc=1.6e8),
              *(ElectromagneticLens(z=lenses[n]["z_m"], turns=lenses[n]["ampere_turns_per_dac"], current=MAG_ROW[n],
                                    Gc=lenses[n]["Gc"], Tc=1e-6) for n in LENSES),
              Rotator(z=1.03, angle=float(np.degrees(0.6))), Plane(z=1.03)]
    a = complex(position_and_angle_response(column)[0])
    before = spot(frame())[0]
    tem3.Def3().SetCLA1(32768 + 3000, 32768)
    displaced = spot(frame())[0] - before
    expected = a * 3000 * shift / 440e-6                                      # px, x + iy
    assert abs(expected) > 5
    assert abs(displaced - expected) < 0.02 * abs(expected)                    # not mirrored, turned by the camera


# ------------------------------------------------------------------------------------------------ the condenser
VIRTUAL_NEOARM = Path(__file__).parents[1] / "examples" / "virtual_neoarm"


def virtual_condenser():
    """The virtual NEOARM's designed condenser, with the fitted objective's front field (control 1)."""
    import json
    from temgym_core.pyjem_simulator import Condenser
    objective = json.loads((VIRTUAL_NEOARM / "virtual_neoarm.json").read_text())["lenses"]["objective"]
    front = dict(z_m=-objective["z_m"], ampere_turns=objective["ampere_turns"], Gc=objective["Gc"], Tc=0.0)
    d = json.loads((VIRTUAL_NEOARM / "condenser.json").read_text())
    return Condenser.from_dict(d, front), d["design"]


def condenser_matrix(condenser, cl3, kick=(0.0, 0.0, 0.0, 0.0)):
    from temgym_core.run import column_matrix
    dac = dict(CL1=condenser.spot_table[2]["CL1"], CL3=cl3, CM=condenser.cm_dac)
    column = condenser.components(dac, hole=0, aperture_position_dac=(2048, 2048), specimen_kick=kick,
                                  objective_control=1.0, voltage_v=200e3) + (Plane(z=0.0),)
    return np.asarray(column_matrix(column, source_z=condenser.source_z_m, voltage=200e3))


def test_the_condenser_deflectors_give_the_shift_and_tilt_asked_for_at_the_specimen():
    condenser, _ = virtual_condenser()
    kick = np.array([3e-8, -2e-8, 4e-4, 1e-4])
    offset = condenser_matrix(condenser, 33866, kick)[:4, 4] - condenser_matrix(condenser, 33866)[:4, 4]
    assert offset == pytest.approx(kick, rel=1e-9, abs=1e-15)


def test_brightness_focuses_the_beam_on_the_specimen_or_makes_it_parallel():
    """At the designed CL3s, a source point lands on one point of the specimen (focused), or a source angle no
    longer changes the angle at the specimen (parallel)."""
    condenser, design = virtual_condenser()
    focused = condenser_matrix(condenser, 33400)                 # the CL3 the design focuses at
    parallel = condenser_matrix(condenser, design["CL3_parallel_at_spot_3"])
    console = condenser_matrix(condenser, 33866)
    assert abs(focused[0, 2]) < 1e-3 * abs(console[0, 2])        # position no longer depends on the source angle
    assert np.abs(parallel[2:4, 2:4]).max() < 1e-3 * np.abs(console[2:4, 2:4]).max()


def test_the_virtual_neoarm_condenser_through_pyjem():
    """Spot size sets CL1 and CL3 from the spot table, a smaller hole lets less current through, and a changed CL2 (not
    on the NEOARM) is refused."""
    from temgym_core.pyjem_simulator import UnmodelledStateError
    scope = SimulatedMicroscope.from_file(VIRTUAL_NEOARM / "virtual_neoarm.json", ht_v=200e3, magnification=40000,
                                          condenser_path=VIRTUAL_NEOARM / "condenser.json", ray_sample_power=14,
                                          detector_samples_per_pixel=1)
    tem3, _ = scope.pyjem()
    lens, eos, apertures = tem3.Lens3(), tem3.EOS3(), tem3.Apt3()
    table = scope.truth.condenser.spot_table
    eos.SelectSpotSize(4)
    assert (lens.GetCL1(), lens.GetCL3()) == (table[4]["CL1"], table[4]["CL3"])
    eos.SelectSpotSize(2)
    assert (lens.GetCL1(), lens.GetCL3(), lens.GetCM()) == (31232, 33866, 26624)

    setting = lambda: scope.physical_setting()
    apertures.SelectKind(1)
    apertures.SetSize(1)
    big = scope.expected_image(setting(), 4).sum()
    apertures.SetSize(4)
    small = scope.expected_image(setting(), 4).sum()
    assert 0 < small < big

    eos.SelectSpotSize(7)                                        # spot 8, the smallest
    assert lens.GetCL1() == table[7]["CL1"]
    setting()
    eos.SelectSpotSize(2)
    lens.SetFLCAbs(1, 1000)
    lens.SetFLCSw(1, 1)
    with pytest.raises(UnmodelledStateError, match="CL2"):
        setting()


# ------------------------------------------------------------------------------------------------ lens hysteresis
def test_lens_hysteresis_lags_by_the_way_a_value_was_reached_and_a_large_swing_wipes_it():
    from temgym_core.pyjem_simulator import LensHysteresis
    law = LensHysteresis(widths_dac=(100.0, 1000.0), fractions=(0.01, 0.002))
    up = law.effective_sequence([30000, 32000])[-1]            # reached from below
    down = law.effective_sequence([34000, 32000])[-1]          # reached from above
    assert up == pytest.approx(32000 - (0.01 * 100 + 0.002 * 1000))
    assert down == pytest.approx(32000 + (0.01 * 100 + 0.002 * 1000))
    small_loop = law.effective_sequence([30000, 32000, 31950, 32000])[-1]
    assert small_loop == pytest.approx(up)                      # a small loop comes back to where it was
    wiped = law.effective_sequence([34000, 32000, 20000, 32000])[-1]
    assert wiped == pytest.approx(up)                           # a large swing wipes the earlier approach from above
    assert LensHysteresis().effective_sequence([1, 5, 3])[-1] == 3


def test_a_lens_with_hysteresis_shows_the_route_through_pyjem():
    """The same IL1 value reached from below and from above gives two slightly different lens controls; without
    hysteresis they are the same."""
    from dataclasses import replace
    from temgym_core.pyjem_simulator import LensHysteresis
    scope = SimulatedMicroscope.from_file(VIRTUAL_NEOARM / "virtual_neoarm.json", ht_v=200e3, magnification=40000,
                                          condenser_path=VIRTUAL_NEOARM / "condenser.json", ray_sample_power=10,
                                          detector_samples_per_pixel=1)
    scope.electronics = replace(scope.electronics, hysteresis={"IL1": LensHysteresis((500.0,), (0.01,))})
    scope.reset()
    lens = scope.pyjem()[0].Lens3()
    value = lens.GetIL1()

    def control_after(route):
        for v in route:
            lens.SetFLCAbs(10, v)
            lens.SetFLCSw(10, 1)
        return scope.physical_setting().projector_controls[1]

    from_below = control_after([value - 3000, value])
    from_above = control_after([value + 3000, value])
    assert from_above - from_below == pytest.approx(2 * 0.01 * 500)
    assert lens.GetIL1() == value                               # the console shows the DAC value, not the lag
