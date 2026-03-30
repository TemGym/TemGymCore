import numpy as np
import pytest

from temgym_core.constants import compute_Rc_from_voltage, deflection_from_excitation, effective_accelerating_potential, voltage_scaling_ratio
from temgym_core.microscope_model import (
    DeflectorConfig, LensConfig, MicroscopeModel, OperatingMode,
    _mode_from_toml_table,
)


def test_operating_mode_interpolate_currents():
    mode = OperatingMode(
        control_values=np.array([10.0, 20.0, 30.0]),
        normalized_currents=np.array(
            [
                [0.0, 0.2],
                [0.5, 0.6],
                [1.0, 1.0],
            ]
        ),
        full_scale_current=10.0,
    )

    np.testing.assert_allclose(mode.interpolate_currents(20.0), [5.0, 6.0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(mode.interpolate_currents(15.0), [2.5, 4.0], rtol=0.0, atol=1e-12)


def test_operating_mode_validation():
    with pytest.raises(ValueError):
        OperatingMode(
            control_values=np.array([10.0, 5.0]),
            normalized_currents=np.array([[0.1], [0.2]]),
            full_scale_current=1.0,
        )

    with pytest.raises(ValueError):
        OperatingMode(
            control_values=np.array([10.0, 20.0]),
            normalized_currents=np.array([[0.1, 0.2]]),
            full_scale_current=1.0,
        )

    with pytest.raises(ValueError):
        OperatingMode(
            control_values=np.array([10.0, 20.0]),
            normalized_currents=np.array([[0.1], [1.2]]),
            full_scale_current=1.0,
        )

    with pytest.raises(ValueError):
        OperatingMode(
            control_values=np.array([10.0, 20.0]),
            normalized_currents=np.array([[-0.1], [0.2]]),
            full_scale_current=1.0,
        )


def test_operating_mode_signed_currents_validation():
    mode = OperatingMode(
        control_values=np.array([0.0, 1.0]),
        normalized_currents=np.array([[-0.2, 0.1], [0.3, -0.5]]),
        full_scale_current=2.0,
        allow_signed_currents=True,
    )
    np.testing.assert_allclose(
        mode.interpolate_normalized_currents(0.5),
        [0.05, -0.2],
        rtol=0.0,
        atol=1e-12,
    )

    with pytest.raises(ValueError):
        OperatingMode(
            control_values=np.array([0.0, 1.0]),
            normalized_currents=np.array([[-1.2], [0.0]]),
            full_scale_current=1.0,
            allow_signed_currents=True,
        )


def test_operating_mode_gc_scales_validation():
    with pytest.raises(ValueError):
        OperatingMode(
            control_values=np.array([0.0, 1.0]),
            normalized_currents=np.array([[0.1], [0.2]]),
            full_scale_current=1.0,
            gc_scales=np.array([1.0, 1.1]),
        )

    with pytest.raises(ValueError):
        OperatingMode(
            control_values=np.array([0.0, 1.0]),
            normalized_currents=np.array([[0.1], [0.2]]),
            full_scale_current=1.0,
            gc_scales=np.array([[1.0], [-0.2]]),
        )


def test_microscope_model_build_components_voltage_scaling():
    v_ref = 100e3
    voltage = 200e3
    V_star_ref = float(effective_accelerating_potential(v_ref))

    model = MicroscopeModel(
        voltage=voltage,
        reference_voltage=v_ref,
        lenses=(
            LensConfig(name="L1", z_position=0.1, turns=200.0, Gc=6.0e-6, Tc=1.0e-4),
            LensConfig(name="L2", z_position=0.2, turns=100.0, Gc=8.0e-6, Tc=2.0e-4),
        ),
        modes={
            "mag": OperatingMode(
                control_values=np.array([1.0, 2.0]),
                normalized_currents=np.array([[0.2, 0.4], [0.6, 0.8]]),
                full_scale_current=10.0,
            )
        },
    )

    components = model.build_components(mode_name="mag", control_value=1.5)
    assert len(components) == 2

    np.testing.assert_allclose(float(components[0].current), 4.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(float(components[1].current), 6.0, rtol=0.0, atol=1e-12)

    # Gc is now pure geometry: Gc_geom = Gc_toml * V*_ref
    np.testing.assert_allclose(float(components[0].Gc), 6.0e-6 * V_star_ref, rtol=1e-12)
    np.testing.assert_allclose(float(components[1].Gc), 8.0e-6 * V_star_ref, rtol=1e-12)

    # tc_voltage_exponent defaults to 0, so Tc remains unchanged.
    np.testing.assert_allclose(float(components[0].Tc), 1.0e-4, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(float(components[1].Tc), 2.0e-4, rtol=0.0, atol=0.0)


def test_microscope_model_tc_voltage_exponent():
    model = MicroscopeModel(
        voltage=300e3,
        reference_voltage=100e3,
        tc_voltage_exponent=1.0,
        lenses=(
            LensConfig(name="L1", z_position=0.1, turns=1.0, Gc=1.0e-6, Tc=2.0e-4),
        ),
        modes={
            "mode": OperatingMode(
                control_values=np.array([0.0, 1.0]),
                normalized_currents=np.array([[0.2], [0.4]]),
                full_scale_current=10.0,
            )
        },
    )
    components = model.build_components(mode_name="mode", control_value=0.5)
    scale = float(voltage_scaling_ratio(300e3, 100e3))
    np.testing.assert_allclose(float(components[0].Tc), 2.0e-4 * scale, rtol=1e-12)


def test_microscope_model_unknown_mode():
    model = MicroscopeModel(
        voltage=200e3,
        lenses=(LensConfig(name="L1", z_position=0.1, turns=1.0, Gc=1.0),),
        modes={
            "known": OperatingMode(
                control_values=np.array([0.0, 1.0]),
                normalized_currents=np.array([[0.1], [0.2]]),
                full_scale_current=1.0,
            )
        },
    )

    with pytest.raises(KeyError):
        model.build_components("unknown", 0.5)


def test_microscope_model_build_components_with_gc_scales():
    v_ref = 100e3
    voltage = 200e3
    V_star_ref = float(effective_accelerating_potential(v_ref))

    mode = OperatingMode(
        control_values=np.array([0.0, 1.0]),
        normalized_currents=np.array([[0.5], [0.5]]),
        full_scale_current=10.0,
        gc_scales=np.array([[1.0], [3.0]]),
    )

    model = MicroscopeModel(
        voltage=voltage,
        reference_voltage=v_ref,
        lenses=(LensConfig(name="L1", z_position=0.1, turns=100.0, Gc=5.0e-6),),
        modes={"m": mode},
    )

    comp = model.build_components("m", 0.5)[0]
    expected_gc = (
        5.0e-6
        * V_star_ref
        * 2.0  # interpolated between 1.0 and 3.0
    )
    np.testing.assert_allclose(float(comp.Gc), expected_gc, rtol=1e-12, atol=0.0)


# --- Deflector tests ---

def _make_model_with_deflector(voltage=200e3, Dc=2.0, turns=500):
    """Helper to build a MicroscopeModel with one lens and one deflector."""
    return MicroscopeModel(
        voltage=voltage,
        lenses=(LensConfig(name="L1", z_position=0.3, turns=1.0, Gc=1.0e-6),),
        deflectors=(
            DeflectorConfig(
                name="D1",
                z_position=0.1,
                spacing=0.005,
                turns=turns,
                Dc=Dc,
                shift_balance_x=-1.0,
                shift_balance_y=-1.0,
                tilt_balance_x=1.0,
                tilt_balance_y=1.0,
            ),
        ),
        modes={
            "mode": OperatingMode(
                control_values=np.array([0.0, 1.0]),
                normalized_currents=np.array([[0.5], [0.5]]),
                full_scale_current=1.0,
            )
        },
    )


def test_deflector_config_fields():
    dc = DeflectorConfig(
        name="D1", z_position=0.1, spacing=0.005, turns=100, Dc=1.5,
        shift_balance_x=-1.0, shift_balance_y=-1.0,
    )
    assert dc.name == "D1"
    assert dc.spacing == 0.005
    assert dc.Dc == 1.5
    assert dc.shift_balance_x == -1.0
    assert dc.tilt_balance_x == 1.0  # default


def test_build_components_no_deflectors_backward_compat():
    """MicroscopeModel without deflectors returns lenses only, same as before."""
    model = MicroscopeModel(
        voltage=200e3,
        lenses=(LensConfig(name="L1", z_position=0.1, turns=1.0, Gc=1.0e-6),),
        modes={
            "mode": OperatingMode(
                control_values=np.array([0.0, 1.0]),
                normalized_currents=np.array([[0.5], [0.5]]),
                full_scale_current=1.0,
            )
        },
    )
    components = model.build_components("mode", 0.5)
    assert len(components) == 1
    from temgym_core.components import ElectromagneticLens
    assert isinstance(components[0], ElectromagneticLens)


def test_build_components_with_deflector_zero_drive():
    """Deflector with zero drive produces a DoubleDeflector with zero shift/tilt."""
    model = _make_model_with_deflector()
    components = model.build_components("mode", 0.5)
    assert len(components) == 2  # 1 deflector + 1 lens

    from temgym_core.components import DoubleDeflector
    defl = [c for c in components if isinstance(c, DoubleDeflector)]
    assert len(defl) == 1
    assert float(defl[0].shift_x) == 0.0
    assert float(defl[0].tilt_x) == 0.0


def test_build_components_deflector_voltage_scaling():
    """Deflection angle = Dc * turns * I / sqrt(V*); doubling voltage reduces it."""
    Dc = 2.0
    turns = 500
    shift_I = 0.01  # 10 mA drive current

    v1 = 100e3
    v2 = 200e3

    model1 = _make_model_with_deflector(voltage=v1, Dc=Dc, turns=turns)
    model2 = _make_model_with_deflector(voltage=v2, Dc=Dc, turns=turns)

    drives = {"D1": (shift_I, 0.0, 0.0, 0.0)}
    comps1 = model1.build_components("mode", 0.5, deflector_drives=drives)
    comps2 = model2.build_components("mode", 0.5, deflector_drives=drives)

    from temgym_core.components import DoubleDeflector
    d1 = [c for c in comps1 if isinstance(c, DoubleDeflector)][0]
    d2 = [c for c in comps2 if isinstance(c, DoubleDeflector)][0]

    # Expected: alpha = Dc * turns * I / sqrt(V*)
    V_star_1 = effective_accelerating_potential(v1)
    V_star_2 = effective_accelerating_potential(v2)
    expected_1 = Dc * turns * shift_I / V_star_1 ** 0.5
    expected_2 = Dc * turns * shift_I / V_star_2 ** 0.5

    np.testing.assert_allclose(float(d1.shift_x), expected_1, rtol=1e-12)
    np.testing.assert_allclose(float(d2.shift_x), expected_2, rtol=1e-12)

    # Ratio should be sqrt(V2*) / sqrt(V1*)
    ratio = float(d1.shift_x) / float(d2.shift_x)
    expected_ratio = (V_star_2 / V_star_1) ** 0.5
    np.testing.assert_allclose(ratio, expected_ratio, rtol=1e-12)


def test_build_components_deflector_balance_passthrough():
    """Balance values from DeflectorConfig are passed through unchanged."""
    model = _make_model_with_deflector()
    drives = {"D1": (0.01, 0.02, 0.0, 0.0)}
    components = model.build_components("mode", 0.5, deflector_drives=drives)

    from temgym_core.components import DoubleDeflector
    defl = [c for c in components if isinstance(c, DoubleDeflector)][0]
    np.testing.assert_allclose(float(defl.shift_balance_x), -1.0, atol=0.0)
    np.testing.assert_allclose(float(defl.shift_balance_y), -1.0, atol=0.0)
    np.testing.assert_allclose(float(defl.tilt_balance_x), 1.0, atol=0.0)
    np.testing.assert_allclose(float(defl.tilt_balance_y), 1.0, atol=0.0)


def test_build_components_sorted_by_z():
    """Components are returned sorted by z-position."""
    model = _make_model_with_deflector()  # deflector at 0.1, lens at 0.3
    components = model.build_components("mode", 0.5)
    z_values = [float(c.z) for c in components]
    assert z_values == sorted(z_values)


def test_deflection_from_excitation_helper():
    """deflection_from_excitation matches manual computation."""
    Dc = 3.0
    NI = 1000.0
    voltage = 200e3
    V_star = effective_accelerating_potential(voltage)
    expected = Dc * NI / V_star ** 0.5

    result = deflection_from_excitation(Dc, NI, voltage)
    np.testing.assert_allclose(float(result), expected, rtol=1e-12)


def test_microscope_model_npz_roundtrip_with_deflectors(tmp_path):
    """DeflectorConfig survives to_npz / from_npz round-trip."""
    model = _make_model_with_deflector()
    filepath = str(tmp_path / "model.npz")
    model.to_npz(filepath)

    loaded = MicroscopeModel.from_npz(filepath)
    assert len(loaded.deflectors) == 1
    d = loaded.deflectors[0]
    assert d.name == "D1"
    np.testing.assert_allclose(d.z_position, 0.1)
    np.testing.assert_allclose(d.spacing, 0.005)
    np.testing.assert_allclose(d.turns, 500)
    np.testing.assert_allclose(d.Dc, 2.0)
    np.testing.assert_allclose(d.shift_balance_x, -1.0)
    np.testing.assert_allclose(d.tilt_balance_x, 1.0)


def test_microscope_model_npz_roundtrip_no_deflectors(tmp_path):
    """Models without deflectors still load correctly (backward compat)."""
    model = MicroscopeModel(
        voltage=200e3,
        lenses=(LensConfig(name="L1", z_position=0.1, turns=1.0, Gc=1.0e-6),),
        modes={
            "mode": OperatingMode(
                control_values=np.array([0.0, 1.0]),
                normalized_currents=np.array([[0.5], [0.5]]),
                full_scale_current=1.0,
            )
        },
    )
    filepath = str(tmp_path / "model.npz")
    model.to_npz(filepath)

    loaded = MicroscopeModel.from_npz(filepath)
    assert len(loaded.deflectors) == 0
    assert len(loaded.lenses) == 1


def test_duplicate_deflector_names_rejected():
    with pytest.raises(ValueError, match="Deflector names must be unique"):
        MicroscopeModel(
            voltage=200e3,
            lenses=(LensConfig(name="L1", z_position=0.1, turns=1.0, Gc=1.0e-6),),
            deflectors=(
                DeflectorConfig(name="D1", z_position=0.1, spacing=0.005, turns=1, Dc=1.0),
                DeflectorConfig(name="D1", z_position=0.2, spacing=0.005, turns=1, Dc=1.0),
            ),
            modes={
                "mode": OperatingMode(
                    control_values=np.array([0.0, 1.0]),
                    normalized_currents=np.array([[0.5], [0.5]]),
                    full_scale_current=1.0,
                )
            },
        )


# ---- _mode_from_toml_table tests ----

def test_mode_from_toml_table_basic():
    table = {
        "headers": ["mag", "L1", "L2"],
        "values": [
            [100.0, 0.5, 1.0],
            [200.0, 0.8, 2.0],
        ],
    }
    mode = _mode_from_toml_table(table, ["L1", "L2"])
    np.testing.assert_array_equal(mode.control_values, [100.0, 200.0])
    assert mode.n_lenses == 2
    assert mode.full_scale_current == 2.0
    np.testing.assert_allclose(
        mode.normalized_currents,
        [[0.25, 0.5], [0.4, 1.0]],
    )


def test_mode_from_toml_table_defaults():
    """Lenses not in headers get default_currents."""
    table = {
        "headers": ["spot", "L1"],
        "values": [[1.0, 0.5], [2.0, 1.0]],
    }
    mode = _mode_from_toml_table(table, ["L1", "L2"], {"L2": 0.3})
    # L2 should have constant 0.3 across both rows
    currents = mode.interpolate_currents(1.5)
    # L2 column is always 0.3; full_scale = max(1.0, 1.0) = 1.0
    assert mode.full_scale_current == 1.0
    np.testing.assert_allclose(
        mode.normalized_currents[:, 1], [0.3, 0.3],
    )


def test_mode_from_toml_table_signed():
    table = {
        "headers": ["ctrl", "L1"],
        "values": [[0.0, -1.5], [1.0, 1.5]],
    }
    mode = _mode_from_toml_table(table, ["L1"])
    assert mode.allow_signed_currents is True
    assert mode.full_scale_current == 1.5


def test_mode_from_toml_table_invalid():
    with pytest.raises(ValueError, match="Invalid mode table"):
        _mode_from_toml_table(
            {"headers": ["only_one"], "values": [[1.0]]},
            ["L1"],
        )


# ---- from_toml tests ----

MINIMAL_TOML = """\
[beam]
voltage_kV = 200

[lenses.L1]
z_m = 0.10
turns = 1000
Gc = 1e-5

[lenses.L2]
z_m = 0.30
turns = 2000
Gc = 2e-5

[modes.imaging.magnification]
headers = ["mag", "L1", "L2"]
values = [[100.0, 0.5, 1.0], [200.0, 0.8, 2.0]]
"""


def test_from_toml_basic(tmp_path):
    toml_path = tmp_path / "test.toml"
    toml_path.write_text(MINIMAL_TOML)

    model = MicroscopeModel.from_toml(str(toml_path))

    assert model.voltage == 200e3
    assert model.reference_voltage == 200e3
    assert len(model.lenses) == 2
    assert model.lenses[0].name == "L1"
    assert model.lenses[1].Gc == pytest.approx(2e-5)
    assert "magnification" in model.modes  # leaf name
    assert model.modes["magnification"].n_lenses == 2


def test_from_toml_mode_renaming(tmp_path):
    toml_path = tmp_path / "test.toml"
    toml_path.write_text(MINIMAL_TOML)

    model = MicroscopeModel.from_toml(
        str(toml_path),
        mode_names={"imaging.magnification": "mag"},
    )
    assert "mag" in model.modes
    assert "magnification" not in model.modes


def test_from_toml_mode_defaults(tmp_path):
    toml_path = tmp_path / "test.toml"
    toml_path.write_text(MINIMAL_TOML)

    # L2 is in the headers → default for L2 should be overridden.
    # Use default for L1 via a renamed mode, but L1 is also in headers,
    # so the table value should win.
    model = MicroscopeModel.from_toml(
        str(toml_path),
        mode_names={"imaging.magnification": "mag"},
        mode_defaults={"mag": {"L1": 999.0}},
    )
    mode = model.modes["mag"]
    # L1 column should be [0.5, 0.8] (from table), not 999.0
    currents_at_100 = mode.interpolate_currents(100.0)
    # full_scale = 2.0, L1 normalized = 0.5/2.0 = 0.25, so current = 0.25*2.0 = 0.5
    assert currents_at_100[0] == pytest.approx(0.5)


def test_from_toml_with_deflectors(tmp_path):
    toml_text = MINIMAL_TOML + """\

[deflectors.D1]
z_m = 0.20
spacing_m = 0.005
turns = 100
Dc = 1.5
shift_balance_x = -1.0
"""
    toml_path = tmp_path / "test.toml"
    toml_path.write_text(toml_text)

    model = MicroscopeModel.from_toml(str(toml_path))
    assert len(model.deflectors) == 1
    d = model.deflectors[0]
    assert d.name == "D1"
    assert d.Dc == pytest.approx(1.5)
    assert d.shift_balance_x == pytest.approx(-1.0)
    assert d.shift_balance_y == pytest.approx(1.0)  # default


def test_from_toml_with_detector(tmp_path):
    toml_text = MINIMAL_TOML + """\

[detector]
z_m = 0.80
pixel_size_m = 15e-6
shape = [2048, 2048]
"""
    toml_path = tmp_path / "test.toml"
    toml_path.write_text(toml_text)

    model = MicroscopeModel.from_toml(str(toml_path))
    assert "detector" in model.auxiliary
    assert model.auxiliary["detector"]["z"] == pytest.approx(0.80)
    assert model.auxiliary["detector"]["pixel_size"] == pytest.approx(15e-6)
    assert model.auxiliary["detector"]["shape"] == [2048, 2048]


def test_from_toml_apertures_in_auxiliary(tmp_path):
    toml_text = MINIMAL_TOML + """\

[apertures.C_aperture]
z_m = 0.25
radii_m = [10e-6, 50e-6]
"""
    toml_path = tmp_path / "test.toml"
    toml_path.write_text(toml_text)

    model = MicroscopeModel.from_toml(str(toml_path))
    assert "apertures" in model.auxiliary
    ap = model.auxiliary["apertures"]["C_aperture"]
    assert ap["z"] == pytest.approx(0.25)
    assert len(ap["radii"]) == 2


def test_from_toml_beam_params_in_auxiliary(tmp_path):
    toml_text = """\
[beam]
voltage_kV = 200
z_source_m = 0.0
virtual_source_diameter_nm = 10.0
source_half_angle_mrad = 0.05

[lenses.L1]
z_m = 0.10
turns = 1000
Gc = 1e-5

[modes.spot]
headers = ["s", "L1"]
values = [[1.0, 0.5]]
"""
    toml_path = tmp_path / "test.toml"
    toml_path.write_text(toml_text)

    model = MicroscopeModel.from_toml(str(toml_path))
    assert model.auxiliary["z_source"] == pytest.approx(0.0)
    assert model.auxiliary["virtual_source_diameter_nm"] == pytest.approx(10.0)
    assert model.auxiliary["source_half_angle_mrad"] == pytest.approx(0.05)


def test_from_toml_reference_voltage_override(tmp_path):
    toml_path = tmp_path / "test.toml"
    toml_path.write_text(MINIMAL_TOML)

    model = MicroscopeModel.from_toml(str(toml_path), reference_voltage=100e3)
    assert model.reference_voltage == pytest.approx(100e3)
    assert model.voltage == pytest.approx(200e3)


def test_from_toml_real_file():
    """Smoke test against the actual microscope.toml in the repo."""
    import pathlib
    toml_path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "examples" / "microscope_models" / "microscope.toml"
    )
    if not toml_path.exists():
        pytest.skip("microscope.toml not found")

    model = MicroscopeModel.from_toml(
        str(toml_path),
        mode_names={
            "illumination.parallel": "spot",
            "imaging.magnification": "mag",
            "imaging.diffraction": "diff",
        },
        mode_defaults={"spot": {"Obj_prefield": 1.0}},
    )

    assert model.voltage == pytest.approx(200e3)
    assert len(model.lenses) == 9
    assert len(model.deflectors) == 1
    assert set(model.modes.keys()) == {"spot", "mag", "diff"}
    assert model.modes["spot"].n_lenses == 9
    assert model.modes["mag"].n_lenses == 9

    # Spot mode: first control value should be 1.0
    np.testing.assert_allclose(
        model.modes["spot"].control_values[0], 1.0,
    )

    # Verify Obj_prefield default was applied in spot mode
    # It should be a constant column (not zero)
    lens_names = [l.name for l in model.lenses]
    obj_idx = lens_names.index("Obj_prefield")
    assert np.all(model.modes["spot"].normalized_currents[:, obj_idx] > 0)

    # Apertures should be in auxiliary
    assert "apertures" in model.auxiliary
    assert "C_aperture" in model.auxiliary["apertures"]

    # Detector should be in auxiliary
    assert "detector" in model.auxiliary

    # Build components should work
    components = model.build_components("spot", 3.0)
    assert len(components) > 0
