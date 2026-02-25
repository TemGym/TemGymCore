import numpy as np
import pytest

from temgym_core.constants import compute_Rc_from_voltage, voltage_scaling_ratio
from temgym_core.microscope_model import LensConfig, MicroscopeModel, OperatingMode


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
    rc_ref_base = float(compute_Rc_from_voltage(v_ref))
    rc_lens_ref = 1.5 * rc_ref_base

    model = MicroscopeModel(
        voltage=voltage,
        reference_voltage=v_ref,
        lenses=(
            LensConfig(name="L1", z_position=0.1, turns=200.0, Gc=6.0e-6, Rc=rc_lens_ref, Tc=1.0e-4),
            LensConfig(name="L2", z_position=0.2, turns=100.0, Gc=8.0e-6, Rc=0.5 * rc_lens_ref, Tc=2.0e-4),
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

    gc_scale = float(voltage_scaling_ratio(voltage, v_ref))
    np.testing.assert_allclose(float(components[0].Gc), 6.0e-6 * gc_scale, rtol=1e-12)
    np.testing.assert_allclose(float(components[1].Gc), 8.0e-6 * gc_scale, rtol=1e-12)

    rc_voltage_base = float(compute_Rc_from_voltage(voltage))
    np.testing.assert_allclose(float(components[0].Rc), 1.5 * rc_voltage_base, rtol=1e-12)
    np.testing.assert_allclose(float(components[1].Rc), 0.75 * rc_voltage_base, rtol=1e-12)

    # tc_voltage_exponent defaults to 0, so Tc remains unchanged.
    np.testing.assert_allclose(float(components[0].Tc), 1.0e-4, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(float(components[1].Tc), 2.0e-4, rtol=0.0, atol=0.0)


def test_microscope_model_tc_voltage_exponent():
    model = MicroscopeModel(
        voltage=300e3,
        reference_voltage=100e3,
        tc_voltage_exponent=1.0,
        lenses=(
            LensConfig(name="L1", z_position=0.1, turns=1.0, Gc=1.0e-6, Rc=1.0e-6, Tc=2.0e-4),
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
        lenses=(LensConfig(name="L1", z_position=0.1, turns=1.0, Gc=1.0, Rc=1.0),),
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
    rc_ref_base = float(compute_Rc_from_voltage(v_ref))

    mode = OperatingMode(
        control_values=np.array([0.0, 1.0]),
        normalized_currents=np.array([[0.5], [0.5]]),
        full_scale_current=10.0,
        gc_scales=np.array([[1.0], [3.0]]),
    )

    model = MicroscopeModel(
        voltage=voltage,
        reference_voltage=v_ref,
        lenses=(LensConfig(name="L1", z_position=0.1, turns=100.0, Gc=5.0e-6, Rc=rc_ref_base),),
        modes={"m": mode},
    )

    comp = model.build_components("m", 0.5)[0]
    expected_gc = (
        5.0e-6
        * float(voltage_scaling_ratio(voltage, v_ref))
        * 2.0  # interpolated between 1.0 and 3.0
    )
    np.testing.assert_allclose(float(comp.Gc), expected_gc, rtol=1e-12, atol=0.0)
