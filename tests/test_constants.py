import numpy as np
import pytest

from temgym_core.constants import (
    compute_Kv_from_voltage,
    compute_NI_Gc_from_lens_parameters,
    effective_accelerating_potential,
    relativistic_voltage_correction,
    relativistic_voltage_factor,
    voltage_scaling_ratio,
)


def test_relativistic_voltage_factor_alias_matches_correction():
    voltage = 200e3
    np.testing.assert_allclose(
        relativistic_voltage_factor(voltage),
        relativistic_voltage_correction(voltage),
        rtol=0.0,
        atol=0.0,
    )


def test_effective_accelerating_potential_definition():
    voltage = 300e3
    expected = voltage * relativistic_voltage_factor(voltage)
    np.testing.assert_allclose(
        effective_accelerating_potential(voltage),
        expected,
        rtol=1e-12,
    )


def test_voltage_scaling_ratio_is_unity_at_reference():
    voltage = 120e3
    ratio = voltage_scaling_ratio(voltage, voltage)
    np.testing.assert_allclose(ratio, 1.0, rtol=0.0, atol=0.0)


def test_voltage_scaling_ratio_decreases_with_higher_voltage():
    reference = 100e3
    high = 300e3
    assert voltage_scaling_ratio(high, reference) < 1.0


def test_compute_ni_gc_from_lens_parameters_round_trip():
    Rc = float(compute_Kv_from_voltage(200e3))
    focal_length = 5e-3
    rotation_angle = 1.0

    NI, Gc = compute_NI_Gc_from_lens_parameters(
        focal_length=focal_length,
        rotation_angle=rotation_angle,
        Rc=Rc,
    )

    expected_f = 1.0 / (Gc * NI**2)
    expected_psi = Rc * NI
    np.testing.assert_allclose(expected_f, focal_length, rtol=1e-12)
    np.testing.assert_allclose(expected_psi, rotation_angle, rtol=1e-12)


@pytest.mark.parametrize(
    "focal_length, rotation_angle, Rc",
    [
        (0.0, 1.0, 1e-4),
        (-1e-3, 1.0, 1e-4),
        (1e-3, 1.0, 0.0),
    ],
)
def test_compute_ni_gc_from_lens_parameters_validation(focal_length, rotation_angle, Rc):
    with pytest.raises(ValueError):
        compute_NI_Gc_from_lens_parameters(
            focal_length=focal_length,
            rotation_angle=rotation_angle,
            Rc=Rc,
        )
