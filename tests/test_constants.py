import warnings

import numpy as np
import pytest

from temgym_core.constants import (
    compute_I0_Cf_from_focal_rotation,
    compute_I0_Gc_from_focal_rotation,
    compute_I0_from_focal_length,
    compute_I0_from_rotation,
    compute_Kv_from_voltage,
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


def test_compute_i0_gc_from_focal_rotation_round_trip():
    Rc = float(compute_Kv_from_voltage(200e3))
    focal_length = 5e-3
    rotation_angle = 1.0

    I0, Gc = compute_I0_Gc_from_focal_rotation(
        focal_length=focal_length,
        rotation_angle=rotation_angle,
        Rc=Rc,
    )

    expected_f = 1.0 / (Gc * I0**2)
    expected_psi = Rc * I0
    np.testing.assert_allclose(expected_f, focal_length, rtol=1e-12)
    np.testing.assert_allclose(expected_psi, rotation_angle, rtol=1e-12)


def test_compute_i0_cf_alias_matches_canonical_and_warns():
    Rc = float(compute_Kv_from_voltage(200e3))
    focal_length = 7e-3
    rotation_angle = 0.5

    canonical = compute_I0_Gc_from_focal_rotation(
        focal_length=focal_length,
        rotation_angle=rotation_angle,
        Rc=Rc,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        alias = compute_I0_Cf_from_focal_rotation(
            focal_length=focal_length,
            rotation_angle=rotation_angle,
            Rc=Rc,
        )

    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    np.testing.assert_allclose(alias[0], canonical[0], rtol=1e-12)
    np.testing.assert_allclose(alias[1], canonical[1], rtol=1e-12)


@pytest.mark.parametrize(
    "focal_length, rotation_angle, Rc",
    [
        (0.0, 1.0, 1e-4),
        (-1e-3, 1.0, 1e-4),
        (1e-3, 1.0, 0.0),
    ],
)
def test_compute_i0_gc_from_focal_rotation_validation(focal_length, rotation_angle, Rc):
    with pytest.raises(ValueError):
        compute_I0_Gc_from_focal_rotation(
            focal_length=focal_length,
            rotation_angle=rotation_angle,
            Rc=Rc,
        )


def test_compute_i0_from_focal_length_round_trip():
    focal_length = 4e-3
    Gc = 3.5e-6
    I0 = float(compute_I0_from_focal_length(focal_length=focal_length, Gc=Gc))
    np.testing.assert_allclose(1.0 / (Gc * I0**2), focal_length, rtol=1e-12)


@pytest.mark.parametrize(
    "focal_length, Gc",
    [
        (0.0, 1e-6),
        (-1e-3, 1e-6),
        (1e-3, 0.0),
        (1e-3, -1e-6),
    ],
)
def test_compute_i0_from_focal_length_validation(focal_length, Gc):
    with pytest.raises(ValueError):
        compute_I0_from_focal_length(focal_length=focal_length, Gc=Gc)


def test_compute_i0_from_rotation_round_trip():
    Rc = float(compute_Kv_from_voltage(200e3))
    rotation_angle = 0.7
    I0 = float(compute_I0_from_rotation(rotation_angle=rotation_angle, Rc=Rc))
    np.testing.assert_allclose(Rc * I0, rotation_angle, rtol=1e-12)


def test_compute_i0_from_rotation_validation():
    with pytest.raises(ValueError):
        compute_I0_from_rotation(rotation_angle=1.0, Rc=0.0)
