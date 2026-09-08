"""Check energy scaling and independent source geometry against physical invariants."""

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
from temgym_core.calibration.condenser_model import (
    condenser_at_voltage,
    initial_parameters,
    magnetic_strength_ratio,
    matrix5,
    physical_geometry,
    projector_at_voltage,
)
from temgym_core.calibration.projector_model import initial_guess
from temgym_core.calibration.projector_model import matrix5 as projector_matrix5
from temgym_core.constants import energy2wavelength


def test_strength_ratio_matches_relativistic_de_broglie_wavelength():
    for voltage in [100000.0, 200000.0, 202000.0, 300000.0]:
        expected = energy2wavelength(voltage) / energy2wavelength(200000.0)
        np.testing.assert_allclose(
            magnetic_strength_ratio(voltage), expected, rtol=2e-08
        )


def test_source_waist_can_move_without_moving_any_lens_or_aperture():
    p = np.r_[initial_parameters(), -3.0, 0.2, -0.15]
    source = np.r_[p[0] + 0.1, p[13:19] + 0.01]
    shifted = condenser_at_voltage(p, 200000.0, source_parameters=source)
    before, after = (
        np.asarray(physical_geometry(p)),
        np.asarray(physical_geometry(shifted)),
    )
    assert abs(after[0] - before[0]) > 1.0
    np.testing.assert_allclose(after[1:], before[1:], atol=1e-12)
    np.testing.assert_allclose(shifted[13:19], source[1:])


def test_reference_voltage_and_symplectic_transport():
    p = np.r_[initial_parameters(), -3.0, 0.2, -0.15]
    np.testing.assert_allclose(condenser_at_voltage(p, 200000.0), p)
    projector = initial_guess()
    controls = np.array([1.0, 1.1, 0.9, 1.0])
    np.testing.assert_allclose(
        projector_at_voltage(projector, controls, 200000.0),
        projector_matrix5(projector, controls),
    )
    symplectic = np.block(
        [[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]]
    )
    for transfer in [
        matrix5(condenser_at_voltage(p, 202000.0), controls[:3]),
        projector_at_voltage(projector, controls, 202000.0),
    ]:
        m = np.asarray(transfer)[:4, :4]
        error = np.abs(m.T @ symplectic @ m - symplectic)
        roundoff = (
            32
            * np.finfo(float).eps
            * (np.abs(m).T @ np.abs(symplectic) @ np.abs(m) + 1)
        )
        assert np.all(error < roundoff)


def test_projector_voltage_changes_every_magnetic_lens():
    p = initial_guess()
    controls = np.array([1.0, 0.9, 1.1, 1.0])
    ratio = float(magnetic_strength_ratio(202000.0))
    expected = projector_matrix5(p, np.r_[ratio, controls * ratio])
    np.testing.assert_allclose(
        projector_at_voltage(p, controls, 202000.0), expected, rtol=1e-12
    )
