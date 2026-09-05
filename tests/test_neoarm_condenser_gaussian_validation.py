"""Decomposition/moment tests only; detector propagation is GPU-only."""
from pathlib import Path
import sys

import jax
import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1]/"examples/lens_inversion"
if str(EXAMPLES) not in sys.path: sys.path.insert(0, str(EXAMPLES))

from neoarm_condenser_gaussian_validation import (
    covariance_to_mixed_source, rank_adapted_source_modes, validate_matrix,
    render_covariance_gaussian_modes, OMEGA,
)
from temgym_core.constants import energy2wavelength
from temgym_core.neoarm_calibration import DetectorGeometry

jax.config.update("jax_enable_x64", True)
EPSILON = float(energy2wavelength(200e3))/(4*np.pi)


def coupled_covariance():
    angle = .37
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    squeeze = np.diag([1.8, .7, 1/1.8, 1/.7])
    rotate = np.block([[rotation, np.zeros((2, 2))], [np.zeros((2, 2)), rotation]])
    curvature = np.array([[.3, -.2], [-.2, -.4]])
    lens = np.block([[np.eye(2), np.zeros((2, 2))], [curvature, np.eye(2)]])
    reorder = [0, 2, 1, 3]
    symplectic = (lens@rotate@squeeze)[np.ix_(reorder, reorder)]
    scales = np.array([8e-9, EPSILON/8e-9]*2)
    return symplectic@np.diag([1.2, 1.2, 1.8, 1.8])@symplectic.T*np.outer(scales, scales)


@pytest.mark.parametrize("covariance", [
    np.diag(np.array([5e-9, 50e-6, 6e-9, 45e-6])**2), coupled_covariance()])
def test_williamson_decomposition_closes_and_is_pure_plus_rank_two(covariance):
    source, metadata = covariance_to_mixed_source(covariance)
    scales = np.sqrt(np.diag(covariance))
    np.testing.assert_allclose(source.total_covariance/np.outer(scales, scales), covariance/np.outer(scales, scales), atol=1e-13)
    centre_eigen = np.linalg.eigvalsh(source.centre_covariance/np.outer(scales, scales))
    assert centre_eigen.min() >= -1e-13
    assert np.sum(centre_eigen > 1e-10) <= 2
    pure = source.pure_covariance
    np.testing.assert_allclose(pure@OMEGA@pure, EPSILON**2*OMEGA, atol=1e-37, rtol=1e-10)
    assert metadata["williamson_symplectic_error"] < 1e-12


def test_gaussian_cubature_recovers_fixed_source_moments_without_normalization():
    covariance = coupled_covariance(); mean = np.array([2e-9, 1e-6, -3e-9, 2e-6])
    source, _ = covariance_to_mixed_source(covariance, mean_state=mean)
    modes, weights, metadata = rank_adapted_source_modes(source, 3)
    states = np.stack([modes.x, modes.dx, modes.y, modes.dy], axis=-1)
    np.testing.assert_allclose(weights.sum(), 1., atol=5e-16)
    np.testing.assert_allclose(weights@states, mean, rtol=1e-13, atol=1e-22)
    residual = states-mean
    measured = source.pure_covariance+np.einsum("i,ij,ik->jk", weights, residual, residual)
    scales = np.sqrt(np.diag(covariance))
    np.testing.assert_allclose(measured/np.outer(scales, scales), covariance/np.outer(scales, scales), atol=1e-12)
    assert metadata["coherent_mode_count"] == 9


def test_pure_source_uses_one_mode_and_subquantum_source_rejected():
    scale = np.array([8e-9, EPSILON/8e-9]*2)
    source, _ = covariance_to_mixed_source(np.diag(scale**2))
    modes, weights, metadata = rank_adapted_source_modes(source, 7)
    assert metadata["coherent_mode_count"] == 1
    assert len(weights) == 1
    with pytest.raises(ValueError): covariance_to_mixed_source(.8*np.diag(scale**2))


def test_symplectic_gate_checks_more_than_unit_determinant():
    validate_matrix(np.diag([1e5, 1e-5, 1e5, 1e-5]))
    with pytest.raises(ValueError): validate_matrix(np.diag([2., 1., 1., .5]))
    with pytest.raises(ValueError): validate_matrix(np.full((4, 4), np.nan))


def test_detector_renderer_refuses_cpu_fallback(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="GPU"):
        render_covariance_gaussian_modes(coupled_covariance(), np.eye(4), DetectorGeometry(), 2e6)
