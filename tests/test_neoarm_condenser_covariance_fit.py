"""Visible-source quotient, quantum completion and conditional fit guards."""
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

EXAMPLES = Path(__file__).resolve().parents[1]/"examples/lens_inversion"
if str(EXAMPLES) not in sys.path: sys.path.insert(0, str(EXAMPLES))

from neoarm_condenser_covariance_fit import (
    BASIS, INVISIBLE, OMEGA, SOURCE_SCALE, visible_coordinates, visible_covariance,
    quantum_completion, geometry_gaps, VisibleSourceModel, svd_gauss_newton,
)
from test_neoarm_condenser_image_recovery import matrix_fixture

jax.config.update("jax_enable_x64", True)


def test_nine_basis_is_orthonormal_and_omits_only_reported_gauge():
    np.testing.assert_allclose(np.einsum("aij,bij->ab", BASIS, BASIS), np.eye(9), atol=2e-16)
    np.testing.assert_allclose(np.einsum("aij,ij->a", BASIS, INVISIBLE), 0., atol=1e-16)
    rng = np.random.default_rng(18)
    a = rng.normal(size=(4, 4)); c = a+a.T
    physical_units = c*np.outer(SOURCE_SCALE, SOURCE_SCALE)
    visible = np.asarray(visible_covariance(visible_coordinates(physical_units)))/np.outer(SOURCE_SCALE, SOURCE_SCALE)
    gauge = np.sum(c*INVISIBLE)/np.sum(INVISIBLE**2)
    np.testing.assert_allclose(visible+gauge*INVISIBLE, c, atol=1e-15)


def test_quantum_interval_matches_known_isotropic_family():
    # 2 I + g N has uncertainty minimum 1-|g|.
    coordinates = visible_coordinates(2*np.diag(SOURCE_SCALE**2))
    result = quantum_completion(coordinates, interval=True)
    assert result["feasible"]
    np.testing.assert_allclose(result["representative"], 0., atol=1e-9)
    np.testing.assert_allclose(result["feasible_interval"], [-1., 1.], atol=1e-12)
    np.testing.assert_allclose(result["margin"], 1., atol=1e-9)


def test_subquantum_visible_covariance_rejected():
    coordinates = visible_coordinates(.7*np.diag(SOURCE_SCALE**2))
    assert not quantum_completion(coordinates)["feasible"]


def test_profiled_completion_can_recover_nonzero_invisible_gauge():
    # Deliberately rotated/squeezed quantum-valid covariance, then discard its
    # invisible scalar; the optimizer must find some physical completion.
    rng = np.random.default_rng(24)
    a = rng.normal(size=(4, 4)); c = a@a.T+np.eye(4)*2
    c += .3*INVISIBLE
    coordinates = visible_coordinates(c*np.outer(SOURCE_SCALE, SOURCE_SCALE))
    result = quantum_completion(coordinates, interval=True)
    assert result["feasible"] and result["margin"] > 0
    completed = result["representative_covariance"]/np.outer(SOURCE_SCALE, SOURCE_SCALE)
    assert np.linalg.eigvalsh(completed+1j*OMEGA).min() >= 0
    np.testing.assert_allclose(visible_coordinates(result["representative_covariance"]), coordinates, atol=3e-15)


def test_geometry_guard_rejects_aperture_inside_moving_cl3():
    model = matrix_fixture(); quotient = VisibleSourceModel(model)
    p = np.r_[np.zeros(12), visible_coordinates(2*np.diag(SOURCE_SCALE**2)), 0., 0.]
    assert quotient.admissible(p)
    model.aperture_z_m = model.base[4]+.25*np.exp(model.base[5])
    assert geometry_gaps(model, model.base)[2] < 0
    assert not quotient.admissible(p)


def test_visible_prediction_and_gauge_completion_agree():
    model = matrix_fixture(); quotient = VisibleSourceModel(model)
    coordinates = visible_coordinates(2*np.diag(SOURCE_SCALE**2))
    control = jnp.array([[.2, -.1, .4, 0., 0., 0., 0.]])
    p = np.r_[np.zeros(12), coordinates, .2, 0.]
    for g in (-.8, 0., .8):
        c = np.asarray(visible_covariance(coordinates))+g*INVISIBLE*np.outer(SOURCE_SCALE, SOURCE_SCALE)
        maps = np.asarray(model.maps(model.base, control[0]))[:, [0, 2]]
        expected = maps@c@maps.transpose(0, 2, 1)/(55e-6)**2
        actual = maps@np.asarray(visible_covariance(coordinates))@maps.transpose(0, 2, 1)/(55e-6)**2
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-16)
    first = quotient.observables(p, control, control)
    p[-1] = .005
    second = quotient.observables(p, control, control)
    np.testing.assert_allclose(first[0], second[0], atol=1e-14)
    np.testing.assert_allclose(np.asarray(second[1])-first[1], [[-.005, -.005, 0.]], atol=1e-14)


def test_focusing_strength_chart_is_invertible_and_jax_differentiable():
    model = matrix_fixture(); quotient = VisibleSourceModel(model)
    optics = model.base.copy(); optics[1::4] += [.08, -.06, .10]
    optics[2::4] *= [1.03, .97, 1.07]
    p = np.r_[quotient.optical_coordinates(optics), visible_coordinates(2*np.diag(SOURCE_SCALE**2)), .2, 0.]
    np.testing.assert_allclose(quotient.decode(p)[0], optics, atol=1e-15)
    controls = jnp.array([[.2, -.1, .4, 0., 0., 0., 0.]])
    def values(x):
        beam, aperture = quotient.observables(x, controls, controls)
        return jnp.r_[beam.reshape(-1), aperture.reshape(-1)]
    derivative = np.asarray(jax.jit(jax.jacfwd(values))(p))
    step = 1e-5
    reference = np.column_stack([(np.asarray(values(p+step*d))-np.asarray(values(p-step*d)))/(2*step) for d in np.eye(23)])
    np.testing.assert_allclose(derivative, reference, atol=1e-7, rtol=1e-4)


def test_svd_solver_does_not_drop_weak_column_or_use_xtol_success():
    def residual(p): return jnp.array([p[0]-2., (p[1]-3.)*1e-4, .2])
    fit = svd_gauss_newton(residual, np.zeros(2), lambda p: True, max_iterations=40)
    assert fit["success"]
    np.testing.assert_allclose(fit["parameters"], [2., 3.], atol=1e-8)
    rejected = svd_gauss_newton(residual, np.zeros(2), lambda p: np.all(p == 0), max_iterations=5)
    assert not rejected["success"]
    assert rejected["reason"] == "physical feasible line search failed"
