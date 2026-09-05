import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.feasible_least_squares import (
    svd_gauss_newton, _trust_region_step, _guard_acceleration,
)


def test_small_acceleration_can_be_uphill_and_is_rejected():
    # Like the actual condenser failure: tiny parameter-space curvature can
    # dominate the stiff response direction and reverse descent.
    jac = np.diag([1e6, 1.]); residual = np.ones(2)
    base = np.array([-1e-6, -1.]); acceleration = np.array([8e-6, 0.])
    assert residual@jac@(base+.5*acceleration) > 0
    guarded, reason, ratio = _guard_acceleration(jac, residual, base, acceleration, np.ones(2), 2.)
    assert ratio < 1e-5 and reason == 'non_descent'
    np.testing.assert_array_equal(guarded, base)


def test_descent_with_nonpositive_predicted_reduction_rejects_acceleration():
    jac = np.diag([1., 1e4]); residual = np.array([1., 0.])
    base = np.array([-.1, 0.]); acceleration = np.array([0., 1e-4])
    guarded, reason, _ = _guard_acceleration(jac, residual, base, acceleration, np.ones(2), 1.)
    assert reason == 'nonpositive_model_reduction'
    np.testing.assert_array_equal(guarded, base)


def test_trust_damping_bracket_expands_until_radius_is_satisfied():
    step, damping, expansions = _trust_region_step(np.eye(1), np.ones(1), np.eye(1), np.array([1e9]), 1e-8)
    assert expansions > 0 and damping >= 1e17-1
    assert np.linalg.norm(step) <= 1e-8*(1+1e-14)


def test_scaled_linear_recovery_and_no_invalid_quality_ratios():
    matrix = jnp.array([[1e3, 1e-3], [2e3, -3e-3]])
    truth = jnp.array([.003, 2000.])
    fit = svd_gauss_newton(lambda p: matrix@(p-truth), np.zeros(2),
        lambda p: np.isfinite(p).all(), native_scales=[.01, 1000.], max_iterations=100)
    assert fit['success']
    np.testing.assert_allclose(fit['parameters'], truth, rtol=1e-8, atol=1e-9)
    for row in fit['history']:
        if 'step_quality' in row:
            assert np.isfinite(row['step_quality'])
            assert row['predicted_reduction'] > 0 and row['actual_reduction'] > 0


def test_nonlinear_rosenbrock_recovery():
    residual = lambda p: jnp.array([10*(p[1]-p[0]**2), 1-p[0]])
    fit = svd_gauss_newton(residual, [-1.2, 1.], lambda p: np.all(np.abs(p) < 5), max_iterations=150)
    assert fit['success'], fit['reason']
    np.testing.assert_allclose(fit['parameters'], [1., 1.], atol=3e-6)


def test_physical_line_search_can_shorten_step_without_false_convergence():
    fit = svd_gauss_newton(lambda p: p-jnp.array([2.]), [0.], lambda p: 0 <= p[0] <= 1,
        max_iterations=8)
    assert not fit['success']
    assert 0 <= fit['parameters'][0] <= 1
    assert fit['normalized_gradient'] > .5
    assert 'line search' in fit['reason'] or 'budget' in fit['reason']


def test_infeasible_accelerated_ray_falls_back_to_feasible_base_gn():
    # At the p1<=0 boundary, curvature suggests a positive p1 component for
    # every nonzero step fraction. The unaccelerated step remains feasible.
    residual = lambda p: jnp.array([p[0]-1., p[1]-.01*p[0]**2])
    fit = svd_gauss_newton(residual, [0., 0.], lambda p: p[1] <= 0, max_iterations=1)
    row = fit['history'][0]
    assert row['acceleration_guard'] == 'accepted_for_line_search'
    assert row['accepted_direction'] == 'base_gauss_newton'
    assert row['attempts'][0]['line_searches'][0]['feasible_trials'] == 0
    np.testing.assert_allclose(fit['parameters'], [1., 0.], atol=1e-14)


def test_iteration_limit_is_not_success_and_invalid_scales_rejected():
    fit = svd_gauss_newton(lambda p: p-jnp.array([100.]), [0.], lambda p: True, max_iterations=1)
    assert not fit['success'] and fit['reason'] == 'iteration budget exhausted'
    for scales in ([0.], [-1.], [np.nan], [1., 2.]):
        with pytest.raises(ValueError, match='scales'):
            svd_gauss_newton(lambda p: p, [1.], lambda p: True, native_scales=scales)


def test_exact_rank_deficiency_does_not_discard_nonzero_directions():
    fit = svd_gauss_newton(lambda p: jnp.array([p[0]+p[1]-3., 2*(p[0]+p[1]-3.)]),
        [0., 0.], lambda p: True, max_iterations=20)
    assert fit['success']
    assert sum(fit['parameters']) == pytest.approx(3., abs=1e-8)
