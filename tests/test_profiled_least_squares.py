import jax
import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.profiled_least_squares import (
    profile_nuisance, profiled_least_squares, _Problem, _inner, _profile_derivatives,
)


def test_inner_holds_outer_fixed_and_preserves_correlated_likelihood():
    root = jnp.linalg.cholesky(jnp.array([[2., .4], [.4, 1.]]))
    residual = lambda p: jnp.linalg.solve(root, jnp.array([p[0]+p[1]-3., p[1]-1.]))
    fit = profile_nuisance(residual, [.7, 0.], lambda p: True, n_outer=1)
    assert fit['success'], fit['reason']
    assert fit['parameters'][0] == .7
    inverse = np.linalg.inv(np.asarray(root@root.T))
    a = np.ones(2)
    expected = -(a@inverse@np.array([.7-3., -1.]))/(a@inverse@a)
    assert fit['parameters'][1] == pytest.approx(expected, abs=1e-10)
    np.testing.assert_allclose(fit['residual'], residual(fit['parameters']))


def test_full_nonlinear_profile_fit_recovers_curved_valley():
    residual = lambda p: jnp.array([30*(p[1]-p[0]**2), p[0]-1.])
    fit = profiled_least_squares(residual, [.5, .24], lambda p: np.max(abs(p)) < 3,
        n_outer=1, max_iterations=12)
    assert fit['success'], fit['reason']
    np.testing.assert_allclose(fit['parameters'], [1., 1.], atol=1e-7)
    assert not fit['priors_added'] and not fit['rank_directions_discarded']
    assert all(row['nuisance_stationary'] for row in fit['history'])


def test_exact_observed_schur_derivative_matches_known_profile():
    residual = lambda p: jnp.array([p[1]-p[0]**2, 2*(p[1]-1.), .2*p[0]])
    x = .8
    p = np.array([x, (x*x+4)/5])
    problem = _Problem(residual, lambda p: True)
    inner = _inner(problem, p, 1, 0, 1e-6)
    assert inner['success']
    derivative = _profile_derivatives(problem, inner, 1)
    factor = derivative['outer_factor'][0, 0]
    assert derivative['hessian'][0, 0]/factor**2 == pytest.approx(4.8*x*x-1.56, rel=1e-12)
    assert derivative['gradient'][0]/factor == pytest.approx(1.6*x*(x*x-1)+.04*x, rel=1e-12)
    expected = np.asarray(jax.jacfwd(lambda z: residual(jnp.array([z, (z*z+4)/5])))(x))
    np.testing.assert_allclose(derivative['profile_jacobian'][:, 0], expected, atol=1e-12)


def test_unconverged_nuisance_blocks_outer_steps_and_implicit_derivatives():
    residual = lambda p: jnp.array([p[0]-1., p[1]-10.])
    fit = profiled_least_squares(residual, [0., 0.], lambda p: True,
        n_outer=1, max_inner_iterations=0)
    assert not fit['success'] and not fit['nuisance_stationary']
    assert fit['history'] == [] and fit['parameters'][0] == 0.
    problem = _Problem(residual, lambda p: True)
    failed = _inner(problem, np.zeros(2), 1, 0, 1e-6)
    with pytest.raises(ValueError, match='stationary nuisance'):
        _profile_derivatives(problem, failed, 1)


def test_boundary_inner_stationarity_is_not_falsely_certified():
    fit = profile_nuisance(lambda p: jnp.array([p[0], p[1]+1.]), [0., 0.],
        lambda p: p[1] >= 0, n_outer=1)
    assert not fit['success'] and not fit['nuisance_stationary']
    assert fit['parameters'][1] == 0
    assert 'Armijo' in fit['reason']


def test_rank_deficiency_reported_without_discarding_outer_direction():
    fit = profiled_least_squares(lambda p: jnp.array([p[0]+p[1]]), [0., 0.],
        lambda p: True, n_outer=1)
    assert not fit['success'] and 'rank-deficient' in fit['reason']
    assert not fit['rank_directions_discarded']


def test_outer_iteration_limit_is_honest():
    fit = profiled_least_squares(lambda p: p-jnp.array([1., 0.]), [0., 0.],
        lambda p: True, n_outer=1, max_iterations=0)
    assert not fit['success'] and fit['nuisance_stationary']
    assert fit['reason'] == 'outer iteration budget exhausted'


@pytest.mark.parametrize('options', [dict(n_outer=0), dict(n_outer=2),
    dict(n_outer=1, max_iterations=-1), dict(n_outer=1, gradient_tolerance=np.nan)])
def test_invalid_options_fail(options):
    with pytest.raises(ValueError):
        profile_nuisance(lambda p: p, [0., 0.], lambda p: True, **options)
