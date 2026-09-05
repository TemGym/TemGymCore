import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_condenser_scaled_cl1_profile import (
    scaled_profiled_least_squares, original_unit_history, with_private_solver,
)
from temgym_core.profiled_least_squares import profiled_least_squares


def test_large_linear_displacement_recovers_with_same_iteration_budget():
    residual = lambda p: p-jnp.array([1., 10000.])
    old = profiled_least_squares(residual, [0., 0.], lambda p: True,
        n_outer=1, max_iterations=3, max_inner_iterations=12)
    assert not old['success']
    fit = scaled_profiled_least_squares(residual, [0., 0.], lambda p: True,
        n_outer=1, scale=1e-3, max_iterations=3, max_inner_iterations=12)
    assert fit['success'], fit['reason']
    np.testing.assert_allclose(fit['parameters'], [1., 10000.], atol=1e-8)
    assert fit['normalized_gradient'] < 1e-6
    assert fit['initial_nuisance_history'][0]['chi_square'] == pytest.approx(100000001.)


def test_entire_likelihood_and_prior_keep_their_relative_weights():
    residual = lambda p: jnp.array([p[0]-2., p[1]-4., p[1]/.5])
    fit = scaled_profiled_least_squares(residual, [0., 0.], lambda p: True,
        n_outer=1, scale=1e-3, max_iterations=5, max_inner_iterations=10)
    assert fit['success'], fit['reason']
    np.testing.assert_allclose(fit['parameters'], [2., .8], atol=1e-8)
    assert fit['residual']@fit['residual'] == pytest.approx(12.8)
    assert fit['history'][-1]['chi_square'] == pytest.approx(12.8)
    np.testing.assert_allclose(fit['jacobian'], [[1., 0.], [0., 1.], [0., 2.]])
    assert fit['unit_conversion']['internal_gradient_tolerance'] == pytest.approx(1e-9)
    assert fit['unit_conversion']['original_gradient_tolerance'] == 1e-6


def test_every_nested_history_cost_and_gradient_is_unscaled():
    data = dict(chi_square=4e-6, whitened_profile_gradient=2e-3,
        profile_hessian_min=.9, trials=[dict(chi_square_change=-1e-6,
            nuisance_history=[dict(normalized_nuisance_gradient=3e-4,
                whitened_nuisance_gradient=5e-4, hessian_step_shift=.01)])])
    value = original_unit_history(data, 1e-3)
    assert value['chi_square'] == 4. and value['whitened_profile_gradient'] == 2.
    assert value['profile_hessian_min'] == .9
    assert value['trials'][0]['chi_square_change'] == -1.
    inner = value['trials'][0]['nuisance_history'][0]
    assert inner['normalized_nuisance_gradient'] == pytest.approx(.3)
    assert inner['whitened_nuisance_gradient'] == .5 and inner['hessian_step_shift'] == .01
    assert data['chi_square'] == 4e-6


def test_solver_callback_does_not_monkeypatch_original_module_globals():
    def original(value):
        return profiled_least_squares(value)
    previous = original.__globals__['profiled_least_squares']
    injected = with_private_solver(original, lambda x: x+3)
    assert injected(2) == 5
    assert original.__globals__['profiled_least_squares'] is previous


@pytest.mark.parametrize('scale', [0., -1., np.nan])
def test_invalid_uniform_scales_rejected(scale):
    with pytest.raises(ValueError):
        scaled_profiled_least_squares(lambda p: p, [1., 1.], lambda p: True,
            scale=scale, n_outer=1)
