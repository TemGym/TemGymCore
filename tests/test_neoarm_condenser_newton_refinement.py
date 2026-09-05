from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_condenser_newton_refinement import refine_training


def test_exact_hessian_refinement_in_scaled_coordinates():
    matrix = jnp.array([[1e5, .003], [3e5, -.002], [2e5, .007]])
    truth = jnp.array([.002, 30.])
    residual = lambda p: matrix@(p-truth)
    fit = refine_training(residual, [.002001, 30.2], lambda p: True, max_iterations=10)
    assert fit['success'], fit['reason']
    np.testing.assert_allclose(fit['parameters'], truth, atol=1e-10)
    for row in fit['history']:
        np.testing.assert_allclose(row['local_hessian_eigenvalues'], 1., rtol=2e-10)
        assert row['information_whitening_error'] < 1e-9


def test_residual_curvature_is_included_not_replaced_by_gauss_newton():
    # At x=1.01, H/(J.TJ) differs from1 because residual0 is nonlinear.
    residual = lambda p: jnp.array([p[0]**2-1., .3*(p[0]-1.)])
    fit = refine_training(residual, [1.01], lambda p: p[0] > .2, max_iterations=10)
    assert fit['success'], fit['reason']
    assert abs(fit['history'][0]['local_hessian_eigenvalues'][0]-1.) > .001
    np.testing.assert_allclose(fit['parameters'], [1.], atol=1e-8)


def test_failed_boundary_and_iteration_budget_do_not_report_success():
    residual = lambda p: p-jnp.array([2.])
    fit = refine_training(residual, [0.], lambda p: p[0] <= .25, max_iterations=1)
    assert not fit['success']
    assert fit['parameters'][0] <= .25
    assert fit['history'][-1]['normalized_gradient'] > 1.
