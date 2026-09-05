from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"examples/lens_inversion"))
from neoarm_last_lens_branch_search import (
    ProfiledResponse, linear_response_design, search_response_branches, distinct_solution,
)
from neoarm_last_lens_tomography import predict_response_parameters

jax.config.update("jax_enable_x64", True)


def response_case():
    t = jnp.linspace(-1., 1., 13)
    nonlinear = jnp.array([jnp.log(12.), -1.5, -.9])
    coefficients = jnp.asarray(np.random.default_rng(723).normal(size=16))
    truth = jnp.r_[nonlinear, coefficients]
    data = predict_response_parameters(truth, t)
    random = np.random.default_rng(712).normal(size=(13, 8, 8))
    covariance = (random@random.transpose(0, 2, 1)+np.eye(8))*1e-6
    cholesky = np.linalg.cholesky(covariance)
    return t, truth, data, cholesky


def test_linear_design_and_correlated_qr_reproduce_full_response():
    t, truth, data, chol = response_case()
    np.testing.assert_allclose(jnp.einsum("nik,k->ni", linear_response_design(truth[:3], t), truth[3:]),
                               data.reshape(len(t), 8), rtol=1e-12, atol=1e-12)
    problem = ProfiledResponse(t, data, chol)
    fitted, residual = problem.solve(truth[:3])
    np.testing.assert_allclose(fitted, truth[3:], rtol=2e-12, atol=2e-12)
    assert np.max(np.abs(residual)) < 1e-8


def test_profiled_metric_matches_direct_whitened_least_squares():
    t, truth, data, chol = response_case()
    shifted = truth[:3]+jnp.array([.1, .08, -.05])
    problem = ProfiledResponse(t, data, chol)
    matrix = np.asarray(problem.whiten(linear_response_design(shifted, t))).reshape(-1, 16)
    expected = np.linalg.lstsq(matrix, np.asarray(problem.whitened_observed), rcond=None)[0]
    actual, residual = problem.solve(shifted)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(residual, matrix@expected-np.asarray(problem.whitened_observed), atol=1e-7)
    jacobian = np.asarray(jax.jacfwd(problem.residual)(shifted))
    eps = 1e-5
    finite = np.stack([(np.asarray(problem.residual(shifted+jnp.eye(3)[i]*eps))-
                       np.asarray(problem.residual(shifted-jnp.eye(3)[i]*eps)))/(2*eps) for i in range(3)], axis=-1)
    np.testing.assert_allclose(jacobian, finite, rtol=2e-6, atol=1e-5)


def test_local_profiled_recovery_does_not_need_upstream_seed():
    t, truth, data, chol = response_case()
    rows = search_response_branches(ProfiledResponse(t, data, chol), q_centres=[-1.5],
        ratios=[8.], slope_seeds=[-.7], max_nfev=100)
    assert len(rows) == 1 and rows[0]["optimizer_success"]
    np.testing.assert_allclose(rows[0]["parameters"], truth, rtol=1e-7, atol=1e-7)


def test_geometry_alternative_counted_when_excitation_law_matches():
    best = np.array([np.log(16.), -1.5, -.9])
    assert distinct_solution([np.log(1.4), -1.5, -.9], best)
    assert distinct_solution([np.log(16.), 1.5, -.9], best)
    assert not distinct_solution(best+1e-6, best)


@pytest.mark.parametrize("bad", ["shape", "upper", "negative", "nan"])
def test_invalid_covariance_contract_rejected(bad):
    t, _, data, chol = response_case()
    if bad == "shape": chol = chol[:, :7, :7]
    elif bad == "upper": chol[0, 0, 1] = 1.
    elif bad == "negative": chol[0, 0, 0] = -1.
    else: chol[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        ProfiledResponse(t, data, chol)
