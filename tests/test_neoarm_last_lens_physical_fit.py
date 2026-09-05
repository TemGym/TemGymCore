from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))

from neoarm_last_lens_physical_fit import (
    RoundLensResponse, analyser_coordinates, seed_from_response_parameters,
    PhysicalResponseProblem, local_information,
)
from neoarm_last_lens_tomography import (
    J4, mechanical_shear, predict_response_parameters, signed_position_response,
)
from temgym_core.jax_least_squares import least_squares


@pytest.fixture
def model():
    return RoundLensResponse((5e-9, 7e-9, 1e-3, 8e-4), 55e-6)


@pytest.fixture
def parameters():
    return np.array([np.log(.36), np.log(15.), -1.4, -1.2, 3., .07, -.02, .2, .008])


def test_physical_round_embedding_and_independent_map(model, parameters):
    controls = jnp.linspace(-1., 1., 13)
    predicted = model.predict(parameters, controls)
    full = model.response_parameters(parameters)
    np.testing.assert_allclose(predicted, predict_response_parameters(full, controls), rtol=2e-12, atol=1e-11)
    length, gap = np.exp(np.asarray(analyser_coordinates(parameters))[:2])
    independent = signed_position_response(parameters[2]+parameters[3]*controls, length, gap,
        model.upstream_matrix(parameters))*(np.array(model.input_steps_grouped)/model.pixel_m)
    np.testing.assert_allclose(predicted, independent, rtol=2e-12, atol=1e-11)


def test_symplecticity_magnetic_shear_and_chart_gauge(model, parameters):
    radial = np.asarray(model.radial_matrix(parameters))
    assert np.linalg.det(radial) == pytest.approx(1., abs=1e-13)
    canonical = np.asarray(model.upstream_matrix(parameters, coordinates='canonical'))
    mechanical = np.asarray(model.upstream_matrix(parameters))
    tt = np.asarray(mechanical_shear(parameters[8]/model.balance_length_m))
    # Balance canonical SI units before checking cancellation against zero.
    ell = model.balance_length_m
    scale = np.diag([np.sqrt(ell)]*2+[1/np.sqrt(ell)]*2)
    cc, mm, tt_balanced = [np.linalg.solve(scale, value@scale) for value in (canonical, mechanical, tt)]
    np.testing.assert_allclose(cc.T@J4@cc, J4, atol=3e-13)
    np.testing.assert_allclose(mm.T@J4@mm, tt_balanced.T@J4@tt_balanced, atol=3e-13)
    other = parameters.copy();other[[5, 7]] += np.pi
    np.testing.assert_allclose(model.upstream_matrix(other), mechanical, rtol=2e-13, atol=2e-8)
    # The row chart covers a=0 without an omitted-entry/cofactor singularity.
    at_zero = parameters.copy();at_zero[5] = np.pi/2
    assert np.all(np.isfinite(jax.jacfwd(model.radial_matrix)(at_zero)))
    assert np.linalg.det(np.asarray(model.radial_matrix(at_zero))) == pytest.approx(1., abs=1e-13)


def test_data_projection_is_exact_on_physical_coefficients(model, parameters):
    seed = seed_from_response_parameters(model.response_parameters(parameters), model)
    np.testing.assert_allclose(model.upstream_matrix(seed), model.upstream_matrix(parameters), rtol=2e-12, atol=2e-9)
    np.testing.assert_allclose(analyser_coordinates(seed), analyser_coordinates(parameters), atol=1e-13)
    np.testing.assert_allclose(model.predict(seed, jnp.linspace(-1., 1., 15)),
                               model.predict(parameters, jnp.linspace(-1., 1., 15)), rtol=2e-11, atol=2e-9)


def test_jit_vmap_first_and_second_derivatives(model, parameters):
    controls = jnp.linspace(-1., 1., 9)
    predict = lambda p: model.predict(p, controls)
    jac = np.asarray(jax.jit(jax.jacfwd(predict))(parameters))
    hess = np.asarray(jax.jit(jax.jacfwd(jax.jacfwd(predict)))(parameters))
    assert np.all(np.isfinite(hess))
    for index in range(9):
        increment = np.eye(9)[index]*1e-6
        difference = (predict(parameters+increment)-predict(parameters-increment))/(2e-6)
        np.testing.assert_allclose(jac[..., index], difference, rtol=3e-7, atol=5e-5)
    batch = jax.jit(jax.vmap(predict))(jnp.stack([parameters, parameters]))
    np.testing.assert_allclose(batch[0], predict(parameters), rtol=3e-13, atol=1e-10)


def test_full_covariance_whitening_and_local_rank(model, parameters):
    controls = np.linspace(-1., 1., 19)
    observed = np.asarray(model.predict(parameters, controls))
    factor = np.eye(8)+.13*np.tril(np.ones((8, 8)), -1)
    covariance = np.broadcast_to(.02**2*(factor@factor.T), (len(controls), 8, 8))
    problem = PhysicalResponseProblem(model, controls, observed+.001, covariance)
    expected = np.linalg.solve(np.linalg.cholesky(covariance), np.full((len(controls), 8, 1), -.001))
    np.testing.assert_allclose(problem.residual(parameters), expected.reshape(-1), rtol=1e-7, atol=1e-9)
    info = local_information(problem, parameters)
    assert info['numerical_rank'] == 9
    assert np.linalg.eigvalsh(info['analyser_covariance']).min() > 0


def test_noiseless_fit_from_data_derived_seed(model, parameters):
    controls = np.linspace(-1., 1., 17)
    response = np.asarray(model.predict(parameters, controls))
    cov = np.broadcast_to(.02**2*np.eye(8), (len(controls), 8, 8))
    problem = PhysicalResponseProblem(model, controls, response, cov)
    # An exact coefficient seed is derived from observables for this algebraic
    # test; the production audit independently searches nonlinear branches.
    seed = seed_from_response_parameters(model.response_parameters(parameters), model)
    seed[[0, 1, 2, 3]] += [.01, -.015, .001, -.002]
    norms = np.linalg.norm(problem.jacobian(seed), axis=0)
    result = least_squares(problem.residual, seed, bounds=(seed-1., seed+1.), x_scale=1/norms,
                           max_nfev=250, gtol=1e-8, ftol=1e-13, xtol=1e-13)
    assert result.success
    assert np.max(np.abs(result.residual)) < 1e-6
    np.testing.assert_allclose(analyser_coordinates(result.x), analyser_coordinates(parameters), atol=1e-8)


def test_shared_calibration_scale_gauge_exact(model, parameters):
    # Ignoring shared position/angle/pitch gains rescales both L and gap by
    # sp*st/pp². Calibration uncertainty therefore cannot be beaten by dose.
    sp, st, pp = 1.02, .97, 1.003
    fp, ft = sp/pp, st/pp
    radial = np.asarray(model.radial_matrix(parameters))
    a, b = radial[0];c, d = radial[1]
    ell = model.balance_length_m
    balanced = np.array([[a*fp, b*ft/ell], [ell*c/ft, d/fp]])
    row_norm = np.linalg.norm(balanced[0])
    transformed = parameters.copy()
    transformed[0] += np.log(fp*ft)
    transformed[4] = np.log(row_norm)
    transformed[5] = np.arctan2(balanced[0, 1], balanced[0, 0])
    transformed[6] = np.dot(balanced[0], balanced[1])/row_norm**2
    transformed[8] *= fp/ft
    controls = np.linspace(-1., 1., 11)
    np.testing.assert_allclose(model.predict(transformed, controls),
        model.predict(parameters, controls)*np.array([fp, fp, ft, ft]), rtol=3e-12, atol=2e-9)
    np.testing.assert_allclose(analyser_coordinates(transformed)[:2]-analyser_coordinates(parameters)[:2],
                               np.log(sp*st/pp**2), atol=1e-14)


def test_invalid_calibration_and_covariance_rejected(model, parameters):
    with pytest.raises(ValueError):
        RoundLensResponse((1., 0., 1., 1.), 1.)
    with pytest.raises(ValueError):
        RoundLensResponse((1., 1., 1., 1.), np.nan)
    with pytest.raises(ValueError):
        model.predict(np.zeros(8), np.arange(5))
    with pytest.raises(ValueError):
        model.upstream_matrix(parameters, coordinates='unknown')
    covariance = np.broadcast_to(np.eye(8), (5, 8, 8)).copy()
    covariance[0, 0, 1] = .2
    with pytest.raises(ValueError):
        PhysicalResponseProblem(model, np.arange(5), np.zeros((5, 2, 4)), covariance)
