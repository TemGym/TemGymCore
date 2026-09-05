import jax
import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.projector_response_tomography import (
    analyser_rows, pair_matrix, predict_response_pair, reconstruct_response_pair,
    reconstruct_map_scan,
)

jax.config.update("jax_enable_x64", True)
P = jnp.array([np.log(.02), np.log(.315), -1.5, -1.])
T = jnp.array([-1., 1.])
STEPS = jnp.array([5e-9, 5e-9, 1e-3, 1e-3])
PIXEL = 55e-6


def sample_maps():
    # Generic astigmatic maps deliberately not restricted to the round fit.
    first = np.array([[8., .3, .1, .03], [.2, 5., -.02, .15],
                      [3., -.2, .16, .01], [-.5, 1., -.02, .25]])
    return np.stack([first, first @ np.diag([1.2, .8, 1/1.2, 1/.8])])


def test_ray_jacobian_matches_closed_form_and_pair_inversion_with_units():
    q = float(P[2]+P[3]*T[0]); length, gap = np.exp(np.asarray(P[:2]))
    c, s = np.cos(-q), np.sin(-q)
    rotation = np.array([[c, -s], [s, c]])
    a = np.cos(q)-gap/length*q*np.sin(q)
    b = length*np.sinc(q/np.pi)+gap*np.cos(q)
    np.testing.assert_allclose(analyser_rows(P, T[0]), np.c_[a*rotation, b*rotation],
                               rtol=1e-13, atol=1e-13)
    for matrix in sample_maps():
        y = predict_response_pair(matrix, P, T, STEPS, PIXEL)
        actual = jax.jit(reconstruct_response_pair)(y, P, T, STEPS, PIXEL)
        np.testing.assert_allclose(actual, matrix, rtol=1e-12, atol=1e-12)


def test_shared_analyser_uncertainty_correlates_upstream_settings():
    maps = sample_maps()
    y = jax.vmap(predict_response_pair, in_axes=(0, None, None, None, None))(
        maps, P, T, STEPS, PIXEL)
    result = reconstruct_map_scan(y, P, T, STEPS, PIXEL, np.eye(32)*1e-8,
        np.diag([.001, .002, .0001, .0002])**2, calibration_data_independent=True)
    np.testing.assert_allclose(result.maps, maps, rtol=1e-12, atol=1e-12)
    assert np.linalg.norm(result.shared_analyser_covariance[:16, 16:]) > 0
    np.testing.assert_array_equal(result.measurement_covariance[:16, 16:], 0.)
    np.testing.assert_allclose(result.joint_covariance,
        result.measurement_covariance+result.shared_analyser_covariance)
    assert np.linalg.eigvalsh(result.joint_covariance).min() > -1e-9


def test_correlated_calibration_and_responses_include_cross_terms():
    maps = sample_maps()[:1]
    y = predict_response_pair(maps[0], P, T, STEPS, PIXEL)[None]
    rng = np.random.default_rng(9)
    factor = rng.normal(size=(20, 25))*1e-5
    covariance = factor@factor.T
    result = reconstruct_map_scan(y, P, T, STEPS, PIXEL, covariance[:16, :16],
        covariance[16:, 16:], calibration_data_independent=False,
        response_analyser_cross_covariance=covariance[:16, 16:])
    def recover(state):
        return reconstruct_response_pair(state[:16].reshape(2,2,4), state[16:],
                                         T, STEPS, PIXEL).ravel()
    derivative = np.asarray(jax.jacfwd(recover)(jnp.r_[y.ravel(), P]))
    np.testing.assert_allclose(result.joint_covariance, derivative@covariance@derivative.T,
                               rtol=1e-11, atol=1e-12)
    assert np.linalg.norm(result.response_analyser_cross_term) > 0


def test_missing_dependence_and_singular_pair_are_rejected():
    y = predict_response_pair(sample_maps()[0], P, T, STEPS, PIXEL)[None]
    with pytest.raises(ValueError, match="cross-covariance"):
        reconstruct_map_scan(y, P, T, STEPS, PIXEL, np.eye(16), np.eye(4),
                             calibration_data_independent=False)
    with pytest.raises(ValueError, match="singular"):
        reconstruct_map_scan(y, P, jnp.array([.1,.1]), STEPS, PIXEL, np.eye(16), np.eye(4),
                             calibration_data_independent=True)
    # Opposite equal-magnitude excitations rotate the same radial row and do
    # not supply a second analyser despite having different control values.
    symmetric = P.at[2].set(0.)
    with pytest.raises(ValueError, match="singular"):
        reconstruct_map_scan(y, symmetric, T, STEPS, PIXEL, np.eye(16), np.eye(4),
                             calibration_data_independent=True)
    with pytest.raises(ValueError, match="positive semidefinite"):
        reconstruct_map_scan(y, P, T, STEPS, PIXEL, -np.eye(16), np.eye(4),
                             calibration_data_independent=True)


def test_duplicate_scans_do_not_average_away_shared_analyser_error():
    y = predict_response_pair(sample_maps()[0], P, T, STEPS, PIXEL)
    scan = np.stack([y, y])
    result = reconstruct_map_scan(scan, P, T, STEPS, PIXEL, np.zeros((32,32)),
        np.diag([.001,.002,.0001,.0002])**2, calibration_data_independent=True)
    average = np.c_[np.eye(16), np.eye(16)]/2
    np.testing.assert_allclose(average@result.joint_covariance@average.T,
                              result.joint_covariance[:16, :16], rtol=1e-12)


def test_input_validation_and_batched_derivatives():
    y = predict_response_pair(sample_maps()[0], P, T, STEPS, PIXEL)[None]
    with pytest.raises(ValueError, match="Positive physical"):
        reconstruct_map_scan(y, P, T, STEPS, 0., np.eye(16), np.eye(4),
                             calibration_data_independent=True)
    derivative = jax.jacfwd(pair_matrix)(P, T)
    assert derivative.shape == (4,4,4)
    assert np.all(np.isfinite(derivative))


def test_small_nonlinear_calibration_perturbations_match_shared_covariance():
    maps = sample_maps()
    y = jax.vmap(predict_response_pair, in_axes=(0, None, None, None, None))(
        maps, P, T, STEPS, PIXEL)
    sigma = np.array([2e-5,3e-5,1e-5,1e-5])
    result = reconstruct_map_scan(y, P, T, STEPS, PIXEL, np.zeros((32,32)),
        np.diag(sigma**2), calibration_data_independent=True)
    def recover(cal):
        return jax.vmap(reconstruct_response_pair, in_axes=(0,None,None,None,None))(
            y, cal, T, STEPS, PIXEL).ravel()
    # Symmetric exact nonlinear perturbations avoid stochastic test tolerances.
    pairs = jax.jit(jax.vmap(recover))(P+np.r_[np.diag(sigma),-np.diag(sigma)])
    finite_factor = (np.asarray(pairs[:4])-np.asarray(pairs[4:])).T/2
    np.testing.assert_allclose(finite_factor@finite_factor.T,
        result.shared_analyser_covariance, rtol=2e-6, atol=1e-14)


def test_common_position_tilt_and_pixel_gains_have_exact_physical_length_gauge():
    # Actual input and detector gains can be absorbed into L,d and a different
    # round magnetic upstream map. Only external calibration fixes this scale.
    position_gain, tilt_gain, pixel_gain = 1.003, .996, 1.002
    length_gain = position_gain*tilt_gain/pixel_gain**2
    j = np.array([[0.,-1.],[1.,0.]])
    radial = np.array([[3.,.04],[2.,1.08/3.]])  # determinant exactly one
    shear = np.eye(4); shear[2:,:2] = 800.*j
    upstream = np.kron(radial, np.eye(2))@shear
    gains = np.array([position_gain]*2+[tilt_gain]*2)
    distorted = predict_response_pair(upstream, P, T, STEPS*gains, PIXEL*pixel_gain)
    effective_upstream = upstream*gains[None,:]/pixel_gain
    effective_upstream[2:] /= length_gain
    effective_parameters = P.at[:2].add(np.log(length_gain))
    predicted = predict_response_pair(effective_upstream, effective_parameters, T, STEPS, PIXEL)
    np.testing.assert_allclose(predicted, distorted, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(effective_upstream),1.,rtol=1e-12)
    omega = np.block([[np.zeros((2,2)),np.eye(2)],[-np.eye(2),np.zeros((2,2))]])
    expected = omega.copy(); expected[:2,:2] = 2*800.*position_gain/tilt_gain*j
    np.testing.assert_allclose(effective_upstream.T@omega@effective_upstream, expected,
                               rtol=1e-12, atol=1e-10)
