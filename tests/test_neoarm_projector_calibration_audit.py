"""Checks of the shared-calibration uncertainty audit, without saved truth.

The numerical fixture is an arbitrary round three-stage column. It does not
load generating optics, image reports, or a calibration fitted from images.
"""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))

from neoarm_projector_calibration_audit import (
    compensation, joint_covariance, svd_information, LENGTH_INDICES,
)
from neoarm_il2_extension_design import ThreeProjectorResponse
from neoarm_last_lens_physical_fit import RoundLensResponse
from temgym_core.physical_nuisance import GaussianPriorFactor
from temgym_core.round_calibration import GAIN_NAMES, calibrated_position_response


@pytest.fixture(scope='module')
def calibration_problem():
    model = ThreeProjectorResponse(
        RoundLensResponse((5e-9, 5e-9, 1e-3, 1e-3), 55e-6))
    parameters = jnp.array([
        np.log(.023), np.log(.27), -1.3, -1.15,
        np.log(.027), np.log(.047), 1.08, .43,
        np.log(.032), np.log(.069), .81, .39,
        2.1, .18, .012, .37, .007,
    ])
    controls = jnp.array([[a, b, c] for a in (-.8, .1, .9)
                         for b in (-.3, .1, .5) for c in (-.4, 0., .4)])
    # A correlated external calibration is deliberately used: the latent
    # coordinate order need not equal the physical gain-name order.
    factor = np.array([[.002, 0., 0.], [.0005, .003, 0.], [-.0002, .0004, .001]])
    prior = GaussianPriorFactor.from_covariance(
        'external_scales', GAIN_NAMES, np.zeros(3), factor@factor.T,
        provenance='Arbitrary test covariance, not instrument metrology.')
    prediction = lambda p, z: calibrated_position_response(
        model.predict(p, controls), prior.values(z)).reshape(-1)
    full = jnp.r_[parameters, jnp.zeros(3)]
    scale = np.linspace(.013, .047, len(controls)*8)
    jac = np.asarray(jax.jit(jax.jacfwd(
        lambda q: prediction(q[:17], q[17:])/scale))(full))
    rank, _, statistical_covariance = svd_information(jac[:, :17])
    assert rank == 17
    derivative = np.asarray(jax.jacfwd(
        lambda z: compensation(parameters, prior.values(z)))(jnp.zeros(3)))
    return model, parameters, controls, prior, jac, statistical_covariance, derivative


def test_actual_gain_compensation_direction_and_exact_finite_family(calibration_problem):
    model, p, controls, *_ = calibration_problem
    gains = jnp.log(jnp.array([1.04, .97, 1.008]))
    moved = compensation(p, gains)
    original = model.predict(p, controls)
    np.testing.assert_allclose(
        calibrated_position_response(model.predict(moved, controls), gains),
        original, rtol=2e-12, atol=3e-12)
    # Actual gains at fixed measured response require the reciprocal geometry
    # scale, unlike the bias caused by completely ignoring those gains.
    expected_shift = -gains[0]-gains[1]+2*gains[2]
    np.testing.assert_allclose(moved[LENGTH_INDICES]-p[LENGTH_INDICES],
                               expected_shift, atol=1e-14)
    excitation_indices = np.array([2, 3, 6, 7, 10, 11])
    np.testing.assert_array_equal(moved[excitation_indices], p[excitation_indices])
    wrong_direction = calibrated_position_response(
        model.predict(compensation(p, -gains), controls), gains)
    assert np.max(np.abs(np.asarray(wrong_direction-original))) > .01


def test_three_exact_data_null_directions_and_one_global_prior(calibration_problem):
    _, _, _, _, jac, _, derivative = calibration_problem
    # Columns [d theta/dz; I] are data nulls, rather than three independently
    # determined physical calibrations. A single shared prior makes them proper.
    family_tangent = np.vstack((derivative, np.eye(3)))
    np.testing.assert_allclose(jac@family_tangent, 0., atol=3e-11)
    rank, singular, covariance = svd_information(jac)
    assert rank == 17 and covariance is None
    assert singular[-3:].max() < singular[0]*1e-10
    prior_rows = np.column_stack((np.zeros((3, 17)), np.eye(3)))
    augmented = np.vstack((jac, prior_rows))
    assert augmented.shape == (len(jac)+3, 20)
    assert svd_information(augmented)[0] == 20


def test_analytic_joint_covariance_matches_full_augmented_likelihood(calibration_problem):
    _, _, _, _, jac, statistical_covariance, derivative = calibration_problem
    augmented = np.vstack((jac, np.column_stack((np.zeros((3, 17)), np.eye(3)))))
    rank, _, numeric = svd_information(augmented)
    analytic = joint_covariance(statistical_covariance, derivative)
    assert rank == 20
    np.testing.assert_allclose(analytic, numeric, rtol=3e-9, atol=3e-11)
    np.testing.assert_array_equal(analytic[17:, 17:], np.eye(3))
    np.testing.assert_array_equal(analytic[:17, 17:], derivative)
    assert np.linalg.eigvalsh(analytic).min() > 0


def test_correlated_physical_gain_floor_and_standard_latent_units(calibration_problem):
    _, p, _, prior, _, _, derivative = calibration_problem
    physical_derivative = np.asarray(jax.jacfwd(lambda g: compensation(p, g))(jnp.zeros(3)))
    np.testing.assert_allclose(derivative, physical_derivative@prior.factor, atol=2e-16)
    sensitivity = np.array([-1., -1., 2.])
    expected_variance = sensitivity@prior.covariance@sensitivity
    np.testing.assert_allclose(np.sum(derivative[LENGTH_INDICES]**2, axis=1),
                               expected_variance, rtol=2e-14)
    # B is d(theta)/d(standard latent), NOT d(theta)/d(physical log gain).
    assert np.linalg.norm(derivative-physical_derivative) > 1.


def test_shared_calibration_cancels_only_with_optical_cross_covariance(calibration_problem):
    model, p, _, prior, _, statistical_covariance, derivative = calibration_problem
    held_controls = jnp.array([[.37, -.18, .17], [-.51, .31, -.22]])
    predict = lambda q: calibrated_position_response(
        model.predict(q[:17], held_controls), prior.values(q[17:])).reshape(-1)
    jac = np.asarray(jax.jit(jax.jacfwd(predict))(jnp.r_[p, jnp.zeros(3)]))
    joint = joint_covariance(statistical_covariance, derivative)
    conditional = jac[:, :17]@statistical_covariance@jac[:, :17].T
    marginalized = jac@joint@jac.T
    np.testing.assert_allclose(marginalized, conditional, rtol=2e-9, atol=1e-13)
    # Dropping optical/gain cross blocks spuriously adds prediction uncertainty.
    incorrect = joint.copy()
    incorrect[:17, 17:] = 0.; incorrect[17:, :17] = 0.
    assert np.linalg.norm(jac@incorrect@jac.T-conditional) > 1e-5


def test_independent_repeats_do_not_repeat_the_shared_prior(calibration_problem):
    _, _, _, _, jac, statistical_covariance, derivative = calibration_problem
    prior_rows = np.column_stack((np.zeros((3, 17)), np.eye(3)))
    # sqrt(R)*J is exactly the information of R independent identical scans.
    repeats = 12
    numeric = svd_information(np.vstack((np.sqrt(repeats)*jac, prior_rows)))[2]
    expected = joint_covariance(statistical_covariance/repeats, derivative)
    np.testing.assert_allclose(numeric, expected, rtol=5e-9, atol=3e-11)
    floor = derivative@derivative.T
    np.testing.assert_allclose(numeric[:17, :17]-statistical_covariance/repeats,
                               floor, rtol=5e-7, atol=3e-11)
    incorrect = svd_information(np.sqrt(repeats)*np.vstack((jac, prior_rows)))[2]
    np.testing.assert_allclose(incorrect[17:, 17:], np.eye(3)/repeats, atol=3e-11)
    assert not np.allclose(incorrect[17:, 17:], expected[17:, 17:])


def test_compensation_jit_vmap_and_first_second_derivatives(calibration_problem):
    _, p, *_ = calibration_problem
    function = lambda gains: compensation(p, gains)
    zero = jnp.zeros(3)
    derivative = np.asarray(jax.jit(jax.jacfwd(function))(zero))
    assert np.all(np.isfinite(jax.jit(jax.jacfwd(jax.jacfwd(function)))(zero)))
    for index in range(3):
        delta = np.eye(3)[index]*1e-6
        finite = (function(delta)-function(-delta))/(2e-6)
        np.testing.assert_allclose(derivative[:, index], finite, rtol=1e-7, atol=4e-10)
    batched = jax.jit(jax.vmap(function))(jnp.array([[0., 0., 0.], [.01, -.005, .003]]))
    np.testing.assert_allclose(batched[0], p, atol=1e-14)
