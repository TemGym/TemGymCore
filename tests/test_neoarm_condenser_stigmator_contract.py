from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_condenser_stigmator_contract import (
    InternalStigmatorModel, round_source, quadrupole, drift, local_information,
    SOURCE_SCALES, APERTURE_RADII_M, GROUPED,
)
from temgym_core.immersed_objective import uniform_field_canonical_matrix


@pytest.fixture(scope='module')
def parameters():
    return jnp.r_[jnp.zeros(12), jnp.array([0., 0., .03, .04]), jnp.array([0., .7, .4])]


@pytest.mark.parametrize('location', ['after_cl1', 'after_cl3'])
def test_pre_stigmator_polarity_is_discrete_invisible_family(location, parameters):
    model = InternalStigmatorModel(location)
    function = jax.jit(model.observables)
    original = np.asarray(function(parameters))
    for index in range(3):
        changed = np.asarray(function(model.polarity_reversal(parameters, index)))
        difference = np.max(abs(changed-original))
        if index < model.insert_before:
            assert difference < 1e-9
        else:
            assert difference > .1


@pytest.mark.parametrize('location', ['after_cl1', 'after_cl3'])
def test_local_geometry_rank_does_not_remove_discrete_polarity_family(location, parameters):
    model = InternalStigmatorModel(location)
    jac = jax.jit(jax.jacfwd(model.observables))(parameters)
    information = local_information(jac)
    assert information['full_rank'] == 19
    assert information['nuisance_rank'] == 7
    assert information['profiled_optical_rank'] == 12
    assert not information['measurement_noise_or_uncertainty_used']


def test_internal_quad_exposes_previously_invisible_source_covariance(parameters):
    model = InternalStigmatorModel('after_cl3')
    derivative = jax.jit(jax.jacfwd(model.observables))
    without_quad = parameters.at[-2:].set(0.)
    assert np.linalg.norm(np.asarray(derivative(without_quad))[:, 15]) < 1e-9
    assert np.linalg.norm(np.asarray(derivative(parameters))[:, 15]) > 1e-4


def test_circular_aperture_counts_reduce_to_round_gaussian_integral(parameters):
    model = InternalStigmatorModel('after_cl3')
    observables = np.asarray(jax.jit(lambda p: model.observables(
        p, include_aperture=True))(parameters)).reshape(27, 12)
    covariance = model.decode(parameters)[1]
    maps = jax.jit(jax.vmap(lambda u: model.maps(parameters, u)[1]))(model.controls)
    aperture_cov = maps@covariance@maps.swapaxes(-1, -2)
    variance = np.asarray(aperture_cov[:, 0, 0])
    expected = np.log(-np.expm1(-np.asarray(APERTURE_RADII_M)[None, :]**2/(2*variance[:, None])))
    np.testing.assert_allclose(observables[:, 9:], expected, rtol=3e-12, atol=2e-12)
    assert np.all(np.diff(observables[:, 9:], axis=1) >= 0)


def test_quad_and_column_maps_are_canonical_symplectic(parameters):
    omega = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]])
    q = np.asarray(quadrupole(jnp.array([21., 12.])))
    np.testing.assert_allclose(q.T@omega@q, omega, atol=0.)
    model = InternalStigmatorModel('after_cl1')
    matrix = np.asarray(model.maps(parameters, model.controls[13])[0])
    np.testing.assert_allclose(matrix.T@omega@matrix, omega, atol=2e-9)


def test_internal_quad_cannot_be_absorbed_as_one_fixed_entrance_covariance(parameters):
    model = InternalStigmatorModel('after_cl1')
    optical, covariance, z_q, quad = model.decode(parameters)
    centre, log_length, bias, slope = optical[0]
    length = jnp.exp(log_length)
    effective = []
    for control in (-.3, .7):
        native = uniform_field_canonical_matrix((bias+slope*control)/length, length)
        upstream = drift(z_q-centre-length/2)@native[jnp.ix_(GROUPED, GROUPED)]@drift(centre-length/2-model.source_z_m)
        pullback = jnp.linalg.solve(upstream, quad@upstream)
        effective.append(np.asarray(pullback@covariance@pullback.T)/np.outer(SOURCE_SCALES, SOURCE_SCALES))
    assert np.linalg.norm(effective[0]-effective[1]) > .1


def test_jit_first_second_derivatives_and_fixed_location_contract(parameters):
    model = InternalStigmatorModel('after_cl3')
    # One canonical matrix suffices for the second-derivative transport check.
    function = lambda p: model.maps(p, model.controls[9])[0]
    jacobian = np.asarray(jax.jit(jax.jacfwd(function))(parameters))
    assert np.all(np.isfinite(jax.jit(jax.jacfwd(jax.jacfwd(function)))(parameters)))
    for index in (0, 2, 6, 16, 17, 18):
        step = np.eye(19)[index]*1e-6
        finite = (function(parameters+step)-function(parameters-step))/(2e-6)
        np.testing.assert_allclose(jacobian[..., index], finite, rtol=2e-6, atol=2e-5)
    with pytest.raises(ValueError):
        InternalStigmatorModel('unknown_hardware_position')
    with pytest.raises(ValueError):
        model.decode(jnp.zeros(18))
