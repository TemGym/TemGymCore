import jax
import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.immersed_objective import (
    SharedObjectiveField,
    canonical_to_mechanical,
    covariance_to_canonical,
    covariance_to_mechanical,
    mechanical_symplectic_form,
    mechanical_to_canonical,
    uniform_field_canonical_matrix,
)
from temgym_core.microscope_inversion import (
    canonical_symplectic_form,
    drift_matrix,
    hard_edge_solenoid_matrix,
    smooth_field_solenoid_matrix,
)


@pytest.fixture(scope="module")
def objective():
    return SharedObjectiveField.from_window(-.05, 0., .05, n_slices_per_side=48)


@pytest.fixture
def parameters():
    return jnp.array([-1.25, .0006, np.log(.004), .2])


def test_uniform_body_matches_hard_edge_and_direct_mechanical_lorentz_helix():
    k, distance = 3.4, .17
    canonical = uniform_field_canonical_matrix(k, distance)
    np.testing.assert_allclose(canonical, hard_edge_solenoid_matrix(k*distance, distance), atol=2e-15)
    transform = mechanical_to_canonical(k)
    mechanical = canonical_to_mechanical(k) @ canonical @ transform
    sine, cosine = np.sin(2*k*distance), np.cos(2*k*distance)
    expected = np.array([
        [1., sine/(2*k), 0., (1-cosine)/(2*k)],
        [0., cosine, 0., sine],
        [0., -(1-cosine)/(2*k), 1., sine/(2*k)],
        [0., -sine, 0., cosine],
    ])
    np.testing.assert_allclose(mechanical, expected, atol=5e-16)
    parallel_ray = np.array([1e-6, 0., 2e-6, 0.])
    np.testing.assert_allclose(mechanical @ parallel_ray, parallel_ray, atol=1e-20)
    standard = np.asarray(canonical_symplectic_form())
    assert np.max(np.abs(mechanical.T @ standard @ mechanical-standard)) > .1
    magnetic = mechanical_symplectic_form(k)
    np.testing.assert_allclose(mechanical.T @ magnetic @ mechanical, magnetic, atol=2e-15)


def test_zero_length_derivatives_and_zero_field_are_finite():
    k = 7.1
    f = lambda distance: uniform_field_canonical_matrix(k, distance)
    expected_generator = np.array([[0., 1., k, 0.], [-k*k, 0., 0., k],
                                   [-k, 0., 0., 1.], [0., -k, -k*k, 0.]])
    np.testing.assert_array_equal(f(0.), np.eye(4))
    np.testing.assert_allclose(jax.jacfwd(f)(0.), expected_generator, atol=1e-14)
    np.testing.assert_allclose(jax.jacfwd(jax.jacfwd(f))(0.), expected_generator @ expected_generator, atol=1e-13)
    np.testing.assert_allclose(uniform_field_canonical_matrix(0., .13), drift_matrix(.13), atol=1e-15)
    derivatives = jax.jacfwd(jax.jacfwd(lambda u: uniform_field_canonical_matrix(u[0], u[1])))(jnp.zeros(2))
    assert np.all(np.isfinite(derivatives))
    np.testing.assert_allclose(f(-.04) @ f(.04), np.eye(4), atol=1e-15)


def test_batched_coordinate_transforms_and_covariance_round_trip():
    k = jnp.array([-31., 0., 19.])
    transform = mechanical_to_canonical(k)
    np.testing.assert_allclose(canonical_to_mechanical(k) @ transform,
                               np.broadcast_to(np.eye(4), (3, 4, 4)), atol=1e-14)
    matrix = np.array([[1., .1, .2, 0.], [0., .9, .1, .1], [0., 0., 1.2, .2], [0., 0., 0., .8]])
    scale = np.diag([30e-9, 2e-3, 35e-9, 1e-3])
    covariance = scale @ matrix @ matrix.T @ scale
    canonical = jax.jit(covariance_to_canonical)(covariance, k)
    expected = np.asarray(transform) @ covariance @ np.swapaxes(np.asarray(transform), -1, -2)
    np.testing.assert_allclose(canonical, expected, rtol=1e-14, atol=1e-28)
    np.testing.assert_allclose(covariance_to_mechanical(canonical, k),
                               np.broadcast_to(covariance, (3, 4, 4)), rtol=1e-13, atol=1e-28)
    np.testing.assert_allclose(jax.vmap(mechanical_to_canonical)(k), transform)


def test_global_normalization_and_shared_sample_split(objective, parameters):
    rates = objective.field_samples(parameters)
    edges = np.asarray(objective.z_edges_m)
    np.testing.assert_allclose(np.sum(np.diff(edges)*rates), parameters[0], atol=3e-16)
    maps = jax.jit(objective.split_maps)(parameters)
    independent_full = smooth_field_solenoid_matrix(edges, rates)
    np.testing.assert_allclose(maps.full, independent_full, rtol=2e-13, atol=3e-12)
    np.testing.assert_allclose(maps.post_mechanical @ maps.pre_mechanical, maps.full, rtol=2e-13, atol=3e-12)
    np.testing.assert_allclose(maps.sample_k_m_inv, objective.field(0., parameters), rtol=1e-15)
    # This field is asymmetric and displaced: its halves must not each receive
    # an independently normalized half of q_total.
    split = objective.sample_index
    upstream_strength = np.sum(np.diff(edges)[:split]*np.asarray(rates)[:split])
    assert abs(upstream_strength-float(parameters[0])/2) > .05


def test_canonical_and_mechanical_endpoint_two_forms(objective, parameters):
    maps = objective.split_maps(parameters)
    canonical = canonical_symplectic_form()
    sample = mechanical_symplectic_form(maps.sample_k_m_inv)
    for matrix in (maps.pre, maps.post, maps.full):
        np.testing.assert_allclose(matrix.T @ canonical @ matrix, canonical, atol=3e-12)
    np.testing.assert_allclose(maps.pre_mechanical.T @ sample @ maps.pre_mechanical, canonical, atol=3e-12)
    np.testing.assert_allclose(maps.post_mechanical.T @ canonical @ maps.post_mechanical, sample, atol=3e-12)


def test_zero_objective_field_is_drift_with_finite_second_derivatives(objective, parameters):
    parameters = parameters.at[0].set(0.)
    maps = objective.split_maps(parameters)
    np.testing.assert_allclose(maps.pre, drift_matrix(.05), atol=1e-15)
    np.testing.assert_allclose(maps.post, drift_matrix(.05), atol=1e-15)
    assert maps.sample_k_m_inv == 0
    derivative = jax.jit(jax.jacfwd(jax.jacfwd(lambda p: objective.split_maps(p).post_mechanical)))(parameters)
    assert np.all(np.isfinite(derivative))


def test_profile_symmetry_and_peak_preserving_asymmetry(objective):
    z = jnp.linspace(-.04, .04, 101)
    symmetric = jnp.array([1., 0., np.log(.004), 0.])
    np.testing.assert_allclose(objective.field(z, symmetric), objective.field(-z, symmetric), rtol=2e-15)
    plus, minus = symmetric.at[3].set(.4), symmetric.at[3].set(-.4)
    np.testing.assert_allclose(objective.field(z, plus), objective.field(-z, minus), rtol=3e-15)
    assert not np.allclose(objective.field(z, plus), objective.field(-z, plus))
    assert float(jax.grad(lambda value: objective.field(value, plus))(0.)) == pytest.approx(0., abs=1e-13)
    assert float(jax.grad(jax.grad(lambda value: objective.field(value, plus)))(0.)) < 0
    assert np.argmax(np.asarray(objective.field(z, plus))) == 50
    details = objective.support_diagnostics(plus)
    assert details["upstream_strength_fraction"] != pytest.approx(.5, abs=.05)


def test_field_and_mechanical_response_derivatives_against_finite_difference(objective, parameters):
    scale = jnp.array([1., .003, 1., .3])
    def response(delta):
        p = parameters + scale*delta
        maps = objective.split_maps(p)
        return jnp.array([maps.pre[0, 0], maps.post_mechanical[0, 2],
                           maps.post_mechanical[0, 1]/.01, maps.sample_k_m_inv*.004])
    zero = jnp.zeros(4)
    jacobian = jax.jit(jax.jacfwd(response))(zero)
    hessian = jax.jit(jax.jacfwd(jax.jacfwd(response)))(zero)
    h = 2e-5
    numerical_jacobian = np.column_stack([(np.asarray(response(h*d))-np.asarray(response(-h*d)))/(2*h)
                                         for d in np.eye(4)])
    derivative = jax.jit(jax.jacfwd(response))
    numerical_hessian = np.stack([(np.asarray(derivative(h*d))-np.asarray(derivative(-h*d)))/(2*h)
                                  for d in np.eye(4)], axis=-1)
    np.testing.assert_allclose(jacobian, numerical_jacobian, rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(hessian, numerical_hessian, rtol=3e-6, atol=2e-7)
    np.testing.assert_allclose(jax.jacrev(response)(zero), jacobian, rtol=3e-13, atol=3e-12)
    field_hessian = jax.jit(jax.hessian(lambda p: objective.field(.001, p)))(parameters)
    assert np.all(np.isfinite(field_hessian))


def test_jit_vmap_field_and_split_maps(objective, parameters):
    inputs = jnp.stack((parameters, parameters.at[3].set(-.2)))
    maps = jax.jit(jax.vmap(objective.split_maps))(inputs)
    assert maps.pre.shape == (2, 4, 4)
    assert maps.post.dtype == jnp.float64
    np.testing.assert_allclose(maps.pre[1], objective.split_maps(inputs[1]).pre, rtol=2e-14)
    values = jax.jit(jax.vmap(lambda p: objective.field(jnp.array([-.001, 0., .002]), p)))(inputs)
    assert values.shape == (2, 3)


@pytest.mark.parametrize("edges,sample", [([-.1, 0., 0., .1], 0.),
    ([-.1, .1, 0.], 0.), ([-.1, 0.], 0.), ([-.1, np.nan, .1], 0.),
    ([-.1, .01, .1], 0.), ([-.1, 0., .1], -.1)])
def test_invalid_fixed_topology_is_rejected(edges, sample):
    with pytest.raises(ValueError):
        SharedObjectiveField(edges, sample)


@pytest.mark.parametrize("count", [0, -1, True, 2.5, [2, 0], [2, 3, 4]])
def test_invalid_slice_counts(count):
    with pytest.raises(ValueError):
        SharedObjectiveField.from_window(-.05, 0., .05, n_slices_per_side=count)


@pytest.mark.parametrize("index,value", [(0, np.nan), (1, .06), (2, -1000.), (2, 1000.), (3, .501)])
def test_invalid_parameters_fail_closed_under_jit(objective, parameters, index, value):
    invalid = parameters.at[index].set(value)
    with pytest.raises(ValueError):
        objective.validate_parameters(invalid)
    assert not bool(jax.jit(objective.parameters_valid)(invalid))
    assert np.all(np.isnan(jax.jit(objective.field_samples)(invalid)))
    # Invalid parameters must not masquerade as valid vacuum outside a window.
    assert np.isnan(jax.jit(lambda p: objective.field(1., p))(invalid))


def test_window_contract_and_resolution_diagnostics(objective, parameters):
    np.testing.assert_array_equal(objective.field(jnp.array([-.051, .051]), parameters), 0.)
    assert np.isfinite(objective.field(-.05, parameters))
    diagnostics = objective.support_diagnostics(parameters)
    assert diagnostics["max_slice_to_width_ratio"] > 0
    assert np.all(np.asarray(diagnostics["boundary_to_peak_field_ratio"]) > 0)
    assert diagnostics["upstream_strength_fraction"]+diagnostics["downstream_strength_fraction"] == pytest.approx(1.)
    assert np.isnan(objective.field(jnp.nan, parameters))


def test_fixed_slice_refinement_converges_without_renormalizing_halves(parameters):
    models = [SharedObjectiveField.from_window(-.05, 0., .05, n_slices_per_side=n) for n in (32, 64, 128, 512)]
    maps = [np.asarray(model.split_maps(parameters).full) for model in models]
    errors = [np.linalg.norm(matrix-maps[-1]) for matrix in maps[:-1]]
    assert errors[1] < .35*errors[0]
    assert errors[2] < .35*errors[1]
    # This test establishes quadrature convergence for one case, not a detector
    # noise-relative image acceptance or a physical field-window validation.
