import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from temgym_core.axial_field_column import (
    FixedAxialFieldMesh, canonical_cf4_step, objective_tail_field,
    round_hamiltonian_exponential,
)
from temgym_core.immersed_objective import (
    SharedObjectiveField, mechanical_symplectic_form, uniform_field_canonical_matrix,
)
from temgym_core.microscope_inversion import canonical_symplectic_form


def generator(k):
    return np.array([[0., 1., k, 0.], [-k*k, 0., 0., k],
                     [-k, 0., 0., 1.], [0., -k, -k*k, 0.]])


def test_analytic_hamiltonian_exponential_including_hyperbolic_and_zero_cases():
    for d, r, f, h in [(1., 3., 9., .1), (.5, -2., -3., .4), (1., 0., 0., .2)]:
        matrix = np.array([[0., d, r, 0.], [-f, 0., 0., r],
                           [-r, 0., 0., d], [0., -r, -f, 0.]])
        np.testing.assert_allclose(round_hamiltonian_exponential(d, r, f, h),
                                   expm(h*matrix), rtol=2e-14, atol=2e-15)
    hessian = jax.jacfwd(jax.jacfwd(lambda v: round_hamiltonian_exponential(1., v[0], v[1], v[2])))(jnp.zeros(3))
    assert np.all(np.isfinite(hessian))


def test_cf4_exact_constant_total_field_and_overlap_cross_term():
    # The objective and controlled fields add before k^2. Dropping 2*k1*k2
    # gives a different Hamiltonian even if the rotation strength is correct.
    objective, controlled, h = 3., 5., .04
    total = canonical_cf4_step(objective+controlled, objective+controlled, h)
    np.testing.assert_allclose(total, uniform_field_canonical_matrix(8., h), atol=2e-15)
    omega = canonical_symplectic_form()
    np.testing.assert_allclose(total.T@omega@total, omega, atol=2e-14)
    omitted_cross_term = round_hamiltonian_exponential(1., 8., 3.**2+5.**2, h)
    assert np.max(np.abs(total-omitted_cross_term)) > .1
    np.testing.assert_array_equal(canonical_cf4_step(7., -2., 0.), np.eye(4))
    assert np.all(np.isfinite(jax.hessian(lambda h: canonical_cf4_step(0., 0., h)[0, 1])(0.)))


def test_cf4_order_against_independent_lorentz_hamiltonian_ode():
    field = lambda z: 1.2+3*z+5*z*z
    reference = solve_ivp(lambda z, y: (generator(field(z))@y.reshape(4, 4)).reshape(-1),
                          (0., .3), np.eye(4).reshape(-1), method='DOP853', rtol=2e-13, atol=2e-14).y[:, -1].reshape(4, 4)
    errors = []
    for count in (2, 4, 8, 16):
        matrix = np.eye(4)
        for left in np.linspace(0., .3, count+1)[:-1]:
            h = .3/count
            matrix = np.asarray(canonical_cf4_step(field(left+(.5-np.sqrt(3)/6)*h),
                              field(left+(.5+np.sqrt(3)/6)*h), h))@matrix
        errors.append(np.linalg.norm(matrix-reference))
    assert errors[1] < .08*errors[0]
    assert errors[2] < .08*errors[1]
    assert errors[3] < .08*errors[2]


@pytest.fixture
def column_data():
    reference = SharedObjectiveField.from_window(-.05, 0., .05, n_slices_per_side=256)
    boundaries = jnp.array([-.3, -.2, -.18, -.05, 0., .05, .2, .22, .35])
    bodies = (-1, 0, -1, -1, -1, -1, 1, -1)
    p = jnp.array([-2.7, -.0005, np.log(.002), .15])
    mesh = FixedAxialFieldMesh.from_boundaries(reference, boundaries, bodies, 4, steps_per_segment=32)
    return mesh, p, boundaries, jnp.array([35., -40.])


def test_tail_profile_uses_original_normalizer_and_nonzero_endpoints(column_data):
    mesh, p, boundaries, rates = column_data
    z = jnp.linspace(-.049, .049, 11)
    np.testing.assert_allclose(objective_tail_field(z, p, mesh.normalization_field),
                               mesh.normalization_field.field(z, p), rtol=3e-15)
    assert objective_tail_field(-.3, p, mesh.normalization_field) != 0
    maps = mesh.split_maps(p, boundaries, rates)
    assert maps.source_k_m_inv != 0 and maps.detector_k_m_inv != 0
    np.testing.assert_allclose(maps.post_mechanical@maps.pre_mechanical, maps.full_mechanical,
                               rtol=2e-12, atol=3e-11)
    omega = canonical_symplectic_form()
    # Strong focusing makes M.T J M a cancellation of large terms. Use the
    # normwise backward error here; the individual-step test is absolute.
    residual = maps.full.T@omega@maps.full-omega
    assert np.linalg.norm(residual)/max(1., np.linalg.norm(maps.full)**2) < 100*np.finfo(float).eps
    final_form = mechanical_symplectic_form(maps.detector_k_m_inv)
    residual = (maps.full_mechanical.T@final_form@maps.full_mechanical
                -mechanical_symplectic_form(maps.source_k_m_inv))
    scale = max(1., np.linalg.norm(maps.full_mechanical)**2*np.linalg.norm(final_form))
    assert np.linalg.norm(residual)/scale < 100*np.finfo(float).eps


def test_geometry_and_field_jit_vmap_first_and_second_derivatives(column_data):
    mesh, p, boundaries, rates = column_data
    def response(delta):
        # Move a body and change its length while holding integrated q fixed.
        length = .02*jnp.exp(delta[1])
        centre = -.19+.01*delta[0]
        edges = boundaries.at[1].set(centre-length/2).at[2].set(centre+length/2)
        fields = rates.at[0].set(.7/length)
        obj = p.at[2].add(delta[2]*.1)
        return mesh.split_maps(obj, edges, fields).full_mechanical[0, :]
    zero = jnp.zeros(3)
    jac = jax.jit(jax.jacfwd(response))(zero)
    hessian = jax.jit(jax.jacfwd(jax.jacfwd(response)))(zero)
    step = 1e-4
    fd = np.column_stack([(np.asarray(response(step*d))-np.asarray(response(-step*d)))/(2*step)
                          for d in np.eye(3)])
    derivative = jax.jit(jax.jacfwd(response))
    fd_hess = np.stack([(np.asarray(derivative(step*d))-np.asarray(derivative(-step*d)))/(2*step)
                         for d in np.eye(3)], axis=-1)
    np.testing.assert_allclose(jac, fd, rtol=2e-7, atol=1e-7)
    np.testing.assert_allclose(hessian, fd_hess, rtol=2e-6, atol=1e-7)
    batched = jax.jit(jax.vmap(response))(jnp.zeros((2, 3)))
    np.testing.assert_allclose(batched[0], response(zero), atol=1e-12)


def test_zero_objective_moving_body_is_exact_and_plane_maps_are_continuous(column_data):
    mesh, p, boundaries, rates = column_data
    p = p.at[0].set(0.)
    def exact(edges, controlled):
        result = jnp.eye(4)
        for a, b, label in zip(edges[:-1], edges[1:], mesh.segment_body_indices):
            result = uniform_field_canonical_matrix(0. if label < 0 else controlled[label], b-a)@result
        return result
    actual = lambda edges: mesh.split_maps(p, edges, rates).full
    np.testing.assert_allclose(actual(boundaries), exact(boundaries, rates), atol=3e-12)
    np.testing.assert_allclose(jax.jacfwd(actual)(boundaries),
                               jax.jacfwd(lambda b: exact(b, rates))(boundaries), rtol=3e-12, atol=2e-10)
    # Inside the first body: source drift followed by the exact partial body.
    at = mesh.canonical_map_to_plane(p, boundaries, rates, -.19)
    expected = uniform_field_canonical_matrix(35., .01)@uniform_field_canonical_matrix(0., .1)
    np.testing.assert_allclose(at, expected, rtol=3e-13, atol=2e-13)
    np.testing.assert_allclose(mesh.canonical_map_to_plane(p, boundaries, rates, boundaries[0]), np.eye(4), atol=1e-15)
    assert np.all(np.isnan(mesh.canonical_map_to_plane(p, boundaries, rates, -.31)))


def test_invalid_geometry_fails_closed_and_truncated_comparison_is_explicit(column_data):
    mesh, p, boundaries, rates = column_data
    bad = boundaries.at[2].set(-.21)
    assert np.all(np.isnan(jax.jit(mesh.split_maps)(p, bad, rates).full))
    truncated = FixedAxialFieldMesh.from_boundaries(mesh.normalization_field, boundaries,
        mesh.segment_body_indices, 4, steps_per_segment=32, truncate_objective=True)
    result = truncated.split_maps(p, boundaries, rates)
    assert result.source_k_m_inv == result.detector_k_m_inv == 0
    assert result.sample_k_m_inv != 0
    assert np.linalg.norm(result.full-mesh.split_maps(p, boundaries, rates).full) > .01
    with pytest.raises(ValueError, match='truncation planes'):
        FixedAxialFieldMesh.from_boundaries(mesh.normalization_field, [-.3, 0., .3],
                                           (-1, -1), 1, truncate_objective=True)
