import jax
import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.nonparaxial_solenoid import (
    kinetic_transverse_momentum, mechanical_slopes, canonical_from_mechanical_slopes,
    uniform_solenoid_hamiltonian, uniform_solenoid_ray,
)
from temgym_core.immersed_objective import uniform_field_canonical_matrix

OMEGA = np.kron(np.eye(2), np.array([[0., 1.], [-1., 0.]]))


@pytest.mark.parametrize('k', [-73., 0., 51.])
def test_paraxial_matrix_and_exact_hamiltonian_vector_field(k):
    p = jnp.array([.00013, .014, -.00021, -.017])
    np.testing.assert_allclose(uniform_solenoid_ray(p, k, .021, order='paraxial'),
        uniform_field_canonical_matrix(k, .021)@p, rtol=1e-13, atol=1e-16)
    tangent = jax.jacfwd(lambda d: uniform_solenoid_ray(p, k, d))(0.)
    expected = OMEGA@jax.grad(uniform_solenoid_hamiltonian)(p, k)
    np.testing.assert_allclose(tangent, expected, rtol=1e-12, atol=1e-13)


def test_uniform_helix_and_kinetic_invariants():
    k, distance = -25., .023
    p = canonical_from_mechanical_slopes(jnp.array([.002, -.001]), jnp.array([.13, -.07]), k)
    uv = kinetic_transverse_momentum(p, k)
    eta = np.sqrt(1-float(uv@uv)); phase = 2*k*distance/eta
    rotated = np.array([[np.cos(phase), np.sin(phase)], [-np.sin(phase), np.cos(phase)]])@uv
    result = uniform_solenoid_ray(p, k, distance)
    np.testing.assert_allclose(kinetic_transverse_momentum(result, k), rotated, atol=1e-15)
    expected_xy = np.asarray(p)[[0, 2]]+np.array([(uv[1]-rotated[1])/(2*k), (rotated[0]-uv[0])/(2*k)])
    np.testing.assert_allclose(np.asarray(result)[[0, 2]], expected_xy, atol=1e-15)
    np.testing.assert_allclose(uniform_solenoid_hamiltonian(result, k), -eta, atol=1e-15)


def test_slopes_conversion_is_not_canonical_slope_shortcut():
    xy, alpha, k = jnp.array([.001, -.002]), jnp.array([.2, -.1]), 40.
    p = canonical_from_mechanical_slopes(xy, alpha, k)
    np.testing.assert_allclose(mechanical_slopes(p, k), alpha, atol=1e-15)
    assert np.linalg.norm(np.asarray(kinetic_transverse_momentum(p, k))-alpha) > .005


def test_nonlinear_canonical_symplecticity_inverse_jit_and_vmap():
    p = jnp.array([.0013, .04, -.0004, -.06]); k, distance = 21., .043
    fun = jax.jit(lambda x: uniform_solenoid_ray(x, k, distance))
    jac = np.asarray(jax.jacfwd(fun)(p))
    np.testing.assert_allclose(jac.T@OMEGA@jac, OMEGA, atol=2e-14)
    np.testing.assert_allclose(uniform_solenoid_ray(fun(p), k, -distance), p, atol=1e-15)
    np.testing.assert_allclose(jax.vmap(fun)(jnp.stack((p, 2*p))), fun(jnp.stack((p, 2*p))), atol=1e-15)
    assert np.isfinite(jax.jacfwd(jax.jacfwd(fun))(p)).all()


def test_exact_drift_and_small_angle_quartic_scaling():
    p = jnp.array([.0001, .08, -.0002, .05])
    expected = np.asarray(p).copy(); expected[[0, 2]] += .2*np.asarray(mechanical_slopes(p, 0.))
    np.testing.assert_allclose(uniform_solenoid_ray(p, 0., .2), expected, atol=1e-15)
    def error(scale, order):
        x = p*scale
        return np.linalg.norm(np.asarray(uniform_solenoid_ray(x, 20., .01)-uniform_solenoid_ray(x, 20., .01, order=order)))
    assert 7.8 < error(1.,'paraxial')/error(.5,'paraxial') < 8.2
    assert 31. < error(1.,'quartic')/error(.5,'quartic') < 33.


def test_invalid_forward_state_not_clipped():
    assert np.isnan(uniform_solenoid_ray(jnp.array([0., 1.1, 0., 0.]), 0., .1)).all()
    with pytest.raises(ValueError): uniform_solenoid_ray(jnp.zeros(4), 1., .1, order='wrong')
