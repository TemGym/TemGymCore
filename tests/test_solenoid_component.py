import jax
import jax.numpy as jnp
import numpy as np

from temgym_core.gaussian import FreeSpacePropagator, make_gaussian
from temgym_core.components import HardEdgeSolenoid, Plane
from temgym_core.run import column_matrix


def reference_column(gaps, lengths, strengths):
    """Independent grouped-coordinate paraxial formula, predating migration."""
    m = jnp.eye(4)
    for i, (length, q) in enumerate(zip(lengths, strengths)):
        drift = jnp.eye(4).at[:2, 2:].set(gaps[i] * jnp.eye(2))
        rotation = jnp.array([[jnp.cos(q), jnp.sin(q)], [-jnp.sin(q), jnp.cos(q)]])
        c, b, power = jnp.cos(q), length * jnp.sinc(q / jnp.pi), -q * jnp.sin(q) / length
        lens = jnp.block([[c * rotation, b * rotation], [power * rotation, c * rotation]])
        m = lens @ drift @ m
    m = jnp.eye(4).at[:2, 2:].set(gaps[-1] * jnp.eye(2)) @ m
    return jnp.eye(5).at[:4, :4].set(m)


def test_component_column_matches_independent_matrix_and_derivatives():
    gaps = jnp.array([.002, .19, .07, .06])
    lengths = jnp.array([.001, .005, .003])
    q = jnp.array([-.48, .7, -.62])
    expected = lambda q: reference_column(gaps, lengths, q)
    entrances = jnp.cumsum(gaps[:-1]) + jnp.r_[0.0, jnp.cumsum(lengths[:-1])]
    actual = lambda q: column_matrix([*(HardEdgeSolenoid(z=z, length=length, q=qi)
                                        for z, length, qi in zip(entrances, lengths, q)),
                                      Plane(z=gaps.sum() + lengths.sum())])
    np.testing.assert_allclose(actual(q), expected(q), rtol=3e-13, atol=1e-10)
    np.testing.assert_allclose(jax.jacfwd(actual)(q), jax.jacfwd(expected)(q),
                               rtol=3e-12, atol=1e-9)


def beam():
    return make_gaussian(x=2e-7, y=-3e-7, dx=.0001, dy=-.0002,
                         waist_x=3e-8, waist_y=4e-8, voltage=200e3)


def test_zero_field_solenoid_is_coherent_free_space():
    source = beam()
    actual = HardEdgeSolenoid(z=0., length=.003, q=0.)(source)
    expected = FreeSpacePropagator()(source, .003)
    for field in ('x', 'y', 'dx', 'dy', 'z', 'pathlength', 'amplitude', 'Q_inv'):
        np.testing.assert_allclose(getattr(actual, field), getattr(expected, field), rtol=2e-12)


def test_coherent_lens_subdivision_preserves_action_amplitude_and_curvature():
    source = beam()
    actual = HardEdgeSolenoid(z=0., length=.003, q=1.1)(source)
    divided = source
    for i in range(20):
        divided = HardEdgeSolenoid(z=i * .003 / 20, length=.003 / 20, q=1.1 / 20)(divided)
    for field in ('x', 'y', 'dx', 'dy', 'amplitude', 'Q_inv'):
        np.testing.assert_allclose(getattr(actual, field), getattr(divided, field), rtol=5e-12)
    np.testing.assert_allclose(actual.pathlength, divided.pathlength, atol=2e-18)


def test_gaussian_flux_conserved_and_coherent_derivative_finite():
    source = beam()
    def flux(q):
        out = HardEdgeSolenoid(z=0., length=.003, q=q)(source)
        return jnp.abs(out.amplitude) ** 2 / jnp.sqrt(jnp.linalg.det(jnp.imag(out.Q_inv)))
    np.testing.assert_allclose(flux(.8), flux(0.), rtol=1e-12)
    assert np.isfinite(jax.grad(flux)(.8))


def test_coherent_solenoid_matches_drift_lens_drift_factorization():
    from temgym_core.components import ABCDTransfer, Lens
    source = beam()
    q, length = .83, .004
    distance = length * np.tan(q / 2) / q
    focal_length = length / (q * np.sin(q))
    propagate = FreeSpacePropagator()
    reference = propagate(source, distance)
    reference = Lens(z=distance, focal_length=focal_length)(reference)
    reference = propagate(reference, distance)
    rotation = jnp.array([[np.cos(q), np.sin(q)], [-np.sin(q), np.cos(q)]])
    reference = ABCDTransfer(A=rotation, B=jnp.zeros((2, 2)), D=rotation)(reference)
    actual = HardEdgeSolenoid(z=0., length=length, q=q)(source)
    for field in ('x', 'y', 'dx', 'dy', 'Q_inv', 'amplitude', 'pathlength'):
        np.testing.assert_allclose(getattr(actual, field), getattr(reference, field), rtol=2e-11)
