from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from temgym_core.coherent_microscope import (
    ApertureGeometry,
    CoherentMicroscope,
    FiniteLensTrain,
    GaussianSourceState,
    WaveNumerics,
    microscope_from_dict,
    microscope_to_dict,
)
from temgym_core.components import Plane
from temgym_core.evaluate import evaluate_gaussians
from temgym_core.gaussian import FreeSpacePropagator
from temgym_core.run import run_to_end


def model():
    return CoherentMicroscope(
        FiniteLensTrain(
            jnp.array([0.0, 0.035, 0.056, 0.0205, 0.002]),
            jnp.array([0.005, 0.005, 0.003, 0.001]),
            jnp.array([0.2, 0.37, 0.4, -0.48]),
        ),
        FiniteLensTrain(
            jnp.array([0.002, 0.19, 0.07, 0.06, 0.05, 0.3]),
            jnp.array([0.0004, 0.005, 0.005, 0.005, 0.001]),
            jnp.array([-0.42, 0.55, 0.65, 0.45, -0.75]),
        ),
        (GaussianSourceState(0.1975), GaussianSourceState(0.2, x=1e-9)),
        ApertureGeometry(0.05, jnp.zeros((2, 2)), jnp.array([2e-6, 5e-6])),
    )


def test_gaussian_resolution_preserves_complex_field_and_derivative():
    numerical = WaveNumerics(specimen_half_width=70e-9, beamlet_width=2e-9)
    points = jnp.array([[0.0, 0.0], [3e-9, -8e-9], [21e-9, 4e-9]])
    grid = SimpleNamespace(shape=(3,), coords=points)

    def fields(size):
        source = GaussianSourceState(0.2, x=2e-9, theta_x=1e-3, rms_size=size * 1e-9)
        beam = run_to_end(
            source.beam(0.0, 200000.0, jnp.zeros(2)),
            (Plane(z=7e-6),),
            propagator=FreeSpacePropagator(include_axial_path=False),
        )
        return (
            evaluate_gaussians(numerical.resolve(beam), grid, method="scan"),
            evaluate_gaussians(beam, grid, method="scan"),
        )

    a, b = fields(8.0)
    np.testing.assert_allclose(a, b, rtol=2e-8, atol=1e-4)
    da, db = jax.jacfwd(fields)(8.0)
    np.testing.assert_allclose(da, db, rtol=2e-7, atol=1e-3)


def test_aperture_flux_against_independent_radial_integral_and_gradient():
    m = model()
    n = WaveNumerics(radial_order=160)
    from numpy.polynomial.legendre import leggauss

    for voltage in (195000.0, 200000.0):
        aperture = m.aperture_component(jnp.ones(4), voltage, 0)
        beam = run_to_end(
            m.sources[0].beam(-m.condenser.extent - 0.1975, voltage, jnp.zeros(2)),
            (*m.condenser_components(jnp.ones(4), voltage)[:2], Plane(z=aperture.z)),
            propagator=FreeSpacePropagator(include_axial_path=False),
        )
        nodes, weights = leggauss(800)
        r = (nodes + 1) / 2 * 12e-6
        intensity = abs(complex(beam.amplitude)) ** 2 * np.exp(
            -float(beam.k * beam.Q_inv.imag[0, 0]) * r * r
        )
        transmission = 1 / (
            1 + np.exp((r - float(aperture.radius)) / float(aperture.edge_width))
        )
        expected = np.sum(weights * 6e-6 * 2 * np.pi * r * intensity * transmission**2)
        actual = m.current_fraction(jnp.ones(4), voltage, 0, 0, n)
        np.testing.assert_allclose(actual, expected, rtol=1e-8)

    def current(fraction):
        return replace(
            m, aperture=replace(m.aperture, gap_fraction=fraction)
        ).current_fraction(jnp.ones(4), 200000.0, 0, 0, n)

    exact = jax.grad(current)(0.05)
    finite = (current(0.05001) - current(0.04999)) / 0.00002
    np.testing.assert_allclose(exact, finite, rtol=1e-7)


def test_serialization_and_dynamic_voltage_source_index():
    m = model()
    clone = microscope_from_dict(microscope_to_dict(m))
    n = WaveNumerics(radial_order=24, azimuthal_order=16)
    f = jax.jit(lambda i, v: clone.current_fraction(jnp.ones(4), v, i, 0, n))
    assert np.isfinite(f(0, 200000.0))
    assert np.isfinite(f(1, 195000.0))
    assert not np.allclose(
        m.transfer_matrix(jnp.ones(4), jnp.ones(5), 200000.0, 0),
        m.transfer_matrix(jnp.ones(4), jnp.ones(5), 195000.0, 1),
    )


def test_envelope_regularizer_has_finite_derivative_for_round_gaussian():
    from temgym_core.components import _regularize_L2k

    def regularize(width):
        return _regularize_L2k(jnp.zeros((2, 2)), width * jnp.eye(2))

    np.testing.assert_allclose(regularize(3.0), 0.0)
    np.testing.assert_allclose(jax.jacfwd(regularize)(3.0), 0.0)
    np.testing.assert_allclose(jax.jacfwd(jax.jacfwd(regularize))(3.0), 0.0)
