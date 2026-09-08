"""Check when a component's local quadratic action resolves a Gaussian beam."""

import jax
import jax.numpy as jnp
import numpy as np

from gaussian_sampling import _support_offsets, sampling_error
from temgym_core.components import (
    AtomicPotential,
    ElectromagneticLens,
    QuadraticPhaseShift,
    SeidelLens,
    SigmoidAperture,
)
from temgym_core.gaussian import make_gaussian
from temgym_core.potential import Si


VOLTAGE = 200e3


def test_support_principal_radii_match_gaussian_waists():
    beam = make_gaussian(voltage=VOLTAGE, waist_x=20e-6, waist_y=35e-6)
    offsets = jax.vmap(_support_offsets)(beam.to_vector())[0]
    radii = np.linalg.norm(np.asarray(offsets[:4]), axis=1)
    np.testing.assert_allclose(np.sort(radii), [20e-6, 20e-6, 35e-6, 35e-6], rtol=1e-12)


def test_quadratic_pathlength_has_zero_residual():
    beam = make_gaussian(voltage=VOLTAGE, waist_x=20e-6, waist_y=25e-6)
    curvature = jnp.array([[2e3, -1e3], [-1e3, 3e3]])

    def pathlength(xy, beam):
        return jnp.dot(jnp.array([1e-3, -2e-3]), xy) + 0.5 * xy @ curvature @ xy

    error = sampling_error(beam, pathlength=pathlength)
    np.testing.assert_allclose(error.phase_max_abs, 0.0, atol=1e-8)


def test_halving_waist_reduces_cubic_residual_eightfold():
    beam = make_gaussian(
        voltage=VOLTAGE, waist_x=jnp.array([30e-6, 15e-6]),
        waist_y=jnp.array([30e-6, 15e-6]),
    )
    error = sampling_error(beam, pathlength=lambda xy, b: 1e4 * xy[0]**3)
    assert error.phase_max_abs[0] > 0.25
    np.testing.assert_allclose(error.phase_max_abs[1] / error.phase_max_abs[0], 1 / 8)


def test_quadratic_component_is_resolved_across_a_bundle():
    beam = make_gaussian(
        x=jnp.array([0., 10e-6]), voltage=VOLTAGE,
        waist_x=20e-6, waist_y=20e-6,
    )
    component = QuadraticPhaseShift(
        z=0., quadratic_phase_shift=jnp.array([[1e3, 2e2], [2e2, -5e2]]),
    )
    error = sampling_error(beam, pathlength=lambda xy, b: component.phase_shift(xy))
    assert error.phase_max_abs.shape == (2,)
    np.testing.assert_allclose(error.phase_max_abs, 0., atol=1e-8)


def test_lens_with_explicit_beam_slopes_is_resolved():
    beam = make_gaussian(
        x=jnp.array([0., 8e-6]), dx=jnp.array([0., 2e-3]),
        voltage=VOLTAGE, waist_x=15e-6, waist_y=18e-6,
    )
    lens = SeidelLens(z=0., focal_length=0.2)
    error = sampling_error(beam, pathlength=lambda xy, b: lens.phase_shift(xy, b.d_xy))
    np.testing.assert_allclose(error.phase_max_abs, 0., atol=1e-7)


def test_aperture_edge_is_unresolved_but_centre_is_resolved():
    radius = 95e-6
    beam = make_gaussian(
        x=jnp.array([0., radius]), voltage=VOLTAGE, waist_x=20e-6, waist_y=20e-6,
    )
    aperture = SigmoidAperture(z=0., radius=radius, edge_width=4e-6, sharpness=16.)
    error = sampling_error(
        beam, log_amplitude=lambda xy, b: aperture.log_transmission(xy),
        support_radius=2.,
    )
    assert error.amplitude_max_abs[0] < 0.05
    assert error.amplitude_max_abs[1] > 0.05


def test_electromagnetic_lens_with_explicit_voltage_is_resolved():
    beam = make_gaussian(voltage=VOLTAGE, waist_x=20e-6, waist_y=20e-6)
    lens = ElectromagneticLens(z=0., turns=1., current=1., Gc=1.)
    error = sampling_error(beam, pathlength=lambda xy, b: lens.phase_shift(xy, b.voltage))
    np.testing.assert_allclose(error.phase_max_abs, 0., atol=1e-8)


def test_atomic_potential_with_explicit_beam_arguments_has_finite_error():
    beam = make_gaussian(
        x=2e-10, y=1e-10, z=3e-10, voltage=VOLTAGE, waist_x=2e-11, waist_y=2e-11,
    )
    atom = AtomicPotential(
        z=3e-10, atom_xyz=jnp.zeros(3), element_params=Si, cutoff_radius=1e-11,
    )
    error = sampling_error(
        beam, pathlength=lambda xy, b: atom.phase_shift(xy, b.z, b.sigma, b.k),
    )
    assert error.phase_max_abs.shape == (1,)
    assert np.isfinite(np.asarray(error.phase_max_abs)).all()
