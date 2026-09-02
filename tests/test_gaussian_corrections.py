import numpy as np
import jax.numpy as jnp

from temgym_core.components import (
    AtomicPotential,
    ElectromagneticLens,
    QuadraticPhaseShift,
    SeidelLens,
    SigmoidAperture,
)
from temgym_core.gaussian import make_gaussian
from temgym_core.gaussian_corrections import (
    diagnose_gaussian_action_sampling,
    estimate_width_scale_for_residual,
    gaussian_1e_radii,
    quadratic_residual_on_beam,
)
from temgym_core.potential import Si


VOLTAGE = 200e3


def test_gaussian_1e_radii_match_voltage_first_constructor_waists():
    beam = make_gaussian(voltage=VOLTAGE, waist_x=20e-6, waist_y=35e-6)

    radii = gaussian_1e_radii(beam)

    np.testing.assert_allclose(
        np.sort(np.asarray(radii)),
        np.array([20e-6, 35e-6]),
        rtol=1e-12,
        atol=1e-18,
    )


def test_quadratic_residual_is_zero_for_quadratic_pathlength():
    beam = make_gaussian(voltage=VOLTAGE, waist_x=20e-6, waist_y=25e-6)
    curvature = jnp.array([[2.0e3, -1.0e3], [-1.0e3, 3.0e3]])

    def quadratic(xy):
        return jnp.dot(jnp.array([1e-3, -2e-3]), xy) + 0.5 * xy @ curvature @ xy

    residual = quadratic_residual_on_beam(quadratic, beam, scale=beam.k)

    np.testing.assert_allclose(np.asarray(residual.max_abs), 0.0, atol=1e-8)


def test_quadratic_residual_detects_cubic_pathlength():
    beam = make_gaussian(voltage=VOLTAGE, waist_x=30e-6, waist_y=30e-6)

    def cubic(xy):
        return 1.0e4 * xy[0] ** 3

    residual = quadratic_residual_on_beam(cubic, beam, scale=beam.k)
    width_scale = estimate_width_scale_for_residual(residual.max_abs, target=0.25)

    assert float(residual.max_abs) > 0.25
    assert 0.1 <= float(width_scale) < 1.0


def test_sampling_diagnostic_accepts_legacy_quadratic_phase_component():
    beam = make_gaussian(
        x=jnp.array([0.0, 10e-6]),
        y=0.0,
        voltage=VOLTAGE,
        waist_x=20e-6,
        waist_y=20e-6,
    )
    component = QuadraticPhaseShift(
        z=0.0,
        quadratic_phase_shift=jnp.array([[1.0e3, 2.0e2], [2.0e2, -5.0e2]]),
    )

    diagnostic = diagnose_gaussian_action_sampling(component, beam, phase_tol=1e-8)

    assert not np.asarray(diagnostic.needs_refinement).any()
    np.testing.assert_allclose(np.asarray(diagnostic.phase_max_abs), 0.0, atol=1e-8)
    np.testing.assert_allclose(
        np.asarray(diagnostic.recommended_width_scale),
        1.0,
        atol=1e-12,
    )


def test_sampling_diagnostic_accepts_legacy_phase_with_slope():
    beam = make_gaussian(
        x=jnp.array([0.0, 8e-6]),
        y=0.0,
        dx=jnp.array([0.0, 2e-3]),
        dy=0.0,
        voltage=VOLTAGE,
        waist_x=15e-6,
        waist_y=18e-6,
    )
    lens = SeidelLens(z=0.0, focal_length=0.2)

    diagnostic = diagnose_gaussian_action_sampling(lens, beam, phase_tol=1e-7)

    assert diagnostic.phase_max_abs.shape == (2,)
    assert not np.asarray(diagnostic.needs_refinement).any()
    np.testing.assert_allclose(np.asarray(diagnostic.phase_max_abs), 0.0, atol=1e-7)


def test_sampling_diagnostic_flags_existing_aperture_edge():
    aperture_radius = 95e-6
    beam = make_gaussian(
        x=jnp.array([0.0, aperture_radius]),
        y=0.0,
        voltage=VOLTAGE,
        waist_x=20e-6,
        waist_y=20e-6,
    )
    aperture = SigmoidAperture(
        z=0.0,
        radius=aperture_radius,
        edge_width=4e-6,
        sharpness=16.0,
    )

    diagnostic = diagnose_gaussian_action_sampling(
        aperture,
        beam,
        amplitude_tol=0.05,
        support_radius=2.0,
    )

    needs_refinement = np.asarray(diagnostic.needs_refinement)
    assert not bool(needs_refinement[0])
    assert bool(needs_refinement[1])
    assert float(diagnostic.amplitude_max_abs[1]) > 0.05
    assert float(diagnostic.recommended_width_scale[1]) < 1.0


class _AllActionNames:
    def pathlength_shift_with_slope(self, xy, dxy):
        return 1.0e4 * xy[0] ** 3 + 0.0 * dxy[0]

    def pathlength_shift(self, xy):
        return 0.0 * xy[0]

    def phase_shift(self, xy):
        return 0.0 * xy[0]


class _PathlengthAndLegacyNames:
    def pathlength_shift(self, xy):
        return 1.0e4 * xy[1] ** 3

    def phase_shift(self, xy):
        return 0.0 * xy[0]


def test_component_action_name_priority():
    beam = make_gaussian(
        voltage=VOLTAGE,
        waist_x=30e-6,
        waist_y=30e-6,
        dx=1e-3,
    )

    with_slope = diagnose_gaussian_action_sampling(
        _AllActionNames(), beam, check_amplitude=False
    )
    pathlength = diagnose_gaussian_action_sampling(
        _PathlengthAndLegacyNames(), beam, check_amplitude=False
    )

    assert bool(np.asarray(with_slope.needs_refinement).any())
    assert bool(np.asarray(pathlength.needs_refinement).any())


def test_sampling_diagnostic_derives_electromagnetic_lens_voltage():
    beam = make_gaussian(voltage=VOLTAGE, waist_x=20e-6, waist_y=20e-6)
    lens = ElectromagneticLens(z=0.0, turns=1.0, current=1.0, Gc=1.0)

    diagnostic = diagnose_gaussian_action_sampling(
        lens,
        beam,
        check_amplitude=False,
    )

    assert not np.asarray(diagnostic.needs_refinement).any()
    np.testing.assert_allclose(np.asarray(diagnostic.phase_max_abs), 0.0, atol=1e-8)


def test_sampling_diagnostic_derives_atomic_potential_beam_arguments():
    beam = make_gaussian(
        x=2e-10,
        y=1e-10,
        z=3e-10,
        voltage=VOLTAGE,
        waist_x=2e-11,
        waist_y=2e-11,
    )
    atom = AtomicPotential(
        z=3e-10,
        atom_xyz=jnp.array([0.0, 0.0, 0.0]),
        element_params=Si,
        cutoff_radius=1e-11,
    )

    diagnostic = diagnose_gaussian_action_sampling(
        atom,
        beam,
        check_amplitude=False,
    )

    assert diagnostic.phase_max_abs.shape == (1,)
    assert np.isfinite(np.asarray(diagnostic.phase_max_abs)).all()
