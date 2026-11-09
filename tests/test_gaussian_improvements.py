import jax
import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.components import (
    ABCDTransfer,
    Detector,
    ElectromagneticLens,
    InterpolatedSample2D,
    Lens,
    LinearAmplitudeShift,
    Rotator,
)
from temgym_core.constants import effective_accelerating_potential, energy2wavelength
from temgym_core.evaluate import (
    evaluate_gaussians,
    evaluate_gaussians_for,
    map_reduce,
)
from temgym_core.gaussian import FreeSpacePropagator, make_gaussian
from temgym_core.run import run_to_end


def _assert_beams_close(actual, expected):
    for field in ("x", "y", "dx", "dy", "z", "pathlength", "amplitude", "Q_inv"):
        np.testing.assert_allclose(
            np.asarray(getattr(actual, field)),
            np.asarray(getattr(expected, field)),
            rtol=2e-11,
            atol=1e-13,
        )


def test_make_gaussian_broadcasts_from_any_constructor_field():
    beam = make_gaussian(
        x=0.0,
        y=jnp.array([-2e-6, 0.0, 3e-6]),
        voltage=jnp.array([80e3, 120e3, 200e3]),
        waist_x=20e-6,
        waist_y=jnp.array([22e-6, 24e-6, 26e-6]),
    )

    assert beam.x.shape == (3,)
    assert beam.voltage.shape == (3,)
    assert beam.Q_inv.shape == (3, 2, 2)
    np.testing.assert_array_equal(np.asarray(beam.x), np.zeros(3))


def test_make_gaussian_rejects_bad_shapes_and_nonphysical_values():
    with pytest.raises(ValueError, match="share one leading length"):
        make_gaussian(x=jnp.zeros(2), y=jnp.zeros(3))
    with pytest.raises(ValueError, match="voltage.*positive"):
        make_gaussian(voltage=-1.0)
    with pytest.raises(ValueError, match="waist_x.*positive"):
        make_gaussian(waist_x=0.0)


def test_direct_batched_pipeline_matches_scalar_vmap():
    beam = make_gaussian(
        x=jnp.array([-4e-6, 0.0, 5e-6]),
        y=jnp.array([1e-6, -2e-6, 3e-6]),
        dx=jnp.array([1e-4, -2e-4, 3e-4]),
        dy=0.0,
        voltage=jnp.array([100e3, 160e3, 220e3]),
        waist_x=18e-6,
        waist_y=23e-6,
    )
    components = (
        Lens(z=0.01, focal_length=0.08),
        LinearAmplitudeShift(
            z=0.012,
            linear_log_amplitude=jnp.array([2.0e3, -1.0e3]),
        ),
        Rotator(z=0.014, angle=23.0),
        Detector(z=0.02, pixel_size=(2e-6, 2e-6), shape=(8, 8)),
    )

    direct = run_to_end(beam, components)
    vmapped = jax.vmap(lambda single: run_to_end(single, components))(beam.to_vector())

    _assert_beams_close(direct, vmapped)


def test_lens_subclass_action_override_is_not_bypassed():
    class PhaseModifiedLens(Lens):
        def phase_shift(self, xy):
            return super().phase_shift(xy) + 2.0e-4 * xy[..., 0]

    beam = make_gaussian(x=0.2, voltage=200e3, waist_x=0.1, waist_y=0.1)
    ideal = Lens(z=0.0, focal_length=2.0)(beam)
    modified = PhaseModifiedLens(z=0.0, focal_length=2.0)(beam)

    np.testing.assert_allclose(modified.dx - ideal.dx, 2.0e-4)


def test_singleton_vector_keeps_its_explicit_batch_axis():
    beam = make_gaussian(
        x=2e-6,
        y=-1e-6,
        voltage=200e3,
        waist_x=20e-6,
        waist_y=25e-6,
    ).to_vector()
    beam = LinearAmplitudeShift(
        z=0.0,
        linear_log_amplitude=jnp.array([1.0e3, -2.0e3]),
    )(beam)
    beam = Rotator(z=0.0, angle=17.0)(beam)
    beam = ABCDTransfer(
        z=0.0,
        A=jnp.eye(2),
        B=jnp.zeros((2, 2)),
    )(beam)

    for field in ("x", "y", "dx", "dy", "z", "pathlength", "amplitude"):
        assert jnp.asarray(getattr(beam, field)).shape == (1,)
    assert beam.r_xy.shape == (1, 2)
    assert beam.d_xy.shape == (1, 2)
    assert beam.Q_inv.shape == (1, 2, 2)


def test_electromagnetic_lens_supports_per_beam_voltage():
    voltages = jnp.array([100e3, 200e3])
    beam = make_gaussian(
        x=jnp.array([1e-3, 1e-3]),
        y=jnp.array([0.5e-3, 0.5e-3]),
        voltage=voltages,
        waist_x=20e-6,
        waist_y=25e-6,
    )
    geometry_constant = 5e-6 * float(effective_accelerating_potential(200e3))
    lens = ElectromagneticLens(
        z=0.0,
        turns=100.0,
        current=50.0,
        Gc=geometry_constant,
        stigmator_strength_x=0.1,
        stigmator_strength_y=-0.05,
    )

    direct = lens(beam)
    vmapped = jax.vmap(lens)(beam.to_vector())

    _assert_beams_close(direct, vmapped)
    assert direct.Q_inv.shape == (2, 2, 2)


def test_free_space_gouy_factor_crosses_determinant_branch_continuously():
    beam = make_gaussian(voltage=200e3, waist_x=30e-6, waist_y=30e-6)
    q_inv = jnp.diag(jnp.array([-2.0 + 0.1j, -3.0 + 0.1j]))
    beam = beam.derive(Q_inv=q_inv)

    out = FreeSpacePropagator()(beam, 1.0)
    modal_factors = 1.0 + jnp.linalg.eigvals(q_inv)
    expected = beam.amplitude * jnp.prod(modal_factors ** (-0.5))
    principal_determinant = beam.amplitude / jnp.sqrt(
        jnp.linalg.det(jnp.eye(2, dtype=q_inv.dtype) + q_inv)
    )

    np.testing.assert_allclose(np.asarray(out.amplitude), np.asarray(expected))
    assert float(jnp.abs(out.amplitude - principal_determinant)) > 0.5


def test_isotropic_free_space_gouy_factor_has_finite_gradient():
    def propagated_real_amplitude(waist):
        beam = make_gaussian(
            voltage=200e3,
            waist_x=waist,
            waist_y=waist,
        )
        return jnp.real(FreeSpacePropagator()(beam, 1.0).amplitude)

    gradient = jax.grad(propagated_real_amplitude)(30e-6)

    assert jnp.isfinite(gradient)


def test_rotator_preserves_anisotropic_field_under_active_rotation():
    beam = make_gaussian(
        x=2e-6,
        y=-3e-6,
        dx=1.5e-4,
        dy=-0.7e-4,
        voltage=200e3,
        waist_x=18e-6,
        waist_y=31e-6,
        phase=0.3,
    )
    q_inv = beam.Q_inv.at[0, 1].set(4.0e3 + 2.0e4j)
    q_inv = q_inv.at[1, 0].set(4.0e3 + 2.0e4j)
    beam = beam.derive(Q_inv=q_inv, pathlength=1.3e-3)
    angle = 31.0
    out = Rotator(z=0.0, angle=angle)(beam)

    theta = jnp.deg2rad(angle)
    rotation = jnp.array(
        [[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]]
    )
    point = jnp.array([7e-6, 5e-6])
    rotated_point = rotation @ point

    def field_at(candidate, xy):
        delta = xy - candidate.r_xy
        action = (
            candidate.pathlength
            + jnp.dot(candidate.d_xy, delta)
            + 0.5 * delta @ candidate.Q_inv @ delta
        )
        return candidate.amplitude * jnp.exp(1j * candidate.k * action)

    np.testing.assert_allclose(
        np.asarray(field_at(out, rotated_point)),
        np.asarray(field_at(beam, point)),
        rtol=2e-11,
        atol=1e-11,
    )


def test_public_evaluator_dispatcher_matches_reference_loop():
    beam = make_gaussian(
        x=jnp.array([-1e-6, 0.5e-6]),
        y=jnp.array([0.0, 1e-6]),
        dx=jnp.array([0.0, 1e-4]),
        voltage=200e3,
        waist_x=20e-6,
        waist_y=22e-6,
        amp=jnp.array([1.0, 0.3]),
    )
    grid = Detector(z=0.0, pixel_size=(2e-6, 2e-6), shape=(7, 6))
    reference = evaluate_gaussians_for(beam, grid)

    scan = evaluate_gaussians(beam, grid, batch_size=1)
    loop = evaluate_gaussians(beam, grid, method="loop")
    pallas = evaluate_gaussians(beam, grid, method="gpu")     # interpreted on a CPU

    for result in (scan, loop, pallas):
        np.testing.assert_allclose(
            np.asarray(result),
            np.asarray(reference),
            rtol=2e-11,
            atol=1e-12,
        )


def test_public_evaluator_broadcasts_shared_derived_fields():
    beam = make_gaussian(
        x=jnp.array([-1e-6, 0.0, 1e-6]),
        voltage=200e3,
        waist_x=20e-6,
        waist_y=20e-6,
    ).derive(amplitude=0.5 + 0.2j)
    grid = Detector(z=0.0, pixel_size=(2e-6, 2e-6), shape=(5, 5))

    reference = evaluate_gaussians_for(beam, grid)
    result = evaluate_gaussians(beam, grid, method="scan", batch_size=2)

    np.testing.assert_allclose(result, reference, rtol=2e-11, atol=1e-12)


def test_public_evaluator_rejects_unknown_method():
    beam = make_gaussian(voltage=200e3)
    grid = Detector(z=0.0, pixel_size=(1.0, 1.0), shape=(2, 2))

    with pytest.raises(ValueError, match="Unknown Gaussian evaluation method"):
        evaluate_gaussians(beam, grid, method="mystery")


def test_map_reduce_accepts_array_and_pytree_inputs():
    values = jnp.arange(7.0)
    expected = jnp.sum(values**2)

    array_result = map_reduce(
        lambda value: value**2,
        jnp.add,
        0.0,
        values,
        batch_size=3,
    )
    pytree_result = map_reduce(
        lambda item: item["left"] * item["right"],
        jnp.add,
        0.0,
        {"left": values, "right": values},
        batch_size=3,
    )

    np.testing.assert_allclose(array_result, expected)
    np.testing.assert_allclose(pytree_result, expected)


def test_interpolated_sample_converts_physical_phase_to_pathlength():
    voltage = 200e3
    physical_phase = 0.7
    amplitude = 0.4
    transmission = np.full(
        (4, 4),
        amplitude * np.exp(1j * physical_phase),
        dtype=np.complex128,
    )
    coordinates = np.linspace(-1e-6, 1e-6, 4)
    sample = InterpolatedSample2D.from_array(
        transmission,
        coordinates,
        coordinates,
        voltage=voltage,
    )
    xy = jnp.array([0.0, 0.0])

    expected_pathlength = physical_phase * energy2wavelength(voltage) / (2 * jnp.pi)
    np.testing.assert_allclose(
        np.asarray(sample.phase_shift(xy)),
        np.asarray(expected_pathlength),
        rtol=1e-11,
        atol=1e-18,
    )
    np.testing.assert_allclose(
        np.asarray(sample.log_transmission(xy)),
        np.log(amplitude),
        rtol=1e-11,
        atol=1e-11,
    )


def test_envelope_regularizer_has_finite_derivative_for_round_gaussian():
    from temgym_core.components import _regularize_L2k

    def regularize(width):
        return _regularize_L2k(jnp.zeros((2, 2)), width * jnp.eye(2))

    np.testing.assert_allclose(regularize(3.0), 0.0)
    np.testing.assert_allclose(jax.jacfwd(regularize)(3.0), 0.0)
    np.testing.assert_allclose(jax.jacfwd(jax.jacfwd(regularize))(3.0), 0.0)
