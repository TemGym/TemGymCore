import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.constants import energy2wavelength
from temgym_core.gaussian import FreeSpacePropagator
from temgym_core.source import (
    _gaussian_cardinal_coefficients,
    airy_input_wave_for_flat_disk,
    circular_input_wave,
    decompose_gaussian,
    finite_probe_wave_focused_to_plane,
    make_waist_divergence_rays,
    point_source_wave_focused_to_plane,
    rectangular_input_wave,
    square_input_wave,
    gaussian_waist,
)


def test_decomposition_preserves_source_and_native_lens_propagation():
    """Choosing an aperture-plane basis must not change the physical source."""
    from dataclasses import replace
    from types import SimpleNamespace
    from temgym_core.components import HardEdgeSolenoid, Plane
    from temgym_core.evaluate import evaluate_gaussians
    from temgym_core.run import run_to_end

    source = gaussian_waist(z=0., voltage=200e3, rms_size=3e-9,
                            x=1e-9, y=-2e-9, theta_x=1e-6, theta_y=-2e-6)
    column = (HardEdgeSolenoid(z=.2, length=.006, q=.22, include_axial_path=False),
              HardEdgeSolenoid(z=.246, length=.004, q=.4, include_axial_path=False),
              Plane(z=.253))
    bundle = decompose_gaussian(source, num_across=96, through=column)
    assert bundle.x.shape == (96**2,)
    np.testing.assert_allclose(bundle.z, source.z)
    fs = FreeSpacePropagator(include_axial_path=False)
    changed = (column[0], replace(column[1], q=.404), column[2])
    for components in ((), column, changed):
        exact = run_to_end(source, components, propagator=fs) if components else source
        sampled = run_to_end(bundle, components, propagator=fs) if components else bundle
        rms = np.sqrt(np.asarray(exact.moments()[1])[0, 0])
        x = np.linspace(-3, 3, 51) * rms + float(exact.x)
        grid = SimpleNamespace(shape=(51,), coords=jnp.asarray(
            np.column_stack((x, np.full_like(x, float(exact.y))))))
        expected = np.asarray(evaluate_gaussians(exact, grid, method="scan"))
        actual = np.asarray(evaluate_gaussians(sampled, grid, method="scan"))
        assert np.linalg.norm(actual-expected) / np.linalg.norm(expected) < 2e-4


def test_refining_decomposition_resolves_sigmoid_transmission():
    from types import SimpleNamespace
    from temgym_core.components import SigmoidAperture
    from temgym_core.evaluate import evaluate_gaussians

    source = gaussian_waist(z=0., voltage=200e3, rms_size=2e-6)
    aperture = SigmoidAperture(z=0., radius=2e-6, edge_width=.12e-6)
    x = np.linspace(1e-6, 3e-6, 81)
    grid = SimpleNamespace(shape=x.shape, coords=jnp.asarray(
        np.column_stack((x, np.zeros_like(x)))))
    expected = np.asarray(evaluate_gaussians(source, grid, method="scan")
                          * jnp.exp(aperture.log_transmission(grid.coords)))
    errors = []
    for n in (128, 256):
        bundle = decompose_gaussian(source, num_across=n)
        actual = np.asarray(evaluate_gaussians(aperture(bundle), grid, method="scan"))
        errors.append(np.linalg.norm(actual-expected) / np.linalg.norm(expected))
    assert errors[1] < errors[0] / 3
    assert errors[1] < .01


def test_decomposition_rejects_invalid_sampling():
    source = gaussian_waist(z=0., voltage=200e3, rms_size=3e-9)
    for kwargs in ({"num_across": 4}, {"num_across": 16, "overlap": 100},
                   {"num_across": 16, "extent_rms": 0}):
        with pytest.raises(ValueError):
            decompose_gaussian(source, **kwargs)



def test_decomposition_roundness_tolerates_relative_rotation_residue():
    from types import SimpleNamespace
    from temgym_core.evaluate import evaluate_gaussians

    source = gaussian_waist(z=0., voltage=200e3, rms_size=50e-9)
    q = source.Q_inv
    residue = 1e-12 * jnp.abs(q[0, 0])
    source = source.derive(Q_inv=q.at[0, 1].set(residue).at[1, 0].set(residue))
    bundle = decompose_gaussian(source, num_across=32)
    x = jnp.linspace(-100e-9, 100e-9, 51)
    grid = SimpleNamespace(shape=(51,), coords=jnp.stack((x, x/2), axis=-1))
    expected = np.asarray(evaluate_gaussians(source, grid, method="scan"))
    actual = np.asarray(evaluate_gaussians(bundle, grid, method="scan"))
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=1e-9)
    # An actual elliptical envelope or cross-axis coupling is still rejected.
    for changed in (q.at[1, 1].multiply(1.01), q.at[0, 1].set(.01*q[0, 0])):
        with pytest.raises(ValueError, match="round"):
            decompose_gaussian(source.derive(Q_inv=changed), num_across=32)

def test_gaussian_cardinal_coefficients_reconstruct_square_samples():
    axis = 9
    width_over_step = 1.1
    ridge = 1e-10
    coordinates = np.linspace(-1.0, 1.0, axis)
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    target = np.exp(-(xx**2 + 0.7 * yy**2)) * np.cos(2.0 * xx - yy)

    coefficients = _gaussian_cardinal_coefficients(
        target,
        width_over_step=width_over_step,
        ridge=ridge,
    )

    indices = np.arange(axis)
    kernel = np.exp(
        -((indices[:, None] - indices[None, :]) / width_over_step) ** 2
    )
    kernel += ridge * np.eye(axis)
    reconstructed = kernel @ coefficients @ kernel.T
    np.testing.assert_allclose(reconstructed, target, rtol=1e-11, atol=1e-11)

    with pytest.raises(ValueError, match="square two-dimensional"):
        _gaussian_cardinal_coefficients(
            np.ones((axis, axis - 1)),
            width_over_step=width_over_step,
        )


def test_airy_input_wave_builds_signed_steered_source_in_coordinate_units():
    axis = 10
    voltage = 200e3
    disk_radius = 100e-6
    focal_length = 10e-3
    target_centre = (25e-6, -15e-6)
    source_m = airy_input_wave_for_flat_disk(
        axis=axis,
        disk_radius=disk_radius,
        focal_length=focal_length,
        voltage=voltage,
        target_centre_xy=target_centre,
        rings=4.0,
    ).to_vector()

    assert np.asarray(source_m.x).shape == (axis * axis,)
    assert np.asarray(source_m.Q_inv).shape == (axis * axis, 2, 2)
    amplitude = np.asarray(source_m.amplitude)
    assert np.all(np.isfinite(amplitude))
    np.testing.assert_allclose(
        np.asarray(source_m.dx),
        target_centre[0] / focal_length,
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        np.asarray(source_m.dy),
        target_centre[1] / focal_length,
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        np.real(np.asarray(source_m.Q_inv)),
        0.0,
        rtol=0.0,
        atol=1e-15,
    )

    wavelength_m = float(np.asarray(energy2wavelength(voltage)))
    phase = (2.0 * np.pi / wavelength_m) * (
        np.asarray(source_m.dx) * np.asarray(source_m.x)
        + np.asarray(source_m.dy) * np.asarray(source_m.y)
    )
    untilted_amplitude = amplitude * np.exp(-1j * phase)
    np.testing.assert_allclose(
        np.imag(untilted_amplitude),
        0.0,
        rtol=0.0,
        atol=1e-11,
    )
    assert np.min(np.real(untilted_amplitude)) < 0.0
    assert np.max(np.real(untilted_amplitude)) > 0.0

    metres_per_micron = 1e-6
    source_um = airy_input_wave_for_flat_disk(
        axis=axis,
        disk_radius=disk_radius / metres_per_micron,
        focal_length=focal_length / metres_per_micron,
        voltage=voltage,
        target_centre_xy=tuple(v / metres_per_micron for v in target_centre),
        rings=4.0,
        wavelength_unit="um",
    ).to_vector()
    np.testing.assert_allclose(
        np.asarray(source_um.x),
        np.asarray(source_m.x) / metres_per_micron,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(source_um.Q_inv),
        np.asarray(source_m.Q_inv) * metres_per_micron,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(source_um.amplitude),
        amplitude,
        rtol=1e-11,
        atol=1e-11,
    )


def test_point_source_focused_to_plane_sets_electron_beam_waist():
    voltage = 200e3
    target_z = 20e-3
    target_waist = 8e-9
    beam = point_source_wave_focused_to_plane(
        n=8,
        half_angle=5e-3,
        waist_at_plane=target_waist,
        voltage=voltage,
        target_z=target_z,
    )

    out = FreeSpacePropagator()(beam, target_z)
    wavelength = energy2wavelength(voltage)
    q_im = jnp.imag(out.Q_inv[:, 0, 0])
    waist_x = jnp.sqrt(wavelength / (jnp.pi * q_im))
    np.testing.assert_allclose(
        np.asarray(waist_x),
        target_waist,
        rtol=1e-11,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        np.sum(np.asarray(beam.amplitude)),
        1.0,
        rtol=0.0,
        atol=1e-15,
    )


def test_point_source_focusing_is_invariant_to_coordinate_unit():
    scale = 1e6
    kwargs = dict(
        n=8,
        half_angle=5e-3,
        waist_at_plane=8e-9,
        voltage=200e3,
        target_z=20e-3,
        z0=1e-3,
        x0=2e-6,
        y0=-3e-6,
    )
    beam_m = point_source_wave_focused_to_plane(**kwargs)
    beam_um = point_source_wave_focused_to_plane(
        **{
            **kwargs,
            "waist_at_plane": kwargs["waist_at_plane"] * scale,
            "target_z": kwargs["target_z"] * scale,
            "z0": kwargs["z0"] * scale,
            "x0": kwargs["x0"] * scale,
            "y0": kwargs["y0"] * scale,
        },
        wavelength_unit="um",
    )

    np.testing.assert_allclose(np.asarray(beam_um.x), np.asarray(beam_m.x) * scale)
    np.testing.assert_allclose(np.asarray(beam_um.y), np.asarray(beam_m.y) * scale)
    np.testing.assert_allclose(np.asarray(beam_um.dx), np.asarray(beam_m.dx))
    np.testing.assert_allclose(np.asarray(beam_um.dy), np.asarray(beam_m.dy))
    np.testing.assert_allclose(
        np.asarray(beam_um.Q_inv),
        np.asarray(beam_m.Q_inv) / scale,
        rtol=1e-12,
        atol=1e-15,
    )


def test_finite_probe_focused_to_plane_maps_between_offset_disks():
    voltage = 200e3
    input_radius = 12e-6
    target_radius = 105e-6
    target_z = 20e-3
    target_waist = 8e-9
    input_center = (2e-6, -3e-6)
    target_center = (25e-6, 10e-6)
    beam = finite_probe_wave_focused_to_plane(
        n=64,
        input_radius=input_radius,
        target_radius=target_radius,
        waist_at_plane=target_waist,
        voltage=voltage,
        target_z=target_z,
        input_center_xy=input_center,
        target_center_xy=target_center,
    )

    input_radii = jnp.hypot(beam.x - input_center[0], beam.y - input_center[1])
    assert float(jnp.max(input_radii)) <= input_radius * (1.0 + 1e-12)

    out = FreeSpacePropagator()(beam, target_z)
    target_radii = jnp.hypot(
        out.x - target_center[0],
        out.y - target_center[1],
    )
    assert float(jnp.max(target_radii)) <= target_radius * (1.0 + 1e-12)
    assert float(jnp.max(target_radii)) > 0.9 * target_radius

    wavelength = energy2wavelength(voltage)
    q_im = jnp.imag(out.Q_inv[:, 0, 0])
    waist_x = jnp.sqrt(wavelength / (jnp.pi * q_im))
    np.testing.assert_allclose(
        np.asarray(waist_x),
        target_waist,
        rtol=1e-11,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        np.sum(np.asarray(beam.amplitude)),
        1.0,
        rtol=0.0,
        atol=1e-15,
    )


def test_focused_source_validation():
    with pytest.raises(ValueError, match="differ"):
        finite_probe_wave_focused_to_plane(
            n=8,
            input_radius=1.0,
            target_radius=1.0,
            waist_at_plane=0.1,
            voltage=200e3,
            target_z=2.0,
            z0=2.0,
        )

    with pytest.raises(ValueError, match="Unknown wavelength unit"):
        point_source_wave_focused_to_plane(
            n=8,
            half_angle=0.1,
            waist_at_plane=0.1,
            voltage=200e3,
            target_z=2.0,
            wavelength_unit="parsec",
        )


def test_square_source_is_the_equal_sided_rectangular_source():
    kwargs = dict(
        waist=0.3,
        voltage=200e3,
        amp=2.5,
        phase=0.2,
        z0=0.4,
        overlap_factor=1.5,
        centre_xy=(0.1, -0.2),
    )
    square = square_input_wave(aperture_length=1.0, **kwargs)
    rectangle = rectangular_input_wave(
        aperture_width=1.0,
        aperture_height=1.0,
        **kwargs,
    )

    for field in ("x", "y", "dx", "dy", "amplitude", "Q_inv"):
        np.testing.assert_allclose(
            np.asarray(getattr(square, field)),
            np.asarray(getattr(rectangle, field)),
        )


def test_uniform_sources_apply_requested_amplitude_and_voltage():
    circular_base = circular_input_wave(1.0, 0.5, 200e3, amp=1.0)
    circular_scaled = circular_input_wave(1.0, 0.5, 200e3, amp=2.5)
    rectangular_base = rectangular_input_wave(1.0, 0.8, 0.3, 200e3, amp=1.0)
    rectangular_scaled = rectangular_input_wave(1.0, 0.8, 0.3, 200e3, amp=2.5)

    np.testing.assert_allclose(
        np.asarray(circular_scaled.amplitude),
        2.5 * np.asarray(circular_base.amplitude),
    )
    np.testing.assert_allclose(
        np.asarray(rectangular_scaled.amplitude),
        2.5 * np.asarray(rectangular_base.amplitude),
    )

    basis = make_waist_divergence_rays(1e-8, voltage=80e3)
    np.testing.assert_array_equal(np.asarray(basis.voltage), np.full(2, 80e3))
