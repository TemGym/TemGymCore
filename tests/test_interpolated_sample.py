"""Physical sample interpolation, Gaussian interaction and JAX derivatives."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gaussian_sampling import sampling_error

from temgym_core.components import InterpolatedSample2D
from temgym_core.gaussian import make_gaussian


def _maps(path, log_amplitude, x, y):
    return dict(pathlength_yx_m=path, log_amplitude_yx=log_amplitude,
                x_coords_m=x, y_coords_m=y)


def _linear_maps():
    x, y = np.linspace(-1.2, 1.2, 9), np.linspace(-0.6, 0.6, 7)
    xx, yy = np.meshgrid(x, y)
    return _maps(3 * xx + 5 * yy, -0.2 + 0.1 * xx - 0.05 * yy, x, y)


def _constant_maps(pathlength=2e-12, log_amplitude=np.log(0.7)):
    x = (np.arange(10) - 4.5) * 0.15e-6
    y = (np.arange(8) - 3.5) * 0.2e-6
    return _maps(np.full((8, 10), pathlength), np.full((8, 10), log_amplitude), x, y)


def test_off_grid_quadratic_values_gradients_and_hessians_are_exact():
    x = np.linspace(-1.1, 1.3, 13)
    y = np.linspace(-0.9, 1.1, 11)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    pathlength = 0.7 + 2.0 * xx - 3.0 * yy + 1.5 * xx**2 + 0.8 * xx * yy - 0.4 * yy**2
    log_amplitude = -0.2 + 0.3 * xx + 0.5 * yy - 0.2 * xx**2 + 0.6 * xx * yy + 0.9 * yy**2
    view = InterpolatedSample2D.from_action_maps(**
        _maps(pathlength, log_amplitude, x, y)
    )
    qx, qy = 0.137, -0.224
    jet = view.action_jet(jnp.asarray((qx, qy)))

    expected_path = 0.7 + 2 * qx - 3 * qy + 1.5 * qx**2 + 0.8 * qx * qy - 0.4 * qy**2
    expected_log = -0.2 + 0.3 * qx + 0.5 * qy - 0.2 * qx**2 + 0.6 * qx * qy + 0.9 * qy**2
    np.testing.assert_allclose(
        (jet.pathlength.value, jet.log_amplitude.value),
        (expected_path, expected_log),
        atol=2e-11,
    )
    np.testing.assert_allclose(
        jet.pathlength.gradient_xy,
        (2 + 3 * qx + 0.8 * qy, -3 + 0.8 * qx - 0.8 * qy),
        atol=2e-10,
    )
    np.testing.assert_allclose(
        jet.log_amplitude.gradient_xy,
        (0.3 - 0.4 * qx + 0.6 * qy, 0.5 + 0.6 * qx + 1.8 * qy),
        atol=2e-10,
    )
    np.testing.assert_allclose(
        jet.pathlength.hessian_xy, ((3.0, 0.8), (0.8, -0.8)), atol=2e-9
    )
    np.testing.assert_allclose(
        jet.log_amplitude.hessian_xy, ((-0.4, 0.6), (0.6, 1.8)), atol=2e-9
    )


def test_pose_rotates_coordinates_gradients_and_hessians():
    level = _linear_maps()
    centre = np.array([2.5, -1.7])
    theta = np.pi / 2
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    local_xy = np.array([0.2, -0.1])
    world_xy = centre + rotation @ local_xy
    view = InterpolatedSample2D.from_action_maps(**
        level, centre_xy_m=centre, rotation_rad=theta
    )

    jet = view.action_jet(jnp.asarray(world_xy))

    np.testing.assert_allclose(jet.pathlength.value, 3 * 0.2 + 5 * -0.1, atol=1e-11)
    np.testing.assert_allclose(
        jet.pathlength.gradient_xy, rotation @ np.array([3.0, 5.0]), atol=1e-10
    )
    np.testing.assert_allclose(jet.pathlength.hessian_xy, 0.0, atol=1e-9)


def test_tilt_foreshortens_coordinates_and_scales_projected_action():
    tilt_x = 0.31
    cosine = np.cos(tilt_x)
    local_xy = np.asarray((0.2, -0.1))
    world_xy = np.asarray((local_xy[0], cosine * local_xy[1]))
    view = InterpolatedSample2D.from_action_maps(**_linear_maps(), tilt_xy_rad=(tilt_x, 0.0))

    jet = view.action_jet(jnp.asarray(world_xy))

    np.testing.assert_allclose(
        jet.pathlength.value,
        (3.0 * local_xy[0] + 5.0 * local_xy[1]) / cosine,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        jet.pathlength.gradient_xy,
        (3.0 / cosine, 5.0 / cosine**2),
        atol=1e-9,
    )
    assert isinstance(view, InterpolatedSample2D)


def test_boundary_modes_are_explicit_and_have_correct_derivatives():
    level = _linear_maps()
    query = jnp.array([2.0, 0.1])

    vacuum = InterpolatedSample2D.from_action_maps(**level, boundary="vacuum")
    np.testing.assert_allclose(vacuum.evaluate_actions(query), [0.0, 0.0], atol=0.0)
    np.testing.assert_allclose(
        vacuum.action_jet(query).pathlength.gradient_xy, 0.0, atol=0.0
    )

    clamped = InterpolatedSample2D.from_action_maps(**level, boundary="constant")
    clamped_jet = clamped.action_jet(query)
    np.testing.assert_allclose(clamped_jet.pathlength.value, 3 * 1.2 + 5 * 0.1, atol=1e-11)
    np.testing.assert_allclose(clamped_jet.pathlength.gradient_xy, [0.0, 5.0], atol=1e-10)

    nan_view = InterpolatedSample2D.from_action_maps(**level, boundary="nan")
    assert np.isnan(np.asarray(nan_view.evaluate_actions(query))).all()

    with pytest.raises(ValueError, match="boundary"):
        InterpolatedSample2D.from_action_maps(**level, boundary="wrap")


def test_periodic_boundary_wraps_values_and_derivatives():
    ny, nx = 8, 12
    pixel = 0.1
    x = (np.arange(nx) - 0.5 * (nx - 1)) * pixel
    y = (np.arange(ny) - 0.5 * (ny - 1)) * pixel
    period_x = nx * pixel
    period_y = ny * pixel
    xx, yy = np.meshgrid(x, y, indexing="xy")
    pathlength = (
        np.cos(2 * np.pi * xx / period_x)
        + 0.2 * np.sin(2 * np.pi * xx / period_x)
        + 0.3 * np.sin(2 * np.pi * yy / period_y)
    )
    level = _maps(pathlength, np.zeros_like(pathlength), x, y)
    view = InterpolatedSample2D.from_action_maps(**level, boundary="periodic")
    point = jnp.array([0.17, -0.13])

    base = view.action_jet(point).pathlength
    wrapped = view.action_jet(point + jnp.array([period_x, -period_y])).pathlength

    np.testing.assert_allclose(wrapped.value, base.value, atol=1e-11)
    np.testing.assert_allclose(wrapped.gradient_xy, base.gradient_xy, atol=1e-10)
    np.testing.assert_allclose(wrapped.hessian_xy, base.hessian_xy, atol=1e-8)

    # Exercise the actual pixel-edge seam, not merely two queries that modulo
    # to the same coordinate.  Value and the first two derivatives converge
    # from opposite sides, which is the C2 contract needed by Gaussian jets.
    xmin, xmax, _, _ = view.domain_bounds_xyxy_m
    epsilon = 1e-8
    left = view.action_jet(jnp.array([xmin + epsilon, 0.13])).pathlength
    right = view.action_jet(jnp.array([xmax - epsilon, 0.13])).pathlength
    np.testing.assert_allclose(left.value, right.value, atol=2e-7)
    np.testing.assert_allclose(left.gradient_xy, right.gradient_xy, atol=2e-6)
    np.testing.assert_allclose(left.hessian_xy, right.hessian_xy, atol=2e-5)


def test_sample_uses_explicit_jet_in_scalar_and_batched_gaussian_action():
    level = _constant_maps()
    view = InterpolatedSample2D.from_action_maps(**level)
    assert isinstance(view, InterpolatedSample2D)
    beam = make_gaussian(
        x=jnp.array([-0.1e-6, 0.1e-6]),
        y=0.0,
        voltage=200e3,
        waist_x=20e-9,
        waist_y=25e-9,
    )

    output = jax.jit(lambda value: view(value))(beam)

    np.testing.assert_allclose(np.abs(output.amplitude), 0.7, rtol=1e-11)
    np.testing.assert_allclose(output.pathlength, 2e-12, atol=1e-24)
    np.testing.assert_allclose(output.dx, 0.0, atol=1e-17)
    np.testing.assert_allclose(output.dy, 0.0, atol=1e-17)


def test_constant_sample_has_zero_quadratic_error():
    view = InterpolatedSample2D.from_action_maps(**_constant_maps())
    beam = make_gaussian(
        x=jnp.array([-0.1e-6, 0.1e-6]),
        y=0.0,
        voltage=200e3,
        waist_x=20e-9,
        waist_y=20e-9,
    )

    diagnostic = sampling_error(
        beam, pathlength=lambda xy, b: view.phase_shift(xy),
        log_amplitude=lambda xy, b: view.log_transmission(xy),
    )

    assert np.all(np.asarray(diagnostic.phase_max_abs) < 1e-10)
    np.testing.assert_allclose(diagnostic.phase_max_abs, 0.0, atol=1e-10)
    np.testing.assert_allclose(diagnostic.amplitude_max_abs, 0.0, atol=1e-10)



def test_diagnostic_flags_a_gaussian_too_wide_for_the_sample():
    voltage = 200e3
    x = np.linspace(-50e-9, 50e-9, 65)
    y = np.linspace(-50e-9, 50e-9, 65)
    xx, _ = np.meshgrid(x, y, indexing="xy")
    phase = 2.0 * np.sin(2 * np.pi * xx / 12e-9)
    from temgym_core.constants import energy2wavelength
    path = phase * float(energy2wavelength(voltage)) / (2 * np.pi)
    level = _maps(path, np.zeros_like(path), x, y)
    view = InterpolatedSample2D.from_action_maps(**level)
    beam = make_gaussian(
        voltage=voltage,
        waist_x=15e-9,
        waist_y=15e-9,
    )

    diagnostic = sampling_error(
        beam, pathlength=lambda xy, b: view.phase_shift(xy), support_radius=2.0,
    )
    assert np.all(np.asarray(diagnostic.phase_max_abs) > 0.1)



def test_sample_actions_are_jittable_and_pose_fields_remain_differentiable():
    view = InterpolatedSample2D.from_action_maps(**_linear_maps())
    points = jnp.array([[0.1, 0.2], [-0.3, 0.05]])

    eager = view.evaluate_actions(points)
    compiled = jax.jit(view.evaluate_actions)(points)

    np.testing.assert_allclose(compiled, eager, rtol=1e-12, atol=1e-12)
    leaves = jax.tree.leaves(view)
    assert any(np.asarray(leaf).shape == (2,) for leaf in leaves)


@pytest.mark.parametrize("boundary", ["vacuum", "constant", "periodic"])
def test_explicit_derivatives_match_jax_with_pose(boundary):
    sample = InterpolatedSample2D.from_action_maps(
        **_linear_maps(), boundary=boundary, centre_xy_m=(0.2, -0.3),
        rotation_rad=0.3, tilt_xy_rad=(0.1, -0.2),
    )
    xy = jnp.array([0.17, -0.11])
    jet = sample.action_jet(xy)
    expected_gradient = jax.jacfwd(sample.evaluate_actions)(xy)
    expected_hessian = jax.jacfwd(jax.jacfwd(sample.evaluate_actions))(xy)
    np.testing.assert_allclose(np.stack([j.gradient_xy for j in jet]), expected_gradient, atol=1e-10)
    np.testing.assert_allclose(np.stack([j.hessian_xy for j in jet]), expected_hessian, atol=1e-9)


def test_action_map_transmission_uses_wavelength_and_unwrapped_path():
    sample = InterpolatedSample2D.from_action_maps(**_constant_maps())
    wavelength = 2.5e-12
    expected = 0.7 * np.exp(2j * np.pi * 2e-12 / wavelength)
    np.testing.assert_allclose(sample.evaluate_complex(jnp.zeros(2), wavelength_m=wavelength), expected, rtol=1e-12)
    with pytest.raises(ValueError, match="wavelength_m"):
        sample.evaluate_complex(jnp.zeros(2))


def test_invalid_maps_are_rejected():
    data = _linear_maps()
    for key, value in [
        ("pathlength_yx_m", np.ones((9, 7))),
        ("log_amplitude_yx", np.full((7, 9), np.nan)),
        ("x_coords_m", np.arange(9)[::-1]),
        ("y_coords_m", np.array([0, 1, 2, 4, 5, 6, 7])),
    ]:
        with pytest.raises(ValueError):
            InterpolatedSample2D.from_action_maps(**(data | {key: value}))


def test_near_opaque_map_has_finite_gaussian_update():
    sample = InterpolatedSample2D.from_action_maps(**_constant_maps(log_amplitude=-100))
    beam = make_gaussian(voltage=200e3, waist_x=20e-9, waist_y=25e-9)
    output = jax.jit(lambda b: sample(b))(beam)
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(output))
    np.testing.assert_allclose(abs(output.amplitude), np.exp(-50), rtol=1e-12)
