import jax
import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.components import ElectromagneticLens, Plane, PolePieceLens
from temgym_core.components.lenses import compute_Rc_from_voltage
from temgym_core.gaussian import make_gaussian
from temgym_core.ray import Ray
from temgym_core.run import column_matrix

VOLTAGE = 200e3
RC = float(compute_Rc_from_voltage(VOLTAGE))


def lens(rotation, gap=0.01, bore_radius=0.004, **kwargs):
    """A lens that turns the image by ``rotation`` radians at 200 kV."""
    return PolePieceLens(z=0.0, turns=1.0, current=rotation / RC, gap=gap, bore_radius=bore_radius, **kwargs)


def test_transfer_matrix_matches_an_independent_ray_integration():
    # The independent check is a Stormer-Verlet integration of the same field with 200,000 fixed
    # steps. An adaptive integrator does not work here. In the field-free tails its error estimate is
    # zero, and one step can jump over the whole field. The cases include a bore 50 times the gap and a
    # gap 50 times the bore.
    cases = jnp.array([(0.3, 0.01, 0.004), (2.0, 0.01, 0.004), (2.5, 0.004, 0.01),
                       (2.5, 0.02, 0.0004), (2.5, 0.0004, 0.02), (3.0, 0.001, 0.001)])
    steps = 200_000

    def both(rotation, gap, bore):
        component = lens(rotation, gap, bore)
        ends = gap / 2 + 20 * bore
        h = 2 * ends / steps

        def kick_drift_kick(rays, z):
            r, slope = rays
            slope = slope - h / 2 * (rotation * component.field_shape(z)) ** 2 * r
            r = r + h * slope
            slope = slope - h / 2 * (rotation * component.field_shape(z + h)) ** 2 * r
            return (r, slope), None

        to_ends = jnp.array([[1.0, -ends], [0.0, 1.0]])
        (r, slope), _ = jax.lax.scan(kick_drift_kick, (to_ends[0], to_ends[1]),
                                     -ends + h * jnp.arange(steps))
        return component.transfer_matrix(VOLTAGE), to_ends @ jnp.stack([r, slope])

    actual, expected = jax.jit(jax.vmap(both))(*cases.T)
    # 200 steps in the class leave an error of about 1e-6 in the strongest case (3 rad, 1 mm).
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-5)


def test_rotation_is_the_same_as_an_electromagnetic_lens_with_the_same_excitation():
    pole_piece = PolePieceLens(z=0.1, turns=500.0, current=3.0, gap=0.008, bore_radius=0.003)
    electromagnetic = ElectromagneticLens(z=0.1, turns=500.0, current=3.0, Gc=1.0)
    assert float(pole_piece.rotation_angle(VOLTAGE)) == pytest.approx(
        float(electromagnetic.rotation_angle(VOLTAGE)), rel=1e-14)


def test_a_weak_lens_has_the_thin_lens_power():
    component = lens(0.02)
    z = np.linspace(-0.3, 0.3, 600001)
    thin_power = 0.02**2 * np.trapezoid(np.asarray(component.field_shape(z)) ** 2, z)
    assert -float(component.transfer_matrix(VOLTAGE)[1, 0]) == pytest.approx(thin_power, rel=1e-3)


def test_column_rotates_focuses_and_differentiates():
    component = PolePieceLens(z=0.05, turns=1.0, current=1.5 / RC, gap=0.01, bore_radius=0.004)
    m = column_matrix([component, Plane(z=0.2)], voltage=VOLTAGE)
    focus = np.asarray(component.transfer_matrix(VOLTAGE))
    rotation = np.array([[np.cos(1.5), -np.sin(1.5)], [np.sin(1.5), np.cos(1.5)]])
    drift = np.array([[1.0, 0.15], [0.0, 1.0]]) @ focus @ np.array([[1.0, 0.05], [0.0, 1.0]])
    np.testing.assert_allclose(m[:2, :2], drift[0, 0] * rotation, atol=1e-12)
    np.testing.assert_allclose(m[:2, 2:4], drift[0, 1] * rotation, atol=1e-12)
    d_gap = jax.grad(lambda gap: column_matrix(
        [PolePieceLens(z=0.05, turns=1.0, current=1.5 / RC, gap=gap, bore_radius=0.004), Plane(z=0.2)],
        voltage=VOLTAGE)[0, 0])(0.01)
    assert np.isfinite(d_gap) and d_gap != 0.0


def test_lens_off_the_axis_acts_about_its_own_axis():
    on_axis, off_axis = lens(1.0), lens(1.0, x0=2e-4, y0=-1e-4)
    ray = Ray(x=3e-4, y=1e-4, dx=1e-3, dy=-2e-3, z=0.0, pathlength=0.0, voltage=VOLTAGE)
    shifted = Ray(x=1e-4, y=2e-4, dx=1e-3, dy=-2e-3, z=0.0, pathlength=0.0, voltage=VOLTAGE)
    a, b = off_axis(ray), on_axis(shifted)
    np.testing.assert_allclose([a.x - 2e-4, a.y + 1e-4, a.dx, a.dy], [b.x, b.y, b.dx, b.dy], atol=1e-15)


def test_gaussian_beams_pass():
    """Gaussian beams are supported since 2026-11-22; tests/test_pole_piece_gaussian.py checks them."""
    out = lens(1.0)(make_gaussian(x=0.0, y=0.0, dx=0.0, dy=0.0, waist_x=1e-8, waist_y=1e-8, voltage=VOLTAGE))
    assert jnp.all(jnp.isfinite(out.Q_inv))


def test_matrix_broadcasts_over_voltage():
    m = lens(1.0).transfer_matrix(jnp.array([100e3, 200e3, 300e3]))
    assert m.shape == (2, 2, 3)


def test_legacy_plot_draws_the_lens_and_follows_its_rotation():
    import matplotlib
    matplotlib.use("Agg")
    from temgym_core.plotting import _compute_cumulative_rotation, plot_model

    component = PolePieceLens(z=0.05, turns=1.0, current=1.2 / RC, gap=0.01, bore_radius=0.004)
    angles = _compute_cumulative_rotation([component, Plane(z=0.2)], voltage=VOLTAGE)
    assert angles[-1] == pytest.approx(1.2)
    rays = Ray(x=jnp.array([0.0, 1e-4]), y=jnp.zeros(2), dx=jnp.zeros(2), dy=jnp.zeros(2),
               z=jnp.zeros(2), pathlength=jnp.zeros(2), voltage=jnp.full(2, VOLTAGE))
    plot_model([component, Plane(z=0.2)], rays=rays, ray_coordinate="x_rot")


def test_field_shape_integrates_to_one_and_zero_current_does_nothing():
    component = lens(1.0, gap=0.01, bore_radius=0.004)
    z = np.linspace(-0.5, 0.5, 1_000_001)
    assert np.trapezoid(np.asarray(component.field_shape(z)), z) == pytest.approx(1.0, rel=1e-6)
    np.testing.assert_allclose(lens(0.0).transfer_matrix(VOLTAGE), np.eye(2), atol=1e-12)


def test_a_long_gap_acts_as_a_uniform_field():
    # With a gap 500 times the bore, the field is nearly uniform over the gap. A uniform field of
    # length S and total rotation q has the power q sin(q) / S, as for HardEdgeSolenoid.
    rotation, gap = 1.5, 0.2
    power = -float(lens(rotation, gap=gap, bore_radius=gap / 500).transfer_matrix(VOLTAGE)[1, 0])
    assert power == pytest.approx(rotation * np.sin(rotation) / gap, rel=0.01)


def test_a_weak_lens_adds_the_thin_lens_optical_path():
    component = lens(0.02)
    focal_length = -1 / float(component.transfer_matrix(VOLTAGE)[1, 0])
    ray = Ray(x=1e-3, y=0.0, dx=0.0, dy=0.0, z=0.0, pathlength=0.0, voltage=VOLTAGE)
    assert float(component(ray).pathlength) == pytest.approx(-(1e-3) ** 2 / (2 * focal_length), rel=1e-3)
