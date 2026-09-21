import jax
import jax.numpy as jnp
import numpy as np

from temgym_core.aberrations import (
    SeidelCoeffs,
    Seidel_object_pos_aperture_pos,
    noll_to_nm,
    zernike_noll,
    zernike_noll_cartesian,
)
from temgym_core.components import SeidelLens, ZernikeAberration
from temgym_core.gaussian import make_gaussian


def _quadratic_taylor_model(fn, centre, point):
    offset = point - centre
    return (
        fn(centre)
        + jnp.dot(jax.grad(fn)(centre), offset)
        + 0.5 * offset @ jax.hessian(fn)(centre) @ offset
    )


def test_cartesian_zernike_matches_polar_and_is_finite_at_origin():
    assert noll_to_nm(4) == (2, 0)
    x = jnp.array([0.0, 0.3, -0.2])
    y = jnp.array([0.0, -0.4, 0.1])
    rho = jnp.hypot(x, y)
    phi = jnp.arctan2(y, x)

    np.testing.assert_allclose(
        np.asarray(zernike_noll_cartesian(4, x, y)),
        np.asarray(zernike_noll(4, rho, phi)),
        rtol=1e-12,
        atol=1e-12,
    )
    hessian = jax.hessian(lambda xy: zernike_noll_cartesian(4, *xy))(
        jnp.zeros(2)
    )
    assert np.isfinite(np.asarray(hessian)).all()


def test_zernike_component_updates_a_batched_electron_beam():
    beam = make_gaussian(
        x=jnp.array([0.0, 0.25]),
        y=0.0,
        voltage=jnp.array([100e3, 200e3]),
        waist_x=0.1,
        waist_y=0.1,
    )
    component = ZernikeAberration(
        z=0.0,
        radius=1.0,
        coefficients={4: 2.0e-9},
    )

    out = component(beam)

    assert out.Q_inv.shape == (2, 2, 2)
    assert np.isfinite(np.asarray(out.Q_inv)).all()
    assert float(jnp.max(jnp.abs(out.Q_inv - beam.Q_inv))) > 0.0


def test_fixed_object_seidel_lens_does_not_depend_on_ray_slope():
    xy = jnp.array([0.03, -0.02])
    coefficients = SeidelCoeffs(C=0.4, D=-0.2, E=0.1)
    lens = SeidelLens(
        z=0.0,
        focal_length=2.0,
        object_plane_dist=9.0,
        object_xy=(0.2, -0.1),
        coeffs=coefficients,
    )

    first = lens.phase_shift(xy, jnp.array([0.4, -0.1]))
    second = lens.phase_shift(xy, jnp.array([-0.2, 0.3]))

    np.testing.assert_allclose(np.asarray(first), np.asarray(second), rtol=0.0, atol=0.0)


def test_quadratic_and_higher_order_seidel_residuals():
    object_xy = jnp.array([0.2, -0.1])
    centre = jnp.array([0.03, 0.04])
    point = jnp.array([-0.2, 0.15])

    def wavefront(aperture_xy, coefficients):
        return Seidel_object_pos_aperture_pos(
            object_xy[0],
            object_xy[1],
            aperture_xy[0],
            aperture_xy[1],
            coefficients,
        )

    def quadratic(xy):
        return wavefront(xy, SeidelCoeffs(C=0.4, D=-0.2))

    def higher_order(xy):
        return wavefront(xy, SeidelCoeffs(B=0.4, F=-0.2))

    np.testing.assert_allclose(
        np.asarray(_quadratic_taylor_model(quadratic, centre, point)),
        np.asarray(quadratic(point)),
        rtol=1e-12,
        atol=1e-12,
    )
    residual = higher_order(point) - _quadratic_taylor_model(
        higher_order,
        centre,
        point,
    )
    assert abs(float(residual)) > 1e-6


def test_distortion_changes_tilt_and_phase_without_extra_curvature():
    """Different pupil rays from one object receive the same cubic kick."""
    from temgym_core.components import DistortedLens

    z1, focal = 0.012, 0.01
    obj = np.array([8e-6, -3e-6])
    xy = np.array([[0., 0.], [20e-6, -10e-6], [-15e-6, 8e-6]])
    slope = (xy - obj) / z1
    E, e = 2e9, -7e8
    beam = make_gaussian(
        x=xy[:, 0], y=xy[:, 1], dx=slope[:, 0], dy=slope[:, 1],
        z=0., voltage=200e3, waist_x=2e-6, waist_y=3e-6,
    )
    lens = DistortedLens(z=0., focal_length=focal, object_plane_dist=z1,
                         IsoDist=E, AnisoDist=e, object_xy=tuple(obj))
    out = jax.jit(lambda b: lens(b))(beam)
    gradient = np.dot(obj, obj) * (E * obj + e * np.array([-obj[1], obj[0]]))
    expected_phase = -np.sum(xy**2, axis=1) / (2*focal) - xy @ gradient

    np.testing.assert_allclose(out.r_xy, xy, atol=1e-18)
    np.testing.assert_allclose(out.d_xy, slope - xy/focal - gradient, atol=1e-15)
    np.testing.assert_allclose(out.pathlength, beam.pathlength + expected_phase, atol=1e-20)
    np.testing.assert_allclose(out.Q_inv, beam.Q_inv - np.eye(2)/focal, atol=1e-12)
    np.testing.assert_allclose(out.amplitude, beam.amplitude, rtol=1e-14)


def test_seidel_reparameterizations_preserve_joint_derivatives():
    from temgym_core.aberrations import (
        Seidel_aperture_pos_aperture_slope,
        Seidel_object_pos_object_slope,
    )

    z1 = .02
    coefficients = SeidelCoeffs(A=2e8, B=1e7, C=3e8, D=-2e8, E=4e8,
                               F=2e8, e=7e8, f=-3e8, c=1e8)
    endpoints = jnp.array([4e-6, -7e-6, 12e-6, 9e-6])

    def original(v):
        return Seidel_object_pos_aperture_pos(*v[:2], *v[2:], coefficients)

    def aperture_version(v):
        slope = (v[2:] - v[:2]) / z1
        return Seidel_aperture_pos_aperture_slope(*v[2:], *slope, z1, coefficients)

    def object_version(v):
        slope = (v[2:] - v[:2]) / z1
        return Seidel_object_pos_object_slope(*v[:2], *slope, z1, coefficients)

    for version in (aperture_version, object_version):
        for operation in (lambda f: f, jax.grad, jax.hessian):
            np.testing.assert_allclose(operation(version)(endpoints),
                                       operation(original)(endpoints),
                                       rtol=2e-13, atol=1e-20)


def test_distorted_lens_honours_explicit_object_and_lens_centre():
    from temgym_core.components import DistortedLens

    obj = (7e-6, -4e-6)
    beam = make_gaussian(x=3e-6, y=5e-6, dx=.01, dy=-.02,
                         z=0., voltage=200e3, waist_x=1e-6, waist_y=1e-6)
    lens = DistortedLens(z=0., focal_length=.02, object_plane_dist=.03,
                         object_xy=obj, x0=2e-6, y0=-1e-6, IsoDist=1e9, AnisoDist=2e9)
    reference = SeidelLens(z=0., focal_length=.02, object_xy=obj,
                           x0=2e-6, y0=-1e-6, coeffs=SeidelCoeffs(E=1e9, e=2e9))
    out, expected = lens(beam), reference(beam)
    for name in ('d_xy', 'pathlength', 'Q_inv'):
        np.testing.assert_allclose(getattr(out, name), getattr(expected, name),
                                   rtol=1e-12, atol=1e-18)


def test_distortion_keeps_outer_object_and_coefficient_derivatives():
    from temgym_core.components import DistortedLens

    z1, focal, pupil = .012, .01, 20e-6
    beam = make_gaussian(x=pupil, y=0., z=0., voltage=200e3,
                         waist_x=1e-6, waist_y=1e-6)

    def kick(obj, coefficient):
        slope = (pupil-obj)/z1
        lens = DistortedLens(z=0., focal_length=focal, object_plane_dist=z1,
                             IsoDist=coefficient)
        out = lens(beam.derive(dx=slope))
        z2 = 1. / (1. / focal - 1. / z1)
        return (out.x + z2 * out.dx + z2 / z1 * obj) / z2

    obj, E = 8e-6, 2e9
    d_obj, d_E = jax.jit(jax.grad(kick, argnums=(0, 1)))(obj, E)
    np.testing.assert_allclose(d_obj, -3*E*obj**2, rtol=1e-12)
    np.testing.assert_allclose(d_E, -obj**3, rtol=1e-12)
