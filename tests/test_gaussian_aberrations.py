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
