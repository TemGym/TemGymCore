"""Native component ray maps: polar equivalence, axis jets and stationarity."""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.jet import jet

from temgym_core.aberrations import (
    KrivanekCoeffs, SeidelCoeffs, W_krivanek, W_krivanek_cartesian,
    grad_W_krivanek, grad_W_krivanek_cartesian,
)
from temgym_core import components
from temgym_core.components import KrivanekLens, Lens, SeidelLens

# The component was referenced in e415ce3 but never committed.
KrivanekLensCartesian = getattr(components, "KrivanekLensCartesian", None)
needs_cartesian_lens = pytest.mark.skipif(
    KrivanekLensCartesian is None, reason="KrivanekLensCartesian is not implemented"
)
# SeidelLens raises NotImplementedError for singular rays; the ray support was never committed.
seidel_ray_not_implemented = pytest.mark.xfail(
    raises=NotImplementedError, strict=True, reason="SeidelLens does not support singular rays"
)
from temgym_core.ray import Ray
from temgym_core.source import gaussian_waist


def all_krivanek_terms():
    rng = np.random.default_rng(31)
    return KrivanekCoeffs(**{
        f.name: float(rng.uniform(-.7, .7)) for f in dataclasses.fields(KrivanekCoeffs)
    })


def test_cartesian_wave_and_gradient_match_polar_off_axis():
    coeffs = all_krivanek_terms()
    xy = np.random.default_rng(7).uniform(-.2, .2, (50, 2))
    x, y = xy.T
    np.testing.assert_allclose(
        W_krivanek_cartesian(x, y, coeffs),
        W_krivanek(jnp.hypot(x, y), jnp.arctan2(y, x), coeffs),
        rtol=1e-12, atol=1e-17,
    )
    np.testing.assert_allclose(grad_W_krivanek_cartesian(x, y, coeffs),
                               grad_W_krivanek(x, y, coeffs), rtol=1e-12, atol=1e-16)


@needs_cartesian_lens
@pytest.mark.parametrize("focal", [.02, -.02])
def test_cartesian_component_matches_legacy_off_axis(focal):
    parameters = dict(z=0., focal_length=focal, x0=2e-5, y0=-1e-5,
                      coeffs=all_krivanek_terms())
    old, new = KrivanekLens(**parameters), KrivanekLensCartesian(**parameters)
    ray = Ray.origin().derive(x=3e-4, y=-2e-4, dx=.03, dy=.02, pathlength=.4)
    a, b = old(ray), new(ray)
    for name in ("r_xy", "d_xy", "pathlength"):
        np.testing.assert_allclose(getattr(a, name), getattr(b, name), rtol=1e-12, atol=1e-16)
    np.testing.assert_allclose(old.phase_shift(ray.r_xy), new.phase_shift(ray.r_xy),
                               rtol=1e-12, atol=1e-16)


@needs_cartesian_lens
def test_cartesian_on_axis_ray_jet_retains_fifth_order_aberration():
    focal, cs, c5 = .02, .03, .7
    lens = KrivanekLensCartesian(z=0., focal_length=focal,
                                coeffs=KrivanekCoeffs(C30=cs, C50=c5))
    def trace(slope):
        return lens(Ray.origin().derive(dx=slope)).dx
    value, series = jet(trace, (jnp.array(0.),),
                        ((jnp.array(1.),) + (jnp.array(0.),)*4,))
    assert float(value) == 0.
    np.testing.assert_allclose(series, [1., 0., -6*cs/focal, 0., -120*c5/focal],
                               rtol=1e-12, atol=1e-12)


@needs_cartesian_lens
def test_cartesian_gaussian_axis_and_high_order_action():
    parameters = dict(z=0., focal_length=.02,
                      coeffs=KrivanekCoeffs(C12=2e-6, phi12=.3, C30=.004))
    old, new = KrivanekLens(**parameters), KrivanekLensCartesian(**parameters)
    beam = gaussian_waist(z=0., voltage=200e3, rms_size=1e-6)
    np.testing.assert_allclose(new(beam).Q_inv, old(beam).Q_inv, rtol=1e-12)
    _, series = jet(lambda x: new.phase_shift(jnp.array([x, 0.])),
                    (jnp.array(0.),), ((jnp.array(1.),)+(jnp.array(0.),)*3,))
    np.testing.assert_allclose(series[3], -6*.004/.02**4, rtol=1e-12)


@seidel_ray_not_implemented
@pytest.mark.parametrize("distance", [0., .04, -.04])
def test_seidel_ray_zero_aberrations_matches_ideal(distance):
    parameters = dict(z=0., focal_length=.02, x0=1e-5, y0=-2e-5)
    lens = SeidelLens(**parameters, object_plane_dist=distance)
    ray = Ray.origin().derive(x=2e-4, y=-1e-4, dx=.002, dy=-.003, pathlength=.7)
    expected, actual = Lens(**parameters)(ray), jax.jit(lens)(ray)
    for name in ("r_xy", "d_xy", "pathlength"):
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name), atol=1e-15)


@seidel_ray_not_implemented
def test_seidel_object_phase_has_the_stationary_position_shift():
    # An object-only phase is invisible to a pupil-only kick. The geometric
    # kernel instead gives an exact position shift and changed outgoing slope.
    distance, focal, A = .04, .02, 2e8
    ray = Ray.origin().derive(x=1e-4, y=-2e-4, dx=.001, dy=-.001)
    obj = np.asarray(ray.r_xy-distance*ray.d_xy)
    go = -4*A*(obj@obj)*obj
    aperture = np.asarray(ray.r_xy) + distance*go
    out = SeidelLens(z=0., focal_length=focal, object_plane_dist=distance,
                     coeffs=SeidelCoeffs(A=A))(ray)
    np.testing.assert_allclose(out.r_xy, aperture, atol=1e-17)
    np.testing.assert_allclose(out.d_xy, ray.d_xy+go-aperture/focal, atol=1e-16)
    action = -aperture@aperture/(2*focal)-A*(obj@obj)**2
    path = action + distance*(np.asarray(ray.d_xy)@go + .5*(go@go))
    np.testing.assert_allclose(out.pathlength, path, atol=1e-18)


@seidel_ray_not_implemented
def test_seidel_ray_stationarity_and_symplectic_map():
    distance = .04
    lens = SeidelLens(z=0., focal_length=.02, object_plane_dist=distance,
        coeffs=SeidelCoeffs(A=.6e8, B=1.2e8, C=-.5e8, D=.7e8,
                           E=-.4e8, F=.9e8, e=.3e8, f=-.2e8, c=.45e8))
    q = jnp.array([1e-4, -2e-4, .001, -.002])
    def trace(q):
        r = lens(Ray.origin().derive(x=q[0], y=q[1], dx=q[2], dy=q[3]))
        return jnp.concatenate((r.r_xy, r.d_xy))
    out = trace(q)
    obj = q[:2]-distance*q[2:]
    # Independent stationarity condition in the object-to-pupil kernel.
    def kernel(o, a):
        return jnp.sum((a-o)**2)/(2*distance) + lens._phase_shift_at_object(a, o)
    go, ga = jax.grad(kernel, argnums=(0, 1))(obj, out[:2])
    np.testing.assert_allclose(-go, q[2:], atol=1e-14)
    np.testing.assert_allclose(ga, out[2:], atol=1e-14)
    matrix = jax.jacfwd(trace)(q)
    symplectic = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]])
    np.testing.assert_allclose(matrix.T@symplectic@matrix, symplectic, atol=1e-11)


@seidel_ray_not_implemented
def test_seidel_fixed_object_and_batched_rays():
    lens = SeidelLens(z=0., focal_length=.02, object_xy=(2e-5, -1e-5),
                      coeffs=SeidelCoeffs(B=1e8, F=2e8))
    rays = Ray.origin().derive(x=jnp.array([1e-4, -2e-4]), y=jnp.array([2e-4, 1e-4]))
    out = lens(rays)
    for i in range(2):
        ray = Ray.origin().derive(x=rays.x[i], y=rays.y[i])
        kick = jax.grad(lambda p: lens.phase_shift(p, ray.d_xy))(ray.r_xy)
        np.testing.assert_allclose(out.d_xy[i], ray.d_xy+kick, atol=1e-15)
        np.testing.assert_allclose(out.pathlength[i], lens.phase_shift(ray.r_xy, ray.d_xy), atol=1e-18)


@seidel_ray_not_implemented
def test_seidel_third_order_jet_of_component_at_one_axis_ray():
    # B gives a known cubic pupil kick, using the full stationary-ray branch.
    B, scale = 1.2e8, 1e-4
    lens = SeidelLens(z=0., focal_length=.02, object_plane_dist=.04,
                      coeffs=SeidelCoeffs(B=B))
    def trace(t):
        return lens(Ray.origin().derive(x=scale*t)).dx
    _, series = jet(trace, (jnp.array(0.),), ((jnp.array(1.), jnp.array(0.), jnp.array(0.)),))
    np.testing.assert_allclose(series[2], -24*B*scale**3, rtol=1e-11, atol=1e-16)
