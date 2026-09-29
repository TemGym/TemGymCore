"""Seidel Gaussian transport checked against the object-to-pupil integral."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.aberrations import SeidelCoeffs
from temgym_core.components import DistortedLens, Lens, SeidelLens
from temgym_core.evaluate import evaluate_gaussians
from temgym_core.gaussian import (
    FreeSpacePropagator, _pupil_action_to_spatial, apply_action_delta, make_gaussian,
)


def field_at(beam, points):
    grid = SimpleNamespace(shape=(len(points),), coords=jnp.asarray(points))
    return np.asarray(evaluate_gaussians(beam, grid, method='scan')).ravel()


def pupil_quadrature(source, points, distance, action, width):
    """Direct quadrature; no Gaussian transport or Taylor code in the oracle."""
    axis = np.linspace(-6 * width, 6 * width, 121)
    xx, yy = np.meshgrid(axis, axis)
    offsets = np.column_stack((xx.ravel(), yy.ravel()))
    objects = offsets + np.asarray(source.r_xy)
    incoming = (
        float(source.pathlength) + offsets @ np.asarray(source.d_xy)
        + .5 * np.einsum('ni,ij,nj->n', offsets, np.asarray(source.Q_inv), offsets)
    )
    fields = []
    k = float(source.k)
    for pupil in np.array_split(points, 16):
        optical_path = (
            incoming[None, :]
            + np.sum((pupil[:, None, :] - objects[None, :, :])**2, axis=-1) / (2 * distance)
            + action(objects[None, :, :], pupil[:, None, :])
        )
        fields.append(np.exp(1j * k * optical_path).sum(axis=1))
    return (np.concatenate(fields) * complex(source.amplitude)
            * (axis[1] - axis[0])**2 * k / (2j * np.pi * distance))


def pupil_points(beam, count=25):
    sigma = 1 / np.sqrt(float(beam.k) * np.imag(np.asarray(beam.Q_inv)[0, 0]))
    axis = np.linspace(-2.5 * sigma, 2.5 * sigma, count)
    xx, yy = np.meshgrid(axis + float(beam.x), axis + float(beam.y))
    return np.column_stack((xx.ravel(), yy.ravel()))


@pytest.mark.parametrize('distance', [.02, -.02])
def test_joint_quadratic_action_matches_diffraction_integral(distance):
    width = 80e-9
    source = make_gaussian(x=.8e-6, y=.25e-6, dx=2e-5, dy=-1e-5,
                           waist_x=width, waist_y=width, z=0., voltage=200e3)
    beam = FreeSpacePropagator()(source, distance)
    value = 2e-12
    gradient = np.array([2e-6, -1e-6, 3e-6, 2e-6])
    hessian = np.array([[3., .4, 2., -.8], [.4, -2., .7, 1.],
                        [2., .7, -4., .2], [-.8, 1., .2, -3.]])
    terms = _pupil_action_to_spatial(beam, distance, value, gradient, hessian)
    xy, slope, amp, path, q = apply_action_delta(beam, *terms)
    out = beam.derive(x=xy[0], y=xy[1], dx=slope[0], dy=slope[1],
                      amplitude=amp, pathlength=path, Q_inv=q)

    def action(objects, pupil):
        shape = np.broadcast_shapes(objects.shape, pupil.shape)
        offsets = np.concatenate((np.broadcast_to(objects - source.r_xy, shape),
                                  np.broadcast_to(pupil - beam.r_xy, shape)), axis=-1)
        return value + offsets @ gradient + .5 * np.einsum('...i,ij,...j->...', offsets, hessian, offsets)

    points = pupil_points(beam)
    reference = pupil_quadrature(source, points, distance, action, width)
    np.testing.assert_allclose(field_at(out, points), reference, rtol=2e-9, atol=1e-11)


@pytest.mark.parametrize('radial,tangential', [(3e12, 0.), (0., 3e12)])
def test_distortion_converges_to_direct_complex_field_quadrature(radial, tangential):
    distance, focal = .02, .01
    errors = []
    for width in (80e-9, 40e-9, 20e-9):
        source = make_gaussian(x=.8e-6, y=.25e-6, dx=2e-5, dy=-1e-5,
                               waist_x=width, waist_y=width, z=0., voltage=200e3)
        beam = FreeSpacePropagator()(source, distance)
        lens = DistortedLens(z=distance, focal_length=focal, object_plane_dist=distance,
                             IsoDist=radial, AnisoDist=tangential)
        out = jax.jit(lambda b: lens(b))(beam)

        def action(objects, pupil):
            r2 = np.sum(objects**2, axis=-1)
            dot = np.sum(objects * pupil, axis=-1)
            cross = objects[..., 0] * pupil[..., 1] - objects[..., 1] * pupil[..., 0]
            return -np.sum(pupil**2, axis=-1) / (2 * focal) - r2 * (radial * dot + tangential * cross)

        points = pupil_points(beam)
        reference = pupil_quadrature(source, points, distance, action, width)
        errors.append(np.linalg.norm(field_at(out, points) - reference) / np.linalg.norm(reference))
    assert errors[2] < errors[1] < errors[0], errors
    assert errors[-1] < .003, errors


@pytest.mark.parametrize('distance', [0., .02, -.02])
def test_zero_seidel_matches_ideal_lens(distance):
    beam = make_gaussian(x=8e-6, y=-3e-6, dx=.001, dy=-.002,
                         waist_x=20e-9, waist_y=30e-9, z=0., voltage=200e3)
    lens = DistortedLens(z=0., focal_length=.01, object_plane_dist=distance,
                         x0=1e-6, y0=-2e-6)
    out = jax.jit(lambda b: lens(b))(beam)
    reference = Lens(z=0., focal_length=.01, x0=1e-6, y0=-2e-6)(beam)
    for name in ('r_xy', 'd_xy', 'Q_inv', 'amplitude', 'pathlength'):
        np.testing.assert_allclose(getattr(out, name), getattr(reference, name), rtol=1e-10, atol=1e-16)


def test_pupil_only_aberration_reduces_to_spatial_action():
    beam = make_gaussian(x=8e-6, y=-3e-6, dx=.001, dy=-.002,
                         waist_x=20e-9, waist_y=30e-9, z=0., voltage=200e3)
    coeffs = SeidelCoeffs(B=1e12)
    mixed = SeidelLens(z=0., focal_length=.01, object_plane_dist=.02, coeffs=coeffs)(beam)
    fixed = SeidelLens(z=0., focal_length=.01, object_xy=(0., 0.), coeffs=coeffs)(beam)
    for name in ('r_xy', 'd_xy', 'Q_inv', 'amplitude', 'pathlength'):
        np.testing.assert_allclose(getattr(mixed, name), getattr(fixed, name), rtol=1e-11, atol=1e-16)


def test_zero_object_distance_reduces_to_the_reparameterized_spatial_phase():
    from temgym_core.gaussian import taylor_expand

    beam = make_gaussian(x=8e-6, y=-3e-6, dx=.001, dy=-.002,
                         waist_x=20e-9, waist_y=30e-9, z=0., voltage=200e3)
    lens = SeidelLens(z=0., focal_length=.01, object_plane_dist=0.,
                      coeffs=SeidelCoeffs(A=2e9, B=3e8, E=1e12, e=2e11))
    terms = taylor_expand(lambda xy: lens.phase_shift(xy, beam.d_xy), beam.r_xy)
    expected = apply_action_delta(beam, *terms)
    actual = lens(beam)
    for value, reference in zip((actual.r_xy, actual.d_xy, actual.amplitude,
                                 actual.pathlength, actual.Q_inv), expected):
        np.testing.assert_allclose(value, reference, rtol=1e-11, atol=1e-16)


def test_distortion_centres_follow_cubic_law_for_a_batched_field():
    distance, focal = .02, .01
    source = make_gaussian(x=jnp.array([.8e-6, -.7e-6, .2e-6]),
                           y=jnp.array([.25e-6, -.3e-6, .6e-6]),
                           dx=jnp.array([1e-5, -2e-5, 0.]), dy=2e-5,
                           waist_x=40e-9, waist_y=40e-9, z=0., voltage=200e3)
    drift = FreeSpacePropagator()
    lens = DistortedLens(z=distance, focal_length=focal, object_plane_dist=distance,
                         IsoDist=-3e12, AnisoDist=2e12)
    out = jax.jit(lambda b: drift(lens(drift(b, distance)), distance))(source)
    obj = np.asarray(source.r_xy)
    kick = np.sum(obj**2, axis=1)[:, None] * (-3e12 * obj + 2e12 * np.c_[-obj[:, 1], obj[:, 0]])
    np.testing.assert_allclose(out.r_xy, -obj - distance * kick, rtol=1e-11, atol=1e-18)


@pytest.mark.parametrize('distortion', [-5e10, -1e11])
def test_negative_projector_distortion_preserves_decaying_gaussians(distortion):
    """Near a distortion fold, rotation/drift must not turn decay into growth."""
    from temgym_core.components import ABCDTransfer, HardEdgeSolenoid
    from temgym_core.gaussian import GaussianBeam

    distance = .0023032826918525534
    # Two beamlets from the failing 500,000-Gaussian microscope image;
    # rescale their coordinates to probe the same fold at either coefficient.
    scale = jnp.sqrt(5e10 / abs(distortion))
    position = scale*jnp.asarray([[2.4082105728813265e-5, -4.788765146492331e-5],
                                  [3.341067852639786e-5, -4.191540933314771e-5]])
    slope = scale*jnp.asarray([[-3.858622473851133e-5, 7.672932352482012e-5],
                               [-5.3533190361475685e-5, 6.716013220556671e-5]])
    cross = .889268526762256+10.236644481126522j
    curvature = jnp.asarray([
        [-.08206534534012783+23.5492040449277j, cross],
        [cross, -.9017741802664251+14.113282708780778j]])
    beam = GaussianBeam(x=position[:, 0], y=position[:, 1],
                        dx=slope[:, 0], dy=slope[:, 1],
                        z=.7329834964770994, amplitude=1e-5+0j,
                        pathlength=0., Q_inv=curvature, voltage=200e3)
    lens = DistortedLens(z=beam.z, focal_length=.002286337233069289,
                         object_plane_dist=distance, IsoDist=distortion)
    rotation = jnp.asarray([[jnp.cos(.37), -jnp.sin(.37)],
                            [jnp.sin(.37), jnp.cos(.37)]])
    rotate = ABCDTransfer(A=rotation, B=jnp.zeros((2, 2)), D=rotation)
    drift = FreeSpacePropagator()

    @jax.jit
    def transport(beam):
        pupil = lens(beam)
        rotated = rotate(pupil)
        finite = HardEdgeSolenoid(z=lens.z, length=.001, q=-.63)(pupil)
        return pupil, rotated, drift(rotated, 1.04375-lens.z), finite

    for out in transport(beam):
        q = np.asarray(out.Q_inv)
        assert np.isfinite(q).all()
        assert (np.linalg.eigvalsh(q.imag) > 0).all()
        assert np.isfinite(np.asarray(out.amplitude)).all()
        values = field_at(out, np.asarray(out.r_xy))
        assert np.isfinite(values).all()
