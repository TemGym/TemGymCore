from jax.numpy import ndarray as NDArray
import jax
import jax.numpy as jnp
import numpy as np
from . import Degrees, Radians
from jax.flatten_util import ravel_pytree

from scipy.constants import e, m_e, h
from typing import (
    Tuple, TypeAlias
)

RadiansJNP = jnp.float64

def _identity():
    # From libertem.corrections.coordinates v0.11.1
    return jnp.eye(3)


def _rotate(radians: 'Radians'):
    # From libertem.corrections.coordinates v0.11.1
    # https://en.wikipedia.org/wiki/Rotation_matrix
    # y, x instead of x, y
    # This function is written with the y-axis flipped - i.e a pi/2 rotation sends
    # a (1, 0) vector to (0, -1). This is the opposition direction to what one 
    # would expect in a conventional cartesian coordinate system.
    return jnp.array([
        (jnp.cos(radians), jnp.sin(radians), 0.),
        (-jnp.sin(radians), jnp.cos(radians), 0.),
        (0., 0., 1.)
    ])


def _scale(pixel_size_yx):
    return jnp.array([
        (1 / pixel_size_yx[1], 0., 0.),
        (0., 1 / pixel_size_yx[0], 0.),
        (0., 0., 1.)
    ])


def _shift(centre_yx):
    return jnp.array([
        (1., 0, centre_yx[1]),
        (0., 1., centre_yx[0])
        (0., 0., 1.)
    ])


def _flip_y():
    # From libertem.corrections.coordinates v0.11.1
    return jnp.array([
        (-1., 0., 0.),
        (0., 1., 0.),
        (0., 0., 1.)
    ])


def metres_to_pixels_transform(centre_yx, 
                               pixel_size_yx, 
                               shape_yx, 
                               flip_y=False, 
                               rotation = 0. # Degrees
):
    if flip_y:
        flip_transform = _flip_y()
    else:
        flip_transform = _identity()

    centre_shift_transform = _shift(centre_yx)
    rotation_transform = _rotate_with_deg_to_rad(rotation)
    scale_transform = _scale(pixel_size_yx)
    pixel_shift_transform = _shift(shape_yx // 2)

    transform = pixel_shift_transform @ scale_transform @ rotation_transform @ flip_transform @ centre_shift_transform

    return transform


def rotation_shift_transform(centre_yx, 
                             flip_y=False, 
                             rotation = 0. # Degrees
):
    if flip_y:
        flip_transform = _flip_y()
    else:
        flip_transform = _identity()

    rotation_transform = _rotate_with_deg_to_rad(rotation)
    centre_shift_transform = _shift(centre_yx)

    transform = centre_shift_transform @ rotation_transform @ flip_transform

    return transform


def _rotate_with_deg_to_rad(degrees: 'Degrees'):
    # From libertem.corrections.coordinates v0.11.1
    return _rotate(jnp.pi / 180 * degrees)


def custom_jacobian_matrix(ray_jac):
    return jnp.array([[ray_jac.x.x, ray_jac.x.y, ray_jac.x.dx, ray_jac.x.dy, ray_jac.x._one],
                      [ray_jac.y.x, ray_jac.y.y, ray_jac.y.dx, ray_jac.y.dy, ray_jac.y._one],
                      [ray_jac.dx.x, ray_jac.dx.y, ray_jac.dx.dx, ray_jac.dx.dy, ray_jac.dx._one],
                      [ray_jac.dy.x, ray_jac.dy.y, ray_jac.dy.dx, ray_jac.dy.dy, ray_jac.dy._one],
                      [ray_jac._one.x, ray_jac._one.y, ray_jac._one.dx, ray_jac._one.dy, ray_jac._one._one]])


@jax.jit
def multi_cumsum_inplace(values, partitions, start):
    def body_fun(i, carry):
        vals, part_idx, part_count = carry
        current_len = partitions[part_idx]

        def reset_part(_):
            # move to the next partition, reset, set start
            new_vals = vals.at[i].set(start)
            return (new_vals, part_idx + 1, 0)

        def continue_part(_):
            # accumulate with previous value
            new_vals = vals.at[i].add(vals[i - 1])
            return (new_vals, part_idx, part_count + 1)

        return jax.lax.cond(part_count == current_len, reset_part, continue_part, None)

    values = values.at[0].set(start)
    values, _, _ = jax.lax.fori_loop(1, values.shape[0], body_fun, (values, 0, 0))
    return values


def concentric_rings(
    num_points_approx: int,
    radius: float,
):
    num_rings = max(
        1,
        int(jnp.floor((-1 + jnp.sqrt(1 + 4 * num_points_approx / jnp.pi)) / 2))
    )

    # Calculate the circumference of each ring
    num_points_kth_ring = jnp.round(
        2 * jnp.pi * jnp.arange(1, num_rings + 1)
    ).astype(int)
    num_rings = num_points_kth_ring.size
    points_per_unit = num_points_approx / num_points_kth_ring.sum()
    points_per_ring = jnp.round(num_points_kth_ring * points_per_unit).astype(int)

    # Make get the radii for the number of circles of rays we need
    radii = jnp.linspace(
        0, radius, num_rings + 1, endpoint=True,
    )[1:]
    div_angle = 2 * jnp.pi / points_per_ring

    params = jnp.stack((radii, div_angle), axis=0)

    # Cupy gave an error here saying that points_per_ring must not be an array
    repeats = points_per_ring

    all_params = jnp.repeat(params, repeats, axis=-1)
    multi_cumsum_inplace(all_params[1, :], points_per_ring, 0.)

    all_radii = all_params[0, :]
    all_angles = all_params[1, :]

    return (
        all_radii * jnp.sin(all_angles),
        all_radii * jnp.cos(all_angles),
    )


def fibonacci_spiral(
    nb_samples: int,
    radius: float,
    alpha=2,
    jnp=jnp,
):
    # From https://github.com/matt77hias/fibpy/blob/master/src/sampling.py
    # Fibonacci spiral sampling in a unit circle
    # Alpha parameter determines smoothness of boundary - default of 2 means a smooth boundary
    # 0 for a rough boundary.
    # Returns a tuple of y, x coordinates of the samples

    ga = jnp.pi * (3.0 - jnp.sqrt(5.0))

    # Boundary points
    jnp_boundary = jnp.round(alpha * jnp.sqrt(nb_samples))

    ii = jnp.arange(nb_samples)
    rr = jnp.where(
        ii > nb_samples - (jnp_boundary + 1),
        radius,
        radius * jnp.sqrt((ii + 0.5) / (nb_samples - 0.5 * (jnp_boundary + 1)))
    )
    rr[0] = 0.
    phi = ii * ga
    y = rr * jnp.sin(phi)
    x = rr * jnp.cos(phi)

    return y, x


def random_coords(num: int, jnp=jnp):
    # generate random points uniformly sampled in x/y
    # within a centred circle of radius 0.5
    # return (y, x)
    key = jax.random.PRNGKey(1)

    yx = jax.random.uniform(key, shape=(int(num * 1.28), 2), minval=-1, maxval=1)  # 1.28 =  4 / np.pi
    radii = jnp.sqrt((yx ** 2).sum(axis=1))
    mask = radii < 1
    yx = yx[mask, :]
    return (
        yx[:, 0],
        yx[:, 1],
    )


def calculate_wavelength(phi_0: float):
    return h / (2 * abs(e) * m_e * phi_0) ** (1 / 2)


def calculate_phi_0(wavelength: float):
    return h ** 2 / (2 * wavelength ** 2 * abs(e) * m_e)


def zero_phase_1D(u, idx_x):
    u_centre = u[idx_x]
    phase_difference = 0 - jnp.angle(u_centre)
    u = u * jnp.exp(1j * phase_difference)
    return u


def zero_phase(u, idx_x, idx_y):
    u_centre = u[idx_x, idx_y]
    phase_difference = 0 - jnp.angle(u_centre)
    u = u * jnp.exp(1j * phase_difference)
    return u
