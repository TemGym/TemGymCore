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


def P2R(radii: NDArray,
        angles: NDArray) -> NDArray:
    return radii * jnp.exp(1j*angles)


def R2P(x: NDArray) -> Tuple[NDArray,
                             NDArray]:
    return jnp.abs(x), jnp.angle(x)

def _flip_y():
    # From libertem.corrections.coordinates v0.11.1
    return jnp.array([
        (-1., 0.),
        (0., 1.)
    ])


def _identity():
    # From libertem.corrections.coordinates v0.11.1
    return jnp.eye(2)


def _rotate(radians: 'Radians'):
    # From libertem.corrections.coordinates v0.11.1
    # https://en.wikipedia.org/wiki/Rotation_matrix
    # y, x instead of x, y
    return jnp.array([
        (jnp.cos(radians), jnp.sin(radians)),
        (-jnp.sin(radians), jnp.cos(radians))
    ])


def _rotate_deg_to_rad(degrees: 'Degrees'):
    # From libertem.corrections.coordinates v0.11.1
    return _rotate(jnp.pi / 180 * degrees)


def metres_to_pixels(
    rays_x, rays_y, shape, pixel_size, flip_y=1, rotation = 0.
):
    if flip_y:
        transform = _flip_y()
    else:
        transform = _identity()

    # Transformations are applied right to left
    transform = _rotate_deg_to_rad(jnp.array(rotation), jnp) @ transform

    y_transformed, x_transformed = (jnp.array((rays_y, rays_x)).T @ transform).T

    sy, sx = shape
    pixel_coords_x = (x_transformed / pixel_size) + (sx // 2)
    pixel_coords_y = (y_transformed / pixel_size) + (sy // 2)

    return (pixel_coords_x, pixel_coords_y)
    


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
