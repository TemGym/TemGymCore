import jax.numpy as jnp
import numpy as np
from scipy.constants import e, m_e, h
from temgym_core.ray import Ray
import jax_dataclasses as jdc
from numba import njit


def custom_jacobian_matrix(ray_jac):
    return jnp.array(
        [
            [ray_jac.x.x, ray_jac.x.y, ray_jac.x.dx, ray_jac.x.dy, ray_jac.x._one],
            [ray_jac.y.x, ray_jac.y.y, ray_jac.y.dx, ray_jac.y.dy, ray_jac.y._one],
            [ray_jac.dx.x, ray_jac.dx.y, ray_jac.dx.dx, ray_jac.dx.dy, ray_jac.dx._one],
            [ray_jac.dy.x, ray_jac.dy.y, ray_jac.dy.dx, ray_jac.dy.dy, ray_jac.dy._one],
            [
                ray_jac._one.x,
                ray_jac._one.y,
                ray_jac._one.dx,
                ray_jac._one.dy,
                ray_jac._one._one,
            ],
        ]
    )


@njit
def multi_cumsum_inplace(values, partitions, start):
    part_idx = 0
    current_part_len = partitions[part_idx]
    part_count = 0
    values[0] = start
    for v_idx in range(1, values.size):
        if current_part_len == part_count:
            part_count = 0
            part_idx += 1
            current_part_len = partitions[part_idx]
            values[v_idx] = start
        else:
            values[v_idx] += values[v_idx - 1]
            part_count += 1


@njit
def inplace_sum(px_y, px_x, mask, frame, buffer):
    h, w = buffer.shape
    n = px_y.size
    for i in range(n):
        py = px_y[i]
        px = px_x[i]
        if mask[i] and (0 <= px_y[i] < h) and (0 <= px_x[i] < w):
            buffer[py, px] += frame[i]


def concentric_rings(
    num_points_approx: int,
    radius: float,
) -> np.ndarray:
    num_rings = max(
        1, int(np.floor((-1 + np.sqrt(1 + 4 * num_points_approx / np.pi)) / 2))
    )

    # Calculate the circumference of each ring
    num_points_kth_ring = np.round(2 * np.pi * np.arange(1, num_rings + 1)).astype(int)
    num_rings = num_points_kth_ring.size
    points_per_unit = num_points_approx / num_points_kth_ring.sum()
    points_per_ring = np.round(num_points_kth_ring * points_per_unit).astype(int)

    # Make get the radii for the number of circles of rays we need
    radii = np.linspace(
        0,
        radius,
        num_rings + 1,
        endpoint=True,
    )[1:]
    div_angle = 2 * np.pi / points_per_ring

    params = np.stack((radii, div_angle), axis=0)

    # Cupy gave an error here saying that points_per_ring must not be an array
    repeats = points_per_ring.tolist()

    all_params = np.repeat(params, repeats, axis=-1)
    multi_cumsum_inplace(all_params[1, :], points_per_ring, 0.0)

    all_radii = all_params[0, :]
    all_angles = all_params[1, :]

    return np.stack(
        (
            all_radii * np.sin(all_angles),
            all_radii * np.cos(all_angles),
        ),
        axis=-1,
    )


def random_coords(num: int) -> np.ndarray:
    # generate random points uniformly sampled in x/y
    # within a centred circle of radius 0.5
    # return (y, x)
    yx = np.random.uniform(
        -1,
        1,
        size=(int(num * 1.28), 2),  # 4 / np.pi
    )
    radii = np.sqrt((yx**2).sum(axis=1))
    mask = radii < 1
    yx = yx[mask, :]
    return yx


def try_ravel(val):
    try:
        return val.ravel()
    except AttributeError:
        return val


def try_reshape(val, maybe_has_shape):
    try:
        return val.reshape(maybe_has_shape.shape)
    except AttributeError:
        return val
