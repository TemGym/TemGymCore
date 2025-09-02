import jax.numpy as jnp
import numpy as np
from numba import njit


def custom_jacobian_matrix(ray_jac):
    """Convert a nested Ray-Jacobian Pytree to a 5×5 homogeneous matrix.

    Parameters
    ----------
    ray_jac : nested structure
        Result of `jax.jacobian` applied to a function returning `Ray`.

    Returns
    -------
    J : jnp.ndarray, shape (5, 5)
        Matrix with rows corresponding to [x, y, dx, dy, 1] derivatives.

    Notes
    -----
    Pure and JIT-friendly; structure mirrors `temgym_core.ray.Ray`. The final
    row keeps the homogeneous variable constant.

    Examples
    --------
    >>> import jax
    >>> from temgym_core.ray import Ray
    >>> f = lambda r: r
    >>> J = custom_jacobian_matrix(jax.jacobian(f)(Ray.origin()))
    >>> J.shape
    (5, 5)
    """
    return jnp.stack(
        [
            jnp.stack([ray_jac.x.x, ray_jac.x.y, ray_jac.x.dx, ray_jac.x.dy, ray_jac.x._one], axis=-1),
            jnp.stack([ray_jac.y.x, ray_jac.y.y, ray_jac.y.dx, ray_jac.y.dy, ray_jac.y._one], axis=-1),
            jnp.stack([ray_jac.dx.x, ray_jac.dx.y, ray_jac.dx.dx, ray_jac.dx.dy, ray_jac.dx._one], axis=-1),
            jnp.stack([ray_jac.dy.x, ray_jac.dy.y, ray_jac.dy.dx, ray_jac.dy.dy, ray_jac.dy._one], axis=-1),
            jnp.stack([ray_jac._one.x, ray_jac._one.y, ray_jac._one.dx, ray_jac._one.dy, ray_jac._one._one], axis=-1),
        ], axis=-1,
    )


@njit
def multi_cumsum_inplace(values, partitions, start):
    """Compute multiple cumulative sums in-place with resets per partition.

    Parameters
    ----------
    values : numpy.ndarray, shape (K,)
        Values to be cumulatively summed. Mutated in-place.
    partitions : numpy.ndarray, shape (R,)
        Number of elements in each partition.
    start : float
        Initial value for each partition.

    Returns
    -------
    None

    Notes
    -----
    Numba-compiled; operates in-place on NumPy arrays. Partitions must sum to
    len(values).
    """
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
    """Accumulate values into a 2D buffer at integer coordinates in-place.

    Parameters
    ----------
    px_y : numpy.ndarray, int
        Row indices.
    px_x : numpy.ndarray, int
        Column indices.
    mask : numpy.ndarray, bool
        Mask selecting valid entries to add.
    frame : numpy.ndarray, float32
        Values to add.
    buffer : numpy.ndarray, float32 or int
        Target accumulator; mutated in-place.

    Returns
    -------
    None

    Notes
    -----
    Numba-compiled; does bounds checking and mask filtering.
    """
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
    """Generate approximately uniform samples on concentric rings.

    Parameters
    ----------
    num_points_approx : int
        Approximate number of points to generate.
    radius : float
        Maximum radius in metres (or radians for angular distributions).

    Returns
    -------
    points : numpy.ndarray, shape (N, 2), float64
        Coordinates (y, x) within the disc of given radius.

    Notes
    -----
    Deterministic layout approximating uniform density.
    """
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
    """Draw uniformly random points within a unit-radius disc.

    Parameters
    ----------
    num : int
        Number of points to draw (approximate; uses rejection sampling).

    Returns
    -------
    yx : numpy.ndarray, shape (N, 2), float64
        Coordinates (y, x) in [-1, 1] within the unit disc.
    """
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
    """Ravel an array-like if possible, otherwise return as-is.

    Parameters
    ----------
    val : Any
        Input array-like or scalar.

    Returns
    -------
    out : Any
        Flattened array or original object.
    """
    try:
        return val.ravel()
    except AttributeError:
        return val


def try_reshape(val, maybe_has_shape):
    """Reshape a value to match a reference's shape if possible.

    Parameters
    ----------
    val : Any
        Array-like to reshape.
    maybe_has_shape : Any
        Reference object providing target `.shape` if present.

    Returns
    -------
    out : Any
        Reshaped array or original value.
    """
    try:
        return val.reshape(maybe_has_shape.shape)
    except AttributeError:
        return val
