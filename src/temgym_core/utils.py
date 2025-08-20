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
        radius * jnp.sqrt((ii + 0.5) / (nb_samples - 0.5 * (jnp_boundary + 1))),
    )
    rr[0] = 0.0
    phi = ii * ga
    y = rr * jnp.sin(phi)
    x = rr * jnp.cos(phi)

    return y, x


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


def calculate_wavelength(phi_0: float):
    return h / (2 * abs(e) * m_e * phi_0) ** (1 / 2)


def calculate_phi_0(wavelength: float):
    return h**2 / (2 * wavelength**2 * abs(e) * m_e)


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


@jdc.pytree_dataclass
# A component that should give a singular jacobian used for testing
class SingularComponent:
    def __call__(self, ray: Ray):
        new_x = ray.x
        new_y = ray.x
        return Ray(
            x=new_x,
            y=new_y,
            dx=ray.dx,
            dy=ray.dy,
            _one=ray._one,
            pathlength=ray.pathlength,
            z=ray.z,
        )


def smiley(size):
    """
    Smiley face test object from https://doi.org/10.1093/micmic/ozad021
    """
    obj = np.ones((size, size), dtype=np.complex64)
    y, x = np.ogrid[-size // 2: size // 2, -size // 2: size // 2]

    outline = (((y * 1.2) ** 2 + x**2) > (110 / 256 * size) ** 2) & (
        ((y * 1.2) ** 2 + x**2) < (120 / 256 * size) ** 2
    )
    obj[outline] = 0.0

    left_eye = ((y + 40 / 256 * size) ** 2 + (x + 40 / 256 * size) ** 2) < (
        20 / 256 * size
    ) ** 2
    obj[left_eye] = 0
    right_eye = (np.abs(y + 40 / 256 * size) < 15 / 256 * size) & (
        np.abs(x - 40 / 256 * size) < 30 / 256 * size
    )
    obj[right_eye] = 0

    nose = (y + 20 / 256 * size + x > 0) & (x < 0) & (y < 10 / 256 * size)

    obj[nose] = (0.05j * x + 0.05j * y)[nose]

    mouth = (
        (((y * 1) ** 2 + x**2) > (50 / 256 * size) ** 2)
        & (((y * 1) ** 2 + x**2) < (70 / 256 * size) ** 2)
        & (y > 20 / 256 * size)
    )

    obj[mouth] = 0

    tongue = (
        ((y - 50 / 256 * size) ** 2 + (x - 50 / 256 * size) ** 2)
        < (20 / 256 * size) ** 2
    ) & ((y**2 + x**2) > (70 / 256 * size) ** 2)
    obj[tongue] = 0

    # This wave modulation introduces a strong signature in the diffraction pattern
    # that allows to confirm the correct scale and orientation.
    signature_wave = np.exp(1j * (3 * y + 7 * x) * 2 * np.pi / size)

    obj += 0.3 * signature_wave - 0.3

    obj = np.abs(obj)

    return obj


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
