import jax_dataclasses as jdc
import jax.numpy as jnp
from numpy.typing import NDArray
from typing import Tuple
from . import (
    PositiveFloat,
    Degrees,
)


@jdc.pytree_dataclass
class Ray:
    matrix: jnp.ndarray  # Shape (5,) vector [x, y, dx, dy, 1]
    z: float
    amplitude: float
    pathlength: float
    wavelength: float
    blocked: float

    @property
    def x(self):
        return self.matrix[..., 0]

    @property
    def y(self):
        return self.matrix[..., 1]

    @property
    def dx(self):
        return self.matrix[..., 2]

    @property
    def dy(self):
        return self.matrix[..., 3]


@jdc.pytree_dataclass
class GaussianRay(Ray):
    w0x: float = 1.0
    Rx: float = 0.0
    w0y: float = 1.0
    Ry: float = 0.0


def propagate(distance, ray: Ray):
    x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy

    new_x = x + dx * distance
    new_y = y + dy * distance

    pathlength = ray.pathlength + distance * jnp.sqrt(1 + dx ** 2 + dy ** 2)

    Ray = ray_matrix(new_x, new_y, dx, dy,
                    ray.z + distance, ray.amplitude,
                    pathlength, ray.wavelength,
                    ray.blocked)
    return Ray


def ray_matrix(x, y, dx, dy,
               z, amplitude,
               pathlength, wavelength,
               blocked):

    new_matrix = jnp.array([x, y, dx, dy, jnp.ones_like(x)]).T  # Doesnt work if all values have 0 shape
    return Ray(
        matrix=new_matrix,
        z=z,
        amplitude=amplitude,
        pathlength=pathlength,
        wavelength=wavelength,
        blocked=blocked
    )
