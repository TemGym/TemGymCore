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
    x: float
    y: float
    dx: float
    dy: float
    _one: 1.0
    z: float
    pathlength: float


def propagate(distance, ray):
    N = jnp.sqrt(1 + ray.dx ** 2 + ray.dy ** 2)
    L = ray.dx / N
    M = ray.dy / N
    

    opl = distance*N

    new_ray = Ray(
        x=ray.x + L / N * distance,
        y=ray.y + M / N * distance,
        dx=ray.dx,
        dy=ray.dy,
        _one=1.0 * ray._one,
        z=ray.z * ray._one + distance,
        pathlength=ray.pathlength + opl,
    )
    return new_ray