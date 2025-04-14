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
    new_ray = Ray(
        x=ray.x + distance * ray.dx,
        y=ray.y + distance * ray.dy,
        dx=ray.dx,
        dy=ray.dy,
        _one=1.0 * ray._one,
        z=ray.z * ray._one + distance,
        pathlength=ray.pathlength + distance,
    )
    return new_ray