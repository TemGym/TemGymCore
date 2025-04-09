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