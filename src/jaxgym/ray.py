import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import Any

from . import (
    PositiveFloat,
    Degrees,
)

@jax.tree_util.register_pytree_node_class
@dataclass
class Ray:
    x: float
    y: float
    dx: float
    dy: float
    z: float
    pathlength: float
    # keys is included for batching but gradient is stopped via stop_gradient
    keys: Any = field(compare=False)

    wavelength: float = 1.0
    _one: float = 1.0

    def tree_flatten(self):
        # include keys in children for vmap, but stop its gradient
        children = (self.x, self.y, self.dx, self.dy,
                    self.z, self.pathlength, self.wavelength, self._one)
        aux = self.keys
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        x, y, dx, dy, z, pathlength, wavelength, _one = children
        keys = aux
        return cls(x=x, y=y, dx=dx, dy=dy,
                   z=z, pathlength=pathlength,
                   wavelength=wavelength, _one=_one,
                   keys=keys)

def propagate(distance, ray):
    new_ray = Ray(
        x=ray.x + ray.dx * distance,
        y=ray.y + ray.dy * distance,
        dx=ray.dx,
        dy=ray.dy,
        _one=ray._one,
        z=ray.z * ray._one + distance,
        pathlength=ray.pathlength + distance,
        wavelength=ray.wavelength,
        keys=ray.keys,
    )
    return new_ray