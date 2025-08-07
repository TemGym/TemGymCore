import numpy as np
import jax.numpy as jnp
import jax_dataclasses as jdc

from .tree_utils import HasParamsMixin
from .ray import Ray
from .utils import concentric_rings, random_coords
from . import CoordsXY


class Source(HasParamsMixin):
    ...


@jdc.pytree_dataclass
class PointSource(Source):
    z: float
    semi_conv: float
    offset_xy: CoordsXY = (0.0, 0.0)

    def __call__(self, ray: Ray):
        return ray

    def generate(self, num: int, random: bool = False) -> np.ndarray:
        semi_conv = self.semi_conv
        offset_xy = self.offset_xy

        if random:
            y, x = random_coords(num) * semi_conv
        else:
            y, x = concentric_rings(num, semi_conv)

        r = np.zeros((x.size, 5), dtype=jnp.float64)  # x, y, theta_x, theta_y, 1

        r[:, 0] += offset_xy[0]
        r[:, 1] += offset_xy[1]
        r[:, 2] = x
        r[:, 3] = y
        r[:, 4] = 1.0

        return r


@jdc.pytree_dataclass
class ParallelBeam(Source):
    z: float
    radius: float
    offset_xy: CoordsXY = (0.0, 0.0)

    def __call__(self, ray: Ray):
        return ray

    def generate(self, num: int, random: bool = False) -> np.ndarray:
        radius = self.radius
        offset_xy = self.offset_xy

        if random:
            y, x = random_coords(num) * radius
        else:
            y, x = concentric_rings(num, radius)

        r = np.zeros((x.size, 5), dtype=jnp.float64)  # x, y, theta_x, theta_y, 1

        r[:, 0] = (x + offset_xy[0])
        r[:, 1] = (y + offset_xy[1])
        r[:, 2] = 0.
        r[:, 3] = 0.
        r[:, 4] = 1.0

        return r
