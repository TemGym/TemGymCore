import numpy as np
import jax_dataclasses as jdc

from .tree_utils import HasParamsMixin
from .ray import Ray
from .utils import concentric_rings, random_coords
from . import CoordsXY


class Source(HasParamsMixin):
    z: float

    def __call__(self, ray: Ray) -> Ray:
        # No-op for API uniformity with Component
        return ray

    def generate_array(self, num: int, random: bool = False) -> np.ndarray:
        raise NotImplementedError

    def make_rays(self, num: int, random: bool = False):
        r = self.generate_array(num, random=random)
        sl = 0 if r.shape[0] == 1 else slice(None)  # if only one ray, Ray will contain scalars
        x = r[sl, 0]
        y = r[sl, 1]
        dx = r[sl, 2]
        dy = r[sl, 3]
        return Ray(x=x, y=y, dx=dx, dy=dy, z=self.z, pathlength=0.)


@jdc.pytree_dataclass
class PointSource(Source):
    z: float
    semi_conv: float
    offset_xy: CoordsXY = (0.0, 0.0)

    def generate_array(self, num: int, random: bool = False) -> np.ndarray:
        semi_conv = self.semi_conv
        offset_xy = self.offset_xy

        if random:
            dyx = random_coords(num) * semi_conv
        else:
            dyx = concentric_rings(num, semi_conv)

        dy, dx = dyx.T

        r = np.zeros((dx.size, 5), dtype=np.float64)  # x, y, theta_x, theta_y, 1
        r[:, 0] += offset_xy[0]
        r[:, 1] += offset_xy[1]
        r[:, 2] = dx
        r[:, 3] = dy
        r[:, 4] = 1.0
        return r


@jdc.pytree_dataclass
class ParallelBeam(Source):
    z: float
    radius: float
    offset_xy: CoordsXY = (0.0, 0.0)

    def generate_array(self, num: int, random: bool = False) -> np.ndarray:
        radius = self.radius
        offset_xy = self.offset_xy

        if random:
            yx = random_coords(num) * radius
        else:
            yx = concentric_rings(num, radius)

        y, x = yx.T

        r = np.zeros((x.size, 5), dtype=np.float64)  # x, y, theta_x, theta_y, 1
        r[:, 0] = (x + offset_xy[0])
        r[:, 1] = (y + offset_xy[1])
        r[:, 4] = 1.0
        return r
