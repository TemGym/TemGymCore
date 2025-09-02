import numpy as np
import jax_dataclasses as jdc

from .tree_utils import HasParamsMixin
from .ray import Ray
from .utils import concentric_rings, random_coords
from . import CoordsXY


class Source(HasParamsMixin):
    """Base class for objects that create sets of initial rays.

    Attributes
    ----------
    z : float
        Axial source position in metres.
    """
    z: float

    def __call__(self, ray: Ray) -> Ray:
        """No-op for API uniformity with Component; returns the input ray.

        Parameters
        ----------
        ray : Ray
            Input ray.

        Returns
        -------
        ray : Ray
            Same ray.
        """
        return ray

    def generate_array(self, num: int, random: bool = False) -> np.ndarray:
        """Generate a (N, 5) array of initial rays [x, y, dx, dy, 1].

        Parameters
        ----------
        num : int
            Approximate number of rays to generate.
        random : bool, default False
            If True, generate a random distribution; otherwise deterministic.

        Returns
        -------
        rays : numpy.ndarray, shape (N, 5), float64
            Rows are [x_m, y_m, dx_rad, dy_rad, 1].

        Raises
        ------
        NotImplementedError
            If called on the base class.
        """
        raise NotImplementedError

    def make_rays(self, num: int, random: bool = False):
        """Build a `Ray` instance from a generated (N, 5) array.

        Parameters
        ----------
        num : int
            Approximate number of rays.
        random : bool, default False
            Generation mode; see `generate_array`.

        Returns
        -------
        rays : Ray
            Ray with vector fields of length N and z set to source z.
        """
        r = self.generate_array(num, random=random)
        sl = 0 if r.shape[0] == 1 else slice(None)  # if only one ray, Ray will contain scalars
        x = r[sl, 0]
        y = r[sl, 1]
        dx = r[sl, 2]
        dy = r[sl, 3]
        return Ray(x=x, y=y, dx=dx, dy=dy, z=self.z, pathlength=0.)


@jdc.pytree_dataclass
class PointSource(Source):
    """Point source with semi-convergence angle around an offset.

    Parameters
    ----------
    z : float
        Axial position in metres.
    semi_conv : float
        Semi-convergence angle (radians).
    offset_xy : CoordsXY, default (0.0, 0.0)
        Position offset (x, y) in metres.

    Notes
    -----
    Deterministic mode uses concentric rings; random mode uses uniform
    disc sampling.
    """
    z: float
    semi_conv: float
    offset_xy: CoordsXY = CoordsXY(x=0.0, y=0.0)

    def generate_array(self, num: int, random: bool = False) -> np.ndarray:
        """Generate rays with varying slopes within a cone of semi-convergence.

        Parameters
        ----------
        num : int
            Approximate number of rays.
        random : bool, default False
            If True, use random placement on rings.

        Returns
        -------
        rays : numpy.ndarray, shape (N, 5), float64
            Rows are [x_m, y_m, dx_rad, dy_rad, 1].
        """
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
    """Parallel beam source filling a circular aperture of given radius.

    Parameters
    ----------
    z : float
        Axial position in metres.
    radius : float
        Aperture radius in metres.
    offset_xy : CoordsXY, default (0.0, 0.0)
        Position offset (x, y) in metres.

    Notes
    -----
    Generated rays have dx=dy=0 with varying positions.
    """
    z: float
    radius: float
    offset_xy: CoordsXY = (0.0, 0.0)

    def generate_array(self, num: int, random: bool = False) -> np.ndarray:
        """Generate uniform samples within a disc aperture.

        Parameters
        ----------
        num : int
            Approximate number of rays.
        random : bool, default False
            Randomized vs deterministic concentric sampling.

        Returns
        -------
        rays : numpy.ndarray, shape (N, 5), float64
            Rows are [x_m, y_m, 0, 0, 1].
        """
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
