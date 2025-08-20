import jax.numpy as jnp
import numpy as np

import jax_dataclasses as jdc
from jaxgym.utils import random_coords, concentric_rings
from jaxgym.ray import Ray
from jaxgym.coordinate_transforms import GridBase
from jaxgym import Degrees, Coords_XY, Scale_YX, Shape_YX


@jdc.pytree_dataclass
class PointSource:
    z: float
    semi_conv: float
    offset_xy: Coords_XY = (0.0, 0.0)

    def step(self, ray: Ray):
        return ray

    def generate(self, num_rays: int, random: bool = False):
        semi_conv = self.semi_conv
        offset_xy = self.offset_xy

        if random:
            y, x = random_coords(num_rays) * semi_conv
        else:
            y, x = concentric_rings(num_rays, semi_conv)

        r = np.zeros((len(x), 5), dtype=np.float64)  # x, y, theta_x, theta_y, 1

        r[:, 0] += offset_xy[0]
        r[:, 1] += offset_xy[1]
        r[:, 2] = x
        r[:, 3] = y
        r[:, 4] = 1.

        return r


@jdc.pytree_dataclass
class ScanGrid(GridBase):
    z: float
    scan_step: Scale_YX
    scan_shape: Shape_YX
    scan_rotation: Degrees

    @property
    def pixel_size(self) -> Scale_YX:
        return self.scan_step

    @property
    def shape(self) -> Shape_YX:
        return self.scan_shape

    @property
    def rotation(self) -> Degrees:
        return self.scan_rotation

    @property
    def flip(self) -> Coords_XY:
        return False


@jdc.pytree_dataclass
class Descanner:
    z: float
    scan_pos_x: float
    scan_pos_y: float
    descan_error: jnp.ndarray

    def step(self, ray: Ray):
        """
        The traditional 5x5 linear ray transfer matrix of an optical system is
               [Axx, Axy, Bxx, Bxy, pos_offset_x],
               [Ayx, Ayy, Byx, Byy, pos_offset_y],
               [Cxx, Cxy, Dxx, Dxy, slope_offset_x],
               [Cyx, Cyy, Dyx, Dyy, slope_offset_y],
               [0.0, 0.0, 0.0, 0.0, 1.0],
        Since the Descanner is designed to only shift or tilt the entire incoming beam,
        with a certain error as a function of scan position, we write the 5th column
        of the ray transfer matrix, which is designed to describe an offset in shift or tilt,
        as a linear function of the scan position (spx, spy) (ignoring scan tilt for now):
        Thus -
            pos_offset_x(spx, spy) = pxo_pxi * spx + pxo_pyi * spy + offpxi
            pos_offset_y(spx, spy) = pyo_pxi * spx + pyo_pyi * spy + offpyi
            slope_offset_x(spx, spy) = sxo_pxi * spx + sxo_pyi * spy + offsxi
            slope_offset_y(spx, spy) = syo_pxi * spx + syo_pyi * spy + offsyi
        which can be represented as another 5x5 transfer matrix that is used to populate
        the 5th column of the ray transfer matrix of the optical system. The jacobian call
        in jaxgym will return the complete 5x5 ray transfer matrix of the optical system
        with the total descan error included in the 5th column.
        """

        sp_x, sp_y = self.scan_pos_x, self.scan_pos_y

        (
            pxo_pxi,  # How position x output scales with respect to scan x position
            pxo_pyi,  # How position x output scales with respect to scan y position
            pyo_pxi,  # How position y output scales with respect to scan x position
            pyo_pyi,  # How position y output scales with respect to scan y position
            sxo_pxi,  # How slope x output scales with respect to scan x position
            sxo_pyi,  # How slope x output scales with respect to scan y position
            syo_pxi,  # How slope y output scales with respect to scan x position
            syo_pyi,  # How slope y output scales with respect to scan y position
            offpxi,  # Constant additive error in x position
            offpyi,  # Constant additive error in y position
            offsxi,  # Constant additive error in x slope
            offsyi,  # Constant additive error in y slope
        ) = self.descan_error

        x, y, dx, dy, _one = ray.x, ray.y, ray.dx, ray.dy, ray._one

        new_x = (
            x
            + (
                sp_x * pxo_pxi
                + sp_y * pxo_pyi
                + offpxi
                - sp_x
            )
            * _one
        )
        new_y = (
            y
            + (
                sp_x * pyo_pxi
                + sp_y * pyo_pyi
                + offpyi
                - sp_y
            )
            * _one
        )

        new_dx = (
            dx
            + (
                sp_x * sxo_pxi
                + sp_y * sxo_pyi
                + offsxi
            )
            * _one
        )
        new_dy = (
            dy
            + (
                sp_x * syo_pxi
                + sp_y * syo_pyi
                + offsyi
            )
            * _one
        )

        one = _one

        return Ray(
            x=new_x,
            y=new_y,
            dx=new_dx,
            dy=new_dy,
            _one=one,
            pathlength=ray.pathlength,
            z=ray.z,
        )


@jdc.pytree_dataclass
class Detector(GridBase):
    z: float
    det_pixel_size: Scale_YX
    det_shape: Shape_YX
    flip_y: bool = False

    @property
    def pixel_size(self) -> Scale_YX:
        return self.det_pixel_size

    @property
    def shape(self) -> Shape_YX:
        return self.det_shape

    @property
    def rotation(self) -> Degrees:
        return 0.

    @property
    def flip(self) -> bool:
        return self.flip_y
