import jax.numpy as jnp
import jax_dataclasses as jdc

from jaxgym.ray import Ray
from jaxgym.coordinate_transforms import GridBase
from jaxgym import Degrees, Coords_XY, Scale_YX, Shape_YX


@jdc.pytree_dataclass
class PointSource:
    z: float
    semi_conv: float

    def step(self, ray: Ray):
        return ray


@jdc.pytree_dataclass
class ScanGrid(GridBase):
    z: float
    scan_step: Scale_YX
    scan_shape: Shape_YX
    scan_rotation: Degrees
    metres_to_pixels_mat: jnp.ndarray = jdc.field(init=False)
    pixels_to_metres_mat: jnp.ndarray = jdc.field(init=False)

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
        the 5th column of the ray transfer matrix of the optical system.
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
    metres_to_pixels_mat: jnp.ndarray = jdc.field(init=False)
    pixels_to_metres_mat: jnp.ndarray = jdc.field(init=False)

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
