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
        sp_x, sp_y = self.scan_pos_x, self.scan_pos_y

        (
            descan_error_xx,
            descan_error_xy,
            descan_error_yx,
            descan_error_yy,
            descan_error_dxx,
            descan_error_dxy,
            descan_error_dyx,
            descan_error_dyy,
            descan_error_offset_x,
            descan_error_offset_y,
            descan_error_offset_dx,
            descan_error_offset_dy,
        ) = self.descan_error

        x, y, dx, dy, _one = ray.x, ray.y, ray.dx, ray.dy, ray._one

        # Apply the negative of the input scan position to all rays equally, i.e descan the beam
        new_x = (
            x
            + (
                sp_x * descan_error_xx
                + sp_y * descan_error_xy
                + descan_error_offset_x
                - sp_x
            )
            * _one
        )
        new_y = (
            y
            + (
                sp_y * descan_error_yy
                + sp_y * descan_error_yx
                + descan_error_offset_y
                - sp_y
            )
            * _one
        )

        new_dx = (
            dx
            + (
                sp_x * descan_error_dxx
                + sp_y * descan_error_dxy
                + descan_error_offset_dx
            )
            * _one
        )
        new_dy = (
            dy
            + (
                sp_y * descan_error_dyy
                + sp_y * descan_error_dyx
                + descan_error_offset_dy
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
