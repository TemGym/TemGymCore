import jax_dataclasses as jdc
import jax.numpy as jnp
from jax.numpy import ndarray as NDArray
from typing import (
    Tuple
)

from .ray import Ray, propagate
from .coordinate_transforms import apply_transformation, pixels_to_metres_transform
from . import (
    Degrees, Coords_XY, Scale_YX, Coords_YX, Pixels_YX, Shape_YX
)
from typing_extensions import TypeAlias
from .ode import solve_ode
import abc
import jax_dataclasses as jdc
from jax.numpy import ndarray as NDArray
from .coordinate_transforms import apply_transformation, pixels_to_metres_transform

Radians: TypeAlias = jnp.float64  # type: ignore
EPS = 1e-12


@jdc.pytree_dataclass
class Lens:
    z: float
    focal_length: float

    def step(self, ray: Ray):
        f = self.focal_length

        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy

        new_dx = -x / f + dx
        new_dy = -y / f + dy

        pathlength = ray.pathlength - (x ** 2 + y ** 2) / (2 * f)
        one = ray._one * 1.0

        return Ray(x=x, y=y, dx=new_dx, dy=new_dy, _one=one, pathlength=pathlength, z=ray.z)
    

@jdc.pytree_dataclass
class ThickLens:
    z_po: float
    z_pi: float
    focal_length: float

    def step(self, ray: Ray):
        f = self.focal_length

        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy

        new_dx = -x / f + dx
        new_dy = -y / f + dy

        pathlength = ray.pathlength - (x ** 2 + y ** 2) / (2 * f)

        new_z = ray.z - (self.z_po - self.z_pi)

        one = ray._one * 1.0

        return Ray(x=x, y=y, dx=new_dx, dy=new_dy, _one=one, pathlength=pathlength, z=new_z)

    @property
    def z(self):
        return self.z_po


@jdc.pytree_dataclass
class Descanner:
    z: float
    offset_x: float
    offset_y: float
    descan_error: jnp.ndarray

    def step(self, ray: Ray):
        offset_x, offset_y = self.offset_x, self.offset_y

        (descan_error_xx, descan_error_xy, descan_error_yx, descan_error_yy,
         descan_error_dxx, descan_error_dxy, descan_error_dyx, descan_error_dyy) = self.descan_error

        descan_error_xx = 1.0 + descan_error_xx
        descan_error_yy = 1.0 + descan_error_yy

        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy

        new_x = x * descan_error_xx + descan_error_xy * y + offset_x
        new_y = y * descan_error_yy + descan_error_yx * x + offset_y

        new_dx = dx + x * descan_error_dxx + y * descan_error_dxy
        new_dy = dy + y * descan_error_dyy + x * descan_error_dyx

        one =  offset_x * x + offset_y * y

        return Ray(x=new_x, y=new_y, dx=new_dx, dy=new_dy, _one=one, pathlength=ray.pathlength, z=ray.z)


@jdc.pytree_dataclass
class ODE:
    z: float
    z_end: float
    phi_lambda: callable
    E_lambda: callable

    def step(self, ray: Ray) -> Ray:
        in_state = jnp.array([ray.x, ray.y, ray.dx, ray.dy, ray.pathlength])

        z_start = self.z
        z_end = self.z_end

        u0 = self.phi_lambda(0.0, 0.0, z_start).astype(jnp.float64)

        out_state, out_z = solve_ode(in_state, z_start, z_end, self.phi_lambda, self.E_lambda, u0)

        x, y, dx, dy, opl = out_state

        return Ray(x=x, y=y, dx=dx, dy=dy, _one=ray._one, pathlength=opl, z=out_z)
        

@jdc.pytree_dataclass
class Deflector:
    z: float
    def_x: float
    def_y: float

    def step(self, ray: Ray):

        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy
        new_dx = dx + self.def_x
        new_dy = dy + self.def_y

        pathlength = ray.pathlength + dx * x + dy * y

        return Ray(x=x, y=y, dx=new_dx, dy=new_dy, _one=ray._one, pathlength=pathlength, z=ray.z)


@jdc.pytree_dataclass
class Rotator:
    z: float
    angle: Degrees

    def step(self, ray: Ray):
            
        angle = jnp.deg2rad(self.angle)

        # Rotate the ray's position
        new_x = ray.x * jnp.cos(angle) - ray.y * jnp.sin(angle)
        new_y = ray.x * jnp.sin(angle) + ray.y * jnp.cos(angle)
        # Rotate the ray's slopes
        new_dx = ray.dx * jnp.cos(angle) - ray.dy * jnp.sin(angle)
        new_dy = ray.dx * jnp.sin(angle) + ray.dy * jnp.cos(angle)

        pathlength = ray.pathlength

        return Ray(x=new_x, y=new_y, dx=new_dx, dy=new_dy, _one=ray._one, pathlength=pathlength, z=ray.z)


@jdc.pytree_dataclass
class DoubleDeflector:
    z: float
    first: Deflector
    second: Deflector

    def step(self, ray: Ray):
        ray = self.first.step(ray)
        z_step = self.second.z - self.first.z
        ray = propagate(z_step, ray)
        ray = self.second.step(ray)

        return ray


@jdc.pytree_dataclass
class InputPlane:
    z: float   

    def step(self, ray: Ray):
        return ray
    

@jdc.pytree_dataclass
class PointSource:
    z: float   
    semi_conv: float

    def step(self, ray: Ray):
        return ray
    

@jdc.pytree_dataclass
class Biprism:
    z: float
    offset: float = 0.
    rotation: Degrees = 0.
    deflection: float = 0.

    def step(
        self, ray: Ray,
    ) -> Ray:

        pos_x, pos_y, dx, dy = ray.x, ray.y, ray.dx, ray.dy

        deflection = self.deflection
        offset = self.offset
        rot = jnp.deg2rad(self.rotation)

        rays_v = jnp.array([pos_x, pos_y]).T

        biprism_loc_v = jnp.array([offset*jnp.cos(rot), offset*jnp.sin(rot)])

        biprism_v = jnp.array([-jnp.sin(rot), jnp.cos(rot)])
        biprism_v /= jnp.linalg.norm(biprism_v)

        rays_v_centred = rays_v - biprism_loc_v

        dot_product = jnp.dot(rays_v_centred, biprism_v) / jnp.dot(biprism_v, biprism_v)
        projection = jnp.outer(dot_product, biprism_v)

        rejection = rays_v_centred - projection
        rejection = rejection/jnp.linalg.norm(rejection, axis=1, keepdims=True)

        # If the ray position is located at [zero, zero], rejection_norm returns a nan,
        # so we convert it to a zero, zero.
        rejection = jnp.nan_to_num(rejection)

        xdeflection_mag = rejection[:, 0]
        ydeflection_mag = rejection[:, 1]

        new_dx = (dx + xdeflection_mag * deflection).squeeze()
        new_dy = (dy + ydeflection_mag * deflection).squeeze()

        pathlength = ray.pathlength + (
            xdeflection_mag * deflection * pos_x + ydeflection_mag * deflection * pos_y
        )

        return Ray(x=pos_x.squeeze(), y=pos_y.squeeze(), dx=new_dx, dy=new_dy, _one=ray._one, pathlength=pathlength, z=ray.z)


# Base class for grid transforms
class GridBase(abc.ABC):
    metres_to_pixels_mat: jnp.ndarray
    pixels_to_metres_mat: jnp.ndarray

    def __post_init__(self):
        object.__setattr__(self, "metres_to_pixels_mat", self.get_metres_to_pixels_transform())
        object.__setattr__(self, "pixels_to_metres_mat", self.get_pixels_to_metres_transform())

    @property
    @abc.abstractmethod
    def pixel_size(self) -> Scale_YX:
        ...

    @property
    @abc.abstractmethod
    def shape(self) -> Shape_YX:
        ...

    @property
    @abc.abstractmethod
    def rotation(self) -> Degrees:
        ...

    @property
    @abc.abstractmethod
    def centre(self) -> Coords_XY:
        ...

    def get_coords(self) -> NDArray:
        shape = self.shape
        y_px = jnp.arange(shape[0])
        x_px = jnp.arange(shape[1])
        yy_px, xx_px = jnp.meshgrid(y_px, x_px, indexing='ij')
        yy_px = yy_px.ravel()
        xx_px = xx_px.ravel()
        coords_x, coords_y = self.pixels_to_metres((yy_px, xx_px))
        coords_xy = jnp.stack((coords_x, coords_y), axis=-1).reshape(-1, 2)
        return coords_xy

    def step(self, ray):
        return ray

    def get_metres_to_pixels_transform(self) -> NDArray:
        # Use the common transform using centre, pixel_size, shape and rotation.
        pixels_to_metres_mat = pixels_to_metres_transform(
            self.centre, self.pixel_size, self.shape, False, self.rotation
        )
        return jnp.linalg.inv(pixels_to_metres_mat)

    def get_pixels_to_metres_transform(self) -> NDArray:
        return pixels_to_metres_transform(
            self.centre, self.pixel_size, self.shape, False, self.rotation
        )

    def metres_to_pixels(self, coords: Coords_XY) -> Pixels_YX:
        coords_x, coords_y = coords
        pixels_y, pixels_x = apply_transformation(coords_y, coords_x, self.metres_to_pixels_mat)
        pixels_y = jnp.round(pixels_y).astype(jnp.int32)
        pixels_x = jnp.round(pixels_x).astype(jnp.int32)
        return pixels_y, pixels_x

    def pixels_to_metres(self, pixels: Pixels_YX) -> Coords_XY:
        pixels_y, pixels_x = pixels
        metres_y, metres_x = apply_transformation(pixels_y, pixels_x, self.pixels_to_metres_mat)
        return metres_x, metres_y

    @property
    def coords(self) -> NDArray:
        return self.get_coords()


@jdc.pytree_dataclass
class ImageGrid(GridBase):
    z: float
    image_pixel_size: Scale_YX
    image_shape: Shape_YX
    image_rotation: Degrees
    image_centre: Coords_XY = (0., 0.)
    image_array: jnp.ndarray = None  # Added image array variable specific to ImageGrid
    metres_to_pixels_mat: jnp.ndarray = jdc.field(init=False)
    pixels_to_metres_mat: jnp.ndarray = jdc.field(init=False)

    @property
    def pixel_size(self) -> Scale_YX:
        return self.image_pixel_size

    @property
    def shape(self) -> Shape_YX:
        return self.image_shape

    @property
    def rotation(self) -> Degrees:
        return self.image_rotation

    @property
    def centre(self) -> Coords_XY:
        return self.image_centre


@jdc.pytree_dataclass
class ScanGrid(GridBase):
    z: float
    scan_step: Scale_YX
    scan_shape: Shape_YX
    scan_rotation: Degrees
    scan_centre: Coords_XY = (0., 0.)
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
    def centre(self) -> Coords_XY:
        return self.scan_centre


@jdc.pytree_dataclass
class Detector(GridBase):
    z: float
    det_pixel_size: Scale_YX
    det_shape: Shape_YX
    det_centre: Coords_XY = (0., 0.)
    det_rotation: Degrees = 0.
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
        return self.det_rotation

    @property
    def centre(self) -> Coords_XY:
        return self.det_centre
