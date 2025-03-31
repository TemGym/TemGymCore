import jax_dataclasses as jdc
import jax.numpy as jnp
from jax.numpy import ndarray as NDArray
from typing import (
    Tuple
)

from .ray import Ray, propagate, ray_matrix
from .coordinate_transforms import _pixels_to_metres, _metres_to_pixels
from . import (
    Degrees, Coords_XY, Scale_YX, Coords_YX, Pixels_YX, Shape_YX
)
from typing_extensions import TypeAlias

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

        Ray = ray_matrix(x, y, new_dx, new_dy,
                        ray.z, ray.amplitude,
                        pathlength, ray.wavelength,
                        ray.blocked)
        return Ray


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
        Ray = ray_matrix(x, y, new_dx, new_dy,
                        new_z, ray.amplitude,
                        pathlength, ray.wavelength,
                        ray.blocked)
        return Ray

    @property
    def z(self):
        return self.z_po

@jdc.pytree_dataclass
class Descanner:
    z: float
    descan_error: Tuple[float, float, float, float]  # Error in the scan position pos_x, y, tilt_x, y
    offset_x: float
    offset_y: float

    def step(self, ray: Ray):
        offset_x, offset_y = self.offset_x, self.offset_y

        (descan_error_xx, descan_error_xy, descan_error, descan_error_yy,
         descan_error_dxx, descan_error_dxy, descan_error_dyx, descan_error_dyy) = self.descan_error
        
        descan_error_xx = 1.0 + descan_error_xx
        descan_error_yy = 1.0 + descan_error_yy
        
        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy

        new_x = x * descan_error_xx + descan_error_xy * y + offset_x
        new_y = y * descan_error_yy + descan_error * x + offset_y

        new_dx = dx + x * descan_error_dxx + y * descan_error_dxy
        new_dy = dy + y * descan_error_dyy + x * descan_error_dyx
        
        pathlength = ray.pathlength - (offset_x * x) - (offset_y * y)

        Ray = ray_matrix(new_x, new_y, new_dx, new_dy,
                         ray.z, ray.amplitude,
                         pathlength, ray.wavelength,
                         ray.blocked)
        return Ray


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

        Ray = ray_matrix(x, y, new_dx, new_dy,
                        ray.z, ray.amplitude,
                        pathlength, ray.wavelength,
                        ray.blocked)
        return Ray

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

        Ray = ray_matrix(new_x, new_y, new_dx, new_dy,
                        ray.z, ray.amplitude,
                        pathlength, ray.wavelength,
                        ray.blocked)
        return Ray

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
class ScanGrid:
    z: float
    scan_centre: Coords_XY = (0., 0.)
    scan_step: Scale_YX
    scan_shape: Shape_YX
    scan_rotation: Degrees

    @property
    def coords(self) -> NDArray:
        return self.get_coords()


    def step(self, ray: Ray):
        return ray


    def get_coords(self) -> Coords_XY:

        scan_shape = self.scan_shape

        y_px = jnp.arange(scan_shape[0])
        x_px = jnp.arange(scan_shape[1])

        yy_px, xx_px = jnp.meshgrid(y_px, x_px, indexing='ij')

        yy_px = yy_px.ravel()
        xx_px = xx_px.ravel()

        scan_coords_x, scan_coords_y = self.pixels_to_metres((yy_px, xx_px))

        scan_coords_xy = jnp.stack((scan_coords_x, scan_coords_y), axis=-1).reshape(-1, 2)

        return scan_coords_xy
    

    def metres_to_pixels(self, coords: Coords_XY) -> Pixels_YX:
        centre = self.scan_centre
        step = self.scan_step
        shape = self.scan_shape
        rotation = self.scan_rotation

        pixels_y, pixels_x = _metres_to_pixels(coords, 
                                              centre,
                                              step,
                                              shape,
                                              rotation)


        return pixels_y, pixels_x


    def pixels_to_metres(self, pixels: Pixels_YX) -> Coords_XY:
        
        centre = self.scan_centre
        step = self.scan_step
        shape = self.scan_shape
        rotation = self.scan_rotation

        coords_x, coords_y = _pixels_to_metres(pixels, 
                                              centre,
                                              step,
                                              shape,
                                              rotation)
    
        return coords_x, coords_y
    
@jdc.pytree_dataclass
class Aperture:
    z: float
    radius: float
    x: float = 0.
    y: float = 0.

    def step(self, ray: Ray):

        pos_x, pos_y, pos_dx, pos_dy = ray.x, ray.y, ray.dx, ray.dy
        distance = jnp.sqrt(
            (pos_x - self.x) ** 2 + (pos_y - self.y) ** 2
        )
        # This code evaluates to 1 if the ray is blocked already,
        # even if the new ray is inside the aperture,
        # evaluates to 1 if the ray was not blocked before and is now,
        # and evaluates to 0 if the ray was not blocked before and is NOT now.
        blocked = jnp.where(distance > self.radius, 1, ray.blocked)

        Ray = ray_matrix(pos_x, pos_y, pos_dx, pos_dy,
                        ray.z, ray.amplitude,
                        ray.pathlength, ray.wavelength,
                        blocked)
        return Ray


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

        Ray = ray_matrix(pos_x.squeeze(), pos_y.squeeze(), new_dx, new_dy,
                        ray.z, ray.amplitude,
                        pathlength, ray.wavelength,
                        ray.blocked)
        return Ray


@jdc.pytree_dataclass
class Detector:
    z: float
    det_centre: Coords_XY = (0., 0.)
    det_pixel_size: Scale_YX
    det_shape: Shape_YX
    det_rotation: Degrees = 0.
    flip_y: bool = False
    

    @property
    def coords(self) -> NDArray:
        return self.get_coords()


    def step(self, ray: Ray):
        return ray


    def get_coords(self) -> Coords_XY:

        det_shape = self.det_shape

        y_px = jnp.arange(det_shape[0])
        x_px = jnp.arange(det_shape[1])

        yy_px, xx_px = jnp.meshgrid(y_px, x_px, indexing='ij')

        yy_px = yy_px.ravel()
        xx_px = xx_px.ravel()

        det_coords_x, det_coords_y = self.pixels_to_metres((yy_px, xx_px))

        det_coords_xy = jnp.stack((det_coords_x, det_coords_y), axis=-1).reshape(-1, 2)

        return det_coords_xy
    

    def metres_to_pixels(self, coords: Coords_XY) -> Pixels_YX:
        centre = self.det_centre
        step = self.det_pixel_size
        shape = self.det_shape
        rotation = self.det_rotation

        pixels_y, pixels_x = _metres_to_pixels(coords, 
                                              centre,
                                              step,
                                              shape,
                                              rotation)

        return pixels_y, pixels_x


    def pixels_to_metres(self, pixels: Pixels_YX) -> Coords_XY:
        
        det_centre = self.det_centre
        det_step = self.det_pixel_size
        det_shape = self.det_shape
        det_rotation = self.det_rotation

        coords_x, coords_y = _pixels_to_metres(pixels, 
                                              det_centre,
                                              det_step,
                                              det_shape,
                                              det_rotation)
    
        return coords_x, coords_y