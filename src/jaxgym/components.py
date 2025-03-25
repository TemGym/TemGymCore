import jax_dataclasses as jdc
import jax.numpy as jnp
from jax.numpy import ndarray as NDArray
from typing import (
    Tuple
)

from .ray import Ray, propagate, ray_matrix
from .utils import R2P, P2R, _identity, _flip_y, _rotate_deg_to_rad
from . import (
    Degrees,
)
from typing_extensions import TypeAlias

Radians: TypeAlias = jnp.float64  # type: ignore

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

        (descan_error_xx, descan_error_xy, descan_error_yx, descan_error_yy,
         descan_error_dxx, descan_error_dxy, descan_error_dyx, descan_error_dyy) = self.descan_error
        
        descan_error_xx = 1.0 + descan_error_xx
        descan_error_yy = 1.0 + descan_error_yy
        
        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy

        new_x = x * descan_error_xx + descan_error_xy * y + offset_x
        new_y = y * descan_error_yy + descan_error_yx * x + offset_y

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
    scan_rotation: jdc.Static[Degrees]
    scan_step: jdc.Static[float]
    scan_shape: jdc.Static[Tuple[int, int]]
    center: jdc.Static[Tuple[float, float]]= (0., 0.)
    coords: jdc.Static[NDArray] = jdc.field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "coords", self.get_coords())

    def step(self, ray: Ray):
        Ray = ray_matrix(ray.x, ray.y, ray.dx, ray.dy,
                        ray.z, ray.amplitude,
                        ray.pathlength, ray.wavelength,
                        ray.blocked)
        return Ray
    

    def get_coords(self):

        centre_x, centre_y = self.center
        scan_shape_y, scan_shape_x = self.scan_shape
        scan_step_y, scan_step_x = self.scan_step
        image_size_y = scan_shape_y * scan_step_y
        image_size_x = scan_shape_x * scan_step_x
        shape_y, shape_x = self.scan_shape

        y_image = jnp.linspace(-image_size_y / 2,
                               image_size_y / 2 - scan_step_y,
                               shape_y, endpoint=True) + centre_y
        
        x_image = jnp.linspace(-image_size_x / 2,
                               image_size_x / 2 - scan_step_x,
                               shape_x, endpoint=True) + centre_x

        y, x = jnp.meshgrid(y_image, x_image, indexing='ij')

        scan_rotation_rad = jnp.deg2rad(self.scan_rotation)

        pos_r, pos_a = R2P(x + y * 1j)
        pos_c = P2R(pos_r, pos_a + scan_rotation_rad)
        y_rot, x_rot = pos_c.imag, pos_c.real

        r = jnp.stack((y_rot, x_rot), axis=-1).reshape(-1, 2)

        return r
    

    def metres_to_pixels(self, yx: Tuple[float, float]) -> Tuple[int, int]:

        scan_step_y, scan_step_x = self.scan_step
        scan_shape_y, scan_shape_x = self.scan_shape

        scan_positions_y, scan_positions_x = yx
        scan_rotation = self.scan_rotation

        transform = _rotate_deg_to_rad(jnp.array(scan_rotation))

        y_transformed, x_transformed = (jnp.array((scan_positions_y, scan_positions_x)).T @ transform).T

        pixel_coords_x = ((x_transformed / scan_step_x) + (scan_shape_x // 2)).astype(jnp.int32)
        pixel_coords_y = ((y_transformed / scan_step_y) + (scan_shape_y // 2)).astype(jnp.int32)

        return pixel_coords_y, pixel_coords_x

    
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
    pixel_size: jdc.Static[float]
    shape: jdc.Static[Tuple[int, int]]
    center: jdc.Static[Tuple[float, float]] = (0., 0.)
    coords: jdc.Static[NDArray] = jdc.field(init=False)
    rotation: jdc.Static[Degrees] = 0.
    flip_y: jdc.Static[bool] = False
    
    def __post_init__(self):
        object.__setattr__(self, "coords", self.get_coords())


    def step(self, ray: Ray):
        return ray


    def get_coords(self):

        flip_y = self.flip_y
        det_rotation = self.rotation
        centre_x, centre_y = self.center
        det_shape_y, det_shape_x = self.shape
        pixel_size = self.pixel_size
        image_size_y = det_shape_y * pixel_size
        image_size_x = det_shape_x * pixel_size

        y_lin = jnp.linspace(-image_size_y / 2,
                       image_size_y / 2 - pixel_size,
                       det_shape_y, endpoint=True) + centre_y

        x_lin = jnp.linspace(-image_size_x / 2,
                       image_size_x / 2 - pixel_size,
                       det_shape_x, endpoint=True) + centre_x
        

        y, x = jnp.meshgrid(y_lin, x_lin, indexing='ij')

        # Shift the coordinates based on the detector center position
        det_positions_y_centred = y - centre_y
        det_positions_x_centred = x - centre_x

        # Create the transformation matrix for rotation of the coordinates
        if flip_y:
            transform = _flip_y()
        else:
            transform = _identity()

        transform = _rotate_deg_to_rad(jnp.array(det_rotation)) @ transform

        # Rotate the coordinates
        y_transformed, x_transformed = (jnp.array((det_positions_y_centred, det_positions_x_centred)).T @ transform).T

        # Shift the coordinates back to the original position
        y_transformed += centre_y
        x_transformed += centre_x

        r = jnp.stack((y_transformed, x_transformed), axis=-1).reshape(-1, 2)

        return r
    

    def metres_to_pixels(self, yx: Tuple[float, float]) -> Tuple[int, int]:
        flip_y = self.flip_y
        pixel_size = self.pixel_size
        det_shape_y, det_shape_x = self.shape
        det_positions_y, det_positions_x = yx
        det_rotation = self.rotation

        if flip_y:
            transform = _flip_y()
        else:
            transform = _identity()

        transform = _rotate_deg_to_rad(jnp.array(det_rotation)) @ transform

        y_transformed, x_transformed = (jnp.array((det_positions_y, det_positions_x)).T @ transform).T

        pixel_coords_x = ((x_transformed / pixel_size) + (det_shape_x // 2)).astype(jnp.int32)       
        pixel_coords_y = ((y_transformed / pixel_size) + (det_shape_y // 2)).astype(jnp.int32)        
        
        return pixel_coords_y, pixel_coords_x
