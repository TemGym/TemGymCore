import jax_dataclasses as jdc
import jax.numpy as jnp
import dataclasses

from .ray import Ray, PixelsRay, propagate
from .utils import random_coords, concentric_rings
from .coordinate_transforms import GridBase
from . import Degrees, CoordsXY, ScaleYX, ShapeYX
from .tree_utils import HasParamsMixin


@jdc.pytree_dataclass
class Plane(HasParamsMixin):
    z: float

    def step(self, ray: Ray):
        return ray


@jdc.pytree_dataclass
class PointSource(HasParamsMixin):
    z: float
    semi_conv: float
    offset_xy: CoordsXY = (0.0, 0.0)

    def step(self, ray: Ray):
        return ray

    def generate(self, num_rays: int, random: bool = False):
        semi_conv = self.semi_conv
        offset_xy = self.offset_xy

        if random:
            y, x = random_coords(num_rays) * semi_conv
        else:
            y, x = concentric_rings(num_rays, semi_conv)

        r = jnp.zeros((num_rays, 5), dtype=jnp.float64)  # x, y, theta_x, theta_y, 1

        r[:, 0] += offset_xy[0]
        r[:, 1] += offset_xy[1]
        r[:, 2] = x
        r[:, 3] = y
        r[:, 4] = 1.0

        return r


@jdc.pytree_dataclass
class Lens(HasParamsMixin):
    z: float
    focal_length: float

    def step(self, ray: Ray):
        f = self.focal_length

        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy

        new_dx = -x / f + dx
        new_dy = -y / f + dy

        pathlength = ray.pathlength - (x**2 + y**2) / (2 * f)
        one = ray._one * 1.0

        return Ray(
            x=x, y=y, dx=new_dx, dy=new_dy, _one=one, pathlength=pathlength, z=ray.z
        )


@jdc.pytree_dataclass
class ScanGrid(HasParamsMixin, GridBase):
    z: float
    scan_step: ScaleYX
    scan_shape: ShapeYX
    scan_rotation: Degrees

    @property
    def pixel_size(self) -> ScaleYX:
        return self.scan_step

    @property
    def shape(self) -> ShapeYX:
        return self.scan_shape

    @property
    def rotation(self) -> Degrees:
        return self.scan_rotation

    @property
    def flip(self) -> CoordsXY:
        return False

    def step(self, ray: Ray):
        return ray

    def from_pixels(self):
        return FromPixelsScanGrid(
            **dataclasses.asdict(self)
        )


@jdc.pytree_dataclass
class FromPixelsScanGrid(ScanGrid):
    def step(self, ray: Ray):
        x_t, y_t = self.pixels_to_metres((ray.y, ray.x))
        ray = Ray(
            x_t,
            y_t,
            dx=ray.dx,
            dy=ray.dy,
            z=ray.z,
            pathlength=ray.pathlength,
        )
        return super().step(ray)


@jdc.pytree_dataclass
class Descanner(HasParamsMixin):
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

        new_x = x + (sp_x * pxo_pxi + sp_y * pxo_pyi + offpxi - sp_x) * _one
        new_y = y + (sp_x * pyo_pxi + sp_y * pyo_pyi + offpyi - sp_y) * _one

        new_dx = dx + (sp_x * sxo_pxi + sp_y * sxo_pyi + offsxi) * _one
        new_dy = dy + (sp_x * syo_pxi + sp_y * syo_pyi + offsyi) * _one

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
class Detector(HasParamsMixin, GridBase):
    z: float
    det_pixel_size: ScaleYX
    det_shape: ShapeYX
    flip_y: bool = False

    @property
    def pixel_size(self) -> ScaleYX:
        return self.det_pixel_size

    @property
    def shape(self) -> ShapeYX:
        return self.det_shape

    @property
    def rotation(self) -> Degrees:
        return 0.0

    @property
    def flip(self) -> bool:
        return self.flip_y

    def step(self, ray: Ray):
        return ray

    def to_pixels(self):
        return ToPixelsDetector(
            **dataclasses.asdict(self)
        )


@jdc.pytree_dataclass
class ToPixelsDetector(Detector):
    def step(self, ray: Ray):
        ray = super().step(ray)
        y_t, x_t = self.metres_to_pixels((ray.x, ray.y), cast=False)
        return PixelsRay(
            x_t,
            y_t,
            dx=ray.dx,
            dy=ray.dy,
            z=ray.z,
            pathlength=ray.pathlength,
        )


@jdc.pytree_dataclass
class ThickLens(HasParamsMixin):
    z_po: float
    z_pi: float
    focal_length: float

    def step(self, ray: Ray):
        f = self.focal_length

        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy

        new_dx = -x / f + dx
        new_dy = -y / f + dy

        pathlength = ray.pathlength - (x**2 + y**2) / (2 * f)

        new_z = ray.z - (self.z_po - self.z_pi)

        one = ray._one * 1.0

        return Ray(
            x=x, y=y, dx=new_dx, dy=new_dy, _one=one, pathlength=pathlength, z=new_z
        )

    @property
    def z(self):
        return self.z_po


@jdc.pytree_dataclass
class Deflector(HasParamsMixin):
    z: float
    def_x: float
    def_y: float

    def step(self, ray: Ray):
        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy
        new_dx = dx + self.def_x
        new_dy = dy + self.def_y

        pathlength = ray.pathlength + dx * x + dy * y

        return Ray(
            x=x,
            y=y,
            dx=new_dx,
            dy=new_dy,
            _one=ray._one,
            pathlength=pathlength,
            z=ray.z,
        )


@jdc.pytree_dataclass
class Rotator(HasParamsMixin):
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

        return Ray(
            x=new_x,
            y=new_y,
            dx=new_dx,
            dy=new_dy,
            _one=ray._one,
            pathlength=pathlength,
            z=ray.z,
        )


@jdc.pytree_dataclass
class DoubleDeflector(HasParamsMixin):
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
class Biprism(HasParamsMixin):
    z: float
    offset: float = 0.0
    rotation: Degrees = 0.0
    deflection: float = 0.0

    def step(
        self,
        ray: Ray,
    ) -> Ray:
        pos_x, pos_y, dx, dy = ray.x, ray.y, ray.dx, ray.dy

        deflection = self.deflection
        offset = self.offset
        rot = jnp.deg2rad(self.rotation)

        rays_v = jnp.array([pos_x, pos_y]).T

        biprism_loc_v = jnp.array([offset * jnp.cos(rot), offset * jnp.sin(rot)])

        biprism_v = jnp.array([-jnp.sin(rot), jnp.cos(rot)])
        biprism_v /= jnp.linalg.norm(biprism_v)

        rays_v_centred = rays_v - biprism_loc_v

        dot_product = jnp.dot(rays_v_centred, biprism_v) / jnp.dot(biprism_v, biprism_v)
        projection = jnp.outer(dot_product, biprism_v)

        rejection = rays_v_centred - projection
        rejection = rejection / jnp.linalg.norm(rejection, axis=1, keepdims=True)

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

        return Ray(
            x=pos_x.squeeze(),
            y=pos_y.squeeze(),
            dx=new_dx,
            dy=new_dy,
            _one=ray._one,
            pathlength=pathlength,
            z=ray.z,
        )
