import numpy as np
from typing import NamedTuple
import jax_dataclasses as jdc
import jax.numpy as jnp

from .ray import Ray, propagate
from .utils import random_coords, concentric_rings
from .coordinate_transforms import GridBase
from . import Degrees, CoordsXY, ScaleYX, ShapeYX
from .tree_utils import HasParamsMixin


class DescanError(NamedTuple):
    pxo_pxi: float = 0.0  # How position x output scales with respect to scan x position
    pxo_pyi: float = 0.0  # How position x output scales with respect to scan y position
    pyo_pxi: float = 0.0  # How position y output scales with respect to scan x position
    pyo_pyi: float = 0.0  # How position y output scales with respect to scan y position
    sxo_pxi: float = 0.0  # How slope x output scales with respect to scan x position
    sxo_pyi: float = 0.0  # How slope x output scales with respect to scan y position
    syo_pxi: float = 0.0  # How slope y output scales with respect to scan x position
    syo_pyi: float = 0.0  # How slope y output scales with respect to scan y position
    offpxi: float = 0.0  # Constant additive error in x position
    offpyi: float = 0.0  # Constant additive error in y position
    offsxi: float = 0.0  # Constant additive error in x slope
    offsyi: float = 0.0  # Constant additive error in y slope

    def as_array(self) -> jnp.ndarray:
        return jnp.array(self)

    def as_matrix(self) -> jnp.ndarray:
        # Not used but represents the equations in descanner()
        return jnp.array(
            [
                [self.pxo_pxi, self.pxo_pyi, 0.0, 0.0, self.offpxi],
                [self.pyo_pxi, self.pyo_pyi, 0.0, 0.0, self.offpyi],
                [self.sxo_pxi, self.sxo_pyi, 0.0, 0.0, self.offsyi],
                [self.syo_pxi, self.syo_pyi, 0.0, 0.0, self.offsyi],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )


@jdc.pytree_dataclass
class Plane(HasParamsMixin):
    z: float

    def __call__(self, ray: Ray):
        return ray


@jdc.pytree_dataclass
class PointSource(HasParamsMixin):
    z: float
    semi_conv: float
    offset_xy: CoordsXY = (0.0, 0.0)

    def __call__(self, ray: Ray):
        return ray

    def generate(self, num_rays: int, random: bool = False) -> np.ndarray:
        semi_conv = self.semi_conv
        offset_xy = self.offset_xy

        if random:
            y, x = random_coords(num_rays) * semi_conv
        else:
            y, x = concentric_rings(num_rays, semi_conv)

        r = np.zeros((x.size, 5), dtype=jnp.float64)  # x, y, theta_x, theta_y, 1

        r[:, 0] += offset_xy[0]
        r[:, 1] += offset_xy[1]
        r[:, 2] = x
        r[:, 3] = y
        r[:, 4] = 1.0

        return r


@jdc.pytree_dataclass
class ParallelBeam(HasParamsMixin):
    z: float
    radius: float
    offset_xy: CoordsXY = (0.0, 0.0)

    def __call__(self, ray: Ray):
        return ray

    def generate(self, num_rays: int, random: bool = False) -> np.ndarray:
        radius = self.radius
        offset_xy = self.offset_xy

        if random:
            y, x = random_coords(num_rays) * radius
        else:
            y, x = concentric_rings(num_rays, radius)

        r = np.zeros((x.size, 5), dtype=jnp.float64)  # x, y, theta_x, theta_y, 1

        r[:, 0] = (x + offset_xy[0])
        r[:, 1] = (y + offset_xy[1])
        r[:, 2] = 0.
        r[:, 3] = 0.
        r[:, 4] = 1.0

        return r


@jdc.pytree_dataclass
class Lens(HasParamsMixin):
    z: float
    focal_length: float

    def __call__(self, ray: Ray):
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

    def __call__(self, ray: Ray):
        return ray


@jdc.pytree_dataclass
class Scanner(HasParamsMixin):
    z: float
    scan_pos_x: float
    scan_pos_y: float
    scan_tilt_x: float = 0.
    scan_tilt_y: float = 0.

    def __call__(self, ray: Ray):
        return ray.derive(
            x=ray.x + self.scan_pos_x * ray._one,
            y=ray.y + self.scan_pos_y * ray._one,
            dx=ray.dx + self.scan_tilt_x * ray._one,
            dy=ray.dy + self.scan_tilt_y * ray._one,
        )


@jdc.pytree_dataclass
class Descanner(HasParamsMixin):
    z: float
    scan_pos_x: float
    scan_pos_y: float
    scan_tilt_x: float = 0.
    scan_tilt_y: float = 0.
    descan_error: DescanError = DescanError()

    def __call__(self, ray: Ray):
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

        de = self.descan_error
        sp_x, sp_y = self.scan_pos_x, self.scan_pos_y
        st_x, st_y = self.scan_tilt_x, self.scan_tilt_y

        return ray.derive(
            x=ray.x + (
                sp_x * de.pxo_pxi
                + sp_y * de.pxo_pyi
                + de.offpxi
                - sp_x
            ) * ray._one,
            y=ray.y + (
                sp_x * de.pyo_pxi
                + sp_y * de.pyo_pyi
                + de.offpyi
                - sp_y
            ) * ray._one,
            dx=ray.dx + (
                sp_x * de.sxo_pxi
                + sp_y * de.sxo_pyi
                + de.offsxi
                - st_x
            ) * ray._one,
            dy=ray.dy + (
                sp_x * de.syo_pxi
                + sp_y * de.syo_pyi
                + de.offsyi
                - st_y
            ) * ray._one
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

    def __call__(self, ray: Ray):
        return ray


@jdc.pytree_dataclass
class ThickLens(HasParamsMixin):
    z_po: float
    z_pi: float
    focal_length: float

    def __call__(self, ray: Ray):
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

    def __call__(self, ray: Ray):
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

    def __call__(self, ray: Ray):
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

    def __call__(self, ray: Ray):
        ray = self.first(ray)
        z_step = self.second.z - self.first.z
        ray = propagate(z_step, ray)
        ray = self.second(ray)

        return ray


@jdc.pytree_dataclass
class Biprism(HasParamsMixin):
    z: float
    offset: float = 0.0
    rotation: Degrees = 0.0
    deflection: float = 0.0

    def __call__(
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
