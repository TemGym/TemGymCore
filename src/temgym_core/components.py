from typing import NamedTuple
import jax_dataclasses as jdc
import jax.numpy as jnp

from .ray import Ray
from .grid import Grid
from . import Degrees, CoordsXY, ScaleYX, ShapeYX
from .tree_utils import HasParamsMixin


class Component(HasParamsMixin):
    """Base component that transforms a ray without side effects.

    Subclasses implement `__call__(ray) -> Ray`. Components are expected to be
    pure and differentiable with JAX.

    Notes
    -----
    All components include a `z` field specifying axial position in metres.
    Components do not change `ray.z`; free-space is handled by propagators.
    """
    def __call__(self, ray: Ray) -> Ray:
        raise NotImplementedError


class DescanError(NamedTuple):
    """Linear descan error coefficients as a function of scan position.

    The descanner introduces position and slope offsets linear in scan position
    (spx, spy). These coefficients parameterize the 5th column of a 5×5 ray
    transfer matrix.

    Parameters
    ----------
    pxo_pxi : float, default 0.0
        d(pos_x_out)/d(scan_pos_x), unitless.
    pxo_pyi : float, default 0.0
        d(pos_x_out)/d(scan_pos_y), unitless.
    pyo_pxi : float, default 0.0
        d(pos_y_out)/d(scan_pos_x), unitless.
    pyo_pyi : float, default 0.0
        d(pos_y_out)/d(scan_pos_y), unitless.
    sxo_pxi : float, default 0.0
        d(slope_x_out)/d(scan_pos_x), rad/m (paraxial small-angle).
    sxo_pyi : float, default 0.0
        d(slope_x_out)/d(scan_pos_y), rad/m.
    syo_pxi : float, default 0.0
        d(slope_y_out)/d(scan_pos_x), rad/m.
    syo_pyi : float, default 0.0
        d(slope_y_out)/d(scan_pos_y), rad/m.
    offpxi : float, default 0.0
        Constant pos_x offset at output, metres.
    offpyi : float, default 0.0
        Constant pos_y offset at output, metres.
    offsxi : float, default 0.0
        Constant slope_x offset at output, radians.
    offsyi : float, default 0.0
        Constant slope_y offset at output, radians.

    Notes
    -----
    Units assume scan positions are in metres in object space.
    TODO: Clarify units for s* coefficients if scan units differ.
    """
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
        """Return coefficients as a 1D array in fixed order.

        Returns
        -------
        coeffs : jnp.ndarray, shape (12,), float32
            Coefficients in the order defined by the NamedTuple.

        Notes
        -----
        Pure and JIT-friendly.
        """
        return jnp.array(self)

    def as_matrix(self) -> jnp.ndarray:
        """Build a 5×5 matrix encoding offsets in the 5th column.

        Returns
        -------
        M : jnp.ndarray, shape (5, 5), float32
            Matrix where the 5th column holds position/slope offsets
            parameterized by this error model.

        Notes
        -----
        Not used directly in the current implementation; provided for
        clarity and potential debugging.
        """
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
class Plane(Component):
    """No-op component located at a plane z.

    Parameters
    ----------
    z : float
        Axial position in metres.

    Notes
    -----
    Pure; returns the input ray unchanged.
    """
    z: float

    def __call__(self, ray: Ray):
        return ray


@jdc.pytree_dataclass
class Lens(Component):
    """Thin lens that changes slopes according to focal length.

    Parameters
    ----------
    z : float
        Axial position in metres.
    focal_length : float
        Focal length in metres. Positive focuses rays.

    Returns
    -------
    Ray
        Ray with updated slopes; positions unchanged at the lens plane.

    Notes
    -----
    Paraxial approximation: `dx' = dx - x/f`, `dy' = dy - y/f`.
    Pathlength increment follows a standard paraxial thin-lens phase term.
    """
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
class ScanGrid(Component, Grid):
    """Scanning grid defining pixel-to-metre mapping at plane z.

    Parameters
    ----------
    z : float
        Axial position in metres.
    pixel_size : ScaleYX
        Pixel size as (y, x) in metres/pixel.
    shape : ShapeYX
        Grid shape as (y, x) in pixels.
    rotation : Degrees, default 0.0
        Grid rotation in degrees, following coordinate transforms module.
    centre : CoordsXY, default (0.0, 0.0)
        Grid centre in metres (x, y).
    flip_y : bool, default False
        If True, flip the y-axis as in detector coordinates.

    Notes
    -----
    Provides coordinate conversion helpers via `Grid`.
    """
    z: float
    pixel_size: ScaleYX
    shape: ShapeYX
    rotation: Degrees = 0.
    centre: CoordsXY = (0., 0)
    flip_y: bool = False

    def __call__(self, ray: Ray):
        return ray


@jdc.pytree_dataclass
class Scanner(Component):
    """Apply scan position and tilt offsets to the ray.

    Parameters
    ----------
    z : float
        Axial position in metres.
    scan_pos_x : float
        Position offset in x, metres.
    scan_pos_y : float
        Position offset in y, metres.
    scan_tilt_x : float, default 0.0
        Slope offset in x, radians.
    scan_tilt_y : float, default 0.0
        Slope offset in y, radians.

    Notes
    -----
    Offsets are added to incoming ray fields.
    """
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
class Descanner(Component):
    """Apply linear descan error as a function of scan position and tilt.

    Parameters
    ----------
    z : float
        Axial position in metres.
    scan_pos_x : float
        Scan position x, metres.
    scan_pos_y : float
        Scan position y, metres.
    scan_tilt_x : float, default 0.0
        Scan tilt x, radians.
    scan_tilt_y : float, default 0.0
        Scan tilt y, radians.
    descan_error : DescanError, default DescanError()
        Linear error coefficients.

    Notes
    -----
    Implements the 5th-column offset of a ray transfer matrix parameterized
    by scan position (and compensates by subtracting scan/tilt). Pure and
    JIT-friendly.
    """
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
        in tem will return the complete 5x5 ray transfer matrix of the optical system
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
class Detector(Component, Grid):
    """Detector grid providing pixel<->metre conversions at plane z.

    Parameters
    ----------
    z : float
        Axial position in metres.
    pixel_size : ScaleYX
        Pixel size as (y, x) in metres/pixel.
    shape : ShapeYX
        Detector shape (y, x) in pixels.
    rotation : Degrees, default 0.0
        Rotation of detector axes in degrees.
    centre : CoordsXY, default (0.0, 0.0)
        Detector centre in metres (x, y).
    flip_y : bool, default False
        If True, flip the y-axis to match display conventions.

    Notes
    -----
    The component itself is a no-op; conversions are on the `Grid` base.
    """
    z: float
    pixel_size: ScaleYX
    shape: ShapeYX
    rotation: Degrees = 0.
    centre: CoordsXY = (0., 0)
    flip_y: bool = False

    def __call__(self, ray: Ray):
        return ray


@jdc.pytree_dataclass
class ThickLens(Component):
    """Thick lens with separate object/image planes and paraxial update.

    Parameters
    ----------
    z_po : float
        Object-side axial position, metres.
    z_pi : float
        Image-side axial position, metres.
    focal_length : float
        Effective focal length, metres.

    Notes
    -----
    Updates slopes as a thin lens and adjusts z by (z_pi - z_po). Pathlength
    updated with a standard paraxial term.
    """
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
        """Return the object-side axial position z_po in metres."""
        return self.z_po


@jdc.pytree_dataclass
class Deflector(Component):
    """Add constant deflections (slopes) to the ray.

    Parameters
    ----------
    z : float
        Axial position in metres.
    def_x : float
        Deflection in x, radians.
    def_y : float
        Deflection in y, radians.

    Notes
    -----
    Pathlength is incremented by dx*x + dy*y (paraxial surrogate).
    """
    z: float
    def_x: float
    def_y: float

    def __call__(self, ray: Ray):
        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy
        return ray.derive(
            dx=dx + self.def_x,
            dy=dy + self.def_y,
            pathlength=ray.pathlength + dx * x + dy * y,
        )


@jdc.pytree_dataclass
class Rotator(Component):
    """Rotate positions and slopes by a given angle around the optical axis.

    Parameters
    ----------
    z : float
        Axial position in metres.
    angle : Degrees
        Rotation angle in degrees.

    Notes
    -----
    Applies the same rotation to (x, y) and (dx, dy).
    """
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
class Biprism(Component):
    """Simulate a biprism that deflects rays away from a line.

    Parameters
    ----------
    z : float
        Axial position in metres.
    offset : float, default 0.0
        Distance of the biprism line from the optical axis, metres.
    rotation : Degrees, default 0.0
        Rotation of the biprism line, degrees.
    deflection : float, default 0.0
        Deflection magnitude applied orthogonal to the line, radians.

    Notes
    -----
    When a ray sits exactly on the line, the rejection direction is
    undefined; NaNs are replaced by zeros. The paraxial pathlength
    increment is proportional to deflection·pos.
    """
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

        # deflection = self.deflection
        # offset = self.offset
        # rot = jnp.deg2rad(self.rotation)

        # rays_v = jnp.array([pos_x, pos_y]).T

        # biprism_loc_v = jnp.array([offset * jnp.cos(rot), offset * jnp.sin(rot)])

        # biprism_v = jnp.array([-jnp.sin(rot), jnp.cos(rot)])
        # biprism_v /= jnp.linalg.norm(biprism_v)

        # rays_v_centred = rays_v - biprism_loc_v

        # dot_product = jnp.dot(rays_v_centred, biprism_v) / jnp.dot(biprism_v, biprism_v)
        # projection = jnp.outer(dot_product, biprism_v)

        # rejection = rays_v_centred - projection
        # rejection = rejection / jnp.linalg.norm(rejection, axis=1, keepdims=True)

        # # If the ray position is located at [zero, zero], rejection_norm returns a nan,
        # # so we convert it to a zero, zero.
        # rejection = jnp.nan_to_num(rejection)

        # xdeflection_mag = rejection[:, 0]
        # ydeflection_mag = rejection[:, 1]

        new_dx = dx + deflection * ray._one * jnp.sign(ray.x)

        pathlength = ray.pathlength + (
            1.0 * deflection * pos_x + 0.0 * deflection * pos_y
        )

        return Ray(
            x=pos_x,
            y=pos_y,
            dx=new_dx,
            dy=dy,
            _one=ray._one,
            pathlength=pathlength,
            z=ray.z,
        )