from typing import NamedTuple, Dict
import jax_dataclasses as jdc
import jax.numpy as jnp

from .ray import Ray
from .grid import Grid
from . import Degrees, CoordsXY, ScaleYX, ShapeYX
from .tree_utils import HasParamsMixin
from .aberrations import grad_W_krivanek, W_krivanek

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
class AberratedLensKrivanek(Lens):
    """Thin lens with Krivanek aberrations.

    Parameters
    ----------
    z : float
        Axial position in metres.
    focal_length : float
        Focal length in metres.
    aber_coeffs : jnp.ndarray

    """
    coeffs: Dict

    def __call__(self, ray: Ray):
        f = self.focal_length
        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy
        coeffs = self.coeffs

        # Paraxial thin lens
        ideal_dx = -x / f + dx
        ideal_dy = -y / f + dy

        alpha = jnp.hypot(ideal_dx, ideal_dy)    # radians
        phi = jnp.arctan2(ideal_dy, ideal_dx)    # radians

        dWx, dWy = grad_W_krivanek(ideal_dx, ideal_dy, coeffs)
        dux, duy = -dWx / f, -dWy / f

        aber_dx = ideal_dx + dux
        aber_dy = ideal_dy + duy

        pathlength = W_krivanek(alpha, phi, coeffs)
        one = ray._one * 1.0

        return Ray(
            x=x, y=y, dx=aber_dx, dy=aber_dy, _one=one, pathlength=pathlength, z=ray.z
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
    The inherited :attr:`Grid.extent` property returns the plot extent
    ``(xmin, xmax, ymin, ymax)`` in metres for convenience.
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
            dx=dx + self.def_x * ray._one,
            dy=dy + self.def_y * ray._one,
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
    def_x: float = 0.0
    side: int = 1

    def __call__(self, ray: Ray):
        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy
        return ray.derive(
            dx=dx + self.def_x * ray._one * jnp.sign(ray.x),
            dy=dy,
            pathlength=ray.pathlength + dx * x + dy * y,
        )


@jdc.pytree_dataclass
class RotatingLens(Lens):
    '''Lens that rotates the beam using the rotator component before applying the lens transformation.'''
    def __init__(self, z: float, focal_length: float, rotation: Degrees):
        super().__init__(z=z, focal_length=focal_length)
        self.rotation = rotation
        self.rotator = Rotator(z=z, angle=rotation)

    def __call__(self, ray: Ray):
        # First apply the rotation to the ray
        rotated_ray = self.rotator(ray)
        # Then apply the lens transformation to the rotated ray
        return super().__call__(rotated_ray)


@jdc.pytree_dataclass
class ElectromagneticLens(Component):
    """Electromagnetic lens with Glaser bell model parameterization.

    Models an unsaturated electromagnetic lens where focal length follows
    f = 1/(Cf·I₀²) and image rotation follows ψ = Kv·I₀. This component
    applies both the thin-lens refraction and accumulated image rotation
    in a single physically-motivated transformation.

    Parameters
    ----------
    z : float
        Axial position in metres.
    I0 : float
        Nominal excitation current in ampere-turns [AT].
    Cf : float
        Lens geometry constant in units [1/(AT²·m)].
        Encodes bore radius, gap width, pole-piece shape, and coil turns.
    Kv : float
        Rotation constant in units [rad/AT].
        Computed from accelerating voltage via `compute_Kv_from_voltage()`.

    Returns
    -------
    Ray
        Ray with updated slopes (lens action) and rotated position/slopes
        (image rotation).

    Notes
    -----
    **Focal Length:** f = 1/(Cf·I₀²)

    **Image Rotation:** ψ = Kv·I₀ [radians]

    **Physics:**
    Based on the Glaser bell model for unsaturated electromagnetic lenses.
    The focal power scales with excitation current squared, and accumulated
    image rotation is proportional to the integrated magnetic field (∝ I₀).

    **Wobble Experiments:**
    To model lens excitation variations (wobble), create multiple instances
    with varied I0 values. For example, with 1% wobble:
    - `ElectromagneticLens(z, I0=I0_nominal, Cf, Kv)`
    - `ElectromagneticLens(z, I0=I0_nominal*1.01, Cf, Kv)`
    - `ElectromagneticLens(z, I0=I0_nominal*1.02, Cf, Kv)`

    **Typical Values:**
    - I0: 10-10,000 AT (ampere-turns)
    - Cf: 10⁻⁶ to 10⁻⁴ [1/(AT²·m)]
    - Kv: ~10⁻⁸ to 10⁻⁷ [rad/AT] for 100-300 kV electrons
    - Focal length: 1 mm to 10 cm
    - Rotation per lens: milliradians to radians

    References
    ----------
    See examples/lens_inversion/n_lens_inversion.ipynb for parameter
    identification from measured transfer matrices.

    Examples
    --------
    >>> from temgym_core.constants import compute_Kv_from_voltage
    >>> # 200 kV electron microscope
    >>> Kv = compute_Kv_from_voltage(200e3)  # rad/AT
    >>> # Typical objective lens
    >>> lens = ElectromagneticLens(
    ...     z=0.0,
    ...     I0=5000.0,      # ampere-turns
    ...     Cf=5e-6,        # 1/(AT²·m)
    ...     Kv=Kv
    ... )
    >>> focal_length = lens.focal_length  # metres
    >>> rotation_rad = lens.rotation_angle  # radians
    """
    z: float
    I0: float
    Cf: float
    Kv: float

    @property
    def focal_length(self) -> float:
        """Compute focal length from Glaser model: f = 1/(Cf·I₀²).

        Returns
        -------
        float
            Focal length in metres.
        """
        return 1.0 / (self.Cf * self.I0**2)

    @property
    def rotation_angle(self) -> float:
        """Compute image rotation angle: ψ = Kv·I₀.

        Returns
        -------
        float
            Rotation angle in radians.
        """
        return self.Kv * self.I0

    def __call__(self, ray: Ray) -> Ray:
        """Apply thin-lens refraction followed by image rotation.

        The transformation sequence:
        1. Paraxial thin-lens: slopes updated by -position/focal_length
        2. Pathlength updated with paraxial phase term
        3. Rotation: (x,y,dx,dy) rotated by accumulated angle ψ

        Parameters
        ----------
        ray : Ray
            Input ray state.

        Returns
        -------
        Ray
            Transformed ray with lens and rotation applied.
        """
        # Extract ray fields
        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy
        f = self.focal_length

        # Apply thin-lens transformation
        new_dx = -x / f + dx
        new_dy = -y / f + dy

        # Update pathlength (paraxial phase)
        pathlength = ray.pathlength - (x**2 + y**2) / (2.0 * f)

        # Apply image rotation
        angle = self.rotation_angle
        cos_a = jnp.cos(angle)
        sin_a = jnp.sin(angle)

        # Rotate position
        rot_x = cos_a * x - sin_a * y
        rot_y = sin_a * x + cos_a * y

        # Rotate slopes
        rot_dx = cos_a * new_dx - sin_a * new_dy
        rot_dy = sin_a * new_dx + cos_a * new_dy

        one = ray._one * 1.0

        return Ray(
            x=rot_x,
            y=rot_y,
            dx=rot_dx,
            dy=rot_dy,
            _one=one,
            pathlength=pathlength,
            z=ray.z,
        )
