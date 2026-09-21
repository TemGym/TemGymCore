from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from numba import njit
import jax_dataclasses as jdc

from .tree_utils import HasParamsMixin
from .gaussian import LENGTH, GaussianBeam, make_gaussian
from .constants import energy2wavelength
from .ray import Ray
from . import CoordsXY

if TYPE_CHECKING:
    from interpax import Interpolator2D


def _electron_wavelength_in_unit(voltage: float, wavelength_unit: str) -> float:
    """Return the relativistic electron wavelength in the coordinate unit."""
    if wavelength_unit not in LENGTH:
        raise ValueError(
            f"Unknown wavelength unit {wavelength_unit!r}; "
            f"expected one of {tuple(LENGTH)}."
        )

    voltage = float(voltage)
    if not np.isfinite(voltage) or voltage <= 0.0:
        raise ValueError("voltage must be finite and positive.")
    wavelength_m = float(np.asarray(energy2wavelength(voltage)))
    return wavelength_m / LENGTH[wavelength_unit]


def _gaussian_cardinal_coefficients(
    target: np.ndarray,
    *,
    width_over_step: float,
    kernel_tolerance: float = 1e-13,
    ridge: float = 1e-10,
) -> np.ndarray:
    """Fit a square equal-width Gaussian lattice with a separable banded solve."""
    from scipy.linalg import solve_banded

    values = np.asarray(target)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("target must be a square two-dimensional array.")
    if values.shape[0] == 0:
        raise ValueError("target must not be empty.")

    axis = int(values.shape[0])
    alpha = float(width_over_step)
    kernel_tolerance = float(kernel_tolerance)
    ridge = float(ridge)
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("width_over_step must be finite and positive.")
    if not 0.0 < kernel_tolerance < 1.0:
        raise ValueError("kernel_tolerance must lie strictly between zero and one.")
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and non-negative.")

    bandwidth = min(
        axis - 1,
        int(np.ceil(alpha * np.sqrt(-np.log(kernel_tolerance)))),
    )
    kernel = np.exp(-(np.arange(bandwidth + 1, dtype=float) / alpha) ** 2)
    banded = np.zeros((2 * bandwidth + 1, axis), dtype=float)
    for offset in range(-bandwidth, bandwidth + 1):
        start = max(0, -offset)
        stop = min(axis, axis - offset)
        banded[bandwidth + offset, start:stop] = kernel[abs(offset)]
    banded[bandwidth] += ridge

    solve_dtype = np.result_type(values, np.float64)
    along_y = solve_banded(
        (bandwidth, bandwidth),
        banded,
        values.astype(solve_dtype, copy=False),
        check_finite=False,
    )
    return solve_banded(
        (bandwidth, bandwidth),
        banded,
        along_y.T,
        check_finite=False,
    ).T


def make_waist_divergence_rays(
    waist: float,
    *,
    voltage: float | None = None,
    wavelength: float | None = None,
    z: float = 0.0,
    x0: float = 0.0,
    y0: float = 0.0,
    pathlength: float = 0.0,
    divergence_sign: float = 1.0,
) -> Ray:
    """Construct a 2-ray waist/divergence paraxial basis bundle."""
    waist = float(waist)
    if waist <= 0:
        raise ValueError(f"`waist` must be > 0, got {waist}.")
    if (voltage is None) == (wavelength is None):
        raise ValueError("Provide exactly one of `voltage` or `wavelength`.")

    if wavelength is None:
        wavelength = float(np.asarray(energy2wavelength(float(voltage))))
    wavelength = float(wavelength)
    if wavelength <= 0:
        raise ValueError(f"`wavelength` must be > 0, got {wavelength}.")

    theta = float(divergence_sign) * wavelength / (np.pi * waist)
    x = np.asarray([float(x0) + waist, float(x0)], dtype=float)
    y = np.asarray([float(y0), float(y0)], dtype=float)
    dx = np.asarray([0.0, theta], dtype=float)
    dy = np.asarray([0.0, 0.0], dtype=float)
    z_vec = np.full((2,), float(z), dtype=float)
    path_vec = np.full((2,), float(pathlength), dtype=float)
    ray_voltage = float(voltage) if voltage is not None else 200e3
    return Ray(
        x=x,
        y=y,
        dx=dx,
        dy=dy,
        z=z_vec,
        pathlength=path_vec,
        voltage=np.full((2,), ray_voltage, dtype=float),
    )


class Source(HasParamsMixin):
    """Base class for objects that create sets of initial rays.

    Attributes
    ----------
    z : float
        Axial source position in metres.
    """
    z: float

    def __call__(self, ray: Ray) -> Ray:
        """No-op for API uniformity with Component; returns the input ray.

        Parameters
        ----------
        ray : Ray
            Input ray.

        Returns
        -------
        ray : Ray
            Same ray.
        """
        return ray

    def generate_array(self, num: int, random: bool = False) -> np.ndarray:
        """Generate a (N, 5) array of initial rays [x, y, dx, dy, 1].

        Parameters
        ----------
        num : int
            Approximate number of rays to generate.
        random : bool, default False
            If True, generate a random distribution; otherwise deterministic.

        Returns
        -------
        rays : numpy.ndarray, shape (N, 5), float64
            Rows are [x_m, y_m, dx_rad, dy_rad, 1].

        Raises
        ------
        NotImplementedError
            If called on the base class.
        """
        raise NotImplementedError

    def make_rays(self, num: int, random: bool = False):
        """Build a `Ray` instance from a generated (N, 5) array.

        Parameters
        ----------
        num : int
            Approximate number of rays.
        random : bool, default False
            Generation mode; see `generate_array`.

        Returns
        -------
        rays : Ray
            Ray with vector fields of length N and z set to source z.
        """
        r = self.generate_array(num, random=random)
        sl = 0 if r.shape[0] == 1 else slice(None)  # if only one ray, Ray will contain scalars
        x = r[sl, 0]
        y = r[sl, 1]
        dx = r[sl, 2]
        dy = r[sl, 3]
        return Ray(x=x, y=y, dx=dx, dy=dy, z=self.z, pathlength=0.)


@jdc.pytree_dataclass
class PointSource(Source):
    """Point source with semi-convergence angle around an offset.

    Parameters
    ----------
    z : float
        Axial position in metres.
    semi_conv : float
        Semi-convergence angle (radians).
    offset_xy : CoordsXY, default (0.0, 0.0)
        Position offset (x, y) in metres.

    Notes
    -----
    Deterministic mode uses concentric rings; random mode uses uniform
    disc sampling.
    """
    z: float
    semi_conv: float
    offset_xy: CoordsXY = (0.0, 0.0)

    def generate_array(self, num: int, random: bool = False) -> np.ndarray:
        """Generate rays with varying slopes within a cone of semi-convergence.

        Parameters
        ----------
        num : int
            Approximate number of rays.
        random : bool, default False
            If True, use random placement on rings.

        Returns
        -------
        rays : numpy.ndarray, shape (N, 5), float64
            Rows are [x_m, y_m, dx_rad, dy_rad, 1].
        """
        semi_conv = self.semi_conv
        offset_xy = self.offset_xy

        if random:
            dyx = random_coords(num) * semi_conv
        else:
            dyx = concentric_rings(num, semi_conv)

        dy, dx = dyx.T

        r = np.zeros((dx.size, 5), dtype=np.float64)  # x, y, theta_x, theta_y, 1
        r[:, 0] += offset_xy[0]
        r[:, 1] += offset_xy[1]
        r[:, 2] = dx
        r[:, 3] = dy
        r[:, 4] = 1.0
        return r


@jdc.pytree_dataclass
class ParallelBeam(Source):
    """Parallel beam source filling a circular aperture of given radius.

    Parameters
    ----------
    z : float
        Axial position in metres.
    radius : float
        Aperture radius in metres.
    offset_xy : CoordsXY, default (0.0, 0.0)
        Position offset (x, y) in metres.

    Notes
    -----
    Generated rays have dx=dy=0 with varying positions.
    """
    z: float
    radius: float
    offset_xy: CoordsXY = (0.0, 0.0)

    def generate_array(self, num: int, random: bool = False) -> np.ndarray:
        """Generate uniform samples within a disc aperture.

        Parameters
        ----------
        num : int
            Approximate number of rays.
        random : bool, default False
            Randomized vs deterministic concentric sampling.

        Returns
        -------
        rays : numpy.ndarray, shape (N, 5), float64
            Rows are [x_m, y_m, 0, 0, 1].
        """
        radius = self.radius
        offset_xy = self.offset_xy

        if random:
            yx = random_coords(num) * radius
        else:
            yx = concentric_rings(num, radius)

        y, x = yx.T

        r = np.zeros((x.size, 5), dtype=np.float64)  # x, y, theta_x, theta_y, 1
        r[:, 0] = (x + offset_xy[0])
        r[:, 1] = (y + offset_xy[1])
        r[:, 4] = 1.0
        return r


def airy_input_wave_for_flat_disk(
    axis: int,
    disk_radius: float,
    focal_length: float,
    voltage: float,
    *,
    z0: float = 0.0,
    target_centre_xy: tuple[float, float] = (0.0, 0.0),
    rings: float = 10.0,
    overlap_factor: float = 1.1,
    taper_fraction: float = 0.12,
    wavelength_unit: str = "m",
) -> GaussianBeam:
    """Represent the Airy/jinc input field that focuses to a flat disk.

    Place an ideal thin lens one focal length after this source and observe the
    field one focal length beyond the lens. Under the paraxial Fourier-transform
    relation, the signed radial input amplitude ``2 J1(u) / u`` produces a
    circular top-hat field of radius ``disk_radius`` in that back focal plane.

    All distances use ``wavelength_unit`` as their coordinate unit. The
    infinite jinc input is truncated after ``rings`` oscillations and smoothly
    tapered at the edge of the sampled disk.
    """
    from scipy.special import j1

    axis = int(axis)
    disk_radius = float(disk_radius)
    focal_length = float(focal_length)
    rings = float(rings)
    overlap_factor = float(overlap_factor)
    taper_fraction = float(taper_fraction)
    wavelength = _electron_wavelength_in_unit(voltage, wavelength_unit)

    if axis < 8:
        raise ValueError("axis must be at least eight.")
    if disk_radius <= 0.0 or focal_length <= 0.0:
        raise ValueError("disk_radius and focal_length must be positive.")
    if rings < 2.0:
        raise ValueError("rings must be at least two.")
    if overlap_factor <= 0.0:
        raise ValueError("overlap_factor must be positive.")
    if not 0.0 < taper_fraction < 1.0:
        raise ValueError("taper_fraction must lie strictly between zero and one.")

    wavenumber = 2.0 * np.pi / wavelength
    max_argument = (rings + 0.25) * np.pi
    extent = max_argument * focal_length / (wavenumber * disk_radius)
    coordinates = np.linspace(-extent, extent, axis)
    spacing = float(coordinates[1] - coordinates[0])
    waist = overlap_factor * spacing
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    radius = np.hypot(xx, yy)
    argument = wavenumber * disk_radius * radius / focal_length

    target = np.ones_like(argument)
    nonzero = argument != 0.0
    target[nonzero] = 2.0 * j1(argument[nonzero]) / argument[nonzero]
    normalized_radius = radius / extent
    taper_start = 1.0 - taper_fraction
    taper_phase = np.clip(
        (normalized_radius - taper_start) / taper_fraction,
        0.0,
        1.0,
    )
    taper = np.where(
        normalized_radius <= taper_start,
        1.0,
        0.5 * (1.0 + np.cos(np.pi * taper_phase)),
    )
    taper = np.where(normalized_radius <= 1.0, taper, 0.0)
    coefficients = _gaussian_cardinal_coefficients(
        target * taper,
        width_over_step=overlap_factor,
    )

    slope_x = float(target_centre_xy[0]) / focal_length
    slope_y = float(target_centre_xy[1]) / focal_length
    tilt_phase = np.exp(1j * wavenumber * (slope_x * xx + slope_y * yy))
    count = axis * axis
    ones = np.ones(count, dtype=float)
    return make_gaussian(
        x=xx.ravel(),
        y=yy.ravel(),
        dx=ones * slope_x,
        dy=ones * slope_y,
        amp=(coefficients * tilt_phase).ravel(),
        phase=np.zeros(count),
        waist_x=ones * waist,
        waist_y=ones * waist,
        rcurv_x=ones * np.inf,
        rcurv_y=ones * np.inf,
        z=ones * z0,
        voltage=ones * voltage,
        wavelength_unit=wavelength_unit,
    )


def circular_input_wave(
    aperture_radius: float,
    waist: float,
    voltage: float,
    *,
    amp: float = 1.0,
    phase: float = 0.0,
    z0: float = 0.0,
    overlap_factor: float = 2.0,
    sampling: str = "fibonacci",
    offset_xy: tuple[float, float] = (0.0, 0.0),
    wavelength_unit: str = "m",
) -> GaussianBeam:
    d = waist / overlap_factor
    area = jnp.pi * aperture_radius**2
    num_rays = int(jnp.ceil(area / (d * d)))

    if sampling.lower() == "fibonacci":
        x0, y0 = fibonacci_spiral(num_rays, aperture_radius)
    else:
        x0, y0 = uniform_disk(num_rays, aperture_radius)

    x0 = x0 + offset_xy[0]
    y0 = y0 + offset_xy[1]

    amp_per_ray = amp * uniform_amp_from_area(num_rays, waist, area)

    beam = make_gaussian(
        x=x0,
        y=y0,
        dx=jnp.zeros_like(x0),
        dy=jnp.zeros_like(y0),
        amp=jnp.ones_like(x0) * amp_per_ray,
        phase=jnp.ones_like(x0) * phase,
        waist_x=jnp.ones_like(x0) * waist,
        waist_y=jnp.ones_like(y0) * waist,
        rcurv_x=jnp.ones_like(x0) * jnp.inf,
        rcurv_y=jnp.ones_like(y0) * jnp.inf,
        z=jnp.ones_like(x0) * z0,
        voltage=jnp.ones_like(x0) * voltage,
        wavelength_unit=wavelength_unit,
    )
    return beam


def point_source_wave(
    n: int,
    half_angle: float,
    waist: float,
    voltage: float,
    *,
    z0: float = 0.0,
    x0: float = 0.0,
    y0: float = 0.0,
    wavelength_unit: str = "m",
) -> GaussianBeam:
    """Point source emitting *n* Gaussians uniformly spread in angle.

    All sub-beams share the same spatial position ``(x0, y0, z0)`` and
    waist, but their slopes are distributed on a Fibonacci spiral in
    angle space filling a cone of ``half_angle`` [rad].

    Parameters
    ----------
    n : int
        Number of Gaussian sub-beams.
    half_angle : float
        Semi-convergence angle [rad].
    waist : float
        RMS Gaussian waist [m] (same for x and y).
    voltage : float
        Accelerating voltage [V].
    z0, x0, y0 : float
        Source position [m].
    wavelength_unit : str
        Length unit for wavelength (default ``"m"``).
    """
    dx, dy = fibonacci_spiral(n, half_angle)
    dx = jnp.asarray(dx)
    dy = jnp.asarray(dy)

    ones = jnp.ones_like(dx)

    return make_gaussian(
        x=ones * x0,
        y=ones * y0,
        dx=dx,
        dy=dy,
        amp=ones / n,
        phase=jnp.zeros_like(dx),
        waist_x=ones * waist,
        waist_y=ones * waist,
        rcurv_x=ones * jnp.inf,
        rcurv_y=ones * jnp.inf,
        z=ones * z0,
        voltage=ones * voltage,
        wavelength_unit=wavelength_unit,
    )


def point_source_wave_focused_to_plane(
    n: int,
    half_angle: float,
    waist_at_plane: float,
    voltage: float,
    target_z: float,
    *,
    z0: float = 0.0,
    x0: float = 0.0,
    y0: float = 0.0,
    wavelength_unit: str = "m",
) -> GaussianBeam:
    """Create a point-source fan with a chosen waist at ``target_z``.

    The beamlet centres and slopes follow a point source while their complex
    curvature is back-propagated from a waist of ``waist_at_plane`` at the
    target plane. All spatial arguments use ``wavelength_unit`` as their
    coordinate unit; ``voltage`` is the accelerating voltage in volts.
    """
    n = int(n)
    half_angle = float(half_angle)
    waist_at_plane = float(waist_at_plane)
    target_z = float(target_z)
    z0 = float(z0)
    if n <= 0:
        raise ValueError("n must be positive.")
    if not np.isfinite(half_angle) or half_angle < 0.0:
        raise ValueError("half_angle must be finite and non-negative.")
    if not np.isfinite(waist_at_plane) or waist_at_plane <= 0.0:
        raise ValueError("waist_at_plane must be finite and positive.")
    if not np.isfinite(target_z) or not np.isfinite(z0):
        raise ValueError("target_z and z0 must be finite.")

    wavelength = _electron_wavelength_in_unit(voltage, wavelength_unit)
    dx, dy = fibonacci_spiral(n, half_angle)
    dx = jnp.asarray(dx)
    dy = jnp.asarray(dy)
    ones = jnp.ones_like(dx)

    q_inv_target = 1j * wavelength / (jnp.pi * waist_at_plane**2)
    distance = target_z - z0
    q_inv_source = q_inv_target / (1.0 - distance * q_inv_target)
    source_im = jnp.imag(q_inv_source)
    source_re = jnp.real(q_inv_source)
    source_waist = jnp.sqrt(wavelength / (jnp.pi * source_im))
    source_rcurv = jnp.where(
        jnp.abs(source_re) > 0.0,
        1.0 / source_re,
        jnp.inf,
    )

    return make_gaussian(
        x=ones * x0,
        y=ones * y0,
        dx=dx,
        dy=dy,
        amp=ones / n,
        phase=jnp.zeros_like(dx),
        waist_x=ones * source_waist,
        waist_y=ones * source_waist,
        rcurv_x=ones * source_rcurv,
        rcurv_y=ones * source_rcurv,
        z=ones * z0,
        voltage=ones * voltage,
        wavelength_unit=wavelength_unit,
    )


def finite_probe_wave_focused_to_plane(
    n: int,
    input_radius: float,
    target_radius: float,
    waist_at_plane: float,
    voltage: float,
    target_z: float,
    *,
    z0: float = 0.0,
    input_center_xy: tuple[float, float] = (0.0, 0.0),
    target_center_xy: tuple[float, float] = (0.0, 0.0),
    wavelength_unit: str = "m",
) -> GaussianBeam:
    """Create a finite probe aimed to cover a disk at ``target_z``.

    Beamlet centres sample ``input_radius`` at ``z0`` and map to the matching
    samples within ``target_radius`` at the target plane. Each beamlet is given
    a waist of ``waist_at_plane`` there. All spatial arguments use
    ``wavelength_unit`` as their coordinate unit.
    """
    n = int(n)
    input_radius = float(input_radius)
    target_radius = float(target_radius)
    waist_at_plane = float(waist_at_plane)
    target_z = float(target_z)
    z0 = float(z0)
    if n <= 0:
        raise ValueError("n must be positive.")
    if not np.isfinite(input_radius) or input_radius < 0.0:
        raise ValueError("input_radius must be finite and non-negative.")
    if not np.isfinite(target_radius) or target_radius < 0.0:
        raise ValueError("target_radius must be finite and non-negative.")
    if not np.isfinite(waist_at_plane) or waist_at_plane <= 0.0:
        raise ValueError("waist_at_plane must be finite and positive.")
    if not np.isfinite(target_z) or not np.isfinite(z0):
        raise ValueError("target_z and z0 must be finite.")

    distance = target_z - z0
    if distance == 0.0:
        raise ValueError("`target_z` must differ from `z0`.")
    wavelength = _electron_wavelength_in_unit(voltage, wavelength_unit)

    unit_x, unit_y = fibonacci_spiral(n, 1.0)
    unit_x = jnp.asarray(unit_x)
    unit_y = jnp.asarray(unit_y)
    source_x = input_center_xy[0] + input_radius * unit_x
    source_y = input_center_xy[1] + input_radius * unit_y
    target_x = target_center_xy[0] + target_radius * unit_x
    target_y = target_center_xy[1] + target_radius * unit_y
    dx = (target_x - source_x) / distance
    dy = (target_y - source_y) / distance
    ones = jnp.ones_like(source_x)

    q_inv_target = 1j * wavelength / (jnp.pi * waist_at_plane**2)
    q_inv_source = q_inv_target / (1.0 - distance * q_inv_target)
    source_im = jnp.imag(q_inv_source)
    source_re = jnp.real(q_inv_source)
    source_waist = jnp.sqrt(wavelength / (jnp.pi * source_im))
    source_rcurv = jnp.where(
        jnp.abs(source_re) > 0.0,
        1.0 / source_re,
        jnp.inf,
    )

    return make_gaussian(
        x=source_x,
        y=source_y,
        dx=dx,
        dy=dy,
        amp=ones / n,
        phase=jnp.zeros_like(source_x),
        waist_x=ones * source_waist,
        waist_y=ones * source_waist,
        rcurv_x=ones * source_rcurv,
        rcurv_y=ones * source_rcurv,
        z=ones * z0,
        voltage=ones * voltage,
        wavelength_unit=wavelength_unit,
    )


def square_input_wave(
    aperture_length: float,
    waist: float,
    voltage: float,
    amp: float = 1.0,
    phase: float = 0.0,
    z0: float = 0.0,
    overlap_factor: float = 2.0,
    centre_xy: tuple[float, float] = (0.0, 0.0),
    wavelength_unit: str = "m",
) -> GaussianBeam:
    """Create a square Gaussian lattice using the rectangular implementation."""
    return rectangular_input_wave(
        aperture_width=aperture_length,
        aperture_height=aperture_length,
        waist=waist,
        voltage=voltage,
        amp=amp,
        phase=phase,
        z0=z0,
        overlap_factor=overlap_factor,
        centre_xy=centre_xy,
        wavelength_unit=wavelength_unit,
    )


def rectangular_input_wave(
    aperture_width: float,
    aperture_height: float,
    waist: float,
    voltage: float,
    amp: float = 1.0,
    phase: float = 0.0,
    z0: float = 0.0,
    overlap_factor: float = 2.0,
    centre_xy: tuple[float, float] = (0.0, 0.0),
    wavelength_unit: str = "m",
) -> GaussianBeam:
    area = aperture_width * aperture_height
    d = waist / overlap_factor

    nx = int(jnp.ceil(aperture_width / d)) + 1
    ny = int(jnp.ceil(aperture_height / d)) + 1
    num_rays = nx * ny

    xs = (jnp.arange(nx) - 0.5 * (nx - 1)) * d
    ys = (jnp.arange(ny) - 0.5 * (ny - 1)) * d
    xg, yg = jnp.meshgrid(xs, ys, indexing="ij")
    x0 = xg.ravel()
    y0 = yg.ravel()

    amp_ray = amp * uniform_amp_from_area(num_rays, waist, area)

    x0 = x0 + centre_xy[0]
    y0 = y0 + centre_xy[1]
    beam = make_gaussian(
        x=x0,
        y=y0,
        dx=jnp.zeros_like(x0),
        dy=jnp.zeros_like(y0),
        amp=jnp.ones_like(x0) * amp_ray,
        phase=jnp.zeros_like(y0) + phase,
        waist_x=jnp.ones_like(x0) * waist,
        waist_y=jnp.ones_like(y0) * waist,
        rcurv_x=jnp.ones_like(x0) * jnp.inf,
        rcurv_y=jnp.ones_like(y0) * jnp.inf,
        z=jnp.ones_like(x0) * z0,
        voltage=jnp.ones_like(x0) * voltage,
        wavelength_unit=wavelength_unit,
    )
    return beam


def sample_input_wave(
    aperture_length: float,
    waist: float,
    voltage: float,
    amp_interpolator: "Interpolator2D",
    phase_interpolator: "Interpolator2D",
    z0: float = 0.0,
    overlap_factor: float = 2.0,
    centre_xy: tuple[float, float] = (0.0, 0.0),
    wavelength_unit: str = "m",
) -> GaussianBeam:
    d = waist / overlap_factor
    nx = int(jnp.ceil(aperture_length / d))
    ny = int(jnp.ceil(aperture_length / d))

    xs = (jnp.arange(nx) - 0.5 * (nx - 1)) * d
    ys = (jnp.arange(ny) - 0.5 * (ny - 1)) * d
    xg, yg = jnp.meshgrid(xs, ys, indexing="ij")
    x0 = xg.ravel()
    y0 = yg.ravel()

    amp_norm = overlap_factor * 2 * jnp.pi

    x0 = x0 + centre_xy[0]
    y0 = y0 + centre_xy[1]

    amplitude = amp_interpolator(x0, y0)
    phase = phase_interpolator(x0, y0)

    beam = make_gaussian(
        x=x0,
        y=y0,
        dx=jnp.zeros_like(x0),
        dy=jnp.zeros_like(y0),
        amp=amplitude / amp_norm,
        phase=phase,
        waist_x=jnp.ones_like(x0) * waist,
        waist_y=jnp.ones_like(y0) * waist,
        rcurv_x=jnp.ones_like(x0) * jnp.inf,
        rcurv_y=jnp.ones_like(y0) * jnp.inf,
        z=jnp.ones_like(x0) * z0,
        voltage=jnp.ones_like(x0) * voltage,
        wavelength_unit=wavelength_unit,
    )
    return beam


def decompose_gaussian(
    beam: GaussianBeam, *, num_across: int, through=(),
    extent_rms: float = 6.0, overlap: float = 1.1,
) -> GaussianBeam:
    """Represent one round coherent Gaussian by ``num_across**2`` beamlets.

    The returned bundle starts at the input beam's plane. ``through`` is an
    optional lossless, linear, round column ending at a sampling plane (usually
    just before an aperture). Choose narrow beamlets there, then map them back
    to the source. Propagating the bundle forward uses the ordinary components.
    The physical source stays the same as the sampling is refined.

    ``extent_rms`` is the lattice half-width in intensity RMS units at that
    plane. ``overlap`` is each beamlet's amplitude 1/e radius / lattice spacing.
    Gaussian convolution sets the weights, rather than treating the beamlets
    as independent emitters. Sum their complex fields, not their intensities.
    This constructs a fixed numerical basis; it is not a differentiable fit.
    """
    import operator
    from .components import ABCDTransfer
    from .gaussian import FreeSpacePropagator
    from .run import column_matrix, run_to_end

    num_across = operator.index(num_across)
    if num_across < 8:
        raise ValueError("num_across must be at least 8.")
    if not np.isfinite(extent_rms) or extent_rms <= 0:
        raise ValueError("extent_rms must be finite and positive.")
    if not np.isfinite(overlap) or overlap <= 0:
        raise ValueError("overlap must be finite and positive.")
    if np.shape(beam.x) != () or np.shape(beam.Q_inv) != (2, 2):
        raise ValueError("Supply one Gaussian, not a beamlet bundle.")
    through = tuple(through)
    propagator = FreeSpacePropagator(include_axial_path=False)
    target = run_to_end(beam, through, propagator=propagator) if through else beam
    q = np.asarray(target.Q_inv)
    if not np.allclose(q, q[0, 0] * np.eye(2), rtol=1e-9, atol=1e-12):
        raise ValueError("The Gaussian must be round at the sampling plane.")
    k, curvature = float(target.k), float(q[0, 0].real)
    if not np.isfinite(q).all() or q[0, 0].imag <= 0:
        raise ValueError("The Gaussian must have finite, positive width.")
    width = np.sqrt(2 / (k * q[0, 0].imag))
    step = extent_rms * width / num_across  # intensity RMS = width / 2
    waist = overlap * step
    weight_width2 = width**2 - waist**2
    if weight_width2 <= 0:
        raise ValueError("Increase num_across: beamlets must be narrower than the beam.")
    axis = (jnp.arange(num_across) - 0.5 * (num_across - 1)) * step
    x, y = jnp.meshgrid(axis, axis)
    x, y = x.ravel(), y.ravel()
    weights = (step**2 * width**2 / (jnp.pi * waist**2 * weight_width2)
               * jnp.exp(-(x*x + y*y) / weight_width2))
    sampled = GaussianBeam(
        x=target.x + x, y=target.y + y,
        dx=target.dx + curvature*x, dy=target.dy + curvature*y,
        z=target.z, voltage=target.voltage, amplitude=target.amplitude * weights,
        Q_inv=jnp.broadcast_to((curvature + 2j / (k * waist**2)) * jnp.eye(2),
                              (num_across**2, 2, 2)),
        pathlength=target.pathlength + target.dx*x + target.dy*y
        + 0.5 * curvature * (x*x + y*y),
    )
    if not through:
        return sampled

    matrix = column_matrix(through, source_z=beam.z, voltage=beam.voltage)
    inverse = jnp.linalg.inv(matrix[:4, :4])
    offset = matrix[:4, 4]
    shifted = sampled.derive(x=sampled.x-offset[0], y=sampled.y-offset[1],
                             dx=sampled.dx-offset[2], dy=sampled.dy-offset[3])
    source = ABCDTransfer(A=inverse[:2, :2], B=inverse[:2, 2:],
                          C=inverse[2:, :2], D=inverse[2:, 2:])(shifted).derive(
        z=beam.z, amplitude=jnp.ones_like(weights), pathlength=jnp.zeros_like(x),
    )
    # Let the native column supply its own action and Gouy factor. Undo these
    # on the source so a forward pass reconstructs the desired sampling basis.
    returned = run_to_end(source, through, propagator=propagator)
    return source.derive(amplitude=sampled.amplitude / returned.amplitude,
                         pathlength=sampled.pathlength - returned.pathlength)


def decompose_gaussian_angular(
    beam: GaussianBeam, *, num_across: int, through=(),
    broadening: float = 8.0, extent_sigma: float = 6.0,
) -> GaussianBeam:
    """Represent one coherent Gaussian by wider packets with sampled tilts.

    Gaussian angular quadrature reconstructs the original field by interference.
    Increasing num_across refines the quadrature without shrinking packet widths
    or increasing its angular extent. This is useful for nanometre probes where
    a very fine spatial decomposition would create nonparaxial narrow packets.
    broadening multiplies each packet's spatial width at the sampling plane.
    through optionally selects that plane via a lossless linear optical column;
    the returned bundle is mapped back to the input plane for native propagation.
    Construction is a fixed host-side choice of basis, not a differentiable fit.
    """
    import operator
    from .gaussian import FreeSpacePropagator
    from .run import run_to_end

    num_across = operator.index(num_across)
    if num_across < 8:
        raise ValueError("num_across must be at least 8.")
    if not np.isfinite(broadening) or broadening <= 1:
        raise ValueError("broadening must be finite and greater than one.")
    if not np.isfinite(extent_sigma) or extent_sigma <= 0:
        raise ValueError("extent_sigma must be finite and positive.")
    if np.shape(beam.x) != () or np.shape(beam.Q_inv) != (2, 2):
        raise ValueError("Supply one Gaussian, not a beamlet bundle.")
    through = tuple(through)
    propagator = FreeSpacePropagator(include_axial_path=False)
    target = run_to_end(beam, through, propagator=propagator) if through else beam
    q = np.asarray(target.Q_inv)
    k = float(target.k)
    covariance = k * q.imag * (1 - broadening**-2)
    if not np.isfinite(covariance).all() or np.linalg.eigvalsh(covariance).min() <= 0:
        raise ValueError("The Gaussian envelope must have positive definite precision.")
    root = np.linalg.cholesky(covariance)
    step = 2 * extent_sigma / num_across
    axis = (np.arange(num_across) + 0.5) * step - extent_sigma
    xx, yy = np.meshgrid(axis, axis)
    standard = np.column_stack((xx.ravel(), yy.ravel()))
    weights = step**2 / (2 * np.pi) * np.exp(-0.5 * np.sum(standard**2, axis=1))
    slopes = jnp.asarray(standard @ root.T / k)
    count = num_across**2
    sampled = jax.tree.map(lambda value: jnp.repeat(value, count, axis=0), target.to_vector())
    sampled = sampled.derive(
        dx=target.dx + slopes[:, 0], dy=target.dy + slopes[:, 1],
        amplitude=target.amplitude * jnp.asarray(weights),
        Q_inv=jnp.broadcast_to(jnp.asarray(q.real + 1j*q.imag/broadening**2), (count, 2, 2)),
    )
    if not through:
        return sampled
    from .components import ABCDTransfer
    from .run import column_matrix

    matrix = column_matrix(through, source_z=beam.z, voltage=beam.voltage)
    inverse = jnp.linalg.inv(matrix[:4, :4])
    offset = matrix[:4, 4]
    shifted = sampled.derive(x=sampled.x-offset[0], y=sampled.y-offset[1],
                             dx=sampled.dx-offset[2], dy=sampled.dy-offset[3])
    source = ABCDTransfer(A=inverse[:2, :2], B=inverse[:2, 2:],
                          C=inverse[2:, :2], D=inverse[2:, 2:])(shifted).derive(
        z=beam.z, amplitude=jnp.ones(count), pathlength=jnp.zeros(count),
    )
    returned = run_to_end(source, through, propagator=propagator)
    return source.derive(amplitude=sampled.amplitude / returned.amplitude,
                         pathlength=sampled.pathlength - returned.pathlength)


def gaussian_waist(*, z, voltage, rms_size, x=0.0, y=0.0, theta_x=0.0, theta_y=0.0):
    """Unit-flux round Gaussian waist; positions and RMS size in metres, slopes in radians."""
    width = 2 * rms_size
    k = 2 * jnp.pi / energy2wavelength(voltage)
    return GaussianBeam(
        x=jnp.asarray(x), y=jnp.asarray(y), dx=theta_x, dy=theta_y,
        z=z, voltage=voltage,
        amplitude=jnp.sqrt(2 / (jnp.pi * width**2)) + 0j,
        Q_inv=1j * (2 / (k * width**2)) * jnp.eye(2), pathlength=0.0,
    )


@njit
def _multi_cumsum_inplace(values, partitions, start):
    """Reset and accumulate angles within each ring; modifies values in place."""
    part_idx = 0
    current_part_len = partitions[part_idx]
    part_count = 0
    values[0] = start
    for v_idx in range(1, values.size):
        if current_part_len == part_count:
            part_count = 0
            part_idx += 1
            current_part_len = partitions[part_idx]
            values[v_idx] = start
        else:
            values[v_idx] += values[v_idx - 1]
            part_count += 1


def concentric_rings(
    num_points_approx: int,
    radius: float,
) -> np.ndarray:
    """Return approximately num_points_approx samples as (y, x) rows inside radius."""
    num_rings = max(
        1, int(np.floor((-1 + np.sqrt(1 + 4 * num_points_approx / np.pi)) / 2))
    )

    # Calculate the circumference of each ring
    num_points_kth_ring = np.round(2 * np.pi * np.arange(1, num_rings + 1)).astype(int)
    num_rings = num_points_kth_ring.size
    points_per_unit = num_points_approx / num_points_kth_ring.sum()
    points_per_ring = np.round(num_points_kth_ring * points_per_unit).astype(int)

    # Sample equally spaced radii, excluding the origin.
    radii = np.linspace(
        0,
        radius,
        num_rings + 1,
        endpoint=True,
    )[1:]
    div_angle = 2 * np.pi / points_per_ring

    params = np.stack((radii, div_angle), axis=0)

    repeats = points_per_ring.tolist()

    all_params = np.repeat(params, repeats, axis=-1)
    _multi_cumsum_inplace(all_params[1, :], points_per_ring, 0.0)

    all_radii = all_params[0, :]
    all_angles = all_params[1, :]

    return np.stack(
        (
            all_radii * np.sin(all_angles),
            all_radii * np.cos(all_angles),
        ),
        axis=-1,
    )


def random_coords(num: int) -> np.ndarray:
    """Return approximately num random (y, x) rows in the unit disk."""
    yx = np.random.uniform(
        -1,
        1,
        size=(int(num * 1.28), 2),  # 4 / np.pi
    )
    radii = np.sqrt((yx**2).sum(axis=1))
    mask = radii < 1
    yx = yx[mask, :]
    return yx


def fibonacci_spiral(
    nb_samples: int,
    radius: float,
    alpha=2,
    **_,
):
    # From https://github.com/matt77hias/fibpy/blob/master/src/sampling.py
    # Fibonacci spiral sampling in a unit circle
    # Alpha parameter determines smoothness of boundary - default of 2 means a smooth boundary
    # 0 for a rough boundary.
    # Returns a tuple of x, y coordinates of the samples

    ga = np.pi * (3.0 - np.sqrt(5.0))

    # Boundary points
    np_boundary = np.round(alpha * np.sqrt(nb_samples))

    ii = np.arange(nb_samples)
    rr = np.where(
        ii > nb_samples - (np_boundary + 1),
        radius,
        radius * np.sqrt((ii + 0.5) / (nb_samples - 0.5 * (np_boundary + 1)))
    )
    rr[0] = 0.
    phi = ii * ga

    x = rr * np.cos(phi)
    y = rr * np.sin(phi)

    return x, y


def uniform_disk(
    nb_samples: int,
    radius: float,
    *,
    waist_radius: float | None = None,
    overlap_factor: float | None = None,
    max_iterations: int = 20,
    **_,
):
    """Return exactly nb_samples as separate x/y arrays on concentric rings.

    Start from the requested beam overlap or area per point, then adjust the
    ring spacing and final ring population to obtain the requested count.
    """
    if nb_samples < 0:
        raise ValueError("nb_samples must be non-negative.")
    if radius <= 0.0 and nb_samples > 0:
        raise ValueError("radius must be positive for non-zero samples.")
    if nb_samples == 0:
        empty = np.empty(0, dtype=float)
        return empty, empty
    if nb_samples == 1:
        return np.array([0.0], dtype=float), np.array([0.0], dtype=float)

    if overlap_factor is not None and overlap_factor <= 0.0:
        raise ValueError("overlap_factor must be positive when provided.")

    area = np.pi * radius**2
    if area == 0.0:
        return np.zeros(nb_samples, dtype=float), np.zeros(nb_samples, dtype=float)

    if waist_radius is not None and overlap_factor is not None:
        base_spacing = waist_radius / overlap_factor
    else:
        base_spacing = np.sqrt(area / nb_samples)
    if base_spacing <= 0.0:
        raise ValueError("Computed spacing must be positive.")

    tol = 1e-12

    def build_rings(spacing: float) -> list[np.ndarray]:
        rings: list[np.ndarray] = []
        rings.append(np.array([[0.0, 0.0]], dtype=float))
        total = 1
        ring_idx = 1
        while True:
            r = ring_idx * spacing
            if r > radius + tol:
                break
            circumference = 2.0 * np.pi * r
            points_on_ring = max(1, int(np.round(circumference / spacing)))
            angles = np.linspace(0.0, 2.0 * np.pi, points_on_ring, endpoint=False, dtype=float)
            coords = np.stack((r * np.cos(angles), r * np.sin(angles)), axis=1)
            rings.append(coords)
            total += points_on_ring
            if total >= nb_samples:
                break
            ring_idx += 1
        return rings

    spacing = base_spacing
    for _ in range(max_iterations):
        rings = build_rings(spacing)
        counts = [ring.shape[0] for ring in rings]
        total_points = sum(counts)

        if total_points < nb_samples:
            ratio = total_points / nb_samples if total_points > 0 else 0.5
            spacing *= max(0.2, np.sqrt(max(ratio, 1e-6)))
            continue

        excess = total_points - nb_samples
        if excess > 0:
            last_ring = rings[-1]
            last_count = last_ring.shape[0]
            if excess >= last_count:
                rings.pop()
                total_points -= last_count
                excess = total_points - nb_samples
                if excess > 0 and rings:
                    last_ring = rings[-1]
                    last_count = last_ring.shape[0]
            if rings and excess > 0:
                keep = last_count - excess
                if keep <= 0:
                    rings.pop()
                else:
                    idx = np.linspace(0, last_count - 1, keep, endpoint=False, dtype=float)
                    idx = np.clip(np.round(idx).astype(int), 0, last_count - 1)
                    rings[-1] = last_ring[idx]
        coords = np.concatenate(rings, axis=0)
        if coords.shape[0] == nb_samples:
            coords = _snap_to_spacing(coords, base_spacing, radius, tol)
            return coords[:, 0], coords[:, 1]
        if coords.shape[0] > nb_samples:
            coords = coords[:nb_samples]
            coords = _snap_to_spacing(coords, base_spacing, radius, tol)
            return coords[:, 0], coords[:, 1]
        spacing *= 0.9

    raise RuntimeError("Could not construct uniform disk sampling with the requested parameters.")


def _snap_to_spacing(coords: np.ndarray, spacing: float, radius: float, tol: float) -> np.ndarray:
    radii = np.linalg.norm(coords, axis=1)
    snapped = coords.copy()
    mask = radii > tol
    if not np.any(mask):
        return snapped

    snap_spacing = np.round(spacing, decimals=12)
    snap_spacing = snap_spacing if snap_spacing > 0 else spacing
    target = np.round(radii[mask] / snap_spacing) * snap_spacing
    max_multiple = np.floor(radius / snap_spacing)
    target = np.clip(target, 0.0, max_multiple * snap_spacing)
    scale = np.where(radii[mask] > 0.0, target / radii[mask], 1.0)
    snapped[mask] *= scale[:, None]
    return snapped


def uniform_amp_from_area(num_gaussians: int, waist: float, area: float) -> float:
    # area = (2*extent)^2 if you tile a square [-extent, extent]^2
    return area / (num_gaussians * jnp.pi * waist**2)
