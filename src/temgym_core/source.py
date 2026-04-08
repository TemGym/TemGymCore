from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import jax_dataclasses as jdc

from .tree_utils import HasParamsMixin
from .gaussian import GaussianBeam, make_gaussian
from .constants import energy2wavelength
from .ray import Ray
from .utils import (
    concentric_rings,
    fibonacci_spiral,
    random_coords,
    uniform_amp_from_area,
    uniform_disk,
)
from . import CoordsXY

if TYPE_CHECKING:
    from interpax import Interpolator2D


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
    return Ray(x=x, y=y, dx=dx, dy=dy, z=z_vec, pathlength=path_vec)


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
    _ = amp
    d = waist / overlap_factor
    area = jnp.pi * aperture_radius**2
    num_rays = int(jnp.ceil(area / (d * d)))

    if sampling.lower() == "fibonacci":
        x0, y0 = fibonacci_spiral(num_rays, aperture_radius)
    else:
        x0, y0 = uniform_disk(num_rays, aperture_radius)

    x0 = x0 + offset_xy[0]
    y0 = y0 + offset_xy[1]

    amp_per_ray = uniform_amp_from_area(num_rays, waist, area)

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

    beam = make_gaussian(
        x=x0,
        y=y0,
        dx=jnp.zeros_like(x0),
        dy=jnp.zeros_like(y0),
        amp=jnp.ones_like(x0) * amp / amp_norm,
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
    _ = amp
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

    amp_ray = uniform_amp_from_area(num_rays, waist, area)

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
