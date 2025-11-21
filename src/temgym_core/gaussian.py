import dataclasses
import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax.nn import softplus
import jax_dataclasses as jdc
from jax import lax
from interpax import Interpolator2D, Interpolator3D

from temgym_core.components import Detector
from temgym_core.aberrations import (
    KrivanekCoeffs,
    SeidelCoeffs,
    Seidel_aperture_pos_aperture_slope,
    W_krivanek
)

from .ray import Ray
from typing import (
    Any,
    Callable,
    NamedTuple,
    Sequence,
    Tuple
)

from ase import units
from .constants import (
    energy2wavelength,
    relativistic_mass_correction
)
from .utils import (
    fibonacci_spiral,
    uniform_disk,
    uniform_amp_from_area,
)

from .potential import potential_smoothed

LENGTH = {
    "m": 1.0,
    "A": 1e-10,
    "angstrom": 1e-10,
}


@jdc.pytree_dataclass(kw_only=True)
class GaussianBeam(Ray):
    amplitude: jnp.ndarray | complex  # complex amplitude + global offsets from propagation
    Q_inv: jnp.ndarray | complex  # 2x2 complex matrix - inverse complex curvature matrix
    voltage: jnp.ndarray | float | None = None  # in eV
    wavelength_unit: jdc.Static[str] = "m"

    def derive(self,
               x: float | jnp.ndarray | None = None,
               y: float | jnp.ndarray | None = None,
               dx: float | jnp.ndarray | None = None,
               dy: float | jnp.ndarray | None = None,
               z: float | jnp.ndarray | None = None,
               amplitude: jnp.ndarray | complex | None = None,
               pathlength: float | jnp.ndarray | None = None,
               Q_inv: jnp.ndarray | complex | None = None,
               voltage: float | jnp.ndarray | None = None,
               wavelength_unit: str | None = None,
               ) -> "GaussianBeam":

        return GaussianBeam(
            x=self.x if x is None else x,
            y=self.y if y is None else y,
            dx=self.dx if dx is None else dx,
            dy=self.dy if dy is None else dy,
            z=self.z if z is None else z,
            amplitude=self.amplitude if amplitude is None else amplitude,
            Q_inv=self.Q_inv if Q_inv is None else Q_inv,
            voltage=self.voltage if voltage is None else voltage,
            pathlength=self.pathlength if pathlength is None else pathlength,
            wavelength_unit=self.wavelength_unit if wavelength_unit is None else wavelength_unit,
        )

    def to_vector(self) -> jnp.ndarray:
        params = {}
        for k, v in dataclasses.asdict(self).items():
            params[k] = v if isinstance(v, str) or v is None else jnp.atleast_1d(v)
        return type(self)(**params)

    @property
    def wavelength(self):
        return energy2wavelength(self.voltage) / LENGTH[self.wavelength_unit]

    @property
    def mass(self):
        return relativistic_mass_correction(self.voltage) * units._me

    @property
    def sigma(self):
        lam = self.wavelength
        return (
            2 * jnp.pi * self.mass * units.kg * units._e * units.C * lam
            / (units._hplanck * units.s * units.J) ** 2
        )

    @property
    def k(self):
        return 2 * jnp.pi / self.wavelength


def make_gaussian(
    x=0.0,
    y=0.0,
    dx=0.0,
    dy=0.0,
    z=0.0,
    voltage: float | jnp.ndarray = 1e5,
    amp=1.0,
    phase=0.0,
    waist_x=1.0,
    waist_y=1.0,
    rcurv_x=jnp.inf,
    rcurv_y=jnp.inf,
    wavelength_unit: str = "m",
) -> "GaussianBeam":

    wavelength = energy2wavelength(voltage) / LENGTH[wavelength_unit]

    x = jnp.atleast_1d(x)
    n_rays = x.shape[0]

    def _bcast_to_n(a):
        a = jnp.atleast_1d(a)
        return a if a.shape[0] == n_rays else jnp.broadcast_to(a, (n_rays,))

    y = _bcast_to_n(y)
    dx = _bcast_to_n(dx)
    dy = _bcast_to_n(dy)

    curv_x = _bcast_to_n(1.0 / rcurv_x)
    curv_y = _bcast_to_n(1.0 / rcurv_y)
    waist_x = _bcast_to_n(waist_x)
    waist_y = _bcast_to_n(waist_y)

    voltage = _bcast_to_n(voltage)

    Q_inv_re = jnp.zeros((n_rays, 2, 2), dtype=jnp.float64)
    Q_inv_re = Q_inv_re.at[:, 0, 0].set(curv_x)
    Q_inv_re = Q_inv_re.at[:, 1, 1].set(curv_y)

    Q_inv_im = jnp.zeros((n_rays, 2, 2), dtype=jnp.float64)
    Q_inv_im = Q_inv_im.at[:, 0, 0].set(wavelength / (jnp.pi * waist_x**2))
    Q_inv_im = Q_inv_im.at[:, 1, 1].set(wavelength / (jnp.pi * waist_y**2))

    Q_inv = (Q_inv_re + 1j * Q_inv_im).astype(jnp.complex128)

    amp = _bcast_to_n(amp)
    phase = _bcast_to_n(phase)
    amplitude = jnp.asarray(amp) * jnp.exp(1j * jnp.asarray(phase))

    ray = GaussianBeam(
        x=x, y=y, dx=dx, dy=dy, z=z,
        amplitude=amplitude, Q_inv=Q_inv, voltage=voltage,
        pathlength=jnp.zeros_like(x),
        _one=jnp.ones_like(x),
        wavelength_unit=wavelength_unit,
    ).to_vector()

    if n_rays == 1:
        def squeeze0(a):
            if a is None:
                return None
            a = jnp.asarray(a)
            return jnp.squeeze(a, axis=0) if (a.ndim > 0 and a.shape[0] == 1) else a
        ray = jax.tree.map(squeeze0, ray)

    return ray


def apply_action_delta(
    ray,
    dS0: complex,
    dS1: jnp.ndarray,
    dS2: jnp.ndarray,
    tiny: float = 1e-30,
):

    k = ray.k
    r0 = ray.r_xy

    # Old action
    S0_old = ray.pathlength  # real
    d_xy_old = ray.d_xy  # (2,) real
    Q_old = ray.Q_inv  # (2,2) complex

    # New action before recentering:
    S0_prime = S0_old + dS0  # complex
    S1_prime = d_xy_old + dS1  # complex (linear)
    Q_prime = Q_old + dS2  # complex (2x2)

    # Recentering: solve Im(Q')·dx = -Im(S1')
    # to find the position shift dx that makes Im(S1_new) = 0
    # This equation comes from differentiating the taylor expansion and
    # setting the imaginary part of the differential to zero - i.e we have an equation
    # that tells us where the gaussian peak is flat, and we solve to find how far we need
    # to shift the coordinates to get there,
    # after a linear imaginary action has been applied.
    ImQ = jnp.imag(Q_prime)
    ImS1 = jnp.imag(S1_prime)

    def solve_dx(args):
        ImQ_, ImS1_ = args
        return jnp.linalg.solve(ImQ_, -ImS1_)

    def zero_dx(args):
        ImQ_, ImS1_ = args
        return jnp.zeros_like(ImS1_)

    det_ImQ = jnp.linalg.det(ImQ)
    dx = lax.cond(
        jnp.abs(det_ImQ) < tiny,
        zero_dx,
        solve_dx,
        (ImQ, ImS1),
    )

    r_xy_new = r0 + dx
    d_xy_new = ray.d_xy + jnp.real(dS1)

    # Shift peak and collect new coefficients
    S0_new = S0_prime + S1_prime @ dx + 0.5 * (dx @ (Q_prime @ dx))
    S1_new = S1_prime + Q_prime @ dx      # (2,) complex
    Q_new = Q_prime  # (2,2) complex

    # By construction after recentering, Im(S1_new) ≈ 0; we keep only the real slope.
    d_xy_new = jnp.real(S1_new)

    # Split S0_new into phase (pathlength) and amplitude factor
    S0_new_re = jnp.real(S0_new)
    S0_new_im = jnp.imag(S0_new)

    pathlength_new = S0_new_re
    amp_factor = jnp.exp(-k * S0_new_im)
    amplitude_new = ray.amplitude * amp_factor

    return r_xy_new, d_xy_new, amplitude_new, pathlength_new, Q_new


def scalar_grad_hess_complex(
    fn: Callable[..., complex],
    x: jnp.ndarray,
    *args: Any,
    diff_argnums: int | Sequence[int] = 0,
) -> Tuple[complex, jnp.ndarray, jnp.ndarray]:
    """
    Return (dS0, grad, hess) where dS0 = fn(x, *args) (complex scalar),
    grad = ∇_x fn (complex vector), hess = sym(∇^2_x fn) (complex matrix).

    Parameters
    ----------
    fn : Callable
        Function returning a complex scalar. The first argument is differentiated.
    x : jnp.ndarray
        Expansion point for the differentiated argument.
    *args :
        Additional positional arguments passed to `fn` but treated as constants
        during differentiation.
    diff_argnums : int or tuple of ints, default 0
        Indices of the arguments of `fn` with respect to which gradients and
        Hessians are taken. By default only the first argument is differentiated.
    """
    full_args = (x, *args)

    def re_fn(*fn_args):  # scalar real
        return jnp.real(fn(*fn_args))

    def im_fn(*fn_args):  # scalar real
        return jnp.imag(fn(*fn_args))

    # evaluate function at x for dS0
    dS0 = fn(*full_args)

    grad_re = jax.grad(re_fn, argnums=diff_argnums)(*full_args)  # (2,)
    grad_im = jax.grad(im_fn, argnums=diff_argnums)(*full_args)  # (2,)
    hess_re = jax.hessian(re_fn, argnums=diff_argnums)(*full_args)  # (2,2)
    hess_im = jax.hessian(im_fn, argnums=diff_argnums)(*full_args)  # (2,2)

    grad = grad_re + 1j * grad_im
    hess = hess_re + 1j * hess_im
    return dS0, grad, hess


@jdc.pytree_dataclass
class Component:
    z: float = 0.0

    def phase_shift(self, xy: jnp.ndarray):
        return 0.0

    def log_transmission(self, xy: jnp.ndarray):
        return 0.0

    def complex_action(self, xy: jnp.ndarray, k: float) -> complex:
        logA = self.log_transmission(xy)
        L = jnp.logaddexp(logA, -50)
        return self.phase_shift(xy) - 1j * (L / k)

    def __call__(self, ray: GaussianBeam) -> GaussianBeam:
        xy_ref = ray.r_xy
        k = ray.k

        dS0, dS1, dS2 = scalar_grad_hess_complex(
            self.complex_action, xy_ref, k
        )

        r_xy, d_xy, amplitude, pathlength, Q_new = apply_action_delta(
            ray, dS0=dS0, dS1=dS1, dS2=dS2
        )

        return ray.derive(
            x=r_xy[0],
            y=r_xy[1],
            dx=d_xy[0],
            dy=d_xy[1],
            z=ray.z,
            amplitude=amplitude,
            pathlength=pathlength,
            Q_inv=Q_new,
        )


@jdc.pytree_dataclass(kw_only=True)
class Lens(Component):
    focal_length: float
    x0: float = 0.0
    y0: float = 0.0

    def phase_shift(self, xy: jnp.ndarray):
        x, y = xy[0] - self.x0, xy[1] - self.y0
        rho2 = x * x + y * y
        return -0.5 * rho2 / self.focal_length


@jdc.pytree_dataclass(kw_only=True)
class KrivanekLens(Lens):
    """Thin lens with Krivanek aberration model applied to the phase."""
    coeffs: jdc.Static[KrivanekCoeffs]
    axis_eps: float = 1e-24

    def phase_shift(self, xy: jnp.ndarray):
        x = xy[0] - self.x0
        y = xy[1] - self.y0
        f = self.focal_length

        rho2 = x * x + y * y

        def _with_aberrations(_):
            rho = jnp.sqrt(rho2)
            phi = jnp.arctan2(y, x)
            alpha = rho / f
            return -0.5 * rho2 / f - W_krivanek(alpha, phi, self.coeffs)

        def _on_axis(_):
            return -0.5 * rho2 / f

        return lax.cond(rho2 > self.axis_eps, _with_aberrations, _on_axis, operand=None)


@jdc.pytree_dataclass(kw_only=True)
class SeidelLens(Lens):
    object_plane_dist: float  # absolute distance from object to lens
    coeffs: SeidelCoeffs = SeidelCoeffs()

    def log_transmission(self, xy):
        return 0.0

    def phase_shift(self, xy, dxy):
        xy = jnp.asarray(xy)
        dxy = jnp.asarray(dxy)
        x_a, y_a = xy[..., 0], xy[..., 1]
        x_ap, y_ap = dxy[..., 0], dxy[..., 1]
        coeffs = self.coeffs
        f = self.focal_length
        rho2 = x_a * x_a + y_a * y_a
        object_plane_dist = self.object_plane_dist
        return -0.5 * rho2 / f - Seidel_aperture_pos_aperture_slope(x_a,
                                                                    y_a,
                                                                    x_ap,
                                                                    y_ap,
                                                                    object_plane_dist,
                                                                    coeffs)

    def complex_action(self, xy: jnp.ndarray, dxy: jnp.ndarray, k: float) -> complex:
        logA = self.log_transmission(xy)
        L = jnp.logaddexp(logA, -50)
        return self.phase_shift(xy, dxy) - 1j * (L / k)

    def __call__(self, ray: GaussianBeam) -> GaussianBeam:
        xy_ref = ray.r_xy
        d_xy = ray.d_xy
        k = ray.k

        dS0, dS1, dS2 = scalar_grad_hess_complex(self.complex_action, xy_ref, d_xy, k)
        r_xy_new, d_xy_new, amplitude_new, pathlength_new, Q_new = apply_action_delta(ray,
                                                                                      dS0=dS0,
                                                                                      dS1=dS1,
                                                                                      dS2=dS2)

        return ray.derive(
            x=r_xy_new[0],
            y=r_xy_new[1],
            dx=d_xy_new[0],
            dy=d_xy_new[1],
            z=ray.z,
            amplitude=amplitude_new,
            pathlength=pathlength_new,
            Q_inv=Q_new,
        )


@jdc.pytree_dataclass(kw_only=True)
class DistortedLens(SeidelLens):
    IsoDist: float = 0.0  # distortion coefficient
    AnisoDist: float = 0.0  # anisotropic distortion coefficient

    def phase_shift(self, xy, dxy):
        xy = jnp.asarray(xy)
        dxy = jnp.asarray(dxy)
        x_a, y_a = xy[..., 0], xy[..., 1]

        f = self.focal_length
        rho2 = x_a * x_a + y_a * y_a

        x_ap, y_ap = dxy[..., 0], dxy[..., 1]
        x_a, y_a = xy[..., 0], xy[..., 1]

        f = self.focal_length
        rho2 = x_a * x_a + y_a * y_a

        x_ap, y_ap = dxy[..., 0], dxy[..., 1]
        coeffs = SeidelCoeffs(E=self.IsoDist, e=self.AnisoDist)
        object_plane_dist = self.object_plane_dist
        return -0.5 * rho2 / f - Seidel_aperture_pos_aperture_slope(x_a,
                                                                    y_a,
                                                                    x_ap,
                                                                    y_ap,
                                                                    object_plane_dist,
                                                                    coeffs)


@jdc.pytree_dataclass
class SigmoidAperture(Component):
    radius: float = 1.0
    edge_width: float = 0.5
    sharpness: float = 1.0
    t_low: float = 0.0
    t_high: float = 1.0
    x0: float = 0.0
    y0: float = 0.0
    eps: float = 1e-15

    def phase_shift(self, xy):
        return 0.0

    def log_transmission(self, xy):
        x, y = xy[0] - self.x0, xy[1] - self.y0

        rho = jnp.sqrt(x * x + y * y + self.eps * self.eps) - self.eps

        w = jnp.maximum(jnp.abs(self.edge_width), self.eps)
        s = jnn.sigmoid(self.sharpness * (rho - self.radius) / w)
        t = self.t_high - (self.t_high - self.t_low) * s
        t_clamped = jnp.clip(t, self.eps, None)
        return jnp.log(t_clamped)


@jdc.pytree_dataclass(kw_only=True)
class Biprism(Component):
    strength: float
    width: float
    length: float | None = None
    theta: float = 0.0
    x0: float = 0.0
    y0: float = 0.0
    sharpness: float = 50.0
    eps: float = 1e-15

    def _uv(self, xy: jnp.ndarray):
        x, y = xy[0], xy[1]
        xr, yr = x - self.x0, y - self.y0
        c, s = jnp.cos(self.theta), jnp.sin(self.theta)
        u = c * xr + s * yr
        v = -s * xr + c * yr
        return u, v

    def phase_shift(self, xy: jnp.ndarray):
        u, _ = self._uv(xy)
        hu = 0.5 * self.width
        eps_u = self.eps * hu
        au = jnp.sqrt(u * u + eps_u * eps_u)
        return -self.strength * au

    def log_transmission(self, xy: jnp.ndarray):
        u, v = self._uv(xy)

        hu = 0.5 * self.width
        eps_u = self.eps * hu
        au = jnp.sqrt(u * u + eps_u * eps_u)
        tx = self.sharpness * (au - hu)
        logA_u = -softplus(-tx)  # smooth rectangular stop in u

        length_is_none = self.length is None

        def _with_length(_):
            length = jnp.asarray(self.length)
            hv = 0.5 * length
            eps_v = self.eps * hu
            av = jnp.sqrt(v * v + eps_v * eps_v)
            ty = self.sharpness * (av - hv)
            return -softplus(-ty)

        def _no_length(_):
            return jnp.asarray(0.0)

        logA_v = lax.cond(length_is_none, _no_length, _with_length, operand=None)

        return logA_u + logA_v


@jdc.pytree_dataclass(kw_only=True)
class ConstantPhaseShift(Component):
    constant_phase_shift: float

    def phase_shift(self, xy: jnp.ndarray):
        return self.constant_phase_shift


@jdc.pytree_dataclass(kw_only=True)
class LinearPhaseShift(Component):
    linear_phase_shift: jnp.ndarray

    def phase_shift(self, xy: jnp.ndarray):
        return jnp.dot(self.linear_phase_shift, xy)


@jdc.pytree_dataclass(kw_only=True)
class QuadraticPhaseShift(Component):
    quadratic_phase_shift: jnp.ndarray

    def phase_shift(self, xy: jnp.ndarray):
        return 0.5 * xy @ self.quadratic_phase_shift @ xy


@jdc.pytree_dataclass(kw_only=True)
class ConstantAmplitudeShift(Component):
    amplitude: float

    def log_transmission(self, xy: jnp.ndarray):
        return jnp.log(self.amplitude)


@jdc.pytree_dataclass(kw_only=True)
class LinearAmplitudeShift(Component):
    linear_amplitude: jnp.ndarray

    def log_transmission(self, xy: jnp.ndarray):
        return jnp.dot(self.linear_log_amplitude, xy)


@jdc.pytree_dataclass(kw_only=True)
class QuadraticAmplitudeShift(Component):
    quadratic_log_amplitude: jnp.ndarray

    def log_transmission(self, xy: jnp.ndarray):
        return 0.5 * xy @ self.quadratic_log_amplitude @ xy


@jdc.pytree_dataclass(kw_only=True)
class MagneticPhaseSample(Component):
    """
    Smooth magnetic phase mask with an internal textured profile.

    Parameters
    ----------
    strength : float
        Peak optical path-length change in metres applied near the centre.
    width, height : float
        Extents of the rectangle (metres) before optional rotation.
    x0, y0 : float
        Centre of the phase object in laboratory coordinates (metres).
    theta : float
        Rotation angle (radians) applied counter-clockwise.
    edge_sharpness : float
        Steepness of the soft-rectangle edges (1/metre). Higher → sharper.
    modulation_strength, skew_strength, radial_strength : float
        Coefficients for internal phase structure to mimic magnetic texture.
    eps : float
        Small constant to keep divisions numerically stable.
    """
    strength: float
    width: float
    height: float
    x0: float = 0.0
    y0: float = 0.0
    theta: float = 0.0
    edge_sharpness: float = 5e6
    modulation_strength: float = 0.3
    skew_strength: float = 0.2
    radial_strength: float = 0.15
    eps: float = 1e-9

    def _local_coords(self, xy):
        x = xy[0] - self.x0
        y = xy[1] - self.y0
        c = jnp.cos(self.theta)
        s = jnp.sin(self.theta)
        u = c * x + s * y
        v = -s * x + c * y
        return u, v

    def _soft_indicator(self, coord, half_extent):
        sharp = self.edge_sharpness
        pos = jax.nn.sigmoid(sharp * (coord + half_extent))
        neg = jax.nn.sigmoid(sharp * (coord - half_extent))
        plateau = jax.nn.sigmoid(sharp * half_extent) - jax.nn.sigmoid(-sharp * half_extent)
        plateau = jnp.maximum(plateau, 1e-9)
        return (pos - neg) / plateau

    def phase_shift(self, xy):
        u, v = self._local_coords(xy)

        hx = 0.5 * self.width
        hy = 0.5 * self.height

        mask = self._soft_indicator(u, hx) * self._soft_indicator(v, hy)

        u_norm = u / (hx + self.eps)
        v_norm = v / (hy + self.eps)
        radial = jnp.sqrt(u_norm * u_norm + v_norm * v_norm + self.eps)

        texture = jnp.sin(jnp.pi * u_norm) * jnp.cos(jnp.pi * v_norm)
        skew = u_norm * v_norm
        radial_term = radial - 0.5

        profile = (
            1.0
            + self.modulation_strength * texture
            + self.skew_strength * skew
            + self.radial_strength * radial_term
        )

        return self.strength * mask * profile


@jdc.pytree_dataclass(kw_only=True)
class InterpolatedSample2D(Component):
    interpolator: Interpolator2D
    method: jdc.Static[str] = "catmull-rom"

    @classmethod
    def from_array(cls, sample, x_coords, y_coords, *, z=0.0, method="cubic"):
        interpolator = Interpolator2D(
            x=x_coords,
            y=y_coords,
            f=sample,
            method=method,
            extrap=1.0, # transmission = 1 outside the sample - no attenuation
        )
        return cls(z=z, interpolator=interpolator, method=method)

    def phase_shift(self, xy):
        z = self.evaluate_complex(xy)
        return jnp.angle(z)

    def log_transmission(self, xy):
        z = self.evaluate_complex(xy)
        amp = jnp.abs(z)
        amp_clamped = jnp.maximum(amp, 1e-15)
        # If amp = 0, return log(1) = 0 to preserve input amplitude
        return jnp.where(amp < 1e-15, 0.0, jnp.log(amp_clamped))

    def evaluate_complex(self, xy):
        return self.interpolator(xy[0], xy[1])


@jdc.pytree_dataclass(kw_only=True)
class InterpolatedFields3D(Component):
    interpolator: Interpolator3D
    method: jdc.Static[str] = "catmull-rom"

    @classmethod
    def from_array(cls, fields, x_coords, y_coords, z_coords, *, method="cubic"):
        """
        fields: array with shape (Nx, Ny, Nz, C)
                e.g. C=2 for [V, A_z] or [V, generator_mag]
        """
        interpolator = Interpolator3D(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            f=fields,
            method=method,
            extrap=0.0,
        )
        return cls(interpolator=interpolator, method=method)

    def evaluate(self, xyz):
        x, y, z = xyz
        return self.interpolator(x, y, z)  # returns (..., C)


@jdc.pytree_dataclass(kw_only=True)
class AtomicPotential():
    atom_xyz: jnp.ndarray  # atom position
    element_params: jnp.ndarray
    cutoff_radius: float  # In angstroms

    def complex_action(self, xy: jnp.ndarray, z: jnp.ndarray, sigma, k: float) -> complex:
        logA = self.log_transmission(xy)
        L = jnp.logaddexp(logA, -50)
        return self.phase_shift(xy, z, sigma, k) - 1j * (L / k)

    def log_transmission(self, xy):
        return 0.0

    def phase_shift(self, xy: jnp.ndarray, z, sigma, k: float) -> complex:
        x, y = xy[0], xy[1]
        r = jnp.sqrt((x - self.atom_xyz[0])**2 + (y - self.atom_xyz[1])**2 +
                     (z - self.atom_xyz[2])**2)
        V = potential_smoothed(r, self.element_params, self.cutoff_radius)

        interaction_constant = sigma / k  # Interaction constant is in radians originally, so we divide by k to get metres
        return -interaction_constant * V

    def __call__(self, ray: GaussianBeam) -> GaussianBeam:
        xy_ref = ray.r_xy
        z = ray.z
        k = ray.k
        sigma = ray.sigma

        dS0, dS1, dS2 = scalar_grad_hess_complex(
            self.complex_action, xy_ref, z, sigma, k
        )

        r_xy, d_xy, amplitude, pathlength, Q_new = apply_action_delta(
            ray, dS0=dS0, dS1=dS1, dS2=dS2
        )

        return ray.derive(
            x=r_xy[0],
            y=r_xy[1],
            dx=d_xy[0],
            dy=d_xy[1],
            z=ray.z,
            amplitude=amplitude,
            pathlength=pathlength,
            Q_inv=Q_new,
        )


@jdc.pytree_dataclass(kw_only=True)
class FourierTransform:
    """
    Meta-component that performs: free-space(f) -> thin lens(f) -> free-space(f),
    which approximates a Fourier transform for a Gaussian beam when the distances
    before and after the lens equal the lens focal length `f`.

    Parameters
    ----------
    f : float | jnp.ndarray
        Focal length (can be scalar or per-ray array).
    x0, y0 : float
        Lens centre offset.
    """
    f: float | jnp.ndarray
    x0: float = 0.0
    y0: float = 0.0

    def __call__(self, ray: GaussianBeam) -> GaussianBeam:
        fs = FreeSpacePropagator()
        # propagate to lens plane
        ray = fs(ray, self.f)
        # apply quadratic phase of a thin lens with focal length f
        lens = Lens(focal_length=self.f, x0=self.x0, y0=self.y0)
        ray = lens(ray)
        # propagate to image plane
        ray = fs(ray, self.f)
        return ray


TransformT = Callable[[Any], Callable[[Any], Tuple[Any, Any]]]


def passthrough_transform(component):
    def inner(ray):
        out = component(ray)
        return out, out
    return inner


class Propagator(NamedTuple):
    distance: float
    propagator: "BaseGaussianPropagator"

    def __call__(self, ray: "GaussianBeam") -> "GaussianBeam":
        return self.propagator(ray, self.distance)


class BaseGaussianPropagator:
    """Abstract base for gaussian-beam propagators.

    Implement `__call__(ray, distance)` in subclasses to return a new GaussianBeam.
    """
    def __call__(self, ray: "GaussianBeam", distance: float) -> "GaussianBeam":
        raise NotImplementedError

    def with_distance(self, distance: float) -> Propagator:
        return Propagator(distance, self)


class FreeSpacePropagator(BaseGaussianPropagator):
    """Full gaussian-beam free-space propagation (2D)."""

    def __call__(self, ray: "GaussianBeam", distance: float) -> "GaussianBeam":
        # Local aliases
        theta = ray.d_xy  # (2,) real
        Q = ray.Q_inv  # (2,2) complex

        # ABCD for Q_inv
        Identity = jnp.eye(2, dtype=jnp.complex128)
        A = Identity + distance * Q  # (2,2) complex
        invA = jnp.linalg.solve(A.T, Identity).T
        detA = jnp.linalg.det(A)

        # New curvature
        Q_new = Q @ invA

        # Centre translation
        r_xy_new = ray.r_xy + distance * theta

        # Pathlength update:
        # + distance (on-axis propagation)
        # + distance * 0.5 * |theta|^2 (obliquity / extra path from tilt)
        theta_sq = jnp.dot(theta, theta)
        pathlength_new = ray.pathlength + distance + 0.5 * distance * theta_sq

        # Amplitude prefactor: det(I + z Q)^(-1/2)
        # This carries both amplitude change and Gouy-like phase.
        amplitude_new = ray.amplitude * detA**(-0.5)

        return ray.derive(
            x=r_xy_new[0],
            y=r_xy_new[1],
            dx=theta[0],
            dy=theta[1],
            z=ray.z + distance,
            amplitude=amplitude_new,
            pathlength=pathlength_new,
            Q_inv=Q_new,
        )


def run_iter(
    ray: GaussianBeam,
    components: Sequence[Any],
    transform: TransformT = passthrough_transform,
    propagator: BaseGaussianPropagator = FreeSpacePropagator(),
) -> Tuple[GaussianBeam, Tuple[GaussianBeam, ...]]:

    rays = []
    r = ray
    for component in components:
        if isinstance(component, (Component, Detector)):
            distance = component.z - r.z
            prop_d = propagator.with_distance(distance)
            r, _out = transform(prop_d)(r)
            rays.append(r)

        r, _out = transform(component)(r)
        rays.append(r)

    return rays


def run_to_end(
    ray: GaussianBeam,
    components: Sequence[Any],
    transform: TransformT = passthrough_transform,
    propagator: BaseGaussianPropagator = FreeSpacePropagator(),
) -> GaussianBeam:
    r = ray
    for component in components:
        if isinstance(component, (Component, Detector)):
            distance = component.z - r.z
            prop_d = propagator.with_distance(distance)
            r, _ = transform(prop_d)(r)
        r, _ = transform(component)(r)
    return r


run_iter_vmapped = jax.jit(jax.vmap(run_iter, in_axes=(0, None)), static_argnums=(3))
run_to_end_vmapped = jax.jit(jax.vmap(run_to_end, in_axes=(0, None)), static_argnums=(3))


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
    offset_xy: Tuple[float, float] = (0.0, 0.0),
    wavelength_unit: str = "m",
) -> GaussianBeam:
    """
    Create a circular distribution of Gaussian beams within a radius R.

    Number of rays is estimated from aperture area, waist, and overlap_factor:

        N ≈ π R^2 / ( (waist / overlap_factor)^2 )
    """

    # --- estimate number of rays ---
    d = waist / overlap_factor
    area = jnp.pi * aperture_radius**2
    num_rays = int(jnp.ceil(area / (d * d)))

    # --- sample rays in the aperture ---
    if sampling.lower() == "fibonacci":
        x0, y0 = fibonacci_spiral(num_rays, aperture_radius)
    else:
        x0, y0 = uniform_disk(num_rays, aperture_radius)

    # apply offset
    x0 = x0 + offset_xy[0]
    y0 = y0 + offset_xy[1]

    # --- amplitude per ray ---
    # ensures energy in overlap area matches continuous beam
    amp_per_ray = uniform_amp_from_area(num_rays, waist, area)

    # --- construct Gaussian beams ---
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


def square_input_wave(
    aperture_length: float,
    waist: float,
    voltage: float,
    amp: float = 1.0,
    phase: float = 0.0,
    z0: float = 0.0,
    overlap_factor: float = 2.0,
    centre_xy: Tuple[float, float] = (0.0, 0.0),
    wavelength_unit: str = "m",
) -> GaussianBeam:

    d = waist / overlap_factor
    Nx = int(jnp.ceil(aperture_length / d))
    Ny = int(jnp.ceil(aperture_length / d))

    xs = (jnp.arange(Nx) - 0.5 * (Nx - 1)) * d
    ys = (jnp.arange(Ny) - 0.5 * (Ny - 1)) * d
    X, Y = jnp.meshgrid(xs, ys, indexing="ij")
    x0 = X.ravel()
    y0 = Y.ravel()

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
    centre_xy: Tuple[float, float] = (0.0, 0.0),
    wavelength_unit: str = "m",
) -> GaussianBeam:
    """
    Create a rectangular array of Gaussian rays covering a rectangle of given
    width and height. The point distribution uses the same lattice generator
    as the square version but scales it to the requested rectangular extents.
    """

    area = aperture_width * aperture_height
    Lx, Ly = aperture_width, aperture_height
    d = waist / overlap_factor  # spacing between centers

    Nx = int(jnp.ceil(Lx / d)) + 1
    Ny = int(jnp.ceil(Ly / d)) + 1
    num_rays = Nx * Ny

    xs = (jnp.arange(Nx) - 0.5 * (Nx - 1)) * d
    ys = (jnp.arange(Ny) - 0.5 * (Ny - 1)) * d
    X, Y = jnp.meshgrid(xs, ys, indexing="ij")
    x0 = X.ravel()
    y0 = Y.ravel()

    amp = uniform_amp_from_area(num_rays, waist, area)

    x0 = x0 + centre_xy[0]
    y0 = y0 + centre_xy[1]
    beam = make_gaussian(
        x=x0,
        y=y0,
        dx=jnp.zeros_like(x0),
        dy=jnp.zeros_like(y0),
        amp=jnp.ones_like(x0) * amp,
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
