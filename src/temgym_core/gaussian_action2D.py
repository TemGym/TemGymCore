import dataclasses
import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax.nn import softplus
import jax_dataclasses as jdc
from jax import lax

from temgym_core.components import Component, Detector
from temgym_core.aberrations import KrivanekCoeffs, Seidel_aperture_pos_aperture_slope, SeidelCoeffs, W_krivanek
from .ray import Ray
from typing import Any, Callable, Generator, NamedTuple, Optional, Sequence, Tuple

from ase import units

from .utils import energy2wavelength, fibonacci_spiral, uniform_disk


def relativistic_mass_correction(energy: float) -> float:
    return 1 + units._e * energy / (units._me * units._c**2)


def _sym(M): return 0.5 * (M + jnp.swapaxes(M, -1, -2))


def center_shift_from_S(S1, S2):
    # We need to find the location of the intensity centre of our gaussian.
    # This might not neccessarily be where the ray is located if for instance we have
    # just passed through a sigmoid aperture - which has the effect of modifying the imaginary part
    # of the action S. This can introduce a linear imaginary action, which means that the intensity
    # centre of the action S. This can introduce a linear imaginary action, which means that the intensity centre of the
    # gaussian no longer aligns with the ray position.
    # This function uses the gradient of the imaginary part of the action to find the intensity centre.
    ImS2 = 0.5 * (jnp.imag(S2) + jnp.imag(S2).T)  # symmetric real
    ImS1 = jnp.imag(S1)
    xi = - jnp.linalg.solve(ImS2, ImS1)
    return xi


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
) -> "GaussianBeam":

    wavelength = energy2wavelength(voltage)
    k = 2.0 * jnp.pi / wavelength

    # --- get batch size from x, then broadcast all 1D params to (n_rays,) ---
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
    k = _bcast_to_n(2.0 * jnp.pi / energy2wavelength(voltage))

    # --- build S2 with leading batch axis ---
    S2_re = jnp.zeros((n_rays, 2, 2), dtype=jnp.float64)
    S2_re = S2_re.at[:, 0, 0].set(curv_x)
    S2_re = S2_re.at[:, 1, 1].set(curv_y)

    S2_im = jnp.zeros((n_rays, 2, 2), dtype=jnp.float64)
    S2_im = S2_im.at[:, 0, 0].set(2.0 / (k * waist_x**2))
    S2_im = S2_im.at[:, 1, 1].set(2.0 / (k * waist_y**2))

    S2 = (S2_re + 1j * S2_im).astype(jnp.complex128)

    amp = _bcast_to_n(amp)
    phase = _bcast_to_n(phase)
    C = jnp.asarray(amp) * jnp.exp(1j * jnp.asarray(phase))

    ray = GaussianBeam(
        x=x, y=y, dx=dx, dy=dy, z=z,
        C=C, S2=S2, voltage=voltage,
        pathlength=jnp.zeros_like(x),
        _one=jnp.ones_like(x),
    ).to_vector()

    if n_rays == 1:
        def squeeze0(a):
            if a is None:
                return None
            a = jnp.asarray(a)
            return jnp.squeeze(a, axis=0) if (a.ndim > 0 and a.shape[0] == 1) else a
        ray = jax.tree.map(squeeze0, ray)

    return ray


@jdc.pytree_dataclass(kw_only=True)
class GaussianBeam(Ray):
    C: jnp.ndarray | complex
    S2: jnp.ndarray
    voltage: jnp.ndarray | float | None = None

    def derive(self,
               x: float | jnp.ndarray | None = None,
               y: float | jnp.ndarray | None = None,
               dx: float | jnp.ndarray | None = None,
               dy: float | jnp.ndarray | None = None,
               z: float | jnp.ndarray | None = None,
               C: jnp.ndarray | complex | None = None,
               S2: jnp.ndarray | None = None,
               voltage: float | jnp.ndarray | None = None,
               pathlength: float | jnp.ndarray | None = None
               ) -> "GaussianBeam":

        return GaussianBeam(
            x=self.x if x is None else x,
            y=self.y if y is None else y,
            dx=self.dx if dx is None else dx,
            dy=self.dy if dy is None else dy,
            z=self.z if z is None else z,
            C=self.C if C is None else C,
            S2=self.S2 if S2 is None else S2,
            voltage=self.voltage if voltage is None else voltage,
            pathlength=self.pathlength if pathlength is None else pathlength
        )

    def to_vector(self) -> jnp.ndarray:
        params = {
            k: jnp.atleast_1d(v)
            for k, v
            in dataclasses.asdict(self).items()
        }
        return type(self)(**params)

    @property
    def wavelength(self) -> float:
        return energy2wavelength(self.voltage)

    @property
    def mass(self) -> float:
        return relativistic_mass_correction(self.voltage) * units._me

    @property
    def sigma(self) -> float:
        return (
            2
            * jnp.pi
            * self.mass
            * units.kg
            * units._e
            * units.C
            * self.wavelength
            / (units._hplanck * units.s * units.J) ** 2
        )

    @property
    def k(self) -> float:
        return 2 * jnp.pi / self.wavelength


def apply_action_delta(
    ray: "GaussianBeam",
    dS0: complex,
    dS1: jnp.ndarray,  # shape (2,)
    dS2: jnp.ndarray,  # shape (2,2)
):
    """
    Apply ΔS(ξ) = dS0 + dS1·ξ + 1/2 ξᵀ dS2 ξ at the current local coords ξ=x - r0,
    then re-center so that Im(S1'+S2' ξ_c) = 0 (intensity maximum at the ray).
    Returns updated (r_xy_new, d_xy_new, C_new, S2_new).
    """
    k = ray.k

    # Total linear/quadratic coefficients *after* the increment
    S1_total = ray.d_xy + dS1
    S2_total = ray.S2 + dS2

    # Choose the re-centering shift to kill the imaginary linear term
    dr_i = center_shift_from_S(S1_total, S2_total)

    # New center (take the real part; imaginary part is a gauge-like tilt in amplitude)
    r_xy_new = ray.r_xy + jnp.real(dr_i)

    # Updated linear coefficient at the new center; store real part as the ray slope
    S1_new = S1_total + S2_total @ dr_i
    d_xy_new = jnp.real(S1_new)

    # Constant term increment to apply to C (only from ΔS, evaluated at ξ=dr_i)
    action_update = (
        dS0
        + jnp.dot(S1_total, dr_i)              # (S1 + dS1)·ξ
        + 0.5 * (dr_i @ S2_total @ dr_i)       # 1/2 ξᵀ (S2 + dS2) ξ
    )

    C_new = ray.C * jnp.exp(1j * k * action_update)
    S2_new = S2_total

    return r_xy_new, d_xy_new, C_new, S2_new


def scalar_grad_hess_complex(
    fn: Callable[..., complex],
    x: jnp.ndarray,
    *args: Any
) -> Tuple[complex, jnp.ndarray, jnp.ndarray]:
    """
    Return (dS0, grad, hess) where dS0 = fn(x, *args) (complex scalar),
    grad = ∇_x fn (complex vector), hess = sym(∇^2_x fn) (complex matrix).
    """
    # evaluate function at x for dS0
    dS0 = fn(x, *args)

    def re_fn(y, *a):  # scalar real
        return jnp.real(fn(y, *a))

    def im_fn(y, *a):  # scalar real
        return jnp.imag(fn(y, *a))

    grad_re = jax.grad(re_fn, argnums=0)(x, *args)  # (2,)
    grad_im = jax.grad(im_fn, argnums=0)(x, *args)  # (2,)
    hess_re = jax.hessian(re_fn, argnums=0)(x, *args)  # (2,2)
    hess_im = jax.hessian(im_fn, argnums=0)(x, *args)  # (2,2)

    grad = grad_re + 1j * grad_im
    hess = _sym(hess_re + 1j * hess_im)
    return dS0, grad, hess


@jdc.pytree_dataclass
class Component2D:
    z: float = 0.0

    def phase_shift(self, xy: jnp.ndarray):
        return 0.0

    def log_transmission(self, xy: jnp.ndarray):
        return 0.0

    def complex_action(self, xy: jnp.ndarray, k: float) -> complex:
        logA = self.log_transmission(xy)
        L = jnp.logaddexp(logA, -20)
        return self.phase_shift(xy) - 1j * (L / k)

    def _apply_single(self, ray: GaussianBeam) -> GaussianBeam:
        xy_ref = jnp.asarray(ray.r_xy, dtype=jnp.float64)
        if xy_ref.ndim != 1:
            raise ValueError("Component2D._apply_single expects a scalar GaussianBeam.")
        k = jnp.squeeze(jnp.asarray(ray.k))

        dS0, dS1, dS2 = scalar_grad_hess_complex(self.complex_action, xy_ref, k)
        r_xy, r_dxy, Cn, S2 = apply_action_delta(ray, dS0=dS0, dS1=dS1, dS2=dS2)

        return ray.derive(
            x=r_xy[..., 0],
            y=r_xy[..., 1],
            dx=r_dxy[..., 0],
            dy=r_dxy[..., 1],
            z=ray.z,
            C=Cn,
            S2=S2
        )

    def __call__(self, ray: GaussianBeam) -> GaussianBeam:
        xy_ref = jnp.asarray(ray.r_xy, dtype=jnp.float64)
        if xy_ref.ndim == 1:
            return self._apply_single(ray)

        batch = xy_ref.shape[0]

        def infer_axes(arr):
            if arr is None:
                return None
            arr = jnp.asarray(arr)
            if arr.ndim == 0:
                return None
            return 0 if arr.shape[0] == batch else None

        in_axes = jax.tree_map(infer_axes, ray)
        vmapped = jax.vmap(lambda r: self._apply_single(r), in_axes=in_axes)
        return vmapped(ray)


@jdc.pytree_dataclass(kw_only=True)
class Lens(Component2D):
    focal_length: float
    x0: float = 0.0
    y0: float = 0.0

    def phase_shift(self, xy: jnp.ndarray):
        x, y = xy[0] - self.x0, xy[1] - self.y0
        rho2 = x * x + y * y
        return -0.5 * rho2 / self.focal_length


@jdc.pytree_dataclass(kw_only=True)
class AberratedLens2D(Component2D):
    focal_length: float
    cubic_coeff: float = 0.0
    quartic_coeff: float = 0.0
    x0: float = 0.0
    y0: float = 0.0
    eps = 1e-14

    def phase_shift(self, xy: jnp.ndarray):
        x, y = xy[0] - self.x0, xy[1] - self.y0
        rho2 = x * x + y * y
        rho3 = rho2 * jnp.sqrt(rho2 + self.eps)
        rho4 = rho2 * rho2
        phase = -0.5 * rho2 / self.focal_length
        phase = phase + self.cubic_coeff * rho3
        phase = phase + self.quartic_coeff * rho4
        return phase


@jdc.pytree_dataclass(kw_only=True)
class KrivanekLens(Component2D):
    """Thin lens with Krivanek aberration model applied to the phase."""
    focal_length: float
    coeffs: jdc.Static[KrivanekCoeffs]
    x0: float = 0.0
    y0: float = 0.0
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
class SeidelLens(Component2D):
    f: float
    z1: float  # absolute distance from object to lens
    coeffs: SeidelCoeffs | None = None  # optional full Seidel coefficients

    def log_transmission(self, xy):
        return 0.0

    def phase_shift(self, xy, dxy):
        x_a, y_a = xy[:, 0], xy[:, 1]
        x_ap, y_ap = dxy[:, 0], dxy[:, 1]
        coeffs = self.coeffs if self.coeffs is not None else SeidelCoeffs()
        return Seidel_aperture_pos_aperture_slope(x_a, y_a, x_ap, y_ap, self.z1, coeffs)

    def complex_action(self, xy: jnp.ndarray, k: float) -> complex:
        logA = self.log_transmission(xy)
        L = jnp.logaddexp(logA, -20)
        return self.phase_shift(xy) - 1j * (L / k)

    def _apply_single(self, ray: GaussianBeam) -> GaussianBeam:
        xy_ref = jnp.asarray(ray.r_xy, dtype=jnp.float64)
        if xy_ref.ndim != 1:
            raise ValueError("Component2D._apply_single expects a scalar GaussianBeam.")
        k = jnp.squeeze(jnp.asarray(ray.k))

        dS0, dS1, dS2 = scalar_grad_hess_complex(self.complex_action, xy_ref, k)
        r_xy, r_dxy, Cn, S2 = apply_action_delta(ray, dS0=dS0, dS1=dS1, dS2=dS2)

        return ray.derive(
            x=r_xy[..., 0],
            y=r_xy[..., 1],
            dx=r_dxy[..., 0],
            dy=r_dxy[..., 1],
            z=ray.z,
            C=Cn,
            S2=S2
        )

    def __call__(self, ray: GaussianBeam) -> GaussianBeam:
        xy_ref = jnp.asarray(ray.r_xy, dtype=jnp.float64)
        if xy_ref.ndim == 1:
            return self._apply_single(ray)

        batch = xy_ref.shape[0]

        def infer_axes(arr):
            if arr is None:
                return None
            arr = jnp.asarray(arr)
            if arr.ndim == 0:
                return None
            return 0 if arr.shape[0] == batch else None

        in_axes = jax.tree_map(infer_axes, ray)
        vmapped = jax.vmap(lambda r: self._apply_single(r), in_axes=in_axes)
        return vmapped(ray)


@jdc.pytree_dataclass(kw_only=True)
class DistortedLens(SeidelLens):
    # Only distortion terms
    E: float = 0.0  # distortion coefficient
    e: float = 0.0  # anisotropic distortion coefficient

    def phase_shift(self, xy, dxy):
        x_a, y_a = xy[:, 0], xy[:, 1]
        x_ap, y_ap = dxy[:, 0], dxy[:, 1]
        coeffs = SeidelCoeffs(A=0.0, B=0.0, C=0.0, D=0.0, E=self.E, F=0.0,
                              e=self.e, f=0.0, c=0.0)
        return Seidel_aperture_pos_aperture_slope(x_a, y_a, x_ap, y_ap, self.z1, coeffs)


@jdc.pytree_dataclass
class SigmoidAperture2D(Component2D):
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
class Biprism(Component2D):
    strength: float
    width: float
    length: float | None = None
    theta: float = 0.0
    x0: float = 0.0
    y0: float = 0.0
    sharpness: float = 50.0
    eps: float = 1e-12

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
        return -self.strength * au  # smooth |u|

    def log_transmission(self, xy: jnp.ndarray):
        u, v = self._uv(xy)

        hu = 0.5 * self.width
        eps_u = self.eps * hu
        au = jnp.sqrt(u * u + eps_u * eps_u)
        tx = self.sharpness * (au - hu)
        logA_u = -softplus(-tx)  # smooth rectangular stop in u

        if self.length is None:
            logA_v = 0.0
        else:
            hv = 0.5 * self.length
            eps_v = self.eps * hu
            av = jnp.sqrt(v * v + eps_v * eps_v)
            ty = self.sharpness * (av - hv)
            logA_v = -softplus(-ty)

        return logA_u + logA_v


@jdc.pytree_dataclass
class ABCDPropagator2D:
    A: jnp.ndarray  # (2,2) real
    B: jnp.ndarray  # (2,2) real
    C: jnp.ndarray  # (2,2) real
    D: jnp.ndarray  # (2,2) real
    L: float = 0.0
    eps: float = 1e-15

    def __call__(self, r: "GaussianBeam") -> "GaussianBeam":
        A, B, C, D = self.A, self.B, self.C, self.D
        k = r.k

        # Quadratic update for the action (keep symmetric to control round-off)
        AB_Q = A + B @ r.S2
        S2 = jnp.linalg.solve(AB_Q.T, (C + D @ r.S2).T).T
        S2 = _sym(S2)

        # Prefactor from the quadratic step: det(A + B S2)^(-1/2), computed stably
        sign, logabs = jnp.linalg.slogdet(AB_Q)
        pref_det = jnp.exp(-0.5 * logabs) / jnp.sqrt(sign)

        # Linear action coefficient prior to re-centering
        S1_temp = jnp.linalg.solve(AB_Q, r.d_xy)

        # Constant action increment for the centred quadratic with a residual linear term
        dS0 = -0.5 * (r.d_xy @ (B @ S1_temp))
        C_temp = r.C * pref_det * jnp.exp(1j * k * (dS0))

        # Re-center so that the imaginary linear coefficient vanishes (intensity maximum)
        dr_i = center_shift_from_S(S1_temp, S2)
        phase_shift = S1_temp @ dr_i + 0.5 * (dr_i @ S2 @ dr_i)
        C_new = C_temp * jnp.exp(1j * k * phase_shift)

        rxy_new = r.r_xy + jnp.real(dr_i)

        S1_new = S1_temp + S2 @ dr_i
        dxy_new = jnp.real(S1_new)

        return r.derive(
            x=rxy_new[..., 0], y=rxy_new[..., 1],
            dx=dxy_new[..., 0], dy=dxy_new[..., 1],
            z=r.z + self.L,
            C=C_new, S2=S2
        )

    @staticmethod
    def free_space(z: float):
        Iden = jnp.eye(2, dtype=jnp.float64)
        Z = jnp.zeros((2, 2), dtype=jnp.float64)
        return ABCDPropagator2D(A=Iden, B=z*Iden, C=Z, D=Iden, L=z)

    @staticmethod
    def thin_lens(fx: float, fy: Optional[float] = None, *, L: float = 0.0):
        if fy is None:
            fy = fx
        Iden = jnp.eye(2, dtype=jnp.float64)
        Z = jnp.zeros((2, 2), dtype=jnp.float64)
        C = jnp.diag(jnp.array([-1.0/fx, -1.0/fy], dtype=jnp.float64))
        return ABCDPropagator2D(A=Iden, B=Z, C=C, D=Iden, L=L)

    @staticmethod
    def rotated_lens(fx: float, fy: float, angle_rad: float, *, L: float = 0.0):
        c, s = jnp.cos(angle_rad), jnp.sin(angle_rad)
        R = jnp.array([[c, -s], [s, c]], dtype=jnp.float64)
        Iden = jnp.eye(2, dtype=jnp.float64)
        Z = jnp.zeros((2, 2), dtype=jnp.float64)
        C = R.T @ jnp.diag(jnp.array([-1.0/fx, -1.0/fy], dtype=jnp.float64)) @ R
        return ABCDPropagator2D(A=Iden, B=Z, C=C, D=Iden, L=L)

    @staticmethod
    def fourier_transform(f: float):
        Iden = jnp.eye(2, dtype=jnp.float64)
        Zero = jnp.zeros((2, 2), dtype=jnp.float64)
        return ABCDPropagator2D(A=Zero, B=f*Iden, C=-(1.0/f)*Iden, D=Zero, L=2*f)

    @staticmethod
    def perfect_imaging(magnification: float, *, L: float = 0.0):
        M = complex(magnification)
        A = jnp.eye(2, dtype=jnp.float64) * M
        D = jnp.eye(2, dtype=jnp.float64) * (1.0/M)
        Z = jnp.zeros((2, 2), dtype=jnp.float64)
        return ABCDPropagator2D(A=A, B=Z, C=Z, D=D, L=L)


TransformT = Callable[[Any], Callable[[Any], Tuple[Any, Any]]]


def passthrough_transform(component):
    def inner(ray):
        out = component(ray)
        return out, out
    return inner


class BaseGaussianPropagator2D:
    def propagate(self, ray: GaussianBeam, distance: float):
        raise NotImplementedError

    def with_distance(self, distance: float):
        return GaussianPropagator2D(distance, self)


class GaussianPropagator2D(NamedTuple):
    distance: float
    propagator: BaseGaussianPropagator2D

    def __call__(self, ray: GaussianBeam):
        return self.propagator.propagate(ray, self.distance)


class FreeSpaceParaxial2D(BaseGaussianPropagator2D):
    def propagate(self, ray: GaussianBeam, distance: float):
        return ABCDPropagator2D.free_space(distance)(ray)


def run_iter(
    ray: GaussianBeam,
    components: Sequence[Any],
    transform: TransformT = passthrough_transform,
    propagator: BaseGaussianPropagator2D = FreeSpaceParaxial2D(),
) -> Generator[Tuple[Any, Any], Any, None]:
    for component in components:
        if isinstance(component, (Component2D, Detector)):
            ray_z = ray.z
            distance = component.z - ray_z
            propagator_d = propagator.with_distance(distance)
            ray, out = transform(propagator_d)(ray)
            yield propagator_d, out

        ray, out = transform(component)(ray)
        yield component, out


def run_to_end(
    ray: GaussianBeam,
    components: Sequence[Any],
    propagator: BaseGaussianPropagator2D = FreeSpaceParaxial2D(),
) -> GaussianBeam:
    for _, ray in run_iter(ray, components, propagator=propagator):
        pass
    return ray


def make_gaussian_plane_wave_circular_aperture(
    aperture_radius: float,
    waist: float,
    num_rays: int,
    voltage: float,
    amp: float = 1.0,
    phase: float = 0.0,
    z0: float = 0.0,
    sampling: str = "uniform, fibonacci",
    offset_xy: Tuple[float, float] = (0.0, 0.0)
) -> GaussianBeam:
    if sampling == "fibonacci":
        x0, y0 = fibonacci_spiral(num_rays, aperture_radius)
    else:
        x0, y0 = uniform_disk(num_rays, aperture_radius)
    x0 = x0 + offset_xy[0]
    y0 = y0 + offset_xy[1]
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
    )
    return beam


def make_gaussian_plane_wave_square_aperture(
    aperture_length: float,
    waist: float,
    num_rays: int,
    voltage: float,
    amp: float = 1.0,
    phase: float = 0.0,
    z0: float = 0.0,
    sampling: str = "uniform, fibonacci",
) -> GaussianBeam:
    # Uniform grid sampling over a square [-aperture_length / 2, aperture_length / 2]^2
    n_x = int(jnp.ceil(jnp.sqrt(num_rays)))
    n_y = int(jnp.ceil(num_rays / n_x))

    xs = jnp.linspace(-aperture_length / 2, aperture_length / 2, n_x, dtype=jnp.float64)
    ys = jnp.linspace(-aperture_length / 2, aperture_length / 2, n_y, dtype=jnp.float64)
    X, Y = jnp.meshgrid(xs, ys, indexing="xy")

    x_flat = X.reshape(-1)
    y_flat = Y.reshape(-1)
    x0 = x_flat[:num_rays]
    y0 = y_flat[:num_rays]

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
    )
    return beam


def make_gaussian_grid_input(waist: float,
                             voltage: float,
                             z0: float,
                             amp: float = 1.0,
                             phase: float = 0.0,
                             n_cells: int = 4,
                             samples_per_line: int = 200,
                             extent: float = 1.0):
    """
    Vectorised creation of a square grid figure.
    Returns:
      points   : (N, 2) array of xy points for all grid lines (float32)
      line_ids : (N,) int32 array indicating which line each point belongs to
                 (0..n_lines-1 are vertical lines, n_lines..2*n_lines-1 are horizontal lines)
    """
    n_lines = n_cells + 1  # includes the outer square
    xs = jnp.linspace(-extent, extent, n_lines, dtype=jnp.float32)  # (n_lines,)
    ys = xs
    t = jnp.linspace(-extent, extent, samples_per_line, dtype=jnp.float32)  # (samples,)

    # Vertical lines: x fixed (one per xs), y varies over t
    vert_x = jnp.broadcast_to(xs[:, None], (n_lines, samples_per_line))   # (n_lines, samples)
    vert_y = jnp.broadcast_to(t[None, :], (n_lines, samples_per_line))    # (n_lines, samples)
    vert_pts = jnp.stack([vert_x, vert_y], axis=-1).reshape(-1, 2)        # (n_lines*samples, 2)

    # Horizontal lines: y fixed (one per ys), x varies over t
    hor_x = jnp.broadcast_to(t[None, :], (n_lines, samples_per_line))     # (n_lines, samples)
    hor_y = jnp.broadcast_to(ys[:, None], (n_lines, samples_per_line))    # (n_lines, samples)
    hor_pts = jnp.stack([hor_x, hor_y], axis=-1).reshape(-1, 2)          # (n_lines*samples, 2)

    points = jnp.concatenate([vert_pts, hor_pts], axis=0).astype(jnp.float32)

    x0, y0 = points[:, 0], points[:, 1]

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
    )
    return beam
