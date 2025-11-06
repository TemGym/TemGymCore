import dataclasses
import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax.nn import softplus
import jax_dataclasses as jdc
from jax import lax

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
    Generator,
    NamedTuple,
    Sequence,
    Tuple
)

from ase import units

from .utils import (
    energy2wavelength,
    relativistic_mass_correction,
    _sym,
    fibonacci_spiral,
    grid_line_area,
    lattice_points_square_cover,
    uniform_disk,
    uniform_amp_from_area
)


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

    S2_re = jnp.zeros((n_rays, 2, 2), dtype=jnp.float64)
    S2_re = S2_re.at[:, 0, 0].set(curv_x)
    S2_re = S2_re.at[:, 1, 1].set(curv_y)

    S2_im = jnp.zeros((n_rays, 2, 2), dtype=jnp.float64)
    S2_im = S2_im.at[:, 0, 0].set(wavelength / (jnp.pi * waist_x**2))
    S2_im = S2_im.at[:, 1, 1].set(wavelength / (jnp.pi * waist_y**2))

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


def apply_action_delta(
    ray,
    dS0: complex,
    dS1: jnp.ndarray,  # shape (2,)
    dS2: jnp.ndarray,  # shape (2,2)
):
    """
    Apply a local quadratic action increment ΔS(ξ) = dS0 + dS1·ξ + 1/2 ξᵀ dS2 ξ
    evaluated at the CURRENT ray center (ξ = r - r0, with r0 = ray.r_xy).

    No re-centering; absolute phase lives in ray.C.

    Updates:
      C   <- C * exp{i k dS0}
      d   <- d + Re(dS1)
      S2  <- S2 + sym(dS2)
      r0  <- r0

    Returns
    -------
    r_xy_new, d_xy_new, C_new, S2_new

    """
    k = ray.k
    r0 = ray.r_xy

    C_new = ray.C * jnp.exp(1j * k * dS0)
    d_xy_new = ray.d_xy + jnp.real(dS1)
    S2_new = ray.S2 + dS2
    r_xy_new = r0

    return r_xy_new, d_xy_new, C_new, S2_new


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
    hess = _sym(hess_re + 1j * hess_im)
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

    def _apply_single(self, ray: GaussianBeam) -> GaussianBeam:
        xy_ref = jnp.asarray(ray.r_xy, dtype=jnp.float64)
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
class Lens(Component):
    focal_length: float
    x0: float = 0.0
    y0: float = 0.0

    def phase_shift(self, xy: jnp.ndarray):
        x, y = xy[0] - self.x0, xy[1] - self.y0
        rho2 = x * x + y * y
        return -0.5 * rho2 / self.focal_length


@jdc.pytree_dataclass(kw_only=True)
class KrivanekLens(Component):
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
class SeidelLens(Component):
    focal_length: float
    z1: float  # absolute distance from object to lens
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
        return -0.5 * rho2 / f - Seidel_aperture_pos_aperture_slope(x_a, y_a, x_ap, y_ap, self.z1, coeffs)

    def complex_action(self, xy: jnp.ndarray, dxy: jnp.ndarray, k: float) -> complex:
        logA = self.log_transmission(xy)
        L = jnp.logaddexp(logA, -20)
        return self.phase_shift(xy, dxy) - 1j * (L / k)

    def _apply_single(self, ray: GaussianBeam) -> GaussianBeam:
        xy_ref = jnp.asarray(ray.r_xy, dtype=jnp.float64)
        if xy_ref.ndim != 1:
            raise ValueError("Component._apply_single expects a scalar GaussianBeam.")
        d_xy = jnp.asarray(ray.d_xy, dtype=jnp.float64)
        k = jnp.squeeze(jnp.asarray(ray.k))

        dS0, dS1, dS2 = scalar_grad_hess_complex(self.complex_action, xy_ref, d_xy, k)
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

        return -0.5 * rho2 / f - Seidel_aperture_pos_aperture_slope(x_a, y_a, x_ap, y_ap, self.z1, coeffs)


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
        return -self.strength * au

    # def log_transmission(self, xy: jnp.ndarray):
    #     u, v = self._uv(xy)

    #     hu = 0.5 * self.width
    #     eps_u = self.eps * hu
    #     au = jnp.sqrt(u * u + eps_u * eps_u)
    #     tx = self.sharpness * (au - hu)
    #     logA_u = -softplus(-tx)  # smooth rectangular stop in u

    #     if self.length is None:
    #         logA_v = 0.0
    #     else:
    #         hv = 0.5 * self.length
    #         eps_v = self.eps * hu
    #         av = jnp.sqrt(v * v + eps_v * eps_v)
    #         ty = self.sharpness * (av - hv)
    #         logA_v = -softplus(-ty)

    #     return logA_u + logA_v


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
    """Full gaussian-beam free-space propagation ()."""

    def __call__(self, ray: "GaussianBeam", distance: float) -> "GaussianBeam":
        I = jnp.eye(2, dtype=jnp.float64)
        theta = ray.d_xy
        A = I + distance * ray.S2
        invA = jnp.linalg.solve(A.T, I).T
        detA = jnp.linalg.det(A)
        Cnew = ray.C * (
            jnp.exp(1j * ray.k * distance)
            * detA ** (-0.5)
            * jnp.exp(1j * ray.k * distance * 0.5 * jnp.dot(theta, theta))
        )
        S2new = ray.S2 @ invA
        xy_new = ray.r_xy + distance * theta

        return ray.derive(
            x=xy_new[..., 0],
            y=xy_new[..., 1],
            dx=theta[..., 0],
            dy=theta[..., 1],
            z=ray.z + distance,
            C=Cnew,
            S2=S2new
        )


def run_iter(
    ray: GaussianBeam,
    components: Sequence[Any],
    transform: TransformT = passthrough_transform,
    propagator: BaseGaussianPropagator = FreeSpacePropagator(),
) -> Generator[Tuple[Any, Any], Any, None]:
    for component in components:
        if isinstance(component, (Component, Detector)):
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
    propagator: BaseGaussianPropagator = FreeSpacePropagator(),
) -> GaussianBeam:
    for _, ray in run_iter(ray, components, propagator=propagator):
        pass
    return ray


def circular_input_wave(
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

    area = jnp.pi * aperture_radius * aperture_radius
    amp = uniform_amp_from_area(num_rays, waist, area)

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


def square_input_wave(
    aperture_length: float,
    waist: float,
    num_rays: int,
    voltage: float,
    amp: float = 1.0,
    phase: float = 0.0,
    z0: float = 0.0,
    centre_xy: Tuple[float, float] = (0.0, 0.0),
    sampling: str = "uniform, fibonacci",
) -> GaussianBeam:

    pts = lattice_points_square_cover(num_rays, aperture_length)
    x0, y0 = pts[:, 0], pts[:, 1]

    area = aperture_length * aperture_length
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
    )
    return beam


def grid_input_wave(waist: float,
                    voltage: float,
                    z0: float,
                    amp: float = 1.0,
                    phase: float = 0.0,
                    n_cells: int = 4,
                    samples_per_line: int = 200,
                    extent: float = 1.0,
                    offset_xy: Tuple[float, float] = (0.0, 0.0)) -> GaussianBeam:
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

    N = 2 * n_lines * samples_per_line  # total number of gaussians
    n_lines = n_cells + 1
    A_obj = grid_line_area(extent, n_cells, waist * 2)  # you choose line_width
    amps = A_obj / (N * jnp.pi * waist**2)

    x0, y0 = x0 + offset_xy[0], y0 + offset_xy[1]
    beam = make_gaussian(
        x=x0,
        y=y0,
        dx=jnp.zeros_like(x0),
        dy=jnp.zeros_like(y0),
        amp=amps,
        phase=jnp.zeros_like(y0) + phase,
        waist_x=jnp.ones_like(x0) * waist,
        waist_y=jnp.ones_like(y0) * waist,
        rcurv_x=jnp.ones_like(x0) * jnp.inf,
        rcurv_y=jnp.ones_like(y0) * jnp.inf,
        z=jnp.ones_like(x0) * z0,
        voltage=jnp.ones_like(x0) * voltage,
    )
    return beam
