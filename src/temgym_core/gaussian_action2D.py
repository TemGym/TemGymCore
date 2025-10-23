import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax_dataclasses as jdc
import numpy as np
from typing import Any, Callable, Generator, NamedTuple, Optional, Sequence, Tuple

from .utils import energy2wavelength, fibonacci_spiral, uniform_disk


def _sym(M: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (M + M.T)


def _slogdet_complex(M: jnp.ndarray) -> jnp.ndarray:
    sign, logabsdet = jnp.linalg.slogdet(M)
    return logabsdet + jnp.log(sign + 0j)


@jdc.pytree_dataclass
class GaussianRay2D:
    """
    ψ(x) = C * exp(i k S(x)),  S(x) = S1·(x - r0) + 1/2 (x - r0)^T S2 (x - r0)
    S1 ∈ C^2, S2 ∈ C^{2×2} (symmetric), r0 ∈ R^2
    """
    C: complex
    S1: jnp.ndarray
    S2: jnp.ndarray
    r0: jnp.ndarray
    k: float = 1.0
    z: float = 0.0

    def action(self, xy: jnp.ndarray) -> jnp.ndarray:
        xy = jnp.asarray(xy, dtype=jnp.float64)
        xi = xy - self.r0
        S2s = _sym(self.S2)
        lin = jnp.einsum('...i,i->...', xi, self.S1)
        quad = 0.5 * jnp.einsum('...i,ij,...j->...', xi, S2s, xi)
        return lin + quad

    def field(self, xy: jnp.ndarray) -> jnp.ndarray:
        return self.C * jnp.exp(1j * self.k * self.action(xy))

    def amplitude(self, xy: jnp.ndarray) -> jnp.ndarray:
        return jnp.abs(self.field(xy))

    def intensity(self, xy: jnp.ndarray) -> jnp.ndarray:
        psi = self.field(xy)
        return jnp.abs(psi) ** 2

    def phase(self, xy: jnp.ndarray) -> jnp.ndarray:
        return jnp.angle(self.field(xy))

    def ray_position(self) -> jnp.ndarray:
        S2s = _sym(self.S2)
        xi_c = jnp.linalg.solve(jnp.imag(S2s), -jnp.imag(self.S1))
        return self.r0 + xi_c

    def ray_slope(self) -> jnp.ndarray:
        xi_c = self.ray_position() - self.r0
        p = jnp.real(self.S1 + _sym(self.S2) @ xi_c)
        return p / self.k

    def recenter(self, C, S1, S2, r0, k):
        S1 = jnp.asarray(S1, dtype=jnp.complex128)
        S2s = _sym(jnp.asarray(S2, dtype=jnp.complex128))
        r0 = jnp.asarray(r0, dtype=jnp.float64)
        xi_c = jnp.linalg.solve(jnp.imag(S2s), -jnp.imag(S1))
        x_c = r0 + xi_c
        phase = (S1 @ xi_c) + 0.5 * (xi_c @ (S2s @ xi_c))
        Cn = C * jnp.exp(1j * k * phase)
        S1n = S1 + S2s @ xi_c
        return Cn, S1n, S2s, x_c

    def _with_params(self, C=None, S1=None, S2=None, r0=None, k=None, z=None):
        C = self.C if C is None else C
        S1 = self.S1 if S1 is None else S1
        S2 = self.S2 if S2 is None else S2
        r0 = self.r0 if r0 is None else r0
        k = self.k if k is None else k
        z = self.z if z is None else z
        C, S1, S2, r0 = self.recenter(C, S1, S2, r0, k)
        return GaussianRay2D(C=C, S1=S1, S2=S2, r0=r0, k=k, z=z)

    @staticmethod
    def make_gaussian(
        x: float, y: float,
        dx: float = 0.0, dy: float = 0.0,
        InitAmp: float = 1.0, InitPhase: float = 0.0,
        waist_x: float = 1.0, waist_y: float = 1.0,
        RadiusOfCurvature_x: float = jnp.inf,
        RadiusOfCurvature_y: float = jnp.inf,
        k: float = 1.0, z: float = 0.0,
    ) -> "GaussianRay2D":
        r0 = jnp.array([x, y], dtype=jnp.float64)
        S1 = jnp.array([k * dx, k * dy], dtype=jnp.complex128)

        def curv(R): return 0.0 if jnp.isinf(R) else 1.0 / R
        S2_real = jnp.diag(jnp.array([curv(RadiusOfCurvature_x), curv(RadiusOfCurvature_y)],
                                     dtype=jnp.float64))
        S2_imag = jnp.diag(jnp.array([
            1.0 / (2.0 * k * max(waist_x ** 2, 1e-12)),
            1.0 / (2.0 * k * max(waist_y ** 2, 1e-12)),
        ], dtype=jnp.float64))
        S2 = (S2_real + 1j * S2_imag).astype(jnp.complex128)
        C = InitAmp * jnp.exp(1j * InitPhase)
        return GaussianRay2D(C=C, S1=S1, S2=S2, r0=r0, k=k, z=float(z))


def apply_action_delta(
    ray: GaussianRay2D,
    dS0: complex = 0.0 + 0.0j,
    dS1: jnp.ndarray | complex = 0.0 + 0.0j,  # scalar or (2,)
    dS2: Optional[jnp.ndarray] = None,        # (2,2) or None
    *, z: Optional[float] = None,
) -> GaussianRay2D:

    phi0, phi1, phi2 = jnp.real(dS0), jnp.real(dS1), jnp.real(dS2)
    ell0 = -ray.k * jnp.imag(dS0)
    ell1 = -ray.k * jnp.imag(dS1)
    ell2 = -ray.k * jnp.imag(dS2)

    Cn = ray.C * jnp.exp(ell0 + 1j * ray.k * phi0)
    S1n = ray.S1 + (phi1 - 1j * ell1 / ray.k)
    S2n = _sym(ray.S2 + (phi2 - 1j * ell2 / ray.k))
    return ray._with_params(C=Cn, S1=S1n, S2=S2n, z=z)


def grad_hess_complex(
    fn: Callable[..., complex],
    x: jnp.ndarray,
    *args: Any
) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
    return grad, hess


@jdc.pytree_dataclass
class Component2D:
    z: float = 0.0

    def phase_shift(self, xy: jnp.ndarray):
        return 0.0

    def log_transmission(self, xy: jnp.ndarray):
        return 0.0

    def complex_action(self, xy: jnp.ndarray, k: float) -> complex:
        return self.phase_shift(xy) - 1j * self.log_transmission(xy) / k

    def __call__(self, ray: GaussianRay2D) -> GaussianRay2D:
        xy_ref = jnp.asarray(ray.ray_position(), dtype=jnp.float64)
        k = ray.k

        # Evaluate ΔS and its derivatives at the intensity center:
        dS0 = self.complex_action(xy_ref, k)
        dS1, dS2 = grad_hess_complex(self.complex_action, xy_ref, k)

        out = apply_action_delta(ray, dS0=dS0, dS1=dS1, dS2=dS2)
        return out._with_params(z=float(self.z))


@jdc.pytree_dataclass(kw_only=True)
class ThinLens2D(Component2D):
    focal_length: float
    center: tuple[float, float] = (0.0, 0.0)

    def phase_shift(self, xy: jnp.ndarray):
        rx, ry = self.center
        x, y = xy[0] - rx, xy[1] - ry
        rho2 = x * x + y * y
        return -0.5 * rho2 / self.focal_length


@jdc.pytree_dataclass(kw_only=True)
class AberratedLens2D(Component2D):
    focal_length: float
    cubic_coeff: float = 0.0
    quartic_coeff: float = 0.0
    center: Tuple[float, float] = (0.0, 0.0)
    eps = 1e-14

    def phase_shift(self, xy: jnp.ndarray):
        rx, ry = self.center
        x, y = xy[0] - rx, xy[1] - ry
        rho2 = x * x + y * y
        rho3 = rho2 * jnp.sqrt(rho2 + self.eps)
        rho4 = rho2 * rho2
        phase = -0.5 * rho2 / self.focal_length
        phase = phase + self.cubic_coeff * rho3
        phase = phase + self.quartic_coeff * rho4
        return phase


@jdc.pytree_dataclass
class SigmoidAperture2D(Component2D):
    radius: float = 1.0
    width: float = 0.5
    t_low: float = 0.0
    t_high: float = 1.0
    center: Tuple[float, float] = (0.0, 0.0)
    eps: float = 1e-12
    def phase_shift(self, xy): return 0.0

    def log_transmission(self, xy):
        rx, ry = self.center
        x, y = xy[0] - rx, xy[1] - ry
        rho = jnp.sqrt(x * x + y * y)
        w = jnp.maximum(jnp.abs(self.width), self.eps)
        s = jnn.sigmoid((rho - self.radius) / w)
        t = self.t_high - (self.t_high - self.t_low) * s
        t_clamped = jnp.clip(t, self.eps, None)
        return jnp.log(t_clamped)


@jdc.pytree_dataclass
class ABCDPropagator2D:
    A: jnp.ndarray  # (2,2) complex
    B: jnp.ndarray  # (2,2) complex
    C: jnp.ndarray  # (2,2) complex
    D: jnp.ndarray  # (2,2) complex
    L: float = 0.0
    eps: float = 1e-12

    def __call__(self, ray: GaussianRay2D) -> GaussianRay2D:
        A, B, C, D = self.A, self.B, self.C, self.D
        L = jnp.asarray(self.L, dtype=jnp.float64)

        # quick singularity guard (both ~0 → non-invertible)
        if (jnp.linalg.norm(A) < self.eps) and (jnp.linalg.norm(B) < self.eps):
            raise ValueError("ABCD matrix is singular: A and B both near-zero.")

        # predicate: imaging-like if ||B|| is tiny
        pred = jnp.less(jnp.linalg.norm(B), self.eps)

        def imaging_fn(r: GaussianRay2D):
            den = A  # B ~ 0
            r0p = jnp.real(den @ r.r0)                     # keep center real
            S2p = (C + D @ r.S2) @ jnp.linalg.solve(den, jnp.eye(2, dtype=den.dtype))
            S2p = _sym(S2p)
            S1p = jnp.linalg.solve(den.T, r.S1 + C @ r0p)  # add C r0' term
            dS0 = 0.5 * (r.r0 @ (C @ r0p))                 # scalar complex
            logdet = _slogdet_complex(den)
            Cp = r.C * jnp.exp(-0.5 * logdet) * jnp.exp(1j * r.k * (dS0 + L))
            return r._with_params(C=Cp, S1=S1p, S2=S2p, r0=r0p, z=r.z + self.L)

        def general_fn(r: GaussianRay2D):
            den = A + B @ r.S2
            den_inv = jnp.linalg.solve(den, jnp.eye(2, dtype=den.dtype))
            S2p = (C + D @ r.S2) @ den_inv
            S2p = _sym(S2p)
            S1p = jnp.linalg.solve(den.T, r.S1)
            t = den_inv @ r.S1
            dS0 = -0.5 * (r.S1 @ (B @ t))                  # scalar complex
            logdet = _slogdet_complex(den)
            Cp = r.C * jnp.exp(-0.5 * logdet) * jnp.exp(1j * r.k * (dS0 + L))
            return r._with_params(C=Cp, S1=S1p, S2=S2p, r0=r.r0, z=r.z + self.L)

        return jax.lax.cond(pred, imaging_fn, general_fn, ray)

    @staticmethod
    def free_space(z: float):
        Iden = jnp.eye(2, dtype=jnp.complex128)
        Z = jnp.zeros((2, 2), dtype=jnp.complex128)
        return ABCDPropagator2D(A=Iden, B=z*Iden, C=Z, D=Iden, L=z)

    @staticmethod
    def thin_lens(fx: float, fy: Optional[float] = None, *, L: float = 0.0):
        if fy is None:
            fy = fx
        Iden = jnp.eye(2, dtype=jnp.complex128)
        Z = jnp.zeros((2, 2), dtype=jnp.complex128)
        C = jnp.diag(jnp.array([-1.0/fx, -1.0/fy], dtype=jnp.complex128))
        return ABCDPropagator2D(A=Iden, B=Z, C=C, D=Iden, L=L)

    @staticmethod
    def rotated_lens(fx: float, fy: float, angle_rad: float, *, L: float = 0.0):
        c, s = jnp.cos(angle_rad), jnp.sin(angle_rad)
        R = jnp.array([[c, -s], [s, c]], dtype=jnp.complex128)
        Iden = jnp.eye(2, dtype=jnp.complex128)
        Z = jnp.zeros((2, 2), dtype=jnp.complex128)
        C = R.T @ jnp.diag(jnp.array([-1.0/fx, -1.0/fy], dtype=jnp.complex128)) @ R
        return ABCDPropagator2D(A=Iden, B=Z, C=C, D=Iden, L=L)

    @staticmethod
    def fourier_transform(f: float):
        Iden = jnp.eye(2, dtype=jnp.complex128)
        Z = jnp.zeros((2, 2), dtype=jnp.complex128)
        return ABCDPropagator2D(A=Z, B=f*Iden, C=-(1.0/f)*Iden, D=Z, L=2*f)

    @staticmethod
    def perfect_imaging(magnification: float, *, L: float = 0.0):
        M = complex(magnification)
        A = jnp.eye(2, dtype=jnp.complex128) * M
        D = jnp.eye(2, dtype=jnp.complex128) * (1.0/M)
        Z = jnp.zeros((2, 2), dtype=jnp.complex128)
        return ABCDPropagator2D(A=A, B=Z, C=Z, D=D, L=L)


TransformT = Callable[[Any], Callable[[Any], Tuple[Any, Any]]]


def passthrough_transform(component):
    def inner(ray):
        out = component(ray)
        return out, out
    return inner


class BaseGaussianPropagator2D:
    def propagate(self, ray: GaussianRay2D, distance: float):
        raise NotImplementedError

    def with_distance(self, distance: float):
        return GaussianPropagator2D(distance, self)


class GaussianPropagator2D(NamedTuple):
    distance: float
    propagator: BaseGaussianPropagator2D

    def __call__(self, ray: GaussianRay2D):
        return self.propagator.propagate(ray, self.distance)


class FreeSpaceParaxial2D(BaseGaussianPropagator2D):
    def propagate(self, ray: GaussianRay2D, distance: float):
        if abs(distance) <= 1e-12:
            return ray
        return ABCDPropagator2D.free_space(distance)(ray)


def run_iter(
    ray: GaussianRay2D,
    components: Sequence[Any],
    transform: TransformT = passthrough_transform,
    propagator: BaseGaussianPropagator2D = FreeSpaceParaxial2D(),
) -> Generator[Tuple[Any, Any], Any, None]:
    current_z = float(getattr(ray, "z", 0.0))
    for component in components:
        if isinstance(component, Component2D):
            distance = float(component.z - current_z)
            if abs(distance) > 1e-12:
                propagator_d = propagator.with_distance(distance)
                ray, out = transform(propagator_d)(ray)
                current_z += distance
                yield propagator_d, out
        ray, out = transform(component)(ray)
        if isinstance(component, Component2D):
            current_z = float(component.z)
        yield component, out


def run_to_end(
    ray: GaussianRay2D,
    components: Sequence[Any],
    propagator: BaseGaussianPropagator2D = FreeSpaceParaxial2D(),
) -> GaussianRay2D:
    for _, ray in run_iter(ray, components, propagator=propagator):
        pass
    return ray


@jdc.pytree_dataclass
class GaussianBeam2D:
    rays: tuple[GaussianRay2D, ...]
    coords: jnp.ndarray
    waist: tuple[float, float]
    radius_of_curvature: tuple[float, float]
    voltage: float
    wavelength: float
    k: float

    def __len__(self) -> int:
        return len(self.rays)

    def __iter__(self):
        return iter(self.rays)

    def stack_parameters(self) -> dict[str, jnp.ndarray]:
        if not self.rays:
            empty_complex = jnp.zeros((0,), dtype=jnp.complex128)
            empty_real = jnp.zeros((0,), dtype=jnp.float64)
            empty_vec = jnp.zeros((0, 2), dtype=jnp.float64)
            empty_mat = jnp.zeros((0, 2, 2), dtype=jnp.complex128)
            return {
                "C": empty_complex,
                "S1": empty_mat[..., 0],  # share empty view
                "S2": empty_mat,
                "r0": empty_vec,
                "z": empty_real,
                "k": jnp.zeros((0,), dtype=jnp.float64),
            }
        Cs = jnp.stack([ray.C for ray in self.rays], axis=0)
        S1s = jnp.stack([ray.S1 for ray in self.rays], axis=0)
        S2s = jnp.stack([ray.S2 for ray in self.rays], axis=0)
        r0s = jnp.stack([ray.r0 for ray in self.rays], axis=0)
        zs = jnp.stack([jnp.asarray(ray.z, dtype=jnp.float64) for ray in self.rays], axis=0)
        ks = jnp.full((len(self),), self.k, dtype=jnp.float64)
        return {"C": Cs, "S1": S1s, "S2": S2s, "r0": r0s, "z": zs, "k": ks}


class GaussianBeamFactory2D:
    """
    Factory for constructing collections of :class:`GaussianRay2D` packets
    with configurable sampling over common aperture shapes.
    """

    def __init__(
        self,
        *,
        voltage: float = 200e3,
        waist_radius: float | None = None,
        overlap_factor: float | None = None,
        normalization: str = "unit",
        sampler: Optional[Callable[..., tuple[np.ndarray, np.ndarray]]] = None,
        sampling: str | None = None,
        sampler_kwargs: Optional[dict[str, Any]] = None,
        initial_z: float = 0.0,
        dtype=jnp.float64,
        wavelength: float | None = None,
        k: float | None = None,
    ):
        self.voltage = float(voltage)
        if k is not None and wavelength is not None:
            raise ValueError("Provide either wavelength or k, not both.")
        if k is not None:
            if k <= 0.0:
                raise ValueError("k must be positive.")
            self.k = float(k)
            self.wavelength = float(2.0 * np.pi / self.k)
        else:
            wl = wavelength if wavelength is not None else energy2wavelength(self.voltage)
            if wl <= 0.0:
                raise ValueError("wavelength must be positive.")
            self.wavelength = float(wl)
            self.k = 2.0 * np.pi / self.wavelength
        if waist_radius is not None and waist_radius <= 0.0:
            raise ValueError("waist_radius must be positive when provided.")
        if overlap_factor is not None and overlap_factor <= 0.0:
            raise ValueError("overlap_factor must be positive when provided.")
        if waist_radius is None and overlap_factor is None:
            raise ValueError("Provide either waist_radius or overlap_factor.")
        self.waist_radius = float(waist_radius) if waist_radius is not None else None
        self.overlap_factor = float(overlap_factor) if overlap_factor is not None else None
        self._last_waist_radius = self.waist_radius

        if normalization not in ("unit", "none"):
            raise ValueError(f"Unknown normalization mode: {normalization}")
        self.normalization = normalization

        if sampler is not None and sampling is not None:
            raise ValueError("Provide either sampler or sampling, not both.")
        if sampler is None:
            sampling = sampling or "fibonacci"
            if sampling == "fibonacci":
                sampler = fibonacci_spiral
            elif sampling == "uniform":
                sampler = uniform_disk
            else:
                raise ValueError(f"Unknown sampling mode: {sampling}")
        self.sampler = sampler
        self.sampling_mode = sampling

        self.sampler_kwargs = sampler_kwargs or {}
        self.initial_z = float(initial_z)
        self.dtype = dtype

    @staticmethod
    def _as_pair(value: float | tuple[float, float] | None, *, default: tuple[float, float]) -> tuple[float, float]:
        if value is None:
            return default
        if np.isscalar(value):
            scalar = float(value)
            return scalar, scalar
        if len(value) != 2:
            raise ValueError("Expected a pair of values.")
        return float(value[0]), float(value[1])

    def _resolve_waist_radius(self, num: int, area: float | None) -> float:
        if self.waist_radius is not None:
            value = self.waist_radius
        else:
            if self.overlap_factor is None:
                raise ValueError(
                    "Cannot resolve waist radius without either a manual waist radius "
                    "or an overlap factor."
                )
            if area is None or area <= 0.0:
                raise ValueError("A positive aperture area is required to infer waist radius.")
            if num <= 0:
                raise ValueError("Number of rays must be positive to infer waist radius.")
            value = float(self.overlap_factor * np.sqrt(area / num))
        self._last_waist_radius = value
        return value

    def _resolve_initial_z(self, value: float | None) -> float:
        return self.initial_z if value is None else float(value)

    @property
    def last_waist_radius(self) -> float | None:
        return self._last_waist_radius

    def _prefactor(self, num: int, area: float | None, waist_radius: float) -> jnp.ndarray:
        if self.normalization == "unit":
            return jnp.ones(num, dtype=jnp.complex128)
        if self.normalization == "none":
            return jnp.ones(num, dtype=jnp.complex128)
        return jnp.ones(num, dtype=jnp.complex128)

    def _build(
        self,
        xs: jnp.ndarray,
        ys: jnp.ndarray,
        *,
        area: float | None,
        waist_xy: tuple[float, float],
        initial_z: float,
        radius_of_curvature_xy: tuple[float, float],
        dx: float,
        dy: float,
        amplitude: float | Sequence[float] = 1.0,
        phase: float | Sequence[float] = 0.0,
    ) -> GaussianBeam2D:
        xs = jnp.asarray(xs, dtype=self.dtype)
        ys = jnp.asarray(ys, dtype=self.dtype)
        num = int(xs.shape[0])
        if num == 0:
            coords = jnp.zeros((0, 2), dtype=self.dtype)
            return GaussianBeam2D(
                rays=tuple(),
                coords=coords,
                waist=waist_xy,
                radius_of_curvature=radius_of_curvature_xy,
                voltage=self.voltage,
                wavelength=self.wavelength,
                k=self.k,
            )

        prefactors = self._prefactor(num, area, waist_xy[0])
        amp_array = (
            jnp.full((num,), float(amplitude), dtype=self.dtype)
            if np.isscalar(amplitude)
            else jnp.asarray(amplitude, dtype=self.dtype)
        )
        if amp_array.shape != (num,):
            raise ValueError("amplitude must broadcast to the number of rays.")
        phase_array = (
            jnp.full((num,), float(phase), dtype=self.dtype)
            if np.isscalar(phase)
            else jnp.asarray(phase, dtype=self.dtype)
        )
        if phase_array.shape != (num,):
            raise ValueError("phase must broadcast to the number of rays.")

        total_amp = amp_array * jnp.abs(prefactors)
        total_phase = phase_array + jnp.angle(prefactors)

        xs_np = np.asarray(xs, dtype=float)
        ys_np = np.asarray(ys, dtype=float)
        amp_np = np.asarray(total_amp, dtype=float)
        phase_np = np.asarray(total_phase, dtype=float)

        rcx, rcy = radius_of_curvature_xy
        wx, wy = waist_xy
        rays = tuple(
            GaussianRay2D.make_gaussian(
                x=float(x),
                y=float(y),
                dx=float(dx),
                dy=float(dy),
                InitAmp=float(amp),
                InitPhase=float(ph),
                waist_x=float(wx),
                waist_y=float(wy),
                RadiusOfCurvature_x=float(rcx),
                RadiusOfCurvature_y=float(rcy),
                k=self.k,
                z=float(initial_z),
            )
            for x, y, amp, ph in zip(xs_np, ys_np, amp_np, phase_np)
        )
        coords = jnp.asarray(np.column_stack((xs_np, ys_np)), dtype=self.dtype)
        return GaussianBeam2D(
            rays=rays,
            coords=coords,
            waist=waist_xy,
            radius_of_curvature=radius_of_curvature_xy,
            voltage=self.voltage,
            wavelength=self.wavelength,
            k=self.k,
        )

    def round_aperture(
        self,
        *,
        aperture_radius: float = 50e-9,
        num_rays: int = 1000,
        sampler: Optional[Callable[..., tuple[np.ndarray, np.ndarray]]] = None,
        sampler_kwargs: Optional[dict[str, Any]] = None,
        initial_z: float | None = None,
        waist_xy: tuple[float, float] | float | None = None,
        radius_of_curvature_xy: tuple[float, float] | float | None = None,
        dx: float = 0.0,
        dy: float = 0.0,
        amplitude: float | Sequence[float] = 1.0,
        phase: float | Sequence[float] = 0.0,
    ) -> GaussianBeam2D:
        sampler = sampler or self.sampler
        combined_kwargs = dict(self.sampler_kwargs)
        if sampler_kwargs:
            combined_kwargs.update(sampler_kwargs)
        area = np.pi * aperture_radius**2 if aperture_radius > 0.0 else 0.0
        base_waist = self._resolve_waist_radius(num_rays, area)
        waist_pair = self._as_pair(waist_xy, default=(base_waist, base_waist))
        rc_pair = self._as_pair(radius_of_curvature_xy, default=(np.inf, np.inf))
        if sampler is uniform_disk:
            combined_kwargs.setdefault("waist_radius", waist_pair[0])
            if self.overlap_factor is not None:
                combined_kwargs.setdefault("overlap_factor", self.overlap_factor)
        xs, ys = sampler(num_rays, radius=aperture_radius, **combined_kwargs)
        initial_z = self._resolve_initial_z(initial_z)
        return self._build(
            xs,
            ys,
            area=area,
            waist_xy=waist_pair,
            initial_z=initial_z,
            radius_of_curvature_xy=rc_pair,
            dx=dx,
            dy=dy,
            amplitude=amplitude,
            phase=phase,
        )

    def elliptical_aperture(
        self,
        *,
        semi_axis_x: float = 50e-9,
        semi_axis_y: float = 30e-9,
        rotation_radians: float = 0.0,
        num_rays: int = 1000,
        sampler: Optional[Callable[..., tuple[np.ndarray, np.ndarray]]] = None,
        sampler_kwargs: Optional[dict[str, Any]] = None,
        initial_z: float | None = None,
        waist_xy: tuple[float, float] | float | None = None,
        radius_of_curvature_xy: tuple[float, float] | float | None = None,
        dx: float = 0.0,
        dy: float = 0.0,
        amplitude: float | Sequence[float] = 1.0,
        phase: float | Sequence[float] = 0.0,
    ) -> GaussianBeam2D:
        if semi_axis_x <= 0.0 or semi_axis_y <= 0.0:
            raise ValueError("Ellipse semi-axes must be positive.")
        sampler = sampler or self.sampler
        combined_kwargs = dict(self.sampler_kwargs)
        if sampler_kwargs:
            combined_kwargs.update(sampler_kwargs)
        area = np.pi * semi_axis_x * semi_axis_y
        base_waist = self._resolve_waist_radius(num_rays, area)
        waist_pair = self._as_pair(waist_xy, default=(base_waist, base_waist))
        rc_pair = self._as_pair(radius_of_curvature_xy, default=(np.inf, np.inf))
        use_uniform_sampler = sampler is uniform_disk
        if use_uniform_sampler:
            effective_radius = np.sqrt(semi_axis_x * semi_axis_y)
            combined_kwargs.setdefault("waist_radius", np.sqrt(waist_pair[0] * waist_pair[1]))
            if self.overlap_factor is not None:
                combined_kwargs.setdefault("overlap_factor", self.overlap_factor)
            ux, uy = sampler(num_rays, radius=effective_radius, **combined_kwargs)
        else:
            ux, uy = sampler(num_rays, radius=1.0, **combined_kwargs)
        ux = jnp.asarray(ux, dtype=self.dtype)
        uy = jnp.asarray(uy, dtype=self.dtype)
        U = jnp.stack([ux, uy], axis=1)
        c, s = jnp.cos(rotation_radians), jnp.sin(rotation_radians)
        if use_uniform_sampler:
            effective_radius = np.sqrt(semi_axis_x * semi_axis_y)
            scale_x = semi_axis_x / effective_radius
            scale_y = semi_axis_y / effective_radius
        else:
            scale_x = semi_axis_x
            scale_y = semi_axis_y
        M = jnp.array(
            [
                [c * scale_x, -s * scale_y],
                [s * scale_x,  c * scale_y],
            ],
            dtype=self.dtype,
        )
        XY = U @ M.T
        xs = XY[:, 0]
        ys = XY[:, 1]
        initial_z = self._resolve_initial_z(initial_z)
        return self._build(
            xs,
            ys,
            area=area,
            waist_xy=waist_pair,
            initial_z=initial_z,
            radius_of_curvature_xy=rc_pair,
            dx=dx,
            dy=dy,
            amplitude=amplitude,
            phase=phase,
        )

    def square_aperture(
        self,
        *,
        side_length: float = 100e-9,
        side_length_y: float | None = None,
        samples_per_side: int | None = None,
        num_rays: int | None = None,
        sampling: str | None = None,
        initial_z: float | None = None,
        waist_xy: tuple[float, float] | float | None = None,
        radius_of_curvature_xy: tuple[float, float] | float | None = None,
        dx: float = 0.0,
        dy: float = 0.0,
        amplitude: float | Sequence[float] = 1.0,
        phase: float | Sequence[float] = 0.0,
    ) -> GaussianBeam2D:
        width = side_length
        height = side_length if side_length_y is None else side_length_y
        if width <= 0.0 or height <= 0.0:
            raise ValueError("side lengths must be positive.")
        if samples_per_side is None and num_rays is None:
            raise ValueError("Provide either samples_per_side or num_rays.")
        area = width * height
        if num_rays is not None and num_rays <= 0:
            raise ValueError("num_rays must be positive when provided.")
        if samples_per_side is not None and samples_per_side <= 0:
            raise ValueError("samples_per_side must be positive when provided.")

        mode = sampling or self.sampling_mode or "fibonacci"
        if mode == "uniform":
            if samples_per_side is not None and num_rays is None:
                coords_x = np.linspace(-0.5 * width, 0.5 * width, samples_per_side, dtype=float)
                coords_y = np.linspace(-0.5 * height, 0.5 * height, samples_per_side, dtype=float)
                X, Y = np.meshgrid(coords_x, coords_y, indexing="xy")
                coords = np.stack((X.reshape(-1), Y.reshape(-1)), axis=1)
                xs = jnp.asarray(coords[:, 0], dtype=self.dtype)
                ys = jnp.asarray(coords[:, 1], dtype=self.dtype)
            else:
                target = num_rays if num_rays is not None else samples_per_side**2
                coords = self._uniform_rectangular_grid(target, width, height)
                xs = jnp.asarray(coords[:, 0], dtype=self.dtype)
                ys = jnp.asarray(coords[:, 1], dtype=self.dtype)
        else:
            samples = samples_per_side
            if samples is None:
                samples = int(np.ceil(np.sqrt(num_rays)))
                samples = max(samples, 1)
            coords_x = jnp.linspace(-0.5 * width, 0.5 * width, samples)
            coords_y = jnp.linspace(-0.5 * height, 0.5 * height, samples)
            X, Y = jnp.meshgrid(coords_x, coords_y, indexing="xy")
            flat_x = X.reshape(-1)
            flat_y = Y.reshape(-1)
            total_points = int(flat_x.shape[0])
            use_count = total_points if num_rays is None else min(num_rays, total_points)
            xs = flat_x[:use_count]
            ys = flat_y[:use_count]

        use_count = int(xs.shape[0])
        waist_value = self._resolve_waist_radius(use_count, area)
        waist_pair = self._as_pair(waist_xy, default=(waist_value, waist_value))
        rc_pair = self._as_pair(radius_of_curvature_xy, default=(np.inf, np.inf))
        initial_z = self._resolve_initial_z(initial_z)
        return self._build(
            xs,
            ys,
            area=area,
            waist_xy=waist_pair,
            initial_z=initial_z,
            radius_of_curvature_xy=rc_pair,
            dx=dx,
            dy=dy,
            amplitude=amplitude,
            phase=phase,
        )

    @staticmethod
    def _uniform_rectangular_grid(num_points: int, width: float, height: float) -> np.ndarray:
        if num_points <= 0:
            raise ValueError("num_points must be positive for uniform rectangular sampling.")
        if width <= 0.0 or height <= 0.0:
            raise ValueError("Rectangle dimensions must be positive.")
        if num_points == 1:
            return np.array([[0.0, 0.0]], dtype=float)

        aspect = width / height
        aspect = aspect if aspect > 0.0 else 1.0
        ny = max(1, int(np.round(np.sqrt(num_points / aspect))))
        nx = max(1, int(np.ceil(num_points / ny)))
        half_x = 0.5 * width
        half_y = 0.5 * height
        xs = np.linspace(-half_x, half_x, nx, dtype=float)
        ys = np.linspace(-half_y, half_y, ny, dtype=float)
        coords = []
        count = 0
        for y in ys:
            if count >= num_points:
                break
            remaining = num_points - count
            if remaining >= nx:
                row_xs = xs
            else:
                drop = nx - remaining
                drop_left = drop // 2
                drop_right = drop - drop_left
                row_xs = xs[drop_left:nx - drop_right]
            for x in row_xs:
                coords.append((x, y))
                count += 1
                if count == num_points:
                    break
        return np.asarray(coords, dtype=float)


def beam_to_gaussian_ray_beta(beam: GaussianBeam2D):
    """
    Convert a :class:`GaussianBeam2D` into the legacy :class:`GaussianRayBeta`
    batch for interoperability with the older propagation utilities.
    """
    from .gaussian import GaussianRayBeta, TaylorExpofAction  # lazy import to avoid cycles

    num = len(beam)
    if num == 0:
        zeros = jnp.zeros((0,), dtype=jnp.float64)
        ones = jnp.zeros((0,), dtype=jnp.float64)
        empty_complex = jnp.zeros((0,), dtype=jnp.complex128)
        S = TaylorExpofAction(
            const=empty_complex,
            lin=jnp.zeros((0, 2), dtype=jnp.complex128),
            quad=jnp.zeros((0, 2, 2), dtype=jnp.complex128),
        )
        return GaussianRayBeta(
            x=zeros,
            y=zeros,
            dx=zeros,
            dy=zeros,
            z=zeros,
            pathlength=zeros,
            _one=ones,
            C=empty_complex,
            S=S,
            voltage=zeros,
        )

    dtype = jnp.float64
    complex_dtype = jnp.complex128
    r0 = jnp.stack([ray.r0 for ray in beam.rays], axis=0).astype(dtype)
    x = r0[:, 0]
    y = r0[:, 1]
    slopes = jnp.stack([ray.ray_slope() for ray in beam.rays], axis=0)
    dx = slopes[:, 0]
    dy = slopes[:, 1]
    z = jnp.stack([jnp.asarray(ray.z, dtype=dtype) for ray in beam.rays], axis=0)
    zeros = jnp.zeros(num, dtype=dtype)
    ones = jnp.ones(num, dtype=dtype)
    C = jnp.stack([ray.C for ray in beam.rays], axis=0).astype(complex_dtype)
    lin = jnp.stack([ray.S1 for ray in beam.rays], axis=0).astype(complex_dtype)
    quad = jnp.stack([ray.S2 for ray in beam.rays], axis=0).astype(complex_dtype)
    const = jnp.zeros(num, dtype=complex_dtype)
    S = TaylorExpofAction(const=const, lin=lin, quad=quad)
    voltage = jnp.full((num,), beam.voltage, dtype=dtype)

    return GaussianRayBeta(
        x=x,
        y=y,
        dx=dx,
        dy=dy,
        z=z,
        pathlength=zeros,
        _one=ones,
        C=C,
        S=S,
        voltage=voltage,
    )
