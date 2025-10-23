import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax_dataclasses as jdc
from typing import Any, Callable, Generator, NamedTuple, Optional, Sequence, Tuple


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
