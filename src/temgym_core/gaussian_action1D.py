import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax_dataclasses as jdc
from typing import Any, Callable, Generator, NamedTuple, Sequence, Tuple


@jdc.pytree_dataclass
class GaussianRay1D:
    C: complex = 1+0j          # global amplitude+phase
    S1: complex = 0+0j         # linear action coeff
    S2: complex = 1j           # quadratic action coeff
    x0: float = 0.0            # expansion origin (intensity centre)
    k: float = 1.0             # wavenumber
    z: float = 0.0             # axial coordinate

    def recenter(self, C, S1, S2, x0, k):
        ray_pos = self._ray_position_from_params(S1, S2, x0)
        delta = ray_pos - x0
        phase = S1 * delta + 0.5 * S2 * delta * delta

        Cn = C * jnp.exp(1j * k * phase)
        S1n = S1 + S2 * delta
        return Cn, S1n, S2, ray_pos

    def _with_params(self, C=None, S1=None, S2=None, x0=None, k=None, z=None):
        C = self.C if C is None else C
        S1 = self.S1 if S1 is None else S1
        S2 = self.S2 if S2 is None else S2
        x0 = self.x0 if x0 is None else x0
        k = self.k if k is None else k
        z = self.z if z is None else z
        C, S1, S2, x0 = self.recenter(C, S1, S2, x0, k)
        return GaussianRay1D(C=C, S1=S1, S2=S2, x0=x0, k=k, z=z)

    def action(self, x):
        xi = x - self.x0
        return self.S1 * xi + 0.5 * self.S2 * xi * xi

    def field(self, x):
        return self.C * jnp.exp(1j * self.k * self.action(x))

    def amplitude(self, x):
        return jnp.abs(self.field(x))

    def intensity(self, x):
        psi = self.field(x)
        return jnp.abs(psi) ** 2

    def phase(self, x):
        return jnp.angle(self.field(x))

    @staticmethod
    def _ray_position_from_params(S1, S2, x0):
        return x0 - S1.imag / (S2.imag)

    @staticmethod
    def _ray_slope_from_params(S1, S2, x0):
        xi = - S1.imag / (S2.imag)
        p = S1 + S2 * xi
        return p.real

    def ray_position(self) -> float:
        # A special utility function which is able to
        # compute the maximum intensity of the gaussian, which
        # can sometimes be different from point about which the gaussian is expanded.
        # This happens in particular when there is a linear amplitude term,
        # which is applied by an aperture and can shift the intensity centre.
        # In general we keep x0 and the intensity centre aligned, so S1.imag is usually 0.0.
        # and x0 matches the maximum intensity position of the gaussian.
        return self._ray_position_from_params(self.S1, self.S2, self.x0)

    def ray_slope(self) -> float:
        # A special utility function which is able to
        # compute the slope of the gaussian, which
        # can sometimes be different from the point about which the gaussian is expanded.
        # This happens in particular when there is a linear amplitude term,
        # so just after an aperture the gaussian can be shifted and then the slope will need
        # to be computed at the intensity centre.
        # In general the ray slope will be computed at the intensity centre,
        # so S1.imag is usually 0.0.
        # xi will be 0.0 in that case.
        return self._ray_slope_from_params(self.S1, self.S2, self.x0) / self.k

    @staticmethod
    def make_gaussian(x: float,
                      dx: float,
                      InitAmp: float = 1.0,
                      InitPhase: float = 0.0,
                      BeamWaist: float = 1.0,
                      RadiusOfCurvature: float = jnp.inf,
                      k: float = 1.0,
                      z: float = 0.0) -> "GaussianRay1D":

        S1_real = k * dx
        S2_real = 0.0 if jnp.isinf(RadiusOfCurvature) else 1.0 / RadiusOfCurvature
        S2_imag = 1.0 / (2.0 * k * (BeamWaist**2))

        S1 = jnp.asarray(S1_real, dtype=jnp.complex128)
        S2 = jnp.asarray(S2_real + 1j * S2_imag, dtype=jnp.complex128)
        C = InitAmp * jnp.exp(1j * InitPhase)
        xc = jnp.asarray(x, dtype=jnp.float64)

        return GaussianRay1D(C=C, S1=S1, S2=S2, x0=xc, k=k, z=float(z))


@jdc.pytree_dataclass
class ABCDPropagator:
    A: complex = 1+0j
    B: complex = 0+0j
    C: complex = 0+0j
    D: complex = 1+0j
    L: float = 0.0         # Propagation Distance for updating ray z and global phase
    eps: float = 1e-12

    def __call__(self, ray: GaussianRay1D) -> GaussianRay1D:
        """Apply this ABCD propagation to a GaussianRay1D (lax.cond branches for differentiability)."""
        # quick Python-level singular check (keeps crashes explicit)
        A_py = complex(self.A); B_py = complex(self.B)
        if abs(A_py) < self.eps and abs(B_py) < self.eps:
            raise ValueError("ABCD matrix is singular: A and B both zero.")

        # make JAX-compatible complex scalars
        Aj = jnp.asarray(complex(self.A), dtype=jnp.complex128)
        Bj = jnp.asarray(complex(self.B), dtype=jnp.complex128)
        Cj = jnp.asarray(complex(self.C), dtype=jnp.complex128)
        Dj = jnp.asarray(complex(self.D), dtype=jnp.complex128)
        L = jnp.asarray(self.L, dtype=jnp.float64)

        # predicate: is B ~ 0 (imaging / magnification-like transform)?
        pred = jnp.less(jnp.abs(Bj), self.eps)

        # imaging case: B ~ 0
        def imaging_fn(r):
            den = Aj
            x0p = jnp.real(den * r.x0)
            S2p = (Cj + Dj * r.S2) / den
            S1p = (r.S1 + Cj * x0p) / den
            dS0 = 0.5 * Cj * r.x0 * x0p
            Cp = r.C * jnp.exp(-0.5 * jnp.log(den)) * jnp.exp(1j * r.k * (dS0 + L))
            return r._with_params(C=Cp, S1=S1p, S2=S2p, x0=x0p, z=r.z + float(self.L))

        # general case: B != 0
        def general_fn(r):
            den = Aj + Bj * r.S2
            S2p = (Cj + Dj * r.S2) / den
            S1p = r.S1 / den
            dS0 = -0.5 * Bj * (r.S1 * r.S1) / den
            Cp = r.C * jnp.exp(-0.5 * jnp.log(den)) * jnp.exp(1j * r.k * (dS0 + L))
            return r._with_params(C=Cp, S1=S1p, S2=S2p, x0=r.x0, z=r.z + float(self.L))

        return jax.lax.cond(pred, imaging_fn, general_fn, ray)

    @staticmethod
    def free_space(z: float):
        # propagation through distance z in free space
        return ABCDPropagator(1.0, z, 0.0, 1.0, z)

    @staticmethod
    def thin_lens(f: float):
        # thin lens of focal length f
        return ABCDPropagator(1.0, 0.0, -1.0 / f, 1.0, 0.0)

    @staticmethod
    def fourier_transform(f: float):
        # system that performs (scaled) Fourier transform: A=0, B=f, C=-1/f, D=0
        return ABCDPropagator(0.0, f, -1.0 / f, 0.0, 2*f)

    @staticmethod
    def perfect_imaging(magnification: float, L: float):
        # perfect imaging with magnification M: A=M, B=0, C=0, D=1/M (simple scaling image)
        M = float(magnification)
        return ABCDPropagator(M, 0.0, 0.0, 1.0 / M, L)

    @staticmethod
    def lens_plus_propagation(f: float, z_before: float = 0.0, z_after: float = 0.0):
        # convenience: propagate z_before -> thin lens f -> propagate z_after
        # Combined ABCD = P_after * F * P_before
        # P(z) = [[1, z],[0,1]] ; F = [[1,0],[-1/f,1]]
        A1, B1, C1, D1 = 1.0, z_before, 0.0, 1.0
        A2, B2, C2, D2 = 1.0, 0.0, -1.0 / f, 1.0
        A3, B3, C3, D3 = 1.0, z_after, 0.0, 1.0

        # multiply matrices (A3*A2*A1 etc.)
        def mul(Aa, Bb, Cc, Dd, Ae, Be, Ce, De):
            return (Aa * Ae + Bb * Ce,
                    Aa * Be + Bb * De,
                    Cc * Ae + Dd * Ce,
                    Cc * Be + Dd * De)

        A12, B12, C12, D12 = mul(A2, B2, C2, D2, A1, B1, C1, D1)
        A, B, C, D = mul(A3, B3, C3, D3, A12, B12, C12, D12)
        return ABCDPropagator(A, B, C, D, L=z_before + z_after)


def apply_action_delta(ray,
                       dS0: complex = 0.0 + 0.0j,
                       dS1: complex = 0.0 + 0.0j,
                       dS2: complex = 0.0 + 0.0j,
                       *,
                       z: float | None = None):
    """Apply a local quadratic complex-action update ΔS to a 1D GaussianRay."""
    dS0 = jnp.asarray(dS0, dtype=jnp.complex128)
    dS1 = jnp.asarray(dS1, dtype=jnp.complex128)
    dS2 = jnp.asarray(dS2, dtype=jnp.complex128)

    phi0 = jnp.real(dS0)
    phi1 = jnp.real(dS1)
    phi2 = jnp.real(dS2)

    ell0 = -ray.k * jnp.imag(dS0)
    ell1 = -ray.k * jnp.imag(dS1)
    ell2 = -ray.k * jnp.imag(dS2)

    Cn = ray.C * jnp.exp(ell0 + 1j * ray.k * phi0)
    S1n = ray.S1 + phi1 - 1j * ell1 / ray.k
    S2n = ray.S2 + phi2 - 1j * ell2 / ray.k

    return ray._with_params(C=Cn, S1=S1n, S2=S2n, z=z)


def apply_phase_delta(ray, phi0=0.0, phi1=0.0, phi2=0.0):
    """Apply a purely real phase ΔS to the ray's complex action."""
    dS0 = jnp.asarray(phi0, dtype=jnp.complex128)
    dS1 = jnp.asarray(phi1, dtype=jnp.complex128)
    dS2 = jnp.asarray(phi2, dtype=jnp.complex128)
    return apply_action_delta(ray, dS0, dS1, dS2)


def apply_log_transmission_delta(ray, L0=0.0, L1=0.0, L2=0.0):
    """Apply a purely imaginary log-transmission ΔS (attenuation profile)."""
    scale = -1j / ray.k
    dS0 = scale * jnp.asarray(L0, dtype=jnp.complex128)
    dS1 = scale * jnp.asarray(L1, dtype=jnp.complex128)
    dS2 = scale * jnp.asarray(L2, dtype=jnp.complex128)
    return apply_action_delta(ray, dS0, dS1, dS2)


def _scalar_complex_vjp(fn, x):
    """Utility: derivative of a scalar complex function with respect to a real scalar."""
    y, pullback = jax.vjp(fn, x)
    seed = jnp.ones_like(y, dtype=y.dtype)
    (deriv,) = pullback(seed)
    return deriv


def grad_complex_action(component, x, k):
    """First derivative ∂ΔS/∂x at position x."""
    x = jnp.asarray(x, dtype=jnp.float64)
    return _scalar_complex_vjp(lambda x_val: component.complex_action(x_val, k), x)


def hess_complex_action(component, x, k):
    """Second derivative ∂²ΔS/∂x² at position x."""
    x = jnp.asarray(x, dtype=jnp.float64)
    grad_fn = lambda x_val: grad_complex_action(component, x_val, k)
    return _scalar_complex_vjp(grad_fn, x)


@jdc.pytree_dataclass
class Component1D:
    """
    Base interface for 1D optical/electron-optical elements acting on GaussianRay1D.

    Subclasses specify the on-axis phase shift ``phase_shift`` and the log
    transmission ``log_transmission``. The default implementation combines
    these into a complex action ΔS and applies its quadratic Taylor expansion
    to the ray via ``apply_action_delta``.
    """
    z: float = 0.0

    def phase_shift(self, x):
        return 0.0

    def log_transmission(self, x):
        return 0.0

    def complex_action(self, x, k):
        phi = self.phase_shift(x)
        L = self.log_transmission(x)
        return phi - 1j * (L / k)

    def expansion(self, x, k):
        dS0 = self.complex_action(x, k)
        dS1 = grad_complex_action(self, x, k)
        dS2 = hess_complex_action(self, x, k)
        return dS0, dS1, dS2

    def __call__(self, ray: GaussianRay1D) -> GaussianRay1D:
        x_ref = jnp.asarray(ray.ray_position(), dtype=jnp.float64)
        k = ray.k
        dS0, dS1, dS2 = self.expansion(x_ref, k)
        updated = apply_action_delta(ray, dS0, dS1, dS2)
        return updated._with_params(z=float(self.z))


@jdc.pytree_dataclass(kw_only=True)
class ThinLens1D(Component1D):
    """Quadratic phase element representing a thin lens in 1D."""
    focal_length: float
    x_center: float = 0.0

    def phase_shift(self, x):
        xi = x - self.x_center
        return -0.5 * (xi * xi) / self.focal_length


@jdc.pytree_dataclass(kw_only=True)
class AberratedLens1D(Component1D):
    """Cubic/quartic-extended thin lens phase in 1D."""
    focal_length: float
    cubic_coeff: float = 0.0
    quartic_coeff: float = 0.0
    x_center: float = 0.0

    def phase_shift(self, x):
        xi = x - self.x_center
        phase = -0.5 * (xi * xi) / self.focal_length
        phase = phase + self.cubic_coeff * (xi ** 3) + self.quartic_coeff * (xi ** 4)
        return phase


@jdc.pytree_dataclass(kw_only=True)
class SigmoidAperture1D(Component1D):
    """Smooth half-plane aperture using a logistic transmission profile."""
    edge_position: float = 0.0
    width: float = 0.5
    t_low: float = 0.0
    t_high: float = 1.0
    eps: float = 1e-12

    def phase_shift(self, x):
        return 0.0

    def log_transmission(self, x):
        width = jnp.maximum(jnp.abs(self.width), self.eps)
        z = (x - self.edge_position) / width
        s = jnn.sigmoid(z)
        t = self.t_low + (self.t_high - self.t_low) * s
        t_clamped = jnp.clip(t, self.eps, None)
        return jnp.log(t_clamped)


# Backwards-compatible aliases
apply_local_gaussian = apply_action_delta
apply_local_phase = apply_phase_delta
apply_local_amplitude = apply_log_transmission_delta
TransformT = Callable[[Any], Callable[[Any], tuple[Any, Any]]]


def passthrough_transform(component):
    def inner(ray):
        out = component(ray)
        return out, out
    return inner


class BaseGaussianPropagator1D:
    def propagate(self, ray: GaussianRay1D, distance: float):
        raise NotImplementedError

    def with_distance(self, distance: float):
        return GaussianPropagator1D(distance, self)


class GaussianPropagator1D(NamedTuple):
    distance: float
    propagator: BaseGaussianPropagator1D

    def __call__(self, ray: GaussianRay1D):
        return self.propagator.propagate(ray, self.distance)


class FreeSpaceParaxial1D(BaseGaussianPropagator1D):
    def propagate(self, ray: GaussianRay1D, distance: float):
        propagator = ABCDPropagator.free_space(distance)
        return propagator(ray)


def run_iter(
    ray: GaussianRay1D,
    components: Sequence[Any],
    transform: TransformT = passthrough_transform,
    propagator: BaseGaussianPropagator1D = FreeSpaceParaxial1D(),
) -> Generator[Tuple[Any, Any], Any, None]:
    """
    Iterate a ray (GaussianRay1D or compatible object) through the model, yielding each step's output.
    Free-space (paraxial) propagation is inserted when component.z != current_z.
    """
    current_z = float(getattr(ray, "z", 0.0))

    for component in components:
        if isinstance(component, Component1D):
            distance = float(component.z - current_z)
            propagator_d = propagator.with_distance(distance)
            ray, out = transform(propagator_d)(ray)
            current_z += distance
            yield propagator_d, out
        ray, out = transform(component)(ray)
        if isinstance(component, Component1D):
            current_z = float(component.z)
        yield component, out


def run_to_end(
    ray: GaussianRay1D,
    components: Sequence[Any],
    propagator: BaseGaussianPropagator1D = FreeSpaceParaxial1D(),
) -> GaussianRay1D:
    """
    Propagate a ray through all components and return the final state.
    """
    for _, ray in run_iter(ray, components, propagator=propagator):
        pass
    return ray
