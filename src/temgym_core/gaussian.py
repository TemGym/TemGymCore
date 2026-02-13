import dataclasses
from typing import Any, Callable, NamedTuple, Sequence, Tuple

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from ase import units
from jax import lax

from .constants import energy2wavelength, relativistic_mass_correction
from .ray import Ray

LENGTH = {
    "m": 1.0,
    "A": 1e-10,
    "angstrom": 1e-10,
}


@jdc.pytree_dataclass(kw_only=True)
class GaussianBeam(Ray):
    amplitude: jnp.ndarray | complex
    Q_inv: jnp.ndarray | complex
    voltage: jnp.ndarray | float | None = None
    wavelength_unit: jdc.Static[str] = "m"

    @property
    def ray_family(self) -> str:
        return "gaussian"

    def derive(
        self,
        x: float | jnp.ndarray | None = None,
        y: float | jnp.ndarray | None = None,
        dx: float | jnp.ndarray | None = None,
        dy: float | jnp.ndarray | None = None,
        z: float | jnp.ndarray | None = None,
        amplitude: jnp.ndarray | complex | None = None,
        pathlength: float | jnp.ndarray | None = None,
        Q_inv: jnp.ndarray | complex | None = None,
        voltage: float | jnp.ndarray | None = None,
        _one: float | jnp.ndarray | None = None,
        wavelength_unit: str | None = None,
    ) -> "GaussianBeam":
        return GaussianBeam(
            x=self.x if x is None else x,
            y=self.y if y is None else y,
            dx=self.dx if dx is None else dx,
            dy=self.dy if dy is None else dy,
            z=self.z if z is None else z,
            amplitude=self.amplitude if amplitude is None else amplitude,
            pathlength=self.pathlength if pathlength is None else pathlength,
            Q_inv=self.Q_inv if Q_inv is None else Q_inv,
            voltage=self.voltage if voltage is None else voltage,
            _one=self._one if _one is None else _one,
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
) -> GaussianBeam:
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
    amplitude = jnp.asarray(amp) * jnp.exp(1j * phase)
    pathlength = jnp.zeros_like(phase)

    ray = GaussianBeam(
        x=x,
        y=y,
        dx=dx,
        dy=dy,
        z=z,
        amplitude=amplitude,
        Q_inv=Q_inv,
        voltage=voltage,
        pathlength=pathlength,
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


def scalar_grad_hess_complex(
    fn: Callable[..., complex],
    x: jnp.ndarray,
    *args: Any,
    diff_argnums: int | Sequence[int] = 0,
) -> Tuple[complex, jnp.ndarray, jnp.ndarray]:
    dS0, grad, hess = taylor_expand(fn, x, *args, diff_argnums=diff_argnums)
    return dS0, grad, hess


def taylor_expand(
    fn: Callable[..., complex],
    x: jnp.ndarray,
    *args: Any,
    diff_argnums: int | Sequence[int] = 0,
) -> Tuple[complex, jnp.ndarray, jnp.ndarray]:
    full_args = (x, *args)

    def re_fn(*fn_args):
        return jnp.real(fn(*fn_args))

    def im_fn(*fn_args):
        return jnp.imag(fn(*fn_args))

    dS0 = fn(*full_args)

    grad_re = jax.grad(re_fn, argnums=diff_argnums)(*full_args)
    grad_im = jax.grad(im_fn, argnums=diff_argnums)(*full_args)

    hess_re = jax.hessian(re_fn, argnums=diff_argnums)(*full_args)
    hess_im = jax.hessian(im_fn, argnums=diff_argnums)(*full_args)

    grad = grad_re + 1j * grad_im
    hess = hess_re + 1j * hess_im
    return dS0, grad, hess


def apply_action_delta(
    ray: GaussianBeam,
    dS0: complex,
    dS1: jnp.ndarray,
    dS2: jnp.ndarray,
    tiny: float = 1e-30,
):
    k = ray.k
    r0 = ray.r_xy

    S0_old = ray.pathlength
    d_xy_old = ray.d_xy
    Q_old = ray.Q_inv

    S0_prime = S0_old + dS0
    S1_prime = d_xy_old + dS1
    Q_prime = Q_old + dS2

    ImQ = jnp.imag(Q_prime)
    ImS1 = jnp.imag(S1_prime)

    def solve_dx(args):
        ImQ_, ImS1_ = args
        return jnp.linalg.solve(ImQ_, -ImS1_)

    def zero_dx(args):
        _, ImS1_ = args
        return jnp.zeros_like(ImS1_)

    det_ImQ = jnp.linalg.det(ImQ)
    dx = lax.cond(
        jnp.abs(det_ImQ) < tiny,
        zero_dx,
        solve_dx,
        (ImQ, ImS1),
    )

    r_xy_new = r0 + dx
    S0_new = S0_prime + S1_prime @ dx + 0.5 * (dx @ (Q_prime @ dx))
    S1_new = S1_prime + Q_prime @ dx
    Q_new = Q_prime

    d_xy_new = jnp.real(S1_new)
    S0_new_re = jnp.real(S0_new)
    S0_new_im = jnp.imag(S0_new)

    pathlength_new = S0_new_re
    amp_factor = jnp.exp(-k * S0_new_im)
    amplitude_new = ray.amplitude * amp_factor

    return r_xy_new, d_xy_new, amplitude_new, pathlength_new, Q_new


class Propagator(NamedTuple):
    distance: float
    propagator: "BaseGaussianPropagator"

    def __call__(self, ray: GaussianBeam) -> GaussianBeam:
        return self.propagator(ray, self.distance)


class BaseGaussianPropagator:
    def __call__(self, ray: GaussianBeam, distance: float) -> GaussianBeam:
        raise NotImplementedError

    def with_distance(self, distance: float) -> Propagator:
        return Propagator(distance, self)


class FreeSpacePropagator(BaseGaussianPropagator):
    def __call__(self, ray: GaussianBeam, distance: float) -> GaussianBeam:
        theta = ray.d_xy
        Q = ray.Q_inv

        identity = jnp.eye(2, dtype=jnp.complex128)
        A = identity + distance * Q
        invA = jnp.linalg.solve(A.T, identity).T
        detA = jnp.linalg.det(A)

        Q_new = Q @ invA
        r_xy_new = ray.r_xy + distance * theta

        theta_sq = jnp.dot(theta, theta)
        pathlength_new = ray.pathlength + distance + 0.5 * distance * theta_sq
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


_REMOVED_SYMBOL_TO_PATH = {
    "Component": "temgym_core.components.Component",
    "Lens": "temgym_core.components.Lens",
    "KrivanekLens": "temgym_core.components.KrivanekLens",
    "SeidelLens": "temgym_core.components.SeidelLens",
    "DistortedLens": "temgym_core.components.DistortedLens",
    "ElectromagneticLens": "temgym_core.components.ElectromagneticLens",
    "ABCDTransfer": "temgym_core.components.ABCDTransfer",
    "SigmoidAperture": "temgym_core.components.SigmoidAperture",
    "Biprism": "temgym_core.components.PhaseBiprism",
    "PhaseBiprism": "temgym_core.components.PhaseBiprism",
    "DeflectionBiprism": "temgym_core.components.DeflectionBiprism",
    "ConstantPhaseShift": "temgym_core.components.ConstantPhaseShift",
    "LinearPhaseShift": "temgym_core.components.LinearPhaseShift",
    "QuadraticPhaseShift": "temgym_core.components.QuadraticPhaseShift",
    "ConstantAmplitudeShift": "temgym_core.components.ConstantAmplitudeShift",
    "LinearAmplitudeShift": "temgym_core.components.LinearAmplitudeShift",
    "QuadraticAmplitudeShift": "temgym_core.components.QuadraticAmplitudeShift",
    "MagneticPhaseSample": "temgym_core.components.MagneticPhaseSample",
    "RandomPhaseSample": "temgym_core.components.RandomPhaseSample",
    "InterpolatedSample2D": "temgym_core.components.InterpolatedSample2D",
    "InterpolatedFields3D": "temgym_core.components.InterpolatedFields3D",
    "AtomicPotential": "temgym_core.components.AtomicPotential",
    "FourierTransform": "temgym_core.components.FourierTransform",
    "Detector": "temgym_core.components.Detector",
    "run_iter": "temgym_core.run.run_iter",
    "run_to_end": "temgym_core.run.run_to_end",
    "run_iter_vmapped": "temgym_core.run.run_iter_vmapped",
    "run_to_end_vmapped": "temgym_core.run.run_to_end_vmapped",
    "circular_input_wave": "temgym_core.source.circular_input_wave",
    "square_input_wave": "temgym_core.source.square_input_wave",
    "rectangular_input_wave": "temgym_core.source.rectangular_input_wave",
    "sample_input_wave": "temgym_core.source.sample_input_wave",
}


def __getattr__(name: str):
    if name in _REMOVED_SYMBOL_TO_PATH:
        target = _REMOVED_SYMBOL_TO_PATH[name]
        raise AttributeError(
            f"`temgym_core.gaussian.{name}` has moved. Import `{target}` instead."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GaussianBeam",
    "make_gaussian",
    "scalar_grad_hess_complex",
    "taylor_expand",
    "apply_action_delta",
    "Propagator",
    "BaseGaussianPropagator",
    "FreeSpacePropagator",
]
