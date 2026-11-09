from __future__ import annotations

import dataclasses
from typing import Any, Callable, NamedTuple, Sequence, Tuple

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jax import lax

from .constants import (
    _COULOMB,
    _JOULE,
    _KG,
    _SECOND,
    ELECTRON_MASS,
    ELEMENTARY_CHARGE,
    PLANCK,
    energy2wavelength,
    relativistic_mass_correction,
)
from .ray import Ray

LENGTH = {
    "m": 1.0,
    "mm": 1e-3,
    "um": 1e-6,
    "micron": 1e-6,
    "nm": 1e-9,
    "A": 1e-10,
    "angstrom": 1e-10,
}


def _as_constructor_vector(name: str, value: Any) -> jnp.ndarray:
    """Normalize a scalar or one-dimensional constructor argument."""
    arr = jnp.asarray(value)
    if arr.ndim == 0:
        return arr[None]
    if arr.ndim == 1:
        return arr
    raise ValueError(f"`{name}` must be scalar or one-dimensional, got shape {arr.shape}.")


def _broadcast_constructor_fields(**fields: Any) -> dict[str, jnp.ndarray]:
    arrays = {name: _as_constructor_vector(name, value) for name, value in fields.items()}
    try:
        shape = jnp.broadcast_shapes(*(arr.shape for arr in arrays.values()))
    except ValueError as exc:
        sizes = ", ".join(f"{name}={arr.shape}" for name, arr in arrays.items())
        raise ValueError(
            "Gaussian constructor fields must be scalar or share one leading "
            f"length; got {sizes}."
        ) from exc
    return {name: jnp.broadcast_to(arr, shape) for name, arr in arrays.items()}


def _validate_positive(name: str, value: jnp.ndarray) -> None:
    """Validate concrete inputs while remaining safe under JAX tracing."""
    try:
        invalid = bool(jnp.any(value <= 0))
    except Exception:
        return
    if invalid:
        raise ValueError(f"`{name}` must be positive.")


def _field_leading_size(name: str, value: Any) -> int:
    if isinstance(value, str) or value is None:
        return 1
    arr = jnp.asarray(value)
    if name == "Q_inv":
        if arr.shape == (2, 2):
            return 1
        if arr.ndim == 3 and arr.shape[-2:] == (2, 2):
            return int(arr.shape[0])
        raise ValueError(f"`Q_inv` must have shape (2, 2) or (N, 2, 2), got {arr.shape}.")
    if arr.ndim == 0:
        return 1
    return int(arr.shape[0])


def _common_leading_size(params: dict[str, Any]) -> int:
    sizes = {
        _field_leading_size(name, value)
        for name, value in params.items()
        if not isinstance(value, str) and value is not None
    }
    non_scalar_sizes = {size for size in sizes if size != 1}
    if len(non_scalar_sizes) > 1:
        details = ", ".join(
            f"{name}={_field_leading_size(name, value)}"
            for name, value in params.items()
            if not isinstance(value, str) and value is not None
        )
        raise ValueError(f"GaussianBeam fields have incompatible leading sizes: {details}.")
    return next(iter(non_scalar_sizes), 1)


def _promote_gaussian_field(name: str, value: Any, size: int) -> Any:
    if isinstance(value, str) or value is None:
        return value
    arr = jnp.asarray(value)
    if name == "Q_inv":
        if arr.shape == (2, 2):
            arr = arr[None, ...]
        elif arr.ndim != 3 or arr.shape[-2:] != (2, 2):
            raise ValueError(f"`Q_inv` must have shape (2, 2) or (N, 2, 2), got {arr.shape}.")
    elif arr.ndim == 0:
        arr = arr[None]

    if arr.shape[0] == size:
        return arr
    if arr.shape[0] == 1:
        return jnp.broadcast_to(arr, (size,) + arr.shape[1:])
    raise ValueError(f"`{name}` has leading size {arr.shape[0]}, expected {size} or 1.")


def _matmul(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    return jnp.einsum("...ij,...jk->...ik", left, right)


def _det_2x2(matrix: jnp.ndarray) -> jnp.ndarray:
    return matrix[..., 0, 0] * matrix[..., 1, 1] - matrix[..., 0, 1] * matrix[..., 1, 0]


def _inverse_2x2(matrix: jnp.ndarray) -> jnp.ndarray:
    a = matrix[..., 0, 0]
    b = matrix[..., 0, 1]
    c = matrix[..., 1, 0]
    d = matrix[..., 1, 1]
    det = a * d - b * c
    adjugate = jnp.stack(
        (jnp.stack((d, -b), axis=-1), jnp.stack((-c, a), axis=-1)),
        axis=-2,
    )
    return adjugate / det[..., None, None]


def _solve_2x2(matrix: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
    inverse = _inverse_2x2(matrix)
    rhs = jnp.asarray(rhs)
    if rhs.ndim == matrix.ndim - 1:
        return jnp.einsum("...ij,...j->...i", inverse, rhs)
    return _matmul(inverse, rhs)


@jdc.pytree_dataclass(kw_only=True)
class GaussianBeam(Ray):
    amplitude: jnp.ndarray | complex
    Q_inv: jnp.ndarray | complex
    voltage: jnp.ndarray | float | None = None
    wavelength_unit: jdc.Static[str] = "m"

    @property
    def ray_family(self) -> str:
        return "gaussian"

    @property
    def r_xy(self) -> jnp.ndarray:
        """Stack positions while preserving an explicit Gaussian batch axis."""
        x, y = jnp.broadcast_arrays(
            jnp.asarray(self.x, dtype=jnp.float64),
            jnp.asarray(self.y, dtype=jnp.float64),
        )
        return jnp.stack((x, y), axis=-1)

    @property
    def d_xy(self) -> jnp.ndarray:
        """Stack slopes while preserving an explicit Gaussian batch axis."""
        dx, dy = jnp.broadcast_arrays(
            jnp.asarray(self.dx, dtype=jnp.float64),
            jnp.asarray(self.dy, dtype=jnp.float64),
        )
        return jnp.stack((dx, dy), axis=-1)

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

    def to_vector(self) -> "GaussianBeam":
        """Return a consistently batched beam, broadcasting scalar fields."""
        params = dataclasses.asdict(self)
        size = _common_leading_size(params)
        params = {
            key: _promote_gaussian_field(key, value, size)
            for key, value in params.items()
        }
        return type(self)(**params)

    def moments(self):
        """Return the 4D position/angle mean and covariance of one Gaussian.

        Positions use the beam's length unit; angles use radians. This is a
        single field's state, not the moments of a coherent sum of beams.
        """
        q = jnp.asarray(self.Q_inv).reshape(2, 2)
        k = jnp.asarray(self.k).reshape(())
        spatial = jnp.linalg.inv(2 * k * q.imag)
        correlation = spatial @ q.real.T
        angular = q.real @ spatial @ q.real.T + jnp.linalg.inv(spatial) / (4 * k**2)
        mean = jnp.concatenate((self.r_xy.reshape(2), self.d_xy.reshape(2)))
        return mean, jnp.block([[spatial, correlation], [correlation.T, angular]])

    @property
    def wavelength(self):
        return energy2wavelength(self.voltage) / LENGTH[self.wavelength_unit]

    @property
    def mass(self):
        return relativistic_mass_correction(self.voltage) * ELECTRON_MASS

    @property
    def sigma(self):
        lam = self.wavelength
        return (
            2 * jnp.pi * self.mass * _KG * ELEMENTARY_CHARGE * _COULOMB * lam
            / (PLANCK * _SECOND * _JOULE) ** 2
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
    if wavelength_unit not in LENGTH:
        raise ValueError(
            f"Unknown wavelength unit {wavelength_unit!r}; expected one of {tuple(LENGTH)}."
        )

    fields = _broadcast_constructor_fields(
        x=x,
        y=y,
        dx=dx,
        dy=dy,
        z=z,
        voltage=voltage,
        amp=amp,
        phase=phase,
        waist_x=waist_x,
        waist_y=waist_y,
        rcurv_x=rcurv_x,
        rcurv_y=rcurv_y,
    )
    _validate_positive("voltage", fields["voltage"])
    _validate_positive("waist_x", fields["waist_x"])
    _validate_positive("waist_y", fields["waist_y"])

    wavelength = energy2wavelength(fields["voltage"]) / LENGTH[wavelength_unit]
    n_rays = fields["x"].shape[0]

    Q_inv_re = jnp.zeros((n_rays, 2, 2), dtype=jnp.float64)
    Q_inv_re = Q_inv_re.at[:, 0, 0].set(1.0 / fields["rcurv_x"])
    Q_inv_re = Q_inv_re.at[:, 1, 1].set(1.0 / fields["rcurv_y"])

    Q_inv_im = jnp.zeros((n_rays, 2, 2), dtype=jnp.float64)
    Q_inv_im = Q_inv_im.at[:, 0, 0].set(
        wavelength / (jnp.pi * fields["waist_x"] ** 2)
    )
    Q_inv_im = Q_inv_im.at[:, 1, 1].set(
        wavelength / (jnp.pi * fields["waist_y"] ** 2)
    )

    ray = GaussianBeam(
        x=fields["x"],
        y=fields["y"],
        dx=fields["dx"],
        dy=fields["dy"],
        z=fields["z"],
        amplitude=jnp.asarray(fields["amp"]) * jnp.exp(1j * fields["phase"]),
        Q_inv=(Q_inv_re + 1j * Q_inv_im).astype(jnp.complex128),
        voltage=fields["voltage"],
        pathlength=jnp.zeros_like(fields["phase"]),
        _one=jnp.ones_like(fields["x"]),
        wavelength_unit=wavelength_unit,
    ).to_vector()

    if n_rays == 1:
        def squeeze0(value):
            if value is None or isinstance(value, str):
                return value
            arr = jnp.asarray(value)
            return jnp.squeeze(arr, axis=0) if arr.ndim > 0 and arr.shape[0] == 1 else arr

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


def _pupil_action_to_spatial(ray, distance, value, gradient, hessian):
    """Integrate a quadratic object/pupil action for one incoming Gaussian.

    The kernel is the Fresnel kernel over ``distance`` multiplied by
    exp(ik * action(object, pupil)). The incident field is specified at the
    pupil, before this action; its reference object field is the inverse
    Fresnel propagation. Derivatives are ordered (object_x, object_y,
    pupil_x, pupil_y), at the unperturbed central-ray endpoints.

    Completing the square in the object integral gives a Schur complement
    and a complex determinant prefactor. Return their spatial action jet
    for apply_action_delta. No object coordinates or image conjugate are
    required. This is exact for a quadratic kernel with a convergent
    Gaussian integral; higher-order actions need a local approximation.
    """
    eye = jnp.eye(2, dtype=ray.Q_inv.dtype)
    go, ga = gradient[:2], gradient[2:]
    hoo, hoa = hessian[:2, :2], hessian[:2, 2:]
    hao, haa = hessian[2:, :2], hessian[2:, 2:]
    t = eye - distance * ray.Q_inv
    r = jnp.linalg.solve(eye + distance * t @ hoo, t)

    # This form avoids subtracting 1/distance terms and includes distance=0.
    d0 = value - 0.5 * distance * go @ r @ go
    d1 = ga + (eye - distance * hao) @ r @ go
    d2 = haa + t @ hoo @ r + hao @ r + r @ hoa - distance * hao @ r @ hoa
    d2 = 0.5 * (d2 + d2.T)

    object_q = jnp.linalg.solve(t, ray.Q_inv)

    def fresnel_root(q):
        determinant = jnp.linalg.det(eye + distance * q)
        root = jnp.sqrt(determinant)
        # Same continuation as FreeSpacePropagator: each modal factor has
        # imaginary part with the sign of distance for a decaying Gaussian.
        return jnp.where(distance * jnp.imag(determinant) < 0.0, -root, root)

    factor = fresnel_root(object_q) / fresnel_root(object_q + hoo)
    d0 = d0 - 1j * jnp.log(factor) / ray.k
    return d0, d1, d2


def apply_action_delta(
    ray: GaussianBeam,
    dS0: complex,
    dS1: jnp.ndarray,
    dS2: jnp.ndarray,
    tiny: float = 1e-30,
):
    """Apply local complex-action coefficients to one beam or a bundle."""
    if jnp.asarray(ray.r_xy).ndim > 1:
        return _apply_action_delta_batched(ray, dS0, dS1, dS2, tiny=tiny)
    return _apply_action_delta_single(ray, dS0, dS1, dS2, tiny=tiny)


def _broadcast_action_term(
    value: Any,
    n: int,
    trailing_shape: tuple[int, ...],
) -> jnp.ndarray:
    arr = jnp.asarray(value)
    target_shape = (n,) + trailing_shape
    if arr.shape == target_shape:
        return arr
    if arr.shape == trailing_shape:
        return jnp.broadcast_to(arr, target_shape)
    if (
        arr.ndim == len(trailing_shape) + 1
        and arr.shape[0] == 1
        and arr.shape[1:] == trailing_shape
    ):
        return jnp.broadcast_to(arr, target_shape)
    return arr


def _apply_action_delta_batched(
    ray: GaussianBeam,
    dS0: complex,
    dS1: jnp.ndarray,
    dS2: jnp.ndarray,
    tiny: float = 1e-30,
):
    r_xy = ray.r_xy
    n = r_xy.shape[0]
    dS0 = _broadcast_action_term(dS0, n, ())
    dS1 = _broadcast_action_term(dS1, n, (2,))
    dS2 = _broadcast_action_term(dS2, n, (2, 2))

    S0_prime = ray.pathlength + dS0
    S1_prime = ray.d_xy + dS1
    Q_prime = ray.Q_inv + dS2

    ImQ = jnp.imag(Q_prime)
    ImS1 = jnp.imag(S1_prime)
    a = ImQ[..., 0, 0]
    b = ImQ[..., 0, 1]
    c = ImQ[..., 1, 0]
    d = ImQ[..., 1, 1]
    det = a * d - b * c
    singular = jnp.abs(det) < tiny
    det_safe = jnp.where(singular, 1.0, det)
    rhs0 = -ImS1[..., 0]
    rhs1 = -ImS1[..., 1]
    dx = jnp.stack(
        (
            (d * rhs0 - b * rhs1) / det_safe,
            (-c * rhs0 + a * rhs1) / det_safe,
        ),
        axis=-1,
    )
    dx = jnp.where(singular[..., None], jnp.zeros_like(dx), dx)

    qdx = jnp.stack(
        (
            Q_prime[..., 0, 0] * dx[..., 0]
            + Q_prime[..., 0, 1] * dx[..., 1],
            Q_prime[..., 1, 0] * dx[..., 0]
            + Q_prime[..., 1, 1] * dx[..., 1],
        ),
        axis=-1,
    )
    r_xy_new = r_xy + dx
    S0_new = (
        S0_prime
        + jnp.sum(S1_prime * dx, axis=-1)
        + 0.5 * jnp.sum(dx * qdx, axis=-1)
    )
    S1_new = S1_prime + qdx

    return (
        r_xy_new,
        jnp.real(S1_new),
        ray.amplitude * jnp.exp(-ray.k * jnp.imag(S0_new)),
        jnp.real(S0_new),
        Q_prime,
    )


def _apply_action_delta_single(
    ray: GaussianBeam,
    dS0: complex,
    dS1: jnp.ndarray,
    dS2: jnp.ndarray,
    tiny: float = 1e-30,
):

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
        return _solve_2x2(ImQ_, -ImS1_)

    def zero_dx(args):
        _, ImS1_ = args
        return jnp.zeros_like(ImS1_)

    dx = lax.cond(
        jnp.abs(_det_2x2(ImQ)) < tiny,
        zero_dx,
        solve_dx,
        (ImQ, ImS1),
    )

    r_xy_new = ray.r_xy + dx
    S0_new = S0_prime + jnp.dot(S1_prime, dx) + 0.5 * jnp.dot(dx, Q_prime @ dx)
    S1_new = S1_prime + Q_prime @ dx
    return (
        r_xy_new,
        jnp.real(S1_new),
        ray.amplitude * jnp.exp(-ray.k * jnp.imag(S0_new)),
        jnp.real(S0_new),
        Q_prime,
    )


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
    """Free-space Gaussian transport: Q, amplitude (Gouy phase) and excess optical path.

    ``pathlength`` gains ``d * (dx**2 + dy**2) / 2``, the same excess path a ray gains in
    :class:`temgym_core.propagator.FreeSpaceParaxial`; the common axial carrier ``k * d`` is
    never accumulated, so phases stay small over metre-long columns.
    """

    def __call__(self, ray: GaussianBeam, distance: float) -> GaussianBeam:
        distance = jnp.asarray(distance, dtype=jnp.float64)
        q = jnp.asarray(ray.Q_inv)
        identity = jnp.eye(2, dtype=q.dtype)
        denominator = identity + distance[..., None, None] * q
        det_a = jnp.linalg.det(denominator)

        # det(I + dQ)**(-1/2) needs a continuation-aware square root. For a
        # physical Gaussian Im(Q)>0, each modal factor stays in the upper
        # half-plane for d>0 (lower for d<0). Their product wraps across the
        # principal determinant branch exactly when d*Im(det) becomes
        # negative. Correcting that sign avoids the Gouy-phase jump without
        # differentiating through a degenerate eigendecomposition.
        determinant_root = jnp.sqrt(det_a)
        determinant_root = jnp.where(
            distance * jnp.imag(det_a) < 0.0,
            -determinant_root,
            determinant_root,
        )
        amplitude_factor = 1.0 / determinant_root
        # Solve once for both Q @ inv(D) and inv(D). Transport Im(Q) by
        # congruence rather than extracting it from a complex quotient: near
        # an astigmatic focus that quotient can lose the tiny positive decay
        # eigenvalue and turn a decaying Gaussian into an overflowing one.
        rhs = jnp.concatenate((jnp.swapaxes(q, -1, -2),
                               jnp.broadcast_to(identity, q.shape)), axis=-1)
        solved = jnp.linalg.solve(jnp.swapaxes(denominator, -1, -2), rhs)
        Q_new = jnp.swapaxes(solved[..., :2], -1, -2)
        inverse = jnp.swapaxes(solved[..., 2:], -1, -2)
        imaginary = jnp.real(jnp.swapaxes(inverse.conj(), -1, -2)
                             @ jnp.imag(q) @ inverse)
        Q_new = jnp.real(Q_new) + 1j*imaginary
        Q_new = .5*(Q_new+jnp.swapaxes(Q_new, -1, -2))

        dx = jnp.asarray(ray.dx, dtype=jnp.float64)
        dy = jnp.asarray(ray.dy, dtype=jnp.float64)
        x_new = ray.x + distance * dx
        y_new = ray.y + distance * dy
        theta_sq = dx * dx + dy * dy

        return ray.derive(
            x=x_new,
            y=y_new,
            dx=dx,
            dy=dy,
            z=ray.z + distance,
            amplitude=ray.amplitude * amplitude_factor,
            pathlength=ray.pathlength + 0.5 * distance * theta_sq,
            Q_inv=Q_new,
        )


def _propagate_free_space_impl(ray: GaussianBeam, distance: float) -> GaussianBeam:
    return FreeSpacePropagator()(ray, distance)


propagate_free_space_jit = jax.jit(_propagate_free_space_impl)


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
    "propagate_free_space_jit",
]
