"""Component interface and the shared local optical-path/amplitude update."""


from __future__ import annotations



import jax
import jax.numpy as jnp

from ..gaussian import (
    GaussianBeam,
    _solve_2x2,
    apply_action_delta,
)
from ..ray import Ray
from ..tree_utils import HasParamsMixin




# Component interface and Gaussian action

class Component(HasParamsMixin):
    """A component transforms a Ray or GaussianBeam at its current plane.

    Call component(beam); dispatch uses beam.ray_family. The runner handles
    free-space travel to component.z. A finite component handles its own body
    propagation. Unsupported beam families raise NotImplementedError.
    """

    def __call__(self, ray: Ray) -> Ray:
        family = getattr(ray, "ray_family", "ray")
        if family == "gaussian":
            return self._call_gaussian(ray)
        if family == "ray":
            return self._call_ray(ray)
        raise TypeError(
            f"Unsupported ray family '{family}' for {type(self).__name__}."
        )

    def _call_ray(self, ray: Ray) -> Ray:
        raise NotImplementedError(
            f"{type(self).__name__} is not implemented for ray_family='ray'."
        )

    def _call_gaussian(self, ray: GaussianBeam) -> GaussianBeam:
        raise NotImplementedError(
            f"{type(self).__name__} is not implemented for ray_family='gaussian'."
        )


def _call_single_or_batched_gaussian(ray: GaussianBeam, single_call):
    if jnp.asarray(ray.r_xy).ndim > 1:
        return jax.vmap(single_call)(ray.to_vector())
    return single_call(ray)


def _default_action_terms(xy_ref: jnp.ndarray):
    return (
        jnp.zeros((), dtype=xy_ref.dtype),
        jnp.zeros_like(xy_ref),
        jnp.zeros((2, 2), dtype=xy_ref.dtype),
    )


def _value_grad_hessian(fn, x: jnp.ndarray):
    value, grad = jax.value_and_grad(fn)(x)
    return value, grad, jax.hessian(fn)(x)


def _uses_default_log_transmission(component: "GaussianActionComponent") -> bool:
    return type(component).log_transmission is GaussianActionComponent.log_transmission


def _min_eigval_2x2(M: jnp.ndarray) -> jnp.ndarray:
    """Minimum eigenvalue of a real symmetric 2×2 matrix."""
    tr = M[..., 0, 0] + M[..., 1, 1]
    # Difference form avoids cancellation at a round Gaussian. At an exact
    # repeated eigenvalue choose the symmetric subgradient (half the trace).
    # Guard the sqrt input itself: masking sqrt(0) afterwards still produces
    # 0 * inf in autodiff, even when the envelope regularizer is inactive.
    disc = (M[..., 0, 0] - M[..., 1, 1]) ** 2 + (M[..., 0, 1] + M[..., 1, 0]) ** 2
    distinct = disc > 0
    root = jnp.sqrt(jnp.where(distinct, disc, 1.))
    return 0.5 * (tr - jnp.where(distinct, root, 0.))


def _regularize_L2k(
    L2k: jnp.ndarray,
    ImQ: jnp.ndarray,
    min_frac: float = 0.1,
) -> jnp.ndarray:
    """Ensure ``ImQ - L2k`` stays positive definite.

    If the minimum eigenvalue of ``ImQ - L2k`` would drop below
    ``min_frac * min_eigval(ImQ)``, a diagonal shift is subtracted
    from *L2k* to bring it back within bounds.
    """
    M = ImQ - L2k
    min_eig_M = _min_eigval_2x2(M)
    min_eig_Q = _min_eigval_2x2(ImQ)
    required = min_frac * min_eig_Q
    shift = jnp.maximum(required - min_eig_M, 0.0)
    return L2k - shift[..., None, None] * jnp.eye(2, dtype=L2k.dtype)


def _apply_gaussian_action(ray, path_terms, log_amplitude_terms=None):
    """Apply local value/gradient/Hessian terms from AD or specimen interpolation."""
    dS0, dS1, dS2 = path_terms
    if log_amplitude_terms is not None:
        log_value, log_gradient, log_hessian = log_amplitude_terms
        k = ray.k
        im_q = jnp.imag(ray.Q_inv)
        # Limit the inferred shift toward higher transmission, then keep the
        # updated Gaussian envelope decaying. Both specimen paths use this rule.
        estimated_shift = _solve_2x2(im_q, log_gradient) / k
        scale = 1.0 / (1.0 + jnp.dot(log_gradient, estimated_shift))
        log_gradient = log_gradient * scale
        log_hessian = log_hessian * scale
        log_hessian_over_k = _regularize_L2k(log_hessian / k, im_q)
        dS0 = dS0 - 1j * (log_value / k)
        dS1 = dS1 - 1j * (log_gradient / k)
        dS2 = dS2 - 1j * log_hessian_over_k

    r_xy, d_xy, amplitude, pathlength, q_new = apply_action_delta(
        ray, dS0=dS0, dS1=dS1, dS2=dS2
    )
    return ray.derive(
        x=r_xy[..., 0], y=r_xy[..., 1],
        dx=d_xy[..., 0], dy=d_xy[..., 1], z=ray.z,
        amplitude=amplitude, pathlength=pathlength, Q_inv=q_new,
    )


class GaussianActionComponent(Component):
    """Base for thin elements described by optical path and field transmission.

    Despite its name, phase_shift(xy) returns optical path, in the beam's
    length unit. log_transmission(xy) returns log field amplitude. Together:

        field multiplier = exp(i k path + log_amplitude)
        complex action  = path - i log_amplitude / k

    Values, gradients and Hessians update amplitude, beam centre and shape.
    The shared action helper limits absorption-driven recentering and keeps
    the Gaussian envelope decaying. Subclasses can provide exact quadratic
    updates or interpolated derivatives instead of automatic differentiation.
    """

    def phase_shift(self, xy: jnp.ndarray):
        return 0.0

    def log_transmission(self, xy: jnp.ndarray):
        return 0.0

    def complex_action(self, xy: jnp.ndarray, k: float) -> complex:
        logA = self.log_transmission(xy)
        L = jnp.logaddexp(logA, -50)
        return self.phase_shift(xy) - 1j * (L / k)

    def _call_gaussian(self, ray: GaussianBeam) -> GaussianBeam:
        return _call_single_or_batched_gaussian(ray, self._call_gaussian_single)

    def _call_gaussian_single(self, ray: GaussianBeam) -> GaussianBeam:
        xy = ray.r_xy
        if type(self).phase_shift is GaussianActionComponent.phase_shift:
            path_terms = _default_action_terms(xy)
        else:
            path_terms = _value_grad_hessian(lambda p: jnp.real(self.phase_shift(p)), xy)

        log_terms = None
        if not _uses_default_log_transmission(self):
            def log_transmission(p):
                return jnp.real(jnp.logaddexp(self.log_transmission(p), -50.0))

            log_terms = _value_grad_hessian(log_transmission, xy)
        return _apply_gaussian_action(ray, path_terms, log_terms)


def _matvec(matrix: jnp.ndarray, vector: jnp.ndarray) -> jnp.ndarray:
    return jnp.einsum("...ij,...j->...i", matrix, vector)


def _matmat(matrix: jnp.ndarray, tensor: jnp.ndarray) -> jnp.ndarray:
    return jnp.einsum("...ij,...jk->...ik", matrix, tensor)


def _rotate_quadratic_form(rotation: jnp.ndarray, q_inv: jnp.ndarray) -> jnp.ndarray:
    """Rotate Q consistently with r' = R r: Q' = R Q R.T."""
    return jnp.einsum("...ij,...jk,...lk->...il", rotation, q_inv, rotation)
