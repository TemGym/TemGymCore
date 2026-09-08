"""Small bounded, scaled least-squares solver with JAX device computations.

This is a *local* damped Gauss--Newton method, not a branch search or a proof
of identifiability.  The host controls acceptance; forward evaluations,
``jacfwd`` derivatives and augmented-SVD steps run on the active JAX device.
No normal equations or numerical-rank truncation are used.  Whiten residuals
before calling this routine when fitting observations with unequal errors.
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class LeastSquaresResult:
    """Local fit and termination evidence; arrays remain on the JAX device."""

    x: jax.Array
    residual: jax.Array
    jacobian: jax.Array  # Derivatives with respect to physical x, not x / scale.
    cost: float  # One half of the sum of squared residuals.
    optimality: float  # Infinity norm of the scaled, projected gradient.
    success: bool
    status: str
    message: str
    nfev: int  # Explicit residual calls; does not count internal AD evaluations.
    njev: int
    nit: int  # Trial steps, including rejected trials.
    history: tuple[dict, ...]


@jax.jit
def _projected_gradient(x, lower, upper, scale, jacobian, residual):
    gradient = (jacobian * scale).T @ residual
    # Exact clipping puts binding coordinates on the bound.  A tolerance
    # proportional to |x| would wrongly immobilize small physical distances.
    active = ((x <= lower) & (gradient > 0)) | ((x >= upper) & (gradient < 0))
    projected = jnp.where(active, 0.0, gradient)
    return projected, active


@jax.jit
def _damped_step(x, lower, upper, scale, jacobian, residual, damping):
    gradient, active = _projected_gradient(x, lower, upper, scale, jacobian, residual)
    matrix = jnp.where(active[None, :], 0.0, jacobian * scale)
    # Normalize the augmented system, not J.T @ J: this avoids squaring its
    # condition number.  Damping is dimensionless after this normalization.
    normalization = jnp.maximum(
        jnp.max(jnp.linalg.norm(matrix, axis=0)), jnp.finfo(matrix.dtype).tiny
    )
    n = x.size
    augmented = jnp.concatenate(
        (matrix / normalization, jnp.sqrt(damping) * jnp.eye(n, dtype=x.dtype))
    )
    rhs = jnp.concatenate((-residual / normalization, jnp.zeros(n, dtype=x.dtype)))
    u, singular, vh = jnp.linalg.svd(augmented, full_matrices=False)
    # Damping gives every direction a positive singular value.  In particular
    # we retain weak and exactly rank-deficient physical directions.
    step = vh.T @ ((u.T @ rhs) / singular)
    step = jnp.where(active, 0.0, step)
    candidate = jnp.clip(x + scale * step, lower, upper)
    step = (candidate - x) / scale
    linear_change = (jacobian * scale) @ step
    prediction = -jnp.dot(gradient, step) - 0.5 * jnp.dot(linear_change, linear_change)
    return candidate, step, prediction


def least_squares(
    residual_fun: Callable,
    x0,
    *,
    bounds,
    x_scale=None,
    jac: Callable | None = None,
    args: tuple = (),
    max_nfev: int = 1000,
    max_iter: int = 1000,
    ftol: float = 1e-10,
    xtol: float = 1e-10,
    gtol: float = 1e-8,
    initial_damping: float = 1e-3,
    jit: bool = True,
) -> LeastSquaresResult:
    """Minimize ``0.5 * sum(residual_fun(x)**2)`` inside a finite box.

    ``residual_fun`` must be JAX differentiable and return a nonempty, real
    vector.  Bounds are two arrays (or scalars) broadcastable to ``x0``;
    each lower bound must be strictly below its upper bound.  Fixed physical
    quantities should therefore be omitted from the fitted parameter vector.
    ``args`` holds dynamic residual inputs such as measured images; reusing
    one residual function avoids recompiling for every frame.
    ``jac`` optionally supplies a JAX-compatible Jacobian. Otherwise jacfwd
    differentiates the residual. Set ``jit=False`` when the residual and a
    supplied Jacobian schedule separately compiled blocks on the host.
    ``x_scale="jac"`` scales each column by its
    inverse norm; accumulated maximum norms prevent growing steps in weak
    directions as the solve progresses. No singular direction is discarded.
    Numeric ``x_scale`` contains positive physical increments, for example metres for
    positions and radians for angles.  It affects conditioning and the scaled
    projected-gradient and step stopping criteria, not the likelihood.

    Jacobians are evaluated only at accepted iterates.  ``nfev`` counts the
    initial and all trial residual calls; ``njev`` counts separate ``jacfwd``
    calls, including their internal forward calculation.  Iteration/evaluation
    limits, nonfinite values and damping exhaustion never imply success.  A
    small *rejected* step never satisfies a convergence test.  The returned
    Jacobian describes the returned point, including at a budget limit.

    The routine respects the configured JAX dtype/device; enable float64
    before creating inputs when the physical scales demand it.  It does not
    force a GPU or silently move a caller's computation to the CPU.
    """
    initial = np.asarray(x0)
    if (
        initial.ndim != 1
        or initial.size == 0
        or np.iscomplexobj(initial)
        or not np.all(np.isfinite(initial))
    ):
        raise ValueError("x0 must be a nonempty finite real vector")
    try:
        lower, upper = (
            np.broadcast_to(np.asarray(b, dtype=float), initial.shape) for b in bounds
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "bounds must be a lower/upper pair broadcastable to x0"
        ) from exc
    if not (
        np.all(np.isfinite(lower))
        and np.all(np.isfinite(upper))
        and np.all(lower < upper)
    ):
        raise ValueError("bounds must be finite with lower < upper")
    if np.any(initial < lower) or np.any(initial > upper):
        raise ValueError("x0 must lie inside bounds; the initial point is not clipped")
    adaptive_scale = isinstance(x_scale, str) and x_scale == "jac"
    try:
        scale = np.broadcast_to(
            np.asarray(
                1.0 if x_scale is None or adaptive_scale else x_scale, dtype=float
            ),
            initial.shape,
        )
    except ValueError as exc:
        raise ValueError("x_scale must be broadcastable to x0") from exc
    if not np.all(np.isfinite(scale) & (scale > 0)):
        raise ValueError("x_scale must be positive and finite")
    for name, limit, minimum in (("max_nfev", max_nfev, 1), ("max_iter", max_iter, 0)):
        if (
            isinstance(limit, bool)
            or not isinstance(limit, (int, np.integer))
            or limit < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if any(not np.isfinite(t) or t < 0 for t in (ftol, xtol, gtol)):
        raise ValueError("tolerances must be finite and nonnegative (zero disables)")
    if not np.isfinite(initial_damping) or initial_damping <= 0:
        raise ValueError("initial_damping must be positive and finite")

    dtype = jnp.result_type(initial, jnp.float32)
    x, lower, upper, scale = (
        jnp.asarray(a, dtype=dtype) for a in (initial, lower, upper, scale)
    )
    # Cast-to-device precision may collapse otherwise distinct physical bounds.
    if not bool(jnp.all((lower < upper) & jnp.isfinite(scale) & (scale > 0))):
        raise ValueError(
            "bounds or scales are not representable in the active JAX dtype"
        )
    # A supplied Jacobian may schedule separately compiled blocks to bound GPU
    # memory. In that case keep the scheduler on the host; all numerical work
    # and the least-squares step still execute in JAX.
    value_fun = jax.jit(residual_fun) if jit else residual_fun
    derivative = jax.jacfwd(residual_fun) if jac is None else jac
    jacobian_fun = jax.jit(derivative) if jit else derivative
    residual = jnp.asarray(value_fun(x, *args))
    if residual.ndim != 1 or residual.size == 0 or jnp.iscomplexobj(residual):
        raise ValueError("residual_fun must return a nonempty real vector")
    nfev, njev, nit = 1, 0, 0
    history = []
    jacobian = jnp.full((residual.size, x.size), jnp.nan, dtype=dtype)
    cost = float(0.5 * jnp.vdot(residual, residual))
    optimality = float("inf")

    def finish(status, message, success=False):
        return LeastSquaresResult(
            x,
            residual,
            jacobian,
            cost,
            optimality,
            success,
            status,
            message,
            nfev,
            njev,
            nit,
            tuple(history),
        )

    if not bool(jnp.all(jnp.isfinite(residual))) or not np.isfinite(cost):
        return finish("nonfinite_initial", "Initial residual or cost is nonfinite.")
    jacobian = jacobian_fun(x, *args)
    njev += 1
    column_norms = jnp.linalg.norm(jacobian, axis=0)
    if adaptive_scale:
        scale = 1.0 / jnp.where(column_norms > 0, column_norms, 1.0)
    damping = float(initial_damping)
    for iteration in range(max_iter + 1):
        if not bool(jnp.all(jnp.isfinite(jacobian))):
            return finish(
                "nonfinite_jacobian", "Jacobian at the current iterate is nonfinite."
            )
        gradient, _ = _projected_gradient(x, lower, upper, scale, jacobian, residual)
        optimality = float(jnp.max(jnp.abs(gradient)))
        if not np.isfinite(optimality):
            return finish(
                "nonfinite_gradient", "Scaled projected gradient is nonfinite."
            )
        if gtol > 0 and optimality <= gtol:
            return finish("gtol", "Scaled projected gradient satisfies gtol.", True)
        if nfev >= max_nfev:
            return finish("evaluation_limit", "Residual evaluation budget exhausted.")
        if iteration == max_iter:
            return finish("iteration_limit", "Trial-step iteration budget exhausted.")
        candidate, step, prediction = _damped_step(
            x, lower, upper, scale, jacobian, residual, damping
        )
        nit += 1
        prediction = float(prediction)
        step_norm = float(jnp.linalg.norm(step))
        if not (
            bool(jnp.all(jnp.isfinite(candidate)))
            and np.isfinite(prediction)
            and np.isfinite(step_norm)
        ):
            return finish("nonfinite_step", "The augmented-SVD step is nonfinite.")
        old_cost = cost
        reduction, ratio = float("-inf"), float("-inf")
        accepted = False
        if prediction > 0 and step_norm > 0:
            trial_residual = value_fun(candidate, *args)
            nfev += 1
            trial_cost = float(0.5 * jnp.vdot(trial_residual, trial_residual))
            if bool(jnp.all(jnp.isfinite(trial_residual))) and np.isfinite(trial_cost):
                reduction = old_cost - trial_cost
                ratio = reduction / prediction
                accepted = reduction > 0 and ratio > 1e-4
        history.append(
            {
                "iteration": nit,
                "cost": old_cost,
                "damping": damping,
                "step_norm": step_norm,
                "predicted_reduction": prediction,
                "actual_reduction": reduction,
                "ratio": ratio,
                "accepted": accepted,
            }
        )
        if accepted:
            previous_scaled_norm = float(jnp.linalg.norm(x / scale))
            x, residual, cost = candidate, trial_residual, trial_cost
            jacobian = jacobian_fun(x, *args)
            njev += 1
            if not bool(jnp.all(jnp.isfinite(jacobian))):
                return finish(
                    "nonfinite_jacobian",
                    "Jacobian at the accepted iterate is nonfinite.",
                )
            if adaptive_scale:
                column_norms = jnp.maximum(
                    column_norms, jnp.linalg.norm(jacobian, axis=0)
                )
                scale = 1.0 / jnp.where(column_norms > 0, column_norms, 1.0)
            gradient, _ = _projected_gradient(
                x, lower, upper, scale, jacobian, residual
            )
            optimality = float(jnp.max(jnp.abs(gradient)))
            if not np.isfinite(optimality):
                return finish(
                    "nonfinite_gradient", "Scaled projected gradient is nonfinite."
                )
            if gtol > 0 and optimality <= gtol:
                return finish("gtol", "Scaled projected gradient satisfies gtol.", True)
            # Require agreement with the linear model, not merely a tiny step
            # caused by increasing damping after repeated failed proposals.
            if ratio > 0.25:
                if (
                    ftol > 0
                    and reduction <= ftol * old_cost
                    and prediction <= ftol * old_cost
                ):
                    return finish(
                        "ftol",
                        "Accepted cost and predicted reduction satisfy ftol.",
                        True,
                    )
                if xtol > 0 and step_norm <= xtol * (xtol + previous_scaled_norm):
                    return finish("xtol", "Accepted scaled step satisfies xtol.", True)
            damping *= 1.0 / 3.0 if ratio > 0.75 else (4.0 if ratio < 0.25 else 1.0)
            damping = max(damping, np.finfo(np.dtype(dtype)).eps ** 2)
        else:
            damping *= 10.0
        if not np.isfinite(damping) or damping > 1e30:
            return finish(
                "stalled", "Damping exhausted without a converged accepted step."
            )
    raise AssertionError("unreachable")
