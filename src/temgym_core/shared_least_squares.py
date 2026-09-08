"""JAX block least squares with independent local and shared parameters.

Eliminate local increments with thin SVDs, then solve for shared increments.
Every residual and correlation is retained. This is the same local
Gauss--Newton problem as a dense joint fit, without storing its large Jacobian.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class LocalResidual:
    """residual(local, shared, *args); args contains recorded data, never truth."""

    residual: object
    args: tuple = ()


@dataclass(frozen=True)
class SharedLeastSquaresResult:
    shared: object
    local: object
    shared_covariance: object
    local_covariances: object
    local_responses: object
    success: bool
    message: str
    history: tuple
    local_residual_rms: object


@jax.jit
def _eliminate(local, shared, error):
    u, s, vt = jnp.linalg.svd(local, full_matrices=False)
    inverse = (vt.T / s) @ u.T
    return (
        shared - u @ (u.T @ shared),
        error - u @ (u.T @ error),
        -inverse @ error,
        -inverse @ shared,
        (vt.T / s) @ (vt / s[:, None]),
        s,
    )


@jax.jit
def _solve(design, error):
    u, s, vt = jnp.linalg.svd(design, full_matrices=False)
    return -(vt.T / s) @ (u.T @ error), (vt.T / s) @ (vt / s[:, None]), s


def fit_shared_least_squares(
    *,
    shared_residual,
    blocks,
    starting_shared,
    starting_local,
    shared_bounds,
    max_iterations=15,
    relative_cost_tolerance=1e-7,
    shared_step_tolerance=1e-5,
):
    """Joint local fit; bounds apply to shared parameters, as in the beam analysis.

    Thin SVD solves use JAX. The host only schedules blocks and accepts steps.
    No soft penalties or truncated singular directions are introduced. Returned
    covariance is re-evaluated at the returned parameters, conditional on the
    supplied likelihood and away from active bounds.
    """
    shared = jnp.asarray(starting_shared)
    local = jnp.asarray(starting_local)
    lower, upper = map(jnp.asarray, shared_bounds)
    if local.ndim != 2 or len(local) != len(blocks) or not len(blocks):
        raise ValueError("Supply one local parameter row per residual block.")
    if not bool(jnp.all((shared > lower) & (shared < upper))):
        raise ValueError("Shared starting parameters must lie inside bounds.")
    common = jax.jit(shared_residual)
    common_jac = jax.jit(jax.jacfwd(shared_residual))
    compiled = {}
    evaluators = []
    for block in blocks:
        if block.residual not in compiled:
            compiled[block.residual] = (
                jax.jit(block.residual),
                jax.jit(jax.jacfwd(block.residual, argnums=(0, 1))),
            )
        evaluators.append((*compiled[block.residual], block.args))

    def cost(loc, glob):
        total = jnp.sum(common(glob) ** 2)
        for values, (residual, _, args) in zip(loc, evaluators):
            total = total + jnp.sum(residual(values, glob, *args) ** 2)
        return float(total)

    def linearize(loc, glob):
        designs, errors = [common_jac(glob)], [common(glob)]
        steps, responses, covariances = [], [], []
        for values, (residual, jacobian, args) in zip(loc, evaluators):
            ll, gg = jacobian(values, glob, *args)
            design, error, step, response, cov, s = _eliminate(
                ll, gg, residual(values, glob, *args)
            )
            if not bool(jnp.all(jnp.isfinite(s) & (s > 0))):
                raise ValueError("Nonfinite or singular local Jacobian.")
            designs.append(design)
            errors.append(error)
            steps.append(step)
            responses.append(response)
            covariances.append(cov)
        step, shared_cov, s = _solve(jnp.concatenate(designs), jnp.concatenate(errors))
        if not bool(jnp.all(jnp.isfinite(s) & (s > 0))):
            raise ValueError("Nonfinite or singular shared Jacobian.")
        responses = jnp.stack(responses)
        return (
            step,
            jnp.stack(steps) + responses @ step,
            shared_cov,
            jnp.stack(covariances),
            responses,
        )

    history = []
    success = False
    message = "Iteration budget exhausted."
    for iteration in range(max_iterations):
        shared_step, local_step, _, _, _ = linearize(local, shared)
        old_cost = cost(local, shared)
        fraction = 1.0
        while fraction >= 2**-14:
            candidate = shared + fraction * shared_step
            new_cost = (
                cost(local + fraction * local_step, candidate)
                if bool(jnp.all((candidate > lower) & (candidate < upper)))
                else np.inf
            )
            if np.isfinite(new_cost) and new_cost <= old_cost + 1e-7:
                break
            fraction /= 2
        if fraction < 2**-14:
            message = "Line search failed."
            break
        local = local + fraction * local_step
        shared = candidate
        norm = float(jnp.linalg.norm(fraction * shared_step))
        history.append(
            {
                "iteration": iteration,
                "chi_squared": new_cost,
                "projector_step": norm,
                "step_fraction": fraction,
            }
        )
        print(
            f"Joint image analysis {iteration}: cost {new_cost:.6g}, shared step {norm:.3g}",
            flush=True,
        )
        if (
            abs(new_cost - old_cost) < relative_cost_tolerance * max(old_cost, 1.0)
            and norm < shared_step_tolerance
        ):
            success = True
            message = "Accepted cost and shared-step tolerances satisfied."
            break
    _, _, cov, local_cov, response = linearize(local, shared)
    rms = jnp.array(
        [
            jnp.sqrt(jnp.mean(residual(values, shared, *args) ** 2))
            for values, (residual, _, args) in zip(local, evaluators)
        ]
    )
    return SharedLeastSquaresResult(
        shared, local, cov, local_cov, response, success, message, tuple(history), rms
    )
