"""SVD trust-region least squares with a user-supplied physical domain.

JAX evaluates residuals, derivatives and directional curvature on its active
device. The small parameter-space SVD and feasibility/line-search loop run on
the host. All singular directions are retained; damping constrains a step and
is never added to the reported likelihood as a parameter prior.

Geodesic acceleration is optional step construction, not permission to move
uphill. It must preserve descent and positive model reduction, and failed
accelerated line searches fall back to the original damped Gauss--Newton step.
"""

import jax
import jax.numpy as jnp
import numpy as np


def _trust_region_step(left, singular, right, residual, radius):
    """Return a dimensionless damped GN step inside radius, with a true bracket."""
    projected = left.T@residual
    numerator = singular*projected

    def step(damping):
        denominator = singular**2+damping
        ratio = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
        return -(right.T@ratio)

    initial = step(0.)
    if np.linalg.norm(initial) <= radius:
        return initial, 0., 0
    low, high = 0., max(float(singular[0]**2), np.finfo(float).tiny)
    expansions = 0
    while np.linalg.norm(step(high)) > radius:
        high *= 4.; expansions += 1
        if not np.isfinite(high) or expansions > 512:
            raise FloatingPointError('Could not bracket the trust-region damping.')
    for _ in range(80):
        middle = (low+high)/2
        if np.linalg.norm(step(middle)) > radius:
            low = middle
        else:
            high = middle
    return step(high), high, expansions


def _model_reduction(jacobian, residual, step):
    response = jacobian@step
    directional = float(residual@response)
    return directional, float(-2*directional-response@response)


def _guard_acceleration(jacobian, residual, base_step, acceleration_step, scales, radius):
    """Safeguard even small accelerations in ill-conditioned coordinates."""
    ratio = np.linalg.norm(acceleration_step/scales)/max(np.linalg.norm(base_step/scales), 1e-300)
    if not np.isfinite(ratio) or ratio >= 1.:
        return base_step, 'curvature_ratio', float(ratio)
    candidate = base_step+.5*acceleration_step
    natural_norm = np.linalg.norm(candidate/scales)
    if natural_norm > radius:
        candidate *= radius/natural_norm
    directional, predicted = _model_reduction(jacobian, residual, candidate)
    if not np.all(np.isfinite(candidate)) or not np.isfinite(directional+predicted):
        return base_step, 'nonfinite_acceleration', float(ratio)
    if directional >= 0:
        return base_step, 'non_descent', float(ratio)
    if predicted <= 0:
        return base_step, 'nonpositive_model_reduction', float(ratio)
    return candidate, 'accepted_for_line_search', float(ratio)


def svd_gauss_newton(residual, initial, admissible, *, max_iterations=100,
                     gradient_tolerance=1e-6, native_scales=None):
    """Minimize original residuals without accepting false physical convergence.

    ``admissible(parameters)`` is a host boolean domain check. ``native_scales``
    defines physical parameter changes of unit trust distance; it defaults to
    ones and must be positive. Returned keys match the earlier example solver.
    Convergence requires the column-normalized unconstrained gradient test;
    a blocked boundary, exhausted budget or failed line search is not success.
    """
    p = np.asarray(initial, dtype=float).copy()
    scales = np.ones_like(p) if native_scales is None else np.asarray(native_scales, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all():
        raise ValueError('Initial coordinates must be a finite nonempty vector.')
    if scales.shape != p.shape or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('Native scales must be positive finite values for every parameter.')
    if max_iterations < 1 or not np.isfinite(gradient_tolerance) or gradient_tolerance <= 0:
        raise ValueError('Positive iteration and gradient limits are required.')
    if not admissible(p):
        raise ValueError('Initializer outside the declared physical domain.')
    fun = jax.jit(lambda value: jnp.asarray(residual(value)).reshape(-1))
    jacobian = jax.jit(jax.jacfwd(fun))
    curvature = jax.jit(lambda value, direction: jax.jvp(
        lambda q: jax.jvp(fun, (q,), (direction,))[1], (value,), (direction,))[1])
    r = np.asarray(fun(p)); evaluations = 1
    history, radius, success = [], 1., False
    reason = 'iteration budget exhausted'
    for iteration in range(max_iterations):
        jac = np.asarray(jacobian(p))
        if not np.isfinite(r).all() or not np.isfinite(jac).all():
            reason = 'nonfinite residual or Jacobian'; break
        norms = np.maximum(np.linalg.norm(jac, axis=0), 1e-12)
        normalized_singular = np.linalg.svd(jac/norms, compute_uv=False)
        gradient = float(np.max(np.abs((jac/norms).T@r)))
        record = dict(iteration=iteration, chi_square=float(r@r), normalized_gradient=gradient,
            singular_min=float(normalized_singular[-1]), singular_max=float(normalized_singular[0]),
            attempts=[])
        history.append(record)
        if gradient < gradient_tolerance:
            success, reason = True, 'normalized gradient tolerance'; break
        left, singular, right = np.linalg.svd(jac*scales, full_matrices=False)
        accepted = False
        for radius_trial in range(8):
            natural, damping, expansions = _trust_region_step(left, singular, right, r, radius)
            base_step = natural*scales
            base_directional, base_predicted = _model_reduction(jac, r, base_step)
            attempt = dict(trust_radius=radius, damping=damping, damping_bracket_expansions=expansions,
                base_natural_norm=float(np.linalg.norm(natural)),
                base_directional_derivative=base_directional, base_predicted_reduction=base_predicted,
                line_searches=[])
            record['attempts'].append(attempt)
            if not np.isfinite(base_directional+base_predicted) or base_directional >= 0 or base_predicted <= 0:
                radius *= .25
                attempt['rejected_reason'] = 'non_descent_or_nonpositive_base_model'; continue
            second = np.asarray(curvature(p, base_step))
            denominator = singular**2+damping
            acceleration = -(right.T@np.divide(singular*(left.T@second), denominator,
                out=np.zeros_like(singular), where=denominator > 0))*scales
            accelerated, guard, ratio = _guard_acceleration(jac, r, base_step, acceleration, scales, radius)
            attempt.update(geodesic_acceleration_ratio=ratio, acceleration_guard=guard)
            candidates = ([('accelerated', accelerated)] if guard == 'accepted_for_line_search' else [])
            candidates.append(('base_gauss_newton', base_step))
            for name, direction in candidates:
                directional, _ = _model_reduction(jac, r, direction)
                line = dict(direction=name, directional_derivative=directional,
                            feasible_trials=0, armijo_rejections=0)
                attempt['line_searches'].append(line)
                for power in range(35):
                    alpha = 2.**(-power)
                    candidate = p+alpha*direction
                    if not admissible(candidate):
                        continue
                    line['feasible_trials'] += 1
                    trial = np.asarray(fun(candidate)); evaluations += 1
                    actual = float(r@r-trial@trial)
                    predicted = _model_reduction(jac, r, alpha*direction)[1]
                    if (np.isfinite(trial).all() and actual > 0 and predicted > 0
                            and trial@trial <= r@r+2e-4*alpha*directional):
                        quality = actual/predicted
                        p, r, accepted = candidate, trial, True
                        record.update(step_fraction=alpha, trust_radius=radius, step_quality=quality,
                            damping=damping, accepted_direction=name,
                            predicted_reduction=predicted, actual_reduction=actual,
                            geodesic_acceleration_ratio=ratio, acceleration_guard=guard)
                        if alpha < .5 or quality < .25:
                            radius *= .25
                        elif quality > .75 and np.linalg.norm(natural) > .8*radius:
                            radius *= 2
                        break
                    line['armijo_rejections'] += 1
                if accepted:
                    break
            if accepted:
                break
            radius *= .25
        if not accepted:
            reason = 'physical feasible descent line search failed'; break
    jac = np.asarray(jacobian(p))
    if np.isfinite(r).all() and np.isfinite(jac).all():
        norms = np.maximum(np.linalg.norm(jac, axis=0), 1e-12)
        gradient = float(np.max(np.abs((jac/norms).T@r)))
        if gradient < gradient_tolerance:
            success, reason = True, 'normalized gradient tolerance'
    else:
        gradient = float('inf')
    return dict(parameters=p, residual=r, jacobian=jac, success=success, reason=reason,
        normalized_gradient=gradient, evaluations=evaluations, history=history)
