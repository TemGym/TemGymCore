"""Feasible, stationary-nuisance profiling of an unchanged JAX residual.

This is a bounded local experiment, not a global optimizer. Parameters are
ordered ``[outer, nuisance]``. The nuisance problem is nonlinear: profiling
does not replace the supplied residual with a covariance-moment likelihood.
There are no additional priors, discarded singular directions, or success
tests based solely on a small step. Active inequality-boundary optima require
a separate constrained/KKT treatment and are deliberately not certified here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def _information_factor(jacobian):
    """Full-rank inverse-information factor, with no singular-value clipping."""
    jacobian = np.asarray(jacobian)
    norms = np.linalg.norm(jacobian, axis=0)
    if not np.isfinite(jacobian).all() or np.any(norms == 0):
        raise ValueError('nonfinite or rank-deficient derivative')
    if jacobian.shape[0] < jacobian.shape[1]:
        raise ValueError('underdetermined derivative; no direction discarded')
    _, singular, right = np.linalg.svd(jacobian/norms, full_matrices=False)
    if singular[-1] <= singular[0]*1e-12:
        raise ValueError('numerically rank-deficient derivative; no direction discarded')
    return (right.T/singular)/norms[:, None]


def _normalized_gradient(jacobian, residual):
    norms = np.linalg.norm(jacobian, axis=0)
    if np.any(norms == 0):
        return np.inf
    return float(np.max(np.abs((jacobian/norms).T@residual)))


class _Problem:
    def __init__(self, residual, admissible):
        self.fun = jax.jit(residual)
        self.derivative = jax.jit(jax.jacfwd(residual))
        self.admissible = admissible
        self.evaluations = self.derivative_evaluations = self.hessian_evaluations = 0

        def local_loss(z, reference, directions):
            r = residual(reference+directions@z)
            return .5*jnp.sum(r*r)

        # Differentiate in the well-scaled local coordinates themselves, not
        # by postmultiplying a badly conditioned raw parameter Hessian.
        self.local_derivatives = jax.jit(lambda p, d: (
            jax.grad(local_loss)(jnp.zeros(d.shape[1]), p, d),
            jax.hessian(local_loss)(jnp.zeros(d.shape[1]), p, d)))

    def value(self, p):
        self.evaluations += 1
        r = np.asarray(self.fun(p))
        if r.ndim != 1 or len(r) == 0 or not np.isfinite(r).all():
            raise ValueError('residual must be a finite nonempty vector')
        return r

    def jacobian(self, p):
        self.derivative_evaluations += 1
        j = np.asarray(self.derivative(p))
        if not np.isfinite(j).all():
            raise ValueError('nonfinite derivative')
        return j

    def local(self, p, directions):
        self.hessian_evaluations += 1
        gradient, hessian = map(np.asarray, self.local_derivatives(p, directions))
        if not np.isfinite(gradient).all() or not np.isfinite(hessian).all():
            raise ValueError('nonfinite local loss derivatives')
        return gradient, (hessian+hessian.T)/2

    def result(self, p, *, success, reason, history, **extra):
        residual, jacobian = self.value(p), self.jacobian(p)
        return dict(parameters=np.asarray(p), residual=residual, jacobian=jacobian,
            success=bool(success), reason=reason, history=history,
            normalized_gradient=_normalized_gradient(jacobian, residual),
            evaluations=self.evaluations, derivative_evaluations=self.derivative_evaluations,
            hessian_evaluations=self.hessian_evaluations, **extra)


def _validate(parameters, n_outer, max_iterations, gradient_tolerance, admissible):
    p = np.asarray(parameters, dtype=float).copy()
    if p.ndim != 1 or not np.isfinite(p).all():
        raise ValueError('A finite parameter vector is required.')
    if not isinstance(n_outer, int) or not 0 < n_outer < len(p):
        raise ValueError('n_outer must leave nonempty outer and nuisance blocks.')
    if not isinstance(max_iterations, int) or max_iterations < 0:
        raise ValueError('max_iterations must be a nonnegative integer.')
    if not np.isfinite(gradient_tolerance) or not 0 < gradient_tolerance < 1:
        raise ValueError('gradient_tolerance must be finite and between zero and one.')
    if not admissible(p):
        raise ValueError('The initial parameters are physically inadmissible.')
    return p


def _inner(problem, initial, n_outer, max_iterations, tolerance, max_backtracking=25):
    p = np.asarray(initial).copy()
    history = []
    success, reason = False, 'nuisance iteration budget exhausted'
    for iteration in range(max_iterations+1):
        try:
            residual, jacobian = problem.value(p), problem.jacobian(p)
            jn = jacobian[:, n_outer:]
            factor = _information_factor(jn)
            directions = np.vstack((np.zeros((n_outer, factor.shape[1])), factor))
            gradient, hessian = problem.local(p, directions)
        except ValueError as error:
            reason = str(error)
            break
        eigenvalues = np.linalg.eigvalsh(hessian)
        whitened_gradient = float(np.max(abs(gradient)))
        normalized_gradient = _normalized_gradient(jn, residual)
        row = dict(iteration=iteration, chi_square=float(residual@residual),
            normalized_nuisance_gradient=normalized_gradient,
            whitened_nuisance_gradient=whitened_gradient,
            local_hessian_min=float(eigenvalues[0]), local_hessian_max=float(eigenvalues[-1]))
        history.append(row)
        # A stationary saddle is not a usable minimum for implicit profiling.
        if max(normalized_gradient, whitened_gradient) < tolerance and eigenvalues[0] > 1e-10:
            success, reason = True, 'stationary feasible nuisance minimum'
            break
        if iteration == max_iterations:
            break
        shift = max(0., 1e-6-float(eigenvalues[0]))
        step = -np.linalg.solve(hessian+shift*np.eye(len(gradient)), gradient)
        step /= max(1., np.linalg.norm(step))
        direction = directions@step
        directional = float(gradient@step)
        if not np.isfinite(direction).all() or directional >= 0:
            reason = 'no finite nuisance descent direction'
            break
        row['hessian_step_shift'] = shift
        row['trials'] = []
        accepted = False
        for trial in range(max_backtracking):
            fraction = 2.**(-trial)
            candidate = p+fraction*direction
            feasible = bool(problem.admissible(candidate))
            attempt = dict(fraction=fraction, feasible=feasible)
            row['trials'].append(attempt)
            if not feasible:
                continue
            try:
                trial_residual = problem.value(candidate)
            except ValueError:
                attempt['finite_residual'] = False
                continue
            change = float(trial_residual@trial_residual-residual@residual)
            attempt['chi_square_change'] = change
            if change < 0 and change <= 2e-4*fraction*directional:
                p, accepted = candidate, True
                row['accepted_fraction'] = fraction
                break
        if not accepted:
            reason = 'no resolved feasible nuisance Armijo improvement'
            break
    return problem.result(p, success=success, reason=reason, history=history,
        nuisance_coordinates=len(p)-n_outer,
        nuisance_stationary=success)


def profile_nuisance(residual, parameters, admissible, *, n_outer,
                     max_iterations=30, gradient_tolerance=1e-6):
    """Hold outer coordinates fixed; solve the unchanged nonlinear inner loss.

    ``success`` requires both column-scaled and inverse-information-scaled
    nuisance gradients below tolerance and a positive exact nuisance Hessian.
    A feasible boundary point or unresolved tiny decrease is not success.
    """
    p = _validate(parameters, n_outer, max_iterations, gradient_tolerance, admissible)
    return _inner(_Problem(residual, admissible), p, n_outer, max_iterations, gradient_tolerance)


def _profile_derivatives(problem, inner, n_outer):
    """Exact implicit derivatives, permitted only at a stationary inner fit."""
    if not inner['success'] or not inner['nuisance_stationary']:
        raise ValueError('A stationary nuisance fit is required before profile derivatives.')
    p, residual, jacobian = inner['parameters'], inner['residual'], inner['jacobian']
    jo, jn = jacobian[:, :n_outer], jacobian[:, n_outer:]
    bn = _information_factor(jn)
    qn = jn@bn
    # Orthogonal projection removes source compensation BEFORE local AD.
    # This is a coordinate transformation, not discarded likelihood data.
    projected_outer = jo-qn@(qn.T@jo)
    bo = _information_factor(projected_outer)
    compensation = -bn@(qn.T@(jo@bo))
    outer = np.vstack((bo, compensation))
    nuisance = np.vstack((np.zeros((n_outer, bn.shape[1])), bn))
    directions = np.column_stack((outer, nuisance))
    gradient, hessian = problem.local(p, directions)
    hoo, hon, hnn = hessian[:n_outer, :n_outer], hessian[:n_outer, n_outer:], hessian[n_outer:, n_outer:]
    if np.linalg.eigvalsh(hnn)[0] <= 1e-10:
        raise ValueError('Nuisance Hessian is not positive definite; implicit minimum undefined.')
    response = -np.linalg.solve(hnn, hon.T)
    profile_directions = outer+nuisance@response
    profiled_hessian = hoo+hon@response
    profiled_hessian = (profiled_hessian+profiled_hessian.T)/2
    # Keeping the tiny nuisance-gradient term makes the finite-tolerance
    # chain rule explicit. It vanishes at an exact inner stationary point.
    profiled_gradient = gradient[:n_outer]+response.T@gradient[n_outer:]
    return dict(gradient=profiled_gradient, hessian=profiled_hessian,
        directions=profile_directions, outer_factor=bo,
        information_whitening_error=float(np.linalg.norm((jacobian@directions).T@(jacobian@directions)-np.eye(len(p)))),
        profile_jacobian=np.linalg.solve(bo.T, (jacobian@profile_directions).T).T)


def profiled_least_squares(residual, initial, admissible, *, n_outer,
                          max_iterations=20, max_inner_iterations=30,
                          gradient_tolerance=1e-6, max_profile_trials=12):
    """Local outer Newton fit with exact, stationary nonlinear nuisance profiles.

    Every trial uses the same residual. The exact observed-Hessian Schur
    complement is differentiated in a full-rank information-scaled chart.
    Trial profiles that fail stationarity are retained in history and rejected.
    Success certifies only a local unconstrained stationary minimum; no branch
    exclusion, covariance coverage, or instrument recovery is implied.
    """
    p = _validate(initial, n_outer, max_iterations, gradient_tolerance, admissible)
    if not isinstance(max_inner_iterations, int) or max_inner_iterations < 0:
        raise ValueError('max_inner_iterations must be a nonnegative integer.')
    if not isinstance(max_profile_trials, int) or max_profile_trials < 1:
        raise ValueError('max_profile_trials must be a positive integer.')
    problem = _Problem(residual, admissible)
    inner = _inner(problem, p, n_outer, max_inner_iterations, gradient_tolerance)
    initial_history = inner['history']
    history = []
    success, reason = False, 'outer iteration budget exhausted'
    profile_jacobian = None
    for iteration in range(max_iterations+1):
        p = inner['parameters']
        if not inner['success']:
            reason = 'unaccepted nuisance profile: '+inner['reason']
            break
        try:
            profile = _profile_derivatives(problem, inner, n_outer)
        except ValueError as error:
            reason = str(error)
            break
        gradient, hessian = profile['gradient'], profile['hessian']
        eigenvalues = np.linalg.eigvalsh(hessian)
        optimality = float(np.max(abs(gradient)))
        residual = inner['residual']
        profile_jacobian = profile['profile_jacobian']
        row = dict(iteration=iteration, chi_square=float(residual@residual),
            whitened_profile_gradient=optimality, normalized_gradient=inner['normalized_gradient'],
            nuisance_stationary=True, nuisance_history=inner['history'],
            profile_hessian_min=float(eigenvalues[0]), profile_hessian_max=float(eigenvalues[-1]),
            information_whitening_error=profile['information_whitening_error'], trials=[])
        history.append(row)
        if max(optimality, inner['normalized_gradient']) < gradient_tolerance and eigenvalues[0] > 1e-10:
            success, reason = True, 'stationary feasible profile and full local minimum'
            break
        if iteration == max_iterations:
            break
        shift = max(0., 1e-6-float(eigenvalues[0]))
        local_step = -np.linalg.solve(hessian+shift*np.eye(n_outer), gradient)
        local_step /= max(1., np.linalg.norm(local_step))
        direction = profile['directions']@local_step
        directional = float(gradient@local_step)
        row['hessian_step_shift'] = shift
        if not np.isfinite(direction).all() or directional >= 0:
            reason = 'no finite outer profile descent direction'
            break
        accepted = False
        for trial in range(max_profile_trials):
            fraction = 2.**(-trial)
            candidate = p+fraction*direction
            feasible = bool(admissible(candidate))
            attempt = dict(fraction=fraction, initial_feasible=feasible)
            row['trials'].append(attempt)
            if not feasible:
                continue
            trial_inner = _inner(problem, candidate, n_outer, max_inner_iterations, gradient_tolerance)
            attempt.update(nuisance_stationary=trial_inner['success'],
                           nuisance_reason=trial_inner['reason'], nuisance_history=trial_inner['history'])
            if not trial_inner['success']:
                continue
            trial_r = trial_inner['residual']
            change = float(trial_r@trial_r-residual@residual)
            attempt['chi_square_change'] = change
            if change < 0 and change <= 2e-4*fraction*directional:
                inner, accepted = trial_inner, True
                row['accepted_fraction'] = fraction
                break
        if not accepted:
            reason = 'no stationary feasible improving profile trial'
            break
    return problem.result(inner['parameters'], success=success, reason=reason, history=history,
        initial_nuisance_history=initial_history, nuisance_stationary=inner['success'],
        profile_jacobian=profile_jacobian, n_outer=n_outer,
        priors_added=False, rank_directions_discarded=False)
