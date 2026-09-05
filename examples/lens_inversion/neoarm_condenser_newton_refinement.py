"""Bounded training-only exact-Hessian refinement of a saved condenser iterate.

The Hessian is differentiated directly in p+Bz coordinates, where columns of
J(p)B are orthonormal. It is not formed by multiplying an ill-conditioned raw
parameter Hessian. Every optical/source/calibration coordinate remains free.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('JAX_ENABLE_X64', 'true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import jax
if __name__ == '__main__':
    jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from neoarm_condenser_extension_recovery import selected_data, beam_residual
from neoarm_condenser_extension_design import load_context, training_residual, covariance_factor, FOLDER
from neoarm_condenser_covariance_fit import VisibleSourceModel, quantum_completion, geometry_gaps
from neoarm_aperture_capture import ROOT


def refine_training(residual, initial, admissible, *, max_iterations=20, gradient_tolerance=1e-6):
    """Short local Newton diagnostic; no truth, heldouts, priors or rank truncation."""
    if max_iterations < 1 or not 0 < gradient_tolerance < 1:
        raise ValueError('Positive bounded iterations and gradient tolerance are required.')
    p = np.asarray(initial, dtype=float).copy()
    if not admissible(p):
        raise ValueError('The saved starting point is outside the physical domain.')
    function = jax.jit(residual)
    derivative = jax.jit(jax.jacfwd(residual))
    def local_loss(z, reference, factor):
        r = residual(reference+factor@z)
        return .5*jnp.sum(r*r)
    local_derivatives = jax.jit(lambda reference, factor: (
        jax.grad(local_loss)(jnp.zeros(len(p)), reference, factor),
        jax.hessian(local_loss)(jnp.zeros(len(p)), reference, factor)))
    history, success = [], False
    reason = 'local Newton iteration limit'
    for iteration in range(max_iterations+1):
        r = np.asarray(function(p)); jac = np.asarray(derivative(p))
        norms = np.linalg.norm(jac, axis=0)
        factor = covariance_factor(jac)
        gradient, hessian = map(np.asarray, local_derivatives(p, factor))
        asymmetry = np.linalg.norm(hessian-hessian.T)/max(np.linalg.norm(hessian), 1e-300)
        hessian = (hessian+hessian.T)/2
        eigenvalues = np.linalg.eigvalsh(hessian)
        normalized_gradient = float(np.max(abs((jac/norms).T@r)))
        if not np.isfinite(gradient).all() or not np.isfinite(hessian).all():
            reason = 'nonfinite exact derivatives'; break
        shift = max(0., 1e-6-float(eigenvalues[0]))
        local_step = -np.linalg.solve(hessian+shift*np.eye(len(p)), gradient)
        decrement = float(np.sqrt(max(0., -gradient@local_step)))
        record = dict(iteration=iteration, chi_square=float(r@r),
            normalized_gradient=normalized_gradient, local_gradient=gradient.tolist(),
            local_hessian_eigenvalues=eigenvalues.tolist(), hessian_relative_asymmetry=float(asymmetry),
            information_whitening_error=float(np.linalg.norm((jac@factor).T@(jac@factor)-np.eye(len(p)))),
            Newton_decrement=decrement, Hessian_diagonal_shift=shift,
            local_step_norm=float(np.linalg.norm(local_step)), line_search=[])
        history.append(record)
        if normalized_gradient < gradient_tolerance and decrement < gradient_tolerance:
            success, reason = True, 'gradient and Newton-decrement tolerances'; break
        if iteration == max_iterations:
            break
        # One unit is a local Gauss--Newton standard error, not an arbitrary
        # physical parameter scale. Limiting it is a step safeguard, not a prior.
        local_step /= max(1., np.linalg.norm(local_step))
        direction = factor@local_step
        directional = float(gradient@local_step)
        if directional >= 0 or not np.isfinite(direction).all():
            reason = 'no finite Newton descent direction'; break
        accepted = False
        for power in range(35):
            alpha = 2.**(-power)
            candidate = p+alpha*direction
            feasible = admissible(candidate)
            trial_record = dict(step_fraction=alpha, feasible=bool(feasible))
            if feasible:
                trial = np.asarray(function(candidate))
                change = float(trial@trial-r@r)
                trial_record.update(chi_square_change=change, armijo_required_change=2e-4*alpha*directional)
                if np.isfinite(trial).all() and change < 0 and change <= 2e-4*alpha*directional:
                    p, accepted = candidate, True
            record['line_search'].append(trial_record)
            if accepted:
                record['accepted_step_fraction'] = alpha
                break
        if not accepted:
            reason = 'no resolved feasible Armijo improvement'; break
    r = np.asarray(function(p)); jac = np.asarray(derivative(p))
    return dict(parameters=p, residual=r, jacobian=jac, success=success, reason=reason,
                history=history, final_information_factor=covariance_factor(jac))


def audit(output, *, max_iterations=20):
    if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
        raise RuntimeError('Use GPU float64 for the training refinement.')
    source_path = FOLDER/'condenser_guarded_recovery.json'
    original = json.loads(source_path.read_text())
    row = next(r for r in original['rows'] if r['initializer'] == 'rough_geometry_data_source')
    old, arrays, repair, aperture, model = load_context()
    quotient = VisibleSourceModel(model)
    array_path = ROOT/original['arrays_path']
    with np.load(array_path, allow_pickle=False) as data:
        fresh = selected_data(data, True)
    old_loss = training_residual(quotient, arrays, aperture)
    new_loss = beam_residual(quotient, fresh)
    residual = lambda p: jnp.r_[old_loss(p), new_loss(p)]
    initial_residual = np.asarray(jax.jit(residual)(np.asarray(row['parameters'])))
    if not np.isclose(initial_residual@initial_residual, row['total_chi_square'], rtol=1e-12, atol=1e-8):
        raise RuntimeError('The initial training likelihood differs from the saved checkpoint.')
    fit = refine_training(residual, row['parameters'], quotient.admissible, max_iterations=max_iterations)
    p = fit['parameters']; optical = np.asarray(quotient.decode(p)[0])
    completion = quantum_completion(p[12:21], interval=True)
    factor = fit['final_information_factor']
    physical_factor = np.asarray(jax.jacfwd(lambda q: quotient.decode(q)[0])(p))@factor
    source_hashes = dict(original['source_hashes'])
    for path in (Path(__file__), ROOT/'tests/test_neoarm_condenser_newton_refinement.py', source_path, array_path):
        source_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = dict(schema='neoarm-condenser-training-Newton-refinement-v1', backend=jax.default_backend(), x64_enabled=True,
        source_hashes=source_hashes, source_report=str(source_path.relative_to(ROOT)),
        initializer='Saved rough-geometry, data-only source training iterate; no truth refinement.',
        initial_parameters=row['parameters'], parameters=p.tolist(),
        success=fit['success'], termination=fit['reason'], history=fit['history'],
        training_total_chi_square=float(fit['residual']@fit['residual']),
        training_data_chi_square=float(fit['residual']@fit['residual']-(p[22]/.005)**2),
        common_radius_prior_count=1, training_data_coordinate_count=original['training_data_coordinate_count'],
        optical_parameters=optical.tolist(), geometry_gaps_m=geometry_gaps(model, optical).tolist(),
        quantum_margin=completion['margin'], invisible_source_covariance_interval=completion['feasible_interval'],
        gauss_newton_parameter_covariance=(factor@factor.T).tolist(),
        gauss_newton_optical_covariance=(physical_factor@physical_factor.T).tolist(),
        covariance_scope='Expected-information local diagnostics, not nonlinear profile intervals; exact observed Hessian spectrum is recorded separately.',
        all_twelve_optical_coordinates_free=True, visible_source_coordinates_free=9,
        heldout_observations_used=False, generating_truth_values_used=False,
        additional_likelihood_priors_added=False, rank_directions_discarded=False,
        complete_microscope_recovery=False, global_uniqueness_proved=False,
        limitations=['This is one bounded local refinement, not a new initialization or branch campaign.',
            'Downstream/objective, source preparation, aperture plane and radius metrology retain their existing conditional contracts.',
            'Fresh and previously inspected heldout observations are untouched by this driver.',
            'A stationary training point still requires predictive, branch and physical-uncertainty audits.'])
    for name, checksum in source_hashes.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != checksum:
            raise RuntimeError(f'A bound source changed: {name}')
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-iterations', type=int, default=20)
    args = parser.parse_args(); jax.config.update('jax_enable_x64', True)
    result = audit(args.output, max_iterations=args.max_iterations)
    print(result['success'], result['termination'], result['training_total_chi_square'],
          result['history'][-1]['normalized_gradient'], flush=True)
