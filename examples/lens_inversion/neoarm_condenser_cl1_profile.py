"""Four fixed-CL1-centre nonlinear profiles of the frozen training likelihood.

Each target uses the covariance of the stationary training fit only. CL1's
centre is fixed exactly, while five other geometry coordinates and seventeen
excitation/source/calibration nuisances are reoptimized. No held-out response
or generating optical/source value is used, and no local profile excludes
other excitation branches or proves full-column identifiability.
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

from temgym_core.profiled_least_squares import profiled_least_squares
from neoarm_condenser_extension_recovery import selected_data, beam_residual
from neoarm_condenser_extension_design import load_context, training_residual, covariance_factor, FOLDER
from neoarm_condenser_covariance_fit import VisibleSourceModel, quantum_completion, geometry_gaps, SOURCE_SCALE
from neoarm_condenser_reference_source import ReferenceSourceChart
from neoarm_shared_objective_audit import provenance_snapshot
from neoarm_aperture_capture import ROOT


FREE_GEOMETRY_INDICES = (1, 4, 5, 8, 9)
NUISANCE_INDICES = (2, 3, 6, 7, 10, 11, *range(12, 23))
FREE_ORDER = np.asarray(FREE_GEOMETRY_INDICES+NUISANCE_INDICES)


def pack_free(reference_parameters):
    return jnp.asarray(reference_parameters)[jnp.asarray(FREE_ORDER)]


def unpack_free(free_parameters, fixed_centre):
    """Expand all22 free coordinates without differentiating the fixed centre."""
    free = jnp.asarray(free_parameters)
    if free.shape != (22,):
        raise ValueError('Exactly22 free coordinates are required.')
    return jnp.zeros(23, dtype=free.dtype).at[0].set(fixed_centre).at[jnp.asarray(FREE_ORDER)].set(free)


def conditional_reference_seed(reference_parameters, covariance, target, admissible,
                               *, max_backtracking=20):
    """Backtrack only the free conditional displacement, never the target."""
    p, covariance = np.asarray(reference_parameters), np.asarray(covariance)
    if p.shape != (23,) or covariance.shape != (23, 23) or not np.isfinite(covariance).all():
        raise ValueError('Finite full reference parameters and covariance are required.')
    if not np.isfinite(target) or covariance[0, 0] <= 0:
        raise ValueError('Finite centre and positive centre variance are required.')
    conditional = covariance[:, 0]/covariance[0, 0]*(target-p[0])
    trials = []
    for trial in range(max_backtracking):
        fraction = 2.**(-trial)
        candidate = p+fraction*conditional
        candidate[0] = target
        feasible = bool(admissible(candidate))
        trials.append(dict(free_displacement_fraction=fraction, fixed_centre=float(candidate[0]), feasible=feasible))
        if feasible:
            return candidate, trials
    return None, trials


def source_diagnostic(parameters):
    completion = quantum_completion(np.asarray(parameters)[12:21], interval=True)
    if not completion['feasible']:
        return dict(quantum_feasible=False, balanced_Hermitian_margin=float(completion['margin']))
    covariance = np.asarray(completion['representative_covariance'])
    balanced = covariance/np.outer(SOURCE_SCALE, SOURCE_SCALE)
    root = np.linalg.cholesky(balanced)
    omega = np.kron(np.eye(2), np.array([[0., 1.], [-1., 0.]]))
    frequencies = np.linalg.eigvalsh(1j*root.T@omega@root)[2:]
    return dict(quantum_feasible=True, balanced_Hermitian_margin=float(completion['margin']),
        representative_emittances_over_quantum_minimum=frequencies.tolist(),
        representative_covariance_m_rad=covariance.tolist(),
        invisible_covariance_interval=list(map(float, completion['feasible_interval'])),
        source_rms_x_alpha_y_alpha=np.sqrt(np.diag(covariance)).tolist(),
        covariance_representative_not_independently_identified=True,
        Hermitian_margin_is_coordinate_dependent=True)


def run(output, *, max_outer_iterations=20, max_inner_iterations=30):
    if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
        raise RuntimeError('Profile fitting requires GPU float64.')
    if not 0 <= max_outer_iterations <= 20 or not 0 <= max_inner_iterations <= 30:
        raise ValueError('This experiment permits at most20 outer and30 inner iterations.')
    stationary_path = FOLDER/'condenser_geometry_profile_diagnostic.json'
    stationary = json.loads(stationary_path.read_text())
    best = stationary['continuation']
    if not best['original_stationary_success']:
        raise ValueError('A stationary full training fit is required.')
    for name, checksum in stationary['source_hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != checksum:
            raise ValueError(f'The stationary checkpoint dependency changed: {name}')
    p = np.asarray(best['original_parameters'])
    old, arrays, _, aperture, model = load_context()
    quotient = VisibleSourceModel(model)
    chart = ReferenceSourceChart(quotient, stationary['reference_control'], stationary['reference_width_m'])
    previous_profile = FOLDER/'condenser_profiled_recovery.json'
    previous = json.loads(previous_profile.read_text())
    array_path = ROOT/previous['arrays_path']
    with np.load(array_path, allow_pickle=False) as archive:
        fresh = selected_data(archive, True)
    old_loss = training_residual(quotient, arrays, aperture)
    fresh_loss = beam_residual(quotient, fresh)
    residual = lambda q: jnp.r_[old_loss(q), fresh_loss(q)]
    function = jax.jit(residual)
    r = np.asarray(function(p))
    if not np.isclose(r@r, best['chi_square'], atol=1e-8, rtol=1e-12):
        raise RuntimeError('The stationary training likelihood did not close.')
    jacobian = np.asarray(jax.jit(jax.jacfwd(residual))(p))
    original_factor = covariance_factor(jacobian)
    reference_p = np.asarray(chart.encode(p))
    decode_derivative = np.asarray(jax.jacfwd(chart.decode)(reference_p))
    reference_factor = np.linalg.solve(decode_derivative, original_factor)
    reference_covariance = reference_factor@reference_factor.T
    sigma = float(np.sqrt(reference_covariance[0, 0]))
    sigma_m = float(model.scales[0]*sigma)
    baseline_centre_m = float(quotient.decode(p)[0][0])
    hashes = dict(stationary['source_hashes'])
    loaded, _ = provenance_snapshot(); hashes.update(loaded)
    for path in (stationary_path, Path(__file__), ROOT/'tests/test_neoarm_condenser_cl1_profile.py', previous_profile, array_path):
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = []
    for offset in (-2., -1., 1., 2.):
        target = float(reference_p[0]+offset*sigma)
        target_m = float(model.base[0]+model.scales[0]*target)
        reference_domain = lambda q: quotient.admissible(np.asarray(chart.decode(q)))
        seed, seed_trials = conditional_reference_seed(reference_p, reference_covariance, target, reference_domain)
        row = dict(offset_in_local_GN_sigma=offset, fixed_centre_coordinate=target,
            fixed_CL1_centre_m=target_m, seed_trials=seed_trials,
            seed_source='Stationary training fit and reference-chart conditional Gaussian covariance only.')
        rows.append(row)
        if seed is None:
            row.update(accepted=False, reason='No feasible fixed-target conditional seed', optimizer_run=False)
            print(offset, False, row['reason'], flush=True)
            continue
        decode = lambda q: chart.decode(unpack_free(q, target))
        free_residual = lambda q: residual(decode(q))
        free_domain = lambda q: quotient.admissible(np.asarray(decode(q)))
        fit = profiled_least_squares(free_residual, np.asarray(pack_free(seed)), free_domain,
            n_outer=5, max_iterations=max_outer_iterations, max_inner_iterations=max_inner_iterations)
        original = np.asarray(decode(fit['parameters']))
        rf = np.asarray(function(original))
        jf = np.asarray(jax.jit(jax.jacfwd(residual))(original))
        free_jacobian = jf[:, np.arange(1, 23)]
        free_gradient = float(np.max(abs((free_jacobian/np.linalg.norm(free_jacobian, axis=0)).T@rf)))
        closure = float(np.max(abs(rf-fit['residual'])))
        fixed_exactly = bool(original[0] == target)
        accepted = bool(fit['success'] and free_gradient < 1e-6 and closure < 1e-5 and fixed_exactly and quotient.admissible(original))
        optical = np.asarray(quotient.decode(original)[0])
        row.update(accepted=accepted, solver_success=fit['success'], reason=fit['reason'], optimizer_run=True,
            initial_original_parameters=np.asarray(chart.decode(seed)).tolist(),
            original_parameters=original.tolist(), free_reference_parameters=fit['parameters'].tolist(),
            fixed_centre_exactly_preserved=fixed_exactly, residual_closure_sigma=closure,
            original_free_coordinate_normalized_gradient=free_gradient,
            training_chi_square=float(rf@rf), delta_training_chi_square=float(rf@rf-r@r),
            training_data_chi_square=float(rf@rf-(original[22]/.005)**2),
            common_radius_prior_chi_square=float((original[22]/.005)**2),
            history=fit['history'], initial_nuisance_history=fit['initial_nuisance_history'],
            residual_evaluations=fit['evaluations'], derivative_evaluations=fit['derivative_evaluations'],
            hessian_evaluations=fit['hessian_evaluations'], source=source_diagnostic(original),
            geometry_gaps_m=geometry_gaps(model, optical).tolist(), optical_parameters=optical.tolist())
        print(offset, accepted, fit['reason'], row['delta_training_chi_square'], free_gradient, flush=True)
    for name, checksum in hashes.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != checksum:
            raise RuntimeError(f'A bound computational input changed: {name}')
    report = dict(schema='neoarm-condenser-CL1-centre-four-point-profile-v1', backend=jax.default_backend(),
        x64_enabled=True, source_hashes=hashes, stationary_report=str(stationary_path.relative_to(ROOT)),
        baseline_parameters=p.tolist(), baseline_training_chi_square=float(r@r),
        baseline_CL1_centre_m=baseline_centre_m, local_GN_CL1_centre_sigma_m=sigma_m,
        reference_control=np.asarray(chart.reference_control).tolist(), reference_width_m=chart.reference_width_m,
        original_GN_covariance=(original_factor@original_factor.T).tolist(),
        reference_GN_covariance=reference_covariance.tolist(), rows=rows,
        all_four_profiles_stationary=all(row['accepted'] for row in rows),
        training_data_coordinate_count=102, common_radius_prior_count=1,
        fixed_optical_coordinate_count=1, free_optical_coordinate_count=11,
        free_geometry_count=5, profiled_excitation_count=6, profiled_source_count=9,
        profiled_additional_calibration_count=2, free_parameter_count_per_profile=22,
        max_outer_iterations=max_outer_iterations, max_inner_iterations=max_inner_iterations,
        generating_values_used=False, heldout_observations_used=False, likelihood_changed=False,
        prior_added=False, global_branch_exclusion=False, confidence_interval_coverage_verified=False,
        limitations=['Each fixed coordinate is one conditional local profile, not a global branch search.',
            'The four target offsets are prescribed from the training-only local covariance, not selected from held-out performance.',
            'Failed points remain unresolved; an unconverged objective is only an upper bound on the conditional minimum.',
            'The source OAM completion remains unobserved and its representative is not independently identified.',
            'The existing conditional objective/downstream, source-preparation and aperture contracts remain.'])
    Path(output).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-outer-iterations', type=int, default=20)
    parser.add_argument('--max-inner-iterations', type=int, default=30)
    args = parser.parse_args()
    run(args.output, max_outer_iterations=args.max_outer_iterations, max_inner_iterations=args.max_inner_iterations)
