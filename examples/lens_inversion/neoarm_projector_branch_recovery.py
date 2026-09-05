"""Response-only recovery using the frozen fresh projector branch challenge.

The old192 and new32 training response coordinates enter once, with their full
measurement covariance. No earlier lens posterior is appended as a prior.
Fresh heldouts are loaded only after the training optimizer results are fixed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time

os.environ.setdefault('JAX_ENABLE_X64', 'true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import jax
import jax.numpy as jnp
import numpy as np

from neoarm_coupled_projector_fit import (
    CoupledProjectorResponse, JointResponseProblem, stage_coordinates, grouped_stage,
    encode_stages, physical_upstream_coordinates, block_diagonal, information,
    PARAMETER_NAMES, PHYSICAL_NAMES,
)
from neoarm_last_lens_physical_fit import RoundLensResponse
from temgym_core.jax_least_squares import least_squares

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT/'examples/generated/neoarm_inversion/imaging_recipe'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_report(path):
    report = json.loads(Path(path).read_text())
    for name, checksum in report['source_hashes'].items():
        if digest(ROOT/name) != checksum:
            raise ValueError(f'Stale recovery dependency: {name}')
    return report


def selected_arrays(arrays, training):
    """Select one split, including off-diagonal response covariance blocks."""
    mask = np.asarray(arrays['training_mask' if training else 'heldout_mask'])
    other = np.asarray(arrays['heldout_mask' if training else 'training_mask'])
    if mask.dtype != bool or other.dtype != bool or mask.shape != other.shape or np.any(mask & other) or not np.all(mask | other):
        raise ValueError('Acquisition masks must be boolean, disjoint and exhaustive.')
    analysers = np.asarray(arrays['pl1_controls'])
    offsets = np.asarray(arrays['il3_controls'])[mask]
    if analysers.shape != (2,):
        raise ValueError('This acquisition has precisely two calibrated analyser states.')
    indices = np.flatnonzero(np.repeat(mask, 16))
    controls = np.array([[p, u] for u in offsets for p in analysers])
    responses = np.asarray(arrays['response'])[mask].reshape(-1, 2, 4)
    covariance = np.asarray(arrays['joint_response_covariance'])[np.ix_(indices, indices)]
    if not all(np.all(np.isfinite(x)) for x in (controls, responses, covariance)):
        raise ValueError('The selected response data must be finite.')
    return controls, responses, covariance


def nominal_preserving_seed(model, base_parameters, q_bias, slope):
    """Change an initializer's IL3 excitation without granting upstream optics."""
    p = np.asarray(base_parameters)
    stages = np.asarray(stage_coordinates(p))
    nominal = np.asarray(grouped_stage(jnp.asarray(stages[4:]), 0.))@np.asarray(model.upstream_matrix(p))
    previous = stages[4:].copy(); previous[2:] = [q_bias, slope]
    prefix = np.linalg.solve(np.asarray(grouped_stage(jnp.asarray(previous), 0.)), nominal)
    return encode_stages(stages[:4], previous, physical_upstream_coordinates(prefix, model.calibration))


def make_starts(model, best, rival):
    seeds = [dict(label='previous_best', parameters=best['parameters']),
             dict(label='previous_interior_rival', parameters=rival['parameters'])]
    for label, base, lower, upper in [('original_interval', best, 0., np.pi),
                                      ('rival_interval', rival, np.pi/2, 3*np.pi/2)]:
        slope = abs(float(base['parameters'][7]))
        for centre in np.linspace(lower+.1*np.pi, upper-.1*np.pi, 5):
            for sign in (-1, 1):
                p = nominal_preserving_seed(model, base['parameters'], centre, sign*slope)
                seeds.append(dict(label=f'{label}_q{centre:.8g}_sign{sign:+d}', parameters=p.tolist()))
    return seeds


def fit_start(problem, seed, old_bounds, *, max_nfev=800):
    initial = np.asarray(seed['parameters'])
    lower, upper = (np.asarray(x).copy() for x in old_bounds)
    # Remove the old artificial interval boundary. Every tested start can move
    # continuously between the original and competing positive-q bias bands.
    lower[6], upper[6] = 0., 3*np.pi/2
    if initial[7] < 0:
        lower[7], upper[7] = -4*np.pi, -.01
    else:
        lower[7], upper[7] = .01, 4*np.pi
    initial = np.clip(initial, lower+1e-12*(upper-lower), upper-1e-12*(upper-lower))
    norms = np.linalg.norm(np.asarray(problem.jacobian(initial)), axis=0)
    if not np.all(np.isfinite(norms) & (norms > 0)):
        return dict(label=seed['label'], optimizer_success=False, status='unresolved_seed_jacobian')
    fit = least_squares(problem.residual, initial, bounds=(lower, upper), x_scale=1/norms,
        max_nfev=max_nfev, max_iter=1800, ftol=1e-11, xtol=1e-11, gtol=1e-7)
    parameters = np.asarray(fit.x)
    jacobian = np.asarray(fit.jacobian)
    normalized_gradient = (jacobian.T@np.asarray(fit.residual))/np.linalg.norm(jacobian, axis=0)
    return dict(label=seed['label'], initial=initial.tolist(), parameters=parameters.tolist(),
        bounds=[lower.tolist(), upper.tolist()], optimizer_success=bool(fit.success),
        status='tested_training_fit' if fit.success else 'unresolved_training_optimizer',
        termination=fit.status, nfev=fit.nfev, chi_square=float(2*fit.cost),
        physical_coordinates=np.asarray(stage_coordinates(parameters)).tolist(),
        active_bounds=np.flatnonzero(np.minimum(abs(parameters-lower), abs(parameters-upper)) < 1e-7).tolist(),
        initial_IL3_q_bias=float(initial[6]), final_IL3_q_bias=float(parameters[6]),
        crossed_original_pi_boundary=bool((initial[6]-np.pi)*(parameters[6]-np.pi) < 0),
        normalized_gradient_max=float(np.max(abs(normalized_gradient))), all_thirteen_coordinates_released=True)


def predictive_diagnostic(model, parameters, parameter_covariance, data):
    controls, responses, covariance = data
    prediction = np.asarray(model.predict(parameters, controls))
    delta = (prediction-responses).reshape(-1)
    white = np.linalg.solve(np.linalg.cholesky(covariance), delta)
    result = dict(controls=controls.tolist(), response=responses.tolist(), response_covariance=covariance.tolist(),
        prediction=prediction.tolist(), chi_square=float(white@white), response_coordinate_count=len(white),
        rms_response_noise_sigma=float(np.sqrt(np.mean(white**2))),
        maximum_absolute_response_error_pixels=float(np.max(abs(delta))))
    if parameter_covariance is not None:
        derivative = np.asarray(jax.jacfwd(lambda p:model.predict(p, controls).reshape(-1))(jnp.asarray(parameters)))
        predictive = covariance+derivative@np.asarray(parameter_covariance)@derivative.T
        w = np.linalg.solve(np.linalg.cholesky(predictive), delta)
        result.update(predictive_chi_square=float(w@w), predictive_rms_sigma=float(np.sqrt(np.mean(w**2))))
    return result


def run(output, *, max_nfev=800):
    if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
        raise RuntimeError('Use GPU float64 for the new response forward/fit.')
    tick = time.perf_counter()
    old_path, boundary_path = FOLDER/'coupled_projector_fit.json', FOLDER/'coupled_projector_branch_boundary.json'
    new_path = FOLDER/'projector_branch_experiment_inference.json'
    old, boundary, new = map(verified_report, (old_path, boundary_path, new_path))
    if not new['paired_data_admitted'] or not new['calibration_response_noise_independent']:
        raise ValueError('The fresh image block must pass admission with independent noise.')
    arrays_path = ROOT/new['arrays_path']
    if digest(arrays_path) != new['arrays_sha256']:
        raise ValueError('Fresh inference NPZ checksum mismatch.')
    # No fresh heldout response is returned, retained or used before fitting.
    with np.load(arrays_path, allow_pickle=False) as arrays:
        fresh_training = selected_arrays(arrays, True)
    calibration = RoundLensResponse(tuple(old['calibrated_input_steps_grouped']), old['detector_pixel_m'])
    if list(calibration.input_steps_grouped) != new['input_steps_grouped'] or calibration.pixel_m != new['detector_pixel_m']:
        raise ValueError('Old and new acquisition calibration conventions differ.')
    model = CoupledProjectorResponse(calibration)
    controls = np.concatenate((np.asarray(old['training_controls']), fresh_training[0]))
    responses = np.concatenate((np.asarray(old['training_response']), fresh_training[1]))
    covariance = block_diagonal([np.asarray(old['training_response_covariance']), fresh_training[2]])
    problem = JointResponseProblem(model, controls, responses, covariance)
    rival = min((row for row in boundary['rows'] if row['optimizer_success'] and not row['active_bounds']),
        key=lambda row:row['chi_square'])
    seeds = make_starts(model, old['best'], rival)
    sources = dict(old['source_hashes']); sources.update(boundary['source_hashes']); sources.update(new['source_hashes'])
    for path in (Path(__file__), old_path, boundary_path, new_path, arrays_path):
        sources[str(path.relative_to(ROOT))] = digest(path)
    rows = []
    for index, seed in enumerate(seeds):
        row = fit_start(problem, seed, old['best']['bounds'], max_nfev=max_nfev)
        row['start_index'] = index; rows.append(row)
        print('fresh recovery', index, row['label'], row['status'], row.get('chi_square'), row.get('nfev'), flush=True)
    converged = [row for row in rows if row['optimizer_success']]
    best = min(converged, key=lambda row:row['chi_square']) if converged else None
    info, fresh_hold, old_hold, truth = {}, {}, {}, {}
    if best is not None:
        parameters = np.asarray(best['parameters'])
        info = information(problem, parameters)
        # This is the FIRST loading of fresh heldout response values for use.
        # The training winner above is fixed, independent of these values.
        with np.load(arrays_path, allow_pickle=False) as arrays:
            fresh_heldout_data = selected_arrays(arrays, False)
        fresh_hold = predictive_diagnostic(model, parameters, info.get('parameter_covariance'), fresh_heldout_data)
        fresh_hold.update(loaded_for_prediction_only_after_all_training_fits=True,
            used_for_initialization_or_training_selection=False, previous_model_selection_evidence=False)
        prior = old['heldout']
        old_hold = predictive_diagnostic(model, parameters, info.get('parameter_covariance'),
            (np.asarray(prior['controls']), np.asarray(prior['response']), np.asarray(prior['response_covariance'])))
        old_hold.update(previous_model_selection_evidence=True, independent_fresh_validation=False,
            used_in_this_training_likelihood=False)
        # Numerically access generating optics only after fitting and fresh
        # validation. Earlier reports' embedded truth diagnostics are unused.
        truth_path = ROOT/new['truth_path']
        if digest(truth_path) != new['truth_sha256']:
            raise ValueError('Postfit generating truth checksum mismatch.')
        generating = json.loads(truth_path.read_text())
        last, previous = generating['PL1'], generating['IL3']
        p1 = [np.log(last['length_m']), np.log(last['detector_gap_m']), last['q_bias'], last['q_slope_per_control']]
        p3 = [np.log(previous['length_m']), np.log(previous['gap_after_exit_m']), previous['q_bias'], previous['q_slope_per_control']]
        prefix = physical_upstream_coordinates(generating['upstream_to_IL3_mechanical'], calibration)
        generating_parameters = encode_stages(p1, p3, prefix)
        error = np.asarray(stage_coordinates(parameters)-stage_coordinates(generating_parameters))
        truth_residual = np.asarray(problem.residual(generating_parameters))
        truth = dict(path=new['truth_path'], sha256=new['truth_sha256'], accessed_for_numerical_diagnostics_only_after_fit=True,
            physical_errors=error.tolist(), lengths_gaps_errors_percent=(100*np.expm1(error[[0,1,4,5]])).tolist(),
            training_chi_square_at_truth=float(truth_residual@truth_residual), training_coordinate_count=len(truth_residual),
            sample_kappa_truth_per_m=float(generating_parameters[12]/calibration.balance_length_m),
            sample_kappa_fitted_per_m=float(parameters[12]/calibration.balance_length_m))
        if info.get('physical_covariance') is not None:
            sigma = np.sqrt(np.diag(np.asarray(info['physical_covariance'])))
            truth.update(physical_errors_sigma=(error/sigma).tolist(),
                lengths_gaps_standard_errors_percent=(100*sigma[[0,1,4,5]]).tolist())
        sources[new['truth_path']] = digest(truth_path)
    if any(digest(ROOT/name) != checksum for name, checksum in sources.items()):
        raise RuntimeError('A pinned inference dependency changed during recovery.')
    report = dict(schema='neoarm-fresh-projector-branch-recovery-v1', source_hashes=sources,
        source_hashes_unchanged_during_execution=True, backend=jax.default_backend(), x64_enabled=jax.config.x64_enabled,
        elapsed_seconds=time.perf_counter()-tick, jax_version=jax.__version__, parameter_names=PARAMETER_NAMES,
        physical_coordinate_names=PHYSICAL_NAMES, old_training_report=str(old_path.relative_to(ROOT)),
        fresh_inference_report=str(new_path.relative_to(ROOT)), fresh_arrays_path=new['arrays_path'], fresh_arrays_sha256=new['arrays_sha256'],
        training_controls=controls.tolist(), training_response=responses.tolist(), training_response_covariance=covariance.tolist(),
        original_training_coordinate_count=len(old['training_response'])*8, new_training_coordinate_count=fresh_training[1].size,
        training_response_coordinate_count=responses.size, all_thirteen_coordinates_released=True,
        calibration_posterior_added_as_prior=False, original_measurements_duplicated=False,
        continuous_IL3_q_bias_bounds_rad=[0., 3*np.pi/2], starts=seeds, rows=rows, best=best,
        start_count=len(rows), converged_start_count=len(converged), unresolved_start_count=len(rows)-len(converged),
        information=info, fresh_heldout=fresh_hold, reused_old_heldout=old_hold, truth_diagnostic=truth,
        fresh_heldout_used_for_initialization_or_training_selection=False, global_uniqueness_proved=False,
        limitations=['The data are conditional synthetic finite-pixel image responses, not an instrument measurement or raw-pixel optical likelihood.',
            'The continuous bias box spans both previously tested positive-bias intervals, not every possible excitation branch.',
            'Both slope signs and22 finite starts are tested; unsuccessful starts remain unresolved.',
            'No previous lens posterior is a prior: original192 and fresh32 response coordinates occur exactly once.',
            'The original conditional geometry cap, negative-first PL1 branch and exact sample/detector calibration assumptions remain.',
            'Fresh heldouts are evaluated only after training is fixed. Older heldouts are explicitly reused model-selection evidence.',
            'The unknown round upstream map and sample magnetic shear remain free; source-shape information is not included.',
            'Shared input-scale uncertainty, drift, non-hard-edge fields and the rest of the column require separate validation.'])
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-nfev', type=int, default=800)
    args = parser.parse_args()
    result = run(args.output, max_nfev=args.max_nfev)
    print(json.dumps({key:result[key] for key in ('start_count','converged_start_count','truth_diagnostic')}, indent=2), flush=True)
