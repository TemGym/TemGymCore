"""Check whether an artificial pi-wide IL3 search boundary creates a rival.

Releases the former q3=pi boundary within [pi/2,3pi/2], with all thirteen
coordinates free and the same original 192 training responses. This tests
particular saved starting solutions, not all possible branches or global
uniqueness. Held-out responses are evaluated only after every refit finishes.
"""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from neoarm_coupled_projector_fit import CoupledProjectorResponse, JointResponseProblem, stage_coordinates
from neoarm_last_lens_physical_fit import ROOT, RoundLensResponse
from temgym_core.jax_least_squares import least_squares


def shifted_bounds(saved_row, interval=(np.pi/2, 3*np.pi/2)):
    """Change only the IL3-bias search interval; retain every other bound."""
    initial = np.asarray(saved_row['parameters'], dtype=float)
    lower, upper = np.asarray(saved_row['bounds'], dtype=float).copy()
    interval = np.asarray(interval)
    if initial.shape != (13,) or lower.shape != (13,) or upper.shape != (13,):
        raise ValueError('A thirteen-coordinate fit and its original bounds are required.')
    if interval.shape != (2,) or not np.all(np.isfinite(interval)) or interval[0] >= interval[1]:
        raise ValueError('Choose an increasing finite IL3-bias interval.')
    lower[6], upper[6] = interval
    if not np.all(np.isfinite(initial)) or np.any(initial < lower) or np.any(initial > upper):
        raise ValueError('The saved seed is outside the shifted interval; it is not clipped.')
    return initial, lower, upper


def run_check(input_path, output_path, *, max_nfev=1200):
    if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
        raise RuntimeError('Run the boundary check on GPU float64.')
    started = perf_counter()
    input_path = Path(input_path).resolve()
    previous = json.loads(input_path.read_text())
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    hashes = dict(previous['source_hashes'])
    for name, checksum in hashes.items():
        if digest(ROOT/name) != checksum:
            raise ValueError(f'Stale coupled response evidence: {name}')
    for source in (input_path, Path(__file__), Path(__file__).with_name('neoarm_coupled_projector_fit.py')):
        hashes[str(source.relative_to(ROOT))] = digest(source)
    model = CoupledProjectorResponse(RoundLensResponse(tuple(previous['calibrated_input_steps_grouped']), previous['detector_pixel_m']))
    problem = JointResponseProblem(model, previous['training_controls'], previous['training_response'], previous['training_response_covariance'])
    interval = (np.pi/2, 3*np.pi/2)
    seeds = [row for row in previous['rows'] if 'parameters' in row and interval[0] <= row['parameters'][6] <= interval[1]]
    rows = []
    for seed in seeds:
        initial, lower, upper = shifted_bounds(seed, interval)
        norms = np.linalg.norm(np.asarray(problem.jacobian(initial)), axis=0)
        fit = least_squares(problem.residual, initial, bounds=(lower, upper), x_scale=1/norms,
            max_nfev=max_nfev, max_iter=2000, ftol=1e-11, xtol=1e-11, gtol=1e-7)
        p = np.asarray(fit.x)
        physical = np.asarray(stage_coordinates(p))
        row = dict(saved_start_index=seed['start_index'], saved_optimizer_success=seed['optimizer_success'],
            saved_chi_square=seed['chi_square'], initial=initial.tolist(), bounds=[lower.tolist(),upper.tolist()],
            parameters=p.tolist(), physical_coordinates=physical.tolist(),
            optimizer_success=bool(fit.success), termination=fit.status, nfev=fit.nfev,
            chi_square=float(2*fit.cost), delta_chi_square_vs_original_best=float(2*fit.cost-previous['best']['chi_square']),
            active_bounds=np.flatnonzero(np.minimum(abs(p-lower),abs(p-upper))<1e-7).tolist(),
            q3_at_original_pi_boundary=bool(abs(p[6]-np.pi)<1e-7),
            q3_at_shifted_boundary=bool(min(abs(p[6]-interval[0]),abs(p[6]-interval[1]))<1e-7),
            all_thirteen_coordinates_free=True,
            status='tested_shifted_interval_fit' if fit.success else 'unresolved_shifted_interval_optimizer')
        rows.append(row)
        print('shifted start',seed['start_index'],row['status'],row['chi_square'], 'q3',p[6], 'bounds',row['active_bounds'],flush=True)
    # No held-out values have influenced compatibility, seeding or optimization.
    hold = previous['heldout']
    hold_controls, hold_data = np.asarray(hold['controls']), np.asarray(hold['response'])
    chol = np.linalg.cholesky(np.asarray(hold['response_covariance']))
    prediction = jax.jit(lambda p:model.predict(p,jnp.asarray(hold_controls)))
    for row in rows:
        expected = np.asarray(prediction(jnp.asarray(row['parameters'])))
        difference = (expected-hold_data).reshape(-1)
        white = np.linalg.solve(chol,difference)
        row['heldout'] = dict(chi_square=float(white@white), response_coordinate_count=len(white),
            maximum_absolute_response_error_pixels=float(np.max(np.abs(difference))),
            predicted_response=expected.tolist(), used_for_refitting=False,
            likelihood_is_fixed_solution_not_profiled=True)
    if any(digest(ROOT/name)!=checksum for name,checksum in hashes.items()):
        raise RuntimeError('A boundary-check dependency changed during execution.')
    report = dict(schema='neoarm-coupled-IL3-shifted-boundary-check-v1', source_hashes=hashes,
        source_hashes_unchanged_during_execution=True, input_path=str(input_path.relative_to(ROOT)), input_sha256=digest(input_path),
        backend=jax.default_backend(), x64_enabled=True, jax_version=jax.__version__, elapsed_seconds=perf_counter()-started,
        original_IL3_bias_interval=[np.pi,2*np.pi], shifted_IL3_bias_interval=list(interval),
        unchanged_conditional_combined_span_bounds_m=previous['conditional_combined_span_bounds_m'],
        unchanged_training_response_coordinate_count=previous['training_response_coordinate_count'],
        original_best_training_chi_square=previous['best']['chi_square'],
        original_best_heldout_chi_square=hold['chi_square'], original_best_heldout_predictive_chi_square=hold['predictive_chi_square'],
        compatible_saved_start_indices=[row['start_index'] for row in seeds], rows=rows,
        start_count=len(rows), converged_start_count=sum(row['optimizer_success'] for row in rows),
        unresolved_start_count=sum(not row['optimizer_success'] for row in rows),
        all_thirteen_coordinates_free=True, added_calibration_prior=False,
        generating_truth_used_for_initialization=False, heldout_used_for_initialization_or_refitting=False,
        global_uniqueness_proved=False, interpretation=[
            'All saved seeds already inside the shifted interval are tested, including compatible failed starts.',
            'A minimum on the new boundary is not an unconstrained stationary competing instrument.',
            'Only specific fitted solutions are evaluated on the untouched56 heldout response coordinates.',
            'Neither a successful local fit nor these finite starts exclude all models in an excitation interval.',
            'The previous geometry/calibration/model assumptions remain unchanged.'])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-nfev',type=int,default=1200)
    args=parser.parse_args()
    jax.config.update('jax_enable_x64',True)
    result=run_check(args.input,args.output,max_nfev=args.max_nfev)
    print('Saved',args.output,'in',result['elapsed_seconds'],'seconds',flush=True)
