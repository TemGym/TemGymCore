"""Training-only geometry/strength-source profiling of the condenser likelihood.

Six centre/log-length coordinates form the outer problem. The six excitation
coordinates, nine visible source-covariance coordinates, collection rotation,
and common radius nuisance are jointly profiled. All twelve optical parameters
remain free overall. Reordering and the fixed-reference source chart change
coordinates only; the 102 data residuals and one radius prior are unchanged.
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

from temgym_core.profiled_least_squares import profile_nuisance, profiled_least_squares
from neoarm_condenser_extension_recovery import selected_data, beam_residual
from neoarm_condenser_extension_design import load_context, training_residual, FOLDER
from neoarm_condenser_covariance_fit import VisibleSourceModel, quantum_completion, geometry_gaps
from neoarm_condenser_reference_source import ReferenceSourceChart
from neoarm_shared_objective_audit import provenance_snapshot
from neoarm_aperture_capture import ROOT


GEOMETRY_INDICES = (0, 1, 4, 5, 8, 9)
NUISANCE_INDICES = (2, 3, 6, 7, 10, 11, *range(12, 23))
PARAMETER_ORDER = np.asarray(GEOMETRY_INDICES+NUISANCE_INDICES)
INVERSE_ORDER = np.argsort(PARAMETER_ORDER)


def geometry_first(parameters):
    return jnp.asarray(parameters)[jnp.asarray(PARAMETER_ORDER)]


def original_order(parameters):
    return jnp.asarray(parameters)[jnp.asarray(INVERSE_ORDER)]


def run(output, *, max_inner_iterations=30, max_outer_iterations=20):
    if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
        raise RuntimeError('The numerical diagnostic requires GPU float64.')
    if not 0 <= max_outer_iterations <= 20:
        raise ValueError('This bounded experiment permits at most20 outer steps.')
    checkpoint_path = FOLDER/'condenser_newton_refinement.json'
    profile_path = FOLDER/'condenser_profiled_recovery.json'
    checkpoint = json.loads(checkpoint_path.read_text())
    profile_report = json.loads(profile_path.read_text())
    profile_row = next(row for row in profile_report['rows'] if row['initializer'] == 'rough_geometry_data_source')
    reference = profile_row['history'][0]
    _, arrays, _, aperture, model = load_context()
    quotient = VisibleSourceModel(model)
    chart = ReferenceSourceChart(quotient, reference['source_reference_control'], reference['source_reference_width_m'])
    array_path = ROOT/profile_report['arrays_path']
    if hashlib.sha256(array_path.read_bytes()).hexdigest() != profile_report['arrays_sha256']:
        raise ValueError('The frozen training archive checksum changed.')
    with np.load(array_path, allow_pickle=False) as archive:
        fresh = selected_data(archive, True)
    old_loss = training_residual(quotient, arrays, aperture)
    fresh_loss = beam_residual(quotient, fresh)
    original_residual = lambda p: jnp.r_[old_loss(p), fresh_loss(p)]
    decode = lambda p: chart.decode(original_order(p))
    residual = lambda p: original_residual(decode(p))
    admissible = lambda p: quotient.admissible(np.asarray(decode(p)))
    cases = [('saved_newton_training', checkpoint['parameters'], checkpoint['training_total_chi_square']),
             ('rough_twelve_outer_profile_training', profile_row['parameters'], profile_row['total_chi_square'])]
    # Bind the currently used computational closure, not every historical
    # optimizer listed in an input report. Input report bytes remain bound too.
    hashes, _ = provenance_snapshot()
    input_paths = [checkpoint_path, profile_path, array_path, Path(__file__),
        ROOT/'tests/test_neoarm_condenser_geometry_profile_diagnostic.py']
    # load_context's old observations and aperture metrology are indirectly
    # referenced by reports; include their exact currently verified artifacts.
    for old_name in ('condenser_image_recovery.json', 'condenser_covariance_fit.json'):
        old_path = FOLDER/old_name
        input_paths.append(old_path)
        old = json.loads(old_path.read_text())
        for key in ('arrays_path', 'array_file'):
            if key in old:
                value = Path(old[key]); input_paths.append(value if value.is_absolute() else ROOT/value)
    for name in aperture['report_input_hashes']:
        input_paths.append(ROOT/name)
    input_paths.append(FOLDER/'aperture_image_fit.json')
    ap_array = Path(aperture['array_file'])
    input_paths.append(ap_array if ap_array.is_absolute() else ROOT/ap_array)
    for path in input_paths:
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()

    def serialize(label, fit, initial, kind):
        original = np.asarray(decode(fit['parameters']))
        r = np.asarray(jax.jit(original_residual)(original))
        jacobian = np.asarray(jax.jit(jax.jacfwd(original_residual))(original))
        gradient = float(np.max(abs((jacobian/np.linalg.norm(jacobian, axis=0)).T@r)))
        optical = np.asarray(quotient.decode(original)[0])
        closure = float(np.max(abs(r-fit['residual'])))
        result = dict(label=label, kind=kind, initial_original_parameters=np.asarray(initial).tolist(),
            original_parameters=original.tolist(), reordered_reference_parameters=fit['parameters'].tolist(),
            solver_success=fit['success'], reason=fit['reason'], history=fit['history'],
            chi_square=float(r@r), data_chi_square=float(r@r-(original[22]/.005)**2),
            original_normalized_gradient=gradient, residual_closure_sigma=closure,
            quantum_margin=quantum_completion(original[12:21])['margin'],
            geometry_gaps_m=geometry_gaps(model, optical).tolist(),
            residual_evaluations=fit['evaluations'], derivative_evaluations=fit['derivative_evaluations'],
            hessian_evaluations=fit['hessian_evaluations'])
        if kind == 'fixed_geometry_inner':
            result['geometry_coordinates_unchanged'] = bool(np.array_equal(original[list(GEOMETRY_INDICES)], np.asarray(initial)[list(GEOMETRY_INDICES)]))
            if not result['geometry_coordinates_unchanged']:
                raise RuntimeError('The fixed geometry changed during nuisance profiling.')
        else:
            result['original_stationary_success'] = bool(fit['success'] and gradient < 1e-6 and closure < 1e-5 and quotient.admissible(original))
        return result

    rows, fits = [], []
    for label, original, expected in cases:
        p = np.asarray(geometry_first(chart.encode(original)))
        check = np.asarray(jax.jit(original_residual)(np.asarray(original)))
        if not np.isclose(check@check, expected, rtol=1e-12, atol=1e-8):
            raise RuntimeError('The frozen original likelihood did not close.')
        closure = float(np.max(abs(np.asarray(jax.jit(residual)(p))-check)))
        if closure > 1e-5:
            raise RuntimeError('The reordered reference chart changed the residual.')
        fit = profile_nuisance(residual, p, admissible, n_outer=6,
            max_iterations=max_inner_iterations)
        row = serialize(label, fit, original, 'fixed_geometry_inner')
        row['initial_residual_closure_sigma'] = closure
        rows.append(row); fits.append(fit)
        print(label, fit['success'], fit['reason'], row['chi_square'], flush=True)
    continuation = None
    if fits[0]['success']:
        fit = profiled_least_squares(residual, fits[0]['parameters'], admissible, n_outer=6,
            max_iterations=max_outer_iterations, max_inner_iterations=max_inner_iterations)
        continuation = serialize('saved_newton_geometry_continuation', fit, checkpoint['parameters'], 'six_geometry_outer')
        continuation['initial_nuisance_history'] = fit['initial_nuisance_history']
        print('geometry_continuation', fit['success'], fit['reason'], continuation['chi_square'], flush=True)
    for name, checksum in hashes.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != checksum:
            raise RuntimeError(f'A bound computational input changed: {name}')
    report = dict(schema='neoarm-condenser-geometry-strength-source-profile-v1',
        backend=jax.default_backend(), x64_enabled=True, source_hashes=hashes,
        geometry_indices=list(GEOMETRY_INDICES), nuisance_indices=list(NUISANCE_INDICES),
        reference_control=np.asarray(chart.reference_control).tolist(),
        reference_width_m=chart.reference_width_m, rows=rows, continuation=continuation,
        max_inner_iterations=max_inner_iterations, max_outer_iterations=max_outer_iterations,
        training_data_coordinate_count=102, common_radius_prior_count=1,
        all_twelve_optical_coordinates_free_in_outer_continuation=True,
        excitation_coordinates_profiled=6, source_coordinates_profiled=9,
        additional_calibration_coordinates_profiled=2,
        generating_truth_values_used=False, heldout_observations_used=False,
        added_priors=False, changed_likelihood=False, discarded_rank_directions=False,
        complete_microscope_recovery=False, global_uniqueness_proved=False,
        limitations=['One bounded continuation from a saved training-only iterate, not independent initialization.',
            'Fixed conditional objective/downstream calibration, source preparation and aperture geometry remain.',
            'Stationary fixed-geometry nuisances alone are not an optical fit.',
            'Neither hardware admission nor full-column branch exclusion is established.'])
    Path(output).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-inner-iterations', type=int, default=30)
    parser.add_argument('--max-outer-iterations', type=int, default=20)
    args = parser.parse_args()
    run(args.output, max_inner_iterations=args.max_inner_iterations, max_outer_iterations=args.max_outer_iterations)
