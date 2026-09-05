"""Bounded fixed-optics nuisance profiles of two frozen training checkpoints.

This driver does not perform optical recovery, inspect generating values, or
evaluate held-out observations. It preserves the original 102-coordinate data
likelihood and its single common aperture-radius calibration prior.
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

from temgym_core.profiled_least_squares import profile_nuisance
from neoarm_condenser_extension_recovery import selected_data, beam_residual
from neoarm_condenser_extension_design import load_context, training_residual, FOLDER
from neoarm_condenser_covariance_fit import VisibleSourceModel, quantum_completion
from neoarm_aperture_capture import ROOT


def run(output, *, max_iterations=20):
    if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
        raise RuntimeError('The numerical audit requires GPU float64.')
    guarded_path = FOLDER/'condenser_guarded_recovery.json'
    newton_path = FOLDER/'condenser_newton_refinement.json'
    guarded = json.loads(guarded_path.read_text())
    newton = json.loads(newton_path.read_text())
    original = next(row for row in guarded['rows'] if row['initializer'] == 'rough_geometry_data_source')
    cases = [('guarded_rough_training', original['parameters'], original['total_chi_square']),
             ('bounded_newton_training', newton['parameters'], newton['training_total_chi_square'])]
    _, arrays, _, aperture, model = load_context()
    quotient = VisibleSourceModel(model)
    array_path = ROOT/guarded['arrays_path']
    with np.load(array_path, allow_pickle=False) as archive:
        fresh = selected_data(archive, True)
    old_loss = training_residual(quotient, arrays, aperture)
    new_loss = beam_residual(quotient, fresh)
    residual = lambda p: jnp.r_[old_loss(p), new_loss(p)]
    rows = []
    for label, parameters, expected in cases:
        initial = np.asarray(parameters)
        value = np.asarray(jax.jit(residual)(initial))
        if not np.isclose(value@value, expected, atol=1e-8, rtol=1e-12):
            raise RuntimeError('The saved training likelihood did not close.')
        fit = profile_nuisance(residual, initial, quotient.admissible,
            n_outer=12, max_iterations=max_iterations)
        if not np.array_equal(fit['parameters'][:12], initial[:12]):
            raise RuntimeError('The fixed optical coordinates changed.')
        row = dict(label=label, initial_parameters=initial.tolist(),
            initial_chi_square=float(value@value), parameters=fit['parameters'].tolist(),
            chi_square=float(fit['residual']@fit['residual']), success=fit['success'],
            reason=fit['reason'], nuisance_history=fit['history'],
            full_normalized_gradient=fit['normalized_gradient'],
            quantum_margin=quantum_completion(fit['parameters'][12:21])['margin'],
            optical_coordinates_unchanged=True,
            residual_evaluations=fit['evaluations'], derivative_evaluations=fit['derivative_evaluations'],
            hessian_evaluations=fit['hessian_evaluations'])
        rows.append(row)
        print(label, row['success'], row['reason'], row['chi_square'], flush=True)
    hashes = dict(newton['source_hashes'])
    paths = (Path(__file__), ROOT/'src/temgym_core/profiled_least_squares.py',
             ROOT/'tests/test_profiled_least_squares.py', guarded_path, newton_path, array_path)
    for path in paths:
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name, expected in hashes.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'A scoped input changed: {name}')
    report = dict(schema='neoarm-condenser-fixed-optics-source-profile-v1',
        backend=jax.default_backend(), x64_enabled=True, source_hashes=hashes, rows=rows,
        training_data_coordinate_count=102, common_radius_prior_count=1,
        outer_optical_coordinates=12, profiled_source_coordinates=9,
        profiled_calibration_coordinates=2, max_nuisance_iterations=max_iterations,
        generating_truth_values_used=False, heldout_observations_used=False,
        optical_fit_performed=False, full_microscope_recovery=False,
        limitations=['Conditional objective/downstream calibration and aperture geometry remain unchanged.',
            'Stationary nuisance profiles are not stationary optical fits or branch exclusions.'])
    Path(output).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-iterations', type=int, default=20)
    args = parser.parse_args()
    run(args.output, max_iterations=args.max_iterations)
