"""Bounded, training-only cold condenser initialization benchmark.

Every seed is rebuilt from the measured-diagram/provisional optical model and
the training covariance observations. No saved fitted geometry, generating
parameter vector, or held-out observation supplies a seed. The published
nominal-model/generator coincidence is explicitly disclosed: only the two
prescribed displaced starts challenge geometry initialization in this fixture.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault('JAX_ENABLE_X64', 'true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import jax
if __name__ == '__main__':
    jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from neoarm_condenser_extension_recovery import selected_data, beam_residual
from neoarm_condenser_extension_design import load_context, training_residual, covariance_factor, FOLDER
from neoarm_condenser_covariance_fit import VisibleSourceModel, visible_coordinates, quantum_completion, geometry_gaps
from neoarm_condenser_image_recovery import linear_source_initializer
from neoarm_condenser_reference_source import ReferenceSourceChart
from neoarm_condenser_scaled_cl1_profile import scaled_profiled_least_squares
from neoarm_condenser_cl1_profile import source_diagnostic
from neoarm_shared_objective_audit import provenance_snapshot
from neoarm_aperture_capture import ROOT


GEOMETRY_INDICES = (0, 1, 4, 5, 8, 9)
NUISANCE_INDICES = (2, 3, 6, 7, 10, 11, *range(12, 23))
ORDER = np.asarray(GEOMETRY_INDICES+NUISANCE_INDICES)
INVERSE = np.argsort(ORDER)


def displaced_diagram_optics(base, sign):
    """Prescribed non-fitted shifts; q/sqrt(L) is held constant initially."""
    if sign not in (-1, 0, 1):
        raise ValueError('Use one of the three prescribed displacement signs.')
    base = np.asarray(base, dtype=float)
    if base.shape != (12,) or not np.isfinite(base).all():
        raise ValueError('A finite twelve-coordinate provisional column is required.')
    value = base.reshape(3, 4).copy()
    value[:, 0] += sign*np.array([1e-3, -.3e-3, .15e-3])
    length_shift = sign*np.array([.08, -.06, .10])
    value[:, 1] += length_shift
    value[:, 2:4] *= np.exp(length_shift[:, None]/2)
    return value.reshape(-1)


def cold_initializers(quotient, beam_controls, beam_data, beam_covariance,
                      aperture_controls, aperture_data, aperture_covariance,
                      *, initializer=linear_source_initializer):
    """Build three source estimates on fixed, prescribed geometry; restore base."""
    model = quotient.model
    base = model.base.copy()
    result = []
    try:
        for sign in (-1, 1, 0):
            model.base = displaced_diagram_optics(base, sign)
            initialized, metadata = initializer(model, beam_controls, beam_data, beam_covariance,
                aperture_controls, aperture_data, aperture_covariance)
            optical, source, angle = model.decode(initialized)
            # The optical chart is relative to the original provisional model,
            # never to a fitted geometry or the temporary displaced base.
            model.base = base.copy()
            p = np.r_[quotient.optical_coordinates(optical), visible_coordinates(source), float(angle), 0.]
            label = 'rough_diagram_data_source' if sign == 0 else f'displaced_diagram_{sign:+}'
            result.append(dict(label=label, sign=sign, parameters=p,
                initial_optical_parameters=np.asarray(optical), source_initialization=metadata,
                geometry_genuinely_displaced_from_diagram=sign != 0))
    finally:
        model.base = base.copy()
    return result


def run(output, *, residual_scale=1e-4, max_outer_iterations=20, max_inner_iterations=30):
    if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
        raise RuntimeError('The cold benchmark requires GPU float64.')
    if not 0 <= max_outer_iterations <= 20 or not 0 <= max_inner_iterations <= 30:
        raise ValueError('The initial benchmark is capped at20 outer and30 inner iterations per seed.')
    started = perf_counter()
    inference_path = FOLDER/'condenser_extension_images.json'
    inference = json.loads(inference_path.read_text())
    array_path = ROOT/inference['arrays_path']
    if hashlib.sha256(array_path.read_bytes()).hexdigest() != inference['arrays_sha256']:
        raise ValueError('The training archive checksum changed.')
    if not inference['all_images_accepted']:
        raise ValueError('The finite-pixel acquisition did not pass admission.')
    with np.load(array_path, allow_pickle=False) as archive:
        fresh = selected_data(archive, True)
    old, arrays, _, aperture, model = load_context()
    quotient = VisibleSourceModel(model)
    beam_controls = np.concatenate((arrays['controls'][:8], fresh[0]))
    beam_data = np.concatenate((arrays['beam_observables'][:8], fresh[1]))
    beam_covariance = np.concatenate((arrays['beam_observable_covariances'][:8], fresh[2]))
    aperture_controls = np.asarray([row['controls'] for row in aperture['records'][:2]])
    seeds = cold_initializers(quotient, beam_controls, beam_data, beam_covariance,
        aperture_controls, arrays['aperture_observables'][:2], arrays['aperture_observable_covariances'][:2])
    rough = next(row for row in seeds if row['sign'] == 0)
    reference_control = arrays['controls'][0]
    optical, source, _, _ = quotient.decode(rough['parameters'])
    m = model.condenser_matrix(optical, reference_control)
    c = np.asarray(m@source@m.T)
    width = float((c[0, 0]*c[2, 2])**.25)
    chart = ReferenceSourceChart(quotient, reference_control, width)
    old_loss = training_residual(quotient, arrays, aperture)
    fresh_loss = beam_residual(quotient, fresh)
    original_residual = lambda p: jnp.r_[old_loss(p), fresh_loss(p)]
    original_fun = jax.jit(original_residual)
    decode = lambda p: chart.decode(jnp.asarray(p)[jnp.asarray(INVERSE)])
    residual = lambda p: original_residual(decode(p))
    domain = lambda p: quotient.admissible(np.asarray(decode(p)))
    # This boolean is already published provenance, not a generating vector
    # used by the algorithm or by seed/design selection.
    coincidence = old['truth_consistency']['diagram_initial_optical_parameters_coincide_with_generator']
    hashes, _ = provenance_snapshot()
    paths = [Path(__file__), ROOT/'tests/test_neoarm_condenser_cold_start.py', inference_path, array_path,
        FOLDER/'condenser_image_recovery.json', ROOT/old['arrays_path'],
        FOLDER/'condenser_covariance_fit.json', FOLDER/'aperture_image_fit.json']
    paths.extend(ROOT/name for name in aperture['report_input_hashes'])
    ap_array = Path(aperture['array_file']); paths.append(ap_array if ap_array.is_absolute() else ROOT/ap_array)
    for path in paths:
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = []
    for seed in seeds:
        initial = seed['parameters']
        reference = np.asarray(chart.encode(initial))[ORDER]
        r0 = np.asarray(original_fun(initial))
        closure0 = float(np.max(abs(np.asarray(jax.jit(residual)(reference))-r0)))
        if closure0 > 1e-5:
            raise RuntimeError('The cold source chart failed likelihood closure.')
        print('starting', seed['label'], float(r0@r0), flush=True)
        fit = scaled_profiled_least_squares(residual, reference, domain,
            scale=residual_scale, n_outer=6, max_iterations=max_outer_iterations,
            max_inner_iterations=max_inner_iterations)
        p = np.asarray(decode(fit['parameters']))
        r = np.asarray(original_fun(p))
        jac = np.asarray(jax.jit(jax.jacfwd(original_residual))(p))
        norms = np.linalg.norm(jac, axis=0)
        gradient = float(np.max(abs((jac/norms).T@r)))
        singular = np.linalg.svd(jac/norms, compute_uv=False)
        factor = None
        try:
            factor = covariance_factor(jac)
            whitened_gradient = float(np.max(abs((jac@factor).T@r)))
            rank_ok = True
        except ValueError:
            whitened_gradient, rank_ok = None, False
        source_info = source_diagnostic(p)
        emittances = source_info.get('representative_emittances_over_quantum_minimum', [])
        interior_source = bool(len(emittances) == 2 and min(emittances) > 1+1e-6)
        closure = float(np.max(abs(r-fit['residual'])))
        accepted = bool(fit['success'] and rank_ok and gradient < 1e-6 and whitened_gradient < 1e-6
            and quotient.admissible(p) and interior_source and closure < 1e-5)
        optical = np.asarray(quotient.decode(p)[0])
        row = dict(label=seed['label'], sign=seed['sign'], accepted=accepted,
            solver_success=fit['success'], reason=fit['reason'],
            initial_parameters=initial.tolist(), initial_optical_parameters=seed['initial_optical_parameters'].tolist(),
            source_initialization=seed['source_initialization'],
            initial_source=source_diagnostic(initial), initial_training_chi_square=float(r0@r0),
            initial_residual_closure_sigma=closure0, original_parameters=p.tolist(),
            optical_parameters=optical.tolist(), training_chi_square=float(r@r),
            training_data_chi_square=float(r@r-(p[22]/.005)**2), radius_prior_chi_square=float((p[22]/.005)**2),
            original_normalized_gradient=gradient, original_information_whitened_gradient=whitened_gradient,
            original_normalized_jacobian_singular_values=singular.tolist(), numerical_full_rank=rank_ok,
            interior_source_completion=interior_source, source=source_info,
            geometry_gaps_m=geometry_gaps(model, optical).tolist(), residual_closure_sigma=closure,
            history=fit['history'], initial_nuisance_history=fit['initial_nuisance_history'],
            solver_unit_conversion=fit['unit_conversion'],
            residual_evaluations=fit['evaluations'], derivative_evaluations=fit['derivative_evaluations'],
            hessian_evaluations=fit['hessian_evaluations'],
            geometry_genuinely_displaced_from_diagram=seed['geometry_genuinely_displaced_from_diagram'])
        rows.append(row)
        print('finished', seed['label'], accepted, fit['reason'], row['training_chi_square'], gradient, flush=True)
    for name, checksum in hashes.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != checksum:
            raise RuntimeError(f'A cold-benchmark input changed: {name}')
    report = dict(schema='neoarm-condenser-cold-start-three-seed-v1', backend=jax.default_backend(),
        x64_enabled=True, source_hashes=hashes, rows=rows, elapsed_seconds=perf_counter()-started,
        provisional_diagram_optical_parameters=model.base.tolist(),
        reference_control=np.asarray(reference_control).tolist(), reference_width_m=width,
        reference_units_from_rough_training_source_seed=True,
        published_nominal_geometry_equals_generator=bool(coincidence),
        published_coincidence_boolean_used_for_disclosure_only=True,
        fitted_geometry_used_for_seeds=False, saved_final_source_or_covariance_used=False,
        numerical_generating_values_used_for_seeds_or_fit=False, heldout_observations_used=False,
        three_seeds_prescribed_before_optimization=True, relative_likelihood_weights_changed=False,
        uniform_residual_scale=residual_scale, original_noise_unit_step_cap=1/residual_scale,
        max_outer_iterations=max_outer_iterations, max_inner_iterations=max_inner_iterations,
        geometry_outer_coordinates=6, excitation_source_calibration_inner_coordinates=17,
        all_twelve_optical_coordinates_free=True, total_free_coordinates=23,
        training_data_coordinate_count=102, common_radius_prior_count=1,
        prior_posterior_used_as_prior=False, global_branch_search_performed=False,
        genuinely_displaced_stationary_fits=sum(row['accepted'] and row['sign'] != 0 for row in rows),
        complete_microscope_recovery=False, hardware_admitted=False,
        limitations=['The rough seed is not a blind geometry test: the published fixture reused the nominal diagram geometry as its generator.',
            'The two displaced seeds perturb centres and log lengths without using a recovered optical solution.',
            'Conditional objective/downstream calibration, source preparation and aperture geometry remain unchanged.',
            'This three-start benchmark is neither global branch exclusion nor uncertainty coverage.',
            'Source covariance moment tomography and cone projection are initialization only; all final fits use the unchanged log-width/correlation residual.'])
    Path(output).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--residual-scale', type=float, default=1e-4)
    args = parser.parse_args()
    run(args.output, residual_scale=args.residual_scale)
