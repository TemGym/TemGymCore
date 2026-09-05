"""Cold moment-guide -> physical condenser -> joint endpoint calibration.

Only the prescribed three-start moment benchmark supplies a condenser seed.
The six endpoint coordinates come from measured signed-input responses. All
29 coordinates are then released in the unchanged joint likelihood: 102
condenser data coordinates, 16 independent endpoint coordinates, one radius
prior. No warm recovery, generating parameters or held-out observations are
loaded for initialization, selection or optimization.
"""

import argparse
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

from neoarm_aperture_capture import ROOT
from neoarm_condenser_cl1_profile import source_diagnostic
from neoarm_condenser_covariance_fit import VisibleSourceModel, geometry_gaps
from neoarm_condenser_endpoint_calibration import EndpointRows, digest
from neoarm_condenser_endpoint_recovery import (
    JointEndpointModel, joint_training_residual, calibration_training, ORDER, INVERSE, verify,
)
from neoarm_condenser_extension_design import load_context, covariance_factor, FOLDER, training_residual
from neoarm_condenser_extension_recovery import selected_data, beam_residual
from neoarm_condenser_reference_source import ReferenceSourceChart
from neoarm_condenser_scaled_cl1_profile import scaled_profiled_least_squares
from neoarm_shared_objective_audit import provenance_snapshot


def select_cold_seed(report):
    """Lowest *original likelihood* accepted result, without reading truth.

    The moment guide's different statistical objective is never used to rank
    candidates. A failed or nonstationary final fit cannot become a seed merely
    by having a smaller reported objective.
    """
    rows = report['rows']
    if len(rows) != 3 or sorted(row['sign'] for row in rows) != [-1, 0, 1]:
        raise ValueError('Exactly the three prescribed cold benchmark starts are required.')
    candidates = []
    for index, row in enumerate(rows):
        fit = row.get('final_exact_fit')
        if fit is None or not fit['accepted']:
            continue
        p = np.asarray(fit['original_parameters'], float)
        chi = float(fit['training_chi_square'])
        gradients = [fit['original_normalized_gradient'], fit['original_information_whitened_gradient']]
        if (p.shape != (23,) or not np.isfinite(p).all() or not np.isfinite(chi) or chi < 0
                or not fit['optimizer_success'] or not fit['numerical_full_rank']
                or not fit['all23_parameters_free'] or any(g is None or not np.isfinite(g)
                    or g < 0 or g >= 1e-6 for g in gradients)):
            raise ValueError('The claimed accepted cold seed fails its stationarity contract.')
        candidates.append((chi, index, row))
    if not candidates:
        raise ValueError('No accepted cold original-likelihood fit is available.')
    _, index, row = min(candidates, key=lambda item: (item[0], item[1]))
    return np.asarray(row['final_exact_fit']['original_parameters'], float), dict(
        row_index=index, label=row['label'], sign=row['sign'],
        accepted_count=len(candidates), prescribed_start_count=3,
        selection_statistic='lowest accepted original training chi-square',
        original_training_chi_square=row['final_exact_fit']['training_chi_square'],
        initial_optical_parameters=row['initial_optical_parameters'],
        source_initialization=row['source_initialization'])


def joint_reference_coordinates(chart):
    """Invertible 23-source chart plus untouched six endpoint coordinates."""
    def encode(p):
        p = jnp.asarray(p)
        return jnp.r_[chart.encode(p[:23]), p[23:]][jnp.asarray(ORDER)]
    def decode(q):
        p = jnp.asarray(q)[jnp.asarray(INVERSE)]
        return jnp.r_[chart.decode(p[:23]), p[23:]]
    return encode, decode


def run(output, *, max_outer_iterations=20, max_inner_iterations=30):
    if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
        raise RuntimeError('The joint fit requires GPU float64.')
    if not 0 < max_outer_iterations <= 20 or not 0 < max_inner_iterations <= 30:
        raise ValueError('This bounded continuation permits at most 20 outer / 30 inner iterations.')
    started = perf_counter()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    seed_path = FOLDER/'condenser_moment_initialization.json'
    benchmark = json.loads(seed_path.read_text())
    verify(benchmark)
    initial_cond, selection = select_cold_seed(benchmark)
    cal_path = FOLDER/'condenser_endpoint_calibration.json'
    calibration_report = json.loads(cal_path.read_text())
    verify(calibration_report)
    cal_array_path = ROOT/calibration_report['arrays_path']
    if digest(cal_array_path) != calibration_report['arrays_sha256'] or not calibration_report['all_images_accepted']:
        raise ValueError('Unchanged admitted calibration images are required.')
    with np.load(cal_array_path, allow_pickle=False) as archive:
        calibration = calibration_training(archive)
    acquisition_path = FOLDER/'condenser_extension_images.json'
    acquisition = json.loads(acquisition_path.read_text())
    verify(acquisition)
    arrays_path = ROOT/acquisition['arrays_path']
    if digest(arrays_path) != acquisition['arrays_sha256']:
        raise ValueError('The condenser training archive changed.')
    with np.load(arrays_path, allow_pickle=False) as archive:
        fresh = selected_data(archive, True)
    _, old_arrays, _, aperture, model = load_context()
    quotient = VisibleSourceModel(model)
    endpoints = EndpointRows(calibration_report['conditional_sample_kappa_per_m'])
    split = model.column.objective.split_maps(jnp.asarray(model.fixed[-4:]))
    joint = JointEndpointModel(quotient, endpoints, split.pre)
    chart = ReferenceSourceChart(quotient, benchmark['reference_control'], benchmark['reference_width_m'])
    encode, decode = joint_reference_coordinates(chart)
    initial_endpoint = endpoints.encode_response(calibration[0])
    initial = np.r_[initial_cond, initial_endpoint]
    if not joint.admissible(initial):
        raise ValueError('The selected cold seed is outside the original physical domain.')
    old_loss, fresh_loss = training_residual(quotient, old_arrays, aperture), beam_residual(quotient, fresh)
    seed_residual = np.asarray(jax.jit(lambda p: jnp.r_[old_loss(p), fresh_loss(p)])(initial_cond))
    seed_chi = float(seed_residual@seed_residual)
    if not np.isclose(seed_chi, selection['original_training_chi_square'], atol=1e-7, rtol=1e-12):
        raise RuntimeError('The selected cold original training likelihood did not close.')
    original_residual = joint_training_residual(joint, old_arrays, aperture, fresh, calibration)
    residual = lambda p: original_residual(decode(p))
    admissible = lambda p: joint.admissible(np.asarray(decode(p)))
    sources, _ = provenance_snapshot()
    for report in (benchmark, calibration_report, acquisition):
        for name, checksum in report['source_hashes'].items():
            if name in sources and sources[name] != checksum:
                raise ValueError(f'Inconsistent dependency checksum: {name}')
            sources[name] = checksum
    for path in (Path(__file__), ROOT/'tests/test_neoarm_condenser_cold_joint_endpoint.py',
                 seed_path, cal_path, cal_array_path, acquisition_path, arrays_path):
        sources[str(path.relative_to(ROOT))] = digest(path)
    print('cold joint starting', selection['label'], seed_chi, flush=True)
    fit = scaled_profiled_least_squares(residual, np.asarray(encode(initial)), admissible,
        scale=1e-3, n_outer=6, max_iterations=max_outer_iterations,
        max_inner_iterations=max_inner_iterations, gradient_tolerance=1e-6)
    p = np.asarray(decode(fit['parameters']))
    r = np.asarray(jax.jit(original_residual)(p))
    jac = np.asarray(jax.jit(jax.jacfwd(original_residual))(p))
    norms = np.linalg.norm(jac, axis=0)
    singular = np.linalg.svd(jac/norms, compute_uv=False)
    rank = int(np.sum(singular > singular[0]*1e-10))
    gradient = float(np.max(abs((jac/norms).T@r)))
    factor = covariance_factor(jac) if rank == 29 else None
    full_gradient = None if factor is None else float(np.max(abs((jac@factor).T@r)))
    source = source_diagnostic(p[:23])
    emittances = source.get('representative_emittances_over_quantum_minimum', [])
    interior = len(emittances) == 2 and min(emittances) > 1+1e-6
    accepted = bool(fit['success'] and rank == 29 and gradient < 1e-6
        and full_gradient < 1e-6 and joint.admissible(p) and interior)
    if len(r) != 119:
        raise RuntimeError('Exactly 118 data coordinates and one radius prior are required.')
    report = dict(schema='neoarm-condenser-cold-joint-endpoint-v1', source_hashes=sources,
        backend=jax.default_backend(), x64_enabled=True,
        cold_seed_selection=selection, cold_benchmark_path=str(seed_path.relative_to(ROOT)),
        cold_benchmark_sha256=digest(seed_path), cold_seed_likelihood_closure=seed_chi-selection['original_training_chi_square'],
        initial_parameters=initial.tolist(), initial_endpoint_from_measured_response=initial_endpoint.tolist(),
        parameters=p.tolist(), condenser_optical_parameters=np.asarray(quotient.decode(p[:23])[0]).tolist(),
        reference_control=benchmark['reference_control'], reference_width_m=benchmark['reference_width_m'],
        total_chi_square=float(r@r), training_data_chi_square=float(r[:-1]@r[:-1]),
        condenser_data_chi_square=float(r[:102]@r[:102]), calibration_data_chi_square=float(r[102:118]@r[102:118]),
        radius_prior_chi_square=float(r[-1]**2), training_data_coordinate_count=118,
        original_condenser_data_coordinate_count=102, calibration_data_coordinate_count=16, radius_prior_count=1,
        free_parameter_count=29, outer_geometry_count=6, inner_nuisance_count=23,
        all_twelve_condenser_optics_free=True, all_six_endpoint_coordinates_free=True,
        original_normalized_gradient=gradient, original_information_whitened_gradient=full_gradient,
        normalized_singular_values=singular.tolist(), numerical_rank=rank,
        optimizer_success=fit['success'], termination=fit['reason'], accepted_stationary_fit=accepted,
        history=fit['history'], initial_nuisance_history=fit.get('initial_nuisance_history', []),
        numerical_residual_scale=1e-3, solver_unit_conversion=fit['unit_conversion'],
        geometry_gaps_m=geometry_gaps(model, quotient.decode(p[:23])[0]).tolist(), source=source,
        warm_recovery_artifact_used_as_seed=False, generating_values_used_as_seed=False,
        heldout_observations_loaded=False, seed_posterior_added_as_prior=False,
        endpoint_posterior_added_as_prior=False, original_data_enter_once=True,
        conditional_fixed_objective_prefield_and_shear=True,
        complete_physical_column_recovery=False, global_uniqueness_proved=False,
        robust_cold_initialization_demonstrated=False, elapsed_seconds=perf_counter()-started,
        limitations=[
            'One successful displaced cold seed among three prescribed moment-guide starts is not robust global initialization.',
            'The moment guide is temporary; only its accepted original-likelihood physical continuation seeds this joint fit.',
            'The benchmark nominal diagram geometry also generated this synthetic fixture; the selected displaced start is separately identified.',
            'Objective prefield, sample magnetic shear, external input scales, pixel pitch and aperture geometry remain conditional.',
            'The six endpoint coordinates are round position rows, not a physical fit of all downstream lenses.',
            'A stable unknown anisotropic source upstream of the fitted condenser system is still assumed.',
            'Local covariance is conditional and is not nonlinear coverage or excitation-branch exclusion.'])
    if factor is not None:
        report['parameter_covariance'] = (factor@factor.T).tolist()
    verify(report)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('cold joint finished', accepted, float(r@r), gradient, full_gradient, fit['reason'], flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.output)
