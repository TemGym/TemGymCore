"""Recover upstream maps from held-out images using a fitted PL1 analyser.

The analyser is fitted on disjoint images. This audit reconstructs all four
upstream rows from paired held-out position responses, carrying its shared
calibration covariance into every map. It prepares the IL3 stage; it does not
fit IL3 or claim that independent noisy maps identify the whole column.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from temgym_core.projector_response_tomography import pair_matrix, reconstruct_map_scan


ROOT = Path(__file__).resolve().parents[2]


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_disjoint_pairs(parameters, controls, *, balance_length_m=.01):
    """Select by calibrated analyser conditioning, never by held-out residuals.

    This small audit takes at most four held-out states. All candidate pairs
    are recorded; a remaining odd state stays in the parent held-out test.
    """
    controls = np.asarray(controls)
    if controls.ndim != 1 or not 2 <= len(controls) <= 4:
        raise ValueError("This held-out audit requires two to four analyser settings.")
    rows = []
    for pair in itertools.combinations(range(len(controls)), 2):
        matrix = np.asarray(pair_matrix(jnp.asarray(parameters), jnp.asarray(controls[list(pair)])))
        condition = float(np.linalg.cond(matrix*np.array([1.,1.,1/balance_length_m,1/balance_length_m])))
        rows.append(dict(indices=list(pair), condition_number=condition if np.isfinite(condition) else None))
    combinations = [selection for selection in itertools.combinations(rows, len(controls)//2)
                    if len({i for row in selection for i in row['indices']}) == 2*len(selection)]
    score = lambda row: np.inf if row['condition_number'] is None else np.log(row['condition_number'])
    selected = min(combinations, key=lambda selection: sum(score(row) for row in selection))
    if any(row['condition_number'] is None or row['condition_number'] > 1e6 for row in selected):
        raise ValueError("No sufficiently independent held-out analyser pair.")
    return [row['indices'] for row in selected], rows


def standardized_chi_square(error, covariance):
    """Avoid mixing raw metres/radians while solving the covariance system."""
    error, covariance = np.asarray(error).ravel(), np.asarray(covariance)
    scales = np.sqrt(np.diag(covariance))
    if np.any(scales <= 0) or not np.all(np.isfinite(scales)):
        raise ValueError("Positive finite uncertainty is required in every recovered map coordinate.")
    correlation = covariance/scales[:,None]/scales[None,:]
    white = np.linalg.solve(np.linalg.cholesky((correlation+correlation.T)/2), error/scales)
    return float(white@white)


def run_audit(calibration_path, output_path):
    if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
        raise RuntimeError("Run the differentiable reconstruction on GPU float64.")
    calibration_path = Path(calibration_path).resolve()
    report = json.loads(calibration_path.read_text())
    paths = {calibration_path, Path(__file__), ROOT/'src/temgym_core/projector_response_tomography.py'}
    for name, checksum in report['source_hashes'].items():
        path = ROOT/name
        if _digest(path) != checksum:
            raise ValueError(f"Stale physical analyser evidence: {name}")
        paths.add(path)
    hashes = {str(path.relative_to(ROOT)):_digest(path) for path in paths}
    if report['best'] is None or report['information']['numerical_rank'] != 9:
        raise ValueError("A fitted, locally full-rank physical analyser is required.")
    cal = report['analyser_calibration']
    if cal['coordinate_names'] != ['log_length_m','log_gap_m','q_bias','q_slope']:
        raise ValueError("Unexpected analyser parameter coordinates.")
    if report['heldout_used_for_initialization_or_fitting']:
        raise ValueError("Peeling validation requires disjoint calibration and held-out images.")
    held = cal['heldout']; controls = np.asarray(held['controls'])
    responses, covariance = np.asarray(held['observed_response']), np.asarray(held['response_covariance'])
    parameters, ccal = np.asarray(cal['coordinates']), np.asarray(cal['marginal_covariance'])
    steps, pixel = np.asarray(cal['calibrated_input_steps_grouped']), cal['detector_pixel_m']
    pairs, candidates = choose_disjoint_pairs(parameters, controls)
    # Pairs have different controls. Each map shares the same four uncertain
    # analyser parameters; their marginal covariances alone would lose this.
    maps, measurement_blocks, jacobians, diagnostics = [], [], [], []
    for indices in pairs:
        cdata = np.zeros((16,16))
        for local, index in enumerate(indices):
            cdata[8*local:8*local+8,8*local:8*local+8] = covariance[index]
        result = reconstruct_map_scan(responses[indices][None], parameters, controls[indices],
            steps, pixel, cdata, ccal, calibration_data_independent=True)
        maps.append(result.maps[0]); measurement_blocks.append(result.measurement_covariance)
        jacobians.append(result.analyser_jacobian)
        diagnostics.append(dict(heldout_indices=indices, controls=controls[indices].tolist(),
            balanced_condition_number=result.balanced_analyser_condition_number))
    maps = np.asarray(maps)
    cmeasure = np.zeros((maps.size,maps.size))
    for i, block in enumerate(measurement_blocks):
        cmeasure[16*i:16*i+16,16*i:16*i+16] = block
    jcal = np.concatenate(jacobians)
    cshared = jcal@ccal@jcal.T
    joint = cmeasure+cshared
    # Generating upstream is used only after held-out reconstruction.
    truth = report.get('truth_diagnostic', {})
    validation = {}
    if truth.get('path'):
        truth_path = ROOT/truth['path']
        if _digest(truth_path) != truth['sha256']:
            raise ValueError("Post-reconstruction truth checksum mismatch.")
        raw = json.loads(truth_path.read_text())
        upstream = np.asarray(raw.get('inputs',raw)['upstream_mechanical'])
        error = maps-upstream
        chi = standardized_chi_square(error, joint)
        individual = error.reshape(-1)/np.sqrt(np.diag(joint))
        validation = dict(upstream_generating_map=upstream.tolist(), truth_accessed_only_after_reconstruction=True,
            joint_chi_square=chi, coordinate_count=maps.size,
            joint_chi_square_z=float((chi-maps.size)/np.sqrt(2*maps.size)),
            maximum_marginal_error_sigma=float(np.max(np.abs(individual))),
            relative_frobenius_map_errors=(np.linalg.norm(error,axis=(1,2))/np.linalg.norm(upstream)).tolist(),
            unscaled_relative_errors_are_not_physical_parameter_errors=True)
    correlations = cshared/np.sqrt(np.diag(joint))[:,None]/np.sqrt(np.diag(joint))[None,:]
    shared_cross = float(np.max(np.abs(correlations[:16,16:]))) if len(maps)>1 else None
    output = dict(schema='neoarm-upstream-heldout-peeling-v1', source_hashes=hashes,
        backend=jax.default_backend(), x64_enabled=True,
        calibration_report_path=str(calibration_path.relative_to(ROOT)),
        calibration_report_sha256=_digest(calibration_path),
        analyser_parameters=parameters.tolist(), analyser_covariance=ccal.tolist(),
        selected_pairs=diagnostics, candidate_pairs=candidates,
        unused_heldout_indices=sorted(set(range(len(controls)))-{i for pair in pairs for i in pair}),
        reconstructed_upstream_maps=maps.tolist(), measurement_covariance=cmeasure.tolist(),
        analyser_jacobian=jcal.tolist(), shared_analyser_covariance=cshared.tolist(),
        full_joint_covariance=joint.tolist(), maximum_cross_map_shared_correlation=shared_cross,
        validation=validation, upstream_lens_parameters_fitted=False,
        source_hashes_unchanged_during_execution=True,
        limitations=['Every analyser pair is selected from training-calibrated conditioning, without observing held-out residuals.',
            'Calibration and reconstruction exposures are disjoint; shared analyser covariance correlates the recovered maps.',
            'Covariance uses a first-order delta method; physical-input and detector calibration are exact here.',
            'The upstream map is reconstructed, not fixed to a round model or independently decomposed into IL lenses.',
            'The complete projector fit must use joint image responses or preserve these cross-setting covariances.'])
    if any(_digest(ROOT/name) != value for name,value in hashes.items()):
        raise RuntimeError("Evidence changed during peeling validation.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration-report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = run_audit(args.calibration_report,args.output)
    print(json.dumps(result['validation'],indent=2))
