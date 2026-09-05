"""Joint physical PL1/IL3 recovery from independent signed-response images.

The original PL1 calibration responses and new paired-PL1/varied-IL3 responses
enter one likelihood. All eight lens parameters and five unknown round upstream
coordinates are released. The fitted PL1 posterior is used ONLY to initialize;
adding it as a prior would count the calibration images twice.

The objective/sample magnetic shear is unknown, not fixed to generating truth.
This conditional round hard-edge example neither fits the rest of the column
nor proves global uniqueness or establishes real-instrument calibration.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
import numpy as np

from neoarm_last_lens_physical_fit import (
    ROOT, RoundLensResponse, load_image_responses, seed_from_response_parameters,
    analyser_coordinates,
)
from neoarm_last_lens_tomography import J2, rotation
from temgym_core.jax_least_squares import least_squares
from temgym_core.projector_response_tomography import reconstruct_map_scan


PARAMETER_NAMES = ('log_combined_span_m', 'log_PL1_span_over_IL3_span',
    'PL1.log_gap_over_length', 'PL1.q_bias', 'PL1.q_slope',
    'IL3.log_gap_over_length', 'IL3.q_bias', 'IL3.q_slope',
    'upstream_log_row_norm', 'upstream_row_angle', 'upstream_row_shear',
    'upstream_rotation_rad', 'sample_kappa_times_balance_length')
PHYSICAL_NAMES = ('PL1.log_length_m', 'PL1.log_gap_m', 'PL1.q_bias', 'PL1.q_slope',
                  'IL3.log_length_m', 'IL3.log_gap_m', 'IL3.q_bias', 'IL3.q_slope')


def stage_coordinates(parameters):
    """Eight physical coordinates; one cap bounds the combined last-two span."""
    p = jnp.asarray(parameters)
    last_span = p[0]-jax.nn.softplus(-p[1])
    previous_span = p[0]-jax.nn.softplus(p[1])
    return jnp.array([last_span-jax.nn.softplus(p[2]), last_span-jax.nn.softplus(-p[2]), p[3], p[4],
        previous_span-jax.nn.softplus(p[5]), previous_span-jax.nn.softplus(-p[5]), p[6], p[7]])


def radial_stage(parameters, controls):
    """Exact hard-edge radial map followed by its physical exit-to-next gap."""
    length, gap = jnp.exp(parameters[:2])
    q = parameters[2]+parameters[3]*jnp.asarray(controls)
    cosine, sine = jnp.cos(q), jnp.sin(q)
    aa = cosine-(gap/length)*q*sine
    bb = length*jnp.sinc(q/jnp.pi)+gap*cosine
    cc, dd = -(q/length)*sine, cosine
    return jnp.stack((jnp.stack((aa, bb), axis=-1), jnp.stack((cc, dd), axis=-1)), axis=-2)


def grouped_stage(parameters, control):
    q = parameters[2]+parameters[3]*control
    return jnp.kron(radial_stage(parameters, control), rotation(-q))


@dataclass(frozen=True)
class CoupledProjectorResponse:
    calibration: RoundLensResponse

    def upstream_matrix(self, parameters, *, coordinates='mechanical'):
        p = jnp.asarray(parameters)
        return self.calibration.upstream_matrix(jnp.concatenate((jnp.zeros(4), p[8:])), coordinates=coordinates)

    def predict(self, parameters, controls):
        """Controls (N,2) are PL1 normalized coordinate and native IL3 offset."""
        p, controls = jnp.asarray(parameters), jnp.asarray(controls)
        if p.shape != (13,) or controls.ndim != 2 or controls.shape[1] != 2:
            raise ValueError('Expected thirteen parameters and (N,2) control coordinates.')
        physical = stage_coordinates(p)
        last, previous = radial_stage(physical[:4], controls[:, 0]), radial_stage(physical[4:], controls[:, 1])
        radial = self.calibration.radial_matrix(jnp.concatenate((jnp.zeros(4), p[8:])))
        row = (last@previous@radial)[:, 0]
        aa, bb = row[:, 0], row[:, 1]
        position = aa[:, None, None]*jnp.eye(2)+(p[12]/self.calibration.balance_length_m)*bb[:, None, None]*J2
        tilt = bb[:, None, None]*jnp.eye(2)
        phase = p[11]-(physical[2]+physical[3]*controls[:, 0])-(physical[6]+physical[7]*controls[:, 1])
        result = rotation(phase)@jnp.concatenate((position, tilt), axis=-1)
        return result*(jnp.asarray(self.calibration.input_steps_grouped)/self.calibration.pixel_m)


def encode_stages(last, previous, upstream_coordinates):
    last, previous = np.asarray(last), np.asarray(previous)
    spans = np.array([np.exp(last[:2]).sum(), np.exp(previous[:2]).sum()])
    return np.r_[np.log(spans.sum()), np.log(spans[0]/spans[1]), last[1]-last[0], last[2:],
                 previous[1]-previous[0], previous[2:], upstream_coordinates]


def physical_upstream_coordinates(upstream, calibration):
    """Data-derived projection in the same global SL2 chart as the PL1 fit."""
    # The arbitrary unit geometry cancels in decoding: it is a chart adapter,
    # not a fitted lens, a physical initializer, or a granted generating length.
    coefficients = np.concatenate((-np.asarray(upstream)[:2], np.asarray(upstream)[2:]))
    coefficients *= np.asarray(calibration.input_steps_grouped)/calibration.pixel_m
    pseudo = np.r_[0., 0., 1., coefficients.ravel()]
    return seed_from_response_parameters(pseudo, calibration)[4:]


class JointResponseProblem:
    def __init__(self, model, controls, responses, covariance):
        self.model = model
        self.controls, self.responses = map(jnp.asarray, (controls, responses))
        count = self.responses.size
        cov = np.asarray(covariance)
        if self.responses.shape != (len(self.controls), 2, 4) or cov.shape != (count, count):
            raise ValueError('Response controls and full joint covariance have incompatible shapes.')
        if not np.all(np.isfinite(cov)) or not np.allclose(cov, cov.T, rtol=1e-12, atol=1e-18):
            raise ValueError('The joint response covariance must be finite and symmetric.')
        self.cholesky = jnp.asarray(np.linalg.cholesky(cov))
        self.residual = jax.jit(lambda p: solve_triangular(self.cholesky,
            (model.predict(p, self.controls)-self.responses).reshape(-1), lower=True))
        self.jacobian = jax.jit(jax.jacfwd(self.residual))


def block_diagonal(blocks):
    sizes = [len(block) for block in blocks]
    result = np.zeros((sum(sizes), sum(sizes)))
    offset = 0
    for size, block in zip(sizes, blocks):
        result[offset:offset+size, offset:offset+size] = block
        offset += size
    return result


def training_relative_maps(scan, controls, *, balance_length_m=.01):
    """Relative maps and diagonal delta-method scale for initialization ONLY.

    We retain the full covariance for the original-data final likelihood. This
    relative initializer deliberately makes no chi-square/significance claim.
    The shared fitted-analyser covariance is propagated by the scan adapter.
    """
    controls = np.asarray(controls)
    reference = np.flatnonzero(np.isclose(controls, 0., rtol=0, atol=1e-12))
    if len(reference) != 1:
        raise ValueError('A single reference IL3 offset zero is required.')
    reference = int(reference[0])
    indices = np.flatnonzero(np.arange(len(controls)) != reference)
    balance = jnp.diag(jnp.array([np.sqrt(balance_length_m)]*2+[1/np.sqrt(balance_length_m)]*2))
    transform = lambda maps: jax.vmap(lambda matrix: jnp.linalg.solve(balance,
        matrix@jnp.linalg.inv(maps[reference])@balance))(maps[jnp.asarray(indices)]).reshape(-1)
    values = np.asarray(transform(jnp.asarray(scan.maps)))
    jac = np.asarray(jax.jacfwd(transform)(jnp.asarray(scan.maps))).reshape(len(values), -1)
    variance = np.einsum('ij,jk,ik->i', jac, scan.joint_covariance, jac)
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(variance) & (variance > 0)):
        raise ValueError('Relative-map uncertainty is singular/nonfinite; initialization is unresolved.')
    return controls[indices], values, np.sqrt(variance), reference


def initialize_previous_stage(scan, controls, *, q_centres=None, max_nfev=300):
    """Data-relative-map starts: both excitation signs and multiple pi bands."""
    ell = .01
    offset, values, sigma, reference = training_relative_maps(scan, controls, balance_length_m=ell)
    balance = jnp.diag(jnp.array([np.sqrt(ell)]*2+[1/np.sqrt(ell)]*2))
    def relative(parameters):
        inverse = jnp.linalg.inv(grouped_stage(parameters, 0.))
        return jax.vmap(lambda u: jnp.linalg.solve(balance,
            grouped_stage(parameters, u)@inverse@balance))(jnp.asarray(offset)).reshape(-1)
    residual = jax.jit(lambda p: (relative(p)-jnp.asarray(values))/jnp.asarray(sigma))
    # Tilt block orientation supplies a data-derived local slope seed, without
    # assuming a field/current calibration or an independent Larmor tag.
    complex_block = lambda block: .5*(np.trace(block)+1j*(block[1, 0]-block[0, 1]))
    tilt_vectors = np.array([[complex_block(m[:2, 2:]), complex_block(m[2:, 2:])] for m in scan.maps])
    overlap = tilt_vectors@tilt_vectors[reference].conj()
    phase = np.unwrap(np.angle(overlap**2))/2
    phase -= phase[reference]
    slope_estimate = -float(np.dot(controls, phase)/np.dot(controls, controls))
    slope_size = float(np.clip(abs(slope_estimate), .05, 4.))
    if q_centres is None:
        q_centres = np.pi*(np.arange(-3, 3)+.5)
    rows = []
    for centre in q_centres:
        for length, gap in ((.012, .045), (.030, .100)):
            for sign in (-1, 1):
                slope_bounds = (-4*np.pi, -.01) if sign < 0 else (.01, 4*np.pi)
                initial = np.array([np.log(length), np.log(gap), centre, sign*slope_size])
                lower = np.array([np.log(.001), np.log(.001), centre-np.pi/2, slope_bounds[0]])
                upper = np.array([np.log(.2), np.log(.7), centre+np.pi/2, slope_bounds[1]])
                norms = np.linalg.norm(np.asarray(jax.jacfwd(residual)(initial)), axis=0)
                fit = least_squares(residual, initial, bounds=(lower, upper), x_scale=1/norms,
                    max_nfev=max_nfev, max_iter=600, ftol=1e-10, xtol=1e-10, gtol=1e-7)
                rows.append(dict(initial=initial.tolist(), parameters=np.asarray(fit.x).tolist(),
                    lower=lower.tolist(), upper=upper.tolist(), optimizer_success=bool(fit.success),
                    termination=fit.status, nfev=fit.nfev, diagonal_relative_map_objective=float(2*fit.cost),
                    metric_is_final_likelihood=False, q_interval=[float(lower[2]), float(upper[2])], slope_sign=sign))
    return rows, dict(reference_index=reference, controls=np.asarray(controls).tolist(),
        estimated_q_slope_from_data=slope_estimate, relative_map_values=values.tolist(),
        relative_map_diagonal_standard_errors=sigma.tolist(), canonical_balance_length_m=ell,
        full_map_joint_covariance=scan.joint_covariance.tolist(), analyser_condition_number=scan.balanced_analyser_condition_number,
        initialization_metric_only=True, generating_optics_used=False)


def fit_joint_start(problem, last_fit, previous_row, *, maximum_span_m=.9, max_nfev=800):
    last = np.asarray(analyser_coordinates(jnp.asarray(last_fit['best']['parameters'])))
    previous = np.asarray(previous_row['parameters']).copy()
    nominal_upstream = np.asarray(problem.model.calibration.upstream_matrix(jnp.asarray(last_fit['best']['parameters'])))
    # If an initializer leaves the total physical box, reduce only its provisional
    # IL3 span; do not modify data or silently admit an out-of-column solution.
    last_span = np.exp(last[:2]).sum()
    previous_span = np.exp(previous[:2]).sum()
    if last_span >= maximum_span_m:
        return dict(optimizer_success=False, status='unresolved_PL1_seed_exceeds_combined_span')
    if last_span+previous_span >= maximum_span_m:
        previous[:2] += np.log(.95*(maximum_span_m-last_span)/previous_span)
    prefix = np.linalg.solve(np.asarray(grouped_stage(jnp.asarray(previous), 0.)), nominal_upstream)
    try:
        upstream = physical_upstream_coordinates(prefix, problem.model.calibration)
    except ValueError as exc:
        return dict(optimizer_success=False, status='unresolved_round_prefix_projection', reason=str(exc))
    initial = encode_stages(last, previous, upstream)
    last_bounds = last_fit['best']['bounds']
    lower = np.array([np.log(.002), -8., np.log(.05), last_bounds[0][2], last_bounds[0][3],
        np.log(.005), previous_row['lower'][2], previous_row['lower'][3], -15., -4*np.pi, -1e5, -4*np.pi, -1e4])
    upper = np.array([np.log(maximum_span_m), 8., np.log(200.), last_bounds[1][2], last_bounds[1][3],
        np.log(1000.), previous_row['upper'][2], previous_row['upper'][3], 15., 4*np.pi, 1e5, 4*np.pi, 1e4])
    initial = np.clip(initial, lower+1e-12*(upper-lower), upper-1e-12*(upper-lower))
    norms = np.linalg.norm(np.asarray(problem.jacobian(initial)), axis=0)
    fit = least_squares(problem.residual, initial, bounds=(lower, upper), x_scale=1/norms,
        max_nfev=max_nfev, max_iter=1500, ftol=1e-11, xtol=1e-11, gtol=1e-7)
    parameters = np.asarray(fit.x)
    return dict(initial=initial.tolist(), parameters=parameters.tolist(), physical_coordinates=np.asarray(stage_coordinates(parameters)).tolist(),
        bounds=[lower.tolist(), upper.tolist()], optimizer_success=bool(fit.success), termination=fit.status, nfev=fit.nfev,
        chi_square=float(2*fit.cost), status='tested_joint_local_fit' if fit.success else 'unresolved_joint_optimizer',
        active_bounds=np.flatnonzero(np.minimum(abs(parameters-lower), abs(parameters-upper)) < 1e-7).tolist(),
        PL1_initialization='independent PL1 image-fit estimate, not generating optics',
        IL3_initialization=previous_row, all_thirteen_coordinates_released=True,
        calibration_posterior_added_as_prior=False)


def information(problem, parameters):
    jac = np.asarray(problem.jacobian(jnp.asarray(parameters)))
    norms = np.linalg.norm(jac, axis=0)
    _, singular, right = np.linalg.svd(jac/norms, full_matrices=False)
    rank = int(np.sum(singular > singular[0]*1e-10))
    result = dict(numerical_rank=rank, free_coordinates=13, normalized_singular_values=singular.tolist(),
                  rank_relative_tolerance=1e-10, no_directions_dropped=True)
    if rank != 13:
        return dict(result, parameter_covariance=None, physical_covariance=None)
    factor = (right.T/singular)/norms[:, None]
    transform = np.asarray(jax.jacfwd(stage_coordinates)(jnp.asarray(parameters)))
    phys = transform@factor
    return dict(result, parameter_covariance=(factor@factor.T).tolist(), physical_covariance=(phys@phys.T).tolist(),
        physical_standard_errors=np.linalg.norm(phys, axis=1).tolist(),
        interpretation='Local untruncated Gauss-Newton fixed-noise covariance, all13 free; no profile/global claim.')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_paired(path):
    path = Path(path).resolve()
    report = json.loads(path.read_text())
    paths = {path}
    for name, checksum in report['source_hashes'].items():
        source = ROOT/name
        if digest(source) != checksum:
            raise ValueError(f'Stale paired response input:{name}')
        paths.add(source)
    arrays_path = ROOT/report['arrays_path']
    if digest(arrays_path) != report['arrays_sha256']:
        raise ValueError('Paired NPZ checksum mismatch.')
    paths.add(arrays_path)
    if not report['paired_data_admitted'] or not report['calibration_response_noise_independent']:
        raise ValueError('Paired image admission and independent calibration noise are required.')
    for check in report['independent_gaussian_checks']:
        if not check['passed']:
            raise ValueError('Independent paired Gaussian validation failed.')
        if 'arrays_path' in check:
            check_path = ROOT/check['arrays_path']
            if digest(check_path) != check['arrays_sha256']:
                raise ValueError('Paired Gaussian validation checksum mismatch.')
            paths.add(check_path)
    with np.load(arrays_path, allow_pickle=False) as saved:
        names = ('response', 'joint_response_covariance', 'pl1_controls', 'il3_controls', 'training_mask', 'heldout_mask')
        data = {key: saved[key].copy() for key in names}
    training, heldout = data['training_mask'], data['heldout_mask']
    if training.dtype != bool or heldout.dtype != bool or np.any(training & heldout) or not np.all(training | heldout):
        raise ValueError('Training and heldout masks must be disjoint and exhaustive.')
    return report, data, paths


def run_fit(pl1_image_report, pl1_fit_report, paired_report, output, *, maximum_span_m=.9, max_nfev=800, initializer_max_nfev=300):
    if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
        raise RuntimeError('Run the JAX coupled fit on GPU float64.')
    started = perf_counter()
    calibration, base_train, base_hold, base_split, paths, base_image = load_image_responses(pl1_image_report)
    paired, data, paired_paths = load_paired(paired_report)
    paths.update(paired_paths)
    pl1_fit_report = Path(pl1_fit_report).resolve()
    previous_fit = json.loads(pl1_fit_report.read_text())
    for name, checksum in previous_fit['source_hashes'].items():
        if digest(ROOT/name) != checksum:
            raise ValueError(f'Stale PL1 initializer: {name}')
        paths.add(ROOT/name)
    # A clean inference-only manifest may repackage the same response data.
    # Compare actual training observations/covariance, not filenames, and do
    # not read its simulation truth to establish initializer compatibility.
    calibrated = previous_fit['analyser_calibration']
    for current, key in zip(base_train, ('training_controls', 'training_response', 'training_response_covariance')):
        if not np.array_equal(np.asarray(current), np.asarray(calibrated[key])):
            raise ValueError('PL1 initializer belongs to different training response data.')
    if tuple(paired['input_steps_grouped']) != calibration.input_steps_grouped or paired['detector_pixel_m'] != calibration.pixel_m:
        raise ValueError('The two acquisition blocks need the same calibrated input/detector convention.')
    paths.update((pl1_fit_report, Path(__file__), ROOT/'src/temgym_core/projector_response_tomography.py'))
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}
    model = CoupledProjectorResponse(calibration)
    def subset(mask):
        indices = np.flatnonzero(np.repeat(mask, 16))
        controls = np.array([[p, u] for u in data['il3_controls'][mask] for p in data['pl1_controls']])
        return controls, data['response'][mask].reshape(-1, 2, 4), data['joint_response_covariance'][np.ix_(indices, indices)]
    paired_train, paired_hold = subset(data['training_mask']), subset(data['heldout_mask'])
    controls = np.concatenate((np.column_stack((base_train[0], np.zeros(len(base_train[0])))), paired_train[0]))
    observations = np.concatenate((base_train[1], paired_train[1]))
    covariance = block_diagonal([*base_train[2], paired_train[2]])
    problem = JointResponseProblem(model, controls, observations, covariance)
    recovered = reconstruct_map_scan(data['response'][data['training_mask']],
        previous_fit['analyser_calibration']['coordinates'], data['pl1_controls'], calibration.input_steps_grouped, calibration.pixel_m,
        paired_train[2], previous_fit['analyser_calibration']['marginal_covariance'],
        calibration_data_independent=True, balance_length_m=.01)
    initializers, init_diagnostics = initialize_previous_stage(recovered, data['il3_controls'][data['training_mask']], max_nfev=initializer_max_nfev)
    rows = []
    for index, initializer in enumerate(initializers):
        fit = fit_joint_start(problem, previous_fit, initializer, maximum_span_m=maximum_span_m, max_nfev=max_nfev)
        fit['start_index'] = index
        rows.append(fit)
        print('coupled start', index, fit['status'], fit.get('chi_square'), fit.get('nfev'), flush=True)
    eligible = [row for row in rows if row['optimizer_success']]
    best = min(eligible, key=lambda row: row['chi_square']) if eligible else None
    diagnostic, heldout, truth = {}, {}, {}
    if best is not None:
        p = np.asarray(best['parameters'])
        diagnostic = information(problem, p)
        hold_controls = np.concatenate((np.column_stack((base_hold[0], np.zeros(len(base_hold[0])))), paired_hold[0]))
        hold_data = np.concatenate((base_hold[1], paired_hold[1]))
        hold_cov = block_diagonal([*base_hold[2], paired_hold[2]])
        prediction = np.asarray(model.predict(p, hold_controls))
        difference = (prediction-hold_data).reshape(-1)
        white = np.linalg.solve(np.linalg.cholesky(hold_cov), difference)
        heldout = dict(controls=hold_controls.tolist(), response=hold_data.tolist(), response_covariance=hold_cov.tolist(),
            prediction=prediction.tolist(), chi_square=float(white@white), response_coordinate_count=len(white),
            rms_response_noise_sigma=float(np.sqrt(np.mean(white**2))), maximum_absolute_response_error_pixels=float(np.max(np.abs(difference))),
            used_for_initialization_or_fit=False, PL1_standalone_count=len(base_hold[0]), paired_IL3_setting_count=int(data['heldout_mask'].sum()))
        if diagnostic.get('parameter_covariance') is not None:
            jac = np.asarray(jax.jacfwd(lambda pp: model.predict(pp, hold_controls).reshape(-1))(p))
            predictive = hold_cov+jac@np.asarray(diagnostic['parameter_covariance'])@jac.T
            white_pred = np.linalg.solve(np.linalg.cholesky(predictive), difference)
            heldout['predictive_chi_square'] = float(white_pred@white_pred)
        # Generating optics are used only after every fit and heldout prediction.
        truth_path = ROOT/paired['truth_path']
        if digest(truth_path) != paired['truth_sha256']:
            raise ValueError('Postfit paired truth checksum mismatch.')
        generating = json.loads(truth_path.read_text())
        last_truth, previous_truth = generating['PL1'], generating['IL3']
        last = np.array([np.log(last_truth['length_m']), np.log(last_truth['detector_gap_m']), last_truth['q_bias'], last_truth['q_slope_per_control']])
        previous = np.array([np.log(previous_truth['length_m']), np.log(previous_truth['gap_after_exit_m']), previous_truth['q_bias'], previous_truth['q_slope_per_control']])
        prefix = physical_upstream_coordinates(np.asarray(generating['upstream_to_IL3_mechanical']), calibration)
        truth_parameters = encode_stages(last, previous, prefix)
        delta = np.asarray(stage_coordinates(p)-stage_coordinates(truth_parameters))
        expected = np.asarray(model.predict(truth_parameters,
            np.array([[p1, u] for u in data['il3_controls'] for p1 in data['pl1_controls']]))).reshape(data['response'].shape)
        truth = dict(path=paired['truth_path'], sha256=paired['truth_sha256'], accessed_only_after_fit=True,
            parameters=truth_parameters.tolist(), physical_errors=delta.tolist(),
            lengths_gaps_errors_percent=(100*np.expm1(delta[[0,1,4,5]])).tolist(),
            noiseless_response_closure_max_abs_pixels=float(np.max(np.abs(expected-np.asarray(generating['expected_response_pixels'])))),
            sample_kappa_m_inv=float(truth_parameters[12]/calibration.balance_length_m))
        if diagnostic.get('physical_covariance') is not None:
            sigma = np.sqrt(np.diag(np.asarray(diagnostic['physical_covariance'])))
            truth.update(physical_errors_sigma=(delta/sigma).tolist(), lengths_gaps_standard_errors_percent=(100*sigma[[0,1,4,5]]).tolist())
        hashes[paired['truth_path']] = digest(truth_path)
    if any(digest(ROOT/name) != value for name, value in hashes.items()):
        raise RuntimeError('A coupled-fit dependency changed during execution.')
    report = dict(schema='neoarm-coupled-physical-PL1-IL3-v1', source_hashes=hashes, source_hashes_unchanged_during_execution=True,
        backend=jax.default_backend(), x64_enabled=True, jax_version=jax.__version__, elapsed_seconds=perf_counter()-started,
        pl1_image_report=str(Path(pl1_image_report).resolve().relative_to(ROOT)), paired_image_report=str(Path(paired_report).resolve().relative_to(ROOT)),
        pl1_arrays_path=base_image['arrays_path'], pl1_arrays_sha256=base_image['arrays_sha256'],
        paired_arrays_path=paired['arrays_path'], paired_arrays_sha256=paired['arrays_sha256'],
        parameter_names=PARAMETER_NAMES, physical_coordinate_names=PHYSICAL_NAMES, balance_length_m=calibration.balance_length_m,
        conditional_combined_span_bounds_m=[.002, maximum_span_m], geometry_allowance_independently_verified=False,
        calibrated_input_steps_grouped=list(calibration.input_steps_grouped), detector_pixel_m=calibration.pixel_m,
        training_controls=controls.tolist(), training_response=observations.tolist(), training_response_covariance=covariance.tolist(),
        training_response_coordinate_count=observations.size, PL1_training_split=base_split,
        paired_training_indices=np.flatnonzero(data['training_mask']).tolist(), paired_heldout_indices=np.flatnonzero(data['heldout_mask']).tolist(),
        initialization=init_diagnostics, initializers=initializers, rows=rows, best=best,
        start_count=len(rows), converged_start_count=len(eligible), unresolved_start_count=len(rows)-len(eligible),
        information=diagnostic, heldout=heldout, truth_diagnostic=truth,
        all_thirteen_coordinates_released=True, calibration_posterior_added_as_prior=False,
        final_likelihood_uses_original_responses=True, generating_upstream_used_for_initialization=False,
        heldout_used_for_initialization_or_fit=False, global_uniqueness_proved=False,
        assumptions=['Round upstream canonical SL2/rotation plus unknown mechanical magnetic shear; source covariance is not granted.',
            'Both lenses have exact hard-edge solenoid bodies and field-free gaps; lens tails/astigmatic mean maps are separate model tests.',
            'Four sample basis calibrations and detector pitch are exact here; systematic shared-scale uncertainty remains.',
            'PL1 branch is conditionally inherited from the independently tested negative first branch, while every parameter remains free.',
            'Six pi-wide IL3 intervals, both slope signs and two provisional geometry seeds do not prove global uniqueness.',
            'Relative-map fits are initialization only; the final full covariance likelihood uses original independent image responses.',
            'There is no calibration posterior prior: the standalone PL1 images occur exactly once in the joint likelihood.',
            'The combined .9m span cap is a conditional gross allowance, not verified geometry.',
            'Failed starts remain unresolved and bound-active alternatives need profile likelihoods.'])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pl1-images', type=Path, required=True)
    parser.add_argument('--pl1-fit', type=Path, required=True)
    parser.add_argument('--paired-images', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--maximum-span-m', type=float, default=.9)
    parser.add_argument('--max-nfev', type=int, default=800)
    parser.add_argument('--initializer-max-nfev', type=int, default=300)
    args = parser.parse_args()
    jax.config.update('jax_enable_x64', True)
    result = run_fit(args.pl1_images, args.pl1_fit, args.paired_images, args.output,
        maximum_span_m=args.maximum_span_m, max_nfev=args.max_nfev, initializer_max_nfev=args.initializer_max_nfev)
    print('Saved', args.output, 'best', None if result['best'] is None else result['best']['chi_square'], flush=True)
