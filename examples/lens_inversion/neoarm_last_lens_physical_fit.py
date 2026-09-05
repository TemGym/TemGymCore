"""Data-initialized PL1 tomography with a physically round upstream map.

Nine coordinates replace the 19-coordinate positive-volume relaxation: last
lens length/gap/excitation (four), round SL(2,R) upstream transport (three),
constant Larmor rotation (one), and unknown sample magnetic shear (one).
The source covariance is neither used nor fixed by this centroid-response fit.

This is a conditional hard-edge/field-free-gap model. A finite branch search,
fixed input calibration, and a provisional geometry cap do not prove global
uniqueness or validate this lens shape on the instrument. The likelihood is
the full correlated image-derived Gaussian *response* likelihood, not the
raw-pixel likelihood. Held-out acquisitions never enter initialization.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from neoarm_last_lens_branch_search import ProfiledResponse, search_response_branches
from neoarm_last_lens_tomography import J2, J4, rotation, mechanical_shear, predict_response_parameters
from temgym_core.jax_least_squares import least_squares


ROOT = Path(__file__).resolve().parents[2]
PARAMETER_NAMES = ('log_total_span_m', 'log_gap_over_length', 'q_bias', 'q_slope',
                   'upstream_log_row_norm', 'upstream_row_angle', 'upstream_row_shear',
                   'upstream_rotation_rad', 'sample_kappa_times_balance_length')
ANALYSER_NAMES = ('log_length_m', 'log_gap_m', 'q_bias', 'q_slope')


@dataclass(frozen=True)
class RoundLensResponse:
    """Round canonical optics followed by a mechanical sample-coordinate map.

    Grouped coordinates are (x,y,theta_x,theta_y), with canonical = T mechanical
    and T=[[I,0],[kappa J2,I]]. The signed electron-optics convention gives final
    rotation phi-q. Gaussian propagation must use the canonical upstream map.

    The balanced SL2 matrix is [[r cos(w),r sin(w)],
    [h r cos(w)-sin(w)/r,h r sin(w)+cos(w)/r]]. Its determinant is exactly one;
    the first row never vanishes, so no chart changes are needed inside JIT.
    In SI, b=balance_length*b_balanced, c=c_balanced/balance_length.
    """

    input_steps_grouped: tuple
    pixel_m: float

    def __post_init__(self):
        steps = np.asarray(self.input_steps_grouped)
        if steps.shape != (4,) or not np.all(np.isfinite(steps) & (steps > 0)):
            raise ValueError('Four positive finite calibrated basis steps are required.')
        if not np.isfinite(self.pixel_m) or self.pixel_m <= 0:
            raise ValueError('A positive finite isotropic detector pitch is required.')

    @property
    def balance_length_m(self):
        steps = np.asarray(self.input_steps_grouped)
        return float(np.sqrt(np.prod(steps[:2])/np.prod(steps[2:])))

    def radial_matrix(self, parameters):
        p = jnp.asarray(parameters)
        r, w, h = jnp.exp(p[4]), p[5], p[6]
        c, s, ell = jnp.cos(w), jnp.sin(w), self.balance_length_m
        return jnp.array([[r*c, ell*r*s], [(h*r*c-s/r)/ell, h*r*s+c/r]])

    def upstream_matrix(self, parameters, *, coordinates='mechanical'):
        p = jnp.asarray(parameters)
        radial = self.radial_matrix(p)
        canonical = jnp.kron(radial, rotation(p[7]))
        if coordinates == 'canonical':
            return canonical
        if coordinates != 'mechanical':
            raise ValueError('Choose canonical or mechanical input coordinates.')
        return canonical@mechanical_shear(p[8]/self.balance_length_m)

    def predict(self, parameters, normalized_dac):
        p, t = jnp.asarray(parameters), jnp.atleast_1d(normalized_dac)
        if p.shape != (9,):
            raise ValueError('The physical round model has nine coordinates.')
        length, gap = jnp.exp(analyser_coordinates(p)[:2])
        q = p[2]+p[3]*t
        lens_a = jnp.cos(q)-(gap/length)*q*jnp.sin(q)
        lens_b = length*jnp.sinc(q/jnp.pi)+gap*jnp.cos(q)
        radial = self.radial_matrix(p)
        aa = lens_a*radial[0, 0]+lens_b*radial[1, 0]
        bb = lens_a*radial[0, 1]+lens_b*radial[1, 1]
        position = aa[:, None, None]*jnp.eye(2)+(p[8]/self.balance_length_m)*bb[:, None, None]*J2
        tilt = bb[:, None, None]*jnp.eye(2)
        response = rotation(p[7]-q)@jnp.concatenate((position, tilt), axis=2)
        return response*(jnp.asarray(self.input_steps_grouped)/self.pixel_m)

    def response_parameters(self, parameters):
        """Exact embedding in the old 19-coordinate relaxation, for tests."""
        p = jnp.asarray(parameters)
        length, gap = jnp.exp(analyser_coordinates(p)[:2])
        upstream = self.upstream_matrix(p)
        w = jnp.concatenate((-(gap/length)*upstream[:2], length*upstream[2:]))
        return jnp.concatenate((p[1:4], (w*(jnp.asarray(self.input_steps_grouped)/self.pixel_m)).reshape(-1)))


def analyser_coordinates(parameters):
    """[log L,log detector gap,q0,q'] with a stable positive span partition."""
    p = jnp.asarray(parameters)
    return jnp.array([p[0]-jax.nn.softplus(p[1]), p[0]-jax.nn.softplus(-p[1]), p[2], p[3]])


def seed_from_response_parameters(parameters, model, *, maximum_span_m=None):
    """Project data-derived QR coefficients into the round model, for a seed.

    The first projected radial row is preserved. The orthogonal part of the
    second row is replaced by that required for determinant one. This is an
    initializer, not a least-squares projection or a claim that noisy QR
    coefficients themselves are physically realizable. No truth is read.
    """
    p = np.asarray(parameters)
    if p.shape != (19,) or not np.all(np.isfinite(p)):
        raise ValueError('Expected nineteen finite response coefficients.')
    factors = np.asarray(model.input_steps_grouped)/model.pixel_m
    w = p[3:].reshape(4, 4)/factors
    sign, log_volume = np.linalg.slogdet(w)
    if sign <= 0:
        raise ValueError('The QR seed has nonpositive calibrated four-volume.')
    gap, ratio = np.exp(.5*log_volume), np.exp(p[0])
    length = gap/ratio
    upstream = np.concatenate((-w[:2]/ratio, w[2:]/length))
    ell = model.balance_length_m
    balance = np.diag([np.sqrt(ell)]*2+[1/np.sqrt(ell)]*2)
    balanced = np.linalg.solve(balance, upstream@balance)
    complex_block = lambda block: .5*(np.trace(block)+1j*(block[1, 0]-block[0, 1]))
    za, zb, zc, zd = [complex_block(balanced[r:r+2, c:c+2]) for r, c in ((0,0),(0,2),(2,0),(2,2))]
    phi = float(np.angle(zb if abs(zb) >= abs(zd) else zd))
    za, zb, zc, zd = np.exp(-1j*phi)*np.array([za, zb, zc, zd])
    a, b, c, d = za.real, zb.real, zc.real, zd.real
    row_norm = np.hypot(a, b)
    if not np.isfinite(row_norm) or row_norm <= 1e-15 or b*b+d*d <= 1e-30:
        raise ValueError('The data-derived round seed has a degenerate radial row.')
    h = (a*c+b*d)/(row_norm**2)
    kappa_ell = (b*za.imag+d*zc.imag)/(b*b+d*d)
    span = length+gap
    if maximum_span_m is not None:
        span = min(span, .95*maximum_span_m)
    return np.array([np.log(span), p[0], p[1], p[2], np.log(row_norm), np.arctan2(b, a), h, phi, kappa_ell])


class PhysicalResponseProblem:
    def __init__(self, model, controls, response, covariance):
        covariance = np.asarray(covariance)
        if covariance.shape != (len(controls), 8, 8) or not np.all(np.isfinite(covariance)):
            raise ValueError('Finite per-setting 8x8 covariance matrices are required.')
        if not np.allclose(covariance, covariance.swapaxes(-1, -2), rtol=1e-12, atol=1e-18):
            raise ValueError('Response covariance must be symmetric.')
        self.model = model
        self.profiled = ProfiledResponse(controls, response, np.linalg.cholesky(covariance))
        self.residual = jax.jit(lambda p: self.profiled.whiten(
            (model.predict(p, self.profiled.controls)-self.profiled.observed).reshape(len(controls), 8, 1)).reshape(-1))
        self.jacobian = jax.jit(jax.jacfwd(self.residual))


def fit_physical_start(problem, qr_row, *, maximum_span_m=.9, max_nfev=500):
    """Fit all nine coordinates; retain seed failures and unclosed starts."""
    row = dict(qr_initial=qr_row['initial'], qr_parameters=qr_row['parameters'],
               qr_optimizer_success=qr_row['optimizer_success'], qr_chi_square=qr_row['chi_square'],
               q_bias_interval=qr_row['q_bias_interval'], slope_sign=qr_row['slope_sign'])
    try:
        initial = seed_from_response_parameters(qr_row['parameters'], problem.model, maximum_span_m=maximum_span_m)
    except ValueError as exc:
        return dict(row, optimizer_success=False, status='unresolved_seed_projection', reason=str(exc))
    # Explicit broad synthetic bounds; none are instrument-authorized currents.
    lower = np.array([np.log(.001), np.log(.2), qr_row['lower'][1], qr_row['lower'][2], -15., -4*np.pi, -1e4, -4*np.pi, -1e4])
    upper = np.array([np.log(maximum_span_m), np.log(200.), qr_row['upper'][1], qr_row['upper'][2], 15., 4*np.pi, 1e4, 4*np.pi, 1e4])
    initial = np.clip(initial, lower+1e-12*(upper-lower), upper-1e-12*(upper-lower))
    jac = np.asarray(problem.jacobian(initial))
    norms = np.linalg.norm(jac, axis=0)
    if not np.all(np.isfinite(norms) & (norms > 0)):
        return dict(row, initial=initial.tolist(), optimizer_success=False, status='unresolved_nonfinite_seed_jacobian')
    result = least_squares(problem.residual, initial, bounds=(lower, upper), x_scale=1/norms,
        max_nfev=max_nfev, max_iter=1000, ftol=1e-11, xtol=1e-11, gtol=1e-7)
    p = np.asarray(result.x)
    gradient = np.asarray(result.jacobian).T@np.asarray(result.residual)
    final_norms = np.linalg.norm(np.asarray(result.jacobian), axis=0)
    geometry = np.asarray(analyser_coordinates(p))
    return dict(row, initial=initial.tolist(), bounds=[lower.tolist(), upper.tolist()], parameters=p.tolist(),
        optimiser_parameter_scale=(1/norms).tolist(), optimizer_success=bool(result.success),
        status='tested_local_fit' if result.success else 'unresolved_optimizer_failure',
        termination=result.status, nfev=result.nfev, chi_square=float(2*result.cost),
        column_normalized_gradient_max=float(np.max(np.abs(gradient)/final_norms)),
        active_bounds=np.flatnonzero(np.minimum(abs(p-lower), abs(p-upper)) < 1e-7).tolist(),
        analyser_coordinates=geometry.tolist(), length_m=float(np.exp(geometry[0])), gap_m=float(np.exp(geometry[1])),
        sample_kappa_m_inv=float(p[8]/problem.model.balance_length_m))


def local_information(problem, parameters):
    """Untruncated, fixed-noise covariance; bound-active fits need profiles."""
    jac = np.asarray(problem.jacobian(jnp.asarray(parameters)))
    norms = np.linalg.norm(jac, axis=0)
    _, singular, right = np.linalg.svd(jac/norms, full_matrices=False)
    rank = int(np.sum(singular > singular[0]*1e-10))
    result = dict(jacobian_column_norms=norms.tolist(), normalized_jacobian_singular_values=singular.tolist(),
                  numerical_rank=rank, relative_rank_tolerance=1e-10, nuisance_coordinates_released=9)
    if rank != 9:
        return dict(result, parameter_covariance=None, analyser_covariance=None,
                    covariance_interpretation='Unresolved rank deficiency; no truncated finite covariance is reported.')
    factor = (right.T/singular)/norms[:, None]
    transform = np.asarray(jax.jacfwd(analyser_coordinates)(jnp.asarray(parameters)))
    analyser_factor = transform@factor
    return dict(result, parameter_covariance=(factor@factor.T).tolist(),
        analyser_covariance=(analyser_factor@analyser_factor.T).tolist(),
        analyser_standard_errors=np.linalg.norm(analyser_factor, axis=1).tolist(),
        covariance_interpretation='Local Gauss-Newton fixed-noise marginal covariance, all nine coordinates released; not a profile or global guarantee.')


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_image_responses(report_path):
    """Verify evidence; split acquisitions before any seeding or optical fit."""
    path = Path(report_path).resolve()
    report = json.loads(path.read_text())
    paths = {path, Path(__file__), Path(__file__).with_name('neoarm_last_lens_branch_search.py'),
             Path(__file__).with_name('neoarm_last_lens_tomography.py'), ROOT/'src/temgym_core/jax_least_squares.py'}
    for name, checksum in report['source_hashes'].items():
        source = ROOT/name
        if _digest(source) != checksum:
            raise ValueError(f'Stale image-response evidence: {name}')
        paths.add(source)
    arrays_path = ROOT/report['arrays_path']
    if _digest(arrays_path) != report['arrays_sha256']:
        raise ValueError('Image NPZ checksum mismatch.')
    paths.add(arrays_path)
    checks = report.get('independent_gaussian_checks', [])
    if not checks or not all(check['passed'] for check in checks):
        raise ValueError('Independent Gaussian validation must pass before optical fitting.')
    for check in checks:
        if 'arrays_path' in check:
            check_path = ROOT/check['arrays_path']
            if _digest(check_path) != check['arrays_sha256']:
                raise ValueError('Independent Gaussian-check NPZ checksum mismatch.')
            paths.add(check_path)
    with np.load(arrays_path, allow_pickle=False) as saved:
        controls, response, covariance = [saved[key].copy() for key in ('accepted_normalized_dac', 'response', 'response_covariance')]
        steps = np.asarray(report.get('input_steps_grouped', np.max(np.abs(saved['input_states_grouped']), axis=0)))
        if 'heldout_response' in saved:
            train = (controls, response, covariance)
            heldout = tuple(saved[key].copy() for key in ('heldout_normalized_dac', 'heldout_response', 'heldout_response_covariance'))
            split = dict(kind='independent_heldout_arrays', training_original_indices=list(range(len(controls))),
                         heldout_original_indices=list(range(len(heldout[0]))), separate_acquisition_arrays=True)
        else:
            hold_mask = np.any(np.isclose(controls[:, None], np.array([-.75, -.25, .25, .75])[None, :], rtol=0, atol=1e-10), axis=1)
            if hold_mask.sum() != 4:
                raise ValueError('Legacy holdout split requires the four predeclared control settings.')
            train = (controls[~hold_mask], response[~hold_mask], covariance[~hold_mask])
            heldout = (controls[hold_mask], response[hold_mask], covariance[hold_mask])
            split = dict(kind='predeclared_control_holdout_before_seeding',
                training_original_indices=np.flatnonzero(~hold_mask).tolist(), heldout_original_indices=np.flatnonzero(hold_mask).tolist(),
                predeclared_heldout_controls=[-.75, -.25, .25, .75], separate_acquisition_arrays=False)
    if 'detector_pixel_m' in report:
        pixel = float(report['detector_pixel_m'])
    else:
        pitches = np.asarray(report['detector']['pixel_size_yx_m'])
        if not np.allclose(pitches, pitches[0], rtol=1e-14, atol=0):
            raise ValueError('This response model requires isotropic calibrated detector pixels.')
        if not np.allclose(report['detector']['affine_xy'], np.eye(2), rtol=0, atol=1e-14):
            raise ValueError('Undo the detector affine calibration before using this response model.')
        pixel = float(pitches[0])
    return RoundLensResponse(tuple(steps), pixel), train, heldout, split, paths, report


def run_fit(image_report_path, output_path, *, truth_path=None, maximum_span_m=.9, max_nfev=500, qr_max_nfev=250):
    if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
        raise RuntimeError('Run the differentiable physical response fit on GPU float64.')
    if not np.isfinite(maximum_span_m) or maximum_span_m <= .001:
        raise ValueError('The conditional span cap must exceed the 1 mm lower bound.')
    start = perf_counter()
    model, train, heldout, split, paths, image_report = load_image_responses(image_report_path)
    hashes = {str(path.relative_to(ROOT)): _digest(path) for path in sorted(paths)}
    problem = PhysicalResponseProblem(model, *train)
    qr_rows = search_response_branches(problem.profiled, max_nfev=qr_max_nfev)
    rows = []
    for index, qr_row in enumerate(qr_rows):
        row = fit_physical_start(problem, qr_row, maximum_span_m=maximum_span_m, max_nfev=max_nfev)
        row['start_index'] = index
        rows.append(row)
        print('physical start', index, row['status'], row.get('chi_square'), row.get('nfev'), flush=True)
    eligible = [row for row in rows if row['optimizer_success']]
    best = min(eligible, key=lambda row: row['chi_square']) if eligible else None
    information, validation, analyser, truth = {}, {}, {}, {}
    if best is not None:
        fitted = np.asarray(best['parameters'])
        information = local_information(problem, fitted)
        prediction = np.asarray(model.predict(fitted, heldout[0]))
        residual = prediction-heldout[1]
        chol = np.linalg.cholesky(heldout[2])
        whitened = np.linalg.solve(chol, residual.reshape(len(heldout[0]), 8, 1)).reshape(-1)
        validation = dict(controls=np.asarray(heldout[0]).tolist(), observed_response=np.asarray(heldout[1]).tolist(),
            response_covariance=np.asarray(heldout[2]).tolist(), predicted_response=prediction.tolist(),
            chi_square=float(whitened@whitened), response_coordinate_count=len(whitened),
            rms_response_noise_sigma=float(np.sqrt(np.mean(whitened**2))),
            maximum_absolute_response_error_pixels=float(np.max(np.abs(residual))), used_for_fit_or_initialization=False)
        if information['parameter_covariance'] is not None:
            jac = np.asarray(jax.jacfwd(lambda p: model.predict(p, heldout[0]).reshape(-1))(fitted))
            measurement_cov = np.zeros((len(whitened), len(whitened)))
            for index, block in enumerate(heldout[2]):
                measurement_cov[8*index:8*index+8, 8*index:8*index+8] = block
            predictive_cov = measurement_cov+jac@np.asarray(information['parameter_covariance'])@jac.T
            pred_white = np.linalg.solve(np.linalg.cholesky(predictive_cov), residual.reshape(-1))
            validation.update(predictive_chi_square=float(pred_white@pred_white),
                              predictive_covariance_includes_cross_setting_parameter_uncertainty=True)
        analyser = dict(coordinate_names=ANALYSER_NAMES, coordinates=best['analyser_coordinates'],
            marginal_covariance=information.get('analyser_covariance'), calibrated_input_steps_grouped=list(model.input_steps_grouped),
            detector_pixel_m=model.pixel_m, training_controls=np.asarray(train[0]).tolist(),
            training_response=np.asarray(train[1]).tolist(), training_response_covariance=np.asarray(train[2]).tolist(),
            **split, heldout=validation)
        # Truth remains unread until every start and heldout prediction is done.
        if truth_path is None and image_report.get('truth_path'):
            truth_path = ROOT/image_report['truth_path']
        if truth_path is not None:
            truth_path = Path(truth_path).resolve()
            if image_report.get('truth_sha256') and _digest(truth_path) != image_report['truth_sha256']:
                raise ValueError('Postfit truth checksum mismatch.')
            raw_truth = json.loads(truth_path.read_text())
            truth_inputs = raw_truth.get('inputs', raw_truth)
            generating = np.asarray(truth_inputs.get('truth_parameters', truth_inputs.get('generating19parameters')))
            if generating.shape != (19,):
                raise ValueError('Postfit truth needs truth_parameters or generating19parameters with shape (19,).')
            truth_physical = seed_from_response_parameters(generating, model)
            expected19 = np.asarray(predict_response_parameters(generating, train[0]))
            expected9 = np.asarray(model.predict(truth_physical, train[0]))
            closure = np.linalg.solve(np.linalg.cholesky(train[2]), (expected9-expected19).reshape(len(train[0]), 8, 1))
            delta = np.asarray(analyser_coordinates(fitted)-analyser_coordinates(truth_physical))
            truth = dict(path=str(truth_path.relative_to(ROOT)), sha256=_digest(truth_path), accessed_only_after_fit=True,
                parameters=truth_physical.tolist(), noiseless_truth_closure_max_abs_pixels=float(np.max(np.abs(expected9-expected19))),
                noiseless_truth_closure_max_abs_sigma=float(np.max(np.abs(closure))),
                analyser_errors=delta.tolist(), length_gap_errors_percent=(100*np.expm1(delta[:2])).tolist(),
                sample_kappa_m_inv=float(truth_physical[8]/model.balance_length_m))
            if information.get('analyser_covariance') is not None:
                sigmas = np.sqrt(np.diag(np.asarray(information['analyser_covariance'])))
                truth.update(analyser_errors_sigma=(delta/sigmas).tolist(),
                             length_gap_standard_errors_percent=(100*sigmas[:2]).tolist())
            hashes[str(truth_path.relative_to(ROOT))] = _digest(truth_path)
    if any(_digest(ROOT/name) != digest for name, digest in hashes.items()):
        raise RuntimeError('Evidence changed during the physical fit.')
    report = dict(schema='neoarm-last-lens-physical-round-fit-v1', source_hashes=hashes,
        source_hashes_unchanged_during_execution=True, image_report_path=str(Path(image_report_path).resolve().relative_to(ROOT)),
        image_report_sha256=_digest(image_report_path), arrays_path=image_report['arrays_path'], arrays_sha256=image_report['arrays_sha256'],
        backend=jax.default_backend(), jax_version=jax.__version__, x64_enabled=True, elapsed_seconds=perf_counter()-start,
        parameter_names=PARAMETER_NAMES, balance_length_m=model.balance_length_m,
        calibrated_input_steps_grouped=list(model.input_steps_grouped), detector_pixel_m=model.pixel_m,
        conditional_span_bounds_m=[.001, maximum_span_m], geometry_allowance_independently_verified=False,
        split=split, training_response_coordinate_count=int(np.asarray(train[1]).size),
        qr_start_count=len(qr_rows), qr_rows=qr_rows, physical_start_count=len(rows), rows=rows, best=best,
        converged_start_count=len(eligible), unresolved_start_count=len(rows)-len(eligible),
        information=information, analyser_calibration=analyser, heldout=validation, truth_diagnostic=truth,
        all_nine_physical_coordinates_released=True, round_upstream_enforced=True, magnetic_symplectic_upstream_enforced=True,
        source_covariance_used_by_optics_fit=False, generating_optics_used_for_initialization=False,
        heldout_used_for_initialization_or_fitting=False, global_uniqueness_proved=False,
        equivalent_coordinate_gauge='Add pi to upstream row angle and upstream rotation simultaneously; the physical upstream map is unchanged.',
        calibration_scale_gauge=dict(
            ignored_shared_gains='L and gap are multiplied by position_gain*tilt_gain/pixel_pitch_gain**2; gap/length is unchanged.',
            log_length_gap_systematic_gradient=[1., 1., -2.],
            gain_coordinate_order=['log_shared_position_gain', 'log_shared_tilt_gain', 'log_pixel_pitch_gain'],
            sample_kappa_gain='position_gain/tilt_gain',
            example_independent_position_tilt_gain_sigma_fraction=.002,
            example_shared_log_length_gap_sigma_fraction=float(np.sqrt(2)*.002),
            gain_covariance_included_in_reported_fit_covariance=False),
        assumptions=['The upstream mean-ray map is round; astigmatic source covariance does not invalidate this constraint.',
            'The final lens is a hard-edge solenoid followed by a field-free detector gap; objective tails/overlap remain a model check.',
            'Four sample mechanical basis inputs, isotropic detector pitch and excitation commands are exactly calibrated here.',
            'The unknown sample magnetic shear is fitted, not supplied by generating truth.',
            'Per-setting response errors have their full image-derived covariance; cross-setting input-calibration errors are not included.',
            'The provisional 0.9 m span cap is conditional, not a measured instrument dimension.',
            'Eight pi-wide phase intervals, both slope signs and two initial ratios form a finite search, not a proof of global exclusion.',
            'Every failed seed/start remains unresolved; covariance is local and cannot replace profile likelihoods.',
            'No upstream truth or heldout response enters QR seeding or physical optimization.'])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--truth', type=Path)
    parser.add_argument('--maximum-span-m', type=float, default=.9)
    parser.add_argument('--max-nfev', type=int, default=500)
    parser.add_argument('--qr-max-nfev', type=int, default=250)
    args = parser.parse_args()
    jax.config.update('jax_enable_x64', True)
    result = run_fit(args.image_report, args.output, truth_path=args.truth,
        maximum_span_m=args.maximum_span_m, max_nfev=args.max_nfev, qr_max_nfev=args.qr_max_nfev)
    print('Saved', args.output, 'best chi2', None if result['best'] is None else result['best']['chi_square'], flush=True)
