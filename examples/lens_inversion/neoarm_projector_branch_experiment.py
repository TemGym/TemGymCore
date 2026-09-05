"""A frozen, fresh image challenge for two image-fitted IL3 branches.

Design reads the two fitted hypotheses, not generating optics. A separate
forward stage then uses the unchanged synthetic column to acquire new images.
The reported design separations are conditional local predictions, not power
guarantees after nonlinear refitting or evidence of instrument feasibility.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import os
import time
import tomllib

os.environ.setdefault('JAX_ENABLE_X64', 'true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import jax
import jax.numpy as jnp
import numpy as np

from neoarm_coupled_projector_fit import (
    CoupledProjectorResponse, stage_coordinates, grouped_stage, block_diagonal,
)
from neoarm_last_lens_physical_fit import RoundLensResponse
from neoarm_last_lens_design import make_relay_model, GROUPED
from neoarm_last_lens_images import image_initializer, signed_response_from_image_fits, render_pure_counts
from neoarm_last_lens_redesigned_images import truth_response_likelihood
from neoarm_pl1_il3_images import assemble_response_covariance
from neoarm_last_lens_tomography import mechanical_shear
from neoarm_mixed_gaussian_validation import MixedGaussianSource
from neoarm_imaging_recipe import column_with_geometry_anchors, single_ray_transfer
from neoarm_inference_manifest import export as export_inference
from temgym_core.gaussian_image_likelihood import (
    make_gaussian_detector_operator, fit_gaussian_image,
    encode_gaussian_image_parameters, gaussian_image_information_factor,
)
from temgym_core.microscope_inversion import drift_matrix, hard_edge_solenoid_matrix
from temgym_core.neoarm_calibration import DetectorGeometry, hard_edge_column_from_config

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT/'examples/generated/neoarm_inversion/imaging_recipe'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_report(path):
    report = json.loads(Path(path).read_text())
    for name, checksum in report['source_hashes'].items():
        if digest(ROOT/name) != checksum:
            raise ValueError(f'Stale branch-experiment source: {name}')
    return report


def signed_inputs(steps):
    result = np.zeros((9, 4))
    for i, step in enumerate(steps):
        result[1+2*i, i], result[2+2*i, i] = step, -step
    return result


def pure_source():
    return MixedGaussianSource(np.diag(np.array([2e-9, 1.5e-9])**2),
        np.zeros((2, 2)), np.zeros((4, 4)), np.zeros(4))


def hypothesis_moments(model, parameters, controls, source):
    """Canonical source covariance and mechanical calibrated input means."""
    p = jnp.asarray(parameters)
    physical = stage_coordinates(p)
    upstream = model.upstream_matrix(p, coordinates='canonical')
    maps = jax.jit(jax.vmap(lambda c: grouped_stage(physical[:4], c[0])
        @ grouped_stage(physical[4:], c[1]) @ upstream))(jnp.asarray(controls))
    covariance = jnp.asarray(source.total_covariance)[np.ix_(GROUPED, GROUPED)]
    spatial_cov = maps[:, :2] @ covariance @ maps[:, :2].swapaxes(-1, -2)
    mechanical = maps @ mechanical_shear(p[12]/model.calibration.balance_length_m)
    means = jnp.einsum('nai,fi->nfa', mechanical[:, :2],
        jnp.asarray(signed_inputs(model.calibration.input_steps_grouped)))
    return np.asarray(means), np.asarray(spatial_cov)


def approximate_response_covariance(covariance_xy_m2, pixel_m, dose):
    """Full-camera centroid approximation, not an exact image information bound."""
    xy = (np.asarray(covariance_xy_m2)/pixel_m**2 + np.eye(2)*(.65**2+1/12))/(2*dose)
    result = np.zeros((len(xy), 8, 8))
    for a in range(2):
        for b in range(2):
            result[:, a*4:(a+1)*4, b*4:(b+1)*4] = xy[:, a, b, None, None]*np.eye(4)
    return result


def local_profile_separation(delta, jacobian, covariance, *, rank_tolerance=1e-10):
    """Linearized nuisance-refit residual; no nonlinear/global power claim."""
    chol = np.linalg.cholesky(covariance)
    white = np.linalg.solve(chol, np.asarray(delta).reshape(-1))
    jac = np.linalg.solve(chol, np.asarray(jacobian).reshape(len(white), -1))
    norms = np.linalg.norm(jac, axis=0)
    left, singular, _ = np.linalg.svd(jac/norms, full_matrices=False)
    rank = int(np.sum(singular > singular[0]*rank_tolerance))
    residual = white-left[:, :rank]@(left[:, :rank].T@white)
    return dict(fixed_hypothesis_separation=float(white@white),
        locally_profiled_separation=float(residual@residual), linearized_nuisance_rank=rank)


def distinct_partition(offsets, training, old_offsets):
    offsets, training = np.asarray(offsets), np.asarray(training)
    if offsets.shape != (4,) or training.shape != (4,) or training.dtype != bool or training.sum() != 2:
        raise ValueError('Exactly four offsets, two training and two held out, are required.')
    if len(np.unique(offsets)) != 4 or np.any(np.isclose(offsets[:, None], np.asarray(old_offsets)[None], rtol=0, atol=1e-10)):
        raise ValueError('All new offsets must be distinct from each other and all previous acquisitions.')
    return training, ~training


def design(output):
    if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
        raise RuntimeError('Branch experiment design/forward evaluation requires GPU float64.')
    best_path, rival_path = FOLDER/'coupled_projector_fit.json', FOLDER/'coupled_projector_branch_boundary.json'
    fitted, boundary = verify_report(best_path), verify_report(rival_path)
    rival = min((row for row in boundary['rows'] if row['optimizer_success'] and not row['active_bounds']),
        key=lambda row: row['chi_square'])
    parameters = [np.asarray(fitted['best']['parameters']), np.asarray(rival['parameters'])]
    model = CoupledProjectorResponse(RoundLensResponse(tuple(fitted['calibrated_input_steps_grouped']), fitted['detector_pixel_m']))
    analysers = np.array([-.5833333333333335, .8333333333333334])
    # Positive controls remain connected to the old nominal setting without
    # crossing the unadmitted negative-side subpixel waist of this analyser pair.
    offsets = np.arange(6, 16)*.05
    controls = np.array([[p, u] for u in offsets for p in analysers])
    old_controls = np.asarray(fitted['training_controls'])
    all_controls = np.concatenate((old_controls, controls))
    predictions = [np.asarray(model.predict(p, all_controls)) for p in parameters]
    jacobians = [np.asarray(jax.jit(jax.jacfwd(lambda pp: model.predict(pp, all_controls)))(p)) for p in parameters]
    means_cov = [hypothesis_moments(model, p, controls, pure_source()) for p in parameters]
    widths = [np.sqrt(np.linalg.eigvalsh(c))/model.calibration.pixel_m for _, c in means_cov]
    covariance = [approximate_response_covariance(c, model.calibration.pixel_m, 1e6) for _, c in means_cov]
    old_cov = np.asarray(fitted['training_response_covariance'])
    candidates = []
    for pair in itertools.combinations(range(len(offsets)), 2):
        selected = np.array([2*i+a for i in pair for a in range(2)])
        if min(w[selected].min() for w in widths) < .30:
            continue
        indices = np.r_[np.arange(len(old_controls)), len(old_controls)+selected]
        scores = [local_profile_separation((predictions[0]-predictions[1])[indices], jacobians[h][indices],
            block_diagonal([old_cov, *covariance[h][selected]])) for h in range(2)]
        candidates.append(dict(indices=list(pair), offsets=offsets[list(pair)].tolist(),
            minimum_local_profile_separation=min(s['locally_profiled_separation'] for s in scores), scores=scores))
    winner = max(candidates, key=lambda row: row['minimum_local_profile_separation'])
    train_offsets = offsets[winner['indices']]
    # Heldouts are chosen geometrically, before noise, not by selecting a winner
    # after seeing future data. They interpolate at 1/3 and 2/3 of the new span.
    held_offsets = np.linspace(offsets[0], offsets[-1], 4)[1:3]
    if np.any(np.isclose(held_offsets[:, None], train_offsets[None], rtol=0, atol=1e-10)):
        raise ValueError('The frozen geometric heldout rule collided with a selected training offset.')
    selected_offsets = np.sort(np.r_[train_offsets, held_offsets])
    old_offsets = np.unique(np.r_[old_controls[:, 1], np.asarray(fitted['heldout']['controls'])[:, 1]])
    training = np.any(np.isclose(selected_offsets[:, None], train_offsets[None], atol=1e-10, rtol=0), axis=1)
    training, heldout = distinct_partition(selected_offsets, training, old_offsets)
    final_controls = np.array([[p, u] for u in selected_offsets for p in analysers])
    detector = DetectorGeometry(z_m=1., mtf_sigma_px=.65)
    operator = jax.jit(make_gaussian_detector_operator(detector, 16))
    factor_fun = jax.jit(lambda p: gaussian_image_information_factor(p, detector, quadrature_order=16))
    exact = []
    for h, p in enumerate(parameters):
        means, cov = hypothesis_moments(model, p, final_controls, pure_source())
        state_cov, captures, rms = [], [], []
        for state, (mean_group, c) in enumerate(zip(means, cov)):
            pars, covs = [], []
            for mean in mean_group:
                encoded = encode_gaussian_image_parameters(mean, c, 1e6, detector)
                factor = np.asarray(factor_fun(encoded))
                inverse = np.linalg.solve(factor, np.eye(6))
                pars.append(np.asarray(encoded)); covs.append(inverse@inverse.T)
                captures.append(float(np.asarray(operator(mean, c, 1e6)).sum()/1e6))
            state_cov.append(signed_response_from_image_fits(pars, covs)[1])
            rms.append((np.sqrt(np.linalg.eigvalsh(c))/model.calibration.pixel_m).tolist())
        exact.append(dict(hypothesis=h, response_covariance=np.asarray(state_cov).tolist(),
            minimum_capture=min(captures), minimum_optical_rms_pixels=float(np.min(rms)),
            maximum_optical_rms_pixels=float(np.max(rms)),
            admitted=bool(min(captures) >= .99 and np.min(rms) >= .30)))
    final_predictions = [np.asarray(model.predict(p, final_controls)) for p in parameters]
    final_jac = [np.asarray(jax.jit(jax.jacfwd(lambda pp: model.predict(pp, final_controls)))(p)) for p in parameters]
    exact_scores = {}
    for label, mask in [('training', training), ('heldout', heldout), ('all', np.ones(4, bool))]:
        ix = np.flatnonzero(np.repeat(mask, 2))
        scores = []
        for h in range(2):
            delta = np.concatenate((predictions[0][:len(old_controls)]-predictions[1][:len(old_controls)],
                (final_predictions[0]-final_predictions[1])[ix]))
            jac = np.concatenate((jacobians[h][:len(old_controls)], final_jac[h][ix]))
            c = block_diagonal([old_cov, *np.asarray(exact[h]['response_covariance'])[ix]])
            scores.append(local_profile_separation(delta, jac, c))
        exact_scores[label] = scores
    sources = dict(fitted['source_hashes'])
    sources.update(boundary['source_hashes'])
    for path in (Path(__file__), best_path, rival_path, ROOT/'examples/lens_inversion/neoarm_inference_manifest.py'):
        sources[str(path.relative_to(ROOT))] = digest(path)
    report = dict(schema='neoarm-projector-branch-challenge-design-v1', backend=jax.default_backend(), source_hashes=sources,
        hypotheses_parameters=[p.tolist() for p in parameters], original_training_data_used=True,
        generating_optics_read_for_design=False, old_heldouts_used_to_choose_hypothesis=False,
        old_heldouts_are_now_previous_model_selection_evidence=True,
        candidate_il3_native_offsets=offsets.tolist(), approximate_candidates=candidates, selected_training_score=winner,
        pl1_controls=analysers.tolist(), il3_controls=selected_offsets.tolist(), training_mask=training.tolist(), heldout_mask=heldout.tolist(),
        exact_finite_pixel_hypothesis_admission=exact, exact_local_design_scores=exact_scores,
        input_steps_grouped=list(model.calibration.input_steps_grouped), detector_pixel_m=model.calibration.pixel_m,
        frozen_before_rendering_or_new_noise=True, incident_electrons_per_frame=1e6,
        all_hypotheses_admitted=all(row['admitted'] for row in exact),
        limitations=['Source covariance is a fixed quantum-valid 2nm/1.5nm pure canonical preparation for conditional image design only.',
            'Only the connected positive control interval0…+.75 is used; the negative-side pair crosses an unadmitted subpixel waist.',
            'Scores profile13 tangent coordinates locally using prior training response information; no nonlinear-refit power guarantee.',
            'Input calibration, detector geometry/PSF and stable alignment are conditional; no systematic-scale nuisance included.',
            'No numerical generating optical parameters or future image noise enter design. Hypothesis reports contain postfit truth diagnostics that are not consulted.',
            'The wider native controls are synthetic study coordinates, not instrument-authorized commands.'])
    if any(digest(ROOT/name) != value for name, value in sources.items()):
        raise RuntimeError('A source changed while freezing the branch design.')
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def acquire(design_path, output):
    """Generate from the unchanged physical column only AFTER design freezes."""
    if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
        raise RuntimeError('Fresh forward images require GPU float64.')
    started = time.perf_counter()
    design_path, output = Path(design_path).resolve(), Path(output).resolve()
    frozen = verify_report(design_path)
    if not frozen['all_hypotheses_admitted'] or not frozen['frozen_before_rendering_or_new_noise']:
        raise ValueError('The two-hypothesis finite-pixel preflight must pass before acquisition.')
    # This is the first numerical use of generating optics in this workflow.
    old_path = FOLDER/'pl1_il3_images.json'
    old = verify_report(old_path)
    truth_input = ROOT/old['truth_path']
    if digest(truth_input) != old['truth_sha256']:
        raise ValueError('The separate generating-model checksum is stale.')
    generating = json.loads(truth_input.read_text())
    config_path = ROOT/'examples/lens_inversion/neoarm_inversion_config.toml'
    geometry_path = ROOT/'examples/microscope_models/measure_microscope_geometry.ipynb'
    config = tomllib.loads(config_path.read_text())
    column = column_with_geometry_anchors(hard_edge_column_from_config(config), geometry_path)
    parameters = jnp.asarray(generating['physical_column_parameters'])
    _, decoded = make_relay_model(column, parameters, generating['generating_sample_magnetic_shear_per_m'])
    lens3, last = decoded[-2][0], decoded[-1][0]
    robust_path = FOLDER/'last_lens_robust_design.json'
    robust = verify_report(robust_path)
    stages = [generating['PL1'], generating['IL3']]
    prefix = jnp.asarray(generating['upstream_to_IL3_canonical'])
    pl1, il3 = np.asarray(frozen['pl1_controls']), np.asarray(frozen['il3_controls'])
    controls = np.array([[p, u] for u in il3 for p in pl1])
    training = np.asarray(frozen['training_mask'], bool)
    heldout = np.asarray(frozen['heldout_mask'], bool)
    distinct_partition(il3, training, old['il3_controls'])
    def physical_map(control):
        p, u = control
        q1 = stages[0]['q_bias']+stages[0]['q_slope_per_control']*p
        q3 = stages[1]['q_bias']+stages[1]['q_slope_per_control']*u
        return (drift_matrix(stages[0]['detector_gap_m'])
            @ hard_edge_solenoid_matrix(q1, stages[0]['length_m'])
            @ drift_matrix(stages[1]['gap_after_exit_m'])
            @ hard_edge_solenoid_matrix(q3, stages[1]['length_m']) @ prefix)
    matrices = np.asarray(jax.jit(jax.vmap(physical_map))(jnp.asarray(controls)))
    commands = np.tile(robust['synthetic_fixed_relay_dacs'], (len(controls), 1))
    commands[:, lens3.channel_index] += controls[:, 1]
    _, _, _, bias1, slope1 = decoded[-1]
    q1 = stages[0]['q_bias']+stages[0]['q_slope_per_control']*controls[:, 0]
    commands[:, last.channel_index] = last.reference_dac+(q1-float(bias1))/float(slope1)
    direct = np.asarray(jax.jit(jax.vmap(lambda u: single_ray_transfer(column, parameters, u)))(jnp.asarray(commands)))
    closure = float(np.max(abs(direct-matrices))/np.max(abs(direct)))
    if closure > 1e-12:
        raise ValueError('Actual column/control map and staged synthetic renderer disagree.')
    source = pure_source()
    steps = np.asarray(frozen['input_steps_grouped']); pixel = frozen['detector_pixel_m']
    input_states = signed_inputs(steps)
    shear = np.asarray(mechanical_shear(generating['generating_sample_magnetic_shear_per_m']))[np.ix_(GROUPED, GROUPED)]
    canonical_means = input_states[:, GROUPED]@shear.T
    spatial = matrices[:, [0, 2]]
    means = np.einsum('nai,fi->nfa', spatial, canonical_means)
    covariance = np.einsum('nai,ij,nbj->nab', spatial, source.total_covariance, spatial)
    widths = np.sqrt(np.linalg.eigvalsh(covariance))/pixel
    detector = DetectorGeometry(z_m=float(parameters[1]), mtf_sigma_px=.65)
    operator = jax.jit(make_gaussian_detector_operator(detector, 16))
    dose, read_noise = frozen['incident_electrons_per_frame'], .6
    sources = dict(frozen['source_hashes']); sources.update(old['source_hashes'])
    for path in (Path(__file__), design_path, old_path, truth_input, robust_path, config_path, geometry_path):
        sources[str(path.relative_to(ROOT))] = digest(path)
    names = ('expected_counts', 'noisy_counts', 'fitted_counts', 'variance_counts',
        'image_parameters', 'image_parameter_covariances', 'image_initial_parameters', 'response', 'response_covariance')
    arrays = {name:[] for name in names}; rows = []
    for state, (mean_group, c) in enumerate(zip(means, covariance)):
        tick = time.perf_counter(); group, analyser = divmod(state, 2)
        image_arrays = [[] for _ in range(4)]; pars, covs, initials, frames = [], [], [], []
        for frame, mean in enumerate(mean_group):
            expected = np.asarray(operator(mean, c, dose)); variance = expected+read_noise**2
            seed = 1040000+100*state+frame
            a, b = jax.random.split(jax.random.PRNGKey(seed))
            observed = np.asarray(jax.random.poisson(a, jnp.asarray(expected), shape=expected.shape)
                +read_noise*jax.random.normal(b, expected.shape))
            initial = image_initializer(observed, detector, read_noise)
            fit = fit_gaussian_image(observed, variance, detector, initial_parameters=initial,
                quadrature_order=16, max_nfev=100)
            failure = None
            try:
                fit.require_accepted()
            except ValueError as error:
                failure = str(error)
            capture = float(expected.sum()/dose)
            frame_admitted = bool(failure is None and widths[state].min() >= .25 and capture >= .99)
            frames.append(dict(frame=frame, noise_seed=seed, admitted=frame_admitted, failure=failure,
                capture_fraction=capture, optimizer_success=bool(fit.optimizer_result.success),
                optimizer_status=fit.optimizer_result.status, nfev=fit.optimizer_result.nfev,
                image_information_rank=fit.numerical_rank, residual_rms_sigma=fit.residual_rms_sigma,
                chi_square_z=fit.chi_square_z, quadrature_rms_sigma=fit.quadrature_rms_sigma,
                quadrature_parameter_shift_sigma=fit.quadrature_parameter_shift_sigma))
            pars.append(fit.parameters); covs.append(fit.parameter_covariance); initials.append(initial)
            for destination, value in zip(image_arrays, (expected, observed, fit.fitted_counts_yx, variance)):
                destination.append(value)
        response, response_cov = signed_response_from_image_fits(pars, covs)
        for name, value in zip(names, (*image_arrays, pars, covs, initials, response, response_cov)):
            arrays[name].append(value)
        rows.append(dict(group_index=group, analyser_index=analyser, il3_control=float(il3[group]),
            pl1_control=float(pl1[analyser]), training=bool(training[group]), held_out=bool(heldout[group]),
            optical_rms_principal_pixels=widths[state].tolist(), frames=frames,
            admitted=all(f['admitted'] for f in frames), elapsed_seconds=time.perf_counter()-tick))
        print(f'Fresh branch state{state}:9 image fits, admitted={rows[-1]["admitted"]}, {time.perf_counter()-tick:.1f}s', flush=True)
    arrays = {name:np.asarray(value).reshape((4, 2)+np.asarray(value).shape[1:]) for name, value in arrays.items()}
    paired_cov, joint_cov = assemble_response_covariance(arrays['response_covariance'])
    arrays.update(paired_response_covariance=paired_cov, joint_response_covariance=joint_cov,
        pl1_controls=pl1, il3_controls=il3, training_mask=training, heldout_mask=heldout,
        raw_dac_values=commands.reshape(4, 2, 7), input_steps_grouped=steps,
        input_states_grouped=input_states, detector_pixel_m=np.asarray(pixel))
    expected_response = ((spatial@shear)[:, :, GROUPED]*steps/pixel).reshape(4, 2, 2, 4)
    likelihood = {name:truth_response_likelihood(arrays['response'][mask].reshape(-1, 2, 4),
        expected_response[mask].reshape(-1, 2, 4), arrays['response_covariance'][mask].reshape(-1, 8, 8))
        for name, mask in [('training', training), ('heldout', heldout)]}
    checks = []
    for state in sorted(set((int(np.argmin(widths[:, 0])), int(np.argmax(widths[:, 1]))))):
        group, analyser = divmod(state, 2)
        frame = int(np.argmax(np.linalg.norm(means[state], axis=1)))
        check_source = MixedGaussianSource(source.position_covariance_xy_m2, source.curvature_xy_per_m,
            source.centre_covariance, canonical_means[frame])
        expected = arrays['expected_counts'][group, analyser, frame]
        sigma = np.sqrt(arrays['variance_counts'][group, analyser, frame])
        renders, refinement, previous = {}, [], None
        for order in (8, 16, 32):
            image, symplectic_error = render_pure_counts(check_source, matrices[state], detector,
                pixel_order=order, incident_electrons=dose)
            renders[order] = np.asarray(image)
            transition = None if previous is None else float(np.linalg.norm((renders[order]-previous)/sigma))
            refinement.append(dict(pixel_quadrature_order=order,
                image_whitened_l2=float(np.linalg.norm((renders[order]-expected)/sigma)),
                transition_whitened_l2=transition))
            previous = renders[order]
        path = output.with_name(output.stem+f'_gaussian_check_{state}.npz')
        np.savez_compressed(path, gaussian_pixel8_counts_yx=renders[8], gaussian_pixel16_counts_yx=renders[16],
            gaussian_pixel32_counts_yx=renders[32], covariance_counts_yx=expected, noise_sigma_yx=sigma,
            transfer_matrix=matrices[state])
        checks.append(dict(state_index=state, frame=frame, pure_gaussian_mode_count=1,
            common_axial_phase_path_m=0., beam_dependent_actions_retained=True,
            balanced_symplectic_error=symplectic_error, refinements=refinement,
            passed=all(r['image_whitened_l2'] < .2 for r in refinement)
                and all(r['transition_whitened_l2'] < .2 for r in refinement[1:]),
            arrays_path=str(path.relative_to(ROOT)), arrays_sha256=digest(path)))
    arrays_path = output.with_suffix('.npz'); np.savez_compressed(arrays_path, **arrays)
    truth_path = output.with_name(output.stem+'_truth.json')
    truth = dict(generating, schema='neoarm-branch-challenge-generating-truth-v1', source_hashes=sources,
        purpose='Post-optimization closure only; generating optics were not numerical design inputs.',
        expected_response_pixels=expected_response.tolist(), il3_controls=il3.tolist(), pl1_controls=pl1.tolist())
    truth_path.write_text(json.dumps(truth, indent=2, allow_nan=False)+'\n')
    admitted = all(row['admitted'] for row in rows) and all(row['passed'] for row in checks)
    report = dict(schema='neoarm-projector-branch-challenge-images-v1', backend=jax.default_backend(),
        x64_enabled=jax.config.x64_enabled, jax_version=jax.__version__, source_hashes=sources,
        design_report_path=str(design_path.relative_to(ROOT)), design_report_sha256=digest(design_path),
        design_frozen_before_new_image_noise=True, generating_truth_used_for_design=False,
        arrays_path=str(arrays_path.relative_to(ROOT)), arrays_sha256=digest(arrays_path),
        truth_path=str(truth_path.relative_to(ROOT)), truth_sha256=digest(truth_path),
        pl1_controls=pl1.tolist(), il3_controls=il3.tolist(), input_steps_grouped=steps.tolist(), detector_pixel_m=pixel,
        control_definitions=old['control_definitions'], training_group_indices=np.flatnonzero(training).tolist(),
        heldout_group_indices=np.flatnonzero(heldout).tolist(), generated_frame_count=72,
        training_frame_count=36, heldout_frame_count=36, incident_electrons_per_frame=dose,
        detector=dict(shape_yx=[256, 256], mtf_sigma_pixels=.65, pixel_quadrature_order=16),
        read_noise_electrons=read_noise, acquisition_groups=rows, truth_likelihood_diagnostics=likelihood,
        independent_gaussian_checks=checks, independent_control_matrix_relative_error=closure,
        all_frame_admissions_passed=all(row['admitted'] for row in rows), paired_data_admitted=admitted,
        calibration_response_noise_independent=True, shared_input_calibration_uncertainty_still_required=True,
        optical_parameters_fitted=False, hardware_commands_admitted=False, elapsed_seconds=time.perf_counter()-started,
        limitations=['This is a fresh branch challenge, not a refit or declaration of the winning instrument.',
            'Two positive IL3 offsets train; two distinct new offsets remain held out. Old heldouts are previous model-selection evidence.',
            'All72 Poisson/read-noise draws use independent seeds disjoint from old acquisitions.',
            'Six image coordinates are fit from data-only moments, with exact synthetic fixed WLS variance; variance estimation remains conditional.',
            'Independent Gaussian field propagation and two pixel-quadrature refinements certify terminal intensities at the selected narrow/wide states.',
            'The objective is fixed across settings in the same independent-half hard-edge surrogate; shared-field tails are not validated here.',
            'The pure canonical source, four input calibrations, PSF, detector geometry and stable alignment are conditional assumptions.',
            'No hardware-safe DAC range, dose feasibility, nonlinear branch power guarantee or full-column recovery is asserted.'])
    if any(digest(ROOT/name) != value for name, value in sources.items()):
        raise RuntimeError('A frozen dependency changed during new acquisition.')
    report['source_hashes_unchanged_during_execution'] = True
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    if admitted:
        inference_path = output.with_name(output.stem+'_inference.json')
        export_inference(output, inference_path, calibration_manifest=FOLDER/'last_lens_redesigned_inference.json')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--design-only', action='store_true')
    args = parser.parse_args()
    design_path = args.output if args.design_only else args.output.with_name(args.output.stem+'_design.json')
    planned = design(design_path)
    print(json.dumps({key:planned[key] for key in ('pl1_controls','il3_controls','training_mask','exact_local_design_scores','all_hypotheses_admitted')}, indent=2), flush=True)
    if not args.design_only:
        result = acquire(design_path, args.output)
        print(json.dumps({key:result[key] for key in ('paired_data_admitted','generated_frame_count','truth_likelihood_diagnostics')}, indent=2))
