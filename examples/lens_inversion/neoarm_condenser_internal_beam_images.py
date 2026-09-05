"""Frozen4me internal-Q beam-image experiment; no optical recovery.

The generating preparation is afterCL1. Neither its source covariance nor Q
location/strength is an inference prior; an alternative component order must
still be tested by a later optical fitter. Hardware dose/rates are unconfirmed.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from neoarm_condenser_autofocus_routes import FOLDER, ROOT
from neoarm_condenser_internal_beam_design import covariance_at, observables_at, information
from neoarm_condenser_actual_analyser_contract import ActualAnalyserStigmator, PIXEL_M
from neoarm_condenser_extension_design import load_context
from neoarm_condenser_extension_images import fit_data_only_image
from neoarm_condenser_gaussian_validation import (
    render_covariance_gaussian_modes, validate_gaussian_modes, covariance_to_mixed_source,
)
from neoarm_shared_objective_audit import provenance_snapshot
from temgym_core.gaussian_image_likelihood import make_gaussian_detector_operator, encode_gaussian_image_parameters, gaussian_image_information_factor
from temgym_core.neoarm_calibration import DetectorGeometry

GROUPED = np.array([0, 2, 1, 3])
DOSE = 4e6
SEED = 1374109


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(report):
    for path, checksum in report['source_hashes'].items():
        if digest(ROOT/path) != checksum:
            raise ValueError(f'Frozen experiment provenance mismatch: {path}')


def roles(training, heldout, count):
    a, b = np.asarray(training), np.asarray(heldout)
    if a.dtype != bool or b.dtype != bool or a.shape != (count,) or b.shape != a.shape or np.any(a&b) or not np.all(a|b):
        raise ValueError('Disjoint exhaustive boolean optical-state roles are required.')
    return a, b


def model_context():
    _, _, _, _, conditional = load_context()
    return ActualAnalyserStigmator('after_cl1', conditional), conditional


def preflight(output):
    if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
        raise RuntimeError('GPU float64 is required.')
    source_path = FOLDER/'condenser_autofocus_routes.json'
    previous = json.loads(source_path.read_text()); verify(previous)
    route = next(r for r in previous['routes'] if r['location'] == 'after_cl1')
    selected = route['selected']; controls = np.asarray(selected['controls']); p = jnp.asarray(route['nominal_parameters'])
    training, heldout = roles(selected['training_mask'], selected['heldout_mask'], len(controls))
    model, conditional = model_context(); detector = DetectorGeometry(mtf_sigma_px=.65)
    c = np.asarray(jax.jit(lambda u: covariance_at(model, p, u))(jnp.asarray(controls)))
    operator = jax.jit(make_gaussian_detector_operator(detector, 32))
    information_fun = jax.jit(lambda x: gaussian_image_information_factor(x, detector, quadrature_order=32))
    noise, capture, ranks = [], [], []
    for covariance in c.reshape(-1, 2, 2):
        encoded = encode_gaussian_image_parameters(np.zeros(2), covariance*PIXEL_M**2, DOSE, detector)
        j = np.asarray(information_fun(encoded)); _, s, v = np.linalg.svd(j, full_matrices=False)
        factor = v.T/s; noise.append((factor@factor.T)[2:5, 2:5]); ranks.append(int(np.sum(s > s[0]*1e-10)))
        capture.append(float(np.asarray(operator(np.zeros(2), covariance*PIXEL_M**2, DOSE)).sum()/DOSE))
    noise = np.asarray(noise).reshape(-1, 2, 3, 3)
    jacobian = np.asarray(jax.jit(jax.jacfwd(lambda pp: observables_at(model, pp, jnp.asarray(controls))))(p))
    info, _ = information(np.linalg.solve(np.linalg.cholesky(noise[training]), jacobian[training]))
    widths = np.sqrt(np.linalg.eigvalsh(c))
    accepted = bool(info['rank'] == 25 and max(info['length_standard_errors_percent']) <= 1.
        and min(ranks) == 6 and widths[:, :, 0].min() >= .4 and min(capture) >= .999)
    sources = dict(previous['source_hashes']); loaded, _ = provenance_snapshot(); sources.update(loaded)
    for path in (Path(__file__), ROOT/'tests/test_neoarm_condenser_internal_beam_images.py', source_path):
        sources[str(path.relative_to(ROOT))] = digest(path)
    report = dict(schema='neoarm-internal-Q-beam-acquisition-design-v1', source_hashes=sources,
        backend=jax.default_backend(), x64_enabled=True, controls=controls.tolist(),
        training_mask=training.tolist(), heldout_mask=heldout.tolist(),
        conditional_analyser_controls=np.asarray(conditional.analyser_controls).tolist(),
        generating_hypothesis_parameters=np.asarray(p).tolist(), generating_component_order='after_cl1',
        generating_parameters_are_not_inference_priors=True, incident_electrons_per_frame=DOSE,
        noise_seed=SEED, frame_count=2*len(controls), training_frame_count=int(2*training.sum()),
        heldout_frame_count=int(2*heldout.sum()), training_information=info,
        minimum_capture=min(capture), minimum_optical_rms_pixels=float(widths.min()),
        maximum_optical_rms_pixels=float(widths.max()), expected_covariance_observable_noise=noise.tolist(),
        design_gates_accepted=accepted, frozen_before_any_new_noise=True,
        mean_field_capture_and_information_only=True, mode_or_noisy_image_acceptance_not_yet_demonstrated=True,
        competing_Q_order_ruled_out=False, source_Q_or_optical_parameters_fixed_in_information=False,
        hardware_dose_rate_or_command_granularity_admitted=False)
    verify(report); output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def acquire(design_path, output):
    if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
        raise RuntimeError('GPU float64 is required.')
    start = perf_counter(); design_path = Path(design_path); design = json.loads(design_path.read_text()); verify(design)
    if not design['design_gates_accepted'] or not design['frozen_before_any_new_noise']:
        raise ValueError('The independently frozen acquisition must pass preflight.')
    model, conditional = model_context(); controls = np.asarray(design['controls']); p = jnp.asarray(design['generating_hypothesis_parameters'])
    training, heldout = roles(design['training_mask'], design['heldout_mask'], len(controls))
    grouped = np.asarray(jax.jit(jax.vmap(lambda u: model.analyser_maps@model.maps(p, u)[0]))(jnp.asarray(controls)))
    matrices = grouped[:, :, GROUPED[:, None], GROUPED[None, :]].reshape(-1, 4, 4)
    source = np.asarray(model.decode(p)[1])[np.ix_(GROUPED, GROUPED)]
    detector = DetectorGeometry(mtf_sigma_px=.65)
    total_covariance = matrices@source@matrices.swapaxes(-1, -2)
    spatial = total_covariance[:, [0, 2]][:, :, [0, 2]]
    width = np.sqrt(np.linalg.eigvalsh(spatial))/PIXEL_M
    pure, _ = covariance_to_mixed_source(source)
    propagated_pure = matrices@pure.pure_covariance@matrices.swapaxes(-1, -2)
    pure_spatial = propagated_pure[:, [0, 2]][:, :, [0, 2]]
    mixture_ratio = np.array([np.linalg.eigvals(np.linalg.solve(a, b)).real.max() for a, b in zip(pure_spatial, spatial)])
    validate_indices = sorted(set([int(np.argmin(width.min(axis=1))), int(np.argmax(width.max(axis=1))), int(np.argmax(mixture_ratio))]))
    sources = dict(design['source_hashes']); sources[str(design_path.relative_to(ROOT))] = digest(design_path)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    # Freeze one accepted modal topology before drawing any acquisition noise.
    validation_attempts = []; accepted_order = None; validation_arrays = {}
    for orders in ((5, 9, 13), (13, 21, 31)):
        records, arrays = validate_gaussian_modes(source, matrices[validate_indices], detector, DOSE,
            names=[f'frame_{i}' for i in validate_indices], hermite_orders=orders, pixel_orders=(8, 16, 32))
        validation_attempts.append(dict(hermite_orders=list(orders), records=records))
        validation_arrays = arrays
        # RMS alone can conceal an identifiable image-parameter bias across
        # many pixels. Impose the stronger whitenedL2 bound before noise.
        if all(row['accepted'] and row['image_rms_sigma']*256 < .2 for row in records):
            accepted_order = orders[-1]; break
    validation_path = output.with_name(output.stem+'_validation.json')
    validation_report = dict(source_hashes=sources, selected_frame_indices=validate_indices,
        attempts=validation_attempts, accepted_hermite_order=accepted_order,
        no_acquisition_noise_drawn=True, generating_values_are_truth_only=True)
    validation_path.write_text(json.dumps(validation_report, indent=2, allow_nan=False)+'\n')
    if accepted_order is None:
        print('Modal/pixel validation failed; no acquisition noise generated.', flush=True)
        return validation_report
    operator = jax.jit(make_gaussian_detector_operator(detector, 32))
    rng = np.random.default_rng(SEED); noisy, fitted, variance_images, image_parameters, image_covariance = [], [], [], [], []
    expected_images = []; reference_parameters = []; rows = []
    for index, (matrix, covariance) in enumerate(zip(matrices, spatial)):
        tick = perf_counter()
        expected, metadata = render_covariance_gaussian_modes(source, matrix, detector, DOSE,
            hermite_order=accepted_order, pixel_order=16)
        expected = np.asarray(expected); reference = np.asarray(operator(np.zeros(2), covariance, DOSE))
        difference = (expected-reference)/np.sqrt(reference+.6**2)
        joint_error = float(np.linalg.norm(difference))
        if joint_error >= .2:
            raise RuntimeError(f'Frame{index} Gaussian-mode/reference joint noise gate failed: {joint_error}')
        observed = rng.poisson(np.maximum(expected, 0.))+rng.normal(0., .6, expected.shape)
        fit, variance, fit_meta = fit_data_only_image(observed, detector)
        parameters = np.asarray(fit.parameters)
        noisy.append(observed); fitted.append(np.asarray(fit.fitted_counts_yx)); variance_images.append(variance)
        image_parameters.append(parameters); image_covariance.append(np.asarray(fit.parameter_covariance)); expected_images.append(expected)
        reference_parameters.append(np.asarray(encode_gaussian_image_parameters(np.zeros(2), covariance, DOSE, detector)))
        row = dict(frame=index, state=index//2, analyser=index%2, training=bool(training[index//2]),
            accepted=fit_meta['accepted'], fit_metadata=fit_meta, numerical_rank=int(fit.numerical_rank),
            chi_square_z=float(fit.chi_square_z), Gaussian_reference_joint_noise_sigma=joint_error,
            Gaussian_reference_rms_noise_sigma=float(np.sqrt(np.mean(difference**2))),
            captured_fraction=float(expected.sum()/DOSE), coherent_mode_count=metadata['coherent_mode_count'],
            elapsed_seconds=perf_counter()-tick)
        rows.append(row); print(json.dumps(row), flush=True)
    shape = (len(controls), 2)
    public_path = output.with_suffix('.npz')
    ip = np.asarray(image_parameters).reshape(*shape, 6); ic = np.asarray(image_covariance).reshape(*shape, 6, 6)
    np.savez_compressed(public_path, controls=controls, training_mask=training, heldout_mask=heldout,
        analyser_controls=np.asarray(conditional.analyser_controls), beam_observables=ip[:, :, 2:5],
        beam_observable_covariances=ic[:, :, 2:5, 2:5], image_parameters=ip, image_parameter_covariances=ic,
        noisy_images=np.asarray(noisy).reshape(*shape, 256, 256), fitted_images=np.asarray(fitted).reshape(*shape, 256, 256),
        fitted_variance_images=np.asarray(variance_images).reshape(*shape, 256, 256))
    truth_path = output.with_name(output.stem+'_truth.npz')
    np.savez_compressed(truth_path, generating_parameters=np.asarray(p), source_covariance=source,
        full_canonical_matrices=matrices.reshape(*shape, 4, 4), expected_images=np.asarray(expected_images).reshape(*shape, 256, 256),
        image_parameters=np.asarray(reference_parameters).reshape(*shape, 6), **validation_arrays)
    delta = ip[:, :, 2:5]-np.asarray(reference_parameters).reshape(*shape, 6)[:, :, 2:5]
    whitened = np.linalg.solve(np.linalg.cholesky(ic[:, :, 2:5, 2:5]), delta[..., None])
    truth_report = dict(truth_path=str(truth_path.relative_to(ROOT)), truth_sha256=digest(truth_path),
        source_hashes=sources, training_observable_truth_chi_square=float(np.sum(whitened[training]**2)),
        heldout_observable_truth_chi_square=float(np.sum(whitened[heldout]**2)),
        truth_values_are_postfit_diagnostics_only=True, generating_Q_order='after_cl1',
        source_coordinate_order=['x', 'pi_x', 'y', 'pi_y'])
    truth_json = output.with_name(output.stem+'_truth.json'); truth_json.write_text(json.dumps(truth_report, indent=2, allow_nan=False)+'\n')
    verify(dict(source_hashes=sources))
    report = dict(schema='neoarm-internal-Q-general-source-beam-images-v1', source_hashes=sources,
        design_path=str(design_path.relative_to(ROOT)), design_sha256=digest(design_path), backend=jax.default_backend(),
        elapsed_seconds=perf_counter()-start, arrays_path=str(public_path.relative_to(ROOT)), arrays_sha256=digest(public_path),
        truth_path=str(truth_path.relative_to(ROOT)), truth_sha256=digest(truth_path),
        truth_report_path=str(truth_json.relative_to(ROOT)), truth_report_sha256=digest(truth_json),
        validation_path=str(validation_path.relative_to(ROOT)), validation_sha256=digest(validation_path),
        controls=controls.tolist(), training_mask=training.tolist(), heldout_mask=heldout.tolist(),
        training_image_coordinate_count=int(6*training.sum()), heldout_image_coordinate_count=int(6*heldout.sum()),
        dose_electrons_per_frame=DOSE, seed=SEED, accepted_hermite_order=accepted_order, frames=rows,
        all_images_accepted=all(row['accepted'] for row in rows),
        public_observables='log RMS_x_pixels, log RMS_y_pixels, atanh correlation; full marginalized3x3 covariance from data-only6-coordinate feasible-GLS image fits.',
        generating_source_Q_or_optics_supplied_as_inference_prior=False, competing_Q_order_ruled_out=False,
        optical_recovery_performed=False, hardware_dose_rate_or_control_safety_admitted=False,
        limitations=['One unchanged physical general Gaussian mixed source and internalQ define the generating case, not known fitting inputs.',
            'The model-autofocus controls were frozen before noise; an actual image-driven autofocus procedure remains unvalidated.',
            'No aperture masks, intermediate bores, source changes, hysteresis, calibration uncertainty or detector dead-time effects are simulated.',
            'Objective/downstream maps and detector/energy remain conditional; all25 source/Q/optical parameters must be free in future recovery.',
            'Heldout image summaries are stored separately by fixed masks and have not entered an optical fit.'])
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n'); return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('preflight', 'acquire'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--design', type=Path)
    args = parser.parse_args(); jax.config.update('jax_enable_x64', True)
    result = preflight(args.output) if args.phase == 'preflight' else acquire(args.design.resolve(), args.output.resolve())
    print('Saved', args.output, flush=True)
