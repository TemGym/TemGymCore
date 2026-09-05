"""GPU numerical admission of the frozen internal-Q beam acquisition.

This new report preserves the original guard-failure artifact. It tests the
same piecewise canonical paraxial model via short validated factors, not the
physical validity of that model at its large internal angles.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from neoarm_condenser_internal_beam_images import ROOT, GROUPED, model_context, verify, digest
from neoarm_condenser_factorized_transport import internal_q_factors, propagate_factors, render_factorized_modes
from neoarm_condenser_gaussian_validation import covariance_to_mixed_source, rank_adapted_source_modes
from temgym_core.evaluate import _beam_field, _prepare_gaussian_params
from temgym_core.gaussian_image_likelihood import make_gaussian_detector_operator, fit_gaussian_image, encode_gaussian_image_parameters
from temgym_core.neoarm_calibration import DetectorGeometry


def noise_error(actual, reference, variance):
    value = (np.asarray(actual)-np.asarray(reference))/np.sqrt(variance)
    return dict(whitened_L2=float(np.linalg.norm(value)), RMS_sigma=float(np.sqrt(np.mean(value**2))))


def phase_comparison(original, refined, covariance_xy):
    """One common phase alignment across all modes, never one phase per mode."""
    angle = np.linspace(0., 2*np.pi, 41, endpoint=False)
    root = np.linalg.cholesky(covariance_xy)
    points = np.concatenate([np.zeros((1, 2)),
        1.5*np.stack([np.cos(angle), np.sin(angle)], axis=1)@root.T])
    evaluate = jax.jit(lambda beam: jax.vmap(_beam_field,
        in_axes=(0, 0, 0, 0, 0, 0, None))(*_prepare_gaussian_params(beam), jnp.asarray(points)))
    a, b = np.asarray(evaluate(original)), np.asarray(evaluate(refined))
    alignment = np.vdot(a, b); alignment /= abs(alignment)
    return dict(complex_field_relative_L2_after_one_global_phase=float(np.linalg.norm(b-alignment*a)/np.linalg.norm(b)),
        common_phase_alignment_rad=float(np.angle(alignment)), mode_dependent_phases_aligned=False,
        probe_point_count=len(points), nonzero_amplitude_norm=float(np.linalg.norm(b)))


def validate(design_path, output, frame_indices=(0, 3, 30)):
    if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
        raise RuntimeError('GPU float64 is required for actual Gaussian forward images.')
    started = perf_counter(); design_path = Path(design_path).resolve()
    design = json.loads(design_path.read_text()); verify(design)
    if not design['design_gates_accepted']:
        raise ValueError('The frozen acquisition must first pass its design gates.')
    model, conditional = model_context(); p = jnp.asarray(design['generating_hypothesis_parameters'])
    source_covariance = np.asarray(model.decode(p)[1])[np.ix_(GROUPED, GROUPED)]
    detector = DetectorGeometry(mtf_sigma_px=.65); dose = design['incident_electrons_per_frame']
    operator = jax.jit(make_gaussian_detector_operator(detector, 32))
    source, _ = covariance_to_mixed_source(source_covariance)
    records, arrays = [], {}
    for frame in frame_indices:
        tick = perf_counter(); control = jnp.asarray(design['controls'][frame//2]); analyser = frame % 2
        grouped = model.analyser_maps[analyser]@model.maps(p, control)[0]
        original_matrix = np.asarray(grouped)[np.ix_(GROUPED, GROUPED)]
        covariance = original_matrix@source_covariance@original_matrix.T
        position = covariance[np.ix_([0, 2], [0, 2])]
        reference = np.asarray(operator(np.zeros(2), position, dose)); variance = reference+.6**2
        factors = internal_q_factors(model, conditional, p, control, analyser)
        product = np.asarray(factors.product())
        mode_rows, actual, previous, accepted_order = [], None, None, None
        for order in (5, 9, 13, 21, 31, 41, 51):
            begin = perf_counter()
            actual, metadata = render_factorized_modes(source_covariance, factors, detector, dose,
                hermite_order=order, pixel_order=8)
            actual = np.asarray(actual)
            row = dict(order=order, mode_count=metadata['coherent_mode_count'],
                reference_error=noise_error(actual, reference, variance),
                transition=None if previous is None else noise_error(actual, previous, variance),
                elapsed_seconds=perf_counter()-begin)
            mode_rows.append(row); previous = actual
            print(json.dumps(dict(frame=frame, mode_refinement=row)), flush=True)
            if len(mode_rows) >= 3 and row['reference_error']['whitened_L2'] < .2 and all(
                    item['transition']['whitened_L2'] < .1 for item in mode_rows[-2:]):
                accepted_order = order; break
        if accepted_order is None:
            records.append(dict(frame=frame, accepted=False, failure='modal_refinement', mode_refinements=mode_rows))
            continue
        # Two successive regroupings of the SAME objective midpoint profile,
        # coupled with smaller physical hard-edge slices. No profile redraw.
        modes, _, _ = rank_adapted_source_modes(source, accepted_order)
        old_beam = propagate_factors(modes, factors); transport_rows = []
        for slices, block in ((16, 32), (32, 16)):
            refined = internal_q_factors(model, conditional, p, control, analyser,
                hardedge_slices=slices, objective_block_size=block)
            new_beam = propagate_factors(modes, refined)
            phase = phase_comparison(old_beam, new_beam, position)
            counts, _ = render_factorized_modes(source_covariance, refined, detector, dose,
                hermite_order=accepted_order, pixel_order=8)
            counts = np.asarray(counts)
            transport_rows.append(dict(hardedge_slices=slices, objective_block_size=block,
                primitive_or_guarded_block_count=len(refined.matrices),
                reference_error=noise_error(counts, reference, variance),
                transition=noise_error(counts, actual, variance), phase=phase))
            old_beam, factors, actual = new_beam, refined, counts
        pixel_rows = []
        for order in (16, 32):
            begin = perf_counter()
            counts, metadata = render_factorized_modes(source_covariance, factors, detector, dose,
                hermite_order=accepted_order, pixel_order=order)
            counts = np.asarray(counts)
            row = dict(order=order, reference_error=noise_error(counts, reference, variance),
                transition=noise_error(counts, actual, variance), elapsed_seconds=perf_counter()-begin)
            pixel_rows.append(row); actual = counts
            print(json.dumps(dict(frame=frame, pixel_refinement=row)), flush=True)
        encoded = np.asarray(encode_gaussian_image_parameters(np.zeros(2), position, dose, detector))
        # This truth-seeded fit is ONLY a noiseless numerical-bias diagnostic,
        # not an initializer or result of the future data-only optical fit.
        fit = fit_gaussian_image(actual, variance, detector, initial_parameters=encoded,
                                 quadrature_order=32, max_nfev=80)
        bias = (np.asarray(fit.parameters)-encoded)/np.sqrt(np.diag(fit.parameter_covariance))
        accepted = bool(fit.optimizer_result.success and max(abs(bias)) < .2
            and noise_error(actual, reference, variance)['whitened_L2'] < .2
            and all(row['transition']['whitened_L2'] < .1 for row in pixel_rows+transport_rows)
            and all(row['phase']['complex_field_relative_L2_after_one_global_phase'] < 1e-5 for row in transport_rows))
        record = dict(frame=frame, accepted=accepted, accepted_hermite_order=accepted_order,
            final_mode_count=metadata['coherent_mode_count'], mode_refinements=mode_rows,
            transport_refinements=transport_rows, pixel_refinements=pixel_rows,
            original_factor_product_relative_closure=float(np.linalg.norm(product-original_matrix)/np.linalg.norm(original_matrix)),
            final_image_error=noise_error(actual, reference, variance),
            noiseless_image_fit_success=bool(fit.optimizer_result.success),
            noiseless_six_image_parameter_bias_sigma=bias.tolist(),
            captured_fraction=float(actual.sum()/dose), elapsed_seconds=perf_counter()-tick)
        records.append(record)
        arrays[f'frame{frame}_gaussian_counts'] = actual
        arrays[f'frame{frame}_reference_counts'] = reference
        print(json.dumps(dict(frame=frame, accepted=accepted, final_error=record['final_image_error'],
                             elapsed_seconds=record['elapsed_seconds'])), flush=True)
    sources = dict(design['source_hashes'])
    paths = (design_path, Path(__file__).resolve(),
        ROOT/'examples/lens_inversion/neoarm_condenser_factorized_transport.py',
        ROOT/'tests/test_neoarm_condenser_factorized_transport.py')
    for path in paths:
        sources[str(path.relative_to(ROOT))] = digest(path)
    verify(dict(source_hashes=sources))
    output = Path(output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    array_path = output.with_suffix('.npz'); np.savez_compressed(array_path, **arrays)
    report = dict(schema='neoarm-factorized-internal-Q-Gaussian-validation-v1', source_hashes=sources,
        design_path=str(design_path.relative_to(ROOT)), design_sha256=digest(design_path),
        arrays_path=str(array_path.relative_to(ROOT)), arrays_sha256=digest(array_path),
        backend=jax.default_backend(), x64_enabled=True, dose_electrons_per_frame=dose,
        records=records, all_selected_frames_accepted=all(row['accepted'] for row in records),
        selected_frame_indices=list(frame_indices), full32_frame_acquisition_performed=False,
        planned_hermite_orders=[5, 9, 13, 21, 31, 41, 51],
        maximum_accepted_hermite_order=max((row.get('accepted_hermite_order', 0) for row in records), default=0),
        objective_midpoint_profile_unchanged=True, original_canonical_guard_unchanged=True,
        assembled_matrix_projected_or_used_for_Gaussian_transport=False,
        beam_dependent_phase_preserved=True, common_axial_phase_omitted=True,
        acquisition_noise_drawn=False, optical_recovery_performed=False,
        elapsed_seconds=perf_counter()-started,
        limitations=['Numerical closure only for the frozen paraxial quadratic model and selected frames.',
            'The canonical covariance reference uses the original composed matrices; it is independent of Gaussian propagation, not an independent physical field model.',
            'C-mini mechanical RMS reaches36mrad; nonparaxial and bore/aperture clearance checks remain unresolved.',
            'Finite objective support and midpoint field discretization are frozen, not revalidated by subdivision of the same primitive factors.',
            'No current readback, upstream kicks, known source/Q inference prior, hardware dose/control admission or optical recovery is implied.'])
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('Saved', output, flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, nargs='+', default=[0, 3, 30])
    args = parser.parse_args(); jax.config.update('jax_enable_x64', True)
    validate(args.design, args.output, args.frames)
