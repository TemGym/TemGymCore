"""Reduce complete lens measurement sets and fit with hard bounds only.

Selection uses measured uncertainty and the previous fitted model. It never
reads generating parameters or validation images. Each retained lens setting
has one ordinary image, four tilt images and four beam-shift images (nine total).
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from finite_projector_pipeline import decode_geometry
from projector_adaptive_experiments import (
    CalibrationMeasurements, fit_displacements_then_all, load_calibration_checkpoint,
    model_to_parameters, prepare_fit, save_fit, score_parameters,
)


@dataclass
class ReductionStudy:
    baseline: object
    directory: Path
    measurements: CalibrationMeasurements
    design_parameters: np.ndarray
    starting_models: list
    controls: np.ndarray
    labels: list


def load_reduction_study(*, original_directory, adaptive_directory):
    """Read measured arrays, the previous fit and the original 20 starting guesses."""
    baseline = load_calibration_checkpoint(original_directory)
    directory = Path(adaptive_directory)
    measurements = CalibrationMeasurements.load(directory / 'wider_combinations/combined_measurements.npz')
    fitted = json.loads((directory / 'wider_combinations/best_fit.json').read_text())
    starts = json.loads((Path(original_directory) / 'beam_shift_landscape/fit_inputs.json').read_text())
    controls, labels = [], []
    for folder, label in [('projector_combinations', 'Mixed projectors'),
                          ('objective_combinations', 'Compensated objective'),
                          ('wider_combinations', 'Wider compensated')]:
        batch = CalibrationMeasurements.load(directory / folder / 'measurements.npz')
        controls.extend(batch.controls)
        labels.extend(f'{label} {i+1}' for i in range(len(batch.controls)))
    controls = np.asarray(controls)
    if len(np.unique(controls, axis=0)) != len(controls):
        raise ValueError('The lens setting bank contains duplicates.')
    for control in controls:
        one = subset_measurements(measurements=measurements, lens_controls=control[None])
        if (len(one.controls), len(one.tilts['lens_controls']), len(one.shifts['lens_controls'])) != (1, 2, 2):
            raise ValueError('Every candidate must have one ordinary image and two pairs of each kind.')
    return ReductionStudy(baseline, directory, measurements, np.asarray(fitted['parameters']),
                          starts['starting_models'], controls, labels)


def _matching_rows(recorded, selected):
    return np.any(np.all(np.isclose(np.asarray(recorded)[:, None], selected[None],
                                   rtol=0, atol=1e-12), axis=2), axis=1)


def subset_measurements(*, measurements, lens_controls):
    """Keep complete recorded observations at selected controls, without duplicating pairs."""
    controls = np.asarray(lens_controls)
    if controls.ndim != 2 or controls.shape[1] != 5 or not len(controls):
        raise ValueError('Supply a nonempty list of five-column lens settings.')
    if len(np.unique(controls, axis=0)) != len(controls):
        raise ValueError('Selected lens settings must be unique.')
    if not _matching_rows(controls, measurements.controls).all():
        raise ValueError('A selected setting has no recorded ordinary image.')
    keep = _matching_rows(measurements.controls, controls)

    def select_pairs(pairs):
        mask = _matching_rows(pairs['lens_controls'], controls)
        return {key: np.asarray(value)[mask].tolist() if key == 'pair_labels'
                else np.asarray(value)[mask] for key, value in pairs.items() if key != 'image_analysis'}

    return CalibrationMeasurements(measurements.controls[keep], measurements.values[keep],
                                   measurements.covariance[keep], select_pairs(measurements.tilts),
                                   select_pairs(measurements.shifts))


def image_count(measurements):
    return len(measurements.controls) + 2 * (len(measurements.tilts['lens_controls'])
                                          + len(measurements.shifts['lens_controls']))


def _prepare(study, measurements):
    return prepare_fit(measurements=measurements,
                       specimen_and_detector=study.baseline.source_experiment.specimen_and_detector,
                       prior_model=study.baseline.prior_model,
                       position_priors=study.baseline.position_priors, use_soft_priors=False)


def _fingerprint(*items):
    digest = hashlib.sha256()
    for item in items:
        digest.update(json.dumps(item, sort_keys=True, default=lambda a: np.asarray(a).tolist()).encode())
    return digest.hexdigest()


def _measurement_fingerprint(m):
    return _fingerprint(m.controls, m.values, m.covariance, m.tilts, m.shifts)


def design_smaller_datasets(*, study, setting_counts, output_directory, reuse_completed=True,
                           minimum_settings_per_family=None):
    """Backward elimination: remove the least informative setting at each step.

    The score is the data-Jacobian log determinant in fixed coordinates,
    whitened using all 28 bundles for numerical stability. No penalty term
    enters this calculation. This is retrospective design using the earlier
    fit; it is not a demonstration of designing a first experiment from scratch.
    """
    counts = sorted(set(int(n) for n in setting_counts))
    if not counts or counts[0] < 4 or counts[-1] > len(study.controls):
        raise ValueError('Request between four and the available number of settings.')
    families = [label.rsplit(' ', 1)[0] for label in study.labels]
    minimum = minimum_settings_per_family or {}
    if (any(name not in families or value < 0 or int(value) != value
            or value > families.count(name) for name, value in minimum.items())
            or sum(minimum.values()) > counts[0]):
        raise ValueError('Family minimum counts must fit inside every requested subset.')
    folder = Path(output_directory)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'selection.json'
    fingerprint = _fingerprint(_measurement_fingerprint(study.measurements), study.design_parameters, counts, minimum)
    if reuse_completed and path.exists():
        previous = json.loads(path.read_text())
        if previous['fingerprint'] == fingerprint:
            return previous
    m = subset_measurements(measurements=study.measurements, lens_controls=study.controls)
    problem = _prepare(study, m)
    u = problem.encode(study.design_parameters)
    jacobian = np.asarray(problem.jacobian(u))
    _, singular, vt = np.linalg.svd(jacobian, full_matrices=False)
    transform = vt.T / singular
    row_controls = np.concatenate([np.repeat(m.controls, 3, axis=0),
        np.repeat(m.tilts['lens_controls'], 2, axis=0),
        np.repeat(m.shifts['lens_controls'], 2, axis=0)])
    blocks = [jacobian[_matching_rows(row_controls, c[None])] @ transform for c in study.controls]

    def log_information(indices):
        s = np.linalg.svd(np.concatenate([blocks[i] for i in indices]), compute_uv=False)
        return float(2 * np.log(np.maximum(s, 1e-30)).sum())

    remaining = list(range(len(blocks)))
    subsets, removed = {}, []
    while len(remaining) >= min(counts):
        if len(remaining) in counts:
            subsets[str(len(remaining))] = remaining.copy()
        if len(remaining) == min(counts):
            break
        scores = [log_information([j for j in remaining if j != i])
                  if sum(families[j] == families[i] for j in remaining) > minimum.get(families[i], 0)
                  else -np.inf for i in remaining]
        index = int(np.argmax(scores))
        removed.append(dict(index=remaining[index], label=study.labels[remaining[index]],
                            remaining_log_information=scores[index]))
        remaining.pop(index)
    selection = dict(fingerprint=fingerprint, method='Backward elimination of data-only information',
                     design_uses='Previous fitted model and measured covariances; no truth or validation data',
                     setting_counts=counts, subsets=subsets, removal_order=removed,
                     lens_controls=study.controls.tolist(), labels=study.labels,
                     minimum_settings_per_family=minimum,
                     acceptance=dict(all_starts_same_solution=True, maximum_training_rms=1.5,
                                     maximum_local_relative_standard_error=0.01))
    path.write_text(json.dumps(selection, indent=2))
    return selection


def _physical_values(p):
    _, lengths, centres, detector = decode_geometry(p)
    return jnp.r_[centres * 1000, lengths * 1000, detector * 1000, p[11],
                  jnp.exp(p[12:16]), p[16] * 1e4, jnp.exp(p[17]) * 1000, jnp.degrees(p[18])]


def _local_uncertainty(problem, parameters):
    u = problem.encode(parameters)
    jacobian = np.asarray(problem.jacobian(u))
    _, singular, vt = np.linalg.svd(jacobian, full_matrices=False)
    tolerance = singular[0] * max(jacobian.shape) * np.finfo(float).eps
    rank = int(np.sum(singular > tolerance))
    derivative = np.asarray(jax.jacfwd(lambda v: _physical_values(problem.decode(v)))(u))
    covariance_factor = derivative @ (vt.T / np.maximum(singular, tolerance))
    standard_error = np.linalg.norm(covariance_factor, axis=1)
    relative = standard_error / np.maximum(np.abs(np.asarray(_physical_values(parameters))), 1e-20)
    return dict(rank=rank, smallest_singular_value=float(singular[-1]),
                covariance_factor_solver_coordinates=(vt.T / np.maximum(singular, tolerance)).tolist(),
                standard_errors=standard_error.tolist(), relative_standard_errors=relative.tolist(),
                maximum_relative_standard_error=float(relative.max()) if rank == 19 else None)


def fit_measurement_dataset(*, measurements, specimen_and_detector, starting_guess,
                            position_bounds, starting_models, output_directory,
                            reuse_completed=True, max_evaluations=5000):
    """Fit measured arrays from independent guesses; return recovery and precision checks.

    All soft penalties are disabled. position_bounds supplies hard bounds and
    numerical scaling only. No truth, previous fitted model or image generator
    is passed to this function.
    """
    folder = Path(output_directory)
    folder.mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint('hard-bounds-only-v1', _measurement_fingerprint(measurements),
                               starting_models, position_bounds,
                               specimen_and_detector, max_evaluations)
    path = folder / 'summary.json'
    if reuse_completed and path.exists():
        record = json.loads(path.read_text())
        if record['fingerprint'] == fingerprint:
            if ('parameters' not in record or
                    'covariance_factor_solver_coordinates' in record['local_uncertainty']):
                return record
            # Older result files contain the same fit but only marginal errors.
            problem = prepare_fit(measurements=measurements, specimen_and_detector=specimen_and_detector,
                                  prior_model=starting_guess, position_priors=position_bounds,
                                  use_soft_priors=False)
            record['local_uncertainty'] = _local_uncertainty(problem, np.asarray(record['parameters']))
            path.write_text(json.dumps(record, indent=2))
            return record
    path.unlink(missing_ok=True)
    problem = prepare_fit(measurements=measurements, specimen_and_detector=specimen_and_detector,
                          prior_model=starting_guess, position_priors=position_bounds,
                          use_soft_priors=False)
    starts = []
    for index, model in enumerate(starting_models):
        fit_path = folder / f'start_{index:02d}.json'
        start_key = _fingerprint(fingerprint, index)
        cached = json.loads(fit_path.read_text()) if reuse_completed and fit_path.exists() else {}
        if cached.get('fingerprint') == start_key:
            record = cached
        else:
            try:
                result = fit_displacements_then_all(problem=problem, starting_parameters=model_to_parameters(model),
                    ordinary_image_count=len(measurements.controls), max_evaluations=max_evaluations)
            except RuntimeError as error:
                record = dict(success=False, message=str(error))
            else:
                save_fit(fit_path, result)
                record = json.loads(fit_path.read_text())
                record['data_rms'] = float(np.sqrt(np.mean(result.data_residual**2)))
                record['active_bounds'] = np.flatnonzero(result.solver_active_mask).tolist()
                # This catches accidental retention of any position or other penalty.
                if len(result.fun) != len(result.data_residual):
                    raise AssertionError('A soft penalty survived in a supposedly data-only fit.')
            record.update(index=index, fingerprint=start_key)
            fit_path.write_text(json.dumps(record, indent=2))
        starts.append(record)
        print(f'{folder.name}: start {index+1}/{len(starting_models)}, '
              f'RMS {record.get("data_rms", float("inf")):.4g}', flush=True)
    valid = [r for r in starts if r['success'] and r.get('data_rms', np.inf) < 1.5]
    summary = dict(fingerprint=fingerprint, image_count=image_count(measurements),
                   ordinary_images=len(measurements.controls), starts=len(starts),
                   acceptable_starts=len(valid), use_soft_priors=False, hard_bounds_retained=True)
    if valid:
        best = min(valid, key=lambda r: r['data_squared_residual'])
        p = np.asarray(best['parameters'])
        values = np.asarray(_physical_values(p))
        agreeing = [r for r in valid if r['data_squared_residual'] <= best['data_squared_residual'] + .01
                    and np.max(np.abs(np.asarray(_physical_values(np.asarray(r['parameters']))) - values)
                               / np.maximum(abs(values), 1e-20)) < .001]
        summary.update(parameters=best['parameters'], model=best['model'],
                       data_rms=best['data_rms'], data_squared_residual=best['data_squared_residual'],
                       agreeing_starts=len(agreeing), local_uncertainty=_local_uncertainty(problem, p),
                       active_bounds=best['active_bounds'])
    else:
        summary.update(agreeing_starts=0, data_rms=None)
    path.write_text(json.dumps(summary, indent=2))
    return summary


def _fit_one_dataset(*, study, **kwargs):
    return fit_measurement_dataset(
        specimen_and_detector=study.baseline.source_experiment.specimen_and_detector,
        starting_guess=study.baseline.prior_model, position_bounds=study.baseline.position_priors,
        **kwargs)


def fit_smaller_datasets(*, study, selection, output_directory, number_of_starts=20,
                        reuse_completed=True):
    """Refit each size from the same original guesses, with no soft priors in either stage.

    Returns convergence and local precision. Generating truth and the 72
    prediction images are deliberately not used to choose the recommended size.
    """
    if number_of_starts < 1 or number_of_starts > len(study.starting_models):
        raise ValueError('Request between one and the saved number of starting guesses.')
    folder = Path(output_directory)
    starts = study.starting_models[:number_of_starts]
    full = _fit_one_dataset(study=study, measurements=study.measurements, starting_models=starts,
                           output_directory=folder / 'full_334', reuse_completed=reuse_completed)
    results = {'full': full}
    for count in sorted(selection['setting_counts'], reverse=True):
        indices = selection['subsets'][str(count)]
        measured = subset_measurements(measurements=study.measurements, lens_controls=study.controls[indices])
        results[str(count)] = _fit_one_dataset(study=study, measurements=measured, starting_models=starts,
                            output_directory=folder / f'settings_{count:02d}', reuse_completed=reuse_completed)
    acceptable = [int(key) for key, r in results.items() if key != 'full'
                  and r['agreeing_starts'] == number_of_starts and not r.get('active_bounds', [1])
                  and r.get('local_uncertainty', {}).get('maximum_relative_standard_error') is not None
                  and r['local_uncertainty']['maximum_relative_standard_error'] < .01]
    report = dict(results=results, recommended_setting_count=min(acceptable) if acceptable else None,
                  acceptance=selection['acceptance'], selection_fixed_before_validation=True)
    (folder / 'fit_comparison.json').write_text(json.dumps(report, indent=2))
    return report


def validate_smaller_dataset(*, study, selection, comparison, output_directory,
                            number_of_starts=20, reuse_completed=True):
    """Score frozen fits, then check the chosen settings on two independent noise repeats.

    These repeat images were independently generated and analysed in the prior
    experiment. They are subset with the already frozen controls. No new image
    synthesis is needed; no discarded frame enters a reduced microscope fit.
    """
    folder = Path(output_directory)
    chosen = comparison['recommended_setting_count']
    if chosen is None:
        raise RuntimeError('No tested size passed the declared precision and recovery checks.')
    fresh = CalibrationMeasurements.load(study.directory / 'fresh_predictions/measurements.npz')
    fresh_problem = _prepare(study, fresh)
    truth_directory = study.baseline.source_experiment.output_directory

    def score(record):
        if 'parameters' not in record:
            return dict(maximum_relative_parameter_error=None, fresh_prediction_rms=None)
        p = np.asarray(record['parameters'])
        errors = score_parameters(parameters=p, truth_directory=truth_directory)
        residual = np.asarray(fresh_problem.data_residual(p))
        return dict(parameter_errors=errors,
                    maximum_relative_parameter_error=max(abs(v['relative_error']) for v in errors.values()),
                    fresh_prediction_rms=float(np.sqrt(np.mean(residual**2))))

    original = {key: score(r) for key, r in comparison['results'].items()}
    controls = study.controls[selection['subsets'][str(chosen)]]
    repeats = []
    for repeat in (1, 2):
        measured = CalibrationMeasurements.load(study.directory / f'noise_repeat_{repeat}/measurements.npz')
        subset = subset_measurements(measurements=measured, lens_controls=controls)
        result = _fit_one_dataset(study=study, measurements=subset,
            starting_models=study.starting_models[:number_of_starts],
            output_directory=folder / f'noise_repeat_{repeat}', reuse_completed=reuse_completed)
        repeats.append(dict(fit=result, **score(result)))
    report = dict(recommended_setting_count=chosen, calibration_images=9*chosen,
                  validation_images=image_count(fresh), validation_refitted=False,
                  selection_uses_validation=False, original=original, independent_noise_repeats=repeats)
    (folder / 'validation.json').write_text(json.dumps(report, indent=2))
    return report
