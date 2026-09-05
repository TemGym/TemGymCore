"""One readable image -> measurement -> microscope-fit procedure.

The simulation function owns generating truth. Image analysis receives images
and public specimen/detector settings. Fitting receives measured responses,
independent starting guesses and hard bounds. Truth is used again only to score.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from finite_projector_pipeline import _generator_parameters, public_specification
from magical_projector_pipeline import make_magical_template
from projector_adaptive_experiments import (
    CalibrationMeasurements, simulate_setting_images, prepare_fit, score_parameters,
)
from projector_beam_shift_pipeline import (
    BeamShiftExperiment, simulate_beam_shift_images, measure_beam_shift_images,
)
from projector_dataset_reduction import fit_measurement_dataset, image_count
from projector_landscape import make_starting_models
from projector_notebook_helpers import Experiment, measure_images
from projector_tilt_pipeline import TiltExperiment, simulate_tilt_images, measure_tilt_images


@dataclass
class CalibrationImages:
    ordinary: Experiment
    tilts: TiltExperiment
    beam_shifts: BeamShiftExperiment

    @property
    def image_count(self):
        return len(self.ordinary.images) + len(self.tilts.images) + len(self.beam_shifts.images)


def load_measurement_protocol(*, path):
    """Read the fixed lens controls and nominal image scales selected by the reduction study."""
    protocol = json.loads(Path(path).read_text())
    controls = np.asarray(protocol['lens_controls'])
    if (controls.ndim != 2 or controls.shape[1] != 5
            or len(protocol['nominal_magnifications']) != len(controls)
            or len(protocol['labels']) != len(controls) or not np.isfinite(controls).all()):
        raise ValueError('Supply five lens controls, one nominal scale and one label per setting.')
    if protocol['tilt_mrad'] != 3 or protocol['beam_shift_nm'] != 10:
        raise ValueError('This demonstrated protocol uses ±3 mrad tilts and ±10 nm beam shifts.')
    return protocol


def simulate_calibration_images(*, protocol, output_directory, random_seed, reuse_completed=True):
    """Generate exactly nine images per setting with the same MAGICAL/solenoid simulator.

    This works from an empty output directory; no previous notebook, fitted
    microscope, generated data or registration parameters are required.
    """
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    request = dict(protocol=protocol, random_seed=int(random_seed), simulator_version=1)
    marker = directory / 'image_request.json'
    if not (reuse_completed and marker.exists() and json.loads(marker.read_text()) == request):
        marker.unlink(missing_ok=True)
        spec = public_specification()
        template, metadata = make_magical_template(spec['voltage_v'], illumination_radius_m=100e-9)
        known = {key: spec[key] for key in ['voltage_v', 'pixel_pitch_m', 'shape_yx', 'feature_sigma_m']}
        known.update(read_noise_std_e=2., specimen=metadata)
        np.savez_compressed(directory / 'specimen_template.npz', **template)
        (directory / 'known_settings.json').write_text(json.dumps(known, indent=2))
        parameters, _ = _generator_parameters()
        (directory / 'scoring_truth.json').write_text(json.dumps({'parameters': parameters.tolist()}, indent=2))
        controls = np.asarray(protocol['lens_controls'])
        source = Experiment(np.empty((0, *known['shape_yx'])), controls,
            np.asarray(protocol['nominal_magnifications']), protocol['labels'], template, known, directory)
        ordinary = simulate_setting_images(source_experiment=source, lens_controls=controls,
            nominal_magnifications=source.nominal_magnifications, image_labels=source.image_labels,
            output_directory=directory, random_seed=random_seed)
        indices = list(range(len(controls)))
        simulate_tilt_images(experiment=ordinary, lens_setting_indices=indices,
                             tilt_mrad=3., random_seed=random_seed+1)
        simulate_beam_shift_images(experiment=ordinary, lens_setting_indices=indices,
                                   shift_nm=10., random_seed=random_seed+2)
        marker.write_text(json.dumps(request, indent=2))
    known = json.loads((directory / 'known_settings.json').read_text())
    data = dict(np.load(directory / 'images.npz'))
    ordinary = Experiment(data['images'], data['lens_controls'], data['nominal_magnifications'],
        data['image_labels'].tolist(), dict(np.load(directory / 'specimen_template.npz')), known, directory)
    tilt_data = dict(np.load(directory / 'tilt_images_3mrad.npz'))
    beam_data = dict(np.load(directory / 'beam_shift_images_10nm.npz'))
    for data in (tilt_data, beam_data):
        data['drift_std_pixels'] = float(data['drift_std_pixels'])
        for key in ['pair_labels', 'image_labels']:
            if key in data:
                data[key] = data[key].tolist()
    return CalibrationImages(ordinary, TiltExperiment(**tilt_data), BeamShiftExperiment(**beam_data))


def measure_calibration_images(*, images, reuse_completed=True):
    """Measure image scale/rotation/blur and signed tilt/beam displacements.

    Registration starts with nominal magnification and an image-based rotation
    search. It receives no true or previously fitted optical parameters.
    """
    directory = images.ordinary.output_directory
    fingerprint = hashlib.sha256()
    for batch in (images.ordinary, images.tilts, images.beam_shifts):
        for value in vars(batch).values():
            if isinstance(value, np.ndarray):
                fingerprint.update(str((value.shape, value.dtype)).encode())
                fingerprint.update(value.tobytes())
    fingerprint.update(json.dumps(images.ordinary.specimen_and_detector, sort_keys=True).encode())
    for key, value in sorted(images.ordinary.specimen_template.items()):
        fingerprint.update(key.encode())
        fingerprint.update(np.asarray(value).tobytes())
    key = fingerprint.hexdigest()
    marker = directory / 'measurement_request.json'
    path = directory / 'measurements.npz'
    if reuse_completed and marker.exists() and json.loads(marker.read_text()) == key and path.exists():
        return CalibrationMeasurements.load(path)
    marker.unlink(missing_ok=True)
    ordinary = images.ordinary
    measured = measure_images(images=ordinary.images, nominal_magnifications=ordinary.nominal_magnifications,
        specimen_template=ordinary.specimen_template, specimen_and_detector=ordinary.specimen_and_detector)
    tilts = measure_tilt_images(tilt_experiment=images.tilts, specimen_template=ordinary.specimen_template,
        specimen_and_detector=ordinary.specimen_and_detector, reference_measurements=measured)
    shifts = measure_beam_shift_images(beam_shift_experiment=images.beam_shifts,
        specimen_template=ordinary.specimen_template, specimen_and_detector=ordinary.specimen_and_detector,
        reference_measurements=measured)
    result = CalibrationMeasurements(ordinary.lens_controls, measured.values, measured.covariance, tilts, shifts)
    result.save(path)
    np.savez_compressed(directory / 'image_analysis.npz', image_parameters=measured.image_parameters,
                        image_error_in_noise_units=measured.image_error_in_noise_units)
    marker.write_text(json.dumps(key))
    return result


def fit_calibration(*, measurements, specimen_and_detector, starting_guess, position_bounds,
                    output_directory, number_of_starts=20, reuse_completed=True):
    """Fit displacements first, then all 19 parameters, with hard bounds and no soft penalties."""
    starts = make_starting_models(prior_model=starting_guess, position_priors=position_bounds,
                                 number_of_starts=number_of_starts, random_seed=482)
    return fit_measurement_dataset(measurements=measurements, specimen_and_detector=specimen_and_detector,
        starting_guess=starting_guess, position_bounds=position_bounds, starting_models=starts,
        output_directory=output_directory, reuse_completed=reuse_completed)


def check_calibration(*, fitted, calibration_images, validation_measurements, starting_guess, position_bounds):
    """Score generating parameters and new measured images without refitting the microscope."""
    if 'parameters' not in fitted:
        raise RuntimeError('No acceptable microscope fit was found.')
    recorded = calibration_images.ordinary.lens_controls
    fresh = validation_measurements.controls
    if np.any(np.all(np.isclose(recorded[:, None], fresh[None], rtol=0, atol=1e-12), axis=2)):
        raise ValueError('Prediction settings must be distinct from calibration settings.')
    problem = prepare_fit(measurements=validation_measurements,
        specimen_and_detector=calibration_images.ordinary.specimen_and_detector,
        prior_model=starting_guess, position_priors=position_bounds, use_soft_priors=False)
    parameters = np.asarray(fitted['parameters'])
    residual = np.asarray(problem.data_residual(parameters))
    factor = np.asarray(fitted['local_uncertainty']['covariance_factor_solver_coordinates'])
    prediction_derivative = np.asarray(problem.jacobian(problem.encode(parameters)))
    propagated = prediction_derivative @ factor
    predictive_covariance = np.eye(len(residual)) + propagated @ propagated.T
    predictive_residual = np.linalg.solve(np.linalg.cholesky(predictive_covariance), residual)
    errors = score_parameters(parameters=parameters, truth_directory=calibration_images.ordinary.output_directory)
    return dict(calibration_images=calibration_images.image_count,
                agreeing_starts=fitted['agreeing_starts'], starts=fitted['starts'],
                parameter_errors=errors,
                maximum_relative_parameter_error=max(abs(v['relative_error']) for v in errors.values()),
                validation_images=image_count(validation_measurements), validation_refitted=False,
                prediction_rms_including_fit_uncertainty=float(np.sqrt(np.mean(predictive_residual**2))),
                prediction_rms=float(np.sqrt(np.mean(residual**2))))


def show_calibration_result(*, fitted, checks):
    """Display a short result table and all 19 parameters in an expandable table."""
    import html
    from IPython.display import HTML, display
    from projector_notebook_helpers import _table
    _table(['Check', 'Result'], [
        ['Images used in fit', str(checks['calibration_images'])],
        ['Starting guesses reaching the same fit', f"{checks['agreeing_starts']}/{checks['starts']}"],
        ['Largest parameter error', f"{100*checks['maximum_relative_parameter_error']:.3f}%"],
        ['Largest local standard error', f"{100*fitted['local_uncertainty']['maximum_relative_standard_error']:.3f}%"],
        ['New images, no refit', f"{checks['validation_images']} images; RMS {checks['prediction_rms']:.3f} noise units"],
        ['Prediction including local fit uncertainty',
         f"RMS {checks['prediction_rms_including_fit_uncertainty']:.3f} uncertainty units"],
        ['Constraints', 'Hard bounds only; no soft penalties'],
    ])
    rows = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in
        [key, f"{r['truth']:.6g}", f"{r['fitted']:.6g}", f"{100*r['relative_error']:+.3f}%"]) + '</tr>'
        for key, r in checks['parameter_errors'].items())
    display(HTML('<details><summary>All 19 parameters: truth, fit and error</summary>'
        '<table><tr><th>Parameter</th><th>Truth</th><th>Fit</th><th>Error</th></tr>' + rows + '</table></details>'))
