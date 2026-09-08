"""Calibrated beam motion over a fixed MAGICAL specimen.

Image analysis separately fits specimen-pattern motion and illumination motion.
The geometric centre of the illumination then supplies A ds + B dtheta.
Only simulation and final scoring receive the generating microscope.
"""
import functools
import json
import operator
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from finite_projector_pipeline import LENSES, matrix5, predict_observables
from projector_landscape import compare_fit_landscapes
from projector_notebook_helpers import _table, model_to_parameters
from projector_tilt_pipeline import pair_tilt_measurements

from temgym_core.intensity import GaussianIntensityDetector
from temgym_core.jax_least_squares import least_squares


@dataclass
class BeamShiftExperiment:
    images: np.ndarray
    lens_controls: np.ndarray
    input_positions_m: np.ndarray
    input_angles_rad: np.ndarray
    nominal_magnifications: np.ndarray
    base_image_indices: np.ndarray
    image_labels: list[str]
    pair_labels: list[str]
    drift_std_pixels: float


def make_beam_shift_image_operator(spec, template):
    """A fixed specimen texture under a movable circular illumination envelope.

    Nine image parameters: log scale, pattern angle, log blur_px, pattern
    translation x/y_px, log dose, log background, beam centre x/y_nm in
    specimen-template coordinates. Beam motion reweights the fixed specimen;
    it does not translate the entire illuminated texture.
    """
    axis = jnp.asarray(template['axis_m'])
    xx, yy = jnp.meshgrid(axis, axis)
    centres = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)
    intensity = jnp.asarray(template['intensity']).ravel()
    radius = spec['specimen']['illumination_radius_m']
    edge_width = spec['specimen']['illumination_edge_m']
    sensor = GaussianIntensityDetector(tuple(spec['shape_yx']))

    def image(p):
        c, s = jnp.cos(p[1]), jnp.sin(p[1])
        rotation = jnp.array([[c, -s], [s, c]])
        xy = jnp.exp(p[0]) * (centres @ rotation.T) / spec['pixel_pitch_m'] + p[3:5]
        # Work on the full known specimen grid so shifting the beam reveals
        # texture outside the original circular footprint as well.
        distance = jnp.sqrt(jnp.sum((centres-p[7:9]*1e-9)**2, axis=1)+1e-40)
        weights = intensity * jax.nn.sigmoid((radius-distance)/edge_width)
        weights = weights / weights.sum()
        return sensor.image_pixels(xy,jnp.exp(2*p[2]),weights,jnp.exp(p[5]),jnp.exp(p[6]))
    return jax.jit(image)


def beam_centre_pixels(image_parameters, pixel_pitch_m):
    """Geometric illumination centre, distinct from a texture-weighted centroid."""
    p = image_parameters
    c, s = jnp.cos(p[1]), jnp.sin(p[1])
    rotation = jnp.array([[c, -s], [s, c]])
    return jnp.exp(p[0]) * rotation @ (p[7:9]*1e-9) / pixel_pitch_m + p[3:5]


def simulate_beam_shift_images(*, experiment, lens_setting_indices, shift_nm,
                               drift_std_pixels=0.003, random_seed=917,
                               angle_change_per_position_rad_per_m=None):
    """Make -/+ x and -/+ y beam-shift pairs on the existing MAGICAL region.

    Rays have r_s=s+u and theta_s=h*u+alpha+epsilon. The beam and its
    wavefront move together; mean angle and curvature stay fixed. The fixed
    specimen is reweighted by the shifted circular illumination. A supplied
    2x2 coupling matrix represents KNOWN accompanying mean-angle changes.
    """
    indices = np.asarray(lens_setting_indices)
    if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in 'iu'
            or np.any(indices < 0) or np.any(indices >= len(experiment.images))):
        raise ValueError('Supply valid base-image indices for beam-shift settings.')
    if not np.isfinite(shift_nm) or shift_nm <= 0 or shift_nm > 20:
        raise ValueError('Supply a beam-shift amplitude above zero and at most 20 nm for this template.')
    if not np.isfinite(drift_std_pixels) or drift_std_pixels < 0:
        raise ValueError('Drift standard deviation must be finite and nonnegative.')
    coupling = np.zeros((2, 2)) if angle_change_per_position_rad_per_m is None else np.asarray(angle_change_per_position_rad_per_m)
    if coupling.shape != (2, 2) or not np.isfinite(coupling).all():
        raise ValueError('Supply a finite 2x2 calibrated position-to-angle coupling.')
    truth = np.asarray(json.loads((experiment.output_directory / 'scoring_truth.json').read_text())['parameters'])
    spec = experiment.specimen_and_detector
    operator = make_beam_shift_image_operator(spec, experiment.specimen_template)
    c, s = np.cos(truth[18]), np.sin(truth[18])
    specimen_rotation = np.array([[c, -s], [s, c]])
    rng = np.random.default_rng(random_seed)
    images, controls, positions, angles, magnifications, base_indices, labels, pair_labels = [], [], [], [], [], [], [], []
    for index in indices:
        commands, nominal = experiment.lens_controls[index], experiment.nominal_magnifications[index]
        shape = np.asarray(predict_observables(truth, commands, spec))
        matrix = np.asarray(matrix5(truth, commands))
        b = matrix[:2, 2:4]
        for axis, name in zip(np.eye(2), ('x', 'y')):
            offset = rng.normal(0, .3, 2)
            pair_labels.append(f'{experiment.image_labels[index]}, {name} shift ±{shift_nm:g} nm')
            for sign in (-1, 1):
                position = sign * shift_nm * 1e-9 * axis
                angle = coupling @ position
                # At fixed specimen coordinate r, the incident mean angle is
                # h*(r-s)+alpha, so texture translation is B*(alpha-h*s).
                texture_shift = b @ (angle-truth[16]*1e4*position) / spec['pixel_pitch_m']
                texture_shift += offset + rng.normal(0, drift_std_pixels, 2)
                beam_in_template_nm = specimen_rotation.T @ position * 1e9
                p = np.r_[shape, texture_shift, np.log(4e7*(nominal/40000.)**2),
                          np.log(3.), beam_in_template_nm]
                mean = np.asarray(operator(p))
                images.append(rng.poisson(mean).astype(float) + rng.normal(0, spec['read_noise_std_e'], mean.shape))
                controls.append(commands)
                positions.append(position)
                angles.append(angle)
                magnifications.append(nominal)
                base_indices.append(index)
                labels.append(f'{experiment.image_labels[index]}, {name} {sign*shift_nm:+g} nm')
    result = BeamShiftExperiment(np.asarray(images), np.asarray(controls), np.asarray(positions),
                                 np.asarray(angles), np.asarray(magnifications), np.asarray(base_indices),
                                 labels, pair_labels, drift_std_pixels)
    np.savez_compressed(experiment.output_directory / f'beam_shift_images_{shift_nm:g}nm.npz', **vars(result))
    return result


def analyse_beam_shift_images(*, images, nominal_magnifications, specimen_template,
                              specimen_and_detector, initial_image_parameters, verbose=True):
    """Fit texture and illumination independently from pixels, with nine parameters.

    Earlier image registrations supply initial scale, angle, blur and shifts.
    The illumination centre starts at zero in every image. Analysis receives
    no calibrated input shifts, true image centres or microscope parameters.
    Full image-parameter covariance is propagated to the illumination centre.
    """
    images = np.asarray(images)
    seeds = np.asarray(initial_image_parameters)
    nominal = np.asarray(nominal_magnifications)
    spec = specimen_and_detector
    if (images.ndim != 3 or not len(images) or images.shape[1:] != tuple(spec['shape_yx'])
            or seeds.shape != (len(images), 7) or nominal.shape != (len(images),)):
        raise ValueError('Supply matching images, magnification labels and seven-parameter reference registrations.')
    if not all(np.isfinite(v).all() for v in (images, seeds, nominal)) or np.any(nominal <= 0):
        raise ValueError('Supply finite image inputs and positive magnification labels.')
    operator = make_beam_shift_image_operator(spec, specimen_template)
    operator_jacobian = jax.jit(jax.jacfwd(operator))
    centre = jax.jit(lambda p: beam_centre_pixels(p, spec['pixel_pitch_m']))
    centre_jacobian = jax.jit(jax.jacfwd(centre))
    parameters, covariances, centres, centre_covariances, errors = [], [], [], [], []
    def residual(p,image,sigma):
        return ((operator(p)-image)/sigma).ravel()
    def derivative_image(p,image,sigma):
        return (operator_jacobian(p)/sigma[...,None]).reshape(-1,9)
    for index, image in enumerate(images):
        lower = np.r_[np.log(nominal[index]/2), -4*np.pi, np.log(.15), -20., -20., np.log([1e3, .01]), -30., -30.]
        upper = np.r_[np.log(nominal[index]*2), 4*np.pi, np.log(60.), 20., 20., np.log([1e12, 1e6]), 30., 30.]
        start = np.clip(np.r_[seeds[index], 0., 0.], lower+1e-8, upper-1e-8)
        sigma = np.sqrt(np.maximum(image, 1.) + spec['read_noise_std_e']**2)
        result = least_squares(residual,start,jac=derivative_image,
                               args=(jnp.asarray(image),jnp.asarray(sigma)),
                               bounds=(lower,upper),x_scale='jac',max_nfev=150,max_iter=300,
                               ftol=1e-9,xtol=1e-9,gtol=1e-5)
        rms = float(np.sqrt(np.mean(result.residual**2)))
        if not result.success or rms > 1.5:
            raise RuntimeError(f'Beam-shift registration failed at frame {index}: {result.message}; RMS {rms:.3f}')
        covariance = np.linalg.inv(np.asarray(result.jacobian).T @ np.asarray(result.jacobian))
        derivative = np.asarray(centre_jacobian(result.x))
        parameters.append(result.x)
        covariances.append(covariance)
        centres.append(np.asarray(centre(result.x)))
        centre_covariances.append(derivative @ covariance @ derivative.T)
        errors.append(rms)
        if verbose:
            print(f'MAGICAL beam-shift image {index+1}/{len(images)}: weighted pixel RMS {rms:.3f}', flush=True)
    return {'image_parameters': np.asarray(parameters), 'parameter_covariance': np.asarray(covariances),
                'beam_centres_pixels': np.asarray(centres), 'beam_centre_covariances': np.asarray(centre_covariances),
                'image_residual_rms': np.asarray(errors)}


def measure_beam_shift_images(*, beam_shift_experiment, specimen_template,
                              specimen_and_detector, reference_measurements):
    """Pixels -> pattern and illumination fits -> beam-centre pair differences."""
    experiment = beam_shift_experiment
    measured = analyse_beam_shift_images(
        images=experiment.images, nominal_magnifications=experiment.nominal_magnifications,
        specimen_template=specimen_template, specimen_and_detector=specimen_and_detector,
        initial_image_parameters=reference_measurements.image_parameters[experiment.base_image_indices],
    )
    result = pair_tilt_measurements(
        lens_controls=experiment.lens_controls, input_tilts_rad=experiment.input_angles_rad,
        positions_pixels=measured['beam_centres_pixels'], position_covariances=measured['beam_centre_covariances'],
        drift_std_pixels=experiment.drift_std_pixels, pair_labels=experiment.pair_labels,
    )
    positions = np.asarray(experiment.input_positions_m)
    if positions.shape != (len(experiment.images), 2) or not np.isfinite(positions).all():
        raise ValueError('Supply finite calibrated x/y beam positions for every exposure.')
    result['position_changes_m'] = positions[1::2]-positions[::2]
    result['image_analysis'] = measured
    return result


def combine_beam_shift_measurements(*measurement_sets):
    keys = ('lens_controls', 'position_changes_m', 'angle_changes_rad',
            'displacements_pixels', 'covariance_pixels')
    result = {key: np.concatenate([m[key] for m in measurement_sets]) for key in keys}
    result['pair_labels'] = functools.reduce(operator.iadd, (list(m['pair_labels']) for m in measurement_sets), [])
    return result


def show_beam_shift_measurements(measurements):
    errors = np.sqrt(np.diagonal(measurements['covariance_pixels'], axis1=1, axis2=2))
    rows = [[label, f'{position[0]*1e9:+g}, {position[1]*1e9:+g}',
             f'{value[0]:+.4f} ± {error[0]:.4f}', f'{value[1]:+.4f} ± {error[1]:.4f}']
            for label, position, value, error in zip(measurements['pair_labels'], measurements['position_changes_m'],
                                                    measurements['displacements_pixels'], errors)]
    _table(['Pair', 'Known dx, dy (nm)', 'Measured dx (pixels)', 'Measured dy (pixels)'], rows)


def compare_beam_shift_fits(*, without_beam_shifts, with_beam_shifts, lens_controls, true_model):
    """Score A, B, curvature and positions; compare the same detector profiles."""
    compare_fit_landscapes(original=without_beam_shifts, extra_wobbles=with_beam_shifts,
                          true_model=true_model, labels=('Tilts', 'Tilts + beam shifts'))
    models = [true_model, without_beam_shifts.best_model, with_beam_shifts.best_model]
    controls = np.unique(np.asarray(lens_controls), axis=0)
    matrices = np.asarray(jax.jit(jax.vmap(lambda p: jax.vmap(lambda c: matrix5(p, c))(controls)))(
        np.array([model_to_parameters(m) for m in models])))
    report, rows = {}, []
    for name, columns in [('A', slice(0, 2)), ('B', slice(2, 4))]:
        blocks = matrices[:, :, :2, columns]
        errors = [float(np.linalg.norm(block-blocks[0])/np.linalg.norm(blocks[0])) for block in blocks[1:]]
        report[f'relative_{name.lower()}_error'] = errors
        rows.append([f'{name} error', '0%', *[f'{100*error:.4f}%' for error in errors]])
    for key, label in [('beam_curvature_per_m', 'Curvature (1/m)'), ('detector_position_mm', 'Detector z (mm)')]:
        values = [model[key] for model in models]
        report[key] = values
        rows.append([label, *[f'{v:.3f}' for v in values]])
    _table(['Quantity', 'Truth', 'Tilts', 'Tilts + beam shifts'], rows)
    rows = []
    for name in LENSES:
        values = [m['lens_positions_mm'][name] for m in models]
        rows.append([name, *[f'{v:.3f}' for v in values],
                     f'{values[1]-values[0]:+.3f}', f'{values[2]-values[0]:+.3f}'])
    _table(['Lens', 'True z (mm)', 'Tilts', 'Tilts + beam shifts', 'Old error (mm)', 'New error (mm)'], rows)
    return report
