"""Extra wobble images and repeatable exploration of the calibration objective.

Only simulate_extra_wobbles reads the hidden generating model. The landscape
fits receive measured arrays and one fixed set of priors, never the truth.
"""
from dataclasses import dataclass, fields
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import jax

from finite_projector_pipeline import LENSES, prepare_column_fit, predict_observables, matrix5
from magical_projector_pipeline import make_magical_image_operator
from projector_notebook_helpers import (
    Experiment, ImageMeasurements, model_to_parameters, parameters_to_model, measure_images, _table,
)


def simulate_extra_wobbles(*, experiment, reference_image_index, wobble_fractions,
                           random_seed=825):
    """Generate only new exposures at existing reference controls, using the same specimen."""
    fractions = np.asarray(wobble_fractions, dtype=float)
    if (fractions.ndim != 1 or not len(fractions) or not np.isfinite(fractions).all()
            or np.any(fractions <= 0) or np.any(fractions > .1) or len(np.unique(fractions)) != len(fractions)):
        raise ValueError("Supply distinct wobble fractions greater than zero and at most 0.1.")
    if not isinstance(reference_image_index, (int, np.integer)) or not 0 <= reference_image_index < len(experiment.images):
        raise ValueError("Supply a valid reference image index.")
    base = experiment.lens_controls[reference_image_index]
    nominal = experiment.nominal_magnifications[reference_image_index]
    controls, labels = [], []
    for fraction in fractions:
        for name, axis in zip(LENSES[1:], np.eye(4)):
            for sign in (-1, 1):
                controls.append(base*(1+sign*fraction*axis))
                labels.append(f"{nominal:,.0f}× {name} {sign*fraction:+.1%}")
    spec = experiment.specimen_and_detector
    truth_record = (experiment.output_directory / "scoring_truth.json").read_text()
    truth = np.asarray(json.loads(truth_record)["parameters"])
    operator = make_magical_image_operator(spec, experiment.specimen_template)
    rng = np.random.default_rng(random_seed)
    images = []
    for commands in controls:
        values = np.asarray(predict_observables(truth, commands, spec))
        p = np.r_[values, rng.normal(0, .3, 2), np.log(4e7*(nominal/40000.)**2), np.log(3.)]
        mean = np.asarray(operator(p))
        images.append(rng.poisson(mean).astype(float)+rng.normal(0, spec["read_noise_std_e"], mean.shape))
    directory = experiment.output_directory / "extra_wobbles"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "scoring_truth.json").write_text(truth_record)
    np.savez_compressed(directory / "images.npz", images=images, lens_controls=controls,
                        nominal_magnifications=np.full(len(images), nominal), image_labels=labels)
    return Experiment(np.asarray(images), np.asarray(controls), np.full(len(images), nominal),
                      labels, experiment.specimen_template, spec, directory)


def combine_image_measurements(*measurement_sets):
    """Concatenate independently analysed frames, preserving their acquisition order."""
    return ImageMeasurements(**{field.name: np.concatenate([getattr(m, field.name) for m in measurement_sets])
                                for field in fields(ImageMeasurements)})


def measure_extra_wobble_images(*, experiment, reference_measurements, reference_image_index):
    """Use an already measured reference image to initialise each new registration.

    This avoids a fresh angle search on the periodic specimen at a substantially
    changed scale. The reference is an image measurement, never lens truth.
    All seven image parameters are independently refitted in every new frame.
    """
    seed = reference_measurements.image_parameters[reference_image_index]
    return measure_images(images=experiment.images, nominal_magnifications=experiment.nominal_magnifications,
                          specimen_template=experiment.specimen_template,
                          specimen_and_detector=experiment.specimen_and_detector,
                          initial_image_parameters=np.tile(seed, (len(experiment.images), 1)))


def combine_tilt_measurements(*measurement_sets):
    """Combine independent pairs; original pairs appear once in the larger experiment."""
    keys = ("lens_controls", "angle_changes_rad", "displacements_pixels", "covariance_pixels")
    result = {key: np.concatenate([m[key] for m in measurement_sets]) for key in keys}
    result["pair_labels"] = sum((list(m["pair_labels"]) for m in measurement_sets), [])
    return result


def make_starting_models(*, prior_model, position_priors, number_of_starts, random_seed=482):
    """One declared guess plus reproducible, dispersed starts within the same bounds.

    Position starts cover the inner 80% of the allowed boxes. Lengths and gains
    vary around the fixed prior model; this is an exploration, not a posterior
    sample or an exhaustive search of the 19-dimensional parameter space.
    """
    from finite_projector_pipeline import position_prior_coordinates
    if number_of_starts < 1:
        raise ValueError("At least one starting model is required.")
    p = model_to_parameters(prior_model)
    encode, decode, lower, upper = position_prior_coordinates(position_priors)
    centre = np.asarray(encode(p))
    length_bounds = np.log(np.asarray(position_priors["length_bounds_mm"])/1000)
    rng = np.random.default_rng(random_seed)
    models = [prior_model]
    for _ in range(number_of_starts-1):
        u = centre.copy()
        u[:6] = lower + rng.uniform(.1, .9, 6)*(upper-lower)
        u[6:11] = np.clip(u[6:11]+rng.normal(0, .25, 5), length_bounds[:, 0]+.01, length_bounds[:, 1]-.01)
        u[11] = rng.uniform(-.6, -.25)
        u[12:16] += rng.uniform(-.25, .25, 4)
        u[16] = rng.uniform(-.2, .2)
        u[17] += rng.uniform(-.25, .25)
        u[18] = rng.uniform(-.4, .4)
        models.append(parameters_to_model(np.asarray(decode(u))))
    return models


@dataclass
class FitLandscape:
    starting_models: list
    fits: list
    detector_positions_mm: np.ndarray
    profile_fits: list

    @property
    def best_fit(self):
        valid = [r for r in self.fits if r.success]
        if not valid:
            raise RuntimeError("No starting point produced a converged fit; inspect the saved diagnostics.")
        return min(valid, key=lambda r: r.cost)

    @property
    def best_model(self):
        return parameters_to_model(self.best_fit.x)


def explore_fit_landscape(*, lens_controls, image_measurements, tilt_measurements,
                          specimen_and_detector, prior_model, position_priors,
                          starting_models, detector_positions_mm, output_directory,
                          max_evaluations=4000):
    """Repeat a fixed fit, then profile detector position with two warm-start sweeps.

    Each profile point fixes one coordinate and refits the other 18. Both
    sweeps start from the best unrestricted fit and visit the grid in opposite
    orders. Keep the lower converged objective at each point. These sampled
    minima can miss branches; no exhaustive minimum count is claimed.
    """
    positions = np.asarray(detector_positions_mm, dtype=float)
    bounds = position_priors["position_bounds_mm"][5]
    if (positions.ndim != 1 or len(positions) < 3 or not np.isfinite(positions).all()
            or np.any(np.diff(positions) <= 0) or positions[0] < bounds[0] or positions[-1] > bounds[1]):
        raise ValueError("Supply at least three increasing detector positions within the prior bounds.")
    problem = prepare_column_fit(
        lens_controls=lens_controls, measured_values=image_measurements.values,
        measurement_covariances=image_measurements.covariance, tilt_measurements=tilt_measurements,
        specimen_and_detector=specimen_and_detector, prior_parameters=model_to_parameters(prior_model),
        position_priors=position_priors,
    )
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "fit_inputs.json").write_text(json.dumps(dict(
        prior_model=prior_model, position_priors=position_priors, starting_models=starting_models,
        detector_positions_mm=positions.tolist(), max_evaluations=max_evaluations), indent=2))
    fits = []
    for i, model in enumerate(starting_models):
        result = problem.solve(model_to_parameters(model), max_evaluations=max_evaluations)
        fits.append(result)
        print(f"Start {i+1}/{len(starting_models)}: total error {2*result.cost:.3f}, "
              f"{'converged' if result.success else 'not converged'}", flush=True)
        _save_results(directory / "starts.json", fits)
    study = FitLandscape(starting_models, fits, positions, [None]*len(positions))
    # Long shallow valleys can exhaust the first budget. Continue promising
    # unfinished runs before interpreting their endpoints as landscape evidence.
    for i, result in enumerate(study.fits):
        if not result.success and 2*result.cost < 2*(2*study.best_fit.cost)+20:
            retry = problem.solve(result.x, max_evaluations=max(12000, 3*max_evaluations))
            if retry.cost <= result.cost:
                study.fits[i] = retry
            print(f"Continued start {i+1}: total error {2*retry.cost:.3f}, "
                  f"{'converged' if retry.success else 'not converged'}", flush=True)
            _save_results(directory / "starts.json", study.fits)
    best = study.best_fit
    for reverse in (False, True):
        seed = best.x
        order = range(len(positions)-1, -1, -1) if reverse else range(len(positions))
        for i in order:
            result = problem.solve(seed, max_evaluations=max_evaluations,
                                   fixed_detector_position_mm=float(positions[i]))
            if result.success:
                seed = result.x
            old = study.profile_fits[i]
            if old is None or (result.success and (not old.success or result.cost < old.cost)):
                study.profile_fits[i] = result
            print(f"Detector {positions[i]:.0f} mm ({'reverse' if reverse else 'forward'}): "
                  f"total error {2*result.cost:.3f}, {'converged' if result.success else 'not converged'}", flush=True)
        _save_results(directory / "detector_profile.json", study.profile_fits)
    # A constrained scan can find a branch missed by the starting guesses.
    candidate = min((r for r in study.profile_fits if r.success), key=lambda r: r.cost, default=None)
    if candidate is not None and candidate.cost < study.best_fit.cost-1e-4:
        polished = problem.solve(candidate.x, max_evaluations=max_evaluations)
        study.fits.append(polished)
        print("Added an unrestricted fit from the better profile branch.", flush=True)
        _save_results(directory / "starts.json", study.fits)
    return study


def _save_results(path, results):
    path.write_text(json.dumps([dict(
        model=parameters_to_model(r.x), parameters=r.x.tolist(),
        total_squared_residual=float(2*r.cost),
        data_squared_residual=float(r.data_residual @ r.data_residual),
        success=bool(r.success), message=str(r.message), evaluations=int(r.nfev),
        optimality=float(r.optimality), fixed_detector_position_mm=r.fixed_detector_position_mm,
    ) for r in results], indent=2))


def load_fit_landscape(output_directory):
    """Read completed searches for comparison, without rerunning their fits.

    Restores the saved scalar data costs as scalars; individual residuals and
    Jacobians are not reconstructed from those summary statistics.
    """
    directory = Path(output_directory)
    inputs = json.loads((directory / 'fit_inputs.json').read_text())
    def read(name):
        records = json.loads((directory / name).read_text())
        return [SimpleNamespace(
            x=np.asarray(r['parameters']), cost=r['total_squared_residual']/2,
            data_squared_residual=r['data_squared_residual'], success=r['success'],
            message=r['message'], nfev=r['evaluations'], optimality=r['optimality'],
            fixed_detector_position_mm=r['fixed_detector_position_mm'],
        ) for r in records]
    fits, profiles = read('starts.json'), read('detector_profile.json')
    positions = np.asarray(inputs['detector_positions_mm'])
    if len(fits) < len(inputs['starting_models']) or len(profiles) != len(positions):
        raise ValueError('The saved landscape search is incomplete.')
    return FitLandscape(inputs['starting_models'], fits, positions, profiles)


def _data_error(result):
    if hasattr(result, 'data_squared_residual'):
        return result.data_squared_residual
    return float(result.data_residual @ result.data_residual)


def compare_fit_landscapes(*, original, extra_wobbles, true_model=None,
                          labels=("Original measurements", "Added 5% and 10% wobbles")):
    """Show fit endpoints and best profiled error; truth is for scoring only."""
    studies = list(zip(labels, (original, extra_wobbles)))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), constrained_layout=True)
    for label, study in studies:
        models = [parameters_to_model(r.x) for r in study.fits]
        costs = np.array([2*r.cost for r in study.fits])
        good = np.array([r.success for r in study.fits])
        heights = costs - 2*study.best_fit.cost
        coordinates = [np.array([m['detector_position_mm'] for m in models]),
                       np.array([m['lens_positions_mm']['PL1'] for m in models]),
                       np.array([m['lens_positions_mm']['Objective'] for m in models])]
        for ax, values in zip(axes, coordinates):
            ax.scatter(values[good], heights[good], label=label, s=25, alpha=.7)
            ax.scatter(values[~good], heights[~good], marker='x', color='gray', s=35)
            ax.set_yscale('symlog', linthresh=1.)
    for ax, name in zip(axes, ['Detector', 'PL1', 'Objective']):
        ax.set(xlabel=f'{name} position (mm)', ylabel='Total error above best fit')
        if true_model is not None:
            z = true_model['detector_position_mm'] if name == 'Detector' else true_model['lens_positions_mm'][name]
            ax.axvline(z, color='black', linestyle=':', linewidth=1)
    axes[0].legend(fontsize=8)
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), constrained_layout=True)
    rows = []
    for label, study in studies:
        success = np.array([r.success for r in study.profile_fits])
        total = np.array([2*r.cost for r in study.profile_fits])
        data = np.array([_data_error(r) for r in study.profile_fits])
        total[~success], data[~success] = np.nan, np.nan
        axes[0].plot(study.detector_positions_mm, total-2*study.best_fit.cost, 'o-', label=label)
        axes[1].plot(study.detector_positions_mm, data-np.nanmin(data), 'o-', label=label)
        close = [r for r in study.fits if r.success and 2*(r.cost-study.best_fit.cost) <= 1.]
        z = [parameters_to_model(r.x)['detector_position_mm'] for r in close]
        rows.append([label, f'{sum(r.success for r in study.fits)}/{len(study.fits)}',
                     len(close), f'{min(z):.2f}–{max(z):.2f}', f'{study.best_model["detector_position_mm"]:.2f}'])
    axes[0].set(ylabel='Total error above best unrestricted fit', title='Detector fixed; other 18 parameters refitted')
    axes[1].set(ylabel='Data error above its smallest scanned value', title='Data contribution at those same fits')
    for ax in axes:
        ax.set_xlabel('Fixed detector position (mm)')
        ax.legend(fontsize=8)
        if true_model is not None:
            ax.axvline(true_model['detector_position_mm'], color='black', linestyle=':', linewidth=1)
    plt.show()
    _table(['Measurements', 'Converged fits', 'Within 1 error unit of best',
            'Their detector positions (mm)', 'Best detector position (mm)'], rows)
    print('An error unit is one squared residual divided by its measurement variance. '
          'The near-best count counts fitted endpoints, not distinct minima. '
          'Profiles use the same fixed priors; the right panel is their data contribution, '
          'not a separately optimized data-only profile.')


def compare_tilt_amplitudes(*, one_mrad, one_and_three_mrad, lens_controls, true_model):
    """Compare geometry and B accuracy on the same controls; truth is scoring only."""
    compare_fit_landscapes(original=one_mrad, extra_wobbles=one_and_three_mrad,
                          true_model=true_model, labels=('±1 mrad', '±1 mrad plus ±3 mrad'))
    controls = np.unique(np.asarray(lens_controls), axis=0)
    matrices = jax.jit(jax.vmap(lambda p: jax.vmap(lambda d: matrix5(p, d))(controls)))
    models = [true_model, one_mrad.best_model, one_and_three_mrad.best_model]
    b = np.asarray(matrices(np.array([model_to_parameters(m) for m in models])))[:, :, :2, 2:4]
    b_errors = np.sqrt(np.sum((b[1:]-b[0])**2, axis=(1,2,3)))/np.linalg.norm(b[0])
    rows = []
    for name in (*LENSES, 'Detector'):
        values = [m['detector_position_mm'] if name == 'Detector' else m['lens_positions_mm'][name] for m in models]
        rows.append([name, *[f'{v:.3f}' for v in values],
                     f'{values[1]-values[0]:+.3f}', f'{values[2]-values[0]:+.3f}'])
    _table(['Element', 'True z (mm)', '±1 mrad fit', '±1 and ±3 mrad fit',
            'Old position error (mm)', 'New position error (mm)'], rows)
    for label, error in zip(('±1 mrad', '±1 mrad plus ±3 mrad'), b_errors):
        print(f'{label}: relative B error at the same measured lens settings = {100*error:.3f}%')
    return dict(relative_b_error_one_mrad=float(b_errors[0]),
                relative_b_error_one_and_three_mrad=float(b_errors[1]),
                detector_error_one_mrad_mm=abs(models[1]['detector_position_mm']-models[0]['detector_position_mm']),
                detector_error_one_and_three_mrad_mm=abs(models[2]['detector_position_mm']-models[0]['detector_position_mm']))
