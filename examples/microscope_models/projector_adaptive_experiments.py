"""Choose informative lens settings, predict new images, measure them, and refit.

Design and fitting use the current inferred microscope and recorded measurements.
Only simulate_setting_images and score_parameters read generating truth.
Five control columns are always Objective, IL1, IL2, IL3, PL1.
"""
from dataclasses import dataclass
import json
from pathlib import Path
from itertools import combinations, product

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from finite_projector_pipeline import matrix5, predict_observables, prepare_column_fit
from magical_projector_pipeline import make_magical_image_operator
from projector_notebook_helpers import Experiment, measure_images, model_to_parameters, parameters_to_model
from projector_landscape import combine_tilt_measurements
from projector_tilt_pipeline import simulate_tilt_images, measure_tilt_images
from projector_beam_shift_pipeline import (
    simulate_beam_shift_images, measure_beam_shift_images, combine_beam_shift_measurements,
)


def five_controls(controls):
    controls = np.asarray(controls)
    if controls.ndim != 2 or controls.shape[1] not in (4, 5):
        raise ValueError('Supply four projector controls or five controls with Objective first.')
    return np.c_[np.ones(len(controls)), controls] if controls.shape[1] == 4 else controls.copy()


@dataclass
class CalibrationMeasurements:
    controls: np.ndarray
    values: np.ndarray
    covariance: np.ndarray
    tilts: dict
    shifts: dict

    def save(self, path):
        np.savez_compressed(path, controls=self.controls, values=self.values, covariance=self.covariance,
                            **{f'tilt_{k}':v for k,v in self.tilts.items() if k != 'image_analysis'},
                            **{f'shift_{k}':v for k,v in self.shifts.items() if k != 'image_analysis'})

    @classmethod
    def load(cls, path):
        data = dict(np.load(path))
        def pairs(prefix):
            result = {k[len(prefix):]:v for k,v in data.items() if k.startswith(prefix)}
            result['pair_labels'] = result['pair_labels'].tolist()
            return result
        return cls(data['controls'], data['values'], data['covariance'], pairs('tilt_'), pairs('shift_'))

    def append(self, other):
        return CalibrationMeasurements(np.concatenate([self.controls, other.controls]),
                                       np.concatenate([self.values, other.values]),
                                       np.concatenate([self.covariance, other.covariance]),
                                       combine_tilt_measurements(self.tilts, other.tilts),
                                       combine_beam_shift_measurements(self.shifts, other.shifts))


def load_existing_measurements(directory):
    directory = Path(directory)
    checkpoint=directory/'beam_shift_landscape/calibration_measurements.npz'
    if checkpoint.exists():
        measurements=CalibrationMeasurements.load(checkpoint)
        measurements.controls=five_controls(measurements.controls)
        for pairs in (measurements.tilts,measurements.shifts):
            pairs['lens_controls']=five_controls(pairs['lens_controls'])
        return measurements
    data = dict(np.load(directory/'beam_shift_baseline_measurements.npz'))
    tilts = {k[5:]:v for k,v in data.items() if k.startswith('tilt_')}
    shifts = dict(np.load(directory/'beam_shift_measured_displacements.npz'))
    for pairs in (tilts, shifts):
        pairs['lens_controls'] = five_controls(pairs['lens_controls'])
        pairs['pair_labels'] = pairs['pair_labels'].tolist()
    measurements=CalibrationMeasurements(five_controls(data['lens_controls']), data['image_values'],
                                         data['image_covariance'], tilts, shifts)
    measurements.save(checkpoint)
    return measurements


def prepare_fit(*, measurements, specimen_and_detector, prior_model, position_priors):
    return prepare_column_fit(
        lens_controls=measurements.controls, measured_values=measurements.values,
        measurement_covariances=measurements.covariance,
        tilt_measurements=measurements.tilts, beam_shift_measurements=measurements.shifts,
        specimen_and_detector=specimen_and_detector, prior_parameters=model_to_parameters(prior_model),
        position_priors=position_priors,
    )


def estimate_strengths_from_beam_rotation(*, beam_shift_measurements, starting_model):
    """Initialise lens strengths from measured beam-shift rotation alone.

    For round solenoids A is a real scalar times R(-sum(q_i)). Therefore
    the angle between an input shift and its image is -sum(q_i), modulo pi.
    A negative radial magnification adds pi and is handled by that wrapping.
    This estimate is only an optimizer start: the joint fit still fits all
    19 parameters using each original measurement once and unchanged priors.
    """
    m=beam_shift_measurements
    if np.any(np.asarray(m['angle_changes_rad']) != 0):
        raise ValueError('Rotation initialisation requires pure calibrated beam shifts.')
    controls=five_controls(m['lens_controls'])
    positions=np.asarray(m['position_changes_m'])
    shifts=np.asarray(m['displacements_pixels'])
    norm2=np.sum(shifts**2,axis=1)
    if np.any(norm2 <= 0) or np.any(np.sum(positions**2,axis=1) <= 0):
        raise ValueError('Rotation needs nonzero input and measured shifts.')
    angles=np.arctan2(shifts[:,1],shifts[:,0])-np.arctan2(positions[:,1],positions[:,0])
    derivative=np.c_[-shifts[:,1],shifts[:,0]]/norm2[:,None]
    variance=np.einsum('ni,nij,nj->n',derivative,m['covariance_pixels'],derivative)
    sigma=np.sqrt(variance)
    p=model_to_parameters(starting_model)
    initial=np.r_[p[11],np.array([1.,1.,1.,-1.])*np.exp(p[12:16])]
    lower=np.r_[-.9,np.exp(p[12:15]-.8),-np.exp(p[15]+.8)]
    upper=np.r_[-.15,np.exp(p[12:15]+.8),-np.exp(p[15]-.8)]
    def residual(q):
        difference=angles+controls@q
        return .5*np.arctan2(np.sin(2*difference),np.cos(2*difference))/sigma
    fit=least_squares(residual,initial,jac=lambda q:controls/sigma[:,None],
                      bounds=(lower,upper),x_scale='jac',max_nfev=1000,
                      xtol=1e-12,ftol=1e-12,gtol=1e-8)
    if not fit.success or np.sqrt(np.mean(fit.fun**2))>2:
        raise RuntimeError('Beam rotations did not give a consistent round-lens strength estimate.')
    model=dict(starting_model)
    model['objective_strength']=float(fit.x[0])
    model['excitation_gains']=dict(zip(['IL1','IL2','IL3','PL1'],np.abs(fit.x[1:]).tolist()))
    return model


def setting_observables(parameters, controls, spec):
    """Seven independent round-column responses for prospective design.

    First three are feature log-scale, rotation and log-blur. Last four are
    x/y detector motion for a 20 nm x shift and a 6 mrad x tilt. The measured
    y-axis pairs repeat this information independently in our round-lens model.
    """
    def one(control):
        matrix = matrix5(parameters, control)
        return jnp.r_[predict_observables(parameters, control, spec),
                      matrix[:2,0]*20e-9/spec['pixel_pitch_m'],
                      matrix[:2,2]*.006/spec['pixel_pitch_m']]
    return jax.vmap(one)(controls)


def make_candidate_settings(*, current_parameters, reference_controls, specimen_and_detector,
                            random_seed=772, random_projector_settings=1000,
                            wider_compensation=False):
    """Screen two-lens grids, mixed projector changes, and refocused objective changes.

    Refocusing adjusts IL1/IL2 in the current fitted model to keep scale and B
    useful. Candidates are retained at 20k–53k effective magnification and
    less than 3 pixels of blur. The optional wider sweep changes IL3 and PL1
    more strongly, with the same visible-image constraints. No truth is used.
    """
    p = jnp.asarray(current_parameters)
    base = five_controls(np.asarray(reference_controls)[None])[0]
    spec = specimen_and_detector
    rng = np.random.default_rng(random_seed)
    rows, categories = [], []
    for fraction in [.025,.05,.1,.2]:
        for i,j in combinations(range(1,5),2):
            for signs in product([-1,1],repeat=2):
                c=base.copy(); c[[i,j]] *= 1+fraction*np.asarray(signs)
                rows.append(c); categories.append('two-lens grid')
    for _ in range(random_projector_settings):
        c=base.copy(); c[1:] *= 1+rng.uniform(-.2,.2,4)
        rows.append(c); categories.append('mixed projectors')

    def focus_error(v, fixed, target_scale, target_b):
        commands=fixed.at[1:3].set(v)
        matrix=matrix5(p,commands)
        e=matrix[:2,:2]+p[16]*1e4*matrix[:2,2:4]
        scale=jnp.linalg.norm(e[:,0])
        return jnp.array([jnp.log(scale/target_scale),
                          (jnp.dot(matrix[:2,2],e[:,0])/scale-target_b)/10])
    error=jax.jit(focus_error); derivative=jax.jit(jax.jacfwd(focus_error))
    requested=[]
    if wider_compensation:
        for objective,il3,pl1,scale,b in product(
                [.9995,1.,1.001,1.002], [.65,.85,1.,1.2,1.4], [.5,.75,1.,1.25,1.5],
                [30000.,50000.], [-.15,.15]):
            c=base.copy();c[0]=objective;c[3:]*=[il3,pl1]
            requested.append((c,scale,b,'wide compensated settings'))
    else:
        for objective,scale,b in product([.9995,.99975,.9999,1.0001,1.00025,1.0005,1.001,1.002],
                                         [30000.,40000.,50000.],[-.1,0.,.1]):
            for _ in range(3):
                c=base.copy();c[0]=objective;c[3:]*=1+rng.uniform(-.1,.1,2)
                requested.append((c,scale,b,'objective compensated settings'))
    for c,scale,b,category in requested:
        fixed=jnp.asarray(c)
        fit=least_squares(lambda v:np.asarray(error(v,fixed,scale,b)),base[1:3],
                          jac=lambda v:np.asarray(derivative(v,fixed,scale,b)),
                          bounds=(base[1:3]*.05,base[1:3]*4.),max_nfev=1000,
                          xtol=1e-11,ftol=1e-11,gtol=1e-9)
        if np.linalg.norm(fit.fun)<1e-5:
            c=c.copy();c[1:3]=fit.x;rows.append(c);categories.append(category)
    rows=np.asarray(rows)
    predictions=np.asarray(jax.jit(lambda c:setting_observables(p,c,spec))(rows))
    scale,blur=np.exp(predictions[:,0]),np.exp(predictions[:,2])
    keep=(scale>=20000)&(scale<=53000)&(blur<3.)
    rows=rows[keep]
    names=['Objective','IL1','IL2','IL3','PL1']
    labels=[', '.join(f'{name} {100*(v-1):+.3g}%' for name,v in zip(names,c/base) if abs(v-1)>1e-9)
            for c in rows]
    return dict(lens_controls=rows, labels=np.asarray(labels), categories=np.asarray(categories)[keep],
                predicted_observables=predictions[keep], reference_controls=base,
                attempted_settings=len(categories)+len(requested)-sum('compensated' in c for c in categories))


def select_informative_settings(*, candidates, current_parameters, current_problem,
                                specimen_and_detector, number_of_settings, allowed=None):
    """Greedy information gain, starting with the CURRENT data and fixed priors.

    Candidate scores use approximate noise from the earlier experiment. Actual
    uncertainties are later measured from the acquired images. An SVD of the
    existing Jacobian provides stable whitening without squaring its condition
    number. Selection receives no truth and no future image measurements.
    """
    controls = np.asarray(candidates['lens_controls'])
    valid = np.ones(len(controls), bool) if allowed is None else np.asarray(allowed, bool)
    u = current_problem.encode(current_parameters)
    existing = np.asarray(current_problem.jacobian(u))
    _, singular, vt = np.linalg.svd(existing, full_matrices=False)
    transform = vt.T / np.maximum(singular, 1e-12)
    jacobian = jax.jit(jax.jacfwd(lambda v: setting_observables(
        current_problem.decode(v), controls, specimen_and_detector)))
    noise = np.array([3.4e-5,2.8e-4,3.8e-3,.00456/np.sqrt(2),.00456/np.sqrt(2),
                      .00456/np.sqrt(2),.00456/np.sqrt(2)])
    h = (np.asarray(jacobian(u))/noise[None,:,None]) @ transform
    information = np.eye(19)
    selected, gains = [], []
    for _ in range(min(number_of_settings, int(valid.sum()))):
        covariance = np.linalg.inv(information)
        blocks = np.eye(7)[None] + h @ covariance @ h.transpose(0,2,1)
        signs, scores = np.linalg.slogdet(blocks)
        scores[~valid | (signs <= 0)] = -np.inf
        index = int(np.argmax(scores))
        selected.append(index); gains.append(float(scores[index]))
        information += h[index].T @ h[index]
        valid[index] = False
    return dict(indices=selected, information_gain=gains, lens_controls=controls[selected],
                labels=np.asarray(candidates['labels'])[selected].tolist())


def simulate_setting_images(*, source_experiment, lens_controls, nominal_magnifications,
                            image_labels, output_directory, random_seed):
    """Ask the simulated instrument for independent images at recorded commands."""
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    record = (source_experiment.output_directory/'scoring_truth.json').read_text()
    truth = np.asarray(json.loads(record)['parameters'])
    spec = source_experiment.specimen_and_detector
    operator = make_magical_image_operator(spec, source_experiment.specimen_template)
    rng = np.random.default_rng(random_seed)
    images = []
    for commands, nominal in zip(lens_controls, nominal_magnifications):
        shape = np.asarray(predict_observables(truth, commands, spec))
        p = np.r_[shape, rng.normal(0,.3,2), np.log(4e7*(nominal/40000.)**2), np.log(3.)]
        mean = np.asarray(operator(p))
        images.append(rng.poisson(mean).astype(float)+rng.normal(0,spec['read_noise_std_e'],mean.shape))
    (directory/'scoring_truth.json').write_text(record)
    result = Experiment(np.asarray(images), np.asarray(lens_controls), np.asarray(nominal_magnifications),
                        list(image_labels), source_experiment.specimen_template, spec, directory)
    np.savez_compressed(directory/'images.npz', images=result.images, lens_controls=result.lens_controls,
                        nominal_magnifications=result.nominal_magnifications, image_labels=result.image_labels)
    return result


def acquire_measurements(*, source_experiment, selection, current_parameters,
                         output_directory, random_seed):
    """Predict first, acquire 9 images per state, then extract measurements.

    Each state has one ordinary image, four +/-3 mrad tilt images, and four
    +/-10 nm beam-shift images. The current fitted model supplies initial image
    guesses, never fixed image parameters; every frame is fitted independently.
    Its predictions are saved before the instrument's images are generated.
    """
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    controls = selection['lens_controls']
    spec = source_experiment.specimen_and_detector
    predictions = np.asarray(setting_observables(current_parameters, controls, spec))
    (directory/'design.json').write_text(json.dumps({
        'lens_control_columns':['Objective','IL1','IL2','IL3','PL1'],
        'lens_controls':np.asarray(controls).tolist(), 'labels':selection['labels'],
        'current_model':parameters_to_model(current_parameters), 'predicted_observables':predictions.tolist(),
        'random_seed':random_seed, 'images_per_setting':9,
    }, indent=2))
    nominal = np.exp(predictions[:,0])
    experiment = simulate_setting_images(
        source_experiment=source_experiment, lens_controls=controls, nominal_magnifications=nominal,
        image_labels=selection['labels'], output_directory=directory, random_seed=random_seed,
    )
    seeds = np.c_[predictions[:,:3], np.zeros((len(controls),2)), np.log(4e7*(nominal/40000.)**2), np.full(len(controls),np.log(3.))]
    images = measure_images(images=experiment.images, nominal_magnifications=nominal,
                            specimen_template=experiment.specimen_template, specimen_and_detector=spec,
                            initial_image_parameters=seeds)
    indices = list(range(len(controls)))
    tilt_images = simulate_tilt_images(experiment=experiment, lens_setting_indices=indices,
                                       tilt_mrad=3., random_seed=random_seed+1)
    tilts = measure_tilt_images(tilt_experiment=tilt_images, specimen_template=experiment.specimen_template,
                                specimen_and_detector=spec, reference_measurements=images)
    beam_images = simulate_beam_shift_images(experiment=experiment, lens_setting_indices=indices,
                                            shift_nm=10., random_seed=random_seed+2)
    shifts = measure_beam_shift_images(beam_shift_experiment=beam_images, specimen_template=experiment.specimen_template,
                                       specimen_and_detector=spec, reference_measurements=images)
    measured = CalibrationMeasurements(controls, images.values, images.covariance, tilts, shifts)
    measured.save(directory/'measurements.npz')
    np.savez_compressed(directory/'image_analysis.npz', image_parameters=images.image_parameters,
                        image_error_in_noise_units=images.image_error_in_noise_units)
    return measured


def save_fit(path, result):
    record=dict(parameters=result.x.tolist(), model=parameters_to_model(result.x),
        total_squared_residual=float(2*result.cost), data_squared_residual=float(result.data_residual @ result.data_residual),
        success=bool(result.success), message=str(result.message), evaluations=int(result.nfev),
        optimality=float(result.optimality))
    if hasattr(result,'displacement_initialization'):
        record['displacement_initialization']=result.displacement_initialization
    Path(path).write_text(json.dumps(record,indent=2))


def fit_displacements_then_all(*, problem, starting_parameters, ordinary_image_count,
                               max_evaluations=5000):
    """Fit 16 optical parameters from A/B responses, then jointly fit all 19.

    Signed detector displacements avoid the log-scale and angle-wrap barriers
    encountered by a poor initial image model. The first stage ignores every
    ordinary scale/rotation/blur observation and all soft priors. It keeps the
    same hard parameter bounds. The final stage restores the original complete
    objective, including its fixed priors, with every measurement used once.
    """
    skip=3*ordinary_image_count
    cache=getattr(problem,'_displacement_initialization_functions',{})
    if skip not in cache:
        residual=jax.jit(lambda u:problem.data_residual(problem.decode(u))[skip:])
        cache[skip]=(residual,jax.jit(jax.jacfwd(residual)))
        problem._displacement_initialization_functions=cache
    residual,derivative=cache[skip]
    u=np.asarray(problem.encode(starting_parameters)).copy()
    if len(residual(u))<16:
        raise ValueError('At least 16 calibrated displacement coordinates are required for optical initialization.')
    def expand(v):
        values=u.copy();values[:16]=v;return values
    initial=least_squares(lambda v:np.asarray(residual(expand(v))),u[:16],
        jac=lambda v:np.asarray(derivative(expand(v)))[:,:16],
        bounds=(problem.lower[:16],problem.upper[:16]),x_scale='jac',max_nfev=max_evaluations,
        ftol=1e-10,xtol=1e-11,gtol=1e-5)
    rms=float(np.sqrt(np.mean(initial.fun**2)))
    if not initial.success or rms>=1.5:
        raise RuntimeError(f'Displacement-only initialization did not find an acceptable fit: RMS {rms:.3g}.')
    optical_start=np.asarray(problem.decode(expand(initial.x)))
    result=problem.solve(optical_start,max_evaluations=max_evaluations)
    result.displacement_initialization=dict(success=bool(initial.success),rms=rms,evaluations=int(initial.nfev),
                                            fitted_optical_parameters=16,optical_parameters=optical_start[:16].tolist())
    return result


def score_parameters(*, parameters, truth_directory):
    """Truth is opened only here, after fitting, to assess parameter recovery."""
    true = np.asarray(json.loads((Path(truth_directory)/'scoring_truth.json').read_text())['parameters'])
    truth, fitted = parameters_to_model(true), parameters_to_model(parameters)
    def flatten(model):
        result = {}
        for key, value in model.items():
            if isinstance(value,dict):
                result.update({f'{key}.{name}':v for name,v in value.items()})
            else:
                result[key] = value
        return result
    a,b=flatten(truth),flatten(fitted)
    return {key:dict(truth=a[key],fitted=b[key],error=b[key]-a[key],relative_error=(b[key]-a[key])/abs(a[key])) for key in a}


@dataclass
class AdaptiveCalibration:
    """Named current model, measured data, fixed priors and simulated instrument."""
    model: dict
    measurements: CalibrationMeasurements
    source_experiment: Experiment
    prior_model: dict
    position_priors: dict
    output_directory: Path
    feedback: dict


def load_calibration_checkpoint(directory):
    """Continue from the completed MAGICAL/tilt/beam-shift notebook."""
    directory=Path(directory)
    spec=json.loads((directory/'acquisition.json').read_text())
    known={key:spec[key] for key in ['voltage_v','pixel_pitch_m','shape_yx','feature_sigma_m','read_noise_std_e','specimen']}
    source=Experiment(np.load(directory/'images.npz')['images'],np.asarray(spec['controls']),
                      np.asarray(spec['nominal_magnifications']),spec['labels'],
                      dict(np.load(directory/'specimen_template.npz')),known,directory)
    inputs=json.loads((directory/'beam_shift_landscape/fit_inputs.json').read_text())
    fits=json.loads((directory/'beam_shift_landscape/starts.json').read_text())
    best=min((r for r in fits if r['success']),key=lambda r:r['total_squared_residual'])
    return AdaptiveCalibration(best['model'],load_existing_measurements(directory),source,
                               inputs['prior_model'],inputs['position_priors'],directory,{})


def run_calibration_round(*, previous, settings_to_measure, vary_objective,
                          wider_compensation, output_directory, random_seed,
                          candidate_seed=772, reuse_completed=True, candidates=None,
                          initialize_from_displacements=False):
    """Choose settings -> save predictions -> make images -> measure -> refit.

    Supply the previous calibration, number/type of new settings and seeds.
    Return an AdaptiveCalibration with the fitted named model, accumulated
    observations and before/after prediction errors. Completed runs can be
    reopened; set reuse_completed=False to regenerate the experiment.
    """
    directory=Path(output_directory);directory.mkdir(parents=True,exist_ok=True)
    if reuse_completed and (directory/'feedback.json').exists():
        design=json.loads((directory/'design.json').read_text())
        if len(design['lens_controls']) != settings_to_measure or design['random_seed'] != random_seed:
            raise ValueError('Saved round uses different settings or seed; choose a new output directory or regenerate.')
        if not np.allclose(model_to_parameters(design['current_model']),model_to_parameters(previous.model),rtol=1e-6,atol=1e-6):
            raise ValueError('Saved round started from a different fitted microscope.')
        best=json.loads((directory/'best_fit.json').read_text())
        feedback=json.loads((directory/'feedback.json').read_text())
        if not best['success'] or feedback['accumulated_fit_rms']>=1.5:
            raise RuntimeError('The saved round does not contain an acceptable microscope fit.')
        return AdaptiveCalibration(best['model'],CalibrationMeasurements.load(directory/'combined_measurements.npz'),
                                   previous.source_experiment,previous.prior_model,previous.position_priors,directory,
                                   feedback)
    (directory/'feedback.json').unlink(missing_ok=True)
    current=model_to_parameters(previous.model)
    spec=previous.source_experiment.specimen_and_detector
    problem=prepare_fit(measurements=previous.measurements,specimen_and_detector=spec,
                        prior_model=previous.prior_model,position_priors=previous.position_priors)
    if candidates is None:
        candidates=make_candidate_settings(current_parameters=current,reference_controls=previous.source_experiment.lens_controls[0],
            specimen_and_detector=spec,random_seed=candidate_seed,wider_compensation=wider_compensation,
            random_projector_settings=500 if wider_compensation else 1000)
    np.savez_compressed(directory/'candidates.npz',**candidates)
    allowed=np.ones(len(candidates['lens_controls']),bool) if wider_compensation else (
        (candidates['lens_controls'][:,0]!=1) if vary_objective else (candidates['lens_controls'][:,0]==1))
    selection=select_informative_settings(candidates=candidates,current_parameters=current,current_problem=problem,
                                         specimen_and_detector=spec,number_of_settings=settings_to_measure,allowed=allowed)
    (directory/'selection.json').write_text(json.dumps({k:v.tolist() if isinstance(v,np.ndarray) else v
                                                       for k,v in selection.items()},indent=2))
    added=acquire_measurements(source_experiment=previous.source_experiment,selection=selection,current_parameters=current,
                              output_directory=directory,random_seed=random_seed)
    new_problem=prepare_fit(measurements=added,specimen_and_detector=spec,
                            prior_model=previous.prior_model,position_priors=previous.position_priors)
    before=np.asarray(new_problem.data_residual(current))
    combined=previous.measurements.append(added);combined.save(directory/'combined_measurements.npz')
    problem=prepare_fit(measurements=combined,specimen_and_detector=spec,
                        prior_model=previous.prior_model,position_priors=previous.position_priors)
    fits=[]
    for i,start in enumerate([current,model_to_parameters(previous.prior_model)]):
        if initialize_from_displacements:
            result=fit_displacements_then_all(problem=problem,starting_parameters=start,
                ordinary_image_count=len(combined.controls),max_evaluations=12000)
        else:
            result=problem.solve(start,max_evaluations=12000)
        save_fit(directory/f'fit_{i}.json',result)
        if result.success and np.sqrt(np.mean(result.data_residual**2))<1.5:
            fits.append(result)
    if not fits:
        raise RuntimeError('No acceptable microscope fit. Convergence alone is insufficient; inspect residuals and initialization.')
    best=min(fits,key=lambda r:r.cost);save_fit(directory/'best_fit.json',best)
    after=np.asarray(new_problem.data_residual(best.x))
    score=score_parameters(parameters=best.x,truth_directory=previous.source_experiment.output_directory)
    (directory/'parameter_errors.json').write_text(json.dumps(score,indent=2))
    feedback=dict(before_prediction_rms=float(np.sqrt(np.mean(before**2))),
                  after_refit_rms=float(np.sqrt(np.mean(after**2))),
                  accumulated_fit_rms=float(np.sqrt(np.mean(best.data_residual**2))),
                  image_count=len(combined.controls)+2*len(combined.tilts['lens_controls'])+2*len(combined.shifts['lens_controls']),
                  maximum_relative_parameter_error=max(abs(v['relative_error']) for v in score.values()))
    (directory/'feedback.json').write_text(json.dumps(feedback,indent=2))
    return AdaptiveCalibration(parameters_to_model(best.x),combined,previous.source_experiment,
                               previous.prior_model,previous.position_priors,directory,feedback)


def check_new_predictions(*, calibration, candidates, settings_to_measure,
                          output_directory, random_seed, excluded_indices=(), reuse_completed=True):
    """Acquire independent images at new settings and score WITHOUT refitting."""
    directory=Path(output_directory);directory.mkdir(parents=True,exist_ok=True)
    if reuse_completed and (directory/'prediction_check.json').exists():
        design=json.loads((directory/'design.json').read_text())
        if len(design['lens_controls']) != settings_to_measure or design['random_seed'] != random_seed:
            raise ValueError('Saved prediction check uses different settings or seed.')
        if not np.allclose(model_to_parameters(design['current_model']),model_to_parameters(calibration.model),rtol=1e-6,atol=1e-6):
            raise ValueError('Saved predictions came from a different fitted microscope.')
        return json.loads((directory/'prediction_check.json').read_text())
    (directory/'prediction_check.json').unlink(missing_ok=True)
    p=model_to_parameters(calibration.model);spec=calibration.source_experiment.specimen_and_detector
    problem=prepare_fit(measurements=calibration.measurements,specimen_and_detector=spec,
                        prior_model=calibration.prior_model,position_priors=calibration.position_priors)
    allowed=np.ones(len(candidates['lens_controls']),bool);allowed[list(excluded_indices)]=False
    # Exclude any previously measured control vector, even if it appears under
    # another candidate index or another experiment label.
    for i,c in enumerate(candidates['lens_controls']):
        if np.any(np.all(np.isclose(calibration.measurements.controls,c,rtol=1e-10,atol=1e-12),axis=1)):
            allowed[i]=False
    selection=select_informative_settings(candidates=candidates,current_parameters=p,current_problem=problem,
        specimen_and_detector=spec,number_of_settings=settings_to_measure,allowed=allowed)
    (directory/'selection.json').write_text(json.dumps({k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in selection.items()},indent=2))
    measured=acquire_measurements(source_experiment=calibration.source_experiment,selection=selection,current_parameters=p,
                                  output_directory=directory,random_seed=random_seed)
    new_problem=prepare_fit(measurements=measured,specimen_and_detector=spec,
                            prior_model=calibration.prior_model,position_priors=calibration.position_priors)
    residual=np.asarray(new_problem.data_residual(p))
    report=dict(prediction_rms=float(np.sqrt(np.mean(residual**2))),image_count=9*settings_to_measure,
                measurement_count=len(residual),refitted=False,maximum_absolute_residual=float(np.max(abs(residual))))
    (directory/'prediction_check.json').write_text(json.dumps(report,indent=2))
    return report


def design_lens_settings(*, calibration, wider_compensation, output_path,
                        random_seed, reuse_completed=True):
    """Supply the current named model; get a bank of visible candidate settings."""
    path=Path(output_path)
    if reuse_completed and path.exists():
        candidates=dict(np.load(path))
        if 'design_model_parameters' in candidates:
            if not np.allclose(candidates['design_model_parameters'],model_to_parameters(calibration.model),rtol=1e-6,atol=1e-6):
                raise ValueError('The saved candidate bank was designed for a different fitted model.')
            if candidates['design_random_seed'] != random_seed or candidates['design_wider_compensation'] != wider_compensation:
                raise ValueError('The saved candidate bank uses different design settings.')
        return candidates
    candidates=make_candidate_settings(current_parameters=model_to_parameters(calibration.model),
        reference_controls=calibration.source_experiment.lens_controls[0],
        specimen_and_detector=calibration.source_experiment.specimen_and_detector,
        random_seed=random_seed,wider_compensation=wider_compensation,
        random_projector_settings=500 if wider_compensation else 1000)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**candidates,design_model_parameters=model_to_parameters(calibration.model),
                        design_random_seed=random_seed,design_wider_compensation=wider_compensation)
    return candidates


def repeat_noise_protocol(*, calibration, calibration_rounds, output_directory,
                          random_seed, reuse_completed=True, initialize_from_displacements=True):
    """Repeat every exposure of the final protocol with independent pixel noise.

    Settings, doses, specimen, and priors stay fixed. Every ordinary, tilt and
    beam-shift image is regenerated; no original noisy measurements are reused.
    The two optical starts are the literal prior and the earlier fitted model.
    """
    directory=Path(output_directory);directory.mkdir(parents=True,exist_ok=True)
    if reuse_completed and (directory/'parameter_errors.json').exists():
        return json.loads((directory/'parameter_errors.json').read_text())
    (directory/'parameter_errors.json').unlink(missing_ok=True)
    source=calibration.source_experiment;spec=source.specimen_and_detector
    metadata=[(five_controls(source.lens_controls),source.nominal_magnifications,source.image_labels)]
    extra=dict(np.load(source.output_directory/'extra_wobbles/images.npz'))
    metadata.append((five_controls(extra['lens_controls']),extra['nominal_magnifications'],extra['image_labels'].tolist()))
    for state in calibration_rounds:
        images=dict(np.load(state.output_directory/'images.npz'))
        metadata.append((five_controls(images['lens_controls']),images['nominal_magnifications'],images['image_labels'].tolist()))
    controls=np.concatenate([m[0] for m in metadata]);nominal=np.concatenate([m[1] for m in metadata])
    labels=sum((list(m[2]) for m in metadata),[])
    if not np.array_equal(controls,calibration.measurements.controls):
        raise ValueError('Supply the acquisition rounds in the same order as the final calibration.')
    experiment=simulate_setting_images(source_experiment=source,lens_controls=controls,nominal_magnifications=nominal,
        image_labels=labels,output_directory=directory,random_seed=random_seed)
    current=model_to_parameters(calibration.model)
    predictions=np.asarray(setting_observables(current,controls,spec))
    seeds=np.c_[predictions[:,:3],np.zeros((len(controls),2)),np.log(4e7*(nominal/40000.)**2),np.full(len(controls),np.log(3.))]
    images=measure_images(images=experiment.images,nominal_magnifications=nominal,specimen_template=source.specimen_template,
                          specimen_and_detector=spec,initial_image_parameters=seeds)
    def groups(pairs,vector_key,factor):
        groups={}
        for commands,vector in zip(pairs['lens_controls'],pairs[vector_key]):
            amplitude=round(float(np.linalg.norm(vector)*factor),10)
            matches=np.flatnonzero(np.all(np.isclose(controls,commands,rtol=1e-12,atol=1e-14),axis=1))
            if not len(matches):raise ValueError('A pair refers to an unknown lens setting.')
            group=groups.setdefault(amplitude,[])
            if int(matches[0]) not in group:group.append(int(matches[0]))
        return groups
    tilts=[]
    for i,(amplitude,indices) in enumerate(groups(calibration.measurements.tilts,'angle_changes_rad',500.).items()):
        frames=simulate_tilt_images(experiment=experiment,lens_setting_indices=indices,tilt_mrad=amplitude,
                                   random_seed=random_seed+i+1)
        tilts.append(measure_tilt_images(tilt_experiment=frames,specimen_template=source.specimen_template,
                                        specimen_and_detector=spec,reference_measurements=images))
    if np.any(calibration.measurements.shifts['angle_changes_rad'] != 0):
        raise ValueError('This repeat helper currently supports pure calibrated beam shifts.')
    shifts=[]
    for i,(amplitude,indices) in enumerate(groups(calibration.measurements.shifts,'position_changes_m',.5e9).items()):
        frames=simulate_beam_shift_images(experiment=experiment,lens_setting_indices=indices,shift_nm=amplitude,
                                         random_seed=random_seed+len(tilts)+i+1)
        shifts.append(measure_beam_shift_images(beam_shift_experiment=frames,specimen_template=source.specimen_template,
                                               specimen_and_detector=spec,reference_measurements=images))
    measurements=CalibrationMeasurements(controls,images.values,images.covariance,
                                         combine_tilt_measurements(*tilts),combine_beam_shift_measurements(*shifts))
    measurements.save(directory/'measurements.npz')
    problem=prepare_fit(measurements=measurements,specimen_and_detector=spec,
                        prior_model=calibration.prior_model,position_priors=calibration.position_priors)
    fits=[]
    for i,start in enumerate([model_to_parameters(calibration.prior_model),current]):
        if initialize_from_displacements:
            result=fit_displacements_then_all(problem=problem,starting_parameters=start,
                ordinary_image_count=len(measurements.controls),max_evaluations=12000)
        else:
            result=problem.solve(start,max_evaluations=12000)
        save_fit(directory/f'fit_{i}.json',result)
        if result.success and np.sqrt(np.mean(result.data_residual**2))<1.5:fits.append(result)
    if not fits:raise RuntimeError('Neither starting model produced an acceptable fit for this noise repeat.')
    best=min(fits,key=lambda r:r.cost);save_fit(directory/'best_fit.json',best)
    errors=score_parameters(parameters=best.x,truth_directory=source.output_directory)
    (directory/'parameter_errors.json').write_text(json.dumps(errors,indent=2))
    (directory/'repeat_protocol.json').write_text(json.dumps(dict(random_seed=random_seed,
        image_count=len(controls)+2*len(measurements.tilts['lens_controls'])+2*len(measurements.shifts['lens_controls']),
        all_images_regenerated=True),indent=2))
    return errors


def run_recovery_checks(*, baseline, calibration_rounds, output_directory, reuse_completed=True):
    """Compare information, priors, detector profiles and two fitting orders."""
    from types import SimpleNamespace
    from projector_landscape import explore_fit_landscape
    folder=Path(output_directory);folder.mkdir(parents=True,exist_ok=True)
    final=calibration_rounds[-1]
    spec=final.source_experiment.specimen_and_detector
    info_path=folder/'identifiability_checks.json'
    if not (reuse_completed and info_path.exists()):
        records={}
        names=['baseline','projector_combinations','objective_combinations','wider_combinations']
        for name,state in zip(names,[baseline,*calibration_rounds]):
            problem=prepare_fit(measurements=state.measurements,specimen_and_detector=spec,
                                prior_model=state.prior_model,position_priors=state.position_priors)
            p=model_to_parameters(state.model);u=problem.encode(p)
            residual=jax.jit(lambda v:problem.data_residual(problem.decode(v)))
            derivative=jax.jit(jax.jacfwd(residual))
            jacobian=np.asarray(derivative(u));_,singular,vt=np.linalg.svd(jacobian,full_matrices=False)
            records[name]=dict(singular_values=singular.tolist(),condition_number=float(singular[0]/singular[-1]))
            if state is final:
                result=least_squares(lambda v:np.asarray(residual(v)),np.asarray(u),
                    jac=lambda v:np.asarray(derivative(v)),bounds=(problem.lower,problem.upper),
                    x_scale='jac',max_nfev=5000,ftol=1e-10,xtol=1e-11,gtol=1e-5)
                if not result.success:raise RuntimeError('The fit without soft priors did not converge.')
                params=np.asarray(problem.decode(result.x));model=parameters_to_model(params)
                errors=score_parameters(parameters=params,truth_directory=final.source_experiment.output_directory)
                records['without_soft_priors']=dict(success=bool(result.success),parameters=params.tolist(),model=model,
                    data_rms=float(np.sqrt(np.mean(result.fun**2))),
                    maximum_relative_parameter_error=max(abs(v['relative_error']) for v in errors.values()),
                    detector_change_mm=model['detector_position_mm']-state.model['detector_position_mm'])
                covariance=(vt.T/singular)@(vt/singular[:,None])
                np.savez_compressed(folder/'final_local_information.npz',singular_values=singular,vectors=vt,
                                    covariance_solver_coordinates=covariance)
        info_path.write_text(json.dumps(records,indent=2))
    landscape=folder/'final_landscape'
    if not (reuse_completed and (landscape/'detector_profile.json').exists()):
        inputs=json.loads((baseline.source_experiment.output_directory/'beam_shift_landscape/fit_inputs.json').read_text())
        z=final.model['detector_position_mm']
        grid=np.unique(np.r_[640.,680.,700.,np.linspace(z-1.,z+1.,9),740.,780.,840.])
        m=final.measurements
        explore_fit_landscape(lens_controls=m.controls,image_measurements=SimpleNamespace(values=m.values,covariance=m.covariance),
            tilt_measurements=m.tilts,beam_shift_measurements=m.shifts,specimen_and_detector=spec,
            prior_model=final.prior_model,position_priors=final.position_priors,starting_models=inputs['starting_models'],
            detector_positions_mm=grid,output_directory=landscape,max_evaluations=5000)
    staged=folder/'ab_initialised_starts';staged.mkdir(exist_ok=True)
    if not (reuse_completed and (staged/'summary.json').exists()):
        inputs=json.loads((baseline.source_experiment.output_directory/'beam_shift_landscape/fit_inputs.json').read_text())
        problem=prepare_fit(measurements=final.measurements,specimen_and_detector=spec,
                            prior_model=final.prior_model,position_priors=final.position_priors)
        summaries=[]
        for i,model in enumerate(inputs['starting_models']):
            try:
                result=fit_displacements_then_all(problem=problem,starting_parameters=model_to_parameters(model),
                    ordinary_image_count=len(final.measurements.controls),max_evaluations=5000)
            except RuntimeError as error:
                summaries.append(dict(index=i,ab_success=False,message=str(error)))
            else:
                save_fit(staged/f'fit_{i}.json',result)
                initial=result.displacement_initialization
                if i==0 and not (folder/'ab_only_geometry.json').exists():
                    p=np.r_[initial['optical_parameters'],model_to_parameters(model)[16:]]
                    errors=score_parameters(parameters=p,truth_directory=final.source_experiment.output_directory)
                    optical={k:v for k,v in errors.items() if k not in
                             ['beam_curvature_per_m','angular_spread_mrad','sample_rotation_deg']}
                    (folder/'ab_only_geometry.json').write_text(json.dumps([dict(
                        start='literal_guess',success=True,data_rms=initial['rms'],optical_parameter_errors=optical,
                        maximum_relative_optical_error=max(abs(v['relative_error']) for v in optical.values()))],indent=2))
                summaries.append(dict(index=i,ab_success=initial['success'],ab_rms=initial['rms'],
                    ab_evaluations=initial['evaluations'],joint_success=bool(result.success),
                    joint_rms=float(np.sqrt(np.mean(result.data_residual**2))),joint_cost=float(2*result.cost),
                    joint_evaluations=int(result.nfev)))
            (staged/'summary.json').write_text(json.dumps(summaries,indent=2))
    return json.loads(info_path.read_text())
