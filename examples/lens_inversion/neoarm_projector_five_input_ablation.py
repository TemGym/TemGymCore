"""Five-input ablation of the existing two-projector Gaussian image experiment.

Keep reference,+/-x position,+/-x tilt images and both measured detector
coordinates. The source shape is unknown: per-image shape/dose nuisances are
profiled by retaining the full 2x2 centroid marginal covariance. One common
alignment offset is analytically profiled for each optical acquisition.

For round optics H_theta_x=B R e_x and H_x=R(A e_x+kappa B e_y).
These two measured vectors identify the round response family across optical
settings, but do NOT observe a generic unconstrained 2x4 response matrix.
No omitted y-input response is filled from generating truth.
"""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
import numpy as np

from neoarm_coupled_projector_fit import CoupledProjectorResponse, stage_coordinates, block_diagonal
from neoarm_last_lens_physical_fit import ROOT, RoundLensResponse
from temgym_core.jax_least_squares import least_squares


FIVE_FRAMES=(0,1,2,5,6)
Y_FRAMES=(3,4,7,8)
PHYSICAL_NAMES=('PL1.log_length_m','PL1.log_gap_m','PL1.q_bias','PL1.q_slope',
                'IL3.log_length_m','IL3.log_gap_m','IL3.q_bias','IL3.q_slope')


class CentroidProfileLikelihood:
    """Use selected original images, never supplemented by model-filled data."""

    def __init__(self,model,controls,image_parameters,image_covariances,input_states,frame_indices):
        indices=np.asarray(frame_indices,dtype=int)
        if indices.ndim!=1 or len(indices)<3 or len(np.unique(indices))!=len(indices) or np.any((indices<0)|(indices>=9)):
            raise ValueError('Select distinct recorded frame indices in0..8.')
        # Select BEFORE finiteness checks: omitted y images cannot affect an
        # x-only fit, even if poisoned or absent from the admitted experiment.
        observed=np.asarray(image_parameters)[:,indices,:2]
        cov=np.asarray(image_covariances)[:,indices,:2,:2]
        controls=np.asarray(controls)
        input_states=np.asarray(input_states)
        if input_states.shape!=(9,4):
            raise ValueError('Recorded physical input states must have shape (9,4).')
        inputs=input_states[indices]
        if observed.shape!=(len(controls),len(indices),2) or cov.shape!=observed.shape+(2,) or controls.shape!=(len(observed),2):
            raise ValueError('Centroid summaries, controls and covariance shapes disagree.')
        if not all(np.all(np.isfinite(v)) for v in (observed,cov,controls,inputs)):
            raise ValueError('Selected observations/calibrations must be finite.')
        if not np.allclose(cov,cov.swapaxes(-1,-2),rtol=1e-11,atol=1e-18):
            raise ValueError('Centroid covariance must be symmetric.')
        self.model=model
        self.frame_indices=tuple(int(i) for i in indices)
        self.controls,self.observed=map(jnp.asarray,(controls,observed))
        self.inputs=jnp.asarray(inputs/np.asarray(model.calibration.input_steps_grouped))
        self.cholesky=jnp.asarray(np.linalg.cholesky(cov))
        design=np.broadcast_to(np.eye(2),cov.shape)
        whitened=np.linalg.solve(np.asarray(self.cholesky),design).reshape(len(controls),2*len(indices),2)
        self.alignment_q=jnp.asarray(np.linalg.qr(whitened,mode='reduced')[0])
        self.residual=jax.jit(self._residual)
        self.jacobian=jax.jit(jax.jacfwd(self._residual))

    @property
    def independent_coordinate_count(self):
        return len(self.controls)*(2*len(self.frame_indices)-2)

    def prediction(self,parameters):
        response=self.model.predict(parameters,self.controls)
        return jnp.einsum('sij,fj->sfi',response,self.inputs)

    def _residual(self,parameters):
        delta=self.prediction(parameters)-self.observed
        white=jax.vmap(jax.vmap(lambda c,d:solve_triangular(c,d,lower=True)))(self.cholesky,delta).reshape(len(self.controls),-1)
        return (white-jnp.einsum('sij,skj,sk->si',self.alignment_q,self.alignment_q,white)).reshape(-1)


def fit_candidate(problem,candidate,*,max_nfev=1000):
    initial=np.asarray(candidate['parameters'])
    lower,upper=np.asarray(candidate['bounds'])
    norms=np.maximum(np.linalg.norm(np.asarray(problem.jacobian(initial)),axis=0),np.finfo(float).tiny)
    fit=least_squares(problem.residual,initial,bounds=(lower,upper),x_scale=np.minimum(1/norms,1.),
        max_nfev=max_nfev,max_iter=1600,ftol=1e-11,xtol=0.,gtol=1e-7)
    p=np.asarray(fit.x);jac=np.asarray(fit.jacobian);norms=np.maximum(np.linalg.norm(jac,axis=0),np.finfo(float).tiny)
    _,singular,right=np.linalg.svd(jac/norms,full_matrices=False)
    rank=int(np.sum(singular>singular[0]*1e-10))
    gradient=jac.T@np.asarray(fit.residual)
    active=((p<=lower)&(gradient>0))|((p>=upper)&(gradient<0))
    result=dict(initial=initial.tolist(),parameters=p.tolist(),bounds=[lower.tolist(),upper.tolist()],
        optimizer_success=bool(fit.success),termination=fit.status,nfev=fit.nfev,chi_square=float(2*fit.cost),
        physical_coordinates=np.asarray(stage_coordinates(p)).tolist(),numerical_rank=rank,
        normalized_jacobian_singular_values=singular.tolist(),all_thirteen_coordinates_free=True,
        final_projected_column_normalized_gradient=float(np.max(np.abs(np.where(active,0.,gradient/norms)))),
        active_bounds=np.flatnonzero(np.minimum(abs(p-lower),abs(p-upper))<1e-7).tolist(),
        status='tested_local_ablation_fit' if fit.success else 'unresolved_ablation_optimizer')
    if rank==13:
        factor=(right.T/singular)/norms[:,None]
        physical=np.asarray(jax.jacfwd(stage_coordinates)(jnp.asarray(p)))@factor
        result.update(parameter_covariance=(factor@factor.T).tolist(),physical_covariance=(physical@physical.T).tolist(),
                      physical_standard_errors=np.linalg.norm(physical,axis=1).tolist())
    else:
        result.update(parameter_covariance=None,physical_covariance=None,physical_standard_errors=None)
    return result


def validate_profile(problem,fit):
    p=jnp.asarray(fit['parameters'])
    residual=np.asarray(problem.residual(p))
    result=dict(profile_chi_square=float(residual@residual),
        independent_coordinate_count=problem.independent_coordinate_count,
        rms_profile_sigma=float(np.sqrt((residual@residual)/problem.independent_coordinate_count)),
        optics_refitted=False,alignment_profiled=True)
    if fit['parameter_covariance'] is not None:
        # Residual projection removes exactly two nuisance coordinates per
        # state. In that whitened subspace I+J C J.T yields the same quadratic
        # form; null projection directions carry zero residual and derivative.
        derivative=np.asarray(problem.jacobian(p))
        covariance=np.eye(len(residual))+derivative@np.asarray(fit['parameter_covariance'])@derivative.T
        w=np.linalg.solve(np.linalg.cholesky(covariance),residual)
        result['predictive_chi_square']=float(w@w)
    return result


def select_y_checks(controls,training_count,old_hold_count,paired_pl1,fresh_train_il3,fresh_hold_il3):
    """Six fixed control/role targets, chosen before any ablation fit."""
    controls=np.asarray(controls)
    low,high=sorted(np.asarray(paired_pl1))
    targets=[(low,0.),(high,0.),(low,float(np.min(fresh_train_il3))),
             (high,float(np.max(fresh_train_il3))),
             (low,float(np.min(fresh_hold_il3))),(high,float(np.max(fresh_hold_il3)))]
    indices=[]
    for i,target in enumerate(targets):
        candidates=np.flatnonzero(np.all(np.isclose(controls,target,rtol=0,atol=1e-10),axis=1))
        candidates=candidates[candidates<training_count] if i<4 else candidates[candidates>=training_count+old_hold_count]
        if not len(candidates):raise ValueError('A predeclared y-check control/role is missing.')
        indices.append(int(candidates[0]))
    if len(set(indices))!=6:raise ValueError('Y-check acquisitions must be distinct.')
    return indices,targets


def y_check_responses(image_parameters,image_covariances):
    """Four omitted images give y-position/y-tilt vectors with full covariance."""
    means=np.asarray(image_parameters)[:,:,:2]
    cov=np.asarray(image_covariances)[:,:,:2,:2]
    transform=np.zeros((4,18))
    for axis in range(2):
        transform[axis*2,3*2+axis]=.5;transform[axis*2,4*2+axis]=-.5
        transform[axis*2+1,7*2+axis]=.5;transform[axis*2+1,8*2+axis]=-.5
    response=np.einsum('ij,sj->si',transform,means.reshape(len(means),18)).reshape(-1,2,2)
    full=np.zeros((len(means),18,18))
    for frame in range(9):full[:,2*frame:2*frame+2,2*frame:2*frame+2]=cov[:,frame]
    response_cov=transform@full@transform.T
    return response,response_cov


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_evidence(recovery_path):
    recovery_path=Path(recovery_path).resolve();recovery=json.loads(recovery_path.read_text())
    hashes=dict(recovery['source_hashes'])
    for path,checksum in hashes.items():
        if digest(ROOT/path)!=checksum:raise ValueError(f'Stale ablation dependency:{path}')
    folder=recovery_path.parent
    manifests=[folder/'last_lens_redesigned_inference.json',folder/'pl1_il3_inference.json',ROOT/recovery['fresh_inference_report']]
    blocks=[];calibrations=[]
    for path in manifests:
        report=json.loads(path.read_text())
        calibrations.append((report['input_steps_grouped'],report['detector_pixel_m']))
        for name,checksum in report['source_hashes'].items():
            if digest(ROOT/name)!=checksum:raise ValueError(f'Stale image input:{name}')
            hashes[name]=checksum
        arrays_path=ROOT/report['arrays_path']
        if digest(arrays_path)!=report['arrays_sha256']:raise ValueError('Image-summary NPZ checksum mismatch.')
        hashes[str(path.relative_to(ROOT))]=digest(path);hashes[str(arrays_path.relative_to(ROOT))]=digest(arrays_path)
        with np.load(arrays_path,allow_pickle=False) as arrays:
            inputs=arrays['input_states_grouped'].copy()
            if len(blocks)==0:
                t=arrays['accepted_normalized_dac'];h=arrays['heldout_normalized_dac']
                block=dict(train_controls=np.column_stack((t,np.zeros(len(t)))),hold_controls=np.column_stack((h,np.zeros(len(h)))),
                    train_y=arrays['image_parameters'].copy(),train_cov=arrays['image_parameter_covariances'].copy(),
                    hold_y=arrays['heldout_image_parameters'].copy(),hold_cov=arrays['heldout_image_parameter_covariances'].copy())
            else:
                train=arrays['training_mask'];hold=arrays['heldout_mask'];pl1=arrays['pl1_controls'];il3=arrays['il3_controls']
                if train.dtype!=bool or hold.dtype!=bool or not np.all(train|hold) or np.any(train&hold):
                    raise ValueError('Acquisition splits must be boolean, disjoint and exhaustive.')
                controls=lambda mask:np.array([[p,u] for u in il3[mask] for p in pl1])
                block=dict(train_controls=controls(train),hold_controls=controls(hold),
                    train_y=arrays['image_parameters'][train].reshape(-1,9,6),train_cov=arrays['image_parameter_covariances'][train].reshape(-1,9,6,6),
                    hold_y=arrays['image_parameters'][hold].reshape(-1,9,6),hold_cov=arrays['image_parameter_covariances'][hold].reshape(-1,9,6,6),
                    paired_pl1=pl1.copy(),train_il3=il3[train].copy(),hold_il3=il3[hold].copy())
            block['inputs']=inputs;blocks.append(block)
    if not all(np.array_equal(b['inputs'],blocks[0]['inputs']) for b in blocks):raise ValueError('Image input-state calibrations differ.')
    if not all(c==calibrations[0] for c in calibrations):raise ValueError('Input-step or detector-pitch calibrations differ.')
    combine=lambda key:np.concatenate([b[key] for b in blocks])
    training=dict(controls=combine('train_controls'),y=combine('train_y'),cov=combine('train_cov'))
    heldout=dict(controls=combine('hold_controls'),y=combine('hold_y'),cov=combine('hold_cov'))
    if not np.array_equal(training['controls'],recovery['training_controls']):raise ValueError('Training controls disagree with frozen recovery.')
    old_hold_count=len(blocks[0]['hold_controls'])+len(blocks[1]['hold_controls'])
    expected_hold=np.concatenate((recovery['reused_old_heldout']['controls'],recovery['fresh_heldout']['controls']))
    if not np.array_equal(heldout['controls'],expected_hold):raise ValueError('Heldout controls disagree with frozen splits.')
    hashes[str(recovery_path.relative_to(ROOT))]=digest(recovery_path);hashes[str(Path(__file__).relative_to(ROOT))]=digest(Path(__file__))
    return recovery,blocks,training,heldout,old_hold_count,hashes


def run_audit(recovery_path,output,*,max_nfev=1000):
    if jax.default_backend()!='gpu' or not jax.config.x64_enabled:raise RuntimeError('Use GPU float64 for the ablation forward/Jacobian fit.')
    started=perf_counter()
    recovery,blocks,training,heldout,old_hold_count,hashes=load_evidence(recovery_path)
    first=json.loads((ROOT/recovery['old_training_report']).read_text())
    model=CoupledProjectorResponse(RoundLensResponse(tuple(first['calibrated_input_steps_grouped']),first['detector_pixel_m']))
    inputs=blocks[0]['inputs']
    all_controls=np.concatenate((training['controls'],heldout['controls']))
    y_indices,y_targets=select_y_checks(all_controls,len(training['controls']),old_hold_count,
        blocks[2]['paired_pl1'],blocks[2]['train_il3'],blocks[2]['hold_il3'])
    eligible=[r for r in recovery['rows'] if r['optimizer_success'] and not r['active_bounds']]
    best=min(eligible,key=lambda r:r['chi_square'])
    rivals=[r for r in eligible if abs(r['parameters'][6]-best['parameters'][6])>.3]
    rival=min(rivals,key=lambda r:r['chi_square'])
    results=[]
    for label,frames in [('five_inputs',FIVE_FRAMES),('nine_inputs_matched_centroid_profile',tuple(range(9)))]:
        problem=CentroidProfileLikelihood(model,training['controls'],training['y'],training['cov'],inputs,frames)
        fits=[]
        for seed_name,seed in [('previous_best',best),('previous_tested_rival',rival)]:
            fit=fit_candidate(problem,seed,max_nfev=max_nfev);fit['seed_label']=seed_name;fits.append(fit)
            print(label,seed_name,fit['optimizer_success'],fit['chi_square'],fit['termination'],flush=True)
        converged=[f for f in fits if f['optimizer_success']]
        winner=min(converged,key=lambda f:f['chi_square']) if converged else None
        distinct=[f for f in converged if winner is not None and abs(f['parameters'][6]-winner['parameters'][6])>.3]
        results.append(dict(label=label,frame_indices=list(frames),training_frame_count=len(training['controls'])*len(frames),
            total_frame_count=len(all_controls)*len(frames),training_independent_coordinates=problem.independent_coordinate_count,
            fits=fits,best=winner,tested_rival_delta_chi_square=min(f['chi_square'] for f in distinct)-winner['chi_square'] if distinct else None))
    # Training winners are fixed before evaluating frozen heldout observations.
    for result in results:
        for fit in result['fits']:
            for name,sl in [('reused_old_heldout',slice(0,old_hold_count)),('frozen_fresh_heldout',slice(old_hold_count,None))]:
                problem=CentroidProfileLikelihood(model,heldout['controls'][sl],heldout['y'][sl],heldout['cov'][sl],inputs,result['frame_indices'])
                fit[name]=validate_profile(problem,fit)
                fit[name].update(used_for_ablation_training_or_winner_selection=False,
                    previously_used_by_full_nine_input_analysis=True,virgin_confirmatory_data=False)
    five=results[0]['best'];nine=results[1]['best'];comparison={};y_check={}
    if five is not None and nine is not None and five['parameter_covariance'] is not None and nine['parameter_covariance'] is not None:
        indices=np.array([0,1,4,5])
        s5=np.asarray(five['physical_standard_errors']);s9=np.asarray(nine['physical_standard_errors'])
        comparison=dict(length_gap_order=['PL1.length','PL1.gap','IL3.length','IL3.gap'],
            five_input_local_standard_errors_percent=(100*s5[indices]).tolist(),
            matched_nine_input_local_standard_errors_percent=(100*s9[indices]).tolist(),
            five_over_nine_standard_error_ratio=(s5[indices]/s9[indices]).tolist(),
            saved_nine_input_signed_response_standard_errors_percent=(100*np.asarray(recovery['information']['physical_standard_errors'])[indices]).tolist(),
            saved_nine_input_signed_response_rival_delta_chi_square=rival['chi_square']-best['chi_square'])
        all_y=np.concatenate((training['y'],heldout['y']))[y_indices]
        all_cov=np.concatenate((training['cov'],heldout['cov']))[y_indices]
        response,cov=y_check_responses(all_y,all_cov)
        controls=all_controls[y_indices]
        predict=lambda p:model.predict(p,jnp.asarray(controls))[:,:,[1,3]]
        predicted=np.asarray(predict(jnp.asarray(five['parameters'])))
        delta=(predicted-response).reshape(-1);covariance=block_diagonal(list(cov))
        white=np.linalg.solve(np.linalg.cholesky(covariance),delta)
        jac=np.asarray(jax.jacfwd(lambda p:predict(p).reshape(-1))(jnp.asarray(five['parameters'])))
        predictive=covariance+jac@np.asarray(five['parameter_covariance'])@jac.T
        w=np.linalg.solve(np.linalg.cholesky(predictive),delta)
        y_check=dict(predeclared_all_acquisition_indices=y_indices,predeclared_control_targets=y_targets,
            extra_frame_indices=list(Y_FRAMES),extra_frame_count=24,proposed_total_frames=219,
            observed_response=response.tolist(),response_covariance=covariance.tolist(),predicted_response=predicted.tolist(),
            response_coordinate_count=24,chi_square=float(white@white),predictive_chi_square=float(w@w),
            maximum_absolute_response_error_pixels=float(np.max(np.abs(delta))),
            used_in_five_input_likelihood=False,previously_used_in_nine_input_analyses=True,
            independent_new_instrument_validation_admitted=False,
            interpretation='Tests round/y-calibration predictions on omitted-from-loss images; not fresh confirmatory data after historical nine-input analysis.')
    if any(digest(ROOT/name)!=value for name,value in hashes.items()):raise RuntimeError('A frozen ablation dependency changed during execution.')
    report=dict(schema='neoarm-five-input-projector-ablation-v1',source_hashes=hashes,source_hashes_unchanged_during_execution=True,
        backend=jax.default_backend(),jax_version=jax.__version__,x64_enabled=True,elapsed_seconds=perf_counter()-started,
        source_recovery_report=str(Path(recovery_path).resolve().relative_to(ROOT)),
        training_acquisition_count=len(training['controls']),heldout_acquisition_count=len(heldout['controls']),
        total_acquisition_count=len(all_controls),total_unique_control_setting_count=len(np.unique(all_controls,axis=0)),
        full_existing_frame_count=len(all_controls)*9,five_input_frame_count=len(all_controls)*5,
        all_thirteen_optical_coordinates_free=True,source_covariance_granted=False,omitted_y_data_filled=False,
        both_detector_coordinates_retained=True,per_acquisition_alignment_coordinates_profiled=2,
        original_signed_response_likelihood_appended=False,results=results,comparison=comparison,y_validation=y_check,
        global_uniqueness_proved=False,x_only_cold_start_demonstrated=False,
        exact_input_calibration_and_round_optics_conditional=True,assumptions=[
            'Both detector coordinates are measured for reference,+/-x position,+/-x tilt; x-only means input basis, not detector readout.',
            'Round canonical optics plus unknown sample magnetic shear constrain the map; a generic2x4 map is not observable from these inputs.',
            'Per-image centroid covariance is the full2x2 marginal of the six-coordinate Gaussian image fit; no source shape or dose is supplied.',
            'A common alignment is profiled per optical acquisition; no centroid information is added twice.',
            'Warm starts are previous full-nine-input image-derived best/rival fits, not generating truth: independent x-only initialization is not demonstrated.',
            'The five- and nine-input matched fits use the same full centroid-profile likelihood; saved signed-response fit is separately identified.',
            'All frozen train/heldout roles are preserved. Historical nine-input analyses already inspected these images; no virgin validation claim.',
            'Six y-check states are selected from controls/roles before ablation fitting. Their24 extra images test roundness/calibration but do not prove absence of astigmatism everywhere.',
            'Stable alignment, exact input/pixel calibration, Gaussian-summary noise and hard-edge round optics remain conditional instrument assumptions.'])
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovery',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-nfev',type=int,default=1000);args=parser.parse_args();jax.config.update('jax_enable_x64',True)
    result=run_audit(args.recovery,args.output,max_nfev=args.max_nfev)
    print('Saved',args.output,result['comparison'],flush=True)
