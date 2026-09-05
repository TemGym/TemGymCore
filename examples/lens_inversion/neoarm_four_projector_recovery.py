"""Training-only noisy-map initialization and joint four-projector recovery."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import os
import time

os.environ.setdefault('JAX_ENABLE_X64','true');os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import jax
import jax.numpy as jnp
import numpy as np
from neoarm_il1_extension_design import FourProjectorResponse,analyser_matrix,reconstruct
from neoarm_il2_extension_design import ThreeProjectorResponse
from neoarm_coupled_projector_fit import grouped_stage,physical_upstream_coordinates,block_diagonal,initialize_previous_stage
from neoarm_last_lens_physical_fit import RoundLensResponse
from neoarm_three_projector_recovery import Problem
from neoarm_projector_branch_recovery import predictive_diagnostic
from temgym_core.jax_least_squares import least_squares

ROOT=Path(__file__).resolve().parents[2];FOLDER=ROOT/'examples/generated/neoarm_inversion/imaging_recipe'
PARAMETER_NAMES=('log_combined_span_m','log_PL1_span_over_IL3_span','log_IL2_span_over_IL3_span','log_IL1_span_over_IL3_span')+tuple(
    f'{lens}.{name}' for lens in ('PL1','IL3','IL2','IL1') for name in ('log_gap_over_length','q_bias','q_slope'))+(
    'prefix_log_row_norm','prefix_row_angle','prefix_row_shear','prefix_rotation_rad','sample_kappa_times_balance_length')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physical_coordinates(parameters):
    p=jnp.asarray(parameters);spans=p[0]+jax.nn.log_softmax(jnp.array([p[1],0.,p[2],p[3]]));values=[]
    for i in range(4):
        index=4+3*i;ratio=p[index];values.extend((spans[i]-jax.nn.softplus(ratio),spans[i]-jax.nn.softplus(-ratio),p[index+1],p[index+2]))
    return jnp.r_[jnp.asarray(values),p[16:]]


def encode_physical(physical):
    p=np.asarray(physical);stages=p[:16].reshape(4,4);spans=np.exp(stages[:,:2]).sum(axis=1)
    values=[np.log(spans.sum()),np.log(spans[0]/spans[1]),np.log(spans[2]/spans[1]),np.log(spans[3]/spans[1])]
    for stage in stages:values.extend((stage[1]-stage[0],stage[2],stage[3]))
    return np.r_[values,p[16:]]


class FittingModel:
    def __init__(self,calibration):self.physical=FourProjectorResponse(calibration)
    def predict(self,p,c):return self.physical.predict(physical_coordinates(p),c)


def selected_data(arrays,training):
    mask=np.asarray(arrays['training_mask' if training else 'heldout_mask']);other=np.asarray(arrays['heldout_mask' if training else 'training_mask'])
    if mask.dtype!=bool or other.dtype!=bool or np.any(mask&other) or not np.all(mask|other):raise ValueError('Acquisition masks must be disjoint exhaustive booleans.')
    pair=np.asarray(arrays['downstream_controls']);offsets=np.asarray(arrays['il1_controls'])[mask]
    if pair.shape!=(2,3):raise ValueError('Two downstream(PL1,IL3,IL2) controls are required.')
    indices=np.flatnonzero(np.repeat(mask,16));controls=np.array([[*a,u] for u in offsets for a in pair])
    response=np.asarray(arrays['response'])[mask].reshape(-1,2,4);covariance=np.asarray(arrays['joint_response_covariance'])[np.ix_(indices,indices)]
    if not all(np.all(np.isfinite(v)) for v in (controls,response,covariance)):raise ValueError('Selected data must be finite.')
    return controls,response,covariance


def fit_seed(problem,calibration,old_physical,initializer,*,max_nfev=1200):
    previous=np.asarray(initializer['parameters']).copy();nominal=np.asarray(ThreeProjectorResponse(calibration).upstream_matrix(old_physical))
    remaining=.9-np.exp(np.asarray(old_physical[:12]).reshape(3,4)[:,:2]).sum()
    if remaining<=0:raise ValueError('The inherited fit exceeds the combined physical span allowance.')
    if np.exp(previous[:2]).sum()>=remaining:previous[:2]+=np.log(.95*remaining/np.exp(previous[:2]).sum())
    prefix=np.linalg.solve(np.asarray(grouped_stage(jnp.asarray(previous),0.)),nominal)
    initial=encode_physical(np.r_[old_physical[:12],previous,physical_upstream_coordinates(prefix,calibration)])
    negative=previous[3]<0
    lower=np.array([np.log(.004),-8.,-8.,-8.,np.log(.05),-np.pi,-4*np.pi,
        np.log(.005),0.,.01,np.log(.005),-3*np.pi,.01,np.log(.005),-3*np.pi,-4*np.pi if negative else .01,
        -15.,-4*np.pi,-1e5,-4*np.pi,-1e4])
    upper=np.array([np.log(.9),8.,8.,8.,np.log(200.),0.,-.01,np.log(1000.),3*np.pi/2,4*np.pi,
        np.log(1000.),3*np.pi,4*np.pi,np.log(1000.),3*np.pi,-.01 if negative else 4*np.pi,15.,4*np.pi,1e5,4*np.pi,1e4])
    initial=np.clip(initial,lower+1e-12*(upper-lower),upper-1e-12*(upper-lower));norms=np.linalg.norm(np.asarray(problem.jacobian(initial)),axis=0)
    if not np.all(np.isfinite(norms)&(norms>0)):return dict(optimizer_success=False,status='unresolved_seed_jacobian',initializer=initializer)
    fit=least_squares(problem.residual,initial,bounds=(lower,upper),x_scale=1/norms,max_nfev=max_nfev,max_iter=2400,ftol=1e-11,xtol=1e-11,gtol=1e-7)
    p=np.asarray(fit.x);jac=np.asarray(fit.jacobian);gradient=jac.T@np.asarray(fit.residual)
    return dict(initial=initial.tolist(),parameters=p.tolist(),physical_coordinates=np.asarray(physical_coordinates(p)).tolist(),bounds=[lower.tolist(),upper.tolist()],
        optimizer_success=bool(fit.success),termination=fit.status,nfev=fit.nfev,chi_square=float(2*fit.cost),initializer=initializer,
        status='tested_local_four_projector_fit' if fit.success else 'unresolved_four_projector_optimizer',
        active_bounds=np.flatnonzero(np.minimum(abs(p-lower),abs(p-upper))<1e-7).tolist(),
        normalized_gradient_max=float(np.max(abs(gradient)/np.linalg.norm(jac,axis=0))),all_twenty_one_coordinates_released=True)


def information(problem,p):
    jac=np.asarray(problem.jacobian(p));norm=np.linalg.norm(jac,axis=0);_,s,v=np.linalg.svd(jac/norm,full_matrices=False);rank=int(np.sum(s>s[0]*1e-10))
    result=dict(rank=rank,free_coordinate_count=21,normalized_singular_values=s.tolist(),parameter_covariance=None,physical_covariance=None,
        interpretation='Local untruncated fixed-noise marginal covariance; broad nonlinear profiles and global alternatives remain separate tests.')
    if rank==21:
        factor=(v.T/s)/norm[:,None];physical=np.asarray(jax.jacfwd(physical_coordinates)(jnp.asarray(p)))@factor
        result.update(parameter_covariance=(factor@factor.T).tolist(),physical_covariance=(physical@physical.T).tolist(),
            length_gap_standard_errors_percent=(100*np.linalg.norm(physical[[0,1,4,5,8,9,12,13]],axis=1)).tolist())
    return result


def run(output,*,max_nfev=1200):
    if jax.default_backend()!='gpu' or not jax.config.x64_enabled:raise RuntimeError('Use GPU float64 for optical propagation and fitting.')
    started=time.perf_counter();old_path=FOLDER/'three_projector_recovery.json';new_path=FOLDER/'il1_extension_images_inference.json'
    old=json.loads(old_path.read_text());new=json.loads(new_path.read_text());sources=dict(old['source_hashes']);sources.update(new['source_hashes'])
    for name,h in sources.items():
        if digest(ROOT/name)!=h:raise ValueError(f'Stale recovery input:{name}')
    if not new['paired_data_admitted'] or not new['calibration_response_noise_independent']:raise ValueError('Admitted independent fresh exposures are required.')
    path=ROOT/new['arrays_path']
    if digest(path)!=new['arrays_sha256']:raise ValueError('Inference NPZ checksum mismatch.')
    with np.load(path,allow_pickle=False) as arrays:
        fresh=selected_data(arrays,True);pair=arrays['downstream_controls'].copy();offsets=arrays['il1_controls'][arrays['training_mask']].copy()
    calibration=RoundLensResponse(tuple(new['input_steps_grouped']),new['detector_pixel_m']);model=FittingModel(calibration)
    controls=np.concatenate((np.column_stack((old['training_controls'],np.zeros(len(old['training_controls'])))),fresh[0]));data=np.concatenate((old['training_response'],fresh[1]))
    covariance=block_diagonal([np.asarray(old['training_response_covariance']),fresh[2]]);problem=Problem(model,controls,data,covariance)
    old_physical=np.asarray(old['best']['physical_coordinates']);calibration_cov=np.asarray(old['information']['physical_covariance'])[:12,:12]
    recovered=reconstruct(fresh[1].reshape(-1,2,2,4),fresh[2],old_physical[:12],calibration_cov,pair,calibration.input_steps_grouped,calibration.pixel_m)
    a=np.asarray(analyser_matrix(old_physical[:12],pair));scan=SimpleNamespace(maps=np.asarray(recovered['maps']),joint_covariance=np.asarray(recovered['joint_covariance']),
        balanced_analyser_condition_number=float(np.linalg.cond(a*np.array([1.,1.,100.,100.]))))
    initializers,init_diagnostic=initialize_previous_stage(scan,offsets,max_nfev=400)
    for source in (Path(__file__),old_path,new_path,path,ROOT/'examples/lens_inversion/neoarm_il1_extension_design.py'):sources[str(source.relative_to(ROOT))]=digest(source)
    rows=[]
    for index,initializer in enumerate(initializers):
        row=fit_seed(problem,calibration,old_physical,initializer,max_nfev=max_nfev);row['start_index']=index;rows.append(row)
        print('four projectors',index,row['status'],row.get('chi_square'),row.get('nfev'),flush=True)
    converged=[row for row in rows if row['optimizer_success']];best=min(converged,key=lambda row:row['chi_square']) if converged else None
    info={};heldout={};truth={}
    if best is not None:
        p=np.asarray(best['parameters']);info=information(problem,p)
        with np.load(path,allow_pickle=False) as arrays:held=selected_data(arrays,False)
        heldout=predictive_diagnostic(model,p,info.get('parameter_covariance'),held);heldout.update(used_for_initialization_or_training_selection=False,loaded_after_training_winner_fixed=True)
        truth_path=ROOT/new['truth_path']
        if digest(truth_path)!=new['truth_sha256']:raise ValueError('Postfit truth checksum mismatch.')
        generating=json.loads(truth_path.read_text());stages=[]
        for name in ('PL1','IL3','IL2','IL1'):
            lens=generating[name];stages.extend((np.log(lens['length_m']),np.log(lens['gap_m']),lens['q_bias'],lens['q_slope']))
        prefix=physical_upstream_coordinates(generating['upstream_to_IL1_mechanical'],calibration);pt=encode_physical(np.r_[stages,prefix])
        difference=np.asarray(physical_coordinates(p)-physical_coordinates(pt));residual=np.asarray(problem.residual(pt));ix=np.array([0,1,4,5,8,9,12,13])
        truth=dict(path=new['truth_path'],sha256=new['truth_sha256'],accessed_after_training_and_fresh_heldout_prediction=True,
            physical_errors=difference.tolist(),length_gap_errors_percent=(100*np.expm1(difference[ix])).tolist(),training_chi_square_at_truth=float(residual@residual),
            training_response_coordinate_count=len(residual))
        if info.get('physical_covariance') is not None:truth['physical_errors_sigma']=(difference/np.sqrt(np.diag(np.asarray(info['physical_covariance'])))).tolist()
        sources[new['truth_path']]=digest(truth_path)
    if any(digest(ROOT/name)!=h for name,h in sources.items()):raise RuntimeError('A frozen inference dependency changed during execution.')
    report=dict(schema='neoarm-four-projector-noisy-image-recovery-v1',source_hashes=sources,source_hashes_unchanged_during_execution=True,backend=jax.default_backend(),
        x64_enabled=jax.config.x64_enabled,elapsed_seconds=time.perf_counter()-started,parameter_names=PARAMETER_NAMES,
        old_training_report=str(old_path.relative_to(ROOT)),new_inference_report=str(new_path.relative_to(ROOT)),new_arrays_path=new['arrays_path'],new_arrays_sha256=new['arrays_sha256'],
        training_controls=controls.tolist(),training_response=data.tolist(),training_response_covariance=covariance.tolist(),original_training_coordinate_count=272,
        new_training_coordinate_count=fresh[1].size,training_response_coordinate_count=data.size,initialization=init_diagnostic,initialization_reconstruction=recovered,
        initializers=initializers,rows=rows,best=best,information=info,fresh_heldout=heldout,truth_diagnostic=truth,start_count=len(rows),converged_start_count=len(converged),
        unresolved_start_count=len(rows)-len(converged),all_twenty_one_coordinates_released=True,calibration_posterior_added_as_prior=False,original_data_duplicated=False,
        generating_IL1_used_for_initialization=False,fresh_heldout_used_for_initialization_or_training_selection=False,global_uniqueness_proved=False,
        limitations=['The old272 and new48 response coordinates occur once in one full covariance likelihood; no calibrated-posterior prior is appended.',
            'Noisy relative maps with shared12-coordinate downstream calibration uncertainty initialize IL1; relative-map weights are initialization only.',
            'IL1 starts cover six pi-wide bias intervals, both slope signs and two generic geometry seeds, without using generating IL1 geometry or excitation.',
            'All21 coordinates are free within the conditional.9m combined span and inherited PL1/IL3/IL2 branch bounds; failed starts remain unresolved.',
            'Fisher predicted weak IL1 length/gap precision. A favourable data realization would not establish robust1% calibration.',
            'The unknown prefix still represents the fixed objective section; this is not a separately identified physical objective or condenser.',
            'Exact input/pixel/PSF calibration, stable alignment, hard-edge fields and exact synthetic WLS variance remain assumptions.',
            'Fresh+.375 IL1 responses are used only after the training winner is fixed. Previous heldouts never enter the new training likelihood.'])
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--max-nfev',type=int,default=1200)
    args=parser.parse_args();result=run(args.output,max_nfev=args.max_nfev);print('Saved',args.output,'best',None if result['best'] is None else result['best']['chi_square'],flush=True)
