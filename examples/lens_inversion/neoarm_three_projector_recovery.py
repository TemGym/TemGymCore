"""Blind, response-only three-projector recovery from fresh IL2 images.

Relative noisy maps initialize IL2. They never add a second calibration prior:
the final likelihood uses the old224 and fresh48 response coordinates once.
Fresh +.375 IL2 data are evaluated only after the training winner is fixed.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import os
import time

os.environ.setdefault('JAX_ENABLE_X64','true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
import numpy as np

from neoarm_il2_extension_design import ThreeProjectorResponse,downstream_pair_matrix,reconstruct_with_shared_calibration
from neoarm_coupled_projector_fit import (
    CoupledProjectorResponse,stage_coordinates,grouped_stage,physical_upstream_coordinates,
    block_diagonal,initialize_previous_stage,
)
from neoarm_last_lens_physical_fit import RoundLensResponse
from neoarm_projector_branch_recovery import predictive_diagnostic
from temgym_core.jax_least_squares import least_squares

ROOT=Path(__file__).resolve().parents[2]
FOLDER=ROOT/'examples/generated/neoarm_inversion/imaging_recipe'
PARAMETER_NAMES=('log_combined_span_m','log_PL1_span_over_IL3_span','log_IL2_span_over_IL3_span',
    'PL1.log_gap_over_length','PL1.q_bias','PL1.q_slope','IL3.log_gap_over_length','IL3.q_bias','IL3.q_slope',
    'IL2.log_gap_over_length','IL2.q_bias','IL2.q_slope','prefix_log_row_norm','prefix_row_angle',
    'prefix_row_shear','prefix_rotation_rad','sample_kappa_times_balance_length')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physical_coordinates(parameters):
    p=jnp.asarray(parameters)
    logspans=p[0]+jax.nn.log_softmax(jnp.array([p[1],0.,p[2]]))
    stages=[]
    for i,index in enumerate((3,6,9)):
        ratio=p[index]
        stages.extend((logspans[i]-jax.nn.softplus(ratio),logspans[i]-jax.nn.softplus(-ratio),p[index+1],p[index+2]))
    return jnp.r_[jnp.asarray(stages),p[12:]]


def encode_physical(physical):
    physical=np.asarray(physical);stages=physical[:12].reshape(3,4)
    spans=np.exp(stages[:,:2]).sum(axis=1)
    result=[np.log(spans.sum()),np.log(spans[0]/spans[1]),np.log(spans[2]/spans[1])]
    for stage in stages:result.extend((stage[1]-stage[0],stage[2],stage[3]))
    return np.r_[result,physical[12:]]


class FittingModel:
    def __init__(self,calibration):self.physical=ThreeProjectorResponse(calibration)
    def predict(self,parameters,controls):return self.physical.predict(physical_coordinates(parameters),controls)


def selected_data(arrays,training):
    mask=np.asarray(arrays['training_mask' if training else 'heldout_mask'])
    other=np.asarray(arrays['heldout_mask' if training else 'training_mask'])
    if mask.dtype!=bool or other.dtype!=bool or np.any(mask&other) or not np.all(mask|other):
        raise ValueError('Disjoint exhaustive boolean acquisition masks are required.')
    pair=np.asarray(arrays['downstream_controls']);offsets=np.asarray(arrays['il2_controls'])[mask]
    if pair.shape!=(2,2):raise ValueError('Two downstream controls (PL1,IL3) are required.')
    indices=np.flatnonzero(np.repeat(mask,16))
    controls=np.array([[*a,u] for u in offsets for a in pair])
    data=np.asarray(arrays['response'])[mask].reshape(-1,2,4)
    covariance=np.asarray(arrays['joint_response_covariance'])[np.ix_(indices,indices)]
    if not all(np.all(np.isfinite(v)) for v in (controls,data,covariance)):raise ValueError('The selected data are not finite.')
    return controls,data,covariance


class Problem:
    def __init__(self,model,controls,data,covariance):
        self.model=model;self.controls=jnp.asarray(controls);self.data=jnp.asarray(data)
        self.covariance=np.asarray(covariance);chol=jnp.asarray(np.linalg.cholesky(covariance))
        self.residual=jax.jit(lambda p:solve_triangular(chol,(model.predict(p,self.controls)-self.data).reshape(-1),lower=True))
        self.jacobian=jax.jit(jax.jacfwd(self.residual))


def fit_seed(problem,calibration,old_parameters,initializer,*,max_nfev=1000):
    downstream=np.asarray(stage_coordinates(old_parameters));previous=np.asarray(initializer['parameters']).copy()
    two=CoupledProjectorResponse(calibration);nominal=np.asarray(two.upstream_matrix(old_parameters))
    remaining=.9-np.exp(downstream.reshape(2,4)[:,:2]).sum()
    if np.exp(previous[:2]).sum()>=remaining:previous[:2]+=np.log(.95*remaining/np.exp(previous[:2]).sum())
    prefix=np.linalg.solve(np.asarray(grouped_stage(jnp.asarray(previous),0.)),nominal)
    projected=physical_upstream_coordinates(prefix,calibration)
    initial=encode_physical(np.r_[downstream,previous,projected])
    negative=previous[3]<0
    lower=np.array([np.log(.003),-8.,-8.,np.log(.05),-np.pi,-4*np.pi,
        np.log(.005),0.,.01,np.log(.005),-3*np.pi,-4*np.pi if negative else .01,
        -15.,-4*np.pi,-1e5,-4*np.pi,-1e4])
    upper=np.array([np.log(.9),8.,8.,np.log(200.),0.,-.01,
        np.log(1000.),3*np.pi/2,4*np.pi,np.log(1000.),3*np.pi,-.01 if negative else 4*np.pi,
        15.,4*np.pi,1e5,4*np.pi,1e4])
    initial=np.clip(initial,lower+1e-12*(upper-lower),upper-1e-12*(upper-lower))
    norms=np.linalg.norm(np.asarray(problem.jacobian(initial)),axis=0)
    if not np.all(np.isfinite(norms)&(norms>0)):return dict(optimizer_success=False,status='unresolved_seed_jacobian',initializer=initializer)
    fit=least_squares(problem.residual,initial,bounds=(lower,upper),x_scale=1/norms,
        max_nfev=max_nfev,max_iter=2000,ftol=1e-11,xtol=1e-11,gtol=1e-7)
    p=np.asarray(fit.x);jac=np.asarray(fit.jacobian);gradient=jac.T@np.asarray(fit.residual)
    return dict(initial=initial.tolist(),parameters=p.tolist(),physical_coordinates=np.asarray(physical_coordinates(p)).tolist(),
        bounds=[lower.tolist(),upper.tolist()],optimizer_success=bool(fit.success),status='tested_local_three_projector_fit' if fit.success else 'unresolved_three_projector_optimizer',
        termination=fit.status,nfev=fit.nfev,chi_square=float(2*fit.cost),initializer=initializer,
        active_bounds=np.flatnonzero(np.minimum(abs(p-lower),abs(p-upper))<1e-7).tolist(),
        normalized_gradient_max=float(np.max(abs(gradient)/np.linalg.norm(jac,axis=0))),all_seventeen_coordinates_released=True)


def information(problem,parameters):
    jac=np.asarray(problem.jacobian(parameters));norms=np.linalg.norm(jac,axis=0)
    _,singular,right=np.linalg.svd(jac/norms,full_matrices=False);rank=int(np.sum(singular>singular[0]*1e-10))
    result=dict(rank=rank,free_coordinate_count=17,normalized_singular_values=singular.tolist(),parameter_covariance=None,physical_covariance=None,
        covariance_interpretation='Untruncated local fixed-noise Gauss-Newton marginal covariance; no global/profile guarantee.')
    if rank==17:
        factor=(right.T/singular)/norms[:,None];physical=np.asarray(jax.jacfwd(physical_coordinates)(jnp.asarray(parameters)))@factor
        result.update(parameter_covariance=(factor@factor.T).tolist(),physical_covariance=(physical@physical.T).tolist(),
            length_gap_standard_errors_percent=(100*np.linalg.norm(physical[[0,1,4,5,8,9]],axis=1)).tolist())
    return result


def run(output,*,max_nfev=1000):
    if jax.default_backend()!='gpu' or not jax.config.x64_enabled:raise RuntimeError('Use GPU float64 for the optical forward/fit.')
    started=time.perf_counter();old_path=FOLDER/'projector_branch_recovery.json';new_path=FOLDER/'il2_extension_images_inference.json'
    old=json.loads(old_path.read_text());new=json.loads(new_path.read_text());sources=dict(old['source_hashes']);sources.update(new['source_hashes'])
    for name,value in sources.items():
        if digest(ROOT/name)!=value:raise ValueError(f'Stale fit input:{name}')
    if not new['paired_data_admitted'] or not new['calibration_response_noise_independent']:raise ValueError('Fresh independent image admission is required.')
    path=ROOT/new['arrays_path']
    if digest(path)!=new['arrays_sha256']:raise ValueError('New response NPZ checksum mismatch.')
    with np.load(path,allow_pickle=False) as arrays:
        fresh=selected_data(arrays,True);pair=arrays['downstream_controls'].copy();offsets=arrays['il2_controls'][arrays['training_mask']].copy()
    calibration=RoundLensResponse(tuple(new['input_steps_grouped']),new['detector_pixel_m']);model=FittingModel(calibration)
    controls=np.concatenate((np.column_stack((old['training_controls'],np.zeros(len(old['training_controls'])))),fresh[0]))
    data=np.concatenate((old['training_response'],fresh[1]));covariance=block_diagonal([np.asarray(old['training_response_covariance']),fresh[2]])
    problem=Problem(model,controls,data,covariance)
    physical_eight=np.asarray(stage_coordinates(old['best']['parameters']))
    reconstructed=reconstruct_with_shared_calibration(fresh[1].reshape(-1,2,2,4),fresh[2],physical_eight,
        np.asarray(old['information']['physical_covariance']),pair,calibration.input_steps_grouped,calibration.pixel_m)
    analyser=np.asarray(downstream_pair_matrix(physical_eight,pair))
    scan=SimpleNamespace(maps=np.asarray(reconstructed['maps']),joint_covariance=np.asarray(reconstructed['joint_covariance']),
        balanced_analyser_condition_number=float(np.linalg.cond(analyser*np.array([1.,1.,100.,100.]))))
    initializers,init_diagnostic=initialize_previous_stage(scan,offsets,max_nfev=400)
    for source in (Path(__file__),old_path,new_path,path,ROOT/'examples/lens_inversion/neoarm_il2_extension_design.py'):
        sources[str(source.relative_to(ROOT))]=digest(source)
    rows=[]
    for index,initializer in enumerate(initializers):
        row=fit_seed(problem,calibration,old['best']['parameters'],initializer,max_nfev=max_nfev)
        row['start_index']=index;rows.append(row)
        print('three projectors',index,row['status'],row.get('chi_square'),row.get('nfev'),flush=True)
    converged=[row for row in rows if row['optimizer_success']];best=min(converged,key=lambda row:row['chi_square']) if converged else None
    info={};heldout={};truth={}
    if best is not None:
        fitted=np.asarray(best['parameters']);info=information(problem,fitted)
        with np.load(path,allow_pickle=False) as arrays:held=selected_data(arrays,False)
        heldout=predictive_diagnostic(model,fitted,info.get('parameter_covariance'),held)
        heldout.update(used_for_initialization_or_training_selection=False,loaded_after_training_winner_fixed=True)
        truth_path=ROOT/new['truth_path']
        if digest(truth_path)!=new['truth_sha256']:raise ValueError('Postfit truth checksum mismatch.')
        generating=json.loads(truth_path.read_text());stages=[]
        for name in ('PL1','IL3','IL2'):
            lens=generating[name];stages.extend((np.log(lens['length_m']),np.log(lens['gap_m']),lens['q_bias'],lens['q_slope']))
        prefix=physical_upstream_coordinates(generating['upstream_to_IL2_mechanical'],calibration)
        truth_parameters=encode_physical(np.r_[stages,prefix]);difference=np.asarray(physical_coordinates(fitted)-physical_coordinates(truth_parameters))
        residual=np.asarray(problem.residual(truth_parameters));indices=np.array([0,1,4,5,8,9])
        truth=dict(path=new['truth_path'],sha256=new['truth_sha256'],accessed_after_training_and_heldout_prediction=True,
            physical_errors=difference.tolist(),length_gap_errors_percent=(100*np.expm1(difference[indices])).tolist(),
            training_chi_square_at_truth=float(residual@residual),training_response_coordinate_count=len(residual))
        if info.get('physical_covariance') is not None:
            sigma=np.sqrt(np.diag(np.asarray(info['physical_covariance'])))
            truth['physical_errors_sigma']=(difference/sigma).tolist()
        sources[new['truth_path']]=digest(truth_path)
    if any(digest(ROOT/name)!=value for name,value in sources.items()):raise RuntimeError('A fit dependency changed during execution.')
    report=dict(schema='neoarm-three-projector-noisy-image-recovery-v1',source_hashes=sources,source_hashes_unchanged_during_execution=True,
        backend=jax.default_backend(),x64_enabled=jax.config.x64_enabled,elapsed_seconds=time.perf_counter()-started,parameter_names=PARAMETER_NAMES,
        old_training_report=str(old_path.relative_to(ROOT)),new_inference_report=str(new_path.relative_to(ROOT)),new_arrays_path=new['arrays_path'],new_arrays_sha256=new['arrays_sha256'],
        training_controls=controls.tolist(),training_response=data.tolist(),training_response_covariance=covariance.tolist(),
        original_training_coordinate_count=224,new_training_coordinate_count=fresh[1].size,training_response_coordinate_count=data.size,
        initialization=init_diagnostic,initialization_reconstruction=reconstructed,initializers=initializers,rows=rows,best=best,
        information=info,fresh_heldout=heldout,truth_diagnostic=truth,start_count=len(rows),converged_start_count=len(converged),unresolved_start_count=len(rows)-len(converged),
        all_seventeen_coordinates_released=True,calibration_posterior_added_as_prior=False,original_data_duplicated=False,
        generating_IL2_used_for_initialization=False,fresh_heldout_used_for_initialization_or_training_selection=False,global_uniqueness_proved=False,
        limitations=['Relative noisy maps initialize IL2 using full shared downstream calibration covariance; the relative diagonal metric is initialization only.',
            'Two generic geometry seeds, six pi-wide IL2 intervals and both slope signs generate starts; no generating IL2 field strength or geometry is granted.',
            'The final likelihood includes original224 and fresh48 response coordinates once, all17 optical coordinates free.',
            'A continuous IL2 bias box−3pi…+3pi allows migration between tested intervals; failed starts remain unresolved.',
            'The combined.9m span cap, negative-first PL1 branch and tested positive IL3 bias range remain conditional assumptions.',
            'Exact input/pixel/PSF calibration, stable alignment and the hard-edge objective/round-optics surrogate remain limitations.',
            'No known source covariance enters the response fit. Raw-pixel shape likelihood and condenser recovery are separate tasks.',
            'Fresh+.375 IL2 images are evaluated only after all training fits finish; no earlier heldout data are added to training.'])
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--max-nfev',type=int,default=1000)
    args=parser.parse_args();result=run(args.output,max_nfev=args.max_nfev);print('Saved',args.output,'best',None if result['best'] is None else result['best']['chi_square'],flush=True)
