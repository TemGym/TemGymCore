"""Conditional noisy condenser recovery after a fresh strong-excitation batch.

Every original and new training observation enters once. The source covariance,
aperture-frame angle, and common radius scale are free alongside all12 optics.
Fresh heldouts are loaded only after all training fits have finished.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time

os.environ.setdefault('JAX_ENABLE_X64','true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import jax
import jax.numpy as jnp
import numpy as np

from neoarm_condenser_extension_design import load_context,training_residual,covariance_factor,FOLDER
from neoarm_condenser_covariance_fit import (
    VisibleSourceModel,svd_gauss_newton,quantum_completion,geometry_gaps,
)
from neoarm_condenser_image_recovery import linear_source_initializer
from neoarm_condenser_covariance_fit import visible_coordinates
from neoarm_aperture_capture import ROOT
from neoarm_shared_objective_audit import provenance_snapshot


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_data(arrays,training):
    mask=np.asarray(arrays['training_mask' if training else 'heldout_mask'])
    other=np.asarray(arrays['heldout_mask' if training else 'training_mask'])
    if mask.dtype!=bool or other.dtype!=bool or np.any(mask&other) or not np.all(mask|other):
        raise ValueError('Disjoint exhaustive boolean acquisition masks are required.')
    result=tuple(np.asarray(arrays[key])[mask] for key in ('controls','beam_observables','beam_observable_covariances'))
    if not all(np.isfinite(a).all() for a in result):raise ValueError('Selected observations must be finite.')
    return result


def beam_residual(quotient,selected):
    controls,data,covariance=map(jnp.asarray,selected)
    root=jnp.linalg.cholesky(covariance)
    def residual(p):
        prediction,_=quotient.observables(p,controls,jnp.zeros((0,7)))
        return jnp.linalg.solve(root,(prediction-data)[...,None]).reshape(-1)
    return residual


def prediction_diagnostic(residual,parameters,covariance):
    r=np.asarray(jax.jit(residual)(parameters))
    jac=np.asarray(jax.jit(jax.jacfwd(residual))(parameters))
    predictive=np.eye(len(r))+jac@covariance@jac.T
    return dict(chi_square=float(r@r),coordinate_count=len(r),
        predictive_chi_square=float(r@np.linalg.solve(predictive,r)),
        residual=r.tolist())


def run(inference_path,output,*,max_iterations=250,solver=None,solver_options=None,
        include_rough_data_start=False):
    if jax.default_backend()!='gpu' or not jax.config.x64_enabled:
        raise RuntimeError('Optical forward and recovery require GPU float64.')
    started=time.perf_counter()
    active_solver=svd_gauss_newton if solver is None else solver
    solver_options={} if solver_options is None else dict(solver_options)
    inference_path=Path(inference_path).resolve();new=json.loads(inference_path.read_text())
    for name,value in new['source_hashes'].items():
        if digest(ROOT/name)!=value:raise ValueError(f'Stale new acquisition: {name}')
    arrays_path=ROOT/new['arrays_path']
    if digest(arrays_path)!=new['arrays_sha256']:raise ValueError('New response array checksum mismatch.')
    if not new['all_images_accepted']:raise ValueError('Only independently admitted images may enter the fit.')
    with np.load(arrays_path,allow_pickle=False) as arrays:fresh=selected_data(arrays,True)
    old,arrays,repair,aperture,model=load_context();quotient=VisibleSourceModel(model)
    old_residual=training_residual(quotient,arrays,aperture)
    new_residual=beam_residual(quotient,fresh)
    def residual(p):return jnp.r_[old_residual(p),new_residual(p)]
    controls=np.concatenate((arrays['controls'][:8],fresh[0]))
    data=np.concatenate((arrays['beam_observables'][:8],fresh[1]))
    covariance=np.concatenate((arrays['beam_observable_covariances'][:8],fresh[2]))
    apcontrols=np.asarray([r['controls'] for r in aperture['records'][:2]])
    initializers=[];base=model.base.copy()
    # The previous unaccepted iterate is an initializer, never a prior.
    initializers.append(('old_training_iterate',np.asarray(repair['records'][0]['parameters'])))
    for sign in ((-1,1,0) if include_rough_data_start else (-1,1)):
        perturbed=base.reshape(3,4).copy()
        perturbed[:,0]+=sign*np.array([1e-3,-.3e-3,.15e-3])
        length_shift=sign*np.array([.08,-.06,.10])
        perturbed[:,1]+=length_shift;perturbed[:,2:4]*=np.exp(length_shift[:,None]/2)
        model.base=perturbed.reshape(-1)
        initialized,_=linear_source_initializer(model,controls,data,covariance,apcontrols,
            arrays['aperture_observables'][:2],arrays['aperture_observable_covariances'][:2])
        optics,source,angle=model.decode(initialized)
        model.base=base.copy()
        p=np.r_[quotient.optical_coordinates(optics),visible_coordinates(source),float(angle),0.]
        label='rough_geometry_data_source' if sign==0 else f'offset_geometry_{sign:+}'
        initializers.append((label,p))
    sources=dict(new['source_hashes']);sources.update(repair['source_hashes']);sources.update(old['source_hashes'])
    loaded_sources,_=provenance_snapshot();sources.update(loaded_sources)
    for path in (Path(__file__),inference_path,arrays_path,FOLDER/'condenser_covariance_fit.json',
                 ROOT/'examples/lens_inversion/neoarm_condenser_extension_design.py'):
        sources[str(path.relative_to(ROOT))]=digest(path)
    rows=[]
    for name,initial in initializers:
        fit=active_solver(residual,initial,quotient.admissible,max_iterations=max_iterations,**solver_options)
        p=np.asarray(fit['parameters']);r=np.asarray(fit['residual'])
        optical=np.asarray(quotient.decode(p)[0]);completion=quantum_completion(p[12:21],interval=True)
        factor=covariance_factor(fit['jacobian']);physical=np.asarray(jax.jacfwd(lambda q:quotient.decode(q)[0])(p))@factor
        rows.append(dict(initializer=name,initial_parameters=initial.tolist(),parameters=p.tolist(),
            optimizer_success=fit['success'],termination=fit['reason'],normalized_gradient=fit['normalized_gradient'],
            evaluations=fit['evaluations'],history=fit['history'],total_chi_square=float(r@r),
            training_data_chi_square=float(r@r-(p[22]/.005)**2),radius_prior_chi_square=float((p[22]/.005)**2),
            optical_parameters=optical.tolist(),parameter_covariance=(factor@factor.T).tolist(),
            optical_covariance=(physical@physical.T).tolist(),physical_standard_errors=np.linalg.norm(physical,axis=1).reshape(3,4).tolist(),
            geometry_gaps_m=geometry_gaps(model,optical).tolist(),quantum_margin=completion['margin'],
            invisible_source_covariance_interval=completion['feasible_interval']))
        print(name,rows[-1]['optimizer_success'],rows[-1]['total_chi_square'],rows[-1]['normalized_gradient'],flush=True)
    converged=[r for r in rows if r['optimizer_success']]
    best=min(converged,key=lambda r:r['total_chi_square']) if converged else None
    held={};old_held={}
    if best is not None:
        with np.load(arrays_path,allow_pickle=False) as data:held_data=selected_data(data,False)
        held=prediction_diagnostic(beam_residual(quotient,held_data),np.asarray(best['parameters']),np.asarray(best['parameter_covariance']))
        held.update(loaded_after_training_winner_fixed=True,used_for_training_or_initialization=False)
        bc=jnp.asarray(arrays['controls'][8:]);ac=jnp.asarray([r['controls'] for r in aperture['records'][2:]])
        bo=jnp.asarray(arrays['beam_observables'][8:]);ao=jnp.asarray(arrays['aperture_observables'][2:])
        br=jnp.asarray(np.linalg.cholesky(arrays['beam_observable_covariances'][8:]))
        ar=jnp.asarray(np.linalg.cholesky(arrays['aperture_observable_covariances'][2:]))
        def old_held_residual(p):
            bp,ap=quotient.observables(p,bc,ac)
            return jnp.r_[jnp.linalg.solve(br,(bp-bo)[...,None]).reshape(-1),
                          jnp.linalg.solve(ar,(ap-ao)[...,None]).reshape(-1)]
        old_held=prediction_diagnostic(old_held_residual,np.asarray(best['parameters']),np.asarray(best['parameter_covariance']))
        old_held.update(previously_inspected_diagnostic_data=True,untouched_confirmatory_data=False,added_to_training=False)
    # Generator values are first accessed after every training fit. The
    # likelihood already treats the downstream/objective maps as calibrated.
    truth_path=ROOT/new['truth_path']
    if digest(truth_path)!=new['truth_sha256']:raise ValueError('Postfit truth checksum mismatch.')
    with np.load(truth_path,allow_pickle=False) as generating:
        true_optics=np.asarray(generating['optical_parameters'])[2:14]
        true_source=np.asarray(generating['source_covariance'])
        true_angle=float(generating['aperture_rotation_rad'])
    true_p=np.r_[quotient.optical_coordinates(true_optics),visible_coordinates(true_source),true_angle,0.]
    rt=np.asarray(jax.jit(residual)(true_p))
    truth=dict(accessed_after_all_training_fits=True,training_chi_square=float(rt@rt),
        training_data_coordinate_count=54+fresh[1].size)
    if best is not None:
        difference=np.asarray(best['optical_parameters'])-true_optics
        standard=np.asarray(best['physical_standard_errors']).reshape(-1)
        truth.update(optical_error_sigma=(difference/standard).reshape(3,4).tolist(),
            centre_errors_mm=(difference.reshape(3,4)[:,0]*1e3).tolist(),
            length_errors_percent=(100*np.expm1(difference.reshape(3,4)[:,1])).tolist())
        best['physically_consistent_local_recovery']=bool(np.max(abs(difference/standard))<3
            and abs((held['predictive_chi_square']-held['coordinate_count'])/np.sqrt(2*held['coordinate_count']))<5)
    sources[new['truth_path']]=digest(truth_path)
    if any(digest(ROOT/name)!=value for name,value in sources.items()):raise RuntimeError('Fit dependency changed during execution.')
    report=dict(schema='neoarm-condenser-fresh-extension-recovery-v1',source_hashes=sources,
        inference_path=str(inference_path.relative_to(ROOT)),arrays_path=str(arrays_path.relative_to(ROOT)),arrays_sha256=digest(arrays_path),
        backend=jax.default_backend(),x64_enabled=jax.config.x64_enabled,elapsed_seconds=time.perf_counter()-started,
        optimizer_implementation=f'{active_solver.__module__}.{active_solver.__name__}',
        rows=rows,best=best,fresh_heldout=held,reused_old_heldout=old_held,truth_diagnostic=truth,
        training_data_coordinate_count=54+fresh[1].size,
        old_training_coordinate_count=54,new_training_coordinate_count=fresh[1].size,
        all_twelve_optical_parameters_released=True,source_visible_coordinate_count=9,
        invisible_source_family_retained=True,aperture_frame_angle_free=True,common_radius_prior_count=1,
        old_posterior_added_as_prior=False,old_heldouts_added_to_training=False,
        generating_source_or_condenser_used_for_initialization=False,
        nontrivially_offset_geometry_starts=True,fresh_heldout_used_for_training_or_selection=False,
        rough_geometry_data_source_start_included=bool(include_rough_data_start),
        complete_full_column_recovery=False,global_uniqueness_proved=False,hardware_admitted=False,
        limitations=['Fixed conditional objective/downstream calibration and aperture-plane position.',
            'One unchanged unknown astigmatic source upstream of CL1; actual stigmator location remains unconfirmed.',
            'Larger synthetic condenser control ranges are not hardware commands or verified connected scans.',
            'Three local starts test initialization only, not all global polarity/excitation branches.',
            'Local covariance is not a nonlinear profile or parameter-refinement acceptance test.'])
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-iterations',type=int,default=250)
    args=parser.parse_args();run(args.inference,args.output,max_iterations=args.max_iterations)
