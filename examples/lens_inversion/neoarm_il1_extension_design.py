"""Conditional IL1 design with all four projectors and unknown prefix free."""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import os
import time
import tomllib

os.environ.setdefault('JAX_ENABLE_X64','true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import jax
import jax.numpy as jnp
import numpy as np

from neoarm_il2_extension_design import ThreeProjectorResponse,connected_component
from neoarm_coupled_projector_fit import radial_stage,grouped_stage,physical_upstream_coordinates,block_diagonal
from neoarm_last_lens_physical_fit import RoundLensResponse
from neoarm_last_lens_design import make_relay_model,GROUPED
from neoarm_last_lens_tomography import rotation,mechanical_shear,J2
from neoarm_projector_branch_experiment import pure_source,signed_inputs,approximate_response_covariance
from neoarm_last_lens_images import signed_response_from_image_fits
from neoarm_imaging_recipe import column_with_geometry_anchors
from temgym_core.neoarm_calibration import DetectorGeometry,hard_edge_column_from_config
from temgym_core.gaussian_image_likelihood import make_gaussian_detector_operator,encode_gaussian_image_parameters,gaussian_image_information_factor

ROOT=Path(__file__).resolve().parents[2];FOLDER=ROOT/'examples/generated/neoarm_inversion/imaging_recipe'
PARAMETER_NAMES=tuple(f'{lens}.{field}' for lens in ('PL1','IL3','IL2','IL1') for field in ('log_length_m','log_gap_m','q_bias','q_slope'))+(
    'prefix_log_row_norm','prefix_row_angle','prefix_row_shear','prefix_rotation_rad','sample_kappa_times_balance_length')


class FourProjectorResponse:
    def __init__(self,calibration):self.calibration=calibration
    def upstream_matrix(self,p,*,coordinates='mechanical'):
        return self.calibration.upstream_matrix(jnp.r_[jnp.zeros(4),jnp.asarray(p)[16:]],coordinates=coordinates)
    def predict(self,parameters,controls):
        p,c=jnp.asarray(parameters),jnp.asarray(controls)
        if p.shape!=(21,) or c.ndim!=2 or c.shape[1]!=4:raise ValueError('Expected21 parameters and(PL1,IL3,IL2,IL1) controls.')
        radial=self.calibration.radial_matrix(jnp.r_[jnp.zeros(4),p[16:]])
        for i in (3,2,1,0):radial=radial_stage(p[4*i:4*i+4],c[:,i])@radial
        aa,bb=radial[:,0,0],radial[:,0,1]
        position=aa[:,None,None]*jnp.eye(2)+(p[20]/self.calibration.balance_length_m)*bb[:,None,None]*J2
        phase=p[19]-sum(p[4*i+2]+p[4*i+3]*c[:,i] for i in range(4))
        result=rotation(phase)@jnp.concatenate((position,bb[:,None,None]*jnp.eye(2)),axis=-1)
        return result*(jnp.asarray(self.calibration.input_steps_grouped)/self.calibration.pixel_m)
    def canonical_maps(self,p,controls):
        p=jnp.asarray(p);prefix=self.upstream_matrix(p,coordinates='canonical')
        def trace(c):
            matrix=prefix
            for i in (3,2,1,0):matrix=grouped_stage(p[4*i:4*i+4],c[i])@matrix
            return matrix
        return jax.vmap(trace)(jnp.asarray(controls))


def extend_prefix(old_physical,provisional,calibration):
    nominal=np.asarray(ThreeProjectorResponse(calibration).upstream_matrix(old_physical))
    prefix=np.linalg.solve(np.asarray(grouped_stage(jnp.asarray(provisional),0.)),nominal)
    return np.r_[old_physical[:12],provisional,physical_upstream_coordinates(prefix,calibration)]


def fisher(jac,covariance):
    matrix=np.linalg.solve(np.linalg.cholesky(covariance),np.asarray(jac).reshape(-1,21));norm=np.linalg.norm(matrix,axis=0)
    _,singular,right=np.linalg.svd(matrix/np.where(norm>0,norm,1.),full_matrices=False);rank=int(np.sum(singular>singular[0]*1e-10))
    result=dict(rank=rank,free_coordinate_count=21,normalized_singular_values=singular.tolist(),parameter_covariance=None)
    if rank==21:
        factor=(right.T/singular)/norm[:,None]
        result.update(parameter_covariance=(factor@factor.T).tolist(),length_gap_standard_errors_percent=(100*np.linalg.norm(factor[[0,1,4,5,8,9,12,13]],axis=1)).tolist())
    return result


def moments(model,p,controls,source_cov):
    maps=model.canonical_maps(p,controls);covariance=jnp.asarray(source_cov)[np.ix_(GROUPED,GROUPED)]
    cov=maps[:,:2]@covariance@maps[:,:2].swapaxes(-1,-2)
    mechanical=maps@mechanical_shear(p[20]/model.calibration.balance_length_m)
    means=jnp.einsum('nai,fi->nfa',mechanical[:,:2],jnp.asarray(signed_inputs(model.calibration.input_steps_grouped)))
    return means,cov


def analyser_matrix(parameters,pair):
    p=jnp.asarray(parameters)
    def rows(c):
        matrix=jnp.eye(4)
        for i in (2,1,0):matrix=grouped_stage(p[4*i:4*i+4],c[i])@matrix
        return matrix[:2]
    return jax.vmap(rows)(jnp.asarray(pair)).reshape(4,4)


def reconstruct(response,covariance,physical_twelve,calibration_covariance,pair,steps,pixel):
    shape=np.asarray(response).shape
    def recover(data,p):
        a=analyser_matrix(p,pair);physical=data.reshape(-1,4,4)*(pixel/jnp.asarray(steps))[None,None,:]
        return jax.vmap(lambda matrix:jnp.linalg.solve(a,matrix))(physical).reshape(-1)
    data=jnp.asarray(response).reshape(-1);p=jnp.asarray(physical_twelve)
    jd,jp=map(np.asarray,jax.jacfwd(recover,argnums=(0,1))(data,p))
    measurement=jd@covariance@jd.T;shared=jp@calibration_covariance@jp.T
    return dict(maps=np.asarray(recover(data,p)).reshape(-1,4,4).tolist(),joint_covariance=(measurement+shared).tolist(),
        measurement_covariance=measurement.tolist(),shared_calibration_covariance=shared.tolist(),response_shape=list(shape),
        proposed_fresh_data_independent_of_previous_calibration_noise=True,initialization_only=True)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(output):
    if jax.default_backend()!='gpu' or not jax.config.x64_enabled:raise RuntimeError('Use GPU float64 for the forward design.')
    started=time.perf_counter();reference_path=FOLDER/'three_projector_recovery.json';reference=json.loads(reference_path.read_text());sources=dict(reference['source_hashes'])
    for name,value in sources.items():
        if digest(ROOT/name)!=value:raise ValueError(f'Stale design input:{name}')
    manifest_path=FOLDER/'il2_extension_images_inference.json';manifest=json.loads(manifest_path.read_text())
    calibration=RoundLensResponse(tuple(manifest['input_steps_grouped']),manifest['detector_pixel_m']);model=FourProjectorResponse(calibration)
    init_path=FOLDER/'measured_initialization.json';robust_path=FOLDER/'last_lens_robust_design.json'
    config_path=ROOT/'examples/lens_inversion/neoarm_inversion_config.toml';geometry_path=ROOT/'examples/microscope_models/measure_microscope_geometry.ipynb'
    initial=json.loads(init_path.read_text());robust=json.loads(robust_path.read_text())
    column=column_with_geometry_anchors(hard_edge_column_from_config(tomllib.loads(config_path.read_text())),geometry_path)
    _,decoded=make_relay_model(column,jnp.asarray(initial['parameters']),0.)
    lens,z,length,_,slope=decoded[-4];_,znext,lnext,_,_=decoded[-3]
    if lens.name!='IL1':raise ValueError('Unexpected provisional lens order.')
    provisional=np.array([np.log(float(length)),np.log(float(znext-lnext/2-z-length/2)),robust['selected_relay_q_IL1_IL2_IL3'][0],float(slope)])
    old_physical=np.asarray(reference['best']['physical_coordinates']);p=extend_prefix(old_physical,provisional,calibration)
    old_controls=np.column_stack((reference['training_controls'],np.zeros(len(reference['training_controls']))));old_cov=np.asarray(reference['training_response_covariance'])
    jacobian=jax.jit(jax.jacfwd(model.predict,argnums=0));old_jac=np.asarray(jacobian(jnp.asarray(p),jnp.asarray(old_controls)))
    old_info=fisher(old_jac,old_cov)
    closure=float(np.max(abs(np.asarray(model.predict(p,old_controls))-np.asarray(ThreeProjectorResponse(calibration).predict(old_physical,np.asarray(reference['training_controls']))))))
    analysers=np.unique(np.asarray(reference['training_controls']),axis=0);grid=np.linspace(-.75,.75,61);zero=30
    source_cov=jnp.asarray(pure_source().total_covariance);moment=jax.jit(lambda pp,c:moments(model,pp,c,source_cov))
    scan=np.array([[*a,u] for a in analysers for u in grid]);means,cov=map(np.asarray,moment(jnp.asarray(p),jnp.asarray(scan)))
    pixel=calibration.pixel_m;rms=np.sqrt(np.linalg.eigvalsh(cov))/pixel;marginal=np.sqrt(np.diagonal(cov,axis1=1,axis2=2))/pixel
    capture=np.maximum(0.,1.-np.sum(np.asarray(jax.scipy.special.erfc((125.-abs(means/pixel))/(np.sqrt(2)*marginal[:,None,:]))),axis=2))
    admitted=((rms[:,0]>=.30)&(capture.min(axis=1)>=.99)).reshape(len(analysers),len(grid));candidates=[]
    for a,b in itertools.combinations(range(len(analysers)),2):
        component=connected_component(admitted[a]&admitted[b],zero)
        if component is None:continue
        left,right=component
        if grid[left]>-.049 or grid[right]<.049:continue
        pair=analysers[[a,b]];condition=float(np.linalg.cond(np.asarray(analyser_matrix(p[:12],pair))*np.array([1.,1.,100.,100.])))
        if not np.isfinite(condition) or condition>1e4:continue
        offsets=np.array([grid[left],0.,grid[right]]);controls=np.array([[*a,u] for u in offsets for a in pair])
        _,cc=map(np.asarray,moment(jnp.asarray(p),jnp.asarray(controls)));newcov=approximate_response_covariance(cc,pixel,1e6)
        info=fisher(np.concatenate((old_jac,np.asarray(jacobian(jnp.asarray(p),jnp.asarray(controls))))),block_diagonal([old_cov,*newcov]))
        if info['rank']!=21:continue
        candidates.append(dict(analyser_indices=[a,b],downstream_controls=pair.tolist(),training_offsets=offsets.tolist(),balanced_analyser_condition=condition,
            approximate_information=info,score=max(info['length_gap_standard_errors_percent'][-2:])))
    candidates.sort(key=lambda row:row['score']);print('IL1 connected candidates',len(candidates),flush=True)
    detector=DetectorGeometry(z_m=float(initial['parameters'][1]),mtf_sigma_px=.65);operator=jax.jit(make_gaussian_detector_operator(detector,16))
    factor=jax.jit(lambda pp:gaussian_image_information_factor(pp,detector,quadrature_order=16));validated=[]
    for candidate in candidates[:3]:
        pair=np.asarray(candidate['downstream_controls']);t=np.asarray(candidate['training_offsets']);held=(t[2] if abs(t[2])>=abs(t[0]) else t[0])/2
        offsets=np.sort(np.r_[t,held]);training=~np.isclose(offsets,held,rtol=0,atol=1e-12);controls=np.array([[*a,u] for u in offsets for a in pair])
        mm,cc=map(np.asarray,moment(jnp.asarray(p),jnp.asarray(controls)));response_cov=[];captures=[];ranks=[]
        for group,c in zip(mm,cc):
            pars=[];covs=[]
            for mean in group:
                encoded=encode_gaussian_image_parameters(mean,c,1e6,detector);f=np.asarray(factor(encoded));_,s,v=np.linalg.svd(f,full_matrices=False)
                ranks.append(int(np.sum(s>s[0]*1e-10)));fac=v.T/s;covs.append(fac@fac.T);pars.append(np.asarray(encoded));captures.append(float(np.asarray(operator(mean,c,1e6)).sum()/1e6))
            response_cov.append(signed_response_from_image_fits(pars,covs)[1])
        select=np.repeat(training,2);rcov=np.asarray(response_cov);newjac=np.asarray(jacobian(jnp.asarray(p),jnp.asarray(controls[select])))
        info=fisher(np.concatenate((old_jac,newjac)),block_diagonal([old_cov,*rcov[select]]))
        recovered=reconstruct(np.asarray(model.predict(p,controls[select])).reshape(3,2,2,4),block_diagonal(list(rcov[select])),p[:12],
            np.asarray(reference['information']['physical_covariance'])[:12,:12],pair,calibration.input_steps_grouped,pixel)
        width=np.sqrt(np.linalg.eigvalsh(cc))/pixel
        row=dict(candidate,il1_native_offsets=offsets.tolist(),training_mask=training.tolist(),heldout_mask=(~training).tolist(),
            exact_finite_pixel_information=info,minimum_capture=min(captures),minimum_image_information_rank=min(ranks),minimum_optical_rms_pixels=float(width.min()),
            maximum_optical_rms_pixels=float(width.max()),response_covariance=rcov.tolist(),initialization_reconstruction=recovered,
            admitted=bool(info['rank']==21 and min(ranks)==6 and min(captures)>=.99 and width.min()>=.30),
            proposed_setting_count=8,proposed_frame_count=72,training_frame_count=54,heldout_frame_count=18)
        validated.append(row);print('IL1 exact',row['analyser_indices'],info.get('length_gap_standard_errors_percent'),flush=True)
    selected=min((row for row in validated if row['admitted']),key=lambda row:max(row['exact_finite_pixel_information']['length_gap_standard_errors_percent'][-2:]),default=None)
    for path in (Path(__file__),reference_path,manifest_path,init_path,robust_path,config_path,geometry_path):sources[str(path.relative_to(ROOT))]=digest(path)
    if any(digest(ROOT/name)!=h for name,h in sources.items()):raise RuntimeError('A source changed during design.')
    report=dict(schema='neoarm-IL1-extension-design-v1',source_hashes=sources,source_hashes_unchanged_during_execution=True,backend=jax.default_backend(),
        elapsed_seconds=time.perf_counter()-started,parameter_names=PARAMETER_NAMES,provisional_IL1_coordinates=provisional.tolist(),provisional_parameters=p.tolist(),
        provisional_IL1_is_not_calibrated=True,original_response_coordinate_count=272,original_information=old_info,nominal_response_closure_max_pixels=closure,
        connected_control_grid=grid.tolist(),downstream_candidate_controls=analysers.tolist(),candidate_pair_count=len(candidates),candidate_shortlist=candidates[:10],
        validated_candidates=validated,selected=selected,all_twenty_one_parameters_free=True,calibration_posterior_added_as_prior=False,
        new_images_generated=False,optical_recovery_demonstrated=False,heldout_responses_used=False,hardware_commands_admitted=False,
        limitations=['IL1 geometry/excitation are provisional design values; the noisy fitted nominal prefix is preserved exactly.',
            'Old272 response coordinates enter once; all21 coordinates including all downstream calibration parameters are released.',
            'The shared12-coordinate downstream calibration covariance is used only to diagnose initialization, not appended as a second prior.',
            'Connected admission is a61-point grid check, not a continuous box or hardware-safe range certificate.',
            'A proposed fresh zero-offset pair avoids reusing old calibration noise in initial map reconstruction.',
            'Finite-pixel Fisher conditions on a pure2nm/1.5nm canonical beam, stable alignment and exact sample/input/detector calibration.',
            'One heldout offset is excluded from the training score. No nonlinear recovery or global branch claim is made.'])
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=audit(args.output);print('Saved',args.output,'admitted',result['selected'] is not None,flush=True)
