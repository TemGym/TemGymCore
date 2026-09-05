"""Fresh native-column images for the frozen four-projector IL1 experiment."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
import tomllib

os.environ.setdefault('JAX_ENABLE_X64','true');os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import jax
import jax.numpy as jnp
import numpy as np
from neoarm_projector_branch_experiment import pure_source,signed_inputs
from neoarm_last_lens_images import image_initializer,signed_response_from_image_fits,render_pure_counts
from neoarm_last_lens_redesigned_images import truth_response_likelihood
from neoarm_last_lens_design import make_relay_model,GROUPED
from neoarm_pl1_il3_images import assemble_response_covariance
from neoarm_mixed_gaussian_validation import MixedGaussianSource
from neoarm_last_lens_tomography import mechanical_shear
from neoarm_imaging_recipe import column_with_geometry_anchors,single_ray_transfer
from neoarm_inference_manifest import inference_arrays
from temgym_core.gaussian_image_likelihood import make_gaussian_detector_operator,fit_gaussian_image
from temgym_core.microscope_inversion import drift_matrix,hard_edge_solenoid_matrix
from temgym_core.neoarm_calibration import DetectorGeometry,hard_edge_column_from_config

ROOT=Path(__file__).resolve().parents[2];FOLDER=ROOT/'examples/generated/neoarm_inversion/imaging_recipe'
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def public_arrays(arrays):
    result=inference_arrays(arrays)
    for key in ('downstream_controls','il1_controls'):result[key]=np.asarray(arrays[key])
    return result


def acquire(output):
    if jax.default_backend()!='gpu' or not jax.config.x64_enabled:raise RuntimeError('Native image generation requires GPU float64.')
    started=time.perf_counter();output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    design_path=FOLDER/'il1_extension_design.json';design=json.loads(design_path.read_text());selected=design['selected']
    if selected is None or not selected['admitted']:raise ValueError('An admitted frozen design is required.')
    sources=dict(design['source_hashes'])
    for name,h in sources.items():
        if digest(ROOT/name)!=h:raise ValueError(f'Stale frozen acquisition source:{name}')
    old_truth_path=FOLDER/'pl1_il3_images_truth.json';old_truth=json.loads(old_truth_path.read_text())
    robust_path=FOLDER/'last_lens_robust_design.json';robust=json.loads(robust_path.read_text())
    config_path=ROOT/'examples/lens_inversion/neoarm_inversion_config.toml';geometry_path=ROOT/'examples/microscope_models/measure_microscope_geometry.ipynb'
    column=column_with_geometry_anchors(hard_edge_column_from_config(tomllib.loads(config_path.read_text())),geometry_path)
    parameters=jnp.asarray(old_truth['physical_column_parameters']);kappa=old_truth['generating_sample_magnetic_shear_per_m']
    relay_model,decoded=make_relay_model(column,parameters,kappa);relay=np.asarray(robust['selected_relay_q_IL1_IL2_IL3'])
    stages={};names=('PL1','IL3','IL2','IL1')
    for i,name in enumerate(names):
        lens,z,length,bias,slope=decoded[-1-i];length=float(length)
        next_entrance=float(parameters[1]) if i==0 else float(decoded[-i][1]-decoded[-i][2]/2)
        stages[name]=dict(length_m=length,gap_m=next_entrance-float(z)-length/2,
            q_bias=old_truth['PL1']['q_bias'] if i==0 else float(relay[3-i]),
            q_slope=old_truth['PL1']['q_slope_per_control'] if i==0 else float(slope))
    prefix=np.asarray(relay_model(jnp.asarray(relay))[0])
    for name in ('IL3','IL2','IL1'):
        lens=stages[name];prefix=np.linalg.solve(np.asarray(drift_matrix(lens['gap_m'])@hard_edge_solenoid_matrix(lens['q_bias'],lens['length_m'])),prefix)
    pair=np.asarray(selected['downstream_controls']);offsets=np.asarray(selected['il1_native_offsets'])
    training=np.asarray(selected['training_mask'],bool);heldout=np.asarray(selected['heldout_mask'],bool)
    controls=np.array([[*a,u] for u in offsets for a in pair]);commands=np.tile(robust['synthetic_fixed_relay_dacs'],(len(controls),1))
    for i in (1,2,3):commands[:,decoded[-1-i][0].channel_index]+=controls[:,i]
    lens,_,_,bias,slope=decoded[-1];commands[:,lens.channel_index]=lens.reference_dac+(stages['PL1']['q_bias']+stages['PL1']['q_slope']*controls[:,0]-float(bias))/float(slope)
    matrices=np.asarray(jax.jit(jax.vmap(lambda u:single_ray_transfer(column,parameters,u)))(jnp.asarray(commands)))
    def staged(c):
        matrix=jnp.asarray(prefix)
        for i in (3,2,1,0):
            v=stages[names[i]];matrix=drift_matrix(v['gap_m'])@hard_edge_solenoid_matrix(v['q_bias']+v['q_slope']*c[i],v['length_m'])@matrix
        return matrix
    independent=np.asarray(jax.jit(jax.vmap(staged))(jnp.asarray(controls)));closure=float(np.max(abs(matrices-independent))/np.max(abs(matrices)))
    if closure>1e-12:raise ValueError('Native full-column and independently composed four-stage maps disagree.')
    input_path=FOLDER/'il2_extension_images_inference.json';calibration=json.loads(input_path.read_text());steps=np.asarray(calibration['input_steps_grouped']);pixel=calibration['detector_pixel_m']
    input_states=signed_inputs(steps);shear=np.asarray(mechanical_shear(kappa))[np.ix_(GROUPED,GROUPED)]
    canonical_means=input_states[:,GROUPED]@shear.T;source=pure_source();spatial=matrices[:,[0,2]]
    means=np.einsum('nai,fi->nfa',spatial,canonical_means);covariance=np.einsum('nai,ij,nbj->nab',spatial,source.total_covariance,spatial)
    widths=np.sqrt(np.linalg.eigvalsh(covariance))/pixel;detector=DetectorGeometry(z_m=float(parameters[1]),mtf_sigma_px=.65)
    operator=jax.jit(make_gaussian_detector_operator(detector,16));dose=1e6;read_noise=.6
    for path in (Path(__file__),design_path,old_truth_path,robust_path,config_path,geometry_path,input_path):sources[str(path.relative_to(ROOT))]=digest(path)
    keys=('expected_counts','noisy_counts','fitted_counts','variance_counts','image_parameters','image_parameter_covariances','image_initial_parameters','response','response_covariance')
    arrays={key:[] for key in keys};rows=[]
    for state,(mm,cc) in enumerate(zip(means,covariance)):
        tick=time.perf_counter();images=[[],[],[],[]];pars=[];covs=[];initials=[];frames=[]
        for frame,mean in enumerate(mm):
            expected=np.asarray(operator(mean,cc,dose));variance=expected+read_noise**2;seed=1160000+100*state+frame
            a,b=jax.random.split(jax.random.PRNGKey(seed));observed=np.asarray(jax.random.poisson(a,jnp.asarray(expected),shape=expected.shape)+read_noise*jax.random.normal(b,expected.shape))
            initial=image_initializer(observed,detector,read_noise);fit=fit_gaussian_image(observed,variance,detector,initial_parameters=initial,quadrature_order=16,max_nfev=100)
            failure=None
            try:fit.require_accepted()
            except ValueError as error:failure=str(error)
            capture=float(expected.sum()/dose);frames.append(dict(frame=frame,noise_seed=seed,failure=failure,
                admitted=bool(failure is None and widths[state].min()>=.25 and capture>=.99),capture_fraction=capture,
                optimizer_success=bool(fit.optimizer_result.success),nfev=fit.optimizer_result.nfev,image_information_rank=fit.numerical_rank,
                residual_rms_sigma=fit.residual_rms_sigma,chi_square_z=fit.chi_square_z,
                quadrature_rms_sigma=fit.quadrature_rms_sigma,quadrature_parameter_shift_sigma=fit.quadrature_parameter_shift_sigma))
            for destination,value in zip(images,(expected,observed,fit.fitted_counts_yx,variance)):destination.append(value)
            pars.append(fit.parameters);covs.append(fit.parameter_covariance);initials.append(initial)
        response,response_cov=signed_response_from_image_fits(pars,covs)
        for key,value in zip(keys,(*images,pars,covs,initials,response,response_cov)):arrays[key].append(value)
        group,analyser=divmod(state,2);rows.append(dict(group_index=group,analyser_index=analyser,controls=controls[state].tolist(),training=bool(training[group]),held_out=bool(heldout[group]),
            optical_rms_principal_pixels=widths[state].tolist(),frames=frames,admitted=all(f['admitted'] for f in frames),elapsed_seconds=time.perf_counter()-tick))
        print('IL1 acquisition',state,'admitted',rows[-1]['admitted'],flush=True)
    arrays={key:np.asarray(value).reshape((4,2)+np.asarray(value).shape[1:]) for key,value in arrays.items()};paired,joint=assemble_response_covariance(arrays['response_covariance'])
    arrays.update(paired_response_covariance=paired,joint_response_covariance=joint,downstream_controls=pair,il1_controls=offsets,
        training_mask=training,heldout_mask=heldout,raw_dac_values=commands.reshape(4,2,7),input_steps_grouped=steps,input_states_grouped=input_states,detector_pixel_m=np.asarray(pixel))
    expected_response=((spatial@shear)[:,:,GROUPED]*steps/pixel).reshape(4,2,2,4)
    likelihood={name:truth_response_likelihood(arrays['response'][mask].reshape(-1,2,4),expected_response[mask].reshape(-1,2,4),arrays['response_covariance'][mask].reshape(-1,8,8))
        for name,mask in [('training',training),('heldout',heldout)]};checks=[]
    for state in sorted(set((int(np.argmin(widths[:,0])),int(np.argmax(widths[:,1]))))):
        group,analyser=divmod(state,2);frame=int(np.argmax(np.linalg.norm(means[state],axis=1)))
        mode=MixedGaussianSource(source.position_covariance_xy_m2,source.curvature_xy_per_m,source.centre_covariance,canonical_means[frame])
        expected=arrays['expected_counts'][group,analyser,frame];sigma=np.sqrt(arrays['variance_counts'][group,analyser,frame]);renders={};refinements=[];previous=None
        for order in (8,16,32):
            image,error=render_pure_counts(mode,matrices[state],detector,pixel_order=order,incident_electrons=dose);renders[order]=np.asarray(image)
            transition=None if previous is None else float(np.linalg.norm((renders[order]-previous)/sigma));refinements.append(dict(pixel_quadrature_order=order,
                image_whitened_l2=float(np.linalg.norm((renders[order]-expected)/sigma)),transition_whitened_l2=transition));previous=renders[order]
        path=output.with_name(output.stem+f'_gaussian_check_{state}.npz');np.savez_compressed(path,gaussian_pixel8_counts_yx=renders[8],gaussian_pixel16_counts_yx=renders[16],
            gaussian_pixel32_counts_yx=renders[32],covariance_counts_yx=expected,noise_sigma_yx=sigma,native_transfer_matrix=matrices[state])
        checks.append(dict(state_index=state,frame=frame,pure_gaussian_mode_count=1,balanced_symplectic_error=error,common_axial_phase_path_m=0.,
            beam_dependent_actions_retained=True,refinements=refinements,passed=all(r['image_whitened_l2']<.2 for r in refinements) and all(r['transition_whitened_l2']<.2 for r in refinements[1:]),
            arrays_path=str(path.relative_to(ROOT)),arrays_sha256=digest(path)))
    arrays_path=output.with_suffix('.npz');np.savez_compressed(arrays_path,**arrays);truth_path=output.with_name(output.stem+'_truth.json')
    truth=dict(schema='neoarm-four-projector-generating-truth-v1',source_hashes=sources,**stages,
        purpose='Postfit diagnostics only; no generating IL1 geometry or excitation enters optical initialization.',
        upstream_to_IL1_canonical=prefix.tolist(),upstream_to_IL1_mechanical=(prefix[np.ix_(GROUPED,GROUPED)]@np.asarray(mechanical_shear(kappa))).tolist(),
        expected_response_pixels=expected_response.tolist(),sample_kappa_per_m=kappa,physical_column_parameters=np.asarray(parameters).tolist())
    truth_path.write_text(json.dumps(truth,indent=2,allow_nan=False)+'\n');admitted=all(r['admitted'] for r in rows) and all(r['passed'] for r in checks)
    report=dict(schema='neoarm-IL1-eight-setting-images-v1',backend=jax.default_backend(),source_hashes=sources,design_path=str(design_path.relative_to(ROOT)),design_sha256=digest(design_path),
        arrays_path=str(arrays_path.relative_to(ROOT)),arrays_sha256=digest(arrays_path),truth_path=str(truth_path.relative_to(ROOT)),truth_sha256=digest(truth_path),
        input_steps_grouped=steps.tolist(),detector_pixel_m=pixel,downstream_controls=pair.tolist(),il1_controls=offsets.tolist(),generated_frame_count=72,
        training_frame_count=54,heldout_frame_count=18,incident_electrons_per_frame=dose,read_noise_electrons=read_noise,acquisition_groups=rows,
        truth_likelihood_diagnostics=likelihood,independent_gaussian_checks=checks,native_column_staged_map_relative_error=closure,paired_data_admitted=admitted,
        calibration_response_noise_independent=True,design_changed_after_new_data=False,optical_parameters_fitted=False,hardware_commands_admitted=False,elapsed_seconds=time.perf_counter()-started,
        limitations=['All72 exposures are new independent Poisson/read-noise draws; the fresh zero-offset pair does not recycle calibration images.',
            'The+.375 IL1 offset is withheld. Optical parameters are not fitted in this acquisition script.',
            'Pure source, stable alignment, exact input/pixel/PSF calibration and exact synthetic WLS variance remain conditional.',
            'Native hard-edge column propagation is independently checked at narrow/wide states with two successive detector-pixel refinements.',
            'No hardware-safe DAC range or one-million-electron retained-frame feasibility is asserted.'])
    if any(digest(ROOT/name)!=h for name,h in sources.items()):raise RuntimeError('A source changed during acquisition.')
    report['source_hashes_unchanged_during_execution']=True;output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if admitted:
        public=public_arrays(arrays);public_path=output.with_name(output.stem+'_inference.npz');np.savez_compressed(public_path,**public)
        manifest_path=output.with_name(output.stem+'_inference.json');public_sources=dict(sources)
        public_sources[str(output.relative_to(ROOT))]=digest(output);public_sources[str(arrays_path.relative_to(ROOT))]=digest(arrays_path)
        manifest=dict(schema='neoarm-four-projector-inference-v1',backend=jax.default_backend(),source_hashes=public_sources,
            arrays_path=str(public_path.relative_to(ROOT)),arrays_sha256=digest(public_path),public_array_keys=sorted(public),truth_path=str(truth_path.relative_to(ROOT)),truth_sha256=digest(truth_path),truth_use='postfit only',
            input_steps_grouped=steps.tolist(),detector_pixel_m=pixel,paired_data_admitted=True,calibration_response_noise_independent=True,
            simulation_audit_path=str(output.relative_to(ROOT)),simulation_audit_sha256=digest(output),generating_truth_fields_in_public_arrays=False,
            independent_gaussian_checks=[dict(passed=True,scope='Independent native-column Gaussian propagation and pixel refinement; details pinned in simulation audit')])
        manifest_path.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=acquire(args.output);print('Saved',args.output,'admitted',result['paired_data_admitted'],flush=True)
