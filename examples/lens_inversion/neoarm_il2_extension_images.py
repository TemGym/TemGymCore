"""Fresh native-column GPU images for the frozen eight-setting IL2 design."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
import tomllib

os.environ.setdefault('JAX_ENABLE_X64','true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
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

ROOT=Path(__file__).resolve().parents[2]
FOLDER=ROOT/'examples/generated/neoarm_inversion/imaging_recipe'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def public_arrays(arrays):
    result=inference_arrays(arrays)
    for key in ('downstream_controls','il2_controls'):
        result[key]=np.asarray(arrays[key])
    return result


def acquire(output):
    if jax.default_backend()!='gpu' or not jax.config.x64_enabled:raise RuntimeError('Use GPU float64 for the forward acquisition.')
    started=time.perf_counter();output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    design_path=FOLDER/'il2_extension_design.json';design=json.loads(design_path.read_text());chosen=design['selected']
    if chosen is None or not chosen['admitted']:raise ValueError('The frozen design is not admitted.')
    sources=dict(design['source_hashes'])
    for name,h in sources.items():
        if digest(ROOT/name)!=h:raise ValueError(f'Stale acquisition design:{name}')
    old_truth_path=FOLDER/'pl1_il3_images_truth.json';truth_old=json.loads(old_truth_path.read_text())
    config_path=ROOT/'examples/lens_inversion/neoarm_inversion_config.toml'
    geometry_path=ROOT/'examples/microscope_models/measure_microscope_geometry.ipynb'
    robust_path=FOLDER/'last_lens_robust_design.json';robust=json.loads(robust_path.read_text())
    column=column_with_geometry_anchors(hard_edge_column_from_config(tomllib.loads(config_path.read_text())),geometry_path)
    parameters=jnp.asarray(truth_old['physical_column_parameters']);kappa=truth_old['generating_sample_magnetic_shear_per_m']
    relay_model,decoded=make_relay_model(column,parameters,kappa);relay=np.asarray(robust['selected_relay_q_IL1_IL2_IL3'])
    lens2,z2,length2,bias2,slope2=decoded[-3];lens3,z3,length3,bias3,slope3=decoded[-2];lens1,z1,length1,bias1,slope1=decoded[-1]
    length2,length3,length1=map(float,(length2,length3,length1))
    gap2=float(z3-length3/2-z2-length2/2);gap3=float(z1-length1/2-z3-length3/2);gap1=float(parameters[1]-z1-length1/2)
    base_canonical=np.asarray(relay_model(jnp.asarray(relay))[0])
    prefix3=np.linalg.solve(np.asarray(drift_matrix(gap3)@hard_edge_solenoid_matrix(relay[2],length3)),base_canonical)
    prefix2=np.linalg.solve(np.asarray(drift_matrix(gap2)@hard_edge_solenoid_matrix(relay[1],length2)),prefix3)
    downstream=np.asarray(chosen['downstream_controls']);offsets=np.asarray(chosen['il2_native_offsets'])
    training=np.asarray(chosen['training_mask'],bool);heldout=np.asarray(chosen['heldout_mask'],bool)
    controls=np.array([[*a,u] for u in offsets for a in downstream])
    commands=np.tile(robust['synthetic_fixed_relay_dacs'],(len(controls),1))
    commands[:,lens2.channel_index]+=controls[:,2];commands[:,lens3.channel_index]+=controls[:,1]
    last=truth_old['PL1'];q1=last['q_bias']+last['q_slope_per_control']*controls[:,0]
    commands[:,lens1.channel_index]=lens1.reference_dac+(q1-float(bias1))/float(slope1)
    matrices=np.asarray(jax.jit(jax.vmap(lambda u:single_ray_transfer(column,parameters,u)))(jnp.asarray(commands)))
    def staged(c):
        qlast=last['q_bias']+last['q_slope_per_control']*c[0]
        return (drift_matrix(gap1)@hard_edge_solenoid_matrix(qlast,length1)
            @drift_matrix(gap3)@hard_edge_solenoid_matrix(relay[2]+float(slope3)*c[1],length3)
            @drift_matrix(gap2)@hard_edge_solenoid_matrix(relay[1]+float(slope2)*c[2],length2)@jnp.asarray(prefix2))
    independent=np.asarray(jax.jit(jax.vmap(staged))(jnp.asarray(controls)))
    closure=float(np.max(abs(matrices-independent))/np.max(abs(matrices)))
    if closure>1e-12:raise ValueError('Native controls and staged physical maps disagree.')
    manifest_path=FOLDER/'projector_branch_experiment_inference.json';manifest=json.loads(manifest_path.read_text())
    steps=np.asarray(manifest['input_steps_grouped']);pixel=manifest['detector_pixel_m'];input_states=signed_inputs(steps)
    shear=np.asarray(mechanical_shear(kappa))[np.ix_(GROUPED,GROUPED)]
    canonical_means=input_states[:,GROUPED]@shear.T;source=pure_source();spatial=matrices[:,[0,2]]
    means=np.einsum('nai,fi->nfa',spatial,canonical_means)
    covariance=np.einsum('nai,ij,nbj->nab',spatial,source.total_covariance,spatial)
    widths=np.sqrt(np.linalg.eigvalsh(covariance))/pixel
    detector=DetectorGeometry(z_m=float(parameters[1]),mtf_sigma_px=.65);dose=1e6;read_noise=.6
    operator=jax.jit(make_gaussian_detector_operator(detector,16))
    for path in (Path(__file__),design_path,old_truth_path,config_path,geometry_path,robust_path,manifest_path):sources[str(path.relative_to(ROOT))]=digest(path)
    keys=('expected_counts','noisy_counts','fitted_counts','variance_counts','image_parameters',
        'image_parameter_covariances','image_initial_parameters','response','response_covariance')
    arrays={key:[] for key in keys};rows=[]
    for state,(mm,cc) in enumerate(zip(means,covariance)):
        tick=time.perf_counter();images=[[],[],[],[]];pars=[];covs=[];initials=[];frames=[]
        for frame,mean in enumerate(mm):
            expected=np.asarray(operator(mean,cc,dose));variance=expected+read_noise**2;seed=1100000+100*state+frame
            a,b=jax.random.split(jax.random.PRNGKey(seed))
            observed=np.asarray(jax.random.poisson(a,jnp.asarray(expected),shape=expected.shape)+read_noise*jax.random.normal(b,expected.shape))
            initial=image_initializer(observed,detector,read_noise)
            fit=fit_gaussian_image(observed,variance,detector,initial_parameters=initial,quadrature_order=16,max_nfev=100)
            failure=None
            try:fit.require_accepted()
            except ValueError as error:failure=str(error)
            capture=float(expected.sum()/dose)
            frames.append(dict(frame=frame,noise_seed=seed,admitted=bool(failure is None and widths[state].min()>=.25 and capture>=.99),
                failure=failure,capture_fraction=capture,optimizer_success=bool(fit.optimizer_result.success),nfev=fit.optimizer_result.nfev,
                image_information_rank=fit.numerical_rank,residual_rms_sigma=fit.residual_rms_sigma,chi_square_z=fit.chi_square_z,
                quadrature_rms_sigma=fit.quadrature_rms_sigma,quadrature_parameter_shift_sigma=fit.quadrature_parameter_shift_sigma))
            for destination,value in zip(images,(expected,observed,fit.fitted_counts_yx,variance)):destination.append(value)
            pars.append(fit.parameters);covs.append(fit.parameter_covariance);initials.append(initial)
        response,response_cov=signed_response_from_image_fits(pars,covs)
        for key,value in zip(keys,(*images,pars,covs,initials,response,response_cov)):arrays[key].append(value)
        group,analyser=divmod(state,2)
        rows.append(dict(group_index=group,analyser_index=analyser,controls=controls[state].tolist(),
            training=bool(training[group]),held_out=bool(heldout[group]),optical_rms_principal_pixels=widths[state].tolist(),
            frames=frames,admitted=all(f['admitted'] for f in frames),elapsed_seconds=time.perf_counter()-tick))
        print('IL2 images',state,'admitted',rows[-1]['admitted'],flush=True)
    arrays={key:np.asarray(value).reshape((4,2)+np.asarray(value).shape[1:]) for key,value in arrays.items()}
    paired,joint=assemble_response_covariance(arrays['response_covariance'])
    arrays.update(paired_response_covariance=paired,joint_response_covariance=joint,downstream_controls=downstream,
        il2_controls=offsets,training_mask=training,heldout_mask=heldout,raw_dac_values=commands.reshape(4,2,7),
        input_steps_grouped=steps,input_states_grouped=input_states,detector_pixel_m=np.asarray(pixel))
    expected_response=((spatial@shear)[:,:,GROUPED]*steps/pixel).reshape(4,2,2,4)
    likelihood={name:truth_response_likelihood(arrays['response'][mask].reshape(-1,2,4),expected_response[mask].reshape(-1,2,4),
        arrays['response_covariance'][mask].reshape(-1,8,8)) for name,mask in [('training',training),('heldout',heldout)]}
    checks=[]
    for state in sorted(set((int(np.argmin(widths[:,0])),int(np.argmax(widths[:,1]))))):
        group,analyser=divmod(state,2);frame=int(np.argmax(np.linalg.norm(means[state],axis=1)))
        mode=MixedGaussianSource(source.position_covariance_xy_m2,source.curvature_xy_per_m,source.centre_covariance,canonical_means[frame])
        expected=arrays['expected_counts'][group,analyser,frame];sigma=np.sqrt(arrays['variance_counts'][group,analyser,frame])
        renders={};refinements=[];previous=None
        for order in (8,16,32):
            image,error=render_pure_counts(mode,matrices[state],detector,pixel_order=order,incident_electrons=dose)
            renders[order]=np.asarray(image);transition=None if previous is None else float(np.linalg.norm((renders[order]-previous)/sigma))
            refinements.append(dict(pixel_quadrature_order=order,image_whitened_l2=float(np.linalg.norm((renders[order]-expected)/sigma)),transition_whitened_l2=transition))
            previous=renders[order]
        path=output.with_name(output.stem+f'_gaussian_check_{state}.npz')
        np.savez_compressed(path,gaussian_pixel8_counts_yx=renders[8],gaussian_pixel16_counts_yx=renders[16],gaussian_pixel32_counts_yx=renders[32],
            covariance_counts_yx=expected,noise_sigma_yx=sigma,native_transfer_matrix=matrices[state])
        checks.append(dict(state_index=state,frame=frame,pure_gaussian_mode_count=1,balanced_symplectic_error=error,
            common_axial_phase_path_m=0.,beam_dependent_actions_retained=True,refinements=refinements,
            passed=all(r['image_whitened_l2']<.2 for r in refinements) and all(r['transition_whitened_l2']<.2 for r in refinements[1:]),
            arrays_path=str(path.relative_to(ROOT)),arrays_sha256=digest(path)))
    arrays_path=output.with_suffix('.npz');np.savez_compressed(arrays_path,**arrays)
    truth_path=output.with_name(output.stem+'_truth.json')
    truth=dict(schema='neoarm-three-projector-generating-truth-v1',source_hashes=sources,
        purpose='Postfit diagnostics only; no optical initializer reads generating values.',
        PL1=dict(length_m=length1,gap_m=gap1,q_bias=last['q_bias'],q_slope=last['q_slope_per_control']),
        IL3=dict(length_m=length3,gap_m=gap3,q_bias=float(relay[2]),q_slope=float(slope3)),
        IL2=dict(length_m=length2,gap_m=gap2,q_bias=float(relay[1]),q_slope=float(slope2)),
        upstream_to_IL2_canonical=prefix2.tolist(),upstream_to_IL2_mechanical=(prefix2[np.ix_(GROUPED,GROUPED)]@np.asarray(mechanical_shear(kappa))).tolist(),
        expected_response_pixels=expected_response.tolist(),sample_kappa_per_m=kappa,physical_column_parameters=np.asarray(parameters).tolist())
    truth_path.write_text(json.dumps(truth,indent=2,allow_nan=False)+'\n')
    admitted=all(row['admitted'] for row in rows) and all(c['passed'] for c in checks)
    report=dict(schema='neoarm-IL2-eight-setting-images-v1',backend=jax.default_backend(),source_hashes=sources,
        design_path=str(design_path.relative_to(ROOT)),design_sha256=digest(design_path),
        arrays_path=str(arrays_path.relative_to(ROOT)),arrays_sha256=digest(arrays_path),truth_path=str(truth_path.relative_to(ROOT)),truth_sha256=digest(truth_path),
        input_steps_grouped=steps.tolist(),detector_pixel_m=pixel,downstream_controls=downstream.tolist(),il2_controls=offsets.tolist(),
        generated_frame_count=72,training_frame_count=54,heldout_frame_count=18,incident_electrons_per_frame=dose,read_noise_electrons=read_noise,
        acquisition_groups=rows,truth_likelihood_diagnostics=likelihood,independent_gaussian_checks=checks,
        native_column_staged_map_relative_error=closure,all_frame_admissions_passed=all(row['admitted'] for row in rows),paired_data_admitted=admitted,
        calibration_response_noise_independent=True,design_changed_after_new_data=False,optical_parameters_fitted=False,hardware_commands_admitted=False,
        elapsed_seconds=time.perf_counter()-started,
        limitations=['Fresh zero-offset data are independent exposures, not recycled old calibration responses.',
            'The positive.375 offset is withheld from optical initialization/training.',
            'All6 image coordinates are fit from data-only starts; exact synthetic fixed WLS variance remains conditional.',
            'Quantum-valid pure source, stable alignment, exact input/pixel/PSF calibration and independent-half hard-edge objective remain assumptions.',
            'No instrument-safe DAC/dose authorization, model-shape validation or optical recovery is asserted.'])
    if any(digest(ROOT/name)!=h for name,h in sources.items()):raise RuntimeError('A pinned source changed during acquisition.')
    report['source_hashes_unchanged_during_execution']=True;output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if admitted:
        public=public_arrays(arrays);public_path=output.with_name(output.stem+'_inference.npz');np.savez_compressed(public_path,**public)
        manifest_path=output.with_name(output.stem+'_inference.json');public_sources=dict(sources)
        public_sources[str(output.relative_to(ROOT))]=digest(output);public_sources[str(arrays_path.relative_to(ROOT))]=digest(arrays_path)
        public_report=dict(schema='neoarm-three-projector-inference-v1',backend=jax.default_backend(),source_hashes=public_sources,
            arrays_path=str(public_path.relative_to(ROOT)),arrays_sha256=digest(public_path),public_array_keys=sorted(public),
            truth_path=str(truth_path.relative_to(ROOT)),truth_sha256=digest(truth_path),truth_use='postfit only',
            input_steps_grouped=steps.tolist(),detector_pixel_m=pixel,paired_data_admitted=True,calibration_response_noise_independent=True,
            simulation_audit_path=str(output.relative_to(ROOT)),simulation_audit_sha256=digest(output),
            generating_truth_fields_in_public_arrays=False,independent_gaussian_checks=[dict(passed=True,scope='Details pinned in simulation audit')])
        manifest_path.write_text(json.dumps(public_report,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=acquire(args.output);print('Saved',args.output,'admitted',result['paired_data_admitted'],flush=True)
