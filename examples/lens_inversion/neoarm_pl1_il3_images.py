"""Fresh paired PL1-analyser images across a separate IL3 control scan.

Five IL3 offsets use two fixed PL1 analyser states. All nine signed specimen
inputs are new exposures independent of the PL1 calibration images. Only
the inner three IL3 offsets are training; the two endpoints are held out.
Optical parameters are not fitted here. Failures remain in the saved data.
"""
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

from neoarm_last_lens_design import make_relay_model,GROUPED
from neoarm_last_lens_images import image_initializer,signed_response_from_image_fits,render_pure_counts
from neoarm_last_lens_redesigned_images import normalized_excitation,truth_response_likelihood
from neoarm_last_lens_tomography import mechanical_shear
from neoarm_mixed_gaussian_validation import MixedGaussianSource
from neoarm_imaging_recipe import column_with_geometry_anchors,single_ray_transfer
from temgym_core.gaussian_image_likelihood import make_gaussian_detector_operator,fit_gaussian_image
from temgym_core.microscope_inversion import drift_matrix,hard_edge_solenoid_matrix
from temgym_core.neoarm_calibration import DetectorGeometry,hard_edge_column_from_config


def assemble_response_covariance(per_state_covariance):
    """Independent exposures; keep each response's full correlated8x8 block."""
    covariance=np.asarray(per_state_covariance)
    if covariance.ndim!=4 or covariance.shape[1:]!=(2,8,8):
        raise ValueError('Expected covariance shape (N,2,8,8).')
    paired=np.zeros((len(covariance),16,16))
    paired[:,:8,:8]=covariance[:,0];paired[:,8:,8:]=covariance[:,1]
    joint=np.zeros((16*len(covariance),16*len(covariance)))
    for i,c in enumerate(paired):joint[16*i:16*(i+1),16*i:16*(i+1)]=c
    return paired,joint


def training_heldout_masks(offsets):
    offsets=np.asarray(offsets,float)
    if offsets.shape!=(5,) or not np.allclose(offsets,[-.1,-.05,0.,.05,.1],rtol=0,atol=1e-14):
        raise ValueError('This frozen pilot uses precisely five declared IL3 offsets.')
    training=np.array([False,True,True,True,False])
    return training,~training


def audit(output):
    if jax.default_backend()!='gpu':raise RuntimeError('Paired forward images require GPU.')
    root=Path(__file__).resolve().parents[2];output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    calibration_path=root/'examples/generated/neoarm_inversion/imaging_recipe/last_lens_redesigned_images.json'
    calibration=json.loads(calibration_path.read_text());digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    sources=dict(calibration['source_hashes'])
    for path in (Path(__file__),calibration_path,root/calibration['arrays_path'],root/calibration['truth_path']):
        sources[str(path.relative_to(root))]=digest(path)
    for name,value in sources.items():
        if digest(root/name)!=value:raise ValueError(f'Stale paired-acquisition dependency:{name}')
    if not calibration['image_generation_admitted']:raise ValueError('The independent PL1 calibration image block is not admitted.')
    design=json.loads((root/calibration['design_report_path']).read_text())
    init=json.loads((root/'examples/generated/neoarm_inversion/imaging_recipe/measured_initialization.json').read_text())
    config=tomllib.loads((root/'examples/lens_inversion/neoarm_inversion_config.toml').read_text())
    column=column_with_geometry_anchors(hard_edge_column_from_config(config),
        root/'examples/microscope_models/measure_microscope_geometry.ipynb')
    parameters=jnp.asarray(init['parameters']);model,decoded=make_relay_model(column,parameters,800.)
    relay=np.asarray(design['selected_relay_q_IL1_IL2_IL3']);base_canonical,base_mechanical=map(np.asarray,model(relay))
    lens3,z3,length3,bias3,slope3=decoded[-2];last,z_last,length_last,bias_last,slope_last=decoded[-1]
    length3,length_last=float(length3),float(length_last)
    gap3=float(z_last-length_last/2-z3-length3/2);gap_last=float(parameters[1]-z_last-length_last/2)
    prefix=np.linalg.solve(np.asarray(drift_matrix(gap3)@hard_edge_solenoid_matrix(relay[2],length3)),base_canonical)
    prefix_grouped=prefix[np.ix_(GROUPED,GROUPED)]
    prefix_mechanical=prefix_grouped@np.asarray(mechanical_shear(800.))
    offsets=np.array([-.1,-.05,0.,.05,.1]);training,heldout=training_heldout_masks(offsets)
    pl1_indices=np.array([5,22]);pl1_q=np.asarray(design['fixed_pl1_q'])[pl1_indices]
    pl1_controls=np.asarray(normalized_excitation(pl1_q));il3_controls=offsets/float(slope3)
    inputs=np.array([5e-9,5e-9,1e-3,1e-3]);pixel=55e-6;dose=1e6;read_noise=.6
    input_states=np.zeros((9,4))
    for i in range(4):input_states[1+2*i,i]=inputs[i];input_states[2+2*i,i]=-inputs[i]
    shear=np.asarray(mechanical_shear(800.))[np.ix_(GROUPED,GROUPED)]
    canonical_means=input_states[:,GROUPED]@shear.T
    source=MixedGaussianSource(np.diag(np.array([2e-9,1.5e-9])**2),np.zeros((2,2)),np.zeros((4,4)),np.zeros(4))
    detector=DetectorGeometry(z_m=float(parameters[1]),mtf_sigma_px=.65)
    operator=jax.jit(make_gaussian_detector_operator(detector,16))
    matrices=np.asarray(jax.jit(jax.vmap(lambda offset:jax.vmap(lambda q:
        drift_matrix(gap_last)@hard_edge_solenoid_matrix(q,length_last)@model(jnp.asarray(relay).at[2].add(offset))[0]
    )(jnp.asarray(pl1_q))))(jnp.asarray(offsets)))
    spatial=matrices[:,:,[0,2]]
    image_means=np.einsum('nsai,fi->nsfa',spatial,canonical_means)
    image_covariance=np.einsum('nsai,ij,nsbj->nsab',spatial,source.total_covariance,spatial)
    widths=np.sqrt(np.linalg.eigvalsh(image_covariance))/pixel
    commands=np.tile(np.asarray(design['synthetic_fixed_relay_dacs']),(5,2,1))
    commands[:,:,lens3.channel_index]+=il3_controls[:,None]
    commands[:,:,last.channel_index]=last.reference_dac+(pl1_q-float(bias_last))/float(slope_last)
    actual=np.asarray(jax.jit(jax.vmap(lambda u:single_ray_transfer(column,parameters,u)))(jnp.asarray(commands.reshape(-1,7)))).reshape(5,2,4,4)
    closure=float(np.max(np.abs(actual-matrices))/np.max(np.abs(actual)))
    if closure>1e-12:raise ValueError('Actual controls do not reproduce the paired optical maps.')
    arrays={name:[] for name in ('expected_counts','noisy_counts','fitted_counts','variance_counts',
        'image_parameters','image_parameter_covariances','image_initial_parameters','response','response_covariance')}
    rows=[]
    for group in range(5):
        group_arrays={name:[] for name in arrays}
        for analyser in range(2):
            started=time.perf_counter();pars=[];covs=[];initials=[];frames=[];images=[[],[],[],[]]
            preflight=bool(widths[group,analyser].min()>=.25)
            for frame,mean in enumerate(image_means[group,analyser]):
                expected=np.asarray(operator(mean,image_covariance[group,analyser],dose));variance=expected+read_noise**2
                seed=980000+1000*group+100*analyser+frame
                key1,key2=jax.random.split(jax.random.PRNGKey(seed))
                observed=np.asarray(jax.random.poisson(key1,jnp.asarray(expected),shape=expected.shape)
                    +read_noise*jax.random.normal(key2,expected.shape))
                initial=image_initializer(observed,detector,read_noise)
                fit=fit_gaussian_image(observed,variance,detector,initial_parameters=initial,quadrature_order=16,max_nfev=100)
                failure=None
                try:fit.require_accepted()
                except ValueError as error:failure=str(error)
                capture=float(expected.sum()/dose)
                admitted=bool(failure is None and preflight and capture>=.99)
                frames.append(dict(frame=frame,noise_seed=seed,admitted=admitted,failure=failure,
                    capture_fraction=capture,optimizer_success=bool(fit.optimizer_result.success),
                    optimizer_status=fit.optimizer_result.status,nfev=fit.optimizer_result.nfev,
                    image_information_rank=fit.numerical_rank,residual_rms_sigma=fit.residual_rms_sigma,
                    chi_square_z=fit.chi_square_z,quadrature_parameter_shift_sigma=fit.quadrature_parameter_shift_sigma,
                    quadrature_rms_sigma=fit.quadrature_rms_sigma))
                pars.append(fit.parameters);covs.append(fit.parameter_covariance);initials.append(initial)
                for destination,value in zip(images,(expected,observed,fit.fitted_counts_yx,variance)):destination.append(value)
            admitted=all(f['admitted'] for f in frames)
            if np.all(np.isfinite(covs)):
                response,covariance_response=signed_response_from_image_fits(pars,covs)
            else:
                response=np.full((2,4),np.nan);covariance_response=np.full((8,8),np.nan)
            for name,value in zip(('expected_counts','noisy_counts','fitted_counts','variance_counts'),images):group_arrays[name].append(value)
            for name,value in [('image_parameters',pars),('image_parameter_covariances',covs),('image_initial_parameters',initials),
                               ('response',response),('response_covariance',covariance_response)]:group_arrays[name].append(value)
            row=dict(il3_offset_rad=float(offsets[group]),il3_control=float(il3_controls[group]),pl1_q_rad=float(pl1_q[analyser]),
                pl1_control=float(pl1_controls[analyser]),training=bool(training[group]),held_out=bool(heldout[group]),
                optical_rms_principal_pixels=widths[group,analyser].tolist(),preflight_width_admitted=preflight,
                frames=frames,admitted=admitted,elapsed_seconds=time.perf_counter()-started)
            rows.append(row);print(f'IL3{offsets[group]:+.2f},PL1 index{pl1_indices[analyser]}:admitted={admitted},{row["elapsed_seconds"]:.1f}s',flush=True)
        for name in arrays:arrays[name].append(group_arrays[name])
    arrays={name:np.asarray(value) for name,value in arrays.items()}
    paired_cov,joint_cov=assemble_response_covariance(arrays['response_covariance'])
    arrays.update(paired_response_covariance=paired_cov,joint_response_covariance=joint_cov,
        pl1_controls=pl1_controls,il3_controls=il3_controls,il3_q_offsets=offsets,pl1_q=pl1_q,
        training_mask=training,heldout_mask=heldout,raw_dac_values=commands,input_steps_grouped=inputs,
        input_states_grouped=input_states,detector_pixel_m=np.asarray(pixel))
    expected_response=np.einsum('nsai,ij->nsaj',spatial,shear)[:,:,:,GROUPED]*inputs/pixel
    admitted=all(row['admitted'] for row in rows)
    likelihood={}
    if admitted:
        for name,mask in [('training',training),('heldout',heldout)]:
            likelihood[name]=truth_response_likelihood(arrays['response'][mask].reshape(-1,2,4),
                expected_response[mask].reshape(-1,2,4),arrays['response_covariance'][mask].reshape(-1,8,8))
    checks=[]
    for flat in sorted(set([int(np.argmin(widths[:,:,0])),int(np.argmax(widths[:,:,1]))])):
        group,analyser=divmod(flat,2);frame=int(np.argmax(np.linalg.norm(image_means[group,analyser],axis=1)))
        check_source=MixedGaussianSource(source.position_covariance_xy_m2,source.curvature_xy_per_m,source.centre_covariance,canonical_means[frame])
        renders={};refinements=[];previous=None;expected=arrays['expected_counts'][group,analyser,frame]
        noise_sigma=np.sqrt(arrays['variance_counts'][group,analyser,frame])
        for order in (8,16,32):
            image,symplectic_error=render_pure_counts(check_source,matrices[group,analyser],detector,pixel_order=order,incident_electrons=dose)
            renders[order]=np.asarray(image);difference=(renders[order]-expected)/noise_sigma
            transition=None if previous is None else float(np.linalg.norm((renders[order]-previous)/noise_sigma))
            refinements.append(dict(pixel_quadrature_order=order,image_whitened_l2=float(np.linalg.norm(difference)),
                                    transition_whitened_l2=transition));previous=renders[order]
        path=output.with_name(output.stem+f'_gaussian_check_{group}_{analyser}.npz')
        np.savez_compressed(path,gaussian_pixel8_counts_yx=renders[8],gaussian_pixel16_counts_yx=renders[16],
            gaussian_pixel32_counts_yx=renders[32],covariance_counts_yx=expected,noise_sigma_yx=noise_sigma,transfer_matrix=matrices[group,analyser])
        checks.append(dict(group_index=group,analyser_index=analyser,frame=frame,pure_gaussian_mode_count=1,
            balanced_symplectic_error=symplectic_error,common_axial_phase_path_m=0.,beam_dependent_actions_retained=True,
            refinements=refinements,passed=all(r['image_whitened_l2']<.2 for r in refinements)
                and all(r['transition_whitened_l2']<.2 for r in refinements[1:]),
            arrays_path=str(path.relative_to(root)),arrays_sha256=digest(path)))
    arrays_path=output.with_suffix('.npz');np.savez_compressed(arrays_path,**arrays)
    truth_path=output.with_name(output.stem+'_truth.json')
    truth=dict(schema='neoarm-pl1-il3-generating-truth-v1',source_hashes=sources,
        purpose='Read only after optimization for recovery/closure diagnostics.',
        upstream_to_IL3_canonical=prefix.tolist(),upstream_to_IL3_mechanical=prefix_mechanical.tolist(),
        canonical_coordinate_order=['x','theta_x','y','theta_y'],mechanical_coordinate_order=['x','y','theta_x','theta_y'],
        generating_sample_magnetic_shear_per_m=800.,
        IL3=dict(length_m=length3,gap_after_exit_m=gap3,q_bias=float(relay[2]),q_slope_per_control=float(slope3)),
        PL1=dict(length_m=length_last,detector_gap_m=gap_last,q_bias=-np.pi/2,q_slope_per_control=-(np.pi/2-.05)),
        expected_response_pixels=expected_response.tolist(),physical_column_parameters=np.asarray(parameters).tolist(),
        physical_column_parameter_names=column.parameter_names,source_canonical_covariance=source.total_covariance.tolist())
    truth_path.write_text(json.dumps(truth,indent=2,allow_nan=False)+'\n')
    report=dict(schema='neoarm-pl1-il3-paired-images-v1',backend=jax.default_backend(),source_hashes=sources,
        scope='New noisy paired analyser images for future coupled PL1/IL3 fitting; no optical fit performed.',
        independent_calibration_report=str(calibration_path.relative_to(root)),independent_calibration_report_sha256=digest(calibration_path),
        calibration_response_noise_independent=True,shared_input_calibration_uncertainty_still_required=True,
        arrays_path=str(arrays_path.relative_to(root)),arrays_sha256=digest(arrays_path),truth_path=str(truth_path.relative_to(root)),truth_sha256=digest(truth_path),
        pl1_controls=pl1_controls.tolist(),il3_controls=il3_controls.tolist(),pl1_q_generating_scan=pl1_q.tolist(),il3_offset_generating_scan=offsets.tolist(),
        control_definitions=dict(PL1='same normalized affine control used by independent PL1 calibration',IL3='native IL3 DAC increment relative to robust fixed relay'),
        training_group_indices=np.flatnonzero(training).tolist(),heldout_group_indices=np.flatnonzero(heldout).tolist(),
        input_steps_grouped=inputs.tolist(),detector_pixel_m=pixel,incident_electrons_per_frame=dose,read_noise_electrons=read_noise,
        detector=dict(shape_yx=[256,256],mtf_sigma_pixels=.65,pixel_quadrature_order=16),generated_frame_count=90,
        training_frame_count=54,heldout_frame_count=36,acquisition_groups=rows,truth_likelihood_diagnostics=likelihood,
        independent_gaussian_checks=checks,independent_control_matrix_relative_error=closure,
        all_frame_admissions_passed=admitted,paired_data_admitted=admitted and all(c['passed'] for c in checks),
        optical_parameters_fitted=False,hardware_commands_admitted=False,
        limitations=['All90 frames are new independent Poisson/read-noise draws; none are reused from PL1 calibration.',
            'Noise covariance across acquisitions is block diagonal only under the declared independent-noise and stable-alignment assumptions.',
            'Shared input-step/pixel calibration uncertainties and analyser-calibration uncertainty still belong in the joint physical fit.',
            'Both endpoint IL3 offsets are held out; no failed endpoint is silently relabelled as training.',
            'The objective is the provenance-bound independent-half hard-edge surrogate, not the newer full-tail shared-field model.',
            'Source preparation and provisional DAC laws remain conditional, not hardware commands.'])
    if any(digest(root/name)!=value for name,value in sources.items()):raise RuntimeError('A pinned source changed during paired acquisition.')
    report['source_hashes_unchanged_during_execution']=True
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=audit(args.output);print(json.dumps({k:result[k] for k in ('paired_data_admitted','generated_frame_count','truth_likelihood_diagnostics')},indent=2))
