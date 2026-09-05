"""Fresh noisy Gaussian images for the margin-aware fixed PL1 acquisition.

No optical optimizer is invoked. Six image coordinates are fitted from data
moments; image-derived responses and covariance are saved separately from
generating optical truth. Three additional acquisition states are held out
from the training response set and the reported training Fisher calculation.
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

from neoarm_last_lens_design import make_relay_model,coefficient_parameters,profiled_geometry_precision,GROUPED
from neoarm_last_lens_images import image_initializer,signed_response_from_image_fits,render_pure_counts
from neoarm_last_lens_tomography import mechanical_shear,predict_response_parameters
from neoarm_imaging_recipe import column_with_geometry_anchors,single_ray_transfer
from neoarm_mixed_gaussian_validation import MixedGaussianSource
from temgym_core.gaussian_image_likelihood import (
    make_gaussian_detector_operator,fit_gaussian_image,encode_gaussian_image_parameters,
)
from temgym_core.microscope_inversion import drift_matrix,hard_edge_solenoid_matrix
from temgym_core.neoarm_calibration import DetectorGeometry,hard_edge_column_from_config


def normalized_excitation(q,q_bias=-np.pi/2,q_slope=-(np.pi/2-.05)):
    return (jnp.asarray(q)-q_bias)/q_slope


def assert_separate_acquisitions(training_controls,heldout_controls):
    a,b=np.asarray(training_controls),np.asarray(heldout_controls)
    if a.ndim!=1 or b.ndim!=1 or not np.all(np.isfinite(np.r_[a,b])):
        raise ValueError('Acquisition coordinates must be finite vectors.')
    if np.any(np.abs(a[:,None]-b[None,:])<1e-10):
        raise ValueError('Held-out acquisition coordinates overlap training.')


def truth_response_likelihood(observed,expected,covariance):
    """No optical fit: audit the full correlated response noise at truth."""
    error=(np.asarray(observed)-np.asarray(expected)).reshape(-1,8)
    white=np.linalg.solve(np.linalg.cholesky(covariance),error[...,None])[...,0]
    chi_square=float(np.sum(white**2));dof=int(white.size)
    return dict(residual_rms_sigma=float(np.sqrt(chi_square/dof)),chi_square=chi_square,
        degrees_of_freedom=dof,chi_square_z=float((chi_square-dof)/np.sqrt(2*dof)),
        maximum_whitened_response_error=float(np.max(np.abs(white))),optical_parameters_fitted=False)


def audit(output,*,incident_electrons=1e6):
    if jax.default_backend()!='gpu':raise RuntimeError('Redesigned forward images require GPU.')
    root=Path(__file__).resolve().parents[2];output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    design_path=root/'examples/generated/neoarm_inversion/imaging_recipe/last_lens_robust_design.json'
    design=json.loads(design_path.read_text());digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    sources=dict(design['source_hashes']);sources.update({str(p.relative_to(root)):digest(p) for p in (Path(__file__),design_path)})
    for name,value in sources.items():
        if digest(root/name)!=value:raise ValueError(f'Stale fixed-acquisition source:{name}')
    if not design['all_tested_scenarios_admitted'] or not all(r['fixed_topology_admitted'] for r in design['approximate_capture_width_corner_checks']):
        raise ValueError('The input fixed-topology design did not pass its declared finite checks.')
    initializer=json.loads((root/'examples/generated/neoarm_inversion/imaging_recipe/measured_initialization.json').read_text())
    config=tomllib.loads((root/'examples/lens_inversion/neoarm_inversion_config.toml').read_text())
    column=column_with_geometry_anchors(hard_edge_column_from_config(config),
        root/'examples/microscope_models/measure_microscope_geometry.ipynb')
    parameters=jnp.asarray(initializer['parameters']);model,decoded=make_relay_model(column,parameters,800.)
    relay=jnp.asarray(design['selected_relay_q_IL1_IL2_IL3']);canonical,mechanical=map(np.asarray,model(relay))
    length=float(decoded[-1][2]);gap=float(parameters[1]-decoded[-1][1]-length/2)
    q_candidates=np.asarray(design['fixed_pl1_q']);selected=np.asarray(design['fixed_selected_state_indices'])
    training_q=q_candidates[selected];heldout_q=np.array([-.37,-2.10,-2.75])
    train_t=np.asarray(normalized_excitation(training_q));held_t=np.asarray(normalized_excitation(heldout_q))
    assert_separate_acquisitions(train_t,held_t)
    inputs=np.asarray(design['input_steps_grouped']);pixel=55e-6;read_noise=.6
    detector=DetectorGeometry(z_m=float(parameters[1]),mtf_sigma_px=.65)
    source=MixedGaussianSource(np.diag(np.array([2e-9,1.5e-9])**2),np.zeros((2,2)),np.zeros((4,4)),np.zeros(4))
    input_states=np.zeros((9,4))
    for i in range(4):input_states[1+2*i,i]=inputs[i];input_states[2+2*i,i]=-inputs[i]
    shear=np.asarray(mechanical_shear(800.))[np.ix_(GROUPED,GROUPED)]
    canonical_means=input_states[:,GROUPED]@shear.T
    operator=jax.jit(make_gaussian_detector_operator(detector,16))
    matrix_fun=jax.jit(lambda q:drift_matrix(gap)@hard_edge_solenoid_matrix(q,length)@jnp.asarray(canonical))
    all_q=np.r_[training_q,heldout_q];matrices=np.asarray(jax.vmap(matrix_fun)(jnp.asarray(all_q)))
    means=np.einsum('nai,fi->nfa',matrices[:,[0,2]],canonical_means)
    spatial_cov=np.einsum('nai,ij,nbj->nab',matrices[:,[0,2]],source.total_covariance,matrices[:,[0,2]])
    widths=np.sqrt(np.linalg.eigvalsh(spatial_cov))/pixel
    if widths.min()<.35:raise ValueError('A selected training or held-out state fails the declared nominal0.35-pixel width margin.')
    controls=np.tile(np.asarray(design['synthetic_fixed_relay_dacs']),(len(all_q),1))
    last,_,_,bias,slope=decoded[-1]
    controls[:,last.channel_index]=last.reference_dac+(all_q-float(bias))/float(slope)
    direct=np.asarray(jax.jit(jax.vmap(lambda u:single_ray_transfer(column,parameters,u)))(jnp.asarray(controls)))
    closure=float(np.max(np.abs(direct-matrices))/np.max(np.abs(direct)))
    if closure>1e-12:raise ValueError('Acquisition controls and independent column ray map disagree.')
    expected_images=[];observed_images=[];fitted_images=[];variance_images=[]
    fitted_parameters=[];parameter_covariances=[];initial_parameters=[];rows=[]
    responses=[];response_covariances=[]
    for state_index,(excitation,mean_group,covariance) in enumerate(zip(all_q,means,spatial_cov)):
        started=time.perf_counter();training=state_index<len(training_q)
        candidate=int(selected[state_index]) if training else state_index-len(training_q)
        seed_base=(930000 if training else 940000)+100*candidate
        pars=[];covs=[];starts=[];frames=[];expected_group=[];observed_group=[];fitted_group=[];variance_group=[]
        for frame_index,mean in enumerate(mean_group):
            expected=np.asarray(operator(mean,covariance,incident_electrons));variance=expected+read_noise**2
            key1,key2=jax.random.split(jax.random.PRNGKey(seed_base+frame_index))
            observed=np.asarray(jax.random.poisson(key1,jnp.asarray(expected),shape=expected.shape)
                +read_noise*jax.random.normal(key2,expected.shape))
            initial=image_initializer(observed,detector,read_noise)
            fit=fit_gaussian_image(observed,variance,detector,initial_parameters=initial,quadrature_order=16,max_nfev=100)
            fit.require_accepted()
            # Truth is first encoded after the data-only optimization finishes.
            true_image_parameters=np.asarray(encode_gaussian_image_parameters(mean,covariance,incident_electrons,detector))
            error_sigma=(fit.parameters-true_image_parameters)/np.sqrt(np.diag(fit.parameter_covariance))
            frames.append(dict(frame=frame_index,noise_seed=seed_base+frame_index,admitted=True,
                optimizer_success=bool(fit.optimizer_result.success),optimizer_status=fit.optimizer_result.status,nfev=fit.optimizer_result.nfev,
                image_information_rank=fit.numerical_rank,fit_residual_rms_sigma=fit.residual_rms_sigma,
                chi_square_z=fit.chi_square_z,capture_fraction=float(expected.sum()/incident_electrons),
                six_image_parameter_error_sigma=error_sigma.tolist(),
                mean_standard_errors_pixels=np.sqrt(np.diag(fit.parameter_covariance)[:2]).tolist(),
                quadrature_rms_sigma=fit.quadrature_rms_sigma,
                quadrature_parameter_shift_sigma=fit.quadrature_parameter_shift_sigma))
            pars.append(fit.parameters);covs.append(fit.parameter_covariance);starts.append(initial)
            expected_group.append(expected);observed_group.append(observed);fitted_group.append(fit.fitted_counts_yx);variance_group.append(variance)
        response,response_cov=signed_response_from_image_fits(pars,covs)
        responses.append(response);response_covariances.append(response_cov)
        expected_images.append(expected_group);observed_images.append(observed_group);fitted_images.append(fitted_group);variance_images.append(variance_group)
        fitted_parameters.append(pars);parameter_covariances.append(covs);initial_parameters.append(starts)
        row=dict(group='training' if training else 'held_out',candidate_index=candidate,q_rad=float(excitation),
            normalized_dac=float(normalized_excitation(excitation)),optical_rms_principal_pixels=widths[state_index].tolist(),
            frames=frames,admitted=True,elapsed_seconds=time.perf_counter()-started)
        rows.append(row);print(f'{row["group"]} state{candidate}:9 image fits admitted,{row["elapsed_seconds"]:.1f}s',flush=True)
    response=np.asarray(responses);response_cov=np.asarray(response_covariances);n_train=len(training_q)
    truth=np.asarray(coefficient_parameters(jnp.asarray(mechanical),length,gap,jnp.asarray(inputs),pixel,-np.pi/2,-(np.pi/2-.05)))
    expected_response=np.asarray(predict_response_parameters(jnp.asarray(truth),normalized_excitation(all_q)))
    physical_expected=np.einsum('nij,jk->nik',matrices[:,[0,2]],shear)[:,:,GROUPED]*inputs/pixel
    response_closure=float(np.max(np.abs(expected_response-physical_expected)))
    if response_closure>1e-10:raise ValueError('Generating response and physical input controls disagree.')
    train_likelihood=truth_response_likelihood(response[:n_train],expected_response[:n_train],response_cov[:n_train])
    held_likelihood=truth_response_likelihood(response[n_train:],expected_response[n_train:],response_cov[n_train:])
    sigma,rank,_,_=profiled_geometry_precision(jnp.asarray(truth),jnp.asarray(train_t),jnp.asarray(response_cov[:n_train]),
        jnp.ones(n_train,bool),jnp.asarray(inputs),pixel)
    independent=[]
    for state_index in sorted(set([int(np.argmin(widths[:,0])),int(np.argmax(widths[:,1]))])):
        frame=int(np.argmax(np.linalg.norm(means[state_index],axis=1)))
        mode_source=MixedGaussianSource(source.position_covariance_xy_m2,source.curvature_xy_per_m,source.centre_covariance,canonical_means[frame])
        renders={};refinements=[];previous=None
        for order in (8,16,32):
            image,symplectic_error=render_pure_counts(mode_source,matrices[state_index],detector,pixel_order=order,
                                                    incident_electrons=incident_electrons)
            renders[order]=np.asarray(image);expected=np.asarray(expected_images[state_index][frame]);variance=np.asarray(variance_images[state_index][frame])
            residual=(renders[order]-expected)/np.sqrt(variance)
            transition=None if previous is None else float(np.linalg.norm((renders[order]-previous)/np.sqrt(variance)))
            refinements.append(dict(pixel_quadrature_order=order,image_whitened_l2=float(np.linalg.norm(residual)),
                                    image_rms_sigma=float(np.sqrt(np.mean(residual**2))),transition_whitened_l2=transition))
            previous=renders[order]
        path=output.with_name(output.stem+f'_gaussian_check_{state_index}.npz')
        np.savez_compressed(path,gaussian_pixel8_counts_yx=renders[8],gaussian_pixel16_counts_yx=renders[16],
            gaussian_pixel32_counts_yx=renders[32],covariance_counts_yx=expected,noise_sigma_yx=np.sqrt(variance),transfer_matrix=matrices[state_index])
        independent.append(dict(state_index=state_index,group=rows[state_index]['group'],frame=frame,pure_gaussian_mode_count=1,
            common_axial_path_m=0.,beam_dependent_action_retained=True,balanced_symplectic_error=symplectic_error,
            refinements=refinements,two_successive_pixel_refinements_passed=all(v['transition_whitened_l2']<.2 for v in refinements[1:]),
            passed=all(v['image_whitened_l2']<.2 for v in refinements) and all(v['transition_whitened_l2']<.2 for v in refinements[1:]),
            arrays_path=str(path.relative_to(root)),arrays_sha256=digest(path)))
    truth_path=output.with_name(output.stem+'_truth.json')
    truth_report=dict(schema='neoarm-redesigned-generating-truth-v1',source_hashes=sources,
        purpose='Post-fit audit only; not an optical optimizer initializer.',
        inputs=dict(truth_parameters=truth.tolist(),input_basis_steps_grouped=inputs.tolist(),detector_pixel_m=pixel,
            generating_length_m=length,generating_detector_gap_m=gap,generating_unknown_kappa_per_m=800.,
            upstream_canonical=canonical.tolist(),upstream_mechanical=mechanical.tolist(),
            normalized_dac=train_t.tolist(),heldout_normalized_dac=held_t.tolist(),
            expected_response_pixels=expected_response[:n_train].tolist(),heldout_expected_response_pixels=expected_response[n_train:].tolist()),
        physical_column_parameters=np.asarray(parameters).tolist(),physical_column_parameter_names=column.parameter_names,
        pure_source_canonical_covariance=source.total_covariance.tolist())
    truth_path.write_text(json.dumps(truth_report,indent=2,allow_nan=False)+'\n')
    arrays_path=output.with_suffix('.npz');arrays={}
    for name,value in dict(expected_counts=expected_images,noisy_counts=observed_images,fitted_counts=fitted_images,
        variance_counts=variance_images,image_parameters=fitted_parameters,image_parameter_covariances=parameter_covariances,
        image_initial_parameters=initial_parameters,response=response,response_covariance=response_cov,raw_dac_values=controls).items():
        arrays[name]=np.asarray(value)[:n_train];arrays['heldout_'+name]=np.asarray(value)[n_train:]
    arrays.update(accepted_normalized_dac=train_t,heldout_normalized_dac=held_t,candidate_q=q_candidates,
        generated_candidate_indices=selected,accepted_candidate_indices=selected,heldout_q=heldout_q,
        input_states_grouped=input_states,input_steps_grouped=inputs,detector_pixel_m=np.asarray(pixel))
    np.savez_compressed(arrays_path,**arrays)
    report=dict(schema='neoarm-last-lens-redesigned-images-v1',backend=jax.default_backend(),jax_version=jax.__version__,
        x64_enabled=jax.config.x64_enabled,source_hashes=sources,source_hashes_unchanged_during_execution=True,
        scope='Fresh noisy images for the fixed robust relay; image metrology only, no optical recovery or hardware recipe.',
        design_report_path=str(design_path.relative_to(root)),design_report_sha256=digest(design_path),
        arrays_path=str(arrays_path.relative_to(root)),arrays_sha256=digest(arrays_path),truth_path=str(truth_path.relative_to(root)),truth_sha256=digest(truth_path),
        input_steps_grouped=inputs.tolist(),detector_pixel_m=pixel,incident_electrons_per_frame=incident_electrons,
        read_noise_electrons=read_noise,detector=dict(shape_yx=[256,256],mtf_sigma_pixels=.65,pixel_quadrature_order=16),
        admitted_state_count=n_train,heldout_state_count=len(heldout_q),training_frame_count=9*n_train,heldout_frame_count=9*len(heldout_q),
        generated_frame_count=9*len(all_q),training_states=rows[:n_train],heldout_states=rows[n_train:],
        training_response_likelihood_at_truth=train_likelihood,heldout_response_likelihood_at_truth=held_likelihood,
        training_only_conditional_fisher=dict(rank=int(rank),geometry_standard_errors_percent=(100*np.asarray(sigma[:2])).tolist(),
            geometry_sigma_with_0p2percent_input_scale_priors=(100*np.sqrt(np.asarray(sigma[:2])**2+.002**2)).tolist(),
            generating_truth_used_only_for_information_audit=True,heldout_frames_used=False),
        independent_gaussian_checks=independent,independent_control_matrix_relative_error=closure,
        generating_response_maximum_closure_error_pixels=response_closure,
        image_generation_admitted=all(v['passed'] for v in independent) and abs(train_likelihood['chi_square_z'])<5 and abs(held_likelihood['chi_square_z'])<5,
        optical_parameters_fitted=False,hardware_commands_admitted=False,
        limitations=['All image starts use data-only thresholded moments; generating image coordinates are evaluated after each optimizer returns.',
            'Poisson plus independent Gaussian read noise is simulated; fixed WLS variance is exact synthetic truth and its uncertainty is not propagated.',
            'Alignment is unchanged within each nine-frame signed-input group; differential drift is not simulated.',
            'The same selected18-state training topology is retained. Held-out exposures are separate and never enter training precision.',
            'The pure source and calibrated specimen inputs remain conditional preparations, not condenser calibration.',
            'Design robustness covers its recorded finite scenarios, not every parameter in a continuous uncertainty box.'])
    if any(digest(root/name)!=value for name,value in sources.items()):raise RuntimeError('A pinned source changed during acquisition.')
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--incident-electrons',type=float,default=1e6);args=parser.parse_args()
    result=audit(args.output,incident_electrons=args.incident_electrons)
    print(json.dumps({k:result[k] for k in ('image_generation_admitted','generated_frame_count',
        'training_response_likelihood_at_truth','heldout_response_likelihood_at_truth','training_only_conditional_fisher')},indent=2))
