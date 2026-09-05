"""Conditional IL1/IL2/IL3 conditioning design for a subsequent PL1 scan.

Screen with continuous-Gaussian centroid information including pixel/PSF
variance. Validate shortlisted settings using the finite-detector image
Jacobian with all six image coordinates free. Every score profiles all
nineteen PL1-tomography coordinates; response amplitudes alone are not an
identifiability criterion. No optical fit or hardware feasibility is proved.
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

from neoarm_imaging_recipe import column_with_geometry_anchors,single_ray_transfer
from neoarm_last_lens_tomography import mechanical_shear,predict_response_parameters,geometry_coordinates
from neoarm_last_lens_images import signed_response_from_image_fits,covariance_factor
from neoarm_mixed_gaussian_validation import MixedGaussianSource
from temgym_core.gaussian_image_likelihood import (
    gaussian_image_information_factor,encode_gaussian_image_parameters,make_gaussian_detector_operator,
)
from temgym_core.microscope_inversion import drift_matrix,hard_edge_solenoid_matrix
from temgym_core.neoarm_calibration import DetectorGeometry,hard_edge_column_from_config

GROUPED=np.array([0,2,1,3])


def response_covariance_from_mean_covariance(mean_covariance):
    """Independent plus/minus frames; both x/y components remain correlated."""
    return jnp.einsum('...ij,kl->...ikjl',.5*jnp.asarray(mean_covariance),jnp.eye(4)).reshape((-1,8,8))


def profiled_geometry_precision(parameters, controls, covariance, admitted, input_steps, pixel_m):
    """Full nineteen-coordinate Fisher SVD, with zero-weight rejected states."""
    jac=jax.jacfwd(predict_response_parameters)(parameters,controls).reshape(-1,8,19)
    chol=jnp.linalg.cholesky(covariance)
    whitened=jnp.linalg.solve(chol,jac)*admitted[:,None,None]
    whitened=whitened.reshape(-1,19)
    norms=jnp.maximum(jnp.linalg.norm(whitened,axis=0),1e-100)
    _,singular,right=jnp.linalg.svd(whitened/norms,full_matrices=False)
    factor=(right.T/jnp.maximum(singular,1e-100))/norms[:,None]
    derivative=jax.jacfwd(geometry_coordinates)(parameters,input_steps,pixel_m)
    sigma=jnp.sqrt(jnp.sum((derivative@factor)**2,axis=1))
    rank=jnp.sum(singular>singular[0]*1e-10)
    return sigma,rank,singular,factor


def make_relay_model(column,parameters,kappa):
    """Actual fixed objective and three finite controlled lenses, before PL1."""
    decoded=tuple(v for v in column._decoded_lenses(parameters) if v[0].entrance_z>=column.sample_z_m)
    if tuple(v[0].name for v in decoded)!=('Obj-post','IL1','IL2','IL3','PL1'):
        raise ValueError('Expected fixed objective postfield and IL1/IL2/IL3/PL1.')
    starts=jnp.array([v[1]-v[2]/2 for v in decoded]);lengths=jnp.array([v[2] for v in decoded])
    exits=starts+lengths;objective_q=decoded[0][3]
    ix=jnp.asarray(GROUPED);shear=mechanical_shear(kappa)
    @jax.jit
    def model(relay_q):
        matrix=jnp.eye(4);cursor=column.sample_z_m
        for i in range(4):
            excitation=objective_q if i==0 else relay_q[i-1]
            matrix=hard_edge_solenoid_matrix(excitation,lengths[i])@drift_matrix(starts[i]-cursor)@matrix
            cursor=exits[i]
        matrix=drift_matrix(starts[4]-cursor)@matrix
        grouped=matrix[ix[:,None],ix[None,:]]
        return matrix,grouped@shear
    return model,decoded


def coefficient_parameters(upstream_mechanical,length,gap,inputs,pixel,q_bias,q_slope):
    s=gap/length;scale=inputs/pixel
    w=jnp.stack((-s*upstream_mechanical[:2]*scale,length*upstream_mechanical[2:]*scale))
    return jnp.r_[jnp.log(s),q_bias,q_slope,w.ravel()]


def candidate_geometry(model,relay_q,controls,length,gap,q_bias,q_slope,source_covariance,means):
    canonical,mechanical=model(relay_q)
    q=q_bias+q_slope*controls
    matrices=jax.vmap(lambda excitation:drift_matrix(gap)@hard_edge_solenoid_matrix(excitation,length)@canonical)(q)
    h=matrices[:,jnp.array([0,2])]
    covariance=jnp.einsum('nai,ij,nbj->nab',h,source_covariance,h)
    image_means=jnp.einsum('nai,fi->nfa',h,means)
    return canonical,mechanical,matrices,covariance,image_means


def design(output,*,candidate_count=4096,shortlist=3):
    if jax.default_backend()!='gpu':
        raise RuntimeError('The forward design and image information require GPU.')
    root=Path(__file__).resolve().parents[2];output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    reference_path=root/'examples/generated/neoarm_inversion/imaging_recipe/last_lens_images.json'
    reference=json.loads(reference_path.read_text());digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for path,value in reference['source_hashes'].items():
        if digest(root/path)!=value:raise ValueError(f'Stale image reference: {path}')
    initializer_path=root/'examples/generated/neoarm_inversion/imaging_recipe/measured_initialization.json'
    init=json.loads(initializer_path.read_text())
    config_path=root/'examples/lens_inversion/neoarm_inversion_config.toml'
    geometry_path=root/'examples/microscope_models/measure_microscope_geometry.ipynb'
    column=column_with_geometry_anchors(hard_edge_column_from_config(tomllib.loads(config_path.read_text())),geometry_path)
    parameters=jnp.asarray(init['parameters']);source=MixedGaussianSource(
        np.diag(np.array([2e-9,1.5e-9])**2),np.zeros((2,2)),np.zeros((4,4)),np.zeros(4))
    kappa=800.;model,decoded=make_relay_model(column,parameters,kappa)
    length=float(decoded[-1][2]);gap=float(parameters[1]-decoded[-1][1]-length/2)
    baseline=np.array([float(v[3]) for v in decoded[1:4]])
    t=jnp.linspace(-1.,1.,25);q_bias=-np.pi/2;q_slope=-(np.pi/2-.05)
    inputs=jnp.array([5e-9,5e-9,1e-3,1e-3]);pixel=55e-6;dose=1e6
    detector=DetectorGeometry(z_m=float(parameters[1]),mtf_sigma_px=.65)
    input_means=np.zeros((9,4))
    for i in range(4):input_means[1+2*i,i]=inputs[i];input_means[2+2*i,i]=-inputs[i]
    shear=np.asarray(mechanical_shear(kappa))[np.ix_(GROUPED,GROUPED)]
    means=jnp.asarray(input_means[:,GROUPED]@shear.T);source_cov=jnp.asarray(source.total_covariance)
    geometry=jax.jit(lambda qs:candidate_geometry(model,qs,t,length,gap,q_bias,q_slope,source_cov,means))
    @jax.jit
    def screen(qs):
        _,mechanical,_,covariance,image_means=geometry(qs)
        rms=jnp.sqrt(jnp.linalg.eigvalsh(covariance))/pixel
        marginal=jnp.sqrt(jnp.diagonal(covariance,axis1=1,axis2=2))/pixel
        # Union-bound capture lower bound, conservatively shrinking the camera
        # by the full3-pixel support of the physical detector PSF.
        margin=125.-jnp.abs(image_means/pixel)
        tail=.5*jax.scipy.special.erfc(margin/(jnp.sqrt(2)*marginal[:,None,:]))
        capture_lower=jnp.maximum(0.,1.-2*jnp.sum(tail,axis=2))
        valid=(rms[:,0]>=.25)&(jnp.min(capture_lower,axis=1)>=.99)
        mean_cov=(covariance/pixel**2+jnp.eye(2)*(.65**2+1/12))/dose
        response_cov=response_covariance_from_mean_covariance(mean_cov)
        p=coefficient_parameters(mechanical,length,gap,inputs,pixel,q_bias,q_slope)
        sigma,rank,singular,_=profiled_geometry_precision(p,t,response_cov,valid,inputs,pixel)
        score=jnp.where((rank==19)&(jnp.sum(valid)>=18),jnp.max(sigma[:2]),jnp.inf)
        return score,sigma,jnp.sum(valid),rms,capture_lower,valid,singular
    batch_screen=jax.jit(jax.vmap(screen))
    rng=np.random.default_rng(99131)
    candidates=np.vstack((baseline,rng.uniform(.05,np.pi-.05,size=(candidate_count,3))))
    search_rows=[]
    def evaluate_batch(candidates,label):
        scores=[];start=time.perf_counter()
        for start_index in range(0,len(candidates),128):
            values=batch_screen(jnp.asarray(candidates[start_index:start_index+128]))
            scores.extend(np.asarray(values[0]).tolist())
        scores=np.asarray(scores)
        search_rows.append(dict(stage=label,candidate_count=len(candidates),finite_count=int(np.isfinite(scores).sum()),
            minimum_geometry_sigma=float(np.min(scores)) if np.isfinite(scores).any() else None,
            elapsed_seconds=time.perf_counter()-start))
        print(search_rows[-1],flush=True)
        return scores
    scores=evaluate_batch(candidates,'uniform_first_branch')
    for width in (.20,.05,.01):
        seeds=candidates[np.argsort(scores)[:16]]
        proposals=np.clip(seeds[:,None,:]+rng.normal(size=(16,24,3))*width,.05,np.pi-.05).reshape(-1,3)
        proposal_scores=evaluate_batch(proposals,f'local_screen_sigma{width}')
        candidates=np.vstack((candidates,proposals));scores=np.r_[scores,proposal_scores]
    selected=[0]
    for i in np.argsort(scores):
        if np.isfinite(scores[i]) and all(np.linalg.norm(candidates[i]-candidates[j])>.03 for j in selected):
            selected.append(int(i))
        if len(selected)>=shortlist+1:break
    operator=jax.jit(make_gaussian_detector_operator(detector,16))
    finer=jax.jit(make_gaussian_detector_operator(detector,32))
    information=jax.jit(lambda p:gaussian_image_information_factor(p,detector,quadrature_order=16))
    rows=[]
    for candidate_index in selected:
        qs=candidates[candidate_index];approx=screen(qs)
        canonical,mechanical,matrices,covariances,image_means=map(np.asarray,geometry(qs))
        valid=np.asarray(approx[5]);response_cov=[];exact_capture=[];image_sigma=[];raw_factors=[]
        for n in np.flatnonzero(valid):
            pc=[];cc=[];factors=[]
            for mean in image_means[n]:
                p=encode_gaussian_image_parameters(mean,covariances[n],dose,detector)
                factor=np.asarray(information(p));_,s,v=np.linalg.svd(factor,full_matrices=False)
                c=(v.T/s)@(v/s[:,None])
                pc.append(p);cc.append(c);factors.append(factor)
                exact_capture.append(float(operator(mean,covariances[n],dose).sum()/dose))
                image_sigma.extend(np.sqrt(np.diag(c)[:2]).tolist())
            response_cov.append(signed_response_from_image_fits(pc,cc)[1]);raw_factors.append(factors)
        p=coefficient_parameters(jnp.asarray(mechanical),length,gap,inputs,pixel,q_bias,q_slope)
        accepted_t=t[valid]
        sigma,rank,singular,factor=profiled_geometry_precision(p,accepted_t,jnp.asarray(response_cov),
            jnp.ones(len(response_cov),dtype=bool),inputs,pixel)
        # Held-out excitation values are not included in the Fisher score.
        held_q=jnp.array([-.37,-1.07,-2.71]);held_t=(held_q-q_bias)/q_slope
        held_jac=jax.jacfwd(predict_response_parameters)(p,held_t).reshape(-1,19)
        held_sigma=jnp.sqrt(jnp.sum((held_jac@factor)**2,axis=1)).reshape(3,2,4)
        narrow=int(np.flatnonzero(valid)[np.argmin(np.asarray(approx[3])[valid,0])])
        far=int(np.argmax(np.linalg.norm(image_means[narrow],axis=1)))
        mean=image_means[narrow,far];variance=covariances[narrow]
        a=np.asarray(operator(mean,variance,dose));b=np.asarray(finer(mean,variance,dose))
        quadrature_l2=float(np.linalg.norm((a-b)/np.sqrt(b+.6**2)))
        controls=np.tile(np.asarray(init['nominal_controls'][0]),(25,1))
        for lens,qvalue in zip(decoded[1:4],qs):
            item,_,_,bias,slope=lens
            controls[:,item.channel_index]=item.reference_dac+(qvalue-float(bias))/float(slope)
        item,_,_,bias,slope=decoded[-1]
        controls[:,item.channel_index]=item.reference_dac+(np.asarray(q_bias+q_slope*t)-float(bias))/float(slope)
        row=dict(candidate_index=candidate_index,label='baseline40k' if candidate_index==0 else 'screen_shortlist',
            relay_q_IL1_IL2_IL3=qs.tolist(),synthetic_dac_channel_names=column.channel_names,
            synthetic_dac_states=controls.tolist(),pl1_q=np.asarray(q_bias+q_slope*t).tolist(),
            admitted_state_indices=np.flatnonzero(valid).tolist(),rejected_state_indices=np.flatnonzero(~valid).tolist(),
            state_count=int(valid.sum()),frame_count=int(valid.sum())*9,
            approximate_geometry_standard_errors_percent=(100*np.asarray(approx[1])[:2]).tolist(),
            exact_image_fisher_geometry_standard_errors_percent=(100*np.asarray(sigma)[:2]).tolist(),
            exact_geometry_sigma_with_0p2percent_input_scale_priors=(100*np.sqrt(np.asarray(sigma[:2])**2+.002**2)).tolist(),
            numerical_rank=int(rank),column_scaled_singular_values=np.asarray(singular).tolist(),
            optical_rms_principal_pixels=np.asarray(approx[3]).tolist(),
            conservative_capture_lower_bound_per_frame=np.asarray(approx[4]).tolist(),
            exact_capture_range=[min(exact_capture),max(exact_capture)],
            exact_per_frame_mean_sigma_pixels=[min(image_sigma),max(image_sigma)],
            shortest_width_quadrature16_to32_whitened_l2=quadrature_l2,
            held_out_pl1_q=np.asarray(held_q).tolist(),held_out_signed_response_sigma_pixels=np.asarray(held_sigma).tolist(),
            upstream_canonical=canonical.tolist(),upstream_mechanical=mechanical.tolist(),
            independent_control_map_maximum_error=None,
            fisher_design_admitted=bool(rank==19 and min(exact_capture)>=.99 and quadrature_l2<.2))
        direct=jax.jit(jax.vmap(lambda u:single_ray_transfer(column,parameters,u)))(jnp.asarray(controls))
        relative=float(jnp.max(jnp.abs(direct-jnp.asarray(matrices)))/jnp.max(jnp.abs(direct)))
        row['independent_control_map_maximum_relative_error']=relative
        row['fisher_design_admitted']=bool(row['fisher_design_admitted'] and relative<1e-12)
        rows.append(row)
        print({'candidate':candidate_index,'relay_q':qs.tolist(),'exact_sigma_percent':row['exact_image_fisher_geometry_standard_errors_percent'],
               'states':row['state_count'],'admitted':row['fisher_design_admitted']},flush=True)
    sources=dict(reference['source_hashes']);sources.update({str(p.relative_to(root)):digest(p) for p in
        (Path(__file__),reference_path,initializer_path)})
    report=dict(schema='neoarm-last-lens-design-v1',backend=jax.default_backend(),x64_enabled=jax.config.x64_enabled,
        scope='Fixed-objective vacuum relay conditioning; image Fisher only, no noisy fit, global uniqueness or hardware recipe.',
        source_hashes=sources,random_seed=99131,search_stages=search_rows,incident_electrons_per_frame=dose,
        input_steps_grouped=np.asarray(inputs).tolist(),source_canonical_covariance=source.total_covariance.tolist(),
        source_wavelength_m=source.wavelength_m,controlled_relay_lenses=['IL1','IL2','IL3'],
        first_branch_absolute_q_bounds=[.05,float(np.pi-.05)],objective_post_q=float(decoded[0][3]),
        candidate_results=rows,screen_candidate_count=len(candidates),
        screening_approximation='Unclipped continuous-Gaussian centroid covariance (spatial_covariance+pixel_variance+PSF_variance)/dose; conservative capture gate.',
        exact_validation='Raw finite-pixel/PSF Gaussian image Jacobian with6 free image coordinates; marginalized2x2 mean blocks; all19 tomography coordinates profiled by SVD.',
        minimum_accepted_states=18,minimum_unblurred_rms_pixels=.25,minimum_capture_fraction=.99,
        limitations=['Objective and source preparation remain fixed to the same provisional measured-geometry initialization.',
            'All DAC coordinates and first-branch bounds are synthetic; actual control safety, hysteresis and aperture transmission are untested.',
            'The0.2percent independent input-scale priors add an exact0.2percent log length/gap uncertainty floor through a coefficient-column gauge.',
            'Three extra held-out excitation responses are predictions of local information, not observed held-out validation.',
            'No ray-amplitude criterion is substituted for the full profiled geometry Fisher score.'])
    if any(digest(root/p)!=v for p,v in sources.items()):raise RuntimeError('A pinned source changed during design.')
    report['source_hashes_unchanged_during_execution']=True
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--candidate-count',type=int,default=4096);parser.add_argument('--shortlist',type=int,default=3)
    args=parser.parse_args();design(args.output,candidate_count=args.candidate_count,shortlist=args.shortlist)
