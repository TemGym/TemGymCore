"""Separate margin-aware relay design; the first design remains immutable.

Select a fixed PL1 subset at nominal unblurred RMS>=0.35pixel. Require that
same subset to retain RMS>=0.25pixel and conservative capture>=99percent at
all eight +/-0.02rad IL-relay corners and source-size factors0.9/1/1.1.
These finite checks are not a certificate for every point in a parameter box.
Validate exact image information at nominal, six signed single-control
perturbations, and both quantum-valid source-size endpoints.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import os
import tomllib

os.environ.setdefault('JAX_ENABLE_X64','true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import jax
import jax.numpy as jnp
import numpy as np

from neoarm_last_lens_design import (
    make_relay_model,candidate_geometry,coefficient_parameters,profiled_geometry_precision,
    response_covariance_from_mean_covariance,GROUPED,
)
from neoarm_last_lens_images import signed_response_from_image_fits
from neoarm_last_lens_tomography import mechanical_shear
from neoarm_imaging_recipe import column_with_geometry_anchors
from neoarm_mixed_gaussian_validation import MixedGaussianSource
from temgym_core.neoarm_calibration import DetectorGeometry,hard_edge_column_from_config
from temgym_core.gaussian_image_likelihood import (
    gaussian_image_information_factor,encode_gaussian_image_parameters,make_gaussian_detector_operator,
)


def resized_pure_covariance(covariance,size_factor):
    """Symplectic resizing preserves wavelength and pure-mode emittances."""
    scales=jnp.array([size_factor,1/size_factor,size_factor,1/size_factor])
    return jnp.asarray(covariance)*scales[:,None]*scales[None,:]


def fixed_selection_gate(rms,capture,selected,*,minimum_width=.25,minimum_capture=.99):
    return jnp.all(jnp.where(selected,(rms[:,0]>=minimum_width)&(jnp.min(capture,axis=1)>=minimum_capture),True))


def audit(output,*,candidate_count=2048):
    if jax.default_backend()!='gpu':raise RuntimeError('Relay forward design requires GPU.')
    root=Path(__file__).resolve().parents[2];output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    original_path=root/'examples/generated/neoarm_inversion/imaging_recipe/last_lens_design.json'
    original=json.loads(original_path.read_text());digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    sources=dict(original['source_hashes'])
    sources.update({str(p.relative_to(root)):digest(p) for p in (Path(__file__),original_path)})
    for path,value in sources.items():
        if digest(root/path)!=value:raise ValueError(f'Stale source:{path}')
    init=json.loads((root/'examples/generated/neoarm_inversion/imaging_recipe/measured_initialization.json').read_text())
    config=tomllib.loads((root/'examples/lens_inversion/neoarm_inversion_config.toml').read_text())
    column=column_with_geometry_anchors(hard_edge_column_from_config(config),
        root/'examples/microscope_models/measure_microscope_geometry.ipynb')
    p=jnp.asarray(init['parameters']);model,decoded=make_relay_model(column,p,800.)
    length=float(decoded[-1][2]);gap=float(p[1]-decoded[-1][1]-length/2)
    t=jnp.linspace(-1.,1.,25);qb=-np.pi/2;qs=-(np.pi/2-.05)
    inputs=jnp.array([5e-9,5e-9,1e-3,1e-3]);pixel=55e-6;dose=1e6
    source=MixedGaussianSource(np.diag(np.array([2e-9,1.5e-9])**2),np.zeros((2,2)),np.zeros((4,4)),np.zeros(4))
    covariance=jnp.asarray(source.total_covariance)
    input_means=np.zeros((9,4))
    for i in range(4):input_means[1+2*i,i]=inputs[i];input_means[2+2*i,i]=-inputs[i]
    shear=np.asarray(mechanical_shear(800.))[np.ix_(GROUPED,GROUPED)]
    means=jnp.asarray(input_means[:,GROUPED]@shear.T)
    detector=DetectorGeometry(z_m=float(p[1]),mtf_sigma_px=.65)
    @jax.jit
    def moments(relay,source_scale):
        _,mechanical,_,cov,image_means=candidate_geometry(model,relay,t,length,gap,qb,qs,
            resized_pure_covariance(covariance,source_scale),means)
        rms=jnp.sqrt(jnp.linalg.eigvalsh(cov))/pixel
        marginal=jnp.sqrt(jnp.diagonal(cov,axis1=1,axis2=2))/pixel
        margin=125.-jnp.abs(image_means/pixel)
        capture=jnp.maximum(0.,1.-jnp.sum(jax.scipy.special.erfc(margin/(jnp.sqrt(2)*marginal[:,None,:])),axis=2))
        return mechanical,cov,image_means,rms,capture
    @jax.jit
    def approximate(relay,source_scale,selected):
        mechanical,cov,_,rms,capture=moments(relay,source_scale)
        rcov=response_covariance_from_mean_covariance((cov/pixel**2+jnp.eye(2)*(.65**2+1/12))/dose)
        parameters=coefficient_parameters(mechanical,length,gap,inputs,pixel,qb,qs)
        sigma,rank,_,_=profiled_geometry_precision(parameters,t,rcov,selected,inputs,pixel)
        return jnp.where((rank==19)&fixed_selection_gate(rms,capture,selected),jnp.max(sigma[:2]),jnp.inf)
    corner_offsets=jnp.asarray(list(itertools.product((-.02,.02),repeat=3)))
    scenario_offsets=jnp.vstack((jnp.zeros((1,3)),jnp.eye(3)*.02,-jnp.eye(3)*.02,jnp.zeros((2,3))))
    scenario_scales=jnp.array([1.]*7+[.9,1.1])
    @jax.jit
    def screen(relay):
        _,_,_,rms,capture=moments(relay,1.)
        selected=(rms[:,0]>=.35)&(jnp.min(capture,axis=1)>=.99)
        def corner(offset):
            def size(scale):
                *_,rr,cc=moments(relay+offset,scale)
                return fixed_selection_gate(rr,cc,selected)
            return jnp.all(jax.vmap(size)(jnp.array([.9,1.,1.1])))
        safe=jnp.all(jax.vmap(corner)(corner_offsets))
        score=approximate(relay,1.,selected)
        return jnp.where(safe&(jnp.sum(selected)>=18),score,jnp.inf)
    batch=jax.jit(jax.vmap(screen));rng=np.random.default_rng(99137)
    seeds=np.array([r['relay_q_IL1_IL2_IL3'] for r in original['candidate_results']])
    candidates=np.vstack((seeds,rng.uniform(.05,np.pi-.05,size=(candidate_count,3))))
    stages=[]
    def evaluate(candidates,label):
        scores=np.concatenate([np.asarray(batch(jnp.asarray(candidates[i:i+128]))) for i in range(0,len(candidates),128)])
        stages.append(dict(label=label,candidate_count=len(candidates),finite_count=int(np.isfinite(scores).sum()),
                           minimum_sigma=float(np.min(scores)) if np.isfinite(scores).any() else None))
        print(stages[-1],flush=True);return scores
    scores=evaluate(candidates,'uniform_and_original_seeds')
    for width in (.2,.05,.01):
        seeds=candidates[np.argsort(scores)[:16]]
        proposals=np.clip(seeds[:,None,:]+rng.normal(size=(16,24,3))*width,.05,np.pi-.05).reshape(-1,3)
        candidate_scores=evaluate(proposals,f'local{width}')
        candidates=np.vstack((candidates,proposals));scores=np.r_[scores,candidate_scores]
    shortlist=np.argsort(scores)[:32]
    robust_scores=[]
    for index in shortlist:
        relay=jnp.asarray(candidates[index]);*_,rms,capture=moments(relay,1.)
        selected=(rms[:,0]>=.35)&(jnp.min(capture,axis=1)>=.99)
        values=jax.vmap(lambda delta,scale:approximate(relay+delta,scale,selected))(scenario_offsets,scenario_scales)
        robust_scores.append(float(jnp.max(values)))
    best_index=int(shortlist[np.argmin(robust_scores)]);best=candidates[best_index]
    *_,rms,capture=moments(best,1.);selected=np.asarray((rms[:,0]>=.35)&(jnp.min(capture,axis=1)>=.99))
    selected_indices=np.flatnonzero(selected);accepted_t=t[selected]
    info=jax.jit(lambda parameters:gaussian_image_information_factor(parameters,detector,quadrature_order=16))
    operator=jax.jit(make_gaussian_detector_operator(detector,16))
    finer=jax.jit(make_gaussian_detector_operator(detector,32))
    rows=[]
    for delta,source_scale in zip(np.asarray(scenario_offsets),np.asarray(scenario_scales)):
        mechanical,cov,image_means,rr,cc=map(np.asarray,moments(best+delta,source_scale))
        response_covariances=[];captures=[];ranks=[]
        for n in selected_indices:
            parameters=[];covariances=[]
            for mean in image_means[n]:
                pp=encode_gaussian_image_parameters(mean,cov[n],dose,detector)
                information=np.asarray(info(pp));_,singular,right=np.linalg.svd(information,full_matrices=False)
                ranks.append(int(np.sum(singular>singular[0]*1e-10)))
                covariance_parameter=(right.T/singular)@(right/singular[:,None])
                parameters.append(pp);covariances.append(covariance_parameter)
                captures.append(float(operator(mean,cov[n],dose).sum()/dose))
            response_covariances.append(signed_response_from_image_fits(parameters,covariances)[1])
        parameters=coefficient_parameters(jnp.asarray(mechanical),length,gap,inputs,pixel,qb,qs)
        sigma,rank,_,_=profiled_geometry_precision(parameters,accepted_t,jnp.asarray(response_covariances),
            jnp.ones(len(selected_indices),bool),inputs,pixel)
        narrow=int(selected_indices[np.argmin(rr[selected,0])]);far=int(np.argmax(np.linalg.norm(image_means[narrow],axis=1)))
        a=np.asarray(operator(image_means[narrow,far],cov[narrow],dose))
        b=np.asarray(finer(image_means[narrow,far],cov[narrow],dose))
        quad=float(np.linalg.norm((b-a)/np.sqrt(b+.6**2)))
        row=dict(relay_q_offset_rad=delta.tolist(),source_intensity_rms_size_factor=float(source_scale),
            fixed_selected_state_indices=selected_indices.tolist(),state_count=len(selected_indices),frame_count=9*len(selected_indices),
            nominal_design_rms_minimum=.35,admission_rms_minimum=.25,
            actual_selected_rms_range=[float(rr[selected].min()),float(rr[selected].max())],
            capture_range=[min(captures),max(captures)],minimum_image_information_rank=min(ranks),
            geometry_fisher_rank=int(rank),geometry_standard_errors_percent=(100*np.asarray(sigma[:2])).tolist(),
            geometry_sigma_with_0p2percent_input_scale_priors=(100*np.sqrt(np.asarray(sigma[:2])**2+.002**2)).tolist(),
            narrow_state_quadrature16_to32_whitened_l2=quad,
            admitted=bool(rank==19 and min(ranks)==6 and rr[selected].min()>=.25 and min(captures)>=.99 and quad<.2))
        rows.append(row);print(row,flush=True)
    corners=[]
    for delta in np.asarray(corner_offsets):
        for source_scale in (.9,1.,1.1):
            *_,rr,cc=map(np.asarray,moments(best+delta,source_scale))
            corners.append(dict(relay_q_offset_rad=delta.tolist(),source_size_factor=source_scale,
                minimum_selected_rms_pixels=float(rr[selected].min()),minimum_conservative_capture=float(cc[selected].min()),
                fixed_topology_admitted=bool(fixed_selection_gate(rr,cc,selected))))
    commands=np.asarray(init['nominal_controls'][0]).copy()
    for lens,q in zip(decoded[1:4],best):
        item,_,_,bias,slope=lens;commands[item.channel_index]=item.reference_dac+(q-float(bias))/float(slope)
    report=dict(schema='neoarm-last-lens-robust-design-v1',backend=jax.default_backend(),source_hashes=sources,
        original_boundary_design_report=str(original_path.relative_to(root)),original_report_sha256=digest(original_path),
        scope='Margin-aware local image-Fisher design; no noisy recovery, global optimum or hardware-safety proof.',
        random_seed=99137,search_stages=stages,total_screen_candidates=len(candidates),best_candidate_index=best_index,
        selected_relay_q_IL1_IL2_IL3=best.tolist(),synthetic_fixed_relay_dacs=commands.tolist(),
        synthetic_dac_channel_names=column.channel_names,objective_post_q=float(decoded[0][3]),
        fixed_pl1_q=np.asarray(qb+qs*t).tolist(),fixed_selected_state_indices=selected_indices.tolist(),
        input_steps_grouped=np.asarray(inputs).tolist(),incident_electrons_per_frame=dose,
        original_source_canonical_covariance=source.total_covariance.tolist(),
        nominal_selection_rms_minimum=.35,perturbed_admission_rms_minimum=.25,
        minimum_capture=.99,relay_perturbation_rad=.02,source_size_factors=[.9,1.,1.1],
        exact_image_fisher_scenarios=rows,approximate_capture_width_corner_checks=corners,
        all_tested_scenarios_admitted=all(r['admitted'] for r in rows) and all(c['fixed_topology_admitted'] for c in corners),
        limitations=['The topology is fixed across all tested perturbations; no state is silently dropped in a difficult scenario.',
            'The24 simultaneous corner/size combinations check widths/capture only; exact Fisher is evaluated at the explicitly listed9 scenarios.',
            'Finite point and corner checks do not certify the entire continuous parameter box.',
            'Each source-size variation changes angular covariance reciprocally to preserve the200kV pure-mode uncertainty bound.',
            'Image Fisher profiles six image coordinates and19 unconstrained tomography coordinates; assumed Poisson/read-noise variance is evaluated at synthetic truth.',
            'All relay DAC values and excitation bounds remain provisional, not approved NEOARM commands.'])
    if any(digest(root/name)!=value for name,value in sources.items()):raise RuntimeError('Pinned source changed.')
    report['source_hashes_unchanged_during_execution']=True
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--candidate-count',type=int,default=2048);args=parser.parse_args()
    audit(args.output,candidate_count=args.candidate_count)
