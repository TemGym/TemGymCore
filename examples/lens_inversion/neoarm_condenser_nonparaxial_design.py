"""Bounded same-emittance source/autofocus search with nonlinear error gates.

Six symplectic source squeezes and the existing broad controls are explored.
All ten source, three Q and twelve optical coordinates remain free in any
reported information calculation. This is a design ceiling, not a recovery,
image-driven autofocus controller, or hardware preparation claim.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault('JAX_ENABLE_X64','true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import jax
if __name__=='__main__':jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np

from neoarm_condenser_nonparaxial_benchmark import gaussian_phase_space_rule,benchmark_state
from neoarm_condenser_internal_beam_images import model_context,verify,digest,ROOT,FOLDER
from neoarm_condenser_general_source_contract import source_coordinates,quantum_margin
from neoarm_condenser_autofocus_routes import autofocus,independent_roles
from neoarm_condenser_internal_beam_design import LOWER,UPPER,DEFOCUS_OFFSETS,covariance_at,observables_at,capture_lower_bound,information
from neoarm_condenser_extension_design import approximate_covariance_noise,greedy_design
from neoarm_condenser_actual_analyser_contract import PIXEL_M
from neoarm_shared_objective_audit import provenance_snapshot
from temgym_core.constants import energy2wavelength
from temgym_core.neoarm_calibration import DetectorGeometry
from temgym_core.gaussian_image_likelihood import make_gaussian_detector_operator,encode_gaussian_image_parameters,gaussian_image_information_factor

DOSE=4e6


def symplectic_source_squeeze(covariance, position_multiplier):
    s=float(position_multiplier)
    if not np.isfinite(s) or s<=0:raise ValueError('Positive finite source squeeze required.')
    scale=np.diag([s,s,1/s,1/s])
    return scale@np.asarray(covariance)@scale.T


def symplectic_emittances(covariance):
    epsilon=float(energy2wavelength(200e3))/(4*np.pi)
    scales=np.array([5e-9,5e-9,epsilon/5e-9,epsilon/5e-9])
    c=np.asarray(covariance)/np.outer(scales,scales)
    l=np.linalg.cholesky(c)
    omega=np.block([[np.zeros((2,2)),np.eye(2)],[-np.eye(2),np.zeros((2,2))]])
    return np.linalg.eigvalsh(1j*l.T@omega@l)[2:]


def nonlinear_observables(model,p,controls,*,order=4,chunk=128):
    source=np.asarray(model.decode(p)[1]);points,weights=gaussian_phase_space_rule(source,order)
    function=jax.jit(jax.vmap(lambda u: benchmark_state(model,p,u,points,weights,
        body_order='exact',exact_drifts=True)[:3]))
    values=[]
    for start in range(0,len(controls),chunk):
        batch=np.asarray(controls[start:start+chunk]);n=len(batch)
        padded=np.pad(batch,((0,chunk-n),(0,0)),mode='edge')
        values.append(tuple(np.asarray(v)[:n] for v in function(jnp.asarray(padded))))
    return tuple(np.concatenate([v[k] for v in values]) for k in range(3))


def exact_pixel_check(model,p,controls,training):
    covariance=np.asarray(jax.jit(lambda u:covariance_at(model,p,u))(jnp.asarray(controls)))
    detector=DetectorGeometry(mtf_sigma_px=.65)
    pixel=jax.jit(lambda x:gaussian_image_information_factor(x,detector,quadrature_order=32))
    render=jax.jit(make_gaussian_detector_operator(detector,32));noise=[];captures=[]
    for c in covariance.reshape(-1,2,2):
        encoded=encode_gaussian_image_parameters(np.zeros(2),c*PIXEL_M**2,DOSE,detector)
        j=np.asarray(pixel(encoded));_,s,v=np.linalg.svd(j,full_matrices=False)
        factor=v.T/s;noise.append((factor@factor.T)[2:5,2:5])
        captures.append(float(np.asarray(render(np.zeros(2),c*PIXEL_M**2,DOSE)).sum()/DOSE))
    noise=np.asarray(noise).reshape(-1,2,3,3);root=np.linalg.cholesky(noise)
    jac=np.asarray(jax.jit(jax.jacfwd(lambda pp:observables_at(model,pp,jnp.asarray(controls))))(p))
    info,_=information(np.linalg.solve(root[training],jac[training]))
    base=np.asarray(jax.jit(lambda u:observables_at(model,p,u))(jnp.asarray(controls)))
    nonlinear={o:nonlinear_observables(model,p,controls,order=o) for o in (6,10,14)}
    shift=np.linalg.solve(root,(nonlinear[14][1]-base)[...,None])[...,0]
    refinements=[]
    for lo,hi in ((6,10),(10,14)):
        difference=np.linalg.solve(root,(nonlinear[hi][1]-nonlinear[lo][1])[...,None])[...,0]
        refinements.append(dict(orders=[lo,hi],maximum_frame_shift_sigma=float(np.linalg.norm(difference,axis=-1).max())))
    return dict(training_information=info,noise_covariance=noise.tolist(),
        minimum_capture=min(captures),nonlinear_frame_shift_sigma=np.linalg.norm(shift,axis=-1).tolist(),
        maximum_nonlinear_frame_shift_sigma=float(np.linalg.norm(shift,axis=-1).max()),
        ray_quadrature_refinements=refinements,minimum_optical_rms_pixels=float(np.sqrt(np.linalg.eigvalsh(covariance)).min()),
        maximum_Cmini_mechanical_principal_RMS_rad=float(nonlinear[14][2][:,2,2:4].max()),
        accepted_nonlinear_moment_design=bool(np.linalg.norm(shift,axis=-1).max()<.2 and min(captures)>=.999
            and max(r['maximum_frame_shift_sigma'] for r in refinements)<.02 and info['rank']==25),
        quantum_wave_and_fringe_accuracy_accepted=False)


def run(output):
    if jax.default_backend()!='gpu' or not jax.config.x64_enabled:raise RuntimeError('GPU float64 required.')
    started=perf_counter();design_path=FOLDER/'condenser_internal_beam_acquisition_design.json'
    original=json.loads(design_path.read_text());verify(original)
    focus_path=FOLDER/'condenser_internal_beam_design.json';focus=json.loads(focus_path.read_text());verify(focus)
    model,_=model_context();initial=jnp.asarray(original['generating_hypothesis_parameters'])
    source=np.asarray(model.decode(initial)[1]);expected_emittances=symplectic_emittances(source)
    axes=[np.unique(np.r_[np.linspace(LOWER[i],UPPER[i],17),np.asarray(focus['common_focus'])[i]+np.linspace(-.02,.02,9)]) for i in range(2)]
    ab=np.array(list(itertools.product(*axes)))
    hashes,_=provenance_snapshot();hashes.update(original['source_hashes'])
    for path in (Path(__file__),ROOT/'tests/test_neoarm_condenser_nonparaxial_design.py',design_path,focus_path,
        ROOT/'src/temgym_core/nonparaxial_solenoid.py',ROOT/'examples/lens_inversion/neoarm_condenser_nonparaxial_benchmark.py'):
        hashes[str(path.relative_to(ROOT))]=digest(path)
    noise_function=jax.jit(jax.vmap(jax.vmap(lambda c:approximate_covariance_noise(c,DOSE))))
    rows=[];best=None
    for squeeze in (1.,2.,4.,8.,16.,32.):
        c=symplectic_source_squeeze(source,squeeze);emittances=symplectic_emittances(c)
        if not np.allclose(emittances,expected_emittances,rtol=1e-10) or quantum_margin(c)<=0:
            raise RuntimeError('Source preparation must preserve emittance and physicality.')
        p=initial.at[12:22].set(jnp.asarray(source_coordinates(c)))
        cm=autofocus(model,p,ab)
        controls=np.array([[a,b,cc+dc] for (a,b),cc in zip(ab,cm) for dc in DEFOCUS_OFFSETS])
        covariance=np.asarray(jax.jit(lambda u:covariance_at(model,p,u))(jnp.asarray(controls)))
        widths=np.sqrt(np.linalg.eigvalsh(covariance));capture=np.asarray(capture_lower_bound(covariance))
        mask=np.all((controls>=LOWER)&(controls<=UPPER),axis=1)&(widths[:,:,0].min(axis=1)>=.4)&(capture.min(axis=1)>=.999)
        candidate=controls[mask]
        row=dict(source_position_multiplier=squeeze,parameters=np.asarray(p).tolist(),source_covariance=c.tolist(),
            quantum_margin=quantum_margin(c),symplectic_emittances_over_quantum_minimum=emittances.tolist(),
            candidate_count=len(controls),linear_capture_and_resolution_admitted_count=len(candidate),
            nonlinear_screen_admitted_count=0,all_admitted_information=None,selected=None)
        if len(candidate):
            cov=covariance[mask];noise=np.asarray(noise_function(jnp.asarray(cov)));root=np.linalg.cholesky(noise)
            base=np.asarray(jax.jit(lambda u:observables_at(model,p,u))(jnp.asarray(candidate)))
            nonlin=nonlinear_observables(model,p,candidate)
            shifts=np.linalg.solve(root,(nonlin[1]-base)[...,None])[...,0]
            error=np.linalg.norm(shifts,axis=-1).max(axis=-1)
            gate=np.isfinite(error)&(error<.1)
            row.update(nonlinear_error_screen_min_median_max_sigma=np.nanquantile(error,[0,.5,1]).tolist(),
                nonlinear_screen_admitted_count=int(gate.sum()),
                nonlinear_admitted_controls=candidate[gate].tolist(),
                nonlinear_admitted_screen_sigma=error[gate].tolist(),
                minimum_error_control=candidate[int(np.nanargmin(error))].tolist())
            candidate=candidate[gate];noise=noise[gate]
            if len(candidate)>=16:
                jac=np.asarray(jax.jit(jax.jacfwd(lambda pp:observables_at(model,pp,jnp.asarray(candidate))))(p))
                whitened=np.linalg.solve(np.linalg.cholesky(noise),jac).reshape(len(candidate),6,25)
                info,factor=information(whitened);row['all_admitted_information']=info
                if factor is not None:
                    chosen,gains,_=greedy_design((whitened@factor*1e3)[None],12)
                    selection,roles=independent_roles(candidate,chosen,4)
                    approx,_=information(whitened[chosen])
                    row['selected']=dict(controls=candidate[selection].tolist(),training_mask=roles.tolist(),
                        heldout_mask=(~roles).tolist(),approximate_training_information=approx,greedy_gains=gains)
                    score=max(approx['length_standard_errors_percent'])
                    if best is None or score<best[0]:best=(score,len(rows),p)
        rows.append(row)
        print('nonparaxial design',squeeze,'linear',row['linear_capture_and_resolution_admitted_count'],
            'nonlinear',row['nonlinear_screen_admitted_count'],row.get('nonlinear_error_screen_min_median_max_sigma'),flush=True)
    if best is not None:
        _,index,p=best;selected=rows[index]['selected']
        selected['finite_pixel_and_refined_ray_check']=exact_pixel_check(model,p,np.asarray(selected['controls']),np.asarray(selected['training_mask']))
    report=dict(schema='neoarm-same-emittance-nonlinear-condenser-design-v1',source_hashes=hashes,
        backend=jax.default_backend(),x64_enabled=True,elapsed_seconds=perf_counter()-started,
        source_squeeze_grid=[1.,2.,4.,8.,16.,32.],CL1_CL3_axes=[a.tolist() for a in axes],
        defocus_offsets=DEFOCUS_OFFSETS.tolist(),Cmini_bounds=[LOWER[2],UPPER[2]],
        initial_source_symplectic_emittances_over_quantum_minimum=expected_emittances.tolist(),rows=rows,
        best_source_row=None if best is None else best[1],incident_electrons_per_frame=DOSE,
        all_twenty_five_parameters_free_in_information=True,source_emittance_not_a_fit_prior=True,
        source_Q_and_optical_truth_granted_to_future_fit=False,
        theoretical_model_autofocus_not_validated_image_controller=True,
        nonlinear_model_scope='Exact ideal condenser bodies and drifts, thin Q unchanged; objective/projector maps still linear.',
        nonlinear_screen='4^4 positive Gaussian phase-space quadrature; <.1 approximate covariance-noise sigma; selected design retested6/10/14 and finite-pixel noise <.2sigma.',
        image_rendering_performed=False,optical_recovery_performed=False,hardware_admitted=False,
        continuous_control_path_validated=False,global_source_preparation_infeasibility_proved=False,
        limitations=['Only a six-point one-parameter symplectic source-preparation family is tested; no arbitrary source covariance or emittance reduction is granted.',
            'The same unknown after-CL1 Q remains; all ten source moments, Q position/strength and twelve optics remain free for information.',
            'Low noise-weighted classical moment error is necessary diagnostic evidence here, not complete nonlinear wave/fringe accuracy.',
            'Controls admitted at isolated sampled vertices do not establish continuous autofocus paths or robustness to preparation error.',
            'A failed finite search cannot prove a physical impossibility; larger source-transform families or different analyser maps remain alternatives.'])
    verify(report);output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    run(parser.parse_args().output)
