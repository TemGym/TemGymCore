"""Nonparaxial condenser-body sensitivity at accepted first-order fits.

No parameter fitting or held-out observation selection occurs. The accepted
23/29-coordinate fits remain valid fits of their stated first-order model.
This diagnostic asks a separate question about its body/drift approximation.
The invisible covariance family is sampled, not granted as measured source.
"""
import argparse
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

from neoarm_condenser_covariance_fit import VisibleSourceModel,quantum_completion,visible_covariance,SOURCE_SCALE,INVISIBLE
from neoarm_condenser_extension_design import load_context,FOLDER
from neoarm_condenser_extension_recovery import selected_data
from neoarm_condenser_endpoint_calibration import EndpointRows,GROUPED,digest
from neoarm_condenser_endpoint_recovery import verify
from neoarm_condenser_cold_joint_endpoint import select_cold_seed
from neoarm_condenser_nonparaxial_benchmark import gaussian_phase_space_rule,benchmark_state,weighted_covariance,covariance_observable
from neoarm_condenser_image_recovery import rotation
from neoarm_condenser_nonparaxial_design import symplectic_emittances
from neoarm_aperture_capture import ROOT
from neoarm_shared_objective_audit import provenance_snapshot
from temgym_core.nonparaxial_solenoid import uniform_solenoid_ray


class AcceptedCondenserAdapter:
    def __init__(self,conditional,optical,source,analyser_rows):
        self.source_z_m=conditional.source_z_m
        self.reference_z_m=conditional.objective_entrance_m
        self.aperture_z_m=conditional.aperture_z_m
        self.insert_before=3 # No Q anywhere in this model.
        self.optical=jnp.asarray(optical).reshape(3,4)
        self.source=jnp.asarray(source)
        self.analyser_maps=jnp.asarray(analyser_rows)
    def decode(self,p):return self.optical,self.source,0.,jnp.eye(4)


def aperture_points(adapter,control,points,*,exact=False):
    state=points[:,jnp.asarray(GROUPED)];cursor=adapter.source_z_m
    order='exact' if exact else 'paraxial'
    for index in range(2):
        z,log_length,bias,slope=adapter.optical[index];length=jnp.exp(log_length)
        state=uniform_solenoid_ray(state,0.,z-length/2-cursor,order=order)
        state=uniform_solenoid_ray(state,(bias+slope*control[index])/length,length,order=order)
        cursor=z+length/2
    state=uniform_solenoid_ray(state,0.,adapter.aperture_z_m-cursor,order=order)
    return state[:,jnp.asarray([0,2])]


def run(output):
    if jax.default_backend()!='gpu' or not jax.config.x64_enabled:raise RuntimeError('GPU float64 required.')
    started=perf_counter();moment_path=FOLDER/'condenser_moment_initialization.json'
    joint_path=FOLDER/'condenser_cold_joint_endpoint.json';cal_path=FOLDER/'condenser_endpoint_calibration.json'
    extension_path=FOLDER/'condenser_extension_images.json'
    moment,joint,cal,extension=[json.loads(path.read_text()) for path in (moment_path,joint_path,cal_path,extension_path)]
    for report in (moment,joint,cal,extension):verify(report)
    if not joint['accepted_stationary_fit']:raise ValueError('An accepted first-order joint fit is required.')
    p23,_=select_cold_seed(moment);p29=np.asarray(joint['parameters'])
    _,arrays,_,aperture,conditional=load_context();quotient=VisibleSourceModel(conditional)
    extension_array=ROOT/extension['arrays_path']
    if digest(extension_array)!=extension['arrays_sha256']:raise ValueError('Training array changed.')
    with np.load(extension_array,allow_pickle=False) as archive:fresh=selected_data(archive,True)
    controls=np.concatenate((arrays['controls'][:8],fresh[0]))
    root=np.linalg.cholesky(np.concatenate((arrays['beam_observable_covariances'][:8],fresh[2])))
    ac=np.asarray([row['controls'] for row in aperture['records'][:2]])
    ar=np.linalg.cholesky(arrays['aperture_observable_covariances'][:2])
    split=conditional.column.objective.split_maps(jnp.asarray(conditional.fixed[-4:]))
    endpoints=EndpointRows(cal['conditional_sample_kappa_per_m'])
    native_old=conditional.downstream@conditional.prefield
    rows_old=native_old[:,jnp.asarray([0,2])][:,:,jnp.asarray(GROUPED)]
    native_new=endpoints.canonical_rows(p29[23:])[:,:,jnp.asarray(GROUPED)]@split.pre
    rows_new=native_new[:,:,jnp.asarray(GROUPED)]
    hashes,_=provenance_snapshot()
    for r in (moment,joint,cal,extension):hashes.update(r['source_hashes'])
    for path in (Path(__file__),ROOT/'tests/test_neoarm_condenser_accepted_nonparaxial_audit.py',moment_path,joint_path,cal_path,extension_path,extension_array):
        hashes[str(path.relative_to(ROOT))]=digest(path)
    results=[]
    for label,p,rows in [('accepted_23',p23,rows_old),('accepted_joint_29',p29[:23],rows_new)]:
        optical=np.asarray(quotient.decode(p)[0]);completion=quantum_completion(p[12:21],interval=True)
        g0=completion['representative'];lo,hi=completion['feasible_interval']
        family=[]
        for fraction in (-.8,0.,.8):
            g=g0+abs(fraction)*((lo if fraction<0 else hi)-g0)
            covariance=np.asarray(visible_covariance(p[12:21]))+g*INVISIBLE*np.outer(SOURCE_SCALE,SOURCE_SCALE)
            source=covariance[np.ix_(GROUPED,GROUPED)]
            adapter=AcceptedCondenserAdapter(conditional,optical,source,rows)
            orders=(6,10,14) if fraction==0 else (10,)
            values={}
            for order in orders:
                points,weights=gaussian_phase_space_rule(source,order)
                para=jax.jit(jax.vmap(lambda u:benchmark_state(adapter,None,u,points,weights)[:3]))(jnp.asarray(controls[:,:3]))
                exact=jax.jit(jax.vmap(lambda u:benchmark_state(adapter,None,u,points,weights,body_order='exact',exact_drifts=True)[:3]))(jnp.asarray(controls[:,:3]))
                def ap(u,ex):
                    point=aperture_points(adapter,u,points,exact=ex)@rotation(p[21]).T*jnp.exp(-p[22])
                    return covariance_observable(weighted_covariance(point,weights))
                ap0=jax.jit(jax.vmap(lambda u:ap(u,False)))(jnp.asarray(ac[:,:3]))
                ap1=jax.jit(jax.vmap(lambda u:ap(u,True)))(jnp.asarray(ac[:,:3]))
                values[order]=(tuple(map(np.asarray,para)),tuple(map(np.asarray,exact)),np.asarray(ap0),np.asarray(ap1))
            before,after,ap_before,ap_after=values[orders[-1]]
            shift=np.linalg.solve(root,(after[1]-before[1])[...,None])[...,0]
            ap_shift=np.linalg.solve(ar,(ap_after-ap_before)[...,None])[...,0]
            refinements=[]
            for low,high in zip(orders[:-1],orders[1:]):
                delta=np.linalg.solve(root,(values[high][1][1]-values[low][1][1])[...,None])[...,0]
                refinements.append(dict(orders=[low,high],maximum_frame_observable_change_sigma=float(np.linalg.norm(delta,axis=-1).max())))
            family.append(dict(invisible_family_fraction=fraction,invisible_coordinate=g,
                source_symplectic_emittances_over_quantum_minimum=symplectic_emittances(source).tolist(),
                source_rms_grouped=np.sqrt(np.diag(source)).tolist(),
                frame_observable_change_sigma=np.linalg.norm(shift,axis=-1).tolist(),
                frame_sigma_min_median_max=np.quantile(np.linalg.norm(shift,axis=-1),[0,.5,1]).tolist(),
                aperture_observable_change_sigma=np.linalg.norm(ap_shift,axis=-1).tolist(),
                maximum_mechanical_principal_RMS_by_body_rad=after[2][:,:,2:4].max(axis=(0,2)).tolist(),
                paraxial_covariance_pixels=before[0].tolist(),nonparaxial_covariance_pixels=after[0].tolist(),
                quadrature_refinements=refinements,
                all_sampled_frame_changes_below_point2sigma=bool(np.linalg.norm(shift,axis=-1).max()<.2 and np.linalg.norm(ap_shift,axis=-1).max()<.2)))
        results.append(dict(label=label,accepted_first_order_parameters=np.asarray(p).tolist(),
            downstream_position_rows_grouped=np.asarray(rows).tolist(),invisible_covariance_interval=[float(lo),float(hi)],
            sampled_covariance_family=family))
        print('accepted condenser nonparaxial',label,family[1]['frame_sigma_min_median_max'],family[1]['aperture_observable_change_sigma'],flush=True)
    report=dict(schema='neoarm-accepted-condenser-nonparaxial-sensitivity-v1',source_hashes=hashes,
        backend=jax.default_backend(),x64_enabled=True,elapsed_seconds=perf_counter()-started,
        beam_training_controls=controls.tolist(),aperture_training_controls=ac.tolist(),
        beam_noise_covariance=np.einsum('...ij,...kj->...ik',root,root).tolist(),
        aperture_noise_covariance=np.einsum('...ij,...kj->...ik',ar,ar).tolist(),cases=results,
        fit_parameters_changed=False,heldout_observations_loaded=False,
        first_order_statistical_fit_invalidated=False,paraxial_physical_accuracy_tested_separately=True,
        objective_projector_nonparaxial_transport_included=False,real_fringe_fields_included=False,
        quantum_wave_transport_validation=False,hardware_validity_proved=False,
        limitations=['This tests the accepted parameters and their actual training controls/noise, not only the separate internal-Q hypothesis.',
            'Three physical invisible-covariance representatives are sampled; nonlinear transport can expose a covariance direction invisible to the fitted first-order model.',
            'Exact uniform-body and drift classical quadrature is not the full nonlinear Maxwell/wave microscope.',
            'The objective and downstream maps remain linear/conditional. Their nonlinear effects may reinforce or compensate these shifts.',
            'Aperture numbers are phase-space spatial-covariance sensitivity at the aperture plane, not an independent finite-sensor collector rendering.',
            'Successful fitting and held-out tests of the original first-order synthetic model do not certify its nonparaxial physical approximation.'])
    verify(report);output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    run(parser.parse_args().output)
