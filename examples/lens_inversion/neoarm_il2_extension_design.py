"""Conditional three-projector design, not three-projector optical recovery.

The downstream IL3/PL1 map comes from measured-image response fitting. IL2 is
not yet calibrated: explicitly provisional factorizations preserve the fitted
nominal map while assessing connected new scans. All17 optical coordinates
enter information calculations; the old responses are included exactly once.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import os
import time
import tomllib

os.environ.setdefault('JAX_ENABLE_X64', 'true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import jax
import jax.numpy as jnp
import numpy as np

from neoarm_coupled_projector_fit import (
    CoupledProjectorResponse, stage_coordinates, radial_stage, grouped_stage,
    physical_upstream_coordinates, block_diagonal,
)
from neoarm_last_lens_physical_fit import RoundLensResponse
from neoarm_last_lens_design import make_relay_model, GROUPED
from neoarm_last_lens_images import signed_response_from_image_fits
from neoarm_projector_branch_experiment import pure_source, signed_inputs, approximate_response_covariance
from neoarm_imaging_recipe import column_with_geometry_anchors
from neoarm_last_lens_tomography import J2, rotation, mechanical_shear
from temgym_core.gaussian_image_likelihood import (
    make_gaussian_detector_operator, gaussian_image_information_factor, encode_gaussian_image_parameters,
)
from temgym_core.neoarm_calibration import DetectorGeometry, hard_edge_column_from_config

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT/'examples/generated/neoarm_inversion/imaging_recipe'
PHYSICAL_NAMES = tuple(f'{lens}.{coordinate}' for lens in ('PL1','IL3','IL2')
    for coordinate in ('log_length_m','log_gap_m','q_bias','q_slope'))
PARAMETER_NAMES = PHYSICAL_NAMES+('prefix_log_row_norm','prefix_row_angle','prefix_row_shear',
    'prefix_rotation_rad','sample_kappa_times_balance_length')


class ThreeProjectorResponse:
    def __init__(self, calibration):
        self.calibration = calibration

    def upstream_matrix(self, parameters, *, coordinates='mechanical'):
        return self.calibration.upstream_matrix(jnp.r_[jnp.zeros(4),jnp.asarray(parameters)[12:]], coordinates=coordinates)

    def predict(self, parameters, controls):
        p, c = jnp.asarray(parameters), jnp.asarray(controls)
        if p.shape != (17,) or c.ndim != 2 or c.shape[1] != 3:
            raise ValueError('Expected17 parameters and controls (PL1,IL3,IL2).')
        radial = self.calibration.radial_matrix(jnp.r_[jnp.zeros(4),p[12:]])
        matrix = radial_stage(p[:4],c[:,0])@radial_stage(p[4:8],c[:,1])@radial_stage(p[8:12],c[:,2])@radial
        aa, bb = matrix[:,0,0], matrix[:,0,1]
        position = aa[:,None,None]*jnp.eye(2)+(p[16]/self.calibration.balance_length_m)*bb[:,None,None]*J2
        angle = bb[:,None,None]*jnp.eye(2)
        phase = p[15]-sum(p[4*i+2]+p[4*i+3]*c[:,i] for i in range(3))
        response = rotation(phase)@jnp.concatenate((position,angle),axis=-1)
        return response*(jnp.asarray(self.calibration.input_steps_grouped)/self.calibration.pixel_m)

    def canonical_maps(self, parameters, controls):
        p = jnp.asarray(parameters)
        prefix = self.upstream_matrix(p, coordinates='canonical')
        return jax.vmap(lambda c:grouped_stage(p[:4],c[0])@grouped_stage(p[4:8],c[1])
            @grouped_stage(p[8:12],c[2])@prefix)(jnp.asarray(controls))


def extend_nominal_fitted_map(two_model, two_parameters, provisional_il2):
    """IL2 is a design scenario; its nominal decomposition adds four gauges."""
    stages = np.asarray(stage_coordinates(two_parameters))
    nominal = np.asarray(two_model.upstream_matrix(two_parameters))
    prefix = np.linalg.solve(np.asarray(grouped_stage(jnp.asarray(provisional_il2),0.)),nominal)
    return np.r_[stages, provisional_il2, physical_upstream_coordinates(prefix,two_model.calibration)]


def connected_component(mask, zero_index):
    mask = np.asarray(mask, bool)
    if not mask[zero_index]:
        return None
    left = right = int(zero_index)
    while left > 0 and mask[left-1]:left -= 1
    while right+1 < len(mask) and mask[right+1]:right += 1
    return left, right


def fisher_summary(jacobian, covariance):
    derivative = np.asarray(jacobian).reshape(-1,17)
    white = np.linalg.solve(np.linalg.cholesky(covariance),derivative)
    norms = np.linalg.norm(white,axis=0)
    _, singular, right = np.linalg.svd(white/np.where(norms>0,norms,1.),full_matrices=False)
    rank = int(np.sum(singular>singular[0]*1e-10))
    result = dict(rank=rank,free_coordinate_count=17,normalized_singular_values=singular.tolist(),
        unexcited_coordinate_indices=np.flatnonzero(norms==0).tolist(),parameter_covariance=None)
    if rank == 17:
        factor = (right.T/singular)/norms[:,None]
        result.update(parameter_covariance=(factor@factor.T).tolist(),
            length_gap_standard_errors_percent=(100*np.linalg.norm(factor[[0,1,4,5,8,9]],axis=1)).tolist(),
            excitation_standard_errors=np.linalg.norm(factor[[2,3,6,7,10,11]],axis=1).tolist())
    return result


def moments(model, parameters, controls, source_covariance):
    p = jnp.asarray(parameters)
    maps = model.canonical_maps(p,controls)
    covariance = jnp.asarray(source_covariance)[np.ix_(GROUPED,GROUPED)]
    cov = maps[:,:2]@covariance@maps[:,:2].swapaxes(-1,-2)
    mechanical = maps@mechanical_shear(p[16]/model.calibration.balance_length_m)
    means = jnp.einsum('nai,fi->nfa',mechanical[:,:2],jnp.asarray(signed_inputs(model.calibration.input_steps_grouped)))
    return means,cov


def downstream_pair_matrix(physical_eight, controls_pair):
    p = jnp.asarray(physical_eight)
    return jax.vmap(lambda c:(grouped_stage(p[:4],c[0])@grouped_stage(p[4:],c[1]))[:2])(
        jnp.asarray(controls_pair)).reshape(4,4)


def reconstruct_with_shared_calibration(response_scan, covariance, physical_eight, analyser_covariance, controls_pair, steps, pixel):
    """Prospective new independent exposures; keep shared8-coordinate calibration."""
    shape = np.asarray(response_scan).shape
    def recover(data, p):
        analyser = downstream_pair_matrix(p,controls_pair)
        physical = data.reshape(-1,4,4)*(pixel/jnp.asarray(steps))[None,None,:]
        return jax.vmap(lambda matrix:jnp.linalg.solve(analyser,matrix))(physical).reshape(-1)
    data, p = jnp.asarray(response_scan).reshape(-1),jnp.asarray(physical_eight)
    values = np.asarray(recover(data,p)).reshape(-1,4,4)
    jd,jp = map(np.asarray,jax.jacfwd(recover,argnums=(0,1))(data,p))
    measurement = jd@covariance@jd.T
    shared = jp@analyser_covariance@jp.T
    return dict(maps=values.tolist(),measurement_covariance=measurement.tolist(),shared_calibration_covariance=shared.tolist(),
        joint_covariance=(measurement+shared).tolist(),response_shape=list(shape),
        calibration_noise_independent_for_proposed_new_exposures=True,
        final_fit_must_use_original_responses_not_this_reconstruction_as_another_prior=True)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(output):
    if jax.default_backend()!='gpu' or not jax.config.x64_enabled:
        raise RuntimeError('Forward design and finite-pixel information use GPU float64.')
    tick=time.perf_counter();reference_path=FOLDER/'projector_branch_recovery.json'
    reference=json.loads(reference_path.read_text());sources=dict(reference['source_hashes'])
    for name,value in sources.items():
        if digest(ROOT/name)!=value:raise ValueError(f'Stale downstream calibration:{name}')
    if reference['best'] is None or reference['information']['numerical_rank']!=13:
        raise ValueError('A full-rank conditional downstream fit is required.')
    # Read only the provisional model for UNCALIBRATED IL2. The downstream
    # physical stages and nominal map come exclusively from the image fit.
    init_path=FOLDER/'measured_initialization.json';robust_path=FOLDER/'last_lens_robust_design.json'
    config_path=ROOT/'examples/lens_inversion/neoarm_inversion_config.toml'
    geometry_path=ROOT/'examples/microscope_models/measure_microscope_geometry.ipynb'
    initial=json.loads(init_path.read_text());robust=json.loads(robust_path.read_text())
    column=column_with_geometry_anchors(hard_edge_column_from_config(tomllib.loads(config_path.read_text())),geometry_path)
    _,decoded=make_relay_model(column,jnp.asarray(initial['parameters']),0.)
    lens2,z2,length2,_,slope2=decoded[-3];_,z3,length3,_,_=decoded[-2]
    if lens2.name!='IL2':raise ValueError('The provisional model ordering is unexpected.')
    provisional=np.array([np.log(float(length2)),np.log(float(z3-length3/2-z2-length2/2)),
        robust['selected_relay_q_IL1_IL2_IL3'][1],float(slope2)])
    # The inference manifest is the authoritative input-calibration contract.
    manifest_path=FOLDER/'projector_branch_experiment_inference.json';manifest=json.loads(manifest_path.read_text())
    calibration=RoundLensResponse(tuple(manifest['input_steps_grouped']),manifest['detector_pixel_m'])
    two_model=CoupledProjectorResponse(calibration);model=ThreeProjectorResponse(calibration)
    p=extend_nominal_fitted_map(two_model,reference['best']['parameters'],provisional)
    old_controls=np.column_stack((reference['training_controls'],np.zeros(len(reference['training_controls']))))
    old_covariance=np.asarray(reference['training_response_covariance'])
    old_jac=np.asarray(jax.jit(jax.jacfwd(lambda pp:model.predict(pp,old_controls)))(p))
    old_info=fisher_summary(old_jac,old_covariance)
    closure=float(np.max(abs(np.asarray(model.predict(p,old_controls))-
        np.asarray(two_model.predict(reference['best']['parameters'],np.asarray(reference['training_controls']))))))
    # Only already calibrated downstream settings are candidates. Their use
    # as future fresh exposures does not reuse old image noise a second time.
    analysers=np.unique(np.asarray(reference['training_controls']),axis=0)
    grid=np.linspace(-.75,.75,61);zero=int(np.argmin(abs(grid)))
    grid_controls=np.array([[*a,u] for a in analysers for u in grid])
    source=pure_source();pixel=calibration.pixel_m;dose=1e6
    source_covariance=jnp.asarray(source.total_covariance)
    moment_fun=jax.jit(lambda pp,c:moments(model,pp,c,source_covariance))
    means,cov=map(np.asarray,moment_fun(jnp.asarray(p),jnp.asarray(grid_controls)))
    rms=np.sqrt(np.linalg.eigvalsh(cov))/pixel
    marginal=np.sqrt(np.diagonal(cov,axis1=1,axis2=2))/pixel
    tail=np.asarray(jax.scipy.special.erfc((125.-abs(means/pixel))/(np.sqrt(2)*marginal[:,None,:])))
    capture=np.maximum(0.,1.-np.sum(tail,axis=2))
    admitted=((rms[:,0]>=.30)&(capture.min(axis=1)>=.99)).reshape(len(analysers),len(grid))
    jac_fun=jax.jit(jax.jacfwd(lambda pp,c:model.predict(pp,c),argnums=0))
    candidates=[]
    for a,b in itertools.combinations(range(len(analysers)),2):
        component=connected_component(admitted[a]&admitted[b],zero)
        if component is None:continue
        left,right=component
        if grid[left]>-.05 or grid[right]<.05:continue
        pair=analysers[[a,b]]
        analyser=np.asarray(downstream_pair_matrix(p[:8],pair))
        condition=float(np.linalg.cond(analyser*np.array([1.,1.,100.,100.])))
        if not np.isfinite(condition) or condition>1e4:continue
        train_offsets=np.array([grid[left],0.,grid[right]])
        controls=np.array([[*a,u] for u in train_offsets for a in pair])
        m,c=map(np.asarray,moment_fun(jnp.asarray(p),jnp.asarray(controls)))
        rcov=approximate_response_covariance(c,pixel,dose)
        jac=np.asarray(jac_fun(jnp.asarray(p),jnp.asarray(controls)))
        info=fisher_summary(np.concatenate((old_jac,jac)),block_diagonal([old_covariance,*rcov]))
        if info['rank']!=17:continue
        candidates.append(dict(analyser_indices=[a,b],downstream_controls=pair.tolist(),
            connected_grid_interval=[float(grid[left]),float(grid[right])],training_il2_native_offsets=train_offsets.tolist(),
            balanced_analyser_condition=condition,approximate_information=info,
            score=max(info['length_gap_standard_errors_percent'][-2:])))
    candidates.sort(key=lambda row:row['score'])
    print('IL2 candidate pairs',len(candidates),'best approximate',None if not candidates else candidates[0]['score'],flush=True)
    detector=DetectorGeometry(z_m=float(initial['parameters'][1]),mtf_sigma_px=.65)
    operator=jax.jit(make_gaussian_detector_operator(detector,16))
    factor_fun=jax.jit(lambda value:gaussian_image_information_factor(value,detector,quadrature_order=16))
    validated=[]
    for candidate in candidates[:3]:
        pair=np.asarray(candidate['downstream_controls']);train_offsets=np.asarray(candidate['training_il2_native_offsets'])
        held_offset=(train_offsets[2] if abs(train_offsets[2])>=abs(train_offsets[0]) else train_offsets[0])/2
        offsets=np.sort(np.r_[train_offsets,held_offset]);mask=~np.isclose(offsets,held_offset,atol=1e-12,rtol=0)
        controls=np.array([[*a,u] for u in offsets for a in pair])
        m,c=map(np.asarray,moment_fun(jnp.asarray(p),jnp.asarray(controls)))
        response_cov=[];captures=[];ranks=[]
        for mm,cc in zip(m,c):
            pars=[];covs=[]
            for mean in mm:
                encoded=encode_gaussian_image_parameters(mean,cc,dose,detector)
                factor=np.asarray(factor_fun(encoded));_,ss,rr=np.linalg.svd(factor,full_matrices=False)
                ranks.append(int(np.sum(ss>ss[0]*1e-10)))
                inverse=(rr.T/ss);covs.append(inverse@inverse.T);pars.append(np.asarray(encoded))
                captures.append(float(np.asarray(operator(mean,cc,dose)).sum()/dose))
            response_cov.append(signed_response_from_image_fits(pars,covs)[1])
        select=np.repeat(mask,2);new_jac=np.asarray(jac_fun(jnp.asarray(p),jnp.asarray(controls[select])))
        exact=fisher_summary(np.concatenate((old_jac,new_jac)),block_diagonal([old_covariance,*np.asarray(response_cov)[select]]))
        reconstruction=reconstruct_with_shared_calibration(np.asarray(model.predict(p,controls[select])).reshape(3,2,2,4),
            block_diagonal(list(np.asarray(response_cov)[select])),p[:8],np.asarray(reference['information']['physical_covariance']),
            pair,calibration.input_steps_grouped,pixel)
        row=dict(candidate,il2_native_offsets=offsets.tolist(),training_mask=mask.tolist(),heldout_mask=(~mask).tolist(),
            new_optical_setting_count=8,new_frame_count=72,new_training_frame_count=54,new_heldout_frame_count=18,
            exact_finite_pixel_information=exact,minimum_capture=min(captures),minimum_image_information_rank=min(ranks),
            minimum_optical_rms_pixels=float(np.sqrt(np.linalg.eigvalsh(c)).min()/pixel),
            maximum_optical_rms_pixels=float(np.sqrt(np.linalg.eigvalsh(c)).max()/pixel),
            response_covariance=np.asarray(response_cov).tolist(),initialization_reconstruction=reconstruction,
            admitted=bool(exact['rank']==17 and min(ranks)==6 and min(captures)>=.99 and np.sqrt(np.linalg.eigvalsh(c)).min()/pixel>=.30))
        validated.append(row);print('Exact IL2',row['analyser_indices'],exact.get('length_gap_standard_errors_percent'),flush=True)
    selected=min((row for row in validated if row['admitted']),
        key=lambda row:max(row['exact_finite_pixel_information']['length_gap_standard_errors_percent'][-2:]),default=None)
    for path in (Path(__file__),reference_path,init_path,robust_path,config_path,geometry_path,manifest_path):
        sources[str(path.relative_to(ROOT))]=digest(path)
    if any(digest(ROOT/name)!=value for name,value in sources.items()):raise RuntimeError('A design dependency changed during evaluation.')
    report=dict(schema='neoarm-IL2-extension-design-v1',backend=jax.default_backend(),source_hashes=sources,
        source_hashes_unchanged_during_execution=True,elapsed_seconds=time.perf_counter()-tick,
        downstream_fit_path=str(reference_path.relative_to(ROOT)),parameter_names=PARAMETER_NAMES,
        provisional_IL2_physical_coordinates=provisional.tolist(),provisional_seventeen_coordinates=p.tolist(),
        provisional_IL2_is_not_a_calibration=True,nominal_response_closure_max_pixels=closure,
        original_response_coordinate_count=224,original_information=old_info,
        original_length_gap_marginal_standard_errors_percent=(100*np.sqrt(np.diag(
            np.asarray(reference['information']['physical_covariance'])))[[0,1,4,5]]).tolist(),
        connected_scan_grid_native=grid.tolist(),downstream_candidate_controls=analysers.tolist(),
        tested_candidate_pair_count=len(candidates),candidate_shortlist=candidates[:10],validated_candidates=validated,selected=selected,
        all_seventeen_optical_coordinates_free=True,calibration_posterior_added_as_prior=False,
        optical_recovery_demonstrated=False,new_images_generated=False,hardware_commands_admitted=False,
        existing_heldout_responses_used=False,
        limitations=['IL2 length/gap/excitation are uncalibrated rough-model design values, while the nominal fitted map is preserved exactly.',
            'Prior224 raw response coordinates are counted once; their uncertainty propagates through all17 free coordinates.',
            'A proposed fresh zero-offset pair makes initial map reconstruction independent of the old calibration image noise.',
            'The shared8-coordinate downstream calibration covariance induces off-diagonal correlations in reconstructed maps; it is not an extra fit prior.',
            'Connected means61 finite admission checks across the local control interval, not a continuous guarantee or safe hardware range.',
            'Finite-pixel Fisher uses all6 image coordinates; the fixed pure canonical source and exact input/pixel/PSF calibrations remain conditional.',
            'The proposed heldout offset contributes no information to the training score. Existing heldouts were not inspected.',
            'No new noisy images, nonlinear three-lens recovery, excitation-branch proof or1% precision claim is made.'])
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=audit(args.output)
    print('Saved',args.output,'candidate admitted',result['selected'] is not None,flush=True)
