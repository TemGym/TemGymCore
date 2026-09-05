"""Fresh condenser measurements selected from the weak original-data directions.

Uses the approximate-geometry, image-initialized source and the unaccepted
training-fit iterate as two provisional design hypotheses. No new noise or
held-out observations choose the design. The downstream/objective calibration
is conditional; controls are synthetic study coordinates, not hardware limits.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import os
import time

os.environ.setdefault('JAX_ENABLE_X64', 'true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import jax
import jax.numpy as jnp
import numpy as np

from neoarm_condenser_covariance_fit import VisibleSourceModel, verified_input, quantum_completion
from neoarm_condenser_image_recovery import CondenserModel, covariance_observable, read_verified_aperture, DEFAULT_APERTURE
from neoarm_aperture_capture import ROOT, XY, load_cases
from temgym_core.gaussian_image_likelihood import (
    make_gaussian_detector_operator, encode_gaussian_image_parameters,
    gaussian_image_information_factor,
)
from temgym_core.neoarm_calibration import DetectorGeometry

FOLDER = ROOT/'examples/generated/neoarm_inversion/imaging_recipe'


def approximate_covariance_noise(optical_covariance_pixels, dose):
    """Full-camera moment approximation used only for candidate ranking."""
    c = jnp.asarray(optical_covariance_pixels)
    broadened = c+jnp.eye(2)*(.65**2+1/12)
    indices = ((0, 0), (0, 1), (1, 1))
    wishart = jnp.array([[(broadened[i,k]*broadened[j,l]+broadened[i,l]*broadened[j,k])/dose
                         for k,l in indices] for i,j in indices])
    def observable(v):
        return covariance_observable(jnp.array([[v[0],v[1]],[v[1],v[2]]]))
    derivative = jax.jacfwd(observable)(c[jnp.array([0,0,1]),jnp.array([0,1,1])])
    return derivative@wishart@derivative.T


def covariance_factor(jacobian):
    """Untruncated inverse-information factor; reject extra null directions."""
    j = np.asarray(jacobian)
    scales = np.linalg.norm(j, axis=0)
    _, s, v = np.linalg.svd(j/scales, full_matrices=False)
    if s[-1] <= s[0]*1e-10:
        raise ValueError('This design requires an explicit full-rank quotient and calibration prior.')
    return (v.T/s)/scales[:,None]


def greedy_design(candidate_factors, count):
    """Maximin D gain over provisional hypotheses, in baseline-whitened units."""
    factors = np.asarray(candidate_factors)
    hypotheses, candidates, _, parameters = factors.shape
    if not 0 < count <= candidates:
        raise ValueError('The requested batch must fit the candidate pool.')
    information = np.tile(np.eye(parameters), (hypotheses,1,1))
    selected, scores = [], []
    for _ in range(count):
        inverse = np.linalg.inv(information)
        innovation = np.einsum('hcai,hij,hcbj->hcab', factors, inverse, factors)
        gain = np.linalg.slogdet(innovation+np.eye(factors.shape[2]))[1]
        robust = gain.min(axis=0)
        robust[selected] = -np.inf
        chosen = int(np.argmax(robust))
        selected.append(chosen); scores.append(float(robust[chosen]))
        information += np.einsum('hai,haj->hij', factors[:,chosen], factors[:,chosen])
    return np.asarray(selected), scores, information


def load_context():
    old, arrays = verified_input(FOLDER/'condenser_image_recovery.json')
    repair_path = FOLDER/'condenser_covariance_fit.json'
    repair, _ = verified_input(repair_path)
    aperture = read_verified_aperture(DEFAULT_APERTURE)
    info_path = next(ROOT/n for n in aperture['report_input_hashes'] if 'information_narrow_probe' in n)
    info = json.loads(info_path.read_text())
    # Only the established conditional downstream/objective and source-plane
    # reference are used here. The fitted condensers are overwritten below.
    _, _, column, fixed = load_cases(info_path)
    analysers = np.asarray(info['projection_states']['controls'])[[0,2]]
    model = CondenserModel(column, fixed, analysers)
    model.analyser_controls = analysers
    return old, arrays, repair, aperture, model


def training_residual(quotient, arrays, aperture):
    bc = jnp.asarray(arrays['controls'][:8])
    ac = jnp.asarray([row['controls'] for row in aperture['records'][:2]])
    b = jnp.asarray(arrays['beam_observables'][:8])
    a = jnp.asarray(arrays['aperture_observables'][:2])
    br = jnp.asarray(np.linalg.cholesky(arrays['beam_observable_covariances'][:8]))
    ar = jnp.asarray(np.linalg.cholesky(arrays['aperture_observable_covariances'][:2]))
    def residual(p):
        bp, ap = quotient.observables(p,bc,ac)
        return jnp.r_[jnp.linalg.solve(br,(bp-b)[...,None]).reshape(-1),
                      jnp.linalg.solve(ar,(ap-a)[...,None]).reshape(-1),p[22]/.005]
    return residual


def moments(quotient, p, controls):
    optical, covariance, _, _ = quotient.decode(p)
    maps = jax.vmap(lambda u: quotient.model.maps(optical,u))(controls)
    output = maps@covariance@maps.swapaxes(-1,-2)
    return output[:,:,XY][:,:,:,XY]/(55e-6)**2


def refocused_candidates(quotient, hypotheses, axes):
    """Bracketed C-mini refinement for each CL1/CL3 pair, before new noise."""
    pairs=np.array(list(itertools.product(*axes[:2])))
    coarse=np.array([[*v,cm,0.,0.,0.,0.] for v in pairs for cm in axes[2]])
    def score(controls):
        cov=jax.vmap(lambda p:moments(quotient,p,controls))(jnp.asarray(hypotheses))
        return jnp.trace(cov,axis1=-2,axis2=-1).sum(axis=(0,2))
    evaluate=jax.jit(score)
    values=np.asarray(evaluate(jnp.asarray(coarse))).reshape(len(pairs),-1)
    index=values.argmin(axis=1)
    lo=np.asarray(axes[2])[np.maximum(index-1,0)]
    hi=np.asarray(axes[2])[np.minimum(index+1,len(axes[2])-1)]
    def controls_at(cm):return np.column_stack((pairs,cm,np.zeros((len(pairs),4))))
    ratio=(np.sqrt(5)-1)/2
    for _ in range(42):
        u=hi-ratio*(hi-lo);v=lo+ratio*(hi-lo)
        fu=np.asarray(evaluate(controls_at(u)));fv=np.asarray(evaluate(controls_at(v)))
        take_left=fu<fv
        hi=np.where(take_left,v,hi);lo=np.where(take_left,lo,u)
    centre=(lo+hi)/2
    candidates=np.concatenate([controls_at(centre+offset) for offset in (-.02,-.01,0.,.01,.02)])
    return candidates[(candidates[:,2]>=axes[2][0])&(candidates[:,2]<=axes[2][-1])]


def information_summary(jacobian, quotient, p):
    factor = covariance_factor(jacobian)
    physical = np.asarray(jax.jacfwd(lambda q: quotient.decode(q)[0])(p))@factor
    return dict(physical_standard_errors=np.linalg.norm(physical,axis=1).reshape(3,4).tolist(),
                condition_number=float(np.linalg.cond(jacobian/np.linalg.norm(jacobian,axis=0))))


def design(output, *, count=8, grid_count=17):
    if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
        raise RuntimeError('Use GPU float64 for the forward design.')
    started=time.perf_counter()
    old, arrays, repair, aperture, model = load_context()
    quotient=VisibleSourceModel(model)
    hypotheses=np.asarray([repair['records'][0][key] for key in ('initial_parameters','parameters')])
    residual=training_residual(quotient,arrays,aperture)
    old_jac=[np.asarray(jax.jit(jax.jacfwd(residual))(p)) for p in hypotheses]
    old_factor=np.asarray([covariance_factor(j) for j in old_jac])
    # Explore stronger first-branch excitations because weak-lens focusing
    # supplies little thickness information. No hardware authority is implied.
    axes=[np.linspace(-.25,2.,grid_count),np.linspace(-.15,2.,grid_count),
          np.linspace(.3,3.5,grid_count+4)]
    controls=refocused_candidates(quotient,hypotheses,axes)
    # Exclude every old acquisition, including the already inspected heldout.
    unseen=np.min(np.linalg.norm(controls[:,None,:3]-arrays['controls'][None,:,:3],axis=-1),axis=1)>1e-8
    covariance_fun=jax.jit(moments,static_argnums=0)
    all_cov=np.asarray([covariance_fun(quotient,jnp.asarray(p),jnp.asarray(controls)) for p in hypotheses])
    rms=np.sqrt(np.linalg.eigvalsh(all_cov))
    marginal=np.sqrt(np.diagonal(all_cov,axis1=-2,axis2=-1)+.65**2)
    capture=1-np.asarray(jax.scipy.special.erfc(125./(np.sqrt(2)*marginal))).sum(axis=-1)
    admitted=unseen&(rms.min(axis=(0,2,3))>=.4)&(capture.min(axis=(0,2))>=.999)
    candidates=controls[admitted]
    if len(candidates)<count+3:
        raise ValueError('Too few jointly captured and resolved prospective settings.')
    covariance=all_cov[:,admitted]
    print('candidate states',len(controls),'admitted',len(candidates),flush=True)
    noise_fun=jax.jit(jax.vmap(jax.vmap(lambda c: approximate_covariance_noise(c,2e6))))
    noise=np.asarray([noise_fun(c) for c in covariance])
    def observable(p, c):
        return jax.vmap(jax.vmap(covariance_observable))(moments(quotient,p,c))
    jacobian_fun=jax.jit(jax.jacfwd(observable,argnums=0))
    candidate_jac=np.asarray([jacobian_fun(p,jnp.asarray(candidates)) for p in hypotheses])
    white=np.linalg.solve(np.linalg.cholesky(noise),candidate_jac).reshape(2,len(candidates),6,23)
    factors=np.einsum('hcri,hij->hcrj',white,old_factor)
    indices,scores,updated=greedy_design(factors,count)
    train=candidates[indices]
    # Three fresh interior controls selected geometrically before rendering.
    # Avoid exact reuse of old/selected controls; they are not optimized using
    # held-out measurements or future noise.
    targets=np.array([.6*train[i]+.4*train[(i+1)%count] for i in range(3)])
    held_indices=[]
    for target in targets:
        distances=np.linalg.norm((candidates-target)[:,:3]/np.array([2.25,2.15,3.2]),axis=1)
        distances[np.r_[indices,held_indices].astype(int)]=np.inf
        held_indices.append(int(np.argmin(distances)))
    held=candidates[held_indices]
    held_cov=np.asarray([covariance_fun(quotient,p,held) for p in hypotheses])
    held_rms=np.sqrt(np.linalg.eigvalsh(held_cov))
    held_capture=1-np.asarray(jax.scipy.special.erfc(125./np.sqrt(2*(np.diagonal(held_cov,axis1=-2,axis2=-1)+.65**2)))).sum(axis=-1)
    if held_rms.min()<.4 or held_capture.min()<.999:
        raise ValueError('Geometrically reserved heldout failed the two-hypothesis preflight.')
    all_selected=np.concatenate((train,held))
    selected_cov=np.concatenate((covariance[:,indices],held_cov),axis=1)
    detector=DetectorGeometry(mtf_sigma_px=.65)
    info_fun=jax.jit(lambda encoded:gaussian_image_information_factor(encoded,detector,quadrature_order=16))
    operator=jax.jit(make_gaussian_detector_operator(detector,16))
    exact_noise=[];captures=[]
    for h in range(2):
        cells=[];caught=[]
        for covs in selected_cov[h]:
            sub=[]
            for c in covs:
                encoded=encode_gaussian_image_parameters(jnp.zeros(2),c*55e-6**2,2e6,detector)
                factor=np.asarray(info_fun(encoded));inverse=np.linalg.solve(factor,np.eye(6))
                sub.append((inverse@inverse.T)[2:5,2:5])
                caught.append(float(operator(jnp.zeros(2),c*55e-6**2,2e6).sum()/2e6))
            cells.append(sub)
        exact_noise.append(cells);captures.append(min(caught))
    exact_noise=np.asarray(exact_noise)
    summary=[]
    for h,p in enumerate(hypotheses):
        derivatives=np.asarray(jacobian_fun(p,all_selected))
        fresh_white=np.linalg.solve(np.linalg.cholesky(exact_noise[h]),derivatives).reshape(-1,6,23)
        combined=np.vstack((old_jac[h],fresh_white[:count].reshape(-1,23)))
        summary.append(dict(hypothesis=h,before=information_summary(old_jac[h],quotient,p),
            after=information_summary(combined,quotient,p),
            minimum_capture=captures[h],minimum_optical_rms_pixels=float(np.sqrt(np.linalg.eigvalsh(selected_cov[h])).min()),
            maximum_optical_rms_pixels=float(np.sqrt(np.linalg.eigvalsh(selected_cov[h])).max())))
    sources=dict(repair['source_hashes']);sources.update(old['source_hashes'])
    for path in (Path(__file__),FOLDER/'condenser_covariance_fit.json',FOLDER/'condenser_image_recovery.json',
                 ROOT/'examples/lens_inversion/neoarm_condenser_covariance_fit.py'):
        sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    report=dict(schema='neoarm-condenser-fresh-extension-design-v1',source_hashes=sources,
        backend=jax.default_backend(),x64_enabled=jax.config.x64_enabled,elapsed_seconds=time.perf_counter()-started,
        hypothesis_parameters=hypotheses.tolist(),hypothesis_labels=['rough_geometry_data_initialized_source','unaccepted_old_training_iterate'],
        conditional_downstream_objective_calibration=True,old_training_data_only_for_design=True,
        old_heldout_observations_used_for_scoring=False,generating_condenser_parameters_used_for_scoring=False,
        candidate_axes=[a.tolist() for a in axes],candidate_count=len(controls),admitted_candidate_count=len(candidates),
        candidate_construction='CL1/CL3 grid, bracketed C-mini minimum-width refinement, five small C-mini offsets; heldouts nearest unused admitted candidates to geometric interior targets.',
        controls=all_selected.tolist(),training_mask=([True]*count+[False]*3),heldout_mask=([False]*count+[True]*3),
        training_count=count,heldout_count=3,new_detector_frames=2*(count+3),incident_electrons_per_frame=2e6,
        greedy_minimum_logdet_gains=scores,exact_finite_pixel_response_covariance=exact_noise.tolist(),
        hypothesis_information=summary,frozen_before_new_images_and_noise=True,
        minimum_all_hypothesis_capture=min(captures),hardware_admitted=False,
        limitations=['Strong prospective CL1/CL3 excitations explore a larger synthetic range; hardware control limits remain unconfirmed.',
            'Two provisional local hypotheses are not full uncertainty or global branch coverage.',
            'Only selected endpoints are preflighted; a connected refocused path and lens-bore/specimen clearance require further checks.',
            'Candidate ranking uses a Gaussian moment approximation; selected-state information uses full pixel-integrated image covariance.',
            'Objective/downstream calibration and unchanged unknown upstream astigmatic preparation remain conditional.',
            'The first frozen design precedes new noise; nonlinear recovery and independent Gaussian-beam validation remain required.'])
    if any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=value for name,value in sources.items()):
        raise RuntimeError('A design dependency changed during execution.')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--grid-count',type=int,default=17)
    args=parser.parse_args()
    result=design(args.output,grid_count=args.grid_count)
    print('Saved',args.output,flush=True)
