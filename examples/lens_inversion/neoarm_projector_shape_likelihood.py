"""Physical common-source shape tests using the original image-fit likelihood.

Centroid responses are NOT appended. Instead we use the first five coordinates
of each six-parameter image fit with their full marginal covariance (free dose
is profiled), and profile a common two-dimensional alignment offset for each
nine-frame optical setting. This is exactly the corresponding Gaussian-summary
profile likelihood, with no second copy of the centroid information.

Source covariance is unknown and quantum admissible. Round optics have an
exact shape gauge: the antisymmetric part of the position--canonical-momentum
cross covariance is invisible. It is not set to generating truth or counted as
an identified source parameter. The old heldouts have already been used for
optical branch comparison; these checks are not new confirmatory data.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
import numpy as np

from neoarm_coupled_projector_fit import (
    CoupledProjectorResponse, grouped_stage, stage_coordinates, load_paired,
)
from neoarm_last_lens_physical_fit import ROOT, load_image_responses
from neoarm_last_lens_tomography import J2, J4
from temgym_core.constants import energy2wavelength
from temgym_core.jax_least_squares import least_squares


SOURCE_NAMES = ('log_X_cholesky_00', 'log_X_cholesky_11', 'X_cholesky_10',
    'H_00', 'H_01', 'H_10', 'H_11', 'log_thermal_determinant_slack',
    'thermal_shape_log_cholesky_00', 'thermal_shape_cholesky_10')


@dataclass(frozen=True)
class QuantumSourceCovariance:
    """Ten-coordinate interior chart for every quantum Gaussian covariance.

    In balanced coordinates epsilon=lambda/(4pi)=1, set C=X H and
    P=H.T X H+X^-1+N, N=a R, det(R)=1 and
    a²=(H10-H01)²+exp(2eta). The Schur complement of Sigma+iJ is then a
    positive Hermitian 2x2 matrix, since det(N)>(H10-H01)². Pure covariances
    are boundary limits eta->-inf, not an assumed known source preparation.
    """

    position_scale_m: float
    wavelength_m: float

    def __post_init__(self):
        if not np.isfinite(self.position_scale_m) or self.position_scale_m <= 0 or not np.isfinite(self.wavelength_m) or self.wavelength_m <= 0:
            raise ValueError('Positive finite position scale and wavelength are required.')

    @property
    def scales(self):
        return np.array([self.position_scale_m]*2+[self.wavelength_m/(4*np.pi*self.position_scale_m)]*2)

    def components(self, parameters):
        p = jnp.asarray(parameters)
        if p.shape != (10,):
            raise ValueError('The physical source chart has ten coordinates.')
        chol = jnp.array([[jnp.exp(p[0]),0.],[p[2],jnp.exp(p[1])]])
        xx = chol@chol.T
        hh = p[3:7].reshape(2,2)
        thermal_chol = jnp.array([[jnp.exp(p[8]),0.],[p[9],jnp.exp(-p[8])]])
        amplitude = jnp.hypot(hh[1,0]-hh[0,1],jnp.exp(p[7]))
        nn = amplitude*(thermal_chol@thermal_chol.T)
        return xx,hh,nn

    def balanced_covariance(self, parameters):
        xx,hh,nn = self.components(parameters)
        cross = xx@hh
        pp = hh.T@xx@hh+jnp.linalg.inv(xx)+nn
        return jnp.block([[xx,cross],[cross.T,pp]])

    def covariance(self, parameters):
        return self.balanced_covariance(parameters)*jnp.asarray(np.outer(self.scales,self.scales))

    def encode_balanced(self, covariance):
        cov = np.asarray(covariance)
        xx, cross, pp = cov[:2,:2], cov[:2,2:], cov[2:,2:]
        chol = np.linalg.cholesky(xx)
        hh = np.linalg.solve(xx,cross)
        nn = pp-hh.T@xx@hh-np.linalg.inv(xx)
        nn = (nn+nn.T)/2
        amplitude = np.sqrt(np.linalg.det(nn))
        slack = amplitude**2-(hh[1,0]-hh[0,1])**2
        if not np.isfinite(slack) or slack <= 0 or np.linalg.eigvalsh(nn).min() <= 0:
            raise ValueError('Encoding requires a strictly interior quantum covariance.')
        thermal_chol = np.linalg.cholesky(nn/amplitude)
        return np.r_[np.log(chol[0,0]),np.log(chol[1,1]),chol[1,0],hh.ravel(),
            .5*np.log(slack),np.log(thermal_chol[0,0]),thermal_chol[1,0]]

    def diagnostic(self, parameters):
        cov = np.asarray(self.balanced_covariance(parameters))
        hermitian = cov+1j*np.asarray(J4)
        eig = np.linalg.eigvalsh(hermitian)
        factor = np.linalg.cholesky(cov)
        emittance = np.linalg.eigvalsh(1j*factor.T@np.asarray(J4)@factor)
        return dict(canonical_covariance_grouped_m_rad=np.asarray(self.covariance(parameters)).tolist(),
            minimum_balanced_uncertainty_eigenvalue=float(eig.min()),
            emittances_over_quantum_minimum=emittance[2:].tolist(),
            quantum_admissible=bool(eig.min() >= -1e-9*max(1.,abs(eig).max())),
            antisymmetric_cross_covariance_balanced=float((cov[1,2]-cov[0,3])/2),
            antisymmetric_cross_covariance_identified=False)


def canonical_position_rows(model, optics, controls):
    physical = stage_coordinates(optics)
    prefix = model.upstream_matrix(optics,coordinates='canonical')
    return jax.vmap(lambda control:(grouped_stage(physical[:4],control[0])@
        grouped_stage(physical[4:],control[1])@prefix)[:2])(controls)


def shape_coordinates(covariance_pixel):
    sigma = jnp.sqrt(jnp.diagonal(covariance_pixel,axis1=-2,axis2=-1))
    correlation = covariance_pixel[...,0,1]/jnp.prod(sigma,axis=-1)
    return jnp.concatenate((jnp.log(sigma),jnp.arctanh(correlation)[...,None]),axis=-1)


def shape_moments(parameters):
    sx,sy = jnp.exp(parameters[:2])
    return jnp.array([sx*sx,jnp.tanh(parameters[2])*sx*sy,sy*sy])


class ImageSummaryLikelihood:
    """Full correlated Gaussian-summary likelihood after dose/alignment profile."""

    def __init__(self, model, source, controls, image_parameters, image_covariances, input_states):
        y, cov = np.asarray(image_parameters),np.asarray(image_covariances)
        controls, inputs = np.asarray(controls),np.asarray(input_states)
        if y.ndim != 3 or y.shape[1:] != (9,6) or cov.shape != y.shape+(6,):
            raise ValueError('Expected (setting,9,6) image fits and full (setting,9,6,6) covariances.')
        if controls.shape != (len(y),2) or inputs.shape != (9,4) or not all(np.all(np.isfinite(a)) for a in (y,cov,controls,inputs)):
            raise ValueError('Image summaries, controls and calibrated inputs must be finite and aligned.')
        if not np.allclose(cov,cov.swapaxes(-1,-2),rtol=1e-11,atol=1e-18):
            raise ValueError('Image-fit covariance must be symmetric.')
        self.model,self.source = model,source
        self.controls,self.observed,self.covariances = map(jnp.asarray,(controls,y,cov))
        self.normalized_inputs = jnp.asarray(inputs/np.asarray(model.calibration.input_steps_grouped))
        # Profiling an unrestricted logdose uses the marginal covariance of
        # the retained five coordinates, not their conditional covariance.
        self.cholesky = jnp.asarray(np.linalg.cholesky(cov[:,:,:5,:5]))
        self.mean_cholesky = jnp.asarray(np.linalg.cholesky(cov[:,:,:2,:2]))
        design = np.tile(np.eye(5)[:,:2],(9,1)).reshape(9,5,2)
        whitened_design = np.linalg.solve(np.asarray(self.cholesky),np.broadcast_to(design,(len(y),9,5,2))).reshape(len(y),45,2)
        self.alignment_q = jnp.asarray(np.linalg.qr(whitened_design,mode='reduced')[0])
        means_design = np.broadcast_to(np.eye(2),(len(y),9,2,2))
        means_design = np.linalg.solve(np.asarray(self.mean_cholesky),means_design).reshape(len(y),18,2)
        self.mean_alignment_q = jnp.asarray(np.linalg.qr(means_design,mode='reduced')[0])
        self.source_residual = jax.jit(self._source_residual)
        self.joint_residual = jax.jit(lambda p:self._source_residual(p[13:],p[:13]))
        self.centroid_residual = jax.jit(self._centroid_residual)

    @property
    def independent_summary_coordinate_count(self):
        return len(self.controls)*(9*5-2)

    def means(self,optics):
        response = self.model.predict(optics,self.controls)
        return jnp.einsum('sij,fj->sfi',response,self.normalized_inputs)

    def shapes(self,optics,source_parameters):
        rows = canonical_position_rows(self.model,optics,self.controls)*jnp.asarray(self.source.scales)[None,None,:]/self.model.calibration.pixel_m
        xx,hh,nn = self.source.components(source_parameters)
        # Sum two PSD terms; do not form a large, nearly cancelling dense
        # covariance quadratic when trial source curvatures are extreme.
        position = rows[:,:,:2]+rows[:,:,2:]@hh.T
        covariance = position@xx@position.swapaxes(-1,-2)+rows[:,:,2:]@(jnp.linalg.inv(xx)+nn)@rows[:,:,2:].swapaxes(-1,-2)
        return shape_coordinates(covariance)

    def _source_residual(self,source_parameters,optics):
        prediction = jnp.concatenate((self.means(optics),
            jnp.broadcast_to(self.shapes(optics,source_parameters)[:,None,:],(len(self.controls),9,3))),axis=-1)
        delta = prediction-self.observed[:,:,:5]
        white = jax.vmap(jax.vmap(lambda l,r:solve_triangular(l,r,lower=True)))(self.cholesky,delta).reshape(len(self.controls),45)
        return (white-jnp.einsum('sij,skj,sk->si',self.alignment_q,self.alignment_q,white)).reshape(-1)

    def _centroid_residual(self,optics):
        delta = self.means(optics)-self.observed[:,:,:2]
        white = jax.vmap(jax.vmap(lambda l,r:solve_triangular(l,r,lower=True)))(self.mean_cholesky,delta).reshape(len(self.controls),18)
        return (white-jnp.einsum('sij,skj,sk->si',self.mean_alignment_q,self.mean_alignment_q,white)).reshape(-1)


def initialize_source(problem,optics):
    """Weighted shape-moment tomography followed by a physical seed projection.

    Exactly one linear round-map gauge is retained as unresolved. The minimum
    norm completion is ONLY an initializer; the final physical fit can vary it.
    Eigenvalue/quantum expansion below is disclosed, not used to alter data.
    """
    rows = np.asarray(canonical_position_rows(problem.model,jnp.asarray(optics),problem.controls)).copy()
    rows *= problem.source.scales[None,None,:]/problem.model.calibration.pixel_m
    indices = [(i,j) for i in range(4) for j in range(i,4)]
    basis = np.zeros((10,4,4))
    for k,(i,j) in enumerate(indices):
        basis[k,i,j]=1.;basis[k,j,i]=1.
    matrices = np.einsum('sai,kij,sbj->skab',rows,basis,rows)
    design = matrices[:,:,[0,0,1],[0,1,1]].transpose(0,2,1)
    shape = np.asarray(problem.observed[:,:,2:5])
    values = np.asarray(jax.vmap(jax.vmap(shape_moments))(jnp.asarray(shape)))
    jac = np.asarray(jax.vmap(jax.vmap(jax.jacfwd(shape_moments)))(jnp.asarray(shape)))
    covariance = np.einsum('sfai,sfij,sfbj->sfab',jac,np.asarray(problem.covariances[:,:,2:5,2:5]),jac)
    chol = np.linalg.cholesky(covariance)
    white_design = np.linalg.solve(chol,np.broadcast_to(design[:,None],(len(rows),9,3,10))).reshape(-1,10)
    white_values = np.linalg.solve(chol,values[...,None]).reshape(-1)
    norms = np.linalg.norm(white_design,axis=0)
    coefficients,_,rank,singular = np.linalg.lstsq(white_design/norms,white_values,rcond=1e-10)
    raw = np.einsum('k,kij->ij',coefficients/norms,basis)
    eig, vectors = np.linalg.eigh(raw)
    floor = max(1e-8,float(eig.max())*1e-10)
    positive = (vectors*np.maximum(eig,floor))@vectors.T
    cholesky = np.linalg.cholesky(positive)
    emittance = np.linalg.eigvalsh(1j*cholesky.T@np.asarray(J4)@cholesky)[2:]
    expansion = max(1.01,1.01/float(emittance.min()))
    admitted_seed = positive*expansion
    parameters = problem.source.encode_balanced(admitted_seed)
    return parameters,dict(linear_shape_information_rank=int(rank),linear_source_coordinate_count=10,
        linear_normalized_singular_values=singular.tolist(), raw_balanced_covariance=raw.tolist(),
        seed_spd_eigenvalue_floor=floor, seed_quantum_expansion=expansion,
        seed_projection_relative_change=float(np.linalg.norm(admitted_seed-raw)/np.linalg.norm(raw)),
        minimum_norm_gauge_is_initialization_only=True, generating_source_used=False)


def source_bounds():
    return (np.array([-8.,-8.,-100.,-1e3,-1e3,-1e3,-1e3,-20.,-10.,-1e4]),
            np.array([8.,8.,100.,1e3,1e3,1e3,1e3,12.,10.,1e4]))


def center_source_gauge(source,parameters):
    """Choose a well-conditioned quantum completion without changing shapes.

    The feasible classical Cauchy bounds bracket the quantum-feasible interval.
    The minimum eigenvalue of Sigma(t)+iJ is concave in scalar t, so a fixed
    ternary refinement selects its maximum. This is a gauge convention, not
    information about the real source's unobserved angular momentum.
    """
    covariance=np.asarray(source.balanced_covariance(parameters))
    cross=covariance[:2,2:]
    bound01=np.sqrt(covariance[0,0]*covariance[3,3])
    bound10=np.sqrt(covariance[1,1]*covariance[2,2])
    left=max(cross[0,1]-bound01,-cross[1,0]-bound10)
    right=min(cross[0,1]+bound01,-cross[1,0]+bound10)
    gauge=np.block([[np.zeros((2,2)),np.asarray(J2)],[-np.asarray(J2),np.zeros((2,2))]])
    minimum=lambda t:float(np.linalg.eigvalsh(covariance+t*gauge+1j*np.asarray(J4)).min())
    original=minimum(0.)
    for _ in range(100):
        first,second=(2*left+right)/3,(left+2*right)/3
        if minimum(first)<minimum(second):left=first
        else:right=second
    shift=(left+right)/2
    improved=minimum(shift)
    if improved<=max(0.,original):
        return np.asarray(parameters),dict(applied=False,original_minimum_quantum_eigenvalue=original)
    updated=source.encode_balanced(covariance+shift*gauge)
    return updated,dict(applied=True,balanced_antisymmetric_cross_shift=shift,
        original_minimum_quantum_eigenvalue=original,centered_minimum_quantum_eigenvalue=improved,
        gauge_chosen_for_conditioning_not_measured=True)


def fit_summary(problem, initial, bounds, *, fixed_optics=None,max_nfev=600):
    residual = problem.joint_residual if fixed_optics is None else jax.jit(lambda p:problem.source_residual(p,jnp.asarray(fixed_optics)))
    initial = np.asarray(initial)
    lower,upper = map(np.asarray,bounds)
    initial = np.clip(initial,lower+1e-12*(upper-lower),upper-1e-12*(upper-lower))
    jac = np.asarray(jax.jacfwd(residual)(jnp.asarray(initial)))
    norms = np.linalg.norm(jac,axis=0)
    # A near-gauge/boundary column must not receive an enormous optimizer
    # coordinate scale. Disabling scaled-x xtol avoids false early stopping
    # when the initial quantum projection is badly conditioned.
    fit = least_squares(residual,initial,bounds=(lower,upper),x_scale=np.minimum(1/norms,1.),
        max_nfev=max_nfev,max_iter=1200,ftol=1e-10,xtol=0.,gtol=1e-7)
    parameters = np.asarray(fit.x)
    optics,source = (parameters[:13],parameters[13:]) if fixed_optics is None else (np.asarray(fixed_optics),parameters)
    final_jac = np.asarray(fit.jacobian)
    final_norms = np.linalg.norm(final_jac,axis=0)
    singular = np.linalg.svd(final_jac/final_norms,compute_uv=False)
    gradient=final_jac.T@np.asarray(fit.residual)
    binding=((parameters<=lower)&(gradient>0))|((parameters>=upper)&(gradient<0))
    projected_gradient=np.where(binding,0.,gradient/final_norms)
    centroid = np.asarray(problem.centroid_residual(jnp.asarray(optics)))
    return dict(parameters=parameters.tolist(), optics=optics.tolist(),source_parameters=source.tolist(),
        optimizer_success=bool(fit.success),termination=fit.status,nfev=fit.nfev,chi_square=float(2*fit.cost),
        final_column_normalized_projected_gradient_max=float(np.max(np.abs(projected_gradient))),
        solver_scaled_optimality=float(fit.optimality),
        centroid_only_profile_chi_square=float(centroid@centroid),
        conditional_shape_chi_square=float(2*fit.cost-centroid@centroid),
        active_bounds=np.flatnonzero(np.minimum(abs(parameters-lower),abs(parameters-upper))<1e-7).tolist(),
        normalized_jacobian_singular_values=singular.tolist(),numerical_rank=int(np.sum(singular>singular[0]*1e-9)),
        relative_rank_tolerance=1e-9,full_coordinate_count=len(parameters),
        source_diagnostic=problem.source.diagnostic(jnp.asarray(source)),
        uncertainty_interpretation='Exact source gauge and possible pure-state boundary preclude an ordinary full-rank covariance claim.',
        fixed_optical_candidate=fixed_optics is not None, initial=initial.tolist(),bounds=[lower.tolist(),upper.tolist()],
        optimizer_success_does_not_imply_noise_consistent_fit=True)


def run_audit(coupled_report,boundary_report,output,*,max_nfev=600,joint=True):
    if jax.default_backend()!='gpu' or not jax.config.x64_enabled:
        raise RuntimeError('Run image-summary forward/derivative fits on GPU float64.')
    started=perf_counter();digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    coupled_path,boundary_path=Path(coupled_report).resolve(),Path(boundary_report).resolve()
    coupled=json.loads(coupled_path.read_text());boundary=json.loads(boundary_path.read_text())
    hashes={**coupled['source_hashes'],**boundary['source_hashes']}
    for name,checksum in hashes.items():
        if digest(ROOT/name)!=checksum:raise ValueError(f'Stale shape input:{name}')
    pl1_path,paired_path=ROOT/coupled['pl1_image_report'],ROOT/coupled['paired_image_report']
    calibration,_,_,_,paths,base=load_image_responses(pl1_path)
    paired,data,paired_paths=load_paired(paired_path);paths.update(paired_paths)
    model=CoupledProjectorResponse(calibration)
    source=QuantumSourceCovariance(float(np.sqrt(np.prod(calibration.input_steps_grouped[:2]))),float(energy2wavelength(200e3)))
    with np.load(ROOT/base['arrays_path'],allow_pickle=False) as saved:
        train_y,train_c=[saved[key].copy() for key in ('image_parameters','image_parameter_covariances')]
        hold_y,hold_c=[saved[key].copy() for key in ('heldout_image_parameters','heldout_image_parameter_covariances')]
        inputs=saved['input_states_grouped'].copy()
    with np.load(ROOT/paired['arrays_path'],allow_pickle=False) as saved:
        y,cov=saved['image_parameters'].copy(),saved['image_parameter_covariances'].copy()
        if not np.array_equal(inputs,saved['input_states_grouped']):raise ValueError('Input-state calibration differs between image blocks.')
    train_y=np.concatenate((train_y,y[data['training_mask']].reshape(-1,9,6)))
    train_c=np.concatenate((train_c,cov[data['training_mask']].reshape(-1,9,6,6)))
    hold_y=np.concatenate((hold_y,y[data['heldout_mask']].reshape(-1,9,6)))
    hold_c=np.concatenate((hold_c,cov[data['heldout_mask']].reshape(-1,9,6,6)))
    problem=ImageSummaryLikelihood(model,source,coupled['training_controls'],train_y,train_c,inputs)
    validation=ImageSummaryLikelihood(model,source,coupled['heldout']['controls'],hold_y,hold_c,inputs)
    paths.update((Path(__file__),coupled_path,boundary_path,ROOT/'src/temgym_core/constants.py'))
    for path in paths:hashes[str(path.relative_to(ROOT))]=digest(path)
    rival=min((r for r in boundary['rows'] if r['optimizer_success']),key=lambda r:r['chi_square'])
    candidates=[('best_mean_ray_branch',coupled['best']),('interior_IL3_rival',rival)]
    rows=[]
    for label,candidate in candidates:
        optics=np.asarray(candidate['parameters'])
        seed,seed_diagnostic=initialize_source(problem,optics)
        profiles=[]
        initializers=[('data_moment_projection',seed)]
        for log_radius in (-1.,0.,1.):
            broad=np.zeros(10);broad[:2]=log_radius;broad[7]=-1.
            initializers.append((f'broad_unknown_source_log_radius_{log_radius}',broad))
        for seed_label,initial in initializers:
            fit=fit_summary(problem,initial,source_bounds(),fixed_optics=optics,max_nfev=max_nfev)
            fit['seed_label']=seed_label
            profiles.append(fit)
            print(label,'fixedsource',seed_label,fit['optimizer_success'],fit['chi_square'],fit['termination'],flush=True)
        eligible=[row for row in profiles if row['optimizer_success']]
        selected=min(eligible,key=lambda row:row['chi_square']) if eligible else None
        joint_fit=None
        if joint and selected is not None:
            lower,upper=source_bounds();opt_lower,opt_upper=np.asarray(candidate['bounds'])
            centered,gauge_diagnostic=center_source_gauge(source,selected['source_parameters'])
            previous_shapes=np.asarray(problem.shapes(jnp.asarray(optics),jnp.asarray(selected['source_parameters'])))
            centered_shapes=np.asarray(problem.shapes(jnp.asarray(optics),jnp.asarray(centered)))
            gauge_diagnostic['maximum_shape_coordinate_change']=float(np.max(np.abs(centered_shapes-previous_shapes)))
            if gauge_diagnostic['maximum_shape_coordinate_change']>1e-9:raise ValueError('Gauge centering changed a physical image shape.')
            selected['joint_initializer_gauge_centering']=gauge_diagnostic
            initial=np.r_[optics,centered]
            joint_fit=fit_summary(problem,initial,(np.r_[opt_lower,lower],np.r_[opt_upper,upper]),max_nfev=max_nfev)
            print(label,'joint',joint_fit['optimizer_success'],joint_fit['chi_square'],joint_fit['termination'],flush=True)
        rows.append(dict(label=label,seed_diagnostic=seed_diagnostic,fixed_source_profiles=profiles,
            best_fixed_source_profile=selected,joint_fit=joint_fit))
    # Evaluate only after all optical/source fits. These exposures were used in
    # earlier mean-ray branch selection, so this is not virgin confirmation.
    for row in rows:
        for key in ('best_fixed_source_profile','joint_fit'):
            fit=row[key]
            if fit is not None:
                residual=np.asarray(validation.source_residual(jnp.asarray(fit['source_parameters']),jnp.asarray(fit['optics'])))
                means=np.asarray(validation.centroid_residual(jnp.asarray(fit['optics'])))
                fit['heldout']=dict(profile_chi_square=float(residual@residual),centroid_profile_chi_square=float(means@means),
                    conditional_shape_chi_square=float(residual@residual-means@means),
                    independent_summary_coordinate_count=validation.independent_summary_coordinate_count,
                    optics_or_source_refitted=False,alignment_and_dose_nuisance_profiled=True,
                    previously_used_for_mean_ray_branch_comparison=True,new_confirmatory_dataset=False)
    if any(digest(ROOT/name)!=checksum for name,checksum in hashes.items()):raise RuntimeError('Shape evidence changed during execution.')
    fixed_profiles_available=all(row['best_fixed_source_profile'] is not None and row['best_fixed_source_profile']['source_diagnostic']['quantum_admissible'] for row in rows)
    joint_accepted=joint and all(row['joint_fit'] is not None and row['joint_fit']['optimizer_success'] and row['joint_fit']['source_diagnostic']['quantum_admissible'] for row in rows)
    report=dict(schema='neoarm-projector-unknown-source-shape-likelihood-v1',source_hashes=hashes,
        source_hashes_unchanged_during_execution=True,backend=jax.default_backend(),x64_enabled=True,jax_version=jax.__version__,elapsed_seconds=perf_counter()-started,
        source_parameter_names=SOURCE_NAMES,source_position_balance_m=source.position_scale_m,wavelength_m=source.wavelength_m,
        source_covariance_granted=False,purity_granted=False,canonical_quantum_admissibility_enforced=True,
        exact_shape_gauge='Add t*J2 to source position/canonical-momentum cross covariance and its transpose to the lower block; every round-map spatial covariance is unchanged.',
        source_shape_identifiable_covariance_coordinate_count=9,
        full_source_covariance_coordinate_count=10,source_gauge_fixed_to_truth=False,
        training_frame_count=int(train_y.shape[0]*9),heldout_frame_count=int(hold_y.shape[0]*9),
        training_independent_summary_coordinate_count=problem.independent_summary_coordinate_count,
        heldout_independent_summary_coordinate_count=validation.independent_summary_coordinate_count,
        old_centroid_response_likelihood_added=False,per_frame_dose_profiled=True,per_setting_alignment_coordinates_profiled=2,
        same_image_centroid_shape_cross_covariance_retained=True,candidates=rows,
        fixed_optical_source_profiles_available=fixed_profiles_available,
        joint_optics_source_convergence_admitted=bool(joint_accepted),accepted=bool(joint_accepted),
        acceptance_scope='Optimizer and physical-source admission for both local joint branches, not global branch exclusion or full-instrument recovery.',
        global_branch_exclusion_proved=False,assumptions=[
            'The per-frame Gaussian-summary likelihood uses the fixed full image-fit covariance, not the raw-pixel likelihood.',
            'Free dose is profiled by retaining the five-coordinate marginal covariance, not a conditional covariance.',
            'Nine frames share one alignment offset per optical setting; source covariance is common across all settings and calibrated kicks.',
            'The original signed-response likelihood is replaced, never appended, so centroid information is not double counted.',
            'Source covariance is canonical and quantum admissible at200kV, but neither covariance nor purity is supplied by truth.',
            'Round optics cannot measure the antisymmetric source cross-covariance; source boundary/gauge directions invalidate naive full-rank error bars.',
            'Previously inspected heldouts are validation/model-selection evidence, not a new confirmatory experiment.',
            'Source stability, exact input calibration, detector model and hard-edge field-free-gap optics remain conditional instrument assumptions.'])
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coupled-report',type=Path,required=True);parser.add_argument('--boundary-report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--max-nfev',type=int,default=600)
    parser.add_argument('--fixed-only',action='store_true');args=parser.parse_args();jax.config.update('jax_enable_x64',True)
    result=run_audit(args.coupled_report,args.boundary_report,args.output,max_nfev=args.max_nfev,joint=not args.fixed_only)
    print('Saved',args.output,'in',result['elapsed_seconds'],'seconds',flush=True)
