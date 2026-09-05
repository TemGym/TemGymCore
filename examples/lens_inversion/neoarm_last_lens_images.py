"""Image-derived, conditional PL1 response tomography on a finite detector.

This deliberately bounded vacuum example uses the unchanged last-lens
coefficient model. Nine candidate excitations receive a reference and signed
four-direction input basis. A predeclared optical-width gate excludes
unresolved crossovers; excluded candidates are never silently admitted.
The source is a quantum-valid pure astigmatic Gaussian, not an independently
assigned position/angle covariance. Dose, known input calibration and the
affine hard-edge excitation law remain conditional assumptions.
"""

from pathlib import Path
import argparse
import hashlib
import json
import os
import time

os.environ.setdefault('JAX_ENABLE_X64', 'true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import jax
import jax.numpy as jnp
import numpy as np

from neoarm_last_lens_tomography import (
    mechanical_shear, predict_response_parameters, geometry_coordinates,
)
from neoarm_mixed_gaussian_validation import MixedGaussianSource, source_modes, detector_quadrature
from temgym_core.gaussian_image_likelihood import (
    make_gaussian_detector_operator, fit_gaussian_image,
    encode_gaussian_image_parameters, gaussian_image_information_factor,
)
from temgym_core.jax_least_squares import least_squares
from temgym_core.microscope_inversion import drift_matrix, hard_edge_solenoid_matrix
from temgym_core.neoarm_calibration import DetectorGeometry
from temgym_core.neoarm_calibration import apply_transfer_matrix_to_gaussians
from temgym_core.evaluate import _beam_field, _prepare_gaussian_params
from temgym_core.gaussian_image_likelihood import apply_detector_psf


def central_response_transform():
    """Map nine (x,y) means to row-major 2x4 central signed responses.

    Frame ordering: reference,+x,-x,+y,-y,+theta_x,-theta_x,+theta_y,-theta_y.
    Reference coefficients are exactly zero: shared-reference correlations
    cancel for central differences, not for the individual one-sided data.
    """
    transform = np.zeros((8, 18))
    for component in range(2):
        for basis in range(4):
            transform[4*component+basis, 2*(1+2*basis)+component] = .5
            transform[4*component+basis, 2*(2+2*basis)+component] = -.5
    return transform


def signed_response_from_image_fits(parameters, covariances):
    parameters, covariances = np.asarray(parameters), np.asarray(covariances)
    if parameters.shape != (9, 6) or covariances.shape != (9, 6, 6):
        raise ValueError('Expected nine six-coordinate image fits with full covariance.')
    means = parameters[:, :2].ravel()
    covariance = np.zeros((18, 18))
    for i in range(9):
        covariance[2*i:2*i+2, 2*i:2*i+2] = covariances[i, :2, :2]
    transform = central_response_transform()
    response_covariance = transform@covariance@transform.T
    if not np.all(np.isfinite(response_covariance)) or np.linalg.eigvalsh(response_covariance).min() <= 0:
        raise ValueError('Signed response covariance is not positive definite.')
    return (transform@means).reshape(2, 4), response_covariance


def image_initializer(observed, detector, read_noise=.6):
    """Data-only thresholded moments initialize, never replace, the raw fit."""
    image = np.asarray(observed)
    signal = np.where(image > max(5*read_noise, .001*float(image.max())), image, 0.)
    total = signal.sum()
    if total <= 0:
        raise ValueError('No significant positive image signal for initialization.')
    yy, xx = np.indices(image.shape)
    coords = np.stack((xx+.5-image.shape[1]/2, yy+.5-image.shape[0]/2), -1)
    mean = np.einsum('ij,ijk->k', signal, coords)/total
    delta = coords-mean
    covariance = np.einsum('ij,ijk,ijl->kl', signal, delta, delta)/total
    covariance -= np.eye(2)*(detector.mtf_sigma_px**2+1/12)
    values, vectors = np.linalg.eigh(covariance)
    covariance = (vectors*np.maximum(values, .3**2))@vectors.T
    sigma = np.sqrt(np.diag(covariance))
    rho = np.clip(covariance[0, 1]/np.prod(sigma), -.95, .95)
    return np.r_[mean, np.log(sigma), np.arctanh(rho), np.log(total)]


def response_residual(parameters, t, observed, inverse_cholesky, scale_sigma=None):
    predicted = predict_response_parameters(parameters[:19], t)
    if scale_sigma is not None:
        # Four log input-scale errors, shared across all states. These can be
        # absorbed into W columns; only the external priors resolve that gauge.
        predicted = predicted*jnp.exp(parameters[19:])[None, None, :]
    residual = jnp.einsum('nij,nj->ni', inverse_cholesky,
        (predicted-observed).reshape(-1, 8)).ravel()
    return residual if scale_sigma is None else jnp.r_[residual, parameters[19:]/scale_sigma]


def covariance_factor(jacobian):
    jacobian = np.asarray(jacobian)
    norms = np.linalg.norm(jacobian, axis=0)
    if np.any(norms == 0):
        raise ValueError('Zero information column.')
    _, singular, right = np.linalg.svd(jacobian/norms, full_matrices=False)
    rank = int(np.sum(singular > singular[0]*1e-10))
    factor = (right.T/singular)/norms[:, None]
    return factor, rank, singular


def render_pure_counts(source, matrix, detector, *, pixel_order, incident_electrons):
    """Existing Gaussian propagation plus independent 2D area quadrature.

    Canonical unit balancing is used ONLY to check symplecticity. It changes
    neither the supplied physical matrix nor the propagated Gaussian. An
    unscaled absolute tolerance on M.T@J@M is invalid for highly magnifying
    columns: its large cancelling products have different physical units.
    """
    if jax.default_backend() != 'gpu':
        raise RuntimeError('Independent Gaussian forward render requires GPU.')
    matrix=np.asarray(matrix);j=np.kron(np.eye(2),np.array([[0.,1.],[-1.,0.]]))
    canonical_units=np.diag([1e-4,1e4,1e-4,1e4])
    error=canonical_units.T@(matrix.T@j@matrix-j)@canonical_units
    if not np.all(np.isfinite(matrix)) or np.max(np.abs(error))>1e-8:
        raise ValueError('Physical transfer fails the canonically balanced symplectic check.')
    if np.any(source.centre_covariance):
        raise ValueError('This bounded independent check accepts a single pure mode only.')
    beam,_=source_modes(source,1)
    beam=apply_transfer_matrix_to_gaussians(beam,matrix,z_out_m=detector.z_m,axial_path_m=0.)
    params=tuple(v[0] for v in _prepare_gaussian_params(beam))
    coords,weights,halo_shape,_=detector_quadrature(detector,pixel_order)
    count=len(coords);tile=128;padding=(-count)%tile
    coords=jnp.pad(jnp.asarray(coords),((0,padding),(0,0),(0,0)))
    coords=coords.reshape(-1,tile*pixel_order**2,2);weights=jnp.asarray(weights)
    @jax.jit
    def evaluate(params,coords):
        def chunk(points):
            field=_beam_field(*params,points)
            return (jnp.abs(field)**2).reshape(tile,pixel_order**2)@weights
        values=jax.lax.map(chunk,coords).ravel()[:count].reshape(halo_shape)
        return incident_electrons*apply_detector_psf(values,detector)
    return evaluate(params,coords),float(np.max(np.abs(error)))


def fit_response(t, response, covariance, initial, truth, inputs, pixel_m, scale_sigma=None):
    inverse = jnp.asarray(np.linalg.inv(np.linalg.cholesky(covariance)))
    start = np.r_[initial, np.zeros(4)] if scale_sigma is not None else initial
    target = np.r_[truth, np.zeros(4)] if scale_sigma is not None else truth
    residual = lambda p: response_residual(p, t, response, inverse, scale_sigma)
    jac = np.asarray(jax.jacfwd(residual)(jnp.asarray(target)))
    norms = np.linalg.norm(jac, axis=0)
    lower = np.r_[np.log(1.), -4., -3., np.full(16, -1e6)]
    upper = np.r_[np.log(100.), -.01, -.01, np.full(16, 1e6)]
    if scale_sigma is not None:
        lower, upper = np.r_[lower, np.full(4, -.1)], np.r_[upper, np.full(4, .1)]
    fit = least_squares(residual, start, bounds=(lower, upper), x_scale=1/norms,
        max_nfev=400, ftol=1e-12, gtol=1e-8, xtol=1e-12)
    factor, rank, singular = covariance_factor(jac)
    geom = lambda p: geometry_coordinates(p[:19], inputs, pixel_m)
    derivative = np.asarray(jax.jacfwd(geom)(jnp.asarray(target)))
    uncertainty = np.sqrt(np.sum((derivative@factor)**2, axis=1))
    difference = np.asarray(geom(fit.x)-geom(jnp.asarray(target)))
    sign = float(jnp.linalg.slogdet(fit.x[3:19].reshape(4, 4))[0])
    return dict(success=bool(fit.success), termination=fit.status, nfev=fit.nfev,
        rank=rank, free_coordinates=len(start), determinant_sign=sign,
        fitted_parameters=np.asarray(fit.x).tolist(),
        residual_rms_sigma=float(np.sqrt(np.mean(np.asarray(fit.residual)**2))),
        geometry_coordinate_names=['log_length','log_gap','log_ratio','q_bias','q_slope'],
        geometry_standard_errors=uncertainty.tolist(),
        length_gap_standard_errors_percent=(100*uncertainty[:2]).tolist(),
        length_gap_errors_percent=(100*np.expm1(difference[:2])).tolist(),
        length_gap_errors_sigma=(difference[:2]/uncertainty[:2]).tolist(),
        column_scaled_singular_values=singular.tolist(),
        conditional_parameter_covariance=(factor@factor.T).tolist(),
        input_scale_prior_sigma=scale_sigma,
        geometry_inference_admitted=bool(fit.success and rank == len(start) and sign > 0))


def audit(output, *, incident_electrons=1e6, state_count=9):
    if jax.default_backend() != 'gpu':
        raise RuntimeError('Image forward simulation requires GPU; no CPU fallback.')
    if state_count < 9 or incident_electrons <= 0:
        raise ValueError('At least nine candidates and a positive incident dose are required.')
    root = Path(__file__).resolve().parents[2]
    previous = root/'examples/generated/neoarm_inversion/imaging_recipe/last_lens_tomography.json'
    reference = json.loads(previous.read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for name, value in reference['source_hashes'].items():
        if digest(root/name) != value:
            raise ValueError(f'Stale upstream reference: {name}')
    paths = [Path(__file__), previous, Path(__file__).with_name('neoarm_mixed_gaussian_validation.py'),
        *(root/'src/temgym_core'/name for name in ('gaussian_image_likelihood.py','gaussian.py',
                                                  'evaluate.py','neoarm_calibration.py'))]
    source_hashes = dict(reference['source_hashes'])
    source_hashes.update({str(p.relative_to(root)):digest(p) for p in paths})
    output = Path(output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    given = reference['inputs']; ix = np.array([0,2,1,3])
    upstream = np.asarray(given['upstream_canonical'])[np.ix_(ix,ix)]
    shear = np.asarray(mechanical_shear(given['generating_unknown_kappa_per_m']))[np.ix_(ix,ix)]
    length, gap = given['generating_length_m'], given['generating_detector_gap_m']
    t = np.linspace(-1.,1.,state_count)
    truth = np.asarray(given['truth_parameters'])
    q = truth[1]+truth[2]*t
    inputs = np.asarray(given['input_basis_steps_grouped']);pixel=given['detector_pixel_m']
    detector = DetectorGeometry(shape_yx=(256,256), pixel_size_yx_m=(pixel,pixel),
                                z_m=.71875, mtf_sigma_px=.65)
    source = MixedGaussianSource(np.diag(np.array([2e-9,1.5e-9])**2), np.zeros((2,2)),
                                np.zeros((4,4)),np.zeros(4))
    operator = jax.jit(make_gaussian_detector_operator(detector,16))
    mean_inputs = np.zeros((9,4))
    for i in range(4):
        mean_inputs[1+2*i,i], mean_inputs[2+2*i,i] = inputs[i],-inputs[i]
    canonical_means = (mean_inputs[:,ix]@shear.T)
    all_expected, all_observed, all_fitted, all_variance = [],[],[],[]
    all_parameters, all_covariances, all_initial = [],[],[]
    rows, responses, covariances, accepted_t, matrices, accepted_indices = [],[],[],[],[],[]
    read_noise=.6
    for state_index,(control,excitation) in enumerate(zip(t,q)):
        start_time=time.perf_counter()
        matrix=np.asarray(drift_matrix(gap)@hard_edge_solenoid_matrix(excitation,length))@upstream
        spatial=matrix[[0,2]]; covariance=spatial@source.total_covariance@spatial.T
        widths=np.sqrt(np.linalg.eigvalsh(covariance))/pixel
        row=dict(candidate_index=state_index, normalized_dac=float(control), q_rad=float(excitation),
                 optical_rms_principal_pixels=widths.tolist())
        # A detector-domain gate, not a hidden optics prior or moving topology.
        if widths.min() < .25:
            row.update(admitted=False, reason='Unblurred principal RMS below fixed0.25-pixel gate; narrow-image width/quadrature not certified.', frames_generated=0)
            rows.append(row);print(f'state{state_index}: excluded RMS{widths}',flush=True);continue
        frame_rows=[];frame_parameters=[];frame_covariances=[];initials=[]
        expected_stack=[];observed_stack=[];fitted_stack=[];variance_stack=[]
        for frame_index,mean in enumerate(canonical_means):
            expected=np.asarray(operator(spatial@mean,covariance,incident_electrons))
            variance=expected+read_noise**2
            seed=91000+100*state_index+frame_index
            poisson_key,read_key=jax.random.split(jax.random.PRNGKey(seed))
            observed=np.asarray(jax.random.poisson(poisson_key,jnp.asarray(expected),shape=expected.shape)
                +read_noise*jax.random.normal(read_key,expected.shape))
            initial=image_initializer(observed,detector,read_noise)
            fit=fit_gaussian_image(observed,variance,detector,initial_parameters=initial,
                                   quadrature_order=16,max_nfev=100)
            try: fit.require_accepted(); admitted=True;reason=None
            except ValueError as error: admitted=False;reason=str(error)
            frame_rows.append(dict(frame=frame_index,seed=seed,admitted=admitted,reason=reason,
                capture_fraction=float(expected.sum()/incident_electrons),
                mean_xy_pixels=np.asarray(spatial@mean/pixel).tolist(),
                rank=fit.numerical_rank,nfev=fit.optimizer_result.nfev,
                fit_success=bool(fit.optimizer_result.success),termination=fit.optimizer_result.status,
                residual_rms_sigma=fit.residual_rms_sigma,chi_square_z=fit.chi_square_z,
                mean_standard_errors_pixels=np.sqrt(np.diag(fit.parameter_covariance)[:2]).tolist(),
                quadrature_rms_sigma=fit.quadrature_rms_sigma,
                quadrature_parameter_shift_sigma=fit.quadrature_parameter_shift_sigma))
            frame_parameters.append(fit.parameters);frame_covariances.append(fit.parameter_covariance)
            expected_stack.append(expected);observed_stack.append(observed);fitted_stack.append(fit.fitted_counts_yx)
            variance_stack.append(variance);initials.append(initial)
        row.update(admitted=all(v['admitted'] for v in frame_rows),frames_generated=9,frames=frame_rows,
                   elapsed_seconds=time.perf_counter()-start_time)
        if row['admitted']:
            response,response_cov=signed_response_from_image_fits(frame_parameters,frame_covariances)
            responses.append(response);covariances.append(response_cov);accepted_t.append(control)
            accepted_indices.append(state_index);matrices.append(matrix)
        rows.append(row)
        all_expected.append(expected_stack);all_observed.append(observed_stack)
        all_fitted.append(fitted_stack);all_variance.append(variance_stack)
        all_parameters.append(frame_parameters);all_covariances.append(frame_covariances);all_initial.append(initials)
        print(f'state{state_index}: admitted={row["admitted"]}, RMS{widths}, {row["elapsed_seconds"]:.1f}s',flush=True)
    if len(responses)<4:
        raise RuntimeError('Too few independently admitted image-derived response states.')
    responses,covariances=np.asarray(responses),np.asarray(covariances)
    initial=np.asarray(reference['inputs']['initial_parameters'])
    response_fit=fit_response(jnp.asarray(accepted_t),jnp.asarray(responses),covariances,initial,truth,inputs,pixel)
    scale_fit=fit_response(jnp.asarray(accepted_t),jnp.asarray(responses),covariances,initial,truth,inputs,pixel,.002)
    # Independent actual pure Gaussian propagation/evaluation for narrowest and
    # broadest admitted states, each with the largest calibrated displacement.
    checks=[]
    widths_all=np.array([np.sqrt(np.linalg.eigvalsh(m[[0,2]]@source.total_covariance@m[[0,2]].T))/pixel for m in matrices])
    for accepted_index in sorted(set([int(np.argmin(widths_all[:,0])),int(np.argmax(widths_all[:,1]))])):
        matrix=matrices[accepted_index];spatial=matrix[[0,2]]
        mean_index=int(np.argmax(np.linalg.norm(canonical_means@spatial.T,axis=1)))
        mode_source=MixedGaussianSource(source.position_covariance_xy_m2,source.curvature_xy_per_m,
                                       source.centre_covariance,canonical_means[mean_index])
        expected=np.asarray(operator(spatial@mode_source.mean_state,
                                     spatial@source.total_covariance@spatial.T,incident_electrons))
        sigma=np.sqrt(expected+read_noise**2)
        renders={}
        for order in (8,16):
            image,symplectic_error=render_pure_counts(mode_source,matrix,detector,
                pixel_order=order,incident_electrons=incident_electrons)
            renders[order]=np.asarray(image)
        difference=(renders[16]-expected)/sigma
        checks.append(dict(candidate_index=accepted_indices[accepted_index],frame=mean_index,
            pure_gaussian_mode_count=1,incoherent_mixture=False,
            balanced_symplectic_maximum_error=symplectic_error,
            image_rms_sigma=float(np.sqrt(np.mean(difference**2))),
            image_whitened_l2=float(np.linalg.norm(difference)),
            quadrature8_to16_whitened_l2=float(np.linalg.norm((renders[16]-renders[8])/sigma)),
            passed=bool(np.linalg.norm(difference)<.2 and np.linalg.norm((renders[16]-renders[8])/sigma)<.2)))
        check_path=output.with_name(f'last_lens_gaussian_check_{accepted_indices[accepted_index]}.npz')
        np.savez_compressed(check_path,
            gaussian_counts_yx=renders[16],gaussian_pixel8_counts_yx=renders[8],
            covariance_counts_yx=expected,noise_sigma_yx=sigma,transfer_matrix=matrix)
        checks[-1].update(arrays_path=str(check_path.relative_to(root)),arrays_sha256=digest(check_path))
    arrays_path=output.with_suffix('.npz')
    np.savez_compressed(arrays_path,expected_counts=np.asarray(all_expected),noisy_counts=np.asarray(all_observed),
        fitted_counts=np.asarray(all_fitted),variance_counts=np.asarray(all_variance),
        image_parameters=np.asarray(all_parameters),image_parameter_covariances=np.asarray(all_covariances),
        image_initial_parameters=np.asarray(all_initial),candidate_normalized_dac=t,candidate_q=q,
        accepted_candidate_indices=np.asarray(accepted_indices),response=responses,response_covariance=covariances,
        accepted_normalized_dac=np.asarray(accepted_t),
        generated_candidate_indices=np.asarray([r['candidate_index'] for r in rows if r['frames_generated']]),
        input_states_grouped=mean_inputs,canonical_source_covariance=source.total_covariance)
    sigma_geometry=np.asarray(response_fit['length_gap_standard_errors_percent'])/100
    # This shot-noise asymptotic rescaling is not a new simulation or a hardware recommendation.
    asymptotic_dose=incident_electrons*(sigma_geometry/.01)**2
    report=dict(schema='neoarm-last-lens-images-v1',backend=jax.default_backend(),jax_version=jax.__version__,
        x64_enabled=jax.config.x64_enabled,source_hashes=source_hashes,
        scope='Vacuum image-derived local PL1 tomography; conditional calibrated inputs and source preparation, not full microscope inversion.',
        incident_electrons_per_frame=incident_electrons,read_noise_electrons=read_noise,
        detector=dict(shape_yx=detector.shape_yx,pixel_size_yx_m=detector.pixel_size_yx_m,
                      mtf_sigma_pixels=detector.mtf_sigma_px,affine_xy=np.asarray(detector.affine_xy).tolist()),
        source=dict(kind='single pure astigmatic Gaussian',intensity_rms_xy_m=[2e-9,1.5e-9],
                    wavelength_m=source.wavelength_m,canonical_covariance=source.total_covariance.tolist()),
        candidate_states=rows,admitted_state_count=len(responses),generated_frame_count=9*len(all_expected),
        response_fit=response_fit,response_fit_with_0p2percent_input_scale_priors=scale_fit,
        independent_gaussian_checks=checks,
        shot_noise_asymptotic_incident_electrons_for_1percent_length_gap=asymptotic_dose.tolist(),
        arrays_path=str(arrays_path.relative_to(root)),arrays_sha256=digest(arrays_path),
        limitations=['Raw observations are Poisson plus independent Gaussian read noise; the WLS variance is exact synthetic truth and held fixed.',
            'The single pure Gaussian and its2nm preparation are conditional; no source-covariance or condenser recovery is claimed.',
            'Finite pixels, detector clipping and external PSF halo are included without detected-flux renormalization.',
            'All six image coordinates are fitted; signed-response covariance uses marginalized2x2 mean blocks and cancels the shared reference exactly.',
            'Reference and signed basis frames assume unchanged alignment within each group; differential drift and calibration error are not simulated.',
            'Four0.2percent log input-scale priors are shared across states and expose an otherwise exact coefficient-column gauge.',
            'Only a locally initialized excitation branch is fitted; all widths and hardware control/slew/aperture limits need instrument checks.',
            'Dose rescaling is an optimistic shot-noise asymptote, not a demonstrated acquisition or hardware-safe dose.'])
    report['image_response_demonstration_passed']=bool(all(c['passed'] for c in checks)
        and response_fit['geometry_inference_admitted'] and scale_fit['geometry_inference_admitted'])
    if any(digest(root/name)!=value for name,value in source_hashes.items()):
        raise RuntimeError('A provenance-bound source changed during this audit.')
    report['source_hashes_unchanged_during_execution']=True
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--incident-electrons',type=float,default=1e6)
    parser.add_argument('--state-count',type=int,default=9)
    args=parser.parse_args()
    result=audit(args.output,incident_electrons=args.incident_electrons,state_count=args.state_count)
    print(json.dumps({k:result[k] for k in ('image_response_demonstration_passed','admitted_state_count',
        'response_fit','response_fit_with_0p2percent_input_scale_priors')},indent=2))
