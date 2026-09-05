"""PL1-only tomography from signed, calibrated sample-to-detector responses.

All matrices here use grouped coordinates [x,y,theta_x,theta_y]. For a final
hard-edge solenoid of length L and following free gap d, let s=d/L and let
F_mech=[U;V] be the unknown upstream map from mechanical sample coordinates.

    R(q) H(q) = cos(q) W0 + q sin(q) W1 + sinc(q) W2
    W0=U+d V,  W1=-s U,  W2=L V
    W0=-W1/s+s W2,  det([W1;W2])=d**2.

Thus d=+sqrt(det([W1;W2])) and L=d/s. The unknown sample magnetic shear
T=[[I,0],[kappa J2,I]] has determinant one, so it does not invalidate this
four-dimensional volume identity. A one-plane determinant is insufficient.
The example fits the upstream coefficient matrices, q bias, q slope and s;
it never grants the sample magnetic field to the estimator. Native excitation
is assumed affine within the sweep. Only a local same-polarity solution is
demonstrated, not exclusion of alternative branches or real-instrument fit.
"""

from pathlib import Path
import argparse
import hashlib
import json
import os

os.environ.setdefault('JAX_ENABLE_X64', 'true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import jax
import jax.numpy as jnp
import numpy as np

from neoarm_imaging_recipe import geometry_anchors_from_notebook
from temgym_core.jax_least_squares import least_squares
from temgym_core.microscope_inversion import drift_matrix, hard_edge_solenoid_matrix

J2 = jnp.array([[0., -1.], [1., 0.]])
J4 = jnp.block([[jnp.zeros((2, 2)), jnp.eye(2)], [-jnp.eye(2), jnp.zeros((2, 2))]])
INTERLEAVED_TO_GROUPED = jnp.array([0, 2, 1, 3])


def rotation(angle):
    c, s = jnp.cos(angle), jnp.sin(angle)
    return jnp.stack((jnp.stack((c, -s), -1), jnp.stack((s, c), -1)), -2)


def mechanical_shear(kappa_per_m):
    """Canonical = T @ mechanical; kappa includes the charge/sign convention."""
    return jnp.eye(4).at[2:, :2].set(kappa_per_m*J2)


def signed_position_response(q, length_m, gap_m, upstream_mechanical, *, frame_rotation_rad=0.):
    """Physical H in metres/radians, with no source-covariance observations."""
    q = jnp.atleast_1d(q)
    a = jnp.cos(q)-(gap_m/length_m)*q*jnp.sin(q)
    b = length_m*jnp.sinc(q/jnp.pi)+gap_m*jnp.cos(q)
    radial = a[:, None, None]*upstream_mechanical[:2]+b[:, None, None]*upstream_mechanical[2:]
    return jnp.einsum('nij,njk->nik', rotation(frame_rotation_rad-q), radial)


def known_q_coefficients(q, response):
    """Fit the three physical 2x4 coefficient matrices at known excitations."""
    q, response = jnp.asarray(q), jnp.asarray(response)
    if q.ndim != 1 or response.shape != (q.size, 2, 4) or q.size < 3:
        raise ValueError('Expected at least three excitations and response shape (N,2,4).')
    design = jnp.stack((jnp.cos(q), q*jnp.sin(q), jnp.sinc(q/jnp.pi)), axis=-1)
    if np.linalg.matrix_rank(np.asarray(design)) != 3:
        raise ValueError('The excitation basis is rank deficient; distinct |q| values are needed.')
    derotated = jnp.einsum('nij,njk->nik', rotation(q), response)
    coefficients = jnp.linalg.lstsq(design, derotated.reshape(q.size, 8), rcond=None)[0]
    return coefficients.reshape(3, 2, 4)


def geometry_from_coefficients(coefficients):
    """Recover L,d and F_mech after undoing detector/input-step calibrations.

    This algebraic diagnostic is intended for exact coefficient checks. Noisy
    inference below fits the response model jointly and propagates covariance.
    Column equilibration changes numerical conditioning, not the SI volume.
    """
    w = np.asarray(coefficients, float)
    if w.shape != (3, 2, 4) or not np.all(np.isfinite(w)):
        raise ValueError('Expected three finite physical 2x4 coefficient matrices.')
    column_scale = np.sqrt(np.sum(w**2, axis=(0, 1)))
    if np.any(column_scale == 0):
        raise ValueError('Four independent calibrated input directions are required.')
    equilibrated = w/column_scale
    relation = np.linalg.lstsq(equilibrated[1:].reshape(2, 8).T,
                               equilibrated[0].ravel(), rcond=None)[0]
    if relation[0] >= 0 or relation[1] <= 0:
        raise ValueError('Coefficient relation does not give a positive detector-gap/length ratio.')
    s = np.sqrt(-relation[1]/relation[0])
    stack = w[1:].reshape(4, 4)
    sign, log_volume = np.linalg.slogdet(stack/column_scale)
    if sign <= 0:
        raise ValueError('The calibrated four-dimensional volume must be positive.')
    gap = np.exp(.5*(log_volume+np.sum(np.log(column_scale))))
    length = gap/s
    upstream = np.concatenate((-w[1]/s, w[2]/length))
    omega = upstream.T@np.asarray(J4)@upstream
    return dict(length_m=float(length), detector_gap_m=float(gap), ratio_s=float(s),
        relation_coefficients=relation.tolist(),
        relation_relative_residual=float(np.linalg.norm(w[0]+w[1]/s-s*w[2])/np.linalg.norm(w[0])),
        recovered_upstream_mechanical=upstream.tolist(), sample_pullback_twoform=omega.tolist(),
        inferred_kappa_per_m=float((omega[1, 0]-omega[0, 1])/4))


def predict_response_parameters(parameters, normalized_dac):
    """Nineteen free coordinates: log(s), q0, dq/dt, W1(8), W2(8)."""
    p, t = jnp.asarray(parameters), jnp.asarray(normalized_dac)
    s, q = jnp.exp(p[0]), p[1]+p[2]*t
    w = p[3:].reshape(2, 2, 4)
    radial = ((q*jnp.sin(q)-jnp.cos(q)/s)[:, None, None]*w[0]
              +(jnp.sinc(q/jnp.pi)+s*jnp.cos(q))[:, None, None]*w[1])
    return jnp.einsum('nij,njk->nik', rotation(-q), radial)


def geometry_coordinates(parameters, input_steps, pixel_m):
    """[log L, log d, log s, q0, dq/dt] for exact Jacobian propagation."""
    # Fitted W blocks are displacements in pixels at the calibrated input
    # steps, not transfer coefficients in SI units. Undo that column scale.
    scale = jnp.asarray(input_steps)/pixel_m
    log_d = .5*(jnp.linalg.slogdet(parameters[3:].reshape(4, 4))[1]-jnp.sum(jnp.log(scale)))
    return jnp.array([log_d-parameters[0], log_d, parameters[0], parameters[1], parameters[2]])


def audit(output):
    root = Path(__file__).resolve().parents[2]
    reference_path = root/'examples/generated/neoarm_inversion/imaging_recipe/measured_initialization.json'
    geometry_path = root/'examples/microscope_models/measure_microscope_geometry.ipynb'
    config_path = root/'examples/lens_inversion/neoarm_inversion_config.toml'
    initialization_source = Path(__file__).with_name('neoarm_measured_initialization.py')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    reference = json.loads(reference_path.read_text())
    for key, path in (('source_sha256', initialization_source), ('geometry_notebook_sha256', geometry_path),
                      ('config_sha256', config_path)):
        if digest(path) != reference[key]:
            raise ValueError(f'Stale synthetic upstream reference: {path}')
    paths = [Path(__file__), reference_path, geometry_path, config_path, initialization_source,
        Path(__file__).with_name('neoarm_imaging_recipe.py'),
        *(root/'src/temgym_core'/name for name in ('jax_least_squares.py', 'microscope_inversion.py'))]
    sources = {str(path.relative_to(root)): digest(path) for path in paths}
    physical = dict(zip(reference['parameter_names'], reference['parameters']))
    anchors = geometry_anchors_from_notebook(geometry_path)
    length = np.exp(physical['PL1.log_length_m'])
    gap = anchors['detector_z_m']-(physical['PL1.z_center_m']+length/2)
    s, native_bias, native_slope = gap/length, physical['PL1.q_bias'], physical['PL1.q_slope_per_dac']
    nominal = jnp.asarray(reference['nominal_matrices'][0])
    tail = drift_matrix(gap)@hard_edge_solenoid_matrix(native_bias, length)
    upstream = jnp.linalg.solve(tail, nominal)
    ix = INTERLEAVED_TO_GROUPED
    upstream = upstream[ix[:, None], ix[None, :]]
    kappa = 800.
    mechanical = upstream@mechanical_shear(kappa)
    t = jnp.linspace(-1., 1., 25)
    q_centre, q_slope = -np.pi/2, -(np.pi/2-.05)
    q = q_centre+q_slope*t
    inputs, pixel_m, sigma = jnp.array([5e-9, 5e-9, 1e-3, 1e-3]), 55e-6, 1e-4
    physical_response = signed_position_response(q, length, gap, mechanical)
    coefficients = known_q_coefficients(q, physical_response)
    known = geometry_from_coefficients(coefficients)
    scaling = inputs/pixel_m
    w = jnp.stack((-s*mechanical[:2]*scaling, length*mechanical[2:]*scaling))
    truth = jnp.concatenate((jnp.array([np.log(s), q_centre, q_slope]), w.ravel()))
    expected = predict_response_parameters(truth, t)
    exact_response_error = float(jnp.max(jnp.abs(expected-physical_response*scaling)))
    observed = expected+sigma*jax.random.normal(jax.random.PRNGKey(731), expected.shape)
    jacobian = np.asarray(jax.jacfwd(predict_response_parameters)(truth, t)).reshape(-1, 19)/sigma
    norms = np.linalg.norm(jacobian, axis=0)
    _, singular, right = np.linalg.svd(jacobian/norms, full_matrices=False)
    rank = int(np.sum(singular > singular[0]*1e-10))
    factor = (right.T/singular)/norms[:, None]
    geometry_jacobian = np.asarray(jax.jacfwd(geometry_coordinates)(truth, inputs, pixel_m))
    geometry_sigma = np.sqrt(np.sum((geometry_jacobian@factor)**2, axis=1))
    initial = np.asarray(truth).copy()
    initial[:3] += [.03, .025, -.025]
    initial[3:] += sigma*np.asarray(jax.random.normal(jax.random.PRNGKey(711), (16,)))
    lower = np.r_[np.log(1.), -4., -3., np.full(16, -1e6)]
    upper = np.r_[np.log(100.), -.01, -.01, np.full(16, 1e6)]
    fit = least_squares(lambda p: ((predict_response_parameters(p, t)-observed)/sigma).ravel(), initial,
        bounds=(lower, upper), x_scale=1/norms, max_nfev=400, ftol=1e-12, gtol=1e-8, xtol=1e-12)
    geometry_error = np.asarray(geometry_coordinates(fit.x, inputs, pixel_m)-geometry_coordinates(truth, inputs, pixel_m))
    determinant_sign = float(jnp.linalg.slogdet(fit.x[3:].reshape(4, 4))[0])
    dof = expected.size-19
    chi_square = float(np.sum(np.asarray(fit.residual)**2))
    native_centre = (q_centre-native_bias)/native_slope
    native_half_span = q_slope/native_slope
    fitted_native_slope = float(fit.x[2]/native_half_span)
    fitted_native_bias = float(fit.x[1]-fitted_native_slope*native_centre)
    report = dict(schema='neoarm-last-lens-tomography-v1',
        scope='Local conditional PL1 hard-edge response tomography; no detector images, hardware recipe or branch uniqueness proof.',
        passed=bool(fit.success and rank == 19 and determinant_sign > 0 and exact_response_error < 1e-10
            and abs(known['length_m']/length-1) < 1e-7 and abs(known['detector_gap_m']/gap-1) < 1e-7
            and np.max(np.abs(geometry_error[:2]/geometry_sigma[:2])) < 3
            and abs((chi_square-dof)/np.sqrt(2*dof)) < 5),
        source_hashes=sources, backend=jax.default_backend(), jax_version=jax.__version__,
        x64_enabled=jax.config.x64_enabled, coordinate_order=['x','y','theta_x','theta_y'],
        known_q_derivation_check=known, exact_model_response_maximum_error_pixels=exact_response_error,
        inputs=dict(upstream_canonical=np.asarray(upstream).tolist(), generating_unknown_kappa_per_m=kappa,
            upstream_mechanical=np.asarray(mechanical).tolist(), generating_length_m=length, generating_detector_gap_m=gap,
            normalized_dac=np.asarray(t).tolist(), generating_q=np.asarray(q).tolist(),
            illustrative_native_dac=np.asarray(native_centre+native_half_span*t).tolist(),
            input_basis_steps_grouped=np.asarray(inputs).tolist(), detector_pixel_m=pixel_m,
            response_noise_standard_deviation_pixels=sigma, expected_response_pixels=np.asarray(expected).tolist(),
            observed_response_pixels=np.asarray(observed).tolist(), noise_seed=731, initializer_seed=711,
            initial_parameters=initial.tolist(), truth_parameters=np.asarray(truth).tolist(),
            lower_bounds=lower.tolist(), upper_bounds=upper.tolist()),
        fit=dict(success=fit.success, termination=fit.status, nfev=fit.nfev, parameters=np.asarray(fit.x).tolist(),
            residual_rms_sigma=float(np.sqrt(chi_square/expected.size)), degrees_of_freedom=dof,
            chi_square_z=float((chi_square-dof)/np.sqrt(2*dof)), coefficient_volume_sign=determinant_sign,
            length_gap_errors_percent=(100*np.expm1(geometry_error[:2])).tolist(),
            length_gap_errors_sigma=(geometry_error[:2]/geometry_sigma[:2]).tolist(),
            q_centre_error_rad=float(geometry_error[3]), q_slope_error_per_normalized_dac=float(geometry_error[4]),
            fitted_native_q_bias=fitted_native_bias, fitted_native_q_slope=fitted_native_slope),
        local_information=dict(parameter_rank=rank, column_scaled_singular_values=singular.tolist(),
            column_scales=(1/norms).tolist(), conditional_parameter_covariance=(factor@factor.T).tolist(),
            geometry_coordinate_names=['log_length','log_gap','log_ratio','q_centre','q_slope'],
            geometry_standard_errors=geometry_sigma.tolist(),
            length_gap_standard_errors_percent=(100*geometry_sigma[:2]).tolist(),
            length_gap_standard_errors_percent_at_0p001_pixel=(1000*geometry_sigma[:2]).tolist(),
            maximum_position_basis_response_pixels=float(jnp.max(jnp.abs(expected[:, :, :2]))),
            maximum_angle_basis_response_pixels=float(jnp.max(jnp.abs(expected[:, :, 2:])))),
        assumptions=['The sample magnetic shear is a generating truth only, not an estimator input or fixed fit parameter.',
            'All sixteen upstream coefficient coordinates and q bias/slope/length-gap ratio are free.',
            'Responses are signed means for four calibrated mechanical inputs; both transverse planes are measured.',
            'Independent homoscedastic response precision is conditional, not established by a detector-image experiment.',
            'The opposite q polarity and other excitation branches are not searched; failed searches would remain unresolved.',
            'Hard-edge finite-field optics and an affine native excitation law are model assumptions requiring held-out tests.',
            'The illustrative native coordinates are synthetic and are not approved hardware commands.',
            'Calibrated source-to-sample condenser measurements remain a separate requirement.'])
    if any(digest(root/path) != value for path, value in sources.items()):
        raise RuntimeError('An audited source changed during the tomography example.')
    report['source_hashes_unchanged_during_execution'] = True
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.output)
    print(json.dumps({'passed': result['passed'], 'fit': result['fit'],
                      'local_information': {key: result['local_information'][key] for key in
                          ('parameter_rank','length_gap_standard_errors_percent','maximum_angle_basis_response_pixels')}}, indent=2))
