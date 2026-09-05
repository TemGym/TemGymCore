"""Shared external calibration uncertainty for the three-projector image fit.

This is an uncertainty audit of an existing stationary fit, not a new fit or
an instrument calibration. Position, tilt and detector-pitch errors are shared
across every frame. The round-lens response has an exact compensating family,
so the data cannot independently determine these three calibration gains.
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
from jax.scipy.linalg import solve_triangular
import numpy as np

from neoarm_il2_extension_design import ThreeProjectorResponse
from neoarm_last_lens_physical_fit import RoundLensResponse
from temgym_core.physical_nuisance import GaussianPriorFactor
from temgym_core.round_calibration import (
    GAIN_NAMES, calibrated_position_response, round_response_gain_family,
)

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT/'examples/generated/neoarm_inversion/imaging_recipe'
LENGTH_INDICES = np.array([0, 1, 4, 5, 8, 9])


def compensation(parameters, log_gains):
    """Physical coordinates giving identical observations at actual gains."""
    p = jnp.asarray(parameters)
    stages, prefix = round_response_gain_family(
        p[:12].reshape(3, 4), p[12:], -jnp.asarray(log_gains))
    return jnp.r_[stages.reshape(-1), prefix]


def svd_information(jacobian):
    jac = np.asarray(jacobian)
    norms = np.linalg.norm(jac, axis=0)
    scales = np.where(norms > 0, norms, 1.)
    _, singular, right = np.linalg.svd(jac/scales, full_matrices=False)
    rank = int(np.sum(singular > singular[0]*1e-10))
    covariance = None
    if rank == jac.shape[1]:
        factor = (right.T/singular)/scales[:, None]
        covariance = factor@factor.T
    return rank, singular, covariance


def joint_covariance(statistical_covariance, compensation_derivative):
    """Independent effective optics and standard calibration latents."""
    c, b = map(np.asarray, (statistical_covariance, compensation_derivative))
    return np.block([[c+b@b.T, b], [b.T, np.eye(b.shape[1])]])


def audit(output, *, calibration_sigma=.002):
    if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
        raise RuntimeError('Use GPU float64 for the optical forward audit.')
    if not np.isfinite(calibration_sigma) or calibration_sigma <= 0:
        raise ValueError('A positive finite illustrative calibration sigma is required.')
    source = FOLDER/'three_projector_recovery.json'
    reference = json.loads(source.read_text())
    sources = dict(reference['source_hashes'])
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for name, checksum in sources.items():
        if digest(ROOT/name) != checksum:
            raise ValueError(f'Stale fit dependency: {name}')
    best = reference['best']
    if (not best['optimizer_success'] or best['active_bounds']
            or best['normalized_gradient_max'] > 1e-5):
        raise ValueError('The audit requires a stationary interior optical fit.')
    manifest_path = ROOT/reference['new_inference_report']
    manifest = json.loads(manifest_path.read_text())
    calibration = RoundLensResponse(tuple(manifest['input_steps_grouped']),
                                   manifest['detector_pixel_m'])
    model = ThreeProjectorResponse(calibration)
    p = jnp.asarray(best['physical_coordinates'])
    controls = jnp.asarray(reference['training_controls'])
    data = jnp.asarray(reference['training_response'])
    chol = jnp.asarray(np.linalg.cholesky(reference['training_response_covariance']))
    prior = GaussianPriorFactor.from_covariance(
        'external_scale_calibration', GAIN_NAMES, np.zeros(3),
        np.eye(3)*calibration_sigma**2,
        provenance='Illustrative independent log-gain uncertainties; not measured NEOARM values.')

    def data_residual(full):
        prediction = calibrated_position_response(
            model.predict(full[:17], controls), prior.values(full[17:]))
        return solve_triangular(chol, (prediction-data).reshape(-1), lower=True)

    def residual(full):
        return jnp.r_[data_residual(full), full[17:]]

    full = jnp.r_[p, jnp.zeros(3)]
    jac = np.asarray(jax.jit(jax.jacfwd(residual))(full))
    data_rank, data_singular, _ = svd_information(jac[:-3])
    rank, singular, numerical_cov = svd_information(jac)
    old_cov = np.asarray(reference['information']['physical_covariance'])
    b = np.asarray(jax.jacfwd(lambda z: compensation(p, prior.values(z)))(jnp.zeros(3)))
    analytic_cov = joint_covariance(old_cov, b)
    relative_covariance_error = np.linalg.norm(numerical_cov-analytic_cov)/np.linalg.norm(analytic_cov)
    likelihood = np.asarray(data_residual(full))
    gradient = jac.T@np.asarray(residual(full))

    # These are finite, exact equal-likelihood families, not a linearized claim.
    family = []
    nominal = np.asarray(model.predict(p, controls))
    for gains in ([-.01, 0., 0.], [0., .01, 0.], [0., 0., .01], [.01, -.006, .004]):
        gain = jnp.asarray(gains)
        transformed = compensation(p, gain)
        changed = np.asarray(calibrated_position_response(model.predict(transformed, controls), gain))
        family.append(dict(log_gains=gains,
            maximum_response_change_pixels=float(np.max(abs(changed-nominal))),
            common_length_ratio=float(jnp.exp(transformed[0]-p[0]))))

    # Include optics/calibration cross-covariance when predicting the same
    # calibrated inputs. No held-out image values are used by this audit.
    held_controls = np.asarray(reference['fresh_heldout']['controls']) if 'controls' in reference['fresh_heldout'] else None
    if held_controls is None:
        with np.load(ROOT/manifest['arrays_path'], allow_pickle=False) as arrays:
            pair = arrays['downstream_controls']
            offsets = arrays['il2_controls'][arrays['heldout_mask']]
            held_controls = np.array([[*a, u] for u in offsets for a in pair])
    predict = lambda q: calibrated_position_response(model.predict(q[:17], held_controls), prior.values(q[17:])).reshape(-1)
    held_jac = np.asarray(jax.jit(jax.jacfwd(predict))(full))
    conditional_prediction_cov = held_jac[:, :17]@old_cov@held_jac[:, :17].T
    marginalized_prediction_cov = held_jac@analytic_cov@held_jac.T
    prediction_error = np.linalg.norm(marginalized_prediction_cov-conditional_prediction_cov)/np.linalg.norm(conditional_prediction_cov)
    floor = np.linalg.norm(b[LENGTH_INDICES], axis=1)
    forecasts = []
    for repeats in (1, 4, 8, 12, 16):
        forecasts.append(dict(independent_repeat_count=repeats,
            length_gap_sigma_percent=(100*np.sqrt(np.diag(old_cov)[LENGTH_INDICES]/repeats+floor**2)).tolist()))
    for path in (Path(__file__), source, ROOT/'src/temgym_core/round_calibration.py',
                 ROOT/'src/temgym_core/physical_nuisance.py'):
        sources[str(path.relative_to(ROOT))] = digest(path)
    report = dict(
        schema='neoarm-three-projector-shared-calibration-audit-v1',
        source_hashes=sources, backend=jax.default_backend(), x64_enabled=jax.config.x64_enabled,
        reference_report=str(source.relative_to(ROOT)),
        interpretation='Exact conditional-fit calibration families and local covariance audit; no new nonlinear fit.',
        calibration_gain_names=GAIN_NAMES, calibration_covariance=prior.covariance.tolist(),
        calibration_latent_names=prior.latent_names, calibration_factor=prior.factor.tolist(),
        joint_covariance_coordinate_convention='First17 physical lens/prefix coordinates, then standard-normal calibration latents z; log_gains=factor@z.',
        physical_calibration_derivative_convention='Derivative of compensated physical coordinates with respect to standard-normal calibration latents z.',
        calibration_provenance=prior.provenance, prior_count=1,
        physical_coordinate_count=17, calibration_latent_count=3,
        data_rank=data_rank, augmented_rank=rank,
        data_normalized_singular_values=data_singular.tolist(),
        augmented_normalized_singular_values=singular.tolist(),
        training_chi_square=float(likelihood@likelihood),
        normalized_gradient_max=float(np.max(abs(gradient)/np.linalg.norm(jac, axis=0))),
        exact_calibration_families=family,
        covariance_closure_relative_error=float(relative_covariance_error),
        shared_input_predictive_covariance_closure_relative_error=float(prediction_error),
        joint_physical_calibration_covariance=analytic_cov.tolist(),
        physical_calibration_derivative=b.tolist(),
        statistical_length_gap_sigma_percent=(100*np.sqrt(np.diag(old_cov)[LENGTH_INDICES])).tolist(),
        marginalized_length_gap_sigma_percent=(100*np.sqrt(np.diag(analytic_cov)[LENGTH_INDICES])).tolist(),
        common_calibration_length_floor_percent=(100*floor).tolist(),
        independent_repeat_forecast=forecasts,
        forecast_scope='Local fixed-noise independent repeats of identical controls; ignores model error, drift, saturation and damage.',
        likelihood_identifies_calibration_gains=False,
        limitations=['Actual sample-position, tilt and detector calibration covariance remains to be measured.',
            'These are errors of already calibrated physical states, not fitted beam-command gains or crosstalk.',
            'Prior uncertainty is shared across frames; repeated observations do not reduce its scale floor.',
            'Exact families assume free round-lens geometry and prefix, inactive bounds, and no independent geometry constraints.',
            'This local uncertainty audit does not establish global branch uniqueness or real-field adequacy.'])
    if (data_rank != 17 or rank != 20 or relative_covariance_error > 1e-7
            or prediction_error > 1e-7 or max(r['maximum_response_change_pixels'] for r in family) > 1e-8):
        raise RuntimeError('Calibration-family or covariance closure failed.')
    if any(digest(ROOT/name) != value for name, value in sources.items()):
        raise RuntimeError('A dependency changed during the audit.')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calibration-sigma', type=float, default=.002)
    args = parser.parse_args()
    result = audit(args.output, calibration_sigma=args.calibration_sigma)
    print('Shared scale floor (%):', result['common_calibration_length_floor_percent'])
    print('Saved', args.output)
