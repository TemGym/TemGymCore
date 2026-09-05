"""Training-only second-moment variable projection for cold initialization.

The guide projects nine unconstrained visible source coefficients by full-rank
weighted QR. Its covariance comes from a delta-method transformation of the
MEASURED log-width/correlation summaries. This temporary statistical objective
is not substituted for the final log-observable likelihood or its single
common aperture-radius prior. Geometry can move while the guide's provisional
source is unphysical; only a physical final guide source may seed the final fit.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault('JAX_ENABLE_X64', 'true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import jax
if __name__ == '__main__':
    jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from temgym_core.feasible_least_squares import svd_gauss_newton
from neoarm_condenser_extension_recovery import selected_data, beam_residual
from neoarm_condenser_extension_design import load_context, training_residual, covariance_factor, FOLDER
from neoarm_condenser_covariance_fit import VisibleSourceModel, BASIS, SOURCE_SCALE, quantum_completion, geometry_gaps
from neoarm_condenser_image_recovery import rotation, finite_json
from neoarm_condenser_reference_source import ReferenceSourceChart
from neoarm_condenser_cold_start import cold_initializers, ORDER, INVERSE
from neoarm_condenser_cl1_profile import source_diagnostic
from neoarm_condenser_scaled_cl1_profile import scaled_profiled_least_squares
from neoarm_shared_objective_audit import provenance_snapshot
from neoarm_aperture_capture import ROOT


def observable_to_moments(observable):
    """[log sigma_x, log sigma_y, atanh rho] -> [Cxx,Cxy,Cyy]."""
    p = jnp.asarray(observable)
    return jnp.array([jnp.exp(2*p[0]), jnp.exp(p[0]+p[1])*jnp.tanh(p[2]), jnp.exp(2*p[1])])


class MomentSourceProjection:
    def __init__(self, quotient, beam_controls, beam_observables, beam_covariances,
                 aperture_controls, aperture_observables, aperture_covariances):
        self.quotient = quotient
        self.beam_controls = jnp.asarray(beam_controls)
        self.aperture_controls = jnp.asarray(aperture_controls)
        measured = np.concatenate((np.asarray(beam_observables).reshape(-1, 3), aperture_observables))
        covariance = np.concatenate((np.asarray(beam_covariances).reshape(-1, 3, 3), aperture_covariances))
        if not np.isfinite(measured).all() or not np.isfinite(covariance).all():
            raise ValueError('Finite measured summaries and covariances are required.')
        moments = np.asarray(jax.vmap(observable_to_moments)(jnp.asarray(measured)))
        derivative = np.asarray(jax.vmap(jax.jacfwd(observable_to_moments))(jnp.asarray(measured)))
        moment_covariance = derivative@covariance@derivative.transpose(0, 2, 1)
        self.roots = jnp.asarray(np.linalg.cholesky(moment_covariance))
        self.target = jnp.linalg.solve(self.roots, jnp.asarray(moments)[..., None]).reshape(-1)
        self.measured_moments = moments
        self.measured_moment_covariance = moment_covariance

    def optical(self, p):
        full = jnp.r_[p[:12], jnp.zeros(9), p[12:14]]
        return self.quotient.decode(full)[0]

    def weighted_design(self, parameters):
        p = jnp.asarray(parameters)
        optical = self.optical(p)
        model = self.quotient.model
        maps = jax.vmap(lambda u: model.maps(optical, u))(self.beam_controls).reshape(-1, 4, 4)
        aperture = jax.vmap(lambda u: model.condenser_matrix(optical, u, to_aperture=True))(self.aperture_controls)
        beam_rows = maps[:, jnp.asarray([0, 2]), :]/55e-6
        aperture_rows = jnp.einsum('ij,ajk->aik', rotation(p[12]), aperture[:, jnp.asarray([0, 2]), :])*jnp.exp(-p[13])
        rows = jnp.concatenate((beam_rows, aperture_rows))*jnp.asarray(SOURCE_SCALE)[None, None, :]
        transported = jnp.einsum('aij,kjl,aml->akim', rows, jnp.asarray(BASIS), rows)
        design = transported[:, :, jnp.asarray([0, 0, 1]), jnp.asarray([0, 1, 1])].transpose(0, 2, 1)
        return jnp.linalg.solve(self.roots, design).reshape(-1, 9)

    def projected(self, parameters):
        design = self.weighted_design(parameters)
        scales = jnp.linalg.norm(design, axis=0)
        normalized = design/scales
        q, r = jnp.linalg.qr(normalized, mode='reduced')
        coordinates = jnp.linalg.solve(r, q.T@self.target)/scales
        # The rank gate rejects an invalid projection; it never drops columns.
        singular = jax.lax.stop_gradient(jnp.linalg.svd(r, compute_uv=False))
        valid = jnp.all(jnp.isfinite(scales)) & jnp.all(scales > 0) & (singular[-1] > singular[0]*1e-12)
        projected_residual = q@(q.T@self.target)-self.target
        projected_residual = jnp.where(valid, projected_residual, jnp.full_like(projected_residual, jnp.nan))
        return coordinates, projected_residual, singular

    def residual(self, parameters):
        _, residual, _ = self.projected(parameters)
        return jnp.r_[residual, parameters[13]/.005]

    def full_parameters(self, parameters):
        source, _, _ = self.projected(parameters)
        return jnp.r_[parameters[:12], source, parameters[12:14]]

    def admissible(self, parameters):
        p = np.asarray(parameters)
        if p.shape != (14,) or not np.isfinite(p).all():
            return False
        if np.any(abs(p[:12]) > np.tile([1., 4., 1., 3.], 3)) or abs(p[12]) > 2*np.pi or abs(p[13]) > .1:
            return False
        return bool(np.min(geometry_gaps(self.quotient.model, np.asarray(self.optical(p)))) > 1e-9)


def run(output, *, max_moment_iterations=150):
    if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
        raise RuntimeError('The initialization experiment requires GPU float64.')
    if not 0 < max_moment_iterations <= 150:
        raise ValueError('The bounded moment guide permits at most150 iterations per start.')
    started = perf_counter()
    inference_path = FOLDER/'condenser_extension_images.json'
    inference = json.loads(inference_path.read_text())
    array_path = ROOT/inference['arrays_path']
    if hashlib.sha256(array_path.read_bytes()).hexdigest() != inference['arrays_sha256']:
        raise ValueError('The frozen training archive changed.')
    with np.load(array_path, allow_pickle=False) as archive:
        fresh = selected_data(archive, True)
    old, arrays, _, aperture, model = load_context()
    quotient = VisibleSourceModel(model)
    controls = np.concatenate((arrays['controls'][:8], fresh[0]))
    observed = np.concatenate((arrays['beam_observables'][:8], fresh[1]))
    covariance = np.concatenate((arrays['beam_observable_covariances'][:8], fresh[2]))
    ac = np.asarray([row['controls'] for row in aperture['records'][:2]])
    ao, av = arrays['aperture_observables'][:2], arrays['aperture_observable_covariances'][:2]
    seeds = cold_initializers(quotient, controls, observed, covariance, ac, ao, av)
    guide = MomentSourceProjection(quotient, controls, observed, covariance, ac, ao, av)
    rough = next(row for row in seeds if row['sign'] == 0)
    optical, source, _, _ = quotient.decode(rough['parameters'])
    ref_control = arrays['controls'][0]
    m = model.condenser_matrix(optical, ref_control)
    c = np.asarray(m@source@m.T)
    width = float((c[0, 0]*c[2, 2])**.25)
    chart = ReferenceSourceChart(quotient, ref_control, width)
    old_loss, new_loss = training_residual(quotient, arrays, aperture), beam_residual(quotient, fresh)
    original_residual = lambda p: jnp.r_[old_loss(p), new_loss(p)]
    function = jax.jit(original_residual)
    decode = lambda p: chart.decode(jnp.asarray(p)[jnp.asarray(INVERSE)])
    final_residual = lambda p: original_residual(decode(p))
    domain = lambda p: quotient.admissible(np.asarray(decode(p)))
    hashes, _ = provenance_snapshot()
    paths = [Path(__file__), ROOT/'tests/test_neoarm_condenser_moment_initialization.py', inference_path, array_path,
        FOLDER/'condenser_image_recovery.json', ROOT/old['arrays_path'], FOLDER/'condenser_covariance_fit.json',
        FOLDER/'aperture_image_fit.json']
    paths.extend(ROOT/name for name in aperture['report_input_hashes'])
    ap_array = Path(aperture['array_file']); paths.append(ap_array if ap_array.is_absolute() else ROOT/ap_array)
    for path in paths:
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = []
    for seed in seeds:
        initial = np.r_[seed['parameters'][:12], seed['parameters'][21:23]]
        print('moment starting', seed['label'], flush=True)
        fit = svd_gauss_newton(guide.residual, initial, guide.admissible,
            max_iterations=max_moment_iterations, gradient_tolerance=1e-6,
            native_scales=np.r_[np.tile([.02, .1, .05, .1], 3), .05, .005])
        p14 = np.asarray(fit['parameters'])
        p = np.asarray(jax.jit(guide.full_parameters)(p14))
        coefficients, projection, singular = map(np.asarray, jax.jit(guide.projected)(p14))
        design = np.asarray(jax.jit(guide.weighted_design)(p14))
        closure = float(np.linalg.norm(design@coefficients-np.asarray(guide.target)-projection)/max(1., np.linalg.norm(guide.target)))
        completion = quantum_completion(p[12:21])
        physical = bool(completion['feasible'] and quotient.admissible(p))
        row = dict(label=seed['label'], sign=seed['sign'],
            initial_optical_parameters=seed['initial_optical_parameters'].tolist(),
            initial14_parameters=initial.tolist(), source_initialization=seed['source_initialization'],
            moment_parameters=p14.tolist(), moment_original23_parameters=p.tolist(),
            moment_optimizer_success=fit['success'], moment_termination=fit['reason'],
            moment_chi_square=float(fit['residual']@fit['residual']),
            moment_normalized_gradient=fit['normalized_gradient'], moment_history=fit['history'],
            source_projection_singular_values=singular.tolist(), weighted_projection_closure=closure,
            source_physical_after_guide=physical, source_completion_margin=float(completion['margin']),
            moment_guide_geometry_gaps_m=geometry_gaps(model, np.asarray(quotient.decode(p)[0])).tolist(),
            moment_geometry_shift_from_seed=(np.asarray(quotient.decode(p)[0]).reshape(3,4)[:, :2]
                -seed['initial_optical_parameters'].reshape(3,4)[:, :2]).tolist(), final_exact_fit=None)
        rows.append(row)
        print('moment finished', seed['label'], fit['success'], row['moment_chi_square'], 'physical', physical, flush=True)
        if physical:
            start = np.asarray(chart.encode(p))[ORDER]
            print('exact starting', seed['label'], float(np.asarray(function(p))@np.asarray(function(p))), flush=True)
            exact = scaled_profiled_least_squares(final_residual, start, domain, scale=1., n_outer=6,
                max_iterations=20, max_inner_iterations=30)
            result = np.asarray(decode(exact['parameters']))
            r = np.asarray(function(result)); jac = np.asarray(jax.jit(jax.jacfwd(original_residual))(result))
            norms = np.linalg.norm(jac, axis=0)
            gradient = float(np.max(abs((jac/norms).T@r)))
            try:
                factor = covariance_factor(jac)
                white_gradient = float(np.max(abs((jac@factor).T@r)))
                full_rank = True
            except ValueError:
                white_gradient, full_rank = None, False
            physical_source = source_diagnostic(result)
            emittances = physical_source.get('representative_emittances_over_quantum_minimum', [])
            interior = len(emittances) == 2 and min(emittances) > 1+1e-6
            accepted = bool(exact['success'] and full_rank and gradient < 1e-6 and white_gradient < 1e-6
                and quotient.admissible(result) and interior)
            row['final_exact_fit'] = dict(accepted=accepted, optimizer_success=exact['success'], reason=exact['reason'],
                original_parameters=result.tolist(), optical_parameters=np.asarray(quotient.decode(result)[0]).tolist(),
                training_chi_square=float(r@r), training_data_chi_square=float(r@r-(result[22]/.005)**2),
                radius_prior_chi_square=float((result[22]/.005)**2), original_normalized_gradient=gradient,
                original_information_whitened_gradient=white_gradient, source=physical_source,
                history=exact['history'], initial_nuisance_history=exact['initial_nuisance_history'],
                numerical_full_rank=full_rank, all23_parameters_free=True)
            print('exact finished', seed['label'], accepted, float(r@r), exact['reason'], flush=True)
    for name, checksum in hashes.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != checksum:
            raise RuntimeError(f'A guide or final-fit input changed: {name}')
    report = dict(schema='neoarm-condenser-moment-variable-projection-initializer-v1', backend=jax.default_backend(),
        x64_enabled=True, source_hashes=hashes, rows=rows, elapsed_seconds=perf_counter()-started,
        measured_moment_values=guide.measured_moments.tolist(),
        measured_delta_method_moment_covariances=guide.measured_moment_covariance.tolist(),
        source_projection_dimension=9, source_rank_truncation=False,
        initializer_source_physicality_enforced=False, final_source_physicality_required=True,
        moment_guide_free_optical_coordinates=12, moment_guide_other_outer_coordinates=2,
        reference_control=np.asarray(ref_control).tolist(), reference_width_m=width,
        numerical_generating_values_used=False, saved_accepted_geometry_used=False, heldout_observations_used=False,
        published_rough_geometry_coincides_with_generator=bool(old['truth_consistency']['diagram_initial_optical_parameters_coincide_with_generator']),
        moment_objective_is_temporary_initializer=True, final_log_observable_likelihood_unchanged=True,
        training_data_coordinate_count=102, existing_common_radius_prior_count=1,
        posterior_added_as_prior=False, complete_microscope_recovery=False, global_uniqueness_proved=False,
        limitations=['The delta-method moment likelihood is only an initialization guide and has different finite-noise behavior from log observables.',
            'The two displaced starts test cold geometry; the rough nominal fixture was historically also the generator.',
            'An unphysical guide source is not admitted to Gaussian simulation or the final physical likelihood.',
            'All previous conditional objective/downstream, source-preparation and aperture assumptions remain.',
            'No failure of this finite initialization benchmark proves nonidentifiability.'])
    Path(output).write_text(json.dumps(finite_json(report), indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-moment-iterations', type=int, default=150)
    args = parser.parse_args()
    run(args.output, max_moment_iterations=args.max_moment_iterations)
