"""Noise-scaled uniform-solenoid-body diagnostic for the frozen internal-Q scan.

This is Gaussian phase-space ray quadrature, not a quantum image renderer.
The exact constant-body Hamiltonian and field-free drift can diagnose a
paraxial model error without inventing a universal angle cutoff. Objective
and projector transport remain conditional linear maps; physical fringes and
quantum corrections are not supplied by the hard-edge model.
"""
import argparse
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

from temgym_core.nonparaxial_solenoid import uniform_solenoid_ray, kinetic_transverse_momentum, mechanical_slopes
from temgym_core.immersed_objective import uniform_field_canonical_matrix
from neoarm_condenser_internal_beam_images import model_context, digest, verify, FOLDER, ROOT
from neoarm_condenser_stigmator_contract import drift, GROUPED
from neoarm_condenser_general_source_contract import quantum_margin
from neoarm_condenser_actual_analyser_contract import PIXEL_M
from neoarm_shared_objective_audit import provenance_snapshot


def gaussian_phase_space_rule(covariance, order):
    """Normalized positive tensor Hermite rule for the declared source Gaussian."""
    if isinstance(order, bool) or not isinstance(order, int) or order < 2:
        raise ValueError('An integer Gaussian quadrature order >=2 is required.')
    covariance = np.asarray(covariance)
    if covariance.shape != (4, 4) or not np.isfinite(covariance).all():
        raise ValueError('A finite positive definite canonical4x4 covariance is required.')
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    index = np.stack(np.meshgrid(*[np.arange(order)]*4, indexing='ij'), axis=-1).reshape(-1, 4)
    weight = np.prod(weights[index]/np.sqrt(np.pi), axis=1)
    points = np.sqrt(2)*nodes[index]@np.linalg.cholesky(covariance).T
    return jnp.asarray(points), jnp.asarray(weight)


def covariance_observable(c):
    sigma = jnp.sqrt(jnp.diag(c))
    return jnp.r_[jnp.log(sigma), jnp.arctanh(c[0, 1]/jnp.prod(sigma))]


def weighted_covariance(points, weights):
    mean = jnp.einsum('n,...ni->...i', weights, points)
    centered = points-mean[..., None, :]
    return jnp.einsum('n,...ni,...nj->...ij', weights, centered, centered)


def propagate_condensers(model, parameters, control, points, *, body_order='paraxial',
                         body_indices=(0, 1, 2), exact_drifts=False):
    """Declared grouped canonical sequence with Q unchanged; no physical fringes.

    The paraxial source covariance is interpreted as normalized canonical
    momentum covariance. At the source its ~100urad momentum/slope difference
    is negligible, but exact mechanical slopes are computed inside each body.
    """
    optical, _, z_q, q_matrix = model.decode(parameters)
    cursor, state = model.source_z_m, points
    bodies = []
    def gap(state, distance):
        if exact_drifts:
            native = uniform_solenoid_ray(state[:, jnp.asarray(GROUPED)], 0., distance)
            return native[:, jnp.asarray(GROUPED)]
        return state@drift(distance).T
    for index in range(3):
        centre, log_length, bias, slope = optical[index]
        length = jnp.exp(log_length)
        if index == model.insert_before:
            state = gap(state, z_q-cursor)@q_matrix.T
            cursor = z_q
        state = gap(state, centre-length/2-cursor)
        k = (bias+slope*control[index])/length
        native = state[:, jnp.asarray(GROUPED)]
        uv = kinetic_transverse_momentum(native, k)
        alpha = mechanical_slopes(native, k)
        bodies.append((uv, alpha))
        chosen_order = body_order if index in body_indices else 'paraxial'
        state = uniform_solenoid_ray(native, k, length, order=chosen_order)[:, jnp.asarray(GROUPED)]
        cursor = centre+length/2
    return gap(state, model.reference_z_m-cursor), bodies


def benchmark_state(model, parameters, control, points, weights, *, body_order='paraxial',
                    body_indices=(0, 1, 2), exact_drifts=False):
    reference, bodies = propagate_condensers(model, parameters, control, points,
        body_order=body_order, body_indices=body_indices, exact_drifts=exact_drifts)
    detected = jnp.einsum('aij,nj->ani', model.analyser_maps[:, :2], reference)/PIXEL_M
    covariance = weighted_covariance(detected, weights)
    observable = jax.vmap(covariance_observable)(covariance)
    body_records = []
    for uv, alpha in bodies:
        r2 = jnp.sum(uv*uv, axis=-1)
        # u,v rotate in each uniform body. Principal RMS and |uv| are
        # constant throughout the body; unlike lab x/y variances no z scan is needed.
        body_records.append(jnp.r_[jnp.sqrt(jnp.linalg.eigvalsh(weighted_covariance(uv, weights))),
            jnp.sqrt(jnp.linalg.eigvalsh(weighted_covariance(alpha, weights))),
            jnp.sum(weights*r2), jnp.sum(weights*r2*r2),
            jnp.sum(weights*(1/jnp.sqrt(1-r2)-1)), jnp.max(r2)])
    return covariance, observable, jnp.stack(body_records), detected


def run(output):
    if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
        raise RuntimeError('The ray quadrature benchmark requires GPU float64.')
    start = perf_counter()
    design_path = FOLDER/'condenser_internal_beam_acquisition_design.json'
    design = json.loads(design_path.read_text()); verify(design)
    model, _ = model_context()
    p = jnp.asarray(design['generating_hypothesis_parameters'])
    controls = jnp.asarray(design['controls'])
    source = np.asarray(model.decode(p)[1])
    roots = np.linalg.cholesky(design['expected_covariance_observable_noise'])
    sources, _ = provenance_snapshot()
    sources.update(design['source_hashes'])
    for path in (Path(__file__), design_path, ROOT/'src/temgym_core/nonparaxial_solenoid.py',
                 ROOT/'tests/test_nonparaxial_solenoid.py', ROOT/'tests/test_neoarm_condenser_nonparaxial_benchmark.py'):
        sources[str(path.relative_to(ROOT))] = digest(path)
    configurations = [dict(name='paraxial', body_order='paraxial'),
        dict(name='Cmini_exact_body', body_order='exact', body_indices=(2,)),
        dict(name='Cmini_quartic_body', body_order='quartic', body_indices=(2,)),
        dict(name='all_exact_bodies', body_order='exact'),
        dict(name='exact_drifts_only', body_order='paraxial', exact_drifts=True),
        dict(name='all_bodies_and_drifts', body_order='exact', exact_drifts=True)]
    results = {}
    for order in (6, 10, 14):
        points, weights = gaussian_phase_space_rule(source, order)
        rows = {}
        for config in configurations:
            kwargs = {k:v for k,v in config.items() if k != 'name'}
            function = jax.jit(jax.vmap(lambda u: benchmark_state(model, p, u, points, weights, **kwargs)[:3]))
            cov, obs, body = map(np.asarray, function(controls))
            if not np.isfinite(cov).all() or not np.isfinite(obs).all() or not np.isfinite(body).all():
                raise RuntimeError('A quadrature node is not in the forward exact-ray domain; do not clip or hide it.')
            rows[config['name']] = dict(covariance=cov, observable=obs, body=body)
        results[order] = rows
        print('nonparaxial quadrature completed', order, flush=True)
    baseline = results[14]['paraxial']
    analytic_maps = np.asarray(jax.jit(jax.vmap(lambda u: model.analyser_maps@model.maps(p, u)[0]))(controls))
    analytic_cov = (analytic_maps@source@analytic_maps.swapaxes(-1, -2))[:, :, :2, :2]/PIXEL_M**2
    analytic_obs = np.asarray(jax.vmap(jax.vmap(covariance_observable))(jnp.asarray(analytic_cov)))
    closure = np.linalg.solve(roots, (baseline['observable']-analytic_obs)[..., None])[..., 0]
    rows = []
    for config in configurations[1:]:
        name = config['name']; current = results[14][name]
        shifts = np.linalg.solve(roots, (current['observable']-baseline['observable'])[..., None])[..., 0]
        refinements = []
        for lo, hi in ((6, 10), (10, 14)):
            changes = np.linalg.solve(roots, (results[hi][name]['observable']-results[lo][name]['observable'])[..., None])[..., 0]
            refinements.append(dict(orders=[lo,hi], maximum_frame_observable_shift_sigma=float(np.linalg.norm(changes, axis=-1).max()),
                maximum_marginal_shift_sigma=float(abs(changes).max())))
        rows.append(dict(name=name, detector_optical_covariance_pixels=current['covariance'].tolist(),
            observable_change=(current['observable']-baseline['observable']).tolist(),
            whitened_observable_change=shifts.tolist(),
            frame_observable_shift_sigma=np.linalg.norm(shifts, axis=-1).tolist(),
            maximum_frame_observable_shift_sigma=float(np.linalg.norm(shifts, axis=-1).max()),
            minimum_frame_observable_shift_sigma=float(np.linalg.norm(shifts, axis=-1).min()),
            detector_optical_principal_rms_pixels=np.sqrt(np.linalg.eigvalsh(current['covariance'])).tolist(),
            quadrature_refinements=refinements,
            quadrature_noise_gate_passed=bool(max(x['maximum_frame_observable_shift_sigma'] for x in refinements) < .02)))
    # Purely symplectic source squeeze: no illicit reduction of source emittance.
    # Existing controls/noise are retained only as a sensitivity comparison;
    # changed illumination would require a new pixel Fisher/capture design.
    squeeze_rows = []
    for squeeze in (2., 4., 8.):
        scale = np.diag([squeeze, squeeze, 1/squeeze, 1/squeeze])
        squeezed = scale@source@scale
        points, weights = gaussian_phase_space_rule(squeezed, 10)
        para = jax.jit(jax.vmap(lambda u: benchmark_state(model,p,u,points,weights)[:3]))(controls)
        exact = jax.jit(jax.vmap(lambda u: benchmark_state(model,p,u,points,weights,
            body_order='exact',body_indices=(2,))[:3]))(controls)
        shift = np.linalg.solve(roots, (np.asarray(exact[1])-np.asarray(para[1]))[...,None])[...,0]
        squeeze_rows.append(dict(symplectic_source_position_multiplier=squeeze,
            source_covariance=squeezed.tolist(), quantum_margin=quantum_margin(squeezed),
            maximum_Cmini_mechanical_principal_RMS_rad=float(np.asarray(para[2])[:,2,2:4].max()),
            optical_principal_rms_pixels=np.sqrt(np.linalg.eigvalsh(np.asarray(para[0]))).tolist(),
            maximum_Cmini_body_shift_in_ORIGINAL_design_noise_sigma=float(np.linalg.norm(shift,axis=-1).max()),
            source_emittance_preserved=True, redesigned_noise_or_capture_information_recomputed=False,
            new_hardware_preparation_proved=False))
    body = baseline['body']
    report = dict(schema='neoarm-uniform-body-nonparaxial-diagnostic-v1',source_hashes=sources,
        backend=jax.default_backend(),x64_enabled=True,elapsed_seconds=perf_counter()-start,
        design_path=str(design_path.relative_to(ROOT)),design_sha256=digest(design_path),
        controls=np.asarray(controls).tolist(),parameters=np.asarray(p).tolist(),source_covariance=source.tolist(),
        quadrature_orders=[6,10,14],phase_space_node_counts=[6**4,10**4,14**4],
        expected_noise_covariance=design['expected_covariance_observable_noise'],
        paraxial_matrix_closure_maximum_observable_sigma=float(np.linalg.norm(closure,axis=-1).max()),
        paraxial_detector_optical_covariance_pixels=baseline['covariance'].tolist(),
        body_diagnostic_fields=['kinetic_momentum_principal_rms_1','kinetic_momentum_principal_rms_2',
            'exact_mechanical_slope_principal_rms_1','exact_mechanical_slope_principal_rms_2',
            'mean_transverse_momentum_squared','mean_transverse_momentum_fourth',
            'mean_axial_path_factor_minus_one','maximum_quadrature_transverse_momentum_squared'],
        paraxial_body_entrance_diagnostics=body.tolist(),
        maximum_mechanical_principal_rms_by_body_rad=body[:,:,2:4].max(axis=(0,2)).tolist(),
        mean_H4_over_H2_by_control_and_body=(body[:,:,5]/(4*body[:,:,4])).tolist(),
        comparisons=rows,source_squeeze_sensitivity=squeeze_rows,
        numerical_generating_values_used_for_FORWARD_test=True,optical_recovery_performed=False,
        no_universal_angle_threshold=True,quantum_wave_validation_performed=False,
        nonparaxial_objective_and_projector_transport_included=False,
        physical_hardedge_fringe_model_validated=False,actual_hardware_bore_or_field_assumed=False,
        references=[dict(url='https://doi.org/10.1007/978-3-319-18317-6_5',
            support='Wiedemann Eq5.40 fixed-energy canonical square-root Hamiltonian; Eq5.41 is its paraxial truncation.'),
            dict(url='https://impactx.readthedocs.io/en/26.08/usage/examples/solenoid_softedge/README.html',
            support='Official ImpactX linear solenoid Hamiltonian/map benchmark; finite soft-edge fields require their specified profile.')],
        conclusions=[
            'The uniform-body exact map is controlled classical dynamics, not a generic exact electron-microscope simulator.',
            'A noise-significant body correction is evidence against assuming paraxial accuracy for this synthetic scan, not a calibrated spherical-aberration coefficient.',
            'Other sections and realistic fringes may change or compensate detector shifts; body-only substitutions are sensitivity tests, not rigorous lower bounds on total error.',
            'The H4 comparison separates leading kinematic curvature from the exact uniform-body resummation.',
            'Source squeezes preserve symplectic emittances but are hypothetical preparations; narrower output may fail detector sampling and needs renewed information design.',
            'The Gaussian phase-space density has unbounded support. Tested quadrature nodes are forward; no global compact-support claim is made.',
            'No acquisition noise, detector images, physical optical fit or hardware commands are generated.'])
    verify(report)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('nonparaxial report',[(r['name'],r['maximum_frame_observable_shift_sigma'],r['quadrature_noise_gate_passed']) for r in rows],flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    run(parser.parse_args().output)
