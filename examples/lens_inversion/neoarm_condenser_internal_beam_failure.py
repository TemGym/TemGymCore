"""Record the frozen internal-Q acquisition's canonical-map prerequisite failure.

This diagnostic does not relax the Gaussian renderer's symplectic guard,
modify a transfer map, render images, or draw acquisition noise.  The matrix
calculations are repeated on GPU; NumPy only inspects the resulting small
arrays using the identical frozen guard expression.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from neoarm_condenser_internal_beam_images import (
    GROUPED, ROOT, digest, model_context, roles, verify,
)
from neoarm_condenser_gaussian_validation import (
    OMEGA, covariance_to_mixed_source, validate_matrix,
)
from neoarm_condenser_stigmator_contract import drift
from temgym_core.constants import energy2wavelength
from temgym_core.immersed_objective import canonical_to_mechanical, uniform_field_canonical_matrix


def guard_diagnostic(matrix):
    """Inspect, but never alter or bypass, the frozen canonical-map guard."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError('A finite 4x4 matrix is required for this diagnostic.')
    residual = matrix.T@OMEGA@matrix-OMEGA
    bound = 1e-9+32*np.finfo(float).eps*(abs(matrix).T@abs(OMEGA)@abs(matrix))
    try:
        validate_matrix(matrix)
        accepted, failure = True, None
    except ValueError as exc:
        accepted, failure = False, str(exc)
    if accepted != bool(np.all(abs(residual) <= bound)):
        raise RuntimeError('Diagnostic no longer matches the frozen renderer guard.')
    return dict(accepted=accepted, exception=failure,
        maximum_absolute_symplectic_residual=float(abs(residual).max()),
        maximum_guard_ratio=float((abs(residual)/bound).max()),
        residual=residual.tolist(), absolute_guard_bound=bound.tolist())


def balanced_diagnostic(matrix, source_covariance, *, left, right):
    """Physical symplectic scaling and conservative last-product error bounds.

    Each plane's scale has conjugate products epsilon=lambda/(4*pi), so this
    change of units preserves the standard canonical form. The forward bound
    covers only the final4x4 product, not errors inherited by either factor.
    The backward bound is a lower bound, not evidence of a small nearby error.
    """
    epsilon = float(energy2wavelength(200e3))/(4*np.pi)
    output = matrix@source_covariance@matrix.T
    def scales(covariance):
        sx, sy = np.sqrt(np.diag(covariance)[[0, 2]])
        return np.array([sx, epsilon/sx, sy, epsilon/sy])
    source_scale, detector_scale = scales(source_covariance), scales(output)
    balanced = matrix*source_scale[None]/detector_scale[:, None]
    residual = balanced.T@OMEGA@balanced-OMEGA
    norm_m = float(np.linalg.norm(balanced, 2)); norm_e = float(np.linalg.norm(residual, 2))
    lower = norm_e/(np.sqrt(norm_m**2+norm_e)+norm_m)
    unit_roundoff = np.finfo(float).eps/2
    gamma4 = 4*unit_roundoff/(1-4*unit_roundoff)
    bound = gamma4*(abs(left)@abs(right))
    return dict(source_scale=source_scale.tolist(), detector_scale=detector_scale.tolist(),
        common_conjugate_scale_m=epsilon,
        maximum_absolute_symplectic_residual=float(abs(residual).max()),
        spectral_symplectic_residual=norm_e, balanced_matrix_spectral_norm=norm_m,
        relative_spectral_distance_to_symplectic_lower_bound=lower/norm_m,
        last_product_balanced_spectral_forward_error_bound=float(np.linalg.norm(
            bound*source_scale[None]/detector_scale[:, None], 2)),
        last_product_per_coordinate_RMS_forward_error_bound=(bound@np.sqrt(np.diag(source_covariance))).tolist(),
        inherited_factor_errors_not_included_in_forward_bound=True,
        source_RMS_xy_m=np.sqrt(np.diag(source_covariance)[[0, 2]]).tolist(),
        source_RMS_canonical_momenta=np.sqrt(np.diag(source_covariance)[[1, 3]]).tolist(),
        detector_RMS_xy_m=np.sqrt(np.diag(output)[[0, 2]]).tolist(),
        detector_RMS_canonical_momenta=np.sqrt(np.diag(output)[[1, 3]]).tolist(),
        detector_RMS_mechanical_angles_rad=np.sqrt(np.diag(output)[[1, 3]]).tolist(),
        source_and_detector_are_field_free=True)


def condenser_stage_maps(model, p, control):
    """Sample9 planes/body for a covariance-angle diagnostic, not wave rendering."""
    optical, _, z_q, q_matrix = model.decode(p)
    matrix, cursor = jnp.eye(4), model.source_z_m
    canonical, mechanical = [], []
    for index in range(3):
        centre, log_length, bias, slope = optical[index]
        length = jnp.exp(log_length)
        if index == model.insert_before:
            matrix = q_matrix@drift(z_q-cursor)@matrix
            cursor = z_q
        matrix = drift(centre-length/2-cursor)@matrix
        k = (bias+slope*control[index])/length
        before = matrix[jnp.ix_(GROUPED, GROUPED)]
        sections = uniform_field_canonical_matrix(k, jnp.linspace(0., length, 9))@before
        canonical.append(sections)
        mechanical.append(canonical_to_mechanical(k)@sections)
        matrix = sections[-1][jnp.ix_(GROUPED, GROUPED)]
        cursor = centre+length/2
    return jnp.stack(canonical), jnp.stack(mechanical)


def audit(design_path, output):
    if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
        raise RuntimeError('Repeat the frozen matrix construction on GPU float64.')
    start = perf_counter()
    design_path = Path(design_path).resolve()
    design = json.loads(design_path.read_text()); verify(design)
    controls = jnp.asarray(design['controls'])
    roles(design['training_mask'], design['heldout_mask'], len(controls))
    model, _ = model_context()
    p = jnp.asarray(design['generating_hypothesis_parameters'])
    grouped = np.asarray(jax.jit(jax.vmap(
        lambda u: model.analyser_maps@model.maps(p, u)[0]))(controls))
    matrices = grouped[:, :, GROUPED[:, None], GROUPED[None, :]].reshape(-1, 4, 4)
    prefixes = np.asarray(jax.jit(jax.vmap(lambda u: model.maps(p, u)[0]))(controls))
    prefixes = prefixes[:, GROUPED[:, None], GROUPED[None, :]]
    analysers = np.asarray(model.analyser_maps)[:, GROUPED[:, None], GROUPED[None, :]]
    source = np.asarray(model.decode(p)[1])[np.ix_(GROUPED, GROUPED)]
    covariance = matrices@source@matrices.swapaxes(-1, -2)
    spatial = covariance[:, [0, 2]][:, :, [0, 2]]
    widths = np.sqrt(np.linalg.eigvalsh(spatial))
    pure, _ = covariance_to_mixed_source(source)
    pure_covariance = matrices@pure.pure_covariance@matrices.swapaxes(-1, -2)
    pure_spatial = pure_covariance[:, [0, 2]][:, :, [0, 2]]
    mixture_ratio = np.array([np.linalg.eigvals(np.linalg.solve(a, b)).real.max()
                             for a, b in zip(pure_spatial, spatial)])
    selected = sorted(set([int(np.argmin(widths.min(axis=1))),
        int(np.argmax(widths.max(axis=1))), int(np.argmax(mixture_ratio))]))
    scales = np.sqrt(np.diag(pure.pure_covariance))
    eigenvalues = np.linalg.eigvalsh(pure.centre_covariance/np.outer(scales, scales))
    rank = int(np.sum(eigenvalues > max(1., float(eigenvalues.max()))*1e-11))
    records = [dict(frame=i, state=i//2, analyser=i%2,
                   matrix=m.tolist(), **guard_diagnostic(m),
                   physically_balanced=balanced_diagnostic(m, source,
                       left=analysers[i%2], right=prefixes[i//2]))
               for i, m in enumerate(matrices)]
    stages = jax.jit(jax.vmap(lambda u: condenser_stage_maps(model, p, u)))(controls)
    stage_rows = []
    for label, maps in zip(('canonical', 'mechanical'), stages):
        maps = np.asarray(maps)
        covariances = maps@source@maps.swapaxes(-1, -2)
        angle = covariances[..., [1, 3]][..., [1, 3], :]
        angular_rms = np.sqrt(np.linalg.eigvalsh(angle))
        stage_rows.append(dict(coordinates=label,
            maximum_principal_angular_RMS_by_lens_rad=angular_rms.max(axis=(0, 2, 3)).tolist(),
            maximum_principal_angular_RMS_rad=float(angular_rms.max()),
            six_RMS_envelope_rad=float(6*angular_rms.max()),
            body_sample_count=9, continuous_z_maximum_not_certified=True))
    sources = dict(design['source_hashes'])
    for path in (design_path, Path(__file__).resolve(),
                 ROOT/'tests/test_neoarm_condenser_internal_beam_failure.py'):
        sources[str(path.relative_to(ROOT))] = digest(path)
    verify(dict(source_hashes=sources))
    report = dict(schema='neoarm-internal-Q-acquisition-prerequisite-failure-v1',
        source_hashes=sources, backend=jax.default_backend(),
        device_kind=jax.devices()[0].device_kind, x64_enabled=True,
        elapsed_diagnostic_seconds=perf_counter()-start,
        design_path=str(design_path.relative_to(ROOT)), design_sha256=digest(design_path),
        design_gates_accepted=design['design_gates_accepted'],
        planned_frame_count=design['frame_count'], selected_validation_frame_indices=selected,
        rejected_frame_indices=[r['frame'] for r in records if not r['accepted']],
        rejected_frame_count=sum(not r['accepted'] for r in records),
        maximum_absolute_symplectic_residual=max(r['maximum_absolute_symplectic_residual'] for r in records),
        maximum_guard_ratio=max(r['maximum_guard_ratio'] for r in records),
        records=records, conditional_analyser_guard_diagnostics=[guard_diagnostic(a) for a in analysers],
        condenser_body_angular_diagnostics=stage_rows,
        objective_and_projector_interior_angles_not_sampled=True,
        maximum_detector_mechanical_angular_RMS_rad=max(max(
            row['physically_balanced']['detector_RMS_mechanical_angles_rad']) for row in records),
        angular_diagnostic_scope='Zero-mean Gaussian covariance has unbounded support; six RMS is an envelope diagnostic, not a maximum ray angle. Detector is field-free so normalized canonical momenta equal mechanical slopes.',
        planned_hermite_order_sequences=[[5, 9, 13], [13, 21, 31]],
        source_centre_mixture_rank=rank,
        planned_incoherent_mode_counts={str(n): n**rank for n in (5, 9, 13, 21, 31)},
        rendered_frame_count=0, completed_modal_orders=[], completed_pixel_refinements=[],
        acquisition_noise_drawn=False, public_inference_arrays_created=False,
        accepted_image_likelihood=False, optical_recovery_performed=False,
        competing_Q_order_ruled_out=False, hardware_admission=False,
        failure_stage='canonical-map validation before first Gaussian-mode propagation',
        outcome='stopped_at_prerequisite_guard_no_images_generated',
        cause_assessment='Accumulated roundoff in strongly squeezed map composition is a hypothesis, not a demonstrated cause. The conditional analyser maps already violate this guard; matching the native full-column map does not certify symplectic accuracy.',
        next_prerequisite='Validate canonical component-wise or numerically balanced Gaussian propagation against an independent detector reference, preserving all beam-dependent phase. Do not relax the guard or silently project the matrices onto a symplectic family.',
        scope='The frozen finite-pixel covariance design remains a conditional information forecast, not an accepted Gaussian-beam image experiment.')
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: report[key] for key in ('outcome', 'rejected_frame_count',
        'maximum_guard_ratio', 'planned_incoherent_mode_counts', 'elapsed_diagnostic_seconds')}))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(); jax.config.update('jax_enable_x64', True)
    audit(args.design, args.output)
