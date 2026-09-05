"""Refocused beam-only design with unknown internal Q and ten source moments.

The conditional optical maps are not refitted here.  Computational hypothesis
and control bounds are fixed before design; all25 inference coordinates enter
every information calculation.  No noisy images or recovery are generated.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from neoarm_condenser_actual_analyser_contract import ActualAnalyserStigmator, PIXEL_M
from neoarm_condenser_general_source_contract import initial_parameters, source_coordinates, quantum_margin
from neoarm_condenser_extension_design import load_context, approximate_covariance_noise, greedy_design, FOLDER
from neoarm_condenser_stigmator_contract import OPTICAL_SCALES, ROOT
from temgym_core.gaussian_image_likelihood import (
    make_gaussian_detector_operator, encode_gaussian_image_parameters, gaussian_image_information_factor,
)
from temgym_core.neoarm_calibration import DetectorGeometry
from temgym_core.jax_least_squares import least_squares

# Frozen computational ranges, copied from the preceding condenser design.
LOWER = np.array([-.25, -.15, .3])
UPPER = np.array([2., 2., 3.5])
LOCAL_HALF_WIDTH = .02
DEFOCUS_OFFSETS = np.array([-.0004, -.0002, -.0001, -.00005, 0., .00005, .0001, .0002, .0004])
RMS_GATE = .4
CAPTURE_GATE = .999
DOSE = 2e6
MTF = .65


def covariance_at(model, parameters, controls):
    """Arbitrary frozen control list; grouped canonical source coordinates."""
    source = model.decode(parameters)[1]
    maps = jax.vmap(lambda u: model.analyser_maps@model.maps(parameters, u)[0])(controls)
    return (maps@source@maps.swapaxes(-1, -2))[:, :, :2, :2]/PIXEL_M**2


def observables_at(model, parameters, controls):
    c = covariance_at(model, parameters, controls)
    sigma = jnp.sqrt(jnp.diagonal(c, axis1=-2, axis2=-1))
    correlation = c[:, :, 0, 1]/jnp.prod(sigma, axis=-1)
    return jnp.concatenate((jnp.log(sigma), jnp.arctanh(correlation)[:, :, None]), axis=-1)


def hypotheses(conditional):
    models = [ActualAnalyserStigmator(name, conditional) for name in ('after_cl1', 'after_cl3')]
    parameters = initial_parameters()
    source = np.asarray(models[0].decode(parameters)[1])
    records = []
    for index, model in enumerate(models):
        records.append(dict(location=model.location, model_index=index, parameters=parameters.tolist(),
            nominal=True, source_scale=1., strength_scale=1., Q_position_change_m=0.))
        for scale, strength, shift in itertools.product((.9, 1.1), (.9, 1.1), (-.002, .002)):
            p = parameters.copy(); p[12:22] = source_coordinates(source*scale**2)
            p[22] += shift/.01; p[23:] *= strength
            if quantum_margin(model.decode(p)[1]) <= 0:
                raise RuntimeError('A frozen hypothesis is not quantum physical.')
            records.append(dict(location=model.location, model_index=index, parameters=p.tolist(), nominal=False,
                source_scale=scale, strength_scale=strength, Q_position_change_m=shift))
    return models, records


def capture_lower_bound(covariance):
    """PSF-safe union bound: the normalized discrete PSF has a3-pixel halo."""
    sigma = jnp.sqrt(jnp.diagonal(covariance, axis1=-2, axis2=-1))
    return jnp.maximum(0., 1.-jnp.sum(jax.scipy.special.erfc(125/(jnp.sqrt(2)*sigma)), axis=-1))


def largest_component(mask):
    """Six-neighbour connected grid; no claim about unsampled interiors."""
    mask = np.asarray(mask, bool)
    unseen = set(map(tuple, np.argwhere(mask))); components = []
    while unseen:
        start = min(unseen); unseen.remove(start); queue = [start]; component = []
        while queue:
            point = queue.pop(); component.append(point)
            for axis in range(mask.ndim):
                for sign in (-1, 1):
                    adjacent = list(point); adjacent[axis] += sign; adjacent = tuple(adjacent)
                    if adjacent in unseen:
                        unseen.remove(adjacent); queue.append(adjacent)
        components.append(component)
    components.sort(key=len, reverse=True)
    accepted = np.zeros_like(mask)
    if components:
        accepted[tuple(np.asarray(components[0]).T)] = True
    return accepted, [len(c) for c in components]


def information(jacobian, *, tolerance=1e-10):
    """Full noise-weighted marginal information; no truncated covariance."""
    j = np.asarray(jacobian).reshape(-1, 25)
    norm = np.linalg.norm(j, axis=0); scaled = j/np.where(norm > 0, norm, 1.)
    _, s, v = np.linalg.svd(scaled, full_matrices=False)
    rank = int(np.sum(s > s[0]*tolerance))
    nuisance = j[:, 12:]/np.where(norm[12:] > 0, norm[12:], 1.)
    u, sn, _ = np.linalg.svd(nuisance, full_matrices=False)
    nrank = int(np.sum(sn > sn[0]*tolerance))
    optical = scaled[:, :12]-u[:, :nrank]@(u[:, :nrank].T@scaled[:, :12])
    ps = np.linalg.svd(optical, compute_uv=False)
    result = dict(rank=rank, nuisance_rank=nrank,
        profiled_optical_rank=int(np.sum(ps > max(ps[0], 1.)*tolerance)),
        column_normalized_singular_values=s.tolist(), profiled_optical_singular_values=ps.tolist(),
        tolerance=tolerance, parameter_covariance=None, optical_covariance=None,
        full_untruncated_covariance_available=False)
    if rank == 25:
        factor = (v.T/s)/norm[:, None]
        physical = OPTICAL_SCALES[:, None]*factor[:12]
        sd = np.linalg.norm(physical, axis=1).reshape(3, 4)
        result.update(parameter_covariance=(factor@factor.T).tolist(), optical_covariance=(physical@physical.T).tolist(),
            centre_standard_errors_mm=(1e3*sd[:, 0]).tolist(), length_standard_errors_percent=(100*sd[:, 1]).tolist(),
            q_bias_standard_errors_rad=sd[:, 2].tolist(), q_slope_standard_errors_rad_per_control=sd[:, 3].tolist(),
            Q_position_standard_error_m=float(.01*np.linalg.norm(factor[22])),
            Q_strength_standard_errors_inverse_m=(30*np.linalg.norm(factor[23:], axis=1)).tolist(),
            full_untruncated_covariance_available=True)
        return result, factor
    return result, None


def common_focus(models, parameters, *, coarse_count=17):
    root = jnp.linalg.cholesky(models[0].decode(parameters)[1])
    def residual(u):
        return jnp.stack([(m.analyser_maps@m.maps(parameters, u)[0]@root)[:, :2]/PIXEL_M for m in models]).reshape(-1)
    function = jax.jit(residual); derivative = jax.jit(jax.jacfwd(residual))
    grid = np.array(list(itertools.product(np.linspace(LOWER[0], UPPER[0], coarse_count),
        np.linspace(LOWER[1], UPPER[1], coarse_count), np.linspace(LOWER[2], UPPER[2], 129))))
    score = np.asarray(jax.jit(jax.vmap(lambda u: jnp.sum(residual(u)**2)))(jnp.asarray(grid)))
    rows = []
    for index in np.argsort(score)[:3]:
        seed = grid[index]; scale = 1/np.linalg.norm(np.asarray(derivative(seed)), axis=0)
        fit = least_squares(function, seed, bounds=(LOWER, UPPER), x_scale=scale,
            max_nfev=220, max_iter=440, gtol=1e-7)
        rows.append(dict(initial=seed.tolist(), controls=np.asarray(fit.x).tolist(),
            optimizer_success=bool(fit.success), objective_trace_pixels2=float(2*fit.cost), nfev=fit.nfev))
    best = min(rows, key=lambda row: row['objective_trace_pixels2'])
    return np.asarray(best['controls']), rows


def refocused_grid(models, p, focus, *, grid_count=17):
    axis_a = focus[0]+np.linspace(-LOCAL_HALF_WIDTH, LOCAL_HALF_WIDTH, grid_count)
    axis_b = focus[1]+np.linspace(-LOCAL_HALF_WIDTH, LOCAL_HALF_WIDTH, grid_count)
    ab = np.array(list(itertools.product(axis_a, axis_b)))
    def objective(c):
        controls = jnp.c_[jnp.asarray(ab), c]
        return sum(jnp.trace(covariance_at(m, p, controls), axis1=-2, axis2=-1).sum(axis=-1) for m in models)
    function = jax.jit(objective)
    lo, hi = jnp.full(len(ab), focus[2]-.01), jnp.full(len(ab), focus[2]+.01)
    ratio = (np.sqrt(5)-1)/2
    for _ in range(55):
        x, y = hi-ratio*(hi-lo), lo+ratio*(hi-lo)
        left = function(x) < function(y)
        lo, hi = jnp.where(left, lo, x), jnp.where(left, y, hi)
    focused = np.asarray((lo+hi)/2)
    controls = np.array([[a, b, c+d] for (a, b), c in zip(ab, focused) for d in DEFOCUS_OFFSETS])
    if np.any(controls < LOWER) or np.any(controls > UPPER):
        raise RuntimeError('The refocused patch left the frozen broad control range.')
    return controls, focused, [axis_a.tolist(), axis_b.tolist(), DEFOCUS_OFFSETS.tolist()]


def audit(output, *, training_count=12, heldout_count=4, grid_count=17):
    if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
        raise RuntimeError('Use GPU float64 for covariance and finite-pixel forward information.')
    started = perf_counter(); old, _, repair, aperture, conditional = load_context()
    models, hypotheses_frozen = hypotheses(conditional); p = jnp.asarray(initial_parameters())
    focus, focus_rows = common_focus(models, p)
    controls, autofocus, axes = refocused_grid(models, p, focus, grid_count=grid_count)
    covariance_fun = [jax.jit(lambda pp, u, m=m: covariance_at(m, pp, u)) for m in models]
    covariance = np.stack([np.asarray(covariance_fun[h['model_index']](jnp.asarray(h['parameters']), jnp.asarray(controls))) for h in hypotheses_frozen])
    widths = np.sqrt(np.linalg.eigvalsh(covariance)); capture = np.asarray(capture_lower_bound(covariance))
    admitted = (widths[:, :, :, 0].min(axis=(0, 2)) >= RMS_GATE)&(capture.min(axis=(0, 2)) >= CAPTURE_GATE)
    component, component_sizes = largest_component(admitted.reshape(grid_count, grid_count, len(DEFOCUS_OFFSETS)))
    indices = np.flatnonzero(component.reshape(-1)); candidate_controls = controls[indices]
    print('Common focus', focus, 'admitted', admitted.sum(), 'largest connected', len(indices), flush=True)
    candidate_info = []; selected = None; exact_rows = []
    if len(indices) >= training_count+heldout_count:
        derivatives, factors, approximate_covariances = [], [], []
        for model in models:
            c = np.asarray(covariance_fun[models.index(model)](p, jnp.asarray(candidate_controls)))
            noise = np.asarray(jax.jit(jax.vmap(jax.vmap(lambda cc: approximate_covariance_noise(cc, DOSE))))(jnp.asarray(c)))
            derivative = np.asarray(jax.jit(jax.jacfwd(lambda pp: observables_at(model, pp, jnp.asarray(candidate_controls))))(p))
            whitened = np.linalg.solve(np.linalg.cholesky(noise), derivative).reshape(len(indices), 6, 25)
            info, factor = information(whitened)
            candidate_info.append(info); derivatives.append(whitened); factors.append(factor); approximate_covariances.append(noise)
            print('Candidate rank', model.location, info['rank'], info.get('length_standard_errors_percent'), flush=True)
        if all(factor is not None for factor in factors):
            normalized = np.stack([j@factor*1e3 for j, factor in zip(derivatives, factors)])
            chosen, gains, _ = greedy_design(normalized, training_count)
            chosen = np.asarray(chosen)
            unused = np.setdiff1d(np.arange(len(indices)), chosen)
            # Heldouts use only geometry, never simulated observations or noise.
            targets = [(candidate_controls[chosen[i]]+candidate_controls[chosen[(i+1)%training_count]])/2 for i in range(heldout_count)]
            hold = []
            scale = np.array([2*LOCAL_HALF_WIDTH, 2*LOCAL_HALF_WIDTH, .001])
            for target in targets:
                cost = np.sum(((candidate_controls[unused]-target)/scale)**2, axis=1)
                index = int(unused[np.argmin(cost)]); hold.append(index); unused = unused[unused != index]
            selection = np.r_[chosen, hold]; selected_controls = candidate_controls[selection]
            detector = DetectorGeometry(mtf_sigma_px=MTF)
            operator = jax.jit(make_gaussian_detector_operator(detector, 32))
            pixel_information = {order: jax.jit(lambda x, order=order: gaussian_image_information_factor(x, detector, quadrature_order=order))
                                 for order in (8, 16, 32)}
            exact_covariances = []
            for h, model in enumerate(models):
                c = np.asarray(covariance_fun[h](p, jnp.asarray(selected_controls)))
                noise_by_order = {order: [] for order in pixel_information}; actual_capture = []; ranks = []
                for cc in c.reshape(-1, 2, 2):
                    encoded = encode_gaussian_image_parameters(np.zeros(2), cc*PIXEL_M**2, DOSE, detector)
                    for order, pixel_function in pixel_information.items():
                        image_j = np.asarray(pixel_function(encoded)); _, singular, right = np.linalg.svd(image_j, full_matrices=False)
                        if order == 32:
                            ranks.append(int(np.sum(singular > singular[0]*1e-10)))
                        cov = (right.T/singular)@(right.T/singular).T
                        noise_by_order[order].append(cov[2:5, 2:5])
                    actual_capture.append(float(np.asarray(operator(np.zeros(2), cc*PIXEL_M**2, DOSE)).sum()/DOSE))
                noise_by_order = {order: np.asarray(noise).reshape(-1, 2, 3, 3) for order, noise in noise_by_order.items()}
                noise = noise_by_order[32]; exact_covariances.append(noise)
                derivative = np.asarray(jax.jit(jax.jacfwd(lambda pp: observables_at(model, pp, jnp.asarray(selected_controls))))(p))
                whitened = np.linalg.solve(np.linalg.cholesky(noise[:training_count]), derivative[:training_count]).reshape(-1, 25)
                info, _ = information(whitened)
                approximate, _ = information(np.asarray(derivatives[h])[chosen])
                refinement = []
                for before, after in ((8, 16), (16, 32)):
                    relative = np.linalg.norm(noise_by_order[before]-noise_by_order[after], axis=(-2, -1))/np.linalg.norm(noise_by_order[after], axis=(-2, -1))
                    refinement.append(dict(quadrature_orders=[before, after],
                        maximum_observable_covariance_relative_change=float(relative.max())))
                exact_rows.append(dict(location=model.location, information=info, approximate_information_same_states=approximate,
                    minimum_image_information_rank=min(ranks), minimum_finite_pixel_capture=min(actual_capture),
                    pixel_information_refinements=refinement,
                    one_percent_length_precision=bool(info['rank'] == 25 and max(info['length_standard_errors_percent']) <= 1.),
                    covariance_observable_uncertainty=noise.tolist(), incidence_dose_per_frame=DOSE))
                print('Exact selected', model.location, info['rank'], info.get('length_standard_errors_percent'), flush=True)
            selected = dict(controls=selected_controls.tolist(), grid_indices=indices[selection].tolist(),
                training_mask=([True]*training_count+[False]*heldout_count), heldout_mask=([False]*training_count+[True]*heldout_count),
                proposed_optical_setting_count=len(selection), proposed_detector_frame_count=2*len(selection),
                training_observable_coordinate_count=6*training_count, heldout_observable_coordinate_count=6*heldout_count,
                greedy_gains=gains, candidate_index=selection.tolist(),
                minimum_corner_rms_pixels=float(widths[:, indices[selection], :, 0].min()),
                maximum_corner_rms_pixels=float(widths[:, indices[selection], :, 1].max()),
                minimum_corner_capture_lower_bound=float(capture[:, indices[selection]].min()),
                finite_pixel_information=exact_rows,
                local_full_rank=all(row['information']['rank'] == 25 for row in exact_rows),
                one_percent_length_precision=all(row['one_percent_length_precision'] for row in exact_rows),
                eligible_for_full_optical_recovery_images=all(row['one_percent_length_precision'] for row in exact_rows))
    previous = FOLDER/'condenser_actual_analyser_contract.json'
    sources = dict(json.loads(previous.read_text())['source_hashes'])
    for path in (Path(__file__), ROOT/'tests/test_neoarm_condenser_internal_beam_design.py', previous,
                 ROOT/'src/temgym_core/gaussian_image_likelihood.py', ROOT/'src/temgym_core/jax_least_squares.py'):
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path, checksum in sources.items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != checksum:
            raise RuntimeError(f'A dependency changed: {path}')
    report = dict(schema='neoarm-internal-Q-refocused-beam-design-v1', source_hashes=sources,
        backend=jax.default_backend(), x64_enabled=True, elapsed_seconds=perf_counter()-started,
        hypotheses_frozen_before_design=hypotheses_frozen,
        hypothesis_bounds='Both Q gaps; nominal source/Q plus16 corners of source RMS±10%, both Q strengths together±10%, and Q position±2mm. These finite corners do not cover a full10D source-shape uncertainty box.',
        control_bounds=[LOWER.tolist(), UPPER.tolist()], conditional_analyser_controls=np.asarray(conditional.analyser_controls).tolist(),
        common_focus=focus.tolist(), focus_design_optimization=focus_rows,
        local_axes=axes, refocused_Cmini=autofocus.tolist(), grid_controls=controls.tolist(),
        rms_gate_pixels=RMS_GATE, capture_gate=CAPTURE_GATE, admitted_grid_mask=admitted.tolist(),
        grid_component_sizes=component_sizes, candidate_grid_indices=indices.tolist(),
        all_candidate_approximate_information=candidate_info, selected=selected,
        outcome=('no_common_admitted_information_design' if selected is None else
            ('conditional_precision_candidate' if selected['eligible_for_full_optical_recovery_images'] else 'captured_but_insufficient_optical_information')),
        next_step=('Do not generate recovery images from the failed common patch. Assess preparation-specific empirical autofocus routes, then lower-magnification analyser pairs if needed.'),
        source_Q_or_optics_fixed_in_information=False, all_twenty_five_coordinates_free=True,
        covariance_noise_method='Candidate ranking: PSF/pixel-broadened Gaussian moment approximation. Selected states: exact finite-pixel6-coordinate image information; full marginalized3x3 covariance retains centroid/dose nuisance.',
        ridge_scope='The1e−6 normalized ridge is used only for greedy state selection; no ridge/prior or truncated direction is included in reported information.',
        physical_optical_recovery_performed=False, noisy_images_generated=False,
        aperture_collection_used=False, upstream_calibrated_kicks_used=False, current_readback_used=False,
        hardware_control_range_or_granularity_confirmed=False, global_branch_identification_proved=False,
        limitations=['This is an ideal conditional beam-only design candidate, not an acquired or recovered instrument.',
            'Objective/downstream maps, camera/PSF, unchanged Gaussian source preparation, isolated hard-edge lenses and thin internalQ remain conditional.',
            'The nominal hypothesis source has10 free covariance coordinates in information; Q position and both strengths are also nuisance parameters.',
            'Corner admission and grid connectivity do not certify continuous uncertainty boxes or safe real-instrument control paths.',
            'Fine C-mini increments are synthetic normalized control changes; actual command quantization/reproducibility must be measured.',
            'The displayed12-state common candidate fails the optical precision test. Camera capture alone does not justify a recovery-image acquisition.',
            'Heldout controls are chosen before images/noise and excluded from all selected training information.',
            'Local marginal uncertainties and rank do not exclude nonlinear/profile families or polarity/excitation branches.',
            'Finite-pixel reference information is not an independent Gaussian-mode propagation or pixel-quadrature refinement gate; those are required before any recovery experiment.'])
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--training-count', type=int, default=12)
    parser.add_argument('--heldout-count', type=int, default=4)
    parser.add_argument('--grid-count', type=int, default=17)
    args = parser.parse_args(); jax.config.update('jax_enable_x64', True)
    result = audit(args.output, training_count=args.training_count, heldout_count=args.heldout_count, grid_count=args.grid_count)
    print('Selected candidate', result['selected'] is not None, flush=True)
