"""Preparation-specific model-autofocus beam-only information preflight.

Different preparations may yield different recorded C-mini controls.  The
controller is not assumed to know Q: perfect model autofocus is a computational
proxy for a future independent image-driven setup stage, not its validation.
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

from neoarm_condenser_internal_beam_design import (
    LOWER, UPPER, DEFOCUS_OFFSETS, RMS_GATE, CAPTURE_GATE, DOSE, MTF,
    covariance_at, observables_at, hypotheses, capture_lower_bound, largest_component, information,
)
from neoarm_condenser_extension_design import load_context, approximate_covariance_noise, greedy_design, FOLDER
from neoarm_condenser_stigmator_contract import ROOT
from neoarm_condenser_general_source_contract import initial_parameters
from neoarm_condenser_actual_analyser_contract import PIXEL_M
from temgym_core.gaussian_image_likelihood import (
    make_gaussian_detector_operator, encode_gaussian_image_parameters, gaussian_image_information_factor,
)
from temgym_core.neoarm_calibration import DetectorGeometry


def autofocus(model, parameters, position_pairs):
    """Coarse bounded global C-mini scan, then fixed-iteration local refinement."""
    ab = np.asarray(position_pairs)
    cm = np.linspace(LOWER[2], UPPER[2], 65)
    scan = np.c_[np.repeat(ab, len(cm), axis=0), np.tile(cm, len(ab))]
    covariance = jax.jit(lambda u: covariance_at(model, parameters, u))
    score = np.asarray(jnp.trace(covariance(jnp.asarray(scan)), axis1=-2, axis2=-1).sum(axis=-1)).reshape(len(ab), len(cm))
    index = score.argmin(axis=1)
    lo, hi = jnp.asarray(cm[np.maximum(index-1, 0)]), jnp.asarray(cm[np.minimum(index+1, len(cm)-1)])
    def objective(c):
        return jnp.trace(covariance(jnp.c_[jnp.asarray(ab), c]), axis1=-2, axis2=-1).sum(axis=-1)
    function = jax.jit(objective); ratio = (np.sqrt(5)-1)/2
    for _ in range(55):
        x, y = hi-ratio*(hi-lo), lo+ratio*(hi-lo)
        left = function(x) < function(y)
        lo, hi = jnp.where(left, lo, x), jnp.where(left, y, hi)
    return np.asarray((lo+hi)/2)


def independent_roles(controls, selected, heldout_count):
    """Select heldout control midpoints without accessing any data or truth."""
    selected = np.asarray(selected, int)
    unused = np.setdiff1d(np.arange(len(controls)), selected)
    if len(unused) < heldout_count:
        raise ValueError('Distinct heldout controls are required.')
    hold = []; scale = UPPER-LOWER
    for i in range(heldout_count):
        target = .5*(controls[selected[i]]+controls[selected[(i+1)%len(selected)]])
        index = int(unused[np.argmin(np.sum(((controls[unused]-target)/scale)**2, axis=1))])
        hold.append(index); unused = unused[unused != index]
    return np.r_[selected, hold], np.r_[np.ones(len(selected), bool), np.zeros(len(hold), bool)]


def exact_information(model, parameters, controls, training, pixel_functions, operator, detector):
    c = np.asarray(jax.jit(lambda u: covariance_at(model, parameters, u))(jnp.asarray(controls)))
    noise = {order: [] for order in pixel_functions}; captures = []; image_ranks = []
    for covariance in c.reshape(-1, 2, 2):
        encoded = encode_gaussian_image_parameters(np.zeros(2), covariance*PIXEL_M**2, DOSE, detector)
        for order, function in pixel_functions.items():
            image_j = np.asarray(function(encoded)); _, s, v = np.linalg.svd(image_j, full_matrices=False)
            if order == 32:
                image_ranks.append(int(np.sum(s > s[0]*1e-10)))
            factor = v.T/s; noise[order].append((factor@factor.T)[2:5, 2:5])
        captures.append(float(np.asarray(operator(np.zeros(2), covariance*PIXEL_M**2, DOSE)).sum()/DOSE))
    noise = {order: np.asarray(value).reshape(-1, 2, 3, 3) for order, value in noise.items()}
    jac = np.asarray(jax.jit(jax.jacfwd(lambda p: observables_at(model, p, jnp.asarray(controls))))(parameters))
    whitening = np.linalg.solve(np.linalg.cholesky(noise[32][training]), jac[training])
    info, _ = information(whitening)
    refinement = []
    for before, after in ((8, 16), (16, 32)):
        difference = np.linalg.norm(noise[before]-noise[after], axis=(-2, -1))/np.linalg.norm(noise[after], axis=(-2, -1))
        refinement.append(dict(orders=[before, after], maximum_observable_covariance_relative_change=float(difference.max())))
    return dict(information=info, full_marginalized_covariance_observable_noise=noise[32].tolist(),
        minimum_capture=min(captures), minimum_image_information_rank=min(image_ranks),
        information_quadrature_refinements=refinement,
        one_percent_length_precision=bool(info['rank'] == 25 and max(info['length_standard_errors_percent']) <= 1.))


def audit(output, *, training_count=12, heldout_count=4, broad_grid_count=33):
    if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
        raise RuntimeError('Use GPU float64 for the covariance/pixel-information preflight.')
    start = perf_counter(); _, _, _, _, conditional = load_context()
    models, frozen = hypotheses(conditional); p = jnp.asarray(initial_parameters())
    failed_path = FOLDER/'condenser_internal_beam_design.json'
    failed = json.loads(failed_path.read_text()); focus = np.asarray(failed['common_focus'])
    axes = [np.unique(np.r_[np.linspace(LOWER[i], UPPER[i], broad_grid_count), focus[i]+np.linspace(-.02, .02, 17)]) for i in range(2)]
    ab = np.array(list(itertools.product(*axes)))
    detector = DetectorGeometry(mtf_sigma_px=MTF)
    operator = jax.jit(make_gaussian_detector_operator(detector, 32))
    pixel_functions = {order: jax.jit(lambda x, order=order: gaussian_image_information_factor(x, detector, quadrature_order=order)) for order in (8, 16, 32)}
    routes = []
    for model_index, model in enumerate(models):
        cm = autofocus(model, p, ab)
        controls = np.array([[a, b, c+d] for (a, b), c in zip(ab, cm) for d in DEFOCUS_OFFSETS])
        covariance_fun = jax.jit(lambda pp, u: covariance_at(model, pp, u))
        covariance = np.asarray(covariance_fun(p, jnp.asarray(controls)))
        widths = np.sqrt(np.linalg.eigvalsh(covariance)); capture = np.asarray(capture_lower_bound(covariance))
        in_bounds = np.all((controls >= LOWER)&(controls <= UPPER), axis=1)
        admitted = in_bounds&(widths[:, :, 0].min(axis=1) >= RMS_GATE)&(capture.min(axis=1) >= CAPTURE_GATE)
        component, sizes = largest_component(admitted.reshape(len(axes[0]), len(axes[1]), len(DEFOCUS_OFFSETS)))
        ids = np.flatnonzero(component.reshape(-1)); candidate = controls[ids]
        print('Autofocus route', model.location, 'admitted', admitted.sum(), 'connected', len(ids), flush=True)
        selected = None; all_information = None
        if len(ids) >= training_count+heldout_count:
            c = covariance[ids]
            noise = np.asarray(jax.jit(jax.vmap(jax.vmap(lambda cc: approximate_covariance_noise(cc, DOSE))))(jnp.asarray(c)))
            jac = np.asarray(jax.jit(jax.jacfwd(lambda pp: observables_at(model, pp, jnp.asarray(candidate))))(p))
            whitened = np.linalg.solve(np.linalg.cholesky(noise), jac).reshape(len(ids), 6, 25)
            all_information, factor = information(whitened)
            print('All approximate', model.location, all_information['rank'], all_information.get('length_standard_errors_percent'), flush=True)
            if factor is not None:
                train, gains, _ = greedy_design((whitened@factor*1e3)[None], training_count)
                selection, role = independent_roles(candidate, train, heldout_count)
                selected_controls = candidate[selection]
                exact = exact_information(model, p, selected_controls, role, pixel_functions, operator, detector)
                approx, _ = information(whitened[np.asarray(train)])
                # This intentionally does not intersect different preparations.
                # Fixed-control corner failures diagnose why measured autofocus,
                # rather than a blindly universal command list, is required.
                corners = []
                for hypothesis in frozen:
                    if hypothesis['model_index'] != model_index:
                        continue
                    cc = np.asarray(covariance_fun(jnp.asarray(hypothesis['parameters']), jnp.asarray(selected_controls)))
                    cw = np.sqrt(np.linalg.eigvalsh(cc)); cp = np.asarray(capture_lower_bound(cc))
                    corners.append(dict(source_scale=hypothesis['source_scale'], strength_scale=hypothesis['strength_scale'],
                        Q_position_change_m=hypothesis['Q_position_change_m'], nominal=hypothesis['nominal'],
                        minimum_rms_pixels=float(cw.min()), maximum_rms_pixels=float(cw.max()),
                        minimum_capture_lower_bound=float(cp.min()),
                        fixed_controls_admitted=bool(cw[:, :, 0].min() >= RMS_GATE and cp.min() >= CAPTURE_GATE)))
                optical = np.asarray(model.decode(p)[0])
                q = optical[:, 2]+selected_controls*optical[:, 3]
                selected = dict(controls=selected_controls.tolist(), grid_indices=ids[selection].tolist(),
                    training_mask=role.tolist(), heldout_mask=(~role).tolist(),
                    proposed_detector_frames=2*len(selection), training_observable_coordinates=int(6*role.sum()),
                    heldout_observable_coordinates=int(6*(~role).sum()), greedy_gains=gains,
                    approximate_information_same_training_states=approx, exact_finite_pixel=exact,
                    minimum_nominal_rms_pixels=float(widths[ids[selection], :, 0].min()),
                    maximum_nominal_rms_pixels=float(widths[ids[selection], :, 1].max()),
                    minimum_nominal_capture_lower_bound=float(capture[ids[selection]].min()),
                    conditional_q_minimum=q.min(axis=0).tolist(), conditional_q_maximum=q.max(axis=0).tolist(),
                    nominal_first_positive_q_branch=bool(q.min() > 0 and q.max() < np.pi),
                    same_controls_without_reautofocus_corner_checks=corners,
                    eligible_for_one_percent_length_recovery_images=exact['one_percent_length_precision'])
                print('Exact route', model.location, exact['information']['rank'], exact['information'].get('length_standard_errors_percent'), flush=True)
        routes.append(dict(location=model.location, nominal_parameters=np.asarray(p).tolist(),
            CL1_CL3_axes=[axis.tolist() for axis in axes], autofocus_Cmini=cm.tolist(),
            defocus_offsets=DEFOCUS_OFFSETS.tolist(), nominal_admitted_mask=admitted.tolist(),
            grid_component_sizes=sizes, candidate_grid_indices=ids.tolist(),
            all_candidate_approximate_information=all_information, selected=selected))
    sources = dict(failed['source_hashes'])
    for path in (Path(__file__), ROOT/'tests/test_neoarm_condenser_autofocus_routes.py', failed_path):
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path, checksum in sources.items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != checksum:
            raise RuntimeError(f'A dependency changed: {path}')
    report = dict(schema='neoarm-preparation-specific-autofocus-routes-v1', source_hashes=sources,
        backend=jax.default_backend(), x64_enabled=True, elapsed_seconds=perf_counter()-start,
        frozen_preparation_hypotheses=frozen, broad_control_bounds=[LOWER.tolist(), UPPER.tolist()],
        actual_conditional_analyser_controls=np.asarray(conditional.analyser_controls).tolist(),
        incident_electrons_per_frame=DOSE, rms_gate_pixels=RMS_GATE, capture_gate=CAPTURE_GATE,
        routes=routes, source_and_Q_are_not_calibrated_inputs_to_information=True,
        all_twenty_five_coordinates_released=True, preparation_intersection_required=False,
        perfect_model_autofocus_is_only_a_controller_proxy=True, image_driven_autofocus_validated=False,
        aperture_collection_used=False, upstream_calibrated_kicks_used=False, current_readback_used=False,
        noisy_recovery_images_generated=False, optical_recovery_performed=False, hardware_admitted=False,
        limitations=['Each observed preparation may produce its own recorded autofocus controls; this is not a universal precomputed hardware recipe.',
            'Independent image-driven autofocus setup frames and their dose, noise, quantization, hysteresis and tracking errors have not been simulated.',
            'All25 source/Q/optical parameters remain free in each information calculation; knowing which hypothesis supplied a design is not granted to a future fitter.',
            'Component order is still a discrete hypothesis. No competing-Q-location fit or global excitation-branch search is performed.',
            'The largest finite grid component checks admitted vertices, not all interpolating hardware trajectories or a continuous uncertainty box.',
            'Selected training information excludes heldout controls. The approximate candidate score and1e−6 greedy ridge are not likelihood priors.',
            'Finite-pixel8/16/32 image-information refinement is not independent Gaussian-mode propagation or noisy-parameter recovery.',
            'If the preparation-specific routes remain weak, lower-magnification calibrated analyser pairs are the next design variable, rather than simply increasing40k image count.'])
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--broad-grid-count', type=int, default=33)
    args = parser.parse_args(); jax.config.update('jax_enable_x64', True)
    audit(args.output, broad_grid_count=args.broad_grid_count)
