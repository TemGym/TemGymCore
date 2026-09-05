"""Independent postfit checks for the conditional condenser/endpoint fit.

No optimizer, acquisition, fit selection, or new noise is run. Generating
values enter only after the saved training selection has been verified.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jnp
import numpy as np

from neoarm_aperture_capture import ROOT, XY
from neoarm_condenser_covariance_fit import VisibleSourceModel, visible_coordinates, quantum_completion
from neoarm_condenser_endpoint_calibration import EndpointRows, GROUPED, STEPS, digest
from neoarm_condenser_endpoint_recovery import JointEndpointModel, joint_training_residual
from neoarm_condenser_extension_design import load_context, covariance_factor, FOLDER
from neoarm_condenser_extension_recovery import selected_data
from neoarm_condenser_image_recovery import covariance_observable, covariance_from_observable
from neoarm_condenser_reference_source import ReferenceSourceChart
from neoarm_shared_objective_audit import provenance_snapshot, verify_provenance_snapshot


def verified_report(path):
    report = json.loads(Path(path).read_text())
    verify_provenance_snapshot(report["source_hashes"])
    return report


def verified_arrays(report):
    path = ROOT/report["arrays_path"]
    if digest(path) != report["arrays_sha256"]:
        raise ValueError("Inference image archive has changed")
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def reconstruct_response(image_parameters, image_covariances):
    """Independent central differences, in pixels, from eight distinct frames."""
    p, c = np.asarray(image_parameters), np.asarray(image_covariances)
    if p.shape != (2, 13, 6) or c.shape != (2, 13, 6, 6):
        raise ValueError("Expected two analysers with thirteen fitted images")
    response = np.stack([.5*(p[:, 1+2*i, :2]-p[:, 2+2*i, :2]) for i in range(4)], -1)
    covariance = np.zeros((2, 8, 8))
    for i in range(4):
        block = .25*(c[:, 1+2*i, :2, :2]+c[:, 2+2*i, :2, :2])
        for a in range(2):
            for b in range(2):
                covariance[:, 4*a+i, 4*b+i] = block[:, a, b]
    return response, covariance


def block_diagnostics(residual):
    r = np.asarray(residual)
    if r.shape != (119,):
        raise ValueError("Expected 96 beam, six aperture, sixteen calibration and one prior residuals")
    return {name: dict(count=b-a, chi_square=float(r[a:b]@r[a:b])) for name, a, b in
        (("beam", 0, 96), ("aperture", 96, 102), ("calibration", 102, 118), ("radius_prior", 118, 119))}


def predicted_check(residual, p, covariance):
    r = np.asarray(jax.jit(residual)(p))
    derivative = np.asarray(jax.jit(jax.jacfwd(residual))(p))
    innovation = np.eye(r.size)+derivative@covariance@derivative.T
    return dict(coordinate_count=r.size, raw_chi_square=float(r@r),
        local_predictive_chi_square=float(r@np.linalg.solve(innovation, r)))


def run(output):
    if jax.default_backend() != "gpu" or not jax.config.x64_enabled:
        raise RuntimeError("The optical and derivative validation requires GPU float64")
    started = perf_counter()
    sources, computational = provenance_snapshot()
    fit_path = FOLDER/"condenser_endpoint_recovery.json"
    cal_path = FOLDER/"condenser_endpoint_calibration.json"
    extension_path = FOLDER/"condenser_extension_images.json"
    fixed_path = FOLDER/"condenser_stationary_validation.json"
    previous_path = FOLDER/"condenser_geometry_profile_diagnostic.json"
    fit, cal, extension, fixed, previous = [verified_report(path) for path in
        (fit_path, cal_path, extension_path, fixed_path, previous_path)]
    if not fit["accepted_stationary_fit"]:
        raise ValueError("Validation requires the already accepted training fit")
    p = np.asarray(fit["parameters"])
    selection_path = fit_path.with_suffix(".selection.json")
    selection = json.loads(selection_path.read_text())
    np.testing.assert_array_equal(selection["parameters"], p)
    calibration, fresh_arrays = verified_arrays(cal), verified_arrays(extension)
    responses, response_covariance = reconstruct_response(
        calibration["image_parameters"], calibration["image_parameter_covariances"])
    response_reconstruction_error = float(np.max(abs(responses-calibration["response"])))
    covariance_reconstruction_error = float(np.max(abs(response_covariance-calibration["response_covariance"])))
    np.testing.assert_allclose(responses, calibration["response"], rtol=0, atol=1e-14)
    np.testing.assert_allclose(response_covariance, calibration["response_covariance"], rtol=1e-13, atol=1e-20)
    _, old_arrays, _, aperture, model = load_context()
    quotient = VisibleSourceModel(model)
    endpoints = EndpointRows(cal["conditional_sample_kappa_per_m"])
    split = model.column.objective.split_maps(jnp.asarray(model.fixed[-4:]))
    joint = JointEndpointModel(quotient, endpoints, split.pre)
    fresh = selected_data(fresh_arrays, True)
    beam_controls = jnp.asarray(np.concatenate((old_arrays["controls"][:8], fresh[0])))
    beam_data = jnp.asarray(np.concatenate((old_arrays["beam_observables"][:8], fresh[1])))
    beam_root = jnp.asarray(np.linalg.cholesky(np.concatenate((old_arrays["beam_observable_covariances"][:8], fresh[2]))))
    ap_controls = jnp.asarray([row["controls"] for row in aperture["records"][:2]])
    ap_data = jnp.asarray(old_arrays["aperture_observables"][:2])
    ap_root = jnp.asarray(np.linalg.cholesky(old_arrays["aperture_observable_covariances"][:2]))
    cal_root = jnp.asarray(np.linalg.cholesky(response_covariance))

    def independent_beam_prediction(parameters, controls):
        optics, source, _, _ = quotient.decode(parameters[:23])
        # Use the mechanical boundary explicitly, unlike the production
        # canonical-only composition, and test cancellation below.
        shear_grouped = endpoints.shear
        mechanical_pre = jnp.linalg.solve(shear_grouped, jnp.asarray(split.pre)[GROUPED])
        mechanical_rows = endpoints.canonical_rows(parameters[23:])@shear_grouped
        pre_rows = mechanical_rows@mechanical_pre
        def one(control):
            rows = pre_rows@model.condenser_matrix(optics, control)
            c = rows@source@jnp.swapaxes(rows, -1, -2)
            return jax.vmap(covariance_observable)(c/endpoints.pixel_m**2)
        return jax.vmap(one)(controls)

    def residual(parameters):
        optics, source, angle, radius = quotient.decode(parameters[:23])
        def at_aperture(control):
            m = model.condenser_matrix(optics, control, to_aperture=True)
            covariance = m@source@m.T
            from neoarm_condenser_image_recovery import rotation
            r = rotation(angle)
            covariance = r@covariance[jnp.ix_(jnp.asarray(XY), jnp.asarray(XY))]@r.T
            return covariance_observable(covariance)-jnp.array([radius, radius, 0.])
        b = independent_beam_prediction(parameters, beam_controls)
        a = jax.vmap(at_aperture)(ap_controls)
        measured_response = (endpoints.canonical_rows(parameters[23:])@endpoints.shear)*STEPS/endpoints.pixel_m
        return jnp.r_[jnp.linalg.solve(beam_root, (b-beam_data)[..., None]).reshape(-1),
            jnp.linalg.solve(ap_root, (a-ap_data)[..., None]).reshape(-1),
            jnp.linalg.solve(cal_root, (measured_response-responses).reshape(2, 8, 1)).reshape(-1), radius/.005]

    native_residual = joint_training_residual(joint, old_arrays, aperture, fresh, (responses, response_covariance))
    r = np.asarray(jax.jit(residual)(p))
    native_r = np.asarray(jax.jit(native_residual)(p))
    closure_sigma = float(np.max(abs(r-native_r)))
    if closure_sigma > 1e-7:
        raise ValueError("Independent mechanical and native canonical likelihoods disagree")
    # Stationarity is assessed on the EXACT original residual; the independent
    # equivalent matrix path can amplify roundoff in this squeezed column.
    jac = np.asarray(jax.jit(jax.jacfwd(native_residual))(p))
    norms = np.linalg.norm(jac, axis=0)
    singular = np.linalg.svd(jac/norms, compute_uv=False)
    gradient = float(np.max(abs((jac/norms).T@native_r)))
    chart = ReferenceSourceChart(quotient, previous["reference_control"], previous["reference_width_m"])
    encode = lambda value: jnp.r_[chart.encode(value[:23]), value[23:]]
    decode = lambda value: jnp.r_[chart.decode(value[:23]), value[23:]]
    reference = np.asarray(encode(p))
    ref_residual = lambda value: native_residual(decode(value))
    ref_jac = np.asarray(jax.jit(jax.jacfwd(ref_residual))(reference))
    ref_factor = covariance_factor(ref_jac)
    original_factor = np.asarray(jax.jacfwd(decode)(reference))@ref_factor
    covariance = original_factor@original_factor.T
    whiten_error = float(np.linalg.norm((jac@original_factor).T@(jac@original_factor)-np.eye(29)))
    full_gradient = float(np.linalg.norm((jac@original_factor).T@native_r))
    hessian = np.asarray(jax.jit(jax.hessian(lambda z: .5*jnp.sum(
        ref_residual(reference+jnp.asarray(ref_factor)@z)**2)))(jnp.zeros(29)))
    hessian_eigenvalues = np.linalg.eigvalsh((hessian+hessian.T)/2)
    physical_derivative = np.asarray(jax.jacfwd(lambda v: quotient.decode(v[:23])[0])(p))
    physical_factor = physical_derivative@original_factor
    standard = np.linalg.norm(physical_factor, axis=1)
    # At the SAME fitted point/data: hold endpoint coordinates fixed. This
    # isolates calibration-uncertainty inflation from movement of the optimum.
    fixed_ref_factor = covariance_factor(ref_jac[:, :23])
    fixed_factor = np.asarray(jax.jacfwd(chart.decode)(reference[:23]))@fixed_ref_factor
    fixed_standard = np.linalg.norm(physical_derivative[:, :23]@fixed_factor, axis=1)

    # All generating values are read after the immutable training selection.
    truth_path = ROOT/extension["truth_path"]
    cal_truth_path = ROOT/cal["truth_path"]
    if digest(truth_path) != extension["truth_sha256"] or digest(cal_truth_path) != cal["truth_sha256"]:
        raise ValueError("Postfit generating values changed")
    with np.load(truth_path, allow_pickle=False) as archive:
        truth_optics = np.asarray(archive["optical_parameters"])[2:14]
        truth_source = np.asarray(archive["source_covariance"])
        truth_angle = float(archive["aperture_rotation_rad"])
    cal_truth = json.loads(cal_truth_path.read_text())
    truth_point = np.r_[quotient.optical_coordinates(truth_optics), visible_coordinates(truth_source),
        truth_angle, 0., cal_truth["endpoint_parameters"]]
    truth_r = np.asarray(jax.jit(native_residual)(truth_point))
    optics = np.asarray(quotient.decode(p[:23])[0])
    errors = optics-truth_optics
    held_cal_data = jnp.asarray(calibration["heldout_response"])
    held_cal_root = jnp.asarray(np.linalg.cholesky(calibration["heldout_covariance"]))
    directions = jnp.asarray(calibration["heldout_directions_grouped"])
    def held_cal(value):
        predicted = jnp.einsum("aij,fj->afi", endpoints.response(value[23:]), directions/STEPS)
        return jnp.linalg.solve(held_cal_root, (predicted-held_cal_data)[..., None]).reshape(-1)
    held = selected_data(fresh_arrays, False)
    held_root = jnp.asarray(np.linalg.cholesky(held[2]))
    held_beam = lambda value: jnp.linalg.solve(held_root,
        (joint.beam_observables(value, held[0])-held[1])[..., None]).reshape(-1)
    predicted = np.asarray(joint.beam_observables(p, held[0]))
    covariances = jax.vmap(jax.vmap(covariance_from_observable))
    diameter_errors = 100*(np.sqrt(np.asarray(jnp.linalg.eigvalsh(covariances(predicted)) /
        jnp.linalg.eigvalsh(covariances(jnp.asarray(held[1])))))-1)
    completion = quantum_completion(p[12:21], interval=True)
    for report in (fit, cal, extension, fixed, previous):
        sources.update(report["source_hashes"])
    for path in (Path(__file__), fit_path, cal_path, extension_path, fixed_path, previous_path,
        selection_path, ROOT/cal["arrays_path"], ROOT/extension["arrays_path"], truth_path, cal_truth_path):
        sources[str(path.relative_to(ROOT))] = digest(path)
    verify_provenance_snapshot(sources)
    report = dict(schema="neoarm-independent-joint-endpoint-validation-v1", source_hashes=sources,
        computational_input_hashes=computational, backend=jax.default_backend(), x64_enabled=True,
        new_noise_or_fit_or_selection_performed=False, generating_values_used_only_after_frozen_training_selection=True,
        coordinate_count=119, data_coordinate_count=118, prior_count=1, free_parameter_count=29,
        independently_reconstructed_response_error_pixels=response_reconstruction_error,
        independently_reconstructed_response_covariance_error=covariance_reconstruction_error,
        canonical_mechanical_residual_closure_sigma=closure_sigma,
        training_total_chi_square=float(native_r@native_r), training_blocks=block_diagnostics(native_r),
        generating_truth_total_chi_square=float(truth_r@truth_r), generating_truth_blocks=block_diagnostics(truth_r),
        normalized_gradient=gradient, full_information_gradient_l2=full_gradient,
        information_whitening_error=whiten_error, normalized_singular_values=singular.tolist(),
        observed_hessian_spectrum=hessian_eigenvalues.tolist(), positive_local_minimum=bool(hessian_eigenvalues[0] > 0),
        postfit_optical_truth_check=dict(lens_order=["CL1", "CL3", "C-mini"],
            coordinate_order=["centre_m", "log_length_m", "q_bias", "q_slope"],
            parameters=optics.reshape(3, 4).tolist(), standard_errors=standard.reshape(3, 4).tolist(),
            errors_sigma=(errors/standard).reshape(3, 4).tolist(),
            centre_errors_mm=(errors.reshape(3, 4)[:, 0]*1e3).tolist(),
            length_errors_percent=(100*np.expm1(errors.reshape(3, 4)[:, 1])).tolist(),
            excitation_relative_errors_percent=(100*(optics.reshape(3, 4)[:, 2:]/truth_optics.reshape(3, 4)[:, 2:]-1)).tolist()),
        same_point_uncertainty_inflation_vs_fixed_endpoints=(standard/fixed_standard).reshape(3, 4).tolist(),
        uncertainty_ratio_vs_previous_fixed_calibration_fit=(standard/np.asarray(
            fixed["postfit_optical_truth_check"]["standard_errors"]).ravel()).reshape(3, 4).tolist(),
        new_heldout_calibration=predicted_check(held_cal, p, covariance),
        previously_inspected_condenser_checks=predicted_check(held_beam, p, covariance),
        previously_inspected_beam_diameter_errors_percent=diameter_errors.tolist(),
        quantum_margin=completion["margin"], invisible_source_covariance_interval=completion["feasible_interval"],
        limitations=["Warm-start conditional condenser plus two endpoint position rows, not full physical column recovery.",
            "Objective prefield, sample shear, input calibration, pitch, PSF and aperture plane remain conditional.",
            "Fresh endpoint checks are independent; condenser checks were previously inspected and are not fresh confirmation.",
            "Local expected-information uncertainties are not nonlinear profile intervals or branch probabilities.",
            "The invisible source covariance family remains free; no upstream stigmator location is inferred."],
        elapsed_seconds=perf_counter()-started)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: report[key] for key in ("training_total_chi_square", "generating_truth_total_chi_square",
        "normalized_gradient", "positive_local_minimum", "elapsed_seconds")}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    run(args.output)
