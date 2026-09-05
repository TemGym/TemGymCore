"""Five-input-per-analyser ablation of the existing joint endpoint experiment.

No new images/noise are generated. Reference, +/-x and +/-tilt-x are retained;
the omitted orthogonal and diagonal responses are previously inspected checks.
All 29 parameters remain free and the common radius prior occurs exactly once.
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

from neoarm_aperture_capture import ROOT
from neoarm_condenser_covariance_fit import VisibleSourceModel, quantum_completion
from neoarm_condenser_endpoint_calibration import EndpointRows, STEPS, digest
from neoarm_condenser_endpoint_recovery import (
    JointEndpointModel, joint_training_residual, prediction_check, ORDER, INVERSE,
)
from neoarm_condenser_endpoint_validation import verified_report, verified_arrays
from neoarm_condenser_extension_design import load_context, covariance_factor, FOLDER
from neoarm_condenser_extension_recovery import selected_data
from neoarm_condenser_reference_source import ReferenceSourceChart
from neoarm_condenser_scaled_cl1_profile import scaled_profiled_least_squares
from neoarm_condenser_stationary_validation import freeze_selection
from neoarm_shared_objective_audit import provenance_snapshot, verify_provenance_snapshot

TRAIN_RESPONSE = np.array([0, 2, 4, 6])
CHECK_RESPONSE = np.array([1, 3, 5, 7])
TRAIN_FRAMES = (0, 1, 2, 5, 6)
CHECK_FRAMES = (3, 4, 7, 8)


def select_response(data, covariance, indices=TRAIN_RESPONSE):
    """Marginal covariance of the selected raw coordinates, NOT whitened rows."""
    data, covariance, indices = np.asarray(data), np.asarray(covariance), np.asarray(indices)
    if data.shape != (2, 2, 4) or covariance.shape != (2, 8, 8):
        raise ValueError("Expected two row-major 2x4 responses with full 8x8 covariances")
    if indices.shape != (4,) or len(np.unique(indices)) != 4 or np.any((indices < 0) | (indices > 7)):
        raise ValueError("Select four distinct response coordinates per analyser")
    selected = data.reshape(2, 8)[:, indices]
    marginal = covariance[:, indices][:, :, indices]
    np.linalg.cholesky(marginal)
    return selected, marginal


def ablated_residual(full_residual, endpoints, calibration):
    data, covariance = select_response(*calibration)
    root = jnp.asarray(np.linalg.cholesky(covariance))
    def residual(p):
        full = full_residual(p)
        selected = endpoints.response(p[23:]).reshape(2, 8)[:, TRAIN_RESPONSE]
        return jnp.r_[full[:102], jnp.linalg.solve(root, (selected-data)[..., None]).reshape(-1), full[-1]]
    return residual


def physical_standard_errors(jacobian, physical_derivative):
    return np.linalg.norm(physical_derivative@covariance_factor(jacobian), axis=1)


def run(output, *, max_outer_iterations=15, max_inner_iterations=25):
    if jax.default_backend() != "gpu" or not jax.config.x64_enabled:
        raise RuntimeError("Optical fitting requires GPU float64")
    started = perf_counter()
    output = Path(output).resolve()
    sources, computational = provenance_snapshot()
    fit_path = FOLDER/"condenser_endpoint_recovery.json"
    cal_path = FOLDER/"condenser_endpoint_calibration.json"
    extension_path = FOLDER/"condenser_extension_images.json"
    chart_path = FOLDER/"condenser_geometry_profile_diagnostic.json"
    previous, calibration_report, extension, chart_report = [verified_report(path) for path in
        (fit_path, cal_path, extension_path, chart_path)]
    if not previous["accepted_stationary_fit"]:
        raise ValueError("The declared warm training start must be stationary")
    calibration_arrays, extension_arrays = verified_arrays(calibration_report), verified_arrays(extension)
    calibration = (calibration_arrays["response"], calibration_arrays["response_covariance"])
    _, old_arrays, _, aperture, model = load_context()
    quotient = VisibleSourceModel(model)
    endpoints = EndpointRows(calibration_report["conditional_sample_kappa_per_m"])
    split = model.column.objective.split_maps(jnp.asarray(model.fixed[-4:]))
    joint = JointEndpointModel(quotient, endpoints, split.pre)
    fresh = selected_data(extension_arrays, True)
    full_residual = joint_training_residual(joint, old_arrays, aperture, fresh, calibration)
    residual = ablated_residual(full_residual, endpoints, calibration)
    initial = np.asarray(previous["parameters"])
    chart = ReferenceSourceChart(quotient, chart_report["reference_control"], chart_report["reference_width_m"])
    encode = lambda p: jnp.r_[chart.encode(p[:23]), p[23:]][jnp.asarray(ORDER)]
    decode = lambda p: jnp.r_[chart.decode(p[jnp.asarray(INVERSE)][:23]), p[jnp.asarray(INVERSE)][23:]]
    ref_residual = lambda p: residual(decode(p))
    admissible = lambda p: joint.admissible(np.asarray(decode(p)))
    derivative = np.asarray(jax.jacfwd(lambda p: quotient.decode(p[:23])[0])(initial))
    initial_full_jac = np.asarray(jax.jit(jax.jacfwd(full_residual))(initial))
    initial_short_jac = np.asarray(jax.jit(jax.jacfwd(residual))(initial))
    full_standard = physical_standard_errors(initial_full_jac, derivative)
    short_standard = physical_standard_errors(initial_short_jac, derivative)
    fit = scaled_profiled_least_squares(ref_residual, np.asarray(encode(initial)), admissible,
        scale=1e-3, n_outer=6, max_iterations=max_outer_iterations,
        max_inner_iterations=max_inner_iterations, gradient_tolerance=1e-6)
    p = np.asarray(decode(fit["parameters"]))
    r = np.asarray(jax.jit(residual)(p))
    jac = np.asarray(jax.jit(jax.jacfwd(residual))(p))
    norms = np.linalg.norm(jac, axis=0)
    singular = np.linalg.svd(jac/norms, compute_uv=False)
    rank = int(np.sum(singular > singular[0]*1e-10))
    gradient = float(np.max(abs((jac/norms).T@r)))
    factor = covariance_factor(jac) if rank == 29 else None
    information_gradient = None if factor is None else float(np.max(abs((jac@factor).T@r)))
    accepted = bool(fit["success"] and rank == 29 and gradient < 1e-6
        and information_gradient < 1e-6 and joint.admissible(p))
    report = dict(schema="neoarm-five-input-endpoint-ablation-v1", backend=jax.default_backend(),
        x64_enabled=True, source_hashes=sources, computational_input_hashes=computational,
        reused_acquisition=True, new_noise_or_render_performed=False,
        training_image_indices_per_analyser=list(TRAIN_FRAMES),
        orthogonal_check_image_indices_per_analyser=list(CHECK_FRAMES),
        diagonal_check_image_indices_per_analyser=[9, 10, 11, 12],
        training_calibration_frame_count=10, orthogonal_check_frame_count=8,
        optional_diagonal_check_frame_count=8,
        condenser_data_coordinate_count=102, calibration_data_coordinate_count=8,
        training_data_coordinate_count=110, radius_prior_count=1, free_parameter_count=29,
        all_twelve_condenser_optics_and_six_endpoint_coordinates_free=True,
        same_point_physical_standard_error_ratio_five_vs_nine_inputs=(short_standard/full_standard).reshape(3, 4).tolist(),
        same_point_standard_error_coordinate_order=["centre_m", "log_length_m", "q_bias", "q_slope"],
        initial_parameters=initial.tolist(), parameters=p.tolist(),
        training_total_chi_square=float(r@r), training_data_chi_square=float(r[:-1]@r[:-1]),
        common_radius_prior_chi_square=float(r[-1]**2),
        training_blocks=dict(condenser=float(r[:102]@r[:102]),
            calibration=float(r[102:110]@r[102:110]), prior=float(r[-1]**2)),
        normalized_gradient=gradient, full_information_gradient=information_gradient,
        normalized_singular_values=singular.tolist(), numerical_rank=rank,
        optimizer_success=fit["success"], termination=fit["reason"], accepted_stationary_fit=accepted,
        history=fit["history"], initial_nuisance_history=fit.get("initial_nuisance_history", []),
        warm_start_uses_previous_full_calibration_training_fit=True,
        orthogonal_and_diagonal_data_are_untouched_confirmation=False,
        prior_relative_weights_changed=False, generating_truth_used=False,
        quantum_margin=quantum_completion(p[12:21])["margin"],
        blind_or_global_or_full_column_recovery=False, hardware_ready=False,
        limitations=["This is a reused-data warm-start measurement ablation, not a fresh confirmatory experiment.",
            "Reference images remain useful alignment/shape checks but contribute zero to signed central differences.",
            "Round canonical endpoints and known sample magnetic shear are assumed; orthogonal inputs are needed to test that model.",
            "Fixed objective prefield/shear, exact input and detector calibration, aperture plane and stable upstream source remain conditional.",
            "The independent source covariance family and global optical branches remain unresolved."])
    if accepted:
        selection = dict(experiment="five-input calibration ablation; one warm training start",
            parameters=p.tolist(), training_chi_square=float(r@r), prior_fit_sha256=digest(fit_path),
            all_orthogonal_and_diagonal_checks_previously_inspected=True)
        selection_path, _ = freeze_selection(output.with_suffix(".selection.json"), selection)
        covariance = factor@factor.T
        data, noise = select_response(*calibration, CHECK_RESPONSE)
        root = jnp.asarray(np.linalg.cholesky(noise))
        orthogonal = lambda p: jnp.linalg.solve(root,
            (endpoints.response(p[23:]).reshape(2, 8)[:, CHECK_RESPONSE]-data)[..., None]).reshape(-1)
        diagonal_data = jnp.asarray(calibration_arrays["heldout_response"])
        diagonal_root = jnp.asarray(np.linalg.cholesky(calibration_arrays["heldout_covariance"]))
        directions = jnp.asarray(calibration_arrays["heldout_directions_grouped"])
        diagonal = lambda p: jnp.linalg.solve(diagonal_root,
            (jnp.einsum("aij,fj->afi", endpoints.response(p[23:]), directions/STEPS)-diagonal_data)[..., None]).reshape(-1)
        held = selected_data(extension_arrays, False)
        held_root = jnp.asarray(np.linalg.cholesky(held[2]))
        beam = lambda p: jnp.linalg.solve(held_root,
            (joint.beam_observables(p, held[0])-held[1])[..., None]).reshape(-1)
        report["reused_orthogonal_checks"] = prediction_check(orthogonal, p, covariance)
        report["reused_diagonal_checks"] = prediction_check(diagonal, p, covariance)
        report["reused_condenser_checks"] = prediction_check(beam, p, covariance)
        physical = np.asarray(jax.jacfwd(lambda p: quotient.decode(p[:23])[0])(p))@factor
        report["physical_standard_errors"] = np.linalg.norm(physical, axis=1).reshape(3, 4).tolist()
        report["parameter_covariance"] = covariance.tolist()
        sources[str(selection_path.relative_to(ROOT))] = digest(selection_path)
    for prior in (previous, calibration_report, extension, chart_report):
        sources.update(prior["source_hashes"])
    for path in (Path(__file__), fit_path, cal_path, extension_path, chart_path,
                 ROOT/calibration_report["arrays_path"], ROOT/extension["arrays_path"]):
        sources[str(path.relative_to(ROOT))] = digest(path)
    verify_provenance_snapshot(sources)
    report["elapsed_seconds"] = perf_counter()-started
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: report[key] for key in ("accepted_stationary_fit", "training_total_chi_square",
        "normalized_gradient", "elapsed_seconds")}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    run(args.output)
