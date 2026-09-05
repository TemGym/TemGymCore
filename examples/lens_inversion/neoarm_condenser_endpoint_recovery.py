"""Joint condenser/endpoint calibration using original independent image data.

All 23 condenser/source/aperture coordinates and six downstream canonical
position-row coordinates are free. The old 102 condenser data coordinates,
16 NEW signed calibration responses and ONE radius prior enter once. The
objective prefield/shear and external input/pixel scales remain conditional.
This is not a physical four-projector or whole-objective recovery claim.
"""

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
from neoarm_condenser_covariance_fit import VisibleSourceModel, quantum_completion, geometry_gaps
from neoarm_condenser_endpoint_calibration import EndpointRows, GROUPED, STEPS, digest
from neoarm_condenser_extension_design import load_context, training_residual, covariance_factor, FOLDER
from neoarm_condenser_extension_recovery import selected_data, beam_residual
from neoarm_condenser_geometry_profile_diagnostic import GEOMETRY_INDICES, NUISANCE_INDICES
from neoarm_condenser_image_recovery import covariance_observable
from neoarm_condenser_reference_source import ReferenceSourceChart
from neoarm_condenser_scaled_cl1_profile import scaled_profiled_least_squares
from neoarm_condenser_stationary_validation import freeze_selection
from neoarm_shared_objective_audit import provenance_snapshot

ORDER = np.asarray((*GEOMETRY_INDICES, *NUISANCE_INDICES, *range(23, 29)))
INVERSE = np.argsort(ORDER)


def verify(report):
    for name, checksum in report["source_hashes"].items():
        if digest(ROOT/name) != checksum:
            raise ValueError(f"Stale input: {name}")


def calibration_training(archive):
    """Return only the declared independent training responses and covariance."""
    data = np.asarray(archive["response"])
    covariance = np.asarray(archive["response_covariance"])
    if data.shape != (2, 2, 4) or covariance.shape != (2, 8, 8):
        raise ValueError("Two calibrated 2x4 responses with separate full8x8 covariances required.")
    if not np.isfinite(data).all() or not np.isfinite(covariance).all():
        raise ValueError("Finite calibration observations required.")
    if not np.allclose(covariance, covariance.swapaxes(-1, -2), rtol=1e-12, atol=1e-18):
        raise ValueError("Symmetric calibration covariance required.")
    np.linalg.cholesky(covariance)
    return data, covariance


class JointEndpointModel:
    def __init__(self, quotient, endpoints, prefield_canonical):
        self.quotient = quotient
        self.endpoints = endpoints
        self.prefield = jnp.asarray(prefield_canonical)

    def beam_observables(self, p, controls):
        optical, source, _, _ = self.quotient.decode(p[:23])
        # Endpoint rows and objective prefield are BOTH canonical. No sample
        # mechanical shear belongs here; it occurs once in calibration images.
        before = self.endpoints.canonical_rows(p[23:])[:, :, GROUPED]@self.prefield
        def image(control):
            rows = before@self.quotient.model.condenser_matrix(optical, control)
            covariance = rows@source@jnp.swapaxes(rows, -1, -2)
            return jax.vmap(covariance_observable)(covariance/self.endpoints.pixel_m**2)
        return jax.vmap(image)(jnp.asarray(controls))

    def admissible(self, p):
        if np.shape(p) != (29,) or not np.isfinite(p).all():
            return False
        ab = np.asarray(p)[23:].reshape(2, 3)[:, :2]
        return bool(np.all(np.linalg.norm(ab, axis=1) > 1e-9)
                    and self.quotient.admissible(np.asarray(p)[:23]))


def joint_training_residual(joint, old, aperture, fresh, calibration):
    controls = np.concatenate((np.asarray(old["controls"][:8]), fresh[0]))
    observations = np.concatenate((np.asarray(old["beam_observables"][:8]), fresh[1]))
    covariance = np.concatenate((np.asarray(old["beam_observable_covariances"][:8]), fresh[2]))
    root = jnp.asarray(np.linalg.cholesky(covariance))
    cal_data, cal_cov = calibration
    cal_root = jnp.asarray(np.linalg.cholesky(cal_cov))
    ap_control = jnp.asarray([row["controls"] for row in aperture["records"][:2]])
    ap_data = jnp.asarray(old["aperture_observables"][:2])
    ap_root = jnp.asarray(np.linalg.cholesky(old["aperture_observable_covariances"][:2]))
    def residual(p):
        beam = joint.beam_observables(p, controls)
        _, ap = joint.quotient.observables(p[:23], jnp.zeros((0, 7)), ap_control)
        cal = joint.endpoints.response(p[23:])
        return jnp.r_[jnp.linalg.solve(root, (beam-observations)[..., None]).reshape(-1),
            jnp.linalg.solve(ap_root, (ap-ap_data)[..., None]).reshape(-1),
            jnp.linalg.solve(cal_root, (cal-cal_data).reshape(2, 8, 1)).reshape(-1), p[22]/.005]
    return residual


def prediction_check(residual, p, covariance):
    r = np.asarray(jax.jit(residual)(p))
    derivative = np.asarray(jax.jit(jax.jacfwd(residual))(p))
    prediction = np.eye(len(r))+derivative@covariance@derivative.T
    return dict(coordinate_count=len(r), raw_chi_square=float(r@r),
        local_predictive_chi_square=float(r@np.linalg.solve(prediction, r)), residual=r.tolist())


def run(output, *, max_outer_iterations=20, max_inner_iterations=30):
    if jax.default_backend() != "gpu" or not jax.config.x64_enabled:
        raise RuntimeError("Forward and recovery require GPU float64.")
    started = perf_counter()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    cal_path = FOLDER/"condenser_endpoint_calibration.json"
    cal_report = json.loads(cal_path.read_text())
    verify(cal_report)
    cal_arrays = ROOT/cal_report["arrays_path"]
    if digest(cal_arrays) != cal_report["arrays_sha256"] or not cal_report["all_images_accepted"]:
        raise ValueError("Admitted calibration images and unchanged arrays required.")
    with np.load(cal_arrays, allow_pickle=False) as archive:
        calibration = calibration_training(archive)
    old, old_arrays, _, aperture, model = load_context()
    quotient = VisibleSourceModel(model)
    endpoints = EndpointRows(cal_report["conditional_sample_kappa_per_m"])
    split = model.column.objective.split_maps(jnp.asarray(model.fixed[-4:]))
    joint = JointEndpointModel(quotient, endpoints, split.pre)
    acquisition_path = FOLDER/"condenser_extension_images.json"
    acquisition = json.loads(acquisition_path.read_text())
    verify(acquisition)
    arrays_path = ROOT/acquisition["arrays_path"]
    if digest(arrays_path) != acquisition["arrays_sha256"]:
        raise ValueError("Condenser training archive checksum mismatch.")
    with np.load(arrays_path, allow_pickle=False) as archive:
        fresh = selected_data(archive, True)
    fit_path = FOLDER/"condenser_geometry_profile_diagnostic.json"
    previous = json.loads(fit_path.read_text())
    verify(previous)
    initial_cond = np.asarray(previous["continuation"]["original_parameters"])
    # Saved TRAINING fit initializes only; its posterior is never a prior.
    chart = ReferenceSourceChart(quotient, previous["reference_control"], previous["reference_width_m"])
    initial = np.r_[initial_cond, endpoints.encode_response(calibration[0])]
    original_residual = joint_training_residual(joint, old_arrays, aperture, fresh, calibration)
    decode = lambda p: jnp.r_[chart.decode(p[jnp.asarray(INVERSE)][:23]), p[jnp.asarray(INVERSE)][23:]]
    encode = lambda p: jnp.r_[chart.encode(p[:23]), p[23:]][jnp.asarray(ORDER)]
    residual = lambda p: original_residual(decode(p))
    admissible = lambda p: joint.admissible(np.asarray(decode(p)))
    # Validate the new conditional map against the original at the declared
    # calibration, solely as a forward-contract check, not an initialization.
    fixed_response = np.asarray(model.downstream)[:, XY][:, :, GROUPED]*STEPS/endpoints.pixel_m
    fixed_rows = endpoints.encode_response(fixed_response)
    closure_point = np.r_[initial_cond, fixed_rows]
    old_prediction, _ = quotient.observables(initial_cond, fresh[0], jnp.zeros((0, 7)))
    closure = float(np.max(abs(np.asarray(joint.beam_observables(closure_point, fresh[0]))-old_prediction)))
    if closure > 1e-8:
        raise RuntimeError("Joint and original condenser forward maps do not close.")
    sources, _ = provenance_snapshot()
    for name in previous["source_hashes"]:
        sources[name] = previous["source_hashes"][name]
    for name in cal_report["source_hashes"]:
        sources[name] = cal_report["source_hashes"][name]
    for path in (Path(__file__), cal_path, cal_arrays, fit_path, acquisition_path, arrays_path):
        sources[str(path.relative_to(ROOT))] = digest(path)
    fit = scaled_profiled_least_squares(residual, np.asarray(encode(initial)), admissible,
        scale=1e-3, n_outer=6, max_iterations=max_outer_iterations,
        max_inner_iterations=max_inner_iterations, gradient_tolerance=1e-6)
    p = np.asarray(decode(fit["parameters"]))
    r = np.asarray(jax.jit(original_residual)(p))
    jac = np.asarray(jax.jit(jax.jacfwd(original_residual))(p))
    norms = np.linalg.norm(jac, axis=0)
    singular = np.linalg.svd(jac/norms, compute_uv=False)
    rank = int(np.sum(singular > singular[0]*1e-10))
    gradient = float(np.max(abs((jac/norms).T@r)))
    factor = covariance_factor(jac) if rank == 29 else None
    full_gradient = None if factor is None else float(np.max(abs((jac@factor).T@r)))
    accepted = bool(fit["success"] and rank == 29 and gradient < 1e-6
        and full_gradient < 1e-6 and joint.admissible(p))
    report = dict(schema="neoarm-joint-condenser-endpoint-recovery-v1", backend=jax.default_backend(),
        source_hashes=sources, all_twelve_condenser_optics_free=True, all_six_endpoint_coordinates_free=True,
        training_data_coordinate_count=118, original_condenser_data_coordinate_count=102,
        new_calibration_data_coordinate_count=16, radius_prior_count=1,
        initial_parameters=initial.tolist(), parameters=p.tolist(), total_chi_square=float(r@r),
        training_data_chi_square=float(r@r-(p[22]/.005)**2), radius_prior_chi_square=float((p[22]/.005)**2),
        normalized_gradient=gradient, full_information_gradient=full_gradient,
        normalized_singular_values=singular.tolist(), numerical_rank=rank,
        optimizer_success=fit["success"], termination=fit["reason"], accepted_stationary_fit=accepted,
        history=fit["history"], initial_nuisance_history=fit.get("initial_nuisance_history", []),
        fixed_calibration_forward_closure=closure, geometry_gaps_m=geometry_gaps(model, quotient.decode(p[:23])[0]).tolist(),
        quantum_margin=quantum_completion(p[12:21])["margin"],
        old_projector_posterior_used=False, previous_condenser_posterior_added_as_prior=False,
        source_initialization_from_previous_training_fit=True, blind_initialization_demonstrated=False,
        conditional_fixed_objective_prefield_and_shear=True, complete_physical_column_recovery=False,
        hardware_ready=False, elapsed_seconds=perf_counter()-started,
        limitations=["Joint conditional condenser and endpoint-map fit, not seven-lens physical recovery.",
            "Same shared-objective synthetic instrument; incompatible earlier projector posterior excluded.",
            "Exact externally calibrated positions/tilts and detector pitch; fixed objective prefield/shear.",
            "Unknown stable upstream astigmatic source, confirmed aperture geometry and clear paths assumed.",
            "One warm training initialization; global branches, cold starts and nonlinear coverage remain separate."])
    if accepted:
        covariance = factor@factor.T
        selection = dict(fit="sole joint training continuation", parameters=p.tolist(),
            training_chi_square=float(r@r), normalized_gradient=gradient,
            calibration_report_sha256=digest(cal_path), prior_fit_sha256=digest(fit_path))
        selection_path, _ = freeze_selection(output.with_suffix(".selection.json"), selection)
        sources[str(selection_path.relative_to(ROOT))] = digest(selection_path)
        with np.load(cal_arrays, allow_pickle=False) as archive:
            held_data = jnp.asarray(archive["heldout_response"])
            held_root = jnp.asarray(np.linalg.cholesky(archive["heldout_covariance"]))
            directions = jnp.asarray(archive["heldout_directions_grouped"])
        def held_calibration(p):
            response = endpoints.response(p[23:])
            predicted = jnp.einsum("aij,fj->afi", response, directions/STEPS)
            return jnp.linalg.solve(held_root, (predicted-held_data)[..., None]).reshape(-1)
        report["new_heldout_calibration"] = prediction_check(held_calibration, p, covariance)
        report["new_heldout_calibration"]["loaded_after_training_selection_frozen"] = True
        with np.load(arrays_path, allow_pickle=False) as archive:
            held = selected_data(archive, False)
        held_root = jnp.asarray(np.linalg.cholesky(held[2]))
        held_residual = lambda p: jnp.linalg.solve(held_root,
            (joint.beam_observables(p, held[0])-held[1])[..., None]).reshape(-1)
        report["previously_inspected_condenser_checks"] = prediction_check(held_residual, p, covariance)
        report["previously_inspected_condenser_checks"]["untouched_confirmatory_data"] = False
        physical = np.asarray(jax.jacfwd(lambda pp: quotient.decode(pp[:23])[0])(p))@factor
        report["parameter_covariance"] = covariance.tolist()
        report["condenser_length_standard_errors_percent"] = (100*np.linalg.norm(physical[[1, 5, 9]], axis=1)).tolist()
        report["condenser_centre_standard_errors_mm"] = (1e3*np.linalg.norm(physical[[0, 4, 8]], axis=1)).tolist()
    verify(report)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print("joint endpoint fit", accepted, float(r@r), gradient, flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-outer-iterations", type=int, default=20)
    parser.add_argument("--max-inner-iterations", type=int, default=30)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    run(args.output, max_outer_iterations=args.max_outer_iterations,
        max_inner_iterations=args.max_inner_iterations)
