"""One bounded continuation of both displaced same-column projector starts.

The first20-iteration attempts remain unchanged. Their final training points
receive20 further iterations with the identical data, physical branch and no
priors. Only then are an accepted winner, reserved images and generating
parameters inspected. This is continuation of two cold-start trajectories,
not a new independent or global initialization demonstration.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
import jax
import numpy as np

from neoarm_shared_projector_recovery import (
    ROOT, FOLDER, digest, verify_provenance_snapshot, load_context, SharedProjectorModel,
    selected_data, make_residual, fit_one, PARAMETER_NAMES,
)


def run(output):
    if jax.default_backend() != "gpu" or not jax.config.x64_enabled:
        raise RuntimeError("GPU float64 is required")
    started = perf_counter()
    seed_path, data_path = FOLDER/"shared_projector_recovery.json", FOLDER/"shared_projector_images.json"
    previous, acquisition = [json.loads(path.read_text()) for path in (seed_path, data_path)]
    for report in (previous, acquisition): verify_provenance_snapshot(report["source_hashes"])
    if previous["best_index"] is not None or previous["post_selection_diagnostics"] is not None:
        raise ValueError("This prescribed continuation requires the untouched unaccepted first checkpoint")
    if [r["prescribed_diagram_displacement_sign"] for r in previous["rows"]] != [-1, 1]:
        raise ValueError("Exactly the original two trajectories must be preserved")
    array_path = ROOT/acquisition["arrays_path"]
    if digest(array_path) != acquisition["arrays_sha256"]:
        raise ValueError("Inference arrays changed")
    with np.load(array_path, allow_pickle=False) as archive: training = selected_data(archive, True)
    _, _, _, _, context = load_context(); model = SharedProjectorModel(context.column, context.fixed)
    rows = []
    for old in previous["rows"]:
        row = fit_one(model, training, np.asarray(old["parameters"]), max_outer_iterations=20)
        row["prescribed_diagram_displacement_sign"] = old["prescribed_diagram_displacement_sign"]
        row["original_cold_parameters"] = old["initial_parameters"]
        row["previous_training_chi_square"] = old["training_chi_square"]
        rows.append(row)
        print("bounded projector continuation", row["prescribed_diagram_displacement_sign"],
            row["accepted"], row.get("training_chi_square"), row.get("original_information_whitened_gradient"), flush=True)
    accepted = [i for i, row in enumerate(rows) if row["accepted"]]
    index = min(accepted, key=lambda i: rows[i]["training_chi_square"]) if accepted else None
    diagnostics = None
    sources = dict(previous["source_hashes"])
    for path in (Path(__file__), seed_path, data_path): sources[str(path.relative_to(ROOT))] = digest(path)
    if index is not None:
        best = rows[index]; p = np.asarray(best["parameters"]); cov = np.asarray(best["parameter_covariance"])
        with np.load(array_path, allow_pickle=False) as archive: held = selected_data(archive, False)
        loss = make_residual(model, held)
        r, jac = np.asarray(jax.jit(loss)(p)), np.asarray(jax.jit(jax.jacfwd(loss))(p))
        predictive = np.eye(len(r))+jac@cov@jac.T
        truth_path = ROOT/acquisition["truth_path"]
        if digest(truth_path) != acquisition["truth_sha256"]: raise ValueError("Post-selection truth changed")
        with np.load(truth_path, allow_pickle=False) as truth:
            generating = np.asarray(truth["stage_coordinates"]); noiseless = np.asarray(truth["expected_response"])
        true_r = np.asarray(jax.jit(make_residual(model, training))(generating))
        closure = np.asarray(jax.jit(model.predict)(generating, np.vstack((training[0], held[0]))))-noiseless
        se = np.sqrt(np.diag(cov))
        diagnostics = dict(heldout_coordinate_count=len(r), heldout_raw_chi_square=float(r@r),
            heldout_linear_predictive_chi_square=float(r@np.linalg.solve(predictive, r)),
            maximum_heldout_response_error_pixels=float(np.max(abs(np.asarray(model.predict(p, held[0]))-held[1]))),
            generating_training_chi_square=float(true_r@true_r), native_noiseless_truth_closure_pixels=float(np.max(abs(closure))),
            generating_parameters=generating.tolist(), length_gap_errors_percent=(100*np.expm1((p-generating).reshape(4, 4)[:, :2])).tolist(),
            length_gap_standard_errors_percent=(100*se.reshape(4, 4)[:, :2]).tolist(),
            parameter_error_local_standard_deviations=((p-generating)/se).tolist(),
            truth_and_heldout_loaded_after_training_selection=True)
        sources[str(truth_path.relative_to(ROOT))] = digest(truth_path)
    verify_provenance_snapshot(sources)
    report = dict(schema="neoarm-shared-physical-projector-bounded-refinement-v1", source_hashes=sources,
        previous_attempt_path=str(seed_path.relative_to(ROOT)), previous_attempt_sha256=digest(seed_path),
        acquisition_path=str(data_path.relative_to(ROOT)), acquisition_sha256=digest(data_path),
        arrays_path=str(array_path.relative_to(ROOT)), arrays_sha256=digest(array_path),
        backend=jax.default_backend(), x64_enabled=True, parameter_names=PARAMETER_NAMES,
        training_response_coordinate_count=96, free_parameter_count=16, prior_coordinate_count=0,
        original_prescribed_start_count=2, accepted_trajectory_count=len(accepted), rows=rows, best_index=index,
        accepted_stationary_fit=index is not None, post_selection_diagnostics=diagnostics,
        maximum_outer_updates_per_original_trajectory=40, source_cold_start_artifact_preserved=True,
        initialization_uses_generating_parameters=False, old_projector_posterior_or_affine_bridge_used=False,
        heldout_data_used_for_initialization_or_selection=False, same_physical_branch_and_likelihood=True,
        fixed_objective_and_detector_anchor_conditional=True, broad_branch_uniqueness_proved=False,
        full_column_recovery=False, elapsed_seconds=perf_counter()-started, limitations=previous["limitations"])
    output = Path(output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); jax.config.update("jax_enable_x64", True); run(args.output)
