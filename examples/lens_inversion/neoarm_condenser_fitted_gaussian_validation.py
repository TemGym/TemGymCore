"""Gaussian-mode forward validation at the frozen stationary condenser fit.

No observations, generating source, noise, optimizer or fit selection enter
this calculation. The six fresh held-out controls are transported using the
fitted physical lenses and one quantum-feasible representative of the fitted
source's explicitly retained invisible covariance family.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from neoarm_aperture_capture import ROOT
from neoarm_condenser_extension_design import load_context, FOLDER
from neoarm_condenser_covariance_fit import VisibleSourceModel, quantum_completion, geometry_gaps
from neoarm_condenser_gaussian_validation import validate_gaussian_modes, covariance_to_mixed_source
from neoarm_shared_objective_audit import provenance_snapshot, verify_provenance_snapshot
from temgym_core.neoarm_calibration import DetectorGeometry


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_report(path):
    report = json.loads(Path(path).read_text())
    for name, expected in report["source_hashes"].items():
        if digest(ROOT/name) != expected: raise ValueError(f"Changed source: {name}")
    return report


def selected_parameters(fit_report, frozen_selection, fit_digest):
    selected = fit_report["continuation"]
    if (not selected or not selected["original_stationary_success"]
            or fit_report["heldout_observations_used"]
            or not frozen_selection["no_heldout_or_generating_values_used"]):
        raise ValueError("A stationary training-only frozen selection is required")
    if frozen_selection["fit_report_sha256"] != fit_digest:
        raise ValueError("The fit report is not the previously frozen selection")
    parameters = np.asarray(selected["original_parameters"], float)
    if parameters.shape != (23,) or not np.all(np.isfinite(parameters)):
        raise ValueError("Expected23 finite visible-source fit coordinates")
    if not np.array_equal(parameters, np.asarray(frozen_selection["parameters"])):
        raise ValueError("Frozen parameters changed after held-out validation")
    return parameters


def heldout_controls(archive):
    mask, training = np.asarray(archive["heldout_mask"]), np.asarray(archive["training_mask"])
    if mask.dtype != bool or training.dtype != bool or np.any(mask & training) or not np.all(mask | training):
        raise ValueError("Frozen disjoint boolean role masks are required")
    # Deliberately read controls, not beam_observables or noisy images.
    controls = np.asarray(archive["controls"])[mask]
    analysers = np.asarray(archive["analyser_controls"])
    if controls.shape != (3, 7) or analysers.shape != (2, 7):
        raise ValueError("The frozen experiment has three heldout states and two analysers")
    return controls, analysers


def run(output):
    if jax.default_backend() != "gpu" or not jax.config.x64_enabled:
        raise RuntimeError("Fitted-source Gaussian forward validation requires GPU float64")
    started = perf_counter(); output = Path(output).resolve()
    fit_path = FOLDER/"condenser_geometry_profile_diagnostic.json"
    validation_path = FOLDER/"condenser_stationary_validation.json"
    inference_path = FOLDER/"condenser_extension_images.json"
    fit_report = verify_report(fit_path)
    validation = verify_report(validation_path)
    parameters = selected_parameters(fit_report, validation["frozen_selection"], digest(fit_path))
    inference = verify_report(inference_path)
    array_path = ROOT/inference["arrays_path"]
    if digest(array_path) != inference["arrays_sha256"]: raise ValueError("Frozen acquisition array checksum mismatch")
    with np.load(array_path, allow_pickle=False) as archive:
        controls, analysers = heldout_controls(archive)
    _, _, _, _, model = load_context()
    quotient = VisibleSourceModel(model)
    if not quotient.admissible(parameters): raise ValueError("Selected fitted geometry/source is no longer physical")
    optical = np.asarray(quotient.decode(jnp.asarray(parameters))[0])
    completion = quantum_completion(parameters[12:21], interval=True)
    covariance = np.asarray(completion["representative_covariance"])
    source, decomposition = covariance_to_mixed_source(covariance)
    physical = model.fixed.copy(); physical[2:14] = optical
    native_controls = controls[:, None]+analysers[None]
    matrices = np.asarray(jax.jit(jax.vmap(lambda u:
        model.column.matrix(physical, u, coordinates="canonical")))(native_controls.reshape(-1, 7)))
    fast = np.asarray(jax.jit(jax.vmap(lambda u: model.maps(optical, u)))(controls)).reshape(-1, 4, 4)
    closure = float(np.max(np.linalg.norm(matrices-fast, axis=(1, 2))/np.linalg.norm(matrices, axis=(1, 2))))
    if closure > 1e-10: raise ValueError("Fitted native-column map closure failed")
    sources, computational_inputs = provenance_snapshot()
    for previous in (fit_report, validation, inference): sources.update(previous["source_hashes"])
    for path in (fit_path, validation_path, inference_path, array_path):
        sources[str(path.relative_to(ROOT))] = digest(path)
    detector = DetectorGeometry(mtf_sigma_px=.65)
    # Dose sets the declared error scale; it is not a fitted optical observable.
    rows, images = validate_gaussian_modes(covariance, matrices, detector, 2e6,
        names=[f"fresh_heldout_state{state}_analyser{analyser}" for state in range(3) for analyser in range(2)])
    verify_provenance_snapshot(sources)
    output.parent.mkdir(parents=True, exist_ok=True)
    array_output = output.with_suffix(".npz")
    np.savez_compressed(array_output, fitted_source_covariance=covariance,
        fitted_source_pure_covariance=source.pure_covariance, fitted_source_centre_covariance=source.centre_covariance,
        fitted_optical_parameters=optical, full_native_fitted_matrices=matrices,
        controls=controls, analyser_controls=analysers, **images)
    report = dict(schema="neoarm-condenser-fitted-gaussian-validation-v1", source_hashes=sources,
        computational_input_hashes=computational_inputs, backend=jax.default_backend(),
        elapsed_seconds=perf_counter()-started, frozen_training_selection=validation["frozen_selection"],
        arrays_path=str(array_output.relative_to(ROOT)), arrays_sha256=digest(array_output),
        dose_for_validation_electrons=2e6, records=rows,
        all_six_fitted_heldout_images_accepted=all(row["accepted"] for row in rows),
        fitted_native_map_relative_closure=closure, geometry_gaps_m=geometry_gaps(model, optical).tolist(),
        fitted_source_decomposition=decomposition, fitted_source_quantum_margin=completion["margin"],
        invisible_covariance_completion_representative=completion["representative"],
        invisible_covariance_feasible_interval=completion["feasible_interval"],
        invisible_covariance_is_not_measured=True, source_reference_plane_is_coordinate_gauge=True,
        fitted_source_reference_z_m=model.source_z_m,
        generating_source_used=False, new_noise_generated=False, observations_used=False,
        fit_optimized_or_reselected=False, earlier_reports_or_sources_modified=False,
        canonical_full_fieldfree_source_to_detector_maps=True,
        no_aperture_or_specimen_masks=True,
        limitations=["Forward validation at one stationary fitted model and six fixed controls, not all parameter/branch families.",
            "The maximum-quantum-margin completion is an unobserved covariance-family representative, not a measured source angular momentum.",
            "The strongly chirped source covariance is represented at the declared reference plane; its reference position is not a recovered physical emitter position.",
            "Shared objective and downstream calibration, input Gaussian-state preparation, vacuum/clearance assumptions remain conditional.",
            "Independent field propagation closure does not establish blind/global optical recovery or hardware readiness."])
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(str(output), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); jax.config.update("jax_enable_x64", True)
    run(args.output)
