"""Freeze the first stationary condenser fit, then inspect fresh predictions.

This is validation only: no optimizer is run and no generating parameter or
held-out observable participates in selection. The fit remains conditional on
the established objective/downstream maps and unknown upstream-source model.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jnp
import numpy as np

from neoarm_condenser_extension_design import load_context, training_residual, covariance_factor, FOLDER
from neoarm_condenser_extension_recovery import selected_data, beam_residual, prediction_diagnostic
from neoarm_condenser_covariance_fit import VisibleSourceModel, quantum_completion, geometry_gaps
from neoarm_condenser_image_recovery import covariance_from_observable
from neoarm_condenser_reference_source import ReferenceSourceChart
from neoarm_shared_objective_audit import provenance_snapshot
from neoarm_aperture_capture import ROOT


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources(report):
    for name, checksum in report["source_hashes"].items():
        if digest(ROOT/name) != checksum:
            raise ValueError(f"Changed computational input: {name}")


def freeze_selection(path, selection):
    """Permit exact reproduction, but never replace a heldout selection."""
    path = Path(path).resolve()
    content = json.dumps(selection, indent=2, allow_nan=False)+"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != content:
            raise ValueError("A different training selection requires a new validation artifact.")
    else:
        path.write_text(content)
    return path, digest(path)


def run(output):
    if jax.default_backend() != "gpu" or not jax.config.x64_enabled:
        raise RuntimeError("The optical checks require GPU float64.")
    output = Path(output).resolve()
    fit_path = FOLDER/"condenser_geometry_profile_diagnostic.json"
    fit_report = json.loads(fit_path.read_text())
    verify_sources(fit_report)
    fit = fit_report["continuation"]
    if not fit or not fit["original_stationary_success"] or fit_report["heldout_observations_used"]:
        raise ValueError("Select a stationary training-only fit before validation.")
    p = jnp.asarray(fit["original_parameters"])
    _, arrays, _, aperture, model = load_context()
    quotient = VisibleSourceModel(model)
    inference_path = FOLDER/"condenser_extension_images.json"
    inference = json.loads(inference_path.read_text())
    verify_sources(inference)
    array_path = ROOT/inference["arrays_path"]
    if digest(array_path) != inference["arrays_sha256"]:
        raise ValueError("Fresh image archive changed.")
    with np.load(array_path, allow_pickle=False) as archive:
        training = selected_data(archive, True)
    old_loss = training_residual(quotient, arrays, aperture)
    new_loss = beam_residual(quotient, training)
    residual = lambda p: jnp.r_[old_loss(p), new_loss(p)]
    r = np.asarray(jax.jit(residual)(p))
    jac = np.asarray(jax.jit(jax.jacfwd(residual))(p))
    gradient = float(np.max(abs((jac/np.linalg.norm(jac, axis=0)).T@r)))
    if gradient >= 1e-6 or not quotient.admissible(np.asarray(p)):
        raise ValueError("The saved original-coordinate fit no longer passes stationarity/physicality.")
    if not np.isclose(r@r, fit["chi_square"], rtol=1e-12, atol=1e-8):
        raise ValueError("The unchanged training likelihood did not reproduce.")
    chart = ReferenceSourceChart(quotient, fit_report["reference_control"], fit_report["reference_width_m"])
    reference = chart.encode(p)
    ref_residual = lambda s: residual(chart.decode(s))
    ref_jac = np.asarray(jax.jit(jax.jacfwd(ref_residual))(reference))
    factor = covariance_factor(ref_jac)
    original_factor = np.asarray(jax.jacfwd(chart.decode)(reference))@factor
    covariance = original_factor@original_factor.T
    whitening_error = float(np.linalg.norm((jac@original_factor).T@(jac@original_factor)-np.eye(23)))
    hessian = np.asarray(jax.jit(jax.hessian(lambda z:
        .5*jnp.sum(ref_residual(reference+jnp.asarray(factor)@z)**2)))(jnp.zeros(23)))
    spectrum = np.linalg.eigvalsh((hessian+hessian.T)/2)
    if spectrum[0] <= 0 or whitening_error > 1e-5:
        raise ValueError("A positive local minimum and stable full-rank information factor are required.")
    # Persist selection BEFORE reading either held-out observations or truth.
    selection_path = output.with_suffix(".selection.json")
    selection = dict(schema="neoarm-condenser-frozen-training-selection-v1",
        fit_report=str(fit_path.relative_to(ROOT)), fit_report_sha256=digest(fit_path),
        parameters=np.asarray(p).tolist(), training_total_chi_square=float(r@r),
        original_normalized_gradient=gradient, candidates="sole stationary training-only continuation",
        no_heldout_or_generating_values_used=True)
    selection_path, selection_hash = freeze_selection(selection_path, selection)

    with np.load(array_path, allow_pickle=False) as archive:
        held_data = selected_data(archive, False)
    held = prediction_diagnostic(beam_residual(quotient, held_data), p, covariance)
    held.update(fresh_selection_frozen_first=True, used_for_training_or_selection=False)
    prediction, _ = quotient.observables(p, held_data[0], jnp.zeros((0, 7)))
    observed_cov = jax.vmap(jax.vmap(covariance_from_observable))(jnp.asarray(held_data[1]))
    predicted_cov = jax.vmap(jax.vmap(covariance_from_observable))(prediction)
    ratios = np.sqrt(np.asarray(jnp.linalg.eigvalsh(predicted_cov)/jnp.linalg.eigvalsh(observed_cov)))
    held["principal_beam_diameter_errors_percent"] = (100*(ratios-1)).tolist()
    held["prediction_scope"] = "Covariance-derived beam images; unknown centroids and absolute dose are not optical observables here."

    bc, ac = jnp.asarray(arrays["controls"][8:]), jnp.asarray([row["controls"] for row in aperture["records"][2:]])
    bo, ao = jnp.asarray(arrays["beam_observables"][8:]), jnp.asarray(arrays["aperture_observables"][2:])
    br, ar = jnp.linalg.cholesky(arrays["beam_observable_covariances"][8:]), jnp.linalg.cholesky(arrays["aperture_observable_covariances"][2:])
    def old_held_residual(p):
        bp, ap = quotient.observables(p, bc, ac)
        return jnp.r_[jnp.linalg.solve(br, (bp-bo)[..., None]).reshape(-1),
                      jnp.linalg.solve(ar, (ap-ao)[..., None]).reshape(-1)]
    reused = prediction_diagnostic(old_held_residual, p, covariance)
    reused.update(previously_inspected_diagnostic_data=True, untouched_confirmatory_data=False)

    truth_path = ROOT/inference["truth_path"]
    if digest(truth_path) != inference["truth_sha256"]:
        raise ValueError("Postfit generating values changed.")
    with np.load(truth_path, allow_pickle=False) as generating:
        truth_optical = np.asarray(generating["optical_parameters"])[2:14]
    optical = np.asarray(quotient.decode(p)[0])
    physical_factor = np.asarray(jax.jacfwd(lambda p: quotient.decode(p)[0])(p))@original_factor
    se = np.linalg.norm(physical_factor, axis=1)
    error = optical-truth_optical
    optical_check = dict(parameters=optical.reshape(3, 4).tolist(),
        coordinate_names=["centre_m", "log_length_m", "q_bias", "q_slope"],
        standard_errors=se.reshape(3, 4).tolist(), errors_sigma=(error/se).reshape(3, 4).tolist(),
        centre_errors_mm=(error.reshape(3, 4)[:, 0]*1e3).tolist(),
        length_errors_percent=(100*np.expm1(error.reshape(3, 4)[:, 1])).tolist(),
        excitation_relative_errors_percent=(100*(optical.reshape(3, 4)[:, 2:]/truth_optical.reshape(3, 4)[:, 2:]-1)).tolist(),
        covariance_scope="Conditional local expected information, not constrained or nonlinear profile intervals.")
    completion = quantum_completion(np.asarray(p)[12:21], interval=True)
    hashes, _ = provenance_snapshot()
    for path in (Path(__file__), fit_path, inference_path, array_path, truth_path, selection_path):
        hashes[str(path.relative_to(ROOT))] = digest(path)
    for name in fit_report["source_hashes"]:
        if name not in hashes:
            hashes[name] = fit_report["source_hashes"][name]
    if digest(fit_path) != selection["fit_report_sha256"] or digest(selection_path) != selection_hash:
        raise RuntimeError("Frozen training selection changed during validation.")
    report = dict(schema="neoarm-condenser-stationary-validation-v1", backend=jax.default_backend(),
        x64_enabled=True, source_hashes=hashes, frozen_selection=selection,
        training_data_chi_square=float(r@r-(p[22]/.005)**2), training_data_coordinate_count=102,
        common_radius_prior_count=1, common_radius_prior_chi_square=float((p[22]/.005)**2),
        all_twelve_optical_parameters_free=True, original_normalized_gradient=gradient,
        full_information_whitening_error=whitening_error, full_observed_hessian_spectrum=spectrum.tolist(),
        fresh_heldout=held, reused_old_heldout=reused, postfit_optical_truth_check=optical_check,
        quantum_margin=completion["margin"], invisible_source_covariance_interval=completion["feasible_interval"],
        geometry_gaps_m=geometry_gaps(model, optical).tolist(),
        parameter_covariance=covariance.tolist(),
        stationary_conditional_training_fit=True,
        fresh_heldout_prediction_consistent=bool(held["predictive_chi_square"] < held["coordinate_count"]+5*np.sqrt(2*held["coordinate_count"])),
        optical_errors_within_three_local_standard_errors=bool(np.max(abs(error/se)) < 3),
        blind_initialization_demonstrated=False, global_uniqueness_proved=False,
        hardware_ready=False, complete_full_column_recovery=False,
        limitations=["This is the first stationary continuation; earlier offset geometry starts did not converge.",
            "The reference optical seed coincides with the generator; downstream/objective calibration is conditional.",
            "The highly chirped unknown source requires physical uncertainty and preparation checks.",
            "The unchanged upstream-astigmatic source assumption is not the unknown internal-stigmator model.",
            "Held-out beam predictions do not measure source properties directly or prove branch uniqueness."])
    verify_sources(report)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args().output)
    print(json.dumps({key: result[key] for key in ("fresh_heldout_prediction_consistent",
        "optical_errors_within_three_local_standard_errors", "postfit_optical_truth_check")}, indent=2))
