"""Acquire the frozen condenser extension with physical Gaussian modes.

Inference arrays contain noisy images, controls/roles and measured image
observables only. Generating optics/source/native maps are separate truth-only
artifacts. The detector-image initializer and iterated noise weights use data,
never the generating covariance, dose or expected image.
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

from temgym_core.gaussian_image_likelihood import (
    make_gaussian_detector_operator, decode_gaussian_image_parameters,
    encode_gaussian_image_parameters, fit_gaussian_image,
)
from temgym_core.neoarm_calibration import DetectorGeometry
from neoarm_aperture_capture import ROOT, XY, load_cases
from neoarm_condenser_image_recovery import CondenserModel, read_verified_aperture, DEFAULT_APERTURE
from neoarm_condenser_gaussian_validation import render_covariance_gaussian_modes, validate_gaussian_modes
from neoarm_shared_objective_audit import provenance_snapshot, verify_provenance_snapshot

FOLDER = ROOT/"examples/generated/neoarm_inversion/imaging_recipe"
DEFAULT_DESIGN = FOLDER/"condenser_extension_design.json"


def read_design(path):
    design = json.loads(Path(path).read_text())
    for name, expected in design["source_hashes"].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen design provenance mismatch: {name}")
    controls = np.asarray(design["controls"], float)
    training, heldout = np.asarray(design["training_mask"], bool), np.asarray(design["heldout_mask"], bool)
    validate_roles(controls, training, heldout)
    if not design["frozen_before_new_images_and_noise"] or design["old_heldout_observations_used_for_scoring"]:
        raise ValueError("The design must precede new noise and exclude held-out observations from scoring")
    return design, controls, training, heldout


def validate_roles(controls, training, heldout):
    if controls.ndim != 2 or controls.shape[1] != 7 or not np.all(np.isfinite(controls)):
        raise ValueError("Finite seven-channel control vectors are required")
    if training.shape != (len(controls),) or heldout.shape != training.shape:
        raise ValueError("Role masks must correspond to optical states")
    if np.any(training & heldout) or not np.all(training | heldout):
        raise ValueError("Every state must have exactly one frozen role")


def data_only_image_initializer(image, detector, *, read_noise=.6):
    """Robust bounded-support signed moments for initialization only.

    A signal-derived bounding box prevents unilluminated read noise from
    dominating narrow-beam second moments. Every pixel remains in the fit.
    """
    image = np.asarray(image, float)
    if image.shape != detector.shape_yx or not np.all(np.isfinite(image)) or image.sum() <= 0:
        raise ValueError("A finite positive-total detector image is required")
    threshold = max(10*read_noise, .02*float(image.max()))
    support = np.argwhere(image > threshold)
    if len(support) < 2: raise ValueError("Insufficient signal for data-only initialization")
    lower, upper = support.min(axis=0), support.max(axis=0)+1
    margin = np.maximum(3, np.ceil(.25*(upper-lower)).astype(int))
    lower = np.maximum(lower-margin, 0); upper = np.minimum(upper+margin, image.shape)
    y, x = np.indices(image.shape)
    coords = np.stack([x+.5-image.shape[1]/2, y+.5-image.shape[0]/2], axis=-1)
    mask = (y >= lower[0]) & (y < upper[0]) & (x >= lower[1]) & (x < upper[1])
    selected = image*mask; total = selected.sum()
    if total <= 0: raise ValueError("Nonpositive signal-box total")
    mean = np.einsum("yx,yxi->i", selected, coords)/total
    delta = coords-mean
    covariance = np.einsum("yx,yxi,yxj->ij", selected, delta, delta)/total
    covariance -= np.eye(2)*(detector.mtf_sigma_px**2+1/12)
    values, vectors = np.linalg.eigh((covariance+covariance.T)/2)
    covariance = (vectors*np.maximum(values, .05**2))@vectors.T
    sigma = np.sqrt(np.diag(covariance))
    correlation = np.clip(covariance[0, 1]/np.prod(sigma), -.99, .99)
    return np.r_[mean, np.log(sigma), np.arctanh(correlation), np.log(image.sum())]


def fit_data_only_image(image, detector, *, quadrature_order=16, max_weight_updates=4):
    """Feasible GLS: image-derived initial state and fitted Poisson weights."""
    if jax.default_backend() != "gpu": raise RuntimeError("Image inference requires GPU")
    initial = data_only_image_initializer(image, detector)
    operator = jax.jit(make_gaussian_detector_operator(detector, quadrature_order))
    predict = lambda p: operator(*decode_gaussian_image_parameters(p, detector))
    variance = np.asarray(predict(initial))+.6**2
    history = []; previous = None; fit = None
    for index in range(max_weight_updates):
        fit = fit_gaussian_image(image, variance, detector, initial_parameters=initial,
            quadrature_order=quadrature_order, max_nfev=200)
        shift = None if previous is None else float(np.max(abs(np.asarray(fit.parameters)-previous)/np.sqrt(np.diag(fit.parameter_covariance))))
        history.append(dict(update=index, optimizer_success=bool(fit.optimizer_result.success),
            evaluations=int(fit.optimizer_result.nfev), parameter_shift_sigma=shift, chi_square_z=float(fit.chi_square_z)))
        if previous is not None and shift < .01: break
        previous = np.asarray(fit.parameters); initial = previous
        variance = np.asarray(fit.fitted_counts_yx)+.6**2
    accepted = bool(fit.optimizer_result.success and fit.numerical_rank == 6 and abs(fit.chi_square_z) < 5
        and fit.quadrature_rms_sigma < .2 and fit.quadrature_parameter_shift_sigma < .2
        and history[-1]["parameter_shift_sigma"] is not None and history[-1]["parameter_shift_sigma"] < .01)
    return fit, variance, dict(accepted=accepted, weight_updates=history,
        initializer="Signal-derived bounding-box signed moments; PSF/pixel variance removed for initialization only; all pixels enter every fit.",
        variance="Feasible GLS using fitted Poisson intensity plus0.6-electron independent read-noise variance; no generating expectation used.",
        variance_estimation_uncertainty_separately_propagated=False)


def acquire(design_path, output, *, seed=914271):
    if jax.default_backend() != "gpu" or not jax.config.x64_enabled:
        raise RuntimeError("Use GPU float64 for acquisition and fitting")
    started = perf_counter()
    design, controls, training, heldout = read_design(design_path)
    design_digest = hashlib.sha256(design_path.read_bytes()).hexdigest()
    aperture = read_verified_aperture(DEFAULT_APERTURE)
    info_path = next(ROOT/n for n in aperture["report_input_hashes"] if "information_narrow_probe" in n)
    info = json.loads(info_path.read_text())
    _, source_covariance, column, generating_parameters = load_cases(info_path)
    analysers = np.asarray(info["projection_states"]["controls"])[[0, 2]]
    if np.any(controls[:, 3:] != 0) or np.any(analysers[:, :3] != 0):
        raise ValueError("Condenser and analyser control fields must be disjoint")
    native_controls = controls[:, None]+analysers[None]
    matrix_fun = jax.jit(jax.vmap(lambda u: column.matrix(generating_parameters, u, coordinates="canonical")))
    native = np.asarray(matrix_fun(native_controls.reshape(-1, 7))).reshape(len(controls), 2, 4, 4)
    model = CondenserModel(column, generating_parameters, analysers)
    fast = np.asarray(jax.jit(jax.vmap(lambda u: model.maps(generating_parameters[2:14], u)))(controls))
    closure = np.linalg.norm(native-fast, axis=(-2, -1))/np.linalg.norm(native, axis=(-2, -1))
    ray = np.array([3e-9, 2e-6, -4e-9, -3e-6])
    traced = np.asarray(jax.jit(jax.vmap(lambda u: column.trace(generating_parameters, u, ray, coordinates="canonical")))(native_controls.reshape(-1, 7)))
    predicted = np.einsum("aij,j->ai", native.reshape(-1, 4, 4), ray)
    trace_error_px = float(np.max(abs((traced-predicted)[:, XY]))/55e-6)
    if closure.max() > 1e-10 or trace_error_px > 1e-7:
        raise ValueError("Native single-ray/full-column map closure failed")
    detector = DetectorGeometry(mtf_sigma_px=.65)
    sources, computational_inputs = provenance_snapshot()
    sources[str(design_path.relative_to(ROOT))] = design_digest
    rng = np.random.default_rng(seed); dose = float(design["incident_electrons_per_frame"])
    expected, noisy_images, observations, uncertainty, parameters, full_covariance, variances, rows = [], [], [], [], [], [], [], []
    operator = jax.jit(make_gaussian_detector_operator(detector, 32))
    transported = native@source_covariance@native.swapaxes(-1, -2)
    image_truth_parameters = []
    for index, matrix in enumerate(native.reshape(-1, 4, 4)):
        tick = perf_counter()
        image, modal_metadata = render_covariance_gaussian_modes(source_covariance, matrix, detector, dose,
            hermite_order=7, pixel_order=16)
        image = np.asarray(image)
        if not np.all(np.isfinite(image)) or np.min(image) < -1e-12:
            raise ValueError("Gaussian-mode image must be finite and nonnegative")
        covariance = transported.reshape(-1, 4, 4)[index][np.ix_(XY, XY)]
        reference = np.asarray(operator(jnp.zeros(2), covariance, dose))
        image_error = float(np.sqrt(np.mean((image-reference)**2/(reference+.6**2))))
        if image_error >= .2: raise ValueError("Independent full-pixel Gaussian/covariance gate failed")
        observed = rng.poisson(np.maximum(image, 0.))+rng.normal(0., .6, image.shape)
        fit, variance, fit_metadata = fit_data_only_image(observed, detector)
        p = np.asarray(fit.parameters)
        expected.append(image); noisy_images.append(observed); observations.append(p[2:5])
        uncertainty.append(fit.parameter_covariance[2:5, 2:5]); parameters.append(p)
        full_covariance.append(fit.parameter_covariance); variances.append(variance)
        image_truth_parameters.append(np.asarray(encode_gaussian_image_parameters(np.zeros(2), covariance, dose, detector)))
        row = dict(frame=index, state=index//2, analyser=index%2, training=bool(training[index//2]),
            accepted=fit_metadata["accepted"], image_fit=fit_metadata,
            image_rank=int(fit.numerical_rank), chi_square_z=float(fit.chi_square_z),
            residual_rms_sigma=float(fit.residual_rms_sigma),
            quadrature_rms_sigma=float(fit.quadrature_rms_sigma),
            quadrature_parameter_shift_sigma=float(fit.quadrature_parameter_shift_sigma),
            estimated_incident_electrons=float(np.exp(p[5])),
            independent_mode_reference_image_rms_sigma=image_error,
            gaussian_mode_count=modal_metadata["coherent_mode_count"],
            elapsed_seconds=perf_counter()-tick)
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("frame", "accepted", "chi_square_z", "independent_mode_reference_image_rms_sigma", "elapsed_seconds")}), flush=True)
    # This is a post-design validation choice, not a change of controls/roles.
    covariances = transported[:, :, XY][:, :, :, XY]/55e-6**2
    rms = np.sqrt(np.linalg.eigvalsh(covariances)).reshape(-1, 2)
    selected = sorted(set([int(np.argmin(rms.min(axis=1))), int(np.argmax(rms.max(axis=1)))]))
    validation_rows, validation_arrays = validate_gaussian_modes(source_covariance,
        native.reshape(-1, 4, 4)[selected], detector, dose, names=[f"extension_frame{i}" for i in selected])
    verify_provenance_snapshot(sources)
    if hashlib.sha256(design_path.read_bytes()).hexdigest() != design_digest:
        raise ValueError("Frozen design changed during acquisition")
    output.parent.mkdir(parents=True, exist_ok=True)
    shape = (len(controls), 2)
    inference_path = output.with_suffix(".npz")
    np.savez_compressed(inference_path, controls=controls, training_mask=training, heldout_mask=heldout,
        analyser_controls=analysers, native_controls=native_controls,
        beam_observables=np.asarray(observations).reshape(*shape, 3),
        beam_observable_covariances=np.asarray(uncertainty).reshape(*shape, 3, 3),
        image_parameters=np.asarray(parameters).reshape(*shape, 6),
        image_parameter_covariances=np.asarray(full_covariance).reshape(*shape, 6, 6),
        noisy_beam_images=np.asarray(noisy_images).reshape(*shape, *detector.shape_yx),
        fitted_variance_images=np.asarray(variances).reshape(*shape, *detector.shape_yx))
    truth_array_path = output.with_name(output.stem+"_truth.npz")
    collection_path = next(ROOT/n for n in aperture["report_input_hashes"] if "aperture_collection.json" in n)
    collection = json.loads(collection_path.read_text())
    collection_a = np.asarray(collection["aperture_collection"]["aperture_to_detector_matrices"][0])[np.ix_(XY, XY)]
    aperture_rotation = float(np.arctan2(collection_a[1, 0], collection_a[0, 0]))
    np.savez_compressed(truth_array_path, expected_gaussian_mode_images=np.asarray(expected).reshape(*shape, *detector.shape_yx),
        source_covariance=source_covariance, optical_parameters=generating_parameters,
        aperture_rotation_rad=aperture_rotation,
        full_native_matrices=native, image_parameters=np.asarray(image_truth_parameters).reshape(*shape, 6),
        native_controls=native_controls, dose_electrons=dose, **validation_arrays)
    truth_parameters = np.asarray(image_truth_parameters)
    truth_difference = np.asarray(observations)-truth_parameters[:, 2:5]
    whitened_truth = np.linalg.solve(np.linalg.cholesky(uncertainty), truth_difference[..., None]).reshape(len(controls), 2, 3)
    truth_report = dict(schema="neoarm-condenser-extension-image-truth-v1", source_hashes=sources,
        design_sha256=design_digest, truth_array_file=str(truth_array_path.relative_to(ROOT)),
        truth_array_sha256=hashlib.sha256(truth_array_path.read_bytes()).hexdigest(),
        generating_values_are_postfit_diagnostics_only=True,
        native_map_maximum_relative_closure=float(closure.max()), native_single_ray_position_closure_pixels=trace_error_px,
        training_observable_truth_chi_square=float(np.sum(whitened_truth[training]**2)),
        heldout_observable_truth_chi_square=float(np.sum(whitened_truth[heldout]**2)),
        per_frame_observable_truth_chi_square=np.sum(whitened_truth**2, axis=-1).tolist(),
        detector_captured_fraction=(np.asarray(expected).sum(axis=(-2, -1)).reshape(*shape)/dose).tolist(),
        selected_refinement_frame_indices=selected, independent_refinement_records=validation_rows)
    truth_path = output.with_name(output.stem+"_truth.json")
    truth_path.write_text(json.dumps(truth_report, indent=2, allow_nan=False)+"\n")
    report = dict(schema="neoarm-condenser-extension-images-v1", source_hashes=sources,
        computational_input_hashes=computational_inputs, design_path=str(design_path.relative_to(ROOT)),
        design_sha256=design_digest, arrays_path=str(inference_path.relative_to(ROOT)),
        arrays_sha256=hashlib.sha256(inference_path.read_bytes()).hexdigest(),
        truth_path=str(truth_array_path.relative_to(ROOT)), truth_sha256=hashlib.sha256(truth_array_path.read_bytes()).hexdigest(),
        truth_format="NPZ: source_covariance(4,4), optical_parameters(full34 shared-column vector), aperture_rotation_rad(scalar). Post-fit checks only.",
        truth_report_file=str(truth_path.relative_to(ROOT)), truth_report_sha256=hashlib.sha256(truth_path.read_bytes()).hexdigest(),
        backend=jax.default_backend(), elapsed_seconds=perf_counter()-started, noise_seed=seed,
        training_mask=training.tolist(), heldout_mask=heldout.tolist(), controls=controls.tolist(),
        training_image_coordinates=int(training.sum()*2*3), heldout_image_coordinates=int(heldout.sum()*2*3),
        frames=rows, all_image_fits_accepted=all(row["accepted"] for row in rows),
        selected_independent_refinements_accepted=all(row["accepted"] for row in validation_rows),
        all_images_accepted=all(row["accepted"] for row in rows+validation_rows),
        generating_images_method="49 actual propagated GaussianBeam modes; separately summed intensities; finite-pixel16x16 integration plusMTF halo; no detector-crop normalization.",
        image_initialization_uses_data_only=True, image_variance_weights_use_generating_expectation=False,
        independent_dose_and_centroid_nuisances_marginalized=True,
        optical_covariance_units="[log rms_x_pixels, log rms_y_pixels, atanh correlation]; full marginalized3x3 covariance.",
        specimen_and_apertures="Vacuum path; no masks in these beam images. Existing separate aperture metrology is unchanged.",
        earlier_training_and_heldout_roles_unchanged=True, aperture_family14_remains_held_out=True,
        hardware_admitted=False,
        limitations=["Stronger synthetic CL1/CL3 commands and unchanged astigmatic input preparation require hardware confirmation.",
            "Objective/downstream maps remain conditional calibration; no upstream deflection/tilt calibration or independent angle tag is granted.",
            "Source is one fixed partially coherent Gaussian state; this does not validate stigmator placement or non-Gaussian source components.",
            "An unblocked specimen vacuum path and absent unmodelled bores/apertures are assumed.",
            "Feasible GLS propagates the fitted local pixel information; finite-sample noise-model uncertainty is not separately modelled.",
            "Image adequacy and observable extraction are not physical lens recovery or global branch identification."])
    report_output = json.dumps(report, indent=2, allow_nan=False)+"\n"
    output.write_text(report_output)
    print(str(output), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=914271)
    args = parser.parse_args(); jax.config.update("jax_enable_x64", True)
    acquire(args.design.resolve(), args.output.resolve(), seed=args.seed)


if __name__ == "__main__":
    main()
