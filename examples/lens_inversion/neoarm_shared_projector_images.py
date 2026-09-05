"""Fresh finite-pixel Gaussian calibration of four shared-column projectors.

Roles and controls come from the immutable design. Physical images use native
canonical maps; fitted centroids/covariances/doses and iterated noise weights
use observed pixels only. Generating expectations are saved separately.
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

from neoarm_aperture_capture import ROOT, XY, ANGLE
from neoarm_condenser_endpoint_calibration import GROUPED, STEPS, PIXEL_M, digest
from neoarm_condenser_extension_design import load_context, FOLDER
from neoarm_condenser_extension_images import fit_data_only_image, validate_roles
from neoarm_last_lens_images import render_pure_counts, signed_response_from_image_fits
from neoarm_mixed_gaussian_validation import MixedGaussianSource
from neoarm_projector_branch_experiment import pure_source, signed_inputs
from neoarm_shared_projector_calibration import SharedProjectorModel
from neoarm_shared_objective_audit import provenance_snapshot, verify_provenance_snapshot
from temgym_core.gaussian_image_likelihood import make_gaussian_detector_operator
from temgym_core.immersed_objective import uniform_field_canonical_matrix, canonical_to_mechanical
from temgym_core.microscope_inversion import drift_matrix
from temgym_core.neoarm_calibration import DetectorGeometry


def projector_body_diagnostics(model, stages, controls, source_covariance, canonical_inputs):
    """Nine points per uniform controlled lens; objective interiors unchecked."""
    native = model.native_parameters(stages)
    def one(control):
        cursor, before = model.objective_exit_m, model.post
        covariances, means = [], []
        for index in range(3, 7):
            centre, log_length, bias, slope = native[2+4*index:6+4*index]
            length = jnp.exp(log_length)
            before = drift_matrix(centre-length/2-cursor)@before
            k = (bias+slope*(control[index]-model.column.lenses[index].reference_dac))/length
            def body(t):
                matrix = canonical_to_mechanical(k)@uniform_field_canonical_matrix(k, t)@before
                cov = matrix@source_covariance@matrix.T
                means = canonical_inputs@matrix.T
                return cov[jnp.ix_(jnp.asarray(ANGLE), jnp.asarray(ANGLE))], means[:, ANGLE]
            c, m = jax.vmap(body)(jnp.linspace(0., length, 9))
            covariances.append(c); means.append(m)
            before = uniform_field_canonical_matrix(k, length)@before
            cursor = centre+length/2
        return jnp.stack(covariances), jnp.stack(means)
    c, m = jax.jit(jax.vmap(one))(jnp.asarray(controls))
    rms = np.sqrt(np.linalg.eigvalsh(np.asarray(c)))
    mean_magnitude = np.linalg.norm(np.asarray(m), axis=-1)
    sample_mechanical = np.asarray(canonical_to_mechanical(model.kappa_per_m))
    sample_cov = sample_mechanical@source_covariance@sample_mechanical.T
    return dict(lens_order=["IL1", "IL2", "IL3", "PL1"], samples_per_lens=9,
        maximum_mechanical_angular_rms_by_lens_rad=rms.max(axis=(0, 2, 3)).tolist(),
        maximum_input_mean_angle_by_lens_rad=mean_magnitude.max(axis=(0, 2, 3)).tolist(),
        maximum_mechanical_angular_rms_rad=float(rms.max()),
        maximum_mean_angle_rad=float(mean_magnitude.max()),
        six_rms_plus_largest_mean_envelope_rad=float(np.max(6*rms.max(axis=-1)+mean_magnitude.max(axis=-1))),
        sample_mechanical_angular_rms_rad=np.sqrt(np.linalg.eigvalsh(sample_cov[np.ix_(ANGLE, ANGLE)])).tolist(),
        objective_interior_angles_checked=False, continuous_z_extrema_certified=False,
        nonlinear_optics_or_hardware_clearance_certified=False,
        scope="Paraxial covariance/mean diagnostics only; Gaussian support is unbounded and six RMS is not a maximum ray angle.")


def acquire(output, *, seed=283149):
    if jax.default_backend() != "gpu" or not jax.config.x64_enabled:
        raise RuntimeError("Image generation and fitting require GPU float64")
    started = perf_counter()
    output = Path(output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    sources, computational = provenance_snapshot()
    design_path = FOLDER/"shared_projector_design.json"
    design = json.loads(design_path.read_text())
    verify_provenance_snapshot(design["source_hashes"])
    controls = np.asarray(design["controls"])
    training, heldout = np.asarray(design["training_mask"]), np.asarray(design["heldout_mask"])
    validate_roles(controls, training, heldout)
    if not design["frozen_before_new_images_and_noise"]:
        raise ValueError("Acquisition roles must precede images and noise")
    _, _, _, _, context = load_context()
    model = SharedProjectorModel(context.column, context.fixed)
    stages = model.stage_coordinates(context.fixed)
    np.testing.assert_array_equal(context.fixed, design["generating_column_parameters"])
    matrices = np.asarray(jax.jit(jax.vmap(lambda u: model.column.matrix(context.fixed, u,
        segment="sample_to_detector", coordinates="canonical")))(controls))
    native_mechanical = np.asarray(jax.jit(jax.vmap(lambda u: model.column.matrix(context.fixed, u,
        segment="sample_to_detector", coordinates="mechanical")))(controls))
    expected_response = native_mechanical[:, XY][:, :, GROUPED]*STEPS/PIXEL_M
    source = pure_source()
    canonical_inputs_grouped = np.asarray(signed_inputs(STEPS))@np.asarray(model.shear).T
    canonical_inputs = canonical_inputs_grouped[:, GROUPED]
    diagnostics = projector_body_diagnostics(model, stages, controls, source.total_covariance, canonical_inputs)
    print("projector angle diagnostics", json.dumps(diagnostics), flush=True)
    detector = DetectorGeometry(z_m=model.detector_z_m, mtf_sigma_px=.65)
    operator = jax.jit(make_gaussian_detector_operator(detector, 16))
    dose, read_noise = float(design["planned_incident_electrons_per_frame"]), .6
    rng = np.random.default_rng(seed)
    arrays = {key: [] for key in ("noisy_counts", "fitted_counts", "fitted_variance", "image_parameters", "image_parameter_covariances")}
    expected, records, refinements = [], [], []
    covariance = matrices[:, XY]@source.total_covariance@matrices[:, XY].swapaxes(-1, -2)
    widths = np.sqrt(np.linalg.eigvalsh(covariance))/PIXEL_M
    refine_states = {int(np.argmin(widths[:, 0])), int(np.argmax(widths[:, 1]))}
    # All native canonical matrices must pass the unchanged renderer guard
    # before producing any frame or drawing acquisition noise.
    omega = np.kron(np.eye(2), np.array([[0., 1.], [-1., 0.]]))
    balance = np.diag([1e-4, 1e4, 1e-4, 1e4])
    errors = balance.T@(matrices.swapaxes(-1, -2)@omega@matrices-omega)@balance
    if np.max(abs(errors)) > 1e-8:
        raise ValueError(f"Native canonical Gaussian guard failed before images: {np.max(abs(errors))}")
    for state, matrix in enumerate(matrices):
        for frame, mean in enumerate(canonical_inputs):
            beam = MixedGaussianSource(source.position_covariance_xy_m2,
                source.curvature_xy_per_m, source.centre_covariance, mean)
            image, error = render_pure_counts(beam, matrix, detector, pixel_order=16, incident_electrons=dose)
            image = np.asarray(image)
            reference = np.asarray(operator(matrix[XY]@mean, covariance[state], dose))
            sigma = np.sqrt(reference+read_noise**2)
            closure = float(np.linalg.norm((image-reference)/sigma))
            if closure >= .2 or widths[state].min() < .35 or image.sum()/dose < .99:
                raise ValueError(f"Image forward/capture contract failed at state{state},frame{frame}")
            observed = rng.poisson(image)+rng.normal(0., read_noise, image.shape)
            fit, variance, metadata = fit_data_only_image(observed, detector)
            if not metadata["accepted"]:
                raise ValueError(f"Data-only image fit failed at state{state},frame{frame}: {metadata}")
            for key, value in zip(arrays, (observed, fit.fitted_counts_yx, variance, fit.parameters, fit.parameter_covariance)):
                arrays[key].append(np.asarray(value))
            expected.append(image)
            records.append(dict(state=state, frame=frame, training=bool(training[state]),
                capture=float(image.sum()/dose), gaussian_reference_closure_l2_sigma=closure,
                balanced_symplectic_error=error, image_fit=metadata))
            if state in refine_states and frame == 5:
                previous = None; rows = []
                for order in (8, 16, 32):
                    value = image if order == 16 else np.asarray(render_pure_counts(beam, matrix,
                        detector, pixel_order=order, incident_electrons=dose)[0])
                    transition = None if previous is None else float(np.linalg.norm((value-previous)/sigma))
                    rows.append(dict(order=order, reference_l2_sigma=float(np.linalg.norm((value-reference)/sigma)),
                        transition_l2_sigma=transition)); previous = value
                if any(row["reference_l2_sigma"] >= .2 for row in rows) or any(row["transition_l2_sigma"] >= .2 for row in rows[1:]):
                    raise ValueError("Selected narrow/wide two-successive pixel refinement failed")
                refinements.append(dict(state=state, frame=frame, refinements=rows))
        print("shared projector state", state, "nine images admitted", flush=True)
    arrays = {name: value.reshape((len(controls), 9)+value.shape[1:])
        for name, values in arrays.items() for value in (np.asarray(values),)}
    pairs = [signed_response_from_image_fits(p, c) for p, c in
        zip(arrays["image_parameters"], arrays["image_parameter_covariances"])]
    response, response_covariance = map(np.asarray, zip(*pairs))
    arrays.update(response=response, response_covariance=response_covariance, native_controls=controls,
        training_mask=training, heldout_mask=heldout, input_steps_grouped=STEPS,
        detector_pixel_m=np.asarray(PIXEL_M))
    path = output.with_suffix(".npz"); np.savez_compressed(path, **arrays)
    truth_path = output.with_name(output.stem+"_truth.npz")
    np.savez_compressed(truth_path, optical_parameters=context.fixed, stage_coordinates=stages,
        expected_response=expected_response, canonical_matrices=matrices,
        expected_counts=np.asarray(expected).reshape(len(controls), 9, *detector.shape_yx),
        source_covariance_canonical=source.total_covariance)
    difference = np.linalg.solve(np.linalg.cholesky(response_covariance),
        (response-expected_response).reshape(len(controls), 8, 1)).reshape(len(controls), 8)
    sources.update(design["source_hashes"])
    for input_path in (Path(__file__), design_path): sources[str(input_path.relative_to(ROOT))] = digest(input_path)
    verify_provenance_snapshot(sources)
    report = dict(schema="neoarm-shared-physical-projector-images-v1", source_hashes=sources,
        computational_input_hashes=computational, backend=jax.default_backend(), x64_enabled=True,
        arrays_path=str(path.relative_to(ROOT)), arrays_sha256=digest(path),
        truth_path=str(truth_path.relative_to(ROOT)), truth_sha256=digest(truth_path),
        design_path=str(design_path.relative_to(ROOT)), design_sha256=digest(design_path),
        frozen_before_noise=True, seed=seed, generated_frame_count=len(controls)*9,
        training_frame_count=int(training.sum()*9), heldout_frame_count=int(heldout.sum()*9),
        training_response_coordinate_count=int(training.sum()*8), heldout_response_coordinate_count=int(heldout.sum()*8),
        all_images_accepted=True, truth_response_likelihood=dict(training_chi_square=float(np.sum(difference[training]**2)),
            heldout_chi_square=float(np.sum(difference[heldout]**2))),
        records=records, selected_pixel_refinements=refinements, angular_diagnostics=diagnostics,
        inference_arrays_contain_generating_values=False, image_fits_use_generating_noise_weights=False,
        old_projector_posterior_or_affine_bridge_used=False, optical_parameters_fitted=False,
        limitations=["Fresh raw Gaussian images in the condenser's exact conditional shared-field column.",
            "Sixteen physical projector coordinates remain to be fitted; objective postfield/shear/detector z are conditional.",
            "Mechanical body-angle diagnostics do not certify nonlinear transport or aperture/hardware clearance.",
            "Pure sample-plane source and exact input/pixel/PSF calibration are conditional; source covariance is not used by the centroid likelihood."],
        elapsed_seconds=perf_counter()-started)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); jax.config.update("jax_enable_x64", True)
    acquire(args.output)
