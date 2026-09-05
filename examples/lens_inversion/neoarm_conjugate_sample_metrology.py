"""Full cubic-specimen image metrology at verified conjugate imaging states.

The data remain the saved GPU Gaussian-reexpansion images. The independent
image template inverse-maps the SAME frozen SampleView, including all its
registered features, and applies the calibrated finite detector MTF. This
conditional benchmark is not a replacement for a defocused wave propagator.
"""

from pathlib import Path
import argparse
import hashlib
import json
import os
import tomllib
import time

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jnp
import numpy as np

from neoarm_workflow import load_saved_acquisition, expected_detector_counts, frozen_sample_view
from temgym_core.constants import energy2wavelength
from temgym_core.jax_least_squares import least_squares
from temgym_core.multiscale_sample import MAGICALSampleAtlas
from temgym_core.neoarm_calibration import hard_edge_column_from_config
from temgym_core.periodic_metrology import _detector_blur


PARAMETER_NAMES = (
    "log_magnification", "rotation_rad", "sample_translation_x_nm", "sample_translation_y_nm",
    "log_incident_electrons", "log_sample_illumination_waist_m", "log_attenuation_scale",
)
PARAMETER_SCALES = np.array([.01, .01, 1., 1., .02, .02, .1])


def make_conjugate_count_predictor(view, detector, source_centre_xy_m, *,
                                   reference_matrix, wavelength_m, shortest_scale_m):
    """Return a seven-parameter JAX image template, after checking B≈0.

    M, image rotation, physical specimen translation, incident dose, sample
    Gaussian waist and overall log-attenuation scale are fitted. Detector
    affine/MTF, sample identity/internal geometry/tilt/in-plane rotation, and
    the independently calibrated beam centre are fixed conditional inputs.
    The rotation parameter describes A; specimen rotation is not double-fitted.
    """
    matrix = np.asarray(reference_matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("reference_matrix must be a finite 4x4 transfer matrix.")
    a, b = matrix[np.ix_([0, 2], [0, 2])], matrix[np.ix_([0, 2], [1, 3])]
    magnification2 = np.linalg.det(a)
    if magnification2 <= 0 or not np.allclose(a @ a.T / magnification2, np.eye(2), atol=1e-8, rtol=0):
        raise ValueError("This conditional template requires a non-singular similarity position map.")
    if (not np.isfinite(shortest_scale_m) or shortest_scale_m <= 0
            or not np.isfinite(wavelength_m) or wavelength_m <= 0):
        raise ValueError("Wavelength and shortest retained sample scale must be positive.")
    phase_bound = np.pi * wavelength_m * np.linalg.norm(np.linalg.solve(a, b), ord=2) / shortest_scale_m**2
    if phase_bound > 1e-4:
        raise ValueError("State is not conjugate at the retained specimen scale; use wave propagation.")
    source_centre = jnp.asarray(source_centre_xy_m)
    if source_centre.shape != (2,) or not np.all(np.isfinite(source_centre)):
        raise ValueError("A finite, independently calibrated sample Gaussian centre is required.")
    optical_xy = jnp.asarray(detector.optical_coords_xy_m)
    pixel_area = np.prod(detector.pixel_size_yx_m) / abs(np.linalg.det(detector.affine_xy))
    view_centre = jnp.asarray(view.centre_xy_m)

    def predict(parameters):
        log_m, angle, tx_nm, ty_nm, log_dose, log_waist, log_contrast = parameters
        c, s = jnp.cos(angle), jnp.sin(angle)
        rotation = jnp.array([[c, -s], [s, c]])
        specimen_xy = jnp.einsum("ji,...j->...i", rotation, optical_xy) * jnp.exp(-log_m)
        translation = jnp.array([tx_nm, ty_nm]) * 1e-9
        # Calls the public cubic action component, never interpolation of a
        # complex array followed by angle. Unused jets are eliminated by JIT.
        log_amplitude = view.log_transmission(specimen_xy + view_centre - translation)
        waist = jnp.exp(log_waist)
        radius2 = jnp.sum((specimen_xy - source_centre)**2, axis=-1)
        incident = 2 / (jnp.pi * waist**2) * jnp.exp(-2 * radius2 / waist**2)
        image = incident * jnp.exp(2 * log_amplitude * jnp.exp(log_contrast))
        counts = image * jnp.exp(log_dose - 2 * log_m) * pixel_area
        return _detector_blur(counts.reshape(-1, 1), detector.shape_yx, detector.mtf_sigma_px).reshape(detector.shape_yx)

    return jax.jit(predict), float(phase_bound)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_summary(result, truth, variance, expected, predict):
    fitted = np.asarray(result.x)
    jacobian = np.asarray(result.jacobian)
    _, singular, vh = np.linalg.svd(jacobian * PARAMETER_SCALES, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * np.finfo(float).eps * max(jacobian.shape)))
    inverse_factor = PARAMETER_SCALES[:, None] * vh.T / singular
    covariance = inverse_factor @ inverse_factor.T if rank == len(truth) else None
    errors = fitted - truth
    sigma = np.sqrt(np.diag(covariance)) if covariance is not None else np.full(truth.shape, np.nan)
    image_error = (np.asarray(predict(result.x)) - expected) / np.sqrt(variance)
    return {
        "optimizer_success": result.success, "termination": result.status,
        "nfev": result.nfev, "njev": result.njev, "optimality": result.optimality,
        "parameters": fitted.tolist(), "parameter_errors": errors.tolist(),
        "parameter_standard_errors": sigma.tolist() if covariance is not None else None,
        "parameter_errors_sigma": (errors / sigma).tolist() if covariance is not None else None,
        "conditional_covariance": covariance.tolist() if covariance is not None else None,
        "jacobian_rank": rank, "scaled_jacobian_singular_values": singular.tolist(),
        "jacobian_parameter_scales": PARAMETER_SCALES.tolist(),
        "data_residual_rms_sigma": float(np.sqrt(np.mean(np.asarray(result.residual)**2))),
        "expected_image_discrepancy_rms_sigma": float(np.sqrt(np.mean(image_error**2))),
        "magnification_relative_error": float(np.expm1(errors[0])),
        "rotation_error_deg": float(np.rad2deg(errors[1])),
    }


def refined_gaussian_image(run, request, view, column, topology, source_centre, factor, directory):
    """Refine only the beam lattice; preserve its verified original coverage."""
    from temgym_core.neoarm_calibration import (
        reexpand_transmitted_field, propagate_gaussians_through_column, render_gaussian_detector_plane,
    )
    if jax.default_backend() != "gpu":
        raise RuntimeError("Fresh Gaussian refinement requires the GPU; no CPU fallback.")
    side = int(round(np.sqrt(topology["beam_count"])))
    if side**2 != topology["beam_count"] or side % 2 != 1:
        raise ValueError("The archived topology is not the expected odd square lattice.")
    original_axes = [centre + np.arange(-(side//2), side//2+1)*topology["spacing_m"]
                     for centre in source_centre]
    for axis, key in zip(original_axes, ("x_sha256", "y_sha256")):
        if hashlib.sha256(axis.tobytes()).hexdigest() != topology[key]:
            raise ValueError("Original Gaussian grid reconstruction failed its recorded checksum.")
    refined_side = 2 * int(np.ceil((side//2)*np.sqrt(factor))) + 1
    x, y = [np.linspace(axis[0], axis[-1], refined_side) for axis in original_axes]
    local = np.asarray(view.world_to_local(jnp.asarray([[x[0], y[0]], [x[0], y[-1]],
                                                        [x[-1], y[0]], [x[-1], y[-1]]])))
    xmin, xmax, ymin, ymax = view.domain_bounds_xyxy_m
    if not np.all((local[:, 0] >= xmin) & (local[:, 0] <= xmax)
                  & (local[:, 1] >= ymin) & (local[:, 1] <= ymax)):
        raise ValueError("Refined grid exceeds the original frozen cubic view.")
    print(f"GPU refinement {factor:g}×: {refined_side**2:,} beamlets; "
          f"{refined_side**2*np.prod(run.detector.shape_yx):,} beamlet/pixel pairs; "
          f"primary beam arrays ~{refined_side**2*160/2**20:.0f} MiB, workspace additional.", flush=True)
    started = time.perf_counter()
    xx, yy = jnp.meshgrid(jnp.asarray(x), jnp.asarray(y), indexing="xy")
    waist = topology["illumination_waist_m"]
    incident = np.sqrt(2/np.pi)/waist * jnp.exp(-((xx-source_centre[0])**2 + (yy-source_centre[1])**2)/waist**2)
    beam, transmitted = reexpand_transmitted_field(
        incident, view, x, y, voltage=200e3, z_m=column.sample_z_m,
        width_over_step_yx=tuple(run.metadata["width_over_step_yx"]), coefficient_method="gpu_kernel_required",
    )
    transmitted_power = float(jnp.sum(jnp.abs(transmitted)**2) * (x[1]-x[0]) * (y[1]-y[0]))
    downstream = propagate_gaussians_through_column(
        beam, column, np.asarray(run.metadata["parameters"]), np.asarray(request.raw_dac_values),
        segment="sample_to_detector", slices_per_lens=run.metadata["slices_per_lens"],
    )
    rendered = render_gaussian_detector_plane(downstream, run.detector,
                                               tile_rows=16, method="gpu_kernel_required")
    elapsed = time.perf_counter() - started
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"frame_{request.frame_id:08d}_beamcount_{refined_side**2}.npz"
    np.savez_compressed(path, intensity_yx=rendered.intensity_yx, field_yx=rendered.field_yx,
                        x_coords_m=x, y_coords_m=y)
    metadata = {
        "requested_beam_count_factor": factor, "beam_count": refined_side**2,
        "grid_shape_yx": [refined_side, refined_side], "grid_spacing_m": float(x[1]-x[0]),
        "grid_bounds_xyxy_m": [x[0], x[-1], y[0], y[-1]],
        "original_grid_sha256_matches": True, "grid_extent_unchanged": True,
        "cubic_level_unchanged": True, "sample_illumination_waist_m": waist,
        "raw_dac_values": list(request.raw_dac_values), "optical_parameters": run.metadata["parameters"],
        "slices_per_lens": run.metadata["slices_per_lens"],
        "elapsed_s": elapsed, "detector_render_elapsed_s": rendered.elapsed_s,
        "backend": jax.default_backend(), "device": str(jax.devices()[0]),
        "device_kind": jax.devices()[0].device_kind, "jax_version": jax.__version__,
        "x64_enabled": jax.config.x64_enabled, "coefficient_method": "gpu_kernel_required",
        "detector_evaluation": "gpu_kernel_required", "transmitted_grid_power": transmitted_power,
        "artifact_path": str(path), "artifact_sha256": digest(path),
    }
    print(f"  Completed Gaussian refinement in {elapsed:.2f} s", flush=True)
    return rendered.intensity_yx, metadata


def audit(root, refinement_directory=None, image_directory=None):
    root = Path(root)
    config_path = root / "examples/lens_inversion/neoarm_inversion_config.toml"
    sources = [Path(__file__), Path(__file__).with_name("neoarm_workflow.py"), config_path,
               *(root / "src/temgym_core" / filename for filename in (
                   "periodic_metrology.py", "multiscale_sample.py", "jax_least_squares.py",
                   "neoarm_calibration.py", "microscope_inversion.py", "evaluate.py",
                   "gaussian.py", "constants.py",
               ))]
    source_hashes = {str(path.relative_to(root)): digest(path) for path in sources}
    config = tomllib.loads(config_path.read_text())
    run = load_saved_acquisition(
        root / "examples/generated/neoarm_inversion/physical_checkpoint_native_imaging_v5_gpu", config_path,
    )
    print("Constructing the exact recorded atlas and frozen views...", flush=True)
    atlas = MAGICALSampleAtlas.from_config(
        config, region_shape_yx=tuple(config["sample"]["region_shape_yx"]),
        authoritative_multilayer_shape_yx=tuple(config["sample"]["authoritative_multilayer_shape_yx"]),
        global_support_margin_m=config["sample"]["atlas_support_margin_m"],
    )
    column = hard_edge_column_from_config(config)
    wavelength = float(energy2wavelength(config["beam"]["voltage_v"]))
    topologies = {row["label"]: row for row in run.metadata["state_topologies"]}
    rows, dose, read_noise = [], 2e7, .6
    for frame_id in (3, 15):
        request = run.request(frame_id)
        view = frozen_sample_view(atlas, run, request, z_m=column.sample_z_m)
        matrix = np.asarray(column.matrix(run.metadata["parameters"], request.raw_dac_values,
                                          segment="sample_to_detector"))
        position = matrix[np.ix_([0, 2], [0, 2])]
        predict, phase_bound = make_conjugate_count_predictor(
            view, run.detector, atlas.pose_translation_xy_m, reference_matrix=matrix,
            wavelength_m=wavelength, shortest_scale_m=request.selected_spacing_m,
        )
        truth = np.array([
            np.log(np.sqrt(abs(np.linalg.det(position)))), np.arctan2(position[1, 0], position[0, 0]),
            *(np.asarray(request.sample_translation_xy_m) / 1e-9), np.log(dose),
            np.log(topologies[request.optical_state_label]["illumination_waist_m"]), 0.,
        ])
        initial = truth + [np.log(1.03), .02, 2., -1., np.log(.97), np.log(1.04), np.log(.9)]
        half_width = np.array([np.log(1.2), .3, 20., 20., np.log(2.), np.log(1.5), np.log(2.)])
        expected = expected_detector_counts(run.image(frame_id), run.detector, dose)
        variance = expected + read_noise**2
        truth_image = np.asarray(predict(truth))
        truth_error = (truth_image - expected) / np.sqrt(variance)
        print(f"Frame {frame_id}, M={request.target_magnification:g}: cubic/archived-Gaussian truth RMS={np.sqrt(np.mean(truth_error**2)):.6g} sigma", flush=True)
        row = {
            "frame_id": frame_id, "magnification": request.target_magnification,
            "selected_region": request.selected_region_name, "selected_cubic_level": request.sample_level_index,
            "frozen_view_bounds_xyxy_m": view.domain_bounds_xyxy_m,
            "phase_bound_rad_at_shortest_scale": phase_bound,
            "truth_parameters": truth.tolist(), "initial_parameters": initial.tolist(),
            "initialization": "all seven parameters deliberately perturbed from recorded nominal calibration; local calibration test, not blind global branch recovery",
            "truth_image_rms_sigma": float(np.sqrt(np.mean(truth_error**2))),
            "truth_image_max_error_sigma": float(np.max(np.abs(truth_error))),
            "beam_count_in_saved_render": topologies[request.optical_state_label]["beam_count"],
            "frame_record_sha256": digest(run.directory / f"frames/frame_{frame_id:08d}.json"),
        }
        rng = np.random.default_rng(51000 + frame_id)
        noisy = rng.poisson(expected) + rng.normal(0, read_noise, expected.shape)
        for name, observed in (("noiseless", expected), ("noisy", noisy)):
            observation_jax, sigma_jax = jnp.asarray(observed), jnp.sqrt(jnp.asarray(variance))
            residual = lambda p: ((predict(p) - observation_jax) / sigma_jax).ravel()
            fit = least_squares(
                residual, initial, bounds=(truth - half_width, truth + half_width),
                x_scale=PARAMETER_SCALES,
                max_nfev=300, gtol=1e-6, ftol=1e-11, xtol=1e-10,
            )
            row[name] = fit_summary(fit, truth, variance, expected, predict)
            print(f"  {name}: success={fit.success}; residual RMS={row[name]['data_residual_rms_sigma']:.6g} sigma", flush=True)
        row["passed_conditional_truth_audit"] = bool(
            row["truth_image_rms_sigma"] < .2
            and row["noiseless"]["optimizer_success"] and row["noisy"]["optimizer_success"]
            and row["noiseless"]["jacobian_rank"] == 7
            and max(abs(x) for x in row["noiseless"]["parameter_errors_sigma"][:4]) < .2
            and .8 < row["noisy"]["data_residual_rms_sigma"] < 1.2
        )
        if frame_id == 3 and refinement_directory is not None:
            previous_expected, previous_fit = expected, row["noiseless"]
            consecutive, refinements = 0, []
            for factor in (2, 4, 8, 16):
                intensity, metadata = refined_gaussian_image(
                    run, request, view, column, topologies[request.optical_state_label],
                    atlas.pose_translation_xy_m, factor, refinement_directory,
                )
                current_expected = expected_detector_counts(intensity, run.detector, dose)
                current_variance = current_expected + read_noise**2
                current_error = (truth_image-current_expected)/np.sqrt(current_variance)
                metadata["truth_image_rms_sigma"] = float(np.sqrt(np.mean(current_error**2)))
                metadata["image_transition_rms_sigma"] = float(np.sqrt(np.mean(
                    ((current_expected-previous_expected)/np.sqrt(current_variance))**2)))
                current_rng = np.random.default_rng(51000 + frame_id)
                current_noisy = current_rng.poisson(current_expected) + current_rng.normal(0, read_noise, expected.shape)
                for name, observed in (("noiseless", current_expected), ("noisy", current_noisy)):
                    observed_jax, sigma_jax = jnp.asarray(observed), jnp.sqrt(jnp.asarray(current_variance))
                    fit = least_squares(
                        lambda p: ((predict(p)-observed_jax)/sigma_jax).ravel(), initial,
                        bounds=(truth-half_width, truth+half_width), x_scale=PARAMETER_SCALES,
                        max_nfev=300, gtol=1e-6, ftol=1e-11, xtol=1e-10,
                    )
                    metadata[name] = fit_summary(fit, truth, current_variance, current_expected, predict)
                current_fit = metadata["noiseless"]
                changes_sigma = (np.asarray(current_fit["parameters"])-previous_fit["parameters"]) / current_fit["parameter_standard_errors"]
                metadata["parameter_transition_sigma"] = changes_sigma.tolist()
                metadata["passed_conditional_truth_audit"] = bool(
                    metadata["truth_image_rms_sigma"] < .2
                    and current_fit["optimizer_success"] and metadata["noisy"]["optimizer_success"]
                    and max(abs(value) for value in current_fit["parameter_errors_sigma"][:4]) < .2
                    and .8 < metadata["noisy"]["data_residual_rms_sigma"] < 1.2)
                metadata["transition_passed"] = bool(
                    metadata["image_transition_rms_sigma"] < .2
                    and np.max(np.abs(changes_sigma[:4])) < .1)
                consecutive = consecutive+1 if metadata["transition_passed"] else 0
                metadata["successive_converged_transitions"] = consecutive
                refinements.append(metadata)
                previous_expected, previous_fit = current_expected, current_fit
                print(f"  Truth RMS={metadata['truth_image_rms_sigma']:.6g}σ; "
                      f"max geometry bias={max(abs(v) for v in current_fit['parameter_errors_sigma'][:4]):.6g}σ; "
                      f"max geometry change={np.max(np.abs(changes_sigma[:4])):.6g}σ", flush=True)
                if consecutive >= 2 and metadata["passed_conditional_truth_audit"]:
                    break
            row["fresh_refinements"] = refinements
            row["fresh_refinement_accepted"] = bool(consecutive >= 2 and refinements[-1]["passed_conditional_truth_audit"])
        if image_directory is not None:
            image_directory.mkdir(parents=True, exist_ok=True)
            selected_expected = current_expected if "fresh_refinements" in row else expected
            selected_fit = row["fresh_refinements"][-1]["noisy"] if "fresh_refinements" in row else row["noisy"]
            export_rng = np.random.default_rng(51000 + frame_id)
            arrays = {
                "expected_counts_yx": selected_expected,
                "noisy_counts_yx": export_rng.poisson(selected_expected) + export_rng.normal(0, read_noise, expected.shape),
                "fitted_counts_yx": np.asarray(predict(jnp.asarray(selected_fit["parameters"]))),
                "noise_sigma_yx": np.sqrt(selected_expected + read_noise**2),
                "template_truth_counts_yx": truth_image,
                "fitted_parameters": np.asarray(selected_fit["parameters"]),
            }
            image_path = image_directory / f"metrology_{int(request.target_magnification)}x.npz"
            np.savez_compressed(image_path, **arrays)
            row["comparison_arrays"] = {
                "path": str(image_path.relative_to(root) if image_path.is_absolute() else image_path),
                "sha256": digest(image_path), "noise_seed": 51000 + frame_id,
                "dose_electrons": dose, "read_noise_e": read_noise,
                "selected_image_source": "fresh_refinement" if "fresh_refinements" in row else "archived_gaussian_render",
                "selected_beam_count": row["fresh_refinements"][-1]["beam_count"] if "fresh_refinements" in row else row["beam_count_in_saved_render"],
                "array_sha256": {name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                                 for name, value in arrays.items()},
            }
        rows.append(row)
    if any(digest(path) != source_hashes[str(path.relative_to(root))] for path in sources):
        raise RuntimeError("An audited source changed during execution; do not bind results to the changed code.")
    return {
        "schema": "neoarm-conjugate-cubic-sample-metrology-v1", "parameter_names": PARAMETER_NAMES,
        "all_archived_images_passed": all(row["passed_conditional_truth_audit"] for row in rows),
        "all_passed": all(row.get("fresh_refinement_accepted", row["passed_conditional_truth_audit"]) for row in rows),
        "archive_manifest_sha256": run.checkpoint.manifest_sha256,
        "backend": jax.default_backend(), "device": str(jax.devices()[0]),
        "sources": source_hashes, "source_hashes_unchanged_during_execution": True,
        "conditional_assumptions": [
            "Detector affine, pixel pitch, MTF and read-noise calibration are exact.",
            "The recorded cubic object, feature registration and sample tilt/rotation are exact.",
            "Input Gaussian centre is calibrated; waist, dose and contrast are released.",
            "All seven parameters are released jointly; no independent-lens geometry is fitted here.",
            "The no-specimen conjugacy test alone is insufficient: saved Gaussian detector images must pass noise-relative closure.",
            "Coverage at the measured states is checked; full allowed fitting-parameter-box view coverage is NOT yet certified.",
            "Any passing result is limited to these conjugate effective-image calibrations, not all acquisition states or physical lens recovery.",
        ], "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refine-40k", action="store_true")
    args = parser.parse_args()
    report = audit(Path(__file__).resolve().parents[2], args.output.parent if args.refine_40k else None,
                   args.output.parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.refine_40k:
        refinement_report = {
            "schema": "neoarm-frozen-cubic-gaussian-image-metrology-refinement-v1",
            "archive_manifest_sha256": report["archive_manifest_sha256"],
            "sources": report["sources"], "conditional_assumptions": report["conditional_assumptions"],
            "parameter_names": report["parameter_names"],
            "rows": [row for row in report["rows"] if "fresh_refinements" in row],
        }
        args.output.with_name("gaussian_metrology_refinement.json").write_text(
            json.dumps(refinement_report, indent=2, allow_nan=False) + "\n")
    print(f"Conditional conjugate-image audit passed: {report['all_passed']}")
