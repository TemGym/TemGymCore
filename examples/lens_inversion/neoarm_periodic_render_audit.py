"""Audit registered multilayer metrology on saved GPU Gaussian specimen images.

This does not redraw a sinusoid, rerender the specimen, fit microscope optics,
or replace archive checksums. The known synthetic column is used only for the
truth audit, an independent no-specimen Gaussian illumination prediction, and
freezing the physical ROI/frequency registration. The frequency fit itself has
no truth penalty. Run with JAX_PLATFORMS=cpu to avoid contending with GPU renders.
"""

from pathlib import Path
import argparse
import hashlib
import json
import os
import tomllib

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np

from neoarm_workflow import load_saved_acquisition, expected_detector_counts
from temgym_core.constants import energy2wavelength
from temgym_core.multiscale_sample import MAGICALSampleAtlas, _sample_projection_numpy
from temgym_core.neoarm_calibration import hard_edge_column_from_config
from temgym_core.periodic_metrology import fit_periodic_frequency


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(root):
    root = Path(root)
    config_path = root / "examples/lens_inversion/neoarm_inversion_config.toml"
    config = tomllib.loads(config_path.read_text())
    run = load_saved_acquisition(
        root / "examples/generated/neoarm_inversion/physical_checkpoint_native_imaging_v5_gpu",
        config_path,
    )
    column = hard_edge_column_from_config(config)
    # Only certificate coordinates/pose are consumed; no raster from this
    # deliberately tiny metadata-only atlas enters any image or likelihood.
    atlas = MAGICALSampleAtlas.from_config(
        config, shape_yx=(16, 16), region_shape_yx=(16, 16),
        authoritative_multilayer_shape_yx=(32, 32), max_levels=1,
    )
    projection, _ = _sample_projection_numpy(atlas.pose_rotation_rad, atlas.pose_tilt_xy_rad)
    detector = run.detector
    optical_xy = np.asarray(detector.optical_coords_xy_m)
    dose, read_noise = 2e7, .6
    period = atlas.certificate.multilayer_period_m
    wavelength = float(energy2wavelength(config["beam"]["voltage_v"]))
    topologies = {row["label"]: row for row in run.metadata["state_topologies"]}
    rows = []
    for frame_id in (3, 15):
        request = run.request(frame_id)
        matrix = np.asarray(column.matrix(run.metadata["parameters"], request.raw_dac_values,
                                           segment="sample_to_detector"))
        position = matrix[np.ix_([0, 2], [0, 2])]
        sample_xy = np.einsum("ij,...j->...i", np.linalg.inv(position), optical_xy)
        local_xy = np.einsum("ij,...j->...i", np.linalg.inv(projection),
                             sample_xy - request.sample_translation_xy_m)
        # The supplied broad sample illumination is not the archive's narrow
        # 3 nm perforation probe, so transporting that unrelated probe would
        # be an invalid flat-field reference. Propagate this image's recorded
        # waist analytically with no specimen instead.
        waist = topologies[request.optical_state_label]["illumination_waist_m"]
        sample_covariance = np.diag([waist**2 / 4, (wavelength / (2*np.pi*waist))**2] * 2)
        sample_mean = np.array([atlas.pose_translation_xy_m[0], 0., atlas.pose_translation_xy_m[1], 0.])
        response = matrix[[0, 2], :]
        covariance = response @ sample_covariance @ response.T
        delta = optical_xy - response @ sample_mean
        illumination = np.exp(-.5 * np.einsum("...i,ij,...j->...", delta, np.linalg.inv(covariance), delta))
        intensity = run.image(frame_id)
        expected_counts = expected_detector_counts(intensity, detector, dose)
        local_per_pixel = np.linalg.inv(projection) @ np.linalg.inv(position) @ np.linalg.inv(detector.affine_xy) @ np.diag(detector.pixel_size_yx_m[::-1])
        frequency = local_per_pixel[0] / period
        margin = 4 * atlas.certificate.interface_width_m + (
            4 * detector.mtf_sigma_px * np.max(np.linalg.svd(local_per_pixel, compute_uv=False))
        )
        candidates = []
        for region in atlas.certificate.regions:
            if region.kind != "multilayer":
                continue
            if request.selected_region_name is not None and region.name != request.selected_region_name:
                continue
            within = np.all(np.abs(local_xy - region.centre_xy_m)
                            < np.asarray(region.size_xy_m) / 2 - margin, axis=-1)
            # Freeze using certificate contrast ranking and no-specimen
            # illumination, not fitted residuals or the noisy observation.
            score = np.sum(illumination[within]) * region.layer_count**2
            candidates.append((score, region, within))
        _, region, valid = max(candidates, key=lambda item: item[0])
        y, x = np.nonzero(valid)
        halo = int(4 * detector.mtf_sigma_px + .5) + 1
        crop = (slice(max(0, y.min()-halo), min(intensity.shape[0], y.max()+halo+1)),
                slice(max(0, x.min()-halo), min(intensity.shape[1], x.max()+halo+1)))
        expected, valid, envelope = expected_counts[crop], valid[crop], illumination[crop]
        variance = expected + read_noise**2
        rng = np.random.default_rng(51000 + frame_id)
        # Draw the full detector first, preserving the published image seed.
        observed_full = rng.poisson(expected_counts) + rng.normal(0, read_noise, intensity.shape)
        row = {
            "frame_id": frame_id, "magnification": request.target_magnification,
            "region": region.name, "crop_yx_start_stop": [[int(s.start), int(s.stop)] for s in crop],
            "valid_pixel_count": int(valid.sum()), "margin_from_region_boundary_m": float(margin),
            "roi_rule": "maximise no-specimen illumination times certificate layer-count squared; exclude four interface widths plus four detector-PSF sigmas",
            "sample_period_m": period, "frequency_truth_xy_per_pixel": frequency.tolist(),
            "registration_search_radius_bins": .75,
            "sample_illumination_waist_m": waist,
            "illumination_reference": "independent analytic no-specimen transport of recorded broad sample Gaussian, before detector MTF",
            "sample_to_detector_A": position.tolist(),
            "sample_to_detector_B_norm_m": float(np.linalg.norm(matrix[np.ix_([0, 2], [1, 3])])),
            "retained_harmonics": 3, "beam_count_in_saved_render": topologies[request.optical_state_label]["beam_count"],
            "frame_record_sha256": digest(run.directory / f"frames/frame_{frame_id:08d}.json"),
        }
        for name, image in (("noiseless", expected), ("noisy", observed_full[crop])):
            try:
                result = fit_periodic_frequency(
                    image, frequency, variance, illumination_yx=envelope, valid_mask_yx=valid,
                    mtf_sigma_px=detector.mtf_sigma_px, search_radius_bins=.75,
                )
                difference = result.frequency_xy_per_pixel - frequency
                whitened = np.linalg.solve(np.linalg.cholesky(result.covariance_xy_per_pixel2), difference)
                row[name] = {
                    "accepted_by_estimator": result.accepted, "reasons": result.reasons,
                    "frequency_xy_per_pixel": result.frequency_xy_per_pixel.tolist(),
                    "conditional_covariance_xy_per_pixel2": result.covariance_xy_per_pixel2.tolist(),
                    "frequency_error_xy_per_pixel": difference.tolist(),
                    "frequency_error_whitened_rms_sigma": float(np.sqrt(np.mean(whitened**2))),
                    "frequency_relative_scale_error": float(np.linalg.norm(result.frequency_xy_per_pixel)/np.linalg.norm(frequency)-1),
                    "detector_residual_rms_sigma": float(np.sqrt(result.chi_squared/valid.sum())),
                    "reduced_chi_squared": result.reduced_chi_squared,
                    "fundamental_delta_chi_squared": result.fundamental_delta_chi_squared,
                    "fft_candidates_xy_per_pixel": result.fft_candidates_xy_per_pixel.tolist(),
                }
            except ValueError as exc:
                row[name] = {"accepted_by_estimator": False, "reasons": [str(exc)]}
        truth = row["noiseless"]
        row["passed_noise_relative_truth_audit"] = bool(
            truth["accepted_by_estimator"] and row["noisy"]["accepted_by_estimator"]
            and truth["frequency_error_whitened_rms_sigma"] < .2
            and truth["detector_residual_rms_sigma"] < .2
        )
        rows.append(row)
    sources = [Path(__file__), root / "src/temgym_core/periodic_metrology.py",
               root / "src/temgym_core/multiscale_sample.py", config_path]
    return {
        "schema": "neoarm-saved-gaussian-periodic-render-audit-v1",
        "synthetic_only": True, "all_passed": all(row["passed_noise_relative_truth_audit"] for row in rows),
        "archive_manifest_sha256": run.checkpoint.manifest_sha256,
        "archive_integrity_checked": True, "render_backend_recorded": run.metadata["jax_backend"],
        "dose_electrons": dose, "read_noise_e": read_noise,
        "sources": {str(path.relative_to(root)): digest(path) for path in sources},
        "caveats": ["Conditional on exact illumination, calibration and certificate; not an instrument likelihood.",
                    "Passing harmonic fixtures alone does not validate Gaussian-rendered specimen images.",
                    "No new Gaussian forward simulation or microscope inversion was run by this audit."],
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = audit(Path(__file__).resolve().parents[2])
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"all_passed": report["all_passed"], "rows": report["rows"]}, indent=2))
