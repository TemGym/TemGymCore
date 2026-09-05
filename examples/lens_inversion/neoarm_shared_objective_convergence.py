"""GPU convergence evidence for the fixed shared-objective synthetic column.

This is a numerical audit, not an optical recovery fit. The initial 256-slice
conjugate solution is held fixed throughout. Uniform refinement holds q_total;
graded meshes and window extensions preserve the analytic physical field
amplitude using q_new=q_reference*normalization_new/normalization_reference.
An unresolved field-window error must not be absorbed by refitting the optics.
"""

from dataclasses import replace
import argparse
import hashlib
import json
from pathlib import Path
import time
import tomllib

import jax
import jax.numpy as jnp
import numpy as np

from temgym_core.constants import energy2wavelength
from temgym_core.gaussian_image_likelihood import (
    GaussianDetectorOperator, encode_gaussian_image_parameters,
    gaussian_image_information_factor, validate_phase_space_covariance,
)
from temgym_core.immersed_objective import SharedObjectiveField
from temgym_core.neoarm_calibration import DetectorGeometry, hard_edge_column_from_config
from neoarm_imaging_recipe import column_with_geometry_anchors
from neoarm_shared_objective import initialize_shared_imaging, shared_imaging_metrics


ROOT = Path(__file__).resolve().parents[2]
SOURCE_RMS = np.array([5e-9, 50e-6, 6e-9, 45e-6])
SOURCE_COVARIANCE = np.diag(SOURCE_RMS**2)
# Explicitly rounded reconnaissance candidate, not a recovered condenser state.
CONDENSER_Q = np.array([.251875, .28875, .78337526])
SAMPLE_KICKS = np.array([[0., 0., 0., 0.], [5e-9, 0., 0., 0.],
                        [0., 20e-6, 0., 0.], [0., 0., 5e-9, 0.],
                        [0., 0., 0., 20e-6]])
KICK_NAMES = ("reference", "position_x_5nm", "angle_x_20urad",
              "position_y_5nm", "angle_y_20urad")
DOSE_ELECTRONS = 20e6
READ_NOISE_ELECTRONS = .6
REFERENCE_SLICES = 8192


def tangent_objective(n_slices_per_side, *, bounds=(-.05, .05),
                      mesh_centre_m=-.0005, mesh_scale_m=.003):
    """Fixed graded quadrature; its nodes never track fitted field parameters."""
    if (isinstance(n_slices_per_side, bool) or int(n_slices_per_side) != n_slices_per_side
            or n_slices_per_side < 1):
        raise ValueError("Choose a positive integer number of slices per side.")
    if (not np.all(np.isfinite([*bounds, mesh_centre_m, mesh_scale_m]))
            or not bounds[0] < 0. < bounds[1] or mesh_scale_m <= 0):
        raise ValueError("The fixed mesh must contain sample zero and have a positive scale.")

    def segment(a, b):
        angles = np.linspace(np.arctan((a-mesh_centre_m)/mesh_scale_m),
                             np.arctan((b-mesh_centre_m)/mesh_scale_m),
                             int(n_slices_per_side)+1)
        values = mesh_centre_m + mesh_scale_m*np.tan(angles)
        values[0], values[-1] = a, b
        return values

    return SharedObjectiveField(tuple(np.r_[segment(bounds[0], 0.),
                                              segment(0., bounds[1])[1:]]), 0.)


def preserve_profile_amplitude(parameters, field, reference_field):
    """Change only q's quadrature convention, never peak field or shape."""
    p = jnp.asarray(parameters)
    new = field.support_diagnostics(p[-4:])["normalization_length_m"]
    old = reference_field.support_diagnostics(p[-4:])["normalization_length_m"]
    return p.at[-4].multiply(new/old)


def compare_images(candidate, reference, reference_information):
    """Conditional Poisson/read WLS discrepancies, including observable precision.

    Empty background pixels can dilute a pixel-RMS check. The six-coordinate
    information factor also measures centroid/width/correlation/dose shifts.
    Its joint Mahalanobis norm bounds every linearized observable's shift in
    standard errors; none of these gates proves optical-parameter convergence.
    """
    inverse = np.linalg.inv(reference_information)
    sigma = np.sqrt(np.sum(inverse**2, axis=2))
    parameter_delta = candidate["image_parameters"]-reference["image_parameters"]
    standardized = parameter_delta/sigma
    joint = np.linalg.norm(np.einsum("nij,nj->ni", reference_information, parameter_delta), axis=1)
    pixel_delta = ((candidate["images"]-reference["images"])
                   / np.sqrt(reference["images"]+READ_NOISE_ELECTRONS**2))
    pixel_rms = np.sqrt(np.mean(pixel_delta**2, axis=(1, 2)))
    return dict(pixel_noise_rms_each_frame=pixel_rms.tolist(),
        maximum_pixel_noise_rms=float(np.max(pixel_rms)),
        maximum_image_coordinate_shift_sigma=float(np.max(np.abs(standardized))),
        maximum_centroid_shift_sigma=float(np.max(np.abs(standardized[:, :2]))),
        maximum_width_correlation_shift_sigma=float(np.max(np.abs(standardized[:, 2:5]))),
        image_coordinate_mahalanobis_each_frame=joint.tolist(),
        maximum_image_coordinate_mahalanobis=float(np.max(joint)),
        pixel_gate_passed=bool(np.max(pixel_rms) < .2),
        joint_observable_gate_passed=bool(np.max(joint) < .2),
        individual_observable_gate_passed=bool(np.max(np.abs(standardized)) < .2))


def source_hashes():
    paths = [Path(__file__), ROOT/"examples/lens_inversion/neoarm_shared_objective.py",
        ROOT/"examples/lens_inversion/neoarm_imaging_recipe.py",
        ROOT/"examples/lens_inversion/neoarm_measured_initialization.py",
        ROOT/"examples/lens_inversion/neoarm_inversion_config.toml",
        ROOT/"examples/microscope_models/measure_microscope_geometry.ipynb"]
    paths += [ROOT/"src/temgym_core"/name for name in (
        "immersed_objective.py", "microscope_inversion.py", "jax_least_squares.py",
        "neoarm_calibration.py", "gaussian_image_likelihood.py", "constants.py")]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def run_convergence_audit():
    """Run the fixed synthetic probes on GPU; return a JSON-safe evidence report."""
    if jax.default_backend() != "gpu" or not jax.config.x64_enabled:
        raise RuntimeError("Run this forward audit on GPU with JAX float64 enabled.")
    hashes_before = source_hashes()
    started = time.monotonic()
    config = tomllib.loads((ROOT/"examples/lens_inversion/neoarm_inversion_config.toml").read_text())
    base = column_with_geometry_anchors(hard_edge_column_from_config(config),
        ROOT/"examples/microscope_models/measure_microscope_geometry.ipynb")
    initial_field = SharedObjectiveField.from_window(-.05, 0., .05, n_slices_per_side=256)
    initialization = initialize_shared_imaging(base, objective=initial_field)
    parameters = jnp.asarray(initialization.parameters)
    controls = np.array(initialization.nominal_controls)
    for lens, q in zip(initialization.column.lenses[:3], CONDENSER_Q):
        controls[:, lens.channel_index] = (lens.reference_dac
            + (q-lens.q_bias)/lens.q_slope_per_dac)
    detector = DetectorGeometry(mtf_sigma_px=.65)
    render = jax.jit(GaussianDetectorOperator(detector, quadrature_order=8))
    encode = jax.jit(lambda mean, covariance: encode_gaussian_image_parameters(
        mean, covariance, DOSE_ELECTRONS, detector))
    information = jax.jit(lambda p: gaussian_image_information_factor(
        p, detector, read_noise_electrons=READ_NOISE_ELECTRONS, quadrature_order=8))
    xy = jnp.array([0, 2])
    covariance = jnp.asarray(SOURCE_COVARIANCE)
    reference_field = SharedObjectiveField.from_window(-.05, 0., .05,
                                                       n_slices_per_side=REFERENCE_SLICES)
    variants, data = {}, {}

    def evaluate(label, field, amplitude_preserved=False):
        p = (preserve_profile_amplitude(parameters, field, reference_field)
             if amplitude_preserved else parameters)
        column = replace(initialization.column, objective=field)
        down, full = [jax.jit(jax.vmap(lambda u: column.matrix(p, u, segment=segment)))(
            jnp.asarray(controls)) for segment in ("sample_to_detector", "full")]
        metrics = np.asarray(jax.vmap(shared_imaging_metrics)(down))
        images, encoded, spatial_covariances = [], [], []
        for full_matrix, downstream in zip(full, down):
            spatial = (full_matrix@covariance@full_matrix.T)[jnp.ix_(xy, xy)]
            spatial_covariances.append(np.asarray(spatial))
            for kick in SAMPLE_KICKS:
                mean = (downstream@jnp.asarray(kick))[xy]
                images.append(render(mean, spatial, DOSE_ELECTRONS))
                encoded.append(encode(mean, spatial))
        images, encoded = np.asarray(jnp.stack(images)), np.asarray(jnp.stack(encoded))
        data[label] = dict(images=images, image_parameters=encoded)
        edges = np.asarray(field.z_edges_m)
        variants[label] = dict(objective_parameters=np.asarray(p[-4:]).tolist(),
            physical_amplitude_preserved_relative_to_uniform_8192=amplitude_preserved,
            window_m=edges[[0, -1]].tolist(),
            slices_per_side=[field.sample_index, len(edges)-1-field.sample_index],
            quadrature_edges_sha256=hashlib.sha256(edges.astype("<f8").tobytes()).hexdigest(),
            support_diagnostics={key: np.asarray(value).tolist()
                                 for key, value in field.support_diagnostics(p[-4:]).items()},
            sample_k_m_inv=float(field.field(0., p[-4:])),
            mechanical_downstream_matrices=np.asarray(down).tolist(),
            field_free_full_matrices=np.asarray(full).tolist(),
            metrics_M_rotation_rad_full_BA_row_major=metrics.tolist(),
            detector_spatial_covariances_m2=np.asarray(spatial_covariances).tolist(),
            detector_eigen_rms_m=np.sqrt(np.linalg.eigvalsh(spatial_covariances)).tolist(),
            expected_detected_electrons=images.sum(axis=(1, 2)).tolist(),
            image_parameters=encoded.tolist(),
            expected_images_float64_sha256=hashlib.sha256(images.astype("<f8").tobytes()).hexdigest())
        print(f"Rendered {label}: M={metrics[:, 0].tolist()}", flush=True)

    for n in (256, 512, 1024, 2048, 4096, 8192, 16384):
        evaluate(f"uniform_{n}", SharedObjectiveField.from_window(-.05, 0., .05, n_slices_per_side=n))
    for n in (128, 256, 512, 1024):
        evaluate(f"tangent_{n}", tangent_objective(n), True)
    for left, right in ((-.075, .075), (-.075, .15), (-.09, .18)):
        counts = (round(-left/.05*REFERENCE_SLICES), round(right/.05*REFERENCE_SLICES))
        evaluate(f"window_{left}_{right}", SharedObjectiveField.from_window(
            left, 0., right, n_slices_per_side=counts), True)

    reference_label = f"uniform_{REFERENCE_SLICES}"
    reference_information = np.stack([np.asarray(information(p))
                                      for p in data[reference_label]["image_parameters"]])
    comparisons = {}
    reference_metrics = np.asarray(variants[reference_label]["metrics_M_rotation_rad_full_BA_row_major"])
    for label, candidate in data.items():
        comparison = compare_images(candidate, data[reference_label], reference_information)
        metrics = np.asarray(variants[label]["metrics_M_rotation_rad_full_BA_row_major"])
        comparison.update(relative_magnification_change=(metrics[:, 0]/reference_metrics[:, 0]-1).tolist(),
            rotation_change_degrees=np.rad2deg(np.angle(np.exp(1j*(metrics[:, 1]-reference_metrics[:, 1])))).tolist(),
            maximum_full_BA_change_m=float(np.max(np.abs(metrics[:, 2:]-reference_metrics[:, 2:]))))
        comparisons[label] = comparison

    successive = {}
    for n in (4096, 8192):
        finer = f"uniform_{2*n}"
        factor = (reference_information if finer == reference_label else
                  np.stack([np.asarray(information(p)) for p in data[finer]["image_parameters"]]))
        successive[f"uniform_{n}_to_{2*n}"] = compare_images(data[f"uniform_{n}"], data[finer], factor)
    hashes_after = source_hashes()
    if hashes_after != hashes_before:
        raise RuntimeError("Audited source changed during generation; rerun before saving evidence.")
    inverse = np.linalg.inv(reference_information)
    return dict(schema="neoarm-shared-objective-convergence-v1", source_hashes=hashes_after,
        backend=jax.default_backend(), devices=[str(device) for device in jax.devices()],
        jax_version=jax.__version__, float64=True, elapsed_seconds=time.monotonic()-started,
        parameter_names=initialization.column.parameter_names,
        fixed_initial_parameters=initialization.parameters.tolist(),
        fixed_controls=controls.tolist(), initialized_magnifications=initialization.magnifications.tolist(),
        initial_objective_slices_per_side=256, source_mean=np.zeros(4).tolist(),
        source_covariance_m_rad=SOURCE_COVARIANCE.tolist(), rounded_condenser_probe_q=CONDENSER_Q.tolist(),
        source_covariance_validation=validate_phase_space_covariance(SOURCE_COVARIANCE, energy2wavelength(200e3)),
        calibrated_mechanical_sample_states=SAMPLE_KICKS.tolist(), sample_state_names=KICK_NAMES,
        frame_order="40k then 80k; each uses the five sample states in declared order",
        dose_electrons=DOSE_ELECTRONS, read_noise_electrons=READ_NOISE_ELECTRONS,
        detector=dict(shape_yx=detector.shape_yx, pixel_size_yx_m=detector.pixel_size_yx_m,
                      mtf_sigma_px=detector.mtf_sigma_px, affine_xy=detector.affine_xy.tolist()),
        detector_quadrature_order=8, reference_variant=reference_label,
        reference_image_coordinate_sigma=np.sqrt(np.sum(inverse**2, axis=2)).tolist(),
        tangent_mesh=dict(centre_m=-.0005, scale_m=.003, parameter_dependent_nodes=False),
        variants=variants, comparisons_to_reference=comparisons, successive_refinements=successive,
        nominal_probe_two_successive_image_refinements_passed=all(
            row["pixel_gate_passed"] and row["joint_observable_gate_passed"] for row in successive.values()),
        fixed_window_physically_validated=False, complete_parameter_box_validated=False,
        recovered_optical_parameter_convergence_demonstrated=False, hardware_commands_admitted=False,
        assumptions=["No refitting of the optical solution or controls during this audit.",
            "Uniform meshes hold q_total; graded/window variants preserve reference analytic field amplitude.",
            "Expected counts, not a random noisy acquisition; variances are expected Poisson counts plus read noise squared.",
            "Per-frame observable covariance is inferred from all finite detector pixels with dose released.",
            "Pixel and observable gates do not certify optical-parameter recovery or the unmeasured field support.",
            "Extended windows remain within declared non-overlapping hard-edge gaps; real fringe-field overlap is not validated."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    report = run_convergence_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print("Saved", args.output, flush=True)
    for key, result in report["successive_refinements"].items():
        print(key, "observable shift:", result["maximum_image_coordinate_mahalanobis"], "sigma", flush=True)


if __name__ == "__main__":
    main()
