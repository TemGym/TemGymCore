"""Calibrated specimen tilts: synthetic images -> image shifts -> column fit.

Pairs use independent exposures and a common arbitrary image offset. The
known drift standard deviation describes independent motion between frames.
Angles are exact in this demonstration and expressed in specimen x/y axes.
"""
from dataclasses import dataclass
import json

import jax
import numpy as np
import matplotlib.pyplot as plt

from finite_projector_pipeline import (
    LENSES, axial_scale_family, predict_observables, predict_tilt_displacements,
)
from magical_projector_pipeline import make_magical_image_operator, analyse_magical_arrays
from projector_notebook_helpers import model_to_parameters, _table


@dataclass
class TiltExperiment:
    images: np.ndarray
    lens_controls: np.ndarray
    input_tilts_rad: np.ndarray
    nominal_magnifications: np.ndarray
    base_image_indices: np.ndarray
    pair_labels: list[str]
    drift_std_pixels: float


def simulate_tilt_images(*, experiment, lens_setting_indices, tilt_mrad,
                         drift_std_pixels=0.003, random_seed=913):
    """Make -/+ x and -/+ y images at each selected existing lens setting.

    Only this generation step reads the hidden microscope. Each pair has a
    shared unknown offset plus independent Gaussian frame drift. The same
    intensity template and angular spread are used at every tilt.
    """
    indices = np.asarray(lens_setting_indices)
    if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in "iu"
            or np.any(indices < 0) or np.any(indices >= len(experiment.images))):
        raise ValueError("Supply valid base-image indices for the tilt settings.")
    if not np.isfinite(tilt_mrad) or tilt_mrad <= 0 or not np.isfinite(drift_std_pixels) or drift_std_pixels < 0:
        raise ValueError("Tilt must be positive and drift standard deviation nonnegative.")
    record = json.loads((experiment.output_directory / "scoring_truth.json").read_text())
    truth = np.asarray(record["parameters"])
    spec = experiment.specimen_and_detector
    operator = make_magical_image_operator(spec, experiment.specimen_template)
    rng = np.random.default_rng(random_seed)
    images, controls, angles, magnifications, base_indices, labels = [], [], [], [], [], []
    for index in indices:
        commands = experiment.lens_controls[index]
        nominal = experiment.nominal_magnifications[index]
        shape_parameters = np.asarray(predict_observables(truth, commands, spec))
        for axis, axis_name in zip(np.eye(2), ("x", "y")):
            offset = rng.normal(0, .3, 2)
            labels.append(f"{experiment.image_labels[index]}, {axis_name} tilt ±{tilt_mrad:g} mrad")
            for sign in (-1, 1):
                angle = sign * tilt_mrad * 1e-3 * axis
                shift = np.asarray(predict_tilt_displacements(
                    truth, commands[None], angle[None], spec["pixel_pitch_m"]))[0]
                position = offset + shift + rng.normal(0, drift_std_pixels, 2)
                p = np.r_[shape_parameters, position, np.log(4e7*(nominal/40000.)**2), np.log(3.)]
                mean = np.asarray(operator(p))
                images.append(rng.poisson(mean).astype(float) + rng.normal(0, spec["read_noise_std_e"], mean.shape))
                controls.append(commands)
                angles.append(angle)
                magnifications.append(nominal)
                base_indices.append(index)
    result = TiltExperiment(np.asarray(images), np.asarray(controls), np.asarray(angles),
                            np.asarray(magnifications), np.asarray(base_indices), labels, drift_std_pixels)
    filename = "tilt_images.npz" if tilt_mrad == 1 else f"tilt_images_{tilt_mrad:g}mrad.npz"
    np.savez_compressed(experiment.output_directory / filename,
                        images=result.images, lens_controls=result.lens_controls,
                        input_tilts_rad=result.input_tilts_rad,
                        nominal_magnifications=result.nominal_magnifications,
                        base_image_indices=result.base_image_indices,
                        pair_labels=result.pair_labels, drift_std_pixels=drift_std_pixels)
    return result


def measure_tilt_images(*, tilt_experiment, specimen_template, specimen_and_detector,
                        reference_measurements):
    """Measure displacements from pixels; no generating parameters are read.

    Base-image registration gives starting values only. Every tilt image is
    independently registered with scale, angle, blur, shift, dose and background
    free. Its shift covariance includes uncertainty in all these quantities.
    Disjoint -/+ pairs give independent displacement errors; common offsets
    cancel. Frame drift adds twice its variance to the pair covariance.
    """
    result = analyse_magical_arrays(
        images=tilt_experiment.images,
        nominal_magnifications=tilt_experiment.nominal_magnifications,
        specimen_template=specimen_template, specimen_and_detector=specimen_and_detector,
        initial_image_parameters=reference_measurements.image_parameters[tilt_experiment.base_image_indices],
    )
    return pair_tilt_measurements(
        lens_controls=tilt_experiment.lens_controls,
        input_tilts_rad=tilt_experiment.input_tilts_rad,
        positions_pixels=result["image_parameters"][:, 3:5],
        position_covariances=result["parameter_covariance"][:, 3:5, 3:5],
        drift_std_pixels=tilt_experiment.drift_std_pixels,
        pair_labels=tilt_experiment.pair_labels,
    )


def pair_tilt_measurements(*, lens_controls, input_tilts_rad, positions_pixels,
                           position_covariances, drift_std_pixels, pair_labels):
    """Subtract disjoint pairs in acquisition order: plus minus minus."""
    controls, angles = np.asarray(lens_controls), np.asarray(input_tilts_rad)
    positions, covariance = np.asarray(positions_pixels), np.asarray(position_covariances)
    n = len(controls)
    if (not n or n % 2 or controls.shape != (n, 4) or angles.shape != (n, 2)
            or positions.shape != (n, 2) or covariance.shape != (n, 2, 2)
            or len(pair_labels) != n//2):
        raise ValueError("Supply two images per tilt pair and matching measurement arrays.")
    if not np.array_equal(controls[::2], controls[1::2]):
        raise ValueError("The projector controls must be fixed within each tilt pair.")
    if not np.isfinite(drift_std_pixels) or drift_std_pixels < 0:
        raise ValueError("Drift standard deviation must be finite and nonnegative.")
    return dict(lens_controls=controls[::2], angle_changes_rad=angles[1::2]-angles[::2],
                displacements_pixels=positions[1::2]-positions[::2],
                covariance_pixels=covariance[1::2]+covariance[::2]+2*drift_std_pixels**2*np.eye(2),
                pair_labels=pair_labels)


def show_tilt_measurements(measurements):
    errors = np.sqrt(np.diagonal(measurements["covariance_pixels"], axis1=1, axis2=2))
    rows = [[label, f"{shift[0]:+.4f} ± {error[0]:.4f}", f"{shift[1]:+.4f} ± {error[1]:.4f}"]
            for label, shift, error in zip(measurements["pair_labels"], measurements["displacements_pixels"], errors)]
    _table(["Lens setting / tilt direction", "Measured dx (pixels)", "Measured dy (pixels)"], rows)


def compare_tilt_fits(*, experiment, image_measurements, tilt_measurements,
                      fit_without_tilts, fit_with_tilts, true_model):
    """Score two independent fits against truth, then test the axial ambiguity.

    The scale scan follows one specified family (all distances scaled and
    illumination curvature/spread rescaled). It is a data-only diagnostic,
    not a confidence interval after refitting all other parameters.
    """
    spec = experiment.specimen_and_detector
    before, after = fit_without_tilts.model, fit_with_tilts.model
    rows = []
    for name in (*LENSES, "Detector"):
        def position(model):
            return model["detector_position_mm"] if name == "Detector" else model["lens_positions_mm"][name]
        z = position(true_model)
        rows.append([name, f"{z:.3f}", f"{position(before):.3f}", f"{position(after):.3f}"])
    _table(["Element", "True z (mm)", "Fit without tilts (mm)", "Fit with tilts (mm)"], rows)

    tilt_controls = tilt_measurements["lens_controls"]
    tilt_angles = tilt_measurements["angle_changes_rad"]
    def shifts(p):
        return predict_tilt_displacements(p, tilt_controls, tilt_angles, spec["pixel_pitch_m"])
    true_shifts = np.asarray(shifts(model_to_parameters(true_model)))
    errors = {}
    for label, model in [("Without tilts", before), ("With tilts", after)]:
        predicted = np.asarray(shifts(model_to_parameters(model)))
        errors[label] = float(np.linalg.norm(predicted-true_shifts)/np.linalg.norm(true_shifts))
        print(f"{label}: relative B error at the measured settings = {100*errors[label]:.2f}%")

    # Evaluate both data costs along exactly the same family through the joint fit.
    p = model_to_parameters(after)
    scales = np.linspace(.9, 1.1, 101)
    family = np.array([axial_scale_family(p, scale) for scale in scales])
    image_whitening = np.linalg.inv(np.linalg.cholesky(image_measurements.covariance))
    tilt_whitening = np.linalg.inv(np.linalg.cholesky(tilt_measurements["covariance_pixels"]))
    predict_images = jax.jit(jax.vmap(lambda params: jax.vmap(
        lambda commands: predict_observables(params, commands, spec))(experiment.lens_controls)))
    difference = np.asarray(predict_images(family))-image_measurements.values
    difference[:, :, 1] = np.arctan2(np.sin(difference[:, :, 1]), np.cos(difference[:, :, 1]))
    image_cost = np.sum(np.einsum("nij,snj->sni", image_whitening, difference)**2, axis=(1, 2))
    shift_difference = np.asarray(jax.jit(jax.vmap(shifts))(family))-tilt_measurements["displacements_pixels"]
    tilt_cost = np.sum(np.einsum("nij,snj->sni", tilt_whitening, shift_difference)**2, axis=(1, 2))
    fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
    ax.plot(scales, image_cost-image_cost.min(), label="Scale, rotation and blur")
    total_cost = image_cost+tilt_cost
    ax.plot(scales, total_cost-total_cost.min(), label="Also measured tilt shifts")
    ax.set(xlabel="Multiplier applied to all axial distances", ylabel="Increase in squared residual / noise²")
    ax.legend()
    plt.show()
    report = dict(relative_b_error_without_tilts=errors["Without tilts"],
                  relative_b_error_with_tilts=errors["With tilts"],
                  image_cost_range_along_scale_family=float(np.ptp(image_cost)),
                  joint_cost_range_along_scale_family=float(np.ptp(total_cost)))
    (experiment.output_directory / "tilt_comparison.json").write_text(json.dumps(report, indent=2))
    return report
