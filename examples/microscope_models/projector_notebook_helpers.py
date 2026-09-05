"""Readable inputs and outputs for the short MAGICAL calibration notebook.

Numerical image analysis and fitting live in the two pipeline modules. This
file names the values passed between stages and keeps plotting out of the
notebook's main function calls.
"""
from dataclasses import dataclass
import html
import json
from pathlib import Path

import jax
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML, display

from finite_projector_pipeline import (
    LENSES, decode_geometry, generate_dataset, fit_column_from_measurements,
    predict_observables,
)
from magical_projector_pipeline import (
    generate_magical_dataset, analyse_magical_arrays, make_magical_image_operator,
)


@dataclass
class Experiment:
    images: np.ndarray
    lens_controls: np.ndarray
    nominal_magnifications: np.ndarray
    image_labels: list[str]
    specimen_template: dict
    specimen_and_detector: dict
    output_directory: Path


@dataclass
class ImageMeasurements:
    """One row per image; covariance uses log scale, radians, and log blur."""
    scale: np.ndarray
    rotation_deg: np.ndarray
    blur_pixels: np.ndarray
    values: np.ndarray
    covariance: np.ndarray
    image_parameters: np.ndarray
    image_error_in_noise_units: np.ndarray


@dataclass
class MicroscopeFit:
    model: dict
    predicted_measurements: np.ndarray
    error_in_noise_units: float
    solver_success: bool
    solver_message: str


def model_to_parameters(model):
    """Convert named values in mm/mrad/degrees to the solver's 19 coordinates."""
    centres = np.array([model["lens_positions_mm"][name] for name in LENSES]) / 1000
    lengths = np.array([model["lens_lengths_mm"][name] for name in LENSES]) / 1000
    detector = model["detector_position_mm"] / 1000
    gaps = np.r_[centres[0] - lengths[0]/2,
                 np.diff(centres) - (lengths[1:] + lengths[:-1])/2,
                 detector - centres[-1] - lengths[-1]/2]
    gains = np.array([model["excitation_gains"][name] for name in LENSES[1:]])
    if np.any(gaps <= 0) or np.any(lengths <= 0) or np.any(gains <= 0) or model["angular_spread_mrad"] <= 0:
        raise ValueError("Starting positions must leave positive gaps; lengths, gain magnitudes, and angular spread must be positive.")
    return np.r_[np.log(gaps), np.log(lengths), model["objective_strength"], np.log(gains),
                 model["beam_curvature_per_m"]/1e4, np.log(model["angular_spread_mrad"]/1000),
                 np.radians(model["sample_rotation_deg"])]


def parameters_to_model(parameters):
    """Give the fitted numbers physical names and units."""
    _, lengths, centres, detector = decode_geometry(parameters)
    return {
        "lens_positions_mm": dict(zip(LENSES, (np.asarray(centres)*1000).tolist())),
        "lens_lengths_mm": dict(zip(LENSES, (np.asarray(lengths)*1000).tolist())),
        "detector_position_mm": float(detector)*1000,
        "objective_strength": float(parameters[11]),
        "excitation_gains": dict(zip(LENSES[1:], np.exp(parameters[12:16]).tolist())),
        "beam_curvature_per_m": float(parameters[16])*1e4,
        "angular_spread_mrad": float(np.exp(parameters[17]))*1000,
        "sample_rotation_deg": float(np.degrees(parameters[18])),
    }


def simulate_experiment(*, magnifications, wobble_fraction, voltage_v,
                        detector_pixel_size_m, detector_shape, output_directory,
                        random_seed=7391):
    """Make MAGICAL images and return the images, controls, and known specimen.

    Each magnification has one reference and two wobbles per intermediate/
    projector lens. All images belong to the same fitting experiment. The true
    optical parameters are saved separately for the final comparison.
    """
    output_directory = Path(output_directory)
    settings_directory = output_directory / "settings"
    generate_dataset(
        directory=settings_directory, magnifications=magnifications,
        wobble_fraction=wobble_fraction, settings_only=True,
        specification={"voltage_v": voltage_v, "pixel_pitch_m": detector_pixel_size_m,
                       "shape_yx": list(detector_shape)},
    )
    specification = generate_magical_dataset(
        base_directory=settings_directory, directory=output_directory, seed=random_seed,
    )
    # The simplified experiment has no split: every supplied image is fitted.
    specification.pop("training_mask", None)
    (output_directory / "acquisition.json").write_text(json.dumps(specification, indent=2))
    known = {key: specification[key] for key in (
        "voltage_v", "pixel_pitch_m", "shape_yx", "feature_sigma_m", "read_noise_std_e", "specimen",
    )}
    return Experiment(
        images=np.load(output_directory / "images.npz")["images"],
        lens_controls=np.asarray(specification["controls"]),
        nominal_magnifications=np.asarray(specification["nominal_magnifications"]),
        image_labels=specification["labels"],
        specimen_template=dict(np.load(output_directory / "specimen_template.npz")),
        specimen_and_detector=known,
        output_directory=output_directory,
    )


def measure_images(*, images, nominal_magnifications, specimen_template, specimen_and_detector):
    """Return scale, rotation, blur, and their uncertainty for every image.

    Nominal magnifications only initialize image registration. No optical
    parameters, lens controls, initial microscope guess, or truth file is read.
    """
    result = analyse_magical_arrays(
        images=images, nominal_magnifications=nominal_magnifications,
        specimen_template=specimen_template, specimen_and_detector=specimen_and_detector,
    )
    values = result["values"]
    return ImageMeasurements(
        scale=np.exp(values[:, 0]), rotation_deg=np.degrees(values[:, 1]),
        blur_pixels=np.exp(values[:, 2]), values=values, covariance=result["covariance"],
        image_parameters=result["image_parameters"],
        image_error_in_noise_units=result["image_residual_rms"],
    )


def fit_microscope(*, lens_controls, image_measurements, specimen_and_detector,
                   starting_guess, position_priors, max_evaluations=10000):
    """Fit all supplied image measurements, and return a named microscope model.

    Supply recorded controls, measured scale/rotation/blur with covariance,
    known specimen and detector values, a named starting model, and position
    priors. The fit reads no files and never receives the generating model.
    Output.model has the same readable keys and units as starting_guess.
    """
    result = fit_column_from_measurements(
        lens_controls=lens_controls, measured_values=image_measurements.values,
        measurement_covariances=image_measurements.covariance,
        specimen_and_detector=specimen_and_detector,
        starting_parameters=model_to_parameters(starting_guess),
        position_priors=position_priors, max_evaluations=max_evaluations,
    )
    if not result.success:
        raise RuntimeError(f"Microscope fit stopped before convergence: {result.message}")
    prediction = jax.jit(jax.vmap(lambda row: predict_observables(result.x, row, specimen_and_detector)))
    return MicroscopeFit(
        model=parameters_to_model(result.x),
        predicted_measurements=np.asarray(prediction(np.asarray(lens_controls))),
        error_in_noise_units=float(np.sqrt(np.mean(result.data_residual**2))),
        solver_success=bool(result.success), solver_message=result.message,
    )


def load_position_priors(path):
    """Read preferred positions, uncertainties, and hard bounds from JSON."""
    return json.loads(Path(path).read_text())


def load_simulated_truth(output_directory):
    """For the final comparison only; this is not an input to either fit."""
    record = json.loads((Path(output_directory) / "scoring_truth.json").read_text())
    return parameters_to_model(np.asarray(record["parameters"]))


def _table(headers, rows):
    def row_html(row, tag):
        return "<tr>" + "".join(f"<{tag}>{html.escape(str(value))}</{tag}>" for value in row) + "</tr>"
    display(HTML("<table>" + row_html(headers, "th") + "".join(row_html(row, "td") for row in rows) + "</table>"))


def show_images(*, images, image_labels, indices):
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), constrained_layout=True)
    for ax, index in zip(axes.ravel(), indices):
        ax.imshow(images[index], origin="lower", cmap="gray")
        ax.set(title=image_labels[index], xlabel="detector pixel", ylabel="detector pixel")
    plt.show()


def show_measurements(*, measurements, image_labels, indices):
    rows = []
    for index in indices:
        rows.append([image_labels[index], f"{measurements.scale[index]:,.1f}",
                     f"{measurements.rotation_deg[index]:.3f}", f"{measurements.blur_pixels[index]:.3f}"])
    _table(["Image", "Measured scale", "Rotation (degrees)", "Blur width (pixels)"], rows)


def show_position_priors(priors):
    rows = []
    for name, mean, sigma, bounds in zip(priors["position_names"], priors["position_mean_mm"],
                                         priors["position_sigma_mm"], priors["position_bounds_mm"]):
        rows.append([name, mean, sigma, f"{bounds[0]} to {bounds[1]}"])
    _table(["Element", "Preferred position (mm)", "Uncertainty (mm)", "Allowed range (mm)"], rows)


def show_image_fit(*, experiment, measurements, fit_result, indices):
    operator = make_magical_image_operator(experiment.specimen_and_detector, experiment.specimen_template)
    fig, axes = plt.subplots(len(indices), 3, figsize=(10, 3.5*len(indices)),
                             squeeze=False, constrained_layout=True)
    for row, index in enumerate(indices):
        image_parameters = measurements.image_parameters[index].copy()
        image_parameters[:3] = fit_result.predicted_measurements[index]
        predicted = np.asarray(operator(image_parameters))
        observed = experiment.images[index]
        minimum, maximum = np.percentile(observed, [1, 99])
        axes[row, 0].imshow(observed, origin="lower", cmap="gray", vmin=minimum, vmax=maximum)
        axes[row, 1].imshow(predicted, origin="lower", cmap="gray", vmin=minimum, vmax=maximum)
        residual = (predicted-observed) / np.sqrt(np.maximum(observed, 1)+experiment.specimen_and_detector["read_noise_std_e"]**2)
        im = axes[row, 2].imshow(residual, origin="lower", cmap="coolwarm", vmin=-4, vmax=4)
        axes[row, 0].set_ylabel(experiment.image_labels[index]+"\ndetector pixel")
        for ax in axes[row]:
            ax.set_xlabel("detector pixel")
    for ax, title in zip(axes[0], ["Simulated image", "Fitted microscope image", "Difference / noise level"]):
        ax.set_title(title)
    fig.colorbar(im, ax=axes[:, 2], shrink=.7)
    plt.show()


def compare_models(*, true_model, fitted_model):
    rows = []
    for name in LENSES:
        true_z, fitted_z = true_model["lens_positions_mm"][name], fitted_model["lens_positions_mm"][name]
        rows.append([name, f"{true_z:.3f}", f"{fitted_z:.3f}", f"{fitted_z-true_z:+.3f}",
                     f"{true_model['lens_lengths_mm'][name]:.3f}", f"{fitted_model['lens_lengths_mm'][name]:.3f}"])
    true_z, fitted_z = true_model["detector_position_mm"], fitted_model["detector_position_mm"]
    rows.append(["Detector", f"{true_z:.3f}", f"{fitted_z:.3f}", f"{fitted_z-true_z:+.3f}", "—", "—"])
    _table(["Element", "True z (mm)", "Fitted z (mm)", "Position error (mm)",
            "True length (mm)", "Fitted length (mm)"], rows)
