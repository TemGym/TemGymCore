"""Projector image records, paired measurements and intensity initialization."""

import functools
import hashlib
import json
import operator
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from temgym_core.evaluate import gaussian_intensity_image
from temgym_core.optimization import least_squares
from .gaussian import make_magical_sample

from .projector_model import (
    LENSES,
    decode_geometry,
    matrix5,
    predict_observables,
    predict_tilt_displacements,
)


def make_magical_template(
    voltage=200000.0, pitch_m=2.5e-9, *, illumination_radius_m=160e-9
):
    """Same fixed nominal specimen as magical_gaussian_imaging.ipynb.

    The ROI is fixed at a multilayer edge so the 600k view is not the central
    perforation or an effectively one-dimensional periodic stripe pattern.
    A soft circular illumination envelope is part of the public template.
    """
    if not np.isfinite(illumination_radius_m) or illumination_radius_m <= 0:
        raise ValueError("Illumination radius must be finite and positive.")
    sample, dimensions, roi = make_magical_sample(voltage)
    n = int(np.ceil(2 * (illumination_radius_m + 30e-9) / pitch_m))
    axis = (np.arange(n) - (n - 1) / 2) * pitch_m
    xx, yy = np.meshgrid(axis, axis)
    coordinates = np.stack((xx, yy), axis=-1)
    actions = np.asarray(sample.evaluate_actions(jnp.asarray(coordinates + roi)))
    intensity = np.exp(2 * actions[..., 1])
    radius = np.hypot(xx, yy)
    illumination = 1 / (1 + np.exp((radius - illumination_radius_m) / 2.5e-9))
    included = radius <= illumination_radius_m + 25e-9
    weights = (intensity * illumination)[included] * pitch_m**2
    weights /= weights.sum()
    metadata = {
        "source": dimensions["source"],
        "phase_contrast": False,
        "roi_centre_atlas_m": roi.tolist(),
        "silicon_width_m": dimensions["silicon_width_m"],
        "sige_width_m": dimensions["sige_width_m"],
        "repeat_m": dimensions["silicon_width_m"]
        + dimensions["sige_width_m"],
        "template_pitch_m": pitch_m,
        "gaussian_component_sigma_m": 2e-9,
        "illumination_radius_m": float(illumination_radius_m),
        "illumination_edge_m": 2.5e-9,
        "description": "Known clean atlas intensity and illumination represented by a positive 2 nm-sigma Gaussian mixture; no background texture or fiducial spots",
    }
    return {
        "centres_m": coordinates[included],
        "weights": weights,
        "axis_m": axis,
        "intensity": intensity,
        "illumination": illumination,
    }, metadata


def make_magical_image_operator(spec, template):
    """Seven fitted quantities: log scale, angle, log width_px, dx, dy, log dose, log background."""
    centres = jnp.asarray(template["centres_m"]) / spec["pixel_pitch_m"]
    weights = jnp.asarray(template["weights"])

    def image(p):
        c, s = jnp.cos(p[1]), jnp.sin(p[1])
        rotation = jnp.array([[c, -s], [s, c]])
        xy = jnp.exp(p[0]) * (centres @ rotation.T) + p[3:5]
        return gaussian_intensity_image(
            xy, jnp.exp(2 * p[2]), weights, jnp.exp(p[5]), jnp.exp(p[6]),
            shape_yx=tuple(spec["shape_yx"]),
        )

    return jax.jit(image)


def analyse_magical_arrays(
    *,
    images,
    nominal_magnifications,
    specimen_template,
    specimen_and_detector,
    verbose=False,
    initial_image_parameters=None,
    require_noise_level=True,
):
    """Measure scale, rotation, and blur from the supplied arrays; read no files."""
    spec, template = specimen_and_detector, specimen_template
    if len(images) != len(nominal_magnifications):
        raise ValueError("Supply one nominal magnification for each image.")
    if initial_image_parameters is not None and np.shape(initial_image_parameters) != (
        len(images),
        7,
    ):
        raise ValueError("Supply seven image-registration starting values per image.")
    operator = make_magical_image_operator(spec, template)
    jacobian = jax.jit(jax.jacfwd(operator))
    measured, covariance, costs, image_parameters, parameter_covariance = (
        [],
        [],
        [],
        [],
        [],
    )
    references = {}

    def residual(p, image, sigma):
        return ((operator(p) - image) / sigma).ravel()

    def jac(p, image, sigma):
        return jacobian(p).reshape(-1, 7) / sigma.ravel()[:, None]

    for index, (image, nominal) in enumerate(zip(images, nominal_magnifications)):
        sigma = np.sqrt(np.maximum(image, 1.0) + spec["read_noise_std_e"] ** 2)
        lower = np.r_[
            np.log(nominal / 2),
            -4 * np.pi,
            np.log(0.15),
            -20.0,
            -20.0,
            np.log(1e4),
            np.log(0.01),
        ]
        upper = np.r_[
            np.log(nominal * 2),
            4 * np.pi,
            np.log(60.0),
            20.0,
            20.0,
            np.log(1e13),
            np.log(1e5),
        ]

        if initial_image_parameters is not None:
            seeds = [
                np.clip(initial_image_parameters[index], lower + 1e-8, upper - 1e-8)
            ]
        elif nominal in references:
            seeds = [references[nominal].copy()]
        else:
            seeds = []
            for angle in np.linspace(-np.pi, np.pi, 72, endpoint=False):
                p = np.r_[
                    np.log(nominal),
                    angle,
                    np.log(nominal * spec["feature_sigma_m"] / spec["pixel_pitch_m"]),
                    0.0,
                    0.0,
                    0.0,
                    np.log(0.01),
                ]
                template_image = np.maximum(np.asarray(operator(p)) - 0.01, 0.0)
                design = (
                    np.column_stack((template_image.ravel(), np.ones(image.size)))
                    / sigma.ravel()[:, None]
                )
                coefficients = np.linalg.lstsq(
                    design, image.ravel() / sigma.ravel(), rcond=None
                )[0]
                p[5:] = np.log(np.maximum(coefficients, [1e4, 0.011]))
                p = np.clip(p, lower + 1e-8, upper - 1e-8)
                seeds.append(
                    (float(np.mean(np.asarray(residual(p, image, sigma)) ** 2)), p)
                )
            seeds = [item[1] for item in sorted(seeds, key=lambda item: item[0])[:2]]
        results = [
            least_squares(
                residual,
                seed,
                jac=jac,
                args=(jnp.asarray(image), jnp.asarray(sigma)),
                bounds=(lower, upper),
                x_scale="jac",
                max_nfev=150,
                max_iter=300,
                ftol=1e-9,
                xtol=1e-9,
                gtol=1e-5,
            )
            for seed in seeds
        ]
        result = min(results, key=lambda r: np.sum(r.residual**2))
        rms = np.sqrt(np.mean(result.residual**2))
        if require_noise_level and (not result.success or rms > 1.5):
            raise RuntimeError(
                f"MAGICAL registration failed at frame {index}: {result.message}; RMS {rms:.3f}"
            )
        if nominal not in references:
            references[nominal] = np.asarray(result.x).copy()
        measured.append(result.x[:3])
        full_covariance = np.linalg.inv(
            np.asarray(result.jacobian).T @ np.asarray(result.jacobian)
        )
        covariance.append(full_covariance[:3, :3])
        parameter_covariance.append(full_covariance)
        costs.append(rms)
        image_parameters.append(result.x)
        if verbose:
            print(
                f"MAGICAL frame {index + 1}/{len(images)}: weighted pixel RMS {rms:.3f}",
                flush=True,
            )
    return {
        "values": np.asarray(measured),
        "covariance": np.asarray(covariance),
        "image_residual_rms": np.asarray(costs),
        "image_parameters": np.asarray(image_parameters),
        "parameter_covariance": np.asarray(parameter_covariance),
    }


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


def model_to_parameters(model):
    """Convert named values in mm/mrad/degrees to the solver's 19 coordinates."""
    centres = np.array([model["lens_positions_mm"][name] for name in LENSES]) / 1000
    lengths = np.array([model["lens_lengths_mm"][name] for name in LENSES]) / 1000
    detector = model["detector_position_mm"] / 1000
    gaps = np.r_[
        centres[0] - lengths[0] / 2,
        np.diff(centres) - (lengths[1:] + lengths[:-1]) / 2,
        detector - centres[-1] - lengths[-1] / 2,
    ]
    gains = np.array([model["excitation_gains"][name] for name in LENSES[1:]])
    if (
        np.any(gaps <= 0)
        or np.any(lengths <= 0)
        or np.any(gains <= 0)
        or model["angular_spread_mrad"] <= 0
    ):
        raise ValueError(
            "Starting positions must leave positive gaps; lengths, gain magnitudes, and angular spread must be positive."
        )
    return np.r_[
        np.log(gaps),
        np.log(lengths),
        model["objective_strength"],
        np.log(gains),
        model["beam_curvature_per_m"] / 1e4,
        np.log(model["angular_spread_mrad"] / 1000),
        np.radians(model["sample_rotation_deg"]),
    ]


def parameters_to_model(parameters):
    """Give the fitted numbers physical names and units."""
    _, lengths, centres, detector = decode_geometry(parameters)
    return {
        "lens_positions_mm": dict(zip(LENSES, (np.asarray(centres) * 1000).tolist())),
        "lens_lengths_mm": dict(zip(LENSES, (np.asarray(lengths) * 1000).tolist())),
        "detector_position_mm": float(detector) * 1000,
        "objective_strength": float(parameters[11]),
        "excitation_gains": dict(zip(LENSES[1:], np.exp(parameters[12:16]).tolist())),
        "beam_curvature_per_m": float(parameters[16]) * 1e4,
        "angular_spread_mrad": float(np.exp(parameters[17])) * 1000,
        "sample_rotation_deg": float(np.degrees(parameters[18])),
    }


def measure_images(
    *,
    images,
    nominal_magnifications,
    specimen_template,
    specimen_and_detector,
    initial_image_parameters=None,
):
    """Return scale, rotation, blur, and their uncertainty for every image.

    Nominal magnifications only initialize image registration. No optical
    parameters, lens controls, initial microscope guess, or truth file is read.
    """
    result = analyse_magical_arrays(
        images=images,
        nominal_magnifications=nominal_magnifications,
        specimen_template=specimen_template,
        specimen_and_detector=specimen_and_detector,
        initial_image_parameters=initial_image_parameters,
    )
    values = result["values"]
    return ImageMeasurements(
        scale=np.exp(values[:, 0]),
        rotation_deg=np.degrees(values[:, 1]),
        blur_pixels=np.exp(values[:, 2]),
        values=values,
        covariance=result["covariance"],
        image_parameters=result["image_parameters"],
        image_error_in_noise_units=result["image_residual_rms"],
    )


@dataclass
class TiltExperiment:
    images: np.ndarray
    lens_controls: np.ndarray
    input_tilts_rad: np.ndarray
    nominal_magnifications: np.ndarray
    base_image_indices: np.ndarray
    pair_labels: list[str]
    drift_std_pixels: float


def simulate_tilt_images(
    *,
    experiment,
    lens_setting_indices,
    tilt_mrad,
    drift_std_pixels=0.003,
    random_seed=913,
):
    """Make -/+ x and -/+ y images at each selected existing lens setting.

    Only this generation step reads the hidden microscope. Each pair has a
    shared unknown offset plus independent Gaussian frame drift. The same
    intensity template and angular spread are used at every tilt.
    """
    indices = np.asarray(lens_setting_indices)
    if (
        indices.ndim != 1
        or not len(indices)
        or indices.dtype.kind not in "iu"
        or np.any(indices < 0)
        or np.any(indices >= len(experiment.images))
    ):
        raise ValueError("Supply valid base-image indices for the tilt settings.")
    if (
        not np.isfinite(tilt_mrad)
        or tilt_mrad <= 0
        or not np.isfinite(drift_std_pixels)
        or drift_std_pixels < 0
    ):
        raise ValueError(
            "Tilt must be positive and drift standard deviation nonnegative."
        )
    record = json.loads(
        (experiment.output_directory / "scoring_truth.json").read_text()
    )
    truth = np.asarray(record["parameters"])
    spec = experiment.specimen_and_detector
    operator = make_magical_image_operator(spec, experiment.specimen_template)
    rng = np.random.default_rng(random_seed)
    images, controls, angles, magnifications, base_indices, labels = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for index in indices:
        commands = experiment.lens_controls[index]
        nominal = experiment.nominal_magnifications[index]
        shape_parameters = np.asarray(predict_observables(truth, commands, spec))
        for axis, axis_name in zip(np.eye(2), ("x", "y")):
            offset = rng.normal(0, 0.3, 2)
            labels.append(
                f"{experiment.image_labels[index]}, {axis_name} tilt ±{tilt_mrad:g} mrad"
            )
            for sign in (-1, 1):
                angle = sign * tilt_mrad * 1e-3 * axis
                shift = np.asarray(
                    predict_tilt_displacements(
                        truth, commands[None], angle[None], spec["pixel_pitch_m"]
                    )
                )[0]
                position = offset + shift + rng.normal(0, drift_std_pixels, 2)
                p = np.r_[
                    shape_parameters,
                    position,
                    np.log(4e7 * (nominal / 40000.0) ** 2),
                    np.log(3.0),
                ]
                mean = np.asarray(operator(p))
                images.append(
                    rng.poisson(mean).astype(float)
                    + rng.normal(0, spec["read_noise_std_e"], mean.shape)
                )
                controls.append(commands)
                angles.append(angle)
                magnifications.append(nominal)
                base_indices.append(index)
    result = TiltExperiment(
        np.asarray(images),
        np.asarray(controls),
        np.asarray(angles),
        np.asarray(magnifications),
        np.asarray(base_indices),
        labels,
        drift_std_pixels,
    )
    filename = (
        "tilt_images.npz" if tilt_mrad == 1 else f"tilt_images_{tilt_mrad:g}mrad.npz"
    )
    np.savez_compressed(
        experiment.output_directory / filename,
        images=result.images,
        lens_controls=result.lens_controls,
        input_tilts_rad=result.input_tilts_rad,
        nominal_magnifications=result.nominal_magnifications,
        base_image_indices=result.base_image_indices,
        pair_labels=result.pair_labels,
        drift_std_pixels=drift_std_pixels,
    )
    return result


def measure_tilt_images(
    *, tilt_experiment, specimen_template, specimen_and_detector, reference_measurements
):
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
        specimen_template=specimen_template,
        specimen_and_detector=specimen_and_detector,
        initial_image_parameters=reference_measurements.image_parameters[
            tilt_experiment.base_image_indices
        ],
    )
    return pair_tilt_measurements(
        lens_controls=tilt_experiment.lens_controls,
        input_tilts_rad=tilt_experiment.input_tilts_rad,
        positions_pixels=result["image_parameters"][:, 3:5],
        position_covariances=result["parameter_covariance"][:, 3:5, 3:5],
        drift_std_pixels=tilt_experiment.drift_std_pixels,
        pair_labels=tilt_experiment.pair_labels,
    )


def pair_tilt_measurements(
    *,
    lens_controls,
    input_tilts_rad,
    positions_pixels,
    position_covariances,
    drift_std_pixels,
    pair_labels,
):
    """Subtract disjoint pairs in acquisition order: plus minus minus."""
    controls, angles = np.asarray(lens_controls), np.asarray(input_tilts_rad)
    positions, covariance = (
        np.asarray(positions_pixels),
        np.asarray(position_covariances),
    )
    n = len(controls)
    if (
        not n
        or n % 2
        or controls.ndim != 2
        or controls.shape[1] not in (4, 5)
        or angles.shape != (n, 2)
        or positions.shape != (n, 2)
        or covariance.shape != (n, 2, 2)
        or len(pair_labels) != n // 2
    ):
        raise ValueError(
            "Supply two images per tilt pair and matching measurement arrays."
        )
    if not np.array_equal(controls[::2], controls[1::2]):
        raise ValueError("The projector controls must be fixed within each tilt pair.")
    if not np.isfinite(drift_std_pixels) or drift_std_pixels < 0:
        raise ValueError("Drift standard deviation must be finite and nonnegative.")
    return dict(
        lens_controls=controls[::2],
        angle_changes_rad=angles[1::2] - angles[::2],
        displacements_pixels=positions[1::2] - positions[::2],
        covariance_pixels=covariance[1::2]
        + covariance[::2]
        + 2 * drift_std_pixels**2 * np.eye(2),
        pair_labels=pair_labels,
    )


@dataclass
class CalibrationMeasurements:
    controls: np.ndarray
    values: np.ndarray
    covariance: np.ndarray
    tilts: dict
    shifts: dict

    def save(self, path):
        np.savez_compressed(
            path,
            controls=self.controls,
            values=self.values,
            covariance=self.covariance,
            **{f"tilt_{k}": v for k, v in self.tilts.items() if k != "image_analysis"},
            **{
                f"shift_{k}": v for k, v in self.shifts.items() if k != "image_analysis"
            },
        )

    @classmethod
    def load(cls, path):
        data = dict(np.load(path))

        def pairs(prefix):
            result = {
                k[len(prefix) :]: v for k, v in data.items() if k.startswith(prefix)
            }
            result["pair_labels"] = result["pair_labels"].tolist()
            return result

        return cls(
            data["controls"],
            data["values"],
            data["covariance"],
            pairs("tilt_"),
            pairs("shift_"),
        )

    def append(self, other):
        return CalibrationMeasurements(
            np.concatenate([self.controls, other.controls]),
            np.concatenate([self.values, other.values]),
            np.concatenate([self.covariance, other.covariance]),
            combine_tilt_measurements(self.tilts, other.tilts),
            combine_beam_shift_measurements(self.shifts, other.shifts),
        )


def simulate_setting_images(
    *,
    source_experiment,
    lens_controls,
    nominal_magnifications,
    image_labels,
    output_directory,
    random_seed,
):
    """Ask the simulated instrument for independent images at recorded commands."""
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    record = (source_experiment.output_directory / "scoring_truth.json").read_text()
    truth = np.asarray(json.loads(record)["parameters"])
    spec = source_experiment.specimen_and_detector
    operator = make_magical_image_operator(spec, source_experiment.specimen_template)
    rng = np.random.default_rng(random_seed)
    images = []
    for commands, nominal in zip(lens_controls, nominal_magnifications):
        shape = np.asarray(predict_observables(truth, commands, spec))
        p = np.r_[
            shape,
            rng.normal(0, 0.3, 2),
            np.log(4e7 * (nominal / 40000.0) ** 2),
            np.log(3.0),
        ]
        mean = np.asarray(operator(p))
        images.append(
            rng.poisson(mean).astype(float)
            + rng.normal(0, spec["read_noise_std_e"], mean.shape)
        )
    (directory / "scoring_truth.json").write_text(record)
    result = Experiment(
        np.asarray(images),
        np.asarray(lens_controls),
        np.asarray(nominal_magnifications),
        list(image_labels),
        source_experiment.specimen_template,
        spec,
        directory,
    )
    np.savez_compressed(
        directory / "images.npz",
        images=result.images,
        lens_controls=result.lens_controls,
        nominal_magnifications=result.nominal_magnifications,
        image_labels=result.image_labels,
    )
    return result


def combine_tilt_measurements(*measurement_sets):
    """Combine independent pairs; original pairs appear once in the larger experiment."""
    keys = (
        "lens_controls",
        "angle_changes_rad",
        "displacements_pixels",
        "covariance_pixels",
    )
    result = {key: np.concatenate([m[key] for m in measurement_sets]) for key in keys}
    result["pair_labels"] = sum((list(m["pair_labels"]) for m in measurement_sets), [])
    return result


def make_starting_models(
    *, prior_model, position_priors, number_of_starts, random_seed=482
):
    """One declared guess plus reproducible, dispersed starts within the same bounds.

    Position starts cover the inner 80% of the allowed boxes. Lengths and gains
    vary around the fixed prior model; this is an exploration, not a posterior
    sample or an exhaustive search of the 19-dimensional parameter space.
    """
    from .projector_model import position_prior_coordinates

    if number_of_starts < 1:
        raise ValueError("At least one starting model is required.")
    p = model_to_parameters(prior_model)
    encode, decode, lower, upper = position_prior_coordinates(position_priors)
    centre = np.asarray(encode(p))
    length_bounds = np.log(np.asarray(position_priors["length_bounds_mm"]) / 1000)
    rng = np.random.default_rng(random_seed)
    models = [prior_model]
    for _ in range(number_of_starts - 1):
        u = centre.copy()
        u[:6] = lower + rng.uniform(0.1, 0.9, 6) * (upper - lower)
        u[6:11] = np.clip(
            u[6:11] + rng.normal(0, 0.25, 5),
            length_bounds[:, 0] + 0.01,
            length_bounds[:, 1] - 0.01,
        )
        u[11] = rng.uniform(-0.6, -0.25)
        u[12:16] += rng.uniform(-0.25, 0.25, 4)
        u[16] = rng.uniform(-0.2, 0.2)
        u[17] += rng.uniform(-0.25, 0.25)
        u[18] = rng.uniform(-0.4, 0.4)
        models.append(parameters_to_model(np.asarray(decode(u))))
    return models


@dataclass
class BeamShiftExperiment:
    images: np.ndarray
    lens_controls: np.ndarray
    input_positions_m: np.ndarray
    input_angles_rad: np.ndarray
    nominal_magnifications: np.ndarray
    base_image_indices: np.ndarray
    image_labels: list[str]
    pair_labels: list[str]
    drift_std_pixels: float


def make_beam_shift_image_operator(spec, template):
    """A fixed specimen texture under a movable circular illumination envelope.

    Nine image parameters: log scale, pattern angle, log blur_px, pattern
    translation x/y_px, log dose, log background, beam centre x/y_nm in
    specimen-template coordinates. Beam motion reweights the fixed specimen;
    it does not translate the entire illuminated texture.
    """
    axis = jnp.asarray(template["axis_m"])
    xx, yy = jnp.meshgrid(axis, axis)
    centres = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)
    intensity = jnp.asarray(template["intensity"]).ravel()
    radius = spec["specimen"]["illumination_radius_m"]
    edge_width = spec["specimen"]["illumination_edge_m"]

    def image(p):
        c, s = jnp.cos(p[1]), jnp.sin(p[1])
        rotation = jnp.array([[c, -s], [s, c]])
        xy = jnp.exp(p[0]) * (centres @ rotation.T) / spec["pixel_pitch_m"] + p[3:5]
        # Work on the full known specimen grid so shifting the beam reveals
        # texture outside the original circular footprint as well.
        distance = jnp.sqrt(jnp.sum((centres - p[7:9] * 1e-9) ** 2, axis=1) + 1e-40)
        weights = intensity * jax.nn.sigmoid((radius - distance) / edge_width)
        weights = weights / weights.sum()
        return gaussian_intensity_image(
            xy, jnp.exp(2 * p[2]), weights, jnp.exp(p[5]), jnp.exp(p[6]),
            shape_yx=tuple(spec["shape_yx"]),
        )

    return jax.jit(image)


def beam_centre_pixels(image_parameters, pixel_pitch_m):
    """Geometric illumination centre, distinct from a texture-weighted centroid."""
    p = image_parameters
    c, s = jnp.cos(p[1]), jnp.sin(p[1])
    rotation = jnp.array([[c, -s], [s, c]])
    return jnp.exp(p[0]) * rotation @ (p[7:9] * 1e-9) / pixel_pitch_m + p[3:5]


def simulate_beam_shift_images(
    *,
    experiment,
    lens_setting_indices,
    shift_nm,
    drift_std_pixels=0.003,
    random_seed=917,
    angle_change_per_position_rad_per_m=None,
):
    """Make -/+ x and -/+ y beam-shift pairs on the existing MAGICAL region.

    Rays have r_s=s+u and theta_s=h*u+alpha+epsilon. The beam and its
    wavefront move together; mean angle and curvature stay fixed. The fixed
    specimen is reweighted by the shifted circular illumination. A supplied
    2x2 coupling matrix represents KNOWN accompanying mean-angle changes.
    """
    indices = np.asarray(lens_setting_indices)
    if (
        indices.ndim != 1
        or not len(indices)
        or indices.dtype.kind not in "iu"
        or np.any(indices < 0)
        or np.any(indices >= len(experiment.images))
    ):
        raise ValueError("Supply valid base-image indices for beam-shift settings.")
    if not np.isfinite(shift_nm) or shift_nm <= 0 or shift_nm > 20:
        raise ValueError(
            "Supply a beam-shift amplitude above zero and at most 20 nm for this template."
        )
    if not np.isfinite(drift_std_pixels) or drift_std_pixels < 0:
        raise ValueError("Drift standard deviation must be finite and nonnegative.")
    coupling = (
        np.zeros((2, 2))
        if angle_change_per_position_rad_per_m is None
        else np.asarray(angle_change_per_position_rad_per_m)
    )
    if coupling.shape != (2, 2) or not np.isfinite(coupling).all():
        raise ValueError("Supply a finite 2x2 calibrated position-to-angle coupling.")
    truth = np.asarray(
        json.loads((experiment.output_directory / "scoring_truth.json").read_text())[
            "parameters"
        ]
    )
    spec = experiment.specimen_and_detector
    operator = make_beam_shift_image_operator(spec, experiment.specimen_template)
    c, s = np.cos(truth[18]), np.sin(truth[18])
    specimen_rotation = np.array([[c, -s], [s, c]])
    rng = np.random.default_rng(random_seed)
    (
        images,
        controls,
        positions,
        angles,
        magnifications,
        base_indices,
        labels,
        pair_labels,
    ) = [], [], [], [], [], [], [], []
    for index in indices:
        commands, nominal = (
            experiment.lens_controls[index],
            experiment.nominal_magnifications[index],
        )
        shape = np.asarray(predict_observables(truth, commands, spec))
        matrix = np.asarray(matrix5(truth, commands))
        b = matrix[:2, 2:4]
        for axis, name in zip(np.eye(2), ("x", "y")):
            offset = rng.normal(0, 0.3, 2)
            pair_labels.append(
                f"{experiment.image_labels[index]}, {name} shift ±{shift_nm:g} nm"
            )
            for sign in (-1, 1):
                position = sign * shift_nm * 1e-9 * axis
                angle = coupling @ position
                # At fixed specimen coordinate r, the incident mean angle is
                # h*(r-s)+alpha, so texture translation is B*(alpha-h*s).
                texture_shift = (
                    b @ (angle - truth[16] * 1e4 * position) / spec["pixel_pitch_m"]
                )
                texture_shift += offset + rng.normal(0, drift_std_pixels, 2)
                beam_in_template_nm = specimen_rotation.T @ position * 1e9
                p = np.r_[
                    shape,
                    texture_shift,
                    np.log(4e7 * (nominal / 40000.0) ** 2),
                    np.log(3.0),
                    beam_in_template_nm,
                ]
                mean = np.asarray(operator(p))
                images.append(
                    rng.poisson(mean).astype(float)
                    + rng.normal(0, spec["read_noise_std_e"], mean.shape)
                )
                controls.append(commands)
                positions.append(position)
                angles.append(angle)
                magnifications.append(nominal)
                base_indices.append(index)
                labels.append(
                    f"{experiment.image_labels[index]}, {name} {sign * shift_nm:+g} nm"
                )
    result = BeamShiftExperiment(
        np.asarray(images),
        np.asarray(controls),
        np.asarray(positions),
        np.asarray(angles),
        np.asarray(magnifications),
        np.asarray(base_indices),
        labels,
        pair_labels,
        drift_std_pixels,
    )
    np.savez_compressed(
        experiment.output_directory / f"beam_shift_images_{shift_nm:g}nm.npz",
        **vars(result),
    )
    return result


def analyse_beam_shift_images(
    *,
    images,
    nominal_magnifications,
    specimen_template,
    specimen_and_detector,
    initial_image_parameters,
    verbose=True,
):
    """Fit texture and illumination independently from pixels, with nine parameters.

    Earlier image registrations supply initial scale, angle, blur and shifts.
    The illumination centre starts at zero in every image. Analysis receives
    no calibrated input shifts, true image centres or microscope parameters.
    Full image-parameter covariance is propagated to the illumination centre.
    """
    images = np.asarray(images)
    seeds = np.asarray(initial_image_parameters)
    nominal = np.asarray(nominal_magnifications)
    spec = specimen_and_detector
    if (
        images.ndim != 3
        or not len(images)
        or images.shape[1:] != tuple(spec["shape_yx"])
        or seeds.shape != (len(images), 7)
        or nominal.shape != (len(images),)
    ):
        raise ValueError(
            "Supply matching images, magnification labels and seven-parameter reference registrations."
        )
    if not all(np.isfinite(v).all() for v in (images, seeds, nominal)) or np.any(
        nominal <= 0
    ):
        raise ValueError(
            "Supply finite image inputs and positive magnification labels."
        )
    operator = make_beam_shift_image_operator(spec, specimen_template)
    operator_jacobian = jax.jit(jax.jacfwd(operator))
    centre = jax.jit(lambda p: beam_centre_pixels(p, spec["pixel_pitch_m"]))
    centre_jacobian = jax.jit(jax.jacfwd(centre))
    parameters, covariances, centres, centre_covariances, errors = [], [], [], [], []

    def residual(p, image, sigma):
        return ((operator(p) - image) / sigma).ravel()

    def derivative_image(p, image, sigma):
        return (operator_jacobian(p) / sigma[..., None]).reshape(-1, 9)

    for index, image in enumerate(images):
        lower = np.r_[
            np.log(nominal[index] / 2),
            -4 * np.pi,
            np.log(0.15),
            -20.0,
            -20.0,
            np.log([1e3, 0.01]),
            -30.0,
            -30.0,
        ]
        upper = np.r_[
            np.log(nominal[index] * 2),
            4 * np.pi,
            np.log(60.0),
            20.0,
            20.0,
            np.log([1e12, 1e6]),
            30.0,
            30.0,
        ]
        start = np.clip(np.r_[seeds[index], 0.0, 0.0], lower + 1e-8, upper - 1e-8)
        sigma = np.sqrt(np.maximum(image, 1.0) + spec["read_noise_std_e"] ** 2)
        result = least_squares(
            residual,
            start,
            jac=derivative_image,
            args=(jnp.asarray(image), jnp.asarray(sigma)),
            bounds=(lower, upper),
            x_scale="jac",
            max_nfev=150,
            max_iter=300,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-5,
        )
        rms = float(np.sqrt(np.mean(result.residual**2)))
        if not result.success or rms > 1.5:
            raise RuntimeError(
                f"Beam-shift registration failed at frame {index}: {result.message}; RMS {rms:.3f}"
            )
        covariance = np.linalg.inv(
            np.asarray(result.jacobian).T @ np.asarray(result.jacobian)
        )
        derivative = np.asarray(centre_jacobian(result.x))
        parameters.append(result.x)
        covariances.append(covariance)
        centres.append(np.asarray(centre(result.x)))
        centre_covariances.append(derivative @ covariance @ derivative.T)
        errors.append(rms)
        if verbose:
            print(
                f"MAGICAL beam-shift image {index + 1}/{len(images)}: weighted pixel RMS {rms:.3f}",
                flush=True,
            )
    return {
        "image_parameters": np.asarray(parameters),
        "parameter_covariance": np.asarray(covariances),
        "beam_centres_pixels": np.asarray(centres),
        "beam_centre_covariances": np.asarray(centre_covariances),
        "image_residual_rms": np.asarray(errors),
    }


def measure_beam_shift_images(
    *,
    beam_shift_experiment,
    specimen_template,
    specimen_and_detector,
    reference_measurements,
):
    """Pixels -> pattern and illumination fits -> beam-centre pair differences."""
    experiment = beam_shift_experiment
    measured = analyse_beam_shift_images(
        images=experiment.images,
        nominal_magnifications=experiment.nominal_magnifications,
        specimen_template=specimen_template,
        specimen_and_detector=specimen_and_detector,
        initial_image_parameters=reference_measurements.image_parameters[
            experiment.base_image_indices
        ],
    )
    result = pair_tilt_measurements(
        lens_controls=experiment.lens_controls,
        input_tilts_rad=experiment.input_angles_rad,
        positions_pixels=measured["beam_centres_pixels"],
        position_covariances=measured["beam_centre_covariances"],
        drift_std_pixels=experiment.drift_std_pixels,
        pair_labels=experiment.pair_labels,
    )
    positions = np.asarray(experiment.input_positions_m)
    if (
        positions.shape != (len(experiment.images), 2)
        or not np.isfinite(positions).all()
    ):
        raise ValueError(
            "Supply finite calibrated x/y beam positions for every exposure."
        )
    result["position_changes_m"] = positions[1::2] - positions[::2]
    result["image_analysis"] = measured
    return result


def combine_beam_shift_measurements(*measurement_sets):
    keys = (
        "lens_controls",
        "position_changes_m",
        "angle_changes_rad",
        "displacements_pixels",
        "covariance_pixels",
    )
    result = {key: np.concatenate([m[key] for m in measurement_sets]) for key in keys}
    result["pair_labels"] = functools.reduce(
        operator.iadd, (list(m["pair_labels"]) for m in measurement_sets), []
    )
    return result


@dataclass
class CalibrationImages:
    ordinary: Experiment
    tilts: TiltExperiment
    beam_shifts: BeamShiftExperiment

    @property
    def image_count(self):
        return (
            len(self.ordinary.images)
            + len(self.tilts.images)
            + len(self.beam_shifts.images)
        )


def measure_calibration_images(*, images, reuse_completed=True):
    """Measure image scale/rotation/blur and signed tilt/beam displacements.

    Registration starts with nominal magnification and an image-based rotation
    search. It receives no true or previously fitted optical parameters.
    """
    directory = images.ordinary.output_directory
    fingerprint = hashlib.sha256()
    for batch in (images.ordinary, images.tilts, images.beam_shifts):
        for value in vars(batch).values():
            if isinstance(value, np.ndarray):
                fingerprint.update(str((value.shape, value.dtype)).encode())
                fingerprint.update(value.tobytes())
    fingerprint.update(
        json.dumps(images.ordinary.specimen_and_detector, sort_keys=True).encode()
    )
    for key, value in sorted(images.ordinary.specimen_template.items()):
        fingerprint.update(key.encode())
        fingerprint.update(np.asarray(value).tobytes())
    key = fingerprint.hexdigest()
    marker = directory / "measurement_request.json"
    path = directory / "measurements.npz"
    if (
        reuse_completed
        and marker.exists()
        and json.loads(marker.read_text()) == key
        and path.exists()
    ):
        return CalibrationMeasurements.load(path)
    marker.unlink(missing_ok=True)
    ordinary = images.ordinary
    measured = measure_images(
        images=ordinary.images,
        nominal_magnifications=ordinary.nominal_magnifications,
        specimen_template=ordinary.specimen_template,
        specimen_and_detector=ordinary.specimen_and_detector,
    )
    tilts = measure_tilt_images(
        tilt_experiment=images.tilts,
        specimen_template=ordinary.specimen_template,
        specimen_and_detector=ordinary.specimen_and_detector,
        reference_measurements=measured,
    )
    shifts = measure_beam_shift_images(
        beam_shift_experiment=images.beam_shifts,
        specimen_template=ordinary.specimen_template,
        specimen_and_detector=ordinary.specimen_and_detector,
        reference_measurements=measured,
    )
    result = CalibrationMeasurements(
        ordinary.lens_controls, measured.values, measured.covariance, tilts, shifts
    )
    result.save(path)
    np.savez_compressed(
        directory / "image_analysis.npz",
        image_parameters=measured.image_parameters,
        image_error_in_noise_units=measured.image_error_in_noise_units,
    )
    marker.write_text(json.dumps(key))
    return result
