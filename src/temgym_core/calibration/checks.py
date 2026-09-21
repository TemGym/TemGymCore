"""Predict new Gaussian beam images and report recovered physical geometry."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .condenser_model import QUANTITIES, physical_geometry, physical_quantity
from .inference import PRESETS, make_predictions, profiled_residual, subset_settings
from .projector_model import LENSES, decode_geometry, projector_coordinates


def microscope_geometry(parameters, decode_projector):
    pp, cp = decode_projector(parameters[:16]), parameters[16:]
    _, lengths, centres, detector = decode_geometry(pp)
    return jnp.r_[
        centres * 1000,
        lengths * 1000,
        detector * 1000,
        physical_geometry(cp),
        physical_quantity(cp, "source_z_second_voltage_mm"),
        cp[29:],
    ]


GEOMETRY_NAMES = (
    [f"{name} position (mm)" for name in LENSES]
    + [f"{name} length (mm)" for name in LENSES]
    + ["Detector position (mm)"]
    + list(QUANTITIES)
    + [
        "Source z, second voltage (mm)",
        "Second aperture x (µm)",
        "Second aperture y (µm)",
    ]
)


def check_beam_microscope(*, directory_path, microscope, render_images=True):
    root = Path(directory_path)
    spec = json.loads((root / "experiment.json").read_text())
    settings = dict(np.load(root / "settings.npz"))
    data = np.load(root / "measurements.npz")
    selected = np.flatnonzero(settings["purpose"] == "check")
    if not len(selected):
        raise ValueError("Reserve independent 'check' settings before checking predictions.")
    if "recorded" in data and not data["recorded"][selected].all():
        raise ValueError("Prediction checks require both aperture images.")
    predict = make_predictions(spec, subset_settings(settings, selected))
    parameters = jnp.asarray(microscope["solver_parameters"])
    values = jnp.asarray(data["values"][selected])
    white = jnp.asarray(np.linalg.inv(np.linalg.cholesky(data["covariance"][selected])))
    residual = jax.jit(lambda p: profiled_residual(predict(p), values, white).ravel())
    error = np.asarray(residual(parameters))
    derivative = np.asarray(jax.jacfwd(residual)(parameters))
    covariance = np.asarray(microscope["covariance"])
    predictive = np.eye(len(error)) + derivative @ covariance @ derivative.T
    standardized = np.linalg.solve(np.linalg.cholesky(predictive), error)
    predicted = np.asarray(predict(parameters))
    pixel_rms = []
    if render_images:
        from .gaussian import render_beam_images

        # One common incident dose per aperture pair is a nuisance, not a fitted lens.
        flux_variance = data["covariance"][selected, :, 5, 5]
        dose = np.sum(
            np.asarray(values)[..., 5] * predicted[..., 5] / flux_variance, axis=1
        ) / np.sum(predicted[..., 5] ** 2 / flux_variance, axis=1)
        generated = []
        original = np.load(root / "images.npz")["images"][selected]
        numerical = json.loads((root / "numerical_checks.json").read_text())
        sampling = np.asarray(
            numerical.get(
                "samples_per_pixel_per_pair",
                np.full(len(settings["voltage_index"]), numerical["samples_per_pixel"]),
            )
        )
        for row, (index, pair) in enumerate(
            render_beam_images(
                specification=spec,
                settings=settings,
                projector_parameters=microscope["projector_parameters"],
                condenser_parameters=microscope["condenser_parameters"],
                selected=selected,
                gaussians_across=numerical["gaussians_across"],
                samples_per_pixel=max(4, int(sampling[selected].max())),
            )
        ):
            image = pair * dose[row] + spec["background_counts"]
            mask = data["regions"][index]
            delta = (original[row] - image) / np.sqrt(
                image + spec["read_noise_counts"] ** 2
            )
            pixel_rms.append(float(np.sqrt(np.mean(delta[mask] ** 2))))
            generated.append(image)
        np.savez_compressed(
            root / "predicted_images.npz", indices=selected, images=generated
        )
    # Truth is read only after frozen-geometry predictions have been evaluated.
    encode, decode, _, _ = projector_coordinates(
        PRESETS / "projector_position_priors.json"
    )
    geometry = lambda p: microscope_geometry(p, decode)
    recovered = np.asarray(geometry(parameters))
    reference = [None] * len(recovered)
    truth_path = root / "scoring_truth.json"
    if truth_path.exists():
        truth = json.loads(truth_path.read_text())
        actual = jnp.r_[
            encode(jnp.asarray(truth["projector_parameters"]))[:16],
            jnp.asarray(truth["condenser_parameters"]),
        ]
        reference = np.asarray(geometry(actual))
    g = np.asarray(jax.jacfwd(geometry)(parameters))
    std = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", g, covariance, g), 0.0))
    rows = [
        {
            "parameter": name,
            "fitted": float(value),
            "truth": None if target is None else float(target),
            "error": None if target is None else float(value - target),
            "standard_deviation": float(sigma),
        }
        for name, value, target, sigma in zip(GEOMETRY_NAMES, recovered, reference, std)
    ]
    report = {
        "fit_residual_rms": microscope["best"]["residual_rms"],
        "prediction_residual_rms": float(np.sqrt(np.mean(standardized**2))),
        "prediction_noise_only_rms": float(np.sqrt(np.mean(error**2))),
        "prediction_pixel_rms": pixel_rms,
        "image_predictions_checked": render_images,
        "geometry": rows,
        "rank": microscope["best"]["rank"],
        "parameter_count": len(parameters),
        "aperture_inserted_for_every_image": True,
        "global_uniqueness_proved": False,
        "truth_available": truth_path.exists(),
    }
    (root / "checks.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
