"""Score the assembled microscope against new measurements and hidden geometry."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from condenser_geometry_audit import physical_quantity
from condenser_image_pipeline import make_image_operator, predict_beam_coordinates
from condenser_simplification_design import QUANTITIES, physical_geometry
from condenser_voltage_experiment import condenser_at_voltage, projector_at_voltage
from finite_projector_pipeline import LENSES, decode_geometry
from full_microscope_apertures import (
    current_fractions,
    effective_parameters,
    make_aperture_transmission,
)
from full_microscope_projector import projector_coordinates, projector_residual
from scipy.linalg import block_diag
from scipy.optimize import nnls


def microscope_geometry(parameters, decode_projector):
    projector, condenser = decode_projector(parameters[:16]), parameters[16:]
    _, lengths, centres, detector = decode_geometry(projector)
    return jnp.r_[
        centres * 1000,
        lengths * 1000,
        detector * 1000,
        physical_geometry(condenser),
        physical_quantity(condenser, "source_z_second_voltage_mm"),
        condenser[29:],
    ]


GEOMETRY_NAMES = (
    [f"{name}_z_mm" for name in LENSES]
    + [f"{name}_length_mm" for name in LENSES]
    + ["detector_z_mm"]
    + [f"condenser_{name}" for name in QUANTITIES]
    + ["condenser_source_z_second_voltage_mm"]
)


def predict_condenser_pixels(*, images_directory, fitted, prediction=False):
    """Freeze both fitted halves; estimate only each frame's dose and background."""
    root = Path(images_directory)
    experiment = json.loads((root / "experiment.json").read_text())
    projector = jnp.asarray(fitted["projector_parameters"])
    condenser = jnp.asarray(fitted["condenser_parameters"])
    errors, per_frame = [], []
    for index, voltage in enumerate(experiment["voltages_v"]):
        folder = root / f"voltage_{index}"
        if prediction:
            folder /= "prediction"
        data = np.load(folder / "images.npz")
        detector = json.loads((folder / "detector.json").read_text())
        maps = np.asarray(
            [
                projector_at_voltage(
                    projector,
                    command,
                    voltage,
                    reference_voltage_v=experiment["reference_voltage_v"],
                )
                for command in data["projector_controls"]
            ]
        )
        render = make_image_operator(
            projector_matrices=maps,
            template=dict(np.load(folder / "template.npz")),
            detector=detector,
        )
        effective = condenser_at_voltage(
            condenser,
            voltage,
            reference_voltage_v=experiment["reference_voltage_v"],
            source_parameters=condenser[22:29] if index else None,
        )
        predictions = []
        for command, frames in zip(data["controls"], data["images"]):
            beam = predict_beam_coordinates(effective, command)
            shapes = np.asarray(render(jnp.r_[beam, jnp.zeros(2 * len(maps))])) - 1.0
            predicted = []
            for shape, observed in zip(shapes, frames):
                noise = np.sqrt(
                    np.maximum(observed, 1.0) + detector["read_noise_e"] ** 2
                )
                design = (
                    np.stack([shape.ravel(), np.ones(shape.size)], axis=1)
                    / noise.ravel()[:, None]
                )
                norms = np.linalg.norm(design, axis=0)
                exposure, _ = nnls(design / norms, observed.ravel() / noise.ravel())
                exposure /= norms
                expected = exposure[0] * shape + exposure[1]
                error = ((expected - observed) / noise).ravel()
                errors.append(error)
                per_frame.append(float(np.sqrt(np.mean(error**2))))
                predicted.append(expected)
            predictions.append(predicted)
        np.savez_compressed(
            folder / "full_microscope_predictions.npz", images=predictions
        )
    error = np.concatenate(errors)
    return {
        "pixel_rms": float(np.sqrt(np.mean(error**2))),
        "worst_frame_rms": max(per_frame),
        "pixel_count": len(error),
        "image_count": len(per_frame),
        "geometry_refitted": False,
        "fitted_nuisance_parameters": "Dose and background only, separately per frame.",
    }


def check_microscope(*, experiment_directory, fitted, bounds_path):
    """Prediction checks first; only the final geometry comparison reads truth."""
    root = Path(experiment_directory)
    _, decode, _, _ = projector_coordinates(bounds_path)
    p = np.r_[fitted["projector_solver_parameters"], fitted["condenser_parameters"]]
    covariance = np.asarray(fitted["covariance"])
    result = {
        "calibration_pixels": predict_condenser_pixels(
            images_directory=root / "condenser", fitted=fitted
        ),
        "prediction_pixels": predict_condenser_pixels(
            images_directory=root / "condenser", fitted=fitted, prediction=True
        ),
    }
    prediction_folder = root / "projector_predictions"
    if prediction_folder.exists():
        residual = projector_residual(
            measurements_directory=prediction_folder, bounds_path=bounds_path
        )
        calibration_parameters = np.asarray(fitted["projector_calibration_parameters"])
        error = np.asarray(residual(calibration_parameters))
        derivative = np.asarray(jax.jacfwd(residual)(calibration_parameters))
        predictive_covariance = (
            np.eye(len(error))
            + derivative
            @ np.asarray(fitted["projector_calibration_covariance"])
            @ derivative.T
        )
        whitened = np.linalg.solve(np.linalg.cholesky(predictive_covariance), error)
        result["projector_predictions"] = {
            "rms": float(np.sqrt(np.mean(whitened**2))),
            "coordinate_count": len(error),
            "refitted": False,
        }
    experiment = json.loads((root / "condenser/experiment.json").read_text())
    errors, variances, derivatives = [], [], []
    for index, voltage in enumerate(experiment["voltages_v"]):
        folder = root / "condenser" / f"voltage_{index}"
        data = np.load(folder / "prediction/current_counts.npz")
        transmitted, reference = (
            data["transmitted_electrons"].astype(float),
            data["reference_electrons"].astype(float),
        )
        values, measurement_covariance = current_fractions(transmitted, reference)
        variances.extend(measurement_covariance)
        transmit = make_aperture_transmission(
            specification=json.loads((folder / "aperture.json").read_text())
        )
        controls = jnp.asarray(data["controls"])

        def predict(
            parameters,
            index=index,
            voltage=voltage,
            transmit=transmit,
            controls=controls,
        ):
            c = parameters[16:]
            effective = effective_parameters(
                c, voltage, experiment["reference_voltage_v"], index
            )
            return jax.vmap(transmit, in_axes=(None, 0))(effective, controls).ravel()

        errors.append(np.asarray(predict(p)) - values.ravel())
        derivatives.append(np.asarray(jax.jacfwd(predict)(p)))
    error, derivative = np.concatenate(errors), np.concatenate(derivatives)
    predictive_covariance = (
        block_diag(*variances) + derivative @ covariance @ derivative.T
    )
    whitened = np.linalg.solve(np.linalg.cholesky(predictive_covariance), error)
    result["current_predictions"] = {
        "rms": float(np.sqrt(np.mean(whitened**2))),
        "coordinate_count": len(error),
        "refitted": False,
    }

    # Opening truth is deliberately last, after prediction errors are fixed.
    true_projector = np.asarray(
        json.loads((root / "projector/scoring_truth.json").read_text())["parameters"]
    )
    true_condenser = np.asarray(
        json.loads((root / "condenser/scoring_truth.json").read_text())["parameters"]
    )
    encode, _, _, _ = projector_coordinates(bounds_path)
    truth = np.r_[np.asarray(encode(true_projector))[:16], true_condenser]
    values = np.asarray(microscope_geometry(p, decode))
    actual = np.asarray(microscope_geometry(truth, decode))
    derivative = np.asarray(jax.jacfwd(lambda x: microscope_geometry(x, decode))(p))
    standard_error = np.sqrt(
        np.maximum(np.diag(derivative @ covariance @ derivative.T), 0.0)
    )
    names = GEOMETRY_NAMES + [
        f"condenser_aperture_{i}_{axis}_um"
        for i in range(2, 2 + len(true_condenser[29:]) // 2)
        for axis in ["x", "y"]
    ]
    result["geometry"] = {
        name: {
            "truth": float(t),
            "fitted": float(v),
            "error": float(v - t),
            "local_sigma": float(s),
            "truth_in_local_95_interval": bool(abs(v - t) <= 1.96 * s),
        }
        for name, t, v, s in zip(names, actual, values, standard_error)
    }
    accepted = [a for a in fitted["condenser_fit"]["starts"] if a["acceptable"]]
    if accepted:
        alternatives = np.asarray(
            [
                microscope_geometry(jnp.asarray(np.r_[p[:16], a["parameters"]]), decode)
                for a in accepted
            ]
        )
        # Condenser physical coordinates do not depend on the projector solver
        # coordinates used here. Report between-start spread separately from
        # local covariance, particularly for ambiguous aperture directions.
        for index in range(11, len(names)):
            result["geometry"][names[index]]["acceptable_start_min"] = float(
                alternatives[:, index].min()
            )
            result["geometry"][names[index]]["acceptable_start_max"] = float(
                alternatives[:, index].max()
            )
    result["alignment_accuracy_assumed"] = experiment.get(
        "alignment_accuracy_assumed", False
    )
    result["scope"] = (
        "One complete simulated acquisition. Full local projector/condenser covariance; no proof of global uniqueness, repeated-noise coverage or real-field-profile accuracy. Known voltages and calibrated input shifts/tilts at both voltages are assumed."
    )
    (root / "checks.json").write_text(json.dumps(result, indent=2))
    return result


def show_result(*, checks):
    from IPython.display import HTML, Markdown, display

    lines = [
        "| Lens | True z (mm) | Fitted z ± 1σ (mm) | True length (mm) | Fitted length ± 1σ (mm) |",
        "|---|---:|---:|---:|---:|",
    ]
    names = [
        ("CL1", "condenser_CL1"),
        ("CL3", "condenser_CL3"),
        ("Mini", "condenser_mini"),
        ("Objective pre-field", "condenser_prefield"),
    ] + [("Objective post-field" if n == "Objective" else n, n) for n in LENSES]
    for label, key in names:
        z, length = (
            checks["geometry"][key + "_z_mm"],
            checks["geometry"][key + "_length_mm"],
        )
        lines.append(
            f"| {label} | {z['truth']:.5g} | {z['fitted']:.6g} ± {z['local_sigma']:.2g} | {length['truth']:.5g} | {length['fitted']:.6g} ± {length['local_sigma']:.2g} |"
        )
    display(Markdown("\n".join(lines)))
    lines = [
        "| Frozen-model check | RMS in uncertainty units |",
        "|---|---:|",
        f"| New full-microscope images | {checks['prediction_pixels']['pixel_rms']:.3f} |",
        f"| New aperture currents | {checks['current_predictions']['rms']:.3f} |",
    ]
    if "projector_predictions" in checks:
        lines.append(
            f"| New projector settings | {checks['projector_predictions']['rms']:.3f} |"
        )
    display(Markdown("\n".join(lines)))
    extra = [
        n
        for n in checks["geometry"]
        if "source" in n or "aperture" in n or "detector" in n
    ]
    rows = []
    for name in extra:
        value = checks["geometry"][name]
        spread = (
            f"{value['acceptable_start_min']:.6g} to {value['acceptable_start_max']:.6g}"
            if "acceptable_start_min" in value
            else "—"
        )
        rows.append(
            f"<tr><td>{name.replace('_', ' ')}</td><td>{value['truth']:.6g}</td>"
            f"<td>{value['fitted']:.6g} ± {value['local_sigma']:.2g}</td><td>{spread}</td></tr>"
        )
    display(
        HTML(
            "<details><summary>Source, aperture and detector positions</summary><table>"
            "<tr><th>Quantity</th><th>Truth</th><th>Fit ± 1σ</th><th>Range across acceptable starts</th></tr>"
            + "".join(rows)
            + "</table></details>"
        )
    )
    ambiguous = [
        name
        for name in extra
        if "aperture" in name
        and checks["geometry"][name].get("acceptable_start_max", 0.0)
        - checks["geometry"][name].get("acceptable_start_min", 0.0)
        > 3.92 * checks["geometry"][name]["local_sigma"]
    ]
    if ambiguous:
        display(
            Markdown(
                "**Some aperture-centre coordinates have several acceptable values.** "
                "Their local error bars describe only the selected minimum; inspect the range across starts above."
            )
        )
