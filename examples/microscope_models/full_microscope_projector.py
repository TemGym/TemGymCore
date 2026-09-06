"""Two-voltage projector calibration from independently measured image pairs.

Physical objective current is always one. Only generation reads lens truth.
The fit uses calibrated position/angle changes to measure A and B, without
assuming common illumination curvature or divergence between voltages.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from condenser_objective_experiment import choose_objective_projector_views
from condenser_voltage_experiment import magnetic_strength_ratio, projector_at_voltage
from finite_projector_pipeline import (
    SIGNS,
    _generator_parameters,
    initial_guess,
    matrix5,
    position_prior_coordinates,
    predict_observables,
    public_specification,
)
from magical_projector_pipeline import make_magical_template
from projector_adaptive_experiments import (
    CalibrationMeasurements,
    simulate_setting_images,
)
from projector_beam_shift_pipeline import (
    BeamShiftExperiment,
    simulate_beam_shift_images,
)
from projector_calibration_workflow import CalibrationImages, measure_calibration_images
from projector_landscape import make_starting_models
from projector_notebook_helpers import Experiment, model_to_parameters
from projector_tilt_pipeline import TiltExperiment, simulate_tilt_images
from scipy.optimize import least_squares
from scipy.stats import chi2


def at_voltage(parameters, voltage, reference=200000.0):
    ratio = magnetic_strength_ratio(voltage, reference)
    return jnp.asarray(parameters).at[11].multiply(ratio).at[12:16].add(jnp.log(ratio))


def plan_fixed_objective_settings(
    *, generating_parameters, protocol, voltage, reference
):
    """Simulated setup adjusts downstream controls to keep the specimen visible.

    Only the resulting commands and nominal scales are supplied to inference.
    In a real acquisition these would be chosen by observing the specimen.
    """
    p = np.asarray(generating_parameters)
    views = []
    for command in protocol["lens_controls"]:
        command = np.asarray(command)
        m = np.asarray(matrix5(p, command))
        angle = p[11] * command[0] + np.dot(
            np.asarray(SIGNS) * np.exp(p[12:16]), command[1:]
        )
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        views.append(
            {
                "controls": command.tolist(),
                "target": [
                    float((rotation @ m[:2, :2])[0, 0]),
                    float((rotation @ m[:2, 2:4])[0, 0]),
                ],
            }
        )
    for v in np.linspace(
        reference, voltage, max(2, int(abs(voltage - reference) / 1000) + 1)
    ):
        _, controls = choose_objective_projector_views(
            {
                "calibrated_projector_parameters": np.asarray(
                    at_voltage(p, v, reference)
                ).tolist(),
                "views": views,
            },
            1.0,
        )
        for view, command in zip(views, controls):
            view["controls"] = command.tolist()
    if not np.all(controls[:, 0] == 1.0):
        raise AssertionError("Objective current must stay fixed.")
    return controls


def simulate_projector_block(
    *, protocol_path, output_directory, voltages_v, random_seed
):
    """Generate ordinary, calibrated-tilt and beam-shift MAGICAL images at both voltages."""
    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(Path(protocol_path).read_text())
    truth, _ = _generator_parameters()
    (root / "scoring_truth.json").write_text(
        json.dumps({"parameters": truth.tolist()}, indent=2)
    )
    (root / "experiment.json").write_text(
        json.dumps(
            {
                "voltages_v": list(voltages_v),
                "reference_voltage_v": voltages_v[0],
                "objective_current": 1.0,
            },
            indent=2,
        )
    )
    for index, voltage in enumerate(voltages_v):
        folder = root / f"voltage_{index}"
        folder.mkdir(exist_ok=True)
        spec = public_specification()
        spec.update(voltage_v=float(voltage), read_noise_std_e=2.0)
        template, metadata = make_magical_template(
            voltage, illumination_radius_m=100e-9
        )
        spec.update(
            specimen=metadata,
            feature_sigma_m=metadata["gaussian_component_sigma_m"],
            scope="Known MAGICAL specimen with calibrated size and circular illumination.",
            illumination_model="Unknown curvature and angular spread, separate at each voltage.",
        )
        spec.pop("feature_centres_m", None)
        parameters = np.asarray(at_voltage(truth, voltage, voltages_v[0])).copy()
        # Illumination can change with voltage. The fit estimates both states;
        # calibrated displacements initialize the shared optical geometry.
        parameters[16] += 0.02 * index
        parameters[17] += np.log(1.03) * index
        controls = plan_fixed_objective_settings(
            generating_parameters=truth,
            protocol=protocol,
            voltage=voltage,
            reference=voltages_v[0],
        )
        (folder / "scoring_truth.json").write_text(
            json.dumps({"parameters": parameters.tolist()})
        )
        (folder / "known_settings.json").write_text(json.dumps(spec, indent=2))
        np.savez_compressed(folder / "specimen_template.npz", **template)
        source = Experiment(
            np.empty((0, *spec["shape_yx"])),
            controls,
            np.asarray(protocol["nominal_magnifications"]),
            [f"Downstream setting {i + 1}" for i in range(len(controls))],
            template,
            spec,
            folder,
        )
        ordinary = simulate_setting_images(
            source_experiment=source,
            lens_controls=controls,
            nominal_magnifications=source.nominal_magnifications,
            image_labels=source.image_labels,
            output_directory=folder,
            random_seed=random_seed + 10 * index,
        )
        indices = list(range(len(controls)))
        simulate_tilt_images(
            experiment=ordinary,
            lens_setting_indices=indices,
            tilt_mrad=3.0,
            random_seed=random_seed + 10 * index + 1,
        )
        simulate_beam_shift_images(
            experiment=ordinary,
            lens_setting_indices=indices,
            shift_nm=10.0,
            random_seed=random_seed + 10 * index + 2,
        )
        print(
            f"Projector: {voltage / 1000:g} kV, {9 * len(controls)} images", flush=True
        )
    return root


def load_projector_images(folder):
    """Load pixels and public acquisition metadata, without reading scoring truth."""
    folder = Path(folder)
    data = dict(np.load(folder / "images.npz"))
    known = json.loads((folder / "known_settings.json").read_text())
    ordinary = Experiment(
        data["images"],
        data["lens_controls"],
        data["nominal_magnifications"],
        data["image_labels"].tolist(),
        dict(np.load(folder / "specimen_template.npz")),
        known,
        folder,
    )
    pair_data = [
        dict(np.load(folder / name))
        for name in ["tilt_images_3mrad.npz", "beam_shift_images_10nm.npz"]
    ]
    for data in pair_data:
        data["drift_std_pixels"] = float(data["drift_std_pixels"])
        for key in ["pair_labels", "image_labels"]:
            if key in data:
                data[key] = data[key].tolist()
    return CalibrationImages(
        ordinary, TiltExperiment(**pair_data[0]), BeamShiftExperiment(**pair_data[1])
    )


def measure_projector_block(*, images_directory):
    root = Path(images_directory)
    experiment = json.loads((root / "experiment.json").read_text())
    for index, _ in enumerate(experiment["voltages_v"]):
        measure_calibration_images(
            images=load_projector_images(root / f"voltage_{index}")
        )
    return root


def projector_coordinates(bounds_path):
    bounds = json.loads(Path(bounds_path).read_text())
    encode, decode, lo, hi = position_prior_coordinates(bounds)
    base = initial_guess()
    lower = np.r_[
        lo,
        np.log(np.asarray(bounds["length_bounds_mm"])[:, 0] / 1000),
        -0.9,
        base[12:16] - 0.8,
    ]
    upper = np.r_[
        hi,
        np.log(np.asarray(bounds["length_bounds_mm"])[:, 1] / 1000),
        -0.15,
        base[12:16] + 0.8,
    ]

    def physical(u):
        return decode(jnp.r_[u, jnp.zeros(3)])

    return encode, physical, lower, upper


def projector_residual(
    *, measurements_directory, bounds_path, include_image_measurements=True
):
    """Original measured A/B responses, one shared physical geometry at both voltages."""
    root = Path(measurements_directory)
    experiment = json.loads((root / "experiment.json").read_text())
    _, decode, _, _ = projector_coordinates(bounds_path)
    blocks, image_blocks = [], []
    for index, voltage in enumerate(experiment["voltages_v"]):
        folder = root / f"voltage_{index}"
        measured = CalibrationMeasurements.load(folder / "measurements.npz")
        pitch = json.loads((folder / "known_settings.json").read_text())[
            "pixel_pitch_m"
        ]
        spec = json.loads((folder / "known_settings.json").read_text())
        image_blocks.append(
            (
                voltage,
                jnp.asarray(measured.controls),
                jnp.asarray(measured.values),
                jnp.asarray(np.linalg.inv(np.linalg.cholesky(measured.covariance))),
                spec,
            )
        )
        for kind, pairs in [("tilts", measured.tilts), ("shifts", measured.shifts)]:
            controls = jnp.asarray(pairs["lens_controls"])
            positions = (
                jnp.zeros_like(jnp.asarray(pairs["angle_changes_rad"]))
                if kind == "tilts"
                else jnp.asarray(pairs["position_changes_m"])
            )
            angles = jnp.asarray(pairs["angle_changes_rad"])
            values = jnp.asarray(pairs["displacements_pixels"])
            white = jnp.asarray(
                np.linalg.inv(np.linalg.cholesky(pairs["covariance_pixels"]))
            )
            blocks.append((voltage, pitch, controls, positions, angles, values, white))

    def residual(u):
        parts = []
        p = decode(u[:16])
        if include_image_measurements:
            for index, (voltage, controls, values, white, spec) in enumerate(
                image_blocks
            ):
                adjusted = (
                    at_voltage(p, voltage, experiment["reference_voltage_v"])
                    .at[16:19]
                    .set(jnp.array([u[16 + 2 * index], u[17 + 2 * index], u[20]]))
                )
                prediction = jax.vmap(
                    lambda command, adjusted=adjusted, spec=spec: predict_observables(
                        adjusted, command, spec
                    )
                )(controls)
                difference = prediction - values
                difference = difference.at[:, 1].set(
                    jnp.arctan2(jnp.sin(difference[:, 1]), jnp.cos(difference[:, 1]))
                )
                parts.append(jnp.einsum("nij,nj->ni", white, difference).ravel())
        for voltage, pitch, controls, positions, angles, values, white in blocks:
            maps = jax.vmap(
                lambda command, voltage=voltage: projector_at_voltage(
                    p,
                    command,
                    voltage,
                    reference_voltage_v=experiment["reference_voltage_v"],
                )
            )(controls)
            predicted = (
                jnp.einsum("nij,nj->ni", maps[:, :2, :2], positions)
                + jnp.einsum("nij,nj->ni", maps[:, :2, 2:4], angles)
            ) / pitch
            parts.append(jnp.einsum("nij,nj->ni", white, predicted - values).ravel())
        return jnp.concatenate(parts)

    return jax.jit(residual)


def fit_projector_block(
    *, measurements_directory, starting_guess_path, bounds_path, number_of_starts=8
):
    root = Path(measurements_directory)
    output = root / "fit"
    output.mkdir(exist_ok=True)
    guess = json.loads(Path(starting_guess_path).read_text())
    bounds = json.loads(Path(bounds_path).read_text())
    encode, decode, lower, upper = projector_coordinates(bounds_path)
    optical_residual = projector_residual(
        measurements_directory=root,
        bounds_path=bounds_path,
        include_image_measurements=False,
    )
    optical_derivative = jax.jit(jax.jacfwd(optical_residual))
    lower = np.r_[lower, -10.0, np.log(0.00015), -10.0, np.log(0.00015), -np.pi]
    upper = np.r_[upper, 10.0, np.log(0.01), 10.0, np.log(0.01), np.pi]
    residual = projector_residual(measurements_directory=root, bounds_path=bounds_path)
    derivative = jax.jit(jax.jacfwd(residual))
    models = make_starting_models(
        prior_model=guess, position_priors=bounds, number_of_starts=number_of_starts
    )
    attempts = []
    for index, model in enumerate(models):
        native = model_to_parameters(model)
        optical_start = np.asarray(encode(native))[:16]
        initial = least_squares(
            optical_residual,
            optical_start,
            jac=optical_derivative,
            bounds=(lower[:16], upper[:16]),
            x_scale="jac",
            max_nfev=5000,
            ftol=1e-10,
            xtol=1e-11,
            gtol=1e-5,
        )
        start = np.r_[
            np.clip(initial.x, lower[:16] + 1e-9, upper[:16] - 1e-9),
            native[16:18],
            native[16:18],
            native[18],
        ]
        fit = least_squares(
            residual,
            start,
            jac=derivative,
            bounds=(lower, upper),
            x_scale="jac",
            max_nfev=5000,
            ftol=1e-10,
            xtol=1e-11,
            gtol=1e-5,
        )
        row = {
            "index": index,
            "starting_parameters": start.tolist(),
            "solver_parameters": fit.x.tolist(),
            "parameters": np.asarray(
                decode(fit.x[:16])
                .at[16:19]
                .set(jnp.array([fit.x[16], fit.x[17], fit.x[20]]))
            ).tolist(),
            "displacement_initialization": {
                "success": bool(initial.success),
                "residual_rms": float(np.sqrt(np.mean(initial.fun**2))),
                "parameters": initial.x.tolist(),
            },
            "success": bool(fit.success),
            "evaluations": fit.nfev,
            "chi_squared": float(fit.fun @ fit.fun),
            "residual_rms": float(np.sqrt(np.mean(fit.fun**2))),
            "message": fit.message,
        }
        attempts.append(row)
        (output / f"start_{index:02d}.json").write_text(json.dumps(row, indent=2))
        print(
            f"Projector start {index + 1}: RMS {row['residual_rms']:.4g}, converged {fit.success}",
            flush=True,
        )
    converged = [r for r in attempts if r["success"]]
    if not converged:
        raise RuntimeError("No projector fit converged; attempts retained.")
    best = min(converged, key=lambda r: r["chi_squared"])
    jac = np.asarray(derivative(np.asarray(best["solver_parameters"])))
    _, singular, vt = np.linalg.svd(jac, full_matrices=False)
    rank = int(np.linalg.matrix_rank(jac / np.linalg.norm(jac, axis=0), tol=1e-9))
    if rank != 21 or best["chi_squared"] > chi2.ppf(0.99, len(jac) - 21):
        raise RuntimeError(
            f"Projector fit does not identify the 16 optical and five illumination parameters at noise level: rank {rank}, RMS {best['residual_rms']:.3g}."
        )
    covariance = (vt.T / singular) @ (vt / singular[:, None])
    for row in attempts:
        row["acceptable"] = bool(
            row["success"]
            and row["chi_squared"] <= chi2.ppf(0.99, len(jac) - 21)
            and row["chi_squared"] <= best["chi_squared"] + chi2.ppf(0.95, 21)
        )
    result = {
        "best_fit": best,
        "starts": attempts,
        "covariance": covariance.tolist(),
        "rank": rank,
        "bounds_lower": lower.tolist(),
        "bounds_upper": upper.tolist(),
        "soft_priors": False,
        "scope": "Shared geometry; calibrated input shifts/tilts; conditional on known voltage and round finite-solenoid model.",
    }
    (output / "result.json").write_text(json.dumps(result, indent=2))
    return result
