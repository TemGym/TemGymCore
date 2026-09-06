"""Image analysis and condenser fitting with a measured, uncertain projector.

The independent specimen beams and the common projector are first estimated
from pixels and the projector calibration measurements. Their full covariance
is retained. Marginalising the projector gives correlated beam measurements
for the condenser fit; Gaussian conditioning then updates the projector and
returns the cross-covariance between both fitted halves. This is a local
likelihood approximation, checked against the original pixels afterwards.
"""

import json
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from condenser_calibration_workflow import measure_images
from condenser_fitting import beam_to_moments, model_moments
from condenser_geometry_audit import search_alternatives
from condenser_image_pipeline import make_image_operator, predict_beam_coordinates
from condenser_voltage_experiment import (
    condenser_at_voltage,
    projector_at_voltage,
)
from full_microscope_apertures import (
    alignment_residual,
    effective_parameters,
    make_aperture_transmission,
)
from full_microscope_projector import projector_coordinates, projector_residual
from scipy.linalg import block_diag
from scipy.stats import qmc


def joint_measurement_covariance(
    conditional_beam_covariances, beam_responses, projector_covariance
):
    """Gaussian block elimination, retaining every correlation through the projector."""
    local = block_diag(*conditional_beam_covariances)
    response = np.concatenate(beam_responses, axis=0)
    cross = response @ projector_covariance
    return local + cross @ response.T, cross


def condition_projector(
    *,
    projector_mean,
    beam_mean,
    beam_covariance,
    beam_projector_covariance,
    projector_covariance,
    fitted_beams,
    beam_derivative,
    condenser_covariance,
):
    """Recover the common projector and full covariance after a marginal beam fit."""
    coupling = np.linalg.solve(beam_covariance, beam_projector_covariance).T
    mean = projector_mean + coupling @ (fitted_beams - beam_mean)
    conditional = projector_covariance - coupling @ beam_projector_covariance
    cross = coupling @ beam_derivative @ condenser_covariance
    covariance = conditional + cross @ beam_derivative.T @ coupling.T
    return mean, np.block([[covariance, cross], [cross.T, condenser_covariance]])


def prepare_condenser_analysis(*, images_directory, projector_fit, reference_voltage_v):
    """Copy public pixels/commands; replace generating maps with fitted projector maps.

    Neither scoring truth nor the generator's reference_projector.json is read.
    Image analysis receives a separate directory with no generating parameters.
    """
    source = Path(images_directory)
    output = source / "analysis"
    output.mkdir(exist_ok=True)
    experiment = json.loads((source / "experiment.json").read_text())
    experiment["calibration_assumption"] = (
        "Projector estimated from images at both voltages; its shared uncertainty "
        "is included in joint beam analysis and the full microscope fit."
    )
    (output / "experiment.json").write_text(json.dumps(experiment, indent=2))
    p = jnp.asarray(projector_fit["best_fit"]["parameters"])
    for index, voltage in enumerate(experiment["voltages_v"]):
        original, target = source / f"voltage_{index}", output / f"voltage_{index}"
        target.mkdir(exist_ok=True)
        data = np.load(original / "images.npz")
        # Only measured pixels and recorded controls cross this boundary.
        commands = data["projector_controls"]
        maps = np.asarray(
            [
                projector_at_voltage(
                    p, command, voltage, reference_voltage_v=reference_voltage_v
                )
                for command in commands
            ]
        )
        np.savez_compressed(
            target / "images.npz",
            images=data["images"],
            controls=data["controls"],
            projector_controls=commands,
            projector_matrices=maps,
        )
        for name in ["template.npz", "detector.json", "counts.npz", "aperture.json"]:
            shutil.copyfile(original / name, target / name)
        measure_images(image_directory=target, initialization_only=True)
        counts = np.load(target / "counts.npz")
        transmitted, reference = (
            counts["transmitted_electrons"].astype(float),
            counts["reference_electrons"].astype(float),
        )
        fractions = transmitted / reference[:, None]
        covariance = (
            np.asarray(jax.vmap(jnp.diag)(transmitted)) / reference[:, None, None] ** 2
            + fractions[:, :, None] * fractions[:, None, :] / reference[:, None, None]
        )
        np.savez_compressed(
            target / "current_measurements.npz",
            controls=counts["controls"],
            values=fractions,
            covariance=covariance,
        )
    if (source / "alignment/alignment_constraint.json").exists():
        (output / "alignment").mkdir(exist_ok=True)
        shutil.copyfile(
            source / "alignment/alignment_constraint.json",
            output / "alignment/alignment_constraint.json",
        )
    return output


def measure_specimen_beams(
    *,
    analysis_directory,
    projector_measurements_directory,
    projector_fit,
    bounds_path,
    max_iterations=15,
):
    """Pixels and measured projector responses -> independent beams and full covariance.

    The condenser geometry and source parameters are completely absent here.
    Per-frame exposure/background are fitted nuisance values. Voltages, input
    tilt/shift calibrations and detector response are known in this first full
    demonstration; voltage readback error is not an additional fitted unknown.
    """
    root = Path(analysis_directory)
    experiment = json.loads((root / "experiment.json").read_text())
    _, decode, lower, upper = projector_coordinates(bounds_path)
    calibration = projector_residual(
        measurements_directory=projector_measurements_directory, bounds_path=bounds_path
    )
    calibration_jacobian = jax.jit(jax.jacfwd(calibration))
    common = np.asarray(projector_fit["best_fit"]["solver_parameters"]).copy()
    common_size = len(common)
    lower, upper = (
        np.asarray(projector_fit["bounds_lower"]),
        np.asarray(projector_fit["bounds_upper"]),
    )
    batches, beams, control_rows, voltage_rows = [], [], [], []
    for index, voltage in enumerate(experiment["voltages_v"]):
        folder = root / f"voltage_{index}"
        data = np.load(folder / "images.npz")
        detector = json.loads((folder / "detector.json").read_text())
        render = make_image_operator(
            projector_matrices=data["projector_matrices"],
            template=dict(np.load(folder / "template.npz")),
            detector=detector,
        )
        commands = jnp.asarray(data["projector_controls"])

        def predict(values, render=render, commands=commands, voltage=voltage):
            p = decode(values[-common_size:][:16])
            maps = jax.vmap(
                lambda c: projector_at_voltage(
                    p, c, voltage, reference_voltage_v=experiment["reference_voltage_v"]
                )
            )(commands)
            return render(values[:-common_size], maps).ravel()

        predict = jax.jit(predict)
        derivative = jax.jit(jax.jacfwd(predict))
        measured = np.load(folder / "measurements.npz")
        for control, frames, initial in zip(
            data["controls"], data["images"], measured["image_parameters"]
        ):
            pixels = frames.ravel()
            weight = 1 / np.sqrt(
                np.maximum(pixels, 1.0) + detector["read_noise_e"] ** 2
            )
            batches.append((predict, derivative, pixels, weight))
            beams.append(initial)
            control_rows.append(control)
            voltage_rows.append(index)
    beams = np.asarray(beams)

    def cost(local, global_parameters):
        total = float(np.sum(np.asarray(calibration(global_parameters)) ** 2))
        for row, (predict, _, pixels, weight) in zip(local, batches):
            error = (
                np.asarray(predict(np.r_[row, global_parameters])) - pixels
            ) * weight
            total += float(error @ error)
        return total

    history, converged = [], False
    for iteration in range(max_iterations):
        shared_design = [np.asarray(calibration_jacobian(common))]
        shared_errors = [np.asarray(calibration(common))]
        local_steps, responses, local_covariances = [], [], []
        for row, (predict, derivative, pixels, weight) in zip(beams, batches):
            values = np.r_[row, common]
            error = (np.asarray(predict(values)) - pixels) * weight
            jacobian = np.asarray(derivative(values)) * weight[:, None]
            local, global_part = jacobian[:, :-common_size], jacobian[:, -common_size:]
            u, s, vt = np.linalg.svd(local, full_matrices=False)
            inverse = (vt.T / s) @ u.T
            shared_design.append(global_part - u @ (u.T @ global_part))
            shared_errors.append(error - u @ (u.T @ error))
            local_steps.append(-inverse @ error)
            responses.append(-inverse @ global_part)
            local_covariances.append((vt.T / s) @ (vt / s[:, None]))
        design, error = np.concatenate(shared_design), np.concatenate(shared_errors)
        u, singular, vt = np.linalg.svd(design, full_matrices=False)
        common_step = -(vt.T / singular) @ (u.T @ error)
        projector_covariance = (vt.T / singular) @ (vt / singular[:, None])
        beam_step = np.asarray(local_steps) + np.asarray(responses) @ common_step
        old_cost = cost(beams, common)
        fraction = 1.0
        while fraction >= 2**-14:
            candidate = common + fraction * common_step
            new_cost = (
                cost(beams + fraction * beam_step, candidate)
                if np.all((candidate > lower) & (candidate < upper))
                else np.inf
            )
            if np.isfinite(new_cost) and new_cost <= old_cost + 1e-7:
                break
            fraction /= 2
        if fraction < 2**-14:
            raise RuntimeError(
                "Joint beam/projector image analysis line search failed."
            )
        beams += fraction * beam_step
        common = candidate
        history.append(
            {
                "iteration": iteration,
                "chi_squared": new_cost,
                "projector_step": float(np.linalg.norm(fraction * common_step)),
                "step_fraction": fraction,
            }
        )
        print(
            f"Joint image analysis {iteration}: cost {new_cost:.6g}, projector step {np.linalg.norm(fraction * common_step):.3g}",
            flush=True,
        )
        if (
            abs(new_cost - old_cost) < 1e-7 * max(old_cost, 1.0)
            and np.linalg.norm(fraction * common_step) < 1e-5
        ):
            converged = True
            break
    pixel_rms = [
        float(
            np.sqrt(
                np.mean(
                    ((np.asarray(predict(np.r_[row, common])) - pixels) * weight) ** 2
                )
            )
        )
        for row, (predict, _, pixels, weight) in zip(beams, batches)
    ]
    if not converged or max(pixel_rms) > 1.5:
        raise RuntimeError(
            f"Corrected beam image analysis failed: converged={converged}, worst RMS={max(pixel_rms):.3g}."
        )
    covariance, beam_projector_covariance = joint_measurement_covariance(
        [c[:7, :7] for c in local_covariances],
        [r[:7] for r in responses],
        projector_covariance,
    )
    np.savez_compressed(
        root / "joint_beam_measurements.npz",
        values=beams[:, :7],
        covariance=covariance,
        conditional_covariance=np.asarray(local_covariances)[:, :7, :7],
        beam_response=np.asarray(responses)[:, :7],
        beam_projector_covariance=beam_projector_covariance,
        projector_covariance=projector_covariance,
        projector_solver_parameters=common,
        projector_bounds_lower=lower,
        projector_bounds_upper=upper,
        controls=control_rows,
        voltage_indices=voltage_rows,
        image_parameters=beams,
    )
    report = {
        "converged": converged,
        "history": history,
        "pixel_rms": pixel_rms,
        "projector_parameters": np.asarray(decode(common[:16])).tolist(),
        "projector_uncertainty_included": True,
        "source_or_condenser_truth_used": False,
        "covariance_scope": "Full local covariance after joint pixel analysis and projector response measurements; dose/background marginalised.",
    }
    (root / "beam_analysis.json").write_text(json.dumps(report, indent=2))
    return root


def condenser_residual(
    *, analysis_directory, use_moments=False, include_projector_uncertainty=True
):
    """Measured beams, correlated through the projector, plus independent currents."""
    root = Path(analysis_directory)
    experiment = json.loads((root / "experiment.json").read_text())
    data = np.load(root / "joint_beam_measurements.npz")
    values = jnp.asarray(data["values"])
    covariance = (
        data["covariance"]
        if include_projector_uncertainty
        else block_diag(*data["conditional_covariance"])
    )
    if use_moments:
        derivative = block_diag(
            *np.asarray(jax.vmap(jax.jacfwd(beam_to_moments))(values))
        )
        covariance = derivative @ covariance @ derivative.T
        values = jax.vmap(beam_to_moments)(values)
    white = jnp.asarray(np.linalg.inv(np.linalg.cholesky(covariance)))
    blocks = []
    for index, voltage in enumerate(experiment["voltages_v"]):
        folder = root / f"voltage_{index}"
        currents = np.load(folder / "current_measurements.npz")
        blocks.append(
            (
                voltage,
                jnp.asarray(data["controls"][data["voltage_indices"] == index]),
                jnp.asarray(currents["controls"]),
                jnp.asarray(currents["values"]),
                jnp.asarray(np.linalg.inv(np.linalg.cholesky(currents["covariance"]))),
                make_aperture_transmission(
                    specification=json.loads((folder / "aperture.json").read_text())
                ),
            )
        )
    alignment = json.loads((root / "alignment/alignment_constraint.json").read_text())

    def residual(p):
        beam_rows, parts = [], []
        for index, (voltage, controls, ac, av, aw, transmit) in enumerate(blocks):
            effective = effective_parameters(
                p, voltage, experiment["reference_voltage_v"], index
            )
            predict = model_moments if use_moments else predict_beam_coordinates
            beam_rows.append(jax.vmap(predict, in_axes=(None, 0))(effective, controls))
            fractions = jax.vmap(transmit, in_axes=(None, 0))(effective, ac)
            parts.append(jnp.einsum("nij,nj->ni", aw, fractions - av).ravel())
        parts.append(alignment_residual(p, alignment))
        return jnp.concatenate(
            [white @ (jnp.concatenate(beam_rows) - values).ravel(), *parts]
        )

    return jax.jit(residual)


def condenser_starting_guesses(
    *, starting_guess_path, number_of_broad_starts=4, number_of_apertures=1
):
    """Four engineering guesses and broad starts; no earlier fit or truth is loaded."""
    from condenser_aperture_experiment import equivalent_aperture_model
    from condenser_calibration_model import initial_parameters

    settings = json.loads(Path(starting_guess_path).read_text())
    model = settings["model"]
    fraction = model["aperture_gap_fraction"]
    p = np.r_[
        np.log(np.asarray(model["gaps_mm"]) / 1000),
        np.log(np.asarray(model["lens_lengths_mm"]) / 1000),
        np.log(model["strength_magnitudes"]),
        np.asarray(model["source_position_nm"]) / 10,
        np.asarray(model["source_angle_mrad"]) * 10,
        np.log(model["source_rms_nm"] * 1e-9),
        np.log(model["source_angular_rms_mrad"] * 1e-3),
        np.log(fraction / (1 - fraction)),
        model["aperture_centre_um"],
    ]

    def extend(base):
        return np.r_[
            base, base[0], base[13:19], np.tile(base[20:22], number_of_apertures - 1)
        ]

    starts = [extend(p), extend(np.r_[initial_parameters(), -3.0, 0.1, -0.1])]
    starts += [
        extend(equivalent_aperture_model(p, objective_strength=-q))
        for q in [0.35, 0.65]
    ]
    lower, upper = extend(settings["bounds_lower"]), extend(settings["bounds_upper"])
    if number_of_broad_starts:
        sample = qmc.LatinHypercube(d=len(lower), seed=713).random(
            number_of_broad_starts
        )
        starts += list(qmc.scale(sample, lower + 1e-7, upper - 1e-7))
    return starts, lower, upper


def fit_condenser_block(
    *, analysis_directory, starting_guess_path, number_of_broad_starts=4
):
    root = Path(analysis_directory)
    experiment = json.loads((root / "experiment.json").read_text())
    starts, lower, upper = condenser_starting_guesses(
        starting_guess_path=starting_guess_path,
        number_of_broad_starts=number_of_broad_starts,
        number_of_apertures=experiment.get("number_of_apertures", 1),
    )
    warm = search_alternatives(
        residual=condenser_residual(analysis_directory=root, use_moments=True),
        starting_parameters=starts,
        bounds_lower=lower,
        bounds_upper=upper,
        output_directory=root / "fit/initialization",
        max_evaluations=2000,
    )
    result = search_alternatives(
        residual=condenser_residual(analysis_directory=root),
        starting_parameters=[
            np.clip(r["parameters"], lower + 1e-9, upper - 1e-9) for r in warm["starts"]
        ],
        bounds_lower=lower,
        bounds_upper=upper,
        output_directory=root / "fit/final",
        max_evaluations=3000,
    )
    result["initialization"] = warm
    result["alignment_constraint"] = json.loads(
        (root / "alignment/alignment_constraint.json").read_text()
    )
    result["soft_geometry_priors"] = False
    if result["covariance"] is None or not any(
        r["acceptable"] for r in result["starts"]
    ):
        raise RuntimeError(
            "No full-rank condenser fit at noise level. Attempts retained."
        )
    (root / "fit/result.json").write_text(json.dumps(result, indent=2))
    return result


def assemble_full_fit(*, analysis_directory, condenser_fit, bounds_path):
    """Condition the measured projector on the fitted condenser and retain cross-covariance."""
    root = Path(analysis_directory)
    experiment = json.loads((root / "experiment.json").read_text())
    measured = np.load(root / "joint_beam_measurements.npz")
    p = jnp.asarray(condenser_fit["best_fit"]["parameters"])
    controls = jnp.asarray(measured["controls"])
    voltages = jnp.asarray(experiment["voltages_v"])[measured["voltage_indices"]]
    indices = jnp.asarray(measured["voltage_indices"])

    def beams(parameters):
        def predict(command, voltage, index):
            source = jnp.where(
                index == 0, jnp.r_[parameters[0], parameters[13:19]], parameters[22:29]
            )
            effective = condenser_at_voltage(
                parameters,
                voltage,
                reference_voltage_v=experiment["reference_voltage_v"],
                source_parameters=source,
            )
            return predict_beam_coordinates(effective, command)

        return jax.vmap(predict)(controls, voltages, indices).ravel()

    projector, full_covariance = condition_projector(
        projector_mean=measured["projector_solver_parameters"],
        beam_mean=measured["values"].ravel(),
        beam_covariance=measured["covariance"],
        beam_projector_covariance=measured["beam_projector_covariance"],
        projector_covariance=measured["projector_covariance"],
        fitted_beams=np.asarray(beams(p)),
        beam_derivative=np.asarray(jax.jacfwd(beams)(p)),
        condenser_covariance=np.asarray(condenser_fit["covariance"]),
    )
    _, decode, _, _ = projector_coordinates(bounds_path)
    lo, hi = measured["projector_bounds_lower"], measured["projector_bounds_upper"]
    if np.any((projector <= lo) | (projector >= hi)):
        raise RuntimeError(
            "Conditional projector update reached a hard bound; nonlinear joint refinement is needed."
        )
    pp = full_covariance[: len(projector), : len(projector)]
    retained = np.r_[np.arange(16), np.arange(len(projector), len(projector) + len(p))]
    covariance = full_covariance[np.ix_(retained, retained)]
    result = {
        "projector_parameters": np.asarray(
            decode(projector[:16])
            .at[16:19]
            .set(jnp.array([projector[16], projector[17], projector[20]]))
        ).tolist(),
        "projector_solver_parameters": projector[:16].tolist(),
        "projector_calibration_parameters": projector.tolist(),
        "projector_calibration_covariance": pp.tolist(),
        "condenser_parameters": np.asarray(p).tolist(),
        "covariance": covariance.tolist(),
        "parameter_order": "16 projector solver coordinates, then 29 condenser native coordinates, then extra aperture centres in micrometres",
        "alignment_accuracy_assumed": experiment.get(
            "alignment_accuracy_assumed", False
        ),
        "projector_uncertainty_included": True,
        "soft_geometry_priors": False,
        "approximation": "Local Gaussian image-analysis likelihood, full shared calibration covariance. Frozen original-pixel checks are required.",
        "condenser_fit": condenser_fit,
    }
    (root / "full_fit.json").write_text(json.dumps(result, indent=2))
    return result
