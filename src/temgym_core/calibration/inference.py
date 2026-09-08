"""Condenser initialization and joint fitting of the original image measurements."""

import json
import shutil
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import block_diag
from scipy.stats import chi2, qmc

from temgym_core.optimization import LeastSquaresResult, least_squares
from temgym_core.optimization import LocalResidual, fit_shared_least_squares

from .apertures import (
    alignment_residual,
    effective_parameters,
    make_aperture_transmission,
)
from .condenser_images import (
    make_image_operator,
    measure_images,
    predict_beam_coordinates,
)
from .condenser_model import (
    beam_to_moments,
    condenser_at_voltage,
    model_moments,
    physical_geometry,
    projector_at_voltage,
)
from .projector import projector_coordinates, projector_residual


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
        if experiment.get("image_model") == "coherent_gaussian":
            shutil.copyfile(
                original / "moment_specimen.npz", target / "moment_specimen.npz"
            )
            if (original / "moment_specimen_local.npz").exists():
                shutil.copyfile(
                    original / "moment_specimen_local.npz",
                    target / "moment_specimen_local.npz",
                )
            measure_condenser_moments(folder=target)
        else:
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
        if experiment.get("image_model") == "coherent_gaussian":
            predict_moments = make_condenser_moment_predictor(folder=folder)
            commands = jnp.asarray(data["projector_controls"])

            def moment_residual(
                local,
                shared,
                values,
                white,
                predict=predict_moments,
                commands=commands,
                voltage=voltage,
            ):
                p = decode(shared[:16])
                maps = jax.vmap(
                    lambda command: projector_at_voltage(
                        p,
                        command,
                        voltage,
                        reference_voltage_v=experiment["reference_voltage_v"],
                    )
                )(commands)
                return jnp.einsum(
                    "nij,nj->ni", white, predict(local, maps) - values
                ).ravel()

            measured = np.load(folder / "measurements.npz")
            for control, values, covariance, initial in zip(
                data["controls"],
                measured["values"],
                measured["covariance"],
                measured["image_parameters"],
            ):
                white = np.linalg.inv(np.linalg.cholesky(covariance))
                batches.append(
                    LocalResidual(
                        moment_residual, (jnp.asarray(values), jnp.asarray(white))
                    )
                )
                beams.append(initial)
                control_rows.append(control)
                voltage_rows.append(index)
        else:
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
                        p,
                        c,
                        voltage,
                        reference_voltage_v=experiment["reference_voltage_v"],
                    )
                )(commands)
                return render(values[:-common_size], maps).ravel()

            predict = jax.jit(predict)

            def pixel_residual(local, shared, pixels, weight, predict=predict):
                return (predict(jnp.r_[local, shared]) - pixels) * weight

            measured = np.load(folder / "measurements.npz")
            for control, frames, initial in zip(
                data["controls"], data["images"], measured["image_parameters"]
            ):
                pixels = frames.ravel()
                weight = 1 / np.sqrt(
                    np.maximum(pixels, 1.0) + detector["read_noise_e"] ** 2
                )
                batches.append(
                    LocalResidual(
                        pixel_residual, (jnp.asarray(pixels), jnp.asarray(weight))
                    )
                )
                beams.append(initial)
                control_rows.append(control)
                voltage_rows.append(index)
    beams = np.asarray(beams)

    result = fit_shared_least_squares(
        shared_residual=calibration,
        blocks=batches,
        starting_shared=common,
        starting_local=beams,
        shared_bounds=(lower, upper),
        max_iterations=max_iterations,
    )
    common, beams = np.asarray(result.shared), np.asarray(result.local)
    projector_covariance = np.asarray(result.shared_covariance)
    local_covariances, responses = (
        np.asarray(result.local_covariances),
        np.asarray(result.local_responses),
    )
    converged, history = result.success, list(result.history)
    pixel_rms = np.asarray(result.local_residual_rms).tolist()
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
        "measurement_rms": pixel_rms,
        "measurement_kind": "image moments"
        if experiment.get("image_model") == "coherent_gaussian"
        else "pixels",
        "projector_parameters": np.asarray(decode(common[:16])).tolist(),
        "projector_uncertainty_included": True,
        "source_or_condenser_truth_used": False,
        "covariance_scope": (
            "Full local covariance from joint image-moment analysis and projector calibration; calibrated dark background; normalized dose."
            if experiment.get("image_model") == "coherent_gaussian"
            else "Full local covariance after joint pixel analysis and projector response measurements; dose/background marginalised."
        ),
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
    from .apertures import equivalent_aperture_model
    from .condenser_model import initial_parameters

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
    gaussian = experiment.get("image_model") == "coherent_gaussian"
    if result["covariance"] is None or not any(
        r["success"] if gaussian else r["acceptable"] for r in result["starts"]
    ):
        raise RuntimeError(
            "No full-rank condenser fit at noise level. Attempts retained."
        )
    (root / "fit/result.json").write_text(json.dumps(result, indent=2))
    return result


def _refine_with_surrogate(
    *, residual, derivative, surrogate, make_state, start, bounds, max_iterations=15
):
    """Refresh costly image physics outside an exact, cheap nonlinear matrix fit.

    Every accepted step decreases the original measured-moment objective. The
    surrogate shares its value and Jacobian at the current model, and its
    geometry-to-beam mapping remains nonlinear. Final covariance always uses
    the original Jacobian, never the surrogate away from its expansion point.
    """
    x = jnp.asarray(start)
    error = residual(x)
    cost = float(0.5 * jnp.vdot(error, error))
    history, nfev, success = [], 1, False
    for iteration in range(max_iterations):
        state = make_state(x)
        inner = least_squares(
            surrogate,
            x,
            args=state,
            bounds=bounds,
            x_scale="jac",
            max_nfev=2000,
            max_iter=4000,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-5,
            initial_damping=1e-9,
        )
        fraction, accepted = 1.0, False
        while fraction >= 2**-12:
            candidate = x + fraction * (inner.x - x)
            model_error = surrogate(candidate, *state)
            predicted = cost - float(0.5 * jnp.vdot(model_error, model_error))
            trial = residual(candidate)
            nfev += 1
            trial_cost = float(0.5 * jnp.vdot(trial, trial))
            reduction = cost - trial_cost
            if (
                np.isfinite(trial_cost)
                and reduction >= 0
                and predicted >= 0
                and reduction >= 0.1 * predicted
            ):
                accepted = True
                break
            fraction /= 2
        history.append(
            {
                "iteration": iteration,
                "chi_squared": 2 * cost,
                "step_fraction": fraction,
                "accepted": accepted,
                "inner_success": inner.success,
                "inner_evaluations": inner.nfev,
                "actual_reduction": reduction,
                "predicted_reduction": predicted,
            }
        )
        print(
            f"Joint outer iteration {iteration}: chi-squared {2 * trial_cost:.8g}, step {fraction:g}, inner evaluations {inner.nfev}",
            flush=True,
        )
        if not accepted:
            break
        previous_cost = cost
        x, error, cost = candidate, trial, trial_cost
        if (
            inner.success
            and reduction <= 1e-7 * (1 + previous_cost)
            and predicted <= 1e-7 * (1 + previous_cost)
        ):
            success = True
            break
    jac = derivative(x)
    scale = jnp.maximum(jnp.linalg.norm(jac, axis=0), 1e-30)
    gradient = jac.T @ error / scale
    lo, hi = map(jnp.asarray, bounds)
    gradient = jnp.where(
        ((x <= lo) & (gradient > 0)) | ((x >= hi) & (gradient < 0)), 0.0, gradient
    )
    optimality = float(jnp.max(jnp.abs(gradient)))
    if optimality < 1e-5:
        success = True
    return LeastSquaresResult(
        x=x,
        residual=error,
        jacobian=jac,
        cost=cost,
        optimality=optimality,
        success=success,
        status="converged" if success else "not_converged",
        message="Original moment likelihood converged."
        if success
        else "Original moment refinement did not converge.",
        nfev=nfev,
        njev=len(history) + 1,
        nit=len(history),
        history=tuple(history),
    )


def _aperture_centre_branches(*, parameters, residual, bounds, centering_sigma_um):
    """Search small, independent centre problems using calibration currents only."""
    p = jnp.asarray(parameters)
    lower, upper = map(np.asarray, bounds)
    groups = [np.array([20, 21])] + [np.array([i, i + 1]) for i in range(29, len(p), 2)]
    base_cost = float(jnp.sum(residual(p) ** 2))
    branches = []
    for indices in groups:
        idx = jnp.asarray(indices)

        def centre_residual(centre, idx=idx):
            return residual(p.at[idx].set(centre))

        centre_residual = jax.jit(centre_residual)
        jac = jax.jit(jax.jacfwd(centre_residual))
        centres = []
        for dx in (-2.0, 0.0, 2.0):
            for dy in (-2.0, 0.0, 2.0):
                guess = np.clip(
                    np.asarray(p)[indices] + centering_sigma_um * np.array([dx, dy]),
                    lower[indices] + 1e-8,
                    upper[indices] - 1e-8,
                )
                fit = least_squares(
                    centre_residual,
                    guess,
                    jac=jac,
                    bounds=(lower[indices], upper[indices]),
                    x_scale="jac",
                    max_nfev=1000,
                    max_iter=2000,
                    ftol=1e-10,
                    gtol=1e-6,
                )
                centre = np.asarray(fit.x)
                if fit.success and not any(
                    np.linalg.norm(centre - c) < 1e-4 for c in centres
                ):
                    centres.append(centre)
                    branches.append(
                        {
                            "indices": indices.tolist(),
                            "centre_um": centre.tolist(),
                            "conditional_delta_chi_squared": 2 * fit.cost - base_cost,
                        }
                    )
    return branches


def refine_microscope_moments(
    *, analysis_directory, condenser_fit, bounds_path, initial_joint_models=None
):
    """Fit the original image moments jointly, without a Gaussian beam-estimate likelihood.

    The intermediate beam fit supplies starting guesses only. Some angular
    variances have asymmetric uncertainties, so their local covariance cannot
    replace the original measured-moment likelihood for final inference.
    Gaussian images are never rendered here. JAX differentiates each specimen
    quadrature and matrix block; the chain rule joins these small Jacobians.
    """
    root = Path(analysis_directory)
    experiment = json.loads((root / "experiment.json").read_text())
    measured = np.load(root / "joint_beam_measurements.npz")
    common = np.asarray(measured["projector_solver_parameters"])
    ncommon = len(common)
    _, decode, _, _ = projector_coordinates(bounds_path)
    calibration = projector_residual(
        measurements_directory=root.parent.parent / "projector",
        bounds_path=bounds_path,
    )
    marginal = condenser_residual(analysis_directory=root)
    nbeam = measured["values"].size

    @jax.jit
    def currents(parameters):
        return marginal(parameters)[nbeam:]

    blocks, beam_groups = [], []
    for index, voltage in enumerate(experiment["voltages_v"]):
        folder = root / f"voltage_{index}"
        images = np.load(folder / "images.npz")
        moments = np.load(folder / "measurements.npz")
        predict = make_condenser_moment_predictor(folder=folder)
        commands = jnp.asarray(images["projector_controls"])

        def beam_coordinates(parameters, control, index=index, voltage=voltage):
            effective = effective_parameters(
                parameters, voltage, experiment["reference_voltage_v"], index
            )
            return predict_beam_coordinates(effective, control)

        beam_value = jax.jit(beam_coordinates)
        beam_groups.append(
            (
                jax.vmap(beam_coordinates, in_axes=(None, 0)),
                jnp.asarray(images["controls"]),
            )
        )

        def image_residual(
            beam,
            projector,
            value,
            white,
            predict=predict,
            commands=commands,
            voltage=voltage,
        ):
            maps = jax.vmap(
                lambda view_command: projector_at_voltage(
                    decode(projector[:16]),
                    view_command,
                    voltage,
                    reference_voltage_v=experiment["reference_voltage_v"],
                )
            )(commands)
            return jnp.einsum("nij,nj->ni", white, predict(beam, maps) - value).ravel()

        image_value = jax.jit(image_residual)
        image_jac = jax.jit(jax.jacfwd(image_residual, argnums=(0, 1)))
        for command, value, covariance in zip(
            images["controls"], moments["values"], moments["covariance"]
        ):
            blocks.append(
                (
                    beam_value,
                    image_value,
                    image_jac,
                    jnp.asarray(command),
                    jnp.asarray(value),
                    jnp.asarray(np.linalg.inv(np.linalg.cholesky(covariance))),
                )
            )

    evaluation_count = 0

    def residual(parameters):
        nonlocal evaluation_count
        pp, cc = parameters[:ncommon], parameters[ncommon:]
        rows = [calibration(pp), currents(cc)]
        for beam, image, _, command, value, white in blocks:
            rows.append(image(beam(cc, command), pp, value, white))
        values = jnp.concatenate(rows)
        evaluation_count += 1
        print(
            f"Joint evaluation {evaluation_count}: chi-squared {float(jnp.vdot(values, values)):.7g}",
            flush=True,
        )
        return values

    cached_state = {}

    def make_state(parameters):
        if "parameters" in cached_state and np.array_equal(
            parameters, cached_state["parameters"]
        ):
            return cached_state["state"]
        started = time.monotonic()
        pp, cc = parameters[:ncommon], parameters[ncommon:]
        beams, errors, local, shared = [], [], [], []
        for beam, image, ij, command, value, white in blocks:
            b = beam(cc, command)
            ll, gg = ij(b, pp, value, white)
            beams.append(b)
            errors.append(image(b, pp, value, white))
            local.append(ll)
            shared.append(gg)
        state = (
            pp,
            jnp.stack(beams),
            jnp.stack(errors),
            jnp.stack(local),
            jnp.stack(shared),
        )
        jax.block_until_ready(state)
        cached_state.update(parameters=np.asarray(parameters).copy(), state=state)
        print(
            f"Updated image-moment derivatives: {time.monotonic() - started:.2f} seconds",
            flush=True,
        )
        return state

    def surrogate(
        parameters, anchor_projector, anchor_beams, anchor_errors, local, shared
    ):
        pp, cc = parameters[:ncommon], parameters[ncommon:]
        beams = jnp.concatenate([fun(cc, controls) for fun, controls in beam_groups])
        errors = (
            anchor_errors
            + jnp.einsum("nij,nj->ni", local, beams - anchor_beams)
            + jnp.einsum("nij,j->ni", shared, pp - anchor_projector)
        )
        return jnp.concatenate((calibration(pp), currents(cc), errors.ravel()))

    surrogate = jax.jit(surrogate)
    surrogate_jac = jax.jit(jax.jacfwd(surrogate))

    def derivative(parameters):
        return surrogate_jac(parameters, *make_state(parameters))

    lower = np.r_[measured["projector_bounds_lower"], condenser_fit["bounds_lower"]]
    upper = np.r_[measured["projector_bounds_upper"], condenser_fit["bounds_upper"]]
    best_cost = condenser_fit["best_fit"]["chi_squared"]
    candidates = [
        r
        for r in condenser_fit["starts"]
        if r["success"]
        and r["chi_squared"]
        <= best_cost + condenser_fit["acceptance"]["delta_chi_squared_limit"]
    ]
    if initial_joint_models is not None:
        candidates = [
            {
                "index": i,
                "parameters": model[ncommon:],
                "projector_solver_parameters": model[:ncommon],
                "origin": "Continuation from a previously fitted data-only model",
            }
            for i, model in enumerate(initial_joint_models)
        ]
    initial_count = len(candidates)
    aperture_branches = []
    attempts, fitted = [], []
    for record in candidates:
        start = np.r_[
            record.get("projector_solver_parameters", common), record["parameters"]
        ]
        print(
            f"Joint image-moment refinement from start {record['index'] + 1}",
            flush=True,
        )
        fit = _refine_with_surrogate(
            residual=residual,
            derivative=derivative,
            surrogate=surrogate,
            make_state=make_state,
            start=start,
            bounds=(lower, upper),
        )
        jac = np.asarray(fit.jacobian)
        singular = np.linalg.svd(
            jac / np.maximum(np.linalg.norm(jac, axis=0), 1e-30), compute_uv=False
        )
        rank = int(np.sum(singular > singular[0] * 1e-9))
        limit = float(chi2.ppf(0.99, max(len(jac) - rank, 1)))
        attempt = {
            "initial_start_index": record["index"],
            "origin": record.get("origin", "Independent condenser multistart fit"),
            "starting_parameters": start.tolist(),
            "success": fit.success,
            "message": fit.message,
            "parameters": fit.x.tolist(),
            "chi_squared": 2 * fit.cost,
            "residual_rms": float(np.sqrt(np.mean(np.asarray(fit.residual) ** 2))),
            "rank": rank,
            "parameter_count": len(start),
            "goodness_of_fit_limit": limit,
            "acceptable": bool(
                fit.success and rank == len(start) and 2 * fit.cost <= limit
            ),
            "evaluations": fit.nfev,
            "iterations": fit.nit,
            "active_bounds": np.flatnonzero(
                (np.asarray(fit.x) == lower) | (np.asarray(fit.x) == upper)
            ).tolist(),
            "history": list(fit.history),
        }
        attempts.append(attempt)
        fitted.append(fit)
        (root / "joint_refinement.json").write_text(
            json.dumps({"starts": attempts}, indent=2)
        )
        print(
            f"Joint refinement RMS {attempt['residual_rms']:.5g}, rank {rank}, acceptable {attempt['acceptable']}",
            flush=True,
        )
        if len(attempts) == initial_count and any(a["acceptable"] for a in attempts):
            best = min(
                (i for i, a in enumerate(attempts) if a["acceptable"]),
                key=lambda i: attempts[i]["chi_squared"],
            )
            anchor = np.asarray(fitted[best].x)
            alignment = json.loads(
                (root / "alignment/alignment_constraint.json").read_text()
            )
            aperture_branches = _aperture_centre_branches(
                parameters=anchor[ncommon:],
                residual=currents,
                bounds=(lower[ncommon:], upper[ncommon:]),
                centering_sigma_um=alignment["isotropic_sigma_m"] * 1e6,
            )
            for branch in aperture_branches:
                idx = np.asarray(branch["indices"]) + ncommon
                centre = np.asarray(branch["centre_um"])
                duplicate = any(
                    np.linalg.norm(np.asarray(f.x)[idx] - centre) < 0.02 for f in fitted
                )
                if duplicate or branch["conditional_delta_chi_squared"] > float(
                    chi2.ppf(0.95, len(anchor))
                ):
                    continue
                candidate = anchor.copy()
                candidate[idx] = centre
                candidates.append(
                    {
                        "index": len(condenser_fit["starts"]) + len(candidates),
                        "parameters": candidate[ncommon:].tolist(),
                        "projector_solver_parameters": candidate[:ncommon].tolist(),
                        "origin": "Aperture-centre alternative from calibration-current search",
                    }
                )
    acceptable = [i for i, a in enumerate(attempts) if a["acceptable"]]
    if not acceptable:
        raise RuntimeError(
            "Joint fit of the original image moments did not pass; attempts retained."
        )
    winner = min(acceptable, key=lambda i: attempts[i]["chi_squared"])
    fit = fitted[winner]
    _, singular, vectors = np.linalg.svd(np.asarray(fit.jacobian), full_matrices=False)
    full_covariance = (vectors.T / singular) @ (vectors / singular[:, None])
    projector, condenser = np.asarray(fit.x[:ncommon]), np.asarray(fit.x[ncommon:])
    retained = np.r_[np.arange(16), np.arange(ncommon, len(fit.x))]
    # Preserve the public condenser-start report, but acceptance now comes from
    # the original measurement likelihood, including the shared projector.
    final_condenser = dict(condenser_fit)
    final_condenser["intermediate_beam_fit"] = condenser_fit
    final_condenser["starts"] = [
        {
            **a,
            "parameters": a["parameters"][ncommon:],
            "joint_rank": a["rank"],
            "joint_parameter_count": a["parameter_count"],
            "rank": len(condenser),
            "parameter_count": len(condenser),
        }
        for a in attempts
    ]
    final_condenser["best_fit"] = final_condenser["starts"][winner]
    final_condenser["covariance"] = full_covariance[ncommon:, ncommon:].tolist()
    final_condenser["acceptance"] = {
        "goodness_of_fit_limit": attempts[winner]["goodness_of_fit_limit"],
        "scope": "Goodness of fit for the full joint original-moment likelihood, including both microscope halves and all nuisance parameters.",
    }
    ncalibration = len(calibration(jnp.asarray(projector)))
    ncurrents = len(currents(jnp.asarray(condenser)))
    errors = np.asarray(fit.residual)
    residual_blocks = {
        name: {
            "coordinate_count": len(values),
            "rms": float(np.sqrt(np.mean(values**2))),
        }
        for name, values in (
            ("projector", errors[:ncalibration]),
            ("currents_and_alignment", errors[ncalibration : ncalibration + ncurrents]),
            ("condenser_image_moments", errors[ncalibration + ncurrents :]),
        )
    }
    result = {
        "projector_parameters": np.asarray(
            decode(projector[:16])
            .at[16:19]
            .set(jnp.array([projector[16], projector[17], projector[20]]))
        ).tolist(),
        "projector_solver_parameters": projector[:16].tolist(),
        "projector_calibration_parameters": projector.tolist(),
        "projector_calibration_covariance": full_covariance[
            :ncommon, :ncommon
        ].tolist(),
        "condenser_parameters": condenser.tolist(),
        "covariance": full_covariance[np.ix_(retained, retained)].tolist(),
        "parameter_order": "16 projector solver coordinates, then condenser/source/aperture coordinates",
        "alignment_accuracy_assumed": experiment.get(
            "alignment_accuracy_assumed", False
        ),
        "projector_uncertainty_included": True,
        "soft_geometry_priors": False,
        "approximation": "Original image moments and their measured covariance; joint nonlinear matrix fit. Covariance is local. Gaussian pixel prediction checks are required.",
        "joint_refinement": {
            "starts": attempts,
            "best_start": winner,
            "residual_blocks": residual_blocks,
            "aperture_centre_search": aperture_branches,
        },
        "condenser_fit": final_condenser,
    }
    (root / "full_fit.json").write_text(json.dumps(result, indent=2))
    return result


def assemble_full_fit(*, analysis_directory, condenser_fit, bounds_path):
    """Condition the measured projector on the fitted condenser and retain cross-covariance."""
    root = Path(analysis_directory)
    experiment = json.loads((root / "experiment.json").read_text())
    if experiment.get("image_model") == "coherent_gaussian":
        return refine_microscope_moments(
            analysis_directory=root,
            condenser_fit=condenser_fit,
            bounds_path=bounds_path,
        )
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


def make_condenser_moment_predictor(*, folder):
    """Use refined known-specimen quadrature where a very small beam needs it."""
    from temgym_core.constants import energy2wavelength

    from .gaussian import load_known_transmission

    folder = Path(folder)
    detector = json.loads((folder / "detector.json").read_text())
    wide_material = load_known_transmission(folder / "moment_specimen.npz")
    wide = _condenser_moment_predictor_from_material(
        material=wide_material, detector=detector
    )
    local_path = folder / "moment_specimen_local.npz"
    if not local_path.exists():
        return wide
    local_material = load_known_transmission(local_path)
    local = _condenser_moment_predictor_from_material(
        material=local_material, detector=detector
    )
    axis = np.asarray(local_material["axis_m"])
    centre = (axis[0] + axis[-1]) / 2
    half_width = (axis[-1] - axis[0]) / 2 - 2 * (axis[1] - axis[0])
    wavelength = energy2wavelength(float(wide_material["voltage_v"]))

    def predict(beam, maps):
        # The same region must contain the illumination and both shifted
        # specimen points used by every pixel-bin characteristic function.
        separation = (
            0.5
            * wavelength
            / detector["pixel_pitch_m"]
            * jnp.max(jnp.sum(jnp.abs(maps[:, :2, 2:4]), axis=1))
        )
        extent = (
            jnp.max(jnp.abs(beam[:2] * 1e-7 - centre))
            + 8 * jnp.exp(beam[4])
            + separation
        )
        return jax.lax.cond(
            extent < half_width, lambda: local(beam, maps), lambda: wide(beam, maps)
        )

    return predict


def _condenser_moment_predictor_from_material(*, material, detector):
    """Seven specimen-beam coordinates and supplied matrices -> measured moments."""
    from temgym_core.moments import (
        predict_detector_moments,
        gaussian_image_characteristic,
        pixel_bin_correction,
    )
    from temgym_core.constants import energy2wavelength

    sample = material["sampled"]
    wavelength = energy2wavelength(float(material["voltage_v"]))
    if detector["psf_sigma_pixels"] < 0.5 or any(n % 2 for n in detector["shape_yx"]):
        raise ValueError(
            "This pixel correction requires even camera dimensions and a known PSF of at least 0.5 pixel RMS."
        )

    def predict(beam, maps):
        mean = jnp.r_[beam[:2] * 1e-7, beam[2:4] * 1e-3]
        spatial = jnp.exp(2 * beam[4]) * jnp.eye(2)
        curvature = beam[5] * 1e4
        covariance = jnp.block(
            [
                [spatial, curvature * spatial],
                [
                    curvature * spatial,
                    curvature**2 * spatial + jnp.exp(2 * beam[6]) * jnp.eye(2),
                ],
            ]
        )
        exit_mean, exit_covariance = sample.gaussian_moments(
            mean, covariance, wavelength
        )

        def predict_frame(matrix):
            moments = predict_detector_moments(
                matrix,
                exit_mean,
                exit_covariance,
                pixel_pitch_m=detector["pixel_pitch_m"],
                psf_sigma_pixels=detector["psf_sigma_pixels"],
            )

            def characteristic(frequency):
                return gaussian_image_characteristic(
                    sample,
                    material["path_m"],
                    mean,
                    covariance,
                    wavelength,
                    matrix,
                    frequency,
                    pixel_pitch_m=detector["pixel_pitch_m"],
                    psf_sigma_pixels=detector["psf_sigma_pixels"],
                )

            return pixel_bin_correction(moments, characteristic)

        return jax.vmap(predict_frame)(maps)

    return predict


def measure_condenser_moments(*, folder):
    """Measure pixels and initialize independent beams using fitted projector maps."""
    from temgym_core.moments import measure_detector_moments, moment_regions
    from temgym_core.constants import energy2wavelength

    folder = Path(folder)
    data = dict(np.load(folder / "images.npz"))
    detector = json.loads((folder / "detector.json").read_text())
    frames = data["images"].reshape(-1, *data["images"].shape[-2:])
    regions = moment_regions(
        images=frames,
        background_counts=detector["background_counts"],
        read_noise_counts=detector["read_noise_e"],
    )
    measured = measure_detector_moments(
        images=frames,
        background_counts=detector["background_counts"],
        read_noise_counts=detector["read_noise_e"],
        regions=regions,
    )
    values = measured["values"].reshape(len(data["controls"]), -1, 5)
    covariance = measured["covariance"].reshape(len(values), -1, 5, 5)
    maps, pitch = data["projector_matrices"], detector["pixel_pitch_m"]
    a, b = maps[:, :2, 0], maps[:, :2, 2]
    design = np.column_stack(((a * a).sum(1), 2 * (a * b).sum(1), (b * b).sum(1)))
    initial = []
    wave = float(
        energy2wavelength(float(np.load(folder / "moment_specimen.npz")["voltage_v"]))
    )
    for row in values:
        mean = np.linalg.lstsq(
            maps[:, :2, :4].reshape(-1, 4), (row[:, :2] * pitch).ravel(), rcond=None
        )[0]
        shape = (
            (row[:, 2] + row[:, 4]) / 2 - detector["psf_sigma_pixels"] ** 2 - 1 / 12
        ) * pitch**2
        vr, cross, va = np.linalg.lstsq(design, shape, rcond=None)[0]
        vr = max(vr, (5e-9) ** 2)
        h = cross / vr
        conditional = max(va - cross**2 / vr, (wave / (4 * np.pi)) ** 2 / vr * 1.1)
        initial.append(
            np.r_[
                mean[:2] / 1e-7,
                mean[2:] / 1e-3,
                0.5 * np.log(vr),
                h / 1e4,
                0.5 * np.log(conditional),
            ]
        )
    predict = make_condenser_moment_predictor(folder=folder)

    def residual(beam, value, white):
        return jnp.einsum(
            "nij,nj->ni", white, predict(beam, jnp.asarray(maps)) - value
        ).ravel()

    residual = jax.jit(residual)
    derivative = jax.jit(jax.jacfwd(residual))
    white = np.linalg.inv(np.linalg.cholesky(covariance))
    lower = np.r_[[-2.0, -2.0, -10.0, -10.0], np.log(3e-9), -10.0, np.log(1e-7)]
    upper = np.r_[[2.0, 2.0, 10.0, 10.0], np.log(120e-9), 10.0, np.log(0.01)]
    fitted = []
    for start, value, weight in zip(initial, values, white):
        start = np.clip(start, lower + 1e-8, upper - 1e-8)
        fit = least_squares(
            residual,
            start,
            args=(jnp.asarray(value), jnp.asarray(weight)),
            jac=derivative,
            bounds=(lower, upper),
            x_scale="jac",
            max_nfev=500,
            max_iter=600,
            gtol=1e-5,
            ftol=1e-10,
            xtol=1e-10,
        )
        if not fit.success or not np.isfinite(fit.x).all():
            raise RuntimeError(f"Independent beam initialization failed: {fit.message}")
        fitted.append(fit.x)
    np.savez_compressed(
        folder / "measurements.npz",
        values=values,
        covariance=covariance,
        image_parameters=fitted,
        regions=regions.reshape(data["images"].shape),
    )
    return folder


def search_alternatives(
    *,
    residual,
    starting_parameters,
    bounds_lower,
    bounds_upper,
    output_directory,
    max_evaluations=2000,
):
    """Fit every supplied valid guess and retain every result, including failures."""
    folder = Path(output_directory)
    folder.mkdir(parents=True, exist_ok=True)
    derivative = jax.jit(jax.jacfwd(residual))
    lower, upper = np.asarray(bounds_lower), np.asarray(bounds_upper)
    records = []
    for index, guess in enumerate(starting_parameters):
        guess = np.asarray(guess)
        if np.any(guess <= lower) or np.any(guess >= upper):
            raise ValueError(
                "Starting guesses must be strictly inside declared physical bounds."
            )
        fit = least_squares(
            residual,
            guess,
            jac=derivative,
            bounds=(lower, upper),
            max_nfev=max_evaluations,
            max_iter=2 * max_evaluations,
            x_scale="jac",
            ftol=1e-9,
            xtol=1e-11,
            gtol=1e-5,
        )
        record = {
            "index": index,
            "starting_parameters": guess.tolist(),
            "parameters": fit.x.tolist(),
            "geometry": np.asarray(physical_geometry(fit.x)).tolist(),
            "success": bool(fit.success),
            "message": str(fit.message),
            "evaluations": int(fit.nfev),
            "chi_squared": float(2 * fit.cost),
            "residual_rms": float(np.sqrt(np.mean(fit.residual**2))),
            "active_bounds": np.flatnonzero(
                (np.asarray(fit.x) == lower) | (np.asarray(fit.x) == upper)
            ).tolist(),
        }
        records.append(record)
        (folder / f"start_{index:03d}.json").write_text(json.dumps(record, indent=2))
        print(
            f"Start {index + 1}: RMS {record['residual_rms']:.4g}, converged {fit.success}",
            flush=True,
        )
    converged = [r for r in records if r["success"]]
    best = min(converged or records, key=lambda r: r["chi_squared"])
    jac = np.asarray(derivative(np.asarray(best["parameters"])))
    singular = np.linalg.svd(
        jac / np.maximum(np.linalg.norm(jac, axis=0), 1e-30), compute_uv=False
    )
    rank = int(np.sum(singular > singular[0] * 1e-9))
    _, raw_singular, vectors = np.linalg.svd(jac, full_matrices=False)
    covariance = (
        ((vectors.T / raw_singular) @ (vectors / raw_singular[:, None]))
        if rank == len(lower)
        else None
    )
    threshold = float(chi2.ppf(0.99, max(len(jac) - rank, 1)))
    delta = float(chi2.ppf(0.95, rank))
    for r in records:
        r["acceptable"] = (
            r["success"]
            and r["chi_squared"] <= threshold
            and r["chi_squared"] <= best["chi_squared"] + delta
        )
    result = {
        "best_fit": best,
        "starts": records,
        "rank": rank,
        "parameter_count": len(lower),
        "covariance": covariance.tolist() if covariance is not None else None,
        "bounds_lower": lower.tolist(),
        "bounds_upper": upper.tolist(),
        "acceptance": {
            "goodness_of_fit_limit": threshold,
            "delta_chi_squared_limit": delta,
            "scope": "Provisional chi-squared approximations conditional on the noise/model; validate with complete noise repetitions.",
        },
    }
    (folder / "search.json").write_text(json.dumps(result, indent=2))
    return result
