"""Measure a shared voltage correction from pixels, without a condenser model."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from condenser_image_pipeline import make_image_operator
from condenser_voltage_experiment import projector_at_voltage
from scipy.linalg import block_diag


def combine_shared_calibration_covariance(
    local_covariances, local_responses, common_information, beam_parameters=7
):
    """Retain the correlations induced by one shared calibration parameter."""
    conditional = block_diag(
        *[cov[:beam_parameters, :beam_parameters] for cov in local_covariances],
        np.zeros((1, 1)),
    )
    common = np.r_[np.asarray(local_responses)[:, :beam_parameters].ravel(), 1.0]
    return conditional + np.outer(common, common) / common_information


def measure_voltage_correction(
    *,
    image_directory,
    projector_views_path,
    reference_voltage_v=200000.0,
    max_iterations=15,
):
    """Known specimen/projector + pixels -> voltage correction and specimen beams.

    Each condenser setting has an independent beam and exposure. The voltage
    correction is shared by this image set. Block elimination fits that one
    common parameter without constructing a huge pixel Jacobian. The returned
    covariance includes correlations between all beams and the voltage.
    Neither source parameters nor condenser geometry nor scoring truth is read.
    """
    folder = Path(image_directory)
    data = np.load(folder / "images.npz")
    measured = np.load(folder / "measurements.npz")
    detector = json.loads((folder / "detector.json").read_text())
    known = json.loads(Path(projector_views_path).read_text())
    projector = jnp.asarray(known["calibrated_projector_parameters"])
    commands = jnp.asarray(data["projector_controls"])
    nominal_voltage = detector["voltage_v"]
    render = make_image_operator(
        projector_matrices=data["projector_matrices"],
        template=dict(np.load(folder / "template.npz")),
        detector=detector,
    )

    def prediction(values):
        maps = jax.vmap(
            lambda command: projector_at_voltage(
                projector,
                command,
                nominal_voltage + values[-1],
                reference_voltage_v=reference_voltage_v,
            )
        )(commands)
        return render(values[:-1], maps).ravel()

    prediction = jax.jit(prediction)
    derivative = jax.jit(jax.jacfwd(prediction))
    pixels = data["images"].reshape(len(data["images"]), -1)
    weights = 1 / np.sqrt(np.maximum(pixels, 1.0) + detector["read_noise_e"] ** 2)
    beams = np.asarray(measured["image_parameters"]).copy()
    correction = 0.0

    def cost(parameters, delta):
        return sum(
            float(
                np.sum(
                    (
                        (np.asarray(prediction(np.r_[row, delta])) - pixels[i])
                        * weights[i]
                    )
                    ** 2
                )
            )
            for i, row in enumerate(parameters)
        )

    history = []
    converged = False
    for iteration in range(max_iterations):
        normal, rhs, steps, couplings, covariances = 0.0, 0.0, [], [], []
        old_cost = 0.0
        for i, row in enumerate(beams):
            value = np.r_[row, correction]
            error = (np.asarray(prediction(value)) - pixels[i]) * weights[i]
            jacobian = np.asarray(derivative(value)) * weights[i, :, None]
            design, voltage_column = jacobian[:, :-1], jacobian[:, -1]
            u, s, vt = np.linalg.svd(design, full_matrices=False)
            inverse = (vt.T / s) @ u.T
            projected_error = error - design @ (inverse @ error)
            projected_voltage = voltage_column - design @ (inverse @ voltage_column)
            normal += float(projected_voltage @ projected_voltage)
            rhs += float(projected_voltage @ projected_error)
            steps.append(-inverse @ error)
            couplings.append(-inverse @ voltage_column)
            covariances.append((vt.T / s) @ (vt / s[:, None]))
            old_cost += float(error @ error)
        if not normal > 0 or not np.isfinite(normal):
            raise RuntimeError("The images do not identify a voltage correction.")
        delta_step = -rhs / normal
        beam_step = np.asarray(steps) + np.asarray(couplings) * delta_step
        fraction = 1.0
        while fraction >= 2**-14:
            trial_beams, trial_correction = (
                beams + fraction * beam_step,
                correction + fraction * delta_step,
            )
            new_cost = cost(trial_beams, trial_correction)
            if np.isfinite(new_cost) and new_cost <= old_cost + 1e-7:
                break
            fraction /= 2
        history.append(
            {
                "iteration": iteration,
                "chi_squared": old_cost,
                "correction_v": correction,
                "proposed_step_v": delta_step,
                "step_fraction": fraction,
            }
        )
        print(
            f"Voltage iteration {iteration}: correction {correction:.6g} V, pixel RMS {np.sqrt(old_cost / pixels.size):.4f}",
            flush=True,
        )
        if fraction < 2**-14:
            raise RuntimeError("Voltage calibration line search did not converge.")
        beams, correction = trial_beams, trial_correction
        if abs(fraction * delta_step) < 1e-5 and abs(new_cost - old_cost) < 1e-7 * max(
            old_cost, 1
        ):
            converged = True
            break
    # Covariance comes from the last linearisation; convergence makes its
    # difference from the final parameter point negligible.
    covariance = combine_shared_calibration_covariance(covariances, couplings, normal)
    np.savez_compressed(
        folder / "voltage_calibrated_measurements.npz",
        controls=data["controls"],
        values=np.r_[beams[:, :7].ravel(), correction],
        covariance=covariance,
        image_parameters=beams,
        nominal_voltage_v=nominal_voltage,
    )
    result = {
        "converged": converged,
        "correction_v": float(correction),
        "voltage_sigma_v": float(1 / np.sqrt(normal)),
        "pixel_rms": float(np.sqrt(new_cost / pixels.size)),
        "history": history,
        "condenser_model_used": False,
    }
    (folder / "voltage_calibration.json").write_text(json.dumps(result, indent=2))
    return result


def make_calibrated_voltage_residual(*, experiment_directory, use_moments=False):
    """Joint measured beams/voltage/current -> condenser fit with calibration uncertainty.

    Reference-voltage images keep their ordinary covariance. At the second
    voltage, the shared voltage correction and all measured beams retain their
    full joint covariance. The last model parameter is that correction in volts.
    """
    from condenser_fitting import beam_to_moments, model_moments
    from condenser_image_pipeline import predict_beam_coordinates
    from condenser_rotation_measurements import make_focus_transmission
    from condenser_voltage_experiment import condenser_at_voltage

    folder = Path(experiment_directory)
    experiment = json.loads((folder / "experiment.json").read_text())
    if len(experiment["voltages_v"]) != 2:
        raise ValueError("This initial calibration test uses exactly two voltages.")
    blocks = []
    alignment_path = folder / "alignment/alignment_constraint.json"
    alignment = (
        json.loads(alignment_path.read_text())
        if experiment.get("aperture_centred_by_current_scan", False)
        else None
    )
    for index, voltage in enumerate(experiment["voltages_v"]):
        target = folder / f"voltage_{index}"
        if index:
            calibration = json.loads((target / "voltage_calibration.json").read_text())
            if not calibration["converged"]:
                raise ValueError("Voltage image calibration has not converged.")
            beam = np.load(target / "voltage_calibrated_measurements.npz")
            values = jnp.asarray(beam["values"])
            covariance = beam["covariance"]
            if use_moments:
                n = len(beam["controls"])

                def convert(v, n=n):
                    return jnp.r_[
                        jax.vmap(beam_to_moments)(v[:-1].reshape(n, 7)).ravel(), v[-1]
                    ]

                transform = np.asarray(jax.jacfwd(convert)(values))
                covariance = transform @ covariance @ transform.T
                values = convert(values)
        else:
            beam = np.load(target / "measurements.npz")
            covariance = block_diag(*beam["covariance"])
            values = jnp.asarray(beam["values"]).ravel()
            if use_moments:
                transform = block_diag(
                    *np.asarray(jax.vmap(jax.jacfwd(beam_to_moments))(beam["values"]))
                )
                covariance = transform @ covariance @ transform.T
                values = jax.vmap(beam_to_moments)(beam["values"]).ravel()
        current = np.load(target / "current_measurements.npz")
        blocks.append(
            (
                voltage,
                jnp.asarray(beam["controls"]),
                values,
                jnp.asarray(np.linalg.inv(np.linalg.cholesky(covariance))),
                jnp.asarray(current["controls"]),
                jnp.asarray(current["values"]),
                jnp.asarray(np.linalg.inv(np.linalg.cholesky(current["covariance"]))),
                make_focus_transmission(
                    specification=json.loads((target / "aperture.json").read_text())
                ),
            )
        )

    def residual(p):
        parts = []
        for index, (voltage, bc, bv, bw, ac, av, aw, transmit) in enumerate(blocks):
            effective = condenser_at_voltage(
                p,
                voltage + (p[29] if index else 0.0),
                reference_voltage_v=experiment["reference_voltage_v"],
                source_parameters=p[22:29] if index else None,
            )
            predict = model_moments if use_moments else predict_beam_coordinates
            beam = jax.vmap(predict, in_axes=(None, 0))(effective, bc).ravel()
            if index:
                beam = jnp.r_[beam, p[29]]
            current = jax.vmap(transmit, in_axes=(None, 0))(effective, ac)
            parts.extend(
                [bw @ (beam - bv), jnp.einsum("nij,nj->ni", aw, current - av).ravel()]
            )
        if alignment is not None:
            from condenser_aperture_experiment import aperture_moments

            mean, _ = aperture_moments(
                p[:22], jnp.asarray(alignment["lens_controls"]), "after"
            )
            parts.append(
                (mean - jnp.asarray(alignment["mean_minus_aperture_centre_m"]))
                / alignment["isotropic_sigma_m"]
            )
        return jnp.concatenate(parts)

    return jax.jit(residual)


def fit_calibrated_voltage_geometry(
    *,
    experiment_directory,
    starting_fit_path,
    max_evaluations=2000,
    output_name="calibrated_fit_beam_coordinates",
):
    """Fit the geometry and the independently measured voltage correction together."""
    from condenser_geometry_audit import search_alternatives

    folder = Path(experiment_directory)
    earlier = json.loads(Path(starting_fit_path).read_text())
    delta = json.loads((folder / "voltage_1/voltage_calibration.json").read_text())[
        "correction_v"
    ]
    # Preserve diverse original starting guesses, including the engineering one.
    starts = [
        np.r_[record["starting_parameters"], delta] for record in earlier["starts"]
    ]
    lower, upper = (
        np.r_[earlier["bounds_lower"], -100.0],
        np.r_[earlier["bounds_upper"], 100.0],
    )
    return search_alternatives(
        residual=make_calibrated_voltage_residual(experiment_directory=folder),
        starting_parameters=starts,
        bounds_lower=lower,
        bounds_upper=upper,
        output_directory=folder / output_name,
        max_evaluations=max_evaluations,
    )


def fit_calibrated_experiment(
    *,
    experiment_directory,
    starting_guess_path,
    number_of_broad_starts=4,
    max_evaluations=2000,
):
    """One bounded fitting procedure, retaining initialization and every final attempt.

    Moment coordinates provide starting candidates. Final comparisons always
    use the measured beam coordinates and their shared calibration covariance.
    Promising unfinished attempts get a further solver budget; failures remain
    in the record. No soft prior or hidden geometry is supplied.
    """
    from condenser_geometry_audit import search_alternatives
    from condenser_voltage_experiment import voltage_starting_guesses

    folder = Path(experiment_directory)
    starts, lower, upper = voltage_starting_guesses(
        starting_guess_path=starting_guess_path,
        number_of_broad_starts=number_of_broad_starts,
    )
    correction = json.loads(
        (folder / "voltage_1/voltage_calibration.json").read_text()
    )["correction_v"]
    starts = [np.r_[p, correction] for p in starts]
    lower, upper = np.r_[lower, -100.0], np.r_[upper, 100.0]
    output = folder / "geometry_fit"
    warm = search_alternatives(
        residual=make_calibrated_voltage_residual(
            experiment_directory=folder, use_moments=True
        ),
        starting_parameters=starts,
        bounds_lower=lower,
        bounds_upper=upper,
        output_directory=output / "initialization",
        max_evaluations=max_evaluations,
    )
    residual = make_calibrated_voltage_residual(experiment_directory=folder)
    # Bound-active initial candidates are moved just inside for the next solver.
    seeds = [
        np.clip(r["parameters"], lower + 1e-9, upper - 1e-9) for r in warm["starts"]
    ]
    fitted = search_alternatives(
        residual=residual,
        starting_parameters=seeds,
        bounds_lower=lower,
        bounds_upper=upper,
        output_directory=output / "attempts",
        max_evaluations=max_evaluations,
    )
    pending = [
        r for r in fitted["starts"] if not r["success"] and r["residual_rms"] < 2
    ]
    attempts = list(fitted["starts"])
    if pending:
        refined = search_alternatives(
            residual=residual,
            starting_parameters=[
                np.clip(r["parameters"], lower + 1e-9, upper - 1e-9) for r in pending
            ],
            bounds_lower=lower,
            bounds_upper=upper,
            output_directory=output / "refinements",
            max_evaluations=3 * max_evaluations,
        )
        for record, original in zip(refined["starts"], pending):
            record["refines_attempt"] = original["index"]
        attempts.extend(refined["starts"])
        if (
            refined["best_fit"]["success"]
            and refined["best_fit"]["chi_squared"] < fitted["best_fit"]["chi_squared"]
        ):
            fitted = refined
    fitted["starts"] = attempts
    fitted["initialization"] = warm
    for record in attempts:
        record["acceptable"] = bool(
            record["success"]
            and record["chi_squared"] <= fitted["acceptance"]["goodness_of_fit_limit"]
            and record["chi_squared"]
            <= fitted["best_fit"]["chi_squared"]
            + fitted["acceptance"]["delta_chi_squared_limit"]
        )
    fitted["measurement_coordinates"] = (
        "beam coordinates with a shared voltage correction"
    )
    fitted["initial_guess_count"] = len(starts)
    (output / "search.json").write_text(json.dumps(fitted, indent=2))
    return fitted


def check_calibrated_predictions(*, experiment_directory, fitted):
    """Predict new pixels with geometry and voltage fixed; fit only dose/background."""
    from condenser_image_pipeline import predict_beam_coordinates
    from condenser_rotation_measurements import make_focus_transmission
    from condenser_voltage_experiment import condenser_at_voltage
    from scipy.optimize import nnls

    folder = Path(experiment_directory)
    experiment = json.loads((folder / "experiment.json").read_text())
    known = json.loads((folder / "reference_projector.json").read_text())
    p = jnp.asarray(fitted["best_fit"]["parameters"])
    errors, nuisance_count = [], 0
    current_errors, current_variances, current_derivatives = [], [], []
    for index, voltage in enumerate(experiment["voltages_v"]):
        target = folder / f"voltage_{index}" / "prediction"
        data = np.load(target / "images.npz")
        detector = json.loads((target / "detector.json").read_text())
        corrected_voltage = voltage + (float(p[29]) if index else 0.0)
        maps = np.asarray(
            [
                projector_at_voltage(
                    jnp.asarray(known["calibrated_projector_parameters"]),
                    command,
                    corrected_voltage,
                    reference_voltage_v=experiment["reference_voltage_v"],
                )
                for command in data["projector_controls"]
            ]
        )
        render = make_image_operator(
            projector_matrices=maps,
            template=dict(np.load(target / "template.npz")),
            detector=detector,
        )
        effective = condenser_at_voltage(
            p,
            corrected_voltage,
            reference_voltage_v=experiment["reference_voltage_v"],
            source_parameters=p[22:29] if index else None,
        )
        predicted_images = []
        for control, frames in zip(data["controls"], data["images"]):
            beam = predict_beam_coordinates(effective, control)
            density = np.asarray(render(jnp.r_[beam, jnp.zeros(2 * len(maps))])) - 1.0
            predicted = []
            for shape, observed in zip(density, frames):
                noise = np.sqrt(
                    np.maximum(observed, 1.0) + detector["read_noise_e"] ** 2
                )
                design = (
                    np.stack([shape.ravel(), np.ones(shape.size)], axis=1)
                    / noise.ravel()[:, None]
                )
                scales = np.linalg.norm(design, axis=0)
                exposure, _ = nnls(design / scales, observed.ravel() / noise.ravel())
                exposure /= scales
                expected = exposure[0] * shape + exposure[1]
                predicted.append(expected)
                errors.append(((expected - observed) / noise).ravel())
                nuisance_count += 2
            predicted_images.append(predicted)
        np.savez_compressed(target / "frozen_predictions.npz", images=predicted_images)
        counts = np.load(target / "current_counts.npz")
        transmitted = counts["transmitted_electrons"].astype(float)
        reference = counts["reference_electrons"].astype(float)
        measured = transmitted / reference
        variance = transmitted / reference**2 + transmitted**2 / reference**3
        spec = json.loads((target.parent / "aperture.json").read_text())
        transmit = make_focus_transmission(specification=spec)
        controls = jnp.asarray(counts["controls"])

        def predict_current(
            parameters,
            voltage=voltage,
            index=index,
            transmit=transmit,
            controls=controls,
        ):
            effective = condenser_at_voltage(
                parameters,
                voltage + (parameters[29] if index else 0.0),
                reference_voltage_v=experiment["reference_voltage_v"],
                source_parameters=parameters[22:29] if index else None,
            )
            return jax.vmap(transmit, in_axes=(None, 0))(effective, controls)[:, 0]

        current_errors.append(np.asarray(predict_current(p)) - measured)
        current_variances.append(variance)
        current_derivatives.append(np.asarray(jax.jacfwd(predict_current)(p)))
    error = np.concatenate(errors)
    result = {
        "pixel_rms": float(np.sqrt(np.mean(error**2))),
        "pixel_chi_squared": float(error @ error),
        "pixel_count": len(error),
        "fitted_exposure_parameters": nuisance_count,
        "geometry_refitted": False,
        "voltage_refitted": False,
        "scope": "Frozen pixel prediction. Only independent dose/background terms are fitted to new frames.",
    }
    difference = np.concatenate(current_errors)
    derivative = np.concatenate(current_derivatives)
    covariance = np.diag(np.concatenate(current_variances))
    covariance += derivative @ np.asarray(fitted["covariance"]) @ derivative.T
    whitened = np.linalg.solve(np.linalg.cholesky(covariance), difference)
    result.update(
        current_prediction_rms=float(np.sqrt(np.mean(whitened**2))),
        current_prediction_chi_squared=float(whitened @ whitened),
        current_prediction_count=len(difference),
        current_prediction_covariance="Independent count noise plus shared fitted-parameter covariance; no refitting.",
    )
    (folder / "prediction_check.json").write_text(json.dumps(result, indent=2))
    return result
