"""Fixed objective current, known voltage changes, and independent source states.

The screening calculation borrows reference measurement covariances. It is an
information forecast, not a simulation of new pixels or evidence of uniqueness.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from condenser_aperture_experiment import aperture_specification
from condenser_fitting import beam_to_moments, model_moments
from condenser_rotation_measurements import make_focus_transmission
from finite_projector_pipeline import matrix5 as projector_matrix5
from scipy.constants import c, electron_mass, elementary_charge


def magnetic_strength_ratio(voltage_v, reference_voltage_v=200000.0):
    """p(reference)/p(voltage); fixed magnetic fields have q proportional to 1/p.

    Relativistic kinetic energy gives (pc)^2 = eV(eV + 2mc^2).
    Voltages are positive kinetic energies per electron charge, in volts.
    """
    rest_voltage = electron_mass * c**2 / elementary_charge
    voltage = jnp.asarray(voltage_v)
    reference = jnp.asarray(reference_voltage_v)
    return jnp.sqrt(
        reference
        * (reference + 2 * rest_voltage)
        / (voltage * (voltage + 2 * rest_voltage))
    )


def condenser_at_voltage(
    parameters, voltage_v, *, reference_voltage_v=200000.0, source_parameters=None
):
    """Return 22 condenser/aperture parameters at the recorded voltage.

    Optional source_parameters = [log(source-to-CL1 gap), source x, y,
    theta_x, theta_y, log(size), log(divergence)], in the model's native units.
    Each voltage may have an independent source waist and all six beam values.
    Lens geometry and aperture coordinates remain shared.
    """
    result = jnp.asarray(parameters[:22])
    if source_parameters is not None:
        source = jnp.asarray(source_parameters)
        result = result.at[0].set(source[0]).at[13:19].set(source[1:])
    ratio = magnetic_strength_ratio(voltage_v, reference_voltage_v)
    return result.at[9:13].add(jnp.log(ratio))


def projector_at_voltage(
    parameters, controls, voltage_v, *, reference_voltage_v=200000.0
):
    """Known 5x5 projector map; every magnetic lens sees the new momentum.

    Controls are physical relative currents. Four downstream controls imply
    fixed objective current of one. This does not assume a new objective-current
    response. Calibration validity at the second voltage must be tested separately.
    """
    ratio = magnetic_strength_ratio(voltage_v, reference_voltage_v)
    p = jnp.asarray(parameters).at[11].multiply(ratio).at[12:16].add(jnp.log(ratio))
    return projector_matrix5(p, controls)


def make_voltage_forecast(
    *,
    image_data,
    current_data,
    current_indices,
    voltages_v,
    aperture_indices=(0,),
    independent_sources=True,
):
    """Whiten predicted moments/currents with reference errors at every voltage.

    Inputs: existing measured beam/current covariances and selected controls.
    Output: differentiable prediction vector for a local information forecast.
    Parameter vector: shared first 22; seven source parameters per extra voltage.
    Three condenser currents are adjusted by 1/ratio to keep their nominal
    strengths unchanged; the objective's physical current stays fixed.
    Projector calibration and unchanged precision are optimistic assumptions here.
    """
    voltage_list = tuple(float(v) for v in voltages_v)
    if not voltage_list or min(voltage_list) <= 0:
        raise ValueError("Supply positive accelerating voltages.")
    transform = np.asarray(jax.vmap(jax.jacfwd(beam_to_moments))(image_data["values"]))
    covariance = transform @ image_data["covariance"] @ transform.transpose(0, 2, 1)
    image_weight = jnp.asarray(np.linalg.inv(np.linalg.cholesky(covariance)))
    image_controls = jnp.asarray(image_data["controls"])
    indices = np.asarray(current_indices, dtype=int)
    radii = np.asarray(aperture_indices, dtype=int)
    cov = current_data["covariance"][indices][:, radii][:, :, radii]
    current_weight = jnp.asarray(np.linalg.inv(np.linalg.cholesky(cov)))
    current_controls = jnp.asarray(current_data["controls"][indices])
    operators = []
    for voltage in voltage_list:
        spec = aperture_specification()
        spec["voltage_v"] = voltage
        spec["radii_m"] = [spec["radii_m"][i] for i in radii]
        operators.append(make_focus_transmission(specification=spec))

    def prediction(p):
        parts = []
        for index, (voltage, transmit) in enumerate(zip(voltage_list, operators)):
            source = (
                p[22 + 7 * (index - 1) : 22 + 7 * index]
                if index and independent_sources
                else None
            )
            effective = condenser_at_voltage(
                p,
                voltage,
                reference_voltage_v=voltage_list[0],
                source_parameters=source,
            )
            ratio = magnetic_strength_ratio(voltage, voltage_list[0])
            beam = jax.vmap(model_moments, in_axes=(None, 0))(
                effective, image_controls / ratio
            )
            current = jax.vmap(transmit, in_axes=(None, 0))(
                effective, current_controls / ratio
            )
            parts.extend(
                [
                    jnp.einsum("nij,nj->ni", image_weight, beam).ravel(),
                    jnp.einsum("nij,nj->ni", current_weight, current).ravel(),
                ]
            )
        return jnp.concatenate(parts)

    return jax.jit(prediction)


def simulate_voltage_experiment(
    *,
    protocol_path,
    projector_views_path,
    output_directory,
    focus_counts_path=None,
    voltages_v=(200000.0, 198000.0),
    view_indices=(0, 1, 2),
    random_seed=13001,
    voltage_offset_v=0.0,
    centre_aperture=False,
):
    """Recorded controls and calibrated maps -> fresh MAGICAL pixels and counts.

    One 4 micrometre aperture, 39 focus-scan settings per voltage, and an open
    reference for each reading. Source properties deliberately change between
    voltages. Only this generating function reads generating lens parameters.
    No ellipse measurement or objective-current sweep is acquired.
    """
    from condenser_calibration_model import generating_parameters
    from condenser_image_pipeline import make_image_operator, predict_beam_coordinates
    from condenser_objective_experiment import choose_objective_projector_views
    from magical_projector_pipeline import make_magical_template

    folder = Path(output_directory)
    folder.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(Path(protocol_path).read_text())
    views = json.loads(Path(projector_views_path).read_text())
    (folder / "reference_projector.json").write_text(json.dumps(views, indent=2))
    base_controls = np.asarray(protocol["calibration_controls"])
    if "current_controls" in protocol:
        base_scan = np.asarray(protocol["current_controls"])
    else:
        scan_indices = np.concatenate(
            [np.arange(g * 41 + 14, g * 41 + 27) for g in range(3)]
        )
        base_scan = np.load(focus_counts_path)["controls"][scan_indices]
    base = np.r_[generating_parameters(), np.log(0.0025 / (0.056 - 0.0025)), 0.2, -0.15]
    if centre_aperture:
        from condenser_aperture_alignment import simulate_and_centre_aperture

        base = simulate_and_centre_aperture(
            condenser_parameters=base,
            output_directory=folder / "alignment",
            random_seed=random_seed + 7000,
        )
    source0 = np.r_[base[0], base[13:19]]
    sources = [source0]
    for index in range(1, len(voltages_v)):
        # Independent waist shifts 0.05 mm; size +3%, divergence -2% per step.
        source = source0.copy()
        source[0] = np.log(np.exp(source0[0]) + index * 0.05e-3)
        source[1:5] += index * np.array([0.02, -0.01, 0.001, -0.001])
        source[5:] += index * np.log([1.03, 0.98])
        sources.append(source)
    rng = np.random.default_rng(random_seed)
    for index, voltage in enumerate(voltages_v):
        target = folder / f"voltage_{index}"
        target.mkdir(exist_ok=True)
        ratio = float(magnetic_strength_ratio(voltage, voltages_v[0]))
        actual_voltage = voltage + (voltage_offset_v if index else 0.0)
        effective = condenser_at_voltage(
            base,
            actual_voltage,
            reference_voltage_v=voltages_v[0],
            source_parameters=sources[index],
        )
        calibration = json.loads(json.dumps(views))
        calibration["views"] = [calibration["views"][i] for i in view_indices]
        p = np.asarray(calibration["calibrated_projector_parameters"])
        p[11] *= ratio
        p[12:16] += np.log(ratio)
        calibration["calibrated_projector_parameters"] = p.tolist()
        try:
            maps, projector_controls = choose_objective_projector_views(
                calibration, 1.0
            )
        except RuntimeError:
            # Follow the calibrated branch in small voltage steps when a direct
            # change gives the underdetermined projector solver a poor start.
            continuation = json.loads(json.dumps(views))
            continuation["views"] = [continuation["views"][i] for i in view_indices]
            for intermediate in np.linspace(
                voltages_v[0], voltage, max(3, int(np.ceil(abs(ratio - 1) / 0.002)) + 1)
            )[1:]:
                step_ratio = float(magnetic_strength_ratio(intermediate, voltages_v[0]))
                adjusted = np.asarray(views["calibrated_projector_parameters"]).copy()
                adjusted[11] *= step_ratio
                adjusted[12:16] += np.log(step_ratio)
                continuation["calibrated_projector_parameters"] = adjusted.tolist()
                maps, projector_controls = choose_objective_projector_views(
                    continuation, 1.0
                )
                for view, command in zip(continuation["views"], projector_controls):
                    view["controls"] = command.tolist()
        actual_maps = np.asarray(
            [
                projector_at_voltage(
                    jnp.asarray(views["calibrated_projector_parameters"]),
                    command,
                    actual_voltage,
                    reference_voltage_v=voltages_v[0],
                )
                for command in projector_controls
            ]
        )
        template, _ = make_magical_template(
            voltage, pitch_m=5e-9, illumination_radius_m=130e-9
        )
        detector = {
            "pixel_pitch_m": 220e-6,
            "shape_yx": [64, 64],
            "read_noise_e": 2.0,
            "psf_sigma_pixels": 0.5,
            "sample_kernel_sigma_m": 2.5e-9,
            "voltage_v": voltage,
        }
        render = make_image_operator(
            projector_matrices=actual_maps, template=template, detector=detector
        )
        controls = base_controls / ratio
        # Instrument setup keeps the illuminated specimen feature visible.
        # At a new voltage an old condenser setting can focus to a sub-nm spot.
        # Adjust only such settings, record the mini-lens command, and do not
        # supply a target beam size as an extra observation to the fitter.
        from scipy.optimize import brentq

        beam_size = jax.jit(
            lambda command, effective=effective: jnp.exp(
                predict_beam_coordinates(effective, command)[4]
            )
        )
        adjusted_settings = []
        for row, control in enumerate(controls):
            size = float(beam_size(control))
            if 10e-9 <= size <= 90e-9:
                continue
            target_size = 30e-9 if size < 10e-9 else 60e-9
            grid = control[2] * np.linspace(0.95, 1.05, 41)

            def size_error(
                mini, control=control, target_size=target_size, beam_size=beam_size
            ):
                return (
                    float(beam_size(jnp.asarray(control).at[2].set(mini))) - target_size
                )

            errors = [size_error(value) for value in grid]
            roots = [
                brentq(size_error, a, b, xtol=1e-12)
                for a, b, fa, fb in zip(grid[:-1], grid[1:], errors[:-1], errors[1:])
                if fa * fb < 0
            ]
            if not roots:
                raise RuntimeError(
                    "Could not keep the specimen feature illuminated in the allowed mini-lens range."
                )
            controls[row, 2] = min(roots, key=lambda value: abs(value - control[2]))
            adjusted_settings.append(row)
        if adjusted_settings:
            print(
                f"{voltage / 1000:g} kV: adjusted mini lens for visible illumination at settings {adjusted_settings}",
                flush=True,
            )
        frames = []
        for control in controls:
            beam = predict_beam_coordinates(effective, control)
            exposure = np.tile(np.log([4e7, 3.0]), len(maps))
            expected = np.asarray(render(np.r_[beam, exposure]))
            frames.append(rng.poisson(expected) + rng.normal(0, 2.0, expected.shape))
        np.savez_compressed(
            target / "images.npz",
            images=frames,
            controls=controls,
            projector_matrices=maps,
            projector_controls=projector_controls,
        )
        np.savez_compressed(target / "template.npz", **template)
        (target / "detector.json").write_text(json.dumps(detector, indent=2))
        spec = aperture_specification()
        spec.update(radii_m=[2e-6], voltage_v=actual_voltage)
        transmit = make_focus_transmission(specification=spec)
        scan = base_scan / ratio
        fractions = np.asarray(jax.vmap(transmit, in_axes=(None, 0))(effective, scan))
        if (
            not np.all(np.isfinite(fractions))
            or fractions.min() < 0
            or fractions.max() > 1.000002
        ):
            raise RuntimeError("Invalid aperture integration.")
        incident = 2e7 * np.exp(rng.normal(0, 0.1, len(scan)))
        np.savez_compressed(
            target / "counts.npz",
            controls=scan,
            transmitted_electrons=rng.poisson(incident[:, None] * fractions),
            reference_electrons=rng.poisson(incident),
        )
        spec["voltage_v"] = voltage
        (target / "aperture.json").write_text(json.dumps(spec, indent=2))
        print(
            f"{voltage / 1000:g} kV: {len(frames) * len(maps)} images, {2 * len(scan)} current readings",
            flush=True,
        )
    (folder / "experiment.json").write_text(
        json.dumps(
            {
                "voltages_v": list(voltages_v),
                "reference_voltage_v": voltages_v[0],
                "view_indices": list(view_indices),
                "random_seed": random_seed,
                "objective_current": 1.0,
                "aperture_diameter_um": 4.0,
                "source_parameters_per_voltage": 7,
                "aperture_centred_by_current_scan": centre_aperture,
                "calibration_assumption": "Known projector model and voltage scaling; exact in this first synthetic experiment.",
            },
            indent=2,
        )
    )
    (folder / "scoring_truth.json").write_text(
        json.dumps(
            {
                "parameters": np.r_[base, np.asarray(sources[1:]).ravel()].tolist(),
                "voltage_offset_v": voltage_offset_v,
            },
            indent=2,
        )
    )
    return folder


def analyse_voltage_experiment(*, experiment_directory, calibrate_voltage=False):
    """Pixels/counts -> beam estimates and current fractions, with covariance.

    Analysis uses the calibrated image maps and public specimen. It never reads
    condenser parameters, source parameters, or the scoring truth.
    """
    from condenser_calibration_workflow import measure_images

    folder = Path(experiment_directory)
    experiment = json.loads((folder / "experiment.json").read_text())
    for index in range(len(experiment["voltages_v"])):
        target = folder / f"voltage_{index}"
        measure_images(
            image_directory=target, initialization_only=calibrate_voltage and index > 0
        )
        counts = np.load(target / "counts.npz")
        transmitted = counts["transmitted_electrons"]
        reference = counts["reference_electrons"]
        values = transmitted / reference[:, None]
        covariance = (
            jax.vmap(jnp.diag)(transmitted) / reference[:, None, None] ** 2
            + values[:, :, None] * values[:, None, :] / reference[:, None, None]
        )
        np.savez_compressed(
            target / "current_measurements.npz",
            controls=counts["controls"],
            values=values,
            covariance=covariance,
        )
    return folder


def make_voltage_residual(*, experiment_directory):
    """Independently measured beams/currents -> residual for the unknown model."""
    folder = Path(experiment_directory)
    experiment = json.loads((folder / "experiment.json").read_text())
    blocks = []
    for index, voltage in enumerate(experiment["voltages_v"]):
        target = folder / f"voltage_{index}"
        beam = np.load(target / "measurements.npz")
        transform = np.asarray(jax.vmap(jax.jacfwd(beam_to_moments))(beam["values"]))
        cov = transform @ beam["covariance"] @ transform.transpose(0, 2, 1)
        current = np.load(target / "current_measurements.npz")
        spec = json.loads((target / "aperture.json").read_text())
        blocks.append(
            (
                voltage,
                jnp.asarray(beam["controls"]),
                jax.vmap(beam_to_moments)(beam["values"]),
                jnp.asarray(np.linalg.inv(np.linalg.cholesky(cov))),
                jnp.asarray(current["controls"]),
                jnp.asarray(current["values"]),
                jnp.asarray(np.linalg.inv(np.linalg.cholesky(current["covariance"]))),
                make_focus_transmission(specification=spec),
            )
        )

    def residual(p):
        parts = []
        for index, (voltage, bc, bv, bw, ac, av, aw, transmit) in enumerate(blocks):
            source = p[22 + 7 * (index - 1) : 22 + 7 * index] if index else None
            effective = condenser_at_voltage(
                p,
                voltage,
                reference_voltage_v=experiment["reference_voltage_v"],
                source_parameters=source,
            )
            beam = jax.vmap(model_moments, in_axes=(None, 0))(effective, bc)
            fractions = jax.vmap(transmit, in_axes=(None, 0))(effective, ac)
            parts.extend(
                [
                    jnp.einsum("nij,nj->ni", bw, beam - bv).ravel(),
                    jnp.einsum("nij,nj->ni", aw, fractions - av).ravel(),
                ]
            )
        return jnp.concatenate(parts)

    return jax.jit(residual)


def voltage_starting_guesses(
    *,
    starting_guess_path,
    number_of_voltages=2,
    number_of_broad_starts=4,
):
    """Literal engineering bounds and varied guesses; no measurement or truth is read."""
    from condenser_aperture_experiment import equivalent_aperture_model
    from condenser_calibration_model import initial_parameters
    from scipy.stats import qmc

    earlier = json.loads(Path(starting_guess_path).read_text())
    p = np.asarray(earlier["best_fit"]["parameters"])[:22]
    lower, upper = (
        np.asarray(earlier["bounds_lower"])[:22],
        np.asarray(earlier["bounds_upper"])[:22],
    )

    def extend(base):
        return np.r_[base, np.tile(np.r_[base[0], base[13:19]], number_of_voltages - 1)]

    starts = [extend(p), extend(np.r_[initial_parameters(), -3.0, 0.1, -0.1])]
    starts += [
        extend(equivalent_aperture_model(p, objective_strength=-q))
        for q in [0.35, 0.65]
    ]
    lower, upper = extend(lower), extend(upper)
    if number_of_broad_starts:
        sample = qmc.LatinHypercube(d=len(lower), seed=713).random(
            number_of_broad_starts
        )
        starts += list(qmc.scale(sample, lower + 1e-7, upper - 1e-7))
    return starts, lower, upper


def fit_voltage_experiment(
    *,
    experiment_directory,
    starting_fit_path,
    number_of_broad_starts=4,
    max_evaluations=2000,
):
    """Fit the earlier fixed-readback experiment; retained for comparison."""
    from condenser_geometry_audit import search_alternatives

    folder = Path(experiment_directory)
    starts, lower, upper = voltage_starting_guesses(
        starting_guess_path=starting_fit_path,
        number_of_voltages=len(
            json.loads((folder / "experiment.json").read_text())["voltages_v"]
        ),
        number_of_broad_starts=number_of_broad_starts,
    )
    residual = make_voltage_residual(experiment_directory=folder)
    return search_alternatives(
        residual=residual,
        starting_parameters=starts,
        bounds_lower=lower,
        bounds_upper=upper,
        output_directory=folder / "fit",
        max_evaluations=max_evaluations,
    )


def simulate_voltage_prediction_images(
    *, experiment_directory, protocol_path, random_seed=14001
):
    """Generate six new condenser settings per voltage for frozen prediction checks.

    This is a generator: it may read scoring truth. The analysis and prediction
    checker below do not. New controls are recorded in a separate prediction folder.
    """
    from condenser_image_pipeline import make_image_operator, predict_beam_coordinates

    folder = Path(experiment_directory)
    experiment = json.loads((folder / "experiment.json").read_text())
    truth = json.loads((folder / "scoring_truth.json").read_text())
    p = np.asarray(truth["parameters"])
    base_controls = np.asarray(
        json.loads(Path(protocol_path).read_text())["prediction_controls"]
    )
    rng = np.random.default_rng(random_seed)
    calibration = json.loads((folder / "reference_projector.json").read_text())
    for index, voltage in enumerate(experiment["voltages_v"]):
        reference = folder / f"voltage_{index}"
        target = reference / "prediction"
        target.mkdir(exist_ok=True)
        recorded = np.load(reference / "images.npz")
        maps = recorded["projector_matrices"]
        actual_voltage = voltage + (
            truth.get("voltage_offset_v", 0.0) if index else 0.0
        )
        actual_maps = np.asarray(
            [
                projector_at_voltage(
                    jnp.asarray(calibration["calibrated_projector_parameters"]),
                    command,
                    actual_voltage,
                    reference_voltage_v=experiment["reference_voltage_v"],
                )
                for command in recorded["projector_controls"]
            ]
        )
        template = dict(np.load(reference / "template.npz"))
        detector = json.loads((reference / "detector.json").read_text())
        render = make_image_operator(
            projector_matrices=actual_maps, template=template, detector=detector
        )
        effective = condenser_at_voltage(
            p,
            actual_voltage,
            reference_voltage_v=experiment["reference_voltage_v"],
            source_parameters=p[22 + 7 * (index - 1) : 22 + 7 * index]
            if index
            else None,
        )
        controls = base_controls / float(
            magnetic_strength_ratio(voltage, experiment["reference_voltage_v"])
        )
        frames = []
        for control in controls:
            beam = predict_beam_coordinates(effective, control)
            expected = np.asarray(
                render(np.r_[beam, np.tile(np.log([4e7, 3.0]), len(maps))])
            )
            frames.append(rng.poisson(expected) + rng.normal(0, 2.0, expected.shape))
        np.savez_compressed(
            target / "images.npz",
            images=frames,
            controls=controls,
            projector_matrices=maps,
            projector_controls=recorded["projector_controls"],
        )
        np.savez_compressed(target / "template.npz", **template)
        (target / "detector.json").write_text(json.dumps(detector, indent=2))
        # Independent current predictions also test the upstream scale and aperture.
        spec = json.loads((reference / "aperture.json").read_text())
        spec["voltage_v"] = actual_voltage
        transmit = make_focus_transmission(specification=spec)
        current_controls = np.array(
            [[cl1, cl3, 1.0] for cl3 in [0.65, 0.85, 1.15, 1.4] for cl1 in [1.75, 1.95]]
        )
        current_controls /= float(
            magnetic_strength_ratio(voltage, experiment["reference_voltage_v"])
        )
        fractions = np.asarray(
            jax.vmap(transmit, in_axes=(None, 0))(effective, current_controls)
        )[:, 0]
        incident = 2e7 * np.exp(rng.normal(0, 0.1, len(current_controls)))
        np.savez_compressed(
            target / "current_counts.npz",
            controls=current_controls,
            transmitted_electrons=rng.poisson(incident * fractions),
            reference_electrons=rng.poisson(incident),
        )
    return folder


def check_voltage_predictions(*, experiment_directory, fitted):
    """Frozen fitted geometry -> new pixels and beam estimates; no truth is read.

    Only dose/background are supplied by independent image analysis. Joint beam
    prediction errors include covariance shared through the fitted microscope.
    """
    from condenser_calibration_workflow import measure_images
    from condenser_image_pipeline import make_image_operator, predict_beam_coordinates

    folder = Path(experiment_directory)
    experiment = json.loads((folder / "experiment.json").read_text())
    blocks, pixel_errors = [], []
    p = jnp.asarray(fitted["best_fit"]["parameters"])
    for index, voltage in enumerate(experiment["voltages_v"]):
        target = folder / f"voltage_{index}" / "prediction"
        measured = np.load(measure_images(image_directory=target))
        data = np.load(target / "images.npz")
        controls = jnp.asarray(data["controls"])
        weight = jnp.asarray(np.linalg.inv(np.linalg.cholesky(measured["covariance"])))
        values = jnp.asarray(measured["values"])

        def predict(parameters, index=index, voltage=voltage, controls=controls):
            effective = condenser_at_voltage(
                parameters,
                voltage,
                reference_voltage_v=experiment["reference_voltage_v"],
                source_parameters=parameters[22 + 7 * (index - 1) : 22 + 7 * index]
                if index
                else None,
            )
            return jax.vmap(predict_beam_coordinates, in_axes=(None, 0))(
                effective, controls
            )

        blocks.append(
            lambda parameters, predict=predict, weight=weight, values=values: (
                jnp.einsum("nij,nj->ni", weight, predict(parameters) - values).ravel()
            )
        )
        detector = json.loads((target / "detector.json").read_text())
        render = make_image_operator(
            projector_matrices=data["projector_matrices"],
            template=dict(np.load(target / "template.npz")),
            detector=detector,
        )
        predicted = []
        for row, beam in enumerate(predict(p)):
            image = np.asarray(
                render(jnp.r_[beam, measured["image_parameters"][row, 7:]])
            )
            noise = np.sqrt(
                np.maximum(data["images"][row], 1.0) + detector["read_noise_e"] ** 2
            )
            pixel_errors.append(((image - data["images"][row]) / noise).ravel())
            predicted.append(image)
        np.savez_compressed(target / "frozen_predictions.npz", images=predicted)

    residual = jax.jit(
        lambda parameters: jnp.concatenate([block(parameters) for block in blocks])
    )
    errors = np.asarray(residual(p))
    derivative = np.asarray(jax.jacfwd(residual)(p))
    covariance = (
        np.eye(len(errors))
        + derivative @ np.asarray(fitted["covariance"]) @ derivative.T
    )
    combined = np.linalg.solve(np.linalg.cholesky(covariance), errors)
    result = {
        "pixel_rms": float(np.sqrt(np.mean(np.concatenate(pixel_errors) ** 2))),
        "beam_rms_with_fitted_uncertainty": float(np.sqrt(np.mean(combined**2))),
        "beam_chi_squared": float(combined @ combined),
        "beam_measurement_count": len(combined),
        "geometry_refitted": False,
    }
    (folder / "prediction_check.json").write_text(json.dumps(result, indent=2))
    return result
