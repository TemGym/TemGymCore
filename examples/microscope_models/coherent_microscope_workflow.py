"""One coherent full-microscope example; all propagation/fitting lives in TemGym."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from temgym_core.coherent_calibration import (
    CalibrationState,
    Parameterization,
    RecordedSettings,
    fit_microscope,
    make_predictor,
    microscope_parameters,
    simulate_observations,
)
from temgym_core.coherent_microscope import (
    WaveNumerics,
    microscope_from_dict,
    microscope_to_dict,
)
from temgym_core.jax_least_squares import least_squares
from temgym_core.multiscale_sample import MAGICALSampleAtlas, SampleView
from temgym_core.solenoid import column_matrix

HERE = Path(__file__).parent


def make_magical_sample(voltage=200000.0):
    """Prescribed complex MAGICAL transmission, including optical path in metres."""
    atlas = MAGICALSampleAtlas.nominal(
        shape_yx=(384, 384),
        region_shape_yx=(128, 128),
        authoritative_multilayer_shape_yx=(256, 256),
        max_levels=3,
        voltage=voltage,
    )
    certificate = replace(
        atlas.certificate,
        features=tuple(
            replace(f, phase_amplitude_rad=0.0, log_amplitude_depth=0.0)
            for f in atlas.certificate.features
        ),
    )
    atlas = MAGICALSampleAtlas.from_certificate(
        certificate,
        shape_yx=(384, 384),
        region_shape_yx=(128, 128),
        authoritative_multilayer_shape_yx=(256, 256),
        max_levels=3,
        global_support_margin_m=atlas.global_support_margin_m,
    )
    sample = SampleView.from_level(atlas.pyramid.levels[0], z=0.0)
    region = next(r for r in certificate.regions if r.name == "multilayer_region_3")
    roi = np.array([4e-9, region.centre_xy_m[1] + region.size_xy_m[1] / 2 - 2e-9])
    return sample, certificate, roi


def load_model(path):
    return microscope_from_dict(json.loads(Path(path).read_text()))


def experiment_numerics(*, refinement=1):
    """Follow a four-waist source footprint; refine the Gaussian count separately."""
    return WaveNumerics(
        specimen_half_width=80e-9,
        beamlet_width=8e-9 / refinement,
        overlap=1.2,
        detector_shape=(12, 12),
        detector_pitch=220e-6,
        pixel_samples=2,
        follow_source=True,
        radial_order=96,
        azimuthal_order=64,
    )


def design_recorded_settings(generating_model):
    """Synthetic setup only: centre/focus views as an operator would, record commands.

    Only these recorded commands go to the fitter. Objective controls remain one.
    Two views per voltage have 40k/60k scale and small opposite defocus; CL1/CL3
    and mini controls vary. This small acquisition tests integration, not a
    measurement design already established to determine every physical parameter.
    """
    images, currents, predictions = [], [], []
    model = generating_model
    projector_seed = jnp.array(
        [
            1.0,
            0.7878671830443049,
            0.8960533754728769,
            0.7326077442697242,
            1.0895397789606636,
        ]
    )
    for index, voltage in enumerate((200000.0, 195000.0)):
        for view in range(3):
            command = jnp.array([1.0 + 0.1 * view, 1.0 - 0.08 * view, 1.25, 1.0])

            # A finite Gaussian waist near the specimen, with a small deliberate
            # offset from exact source imaging. This does not change body geometry.
            def focus(mini, command=command, index=index, voltage=voltage):
                cc = command.at[2].set(mini[0])
                source_z = (
                    -model.condenser.extent - model.sources[index].upstream_distance
                )
                m = column_matrix(
                    model.condenser_components(cc, voltage), source_z=source_z
                )
                q = (
                    jnp.sum(model.condenser.strengths * cc)
                    * model.condenser.components(
                        cc, voltage, model.reference_voltage, 0.0
                    )[0].q
                    / (model.condenser.strengths[0] * cc[0])
                )
                # Trace after undoing rotation, signed radial B.
                r = jnp.array([[jnp.cos(q), -jnp.sin(q)], [jnp.sin(q), jnp.cos(q)]])
                return jnp.array([(r @ m[:2, 2:4])[0, 0] / 0.001 - 0.5])

            result = least_squares(
                focus, jnp.array([1.25]), bounds=([0.8], [1.8]), max_nfev=100
            )
            if not result.success or np.linalg.norm(result.residual) > 1e-5:
                raise RuntimeError("Synthetic condenser setup did not focus.")
            command = command.at[2].set(result.x[0])
            target_mag = 40000.0 if view == 0 else 60000.0
            defocus = (-4.0 if view == 0 else 4.0) * 1e-6

            def imaging(knobs, voltage=voltage, target_mag=target_mag, defocus=defocus):
                pc = projector_seed.at[1:].set(knobs)
                m = column_matrix(model.projector_components(pc, voltage))
                a, b = m[:2, :2], m[:2, 2:4]
                return jnp.array(
                    [
                        jnp.log(jnp.linalg.norm(a[:, 0]) / target_mag),
                        (jnp.trace(jnp.linalg.solve(a, b)) / 2 - defocus) / 1e-4,
                    ]
                )

            result = least_squares(
                imaging,
                projector_seed[1:],
                bounds=(0.05, 2.5),
                max_nfev=1200,
                initial_damping=1e-7,
                ftol=1e-12,
                gtol=1e-10,
            )
            if not result.success or np.linalg.norm(result.residual) > 1e-5:
                raise RuntimeError(
                    f"Synthetic projector setup did not focus: {result.message}, {result.residual}"
                )
            pc = projector_seed.at[1:].set(result.x)
            row = {
                "voltage": voltage,
                "source_index": index,
                "condenser": np.asarray(command).tolist(),
                "projector": np.asarray(pc).tolist(),
                "tilt_xy": [0.0, 0.0],
                "shift_xy": [0.0, 0.0],
            }
            if view < 2:
                images.append(row)
                images.append(
                    dict(
                        row,
                        tilt_xy=[0.001, 0.0] if view == 0 else [0.0, 0.001],
                        shift_xy=[5e-9, 0.0] if view == 0 else [0.0, 5e-9],
                    )
                )
                for hole in (-1, 0, 1):
                    currents.append(
                        {
                            "voltage": voltage,
                            "source_index": index,
                            "condenser": row["condenser"],
                            "hole": hole,
                        }
                    )
            else:
                predictions.append(row)
    return RecordedSettings(tuple(images), tuple(currents)), RecordedSettings(
        tuple(predictions), ()
    )


def acquisition_state(model, number_of_images):
    return CalibrationState(
        model,
        jnp.full(number_of_images, 2e6),
        jnp.full(number_of_images, 3.0),
        jnp.full(len(model.sources), 1e7),
    )


def simulate_experiment(
    *, generating_model_path, output_directory, numerical, random_seed=9101
):
    """Read generating geometry; save image/current arrays and recorded settings.

    Returns only the data directory. Fitting reads no generating model.
    """
    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    model = load_model(generating_model_path)
    settings, prediction_settings = design_recorded_settings(model)
    sample, _, roi = make_magical_sample()
    sample = replace(sample, centre_xy_m=jnp.asarray(-roi))
    for name, rows in (("calibration", settings), ("prediction", prediction_settings)):
        predictor = make_predictor(rows, sample, numerical)
        state = acquisition_state(model, len(rows.images))
        data = simulate_observations(
            predictor, state, random_seed=random_seed + (name == "prediction")
        )
        np.savez_compressed(root / f"{name}.npz", **data)
        (root / f"{name}_settings.json").write_text(
            json.dumps({"images": rows.images, "currents": rows.currents}, indent=2)
        )
    (root / "generation.json").write_text(
        json.dumps(
            {
                "random_seed": random_seed,
                "generating_model": microscope_to_dict(model),
                "numerical": {
                    **asdict(numerical),
                    "number_of_gaussians": len(numerical.centres),
                },
            },
            indent=2,
        )
    )
    return root


def load_measurements(directory, name="calibration"):
    root = Path(directory)
    rows = json.loads((root / f"{name}_settings.json").read_text())
    return RecordedSettings(tuple(rows["images"]), tuple(rows["currents"])), dict(
        np.load(root / f"{name}.npz")
    )


def fit_experiment(
    *,
    measurement_directory,
    starting_model_path,
    numerical,
    number_of_starts=2,
    max_evaluations=100,
):
    """Supply measured arrays, independent geometry, bounds and numerical accuracy.

    Return the fitted TemGym state and all-start diagnostics. All nine lens
    gains/lengths, gaps, source states and hole positions are free. Image/current
    exposures are nuisance parameters; detector background is assumed measured.
    The fully coherent example fixes additional incoherent angular spread at zero.
    """
    settings, data = load_measurements(measurement_directory)
    sample, _, roi = make_magical_sample()
    sample = replace(sample, centre_xy_m=jnp.asarray(-roi))
    initial = acquisition_state(load_model(starting_model_path), len(settings.images))
    parameters = Parameterization(
        initial, microscope_parameters(initial, fit_exposures=True)
    )
    rng = np.random.default_rng(471)
    starts = [parameters.initial]
    for _ in range(number_of_starts - 1):
        starts.append(
            np.clip(
                parameters.initial + rng.normal(0, 0.04, len(parameters.initial)),
                *parameters.bounds,
            )
        )
    fitted, report = fit_microscope(
        predictor=make_predictor(settings, sample, numerical),
        observations=data,
        parameterization=parameters,
        starting_coordinates=starts,
        max_evaluations=max_evaluations,
    )
    root = Path(measurement_directory)
    (root / "fit.json").write_text(json.dumps(report, indent=2))
    (root / "fitted_model.json").write_text(
        json.dumps(microscope_to_dict(fitted.microscope), indent=2)
    )
    np.savez(
        root / "fitted_acquisition.npz",
        image_dose=fitted.image_dose,
        image_background=fitted.image_background,
        current_dose=fitted.current_dose,
    )
    return fitted, report


def check_experiment(*, measurement_directory, fitted_state, fit_report, numerical):
    """Read truth only AFTER fitting; also predict unused settings and refine waves."""
    root = Path(measurement_directory)
    sample, _, roi = make_magical_sample()
    sample = replace(sample, centre_xy_m=jnp.asarray(-roi))
    settings, observed = load_measurements(root, "prediction")
    calibration_settings, _ = load_measurements(root)
    calibration_sources = np.asarray(
        [row["source_index"] for row in calibration_settings.images]
    )
    # Predict unused images at the fitted exposure for that voltage. Their
    # observed pixels and the hidden generating exposures never set brightness.
    prediction_dose = jnp.asarray(
        [
            np.median(
                np.asarray(fitted_state.image_dose)[
                    calibration_sources == row["source_index"]
                ]
            )
            for row in settings.images
        ]
    )
    state = replace(
        acquisition_state(fitted_state.microscope, len(settings.images)),
        image_dose=prediction_dose,
        current_dose=fitted_state.current_dose,
    )
    predicted, _ = make_predictor(settings, sample, numerical)(state)
    fine = replace(
        numerical,
        beamlet_width=numerical.beamlet_width / 2,
        radial_order=2 * numerical.radial_order,
        azimuthal_order=2 * numerical.azimuthal_order,
    )
    refined, _ = make_predictor(settings, sample, fine)(state)
    truth = microscope_from_dict(
        json.loads((root / "generation.json").read_text())["generating_model"]
    )
    # Check the actual acquisition too, not only a conveniently chosen new view.
    calibration_settings, _ = load_measurements(root)
    checks_by_state = {}
    for label, candidate in (
        ("generating", acquisition_state(truth, len(calibration_settings.images))),
        ("fitted", fitted_state),
    ):
        coarse_images, coarse_currents = make_predictor(
            calibration_settings, sample, numerical
        )(candidate)
        fine_images, fine_currents = make_predictor(calibration_settings, sample, fine)(
            candidate
        )
        differences = np.linalg.norm(
            np.asarray(fine_images - coarse_images).reshape(
                len(calibration_settings.images), -1
            ),
            axis=1,
        )
        denominators = np.linalg.norm(
            np.asarray(fine_images - candidate.image_background[:, None, None]).reshape(
                len(calibration_settings.images), -1
            ),
            axis=1,
        )
        checks_by_state[label] = {
            "worst_image_refinement_change": float(
                np.max(differences / np.maximum(denominators, 1e-100))
            ),
            "worst_current_refinement_change": float(
                np.max(
                    np.abs(np.asarray(fine_currents - coarse_currents))
                    / np.maximum(np.asarray(fine_currents), 1e-100)
                )
            ),
        }
    pixel_refined = replace(numerical, pixel_samples=2 * numerical.pixel_samples)
    reference_state = acquisition_state(truth, len(calibration_settings.images))
    pixels_coarse, _ = make_predictor(calibration_settings, sample, numerical)(
        reference_state
    )
    pixels_fine, _ = make_predictor(calibration_settings, sample, pixel_refined)(
        reference_state
    )
    pixel_change = float(
        np.linalg.norm(pixels_coarse - pixels_fine)
        / np.linalg.norm(pixels_fine - reference_state.image_background[:, None, None])
    )
    params = microscope_parameters(acquisition_state(truth, len(settings.images)))
    truth_values = Parameterization(
        acquisition_state(truth, len(settings.images)), params
    ).values(acquisition_state(truth, len(settings.images)))
    # Scoring does not feed errors back to the solver or choose starting models.
    fitted_values = Parameterization(
        acquisition_state(truth, len(settings.images)), params
    ).values(state)
    errors = [
        {
            "name": name,
            "truth": value,
            "fitted": fitted_values[name],
            "error": fitted_values[name] - value,
        }
        for name, value in truth_values.items()
    ]
    check = {
        "prediction_residual_rms": float(
            np.sqrt(
                np.mean(
                    (
                        (np.asarray(predicted) - observed["images"])
                        / observed["image_sigma"]
                    )
                    ** 2
                )
            )
        ),
        "refinement_relative_change": float(
            np.linalg.norm(np.asarray(refined) - predicted) / np.linalg.norm(refined)
        ),
        "geometry_errors": errors,
        "acquisition_refinement": checks_by_state,
        "pixel_refinement_change": pixel_change,
        "data_fit_accepted": fit_report.get("data_fit_accepted", False),
        "solver_converged": fit_report["starts"][fit_report["best_index"]]["success"],
        "rank": fit_report["rank"],
        "number_of_parameters": fit_report["number_of_parameters"],
        "scope": "Small coherent integration experiment; neither recovery nor global uniqueness is assumed.",
    }
    check["sampling_passed"] = (
        check["refinement_relative_change"] < 0.005
        and pixel_change < 0.005
        and all(
            row["worst_image_refinement_change"] < 0.005
            and row["worst_current_refinement_change"] < 0.005
            for row in checks_by_state.values()
        )
    )
    check["calibration_accepted"] = bool(
        check["data_fit_accepted"]
        and check["sampling_passed"]
        and check["prediction_residual_rms"] < 1.2
    )
    check["unique_geometry_established"] = False
    (root / "checks.json").write_text(json.dumps(check, indent=2))
    np.savez_compressed(
        root / "prediction_comparison.npz",
        observed=observed["images"],
        predicted=predicted,
        refined=refined,
    )
    return check


def show_experiment(measurement_directory, report=None, checks=None):
    import matplotlib.pyplot as plt

    settings, data = load_measurements(measurement_directory)
    fig, axes = plt.subplots(2, 4, figsize=(11, 5))
    for ax, row, frame in zip(axes.ravel(), settings.images, data["images"]):
        ax.imshow(frame, origin="lower", cmap="gray")
        ax.set_title(f"{row['voltage'] / 1000:g} kV")
        ax.axis("off")
    fig.tight_layout()
    if report is not None:
        print(
            f"{report['number_of_parameters']} fitted parameters; local rank {report['rank']}."
        )
        for row in report["starts"]:
            print(
                f"Start {row['index'] + 1}: converged={row['success']}, residual RMS={row['residual_rms']:.3f}"
            )
    if checks is not None:
        print(
            f"New-image residual RMS: {checks['prediction_residual_rms']:.3f} (noise should be about 1)."
        )
        print(
            f"Gaussian refinement change: {checks['refinement_relative_change']:.2%}; target <0.5%."
        )
        print(
            f"Sampling accepted: {checks['sampling_passed']}; calibration accepted: {checks['calibration_accepted']}."
        )
        print("A converged fit does not by itself establish unique geometry.")
    return fig


TILT_PARAMETER_NAMES = ("detector_gap_m", "IL1_gain", "PL1_gain")
TILT_PARAMETER_INDICES = (5, 12, 15)


def generate_gaussian_tilt_experiment(
    *, reference_experiment, output_directory, tilt_mrad=1.0, random_seed=8021
):
    """Save 12 native Gaussian images; only this step reads generating optics.

    Three recorded projector settings, with independent +/- x and y tilt
    pairs. A fixed prescribed specimen exit wave receives a uniform phase
    ramp; its amplitude is unchanged. Apertures are retracted. This first
    bridge test varies three unknown projector parameters, with all remaining
    geometry supplied as known. It is not a full source/condenser inversion.
    """
    from time import perf_counter

    import jax
    from coherent_forward_validation import (
        _recorded_gaussian_case,
        matched_exit_gaussians,
    )

    from temgym_core.coherent_image import blurred_gaussian_image
    from temgym_core.evaluate import gpu_kernel_available
    from temgym_core.gaussian import FreeSpacePropagator
    from temgym_core.run import run_to_end

    if not np.isfinite(tilt_mrad) or not 0 < tilt_mrad <= 1:
        raise ValueError("Use tilts between zero and 1 mrad in this detector field.")
    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    reference = Path(reference_experiment)
    truth = np.asarray(
        json.loads((reference / "projector/scoring_truth.json").read_text())[
            "parameters"
        ]
    )
    controls = np.load(reference / "condenser/voltage_0/images.npz")[
        "projector_controls"
    ]
    method = "spatial" if gpu_kernel_available() else "scan"
    template, detector, beam, _, _, _ = _recorded_gaussian_case(
        reference,
        0,
        3,
        0,
        (256, 256),
        2,
    )
    source, _ = matched_exit_gaussians(
        template=template,
        detector=detector,
        beam_coordinates=beam,
        step_nm=0.5,
    )
    rng = np.random.default_rng(random_seed)
    dose, background, read_noise = 4e7, 3.0, 2.0
    images, means, commands, angles, labels, timings = [], [], [], [], [], []
    for view in range(len(controls)):
        _, _, _, column, sensor, blur = _recorded_gaussian_case(
            reference,
            0,
            3,
            view,
            (256, 256),
            2,
        )
        sensor = replace(sensor, pixel_pitch_m=55e-6)

        def forward(input_beam, tilt, column=column, sensor=sensor, blur=blur):
            phase = input_beam.k * (input_beam.x * tilt[0] + input_beam.y * tilt[1])
            tilted = input_beam.derive(
                dx=input_beam.dx + tilt[0],
                dy=input_beam.dy + tilt[1],
                amplitude=input_beam.amplitude * jnp.exp(1j * phase),
            )
            output = run_to_end(
                tilted,
                column,
                propagator=FreeSpacePropagator(include_axial_path=False),
            )
            return blurred_gaussian_image(
                output,
                sensor,
                blur_sigma_m=blur,
                padding_pixels=16,
                method=method,
            )

        render = jax.jit(forward)
        for axis, name in zip(np.eye(2), ("x", "y")):
            for sign in (-1, 1):
                tilt = sign * tilt_mrad * 1e-3 * axis
                start = perf_counter()
                image = (
                    dose * np.asarray(render(source, jnp.asarray(tilt))) + background
                )
                timings.append(perf_counter() - start)
                means.append(image)
                images.append(
                    rng.poisson(image).astype(float)
                    + rng.normal(0, read_noise, image.shape)
                )
                commands.append(controls[view])
                angles.append(tilt)
                labels.append(f"View {view}, {name} tilt {sign * tilt_mrad:+g} mrad")
        print(f"Generated Gaussian tilt pairs at projector view {view}.", flush=True)
    fixed = truth.tolist()
    for index in TILT_PARAMETER_INDICES:
        fixed[index] = None  # The fitted values are absent from known settings.
    settings = {
        "voltage_v": 200000.0,
        "pixel_pitch_m": 55e-6,
        "shape_yx": [256, 256],
        "background_counts": background,
        "read_noise_counts": read_noise,
        "lens_controls": np.asarray(commands).tolist(),
        "tilts_xy_rad": np.asarray(angles).tolist(),
        "labels": labels,
        "fixed_projector_parameters": fixed,
        "fitted_parameter_names": list(TILT_PARAMETER_NAMES),
        "assumptions": "Calibrated pure specimen tilt; unchanged exit-wave amplitude; no aperture clipping or frame drift; remaining projector parameters known.",
    }
    np.savez_compressed(root / "images.npz", images=images)
    (root / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    np.savez_compressed(root / "scoring_images.npz", mean_images=means)
    (root / "scoring_truth.json").write_text(
        json.dumps(
            {
                "projector_parameters": truth.tolist(),
                "fitted_values": dict(
                    zip(
                        TILT_PARAMETER_NAMES,
                        np.exp(truth[list(TILT_PARAMETER_INDICES)]).tolist(),
                    )
                ),
                "device": jax.devices()[0].device_kind,
                "method": method,
                "gaussians": int(source.x.size),
                "frame_seconds_including_compilation": timings,
                "random_seed": random_seed,
            },
            indent=2,
        )
        + "\n"
    )
    return root


def measure_gaussian_tilt_images(*, experiment_directory):
    """Read pixels and detector calibration; return measured shifts/covariances."""
    from temgym_core.coherent_calibration import measure_image_centroids

    root = Path(experiment_directory)
    settings = json.loads((root / "settings.json").read_text())
    images = np.load(root / "images.npz")["images"]
    measured = measure_image_centroids(
        images=images,
        background_counts=settings["background_counts"],
        read_noise_counts=settings["read_noise_counts"],
    )
    centres, covariance = measured["centres_xy_pixels"], measured["covariances_pixels2"]
    result = {
        "displacements_xy_pixels": centres[1::2] - centres[::2],
        "covariances_pixels2": covariance[1::2] + covariance[::2],
        "signal_counts": measured["signal_counts"],
    }
    np.savez_compressed(root / "measured_shifts.npz", **result)
    return result


def fit_gaussian_tilt_measurements(
    *,
    measurements,
    known_settings,
    starting_parameters,
    parameter_bounds,
    additional_starts=(),
):
    """Fit physical parameters using native TemGym matrices, without images.

    Inputs contain measured shifts, their full 2x2 covariance, recorded lens
    controls/tilts, known fixed optics, and independent physical guesses/bounds.
    No generating model or Gaussian renderer is accessed here.
    """
    from time import perf_counter

    from finite_projector_pipeline import predict_tilt_displacements

    settings = known_settings
    indices = np.asarray(TILT_PARAMETER_INDICES)
    fixed = settings["fixed_projector_parameters"]
    if any(fixed[index] is not None for index in indices):
        raise ValueError("Unknown parameters must be absent from known settings.")
    base = jnp.asarray([0.0 if value is None else value for value in fixed])
    controls = jnp.asarray(settings["lens_controls"])[::2]
    tilts = jnp.asarray(settings["tilts_xy_rad"])
    changes = tilts[1::2] - tilts[::2]
    observed = jnp.asarray(measurements["displacements_xy_pixels"])
    cholesky = jnp.linalg.cholesky(jnp.asarray(measurements["covariances_pixels2"]))

    def predictions(log_parameters):
        full = base.at[indices].set(log_parameters)
        return predict_tilt_displacements(
            full, controls, changes, settings["pixel_pitch_m"]
        )

    def residual(log_parameters):
        difference = predictions(log_parameters) - observed
        return jnp.linalg.solve(cholesky, difference[..., None]).ravel()

    lower = np.log([parameter_bounds[name][0] for name in TILT_PARAMETER_NAMES])
    upper = np.log([parameter_bounds[name][1] for name in TILT_PARAMETER_NAMES])
    rows, solutions = [], []
    start = perf_counter()
    for guess in (starting_parameters, *additional_starts):
        initial = np.log([guess[name] for name in TILT_PARAMETER_NAMES])
        result = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            x_scale="jac",
            max_nfev=300,
            max_iter=300,
            gtol=1e-6,
        )
        solutions.append(result)
        rows.append(
            {
                "starting_parameters": guess,
                "fitted_parameters": dict(
                    zip(TILT_PARAMETER_NAMES, np.exp(np.asarray(result.x)).tolist())
                ),
                "success": bool(result.success),
                "message": result.message,
                "scaled_gradient": float(result.optimality),
                "residual_rms": float(
                    np.sqrt(np.mean(np.asarray(result.residual) ** 2))
                ),
            }
        )
    best_index = int(
        np.argmin([np.sum(np.asarray(item.residual) ** 2) for item in solutions])
    )
    best = solutions[best_index]
    jacobian = np.asarray(best.jacobian)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    covariance_log = np.linalg.inv(jacobian.T @ jacobian)
    values = np.exp(np.asarray(best.x))
    covariance_physical = values[:, None] * covariance_log * values[None, :]
    return {
        "starts": rows,
        "best_index": best_index,
        "fitted_parameters": rows[best_index]["fitted_parameters"],
        "parameter_std": dict(
            zip(TILT_PARAMETER_NAMES, np.sqrt(np.diag(covariance_physical)).tolist())
        ),
        "predicted_shifts_xy_pixels": np.asarray(predictions(best.x)).tolist(),
        "jacobian_singular_values": singular_values.tolist(),
        "local_rank": int(np.linalg.matrix_rank(jacobian)),
        "fit_seconds_including_compilation": perf_counter() - start,
        "gaussian_images_rendered_during_fit": 0,
        "scope": "Three-parameter projector recovery, with remaining optics known; not full microscope recovery or global uniqueness.",
    }


def check_gaussian_tilt_fit(*, experiment_directory, measurements, fit_result):
    """Only scoring reads the true optics and noiseless generating images."""
    from finite_projector_pipeline import predict_tilt_displacements

    from temgym_core.coherent_calibration import measure_image_centroids

    root = Path(experiment_directory)
    settings = json.loads((root / "settings.json").read_text())
    truth = json.loads((root / "scoring_truth.json").read_text())
    angles = np.asarray(settings["tilts_xy_rad"])
    expected = np.asarray(
        predict_tilt_displacements(
            jnp.asarray(truth["projector_parameters"]),
            jnp.asarray(settings["lens_controls"])[::2],
            jnp.asarray(angles[1::2] - angles[::2]),
            settings["pixel_pitch_m"],
        )
    )
    means = np.load(root / "scoring_images.npz")["mean_images"]
    centres = measure_image_centroids(
        images=means,
        background_counts=settings["background_counts"],
        read_noise_counts=0,
    )["centres_xy_pixels"]
    noiseless_shifts = centres[1::2] - centres[::2]
    chol = np.linalg.cholesky(measurements["covariances_pixels2"])

    def rms(error):
        return float(np.sqrt(np.mean(np.linalg.solve(chol, error[..., None]) ** 2)))

    errors = {
        name: fit_result["fitted_parameters"][name] - value
        for name, value in truth["fitted_values"].items()
    }
    signal = means - settings["background_counts"]
    edge = np.ones(signal.shape[1:], dtype=bool)
    edge[8:-8, 8:-8] = False
    starts = np.array(
        [
            [row["fitted_parameters"][name] for name in TILT_PARAMETER_NAMES]
            for row in fit_result["starts"]
        ]
    )
    report = {
        "image_count": len(means),
        "pair_count": len(expected),
        "true_parameters": truth["fitted_values"],
        "parameter_errors": errors,
        "mean_image_shift_bias_rms_noise": rms(noiseless_shifts - expected),
        "mean_image_shift_max_error_pixels": float(
            np.max(np.abs(noiseless_shifts - expected))
        ),
        "measured_shift_error_rms_noise": rms(
            measurements["displacements_xy_pixels"] - expected
        ),
        "maximum_edge_signal_fraction": float(
            np.max(signal[:, edge].sum(axis=1) / signal.sum(axis=(1, 2)))
        ),
        "maximum_relative_spread_between_starts": float(
            np.max(np.ptp(starts, axis=0) / starts.mean(axis=0))
        ),
        "fit": fit_result,
        "generation": {
            key: truth[key]
            for key in (
                "device",
                "method",
                "gaussians",
                "frame_seconds_including_compilation",
            )
        },
        "scope": fit_result["scope"],
    }
    report["measurement_bridge_passed"] = bool(
        report["mean_image_shift_bias_rms_noise"] < 0.2
    )
    report["recovery_passed"] = bool(
        report["measurement_bridge_passed"]
        and fit_result["local_rank"] == 3
        and all(row["success"] for row in fit_result["starts"])
        and fit_result["starts"][fit_result["best_index"]]["residual_rms"] < 2
        and abs(errors["detector_gap_m"]) < 0.1e-3
        and abs(errors["IL1_gain"]) / truth["fitted_values"]["IL1_gain"] < 0.01
        and abs(errors["PL1_gain"]) / truth["fitted_values"]["PL1_gain"] < 0.01
        and report["maximum_relative_spread_between_starts"] < 1e-4
    )
    (root / "recovery.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def show_gaussian_tilt_recovery(*, report):
    """Compact notebook result with physical units and explicit recovery scope."""
    from IPython.display import Markdown, display

    rows = []
    for name in TILT_PARAMETER_NAMES:
        scale = 1000 if name == "detector_gap_m" else 1
        label = "Detector gap (mm)" if scale == 1000 else name.replace("_", " ")
        true = report["true_parameters"][name] * scale
        fitted = report["fit"]["fitted_parameters"][name] * scale
        uncertainty = report["fit"]["parameter_std"][name] * scale
        rows.append(
            f"| {label} | {true:.6g} | {fitted:.6g} | {fitted - true:+.3g} | {uncertainty:.3g} |"
        )
    display(
        Markdown(
            "| Parameter | True | Fitted | Error | Local standard error |\n"
            "|---|---:|---:|---:|---:|\n" + "\n".join(rows)
        )
    )
    print(
        f"Gaussian-image shift bias: {report['mean_image_shift_bias_rms_noise']:.3g} measurement noise units."
    )
    print(
        f"Matrix fit time, including compilation: {report['fit']['fit_seconds_including_compilation']:.2f} s."
    )
    print(
        f"Recovery passed: {report['recovery_passed']}. Remaining projector parameters were held fixed."
    )


def show_gaussian_tilt_images(*, experiment_directory):
    """Show the x-tilt pairs at the three recorded projector settings."""
    import matplotlib.pyplot as plt

    root = Path(experiment_directory)
    data = np.load(root / "images.npz")["images"]
    settings = json.loads((root / "settings.json").read_text())
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharex=True, sharey=True)
    for view in range(3):
        vmax = np.percentile(data[4 * view : 4 * view + 2], 99.8)
        for sign_index in range(2):
            index = 4 * view + sign_index
            ax = axes[sign_index, view]
            ax.imshow(data[index], origin="lower", cmap="gray", vmin=0, vmax=vmax)
            ax.set_title(settings["labels"][index])
            ax.set_xlabel("Detector x (pixels)")
            ax.set_ylabel("Detector y (pixels)")
    fig.tight_layout()
    return fig


def prepare_known_transmission(
    *, template, voltage_v, kernel_sigma_m, step_nm=0.5, half_width_nm=192.0
):
    """Sample the same prescribed MAGICAL amplitude and phase for both paths.

    The amplitude target is the earlier known smoothed intensity template;
    the optical path is the native MAGICAL specimen. No unknown microscope
    quantity enters this preparation. Derivatives are taken from the known
    specimen, not estimated from the generating beam.
    """
    import jax

    from temgym_core.coherent_moments import SampledTransmission

    half = int(np.ceil(half_width_nm / step_nm))
    axis = np.arange(-half, half + 1) * step_nm * 1e-9
    old = np.asarray(template["axis_m"])
    delta = axis[:, None] - old[None, :]
    variance = kernel_sigma_m**2
    g = np.exp(-0.5 * delta**2 / variance) / np.sqrt(2 * np.pi * variance)
    gradient = -delta * g / variance
    intensity = g @ template["intensity"] @ g.T
    dx = g @ template["intensity"] @ gradient.T
    dy = gradient @ template["intensity"] @ g.T
    xy = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
    sample, _, roi = make_magical_sample(voltage_v)

    def actions(point):
        return sample.evaluate_actions(point + jnp.asarray(roi))[0]

    path = jax.jit(jax.vmap(actions))(jnp.asarray(xy))
    path_gradient = jax.jit(jax.vmap(jax.jacfwd(actions)))(jnp.asarray(xy))
    amplitude_gradient = 0.5 * np.stack((dx, dy), axis=-1).reshape(-1, 2)
    amplitude_gradient /= np.maximum(intensity.ravel(), 1e-300)[:, None]
    maximum = intensity.max()
    sampled = SampledTransmission(
        jnp.asarray(xy),
        jnp.asarray(np.maximum(intensity.ravel() / maximum, 1e-300)),
        jnp.asarray(amplitude_gradient),
        path_gradient,
    )
    return {
        "sampled": sampled,
        "path_m": np.asarray(path),
        "axis_m": axis,
        "voltage_v": voltage_v,
        "step_nm": step_nm,
    }


def gaussian_exit_wave(
    *,
    transmission,
    profile,
    position_m=(0.0, 0.0),
    angle_rad=(0.0, 0.0),
    curvature_per_m=0.0,
    rms_size_m=None,
    radius_m=None,
    edge_m=None,
    specimen_rotation_rad=0.0,
    overlap=1.1,
):
    """Represent a prescribed coherent exit wave with actual TemGym Gaussians.

    Beamlets share a common quadratic phase; cardinal coefficients represent
    the complex specimen and illumination amplitude. Padding to size buckets
    reuses compiled GPU kernels while zero-amplitude entries carry no field.
    """
    from temgym_core.constants import energy2wavelength
    from temgym_core.gaussian import make_gaussian
    from temgym_core.source import _gaussian_cardinal_coefficients

    sampled = transmission["sampled"]
    c, s = np.cos(specimen_rotation_rad), np.sin(specimen_rotation_rad)
    rotation = np.array([[c, -s], [s, c]])
    xy = np.asarray(sampled.coordinates_m) @ rotation.T
    delta = xy - np.asarray(position_m)
    if profile == "gaussian":
        illumination = np.exp(-0.5 * np.sum(delta**2, axis=1) / rms_size_m**2)
    elif profile == "circle":
        from scipy.special import expit

        illumination = expit((radius_m - np.linalg.norm(delta, axis=1)) / edge_m)
    else:
        raise ValueError("Choose gaussian or circle illumination.")
    intensity = np.asarray(sampled.intensity) * illumination
    step = transmission["step_nm"] * 1e-9
    intensity /= intensity.sum() * step**2
    k = 2 * np.pi / float(energy2wavelength(transmission["voltage_v"]))
    shape = (len(transmission["axis_m"]),) * 2
    target = (np.sqrt(intensity) * np.exp(1j * k * transmission["path_m"])).reshape(
        shape
    )
    coefficients = _gaussian_cardinal_coefficients(
        target, width_over_step=overlap
    ).ravel()
    keep = np.flatnonzero(np.abs(coefficients) > np.max(np.abs(coefficients)) * 1e-12)
    count = len(keep)
    size = (
        4096 * 2 ** int(np.ceil(np.log2(max(1.0, count / 4096))))
        if count < 65536
        else 65536 * int(np.ceil(count / 65536))
    )
    selected = keep[np.arange(size) % count]
    amplitude = coefficients[selected].copy()
    amplitude[count:] = 0
    positions = xy[selected]
    theta = np.asarray(angle_rad) - curvature_per_m * np.asarray(position_m)
    slopes = curvature_per_m * positions + theta
    phase = k * (
        0.5 * curvature_per_m * np.sum(positions**2, axis=1) + positions @ theta
    )
    return make_gaussian(
        x=positions[:, 0],
        y=positions[:, 1],
        dx=slopes[:, 0],
        dy=slopes[:, 1],
        amp=amplitude,
        phase=phase,
        waist_x=overlap * step,
        waist_y=overlap * step,
        rcurv_x=np.inf if curvature_per_m == 0 else 1 / curvature_per_m,
        rcurv_y=np.inf if curvature_per_m == 0 else 1 / curvature_per_m,
        voltage=transmission["voltage_v"],
    )


def condenser_gaussian_state(
    *, parameters, controls, voltage_v, aperture_fraction=None
):
    """Generate the specimen state by native Gaussian source/lens propagation."""
    from temgym_core.coherent_microscope import (
        GaussianSourceState,
        propagate_gaussian_mixture,
    )
    from temgym_core.constants import energy2wavelength
    from temgym_core.solenoid import solenoid_column

    p = jnp.asarray(parameters)
    gaps, lengths = jnp.exp(p[:5]), jnp.exp(p[5:9])
    strengths = jnp.exp(p[9:13]) * jnp.r_[jnp.asarray(controls), -1.0]
    z = -(jnp.sum(gaps) + jnp.sum(lengths))
    source = GaussianSourceState(
        upstream_distance=gaps[0],
        x=p[13] * 1e-8,
        y=p[14] * 1e-8,
        theta_x=p[15] * 1e-4,
        theta_y=p[16] * 1e-4,
        rms_size=jnp.exp(p[17]),
    ).beam(z, voltage_v, jnp.zeros(2))
    floor = (energy2wavelength(voltage_v) / (4 * jnp.pi * jnp.exp(p[17]))) ** 2
    extra = jnp.eye(4).at[:2, :2].set(0) * (jnp.exp(2 * p[18]) - floor)
    column = tuple(
        replace(lens, include_axial_path=False) if hasattr(lens, "q") else lens
        for lens in solenoid_column(gaps, lengths, strengths, source_z=z)
    )
    if aperture_fraction is not None:
        from temgym_core.components import Plane

        aperture_z = column[1].z + column[1].length + aperture_fraction * gaps[2]
        column = (*column[:2], Plane(z=aperture_z))
    return propagate_gaussian_mixture(source, column, extra)


def save_known_transmission(path, transmission):
    """Save only known specimen/grid quantities, never lens or source parameters."""
    t = transmission["sampled"]
    np.savez_compressed(
        path,
        coordinates_m=t.coordinates_m,
        intensity=t.intensity,
        log_amplitude_gradient=t.log_amplitude_gradient,
        path_gradient=t.path_gradient,
        path_m=transmission["path_m"],
        axis_m=transmission["axis_m"],
        voltage_v=transmission["voltage_v"],
        step_nm=transmission["step_nm"],
    )


def load_known_transmission(path):
    from temgym_core.coherent_moments import SampledTransmission

    data = dict(np.load(path))
    sampled = SampledTransmission(
        *(
            jnp.asarray(data[name])
            for name in (
                "coordinates_m",
                "intensity",
                "log_amplitude_gradient",
                "path_gradient",
            )
        )
    )
    return {
        "sampled": sampled,
        **{
            name: data[name]
            for name in (
                "path_m",
                "axis_m",
                "voltage_v",
                "step_nm",
            )
        },
    }


def make_gaussian_renderer(
    *,
    shape_yx,
    pixel_pitch_m,
    samples_per_pixel,
    psf_sigma_pixels=0.0,
    padding_pixels=8,
):
    """Native field renderer with optional padded regions inside the full camera.

    Region choice only saves work in negligible signal tails. Pixel pitch and
    coherent quadrature stay fixed; verify the omitted signal independently.
    """
    import jax

    from temgym_core.coherent_image import CoherentDetector, blurred_gaussian_image
    from temgym_core.evaluate import gpu_kernel_available
    from temgym_core.gaussian import FreeSpacePropagator
    from temgym_core.run import run_to_end

    method = "spatial" if gpu_kernel_available() else "scan"
    compiled = {}

    def render(
        source, components, angular_variance, offset_pixels, *, region_shape_yx=None
    ):
        shape = tuple(shape_yx if region_shape_yx is None else region_shape_yx)
        if any(n > full or n < 2 or (full - n) % 2 for n, full in zip(shape, shape_yx)):
            raise ValueError(
                "Use a centred region with even padding inside the camera."
            )
        if shape not in compiled:
            sensor = CoherentDetector(shape, pixel_pitch_m, 0.0, samples_per_pixel)

            def evaluate(source, components, angular_variance, offset_pixels):
                output = run_to_end(
                    source,
                    components,
                    propagator=FreeSpacePropagator(include_axial_path=False),
                )
                output = output.derive(
                    x=output.x + offset_pixels[0] * pixel_pitch_m,
                    y=output.y + offset_pixels[1] * pixel_pitch_m,
                )
                matrix = column_matrix(components)
                blur = jnp.sqrt(
                    angular_variance * jnp.sum(matrix[:2, 2] ** 2)
                    + (psf_sigma_pixels * pixel_pitch_m) ** 2
                )
                return blurred_gaussian_image(
                    output,
                    sensor,
                    blur_sigma_m=blur,
                    padding_pixels=padding_pixels,
                    method=method,
                )

            compiled[shape] = jax.jit(evaluate)
        frame = compiled[shape](source, components, angular_variance, offset_pixels)
        return jnp.pad(
            frame,
            tuple(
                ((full - n) // 2, (full - n) // 2) for n, full in zip(shape, shape_yx)
            ),
        )

    return render


def gaussian_rendering_region(*, detector_moments, shape_yx, standard_deviations=8.0):
    """Propose a centred integration region; convergence checks remain mandatory."""
    value = np.asarray(detector_moments)
    extent = np.abs(value[:2]) + standard_deviations * np.sqrt(
        np.maximum(value[[2, 4]], 0)
    )
    width = max(32, 16 * int(np.ceil(2 * extent.max() / 16)))
    return tuple(min(width, n) for n in shape_yx)


def replace_projector_with_gaussian_images(
    *,
    directory,
    random_seed=41001,
    step_nm=0.5,
    shape_yx=(256, 256),
    samples_per_pixel=4,
):
    """Generate all original projector controls, tilts and shifts with Gaussians."""
    from time import perf_counter

    import jax
    from finite_projector_pipeline import projector_components

    root = Path(directory)
    experiment = json.loads((root / "experiment.json").read_text())
    report = {
        "image_model": "coherent_gaussian",
        "frames": [],
        "source_construction_seconds": [],
    }
    for voltage_index, voltage in enumerate(experiment["voltages_v"]):
        folder = root / f"voltage_{voltage_index}"
        spec = json.loads((folder / "known_settings.json").read_text())
        truth = np.asarray(
            json.loads((folder / "scoring_truth.json").read_text())["parameters"]
        )
        spec.update(
            shape_yx=list(shape_yx),
            image_model="coherent_gaussian",
            psf_sigma_pixels=0.0,
            background_counts=3.0,
            moment_regions=True,
        )
        spec["specimen"].update(
            phase_contrast=True,
            description="Known smoothed MAGICAL intensity target and native optical-path phase, represented by coherent Gaussian beamlets.",
        )
        (folder / "known_settings.json").write_text(json.dumps(spec, indent=2) + "\n")
        material = prepare_known_transmission(
            template=dict(np.load(folder / "specimen_template.npz")),
            voltage_v=voltage,
            kernel_sigma_m=spec["feature_sigma_m"],
            step_nm=step_nm,
        )
        save_known_transmission(folder / "moment_specimen.npz", material)
        renderer = make_gaussian_renderer(
            shape_yx=shape_yx,
            pixel_pitch_m=spec["pixel_pitch_m"],
            samples_per_pixel=samples_per_pixel,
        )
        source_cache = {}
        rng = np.random.default_rng(random_seed + voltage_index)
        for filename, kind in (
            ("images.npz", "ordinary"),
            ("tilt_images_3mrad.npz", "tilts"),
            ("beam_shift_images_10nm.npz", "shifts"),
        ):
            data = dict(np.load(folder / filename))
            positions = data.get(
                "input_positions_m", np.zeros((len(data["images"]), 2))
            )
            angles = data.get(
                "input_angles_rad",
                data.get("input_tilts_rad", np.zeros_like(positions)),
            )
            frames, means = [], []
            common = np.zeros(2)
            for index, (controls, position, angle, mag) in enumerate(
                zip(
                    data["lens_controls"],
                    positions,
                    angles,
                    data["nominal_magnifications"],
                )
            ):
                key = tuple(position)
                if key not in source_cache:
                    start = perf_counter()
                    source_cache[key] = gaussian_exit_wave(
                        transmission=material,
                        profile="circle",
                        position_m=position,
                        curvature_per_m=truth[16] * 1e4,
                        radius_m=spec["specimen"]["illumination_radius_m"],
                        edge_m=spec["specimen"]["illumination_edge_m"],
                        specimen_rotation_rad=truth[18],
                    )
                    report["source_construction_seconds"].append(perf_counter() - start)
                source = source_cache[key]
                source = source.derive(
                    dx=source.dx + angle[0],
                    dy=source.dy + angle[1],
                    amplitude=source.amplitude
                    * jnp.exp(
                        1j * source.k * (source.x * angle[0] + source.y * angle[1])
                    ),
                )
                column = tuple(
                    replace(c, include_axial_path=False) if hasattr(c, "q") else c
                    for c in projector_components(
                        jnp.asarray(truth), jnp.asarray(controls)
                    )
                )
                if kind == "ordinary" or index % 2 == 0:
                    common = rng.normal(0, 0.3, 2)
                offset = common + (
                    rng.normal(0, float(data["drift_std_pixels"]), 2)
                    if kind != "ordinary"
                    else 0.0
                )
                start = perf_counter()
                mean = np.asarray(
                    renderer(
                        source, column, jnp.exp(2 * truth[17]), jnp.asarray(offset)
                    )
                )
                mean = mean * (4e7 * (mag / 40000.0) ** 2) + 3.0
                elapsed = perf_counter() - start
                frames.append(
                    rng.poisson(mean).astype(float) + rng.normal(0, 2.0, mean.shape)
                )
                means.append(mean)
                report["frames"].append(
                    {
                        "voltage_index": voltage_index,
                        "kind": kind,
                        "index": index,
                        "seconds": elapsed,
                        "gaussians": int(source.x.size),
                    }
                )
                if index % 5 == 0:
                    print(
                        f"Gaussian projector {voltage / 1000:g} kV, {kind}: {index + 1}/{len(positions)} ({elapsed:.2f}s)",
                        flush=True,
                    )
            data["images"] = np.asarray(frames)
            np.savez_compressed(folder / filename, **data)
            np.savez_compressed(folder / ("scoring_gaussian_" + filename), images=means)
        del source_cache
        jax.clear_caches()
    experiment.update(image_model="coherent_gaussian", detector_shape_yx=list(shape_yx))
    (root / "experiment.json").write_text(json.dumps(experiment, indent=2) + "\n")
    (root / "gaussian_generation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def replace_condenser_with_gaussian_images(
    *,
    directory,
    random_seed=42001,
    step_nm=0.5,
    shape_yx=(256, 256),
    samples_per_pixel=8,
):
    """Generate all condenser settings from native Gaussian source/lens transport."""
    import hashlib
    import inspect
    from time import perf_counter

    import jax
    from condenser_image_pipeline import beam_coordinates
    from condenser_voltage_experiment import condenser_at_voltage
    from finite_projector_pipeline import projector_components
    from full_microscope_projector import at_voltage

    from temgym_core.coherent_moments import detector_moments
    from temgym_core.constants import energy2wavelength

    root = Path(directory)
    experiment = json.loads((root / "experiment.json").read_text())
    truth = np.asarray(
        json.loads((root / "scoring_truth.json").read_text())["parameters"]
    )
    projector = np.asarray(
        json.loads((root.parent / "projector/scoring_truth.json").read_text())[
            "parameters"
        ]
    )
    native_state = jax.jit(
        lambda p, c, v: condenser_gaussian_state(parameters=p, controls=c, voltage_v=v)
    )
    report = {"image_model": "coherent_gaussian", "frames": [], "beam_states": []}
    for voltage_index, voltage in enumerate(experiment["voltages_v"]):
        folder = root / f"voltage_{voltage_index}"
        detector = json.loads((folder / "detector.json").read_text())
        detector.update(
            shape_yx=list(shape_yx),
            image_model="coherent_gaussian",
            background_counts=3.0,
            moment_regions=True,
        )
        material = prepare_known_transmission(
            template=dict(np.load(folder / "template.npz")),
            voltage_v=voltage,
            kernel_sigma_m=detector["sample_kernel_sigma_m"],
            step_nm=step_nm,
        )
        save_known_transmission(folder / "moment_specimen.npz", material)
        effective = condenser_at_voltage(
            truth, voltage, source_parameters=truth[22:29] if voltage_index else None
        )
        pp = at_voltage(projector, voltage, experiment["reference_voltage_v"])
        renderer = make_gaussian_renderer(
            shape_yx=shape_yx,
            pixel_pitch_m=detector["pixel_pitch_m"],
            samples_per_pixel=samples_per_pixel,
            psf_sigma_pixels=detector["psf_sigma_pixels"],
        )
        for prediction in (False, True):
            target = folder / "prediction" if prediction else folder
            data = dict(np.load(target / "images.npz"))
            (target / "detector.json").write_text(json.dumps(detector, indent=2) + "\n")
            rng = np.random.default_rng(
                random_seed + 100 * voltage_index + 1000 * prediction
            )
            images, means = [], []
            checkpoint = target / "gaussian_image_progress.npz"
            digest = hashlib.sha256()
            for array in (
                truth,
                projector,
                data["controls"],
                data["projector_controls"],
            ):
                digest.update(np.asarray(array).tobytes())
            digest.update(
                json.dumps(
                    {
                        "voltage": voltage,
                        "step_nm": step_nm,
                        "shape": shape_yx,
                        "samples": samples_per_pixel,
                        "detector": detector,
                    },
                    sort_keys=True,
                ).encode()
            )
            digest.update(
                (
                    inspect.getsource(gaussian_exit_wave)
                    + inspect.getsource(make_gaussian_renderer)
                ).encode()
            )
            signature = digest.hexdigest()
            if checkpoint.exists():
                saved = np.load(checkpoint)
                if str(saved["signature"]) == signature:
                    means = list(saved["images"])
            for index, controls in enumerate(data["controls"]):
                if index < len(means):
                    continue
                mean, covariance = native_state(
                    effective, jnp.asarray(controls), jnp.asarray(voltage)
                )
                beam = np.asarray(beam_coordinates(mean, covariance))
                floor = float(energy2wavelength(voltage)) / (
                    4 * np.pi * np.exp(beam[4])
                )
                extra = np.exp(2 * beam[6]) - floor**2
                if extra < 0:
                    raise ValueError(
                        "Generating source violates the Gaussian uncertainty bound."
                    )
                source = gaussian_exit_wave(
                    transmission=material,
                    profile="gaussian",
                    position_m=beam[:2] * 1e-7,
                    angle_rad=beam[2:4] * 1e-3,
                    curvature_per_m=beam[5] * 1e4,
                    rms_size_m=np.exp(beam[4]),
                )
                exit_mean, exit_covariance = material["sampled"].gaussian_moments(
                    mean, covariance, energy2wavelength(voltage)
                )
                row = []
                for view, command in enumerate(data["projector_controls"]):
                    column = tuple(
                        replace(c, include_axial_path=False) if hasattr(c, "q") else c
                        for c in projector_components(pp, jnp.asarray(command))
                    )
                    moments = detector_moments(
                        column_matrix(column),
                        exit_mean,
                        exit_covariance,
                        pixel_pitch_m=detector["pixel_pitch_m"],
                        psf_sigma_pixels=detector["psf_sigma_pixels"],
                    )
                    region = gaussian_rendering_region(
                        detector_moments=moments, shape_yx=shape_yx
                    )
                    start = perf_counter()
                    frame = (
                        4e7
                        * np.asarray(
                            renderer(
                                source,
                                column,
                                extra,
                                jnp.zeros(2),
                                region_shape_yx=region,
                            )
                        )
                        + 3.0
                    )
                    elapsed = perf_counter() - start
                    row.append(frame)
                    report["frames"].append(
                        {
                            "voltage_index": voltage_index,
                            "prediction": prediction,
                            "setting": index,
                            "view": view,
                            "seconds": elapsed,
                            "gaussians": int(source.x.size),
                            "rendered_region_yx": list(region),
                        }
                    )
                report["beam_states"].append(
                    {
                        "voltage_index": voltage_index,
                        "prediction": prediction,
                        "setting": index,
                        "coordinates": beam.tolist(),
                    }
                )
                means.append(row)
                np.savez_compressed(checkpoint, signature=signature, images=means)
                print(
                    f"Gaussian condenser {voltage / 1000:g} kV, prediction={prediction}: {index + 1}/{len(data['controls'])}",
                    flush=True,
                )
            images = [
                rng.poisson(row).astype(float)
                + rng.normal(0, detector["read_noise_e"], np.shape(row))
                for row in means
            ]
            data["images"] = np.asarray(images)
            np.savez_compressed(target / "images.npz", **data)
            np.savez_compressed(target / "scoring_gaussian_images.npz", images=means)
        jax.clear_caches()
    experiment.update(image_model="coherent_gaussian", detector_shape_yx=list(shape_yx))
    (root / "experiment.json").write_text(json.dumps(experiment, indent=2) + "\n")
    (root / "gaussian_generation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def replace_aperture_with_gaussian_counts(
    *, directory, random_seed=44001, radial_order=256, azimuthal_order=128
):
    """Propagate a native Gaussian source to each aperture and integrate |t psi|²."""
    from types import SimpleNamespace

    import jax
    from full_microscope_apertures import aperture_centres, effective_parameters
    from numpy.polynomial.legendre import leggauss

    from temgym_core.coherent_microscope import GaussianSourceState
    from temgym_core.components import SigmoidAperture
    from temgym_core.evaluate import evaluate_gaussians

    root = Path(directory)
    experiment = json.loads((root / "experiment.json").read_text())
    truth = np.asarray(
        json.loads((root / "scoring_truth.json").read_text())["parameters"]
    )
    nodes, weights = leggauss(radial_order)
    nodes, weights = jnp.asarray((nodes + 1) / 2), jnp.asarray(weights / 2)
    theta = jnp.arange(azimuthal_order) * 2 * jnp.pi / azimuthal_order
    report = {
        "forward_model": "native Gaussian transport and squared SigmoidAperture transmission",
        "radial_order": radial_order,
        "azimuthal_order": azimuthal_order,
        "blocks": [],
    }
    for index, voltage in enumerate(experiment["voltages_v"]):
        folder = root / f"voltage_{index}"
        spec = json.loads((folder / "aperture.json").read_text())
        p = effective_parameters(
            truth, voltage, experiment["reference_voltage_v"], index
        )
        centres = aperture_centres(p, len(spec["radii_m"])) * 1e-6

        def transmit(command, p=p, voltage=voltage, spec=spec, centres=centres):
            mean, covariance = condenser_gaussian_state(
                parameters=p,
                controls=command,
                voltage_v=voltage,
                aperture_fraction=jax.nn.sigmoid(p[19]),
            )
            # The transported mixture has an exact Gaussian intensity at this
            # plane. Its phase does not affect an integrated aperture current.
            incident = GaussianSourceState(
                0.0, x=mean[0], y=mean[1], rms_size=jnp.sqrt(covariance[0, 0])
            ).beam(0.0, voltage, jnp.zeros(2))
            fractions = []
            for radius, centre in zip(spec["radii_m"], centres):
                aperture = SigmoidAperture(
                    z=0.0,
                    radius=radius,
                    edge_width=spec["edge_width_m"],
                    x0=centre[0],
                    y0=centre[1],
                )
                narrow = jnp.sqrt(covariance[0, 0]) < radius / 3
                extent = jnp.where(
                    narrow,
                    8 * jnp.sqrt(covariance[0, 0]),
                    radius + 16 * spec["edge_width_m"],
                )
                origin = jnp.where(narrow, mean[:2], centre)
                r = nodes * extent
                xx, yy = (
                    origin[0] + r[:, None] * jnp.cos(theta),
                    origin[1] + r[:, None] * jnp.sin(theta),
                )
                coordinates = jnp.stack((xx, yy), axis=-1).reshape(-1, 2)
                grid = SimpleNamespace(coords=coordinates, shape=xx.shape)
                intensity = (
                    jnp.abs(evaluate_gaussians(incident, grid, method="scan")) ** 2
                )
                transmission = jnp.exp(
                    2 * aperture.log_transmission(coordinates)
                ).reshape(xx.shape)
                fractions.append(
                    jnp.sum(
                        intensity
                        * transmission
                        * (weights * nodes * extent**2)[:, None]
                    )
                    * 2
                    * jnp.pi
                    / azimuthal_order
                )
            return jnp.stack(fractions)

        calculate = jax.jit(jax.vmap(transmit))
        for prediction in (False, True):
            path = folder / (
                "prediction/current_counts.npz" if prediction else "counts.npz"
            )
            data = dict(np.load(path))
            fractions = np.asarray(calculate(jnp.asarray(data["controls"])))
            if (
                not np.isfinite(fractions).all()
                or fractions.min() < 0
                or fractions.max() > 1.000001
            ):
                raise ValueError("Invalid native Gaussian aperture transmission.")
            rng = np.random.default_rng(random_seed + 100 * index + 1000 * prediction)
            dose = 2e7 * np.exp(rng.normal(0, 0.1, len(fractions)))
            data["reference_electrons"] = rng.poisson(dose)
            data["transmitted_electrons"] = rng.poisson(dose[:, None] * fractions)
            np.savez_compressed(path, **data)
            report["blocks"].append(
                {
                    "voltage_index": index,
                    "prediction": prediction,
                    "fractions": fractions.tolist(),
                }
            )
    (root / "gaussian_current_generation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def refine_condenser_pixel_integration(
    *,
    directory,
    random_seed=42001,
    variance_threshold_pixels=4.0,
    samples_per_pixel=16,
    selected_frames=None,
    gaussian_overlap=1.1,
):
    """Refine narrow coherent spots before drawing the final detector noise.

    The physical experiment is unchanged. Smaller coherent structures require
    finer integration inside each recorded pixel, including before detector blur.
    """
    import jax
    from condenser_image_pipeline import beam_coordinates
    from condenser_voltage_experiment import condenser_at_voltage
    from finite_projector_pipeline import projector_components
    from full_microscope_projector import at_voltage

    from temgym_core.coherent_moments import detector_moments
    from temgym_core.constants import energy2wavelength

    root = Path(directory)
    experiment = json.loads((root / "experiment.json").read_text())
    truth = jnp.asarray(
        json.loads((root / "scoring_truth.json").read_text())["parameters"]
    )
    projector = jnp.asarray(
        json.loads((root.parent / "projector/scoring_truth.json").read_text())[
            "parameters"
        ]
    )
    native = jax.jit(
        lambda p, c, v: condenser_gaussian_state(parameters=p, controls=c, voltage_v=v)
    )
    report = {
        "samples_per_pixel": samples_per_pixel,
        "variance_threshold_pixels": variance_threshold_pixels,
        "frames": [],
    }
    previous_path = root / "gaussian_pixel_refinement.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        for row in previous["frames"]:
            row.setdefault("samples_per_pixel", previous["samples_per_pixel"])
        report["frames"] = previous["frames"]
    selected = (
        None if selected_frames is None else {tuple(row) for row in selected_frames}
    )
    for index, voltage in enumerate(experiment["voltages_v"]):
        parent = root / f"voltage_{index}"
        detector = json.loads((parent / "detector.json").read_text())
        material = load_known_transmission(
            parent
            / (
                "gaussian_specimen.npz"
                if (parent / "gaussian_specimen.npz").exists()
                else "moment_specimen.npz"
            )
        )
        wave = energy2wavelength(voltage)
        sample_moments = jax.jit(
            lambda mean, cov, sample=material["sampled"], wave=wave: (
                sample.gaussian_moments(mean, cov, wave)
            )
        )
        effective = condenser_at_voltage(
            truth, voltage, source_parameters=truth[22:29] if index else None
        )
        pp = at_voltage(projector, voltage, experiment["reference_voltage_v"])
        render = make_gaussian_renderer(
            shape_yx=detector["shape_yx"],
            pixel_pitch_m=detector["pixel_pitch_m"],
            samples_per_pixel=samples_per_pixel,
            psf_sigma_pixels=detector["psf_sigma_pixels"],
        )
        for prediction in (False, True):
            folder = parent / "prediction" if prediction else parent
            data = dict(np.load(folder / "images.npz"))
            images = np.load(folder / "scoring_gaussian_images.npz")["images"]
            for setting, controls in enumerate(data["controls"]):
                mean, cov = native(
                    effective, jnp.asarray(controls), jnp.asarray(voltage)
                )
                exit_mean, exit_cov = sample_moments(mean, cov)
                source = None
                for view, command in enumerate(data["projector_controls"]):
                    if (
                        selected is not None
                        and (index, prediction, setting, view) not in selected
                    ):
                        continue
                    column = tuple(
                        replace(c, include_axial_path=False) if hasattr(c, "q") else c
                        for c in projector_components(pp, jnp.asarray(command))
                    )
                    moments = detector_moments(
                        column_matrix(column),
                        exit_mean,
                        exit_cov,
                        pixel_pitch_m=detector["pixel_pitch_m"],
                        psf_sigma_pixels=detector["psf_sigma_pixels"],
                    )
                    if (
                        selected is None
                        and min(np.asarray(moments)[[2, 4]])
                        >= variance_threshold_pixels
                    ):
                        continue
                    if source is None:
                        beam = np.asarray(beam_coordinates(mean, cov))
                        extra = (
                            np.exp(2 * beam[6])
                            - (float(wave) / (4 * np.pi * np.exp(beam[4]))) ** 2
                        )
                        source = gaussian_exit_wave(
                            transmission=material,
                            profile="gaussian",
                            position_m=beam[:2] * 1e-7,
                            angle_rad=beam[2:4] * 1e-3,
                            curvature_per_m=beam[5] * 1e4,
                            rms_size_m=np.exp(beam[4]),
                            overlap=gaussian_overlap,
                        )
                    region = gaussian_rendering_region(
                        detector_moments=moments, shape_yx=detector["shape_yx"]
                    )
                    updated = (
                        4e7
                        * np.asarray(
                            render(
                                source,
                                column,
                                extra,
                                jnp.zeros(2),
                                region_shape_yx=region,
                            )
                        )
                        + 3
                    )
                    difference = (updated - images[setting, view]) / np.sqrt(
                        updated + detector["read_noise_e"] ** 2
                    )
                    report["frames"].append(
                        {
                            "samples_per_pixel": samples_per_pixel,
                            "gaussian_overlap": gaussian_overlap,
                            "voltage_index": index,
                            "prediction": prediction,
                            "setting": setting,
                            "view": view,
                            "pixel_change_rms": float(np.sqrt(np.mean(difference**2))),
                            "region_yx": list(region),
                        }
                    )
                    images[setting, view] = updated
                    print(
                        f"Refined Gaussian pixels {voltage / 1000:g} kV, new={prediction}, setting {setting + 1}, view {view + 1}",
                        flush=True,
                    )
            rng = np.random.default_rng(random_seed + 100 * index + 1000 * prediction)
            data["images"] = np.asarray(
                [
                    rng.poisson(row).astype(float)
                    + rng.normal(0, detector["read_noise_e"], np.shape(row))
                    for row in images
                ]
            )
            np.savez_compressed(folder / "images.npz", **data)
            np.savez_compressed(folder / "scoring_gaussian_images.npz", images=images)
            checkpoint = dict(np.load(folder / "gaussian_image_progress.npz"))
            np.savez_compressed(
                folder / "gaussian_image_progress.npz",
                signature=str(checkpoint["signature"]) + "; refined pixels",
                images=images,
            )
        jax.clear_caches()
    (root / "gaussian_pixel_refinement.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def prepare_condenser_moment_specimens(*, directory, step_nm=0.25):
    """Refine known-specimen quadrature independently of the Gaussian image basis."""
    import shutil

    root = Path(directory)
    experiment = json.loads((root / "experiment.json").read_text())
    for index, voltage in enumerate(experiment["voltages_v"]):
        folder = root / f"voltage_{index}"
        coarse = folder / "gaussian_specimen.npz"
        if not coarse.exists():
            shutil.copyfile(folder / "moment_specimen.npz", coarse)
        detector = json.loads((folder / "detector.json").read_text())
        fine = prepare_known_transmission(
            template=dict(np.load(folder / "template.npz")),
            voltage_v=voltage,
            kernel_sigma_m=detector["sample_kernel_sigma_m"],
            step_nm=step_nm,
        )
        save_known_transmission(folder / "moment_specimen.npz", fine)
        local = prepare_known_transmission(
            template=dict(np.load(folder / "template.npz")),
            voltage_v=voltage,
            kernel_sigma_m=detector["sample_kernel_sigma_m"],
            step_nm=0.0625,
            half_width_nm=32.0,
        )
        save_known_transmission(folder / "moment_specimen_local.npz", local)
    (root / "moment_quadrature.json").write_text(
        json.dumps(
            {
                "step_nm": step_nm,
                "local_step_nm": 0.0625,
                "local_half_width_nm": 32.0,
                "scope": "Known specimen integrals only. Gaussian image basis is retained independently.",
            },
            indent=2,
        )
        + "\n"
    )


def check_gaussian_moment_numerics(
    *, directory, rms_tolerance=0.1, coordinate_tolerance=0.3
):
    """Generator audit: compare two forward calculations before any inverse fit.

    Generating parameters are used only to assess integration accuracy. The
    fitting measurements are always extracted from the independent noisy pixels.
    """
    import jax
    from condenser_image_pipeline import beam_coordinates
    from condenser_voltage_experiment import condenser_at_voltage, projector_at_voltage
    from full_microscope_inference import make_condenser_moment_predictor

    from temgym_core.coherent_moments import measure_detector_moments, moment_regions

    root = Path(directory)
    experiment = json.loads((root / "experiment.json").read_text())
    truth = jnp.asarray(
        json.loads((root / "scoring_truth.json").read_text())["parameters"]
    )
    pp = jnp.asarray(
        json.loads((root.parent / "projector/scoring_truth.json").read_text())[
            "parameters"
        ]
    )
    native = jax.jit(
        lambda p, c, v: condenser_gaussian_state(parameters=p, controls=c, voltage_v=v)
    )
    report = {
        "frames": [],
        "passed": True,
        "rms_tolerance": rms_tolerance,
        "coordinate_tolerance": coordinate_tolerance,
    }
    for index, voltage in enumerate(experiment["voltages_v"]):
        parent = root / f"voltage_{index}"
        detector = json.loads((parent / "detector.json").read_text())
        predict = jax.jit(make_condenser_moment_predictor(folder=parent))
        effective = condenser_at_voltage(
            truth, voltage, source_parameters=truth[22:29] if index else None
        )
        for prediction in (False, True):
            folder = parent / "prediction" if prediction else parent
            data = np.load(folder / "images.npz")
            means = np.load(folder / "scoring_gaussian_images.npz")["images"]
            shape = (-1, *means.shape[-2:])
            regions = moment_regions(
                images=data["images"].reshape(shape),
                background_counts=detector["background_counts"],
                read_noise_counts=detector["read_noise_e"],
            )
            observed = measure_detector_moments(
                images=data["images"].reshape(shape),
                background_counts=detector["background_counts"],
                read_noise_counts=detector["read_noise_e"],
                regions=regions,
            )
            noiseless = measure_detector_moments(
                images=means.reshape(shape),
                background_counts=detector["background_counts"],
                read_noise_counts=detector["read_noise_e"],
                regions=regions,
            )
            white = np.linalg.inv(np.linalg.cholesky(observed["covariance"]))
            maps = jnp.asarray(
                [
                    projector_at_voltage(
                        pp,
                        c,
                        voltage,
                        reference_voltage_v=experiment["reference_voltage_v"],
                    )
                    for c in data["projector_controls"]
                ]
            )
            expected = []
            for command in data["controls"]:
                mean, covariance = native(
                    effective, jnp.asarray(command), jnp.asarray(voltage)
                )
                expected.append(
                    np.asarray(predict(beam_coordinates(mean, covariance), maps))
                )
            error = np.einsum(
                "nij,nj->ni",
                white,
                np.asarray(expected).reshape(-1, 5) - noiseless["values"],
            )
            for setting, row in enumerate(error.reshape(len(means), -1, 5)):
                for view, values in enumerate(row):
                    rms, maximum = (
                        float(np.sqrt(np.mean(values**2))),
                        float(abs(values).max()),
                    )
                    passed = rms < rms_tolerance and maximum < coordinate_tolerance
                    report["frames"].append(
                        {
                            "voltage_index": index,
                            "prediction": prediction,
                            "setting": setting,
                            "view": view,
                            "rms": rms,
                            "maximum": maximum,
                            "passed": passed,
                        }
                    )
                    report["passed"] = report["passed"] and passed
        jax.clear_caches()
    report["rms"] = float(
        np.sqrt(np.mean([row["rms"] ** 2 for row in report["frames"]]))
    )
    (root / "gaussian_moment_validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def ensure_gaussian_moment_numerics(*, directory, random_seed=42001):
    """Refine detector quadrature and Gaussian interpolation before fitting."""
    root = Path(directory)
    report = check_gaussian_moment_numerics(directory=root)

    def key(row):
        return (row["voltage_index"], row["prediction"], row["setting"], row["view"])

    for samples in (16, 32, 64):
        if report["passed"]:
            return report
        history = json.loads((root / "gaussian_pixel_refinement.json").read_text())
        latest = {
            key(row): row.get("samples_per_pixel", history["samples_per_pixel"])
            for row in history["frames"]
        }
        selected = [
            key(row)
            for row in report["frames"]
            if not row["passed"] and latest.get(key(row), 8) < samples
        ]
        if not selected:
            continue
        print(
            f"Refining {len(selected)} flagged Gaussian images at {samples} samples per pixel.",
            flush=True,
        )
        refine_condenser_pixel_integration(
            directory=root,
            random_seed=random_seed,
            samples_per_pixel=samples,
            selected_frames=selected,
        )
        report = check_gaussian_moment_numerics(directory=root)
    if not report["passed"]:
        selected = [key(row) for row in report["frames"] if not row["passed"]]
        print(f"Refining Gaussian interpolation in {len(selected)} images.", flush=True)
        refine_condenser_pixel_integration(
            directory=root,
            random_seed=random_seed,
            samples_per_pixel=32,
            selected_frames=selected,
            gaussian_overlap=1.5,
        )
        report = check_gaussian_moment_numerics(directory=root)
    if not report["passed"]:
        raise RuntimeError(
            "Gaussian image/moment numerical checks failed; see gaussian_moment_validation.json before fitting."
        )
    return report
