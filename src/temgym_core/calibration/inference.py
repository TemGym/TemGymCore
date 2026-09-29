"""Fit aperture-in image measurements with native matrices and aperture moments."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import chi2

from temgym_core.moments import (
    measure_detector_moments,
    moment_regions,
    predict_detector_moments,
)
from temgym_core.optimization import least_squares

from .apertures import aperture_centres, effective_parameters, make_aperture_moments
from .condenser_model import condenser_starting_guesses, first_crossover_fraction
from .projector_model import (
    at_voltage,
    make_starting_models,
    matrix5,
    model_to_parameters,
    projector_coordinates,
)

PRESETS = Path(__file__).parent / "presets"
KICK_UNITS = np.array([20e-9, 20e-9, 3e-3, 3e-3])


def subset_settings(settings, selected):
    return {key: value[selected] for key, value in settings.items()}


def make_predictions(specification, settings, *, radial_order=192, angular_order=64):
    """47 physical coordinates → five image moments and transmitted fraction per hole."""
    _, decode, _, _ = projector_coordinates(PRESETS / "projector_position_priors.json")
    groups = []
    for vi, voltage in enumerate(specification["voltages_v"]):
        selected = np.flatnonzero(settings["voltage_index"] == vi)
        controls, inverse = np.unique(
            settings["condenser_controls"][selected], axis=0, return_inverse=True
        )
        apertures = [
            make_aperture_moments(
                radius_m=d * 0.5e-6,
                edge_width_m=specification["aperture_edge_width_m"],
                voltage_v=voltage,
                radial_order=radial_order,
                angular_order=angular_order,
                diffraction=specification.get("aperture_diffraction", True),
            )
            for d in specification["aperture_diameters_um"]
        ]
        groups.append(
            (
                vi,
                voltage,
                jnp.asarray(selected),
                jnp.asarray(controls),
                jnp.asarray(inverse),
                jnp.asarray(settings["projector_controls"][selected]),
                jnp.asarray(settings["specimen_kick"][selected]),
                apertures,
            )
        )

    def predict(parameters):
        result = jnp.zeros((len(settings["voltage_index"]), 2, 6))
        for (
            vi,
            voltage,
            selected,
            controls,
            inverse,
            commands,
            kicks,
            apertures,
        ) in groups:
            pp = at_voltage(
                decode(parameters[:16]), voltage, specification["voltages_v"][0]
            )
            cp = effective_parameters(
                parameters[16:], voltage, specification["voltages_v"][0], vi
            )
            centres = aperture_centres(cp, 2) * 1e-6
            maps = jax.vmap(matrix5, in_axes=(None, 0))(pp, commands)
            for ai, aperture in enumerate(apertures):
                mean, covariance, flux = jax.vmap(aperture, in_axes=(None, 0, None))(
                    cp, controls, centres[ai]
                )
                moments = jax.vmap(
                    lambda m, u, s: predict_detector_moments(
                        m,
                        u,
                        s,
                        pixel_pitch_m=specification["pixel_pitch_m"],
                        psf_sigma_pixels=specification["psf_sigma_pixels"],
                    )
                )(maps, mean[inverse] + kicks, covariance[inverse])
                result = result.at[selected, ai, :5].set(moments)
                result = result.at[selected, ai, 5].set(flux[inverse])
        return result

    return jax.jit(predict)


def profiled_residual(prediction, values, whitener):
    """Eliminate one unknown incident dose shared by each two-aperture pair.

    Use the full moment/intensity covariance. No aperture-out reference is used.
    """
    base = jnp.einsum(
        "...ij,...j->...i", whitener, prediction.at[..., 5].set(0.0) - values
    )
    direction = whitener[..., :, 5] * prediction[..., 5, None]
    dose = -jnp.sum(base * direction, axis=(-2, -1)) / jnp.sum(
        direction**2, axis=(-2, -1)
    )
    return base + jnp.maximum(dose, 0.0)[:, None, None] * direction


def measured_images(images, specification):
    shape = images.shape
    frames = images.reshape(-1, *shape[-2:])
    recorded = np.isfinite(frames).all(axis=(1, 2))
    if not np.all(recorded | np.isnan(frames).all(axis=(1, 2))):
        raise ValueError("A frame must be finite or entirely NaN for an unrecorded aperture.")
    if not recorded.any():
        raise ValueError("No recorded images were supplied.")
    regions = moment_regions(
        images=frames[recorded],
        background_counts=specification["background_counts"],
        read_noise_counts=specification["read_noise_counts"],
        margin_pixels=specification.get("moment_margin_pixels", 48),
    )
    measured = measure_detector_moments(
        images=frames[recorded],
        background_counts=specification["background_counts"],
        read_noise_counts=specification["read_noise_counts"],
        regions=regions,
    )
    values = np.zeros((len(frames), 6))
    values[recorded] = np.column_stack((measured["values"], measured["signal_counts"]))
    covariance = np.broadcast_to(np.eye(6), (len(frames), 6, 6)).copy()
    covariance[recorded] = measured["covariance_with_flux"]
    masks = np.zeros(frames.shape, dtype=bool)
    masks[recorded] = regions
    return values.reshape(*shape[:2], 6), covariance.reshape(*shape[:2], 6, 6), masks.reshape(shape)


def measure_beam_images(*, directory_path):
    """Only recorded pixels, detector calibration and recorded commands are read."""
    root = Path(directory_path)
    spec = json.loads((root / "experiment.json").read_text())
    settings = dict(np.load(root / "settings.npz"))
    images = np.load(root / "images.npz")["images"]
    recorded = np.isfinite(images).all(axis=(-2, -1))
    values, covariance, regions = measured_images(images, spec)
    np.savez_compressed(
        root / "measurements.npz", values=values, covariance=covariance, regions=regions,
        recorded=recorded,
    )
    save_projector_responses(root, settings, values, covariance, recorded=recorded)


def save_projector_responses(root, settings, values, covariance, recorded=None):
    """Form signed steering differences from measured or simulated moments."""
    responses, errors, commands, voltages, kicks, apertures = [], [], [], [], [], []
    for group in np.unique(settings["response_group"]):
        if group < 0:
            continue
        rows = list(
            np.flatnonzero(
                (settings["response_group"] == group)
                & np.any(settings["specimen_kick"] != 0, axis=1)
            )
        )
        while rows:
            plus = rows.pop(0)
            opposite = [
                row
                for row in rows
                if np.array_equal(
                    settings["specimen_kick"][row], -settings["specimen_kick"][plus]
                )
            ]
            if len(opposite) != 1:
                raise ValueError(
                    "Each measured kick needs one equal and opposite partner."
                )
            minus = opposite[0]
            rows.remove(minus)
            for ai in range(2):
                if recorded is not None and not (recorded[plus, ai] and recorded[minus, ai]):
                    continue
                responses.append((values[plus, ai, :2] - values[minus, ai, :2]) / 2)
                errors.append(
                    (covariance[plus, ai, :2, :2] + covariance[minus, ai, :2, :2]) / 4
                )
                kicks.append(
                    (settings["specimen_kick"][plus] - settings["specimen_kick"][minus])
                    / 2
                )
                commands.append(settings["projector_controls"][plus])
                voltages.append(settings["voltage_index"][plus])
                apertures.append(ai)
    np.savez_compressed(
        root / "projector_responses.npz",
        values=responses,
        covariance=errors,
        controls=commands,
        voltage_index=voltages,
        specimen_kicks=kicks,
        aperture_index=apertures,
    )


def numerical_image_check(*, directory_path, means, truth):
    """Compare two forward calculations; truth is used only for this renderer audit."""
    root = Path(directory_path)
    spec = json.loads((root / "experiment.json").read_text())
    settings = dict(np.load(root / "settings.npz"))
    _, covariance, regions = measured_images(
        np.load(root / "images.npz")["images"], spec
    )
    m = measure_detector_moments(
        images=means.reshape(-1, *means.shape[-2:]),
        background_counts=0.0,
        read_noise_counts=0.0,
        regions=regions.reshape(-1, *means.shape[-2:]),
    )
    values = np.column_stack((m["values"], m["signal_counts"])).reshape(
        *means.shape[:2], 6
    )
    encode, _, _, _ = projector_coordinates(PRESETS / "projector_position_priors.json")
    parameters = jnp.r_[
        encode(jnp.asarray(truth["projector_parameters"]))[:16],
        jnp.asarray(truth["condenser_parameters"]),
    ]
    predict = make_predictions(spec, settings)
    expected = predict(parameters)
    white = jnp.asarray(np.linalg.inv(np.linalg.cholesky(covariance)))
    error = np.asarray(profiled_residual(expected, jnp.asarray(values), white))
    rms = np.sqrt(np.mean(error**2, axis=(1, 2)))
    print(
        f"Image/moment numerical RMS: {np.sqrt(np.mean(error**2)):.5g}; worst pair {max(rms):.5g}",
        flush=True,
    )
    return {
        "passed": bool(max(rms) < 0.2),
        "rms": float(np.sqrt(np.mean(error**2))),
        "worst_pair_rms": float(max(rms)),
        "flagged_pairs": np.flatnonzero(rms >= 0.2).tolist(),
        "rms_per_pair": rms.tolist(),
    }


def run_starts(
    residual,
    starts,
    bounds,
    *,
    output_path,
    profiled_parameters=0,
    max_evaluations=3000,
    constraints=None,
    solver_options=None,
):
    """Keep every independent start; rank and noise agreement determine acceptance."""
    data_residual = jax.jit(residual)
    if constraints is not None:
        residual = lambda p: jnp.r_[data_residual(p), 1000 * constraints(p)]
    residual = jax.jit(residual)
    derivative = jax.jit(jax.jacfwd(residual))
    records, fits = [], []
    options = {"ftol": 1e-9, "xtol": 1e-10, "gtol": 1e-5, "x_scale": "jac"}
    options.update(solver_options or {})
    for index, start in enumerate(starts):
        fit = least_squares(
            residual,
            np.clip(start, np.asarray(bounds[0]) + 1e-9, np.asarray(bounds[1]) - 1e-9),
            jac=derivative,
            bounds=bounds,
            max_nfev=max_evaluations,
            max_iter=2 * max_evaluations,
            **options,
        )
        measured_error = np.asarray(data_residual(fit.x))
        violation = 0.0 if constraints is None else float(jnp.max(constraints(fit.x)))
        jac = np.asarray(fit.jacobian)[: len(measured_error)]
        scaled = jac / np.maximum(np.linalg.norm(jac, axis=0), 1e-30)
        singular = np.linalg.svd(scaled, compute_uv=False)
        rank = int(np.sum(singular > singular[0] * 1e-9))
        dof = max(jac.shape[0] - profiled_parameters - rank, 1)
        record = {
            "index": index,
            "parameters": np.asarray(fit.x).tolist(),
            "starting_parameters": np.asarray(start).tolist(),
            "success": bool(fit.success),
            "message": fit.message,
            "termination_status": fit.status,
            "optimality": fit.optimality,
            "solver_options": options,
            "scaled_condition_number": float(singular[0] / singular[-1]),
            "chi_squared": float(np.sum(measured_error**2)),
            "residual_rms": float(np.sqrt(np.mean(measured_error**2))),
            "constraint_max_violation": violation,
            "rank": rank,
            "parameter_count": len(start),
            "degrees_of_freedom": dof,
            "evaluations": fit.nfev,
            "acceptable": bool(
                fit.success
                and rank == len(start)
                and np.sum(measured_error**2) < chi2.ppf(0.99, dof)
                and violation < 1e-6
            ),
        }
        records.append(record)
        fits.append(fit)
        Path(output_path).write_text(json.dumps({"starts": records}, indent=2) + "\n")
        print(
            f"Start {index + 1}: RMS {record['residual_rms']:.5g}, rank {rank}/{len(start)}, acceptable {record['acceptable']}",
            flush=True,
        )
    valid = (
        [i for i, r in enumerate(records) if r["acceptable"]]
        or [i for i, r in enumerate(records) if r["success"]]
        or list(range(len(records)))
    )
    winner = min(valid, key=lambda i: records[i]["chi_squared"])
    fit = fits[winner]
    _, singular, vt = np.linalg.svd(
        np.asarray(fit.jacobian)[: len(data_residual(fit.x))], full_matrices=False
    )
    covariance = (
        (vt.T / singular**2) @ vt if records[winner]["rank"] == len(fit.x) else None
    )
    return {
        "best": records[winner],
        "starts": records,
        "covariance": None if covariance is None else covariance.tolist(),
    }


def fit_projector_responses(
    *, response, voltages_v, pixel_pitch_m, number_of_starts, output_path,
    prior_model=None,
):
    """Fit the 16 specimen-to-detector coordinates to paired steering responses.

    Each response is half the detector-centre difference, in pixels, between
    equal and opposite calibrated specimen kicks at one projector setting.
    Only the A and B blocks of the projector matrix enter; the condenser,
    source and apertures play no part. ``prior_model`` replaces the preset
    starting guess (same named form as ``projector_starting_guess.json``).
    Returns the ``run_starts`` record in encoded coordinates; decode with
    ``projector_coordinates``.
    """
    encode, decode, plo, phi = projector_coordinates(
        PRESETS / "projector_position_priors.json"
    )
    rw = jnp.asarray(np.linalg.inv(np.linalg.cholesky(response["covariance"])))
    rv = jnp.asarray(response["values"])
    commands = jnp.asarray(response["controls"])
    voltages = jnp.asarray(voltages_v)[np.asarray(response["voltage_index"])]
    kicks = jnp.asarray(response["specimen_kicks"])

    def projector_residual(u):
        maps = jax.vmap(
            lambda c, v: matrix5(at_voltage(decode(u), v, voltages_v[0]), c)
        )(commands, voltages)
        predicted = jnp.einsum("nij,nj->ni", maps[:, :2, :4], kicks) / pixel_pitch_m
        return jnp.einsum("nij,nj->ni", rw, predicted - rv).ravel()

    if prior_model is None:
        prior_model = json.loads((PRESETS / "projector_starting_guess.json").read_text())
    guesses = make_starting_models(
        prior_model=prior_model,
        position_priors=json.loads(
            (PRESETS / "projector_position_priors.json").read_text()
        ),
        number_of_starts=number_of_starts,
    )
    starts = [encode(model_to_parameters(g))[:16] for g in guesses]
    print("Fit the projector response to calibrated shifts and tilts", flush=True)
    return run_starts(projector_residual, starts, (plo, phi), output_path=output_path)


def fit_beam_microscope(*, directory_path, number_of_starts, projector_aperture_um=None):
    root = Path(directory_path)
    spec = json.loads((root / "experiment.json").read_text())
    settings = dict(np.load(root / "settings.npz"))
    data = dict(np.load(root / "measurements.npz"))
    selected = settings["purpose"] != "check"
    training = subset_settings(settings, selected)
    active = data.get("recorded", np.ones(data["values"].shape[:2], dtype=bool))[selected].copy()
    aperture_index = None
    if projector_aperture_um is not None:
        aperture_index = spec["aperture_diameters_um"].index(projector_aperture_um)
        active[training["purpose"] == "projector", 1 - aperture_index] = False
    if not active.any(axis=1).all():
        raise ValueError("Every training setting needs its selected aperture image.")
    if not active[training["purpose"] != "projector"].all():
        raise ValueError("Condenser settings require both aperture images.")
    crossover_bounds = spec.get("crossover_fraction_bounds")
    crossover_groups = [
        (
            vi,
            jnp.asarray(
                np.unique(
                    training["condenser_controls"][training["voltage_index"] == vi],
                    axis=0,
                )
            ),
        )
        for vi in range(len(spec["voltages_v"]))
    ]

    def crossover_constraints(cp):
        violations = []
        for vi, controls in crossover_groups:
            effective = effective_parameters(
                cp, spec["voltages_v"][vi], spec["voltages_v"][0], vi
            )
            fractions = jax.vmap(first_crossover_fraction, in_axes=(None, 0))(
                effective, controls
            )
            q = jnp.exp(effective[9]) * controls[:, 0]
            violations.extend(
                [
                    jnp.maximum(crossover_bounds[0] - fractions, 0.0),
                    jnp.maximum(fractions - crossover_bounds[1], 0.0),
                    jnp.maximum(q - jnp.pi / 2, 0.0),
                ]
            )
        return jnp.concatenate(violations)

    values = jnp.asarray(data["values"][selected])
    white = jnp.asarray(np.linalg.inv(np.linalg.cholesky(data["covariance"][selected])))
    white = white * jnp.asarray(active[..., None, None])
    encode, decode, plo, phi = projector_coordinates(
        PRESETS / "projector_position_priors.json"
    )
    response = dict(np.load(root / "projector_responses.npz"))
    if aperture_index is not None:
        # Legacy two-aperture response files are interleaved; imported sparse
        # acquisitions explicitly record each response's aperture index.
        indices = response.get("aperture_index", np.arange(len(response["values"])) % 2)
        response = {key: value[indices == aperture_index] for key, value in response.items()}
    projector = fit_projector_responses(
        response=response,
        voltages_v=spec["voltages_v"],
        pixel_pitch_m=spec["pixel_pitch_m"],
        number_of_starts=number_of_starts,
        output_path=root / "projector_starts.json",
    )
    if not projector["best"]["acceptable"]:
        raise RuntimeError("No acceptable projector fit; see projector_starts.json.")
    pp = jnp.asarray(projector["best"]["parameters"])
    predict = make_predictions(spec, training)

    def full_residual(parameters):
        # Exclude unused rows, including from the degrees-of-freedom count.
        # Profiling with the masked whitener leaves one-aperture dose unknown.
        return profiled_residual(predict(parameters), values, white)[active].ravel()

    cstarts, clo, chi = condenser_starting_guesses(
        starting_guess_path=PRESETS / "microscope_condenser_starting_guess.json",
        number_of_broad_starts=max(0, number_of_starts - 4),
        number_of_apertures=2,
    )
    print(
        "Fit the condenser and source, with the projector fixed for initialization",
        flush=True,
    )
    condenser = run_starts(
        lambda cp: full_residual(jnp.r_[pp, cp]),
        cstarts[:number_of_starts],
        (clo, chi),
        output_path=root / "condenser_starts.json",
        profiled_parameters=len(values),
        constraints=crossover_constraints if crossover_bounds is not None else None,
        # Coordinates already use log lengths/strengths and scaled source units.
        # Extra Jacobian scaling at a distant guess can pin source parameters
        # to their bounds before the lenses approach the observed focus.
        solver_options={"x_scale": 1.0, "geodesic": True},
    )
    # The final likelihood returns to the original images; the initialization
    # estimates and response fits are not counted as additional measurements.
    best = condenser["best"]["chi_squared"]
    candidates = [
        r
        for r in condenser["starts"]
        if r["success"] and r["chi_squared"] <= best + chi2.ppf(0.95, len(clo))
    ]
    if not candidates:
        raise RuntimeError(
            "Condenser initialization did not converge; see condenser_starts.json."
        )
    # Exact synthetic equations need an absolute residual target. A small
    # gradient alone can coexist with large errors in weak geometry directions.
    noiseless = (
        spec.get("measurement_noise_added") is False
        and not (root / "images.npz").exists()
    )
    refinement = {
        "geodesic": True,
        "gtol": 0.0 if noiseless else 1e-10,
        "ftol": 0.0 if noiseless else 1e-11,
        "xtol": 0.0 if noiseless else 1e-12,
        "residual_tol": 1e-8 if noiseless else 0.0,
        "initial_damping": 1e-6,
    }
    print("Refine both halves against the same original beam measurements", flush=True)
    joint = run_starts(
        full_residual,
        [jnp.r_[pp, jnp.asarray(r["parameters"])] for r in candidates],
        (np.r_[plo, clo], np.r_[phi, chi]),
        output_path=root / "joint_starts.json",
        profiled_parameters=len(values),
        constraints=(lambda p: crossover_constraints(p[16:]))
        if crossover_bounds is not None
        else None,
        # Initial fits select a basin. A loose gradient threshold is insufficient
        # to recover geometry along the joint model's weak, curved directions.
        solver_options=refinement,
        max_evaluations=6000,
    )
    result = {
        "projector_parameters": np.asarray(
            decode(jnp.asarray(joint["best"]["parameters"][:16]))
        ).tolist(),
        "condenser_parameters": joint["best"]["parameters"][16:],
        "solver_parameters": joint["best"]["parameters"],
        "covariance": joint["covariance"],
        "best": joint["best"],
        "starts": joint["starts"],
        "projector_initialization": projector,
        "condenser_initialization": condenser,
        "soft_geometry_priors": False,
        "crossover_fraction_bounds": crossover_bounds,
        "crossover_constraint_method": "Feasibility penalty with a final violation limit of 1e-6; excluded from data chi-squared, rank and local covariance.",
        "aperture_out_reference": False,
        "profiled_dose_parameters": len(values),
        "projector_aperture_um": projector_aperture_um,
        "fitted_image_count": int(active.sum()),
    }
    (root / "fit.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["best"]["acceptable"]:
        raise RuntimeError(
            "The joint fit is not full-rank at noise level; inspect joint_starts.json."
        )
    return result
