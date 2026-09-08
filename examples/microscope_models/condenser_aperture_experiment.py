"""Joint condenser inversion from MAGICAL images and TemGym aperture currents.

All 19 condenser/source parameters remain free. Three additional unknowns are
the aperture position within the drift beside CL3 and its transverse centre.
The open-aperture image measurements and aperture-current measurements are
separate acquisitions. Clipped beams are never analysed as uncut Gaussians.
"""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from condenser_calibration_model import (
    equivalent_parameters,
    generating_parameters,
    initial_parameters,
    named_parameters,
    source_moments,
)
from condenser_fitting import beam_to_moments, model_moments
from jax.scipy.special import i0e
from scipy.optimize import least_squares

from temgym_core.components import SigmoidAperture
from temgym_core.gaussian import make_gaussian
from temgym_core.solenoid import solenoid_column_matrix


def aperture_specification():
    """Public, calibrated aperture sizes; these are synthetic instrument settings."""
    return {
        "radii_m": [2e-6, 5e-6, 10e-6],
        "edge_width_m": 0.15e-6,
        "beamlet_waist_m": 0.005e-6,
        "voltage_v": 200000.0,
        "radial_order": 192,
    }


def aperture_moments(parameters, controls, side):
    """Round illumination just before the aperture, in x/y/theta_x/theta_y order."""
    p = jnp.asarray(parameters)
    gaps, lengths = jnp.exp(p[:5]), jnp.exp(p[5:9])
    strengths = jnp.exp(p[9:12]) * controls
    fraction = jax.nn.sigmoid(p[19])
    if side == "before":
        aperture_gaps = jnp.array([gaps[0], gaps[1] * (1 - fraction)])
        transfer = solenoid_column_matrix(aperture_gaps, lengths[:1], strengths[:1])[:4, :4]
    elif side == "after":
        aperture_gaps = jnp.array([gaps[0], gaps[1], gaps[2] * fraction])
        transfer = solenoid_column_matrix(aperture_gaps, lengths[:2], strengths[:2])[:4, :4]
    else:
        raise ValueError("Aperture side must be before or after CL3.")
    mean, covariance = source_moments(p[:19])
    return (transfer @ mean)[:2] - p[20:22] * 1e-6, (
        transfer @ covariance @ transfer.T
    )[0, 0]


def named_aperture_parameters(parameters, side):
    result = named_parameters(parameters[:19])
    fraction = float(jax.nn.sigmoid(parameters[19]))
    if side == "after":
        offset = np.exp(parameters[6]) / 2 + np.exp(parameters[2]) * fraction
    else:
        offset = -np.exp(parameters[6]) / 2 - np.exp(parameters[1]) * fraction
    result.update(
        aperture_z_mm=result["CL3.z_mm"] + 1000 * offset,
        aperture_offset_from_CL3_mm=1000 * offset,
        aperture_x_um=float(parameters[20]),
        aperture_y_um=float(parameters[21]),
    )
    return result


def make_transmission_operator(*, specification, side):
    """Propagate an incoherent Gaussian-beamlet ensemble through SigmoidAperture.

    Each radial beamlet is an actual TemGym GaussianBeam. Its integrated output
    power is calculated after applying the component, including the changed
    amplitude and width. Circular symmetry allows analytical angular integration
    of the incident round Gaussian ensemble. No hard-aperture substitute is used.
    """
    if not jax.config.x64_enabled:
        raise ValueError("This calibration requires JAX 64-bit precision.")
    nodes, weights = np.polynomial.legendre.leggauss(specification["radial_order"])
    waist = specification["beamlet_waist_m"]
    radii, power_weights = [], []
    for radius in specification["radii_m"]:
        limit = radius + 12 * specification["edge_width_m"]
        radial = (nodes + 1) * limit / 2
        beam = make_gaussian(
            x=jnp.asarray(radial),
            y=0.0,
            voltage=specification["voltage_v"],
            waist_x=waist,
            waist_y=waist,
        )
        aperture = SigmoidAperture(
            radius=radius,
            edge_width=specification["edge_width_m"],
            sharpness=1.0,
            z=0.0,
        )
        transmitted = aperture(beam)
        ratio = jnp.abs(transmitted.amplitude / beam.amplitude) ** 2 * jnp.sqrt(
            jnp.linalg.det(jnp.imag(beam.Q_inv))
            / jnp.linalg.det(jnp.imag(transmitted.Q_inv))
        )
        radii.append(radial)
        power_weights.append(np.asarray(ratio) * weights * limit / 2)
    radial = jnp.asarray(radii)
    power_weights = jnp.asarray(power_weights)

    def transmission(parameters, controls):
        mean, variance = aperture_moments(parameters, controls, side)
        # Centre distribution plus beamlet width equals the full incident width.
        variance = variance - waist**2 / 4
        offset = jnp.sqrt(jnp.dot(mean, mean) + 1e-40)
        density = (
            radial
            / variance
            * jnp.exp(-((radial - offset) ** 2) / (2 * variance))
            * i0e(radial * offset / variance)
        )
        return jnp.sum(density * power_weights, axis=1)

    return jax.jit(transmission)


def generate_aperture_currents(
    *, output_directory, specification=None, random_seed=2606
):
    """Generate Poisson current readings. Only this step knows the hidden model."""
    folder = Path(output_directory)
    folder.mkdir(parents=True, exist_ok=True)
    spec = aperture_specification() if specification is None else specification
    truth = np.r_[
        generating_parameters(), np.log(0.0025 / (0.056 - 0.0025)), 0.2, -0.15
    ]
    rng = np.random.default_rng(random_seed)
    a, b = np.meshgrid(np.linspace(0.75, 1.35, 5), np.linspace(0.72, 1.32, 5))
    controls = np.c_[a.ravel(), b.ravel(), np.ones(a.size)]
    prediction_controls = np.c_[rng.uniform(0.75, 1.35, (9, 2)), np.ones(9)]
    operator = make_transmission_operator(specification=spec, side="after")
    for name, commands in [
        ("calibration", controls),
        ("prediction", prediction_controls),
    ]:
        fractions = np.asarray(jax.vmap(operator, in_axes=(None, 0))(truth, commands))
        if not np.all((fractions >= 0) & (fractions <= 1.000001)):
            raise RuntimeError("Check Gaussian aperture integration resolution.")
        incident = 2e7 * np.exp(rng.normal(0, 0.1, len(commands)))
        counts = rng.poisson(incident[:, None] * fractions)
        reference = rng.poisson(incident)
        np.savez_compressed(
            folder / f"{name}_counts.npz",
            controls=commands,
            transmitted_electrons=counts,
            reference_electrons=reference,
        )
    (folder / "aperture.json").write_text(json.dumps(spec, indent=2))
    (folder / "scoring_truth.json").write_text(
        json.dumps(
            {
                "parameters": truth.tolist(),
                "aperture_side": "after",
                "random_seed": random_seed,
            },
            indent=2,
        )
    )
    return folder


def measure_aperture_currents(*, input_directory, split="calibration"):
    """Convert counts to transmission ratios, keeping shared-reference covariance."""
    folder = Path(input_directory)
    data = np.load(folder / f"{split}_counts.npz")
    counts = data["transmitted_electrons"].astype(float)
    reference = data["reference_electrons"].astype(float)
    values = counts / reference[:, None]
    covariance = (
        np.eye(3)[None]
        * np.maximum(counts, 1)[:, None, :]
        / reference[:, None, None] ** 2
        + counts[:, :, None] * counts[:, None, :] / reference[:, None, None] ** 3
    )
    path = folder / f"{split}_measurements.npz"
    np.savez(path, controls=data["controls"], values=values, covariance=covariance)
    return path


def make_joint_residual(
    *, beam_measurements_path, aperture_measurements_path, specification, side
):
    beam = np.load(beam_measurements_path)
    aperture = np.load(aperture_measurements_path)
    beam_values = jax.vmap(beam_to_moments)(beam["values"])
    jac = np.asarray(jax.vmap(jax.jacfwd(beam_to_moments))(beam["values"]))
    beam_cov = jac @ beam["covariance"] @ jac.transpose(0, 2, 1)
    beam_whitening = jnp.asarray(np.linalg.inv(np.linalg.cholesky(beam_cov)))
    aperture_whitening = jnp.asarray(
        np.linalg.inv(np.linalg.cholesky(aperture["covariance"]))
    )
    beam_controls = jnp.asarray(beam["controls"])
    aperture_controls = jnp.asarray(aperture["controls"])
    measured_transmission = jnp.asarray(aperture["values"])
    transmission = make_transmission_operator(specification=specification, side=side)

    def residual(p):
        predicted_beam = jax.vmap(model_moments, in_axes=(None, 0))(
            p[:19], beam_controls
        )
        predicted_current = jax.vmap(transmission, in_axes=(None, 0))(
            p, aperture_controls
        )
        return jnp.r_[
            jnp.einsum(
                "nij,nj->ni", beam_whitening, predicted_beam - beam_values
            ).ravel(),
            jnp.einsum(
                "nij,nj->ni",
                aperture_whitening,
                predicted_current - measured_transmission,
            ).ravel(),
        ]

    return jax.jit(residual)


def equivalent_aperture_model(parameters, *, objective_strength):
    """Exact surviving ambiguity: translate the upstream assembly and rotate it.

    The aperture keeps its fractional location beside CL3. Its unknown transverse
    centre rotates with the unknown source. Every circular-aperture transmission
    and every specimen beam remain unchanged.
    """
    p = np.asarray(parameters)
    other = np.r_[
        equivalent_parameters(p[:19], objective_strength=objective_strength), p[19:]
    ]
    angle = objective_strength + np.exp(p[12])
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    other[20:22] = rotation @ p[20:22]
    return other


def fit_joint_condenser(
    *,
    beam_measurements_path,
    aperture_measurements_path,
    specification,
    output_directory,
    starting_parameters,
    starts_per_side=8,
    max_evaluations=3000,
    random_seed=83,
    reuse_completed=True,
    verbose=True,
):
    """Fit all 22 unknowns without fixing a gauge or adding soft priors."""
    folder = Path(output_directory)
    folder.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(Path(__file__).read_bytes())
    for path in (
        beam_measurements_path,
        aperture_measurements_path,
        Path(__file__).with_name("condenser_calibration_model.py"),
    ):
        digest.update(Path(path).read_bytes())
    digest.update(
        json.dumps(
            {
                "specification": specification,
                "starting_parameters": np.asarray(starting_parameters).tolist(),
                "starts": starts_per_side,
                "max_evaluations": max_evaluations,
                "seed": random_seed,
            },
            sort_keys=True,
        ).encode()
    )
    signature = digest.hexdigest()
    summary_path = folder / "joint_fit_summary.json"
    if reuse_completed and summary_path.exists():
        saved = json.loads(summary_path.read_text())
        if saved.get("signature") == signature:
            return saved
    bounds_centre = initial_parameters()
    lower = np.r_[
        bounds_centre[:9] - np.log(2.5),
        bounds_centre[9:13] - np.log(1.8),
        [-5, -5, -1, -1],
        np.log([1e-9, 10e-6]),
        -7.0,
        -3.0,
        -3.0,
    ]
    upper = np.r_[
        bounds_centre[:9] + np.log(2.5),
        bounds_centre[9:13] + np.log(1.8),
        [5, 5, 1, 1],
        np.log([80e-9, 300e-6]),
        0.0,
        3.0,
        3.0,
    ]
    start = np.r_[np.asarray(starting_parameters), -3.0, 0.0, 0.0]
    rng = np.random.default_rng(random_seed)
    records = []
    for side in ("before", "after"):
        residual = make_joint_residual(
            beam_measurements_path=beam_measurements_path,
            aperture_measurements_path=aperture_measurements_path,
            specification=specification,
            side=side,
        )
        derivative = jax.jit(jax.jacfwd(residual))
        for index in range(starts_per_side):
            guess = start.copy()
            if index:
                guess[:13] += rng.normal(0, 0.035, 13)
                guess[13:17] += rng.normal(0, 0.01, 4)
                guess[17:19] += rng.normal(0, 0.06, 2)
                guess[19:] += rng.normal(0, [0.4, 0.15, 0.15], 3)
            guess = np.clip(guess, lower + 1e-8, upper - 1e-8)
            fit = least_squares(
                lambda p, residual=residual: np.asarray(residual(p)),
                guess,
                jac=lambda p, derivative=derivative: np.asarray(derivative(p)),
                bounds=(lower, upper),
                x_scale="jac",
                ftol=1e-7,
                xtol=1e-10,
                gtol=1e-5,
                max_nfev=max_evaluations,
            )
            record = {
                "side": side,
                "index": index,
                "parameters": fit.x.tolist(),
                "success": bool(fit.success),
                "message": str(fit.message),
                "evaluations": int(fit.nfev),
                "chi_squared": float(2 * fit.cost),
                "residual_rms": float(np.sqrt(np.mean(fit.fun**2))),
                "model": named_aperture_parameters(fit.x, side),
                "active_bounds": np.flatnonzero(fit.active_mask).tolist(),
            }
            records.append(record)
            (folder / f"{side}_{index:02d}.json").write_text(
                json.dumps(record, indent=2)
            )
            if verbose:
                print(
                    f"Aperture {side} CL3, start {index + 1}/{starts_per_side}: "
                    f"RMS {record['residual_rms']:.3f}, success={fit.success}",
                    flush=True,
                )
    best = min(records, key=lambda r: r["chi_squared"])
    residual = make_joint_residual(
        beam_measurements_path=beam_measurements_path,
        aperture_measurements_path=aperture_measurements_path,
        specification=specification,
        side=best["side"],
    )
    p = np.asarray(best["parameters"])
    singular = np.linalg.svd(jax.jacfwd(residual)(p), compute_uv=False)
    family = []
    for strength in (-0.35, -0.45, -0.6):
        alternative = equivalent_aperture_model(p, objective_strength=strength)
        family.append(
            {
                "parameters": alternative.tolist(),
                "model": named_aperture_parameters(alternative, best["side"]),
                "maximum_whitened_prediction_change": float(
                    np.max(abs(np.asarray(residual(alternative) - residual(p))))
                ),
            }
        )
    result = {
        "signature": signature,
        "best_fit": best,
        "starts": records,
        "parameter_count": 22,
        "free_parameter_count": 22,
        "soft_priors": False,
        "fixed_physical_parameters": [],
        "singular_values": singular.tolist(),
        "rank": int(np.sum(singular > singular[0] * 1e-11)),
        "equivalent_models": family,
        "bounds_lower": lower.tolist(),
        "bounds_upper": upper.tolist(),
    }
    (folder / "joint_fit_summary.json").write_text(json.dumps(result, indent=2))
    return result


def evaluate_aperture_experiment(
    *,
    fitted,
    beam_prediction_path,
    current_prediction_path,
    current_calibration_path,
    specification,
    scoring_truth_path,
):
    """Freeze the joint fit, check new measurements, then score physical parameters."""
    from condenser_calibration_workflow import check_predictions

    beam_check = check_predictions(
        fitted=fitted, measurements_path=beam_prediction_path
    )
    p = np.asarray(fitted["best_fit"]["parameters"])
    operator = make_transmission_operator(
        specification=specification, side=fitted["best_fit"]["side"]
    )
    prediction = np.load(current_prediction_path)
    expected = np.asarray(
        jax.vmap(operator, in_axes=(None, 0))(p, prediction["controls"])
    )
    errors = np.linalg.solve(
        np.linalg.cholesky(prediction["covariance"]),
        (expected - prediction["values"])[..., None],
    )[..., 0]
    # A before-CL3 aperture cannot respond to CL3. Fit an unrestricted mean
    # transmission vector for each CL1 value to give an optimistic lower bound
    # on the before-CL3 model's current residual, independent of lens fitting.
    measured = np.load(current_calibration_path)
    chi_lower = 0.0
    for control in np.unique(measured["controls"][:, 0]):
        mask = np.isclose(measured["controls"][:, 0], control)
        precision = np.linalg.inv(measured["covariance"][mask])
        values = measured["values"][mask]
        mean = np.linalg.solve(
            precision.sum(axis=0), np.einsum("nij,nj->i", precision, values)
        )
        delta = values - mean
        chi_lower += float(np.einsum("ni,nij,nj->", delta, precision, delta))
    truth_record = json.loads(Path(scoring_truth_path).read_text())
    truth = np.asarray(truth_record["parameters"])
    truth_named = named_aperture_parameters(truth, truth_record["aperture_side"])
    fitted_named = named_aperture_parameters(p, fitted["best_fit"]["side"])
    return {
        "image_predictions": beam_check,
        "current_prediction_rms": float(np.sqrt(np.mean(errors**2))),
        "current_prediction_readings": int(errors.size),
        "before_CL3_current_chi_squared_lower_bound": chi_lower,
        "before_CL3_current_rms_lower_bound": float(
            np.sqrt(chi_lower / measured["values"].size)
        ),
        "controlled_strength_error_percent": float(
            100 * np.max(abs(np.exp(p[9:12] - truth[9:12]) - 1))
        ),
        "parameter_comparison": [
            {"parameter": k, "truth": v, "fit": fitted_named[k]}
            for k, v in truth_named.items()
        ],
        "scope": "Prediction residuals use measurement uncertainty only; truth is read only for scoring.",
    }


def show_aperture_result(*, fitted, checks, current_measurements_path, specification):
    """Short notebook result, a parameter table, and measured current curves."""
    import matplotlib.pyplot as plt
    from IPython.display import Markdown, display

    best = fitted["best_fit"]
    matching = [r for r in fitted["starts"] if r["side"] == best["side"]]
    converged = sum(r["success"] for r in matching)
    display(
        Markdown(
            f"**Result: the condenser is still not uniquely determined.** "
            f"The best model places the aperture **{best['side']} CL3** and fits all 22 parameters. "
            f"The data constrain {fitted['rank']} independent directions.\n\n"
            f"Joint calibration RMS: **{best['residual_rms']:.3f}**. New image-measurement RMS: "
            f"**{checks['image_predictions']['measurement_rms']:.3f}**. New current-measurement RMS: "
            f"**{checks['current_prediction_rms']:.3f}**.\n\n"
            f"{converged}/{len(matching)} starts on the preferred side met the optimizer stopping criteria. "
            "The other starts are retained in the report; this is not a completed search for all minima. "
            "An exact family of different pre-field/source/aperture models has identical predictions."
        )
    )
    data = np.load(current_measurements_path)
    mask = np.isclose(data["controls"][:, 0], 1.05)
    commands = data["controls"][mask]
    operator = make_transmission_operator(
        specification=specification, side=best["side"]
    )
    predicted = np.asarray(
        jax.vmap(operator, in_axes=(None, 0))(jnp.asarray(best["parameters"]), commands)
    )
    _fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    for i, radius in enumerate(specification["radii_m"]):
        axes[0].errorbar(
            commands[:, 1],
            data["values"][mask, i],
            yerr=np.sqrt(data["covariance"][mask, i, i]),
            fmt="o",
            label=f"{2 * radius * 1e6:g} µm diameter",
        )
        axes[0].plot(commands[:, 1], predicted[:, i], color=f"C{i}")
    axes[0].set(
        xlabel="CL3 command (CL1 fixed)",
        ylabel="Transmitted / open current",
        title="Aperture data and joint fit",
    )
    axes[0].legend(fontsize=8)
    singular = np.asarray(fitted["singular_values"])
    axes[1].semilogy(np.arange(1, 23), singular / singular[0], "o-")
    axes[1].set(
        xlabel="Sensitivity direction",
        ylabel="Relative sensitivity",
        title="One exact ambiguity remains",
    )
    plt.show()
    table = "| Parameter | Truth | Joint fit |\n|---|---:|---:|\n"
    table += "\n".join(
        f"| {r['parameter']} | {r['truth']:.5g} | {r['fit']:.5g} |"
        for r in checks["parameter_comparison"]
    )
    display(Markdown(table))
    family = "| Equally predictive model | Pre-field strength | Pre-field length (mm) | Aperture z (mm) |\n|---|---:|---:|---:|\n"
    for i, member in enumerate(fitted["equivalent_models"]):
        model = member["model"]
        family += f"| {i + 1} | {model['Objective pre-field.strength']:.3f} | {model['Objective pre-field.length_mm']:.4f} | {model['aperture_z_mm']:.4f} |\n"
    display(Markdown(family))
