"""Calibrate an aperture-centering scan without calibrated movement distances.

The known aperture radius and absolute transmitted fraction establish a length
scale. A positive movement metric allows unknown gains and skew; its overall
rotation is immaterial for finding the current maximum. No condenser geometry
enters this image-free current analysis.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from condenser_aperture_experiment import aperture_specification
from condenser_rotation_measurements import make_radial_transmission
from scipy.optimize import least_squares


def make_alignment_operator(*, aperture, commands):
    """Six unknowns: peak x/y, log beam RMS, log movement gains x/y, and skew.

    Beam RMS and gains are in micrometres. Commands are arbitrary actuator units.
    The upper-triangular movement map represents its metric, not an assertion
    that the actuator axes are aligned with the electron beam's x/y axes.
    """
    power = make_radial_transmission(specification=aperture)
    commands = jnp.asarray(commands)

    def predict(p):
        movement = jnp.array([[jnp.exp(p[3]), p[5]], [0.0, jnp.exp(p[4])]])
        means = (commands - p[:2]) @ movement.T * 1e-6
        return jax.vmap(power, in_axes=(0, None))(means, jnp.exp(2 * p[2]) * 1e-12)[
            :, 0
        ]

    return jax.jit(predict)


def simulate_alignment_scan(*, output_directory, random_seed=17001):
    """Generate 25 positions with open references: a candidate, not a microscope recipe yet."""
    folder = Path(output_directory)
    folder.mkdir(parents=True, exist_ok=True)
    xx, yy = np.meshgrid(np.linspace(-4.0, 4.0, 5), np.linspace(-4.0, 4.0, 5))
    commands = np.stack([xx.ravel(), yy.ravel()], axis=1)
    aperture = aperture_specification()
    aperture["radii_m"] = [2e-6]
    parameters = np.r_[0.3, -0.2, np.log([1.7, 0.55, 0.45]), 0.12]
    fractions = np.asarray(
        make_alignment_operator(aperture=aperture, commands=commands)(parameters)
    )
    rng = np.random.default_rng(random_seed)
    incident = 2e7 * np.exp(rng.normal(0, 0.1, len(commands)))
    np.savez_compressed(
        folder / "scan.npz",
        commands=commands,
        transmitted=rng.poisson(incident * fractions),
        reference=rng.poisson(incident),
    )
    (folder / "aperture.json").write_text(json.dumps(aperture, indent=2))
    (folder / "scoring_truth.json").write_text(
        json.dumps(
            {"parameters": parameters.tolist(), "random_seed": random_seed}, indent=2
        )
    )
    return folder


def measure_alignment_scan(*, input_directory):
    """Current readings + aperture size -> current maximum and centring uncertainty.

    The output uncertainty is an orientation-independent physical bound. It is
    a measurement uncertainty, not a soft prior on the aperture position.
    """
    folder = Path(input_directory)
    data = np.load(folder / "scan.npz")
    aperture = json.loads((folder / "aperture.json").read_text())
    predict = make_alignment_operator(aperture=aperture, commands=data["commands"])
    values = data["transmitted"] / data["reference"]
    variance = (
        data["transmitted"] / data["reference"] ** 2 + values**2 / data["reference"]
    )
    weight = jnp.asarray(1 / np.sqrt(variance))
    residual = jax.jit(lambda p: (predict(p) - values) * weight)
    derivative = jax.jit(jax.jacfwd(residual))
    lower, upper = (
        np.r_[-4.0, -4.0, np.log([0.1, 0.01, 0.01]), -10.0],
        np.r_[4.0, 4.0, np.log([20.0, 10.0, 10.0]), 10.0],
    )
    guesses = [
        np.r_[0.0, 0.0, np.log([size, gain, gain]), 0.0]
        for size, gain in [(1.5, 0.5), (3.0, 1.0), (0.7, 0.2)]
    ]
    fits = [
        least_squares(
            lambda p: np.asarray(residual(p)),
            guess,
            jac=lambda p: np.asarray(derivative(p)),
            bounds=(lower, upper),
            x_scale="jac",
            max_nfev=500,
            ftol=1e-10,
            xtol=1e-11,
            gtol=1e-6,
        )
        for guess in guesses
    ]
    good = [fit for fit in fits if fit.success]
    if not good:
        raise RuntimeError("Aperture-centering scan did not converge.")
    best = min(good, key=lambda fit: fit.cost)
    _, singular, vt = np.linalg.svd(best.jac, full_matrices=False)
    covariance = (vt.T / singular) @ (vt / singular[:, None])
    p = best.x
    movement = np.array([[np.exp(p[3]), p[5]], [0.0, np.exp(p[4])]])
    physical_covariance = movement @ covariance[:2, :2] @ movement.T
    result = {
        "peak_command": p[:2].tolist(),
        "beam_rms_um": float(np.exp(p[2])),
        "movement_metric_um2": (movement.T @ movement).tolist(),
        "peak_covariance_command_units": covariance[:2, :2].tolist(),
        "isotropic_centring_sigma_um": float(
            np.sqrt(np.linalg.eigvalsh(physical_covariance).max())
        ),
        "residual_rms": float(np.sqrt(np.mean(best.fun**2))),
        "parameters": p.tolist(),
        "covariance": covariance.tolist(),
        "starts": [
            {
                "parameters": f.x.tolist(),
                "success": bool(f.success),
                "chi_squared": float(2 * f.cost),
            }
            for f in fits
        ],
        "scope": "Assumes a stable round Gaussian illumination, calibrated aperture and a locally linear movement map. The achieved maximum must be checked after moving.",
    }
    (folder / "measurement.json").write_text(json.dumps(result, indent=2))
    return result


def simulate_and_centre_aperture(
    *,
    condenser_parameters,
    output_directory,
    lens_controls=(1.75, 0.75, 1.0),
    random_seed=17001,
):
    """Simulated acquisition: scan a broad beam, estimate its maximum, then centre.

    Only generation uses condenser_parameters. The centring analysis receives
    the scan counts, commands and known aperture size. Movement is assumed
    repeatable in this first test; its real feasibility is still to be confirmed.
    """
    from condenser_aperture_experiment import aperture_moments

    folder = Path(output_directory)
    folder.mkdir(parents=True, exist_ok=True)
    p = np.asarray(condenser_parameters).copy()
    relative_mean, variance = aperture_moments(p, jnp.asarray(lens_controls), "after")
    rotation = np.array([[np.cos(0.4), -np.sin(0.4)], [np.sin(0.4), np.cos(0.4)]])
    movement = rotation @ np.array([[0.55, 0.12], [0.0, 0.45]])
    peak = np.linalg.solve(movement, np.asarray(relative_mean) * 1e6)
    parameters = np.r_[
        peak, 0.5 * np.log(float(variance) * 1e12), np.log([0.55, 0.45]), 0.12
    ]
    xx, yy = np.meshgrid(np.linspace(-4.0, 4.0, 5), np.linspace(-4.0, 4.0, 5))
    commands = np.stack([xx.ravel(), yy.ravel()], axis=1)
    aperture = aperture_specification()
    aperture["radii_m"] = [2e-6]
    fractions = np.asarray(
        make_alignment_operator(aperture=aperture, commands=commands)(parameters)
    )
    rng = np.random.default_rng(random_seed)
    incident = 2e7 * np.exp(rng.normal(0, 0.1, len(commands)))
    np.savez_compressed(
        folder / "scan.npz",
        commands=commands,
        transmitted=rng.poisson(incident * fractions),
        reference=rng.poisson(incident),
    )
    (folder / "aperture.json").write_text(json.dumps(aperture, indent=2))
    measured = measure_alignment_scan(input_directory=folder)
    if measured["residual_rms"] > 1.5:
        raise RuntimeError(
            "Alignment scan does not support the assumed beam/movement model."
        )
    # This is the simulated physical move commanded by the independent analysis.
    p[20:22] += movement @ np.asarray(measured["peak_command"])
    (folder / "scoring_truth.json").write_text(
        json.dumps(
            {
                "scan_parameters": parameters.tolist(),
                "actual_movement_um_per_command": movement.tolist(),
                "centred_condenser_parameters": p.tolist(),
                "random_seed": random_seed,
            },
            indent=2,
        )
    )
    constraint = {
        "lens_controls": list(lens_controls),
        "mean_minus_aperture_centre_m": [0.0, 0.0],
        "isotropic_sigma_m": measured["isotropic_centring_sigma_um"] * 1e-6,
        "current_readings": 50,
        "scope": "Current-maximum measurement from an uncalibrated 2D aperture scan. Repeatable movement is assumed.",
    }
    (folder / "alignment_constraint.json").write_text(json.dumps(constraint, indent=2))
    return p
