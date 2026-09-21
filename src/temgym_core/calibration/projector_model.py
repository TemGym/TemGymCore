"""Physical projector parameters mapped to native finite-solenoid components."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from temgym_core.components import solenoid_column
from temgym_core.run import column_matrix


def round_matrix(vector):
    """A two-number representation of an isotropic scale and rotation."""
    x, y = vector
    return jnp.array([[x, -y], [y, x]])


def choose_objective_projector_views(views, objective_command):
    """Keep the five useful nominal conjugates by adjusting downstream lenses.

    Setup uses the previously calibrated model with its nominal q(u) law. It
    never uses the hidden response slopes. Actual images may differ from these
    planned maps, which is why the image analysis measures a free pattern.
    """
    p = jnp.asarray(views["calibrated_projector_parameters"])
    matrix = jax.jit(matrix5)
    planned, commands = [], []
    for view in views["views"]:
        magnification, target_b = view["target"]
        start = np.asarray(view["controls"][1:])

        def error(v, magnification=magnification, target_b=target_b):
            command = jnp.r_[objective_command, v]
            m = matrix(p, command)
            angle = p[11] * objective_command + jnp.dot(SIGNS * jnp.exp(p[12:16]), v)
            rotation = round_matrix(jnp.array([jnp.cos(angle), jnp.sin(angle)]))
            aa = rotation @ m[:2, :2]
            bb = rotation @ m[:2, 2:4]
            return jnp.array(
                [
                    aa[0, 0] / magnification - 1,
                    (bb[0, 0] - target_b) / (magnification * 1e-4),
                ]
            )

        objective = jax.jit(error)
        derivative = jax.jit(jax.jacfwd(error))
        fit = least_squares(
            lambda v, objective=objective: np.asarray(objective(v)),
            start,
            jac=lambda v, derivative=derivative: np.asarray(derivative(v)),
            bounds=(0.05, 4.5),
            max_nfev=500,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        if np.linalg.norm(fit.fun) > 1e-7:
            # Two target coefficients leave freedom in four lens currents.
            # A nearby starting point can escape a turning point of one branch.
            for direction in np.r_[np.eye(4), -np.eye(4)]:
                trial = least_squares(
                    lambda v, objective=objective: np.asarray(objective(v)),
                    np.clip(start * (1 + 0.03 * direction), 0.050001, 4.499999),
                    jac=lambda v, derivative=derivative: np.asarray(derivative(v)),
                    bounds=(0.05, 4.5),
                    max_nfev=1000,
                    ftol=1e-12,
                    xtol=1e-12,
                    gtol=1e-12,
                )
                if np.linalg.norm(trial.fun) < np.linalg.norm(fit.fun):
                    fit = trial
                if np.linalg.norm(fit.fun) <= 1e-7:
                    break
            if np.linalg.norm(fit.fun) > 1e-7:
                raise RuntimeError(
                    f"Could not keep projector target A={magnification}, B={target_b}; residual {np.linalg.norm(fit.fun):.3g}."
                )
        command = np.r_[objective_command, fit.x]
        commands.append(command)
        planned.append(np.asarray(matrix(p, command)))
    return np.asarray(planned), np.asarray(commands)


LENSES = ("Objective", "IL1", "IL2", "IL3", "PL1")


SIGNS = jnp.array([1.0, 1.0, 1.0, -1.0])


def initial_guess():
    """Literal engineering guesses, independently specified from the generator."""
    return np.r_[
        np.log([0.0025, 0.19, 0.08, 0.05, 0.06, 0.34]),
        np.log([0.0005, 0.006, 0.004, 0.006, 0.0012]),
        -0.40,
        np.log([0.50, 0.60, 0.50, 0.70]),
    ]


def decode_geometry(p):
    gaps, lengths = jnp.exp(p[:6]), jnp.exp(p[6:11])
    entrances = jnp.cumsum(gaps[:-1]) + jnp.r_[0.0, jnp.cumsum(lengths[:-1])]
    return gaps, lengths, entrances + lengths / 2, jnp.sum(gaps) + jnp.sum(lengths)


def projector_components(p, controls):
    """Executable TemGym projector, from specimen to detector.

    Four controls mean IL1, IL2, IL3, PL1 with the objective held at one.
    Five controls mean Objective, IL1, IL2, IL3, PL1. The objective control
    multiplies its fitted reference strength, just as for the other lenses.
    """
    controls = jnp.asarray(controls)
    if controls.shape not in ((4,), (5,)):
        raise ValueError(
            "Supply four projector controls or five controls including the objective first."
        )
    gaps, lengths, _, _ = decode_geometry(p)
    objective = controls[0] if controls.shape == (5,) else 1.0
    q = jnp.r_[p[11] * objective, SIGNS * jnp.exp(p[12:16]) * controls[-4:]]
    return solenoid_column(gaps, lengths, q)


def matrix5(p, controls):
    """Grouped 5×5 ray map of the same components used for coherent imaging."""
    return column_matrix(projector_components(p, controls))


def generating_parameters():
    """Declared synthetic geometry and excitations; never a fitting input."""
    centres = np.array([0.0023, 0.2135239554, 0.2904801564, 0.3529801564, 0.4079834965])
    lengths = np.array([0.0004, 0.005, 0.005, 0.005, 0.001])
    gaps = np.r_[
        centres[0] - lengths[0] / 2,
        np.diff(centres) - (lengths[1:] + lengths[:-1]) / 2,
        0.71875 - centres[-1] - lengths[-1] / 2,
    ]

    return np.r_[
        np.log(gaps),
        np.log(lengths),
        -0.4257450165772056,
        np.log([0.55, 0.65, 0.45, 0.75]),
    ]


def position_prior_coordinates(priors):
    """Map bounded centre/detector coordinates to the column's log gaps.

    Position coordinates are standardized by the stated prior standard
    deviations. Bounds must guarantee positive gaps even at their corners;
    this makes every optimizer iterate an ordered, non-overlapping column.
    """
    mean = np.asarray(priors["position_mean_mm"], dtype=float)
    sigma = np.asarray(priors["position_sigma_mm"], dtype=float)
    bounds = np.asarray(priors["position_bounds_mm"], dtype=float)
    lengths = np.asarray(priors["length_bounds_mm"], dtype=float)
    if (
        mean.shape != (6,)
        or sigma.shape != (6,)
        or bounds.shape != (6, 2)
        or lengths.shape != (5, 2)
    ):
        raise ValueError("Expected six position priors and five lens-length bounds.")
    if (
        not all(np.isfinite(a).all() for a in (mean, sigma, bounds, lengths))
        or np.any(sigma <= 0)
        or np.any(bounds[:, 1] <= bounds[:, 0])
        or np.any(lengths[:, 0] <= 0)
        or np.any(lengths[:, 1] <= lengths[:, 0])
        or np.any(mean < bounds[:, 0])
        or np.any(mean > bounds[:, 1])
    ):
        raise ValueError(
            "Prior means, widths, and bounds must be finite and consistent."
        )
    minimum_gaps = np.r_[
        bounds[0, 0] - lengths[0, 1] / 2,
        bounds[1:5, 0] - bounds[:4, 1] - (lengths[1:, 1] + lengths[:-1, 1]) / 2,
        bounds[5, 0] - bounds[4, 1] - lengths[4, 1] / 2,
    ]
    if np.any(minimum_gaps <= 0):
        raise ValueError(
            "Position/length bounds must guarantee positive non-overlapping gaps."
        )

    def encode(p):
        _, _, centres, detector = decode_geometry(p)
        return (
            jnp.asarray(p).at[:6].set((jnp.r_[centres, detector] * 1000 - mean) / sigma)
        )

    def decode(u):
        z = (mean + sigma * u[:6]) / 1000
        length = jnp.exp(u[6:11])
        gaps = jnp.r_[
            z[0] - length[0] / 2,
            jnp.diff(z[:5]) - (length[1:] + length[:-1]) / 2,
            z[5] - z[4] - length[4] / 2,
        ]
        return jnp.asarray(u).at[:6].set(jnp.log(gaps))

    return encode, decode, (bounds[:, 0] - mean) / sigma, (bounds[:, 1] - mean) / sigma


def model_to_parameters(model):
    """Convert named lens geometry and strengths to 16 physical coordinates."""
    centres = np.array([model["lens_positions_mm"][name] for name in LENSES]) / 1000
    lengths = np.array([model["lens_lengths_mm"][name] for name in LENSES]) / 1000
    detector = model["detector_position_mm"] / 1000
    gaps = np.r_[
        centres[0] - lengths[0] / 2,
        np.diff(centres) - (lengths[1:] + lengths[:-1]) / 2,
        detector - centres[-1] - lengths[-1] / 2,
    ]
    gains = np.array([model["excitation_gains"][name] for name in LENSES[1:]])
    if np.any(gaps <= 0) or np.any(lengths <= 0) or np.any(gains <= 0):
        raise ValueError(
            "Starting positions must leave positive gaps; lengths and gain magnitudes must be positive."
        )
    return np.r_[
        np.log(gaps),
        np.log(lengths),
        model["objective_strength"],
        np.log(gains),
    ]


def parameters_to_model(parameters):
    """Give the fitted numbers physical names and units."""
    _, lengths, centres, detector = decode_geometry(parameters)
    return {
        "lens_positions_mm": dict(zip(LENSES, (np.asarray(centres) * 1000).tolist())),
        "lens_lengths_mm": dict(zip(LENSES, (np.asarray(lengths) * 1000).tolist())),
        "detector_position_mm": float(detector) * 1000,
        "objective_strength": float(parameters[11]),
        "excitation_gains": dict(zip(LENSES[1:], np.exp(parameters[12:16]).tolist())),
    }


def make_starting_models(
    *, prior_model, position_priors, number_of_starts, random_seed=482
):
    """One declared guess plus reproducible, dispersed starts within the same bounds.

    Position starts cover the inner 80% of the allowed boxes. Lengths and gains
    vary around the fixed prior model; this is an exploration, not a posterior
    sample or an exhaustive search of the parameter space.
    """
    if number_of_starts < 1:
        raise ValueError("At least one starting model is required.")
    p = model_to_parameters(prior_model)
    encode, decode, lower, upper = position_prior_coordinates(position_priors)
    centre = np.asarray(encode(p))
    length_bounds = np.log(np.asarray(position_priors["length_bounds_mm"]) / 1000)
    rng = np.random.default_rng(random_seed)
    models = [prior_model]
    for _ in range(number_of_starts - 1):
        u = centre.copy()
        u[:6] = lower + rng.uniform(0.1, 0.9, 6) * (upper - lower)
        u[6:11] = np.clip(
            u[6:11] + rng.normal(0, 0.25, 5),
            length_bounds[:, 0] + 0.01,
            length_bounds[:, 1] - 0.01,
        )
        u[11] = rng.uniform(-0.6, -0.25)
        u[12:16] += rng.uniform(-0.25, 0.25, 4)
        models.append(parameters_to_model(np.asarray(decode(u))))
    return models


def at_voltage(parameters, voltage, reference=200000.0):
    from .condenser_model import magnetic_strength_ratio

    ratio = magnetic_strength_ratio(voltage, reference)
    return jnp.asarray(parameters).at[11].multiply(ratio).at[12:16].add(jnp.log(ratio))


def projector_coordinates(bounds_path):
    import json
    from pathlib import Path

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
        return decode(u)

    return encode, physical, lower, upper
