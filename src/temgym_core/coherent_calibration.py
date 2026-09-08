"""Named, bounded JAX calibration of coherent microscope pixels and currents.

This module owns predictions, likelihoods, multiple starts and local uncertainty.
An example supplies instrument settings, known specimen and independent guesses.
"""

from dataclasses import dataclass, is_dataclass, replace

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np

from .jax_least_squares import least_squares


def measure_image_centroids(*, images, background_counts, read_noise_counts):
    """Intensity centroids and shot/read-noise covariance from measured pixels.

    Coordinates are detector x/y pixels relative to the array centre. The
    calibrated background is subtracted without clipping negative residuals.
    Assumes independent pixels and known background; detector cropping can
    bias a centroid and must be checked by the acquisition. No optical model
    or true image positions enter this measurement.
    """
    images = np.asarray(images, dtype=float)
    if images.ndim != 3 or not np.isfinite(images).all():
        raise ValueError("Supply finite images with shape (frames, y, x).")
    if not np.isfinite([background_counts, read_noise_counts]).all() or min(
        background_counts, read_noise_counts
    ) < 0:
        raise ValueError("Supply nonnegative calibrated background and read noise.")
    height, width = images.shape[1:]
    yy, xx = np.meshgrid(
        np.arange(height) - (height - 1) / 2,
        np.arange(width) - (width - 1) / 2, indexing="ij",
    )
    coordinates = np.stack((xx, yy), axis=-1)
    signal = images - background_counts
    flux = signal.sum(axis=(1, 2))
    if np.any(flux <= 0):
        raise ValueError("Each image must contain positive background-subtracted flux.")
    centres = np.einsum("nyx,yxc->nc", signal, coordinates) / flux[:, None]
    delta = coordinates[None] - centres[:, None, None]
    variance = np.maximum(images, 0) + read_noise_counts**2
    covariance = np.einsum("nyx,nyxi,nyxj->nij", variance, delta, delta)
    covariance /= flux[:, None, None] ** 2
    return {
        "centres_xy_pixels": centres,
        "covariances_pixels2": covariance,
        "signal_counts": flux,
    }


@jdc.pytree_dataclass
class CalibrationState:
    microscope: object
    image_dose: object
    image_background: object
    current_dose: object


def _get(state, path):
    for key in path:
        state = getattr(state, key) if isinstance(key, str) else state[key]
    return state


def _set(state, path, value):
    if not path:
        return value
    key, *tail = path
    child = getattr(state, key) if isinstance(key, str) else state[key]
    child = _set(child, tail, value)
    if is_dataclass(state):
        return replace(state, **{key: child})
    if isinstance(state, tuple):
        return tuple(child if i == key else old for i, old in enumerate(state))
    return jnp.asarray(state).at[key].set(child)


@dataclass(frozen=True)
class FitParameter:
    """Named physical quantity and engineering bounds, in SI units.

    A logarithmic coordinate preserves a known sign; scale sets displayed and
    numerical units. Bounds are physical, e.g. metres for a lens length.
    """

    name: str
    path: tuple
    lower: float
    upper: float
    scale: float = 1.0
    logarithmic: bool = False
    sign: float = 1.0

    def encode(self, value):
        return (
            jnp.log(value / (self.sign * self.scale))
            if self.logarithmic
            else value / self.scale
        )

    def decode(self, value):
        return (
            self.sign * self.scale * jnp.exp(value)
            if self.logarithmic
            else self.scale * value
        )


class Parameterization:
    def __init__(self, starting_state, parameters):
        self.starting_state = starting_state
        self.parameters = tuple(parameters)
        if not self.parameters or len({p.name for p in self.parameters}) != len(
            self.parameters
        ):
            raise ValueError("Supply distinct named fit parameters.")
        if len({p.path for p in self.parameters}) != len(self.parameters):
            raise ValueError("A physical field may occur only once in a fit.")
        lower, upper = [], []
        for p in self.parameters:
            if (
                not np.isfinite([p.lower, p.upper, p.scale]).all()
                or p.lower >= p.upper
                or p.scale <= 0
            ):
                raise ValueError(f"Invalid physical bounds/scale: {p.name}")
            if p.logarithmic and (
                p.sign not in (-1.0, 1.0)
                or p.lower * p.sign <= 0
                or p.upper * p.sign <= 0
            ):
                raise ValueError(
                    f"Logarithmic bounds must retain the declared sign: {p.name}"
                )
            endpoints = sorted(float(p.encode(v)) for v in (p.lower, p.upper))
            lower.append(endpoints[0])
            upper.append(endpoints[1])
        self.bounds = (np.asarray(lower), np.asarray(upper))
        self.initial = np.asarray(self.encode(starting_state))
        if (
            not np.isfinite(self.initial).all()
            or np.any(self.initial < lower)
            or np.any(self.initial > upper)
        ):
            raise ValueError(
                "Engineering starting state must lie within all declared bounds."
            )

    def encode(self, state):
        return jnp.stack([p.encode(_get(state, p.path)) for p in self.parameters])

    def decode(self, coordinates):
        state = self.starting_state
        for p, value in zip(self.parameters, coordinates):
            state = _set(state, p.path, p.decode(value))
        return state

    def values(self, state):
        return {p.name: float(_get(state, p.path)) for p in self.parameters}


@dataclass(frozen=True)
class RecordedSettings:
    """Recorded settings: separate image frames and aperture-current readings.

    Each row includes voltage, source index and four condenser commands.
    Images also have five projector commands, tilt_xy and shift_xy. Currents
    use hole=-1 for an open reference. Objective commands remain one when fixed.
    """

    images: tuple
    currents: tuple


def make_predictor(settings, specimen, numerical):
    images, currents = settings.images, settings.currents

    def pack(rows, key):
        return jnp.asarray([row[key] for row in rows])

    if images:
        image_rows = tuple(
            pack(images, key)
            for key in (
                "voltage",
                "source_index",
                "condenser",
                "projector",
                "tilt_xy",
                "shift_xy",
            )
        )
    if currents:
        current_rows = tuple(
            pack(currents, key)
            for key in ("voltage", "source_index", "condenser", "hole")
        )

    def predict(state):
        model = state.microscope
        if images:

            def frame(row):
                voltage, index, condenser, projector, tilt, shift, dose, background = (
                    row
                )
                return model.image(
                    condenser,
                    projector,
                    voltage,
                    index,
                    specimen,
                    numerical,
                    dose=dose,
                    background=background,
                    tilt_xy=tilt,
                    shift_xy=shift,
                )

            pixels = jax.lax.map(
                frame, (*image_rows, state.image_dose, state.image_background)
            )
        else:
            pixels = jnp.empty((0, *numerical.detector_shape))
        if currents:

            def reading(row):
                voltage, index, controls, hole = row
                fraction = jax.lax.cond(
                    hole < 0,
                    lambda: jnp.asarray(1.0),
                    lambda: model.current_fraction(
                        controls, voltage, index, hole, numerical
                    ),
                )
                return fraction * state.current_dose[index]

            counts = jax.lax.map(reading, current_rows)
        else:
            counts = jnp.empty((0,))
        return pixels, counts

    return jax.jit(predict)


def simulate_observations(predict, generating_state, *, random_seed=1, read_noise=0.0):
    """Generation only: return arrays; no generating parameters accompany them."""
    pixels, currents = (np.asarray(v) for v in predict(generating_state))
    if (
        not np.isfinite(pixels).all()
        or not np.isfinite(currents).all()
        or np.any(pixels < 0)
        or np.any(currents < 0)
    ):
        raise ValueError(
            "Generating model produced invalid counts; check physical sampling."
        )
    rng = np.random.default_rng(random_seed)
    measured_images = rng.poisson(pixels).astype(float) + rng.normal(
        0, read_noise, pixels.shape
    )
    measured_currents = rng.poisson(currents).astype(float)
    return {
        "images": measured_images,
        "currents": measured_currents,
        "image_sigma": np.sqrt(np.maximum(measured_images, 1.0) + read_noise**2),
        "current_sigma": np.sqrt(np.maximum(measured_currents, 1.0)),
    }


def fit_microscope(
    *,
    predictor,
    observations,
    parameterization,
    starting_coordinates=None,
    max_evaluations=200,
    relative_cost_tolerance=1e-8,
):
    """Fit original coherent pixels/current counts from every independent start.

    Bounds are not soft priors. Rank deficiency yields no finite covariance;
    solver convergence and small residuals never certify unique geometry.
    """
    observed_images = jnp.asarray(observations["images"])
    observed_currents = jnp.asarray(observations["currents"])
    image_sigma, current_sigma = (
        jnp.asarray(observations[key]) for key in ("image_sigma", "current_sigma")
    )
    if any(
        not np.isfinite(np.asarray(v)).all() or np.any(np.asarray(v) <= 0)
        for v in (image_sigma, current_sigma)
    ):
        raise ValueError("Measurement uncertainties must be positive and finite.")
    if (
        image_sigma.shape != observed_images.shape
        or current_sigma.shape != observed_currents.shape
    ):
        raise ValueError("Supply one uncertainty for every recorded pixel/current.")
    if not all(
        np.isfinite(np.asarray(v)).all() for v in (observed_images, observed_currents)
    ):
        raise ValueError("Observations must be finite.")

    def residual(coordinates):
        pixels, currents = predictor(parameterization.decode(coordinates))
        if (
            pixels.shape != observed_images.shape
            or currents.shape != observed_currents.shape
        ):
            raise ValueError("Prediction and observation shapes differ.")
        return jnp.concatenate(
            (
                ((pixels - observed_images) / image_sigma).ravel(),
                ((currents - observed_currents) / current_sigma).ravel(),
            )
        )

    residual = jax.jit(residual)
    starts = (
        [parameterization.initial]
        if starting_coordinates is None
        else starting_coordinates
    )
    records = []
    fits = []
    for index, start in enumerate(starts):
        result = least_squares(
            residual,
            start,
            bounds=parameterization.bounds,
            max_nfev=max_evaluations,
            max_iter=max_evaluations,
            ftol=relative_cost_tolerance,
            gtol=1e-5,
        )
        fits.append(result)
        record = {
            "index": index,
            "success": result.success,
            "message": result.message,
            "evaluations": result.nfev,
            "chi_squared": 2 * result.cost if np.isfinite(result.cost) else None,
            "residual_rms": float(jnp.sqrt(jnp.mean(result.residual**2)))
            if np.isfinite(result.cost)
            else None,
            "starting_coordinates": np.asarray(start).tolist(),
            "coordinates": np.asarray(result.x).tolist(),
            "values": parameterization.values(parameterization.decode(result.x)),
        }
        records.append(record)
        print(
            f"Start {index + 1}: RMS {record['residual_rms']}; {result.message}",
            flush=True,
        )
    if not fits:
        raise ValueError("Supply at least one starting guess.")
    candidates = [i for i, fit in enumerate(fits) if fit.success] or list(
        range(len(fits))
    )
    best_index = min(candidates, key=lambda i: fits[i].cost)
    best = fits[best_index]
    jacobian = np.asarray(best.jacobian)
    norms = np.linalg.norm(jacobian, axis=0)
    jacobian_finite = bool(np.isfinite(jacobian).all())
    singular = (
        np.linalg.svd(jacobian / np.maximum(norms, 1e-300), compute_uv=False)
        if jacobian_finite
        else np.array([])
    )
    rank = int(np.sum(singular > singular[0] * 1e-9)) if singular.size else None
    covariance = None
    if rank == jacobian.shape[1]:
        _, s, vt = np.linalg.svd(jacobian, full_matrices=False)
        coordinate_covariance = (vt.T / s) @ (vt / s[:, None])
        scales = np.asarray(
            [
                jax.grad(p.decode)(best.x[i])
                for i, p in enumerate(parameterization.parameters)
            ]
        )
        covariance = coordinate_covariance * scales[:, None] * scales[None, :]
    degrees_of_freedom = (
        max(0, int(best.residual.size) - rank) if rank is not None else None
    )
    p_value = (
        float(jax.scipy.special.gammaincc(degrees_of_freedom / 2, best.cost))
        if degrees_of_freedom
        else None
    )
    lower, upper = parameterization.bounds
    on_bound = (
        np.minimum(np.asarray(best.x) - lower, upper - np.asarray(best.x))
        / (upper - lower)
    ) < 1e-5
    report = {
        "starts": records,
        "best_index": best_index,
        "rank": rank,
        "jacobian_finite": jacobian_finite,
        "degrees_of_freedom": degrees_of_freedom,
        "chi_squared_upper_tail": p_value,
        "data_fit_accepted": bool(
            best.success and p_value is not None and p_value > 0.01
        ),
        "parameters_at_bounds": [
            p.name for p, active in zip(parameterization.parameters, on_bound) if active
        ],
        "number_of_parameters": jacobian.shape[1],
        "normalized_singular_values": singular.tolist(),
        "covariance": None if covariance is None else covariance.tolist(),
        "parameter_names": [p.name for p in parameterization.parameters],
        "covariance_scope": "Local physical-parameter covariance, conditional on any fixed fields; no global uniqueness claim.",
    }
    return parameterization.decode(best.x), report


def microscope_parameters(
    starting_state,
    *,
    fit_angular_spread=False,
    fit_exposures=False,
    fit_backgrounds=False,
):
    """Engineering bounds for all source/lens/aperture geometry, without priors.

    Positive lengths/gains vary by a factor of two around the supplied guess;
    polarities, lens order, aperture diameters and detector pixels are known.
    Transverse hole centres are bounded at ±0.2 µm (the alignment assumption).
    Source positions/angles can differ independently at each voltage.
    Extra angular spread requires multiple coherent modes (angular_order >= 2).
    """
    parameters = []

    def positive(name, path, scale=1.0, sign=1.0):
        value = float(_get(starting_state, path))
        a, b = sorted((value / 2, value * 2))
        parameters.append(FitParameter(name, path, a, b, scale, True, sign))

    for train, names in (
        ("condenser", ("CL1", "CL3", "Mini", "Objective pre")),
        ("projector", ("Objective post", "IL1", "IL2", "IL3", "PL1")),
    ):
        root = ("microscope", train)
        for i in range(1 if train == "condenser" else 0, len(names) + 1):
            positive(f"{train} gap {i} (m)", (*root, "gaps", i), 1e-3)
        for i, name in enumerate(names):
            positive(f"{name} length (m)", (*root, "lengths", i), 1e-3)
            positive(
                f"{name} q gain (rad)",
                (*root, "strengths", i),
                1.0,
                np.sign(float(_get(starting_state, (*root, "strengths", i)))),
            )
    for i, source in enumerate(starting_state.microscope.sources):
        root = ("microscope", "sources", i)
        for field in ("upstream_distance", "rms_size"):
            positive(
                f"source {i} {field}",
                (*root, field),
                1e-3 if field == "upstream_distance" else 1e-9,
            )
        for field, limit, scale in (
            ("x", 30e-9, 1e-9),
            ("y", 30e-9, 1e-9),
            ("theta_x", 1e-4, 1e-5),
            ("theta_y", 1e-4, 1e-5),
        ):
            parameters.append(
                FitParameter(
                    f"source {i} {field}", (*root, field), -limit, limit, scale
                )
            )
        if fit_angular_spread:
            positive(
                f"source {i} additional_angular_rms",
                (*root, "additional_angular_rms"),
                1e-5,
            )
    parameters.append(
        FitParameter(
            "aperture gap fraction",
            ("microscope", "aperture", "gap_fraction"),
            0.002,
            0.98,
        )
    )
    for hole in range(len(starting_state.microscope.aperture.radii)):
        for axis in range(2):
            parameters.append(
                FitParameter(
                    f"hole {hole} centre {'xy'[axis]} (m)",
                    ("microscope", "aperture", "centres_xy", hole, axis),
                    -0.2e-6,
                    0.2e-6,
                    1e-6,
                )
            )
    if fit_exposures:
        for field in ("image_dose", "current_dose"):
            for i in range(len(getattr(starting_state, field))):
                positive(f"{field} {i}", (field, i))
    if fit_backgrounds:
        for i in range(len(starting_state.image_background)):
            parameters.append(
                FitParameter(
                    f"image background {i}", ("image_background", i), 0.0, 100.0, 10.0
                )
            )
    return tuple(parameters)
