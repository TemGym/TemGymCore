"""MAGICAL images of a Gaussian illumination measured through calibrated projectors.

The beam at the specimen has an unknown mean position/angle, round size,
position-angle correlation and conditional angular spread (seven numbers).
Image analysis estimates these independently at each condenser setting. It has
no access to the condenser parameters or its generating source.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from condenser_calibration_model import generating_parameters, specimen_moments
from magical_projector_pipeline import make_magical_template
from scipy.optimize import brentq, least_squares

from temgym_core.gaussian_image_likelihood import normal_interval_probability


def beam_coordinates(mean, covariance):
    """Dimensionless means, log spatial RMS, scaled curvature and log conditional angular RMS."""
    variance = covariance[0, 0]
    curvature = covariance[0, 2] / variance
    conditional = covariance[2, 2] - covariance[0, 2] ** 2 / variance
    return jnp.r_[
        mean[:2] / 1e-7,
        mean[2:] / 1e-3,
        0.5 * jnp.log(variance),
        curvature / 1e4,
        0.5 * jnp.log(jnp.maximum(conditional, 1e-30)),
    ]


def predict_beam_coordinates(parameters, controls):
    mean, covariance = specimen_moments(parameters, controls)
    beam = beam_coordinates(mean, covariance)
    # Round symplectic transport preserves emittance. This expression avoids
    # subtracting almost equal variances for strongly converging illumination.
    return beam.at[6].set(
        parameters[17] + parameters[18] - 0.5 * jnp.log(covariance[0, 0])
    )


def make_image_operator(*, projector_matrices, template, detector):
    """Integrate a known Gaussian-mixture specimen times a Gaussian beam exactly.

    Multiplying each specimen component by the illumination gives a Gaussian
    with an updated centre and width. Conditional source angles then propagate
    through the known projector. A known detector PSF is included in the width.
    The operator takes seven beam coordinates followed by dose/background pairs.
    """
    matrices = jnp.asarray(projector_matrices)
    a, b = matrices[:, :2, :2], matrices[:, :2, 2:4]
    # This renderer integrates circular Gaussians. General astigmatic maps
    # would require a full two-dimensional Gaussian pixel integral.
    for block in (np.asarray(a), np.asarray(b)):
        if not (
            np.allclose(block[:, 0, 0], block[:, 1, 1])
            and np.allclose(block[:, 0, 1], -block[:, 1, 0])
        ):
            raise ValueError(
                "This renderer requires round, rotational projector blocks."
            )
    if not np.allclose(np.asarray(matrices)[:, :2, 4], 0.0):
        raise ValueError("Remove calibrated detector offsets before using these maps.")
    axis = jnp.asarray(template["axis_m"])
    xx, yy = jnp.meshgrid(axis, axis)
    centres = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)
    transmission = jnp.asarray(template["intensity"]).ravel()
    kernel_variance = detector["sample_kernel_sigma_m"] ** 2
    pitch = detector["pixel_pitch_m"]
    ny, nx = detector["shape_yx"]
    xe = jnp.arange(nx + 1) - nx / 2
    ye = jnp.arange(ny + 1) - ny / 2

    def render(parameters, projector_matrices_override=None):
        # A known magnetic projector model can supply a differentiable voltage
        # correction while each specimen beam is measured independently.
        if projector_matrices_override is None:
            a_used, b_used = a, b
        else:
            a_used = projector_matrices_override[:, :2, :2]
            b_used = projector_matrices_override[:, :2, 2:4]
        mean_position, mean_angle = parameters[:2] * 1e-7, parameters[2:4] * 1e-3
        spatial_variance = jnp.exp(2 * parameters[4])
        curvature, angular_variance = parameters[5] * 1e4, jnp.exp(2 * parameters[6])
        combined = spatial_variance + kernel_variance
        product_centres = (
            spatial_variance * centres + kernel_variance * mean_position
        ) / combined
        product_variance = spatial_variance * kernel_variance / combined
        log_weights = jnp.log(transmission) - jnp.sum(
            (centres - mean_position) ** 2, axis=1
        ) / (2 * combined)
        weights = jax.nn.softmax(log_weights)
        maps = a_used + curvature * b_used
        shifts = jnp.einsum("nij,j->ni", b_used, mean_angle - curvature * mean_position)
        detector_centres = (
            jnp.einsum("nij,kj->nki", maps, product_centres) + shifts[:, None]
        ) / pitch
        variance = (
            product_variance * jnp.sum(maps[:, :, 0] ** 2, axis=1)
            + angular_variance * jnp.sum(b_used[:, :, 0] ** 2, axis=1)
        ) / pitch**2
        widths = jnp.sqrt(variance + detector["psf_sigma_pixels"] ** 2)
        exposures = parameters[7:].reshape(-1, 2)

        def frame(xy, width, exposure):
            px = normal_interval_probability(
                (xe[:-1, None] - xy[:, 0]) / width, (xe[1:, None] - xy[:, 0]) / width
            )
            py = normal_interval_probability(
                (ye[:-1, None] - xy[:, 1]) / width, (ye[1:, None] - xy[:, 1]) / width
            )
            return jnp.exp(exposure[0]) * ((py * weights) @ px.T) + jnp.exp(exposure[1])

        return jax.vmap(frame)(detector_centres, widths, exposures)

    return jax.jit(render)


def choose_condenser_controls(parameters, *, reference_settings=12, random_seed=534):
    """Simulated instrument setup: tune the mini to a visible illumination range.

    This is generation/setup, not inference. It uses the generating model to
    make a reproducible instrument scenario. No parameters or beam moments from
    this step are passed to the image or condenser fit.
    """
    from condenser_calibration_model import matrix5

    rng = np.random.default_rng(random_seed)
    matrix = jax.jit(matrix5)
    moments = jax.jit(specimen_moments)
    controls = []
    candidates = rng.uniform([0.65, 0.65], [1.45, 1.4], (reference_settings * 25, 2))
    for index, pair in enumerate(candidates):
        target_b = [0.00035, -0.00035, 0.00065, -0.00065][index % 4]

        def error(mini, pair=pair, target_b=target_b):
            m = np.asarray(matrix(parameters, np.r_[pair, mini]))
            # Derotate using the known signed sum only inside simulated setup.
            q = np.dot(np.exp(parameters[9:12]), np.r_[pair, mini]) - np.exp(
                parameters[12]
            )
            return np.cos(-q) * m[0, 2] + np.sin(-q) * m[1, 2] - target_b

        grid = np.linspace(0.6, 1.7, 65)
        values = [error(v) for v in grid]
        roots = [
            brentq(error, x, y)
            for x, y, fx, fy in zip(grid[:-1], grid[1:], values[:-1], values[1:])
            if fx * fy < 0
        ]
        if not roots:
            continue
        row = np.r_[pair, min(roots, key=lambda v: abs(v - 1.15))]
        _, covariance = moments(parameters, row)
        if np.sqrt(covariance[2, 2]) > 0.9e-3:
            continue
        controls.append(row)
        if len(controls) == reference_settings:
            break
    if len(controls) != reference_settings:
        raise RuntimeError("Could not find enough visible condenser settings.")
    return np.asarray(controls)


def simulate_condenser_images(
    *, projector_matrices, output_directory, settings=12, random_seed=901, controls=None
):
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    parameters = generating_parameters()
    controls = (
        choose_condenser_controls(parameters, reference_settings=settings)
        if controls is None
        else np.asarray(controls)
    )
    template, _metadata = make_magical_template(
        200000.0, pitch_m=5e-9, illumination_radius_m=130e-9
    )
    detector = {
        "pixel_pitch_m": 220e-6,
        "shape_yx": [64, 64],
        "read_noise_e": 2.0,
        "psf_sigma_pixels": 0.5,
        "sample_kernel_sigma_m": 2.5e-9,
        "voltage_v": 200000.0,
        "detector_origin": "Known from projector calibration; no unknown inter-frame drift in this test",
    }
    operator = make_image_operator(
        projector_matrices=projector_matrices, template=template, detector=detector
    )
    rng = np.random.default_rng(random_seed)
    images, beam_truth = [], []
    for index, commands in enumerate(controls):
        beam = np.asarray(predict_beam_coordinates(parameters, commands))
        exposure = np.tile(np.log([4e7, 3.0]), (len(projector_matrices), 1)).ravel()
        mean = np.asarray(operator(np.r_[beam, exposure]))
        images.append(
            rng.poisson(mean).astype(float)
            + rng.normal(0, detector["read_noise_e"], mean.shape)
        )
        beam_truth.append(beam)
        print(f"Generated condenser setting {index + 1}/{len(controls)}", flush=True)
    np.savez_compressed(
        directory / "images.npz",
        images=np.asarray(images),
        controls=controls,
        projector_matrices=projector_matrices,
    )
    np.savez_compressed(directory / "template.npz", **template)
    (directory / "detector.json").write_text(json.dumps(detector, indent=2))
    (directory / "scoring_truth.json").write_text(
        json.dumps(
            {
                "parameters": parameters.tolist(),
                "beam_coordinates": np.asarray(beam_truth).tolist(),
                "random_seed": random_seed,
            },
            indent=2,
        )
    )
    return directory


def measure_condenser_images(
    *, input_directory, max_evaluations=200, initialization_only=False
):
    """Fit the specimen illumination in each set of projector views, independently.

    When initialization_only is True, nominal-map residuals may exceed the
    quality threshold: these are guesses for a later calibrated pixel fit, not
    accepted measurements. That later fit must validate its pixel residuals.

    Only images, calibrated projector maps and the public specimen/detector
    model are read. The source and condenser generating parameters are absent.
    """
    directory = Path(input_directory)
    data = np.load(directory / "images.npz")
    images, projectors = data["images"], data["projector_matrices"]
    detector = json.loads((directory / "detector.json").read_text())
    operator = make_image_operator(
        projector_matrices=projectors,
        template=dict(np.load(directory / "template.npz")),
        detector=detector,
    )
    derivative = jax.jit(jax.jacfwd(operator))
    n = len(projectors)
    lower = np.r_[
        [-2, -2, -10, -10],
        np.log(3e-9),
        -10.0,
        np.log(1e-7),
        np.tile(np.log([1e5, 0.01]), n),
    ]
    upper = np.r_[
        [2, 2, 10, 10],
        np.log(120e-9),
        10.0,
        np.log(0.01),
        np.tile(np.log([1e10, 100.0]), n),
    ]
    results, covariances, all_parameters, rms_values = [], [], [], []
    for index, frames in enumerate(images):
        noise = np.sqrt(np.maximum(frames, 1.0) + detector["read_noise_e"] ** 2)

        def residual(p, frames=frames, noise=noise):
            return ((np.asarray(operator(p)) - frames) / noise).ravel()

        def jacobian(p, noise=noise):
            return (
                np.asarray(derivative(p)).reshape(-1, len(p)) / noise.ravel()[:, None]
            )

        # Broad curvature search, using only pixel residuals. The first view is
        # focused at the specimen, so it supplies a rough illumination width.
        weights = np.maximum(frames[0] - np.median(frames[0, :4]), 0.0)
        yy, xx = np.indices(weights.shape)
        centroid = (
            np.array([(weights * xx).sum(), (weights * yy).sum()]) / weights.sum()
        )
        a = projectors[0, :2, :2]
        centre = np.linalg.solve(
            a,
            (centroid - (np.array(weights.shape[::-1]) - 1) / 2)
            * detector["pixel_pitch_m"],
        )
        variance = ((xx - centroid[0]) ** 2 + (yy - centroid[1]) ** 2) * weights
        size = (
            np.sqrt(variance.sum() / weights.sum() / 2)
            * detector["pixel_pitch_m"]
            / np.linalg.norm(a[:, 0])
        )
        candidates = []
        for h in np.linspace(-4.0, 4.0, 17):
            p = np.r_[
                centre / 1e-7,
                [0.0, 0.0],
                np.log(np.clip(size, 10e-9, 80e-9)),
                h,
                np.log(20e-6),
                np.tile(np.log([4e7, 3.0]), n),
            ]
            candidates.append((np.mean(residual(p) ** 2), p))
        fits = []
        for _, seed in sorted(candidates, key=lambda item: item[0])[:2]:
            fits.append(
                least_squares(
                    residual,
                    seed,
                    jac=jacobian,
                    bounds=(lower, upper),
                    max_nfev=max_evaluations,
                    x_scale="jac",
                    ftol=1e-9,
                    xtol=1e-10,
                    gtol=1e-5,
                )
            )
        fit = min(fits, key=lambda f: f.cost)
        rms = float(np.sqrt(np.mean(fit.fun**2)))
        if not fit.success or (rms > 1.5 and not initialization_only):
            raise RuntimeError(
                f"Image fit failed at setting {index}: {fit.message}; RMS {rms:.3g}"
            )
        _, s, vt = np.linalg.svd(fit.jac, full_matrices=False)
        covariance = (vt.T / s) @ (vt / s[:, None])
        results.append(fit.x[:7])
        covariances.append(covariance[:7, :7])
        all_parameters.append(fit.x)
        rms_values.append(rms)
        print(
            f"Measured setting {index + 1}/{len(images)}: pixel RMS {rms:.3f}, "
            f"size {np.exp(fit.x[4]) * 1e9:.2f} nm, curvature {fit.x[5] * 1e4:.1f}/m, "
            f"conditional spread {np.exp(fit.x[6]) * 1e6:.2f} µrad",
            flush=True,
        )
    np.savez_compressed(
        directory / "measurements.npz",
        controls=data["controls"],
        values=results,
        covariance=covariances,
        image_parameters=all_parameters,
        pixel_residual_rms=rms_values,
        initialization_only=initialization_only,
    )
    return directory / "measurements.npz"
