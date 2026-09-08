"""Image-based beam initialization before the joint moment fit."""

import hashlib
import json
from contextlib import redirect_stdout
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from temgym_core.intensity import GaussianIntensityDetector, GaussianIntensitySample
from temgym_core.jax_least_squares import least_squares

from .condenser_model import specimen_moments


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
    sample = GaussianIntensitySample(centres, transmission, kernel_variance)
    sensor = GaussianIntensityDetector(
        tuple(detector["shape_yx"]), pitch, detector["psf_sigma_pixels"]
    )

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
        product_centres, product_variance, weights = sample.illuminate(
            mean_position, spatial_variance
        )
        maps = a_used + curvature * b_used
        shifts = jnp.einsum("nij,j->ni", b_used, mean_angle - curvature * mean_position)
        detector_centres = (
            jnp.einsum("nij,kj->nki", maps, product_centres) + shifts[:, None]
        ) / pitch
        variance = (
            product_variance * jnp.sum(maps[:, :, 0] ** 2, axis=1)
            + angular_variance * jnp.sum(b_used[:, :, 0] ** 2, axis=1)
        ) / pitch**2
        exposures = parameters[7:].reshape(-1, 2)

        def frame(xy, variance, exposure):
            return sensor.image_pixels(
                xy, variance, weights, jnp.exp(exposure[0]), jnp.exp(exposure[1])
            )

        return jax.vmap(frame)(detector_centres, variance, exposures)

    return jax.jit(render)


def measure_condenser_images(
    *,
    input_directory,
    max_evaluations=200,
    initialization_only=False,
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
    image_fit_converged = []

    def residual(p, frames, noise):
        return ((operator(p) - frames) / noise).ravel()

    def jacobian(p, frames, noise):
        return derivative(p).reshape(-1, len(p)) / noise.ravel()[:, None]

    for index, frames in enumerate(images):
        noise = np.sqrt(np.maximum(frames, 1.0) + detector["read_noise_e"] ** 2)

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
            candidates.append((np.mean(np.asarray(residual(p, frames, noise)) ** 2), p))
        fits = []
        for _, seed in sorted(candidates, key=lambda item: item[0])[:2]:
            fits.append(
                least_squares(
                    residual,
                    seed,
                    jac=jacobian,
                    args=(jnp.asarray(frames), jnp.asarray(noise)),
                    bounds=(lower, upper),
                    max_nfev=max_evaluations,
                    max_iter=2 * max_evaluations,
                    x_scale="jac",
                    ftol=1e-9,
                    xtol=1e-10,
                    gtol=1e-5,
                )
            )
        fit = min(fits, key=lambda f: f.cost)
        rms = float(np.sqrt(np.mean(fit.residual**2)))
        if not fit.success or (rms > 1.5 and not initialization_only):
            raise RuntimeError(
                f"Image fit failed at setting {index}: {fit.message}; RMS {rms:.3g}"
            )
        _, s, vt = np.linalg.svd(fit.jacobian, full_matrices=False)
        covariance = (vt.T / s) @ (vt / s[:, None])
        results.append(fit.x[:7])
        covariances.append(covariance[:7, :7])
        all_parameters.append(fit.x)
        rms_values.append(rms)
        image_fit_converged.append(fit.success)
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
        image_fit_converged=image_fit_converged,
        initialization_only=initialization_only,
    )
    return directory / "measurements.npz"


def _fingerprint(paths, settings):
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for path in paths:
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _cached(folder, stage, signature, outputs):
    record = Path(folder) / f"{stage}_cache.json"
    return (
        record.exists()
        and json.loads(record.read_text())["signature"] == signature
        and all(Path(path).exists() for path in outputs)
    )


def _record(folder, stage, signature):
    (Path(folder) / f"{stage}_cache.json").write_text(
        json.dumps({"signature": signature})
    )


def measure_images(*, image_directory, reuse_completed=True, initialization_only=False):
    """Supply images and known imaging model; return measured beams with covariance."""
    folder = Path(image_directory)
    signature = _fingerprint(
        [
            folder / "images.npz",
            folder / "template.npz",
            folder / "detector.json",
            Path(__file__).with_name("condenser_images.py"),
        ],
        {"initialization_only": initialization_only},
    )
    output = folder / "measurements.npz"
    if not reuse_completed or not _cached(folder, "measurement", signature, [output]):
        with (folder / "measurement.log").open("w") as log, redirect_stdout(log):
            measure_condenser_images(
                input_directory=folder, initialization_only=initialization_only
            )
        _record(folder, "measurement", signature)
    print(
        f"Measured beams ready for {len(np.load(output)['controls'])} condenser settings."
    )
    return output
