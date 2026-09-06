"""Independent ellipse-rotation images and aperture sweeps through focus."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from condenser_aperture_experiment import (
    aperture_moments,
    aperture_specification,
    measure_aperture_currents,
)
from condenser_calibration_model import generating_parameters, matrix5, source_moments
from jax.scipy.special import i0e
from scipy.optimize import least_squares, minimize_scalar

from temgym_core.components import SigmoidAperture
from temgym_core.gaussian import make_gaussian


def make_radial_transmission(*, specification, order=256, edge_samples=256):
    """Integrate actual SigmoidAperture beamlet power, including a narrow crossover.

    The radial interval follows the Gaussian's support. A fixed interval can
    miss a nanometre crossover in a micrometre aperture. A tabulated component
    power curve keeps differentiation inexpensive; its convergence is tested.
    """
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes, weights = jnp.asarray(nodes), jnp.asarray(weights)
    waist = 0.5e-9
    tables = []
    for radius in specification["radii_m"]:
        limit = radius + 12 * specification["edge_width_m"]
        grid = jnp.linspace(
            1e-15,
            limit,
            int(np.ceil(limit / specification["edge_width_m"] * edge_samples)) + 1,
        )
        beam = make_gaussian(
            x=grid,
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
        out = aperture(beam)
        power = jnp.abs(out.amplitude / beam.amplitude) ** 2 * jnp.sqrt(
            jnp.linalg.det(jnp.imag(beam.Q_inv)) / jnp.linalg.det(jnp.imag(out.Q_inv))
        )
        tables.append((grid, power, limit))

    def transmission(mean, variance):
        variance = variance - waist**2 / 4
        sigma = jnp.sqrt(variance)
        offset = jnp.sqrt(jnp.dot(mean, mean) + 1e-40)
        results = []
        for grid, power, limit in tables:
            lower = jnp.maximum(0.0, offset - 10 * sigma)
            upper = jnp.minimum(limit, offset + 10 * sigma)
            width = jnp.maximum(upper - lower, 0.0)
            radial = lower + (nodes + 1) * width / 2
            density = (
                radial
                / variance
                * jnp.exp(-((radial - offset) ** 2) / (2 * variance))
                * i0e(radial * offset / variance)
            )
            results.append(
                jnp.sum(weights * width / 2 * density * jnp.interp(radial, grid, power))
            )
        return jnp.stack(results)

    return jax.jit(transmission)


def make_focus_transmission(*, specification, order=256, edge_samples=256):
    """Condenser/source parameters -> power through the physical aperture."""
    power = make_radial_transmission(
        specification=specification, order=order, edge_samples=edge_samples
    )

    def transmission(p, controls):
        mean, variance = aperture_moments(p, controls, "after")
        return power(mean, variance)

    return jax.jit(transmission)


def make_ellipse_renderer(shape=(64, 64), integration_order=4):
    """Pixel-integrated elliptical Gaussian beam; no specimen features or lens parameters."""
    nodes, weights = np.polynomial.legendre.leggauss(integration_order)
    yy, xx = jnp.indices(shape)
    x = xx[..., None, None] + jnp.asarray(nodes)[None, None, :, None] / 2
    y = yy[..., None, None] + jnp.asarray(nodes)[None, None, None, :] / 2
    quadrature = jnp.asarray(weights[:, None] * weights[None, :]) / 4

    def image(p):
        dx, dy = x - p[0], y - p[1]
        c, s = jnp.cos(p[4]), jnp.sin(p[4])
        a, b = c * dx + s * dy, -s * dx + c * dy
        density = jnp.exp(
            -0.5 * ((a / jnp.exp(p[2])) ** 2 + (b / jnp.exp(p[3])) ** 2)
        ) / (2 * jnp.pi * jnp.exp(p[2] + p[3]))
        return jnp.exp(p[5]) * jnp.sum(density * quadrature, axis=(-2, -1)) + jnp.exp(
            p[6]
        )

    return jax.jit(image)


def generate_rotation_and_focus(
    *, projector_views_path, protocol_path, output_directory, random_seed=9101
):
    """Generate ellipse images and bracketed current sweeps. Only generation knows truth."""
    folder = Path(output_directory)
    folder.mkdir(parents=True, exist_ok=True)
    p = np.r_[generating_parameters(), np.log(0.0025 / (0.056 - 0.0025)), 0.2, -0.15]
    views = json.loads(Path(projector_views_path).read_text())
    projector = np.asarray(views["views"][0]["matrix"])
    controls = np.asarray(
        json.loads(Path(protocol_path).read_text())["calibration_controls"]
    )
    rng = np.random.default_rng(random_seed)
    # Stable astigmatic illumination prepared upstream of CL1. Its orientation
    # is unknown to inference, and its widths are measured separately per image.
    source_angle = 0.35
    axis_ratio = 1.6
    rot = np.array(
        [
            [np.cos(source_angle), -np.sin(source_angle)],
            [np.sin(source_angle), np.cos(source_angle)],
        ]
    )
    shape = rot @ np.diag([axis_ratio, 1 / axis_ratio]) @ rot.T
    mean, cov = source_moments(p)
    cov = np.asarray(cov).copy()
    cov[:2, :2] = np.exp(2 * p[17]) * shape
    cov[2:, 2:] = np.exp(2 * p[18]) * shape
    renderer = make_ellipse_renderer()
    images = []
    for c in controls:
        transfer = projector[:4, :4] @ np.asarray(matrix5(p, c))[:4, :4]
        camera_mean = (transfer @ mean)[:2] / 220e-6 + 31.5
        camera_cov = (transfer @ cov @ transfer.T)[:2, :2] / 220e-6**2 + np.eye(
            2
        ) * 0.5**2
        values, vectors = np.linalg.eigh(camera_cov)
        angle = np.arctan2(vectors[1, -1], vectors[0, -1])
        params = np.r_[
            camera_mean, 0.5 * np.log(values[::-1]), angle, np.log([2e7, 3.0])
        ]
        intensity = np.asarray(renderer(params))
        images.append(rng.poisson(intensity) + rng.normal(0, 2.0, intensity.shape))
    np.savez_compressed(
        folder / "rotation_images.npz",
        images=images,
        controls=controls,
        projector_rotation=np.arctan2(projector[1, 0], projector[0, 0]),
        read_noise_e=2.0,
    )

    spec = aperture_specification()
    transmit = make_focus_transmission(specification=spec)
    # Setup locates crossovers in the simulated instrument, then records the
    # controls. No fitted crossover location is passed to inference.
    commands = []
    groups = []
    peaks = []
    for group, cl3 in enumerate([0.75, 1.0, 1.3]):
        result = minimize_scalar(
            lambda cl1, cl3=cl3: float(
                aperture_moments(p, jnp.array([cl1, cl3, 1.0]), "after")[1]
            ),
            bounds=(1.3, 3.0),
            method="bounded",
            options={"xatol": 1e-12},
        )
        peak = result.x
        peaks.append(peak)
        for cl1 in peak + np.linspace(-0.65, 0.65, 41):
            commands.append([cl1, cl3, 1.0])
            groups.append(group)
    commands = np.asarray(commands)
    fractions = np.asarray(
        jax.vmap(transmit, in_axes=(None, 0))(p, jnp.asarray(commands))
    )
    if fractions.min() < 0 or fractions.max() > 1.000002:
        raise RuntimeError("Aperture integration failed.")
    incident = 2e7 * np.exp(rng.normal(0, 0.1, len(commands)))
    np.savez_compressed(
        folder / "calibration_counts.npz",
        controls=commands,
        transmitted_electrons=rng.poisson(incident[:, None] * fractions),
        reference_electrons=rng.poisson(incident),
        groups=groups,
    )
    (folder / "aperture.json").write_text(json.dumps(spec, indent=2))
    (folder / "scoring_truth.json").write_text(
        json.dumps(
            {
                "parameters": p.tolist(),
                "source_angle": source_angle,
                "axis_ratio": axis_ratio,
                "focus_controls": peaks,
                "random_seed": random_seed,
            },
            indent=2,
        )
    )
    return folder


def measure_rotation_and_focus(*, input_directory):
    """Fit each ellipse from pixels, subtract known projector rotation, measure currents."""
    folder = Path(input_directory)
    data = np.load(folder / "rotation_images.npz")
    renderer = make_ellipse_renderer()
    derivative = jax.jit(jax.jacfwd(renderer))
    angles = []
    variances = []
    pixel_rms = []
    parameters = []
    for frames in data["images"]:
        background = np.median(np.r_[frames[:3].ravel(), frames[-3:].ravel()])
        weights = np.maximum(frames - background, 0.0)
        yy, xx = np.indices(frames.shape)
        centre = np.array([(weights * xx).sum(), (weights * yy).sum()]) / weights.sum()
        coords = np.stack([xx - centre[0], yy - centre[1]], axis=-1)
        cov = np.einsum("ij,ija,ijb->ab", weights, coords, coords) / weights.sum()
        values, vectors = np.linalg.eigh(cov)
        guess = np.r_[
            centre,
            0.5 * np.log(values[::-1]),
            np.arctan2(vectors[1, -1], vectors[0, -1]),
            np.log([weights.sum(), max(background, 0.1)]),
        ]
        noise = np.sqrt(np.maximum(frames, 1.0) + float(data["read_noise_e"]) ** 2)
        fit = least_squares(
            lambda p, frames=frames, noise=noise: (
                (np.asarray(renderer(p)) - frames) / noise
            ).ravel(),
            guess,
            jac=lambda p, noise=noise: (
                np.asarray(derivative(p)).reshape(-1, 7) / noise.ravel()[:, None]
            ),
            max_nfev=100,
            x_scale="jac",
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-5,
        )
        if not fit.success:
            raise RuntimeError("Ellipse fit did not converge.")
        _, s, vt = np.linalg.svd(fit.jac, full_matrices=False)
        cov = (vt.T / s) @ (vt / s[:, None])
        angle = (
            fit.x[4]
            + (np.pi / 2 if fit.x[3] > fit.x[2] else 0)
            - float(data["projector_rotation"])
        )
        angles.append(0.5 * np.arctan2(np.sin(2 * angle), np.cos(2 * angle)))
        variances.append(cov[4, 4])
        pixel_rms.append(float(np.sqrt(np.mean(fit.fun**2))))
        parameters.append(fit.x)
    np.savez_compressed(
        folder / "rotation_measurements.npz",
        controls=data["controls"],
        angles=angles,
        variance=variances,
        pixel_rms=pixel_rms,
        image_parameters=parameters,
    )
    measure_aperture_currents(input_directory=folder)
    return {
        "rotation": folder / "rotation_measurements.npz",
        "current": folder / "calibration_measurements.npz",
    }
