"""Analytic incoherent Gaussian intensity components.

These integrate a phase-space intensity distribution, not a coherent exit wave.
They complement GaussianBeam without changing its amplitude conventions.
Distances are metres except for the explicitly pixel-based detector method.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .gaussian_image_likelihood import normal_interval_probability


@dataclass(frozen=True)
class GaussianIntensitySample:
    """Positive Gaussian-mixture specimen with a common round kernel."""

    centres: object
    transmission: object
    kernel_variance: object

    def illuminate(self, mean_position, spatial_variance):
        combined = spatial_variance + self.kernel_variance
        centres = (
            spatial_variance * self.centres + self.kernel_variance * mean_position
        ) / combined
        variance = spatial_variance * self.kernel_variance / combined
        log_weights = jnp.log(self.transmission) - jnp.sum(
            (self.centres - mean_position) ** 2, axis=-1
        ) / (2 * combined)
        return centres, variance, jax.nn.softmax(log_weights)


@dataclass(frozen=True)
class GaussianIntensityDetector:
    """Integrate round Gaussian spots over finite pixels, including a known PSF.

    Arbitrary astigmatic/correlated spots require a different pixel integral.
    Weights are integrated component intensities; flux outside the finite
    sensor is lost rather than renormalized onto visible pixels.
    """

    shape_yx: tuple
    pixel_pitch_m: float = 1.0
    psf_sigma_pixels: float = 0.0

    def image_pixels(self, centres, variance, weights, dose=1.0, background=0.0):
        ny, nx = self.shape_yx
        xe, ye = jnp.arange(nx + 1) - nx / 2, jnp.arange(ny + 1) - ny / 2
        width = jnp.sqrt(variance + self.psf_sigma_pixels**2)
        px = normal_interval_probability(
            (xe[:-1, None] - centres[:, 0]) / width,
            (xe[1:, None] - centres[:, 0]) / width,
        )
        py = normal_interval_probability(
            (ye[:-1, None] - centres[:, 1]) / width,
            (ye[1:, None] - centres[:, 1]) / width,
        )
        return dose * ((py * weights) @ px.T) + background

    def image(self, centres_m, variance_m2, weights, dose=1.0, background=0.0):
        return self.image_pixels(
            centres_m / self.pixel_pitch_m,
            variance_m2 / self.pixel_pitch_m**2,
            weights,
            dose,
            background,
        )


def radial_aperture_transmission(
    *,
    radii_m,
    edge_width_m,
    voltage_v,
    order=256,
    edge_samples=256,
    beamlet_waist_m=0.5e-9,
):
    """Round Gaussian phase-space flux through TemGym sigmoid apertures.

    The radial quadrature follows the beam support, retaining narrow crossovers.
    Component beamlet powers are tabulated once; interpolation and quadrature
    use JAX. Refine both orders separately when using a new physical regime.
    """
    import numpy as np
    from jax.scipy.special import i0e

    from .components import SigmoidAperture
    from .gaussian import make_gaussian

    nodes, weights = map(jnp.asarray, np.polynomial.legendre.leggauss(order))
    tables = []
    for radius in radii_m:
        limit = radius + 12 * edge_width_m
        grid = jnp.linspace(
            1e-15, limit, int(np.ceil(limit / edge_width_m * edge_samples)) + 1
        )
        beam = make_gaussian(
            x=grid,
            y=0.0,
            voltage=voltage_v,
            waist_x=beamlet_waist_m,
            waist_y=beamlet_waist_m,
        )
        out = SigmoidAperture(
            radius=radius, edge_width=edge_width_m, sharpness=1.0, z=0.0
        )(beam)
        power = jnp.abs(out.amplitude / beam.amplitude) ** 2 * jnp.sqrt(
            jnp.linalg.det(jnp.imag(beam.Q_inv)) / jnp.linalg.det(jnp.imag(out.Q_inv))
        )
        tables.append((grid, power, limit))

    def transmission(mean, variance):
        variance = variance - beamlet_waist_m**2 / 4
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


@jax.jit
def fit_intensity_scale_and_background(shape, image, noise):
    """Exact two-variable nonnegative least squares for exposure and background."""
    design = (
        jnp.stack((shape.ravel(), jnp.ones(shape.size)), axis=1)
        / noise.ravel()[:, None]
    )
    target = image.ravel() / noise.ravel()
    norms = jnp.linalg.norm(design, axis=0)
    matrix = design / jnp.where(norms > 0, norms, 1.0)
    solution = jnp.linalg.lstsq(matrix, target, rcond=None)[0]
    single = jnp.maximum(matrix.T @ target, 0.0)
    candidates = jnp.stack(
        (
            jnp.zeros(2),
            jnp.array([single[0], 0.0]),
            jnp.array([0.0, single[1]]),
            solution,
        )
    )
    costs = jnp.sum((matrix @ candidates.T - target[:, None]) ** 2, axis=0)
    costs = costs.at[3].set(jnp.where(jnp.all(solution >= 0), costs[3], jnp.inf))
    return candidates[jnp.argmin(costs)] / jnp.where(norms > 0, norms, 1.0)
