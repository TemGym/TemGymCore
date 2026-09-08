"""Coherent TemGym image formation with an explicit detector integral.

The spatial evaluator is an optional forward accelerator. Use scan for JAX
derivatives. An incoherent source ensemble is represented by separate beam
bundles: sum intensities between modes, complex fields within each bundle.
"""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.signal import convolve2d
from jax.scipy.special import ndtr

from .evaluate import evaluate_gaussians
from .run import run_to_end


@dataclass(frozen=True)
class CoherentDetector:
    """Square detector pixels, midpoint quadrature and optional Gaussian PSF.

    ``image`` returns integrated intensity, without field-of-view normalization.
    Refine ``samples_per_pixel`` to check that fringes are resolved. Pixel size
    and z are metres. PSF width is measured in detector pixels.
    """

    shape_yx: tuple
    pixel_pitch_m: float
    z: float
    samples_per_pixel: int = 2
    psf_sigma_pixels: float = 0.0

    def __post_init__(self):
        if (
            len(self.shape_yx) != 2
            or any(int(n) != n or n <= 0 for n in self.shape_yx)
            or not np.isfinite(self.pixel_pitch_m)
            or self.pixel_pitch_m <= 0
            or not isinstance(self.samples_per_pixel, int)
            or self.samples_per_pixel < 1
            or not np.isfinite(self.psf_sigma_pixels)
            or self.psf_sigma_pixels < 0
        ):
            raise ValueError(
                "Positive shape, pitch and integer quadrature order are required."
            )

    @property
    def grid(self):
        n = self.samples_per_pixel
        ny, nx = self.shape_yx
        x = (jnp.arange(nx * n) + 0.5 - nx * n / 2) * self.pixel_pitch_m / n
        y = (jnp.arange(ny * n) + 0.5 - ny * n / 2) * self.pixel_pitch_m / n
        xx, yy = jnp.meshgrid(x, y)
        return SimpleNamespace(
            shape=(ny * n, nx * n), coords=jnp.stack((xx.ravel(), yy.ravel()), axis=-1)
        )

    def image(self, beam, *, method="scan"):
        field = evaluate_gaussians(beam, self.grid, method=method, tile_beams=32)
        n = self.samples_per_pixel
        ny, nx = self.shape_yx
        image = (jnp.abs(field) ** 2).reshape(ny, n, nx, n).mean(axis=(1, 3))
        image = image * self.pixel_pitch_m**2
        if self.psf_sigma_pixels > 0:
            radius = int(np.ceil(4 * self.psf_sigma_pixels))
            axis = jnp.arange(-radius, radius + 1)
            kernel = jnp.exp(-0.5 * (axis / self.psf_sigma_pixels) ** 2)
            kernel = kernel / kernel.sum()
            image = convolve2d(image, kernel[:, None] * kernel[None, :], mode="same")
        return image


def microscope_image(source, components, detector, *, method="scan"):
    """Propagate a Gaussian bundle through source-to-detector components.

    Include specimen and apertures in ``components`` at their physical planes.
    A final Plane supplies the detector distance. The same function can be
    JIT compiled and differentiated with method='scan'.
    """
    from .components import Plane
    from .gaussian import FreeSpacePropagator
    from .solenoid import HardEdgeSolenoid

    # Drop the common axial carrier before adding tiny path differences. This
    # changes no intensity or relative phase within an equal-voltage mode.
    components = tuple(
        replace(c, include_axial_path=False) if isinstance(c, HardEdgeSolenoid) else c
        for c in components
    )
    beam = run_to_end(
        source,
        (*components, Plane(z=detector.z)),
        propagator=FreeSpacePropagator(include_axial_path=False),
    )
    return detector.image(beam, method=method)


def ensemble_image(sources, weights, components, detector, *, method="scan"):
    """Incoherent sum of coherent illumination modes, with supplied weights.

    Weights include mode dose; they are not silently normalized. Each source
    bundle must describe one coherent mode, not an incoherent ray ensemble.
    """
    if len(sources) != len(weights) or not len(sources):
        raise ValueError("Supply one weight per coherent source mode.")
    return sum(
        weight * microscope_image(source, components, detector, method=method)
        for source, weight in zip(sources, weights)
    )


def blurred_gaussian_image(
    beam, detector, *, blur_sigma_m, padding_pixels=4, method="scan", tile_beams=32
):
    """Direct coherent Gaussian sum, then real-space Gaussian intensity blur.

    Input angular spread in round linear optics translates each coherent image
    by B*angle. Its Gaussian ensemble can therefore be averaged by convolution,
    after summing complex Gaussian fields. Supply the combined detector-plane
    RMS of that spread and the detector PSF. Neither propagation nor image
    evaluation uses FFTs. Refine pixel quadrature and padding independently.
    ``tile_beams`` controls GPU work grouping, without changing the sampling.
    """
    if not isinstance(padding_pixels, int) or padding_pixels < 1:
        raise ValueError("Supply a positive integer detector padding.")
    n = detector.samples_per_pixel
    ny, nx = detector.shape_yx
    pad = padding_pixels * n
    x = (
        (jnp.arange(nx * n + 2 * pad) + 0.5 - (nx * n + 2 * pad) / 2)
        * detector.pixel_pitch_m
        / n
    )
    y = (
        (jnp.arange(ny * n + 2 * pad) + 0.5 - (ny * n + 2 * pad) / 2)
        * detector.pixel_pitch_m
        / n
    )
    xx, yy = jnp.meshgrid(x, y)
    grid = SimpleNamespace(
        shape=xx.shape, coords=jnp.stack((xx.ravel(), yy.ravel()), axis=-1)
    )
    field = evaluate_gaussians(beam, grid, method=method, tile_beams=tile_beams)
    intensity = jnp.abs(field) ** 2
    # Integrate the Gaussian blur kernel over each detector pixel analytically.
    # Averaging an already blurred sampled image would add a separate O(h²)
    # pixel-boundary error. Only the coherent intensity integral is sampled here.
    sigma = jnp.maximum(blur_sigma_m, 1e-30)

    def response(count, positions):
        edges = (jnp.arange(count + 1) - count / 2) * detector.pixel_pitch_m
        values = ndtr((edges[:, None] - positions[None, :]) / sigma)
        return values[1:] - values[:-1]

    return (
        response(ny, y)
        @ intensity
        @ response(nx, x).T
        * (detector.pixel_pitch_m / n) ** 2
    )


def make_image_model(
    *,
    source_from_parameters,
    components_from_parameters,
    detector_from_parameters,
    method="scan",
):
    """Compile a physical-parameter → pixel-intensity forward model.

    The supplied constructors keep parameter meaning with the microscope
    configuration. They must be JAX-traceable; discretizations stay fixed
    during a fit. Build source sampling outside them and vary its fields.
    """

    def predict(parameters):
        return microscope_image(
            source_from_parameters(parameters),
            components_from_parameters(parameters),
            detector_from_parameters(parameters),
            method=method,
        )

    return jax.jit(predict)


def fit_images(
    *,
    predict_images,
    images,
    noise_std,
    starting_parameters,
    bounds,
    parameter_scale=None,
    max_evaluations=500,
    relative_cost_tolerance=1e-8,
    gradient_tolerance=1e-5,
):
    """Fit supplied pixels with TemGym JAX least squares and known pixel errors.

    The predictor must use a differentiable renderer ('scan'). Detector dose,
    background and source uncertainty must be included explicitly by the
    predictor if unknown. No generating parameters or priors are supplied.
    """
    from .jax_least_squares import least_squares

    observed, sigma = jnp.asarray(images), jnp.asarray(noise_std)
    if not np.all(np.isfinite(np.asarray(sigma))) or np.any(np.asarray(sigma) <= 0):
        raise ValueError("Pixel noise standard deviations must be positive and finite.")
    if not np.all(np.isfinite(np.asarray(observed))):
        raise ValueError("Observed images must be finite.")
    if np.broadcast_shapes(observed.shape, sigma.shape) != observed.shape:
        raise ValueError("Noise must broadcast to the image shape.")

    def residual(parameters):
        predicted = predict_images(parameters)
        if predicted.shape != observed.shape:
            raise ValueError("Predicted and measured image shapes must match.")
        return ((predicted - observed) / sigma).ravel()

    return least_squares(
        residual,
        starting_parameters,
        bounds=bounds,
        x_scale=parameter_scale,
        max_nfev=max_evaluations,
        max_iter=max_evaluations,
        ftol=relative_cost_tolerance,
        gtol=gradient_tolerance,
    )
