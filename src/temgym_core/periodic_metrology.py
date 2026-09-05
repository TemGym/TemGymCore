"""Fail-closed, registered-frequency metrology for resolved multilayer images.

This is deliberately separate from the provenance-pinned legacy Fourier-peak
extractor. FFT maxima initialise an image-domain harmonic fit; the FFT never
provides the measurement or its covariance. A missing/ambiguous fundamental,
inadequate template, or search-boundary optimum is not a valid measurement.

The conditional image model is an independently measured illumination field
times a quadratic background plus a periodic harmonic series. Harmonic phases
and amplitudes are nuisance parameters, so spatially invariant defocus/MTF
contrast changes do not change the fitted frequency. This is not a general
finite-specimen, lattice, diffraction or dynamical-scattering model. In
particular, arbitrary illumination errors and edge/fringe structure must pass
the image residual gate; they are not absorbed into a larger error bar.

Images have (y, x) ordering; frequencies are (x, y), in cycles per detector
pixel, NOT radians/pixel or FFT bins. Variance is fixed, independently supplied
Poisson-plus-read variance. Its calibration and the illumination/certificate
uncertainties must be propagated separately before an instrument likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


def _coordinates(shape_yx):
    ny, nx = shape_yx
    y, x = jnp.meshgrid(
        (jnp.arange(ny) - (ny - 1) / 2) / ny,
        (jnp.arange(nx) - (nx - 1) / 2) / nx,
        indexing="ij",
    )
    return x.ravel(), y.ravel()


def _design(frequency_bins_xy, xy, illumination, harmonics):
    x, y = xy
    phase = 2 * jnp.pi * (frequency_bins_xy[0] * x + frequency_bins_xy[1] * y)
    columns = [jnp.ones_like(x), x, y, x * x, x * y, y * y]
    for order in range(1, harmonics + 1):
        columns.extend((jnp.cos(order * phase), jnp.sin(order * phase)))
    return jnp.stack(columns, axis=1) * illumination[:, None]


def _detector_blur(matrix, shape_yx, sigma):
    """Separable nearest-edge Gaussian PSF, with SciPy's four-sigma support."""
    if sigma == 0:
        return matrix
    radius = int(4 * sigma + .5)
    axis = jnp.arange(-radius, radius + 1)
    weights = jnp.exp(-.5 * (axis / sigma)**2)
    weights /= jnp.sum(weights)
    planes = jnp.moveaxis(matrix.reshape(*shape_yx, -1), -1, 0)[..., None]
    for direction in (0, 1):
        padding = ((0, 0), (radius, radius) if direction == 0 else (0, 0),
                   (radius, radius) if direction == 1 else (0, 0), (0, 0))
        kernel = weights.reshape((-1, 1, 1, 1) if direction == 0 else (1, -1, 1, 1))
        planes = jax.lax.conv_general_dilated(
            jnp.pad(planes, padding, mode="edge"), kernel, (1, 1), "VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
    return jnp.moveaxis(planes[..., 0], 0, -1).reshape(matrix.shape)


def periodic_template(
    frequency_xy_per_pixel,
    coefficients,
    shape_yx,
    *,
    illumination_yx=None,
    harmonics=3,
    mtf_sigma_px=0.,
):
    """Differentiable model; coefficient order is 1,x,y,x²,xy,y²,cos1,sin1,...

    Polynomial coordinates are centred and divided by the respective image
    width/height. ``shape_yx``, ``harmonics`` and ``mtf_sigma_px`` are static
    under JIT. Illumination is BEFORE the detector MTF; the Gaussian MTF acts
    on the entire modulated finite envelope, not just harmonic amplitudes.
    This function does not clip negative values or renormalise image intensity.
    """
    illumination = (
        jnp.ones(shape_yx) if illumination_yx is None else jnp.asarray(illumination_yx)
    )
    bins = jnp.asarray(frequency_xy_per_pixel) * jnp.asarray(shape_yx[::-1])
    matrix = _design(bins, _coordinates(shape_yx), illumination.ravel(), harmonics)
    matrix = _detector_blur(matrix, shape_yx, mtf_sigma_px)
    return (matrix @ jnp.asarray(coefficients)).reshape(shape_yx)


@dataclass(frozen=True)
class PeriodicFrequencyMeasurement:
    """Conditional result; check ``accepted`` before consuming any covariance."""

    accepted: bool
    reasons: tuple[str, ...]
    frequency_xy_per_pixel: np.ndarray
    covariance_xy_per_pixel2: np.ndarray
    coefficients: np.ndarray
    fitted_image_yx: np.ndarray
    chi_squared: float
    reduced_chi_squared: float
    fundamental_delta_chi_squared: float
    alternative_frequencies_xy_per_pixel: np.ndarray
    fft_candidates_xy_per_pixel: np.ndarray
    converged: bool
    uncertainty_scope: str = "conditional on fixed illumination and detector variance"

    def require_accepted(self):
        """Prevent an unresolved feature from silently entering a likelihood."""
        if not self.accepted:
            raise ValueError("Unresolved periodic feature: " + "; ".join(self.reasons))
        return self


def _fft_seeds(image, expected_bins, radius, dc_exclusion_bins):
    """Detect on the original spectrum, then apply masks to candidates only."""
    ny, nx = image.shape
    window = np.hanning(ny)[:, None] * np.hanning(nx)[None, :]
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2((image - image.mean()) * window)))
    interior = spectrum[1:-1, 1:-1]
    maximum = np.ones(interior.shape, dtype=bool)
    strict = np.zeros(interior.shape, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                neighbour = spectrum[1 + dy:ny - 1 + dy, 1 + dx:nx - 1 + dx]
                maximum &= interior >= neighbour
                strict |= interior > neighbour
    rows, columns = np.nonzero(maximum & strict & (interior > 0))
    rows, columns = rows + 1, columns + 1
    vectors = np.column_stack((columns - nx // 2, rows - ny // 2)).astype(float)
    selected = (np.linalg.norm(vectors, axis=1) > dc_exclusion_bins) & (
        np.linalg.norm(vectors - expected_bins, axis=1) < radius
    )
    vectors = vectors[selected]
    power = spectrum[rows[selected], columns[selected]]
    return vectors[np.argsort(power)[::-1]][:8]


def fit_periodic_frequency(
    image_yx,
    expected_frequency_xy_per_pixel,
    variance_yx,
    *,
    illumination_yx=None,
    valid_mask_yx=None,
    harmonics=3,
    mtf_sigma_px=0.,
    search_radius_bins=1.5,
    dc_exclusion_bins=2.0,
    minimum_fundamental_delta_chi_squared=36.0,
    maximum_reduced_chi_squared=1.5,
    ambiguity_delta_chi_squared=9.0,
    max_iterations=60,
):
    """Fit one registered periodic family with JAX derivatives, without SciPy.

    ``expected_frequency_xy_per_pixel`` defines a frozen registration window,
    not a prior/penalty or measurement. The search disk is in anisotropically
    scaled FFT bins. It must be set independently of each noisy frame, and
    must not cross a harmonic alias or Nyquist. Multiple FFT seeds and a small
    fixed offset grid are fitted; ``alternative_frequencies`` reports distinct
    competitive minima that were found, not a proof no others exist.

    At least three cycles across the detector, an unmasked FFT maximum and a
    significant fundamental are required. Illumination must be specified before
    the detector MTF (for example by a fitted vacuum-beam model); a blurred
    flat-field image is not an interchangeable input when ``mtf_sigma_px>0``.
    Unknown variance is not estimated from the fit residuals:
    systematic model errors must fail the goodness-of-fit gate. The covariance
    is the local variable-projection Gauss--Newton inverse information, with
    linear illumination/background/harmonic coefficients profiled out.

    An optional frozen boolean ``valid_mask_yx`` selects raw detector pixels
    inside a registered polygon, without resampling/introducing correlations.
    The full image still initialises the FFT; only valid pixels enter the fit
    and its degrees of freedom. Leave an MTF halo around the physical feature
    boundary: the mask is applied after, not before, detector convolution.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Periodic metrology requires jax_enable_x64=True.")
    image = np.asarray(image_yx, dtype=float)
    if image.ndim != 2 or min(image.shape) < 16 or not np.all(np.isfinite(image)):
        raise ValueError("image_yx must be finite and at least 16x16.")
    variance = np.broadcast_to(np.asarray(variance_yx, dtype=float), image.shape)
    illumination = (
        np.ones_like(image) if illumination_yx is None else np.asarray(illumination_yx, dtype=float)
    )
    valid = np.ones(image.shape, dtype=bool) if valid_mask_yx is None else np.asarray(valid_mask_yx)
    if valid.shape != image.shape or valid.dtype.kind != "b" or np.count_nonzero(valid) < 64:
        raise ValueError("valid_mask_yx must be boolean, image-shaped, and retain at least 64 pixels.")
    expected = np.asarray(expected_frequency_xy_per_pixel, dtype=float)
    if expected.shape != (2,) or not np.all(np.isfinite(expected)):
        raise ValueError("Expected frequency must be a finite (x,y) pair.")
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0):
        raise ValueError("Variance must be finite and strictly positive.")
    if (illumination.shape != image.shape or not np.all(np.isfinite(illumination))
            or np.any(illumination < 0) or not np.any(illumination > 0)):
        raise ValueError("Illumination must be finite, non-negative and match the image.")
    if np.count_nonzero(valid & (illumination > 0)) < 64:
        raise ValueError("At least 64 valid pixels must receive nonzero illumination.")
    if int(harmonics) != harmonics or not 1 <= harmonics <= 8:
        raise ValueError("harmonics must be an integer from 1 to 8.")
    if int(max_iterations) != max_iterations or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer.")
    limits = np.asarray([search_radius_bins, minimum_fundamental_delta_chi_squared,
                         maximum_reduced_chi_squared, ambiguity_delta_chi_squared])
    if not np.all(np.isfinite(limits)) or np.any(limits <= 0):
        raise ValueError("Search radius and likelihood thresholds must be positive finite numbers.")
    if not np.isfinite(dc_exclusion_bins) or dc_exclusion_bins < 0:
        raise ValueError("DC exclusion must be finite and non-negative.")
    if not np.isfinite(mtf_sigma_px) or not 0 <= mtf_sigma_px <= 8:
        raise ValueError("MTF sigma must be finite and between zero and eight pixels.")
    scale = np.asarray(image.shape[::-1], dtype=float)
    expected_bins = expected * scale
    if np.linalg.norm(expected_bins) - search_radius_bins < 3:
        raise ValueError("The entire search window must retain at least three specimen cycles.")
    if np.any(harmonics * (np.abs(expected_bins) + search_radius_bins) / scale >= .5):
        raise ValueError("Retained harmonics and their search window must be below Nyquist.")

    xy = _coordinates(image.shape)
    points = np.column_stack([np.asarray(coordinate)[valid.ravel()] for coordinate in xy])
    phases = points @ expected_bins
    separation = points[np.argmax(phases)] - points[np.argmin(phases)]
    if np.ptp(phases) - search_radius_bins * np.linalg.norm(separation) < 3:
        raise ValueError("Valid pixels must span at least three specimen cycles throughout the search window.")
    illumination_jax = jnp.asarray(illumination.ravel())
    inverse_sigma = jnp.asarray(valid.ravel() / np.sqrt(variance.ravel()))
    data = jnp.asarray(image.ravel()) * inverse_sigma

    def image_design(bins):
        matrix = _design(bins, xy, illumination_jax, harmonics)
        return _detector_blur(matrix, image.shape, mtf_sigma_px)

    def linear_fit(bins):
        matrix = image_design(bins) * inverse_sigma[:, None]
        gram = matrix.T @ matrix
        # A roundoff-sized stabiliser, not a physical coefficient prior.
        coefficient = jnp.linalg.solve(gram + 1e-14 * jnp.diag(jnp.diag(gram)), matrix.T @ data)
        return matrix @ coefficient - data, coefficient

    residual = jax.jit(lambda bins: linear_fit(bins)[0])
    value_jacobian = jax.jit(lambda bins: (linear_fit(bins)[0], jax.jacfwd(lambda q: linear_fit(q)[0])(bins)))
    fitted_coefficients = jax.jit(lambda bins: linear_fit(bins)[1])
    candidates = _fft_seeds(image, expected_bins, search_radius_bins, dc_exclusion_bins)
    offsets = np.asarray([[0, 0], [.5, 0], [-.5, 0], [0, .5], [0, -.5]])
    seeds = np.vstack((candidates, expected_bins + min(.8, search_radius_bins / 2) * offsets))
    solutions = []
    for seed in seeds:
        q, damping, converged = seed.copy(), 1e-3, False
        for _ in range(max_iterations):
            r, jac = (np.asarray(value) for value in value_jacobian(q))
            normal, gradient = jac.T @ jac, jac.T @ r
            # Gradient is measured in local standard-error units. Resolving
            # the optimum to 1e-5 sigma is already far below detector noise;
            # a machine-epsilon cost-decrease test can stall on noisy images.
            if np.max(np.abs(gradient) / np.sqrt(np.maximum(np.diag(normal), 1e-30))) < 1e-5:
                converged = True
                break
            step = np.linalg.solve(normal + damping * np.diag(np.maximum(np.diag(normal), 1e-20)), -gradient)
            step *= min(1., .5 / max(np.linalg.norm(step), 1e-30))
            trial = q + step
            displacement = trial - expected_bins
            trial = expected_bins + displacement * min(1., search_radius_bins / max(np.linalg.norm(displacement), 1e-30))
            new_r = np.asarray(residual(trial))
            if new_r @ new_r < r @ r:
                q = trial
                damping = max(damping / 3, 1e-10)
                if np.linalg.norm(step) < 1e-9:
                    converged = True
                    break
            else:
                damping *= 10
                if damping > 1e12:
                    break
        r, jac = (np.asarray(value) for value in value_jacobian(q))
        solutions.append((float(r @ r), q, jac, converged))
    solutions.sort(key=lambda solution: solution[0])
    chi_squared, bins, jac, converged = solutions[0]
    coefficient = np.asarray(fitted_coefficients(bins))
    alternatives = []
    for cost, other, _, _ in solutions[1:]:
        if cost - chi_squared <= ambiguity_delta_chi_squared and np.linalg.norm(other - bins) > .05:
            if not any(np.linalg.norm(other - previous) <= .05 for previous in alternatives):
                alternatives.append(other)
    matrix = np.asarray(image_design(bins)) * np.asarray(inverse_sigma)[:, None]
    null_matrix = np.delete(matrix, (6, 7), axis=1)
    null_coefficient = np.linalg.lstsq(null_matrix, np.asarray(data), rcond=None)[0]
    null_residual = null_matrix @ null_coefficient - np.asarray(data)
    fundamental_delta = max(0., float(null_residual @ null_residual) - chi_squared)
    degrees_of_freedom = np.count_nonzero(valid) - matrix.shape[1] - 2
    reduced = chi_squared / degrees_of_freedom
    reasons = []
    if not converged:
        reasons.append("frequency optimiser did not converge")
    if np.linalg.norm(bins - expected_bins) >= .98 * search_radius_bins:
        reasons.append("frequency lies on the registration search boundary")
    if fundamental_delta < minimum_fundamental_delta_chi_squared:
        reasons.append("no significant fundamental modulation")
    if not len(candidates) or np.min(np.linalg.norm(candidates - bins, axis=1)) > 1.5:
        reasons.append("no corresponding maximum in the unmasked Fourier spectrum")
    if reduced > maximum_reduced_chi_squared:
        reasons.append("harmonic image model is inconsistent with supplied detector noise")
    if alternatives:
        reasons.append("multiple competitive frequency minima")
    if np.linalg.matrix_rank(matrix) < matrix.shape[1] or np.linalg.matrix_rank(jac) < 2:
        reasons.append("frequency or linear nuisance parameters are rank deficient")
    try:
        covariance = np.linalg.inv(jac.T @ jac) / scale[:, None] / scale[None, :]
        if not np.all(np.isfinite(covariance)) or np.min(np.linalg.eigvalsh(covariance)) <= 0:
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        covariance = np.full((2, 2), np.nan)
        reasons.append("frequency information is singular")
    fitted_image = np.asarray(periodic_template(bins / scale, coefficient, image.shape,
                                                illumination_yx=illumination, harmonics=harmonics,
                                                mtf_sigma_px=mtf_sigma_px))
    return PeriodicFrequencyMeasurement(
        not reasons, tuple(reasons), bins / scale, covariance, coefficient, fitted_image,
        chi_squared, reduced, fundamental_delta,
        np.asarray(alternatives).reshape(-1, 2) / scale, candidates / scale, converged,
    )
