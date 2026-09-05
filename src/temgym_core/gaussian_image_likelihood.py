"""Finite-pixel images of transported Gaussian phase-space distributions.

This is the detector model for the imaging-only condenser experiment. A
partially coherent Gaussian beam has a Gaussian Wigner distribution; its
spatial marginal after a first-order map is consequently Gaussian. This
module computes that marginal, including pixel area, detector affine, and
light arriving from outside the detector crop through a finite discrete PSF.
It supplies no independent ellipse-angle observation.

The PSF acts on the expected arrival distribution. Independent Poisson/read
noise applied afterwards is a declared synthetic detector model, not a model
of correlated charge sharing, saturation, or threshold losses.
"""

from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr


def detector_psf_kernel(mtf_sigma_px):
    """Return a normalized one-dimensional, four-sigma discrete PSF.

    The radius is ceil(4 sigma). Evaluate the unblurred expected counts on a
    halo of this radius before applying the PSF; replicating a detector edge
    would substitute artificial illumination outside the field of view.
    """
    sigma = float(mtf_sigma_px)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("mtf_sigma_px must be finite and non-negative.")
    if sigma == 0:
        return np.ones(1)
    radius = int(np.ceil(4 * sigma))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma)**2)
    return kernel / kernel.sum()


def apply_detector_psf(halo_counts_yx, detector):
    """Blur a physical illumination halo and return the valid detector crop."""
    kernel = jnp.asarray(detector_psf_kernel(detector.mtf_sigma_px))
    radius = (kernel.size - 1) // 2
    expected_shape = tuple(int(n) + 2 * radius for n in detector.shape_yx)
    values = jnp.asarray(halo_counts_yx)
    if values.shape != expected_shape:
        raise ValueError(f"Expected counts on detector plus PSF halo {expected_shape}.")
    if radius == 0:
        return values
    values = values[None, :, :, None]
    dimensions = ("NHWC", "HWIO", "NHWC")
    values = jax.lax.conv_general_dilated(values, kernel[None, :, None, None],
        (1, 1), "VALID", dimension_numbers=dimensions)
    values = jax.lax.conv_general_dilated(values, kernel[:, None, None, None],
        (1, 1), "VALID", dimension_numbers=dimensions)
    return values[0, :, :, 0]


def normal_interval_probability(lower, upper):
    """Stable standard-normal probability on an ordered interval."""
    # Positive tails use symmetry to avoid subtracting two numbers near one.
    return jnp.where(lower > 0, ndtr(-lower) - ndtr(-upper),
                     ndtr(upper) - ndtr(lower))


@lru_cache(maxsize=16)
def _quadrature(order):
    if isinstance(order, bool) or int(order) != order or order < 1:
        raise ValueError("quadrature_order must be a positive integer.")
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    return nodes, weights


@dataclass(frozen=True, eq=False)
class GaussianDetectorOperator:
    """Differentiable counts, with no normalization by detected intensity.

    Mean and covariance are in optical lab (x,y) coordinates, before the
    detector affine. ``incident_electrons`` is the population of the entire
    spatial Gaussian, including electrons outside the detector. Quadrature is
    static across fitting; compare finer orders over the allowed parameter
    range before admitting very narrow beams into an inversion.
    """

    detector: object
    quadrature_order: int = 8

    def __post_init__(self):
        _quadrature(self.quadrature_order)
        detector_psf_kernel(self.detector.mtf_sigma_px)

    @property
    def halo_pixels(self):
        return (detector_psf_kernel(self.detector.mtf_sigma_px).size - 1) // 2

    def __call__(self, mean_xy_m, covariance_xy_m2, incident_electrons=1.):
        mean, covariance = jnp.asarray(mean_xy_m), jnp.asarray(covariance_xy_m2)
        if mean.shape != (2,) or covariance.shape != (2, 2):
            raise ValueError("A spatial Gaussian requires a (2,) mean and (2,2) covariance.")
        if jnp.ndim(incident_electrons) != 0:
            raise ValueError("incident_electrons must be scalar; use vmap for exposures.")
        affine = jnp.asarray(self.detector.affine_xy)
        mean = affine @ (mean - jnp.asarray(self.detector.centre_xy_m))
        covariance = affine @ covariance @ affine.T
        ny, nx = self.detector.shape_yx
        py, px = self.detector.pixel_size_yx_m
        halo = self.halo_pixels
        x_centres = (jnp.arange(-halo, nx + halo) + .5 - nx / 2) * px
        y_lower = (jnp.arange(-halo, ny + halo) - ny / 2) * py
        nodes, weights = (jnp.asarray(a) for a in _quadrature(self.quadrature_order))
        x_delta = x_centres[:, None] + .5 * px * nodes - mean[0]
        variance_x = covariance[0, 0]
        slope = covariance[1, 0] / variance_x
        conditional_sigma = jnp.sqrt(covariance[1, 1] - covariance[1, 0] * slope)
        conditional_mean = mean[1] + slope * x_delta
        lower = (y_lower[:, None, None] - conditional_mean[None, :, :]) / conditional_sigma
        upper = lower + py / conditional_sigma
        probability_y = normal_interval_probability(lower, upper)
        density_x = jnp.exp(-.5 * x_delta**2 / variance_x) / jnp.sqrt(2 * jnp.pi * variance_x)
        halo_probability = .5 * px * jnp.sum(probability_y * density_x * weights, axis=-1)
        return incident_electrons * apply_detector_psf(halo_probability, self.detector)


def make_gaussian_detector_operator(detector, quadrature_order=8):
    """Construct the reusable finite-pixel operator; JIT the returned callable."""
    return GaussianDetectorOperator(detector, quadrature_order)


def transported_spatial_moments(matrix, mean_state, covariance_state):
    """Project [x,theta_x,y,theta_y] moments through a genuine 4x4 map."""
    matrix, mean, covariance = map(jnp.asarray, (matrix, mean_state, covariance_state))
    if matrix.shape != (4, 4) or mean.shape != (4,) or covariance.shape != (4, 4):
        raise ValueError("Expected a 4x4 map, four-state mean, and 4x4 covariance.")
    position_map = matrix[jnp.asarray((0, 2)), :]
    return position_map @ mean, position_map @ covariance @ position_map.T


def fixed_variance_image_residual(prediction, observed, variance):
    """Whiten raw pixels using a fixed, externally supplied noise variance.

    This is weighted least squares, not an exact Poisson/read-noise likelihood.
    A variance estimated from an independent dose exposure or declared design
    is held fixed; normalizing either image by its own total loses the flux
    information needed for aperture measurements. Negative dark-corrected
    observations are retained. Non-positive variances yield non-finite output
    and must fail the fit's admission checks.
    """
    prediction, observed, variance = map(jnp.asarray, (prediction, observed, variance))
    if prediction.shape != observed.shape or variance.shape != observed.shape:
        raise ValueError("Prediction, observation, and variance shapes must agree.")
    return ((prediction - observed) / jnp.sqrt(variance)).reshape(-1)


def validate_phase_space_covariance(covariance, wavelength_m, *,
                                    position_scale_m=1e-8, relative_tolerance=1e-10):
    """Reject Gaussian source covariances violating the wave uncertainty bound.

    [x,theta_x,y,theta_y] has minimum eigen-emittance lambda/(4 pi).
    Congruence scaling balances the Hermitian uncertainty matrix before its
    eigenvalues are tested. Positive definiteness alone is insufficient.
    This host-side check belongs at design entry, not inside a compiled loss.
    """
    covariance = np.asarray(covariance, float)
    if covariance.shape != (4, 4) or not np.all(np.isfinite(covariance)):
        raise ValueError("Phase-space covariance must be a finite 4x4 array.")
    if not np.isfinite(wavelength_m) or wavelength_m <= 0:
        raise ValueError("wavelength_m must be finite and positive.")
    if not np.isfinite(position_scale_m) or position_scale_m <= 0:
        raise ValueError("position_scale_m must be finite and positive.")
    if not np.isfinite(relative_tolerance) or relative_tolerance < 0:
        raise ValueError("relative_tolerance must be finite and non-negative.")
    epsilon = wavelength_m / (4 * np.pi)
    scale = np.array([position_scale_m, epsilon / position_scale_m] * 2)
    balanced = covariance / np.outer(scale, scale)
    if not np.allclose(balanced, balanced.T, rtol=0., atol=relative_tolerance):
        raise ValueError("Phase-space covariance must be symmetric.")
    if np.linalg.eigvalsh(balanced).min() <= 0:
        raise ValueError("Phase-space covariance must be positive definite.")
    symplectic = np.kron(np.eye(2), np.array([[0., 1.], [-1., 0.]]))
    eigenvalues = np.linalg.eigvalsh(balanced + 1j * symplectic)
    tolerance = relative_tolerance * max(1., np.max(np.abs(eigenvalues)))
    if eigenvalues.min() < -tolerance:
        raise ValueError("Source covariance violates the Gaussian wave uncertainty bound.")
    return {"minimum_balanced_uncertainty_eigenvalue": float(eigenvalues.min()),
            "minimum_emittance_m_rad": float(epsilon), "tolerance": float(tolerance)}


GAUSSIAN_IMAGE_PARAMETER_NAMES = (
    "mean_x_pixel", "mean_y_pixel", "log_rms_x_pixel", "log_rms_y_pixel",
    "atanh_correlation_xy", "log_incident_electrons",
)


def decode_gaussian_image_parameters(parameters, detector):
    """Decode six unbounded-beam coordinates in the physical detector frame.

    Pixel coordinates have their origin at the detector centre. These describe
    the distribution before finite-detector truncation and before the PSF.
    The inverse affine maps them into the optical frame used by the renderer.
    """
    p = jnp.asarray(parameters)
    if p.shape != (6,):
        raise ValueError("Expected six Gaussian image coordinates.")
    pixel_xy = jnp.asarray(detector.pixel_size_yx_m[::-1])
    mean_physical = p[:2] * pixel_xy
    sigma = jnp.exp(p[2:4]) * pixel_xy
    cross = jnp.tanh(p[4]) * jnp.prod(sigma)
    covariance_physical = jnp.array([[sigma[0]**2, cross], [cross, sigma[1]**2]])
    inverse = jnp.linalg.inv(jnp.asarray(detector.affine_xy))
    mean = inverse @ mean_physical + jnp.asarray(detector.centre_xy_m)
    return mean, inverse @ covariance_physical @ inverse.T, jnp.exp(p[5])


def encode_gaussian_image_parameters(mean_xy_m, covariance_xy_m2, incident_electrons, detector):
    """Encode transported optical moments without extracting another angle."""
    affine = jnp.asarray(detector.affine_xy)
    mean = affine @ (jnp.asarray(mean_xy_m) - jnp.asarray(detector.centre_xy_m))
    covariance = affine @ jnp.asarray(covariance_xy_m2) @ affine.T
    sigma = jnp.sqrt(jnp.diag(covariance))
    pixel_xy = jnp.asarray(detector.pixel_size_yx_m[::-1])
    correlation = covariance[0, 1] / jnp.prod(sigma)
    return jnp.concatenate((mean / pixel_xy, jnp.log(sigma / pixel_xy),
                            jnp.array([jnp.arctanh(correlation), jnp.log(incident_electrons)])))


def gaussian_image_information_factor(parameters, detector, *, variance_yx=None,
                                      read_noise_electrons=.6, quadrature_order=8):
    """Six-column SVD factor preserving the local raw-pixel WLS information.

    If ``J`` is the whitened pixel Jacobian, the returned ``R`` satisfies
    ||R delta||² = ||J delta||² without forming J.T@J. Composing R with the
    six image-coordinate derivatives permits lens-design Fisher calculations
    without a pixels-by-optical-parameters allocation. No measured ellipse
    precision is granted externally. This local factor is not a substitute
    for fitting and validating the observed images.

    When variance is omitted, use predicted Poisson variance plus independent
    read-noise variance as an explicitly conditional design calculation.
    """
    if not np.isfinite(read_noise_electrons) or read_noise_electrons <= 0:
        raise ValueError("A finite positive read-noise floor is required.")
    operator = make_gaussian_detector_operator(detector, quadrature_order)
    predict = lambda p: operator(*decode_gaussian_image_parameters(p, detector))
    parameters = jnp.asarray(parameters)
    expected = predict(parameters)
    variance = expected + read_noise_electrons**2 if variance_yx is None else jnp.asarray(variance_yx)
    if variance.shape != detector.shape_yx:
        raise ValueError("Fixed variance must match the detector shape.")
    derivative = jax.jacfwd(predict)(parameters).reshape(-1, 6)
    whitened = derivative / jnp.sqrt(variance.reshape(-1, 1))
    _, singular, right = jnp.linalg.svd(whitened, full_matrices=False)
    return singular[:, None] * right


@dataclass(frozen=True)
class GaussianImageFit:
    parameters: np.ndarray
    parameter_covariance: np.ndarray
    fitted_counts_yx: np.ndarray
    whitened_parameter_jacobian: np.ndarray
    optimizer_result: object
    degrees_of_freedom: int
    residual_rms_sigma: float
    chi_square_z: float
    numerical_rank: int
    quadrature_rms_sigma: float
    quadrature_parameter_shift_sigma: float

    def require_accepted(self, *, maximum_chi_square_z=5.):
        """Admit a noisy fit only after optimization, rank, and image checks.

        The upper chi-square check diagnoses excess image mismatch. A lower
        value may be noiseless reference data; it is retained, not used to
        inflate or shrink the externally supplied variance.
        """
        if (not self.optimizer_result.success or self.numerical_rank != 6
                or not np.isfinite(self.chi_square_z)
                or self.chi_square_z > maximum_chi_square_z
                or not np.isfinite(self.quadrature_rms_sigma)
                or not np.isfinite(self.quadrature_parameter_shift_sigma)
                or self.quadrature_rms_sigma > .2
                or self.quadrature_parameter_shift_sigma > .2):
            raise ValueError("Gaussian image fit failed convergence, rank, pixel-residual, or quadrature admission.")
        return self


def fit_gaussian_image(observed_counts_yx, variance_yx, detector, *, initial_parameters=None,
                       quadrature_order=8, max_nfev=160):
    """Fit raw finite-detector pixels and retain the full six-parameter covariance.

    The initializer uses signed data moments only as a starting estimate.
    Truncation and detector blur enter every fitted prediction. The returned
    covariance is an unrescaled local WLS covariance from a Jacobian SVD;
    uncertainty in the independent variance estimate is not propagated here.
    Physical lens fits must profile dose, alignment, detector calibration,
    and source covariance as appropriate to the actual acquisition.
    """
    from .jax_least_squares import least_squares

    observed, variance = np.asarray(observed_counts_yx, float), np.asarray(variance_yx, float)
    if observed.shape != detector.shape_yx or variance.shape != observed.shape:
        raise ValueError("Observed counts and fixed variance must match detector shape.")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(variance)) or np.any(variance <= 0):
        raise ValueError("Observations must be finite and variances finite and positive.")
    ny, nx = detector.shape_yx
    if observed.size <= 6:
        raise ValueError("More than six pixels are required for a six-parameter fit.")
    if initial_parameters is None:
        total = observed.sum()
        if total <= 0:
            raise ValueError("Positive total signal is required for moment initialization.")
        yy, xx = np.indices(observed.shape)
        coords = np.stack((xx+.5-nx/2, yy+.5-ny/2), axis=-1)
        centre = np.einsum('yx,yxi->i', observed, coords) / total
        delta = coords - centre
        covariance = np.einsum('yx,yxi,yxj->ij', observed, delta, delta) / total
        if np.linalg.eigvalsh(covariance).min() <= 0:
            raise ValueError("Signed moment initializer is not positive definite; supply an explicit initial state.")
        sigma = np.sqrt(np.diag(covariance))
        rho = np.clip(covariance[0, 1]/np.prod(sigma), -.99, .99)
        initial_parameters = np.r_[centre, np.log(sigma), np.arctanh(rho), np.log(total)]
    initial = np.asarray(initial_parameters, float)
    if initial.shape != (6,) or not np.all(np.isfinite(initial)):
        raise ValueError("Initial image parameters must be six finite values.")
    # Broad detector-relative inference bounds, not optical/hardware bounds.
    lower = np.array([-2*nx, -2*ny, np.log(.1), np.log(.1), -4., np.log(1e-3)])
    upper = np.array([2*nx, 2*ny, np.log(100*nx), np.log(100*ny), 4., np.log(1e15)])
    operator = make_gaussian_detector_operator(detector, quadrature_order)
    predict = lambda p: operator(*decode_gaussian_image_parameters(p, detector))
    residual = lambda p: fixed_variance_image_residual(predict(p), observed, variance)
    fit = least_squares(residual, initial, bounds=(lower, upper),
        x_scale=np.array([10., 10., .2, .2, .2, .2]), max_nfev=max_nfev,
        ftol=1e-11, xtol=1e-10, gtol=1e-8)
    jacobian = np.asarray(fit.jacobian)
    left, singular, right = np.linalg.svd(jacobian, full_matrices=False)
    rank = int(np.sum(singular > singular[0]*1e-10)) if singular[0] > 0 else 0
    parameter_covariance = (right.T / singular**2) @ right if rank == 6 else np.full((6, 6), np.nan)
    dof = observed.size - 6
    chi_square = 2 * fit.cost
    fitted_counts = np.asarray(predict(fit.x))
    finer_operator = make_gaussian_detector_operator(detector, 2*quadrature_order)
    finer_counts = np.asarray(finer_operator(*decode_gaussian_image_parameters(fit.x, detector)))
    quadrature_error = ((finer_counts-fitted_counts)/np.sqrt(variance)).reshape(-1)
    quadrature_shift = (right.T @ ((left.T @ quadrature_error)/singular)
                        if rank == 6 else np.full(6, np.nan))
    shift_sigma = np.abs(quadrature_shift)/np.sqrt(np.diag(parameter_covariance))
    return GaussianImageFit(np.asarray(fit.x), parameter_covariance, fitted_counts,
        jacobian, fit, dof, float(np.sqrt(chi_square/observed.size)),
        float((chi_square-dof)/np.sqrt(2*dof)), rank,
        float(np.sqrt(np.mean(quadrature_error**2))), float(np.max(shift_sigma)))
