"""Measure image moments, then predict the same quantities with lens matrices.

``measure_detector_moments`` takes recorded pixels and returns five values per
image: centre x/y, variance x, covariance xy, variance y. Its covariance output
is the uncertainty of those five measurements, including their correlations.

``predict_detector_moments`` takes a 5x5 microscope matrix and the specimen's
4D position/angle mean and covariance, and predicts those five pixel quantities.
``SampledTransmission`` supplies the specimen integrals for that prediction.
The final two functions correct pixel sampling for narrow condenser images.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.nn import softmax


def measure_detector_moments(
    *, images, background_counts, read_noise_counts, regions=None
):
    """Measure five image moments with full covariance from independent pixels.

    Calibrated dark background is subtracted without clipping signal values.
    Shot/read noise propagates through the normalized moments using the delta
    method. Drift moves centroids; unknown drift must be fitted or eliminated
    by paired measurements; pixel noise alone does not describe drift.
    """
    images = np.asarray(images, dtype=float)
    if images.ndim != 3 or not np.isfinite(images).all():
        raise ValueError("Supply finite images of shape (frames, height, width).")
    yy, xx = np.indices(images.shape[1:], dtype=float)
    xx -= (images.shape[2] - 1) / 2
    yy -= (images.shape[1] - 1) / 2
    regions = (
        np.ones(images.shape, dtype=bool)
        if regions is None
        else np.broadcast_to(regions, images.shape)
    )
    signal = (images - background_counts) * regions
    flux = signal.sum(axis=(1, 2))
    if np.any(flux <= 0):
        raise ValueError("Each frame must have positive background-subtracted flux.")
    x_mean = (signal * xx).sum(axis=(1, 2)) / flux
    y_mean = (signal * yy).sum(axis=(1, 2)) / flux
    x, y = xx - x_mean[:, None, None], yy - y_mean[:, None, None]
    centred_products = np.stack((x * x, x * y, y * y), axis=-1)
    second_moments = np.einsum("nyx,nyxk->nk", signal, centred_products) / flux[:, None]
    values = np.column_stack((x_mean, y_mean, second_moments))
    measurement_derivative = np.concatenate(
        (
            x[..., None],
            y[..., None],
            centred_products - second_moments[:, None, None],
        ),
        axis=-1,
    )
    measurement_derivative /= flux[:, None, None, None]
    pixel_noise_variance = (np.maximum(images, 0) + read_noise_counts**2) * regions
    measurement_covariance = np.einsum(
        "nyx,nyxi,nyxj->nij",
        pixel_noise_variance,
        measurement_derivative,
        measurement_derivative,
    )
    return {"values": values, "covariance": measurement_covariance, "signal_counts": flux}


def predict_detector_moments(matrix, mean, covariance, *, pixel_pitch_m, psf_sigma_pixels):
    """Predict [mean x, mean y, var x, cov xy, var y] in detector pixel units.

    The 5×5 matrix acts on [x, y, theta_x, theta_y, 1]. The specimen mean and
    covariance contain the first four coordinates, using metres and radians.
    The 1/12 term accounts for assigning integrated counts to pixel centres.
    It is the resolved-image limit; sampling and finite-detector cropping must
    be checked against the actual pixel measurements.
    """
    mapping = matrix[:2, :4]
    centre = (mapping @ mean + matrix[:2, 4]) / pixel_pitch_m
    image_covariance = mapping @ covariance @ mapping.T / pixel_pitch_m**2
    image_covariance += jnp.eye(2) * (psf_sigma_pixels**2 + 1 / 12)
    return jnp.array(
        [*centre, image_covariance[0, 0], image_covariance[0, 1], image_covariance[1, 1]]
    )


def moment_regions(
    *,
    images,
    background_counts,
    read_noise_counts,
    margin_pixels=8,
    threshold_sigma=8.0,
):
    """Choose padded signal rectangles from pixels, with no microscope model.

    Smoothing is only used to locate the signal. Moments use the unsmoothed
    recorded pixels. The mask is retained so noiseless/sampling checks use
    exactly the same measurement region as the noisy experiment.
    """
    from scipy.ndimage import gaussian_filter

    images = np.asarray(images)
    masks = np.zeros(images.shape, dtype=bool)
    threshold = (
        threshold_sigma
        * np.sqrt(background_counts + read_noise_counts**2)
        / (4 * np.sqrt(np.pi))
    )
    for image, mask in zip(images, masks):
        signal = gaussian_filter(image - background_counts, sigma=2.0)
        y, x = np.nonzero(signal > threshold)
        if not len(x):
            raise ValueError("No beam is distinguishable from detector noise.")
        y0, y1 = (
            max(0, y.min() - margin_pixels),
            min(image.shape[0], y.max() + margin_pixels + 1),
        )
        x0, x1 = (
            max(0, x.min() - margin_pixels),
            min(image.shape[1], x.max() + margin_pixels + 1),
        )
        mask[y0:y1, x0:x1] = True
    return masks


@dataclass(frozen=True)
class SampledTransmission:
    """Known complex specimen sampled on a uniform integration grid.

    ``intensity`` is |t|², ``log_amplitude_gradient`` is grad(log|t|) in 1/m,
    and ``path_gradient`` is grad(optical path), in radians of ray slope.
    Arrays are flattened over the same grid. Constant cell area cancels in
    the normalized moments; refine this grid independently of Gaussian count.
    """

    coordinates_m: jax.Array
    intensity: jax.Array
    log_amplitude_gradient: jax.Array
    path_gradient: jax.Array

    def _exit_moments(
        self,
        log_illumination,
        amplitude_gradient,
        angles,
        extra_angular_covariance,
        wavelength,
    ):
        xy = jnp.asarray(self.coordinates_m)
        weights = softmax(log_illumination + jnp.log(jnp.asarray(self.intensity)))
        state = jnp.concatenate((xy, angles + self.path_gradient), axis=1)
        mean = weights @ state
        centred = state - mean
        covariance = jnp.einsum("n,ni,nj->ij", weights, centred, centred)
        amplitude_gradient = amplitude_gradient + self.log_amplitude_gradient
        diffraction_covariance = (wavelength / (2 * jnp.pi)) ** 2 * jnp.einsum(
            "n,ni,nj->ij",
            weights,
            amplitude_gradient,
            amplitude_gradient,
        )
        covariance = covariance.at[2:, 2:].add(
            diffraction_covariance + extra_angular_covariance
        )
        return mean, covariance

    def gaussian_moments(self, mean, covariance, wavelength):
        """Apply the specimen to a Gaussian phase-space illumination state.

        The incident conditional angular covariance includes the coherent
        Gaussian diffraction floor. Subtract it before adding the exit wave's
        amplitude-gradient term, so diffraction is counted exactly once.
        A physical incident state must satisfy the uncertainty bound.
        """
        spatial = covariance[:2, :2]
        inverse = jnp.linalg.inv(spatial)
        correlation = covariance[2:, :2] @ inverse
        conditional = covariance[2:, 2:] - correlation @ covariance[:2, 2:]
        floor = (wavelength / (4 * jnp.pi)) ** 2 * inverse
        delta = self.coordinates_m - mean[:2]
        amplitude_gradient = -0.5 * (delta @ inverse)
        illumination = -0.5 * jnp.einsum("ni,ij,nj->n", delta, inverse, delta)
        angles = mean[2:] + delta @ correlation.T
        return self._exit_moments(
            illumination, amplitude_gradient, angles, conditional - floor, wavelength
        )

    def circular_moments(
        self,
        *,
        radius_m,
        edge_m,
        position_m,
        angle_rad,
        curvature_per_m,
        angular_rms_rad,
        wavelength,
    ):
        """Soft circular illumination with a coherent field and extra tilt spread."""
        from jax.nn import log_sigmoid, sigmoid

        delta = self.coordinates_m - position_m
        radius = jnp.sqrt(jnp.sum(delta**2, axis=1) + 1e-40)
        edge_coordinate = (radius_m - radius) / edge_m
        log_intensity = log_sigmoid(edge_coordinate)
        amplitude_gradient = (
            -0.5
            * sigmoid(-edge_coordinate)[:, None]
            * delta
            / (edge_m * radius[:, None])
        )
        angles = angle_rad + curvature_per_m * delta
        extra_angular_covariance = jnp.eye(2) * angular_rms_rad**2
        return self._exit_moments(
            log_intensity, amplitude_gradient, angles, extra_angular_covariance, wavelength
        )


def gaussian_image_characteristic(
    sample,
    optical_path_m,
    mean,
    covariance,
    wavelength,
    matrix,
    frequency,
    *,
    pixel_pitch_m,
    psf_sigma_pixels,
):
    """Image characteristic function from a specimen-plane mutual-intensity integral.

    ``frequency`` is in radians per detector pixel. The matrix maps its angular
    part to a separation between two specimen points. This predicts pixel-bin
    corrections without making an image. Known transmission is interpolated on
    its uniform grid; refine that integration separately from the Gaussian basis.
    """
    from jax.scipy.ndimage import map_coordinates

    xy = jnp.asarray(sample.coordinates_m)
    side = int(np.sqrt(len(xy)))
    step = xy[1, 0] - xy[0, 0]
    mapping = matrix[:2, :4] / pixel_pitch_m
    alpha, beta = mapping[:, :2].T @ frequency, mapping[:, 2:].T @ frequency
    k = 2 * jnp.pi / wavelength
    separation = beta / k
    inverse = jnp.linalg.inv(covariance[:2, :2])
    curvature = covariance[2:, :2] @ inverse
    extra = (
        covariance[2:, 2:]
        - curvature @ covariance[:2, 2:]
        - (wavelength / (4 * jnp.pi)) ** 2 * inverse
    )
    delta = xy - mean[:2]
    illumination = jnp.exp(-0.5 * jnp.einsum("ni,ij,nj->n", delta, inverse, delta))
    normalization = jnp.sum(illumination * sample.intensity)
    amplitude = jnp.sqrt(sample.intensity).reshape(side, side)
    path = jnp.asarray(optical_path_m).reshape(side, side)
    shifted = []
    for sign in (-1, 1):
        points = (xy + sign * 0.5 * separation - xy[0]) / step
        coordinates = jnp.stack((points[:, 1], points[:, 0]))
        shifted.append(
            (
                map_coordinates(
                    amplitude, coordinates, order=1, mode="constant", cval=0.0
                ),
                map_coordinates(path, coordinates, order=1, mode="constant", cval=0.0),
            )
        )
    phase = xy @ alpha + k * (
        (mean[2:] + delta @ curvature.T) @ separation + shifted[1][1] - shifted[0][1]
    )
    integrand = illumination * shifted[0][0] * shifted[1][0] * jnp.exp(1j * phase)
    damping = jnp.exp(
        -0.125 * separation @ inverse @ separation
        - 0.5 * beta @ extra @ beta
        - 0.5 * psf_sigma_pixels**2 * jnp.sum(frequency**2)
    )
    return (
        damping
        * jnp.sum(integrand)
        / normalization
        * jnp.exp(1j * frequency @ matrix[:2, 4] / pixel_pitch_m)
    )


def pixel_bin_correction(moments, characteristic):
    """First Fourier harmonic of assigning integrated counts to pixel centres.

    For a Gaussian detector PSF >=0.5 pixel RMS, the second harmonic is
    suppressed below 2.7e-9 before specimen coherence is included. ``moments``
    contains the continuous mean/covariance plus the usual 1/12 diagonal term.
    The even-sized detector has centres at half-integer pixel coordinates.
    """
    frequencies = 2 * jnp.pi * jnp.eye(2)
    chi = jax.vmap(characteristic)(frequencies)
    derivative = jax.vmap(jax.jacfwd(characteristic))(frequencies)
    shift = chi.imag / jnp.pi
    diagonal = -2 * jnp.diag(derivative).real / jnp.pi + chi.real / jnp.pi**2
    diagonal -= 2 * moments[:2] * shift + shift**2
    mixed = -(derivative[0, 1] + derivative[1, 0]).real / jnp.pi
    mixed += (
        characteristic(jnp.array([2 * jnp.pi, -2 * jnp.pi]))
        - characteristic(jnp.array([2 * jnp.pi, 2 * jnp.pi]))
    ).real / (2 * jnp.pi**2)
    mixed -= moments[0] * shift[1] + moments[1] * shift[0] + shift[0] * shift[1]
    return moments + jnp.array([shift[0], shift[1], diagonal[0], mixed, diagonal[1]])
