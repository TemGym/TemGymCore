"""Measure image moments and predict them with TemGym ray matrices.

The specimen integrals include its phase gradient and amplitude-gradient
diffraction. They predict image measurements without rendering detector pixels.
The final section provides the faster intensity approximation for initialization.
Coordinates are [x, y, theta_x, theta_y], with optical path measured in metres.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.nn import softmax
from jax.scipy.special import ndtr


@dataclass(frozen=True)
class SampledTransmission:
    """Known complex specimen sampled on a uniform integration grid.

    ``intensity`` is |t|², ``log_amplitude_gradient`` is grad(log|t|) in 1/m,
    and ``path_gradient`` is grad(optical path), in radians of ray slope.
    Arrays are flattened over the same grid. Constant cell area cancels in
    the normalized moments; refine this grid independently of Gaussian count.
    """

    coordinates_m: object
    intensity: object
    log_amplitude_gradient: object
    path_gradient: object

    def _moments(self, log_illumination, amplitude_gradient, angles, extra, wavelength):
        xy = jnp.asarray(self.coordinates_m)
        weights = softmax(log_illumination + jnp.log(jnp.asarray(self.intensity)))
        state = jnp.concatenate((xy, angles + self.path_gradient), axis=1)
        mean = weights @ state
        centred = state - mean
        covariance = jnp.einsum("n,ni,nj->ij", weights, centred, centred)
        gradient = amplitude_gradient + self.log_amplitude_gradient
        diffraction = (wavelength / (2 * jnp.pi)) ** 2 * jnp.einsum(
            "n,ni,nj->ij",
            weights,
            gradient,
            gradient,
        )
        covariance = covariance.at[2:, 2:].add(diffraction + extra)
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
        gradient = -0.5 * (delta @ inverse)
        illumination = -0.5 * jnp.einsum("ni,ij,nj->n", delta, inverse, delta)
        angles = mean[2:] + delta @ correlation.T
        return self._moments(
            illumination, gradient, angles, conditional - floor, wavelength
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
        gradient = (
            -0.5
            * sigmoid(-edge_coordinate)[:, None]
            * delta
            / (edge_m * radius[:, None])
        )
        angles = angle_rad + curvature_per_m * delta
        extra = jnp.eye(2) * angular_rms_rad**2
        return self._moments(log_intensity, gradient, angles, extra, wavelength)


def detector_moments(matrix, mean, covariance, *, pixel_pitch_m, psf_sigma_pixels):
    """Predict [mean x, mean y, var x, cov xy, var y] in detector pixel units.

    The 1/12 term accounts for assigning integrated counts to pixel centres.
    It is the resolved-image limit; sampling and finite-detector cropping must
    be checked against the actual pixel measurements.
    """
    mapping = matrix[:2, :4]
    centre = (mapping @ mean + matrix[:2, 4]) / pixel_pitch_m
    shape = mapping @ covariance @ mapping.T / pixel_pitch_m**2
    shape += jnp.eye(2) * (psf_sigma_pixels**2 + 1 / 12)
    return jnp.array([*centre, shape[0, 0], shape[0, 1], shape[1, 1]])


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


def measure_detector_moments(
    *, images, background_counts, read_noise_counts, regions=None
):
    """Measure five image moments with full covariance from independent pixels.

    Calibrated dark background is subtracted without clipping signal values.
    Shot/read noise propagates through the normalized moments using the delta
    method. Drift moves centroids; unknown drift must be fitted or eliminated
    by paired measurements, not absorbed into this covariance silently.
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
    shape_fields = np.stack((x * x, x * y, y * y), axis=-1)
    shapes = np.einsum("nyx,nyxk->nk", signal, shape_fields) / flux[:, None]
    values = np.column_stack((x_mean, y_mean, shapes))
    influence = np.concatenate(
        (x[..., None], y[..., None], shape_fields - shapes[:, None, None]), axis=-1
    )
    influence /= flux[:, None, None, None]
    variance = (np.maximum(images, 0) + read_noise_counts**2) * regions
    covariance = np.einsum("nyx,nyxi,nyxj->nij", variance, influence, influence)
    return {"values": values, "covariance": covariance, "signal_counts": flux}


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
    import jax

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


def gaussian_phase_space_moments(beam):
    """Exact position/angle moments of a single coherent Gaussian beam."""
    q = jnp.asarray(beam.Q_inv).reshape(2, 2)
    k = jnp.asarray(beam.k).reshape(())
    spatial = jnp.linalg.inv(2 * k * q.imag)
    correlation = spatial @ q.real.T
    angular = q.real @ spatial @ q.real.T + jnp.linalg.inv(spatial) / (4 * k**2)
    mean = jnp.concatenate((beam.r_xy.reshape(2), beam.d_xy.reshape(2)))
    return mean, jnp.block([[spatial, correlation], [correlation.T, angular]])


def propagate_gaussian_mixture(source, components, extra_covariance):
    """Transport a Gaussian field plus Gaussian-distributed ray-centre offsets.

    The coherent component traverses the actual lens components. First-order
    transport maps the incoherent centre distribution through the same column.
    Their covariances add; no ray/intensity replacement of field interference
    is made when the resulting state is later decomposed at the specimen.
    """
    from .gaussian import FreeSpacePropagator
    from .run import column_matrix, run_to_end

    output = run_to_end(
        source,
        components,
        propagator=FreeSpacePropagator(include_axial_path=False),
    )
    mean, covariance = gaussian_phase_space_moments(output)
    matrix = column_matrix(components, source_z=source.z)[:4, :4]
    return mean, covariance + matrix @ extra_covariance @ matrix.T


# Fast intensity approximation used for initialization.


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


def gaussian_intensity_image(
    centres,
    variance,
    weights,
    dose=1.0,
    background=0.0,
    *,
    shape_yx,
    psf_sigma_pixels=0.0,
):
    """Fast intensity approximation: integrate independent round Gaussian spots.

    Centres, variance and PSF use pixel units. Weights are integrated spot
    intensities. Off-detector flux is lost; no field interference is included.
    """
    ny, nx = shape_yx
    xe, ye = jnp.arange(nx + 1) - nx / 2, jnp.arange(ny + 1) - ny / 2
    width = jnp.sqrt(variance + psf_sigma_pixels**2)
    px = normal_interval_probability(
        (xe[:-1, None] - centres[:, 0]) / width,
        (xe[1:, None] - centres[:, 0]) / width,
    )
    py = normal_interval_probability(
        (ye[:-1, None] - centres[:, 1]) / width,
        (ye[1:, None] - centres[:, 1]) / width,
    )
    return dose * ((py * weights) @ px.T) + background


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


def normal_interval_probability(lower, upper):
    """Stable standard-normal probability on an ordered interval."""
    # Positive tails use symmetry to avoid subtracting two numbers near one.
    return jnp.where(lower > 0, ndtr(-lower) - ndtr(-upper), ndtr(upper) - ndtr(lower))
