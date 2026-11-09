"""One microscope setting as a native TemGym column (condenser, aperture, deflectors,
projector) and its ray-traced detector image."""

from functools import lru_cache

import numpy as np


def setting_column(
    condenser_parameters,
    projector_parameters,
    *,
    voltage_v,
    reference_voltage_v,
    source_index,
    condenser_controls,
    projector_controls,
    specimen_kick,
    aperture_index,
    aperture_diameter_um,
    aperture_edge_width_m,
    gap_fractions=(0.3, 0.7),
    specimen=None,
):
    """Native source-to-detector column for one physical microscope setting.

    Parameters
    ----------
    condenser_parameters, projector_parameters : array_like
        Physical model parameters (31 and 16 values), as saved in
        ``scoring_truth.json`` or decoded from a fit.
    voltage_v, reference_voltage_v : float
        Beam voltage and the voltage at which the parameters are stated.
    source_index : int
        0 for the reference voltage's source, 1 for the second voltage's source.
    condenser_controls, projector_controls : array_like
        Relative lens controls: CL1, CL3, mini; and Objective, IL1, IL2, IL3, PL1.
    specimen_kick : array_like
        Calibrated specimen displacement and angle ``[x, y, theta_x, theta_y]`` in
        metres and radians, realised by a post-aperture double deflector.
    aperture_index : int
        Which of the two fitted aperture centres to use.
    aperture_diameter_um, aperture_edge_width_m : float
        Inserted aperture diameter and its sigmoid edge width.
    gap_fractions : tuple of float, optional
        Illustrative deflector positions as fractions of the aperture-to-mini gap.
    specimen : Component, optional
        A thin specimen at the specimen plane, between the condenser and the projector.

    Returns
    -------
    column : tuple
        Native components from the condenser to the detector plane.
    mean, covariance : numpy.ndarray
        Source ray moments (4 and 4x4).
    source_z : float
        Source position relative to the specimen, in metres.
    """
    from .apertures import aperture_centres, effective_parameters
    from .condenser import (
        condenser_with_aperture,
        condenser_with_steering,
        source_moments,
    )
    from .projector import at_voltage, projector_components

    cp = effective_parameters(
        condenser_parameters, voltage_v, reference_voltage_v, source_index
    )
    pp = at_voltage(projector_parameters, voltage_v, reference_voltage_v)
    column = condenser_with_aperture(
        cp,
        condenser_controls,
        radius_m=aperture_diameter_um * 0.5e-6,
        edge_width_m=aperture_edge_width_m,
        centre_m=aperture_centres(cp, 2)[aperture_index] * 1e-6,
    )
    column = condenser_with_steering(
        column,
        specimen_kick,
        voltage_v=voltage_v,
        gap_fractions=gap_fractions,
    )
    column = (
        *column,
        *(() if specimen is None else (specimen,)),
        *projector_components(pp, projector_controls),
    )
    mean, covariance = map(np.asarray, source_moments(cp, voltage_v))
    return column, mean, covariance, -np.exp(np.asarray(cp[:9])).sum()


@lru_cache(maxsize=4)
def _ray_renderer(shape):
    """Compile native ray transport and weighted detector binning once per shape."""
    import jax
    import jax.numpy as jnp

    from temgym_core.components import Detector, GaussianActionComponent
    from temgym_core.components._base import _uses_default_log_transmission
    from temgym_core.run import run_iter

    @jax.jit
    def render(column, source_rays, pixel_pitch):
        weights = jnp.ones_like(source_rays.x)
        for component, rays in run_iter(source_rays, column):
            if isinstance(component, GaussianActionComponent) and not _uses_default_log_transmission(component):
                # Native Rays have no amplitude; accumulate intensity separately (apertures, specimens).
                weights *= jnp.exp(2 * component.log_transmission(rays.r_xy))
        detector = Detector(
            z=rays.z, shape=shape, pixel_size=(pixel_pitch, pixel_pitch)
        )
        iy, ix = detector.metres_to_pixels((rays.x, rays.y))
        inside = (iy >= 0) & (iy < shape[0]) & (ix >= 0) & (ix < shape[1])
        index = jnp.where(inside, iy * shape[1] + ix, 0)
        return (
            jnp.bincount(
                index, weights=jnp.where(inside, weights, 0), length=shape[0] * shape[1]
            ).reshape(shape)
            / weights.size
        )

    return render


@lru_cache(maxsize=1)
def _source_samples(power):
    from scipy.special import ndtri
    from scipy.stats import qmc

    # Fixed, evenly distributed samples reduce sampling speckle between settings.
    return ndtri(qmc.Sobol(d=4, scramble=True, seed=24003).random_base2(power))


def render_rays(
    column,
    mean,
    covariance,
    source_z,
    voltage_v,
    *,
    shape_yx,
    pixel_pitch_m,
    psf_sigma_pixels,
    ray_sample_power=18,
    detector_samples_per_pixel=1,
):
    """Expected detector signal per incident electron from native rays.

    Parameters
    ----------
    column : tuple
        Native components ending at the detector plane, e.g. from
        :func:`setting_column`.
    mean, covariance : array_like
        Source ray moments; the source is sampled with fixed Sobol points.
    source_z : float
        Source position in metres.
    voltage_v : float
        Beam voltage.
    shape_yx : tuple of int
        Detector rows and columns.
    pixel_pitch_m : float
        Detector pixel pitch.
    psf_sigma_pixels : float
        Gaussian detector point-spread function, applied before pixel integration.
    ray_sample_power : int, optional
        Trace ``2**ray_sample_power`` rays.
    detector_samples_per_pixel : int, optional
        Internal sub-samples per pixel axis before the PSF and binning.

    Returns
    -------
    numpy.ndarray
        Transmitted fraction per pixel (sums to the transmitted fraction on the
        detector); no aperture diffraction and no noise.
    """
    from scipy.signal import fftconvolve

    from temgym_core.ray import Ray

    samples = (
        mean
        + _source_samples(ray_sample_power)
        @ np.linalg.cholesky(covariance).T
    )
    rays = Ray(*samples.T, z=source_z, pathlength=0.0, voltage=voltage_v)
    # Apply the detector PSF before integrating detector pixels. A fine internal
    # grid avoids rounding sparse rays to whole pixels before the blur.
    sampling = detector_samples_per_pixel
    height, width = shape_yx
    image = np.asarray(
        _ray_renderer((height * sampling, width * sampling))(
            column, rays, pixel_pitch_m / sampling
        )
    )
    # Blur only occupied rows/columns plus the complete PSF support. Keeping the
    # crop on pixel boundaries gives the same image without filtering empty space.
    occupied = [np.flatnonzero(image.any(axis=axis)) for axis in (1, 0)]
    result = np.zeros((height, width))
    if all(len(indices) for indices in occupied):
        halo = int(np.ceil(8 * psf_sigma_pixels)) + 1
        bounds = [
            (max(0, indices[0] // sampling - halo),
             min(size, indices[-1] // sampling + halo + 1))
            for indices, size in zip(occupied, (height, width))
        ]
        (y0, y1), (x0, x1) = bounds
        crop = image[y0 * sampling:y1 * sampling, x0 * sampling:x1 * sampling]
        # Separable FFT convolution is fast even on the fine detector grid.
        # Retain eight sigma: four-sigma truncation biases high-dose beam widths.
        sigma = psf_sigma_pixels * sampling
        blurred = crop
        if sigma > 0:
            radius = int(8 * sigma + 0.5)
            kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
            kernel /= kernel.sum()
            blurred = fftconvolve(crop, kernel[:, None], mode="same", axes=0)
            blurred = fftconvolve(blurred, kernel[None, :], mode="same", axes=1)
            blurred = np.maximum(blurred, 0)  # Remove FFT round-off below zero.
        result[y0:y1, x0:x1] = blurred.reshape(
            y1 - y0, sampling, x1 - x0, sampling
        ).sum(axis=(1, 3))
    return result
