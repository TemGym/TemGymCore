"""Differentiable finite-detector centroids of transported Gaussian probes.

Matches centre-sampled detector intensities and the nearest-edge Gaussian MTF
used by the renderer. It is not a pixel-integrated or saturation model.
"""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import gaussian_filter1d


@lru_cache(maxsize=16)
def _centroid_operator(shape, pixel_size, centre, affine_values, mtf_sigma):
    ny, nx = shape
    py, px = pixel_size
    yy, xx = np.indices(shape, dtype=float)
    physical = np.stack(((xx+.5-nx/2)*px, (yy+.5-ny/2)*py), axis=-1)
    affine = np.asarray(affine_values).reshape(2, 2)
    optical = np.linalg.solve(affine, physical.reshape(-1, 2).T).T + np.asarray(centre)
    # Apply the transpose of the separable detector blur to moment weights.
    # This exactly retains nearest-edge padding without convolving an image
    # every time the optimizer evaluates a centroid.
    kx = gaussian_filter1d(np.eye(nx), mtf_sigma, axis=0, mode="nearest") if mtf_sigma else np.eye(nx)
    ky = gaussian_filter1d(np.eye(ny), mtf_sigma, axis=0, mode="nearest") if mtf_sigma else np.eye(ny)
    x0, x1, x2 = (kx.T@values for values in (np.ones(nx), np.arange(nx), np.arange(nx)**2))
    y0, y1, y2 = (ky.T@values for values in (np.ones(ny), np.arange(ny), np.arange(ny)**2))
    weights = np.stack((np.outer(y0, x0), np.outer(y0, x1), np.outer(y1, x0),
                        np.outer(y0, x2), np.outer(y1, x1), np.outer(y2, x0)), axis=-1).reshape(-1, 6)

    def evaluate(mean_xy, covariance_xy):
        delta = jnp.asarray(optical)-mean_xy
        precision_delta = jnp.linalg.solve(covariance_xy, delta.T).T
        exponent = -.5*jnp.sum(delta*precision_delta, axis=-1)
        intensity = jnp.exp(exponent-jnp.max(exponent))
        moment_weights = jnp.asarray(weights)
        totals = intensity@moment_weights
        return totals[1:]/totals[0], intensity, precision_delta, totals[0]

    @jax.custom_jvp
    def centroid(mean_xy, covariance_xy):
        return evaluate(mean_xy, covariance_xy)[0]

    @centroid.defjvp
    def centroid_jvp(primals, tangents):
        result, intensity, v, normalization = evaluate(*primals)
        weights_array = jnp.asarray(weights)
        centered_weights = weights_array[:, 1:]-weights_array[:, :1]*result
        d_mean = centered_weights.T@(intensity[:, None]*v)/normalization
        d_covariance = .5*jnp.einsum("ni,n,nj,nk->ijk", centered_weights, intensity, v, v)/normalization
        tangent = d_mean@tangents[0]+jnp.einsum("ijk,jk->i", d_covariance, tangents[1])
        return result, tangent

    return jax.jit(centroid)


def finite_gaussian_raw_moments(mean_xy_m, covariance_xy_m2, detector):
    """Expected ``x, y, x², xy, y²`` in finite-detector pixel coordinates.

    Mean and positive-definite covariance are in optical lab coordinates,
    before the detector affine. Covariance is the beam's physical covariance,
    not uncertainty on a commanded displacement. Differentiation uses Gaussian
    score moments, avoiding a parameter-by-pixel derivative allocation.
    """
    mean, covariance = jnp.asarray(mean_xy_m), jnp.asarray(covariance_xy_m2)
    if mean.shape != (2,) or covariance.shape != (2, 2):
        raise ValueError("Expected a (2,) mean and (2,2) covariance.")
    operator = _centroid_operator(tuple(detector.shape_yx), tuple(detector.pixel_size_yx_m),
                                 tuple(detector.centre_xy_m),
                                 tuple(np.asarray(detector.affine_xy).ravel()),
                                 float(detector.mtf_sigma_px))
    return operator(mean, covariance)


def finite_gaussian_centroid(mean_xy_m, covariance_xy_m2, detector):
    """Finite-detector centroid, including the detector affine and MTF."""
    return finite_gaussian_raw_moments(mean_xy_m, covariance_xy_m2, detector)[:2]
