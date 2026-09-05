"""Recover aperture-image scale and beam covariance from finite detector counts.

This deliberately conditional diagnostic uses the verified aperture-conjugate
collection state, not ordinary 40k specimen images. Each coherent mode is
propagated by the exact B=0 coordinate-image limit. The specimen and every
intermediate bore are assumed clear. Circular apertures supply a length scale,
but cannot supply an absolute rotation reference or angular covariance.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from temgym_core.gaussian_image_likelihood import (
    apply_detector_psf, detector_psf_kernel, normal_interval_probability,
    make_gaussian_detector_operator,
)
from temgym_core.jax_least_squares import least_squares
from temgym_core.neoarm_calibration import DetectorGeometry
from temgym_core.constants import energy2wavelength
from neoarm_aperture_capture import (
    ROOT, XY, RADII_M, CENTRES_M, load_cases, thermal_source_modes,
)


DEFAULT_COLLECTION = ROOT/"examples/generated/neoarm_inversion/imaging_recipe/aperture_collection.json"


def require_gpu():
    if jax.default_backend() != "gpu":
        raise RuntimeError("Physical aperture-image simulation and fitting require GPU.")


def encode_spatial(mean_px, covariance_px2):
    sigma = np.sqrt(np.diag(covariance_px2))
    rho = covariance_px2[0, 1]/np.prod(sigma)
    return np.r_[mean_px, np.log(sigma), np.arctanh(rho)]


def decode_spatial(p):
    sigma = jnp.exp(p[2:4]); rho = jnp.tanh(p[4])
    return p[:2], jnp.array([[sigma[0]**2, rho*sigma[0]*sigma[1]],
                            [rho*sigma[0]*sigma[1], sigma[1]**2]])


@dataclass(frozen=True, eq=False)
class AperturePixelOperator:
    """Circle-clipped Gaussian pixel probabilities with explicit edge geometry.

    Integrate Gaussian conditional-y probabilities over x. Split x intervals
    at every intersection with a horizontal pixel edge; a sine substitution
    removes the square-root endpoint singularity at the circle's extrema.
    No hard-mask differentiation or detected-power normalization is used.
    Coordinates are detector pixels, before the discrete halo PSF.
    """

    detector: object
    order: int = 12

    def __post_init__(self):
        if self.order < 2:
            raise ValueError("Circle integration needs quadrature order >=2.")
        if not np.allclose(self.detector.affine_xy, np.eye(2)):
            raise ValueError("This diagnostic currently uses calibrated orthogonal detector pixels.")
        if self.detector.pixel_size_yx_m[0] != self.detector.pixel_size_yx_m[1]:
            raise ValueError("This diagnostic currently requires square pixels.")

    @property
    def halo(self):
        return (len(detector_psf_kernel(self.detector.mtf_sigma_px))-1)//2

    def geometry(self, radius_px, centre_px):
        """Return x nodes, weights and clipped y intervals on the PSF halo."""
        ny, nx = self.detector.shape_yx
        xlo = jnp.arange(-self.halo, nx+self.halo)-nx/2
        ylo = jnp.arange(-self.halo, ny+self.halo)-ny/2
        xl, yl = jnp.meshgrid(xlo, ylo, indexing="xy")
        cx, cy = centre_px
        left = jnp.clip(xl, cx-radius_px, cx+radius_px)
        right = jnp.clip(xl+1, cx-radius_px, cx+radius_px)
        cuts = [left, right]
        for edge in (yl, yl+1):
            # Inactive square roots have finite derivatives under JAX.
            radical = radius_px**2-(edge-cy)**2
            root = jnp.sqrt(jnp.maximum(radical, 1e-24))
            for sign in (-1, 1):
                candidate = jnp.where(radical > 0, cx+sign*root, left)
                cuts.append(jnp.clip(candidate, left, right))
        cuts = jnp.sort(jnp.stack(cuts, axis=-1), axis=-1)
        # x=cx+r*sin(t) regularizes both circular extrema.
        normalized = jnp.clip((cuts-cx)/radius_px, -1+1e-14, 1-1e-14)
        angle = jnp.arcsin(normalized)
        nodes, weights = (jnp.asarray(a) for a in np.polynomial.legendre.leggauss(self.order))
        middle = .5*(angle[..., 1:]+angle[..., :-1])
        half = .5*(angle[..., 1:]-angle[..., :-1])
        t = middle[..., None]+half[..., None]*nodes
        x = cx+radius_px*jnp.sin(t)
        weight = half[..., None]*weights*radius_px*jnp.cos(t)
        height = radius_px*jnp.cos(t)
        lower = jnp.maximum(yl[..., None, None], cy-height)
        upper = jnp.minimum(yl[..., None, None]+1, cy+height)
        return x, weight, lower, jnp.maximum(lower, upper)

    def __call__(self, mean_px, covariance_px2, radius_px, centre_px, incident_electrons=1.):
        x, weight, lower, upper = self.geometry(radius_px, centre_px)
        dx = x-mean_px[0]
        variance = covariance_px2[0, 0]
        slope = covariance_px2[1, 0]/variance
        sigma_y = jnp.sqrt(covariance_px2[1, 1]-covariance_px2[1, 0]*slope)
        mean_y = mean_px[1]+slope*dx
        probability_y = normal_interval_probability((lower-mean_y)/sigma_y, (upper-mean_y)/sigma_y)
        density_x = jnp.exp(-.5*dx*dx/variance)/jnp.sqrt(2*jnp.pi*variance)
        halo = jnp.sum(weight*density_x*probability_y, axis=(-1, -2))
        return incident_electrons*apply_detector_psf(halo, self.detector)


def make_predictor(detector, order=12, *, independent_shapes=False):
    """16 coordinates: spatial5, logM, aperture centres6, frame logdoses4."""
    aperture = AperturePixelOperator(detector, order)
    gaussian = make_gaussian_detector_operator(detector, order)
    pixel = detector.pixel_size_yx_m[1]

    def predict(p):
        mean, covariance = decode_spatial(p[:5])
        magnification = jnp.exp(p[5])
        centres = p[6:12].reshape(3, 2)
        doses = jnp.exp(p[12:16])
        aperture_out = gaussian(mean*pixel, covariance*pixel**2, doses[0])
        if independent_shapes:
            means, covariances = jax.vmap(decode_spatial)(p[16:31].reshape(3, 5))
            aperture_in = jax.lax.map(lambda args: aperture(*args),
                (means, covariances, magnification*jnp.asarray(RADII_M)/pixel, centres, doses[1:]))
        else:
            aperture_in = jax.lax.map(lambda args: aperture(mean, covariance, *args),
                (magnification*jnp.asarray(RADII_M)/pixel, centres, doses[1:]))
        return jnp.concatenate([aperture_out[None], aperture_in])
    return predict


def _thermal_polynomial_sum(coordinate, ratio, modes):
    """Sum unrenormalized incoherent mode intensities by stable recurrence."""
    previous = jnp.ones_like(coordinate)
    value = (1-ratio)*previous**2
    if modes == 1:
        return value
    current = jnp.sqrt(2.)*coordinate
    value += (1-ratio)*ratio*current**2
    for n in range(1, modes-1):
        following = jnp.sqrt(2/(n+1))*coordinate*current-jnp.sqrt(n/(n+1))*previous
        value += (1-ratio)*ratio**(n+1)*following**2
        previous, current = current, following
    return value


def mode_reference(p, source_covariance, source_to_detector, detector, *, order=16, y_order=8, modes=10):
    """Independent two-dimensional pixel integration of propagated wave modes.

    At B=0 every mode obeys psi_d(r)=exp(i quadratic phase)*psi_ap(A^-1 r)/sqrt(|det A|).
    The common phase cancels only AFTER forming each mode intensity; no
    Gaussian covariance is substituted for the truncated mode sum. Source
    propagation gives the principal axes and pure-mode widths at detector.
    The finite thermal probabilities are never renormalized.
    """
    require_gpu()
    wavelength = float(energy2wavelength(200e3))
    pure, weights, omitted = thermal_source_modes(source_covariance, wavelength, modes_per_axis=modes)
    propagated = source_to_detector@pure@source_to_detector.T
    spatial = propagated[np.ix_(XY, XY)]/detector.pixel_size_yx_m[1]**2
    axes = source_to_detector[np.ix_(XY, XY)]
    axes /= np.sqrt(np.sum(axes*axes)/2)
    variances = np.diag(axes.T@spatial@axes)
    epsilon = wavelength/(4*np.pi)
    emittance = np.sqrt(np.diag(source_covariance)[::2]*np.diag(source_covariance)[1::2])
    ratio = jnp.asarray((emittance-epsilon)/(emittance+epsilon))
    operator = AperturePixelOperator(detector, order)
    nodes, weights_y = (jnp.asarray(a) for a in np.polynomial.legendre.leggauss(y_order))
    mean, _ = decode_spatial(jnp.asarray(p[:5]))

    def intensity(x, y):
        dx, dy = x-mean[0], y-mean[1]
        first = axes[0, 0]*dx+axes[1, 0]*dy
        second = axes[0, 1]*dx+axes[1, 1]*dy
        gaussian = jnp.exp(-.5*(first**2/variances[0]+second**2/variances[1]))/(2*jnp.pi*np.sqrt(np.prod(variances)))
        return gaussian*_thermal_polynomial_sum(first/np.sqrt(2*variances[0]), ratio[0], modes)*_thermal_polynomial_sum(second/np.sqrt(2*variances[1]), ratio[1], modes)

    @jax.jit
    def aperture_counts(radius, centre, dose):
        x, weight, lower, upper = operator.geometry(radius, centre)
        middle, half = .5*(lower+upper), .5*(upper-lower)
        def one_y(i, integral):
            return integral+weights_y[i]*intensity(x, middle+half*nodes[i])
        integral = jax.lax.fori_loop(0, y_order, one_y, jnp.zeros_like(x))*half
        return dose*apply_detector_psf(jnp.sum(integral*weight, axis=(-1, -2)), detector)

    @jax.jit
    def unmasked_counts(dose):
        ny, nx = detector.shape_yx
        centres_x = jnp.arange(-operator.halo, nx+operator.halo)+.5-nx/2
        centres_y = jnp.arange(-operator.halo, ny+operator.halo)+.5-ny/2
        xx, yy = jnp.meshgrid(centres_x, centres_y, indexing="xy")
        def one(i, integral):
            ix, iy = i//y_order, i % y_order
            return integral+.25*weights_y[ix]*weights_y[iy]*intensity(xx+.5*nodes[ix], yy+.5*nodes[iy])
        halo = jax.lax.fori_loop(0, y_order*y_order, one, jnp.zeros_like(xx))
        return dose*apply_detector_psf(halo, detector)

    radii = np.exp(p[5])*RADII_M/detector.pixel_size_yx_m[1]
    centres = np.asarray(p[6:12]).reshape(3, 2); doses = np.exp(p[12:16])
    images = [np.asarray(unmasked_counts(doses[0]))]
    images.extend(np.asarray(aperture_counts(radius, centre, dose)) for radius, centre, dose in zip(radii, centres, doses[1:]))
    return np.asarray(images), dict(coherent_modes=modes**2, omitted_source_probability=omitted,
        x_order=order, y_order=y_order, propagation="Exact aperture-conjugate wave coordinate-image limit; independent 2D Hermite-mode pixel integration.")


def verify_report(path):
    report = json.loads(Path(path).read_text())
    for name, expected in report["source_hashes"].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Stale input dependency: {name}")
    upstream = Path(report["information_report_file"])
    if hashlib.sha256(upstream.read_bytes()).hexdigest() != report["information_report_sha256"]:
        raise ValueError("The information report no longer matches the collection-state provenance.")
    return report


def local_covariance(jacobian, relative_threshold=1e-10):
    jacobian = np.asarray(jacobian)
    _, singular, right = np.linalg.svd(jacobian, full_matrices=False)
    rank = int(np.sum(singular > singular[0]*relative_threshold))
    covariance = (right.T/singular**2)@right if rank == jacobian.shape[1] else np.full((jacobian.shape[1],)*2, np.nan)
    return covariance, singular, rank


def data_only_initializer(observed, detector, *, magnification_seed=100., predictor=None):
    """Initialize from aperture-out signed moments and all four raw count sums.

    No optical truth, source covariance or known incident dose is an argument.
    The common magnification seed is a broad design value, not a measurement;
    all unknown aperture centres start at the detector origin. Coverage is
    evaluated from these provisional image parameters without normalization.
    """
    counts = np.asarray(observed, float)
    if counts.shape != (4, *detector.shape_yx) or not np.isfinite(counts).all():
        raise ValueError("Four finite aperture-out/in images are required.")
    if not np.isfinite(magnification_seed) or magnification_seed <= 0:
        raise ValueError("The nominal magnification seed must be positive.")
    totals = counts.sum(axis=(-1, -2))
    if np.any(totals <= 0):
        raise ValueError("Each dark-corrected exposure needs positive total signal.")
    ny, nx = detector.shape_yx
    yy, xx = np.indices((ny, nx))
    positions = np.stack([xx+.5-nx/2, yy+.5-ny/2], axis=-1)
    mean = np.einsum("yx,yxi->i", counts[0], positions)/totals[0]
    delta = positions-mean
    covariance = np.einsum("yx,yxi,yxj->ij", counts[0], delta, delta)/totals[0]
    kernel = detector_psf_kernel(detector.mtf_sigma_px)
    offsets = np.arange(len(kernel))-(len(kernel)-1)/2
    # Signed pixel-centre moments of a broad smooth beam include quantization
    # and the known discrete detector PSF. These corrections are initialization
    # only: the actual forward fit still integrates the full detector model.
    covariance -= (np.sum(kernel*offsets**2)+1/12)*np.eye(2)
    if np.linalg.eigvalsh(covariance).min() <= 0:
        raise ValueError("Aperture-out moment covariance is not positive; acquire more signal or supply a different initializer.")
    initial = np.r_[encode_spatial(mean, covariance), np.log(magnification_seed), np.zeros(6), np.zeros(4)]
    if predictor is None:
        require_gpu()
        predictor = jax.jit(make_predictor(detector))
    coverage = np.asarray(predictor(initial)).sum(axis=(-1, -2))
    if coverage.shape != (4,) or np.any(coverage <= 0) or not np.isfinite(coverage).all():
        raise ValueError("Initial finite-camera coverage must be finite and positive.")
    initial[12:16] = np.log(totals/coverage)
    return initial


def accepted_image_fit(report):
    """Two-sided noisy goodness-of-fit and independent numerical gates."""
    return bool(report["optimizer_success"]
        and report["jacobian_rank"] == report["parameter_count"]
        and abs(report["chi_square_z"]) < 5
        and report["reference_induced_maximum_parameter_shift_sigma"] < .2
        and report["fit_quadrature_refinement_maximum_parameter_shift_sigma"] < .2
        and all(refined["optimizer_success"] and refined["maximum_parameter_change_sigma"] < .1
                for refined in report["two_successive_complete_refinement_fits"]))


def run_fit(truth, source_covariance, source_to_detector, detector, *, seed=4951,
            fit_order=12, reference_order=24, modes=10, max_nfev=100):
    require_gpu()
    predictor = jax.jit(make_predictor(detector, fit_order))
    predicted_truth = np.asarray(predictor(truth))
    coarse_reference, coarse_metadata = mode_reference(truth, source_covariance, source_to_detector,
        detector, order=reference_order, modes=modes)
    reference, reference_metadata = mode_reference(truth, source_covariance, source_to_detector,
        detector, order=2*reference_order, y_order=12, modes=modes+2)
    variance = reference+.6**2
    reference_error = (reference-predicted_truth)/np.sqrt(variance)
    rng = np.random.default_rng(seed)
    observed = rng.poisson(reference)+rng.normal(0, .6, reference.shape)
    initial = data_only_initializer(observed, detector, magnification_seed=100., predictor=predictor)
    # Broad image-coordinate bounds; these are not hardware or physical priors.
    lower = np.r_[[-64, -64], [np.log(1), np.log(1), -3], np.log(60), np.full(6, -30), np.full(4, np.log(1e3))]
    upper = np.r_[[64, 64], [np.log(150), np.log(150), 3], np.log(240), np.full(6, 30), np.full(4, np.log(1e10))]
    scales = np.r_[[1., 1., .1, .1, .1, .03], np.ones(6), np.full(4, .1)]
    residual = lambda p: ((predictor(p)-observed)/jnp.sqrt(variance)).reshape(-1)
    result = least_squares(residual, initial, bounds=(lower, upper), x_scale=scales,
        max_nfev=max_nfev, ftol=1e-10, xtol=1e-10, gtol=1e-7)
    fitted = np.asarray(result.x)
    covariance, singular, rank = local_covariance(result.jacobian)
    jac = np.asarray(result.jacobian)
    reference_shift = np.linalg.lstsq(jac, reference_error.reshape(-1), rcond=1e-10)[0]
    finer = np.asarray(jax.jit(make_predictor(detector, 2*fit_order))(fitted))
    fitted_counts = np.asarray(predictor(fitted))
    refinement_error = ((finer-fitted_counts)/np.sqrt(variance)).reshape(-1)
    refinement_shift = np.linalg.lstsq(jac, refinement_error, rcond=1e-10)[0]
    refinement_fits = []
    previous = fitted
    for order in (2*fit_order, 4*fit_order):
        refined_predictor = jax.jit(make_predictor(detector, order))
        refined_residual = lambda p: ((refined_predictor(p)-observed)/jnp.sqrt(variance)).reshape(-1)
        refinement_initial = previous+.1*np.sqrt(np.diag(covariance))*np.where(np.arange(len(previous)) % 2, -1, 1)
        refined_fit = least_squares(refined_residual, refinement_initial, bounds=(lower, upper),
            x_scale=scales, max_nfev=max_nfev, ftol=1e-10, xtol=1e-10, gtol=1e-7)
        refined_parameters = np.asarray(refined_fit.x)
        refinement_fits.append(dict(order=order, optimizer_success=bool(refined_fit.success),
            nfev=int(refined_fit.nfev), maximum_parameter_change_sigma=float(np.max(
                abs(refined_parameters-previous)/np.sqrt(np.diag(covariance)))),
            initialization="Alternating0.1standard-error perturbation of previous fit, avoiding a zero-step roundoff termination."))
        previous = refined_parameters
    errors_sigma = (fitted-truth)/np.sqrt(np.diag(covariance))
    dof = observed.size-len(truth)
    chi_square = float(np.sum((fitted_counts-observed)**2/variance))
    # The p5 scale is entirely absent from the aperture-out image if image-space
    # covariance and dose are unknown. Show this structural information loss.
    out_rank = local_covariance(jac[:observed[0].size])[2]
    _, truth_cov = decode_spatial(truth); _, fit_cov = decode_spatial(fitted)
    physical_cov_truth = np.asarray(truth_cov)*(detector.pixel_size_yx_m[1]/np.exp(truth[5]))**2
    physical_cov_fit = np.asarray(fit_cov)*(detector.pixel_size_yx_m[1]/np.exp(fitted[5]))**2
    def physical_observables(p):
        log_scale = jnp.log(detector.pixel_size_yx_m[1])-p[5]
        return jnp.r_[p[:2]*jnp.exp(log_scale), p[2:4]+log_scale, p[4], p[5]]
    observable_jacobian = np.asarray(jax.jacfwd(physical_observables)(fitted))
    profiled_observable_covariance = observable_jacobian@covariance@observable_jacobian.T
    report = dict(optimizer_success=bool(result.success), optimizer_message=str(result.message),
        nfev=int(result.nfev), truth_parameters=truth.tolist(), initial_parameters=initial.tolist(),
        fitted_parameters=fitted.tolist(), parameter_standard_errors=np.sqrt(np.diag(covariance)).tolist(),
        parameter_covariance=covariance.tolist(), parameter_error_standard_deviations=errors_sigma.tolist(),
        jacobian_rank=rank, parameter_count=len(truth), singular_values=singular.tolist(),
        aperture_out_only_rank=out_rank, residual_rms_sigma=float(np.sqrt(chi_square/observed.size)),
        chi_square_z=float((chi_square-dof)/np.sqrt(2*dof)),
        truth_closure_weighted_rms_sigma=float(np.sqrt(np.mean(reference_error**2))),
        reference_induced_maximum_parameter_shift_sigma=float(np.max(abs(reference_shift)/np.sqrt(np.diag(covariance)))),
        fit_quadrature_refinement_rms_sigma=float(np.sqrt(np.mean(refinement_error**2))),
        fit_quadrature_refinement_maximum_parameter_shift_sigma=float(np.max(abs(refinement_shift)/np.sqrt(np.diag(covariance)))),
        two_successive_complete_refinement_fits=refinement_fits,
        independent_reference_refinement_rms_sigma=float(np.sqrt(np.mean((reference-coarse_reference)**2/variance))),
        magnification_truth=float(np.exp(truth[5])), magnification_fitted=float(np.exp(fitted[5])),
        magnification_relative_error=float(np.expm1(fitted[5]-truth[5])),
        magnification_relative_standard_error=float(np.sqrt(covariance[5, 5])),
        aperture_plane_covariance_detector_aligned_truth_m2=physical_cov_truth.tolist(),
        aperture_plane_covariance_detector_aligned_fitted_m2=physical_cov_fit.tolist(),
        aperture_plane_covariance_relative_error=float(np.linalg.norm(physical_cov_fit-physical_cov_truth)/np.linalg.norm(physical_cov_truth)),
        profiled_image_observable_names=["mean_x_px", "mean_y_px", "log_sigma_x_px", "log_sigma_y_px", "atanh_correlation", "log_magnification"],
        profiled_image_observables=fitted[:6].tolist(),
        profiled_image_observable_covariance=covariance[:6, :6].tolist(),
        profiled_aperture_observable_names=["mean_x_m", "mean_y_m", "log_sigma_x_m", "log_sigma_y_m", "atanh_correlation", "log_magnification"],
        profiled_aperture_observables=np.asarray(physical_observables(fitted)).tolist(),
        profiled_aperture_observable_covariance=profiled_observable_covariance.tolist(),
        nuisance_marginalization="Full16-coordinate covariance; retain6x6 block after fitting every dose and aperture centre. Do not condition on fitted nuisance truth.",
        dose_relative_errors=np.expm1(fitted[12:16]-truth[12:16]).tolist(),
        reference=reference_metadata, coarser_reference=coarse_metadata,
        radius_calibration="Radii fixed for this fit; shared fractional radius uncertainty is an exact opposite magnification scale gauge.",
        common_radius_0p5_percent_prior_scale_uncertainty_floor=.005,
        absolute_rotation_identifiable=False, angular_covariance_identifiable=False,
        independent_frame_doses_fitted=True, total_transmission_from_count_ratios_alone_identifiable=False,
        source_covariance_used_as_fit_prior=False,
        intensity_shape_constant_across_aperture_insertions_assumed=True,
        fixed_variance_source="Declared synthetic reference expectation plus read variance, held fixed for WLS; not an exact Poisson/read-noise likelihood.",
        initialization="Data-only aperture-out signed moments, aperture centres0, broad common Mseed100, and each raw image sum divided by predicted coverage; no truth/source/dose inputs.")
    report["accepted"] = accepted_image_fit(report)
    arrays = dict(truth=truth, fitted=fitted, covariance=covariance, reference_images=reference,
        noisy_images=observed, predicted_images=fitted_counts, variance_images=variance)
    return report, arrays


def independent_shape_ablation(data, detector, *, order=12):
    """Release a separate incident spatial marginal in every aperture exposure.

    The simulated truth is stable, but the fit is not allowed to assume that.
    This tests local information loss, not a demonstrated arbitrary-drift
    hardware correction. Dose and aperture centre remain free.
    """
    require_gpu()
    common = data["fitted"]
    initial = np.r_[common, np.tile(common[:5], 3)]
    truth = np.r_[data["truth"], np.tile(data["truth"][:5], 3)]
    predictor = jax.jit(make_predictor(detector, order, independent_shapes=True))
    residual = lambda p: ((predictor(p)-data["noisy_images"])/jnp.sqrt(data["variance_images"])).reshape(-1)
    spatial_lower = np.array([-64, -64, np.log(1), np.log(1), -3])
    spatial_upper = np.array([64, 64, np.log(150), np.log(150), 3])
    lower = np.r_[spatial_lower, np.log(60), np.full(6, -30), np.full(4, np.log(1e3)), np.tile(spatial_lower, 3)]
    upper = np.r_[spatial_upper, np.log(240), np.full(6, 30), np.full(4, np.log(1e10)), np.tile(spatial_upper, 3)]
    scales = np.r_[[1., 1., .1, .1, .1, .03], np.ones(6), np.full(4, .1), np.tile([1., 1., .1, .1, .1], 3)]
    result = least_squares(residual, initial, bounds=(lower, upper), x_scale=scales,
        max_nfev=120, ftol=1e-10, xtol=1e-10, gtol=1e-7)
    covariance, singular, rank = local_covariance(result.jacobian)
    fitted = np.asarray(result.x)
    report = dict(optimizer_success=bool(result.success), optimizer_message=str(result.message),
        jacobian_rank=rank, parameter_count=31, nfev=int(result.nfev),
        magnification_relative_error=float(np.expm1(fitted[5]-truth[5])),
        magnification_relative_standard_error=float(np.sqrt(covariance[5, 5])),
        residual_rms_sigma=float(np.sqrt(2*result.cost/data["noisy_images"].size)),
        fitted_parameters=fitted.tolist(), parameter_covariance=covariance.tolist(),
        singular_values=singular.tolist(),
        intensity_shape_common_across_aperture_insertions_assumed=False,
        independent_frame_doses_fitted=True,
        simulated_truth="Stable incident beam; arbitrary per-frame mean/covariance nuisance released in the fit.",
        interpretation="Aperture edge scale survives without a shared beam shape. A lens-response measurement still needs recorded and reproducible optical states.")
    return report, dict(independent_shape_fitted=fitted, independent_shape_covariance=covariance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", type=int, nargs="+", default=[0, 2, 14])
    parser.add_argument("--dose", type=float, default=2e6)
    parser.add_argument("--fit-order", type=int, default=12)
    parser.add_argument("--reference-order", type=int, default=24)
    parser.add_argument("--modes", type=int, default=10)
    parser.add_argument("--independent-shape-ablation", action="store_true")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True); require_gpu()
    started = perf_counter()
    collection_report = verify_report(args.collection)
    collection = collection_report["aperture_collection"]
    if not collection["collection_ray_map_verified"] or collection["maximum_aperture_to_detector_B_norm_m"] > 1e-8:
        raise ValueError("A verified aperture-conjugate collection state is required.")
    cases, source_covariance, column, parameters = load_cases(Path(collection_report["information_report_file"]))
    from neoarm_shared_objective_audit import provenance_snapshot, verify_provenance_snapshot
    sources, inputs = provenance_snapshot()
    report_inputs = {str(args.collection.relative_to(ROOT) if args.collection.is_absolute() else args.collection):
                    hashlib.sha256(args.collection.read_bytes()).hexdigest()}
    information_path = Path(collection_report["information_report_file"])
    report_inputs[str(information_path.relative_to(ROOT) if information_path.is_absolute() else information_path)] = hashlib.sha256(information_path.read_bytes()).hexdigest()
    detector = DetectorGeometry(mtf_sigma_px=.65)
    records, arrays = [], {}
    for family in args.families:
        control = np.asarray(collection["controls"][family])
        full = np.asarray(jax.jit(lambda u: column.matrix(parameters, u))(control))
        covariance = full@source_covariance@full.T
        matrix = np.asarray(collection["aperture_to_detector_matrices"][family])
        a = matrix[np.ix_(XY, XY)]
        b = matrix[np.ix_(XY, np.array([1, 3]))]
        magnification = np.sqrt(np.sum(a*a)/2)
        centres = (CENTRES_M@a.T)/55e-6
        truth = np.r_[encode_spatial(np.zeros(2), covariance[np.ix_(XY, XY)]/(55e-6)**2),
            np.log(magnification), centres.reshape(-1), np.log(args.dose*np.array([1., .98, 1.03, 1.01]))]
        record, data = run_fit(truth, source_covariance, full, detector,
            seed=4951+family, fit_order=args.fit_order, reference_order=args.reference_order, modes=args.modes)
        record.update(family_index=family, label=collection["labels"][family], controls=control.tolist())
        source_to_aperture = cases[2*family].source_to_aperture
        aperture_covariance = source_to_aperture@source_covariance@source_to_aperture.T
        exact_spatial = a@aperture_covariance[np.ix_(XY, XY)]@a.T
        unmasked_operator = jax.jit(make_gaussian_detector_operator(detector, 24))
        actual_unmasked = np.asarray(unmasked_operator(jnp.zeros(2), covariance[np.ix_(XY, XY)], args.dose))
        zero_b_unmasked = np.asarray(unmasked_operator(jnp.zeros(2), exact_spatial, args.dose))
        difference = (actual_unmasked-zero_b_unmasked)/np.sqrt(actual_unmasked+.6**2)
        record.update(actual_aperture_to_detector_B_norm_m=float(np.linalg.norm(b)),
            actual_aperture_equivalent_defocus_B_over_A_m=np.linalg.solve(a, b).tolist(),
            actual_B_unmasked_image_difference_noise_norm=float(np.linalg.norm(difference)),
            actual_B_unmasked_image_difference_rms_sigma=float(np.sqrt(np.mean(difference**2))),
            actual_B_masked_edge_wave_error_certified=False,
            finite_B_warning="Unmasked residual-B effect checked against noise; sharp-aperture wave propagation at nonzero B is not certified by this conjugate-limit fit.")
        if args.independent_shape_ablation and family == args.families[0]:
            record["independent_shape_ablation"], extra_arrays = independent_shape_ablation(data, detector, order=args.fit_order)
            data.update(extra_arrays)
        records.append(record)
        arrays.update({f"family_{family}_{key}": value for key, value in data.items()})
        print(json.dumps({key: record[key] for key in ("family_index", "accepted", "jacobian_rank", "magnification_relative_error", "magnification_relative_standard_error", "truth_closure_weighted_rms_sigma", "reference_induced_maximum_parameter_shift_sigma", "nfev")}), flush=True)
    verify_provenance_snapshot(sources)
    for name, expected in report_inputs.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Input collection report changed during the fit.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    array_path = args.output.with_suffix(".npz")
    np.savez_compressed(array_path, **arrays)
    report = dict(schema="neoarm-aperture-image-fit-v1", backend=jax.default_backend(),
        source_hashes=sources, computational_input_hashes=inputs, report_input_hashes=report_inputs,
        source_snapshot_verified_unchanged=True,
        array_file=str(array_path), array_sha256=hashlib.sha256(array_path.read_bytes()).hexdigest(),
        records=records, elapsed_seconds=perf_counter()-started,
        acquisition=dict(families=args.families, radii_m=RADII_M.tolist(),
            detector_shape_yx=list(detector.shape_yx), detector_pixel_size_yx_m=list(detector.pixel_size_yx_m),
            detector_psf_sigma_px=detector.mtf_sigma_px, nominal_incident_electrons_per_frame=args.dose,
            fit_order=args.fit_order, reference_order=args.reference_order, reference_modes_per_axis=args.modes,
            independent_shape_ablation=args.independent_shape_ablation),
        assumptions=["Verified exact aperture-conjugate collection map; departure from conjugacy not fitted.",
            "Vacuum over at least4.33um sample radius plus centring/pose margins; original40nm hole is not applied.",
            "No other aperture or lens-bore clipping; scalar paraxial field with fixed truncated objective.",
            "Known detector pixel geometry and PSF; independent Poisson plus0.6e Gaussian read noise.",
            "Known circular radii, unknown individual aperture centres and independent frame doses.",
            "Spatial Gaussian marginal is fitted freely; source angular covariance and stigmator location are not granted."],
        hardware_admitted=False, coupled_condenser_recovery_demonstrated=False,
        absolute_aperture_rotation_identifiable=False, specimen_vacuum_assumed=True,
        finite_B_masked_wave_error_certified=False,
        certified_nonzero_B_over_A_tolerance_m=None,
        next_collection_measurement="Locate aperture-edge best focus from images with the objective fixed. Repeat aperture-out frames between insertions to test drift. Fit finite-B/blur or certify a noise-relative misfocus tolerance before using these observables in hardware inversion.",
        aperture_radius_requirement="Nominal10/20/30um values are synthetic inputs, not demonstrated0.5% hardware certificates.",
        all_selected_exact_conjugate_image_fits_accepted=all(record["accepted"] for record in records))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
