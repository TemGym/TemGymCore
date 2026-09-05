"""Conditional probe-image likelihood for the coupled recovery benchmark.

Uses every archived perforation/basis exposure, without clipping negative
dark-corrected pixel values. Specimen Fourier/layer and Bragg estimators are
kept outside this benchmark until their own truth-closure tests pass.
"""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from neoarm_workflow import load_saved_acquisition, expected_detector_counts
from temgym_core.detector_moments import finite_gaussian_raw_moments


def signed_raw_moments(counts, expected, read_noise_e):
    """Unclipped moments and delta-method Poisson/read-noise covariance.

    ``expected`` supplies the fixed synthetic noise variance, not the measured
    moment. Experimental use requires an independently checked variance model.
    The approximation is conditional on the calibrated input and valid at high
    total count; a positive total at >20 sigma is required.
    """
    counts, expected = np.asarray(counts, float), np.asarray(expected, float)
    if counts.shape != expected.shape or counts.ndim != 2:
        raise ValueError("Expected equally sized two-dimensional images.")
    if not np.all(np.isfinite(counts)) or not np.all(np.isfinite(expected)) or np.any(expected < 0):
        raise ValueError("Images must be finite; expected counts nonnegative.")
    if not np.isfinite(read_noise_e) or read_noise_e < 0:
        raise ValueError("Read noise must be finite and nonnegative.")
    variance = expected + read_noise_e**2
    total = counts.sum()
    if total <= 20*np.sqrt(variance.sum()):
        raise ValueError("Insufficient total-count SNR for moment linearization.")
    yy, xx = np.indices(counts.shape, dtype=float)
    weights = np.stack((xx, yy, xx**2, xx*yy, yy**2), axis=-1).reshape(-1, 5)
    measured = counts.ravel()@weights/total
    # Fixed covariance evaluated at the deterministic detector expectation.
    expected_total = expected.sum()
    expected_moments = expected.ravel()@weights/expected_total
    score = (weights-expected_moments)/expected_total
    covariance = score.T@(variance.ravel()[:, None]*score)
    return measured, covariance


def infer_unbounded_gaussians(likelihood, count_images):
    """Fit measured finite moments in a non-saturating Gaussian coordinate.

    Log-count quadratic fits provide an image-derived seed; the actual fit
    uses the finite detector and MTF. These reconstructions initialize optics,
    while the final likelihood remains the original measured moments.
    """
    unit = 1e-3
    coordinates = likelihood.detector.optical_coords_xy_m.reshape(-1, 2)/unit
    x, y = coordinates.T
    design = np.column_stack((np.ones(x.size), x, y, x*x+y*y))

    def residual(parameters, observed, whitener):
        prediction = finite_gaussian_raw_moments(parameters[:2]*unit,
            jnp.eye(2)*jnp.exp(2*parameters[2])*unit**2, likelihood.detector)
        return whitener@(prediction-observed)

    function, jacobian = jax.jit(residual), jax.jit(jax.jacfwd(residual))
    fits, covariances, diagnostics = [], [], []
    for counts, observed, whitener in zip(count_images, likelihood.observed, likelihood.whitener):
        values = np.asarray(counts).ravel()
        mask = values > 10
        weighted = np.sqrt(values[mask])
        coefficients = np.linalg.lstsq(design[mask]*weighted[:, None],
                                       np.log(values[mask])*weighted, rcond=None)[0]
        precision = max(-2*coefficients[3], 1e-4)
        initial = np.r_[coefficients[1:3]/precision, -.5*np.log(precision)]
        fit = least_squares(lambda p:np.asarray(function(p, observed, whitener)), initial,
                            jac=lambda p:np.asarray(jacobian(p, observed, whitener)),
                            x_scale="jac", max_nfev=100, ftol=1e-10, xtol=1e-10, gtol=1e-7)
        if not fit.success or not np.all(np.isfinite(fit.x)):
            raise RuntimeError("A finite-image Gaussian reconstruction failed; no optical seed admitted.")
        fits.append(fit.x)
        covariances.append(np.linalg.inv(fit.jac.T@fit.jac))
        diagnostics.append({"cost":float(fit.cost), "nfev":fit.nfev})
    return np.asarray(fits), np.asarray(covariances), diagnostics


@dataclass
class ProbeImageLikelihood:
    column: object
    detector: object
    requests: tuple
    observed: np.ndarray
    covariance: np.ndarray
    sample_mean: np.ndarray
    sample_covariance: np.ndarray
    held_mask: np.ndarray
    provenance: dict

    def __post_init__(self):
        self.n_joint_parameters = self.column.n_parameters
        self.n_nuisance_latents = 0
        self.calibration_latent_names = ()
        self.nuisance_scales = np.empty(0)
        self.train_indices = np.flatnonzero(~self.held_mask)
        self.held_indices = np.flatnonzero(self.held_mask)
        self.layout = SimpleNamespace(observation_keys=tuple(
            f"frame_{self.requests[i].frame_id}.{name}" for i in self.train_indices
            for name in ("x", "y", "xx", "xy", "yy")))
        self.whitener = jnp.asarray(np.linalg.inv(np.linalg.cholesky(self.covariance)))
        dacs = jnp.asarray([r.raw_dac_values for r in self.requests])
        means = jnp.asarray([r.sample_state.as_array()+self.sample_mean for r in self.requests])
        covariance = jnp.asarray(self.sample_covariance)

        def one(parameters, control, mean):
            matrix = self.column.matrix(parameters, control, segment="sample_to_detector")[[0, 2], :]
            return finite_gaussian_raw_moments(matrix@mean, matrix@covariance@matrix.T, self.detector)

        self.predict = jax.jit(jax.vmap(one, in_axes=(None, 0, 0)))
        self.dacs, self.means = dacs, means

    def join_parameters(self, parameters):
        return np.asarray(parameters)

    def all_residuals(self, parameters):
        difference = self.predict(parameters, self.dacs, self.means)-jnp.asarray(self.observed)
        return jnp.einsum("nij,nj->ni", self.whitener, difference)

    def data_residual(self, parameters):
        return self.all_residuals(parameters)[self.train_indices].reshape(-1)

    def joint_residual(self, parameters):
        return self.data_residual(parameters)

    def held_residual(self, parameters):
        return self.all_residuals(parameters)[self.held_indices].reshape(-1)


def make_probe_likelihood(snapshot, root, column, sample_mean, sample_covariance):
    """Read the same GPU arrays and reproduce their original Poisson draws."""
    if jax.default_backend() != "gpu":
        raise RuntimeError("The NEOARM Gaussian forward benchmark requires the GPU.")
    root = Path(root)
    run = load_saved_acquisition(
        root / "examples/generated/neoarm_inversion/physical_checkpoint_native_imaging_v5_gpu",
        root / "examples/lens_inversion/neoarm_inversion_config.toml")
    requests, observed, covariances, held, count_images = [], [], [], [], []
    metadata = snapshot["dataset"].metadata["frame_metadata"]
    for request in run.requests:
        if request.measurement not in {"perforation", "position_basis", "angle_basis", "basis_orthogonal", "basis_negative"}:
            continue
        record = metadata[str(request.frame_id)]
        if record["dark_offset_e"] != 0:
            raise ValueError("This conditional benchmark expects the archive's zero dark-count model.")
        expected = expected_detector_counts(run.image(request.frame_id), run.detector, record["dose_electrons"])
        rng = np.random.default_rng(record["noise_seed"])
        counts = rng.poisson(expected).astype(float)
        if record["read_noise_e"]:
            counts += rng.normal(0, record["read_noise_e"], expected.shape)
        measured, covariance = signed_raw_moments(counts, expected, record["read_noise_e"])
        requests.append(request)
        observed.append(measured)
        covariances.append(covariance)
        count_images.append(counts)
        held.append(snapshot["dataset"].plan.states[request.optical_state_index].role == "held_out")
    likelihood = ProbeImageLikelihood(column, run.detector, tuple(requests), np.asarray(observed),
        np.asarray(covariances), sample_mean, sample_covariance, np.asarray(held),
        {"manifest_sha256":run.checkpoint.manifest_sha256,
         "input_calibration":"conditional on exact synthetic sample-plane states and physical beam covariance",
         "noise_covariance":"analytic fixed Poisson plus read noise; no bootstrap",
         "omitted_families":"all material/Fourier/layer/Bragg estimators; pending independent estimator closure"})
    (likelihood.gaussian_estimates, likelihood.gaussian_estimate_covariances,
     likelihood.gaussian_fit_diagnostics) = infer_unbounded_gaussians(likelihood, count_images)
    return likelihood
