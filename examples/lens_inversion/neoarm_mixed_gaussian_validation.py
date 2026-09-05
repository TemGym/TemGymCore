"""Incoherent astigmatic Gaussian-mode transport versus covariance images.

A common pure Gaussian plus a positive Gaussian distribution of 4D centres
defines a physically valid mixed Gaussian Wigner distribution. A positive
Gauss--Hermite cubature approximates that one fixed source distribution at
every optical state. Existing GaussianBeam propagation and field evaluation
are used mode by mode; weighted INTENSITIES, never amplitudes, are summed.

This validates a partial-coherence model and detector integration, not a
unique coherent-mode decomposition, a source preparation control, or recovery
of an unknown microscope. State order is [x, theta_x, y, theta_y].
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import hashlib
import json
import os
import time

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jnp
import numpy as np

from temgym_core.constants import energy2wavelength
from temgym_core.evaluate import _beam_field, _prepare_gaussian_params
from temgym_core.gaussian import make_gaussian
from temgym_core.neoarm_calibration import DetectorGeometry, apply_transfer_matrix_to_gaussians

J4 = np.kron(np.eye(2), np.array([[0., 1.], [-1., 0.]]))
POSITION = [0, 2]
ANGLE = [1, 3]


def _symmetric(name, value, shape):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {shape} matrix.")
    scale = max(float(np.max(np.abs(array))), np.finfo(float).tiny)
    if np.max(np.abs(array-array.T)) > 1e-12*scale:
        raise ValueError(f"{name} must be symmetric.")
    return .5*(array+array.T)


def physical_emittances(covariance, wavelength_m):
    """Check Gaussian uncertainty and return its two symplectic emittances."""
    covariance = _symmetric("covariance", covariance, (4, 4))
    if np.any(np.diag(covariance) <= 0) or not np.isfinite(wavelength_m) or wavelength_m <= 0:
        raise ValueError("Covariance diagonals and wavelength must be positive.")
    rscale = np.sqrt(np.mean(np.diag(covariance)[POSITION]))
    pscale = np.sqrt(np.mean(np.diag(covariance)[ANGLE]))
    scales = np.array([rscale, pscale, rscale, pscale])
    normalized = covariance / scales[:, None] / scales[None, :]
    epsilon = wavelength_m/(4*np.pi)
    if np.linalg.eigvalsh(normalized+1j*epsilon/(rscale*pscale)*J4).min() < -1e-10:
        raise ValueError("Covariance violates the Gaussian uncertainty bound lambda/(4*pi).")
    values = np.sort(np.abs(np.linalg.eigvals(1j*J4@normalized)))*rscale*pscale
    return .5*(values[::2]+values[1::2])


@dataclass(frozen=True)
class MixedGaussianSource:
    position_covariance_xy_m2: np.ndarray
    curvature_xy_per_m: np.ndarray
    centre_covariance: np.ndarray
    mean_state: np.ndarray
    voltage_v: float = 200e3

    def __post_init__(self):
        position = _symmetric("position covariance", self.position_covariance_xy_m2, (2, 2))
        curvature = _symmetric("curvature", self.curvature_xy_per_m, (2, 2))
        centres = _symmetric("centre covariance", self.centre_covariance, (4, 4))
        mean = np.asarray(self.mean_state, dtype=float)
        if np.linalg.eigvalsh(position).min() <= 0 or mean.shape != (4,) or not np.all(np.isfinite(mean)):
            raise ValueError("Pure-mode position covariance must be positive definite; mean must be a finite 4-vector.")
        if not np.isfinite(self.voltage_v) or self.voltage_v <= 0:
            raise ValueError("Voltage must be positive and finite.")
        for name, value in (("position_covariance_xy_m2", position), ("curvature_xy_per_m", curvature),
                            ("centre_covariance", centres), ("mean_state", mean)):
            object.__setattr__(self, name, value)
        scales = np.sqrt(np.diag(self.pure_covariance))
        normalized = centres/scales[:, None]/scales[None, :]
        if np.linalg.eigvalsh(normalized).min() < -1e-12:
            raise ValueError("The incoherent centre covariance must be positive semidefinite.")
        physical_emittances(self.total_covariance, self.wavelength_m)

    @property
    def wavelength_m(self):
        return float(energy2wavelength(self.voltage_v))

    @property
    def q_inverse(self):
        return self.curvature_xy_per_m + 1j*self.wavelength_m/(4*np.pi)*np.linalg.inv(self.position_covariance_xy_m2)

    @property
    def pure_covariance(self):
        p, h = self.position_covariance_xy_m2, self.curvature_xy_per_m
        epsilon = self.wavelength_m/(4*np.pi)
        grouped = np.block([[p, p@h], [h@p, h@p@h+epsilon**2*np.linalg.inv(p)]])
        return grouped[np.ix_([0, 2, 1, 3], [0, 2, 1, 3])]

    @property
    def total_covariance(self):
        return self.pure_covariance+self.centre_covariance


def source_modes(source, order):
    """One source-space four-dimensional positive Gauss--Hermite cubature."""
    if isinstance(order, bool) or int(order) != order or order < 1:
        raise ValueError("Hermite order must be a positive integer.")
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    nodes, weights = np.sqrt(2)*nodes, weights/np.sqrt(np.pi)
    coordinates = np.stack(np.meshgrid(*([nodes]*4), indexing="ij"), axis=-1).reshape(-1, 4)
    weight_grid = np.stack(np.meshgrid(*([weights]*4), indexing="ij"), axis=-1)
    mode_weights = np.prod(weight_grid, axis=-1).ravel()
    scales = np.sqrt(np.diag(source.pure_covariance))
    eigenvalues, eigenvectors = np.linalg.eigh(source.centre_covariance/scales[:, None]/scales[None, :])
    factor = scales[:, None]*eigenvectors*np.sqrt(np.maximum(eigenvalues, 0))
    centres = source.mean_state+coordinates@factor.T
    amplitude = 1/np.sqrt(2*np.pi*np.sqrt(np.linalg.det(source.position_covariance_xy_m2)))
    modes = make_gaussian(x=centres[:, 0], dx=centres[:, 1], y=centres[:, 2], dy=centres[:, 3],
                          z=0., voltage=source.voltage_v, amp=amplitude).derive(
                              Q_inv=jnp.broadcast_to(jnp.asarray(source.q_inverse), (len(centres), 2, 2)))
    return modes, mode_weights


def detector_quadrature(detector, order):
    """Independent 2D pixel-area quadrature, including the detector-MTF halo."""
    if isinstance(order, bool) or int(order) != order or order < 1:
        raise ValueError("Pixel quadrature order must be a positive integer.")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    radius = int(np.ceil(4*detector.mtf_sigma_px))
    ny, nx = detector.shape_yx
    py, px = detector.pixel_size_yx_m
    yy, xx = np.meshgrid((np.arange(-radius, ny+radius)-(ny-1)/2)*py,
                         (np.arange(-radius, nx+radius)-(nx-1)/2)*px, indexing="ij")
    oy, ox = np.meshgrid(nodes*py/2, nodes*px/2, indexing="ij")
    physical = np.stack((xx, yy), axis=-1).reshape(-1, 1, 2)+np.stack((ox, oy), axis=-1).reshape(1, -1, 2)
    optical = np.einsum("ij,...j->...i", np.linalg.inv(detector.affine_xy), physical)+detector.centre_xy_m
    area_weights = np.outer(weights, weights).ravel()*px*py/(4*abs(np.linalg.det(detector.affine_xy)))
    return optical, area_weights, (ny+2*radius, nx+2*radius), radius


def _blur_halo(counts, sigma, radius):
    if radius == 0:
        return counts
    axis = jnp.arange(-radius, radius+1)
    kernel = jnp.exp(-.5*(axis/sigma)**2)
    kernel /= kernel.sum()
    value = counts[None, ..., None]
    for shape in ((-1, 1, 1, 1), (1, -1, 1, 1)):
        value = jax.lax.conv_general_dilated(value, kernel.reshape(shape), (1, 1), "VALID",
                                            dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return value[0, ..., 0]


def render_mixed_counts(source, matrix, detector, *, hermite_order, pixel_order=4,
                        incident_electrons=2e7, mode_batch=32, pixel_tile=512, require_gpu=True):
    """Propagate/evaluate independent modes on GPU, integrating their intensities."""
    if require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError("Mixed-beam forward validation requires a GPU; no CPU fallback.")
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)) or not np.allclose(matrix.T@J4@matrix, J4, atol=1e-8, rtol=1e-8):
        raise ValueError("The transfer matrix must be finite and symplectic.")
    if not np.isfinite(incident_electrons) or incident_electrons <= 0:
        raise ValueError("Incident electron count must be positive and finite.")
    if mode_batch < 1 or pixel_tile < 1:
        raise ValueError("Mode and pixel tile sizes must be positive.")
    modes, weights = source_modes(source, hermite_order)
    # A common axial phase is irrelevant to every independently squared mode.
    propagated = apply_transfer_matrix_to_gaussians(modes, matrix, z_out_m=detector.z_m, axial_path_m=0.)
    parameters = _prepare_gaussian_params(propagated)
    mode_padding = (-len(weights)) % mode_batch
    parameters = tuple(jnp.pad(value, ((0, mode_padding),)+((0, 0),)*(value.ndim-1))
                       .reshape((-1, mode_batch)+value.shape[1:]) for value in parameters)
    weights = jnp.pad(jnp.asarray(weights), (0, mode_padding)).reshape(-1, mode_batch)
    coordinates, area_weights, halo_shape, radius = detector_quadrature(detector, pixel_order)
    pixel_count = coordinates.shape[0]
    pixel_padding = (-pixel_count) % pixel_tile
    coordinates = jnp.pad(jnp.asarray(coordinates), ((0, pixel_padding), (0, 0), (0, 0)))
    coordinates = coordinates.reshape(-1, pixel_tile*pixel_order**2, 2)
    area_weights = jnp.asarray(area_weights)

    @jax.jit
    def evaluate(parameters, weights, coordinates, area_weights):
        def pixel_chunk(coords):
            def mode_chunk(total, values):
                params, weight = values
                fields = jax.vmap(_beam_field, in_axes=(0, 0, 0, 0, 0, 0, None))(*params, coords)
                return total+weight@(jnp.real(fields)**2+jnp.imag(fields)**2), None
            density, _ = jax.lax.scan(mode_chunk, jnp.zeros(coords.shape[0]), (parameters, weights))
            return density.reshape(pixel_tile, pixel_order**2)@area_weights
        pixels = jax.lax.map(pixel_chunk, coordinates).reshape(-1)[:pixel_count].reshape(halo_shape)
        return _blur_halo(pixels, detector.mtf_sigma_px, radius)*incident_electrons

    return evaluate(parameters, weights, coordinates, area_weights)


def default_source():
    angle = .29
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    position = rotation@np.diag(np.array([35e-9, 20e-9])**2)@rotation.T
    curvature = np.array([[250., -120.], [-120., -100.]])
    scale = np.diag([12e-9, 6e-6, 10e-9, 4e-6])
    lower = np.array([[1., 0., 0., 0.], [.25, 1., 0., 0.], [-.15, .3, 1., 0.], [.2, -.1, .35, 1.]])
    factor = scale@lower
    return MixedGaussianSource(position, curvature, factor@factor.T, np.array([6e-9, 2e-6, -4e-9, -3e-6]))


def imaging_map(magnification, angle, specimen_defocus_m):
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    matrix = np.zeros((4, 4))
    matrix[np.ix_(POSITION, POSITION)] = magnification*rotation
    matrix[np.ix_(POSITION, ANGLE)] = magnification*rotation*specimen_defocus_m
    matrix[np.ix_(ANGLE, ANGLE)] = rotation/magnification
    return matrix


def audit(output):
    from temgym_core.gaussian_image_likelihood import (
        make_gaussian_detector_operator, fit_gaussian_image, encode_gaussian_image_parameters,
        GAUSSIAN_IMAGE_PARAMETER_NAMES,
    )

    output = Path(output)
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__), *(root/'src/temgym_core'/name for name in (
        'gaussian_image_likelihood.py', 'gaussian.py', 'evaluate.py', 'neoarm_calibration.py',
        'constants.py', 'jax_least_squares.py'))]
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    source_hashes = {str(path.relative_to(root)): digest(path) for path in paths}
    source = default_source()
    detector = DetectorGeometry(shape_yx=(256, 256), pixel_size_yx_m=(55e-6, 55e-6),
                                affine_xy=np.array([[1.006, .025], [-.028, .994]]),
                                centre_xy_m=np.array([.2e-3, -.1e-3]), mtf_sigma_px=.65, z_m=1.)
    operator = make_gaussian_detector_operator(detector, quadrature_order=8)
    finer_operator = make_gaussian_detector_operator(detector, quadrature_order=16)
    rows = []
    output.parent.mkdir(parents=True, exist_ok=True)
    for defocus in (-100e-6, 0., 100e-6):
        matrix = imaging_map(4e4, .37, defocus)
        response = matrix[POSITION]
        expected = np.asarray(operator(response@source.mean_state,
                                       response@source.total_covariance@response.T, 2e7))
        finer_expected = np.asarray(finer_operator(response@source.mean_state,
                                                  response@source.total_covariance@response.T, 2e7))
        sigma = np.sqrt(expected+.6**2)
        previous, consecutive, refinements = None, 0, []
        for order in (3, 5, 7, 9, 13):
            started = time.perf_counter()
            actual = np.asarray(render_mixed_counts(source, matrix, detector, hermite_order=order))
            elapsed = time.perf_counter()-started
            error = float(np.sqrt(np.mean(((actual-expected)/sigma)**2)))
            transition = None if previous is None else float(np.sqrt(np.mean(((actual-previous)/sigma)**2)))
            consecutive = consecutive+1 if transition is not None and transition < .2 else 0
            refinements.append({'hermite_order': order, 'coherent_mode_count': order**4,
                                'image_rms_sigma': error, 'transition_rms_sigma': transition,
                                'elapsed_s': elapsed, 'successive_converged_transitions': consecutive})
            print(f"d={defocus*1e6:+g} um; {order**4:,} incoherent modes; error={error:.6g} sigma; {elapsed:.2f}s", flush=True)
            previous = actual
            if error < .2 and consecutive >= 2:
                break
        fine_pixel_images, quadrature_checks = {}, []
        previous_pixel_image = actual
        for pixel_order in (8, 16):
            started = time.perf_counter()
            pixel_image = np.asarray(render_mixed_counts(source, matrix, detector, hermite_order=order,
                                                          pixel_order=pixel_order))
            elapsed = time.perf_counter()-started
            discrepancy = float(np.sqrt(np.mean(((pixel_image-expected)/sigma)**2)))
            difference = float(np.sqrt(np.mean(((pixel_image-previous_pixel_image)/sigma)**2)))
            quadrature_checks.append({'pixel_order_each_axis': pixel_order,
                                       'image_rms_sigma': discrepancy,
                                       'transition_rms_sigma': difference, 'elapsed_s': elapsed})
            fine_pixel_images[pixel_order] = pixel_image
            previous_pixel_image = pixel_image
            print(f"  pixel quadrature {pixel_order}x{pixel_order}: error={discrepancy:.6g} sigma; "
                  f"transition={difference:.6g} sigma; {elapsed:.2f}s", flush=True)
        truth_parameters = np.asarray(encode_gaussian_image_parameters(
            response@source.mean_state, response@source.total_covariance@response.T, 2e7, detector))
        initial_parameters = truth_parameters+np.array([.7, -.5, .02, -.015, .025, np.log(.97)])
        image_fit = fit_gaussian_image(fine_pixel_images[16], sigma**2, detector,
                                        initial_parameters=initial_parameters)
        image_fit.require_accepted()
        parameter_sigma = np.sqrt(np.diag(image_fit.parameter_covariance))
        parameter_bias = (image_fit.parameters-truth_parameters)/parameter_sigma
        u, singular, vh = np.linalg.svd(image_fit.whitened_parameter_jacobian, full_matrices=False)
        image_error = ((fine_pixel_images[16]-expected)/sigma).ravel()
        linear_bias = (vh.T@((u.T@image_error)/singular))/parameter_sigma
        fit_metadata = {
            'parameter_names': GAUSSIAN_IMAGE_PARAMETER_NAMES, 'truth_parameters': truth_parameters.tolist(),
            'initial_parameters': initial_parameters.tolist(), 'fitted_parameters': image_fit.parameters.tolist(),
            'conditional_parameter_covariance': image_fit.parameter_covariance.tolist(),
            'parameter_bias_sigma': parameter_bias.tolist(), 'linearized_parameter_bias_sigma': linear_bias.tolist(),
            'maximum_parameter_bias_sigma': float(np.max(np.abs(parameter_bias))),
            'maximum_linearized_parameter_bias_sigma': float(np.max(np.abs(linear_bias))),
            'numerical_rank': image_fit.numerical_rank, 'optimizer_success': image_fit.optimizer_result.success,
            'termination': image_fit.optimizer_result.status,
            'image_residual_rms_sigma': image_fit.residual_rms_sigma,
            'fit_pixel8_vs16_rms_sigma': image_fit.quadrature_rms_sigma,
            'fit_pixel8_vs16_maximum_parameter_shift_sigma': image_fit.quadrature_parameter_shift_sigma,
            'scope': 'Six effective image parameters only, conditioned on exact detector/source distribution; no extra ellipse-angle datum.',
        }
        image_path = output.with_name(f'mixed_gaussian_defocus_{defocus*1e6:+.0f}um.npz')
        np.savez_compressed(image_path, gaussian_mode_counts_yx=actual, covariance_counts_yx=expected,
                            gaussian_mode_counts_pixel8_yx=fine_pixel_images[8],
                            gaussian_mode_counts_pixel16_yx=fine_pixel_images[16],
                            fitted_gaussian_counts_yx=image_fit.fitted_counts_yx,
                            noise_sigma_yx=sigma, source_covariance=source.total_covariance, transfer_matrix=matrix)
        rows.append({'specimen_defocus_m': defocus, 'magnification': 4e4,
                     'passed': bool(error < .2 and consecutive >= 2
                                    and all(q['transition_rms_sigma'] < .2 and q['image_rms_sigma'] < .2 for q in quadrature_checks)
                                    and np.max(np.abs(parameter_bias)) < .2 and np.max(np.abs(linear_bias)) < .2),
                     'refinements': refinements, 'two_dimensional_pixel_quadrature': quadrature_checks,
                     'image_parameter_fit': fit_metadata,
                     'analytic_pixel8_vs16_rms_sigma': float(np.sqrt(np.mean(((expected-finer_expected)/sigma)**2))),
                     'artifact': str(image_path), 'artifact_sha256': digest(image_path)})
    if any(digest(root/path) != value for path, value in source_hashes.items()):
        raise RuntimeError('An audited source changed during execution.')
    result = {'schema': 'neoarm-physical-mixed-gaussian-validation-v1', 'all_passed': all(r['passed'] for r in rows),
              'backend': jax.default_backend(), 'device': str(jax.devices()[0]),
              'source_hashes': source_hashes, 'source_hashes_unchanged_during_execution': True,
              'coordinate_order': ['x', 'theta_x', 'y', 'theta_y'],
              'voltage_v': source.voltage_v, 'source_mean_state': source.mean_state.tolist(),
              'detector': {'shape_yx': list(detector.shape_yx), 'pixel_size_yx_m': list(detector.pixel_size_yx_m),
                           'affine_xy': np.asarray(detector.affine_xy).tolist(),
                           'centre_xy_m': np.asarray(detector.centre_xy_m).tolist(),
                           'mtf_sigma_px': detector.mtf_sigma_px, 'incident_electrons': 2e7, 'read_noise_e': .6},
              'pure_mode_covariance': source.pure_covariance.tolist(),
              'centre_covariance': source.centre_covariance.tolist(), 'total_covariance': source.total_covariance.tolist(),
              'minimum_emittance_m_rad': source.wavelength_m/(4*np.pi),
              'symplectic_emittances_m_rad': physical_emittances(source.total_covariance, source.wavelength_m).tolist(),
              'assumptions': ['Incoherent Gaussian distribution of pure Gaussian-mode centres; same source at every state.',
                              'Quadrature weights multiply intensities, not field amplitudes.',
                              'Pixel area integration and an extended optical halo precede the discrete detector MTF.',
                              'No sample, source-preparation control, condenser stigmator location or independent rotation tag is assumed.',
                              'The +/-100 micrometre maps are synthetic engine tests, not claims of reachable NEOARM DAC states.',
                              'This is a Gaussian partial-coherence transport validation, not a proof of full microscope inversion.'],
              'rows': rows}
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(f"All mixed-Gaussian validation states passed: {audit(args.output)['all_passed']}")
