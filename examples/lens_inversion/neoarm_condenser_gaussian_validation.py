"""Independent finite-pixel Gaussian-mode validation for condenser images.

A Williamson decomposition gives a fixed pure Gaussian plus a positive,
rank-at-most-two distribution of source centres. This is an exact Gaussian
mixed state before cubature: modes propagate as GaussianBeam objects, then
their separately evaluated field intensities are summed. No detector crop is
renormalized and no covariance-image operator is used by the mode renderer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from temgym_core.constants import energy2wavelength
from temgym_core.evaluate import _beam_field, _prepare_gaussian_params
from temgym_core.gaussian import make_gaussian
from temgym_core.gaussian_image_likelihood import (
    make_gaussian_detector_operator, fit_gaussian_image,
    encode_gaussian_image_parameters, validate_phase_space_covariance,
)
from temgym_core.neoarm_calibration import DetectorGeometry, apply_transfer_matrix_to_gaussians
from neoarm_mixed_gaussian_validation import MixedGaussianSource, _blur_halo

ROOT = Path(__file__).resolve().parents[2]
XY, ANGLE = [0, 2], [1, 3]
OMEGA = np.kron(np.eye(2), [[0., 1.], [-1., 0.]])


def covariance_to_mixed_source(covariance, *, voltage_v=200e3, mean_state=None):
    """Return one unchanged, physical pure-plus-centres representation.

    In each Williamson pair, diag(nu,1/nu) is pure and bounded above by
    diag(nu,nu). Thus only one noisy centre coordinate per pair is needed.
    The choice of pair axes is a decomposition convention, not source data.
    """
    covariance = np.asarray(covariance, dtype=float)
    wavelength = float(energy2wavelength(voltage_v)); epsilon = wavelength/(4*np.pi)
    validate_phase_space_covariance(covariance, wavelength)
    length = float(np.sqrt(np.mean(np.diag(covariance)[XY])))
    scales = np.array([length, epsilon/length]*2)
    balanced = covariance/np.outer(scales, scales)
    values, vectors = np.linalg.eigh(balanced)
    root = (vectors*np.sqrt(values))@vectors.T
    frequencies, eigenvectors = np.linalg.eigh(1j*root@OMEGA@root)
    positive = np.flatnonzero(frequencies > 0)
    if len(positive) != 2: raise ValueError("Williamson spectrum must have two positive modes")
    nu = frequencies[positive]
    if nu.min() < 1-1e-8: raise ValueError("Sub-quantum symplectic emittance")
    orthogonal = np.column_stack([axis for k in positive for axis in
                                 (np.sqrt(2)*eigenvectors[:, k].real, -np.sqrt(2)*eigenvectors[:, k].imag)])
    williamson = root@orthogonal/np.sqrt(np.repeat(nu, 2))
    diagonal_pure = np.ravel(np.column_stack([nu, 1/nu]))
    pure_balanced = (williamson*diagonal_pure)@williamson.T
    centres_balanced = balanced-pure_balanced
    cvalues, cvectors = np.linalg.eigh((centres_balanced+centres_balanced.T)/2)
    if cvalues.min() < -1e-9*max(1., np.max(np.diag(balanced))):
        raise ValueError("Williamson pure covariance is not below total covariance")
    cvalues = np.maximum(cvalues, 0.)
    centres_balanced = (cvectors*cvalues)@cvectors.T
    pure = pure_balanced*np.outer(scales, scales)
    centres = centres_balanced*np.outer(scales, scales)
    position = pure[np.ix_(XY, XY)]
    curvature = np.linalg.solve(position, pure[np.ix_(XY, ANGLE)])
    curvature = (curvature+curvature.T)/2
    source = MixedGaussianSource(position, curvature, centres,
        np.zeros(4) if mean_state is None else np.asarray(mean_state), voltage_v)
    residual = (source.total_covariance-covariance)/np.outer(scales, scales)
    if np.linalg.norm(residual) > 1e-8*np.linalg.norm(balanced):
        raise ValueError("Mixed source decomposition does not close on input covariance")
    metadata = dict(symplectic_emittances_m_rad=(nu*epsilon).tolist(),
        minimum_quantum_emittance_m_rad=epsilon,
        pure_plus_centres_relative_covariance_error=float(np.linalg.norm(residual)/np.linalg.norm(balanced)),
        williamson_symplectic_error=float(np.linalg.norm(williamson.T@OMEGA@williamson-OMEGA)),
        source_decomposition_is_not_an_observed_coherent_mode_basis=True)
    return source, metadata


def rank_adapted_source_modes(source, order):
    if isinstance(order, bool) or int(order) != order or order < 1:
        raise ValueError("Hermite order must be a positive integer")
    scales = np.sqrt(np.diag(source.pure_covariance))
    normalized = source.centre_covariance/np.outer(scales, scales)
    values, vectors = np.linalg.eigh(normalized)
    keep = values > max(1., float(np.max(values)))*1e-11
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    nodes, weights = np.sqrt(2)*nodes, weights/np.sqrt(np.pi)
    rank = int(np.sum(keep))
    if rank:
        coordinates = np.stack(np.meshgrid(*([nodes]*rank), indexing="ij"), axis=-1).reshape(-1, rank)
        weights_grid = np.stack(np.meshgrid(*([weights]*rank), indexing="ij"), axis=-1)
        probabilities = np.prod(weights_grid, axis=-1).ravel()
        factor = scales[:, None]*vectors[:, keep]*np.sqrt(values[keep])
        centres = source.mean_state+coordinates@factor.T
    else:
        probabilities = np.ones(1); centres = source.mean_state[None]
    amplitude = 1/np.sqrt(2*np.pi*np.sqrt(np.linalg.det(source.position_covariance_xy_m2)))
    modes = make_gaussian(x=centres[:, 0], dx=centres[:, 1], y=centres[:, 2], dy=centres[:, 3],
        z=0., voltage=source.voltage_v, amp=amplitude).derive(
            Q_inv=jnp.broadcast_to(jnp.asarray(source.q_inverse), (len(centres), 2, 2)))
    discarded = np.where(keep, 0., values)
    return modes, probabilities, dict(centre_distribution_rank=rank, coherent_mode_count=len(centres),
        probability_sum=float(probabilities.sum()), discarded_normalized_centre_eigenvalues=discarded.tolist())


def validate_matrix(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("A finite canonical 4x4 source-to-detector map is required")
    # Full-column high-magnification maps are strongly squeezed. The formally
    # zero position-position block subtracts terms near 1e10; use its explicit
    # floating-point multiplication bound, not a blanket relaxed tolerance.
    roundoff = 32*np.finfo(float).eps*(abs(matrix).T@abs(OMEGA)@abs(matrix))
    if np.any(abs(matrix.T@OMEGA@matrix-OMEGA) > 1e-9+roundoff):
        raise ValueError("Source-to-detector matrix must be symplectic in canonical coordinates")
    return matrix


def render_covariance_gaussian_modes(covariance, matrix, detector, dose, *, hermite_order=7,
                                     pixel_order=8, voltage_v=200e3, mean_state=None,
                                     mode_batch=16, pixel_tile=128):
    """Actual independent GaussianBeam intensity propagation on GPU.

    Pixel centres and quadrature offsets are streamed separately: order32
    does not allocate a complete detector-times-quadrature coordinate array.
    """
    if jax.default_backend() != "gpu": raise RuntimeError("Gaussian forward simulation requires GPU")
    if not np.isfinite(dose) or dose <= 0: raise ValueError("Incident dose must be positive")
    if pixel_order < 1 or int(pixel_order) != pixel_order: raise ValueError("Invalid pixel quadrature order")
    if mode_batch < 1 or pixel_tile < 1: raise ValueError("Invalid stream tile")
    matrix = validate_matrix(matrix)
    source, metadata = covariance_to_mixed_source(covariance, voltage_v=voltage_v, mean_state=mean_state)
    modes, probabilities, mode_metadata = rank_adapted_source_modes(source, hermite_order)
    propagated = apply_transfer_matrix_to_gaussians(modes, matrix, z_out_m=detector.z_m, axial_path_m=0.)
    parameters = _prepare_gaussian_params(propagated)
    mode_padding = (-len(probabilities)) % mode_batch
    parameters = tuple(jnp.pad(value, ((0, mode_padding),)+((0, 0),)*(value.ndim-1))
        .reshape((-1, mode_batch)+value.shape[1:]) for value in parameters)
    probabilities = jnp.pad(jnp.asarray(probabilities), (0, mode_padding)).reshape(-1, mode_batch)
    radius = int(np.ceil(4*detector.mtf_sigma_px))
    ny, nx = detector.shape_yx; py, px = detector.pixel_size_yx_m
    yy, xx = np.meshgrid((np.arange(-radius, ny+radius)-(ny-1)/2)*py,
                        (np.arange(-radius, nx+radius)-(nx-1)/2)*px, indexing="ij")
    inverse = np.linalg.inv(detector.affine_xy)
    centres = np.stack([xx, yy], axis=-1).reshape(-1, 2)@inverse.T+detector.centre_xy_m
    nodes, weights = np.polynomial.legendre.leggauss(pixel_order)
    oy, ox = np.meshgrid(nodes*py/2, nodes*px/2, indexing="ij")
    offsets = np.stack([ox, oy], axis=-1).reshape(-1, 2)@inverse.T
    area_weights = np.outer(weights, weights).ravel()*px*py/(4*abs(np.linalg.det(detector.affine_xy)))
    pixel_count = len(centres); pixel_padding = (-pixel_count) % pixel_tile
    centres = jnp.pad(jnp.asarray(centres), ((0, pixel_padding), (0, 0))).reshape(-1, pixel_tile, 2)

    @jax.jit
    def evaluate(parameters, probabilities, centres, offsets, area_weights):
        def pixel_chunk(pixel_centres):
            coordinates = (pixel_centres[:, None]+offsets[None]).reshape(-1, 2)
            def mode_chunk(total, values):
                params, probability = values
                fields = jax.vmap(_beam_field, in_axes=(0, 0, 0, 0, 0, 0, None))(*params, coordinates)
                return total+probability@(jnp.real(fields)**2+jnp.imag(fields)**2), None
            density, _ = jax.lax.scan(mode_chunk, jnp.zeros(len(coordinates)), (parameters, probabilities))
            return density.reshape(pixel_tile, pixel_order**2)@area_weights
        pixels = jax.lax.map(pixel_chunk, centres).reshape(-1)[:pixel_count].reshape(ny+2*radius, nx+2*radius)
        return _blur_halo(pixels, detector.mtf_sigma_px, radius)*dose

    counts = evaluate(parameters, probabilities, centres, jnp.asarray(offsets), jnp.asarray(area_weights))
    metadata.update(mode_metadata, hermite_order=hermite_order, pixel_order=pixel_order,
        backend=jax.default_backend(), detector_crop_renormalized=False,
        maximum_stream_field_array_elements=mode_batch*pixel_tile*pixel_order**2,
        source_covariance=np.asarray(covariance).tolist(), full_canonical_matrix=matrix.tolist())
    return counts, metadata


def validate_gaussian_modes(covariance, matrices, detector, dose, *, names=None,
                            hermite_orders=(3, 5, 7), pixel_orders=(8, 16, 32)):
    """Selected-state convergence, independent image and observable checks."""
    if len(hermite_orders) < 3 or len(pixel_orders) < 3:
        raise ValueError("Two successive refinements are required separately for modes and pixel integration")
    rows, arrays = [], {}
    operator = jax.jit(make_gaussian_detector_operator(detector, pixel_orders[-1]))
    for index, matrix in enumerate(matrices):
        started = perf_counter(); matrix = validate_matrix(matrix)
        position = matrix[XY]@covariance@matrix[XY].T
        expected = np.asarray(operator(jnp.zeros(2), position, dose))
        variance = expected+.6**2
        discrepancy = lambda a, b: float(np.sqrt(np.mean((a-b)**2/variance)))
        previous = None; mode_rows = []; pixel_rows = []
        for order in hermite_orders:
            tick = perf_counter()
            actual, metadata = render_covariance_gaussian_modes(covariance, matrix, detector, dose,
                hermite_order=order, pixel_order=pixel_orders[0])
            actual = np.asarray(actual)
            mode_rows.append(dict(hermite_order=order, mode_count=metadata["coherent_mode_count"],
                image_rms_sigma=discrepancy(actual, expected),
                transition_rms_sigma=None if previous is None else discrepancy(actual, previous), elapsed_seconds=perf_counter()-tick))
            previous = actual
        for pixel_order in pixel_orders[1:]:
            tick = perf_counter()
            actual, metadata = render_covariance_gaussian_modes(covariance, matrix, detector, dose,
                hermite_order=hermite_orders[-1], pixel_order=pixel_order)
            actual = np.asarray(actual)
            pixel_rows.append(dict(pixel_order=pixel_order, image_rms_sigma=discrepancy(actual, expected),
                transition_rms_sigma=discrepancy(actual, previous), elapsed_seconds=perf_counter()-tick))
            previous = actual
        reference_parameters = np.asarray(encode_gaussian_image_parameters(np.zeros(2), position, dose, detector))
        fit = fit_gaussian_image(actual, variance, detector, initial_parameters=reference_parameters,
            quadrature_order=pixel_orders[-1], max_nfev=80)
        delta = np.asarray(fit.parameters)-reference_parameters
        standard_errors = np.sqrt(np.diag(fit.parameter_covariance))
        parameter_bias = delta/standard_errors
        information_bias = float(np.linalg.norm(np.asarray(fit.whitened_parameter_jacobian)@delta))
        accepted = bool(fit.optimizer_result.success and discrepancy(actual, expected) < .2
            and max(abs(parameter_bias)) < .2 and information_bias < .2
            and all(row["transition_rms_sigma"] < .2 for row in mode_rows[-2:])
            and all(row["transition_rms_sigma"] < .2 for row in pixel_rows[-2:]))
        row = dict(name=str(index) if names is None else names[index], accepted=accepted,
            mode_refinements=mode_rows, pixel_refinements=pixel_rows, metadata=metadata,
            captured_fraction_gaussian_modes=float(actual.sum()/dose), captured_fraction_reference=float(expected.sum()/dose),
            image_rms_sigma=discrepancy(actual, expected), observable_fit_optimizer_success=bool(fit.optimizer_result.success),
            fitted_image_parameter_bias_sigma=parameter_bias.tolist(), observable_joint_bias_sigma=information_bias,
            elapsed_seconds=perf_counter()-started)
        arrays[f"state{index}_reference_counts"] = expected
        arrays[f"state{index}_gaussian_mode_counts"] = actual
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("name", "accepted", "image_rms_sigma", "observable_joint_bias_sigma", "elapsed_seconds")}), flush=True)
    return rows, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-npz", type=Path, help="Arrays source_covariance(4,4), matrices(n,4,4); optional names.")
    parser.add_argument("--dose", type=float, default=2e6)
    args = parser.parse_args(); args.output = args.output.resolve()
    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "gpu": raise RuntimeError("All forward images must run on GPU")
    inputs = {}
    if args.input_npz is not None:
        data = np.load(args.input_npz, allow_pickle=False)
        covariance, matrices = data["source_covariance"], data["matrices"]
        names = data["names"].astype(str).tolist() if "names" in data else None
        inputs[str(args.input_npz.resolve())] = hashlib.sha256(args.input_npz.read_bytes()).hexdigest()
    else:
        from neoarm_aperture_capture import load_cases
        path = ROOT/"examples/generated/neoarm_inversion/imaging_recipe/shared_objective_information_narrow_probe.json"
        cases, covariance, _, _ = load_cases(path)
        selected = [0, 28]
        matrices = np.asarray([cases[i].source_to_detector for i in selected])
        names = ["focused_family0_40k", "narrow_family14_40k"]
        inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    from neoarm_shared_objective_audit import provenance_snapshot, verify_provenance_snapshot
    sources, computational_inputs = provenance_snapshot()
    started = perf_counter(); detector = DetectorGeometry(mtf_sigma_px=.65)
    rows, arrays = validate_gaussian_modes(covariance, matrices, detector, args.dose, names=names)
    verify_provenance_snapshot(sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    array_path = args.output.with_suffix(".npz"); np.savez_compressed(array_path, **arrays,
        source_covariance=covariance, matrices=matrices)
    report = dict(schema="neoarm-condenser-independent-gaussian-modes-v1", source_hashes=sources,
        computational_input_hashes=computational_inputs, input_hashes=inputs,
        array_file=str(array_path.relative_to(ROOT)), array_sha256=hashlib.sha256(array_path.read_bytes()).hexdigest(),
        backend=jax.default_backend(), dose_electrons=args.dose, elapsed_seconds=perf_counter()-started,
        records=rows, all_selected_images_accepted=all(row["accepted"] for row in rows),
        selected_states_only=True, validation_over_parameter_bounds_demonstrated=False,
        aperture_and_specimen_masks_present=False,
        physical_scope="Mixed Gaussian source; paraxial canonical full-column maps; unmasked vacuum path; finite-pixel MTF halo; incident dose is not detector-crop normalization.")
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(str(args.output), flush=True)


if __name__ == "__main__":
    main()
