"""Primitive canonical Gaussian transport for the frozen internal-Q experiment.

Every physical drift, hard-edge slice, thin quadrupole and original objective
midpoint slice is checked by the unchanged canonical-map guard. No assembled
map is projected, repaired or supplied to Gaussian propagation. Objective
subdivision repeats each original constant midpoint field: it tests numerical
transport, not convergence of the physical field model or paraxial adequacy.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from neoarm_condenser_gaussian_validation import (
    validate_matrix, covariance_to_mixed_source, rank_adapted_source_modes,
)
from neoarm_mixed_gaussian_validation import _blur_halo
from temgym_core.evaluate import _beam_field, _prepare_gaussian_params
from temgym_core.immersed_objective import uniform_field_canonical_matrix
from temgym_core.microscope_inversion import drift_matrix
from temgym_core.neoarm_calibration import apply_transfer_matrix_to_gaussians

ORDER = np.array([0, 2, 1, 3])


@dataclass(frozen=True)
class CanonicalFactors:
    matrices: np.ndarray
    output_z_m: np.ndarray
    labels: tuple

    def __post_init__(self):
        matrices = np.asarray(self.matrices, dtype=float)
        z = np.asarray(self.output_z_m, dtype=float)
        if (matrices.ndim != 3 or matrices.shape[1:] != (4, 4)
                or z.shape != (len(matrices),) or len(self.labels) != len(matrices)
                or not np.isfinite(z).all() or np.any(np.diff(z) < -1e-14)):
            raise ValueError('Ordered finite primitive factors and physical output planes are required.')
        for matrix in matrices:
            validate_matrix(matrix)
        object.__setattr__(self, 'matrices', matrices)
        object.__setattr__(self, 'output_z_m', z)

    def product(self):
        """Diagnostic only; this ill-conditioned product is not a wave operator."""
        return jax.lax.scan(lambda carry, matrix: (matrix@carry, None),
                            jnp.eye(4), jnp.asarray(self.matrices))[0]


def internal_q_factors(model, conditional, parameters, control, analyser_index,
                       *, hardedge_slices=8, objective_subdivisions=1,
                       objective_block_size=64):
    """Reconstruct the exact frozen field's canonical factors, in propagation order."""
    for count in (hardedge_slices, objective_subdivisions, objective_block_size):
        if isinstance(count, bool) or int(count) != count or count < 1:
            raise ValueError('Positive integer subdivision counts are required.')
    if analyser_index not in (0, 1) or np.shape(control) != (3,):
        raise ValueError('Use one of the two frozen analysers and three condenser controls.')
    if np.any(np.asarray(conditional.reference_controls) != 0):
        raise ValueError('The frozen internal-Q model assumes zero condenser DAC references.')
    optical, _, z_q, q_matrix = model.decode(parameters)
    matrices, zs, labels = [], [], []
    cursor = model.source_z_m

    def add(matrix, z, label):
        matrices.append(jnp.asarray(matrix)[None]); zs.append(jnp.asarray(z)[None]); labels.append(label)

    def gap(target):
        nonlocal cursor
        if float(target)-float(cursor) < -1e-14:
            raise ValueError('Declared physical factors overlap or change component order.')
        add(drift_matrix(target-cursor), target, 'drift')
        cursor = target

    def body(centre, length, q, name):
        nonlocal cursor
        gap(centre-length/2)
        entrance = cursor
        for index in range(hardedge_slices):
            add(uniform_field_canonical_matrix(q/length, length/hardedge_slices),
                entrance+length*(index+1)/hardedge_slices, name)
        cursor = centre+length/2

    for index, name in enumerate(('CL1', 'CL3', 'C-mini')):
        if index == model.insert_before:
            gap(z_q); add(q_matrix[jnp.ix_(ORDER, ORDER)], z_q, 'internal-Q')
        centre, log_length, bias, slope = optical[index]
        body(centre, jnp.exp(log_length), bias+slope*control[index], name)
    objective = conditional.column.objective
    gap(objective.z_edges_m[0])
    edges = jnp.asarray(objective.z_edges_m)
    rates = objective.field_samples(jnp.asarray(conditional.fixed[-4:]))
    distances = jnp.diff(edges)/objective_subdivisions
    objective_matrices = uniform_field_canonical_matrix(rates, distances)
    objective_matrices = jnp.repeat(objective_matrices, objective_subdivisions, axis=0)
    objective_z = (edges[:-1, None]+distances[:, None]*jnp.arange(1, objective_subdivisions+1)[None]).reshape(-1)
    # Short blocks avoid thousands of tiny GPU launches. Validate BOTH all
    # physical primitives and each short product; no strongly squeezed full
    # column product is used. Reducing block size tests this grouping error.
    for primitive in np.asarray(objective_matrices):
        validate_matrix(primitive)
    padding = (-len(objective_matrices)) % objective_block_size
    objective_matrices = jnp.concatenate((objective_matrices, jnp.tile(jnp.eye(4), (padding, 1, 1))))
    objective_z = jnp.pad(objective_z, (0, padding), mode='edge')
    blocks = objective_matrices.reshape(-1, objective_block_size, 4, 4)
    def product(block):
        return jax.lax.scan(lambda carry, m: (m@carry, None), jnp.eye(4), block)[0]
    matrices.append(jax.jit(jax.vmap(product))(blocks))
    zs.append(objective_z.reshape(-1, objective_block_size)[:, -1])
    labels.extend(('objective-midpoint-block',)*len(blocks))
    cursor = objective.z_edges_m[-1]
    controls = np.asarray(conditional.analyser_controls)[analyser_index]
    for lens, centre, length, bias, slope in conditional.column._decoded_lenses(conditional.fixed)[3:]:
        q = bias+slope*(controls[lens.channel_index]-lens.reference_dac)
        body(centre, length, q, lens.name)
    gap(conditional.fixed[1])
    return CanonicalFactors(np.asarray(jnp.concatenate(matrices)), np.asarray(jnp.concatenate(zs)), tuple(labels))


@jax.jit
def _propagate(beam, matrices, output_z):
    # Each step uses the existing Gaussian law, including its quadratic
    # beam-dependent action. Do not reinterpret canonical momenta as slopes
    # at the immersed sample or insert an extra fringe/shear there.
    def advance(current, values):
        matrix, z = values
        return apply_transfer_matrix_to_gaussians(current, matrix, z_out_m=z, axial_path_m=0.), None
    result, _ = jax.lax.scan(advance, beam, (matrices, output_z))
    # Endpoint actions telescope exactly for a homogeneous quadratic optical
    # Hamiltonian. Evaluate the same beam-dependent total once to avoid loss
    # of low bits during cancellation between many intermediate actions.
    action = .5*(jnp.sum(result.r_xy*result.d_xy, axis=-1)-jnp.sum(beam.r_xy*beam.d_xy, axis=-1))
    return result.derive(pathlength=beam.pathlength+action)


def propagate_factors(beam, factors):
    """Preserve the canonical field; only a common axial phase is omitted."""
    if not isinstance(factors, CanonicalFactors):
        raise TypeError('Validated primitive CanonicalFactors are required.')
    return _propagate(beam, jnp.asarray(factors.matrices), jnp.asarray(factors.output_z_m))


def evaluate_incoherent_modes(beam, probabilities, detector, dose, *, pixel_order=8,
                              mode_batch=16, pixel_tile=128):
    """GPU complex-field evaluation, intensity sum, finite pixels and PSF halo.

    This intentionally repeats the independently tested streamed2D quadrature
    policy, but accepts an already propagated bundle instead of a full map.
    """
    if jax.default_backend() != 'gpu':
        raise RuntimeError('Forward Gaussian images require GPU; no CPU fallback.')
    if not np.isfinite(dose) or dose <= 0:
        raise ValueError('Incident dose must be finite and positive.')
    for value in (pixel_order, mode_batch, pixel_tile):
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError('Positive integer pixel/mode tile sizes are required.')
    probability = np.asarray(probabilities)
    if probability.shape != (beam.x.shape[0],) or np.any(probability < 0) or not np.isclose(probability.sum(), 1.):
        raise ValueError('One normalized nonnegative probability per incoherent mode is required.')
    parameters = _prepare_gaussian_params(beam)
    padding = (-len(probability)) % mode_batch
    parameters = tuple(jnp.pad(value, ((0, padding),)+((0, 0),)*(value.ndim-1))
        .reshape((-1, mode_batch)+value.shape[1:]) for value in parameters)
    probabilities = jnp.pad(jnp.asarray(probability), (0, padding)).reshape(-1, mode_batch)
    radius = int(np.ceil(4*detector.mtf_sigma_px))
    ny, nx = detector.shape_yx; py, px = detector.pixel_size_yx_m
    yy, xx = np.meshgrid((np.arange(-radius, ny+radius)-(ny-1)/2)*py,
                        (np.arange(-radius, nx+radius)-(nx-1)/2)*px, indexing='ij')
    inverse = np.linalg.inv(detector.affine_xy)
    centres = np.stack([xx, yy], axis=-1).reshape(-1, 2)@inverse.T+detector.centre_xy_m
    nodes, weights = np.polynomial.legendre.leggauss(pixel_order)
    oy, ox = np.meshgrid(nodes*py/2, nodes*px/2, indexing='ij')
    offsets = np.stack([ox, oy], axis=-1).reshape(-1, 2)@inverse.T
    area = np.outer(weights, weights).ravel()*px*py/(4*abs(np.linalg.det(detector.affine_xy)))
    pixel_count = len(centres); padding = (-pixel_count) % pixel_tile
    centres = jnp.pad(jnp.asarray(centres), ((0, padding), (0, 0))).reshape(-1, pixel_tile, 2)

    @jax.jit
    def evaluate(parameters, probabilities, centres, offsets, area):
        def chunk(pixel_centres):
            points = (pixel_centres[:, None]+offsets[None]).reshape(-1, 2)
            def add(total, values):
                params, probability = values
                fields = jax.vmap(_beam_field, in_axes=(0, 0, 0, 0, 0, 0, None))(*params, points)
                return total+probability@(jnp.real(fields)**2+jnp.imag(fields)**2), None
            density, _ = jax.lax.scan(add, jnp.zeros(len(points)), (parameters, probabilities))
            return density.reshape(pixel_tile, pixel_order**2)@area
        counts = jax.lax.map(chunk, centres).reshape(-1)[:pixel_count].reshape(ny+2*radius, nx+2*radius)
        return _blur_halo(counts, detector.mtf_sigma_px, radius)*dose
    return evaluate(parameters, probabilities, centres, jnp.asarray(offsets), jnp.asarray(area))


def render_factorized_modes(covariance, factors, detector, dose, *, hermite_order=13,
                             pixel_order=8, voltage_v=200e3):
    source, metadata = covariance_to_mixed_source(covariance, voltage_v=voltage_v)
    modes, probabilities, mode_metadata = rank_adapted_source_modes(source, hermite_order)
    propagated = propagate_factors(modes, factors)
    counts = evaluate_incoherent_modes(propagated, probabilities, detector, dose, pixel_order=pixel_order)
    metadata.update(mode_metadata, primitive_count=len(factors.matrices),
        hermite_order=hermite_order, pixel_order=pixel_order, backend=jax.default_backend(),
        common_axial_phase_omitted=True, beam_dependent_endpoint_action_preserved=True,
        factor_guards_unchanged=True, detector_crop_renormalized=False)
    return counts, metadata
