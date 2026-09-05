"""Finite-area, halo-correct detector adapter for regular coherent beam grids.

The accelerator retains the actual Gaussian fields and endpoint phases. It
omits only distant neighbours with exponentially small field envelopes; that
approximation is audited against exhaustive GPU evaluation. It is not a
replacement for a general beam renderer: irregular grids or broad overlapping
beams fail the host-side admission contract. A frozen topology is required.
"""

from dataclasses import dataclass, asdict
from functools import partial
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import os
import time
import tomllib

os.environ.setdefault('JAX_ENABLE_X64', 'true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import gaussian_filter

from neoarm_measured_specimen import frozen_measured_view, require_view_coverage, detector_footprint
from neoarm_imaging_recipe import geometry_anchors_from_notebook
from neoarm_mixed_gaussian_validation import detector_quadrature
from temgym_core.constants import energy2wavelength
from temgym_core.evaluate import _beam_field, _prepare_gaussian_params, evaluate_gaussians
from temgym_core.gaussian_image_likelihood import apply_detector_psf
from temgym_core.multiscale_sample import MAGICALSampleAtlas
from temgym_core.neoarm_calibration import DetectorGeometry, reexpand_transmitted_field, apply_transfer_matrix_to_gaussians


@partial(jax.jit, static_argnames=('shape_yx', 'radius', 'point_tile'))
def evaluate_regular_lattice(parameters, coords_xy_m, *, shape_yx, radius, point_tile=512):
    """Coherent local sums; JAX derivatives retain all actual field parameters.

    Call ``RegularGaussianLattice.from_beam`` to validate the regular-grid and
    support contract first. Neighbour membership is piecewise constant; its
    neglected boundary fields must independently satisfy the truncation gate.
    """
    ny, nx = shape_yx
    centres = parameters[0]
    origin = centres[0]
    steps = jnp.stack((centres[1]-origin, centres[nx]-origin), axis=-1)
    inverse = jnp.linalg.inv(steps)
    axis = jnp.arange(-radius, radius+1)
    oy, ox = jnp.meshgrid(axis, axis, indexing='xy')
    offsets = jnp.stack((ox.ravel(), oy.ravel()), axis=-1)
    count = coords_xy_m.shape[0]
    padded = jnp.pad(coords_xy_m, ((0, (-count) % point_tile), (0, 0))).reshape(-1, point_tile, 2)

    def point(coords):
        centre_index = jnp.floor(inverse@(coords-origin)).astype(jnp.int32)
        indices = centre_index+offsets
        valid = ((indices[:, 0] >= 0) & (indices[:, 0] < nx)
                 & (indices[:, 1] >= 0) & (indices[:, 1] < ny))
        flat = jnp.clip(indices[:, 1], 0, ny-1)*nx+jnp.clip(indices[:, 0], 0, nx-1)
        selected = tuple(value[flat] for value in parameters)
        fields = jax.vmap(_beam_field, in_axes=(0, 0, 0, 0, 0, 0, None))(*selected, coords)
        return jnp.sum(jnp.where(valid, fields, 0.))

    return jax.lax.map(jax.vmap(point), padded).ravel()[:count]


@dataclass(frozen=True)
class RegularGaussianLattice:
    parameters: tuple
    shape_yx: tuple
    radius: int
    validation: dict

    @classmethod
    def from_beam(cls, beam, shape_yx, *, field_envelope_tolerance=1e-16,
                  maximum_radius=24, require_gpu=True):
        if require_gpu and jax.default_backend() != 'gpu':
            raise RuntimeError('Finite-pixel coherent forward simulation requires the GPU.')
        shape = tuple(int(n) for n in shape_yx)
        if len(shape) != 2 or min(shape) < 2:
            raise ValueError('A two-dimensional grid of at least 2x2 is required.')
        if not np.isfinite(field_envelope_tolerance) or not 0 < field_envelope_tolerance < 1:
            raise ValueError('Field envelope tolerance must lie strictly between zero and one.')
        parameters = tuple(_prepare_gaussian_params(beam))
        centres, _, _, _, q, k = parameters
        ny, nx = shape
        if centres.shape[0] != nx*ny:
            raise ValueError('Beam count differs from the declared grid shape.')
        origin = centres[0]
        steps = jnp.stack((centres[1]-origin, centres[nx]-origin), axis=-1)
        inverse = jnp.linalg.inv(steps)
        yy, xx = jnp.meshgrid(jnp.arange(ny), jnp.arange(nx), indexing='ij')
        expected_indices = jnp.stack((xx.ravel(), yy.ravel()), axis=-1)
        grid_error = float(jnp.max(jnp.abs((centres-origin)@inverse.T-expected_indices)))
        common_q = float(jnp.max(jnp.abs(q-q[0])))
        common_k = float(jnp.max(jnp.abs(k-k[0])))
        if not np.isfinite(grid_error) or grid_error > 1e-6 or common_q != 0 or common_k != 0:
            raise ValueError('Regular positions, identical Q matrices and one wavelength are required.')
        envelope = np.asarray(.5*k[0]*steps.T@jnp.imag(q[0])@steps)
        minimum_decay = np.linalg.eigvalsh(envelope).min()
        if not np.isfinite(minimum_decay) or minimum_decay <= 0:
            raise ValueError('Gaussian field envelopes must decay in both grid directions.')
        radius = int(np.ceil(np.sqrt(-np.log(field_envelope_tolerance)/minimum_decay)))+1
        if radius > maximum_radius:
            raise ValueError('The Gaussian support is too broad for this bounded-neighbour adapter.')
        return cls(parameters, shape, radius, dict(maximum_position_grid_error=grid_error,
            maximum_q_difference=common_q, maximum_wavenumber_difference=common_k,
            minimum_grid_field_decay=minimum_decay, field_envelope_tolerance=field_envelope_tolerance,
            neighbourhood_radius=radius, maximum_neighbours_per_query=(2*radius+1)**2,
            tail_bound_scope='Single omitted field envelope, not an image-error or cancellation bound.'))

    def __call__(self, coords_xy_m, *, radius=None):
        return evaluate_regular_lattice(self.parameters, jnp.asarray(coords_xy_m), shape_yx=self.shape_yx,
                                        radius=self.radius if radius is None else radius)


def integrated_gaussian_counts(lattice, detector, *, incident_electrons, quadrature_order=8, radius=None):
    """Integrate coherent intensity over pixels and the real external PSF halo."""
    if jax.default_backend() != 'gpu':
        raise RuntimeError('Finite-pixel Gaussian forward simulation requires the GPU.')
    coordinates, weights, halo_shape, _ = detector_quadrature(detector, quadrature_order)
    fields = lattice(jnp.asarray(coordinates.reshape(-1, 2)), radius=radius)
    intensity = jnp.real(fields)**2+jnp.imag(fields)**2
    counts = (intensity.reshape(-1, len(weights))@jnp.asarray(weights)).reshape(halo_shape)
    return incident_electrons*apply_detector_psf(counts, detector)


def make_finite_pixel_conjugate_predictor(view, detector, source_centre, *, reference_matrix,
                                         wavelength_m, shortest_scale_m, quadrature_order=8):
    """Differentiable seven-parameter cubic reference, independent of beamlets."""
    matrix = np.asarray(reference_matrix, float)
    a, b = matrix[np.ix_([0, 2], [0, 2])], matrix[np.ix_([0, 2], [1, 3])]
    determinant = np.linalg.det(a)
    if determinant <= 0 or not np.allclose(a@a.T/determinant, np.eye(2), atol=1e-8, rtol=0):
        raise ValueError('The conjugate template requires a similarity position map.')
    phase_bound = np.pi*wavelength_m*np.linalg.norm(np.linalg.solve(a, b), ord=2)/shortest_scale_m**2
    if not np.isfinite(phase_bound) or phase_bound > 1e-4:
        raise ValueError('The finite-pixel cubic intensity reference requires conjugacy at the retained scale.')
    coordinates, weights, halo_shape, _ = detector_quadrature(detector, quadrature_order)
    coordinates, weights, source = map(jnp.asarray, (coordinates, weights, source_centre))

    @jax.jit
    def predict(parameters):
        log_m, angle, tx_nm, ty_nm, log_dose, log_waist, log_contrast = parameters
        c, s = jnp.cos(angle), jnp.sin(angle)
        rotation = jnp.array([[c, -s], [s, c]])
        points = coordinates@rotation*jnp.exp(-log_m)
        query = points+jnp.asarray(view.centre_xy_m)-jnp.array([tx_nm, ty_nm])*1e-9
        log_amplitude = view.log_transmission(query.reshape(-1, 2)).reshape(points.shape[:-1])
        waist = jnp.exp(log_waist)
        incident = 2/(jnp.pi*waist**2)*jnp.exp(-2*jnp.sum((points-source)**2, axis=-1)/waist**2)
        intensity = incident*jnp.exp(2*log_amplitude*jnp.exp(log_contrast))
        halo = (intensity@weights).reshape(halo_shape)*jnp.exp(log_dose-2*log_m)
        return apply_detector_psf(halo, detector)

    return predict, float(phase_bound)


def _rms(error, sigma):
    return float(np.sqrt(np.mean((np.asarray(error)/sigma)**2)))


def audit(output):
    root = Path(__file__).resolve().parents[2]
    output = Path(output)
    recorded_path = root/'examples/generated/neoarm_inversion/imaging_recipe/measured_specimen.json'
    recorded = json.loads(recorded_path.read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for path, expected in recorded['source_hashes'].items():
        if digest(root/path) != expected:
            raise ValueError(f'Stale frozen-specimen source: {path}')
    paths = [Path(__file__), recorded_path, Path(__file__).with_name('neoarm_mixed_gaussian_validation.py'),
             root/'src/temgym_core/gaussian_image_likelihood.py',
             *(root/p for p in recorded['source_hashes'])]
    hashes = {str(p.relative_to(root)): digest(p) for p in paths}
    config = tomllib.loads((root/'examples/lens_inversion/neoarm_inversion_config.toml').read_text())
    geometry = geometry_anchors_from_notebook(root/'examples/microscope_models/measure_microscope_geometry.ipynb')
    atlas = MAGICALSampleAtlas.from_config(config,
        region_shape_yx=tuple(config['sample']['region_shape_yx']),
        authoritative_multilayer_shape_yx=tuple(config['sample']['authoritative_multilayer_shape_yx']),
        global_support_margin_m=config['sample']['atlas_support_margin_m'])
    rows = []
    output.parent.mkdir(parents=True, exist_ok=True)
    for state in recorded['rows']:
        source, waist = np.array(state['source_centre_xy_m']), state['source_waist_m']
        final = state['refinements'][-1]
        detector = DetectorGeometry(shape_yx=tuple(config['detector']['shape_yx']),
            pixel_size_yx_m=tuple(config['detector']['pixel_size_yx_m']),
            mtf_sigma_px=config['detector']['mtf_sigma_px'], z_m=geometry['detector_z_m'])
        # z only labels the exact transfer endpoint; no extra axial path is added.
        matrix = np.array(state['transfer_matrix'])
        view, _, _ = frozen_measured_view(atlas, matrix, detector, source, 2.5*waist, 0.)
        if asdict(view.selection) != state['selection']:
            # JSON round-trips tuples into lists; compare canonical JSON values.
            if json.loads(json.dumps(asdict(view.selection))) != state['selection']:
                raise ValueError('The frozen cubic selection changed.')
        axes = [centre+np.linspace(-2.5*waist, 2.5*waist, final['grid_shape_yx'][0]) for centre in source]
        for axis, key in zip(axes, ('grid_x_sha256', 'grid_y_sha256')):
            if hashlib.sha256(axis.tobytes()).hexdigest() != final[key]:
                raise ValueError('The frozen coherent topology changed.')
        halo_radius = int(np.ceil(4*detector.mtf_sigma_px))
        halo_detector = DetectorGeometry(shape_yx=tuple(n+2*halo_radius for n in detector.shape_yx),
            pixel_size_yx_m=detector.pixel_size_yx_m, affine_xy=detector.affine_xy,
            centre_xy_m=detector.centre_xy_m, z_m=detector.z_m)
        halo_world = detector_footprint(matrix, halo_detector)
        halo_clearance = require_view_coverage(view, halo_world, label='physical detector/MTF pixel edges')
        started = time.perf_counter()
        xx, yy = jnp.meshgrid(*map(jnp.asarray, axes), indexing='xy')
        incident = np.sqrt(2/np.pi)/waist*jnp.exp(-((xx-source[0])**2+(yy-source[1])**2)/waist**2)
        beam, _ = reexpand_transmitted_field(incident, view, *axes, voltage=config['beam']['voltage_v'], z_m=0.,
            width_over_step_yx=(1.25, 1.25), coefficient_method='gpu_kernel_required')
        beam = apply_transfer_matrix_to_gaussians(beam, matrix, z_out_m=detector.z_m, axial_path_m=0.)
        lattice = RegularGaussianLattice.from_beam(beam, final['grid_shape_yx'])
        print(f"{state['magnification']:g}x: {final['beam_count']:,} coherent beamlets, "
              f"{lattice.radius}-step neighbourhood; fresh GPU coefficients/propagation.", flush=True)
        coords, _, _, _ = detector_quadrature(detector, 8)
        sample_indices = np.random.default_rng(841).integers(0, coords.shape[0]*coords.shape[1], size=64)
        check_coords = jnp.asarray(coords.reshape(-1, 2)[sample_indices])
        check_grid = SimpleNamespace(coords=check_coords, shape=(64,))
        exhaustive_pallas = np.asarray(evaluate_gaussians(beam, check_grid, method='gpu_kernel_required'))
        # This is still GPU work. The all-beam JAX scan shares the elementary
        # field arithmetic but not neighbour selection with the accelerator.
        exhaustive = np.asarray(evaluate_gaussians(beam, check_grid, method='scan', batch_size=8192))
        accelerated = np.asarray(lattice(check_coords))
        wider = np.asarray(lattice(check_coords, radius=lattice.radius+2))
        field_error = float(np.linalg.norm(accelerated-exhaustive)/np.linalg.norm(exhaustive))
        pallas_field_error = float(np.linalg.norm(accelerated-exhaustive_pallas)/np.linalg.norm(exhaustive_pallas))
        pallas_intensity_error = float(np.linalg.norm(np.abs(accelerated)**2-np.abs(exhaustive_pallas)**2)
                                       /np.linalg.norm(np.abs(exhaustive_pallas)**2))
        count_scale = (recorded['incident_electrons']*np.prod(detector.pixel_size_yx_m)
                       /abs(np.linalg.det(detector.affine_xy)))
        local_point_counts = count_scale*np.abs(accelerated)**2
        backend_point_errors = [
            (count_scale*np.abs(reference)**2-local_point_counts)
            /np.sqrt(local_point_counts+recorded['read_noise_e']**2)
            for reference in (exhaustive, exhaustive_pallas)]
        backend_point_maximum_sigma = float(np.max(np.abs(backend_point_errors)))
        carrier = np.asarray(check_coords)
        carrier_phase = float(np.max(np.abs(.5*float(lattice.parameters[5][0])
            *np.einsum('ni,ij,nj->n', carrier, np.real(np.asarray(lattice.parameters[4][0])), carrier))))
        neighbour_error = float(np.linalg.norm(accelerated-wider)/np.linalg.norm(wider))
        counts, references, checks = {}, {}, []
        truth = np.array(state['truth_parameters'])
        for order in (1, 4, 8, 16):
            begin = time.perf_counter()
            counts[order] = np.asarray(integrated_gaussian_counts(lattice, detector,
                incident_electrons=recorded['incident_electrons'], quadrature_order=order))
            predict, phase_bound = make_finite_pixel_conjugate_predictor(view, detector, source,
                reference_matrix=matrix, wavelength_m=float(energy2wavelength(config['beam']['voltage_v'])),
                shortest_scale_m=state['selection']['selected_spacing_m'], quadrature_order=order)
            references[order] = np.asarray(predict(truth))
            sigma = np.sqrt(references[order]+recorded['read_noise_e']**2)
            difference = (counts[order]-references[order])/sigma
            checks.append(dict(quadrature_order=order, image_rms_sigma=float(np.sqrt(np.mean(difference**2))),
                whitened_image_l2=float(np.linalg.norm(difference)), elapsed_s=time.perf_counter()-begin))
            print(f"  pixel {order}x{order}: RMS={checks[-1]['image_rms_sigma']:.6g} sigma; "
                  f"L2={checks[-1]['whitened_image_l2']:.6g}; {checks[-1]['elapsed_s']:.2f}s", flush=True)
        sigma = np.sqrt(references[16]+recorded['read_noise_e']**2)
        previous_artifact = root/state['selected_artifact']
        if digest(previous_artifact) != state['selected_artifact_sha256']:
            raise ValueError('The archived centre-sampled comparison image changed.')
        with np.load(previous_artifact, allow_pickle=False) as data:
            previous_counts = data['expected_counts_yx']
        centre_fields = np.asarray(lattice(jnp.asarray(detector.optical_coords_xy_m.reshape(-1, 2))))
        legacy_counts = gaussian_filter(np.abs(centre_fields.reshape(detector.shape_yx))**2,
            detector.mtf_sigma_px, mode='nearest')*np.prod(detector.pixel_size_yx_m)*recorded['incident_electrons']
        transition = dict(gaussian_pixel8_vs16_rms_sigma=_rms(counts[8]-counts[16], sigma),
            gaussian_pixel8_vs16_whitened_l2=float(np.linalg.norm((counts[8]-counts[16])/sigma)),
            cubic_pixel8_vs16_rms_sigma=_rms(references[8]-references[16], sigma),
            cubic_pixel8_vs16_whitened_l2=float(np.linalg.norm((references[8]-references[16])/sigma)),
            legacy_reproduction_rms_sigma=_rms(legacy_counts-previous_counts, sigma),
            centre_halo_vs_legacy_mtf_rms_sigma=_rms(counts[1]-legacy_counts, sigma),
            finite_pixel_vs_centre_same_halo_rms_sigma=_rms(counts[16]-counts[1], sigma),
            finite_pixel_vs_legacy_total_rms_sigma=_rms(counts[16]-previous_counts, sigma))
        image_path = output.with_name(f"finite_pixel_specimen_{int(state['magnification'])}x.npz")
        np.savez_compressed(image_path, gaussian_counts_yx=counts[16], cubic_reference_counts_yx=references[16],
            gaussian_pixel8_counts_yx=counts[8], cubic_pixel8_counts_yx=references[8],
            centre_halo_counts_yx=counts[1], legacy_centre_counts_yx=previous_counts, noise_sigma_yx=sigma,
            raw_dac_values=np.array(state['controls']), transfer_matrix=matrix)
        strict_complex_agreement = bool(field_error < 1e-9 and pallas_field_error < 1e-9)
        passed = bool(np.isfinite(field_error) and neighbour_error < 1e-9
            and backend_point_maximum_sigma < .2 and checks[-1]['image_rms_sigma'] < .2
            and checks[-1]['whitened_image_l2'] < .2 and transition['gaussian_pixel8_vs16_whitened_l2'] < .1
            and transition['cubic_pixel8_vs16_whitened_l2'] < .1 and transition['legacy_reproduction_rms_sigma'] < .001)
        rows.append(dict(magnification=state['magnification'], passed=passed, beam_count=final['beam_count'],
            frozen_selection=state['selection'], frozen_topology_sha256=[final['grid_x_sha256'], final['grid_y_sha256']],
            neighbour_validation=lattice.validation, exhaustive_gpu_test_point_count=64,
            exhaustive_field_relative_l2=field_error, expanded_neighbourhood_field_relative_l2=neighbour_error,
            exhaustive_pallas_field_relative_l2=pallas_field_error,
            exhaustive_pallas_intensity_relative_l2=pallas_intensity_error,
            strict_complex_backend_agreement_passed=strict_complex_agreement,
            strict_complex_backend_relative_tolerance=1e-9,
            maximum_sampled_backend_point_count_error_sigma=backend_point_maximum_sigma,
            maximum_sampled_common_detector_carrier_phase_rad=carrier_phase,
            intensity_admission_passed=passed,
            minimum_cubic_halo_clearance_m=halo_clearance, phase_bound_rad=phase_bound,
            pixel_quadrature=checks, transitions=transition, total_elapsed_s=time.perf_counter()-started,
            artifact=str(image_path.relative_to(root) if image_path.is_absolute() else image_path),
            artifact_sha256=digest(image_path)))
    if any(digest(root/p) != expected for p, expected in hashes.items()):
        raise RuntimeError('An audited source changed during execution.')
    report = dict(schema='neoarm-finite-pixel-conjugate-specimen-v1', all_passed=all(r['passed'] for r in rows),
        admission_scope='Terminal-detector intensity only; no interchangeable downstream complex wavefront is certified.',
        all_strict_complex_backend_agreement_passed=all(r['strict_complex_backend_agreement_passed'] for r in rows),
        backend=jax.default_backend(), device_kind=jax.devices()[0].device_kind,
        source_hashes=hashes, source_hashes_unchanged_during_execution=True,
        incident_electrons=recorded['incident_electrons'], read_noise_e=recorded['read_noise_e'],
        assumptions=['Same frozen physical cubic views and coherent topologies as the measured-geometry image benchmark.',
            'Actual Gaussian complex fields, coefficients and beam-dependent endpoint actions are retained.',
            'The detector integral uses optical area weights, physical external illumination, normalized PSF and valid crop.',
            'Neighbour support is admitted by an envelope precondition plus independent exhaustive GPU point checks.',
            'Large common detector carrier phases produce separately reported backend complex-field rounding differences.',
            'The original strict 1e-9 complex agreement diagnostic is retained; intensity admission does not assert it passed.',
            'Pixel8-to16 convergence is tested; this is not a proof for arbitrary finer specimen features or defocused beams.',
            'Whitened image L2 bounds local linearized parameter bias in sigma; no new nonlinear lens or seven-parameter fit is claimed.',
            'The finite-pixel arrays are new results; legacy centre-sampled evidence is preserved unchanged.'], rows=rows)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(f'Finite-pixel specimen validation passed: {audit(args.output)["all_passed"]}')
