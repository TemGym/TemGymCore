"""Fresh GPU specimen images on the diagram-anchored low-excitation column.

This is a conditional conjugate-image benchmark, not a hardware DAC recipe.
The invariant cubic atlas is selected afresh for the actual 40k/80k matrices.
Coherent Gaussian re-expansion and an independent cubic intensity template
must agree before the seven effective image parameters are admitted. Detector
pixels remain centre-sampled here; the mixed-beam pixel-area proof does not
validate this different specimen renderer.
"""

from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
import tomllib

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jnp
import numpy as np

from neoarm_conjugate_sample_metrology import (
    make_conjugate_count_predictor, fit_summary, PARAMETER_NAMES, PARAMETER_SCALES,
)
from neoarm_imaging_recipe import column_with_geometry_anchors
from neoarm_measured_initialization import initialize_measured_imaging
from neoarm_workflow import expected_detector_counts
from temgym_core.constants import energy2wavelength
from temgym_core.jax_least_squares import least_squares
from temgym_core.multiscale_sample import MAGICALSampleAtlas, select_sample_view
from temgym_core.neoarm_calibration import (
    DetectorGeometry, hard_edge_column_from_config, reexpand_transmitted_field,
    apply_transfer_matrix_to_gaussians, render_gaussian_detector_plane,
    propagate_gaussians_through_column,
)

POSITION = [0, 2]


def detector_footprint(matrix, detector):
    """Actual pixel-edge footprint in specimen coordinates, including affine."""
    matrix = np.asarray(matrix, float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("Expected a finite 4x4 transfer matrix.")
    ny, nx = detector.shape_yx
    py, px = detector.pixel_size_yx_m
    corners = np.array([[-nx*px/2, -ny*py/2], [-nx*px/2, ny*py/2],
                        [nx*px/2, -ny*py/2], [nx*px/2, ny*py/2]])
    optical = corners@np.linalg.inv(detector.affine_xy).T + detector.centre_xy_m
    return optical@np.linalg.inv(matrix[np.ix_(POSITION, POSITION)]).T


def require_view_coverage(view, world_points, *, label):
    """A crop never silently treats unrepresented specimen pixels as vacuum."""
    local = np.asarray(view.world_to_local(jnp.asarray(world_points)))
    xmin, xmax, ymin, ymax = view.domain_bounds_xyxy_m
    slack = np.stack((local[:, 0]-xmin, xmax-local[:, 0],
                      local[:, 1]-ymin, ymax-local[:, 1]), axis=-1)
    if not np.all(np.isfinite(local)) or np.min(slack) < 0:
        raise ValueError(f"Frozen cubic view does not cover {label}.")
    return float(np.min(slack))


def frozen_measured_view(atlas, matrix, detector, source_centre, half_grid_extent, z_m):
    """Select one actual-matrix crop covering detector and fixed beam support."""
    if not np.allclose(detector.centre_xy_m, 0., atol=0., rtol=0.):
        raise ValueError("This acquisition currently requires an on-axis detector centre.")
    jacobian = detector.affine_xy@np.asarray(matrix)[np.ix_(POSITION, POSITION)]
    kwargs = dict(transfer_jacobian_xy=jacobian, detector_shape_yx=detector.shape_yx,
                  detector_pixel_size_yx_m=detector.pixel_size_yx_m,
                  sample_centre_xy_m=(0., 0.), z=z_m)
    preliminary = select_sample_view(atlas, **kwargs)
    centre = np.asarray(source_centre)
    grid_corners = centre+half_grid_extent*np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    local_grid = np.asarray(preliminary.world_to_local(jnp.asarray(grid_corners)))
    xmin, xmax, ymin, ymax = preliminary.selection.local_bounds_xyxy_m
    margin = max(0., xmin-local_grid[:, 0].min(), local_grid[:, 0].max()-xmax,
                 ymin-local_grid[:, 1].min(), local_grid[:, 1].max()-ymax)
    # Interpolation support is separate from detector and Gaussian support.
    halo_margin = 8*preliminary.selection.selected_spacing_m
    view = select_sample_view(atlas, gaussian_support_margin_m=margin,
                              interpolation_halo_margin_m=halo_margin, **kwargs)
    if not view.selection.meets_resolution:
        raise ValueError("The frozen atlas level cannot cover/resolve this measured-matrix acquisition.")
    footprint = detector_footprint(matrix, detector)
    margins = {
        'detector_edges_minimum_local_clearance_m': require_view_coverage(view, footprint, label='detector pixel edges'),
        'gaussian_grid_minimum_local_clearance_m': require_view_coverage(view, grid_corners, label='Gaussian grid'),
        'full_fitting_parameter_box_certified': False,
    }
    return view, footprint, margins


def render_state(view, matrix, detector, *, source_centre, waist_m, half_grid_extent,
                 side, voltage_v, z_m, slice_reference=None):
    """Fresh Gaussian coefficients, exact hard-edge transfer, GPU image field."""
    if jax.default_backend() != 'gpu':
        raise RuntimeError('Fresh specimen forward simulation requires the GPU.')
    axes = [centre+np.linspace(-half_grid_extent, half_grid_extent, side) for centre in source_centre]
    corners = np.array([[axes[0][i], axes[1][j]] for i in (0, -1) for j in (0, -1)])
    require_view_coverage(view, corners, label='concrete Gaussian grid')
    print(f'GPU {side**2:,} coherent beamlets; primary arrays ~{side**2*160/2**20:.0f} MiB; '
          f'{side**2*np.prod(detector.shape_yx):,} beamlet/pixel pairs.', flush=True)
    started = time.perf_counter()
    xx, yy = jnp.meshgrid(*map(jnp.asarray, axes), indexing='xy')
    incident = np.sqrt(2/np.pi)/waist_m*jnp.exp(-((xx-source_centre[0])**2+(yy-source_centre[1])**2)/waist_m**2)
    beam, transmitted = reexpand_transmitted_field(
        incident, view, *axes, voltage=voltage_v, z_m=z_m,
        width_over_step_yx=(1.25, 1.25), coefficient_method='gpu_kernel_required')
    maximum_q_difference = float(jnp.max(jnp.abs(beam.Q_inv-beam.Q_inv[0])))
    if maximum_q_difference != 0.:
        raise ValueError('The direct-map common metaplectic phase argument requires identical Gaussian Q matrices.')
    # The full matrix is computed from the actual finite hard-edge column.
    # Remove only common axial phase, retaining every centre-dependent action.
    downstream = apply_transfer_matrix_to_gaussians(beam, matrix, z_out_m=detector.z_m, axial_path_m=0.)
    rendered = render_gaussian_detector_plane(downstream, detector, tile_rows=16, method='gpu_kernel_required')
    intensity = np.asarray(rendered.intensity_yx)
    elapsed = time.perf_counter()-started
    power = float(jnp.sum(jnp.abs(transmitted)**2)*(axes[0][1]-axes[0][0])*(axes[1][1]-axes[1][0]))
    slice_check = None
    if slice_reference is not None:
        column, parameters, controls, dose, read_noise = slice_reference
        comparison_started = time.perf_counter()
        sliced = propagate_gaussians_through_column(beam, column, parameters, controls,
            segment='sample_to_detector', slices_per_lens=64)
        sliced_render = render_gaussian_detector_plane(sliced, detector, tile_rows=16, method='gpu_kernel_required')
        direct_field, sliced_field = rendered.field_yx, sliced_render.field_yx
        phase = np.angle(np.vdot(sliced_field, direct_field))
        field_error = np.linalg.norm(sliced_field*np.exp(1j*phase)-direct_field)/np.linalg.norm(direct_field)
        direct_counts = expected_detector_counts(intensity, detector, dose)
        sliced_counts = expected_detector_counts(sliced_render.intensity_yx, detector, dose)
        image_error = np.sqrt(np.mean(((sliced_counts-direct_counts)/np.sqrt(direct_counts+read_noise**2))**2))
        slice_check = {'slices_per_lens': 64, 'global_field_phase_alignment_rad': float(phase),
            'phase_aligned_field_relative_l2_error': float(field_error),
            'intensity_rms_sigma': float(image_error), 'passed_image_gate': bool(image_error < .2),
            'elapsed_s': time.perf_counter()-comparison_started,
            'scope': 'Single new-state comparison, not two successive slice refinements.'}
    return intensity, {
        'beam_count': side**2, 'grid_shape_yx': [side, side],
        'grid_bounds_xyxy_m': [axes[0][0], axes[0][-1], axes[1][0], axes[1][-1]],
        'grid_spacing_m': axes[0][1]-axes[0][0], 'elapsed_s': elapsed,
        'detector_render_elapsed_s': rendered.elapsed_s, 'transmitted_grid_power': power,
        'grid_x_sha256': hashlib.sha256(axes[0].tobytes()).hexdigest(),
        'grid_y_sha256': hashlib.sha256(axes[1].tobytes()).hexdigest(),
        'maximum_input_q_matrix_difference': maximum_q_difference,
        'direct_vs_64_slice_comparison': slice_check,
    }


def audit(output):
    output = Path(output)
    root = Path(__file__).resolve().parents[2]
    configuration = root/'examples/lens_inversion/neoarm_inversion_config.toml'
    geometry = root/'examples/microscope_models/measure_microscope_geometry.ipynb'
    paths = [Path(__file__), configuration, geometry,
             *(Path(__file__).with_name(name) for name in ('neoarm_conjugate_sample_metrology.py',
                 'neoarm_measured_initialization.py', 'neoarm_imaging_recipe.py', 'neoarm_workflow.py')),
             *(root/'src/temgym_core'/name for name in ('multiscale_sample.py', 'neoarm_calibration.py',
                 'microscope_inversion.py', 'gaussian.py', 'evaluate.py', 'periodic_metrology.py',
                 'constants.py', 'jax_least_squares.py'))]
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    sources = {str(path.relative_to(root)): digest(path) for path in paths}
    config = tomllib.loads(configuration.read_text())
    column = column_with_geometry_anchors(hard_edge_column_from_config(config), geometry)
    initialization = initialize_measured_imaging(column)
    column = initialization.column
    detector = DetectorGeometry(shape_yx=tuple(config['detector']['shape_yx']),
        pixel_size_yx_m=tuple(config['detector']['pixel_size_yx_m']),
        mtf_sigma_px=config['detector']['mtf_sigma_px'], z_m=column.detector_z_m)
    atlas = MAGICALSampleAtlas.from_config(config,
        region_shape_yx=tuple(config['sample']['region_shape_yx']),
        authoritative_multilayer_shape_yx=tuple(config['sample']['authoritative_multilayer_shape_yx']),
        global_support_margin_m=config['sample']['atlas_support_margin_m'])
    source_centre = np.asarray(atlas.pose_translation_xy_m)
    voltage, dose, read_noise = config['beam']['voltage_v'], 2e7, .6
    # Independently declared sample-plane Gaussian illumination. These are
    # the same pilot widths, not covariances inferred from the old condenser.
    waists = (1.1002091859399691e-7, 5.276761604902715e-8)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (magnification, controls, matrix, waist) in enumerate(zip(
            initialization.magnifications, initialization.nominal_controls, initialization.nominal_matrices, waists)):
        extent = 2.5*waist
        view, footprint, coverage = frozen_measured_view(atlas, matrix, detector, source_centre, extent, column.sample_z_m)
        predict, phase_bound = make_conjugate_count_predictor(view, detector, source_centre,
            reference_matrix=matrix, wavelength_m=float(energy2wavelength(voltage)),
            shortest_scale_m=view.selection.selected_spacing_m)
        a = matrix[np.ix_(POSITION, POSITION)]
        truth = np.array([np.log(np.sqrt(np.linalg.det(a))), np.arctan2(a[1, 0], a[0, 0]),
                          *(np.asarray(view.centre_xy_m)/1e-9), np.log(dose), np.log(waist), 0.])
        initial = truth+np.array([np.log(1.03), .02, 2., -1., np.log(.97), np.log(1.04), np.log(.9)])
        half_width = np.array([np.log(1.2), .3, 20., 20., np.log(2.), np.log(1.5), np.log(2.)])
        reference = np.asarray(predict(truth))
        refinements, previous, previous_fit, consecutive = [], None, None, 0
        sides = (1621, 2293, 3241) if index == 0 else (779, 1103, 1557)
        for side in sides:
            intensity, record = render_state(view, matrix, detector, source_centre=source_centre,
                waist_m=waist, half_grid_extent=extent, side=side, voltage_v=voltage, z_m=column.sample_z_m,
                slice_reference=((column, initialization.parameters, controls, dose, read_noise)
                                 if index == 1 and side == sides[0] else None))
            expected = expected_detector_counts(intensity, detector, dose)
            variance = expected+read_noise**2
            noise_seed = 62040+40*index
            rng = np.random.default_rng(noise_seed)
            noisy = rng.poisson(expected)+rng.normal(0., read_noise, expected.shape)
            for name, observed in (('noiseless', expected), ('noisy', noisy)):
                observed_jax, sigma_jax = jnp.asarray(observed), jnp.sqrt(jnp.asarray(variance))
                fit = least_squares(lambda p: ((predict(p)-observed_jax)/sigma_jax).ravel(), initial,
                    bounds=(truth-half_width, truth+half_width), x_scale=PARAMETER_SCALES,
                    max_nfev=300, gtol=1e-6, ftol=1e-11, xtol=1e-10)
                record[name] = fit_summary(fit, truth, variance, expected, predict)
            record['truth_image_rms_sigma'] = float(np.sqrt(np.mean(((expected-reference)/np.sqrt(variance))**2)))
            current_fit = record['noiseless']
            record['maximum_parameter_bias_sigma'] = float(np.max(np.abs(current_fit['parameter_errors_sigma'])))
            record['parameter_transition_sigma'] = (None if previous_fit is None else
                ((np.array(current_fit['parameters'])-previous_fit['parameters'])/current_fit['parameter_standard_errors']).tolist())
            record['image_transition_rms_sigma'] = (None if previous is None else
                float(np.sqrt(np.mean(((expected-previous)/np.sqrt(variance))**2))))
            transition_passed = (previous_fit is not None and record['image_transition_rms_sigma'] < .2
                                and np.max(np.abs(record['parameter_transition_sigma'])) < .1)
            consecutive = consecutive+1 if transition_passed else 0
            record['successive_converged_transitions'] = consecutive
            record['passed_conditional_image_audit'] = bool(record['truth_image_rms_sigma'] < .2
                and record['maximum_parameter_bias_sigma'] < .2 and current_fit['jacobian_rank'] == 7
                and current_fit['optimizer_success'] and record['noisy']['optimizer_success']
                and .8 < record['noisy']['data_residual_rms_sigma'] < 1.2)
            arrays = dict(expected_counts_yx=expected, noisy_counts_yx=noisy,
                fitted_counts_yx=np.asarray(predict(jnp.asarray(record['noisy']['parameters']))),
                noise_sigma_yx=np.sqrt(variance), template_truth_counts_yx=reference,
                transfer_matrix=matrix, raw_dac_values=controls, optical_parameters=initialization.parameters,
                footprint_specimen_xy_m=footprint)
            image_path = output.with_name(f'measured_specimen_{int(magnification)}x_beamcount_{side**2}.npz')
            np.savez_compressed(image_path, **arrays)
            record['artifact'] = str(image_path.relative_to(root) if image_path.is_absolute() else image_path)
            record['artifact_sha256'] = digest(image_path)
            record['array_sha256'] = {key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                                      for key, value in arrays.items()}
            refinements.append(record)
            previous, previous_fit = expected, current_fit
            print(f'{magnification:g}x: truth {record["truth_image_rms_sigma"]:.6g} sigma; '
                  f'max seven-parameter bias {record["maximum_parameter_bias_sigma"]:.6g} sigma; '
                  f'converged transitions {consecutive}.', flush=True)
        rows.append(dict(magnification=magnification, rotation_rad=truth[1], source_waist_m=waist,
            source_centre_xy_m=source_centre.tolist(), phase_bound_rad_at_shortest_scale=phase_bound,
            controls=controls.tolist(), optical_parameters=initialization.parameters.tolist(),
            transfer_matrix=matrix.tolist(), selection=asdict(view.selection),
            sample_view_bounds_xyxy_m=view.domain_bounds_xyxy_m, coverage=coverage,
            truth_parameters=truth.tolist(), initial_parameters=initial.tolist(), noise_seed=noise_seed,
            refinements=refinements, selected_artifact=refinements[-1]['artifact'],
            selected_artifact_sha256=refinements[-1]['artifact_sha256'],
            passed=bool(consecutive >= 2 and refinements[-1]['passed_conditional_image_audit'])))
    if any(digest(root/path) != value for path, value in sources.items()):
        raise RuntimeError('A source changed during the measured-specimen audit.')
    report = dict(schema='neoarm-measured-geometry-specimen-v1', all_passed=all(row['passed'] for row in rows),
        backend=jax.default_backend(), device=str(jax.devices()[0]), device_kind=jax.devices()[0].device_kind,
        jax_version=jax.__version__, x64_enabled=jax.config.x64_enabled, source_hashes=sources,
        source_hashes_unchanged_during_execution=True, parameter_names=PARAMETER_NAMES,
        incident_electrons=dose, read_noise_e=read_noise, specimen_identity=atlas.certificate.version,
        objective_fixed_between_states=True, fixed_objective_biases={lens.name: lens.q_bias for lens in column.lenses
            if lens.name in ('Obj-prefield', 'Obj-post')},
        assumptions=['Geometry anchors come from the microscope diagram; thicknesses and DAC laws remain provisional.',
            'All images are freshly re-expanded and propagated; no archived image is rotated or rescaled.',
            'Same frozen physical cubic atlas; calibrated pure sample-plane Gaussian source, not condenser recovery.',
            'Exact full hard-edge transfer map with common axial phase removed; beam-dependent actions retained.',
            'Independent reference uses conjugate cubic intensity, conditioned on negligible B at the retained scale.',
            'Detector intensity is centre-sampled, multiplied by optical pixel area, then legacy finite-grid MTF.',
            'Pixel-area integration, MTF halo correctness for specimen images and full fitting-box view coverage remain unproved.',
            'Seven effective image parameters are fitted, not physical lens parameters or hardware DAC calibration.'], rows=rows)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(f'Measured-geometry conditional specimen audit passed: {audit(args.output)["all_passed"]}')
