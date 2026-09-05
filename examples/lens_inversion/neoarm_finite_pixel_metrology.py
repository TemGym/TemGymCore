"""Seven-parameter calibration of fresh finite-pixel coherent specimen images.

This consumes the independently validated Gaussian pixel16/physical-halo
images. The cubic image likelihood uses pixel8 quadrature, with pixel16 checks
at the fitted parameters. It is conditional image metrology, not a physical
lens fit, and never changes the older centre-sampled checkpoints.
"""

from pathlib import Path
import argparse
import hashlib
import json
import os
import tomllib

os.environ.setdefault('JAX_ENABLE_X64', 'true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import jax
import jax.numpy as jnp
import numpy as np

from neoarm_finite_pixel_specimen import make_finite_pixel_conjugate_predictor
from neoarm_measured_specimen import frozen_measured_view
from neoarm_imaging_recipe import geometry_anchors_from_notebook
from neoarm_conjugate_sample_metrology import fit_summary, PARAMETER_NAMES, PARAMETER_SCALES
from temgym_core.constants import energy2wavelength
from temgym_core.jax_least_squares import least_squares
from temgym_core.multiscale_sample import MAGICALSampleAtlas
from temgym_core.neoarm_calibration import DetectorGeometry


def audit(output):
    if jax.default_backend() != 'gpu':
        raise RuntimeError('Finite-pixel specimen metrology requires the GPU; no CPU fallback.')
    root = Path(__file__).resolve().parents[2]
    output = Path(output)
    input_report = root/'examples/generated/neoarm_inversion/imaging_recipe/finite_pixel_specimen.json'
    finite = json.loads(input_report.read_text())
    if not finite['all_passed'] or not finite['admission_scope'].startswith('Terminal-detector intensity only'):
        raise ValueError('The finite-pixel terminal-intensity audit must pass first.')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for path, value in finite['source_hashes'].items():
        if digest(root/path) != value:
            raise ValueError(f'Stale finite-pixel source: {path}')
    sources = {str(Path(__file__).relative_to(root)): digest(Path(__file__)),
               str(input_report.relative_to(root)): digest(input_report), **finite['source_hashes']}
    measured = json.loads((root/'examples/generated/neoarm_inversion/imaging_recipe/measured_specimen.json').read_text())
    measured_by_mag = {row['magnification']: row for row in measured['rows']}
    config = tomllib.loads((root/'examples/lens_inversion/neoarm_inversion_config.toml').read_text())
    geometry = geometry_anchors_from_notebook(root/'examples/microscope_models/measure_microscope_geometry.ipynb')
    atlas = MAGICALSampleAtlas.from_config(config,
        region_shape_yx=tuple(config['sample']['region_shape_yx']),
        authoritative_multilayer_shape_yx=tuple(config['sample']['authoritative_multilayer_shape_yx']),
        global_support_margin_m=config['sample']['atlas_support_margin_m'])
    detector = DetectorGeometry(shape_yx=tuple(config['detector']['shape_yx']),
        pixel_size_yx_m=tuple(config['detector']['pixel_size_yx_m']),
        mtf_sigma_px=config['detector']['mtf_sigma_px'], z_m=geometry['detector_z_m'])
    rows = []
    output.parent.mkdir(parents=True, exist_ok=True)
    for index, finite_row in enumerate(finite['rows']):
        state = measured_by_mag[finite_row['magnification']]
        array_path = root/finite_row['artifact']
        if digest(array_path) != finite_row['artifact_sha256']:
            raise ValueError('The Gaussian finite-pixel image checksum changed.')
        with np.load(array_path, allow_pickle=False) as saved:
            expected = saved['gaussian_counts_yx']
            np.testing.assert_array_equal(saved['raw_dac_values'], state['controls'])
            np.testing.assert_array_equal(saved['transfer_matrix'], state['transfer_matrix'])
        matrix, source = np.asarray(state['transfer_matrix']), np.asarray(state['source_centre_xy_m'])
        view, _, _ = frozen_measured_view(atlas, matrix, detector, source, 2.5*state['source_waist_m'], 0.)
        predictors = {order: make_finite_pixel_conjugate_predictor(view, detector, source,
            reference_matrix=matrix, wavelength_m=float(energy2wavelength(config['beam']['voltage_v'])),
            shortest_scale_m=state['selection']['selected_spacing_m'], quadrature_order=order)[0]
            for order in (8, 16)}
        predict = predictors[8]
        truth, initial = np.array(state['truth_parameters']), np.array(state['initial_parameters'])
        half_width = np.array([np.log(1.2), .3, 20., 20., np.log(2.), np.log(1.5), np.log(2.)])
        variance = expected+finite['read_noise_e']**2
        noise_seed = 72040+40*index
        rng = np.random.default_rng(noise_seed)
        noisy = rng.poisson(expected)+rng.normal(0., finite['read_noise_e'], expected.shape)
        row = dict(magnification=state['magnification'], truth_parameters=truth.tolist(),
            initial_parameters=initial.tolist(), input_artifact=finite_row['artifact'],
            input_artifact_sha256=finite_row['artifact_sha256'], noise_seed=noise_seed,
            beam_count=finite_row['beam_count'], full_fitting_parameter_box_coverage_certified=False)
        for name, observed in (('noiseless', expected), ('noisy', noisy)):
            target, sigma = jnp.asarray(observed), jnp.sqrt(jnp.asarray(variance))
            fit = least_squares(lambda p: ((predict(p)-target)/sigma).ravel(), initial,
                bounds=(truth-half_width, truth+half_width), x_scale=PARAMETER_SCALES,
                max_nfev=300, gtol=1e-6, ftol=1e-11, xtol=1e-10)
            summary = fit_summary(fit, truth, variance, expected, predict)
            delta = ((np.asarray(predictors[16](fit.x))-np.asarray(predict(fit.x)))/np.sqrt(variance)).ravel()
            left, singular, right = np.linalg.svd(np.asarray(fit.jacobian)*PARAMETER_SCALES, full_matrices=False)
            shift = PARAMETER_SCALES*(right.T@((left.T@delta)/singular))
            summary['pixel8_to16_parameter_shift_sigma'] = (shift/summary['parameter_standard_errors']).tolist()
            summary['pixel8_to16_rms_sigma'] = float(np.sqrt(np.mean(delta**2)))
            row[name] = summary
            print(f"{state['magnification']:g}x {name}: residual {summary['data_residual_rms_sigma']:.6g} sigma; "
                  f"maximum parameter error {max(map(abs, summary['parameter_errors_sigma'])):.6g} sigma", flush=True)
        row['passed'] = bool(row['noiseless']['optimizer_success'] and row['noisy']['optimizer_success']
            and row['noiseless']['jacobian_rank'] == row['noisy']['jacobian_rank'] == 7
            and max(map(abs, row['noiseless']['parameter_errors_sigma'])) < .2
            and .8 < row['noisy']['data_residual_rms_sigma'] < 1.2
            and all(max(map(abs, row[name]['pixel8_to16_parameter_shift_sigma'])) < .2 for name in ('noiseless', 'noisy')))
        arrays = dict(expected_counts_yx=expected, noisy_counts_yx=noisy,
            fitted_counts_yx=np.asarray(predict(jnp.asarray(row['noisy']['parameters']))),
            noise_sigma_yx=np.sqrt(variance), template_truth_counts_yx=np.asarray(predict(truth)))
        comparison_path = output.with_name(f"finite_pixel_metrology_{int(state['magnification'])}x.npz")
        np.savez_compressed(comparison_path, **arrays)
        row['comparison_arrays'] = dict(path=str(comparison_path.relative_to(root) if comparison_path.is_absolute() else comparison_path),
            sha256=digest(comparison_path), array_sha256={name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                                                        for name, value in arrays.items()})
        rows.append(row)
    if any(digest(root/path) != value for path, value in sources.items()):
        raise RuntimeError('An audited source changed during finite-pixel metrology.')
    report = dict(schema='neoarm-finite-pixel-seven-parameter-metrology-v1', all_passed=all(row['passed'] for row in rows),
        backend=jax.default_backend(), device_kind=jax.devices()[0].device_kind, parameter_names=PARAMETER_NAMES,
        source_hashes=sources, source_hashes_unchanged_during_execution=True,
        incident_electrons=finite['incident_electrons'], read_noise_e=finite['read_noise_e'],
        assumptions=['Fresh independent Gaussian pixel16 data; seven-parameter cubic pixel8 likelihood with physical MTF halo.',
            'All seven image parameters begin perturbed and are released jointly.',
            'Detector geometry, certificate, specimen internal pose and input centre are exactly calibrated conditional inputs.',
            'Variance is the exact synthetic expectation plus read-noise variance; uncertainty in its calibration is not fitted.',
            'Pixel8-to16 parameter shifts are tested at both final fits; full fitting-box crop coverage is not certified.',
            'No physical lens recovery, equivalent-branch exclusion or interchangeable complex-field backend is claimed.'], rows=rows)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(f'Finite-pixel seven-parameter metrology passed: {audit(args.output)["all_passed"]}')
