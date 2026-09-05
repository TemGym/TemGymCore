"""Ideal general-source rank through two actual conditional analyser maps.

This is a small-matrix/capture preflight, not detector-image generation.  The
same27 Cartesian controls and unchanged source/Q hypotheses are retained.
"""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from neoarm_condenser_general_source_contract import (
    GeneralSourceStigmator, initial_parameters, information, ROOT,
)
from neoarm_condenser_extension_design import load_context, FOLDER
from temgym_core.microscope_inversion import drift_matrix

GROUPED = np.array([0, 2, 1, 3])
PIXEL_M = 55e-6


def compose_analyser_maps(conditional_model, reference_z_m):
    """Canonical grouped reference state to grouped field-free detector state.

    Conditional prefield converts canonical objective entrance to mechanical
    sample coordinates; downstream maps consume that mechanical sample state.
    No extra sample magnetic shear or independent rotation tag is appended.
    """
    native = conditional_model.downstream@conditional_model.prefield@drift_matrix(
        conditional_model.objective_entrance_m-reference_z_m)
    return native[:, GROUPED[:, None], GROUPED[None, :]]


class ActualAnalyserStigmator(GeneralSourceStigmator):
    def __init__(self, location, conditional_model):
        super().__init__(location)
        self.analyser_maps = compose_analyser_maps(conditional_model, self.reference_z_m)

    def detector_maps(self, parameters):
        before = jax.vmap(lambda u: self.maps(parameters, u)[0])(self.controls)
        return self.analyser_maps[None]@before[:, None]

    def detector_covariance_pixels(self, parameters):
        m = self.detector_maps(parameters)
        covariance = self.decode(parameters)[1]
        return (m@covariance@jnp.swapaxes(m, -1, -2))[:, :, :2, :2]/PIXEL_M**2

    def observables(self, parameters):
        c = self.detector_covariance_pixels(parameters)
        sigma = jnp.sqrt(jnp.diagonal(c, axis1=-2, axis2=-1))
        return jnp.concatenate((jnp.log(sigma),
            (c[:, :, 0, 1]/jnp.prod(sigma, axis=-1))[:, :, None]), axis=-1).reshape(-1)


def gaussian_rectangle_capture(covariance_pixels, *, half_width_pixels=128., order=32):
    """Centered unblurred Gaussian probability on a square; no image array.

    This quadrature is used only for the extremely broad preflight beams and
    independently refined. It is not a replacement finite-pixel renderer.
    """
    if order < 2 or half_width_pixels <= 0:
        raise ValueError('Positive square half-width and order>=2 are required.')
    nodes, weights = np.polynomial.legendre.leggauss(order)
    x, y = np.meshgrid(nodes*half_width_pixels, nodes*half_width_pixels, indexing='xy')
    points = jnp.asarray(np.stack((x, y), axis=-1).reshape(-1, 2))
    weight = jnp.asarray(np.outer(weights, weights).reshape(-1)*half_width_pixels**2)
    c = jnp.asarray(covariance_pixels)
    exponent = jnp.einsum('ni,...ij,nj->...n', points, jnp.linalg.inv(c), points)
    return jnp.sum(jnp.exp(-.5*exponent)*weight, axis=-1)/(2*jnp.pi*jnp.sqrt(jnp.linalg.det(c)))


def audit(output):
    if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
        raise RuntimeError('Use GPU float64 for the optical/capture preflight.')
    old, _, repair, aperture, conditional = load_context()
    if np.any(np.asarray(conditional.reference_controls) != 0):
        raise ValueError('The retained Cartesian control convention requires zero reference DACs.')
    p = initial_parameters(); rows = []
    for location in ('after_cl1', 'after_cl3'):
        model = ActualAnalyserStigmator(location, conditional)
        covariance = np.asarray(jax.jit(model.detector_covariance_pixels)(p))
        jacobian = np.asarray(jax.jit(jax.jacfwd(model.observables))(p))
        rank = information(jacobian)
        width = np.sqrt(np.linalg.eigvalsh(covariance))
        captures = []
        for order in (16, 32):
            captures.append(np.asarray(jax.jit(lambda c: gaussian_rectangle_capture(c, order=order))(covariance)))
        delta = float(np.max(abs(captures[1]-captures[0])))
        if delta > 1e-10:
            raise RuntimeError('The broad-beam detector-probability estimate failed refinement.')
        admitted = (width[:, :, 0] >= .4)&(captures[-1] >= .999)
        valid_info = (information(jacobian[np.repeat(admitted.reshape(-1), 3)])
                      if np.any(admitted) else None)
        # Native full-column closure with Q disabled tests canonical/mechanical
        # ordering without silently adding a second objective magnetic shear.
        zero = p.copy(); zero[23:] = 0.
        controls = np.zeros((len(model.controls), 7)); controls[:, :3] = model.controls
        native_controls = controls[:, None]+np.asarray(conditional.analyser_controls)[None]
        parameters = np.asarray(conditional.fixed).copy()
        parameters[2:14] = np.asarray(model.decode(zero)[0]).reshape(-1)
        native = np.asarray(jax.jit(jax.vmap(lambda u: conditional.column.matrix(
            jnp.asarray(parameters), u, coordinates='canonical')))(native_controls.reshape(-1, 7)))
        grouped = native[:, GROUPED[:, None], GROUPED[None, :]].reshape(len(controls), 2, 4, 4)
        staged = np.asarray(jax.jit(model.detector_maps)(zero))
        closure = float(np.max(np.linalg.norm(staged-grouped, axis=(-2, -1))/np.linalg.norm(grouped, axis=(-2, -1))))
        if closure > 1e-9:
            raise RuntimeError('Native full-column coordinate closure failed.')
        rows.append(dict(location=location, parameters=p.tolist(), controls=np.asarray(model.controls).tolist(),
            source_covariance=np.asarray(model.decode(p)[1]).tolist(), quadrupole_position_m=float(model.decode(p)[2]),
            quadrupole_strengths_inverse_m=(30*p[23:]).tolist(),
            grouped_reference_to_detector_maps=np.asarray(model.analyser_maps).tolist(),
            information=rank, optical_covariance_pixels=covariance.tolist(),
            principal_optical_rms_pixels=width.tolist(), detector_capture_estimate=captures[-1].tolist(),
            detector_capture_refinement_maximum_absolute_change=delta,
            minimum_optical_rms_pixels=float(width.min()), maximum_optical_rms_pixels=float(width.max()),
            minimum_capture=float(captures[-1].min()), maximum_capture=float(captures[-1].max()),
            admitted_frame_mask=admitted.tolist(), admitted_frame_count=int(admitted.sum()),
            admitted_complete_pair_count=int(admitted.all(axis=1).sum()), admitted_information=valid_info,
            zero_Q_native_full_column_relative_matrix_closure=closure))
    previous = FOLDER/'condenser_general_source_contract.json'
    general = json.loads(previous.read_text())
    sources = dict(general['source_hashes'])
    for report in (old, repair, aperture):
        for key in ('source_hashes', 'computational_input_hashes', 'report_input_hashes'):
            sources.update(report.get(key, {}))
    for path in (Path(__file__), ROOT/'tests/test_neoarm_condenser_actual_analyser_contract.py', previous,
                 ROOT/'examples/lens_inversion/neoarm_condenser_extension_design.py'):
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path, checksum in sources.items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != checksum:
            raise RuntimeError(f'A bound source changed: {path}')
    report = dict(schema='neoarm-general-source-actual-two-analyser-contract-v1',
        backend=jax.default_backend(), x64_enabled=True, source_hashes=sources,
        actual_conditional_analyser_controls=np.asarray(conditional.analyser_controls).tolist(),
        objective_entrance_m=conditional.objective_entrance_m, source_reference_m=conditional.source_z_m,
        detector_shape_yx=[256, 256], detector_pixel_m=PIXEL_M, minimum_rms_gate_pixels=.4, minimum_capture_gate=.999,
        capture_estimate='Centered continuous Gaussian probability over256x256 sensor;16→32 tensor Gauss-Legendre refinement; no crop normalization. Pixel MTF omitted for this broad-beam feasibility estimate.',
        covariance_observables='Per state and analyser: log RMS_x, log RMS_y, correlation.162 ideal scalar observations, not measured image summaries.',
        cases=rows, unchanged_general_source_and_Q_hypotheses=True,
        objective_and_downstream_maps_conditional=True, source_or_Q_assumed_known_for_information=False,
        all_ten_source_moments_and_three_Q_coordinates_profiled=True,
        aperture_collector_data_used=False, independent_kicks_or_current_readback_used=False,
        new_images_generated=False, recovery_performed=False, hardware_feasibility_proved=False,
        limitations=[
            'Two specific conditional analyser maps replace the ideal three-projection ceiling. Their calibration uncertainty is not propagated.',
            'The displayed Cartesian controls are retained without condenser refocusing; these broad beams fail capture and cannot justify a beam-only experiment.',
            'Formal local rank without camera admission is not usable information, precision, an accepted image likelihood or global uniqueness.',
            'The source and nonzero internal Q are declared synthetic hypotheses, not instrument truth or calibrated source/Q inputs. All13 nuisance coordinates remain free in the rank audit.',
            'No pixel-integrated images or noisy covariances are generated. MTF and clipping would have to enter a future admitted image likelihood.',
            'A beam-only route needs a new refocused connected excitation design, followed by finite-pixel information, source stability, noisy recovery and branch checks.',
            'An aperture collector is not mathematically required by this generic ideal local rank, but its practical replacement has not been demonstrated.'])
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); jax.config.update('jax_enable_x64', True)
    result = audit(args.output)
    print([(row['location'], row['information']['full_rank'], row['maximum_capture']) for row in result['cases']], flush=True)
