"""Finite-camera aperture throughput: geometric joint acceptance and GPU waves.

The partially coherent Gaussian source is a positive mixture of transported
Hermite-Gaussian modes, not one coherent beam assigned an impossible covariance.
Every mode is multiplied by the physical circular aperture before a GPU Collins
integral. Camera clipping is never removed by renormalizing detected intensity.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from functools import partial
import hashlib
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr, ndtri, log_ndtr
from scipy.stats import qmc

from temgym_core.constants import energy2wavelength
from temgym_core.critical_line_calibration import aperture_edge_scan_from_moments
from temgym_core.gaussian_image_likelihood import validate_phase_space_covariance, make_gaussian_detector_operator
from temgym_core.neoarm_calibration import DetectorGeometry


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INFORMATION = ROOT/"examples/generated/neoarm_inversion/imaging_recipe/shared_objective_information_narrow_probe.json"
XY, ANGLE = np.array([0, 2]), np.array([1, 3])
HALF_CAMERA_M = 128*55e-6
PSF_SIGMA_M = .65*55e-6
RADII_M = np.array([10e-6, 20e-6, 30e-6])
CENTRES_M = np.array([[.5e-6, -.3e-6], [.4e-6, .2e-6], [-.2e-6, .4e-6]])


def camera_acceptance(position_xy, *, half_width_m=HALF_CAMERA_M, psf_sigma_m=PSF_SIGMA_M):
    """Probability that a Gaussian detector PSF falls inside the camera."""
    upper = ndtr((half_width_m-position_xy)/psf_sigma_m)
    lower = ndtr((-half_width_m-position_xy)/psf_sigma_m)
    return jnp.prod(upper-lower, axis=-1)


def czt_axis(field, *, input_start, input_step, output_start, output_step,
             output_count, wavelength_times_b, axis=-1):
    """Uniform physical Fourier quadrature using a JAX Bluestein transform.

    Computes dx*sum_j f(x_j) exp(-2pi i*x_j*X_k/(lambda*B)). This is a
    transform algorithm, not a Gaussian propagator or a fitted transfer map.
    """
    values = jnp.moveaxis(field, axis, -1)
    count = values.shape[-1]
    fft_count = 1 << (count+output_count-2).bit_length()
    j, k = jnp.arange(count), jnp.arange(output_count)
    alpha = 2*jnp.pi*input_step*output_step/wavelength_times_b
    premultiply = jnp.exp(-1j*(2*jnp.pi*input_step*output_start*j/wavelength_times_b + .5*alpha*j*j))
    padded = jnp.pad(values*premultiply, [(0, 0)]*(values.ndim-1)+[(0, fft_count-count)])
    index = jnp.arange(fft_count)
    difference = jnp.where(index < output_count, index, index-fft_count)
    valid = (index < output_count) | (index >= fft_count-count+1)
    chirp = jnp.where(valid, jnp.exp(.5j*alpha*difference*difference), 0.)
    transformed = jnp.fft.ifft(jnp.fft.fft(padded)*jnp.fft.fft(chirp))[..., :output_count]
    output_coords = output_start+k*output_step
    transformed *= input_step*jnp.exp(-1j*(.5*alpha*k*k + 2*jnp.pi*input_start*output_coords/wavelength_times_b))
    return jnp.moveaxis(transformed, -1, axis)


def thermal_source_modes(source_covariance, wavelength_m, *, modes_per_axis=6):
    """Pure squeezed covariance and exact thermal weights for a diagonal source.

    The geometrically decaying mode probabilities are NOT renormalized when
    truncated. The omitted probability is an explicit bound on any lost power.
    """
    covariance = np.asarray(source_covariance)
    validate_phase_space_covariance(covariance, wavelength_m)
    if not np.allclose(covariance, np.diag(np.diag(covariance)), rtol=0, atol=1e-30):
        raise ValueError("This reference currently requires a diagonal field-free source covariance.")
    epsilon = wavelength_m/(4*np.pi)
    emittance = np.sqrt(np.diag(covariance)[::2]*np.diag(covariance)[1::2])
    ratio = (emittance-epsilon)/(emittance+epsilon)
    pure = covariance*np.repeat(epsilon/emittance, 2)
    one_dimensional = (1-ratio[:, None])*ratio[:, None]**np.arange(modes_per_axis)
    weights = np.outer(*one_dimensional)
    return pure, weights, float(1-weights.sum())


def normalized_hermite_modes(coordinates, modes):
    """H_n(x)/(sqrt(2^n*n!)), evaluated by a stable three-term recurrence."""
    result = [jnp.ones_like(coordinates)]
    if modes > 1:
        result.append(jnp.sqrt(2.)*coordinates)
    for n in range(1, modes-1):
        result.append(jnp.sqrt(2/(n+1))*coordinates*result[-1]-jnp.sqrt(n/(n+1))*result[-2])
    return jnp.stack(result)


def circle_cell_fraction(x, y, step, radius, centre, *, subpixels=4):
    """Area quadrature of a sharp circular aperture; no softened specimen edge."""
    offsets = (jnp.arange(subpixels)+.5)/subpixels-.5
    fraction = jnp.zeros((len(y), len(x)))
    for a in range(subpixels):
        for b in range(subpixels):
            dx = x[None, :]+step*offsets[a]-centre[0]
            dy = y[:, None]+step*offsets[b]-centre[1]
            fraction += (dx*dx+dy*dy <= radius*radius)/subpixels**2
    return fraction


@dataclass(frozen=True)
class CaptureCase:
    label: str
    magnification: float
    controls: np.ndarray
    source_to_aperture: np.ndarray
    source_to_detector: np.ndarray

    @property
    def aperture_to_detector(self):
        return np.linalg.solve(self.source_to_aperture.T, self.source_to_detector.T).T


def load_cases(path=DEFAULT_INFORMATION):
    from neoarm_shared_objective import SharedObjectiveColumn, shared_imaging_metrics
    from neoarm_measured_information_audit import measured_column
    from temgym_core.immersed_objective import SharedObjectiveField
    report = json.loads(Path(path).read_text())
    for name, expected in report["source_hashes"].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Stale information-report dependency: {name}")
    p = np.asarray(report["truth_optical_parameters"])
    objective = SharedObjectiveField(tuple(report["field_edges_m"]), 0.)
    column = SharedObjectiveColumn.from_hard_edge(measured_column(), objective=objective,
                                                objective_initial=tuple(p[-4:]))
    projectors = np.unique(np.asarray(report["image_controls"])[:, 3:], axis=0)
    trial = np.column_stack([np.zeros((len(projectors), 3)), projectors])
    maps = jax.jit(jax.vmap(lambda u: column.matrix(p, u, segment="sample_to_detector")))(trial)
    metrics = np.asarray(jax.vmap(shared_imaging_metrics)(maps))
    nominal = []
    for target in (40000., 80000.):
        score = abs(np.log(metrics[:, 0]/target))+np.linalg.norm(metrics[:, 2:], axis=1)/1e-6
        index = int(np.argmin(score))
        if abs(metrics[index, 0]/target-1) > 1e-8 or np.linalg.norm(metrics[index, 2:]) > 1e-10:
            raise ValueError("The saved design has no required specimen-conjugate nominal state.")
        nominal.append(trial[index])
    illumination = list(np.asarray(report["illumination_controls"]))
    labels = [f"illumination_{i}" for i in range(len(illumination))]
    probe = report["focus_search"].get("calibration_probe")
    if probe:
        illumination.append(np.asarray(probe["controls"])); labels.append("calibration_probe")
    controls, case_labels, magnifications = [], [], []
    for label, upstream in zip(labels, illumination):
        for target, downstream in zip((40000., 80000.), nominal):
            u = downstream.copy(); u[:3] = upstream[:3]
            controls.append(u); case_labels.append(label); magnifications.append(target)
    aperture_z = -.42083333333333334
    at_aperture = np.asarray(jax.jit(jax.vmap(lambda u: column.matrix_to_plane(p, u, aperture_z)))(jnp.asarray(controls)))
    at_detector = np.asarray(jax.jit(jax.vmap(lambda u: column.matrix(p, u)))(jnp.asarray(controls)))
    cases = [CaptureCase(label, mag, u, a, d) for label, mag, u, a, d in
             zip(case_labels, magnifications, controls, at_aperture, at_detector)]
    return cases, np.asarray(report["source_covariance_m_rad"]), column, p


def geometric_capture(cases, source_covariance, *, sobol_power=18, repetitions=8):
    """Scrambled Sobol joint acceptance of aperture position and detector hit.

    This truncates a positive Gaussian Wigner distribution geometrically and
    therefore omits edge diffraction. Its discrepancy against the wave result
    is a model error, not Monte Carlo error. Independent scrambles estimate
    integration uncertainty; they are not a rigorous probabilistic bound.
    """
    if jax.default_backend() != "gpu":
        raise RuntimeError("The aperture/detector forward audit requires GPU.")
    root = np.linalg.cholesky(source_covariance)
    aperture = jnp.asarray([case.source_to_aperture[XY]@root for case in cases])
    detector = jnp.asarray([case.source_to_detector[XY]@root for case in cases])

    @jax.jit
    def evaluate(normal):
        def at_case(a, d):
            positions = normal@a.T
            hits = camera_acceptance(normal@d.T)
            inside = jnp.sum((positions[:, None, :]-jnp.asarray(CENTRES_M)[None])**2, axis=-1) <= jnp.asarray(RADII_M)**2
            return jnp.mean(inside.astype(normal.dtype), axis=0), jnp.mean(inside*hits[:, None], axis=0), jnp.mean(hits)
        return jax.lax.map(lambda pair: at_case(*pair), (aperture, detector))

    estimates = []
    for seed in range(repetitions):
        uniforms = qmc.Sobol(4, scramble=True, seed=1907+seed).random_base2(sobol_power)
        estimates.append(tuple(np.asarray(value) for value in evaluate(ndtri(jnp.clip(jnp.asarray(uniforms), 1e-12, 1-1e-12)))))
    total, captured, outside = [np.asarray([value[i] for value in estimates]) for i in range(3)]
    return dict(total_transmission=total.mean(axis=0), detected_aperture_in=captured.mean(axis=0),
        detected_aperture_out=outside.mean(axis=0),
        aperture_in_standard_error=captured.std(axis=0, ddof=1)/np.sqrt(repetitions),
        total_standard_error=total.std(axis=0, ddof=1)/np.sqrt(repetitions),
        fraction_of_transmitted_captured=captured.mean(axis=0)/total.mean(axis=0),
        finite_camera_in_out_ratio=captured.mean(axis=0)/outside.mean(axis=0)[:, None],
        sobol_power=sobol_power, independent_scrambles=repetitions, approximation="geometric; no edge diffraction")


def exact_aperture_totals(cases, covariance):
    values = []
    for case in cases:
        at_aperture = case.source_to_aperture@covariance@case.source_to_aperture.T
        result = aperture_edge_scan_from_moments(jnp.zeros((3, 2)), at_aperture[np.ix_(XY, XY)],
            aperture_centre_xy_m=jnp.asarray(CENTRES_M), aperture_radius_m=jnp.asarray(RADII_M),
            incident_dose_electrons=1., radial_order=64, azimuthal_order=192)
        values.append(np.asarray(result.expected_transmitted_electrons[:, 0]))
    return np.asarray(values)


@partial(jax.jit, static_argnames=("output_count",))
def _propagate_mode_integrals(indices, amplitude, modes_x, modes_y, phase_factor, masks,
                             integration_weights, input_start, input_step, output_start,
                             output_step, wavelength_b, *, output_count):
    """Stream modes; large field arrays are device arguments, not JIT constants."""
    def one(pair):
        nx, ny = pair
        incoming = amplitude*modes_x[nx]*modes_y[ny]*phase_factor
        def at_aperture(mask):
            field = czt_axis(incoming*mask, input_start=input_start, input_step=input_step,
                output_start=output_start, output_step=output_step, output_count=output_count,
                wavelength_times_b=wavelength_b, axis=-1)
            field = czt_axis(field, input_start=input_start, input_step=input_step,
                output_start=output_start, output_step=output_step, output_count=output_count,
                wavelength_times_b=wavelength_b, axis=-2)/wavelength_b
            return jnp.einsum("yx,kyx->k", jnp.abs(field)**2, integration_weights)*output_step**2
        return jax.lax.map(at_aperture, masks)
    return jax.lax.map(one, indices)


def wave_capture(case, source_covariance, *, input_count=1024, output_count=1024,
                 modes_per_axis=6, aperture_subpixels=4, mode_batch=4,
                 include_unmasked_grid=False):
    """GPU mixed-mode Collins integral after physical aperture multiplication."""
    if jax.default_backend() != "gpu":
        raise RuntimeError("Physical aperture-wave propagation must run on GPU.")
    wavelength = float(energy2wavelength(200e3))
    pure, weights, omitted = thermal_source_modes(source_covariance, wavelength, modes_per_axis=modes_per_axis)
    transform = case.aperture_to_detector
    b_block = transform[np.ix_(XY, ANGLE)]
    b = np.sqrt(np.sum(b_block*b_block)/2)
    if b < 1e-10:
        raise ValueError("An aperture-conjugate plane requires the direct-imaging limit, not this Collins kernel.")
    rotation = b_block/b
    a_matrix = rotation.T@transform[np.ix_(XY, XY)]
    a = np.trace(a_matrix)/2
    if not np.allclose(rotation.T@rotation, np.eye(2), atol=1e-7) or not np.allclose(a_matrix, a*np.eye(2), atol=1e-6, rtol=1e-10):
        raise ValueError("This independent reference requires a round section map plus rotation.")
    aperture_covariance = case.source_to_aperture@pure@case.source_to_aperture.T
    spatial = aperture_covariance[np.ix_(XY, XY)]
    phase = aperture_covariance[np.ix_(ANGLE, XY)]@np.linalg.inv(spatial)
    phase = .5*(phase+phase.T)
    source_axes = case.source_to_aperture[np.ix_(XY, XY)]
    source_axes /= np.sqrt(np.sum(source_axes*source_axes)/2)
    variances = np.diag(source_axes.T@spatial@source_axes)
    full_covariance = case.source_to_aperture@source_covariance@case.source_to_aperture.T
    extent = 1.2*RADII_M.max()+np.max(np.abs(CENTRES_M))
    if include_unmasked_grid:
        extent = max(extent, 8*np.sqrt(np.linalg.eigvalsh(full_covariance[np.ix_(XY, XY)]).max()))
    input_step = 2*extent/input_count
    coords = (jnp.arange(input_count)+.5)*input_step-extent
    output_extent = HALF_CAMERA_M*np.max(np.sum(abs(rotation), axis=0))+5*PSF_SIGMA_M
    output_step = 2*output_extent/output_count
    out = (jnp.arange(output_count)+.5)*output_step-output_extent
    combined_phase = phase+a/b*np.eye(2)
    # Bound phase cycles per input-grid step, including all requested output frequencies.
    max_phase_gradient = (np.max(np.sum(abs(combined_phase), axis=1))*extent+output_extent/abs(b))/wavelength
    cycles_per_step = input_step*max_phase_gradient
    if cycles_per_step > .25:
        raise ValueError(f"Input sampling needs refinement: {cycles_per_step:.3g} cycles/step exceeds0.25.")
    yy, xx = jnp.meshgrid(coords, coords, indexing="ij")
    positions = jnp.stack([xx, yy], axis=-1)
    local = positions@jnp.asarray(source_axes)
    modes_x = normalized_hermite_modes(local[..., 0]/np.sqrt(2*variances[0]), modes_per_axis)
    modes_y = normalized_hermite_modes(local[..., 1]/np.sqrt(2*variances[1]), modes_per_axis)
    amplitude = jnp.exp(-.25*jnp.einsum("...i,ij,...j->...", positions, jnp.linalg.inv(jnp.asarray(spatial)), positions))
    amplitude /= jnp.sqrt(2*jnp.pi*jnp.sqrt(jnp.linalg.det(jnp.asarray(spatial))))
    phase_factor = jnp.exp(1j*jnp.pi/wavelength*jnp.einsum("...i,ij,...j->...", positions, jnp.asarray(combined_phase), positions))
    masks = jnp.stack([circle_cell_fraction(coords, coords, input_step, radius, centre,
                                          subpixels=aperture_subpixels) for radius, centre in zip(RADII_M, CENTRES_M)])
    if include_unmasked_grid:
        masks = jnp.concatenate([masks, jnp.ones((1, input_count, input_count))])
    output_yy, output_xx = jnp.meshgrid(out, out, indexing="ij")
    output_positions = jnp.stack([output_xx, output_yy], axis=-1)@jnp.asarray(rotation).T
    camera_weight = camera_acceptance(output_positions)
    dx, dy = output_positions[..., 0], output_positions[..., 1]
    integration_weights = camera_weight[None]*jnp.stack([jnp.ones_like(dx), dx*dx, dx*dy, dy*dy])
    wavelength_b = wavelength*b
    index = np.array(np.meshgrid(np.arange(modes_per_axis), np.arange(modes_per_axis), indexing="ij")).reshape(2, -1).T
    powers = []
    for start in range(0, len(index), mode_batch):
        powers.extend(np.asarray(_propagate_mode_integrals(jnp.asarray(index[start:start+mode_batch]),
            amplitude, modes_x, modes_y, phase_factor, masks, integration_weights,
            -extent+.5*input_step, input_step, -output_extent+.5*output_step, output_step,
            wavelength_b, output_count=output_count)))
    moments = np.tensordot(np.asarray(weights).reshape(-1), np.asarray(powers), axes=(0, 0))
    captured = moments[:, 0]
    full_detector_covariance = case.source_to_detector@source_covariance@case.source_to_detector.T
    gaussian_detector = make_gaussian_detector_operator(DetectorGeometry(mtf_sigma_px=.65), 16)
    unmasked = float(jax.jit(lambda: jnp.sum(gaussian_detector(jnp.zeros(2),
        jnp.asarray(full_detector_covariance[np.ix_(XY, XY)]), 1.)))())
    grid_covariance, moment_error = None, None
    if include_unmasked_grid:
        xx, xy, yy = moments[-1, 1:]/captured[-1]
        grid_covariance = np.array([[xx, xy], [xy, yy]])
        reference_covariance = full_detector_covariance[np.ix_(XY, XY)]
        moment_error = float(np.linalg.norm(grid_covariance-reference_covariance)/np.linalg.norm(reference_covariance))
    return dict(detected_aperture_in=captured[:3].tolist(), detected_aperture_out=unmasked,
        finite_camera_in_out_ratio=(captured[:3]/unmasked).tolist(),
        aperture_out_reference="Analytic transported mixed-Gaussian intensity, integrated over finite pixels and detector PSF; not normalized to one.",
        unmasked_grid_capture=float(captured[3]) if include_unmasked_grid else None,
        unmasked_grid_minus_reference=float(captured[3]-unmasked) if include_unmasked_grid else None,
        unmasked_grid_pre_psf_covariance_m2=grid_covariance.tolist() if grid_covariance is not None else None,
        unmasked_grid_vs_untruncated_analytic_covariance_relative_error=moment_error,
        aperture_to_detector_b_m=float(b), aperture_to_detector_a=float(a),
        aperture_to_detector_rotation_rad=float(np.arctan2(rotation[1, 0], rotation[0, 0])),
        aperture_pure_gaussian_eigen_rms_m=np.sqrt(np.linalg.eigvalsh(spatial)).tolist(),
        input_count=input_count, output_count=output_count, aperture_subpixels=aperture_subpixels,
        input_extent_m=extent, output_extent_m=output_extent, input_phase_cycles_per_step=float(cycles_per_step),
        coherent_mode_count=modes_per_axis**2, omitted_source_probability=omitted,
        source_covariance_physical=True, detector_intensity_renormalized=False,
        specimen_transmission="Vacuum over the entire illuminated specimen-plane region; no40nm hole is applied.",
        approximation="paraxial scalar wave; thermal Hermite modes; fixed truncated objective field; Gaussian detector PSF")


def aperture_imaging_collection(cases, covariance, column, parameters, *, aperture_magnification=120.):
    """Construct a separate aperture-image collection state with fixed objective.

    C-mini is downstream of the aperture, so its collection setting does not
    change aperture-plane illumination. Projector relay strengths are converted
    back to the ORIGINAL DAC-native parameterization; no truth is rebased in
    the recorded control vector. This auxiliary state is not a40k sample image.
    """
    from neoarm_shared_objective import initialize_shared_imaging
    if jax.default_backend() != "gpu":
        raise RuntimeError("Collection-state finite-camera simulation requires GPU.")
    p = jnp.asarray(parameters)
    base = cases[0]
    mini = next(lens for lens in column.lenses if lens.name == "C-mini")
    inverse_aperture = jnp.linalg.inv(jnp.asarray(base.source_to_aperture))

    def aperture_to_sample(q):
        u = jnp.asarray(base.controls).at[mini.channel_index].set(mini.reference_dac+(q-p[12])/p[13])
        return column.matrix(p, u, segment="source_to_sample")@inverse_aperture

    def derotated_b(q):
        vector = aperture_to_sample(q)[jnp.ix_(jnp.asarray(XY), jnp.asarray(ANGLE))][:, 0]
        return jnp.array([jnp.cos(q)*vector[0]-jnp.sin(q)*vector[1],
                          jnp.sin(q)*vector[0]+jnp.cos(q)*vector[1]])

    reference = derotated_b(.01); reference /= jnp.linalg.norm(reference)
    residual = jax.jit(lambda q: jnp.dot(derotated_b(q), reference))
    q_grid = np.linspace(.001, np.pi-.001, 257)
    response = np.asarray(jax.jit(jax.vmap(residual))(q_grid))
    crossings = np.flatnonzero(response[:-1]*response[1:] <= 0)
    if not len(crossings):
        raise ValueError("No first-branch aperture-to-sample conjugacy was found.")
    bracket = q_grid[crossings[0]:crossings[0]+2]

    @jax.jit
    def bisect(low, high):
        def step(_, endpoints):
            lo, hi = endpoints; mid = (lo+hi)/2
            left = residual(lo)*residual(mid) <= 0
            return jnp.where(left, lo, mid), jnp.where(left, mid, hi)
        lo, hi = jax.lax.fori_loop(0, 52, step, (low, high))
        return (lo+hi)/2

    q_mini = float(bisect(*bracket))
    pre = np.asarray(aperture_to_sample(q_mini))
    aperture_to_sample_mag = np.sqrt(np.sum(pre[np.ix_(XY, XY)]**2)/2)
    specimen_magnification = aperture_magnification/aperture_to_sample_mag
    # Synchronize the initializer's lens metadata with the supplied parameters.
    actual_column = replace(column, source_z_m=float(p[0]), detector_z_m=float(p[1]),
        objective_initial=tuple(np.asarray(p[-4:])), lenses=tuple(replace(lens,
            z_center_m=float(p[2+4*i]), length_m=float(jnp.exp(p[3+4*i])),
            q_bias=float(p[4+4*i]), q_slope_per_dac=float(p[5+4*i]))
            for i, lens in enumerate(column.lenses)))
    relay = initialize_shared_imaging(actual_column, magnifications=(specimen_magnification,))
    if not np.allclose(relay.parameters[-4:], p[-4:], rtol=0, atol=1e-11):
        raise ValueError("Collection initialization attempted to change the fixed objective field.")
    collection = np.asarray(base.controls).copy()
    collection[2] = mini.reference_dac+(q_mini-float(p[12]))/float(p[13])
    for i, lens in enumerate(column.lenses[3:], start=3):
        collection[lens.channel_index] = lens.reference_dac+(relay.parameters[4+4*i]-float(p[4+4*i]))/float(p[5+4*i])
    upstream = cases[::2]
    controls = np.tile(collection, (len(upstream), 1))
    controls[:, :2] = np.asarray([case.controls[:2] for case in upstream])
    full = np.asarray(jax.jit(jax.vmap(lambda u: column.matrix(p, u)))(controls))
    sample = np.asarray(jax.jit(jax.vmap(lambda u: column.matrix(p, u, segment="source_to_sample")))(controls))
    aperture = np.asarray(jax.jit(jax.vmap(lambda u: column.matrix_to_plane(p, u, -.42083333333333334)))(controls))
    transfer = np.array([np.linalg.solve(a.T, d.T).T for a, d in zip(aperture, full)])
    a = transfer[:, XY][:, :, XY]; b = transfer[:, XY][:, :, ANGLE]
    achieved = np.sqrt(np.sum(a*a, axis=(1, 2))/2)
    if np.max(abs(achieved/aperture_magnification-1)) > 1e-8 or np.max(np.linalg.norm(b, axis=(1, 2))) > 1e-8:
        raise ValueError("The original physical column fails aperture-image collection checks.")
    sample_covariance = sample@covariance@sample.transpose(0, 2, 1)
    detector_covariance = full@covariance@full.transpose(0, 2, 1)
    source_to_aperture_covariance = aperture@covariance@aperture.transpose(0, 2, 1)
    sample_rms = np.sqrt(np.linalg.eigvalsh(sample_covariance[:, XY][:, :, XY]))
    detector_rms = np.sqrt(np.linalg.eigvalsh(detector_covariance[:, XY][:, :, XY]))
    aperture_rms = np.sqrt(np.linalg.eigvalsh(source_to_aperture_covariance[:, XY][:, :, XY]))
    detector_operator = make_gaussian_detector_operator(DetectorGeometry(mtf_sigma_px=.65), 16)
    aperture_out_capture = np.asarray(jax.jit(jax.vmap(lambda cov: jnp.sum(detector_operator(jnp.zeros(2),
        cov[jnp.ix_(jnp.asarray(XY), jnp.asarray(XY))], 1.))))(detector_covariance))
    exact = exact_aperture_totals([CaptureCase(case.label, specimen_magnification, u, ap, d)
        for case, u, ap, d in zip(upstream, controls, aperture, full)], covariance)
    # In the exact conjugate limit the entire transmitted wave is a coordinate
    # image of the aperture. A Gaussian-PSF tail bound certifies finite capture.
    centre_detector = np.einsum("kij,rj->kri", a, CENTRES_M)
    margin = HALF_CAMERA_M-np.max(abs(centre_detector), axis=-1)-achieved[:, None]*RADII_M
    log_escape_bound = np.log(4.)+np.asarray(log_ndtr(-jnp.asarray(margin)/PSF_SIGMA_M))
    hole_transmission = []
    for cov in sample_covariance:
        value = aperture_edge_scan_from_moments(jnp.zeros((1, 2)), cov[np.ix_(XY, XY)],
            aperture_centre_xy_m=jnp.zeros(2), aperture_radius_m=40e-9,
            incident_dose_electrons=1., radial_order=32, azimuthal_order=96)
        hole_transmission.append(float(value.expected_transmitted_electrons[0, 0]))
    phases = np.asarray([[p[4+4*i]+p[5+4*i]*(u[i]-column.lenses[i].reference_dac)
                         for i in range(7)] for u in controls], float)
    return dict(controls=controls.tolist(), labels=[case.label for case in upstream], phases_q=phases.tolist(),
        aperture_to_detector_matrices=transfer.tolist(), achieved_aperture_magnification=achieved.tolist(),
        maximum_aperture_to_detector_B_norm_m=float(np.linalg.norm(b, axis=(1, 2)).max()),
        maximum_aperture_map_change_from_original=float(max(np.max(abs(ap-case.source_to_aperture))
            for ap, case in zip(aperture, upstream))),
        objective_parameter_change=(relay.parameters[-4:]-np.asarray(p[-4:])).tolist(),
        objective_controls_fixed=True, same_sign_first_branch_projector_phases=bool(np.all(abs(phases[:, 3:]) < np.pi)),
        auxiliary_specimen_magnification=float(specimen_magnification),
        aperture_to_sample_magnification=float(aperture_to_sample_mag),
        aperture_plane_eigen_rms_m=aperture_rms.tolist(), sample_plane_eigen_rms_m=sample_rms.tolist(),
        detector_eigen_rms_m=detector_rms.tolist(), aperture_out_camera_fraction=aperture_out_capture.tolist(),
        aperture_in_camera_escape_log_probability_upper_bound=log_escape_bound.tolist(),
        aperture_image_detector_edge_margin_m=margin.tolist(),
        aperture_plane_transmission=exact.tolist(),
        ideal_conjugate_camera_in_out_ratio=(exact/aperture_out_capture[:, None]).tolist(),
        required_clear_sample_radius_six_rms_m=float(6*sample_rms.max()),
        original_40nm_radius_hole_aperture_out_transmission=hole_transmission,
        specimen_plane_maximum_aperture_image_radius_m=float(aperture_to_sample_mag*(RADII_M+np.linalg.norm(CENTRES_M, axis=1)).max()),
        collection_ray_map_verified=True, wave_route="Exact conjugate coordinate-image limit, no intensity renormalization; residual B checked numerically.",
        all_other_apertures_and_lens_bore_clearance_verified=False,
        hardware_commands_admitted=False, existing_40nm_hole_adequate=bool(min(hole_transmission) >= .999),
        warning="Requires a micrometre-scale clear specimen path plus pose margins. The original40nm synthetic hole blocks most collection flux.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--information", type=Path, default=DEFAULT_INFORMATION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wave-case-indices", type=int, nargs="*", default=[])
    parser.add_argument("--input-count", type=int, default=1024)
    parser.add_argument("--output-count", type=int, default=1024)
    parser.add_argument("--modes-per-axis", type=int, default=6)
    parser.add_argument("--sobol-power", type=int, default=18)
    parser.add_argument("--include-unmasked-grid", action="store_true")
    parser.add_argument("--collection", action="store_true")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "gpu":
        raise RuntimeError("The aperture-capture forward audit requires GPU.")
    started = perf_counter()
    initial_source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    information_hash = hashlib.sha256(args.information.read_bytes()).hexdigest()
    cases, covariance, column, parameters = load_cases(args.information)
    from neoarm_shared_objective_audit import provenance_snapshot, verify_provenance_snapshot
    sources, computational_inputs = provenance_snapshot()
    geometric = geometric_capture(cases, covariance, sobol_power=args.sobol_power)
    exact = exact_aperture_totals(cases, covariance)
    waves = {}
    for index in args.wave_case_indices:
        try:
            result = wave_capture(cases[index], covariance, input_count=args.input_count,
                output_count=args.output_count, modes_per_axis=args.modes_per_axis,
                include_unmasked_grid=args.include_unmasked_grid)
        except ValueError as error:
            waves[str(index)] = {"status": "not_evaluated", "reason": str(error)}
            print("Wave case", index, "not evaluated:", error, flush=True)
            continue
        result["fraction_of_transmitted_captured"] = (np.array(result["detected_aperture_in"])/exact[index]).tolist()
        waves[str(index)] = result
        print("Wave case", index, "capture", result["fraction_of_transmitted_captured"], flush=True)
    collection = aperture_imaging_collection(cases, covariance, column, parameters) if args.collection else None
    verify_provenance_snapshot(sources)
    if (hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != initial_source_hash
            or hashlib.sha256(args.information.read_bytes()).hexdigest() != information_hash):
        raise RuntimeError("Aperture-audit source or input report changed during execution.")
    report = dict(schema="neoarm-aperture-capture-v1", backend=jax.default_backend(),
        information_report_sha256=information_hash, information_report_file=str(args.information),
        source_sha256=initial_source_hash, source_hashes=sources,
        computational_input_hashes=computational_inputs, source_snapshot_verified_unchanged=True,
        cases=[dict(label=c.label, magnification=c.magnification, controls=c.controls.tolist()) for c in cases],
        aperture_radii_m=RADII_M.tolist(), aperture_centres_m=CENTRES_M.tolist(),
        source_covariance=covariance.tolist(), exact_aperture_plane_transmission=exact.tolist(),
        geometric={key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in geometric.items()},
        wave=waves, aperture_collection=collection, elapsed_seconds=perf_counter()-started,
        hardware_commands_admitted=False, complete_wave_convergence_proved=False,
        objective_field_tail_model_validated=False,
        specimen_plane_vacuum_assumed=True, other_apertures_and_lens_bores_modelled=False,
        count_warning="An aperture-out finite-camera count is not an independent total incident-dose measurement.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
