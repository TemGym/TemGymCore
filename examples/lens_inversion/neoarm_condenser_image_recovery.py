"""Conditional condenser recovery from noisy beam and aperture-image observables.

No upstream beam-command calibration or independent ellipse-angle tag enters
the fit. The source reference plane is a declared coordinate gauge; all ten
covariance directions remain free, including the invisible antisymmetric
position-angle covariance. Objective/downstream maps are an explicit external
calibration for this first-stage experiment, not recovered instrument truth.
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
from temgym_core.critical_line_calibration import EntranceCovarianceChart
from temgym_core.gaussian_image_likelihood import (
    make_gaussian_detector_operator, fit_gaussian_image,
    validate_phase_space_covariance,
)
from temgym_core.immersed_objective import uniform_field_canonical_matrix, canonical_to_mechanical
from temgym_core.jax_least_squares import least_squares
from temgym_core.microscope_inversion import drift_matrix
from temgym_core.neoarm_calibration import DetectorGeometry
from neoarm_aperture_capture import ROOT, XY, load_cases
from neoarm_aperture_image_fit import decode_spatial
from neoarm_measured_information_audit import measured_column
from neoarm_shared_objective_audit import provenance_snapshot, verify_provenance_snapshot


DEFAULT_APERTURE = ROOT/"examples/generated/neoarm_inversion/imaging_recipe/aperture_image_fit.json"
TRAIN_FAMILIES = (0, 1, 2, 3, 4, 5, 12, 13)
HELDOUT_FAMILIES = (6, 7, 14)
SOURCE_REFERENCE_SCALES = np.array([10e-9, 40e-6, 10e-9, 40e-6])
OMEGA = np.kron(np.eye(2), np.array([[0., 1.], [-1., 0.]]))


def covariance_observable(covariance):
    sigma = jnp.sqrt(jnp.diag(covariance))
    return jnp.r_[jnp.log(sigma), jnp.arctanh(covariance[0, 1]/jnp.prod(sigma))]


def covariance_from_observable(p):
    return decode_spatial(jnp.r_[jnp.zeros(2), p])[1]


def rotation(angle):
    c, s = jnp.cos(angle), jnp.sin(angle)
    return jnp.array([[c, -s], [s, c]])


def read_verified_aperture(path):
    report = json.loads(Path(path).read_text())
    for section in ("source_hashes", "report_input_hashes"):
        for name, expected in report[section].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Stale aperture metrology input: {name}")
    array_path = Path(report["array_file"])
    if not array_path.is_absolute():
        array_path = ROOT/array_path
    if hashlib.sha256(array_path.read_bytes()).hexdigest() != report["array_sha256"]:
        raise ValueError("Aperture metrology array checksum mismatch.")
    if not report["all_selected_exact_conjugate_image_fits_accepted"]:
        raise ValueError("Aperture image metrology must pass its declared gates.")
    return report


class CondenserModel:
    """Fast three-lens JAX map with precomputed conditional objective/analyser maps."""

    def __init__(self, column, fixed_parameters, analyser_controls, *, aperture_z_m=-.42083333333333334):
        self.column = column
        self.fixed = np.asarray(fixed_parameters)
        self.source_z_m = float(self.fixed[0])
        self.aperture_z_m = float(aperture_z_m)
        self.objective_entrance_m = float(column.objective.z_edges_m[0])
        split = jax.jit(lambda p: column.objective.split_maps(p))(jnp.asarray(self.fixed[-4:]))
        self.prefield = canonical_to_mechanical(split.sample_k_m_inv)@split.pre
        self.downstream = jax.jit(jax.vmap(lambda u: column.matrix(self.fixed, u,
            segment="sample_to_detector", coordinates="mechanical")))(jnp.asarray(analyser_controls))
        self.channels = tuple(lens.channel_index for lens in column.lenses[:3])
        self.reference_controls = jnp.asarray([lens.reference_dac for lens in column.lenses[:3]])
        self.chart = EntranceCovarianceChart(np.diag(SOURCE_REFERENCE_SCALES**2))
        self.base = np.asarray(measured_column().nominal_parameters[:14])[2:14]
        # measured_column has the same condenser layout; source/detector first.
        self.scales = np.tile([.01, .1, .1, .1], 3)

    def condenser_matrix(self, optical, control, *, to_aperture=False, source_z_m=None):
        matrix = jnp.eye(4)
        cursor = self.source_z_m if source_z_m is None else source_z_m
        count = 2 if to_aperture else 3
        for i in range(count):
            centre, log_length, bias, slope = optical[4*i:4*i+4]
            length = jnp.exp(log_length)
            matrix = drift_matrix(centre-length/2-cursor)@matrix
            q = bias+slope*(control[self.channels[i]]-self.reference_controls[i])
            matrix = uniform_field_canonical_matrix(q/length, length)@matrix
            cursor = centre+length/2
        target = self.aperture_z_m if to_aperture else self.objective_entrance_m
        return drift_matrix(target-cursor)@matrix

    def maps(self, optical, control, *, source_z_m=None):
        before = self.condenser_matrix(optical, control, source_z_m=source_z_m)
        return jnp.einsum("aij,jk->aik", self.downstream@self.prefield, before)

    def decode(self, p):
        return jnp.asarray(self.base)+jnp.asarray(self.scales)*p[:12], self.chart.covariance(p[12:22]), p[22]

    def observables(self, p, beam_controls, aperture_controls):
        optical, covariance, collection_rotation = self.decode(p)
        def beam(control):
            maps = self.maps(optical, control)
            output = maps@covariance@jnp.swapaxes(maps, -1, -2)
            return jax.vmap(covariance_observable)(output[:, XY][:, :, XY]/(55e-6)**2)
        def aperture(control):
            matrix = self.condenser_matrix(optical, control, to_aperture=True)
            output = matrix@covariance@matrix.T
            r = rotation(collection_rotation)
            return covariance_observable(r@output[jnp.ix_(jnp.asarray(XY), jnp.asarray(XY))]@r.T)
        return jax.vmap(beam)(jnp.asarray(beam_controls)), jax.vmap(aperture)(jnp.asarray(aperture_controls))

    def physicality_residual(self, p):
        _, covariance, _ = self.decode(p)
        epsilon = energy2wavelength(200e3)/(4*jnp.pi)
        scale = jnp.array([10e-9, epsilon/10e-9]*2)
        balanced = covariance/jnp.outer(scale, scale)
        eigenvalues = jnp.linalg.eigvalsh(balanced+1j*jnp.asarray(OMEGA))
        return jnp.minimum(eigenvalues, 0.)*1e4


def covariance_chart_coordinates(model, covariance):
    whitened = covariance/np.outer(SOURCE_REFERENCE_SCALES, SOURCE_REFERENCE_SCALES)
    eigenvalues, eigenvectors = np.linalg.eigh(whitened)
    if np.any(eigenvalues <= 0):
        raise ValueError("The initialized source covariance must be positive definite.")
    generator = (eigenvectors*np.log(eigenvalues))@eigenvectors.T
    return np.einsum("kij,ij->k", model.chart.symmetric_basis, generator)


def linear_source_initializer(model, controls, beam_observables, beam_covariances,
                              aperture_controls, aperture_observables, aperture_covariances):
    """Data-only linear covariance tomography on the approximate diagram optics.

    Grid the unknown common aperture-frame angle. The minimum-norm linear
    solution picks an explicit representative of the invisible covariance
    direction. Projecting to the quantum cone is initialization only.
    """
    optical = jnp.asarray(model.base)
    maps = np.asarray(jax.jit(jax.vmap(lambda u: model.maps(optical, u)))(jnp.asarray(controls))).reshape(-1, 4, 4)
    aperture_maps = np.asarray(jax.jit(jax.vmap(lambda u: model.condenser_matrix(optical, u, to_aperture=True)))(jnp.asarray(aperture_controls)))
    basis = np.asarray(model.chart.symmetric_basis)*np.outer(SOURCE_REFERENCE_SCALES, SOURCE_REFERENCE_SCALES)
    observed = np.concatenate([np.asarray(beam_observables).reshape(-1, 3), aperture_observables])
    measurement_cov = np.concatenate([np.asarray(beam_covariances).reshape(-1, 3, 3), aperture_covariances])
    def linear_quantities(o):
        c = covariance_from_observable(o)
        return jnp.array([c[0, 0], c[0, 1], c[1, 1]])
    targets = np.asarray(jax.vmap(linear_quantities)(observed))
    derivative = np.asarray(jax.vmap(jax.jacfwd(linear_quantities))(observed))
    roots = np.linalg.cholesky(derivative@measurement_cov@derivative.transpose(0, 2, 1))
    target = np.linalg.solve(roots, targets[..., None]).reshape(-1)
    best = None
    for angle in np.linspace(-np.pi, np.pi, 49, endpoint=False):
        r = np.asarray(rotation(angle))
        rows = np.concatenate([maps[:, XY]/55e-6, np.einsum("ij,ajk->aik", r, aperture_maps[:, XY])])
        transported = np.einsum("aij,kjl,aml->akim", rows, basis, rows)
        design = transported[:, :, (0, 0, 1), (0, 1, 1)].transpose(0, 2, 1)
        whitened = np.linalg.solve(roots, design).reshape(-1, 10)
        coordinates = np.linalg.lstsq(whitened, target, rcond=1e-10)[0]
        cost = np.linalg.norm(whitened@coordinates-target)**2
        if best is None or cost < best[0]:
            best = (cost, angle, np.einsum("k,kij->ij", coordinates, basis))
    covariance = best[2]
    epsilon = float(energy2wavelength(200e3))/(4*np.pi)
    scale = np.array([10e-9, epsilon/10e-9]*2)
    balanced = covariance/np.outer(scale, scale)
    shift = max(0., .01-np.linalg.eigvalsh(balanced+1j*OMEGA).min())
    covariance += shift*np.diag(scale**2)
    validate_phase_space_covariance(covariance, energy2wavelength(200e3))
    return np.r_[np.zeros(12), covariance_chart_coordinates(model, covariance), best[1]], dict(
        aperture_rotation_grid_count=49, selected_rotation_rad=float(best[1]),
        linear_covariance_cost=float(best[0]), initial_quantum_cone_diagonal_shift=float(shift),
        optics="Approximate measured-diagram condenser positions/lengths and provisional excitation law; not fitted truth parameters.",
        source_covariance="Linear fit to measured covariance observations; minimum-norm null representative, then physicality projection.")


def generate_beam_measurements(model, truth_optical, source_covariance, controls, detector, *, dose=2e6, seed=8521):
    if jax.default_backend() != "gpu":
        raise RuntimeError("Detector image generation and fitting must use GPU.")
    maps = np.asarray(jax.jit(jax.vmap(lambda u: model.maps(truth_optical, u)))(jnp.asarray(controls)))
    covariance = maps@source_covariance@maps.transpose(0, 1, 3, 2)
    operator = jax.jit(make_gaussian_detector_operator(detector, 16))
    rng = np.random.default_rng(seed)
    observed, estimates, uncertainty, expected, rows = [], [], [], [], []
    for i, cov in enumerate(covariance.reshape(-1, 4, 4)):
        frame_dose = dose*(1+.025*np.sin(i))
        prediction = np.asarray(operator(jnp.zeros(2), cov[np.ix_(XY, XY)], frame_dose))
        variance = prediction+.6**2
        noisy = rng.poisson(prediction)+rng.normal(0., .6, prediction.shape)
        fit = fit_gaussian_image(noisy, variance, detector, quadrature_order=16, max_nfev=160)
        fit.require_accepted()
        if abs(fit.chi_square_z) > 5:
            raise ValueError("Two-sided noisy-image goodness of fit failed.")
        p = np.asarray(fit.parameters)
        estimates.append(p[2:5]); uncertainty.append(fit.parameter_covariance[2:5, 2:5])
        observed.append(noisy); expected.append(prediction)
        rows.append(dict(frame=i, optimizer_success=bool(fit.optimizer_result.success),
            image_rank=int(fit.numerical_rank), captured_fraction=float(prediction.sum()/frame_dose),
            residual_rms_sigma=fit.residual_rms_sigma, chi_square_z=fit.chi_square_z,
            incident_dose_estimate=float(np.exp(p[5])), incident_dose_truth=frame_dose,
            covariance_marginalizes_centroid_and_dose=True))
    shape = (len(controls), len(model.downstream))
    return np.asarray(estimates).reshape(*shape, 3), np.asarray(uncertainty).reshape(*shape, 3, 3), rows, dict(
        noisy_beam_images=np.asarray(observed).reshape(*shape, *detector.shape_yx),
        expected_beam_images=np.asarray(expected).reshape(*shape, *detector.shape_yx),
        beam_observables=np.asarray(estimates).reshape(*shape, 3),
        beam_observable_covariances=np.asarray(uncertainty).reshape(*shape, 3, 3))


def profile_optical_information(jacobian, count=12):
    matrix = np.asarray(jacobian)
    nuisance = matrix[:, count:]
    left, singular, _ = np.linalg.svd(nuisance, full_matrices=False)
    nuisance_rank = int(np.sum(singular > singular[0]*1e-10))
    profiled = matrix[:, :count]-left[:, :nuisance_rank]@(left[:, :nuisance_rank].T@matrix[:, :count])
    _, singular, right = np.linalg.svd(profiled, full_matrices=False)
    rank = int(np.sum(singular > singular[0]*1e-10))
    covariance = (right.T/singular**2)@right if rank == count else np.full((count, count), np.nan)
    return dict(optical_rank=rank, nuisance_rank=nuisance_rank, singular_values=singular.tolist(),
                normalized_parameter_standard_errors=np.sqrt(np.diag(covariance)).tolist()), covariance


def finite_json(value):
    """Represent unidentified/non-finite numeric diagnostics as explicit nulls."""
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def postfit_truth_check(model, column, optical_truth, source_truth, beam_controls,
                        beam_observed, beam_covariance, aperture_rows, aperture_covariance,
                        collection_report, training_count):
    """Generating values are accessed here only as a post-fit consistency test."""
    aperture_controls = np.asarray([row["controls"] for row in aperture_rows])
    collection_map = np.asarray(collection_report["aperture_collection"]["aperture_to_detector_matrices"][0])
    a = collection_map[np.ix_(XY, XY)]
    angle = float(np.arctan2(a[1, 0], a[0, 0]))
    parameters = np.r_[(optical_truth[2:14]-model.base)/model.scales,
        covariance_chart_coordinates(model, source_truth), angle]
    beam, aperture = model.observables(parameters, beam_controls, aperture_controls)
    beam_error = np.linalg.solve(np.linalg.cholesky(beam_covariance),
        (np.asarray(beam)-beam_observed)[..., None]).reshape(len(beam_controls), -1)
    aperture_observed = np.asarray([row["profiled_aperture_observables"][2:5] for row in aperture_rows])
    aperture_error = np.linalg.solve(np.linalg.cholesky(aperture_covariance),
        (np.asarray(aperture)-aperture_observed)[..., None]).reshape(len(aperture_rows), -1)
    aperture_truth = np.asarray([covariance_observable(np.asarray(
        row["aperture_plane_covariance_detector_aligned_truth_m2"])) for row in aperture_rows])
    # Compare the optimized fast factorization against the actual full-column
    # ray function, rather than checking the renderer against itself.
    full_controls = np.repeat(beam_controls, len(model.downstream), axis=0)
    full_controls[:, 3:] = np.tile(model.analyser_controls[:, 3:], (len(beam_controls), 1))
    full_maps = np.asarray(jax.jit(jax.vmap(lambda u: column.matrix(optical_truth, u)))(full_controls))
    fast_maps = np.asarray(jax.jit(jax.vmap(lambda u: model.maps(optical_truth[2:14], u)))(jnp.asarray(beam_controls))).reshape(-1, 4, 4)
    relative_error = np.linalg.norm(full_maps-fast_maps, axis=(1, 2))/np.linalg.norm(full_maps, axis=(1, 2))
    actual_covariance = full_maps@source_truth@full_maps.transpose(0, 2, 1)
    actual_observables = np.asarray(jax.vmap(covariance_observable)(actual_covariance[:, XY][:, :, XY]/55e-6**2)).reshape(np.asarray(beam).shape)
    beam_closure = np.linalg.solve(np.linalg.cholesky(beam_covariance),
        (np.asarray(beam)-actual_observables)[..., None]).reshape(-1)
    aperture_closure = np.linalg.solve(np.linalg.cholesky(aperture_covariance),
        (np.asarray(aperture)-aperture_truth)[..., None]).reshape(-1)
    return dict(accessed_only_after_optical_fit=True, collection_rotation_rad=angle,
        beam_training_chi_square=float(np.sum(beam_error[:training_count]**2)),
        aperture_training_chi_square=float(np.sum(aperture_error**2)),
        total_training_chi_square=float(np.sum(beam_error[:training_count]**2)+np.sum(aperture_error**2)),
        beam_heldout_chi_square=float(np.sum(beam_error[training_count:]**2)),
        beam_noiseless_full_column_closure_maximum_sigma=float(np.max(abs(beam_closure))),
        aperture_noiseless_closure_maximum_sigma=float(np.max(abs(aperture_closure))),
        fast_factorization_vs_full_column_maximum_relative_matrix_error=float(relative_error.max()),
        diagram_initial_optical_parameters_coincide_with_generator=bool(np.allclose(model.base, optical_truth[2:14], rtol=0, atol=1e-14)),
        interpretation="Likelihood closure is a model/frame check, not convergence or recovery evidence.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aperture", type=Path, default=DEFAULT_APERTURE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dose", type=float, default=2e6)
    parser.add_argument("--maximum-optical-evaluations", type=int, default=1500)
    parser.add_argument("--initial-damping", type=float, default=1e-10)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "gpu":
        raise RuntimeError("The condenser image demonstration requires GPU.")
    started = perf_counter()
    aperture_report = read_verified_aperture(args.aperture)
    information_path = next(ROOT/name for name in aperture_report["report_input_hashes"] if "information_narrow_probe" in name)
    information = json.loads(information_path.read_text())
    collection_path = next(ROOT/name for name in aperture_report["report_input_hashes"] if "aperture_collection.json" in name)
    collection_report = json.loads(collection_path.read_text())
    cases, source_covariance, column, truth_parameters = load_cases(information_path)
    analysers = np.asarray(information["projection_states"]["controls"])[[0, 2]]
    model = CondenserModel(column, truth_parameters, analysers)
    model.analyser_controls = analysers
    sources, inputs = provenance_snapshot()
    controls = np.asarray([case.controls for case in cases[::2]])
    selected = np.array(TRAIN_FAMILIES+HELDOUT_FAMILIES)
    detector = DetectorGeometry(mtf_sigma_px=.65)
    observed, covariance, image_rows, arrays = generate_beam_measurements(model, truth_parameters[2:14],
        source_covariance, controls[selected], detector, dose=args.dose)
    aperture_rows = aperture_report["records"]
    apcontrols = np.asarray([row["controls"] for row in aperture_rows])
    aperture_observed = np.asarray([row["profiled_aperture_observables"][2:5] for row in aperture_rows])
    aperture_covariance = np.asarray([np.asarray(row["profiled_aperture_observable_covariance"])[2:5, 2:5] for row in aperture_rows])
    count = len(TRAIN_FAMILIES)
    training_controls, training_observed, training_covariance = controls[selected[:count]], observed[:count], covariance[:count]
    initial, initialization = linear_source_initializer(model, training_controls, training_observed, training_covariance,
        apcontrols, aperture_observed, aperture_covariance)
    roots = jnp.asarray(np.linalg.cholesky(training_covariance))
    aproots = jnp.asarray(np.linalg.cholesky(aperture_covariance))
    def data_residual(p):
        beam, aperture = model.observables(p, training_controls, apcontrols)
        return jnp.r_[jnp.linalg.solve(roots, (beam-training_observed)[..., None]).reshape(-1),
                      jnp.linalg.solve(aproots, (aperture-aperture_observed)[..., None]).reshape(-1)]
    def residual(p):
        return jnp.r_[data_residual(p), model.physicality_residual(p)]
    lower = np.r_[np.tile([-1., -4., -1., -3.], 3), np.full(10, -8.), -2*np.pi]
    upper = np.r_[np.tile([1., 4., 1., 3.], 3), np.full(10, 8.), 2*np.pi]
    initial_jacobian = np.asarray(jax.jit(jax.jacfwd(data_residual))(initial))
    sensitivity_scale = 1/np.maximum(np.linalg.norm(initial_jacobian, axis=0), 1e-8)
    fit = least_squares(residual, initial, bounds=(lower, upper), x_scale=sensitivity_scale,
        max_nfev=args.maximum_optical_evaluations, max_iter=args.maximum_optical_evaluations,
        initial_damping=args.initial_damping, ftol=1e-11, xtol=1e-10, gtol=1e-7)
    parameters = np.asarray(fit.x)
    optical, fitted_source, angle = model.decode(parameters)
    data_jacobian = np.asarray(jax.jit(jax.jacfwd(data_residual))(parameters))
    diagnostic, optical_covariance = profile_optical_information(data_jacobian)
    errors = (np.asarray(optical)-truth_parameters[2:14]).reshape(3, 4)
    predicted_holdout = np.asarray(model.observables(parameters, controls[selected[count:]], apcontrols)[0])
    holdout_error = np.linalg.solve(np.linalg.cholesky(covariance[count:]),
        (predicted_holdout-observed[count:])[..., None]).reshape(-1)
    physicality = validate_phase_space_covariance(np.asarray(fitted_source), energy2wavelength(200e3))
    residual_values = np.asarray(data_residual(parameters))
    singular = np.linalg.svd(data_jacobian, compute_uv=False)
    rank = int(np.sum(singular > singular[0]*1e-10))
    # This full-source covariance perturbation is invisible to any round
    # position projection; report it explicitly rather than deleting a column.
    antisymmetric = np.zeros((4, 4)); antisymmetric[0, 3]=antisymmetric[3, 0]=1e-14
    antisymmetric[1, 2]=antisymmetric[2, 1]=-1e-14
    source_derivative = np.asarray(jax.jacfwd(model.chart.covariance)(parameters[12:22])).reshape(16, 10)
    null_coordinates = np.linalg.lstsq(source_derivative/np.outer(SOURCE_REFERENCE_SCALES, SOURCE_REFERENCE_SCALES).reshape(-1, 1),
        (antisymmetric/np.outer(SOURCE_REFERENCE_SCALES, SOURCE_REFERENCE_SCALES)).reshape(-1), rcond=1e-12)[0]
    null_vector = np.r_[np.zeros(12), null_coordinates, 0.]
    truth_check = postfit_truth_check(model, column, truth_parameters, source_covariance,
        controls[selected], observed, covariance, aperture_rows, aperture_covariance,
        collection_report, count)
    verify_provenance_snapshot(sources)
    arrays.update(fitted_parameters=parameters, initial_parameters=initial, data_jacobian=data_jacobian,
        fitted_source_covariance=np.asarray(fitted_source), optical_parameter_covariance=optical_covariance,
        selected_families=selected, controls=controls[selected], aperture_observables=aperture_observed,
        aperture_observable_covariances=aperture_covariance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    array_path = args.output.with_suffix(".npz"); np.savez_compressed(array_path, **arrays)
    sigma = np.sqrt(np.diag(optical_covariance))*model.scales
    report = dict(schema="neoarm-condenser-image-recovery-v1", source_hashes=sources,
        computational_input_hashes=inputs, aperture_report_path=str(args.aperture),
        aperture_report_sha256=hashlib.sha256(args.aperture.read_bytes()).hexdigest(),
        information_report_sha256=hashlib.sha256(information_path.read_bytes()).hexdigest(),
        arrays_path=str(array_path), arrays_sha256=hashlib.sha256(array_path.read_bytes()).hexdigest(),
        backend=jax.default_backend(), elapsed_seconds=perf_counter()-started, dose_electrons=args.dose,
        train_families=list(TRAIN_FAMILIES), heldout_families=list(HELDOUT_FAMILIES), image_fits=image_rows,
        initialization=initialization, initial_parameters=initial.tolist(), fitted_parameters=parameters.tolist(),
        optimizer_success=bool(fit.success), optimizer_message=str(fit.message), nfev=int(fit.nfev),
        optimizer_optimality=float(fit.optimality), optimizer_history=list(fit.history),
        optimizer_initial_damping=args.initial_damping, optimizer_sensitivity_scale=sensitivity_scale.tolist(),
        truth_consistency=truth_check,
        data_chi_square=float(residual_values@residual_values), data_coordinate_count=len(residual_values),
        data_jacobian_rank=rank, full_parameter_count=23, optical_information=diagnostic,
        optical_parameters_truth=truth_parameters[2:14].tolist(), optical_parameters_fitted=np.asarray(optical).tolist(),
        lens_errors_z_m_logL_qbias_qslope=errors.tolist(), optical_parameter_standard_errors=sigma.tolist(),
        heldout_chi_square=float(holdout_error@holdout_error), heldout_coordinate_count=len(holdout_error),
        fitted_source_covariance=np.asarray(fitted_source).tolist(), source_physicality=physicality,
        covariance_antisymmetric_null_jacobian_norm=float(np.linalg.norm(data_jacobian@null_vector)),
        source_reference_plane_m=model.source_z_m, source_reference_plane_is_coordinate_gauge=True,
        all_ten_source_covariance_directions_released=True, independent_angle_tag=False,
        objective_and_downstream_calibration_held_fixed=True, collection_rotation_free=True,
        aperture_radius_calibration_held_exact=True,
        common_aperture_radius_scale_nuisance_fitted=False,
        objective_downstream_and_aperture_calibration_uncertainty_propagated=False,
        local_information_evaluated_at_unconverged_iterate=not bool(fit.success),
        heldout_scope="Beam-image modality only: aperture covariance for family14 participates in training, so family14 is not a wholly held-out optical state.",
        blind_separated_geometry_initialization_demonstrated=False,
        optical_parameter_refinement_demonstrated=False,
        next_step="Resolve correlated-valley optimization using the same likelihood, then reserve aperture family14 with its beam images and propagate common radius, aperture-plane, and objective/downstream calibration uncertainty.",
        aperture_finite_B_metrology_conditional=True, upstream_astigmatic_source_preparation_conditional=True,
        hardware_admitted=False, complete_condenser_recovery_accepted=False,
        assumptions=["Fixed but unknown astigmatic source covariance upstream of CL1; physical preparation/stigmator location unconfirmed.",
            "Aperture spatial metrology conditional on exact conjugacy, known radii, and clear vacuum path; collection rotation remains free.",
            "Source position is a coordinate-reference gauge, not a recovered physical source location.",
            "Independent dose and centroid nuisances are marginalized by each measured-image covariance fit.",
            "Objective field and downstream analysers treated as externally calibrated for this initial conditional experiment.",
            "No high-magnification total-flux assumption, upstream tilt calibration, or independent Larmor-angle measurement."])
    args.output.write_text(json.dumps(finite_json(report), indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: report[key] for key in ["optimizer_success", "nfev", "data_chi_square", "data_jacobian_rank", "heldout_chi_square"]}), flush=True)


if __name__ == "__main__":
    main()
