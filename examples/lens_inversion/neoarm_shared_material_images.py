"""GPU specimen images and data-initialized metrology in the condenser column.

One fixed cubic view and three Gaussian refinements are used at a conjugate
40k state. The probe is prescribed at the sample, not recovered upstream.
Only terminal intensity is admitted by the conjugate reference; this is not
an interchangeable complex-wavefront or physical lens-recovery benchmark.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import tomllib

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jnp
import numpy as np

from neoarm_aperture_capture import ROOT, XY
from neoarm_condenser_endpoint_calibration import EndpointRows, digest
from neoarm_condenser_extension_design import FOLDER
from neoarm_conjugate_sample_metrology import PARAMETER_NAMES, PARAMETER_SCALES
from neoarm_finite_pixel_specimen import (
    RegularGaussianLattice, integrated_gaussian_counts,
    make_finite_pixel_conjugate_predictor,
)
from neoarm_measured_specimen import frozen_measured_view, require_view_coverage
from neoarm_mixed_gaussian_validation import detector_quadrature
from neoarm_shared_objective_audit import verify_provenance_snapshot
from temgym_core.constants import energy2wavelength
from temgym_core.evaluate import evaluate_gaussians
from temgym_core.jax_least_squares import least_squares
from temgym_core.multiscale_sample import MAGICALSampleAtlas
from temgym_core.neoarm_calibration import (
    DetectorGeometry, apply_transfer_matrix_to_gaussians, reexpand_transmitted_field,
)


@dataclass(frozen=True)
class TranslatedActionView:
    """A rigid specimen displacement; the atlas raster remains unchanged."""
    view: object
    translation_xy_m: object

    def evaluate_actions(self, points):
        return self.view.evaluate_actions(points-jnp.asarray(self.translation_xy_m))


def fitting_box(nominal_magnification, measured_rotation_rad):
    """Acquisition-scale bounds independent of generating image parameters."""
    centre = np.array([np.log(nominal_magnification), measured_rotation_rad,
                       0., 0., np.log(2e7), np.log(110e-9), 0.])
    width = np.array([np.log(1.1), .15, 20., 20., np.log(20.), np.log(2.), np.log(5.)])
    return centre-width, centre+width


def box_coverage(view, detector, bounds, *, grid_halfwidth_m, source_centre):
    """Analytic enclosure for all rotations, magnifications and translations.

    Each local corner coordinate is a*cos(angle)+b*sin(angle). Evaluate
    endpoints and its stationary angles, then the smallest magnification;
    add the exact translation-box margins. No sampled-angle coverage claim.
    """
    lower, upper = map(np.asarray, bounds)
    halo = int(np.ceil(4*detector.mtf_sigma_px))
    half = (np.asarray(detector.shape_yx)[::-1]/2+halo)*np.asarray(detector.pixel_size_yx_m)[::-1]
    corners = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])*half
    optical = corners@np.linalg.inv(detector.affine_xy).T+detector.centre_xy_m
    radius = np.linalg.norm(optical, axis=1).max()*np.exp(-lower[0])
    transform = np.linalg.inv(np.asarray(view.projection_world_from_local))
    maxima = np.zeros(2)
    for point in optical:
        rotated_point = np.array([point[1], -point[0]])
        for axis in range(2):
            a, b = transform[axis]@point, transform[axis]@rotated_point
            critical = np.arctan2(b, a)
            candidates = [lower[1], upper[1]]
            start = int(np.ceil((lower[1]-critical)/np.pi))
            end = int(np.floor((upper[1]-critical)/np.pi))
            candidates.extend(critical+k*np.pi for k in range(start, end+1))
            maxima[axis] = max(maxima[axis], max(abs(a*np.cos(t)+b*np.sin(t)) for t in candidates))
    tlo, thi = lower[2:4]*1e-9, upper[2:4]*1e-9
    midpoint = -.5*(tlo+thi)
    local_midpoint = transform@midpoint
    local_radius = maxima*np.exp(-lower[0])+abs(transform)@(.5*(thi-tlo))
    xmin, xmax, ymin, ymax = view.domain_bounds_xyxy_m
    slack = np.r_[local_midpoint-local_radius-[xmin, ymin], [xmax, ymax]-local_midpoint-local_radius]
    if not np.isfinite(slack).all() or slack.min() < 0:
        raise ValueError("Cubic view does not enclose the complete image-fitting box.")
    grid = np.asarray(source_centre)+grid_halfwidth_m*np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    grid_clearance = require_view_coverage(view, grid, label="frozen Gaussian grid")
    return dict(full_image_fitting_box_covered=True,
        minimum_local_box_clearance_m=float(slack.min()),
        maximum_inverse_detector_radius_m=float(radius),
        gaussian_grid_clearance_m=grid_clearance,
        scope="Fixed cubic crop and level; not uniform Gaussian-error validation over every fitted parameter.")


def measured_initializer(observed, detector, response, kappa, bounds):
    """Use measured calibration for orientation, detector moments for width.

    No generating expectation, fitted optical posterior or truth parameters
    are accepted by this interface. Calibration enters initialization only.
    """
    image = np.asarray(observed, float)
    if image.shape != detector.shape_yx or not np.isfinite(image).all():
        raise ValueError("A finite detector image with the declared shape is required.")
    endpoint = EndpointRows(kappa)
    encoded = endpoint.encode_response(response).reshape(2, 3)
    magnification = float(np.mean(abs(encoded[:, 0])*endpoint.pixel_m/endpoint.steps[0]))
    angle = float(np.angle(np.mean(np.exp(1j*encoded[:, 2]))))
    weights = np.maximum(image, 0.)
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Image initialization requires positive measured signal.")
    coords = np.asarray(detector.optical_coords_xy_m)
    centre = np.sum(coords*weights[..., None], axis=(0, 1))/total
    radius2 = float(np.sum(weights*np.sum((coords-centre)**2, axis=-1))/total)
    waist = np.sqrt(2*radius2)/magnification
    p = np.array([np.log(magnification), angle, 0., 0., np.log(total), np.log(waist), 0.])
    lower, upper = map(np.asarray, bounds)
    return np.clip(p, lower+.02*(upper-lower), upper-.02*(upper-lower))


def fit_image(observed, predictor, initial, bounds, *, read_noise_e=.6):
    """Feasible GLS with observed initialization and fitted Poisson weights."""
    data = jnp.asarray(observed)
    p = jnp.asarray(initial)
    variance = jnp.maximum(data, 0.)+read_noise_e**2
    history = []
    for iteration in range(5):
        fixed_sigma = jnp.sqrt(variance)
        fit = least_squares(lambda x: ((predictor(x)-data)/fixed_sigma).ravel(), p,
            bounds=bounds, x_scale=PARAMETER_SCALES, max_nfev=120,
            gtol=1e-6, ftol=1e-12, xtol=1e-11)
        next_variance = jnp.maximum(predictor(fit.x), 0.)+read_noise_e**2
        change = float(jnp.max(abs(next_variance/variance-1)))
        history.append(dict(iteration=iteration, optimizer_success=bool(fit.success),
            termination=fit.status, maximum_relative_weight_change=change,
            scaled_gradient=float(fit.optimality)))
        p, variance = fit.x, next_variance
        if iteration >= 1 and change < 1e-5:
            break
    residual = (predictor(p)-data).ravel()/jnp.sqrt(variance).ravel()
    jacobian = np.asarray(jax.jacfwd(predictor)(p)).reshape(-1, 7)/np.sqrt(np.asarray(variance)).ravel()[:, None]
    _, singular, vh = np.linalg.svd(jacobian*PARAMETER_SCALES, full_matrices=False)
    rank = int(np.sum(singular > singular[0]*np.finfo(float).eps*max(jacobian.shape)))
    inverse = PARAMETER_SCALES[:, None]*vh.T/singular
    covariance = inverse@inverse.T
    gradient = jacobian.T@np.asarray(residual)
    normalized_gradient = np.max(abs(gradient)/np.linalg.norm(jacobian, axis=0))
    accepted = bool(fit.success and rank == 7 and change < 1e-5 and normalized_gradient < 1e-4)
    return dict(accepted=accepted, parameters=np.asarray(p).tolist(), covariance=covariance.tolist(),
        parameter_standard_errors=np.sqrt(np.diag(covariance)).tolist(), rank=rank,
        normalized_gradient=float(normalized_gradient),
        data_rms_sigma=float(np.sqrt(np.mean(np.asarray(residual)**2))), history=history), np.asarray(variance)


def run(output):
    if jax.default_backend() != "gpu" or not jax.config.x64_enabled:
        raise RuntimeError("The material forward simulation and fit require GPU float64.")
    started = perf_counter()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    design_path = FOLDER/"shared_material_design.json"
    calibration_path = FOLDER/"condenser_endpoint_calibration.json"
    design, calibration = [json.loads(p.read_text()) for p in (design_path, calibration_path)]
    sources = {}
    for report in (design, calibration):
        verify_provenance_snapshot(report["source_hashes"])
        sources.update(report["source_hashes"])
    array_path = ROOT/calibration["arrays_path"]
    if digest(array_path) != calibration["arrays_sha256"]:
        raise ValueError("Measured calibration archive changed.")
    with np.load(array_path, allow_pickle=False) as data:
        response = data["response"]
    for path in (Path(__file__), ROOT/"tests/test_neoarm_shared_material_images.py",
                 design_path, calibration_path, array_path,
                 Path(__file__).with_name("neoarm_finite_pixel_specimen.py")):
        sources[str(path.relative_to(ROOT))] = digest(path)
    config = tomllib.loads((ROOT/"examples/lens_inversion/neoarm_inversion_config.toml").read_text())
    atlas = MAGICALSampleAtlas.from_config(config,
        region_shape_yx=tuple(config["sample"]["region_shape_yx"]),
        authoritative_multilayer_shape_yx=tuple(config["sample"]["authoritative_multilayer_shape_yx"]),
        global_support_margin_m=config["sample"]["atlas_support_margin_m"])
    matrix = np.asarray(design["canonical_sample_to_detector"])
    detector = DetectorGeometry(mtf_sigma_px=.65, z_m=float(design["fixed_optical_parameters"][1]))
    source = np.asarray(atlas.pose_translation_xy_m)
    waist, dose, read_noise, halfwidth = 110e-9, 2e7, .6, 275e-9
    stage_translation = np.array([3e-9, -4e-9])
    view, footprint, _ = frozen_measured_view(atlas, matrix, detector, source, halfwidth, 0.)
    generating_view = TranslatedActionView(view, stage_translation)
    kappa = calibration["conditional_sample_kappa_per_m"]
    encoded = EndpointRows(kappa).encode_response(response).reshape(2, 3)
    rotation_seed = float(np.angle(np.mean(np.exp(1j*encoded[:, 2]))))
    bounds = fitting_box(40000., rotation_seed)
    coverage = box_coverage(view, detector, bounds, grid_halfwidth_m=halfwidth, source_centre=source)
    grid_corners = source+halfwidth*np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    coverage["shifted_generator_grid_clearance_m"] = require_view_coverage(
        view, grid_corners-stage_translation, label="translated specimen Gaussian grid")
    predictors = {order: make_finite_pixel_conjugate_predictor(view, detector, source,
        reference_matrix=matrix, wavelength_m=float(energy2wavelength(200e3)),
        shortest_scale_m=view.selection.selected_spacing_m, quadrature_order=order)[0]
        for order in (4, 8, 16)}
    a = matrix[np.ix_(XY, XY)]
    truth = np.array([np.log(np.sqrt(np.linalg.det(a))), np.arctan2(a[1, 0], a[0, 0]),
        *((np.asarray(view.centre_xy_m)+stage_translation)/1e-9), np.log(dose), np.log(waist), 0.])
    reference = np.asarray(predictors[16](truth))
    sigma = np.sqrt(reference+read_noise**2)
    records, images = [], []
    for side in (1621, 2293, 3241):
        begin = perf_counter()
        print(f"GPU {side**2:,} beamlets; primary beam arrays about {side**2*160/2**30:.2f} GiB; "
              "three finite-pixel quadratures and exhaustive point checks.", flush=True)
        axes = [centre+np.linspace(-halfwidth, halfwidth, side) for centre in source]
        xx, yy = jnp.meshgrid(*map(jnp.asarray, axes), indexing="xy")
        incident = np.sqrt(2/np.pi)/waist*jnp.exp(-((xx-source[0])**2+(yy-source[1])**2)/waist**2)
        beam, _ = reexpand_transmitted_field(incident, generating_view, *axes, voltage=200e3, z_m=0.,
            width_over_step_yx=(1.25, 1.25), coefficient_method="gpu_kernel_required")
        downstream = apply_transfer_matrix_to_gaussians(beam, matrix, z_out_m=detector.z_m, axial_path_m=0.)
        lattice = RegularGaussianLattice.from_beam(downstream, (side, side))
        counts = {order: np.asarray(integrated_gaussian_counts(lattice, detector,
            incident_electrons=dose, quadrature_order=order)) for order in (4, 8, 16)}
        coords, _, _, _ = detector_quadrature(detector, 8)
        indices = np.random.default_rng(81932).integers(0, coords.shape[0]*coords.shape[1], size=32)
        selected = jnp.asarray(coords.reshape(-1, 2)[indices])
        exhaustive = np.asarray(evaluate_gaussians(downstream, SimpleNamespace(coords=selected, shape=(32,)),
            method="gpu_kernel_required"))
        fast = np.asarray(lattice(selected))
        wider = np.asarray(lattice(selected, radius=lattice.radius+2))
        count_scale = dose*np.prod(detector.pixel_size_yx_m)
        sampled_error = (count_scale*(abs(fast)**2-abs(exhaustive)**2))/np.sqrt(count_scale*abs(exhaustive)**2+read_noise**2)
        row = dict(beam_count=side**2, grid_shape_yx=[side, side], spacing_m=float(axes[0][1]-axes[0][0]),
            axis_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in axes],
            neighbourhood=lattice.validation, gaussian_reference_rms_sigma=float(np.sqrt(np.mean(((counts[16]-reference)/sigma)**2))),
            gaussian_reference_joint_sigma=float(np.linalg.norm((counts[16]-reference)/sigma)),
            pixel4_to8_joint_sigma=float(np.linalg.norm((counts[4]-counts[8])/sigma)),
            pixel8_to16_joint_sigma=float(np.linalg.norm((counts[8]-counts[16])/sigma)),
            exhaustive_point_count=32, maximum_exhaustive_point_error_sigma=float(np.max(abs(sampled_error))),
            complex_exhaustive_relative_error=float(np.linalg.norm(fast-exhaustive)/np.linalg.norm(exhaustive)),
            wider_neighbour_relative_error=float(np.linalg.norm(fast-wider)/np.linalg.norm(wider)),
            elapsed_seconds=perf_counter()-begin)
        row["terminal_intensity_gate_passed"] = bool(row["gaussian_reference_joint_sigma"] < .2
            and row["pixel8_to16_joint_sigma"] < .1 and row["maximum_exhaustive_point_error_sigma"] < .2
            and row["wider_neighbour_relative_error"] < 1e-9)
        records.append(row); images.append(counts[16]); print(json.dumps(row), flush=True)
        del beam, downstream, lattice, incident, xx, yy
    numerical_accepted = all(r["terminal_intensity_gate_passed"] for r in records)
    seed = 9871941
    rng = np.random.default_rng(seed)
    expected = images[-1]
    observed = rng.poisson(np.maximum(expected, 0.))+rng.normal(0., read_noise, expected.shape) if numerical_accepted else None
    fits, variances = [], []
    if numerical_accepted:
        initial = measured_initializer(observed, detector, response, kappa, bounds)
        for image in images:
            fit, variance = fit_image(image, predictors[8], initial, bounds)
            fits.append(fit); variances.append(variance)
        noisy_fit, noisy_variance = fit_image(observed, predictors[8], initial, bounds)
        fitted = np.asarray(noisy_fit["parameters"])
        noisy_fit["postfit_truth_errors_sigma"] = ((fitted-truth)/noisy_fit["parameter_standard_errors"]).tolist()
        noisy_fit["magnification_relative_error"] = float(np.expm1(fitted[0]-truth[0]))
        noisy_fit["rotation_error_degrees"] = float(np.rad2deg(fitted[1]-truth[1]))
        for index, fit in enumerate(fits):
            p, error = np.asarray(fit["parameters"]), np.asarray(fit["parameter_standard_errors"])
            fit["postfit_truth_bias_sigma"] = ((p-truth)/error).tolist()
            fit["previous_refinement_shift_sigma"] = None if index == 0 else ((p-fits[index-1]["parameters"])/error).tolist()
        transitions = [max(abs(np.asarray(f["previous_refinement_shift_sigma"]))) for f in fits[1:]]
        accepted = bool(noisy_fit["accepted"] and all(f["accepted"] for f in fits)
            and max(transitions) < .1 and max(abs(np.asarray(fits[-1]["postfit_truth_bias_sigma"]))) < .2
            and .8 < noisy_fit["data_rms_sigma"] < 1.2)
    else:
        initial, noisy_fit, noisy_variance, accepted = None, None, None, False
    arrays = dict(expected_counts_yx=expected, cubic_reference_counts_yx=reference,
        refinement_expected_counts_yx=np.asarray(images), canonical_transfer_matrix=matrix,
        controls=np.asarray(design["controls"]), footprint_sample_xy_m=footprint)
    if observed is not None:
        arrays.update(noisy_counts_yx=observed, fitted_counts_yx=np.asarray(predictors[8](jnp.asarray(noisy_fit["parameters"]))),
            fitted_variance_yx=noisy_variance)
    image_path = output.with_suffix(".npz")
    np.savez_compressed(image_path, **arrays)
    verify_provenance_snapshot(sources)
    report = dict(schema="neoarm-shared-column-material-images-v1", backend=jax.default_backend(),
        device_kind=jax.devices()[0].device_kind, source_hashes=sources, arrays_path=str(image_path.relative_to(ROOT)),
        arrays_sha256=digest(image_path), all_passed=accepted, numerical_intensity_gates_passed=numerical_accepted,
        parameter_names=PARAMETER_NAMES, initial_parameters=None if initial is None else initial.tolist(),
        fitting_bounds=np.asarray(bounds).tolist(), truth_parameters_postfit_only=truth.tolist(),
        generation_uses_shared_condenser_column=True, physical_projector_or_condenser_parameters_fitted=False,
        illumination_is_prescribed_sample_plane_probe=True, fitted_condenser_source_propagated=False,
        canonical_input_probe=True, fixed_cubic_selection=asdict(view.selection), coverage=coverage,
        source_centre_xy_m=source.tolist(), source_waist_m=waist, source_grid_halfwidth_m=halfwidth,
        specimen_rigid_displacement_xy_m=stage_translation.tolist(), refinements=records,
        noiseless_fits=fits, noisy_fit=noisy_fit, acquisition_seed=seed, incident_electrons=dose,
        acquisition_noise_drawn=observed is not None, read_noise_e=read_noise,
        fit_uses_generating_weights=False, measured_calibration_used_only_for_initialization=True,
        elapsed_seconds=perf_counter()-started,
        limitations=["One shared-objective conjugate40k state; objective field and detector/certificate geometry conditional.",
            "The same invariant synthetic atlas contains nominal contrasts, not a real specimen certificate.",
            "Terminal-intensity Gaussian/cubic reference; complex-field and nonparaxial physical validity are separate.",
            "Full fitting-box cubic coverage is certified; Gaussian numerical errors sampled at the generating optical state.",
            "Seven image parameters, not a full-column lens recovery; prescribed illumination has not been inferred from condenser data."])
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(all_passed=accepted, numerical_gates=numerical_accepted,
        noisy_fit=noisy_fit, elapsed_seconds=report["elapsed_seconds"])), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    run(args.output)
