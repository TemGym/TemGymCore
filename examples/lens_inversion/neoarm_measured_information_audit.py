"""Measured-diagram condenser reconnaissance without a fictitious rotation tag.

Only Gaussian second moments and local information matrices are computed here;
this is not a detector-image renderer or hardware-ready acquisition recipe.
In particular an equivalent B/A diversity must be supplied by the actual
downstream control solution.  The 100 um case is merely an upper bound.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tomllib

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erfc

from temgym_core.constants import energy2wavelength
from temgym_core.critical_line_calibration import (
    EntranceCovarianceChart, aperture_edge_scan_from_moments,
    hard_edge_matrix_to_plane,
)
from temgym_core.neoarm_calibration import DetectorGeometry, hard_edge_column_from_config
from neoarm_imaging_recipe import column_with_geometry_anchors, geometry_anchors_from_notebook
from neoarm_imaging_information_audit import profile_targets, spatial_summary


ROOT = Path(__file__).resolve().parents[2]
DIAGRAM = ROOT / "examples/microscope_models/measure_microscope_geometry.ipynb"
CONFIG = ROOT / "examples/lens_inversion/neoarm_inversion_config.toml"
MICROSCOPE_SETTINGS = ROOT / "examples/microscope_models/microscope.toml"


def measured_column():
    return column_with_geometry_anchors(
        hard_edge_column_from_config(tomllib.loads(CONFIG.read_text())), DIAGRAM
    )


def source_covariance():
    """A physical, slightly non-round synthetic source, not a calibrated tag.

    Nominal source-plane RMS widths are 5/6 nm and angular widths 50/45 urad.
    Both emittances exceed lambda/(4pi) at 200 kV.  All ten covariance
    directions are subsequently profiled, as is the source reference plane.
    """
    return np.diag(np.array((5e-9, 50e-6, 6e-9, 45e-6)) ** 2)


def symplectic_emittances(covariance):
    omega = np.kron(np.eye(2), np.array(((0., 1.), (-1., 0.))))
    return np.sort(np.abs(np.linalg.eigvals(omega @ covariance)))[::2]


def control_for_q(column, q):
    """Provisional DAC coordinates, never a conversion to real NEOARM commands."""
    control = jnp.zeros(len(column.channel_names))
    for index, name in enumerate(("CL1", "CL3", "C-mini")):
        lens = next(lens for lens in column.lenses if lens.name == name)
        control = control.at[lens.channel_index].set(
            lens.reference_dac + (q[index] - lens.q_bias) / lens.q_slope_per_dac
        )
    return control


def capture_lower_bound(covariance_xy, *, magnification=40000.):
    """Conservative centered-square capture from exact Gaussian marginal tails.

    No image is rendered.  The union bound is valid for rotated/correlated
    ellipses.  It includes the declared Gaussian detector MTF in quadrature.
    """
    half_width = 128 * 55e-6 / magnification
    variance = np.diagonal(np.asarray(covariance_xy), axis1=-2, axis2=-1)
    sigma = np.sqrt(variance + (0.65 * 55e-6 / magnification) ** 2)
    return np.maximum(0., 1. - np.sum(erfc(half_width / (np.sqrt(2.) * sigma)), axis=-1))


def search_illumination_candidates(column=None, *, grid_points=45,
                                   diversity_m=100e-6, target_b_m=0.0007):
    """Find low-order focus-band candidates in the measured geometry.

    CL1/CL3 scan a declared synthetic q rectangle; C-mini is solved on its
    first sign-changing root below 1.2 rad.  This is coarse reconnaissance,
    not branch certification or D-optimal design. No old covariance is reused.
    """
    column = measured_column() if column is None else column
    parameters = jnp.asarray(column.nominal_parameters)
    covariance = jnp.asarray(source_covariance())
    aperture_z = geometry_anchors_from_notebook(DIAGRAM)["C-aperture"]

    def matrix(q):
        return column.matrix(parameters, control_for_q(column, q), segment="source_to_sample")

    def unrotated_b(q):
        transport = matrix(q)
        prefield = next(lens.q_bias for lens in column.lenses if lens.name == "Obj-prefield")
        rotation = jnp.sum(q) + prefield
        return transport[0, 1] * jnp.cos(rotation) - transport[2, 1] * jnp.sin(rotation)

    pairs = np.array(np.meshgrid(np.linspace(.02, .55, grid_points),
                                np.linspace(.02, .45, grid_points))).reshape(2, -1).T
    qmini = np.linspace(1e-5, 1.2, 129)
    grid = np.stack((np.broadcast_to(pairs[:, 0, None], (len(pairs), len(qmini))),
                     np.broadcast_to(pairs[:, 1, None], (len(pairs), len(qmini))),
                     np.broadcast_to(qmini, (len(pairs), len(qmini)))), axis=-1)
    response = np.asarray(jax.jit(jax.vmap(unrotated_b))(jnp.asarray(grid.reshape(-1, 3))))
    response = response.reshape(len(pairs), len(qmini)) - target_b_m
    crossing = response[:, :-1] * response[:, 1:] <= 0
    eligible = crossing.any(axis=1)
    first = crossing.argmax(axis=1)[eligible]
    selected_pairs = jnp.asarray(pairs[eligible])

    @jax.jit
    def refine(lower, upper):
        def residual(q):
            return jax.vmap(unrotated_b)(jnp.column_stack((selected_pairs, q))) - target_b_m

        def step(_, bounds):
            lower, upper = bounds
            midpoint = (lower + upper) / 2
            left = residual(lower) * residual(midpoint) <= 0
            return jnp.where(left, lower, midpoint), jnp.where(left, midpoint, upper)

        lower, upper = jax.lax.fori_loop(0, 42, step, (lower, upper))
        return jnp.column_stack((selected_pairs, (lower + upper) / 2))

    q = np.asarray(refine(jnp.asarray(qmini[first]), jnp.asarray(qmini[first + 1])))
    sample_matrix = np.asarray(jax.jit(jax.vmap(matrix))(q))
    sample_covariance = sample_matrix @ np.asarray(covariance) @ sample_matrix.transpose(0, 2, 1)
    aperture_matrix = np.asarray(jax.jit(jax.vmap(lambda value: hard_edge_matrix_to_plane(
        column, parameters, control_for_q(column, value), aperture_z)))(q))
    aperture_covariance = aperture_matrix @ np.asarray(covariance) @ aperture_matrix.transpose(0, 2, 1)
    aperture_rms = np.sqrt(np.linalg.eigvalsh(aperture_covariance[:, (0, 2)][:, :, (0, 2)]))
    projected = []
    for distance in (-diversity_m, 0., diversity_m):
        drift = np.eye(4); drift[0, 1] = distance; drift[2, 3] = distance
        transformed = drift @ sample_covariance @ drift.T
        projected.append(transformed[:, (0, 2)][:, :, (0, 2)])
    projected = np.stack(projected, axis=1)
    widths = np.sqrt(np.linalg.eigvalsh(projected))
    capture = capture_lower_bound(projected)
    keep = ((capture.min(axis=1) >= .995) & (widths.min(axis=(1, 2)) >= 5e-9)
            & (aperture_rms.min(axis=1) >= 3e-6) & (aperture_rms.max(axis=1) <= 30e-6))
    return {
        "q": q[keep],
        "controls": np.asarray(jax.vmap(lambda value: control_for_q(column, value))(q[keep])),
        "sample_covariance": sample_covariance[keep],
        "projected_eigen_rms_m": widths[keep],
        "minimum_capture_probability": capture[keep].min(axis=1),
        "aperture_eigen_rms_m": aperture_rms[keep],
        "searched_pair_count": len(pairs), "root_count": len(q),
        "diversity_m": diversity_m,
    }


def select_spread_candidates(candidates, count=16):
    """Space-filling subset of the focus band; no held-out or truth-fit score."""
    q = np.asarray(candidates["q"])
    if len(q) < count:
        raise ValueError(f"Only {len(q)} eligible candidates for requested {count} states.")
    scaled = (q - q.min(axis=0)) / np.maximum(np.ptp(q, axis=0), 1e-12)
    selected = [int(np.argmin(np.linalg.norm(scaled - .5, axis=1)))]
    distance = np.full(len(q), np.inf)
    for _ in range(count - 1):
        distance = np.minimum(distance, np.linalg.norm(scaled - scaled[selected[-1]], axis=1))
        distance[selected] = -np.inf
        selected.append(int(np.argmax(distance)))
    return np.asarray(selected)


def profile_measured_targets(matrix):
    """Distinguish numerical nulls, a relative reporting threshold, and noise."""
    matrix = np.asarray(matrix)
    result = profile_targets(matrix)
    nuisance = matrix[:, 12:]
    norms = np.linalg.norm(nuisance, axis=0)
    active = norms > np.finfo(float).tiny
    left, singular, _ = np.linalg.svd(nuisance[:, active] / norms[active], full_matrices=False)
    rank = np.sum(singular > 1e-10 * singular[0])
    target = matrix[:, :12] - left[:, :rank] @ (left[:, :rank].T @ matrix[:, :12])
    _, singular, right = np.linalg.svd(target, full_matrices=False)
    tolerance = np.finfo(float).eps * max(matrix.shape) * singular[0]
    numerical_rank = int(np.sum(singular > tolerance))
    result.update(
        numerical_target_rank=numerical_rank,
        numerical_rank_tolerance=float(tolerance),
        noise_resolved_target_rank=int(np.sum(singular > 1.)),
        relative_reporting_rank=result["target_rank"],
        rank_warning="target_rank uses legacy relative1e-7 threshold, not proof of an exact null; numerical/noise-resolved ranks are reported separately",
    )
    if numerical_rank == 12:
        uncertainty = np.sqrt(np.sum(right.T**2 / singular**2, axis=1))
        result["numerically_full_rank_maximum_scaled_sigma"] = float(uncertainty.max())
        result["numerically_full_rank_scaled_sigmas"] = uncertainty.tolist()
    return result


def pixel_information_rows(spatial_values, spatial_jacobian, rotations, *, dose_electrons,
                           projection_scale=40000., quadrature_order=8):
    """GPU finite-pixel information, with per-image centre and dose profiled."""
    from temgym_core.gaussian_image_likelihood import (
        encode_gaussian_image_parameters, gaussian_image_information_factor,
    )
    if jax.default_backend() != "gpu":
        raise RuntimeError("Raw detector-pixel information requires GPU forward simulation.")
    detector = DetectorGeometry(mtf_sigma_px=.65)

    def encode(summary, rotation):
        sigma = jnp.exp(summary[:2])
        covariance = jnp.array(((sigma[0]**2, summary[2] * jnp.prod(sigma)),
                                (summary[2] * jnp.prod(sigma), sigma[1]**2)))
        cosine, sine = jnp.cos(rotation), jnp.sin(rotation)
        transform = projection_scale * jnp.array(((cosine, -sine), (sine, cosine)))
        return encode_gaussian_image_parameters(jnp.zeros(2), transform @ covariance @ transform.T,
                                                dose_electrons, detector)

    values = jnp.asarray(spatial_values)
    rotations = jnp.asarray(rotations)
    encoded = np.asarray(jax.vmap(encode)(values, rotations))
    encoded_derivative = np.asarray(jax.vmap(jax.jacfwd(encode))(values, rotations))
    information = jax.jit(jax.vmap(lambda p: gaussian_image_information_factor(
        p, detector, read_noise_electrons=.6, quadrature_order=quadrature_order)))
    rows = []
    for start in range(0, len(encoded), 4):
        factors = np.asarray(information(jnp.asarray(encoded[start:start + 4])))
        for local_index, factor in enumerate(factors):
            index = start + local_index
            nuisance = factor[:, (0, 1, 5)]
            left, singular, _ = np.linalg.svd(nuisance, full_matrices=False)
            rank = np.sum(singular > 1e-10 * singular[0])
            retained = factor - left[:, :rank] @ (left[:, :rank].T @ factor)
            rows.append(retained @ encoded_derivative[index] @ spatial_jacobian[index])
    return np.asarray(rows).reshape(-1, spatial_jacobian.shape[-1])


def angular_information_rows(column, controls, *, pixel_information, dose_electrons=20e6):
    """Conditional calibrated 0.4 m Fourier-plane information, not a control solution."""
    nominal = jnp.asarray(column.nominal_parameters)
    names = tuple(f"{lens}.{field}" for lens in ("CL1", "CL3", "C-mini")
                  for field in ("z_center_m", "log_length_m", "q_bias", "q_slope_per_dac"))
    indices = jnp.asarray([column.parameter_names.index(name) for name in names])
    pre = jnp.asarray([column.parameter_names.index("Obj-prefield." + field)
                      for field in ("z_center_m", "log_length_m", "q_bias")])
    chart = EntranceCovarianceChart(source_covariance())
    angle = jnp.array((1, 3))

    def predict(values):
        parameters = nominal.at[indices].add(values[:12] * jnp.array([.01, 1., 1., 1.] * 3))
        parameters = parameters.at[0].add(.01 * values[22])
        parameters = parameters.at[pre].add(values[23:26] * jnp.array((.01, 1., 1.)))
        covariance = chart.covariance(values[12:22])

        def at_state(control):
            matrix = column.matrix(parameters, control, segment="source_to_sample")
            transported = matrix @ covariance @ matrix.T
            return spatial_summary(transported[jnp.ix_(angle, angle)]).at[:2].add(values[42])

        return jax.vmap(at_state)(jnp.asarray(controls))

    nominal_summary = np.asarray(predict(jnp.zeros(43)))
    derivative = np.asarray(jax.jacfwd(predict)(jnp.zeros(43)))
    if pixel_information:
        rows = pixel_information_rows(nominal_summary, derivative, np.zeros(len(controls)),
                                      dose_electrons=dose_electrons, projection_scale=.4,
                                      quadrature_order=32)
    else:
        rows = derivative.reshape(-1, 43) / .002
    prior = np.zeros((1, 43)); prior[0, 42] = 1 / .002
    return np.vstack((rows, prior)), {
        "scope": "independently calibrated Fourier-plane upper bound, no actual diffraction controls established",
        "camera_length_m": .4, "angular_scale_prior_sigma": .002,
        "angular_rms_pixel_range": (np.exp(nominal_summary[:, :2]) * .4 / 55e-6).min(axis=0).tolist()
            + (np.exp(nominal_summary[:, :2]) * .4 / 55e-6).max(axis=0).tolist(),
        "n_angular_frames": len(controls), "image_backend": jax.default_backend() if pixel_information else "not_rendered",
        "quadrature_order": 32 if pixel_information else None,
    }


def measured_information_matrix(column, controls, diversities_m, *,
                                summary_sigma=.002, dose_electrons=20e6,
                                pixel_information=False, projection_rotations_rad=None):
    """Image-covariance and centered total-count Jacobian with honest nuisances.

    Source covariance(10), source reference z, objective prefield(3), image
    scale, B/A scale, each aperture radius, aperture z, source mean(4), and
    each aperture centre(2) are released. The count block remains optimistic:
    it assumes all transmitted electrons are counted, not merely those inside
    a finite 40k detector window. There is no standalone Larmor-angle datum.
    """
    controls = jnp.asarray(controls)
    diversities = jnp.asarray(diversities_m)
    nominal = jnp.asarray(column.nominal_parameters)
    target_names = tuple(f"{lens}.{field}" for lens in ("CL1", "CL3", "C-mini")
                         for field in ("z_center_m", "log_length_m", "q_bias", "q_slope_per_dac"))
    target_indices = jnp.asarray([column.parameter_names.index(name) for name in target_names])
    scales = jnp.asarray([.01, 1., 1., 1.] * 3)
    pre_indices = jnp.asarray([column.parameter_names.index("Obj-prefield." + name)
                              for name in ("z_center_m", "log_length_m", "q_bias")])
    chart = EntranceCovarianceChart(source_covariance())
    aperture_z = geometry_anchors_from_notebook(DIAGRAM)["C-aperture"]
    available_radii = tomllib.loads(MICROSCOPE_SETTINGS.read_text())["apertures"]["C_aperture"]["radii_m"]
    if len(available_radii) < 3 or np.any(np.asarray(available_radii[:3]) <= 0):
        raise ValueError("Three positive aperture radii are needed for this ablation.")
    radii = jnp.asarray(available_radii[:3])
    xy = jnp.array((0, 2))
    source_mean_scales = jnp.array((5e-9, 50e-6, 5e-9, 50e-6))
    centre_nominal = jnp.array(((.5e-6, -.3e-6), (.4e-6, .2e-6), (-.2e-6, .4e-6)))

    def predict(values):
        parameters = nominal.at[target_indices].add(values[:12] * scales)
        parameters = parameters.at[0].add(.01 * values[22])
        parameters = parameters.at[pre_indices].add(values[23:26] * jnp.array((.01, 1., 1.)))
        covariance = chart.covariance(values[12:22])
        mean = values[32:36] * source_mean_scales

        def at_state(control):
            sample_matrix = column.matrix(parameters, control, segment="source_to_sample")
            sample_covariance = sample_matrix @ covariance @ sample_matrix.T

            def at_diversity(distance):
                drift = jnp.eye(4).at[0, 1].set(distance * jnp.exp(values[27]))
                drift = drift.at[2, 3].set(distance * jnp.exp(values[27]))
                projected = drift @ sample_covariance @ drift.T
                return spatial_summary(projected[jnp.ix_(xy, xy)]).at[:2].add(values[26])

            aperture_matrix = hard_edge_matrix_to_plane(column, parameters, control, aperture_z)
            drift = jnp.eye(4).at[0, 1].set(.001 * values[31]).at[2, 3].set(.001 * values[31])
            aperture_matrix = drift @ aperture_matrix
            at_aperture = aperture_matrix @ covariance @ aperture_matrix.T
            aperture_mean = (aperture_matrix @ mean)[xy]
            counts = aperture_edge_scan_from_moments(
                jnp.broadcast_to(aperture_mean, (3, 2)), at_aperture[jnp.ix_(xy, xy)],
                aperture_centre_xy_m=centre_nominal + 1e-6 * values[36:42].reshape(3, 2),
                aperture_radius_m=radii * jnp.exp(values[28:31]),
                incident_dose_electrons=dose_electrons,
                radial_order=48, azimuthal_order=128,
            ).expected_transmitted_electrons[:, 0]
            return jax.vmap(at_diversity)(diversities).reshape(-1), jnp.log(counts / dose_electrons)

        spatial, log_counts = jax.vmap(at_state)(controls)
        return jnp.concatenate((spatial.reshape(-1), log_counts.reshape(-1)))

    zero = jnp.zeros(42)
    expected = np.asarray(predict(zero))
    n_spatial = len(controls) * len(diversities_m) * 3
    transmission = np.exp(expected[n_spatial:])
    sigma = np.concatenate((np.full(n_spatial, summary_sigma),
                            np.sqrt(1 / (dose_electrons * transmission) + 1 / dose_electrons)))
    derivative = np.asarray(jax.jacfwd(predict)(zero))
    jacobian = derivative / sigma[:, None]
    if pixel_information:
        if projection_rotations_rad is None:
            raise ValueError("Actual calibrated projection rotations are needed for finite-pixel information.")
        spatial = pixel_information_rows(
            expected[:n_spatial].reshape(-1, 3), derivative[:n_spatial].reshape(-1, 3, 42),
            np.tile(projection_rotations_rad, len(controls)), dose_electrons=dose_electrons,
        )
        jacobian = np.vstack((spatial, jacobian[n_spatial:]))
        n_spatial = len(spatial)
    priors = np.zeros((5, 42))
    priors[0, 26] = 1 / .002
    priors[1, 27] = 1 / .002
    priors[np.arange(2, 5), np.arange(28, 31)] = 1 / .005
    return jacobian, priors, {
        "n_spatial_rows": n_spatial,
        "n_optical_settings": len(controls),
        "n_spatial_images": len(controls) * len(diversities_m),
        "n_aperture_inserted_frames": len(controls) * 3,
        "n_independent_dose_reference_frames": len(controls) * 3,
        "transmission_range": [float(transmission.min()), float(transmission.max())],
        "diversities_m": list(diversities_m),
        "conditional_covariance_summary_sigma": summary_sigma,
        "image_scale_prior_sigma": .002, "diversity_scale_prior_sigma": .002,
        "aperture_radius_prior_sigma": .005,
        "target_names": target_names,
        "source_and_aperture_centres_profiled": True,
        "aperture_plane_parameter_scale_m": .001,
        "total_flux_capture_verified": False,
        "spatial_noise_model": "finite256x256pixels Poisson-plus-readnoise localWLS; perimagecentre/doseprofiled" if pixel_information else "granted covariance-summary precision",
        "image_backend": jax.default_backend() if pixel_information else "not_rendered",
    }


def run_measured_audit(*, grid_points=45, state_count=16, pixel_information=False,
                       off_band_q_steps=(), include_angular=False):
    column = measured_column()
    candidates = search_illumination_candidates(column, grid_points=grid_points)
    selected = select_spread_candidates(candidates, state_count)
    selected_q = candidates["q"][selected]
    extra = []
    for step in off_band_q_steps:
        if not np.isfinite(step) or step <= 0:
            raise ValueError("Off-band q steps must be finite and positive.")
        for lens in range(3):
            for sign in (-1, 1):
                value = selected_q[0].copy(); value[lens] += sign * step
                extra.append(value)
    if extra:
        selected_q = np.vstack((selected_q, extra))
    controls = np.asarray(jax.vmap(lambda q: control_for_q(column, q))(selected_q))
    initial_path = ROOT / "examples/generated/neoarm_inversion/imaging_recipe/measured_initialization.json"
    initialization = json.loads(initial_path.read_text())
    if initialization["geometry_notebook_sha256"] != hashlib.sha256(DIAGRAM.read_bytes()).hexdigest():
        raise ValueError("Initializer was not generated from this measured geometry.")
    routes = {
        "hypothetical_100um": ((-100e-6, 0., 100e-6), (0., 0., 0.)),
    }
    for path in initialization["bounded_imaging_paths"]:
        if path["nominal_magnification"] != 40000.:
            continue
        endpoints = {int(route["direction"]): route["points"][-1]
                     for route in path["paths"] if route["endpoint_established"]}
        if set(endpoints) != {-1, 1}:
            raise ValueError("Both measured control endpoints must be established.")
        nominal = initialization["nominal_metrics"][0]
        metrics = [endpoints[-1]["achieved_metrics"], nominal, endpoints[1]["achieved_metrics"]]
        label = f"measured_control_halfwidth_{path['control_half_width']:g}"
        routes[label] = (tuple(value[1] for value in metrics), tuple(value[2] for value in metrics))
    cases = {}
    angular_rows, angular_contract = angular_information_rows(
        column, controls, pixel_information=pixel_information,
    ) if include_angular else (None, None)
    for label, (distances, rotations) in routes.items():
        matrix, priors, contract = measured_information_matrix(
            column, controls, distances, pixel_information=pixel_information,
            projection_rotations_rad=rotations,
        )
        for fixed_plane in (False, True):
            joined = np.vstack((matrix, priors))
            if fixed_plane:
                joined = np.delete(joined, 31, axis=1)
            key = label + ("_conditional_fixed_aperture_z" if fixed_plane else "_free_aperture_z")
            cases[key] = profile_measured_targets(joined)
            cases[key]["contract"] = contract
        spatial = matrix[:contract["n_spatial_rows"]]
        cases[label + "_without_aperture_counts"] = profile_measured_targets(np.vstack((spatial, priors)))
        if include_angular:
            joined = np.vstack((np.pad(matrix, ((0, 0), (0, 1))),
                                np.pad(priors, ((0, 0), (0, 1))), angular_rows))
            cases[label + "_plus_angular_free_aperture_z"] = profile_measured_targets(joined)
            cases[label + "_plus_angular_conditional_fixed_aperture_z"] = profile_measured_targets(
                np.delete(joined, 31, axis=1))
        # This is explicitly a conditional upper bound, not an imposed prior:
        # it reveals whether an independently known prefield would help.
        cases[label + "_conditional_known_prefield_and_aperture_z"] = profile_measured_targets(
            np.delete(np.vstack((matrix, priors)), (23, 24, 25, 31), axis=1))
        for coordinate, name in ((23, "z"), (24, "length"), (25, "q")):
            cases[label + f"_conditional_known_prefield_{name}_and_aperture_z"] = profile_measured_targets(
                np.delete(np.vstack((matrix, priors)), (coordinate, 31), axis=1))
    return {
        "scope": "local measured-geometry Gaussian-moment and total-flux upper-bound audit, not raw-image recovery",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "diagram_sha256": hashlib.sha256(DIAGRAM.read_bytes()).hexdigest(),
        "initializer_report_sha256": hashlib.sha256(initial_path.read_bytes()).hexdigest(),
        "initializer_source_sha256": initialization["source_sha256"],
        "microscope_settings_sha256": hashlib.sha256(MICROSCOPE_SETTINGS.read_bytes()).hexdigest(),
        "source_covariance": source_covariance().tolist(),
        "source_emittances_m_rad": symplectic_emittances(source_covariance()).tolist(),
        "quantum_minimum_m_rad": float(energy2wavelength(200000.)) / (4 * np.pi),
        "candidate_count": len(candidates["q"]), "selected_indices": selected.tolist(),
        "selected_controls": controls.tolist(), "selected_q": selected_q.tolist(),
        "focus_band_minimum_capture": candidates["minimum_capture_probability"][selected].tolist(),
        "focus_band_aperture_eigen_rms_m": candidates["aperture_eigen_rms_m"][selected].tolist(),
        "off_band_q_steps": list(off_band_q_steps),
        "off_band_states_not_filtered_by_capture": True,
        "angular_measurement_contract": angular_contract,
        "cases": cases, "hardware_ready": False,
        "source_reference_plane_is_a_covariance_transport_gauge": True,
        "pixel_information": bool(pixel_information),
        "geometry_warning": "Nominal CL3-exit/aperture clearance is only 0.833 mm; all drift-plane calculations are conditional on preserving membership. Diagram coordinates and radii are approximate, not certified priors.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--grid-points", type=int, default=45)
    parser.add_argument("--information", action="store_true")
    parser.add_argument("--pixel-information", action="store_true", help="Use GPU finite-pixel Fisher information instead of granted summary noise.")
    parser.add_argument("--off-band-q-steps", type=float, nargs="*", default=[])
    parser.add_argument("--angular-information", action="store_true")
    args = parser.parse_args(argv)
    jax.config.update("jax_enable_x64", True)
    candidates = None if args.information else search_illumination_candidates(grid_points=args.grid_points)
    result = run_measured_audit(grid_points=args.grid_points, pixel_information=args.pixel_information,
                               off_band_q_steps=args.off_band_q_steps,
                               include_angular=args.angular_information) if args.information else {
        "scope": "measured-diagram synthetic condenser moment reconnaissance; no hardware or inversion admission",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "diagram_sha256": hashlib.sha256(DIAGRAM.read_bytes()).hexdigest(),
        "source_covariance": source_covariance().tolist(),
        "source_emittances_m_rad": symplectic_emittances(source_covariance()).tolist(),
        "quantum_minimum_m_rad": float(energy2wavelength(200000.)) / (4 * np.pi),
        "candidate_count": len(candidates["q"]),
        "candidates": {key: value.tolist() if isinstance(value, np.ndarray) else value
                       for key, value in candidates.items()},
        "hardware_ready": False,
        "warning": "Projection diversity is a conditional requirement, not proven reachable in this measured geometry; aperture-position and radius values are approximate, not certified priors.",
    }
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
        print(f"Saved measured-geometry reconnaissance: {args.output}")
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
