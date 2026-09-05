"""Joint, finite-pixel information audit for a shared immersed objective.

The data are ordinary aperture-out Gaussian images, calibrated *mechanical*
sample-plane kicks, and an optional optimistic total-aperture-flux block.
There is no independent ellipse/Larmor-angle observation and no calibrated
upstream beam command. Local information is not nonlinear or branch recovery.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from importlib.metadata import version as distribution_version
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from temgym_core.constants import energy2wavelength
from temgym_core.critical_line_calibration import EntranceCovarianceChart, aperture_edge_scan_from_moments
from temgym_core.gaussian_image_likelihood import (
    encode_gaussian_image_parameters, gaussian_image_information_factor,
    validate_phase_space_covariance,
)
from temgym_core.neoarm_calibration import DetectorGeometry


XY = jnp.array([0, 2])
SOURCE_COVARIANCE = np.diag(np.array([5e-9, 50e-6, 6e-9, 45e-6])**2)
SOURCE_SCALES = np.sqrt(np.diag(SOURCE_COVARIANCE))


def provenance_snapshot(root=None):
    """Hash computational inputs and the conservatively loaded repo closure.

    Imported repository modules are included even when only part of a module
    is used. Third-party code is represented by distribution versions, not
    misleadingly treated as repository source. Capture before computation and
    verify after it so edits during a long GPU run cannot silently rebind data.
    """
    root = Path(__file__).resolve().parents[2] if root is None else Path(root).resolve()
    files = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        if path.is_relative_to(root):
            relative = path.relative_to(root)
            if relative.parts[0] in {"src", "examples"} and path.is_file():
                files.add(path)
    inputs = ("examples/lens_inversion/neoarm_inversion_config.toml",
              "examples/microscope_models/measure_microscope_geometry.ipynb")
    files.update(root/path for path in inputs)
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(files)}
    return hashes, {path: hashes[path] for path in inputs}


def verify_provenance_snapshot(expected, root=None):
    root = Path(__file__).resolve().parents[2] if root is None else Path(root).resolve()
    changed = [name for name, digest in expected.items()
               if not (root/name).is_file() or hashlib.sha256((root/name).read_bytes()).hexdigest() != digest]
    if changed:
        raise RuntimeError("Repository inputs changed during the audit: " + ", ".join(changed))


def information_artifact_metadata(path):
    if path is None:
        return None
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                size_bytes=path.stat().st_size, format="numpy-npz",
                contents="Whitened local information rows and parameter names; no pickle objects.")


@dataclass(frozen=True)
class ImageDesign:
    """One row per frame; alignment is shared only within the stated group.

    ``kick_index=-1`` is the reference state. Other rows index four calibrated
    sample-plane states whose *absolute* 4x4 calibration covariances are given
    by ``calibration_cholesky``. Calibration errors are shared across settings.
    They are measurement uncertainty, not unconstrained command crosstalk.
    """

    controls: np.ndarray
    kick_index: np.ndarray
    alignment_group: np.ndarray
    calibrated_states: np.ndarray
    calibration_cholesky: np.ndarray

    def __post_init__(self):
        n = len(self.controls)
        if np.shape(self.controls) != (n, 7) or not np.all(np.isfinite(self.controls)):
            raise ValueError("Expected finite seven-channel controls for every frame.")
        if np.shape(self.kick_index) != (n,) or np.shape(self.alignment_group) != (n,):
            raise ValueError("Every frame needs a kick and alignment-group index.")
        if not np.all(np.isin(self.kick_index, [-1, 0, 1, 2, 3])):
            raise ValueError("Kick indices must be reference -1 or calibrated state 0..3.")
        if np.shape(self.calibrated_states) != (4, 4) or np.shape(self.calibration_cholesky) != (4, 4, 4):
            raise ValueError("Four physical sample states and four 4x4 calibration factors are required.")
        if not np.all(np.isfinite(self.calibrated_states)) or not np.all(np.isfinite(self.calibration_cholesky)):
            raise ValueError("Sample-state calibration must be finite.")
        if not np.all(np.equal(np.mod(self.alignment_group, 1), 0)) or np.any(self.alignment_group < 0):
            raise ValueError("Alignment-group indices must be nonnegative integers.")


def optical_parameter_scales(column):
    scales = np.array([.01, .01] + [.01, 1., 1., 1.]*7 + [1., .001, 1., 1.])
    if len(column.parameter_names) != len(scales):
        raise ValueError("Expected the 34-parameter shared-objective column.")
    return scales


class JointImagePredictor:
    """Differentiable six-parameter image model, with an explicit nuisance chart.

    All 34 optical parameters are varied jointly. The ten source covariance
    directions and source reference plane are retained, exposing their gauge.
    The exponential SPD covariance chart is only a *local* physical chart;
    nonlinear recovery must also enforce the quantum covariance inequality.
    """

    def __init__(self, column, design, *, parameters=None, detector=None,
                 dose_electrons=20e6, source_covariance=SOURCE_COVARIANCE,
                 aperture_z_m=-.42083333333333334, aperture_radii_m=(10e-6, 20e-6, 30e-6)):
        self.column, self.design = column, design
        self.nominal = jnp.asarray(column.nominal_parameters if parameters is None else parameters)
        self.scales = jnp.asarray(optical_parameter_scales(column))
        self.detector = DetectorGeometry(mtf_sigma_px=.65) if detector is None else detector
        if not np.isfinite(dose_electrons) or dose_electrons <= 0:
            raise ValueError("A positive incident electron dose is required.")
        validate_phase_space_covariance(source_covariance, energy2wavelength(200e3))
        self.chart = EntranceCovarianceChart(source_covariance)
        self.source_mean_scales = jnp.sqrt(jnp.diag(jnp.asarray(source_covariance)))
        self.dose = float(dose_electrons)
        self.aperture_z = float(aperture_z_m)
        self.radii = jnp.asarray(aperture_radii_m)
        if self.radii.shape != (3,) or np.any(np.asarray(self.radii) <= 0):
            raise ValueError("Three positive independently calibrated aperture radii are required.")
        # Fixed indices keep generated information artifacts easy to audit.
        self.covariance_slice = slice(34, 44)
        self.mean_slice = slice(44, 48)
        self.calibration_slice = slice(48, 64)
        self.image_scale_index = 64
        self.radius_slice = slice(65, 68)
        self.aperture_plane_index = 68
        self.aperture_centre_slice = slice(69, 75)
        self.n_parameters = 75
        self.parameter_names = tuple(column.parameter_names) + tuple(
            f"source_covariance.{i}" for i in range(10)) + tuple(
            f"source_mean.{i}" for i in range(4)) + tuple(
            f"sample_calibration.state_{i}.coordinate_{j}" for i in range(4) for j in range(4)
        ) + ("detector.log_scale",) + tuple(f"aperture_{i}.log_radius" for i in range(3)) + (
            "aperture.z_m",) + tuple(f"aperture_{i}.centre_{axis}" for i in range(3) for axis in "xy")

    def decode(self, values):
        values = jnp.asarray(values)
        optical = self.nominal + self.scales*values[:34]
        covariance = self.chart.covariance(values[self.covariance_slice])
        mean = self.source_mean_scales*values[self.mean_slice]
        errors = values[self.calibration_slice].reshape(4, 4)
        kicks = jnp.asarray(self.design.calibrated_states) + jnp.einsum(
            "kij,kj->ki", jnp.asarray(self.design.calibration_cholesky), errors)
        return optical, covariance, mean, kicks

    def __call__(self, values):
        optical, covariance, mean, kicks = self.decode(values)
        scale = jnp.exp(values[self.image_scale_index])
        all_kicks = jnp.concatenate((jnp.zeros((1, 4)), kicks), axis=0)

        def one(control, kick_index):
            full = self.column.matrix(optical, control, segment="full")
            downstream = self.column.matrix(optical, control, segment="sample_to_detector",
                                             coordinates="mechanical")
            position = (full@mean + downstream@all_kicks[kick_index+1])[XY]*scale
            transported = full@covariance@full.T
            spatial = transported[jnp.ix_(XY, XY)]*scale**2
            return encode_gaussian_image_parameters(position, spatial, self.dose, self.detector)

        return jax.vmap(one)(jnp.asarray(self.design.controls), jnp.asarray(self.design.kick_index))

    def log_transmission(self, values, aperture_controls):
        """Optimistic all-electron throughput, NOT finite-camera image counts."""
        optical, covariance, mean, _ = self.decode(values)
        radius = self.radii*jnp.exp(values[self.radius_slice])
        aperture_z = self.aperture_z + .001*values[self.aperture_plane_index]
        centre = jnp.array([[.5e-6, -.3e-6], [.4e-6, .2e-6], [-.2e-6, .4e-6]])
        centre = centre + 1e-6*values[self.aperture_centre_slice].reshape(3, 2)

        def one(control):
            matrix = self.column.matrix_to_plane(optical, control, aperture_z)
            at_aperture = matrix@covariance@matrix.T
            counts = aperture_edge_scan_from_moments(
                jnp.broadcast_to((matrix@mean)[XY], (3, 2)), at_aperture[jnp.ix_(XY, XY)],
                aperture_centre_xy_m=centre, aperture_radius_m=radius,
                incident_dose_electrons=self.dose, radial_order=48, azimuthal_order=128,
            ).expected_transmitted_electrons[:, 0]
            return jnp.log(counts/self.dose)
        return jax.vmap(one)(jnp.asarray(aperture_controls))

    def prior_rows(self, *, detector_scale_sigma=.002, radius_relative_sigma=.005,
                   aperture_plane_sigma_m=None):
        rows = np.zeros((20 + (aperture_plane_sigma_m is not None), self.n_parameters))
        rows[np.arange(16), np.arange(48, 64)] = 1.
        rows[16, 64] = 1/detector_scale_sigma
        rows[np.arange(17, 20), np.arange(65, 68)] = 1/radius_relative_sigma
        if aperture_plane_sigma_m is not None:
            if aperture_plane_sigma_m <= 0:
                raise ValueError("An aperture-plane prior must be positive; do not silently fix truth.")
            rows[20, 68] = .001/aperture_plane_sigma_m
        return rows


def remove_column_space(matrix, nuisance, *, relative_tolerance=1e-10):
    """Stable projection without forming or inverting a Fisher normal matrix."""
    nuisance = np.asarray(nuisance)
    norms = np.linalg.norm(nuisance, axis=0)
    active = norms > np.finfo(float).tiny
    if not np.any(active):
        return np.asarray(matrix)
    left, singular, _ = np.linalg.svd(nuisance[:, active]/norms[active], full_matrices=False)
    rank = np.sum(singular > relative_tolerance*singular[0])
    return matrix-left[:, :rank]@(left[:, :rank].T@matrix)


def profile_information(matrix, names, target_indices):
    """Report weak finite directions separately from exact/numerical nulls."""
    matrix = np.asarray(matrix)
    target_indices = np.asarray(target_indices, int)
    nuisance = np.setdiff1d(np.arange(matrix.shape[1]), target_indices)
    retained = remove_column_space(matrix[:, target_indices], matrix[:, nuisance])
    _, singular, right = np.linalg.svd(retained, full_matrices=False)
    tolerance = np.finfo(float).eps*max(matrix.shape)*(singular[0] if len(singular) else 0.)
    rank = int(np.sum(singular > tolerance))
    report = dict(target_names=[names[i] for i in target_indices], numerical_rank=rank,
                  target_count=len(target_indices), noise_resolved_rank=int(np.sum(singular > 1.)),
                  singular_values=singular.tolist(), numerical_tolerance=float(tolerance),
                  weakest_scaled_direction=right[-1].tolist(), global_uniqueness_proved=False)
    if rank == len(target_indices):
        sigma = np.sqrt(np.sum(right.T**2/singular**2, axis=1))
        report.update(scaled_standard_deviations=sigma.tolist(), maximum_scaled_sigma=float(sigma.max()))
    return report


def finite_pixel_information(predictor, *, quadrature_order=8, batch_size=4):
    """GPU exact local pixel WLS information, per-frame dose profiled.

    Image alignment is shared across reference/basis frames at each optical
    setting; profiling a separate centre per frame would delete the calibrated
    sample response. The full six-parameter pixel metric retains shape/centre
    correlations and finite-detector clipping.
    """
    if jax.default_backend() != "gpu":
        raise RuntimeError("Finite-pixel forward simulation must run on GPU.")
    zero = jnp.zeros(predictor.n_parameters)
    expected = np.asarray(jax.jit(predictor)(zero))
    derivative = np.asarray(jax.jit(jax.jacfwd(predictor))(zero))
    factor = jax.jit(jax.vmap(lambda p: gaussian_image_information_factor(
        p, predictor.detector, read_noise_electrons=.6, quadrature_order=quadrature_order)))
    groups, inverse = np.unique(predictor.design.alignment_group, return_inverse=True)
    rows = np.zeros((len(expected), 6, predictor.n_parameters+2*len(groups)))
    for start in range(0, len(expected), batch_size):
        factors = np.asarray(factor(jnp.asarray(expected[start:start+batch_size])))
        for local, metric in enumerate(factors):
            index = start+local
            retained = remove_column_space(metric, metric[:, 5:6])
            rows[index, :, :predictor.n_parameters] = retained@derivative[index]
            offset = predictor.n_parameters+2*inverse[index]
            rows[index, :, offset:offset+2] = retained[:, :2]
    names = predictor.parameter_names + tuple(f"alignment_group_{g}.{axis}" for g in groups for axis in "xy")
    return rows.reshape(-1, rows.shape[-1]), expected, names


def joint_information(predictor, *, aperture_controls=None, aperture_plane_sigma_m=None,
                      quadrature_order=8, information_sink=None):
    images, expected, names = finite_pixel_information(predictor, quadrature_order=quadrature_order)
    priors = predictor.prior_rows(aperture_plane_sigma_m=aperture_plane_sigma_m)
    priors = np.pad(priors, ((0, 0), (0, images.shape[1]-priors.shape[1])))
    matrix = np.vstack((images, priors))
    cases = {"images_and_calibrated_sample_states": matrix}
    transmission_range = None
    if aperture_controls is not None:
        zero = jnp.zeros(predictor.n_parameters)
        fun = lambda p: predictor.log_transmission(p, aperture_controls).reshape(-1)
        log_t = np.asarray(jax.jit(fun)(zero))
        jac = np.asarray(jax.jit(jax.jacfwd(fun))(zero))
        transmission = np.exp(log_t)
        # One independent aperture-out dose reference for every inserted frame.
        sigma = np.sqrt(1/(predictor.dose*transmission)+1/predictor.dose)
        counts = np.pad(jac/sigma[:, None], ((0, 0), (0, images.shape[1]-jac.shape[1])))
        cases["plus_optimistic_centered_total_flux"] = np.vstack((matrix, counts))
        transmission_range = [float(transmission.min()), float(transmission.max())]
        for precision_m in (.01, .001):
            anchors = np.zeros((2, matrix.shape[1]))
            anchors[0, 1] = .01/precision_m
            anchors[1, predictor.aperture_plane_index] = .001/precision_m
            cases[f"conditional_detector_and_aperture_plane_metrology_{precision_m:g}_m"] = np.vstack(
                (matrix, counts, anchors))
    report = {}
    for label, jac in cases.items():
        report[label] = {
            "condenser": profile_information(jac, names, np.arange(2, 14)),
            "projectors": profile_information(jac, names, np.arange(14, 30)),
            "shared_objective": profile_information(jac, names, np.arange(30, 34)),
        }
    if information_sink is not None:
        information_sink(cases, names)
    gauge_matrix = cases.get("plus_optimistic_centered_total_flux", matrix)
    covariance_columns = gauge_matrix[:, 34:44]
    covariance_norms = np.linalg.norm(covariance_columns, axis=0)
    normalized = covariance_columns/np.maximum(covariance_norms, np.finfo(float).tiny)
    source_z_column = gauge_matrix[:, 0]
    source_z_fit = normalized@np.linalg.lstsq(normalized, source_z_column, rcond=1e-10)[0]
    gauge_relative_residual = np.linalg.norm(source_z_column-source_z_fit)/np.linalg.norm(source_z_column)
    return dict(cases=report, encoded_images=expected.tolist(), image_count=len(expected),
                parameter_names=names, source_covariance_m_rad=SOURCE_COVARIANCE.tolist(),
                transmission_range=transmission_range, optical_parameters_released=34,
                source_covariance_nuisance_count=10, source_reference_plane_released=True,
                source_reference_plane_gauge="Moving the reference plane and transporting the unknown source covariance/mean leaves all downstream data invariant.",
                source_reference_plane_gauge_relative_jacobian_residual=float(gauge_relative_residual),
                optical_parameter_scales=np.asarray(predictor.scales).tolist(),
                objective_prefield_fixed=False, independent_ellipse_angle_tag=False,
                aperture_plane_prior_sigma_m=aperture_plane_sigma_m,
                aperture_radius_relative_prior_sigma=.005, detector_scale_relative_prior_sigma=.002,
                total_flux_capture_verified=False, branch_uniqueness_proved=False,
                noisy_recovery_demonstrated=False, hardware_commands_admitted=False,
                backend=jax.default_backend(), quadrature_order=quadrature_order)


def focus_candidates(initialization, *, grid_points=33, state_count=8,
                     calibration_probe_target_m=None):
    """First sampled C-mini covariance minimum; no assumed Larmor-angle datum.

    This is a space-filling synthetic design, not an optimized hardware scan.
    It searches the new field's actual source-to-sample spatial covariance.
    The source covariance is a design truth, but is fully released in the fit.
    """
    from neoarm_measured_information_audit import control_for_q, select_spread_candidates
    column, parameters = initialization.column, jnp.asarray(initialization.parameters)
    covariance = jnp.asarray(SOURCE_COVARIANCE)

    def sample_covariance(q):
        matrix = column.matrix(parameters, control_for_q(column, q), segment="source_to_sample")
        propagated = matrix@covariance@matrix.T
        return propagated[jnp.ix_(XY, XY)]

    score = lambda q: jnp.trace(sample_covariance(q))
    pairs = np.array(np.meshgrid(np.linspace(.02, .55, grid_points),
                                np.linspace(.02, .45, grid_points))).reshape(2, -1).T
    mini = np.linspace(1e-5, 1.2, 129)
    grid = np.stack([np.broadcast_to(pairs[:, 0, None], (len(pairs), len(mini))),
                     np.broadcast_to(pairs[:, 1, None], (len(pairs), len(mini))),
                     np.broadcast_to(mini, (len(pairs), len(mini)))], axis=-1)
    scores = np.asarray(jax.jit(jax.vmap(score))(jnp.asarray(grid.reshape(-1, 3)))).reshape(len(pairs), -1)
    minima = (scores[:, 1:-1] < scores[:, :-2]) & (scores[:, 1:-1] < scores[:, 2:])
    eligible = minima.any(axis=1)
    indices = minima.argmax(axis=1)[eligible]+1
    selected_pairs = jnp.asarray(pairs[eligible])

    @jax.jit
    def refine(lower, upper):
        def step(_, bounds):
            low, high = bounds
            left, right = (2*low+high)/3, (low+2*high)/3
            lhs = jax.vmap(score)(jnp.column_stack([selected_pairs, left]))
            rhs = jax.vmap(score)(jnp.column_stack([selected_pairs, right]))
            return jnp.where(lhs < rhs, low, left), jnp.where(lhs < rhs, right, high)
        low, high = jax.lax.fori_loop(0, 48, step, (lower, upper))
        return jnp.column_stack([selected_pairs, (low+high)/2])

    q = np.asarray(refine(jnp.asarray(mini[indices-1]), jnp.asarray(mini[indices+1])))
    sample = np.asarray(jax.jit(jax.vmap(sample_covariance))(q))
    widths = np.sqrt(np.linalg.eigvalsh(sample))
    calibration_probe = None
    if calibration_probe_target_m is not None:
        if calibration_probe_target_m <= 0:
            raise ValueError("The calibration probe target must be positive.")
        index = int(np.argmin(np.abs(np.log(np.sqrt(np.prod(widths, axis=1))/calibration_probe_target_m))))
        calibration_probe = dict(target_rms_m=calibration_probe_target_m, q=q[index].tolist(),
            controls=np.asarray(control_for_q(column, q[index])).tolist(),
            sample_eigen_rms_m=widths[index].tolist(), upstream_command_calibration_assumed=False)
    # At 40k the detector half-field is 176 nm: this rejects invisible focus
    # candidates, not deliberately clipped off-band measurements added later.
    accepted = np.isfinite(widths).all(axis=1) & (widths.min(axis=1) >= 3e-9) & (widths.max(axis=1) < 70e-9)
    kept = q[accepted]
    selected = select_spread_candidates({"q": kept}, count=state_count)
    q = kept[selected]
    controls = np.asarray(jax.vmap(lambda value: control_for_q(column, value))(q))
    return dict(q=q, controls=controls, sample_eigen_rms_m=widths[accepted][selected],
                searched_pairs=len(pairs), first_minima_count=len(indices), accepted_count=int(accepted.sum()),
                calibration_probe=calibration_probe)


def make_design(initialization, condenser_controls, projection_controls, *,
                projector_wobble=.05, position_step_m=5e-9, angle_step_rad=.0002,
                calibration_relative_sigma=.002, wide_pl1_scan=False, calibration_controls=None):
    """Two-magnification pilot with actual projector controls and shared kicks.

    The stated sample-state calibration precision is a conditional requirement,
    never proof that a deflector command is calibrated on the instrument.
    """
    controls, kick, groups = [], [], []
    group = 0
    for illumination in condenser_controls:
        for projection in projection_controls:
            u = np.asarray(projection).copy(); u[:3] = illumination[:3]
            controls.append(u); kick.append(-1); groups.append(group); group += 1
    reference_illumination = (np.asarray(condenser_controls)[0, :3] if calibration_controls is None
                              else np.asarray(calibration_controls)[:3])
    for nominal in initialization.nominal_controls:
        settings = [np.asarray(nominal).copy()]
        for index in range(3, 7):
            for sign in (-1., 1.):
                u = np.asarray(nominal).copy(); u[index] += sign*projector_wobble
                settings.append(u)
        if wide_pl1_scan:
            lens = next(lens for lens in initialization.column.lenses if lens.name == "PL1")
            sign = np.sign(lens.q_bias) or 1.
            for magnitude in np.linspace(.05, np.pi-.05, 9):
                u = np.asarray(nominal).copy()
                u[lens.channel_index] = lens.reference_dac + (sign*magnitude-lens.q_bias)/lens.q_slope_per_dac
                settings.append(u)
        for u in settings:
            u[:3] = reference_illumination
            for basis in (-1, 0, 1, 2, 3):
                controls.append(u.copy()); kick.append(basis); groups.append(group)
            group += 1
    steps = np.array([position_step_m, angle_step_rad, position_step_m, angle_step_rad])
    states = np.diag(steps)
    # Each coordinate's absolute calibration sigma is declared. Off-axis errors
    # are not set exactly zero, and the errors are shared across optical states.
    cholesky = np.broadcast_to(np.diag(steps*calibration_relative_sigma), (4, 4, 4)).copy()
    return ImageDesign(np.asarray(controls), np.asarray(kick), np.asarray(groups), states, cholesky)


def fixed_tangent_field(slices_per_side=1024):
    """Fixed mesh refined near the nominal field; never changes during fitting.

    This improves quadrature of a *truncated* field. It does not justify its
    finite support or neglect of overlap between objective tails and C-mini.
    """
    from temgym_core.immersed_objective import SharedObjectiveField
    parts = []
    for a, b in ((-.05, 0.), (0., .05)):
        angles = np.linspace(np.arctan((a+.0005)/.003), np.arctan((b+.0005)/.003), slices_per_side+1)
        edge = -.0005+.003*np.tan(angles)
        edge[0], edge[-1] = a, b
        parts.append(edge)
    return SharedObjectiveField(tuple(np.concatenate((parts[0], parts[1][1:]))), 0.)


def check_forward_accuracy(report):
    """Two held-physics field/pixel refinements on every designed image.

    The field's physical peak amplitude is preserved when midpoint
    normalization changes. No lens strengths or controls are re-fitted.
    This tests the fixed truncated support only, not missing objective tails.
    """
    from neoarm_shared_objective import SharedObjectiveColumn
    from neoarm_measured_information_audit import measured_column
    from temgym_core.gaussian_image_likelihood import (
        make_gaussian_detector_operator, decode_gaussian_image_parameters,
    )
    if jax.default_backend() != "gpu" or report["field_mesh"] != "tangent":
        raise RuntimeError("This refinement check requires GPU and a fixed tangent mesh.")
    p = np.asarray(report["truth_optical_parameters"])
    design = ImageDesign(np.asarray(report["image_controls"]), np.asarray(report["image_kick_index"]),
        np.asarray(report["image_alignment_group"]), np.asarray(report["calibrated_sample_states"]),
        np.asarray(report["calibration_cholesky"]))
    n = report["field_slices_per_side"]
    field = fixed_tangent_field(n)
    column = SharedObjectiveColumn.from_hard_edge(measured_column(), objective=field,
                                                  objective_initial=tuple(p[-4:]))
    predictor = JointImagePredictor(column, design, parameters=p)
    values = [np.asarray(report["encoded_images"])]
    normalization = float(field.support_diagnostics(p[-4:])["normalization_length_m"])
    for count in (2*n, 4*n):
        refined = fixed_tangent_field(count)
        ratio = float(refined.support_diagnostics(p[-4:])["normalization_length_m"])/normalization
        newp = p.copy(); newp[-4] *= ratio
        new_column = SharedObjectiveColumn.from_hard_edge(measured_column(), objective=refined,
                                                          objective_initial=tuple(newp[-4:]))
        fun = JointImagePredictor(new_column, design, parameters=newp)
        values.append(np.asarray(jax.jit(fun)(jnp.zeros(75))))
    order = report["quadrature_order"]
    operators = [make_gaussian_detector_operator(predictor.detector, q) for q in (order, 2*order, 4*order)]
    render = [jax.jit(jax.vmap(lambda x, op=op: op(*decode_gaussian_image_parameters(x, predictor.detector))))
              for op in operators]
    factor = jax.jit(jax.vmap(lambda x: gaussian_image_information_factor(
        x, predictor.detector, quadrature_order=2*order)))
    metrics = {key: [] for key in ("quadrature_first_rms_sigma", "quadrature_second_rms_sigma",
        "field_first_rms_sigma", "field_second_rms_sigma",
        "field_first_image_coordinate_mahalanobis_sigma", "field_second_image_coordinate_mahalanobis_sigma")}
    for start in range(0, len(values[0]), 4):
        v = [jnp.asarray(value[start:start+4]) for value in values]
        low, medium, high = [np.asarray(op(v[0])) for op in render]
        scale = np.sqrt(high+.6**2)
        f1, f2 = np.asarray(render[1](v[1])), np.asarray(render[1](v[2]))
        metric = np.asarray(factor(v[0]))
        for name, delta in (("quadrature_first", medium-low), ("quadrature_second", high-medium),
                            ("field_first", f1-medium), ("field_second", f2-f1)):
            metrics[name+"_rms_sigma"].extend(np.sqrt(np.mean((delta/scale)**2, axis=(1, 2))).tolist())
        for name, delta in (("field_first", v[1]-v[0]), ("field_second", v[2]-v[1])):
            metrics[name+"_image_coordinate_mahalanobis_sigma"].extend(
                np.linalg.norm(np.einsum("nij,nj->ni", metric, np.asarray(delta)), axis=1).tolist())
    # A pixel residual's full Euclidean norm also bounds its projection onto
    # every normalized image-parameter score direction.
    for name in ("quadrature_first", "quadrature_second"):
        metrics[name+"_full_pixel_norm_sigma"] = (np.asarray(metrics[name+"_rms_sigma"])*
                                                   np.sqrt(np.prod(predictor.detector.shape_yx))).tolist()
    return dict(field_slice_counts=[n, 2*n, 4*n], pixel_quadrature_orders=[order, 2*order, 4*order],
        per_frame=metrics, maxima={key: float(np.max(value)) for key, value in metrics.items()},
        all_tested_differences_below_point_two_sigma=all(np.max(value) < .2 for value in metrics.values()),
        optical_parameters_held_fixed=True, physical_peak_amplitude_preserved=True,
        finite_support_validated=False, complete_parameter_box_tested=False,
        recovered_parameter_refinement_tested=False)


def run_audit(*, grid_points=33, state_count=8, off_band_q_step=.015,
              aperture_plane_sigma_m=None, angle_step_rad=.0002, slices_per_side=256,
              projector_wobble=.05, quadrature_order=8, field_mesh="uniform",
              information_output=None, wide_pl1_scan=False, check_forward=False,
              calibration_probe_target_m=None):
    from neoarm_measured_information_audit import measured_column, control_for_q
    from neoarm_shared_objective import initialize_shared_imaging, shared_projection_states
    from temgym_core.immersed_objective import SharedObjectiveField
    started = perf_counter()
    source_hashes, input_hashes = provenance_snapshot()
    objective = (fixed_tangent_field(slices_per_side) if field_mesh == "tangent" else
                 SharedObjectiveField.from_window(-.05, 0., .05, n_slices_per_side=slices_per_side))
    initialization = initialize_shared_imaging(measured_column(), objective=objective)
    candidates = focus_candidates(initialization, grid_points=grid_points, state_count=state_count,
                                  calibration_probe_target_m=calibration_probe_target_m)
    q = candidates["q"].copy()
    if off_band_q_step:
        extra = [q[0] + sign*off_band_q_step*np.eye(3)[axis]
                 for axis in range(3) for sign in (-1., 1.)]
        q = np.vstack([q, extra])
    illumination = np.asarray(jax.vmap(lambda value: control_for_q(initialization.column, value))(q))
    projection = shared_projection_states(initialization)
    calibration_controls = (candidates["calibration_probe"]["controls"]
                            if candidates["calibration_probe"] is not None else None)
    design = make_design(initialization, illumination, projection["controls"], angle_step_rad=angle_step_rad,
                         projector_wobble=projector_wobble, wide_pl1_scan=wide_pl1_scan,
                         calibration_controls=calibration_controls)
    predictor = JointImagePredictor(initialization.column, design, parameters=initialization.parameters)
    def save_information(cases, names):
        if information_output is not None:
            information_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(information_output, parameter_names=np.asarray(names), **cases)

    report = joint_information(predictor, aperture_controls=illumination,
                               aperture_plane_sigma_m=aperture_plane_sigma_m,
                               quadrature_order=quadrature_order, information_sink=save_information)
    report.update(schema="neoarm-shared-objective-joint-information-v1",
        source_hashes=source_hashes, computational_input_hashes=input_hashes,
        provenance_scope="Conservative loaded repository dependency closure plus computational notebook/TOML inputs; snapshots verified before/after calculation.",
        runtime_versions={name: distribution_version(name) for name in ("jax", "jaxlib", "numpy", "scipy")},
        python_version=platform.python_version(),
        truth_optical_parameters=initialization.parameters.tolist(),
        focus_search={key: value.tolist() if isinstance(value, np.ndarray) else value
                      for key, value in candidates.items()},
        illumination_q=q.tolist(), illumination_controls=illumination.tolist(),
        projection_states=projection, image_controls=design.controls.tolist(),
        image_kick_index=design.kick_index.tolist(), image_alignment_group=design.alignment_group.tolist(),
        calibrated_sample_states=design.calibrated_states.tolist(),
        calibration_cholesky=design.calibration_cholesky.tolist(),
        synthetic_controls_only=True, objective_controls_fixed=True,
        field_slices_per_side=slices_per_side, field_mesh=field_mesh,
        field_edges_m=list(objective.z_edges_m), field_discretization_converged=False,
        field_finite_support_validated=False,
        information_matrix_file=str(information_output) if information_output else None,
        projector_one_lens_wobble=projector_wobble, incident_electrons_per_frame=predictor.dose,
        wide_same_sign_first_branch_pl1_scan=wide_pl1_scan,
        gaussian_detector=dict(shape_yx=predictor.detector.shape_yx,
            pixel_size_yx_m=predictor.detector.pixel_size_yx_m, mtf_sigma_px=predictor.detector.mtf_sigma_px,
            read_noise_electrons=.6),
        assumptions=["Common objective field shape is synthetic and must be checked for model error.",
            "Four mechanical sample-plane input states have the stated independent metrology covariance.",
            "Mean alignment is constant within each five-frame basis group; dose can change per frame.",
            "All ten source covariance nuisance coordinates are free; no calibrated stigmator or upstream command.",
            "Centered aperture total flux requires independent dose references and validated full capture.",
            "Approximate aperture radii are not yet certified to the conditional 0.5 percent precision.",
            "Local information and numerical rank are not noisy recovery or branch uniqueness."])
    if check_forward:
        report["forward_refinement"] = check_forward_accuracy(report)
        report["selected_frame_forward_refinements_passed"] = report["forward_refinement"][
            "all_tested_differences_below_point_two_sigma"]
    verify_provenance_snapshot(source_hashes)
    final_hashes, _ = provenance_snapshot()
    additional = sorted(set(final_hashes)-set(source_hashes))
    if additional:
        raise RuntimeError("Preload these repository dependencies before rerunning: " + ", ".join(additional))
    report["information_matrix_artifact"] = information_artifact_metadata(information_output)
    report["information_matrix_sha256"] = (report["information_matrix_artifact"]["sha256"]
                                           if information_output is not None else None)
    report["source_snapshot_verified_unchanged"] = True
    report["elapsed_seconds"] = perf_counter()-started
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--grid-points", type=int, default=33)
    parser.add_argument("--state-count", type=int, default=8)
    parser.add_argument("--off-band-q-step", type=float, default=.015)
    parser.add_argument("--aperture-plane-sigma-m", type=float)
    parser.add_argument("--angle-step-rad", type=float, default=.0002)
    parser.add_argument("--slices-per-side", type=int, default=256)
    parser.add_argument("--projector-wobble", type=float, default=.05)
    parser.add_argument("--quadrature-order", type=int, default=8)
    parser.add_argument("--field-mesh", choices=("uniform", "tangent"), default="uniform")
    parser.add_argument("--information-output", type=Path)
    parser.add_argument("--wide-pl1-scan", action="store_true")
    parser.add_argument("--check-forward", action="store_true")
    parser.add_argument("--calibration-probe-rms-m", type=float)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "gpu":
        raise RuntimeError("The coupled finite-pixel information audit requires a GPU.")
    report = run_audit(grid_points=args.grid_points, state_count=args.state_count,
        off_band_q_step=args.off_band_q_step, aperture_plane_sigma_m=args.aperture_plane_sigma_m,
        angle_step_rad=args.angle_step_rad, slices_per_side=args.slices_per_side,
        projector_wobble=args.projector_wobble, quadrature_order=args.quadrature_order,
        field_mesh=args.field_mesh, information_output=args.information_output,
        wide_pl1_scan=args.wide_pl1_scan, check_forward=args.check_forward,
        calibration_probe_target_m=args.calibration_probe_rms_m)
    report["invocation_argv"] = sys.argv
    report["jax_enable_x64"] = bool(jax.config.x64_enabled)
    output = json.dumps(report, indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
        print(json.dumps({key: {group: (value["numerical_rank"], value.get("maximum_scaled_sigma"))
                              for group, value in case.items()} for key, case in report["cases"].items()}, indent=2))
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
