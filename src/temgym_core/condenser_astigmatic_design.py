"""Conditional no-commanded-tilt condenser design with an astigmatic tag.

The design in this module is deliberately fail closed.  It combines the
translated C-aperture Poisson block from :mod:`condenser_branch_b` with
sample-plane spatial covariance, indexed-Si angular covariance, and the
orientation of a deliberately astigmatic incident-wave tag.  No calibrated
deflector drive or commanded sample tilt appears in the design matrix.

The committed controls are normalized synthetic model coordinates, not safe
NEOARM commands.  The reported sub-one-percent number is a nominal-truth,
matched-hard-edge Fisher result with almost no margin.  Detector sampling,
MTF, disc-edge SNR, saturation, overlap, nonlinear recovery, and truth-envelope
robustness are explicitly unverified and therefore hardware admission is
always refused by the committed manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from .condenser_branch_b import (
    NORMALIZED_CONTROL_WARNING,
    BranchBLikelihood,
    load_branch_b_experiment,
    make_synthetic_branch_b_observations,
)
from .constants import energy2wavelength
from .critical_line_calibration import (
    EntranceCovarianceChart,
    axial_plane_covariance_observables_with_entrance_covariance,
)


ASTIGMATIC_DESIGN_SCHEMA = (
    "temgym.neoarm.condenser-astigmatic-no-tilt-design/v1"
)
EXPECTED_ASTIGMATIC_DESIGN_SHA256 = (
    "f0e6db465d926e55311d84e55a7174e9077cf3b685deab51390a7e8814bba705"
)
DEFAULT_ASTIGMATIC_DESIGN = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "lens_inversion"
    / "neoarm_condenser_astigmatic_design.json"
)

_SPATIAL_OBSERVABLE_INDICES = (0, 2, 5)
_ANGULAR_OBSERVABLE_INDICES = (1, 3, 8)
_EXPECTED_TARGET_COUNT = 12
_EXPECTED_SAMPLE_ACQUISITIONS = 48
_EXPECTED_UNIQUE_SAMPLE_STATES = 13
_EXPECTED_MATRIX_SHAPE = (428, 43)


class AstigmaticDesignProvenanceError(ValueError):
    """Raised when the immutable design or its Branch-B parent changed."""


class AstigmaticDesignAdmissionError(RuntimeError):
    """Raised when a conditional local design is presented as hardware ready."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_array(
    value: Any,
    name: str,
    *,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if shape is not None and array.shape != shape:
        raise AstigmaticDesignProvenanceError(
            f"{name} must have shape {shape}; got {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise AstigmaticDesignProvenanceError(f"{name} is not finite.")
    result = array.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class AstigmaticCondenserDesign:
    """Validated immutable acquisition and uncertainty contract."""

    path: Path
    sha256: str
    branch_b_payload_path: Path
    branch_b_payload_sha256: str
    controls: np.ndarray
    repeat_counts: np.ndarray
    covariance_summary_sigma: float
    tag_orientation_sigma_rad: float
    realspace_log_scale_prior_sigma: float
    angular_log_scale_prior_sigma: float
    planned_camera_length_m: float
    detector_pixel_size_m: float
    detector_shape_yx: tuple[int, int]
    detector_mtf_sigma_px: float
    si_111_spacing_m: float
    reference_voltage_v: float
    hardware_ready: bool
    angular_observability_verified: bool
    nonlinear_recovery_verified: bool
    truth_envelope_verified: bool
    held_out_verified: bool
    hardware_gates: Mapping[str, Mapping[str, Any]]
    expected_audit: Mapping[str, Any]
    frame_accounting: Mapping[str, Any]
    scope: str
    coordinate_status: str

    def __post_init__(self) -> None:
        controls = _frozen_array(
            self.controls,
            "controls",
            shape=(_EXPECTED_UNIQUE_SAMPLE_STATES, 7),
        )
        repeats = np.asarray(self.repeat_counts, dtype=int)
        if repeats.shape != (_EXPECTED_UNIQUE_SAMPLE_STATES,) or np.any(repeats <= 0):
            raise AstigmaticDesignProvenanceError(
                "repeat_counts must contain thirteen positive integers."
            )
        if int(np.sum(repeats)) != _EXPECTED_SAMPLE_ACQUISITIONS:
            raise AstigmaticDesignProvenanceError(
                "The astigmatic design must contain 48 sample acquisitions."
            )
        if not np.allclose(controls[:, 3:], 0.0, rtol=0.0, atol=0.0):
            raise AstigmaticDesignProvenanceError(
                "Downstream normalized controls must remain exactly zero."
            )
        repeats = repeats.copy()
        repeats.setflags(write=False)
        positive = (
            self.covariance_summary_sigma,
            self.tag_orientation_sigma_rad,
            self.realspace_log_scale_prior_sigma,
            self.angular_log_scale_prior_sigma,
            self.planned_camera_length_m,
            self.detector_pixel_size_m,
            self.detector_mtf_sigma_px,
            self.si_111_spacing_m,
            self.reference_voltage_v,
        )
        if not np.all(np.isfinite(positive)) or min(positive) <= 0.0:
            raise AstigmaticDesignProvenanceError(
                "Every physical/noise scale must be finite and positive."
            )
        if self.coordinate_status != NORMALIZED_CONTROL_WARNING:
            raise AstigmaticDesignProvenanceError(
                "The normalized-control warning is missing."
            )
        admission_flags = (
            self.hardware_ready,
            self.angular_observability_verified,
            self.nonlinear_recovery_verified,
            self.truth_envelope_verified,
            self.held_out_verified,
        )
        if any(admission_flags):
            raise AstigmaticDesignProvenanceError(
                "The committed conditional design must remain fail closed."
            )
        if any(
            bool(gate.get("verified_on_instrument"))
            for gate in self.hardware_gates.values()
        ):
            raise AstigmaticDesignProvenanceError(
                "No detector-feasibility gate has instrument evidence yet."
            )
        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "repeat_counts", repeats)
        object.__setattr__(
            self, "hardware_gates", MappingProxyType(dict(self.hardware_gates))
        )
        object.__setattr__(
            self, "expected_audit", MappingProxyType(dict(self.expected_audit))
        )
        object.__setattr__(
            self, "frame_accounting", MappingProxyType(dict(self.frame_accounting))
        )

    @property
    def expanded_controls(self) -> np.ndarray:
        expanded = np.repeat(self.controls, self.repeat_counts, axis=0)
        expanded.setflags(write=False)
        return expanded

    @property
    def conservative_raw_frame_count(self) -> int:
        return int(self.frame_accounting["conservative_total_raw_frames"])

    @property
    def joint_covariance_raw_frame_count(self) -> int:
        return int(self.frame_accounting["joint_dense_covariance_total_raw_frames"])


@dataclass(frozen=True)
class AstigmaticDesignMatrix:
    """Whitened local Jacobian and its exact row/column ordering."""

    matrix: np.ndarray
    row_names: tuple[str, ...]
    column_names: tuple[str, ...]

    def __post_init__(self) -> None:
        matrix = _frozen_array(self.matrix, "matrix", shape=_EXPECTED_MATRIX_SHAPE)
        rows = tuple(str(value) for value in self.row_names)
        columns = tuple(str(value) for value in self.column_names)
        if matrix.shape != (len(rows), len(columns)):
            raise ValueError("Matrix dimensions do not match its names.")
        if len(rows) != len(set(rows)) or len(columns) != len(set(columns)):
            raise ValueError("Audit row and column names must be unique.")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "row_names", rows)
        object.__setattr__(self, "column_names", columns)


@dataclass(frozen=True)
class AstigmaticDesignAudit:
    """Profiled target diagnostics and nominal detector geometry."""

    target_rank: int
    nuisance_rank: int
    singular_values: np.ndarray
    scaled_standard_uncertainties: np.ndarray
    weakest_target_direction: np.ndarray
    target_names: tuple[str, ...]
    sample_equivalent_rms_range_m: np.ndarray
    angular_eigen_rms_range_rad: np.ndarray
    angular_eigen_rms_range_px: np.ndarray
    si_111_radius_px: float
    maximum_three_sigma_angular_support_px: float
    nominal_detector_geometry_pass: bool
    hardware_ready: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "singular_values",
            _frozen_array(self.singular_values, "singular_values", shape=(12,)),
        )
        object.__setattr__(
            self,
            "scaled_standard_uncertainties",
            _frozen_array(
                self.scaled_standard_uncertainties,
                "scaled_standard_uncertainties",
                shape=(12,),
            ),
        )
        object.__setattr__(
            self,
            "weakest_target_direction",
            _frozen_array(
                self.weakest_target_direction,
                "weakest_target_direction",
                shape=(12,),
            ),
        )
        object.__setattr__(
            self,
            "sample_equivalent_rms_range_m",
            _frozen_array(
                self.sample_equivalent_rms_range_m,
                "sample_equivalent_rms_range_m",
                shape=(2,),
            ),
        )
        object.__setattr__(
            self,
            "angular_eigen_rms_range_rad",
            _frozen_array(
                self.angular_eigen_rms_range_rad,
                "angular_eigen_rms_range_rad",
                shape=(2, 2),
            ),
        )
        object.__setattr__(
            self,
            "angular_eigen_rms_range_px",
            _frozen_array(
                self.angular_eigen_rms_range_px,
                "angular_eigen_rms_range_px",
                shape=(2, 2),
            ),
        )
        if len(self.target_names) != 12:
            raise ValueError("Expected twelve target names.")

    @property
    def minimum_singular_value_ratio(self) -> float:
        return float(self.singular_values[-1] / self.singular_values[0])

    @property
    def maximum_scaled_standard_uncertainty(self) -> float:
        return float(np.max(self.scaled_standard_uncertainties))

    @property
    def conditional_precision_pass(self) -> bool:
        return bool(
            self.target_rank == 12
            and self.maximum_scaled_standard_uncertainty < 0.01
        )


def load_astigmatic_condenser_design(
    path: str | Path = DEFAULT_ASTIGMATIC_DESIGN,
    *,
    repository_root: str | Path | None = None,
) -> AstigmaticCondenserDesign:
    """Load and checksum-verify the conditional design manifest."""

    design_path = Path(path).resolve()
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    digest = _sha256(design_path)
    if digest != EXPECTED_ASTIGMATIC_DESIGN_SHA256:
        raise AstigmaticDesignProvenanceError(
            "The astigmatic condenser design manifest checksum changed."
        )
    document = json.loads(design_path.read_text(encoding="utf-8"))
    if document.get("schema") != ASTIGMATIC_DESIGN_SCHEMA:
        raise AstigmaticDesignProvenanceError("Unsupported design schema.")
    if document.get("coordinate_status") != NORMALIZED_CONTROL_WARNING:
        raise AstigmaticDesignProvenanceError(
            "The normalized-control warning changed."
        )

    parent = document["provenance"]["branch_b_payload"]
    parent_path = root / parent["path"]
    if _sha256(parent_path) != parent["sha256"]:
        raise AstigmaticDesignProvenanceError(
            "The bound Branch-B design payload changed."
        )
    parent_document = json.loads(parent_path.read_text(encoding="utf-8"))
    lower, upper = np.asarray(
        parent_document["model_control_contract"][
            "safe_bounds_CL1_CL3_Cmini"
        ],
        dtype=float,
    )

    states = document["sample_block"]["states"]
    controls3 = np.asarray(
        [state["controls_CL1_CL3_Cmini"] for state in states], dtype=float
    )
    repeats = np.asarray([state["repeat_count"] for state in states], dtype=int)
    if controls3.shape != (_EXPECTED_UNIQUE_SAMPLE_STATES, 3):
        raise AstigmaticDesignProvenanceError("Unexpected selected-state count.")
    if np.any(controls3 < lower[None, :]) or np.any(controls3 > upper[None, :]):
        raise AstigmaticDesignProvenanceError(
            "A selected normalized control lies outside the frozen safe box."
        )
    controls = np.concatenate(
        (controls3, np.zeros((_EXPECTED_UNIQUE_SAMPLE_STATES, 4))), axis=1
    )

    q_contract = parent_document["model_control_contract"]["affine_q_per_m"]
    q = (
        np.asarray(q_contract["bias"])[None, :]
        + controls3 * np.asarray(q_contract["slope"])[None, :]
    )
    expected_q = document["sample_block"][
        "selected_integrated_larmor_q_range_dimensionless"
    ]
    np.testing.assert_allclose(
        np.min(q, axis=0), expected_q["minimum"], rtol=2.0e-12, atol=1.0e-14
    )
    np.testing.assert_allclose(
        np.max(q, axis=0), expected_q["maximum"], rtol=2.0e-12, atol=1.0e-14
    )

    sample = document["sample_block"]
    detector = document["angular_detector_feasibility"]
    admission = document["hardware_admission"]
    return AstigmaticCondenserDesign(
        path=design_path,
        sha256=digest,
        branch_b_payload_path=parent_path,
        branch_b_payload_sha256=str(parent["sha256"]),
        controls=controls,
        repeat_counts=repeats,
        covariance_summary_sigma=float(
            sample["spatial_and_angular_covariance_summary_sigma"]
        ),
        tag_orientation_sigma_rad=float(sample["tag_orientation_sigma_rad"]),
        realspace_log_scale_prior_sigma=float(
            sample["realspace_log_scale_prior_sigma"]
        ),
        angular_log_scale_prior_sigma=float(
            sample["indexed_si_log_angular_scale_prior_sigma"]
        ),
        planned_camera_length_m=float(
            detector["planned_validation_camera_length_m"]
        ),
        detector_pixel_size_m=float(detector["detector_pixel_size_m"]),
        detector_shape_yx=tuple(int(value) for value in detector["detector_shape_yx"]),
        detector_mtf_sigma_px=float(detector["detector_mtf_sigma_px"]),
        si_111_spacing_m=float(detector["si_111_spacing_m"]),
        reference_voltage_v=float(detector["reference_voltage_V"]),
        hardware_ready=bool(admission["ready"]),
        angular_observability_verified=bool(
            admission["angular_observability_verified"]
        ),
        nonlinear_recovery_verified=bool(
            admission["nonlinear_recovery_verified"]
        ),
        truth_envelope_verified=bool(admission["truth_envelope_verified"]),
        held_out_verified=bool(admission["held_out_verified"]),
        hardware_gates=detector["gates"],
        expected_audit=document["fisher_audit"],
        frame_accounting=document["frame_accounting"],
        scope=str(document["scope"]),
        coordinate_status=str(document["coordinate_status"]),
    )


def _branch_likelihood(design: AstigmaticCondenserDesign) -> BranchBLikelihood:
    experiment = load_branch_b_experiment(design.branch_b_payload_path)
    if experiment.payload_sha256 != design.branch_b_payload_sha256:
        raise AstigmaticDesignProvenanceError(
            "The loaded Branch-B experiment differs from the bound parent."
        )
    return BranchBLikelihood(
        experiment,
        make_synthetic_branch_b_observations(experiment),
    )


def build_astigmatic_design_matrix(
    design: AstigmaticCondenserDesign,
    likelihood: BranchBLikelihood | None = None,
) -> tuple[AstigmaticDesignMatrix, np.ndarray]:
    """Build the 428-by-43 x64 matrix and return nominal sample summaries.

    The second return value is the unwhitened full ten-coordinate covariance
    chart at every repeated sample acquisition.  It is used only for the
    detector-geometry audit.
    """

    if not isinstance(design, AstigmaticCondenserDesign):
        raise TypeError("design must be an AstigmaticCondenserDesign.")
    jax.config.update("jax_enable_x64", True)
    branch = _branch_likelihood(design) if likelihood is None else likelihood
    if branch.experiment.payload_sha256 != design.branch_b_payload_sha256:
        raise AstigmaticDesignProvenanceError(
            "The supplied likelihood is not bound to this design."
        )

    parameters = jnp.asarray(branch.nominal_parameters)
    old_jacobian = np.asarray(branch.jacobian(parameters), dtype=float)
    keep_old_columns = np.asarray(
        (*range(26), 42, 43, *range(44, 56)), dtype=int
    )
    retained_names = tuple(branch.parameter_names[index] for index in keep_old_columns)
    column_names = retained_names + (
        "nuisance.shared_log_realspace_scale",
        "nuisance.shared_indexed_si_log_angular_scale",
        "nuisance.shared_astigmatic_tag_angle_offset_rad",
    )

    edge_slice = branch.residual_slices["edge_poisson"]
    prior_slice = branch.residual_slices["shared_priors"]
    edge = old_jacobian[edge_slice][:, keep_old_columns]
    parent_priors = old_jacobian[prior_slice][:, keep_old_columns]
    scale_priors = np.zeros((2, len(column_names)))
    scale_priors[0, 40] = 1.0 / design.realspace_log_scale_prior_sigma
    scale_priors[1, 41] = 1.0 / design.angular_log_scale_prior_sigma

    expanded_controls = design.expanded_controls
    experiment = branch.experiment
    physical0 = jnp.asarray(experiment.column.nominal_parameters)
    eta0 = jnp.zeros(10, dtype=physical0.dtype)
    chart = EntranceCovarianceChart(experiment.entrance_reference_covariance)

    def covariance_response(physical: Any, eta: Any) -> jnp.ndarray:
        return axial_plane_covariance_observables_with_entrance_covariance(
            experiment.column,
            physical,
            eta,
            jnp.asarray(expanded_controls),
            chart,
            plane_z_m=experiment.sample_z_m,
            mode="full",
        )

    summaries = np.asarray(covariance_response(physical0, eta0), dtype=float)
    jacobian_physical = np.asarray(
        jax.jacfwd(lambda value: covariance_response(value, eta0))(physical0),
        dtype=float,
    )
    jacobian_eta = np.asarray(
        jax.jacfwd(lambda value: covariance_response(physical0, value))(eta0),
        dtype=float,
    )

    selected = np.asarray(
        (*_SPATIAL_OBSERVABLE_INDICES, *_ANGULAR_OBSERVABLE_INDICES),
        dtype=int,
    )
    sample = np.zeros((_EXPECTED_SAMPLE_ACQUISITIONS, 7, len(column_names)))
    target_indices = np.asarray(experiment.target_parameter_indices)
    target_scales = np.asarray(experiment.target_parameter_scales)
    upstream_indices = np.asarray(experiment.upstream_parameter_indices)
    upstream_scales = np.asarray(experiment.upstream_parameter_scales)
    sample[:, :6, :12] = (
        jacobian_physical[:, selected][:, :, target_indices]
        * target_scales[None, None, :]
    )
    sample[:, :6, 12:22] = jacobian_eta[:, selected]
    sample[:, :6, 22:26] = (
        jacobian_physical[:, selected][:, :, upstream_indices]
        * upstream_scales[None, None, :]
    )
    sample[:, 0, 40] = 1.0
    sample[:, 1, 40] = 1.0
    sample[:, 3, 41] = 1.0
    sample[:, 4, 41] = 1.0
    sample[:, :6] /= design.covariance_summary_sigma

    # A shared unknown tag orientation is deliberately profiled.  The hard-edge
    # round-lens identity is Phi=-sum(q), with q affine in the recorded raw
    # controls.  Absolute q-bias sums remain gauged against this offset; command
    # differences constrain the slopes.
    sample[:, 6, 42] = 1.0
    for column in (2, 6, 10):
        sample[:, 6, column] = -target_scales[column]
    for lens_index, column in enumerate((3, 7, 11)):
        sample[:, 6, column] = (
            -expanded_controls[:, lens_index] * target_scales[column]
        )
    upstream_names = experiment.parameter_layout.upstream_names
    objective_q = upstream_names.index("Obj-prefield.q_bias")
    sample[:, 6, 22 + objective_q] = -upstream_scales[objective_q]
    sample[:, 6] /= design.tag_orientation_sigma_rad

    base = np.zeros((92, len(column_names)))
    base[:88, :40] = edge
    base[88:90, :40] = parent_priors
    base[90:] = scale_priors
    matrix = np.concatenate((base, sample.reshape(-1, len(column_names))))

    row_names = (
        tuple(branch.residual_names[edge_slice])
        + tuple(branch.residual_names[prior_slice])
        + (
            "prior.registered_sample_log_realspace_scale",
            "prior.indexed_si_log_angular_scale",
        )
        + tuple(
            f"sample.acquisition_{acquisition:02d}.{observable}"
            for acquisition in range(_EXPECTED_SAMPLE_ACQUISITIONS)
            for observable in (
                "log_rms_x",
                "log_rms_y",
                "correlation_x_y",
                "log_rms_theta_x",
                "log_rms_theta_y",
                "correlation_theta_x_theta_y",
                "astigmatic_tag_orientation",
            )
        )
    )
    return AstigmaticDesignMatrix(matrix, row_names, column_names), summaries


def _eigen_rms(
    log_x: np.ndarray, log_y: np.ndarray, correlation: np.ndarray
) -> np.ndarray:
    sigma_x = np.exp(log_x)
    sigma_y = np.exp(log_y)
    covariance = correlation * sigma_x * sigma_y
    trace = sigma_x**2 + sigma_y**2
    discriminant = np.sqrt(
        (sigma_x**2 - sigma_y**2) ** 2 + 4.0 * covariance**2
    )
    return np.stack(
        (
            np.sqrt(0.5 * (trace - discriminant)),
            np.sqrt(0.5 * (trace + discriminant)),
        ),
        axis=-1,
    )


def recompute_astigmatic_design_audit(
    design: AstigmaticCondenserDesign | None = None,
    *,
    verify_expected: bool = True,
) -> AstigmaticDesignAudit:
    """Recompute the nominal local rank and detector-geometry predictions."""

    selected = load_astigmatic_condenser_design() if design is None else design
    matrix_contract, summaries = build_astigmatic_design_matrix(selected)
    matrix = matrix_contract.matrix
    target = matrix[:, :_EXPECTED_TARGET_COUNT]
    nuisance = matrix[:, _EXPECTED_TARGET_COUNT:]
    norms = np.linalg.norm(nuisance, axis=0)
    active = norms > np.finfo(float).tiny
    normalized = nuisance[:, active] / norms[active][None, :]
    left, nuisance_singular, _ = np.linalg.svd(normalized, full_matrices=False)
    nuisance_rank = int(
        np.sum(nuisance_singular > 1.0e-10 * nuisance_singular[0])
    )
    projected = target - left[:, :nuisance_rank] @ (
        left[:, :nuisance_rank].T @ target
    )
    _left, singular, right = np.linalg.svd(projected, full_matrices=False)
    target_rank = int(np.sum(singular > 1.0e-7 * singular[0]))
    uncertainty = (
        np.sqrt(np.diag(np.linalg.inv(projected.T @ projected)))
        if target_rank == _EXPECTED_TARGET_COUNT
        else np.full(_EXPECTED_TARGET_COUNT, np.inf)
    )

    spatial_rms = np.sqrt(
        0.5 * (np.exp(2.0 * summaries[:, 0]) + np.exp(2.0 * summaries[:, 2]))
    )
    angular_eigen = _eigen_rms(
        summaries[:, 1], summaries[:, 3], summaries[:, 8]
    )
    angular_range = np.stack(
        (np.min(angular_eigen, axis=0), np.max(angular_eigen, axis=0))
    )
    px_per_rad = selected.planned_camera_length_m / selected.detector_pixel_size_m
    angular_px = angular_range * px_per_rad
    wavelength = float(np.asarray(energy2wavelength(selected.reference_voltage_v)))
    si_radius_px = (
        wavelength
        / selected.si_111_spacing_m
        * selected.planned_camera_length_m
        / selected.detector_pixel_size_m
    )
    maximum_support = 3.0 * float(angular_px[1, 1])
    detector_half_width = 0.5 * min(selected.detector_shape_yx)
    geometry_pass = bool(
        si_radius_px >= 3.0
        and si_radius_px <= 96.0
        and angular_px[0, 0] >= 1.5 * selected.detector_mtf_sigma_px
        and si_radius_px + maximum_support < detector_half_width
        and 2.0 * maximum_support < si_radius_px
    )
    audit = AstigmaticDesignAudit(
        target_rank=target_rank,
        nuisance_rank=nuisance_rank,
        singular_values=singular,
        scaled_standard_uncertainties=uncertainty,
        weakest_target_direction=right[-1],
        target_names=tuple(matrix_contract.column_names[:12]),
        sample_equivalent_rms_range_m=np.asarray(
            (np.min(spatial_rms), np.max(spatial_rms))
        ),
        angular_eigen_rms_range_rad=angular_range,
        angular_eigen_rms_range_px=angular_px,
        si_111_radius_px=float(si_radius_px),
        maximum_three_sigma_angular_support_px=maximum_support,
        nominal_detector_geometry_pass=geometry_pass,
        hardware_ready=False,
    )

    if verify_expected:
        expected = selected.expected_audit
        if audit.target_rank != int(expected["profiled_target_rank"]):
            raise AstigmaticDesignProvenanceError("Target rank changed.")
        if audit.nuisance_rank != int(expected["profiled_nuisance_rank"]):
            raise AstigmaticDesignProvenanceError("Nuisance rank changed.")
        np.testing.assert_allclose(
            audit.singular_values,
            expected["profiled_target_singular_values"],
            rtol=3.0e-8,
            atol=1.0e-8,
        )
        np.testing.assert_allclose(
            audit.scaled_standard_uncertainties,
            expected["scaled_parameter_standard_uncertainties"],
            rtol=3.0e-8,
            atol=1.0e-10,
        )
        expected_weakest = np.asarray(expected["weakest_target_direction"])
        sign = np.sign(np.dot(audit.weakest_target_direction, expected_weakest))
        np.testing.assert_allclose(
            sign * audit.weakest_target_direction,
            expected_weakest,
            rtol=3.0e-8,
            atol=1.0e-9,
        )
    return audit


def require_astigmatic_hardware_readiness(
    design: AstigmaticCondenserDesign,
    audit: AstigmaticDesignAudit,
) -> None:
    """Fail closed until local precision and every measured gate pass."""

    if not audit.conditional_precision_pass:
        raise AstigmaticDesignAdmissionError(
            "The astigmatic condenser design fails its conditional precision gate."
        )
    unverified = tuple(
        name
        for name, gate in design.hardware_gates.items()
        if not bool(gate.get("verified_on_instrument"))
    )
    if (
        not design.hardware_ready
        or not design.angular_observability_verified
        or not design.nonlinear_recovery_verified
        or not design.truth_envelope_verified
        or not design.held_out_verified
        or unverified
        or not audit.hardware_ready
    ):
        raise AstigmaticDesignAdmissionError(
            "The local astigmatic design is not hardware ready; unverified gates: "
            + ", ".join(unverified)
        )


__all__ = [
    "ASTIGMATIC_DESIGN_SCHEMA",
    "DEFAULT_ASTIGMATIC_DESIGN",
    "EXPECTED_ASTIGMATIC_DESIGN_SHA256",
    "AstigmaticCondenserDesign",
    "AstigmaticDesignAdmissionError",
    "AstigmaticDesignAudit",
    "AstigmaticDesignMatrix",
    "AstigmaticDesignProvenanceError",
    "build_astigmatic_design_matrix",
    "load_astigmatic_condenser_design",
    "recompute_astigmatic_design_audit",
    "require_astigmatic_hardware_readiness",
]
