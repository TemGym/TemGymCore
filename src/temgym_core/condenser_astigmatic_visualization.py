"""Deterministic figures for the conditional astigmatic no-tilt pilot.

This module visualises the *observable-level* predictions already present in
the checksum-verified astigmatic condenser design.  It intentionally does not
simulate coherent wave propagation, specimen scattering, counting noise, or
hardware response.  Consequently these figures are planning previews, never
GPU-wave validation or instrument evidence.

The preview uses the nominal hard-edge covariance model at the thirteen
selected states.  A representative state is selected by the largest angular
major-axis RMS among states which pass the manifest's strict sample-size and
nominal native-detector geometry filters.  This makes the choice deterministic
and exposes the most demanding accepted diffraction-disc support.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

from .condenser_astigmatic_design import (
    ASTIGMATIC_DESIGN_SCHEMA,
    AstigmaticCondenserDesign,
    AstigmaticDesignAudit,
    AstigmaticDesignProvenanceError,
    load_astigmatic_condenser_design,
    recompute_astigmatic_design_audit,
)
from .condenser_branch_b import load_branch_b_experiment
from .critical_line_calibration import (
    axial_plane_covariance_observables_with_entrance_covariance,
)
from .plotting import NEOARM_RAY_COLORS, export_ray_figure


PLANNING_PREVIEW_SCHEMA = (
    "temgym.neoarm.condenser-astigmatic-observable-planning-preview/v1"
)
PLANNING_PREVIEW_DISCLAIMER = (
    "OBSERVABLE-LEVEL PLANNING PREVIEW · NOT GPU WAVE VALIDATION · "
    "NOT HARDWARE EVIDENCE"
)
DEFAULT_SI_LATTICE_AZIMUTH_RAD = np.deg2rad(18.0)
SI_111_BASIS_SEPARATION_RAD = float(np.arccos(1.0 / 3.0))

_EXPECTED_UNIQUE_STATES = 13
_EXPECTED_ACQUISITIONS = 48
_EXPECTED_SUMMARY_SIZE = 10


class AstigmaticPlanningPreviewError(ValueError):
    """Raised when a plotting input violates the audited planning contract."""


def _frozen_array(
    value: object,
    name: str,
    *,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if shape is not None and array.shape != shape:
        raise AstigmaticPlanningPreviewError(
            f"{name} must have shape {shape}; got {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise AstigmaticPlanningPreviewError(f"{name} must be finite.")
    result = array.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class AstigmaticPlanningPreview:
    """Immutable numerical content for the three-panel planning figure.

    Covariance matrices use conventional Cartesian ``(x, y)`` ordering.
    Raster arrays use conventional image ``(y, x)`` ordering.  The diffraction
    image is a relative peak-density preview; it contains no claimed absolute
    scattering intensity.
    """

    design_manifest_path: Path
    design_manifest_sha256: str
    coordinate_status: str
    scope: str
    selected_state_index: int
    selected_repeat_count: int
    selected_control_CL1_CL3_Cmini: np.ndarray
    strict_sample_equivalent_rms_filter_m: np.ndarray
    strict_state_mask: np.ndarray
    spatial_covariances_m2: np.ndarray
    angular_covariances_rad2: np.ndarray
    spatial_axis_rms_m: np.ndarray
    angular_axis_rms_rad: np.ndarray
    angular_axis_rms_px: np.ndarray
    spatial_orientation_rad: np.ndarray
    angular_orientation_rad: np.ndarray
    relative_spatial_orientation_rad: np.ndarray
    relative_angular_orientation_rad: np.ndarray
    repeat_counts: np.ndarray
    realspace_x_m: np.ndarray
    realspace_y_m: np.ndarray
    realspace_relative_intensity_yx: np.ndarray
    detector_x_m: np.ndarray
    detector_y_m: np.ndarray
    diffraction_relative_density_yx: np.ndarray
    bragg_vectors_px_xy: np.ndarray
    bragg_labels: tuple[str, str]
    lattice_azimuth_rad: float
    si_111_radius_px: float
    camera_length_m: float
    detector_pixel_size_m: float
    detector_mtf_sigma_px: float
    covariance_summary_sigma: float
    tag_orientation_sigma_rad: float
    audit_target_rank: int
    audit_maximum_scaled_standard_uncertainty: float
    nominal_detector_geometry_pass: bool
    hardware_ready: bool = False

    def __post_init__(self) -> None:
        arrays = {
            "selected_control_CL1_CL3_Cmini": (
                self.selected_control_CL1_CL3_Cmini,
                (3,),
            ),
            "strict_sample_equivalent_rms_filter_m": (
                self.strict_sample_equivalent_rms_filter_m,
                (2,),
            ),
            "spatial_covariances_m2": (
                self.spatial_covariances_m2,
                (_EXPECTED_UNIQUE_STATES, 2, 2),
            ),
            "angular_covariances_rad2": (
                self.angular_covariances_rad2,
                (_EXPECTED_UNIQUE_STATES, 2, 2),
            ),
            "spatial_axis_rms_m": (
                self.spatial_axis_rms_m,
                (_EXPECTED_UNIQUE_STATES, 2),
            ),
            "angular_axis_rms_rad": (
                self.angular_axis_rms_rad,
                (_EXPECTED_UNIQUE_STATES, 2),
            ),
            "angular_axis_rms_px": (
                self.angular_axis_rms_px,
                (_EXPECTED_UNIQUE_STATES, 2),
            ),
            "spatial_orientation_rad": (
                self.spatial_orientation_rad,
                (_EXPECTED_UNIQUE_STATES,),
            ),
            "angular_orientation_rad": (
                self.angular_orientation_rad,
                (_EXPECTED_UNIQUE_STATES,),
            ),
            "relative_spatial_orientation_rad": (
                self.relative_spatial_orientation_rad,
                (_EXPECTED_UNIQUE_STATES,),
            ),
            "relative_angular_orientation_rad": (
                self.relative_angular_orientation_rad,
                (_EXPECTED_UNIQUE_STATES,),
            ),
            "realspace_x_m": (self.realspace_x_m, None),
            "realspace_y_m": (self.realspace_y_m, None),
            "detector_x_m": (self.detector_x_m, None),
            "detector_y_m": (self.detector_y_m, None),
            "bragg_vectors_px_xy": (self.bragg_vectors_px_xy, (2, 2)),
        }
        frozen: dict[str, np.ndarray] = {}
        for name, (values, shape) in arrays.items():
            frozen[name] = _frozen_array(values, name, shape=shape)

        strict = np.asarray(self.strict_state_mask, dtype=bool)
        if strict.shape != (_EXPECTED_UNIQUE_STATES,):
            raise AstigmaticPlanningPreviewError(
                "strict_state_mask must have shape (13,)."
            )
        strict = strict.copy()
        strict.setflags(write=False)
        object.__setattr__(self, "strict_state_mask", strict)

        repeats = np.asarray(self.repeat_counts, dtype=int)
        if (
            repeats.shape != (_EXPECTED_UNIQUE_STATES,)
            or np.any(repeats <= 0)
            or int(np.sum(repeats)) != _EXPECTED_ACQUISITIONS
        ):
            raise AstigmaticPlanningPreviewError(
                "repeat_counts must encode 48 positive acquisitions at 13 states."
            )
        repeats = repeats.copy()
        repeats.setflags(write=False)
        object.__setattr__(self, "repeat_counts", repeats)

        realspace_shape = (
            frozen["realspace_y_m"].size,
            frozen["realspace_x_m"].size,
        )
        detector_shape = (
            frozen["detector_y_m"].size,
            frozen["detector_x_m"].size,
        )
        frozen["realspace_relative_intensity_yx"] = _frozen_array(
            self.realspace_relative_intensity_yx,
            "realspace_relative_intensity_yx",
            shape=realspace_shape,
        )
        frozen["diffraction_relative_density_yx"] = _frozen_array(
            self.diffraction_relative_density_yx,
            "diffraction_relative_density_yx",
            shape=detector_shape,
        )
        for name, value in frozen.items():
            object.__setattr__(self, name, value)

        if not 0 <= int(self.selected_state_index) < _EXPECTED_UNIQUE_STATES:
            raise AstigmaticPlanningPreviewError("selected_state_index is invalid.")
        if not strict[int(self.selected_state_index)]:
            raise AstigmaticPlanningPreviewError(
                "The selected state must satisfy every strict planning filter."
            )
        if int(self.selected_repeat_count) <= 0:
            raise AstigmaticPlanningPreviewError(
                "selected_repeat_count must be positive."
            )
        positive = (
            self.si_111_radius_px,
            self.camera_length_m,
            self.detector_pixel_size_m,
            self.detector_mtf_sigma_px,
            self.covariance_summary_sigma,
            self.tag_orientation_sigma_rad,
            self.audit_maximum_scaled_standard_uncertainty,
        )
        if not np.all(np.isfinite(positive)) or min(positive) <= 0.0:
            raise AstigmaticPlanningPreviewError(
                "All preview scales must be finite and positive."
            )
        if bool(self.hardware_ready):
            raise AstigmaticPlanningPreviewError(
                "A planning preview cannot be marked hardware ready."
            )
        if not bool(self.nominal_detector_geometry_pass):
            raise AstigmaticPlanningPreviewError(
                "The selected design does not pass its nominal geometry audit."
            )
        if np.max(frozen["realspace_relative_intensity_yx"]) > 1.0 + 1.0e-12:
            raise AstigmaticPlanningPreviewError(
                "Real-space relative intensity must not exceed one."
            )
        if np.max(frozen["diffraction_relative_density_yx"]) > 1.0 + 1.0e-12:
            raise AstigmaticPlanningPreviewError(
                "Diffraction relative density must not exceed one."
            )
        if abs(np.linalg.det(frozen["bragg_vectors_px_xy"])) <= 1.0e-12:
            raise AstigmaticPlanningPreviewError(
                "The indexed-Si basis vectors must be noncollinear."
            )

    @property
    def selected_spatial_axis_rms_m(self) -> np.ndarray:
        return self.spatial_axis_rms_m[self.selected_state_index]

    @property
    def selected_angular_axis_rms_rad(self) -> np.ndarray:
        return self.angular_axis_rms_rad[self.selected_state_index]

    @property
    def selected_angular_axis_rms_px(self) -> np.ndarray:
        return self.angular_axis_rms_px[self.selected_state_index]

    def export_metadata(self) -> dict[str, object]:
        """Return explicit, JSON-safe provenance for figure export."""

        return {
            "schema": PLANNING_PREVIEW_SCHEMA,
            "title": "Conditional astigmatic no-tilt planning preview",
            "status": PLANNING_PREVIEW_DISCLAIMER,
            "evidence_class": "nominal observable-level planning only",
            "gpu_wave_validation": False,
            "hardware_evidence": False,
            "hardware_ready": False,
            "design_manifest": str(self.design_manifest_path),
            "design_manifest_sha256": self.design_manifest_sha256,
            "design_scope": self.scope,
            "coordinate_status": self.coordinate_status,
            "selected_state_index": int(self.selected_state_index),
            "selected_repeat_count": int(self.selected_repeat_count),
            "selected_normalized_controls_CL1_CL3_Cmini": (
                self.selected_control_CL1_CL3_Cmini.tolist()
            ),
            "selection_rule": (
                "largest angular major-axis RMS among states passing the "
                "manifest strict sample-size and native-detector geometry filters"
            ),
            "strict_sample_equivalent_rms_filter_m": (
                self.strict_sample_equivalent_rms_filter_m.tolist()
            ),
            "camera_length_m": float(self.camera_length_m),
            "detector_shape_yx": [
                int(self.detector_y_m.size),
                int(self.detector_x_m.size),
            ],
            "detector_pixel_size_m": float(self.detector_pixel_size_m),
            "detector_mtf_sigma_px": float(self.detector_mtf_sigma_px),
            "si_111_radius_px": float(self.si_111_radius_px),
            "si_basis_labels": list(self.bragg_labels),
            "si_basis_separation_deg": float(
                np.rad2deg(SI_111_BASIS_SEPARATION_RAD)
            ),
            "assumed_registered_lattice_azimuth_deg": float(
                np.rad2deg(self.lattice_azimuth_rad)
            ),
            "selected_spatial_axis_rms_nm": (
                self.selected_spatial_axis_rms_m * 1.0e9
            ).tolist(),
            "selected_angular_axis_rms_mrad": (
                self.selected_angular_axis_rms_rad * 1.0e3
            ).tolist(),
            "selected_angular_axis_rms_px": (
                self.selected_angular_axis_rms_px.tolist()
            ),
            "audit_target_rank": int(self.audit_target_rank),
            "audit_maximum_scaled_standard_uncertainty": float(
                self.audit_maximum_scaled_standard_uncertainty
            ),
            "nominal_detector_geometry_pass": bool(
                self.nominal_detector_geometry_pass
            ),
            "intensity_contract": (
                "relative deterministic Gaussian peak density; no absolute "
                "specimen scattering, dose, counting noise, or detector response"
            ),
        }


def _manifest_strict_sample_filter(design: AstigmaticCondenserDesign) -> np.ndarray:
    document = json.loads(design.path.read_text(encoding="utf-8"))
    if document.get("schema") != ASTIGMATIC_DESIGN_SCHEMA:
        raise AstigmaticPlanningPreviewError("Unsupported design manifest schema.")
    try:
        limits = document["provenance"]["candidate_search"][
            "strict_sample_equivalent_rms_filter_m"
        ]
    except KeyError as error:
        raise AstigmaticPlanningPreviewError(
            "The design manifest has no strict sample-size filter."
        ) from error
    limits_array = _frozen_array(limits, "strict sample filter", shape=(2,))
    if limits_array[0] <= 0.0 or limits_array[1] <= limits_array[0]:
        raise AstigmaticPlanningPreviewError("The strict sample filter is invalid.")
    return limits_array


def nominal_astigmatic_covariance_summaries(
    design: AstigmaticCondenserDesign,
) -> np.ndarray:
    """Evaluate only the nominal 48-state covariance summaries.

    Unlike :func:`build_astigmatic_design_matrix`, this helper does not form
    parameter Jacobians.  It is intended for plotting after the separate
    Fisher audit has already established the local design contract.
    """

    if not isinstance(design, AstigmaticCondenserDesign):
        raise TypeError("design must be an AstigmaticCondenserDesign.")
    jax.config.update("jax_enable_x64", True)
    experiment = load_branch_b_experiment(design.branch_b_payload_path)
    if experiment.payload_sha256 != design.branch_b_payload_sha256:
        raise AstigmaticDesignProvenanceError(
            "The loaded Branch-B experiment differs from the bound design."
        )
    summaries = np.asarray(
        axial_plane_covariance_observables_with_entrance_covariance(
            experiment.column,
            jnp.asarray(experiment.column.nominal_parameters),
            jnp.zeros(10, dtype=jnp.float64),
            jnp.asarray(design.expanded_controls),
            experiment.entrance_covariance_chart,
            plane_z_m=experiment.sample_z_m,
            mode="full",
        ),
        dtype=float,
    )
    return _frozen_array(
        summaries,
        "nominal covariance summaries",
        shape=(_EXPECTED_ACQUISITIONS, _EXPECTED_SUMMARY_SIZE),
    )


def _collapse_repeated_summaries(
    design: AstigmaticCondenserDesign,
    summaries: np.ndarray,
) -> np.ndarray:
    repeated = _frozen_array(
        summaries,
        "summaries",
        shape=(_EXPECTED_ACQUISITIONS, _EXPECTED_SUMMARY_SIZE),
    )
    starts = np.concatenate(((0,), np.cumsum(design.repeat_counts)[:-1]))
    unique = repeated[starts]
    for state, (start, count) in enumerate(zip(starts, design.repeat_counts)):
        block = repeated[int(start) : int(start + count)]
        if not np.allclose(block, unique[state], rtol=2.0e-12, atol=2.0e-13):
            raise AstigmaticPlanningPreviewError(
                f"Repeated summaries for state {state} are not identical."
            )
    return _frozen_array(
        unique,
        "unique summaries",
        shape=(_EXPECTED_UNIQUE_STATES, _EXPECTED_SUMMARY_SIZE),
    )


def _covariance_from_chart(
    summaries: np.ndarray,
    indices: tuple[int, int, int],
) -> np.ndarray:
    log_x = summaries[:, indices[0]]
    log_y = summaries[:, indices[1]]
    correlation = summaries[:, indices[2]]
    if np.any(np.abs(correlation) >= 1.0):
        raise AstigmaticPlanningPreviewError(
            "A nominal covariance correlation is outside (-1, 1)."
        )
    sigma_x = np.exp(log_x)
    sigma_y = np.exp(log_y)
    covariance = np.empty((summaries.shape[0], 2, 2), dtype=float)
    covariance[:, 0, 0] = sigma_x**2
    covariance[:, 1, 1] = sigma_y**2
    covariance[:, 0, 1] = correlation * sigma_x * sigma_y
    covariance[:, 1, 0] = covariance[:, 0, 1]
    return covariance


def _axes_and_orientation(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.any(eigenvalues <= 0.0):
        raise AstigmaticPlanningPreviewError(
            "The nominal covariance is not positive definite."
        )
    axes = np.sqrt(eigenvalues)
    major_vectors = eigenvectors[:, :, 1]
    angles = np.arctan2(major_vectors[:, 1], major_vectors[:, 0])
    angles = (angles + 0.5 * np.pi) % np.pi - 0.5 * np.pi
    return axes, angles


def _relative_unoriented_angle(angles: np.ndarray) -> np.ndarray:
    # Principal axes are unoriented: theta and theta + pi are identical.
    unwrapped = 0.5 * np.unwrap(2.0 * np.asarray(angles, dtype=float))
    return unwrapped - unwrapped[0]


def _gaussian_on_grid(
    x: np.ndarray,
    y: np.ndarray,
    covariance: np.ndarray,
    *,
    center_xy: Sequence[float] = (0.0, 0.0),
) -> np.ndarray:
    xx, yy = np.meshgrid(x, y, indexing="xy")
    delta = np.stack((xx - center_xy[0], yy - center_xy[1]), axis=-1)
    precision = np.linalg.inv(covariance)
    exponent = np.einsum("...i,ij,...j->...", delta, precision, delta)
    return np.exp(-0.5 * exponent)


def _build_diffraction_density(
    detector_x_px: np.ndarray,
    detector_y_px: np.ndarray,
    angular_covariance_px2: np.ndarray,
    *,
    mtf_sigma_px: float,
    bragg_vectors_px_xy: np.ndarray,
) -> np.ndarray:
    # The display includes the manifest's nominal Gaussian MTF blur, but the
    # reported angular widths remain the pre-MTF observable widths.
    displayed_covariance = angular_covariance_px2 + (
        float(mtf_sigma_px) ** 2 * np.eye(2)
    )
    centers = np.concatenate(
        (
            np.zeros((1, 2)),
            bragg_vectors_px_xy,
            -bragg_vectors_px_xy,
        ),
        axis=0,
    )
    relative_weights = np.asarray((1.0, 0.32, 0.25, 0.32, 0.25))
    density = np.zeros((detector_y_px.size, detector_x_px.size), dtype=float)
    for center, weight in zip(centers, relative_weights):
        density += weight * _gaussian_on_grid(
            detector_x_px,
            detector_y_px,
            displayed_covariance,
            center_xy=center,
        )
    maximum = float(np.max(density))
    if maximum <= 0.0 or not np.isfinite(maximum):
        raise AstigmaticPlanningPreviewError(
            "The nominal diffraction density could not be normalised."
        )
    return density / maximum


def build_astigmatic_planning_preview(
    design: AstigmaticCondenserDesign | None = None,
    *,
    audit: AstigmaticDesignAudit | None = None,
    summaries: np.ndarray | None = None,
    lattice_azimuth_rad: float = DEFAULT_SI_LATTICE_AZIMUTH_RAD,
    realspace_grid_points: int = 401,
) -> AstigmaticPlanningPreview:
    """Build a deterministic real-space/diffraction planning preview.

    Pass an already recomputed ``audit`` and nominal ``summaries`` in a
    notebook to avoid repeating work performed by earlier audit cells.  When
    omitted, both are obtained from the checksum-verified default design.
    """

    selected_design = (
        load_astigmatic_condenser_design() if design is None else design
    )
    if not isinstance(selected_design, AstigmaticCondenserDesign):
        raise TypeError("design must be an AstigmaticCondenserDesign.")
    selected_audit = (
        recompute_astigmatic_design_audit(selected_design)
        if audit is None
        else audit
    )
    if not isinstance(selected_audit, AstigmaticDesignAudit):
        raise TypeError("audit must be an AstigmaticDesignAudit.")
    if selected_design.hardware_ready or selected_audit.hardware_ready:
        raise AstigmaticPlanningPreviewError(
            "The conditional design must remain explicitly fail closed."
        )
    if not selected_audit.conditional_precision_pass:
        raise AstigmaticPlanningPreviewError(
            "The conditional local precision audit no longer passes."
        )
    if not selected_audit.nominal_detector_geometry_pass:
        raise AstigmaticPlanningPreviewError(
            "The nominal native-detector geometry audit no longer passes."
        )
    if not np.isfinite(lattice_azimuth_rad):
        raise AstigmaticPlanningPreviewError("lattice_azimuth_rad must be finite.")
    realspace_grid_points = int(realspace_grid_points)
    if realspace_grid_points < 129 or realspace_grid_points % 2 == 0:
        raise AstigmaticPlanningPreviewError(
            "realspace_grid_points must be an odd integer of at least 129."
        )

    repeated_summaries = (
        nominal_astigmatic_covariance_summaries(selected_design)
        if summaries is None
        else summaries
    )
    unique = _collapse_repeated_summaries(selected_design, repeated_summaries)
    spatial_covariance = _covariance_from_chart(unique, (0, 2, 5))
    angular_covariance = _covariance_from_chart(unique, (1, 3, 8))
    spatial_axes, spatial_angles = _axes_and_orientation(spatial_covariance)
    angular_axes, angular_angles = _axes_and_orientation(angular_covariance)

    px_per_rad = (
        selected_design.planned_camera_length_m
        / selected_design.detector_pixel_size_m
    )
    angular_axes_px = angular_axes * px_per_rad
    equivalent_spatial_rms = np.sqrt(
        0.5 * np.trace(spatial_covariance, axis1=1, axis2=2)
    )
    strict_limits = _manifest_strict_sample_filter(selected_design)
    half_detector_px = 0.5 * min(selected_design.detector_shape_yx)
    radius_px = float(selected_audit.si_111_radius_px)
    geometry_mask = (
        (angular_axes_px[:, 0] >= 1.5 * selected_design.detector_mtf_sigma_px)
        & (radius_px + 3.0 * angular_axes_px[:, 1] < half_detector_px)
        & (6.0 * angular_axes_px[:, 1] < radius_px)
        & (radius_px >= 3.0)
        & (radius_px <= 96.0)
    )
    strict_mask = (
        (equivalent_spatial_rms >= strict_limits[0])
        & (equivalent_spatial_rms <= strict_limits[1])
        & geometry_mask
    )
    candidates = np.flatnonzero(strict_mask)
    if candidates.size == 0:
        raise AstigmaticPlanningPreviewError(
            "No selected optical state passes every strict planning filter."
        )
    selected_state = int(
        candidates[np.argmax(angular_axes_px[candidates, 1])]
    )

    # Four major-axis standard deviations give an honest, symmetric
    # sample-plane field of view.  The odd grid places the beam centroid on a
    # pixel centre.
    realspace_half_width = 4.0 * float(spatial_axes[selected_state, 1])
    realspace_x = np.linspace(
        -realspace_half_width,
        realspace_half_width,
        realspace_grid_points,
    )
    realspace_y = realspace_x.copy()
    realspace_intensity = _gaussian_on_grid(
        realspace_x,
        realspace_y,
        spatial_covariance[selected_state],
    )

    detector_ny, detector_nx = selected_design.detector_shape_yx
    detector_x_px = np.arange(detector_nx, dtype=float) - 0.5 * (detector_nx - 1)
    detector_y_px = np.arange(detector_ny, dtype=float) - 0.5 * (detector_ny - 1)
    basis_angles = np.asarray(
        (
            float(lattice_azimuth_rad),
            float(lattice_azimuth_rad) + SI_111_BASIS_SEPARATION_RAD,
        )
    )
    bragg_vectors = radius_px * np.stack(
        (np.cos(basis_angles), np.sin(basis_angles)), axis=1
    )
    diffraction = _build_diffraction_density(
        detector_x_px,
        detector_y_px,
        angular_covariance[selected_state] * px_per_rad**2,
        mtf_sigma_px=selected_design.detector_mtf_sigma_px,
        bragg_vectors_px_xy=bragg_vectors,
    )

    spatial_range = np.asarray(
        (np.min(equivalent_spatial_rms), np.max(equivalent_spatial_rms))
    )
    angular_range = np.stack(
        (np.min(angular_axes, axis=0), np.max(angular_axes, axis=0))
    )
    np.testing.assert_allclose(
        spatial_range,
        selected_audit.sample_equivalent_rms_range_m,
        rtol=3.0e-8,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        angular_range,
        selected_audit.angular_eigen_rms_range_rad,
        rtol=3.0e-8,
        atol=1.0e-14,
    )

    return AstigmaticPlanningPreview(
        design_manifest_path=selected_design.path,
        design_manifest_sha256=selected_design.sha256,
        coordinate_status=selected_design.coordinate_status,
        scope=selected_design.scope,
        selected_state_index=selected_state,
        selected_repeat_count=int(selected_design.repeat_counts[selected_state]),
        selected_control_CL1_CL3_Cmini=selected_design.controls[selected_state, :3],
        strict_sample_equivalent_rms_filter_m=strict_limits,
        strict_state_mask=strict_mask,
        spatial_covariances_m2=spatial_covariance,
        angular_covariances_rad2=angular_covariance,
        spatial_axis_rms_m=spatial_axes,
        angular_axis_rms_rad=angular_axes,
        angular_axis_rms_px=angular_axes_px,
        spatial_orientation_rad=spatial_angles,
        angular_orientation_rad=angular_angles,
        relative_spatial_orientation_rad=_relative_unoriented_angle(
            spatial_angles
        ),
        relative_angular_orientation_rad=_relative_unoriented_angle(
            angular_angles
        ),
        repeat_counts=selected_design.repeat_counts,
        realspace_x_m=realspace_x,
        realspace_y_m=realspace_y,
        realspace_relative_intensity_yx=realspace_intensity,
        detector_x_m=(detector_x_px * selected_design.detector_pixel_size_m),
        detector_y_m=(detector_y_px * selected_design.detector_pixel_size_m),
        diffraction_relative_density_yx=diffraction,
        bragg_vectors_px_xy=bragg_vectors,
        bragg_labels=(r"Si $(111)$", r"Si $(1\bar{1}1)$"),
        lattice_azimuth_rad=float(lattice_azimuth_rad),
        si_111_radius_px=radius_px,
        camera_length_m=selected_design.planned_camera_length_m,
        detector_pixel_size_m=selected_design.detector_pixel_size_m,
        detector_mtf_sigma_px=selected_design.detector_mtf_sigma_px,
        covariance_summary_sigma=selected_design.covariance_summary_sigma,
        tag_orientation_sigma_rad=selected_design.tag_orientation_sigma_rad,
        audit_target_rank=selected_audit.target_rank,
        audit_maximum_scaled_standard_uncertainty=(
            selected_audit.maximum_scaled_standard_uncertainty
        ),
        nominal_detector_geometry_pass=(
            selected_audit.nominal_detector_geometry_pass
        ),
        hardware_ready=False,
    )


def _image_extent(coordinates: np.ndarray) -> tuple[float, float]:
    if coordinates.ndim != 1 or coordinates.size < 2:
        raise AstigmaticPlanningPreviewError("Image coordinates are incomplete.")
    step = float(np.median(np.diff(coordinates)))
    if step <= 0.0 or not np.allclose(np.diff(coordinates), step):
        raise AstigmaticPlanningPreviewError(
            "Image coordinates must be uniformly increasing."
        )
    return float(coordinates[0] - 0.5 * step), float(
        coordinates[-1] + 0.5 * step
    )


def _add_horizontal_scale_bar(
    ax: mpl.axes.Axes,
    *,
    length: float,
    label: str,
    color: str = "white",
) -> None:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x0 = xmin + 0.065 * (xmax - xmin)
    y0 = ymin + 0.075 * (ymax - ymin)
    ax.plot(
        (x0, x0 + length),
        (y0, y0),
        color=color,
        lw=3.0,
        solid_capstyle="butt",
        zorder=20,
    )
    ax.text(
        x0 + 0.5 * length,
        y0 + 0.025 * (ymax - ymin),
        label,
        color=color,
        ha="center",
        va="bottom",
        fontsize=8.0,
        weight="bold",
        zorder=20,
    )


def _ellipse_patch(
    covariance: np.ndarray,
    *,
    center_xy: Sequence[float] = (0.0, 0.0),
    sigma: float = 1.0,
    scale: float = 1.0,
    **kwargs: object,
) -> Ellipse:
    axes, angle = _axes_and_orientation(np.asarray(covariance)[None, :, :])
    minor, major = axes[0] * float(scale)
    return Ellipse(
        xy=center_xy,
        width=2.0 * sigma * major,
        height=2.0 * sigma * minor,
        angle=float(np.rad2deg(angle[0])),
        fill=False,
        **kwargs,
    )


def plot_astigmatic_planning_preview(
    preview: AstigmaticPlanningPreview,
    *,
    figsize: tuple[float, float] = (16.2, 5.7),
) -> tuple[mpl.figure.Figure, np.ndarray]:
    """Render the real-space, diffraction, and relative-orientation panels."""

    if not isinstance(preview, AstigmaticPlanningPreview):
        raise TypeError("preview must be an AstigmaticPlanningPreview.")
    colors = NEOARM_RAY_COLORS
    with mpl.rc_context(
        {
            "font.size": 9.0,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    ):
        fig, axes = plt.subplots(
            1,
            3,
            figsize=figsize,
            width_ratios=(1.0, 1.0, 1.18),
            layout="constrained",
        )
        fig.get_layout_engine().set(h_pad=0.16, w_pad=0.08, hspace=0.02)

        state = preview.selected_state_index
        spatial_cov_nm2 = preview.spatial_covariances_m2[state] * 1.0e18
        x_nm = preview.realspace_x_m * 1.0e9
        y_nm = preview.realspace_y_m * 1.0e9
        xmin, xmax = _image_extent(x_nm)
        ymin, ymax = _image_extent(y_nm)
        real_image = axes[0].imshow(
            preview.realspace_relative_intensity_yx,
            origin="lower",
            extent=(xmin, xmax, ymin, ymax),
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            rasterized=True,
        )
        for sigma, linestyle, alpha in ((1.0, "-", 0.95), (2.0, "--", 0.8)):
            axes[0].add_patch(
                _ellipse_patch(
                    spatial_cov_nm2,
                    sigma=sigma,
                    edgecolor="white",
                    lw=1.2,
                    linestyle=linestyle,
                    alpha=alpha,
                )
            )
        axes[0].plot(0.0, 0.0, "+", color="white", ms=8, mew=1.2)
        axes[0].set(
            xlabel="sample x (nm)",
            ylabel="sample y (nm)",
            title=(
                f"(a) Sample-plane astigmatic beam · state {state:02d}\n"
                "relative incident intensity; 1σ/2σ covariance"
            ),
        )
        axes[0].set_aspect("equal")
        controls = preview.selected_control_CL1_CL3_Cmini
        axes[0].text(
            0.025,
            0.975,
            (
                "normalized (CL1, CL3, C-mini)\n"
                f"= ({controls[0]:+.3f}, {controls[1]:+.3f}, "
                f"{controls[2]:+.3f}); repeats = {preview.selected_repeat_count}"
            ),
            transform=axes[0].transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            color="#111827",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.86,
            },
            zorder=30,
        )
        _add_horizontal_scale_bar(
            axes[0], length=250.0, label="250 nm", color="white"
        )
        real_cbar = fig.colorbar(real_image, ax=axes[0], fraction=0.047, pad=0.025)
        real_cbar.set_label("relative intensity")

        detector_x_mm = preview.detector_x_m * 1.0e3
        detector_y_mm = preview.detector_y_m * 1.0e3
        dx0, dx1 = _image_extent(detector_x_mm)
        dy0, dy1 = _image_extent(detector_y_mm)
        diffraction_image = axes[1].imshow(
            preview.diffraction_relative_density_yx,
            origin="lower",
            extent=(dx0, dx1, dy0, dy1),
            cmap="cividis",
            norm=mpl.colors.PowerNorm(gamma=0.45, vmin=0.0, vmax=1.0),
            interpolation="nearest",
            rasterized=True,
        )
        pixel_mm = preview.detector_pixel_size_m * 1.0e3
        vector_colors = (colors["position"], colors["angle"])
        angular_cov_px2 = (
            preview.angular_covariances_rad2[state]
            * (preview.camera_length_m / preview.detector_pixel_size_m) ** 2
        )
        for vector, label, color in zip(
            preview.bragg_vectors_px_xy,
            preview.bragg_labels,
            vector_colors,
        ):
            vector_mm = vector * pixel_mm
            axes[1].annotate(
                "",
                xy=vector_mm,
                xytext=(0.0, 0.0),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "lw": 1.45,
                    "shrinkA": 2.0,
                    "shrinkB": 1.0,
                },
                zorder=15,
            )
            axes[1].text(
                *(1.10 * vector_mm),
                label,
                color=color,
                fontsize=8.5,
                weight="bold",
                ha="center",
                va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76},
                zorder=16,
            )
            axes[1].add_patch(
                _ellipse_patch(
                    angular_cov_px2,
                    center_xy=vector_mm,
                    sigma=2.0,
                    scale=pixel_mm,
                    edgecolor="white",
                    lw=1.15,
                    linestyle="--",
                    alpha=0.95,
                    zorder=14,
                )
            )
        axes[1].plot(0.0, 0.0, "+", color="white", ms=7, mew=1.1)
        axes[1].set(
            xlabel="detector x (mm)",
            ylabel="detector y (mm)",
            title=(
                "(b) Indexed-Si diffraction geometry · L = 0.400 m\n"
                "native 256×256, 55 µm pixels; dashed = angular 2σ"
            ),
        )
        axes[1].set_aspect("equal")
        minor_mrad, major_mrad = preview.selected_angular_axis_rms_rad * 1.0e3
        minor_px, major_px = preview.selected_angular_axis_rms_px
        axes[1].text(
            0.025,
            0.975,
            (
                f"Si {{111}} radius = {preview.si_111_radius_px:.2f} px\n"
                f"angular 1σ = {minor_mrad:.3f}, {major_mrad:.3f} mrad\n"
                f"projected 1σ = {minor_px:.2f}, {major_px:.2f} px\n"
                f"display MTF σ = {preview.detector_mtf_sigma_px:.2f} px\n"
                f"assumed registered azimuth = "
                f"{np.rad2deg(preview.lattice_azimuth_rad):.1f}°"
            ),
            transform=axes[1].transAxes,
            ha="left",
            va="top",
            fontsize=7.7,
            color="#111827",
            bbox={
                "boxstyle": "round,pad=0.34",
                "facecolor": "white",
                "edgecolor": "#CBD5E1",
                "alpha": 0.92,
            },
            zorder=30,
        )
        _add_horizontal_scale_bar(
            axes[1], length=2.0, label="2 mm", color="white"
        )
        diffraction_cbar = fig.colorbar(
            diffraction_image, ax=axes[1], fraction=0.047, pad=0.025
        )
        diffraction_cbar.set_label("relative planning density")

        states = np.arange(_EXPECTED_UNIQUE_STATES)
        spatial_relative_deg = np.rad2deg(
            preview.relative_spatial_orientation_rad
        )
        angular_relative_deg = np.rad2deg(
            preview.relative_angular_orientation_rad
        )
        marker_sizes = 23.0 + 5.5 * np.sqrt(preview.repeat_counts)
        axes[2].axhline(0.0, color="#9CA3AF", lw=0.8, zorder=0)
        axes[2].axvspan(
            state - 0.35,
            state + 0.35,
            color="#E5E7EB",
            alpha=0.75,
            zorder=0,
            label=f"selected state {state:02d}",
        )
        axes[2].plot(
            states,
            spatial_relative_deg,
            color=colors["position"],
            lw=1.45,
            alpha=0.9,
            zorder=2,
        )
        axes[2].scatter(
            states,
            spatial_relative_deg,
            s=marker_sizes,
            facecolor="white",
            edgecolor=colors["position"],
            linewidth=1.35,
            label="sample-plane ellipse",
            zorder=3,
        )
        axes[2].plot(
            states,
            angular_relative_deg,
            color=colors["angle"],
            lw=1.45,
            alpha=0.9,
            zorder=2,
        )
        axes[2].scatter(
            states,
            angular_relative_deg,
            s=marker_sizes,
            marker="s",
            facecolor="white",
            edgecolor=colors["angle"],
            linewidth=1.35,
            label="angular ellipse",
            zorder=3,
        )
        axes[2].scatter(
            (state, state),
            (spatial_relative_deg[state], angular_relative_deg[state]),
            s=marker_sizes[state] + 75.0,
            facecolor="none",
            edgecolor="#111827",
            linewidth=1.1,
            zorder=4,
        )
        axes[2].set(
            xlim=(-0.55, _EXPECTED_UNIQUE_STATES - 0.45),
            xticks=states,
            xlabel="selected normalized optical state",
            ylabel="principal-axis change from state 00 (deg, modulo 180°)",
            title=(
                "(c) State-relative ellipse orientation\n"
                "marker area increases with planned repeat count"
            ),
        )
        axes[2].grid(axis="y", color="#E5E7EB", lw=0.7)
        axes[2].legend(loc="best", fontsize=7.9)
        axes[2].text(
            0.02,
            0.02,
            (
                f"orientation summary σ = "
                f"{np.rad2deg(preview.tag_orientation_sigma_rad):.3f}°\n"
                "common absolute angle is profiled"
            ),
            transform=axes[2].transAxes,
            ha="left",
            va="bottom",
            fontsize=7.7,
            color="#374151",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "#D1D5DB",
                "alpha": 0.92,
            },
        )

        fig.suptitle(
            (
                "Conditional deliberately astigmatic no-commanded-tilt pilot\n"
                f"{PLANNING_PREVIEW_DISCLAIMER}"
            ),
            color="#991B1B",
            fontsize=11.5,
            weight="bold",
        )
        fig.text(
            0.5,
            0.002,
            (
                "Controls are normalized synthetic model coordinates, not "
                "certified NEOARM DAC commands. Diffraction intensity is a "
                "relative Gaussian geometry preview, not specimen scattering."
            ),
            ha="center",
            va="bottom",
            color="#4B5563",
            fontsize=7.6,
        )
    return fig, np.asarray(axes, dtype=object)


def export_astigmatic_planning_preview(
    preview: AstigmaticPlanningPreview,
    path_stem: str | Path,
    *,
    formats: Sequence[str] = ("png", "svg", "pdf"),
    dpi: int = 300,
    close: bool = False,
) -> dict[str, Path]:
    """Render and export the preview with fail-closed provenance metadata."""

    fig, _axes = plot_astigmatic_planning_preview(preview)
    outputs = export_ray_figure(
        fig,
        path_stem,
        formats=formats,
        dpi=dpi,
        metadata=preview.export_metadata(),
    )
    if close:
        plt.close(fig)
    return outputs


__all__ = [
    "AstigmaticPlanningPreview",
    "AstigmaticPlanningPreviewError",
    "DEFAULT_SI_LATTICE_AZIMUTH_RAD",
    "PLANNING_PREVIEW_DISCLAIMER",
    "PLANNING_PREVIEW_SCHEMA",
    "SI_111_BASIS_SEPARATION_RAD",
    "build_astigmatic_planning_preview",
    "export_astigmatic_planning_preview",
    "nominal_astigmatic_covariance_summaries",
    "plot_astigmatic_planning_preview",
]
