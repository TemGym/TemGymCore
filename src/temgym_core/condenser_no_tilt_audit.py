"""Fail-closed audit for a condenser experiment without commanded beam tilt.

This module deliberately audits an *upper bound* on the information available
from the paper-inspired experiment.  It retains the existing, richer
translated C-aperture Poisson scan, but removes every calibrated deflector and
sample-angle row.  The remaining sample measurements are quantities that can
be extracted from registered intensity images: two spatial widths and their
correlation.  Relative rotations are added only as differences to a reference
state, as obtainable from a deliberately astigmatic incident beam generated
upstream of every condenser lens being fitted.  If the tag is generated at a
different plane, that plane and its quadrupole/transport map must be explicit
model components.

Consequently, failure of this audit also rules out a design using only centred
aperture throughput, beam-radius wobble curves, and relative rotations at the
same optical states.  Those measurements are no richer than the retained
translated-aperture and two-dimensional spatial-covariance data.

An optional angular-radius block represents a convergence measurement from an
indexed Si diffraction pattern.  Its calibration is explicit.  In the
preferred ratio, disk radius is divided by a Si reciprocal-vector radius, so
camera length and detector pixel scale cancel; the remaining shared scale is
the certified ``lambda/d`` (including high-voltage uncertainty).  A shared
real-space detector-radius scale is always profiled and may receive a separate
finite prior from the registered multiscale specimen/downstream calibration.
The result is a local matched-model Fisher audit, never an instrument-readiness
certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from .condenser_branch_b import BranchBLikelihood


POSITION_OBSERVABLE_INDICES = (0, 2, 5)
ANGULAR_RADIUS_OBSERVABLE_INDICES = (1, 3)


class NoTiltRouteAdmissionError(RuntimeError):
    """Raised when a local no-tilt audit is used as an instrument certificate."""


AngularScaleMode = Literal["none", "unanchored", "si_ratio_prior", "fixed"]


@dataclass(frozen=True)
class NoTiltAuditMatrix:
    """Whitened local design matrix and its exact row/column contracts."""

    matrix: np.ndarray
    row_names: tuple[str, ...]
    column_names: tuple[str, ...]
    angular_scale_mode: AngularScaleMode
    profiles_realspace_scale: bool = True

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
            raise ValueError("matrix must be a finite two-dimensional array.")
        rows = tuple(str(value) for value in self.row_names)
        columns = tuple(str(value) for value in self.column_names)
        if matrix.shape != (len(rows), len(columns)):
            raise ValueError("matrix shape does not match its row/column names.")
        if len(columns) < 13 or len(set(rows)) != len(rows):
            raise ValueError("audit names are incomplete or duplicated.")
        frozen = matrix.copy()
        frozen.setflags(write=False)
        object.__setattr__(self, "matrix", frozen)
        object.__setattr__(self, "row_names", rows)
        object.__setattr__(self, "column_names", columns)
        if self.angular_scale_mode not in (
            "none",
            "unanchored",
            "si_ratio_prior",
            "fixed",
        ):
            raise ValueError("Unknown angular_scale_mode.")


@dataclass(frozen=True)
class NoTiltRankAudit:
    """Profiled target-rank result for one local no-tilt design matrix."""

    target_names: tuple[str, ...]
    target_rank: int
    nuisance_rank: int
    target_singular_values: np.ndarray
    weakest_target_direction: np.ndarray
    scaled_standard_uncertainties: np.ndarray
    structural_pass: bool
    practical_pass: bool
    practical_sigma_limit: float
    scope: str

    def __post_init__(self) -> None:
        names = tuple(str(value) for value in self.target_names)
        singular = np.asarray(self.target_singular_values, dtype=float)
        weakest = np.asarray(self.weakest_target_direction, dtype=float)
        uncertainty = np.asarray(self.scaled_standard_uncertainties, dtype=float)
        if len(names) != 12 or singular.shape != (12,) or weakest.shape != (12,):
            raise ValueError("The no-tilt audit requires exactly twelve targets.")
        if uncertainty.shape != (12,):
            raise ValueError("scaled_standard_uncertainties must have shape (12,).")
        if not np.all(np.isfinite(singular)) or not np.all(np.isfinite(weakest)):
            raise ValueError("Rank diagnostics must be finite.")
        for field_name, values in (
            ("target_singular_values", singular),
            ("weakest_target_direction", weakest),
            ("scaled_standard_uncertainties", uncertainty),
        ):
            frozen = values.copy()
            frozen.setflags(write=False)
            object.__setattr__(self, field_name, frozen)
        object.__setattr__(self, "target_names", names)

    @property
    def minimum_singular_value_ratio(self) -> float:
        if self.target_singular_values[0] <= 0.0:
            return 0.0
        return float(
            self.target_singular_values[-1] / self.target_singular_values[0]
        )

    @property
    def maximum_scaled_standard_uncertainty(self) -> float:
        return float(np.max(self.scaled_standard_uncertainties))

    @property
    def null_family(self) -> dict[str, float]:
        return dict(zip(self.target_names, self.weakest_target_direction, strict=True))


def _require_nominal_branch_b(likelihood: BranchBLikelihood) -> None:
    if not isinstance(likelihood, BranchBLikelihood):
        raise TypeError("likelihood must be a BranchBLikelihood.")
    experiment = likelihood.experiment
    if experiment.tomography_controls.shape != (32, 7):
        raise ValueError("The no-tilt audit requires the frozen 32-state design.")
    if len(likelihood.parameter_names) != 56:
        raise ValueError("Unexpected Branch-B parameter layout.")
    expected = (
        "CL1.z_center_m",
        "CL1.log_length_m",
        "CL1.q_bias",
        "CL1.q_slope_per_dac",
        "CL3.z_center_m",
        "CL3.log_length_m",
        "CL3.q_bias",
        "CL3.q_slope_per_dac",
        "C-mini.z_center_m",
        "C-mini.log_length_m",
        "C-mini.q_bias",
        "C-mini.q_slope_per_dac",
    )
    if experiment.parameter_layout.target_names != expected:
        raise ValueError("The twelve-target condenser ordering changed.")


def build_no_tilt_audit_matrix(
    likelihood: BranchBLikelihood,
    *,
    angular_scale_mode: AngularScaleMode = "none",
    position_summary_sigma: float = 0.002,
    rotation_sigma_rad: float = 0.02,
    angular_radius_sigma: float = 0.002,
    si_log_angular_scale_prior_sigma: float = 0.002,
    realspace_log_scale_prior_sigma: float | None = None,
    profile_realspace_scale: bool = True,
) -> NoTiltAuditMatrix:
    """Build the whitened local matrix for the paper-informed no-tilt route.

    The C-aperture rows are raw Poisson rows and retain the per-state dose,
    aperture-radius, stage-scale, entrance-covariance, source, and
    objective-prefield nuisances already present in ``BranchBLikelihood``.
    Deflector rows are excluded.  A further unanchored real-space detector
    radius scale is appended and profiled.

    Relative-rotation rows use raw lens commands and the exact hard-edge
    identity ``rotation = -sum(q)``.  The primary experimental tag is a
    deliberately astigmatic incident beam placed upstream of CL1.  Its command
    strength need not be calibrated: the shared initial ellipse angle cancels
    against state zero and its ellipticity/orientation remain in the fitted
    entrance covariance.  A downstream tag is invalid unless its generation
    plane and transport map are included.  A true defocused Ronchigram carrying the
    incident angle-to-sample map is an alternative; an ordinary downstream
    specimen image is not such a measurement.  Rotation rows here represent
    separate tagged exposures with their own uncertainty.  If orientation and
    spatial covariance are extracted from the same pixels, callers must use a
    joint bootstrap covariance; treating them as independent, as this design
    audit does, is only an optimistic upper bound.

    Unless ``angular_scale_mode='none'``, each state contributes
    ``0.5*(log sigma_theta_x + log sigma_theta_y)``.  A shared calibration
    nuisance with the declared finite Si/voltage prior is added explicitly.
    """

    _require_nominal_branch_b(likelihood)
    if angular_scale_mode not in (
        "none",
        "unanchored",
        "si_ratio_prior",
        "fixed",
    ):
        raise ValueError("Unknown angular_scale_mode.")
    positive = (
        position_summary_sigma,
        rotation_sigma_rad,
        angular_radius_sigma,
        si_log_angular_scale_prior_sigma,
    )
    if not np.all(np.isfinite(positive)) or min(positive) <= 0.0:
        raise ValueError("All measurement/prior scales must be finite and positive.")

    experiment = likelihood.experiment
    parameters = jnp.asarray(likelihood.nominal_parameters)
    full_jacobian = np.asarray(likelihood.jacobian(parameters), dtype=float)
    tomography_jacobian = np.asarray(
        jax.jacfwd(
            lambda values: likelihood.predict(values).tomography_observables
        )(parameters),
        dtype=float,
    )

    edge_slice = likelihood.residual_slices["edge_poisson"]
    prior_slice = likelihood.residual_slices["shared_priors"]
    edge = full_jacobian[edge_slice]
    priors = full_jacobian[prior_slice]

    position = tomography_jacobian[:, POSITION_OBSERVABLE_INDICES, :].reshape(
        -1, full_jacobian.shape[1]
    ) / float(position_summary_sigma)
    base = np.concatenate((edge, position, priors))

    row_names = (
        tuple(likelihood.residual_names[edge_slice])
        + tuple(
            f"sample_spatial.state_{state:02d}.{name}"
            for state in range(32)
            for name in ("log_rms_x", "log_rms_y", "correlation_x_y")
        )
        + tuple(likelihood.residual_names[prior_slice])
    )
    column_names = tuple(likelihood.parameter_names)

    if profile_realspace_scale:
        detector_scale = np.zeros(base.shape[0])
        detector_scale[edge.shape[0] : edge.shape[0] + position.shape[0]] = np.tile(
            np.asarray((1.0, 1.0, 0.0)) / float(position_summary_sigma), 32
        )
        base = np.column_stack((base, detector_scale))
        column_names += ("nuisance.shared_log_realspace_detector_radius_scale",)

    if realspace_log_scale_prior_sigma is not None:
        if not profile_realspace_scale:
            raise ValueError(
                "A real-space scale prior requires profile_realspace_scale=True."
            )
        realspace_sigma = float(realspace_log_scale_prior_sigma)
        if not np.isfinite(realspace_sigma) or realspace_sigma <= 0.0:
            raise ValueError(
                "realspace_log_scale_prior_sigma must be finite and positive."
            )
        scale_prior = np.zeros((1, base.shape[1]), dtype=float)
        scale_prior[0, -1] = 1.0 / realspace_sigma
        base = np.concatenate((base, scale_prior), axis=0)
        row_names += ("prior.registered_sample_log_realspace_scale",)

    rotation = np.zeros((31, base.shape[1]), dtype=float)
    controls = np.asarray(experiment.tomography_controls)
    scales = np.asarray(experiment.target_parameter_scales)
    slope_target_indices = (3, 7, 11)
    for state in range(1, 32):
        for lens_index, target_index in enumerate(slope_target_indices):
            rotation[state - 1, target_index] = -(
                controls[state, lens_index] - controls[0, lens_index]
            ) * scales[target_index] / float(rotation_sigma_rad)
    base = np.concatenate((base, rotation), axis=0)
    row_names += tuple(
        f"astigmatic_illumination.relative_orientation.state_{state:02d}_minus_00"
        for state in range(1, 32)
    )

    if angular_scale_mode != "none":
        has_scale_nuisance = angular_scale_mode in (
            "unanchored",
            "si_ratio_prior",
        )
        if has_scale_nuisance:
            base = np.column_stack((base, np.zeros(base.shape[0])))
            column_names += ("nuisance.indexed_si_log_angular_scale",)
        angular = np.zeros((32, base.shape[1]), dtype=float)
        angular[:, :56] = 0.5 * (
            tomography_jacobian[:, ANGULAR_RADIUS_OBSERVABLE_INDICES[0], :]
            + tomography_jacobian[:, ANGULAR_RADIUS_OBSERVABLE_INDICES[1], :]
        ) / float(angular_radius_sigma)
        if has_scale_nuisance:
            angular[:, -1] = 1.0 / float(angular_radius_sigma)
        base = np.concatenate((base, angular), axis=0)
        row_names += tuple(
            f"indexed_si_diffraction.angular_radius.state_{state:02d}"
            for state in range(32)
        )
        if angular_scale_mode == "si_ratio_prior":
            calibration_prior = np.zeros((1, base.shape[1]), dtype=float)
            calibration_prior[0, -1] = 1.0 / float(
                si_log_angular_scale_prior_sigma
            )
            base = np.concatenate((base, calibration_prior), axis=0)
            row_names += ("prior.indexed_si_lambda_over_d_scale",)

    return NoTiltAuditMatrix(
        base,
        row_names,
        column_names,
        angular_scale_mode,
        bool(profile_realspace_scale),
    )


def profile_no_tilt_targets(
    audit_matrix: NoTiltAuditMatrix,
    *,
    target_names: tuple[str, ...],
    nuisance_relative_rtol: float = 1.0e-10,
    target_relative_rtol: float = 1.0e-7,
    practical_sigma_limit: float = 0.01,
) -> NoTiltRankAudit:
    """Profile every nuisance column and diagnose the twelve target columns."""

    if len(target_names) != 12:
        raise ValueError("target_names must contain twelve values.")
    if not 0.0 < nuisance_relative_rtol < 1.0:
        raise ValueError("nuisance_relative_rtol must lie in (0, 1).")
    if not 0.0 < target_relative_rtol < 1.0:
        raise ValueError("target_relative_rtol must lie in (0, 1).")
    if not np.isfinite(practical_sigma_limit) or practical_sigma_limit <= 0.0:
        raise ValueError("practical_sigma_limit must be finite and positive.")

    matrix = audit_matrix.matrix
    target = matrix[:, :12]
    nuisance = matrix[:, 12:]
    norms = np.linalg.norm(nuisance, axis=0)
    active = norms > np.finfo(float).tiny
    normalized = nuisance[:, active] / norms[active][None, :]
    if normalized.shape[1]:
        left, nuisance_singular, _ = np.linalg.svd(normalized, full_matrices=False)
        nuisance_rank = int(
            np.sum(
                nuisance_singular
                > nuisance_relative_rtol * nuisance_singular[0]
            )
        )
        projected = target - left[:, :nuisance_rank] @ (
            left[:, :nuisance_rank].T @ target
        )
    else:
        nuisance_rank = 0
        projected = target

    _left, singular, right = np.linalg.svd(projected, full_matrices=False)
    target_rank = int(np.sum(singular > target_relative_rtol * singular[0]))
    if target_rank == 12:
        covariance = (right.T * (1.0 / singular**2)) @ right
        uncertainty = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    else:
        uncertainty = np.full(12, np.inf)
    structural = target_rank == 12
    practical = bool(
        structural
        and np.all(np.isfinite(uncertainty))
        and float(np.max(uncertainty)) <= practical_sigma_limit
    )
    return NoTiltRankAudit(
        target_names=target_names,
        target_rank=target_rank,
        nuisance_rank=nuisance_rank,
        target_singular_values=singular,
        weakest_target_direction=right[-1],
        scaled_standard_uncertainties=uncertainty,
        structural_pass=structural,
        practical_pass=practical,
        practical_sigma_limit=float(practical_sigma_limit),
        scope=(
            "local matched-model optimistic upper-bound audit; normalized "
            "controls; independent summary blocks; not nonlinear recovery or "
            "hardware admission"
        ),
    )


def audit_no_tilt_route(
    likelihood: BranchBLikelihood,
    *,
    angular_scale_mode: AngularScaleMode = "none",
    position_summary_sigma: float = 0.002,
    rotation_sigma_rad: float = 0.02,
    angular_radius_sigma: float = 0.002,
    si_log_angular_scale_prior_sigma: float = 0.002,
    realspace_log_scale_prior_sigma: float | None = None,
    profile_realspace_scale: bool = True,
    practical_sigma_limit: float = 0.01,
) -> NoTiltRankAudit:
    """Run the complete fail-closed audit from a Branch-B optical model."""

    matrix = build_no_tilt_audit_matrix(
        likelihood,
        angular_scale_mode=angular_scale_mode,
        position_summary_sigma=position_summary_sigma,
        rotation_sigma_rad=rotation_sigma_rad,
        angular_radius_sigma=angular_radius_sigma,
        si_log_angular_scale_prior_sigma=si_log_angular_scale_prior_sigma,
        realspace_log_scale_prior_sigma=realspace_log_scale_prior_sigma,
        profile_realspace_scale=profile_realspace_scale,
    )
    return profile_no_tilt_targets(
        matrix,
        target_names=likelihood.experiment.parameter_layout.target_names,
        practical_sigma_limit=practical_sigma_limit,
    )


def require_practical_no_tilt_admission(audit: NoTiltRankAudit) -> None:
    """Fail closed unless both structural rank and the precision gate pass."""

    if not audit.structural_pass:
        raise NoTiltRouteAdmissionError(
            f"No-tilt condenser route has target rank {audit.target_rank}/12; "
            "an absolute angular/wavefront or independent axial anchor is required."
        )
    if not audit.practical_pass:
        raise NoTiltRouteAdmissionError(
            "No-tilt condenser route is formally full rank but fails the declared "
            f"precision gate: max scaled sigma={audit.maximum_scaled_standard_uncertainty:.6g}, "
            f"limit={audit.practical_sigma_limit:.6g}."
        )


__all__ = [
    "ANGULAR_RADIUS_OBSERVABLE_INDICES",
    "AngularScaleMode",
    "NoTiltAuditMatrix",
    "NoTiltRankAudit",
    "NoTiltRouteAdmissionError",
    "POSITION_OBSERVABLE_INDICES",
    "audit_no_tilt_route",
    "build_no_tilt_audit_matrix",
    "profile_no_tilt_targets",
    "require_practical_no_tilt_admission",
]
