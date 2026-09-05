"""Measured raw-DAC continuation for hard-edge Larmor branch selection.

No lens-current readback is required by this module.  Instead, it designs a
connected sequence in the commanded raw DAC coordinate, bounds every possible
``q`` increment using a conservative ``max |dq/dDAC|``, and then checks the
*measured* orientation increments with their covariance before unwrapping.

The distinction between a connected image sequence and approach metadata is
intentional.  In particular, ``FrameRequest.dac_ramp_steps`` in the compact
NEOARM protocol describes how an endpoint is approached; it is not evidence
that an image observable was measured at every intermediate point.  A legacy
endpoint ramp therefore cannot pass :func:`audit_continuation_manifest`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Literal, Mapping, Sequence

import numpy as np


MAX_CONTINUATION_PHASE_STEP_RAD = float(np.pi / 4.0)
CONTINUATION_MANIFEST_SCHEMA = "temgym.neoarm.phase-branch-continuation/v1"

ContinuationObservableKind = Literal[
    "oriented_ellipse_axis",
    "signed_response_vector",
]
ContinuationRoute = Literal[
    "upstream_astigmatic_tag",
    "downstream_calibrated_response",
]
ContinuationFrameRole = Literal[
    "astigmatic_tag",
    "on_axis",
    "position_plus",
    "tilt_plus",
]
ContinuationEvidenceOrigin = Literal[
    "new_connected_scan",
    "legacy_endpoint_ramp_metadata",
]


def _immutable_float_array(
    values: Any,
    name: str,
    *,
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions.")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    result = array.copy()
    result.setflags(write=False)
    return result


def _immutable_bool_array(values: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(values, dtype=bool)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}.")
    result = array.copy()
    result.setflags(write=False)
    return result


def _validated_covariance(
    values: Any,
    name: str,
    size: int,
    *,
    positive_definite: bool = False,
) -> np.ndarray:
    covariance = _immutable_float_array(values, name, shape=(size, size))
    scale = max(float(np.linalg.norm(covariance, ord=2)), np.finfo(float).tiny)
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-13 * scale):
        raise ValueError(f"{name} must be symmetric.")
    eigenvalues = np.linalg.eigvalsh(covariance)
    tolerance = 1e-12 * scale
    if float(eigenvalues[0]) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite.")
    if positive_definite and float(eigenvalues[0]) <= tolerance:
        raise ValueError(f"{name} must be positive definite for GLS fitting.")
    return covariance


@dataclass(frozen=True)
class DACContinuationGrid:
    """A connected one-channel raw-DAC scan with a certified phase-step bound."""

    raw_dac_values: np.ndarray
    anchor_raw_dac_values: tuple[float, ...]
    max_abs_dq_per_dac: float
    target_max_abs_delta_q_rad: float
    predicted_abs_delta_q_bounds_rad: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        values = _immutable_float_array(
            self.raw_dac_values, "raw_dac_values", ndim=1
        )
        if values.size < 2:
            raise ValueError("A continuation grid requires at least two points.")
        differences = np.diff(values)
        increasing = bool(values[-1] > values[0])
        if increasing:
            monotone = np.all(differences > 0.0)
        else:
            monotone = np.all(differences < 0.0)
        if not monotone:
            raise ValueError("raw_dac_values must be strictly monotone.")

        derivative_bound = float(self.max_abs_dq_per_dac)
        target = float(self.target_max_abs_delta_q_rad)
        if not np.isfinite(derivative_bound) or derivative_bound < 0.0:
            raise ValueError("max_abs_dq_per_dac must be finite and non-negative.")
        if (
            not np.isfinite(target)
            or target <= 0.0
            or target > MAX_CONTINUATION_PHASE_STEP_RAD
        ):
            raise ValueError(
                "target_max_abs_delta_q_rad must lie in (0, pi/4]."
            )

        anchors = tuple(float(value) for value in self.anchor_raw_dac_values)
        if not np.all(np.isfinite(anchors)):
            raise ValueError("anchor_raw_dac_values must be finite.")
        if len(set(anchors)) != len(anchors):
            raise ValueError("anchor_raw_dac_values must be unique.")
        lower = float(min(values[0], values[-1]))
        upper = float(max(values[0], values[-1]))
        for anchor in anchors:
            if not lower <= anchor <= upper:
                raise ValueError("Every anchor must lie between the endpoints.")
            if not np.any(values == anchor):
                raise ValueError(
                    f"Requested anchor {anchor!r} is not represented exactly."
                )

        predicted = np.abs(differences) * derivative_bound
        tolerance = 32.0 * np.finfo(float).eps * max(1.0, target)
        if np.any(predicted > target + tolerance):
            raise ValueError(
                "The raw-DAC grid violates its conservative |delta q| bound."
            )
        predicted = predicted.copy()
        predicted.setflags(write=False)
        object.__setattr__(self, "raw_dac_values", values)
        object.__setattr__(self, "anchor_raw_dac_values", anchors)
        object.__setattr__(self, "max_abs_dq_per_dac", derivative_bound)
        object.__setattr__(self, "target_max_abs_delta_q_rad", target)
        object.__setattr__(self, "predicted_abs_delta_q_bounds_rad", predicted)

    @property
    def start_raw_dac(self) -> float:
        return float(self.raw_dac_values[0])

    @property
    def stop_raw_dac(self) -> float:
        return float(self.raw_dac_values[-1])

    @property
    def maximum_predicted_abs_delta_q_rad(self) -> float:
        return float(np.max(self.predicted_abs_delta_q_bounds_rad))

    @property
    def anchor_indices(self) -> tuple[int, ...]:
        return tuple(
            int(np.flatnonzero(self.raw_dac_values == anchor)[0])
            for anchor in self.anchor_raw_dac_values
        )


def design_raw_dac_continuation_grid(
    start_raw_dac: float,
    stop_raw_dac: float,
    *,
    max_abs_dq_per_dac: float,
    anchor_raw_dac_values: Sequence[float] = (),
    target_max_abs_delta_q_rad: float = MAX_CONTINUATION_PHASE_STEP_RAD,
) -> DACContinuationGrid:
    """Design the coarsest connected grid satisfying ``|delta q| <= target``.

    The endpoints and every requested anchor are copied into the result
    exactly.  Intermediate values are inserted separately within every
    endpoint/anchor interval, so an anchor is never displaced by rounding.
    ``max_abs_dq_per_dac`` must be an envelope valid over the complete scan,
    not merely the derivative at the nominal point.
    """

    start = float(start_raw_dac)
    stop = float(stop_raw_dac)
    derivative_bound = float(max_abs_dq_per_dac)
    target = float(target_max_abs_delta_q_rad)
    if not np.all(np.isfinite((start, stop))) or start == stop:
        raise ValueError("Raw-DAC endpoints must be finite and distinct.")
    if not np.isfinite(derivative_bound) or derivative_bound < 0.0:
        raise ValueError("max_abs_dq_per_dac must be finite and non-negative.")
    if (
        not np.isfinite(target)
        or target <= 0.0
        or target > MAX_CONTINUATION_PHASE_STEP_RAD
    ):
        raise ValueError("target_max_abs_delta_q_rad must lie in (0, pi/4].")

    requested_anchors = tuple(float(value) for value in anchor_raw_dac_values)
    if not np.all(np.isfinite(requested_anchors)):
        raise ValueError("anchor_raw_dac_values must be finite.")
    if len(set(requested_anchors)) != len(requested_anchors):
        raise ValueError("anchor_raw_dac_values must be unique.")
    lower, upper = sorted((start, stop))
    if any(not lower <= value <= upper for value in requested_anchors):
        raise ValueError("Every anchor must lie between the endpoints.")

    interior = tuple(
        value for value in requested_anchors if value != start and value != stop
    )
    interior = tuple(sorted(interior, reverse=stop < start))
    waypoints = (start, *interior, stop)
    pieces: list[np.ndarray] = []
    for interval_index, (left, right) in enumerate(
        zip(waypoints[:-1], waypoints[1:], strict=True)
    ):
        phase_span_bound = abs(right - left) * derivative_bound
        subdivisions = max(1, int(math.ceil(phase_span_bound / target)))
        points = np.linspace(left, right, subdivisions + 1, dtype=float)
        # np.linspace promises its endpoints, but assign explicitly because
        # exact anchor identity is part of the serialized acquisition contract.
        points[0] = left
        points[-1] = right
        if interval_index:
            points = points[1:]
        pieces.append(points)
    values = np.concatenate(pieces)
    return DACContinuationGrid(
        raw_dac_values=values,
        anchor_raw_dac_values=requested_anchors,
        max_abs_dq_per_dac=derivative_bound,
        target_max_abs_delta_q_rad=target,
    )


@dataclass(frozen=True)
class PhaseContinuationResult:
    """Unwrapped orientation sequence and both continuation safety gates."""

    observable_kind: ContinuationObservableKind
    raw_dac_values: np.ndarray
    principal_angles_rad: np.ndarray
    unwrapped_angles_rad: np.ndarray
    angle_covariance_rad2: np.ndarray
    wrapped_interval_deltas_rad: np.ndarray
    interval_standard_uncertainties_rad: np.ndarray
    uncertainty_gate_values_rad: np.ndarray
    interval_gate_passed: np.ndarray
    sigma_multiplier: float
    maximum_allowed_interval_rad: float
    design_maximum_abs_delta_q_rad: float
    passed: bool
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.observable_kind not in (
            "oriented_ellipse_axis",
            "signed_response_vector",
        ):
            raise ValueError("Unknown continuation observable kind.")
        values = _immutable_float_array(
            self.raw_dac_values, "raw_dac_values", ndim=1
        )
        size = values.size
        if size < 2:
            raise ValueError("A continuation result requires at least two points.")
        arrays = {
            "principal_angles_rad": _immutable_float_array(
                self.principal_angles_rad, "principal_angles_rad", shape=(size,)
            ),
            "unwrapped_angles_rad": _immutable_float_array(
                self.unwrapped_angles_rad, "unwrapped_angles_rad", shape=(size,)
            ),
            "angle_covariance_rad2": _validated_covariance(
                self.angle_covariance_rad2, "angle_covariance_rad2", size
            ),
            "wrapped_interval_deltas_rad": _immutable_float_array(
                self.wrapped_interval_deltas_rad,
                "wrapped_interval_deltas_rad",
                shape=(size - 1,),
            ),
            "interval_standard_uncertainties_rad": _immutable_float_array(
                self.interval_standard_uncertainties_rad,
                "interval_standard_uncertainties_rad",
                shape=(size - 1,),
            ),
            "uncertainty_gate_values_rad": _immutable_float_array(
                self.uncertainty_gate_values_rad,
                "uncertainty_gate_values_rad",
                shape=(size - 1,),
            ),
            "interval_gate_passed": _immutable_bool_array(
                self.interval_gate_passed,
                "interval_gate_passed",
                (size - 1,),
            ),
        }
        sigma_multiplier = float(self.sigma_multiplier)
        maximum = float(self.maximum_allowed_interval_rad)
        design_maximum = float(self.design_maximum_abs_delta_q_rad)
        if not np.isfinite(sigma_multiplier) or sigma_multiplier <= 0.0:
            raise ValueError("sigma_multiplier must be finite and positive.")
        if not np.isfinite(maximum) or not 0.0 < maximum <= np.pi / 4.0:
            raise ValueError("maximum_allowed_interval_rad must lie in (0, pi/4].")
        if not np.isfinite(design_maximum) or design_maximum < 0.0:
            raise ValueError("design_maximum_abs_delta_q_rad must be non-negative.")
        reasons = tuple(str(reason) for reason in self.failure_reasons)
        if bool(self.passed) != bool(np.all(arrays["interval_gate_passed"])):
            raise ValueError("passed is inconsistent with interval_gate_passed.")
        if bool(self.passed) == bool(reasons):
            raise ValueError("failure_reasons must be empty exactly when passed.")
        object.__setattr__(self, "raw_dac_values", values)
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "sigma_multiplier", sigma_multiplier)
        object.__setattr__(self, "maximum_allowed_interval_rad", maximum)
        object.__setattr__(
            self, "design_maximum_abs_delta_q_rad", design_maximum
        )
        object.__setattr__(self, "failure_reasons", reasons)

    @property
    def failed_interval_indices(self) -> tuple[int, ...]:
        """Indices ``i`` for failed intervals from point ``i`` to ``i+1``."""

        return tuple(int(value) for value in np.flatnonzero(~self.interval_gate_passed))

    @property
    def periodicity_rad(self) -> float:
        return float(
            np.pi if self.observable_kind == "oriented_ellipse_axis" else 2.0 * np.pi
        )


class PhaseContinuationError(RuntimeError):
    """Fail-closed branch error carrying the complete interval diagnostics."""

    def __init__(self, diagnostics: PhaseContinuationResult):
        self.diagnostics = diagnostics
        failed = ", ".join(str(value) for value in diagnostics.failed_interval_indices)
        super().__init__(
            "Phase continuation failed the conservative interval gate at "
            f"interval(s) {failed}. No branch-unwrapped values are admitted."
        )


def _phase_continuation_result(
    grid: DACContinuationGrid,
    principal_angles_rad: np.ndarray,
    angle_covariance_rad2: np.ndarray,
    *,
    observable_kind: ContinuationObservableKind,
    periodicity_rad: float,
    sigma_multiplier: float,
    fail_closed: bool,
) -> PhaseContinuationResult:
    angles = _immutable_float_array(
        principal_angles_rad, "principal_angles_rad", ndim=1
    )
    size = grid.raw_dac_values.size
    if angles.shape != (size,):
        raise ValueError(
            "The angle sequence must contain one value per raw-DAC grid point."
        )
    covariance = _validated_covariance(
        angle_covariance_rad2, "angle_covariance_rad2", size
    )
    sigma_factor = float(sigma_multiplier)
    if not np.isfinite(sigma_factor) or sigma_factor <= 0.0:
        raise ValueError("sigma_multiplier must be finite and positive.")

    raw_differences = np.diff(angles)
    wrapped = np.angle(np.exp(1j * (2.0 * np.pi / periodicity_rad) * raw_differences))
    wrapped = wrapped * periodicity_rad / (2.0 * np.pi)
    unwrapped = np.empty_like(angles)
    unwrapped[0] = angles[0]
    unwrapped[1:] = angles[0] + np.cumsum(wrapped)

    interval_variances = (
        np.diag(covariance)[:-1]
        + np.diag(covariance)[1:]
        - 2.0 * covariance[np.arange(size - 1), np.arange(1, size)]
    )
    covariance_scale = max(
        float(np.linalg.norm(covariance, ord=2)), np.finfo(float).tiny
    )
    if np.any(interval_variances < -1e-12 * covariance_scale):
        raise ValueError("The covariance gives a negative interval variance.")
    interval_sigma = np.sqrt(np.maximum(interval_variances, 0.0))
    gate_values = np.abs(wrapped) + sigma_factor * interval_sigma
    gate_passed = gate_values <= MAX_CONTINUATION_PHASE_STEP_RAD
    reasons: tuple[str, ...]
    if np.all(gate_passed):
        reasons = ()
    else:
        failed = tuple(int(value) for value in np.flatnonzero(~gate_passed))
        reasons = (f"interval_uncertainty_gate_failed:{failed}",)
    result = PhaseContinuationResult(
        observable_kind=observable_kind,
        raw_dac_values=grid.raw_dac_values,
        principal_angles_rad=angles,
        unwrapped_angles_rad=unwrapped,
        angle_covariance_rad2=covariance,
        wrapped_interval_deltas_rad=wrapped,
        interval_standard_uncertainties_rad=interval_sigma,
        uncertainty_gate_values_rad=gate_values,
        interval_gate_passed=gate_passed,
        sigma_multiplier=sigma_factor,
        maximum_allowed_interval_rad=MAX_CONTINUATION_PHASE_STEP_RAD,
        design_maximum_abs_delta_q_rad=grid.maximum_predicted_abs_delta_q_rad,
        passed=bool(np.all(gate_passed)),
        failure_reasons=reasons,
    )
    if fail_closed and not result.passed:
        raise PhaseContinuationError(result)
    return result


def unwrap_oriented_ellipse_angles(
    grid: DACContinuationGrid,
    measured_axis_angles_rad: Sequence[float] | np.ndarray,
    angle_covariance_rad2: np.ndarray,
    *,
    sigma_multiplier: float = 5.0,
    fail_closed: bool = True,
) -> PhaseContinuationResult:
    """Unwrap a pi-periodic ellipse axis by applying a doubled-angle map.

    The returned intercept is deliberately not forced to zero: it contains the
    common, unknown orientation of the incident astigmatic tag.  A result is
    admitted only when every measured interval satisfies
    ``|wrapped delta psi| + sigma_multiplier*sigma_delta <= pi/4``.
    """

    measured = _immutable_float_array(
        measured_axis_angles_rad, "measured_axis_angles_rad", ndim=1
    )
    # Canonicalize the unoriented axis with exp(2j*psi); branch continuation is
    # subsequently performed with period pi.
    principal = 0.5 * np.angle(np.exp(2.0j * measured))
    return _phase_continuation_result(
        grid,
        principal,
        angle_covariance_rad2,
        observable_kind="oriented_ellipse_axis",
        periodicity_rad=float(np.pi),
        sigma_multiplier=sigma_multiplier,
        fail_closed=fail_closed,
    )


def _expanded_response_vector_covariance(
    response_vector_covariance_xy: np.ndarray,
    size: int,
) -> np.ndarray:
    covariance_input = np.asarray(response_vector_covariance_xy, dtype=float)
    if covariance_input.shape == (size, 2, 2):
        expanded = np.zeros((2 * size, 2 * size), dtype=float)
        for index in range(size):
            expanded[2 * index : 2 * index + 2, 2 * index : 2 * index + 2] = (
                covariance_input[index]
            )
        covariance_input = expanded
    return _validated_covariance(
        covariance_input,
        "response_vector_covariance_xy",
        2 * size,
    )


def _response_angle_covariance(
    response_vectors_xy: np.ndarray,
    response_vector_covariance_xy: np.ndarray,
) -> np.ndarray:
    size = response_vectors_xy.shape[0]
    covariance = _expanded_response_vector_covariance(
        response_vector_covariance_xy, size
    )
    jacobian = np.zeros((size, 2 * size), dtype=float)
    for index, (x_value, y_value) in enumerate(response_vectors_xy):
        radius_squared = x_value * x_value + y_value * y_value
        jacobian[index, 2 * index] = -y_value / radius_squared
        jacobian[index, 2 * index + 1] = x_value / radius_squared
    return jacobian @ covariance @ jacobian.T


def unwrap_signed_response_vector_rotations(
    grid: DACContinuationGrid,
    response_vectors_xy: np.ndarray,
    *,
    response_vector_covariance_xy: np.ndarray | None = None,
    angle_covariance_rad2: np.ndarray | None = None,
    minimum_response_norm: float = 0.0,
    sigma_multiplier: float = 5.0,
    fail_closed: bool = True,
) -> PhaseContinuationResult:
    """Unwrap signed, 2pi-periodic rotations of calibrated 2-D responses.

    Exactly one covariance representation is required.  Cartesian covariance
    may have shape ``(2*n, 2*n)`` (including cross-frame correlations) or
    ``(n, 2, 2)`` (independent frame blocks).  It is propagated to angle space
    with the analytic ``atan2`` Jacobian.  The Cartesian covariance must
    already include shared on-axis subtraction, detector calibration, and
    calibrated-input uncertainty; storing sample-state covariance in an
    acquisition manifest does not propagate it automatically.

    When Cartesian covariance is supplied, every marginal confidence region
    must remain strictly outside ``minimum_response_norm``.  The angle-only
    route cannot perform that Cartesian identifiability check, so its caller is
    responsible for certifying a non-zero response before use.
    """

    vectors = _immutable_float_array(
        response_vectors_xy, "response_vectors_xy", ndim=2
    )
    size = grid.raw_dac_values.size
    if vectors.shape != (size, 2):
        raise ValueError(f"response_vectors_xy must have shape ({size}, 2).")
    threshold = float(minimum_response_norm)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("minimum_response_norm must be finite and non-negative.")
    sigma_factor = float(sigma_multiplier)
    if not np.isfinite(sigma_factor) or sigma_factor <= 0.0:
        raise ValueError("sigma_multiplier must be finite and positive.")
    norms = np.linalg.norm(vectors, axis=1)
    numerical_floor = np.sqrt(np.finfo(float).tiny)
    if np.any(norms <= max(threshold, numerical_floor)):
        raise ValueError(
            "Every signed response vector must exceed minimum_response_norm; "
            "the rotation of a zero response is undefined."
        )
    if (response_vector_covariance_xy is None) == (angle_covariance_rad2 is None):
        raise ValueError(
            "Provide exactly one of response_vector_covariance_xy or "
            "angle_covariance_rad2."
        )
    if angle_covariance_rad2 is None:
        cartesian_covariance = _expanded_response_vector_covariance(
            np.asarray(response_vector_covariance_xy, dtype=float), size
        )
        # atan2 linearization is not admissible when the Cartesian confidence
        # region can reach a zero/too-small response.  The largest eigenvalue
        # of each marginal 2-D covariance gives a conservative Euclidean ball
        # even when the complete covariance contains cross-frame correlation.
        marginal_radius = np.empty(size, dtype=float)
        for index in range(size):
            block = cartesian_covariance[
                2 * index : 2 * index + 2,
                2 * index : 2 * index + 2,
            ]
            maximum_variance = max(float(np.linalg.eigvalsh(block)[-1]), 0.0)
            marginal_radius[index] = sigma_factor * math.sqrt(maximum_variance)
        if np.any(norms - marginal_radius <= threshold):
            failed = tuple(
                int(value)
                for value in np.flatnonzero(norms - marginal_radius <= threshold)
            )
            raise ValueError(
                "The Cartesian response confidence region reaches "
                "minimum_response_norm at point(s) "
                f"{failed}; signed rotation is not identifiable."
            )
        covariance = _response_angle_covariance(vectors, cartesian_covariance)
    else:
        covariance = _validated_covariance(
            angle_covariance_rad2, "angle_covariance_rad2", size
        )
    principal = np.arctan2(vectors[:, 1], vectors[:, 0])
    return _phase_continuation_result(
        grid,
        principal,
        covariance,
        observable_kind="signed_response_vector",
        periodicity_rad=float(2.0 * np.pi),
        sigma_multiplier=sigma_factor,
        fail_closed=fail_closed,
    )


@dataclass(frozen=True)
class RelativeQSlopeFit:
    """GLS fit of relative hard-edge strength with a free orientation intercept."""

    reference_raw_dac: float
    unknown_orientation_intercept_rad: float
    q_slope_per_dac: float
    covariance_intercept_q_slope: np.ndarray
    fitted_angles_rad: np.ndarray
    residual_angles_rad: np.ndarray
    chi_square: float
    degrees_of_freedom: int
    observed_rotation_per_q_sign: float
    intercept_periodicity_rad: float
    absolute_q_reference_identified: bool = False

    def __post_init__(self) -> None:
        scalars = (
            self.reference_raw_dac,
            self.unknown_orientation_intercept_rad,
            self.q_slope_per_dac,
            self.chi_square,
            self.observed_rotation_per_q_sign,
            self.intercept_periodicity_rad,
        )
        if not np.all(np.isfinite(scalars)) or self.chi_square < 0.0:
            raise ValueError("Relative-q fit scalars must be finite and physical.")
        covariance = _validated_covariance(
            self.covariance_intercept_q_slope,
            "covariance_intercept_q_slope",
            2,
            positive_definite=True,
        )
        fitted = _immutable_float_array(
            self.fitted_angles_rad, "fitted_angles_rad", ndim=1
        )
        residual = _immutable_float_array(
            self.residual_angles_rad,
            "residual_angles_rad",
            shape=fitted.shape,
        )
        if self.degrees_of_freedom != fitted.size - 2:
            raise ValueError("degrees_of_freedom must equal n_points - 2.")
        if self.observed_rotation_per_q_sign not in (-1.0, 1.0):
            raise ValueError("observed_rotation_per_q_sign must be -1 or +1.")
        if self.absolute_q_reference_identified:
            raise ValueError(
                "A relative rotation fit cannot identify the absolute q reference."
            )
        object.__setattr__(self, "covariance_intercept_q_slope", covariance)
        object.__setattr__(self, "fitted_angles_rad", fitted)
        object.__setattr__(self, "residual_angles_rad", residual)

    @property
    def q_slope_standard_uncertainty(self) -> float:
        return float(np.sqrt(self.covariance_intercept_q_slope[1, 1]))

    @property
    def intercept_standard_uncertainty_rad(self) -> float:
        return float(np.sqrt(self.covariance_intercept_q_slope[0, 0]))


def fit_relative_q_slope(
    continuation: PhaseContinuationResult,
    *,
    reference_raw_dac: float | None = None,
    observed_rotation_per_q_sign: Literal[-1, 1] = -1,
) -> RelativeQSlopeFit:
    """Fit ``psi = psi0 + sign*q_slope*(DAC-reference)`` by GLS.

    For the hard-edge convention used by TemGymCore, transverse Larmor
    rotation is ``-q``, so ``observed_rotation_per_q_sign=-1`` is the default.
    The fitted intercept remains a shared unknown tag/detector orientation;
    this relative experiment does not invent an absolute ``q_ref``.
    """

    if not isinstance(continuation, PhaseContinuationResult):
        raise TypeError("continuation must be a PhaseContinuationResult.")
    if not continuation.passed:
        raise PhaseContinuationError(continuation)
    sign = float(observed_rotation_per_q_sign)
    if sign not in (-1.0, 1.0):
        raise ValueError("observed_rotation_per_q_sign must be -1 or +1.")
    dacs = continuation.raw_dac_values
    reference = (
        float(np.mean(dacs))
        if reference_raw_dac is None
        else float(reference_raw_dac)
    )
    if not np.isfinite(reference):
        raise ValueError("reference_raw_dac must be finite.")
    covariance = _validated_covariance(
        continuation.angle_covariance_rad2,
        "angle_covariance_rad2",
        dacs.size,
        positive_definite=True,
    )
    design = np.column_stack((np.ones(dacs.size), sign * (dacs - reference)))
    cholesky = np.linalg.cholesky(covariance)
    whitened_design = np.linalg.solve(cholesky, design)
    whitened_angles = np.linalg.solve(
        cholesky, continuation.unwrapped_angles_rad
    )
    if np.linalg.matrix_rank(whitened_design) != 2:
        raise ValueError("The continuation grid cannot identify intercept and slope.")
    normal = whitened_design.T @ whitened_design
    parameter_covariance = np.linalg.inv(normal)
    parameters = parameter_covariance @ (
        whitened_design.T @ whitened_angles
    )
    fitted = design @ parameters
    residual = continuation.unwrapped_angles_rad - fitted
    whitened_residual = np.linalg.solve(cholesky, residual)
    return RelativeQSlopeFit(
        reference_raw_dac=reference,
        unknown_orientation_intercept_rad=float(parameters[0]),
        q_slope_per_dac=float(parameters[1]),
        covariance_intercept_q_slope=parameter_covariance,
        fitted_angles_rad=fitted,
        residual_angles_rad=residual,
        chi_square=float(whitened_residual @ whitened_residual),
        degrees_of_freedom=int(dacs.size - 2),
        observed_rotation_per_q_sign=sign,
        intercept_periodicity_rad=continuation.periodicity_rad,
        absolute_q_reference_identified=False,
    )


@dataclass(frozen=True)
class QReferenceTransform:
    """A linear ``q`` law expressed as legacy bias/slope at a new DAC origin."""

    source_reference_raw_dac: float
    target_reference_raw_dac: float
    q_bias: float
    q_slope_per_dac: float
    covariance_q_bias_q_slope: np.ndarray
    jacobian: np.ndarray

    def __post_init__(self) -> None:
        values = (
            self.source_reference_raw_dac,
            self.target_reference_raw_dac,
            self.q_bias,
            self.q_slope_per_dac,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("q-reference transform values must be finite.")
        covariance = _validated_covariance(
            self.covariance_q_bias_q_slope,
            "covariance_q_bias_q_slope",
            2,
        )
        jacobian = _immutable_float_array(
            self.jacobian, "jacobian", shape=(2, 2)
        )
        object.__setattr__(self, "covariance_q_bias_q_slope", covariance)
        object.__setattr__(self, "jacobian", jacobian)


def transform_q_reference_covariance(
    q_at_source_reference: float,
    q_slope_per_dac: float,
    covariance_q_reference_q_slope: np.ndarray,
    *,
    source_reference_raw_dac: float,
    target_reference_raw_dac: float,
) -> QReferenceTransform:
    """Move ``(q_ref, slope)`` and its covariance to a legacy DAC reference."""

    q_reference = float(q_at_source_reference)
    slope = float(q_slope_per_dac)
    source = float(source_reference_raw_dac)
    target = float(target_reference_raw_dac)
    if not np.all(np.isfinite((q_reference, slope, source, target))):
        raise ValueError("q law and DAC references must be finite.")
    covariance = _validated_covariance(
        covariance_q_reference_q_slope,
        "covariance_q_reference_q_slope",
        2,
    )
    displacement = target - source
    jacobian = np.asarray(((1.0, displacement), (0.0, 1.0)), dtype=float)
    transformed = jacobian @ covariance @ jacobian.T
    return QReferenceTransform(
        source_reference_raw_dac=source,
        target_reference_raw_dac=target,
        q_bias=q_reference + slope * displacement,
        q_slope_per_dac=slope,
        covariance_q_bias_q_slope=transformed,
        jacobian=jacobian,
    )


HARD_EDGE_SIGN_EXPLANATION = (
    "The exact n*pi alias transform preserves the sign of q whenever the "
    "physical length remains positive. The replacement q -> -q is not a "
    "member of that family: it reverses signed Larmor rotation and must be "
    "tested as a separate polarity hypothesis, or fixed by a signed "
    "response-vector measurement."
)


@dataclass(frozen=True)
class HardEdgePhaseAlias:
    """One exact single-operating-point hard-edge matrix alias."""

    branch_index: int
    q: float
    length_m: float
    is_base_branch: bool

    def __post_init__(self) -> None:
        if int(self.branch_index) != self.branch_index:
            raise ValueError("branch_index must be an integer.")
        if not np.all(np.isfinite((self.q, self.length_m))) or self.length_m <= 0.0:
            raise ValueError("Alias q and length must be finite and length positive.")
        object.__setattr__(self, "branch_index", int(self.branch_index))
        object.__setattr__(self, "q", float(self.q))
        object.__setattr__(self, "length_m", float(self.length_m))
        if bool(self.is_base_branch) != (self.branch_index == 0):
            raise ValueError("is_base_branch must be equivalent to branch_index == 0.")


@dataclass(frozen=True)
class HardEdgeAliasEnumeration:
    """Exact anchor-point phase/length aliases admitted by finite bounds."""

    base_q: float
    base_length_m: float
    q_bounds: tuple[float, float]
    length_bounds_m: tuple[float, float]
    aliases: tuple[HardEdgePhaseAlias, ...]
    q_sign_is_separate: bool = True
    sign_explanation: str = HARD_EDGE_SIGN_EXPLANATION

    def __post_init__(self) -> None:
        aliases = tuple(self.aliases)
        if len({alias.branch_index for alias in aliases}) != len(aliases):
            raise ValueError("Alias branch indices must be unique.")
        if tuple(sorted(aliases, key=lambda value: value.branch_index)) != aliases:
            raise ValueError("Aliases must be sorted by branch index.")
        if not self.q_sign_is_separate or self.sign_explanation != HARD_EDGE_SIGN_EXPLANATION:
            raise ValueError("The separate q-sign interpretation is mandatory.")
        object.__setattr__(self, "aliases", aliases)

    @property
    def branch_indices(self) -> tuple[int, ...]:
        return tuple(alias.branch_index for alias in self.aliases)


def enumerate_hard_edge_phase_aliases(
    base_q: float,
    base_length_m: float,
    *,
    q_bounds: tuple[float, float],
    length_bounds_m: tuple[float, float],
) -> HardEdgeAliasEnumeration:
    """Enumerate every exact anchor-point phase branch inside finite bounds.

    The identity follows directly from the exact hard-edge transfer matrix:
    both its radial block and Larmor rotation acquire ``(-1)**n`` and those
    factors cancel.  Positive physical length automatically restricts this
    enumeration to aliases having the same sign as ``base_q``.

    This is a matrix identity at one operating point, not a fixed-length alias
    of an entire connected ``q(DAC)`` scan.  For non-zero ``n``, the transformed
    length ``length*(q+n*pi)/q`` changes with ``q``.  Enumerated branches must
    therefore be treated as anchor hypotheses and tested against all measured
    continuation points jointly.
    """

    q_value = float(base_q)
    length = float(base_length_m)
    q_limits = tuple(float(value) for value in q_bounds)
    length_limits = tuple(float(value) for value in length_bounds_m)
    if not np.all(np.isfinite((q_value, length, *q_limits, *length_limits))):
        raise ValueError("Alias values and bounds must be finite.")
    if q_value == 0.0:
        raise ValueError("base_q must be non-zero for the exact alias transform.")
    if length <= 0.0:
        raise ValueError("base_length_m must be positive.")
    if len(q_limits) != 2 or not q_limits[0] < q_limits[1]:
        raise ValueError("q_bounds must be a finite increasing pair.")
    if (
        len(length_limits) != 2
        or length_limits[0] <= 0.0
        or not length_limits[0] < length_limits[1]
    ):
        raise ValueError("length_bounds_m must be a positive increasing pair.")

    lower_estimate = int(math.ceil((q_limits[0] - q_value) / np.pi))
    upper_estimate = int(math.floor((q_limits[1] - q_value) / np.pi))
    # The quotient used to infer the integer range can round just across an
    # integer when a closed bound was itself formed as q + n*pi.  Search one
    # neighbouring integer on each side and let the direct closed-bound tests
    # below make the final decision.  This preserves exact inclusive bounds
    # without introducing a scale-dependent numerical tolerance.
    lower_n = lower_estimate - 1
    upper_n = upper_estimate + 1
    if upper_n - lower_n + 1 > 1_000_000:
        raise ValueError(
            "Alias bounds imply a search window of more than one million "
            "phase branches."
        )
    aliases: list[HardEdgePhaseAlias] = []
    for branch_index in range(lower_n, upper_n + 1):
        candidate_q = q_value + branch_index * np.pi
        candidate_length = length * candidate_q / q_value
        if not q_limits[0] <= candidate_q <= q_limits[1]:
            continue
        if not length_limits[0] <= candidate_length <= length_limits[1]:
            continue
        if candidate_length <= 0.0:
            continue
        aliases.append(
            HardEdgePhaseAlias(
                branch_index=branch_index,
                q=candidate_q,
                length_m=candidate_length,
                is_base_branch=branch_index == 0,
            )
        )
    return HardEdgeAliasEnumeration(
        base_q=q_value,
        base_length_m=length,
        q_bounds=q_limits,
        length_bounds_m=length_limits,
        aliases=tuple(aliases),
    )


@dataclass(frozen=True)
class ContinuationFrame:
    """One image-bearing point in a connected phase-continuation scan."""

    frame_id: str
    sequence_index: int
    grid_index: int
    raw_dac_value: float
    route: ContinuationRoute
    role: ContinuationFrameRole
    sample_state_x_theta_x_y_theta_y: tuple[float, float, float, float] | None
    image_record_id: str | None = None

    def __post_init__(self) -> None:
        frame_id = str(self.frame_id).strip()
        if not frame_id:
            raise ValueError("frame_id must be non-empty.")
        if self.sequence_index < 0 or self.grid_index < 0:
            raise ValueError("Frame indices must be non-negative.")
        raw_dac = float(self.raw_dac_value)
        if not np.isfinite(raw_dac):
            raise ValueError("raw_dac_value must be finite.")
        if self.route not in (
            "upstream_astigmatic_tag",
            "downstream_calibrated_response",
        ):
            raise ValueError("Unknown continuation route.")
        allowed_roles = (
            ("astigmatic_tag",)
            if self.route == "upstream_astigmatic_tag"
            else ("on_axis", "position_plus", "tilt_plus")
        )
        if self.role not in allowed_roles:
            raise ValueError(f"Role {self.role!r} is incompatible with {self.route!r}.")
        state = self.sample_state_x_theta_x_y_theta_y
        if self.route == "upstream_astigmatic_tag":
            if state is not None:
                raise ValueError("An upstream astigmatic-tag frame has no sample input state.")
        else:
            if state is None:
                raise ValueError("A downstream response frame requires a sample input state.")
            state = tuple(float(value) for value in state)
            if len(state) != 4 or not np.all(np.isfinite(state)):
                raise ValueError("Sample states must contain four finite values.")
            x_value, theta_x, y_value, theta_y = state
            if self.role == "on_axis" and any(value != 0.0 for value in state):
                raise ValueError("The on-axis sample state must be exactly zero.")
            if self.role == "position_plus" and (
                (x_value == 0.0 and y_value == 0.0)
                or theta_x != 0.0
                or theta_y != 0.0
            ):
                raise ValueError(
                    "position_plus requires non-zero position and zero tilt."
                )
            if self.role == "tilt_plus" and (
                (theta_x == 0.0 and theta_y == 0.0)
                or x_value != 0.0
                or y_value != 0.0
            ):
                raise ValueError("tilt_plus requires non-zero tilt and zero position.")
        record_id = self.image_record_id
        if record_id is not None:
            record_id = str(record_id).strip()
            if not record_id:
                raise ValueError("image_record_id must be non-empty or None.")
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "sequence_index", int(self.sequence_index))
        object.__setattr__(self, "grid_index", int(self.grid_index))
        object.__setattr__(self, "raw_dac_value", raw_dac)
        object.__setattr__(self, "sample_state_x_theta_x_y_theta_y", state)
        object.__setattr__(self, "image_record_id", record_id)


@dataclass(frozen=True)
class ExperimentalContinuationManifest:
    """Immutable schema for one lens/channel connected continuation sequence."""

    acquisition_id: str
    lens_name: str
    raw_dac_channel_name: str
    route: ContinuationRoute
    grid: DACContinuationGrid
    frames: tuple[ContinuationFrame, ...]
    fixed_raw_dac_channels: tuple[tuple[str, float], ...] = ()
    tag_generation_z_m: float | None = None
    varied_lens_entrance_z_m: float | None = None
    sample_state_covariance: np.ndarray | None = None
    evidence_origin: ContinuationEvidenceOrigin = "new_connected_scan"
    schema: str = CONTINUATION_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        for name in ("acquisition_id", "lens_name", "raw_dac_channel_name"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty.")
            object.__setattr__(self, name, value)
        if self.route not in (
            "upstream_astigmatic_tag",
            "downstream_calibrated_response",
        ):
            raise ValueError("Unknown continuation route.")
        if not isinstance(self.grid, DACContinuationGrid):
            raise TypeError("grid must be a DACContinuationGrid.")
        frames = tuple(self.frames)
        if any(not isinstance(frame, ContinuationFrame) for frame in frames):
            raise TypeError("frames must contain ContinuationFrame objects.")
        fixed = tuple((str(name).strip(), float(value)) for name, value in self.fixed_raw_dac_channels)
        if any(not name or not np.isfinite(value) for name, value in fixed):
            raise ValueError("Fixed raw-DAC channels must be named and finite.")
        if len({name for name, _ in fixed}) != len(fixed):
            raise ValueError("Fixed raw-DAC channel names must be unique.")
        if self.raw_dac_channel_name in {name for name, _ in fixed}:
            raise ValueError("The scanned channel cannot also be a fixed channel.")
        if self.evidence_origin not in (
            "new_connected_scan",
            "legacy_endpoint_ramp_metadata",
        ):
            raise ValueError("Unknown continuation evidence origin.")

        covariance: np.ndarray | None
        if self.route == "upstream_astigmatic_tag":
            if self.tag_generation_z_m is None or self.varied_lens_entrance_z_m is None:
                raise ValueError(
                    "An upstream tag requires its generation z and lens entrance z."
                )
            tag_z = float(self.tag_generation_z_m)
            entrance_z = float(self.varied_lens_entrance_z_m)
            if not np.all(np.isfinite((tag_z, entrance_z))) or not tag_z < entrance_z:
                raise ValueError(
                    "The astigmatic tag must be generated upstream of the varied lens."
                )
            if self.sample_state_covariance is not None:
                raise ValueError("The upstream tag route has no sample-state covariance.")
            covariance = None
            object.__setattr__(self, "tag_generation_z_m", tag_z)
            object.__setattr__(self, "varied_lens_entrance_z_m", entrance_z)
        else:
            if self.tag_generation_z_m is not None or self.varied_lens_entrance_z_m is not None:
                raise ValueError("A downstream response route must not declare a tag plane.")
            if self.sample_state_covariance is None:
                raise ValueError(
                    "A downstream response route requires calibrated sample-state covariance."
                )
            covariance = _validated_covariance(
                self.sample_state_covariance, "sample_state_covariance", 4
            )
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "fixed_raw_dac_channels", fixed)
        object.__setattr__(self, "sample_state_covariance", covariance)
        object.__setattr__(self, "schema", str(self.schema))


def _continuation_roles(route: ContinuationRoute) -> tuple[ContinuationFrameRole, ...]:
    if route == "upstream_astigmatic_tag":
        return ("astigmatic_tag",)
    if route == "downstream_calibrated_response":
        return ("on_axis", "position_plus", "tilt_plus")
    raise ValueError("Unknown continuation route.")


def build_experimental_continuation_manifest(
    grid: DACContinuationGrid,
    *,
    acquisition_id: str,
    lens_name: str,
    raw_dac_channel_name: str,
    route: ContinuationRoute,
    fixed_raw_dac_channels: Mapping[str, float] | Sequence[tuple[str, float]] = (),
    tag_generation_z_m: float | None = None,
    varied_lens_entrance_z_m: float | None = None,
    position_plus_state_x_theta_x_y_theta_y: tuple[float, float, float, float] | None = None,
    tilt_plus_state_x_theta_x_y_theta_y: tuple[float, float, float, float] | None = None,
    sample_state_covariance: np.ndarray | None = None,
    recorded_image_ids: Sequence[str | None] | None = None,
    evidence_origin: ContinuationEvidenceOrigin = "new_connected_scan",
) -> ExperimentalContinuationManifest:
    """Build the exact image manifest for an upstream or downstream scan.

    An upstream scan records one astigmatic-tag image at every DAC point.  A
    downstream scan records a triplet at every point: on-axis, calibrated
    positive position, and calibrated positive tilt.  Supplying endpoint
    images alone therefore produces an incomplete audit, even if the hardware
    was internally ramped between them.
    """

    if not isinstance(grid, DACContinuationGrid):
        raise TypeError("grid must be a DACContinuationGrid.")
    roles = _continuation_roles(route)
    if route == "downstream_calibrated_response":
        if position_plus_state_x_theta_x_y_theta_y is None:
            raise ValueError("The downstream route requires position_plus_state.")
        if tilt_plus_state_x_theta_x_y_theta_y is None:
            raise ValueError("The downstream route requires tilt_plus_state.")
    states: dict[ContinuationFrameRole, tuple[float, float, float, float] | None] = {
        "astigmatic_tag": None,
        "on_axis": (0.0, 0.0, 0.0, 0.0),
        "position_plus": position_plus_state_x_theta_x_y_theta_y,
        "tilt_plus": tilt_plus_state_x_theta_x_y_theta_y,
    }
    expected_count = int(grid.raw_dac_values.size * len(roles))
    if recorded_image_ids is None:
        records: tuple[str | None, ...] = (None,) * expected_count
    else:
        records = tuple(recorded_image_ids)
        if len(records) != expected_count:
            raise ValueError(
                f"recorded_image_ids must contain {expected_count} entries."
            )
    if isinstance(fixed_raw_dac_channels, Mapping):
        fixed = tuple(
            sorted(
                (str(name), float(value))
                for name, value in fixed_raw_dac_channels.items()
            )
        )
    else:
        fixed = tuple(fixed_raw_dac_channels)

    frames: list[ContinuationFrame] = []
    sequence_index = 0
    for grid_index, raw_dac in enumerate(grid.raw_dac_values):
        for role in roles:
            frame_id = (
                f"{acquisition_id}:{lens_name}:{grid_index:04d}:{role}"
            )
            frames.append(
                ContinuationFrame(
                    frame_id=frame_id,
                    sequence_index=sequence_index,
                    grid_index=grid_index,
                    raw_dac_value=float(raw_dac),
                    route=route,
                    role=role,
                    sample_state_x_theta_x_y_theta_y=states[role],
                    image_record_id=records[sequence_index],
                )
            )
            sequence_index += 1
    return ExperimentalContinuationManifest(
        acquisition_id=acquisition_id,
        lens_name=lens_name,
        raw_dac_channel_name=raw_dac_channel_name,
        route=route,
        grid=grid,
        frames=tuple(frames),
        fixed_raw_dac_channels=fixed,
        tag_generation_z_m=tag_generation_z_m,
        varied_lens_entrance_z_m=varied_lens_entrance_z_m,
        sample_state_covariance=sample_state_covariance,
        evidence_origin=evidence_origin,
    )


@dataclass(frozen=True)
class ContinuationManifestAudit:
    """Structural and measured-evidence audit for a continuation manifest."""

    expected_frame_count: int
    present_frame_count: int
    recorded_frame_count: int
    expected_grid_point_count: int
    completely_recorded_grid_point_count: int
    schema_passed: bool
    frame_structure_passed: bool
    endpoint_anchor_coverage_passed: bool
    connected_order_passed: bool
    new_measurement_origin_passed: bool
    structural_passed: bool
    measured_continuity_passed: bool
    failure_reasons: tuple[str, ...]
    scope: str

    def __post_init__(self) -> None:
        counts = (
            self.expected_frame_count,
            self.present_frame_count,
            self.recorded_frame_count,
            self.expected_grid_point_count,
            self.completely_recorded_grid_point_count,
        )
        if any(int(value) != value or value < 0 for value in counts):
            raise ValueError("Manifest audit counts must be non-negative integers.")
        if self.structural_passed != bool(
            self.schema_passed
            and self.frame_structure_passed
            and self.endpoint_anchor_coverage_passed
            and self.connected_order_passed
        ):
            raise ValueError("structural_passed is inconsistent with component gates.")
        expected_measured = bool(
            self.structural_passed
            and self.new_measurement_origin_passed
            and self.recorded_frame_count == self.expected_frame_count
            and self.completely_recorded_grid_point_count
            == self.expected_grid_point_count
        )
        if self.measured_continuity_passed != expected_measured:
            raise ValueError(
                "measured_continuity_passed is inconsistent with evidence gates."
            )
        reasons = tuple(str(reason) for reason in self.failure_reasons)
        if self.measured_continuity_passed == bool(reasons):
            raise ValueError(
                "failure_reasons must be empty exactly when measured evidence passes."
            )
        object.__setattr__(self, "failure_reasons", reasons)


def audit_continuation_manifest(
    manifest: ExperimentalContinuationManifest,
) -> ContinuationManifestAudit:
    """Audit exact frame coverage; legacy endpoint ramps always fail evidence."""

    if not isinstance(manifest, ExperimentalContinuationManifest):
        raise TypeError("manifest must be an ExperimentalContinuationManifest.")
    roles = _continuation_roles(manifest.route)
    grid_size = int(manifest.grid.raw_dac_values.size)
    expected_pairs = tuple(
        (grid_index, role)
        for grid_index in range(grid_size)
        for role in roles
    )
    expected_count = len(expected_pairs)
    frames = manifest.frames
    frame_pairs = tuple((frame.grid_index, frame.role) for frame in frames)
    ids_unique = len({frame.frame_id for frame in frames}) == len(frames)
    recorded_ids = tuple(
        frame.image_record_id
        for frame in frames
        if frame.image_record_id is not None
    )
    recorded_ids_unique = len(set(recorded_ids)) == len(recorded_ids)
    sequence_indices = tuple(frame.sequence_index for frame in frames)
    sequence_ok = sequence_indices == tuple(range(len(frames)))
    route_ok = all(frame.route == manifest.route for frame in frames)
    index_and_dac_ok = all(
        0 <= frame.grid_index < grid_size
        and frame.raw_dac_value == manifest.grid.raw_dac_values[frame.grid_index]
        for frame in frames
    )
    # A connected response scan must apply the same calibrated input basis at
    # every DAC point.  Otherwise a changing input state can masquerade as a
    # lens rotation.  This is deliberately audited after construction as well
    # as guaranteed by the builder, so hand-edited/decoded manifests cannot
    # silently weaken the experiment.
    sample_states_consistent = all(
        len(
            {
                frame.sample_state_x_theta_x_y_theta_y
                for frame in frames
                if frame.role == role
            }
        )
        <= 1
        for role in roles
    )
    frame_structure = bool(
        ids_unique
        and recorded_ids_unique
        and route_ok
        and index_and_dac_ok
        and sample_states_consistent
        and frame_pairs == expected_pairs
    )
    connected_order = bool(sequence_ok and frame_pairs == expected_pairs)

    covered_pairs = set(frame_pairs)
    special_indices = {0, grid_size - 1, *manifest.grid.anchor_indices}
    endpoint_anchor_coverage = all(
        (grid_index, role) in covered_pairs
        for grid_index in special_indices
        for role in roles
    )
    recorded_count = sum(frame.image_record_id is not None for frame in frames)
    complete_points = 0
    for grid_index in range(grid_size):
        point_frames = [
            frame for frame in frames if frame.grid_index == grid_index
        ]
        if (
            len(point_frames) == len(roles)
            and {frame.role for frame in point_frames} == set(roles)
            and all(frame.image_record_id is not None for frame in point_frames)
        ):
            complete_points += 1
    schema_passed = manifest.schema == CONTINUATION_MANIFEST_SCHEMA
    new_origin = manifest.evidence_origin == "new_connected_scan"
    structural = bool(
        schema_passed
        and frame_structure
        and endpoint_anchor_coverage
        and connected_order
    )
    measured = bool(
        structural
        and new_origin
        and recorded_count == expected_count
        and complete_points == grid_size
    )
    reasons: list[str] = []
    if not schema_passed:
        reasons.append("schema_mismatch")
    if not frame_structure:
        reasons.append("frame_role_or_grid_coverage")
    if not recorded_ids_unique:
        reasons.append("image_record_id_reused")
    if not sample_states_consistent:
        reasons.append("sample_input_state_changed_within_scan")
    if not endpoint_anchor_coverage:
        reasons.append("endpoint_or_anchor_missing")
    if not connected_order:
        reasons.append("acquisition_order_not_connected")
    if not new_origin:
        reasons.append("legacy_endpoint_ramp_is_not_measured_continuation")
    if recorded_count != expected_count or complete_points != grid_size:
        reasons.append("intermediate_images_not_completely_recorded")
    return ContinuationManifestAudit(
        expected_frame_count=expected_count,
        present_frame_count=len(frames),
        recorded_frame_count=recorded_count,
        expected_grid_point_count=grid_size,
        completely_recorded_grid_point_count=complete_points,
        schema_passed=schema_passed,
        frame_structure_passed=frame_structure,
        endpoint_anchor_coverage_passed=endpoint_anchor_coverage,
        connected_order_passed=connected_order,
        new_measurement_origin_passed=new_origin,
        structural_passed=structural,
        measured_continuity_passed=measured,
        failure_reasons=tuple(reasons),
        scope=(
            "Measured branch-continuation evidence for one raw-DAC lens channel; "
            "not an absolute field-current calibration."
        ),
    )


def continuation_manifest_to_payload(
    manifest: ExperimentalContinuationManifest,
) -> dict[str, Any]:
    """Return a JSON-compatible payload without weakening the audit contract."""

    if not isinstance(manifest, ExperimentalContinuationManifest):
        raise TypeError("manifest must be an ExperimentalContinuationManifest.")
    return {
        "schema": manifest.schema,
        "acquisition_id": manifest.acquisition_id,
        "lens_name": manifest.lens_name,
        "raw_dac_channel_name": manifest.raw_dac_channel_name,
        "route": manifest.route,
        "evidence_origin": manifest.evidence_origin,
        "grid": {
            "raw_dac_values": manifest.grid.raw_dac_values.tolist(),
            "anchor_raw_dac_values": list(manifest.grid.anchor_raw_dac_values),
            "max_abs_dq_per_dac": manifest.grid.max_abs_dq_per_dac,
            "target_max_abs_delta_q_rad": (
                manifest.grid.target_max_abs_delta_q_rad
            ),
        },
        "fixed_raw_dac_channels": [
            [name, value] for name, value in manifest.fixed_raw_dac_channels
        ],
        "tag_generation_z_m": manifest.tag_generation_z_m,
        "varied_lens_entrance_z_m": manifest.varied_lens_entrance_z_m,
        "sample_state_covariance": (
            None
            if manifest.sample_state_covariance is None
            else manifest.sample_state_covariance.tolist()
        ),
        "frames": [
            {
                "frame_id": frame.frame_id,
                "sequence_index": frame.sequence_index,
                "grid_index": frame.grid_index,
                "raw_dac_value": frame.raw_dac_value,
                "route": frame.route,
                "role": frame.role,
                "sample_state_x_theta_x_y_theta_y": (
                    None
                    if frame.sample_state_x_theta_x_y_theta_y is None
                    else list(frame.sample_state_x_theta_x_y_theta_y)
                ),
                "image_record_id": frame.image_record_id,
            }
            for frame in manifest.frames
        ],
    }


def continuation_manifest_from_payload(
    payload: Mapping[str, Any],
) -> ExperimentalContinuationManifest:
    """Parse a continuation payload while retaining legacy-origin provenance."""

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping.")
    required = {
        "schema",
        "acquisition_id",
        "lens_name",
        "raw_dac_channel_name",
        "route",
        "evidence_origin",
        "grid",
        "fixed_raw_dac_channels",
        "tag_generation_z_m",
        "varied_lens_entrance_z_m",
        "sample_state_covariance",
        "frames",
    }
    if set(payload) != required:
        missing = sorted(required - set(payload))
        extra = sorted(set(payload) - required)
        raise ValueError(f"Continuation payload keys differ; missing={missing}, extra={extra}.")
    grid_payload = payload["grid"]
    if not isinstance(grid_payload, Mapping):
        raise ValueError("grid must be a mapping.")
    grid_required = {
        "raw_dac_values",
        "anchor_raw_dac_values",
        "max_abs_dq_per_dac",
        "target_max_abs_delta_q_rad",
    }
    if set(grid_payload) != grid_required:
        raise ValueError("Continuation grid payload keys differ from the v1 schema.")
    grid = DACContinuationGrid(
        raw_dac_values=np.asarray(grid_payload["raw_dac_values"], dtype=float),
        anchor_raw_dac_values=tuple(grid_payload["anchor_raw_dac_values"]),
        max_abs_dq_per_dac=float(grid_payload["max_abs_dq_per_dac"]),
        target_max_abs_delta_q_rad=float(
            grid_payload["target_max_abs_delta_q_rad"]
        ),
    )
    frame_payloads = payload["frames"]
    if not isinstance(frame_payloads, Sequence) or isinstance(frame_payloads, (str, bytes)):
        raise ValueError("frames must be a sequence.")
    frames: list[ContinuationFrame] = []
    frame_required = {
        "frame_id",
        "sequence_index",
        "grid_index",
        "raw_dac_value",
        "route",
        "role",
        "sample_state_x_theta_x_y_theta_y",
        "image_record_id",
    }
    for frame_payload in frame_payloads:
        if not isinstance(frame_payload, Mapping) or set(frame_payload) != frame_required:
            raise ValueError("A frame differs from the v1 continuation schema.")
        state = frame_payload["sample_state_x_theta_x_y_theta_y"]
        frames.append(
            ContinuationFrame(
                frame_id=frame_payload["frame_id"],
                sequence_index=int(frame_payload["sequence_index"]),
                grid_index=int(frame_payload["grid_index"]),
                raw_dac_value=float(frame_payload["raw_dac_value"]),
                route=frame_payload["route"],
                role=frame_payload["role"],
                sample_state_x_theta_x_y_theta_y=(
                    None if state is None else tuple(state)
                ),
                image_record_id=frame_payload["image_record_id"],
            )
        )
    sample_covariance = payload["sample_state_covariance"]
    return ExperimentalContinuationManifest(
        acquisition_id=payload["acquisition_id"],
        lens_name=payload["lens_name"],
        raw_dac_channel_name=payload["raw_dac_channel_name"],
        route=payload["route"],
        grid=grid,
        frames=tuple(frames),
        fixed_raw_dac_channels=tuple(
            (str(pair[0]), float(pair[1]))
            for pair in payload["fixed_raw_dac_channels"]
        ),
        tag_generation_z_m=payload["tag_generation_z_m"],
        varied_lens_entrance_z_m=payload["varied_lens_entrance_z_m"],
        sample_state_covariance=(
            None
            if sample_covariance is None
            else np.asarray(sample_covariance, dtype=float)
        ),
        evidence_origin=payload["evidence_origin"],
        schema=payload["schema"],
    )


__all__ = [
    "MAX_CONTINUATION_PHASE_STEP_RAD",
    "CONTINUATION_MANIFEST_SCHEMA",
    "ContinuationObservableKind",
    "ContinuationRoute",
    "ContinuationFrameRole",
    "ContinuationEvidenceOrigin",
    "DACContinuationGrid",
    "design_raw_dac_continuation_grid",
    "PhaseContinuationResult",
    "PhaseContinuationError",
    "unwrap_oriented_ellipse_angles",
    "unwrap_signed_response_vector_rotations",
    "RelativeQSlopeFit",
    "fit_relative_q_slope",
    "QReferenceTransform",
    "transform_q_reference_covariance",
    "HARD_EDGE_SIGN_EXPLANATION",
    "HardEdgePhaseAlias",
    "HardEdgeAliasEnumeration",
    "enumerate_hard_edge_phase_aliases",
    "ContinuationFrame",
    "ExperimentalContinuationManifest",
    "build_experimental_continuation_manifest",
    "ContinuationManifestAudit",
    "audit_continuation_manifest",
    "continuation_manifest_to_payload",
    "continuation_manifest_from_payload",
]
