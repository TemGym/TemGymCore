"""Honest matched-model and local-null reporting for microscope inversion.

This module is intentionally separate from the checksum-bound forward/inversion
implementation.  It consumes the existing singular-subspace objects and adds
reporting semantics without changing their numerical construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

from .microscope_inversion import (
    EquivalentSolutionFamily,
    IdentifiableParameterSubspace,
)


ArrayLike = Any


def _immutable_float_array(values: ArrayLike) -> np.ndarray:
    """Return an owned, read-only floating-point array for report dataclasses."""

    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ParameterRecovery:
    """Matched-model recovery status for one named physical parameter.

    ``scaled_error`` and ``one_percent_gate_passed`` are deliberately ``None``
    when the parameter has appreciable weight in the local null space.  This
    prevents a convenient representative of an equivalent-solution family from
    being reported as an independently recovered lens parameter.
    """

    index: int
    name: str
    truth: float
    fitted: float
    scale: float
    null_space_weight: float
    identifiable_individually: bool
    included_in_named_gate: bool
    scaled_error: float | None
    named_metric_kind: str | None
    named_metric_error: float | None
    one_percent_gate_passed: bool | None


@dataclass(frozen=True)
class MatchedModelRecovery:
    """Recovery diagnostics in the experimentally identifiable SVD subspace.

    Active and null errors are dimensionless coordinates obtained after
    dividing physical parameter errors by the caller's declared scales.  The
    acceptance threshold is strictly less than 0.01, i.e. one percent of one
    declared scale in every active singular combination.
    """

    parameter_names: tuple[str, ...]
    truth_parameters: np.ndarray
    fitted_parameters: np.ndarray
    parameter_scales: np.ndarray
    scaled_parameter_error: np.ndarray
    active_combination_error_scaled: np.ndarray
    null_combination_error_scaled: np.ndarray
    maximum_active_combination_error_scaled: float
    one_percent_threshold: float
    active_one_percent_gate_passed: bool
    negligible_null_weight_threshold: float
    named_metric_policy: str
    requested_parameter_names: tuple[str, ...]
    parameters: tuple[ParameterRecovery, ...]

    @property
    def identifiable_parameter_names(self) -> tuple[str, ...]:
        return tuple(
            item.name for item in self.parameters if item.identifiable_individually
        )

    @property
    def non_identifiable_parameter_names(self) -> tuple[str, ...]:
        return tuple(
            item.name for item in self.parameters if not item.identifiable_individually
        )

    @property
    def gated_parameter_names(self) -> tuple[str, ...]:
        """Requested coordinates eligible for an individual physical gate."""

        return tuple(
            item.name
            for item in self.parameters
            if item.included_in_named_gate and item.identifiable_individually
        )

    @property
    def non_identifiable_requested_parameter_names(self) -> tuple[str, ...]:
        return tuple(
            item.name
            for item in self.parameters
            if item.included_in_named_gate and not item.identifiable_individually
        )

    @property
    def named_one_percent_gate_passed(self) -> bool:
        """Whether at least one eligible requested coordinate and all such gates pass."""

        individual = tuple(
            item.one_percent_gate_passed
            for item in self.parameters
            if item.included_in_named_gate and item.identifiable_individually
        )
        return bool(individual) and all(value is True for value in individual)

    @property
    def one_percent_gate_passed(self) -> bool:
        """Return the policy-appropriate one-percent acceptance gate.

        Under ``declared_scale`` this preserves the original conservative gate:
        both scaled active combinations and eligible named coordinates must
        pass.  Under ``physical_optical`` the SVD-coordinate result remains a
        diagnostic because its value depends on arbitrary design scales; only
        explicitly requested, individually identifiable optical coordinates
        enter this aggregate gate.
        """

        if self.named_metric_policy == "physical_optical":
            return self.named_one_percent_gate_passed
        return bool(
            self.active_one_percent_gate_passed
            and self.named_one_percent_gate_passed
        )


def assess_matched_model_recovery(
    subspace: IdentifiableParameterSubspace,
    fitted_parameters: ArrayLike,
    truth_parameters: ArrayLike,
    *,
    negligible_null_weight: float = 1.0e-8,
    named_metric_policy: str = "declared_scale",
    optical_parameter_names: tuple[str, ...] | None = None,
) -> MatchedModelRecovery:
    """Assess matched-model recovery without claiming null-mixed parameters.

    The signed physical error is first divided by ``subspace.parameter_scales``
    and then projected onto the orthonormal active and null bases.  The active
    recovery gate passes only when at least one active direction exists and the
    largest absolute active coordinate error is *strictly* below 0.01.

    A named physical parameter is separately reported and gated only if the
    sum of squared loadings of its row in ``null_basis_scaled`` is no greater
    than ``negligible_null_weight``.  Otherwise its error fields are ``None``
    and it is explicitly classified as non-identifiable.

    ``named_metric_policy="declared_scale"`` retains the dimensionless error
    ``(fit-truth)/scale`` and, by default, requests every parameter.  For a
    physical recovery claim use ``"physical_optical"`` and explicitly provide
    ``optical_parameter_names`` so calibration and route nuisances cannot enter
    the optics gate.  A ``*.log_length_m`` coordinate then uses absolute log
    error; other nonzero-truth optical coordinates use fractional truth error;
    and a zero-truth coordinate uses its declared scale as an absolute
    reference.  Every named gate is strictly below 0.01.
    """

    if not isinstance(subspace, IdentifiableParameterSubspace):
        raise TypeError("subspace must be an IdentifiableParameterSubspace.")
    fitted = np.asarray(fitted_parameters, dtype=float)
    truth = np.asarray(truth_parameters, dtype=float)
    expected = (subspace.n_parameters,)
    if fitted.shape != expected or not np.all(np.isfinite(fitted)):
        raise ValueError(f"fitted_parameters must be finite with shape {expected}.")
    if truth.shape != expected or not np.all(np.isfinite(truth)):
        raise ValueError(f"truth_parameters must be finite with shape {expected}.")
    null_tolerance = float(negligible_null_weight)
    if not np.isfinite(null_tolerance) or not 0.0 <= null_tolerance < 1.0:
        raise ValueError(
            "negligible_null_weight must be finite and lie in [0, 1)."
        )
    policy = str(named_metric_policy)
    if policy not in {"declared_scale", "physical_optical"}:
        raise ValueError(
            "named_metric_policy must be 'declared_scale' or 'physical_optical'."
        )
    names = tuple(subspace.parameter_names)
    if optical_parameter_names is None:
        if policy == "physical_optical":
            raise ValueError(
                "physical_optical reporting requires explicit optical_parameter_names."
            )
        requested_names = names
    else:
        requested_names = tuple(optical_parameter_names)
        if not requested_names or len(set(requested_names)) != len(requested_names):
            raise ValueError("optical_parameter_names must be non-empty and unique.")
        unknown = tuple(name for name in requested_names if name not in names)
        if unknown:
            raise ValueError(
                f"optical_parameter_names contains unknown names: {unknown}."
            )
    requested = set(requested_names)

    scales = np.asarray(subspace.parameter_scales, dtype=float)
    active_basis = np.asarray(subspace.active_basis_scaled, dtype=float)
    null_basis = np.asarray(subspace.null_basis_scaled, dtype=float)
    if scales.shape != expected or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("subspace contains invalid parameter scales.")
    if active_basis.shape != (subspace.n_parameters, subspace.rank):
        raise ValueError("subspace active basis has an inconsistent shape.")
    if null_basis.shape != (subspace.n_parameters, subspace.nullity):
        raise ValueError("subspace null basis has an inconsistent shape.")

    scaled_error = (fitted - truth) / scales
    active_error = active_basis.T @ scaled_error
    null_error = null_basis.T @ scaled_error
    maximum_active = (
        float(np.max(np.abs(active_error))) if active_error.size else 0.0
    )
    threshold = 0.01
    active_passed = bool(active_error.size and maximum_active < threshold)
    null_weights = np.sum(null_basis**2, axis=1)

    parameter_reports: list[ParameterRecovery] = []
    for index, name in enumerate(subspace.parameter_names):
        identifiable = bool(null_weights[index] <= null_tolerance)
        included = name in requested
        parameter_error = (
            float(scaled_error[index]) if identifiable and included else None
        )
        if parameter_error is None:
            metric_kind = None
            metric_error = None
        elif policy == "declared_scale":
            metric_kind = "absolute_fraction_of_declared_scale"
            metric_error = float(abs(parameter_error))
        elif name == "log_length_m" or name.endswith(".log_length_m") or name.endswith(
            "_log_length_m"
        ):
            metric_kind = "absolute_log_length_error"
            metric_error = float(abs(fitted[index] - truth[index]))
        elif name in {"source_z_m", "detector_z_m"} or name.endswith(".z_m"):
            # Absolute axial coordinates depend on the arbitrary choice of
            # z-origin.  Measure recovery against the declared local axial
            # scale instead of dividing by the coordinate value itself.
            metric_kind = "absolute_axial_error_over_declared_scale"
            metric_error = float(abs((fitted[index] - truth[index]) / scales[index]))
        elif truth[index] != 0.0:
            metric_kind = "absolute_fractional_truth_error"
            metric_error = float(abs((fitted[index] - truth[index]) / truth[index]))
        else:
            metric_kind = "absolute_error_over_declared_zero_reference_scale"
            metric_error = float(abs((fitted[index] - truth[index]) / scales[index]))
        parameter_reports.append(
            ParameterRecovery(
                index=index,
                name=name,
                truth=float(truth[index]),
                fitted=float(fitted[index]),
                scale=float(scales[index]),
                null_space_weight=float(null_weights[index]),
                identifiable_individually=identifiable,
                included_in_named_gate=included,
                scaled_error=parameter_error,
                named_metric_kind=metric_kind,
                named_metric_error=metric_error,
                one_percent_gate_passed=(
                    bool(metric_error < threshold) if metric_error is not None else None
                ),
            )
        )

    return MatchedModelRecovery(
        parameter_names=tuple(subspace.parameter_names),
        truth_parameters=_immutable_float_array(truth),
        fitted_parameters=_immutable_float_array(fitted),
        parameter_scales=_immutable_float_array(scales),
        scaled_parameter_error=_immutable_float_array(scaled_error),
        active_combination_error_scaled=_immutable_float_array(active_error),
        null_combination_error_scaled=_immutable_float_array(null_error),
        maximum_active_combination_error_scaled=maximum_active,
        one_percent_threshold=threshold,
        active_one_percent_gate_passed=active_passed,
        negligible_null_weight_threshold=null_tolerance,
        named_metric_policy=policy,
        requested_parameter_names=requested_names,
        parameters=tuple(parameter_reports),
    )


@dataclass(frozen=True)
class NullFamilyLoading:
    """One dominant named loading of a unit direction in scaled coordinates."""

    parameter_index: int
    parameter_name: str
    scaled_loading: float
    absolute_scaled_loading: float


@dataclass(frozen=True)
class NullFamilyEndpoint:
    """Numerically evaluated endpoint of a bounded local-null direction."""

    signed_scaled_coordinate: float
    parameters: np.ndarray
    cost: float
    delta_cost: float
    active_bounds: tuple[str, ...]


@dataclass(frozen=True)
class LocalNullFamilyVerification:
    """Sampled straight-segment verification of one local null tangent."""

    family_index: int
    direction_scaled: np.ndarray
    direction_physical: np.ndarray
    dominant_loadings: tuple[NullFamilyLoading, ...]
    negative_endpoint: NullFamilyEndpoint
    positive_endpoint: NullFamilyEndpoint
    sampled_scaled_coordinates: np.ndarray
    sampled_costs: np.ndarray
    sampled_delta_costs: np.ndarray
    maximum_absolute_endpoint_delta_cost: float
    maximum_absolute_sampled_delta_cost: float
    requested_negative_scaled_coordinate: float
    requested_positive_scaled_coordinate: float
    negative_physical_bound_reached: bool
    positive_physical_bound_reached: bool
    status: str

    @property
    def verified_straight_equivalent(self) -> bool:
        """Whether every sampled point on this straight tangent segment is flat."""

        return self.status == "verified_straight_equivalent"

    @property
    def verified_equivalent(self) -> bool:
        """Compatibility alias; this does not claim a profiled nonlinear family."""

        return self.verified_straight_equivalent


@dataclass(frozen=True)
class LocalNullFamilyReport:
    """Sampled straight-segment audit for every direction in a local null basis."""

    parameter_names: tuple[str, ...]
    base_parameters: np.ndarray
    parameter_scales: np.ndarray
    base_cost: float
    delta_cost_tolerance: float
    straight_sample_count: int
    families: tuple[LocalNullFamilyVerification, ...]

    @property
    def all_verified_straight_equivalent(self) -> bool:
        return bool(self.families) and all(
            family.verified_straight_equivalent for family in self.families
        )

    @property
    def all_verified_equivalent(self) -> bool:
        """Compatibility alias; no nonlinear profiling or reoptimization is implied."""

        return self.all_verified_straight_equivalent


def _finite_residual_cost(
    residual_function: Callable[[jnp.ndarray], ArrayLike],
    parameters: np.ndarray,
    *,
    expected_size: int | None = None,
) -> tuple[float, int]:
    residuals = np.asarray(residual_function(jnp.asarray(parameters)), dtype=float).ravel()
    if residuals.size == 0 or not np.all(np.isfinite(residuals)):
        raise ValueError("residual_function must return a non-empty finite array.")
    if expected_size is not None and residuals.size != expected_size:
        raise ValueError("residual_function changed residual-vector size at an endpoint.")
    return float(0.5 * residuals @ residuals), int(residuals.size)


def _optional_finite_residual_cost(
    residual_function: Callable[[jnp.ndarray], ArrayLike],
    parameters: np.ndarray,
    *,
    expected_size: int,
) -> float | None:
    """Return a finite cost, while treating model-domain failures as inadmissible.

    A hard-edge trial may meet every declared parameter bound and still place an
    inverse-map observable exactly at a conjugacy.  Exceptions and non-finite
    values at such a trial delimit the connected evaluable segment; a changed
    residual-vector size remains a programming/schema error and is never
    hidden as an optical-domain boundary.
    """

    try:
        residuals = np.asarray(
            residual_function(jnp.asarray(parameters)), dtype=float
        ).ravel()
    except Exception:  # The caller records this point as outside the model domain.
        return None
    if residuals.size != expected_size:
        raise ValueError("residual_function changed residual-vector size at an endpoint.")
    if residuals.size == 0 or not np.all(np.isfinite(residuals)):
        return None
    return float(0.5 * residuals @ residuals)


def _connected_finite_coordinate(
    residual_function: Callable[[jnp.ndarray], ArrayLike],
    base_parameters: np.ndarray,
    direction_physical: np.ndarray,
    requested_coordinate: float,
    *,
    expected_size: int,
    probe_count: int,
    bisection_steps: int = 48,
) -> tuple[float, float, bool]:
    """Find the finite segment connected to zero along one signed direction."""

    requested = float(requested_coordinate)
    if requested == 0.0:
        cost = _optional_finite_residual_cost(
            residual_function, base_parameters, expected_size=expected_size
        )
        if cost is None:
            raise RuntimeError("The already-validated base point became non-finite.")
        return 0.0, cost, True
    fractions = np.linspace(0.0, 1.0, max(int(probe_count), 3))[1:]
    last_coordinate = 0.0
    last_cost = _optional_finite_residual_cost(
        residual_function, base_parameters, expected_size=expected_size
    )
    if last_cost is None:
        raise RuntimeError("The already-validated base point became non-finite.")
    first_invalid: float | None = None
    for fraction in fractions:
        coordinate = requested * float(fraction)
        cost = _optional_finite_residual_cost(
            residual_function,
            base_parameters + coordinate * direction_physical,
            expected_size=expected_size,
        )
        if cost is None:
            first_invalid = coordinate
            break
        last_coordinate = coordinate
        last_cost = cost
    if first_invalid is None:
        return requested, float(last_cost), True

    finite_coordinate = last_coordinate
    finite_cost = float(last_cost)
    invalid_coordinate = first_invalid
    for _ in range(int(bisection_steps)):
        midpoint = 0.5 * (finite_coordinate + invalid_coordinate)
        cost = _optional_finite_residual_cost(
            residual_function,
            base_parameters + midpoint * direction_physical,
            expected_size=expected_size,
        )
        if cost is None:
            invalid_coordinate = midpoint
        else:
            finite_coordinate = midpoint
            finite_cost = cost
    # Pull one representable step toward the known-finite base so downstream
    # profiles do not begin on a floating-point classification knife edge.
    safe_coordinate = float(np.nextafter(finite_coordinate, 0.0))
    safe_cost = _optional_finite_residual_cost(
        residual_function,
        base_parameters + safe_coordinate * direction_physical,
        expected_size=expected_size,
    )
    if safe_cost is not None:
        finite_coordinate = safe_coordinate
        finite_cost = safe_cost
    return float(finite_coordinate), float(finite_cost), False


def _finite_bounds(
    bounds: tuple[ArrayLike, ArrayLike], n_parameters: int
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.broadcast_to(np.asarray(bounds[0], dtype=float), (n_parameters,)).copy()
    upper = np.broadcast_to(np.asarray(bounds[1], dtype=float), (n_parameters,)).copy()
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("verify_local_null_families requires finite bounds.")
    if np.any(lower >= upper):
        raise ValueError("Every lower bound must be smaller than its upper bound.")
    return lower, upper


def _direction_extents(
    parameters: np.ndarray,
    direction: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[float, float]:
    positive_limits = np.full(direction.shape, np.inf, dtype=float)
    negative_limits = np.full(direction.shape, np.inf, dtype=float)
    positive = direction > 0.0
    negative = direction < 0.0
    positive_limits[positive] = (upper[positive] - parameters[positive]) / direction[
        positive
    ]
    positive_limits[negative] = (lower[negative] - parameters[negative]) / direction[
        negative
    ]
    negative_limits[negative] = (upper[negative] - parameters[negative]) / (-direction[
        negative
    ])
    negative_limits[positive] = (parameters[positive] - lower[positive]) / direction[
        positive
    ]
    return float(np.min(negative_limits)), float(np.min(positive_limits))


def verify_local_null_families(
    family: EquivalentSolutionFamily,
    residual_function: Callable[[jnp.ndarray], ArrayLike],
    *,
    bounds: tuple[ArrayLike, ArrayLike],
    parameter_scales: ArrayLike,
    delta_cost_tolerance: float,
    dominant_loading_count: int = 3,
    straight_sample_count: int = 9,
) -> LocalNullFamilyReport:
    """Sample every bounded straight tangent segment of a local null basis.

    Each physical null vector is renormalized to unit Euclidean length after
    division by ``parameter_scales``.  The positive and negative coordinates
    therefore have the common meaning "declared-scale units".  Both directions
    are extended toward the first finite physical box bound and evaluated with
    the caller's actual residual.  If a hard-edge conjugacy or another
    non-finite model-domain boundary is encountered first, the connected finite
    segment is located by probing and bisection and the family is explicitly
    labelled ``finite_domain_truncated`` rather than aborting unrelated null
    audits or being called equivalent.  At least five points are evaluated
    across the retained segment, always including the negative endpoint, zero,
    and the positive endpoint.  A family is labelled
    ``verified_straight_equivalent`` only when both physical bounds were reached
    and the absolute cost change at every sampled point is no greater than
    ``delta_cost_tolerance``; otherwise it remains a classified candidate.  This
    is explicitly not a curved-family profile and does not reoptimize
    coordinates orthogonal to the tangent.

    Finite bounds are required because an unbounded local tangent has no
    numerically defined admissible endpoint.
    """

    if not isinstance(family, EquivalentSolutionFamily):
        raise TypeError("family must be an EquivalentSolutionFamily.")
    if not callable(residual_function):
        raise TypeError("residual_function must be callable.")
    parameters = np.asarray(family.parameters, dtype=float)
    n_parameters = parameters.size
    if parameters.shape != (len(family.parameter_names),) or not np.all(
        np.isfinite(parameters)
    ):
        raise ValueError("family parameters and names are inconsistent or non-finite.")
    basis = np.asarray(family.basis, dtype=float)
    if basis.shape != (n_parameters, family.nullity) or not np.all(np.isfinite(basis)):
        raise ValueError("family basis has an inconsistent shape or non-finite values.")
    scales = np.asarray(parameter_scales, dtype=float)
    if scales.shape != (n_parameters,) or not np.all(np.isfinite(scales)) or np.any(
        scales <= 0.0
    ):
        raise ValueError(
            "parameter_scales must be a finite positive vector matching the family."
        )
    tolerance = float(delta_cost_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("delta_cost_tolerance must be finite and non-negative.")
    if int(dominant_loading_count) != dominant_loading_count or int(
        dominant_loading_count
    ) < 1:
        raise ValueError(
            "dominant_loading_count must be a positive integer."
        )
    loading_count = min(int(dominant_loading_count), n_parameters)
    if int(straight_sample_count) != straight_sample_count or int(
        straight_sample_count
    ) < 5:
        raise ValueError("straight_sample_count must be an integer of at least five.")
    sample_count = int(straight_sample_count)
    lower, upper = _finite_bounds(bounds, n_parameters)
    if np.any(parameters < lower) or np.any(parameters > upper):
        raise ValueError("family base parameters must lie within bounds.")

    base_cost, residual_size = _finite_residual_cost(residual_function, parameters)
    verifications: list[LocalNullFamilyVerification] = []
    for family_index in range(family.nullity):
        direction_scaled = basis[:, family_index] / scales
        norm = float(np.linalg.norm(direction_scaled))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError(f"family basis column {family_index} has zero scaled norm.")
        direction_scaled = direction_scaled / norm
        # SVD-vector signs are arbitrary.  Canonicalize them so that reports are
        # reproducible and the largest-magnitude named loading is positive.
        anchor = int(np.argmax(np.abs(direction_scaled)))
        if direction_scaled[anchor] < 0.0:
            direction_scaled = -direction_scaled
        direction_physical = scales * direction_scaled
        negative_extent, positive_extent = _direction_extents(
            parameters, direction_physical, lower, upper
        )
        if not np.isfinite(positive_extent) or not np.isfinite(negative_extent):
            raise ValueError(
                f"family direction {family_index} does not meet a finite bound."
            )
        if positive_extent < 0.0 or negative_extent < 0.0:
            raise RuntimeError("Computed a negative admissible null-family extent.")

        def endpoint(signed_coordinate: float, cost: float) -> NullFamilyEndpoint:
            endpoint_parameters = parameters + signed_coordinate * direction_physical
            endpoint_parameters = np.minimum(
                np.maximum(endpoint_parameters, lower), upper
            )
            bound_atol = 128.0 * np.finfo(float).eps
            hits: list[str] = []
            for index, name in enumerate(family.parameter_names):
                scale = max(1.0, abs(endpoint_parameters[index]))
                if abs(endpoint_parameters[index] - lower[index]) <= bound_atol * scale:
                    hits.append(f"{name}:lower")
                if abs(endpoint_parameters[index] - upper[index]) <= bound_atol * scale:
                    hits.append(f"{name}:upper")
            return NullFamilyEndpoint(
                signed_scaled_coordinate=float(signed_coordinate),
                parameters=_immutable_float_array(endpoint_parameters),
                cost=cost,
                delta_cost=float(cost - base_cost),
                active_bounds=tuple(hits),
            )

        requested_negative = -negative_extent
        requested_positive = positive_extent
        probe_count = max(4 * sample_count + 1, 33)
        (
            finite_negative,
            finite_negative_cost,
            negative_bound_reached,
        ) = _connected_finite_coordinate(
            residual_function,
            parameters,
            direction_physical,
            requested_negative,
            expected_size=residual_size,
            probe_count=probe_count,
        )
        (
            finite_positive,
            finite_positive_cost,
            positive_bound_reached,
        ) = _connected_finite_coordinate(
            residual_function,
            parameters,
            direction_physical,
            requested_positive,
            expected_size=residual_size,
            probe_count=probe_count,
        )
        negative_count = sample_count // 2 + 1
        positive_count = sample_count - negative_count + 1
        sampled_coordinates = np.concatenate(
            (
                np.linspace(finite_negative, 0.0, negative_count),
                np.linspace(0.0, finite_positive, positive_count)[1:],
            )
        )
        sampled_costs = np.empty(sample_count, dtype=float)
        sampled_delta_costs = np.empty(sample_count, dtype=float)
        invalid_sample = False
        for sample_index, coordinate in enumerate(sampled_coordinates):
            if coordinate == 0.0:
                cost = base_cost
            else:
                sample_parameters = parameters + coordinate * direction_physical
                optional_cost = _optional_finite_residual_cost(
                    residual_function,
                    sample_parameters,
                    expected_size=residual_size,
                )
                if optional_cost is None:
                    invalid_sample = True
                    cost = np.nan
                else:
                    cost = optional_cost
            sampled_costs[sample_index] = cost
            sampled_delta_costs[sample_index] = cost - base_cost

        negative_endpoint = endpoint(finite_negative, finite_negative_cost)
        positive_endpoint = endpoint(finite_positive, finite_positive_cost)
        maximum_endpoint_delta = float(
            max(abs(negative_endpoint.delta_cost), abs(positive_endpoint.delta_cost))
        )
        maximum_sampled_delta = (
            float("inf")
            if invalid_sample
            else float(np.max(np.abs(sampled_delta_costs)))
        )
        order = np.argsort(-np.abs(direction_scaled), kind="stable")[:loading_count]
        loadings = tuple(
            NullFamilyLoading(
                parameter_index=int(index),
                parameter_name=family.parameter_names[int(index)],
                scaled_loading=float(direction_scaled[index]),
                absolute_scaled_loading=float(abs(direction_scaled[index])),
            )
            for index in order
        )
        if finite_negative == 0.0 or finite_positive == 0.0:
            status = "degenerate_bounded_segment"
        elif not negative_bound_reached or not positive_bound_reached:
            status = "finite_domain_truncated"
        elif invalid_sample:
            status = "invalid_straight_segment"
        elif maximum_sampled_delta <= tolerance:
            status = "verified_straight_equivalent"
        else:
            status = "local_null_candidate"
        verifications.append(
            LocalNullFamilyVerification(
                family_index=family_index,
                direction_scaled=_immutable_float_array(direction_scaled),
                direction_physical=_immutable_float_array(direction_physical),
                dominant_loadings=loadings,
                negative_endpoint=negative_endpoint,
                positive_endpoint=positive_endpoint,
                sampled_scaled_coordinates=_immutable_float_array(sampled_coordinates),
                sampled_costs=_immutable_float_array(sampled_costs),
                sampled_delta_costs=_immutable_float_array(sampled_delta_costs),
                maximum_absolute_endpoint_delta_cost=maximum_endpoint_delta,
                maximum_absolute_sampled_delta_cost=maximum_sampled_delta,
                requested_negative_scaled_coordinate=float(requested_negative),
                requested_positive_scaled_coordinate=float(requested_positive),
                negative_physical_bound_reached=negative_bound_reached,
                positive_physical_bound_reached=positive_bound_reached,
                status=status,
            )
        )

    return LocalNullFamilyReport(
        parameter_names=tuple(family.parameter_names),
        base_parameters=_immutable_float_array(parameters),
        parameter_scales=_immutable_float_array(scales),
        base_cost=base_cost,
        delta_cost_tolerance=tolerance,
        straight_sample_count=sample_count,
        families=tuple(verifications),
    )


__all__ = [
    "LocalNullFamilyReport",
    "LocalNullFamilyVerification",
    "MatchedModelRecovery",
    "NullFamilyEndpoint",
    "NullFamilyLoading",
    "ParameterRecovery",
    "assess_matched_model_recovery",
    "verify_local_null_families",
]
