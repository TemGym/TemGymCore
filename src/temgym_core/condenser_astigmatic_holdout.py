"""Leak-free held-out validation for the astigmatic condenser experiment.

The checksum-bound astigmatic design contains three singleton sample states
that can be removed from fitting while retaining all twelve locally
identifiable CL1/CL3/C-mini target directions below the declared one-percent
scaled-uncertainty threshold.  This module declares that split and keeps the
split on *unwhitened acquisition blocks*.  Consequently, held-out values can
never enter a training residual through a dense Cholesky solve.

The default split is a matched-hard-edge, local-design result.  It is not an
instrument-readiness claim: nonlinear recovery, model discrepancy, detector
feasibility, and truth-envelope robustness remain separate gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from .condenser_astigmatic_design import (
    AstigmaticCondenserDesign,
    AstigmaticDesignProvenanceError,
    build_astigmatic_design_matrix,
)
from .condenser_astigmatic_likelihood import AstigmaticCondenserLikelihood


DEFAULT_ASTIGMATIC_HELD_OUT_STATE_INDICES = (2, 3, 7)
"""Zero-based manifest state indices reserved for prediction only."""

ASTIGMATIC_SAMPLE_OBSERVABLE_NAMES = (
    "log_rms_x",
    "log_rms_y",
    "correlation_x_y",
    "log_rms_theta_x",
    "log_rms_theta_y",
    "correlation_theta_x_theta_y",
    "astigmatic_tag_orientation",
)

_SAMPLE_ACQUISITION_COUNT = 48
_OBSERVABLES_PER_ACQUISITION = 7
_BASE_RESIDUAL_COUNT = 92
_PARAMETER_COUNT = 43
_TARGET_COUNT = 12
_EXPECTED_DEFAULT_TRAINING_SHAPE = (407, 43)
_EXPECTED_DEFAULT_FULL_RANK = 41
_EXPECTED_DEFAULT_NUISANCE_RANK = 29
_EXPECTED_DEFAULT_TARGET_RANK = 12
_EXPECTED_DEFAULT_MAXIMUM_SCALED_UNCERTAINTY = 0.009980278123994601
_EXPECTED_DEFAULT_SCALED_UNCERTAINTIES = np.asarray(
    (
        0.0017081818,
        0.0099802781,
        0.0013192682,
        0.0011452800,
        0.0002092339,
        0.0063354988,
        0.0021379453,
        0.0018453217,
        0.0002880796,
        0.0084525911,
        0.0095837689,
        0.0019777526,
    ),
    dtype=float,
)


class AstigmaticHeldOutLeakageError(ValueError):
    """Raised when a dense covariance cannot support a valid Gaussian split."""


class AstigmaticHeldOutAuditError(RuntimeError):
    """Raised when the checksum-bound default held-out audit changes."""


def _immutable_array(
    value: Any,
    name: str,
    *,
    shape: tuple[int, ...] | None = None,
    dtype: Any = float,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    result = array.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class AstigmaticHeldOutSplit:
    """Acquisition-level train/test indices and marginal noise factors."""

    design_sha256: str
    branch_b_payload_sha256: str
    held_out_state_indices: tuple[int, ...]
    held_out_acquisition_indices: tuple[int, ...]
    training_acquisition_indices: tuple[int, ...]
    held_out_sample_indices: np.ndarray
    training_sample_indices: np.ndarray
    held_out_full_residual_indices: np.ndarray
    training_full_residual_indices: np.ndarray
    held_out_from_training_regression: np.ndarray
    held_out_noise_cholesky: np.ndarray
    training_noise_cholesky: np.ndarray
    maximum_train_held_out_noise_correlation: float

    def __post_init__(self) -> None:
        held_states = tuple(int(value) for value in self.held_out_state_indices)
        held_acquisitions = tuple(
            int(value) for value in self.held_out_acquisition_indices
        )
        training_acquisitions = tuple(
            int(value) for value in self.training_acquisition_indices
        )
        if not held_states or tuple(sorted(set(held_states))) != held_states:
            raise ValueError("held_out_state_indices must be unique and sorted.")
        if not held_acquisitions or tuple(sorted(set(held_acquisitions))) != (
            held_acquisitions
        ):
            raise ValueError(
                "held_out_acquisition_indices must be unique and sorted."
            )
        if set(held_acquisitions) & set(training_acquisitions):
            raise ValueError("Training and held-out acquisitions overlap.")
        if tuple(sorted((*held_acquisitions, *training_acquisitions))) != tuple(
            range(_SAMPLE_ACQUISITION_COUNT)
        ):
            raise ValueError("The acquisition split must cover all 48 acquisitions.")

        held_size = len(held_acquisitions) * _OBSERVABLES_PER_ACQUISITION
        training_size = (
            len(training_acquisitions) * _OBSERVABLES_PER_ACQUISITION
        )
        arrays = {
            "held_out_sample_indices": _immutable_array(
                self.held_out_sample_indices,
                "held_out_sample_indices",
                shape=(held_size,),
                dtype=int,
            ),
            "training_sample_indices": _immutable_array(
                self.training_sample_indices,
                "training_sample_indices",
                shape=(training_size,),
                dtype=int,
            ),
            "held_out_full_residual_indices": _immutable_array(
                self.held_out_full_residual_indices,
                "held_out_full_residual_indices",
                shape=(held_size,),
                dtype=int,
            ),
            "training_full_residual_indices": _immutable_array(
                self.training_full_residual_indices,
                "training_full_residual_indices",
                shape=(_BASE_RESIDUAL_COUNT + training_size,),
                dtype=int,
            ),
            "held_out_noise_cholesky": _immutable_array(
                self.held_out_noise_cholesky,
                "held_out_noise_cholesky",
                shape=(held_size, held_size),
            ),
            "held_out_from_training_regression": _immutable_array(
                self.held_out_from_training_regression,
                "held_out_from_training_regression",
                shape=(held_size, training_size),
            ),
            "training_noise_cholesky": _immutable_array(
                self.training_noise_cholesky,
                "training_noise_cholesky",
                shape=(training_size, training_size),
            ),
        }
        for name in ("design_sha256", "branch_b_payload_sha256"):
            digest = str(getattr(self, name)).strip()
            if len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
            object.__setattr__(self, name, digest)
        correlation = float(self.maximum_train_held_out_noise_correlation)
        if not np.isfinite(correlation) or correlation < 0.0:
            raise ValueError(
                "maximum_train_held_out_noise_correlation must be finite and "
                "non-negative."
            )
        object.__setattr__(self, "held_out_state_indices", held_states)
        object.__setattr__(
            self, "held_out_acquisition_indices", held_acquisitions
        )
        object.__setattr__(
            self, "training_acquisition_indices", training_acquisitions
        )
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        object.__setattr__(
            self, "maximum_train_held_out_noise_correlation", correlation
        )

    @property
    def held_out_observation_count(self) -> int:
        return int(self.held_out_sample_indices.size)

    @property
    def training_residual_count(self) -> int:
        return int(self.training_full_residual_indices.size)


@dataclass(frozen=True)
class AstigmaticHeldOutTrainingAudit:
    """Nominal Fisher audit after removing the registered held-out states."""

    training_matrix_shape: tuple[int, int]
    held_out_matrix_shape: tuple[int, int]
    full_rank: int
    full_nullity: int
    nuisance_rank: int
    profiled_target_rank: int
    held_out_response_rank: int
    target_names: tuple[str, ...]
    profiled_target_singular_values: np.ndarray
    scaled_target_standard_uncertainties: np.ndarray
    maximum_scaled_target_standard_uncertainty: float
    conditional_precision_pass: bool
    checksum_contract_verified: bool

    def __post_init__(self) -> None:
        if self.training_matrix_shape[1] != _PARAMETER_COUNT:
            raise ValueError("The training matrix must have 43 columns.")
        if self.held_out_matrix_shape[1] != _PARAMETER_COUNT:
            raise ValueError("The held-out matrix must have 43 columns.")
        if len(self.target_names) != _TARGET_COUNT:
            raise ValueError("Expected twelve target names.")
        singular = _immutable_array(
            self.profiled_target_singular_values,
            "profiled_target_singular_values",
            shape=(_TARGET_COUNT,),
        )
        uncertainty = _immutable_array(
            self.scaled_target_standard_uncertainties,
            "scaled_target_standard_uncertainties",
            shape=(_TARGET_COUNT,),
        )
        maximum = float(self.maximum_scaled_target_standard_uncertainty)
        if not np.isfinite(maximum) or maximum < 0.0:
            raise ValueError("Maximum target uncertainty must be finite and non-negative.")
        object.__setattr__(self, "profiled_target_singular_values", singular)
        object.__setattr__(
            self, "scaled_target_standard_uncertainties", uncertainty
        )
        object.__setattr__(
            self, "maximum_scaled_target_standard_uncertainty", maximum
        )


@dataclass(frozen=True)
class AstigmaticHeldOutEvaluation:
    """Post-fit residuals for the 21 observations never used by training."""

    state_indices: tuple[int, ...]
    acquisition_indices: tuple[int, ...]
    residual: np.ndarray
    raw_difference: np.ndarray
    conditional_innovation: np.ndarray
    predictive_covariance: np.ndarray
    chi_square: float
    degrees_of_freedom: int
    rms_standardized_residual: float
    maximum_absolute_standardized_residual: float

    def __post_init__(self) -> None:
        size = len(self.acquisition_indices) * _OBSERVABLES_PER_ACQUISITION
        residual = _immutable_array(
            self.residual, "residual", shape=(size,)
        )
        difference = _immutable_array(
            self.raw_difference, "raw_difference", shape=(size,)
        )
        innovation = _immutable_array(
            self.conditional_innovation,
            "conditional_innovation",
            shape=(size,),
        )
        covariance = _immutable_array(
            self.predictive_covariance,
            "predictive_covariance",
            shape=(size, size),
        )
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "raw_difference", difference)
        object.__setattr__(self, "conditional_innovation", innovation)
        object.__setattr__(self, "predictive_covariance", covariance)


def _state_acquisition_indices(
    design: AstigmaticCondenserDesign,
    state_indices: Sequence[int],
) -> tuple[int, ...]:
    states = tuple(int(value) for value in state_indices)
    if not states or tuple(sorted(set(states))) != states:
        raise ValueError("state_indices must be unique, non-empty, and sorted.")
    if states[0] < 0 or states[-1] >= design.controls.shape[0]:
        raise IndexError("A held-out state index lies outside the design.")
    starts = np.cumsum(np.r_[0, design.repeat_counts[:-1]])
    stops = np.cumsum(design.repeat_counts)
    return tuple(
        acquisition
        for state in states
        for acquisition in range(int(starts[state]), int(stops[state]))
    )


def _flattened_sample_indices(acquisitions: Sequence[int]) -> np.ndarray:
    result = np.concatenate(
        tuple(
            np.arange(
                acquisition * _OBSERVABLES_PER_ACQUISITION,
                (acquisition + 1) * _OBSERVABLES_PER_ACQUISITION,
                dtype=int,
            )
            for acquisition in acquisitions
        )
    )
    result.setflags(write=False)
    return result


def make_astigmatic_held_out_split(
    likelihood: AstigmaticCondenserLikelihood,
    *,
    held_out_state_indices: Sequence[int] = (
        DEFAULT_ASTIGMATIC_HELD_OUT_STATE_INDICES
    ),
) -> AstigmaticHeldOutSplit:
    """Create a covariance-safe acquisition split.

    The full sample covariance is first reconstructed, then marginalized into
    training and held-out blocks.  Cross-block covariance is handled by the
    exact Gaussian conditional: training uses the marginal ``C_TT`` while the
    test innovation uses ``C_HT C_TT^-1`` and the Schur complement.  This is
    stricter than masking an already-whitened residual, which can leak
    held-out values into training.
    """

    if not isinstance(likelihood, AstigmaticCondenserLikelihood):
        raise TypeError("likelihood must be an AstigmaticCondenserLikelihood.")
    states = tuple(int(value) for value in held_out_state_indices)
    held_acquisitions = _state_acquisition_indices(likelihood.design, states)
    training_acquisitions = tuple(
        index
        for index in range(_SAMPLE_ACQUISITION_COUNT)
        if index not in set(held_acquisitions)
    )
    held = _flattened_sample_indices(held_acquisitions)
    training = _flattened_sample_indices(training_acquisitions)

    full_factor = np.asarray(
        likelihood.observations.sample_noise_cholesky, dtype=float
    )
    covariance = full_factor @ full_factor.T
    covariance_training = covariance[np.ix_(training, training)]
    covariance_held = covariance[np.ix_(held, held)]
    cross = covariance[np.ix_(training, held)]
    denominator = np.sqrt(
        np.diag(covariance_training)[:, None]
        * np.diag(covariance_held)[None, :]
    )
    maximum_correlation = float(
        np.max(np.abs(cross) / denominator, initial=0.0)
    )
    try:
        training_factor = np.linalg.cholesky(covariance_training)
        covariance_training_inverse_cross = np.linalg.solve(
            training_factor.T,
            np.linalg.solve(training_factor, cross),
        )
        held_from_training = covariance_training_inverse_cross.T
        conditional_covariance = covariance_held - (
            cross.T @ covariance_training_inverse_cross
        )
        conditional_covariance = 0.5 * (
            conditional_covariance + conditional_covariance.T
        )
        held_factor = np.linalg.cholesky(conditional_covariance)
    except np.linalg.LinAlgError as exc:
        raise AstigmaticHeldOutLeakageError(
            "The dense sample covariance does not yield positive-definite "
            "training and held-out conditional blocks."
        ) from exc

    held_full = _BASE_RESIDUAL_COUNT + held
    training_full = np.concatenate(
        (np.arange(_BASE_RESIDUAL_COUNT, dtype=int), _BASE_RESIDUAL_COUNT + training)
    )
    return AstigmaticHeldOutSplit(
        design_sha256=likelihood.design.sha256,
        branch_b_payload_sha256=likelihood.design.branch_b_payload_sha256,
        held_out_state_indices=states,
        held_out_acquisition_indices=held_acquisitions,
        training_acquisition_indices=training_acquisitions,
        held_out_sample_indices=held,
        training_sample_indices=training,
        held_out_full_residual_indices=held_full,
        training_full_residual_indices=training_full,
        held_out_from_training_regression=held_from_training,
        held_out_noise_cholesky=held_factor,
        training_noise_cholesky=training_factor,
        maximum_train_held_out_noise_correlation=maximum_correlation,
    )


def _verify_split(
    likelihood: AstigmaticCondenserLikelihood,
    split: AstigmaticHeldOutSplit,
) -> None:
    if not isinstance(split, AstigmaticHeldOutSplit):
        raise TypeError("split must be an AstigmaticHeldOutSplit.")
    if split.design_sha256 != likelihood.design.sha256:
        raise AstigmaticDesignProvenanceError(
            "The held-out split belongs to a different astigmatic design."
        )
    if (
        split.branch_b_payload_sha256
        != likelihood.design.branch_b_payload_sha256
    ):
        raise AstigmaticDesignProvenanceError(
            "The held-out split belongs to a different Branch-B payload."
        )


def _sample_difference(
    likelihood: AstigmaticCondenserLikelihood,
    parameters: Any,
) -> jnp.ndarray:
    prediction = likelihood.predict(parameters).sample_observables
    difference = prediction - jnp.asarray(
        likelihood.observations.sample_observables
    )
    wrapped_tag = 0.5 * jnp.arctan2(
        jnp.sin(2.0 * difference[:, 6]),
        jnp.cos(2.0 * difference[:, 6]),
    )
    return difference.at[:, 6].set(wrapped_tag).reshape(-1)


def astigmatic_training_residual(
    likelihood: AstigmaticCondenserLikelihood,
    parameters: Any,
    split: AstigmaticHeldOutSplit,
) -> jnp.ndarray:
    """Return 88 edge + 4 prior + 315 sample rows, with no test leakage."""

    _verify_split(likelihood, split)
    difference = _sample_difference(likelihood, parameters)
    selected = difference[jnp.asarray(split.training_sample_indices)]
    sample_residual = jsp.linalg.solve_triangular(
        jnp.asarray(split.training_noise_cholesky), selected, lower=True
    )
    return jnp.concatenate(
        (
            likelihood.edge_data_residual(parameters),
            likelihood.prior_residual(parameters),
            sample_residual,
        )
    )


def astigmatic_held_out_residual(
    likelihood: AstigmaticCondenserLikelihood,
    fitted_parameters: Any,
    split: AstigmaticHeldOutSplit,
) -> jnp.ndarray:
    """Return the 21 sample rows that were unavailable to fitting."""

    _verify_split(likelihood, split)
    difference = _sample_difference(likelihood, fitted_parameters)
    training = difference[jnp.asarray(split.training_sample_indices)]
    held_out = difference[jnp.asarray(split.held_out_sample_indices)]
    selected = held_out - jnp.asarray(
        split.held_out_from_training_regression
    ) @ training
    return jsp.linalg.solve_triangular(
        jnp.asarray(split.held_out_noise_cholesky), selected, lower=True
    )


def evaluate_astigmatic_held_out(
    likelihood: AstigmaticCondenserLikelihood,
    fitted_parameters: Any,
    split: AstigmaticHeldOutSplit,
    *,
    additional_predictive_covariance: Any | None = None,
) -> AstigmaticHeldOutEvaluation:
    """Evaluate test-only prediction errors without refitting any coordinate.

    ``additional_predictive_covariance`` is an optional 21-by-21 covariance in
    the *unwhitened conditional-innovation units*.  A caller can supply
    ``J_innovation @ Cov(fit) @ J_innovation.T`` plus independently validated
    model discrepancy.  Here ``J_innovation = J_H - C_HT C_TT^-1 J_T``.
    With no addition, the returned residual uses conditional measurement noise
    only.
    """

    _verify_split(likelihood, split)
    all_difference = np.asarray(
        _sample_difference(likelihood, fitted_parameters), dtype=float
    )
    difference = all_difference[split.held_out_sample_indices]
    training_difference = all_difference[split.training_sample_indices]
    innovation = difference - (
        split.held_out_from_training_regression @ training_difference
    )
    measurement_factor = np.asarray(split.held_out_noise_cholesky)
    covariance = measurement_factor @ measurement_factor.T
    if additional_predictive_covariance is not None:
        addition = np.asarray(additional_predictive_covariance, dtype=float)
        expected_shape = (split.held_out_observation_count,) * 2
        if addition.shape != expected_shape or not np.all(np.isfinite(addition)):
            raise ValueError(
                "additional_predictive_covariance must be a finite "
                f"{expected_shape} array."
            )
        addition = 0.5 * (addition + addition.T)
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(addition)))
        tolerance = 1.0e-12 * max(
            float(np.max(np.abs(np.diag(addition)), initial=0.0)), 1.0
        )
        if minimum_eigenvalue < -tolerance:
            raise ValueError(
                "additional_predictive_covariance must be positive semidefinite."
            )
        covariance = covariance + addition
    factor = np.linalg.cholesky(covariance)
    residual = np.linalg.solve(factor, innovation)
    chi_square = float(residual @ residual)
    return AstigmaticHeldOutEvaluation(
        state_indices=split.held_out_state_indices,
        acquisition_indices=split.held_out_acquisition_indices,
        residual=residual,
        raw_difference=difference,
        conditional_innovation=innovation,
        predictive_covariance=covariance,
        chi_square=chi_square,
        degrees_of_freedom=split.held_out_observation_count,
        rms_standardized_residual=float(np.sqrt(np.mean(residual**2))),
        maximum_absolute_standardized_residual=float(
            np.max(np.abs(residual), initial=0.0)
        ),
    )


def _matrix_rank(singular_values: np.ndarray, relative_tolerance: float) -> int:
    if singular_values.size == 0 or singular_values[0] == 0.0:
        return 0
    return int(
        np.sum(singular_values > relative_tolerance * singular_values[0])
    )


def audit_astigmatic_held_out_training(
    likelihood: AstigmaticCondenserLikelihood,
    split: AstigmaticHeldOutSplit,
    *,
    verify_default_contract: bool = True,
    full_relative_tolerance: float = 1.0e-7,
    nuisance_relative_tolerance: float = 1.0e-10,
    target_relative_tolerance: float = 1.0e-7,
) -> AstigmaticHeldOutTrainingAudit:
    """Recompute rank and profiled precision after the registered split."""

    _verify_split(likelihood, split)
    tolerances = (
        full_relative_tolerance,
        nuisance_relative_tolerance,
        target_relative_tolerance,
    )
    if not np.all(np.isfinite(tolerances)) or min(tolerances) <= 0.0:
        raise ValueError("Rank tolerances must be finite and positive.")
    matrix_contract, _ = build_astigmatic_design_matrix(likelihood.design)
    matrix = matrix_contract.matrix
    training = matrix[split.training_full_residual_indices]
    held_out = matrix[split.held_out_full_residual_indices]

    full_singular = np.linalg.svd(training, compute_uv=False)
    full_rank = _matrix_rank(full_singular, full_relative_tolerance)
    target = training[:, :_TARGET_COUNT]
    nuisance = training[:, _TARGET_COUNT:]
    nuisance_norms = np.linalg.norm(nuisance, axis=0)
    active = nuisance_norms > np.finfo(float).tiny
    normalized_nuisance = nuisance[:, active] / nuisance_norms[active][None, :]
    nuisance_left, nuisance_singular, _ = np.linalg.svd(
        normalized_nuisance, full_matrices=False
    )
    nuisance_rank = _matrix_rank(
        nuisance_singular, nuisance_relative_tolerance
    )
    projected_target = target - nuisance_left[:, :nuisance_rank] @ (
        nuisance_left[:, :nuisance_rank].T @ target
    )
    target_singular = np.linalg.svd(projected_target, compute_uv=False)
    target_rank = _matrix_rank(target_singular, target_relative_tolerance)
    uncertainties = (
        np.sqrt(np.diag(np.linalg.inv(projected_target.T @ projected_target)))
        if target_rank == _TARGET_COUNT
        else np.full(_TARGET_COUNT, np.inf)
    )
    maximum = float(np.max(uncertainties))
    held_out_rank = _matrix_rank(
        np.linalg.svd(held_out, compute_uv=False), full_relative_tolerance
    )

    is_default = (
        split.held_out_state_indices
        == DEFAULT_ASTIGMATIC_HELD_OUT_STATE_INDICES
    )
    verified = False
    if verify_default_contract:
        if not is_default:
            raise AstigmaticHeldOutAuditError(
                "The checksum-bound verification applies only to default states "
                "(2, 3, 7)."
            )
        expected = (
            training.shape == _EXPECTED_DEFAULT_TRAINING_SHAPE
            and full_rank == _EXPECTED_DEFAULT_FULL_RANK
            and nuisance_rank == _EXPECTED_DEFAULT_NUISANCE_RANK
            and target_rank == _EXPECTED_DEFAULT_TARGET_RANK
        )
        if not expected:
            raise AstigmaticHeldOutAuditError(
                "The default held-out training rank contract changed."
            )
        try:
            np.testing.assert_allclose(
                uncertainties,
                _EXPECTED_DEFAULT_SCALED_UNCERTAINTIES,
                rtol=6.0e-8,
                atol=2.0e-10,
            )
            np.testing.assert_allclose(
                maximum,
                _EXPECTED_DEFAULT_MAXIMUM_SCALED_UNCERTAINTY,
                rtol=6.0e-8,
                atol=2.0e-10,
            )
        except AssertionError as exc:
            raise AstigmaticHeldOutAuditError(
                "The default held-out precision contract changed."
            ) from exc
        verified = True

    return AstigmaticHeldOutTrainingAudit(
        training_matrix_shape=training.shape,
        held_out_matrix_shape=held_out.shape,
        full_rank=full_rank,
        full_nullity=_PARAMETER_COUNT - full_rank,
        nuisance_rank=nuisance_rank,
        profiled_target_rank=target_rank,
        held_out_response_rank=held_out_rank,
        target_names=tuple(matrix_contract.column_names[:_TARGET_COUNT]),
        profiled_target_singular_values=target_singular,
        scaled_target_standard_uncertainties=uncertainties,
        maximum_scaled_target_standard_uncertainty=maximum,
        conditional_precision_pass=bool(
            target_rank == _TARGET_COUNT and maximum < 0.01
        ),
        checksum_contract_verified=verified,
    )


__all__ = [
    "ASTIGMATIC_SAMPLE_OBSERVABLE_NAMES",
    "DEFAULT_ASTIGMATIC_HELD_OUT_STATE_INDICES",
    "AstigmaticHeldOutAuditError",
    "AstigmaticHeldOutEvaluation",
    "AstigmaticHeldOutLeakageError",
    "AstigmaticHeldOutSplit",
    "AstigmaticHeldOutTrainingAudit",
    "astigmatic_held_out_residual",
    "astigmatic_training_residual",
    "audit_astigmatic_held_out_training",
    "evaluate_astigmatic_held_out",
    "make_astigmatic_held_out_split",
]
