"""Observation contracts for the conditional NEOARM condenser Branch A.

The Branch-A experiment has two derived-data blocks with fixed shapes:

* twelve state-major, ten-coordinate phase-space covariance summaries; and
* three state-major by four command-major central-deflector responses, each
  ordered as ``(x, theta_x, y, theta_y)``.

Instrument observations must carry acquisition and covariance provenance.  A
vector of independent standard deviations is intentionally only a synthetic
convenience: it cannot certify an instrument data set because the ten
tomography summaries reconstructed from five images are generally correlated.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping, Sequence

import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np


EvidenceKind = Literal["instrument", "synthetic"]
CovarianceRepresentation = Literal[
    "global_covariance",
    "per_group_covariance",
    "global_cholesky",
    "per_group_cholesky",
    "diagonal_standard_deviations",
]

TOMOGRAPHY_OBSERVABLE_NAMES = (
    "log_rms_x_m",
    "log_rms_theta_x_rad",
    "log_rms_y_m",
    "log_rms_theta_y_rad",
    "correlation_x_theta_x",
    "correlation_x_y",
    "correlation_x_theta_y",
    "correlation_theta_x_y",
    "correlation_theta_x_theta_y",
    "correlation_y_theta_y",
)

DEFLECTOR_RESPONSE_OBSERVABLE_NAMES = (
    "sample_central_half_difference_x_m",
    "sample_central_half_difference_theta_x_rad",
    "sample_central_half_difference_y_m",
    "sample_central_half_difference_theta_y_rad",
)

ENTRANCE_COVARIANCE_COORDINATE_NAMES = (
    "entrance_covariance_whitened_log_0",
    "entrance_covariance_whitened_log_1",
    "entrance_covariance_whitened_log_2",
    "entrance_covariance_whitened_log_3",
    "entrance_covariance_whitened_symmetric_0_1",
    "entrance_covariance_whitened_symmetric_0_2",
    "entrance_covariance_whitened_symmetric_0_3",
    "entrance_covariance_whitened_symmetric_1_2",
    "entrance_covariance_whitened_symmetric_1_3",
    "entrance_covariance_whitened_symmetric_2_3",
)

DEFAULT_DEFLECTOR_PLANE_NAMES = ("upper", "lower")
DEFAULT_DEFLECTOR_COMMAND_NAMES = ("shift_x", "shift_y", "tilt_x", "tilt_y")
DEFAULT_OBJECTIVE_PREFIELD_PARAMETER_NAMES = (
    "Obj-prefield.z_center_m",
    "Obj-prefield.log_length_m",
    "Obj-prefield.q_bias",
)


def _immutable_array(value: object, name: str, *, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    result = array.copy()
    result.setflags(write=False)
    return result


def _nonempty(value: object, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _unique_names(values: Sequence[object], name: str, count: int) -> tuple[str, ...]:
    names = tuple(_nonempty(value, name) for value in values)
    if len(names) != count or len(set(names)) != count:
        raise ValueError(f"{name} must contain {count} unique names.")
    return names


@dataclass(frozen=True)
class ObservationProvenance:
    """Auditable origin of observations and their uncertainty model."""

    evidence_kind: EvidenceKind
    dataset_id: str
    manifest_sha256: str
    source_uri: str
    extraction_method: str
    covariance_method: str

    def __post_init__(self) -> None:
        if self.evidence_kind not in ("instrument", "synthetic"):
            raise ValueError("evidence_kind must be 'instrument' or 'synthetic'.")
        for field_name in (
            "dataset_id",
            "source_uri",
            "extraction_method",
            "covariance_method",
        ):
            object.__setattr__(self, field_name, _nonempty(getattr(self, field_name), field_name))
        if self.evidence_kind == "instrument" and self.source_uri.startswith(
            "synthetic:"
        ):
            raise ValueError("Instrument provenance cannot use a synthetic source URI.")
        digest = _nonempty(self.manifest_sha256, "manifest_sha256").lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("manifest_sha256 must be a 64-character hexadecimal digest.")
        object.__setattr__(self, "manifest_sha256", digest)


def _validate_cholesky(value: object, name: str, size: int) -> np.ndarray:
    factor = _immutable_array(value, name, ndim=2)
    if factor.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size}).")
    scale = max(float(np.max(np.abs(factor))), np.finfo(float).tiny)
    tolerance = 100.0 * np.finfo(float).eps * scale
    if not np.allclose(factor, np.tril(factor), rtol=0.0, atol=tolerance):
        raise ValueError(f"{name} must be lower triangular.")
    if np.any(np.diag(factor) <= 0.0):
        raise ValueError(f"{name} must have a strictly positive diagonal.")
    return factor


def _factor_covariance(value: object, name: str, size: int) -> np.ndarray:
    covariance = _immutable_array(value, name, ndim=2)
    if covariance.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size}).")
    scale = max(float(np.max(np.abs(covariance))), np.finfo(float).tiny)
    tolerance = 100.0 * np.finfo(float).eps * scale
    if not np.allclose(covariance, covariance.T, rtol=1.0e-10, atol=tolerance):
        raise ValueError(f"{name} must be symmetric.")
    try:
        return _immutable_array(
            np.linalg.cholesky(0.5 * (covariance + covariance.T)),
            name + " cholesky",
            ndim=2,
        )
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite.") from error


def _block_diagonal(factors: np.ndarray) -> np.ndarray:
    group_count, width, _ = factors.shape
    result = np.zeros((group_count * width, group_count * width), dtype=float)
    for index, factor in enumerate(factors):
        start = index * width
        result[start : start + width, start : start + width] = factor
    return result


@dataclass(frozen=True)
class GaussianWhitening:
    """A positive-definite Gaussian uncertainty in a declared representation."""

    cholesky: np.ndarray
    group_shape: tuple[int, int]
    representation: CovarianceRepresentation

    def __post_init__(self) -> None:
        if self.representation not in (
            "global_covariance",
            "per_group_covariance",
            "global_cholesky",
            "per_group_cholesky",
            "diagonal_standard_deviations",
        ):
            raise ValueError("Unknown covariance representation.")
        if len(self.group_shape) != 2:
            raise ValueError("group_shape must be (group_count, group_width).")
        group_count, group_width = map(int, self.group_shape)
        if group_count < 1 or group_width < 1 or (group_count, group_width) != self.group_shape:
            raise ValueError("group_shape entries must be positive integers.")
        factor = _validate_cholesky(
            self.cholesky, "cholesky", group_count * group_width
        )
        object.__setattr__(self, "cholesky", factor)
        object.__setattr__(self, "group_shape", (group_count, group_width))

    @property
    def size(self) -> int:
        return self.group_shape[0] * self.group_shape[1]

    @property
    def is_diagonal_shorthand(self) -> bool:
        return self.representation == "diagonal_standard_deviations"

    @property
    def covariance(self) -> np.ndarray:
        covariance = self.cholesky @ self.cholesky.T
        covariance.setflags(write=False)
        return covariance

    def whiten(self, difference: object) -> jnp.ndarray:
        values = jnp.asarray(difference)
        if values.size != self.size:
            raise ValueError(f"difference must contain exactly {self.size} values.")
        return jsp.linalg.solve_triangular(
            jnp.asarray(self.cholesky, dtype=values.dtype),
            values.reshape(-1),
            lower=True,
        )

    @classmethod
    def from_global_covariance(
        cls, covariance: object, *, group_shape: tuple[int, int]
    ) -> "GaussianWhitening":
        size = int(group_shape[0]) * int(group_shape[1])
        return cls(
            _factor_covariance(covariance, "covariance", size),
            group_shape,
            "global_covariance",
        )

    @classmethod
    def from_per_group_covariance(cls, covariance: object) -> "GaussianWhitening":
        values = _immutable_array(covariance, "covariance", ndim=3)
        if values.shape[1] != values.shape[2] or values.shape[0] < 1:
            raise ValueError("per-group covariance must have shape (groups, width, width).")
        factors = np.stack(
            [
                _factor_covariance(block, f"covariance[{index}]", values.shape[1])
                for index, block in enumerate(values)
            ]
        )
        return cls(
            _block_diagonal(factors),
            (values.shape[0], values.shape[1]),
            "per_group_covariance",
        )

    @classmethod
    def from_global_cholesky(
        cls, cholesky: object, *, group_shape: tuple[int, int]
    ) -> "GaussianWhitening":
        size = int(group_shape[0]) * int(group_shape[1])
        return cls(
            _validate_cholesky(cholesky, "cholesky", size),
            group_shape,
            "global_cholesky",
        )

    @classmethod
    def from_per_group_cholesky(cls, cholesky: object) -> "GaussianWhitening":
        values = _immutable_array(cholesky, "cholesky", ndim=3)
        if values.shape[1] != values.shape[2] or values.shape[0] < 1:
            raise ValueError("per-group cholesky must have shape (groups, width, width).")
        factors = np.stack(
            [
                _validate_cholesky(block, f"cholesky[{index}]", values.shape[1])
                for index, block in enumerate(values)
            ]
        )
        return cls(
            _block_diagonal(factors),
            (values.shape[0], values.shape[1]),
            "per_group_cholesky",
        )

    @classmethod
    def from_standard_deviations(
        cls, standard_deviations: object, *, group_shape: tuple[int, int]
    ) -> "GaussianWhitening":
        size = int(group_shape[0]) * int(group_shape[1])
        sigma = _immutable_array(standard_deviations, "standard_deviations")
        try:
            sigma = np.broadcast_to(sigma, group_shape).reshape(-1)
        except ValueError as error:
            raise ValueError("standard_deviations must broadcast to group_shape.") from error
        if sigma.size != size or np.any(sigma <= 0.0):
            raise ValueError("standard_deviations must be finite and strictly positive.")
        return cls(np.diag(sigma), group_shape, "diagonal_standard_deviations")


@dataclass(frozen=True)
class BranchATomographyObservations:
    """Twelve state-major ten-coordinate covariance-tomography summaries."""

    values: np.ndarray
    uncertainty: GaussianWhitening
    state_ids: tuple[str, ...] = tuple(f"tomography_state_{index}" for index in range(12))

    def __post_init__(self) -> None:
        values = _immutable_array(self.values, "values", ndim=2)
        if values.shape != (12, 10):
            raise ValueError("Branch-A tomography values must have shape (12, 10).")
        if not isinstance(self.uncertainty, GaussianWhitening):
            raise TypeError("uncertainty must be GaussianWhitening.")
        if self.uncertainty.group_shape != (12, 10):
            raise ValueError("Tomography uncertainty group_shape must be (12, 10).")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "state_ids", _unique_names(self.state_ids, "state_ids", 12))

    @property
    def observation_names(self) -> tuple[str, ...]:
        return tuple(
            f"{state_id}.{observable}"
            for state_id in self.state_ids
            for observable in TOMOGRAPHY_OBSERVABLE_NAMES
        )

    def whitened_residual(self, prediction: object) -> jnp.ndarray:
        predicted = jnp.asarray(prediction)
        if predicted.shape != (12, 10):
            raise ValueError("Tomography prediction must have shape (12, 10).")
        return self.uncertainty.whiten(predicted - jnp.asarray(self.values))


@dataclass(frozen=True)
class BranchADeflectorObservations:
    """Three states by four commands by four central-difference responses."""

    values: np.ndarray
    uncertainty: GaussianWhitening
    state_ids: tuple[str, ...] = tuple(f"deflector_state_{index}" for index in range(3))
    command_names: tuple[str, ...] = DEFAULT_DEFLECTOR_COMMAND_NAMES

    def __post_init__(self) -> None:
        values = _immutable_array(self.values, "values", ndim=3)
        if values.shape != (3, 4, 4):
            raise ValueError("Branch-A deflector values must have shape (3, 4, 4).")
        if not isinstance(self.uncertainty, GaussianWhitening):
            raise TypeError("uncertainty must be GaussianWhitening.")
        if self.uncertainty.group_shape != (12, 4):
            raise ValueError("Deflector uncertainty group_shape must be (12, 4).")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "state_ids", _unique_names(self.state_ids, "state_ids", 3))
        object.__setattr__(
            self, "command_names", _unique_names(self.command_names, "command_names", 4)
        )

    @property
    def observation_names(self) -> tuple[str, ...]:
        return tuple(
            f"{state_id}.{command}.{observable}"
            for state_id in self.state_ids
            for command in self.command_names
            for observable in DEFLECTOR_RESPONSE_OBSERVABLE_NAMES
        )

    def whitened_residual(self, prediction: object) -> jnp.ndarray:
        predicted = jnp.asarray(prediction)
        if predicted.shape != (3, 4, 4):
            raise ValueError("Deflector prediction must have shape (3, 4, 4).")
        return self.uncertainty.whiten(predicted - jnp.asarray(self.values))


@dataclass(frozen=True)
class ObjectivePrefieldAnchorObservation:
    """Independent Gaussian observation or prior on Obj-prefield parameters."""

    values: np.ndarray
    uncertainty: GaussianWhitening
    provenance: ObservationProvenance
    parameter_names: tuple[str, ...] = DEFAULT_OBJECTIVE_PREFIELD_PARAMETER_NAMES

    def __post_init__(self) -> None:
        values = _immutable_array(self.values, "values", ndim=1)
        if values.size < 1:
            raise ValueError("Obj-prefield anchor values must be non-empty.")
        if not isinstance(self.provenance, ObservationProvenance):
            raise ValueError("Obj-prefield anchor provenance is required.")
        if not isinstance(self.uncertainty, GaussianWhitening):
            raise TypeError("uncertainty must be GaussianWhitening.")
        if self.uncertainty.group_shape != (1, values.size):
            raise ValueError("Anchor uncertainty group_shape must be (1, n_parameters).")
        names = _unique_names(self.parameter_names, "parameter_names", values.size)
        if not all(name.startswith("Obj-prefield.") for name in names):
            raise ValueError("Every anchor parameter must belong to Obj-prefield.")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "parameter_names", names)

    def whitened_residual(self, prediction: object) -> jnp.ndarray:
        predicted = jnp.asarray(prediction)
        if predicted.shape != self.values.shape:
            raise ValueError(f"Anchor prediction must have shape {self.values.shape}.")
        return self.uncertainty.whiten(predicted - jnp.asarray(self.values))


@dataclass(frozen=True)
class OrderedParameterBlock:
    name: str
    start: int
    stop: int
    parameter_names: tuple[str, ...]

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)


@dataclass(frozen=True)
class BranchAParameterOrder:
    """Exact global parameter order shared by Branch A and downstream data."""

    parameter_names: tuple[str, ...]
    blocks: tuple[OrderedParameterBlock, ...]

    @property
    def block_slices(self) -> Mapping[str, slice]:
        return MappingProxyType({block.name: block.slice for block in self.blocks})


def deflector_gain_parameter_names(
    plane_names: Sequence[str] = DEFAULT_DEFLECTOR_PLANE_NAMES,
    command_names: Sequence[str] = DEFAULT_DEFLECTOR_COMMAND_NAMES,
) -> tuple[str, ...]:
    """Return the C-order ``(plane, axis, command)`` sixteen-gain layout."""

    planes = _unique_names(plane_names, "plane_names", 2)
    commands = _unique_names(command_names, "command_names", 4)
    return tuple(
        f"{plane}.kick_gain_{axis}_rad_per_{command}"
        for plane in planes
        for axis in ("x", "y")
        for command in commands
    )


def branch_a_parameter_order(
    optical_parameter_names: Sequence[str],
    calibration_latent_names: Sequence[str] = (),
    discrepancy_latent_names: Sequence[str] = (),
    *,
    prefix: str = "actual_hardware",
) -> BranchAParameterOrder:
    """Build the notebook-compatible optical/calibration/covariance/gain order."""

    prefix = _nonempty(prefix, "prefix")
    groups = (
        ("optical", tuple(map(str, optical_parameter_names))),
        ("calibration", tuple(map(str, calibration_latent_names))),
        ("discrepancy", tuple(map(str, discrepancy_latent_names))),
        (
            "entrance_covariance",
            tuple(f"{prefix}.entrance.{name}" for name in ENTRANCE_COVARIANCE_COORDINATE_NAMES),
        ),
        (
            "deflector_gain",
            tuple(f"{prefix}.{name}" for name in deflector_gain_parameter_names()),
        ),
    )
    flattened = tuple(name for _, names in groups for name in names)
    if any(not name.strip() for name in flattened) or len(flattened) != len(set(flattened)):
        raise ValueError("All joint parameter names must be non-empty and globally unique.")
    blocks = []
    cursor = 0
    for name, names in groups:
        blocks.append(OrderedParameterBlock(name, cursor, cursor + len(names), names))
        cursor += len(names)
    return BranchAParameterOrder(flattened, tuple(blocks))


@dataclass(frozen=True)
class BranchAObservationDataset:
    """Provenance-bearing Branch-A data and JAX residual assembly."""

    tomography: BranchATomographyObservations
    deflector: BranchADeflectorObservations
    provenance: ObservationProvenance | None
    objective_prefield_anchor: ObjectivePrefieldAnchorObservation | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tomography, BranchATomographyObservations):
            raise TypeError("tomography must be BranchATomographyObservations.")
        if not isinstance(self.deflector, BranchADeflectorObservations):
            raise TypeError("deflector must be BranchADeflectorObservations.")
        if not isinstance(self.provenance, ObservationProvenance):
            raise ValueError("Branch-A observation provenance is required.")
        if self.provenance.evidence_kind == "instrument":
            if self.tomography.uncertainty.is_diagonal_shorthand:
                raise ValueError(
                    "Instrument tomography cannot be certified from diagonal-only "
                    "standard-deviation shorthand; provide per-state or global "
                    "covariance/Cholesky."
                )
            if self.deflector.uncertainty.is_diagonal_shorthand:
                raise ValueError(
                    "Instrument deflector data cannot be certified from diagonal-only "
                    "standard-deviation shorthand; provide block or global covariance/Cholesky."
                )
        anchor = self.objective_prefield_anchor
        if anchor is not None:
            if not isinstance(anchor, ObjectivePrefieldAnchorObservation):
                raise TypeError("objective_prefield_anchor has the wrong type.")
            if anchor.provenance.dataset_id == self.provenance.dataset_id:
                raise ValueError(
                    "The Obj-prefield anchor must have independent provenance, not the "
                    "same dataset_id as the Branch-A observations."
                )
            if (
                self.provenance.evidence_kind == "instrument"
                and anchor.provenance.evidence_kind != "instrument"
            ):
                raise ValueError("An instrument Branch-A data set requires an instrument anchor.")

    @property
    def evidence_kind(self) -> EvidenceKind:
        return self.provenance.evidence_kind

    @property
    def instrument_certified(self) -> bool:
        """Whether the two Branch-A data blocks pass instrument-data checks."""

        return self.evidence_kind == "instrument"

    @property
    def conditional_route_complete(self) -> bool:
        """Whether instrument data also carry the independent Branch-A anchor."""

        return self.instrument_certified and self.objective_prefield_anchor is not None

    @property
    def observation_names(self) -> tuple[str, ...]:
        names = self.tomography.observation_names + self.deflector.observation_names
        if self.objective_prefield_anchor is not None:
            names += tuple(
                f"objective_prefield_anchor.{name}"
                for name in self.objective_prefield_anchor.parameter_names
            )
        return names

    @property
    def residual_block_slices(self) -> Mapping[str, slice]:
        tomography_stop = len(self.tomography.observation_names)
        deflector_stop = tomography_stop + len(self.deflector.observation_names)
        result = {
            "tomography": slice(0, tomography_stop),
            "deflector": slice(tomography_stop, deflector_stop),
        }
        if self.objective_prefield_anchor is not None:
            result["objective_prefield_anchor"] = slice(
                deflector_stop, deflector_stop + self.objective_prefield_anchor.values.size
            )
        return MappingProxyType(result)

    def whitened_residual(
        self,
        tomography_prediction: object,
        deflector_prediction: object,
        objective_prefield_prediction: object | None = None,
    ) -> jnp.ndarray:
        parts = [
            self.tomography.whitened_residual(tomography_prediction),
            self.deflector.whitened_residual(deflector_prediction),
        ]
        if self.objective_prefield_anchor is None:
            if objective_prefield_prediction is not None:
                raise ValueError("No Obj-prefield anchor is present in this data set.")
        else:
            if objective_prefield_prediction is None:
                raise ValueError("objective_prefield_prediction is required by the anchor.")
            parts.append(
                self.objective_prefield_anchor.whitened_residual(
                    objective_prefield_prediction
                )
            )
        return jnp.concatenate(tuple(parts))

    @classmethod
    def synthetic_from_predictions(
        cls,
        tomography_prediction: object,
        tomography_uncertainty: GaussianWhitening,
        deflector_prediction: object,
        deflector_uncertainty: GaussianWhitening,
        *,
        seed: int,
        simulation_id: str,
        objective_prefield_prediction: object | None = None,
        objective_prefield_uncertainty: GaussianWhitening | None = None,
        objective_prefield_parameter_names: Sequence[str] = (
            DEFAULT_OBJECTIVE_PREFIELD_PARAMETER_NAMES
        ),
    ) -> "BranchAObservationDataset":
        """Draw a deterministic matched-model data set labelled ``synthetic``."""

        simulation_id = _nonempty(simulation_id, "simulation_id")
        tomography_mean = _immutable_array(
            tomography_prediction, "tomography_prediction", ndim=2
        )
        deflector_mean = _immutable_array(
            deflector_prediction, "deflector_prediction", ndim=3
        )
        if tomography_mean.shape != (12, 10) or deflector_mean.shape != (3, 4, 4):
            raise ValueError(
                "Synthetic predictions must have shapes (12, 10) and (3, 4, 4)."
            )
        if tomography_uncertainty.group_shape != (12, 10):
            raise ValueError("tomography_uncertainty must have group_shape (12, 10).")
        if deflector_uncertainty.group_shape != (12, 4):
            raise ValueError("deflector_uncertainty must have group_shape (12, 4).")
        rng = np.random.default_rng(seed)
        tomography_values = (
            tomography_mean.reshape(-1)
            + tomography_uncertainty.cholesky @ rng.standard_normal(120)
        )
        deflector_values = (
            deflector_mean.reshape(-1)
            + deflector_uncertainty.cholesky @ rng.standard_normal(48)
        )
        digest = hashlib.sha256()
        digest.update(simulation_id.encode("utf-8"))
        for array in (
            tomography_mean,
            tomography_uncertainty.cholesky,
            deflector_mean,
            deflector_uncertainty.cholesky,
        ):
            digest.update(np.asarray(array, dtype=np.float64).tobytes(order="C"))
        digest.update(str(int(seed)).encode("ascii"))
        provenance = ObservationProvenance(
            evidence_kind="synthetic",
            dataset_id=simulation_id,
            manifest_sha256=digest.hexdigest(),
            source_uri=f"synthetic://neoarm-branch-a/{simulation_id}",
            extraction_method="matched-model predictions plus seeded Gaussian draw",
            covariance_method="declared synthetic Gaussian covariance",
        )
        anchor = None
        if (objective_prefield_prediction is None) != (
            objective_prefield_uncertainty is None
        ):
            raise ValueError(
                "Provide both objective_prefield_prediction and its uncertainty, or neither."
            )
        if objective_prefield_prediction is not None:
            anchor_mean = _immutable_array(
                objective_prefield_prediction, "objective_prefield_prediction", ndim=1
            )
            if objective_prefield_uncertainty.group_shape != (1, anchor_mean.size):
                raise ValueError(
                    "objective_prefield_uncertainty has the wrong group_shape."
                )
            anchor_values = (
                anchor_mean
                + objective_prefield_uncertainty.cholesky
                @ rng.standard_normal(anchor_mean.size)
            )
            anchor_hasher = hashlib.sha256(
                digest.digest() + b"independent-objective-prefield-anchor"
            )
            anchor_hasher.update(
                np.asarray(anchor_mean, dtype=np.float64).tobytes(order="C")
            )
            anchor_hasher.update(
                np.asarray(
                    objective_prefield_uncertainty.cholesky, dtype=np.float64
                ).tobytes(order="C")
            )
            anchor_digest = anchor_hasher.hexdigest()
            anchor = ObjectivePrefieldAnchorObservation(
                anchor_values,
                objective_prefield_uncertainty,
                ObservationProvenance(
                    evidence_kind="synthetic",
                    dataset_id=simulation_id + ":independent-objective-prefield-anchor",
                    manifest_sha256=anchor_digest,
                    source_uri=(
                        f"synthetic://neoarm-branch-a/{simulation_id}/"
                        "objective-prefield-anchor"
                    ),
                    extraction_method="independent seeded Gaussian anchor draw",
                    covariance_method="declared synthetic Gaussian covariance",
                ),
                tuple(objective_prefield_parameter_names),
            )
        return cls(
            BranchATomographyObservations(
                tomography_values.reshape(12, 10), tomography_uncertainty
            ),
            BranchADeflectorObservations(
                deflector_values.reshape(3, 4, 4), deflector_uncertainty
            ),
            provenance,
            anchor,
        )


__all__ = [
    "BranchADeflectorObservations",
    "BranchAObservationDataset",
    "BranchAParameterOrder",
    "BranchATomographyObservations",
    "CovarianceRepresentation",
    "DEFAULT_DEFLECTOR_COMMAND_NAMES",
    "DEFAULT_DEFLECTOR_PLANE_NAMES",
    "DEFAULT_OBJECTIVE_PREFIELD_PARAMETER_NAMES",
    "DEFLECTOR_RESPONSE_OBSERVABLE_NAMES",
    "ENTRANCE_COVARIANCE_COORDINATE_NAMES",
    "EvidenceKind",
    "GaussianWhitening",
    "ObjectivePrefieldAnchorObservation",
    "ObservationProvenance",
    "OrderedParameterBlock",
    "TOMOGRAPHY_OBSERVABLE_NAMES",
    "branch_a_parameter_order",
    "deflector_gain_parameter_names",
]
