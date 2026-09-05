"""Bind the no-tilt astigmatic condenser likelihood to a full-column fit.

The checksum-bound :class:`~temgym_core.condenser_astigmatic_likelihood.AstigmaticCondenserLikelihood`
uses scaled local coordinates for the twelve CL1/CL3/C-mini parameters and
four source/objective-prefield parameters.  A full-column inverse problem,
however, must estimate those same quantities only once, in the physical
coordinate vector consumed by :class:`HardEdgeColumnParameterization`.

This module supplies that exact change of coordinates.  The 36 physical
column parameters stay at the front of the joint vector and the 27 genuinely
astigmatic-acquisition nuisance coordinates follow them.  In particular, the
adapter contains no commanded beam tilt, deflector response, or K16 map.
Twenty downstream physical parameters are intentionally invisible to the
condenser block; a downstream image/diffraction likelihood is responsible for
them in the combined fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .condenser_astigmatic_likelihood import (
    AstigmaticCondenserLikelihood,
    AstigmaticCondenserPrediction,
)
from .neoarm_calibration import HardEdgeColumnParameterization


_NATIVE_PARAMETER_COUNT = 43
_TARGET_NATIVE_SLICE = slice(0, 12)
_ENTRANCE_NATIVE_SLICE = slice(12, 22)
_UPSTREAM_NATIVE_SLICE = slice(22, 26)
_ACQUISITION_NATIVE_SLICE = slice(26, 43)
_OUTER_NATIVE_INDICES = tuple(range(12, 22)) + tuple(range(26, 43))
_OUTER_PARAMETER_COUNT = len(_OUTER_NATIVE_INDICES)


def _immutable_array(
    value: Any,
    name: str,
    *,
    shape: tuple[int, ...],
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    result = array.copy()
    result.setflags(write=False)
    return result


def _validate_column_identity(
    shared: HardEdgeColumnParameterization,
    bound: HardEdgeColumnParameterization,
) -> None:
    """Reject a silently different full-column coordinate convention."""

    if not isinstance(shared, HardEdgeColumnParameterization):
        raise TypeError("shared_column must be a HardEdgeColumnParameterization.")
    if tuple(shared.parameter_names) != tuple(bound.parameter_names):
        raise ValueError(
            "The shared full-column parameter order differs from the "
            "astigmatic likelihood's checksum-bound column."
        )
    if tuple(shared.channel_names) != tuple(bound.channel_names):
        raise ValueError(
            "The shared full-column raw-control order differs from the "
            "astigmatic likelihood's checksum-bound column."
        )
    if not np.array_equal(
        np.asarray(shared.nominal_parameters),
        np.asarray(bound.nominal_parameters),
    ):
        raise ValueError(
            "The shared full-column nominal physical gauge differs from the "
            "astigmatic likelihood's checksum-bound gauge."
        )
    if float(shared.sample_z_m) != float(bound.sample_z_m):
        raise ValueError(
            "The shared full-column sample plane differs from the "
            "astigmatic likelihood's checksum-bound sample plane."
        )
    if len(shared.lenses) != len(bound.lenses):
        raise ValueError("The shared full-column lens topology differs.")
    for shared_lens, bound_lens in zip(shared.lenses, bound.lenses, strict=True):
        shared_identity = (
            shared_lens.name,
            shared_lens.channel_index,
            float(shared_lens.reference_dac),
        )
        bound_identity = (
            bound_lens.name,
            bound_lens.channel_index,
            float(bound_lens.reference_dac),
        )
        if shared_identity != bound_identity:
            raise ValueError(
                "The shared full-column lens/control topology differs at "
                f"{bound_lens.name!r}."
            )


@dataclass(frozen=True)
class AstigmaticOuterParameterContract:
    """Finite optimizer contract for the 27 acquisition-only coordinates.

    The stored units are the native likelihood units.  Edge-centre coordinates
    are micrometres; log quantities and entrance-covariance chart coordinates
    are dimensionless; the final tag orientation is radians and pi-periodic.
    Bounds are deliberately finite so a global fit cannot wander into an
    undeclared acquisition model.
    """

    names: tuple[str, ...]
    native_parameter_indices: tuple[int, ...]
    reference: np.ndarray
    scales: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        names = tuple(str(value).strip() for value in self.names)
        indices = tuple(int(value) for value in self.native_parameter_indices)
        if len(names) != _OUTER_PARAMETER_COUNT or len(set(names)) != len(names):
            raise ValueError("Expected 27 unique astigmatic outer-parameter names.")
        if any(not value for value in names):
            raise ValueError("Astigmatic outer-parameter names must be non-empty.")
        if indices != _OUTER_NATIVE_INDICES:
            raise ValueError(
                "Astigmatic outer coordinates must retain the exact native "
                "12:22 then 26:43 ordering."
            )
        if any(
            "k16" in name.lower()
            or "deflector" in name.lower()
            or "tilt_command" in name.lower()
            for name in names
        ):
            raise ValueError(
                "The no-commanded-tilt adapter cannot contain K16, deflector, "
                "or commanded-tilt coordinates."
            )
        arrays = {
            field: _immutable_array(
                getattr(self, field), field, shape=(_OUTER_PARAMETER_COUNT,)
            )
            for field in ("reference", "scales", "lower", "upper")
        }
        if np.any(arrays["scales"] <= 0.0):
            raise ValueError("Every astigmatic outer-parameter scale must be positive.")
        if np.any(arrays["lower"] >= arrays["upper"]):
            raise ValueError("Every astigmatic outer bound must have lower < upper.")
        if np.any(arrays["reference"] < arrays["lower"]) or np.any(
            arrays["reference"] > arrays["upper"]
        ):
            raise ValueError("The astigmatic outer reference must lie within bounds.")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "native_parameter_indices", indices)
        for field, array in arrays.items():
            object.__setattr__(self, field, array)

    @property
    def size(self) -> int:
        return _OUTER_PARAMETER_COUNT

    @property
    def block_slices(self) -> Mapping[str, slice]:
        return MappingProxyType(
            {
                "entrance_covariance": slice(0, 10),
                "aperture_and_stage": slice(10, 12),
                "edge_local": slice(12, 24),
                "sample_scales": slice(24, 26),
                "astigmatic_tag_orientation": slice(26, 27),
            }
        )


def default_astigmatic_outer_parameter_contract(
    likelihood: AstigmaticCondenserLikelihood,
    *,
    entrance_coordinate_bound: float = 0.35,
    edge_centre_bound_um: float = 5.0,
    edge_log_dose_bound: float = 0.5,
    prior_sigma_bound: float = 5.0,
    tag_orientation_bound_rad: float = 0.5 * np.pi,
    tag_orientation_scale_rad: float = 0.1,
) -> AstigmaticOuterParameterContract:
    """Build the declared finite nuisance box for full-column optimization."""

    if not isinstance(likelihood, AstigmaticCondenserLikelihood):
        raise TypeError("likelihood must be an AstigmaticCondenserLikelihood.")
    positive = (
        entrance_coordinate_bound,
        edge_centre_bound_um,
        edge_log_dose_bound,
        prior_sigma_bound,
        tag_orientation_bound_rad,
        tag_orientation_scale_rad,
    )
    if not np.all(np.isfinite(positive)) or min(positive) <= 0.0:
        raise ValueError("All outer-contract widths and scales must be positive.")

    branch = likelihood.branch_experiment
    design = likelihood.design
    native_indices = _OUTER_NATIVE_INDICES
    names = tuple(likelihood.parameter_names[index] for index in native_indices)
    reference = np.asarray(likelihood.nominal_parameters)[list(native_indices)]

    scales = np.concatenate(
        (
            np.full(10, 0.1),
            (
                branch.aperture_log_radius_prior_sigma,
                branch.aperture_stage_log_scale_prior_sigma,
            ),
            np.tile((1.0, 1.0, 0.1), 4),
            (
                design.realspace_log_scale_prior_sigma,
                design.angular_log_scale_prior_sigma,
            ),
            (tag_orientation_scale_rad,),
        )
    )
    half_width = np.concatenate(
        (
            np.full(10, entrance_coordinate_bound),
            prior_sigma_bound
            * np.asarray(
                (
                    branch.aperture_log_radius_prior_sigma,
                    branch.aperture_stage_log_scale_prior_sigma,
                )
            ),
            np.tile(
                (edge_centre_bound_um, edge_centre_bound_um, edge_log_dose_bound),
                4,
            ),
            prior_sigma_bound
            * np.asarray(
                (
                    design.realspace_log_scale_prior_sigma,
                    design.angular_log_scale_prior_sigma,
                )
            ),
            (tag_orientation_bound_rad,),
        )
    )
    return AstigmaticOuterParameterContract(
        names=names,
        native_parameter_indices=native_indices,
        reference=reference,
        scales=scales,
        lower=reference - half_width,
        upper=reference + half_width,
    )


@dataclass(frozen=True)
class AstigmaticFullColumnRankDiagnostics:
    """Structural ranks of the adapter's local whitened Jacobian."""

    joint_rank: int
    joint_nullity: int
    outer_rank: int
    profiled_shared_physical_rank: int
    profiled_condenser_target_rank: int
    shared_physical_parameter_names: tuple[str, ...]
    unobserved_physical_parameter_names: tuple[str, ...]
    joint_singular_values: np.ndarray
    outer_singular_values: np.ndarray
    profiled_shared_physical_singular_values: np.ndarray
    profiled_condenser_target_singular_values: np.ndarray
    relative_tolerance: float
    nuisance_relative_tolerance: float

    def __post_init__(self) -> None:
        counts = (
            self.joint_rank,
            self.joint_nullity,
            self.outer_rank,
            self.profiled_shared_physical_rank,
            self.profiled_condenser_target_rank,
        )
        if any(int(value) != value or int(value) < 0 for value in counts):
            raise ValueError("Rank counts must be non-negative integers.")
        for name in (
            "joint_singular_values",
            "outer_singular_values",
            "profiled_shared_physical_singular_values",
            "profiled_condenser_target_singular_values",
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.ndim != 1 or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite one-dimensional array.")
            value = value.copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        tolerances = (self.relative_tolerance, self.nuisance_relative_tolerance)
        if not np.all(np.isfinite(tolerances)) or min(tolerances) <= 0.0:
            raise ValueError("Rank tolerances must be finite and positive.")


def _equilibrated_singular_values(
    matrix: np.ndarray,
    *,
    active_norm_rtol: float = 1.0e-12,
) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("Rank analysis requires a finite matrix.")
    norms = np.linalg.norm(values, axis=0)
    largest = float(np.max(norms, initial=0.0))
    active = norms > active_norm_rtol * largest if largest else np.zeros_like(norms, bool)
    if not np.any(active):
        return np.empty(0, dtype=float)
    return np.linalg.svd(values[:, active] / norms[active][None, :], compute_uv=False)


def _rank(singular_values: np.ndarray, relative_tolerance: float) -> int:
    singular = np.asarray(singular_values)
    return (
        0
        if singular.size == 0 or singular[0] == 0.0
        else int(np.sum(singular > relative_tolerance * singular[0]))
    )


def _profile_columns(
    targets: np.ndarray,
    nuisance: np.ndarray,
    *,
    nuisance_relative_tolerance: float,
) -> tuple[np.ndarray, int, np.ndarray]:
    nuisance_values = np.asarray(nuisance, dtype=float)
    norms = np.linalg.norm(nuisance_values, axis=0)
    largest = float(np.max(norms, initial=0.0))
    active = norms > 1.0e-12 * largest if largest else np.zeros_like(norms, bool)
    if not np.any(active):
        return np.asarray(targets, dtype=float), 0, np.empty(0)
    left, singular, _ = np.linalg.svd(
        nuisance_values[:, active] / norms[active][None, :],
        full_matrices=False,
    )
    nuisance_rank = _rank(singular, nuisance_relative_tolerance)
    resolved_left = left[:, :nuisance_rank]
    target_values = np.asarray(targets, dtype=float)
    profiled = target_values - resolved_left @ (resolved_left.T @ target_values)
    return profiled, nuisance_rank, singular


@dataclass(frozen=True)
class AstigmaticFullColumnAdapter:
    """JAX-compatible physical-column plus 27-nuisance likelihood adapter."""

    likelihood: AstigmaticCondenserLikelihood
    shared_column: HardEdgeColumnParameterization
    outer_contract: AstigmaticOuterParameterContract

    def __post_init__(self) -> None:
        if not isinstance(self.likelihood, AstigmaticCondenserLikelihood):
            raise TypeError("likelihood must be an AstigmaticCondenserLikelihood.")
        if not isinstance(self.outer_contract, AstigmaticOuterParameterContract):
            raise TypeError("outer_contract must be an AstigmaticOuterParameterContract.")
        _validate_column_identity(
            self.shared_column,
            self.likelihood.branch_experiment.column,
        )
        expected_names = tuple(
            self.likelihood.parameter_names[index]
            for index in self.outer_contract.native_parameter_indices
        )
        if self.outer_contract.names != expected_names:
            raise ValueError(
                "outer_contract names do not match the likelihood's native ordering."
            )

    @classmethod
    def from_likelihood(
        cls,
        likelihood: AstigmaticCondenserLikelihood,
        shared_column: HardEdgeColumnParameterization | None = None,
        *,
        outer_contract: AstigmaticOuterParameterContract | None = None,
    ) -> "AstigmaticFullColumnAdapter":
        """Construct an adapter and verify the shared physical-column identity."""

        if not isinstance(likelihood, AstigmaticCondenserLikelihood):
            raise TypeError("likelihood must be an AstigmaticCondenserLikelihood.")
        selected_column = (
            likelihood.branch_experiment.column
            if shared_column is None
            else shared_column
        )
        selected_contract = (
            default_astigmatic_outer_parameter_contract(likelihood)
            if outer_contract is None
            else outer_contract
        )
        return cls(likelihood, selected_column, selected_contract)

    @property
    def optical_parameter_count(self) -> int:
        return self.shared_column.n_parameters

    @property
    def outer_parameter_count(self) -> int:
        return self.outer_contract.size

    @property
    def joint_parameter_count(self) -> int:
        return self.optical_parameter_count + self.outer_parameter_count

    @property
    def target_physical_indices(self) -> tuple[int, ...]:
        return tuple(
            int(value) for value in self.likelihood.branch_experiment.target_parameter_indices
        )

    @property
    def upstream_physical_indices(self) -> tuple[int, ...]:
        return tuple(
            int(value)
            for value in self.likelihood.branch_experiment.upstream_parameter_indices
        )

    @property
    def shared_physical_indices(self) -> tuple[int, ...]:
        return self.target_physical_indices + self.upstream_physical_indices

    @property
    def unobserved_physical_indices(self) -> tuple[int, ...]:
        shared = set(self.shared_physical_indices)
        return tuple(
            index for index in range(self.optical_parameter_count) if index not in shared
        )

    @property
    def joint_parameter_names(self) -> tuple[str, ...]:
        return tuple(self.shared_column.parameter_names) + self.outer_contract.names

    @property
    def joint_reference(self) -> np.ndarray:
        result = np.concatenate(
            (
                np.asarray(self.shared_column.nominal_parameters),
                self.outer_contract.reference,
            )
        )
        result.setflags(write=False)
        return result

    def joint_scales(self, optical_parameter_scales: Sequence[float]) -> np.ndarray:
        optical = _immutable_array(
            optical_parameter_scales,
            "optical_parameter_scales",
            shape=(self.optical_parameter_count,),
        )
        if np.any(optical <= 0.0):
            raise ValueError("Every optical parameter scale must be positive.")
        result = np.concatenate((optical, self.outer_contract.scales))
        result.setflags(write=False)
        return result

    def joint_bounds(
        self,
        optical_bounds: tuple[Sequence[float], Sequence[float]],
    ) -> tuple[np.ndarray, np.ndarray]:
        optical_lower = _immutable_array(
            optical_bounds[0],
            "optical_lower",
            shape=(self.optical_parameter_count,),
        )
        optical_upper = _immutable_array(
            optical_bounds[1],
            "optical_upper",
            shape=(self.optical_parameter_count,),
        )
        if np.any(optical_lower >= optical_upper):
            raise ValueError("Every optical bound must have lower < upper.")
        lower = np.concatenate((optical_lower, self.outer_contract.lower))
        upper = np.concatenate((optical_upper, self.outer_contract.upper))
        lower.setflags(write=False)
        upper.setflags(write=False)
        return lower, upper

    def split_joint_parameters(
        self, joint_parameters: Any
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        values = jnp.asarray(joint_parameters)
        if values.shape != (self.joint_parameter_count,):
            raise ValueError(
                f"joint_parameters must have shape ({self.joint_parameter_count},)."
            )
        return (
            values[: self.optical_parameter_count],
            values[self.optical_parameter_count :],
        )

    def join_parameters(
        self,
        physical_parameters: Any,
        outer_parameters: Any | None = None,
    ) -> jnp.ndarray:
        physical = jnp.asarray(physical_parameters)
        if physical.shape != (self.optical_parameter_count,):
            raise ValueError(
                f"physical_parameters must have shape ({self.optical_parameter_count},)."
            )
        outer = (
            jnp.asarray(self.outer_contract.reference, dtype=physical.dtype)
            if outer_parameters is None
            else jnp.asarray(outer_parameters)
        )
        if outer.shape != (self.outer_parameter_count,):
            raise ValueError(
                f"outer_parameters must have shape ({self.outer_parameter_count},)."
            )
        return jnp.concatenate((physical, outer))

    def likelihood_parameters_from_parts(
        self,
        physical_parameters: Any,
        outer_parameters: Any,
    ) -> jnp.ndarray:
        """Map one physical vector and 27 outer coordinates to native order."""

        physical = jnp.asarray(physical_parameters)
        outer = jnp.asarray(outer_parameters)
        if physical.shape != (self.optical_parameter_count,):
            raise ValueError(
                f"physical_parameters must have shape ({self.optical_parameter_count},)."
            )
        if outer.shape != (self.outer_parameter_count,):
            raise ValueError(
                f"outer_parameters must have shape ({self.outer_parameter_count},)."
            )
        branch = self.likelihood.branch_experiment
        values = jnp.asarray(self.likelihood.nominal_parameters, dtype=physical.dtype)
        target_indices = jnp.asarray(branch.target_parameter_indices)
        upstream_indices = jnp.asarray(branch.upstream_parameter_indices)
        bound_nominal = jnp.asarray(branch.column.nominal_parameters, dtype=physical.dtype)
        values = values.at[_TARGET_NATIVE_SLICE].set(
            (physical[target_indices] - bound_nominal[target_indices])
            / jnp.asarray(branch.target_parameter_scales, dtype=physical.dtype)
        )
        values = values.at[_UPSTREAM_NATIVE_SLICE].set(
            (physical[upstream_indices] - bound_nominal[upstream_indices])
            / jnp.asarray(branch.upstream_parameter_scales, dtype=physical.dtype)
        )
        return values.at[jnp.asarray(_OUTER_NATIVE_INDICES)].set(
            outer.astype(physical.dtype)
        )

    def likelihood_parameters(self, joint_parameters: Any) -> jnp.ndarray:
        physical, outer = self.split_joint_parameters(joint_parameters)
        return self.likelihood_parameters_from_parts(physical, outer)

    def predict_from_parts(
        self,
        physical_parameters: Any,
        outer_parameters: Any,
    ) -> AstigmaticCondenserPrediction:
        return self.likelihood.predict(
            self.likelihood_parameters_from_parts(physical_parameters, outer_parameters)
        )

    def residual_from_parts(
        self,
        physical_parameters: Any,
        outer_parameters: Any,
    ) -> jnp.ndarray:
        return self.likelihood.residual(
            self.likelihood_parameters_from_parts(physical_parameters, outer_parameters)
        )

    def data_residual_from_parts(
        self,
        physical_parameters: Any,
        outer_parameters: Any,
    ) -> jnp.ndarray:
        return self.likelihood.data_residual(
            self.likelihood_parameters_from_parts(physical_parameters, outer_parameters)
        )

    def prior_residual_from_parts(
        self,
        physical_parameters: Any,
        outer_parameters: Any,
    ) -> jnp.ndarray:
        return self.likelihood.prior_residual(
            self.likelihood_parameters_from_parts(physical_parameters, outer_parameters)
        )

    def residual(self, joint_parameters: Any) -> jnp.ndarray:
        physical, outer = self.split_joint_parameters(joint_parameters)
        return self.residual_from_parts(physical, outer)

    def jacobian(self, joint_parameters: Any) -> jnp.ndarray:
        return jax.jacfwd(self.residual)(jnp.asarray(joint_parameters))

    def coordinate_jacobian(self, *, dtype: Any = jnp.float64) -> jnp.ndarray:
        """Return d(native 43)/d(full-column 36 plus outer 27) analytically."""

        branch = self.likelihood.branch_experiment
        matrix = jnp.zeros(
            (_NATIVE_PARAMETER_COUNT, self.joint_parameter_count), dtype=dtype
        )
        matrix = matrix.at[
            jnp.arange(_TARGET_NATIVE_SLICE.start, _TARGET_NATIVE_SLICE.stop),
            jnp.asarray(branch.target_parameter_indices),
        ].set(1.0 / jnp.asarray(branch.target_parameter_scales, dtype=dtype))
        matrix = matrix.at[
            jnp.arange(_UPSTREAM_NATIVE_SLICE.start, _UPSTREAM_NATIVE_SLICE.stop),
            jnp.asarray(branch.upstream_parameter_indices),
        ].set(1.0 / jnp.asarray(branch.upstream_parameter_scales, dtype=dtype))
        outer_columns = self.optical_parameter_count + jnp.arange(
            self.outer_parameter_count
        )
        return matrix.at[
            jnp.asarray(_OUTER_NATIVE_INDICES), outer_columns
        ].set(1.0)

    def local_rank_diagnostics(
        self,
        joint_parameters: Any | None = None,
        *,
        relative_tolerance: float = 1.0e-7,
        nuisance_relative_tolerance: float = 1.0e-10,
    ) -> AstigmaticFullColumnRankDiagnostics:
        """Report the adapter block's joint and nuisance-profiled ranks."""

        tolerances = (relative_tolerance, nuisance_relative_tolerance)
        if not np.all(np.isfinite(tolerances)) or min(tolerances) <= 0.0:
            raise ValueError("Rank tolerances must be finite and positive.")
        point = self.joint_reference if joint_parameters is None else joint_parameters
        jacobian = np.asarray(self.jacobian(jnp.asarray(point)), dtype=float)
        if not np.all(np.isfinite(jacobian)):
            raise FloatingPointError("The astigmatic adapter Jacobian is not finite.")

        physical_count = self.optical_parameter_count
        outer = jacobian[:, physical_count:]
        joint_singular = _equilibrated_singular_values(jacobian)
        outer_singular = _equilibrated_singular_values(outer)
        outer_rank = _rank(outer_singular, nuisance_relative_tolerance)

        shared_indices = np.asarray(self.shared_physical_indices, dtype=int)
        shared = jacobian[:, shared_indices]
        profiled_shared, _, _ = _profile_columns(
            shared,
            outer,
            nuisance_relative_tolerance=nuisance_relative_tolerance,
        )
        profiled_shared_singular = _equilibrated_singular_values(profiled_shared)

        target_indices = np.asarray(self.target_physical_indices, dtype=int)
        upstream_indices = np.asarray(self.upstream_physical_indices, dtype=int)
        target = jacobian[:, target_indices]
        target_nuisance = np.concatenate(
            (outer, jacobian[:, upstream_indices]), axis=1
        )
        profiled_target, _, _ = _profile_columns(
            target,
            target_nuisance,
            nuisance_relative_tolerance=nuisance_relative_tolerance,
        )
        profiled_target_singular = _equilibrated_singular_values(profiled_target)

        joint_rank = _rank(joint_singular, relative_tolerance)
        return AstigmaticFullColumnRankDiagnostics(
            joint_rank=joint_rank,
            joint_nullity=self.joint_parameter_count - joint_rank,
            outer_rank=outer_rank,
            profiled_shared_physical_rank=_rank(
                profiled_shared_singular, relative_tolerance
            ),
            profiled_condenser_target_rank=_rank(
                profiled_target_singular, relative_tolerance
            ),
            shared_physical_parameter_names=tuple(
                self.shared_column.parameter_names[index]
                for index in self.shared_physical_indices
            ),
            unobserved_physical_parameter_names=tuple(
                self.shared_column.parameter_names[index]
                for index in self.unobserved_physical_indices
            ),
            joint_singular_values=joint_singular,
            outer_singular_values=outer_singular,
            profiled_shared_physical_singular_values=profiled_shared_singular,
            profiled_condenser_target_singular_values=profiled_target_singular,
            relative_tolerance=float(relative_tolerance),
            nuisance_relative_tolerance=float(nuisance_relative_tolerance),
        )


def make_astigmatic_full_column_adapter(
    likelihood: AstigmaticCondenserLikelihood,
    shared_column: HardEdgeColumnParameterization | None = None,
    *,
    outer_contract: AstigmaticOuterParameterContract | None = None,
) -> AstigmaticFullColumnAdapter:
    """Convenience constructor for notebook/global-likelihood integration."""

    return AstigmaticFullColumnAdapter.from_likelihood(
        likelihood,
        shared_column,
        outer_contract=outer_contract,
    )


__all__ = [
    "AstigmaticFullColumnAdapter",
    "AstigmaticFullColumnRankDiagnostics",
    "AstigmaticOuterParameterContract",
    "default_astigmatic_outer_parameter_contract",
    "make_astigmatic_full_column_adapter",
]
