"""Nuisance-profiled identifiable subspaces for physical likelihood fits.

The ordinary SVD of a stacked, whitened likelihood can be a poor admission
rule when one observation block is orders of magnitude more sensitive than
another.  A relative cutoff set by that largest singular value can silently
discard every useful direction in the weaker block.  This module separates
the two decisions needed by a physical inverse problem:

* every numerically resolved nuisance direction is retained so that weak
  calibration or acquisition coordinates are profiled rather than frozen;
* optical directions are admitted from the nuisance-profiled optical
  Jacobian using an explicit absolute singular-value (SNR) threshold.

Rejected optical directions are generally confounded with nuisance rather
than pure coordinate axes.  Their reported practical-null vectors therefore
include the least-squares nuisance compensation.  The active basis is the
orthogonal complement of those compensated vectors, preserving the
orthonormal active/null contract used by :mod:`microscope_inversion`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Literal, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize

from .microscope_inversion import (
    DirectionProfileLikelihood,
    IdentifiableParameterSubspace,
    IdentifiableSubspaceFit,
    InversionResult,
    fit_least_squares,
)


ArrayLike = Any
OptimizerCostTransform = Literal["raw", "log_soft_norm"]


def _normalise_optimizer_cost_conditioning(
    transform: OptimizerCostTransform | str,
    residual_norm_scale: float | None,
) -> tuple[OptimizerCostTransform, float | None]:
    """Validate solver-only conditioning without changing the likelihood."""

    if not isinstance(transform, str):
        raise TypeError("optimizer_cost_transform must be a string.")
    if transform not in {"raw", "log_soft_norm"}:
        raise ValueError(
            "optimizer_cost_transform must be 'raw' or 'log_soft_norm'."
        )
    scale = None if residual_norm_scale is None else float(residual_norm_scale)
    if scale is not None and (not np.isfinite(scale) or scale <= 0.0):
        raise ValueError("optimizer_residual_norm_scale must be finite and positive.")
    if scale is not None:
        scale_squared = scale * scale
        if not np.isfinite(scale_squared) or scale_squared <= 0.0:
            raise ValueError(
                "optimizer_residual_norm_scale squared must be finite and positive."
            )
    if transform == "raw" and scale is not None:
        raise ValueError(
            "optimizer_residual_norm_scale is only meaningful when "
            "optimizer_cost_transform='log_soft_norm'."
        )
    return transform, scale


def _condition_slsqp_cost_and_gradient(
    raw_cost: float,
    raw_gradient: np.ndarray,
    *,
    transform: OptimizerCostTransform,
    residual_norm_scale_squared: float,
) -> tuple[float, np.ndarray]:
    r"""Return a minimizer-preserving cost used only by SLSQP.

    For ``log_soft_norm`` this evaluates

    .. math::

       F(c) = \frac{1}{2}\log\left(1 + \frac{2c}{\tau^2}\right),
       \qquad
       \nabla F = \frac{\nabla c}{\tau^2 + 2c},

    where :math:`c=\|r\|_2^2/2`.  The transform is smooth and strictly
    increasing, so it preserves every constrained minimum and the ordering of
    candidate solutions.  It merely prevents a statistically valid but very
    sensitive whitened likelihood from presenting SLSQP with gradients of
    order :math:`\|J\|^2` far from the solution.
    """

    value = float(raw_cost)
    gradient = np.asarray(raw_gradient, dtype=float)
    if transform == "raw":
        return value, gradient
    tau_squared = float(residual_norm_scale_squared)
    if not np.isfinite(tau_squared) or tau_squared <= 0.0:
        raise ValueError("residual_norm_scale_squared must be finite and positive.")
    if value < 0.0:
        raise ValueError("raw_cost must be non-negative.")
    if value == 0.0:
        conditioned_value = 0.0
    else:
        # Work entirely in log space: both ``2 * value`` and
        # ``0.5 * tau_squared`` can overflow/underflow even though their ratio
        # and the requested transformed objective are finite.
        log_ratio = np.log(value) + np.log(2.0) - np.log(tau_squared)
        conditioned_value = 0.5 * float(np.logaddexp(0.0, log_ratio))
    # Scale the sum before inversion.  Directly forming ``tau_squared +
    # 2 * value`` may overflow at the top of the float64 range, while first
    # taking ``1 / max(tau_squared, value)`` may overflow at the bottom even
    # when the final quotient is representable.
    denominator_scale = max(tau_squared, value)
    scaled_denominator = (
        tau_squared / denominator_scale
        + 2.0 * (value / denominator_scale)
    )
    if denominator_scale <= np.finfo(float).max / scaled_denominator:
        conditioned_gradient = gradient / (
            denominator_scale * scaled_denominator
        )
    else:
        # The mathematical denominator is above float64's range.  Divide in
        # stages so a representable subnormal result is not rounded to zero.
        conditioned_gradient = (
            gradient / denominator_scale
        ) / scaled_denominator
    return conditioned_value, conditioned_gradient


def _immutable(values: ArrayLike, *, ndim: int | None = None) -> np.ndarray:
    result = np.array(values, dtype=float, copy=True)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"Expected a {ndim}-dimensional array.")
    if not np.all(np.isfinite(result)):
        raise ValueError("Arrays must contain only finite values.")
    result.setflags(write=False)
    return result


def _rank(singular_values: np.ndarray, threshold: float) -> int:
    return int(np.sum(np.asarray(singular_values, dtype=float) > float(threshold)))


@dataclass(frozen=True)
class NuisanceProfiledSubspace:
    """An optical-SNR-qualified subspace with all resolved nuisance modes."""

    subspace: IdentifiableParameterSubspace
    optical_subspace: IdentifiableParameterSubspace
    jacobian_shape: tuple[int, int]
    optical_parameter_count: int
    nuisance_parameter_count: int
    optical_rank: int
    nuisance_rank: int
    optical_minimum_singular_value: float
    nuisance_rank_rtol: float
    nuisance_numerical_threshold: float
    profiled_optical_singular_values: np.ndarray
    nuisance_singular_values: np.ndarray
    joint_singular_values: np.ndarray
    maximum_practical_null_residual_norm: float

    def __post_init__(self) -> None:
        if not isinstance(self.subspace, IdentifiableParameterSubspace):
            raise TypeError("subspace must be IdentifiableParameterSubspace.")
        if not isinstance(self.optical_subspace, IdentifiableParameterSubspace):
            raise TypeError("optical_subspace must be IdentifiableParameterSubspace.")
        rows, columns = (int(value) for value in self.jacobian_shape)
        optical = int(self.optical_parameter_count)
        nuisance = int(self.nuisance_parameter_count)
        if rows < 1 or columns < 1 or optical < 1 or nuisance < 0:
            raise ValueError("Jacobian and parameter counts must be positive.")
        if optical + nuisance != columns or columns != self.subspace.n_parameters:
            raise ValueError("Parameter counts do not match the subspace.")
        if self.optical_subspace.n_parameters != optical:
            raise ValueError("optical_subspace has the wrong parameter count.")
        if self.optical_rank + self.nuisance_rank != self.subspace.rank:
            raise ValueError("Component ranks do not add to the active rank.")
        if self.optical_subspace.rank != self.optical_rank:
            raise ValueError("optical_subspace has the wrong active rank.")
        if not 0 <= self.optical_rank <= optical:
            raise ValueError("optical_rank lies outside the optical dimension.")
        if not 0 <= self.nuisance_rank <= nuisance:
            raise ValueError("nuisance_rank lies outside the nuisance dimension.")
        if (
            not np.isfinite(self.optical_minimum_singular_value)
            or self.optical_minimum_singular_value <= 0.0
        ):
            raise ValueError("optical_minimum_singular_value must be positive.")
        if not np.isfinite(self.nuisance_rank_rtol) or not 0 < self.nuisance_rank_rtol < 1:
            raise ValueError("nuisance_rank_rtol must lie strictly in (0, 1).")
        if (
            not np.isfinite(self.nuisance_numerical_threshold)
            or self.nuisance_numerical_threshold < 0.0
            or not np.isfinite(self.maximum_practical_null_residual_norm)
            or self.maximum_practical_null_residual_norm < 0.0
        ):
            raise ValueError("Stored thresholds and residual norms must be finite.")
        arrays = {
            "profiled_optical_singular_values": self.profiled_optical_singular_values,
            "nuisance_singular_values": self.nuisance_singular_values,
            "joint_singular_values": self.joint_singular_values,
        }
        for name, values in arrays.items():
            object.__setattr__(self, name, _immutable(values, ndim=1))
        object.__setattr__(self, "jacobian_shape", (rows, columns))
        object.__setattr__(self, "optical_parameter_count", optical)
        object.__setattr__(self, "nuisance_parameter_count", nuisance)
        object.__setattr__(self, "optical_rank", int(self.optical_rank))
        object.__setattr__(self, "nuisance_rank", int(self.nuisance_rank))


@dataclass(frozen=True)
class ActiveSubspaceStability:
    r"""Basis-invariant comparison of two active spaces in scaled coordinates.

    ``principal_cosines`` and ``principal_angles_rad`` contain the complete
    principal spectrum, ordered from the best-aligned to the worst-aligned
    direction.  When the ranks differ, the shorter spectrum is padded with
    zero cosines (right-angle directions).  Consequently a rank change is
    represented explicitly by a maximum principal angle of :math:`\pi/2` and
    a unit spectral-projector distance instead of being mistaken for stable
    overlap in the shared directions.
    """

    ambient_dimension: int
    first_rank: int
    second_rank: int
    ranks_match: bool
    principal_cosines: np.ndarray
    principal_angles_rad: np.ndarray
    maximum_principal_angle_rad: float
    spectral_projector_distance: float

    def __post_init__(self) -> None:
        ambient = int(self.ambient_dimension)
        first_rank = int(self.first_rank)
        second_rank = int(self.second_rank)
        if ambient < 1:
            raise ValueError("ambient_dimension must be positive.")
        if not 0 <= first_rank <= ambient or not 0 <= second_rank <= ambient:
            raise ValueError("Subspace ranks must lie within the ambient dimension.")
        if bool(self.ranks_match) != (first_rank == second_rank):
            raise ValueError("ranks_match is inconsistent with the stored ranks.")
        cosines = _immutable(self.principal_cosines, ndim=1)
        angles = _immutable(self.principal_angles_rad, ndim=1)
        expected = max(first_rank, second_rank)
        if cosines.shape != (expected,) or angles.shape != (expected,):
            raise ValueError("Principal spectra must have length equal to the larger rank.")
        tolerance = 64.0 * np.finfo(float).eps
        if np.any(cosines < -tolerance) or np.any(cosines > 1.0 + tolerance):
            raise ValueError("Principal cosines must lie in [0, 1].")
        if np.any(angles < -tolerance) or np.any(angles > np.pi / 2.0 + tolerance):
            raise ValueError("Principal angles must lie in [0, pi/2].")
        maximum_angle = float(self.maximum_principal_angle_rad)
        projector_distance = float(self.spectral_projector_distance)
        if not np.isfinite(maximum_angle) or not 0.0 <= maximum_angle <= np.pi / 2.0:
            raise ValueError("maximum_principal_angle_rad must lie in [0, pi/2].")
        if (
            not np.isfinite(projector_distance)
            or not 0.0 <= projector_distance <= 1.0
        ):
            raise ValueError("spectral_projector_distance must lie in [0, 1].")
        expected_maximum = float(angles[-1]) if angles.size else 0.0
        if not np.isclose(maximum_angle, expected_maximum, rtol=0.0, atol=tolerance):
            raise ValueError("maximum_principal_angle_rad is inconsistent with the spectrum.")
        object.__setattr__(self, "ambient_dimension", ambient)
        object.__setattr__(self, "first_rank", first_rank)
        object.__setattr__(self, "second_rank", second_rank)
        object.__setattr__(self, "ranks_match", bool(self.ranks_match))
        object.__setattr__(self, "principal_cosines", cosines)
        object.__setattr__(self, "principal_angles_rad", angles)
        object.__setattr__(self, "maximum_principal_angle_rad", maximum_angle)
        object.__setattr__(self, "spectral_projector_distance", projector_distance)


def _scaled_active_basis_and_metadata(
    value: ArrayLike,
    *,
    name: str,
) -> tuple[np.ndarray, tuple[str, ...] | None, np.ndarray | None]:
    """Return an explicit scaled active basis and optional coordinate metadata."""

    if isinstance(value, NuisanceProfiledSubspace):
        value = value.subspace
    if isinstance(value, IdentifiableParameterSubspace):
        basis = np.asarray(value.active_basis_scaled, dtype=float)
        names: tuple[str, ...] | None = tuple(value.parameter_names)
        scales: np.ndarray | None = np.asarray(value.parameter_scales, dtype=float)
    else:
        basis = np.asarray(value, dtype=float)
        names = None
        scales = None
    if basis.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional scaled active basis.")
    if basis.shape[0] < 1:
        raise ValueError(f"{name} must have a positive ambient dimension.")
    if basis.shape[1] > basis.shape[0]:
        raise ValueError(f"{name} cannot have rank larger than its ambient dimension.")
    if not np.all(np.isfinite(basis)):
        raise ValueError(f"{name} must contain only finite values.")
    if basis.shape[1]:
        left_vectors, singular_values, _right_vectors = np.linalg.svd(
            basis, full_matrices=False
        )
        numerical_threshold = (
            np.finfo(float).eps * max(basis.shape) * float(singular_values[0])
        )
        if np.any(singular_values <= numerical_threshold):
            raise ValueError(f"{name} columns must be linearly independent.")
        basis = left_vectors[:, : basis.shape[1]]
    else:
        basis = np.empty((basis.shape[0], 0), dtype=float)
    return basis, names, scales


def compare_scaled_active_subspaces(
    first: ArrayLike,
    second: ArrayLike,
) -> ActiveSubspaceStability:
    r"""Compare active column spaces expressed in common scaled coordinates.

    Arguments may be explicit ``(n_parameters, rank)`` spanning matrices,
    :class:`IdentifiableParameterSubspace` instances, or complete
    :class:`NuisanceProfiledSubspace` designs.  Explicit spanning matrices are
    orthonormalized internally, so the result is invariant to column order,
    sign, rotation, and nonsingular changes of basis.  Object inputs also
    verify that parameter names and parameter scales describe the same scaled
    coordinate system.

    Equal zero-rank spaces are exactly stable: both spectra are empty and both
    scalar distances are zero.  A rank mismatch is returned as a valid,
    explicitly non-matching comparison with missing directions padded at
    :math:`\pi/2`; ambient-dimension mismatch remains an input error.
    """

    first_basis, first_names, first_scales = _scaled_active_basis_and_metadata(
        first, name="first"
    )
    second_basis, second_names, second_scales = _scaled_active_basis_and_metadata(
        second, name="second"
    )
    if first_basis.shape[0] != second_basis.shape[0]:
        raise ValueError("Scaled active subspaces must have equal ambient dimensions.")
    if first_names is not None and second_names is not None:
        if first_names != second_names:
            raise ValueError(
                "Scaled active subspaces must use the same parameter ordering."
            )
        if not np.allclose(first_scales, second_scales, rtol=1.0e-12, atol=0.0):
            raise ValueError(
                "Scaled active subspaces must use the same parameter scales."
            )

    first_rank = first_basis.shape[1]
    second_rank = second_basis.shape[1]
    shared_rank = min(first_rank, second_rank)
    if shared_rank:
        shared_cosines = np.linalg.svd(
            first_basis.T @ second_basis, compute_uv=False
        )
        shared_cosines = np.clip(shared_cosines, 0.0, 1.0)
    else:
        shared_cosines = np.empty(0, dtype=float)
    missing_rank = max(first_rank, second_rank) - shared_rank
    principal_cosines = np.concatenate(
        (shared_cosines, np.zeros(missing_rank, dtype=float))
    )
    principal_angles = np.arccos(principal_cosines)

    first_projector = first_basis @ first_basis.T
    second_projector = second_basis @ second_basis.T
    projector_eigenvalues = np.linalg.eigvalsh(first_projector - second_projector)
    projector_distance = float(
        np.clip(np.max(np.abs(projector_eigenvalues)), 0.0, 1.0)
    )
    maximum_angle = float(principal_angles[-1]) if principal_angles.size else 0.0
    return ActiveSubspaceStability(
        ambient_dimension=first_basis.shape[0],
        first_rank=first_rank,
        second_rank=second_rank,
        ranks_match=first_rank == second_rank,
        principal_cosines=principal_cosines,
        principal_angles_rad=principal_angles,
        maximum_principal_angle_rad=maximum_angle,
        spectral_projector_distance=projector_distance,
    )


def nuisance_profiled_identifiable_subspace(
    jacobian: ArrayLike,
    reference_parameters: ArrayLike,
    parameter_scales: ArrayLike,
    *,
    optical_parameter_count: int,
    optical_minimum_singular_value: float,
    nuisance_rank_rtol: float = 1.0e-10,
    parameter_names: Sequence[str] = (),
) -> NuisanceProfiledSubspace:
    """Build a compensated practical-null and its orthogonal active complement.

    ``jacobian`` must be the actual whitened residual Jacobian.  Singular
    values therefore have the direct interpretation of residual SNR produced
    by a one-unit displacement in the declared scaled coordinates.
    ``optical_minimum_singular_value`` is applied only after projecting optical
    response off the complete numerically resolved nuisance response space.
    """

    matrix = np.asarray(jacobian, dtype=float)
    reference = np.asarray(reference_parameters, dtype=float)
    scales = np.asarray(parameter_scales, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("jacobian must be a non-empty two-dimensional array.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("jacobian must contain only finite values.")
    if reference.shape != (matrix.shape[1],) or not np.all(np.isfinite(reference)):
        raise ValueError("reference_parameters must be finite and match the columns.")
    if scales.shape != reference.shape or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("parameter_scales must be finite, positive, and match parameters.")
    optical_count = int(optical_parameter_count)
    if optical_count != optical_parameter_count or not 0 < optical_count <= matrix.shape[1]:
        raise ValueError("optical_parameter_count must be a valid integer split point.")
    nuisance_count = matrix.shape[1] - optical_count
    optical_threshold = float(optical_minimum_singular_value)
    nuisance_rtol = float(nuisance_rank_rtol)
    if not np.isfinite(optical_threshold) or optical_threshold <= 0:
        raise ValueError("optical_minimum_singular_value must be finite and positive.")
    if not np.isfinite(nuisance_rtol) or not 0 < nuisance_rtol < 1:
        raise ValueError("nuisance_rank_rtol must lie strictly in (0, 1).")
    names = tuple(parameter_names) or tuple(
        f"parameter_{index}" for index in range(matrix.shape[1])
    )
    if len(names) != matrix.shape[1] or len(set(names)) != len(names):
        raise ValueError("parameter_names must be unique and match parameters.")

    scaled = matrix * scales[None, :]
    optical_matrix = scaled[:, :optical_count]
    nuisance_matrix = scaled[:, optical_count:]

    if nuisance_count:
        nuisance_left, nuisance_singular, nuisance_right_t = np.linalg.svd(
            nuisance_matrix, full_matrices=True
        )
        nuisance_threshold = (
            nuisance_rtol * float(nuisance_singular[0])
            if nuisance_singular.size
            else 0.0
        )
        nuisance_rank = _rank(nuisance_singular, nuisance_threshold)
        nuisance_left_active = nuisance_left[:, :nuisance_rank]
        nuisance_right_active = nuisance_right_t[:nuisance_rank, :].T
        nuisance_right_null = nuisance_right_t[nuisance_rank:, :].T
        profiled_optical = optical_matrix - nuisance_left_active @ (
            nuisance_left_active.T @ optical_matrix
        )
    else:
        nuisance_singular = np.empty(0, dtype=float)
        nuisance_threshold = 0.0
        nuisance_rank = 0
        nuisance_left_active = np.empty((matrix.shape[0], 0), dtype=float)
        nuisance_right_active = np.empty((0, 0), dtype=float)
        nuisance_right_null = np.empty((0, 0), dtype=float)
        profiled_optical = optical_matrix.copy()

    _optical_left, optical_singular, optical_right_t = np.linalg.svd(
        profiled_optical, full_matrices=True
    )
    optical_rank = _rank(optical_singular, optical_threshold)
    optical_rejected = optical_right_t[optical_rank:, :].T

    practical_null_blocks: list[np.ndarray] = []
    if optical_rejected.shape[1]:
        if nuisance_rank:
            nuisance_compensation = -nuisance_right_active @ (
                (nuisance_left_active.T @ (optical_matrix @ optical_rejected))
                / nuisance_singular[:nuisance_rank, None]
            )
        else:
            nuisance_compensation = np.zeros(
                (nuisance_count, optical_rejected.shape[1]), dtype=float
            )
        practical_null_blocks.append(
            np.vstack((optical_rejected, nuisance_compensation))
        )
    if nuisance_right_null.shape[1]:
        practical_null_blocks.append(
            np.vstack(
                (
                    np.zeros((optical_count, nuisance_right_null.shape[1])),
                    nuisance_right_null,
                )
            )
        )

    nullity = matrix.shape[1] - optical_rank - nuisance_rank
    if practical_null_blocks:
        candidates = np.concatenate(practical_null_blocks, axis=1)
        if candidates.shape != (matrix.shape[1], nullity):
            raise RuntimeError("Practical-null construction has an inconsistent shape.")
        candidate_rank = np.linalg.matrix_rank(candidates)
        if candidate_rank != nullity:
            raise RuntimeError("Practical-null candidates are not linearly independent.")
        complete_q, _ = np.linalg.qr(candidates, mode="complete")
        null_basis = complete_q[:, :nullity]
        active_basis = complete_q[:, nullity:]
    else:
        null_basis = np.empty((matrix.shape[1], 0), dtype=float)
        active_basis = np.eye(matrix.shape[1], dtype=float)

    if active_basis.shape[1] != optical_rank + nuisance_rank:
        raise RuntimeError("Active basis has the wrong dimension.")
    orthogonality = active_basis.T @ null_basis
    if orthogonality.size and np.max(np.abs(orthogonality)) > 1.0e-10:
        raise RuntimeError("Active and practical-null bases are not orthogonal.")
    practical_null_residual = scaled @ null_basis
    maximum_null_residual = (
        float(np.max(np.linalg.norm(practical_null_residual, axis=0)))
        if nullity
        else 0.0
    )
    joint_singular = np.linalg.svd(scaled, compute_uv=False)
    largest_joint = float(joint_singular[0]) if joint_singular.size else 0.0
    reported_rtol = min(optical_threshold / max(largest_joint, optical_threshold), 0.5)
    subspace = IdentifiableParameterSubspace(
        reference_parameters=reference.copy(),
        parameter_scales=scales.copy(),
        parameter_names=names,
        active_basis_scaled=active_basis,
        null_basis_scaled=null_basis,
        active_basis_physical=scales[:, None] * active_basis,
        null_basis_physical=scales[:, None] * null_basis,
        singular_values=joint_singular,
        rank=active_basis.shape[1],
        threshold=optical_threshold,
        rank_rtol=reported_rtol,
    )
    largest_profiled_optical = (
        float(optical_singular[0]) if optical_singular.size else 0.0
    )
    optical_reported_rtol = min(
        optical_threshold / max(largest_profiled_optical, optical_threshold), 0.5
    )
    optical_active = optical_right_t[:optical_rank, :].T
    optical_null = optical_right_t[optical_rank:, :].T
    optical_subspace = IdentifiableParameterSubspace(
        reference_parameters=reference[:optical_count].copy(),
        parameter_scales=scales[:optical_count].copy(),
        parameter_names=names[:optical_count],
        active_basis_scaled=optical_active,
        null_basis_scaled=optical_null,
        active_basis_physical=scales[:optical_count, None] * optical_active,
        null_basis_physical=scales[:optical_count, None] * optical_null,
        singular_values=optical_singular,
        rank=optical_rank,
        threshold=optical_threshold,
        rank_rtol=optical_reported_rtol,
    )
    return NuisanceProfiledSubspace(
        subspace=subspace,
        optical_subspace=optical_subspace,
        jacobian_shape=matrix.shape,
        optical_parameter_count=optical_count,
        nuisance_parameter_count=nuisance_count,
        optical_rank=optical_rank,
        nuisance_rank=nuisance_rank,
        optical_minimum_singular_value=optical_threshold,
        nuisance_rank_rtol=nuisance_rtol,
        nuisance_numerical_threshold=nuisance_threshold,
        profiled_optical_singular_values=optical_singular,
        nuisance_singular_values=nuisance_singular,
        joint_singular_values=joint_singular,
        maximum_practical_null_residual_norm=maximum_null_residual,
    )


def _normalise_bounds(
    bounds: tuple[ArrayLike, ArrayLike], n_parameters: int
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.broadcast_to(np.asarray(bounds[0], dtype=float), (n_parameters,)).copy()
    upper = np.broadcast_to(np.asarray(bounds[1], dtype=float), (n_parameters,)).copy()
    if np.any(lower >= upper):
        raise ValueError("Every lower bound must be smaller than its upper bound.")
    return lower, upper


def _integer_argument(
    value: Any,
    name: str,
    *,
    minimum: int,
) -> int:
    """Return an exactly integral scalar without silently truncating it."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}.")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"{name} must be an integer greater than or equal to {minimum}."
        ) from error
    if converted < minimum or converted != value:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}.")
    return converted


def _conservative_active_bounds(
    reference: np.ndarray,
    physical_basis: np.ndarray,
    full_bounds: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = full_bounds
    tolerance = 64.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(reference))
    if np.any(reference < lower - tolerance) or np.any(reference > upper + tolerance):
        raise ValueError("reference_parameters lie outside the physical bounds.")
    count = physical_basis.shape[1]
    coordinate_lower = np.full(count, -np.inf)
    coordinate_upper = np.full(count, np.inf)
    if count == 0:
        return coordinate_lower, coordinate_upper
    negative_margin = (lower - reference) / count
    positive_margin = (upper - reference) / count
    for coordinate in range(count):
        coefficients = physical_basis[:, coordinate]
        positive = coefficients > 0.0
        negative = coefficients < 0.0
        if np.any(positive):
            coordinate_lower[coordinate] = max(
                coordinate_lower[coordinate],
                float(np.max(negative_margin[positive] / coefficients[positive])),
            )
            coordinate_upper[coordinate] = min(
                coordinate_upper[coordinate],
                float(np.min(positive_margin[positive] / coefficients[positive])),
            )
        if np.any(negative):
            coordinate_lower[coordinate] = max(
                coordinate_lower[coordinate],
                float(np.max(positive_margin[negative] / coefficients[negative])),
            )
            coordinate_upper[coordinate] = min(
                coordinate_upper[coordinate],
                float(np.min(negative_margin[negative] / coefficients[negative])),
            )
    if np.any(coordinate_lower >= coordinate_upper):
        raise ValueError("The physical bounds contain no conservative active box.")
    return coordinate_lower, coordinate_upper


def nuisance_profiled_active_bounds(
    design: NuisanceProfiledSubspace,
    bounds: tuple[ArrayLike, ArrayLike],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the finite conservative active box before launching a fit."""

    if not isinstance(design, NuisanceProfiledSubspace):
        raise TypeError("design must be NuisanceProfiledSubspace.")
    subspace = design.subspace
    lower, upper = _normalise_bounds(bounds, subspace.n_parameters)
    active_lower, active_upper = _conservative_active_bounds(
        np.asarray(subspace.reference_parameters),
        np.asarray(subspace.active_basis_physical),
        (lower, upper),
    )
    return active_lower.copy(), active_upper.copy()


def _active_polytope_marginal_bounds(
    reference: np.ndarray,
    physical_basis: np.ndarray,
    full_bounds: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact coordinate marginals of the bound-constrained active polytope.

    The physical box becomes a convex polytope after a rotated identifiable-
    subspace embedding.  Unlike :func:`_conservative_active_bounds`, these
    marginal limits do not define a Cartesian box: they are used only to help
    the constrained optimizer and to normalize diagnostics.  Feasibility is
    always enforced by the complete set of linear physical-box constraints.
    """

    constraint_matrix, constraint_lower, constraint_upper = (
        _normalised_active_polytope_constraints(
            reference, physical_basis, full_bounds
        )
    )
    rank = physical_basis.shape[1]
    inequality_matrix = np.concatenate(
        (constraint_matrix, -constraint_matrix), axis=0
    )
    inequality_upper = np.concatenate(
        (constraint_upper, -constraint_lower)
    )
    coordinate_bounds = [(None, None)] * rank
    marginal_lower = np.empty(rank, dtype=float)
    marginal_upper = np.empty(rank, dtype=float)
    linprog_options = {
        "primal_feasibility_tolerance": 1.0e-10,
        "dual_feasibility_tolerance": 1.0e-10,
    }
    for coordinate in range(rank):
        objective = np.zeros(rank, dtype=float)
        objective[coordinate] = 1.0
        minimum = linprog(
            objective,
            A_ub=inequality_matrix,
            b_ub=inequality_upper,
            bounds=coordinate_bounds,
            method="highs",
            options=linprog_options,
        )
        maximum = linprog(
            -objective,
            A_ub=inequality_matrix,
            b_ub=inequality_upper,
            bounds=coordinate_bounds,
            method="highs",
            options=linprog_options,
        )
        if not minimum.success or not maximum.success:
            raise ValueError(
                "The active physical-bound polytope is empty or unbounded."
            )
        marginal_lower[coordinate] = float(minimum.fun)
        marginal_upper[coordinate] = float(-maximum.fun)
    if (
        not np.all(np.isfinite(marginal_lower))
        or not np.all(np.isfinite(marginal_upper))
        or np.any(marginal_lower >= marginal_upper)
    ):
        raise ValueError("The active physical-bound polytope has invalid marginals.")
    return marginal_lower, marginal_upper


def _normalised_active_polytope_constraints(
    reference: np.ndarray,
    physical_basis: np.ndarray,
    full_bounds: tuple[np.ndarray, np.ndarray],
    *,
    require_reference_feasible: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map a physical box to dimensionless active-space inequalities.

    Dividing every physical row by its own box width leaves the feasible set
    unchanged while preventing metre-, radian-, and lens-law rows from giving
    SLSQP and HiGHS incompatible absolute feasibility scales.
    """

    lower, upper = full_bounds
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError(
            "Full-polytope fitting requires finite physical parameter bounds."
        )
    if require_reference_feasible and (
        np.any(reference < lower) or np.any(reference > upper)
    ):
        raise ValueError("reference_parameters lie outside the physical bounds.")
    widths = upper - lower
    constraint_matrix = physical_basis / widths[:, None]
    constraint_lower = (lower - reference) / widths
    constraint_upper = (upper - reference) / widths
    if (
        not np.all(np.isfinite(constraint_matrix))
        or not np.all(np.isfinite(constraint_lower))
        or not np.all(np.isfinite(constraint_upper))
    ):
        raise ValueError(
            "Physical bounds are too narrow to form finite normalized constraints."
        )
    return constraint_matrix, constraint_lower, constraint_upper


def _active_polytope_interior_point(
    constraint_matrix: np.ndarray,
    constraint_lower: np.ndarray,
    constraint_upper: np.ndarray,
) -> np.ndarray:
    """Return a Chebyshev centre in active coordinates.

    Hit-and-run can be effectively immobile at a high-dimensional vertex: a
    random line through a corner almost never intersects the interior.  The
    centre is used for the sampling chain only; the caller's requested initial
    point is still retained as the first optimization start.
    """

    rank = constraint_matrix.shape[1]
    row_norms = np.linalg.norm(constraint_matrix, axis=1)
    active_rows = row_norms > 0.0
    inactive_rows = ~active_rows
    tolerance = 1.0e-10
    if np.any(inactive_rows) and (
        np.any(constraint_lower[inactive_rows] > tolerance)
        or np.any(constraint_upper[inactive_rows] < -tolerance)
    ):
        raise ValueError("The active physical-bound polytope is empty.")
    if not np.any(active_rows):
        raise ValueError("The active physical-bound polytope is unbounded.")
    matrix = constraint_matrix[active_rows]
    norms = row_norms[active_rows]
    objective = np.zeros(rank + 1, dtype=float)
    objective[-1] = -1.0
    inequalities = np.concatenate(
        (
            np.column_stack((matrix, norms)),
            np.column_stack((-matrix, norms)),
        ),
        axis=0,
    )
    inequality_upper = np.concatenate(
        (constraint_upper[active_rows], -constraint_lower[active_rows])
    )
    result = linprog(
        objective,
        A_ub=inequalities,
        b_ub=inequality_upper,
        bounds=[(None, None)] * rank + [(0.0, None)],
        method="highs",
        options={
            "primal_feasibility_tolerance": 1.0e-10,
            "dual_feasibility_tolerance": 1.0e-10,
        },
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError(
            "The active physical-bound polytope has no finite relative interior."
        )
    radius = float(result.x[-1])
    if radius <= 0.0:
        raise ValueError(
            "The active physical-bound polytope has no full-dimensional interior."
        )
    centre = np.asarray(result.x[:-1], dtype=float)
    embedded = constraint_matrix @ centre
    tolerance = 1.0e-9
    if np.any(embedded < constraint_lower - tolerance) or np.any(
        embedded > constraint_upper + tolerance
    ):
        raise RuntimeError("The computed active-polytope centre is infeasible.")
    return centre


def _polytope_line_interval(
    point: np.ndarray,
    direction: np.ndarray,
    constraint_matrix: np.ndarray,
    constraint_lower: np.ndarray,
    constraint_upper: np.ndarray,
) -> tuple[float, float]:
    """Return the complete feasible interval along one active-space line."""

    embedded = constraint_matrix @ point
    velocity = constraint_matrix @ direction
    tolerance = 1.0e-9
    if np.any(embedded < constraint_lower - tolerance) or np.any(
        embedded > constraint_upper + tolerance
    ):
        raise ValueError("Hit-and-run point lies outside the active polytope.")
    line_lower = -np.inf
    line_upper = np.inf
    positive = velocity > 0.0
    negative = velocity < 0.0
    if np.any(positive):
        line_lower = max(
            line_lower,
            float(
                np.max(
                    (constraint_lower[positive] - embedded[positive])
                    / velocity[positive]
                )
            ),
        )
        line_upper = min(
            line_upper,
            float(
                np.min(
                    (constraint_upper[positive] - embedded[positive])
                    / velocity[positive]
                )
            ),
        )
    if np.any(negative):
        line_lower = max(
            line_lower,
            float(
                np.max(
                    (constraint_upper[negative] - embedded[negative])
                    / velocity[negative]
                )
            ),
        )
        line_upper = min(
            line_upper,
            float(
                np.min(
                    (constraint_lower[negative] - embedded[negative])
                    / velocity[negative]
                )
            ),
        )
    if not np.isfinite(line_lower) or not np.isfinite(line_upper):
        raise ValueError("The active physical-bound polytope is unbounded.")
    if line_lower > line_upper + tolerance:
        raise RuntimeError("Computed an empty hit-and-run line interval.")
    return line_lower, line_upper


@dataclass(frozen=True)
class _HitAndRunStartPlan:
    """Immutable provenance for deterministic polytope multistarts.

    ``chain_indices`` uses ``-1`` for the caller-supplied ``active_x0`` and
    zero-based indices for independently seeded hit-and-run chains.  This is a
    private implementation record rather than part of the long-lived
    :class:`InversionResult` API; the public fitter also summarizes it in the
    result message.
    """

    starts: np.ndarray
    chain_indices: np.ndarray
    within_chain_indices: np.ndarray
    chain_seed_words: np.ndarray
    chain_sample_counts: np.ndarray

    def __post_init__(self) -> None:
        starts = _immutable(self.starts, ndim=2)
        chain_indices = np.array(self.chain_indices, dtype=int, copy=True)
        within_chain_indices = np.array(
            self.within_chain_indices, dtype=int, copy=True
        )
        chain_seed_words = np.array(self.chain_seed_words, dtype=np.uint64, copy=True)
        chain_sample_counts = np.array(
            self.chain_sample_counts, dtype=int, copy=True
        )
        if chain_indices.shape != (starts.shape[0],):
            raise ValueError("chain_indices must have one entry per start.")
        if within_chain_indices.shape != chain_indices.shape:
            raise ValueError("within_chain_indices must have one entry per start.")
        if chain_seed_words.ndim != 1 or chain_sample_counts.shape != chain_seed_words.shape:
            raise ValueError("Chain provenance arrays must be one-dimensional and aligned.")
        if starts.shape[0] < 1 or chain_indices[0] != -1 or within_chain_indices[0] != -1:
            raise ValueError("The first start must be the caller-supplied initial point.")
        if np.any(chain_sample_counts < 1):
            raise ValueError("Every instantiated chain must contribute at least one start.")
        if int(np.sum(chain_sample_counts)) != starts.shape[0] - 1:
            raise ValueError("Chain sample counts do not match the stored starts.")
        for values in (
            chain_indices,
            within_chain_indices,
            chain_seed_words,
            chain_sample_counts,
        ):
            values.setflags(write=False)
        object.__setattr__(self, "starts", starts)
        object.__setattr__(self, "chain_indices", chain_indices)
        object.__setattr__(self, "within_chain_indices", within_chain_indices)
        object.__setattr__(self, "chain_seed_words", chain_seed_words)
        object.__setattr__(self, "chain_sample_counts", chain_sample_counts)


def _hit_and_run_chain_samples(
    initial: np.ndarray,
    constraint_matrix: np.ndarray,
    constraint_lower: np.ndarray,
    constraint_upper: np.ndarray,
    *,
    count: int,
    generator: np.random.Generator,
    burn: int,
    thin: int,
    interior_fraction: float,
    excluded_point: np.ndarray,
) -> np.ndarray:
    """Draw ``count`` retained samples from one independently seeded chain."""

    rank = initial.size
    current = initial.copy()
    samples: list[np.ndarray] = []
    # A full-dimensional bounded polytope makes an exact return to the
    # excluded point a probability-zero event, but explicitly skipping it
    # guarantees that ``active_x0`` occurs exactly once in the complete plan.
    max_steps = burn + thin * count + max(128, 32 * rank * count)
    retained_steps = 0
    for step in range(1, max_steps + 1):
        direction = generator.normal(size=rank)
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
            continue
        direction /= norm
        lower, upper = _polytope_line_interval(
            current,
            direction,
            constraint_matrix,
            constraint_lower,
            constraint_upper,
        )
        centre = 0.5 * (lower + upper)
        half_width = 0.5 * (upper - lower) * interior_fraction
        coordinate = (
            centre
            if half_width <= np.finfo(float).eps * max(1.0, abs(centre))
            else generator.uniform(centre - half_width, centre + half_width)
        )
        current = current + coordinate * direction
        if step <= burn or (step - burn) % thin != 0:
            continue
        if np.array_equal(current, excluded_point):
            continue
        samples.append(current.copy())
        retained_steps += 1
        if retained_steps == count:
            break
    if len(samples) != count:
        raise RuntimeError("Hit-and-run failed to generate the requested starts.")
    return np.asarray(samples, dtype=float)


def _hit_and_run_polytope_start_plan(
    initial: np.ndarray,
    chain_initial: np.ndarray,
    constraint_matrix: np.ndarray,
    constraint_lower: np.ndarray,
    constraint_upper: np.ndarray,
    *,
    count: int,
    seed: int,
    chains: int,
    burnin: int | None,
    thinning: int | None,
    interior_fraction: float,
) -> _HitAndRunStartPlan:
    """Generate an independently seeded, deterministic multichain plan."""

    if int(count) != count or count < 1:
        raise ValueError("multistart must be a positive integer.")
    if int(chains) != chains or chains < 1:
        raise ValueError("hit_and_run_chains must be a positive integer.")
    if not np.isfinite(interior_fraction) or not 0.0 < interior_fraction < 1.0:
        raise ValueError("interior_fraction must lie strictly in (0, 1).")
    rank = initial.size
    burn = max(128, 8 * rank) if burnin is None else int(burnin)
    thin = max(8, rank // 2) if thinning is None else int(thinning)
    if burn < 0 or thin < 1:
        raise ValueError("hit-and-run burnin must be non-negative and thinning positive.")
    initial = np.asarray(initial, dtype=float)
    chain_initial = np.asarray(chain_initial, dtype=float)
    if chain_initial.shape != initial.shape or not np.all(np.isfinite(chain_initial)):
        raise ValueError("chain_initial must be finite and match initial.")
    if int(count) == 1:
        return _HitAndRunStartPlan(
            starts=initial[None, :],
            chain_indices=np.asarray((-1,), dtype=int),
            within_chain_indices=np.asarray((-1,), dtype=int),
            chain_seed_words=np.empty(0, dtype=np.uint64),
            chain_sample_counts=np.empty(0, dtype=int),
        )

    sample_count = int(count) - 1
    chain_count = min(int(chains), sample_count)
    chain_sample_counts = np.full(
        chain_count, sample_count // chain_count, dtype=int
    )
    chain_sample_counts[: sample_count % chain_count] += 1
    seed_sequences = np.random.SeedSequence(int(seed)).spawn(chain_count)
    chain_seed_words = np.asarray(
        [sequence.generate_state(1, dtype=np.uint64)[0] for sequence in seed_sequences],
        dtype=np.uint64,
    )

    starts = [initial.copy()]
    chain_indices = [-1]
    within_chain_indices = [-1]
    for chain_index, (sample_allocation, sequence) in enumerate(
        zip(chain_sample_counts, seed_sequences, strict=True)
    ):
        samples = _hit_and_run_chain_samples(
            chain_initial,
            constraint_matrix,
            constraint_lower,
            constraint_upper,
            count=int(sample_allocation),
            generator=np.random.default_rng(sequence),
            burn=burn,
            thin=thin,
            interior_fraction=interior_fraction,
            excluded_point=initial,
        )
        starts.extend(samples)
        chain_indices.extend([chain_index] * int(sample_allocation))
        within_chain_indices.extend(range(int(sample_allocation)))
    if len(starts) != int(count):
        raise RuntimeError("Hit-and-run failed to generate the requested starts.")
    result = np.asarray(starts, dtype=float)
    embedded = result @ constraint_matrix.T
    if np.any(embedded < constraint_lower[None, :]) or np.any(
        embedded > constraint_upper[None, :]
    ):
        raise RuntimeError("Hit-and-run generated a point outside the active polytope.")
    if np.count_nonzero(np.all(result == initial[None, :], axis=1)) != 1:
        raise RuntimeError("The caller-supplied active_x0 was not retained exactly once.")
    return _HitAndRunStartPlan(
        starts=result,
        chain_indices=np.asarray(chain_indices, dtype=int),
        within_chain_indices=np.asarray(within_chain_indices, dtype=int),
        chain_seed_words=chain_seed_words,
        chain_sample_counts=chain_sample_counts,
    )


def _hit_and_run_polytope_starts(
    initial: np.ndarray,
    chain_initial: np.ndarray,
    constraint_matrix: np.ndarray,
    constraint_lower: np.ndarray,
    constraint_upper: np.ndarray,
    *,
    count: int,
    seed: int,
    burnin: int | None,
    thinning: int | None,
    interior_fraction: float,
) -> np.ndarray:
    """Backward-compatible single-chain start-array helper."""

    return _hit_and_run_polytope_start_plan(
        initial,
        chain_initial,
        constraint_matrix,
        constraint_lower,
        constraint_upper,
        count=count,
        seed=seed,
        chains=1,
        burnin=burnin,
        thinning=thinning,
        interior_fraction=interior_fraction,
    ).starts.copy()


def fit_nuisance_profiled_polytope(
    residual_function: Callable[[jnp.ndarray], ArrayLike],
    design: NuisanceProfiledSubspace,
    *,
    bounds: tuple[ArrayLike, ArrayLike],
    active_x0: ArrayLike | None = None,
    multistart: int = 1,
    seed: int = 0,
    hit_and_run_burnin: int | None = None,
    hit_and_run_thinning: int | None = None,
    hit_and_run_chains: int | None = None,
    hit_and_run_interior_fraction: float = 0.995,
    max_nfev: int | None = None,
    ftol: float = 1.0e-10,
    fit_rank_rtol: float | None = None,
    optimizer_cost_transform: OptimizerCostTransform = "raw",
    optimizer_residual_norm_scale: float | None = None,
) -> IdentifiableSubspaceFit:
    """Fit across the complete bound-feasible identifiable polytope.

    Physical box bounds become coupled linear constraints after projection into
    the nuisance-profiled active basis.  This fitter keeps those constraints
    exactly, uses independently seeded hit-and-run chains to distribute starts
    through the full convex polytope, and minimizes the whitened linear least-
    squares cost with SLSQP.  Unless ``hit_and_run_chains`` is given, up to
    four chains share the sampled starts as evenly as possible.  ``active_x0``
    is always retained exactly once, so ``multistart`` remains the total number
    of optimizer starts rather than a per-chain count.
    Rejected practical-null coordinates remain fixed at the declared reference
    gauge and must be audited separately with profile likelihoods.

    ``optimizer_cost_transform='log_soft_norm'`` conditions only the scalar
    objective supplied to SLSQP.  It applies a smooth, strictly monotone
    logarithmic transform to the *total* raw least-squares cost, with
    ``optimizer_residual_norm_scale`` as :math:`\\tau`.  When the scale is
    omitted, :math:`\\tau=\\sqrt{m}` for ``m`` whitened residuals.  This is not a
    robust per-residual loss and does not change the maximum-likelihood
    solution.  Returned costs, gradients, Jacobians, and Fisher covariances
    always use the untransformed whitened residual.  ``ftol`` applies to the
    selected SLSQP objective.

    This is the appropriate multistart route when basin evidence is intended to
    cover the declared active physical box.  The older
    :func:`fit_nuisance_profiled_subspace` remains useful for inexpensive local
    work inside a conservative Cartesian box.
    """

    if not isinstance(design, NuisanceProfiledSubspace):
        raise TypeError("design must be NuisanceProfiledSubspace.")
    if not callable(residual_function):
        raise TypeError("residual_function must be callable.")
    if not np.isfinite(ftol) or ftol <= 0.0:
        raise ValueError("ftol must be finite and positive.")
    optimizer_transform, requested_optimizer_scale = (
        _normalise_optimizer_cost_conditioning(
            optimizer_cost_transform, optimizer_residual_norm_scale
        )
    )
    start_count = _integer_argument(multistart, "multistart", minimum=1)
    seed_value = _integer_argument(seed, "seed", minimum=0)
    burnin = (
        None
        if hit_and_run_burnin is None
        else _integer_argument(
            hit_and_run_burnin, "hit_and_run_burnin", minimum=0
        )
    )
    thinning = (
        None
        if hit_and_run_thinning is None
        else _integer_argument(
            hit_and_run_thinning, "hit_and_run_thinning", minimum=1
        )
    )
    requested_chain_count = (
        4
        if hit_and_run_chains is None
        else _integer_argument(
            hit_and_run_chains, "hit_and_run_chains", minimum=1
        )
    )
    iteration_limit = (
        None
        if max_nfev is None
        else _integer_argument(max_nfev, "max_nfev", minimum=1)
    )

    subspace = design.subspace
    if subspace.rank < 1:
        raise ValueError("The nuisance-profiled design has no active coordinates.")
    reference = np.asarray(subspace.reference_parameters, dtype=float)
    physical_basis = np.asarray(subspace.active_basis_physical, dtype=float)
    lower, upper = _normalise_bounds(bounds, subspace.n_parameters)
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError(
            "fit_nuisance_profiled_polytope requires finite physical bounds."
        )
    constraint_matrix, constraint_lower, constraint_upper = (
        _normalised_active_polytope_constraints(
            reference, physical_basis, (lower, upper)
        )
    )
    active_lower, active_upper = _active_polytope_marginal_bounds(
        reference, physical_basis, (lower, upper)
    )
    initial = (
        np.zeros(subspace.rank, dtype=float)
        if active_x0 is None
        else np.asarray(active_x0, dtype=float)
    )
    if initial.shape != (subspace.rank,) or not np.all(np.isfinite(initial)):
        raise ValueError(f"active_x0 must be finite with shape ({subspace.rank},).")
    initial_constraint_values = constraint_matrix @ initial
    feasibility_tolerance = 1.0e-9
    if np.any(initial_constraint_values < constraint_lower) or np.any(
        initial_constraint_values > constraint_upper
    ):
        raise ValueError("active_x0 lies outside the active physical-bound polytope.")

    chain_initial = (
        initial.copy()
        if start_count == 1
        else _active_polytope_interior_point(
            constraint_matrix, constraint_lower, constraint_upper
        )
    )

    start_plan = _hit_and_run_polytope_start_plan(
        initial,
        chain_initial,
        constraint_matrix,
        constraint_lower,
        constraint_upper,
        count=start_count,
        seed=seed_value,
        chains=requested_chain_count,
        burnin=burnin,
        thinning=thinning,
        interior_fraction=float(hit_and_run_interior_fraction),
    )
    starts = np.asarray(start_plan.starts)
    # LP marginals are redundant helper bounds, while the complete linear
    # constraints define feasibility.  Include every independently verified
    # start in those helper bounds despite possible last-bit LP roundoff.
    active_lower = np.minimum(active_lower, np.min(starts, axis=0))
    active_upper = np.maximum(active_upper, np.max(starts, axis=0))

    jax_coordinate_dtype = jnp.asarray(initial).dtype

    def active_residual(coordinates: jnp.ndarray) -> jnp.ndarray:
        return jnp.ravel(
            residual_function(subspace.parameters_from_active(coordinates))
        )

    def objective(coordinates: jnp.ndarray) -> jnp.ndarray:
        residual = active_residual(coordinates)
        return 0.5 * jnp.real(jnp.vdot(residual, residual))

    objective_and_gradient = jax.jit(jax.value_and_grad(objective))
    residual_jacobian = jax.jit(jax.jacfwd(active_residual))
    # Compile before entering SciPy so compile latency is not charged to one
    # arbitrarily selected start.
    compiled_residual = active_residual(
        jnp.asarray(initial, dtype=jax_coordinate_dtype)
    )
    if compiled_residual.size == 0:
        raise ValueError("residual_function must return at least one residual.")
    if not np.issubdtype(np.dtype(compiled_residual.dtype), np.floating):
        raise TypeError("residual_function must return real floating-point residuals.")
    compiled_value, compiled_gradient = objective_and_gradient(
        jnp.asarray(initial, dtype=jax_coordinate_dtype)
    )
    compiled_value.block_until_ready()
    compiled_gradient.block_until_ready()
    compiled_residual.block_until_ready()
    if not np.all(np.isfinite(np.asarray(compiled_residual))):
        raise FloatingPointError(
            "residual_function returned non-finite values at active_x0."
        )
    optimizer_scale = (
        np.sqrt(float(compiled_residual.size))
        if requested_optimizer_scale is None
        else requested_optimizer_scale
    )
    optimizer_scale_squared = float(optimizer_scale) * float(optimizer_scale)
    if not np.isfinite(optimizer_scale_squared) or optimizer_scale_squared <= 0.0:
        raise ValueError(
            "optimizer_residual_norm_scale squared must be finite and positive."
        )

    def scipy_objective(coordinates: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = objective_and_gradient(
            jnp.asarray(coordinates, dtype=jax_coordinate_dtype)
        )
        scalar = float(value)
        derivative = np.asarray(gradient, dtype=float)
        if not np.isfinite(scalar) or not np.all(np.isfinite(derivative)):
            return float(np.finfo(float).max / 1.0e100), np.zeros_like(coordinates)
        return _condition_slsqp_cost_and_gradient(
            scalar,
            derivative,
            transform=optimizer_transform,
            residual_norm_scale_squared=optimizer_scale_squared,
        )

    linear_constraint = LinearConstraint(
        constraint_matrix,
        constraint_lower,
        constraint_upper,
    )
    coordinate_bounds = Bounds(active_lower, active_upper)
    options: dict[str, object] = {"ftol": float(ftol), "disp": False}
    if iteration_limit is not None:
        # SLSQP exposes an iteration rather than a function-evaluation limit;
        # retain the established max_nfev API name for fitter compatibility.
        options["maxiter"] = iteration_limit
    fits = [
        minimize(
            scipy_objective,
            start,
            method="SLSQP",
            jac=True,
            bounds=coordinate_bounds,
            constraints=(linear_constraint,),
            options=options,
        )
        for start in starts
    ]
    solution_coordinates = np.asarray([fit.x for fit in fits], dtype=float)
    solution_constraint_values = solution_coordinates @ constraint_matrix.T
    feasible = np.all(
        (solution_constraint_values >= constraint_lower[None, :] - feasibility_tolerance)
        & (solution_constraint_values <= constraint_upper[None, :] + feasibility_tolerance),
        axis=1,
    )
    finite_solutions = np.all(np.isfinite(solution_coordinates), axis=1)
    final_evaluations: list[tuple[float, np.ndarray]] = []
    for solution, finite in zip(
        solution_coordinates, finite_solutions, strict=True
    ):
        if not finite:
            final_evaluations.append(
                (np.inf, np.full(subspace.rank, np.nan, dtype=float))
            )
            continue
        value, gradient = objective_and_gradient(
            jnp.asarray(solution, dtype=jax_coordinate_dtype)
        )
        scalar = float(value)
        derivative = np.asarray(gradient, dtype=float)
        if not np.isfinite(scalar) or not np.all(np.isfinite(derivative)):
            scalar = np.inf
            derivative = np.full(subspace.rank, np.nan, dtype=float)
        final_evaluations.append((scalar, derivative))
    # Re-evaluate every returned point with the actual compiled objective.  A
    # stale or inaccurate ``OptimizeResult.fun`` must not hide a lower-cost
    # feasible candidate, especially when that candidate exhausted maxiter.
    start_costs = np.asarray([item[0] for item in final_evaluations], dtype=float)
    gradients = np.asarray([item[1] for item in final_evaluations], dtype=float)
    usable = feasible & np.isfinite(start_costs) & finite_solutions
    successes = np.asarray(
        [bool(fit.success) for fit in fits], dtype=bool
    ) & usable
    finite_feasible = np.flatnonzero(usable)
    if finite_feasible.size:
        best_index = int(
            finite_feasible[np.argmin(start_costs[finite_feasible])]
        )
    else:
        raise RuntimeError(
            "Every full-polytope optimization returned a non-finite or "
            "physically infeasible solution."
        )
    best = fits[best_index]
    best_coordinates = np.asarray(best.x, dtype=float)
    residuals = np.asarray(
        active_residual(
            jnp.asarray(best_coordinates, dtype=jax_coordinate_dtype)
        ),
        dtype=float,
    )
    final_jacobian = np.asarray(
        residual_jacobian(
            jnp.asarray(best_coordinates, dtype=jax_coordinate_dtype)
        ),
        dtype=float,
    )
    if not np.all(np.isfinite(residuals)) or not np.all(np.isfinite(final_jacobian)):
        raise RuntimeError("The selected full-polytope solution is non-finite.")

    _left, singular, right_t = np.linalg.svd(final_jacobian, full_matrices=True)
    relative = (
        np.finfo(float).eps * max(final_jacobian.shape)
        if fit_rank_rtol is None
        else float(fit_rank_rtol)
    )
    if not np.isfinite(relative) or not 0.0 < relative < 1.0:
        raise ValueError("fit_rank_rtol must lie strictly in (0, 1).")
    threshold = relative * (float(singular[0]) if singular.size else 0.0)
    fit_rank = _rank(singular, threshold)
    vectors = right_t[:fit_rank, :].T
    fisher_covariance = (
        vectors @ np.diag(1.0 / singular[:fit_rank] ** 2) @ vectors.T
        if fit_rank
        else np.zeros((subspace.rank, subspace.rank), dtype=float)
    )
    fit_null = right_t[fit_rank:, :].T
    active_covariance = fisher_covariance.copy()
    active_errors = np.sqrt(np.maximum(np.diag(fisher_covariance), 0.0))
    if fit_null.size:
        affected = np.sum(fit_null**2, axis=1) > 100.0 * np.finfo(float).eps
        active_covariance[affected, :] = np.nan
        active_covariance[:, affected] = np.nan
        active_errors[affected] = np.inf

    active_mask = np.zeros_like(solution_coordinates, dtype=int)
    coordinate_tolerance = 1.0e-7 * np.maximum(
        active_upper - active_lower, np.finfo(float).tiny
    )
    active_mask[
        np.abs(solution_coordinates - active_lower[None, :])
        <= coordinate_tolerance[None, :]
    ] = -1
    active_mask[
        np.abs(solution_coordinates - active_upper[None, :])
        <= coordinate_tolerance[None, :]
    ] = 1
    active_fit = InversionResult(
        parameters=best_coordinates,
        parameter_names=tuple(
            f"active_profiled_{index}" for index in range(subspace.rank)
        ),
        success=bool(successes[best_index]),
        cost=float(0.5 * residuals @ residuals),
        residuals=residuals,
        jacobian=final_jacobian,
        singular_values=singular,
        covariance=active_covariance,
        identifiable_covariance=fisher_covariance,
        parameter_standard_errors=active_errors,
        null_space_basis=fit_null,
        jacobian_rank=fit_rank,
        degrees_of_freedom=max(residuals.size - fit_rank, 1),
        message=(
            "SLSQP over the complete linearly constrained active physical-bound "
            "polytope; optimizer objective "
            + (
                "is the raw whitened least-squares cost; "
                if optimizer_transform == "raw"
                else (
                    "is a strictly monotone log-soft transform of the total raw "
                    f"cost with residual-norm scale {optimizer_scale:.12g}; "
                    "reported costs/gradients/Fisher quantities remain raw; "
                )
            )
            + f"{start_plan.chain_sample_counts.size} independently seeded "
            "hit-and-run chain(s) with per-chain sampled-start counts "
            f"{start_plan.chain_sample_counts.tolist()} and root seed {seed_value}: "
            f"{best.message}"
        ),
        nfev=int(getattr(best, "nfev", 0)),
        njev=(
            None
            if getattr(best, "njev", None) is None
            else int(getattr(best, "njev"))
        ),
        start_parameters=starts,
        start_costs=start_costs,
        solution_parameters=solution_coordinates,
        start_success=successes,
        start_gradients=gradients,
        start_active_mask=active_mask,
    )
    # Report the same NumPy float64 embedding used by the physical constraints;
    # do not silently round the final parameters through JAX float32 when x64
    # is disabled in an otherwise low-precision caller.
    parameters = reference + physical_basis @ best_coordinates
    identifiable_covariance = (
        physical_basis @ fisher_covariance @ physical_basis.T
    )
    projected_errors = np.sqrt(
        np.maximum(np.diag(identifiable_covariance), 0.0)
    )
    return IdentifiableSubspaceFit(
        subspace=subspace,
        active_fit=active_fit,
        parameters=parameters,
        active_coordinates=best_coordinates,
        active_coordinate_bounds=(active_lower.copy(), active_upper.copy()),
        active_fisher_covariance=fisher_covariance,
        identifiable_covariance=identifiable_covariance,
        projected_parameter_standard_errors=projected_errors,
        fit_singular_values=singular,
        fit_rank=fit_rank,
        fit_threshold=threshold,
        fit_null_basis_active=fit_null,
    )


def profile_nuisance_profiled_polytope_direction(
    residual_function: Callable[[jnp.ndarray], ArrayLike],
    fit: IdentifiableSubspaceFit,
    scaled_direction: ArrayLike,
    grid: ArrayLike,
    *,
    bounds: tuple[ArrayLike, ArrayLike],
    direction_name: str = "scaled_direction",
    multistart: int = 4,
    seed: int = 0,
    hit_and_run_burnin: int | None = None,
    hit_and_run_thinning: int | None = None,
    hit_and_run_chains: int | None = None,
    hit_and_run_interior_fraction: float = 0.995,
    max_nfev: int | None = None,
    ftol: float = 1.0e-10,
    optimizer_cost_transform: OptimizerCostTransform = "raw",
    optimizer_residual_norm_scale: float | None = None,
) -> DirectionProfileLikelihood:
    """Profile one direction over every feasible orthogonal active coordinate.

    The ordinary profile helper uses a conservative Cartesian box because
    ``scipy.optimize.least_squares`` cannot represent the coupled physical
    bounds induced by a rotated SVD basis.  This version instead constructs the
    exact conditional polytope at each fixed grid coordinate, samples it with
    up to four independently seeded hit-and-run chains by default, and
    optimizes every start with normalized SLSQP constraints.  The chosen base
    or warm start is retained exactly once and ``multistart`` is the total
    number of starts per slice.  It therefore cannot overstate curvature
    merely because compensating active coordinates fell outside an artificial
    ``1 / rank`` box.

    ``optimizer_cost_transform='log_soft_norm'`` has the same solver-only,
    minimizer-preserving semantics as in :func:`fit_nuisance_profiled_polytope`.
    Profile costs remain raw whitened least-squares costs.
    """

    if not isinstance(fit, IdentifiableSubspaceFit):
        raise TypeError("fit must be an IdentifiableSubspaceFit.")
    if not callable(residual_function):
        raise TypeError("residual_function must be callable.")
    if not np.isfinite(ftol) or ftol <= 0.0:
        raise ValueError("ftol must be finite and positive.")
    optimizer_transform, requested_optimizer_scale = (
        _normalise_optimizer_cost_conditioning(
            optimizer_cost_transform, optimizer_residual_norm_scale
        )
    )
    start_count = _integer_argument(multistart, "multistart", minimum=1)
    seed_value = _integer_argument(seed, "seed", minimum=0)
    burnin = (
        None
        if hit_and_run_burnin is None
        else _integer_argument(
            hit_and_run_burnin, "hit_and_run_burnin", minimum=0
        )
    )
    thinning = (
        None
        if hit_and_run_thinning is None
        else _integer_argument(
            hit_and_run_thinning, "hit_and_run_thinning", minimum=1
        )
    )
    requested_chain_count = (
        4
        if hit_and_run_chains is None
        else _integer_argument(
            hit_and_run_chains, "hit_and_run_chains", minimum=1
        )
    )
    iteration_limit = (
        None
        if max_nfev is None
        else _integer_argument(max_nfev, "max_nfev", minimum=1)
    )
    direction = np.asarray(scaled_direction, dtype=float)
    if direction.shape != (fit.subspace.n_parameters,) or not np.all(
        np.isfinite(direction)
    ):
        raise ValueError("scaled_direction must be finite and match full parameters.")
    direction_norm = float(np.linalg.norm(direction))
    if not np.isfinite(direction_norm) or direction_norm <= 0.0:
        raise ValueError("scaled_direction must be non-zero.")
    direction /= direction_norm
    values = np.asarray(grid, dtype=float)
    if (
        values.ndim != 1
        or values.size < 2
        or not np.all(np.isfinite(values))
        or len(np.unique(values)) != values.size
    ):
        raise ValueError("grid must contain at least two distinct finite coordinates.")
    lower, upper = _normalise_bounds(bounds, fit.subspace.n_parameters)
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("Full-polytope profiling requires finite physical bounds.")

    active = np.asarray(fit.subspace.active_basis_scaled, dtype=float)
    active_component = active.T @ direction
    if float(np.linalg.norm(active_component)) <= 256.0 * np.finfo(float).eps:
        coefficient_basis = np.eye(fit.subspace.rank)
    else:
        _left, _singular, right_t = np.linalg.svd(
            active_component[None, :], full_matrices=True
        )
        coefficient_basis = right_t[1:, :].T
    orthogonal_active = active @ coefficient_basis
    physical_orthogonal = (
        np.asarray(fit.subspace.parameter_scales)[:, None] * orthogonal_active
    )
    physical_direction = np.asarray(fit.subspace.parameter_scales) * direction
    free_rank = physical_orthogonal.shape[1]
    jax_coordinate_dtype = jnp.asarray(fit.active_coordinates).dtype

    def conditional_residual(
        free_coordinates: jnp.ndarray, base_parameters: jnp.ndarray
    ) -> jnp.ndarray:
        parameters = base_parameters + jnp.asarray(physical_orthogonal) @ free_coordinates
        return jnp.ravel(residual_function(parameters))

    def conditional_objective(
        free_coordinates: jnp.ndarray, base_parameters: jnp.ndarray
    ) -> jnp.ndarray:
        residual = conditional_residual(free_coordinates, base_parameters)
        return 0.5 * jnp.real(jnp.vdot(residual, residual))

    conditional_value_gradient = jax.jit(
        jax.value_and_grad(conditional_objective, argnums=0)
    )
    costs: list[float] = []
    conditional_parameters: list[np.ndarray] = []
    successes: list[bool] = []
    previous = np.zeros(free_rank, dtype=float)
    for grid_index, coordinate in enumerate(values):
        base = np.asarray(fit.parameters, dtype=float) + coordinate * physical_direction
        if free_rank == 0:
            widths = upper - lower
            normalized_base = (base - lower) / widths
            if np.any(normalized_base < -1.0e-10) or np.any(
                normalized_base > 1.0 + 1.0e-10
            ):
                raise ValueError(
                    f"Profile grid coordinate {coordinate:g} has an empty "
                    "conditional physical-bound slice."
                )
            residual = np.asarray(residual_function(jnp.asarray(base)), dtype=float).ravel()
            if residual.size == 0 or not np.all(np.isfinite(residual)):
                raise FloatingPointError(
                    f"Profile residual is non-finite at coordinate {coordinate:g}."
                )
            conditional_parameters.append(base)
            costs.append(float(0.5 * residual @ residual))
            successes.append(True)
            continue

        constraint_matrix, constraint_lower, constraint_upper = (
            _normalised_active_polytope_constraints(
                base,
                physical_orthogonal,
                (lower, upper),
                require_reference_feasible=False,
            )
        )
        previous_values = constraint_matrix @ previous
        previous_feasible = not (
            np.any(previous_values < constraint_lower - 1.0e-9)
            or np.any(previous_values > constraint_upper + 1.0e-9)
        )
        zero_feasible = bool(
            np.all(constraint_lower <= 1.0e-9)
            and np.all(constraint_upper >= -1.0e-9)
        )
        # Seed the zero-coordinate slice with the reported global fit.  This
        # prevents a local conditional solve from reporting a higher cost at
        # grid zero merely because a warm start entered a different basin.
        if coordinate == 0.0 and zero_feasible:
            profile_initial = np.zeros(free_rank, dtype=float)
        elif previous_feasible:
            profile_initial = previous.copy()
        elif zero_feasible:
            profile_initial = np.zeros(free_rank, dtype=float)
        else:
            profile_initial = _active_polytope_interior_point(
                constraint_matrix, constraint_lower, constraint_upper
            )
        chain_initial = (
            profile_initial.copy()
            if start_count == 1
            else _active_polytope_interior_point(
                constraint_matrix, constraint_lower, constraint_upper
            )
        )
        start_plan = _hit_and_run_polytope_start_plan(
            profile_initial,
            chain_initial,
            constraint_matrix,
            constraint_lower,
            constraint_upper,
            count=start_count,
            seed=seed_value + 104729 * grid_index,
            chains=requested_chain_count,
            burnin=burnin,
            thinning=thinning,
            interior_fraction=float(hit_and_run_interior_fraction),
        )
        starts = np.asarray(start_plan.starts)
        base_jax = jnp.asarray(base)
        compiled_residual = conditional_residual(
            jnp.asarray(profile_initial, dtype=jax_coordinate_dtype), base_jax
        )
        if compiled_residual.size == 0:
            raise ValueError("residual_function must return at least one residual.")
        if not np.issubdtype(np.dtype(compiled_residual.dtype), np.floating):
            raise TypeError("residual_function must return real floating-point residuals.")
        if not np.all(np.isfinite(np.asarray(compiled_residual))):
            raise FloatingPointError(
                f"Profile residual is non-finite at coordinate {coordinate:g}."
            )
        optimizer_scale = (
            np.sqrt(float(compiled_residual.size))
            if requested_optimizer_scale is None
            else requested_optimizer_scale
        )
        optimizer_scale_squared = float(optimizer_scale) * float(optimizer_scale)
        if not np.isfinite(optimizer_scale_squared) or optimizer_scale_squared <= 0.0:
            raise ValueError(
                "optimizer_residual_norm_scale squared must be finite and positive."
            )

        def scipy_objective(free_coordinates: np.ndarray) -> tuple[float, np.ndarray]:
            value, gradient = conditional_value_gradient(
                jnp.asarray(free_coordinates, dtype=jax_coordinate_dtype), base_jax
            )
            scalar = float(value)
            derivative = np.asarray(gradient, dtype=float)
            if not np.isfinite(scalar) or not np.all(np.isfinite(derivative)):
                return float(np.finfo(float).max / 1.0e100), np.zeros_like(
                    free_coordinates
                )
            return _condition_slsqp_cost_and_gradient(
                scalar,
                derivative,
                transform=optimizer_transform,
                residual_norm_scale_squared=optimizer_scale_squared,
            )

        options: dict[str, object] = {"ftol": float(ftol), "disp": False}
        if iteration_limit is not None:
            options["maxiter"] = iteration_limit
        linear_constraint = LinearConstraint(
            constraint_matrix, constraint_lower, constraint_upper
        )
        optimizations = [
            minimize(
                scipy_objective,
                start,
                method="SLSQP",
                jac=True,
                constraints=(linear_constraint,),
                options=options,
            )
            for start in starts
        ]
        solutions = np.asarray([result.x for result in optimizations], dtype=float)
        solution_constraints = solutions @ constraint_matrix.T
        feasible = np.all(
            (solution_constraints >= constraint_lower[None, :] - 1.0e-9)
            & (solution_constraints <= constraint_upper[None, :] + 1.0e-9),
            axis=1,
        )
        finite_solutions = np.all(np.isfinite(solutions), axis=1)
        solution_costs = np.full(len(optimizations), np.inf, dtype=float)
        for solution_index, (solution, finite) in enumerate(
            zip(solutions, finite_solutions, strict=True)
        ):
            if not finite:
                continue
            value, gradient = conditional_value_gradient(
                jnp.asarray(solution, dtype=jax_coordinate_dtype), base_jax
            )
            scalar = float(value)
            derivative = np.asarray(gradient, dtype=float)
            if np.isfinite(scalar) and np.all(np.isfinite(derivative)):
                solution_costs[solution_index] = scalar
        usable = (
            feasible
            & np.isfinite(solution_costs)
            & finite_solutions
        )
        converged = usable & np.asarray(
            [bool(result.success) for result in optimizations], dtype=bool
        )
        fallback = np.flatnonzero(usable)
        if fallback.size:
            best_index = int(fallback[np.argmin(solution_costs[fallback])])
        else:
            raise RuntimeError(
                f"Every conditional profile optimization failed at {coordinate:g}."
            )
        previous = solutions[best_index]
        parameters = base + physical_orthogonal @ previous
        residual = np.asarray(
            residual_function(jnp.asarray(parameters)), dtype=float
        ).ravel()
        if residual.size == 0 or not np.all(np.isfinite(residual)):
            raise FloatingPointError(
                f"Selected profile solution is non-finite at {coordinate:g}."
            )
        conditional_parameters.append(parameters)
        costs.append(float(0.5 * residual @ residual))
        successes.append(bool(converged[best_index]))

    return DirectionProfileLikelihood(
        direction_name=str(direction_name),
        direction_scaled=direction.copy(),
        direction_physical=physical_direction.copy(),
        grid=values.copy(),
        costs=np.asarray(costs, dtype=float),
        conditional_parameters=np.asarray(conditional_parameters, dtype=float),
        success=np.asarray(successes, dtype=bool),
        orthogonal_active_basis_scaled=orthogonal_active.copy(),
    )


def fit_nuisance_profiled_subspace(
    residual_function: Callable[[jnp.ndarray], ArrayLike],
    design: NuisanceProfiledSubspace,
    *,
    bounds: tuple[ArrayLike, ArrayLike] = (-np.inf, np.inf),
    active_x0: ArrayLike | None = None,
    jacobian_mode: str = "forward",
    fit_rank_rtol: float | None = None,
    **fit_options: Any,
) -> IdentifiableSubspaceFit:
    """Fit the precomputed nuisance-profiled active basis.

    The returned object is the existing :class:`IdentifiableSubspaceFit`, so
    downstream profile-likelihood and plotting APIs remain compatible.  The
    Fisher covariance uses the absolute scale of the already-whitened
    residual and is not rescaled by the realized residual variance.
    """

    if not isinstance(design, NuisanceProfiledSubspace):
        raise TypeError("design must be NuisanceProfiledSubspace.")
    if not callable(residual_function):
        raise TypeError("residual_function must be callable.")
    subspace = design.subspace
    reference = np.asarray(subspace.reference_parameters, dtype=float)
    lower, upper = _normalise_bounds(bounds, subspace.n_parameters)
    active_bounds = nuisance_profiled_active_bounds(design, (lower, upper))
    initial = (
        np.zeros(subspace.rank, dtype=float)
        if active_x0 is None
        else np.asarray(active_x0, dtype=float)
    )
    if initial.shape != (subspace.rank,) or not np.all(np.isfinite(initial)):
        raise ValueError(f"active_x0 must be finite with shape ({subspace.rank},).")
    if np.any(initial < active_bounds[0]) or np.any(initial > active_bounds[1]):
        raise ValueError("active_x0 lies outside the conservative active bounds.")
    if jacobian_mode not in {"forward", "reverse"}:
        raise ValueError("jacobian_mode must be 'forward' or 'reverse'.")

    def active_residual(coordinates: jnp.ndarray) -> jnp.ndarray:
        return jnp.ravel(
            residual_function(subspace.parameters_from_active(coordinates))
        )

    options = dict(fit_options)
    for forbidden in ("bounds", "parameter_names", "jacobian_mode"):
        if forbidden in options:
            raise TypeError(f"{forbidden} is an explicit fitter argument.")
    options.setdefault("loss", "linear")
    active_fit = fit_least_squares(
        active_residual,
        initial,
        bounds=active_bounds,
        parameter_names=tuple(
            f"active_profiled_{index}" for index in range(subspace.rank)
        ),
        jacobian_mode=jacobian_mode,
        **options,
    )
    parameters = np.asarray(
        subspace.parameters_from_active(jnp.asarray(active_fit.parameters)), dtype=float
    )
    tolerance = 128.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(parameters))
    if np.any(parameters < lower - tolerance) or np.any(parameters > upper + tolerance):
        raise RuntimeError("Embedded optimizer result escaped the physical bounds.")

    final_jacobian = np.asarray(active_fit.jacobian, dtype=float)
    _left, singular, right_t = np.linalg.svd(final_jacobian, full_matrices=True)
    relative = (
        np.finfo(float).eps * max(final_jacobian.shape)
        if fit_rank_rtol is None
        else float(fit_rank_rtol)
    )
    if not np.isfinite(relative) or not 0 < relative < 1:
        raise ValueError("fit_rank_rtol must lie strictly in (0, 1).")
    threshold = relative * (float(singular[0]) if singular.size else 0.0)
    fit_rank = _rank(singular, threshold)
    vectors = right_t[:fit_rank, :].T
    fisher_covariance = (
        vectors @ np.diag(1.0 / singular[:fit_rank] ** 2) @ vectors.T
        if fit_rank
        else np.zeros((subspace.rank, subspace.rank), dtype=float)
    )
    fit_null = right_t[fit_rank:, :].T
    active_covariance = fisher_covariance.copy()
    active_errors = np.sqrt(np.maximum(np.diag(fisher_covariance), 0.0))
    if fit_null.size:
        affected = np.sum(fit_null**2, axis=1) > 100.0 * np.finfo(float).eps
        active_covariance[affected, :] = np.nan
        active_covariance[:, affected] = np.nan
        active_errors[affected] = np.inf
    active_fit = replace(
        active_fit,
        singular_values=singular,
        covariance=active_covariance,
        identifiable_covariance=fisher_covariance,
        parameter_standard_errors=active_errors,
        null_space_basis=fit_null,
        jacobian_rank=fit_rank,
        degrees_of_freedom=max(active_fit.residuals.size - fit_rank, 1),
    )
    physical_basis = np.asarray(subspace.active_basis_physical)
    identifiable_covariance = physical_basis @ fisher_covariance @ physical_basis.T
    projected_errors = np.sqrt(
        np.maximum(np.diag(identifiable_covariance), 0.0)
    )
    return IdentifiableSubspaceFit(
        subspace=subspace,
        active_fit=active_fit,
        parameters=parameters,
        active_coordinates=np.asarray(active_fit.parameters, dtype=float),
        active_coordinate_bounds=(active_bounds[0].copy(), active_bounds[1].copy()),
        active_fisher_covariance=fisher_covariance,
        identifiable_covariance=identifiable_covariance,
        projected_parameter_standard_errors=projected_errors,
        fit_singular_values=singular,
        fit_rank=fit_rank,
        fit_threshold=threshold,
        fit_null_basis_active=fit_null,
    )


__all__ = [
    "ActiveSubspaceStability",
    "NuisanceProfiledSubspace",
    "compare_scaled_active_subspaces",
    "fit_nuisance_profiled_polytope",
    "fit_nuisance_profiled_subspace",
    "nuisance_profiled_active_bounds",
    "nuisance_profiled_identifiable_subspace",
    "profile_nuisance_profiled_polytope_direction",
]
