"""Native hard-edge diffraction-mode design and rendering for NEOARM.

The routines in this module deliberately use the physical downstream column.
They do not insert an ideal Fourier relay.  A diffraction mode is defined by
the sample-to-detector transfer blocks

``A ~= 0`` and ``B ~= L R``,

where ``L`` is the requested camera length and ``R`` is the (generally
DAC-dependent) Larmor rotation.  The hard-edge column is round, so the two
scalar conditions ``|A| = 0`` and ``|B| = L`` are sufficient; the returned
diagnostics nevertheless check the complete 4x4 matrix, the full 2x2 blocks,
round structure, and symplecticity.

All control coordinates are raw/normalised DAC values.  No electrical-current
readback or hidden current calibration is assumed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc

from .gaussian import GaussianBeam
from .microscope_inversion import symplectic_defect
from .neoarm_calibration import (
    DetectorGeometry,
    FrameRequest,
    GaussianRenderResult,
    GaussianWorkEstimate,
    HardEdgeColumnParameterization,
    propagate_gaussians_through_column,
    render_gaussian_detector_plane,
)


_POSITION = (0, 2)
_ANGLE = (1, 3)


def _readonly_array(value: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite array with shape {shape}.")
    array = array.copy()
    array.setflags(write=False)
    return array


def _position_blocks(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(matrix)
    return (
        values[np.ix_(_POSITION, _POSITION)],
        values[np.ix_(_POSITION, _ANGLE)],
        values[np.ix_(_ANGLE, _POSITION)],
        values[np.ix_(_ANGLE, _ANGLE)],
    )


def _closest_proper_rotation(block: np.ndarray) -> np.ndarray:
    left, _singular, right_t = np.linalg.svd(np.asarray(block, dtype=float))
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    return rotation


def _round_structure_defect(matrix: np.ndarray, rotation: np.ndarray) -> float:
    residuals = []
    blocks = _position_blocks(matrix)
    matrix_scale = max(float(np.linalg.norm(matrix, ord="fro")), 1.0)
    for block in blocks:
        coefficient = 0.5 * float(np.trace(rotation.T @ block))
        residuals.append(np.linalg.norm(block - coefficient * rotation, ord="fro"))
    return float(np.sqrt(np.sum(np.square(residuals))) / matrix_scale)


def _stable_mode_signature(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    dacs: np.ndarray,
    target_camera_length_m: float,
    adjusted_channel_names: Sequence[str],
) -> str:
    payload = {
        "schema": "temgym-native-diffraction-mode-v1",
        "channel_names": list(column.channel_names),
        "adjusted_channel_names": list(adjusted_channel_names),
        "target_camera_length_m": float(target_camera_length_m),
        "sample_z_m": float(column.sample_z_m),
        "detector_z_m": float(column.detector_z_m),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for name, values in (("parameters", parameters), ("raw_dacs", dacs)):
        array = np.ascontiguousarray(values, dtype="<f8")
        digest.update(name.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return f"native-hard-edge-diffraction-v1:{digest.hexdigest()}"


@dataclass(frozen=True)
class NativeDiffractionMode:
    """One solved physical-column diffraction mode and its diagnostics."""

    target_camera_length_m: float
    achieved_camera_length_m: float
    diffraction_rotation_rad: float
    raw_dac_values: np.ndarray
    adjusted_channel_names: tuple[str, ...]
    sample_to_detector_matrix: np.ndarray
    diffraction_rotation_matrix: np.ndarray
    a_block_norm: float
    camera_length_relative_error: float
    b_rotation_relative_residual: float
    round_structure_defect: float
    symplectic_defect_norm: float
    constraint_residual: np.ndarray
    constraint_jacobian_wrt_dacs: np.ndarray
    constraint_jacobian_wrt_parameters: np.ndarray
    active_bound_mask: np.ndarray
    optimizer_evaluations: int
    optimizer_status: int
    feasible: bool
    failure_reasons: tuple[str, ...]
    model_signature: str

    def __post_init__(self) -> None:
        target = float(self.target_camera_length_m)
        achieved = float(self.achieved_camera_length_m)
        scalars = (
            target,
            achieved,
            self.diffraction_rotation_rad,
            self.a_block_norm,
            self.camera_length_relative_error,
            self.b_rotation_relative_residual,
            self.round_structure_defect,
            self.symplectic_defect_norm,
        )
        if not np.all(np.isfinite(scalars)) or target <= 0.0 or achieved <= 0.0:
            raise ValueError("Native diffraction diagnostics must be finite and positive where required.")
        names = tuple(str(name) for name in self.adjusted_channel_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("adjusted_channel_names must be non-empty and unique.")
        dacs = np.asarray(self.raw_dac_values, dtype=float)
        parameter_jacobian = np.asarray(self.constraint_jacobian_wrt_parameters, dtype=float)
        object.__setattr__(
            self,
            "raw_dac_values",
            _readonly_array(dacs, shape=dacs.shape, name="raw_dac_values"),
        )
        object.__setattr__(
            self,
            "sample_to_detector_matrix",
            _readonly_array(
                self.sample_to_detector_matrix,
                shape=(4, 4),
                name="sample_to_detector_matrix",
            ),
        )
        object.__setattr__(
            self,
            "diffraction_rotation_matrix",
            _readonly_array(
                self.diffraction_rotation_matrix,
                shape=(2, 2),
                name="diffraction_rotation_matrix",
            ),
        )
        object.__setattr__(
            self,
            "constraint_residual",
            _readonly_array(self.constraint_residual, shape=(3,), name="constraint_residual"),
        )
        object.__setattr__(
            self,
            "constraint_jacobian_wrt_dacs",
            _readonly_array(
                self.constraint_jacobian_wrt_dacs,
                shape=(3, dacs.size),
                name="constraint_jacobian_wrt_dacs",
            ),
        )
        if parameter_jacobian.ndim != 2 or parameter_jacobian.shape[0] != 3:
            raise ValueError("constraint_jacobian_wrt_parameters must have shape (3, n_parameters).")
        object.__setattr__(
            self,
            "constraint_jacobian_wrt_parameters",
            _readonly_array(
                parameter_jacobian,
                shape=parameter_jacobian.shape,
                name="constraint_jacobian_wrt_parameters",
            ),
        )
        bound_mask = np.asarray(self.active_bound_mask, dtype=bool)
        if bound_mask.shape != (len(names),):
            raise ValueError("active_bound_mask must have one value per adjusted channel.")
        bound_mask = bound_mask.copy()
        bound_mask.setflags(write=False)
        object.__setattr__(self, "active_bound_mask", bound_mask)
        object.__setattr__(self, "adjusted_channel_names", names)
        object.__setattr__(self, "failure_reasons", tuple(str(v) for v in self.failure_reasons))
        signature = str(self.model_signature).strip()
        if not signature:
            raise ValueError("model_signature must be non-empty.")
        object.__setattr__(self, "model_signature", signature)

    @property
    def a_block(self) -> np.ndarray:
        return _position_blocks(self.sample_to_detector_matrix)[0]

    @property
    def b_block(self) -> np.ndarray:
        return _position_blocks(self.sample_to_detector_matrix)[1]

    @property
    def b_target_block(self) -> np.ndarray:
        return self.target_camera_length_m * self.diffraction_rotation_matrix

@dataclass(frozen=True)
class NativeDiffractionDesign:
    """A continuation-ordered collection of native diffraction modes."""

    modes: tuple[NativeDiffractionMode, ...]

    def __post_init__(self) -> None:
        modes = tuple(self.modes)
        if not modes:
            raise ValueError("NativeDiffractionDesign requires at least one mode.")
        object.__setattr__(self, "modes", modes)

    @property
    def all_targets_feasible(self) -> bool:
        return all(mode.feasible for mode in self.modes)

    @property
    def target_camera_lengths_m(self) -> np.ndarray:
        return np.asarray([mode.target_camera_length_m for mode in self.modes])

    @property
    def raw_dac_values(self) -> np.ndarray:
        return np.stack([mode.raw_dac_values for mode in self.modes])

    @property
    def maximum_a_block_norm(self) -> float:
        return max(mode.a_block_norm for mode in self.modes)

    @property
    def maximum_camera_length_relative_error(self) -> float:
        return max(mode.camera_length_relative_error for mode in self.modes)


class NativeDiffractionInfeasibleError(RuntimeError):
    """One or more requested camera lengths are not reachable in the DAC box."""

    def __init__(self, design: NativeDiffractionDesign):
        self.design = design
        details = ["Native hard-edge diffraction design is infeasible within the declared DAC bounds."]
        for mode in design.modes:
            if mode.feasible:
                continue
            details.append(
                f"L={mode.target_camera_length_m:.9g} m: "
                f"A_norm={mode.a_block_norm:.6g}, "
                f"camera_relative_error={mode.camera_length_relative_error:.6g}, "
                f"reasons={','.join(mode.failure_reasons)}"
            )
        super().__init__("\n".join(details))


def _active_bounds(
    bounds: tuple[float | Sequence[float], float | Sequence[float]],
    *,
    n_channels: int,
    active_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    converted = []
    for label, value in zip(("lower", "upper"), bounds, strict=True):
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            array = np.full(active_indices.size, float(array))
        elif array.shape == (n_channels,):
            array = array[active_indices]
        elif array.shape != (active_indices.size,):
            raise ValueError(
                f"{label} DAC bound must be scalar, one value per channel, or one value per adjusted channel."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{label} DAC bounds must be finite.")
        converted.append(array)
    lower, upper = converted
    if np.any(lower >= upper):
        raise ValueError("Every lower DAC bound must be strictly below its upper bound.")
    return lower, upper


def _validate_adjusted_channels(
    column: HardEdgeColumnParameterization,
    adjusted_channel_names: Sequence[str],
) -> tuple[tuple[str, ...], np.ndarray]:
    names = tuple(str(name) for name in adjusted_channel_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("adjusted_channel_names must be non-empty and unique.")
    indices = []
    controlled_downstream = {
        column.channel_names[lens.channel_index]
        for lens in column.lenses
        if lens.channel_index is not None and lens.entrance_z >= column.sample_z_m
    }
    for name in names:
        if name not in column.channel_names:
            raise ValueError(f"Unknown DAC channel {name!r}.")
        if name not in controlled_downstream:
            raise ValueError(f"DAC channel {name!r} is not a controllable downstream lens.")
        indices.append(column.channel_names.index(name))
    return names, np.asarray(indices, dtype=int)


def _candidate_seeds(
    base: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    multistart_count: int,
    continuation_seed: np.ndarray | None,
) -> tuple[np.ndarray, ...]:
    seeds = [np.clip(base, lower, upper)]
    if continuation_seed is not None:
        seeds.insert(0, np.clip(continuation_seed, lower, upper))
    span = upper - lower
    local_step = np.minimum(0.05 * span, 10.0)
    for index in range(base.size):
        direction = np.zeros_like(base)
        direction[index] = local_step[index]
        seeds.append(np.clip(base + direction, lower, upper))
        seeds.append(np.clip(base - direction, lower, upper))
    unit = qmc.Halton(base.size, scramble=False).random(int(multistart_count))
    seeds.extend(lower + unit * span)
    unique: list[np.ndarray] = []
    for seed in seeds:
        if any(np.allclose(seed, prior, rtol=0.0, atol=1e-12) for prior in unique):
            continue
        unique.append(np.asarray(seed, dtype=float))
    return tuple(unique)


def solve_native_diffraction_modes(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    target_camera_lengths_m: Sequence[float],
    *,
    base_dac_values: Sequence[float] | np.ndarray,
    adjusted_channel_names: Sequence[str] = ("IL1", "PL1"),
    dac_bounds: tuple[float | Sequence[float], float | Sequence[float]] = (-200.0, 200.0),
    multistart_count: int = 16,
    maximum_function_evaluations: int = 1200,
    a_block_norm_atol: float = 1e-7,
    camera_length_rtol: float = 1e-7,
    b_rotation_relative_atol: float = 1e-9,
    round_structure_atol: float = 1e-9,
    symplectic_defect_atol: float = 2e-7,
    require_feasible: bool = True,
) -> NativeDiffractionDesign:
    """Solve physical downstream DACs for one or more camera lengths.

    ``base_dac_values`` may be one shared row or one row per requested mode.
    Only ``adjusted_channel_names`` are changed, so calibrated illumination
    settings upstream of the specimen remain exactly those in each input row.
    Targets are solved in increasing camera-length order for continuation and
    returned in the caller's original order.  Deterministic Halton multistarts
    protect against the many periodic hard-edge branches.
    """

    parameter_values = np.asarray(parameters, dtype=float)
    if parameter_values.shape != (column.n_parameters,) or not np.all(np.isfinite(parameter_values)):
        raise ValueError(f"parameters must be finite with shape ({column.n_parameters},).")
    targets = np.asarray(target_camera_lengths_m, dtype=float)
    if targets.ndim != 1 or targets.size == 0 or not np.all(np.isfinite(targets)) or np.any(targets <= 0.0):
        raise ValueError("target_camera_lengths_m must be a finite positive vector.")
    names, active = _validate_adjusted_channels(column, adjusted_channel_names)
    channel_count = len(column.channel_names)
    base_rows = np.asarray(base_dac_values, dtype=float)
    if base_rows.shape == (channel_count,):
        base_rows = np.broadcast_to(base_rows, (targets.size, channel_count)).copy()
    if base_rows.shape != (targets.size, channel_count) or not np.all(np.isfinite(base_rows)):
        raise ValueError(
            "base_dac_values must be one finite channel row or one row per requested camera length."
        )
    if isinstance(multistart_count, (bool, np.bool_)) or int(multistart_count) != multistart_count or multistart_count < 1:
        raise ValueError("multistart_count must be a positive integer.")
    if (
        isinstance(maximum_function_evaluations, (bool, np.bool_))
        or int(maximum_function_evaluations) != maximum_function_evaluations
        or maximum_function_evaluations < 10
    ):
        raise ValueError("maximum_function_evaluations must be an integer of at least ten.")
    tolerances = np.asarray(
        (
            a_block_norm_atol,
            camera_length_rtol,
            b_rotation_relative_atol,
            round_structure_atol,
            symplectic_defect_atol,
        ),
        dtype=float,
    )
    if not np.all(np.isfinite(tolerances)) or np.any(tolerances < 0.0):
        raise ValueError("Diffraction-mode tolerances must be finite and non-negative.")
    lower, upper = _active_bounds(
        dac_bounds,
        n_channels=channel_count,
        active_indices=active,
    )

    active_jax = jnp.asarray(active)
    position_jax = jnp.asarray(_POSITION)
    angle_jax = jnp.asarray(_ANGLE)
    fixed_parameters = jnp.asarray(parameter_values)

    def normalized_constraints(
        active_values: jnp.ndarray,
        target: jnp.ndarray,
        base_row: jnp.ndarray,
    ) -> jnp.ndarray:
        dacs = base_row.at[active_jax].set(active_values)
        matrix = column.matrix(fixed_parameters, dacs, segment="sample_to_detector")
        a_block = matrix[jnp.ix_(position_jax, position_jax)]
        b_block = matrix[jnp.ix_(position_jax, angle_jax)]
        b_length = jnp.sqrt(0.5 * jnp.sum(b_block * b_block))
        return jnp.asarray((a_block[0, 0], a_block[1, 0], b_length / target - 1.0))

    compiled_constraints = jax.jit(normalized_constraints)
    compiled_jacobian = jax.jit(jax.jacfwd(normalized_constraints, argnums=0))

    modes_by_original_index: list[NativeDiffractionMode | None] = [None] * targets.size
    continuation_seed: np.ndarray | None = None
    for original_index in np.argsort(targets, kind="stable"):
        target = float(targets[original_index])
        base_row = np.asarray(base_rows[original_index], dtype=float)
        active_base = base_row[active]

        def objective(candidate: np.ndarray) -> np.ndarray:
            return np.asarray(compiled_constraints(candidate, target, base_row), dtype=float)

        def objective_jacobian(candidate: np.ndarray) -> np.ndarray:
            return np.asarray(compiled_jacobian(candidate, target, base_row), dtype=float)

        best_key: tuple[float, ...] | None = None
        best_result = None
        best_mode_diagnostics = None
        for seed in _candidate_seeds(
            active_base,
            lower,
            upper,
            int(multistart_count),
            continuation_seed,
        ):
            result = least_squares(
                objective,
                seed,
                jac=objective_jacobian,
                bounds=(lower, upper),
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
                max_nfev=int(maximum_function_evaluations),
            )
            dacs = base_row.copy()
            dacs[active] = np.asarray(result.x, dtype=float)
            matrix = np.asarray(
                column.matrix(parameter_values, dacs, segment="sample_to_detector"),
                dtype=float,
            )
            a_block, b_block, _c_block, _d_block = _position_blocks(matrix)
            a_norm = float(np.linalg.norm(a_block, ord="fro") / np.sqrt(2.0))
            achieved_length = float(np.linalg.norm(b_block, ord="fro") / np.sqrt(2.0))
            rotation = _closest_proper_rotation(b_block)
            b_rotation_residual = float(
                np.linalg.norm(b_block - achieved_length * rotation, ord="fro")
                / (np.sqrt(2.0) * target)
            )
            camera_error = abs(achieved_length / target - 1.0)
            round_defect = _round_structure_defect(matrix, rotation)
            symplectic_norm = float(
                np.linalg.norm(np.asarray(symplectic_defect(matrix)), ord="fro")
            )
            failures = tuple(
                label
                for label, failed in (
                    ("sample_conjugacy_A", a_norm > a_block_norm_atol),
                    ("camera_length", camera_error > camera_length_rtol),
                    ("B_scaled_rotation", b_rotation_residual > b_rotation_relative_atol),
                    ("round_column_structure", round_defect > round_structure_atol),
                    ("symplecticity", symplectic_norm > symplectic_defect_atol),
                )
                if failed
            )
            feasible = not failures
            constraint_norm = float(np.linalg.norm(objective(result.x)))
            reference = active_base if continuation_seed is None else continuation_seed
            scale = upper - lower
            # Once a candidate satisfies the declared physical tolerances,
            # select the nearest continuation branch rather than rewarding
            # meaningless differences at the 1e-14 optimizer floor.
            key = (
                0.0 if feasible else 1.0,
                0.0 if feasible else constraint_norm,
                float(np.linalg.norm((result.x - reference) / scale)),
                float(np.linalg.norm((result.x - active_base) / scale)),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_result = result
                best_mode_diagnostics = (
                    dacs,
                    matrix,
                    a_norm,
                    achieved_length,
                    rotation,
                    b_rotation_residual,
                    camera_error,
                    round_defect,
                    symplectic_norm,
                    feasible,
                    failures,
                )
        if best_result is None or best_mode_diagnostics is None:
            raise RuntimeError("No native diffraction optimization candidate was produced.")
        (
            dacs,
            matrix,
            a_norm,
            achieved_length,
            rotation,
            b_rotation_residual,
            camera_error,
            round_defect,
            symplectic_norm,
            feasible,
            failures,
        ) = best_mode_diagnostics
        continuation_seed = np.asarray(best_result.x, dtype=float)

        dacs_jax = jnp.asarray(dacs)
        target_jax = jnp.asarray(target, dtype=dacs_jax.dtype)

        def raw_constraints_from_dacs(all_dacs: jnp.ndarray) -> jnp.ndarray:
            native_matrix = column.matrix(
                fixed_parameters, all_dacs, segment="sample_to_detector"
            )
            a = native_matrix[jnp.ix_(position_jax, position_jax)]
            b = native_matrix[jnp.ix_(position_jax, angle_jax)]
            length = jnp.sqrt(0.5 * jnp.sum(b * b))
            return jnp.asarray((a[0, 0], a[1, 0], length - target_jax))

        def raw_constraints_from_parameters(parameter_row: jnp.ndarray) -> jnp.ndarray:
            native_matrix = column.matrix(
                parameter_row, dacs_jax, segment="sample_to_detector"
            )
            a = native_matrix[jnp.ix_(position_jax, position_jax)]
            b = native_matrix[jnp.ix_(position_jax, angle_jax)]
            length = jnp.sqrt(0.5 * jnp.sum(b * b))
            return jnp.asarray((a[0, 0], a[1, 0], length - target_jax))

        residual = np.asarray(raw_constraints_from_dacs(dacs_jax), dtype=float)
        dac_jacobian = np.asarray(jax.jacfwd(raw_constraints_from_dacs)(dacs_jax), dtype=float)
        parameter_jacobian = np.asarray(
            jax.jacfwd(raw_constraints_from_parameters)(fixed_parameters), dtype=float
        )
        active_bound_mask = np.isclose(best_result.x, lower, rtol=0.0, atol=1e-9) | np.isclose(
            best_result.x, upper, rtol=0.0, atol=1e-9
        )
        mode = NativeDiffractionMode(
            target_camera_length_m=target,
            achieved_camera_length_m=achieved_length,
            diffraction_rotation_rad=float(np.arctan2(rotation[1, 0], rotation[0, 0])),
            raw_dac_values=dacs,
            adjusted_channel_names=names,
            sample_to_detector_matrix=matrix,
            diffraction_rotation_matrix=rotation,
            a_block_norm=a_norm,
            camera_length_relative_error=camera_error,
            b_rotation_relative_residual=b_rotation_residual,
            round_structure_defect=round_defect,
            symplectic_defect_norm=symplectic_norm,
            constraint_residual=residual,
            constraint_jacobian_wrt_dacs=dac_jacobian,
            constraint_jacobian_wrt_parameters=parameter_jacobian,
            active_bound_mask=active_bound_mask,
            optimizer_evaluations=int(best_result.nfev),
            optimizer_status=int(best_result.status),
            feasible=feasible,
            failure_reasons=failures,
            model_signature=_stable_mode_signature(
                column,
                parameter_values,
                dacs,
                target,
                names,
            ),
        )
        modes_by_original_index[original_index] = mode

    design = NativeDiffractionDesign(
        modes=tuple(mode for mode in modes_by_original_index if mode is not None)
    )
    if len(design.modes) != targets.size:
        raise RuntimeError("Internal diffraction design ordering error.")
    if require_feasible and not design.all_targets_feasible:
        raise NativeDiffractionInfeasibleError(design)
    return design


def solve_native_diffraction_mode(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    target_camera_length_m: float,
    *,
    base_dac_values: Sequence[float],
    **kwargs: object,
) -> NativeDiffractionMode:
    """Convenience wrapper for a single physical-column diffraction mode."""

    return solve_native_diffraction_modes(
        column,
        parameters,
        (target_camera_length_m,),
        base_dac_values=base_dac_values,
        **kwargs,
    ).modes[0]


def verify_native_diffraction_mode(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    mode: NativeDiffractionMode,
    *,
    matrix_rtol: float = 2e-10,
    matrix_atol: float = 2e-12,
) -> np.ndarray:
    """Recompute and verify that ``mode`` belongs to this exact native column."""

    recomputed = np.asarray(
        column.matrix(parameters, mode.raw_dac_values, segment="sample_to_detector"),
        dtype=float,
    )
    if not np.allclose(
        recomputed,
        mode.sample_to_detector_matrix,
        rtol=matrix_rtol,
        atol=matrix_atol,
    ):
        raise ValueError(
            "The diffraction mode is stale or belongs to a different hard-edge column/parameter state."
        )
    if not mode.feasible:
        raise ValueError(
            "The diffraction mode failed its physical feasibility checks: "
            + ", ".join(mode.failure_reasons)
        )
    return recomputed


@dataclass(frozen=True)
class BoundNativeDiffractionRequest:
    """A diffraction request paired with its solved native-column mode."""

    request: FrameRequest
    mode: NativeDiffractionMode

    @property
    def render_metadata(self) -> Mapping[str, object]:
        metadata = {
            "schema": "temgym-native-diffraction-request-v1",
            "frame_id": int(self.request.frame_id),
            "model_scope": "neoarm_physical_column",
            "renderer_family": "hard_edge_native_gaussian",
            "downstream_diffraction_model": "exact HardEdgeColumnParameterization sample_to_detector segment",
            "target_camera_length_m": float(self.mode.target_camera_length_m),
            "achieved_camera_length_m": float(self.mode.achieved_camera_length_m),
            "diffraction_rotation_rad": float(self.mode.diffraction_rotation_rad),
            "a_block_norm": float(self.mode.a_block_norm),
            "camera_length_relative_error": float(self.mode.camera_length_relative_error),
            "b_rotation_relative_residual": float(self.mode.b_rotation_relative_residual),
            "round_structure_defect": float(self.mode.round_structure_defect),
            "symplectic_defect_norm": float(self.mode.symplectic_defect_norm),
            "adjusted_channel_names": self.mode.adjusted_channel_names,
            "combined_raw_dac_values": tuple(float(v) for v in self.mode.raw_dac_values),
            "native_mode_signature": self.mode.model_signature,
            "uses_ideal_relay": False,
        }
        return MappingProxyType(metadata)


def _camera_lengths_for_requests(
    requests: Sequence[FrameRequest],
    values: float | Sequence[float] | Mapping[int | str, float],
) -> np.ndarray:
    if isinstance(values, Mapping):
        resolved = []
        for request in requests:
            if request.frame_id in values:
                resolved.append(values[request.frame_id])
            elif request.optical_state_label in values:
                resolved.append(values[request.optical_state_label])
            else:
                raise KeyError(
                    "No target camera length for diffraction frame "
                    f"{request.frame_id} ({request.optical_state_label!r})."
                )
        array = np.asarray(resolved, dtype=float)
    else:
        array = np.asarray(values, dtype=float)
        if array.ndim == 0:
            array = np.full(len(requests), float(array))
    if array.shape != (len(requests),) or not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError("Target camera lengths must be finite, positive, and match the request count.")
    return array


def bind_native_diffraction_requests(
    requests: Sequence[FrameRequest],
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    target_camera_lengths_m: float | Sequence[float] | Mapping[int | str, float],
    **solver_kwargs: object,
) -> tuple[BoundNativeDiffractionRequest, ...]:
    """Replace only downstream DACs in diffraction :class:`FrameRequest`s.

    The old endpoint-to-approach-start displacement is retained exactly, so
    the manifest's forward/reverse preconditioning contract still applies to
    the solved endpoint.  Each binding exposes serialisable render metadata
    that explicitly declares use of the native hard-edge column and absence of
    an ideal relay.
    """

    request_tuple = tuple(requests)
    if not request_tuple:
        raise ValueError("At least one diffraction FrameRequest is required.")
    if any(request.measurement != "indexed_diffraction" for request in request_tuple):
        raise ValueError("bind_native_diffraction_requests accepts only indexed_diffraction rows.")
    if any(len(request.raw_dac_values) != len(column.channel_names) for request in request_tuple):
        raise ValueError("Every request must contain one raw DAC value per column channel.")
    targets = _camera_lengths_for_requests(request_tuple, target_camera_lengths_m)
    base_rows = np.asarray([request.raw_dac_values for request in request_tuple], dtype=float)
    design = solve_native_diffraction_modes(
        column,
        parameters,
        targets,
        base_dac_values=base_rows,
        **solver_kwargs,
    )
    bindings = []
    for request, mode in zip(request_tuple, design.modes, strict=True):
        old_endpoint = np.asarray(request.raw_dac_values, dtype=float)
        approach_delta = np.asarray(request.dac_approach_start_values, dtype=float) - old_endpoint
        new_endpoint = np.asarray(mode.raw_dac_values, dtype=float)
        bound_request = replace(
            request,
            raw_dac_values=tuple(float(value) for value in new_endpoint),
            dac_approach_start_values=tuple(float(value) for value in new_endpoint + approach_delta),
        )
        bindings.append(BoundNativeDiffractionRequest(bound_request, mode))
    return tuple(bindings)


def propagate_native_diffraction_gaussians(
    beam_at_sample: GaussianBeam,
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    mode: NativeDiffractionMode,
    *,
    slices_per_lens: int = 64,
) -> GaussianBeam:
    """Propagate a sample-plane bundle through the solved physical column."""

    verify_native_diffraction_mode(column, parameters, mode)
    return propagate_gaussians_through_column(
        beam_at_sample,
        column,
        parameters,
        mode.raw_dac_values,
        segment="sample_to_detector",
        slices_per_lens=slices_per_lens,
    )


def render_native_diffraction_detector(
    beam_at_sample: GaussianBeam,
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    mode: NativeDiffractionMode,
    detector: DetectorGeometry,
    *,
    slices_per_lens: int = 64,
    tile_rows: int = 32,
    beam_batch_size: int = 128,
    method: str = "auto",
    normalise: bool = False,
    workload: GaussianWorkEstimate | None = None,
) -> GaussianRenderResult:
    """Render a diffraction field using only the native downstream lenses."""

    detector_beam = propagate_native_diffraction_gaussians(
        beam_at_sample,
        column,
        parameters,
        mode,
        slices_per_lens=slices_per_lens,
    )
    return render_gaussian_detector_plane(
        detector_beam,
        detector,
        tile_rows=tile_rows,
        beam_batch_size=beam_batch_size,
        method=method,
        normalise=normalise,
        workload=workload,
    )


__all__ = [
    "NativeDiffractionMode",
    "NativeDiffractionDesign",
    "NativeDiffractionInfeasibleError",
    "BoundNativeDiffractionRequest",
    "solve_native_diffraction_mode",
    "solve_native_diffraction_modes",
    "verify_native_diffraction_mode",
    "bind_native_diffraction_requests",
    "propagate_native_diffraction_gaussians",
    "render_native_diffraction_detector",
]
