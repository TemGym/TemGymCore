"""Raw-DAC critical-line and aperture-throughput calibration utilities.

The routines in this module turn the most useful part of a coupled lens sweep
into explicit observables: signed beam-size changes under a small lens wobble,
and absolute current transmitted through a known circular aperture.  They do
not require electrical-current readback.  The independent variable is the
same raw (or normalized raw) DAC coordinate recorded in the acquisition
manifest, and the fitted optical law maps that coordinate directly to the
dimensionless integrated Larmor strength ``q``.

Consequently, this module can identify a predictive ``DAC -> q`` law but
cannot factor it into DAC-to-current gain and current-to-field calibration:
``I -> c I`` and ``field_per_amp -> field_per_amp / c`` is an exact gauge.
Changing the declared DAC reference likewise trades intercept against slope.
Longitudinal hard-edge geometry is an effective representation; external
first-order images cannot prove that the real axial field is rectangular.

All phase-space states use the TemGymCore ordering
``[x, theta_x, y, theta_y]``.  These first-order observables are intended for
experiment design and robust initialization; wave-optical specimen images
remain the held-out validation of the fitted column.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from .microscope_inversion import drift_matrix, hard_edge_solenoid_matrix
from .neoarm_calibration import HardEdgeColumnParameterization
from .physical_nuisance import (
    BlockwiseIdentifiabilityDiagnostics,
    diagnose_blockwise_identifiability,
)


_POSITION = jnp.asarray((0, 2))

CONDENSER_PROBE_OBSERVABLE_NAMES = (
    "sample_centroid_x_m",
    "sample_centroid_y_m",
    "sample_mean_tilt_x_rad",
    "sample_mean_tilt_y_rad",
    "log_sample_rms_radius_m",
    "log_sample_rms_angle_rad",
    "sample_position_angle_correlation",
    "sample_canonical_angular_momentum_correlation",
)

PHASE_SPACE_COVARIANCE_OBSERVABLE_NAMES = (
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

POSITION_COVARIANCE_OBSERVABLE_NAMES = (
    "log_rms_x_m",
    "log_rms_y_m",
    "correlation_x_y",
)

DEFLECTOR_SAMPLE_CENTRAL_DIFFERENCE_OBSERVABLE_NAMES = (
    "sample_central_half_difference_x_m",
    "sample_central_half_difference_theta_x_rad",
    "sample_central_half_difference_y_m",
    "sample_central_half_difference_theta_y_rad",
)

DEFLECTOR_SAMPLE_POSITION_DIFFERENCE_OBSERVABLE_NAMES = (
    DEFLECTOR_SAMPLE_CENTRAL_DIFFERENCE_OBSERVABLE_NAMES[0],
    DEFLECTOR_SAMPLE_CENTRAL_DIFFERENCE_OBSERVABLE_NAMES[2],
)

_ANGULAR_KICK_BASIS = jnp.asarray(
    (
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 0.0),
        (0.0, 1.0),
    )
)

_DEFLECTOR_DIFFERENCE_MODE_INDICES = {
    "position": (0, 2),
    "full": tuple(range(4)),
}

_PROBE_MODE_INDICES = {
    # A registered perforation edge scan can recover these three quantities.
    "position": (0, 1, 4),
    # A calibrated diffraction-plane exposure can recover these three.
    "angle": (2, 3, 5),
    # Position image + diffraction image + a defocus/complex-probe retrieval.
    "complex_radial": (4, 5, 6),
    # The last entry needs a calibrated coupled projection or an entrance basis;
    # ordinary round images leave it null (the 9/10 covariance-tomography mode).
    "full": tuple(range(8)),
}

_COVARIANCE_MODE_INDICES = {
    "position": (0, 2, 5),
    "full": tuple(range(10)),
}

_PHASE_SPACE_CORRELATION_INDICES = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
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


def _orthonormal_symmetric_basis_4x4() -> np.ndarray:
    """Return the canonical Frobenius-orthonormal symmetric 4x4 basis."""

    basis: list[np.ndarray] = []
    for index in range(4):
        element = np.zeros((4, 4), dtype=float)
        element[index, index] = 1.0
        basis.append(element)
    for row, column in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        element = np.zeros((4, 4), dtype=float)
        element[row, column] = element[column, row] = 1.0 / np.sqrt(2.0)
        basis.append(element)
    return np.stack(basis)


@dataclass(frozen=True)
class EntranceCovarianceChart:
    r"""A global ten-coordinate SPD chart about an entrance covariance.

    Let ``C0 = L L.T`` be the Cholesky factorization of the calibrated
    reference covariance and let ``H(eta)`` be a symmetric matrix expanded in
    a Frobenius-orthonormal basis.  This chart evaluates

    ``C(eta) = L exp(H(eta)) L.T``.

    It recovers ``C0`` at the origin and remains symmetric positive definite
    for every finite, numerically representable coordinate vector.  The
    coordinates are dimensionless perturbations in the whitened reference
    phase space, which avoids mixing metre and radian covariance entries in a
    nonlinear optimizer.  The returned array is built entirely with JAX and
    can be used directly inside ``jit``, ``jacfwd``, or a nonlinear residual.
    """

    reference_covariance: np.ndarray
    coordinate_names: tuple[str, ...] = ENTRANCE_COVARIANCE_COORDINATE_NAMES
    cholesky_factor: np.ndarray = field(init=False, repr=False)
    symmetric_basis: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        covariance = _state_covariance(self.reference_covariance)
        covariance = np.asarray(covariance, dtype=float)
        names = tuple(str(name).strip() for name in self.coordinate_names)
        if (
            len(names) != 10
            or any(not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(
                "coordinate_names must contain ten unique non-empty names."
            )
        cholesky = np.linalg.cholesky(covariance)
        basis = _orthonormal_symmetric_basis_4x4()
        for field_name, values in (
            ("reference_covariance", covariance),
            ("cholesky_factor", cholesky),
            ("symmetric_basis", basis),
        ):
            frozen = np.asarray(values, dtype=float).copy()
            frozen.setflags(write=False)
            object.__setattr__(self, field_name, frozen)
        object.__setattr__(self, "coordinate_names", names)

    @property
    def n_coordinates(self) -> int:
        return 10

    def covariance(self, coordinates: np.ndarray) -> jnp.ndarray:
        """Decode dimensionless chart coordinates into a 4x4 SPD covariance."""

        eta = jnp.asarray(coordinates)
        if eta.shape != (self.n_coordinates,):
            raise ValueError(
                f"coordinates must have shape ({self.n_coordinates},)."
            )
        if jnp.issubdtype(eta.dtype, jnp.complexfloating):
            raise ValueError("coordinates must be real-valued.")
        eta = eta.astype(jnp.result_type(eta, jnp.asarray(1.0)))
        basis = jnp.asarray(self.symmetric_basis, dtype=eta.dtype)
        generator = jnp.einsum("k,kij->ij", eta, basis)
        reference_factor = jnp.asarray(self.cholesky_factor, dtype=eta.dtype)
        covariance = reference_factor @ jsp.linalg.expm(generator) @ reference_factor.T
        return 0.5 * (covariance + covariance.T)


def _state_vector(name: str, value: Sequence[float]) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (4,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite length-four state vector.")
    return array


def _state_covariance(value: np.ndarray) -> np.ndarray | jnp.ndarray:
    covariance = jnp.asarray(value)
    if covariance.shape != (4, 4):
        raise ValueError("source_covariance must be a finite 4x4 matrix.")
    # Numerical inputs receive the full public validation contract.  During a
    # JAX transformation the covariance may be a tracer produced by the SPD
    # chart above; converting that tracer to NumPy would break differentiation.
    # Shape is static and positive definiteness is guaranteed by the chart.
    if isinstance(covariance, jax.core.Tracer):
        return covariance
    covariance_np = np.asarray(covariance, dtype=float)
    if not np.all(np.isfinite(covariance_np)):
        raise ValueError("source_covariance must be a finite 4x4 matrix.")
    if not np.allclose(covariance_np, covariance_np.T, rtol=1e-10, atol=1e-20):
        raise ValueError("source_covariance must be symmetric.")
    if np.linalg.eigvalsh(covariance_np).min() <= 0.0:
        raise ValueError("source_covariance must be positive definite.")
    return covariance_np


def hard_edge_matrix_to_plane(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    dac_values: np.ndarray,
    plane_z_m: float,
) -> jnp.ndarray:
    """Return the source-to-internal-plane hard-edge transfer matrix.

    Lens membership is fixed by the initial geometry, just as it is in
    :meth:`HardEdgeColumnParameterization.matrix`.  The chosen plane must lie
    in a nominal drift gap rather than inside a lens body, so membership cannot
    switch inside a JIT-compiled fit.
    """

    plane = float(plane_z_m)
    if not np.isfinite(plane) or not column.source_z_m < plane <= column.detector_z_m:
        raise ValueError("plane_z_m must lie after the source and at/before the detector.")
    for lens in column.lenses:
        if lens.entrance_z < plane < lens.exit_z:
            raise ValueError(
                f"plane_z_m lies inside nominal lens body {lens.name!r}; use a drift gap."
            )
    values = jnp.asarray(parameters)
    dacs = jnp.asarray(dac_values)
    if values.shape != (column.n_parameters,):
        raise ValueError(f"parameters must have shape ({column.n_parameters},).")
    if dacs.shape != (len(column.channel_names),):
        raise ValueError(
            f"dac_values must have shape ({len(column.channel_names)},)."
        )
    total = jnp.eye(4, dtype=values.dtype)
    axial_cursor = values[0]
    for lens, centre, length, q_bias, q_slope in column._decoded_lenses(values):
        if lens.exit_z > plane:
            continue
        entrance = centre - 0.5 * length
        total = drift_matrix(entrance - axial_cursor) @ total
        displacement = (
            jnp.asarray(0.0, dtype=values.dtype)
            if lens.channel_index is None
            else dacs[lens.channel_index] - lens.reference_dac
        )
        q = q_bias + q_slope * displacement
        total = hard_edge_solenoid_matrix(q, length) @ total
        axial_cursor = centre + 0.5 * length
    return drift_matrix(jnp.asarray(plane, dtype=values.dtype) - axial_cursor) @ total


def hard_edge_matrix_between_planes(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    dac_values: np.ndarray,
    *,
    from_z_m: float,
    to_z_m: float,
) -> jnp.ndarray:
    """Return the first-order transfer between two fixed drift planes.

    Both planes use the same source-referenced hard-edge column, and the
    source dependence is then removed algebraically as

    ``M(from -> to) = M(source -> to) @ inv(M(source -> from))``.

    The solve is implemented with JAX rather than a materialized inverse, so
    the result remains differentiable with respect to every optical parameter
    and raw-DAC value.  Plane locations are deliberately static Python values:
    callers fitting lens geometry must separately certify that both locations
    stay outside every lens body over the complete parameter box (for example
    with :func:`validate_hard_edge_internal_plane_bounds`).
    """

    start = float(from_z_m)
    stop = float(to_z_m)
    if (
        not np.all(np.isfinite((start, stop)))
        or not column.source_z_m < start < stop <= column.detector_z_m
    ):
        raise ValueError(
            "Expected source_z_m < from_z_m < to_z_m <= detector_z_m."
        )
    source_to_start = hard_edge_matrix_to_plane(
        column, parameters, dac_values, start
    )
    source_to_stop = hard_edge_matrix_to_plane(
        column, parameters, dac_values, stop
    )
    # Right multiplication by inv(source_to_start), expressed as a solve.
    return jnp.linalg.solve(source_to_start.T, source_to_stop.T).T


def _known_deflector_planes(
    plane_z_m: float | Sequence[float],
    *,
    observation_z_m: float,
    column: HardEdgeColumnParameterization,
) -> np.ndarray:
    planes = np.atleast_1d(np.asarray(plane_z_m, dtype=float))
    if planes.ndim != 1 or planes.size not in (1, 2):
        raise ValueError("plane_z_m must describe exactly one or two planes.")
    if not np.all(np.isfinite(planes)) or np.any(np.diff(planes) <= 0.0):
        raise ValueError("Deflector planes must be finite and strictly increasing.")
    if not np.all(
        (planes > column.source_z_m) & (planes < float(observation_z_m))
    ):
        raise ValueError(
            "Every deflector plane must lie after the source and before the "
            "observation plane."
        )
    return planes


def ideal_angular_kick_response_matrices(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    dac_values: np.ndarray,
    plane_z_m: float | Sequence[float],
    *,
    observation_z_m: float | None = None,
) -> jnp.ndarray:
    r"""Return one/two ideal angular-kick responses at a downstream plane.

    The returned tensor has shape ``(n_planes, 4, 2)``.  Its final axis is the
    physical kick ``[delta_theta_x, delta_theta_y]`` in radians; its middle
    axis follows the TemGymCore state order ``[x, theta_x, y, theta_y]``.
    Therefore ``response[p] @ kick_xy[p]`` is the downstream phase-space
    displacement caused by deflector plane ``p``.  The specimen plane is the
    default observation plane.

    This is the ideal thin-kick model.  A finite coil length, hysteresis, or
    command electronics belongs in a separately declared calibration model,
    not in the condenser lens parameters.
    """

    observation = (
        float(column.sample_z_m)
        if observation_z_m is None
        else float(observation_z_m)
    )
    if (
        not np.isfinite(observation)
        or not column.source_z_m < observation <= column.detector_z_m
    ):
        raise ValueError("observation_z_m must lie inside the downstream column.")
    planes = _known_deflector_planes(
        plane_z_m, observation_z_m=observation, column=column
    )
    responses = []
    for plane in planes:
        transfer = hard_edge_matrix_between_planes(
            column,
            parameters,
            dac_values,
            from_z_m=float(plane),
            to_z_m=observation,
        )
        responses.append(
            transfer @ jnp.asarray(_ANGULAR_KICK_BASIS, dtype=transfer.dtype)
        )
    return jnp.stack(tuple(responses), axis=0)


def _runtime_real_array(name: str, value: np.ndarray) -> jnp.ndarray:
    """Validate numerical arrays without concretizing values under JAX."""

    array = jnp.asarray(value)
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise ValueError(f"{name} must be real-valued.")
    if not isinstance(array, jax.core.Tracer) and not np.all(
        np.isfinite(np.asarray(array))
    ):
        raise ValueError(f"{name} must contain only finite values.")
    return array.astype(jnp.result_type(array, jnp.asarray(1.0)))


@dataclass(frozen=True)
class DeflectorCommandKickMap:
    r"""Affine raw-command calibration for one or two thin-kick planes.

    ``gain_xy_rad_per_command[p, a, c]`` maps command channel ``c`` to the
    ``x``/``y`` angular kick at physical plane ``p``.  ``kick_bias_xy_rad`` is
    the command-independent kick.  Both are independent calibration nuisance
    parameters; they must not be silently absorbed into a condenser-lens DAC
    law.

    The flattened calibration parameter order is all kick biases in C order,
    followed by all gains in C order.  :attr:`parameter_names` records that
    order explicitly.  Passing this vector to :meth:`kicks` leaves the whole
    command-to-observable path compatible with ``jax.jit`` and ``jax.jacfwd``.
    Because the entries carry different physical scales, Fisher calculations
    should column-scale them by their calibration uncertainties.
    """

    plane_z_m: np.ndarray
    plane_names: tuple[str, ...]
    command_names: tuple[str, ...]
    reference_command: np.ndarray
    kick_bias_xy_rad: np.ndarray
    gain_xy_rad_per_command: np.ndarray

    def __post_init__(self) -> None:
        planes = np.atleast_1d(np.asarray(self.plane_z_m, dtype=float))
        plane_names = tuple(str(name).strip() for name in self.plane_names)
        command_names = tuple(str(name).strip() for name in self.command_names)
        reference = np.asarray(self.reference_command, dtype=float)
        bias = np.asarray(self.kick_bias_xy_rad, dtype=float)
        gain = np.asarray(self.gain_xy_rad_per_command, dtype=float)
        n_planes = planes.size
        n_commands = len(command_names)
        if planes.ndim != 1 or n_planes not in (1, 2):
            raise ValueError("plane_z_m must describe exactly one or two planes.")
        if not np.all(np.isfinite(planes)) or np.any(np.diff(planes) <= 0.0):
            raise ValueError("plane_z_m must be finite and strictly increasing.")
        if (
            len(plane_names) != n_planes
            or any(not name for name in plane_names)
            or len(set(plane_names)) != n_planes
        ):
            raise ValueError("plane_names must uniquely name every deflector plane.")
        if (
            n_commands < 1
            or any(not name for name in command_names)
            or len(set(command_names)) != n_commands
        ):
            raise ValueError("command_names must be non-empty and unique.")
        if reference.shape != (n_commands,):
            raise ValueError(f"reference_command must have shape ({n_commands},).")
        if bias.shape != (n_planes, 2):
            raise ValueError(
                f"kick_bias_xy_rad must have shape ({n_planes}, 2)."
            )
        if gain.shape != (n_planes, 2, n_commands):
            raise ValueError(
                "gain_xy_rad_per_command must have shape "
                f"({n_planes}, 2, {n_commands})."
            )
        if not all(np.all(np.isfinite(values)) for values in (reference, bias, gain)):
            raise ValueError("Deflector command-map arrays must contain finite values.")
        for field_name, values in (
            ("plane_z_m", planes),
            ("reference_command", reference),
            ("kick_bias_xy_rad", bias),
            ("gain_xy_rad_per_command", gain),
        ):
            frozen = np.asarray(values, dtype=float).copy()
            frozen.setflags(write=False)
            object.__setattr__(self, field_name, frozen)
        object.__setattr__(self, "plane_names", plane_names)
        object.__setattr__(self, "command_names", command_names)

    @property
    def n_planes(self) -> int:
        return len(self.plane_names)

    @property
    def n_commands(self) -> int:
        return len(self.command_names)

    @property
    def n_parameters(self) -> int:
        return 2 * self.n_planes * (1 + self.n_commands)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        axes = ("x", "y")
        names = [
            f"{plane}.kick_bias_{axis}_rad"
            for plane in self.plane_names
            for axis in axes
        ]
        names.extend(
            f"{plane}.kick_gain_{axis}_rad_per_{command}"
            for plane in self.plane_names
            for axis in axes
            for command in self.command_names
        )
        return tuple(names)

    @property
    def nominal_parameters(self) -> np.ndarray:
        values = np.concatenate(
            (self.kick_bias_xy_rad.reshape(-1), self.gain_xy_rad_per_command.reshape(-1))
        )
        values.setflags(write=False)
        return values

    def decode_parameters(
        self, calibration_parameters: np.ndarray | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Decode a differentiable flattened bias/gain nuisance vector."""

        values = _runtime_real_array(
            "calibration_parameters",
            self.nominal_parameters
            if calibration_parameters is None
            else calibration_parameters,
        )
        if values.shape != (self.n_parameters,):
            raise ValueError(
                f"calibration_parameters must have shape ({self.n_parameters},)."
            )
        split = 2 * self.n_planes
        bias = values[:split].reshape((self.n_planes, 2))
        gain = values[split:].reshape((self.n_planes, 2, self.n_commands))
        return bias, gain

    def kicks(
        self,
        command_values: np.ndarray,
        calibration_parameters: np.ndarray | None = None,
    ) -> jnp.ndarray:
        """Return kick arrays with shape ``(n_states, n_planes, 2)``."""

        commands = _runtime_real_array("command_values", command_values)
        if commands.ndim == 1:
            commands = commands[None, :]
        if commands.ndim != 2 or commands.shape[1] != self.n_commands:
            raise ValueError(
                "command_values must have shape (n_states, n_commands) or "
                "(n_commands,)."
            )
        bias, gain = self.decode_parameters(calibration_parameters)
        reference = jnp.asarray(self.reference_command, dtype=commands.dtype)
        displacement = commands - reference
        return bias[None, :, :] + jnp.einsum(
            "pac,nc->npa", gain, displacement
        )


class DeflectorCentralDifference(NamedTuple):
    """Source-cancelling central half difference at the specimen plane.

    ``mean_half_difference`` has shape ``(2,)`` in ``position`` mode and
    ``(4,)`` in ``full`` mode.  ``covariance_half_difference`` has the matching
    square shape and is identically zero for a deterministic ideal kick: both
    signed commands propagate the same entrance covariance.
    """

    mean_half_difference: jnp.ndarray
    covariance_half_difference: jnp.ndarray


def deflector_command_phase_space_moments(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    dac_values: np.ndarray,
    source_mean_state: Sequence[float],
    source_covariance: np.ndarray,
    command_values: np.ndarray,
    command_map: DeflectorCommandKickMap,
    *,
    calibration_parameters: np.ndarray | None = None,
    observation_z_m: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Predict mean/covariance for commanded ideal deflector exposures.

    The return shapes are ``(n_states, 4)`` and ``(n_states, 4, 4)``.  Commands
    alter only the mean in this first-order deterministic-kick model.  The
    covariance is repeated explicitly so a caller cannot accidentally treat a
    command-pair width change as independent condenser information.
    """

    if not isinstance(command_map, DeflectorCommandKickMap):
        raise TypeError("command_map must be a DeflectorCommandKickMap.")
    observation = (
        float(column.sample_z_m)
        if observation_z_m is None
        else float(observation_z_m)
    )
    # This also validates the known physical plane ordering.
    responses = ideal_angular_kick_response_matrices(
        column,
        parameters,
        dac_values,
        command_map.plane_z_m,
        observation_z_m=observation,
    )
    mean = _runtime_real_array("source_mean_state", source_mean_state)
    if mean.shape != (4,):
        raise ValueError("source_mean_state must have shape (4,).")
    covariance = jnp.asarray(_state_covariance(source_covariance))
    source_to_observation = hard_edge_matrix_to_plane(
        column, parameters, dac_values, observation
    )
    base_mean, propagated_covariance = propagate_phase_space_gaussian(
        source_to_observation, mean, covariance
    )
    kicks = command_map.kicks(command_values, calibration_parameters)
    displacements = jnp.einsum("pia,npa->ni", responses, kicks)
    output_mean = base_mean[None, :] + displacements
    output_covariance = jnp.broadcast_to(
        propagated_covariance,
        (output_mean.shape[0], 4, 4),
    )
    return output_mean, output_covariance


def central_deflector_command_difference(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    dac_values: np.ndarray,
    source_mean_state: Sequence[float],
    source_covariance: np.ndarray,
    centre_command: np.ndarray,
    command_delta: np.ndarray,
    command_map: DeflectorCommandKickMap,
    *,
    calibration_parameters: np.ndarray | None = None,
    mode: str = "position",
) -> DeflectorCentralDifference:
    r"""Return the specimen response to commands ``centre +/- delta``.

    The result is ``(plus - minus) / 2`` rather than a derivative because a
    multichannel ``command_delta`` has no unique scalar denominator.  For the
    affine kick map it equals the downstream response to ``gain @ delta``:
    entrance mean, entrance covariance, static deflector bias, and reference
    command all cancel.  Divide by a declared scalar step externally when a
    one-channel derivative is desired.
    """

    if not isinstance(command_map, DeflectorCommandKickMap):
        raise TypeError("command_map must be a DeflectorCommandKickMap.")
    if mode not in _DEFLECTOR_DIFFERENCE_MODE_INDICES:
        raise ValueError(
            "mode must be one of "
            f"{tuple(_DEFLECTOR_DIFFERENCE_MODE_INDICES)}, got {mode!r}."
        )
    centre = _runtime_real_array("centre_command", centre_command)
    delta = _runtime_real_array("command_delta", command_delta)
    expected_shape = (command_map.n_commands,)
    if centre.shape != expected_shape or delta.shape != expected_shape:
        raise ValueError(
            f"centre_command and command_delta must have shape {expected_shape}."
        )
    if not isinstance(delta, jax.core.Tracer) and not np.any(np.asarray(delta) != 0.0):
        raise ValueError("command_delta must contain at least one non-zero command.")
    # Validate the nuisance inputs that cancel from the observable.  The
    # central affine difference is then evaluated analytically rather than by
    # subtracting two potentially much larger centroids.
    source_mean = _runtime_real_array("source_mean_state", source_mean_state)
    if source_mean.shape != (4,):
        raise ValueError("source_mean_state must have shape (4,).")
    _state_covariance(source_covariance)
    responses = ideal_angular_kick_response_matrices(
        column,
        parameters,
        dac_values,
        observation_z_m=column.sample_z_m,
        plane_z_m=command_map.plane_z_m,
    )
    _, gain = command_map.decode_parameters(calibration_parameters)
    central_kick = jnp.einsum("pac,c->pa", gain, delta)
    mean_half_difference = jnp.einsum("pia,pa->i", responses, central_kick)
    covariance_half_difference = jnp.zeros(
        (4, 4), dtype=mean_half_difference.dtype
    )
    indices = jnp.asarray(_DEFLECTOR_DIFFERENCE_MODE_INDICES[mode])
    return DeflectorCentralDifference(
        mean_half_difference[indices],
        covariance_half_difference[jnp.ix_(indices, indices)],
    )


@dataclass(frozen=True)
class HardEdgeInternalPlaneBounds:
    """Independent parameter box certified around a fixed inter-lens plane."""

    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    plane_z_m: float
    upstream_lens_name: str
    downstream_lens_name: str
    minimum_clearance_m: float
    worst_case_upstream_exit_z_m: float
    worst_case_downstream_entrance_z_m: float
    was_restricted: bool = False

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower_bounds, dtype=float)
        upper = np.asarray(self.upper_bounds, dtype=float)
        if (
            lower.ndim != 1
            or upper.shape != lower.shape
            or np.any(np.isnan(lower))
            or np.any(np.isnan(upper))
            or np.any(lower > upper)
        ):
            raise ValueError("lower_bounds and upper_bounds must form a valid box.")
        values = (
            self.plane_z_m,
            self.minimum_clearance_m,
            self.worst_case_upstream_exit_z_m,
            self.worst_case_downstream_entrance_z_m,
        )
        if not np.all(np.isfinite(values)) or self.minimum_clearance_m < 0.0:
            raise ValueError("Plane-bound certificate values must be finite and valid.")
        upstream = str(self.upstream_lens_name).strip()
        downstream = str(self.downstream_lens_name).strip()
        if not upstream or not downstream or upstream == downstream:
            raise ValueError("Lens names must be distinct and non-empty.")
        tolerance = 32.0 * np.finfo(float).eps * max(
            1.0,
            abs(float(self.plane_z_m)),
            abs(float(self.worst_case_upstream_exit_z_m)),
            abs(float(self.worst_case_downstream_entrance_z_m)),
        )
        if (
            self.upstream_clearance_m + tolerance < self.minimum_clearance_m
            or self.downstream_clearance_m + tolerance < self.minimum_clearance_m
        ):
            raise ValueError("The parameter box does not satisfy its clearance claim.")
        lower = lower.copy()
        upper = upper.copy()
        lower.setflags(write=False)
        upper.setflags(write=False)
        object.__setattr__(self, "lower_bounds", lower)
        object.__setattr__(self, "upper_bounds", upper)
        object.__setattr__(self, "upstream_lens_name", upstream)
        object.__setattr__(self, "downstream_lens_name", downstream)
        object.__setattr__(self, "was_restricted", bool(self.was_restricted))

    @property
    def upstream_clearance_m(self) -> float:
        return float(self.plane_z_m - self.worst_case_upstream_exit_z_m)

    @property
    def downstream_clearance_m(self) -> float:
        return float(self.worst_case_downstream_entrance_z_m - self.plane_z_m)

    @property
    def worst_case_clearance_m(self) -> float:
        return min(self.upstream_clearance_m, self.downstream_clearance_m)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return mutable copies suitable for a numerical optimizer."""

        return self.lower_bounds.copy(), self.upper_bounds.copy()


class HardEdgeInternalPlaneConstraint(NamedTuple):
    """A fixed plane that must remain in one nominal inter-lens gap."""

    plane_z_m: float
    upstream_lens_name: str
    downstream_lens_name: str
    minimum_clearance_m: float = 0.0


def _hard_edge_parameter_box(
    column: HardEdgeColumnParameterization,
    bounds: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if len(bounds) != 2:
        raise ValueError("bounds must contain lower and upper arrays.")
    try:
        lower = np.broadcast_to(
            np.asarray(bounds[0], dtype=float), (column.n_parameters,)
        ).copy()
        upper = np.broadcast_to(
            np.asarray(bounds[1], dtype=float), (column.n_parameters,)
        ).copy()
    except ValueError as error:
        raise ValueError(
            f"bounds must broadcast to ({column.n_parameters},)."
        ) from error
    if (
        np.any(np.isnan(lower))
        or np.any(np.isnan(upper))
        or np.any(lower > upper)
    ):
        raise ValueError(
            "bounds must not contain NaNs and must satisfy lower <= upper."
        )
    return lower, upper


def _hard_edge_internal_plane_geometry(
    column: HardEdgeColumnParameterization,
    upstream_lens_name: str,
    downstream_lens_name: str,
) -> tuple[str, str, int, int, int, int]:
    upstream = str(upstream_lens_name).strip()
    downstream = str(downstream_lens_name).strip()
    lens_names = tuple(lens.name for lens in column.lenses)
    missing = tuple(name for name in (upstream, downstream) if name not in lens_names)
    if missing:
        raise ValueError(f"Unknown hard-edge lens names: {missing}.")
    upstream_index = lens_names.index(upstream)
    downstream_index = lens_names.index(downstream)
    if downstream_index != upstream_index + 1:
        raise ValueError("The named lenses must be adjacent and in axial order.")
    names = column.parameter_names
    return (
        upstream,
        downstream,
        names.index(f"{upstream}.z_center_m"),
        names.index(f"{upstream}.log_length_m"),
        names.index(f"{downstream}.z_center_m"),
        names.index(f"{downstream}.log_length_m"),
    )


def _internal_plane_bound_certificate(
    column: HardEdgeColumnParameterization,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    plane_z_m: float,
    upstream_lens_name: str,
    downstream_lens_name: str,
    minimum_clearance_m: float,
    was_restricted: bool,
) -> HardEdgeInternalPlaneBounds:
    plane = float(plane_z_m)
    clearance = float(minimum_clearance_m)
    if not np.isfinite(plane) or not column.source_z_m < plane < column.detector_z_m:
        raise ValueError("plane_z_m must lie strictly inside the nominal column.")
    if not np.isfinite(clearance) or clearance < 0.0:
        raise ValueError("minimum_clearance_m must be finite and non-negative.")
    upstream, downstream, up_z, up_log_l, down_z, down_log_l = (
        _hard_edge_internal_plane_geometry(
            column, upstream_lens_name, downstream_lens_name
        )
    )
    upstream_lens = column.lenses[
        tuple(lens.name for lens in column.lenses).index(upstream)
    ]
    downstream_lens = column.lenses[
        tuple(lens.name for lens in column.lenses).index(downstream)
    ]
    if not upstream_lens.exit_z <= plane <= downstream_lens.entrance_z:
        raise ValueError("The plane must lie in the named nominal inter-lens gap.")
    relevant = np.asarray(
        (
            lower[up_z], upper[up_z], lower[up_log_l], upper[up_log_l],
            lower[down_z], upper[down_z], lower[down_log_l], upper[down_log_l],
        )
    )
    if not np.all(np.isfinite(relevant)):
        raise ValueError("Named-lens centre and log-length bounds must be finite.")
    with np.errstate(over="ignore"):
        upstream_max_length = float(np.exp(upper[up_log_l]))
        downstream_max_length = float(np.exp(upper[down_log_l]))
    if not np.all(np.isfinite((upstream_max_length, downstream_max_length))):
        raise ValueError("Named-lens maximum lengths must be finite.")
    return HardEdgeInternalPlaneBounds(
        lower_bounds=lower,
        upper_bounds=upper,
        plane_z_m=plane,
        upstream_lens_name=upstream,
        downstream_lens_name=downstream,
        minimum_clearance_m=clearance,
        worst_case_upstream_exit_z_m=float(
            upper[up_z] + 0.5 * upstream_max_length
        ),
        worst_case_downstream_entrance_z_m=float(
            lower[down_z] - 0.5 * downstream_max_length
        ),
        was_restricted=was_restricted,
    )


def validate_hard_edge_internal_plane_bounds(
    column: HardEdgeColumnParameterization,
    bounds: tuple[np.ndarray, np.ndarray],
    *,
    plane_z_m: float,
    upstream_lens_name: str,
    downstream_lens_name: str,
    minimum_clearance_m: float = 0.0,
) -> HardEdgeInternalPlaneBounds:
    """Prove that every corner of a box keeps a plane between two bodies.

    Body edges are monotone in centre and positive length, so the adverse
    centre bounds paired with maximum log-lengths analytically cover the whole
    Cartesian box.  Unsafe bounds fail closed with ``ValueError``.
    """

    lower, upper = _hard_edge_parameter_box(column, bounds)
    return _internal_plane_bound_certificate(
        column,
        lower,
        upper,
        plane_z_m=plane_z_m,
        upstream_lens_name=upstream_lens_name,
        downstream_lens_name=downstream_lens_name,
        minimum_clearance_m=minimum_clearance_m,
        was_restricted=False,
    )


def restrict_hard_edge_internal_plane_bounds(
    column: HardEdgeColumnParameterization,
    bounds: tuple[np.ndarray, np.ndarray],
    *,
    plane_z_m: float,
    upstream_lens_name: str,
    downstream_lens_name: str,
    minimum_clearance_m: float = 0.0,
    reference_parameters: np.ndarray | None = None,
) -> HardEdgeInternalPlaneBounds:
    """Restrict a box so a fixed plane cannot enter either named lens body.

    The reference (nominal parameters by default) is retained.  For each lens,
    the adverse centre excursion and maximum physical-length excursion are
    reduced by a common factor.  The returned axis-aligned sub-box is then
    certified analytically by :func:`validate_hard_edge_internal_plane_bounds`.
    """

    original_lower, original_upper = _hard_edge_parameter_box(column, bounds)
    lower = original_lower.copy()
    upper = original_upper.copy()
    plane = float(plane_z_m)
    clearance = float(minimum_clearance_m)
    if not np.isfinite(plane) or not column.source_z_m < plane < column.detector_z_m:
        raise ValueError("plane_z_m must lie strictly inside the nominal column.")
    if not np.isfinite(clearance) or clearance < 0.0:
        raise ValueError("minimum_clearance_m must be finite and non-negative.")
    upstream, downstream, up_z, up_log_l, down_z, down_log_l = (
        _hard_edge_internal_plane_geometry(
            column, upstream_lens_name, downstream_lens_name
        )
    )
    lens_names = tuple(lens.name for lens in column.lenses)
    upstream_lens = column.lenses[lens_names.index(upstream)]
    downstream_lens = column.lenses[lens_names.index(downstream)]
    if not upstream_lens.exit_z <= plane <= downstream_lens.entrance_z:
        raise ValueError("The plane must lie in the named nominal inter-lens gap.")

    reference = np.asarray(
        column.nominal_parameters
        if reference_parameters is None
        else reference_parameters,
        dtype=float,
    )
    if reference.shape != (column.n_parameters,) or not np.all(np.isfinite(reference)):
        raise ValueError(
            f"reference_parameters must be finite with shape ({column.n_parameters},)."
        )
    if np.any(reference < lower) or np.any(reference > upper):
        raise ValueError("reference_parameters must lie inside bounds.")

    def physical_length(log_length: float) -> float:
        with np.errstate(over="ignore"):
            length = float(np.exp(log_length))
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError("Named-lens log-length bounds must decode finitely.")
        return length

    up_ref_length = physical_length(reference[up_log_l])
    down_ref_length = physical_length(reference[down_log_l])
    available_upstream = (
        plane - clearance - reference[up_z] - 0.5 * up_ref_length
    )
    available_downstream = (
        reference[down_z] - 0.5 * down_ref_length - plane - clearance
    )
    tolerance = 32.0 * np.finfo(float).eps * max(1.0, abs(plane))
    if available_upstream < -tolerance or available_downstream < -tolerance:
        raise ValueError(
            "reference_parameters do not keep the requested clearance from the plane."
        )
    available_upstream = max(available_upstream, 0.0)
    available_downstream = max(available_downstream, 0.0)

    up_requested_length = physical_length(upper[up_log_l])
    up_consumption = (
        upper[up_z]
        - reference[up_z]
        + 0.5 * (up_requested_length - up_ref_length)
    )
    if up_consumption > available_upstream:
        factor = available_upstream / up_consumption
        if factor > 0.0:
            factor = float(np.nextafter(factor, 0.0))
        upper[up_z] = reference[up_z] + factor * (
            upper[up_z] - reference[up_z]
        )
        upper[up_log_l] = np.log(
            up_ref_length + factor * (up_requested_length - up_ref_length)
        )

    down_requested_length = physical_length(upper[down_log_l])
    down_consumption = (
        reference[down_z]
        - lower[down_z]
        + 0.5 * (down_requested_length - down_ref_length)
    )
    if down_consumption > available_downstream:
        factor = available_downstream / down_consumption
        if factor > 0.0:
            factor = float(np.nextafter(factor, 0.0))
        lower[down_z] = reference[down_z] - factor * (
            reference[down_z] - lower[down_z]
        )
        upper[down_log_l] = np.log(
            down_ref_length
            + factor * (down_requested_length - down_ref_length)
        )

    if np.any(lower > upper):
        raise ValueError("Plane restriction leaves no valid independent parameter box.")
    return _internal_plane_bound_certificate(
        column,
        lower,
        upper,
        plane_z_m=plane,
        upstream_lens_name=upstream,
        downstream_lens_name=downstream,
        minimum_clearance_m=clearance,
        was_restricted=not (
            np.array_equal(lower, original_lower)
            and np.array_equal(upper, original_upper)
        ),
    )


def _hard_edge_internal_plane_constraints(
    constraints: Sequence[HardEdgeInternalPlaneConstraint],
) -> tuple[HardEdgeInternalPlaneConstraint, ...]:
    values = tuple(constraints)
    if not values:
        raise ValueError("constraints must contain at least one fixed plane.")
    if not all(
        isinstance(value, HardEdgeInternalPlaneConstraint) for value in values
    ):
        raise TypeError(
            "constraints must contain HardEdgeInternalPlaneConstraint values."
        )
    return values


def validate_hard_edge_internal_planes_bounds(
    column: HardEdgeColumnParameterization,
    bounds: tuple[np.ndarray, np.ndarray],
    constraints: Sequence[HardEdgeInternalPlaneConstraint],
) -> tuple[HardEdgeInternalPlaneBounds, ...]:
    """Certify one parameter box against several fixed internal planes.

    The certificates preserve the order of ``constraints`` and all contain
    the same immutable lower and upper bounds.  Validation fails closed if
    any one plane could enter either adjacent hard-edge body.
    """

    lower, upper = _hard_edge_parameter_box(column, bounds)
    specifications = _hard_edge_internal_plane_constraints(constraints)
    certificates: list[HardEdgeInternalPlaneBounds] = []
    for index, specification in enumerate(specifications):
        try:
            certificate = _internal_plane_bound_certificate(
                column,
                lower,
                upper,
                plane_z_m=specification.plane_z_m,
                upstream_lens_name=specification.upstream_lens_name,
                downstream_lens_name=specification.downstream_lens_name,
                minimum_clearance_m=specification.minimum_clearance_m,
                was_restricted=False,
            )
        except ValueError as error:
            raise ValueError(
                "Internal-plane constraint "
                f"{index} at z={specification.plane_z_m:.9g} m failed: {error}"
            ) from error
        certificates.append(certificate)
    return tuple(certificates)


def restrict_hard_edge_internal_planes_bounds(
    column: HardEdgeColumnParameterization,
    bounds: tuple[np.ndarray, np.ndarray],
    constraints: Sequence[HardEdgeInternalPlaneConstraint],
    *,
    reference_parameters: np.ndarray | None = None,
) -> tuple[HardEdgeInternalPlaneBounds, ...]:
    """Intersect a box with several fixed-plane clearance constraints.

    Each plane independently produces an inner box from the same requested
    bounds.  Their componentwise intersection is therefore independent of
    constraint order and retains ``reference_parameters``.  Every plane is
    then re-certified against that same final box.  This final pass prevents
    a caller from accidentally treating certificates from different
    intermediate boxes as one design.
    """

    original_lower, original_upper = _hard_edge_parameter_box(column, bounds)
    specifications = _hard_edge_internal_plane_constraints(constraints)
    restricted_lower = [original_lower]
    restricted_upper = [original_upper]
    for index, specification in enumerate(specifications):
        try:
            certificate = restrict_hard_edge_internal_plane_bounds(
                column,
                (original_lower, original_upper),
                plane_z_m=specification.plane_z_m,
                upstream_lens_name=specification.upstream_lens_name,
                downstream_lens_name=specification.downstream_lens_name,
                minimum_clearance_m=specification.minimum_clearance_m,
                reference_parameters=reference_parameters,
            )
        except ValueError as error:
            raise ValueError(
                "Internal-plane constraint "
                f"{index} at z={specification.plane_z_m:.9g} m is infeasible: "
                f"{error}"
            ) from error
        plane_lower, plane_upper = certificate.bounds
        restricted_lower.append(plane_lower)
        restricted_upper.append(plane_upper)

    lower = np.maximum.reduce(restricted_lower)
    upper = np.minimum.reduce(restricted_upper)
    if np.any(lower > upper):
        raise ValueError(
            "Internal-plane restrictions leave no common parameter box."
        )

    was_restricted = not (
        np.array_equal(lower, original_lower)
        and np.array_equal(upper, original_upper)
    )
    certificates = validate_hard_edge_internal_planes_bounds(
        column,
        (lower, upper),
        specifications,
    )
    return tuple(
        HardEdgeInternalPlaneBounds(
            lower_bounds=certificate.lower_bounds,
            upper_bounds=certificate.upper_bounds,
            plane_z_m=certificate.plane_z_m,
            upstream_lens_name=certificate.upstream_lens_name,
            downstream_lens_name=certificate.downstream_lens_name,
            minimum_clearance_m=certificate.minimum_clearance_m,
            worst_case_upstream_exit_z_m=(
                certificate.worst_case_upstream_exit_z_m
            ),
            worst_case_downstream_entrance_z_m=(
                certificate.worst_case_downstream_entrance_z_m
            ),
            was_restricted=was_restricted,
        )
        for certificate in certificates
    )


def propagate_phase_space_gaussian(
    matrix: np.ndarray,
    mean_state: Sequence[float],
    covariance: np.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Propagate the mean and covariance of a first-order Gaussian beam."""

    transform = jnp.asarray(matrix)
    mean = jnp.asarray(mean_state)
    sigma = jnp.asarray(covariance)
    if transform.shape != (4, 4) or mean.shape != (4,) or sigma.shape != (4, 4):
        raise ValueError("Expected a 4x4 matrix, length-four mean, and 4x4 covariance.")
    return transform @ mean, transform @ sigma @ transform.T


def phase_space_covariance_observables(
    covariance: np.ndarray,
    *,
    mode: str = "full",
) -> jnp.ndarray:
    r"""Return a unit-stable observable chart for a 4D beam covariance.

    The full chart contains the four logarithmic RMS widths followed by all
    six Pearson correlations in :data:`PHASE_SPACE_COVARIANCE_OBSERVABLE_NAMES`
    order.  It is locally one-to-one on the cone of positive-definite 4x4
    covariances and, unlike raw covariance entries, does not mix metre,
    radian, and squared units in the same numerical rank test.  ``position``
    retains only ``log(sigma_x)``, ``log(sigma_y)``, and ``corr(x, y)``: the
    quantities available from one registered intensity screen.

    The function is composed only of JAX operations after static validation,
    so it may be used inside ``jit`` and differentiated through a propagated
    covariance or :class:`EntranceCovarianceChart`.
    """

    if mode not in _COVARIANCE_MODE_INDICES:
        raise ValueError(
            f"mode must be one of {tuple(_COVARIANCE_MODE_INDICES)}, got {mode!r}."
        )
    sigma = jnp.asarray(_state_covariance(covariance))
    dtype = jnp.result_type(sigma, jnp.asarray(1.0))
    sigma = sigma.astype(dtype)
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    variances = jnp.maximum(jnp.diag(sigma), tiny)
    rms = jnp.sqrt(variances)
    correlations = jnp.stack(
        tuple(
            sigma[row, column] / (rms[row] * rms[column])
            for row, column in _PHASE_SPACE_CORRELATION_INDICES
        )
    )
    full = jnp.concatenate((jnp.log(rms), correlations))
    return full[jnp.asarray(_COVARIANCE_MODE_INDICES[mode])]


def axial_plane_covariance_observables(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    dac_values: np.ndarray,
    source_covariance: np.ndarray,
    *,
    plane_z_m: float,
    mode: str = "full",
) -> jnp.ndarray:
    """Predict covariance summaries at a known axial plane for DAC states.

    ``dac_values`` may be one raw-DAC row or a batch; the returned value always
    has shape ``(n_states, n_observables)``.  The plane must be a fixed drift
    gap accepted by :func:`hard_edge_matrix_to_plane`.  A full ten-component
    summary requires calibrated phase-space tomography (for example, paired
    image/diffraction and defocus-diverse measurements); ``position`` is the
    deliberately weaker single-screen contract.
    """

    if mode not in _COVARIANCE_MODE_INDICES:
        raise ValueError(
            f"mode must be one of {tuple(_COVARIANCE_MODE_INDICES)}, got {mode!r}."
        )
    covariance = jnp.asarray(_state_covariance(source_covariance))
    controls = jnp.asarray(dac_values)
    if controls.ndim == 1:
        controls = controls[None, :]
    if controls.ndim != 2 or controls.shape[1] != len(column.channel_names):
        raise ValueError(
            "dac_values must have shape (n_states, n_channels) or (n_channels,)."
        )

    def one_state(dacs: jnp.ndarray) -> jnp.ndarray:
        matrix = hard_edge_matrix_to_plane(column, parameters, dacs, plane_z_m)
        propagated = matrix @ covariance @ matrix.T
        return phase_space_covariance_observables(propagated, mode=mode)

    return jax.vmap(one_state)(controls)


def axial_plane_covariance_observables_with_entrance_covariance(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    covariance_coordinates: np.ndarray,
    dac_values: np.ndarray,
    covariance_chart: EntranceCovarianceChart,
    *,
    plane_z_m: float,
    mode: str = "full",
) -> jnp.ndarray:
    """Predict a known-plane covariance chart with entrance covariance free."""

    if not isinstance(covariance_chart, EntranceCovarianceChart):
        raise TypeError("covariance_chart must be an EntranceCovarianceChart.")
    return axial_plane_covariance_observables(
        column,
        parameters,
        dac_values,
        covariance_chart.covariance(covariance_coordinates),
        plane_z_m=plane_z_m,
        mode=mode,
    )


def position_moments(
    matrix: np.ndarray,
    mean_state: Sequence[float],
    covariance: np.ndarray,
    *,
    affine_xy: np.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return transverse position mean/covariance after a transfer matrix."""

    mean, sigma = propagate_phase_space_gaussian(matrix, mean_state, covariance)
    affine = jnp.eye(2, dtype=mean.dtype) if affine_xy is None else jnp.asarray(affine_xy)
    if affine.shape != (2, 2):
        raise ValueError("affine_xy must have shape (2, 2).")
    transverse_mean = affine @ mean[_POSITION]
    transverse_covariance = affine @ sigma[jnp.ix_(_POSITION, _POSITION)] @ affine.T
    return transverse_mean, transverse_covariance


def rms_beam_radius(covariance_xy_m2: np.ndarray) -> jnp.ndarray:
    """Return the one-axis-equivalent RMS radius ``sqrt(trace(C)/2)``."""

    covariance = jnp.asarray(covariance_xy_m2)
    if covariance.shape != (2, 2):
        raise ValueError("covariance_xy_m2 must have shape (2, 2).")
    return jnp.sqrt(jnp.maximum(0.5 * jnp.trace(covariance), 0.0))


def phase_space_probe_observables(
    mean_state: np.ndarray,
    covariance: np.ndarray,
) -> jnp.ndarray:
    """Return stable sample-plane observables from a paraxial Gaussian.

    The output order is :data:`CONDENSER_PROBE_OBSERVABLE_NAMES`.  The two
    logarithmic radii and the symmetric position--angle correlation form the
    ``complex_radial`` contract used for condenser experiment design.  They
    can be recovered, respectively, from a registered sample/perforation
    image, a calibrated diffraction view, and defocus diversity or complex
    probe retrieval.  The final antisymmetric covariance is intentionally
    separate: ordinary image/defocus/diffraction screens cannot observe it.

    This function describes the beam *at* the specimen; it does not turn a
    calibrated specimen-plane input into an upstream measurement.  Inferring
    the condenser still requires a stable, independently characterized state
    at a plane before ``CL1`` (or an equivalent source nuisance model).
    """

    mean = jnp.asarray(mean_state)
    sigma = jnp.asarray(covariance)
    if mean.shape != (4,) or sigma.shape != (4, 4):
        raise ValueError("Expected a length-four mean and 4x4 covariance.")
    position_variance = 0.5 * (sigma[0, 0] + sigma[2, 2])
    angle_variance = 0.5 * (sigma[1, 1] + sigma[3, 3])
    dtype = jnp.result_type(mean, sigma)
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    position_variance = jnp.maximum(position_variance, tiny)
    angle_variance = jnp.maximum(angle_variance, tiny)
    normalization = jnp.sqrt(position_variance * angle_variance)
    symmetric_correlation = 0.5 * (sigma[0, 1] + sigma[2, 3]) / normalization
    canonical_correlation = 0.5 * (sigma[0, 3] - sigma[2, 1]) / normalization
    return jnp.stack(
        (
            mean[0],
            mean[2],
            mean[1],
            mean[3],
            0.5 * jnp.log(position_variance),
            0.5 * jnp.log(angle_variance),
            symmetric_correlation,
            canonical_correlation,
        )
    )


def sample_plane_probe_observables(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    dac_values: np.ndarray,
    source_mean_state: Sequence[float],
    source_covariance: np.ndarray,
    *,
    mode: str = "complex_radial",
) -> jnp.ndarray:
    """Predict measurable probe summaries for one or more condenser states.

    ``dac_values`` may be one raw-DAC row or a batch.  The return value always
    has a leading state axis.  ``mode='position'`` corresponds to a registered
    sample/perforation image, ``'angle'`` to a calibrated diffraction view,
    and ``'complex_radial'`` to their radii plus a defocus-derived wavefront
    curvature.  ``'full'`` is an audit contract, not a claim that one detector
    frame exposes all eight quantities.
    """

    if mode not in _PROBE_MODE_INDICES:
        raise ValueError(
            f"mode must be one of {tuple(_PROBE_MODE_INDICES)}, got {mode!r}."
        )
    mean = jnp.asarray(_state_vector("source_mean_state", source_mean_state))
    covariance = jnp.asarray(_state_covariance(source_covariance))
    controls = jnp.asarray(dac_values)
    if controls.ndim == 1:
        controls = controls[None, :]
    if controls.ndim != 2 or controls.shape[1] != len(column.channel_names):
        raise ValueError(
            "dac_values must have shape (n_states, n_channels) or (n_channels,)."
        )

    def one_state(dacs: jnp.ndarray) -> jnp.ndarray:
        matrix = hard_edge_matrix_to_plane(
            column, parameters, dacs, column.sample_z_m
        )
        propagated_mean, propagated_covariance = propagate_phase_space_gaussian(
            matrix, mean, covariance
        )
        return phase_space_probe_observables(
            propagated_mean, propagated_covariance
        )

    all_observables = jax.vmap(one_state)(controls)
    indices = jnp.asarray(_PROBE_MODE_INDICES[mode])
    return all_observables[:, indices]


def sample_plane_probe_observables_with_entrance_covariance(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    covariance_coordinates: np.ndarray,
    dac_values: np.ndarray,
    source_mean_state: Sequence[float],
    covariance_chart: EntranceCovarianceChart,
    *,
    mode: str = "complex_radial",
) -> jnp.ndarray:
    """Predict specimen-plane probes with a released entrance covariance.

    ``covariance_coordinates`` are the ten dimensionless coordinates of
    :class:`EntranceCovarianceChart`.  Keeping optical parameters and entrance
    nuisance coordinates as separate arguments makes this function convenient
    both for a joint nonlinear residual and for blockwise Jacobian audits.
    """

    if not isinstance(covariance_chart, EntranceCovarianceChart):
        raise TypeError("covariance_chart must be an EntranceCovarianceChart.")
    covariance = covariance_chart.covariance(covariance_coordinates)
    return sample_plane_probe_observables(
        column,
        parameters,
        dac_values,
        source_mean_state,
        covariance,
        mode=mode,
    )


def circular_perforation_signal(
    mean_xy_m: Sequence[float],
    covariance_xy_m2: np.ndarray,
    perforation_centre_xy_m: Sequence[float],
    perforation_radius_m: float,
    *,
    vacuum_signal: float = 1.0,
    material_signal: float = 0.0,
    radial_order: int = 32,
    azimuthal_order: int = 96,
) -> jnp.ndarray:
    """Predict normalized counts from a Gaussian crossing a circular hole.

    ``vacuum_signal`` and ``material_signal`` are separately calibrated
    detector signals for illumination wholly inside and outside the hole.  A
    stage scan across the certified perforation therefore provides a direct
    sample-plane centroid/radius measurement without inventing a new sample.
    Contrast or dose uncertainty must be included as a nuisance in a real fit.
    """

    centre = jnp.asarray(perforation_centre_xy_m)
    if centre.shape != (2,):
        raise ValueError("perforation_centre_xy_m must have shape (2,).")
    if not np.all(np.isfinite((vacuum_signal, material_signal))):
        raise ValueError("vacuum_signal and material_signal must be finite.")
    inside_probability = circular_gaussian_aperture_transmission(
        jnp.asarray(mean_xy_m) - centre,
        covariance_xy_m2,
        perforation_radius_m,
        radial_order=radial_order,
        azimuthal_order=azimuthal_order,
    )
    return material_signal + (vacuum_signal - material_signal) * inside_probability


def sample_perforation_edge_scan(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    dac_values: np.ndarray,
    source_mean_state: Sequence[float],
    source_covariance: np.ndarray,
    *,
    stage_offsets_xy_m: np.ndarray,
    perforation_centre_xy_m: Sequence[float],
    perforation_radius_m: float,
    vacuum_signal: float = 1.0,
    material_signal: float = 0.0,
    radial_order: int = 32,
    azimuthal_order: int = 96,
) -> jnp.ndarray:
    """Predict a registered two-dimensional perforation edge scan.

    Positive stage offsets translate the perforation centre in the microscope
    frame.  The returned samples are detector-integrated intensity fractions,
    so the scan is directly measurable even though its fitted centroid and
    radius are subsequently used as compact observables.
    """

    offsets = jnp.asarray(stage_offsets_xy_m)
    if offsets.ndim != 2 or offsets.shape[1] != 2:
        raise ValueError("stage_offsets_xy_m must have shape (n_positions, 2).")
    matrix = hard_edge_matrix_to_plane(
        column, parameters, dac_values, column.sample_z_m
    )
    mean_xy, covariance_xy = position_moments(
        matrix, source_mean_state, source_covariance
    )
    centre = jnp.asarray(perforation_centre_xy_m)
    return jax.vmap(
        lambda offset: circular_perforation_signal(
            mean_xy,
            covariance_xy,
            centre + offset,
            perforation_radius_m,
            vacuum_signal=vacuum_signal,
            material_signal=material_signal,
            radial_order=radial_order,
            azimuthal_order=azimuthal_order,
        )
    )(offsets)


def detector_log_radius(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    dac_values: np.ndarray,
    source_mean_state: Sequence[float],
    source_covariance: np.ndarray,
    *,
    detector_affine_xy: np.ndarray | None = None,
    observation_z_m: float | None = None,
) -> jnp.ndarray:
    """Predict log RMS beam radius for a vacuum exposure.

    By default the radius is evaluated at the detector, preserving the
    historical behaviour of this function.  ``observation_z_m`` may instead
    select an explicit internal drift plane, notably ``column.sample_z_m``.
    The optional affine is applied to transverse coordinates at whichever
    plane is selected.
    """

    matrix = (
        column.matrix(parameters, dac_values, segment="full")
        if observation_z_m is None
        else hard_edge_matrix_to_plane(
            column, parameters, dac_values, observation_z_m
        )
    )
    _, covariance = position_moments(
        matrix,
        source_mean_state,
        source_covariance,
        affine_xy=detector_affine_xy,
    )
    return jnp.log(jnp.maximum(rms_beam_radius(covariance), jnp.finfo(matrix.dtype).tiny))


def signed_wobble_response(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    dac_values: np.ndarray,
    source_mean_state: Sequence[float],
    source_covariance: np.ndarray,
    *,
    wobble_channel_index: int,
    wobble_step: float,
    detector_affine_xy: np.ndarray | None = None,
    observation_z_m: float | None = None,
) -> jnp.ndarray:
    """Central finite difference of log beam radius in raw-DAC coordinates.

    ``observation_z_m=None`` measures the detector radius.  Passing a physical
    z coordinate evaluates both wobble states at that fixed internal plane.
    """

    channel = int(wobble_channel_index)
    if channel < 0 or channel >= len(column.channel_names):
        raise ValueError("wobble_channel_index is outside the column controls.")
    step = float(wobble_step)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("wobble_step must be finite and positive.")
    dacs = jnp.asarray(dac_values)
    direction = jnp.zeros_like(dacs).at[channel].set(step)
    plus = detector_log_radius(
        column,
        parameters,
        dacs + direction,
        source_mean_state,
        source_covariance,
        detector_affine_xy=detector_affine_xy,
        observation_z_m=observation_z_m,
    )
    minus = detector_log_radius(
        column,
        parameters,
        dacs - direction,
        source_mean_state,
        source_covariance,
        detector_affine_xy=detector_affine_xy,
        observation_z_m=observation_z_m,
    )
    return (plus - minus) / (2.0 * step)


def condenser_reconnaissance_frame_observables(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    paired_dac_values: np.ndarray,
    source_mean_state: Sequence[float],
    source_covariance: np.ndarray,
    *,
    aperture_z_m: float,
    aperture_radius_m: float,
    observation_z_m: float | None = None,
    radial_order: int = 32,
    azimuthal_order: int = 96,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Predict per-frame radii and aperture counts for paired lens wobbles.

    The DAC array has shape ``(n_pairs, 2, n_channels)`` with the negative and
    positive wobble in axis one.  Both returned arrays have shape
    ``(n_pairs, 2)``.  Keeping the absolute log radii is useful: collapsing
    each pair to only a derivative can discard an identifiable direction.
    """

    controls = jnp.asarray(paired_dac_values)
    if controls.ndim != 3 or controls.shape[1:] != (
        2,
        len(column.channel_names),
    ):
        raise ValueError(
            "paired_dac_values must have shape (n_pairs, 2, n_channels)."
        )
    mean = jnp.asarray(_state_vector("source_mean_state", source_mean_state))
    covariance = jnp.asarray(_state_covariance(source_covariance))
    observation_plane = (
        column.sample_z_m if observation_z_m is None else float(observation_z_m)
    )

    def one_frame(dacs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        observation_matrix = hard_edge_matrix_to_plane(
            column, parameters, dacs, observation_plane
        )
        _, observation_covariance = position_moments(
            observation_matrix, mean, covariance
        )
        aperture_matrix = hard_edge_matrix_to_plane(
            column, parameters, dacs, aperture_z_m
        )
        aperture_mean, aperture_covariance = position_moments(
            aperture_matrix, mean, covariance
        )
        log_radius = jnp.log(
            jnp.maximum(
                rms_beam_radius(observation_covariance),
                jnp.finfo(observation_matrix.dtype).tiny,
            )
        )
        transmission = circular_gaussian_aperture_transmission(
            aperture_mean,
            aperture_covariance,
            aperture_radius_m,
            radial_order=radial_order,
            azimuthal_order=azimuthal_order,
        )
        return log_radius, transmission

    flat_controls = controls.reshape((-1, controls.shape[-1]))
    log_radius, transmission = jax.vmap(one_frame)(flat_controls)
    shape = controls.shape[:2]
    return log_radius.reshape(shape), transmission.reshape(shape)


def condenser_reconnaissance_response(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    paired_dac_values: np.ndarray,
    source_mean_state: Sequence[float],
    source_covariance: np.ndarray,
    *,
    wobble_step: float,
    aperture_z_m: float,
    aperture_radius_m: float,
    observation_z_m: float | None = None,
    retain_absolute_log_radius: bool = False,
    radial_order: int = 32,
    azimuthal_order: int = 96,
) -> jnp.ndarray:
    """Return the differentiable historical critical-line response vector.

    The historical compact contract contains one signed log-radius derivative
    per pair followed by both absolute aperture transmissions.  Setting
    ``retain_absolute_log_radius`` additionally appends both frame radii.  The
    latter costs no extra acquisition frames when the probe retrieval already
    produced the two radii.  Because the signed derivative is an exact linear
    combination of those radii, callers must not assign independent errors to
    all three quantities.  Use
    :func:`condenser_independent_reconnaissance_response` for a diagonal-noise
    likelihood, or supply the full (singular) covariance explicitly.
    """

    step = float(wobble_step)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("wobble_step must be finite and positive.")
    log_radius, transmission = condenser_reconnaissance_frame_observables(
        column,
        parameters,
        paired_dac_values,
        source_mean_state,
        source_covariance,
        aperture_z_m=aperture_z_m,
        aperture_radius_m=aperture_radius_m,
        observation_z_m=observation_z_m,
        radial_order=radial_order,
        azimuthal_order=azimuthal_order,
    )
    signed_response = (log_radius[:, 1] - log_radius[:, 0]) / (2.0 * step)
    blocks = [signed_response, transmission.reshape(-1)]
    if retain_absolute_log_radius:
        blocks.append(log_radius.reshape(-1))
    return jnp.concatenate(tuple(blocks))


def condenser_independent_reconnaissance_response(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    paired_dac_values: np.ndarray,
    source_mean_state: Sequence[float],
    source_covariance: np.ndarray,
    *,
    aperture_z_m: float,
    aperture_radius_m: float,
    observation_z_m: float | None = None,
    radial_order: int = 32,
    azimuthal_order: int = 96,
) -> jnp.ndarray:
    """Return algebraically non-duplicated transmissions and log radii.

    The output ordering is ``[transmission.reshape(-1), log_radius.reshape(-1)]``.
    Each pair therefore contributes four scalar measurements derived from its
    two physical frames.  A signed critical-line derivative may still be used
    to adapt the scan or displayed as a diagnostic, but it is deliberately not
    repeated in this vector because it contains no information independent of
    the two absolute log radii.  The historical API name is retained for
    compatibility: these observables are algebraically independent, not
    necessarily statistically independent.  Transmission and radius extracted
    from the same image may remain correlated and should use their measured
    joint covariance in a likelihood.
    """

    log_radius, transmission = condenser_reconnaissance_frame_observables(
        column,
        parameters,
        paired_dac_values,
        source_mean_state,
        source_covariance,
        aperture_z_m=aperture_z_m,
        aperture_radius_m=aperture_radius_m,
        observation_z_m=observation_z_m,
        radial_order=radial_order,
        azimuthal_order=azimuthal_order,
    )
    return jnp.concatenate((transmission.reshape(-1), log_radius.reshape(-1)))


@dataclass(frozen=True)
class CriticalLineScan:
    """Signed wobble response evaluated along one raw-DAC scan axis."""

    scan_channel_index: int
    wobble_channel_index: int
    scan_values: np.ndarray
    response: np.ndarray
    zero_crossings: np.ndarray
    observation_z_m: float | None = None


def linear_zero_crossings(x: Sequence[float], y: Sequence[float]) -> np.ndarray:
    """Return linearly interpolated sign-changing roots without extrapolation."""

    coordinate = np.asarray(x, dtype=float)
    values = np.asarray(y, dtype=float)
    if coordinate.ndim != 1 or values.shape != coordinate.shape or coordinate.size < 2:
        raise ValueError("x and y must be equal one-dimensional arrays of length >= 2.")
    if not np.all(np.isfinite(coordinate)) or not np.all(np.isfinite(values)):
        raise ValueError("x and y must contain only finite values.")
    if np.any(np.diff(coordinate) <= 0.0):
        raise ValueError("x must be strictly increasing.")
    roots: list[float] = []
    for x0, x1, y0, y1 in zip(
        coordinate[:-1], coordinate[1:], values[:-1], values[1:], strict=True
    ):
        if y0 == 0.0:
            roots.append(float(x0))
        if y0 * y1 < 0.0:
            roots.append(float(x0 - y0 * (x1 - x0) / (y1 - y0)))
    if values[-1] == 0.0:
        roots.append(float(coordinate[-1]))
    return np.unique(np.asarray(roots, dtype=float))


def critical_line_scan(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    reference_dac_values: Sequence[float],
    source_mean_state: Sequence[float],
    source_covariance: np.ndarray,
    *,
    scan_channel_index: int,
    wobble_channel_index: int,
    scan_values: Sequence[float],
    wobble_step: float,
    detector_affine_xy: np.ndarray | None = None,
    observation_z_m: float | None = None,
) -> CriticalLineScan:
    """Evaluate a one-dimensional critical-line experiment.

    A root means that the measured beam diameter is locally insensitive to the
    wobbled lens.  It is a compact alternative to retaining a full two-lens
    sweep, although the full response curve should still be archived.  The
    detector is used by default; ``observation_z_m`` selects a fixed internal
    drift plane such as the specimen plane.
    """

    scan_channel = int(scan_channel_index)
    wobble_channel = int(wobble_channel_index)
    if scan_channel < 0 or scan_channel >= len(column.channel_names):
        raise ValueError("scan_channel_index is outside the column controls.")
    scan = np.asarray(scan_values, dtype=float)
    if scan.ndim != 1 or scan.size < 2 or np.any(np.diff(scan) <= 0.0):
        raise ValueError("scan_values must be a strictly increasing vector.")
    reference = np.asarray(reference_dac_values, dtype=float)
    if reference.shape != (len(column.channel_names),):
        raise ValueError("reference_dac_values has the wrong number of controls.")
    responses = []
    for value in scan:
        dacs = reference.copy()
        dacs[scan_channel] = value
        responses.append(
            float(
                signed_wobble_response(
                    column,
                    parameters,
                    dacs,
                    source_mean_state,
                    source_covariance,
                    wobble_channel_index=wobble_channel,
                    wobble_step=wobble_step,
                    detector_affine_xy=detector_affine_xy,
                    observation_z_m=observation_z_m,
                )
            )
        )
    response = np.asarray(responses)
    return CriticalLineScan(
        scan_channel,
        wobble_channel,
        scan.copy(),
        response,
        linear_zero_crossings(scan, response),
        None if observation_z_m is None else float(observation_z_m),
    )


def circular_gaussian_aperture_transmission(
    mean_xy_m: Sequence[float],
    covariance_xy_m2: np.ndarray,
    radius_m: float | np.ndarray,
    *,
    radial_order: int = 32,
    azimuthal_order: int = 96,
) -> jnp.ndarray:
    """Integrate an arbitrary 2D Gaussian over a hard circular aperture.

    Fixed Gauss--Legendre radial quadrature and a periodic azimuthal rule keep
    this differentiable with respect to beam moments and aperture radius.  It
    is more accurate than a beam-diameter clipping heuristic near the edge.
    """

    # Do not coerce the radius through ``float`` here.  Radius is an explicit
    # calibration nuisance in aperture-edge fits, and concretising it would
    # silently break ``jax.jacfwd``/``jax.jit``.  Numerical callers still get
    # the same eager validation; transformed callers inherit positivity from
    # their parameterisation (normally ``r_ref * exp(log_radius_delta)``).
    radius = _runtime_real_array("radius_m", radius_m)
    if radius.shape != ():
        raise ValueError("radius_m must be scalar.")
    if not isinstance(radius, jax.core.Tracer) and float(radius) <= 0.0:
        raise ValueError("radius_m must be finite and positive.")
    radial_order = int(radial_order)
    azimuthal_order = int(azimuthal_order)
    if radial_order < 8 or azimuthal_order < 16:
        raise ValueError("Use radial_order >= 8 and azimuthal_order >= 16.")
    mean = _runtime_real_array("mean_xy_m", mean_xy_m)
    covariance = _runtime_real_array("covariance_xy_m2", covariance_xy_m2)
    if mean.shape != (2,) or covariance.shape != (2, 2):
        raise ValueError("Expected a length-two mean and 2x2 covariance.")
    nodes_np, weights_np = np.polynomial.legendre.leggauss(radial_order)
    dtype = jnp.result_type(mean, covariance, radius)
    radius = radius.astype(dtype)
    radial = 0.5 * radius * (jnp.asarray(nodes_np, dtype=dtype) + 1.0)
    radial_weights = 0.5 * radius * jnp.asarray(weights_np, dtype=dtype)
    angle = 2.0 * jnp.pi * jnp.arange(azimuthal_order) / azimuthal_order
    xx = radial[:, None] * jnp.cos(angle)[None, :]
    yy = radial[:, None] * jnp.sin(angle)[None, :]
    points = jnp.stack((xx, yy), axis=-1)
    delta = points - mean
    inverse = jnp.linalg.inv(covariance)
    exponent = jnp.einsum("...i,ij,...j->...", delta, inverse, delta)
    density = jnp.exp(-0.5 * exponent) / (
        2.0 * jnp.pi * jnp.sqrt(jnp.linalg.det(covariance))
    )
    angular_step = 2.0 * jnp.pi / azimuthal_order
    probability = jnp.sum(
        radial_weights[:, None] * radial[:, None] * angular_step * density
    )
    return jnp.clip(probability, 0.0, 1.0)


class ApertureEdgeScanPrediction(NamedTuple):
    """Absolute predictions for registered condenser-aperture edge images.

    Every array retains a leading optical-state axis, even for a single state.
    ``transmission`` and ``expected_transmitted_electrons`` have shape
    ``(n_states, n_scan_positions)``.  The other fields make the physical
    moments and all fitted aperture nuisances inspectable rather than hiding
    them in a closure or normalising away the incident dose.

    ``aperture_out_expected_electrons`` is the expectation for one exposure
    acquired with the aperture removed.  It is the direct dose reference used
    by :class:`ApertureEdgeScanDesign`; detector background and gain belong in
    separately declared detector nuisances.
    """

    beam_mean_xy_m: jnp.ndarray
    beam_covariance_xy_m2: jnp.ndarray
    aperture_centre_xy_m: jnp.ndarray
    aperture_radius_m: jnp.ndarray
    incident_dose_electrons: jnp.ndarray
    scan_beam_mean_xy_m: jnp.ndarray
    scan_aperture_centre_xy_m: jnp.ndarray
    relative_beam_mean_xy_m: jnp.ndarray
    transmission: jnp.ndarray
    expected_transmitted_electrons: jnp.ndarray
    aperture_out_expected_electrons: jnp.ndarray


def _broadcast_aperture_state_value(
    name: str,
    value: np.ndarray,
    *,
    n_states: int,
    trailing_shape: tuple[int, ...],
) -> jnp.ndarray:
    """Broadcast one explicit aperture nuisance over optical states."""

    array = _runtime_real_array(name, value)
    if array.shape == trailing_shape:
        array = jnp.broadcast_to(array, (n_states,) + trailing_shape)
    elif array.shape != (n_states,) + trailing_shape:
        expected = (
            f"{trailing_shape} or {(n_states,) + trailing_shape}"
            if trailing_shape
            else f"() or ({n_states},)"
        )
        raise ValueError(f"{name} must have shape {expected}.")
    return array


def aperture_edge_scan_from_moments(
    beam_mean_xy_m: np.ndarray,
    beam_covariance_xy_m2: np.ndarray,
    *,
    aperture_centre_xy_m: np.ndarray,
    aperture_radius_m: float | np.ndarray,
    incident_dose_electrons: float | np.ndarray,
    aperture_centre_offsets_xy_m: np.ndarray | None = None,
    beam_offsets_xy_m: np.ndarray | None = None,
    radial_order: int = 32,
    azimuthal_order: int = 96,
) -> ApertureEdgeScanPrediction:
    """Predict raw edge-scan transmissions and counts from beam moments.

    Parameters are in one physical aperture-plane coordinate system.  The
    beam moments may describe one state, ``(2,)``/``(2,2)``, or a batch,
    ``(n_states,2)``/``(n_states,2,2)``.  The aperture centre, radius, and
    incident dose are *explicit fit inputs*: each accepts one common value or
    one value per optical state.

    Exactly one of ``aperture_centre_offsets_xy_m`` and ``beam_offsets_xy_m``
    may be supplied.  A common scan has shape ``(n_positions,2)``; a state-
    specific scan has shape ``(n_states,n_positions,2)``.  A positive beam
    offset is precisely equivalent to the negative aperture-centre offset,
    which permits the same likelihood to represent either a calibrated
    aperture translation or an upstream deflector translation.  With neither
    argument supplied, one exposure at zero offset is predicted.

    The expected count is ``incident_dose_electrons * transmission``.  No
    per-frame normalisation is performed because the absolute intensity is
    part of the condenser constraint.
    """

    mean = _runtime_real_array("beam_mean_xy_m", beam_mean_xy_m)
    if mean.shape == (2,):
        mean = mean[None, :]
    if mean.ndim != 2 or mean.shape[1] != 2:
        raise ValueError("beam_mean_xy_m must have shape (2,) or (n_states, 2).")
    n_states = mean.shape[0]

    covariance = _runtime_real_array(
        "beam_covariance_xy_m2", beam_covariance_xy_m2
    )
    if covariance.shape == (2, 2):
        covariance = jnp.broadcast_to(covariance, (n_states, 2, 2))
    elif covariance.shape != (n_states, 2, 2):
        raise ValueError(
            "beam_covariance_xy_m2 must have shape (2, 2) or "
            "(n_states, 2, 2)."
        )

    centre = _broadcast_aperture_state_value(
        "aperture_centre_xy_m",
        aperture_centre_xy_m,
        n_states=n_states,
        trailing_shape=(2,),
    )
    radius = _broadcast_aperture_state_value(
        "aperture_radius_m",
        aperture_radius_m,
        n_states=n_states,
        trailing_shape=(),
    )
    dose = _broadcast_aperture_state_value(
        "incident_dose_electrons",
        incident_dose_electrons,
        n_states=n_states,
        trailing_shape=(),
    )
    for name, values in (
        ("aperture_radius_m", radius),
        ("incident_dose_electrons", dose),
    ):
        if not isinstance(values, jax.core.Tracer) and np.any(
            np.asarray(values) <= 0.0
        ):
            raise ValueError(f"{name} must be strictly positive.")

    if aperture_centre_offsets_xy_m is not None and beam_offsets_xy_m is not None:
        raise ValueError(
            "Supply aperture_centre_offsets_xy_m or beam_offsets_xy_m, not both."
        )
    offsets_name = (
        "beam_offsets_xy_m"
        if beam_offsets_xy_m is not None
        else "aperture_centre_offsets_xy_m"
    )
    offsets_value = (
        beam_offsets_xy_m
        if beam_offsets_xy_m is not None
        else aperture_centre_offsets_xy_m
    )
    if offsets_value is None:
        offsets = jnp.zeros((n_states, 1, 2), dtype=mean.dtype)
    else:
        offsets = _runtime_real_array(offsets_name, offsets_value)
        if offsets.shape == (2,):
            offsets = offsets[None, :]
        if offsets.ndim == 2 and offsets.shape[1] == 2:
            offsets = jnp.broadcast_to(
                offsets[None, :, :], (n_states, offsets.shape[0], 2)
            )
        elif offsets.ndim != 3 or offsets.shape[0] != n_states or offsets.shape[2] != 2:
            raise ValueError(
                f"{offsets_name} must have shape (2,), (n_positions, 2), or "
                "(n_states, n_positions, 2)."
            )

    zeros = jnp.zeros_like(offsets)
    if beam_offsets_xy_m is None:
        scan_beam_mean = mean[:, None, :] + zeros
        scan_centre = centre[:, None, :] + offsets
    else:
        scan_beam_mean = mean[:, None, :] + offsets
        scan_centre = centre[:, None, :] + zeros
    relative_mean = scan_beam_mean - scan_centre

    def one_state(
        state_relative_mean: jnp.ndarray,
        state_covariance: jnp.ndarray,
        state_radius: jnp.ndarray,
    ) -> jnp.ndarray:
        return jax.vmap(
            lambda relative: circular_gaussian_aperture_transmission(
                relative,
                state_covariance,
                state_radius,
                radial_order=radial_order,
                azimuthal_order=azimuthal_order,
            )
        )(state_relative_mean)

    transmission = jax.vmap(one_state)(relative_mean, covariance, radius)
    expected_counts = transmission * dose[:, None]
    return ApertureEdgeScanPrediction(
        beam_mean_xy_m=mean,
        beam_covariance_xy_m2=covariance,
        aperture_centre_xy_m=centre,
        aperture_radius_m=radius,
        incident_dose_electrons=dose,
        scan_beam_mean_xy_m=scan_beam_mean,
        scan_aperture_centre_xy_m=scan_centre,
        relative_beam_mean_xy_m=relative_mean,
        transmission=transmission,
        expected_transmitted_electrons=expected_counts,
        aperture_out_expected_electrons=dose,
    )


def condenser_aperture_edge_scan(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    dac_values: np.ndarray,
    source_mean_state: Sequence[float],
    source_covariance: np.ndarray,
    *,
    aperture_z_m: float,
    aperture_centre_xy_m: np.ndarray,
    aperture_radius_m: float | np.ndarray,
    incident_dose_electrons: float | np.ndarray,
    aperture_centre_offsets_xy_m: np.ndarray | None = None,
    beam_offsets_xy_m: np.ndarray | None = None,
    radial_order: int = 32,
    azimuthal_order: int = 96,
) -> ApertureEdgeScanPrediction:
    """Predict a condenser-aperture edge scan through the hard-edge column.

    ``dac_values`` may contain one fixed optical state or a batch of states.
    The returned aperture-plane mean and covariance are the actual propagated
    first-order moments used by the circular integral, making it possible to
    audit the physical measurement plane independently of the count curve.
    Aperture geometry and dose remain explicit per-state nuisance arguments.
    """

    controls = _runtime_real_array("dac_values", dac_values)
    if controls.ndim == 1:
        controls = controls[None, :]
    if controls.ndim != 2 or controls.shape[1] != len(column.channel_names):
        raise ValueError(
            "dac_values must have shape (n_channels,) or (n_states, n_channels)."
        )
    mean = _runtime_real_array("source_mean_state", source_mean_state)
    covariance = jnp.asarray(_state_covariance(source_covariance))
    if mean.shape != (4,):
        raise ValueError("source_mean_state must have shape (4,).")

    def one_state(dacs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        matrix = hard_edge_matrix_to_plane(
            column, parameters, dacs, aperture_z_m
        )
        return position_moments(matrix, mean, covariance)

    beam_mean, beam_covariance = jax.vmap(one_state)(controls)
    return aperture_edge_scan_from_moments(
        beam_mean,
        beam_covariance,
        aperture_centre_xy_m=aperture_centre_xy_m,
        aperture_radius_m=aperture_radius_m,
        incident_dose_electrons=incident_dose_electrons,
        aperture_centre_offsets_xy_m=aperture_centre_offsets_xy_m,
        beam_offsets_xy_m=beam_offsets_xy_m,
        radial_order=radial_order,
        azimuthal_order=azimuthal_order,
    )


@dataclass(frozen=True)
class ApertureEdgeScanDesign:
    """Immutable raw-frame contract for registered aperture-edge scans.

    For each unit normal ``n`` and signed edge offset ``s``, the aperture
    centre is commanded to ``n * (reference_radius_m + s)`` relative to the
    calibrated beam-centred position.  These are known commands, not fitted
    aperture-centre nuisances.  The latter remain explicit inputs to
    :func:`condenser_aperture_edge_scan`.
    """

    scan_normals_xy: np.ndarray
    signed_edge_offsets_m: np.ndarray
    reference_radius_m: float
    include_aperture_out_reference: bool = True

    def __post_init__(self) -> None:
        normals = np.asarray(self.scan_normals_xy, dtype=float)
        if normals.shape == (2,):
            normals = normals[None, :]
        if (
            normals.ndim != 2
            or normals.shape[1] != 2
            or normals.shape[0] < 1
            or not np.all(np.isfinite(normals))
        ):
            raise ValueError(
                "scan_normals_xy must have shape (2,) or (n_normals, 2)."
            )
        norms = np.linalg.norm(normals, axis=1)
        if np.any(norms <= 0.0):
            raise ValueError("scan_normals_xy must not contain a zero vector.")
        normals = normals / norms[:, None]
        signed = np.asarray(self.signed_edge_offsets_m, dtype=float)
        if signed.ndim != 1 or signed.size < 1 or not np.all(np.isfinite(signed)):
            raise ValueError(
                "signed_edge_offsets_m must be a non-empty finite vector."
            )
        if signed.size > 1 and np.any(np.diff(signed) <= 0.0):
            raise ValueError("signed_edge_offsets_m must be strictly increasing.")
        radius = float(self.reference_radius_m)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("reference_radius_m must be finite and positive.")
        if np.any(radius + signed <= 0.0):
            raise ValueError(
                "Every reference_radius_m + signed edge offset must be positive."
            )
        for name, values in (
            ("scan_normals_xy", normals),
            ("signed_edge_offsets_m", signed),
        ):
            frozen = values.copy()
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)
        object.__setattr__(self, "reference_radius_m", radius)
        object.__setattr__(
            self,
            "include_aperture_out_reference",
            bool(self.include_aperture_out_reference),
        )

    @property
    def n_normals(self) -> int:
        return int(self.scan_normals_xy.shape[0])

    @property
    def n_offsets(self) -> int:
        return int(self.signed_edge_offsets_m.size)

    @property
    def n_scan_frames(self) -> int:
        return self.n_normals * self.n_offsets

    @property
    def n_reference_frames(self) -> int:
        return int(self.include_aperture_out_reference)

    @property
    def n_frames(self) -> int:
        return self.n_scan_frames + self.n_reference_frames

    @property
    def aperture_centre_offsets_xy_m(self) -> np.ndarray:
        offsets = self.scan_normals_xy[:, None, :] * (
            self.reference_radius_m + self.signed_edge_offsets_m[None, :, None]
        )
        flattened = np.asarray(offsets.reshape((-1, 2)))
        flattened.setflags(write=False)
        return flattened

    @property
    def scan_normal_indices(self) -> np.ndarray:
        indices = np.repeat(np.arange(self.n_normals), self.n_offsets)
        indices.setflags(write=False)
        return indices

    @property
    def scan_signed_edge_offsets_m(self) -> np.ndarray:
        offsets = np.tile(self.signed_edge_offsets_m, self.n_normals)
        offsets.setflags(write=False)
        return offsets

    @property
    def frame_kinds(self) -> tuple[str, ...]:
        kinds = ("aperture_edge",) * self.n_scan_frames
        if self.include_aperture_out_reference:
            kinds += ("aperture_out_dose_reference",)
        return kinds

    def raw_expected_counts(
        self, prediction: ApertureEdgeScanPrediction
    ) -> jnp.ndarray:
        """Assemble raw expected counts in the declared frame order."""

        if prediction.expected_transmitted_electrons.ndim != 2:
            raise ValueError("prediction must retain its optical-state axis.")
        if prediction.expected_transmitted_electrons.shape[1] != self.n_scan_frames:
            raise ValueError(
                "prediction scan-position count does not match this design."
            )
        blocks = [prediction.expected_transmitted_electrons]
        if self.include_aperture_out_reference:
            blocks.append(prediction.aperture_out_expected_electrons[:, None])
        return jnp.concatenate(tuple(blocks), axis=1)


def three_normal_aperture_edge_design(
    reference_radius_m: float,
    edge_step_m: float,
    *,
    include_aperture_out_reference: bool = True,
) -> ApertureEdgeScanDesign:
    """Return the isotropic three-normal, seven-offset raw-image contract.

    Normals at 0, 120, and 240 degrees give a balanced two-dimensional centre
    constraint.  Each normal receives offsets ``[-3,-2,...,+3] * edge_step``
    around the certified reference edge.  With the default aperture-out dose
    reference this is exactly ``3 * 7 + 1 = 22`` raw detector frames per fixed
    optical state.  The raw count vector, rather than a fitted sigmoid or a
    normalised transmission curve, is the intended likelihood input.
    """

    step = float(edge_step_m)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("edge_step_m must be finite and positive.")
    angles = 2.0 * np.pi * np.arange(3, dtype=float) / 3.0
    normals = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    return ApertureEdgeScanDesign(
        scan_normals_xy=normals,
        signed_edge_offsets_m=np.arange(-3, 4, dtype=float) * step,
        reference_radius_m=reference_radius_m,
        include_aperture_out_reference=include_aperture_out_reference,
    )


@dataclass(frozen=True)
class ApertureSweep:
    """Absolute aperture transmission and incident radius along a DAC scan."""

    scan_channel_index: int
    scan_values: np.ndarray
    transmission: np.ndarray
    incident_rms_radius_m: np.ndarray


def condenser_aperture_sweep(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    reference_dac_values: Sequence[float],
    source_mean_state: Sequence[float],
    source_covariance: np.ndarray,
    *,
    aperture_z_m: float,
    aperture_radius_m: float,
    scan_channel_index: int,
    scan_values: Sequence[float],
    radial_order: int = 32,
    azimuthal_order: int = 96,
) -> ApertureSweep:
    """Predict the absolute-current curve used to calibrate condenser optics."""

    scan_channel = int(scan_channel_index)
    if scan_channel < 0 or scan_channel >= len(column.channel_names):
        raise ValueError("scan_channel_index is outside the column controls.")
    scan = np.asarray(scan_values, dtype=float)
    reference = np.asarray(reference_dac_values, dtype=float)
    mean = _state_vector("source_mean_state", source_mean_state)
    covariance = _state_covariance(source_covariance)
    if scan.ndim != 1 or scan.size < 2 or np.any(np.diff(scan) <= 0.0):
        raise ValueError("scan_values must be a strictly increasing vector.")
    if reference.shape != (len(column.channel_names),):
        raise ValueError("reference_dac_values has the wrong number of controls.")
    transmissions: list[float] = []
    radii: list[float] = []
    for value in scan:
        dacs = reference.copy()
        dacs[scan_channel] = value
        matrix = hard_edge_matrix_to_plane(
            column, parameters, dacs, aperture_z_m
        )
        transverse_mean, transverse_covariance = position_moments(
            matrix, mean, covariance
        )
        transmissions.append(
            float(
                circular_gaussian_aperture_transmission(
                    transverse_mean,
                    transverse_covariance,
                    aperture_radius_m,
                    radial_order=radial_order,
                    azimuthal_order=azimuthal_order,
                )
            )
        )
        radii.append(float(rms_beam_radius(transverse_covariance)))
    return ApertureSweep(
        scan_channel,
        scan.copy(),
        np.asarray(transmissions),
        np.asarray(radii),
    )


def condenser_parameter_indices(
    column: HardEdgeColumnParameterization,
    lens_names: Sequence[str] = ("CL1", "CL3", "C-mini"),
) -> np.ndarray:
    """Return the physical-parameter columns belonging to named condensers."""

    requested = tuple(str(name) for name in lens_names)
    available = {lens.name for lens in column.lenses}
    missing = tuple(name for name in requested if name not in available)
    if missing:
        raise ValueError(f"Unknown condenser lens names: {missing}.")
    indices = np.asarray(
        [
            index
            for index, name in enumerate(column.parameter_names)
            if any(name.startswith(f"{lens_name}.") for lens_name in requested)
        ],
        dtype=int,
    )
    indices.setflags(write=False)
    return indices


def _rank_from_singular_values(
    singular_values: np.ndarray,
    rtol: float,
) -> tuple[int, float]:
    largest = float(singular_values[0]) if singular_values.size else 0.0
    threshold = float(rtol * largest)
    return int(np.sum(singular_values > threshold)), threshold


@dataclass(frozen=True)
class CondenserIdentifiabilityAudit:
    """Local rank of a noise- and parameter-scaled condenser experiment.

    The rank is conditional on every parameter omitted from
    ``parameter_indices``.  In particular, ``12/12`` for CL1/CL3/C-mini does
    not imply that source-plane moments, the objective prefield, or a magnetic
    field profile are independently identified.
    """

    parameter_indices: np.ndarray
    parameter_names: tuple[str, ...]
    scaled_jacobian: np.ndarray
    singular_values: np.ndarray
    right_singular_vectors: np.ndarray
    rank: int
    threshold: float
    rank_rtol: float

    def __post_init__(self) -> None:
        indices = np.asarray(self.parameter_indices, dtype=int)
        jacobian = np.asarray(self.scaled_jacobian, dtype=float)
        singular = np.asarray(self.singular_values, dtype=float)
        right = np.asarray(self.right_singular_vectors, dtype=float)
        names = tuple(self.parameter_names)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("parameter_indices must be a non-empty vector.")
        if jacobian.ndim != 2 or jacobian.shape[1] != indices.size:
            raise ValueError("scaled_jacobian has the wrong parameter dimension.")
        if singular.shape != (min(jacobian.shape),):
            raise ValueError("singular_values has the wrong shape.")
        if right.shape != (indices.size, indices.size):
            raise ValueError("right_singular_vectors has the wrong shape.")
        if len(names) != indices.size:
            raise ValueError("parameter_names must match parameter_indices.")
        if not 0 <= int(self.rank) <= indices.size:
            raise ValueError("rank is outside the parameter dimension.")
        for name, value in (
            ("threshold", self.threshold),
            ("rank_rtol", self.rank_rtol),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        for field_name, array in (
            ("parameter_indices", indices),
            ("scaled_jacobian", jacobian),
            ("singular_values", singular),
            ("right_singular_vectors", right),
        ):
            frozen = array.copy()
            frozen.setflags(write=False)
            object.__setattr__(self, field_name, frozen)
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "rank", int(self.rank))

    @property
    def nullity(self) -> int:
        return len(self.parameter_names) - self.rank

    @property
    def relative_singular_values(self) -> np.ndarray:
        if self.singular_values.size == 0 or self.singular_values[0] == 0.0:
            return np.zeros_like(self.singular_values)
        values = self.singular_values / self.singular_values[0]
        values.setflags(write=False)
        return values


def audit_condenser_identifiability(
    column: HardEdgeColumnParameterization,
    prediction_function: Callable[[jnp.ndarray], jnp.ndarray],
    parameters: np.ndarray,
    *,
    standard_deviations: np.ndarray | float,
    parameter_scales: np.ndarray,
    lens_names: Sequence[str] = ("CL1", "CL3", "C-mini"),
    parameter_indices: Sequence[int] | None = None,
    rank_rtol: float = 1.0e-7,
) -> CondenserIdentifiabilityAudit:
    """Differentiate a physical likelihood and report conditional rank.

    Rows are divided by experimental standard deviations and columns are
    multiplied by declared physical parameter scales before the SVD.  This
    avoids declaring metre- or covariance-valued rows null merely because
    their numerical units are small.
    """

    values = np.asarray(parameters, dtype=float)
    scales = np.asarray(parameter_scales, dtype=float)
    if values.shape != (column.n_parameters,) or not np.all(np.isfinite(values)):
        raise ValueError(f"parameters must have shape ({column.n_parameters},).")
    if (
        scales.shape != values.shape
        or not np.all(np.isfinite(scales))
        or np.any(scales <= 0.0)
    ):
        raise ValueError(
            "parameter_scales must be finite, positive, and match parameters."
        )
    if not np.isfinite(rank_rtol) or rank_rtol <= 0.0:
        raise ValueError("rank_rtol must be finite and positive.")
    indices = (
        condenser_parameter_indices(column, lens_names)
        if parameter_indices is None
        else np.asarray(parameter_indices, dtype=int)
    )
    if (
        indices.ndim != 1
        or indices.size == 0
        or np.any(indices < 0)
        or np.any(indices >= column.n_parameters)
    ):
        raise ValueError("parameter_indices contains an invalid parameter column.")
    prediction = np.asarray(prediction_function(jnp.asarray(values)), dtype=float).reshape(-1)
    if prediction.size == 0 or not np.all(np.isfinite(prediction)):
        raise ValueError("prediction_function must return finite observations.")
    try:
        deviations = np.broadcast_to(
            np.asarray(standard_deviations, dtype=float), prediction.shape
        )
    except ValueError as error:
        raise ValueError("standard_deviations must broadcast to the prediction.") from error
    if not np.all(np.isfinite(deviations)) or np.any(deviations <= 0.0):
        raise ValueError("standard_deviations must be finite and positive.")
    jacobian = np.asarray(
        jax.jacfwd(lambda p: jnp.ravel(prediction_function(p)))(jnp.asarray(values))
    )
    if jacobian.shape != (prediction.size, column.n_parameters):
        raise ValueError("prediction_function returned an inconsistent JAX shape.")
    scaled = (
        jacobian[:, indices]
        * scales[indices][None, :]
        / deviations[:, None]
    )
    _, singular, right = np.linalg.svd(scaled, full_matrices=True)
    rank, threshold = _rank_from_singular_values(singular, float(rank_rtol))
    return CondenserIdentifiabilityAudit(
        parameter_indices=indices,
        parameter_names=tuple(column.parameter_names[index] for index in indices),
        scaled_jacobian=scaled,
        singular_values=singular,
        right_singular_vectors=right,
        rank=rank,
        threshold=threshold,
        rank_rtol=float(rank_rtol),
    )


@dataclass(frozen=True)
class EntranceCovarianceNuisanceAudit:
    """Blockwise optical rank after releasing all entrance covariance modes.

    ``diagnostics`` is the unmodified result from
    :func:`physical_nuisance.diagnose_blockwise_identifiability`.  It contains
    raw and column-equilibrated conditional, nuisance, joint, and
    nuisance-profiled spectra for every block and for their shared-nuisance
    combination.  The convenience properties below report the equilibrated
    structural ranks; raw spectra remain available for information-strength
    comparisons.
    """

    parameter_indices: np.ndarray
    parameter_names: tuple[str, ...]
    covariance_coordinate_names: tuple[str, ...]
    diagnostics: BlockwiseIdentifiabilityDiagnostics

    def __post_init__(self) -> None:
        indices = np.asarray(self.parameter_indices, dtype=int)
        names = tuple(self.parameter_names)
        covariance_names = tuple(self.covariance_coordinate_names)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("parameter_indices must be a non-empty vector.")
        if len(names) != indices.size:
            raise ValueError("parameter_names must match parameter_indices.")
        if len(covariance_names) != 10 or len(set(covariance_names)) != 10:
            raise ValueError("Expected ten unique covariance coordinate names.")
        if not isinstance(self.diagnostics, BlockwiseIdentifiabilityDiagnostics):
            raise TypeError("diagnostics must be BlockwiseIdentifiabilityDiagnostics.")
        if self.diagnostics.optical_parameter_count != indices.size:
            raise ValueError("diagnostics has the wrong optical parameter count.")
        if self.diagnostics.nuisance_parameter_count != len(covariance_names):
            raise ValueError("diagnostics has the wrong nuisance parameter count.")
        indices = indices.copy()
        indices.setflags(write=False)
        object.__setattr__(self, "parameter_indices", indices)
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "covariance_coordinate_names", covariance_names)

    @property
    def conditional_rank(self) -> int:
        """Column-equilibrated optical rank with covariance held fixed."""

        return self.diagnostics.combined.column_equilibrated_conditional_optical.rank

    @property
    def nuisance_profiled_rank(self) -> int:
        """Column-equilibrated optical rank after data-only nuisance profiling."""

        return self.diagnostics.combined.column_equilibrated_profiled_optical.rank

    @property
    def nuisance_rank(self) -> int:
        """Column-equilibrated rank of the ten covariance nuisance columns."""

        return self.diagnostics.combined.column_equilibrated_nuisance.rank


@dataclass(frozen=True)
class AxialPlaneCovarianceBlock:
    """One calibrated covariance-tomography block at a fixed axial plane.

    ``mode='position'`` declares the three covariance summaries available
    from an intensity screen.  ``mode='full'`` declares all ten phase-space
    covariance summaries and therefore carries the stronger experimental
    requirement documented by :func:`axial_plane_covariance_observables`.
    Standard deviations may be scalar, one value per summary, one value per
    state and summary, or a fully flattened vector.
    """

    name: str
    plane_z_m: float
    dac_values: np.ndarray
    standard_deviations: np.ndarray | float
    mode: str = "full"

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        plane = float(self.plane_z_m)
        controls = np.asarray(self.dac_values, dtype=float)
        deviations = np.asarray(self.standard_deviations, dtype=float)
        mode = str(self.mode)
        if not name:
            raise ValueError("name must be non-empty.")
        if not np.isfinite(plane):
            raise ValueError("plane_z_m must be finite.")
        if controls.ndim == 1:
            controls = controls[None, :]
        if controls.ndim != 2 or controls.shape[0] == 0 or not np.all(
            np.isfinite(controls)
        ):
            raise ValueError(
                "dac_values must be a finite one- or two-dimensional array."
            )
        if mode not in _COVARIANCE_MODE_INDICES:
            raise ValueError(
                f"mode must be one of {tuple(_COVARIANCE_MODE_INDICES)}, got {mode!r}."
            )
        component_count = len(_COVARIANCE_MODE_INDICES[mode])
        permitted_shapes = {
            (),
            (component_count,),
            (controls.shape[0], component_count),
            (controls.shape[0] * component_count,),
        }
        if deviations.shape not in permitted_shapes:
            raise ValueError(
                "standard_deviations must be scalar, one value per covariance "
                "summary, one value per state/summary, or fully flattened."
            )
        if not np.all(np.isfinite(deviations)) or np.any(deviations <= 0.0):
            raise ValueError("standard_deviations must be finite and positive.")
        controls = controls.copy()
        deviations = deviations.copy()
        controls.setflags(write=False)
        deviations.setflags(write=False)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "plane_z_m", plane)
        object.__setattr__(self, "dac_values", controls)
        object.__setattr__(self, "standard_deviations", deviations)
        object.__setattr__(self, "mode", mode)

    @property
    def observable_names(self) -> tuple[str, ...]:
        if self.mode == "position":
            return POSITION_COVARIANCE_OBSERVABLE_NAMES
        return PHASE_SPACE_COVARIANCE_OBSERVABLE_NAMES

    def flattened_standard_deviations(self) -> np.ndarray:
        """Return one immutable standard deviation per flattened prediction."""

        shape = (self.dac_values.shape[0], len(self.observable_names))
        deviations = np.asarray(self.standard_deviations)
        if deviations.shape == (shape[0] * shape[1],):
            flattened = deviations.copy()
        else:
            flattened = np.broadcast_to(deviations, shape).reshape(-1).copy()
        flattened.setflags(write=False)
        return flattened


def audit_axial_plane_covariance_nuisance_identifiability(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    covariance_chart: EntranceCovarianceChart,
    covariance_blocks: Sequence[AxialPlaneCovarianceBlock],
    *,
    parameter_scales: np.ndarray,
    parameter_indices: Sequence[int],
    baseline_prediction_function: Callable[
        [jnp.ndarray, jnp.ndarray], jnp.ndarray
    ]
    | None = None,
    baseline_standard_deviations: np.ndarray | float | None = None,
    baseline_block_name: str = "baseline",
    rank_rtol: float = 1.0e-7,
    column_norm_atol: float = 0.0,
) -> EntranceCovarianceNuisanceAudit:
    """Audit optical rank from one or more known-plane covariance summaries.

    All blocks share the ten coordinates of ``covariance_chart``.  An optional
    baseline callback receives the reconstructed full optical vector and the
    decoded entrance covariance, which allows critical-line, aperture-count,
    and specimen-plane probe observations to be combined without teaching
    this reusable audit about their individual likelihoods.

    A position-only block is intentionally not promoted to a phase-space
    measurement: it contributes only two widths and one transverse
    correlation.  A ``full`` block requires an experimentally calibrated
    reconstruction of all four widths and six correlations.  As with the
    existing entrance-covariance audit, this is a local, data-only rank test;
    it adds no prior to manufacture identifiability.
    """

    if not isinstance(covariance_chart, EntranceCovarianceChart):
        raise TypeError("covariance_chart must be an EntranceCovarianceChart.")
    blocks = tuple(covariance_blocks)
    if not blocks or any(
        not isinstance(block, AxialPlaneCovarianceBlock) for block in blocks
    ):
        raise ValueError(
            "covariance_blocks must contain at least one AxialPlaneCovarianceBlock."
        )
    block_names = tuple(block.name for block in blocks)
    if len(set(block_names)) != len(block_names):
        raise ValueError("Covariance block names must be unique.")
    baseline_name = str(baseline_block_name).strip()
    if not baseline_name or (
        baseline_prediction_function is not None and baseline_name in block_names
    ):
        raise ValueError("baseline_block_name must be non-empty and unique.")
    if (baseline_prediction_function is None) != (
        baseline_standard_deviations is None
    ):
        raise ValueError(
            "baseline_prediction_function and baseline_standard_deviations "
            "must be supplied together."
        )

    values = np.asarray(parameters, dtype=float)
    scales = np.asarray(parameter_scales, dtype=float)
    indices = np.asarray(parameter_indices, dtype=int)
    if values.shape != (column.n_parameters,) or not np.all(np.isfinite(values)):
        raise ValueError(f"parameters must have shape ({column.n_parameters},).")
    if (
        scales.shape != values.shape
        or not np.all(np.isfinite(scales))
        or np.any(scales <= 0.0)
    ):
        raise ValueError(
            "parameter_scales must be finite, positive, and match parameters."
        )
    if (
        indices.ndim != 1
        or indices.size == 0
        or len(np.unique(indices)) != indices.size
        or np.any(indices < 0)
        or np.any(indices >= column.n_parameters)
    ):
        raise ValueError("parameter_indices contains an invalid or repeated column.")
    if not np.isfinite(rank_rtol) or not 0.0 < rank_rtol < 1.0:
        raise ValueError("rank_rtol must lie strictly in (0, 1).")
    if not np.isfinite(column_norm_atol) or column_norm_atol < 0.0:
        raise ValueError("column_norm_atol must be finite and non-negative.")
    channel_count = len(column.channel_names)
    for block in blocks:
        if block.dac_values.shape[1] != channel_count:
            raise ValueError(
                f"Covariance block {block.name!r} has the wrong DAC dimension."
            )
        # Perform the fixed-plane membership check before entering a JAX trace.
        hard_edge_matrix_to_plane(
            column, values, block.dac_values[0], block.plane_z_m
        )

    selected_reference = values[indices]
    joint_reference = jnp.asarray(
        np.concatenate((selected_reference, np.zeros(covariance_chart.n_coordinates)))
    )
    jax_indices = jnp.asarray(indices)

    def decode_joint(joint: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        optical = jnp.asarray(values).at[jax_indices].set(joint[: indices.size])
        covariance = covariance_chart.covariance(joint[indices.size :])
        return optical, covariance

    def whitened_jacobian(
        prediction_function: Callable[[jnp.ndarray], jnp.ndarray],
        standard_deviations: np.ndarray | float,
    ) -> np.ndarray:
        prediction = np.asarray(
            prediction_function(joint_reference), dtype=float
        ).reshape(-1)
        if prediction.size == 0 or not np.all(np.isfinite(prediction)):
            raise ValueError("Prediction blocks must be finite and non-empty.")
        try:
            deviations = np.broadcast_to(
                np.asarray(standard_deviations, dtype=float), prediction.shape
            )
        except ValueError as error:
            raise ValueError(
                "Standard deviations must broadcast to their prediction block."
            ) from error
        if not np.all(np.isfinite(deviations)) or np.any(deviations <= 0.0):
            raise ValueError("Standard deviations must be finite and positive.")
        jacobian = np.asarray(
            jax.jacfwd(lambda joint: jnp.ravel(prediction_function(joint)))(
                joint_reference
            ),
            dtype=float,
        )
        if jacobian.shape != (prediction.size, joint_reference.size):
            raise RuntimeError("Prediction block returned an inconsistent JAX shape.")
        return jacobian / deviations[:, None]

    jacobian_blocks: list[tuple[str, np.ndarray]] = []
    if baseline_prediction_function is not None:

        def baseline_prediction(joint: jnp.ndarray) -> jnp.ndarray:
            optical, covariance = decode_joint(joint)
            return jnp.ravel(baseline_prediction_function(optical, covariance))

        jacobian_blocks.append(
            (
                baseline_name,
                whitened_jacobian(
                    baseline_prediction,
                    baseline_standard_deviations,
                ),
            )
        )

    for block in blocks:

        def covariance_prediction(
            joint: jnp.ndarray,
            block: AxialPlaneCovarianceBlock = block,
        ) -> jnp.ndarray:
            optical = jnp.asarray(values).at[jax_indices].set(
                joint[: indices.size]
            )
            return axial_plane_covariance_observables_with_entrance_covariance(
                column,
                optical,
                joint[indices.size :],
                block.dac_values,
                covariance_chart,
                plane_z_m=block.plane_z_m,
                mode=block.mode,
            ).reshape(-1)

        jacobian_blocks.append(
            (
                block.name,
                whitened_jacobian(
                    covariance_prediction,
                    block.flattened_standard_deviations(),
                ),
            )
        )

    diagnostics = diagnose_blockwise_identifiability(
        jacobian_blocks,
        optical_parameter_count=indices.size,
        parameter_scales=np.concatenate(
            (scales[indices], np.ones(covariance_chart.n_coordinates))
        ),
        rank_rtol=rank_rtol,
        column_norm_atol=column_norm_atol,
    )
    return EntranceCovarianceNuisanceAudit(
        parameter_indices=indices,
        parameter_names=tuple(column.parameter_names[index] for index in indices),
        covariance_coordinate_names=covariance_chart.coordinate_names,
        diagnostics=diagnostics,
    )


def audit_entrance_covariance_nuisance_identifiability(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    probe_dac_values: np.ndarray,
    source_mean_state: Sequence[float],
    covariance_chart: EntranceCovarianceChart,
    *,
    probe_standard_deviations: np.ndarray | float,
    parameter_scales: np.ndarray,
    mode: str = "complex_radial",
    lens_names: Sequence[str] = ("CL1", "CL3", "C-mini"),
    parameter_indices: Sequence[int] | None = None,
    baseline_prediction_function: Callable[
        [jnp.ndarray, jnp.ndarray], jnp.ndarray
    ]
    | None = None,
    baseline_standard_deviations: np.ndarray | float | None = None,
    baseline_block_name: str = "condenser_reconnaissance",
    probe_block_name: str = "sample_plane_probes",
    rank_rtol: float = 1.0e-7,
    column_norm_atol: float = 0.0,
) -> EntranceCovarianceNuisanceAudit:
    """Audit optical rank while profiling a shared entrance covariance.

    The required probe block is evaluated with
    :func:`sample_plane_probe_observables_with_entrance_covariance`.  An
    optional baseline callback can add the critical-line/aperture experiment;
    it receives the full optical parameter vector and decoded 4x4 entrance
    covariance, so the same callback can later be reused in a nonlinear
    residual.  Every row is whitened by its declared experimental standard
    deviation before the existing blockwise physical-nuisance diagnostic is
    called.  The ten covariance coordinates remain shared across all blocks.

    This is a local data-only audit.  It deliberately does not add a prior to
    make a weak optical direction appear identified.
    """

    if not isinstance(covariance_chart, EntranceCovarianceChart):
        raise TypeError("covariance_chart must be an EntranceCovarianceChart.")
    values = np.asarray(parameters, dtype=float)
    scales = np.asarray(parameter_scales, dtype=float)
    if values.shape != (column.n_parameters,) or not np.all(np.isfinite(values)):
        raise ValueError(f"parameters must have shape ({column.n_parameters},).")
    if (
        scales.shape != values.shape
        or not np.all(np.isfinite(scales))
        or np.any(scales <= 0.0)
    ):
        raise ValueError(
            "parameter_scales must be finite, positive, and match parameters."
        )
    if mode not in _PROBE_MODE_INDICES:
        raise ValueError(f"Unknown probe mode {mode!r}.")
    if not np.isfinite(rank_rtol) or not 0.0 < rank_rtol < 1.0:
        raise ValueError("rank_rtol must lie strictly in (0, 1).")
    if not np.isfinite(column_norm_atol) or column_norm_atol < 0.0:
        raise ValueError("column_norm_atol must be finite and non-negative.")
    indices = (
        condenser_parameter_indices(column, lens_names)
        if parameter_indices is None
        else np.asarray(parameter_indices, dtype=int)
    )
    if (
        indices.ndim != 1
        or indices.size == 0
        or len(np.unique(indices)) != indices.size
        or np.any(indices < 0)
        or np.any(indices >= column.n_parameters)
    ):
        raise ValueError("parameter_indices contains an invalid or repeated column.")
    baseline_name = str(baseline_block_name).strip()
    probe_name = str(probe_block_name).strip()
    if not baseline_name or not probe_name or baseline_name == probe_name:
        raise ValueError("Block names must be non-empty and distinct.")
    if (baseline_prediction_function is None) != (
        baseline_standard_deviations is None
    ):
        raise ValueError(
            "baseline_prediction_function and baseline_standard_deviations "
            "must be supplied together."
        )

    controls = np.asarray(probe_dac_values, dtype=float)
    if controls.ndim == 1:
        controls = controls[None, :]
    if (
        controls.ndim != 2
        or controls.shape[1] != len(column.channel_names)
        or controls.shape[0] == 0
        or not np.all(np.isfinite(controls))
    ):
        raise ValueError("probe_dac_values must be finite (n_states, n_channels).")
    source_mean = _state_vector("source_mean_state", source_mean_state)
    selected_reference = values[indices]
    joint_reference = jnp.asarray(
        np.concatenate((selected_reference, np.zeros(covariance_chart.n_coordinates)))
    )
    jax_indices = jnp.asarray(indices)

    def decode_joint(joint: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        optical = jnp.asarray(values).at[jax_indices].set(joint[: indices.size])
        covariance = covariance_chart.covariance(joint[indices.size :])
        return optical, covariance

    def probe_prediction(joint: jnp.ndarray) -> jnp.ndarray:
        optical = jnp.asarray(values).at[jax_indices].set(joint[: indices.size])
        return sample_plane_probe_observables_with_entrance_covariance(
            column,
            optical,
            joint[indices.size :],
            controls,
            source_mean,
            covariance_chart,
            mode=mode,
        ).reshape(-1)

    def whitened_jacobian(
        prediction_function: Callable[[jnp.ndarray], jnp.ndarray],
        standard_deviations: np.ndarray | float,
    ) -> np.ndarray:
        prediction = np.asarray(
            prediction_function(joint_reference), dtype=float
        ).reshape(-1)
        if prediction.size == 0 or not np.all(np.isfinite(prediction)):
            raise ValueError("Prediction blocks must be finite and non-empty.")
        try:
            deviations = np.broadcast_to(
                np.asarray(standard_deviations, dtype=float), prediction.shape
            )
        except ValueError as error:
            raise ValueError(
                "Standard deviations must broadcast to their prediction block."
            ) from error
        if not np.all(np.isfinite(deviations)) or np.any(deviations <= 0.0):
            raise ValueError("Standard deviations must be finite and positive.")
        jacobian = np.asarray(
            jax.jacfwd(lambda joint: jnp.ravel(prediction_function(joint)))(
                joint_reference
            ),
            dtype=float,
        )
        if jacobian.shape != (prediction.size, joint_reference.size):
            raise RuntimeError("Prediction block returned an inconsistent JAX shape.")
        return jacobian / deviations[:, None]

    blocks: list[tuple[str, np.ndarray]] = []
    if baseline_prediction_function is not None:

        def baseline_prediction(joint: jnp.ndarray) -> jnp.ndarray:
            optical, covariance = decode_joint(joint)
            return jnp.ravel(baseline_prediction_function(optical, covariance))

        blocks.append(
            (
                baseline_name,
                whitened_jacobian(
                    baseline_prediction,
                    baseline_standard_deviations,
                ),
            )
        )
    blocks.append(
        (probe_name, whitened_jacobian(probe_prediction, probe_standard_deviations))
    )
    joint_scales = np.concatenate(
        (scales[indices], np.ones(covariance_chart.n_coordinates))
    )
    diagnostics = diagnose_blockwise_identifiability(
        blocks,
        optical_parameter_count=indices.size,
        parameter_scales=joint_scales,
        rank_rtol=rank_rtol,
        column_norm_atol=column_norm_atol,
    )
    return EntranceCovarianceNuisanceAudit(
        parameter_indices=indices,
        parameter_names=tuple(column.parameter_names[index] for index in indices),
        covariance_coordinate_names=covariance_chart.coordinate_names,
        diagnostics=diagnostics,
    )


@dataclass(frozen=True)
class CondenserProbeDesignSelection:
    """Greedy rank-revealing selection of extra specimen probe states."""

    selected_candidate_indices: np.ndarray
    selected_dac_values: np.ndarray
    scaled_jacobian: np.ndarray
    singular_values: np.ndarray
    baseline_rank: int
    achieved_rank: int
    target_rank: int
    parameter_names: tuple[str, ...]
    observable_mode: str
    measurement_contract: str
    minimum_relative_singular_value: float | None = None
    achieved_relative_singular_value: float = 0.0

    def __post_init__(self) -> None:
        selected = np.asarray(self.selected_candidate_indices, dtype=int)
        dacs = np.asarray(self.selected_dac_values, dtype=float)
        jacobian = np.asarray(self.scaled_jacobian, dtype=float)
        singular = np.asarray(self.singular_values, dtype=float)
        names = tuple(self.parameter_names)
        if selected.ndim != 1 or dacs.ndim != 2 or dacs.shape[0] != selected.size:
            raise ValueError("Selected indices and DAC rows are inconsistent.")
        if jacobian.ndim != 2 or jacobian.shape[1] != len(names):
            raise ValueError("scaled_jacobian and parameter_names are inconsistent.")
        if singular.shape != (min(jacobian.shape),):
            raise ValueError("singular_values has the wrong shape.")
        if not (
            0 <= int(self.baseline_rank)
            <= int(self.target_rank)
            <= len(names)
            and int(self.target_rank) >= 1
            and int(self.baseline_rank)
            <= int(self.achieved_rank)
            <= len(names)
        ):
            raise ValueError("Rank fields are inconsistent.")
        minimum_relative = self.minimum_relative_singular_value
        if minimum_relative is not None and (
            not np.isfinite(minimum_relative)
            or minimum_relative <= 0.0
            or minimum_relative > 1.0
        ):
            raise ValueError(
                "minimum_relative_singular_value must lie in (0, 1]."
            )
        if (
            not np.isfinite(self.achieved_relative_singular_value)
            or self.achieved_relative_singular_value < 0.0
            or self.achieved_relative_singular_value > 1.0
        ):
            raise ValueError(
                "achieved_relative_singular_value must lie in [0, 1]."
            )
        for field_name, array in (
            ("selected_candidate_indices", selected),
            ("selected_dac_values", dacs),
            ("scaled_jacobian", jacobian),
            ("singular_values", singular),
        ):
            array = array.copy()
            array.setflags(write=False)
            object.__setattr__(self, field_name, array)
        object.__setattr__(self, "parameter_names", names)

    @property
    def reached_rank_target(self) -> bool:
        return self.achieved_rank >= self.target_rank

    @property
    def conditioning_target_met(self) -> bool:
        return (
            self.minimum_relative_singular_value is None
            or self.achieved_relative_singular_value
            >= self.minimum_relative_singular_value
        )

    @property
    def reached_target(self) -> bool:
        return self.reached_rank_target and self.conditioning_target_met


def greedy_condenser_probe_design(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    candidate_dac_values: np.ndarray,
    source_mean_state: Sequence[float],
    source_covariance: np.ndarray,
    *,
    baseline_scaled_jacobian: np.ndarray,
    parameter_scales: np.ndarray,
    observable_standard_deviations: np.ndarray | float,
    observable_mode: str = "complex_radial",
    lens_names: Sequence[str] = ("CL1", "CL3", "C-mini"),
    parameter_indices: Sequence[int] | None = None,
    rank_rtol: float = 1.0e-7,
    target_rank: int | None = None,
    maximum_states: int | None = None,
    minimum_relative_singular_value: float | None = None,
) -> CondenserProbeDesignSelection:
    """Select the smallest candidate-state prefix that reaches target rank.

    Selection first maximizes rank, then the pseudo-log-determinant of the
    resolved singular values.  Each candidate is one *optical state*; the
    number of detector frames needed to obtain its observable block depends on
    ``observable_mode``.  For ``complex_radial`` the physical contract is a
    registered position image, a diffraction image, and one defocus/complex-
    probe curvature measurement on the same specimen.  By default selection
    stops as soon as ``target_rank`` is reached.  Setting
    ``minimum_relative_singular_value`` keeps adding states until the target
    singular direction also reaches the requested fraction of the largest
    singular value, or until ``maximum_states`` is exhausted.
    """

    if observable_mode not in _PROBE_MODE_INDICES:
        raise ValueError(f"Unknown observable_mode {observable_mode!r}.")
    values = np.asarray(parameters, dtype=float)
    candidates = np.asarray(candidate_dac_values, dtype=float)
    scales = np.asarray(parameter_scales, dtype=float)
    if values.shape != (column.n_parameters,) or scales.shape != values.shape:
        raise ValueError("parameters and parameter_scales must match the column.")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("parameters and parameter_scales must be finite; scales positive.")
    if candidates.ndim != 2 or candidates.shape[1] != len(column.channel_names):
        raise ValueError("candidate_dac_values must have shape (n_states, n_channels).")
    if candidates.shape[0] == 0 or not np.all(np.isfinite(candidates)):
        raise ValueError("At least one finite candidate DAC row is required.")
    if not np.isfinite(rank_rtol) or rank_rtol <= 0.0:
        raise ValueError("rank_rtol must be finite and positive.")
    if minimum_relative_singular_value is not None and (
        not np.isfinite(minimum_relative_singular_value)
        or minimum_relative_singular_value <= 0.0
        or minimum_relative_singular_value > 1.0
    ):
        raise ValueError(
            "minimum_relative_singular_value must lie in (0, 1]."
        )
    indices = (
        condenser_parameter_indices(column, lens_names)
        if parameter_indices is None
        else np.asarray(parameter_indices, dtype=int)
    )
    if (
        indices.ndim != 1
        or indices.size == 0
        or np.any(indices < 0)
        or np.any(indices >= column.n_parameters)
    ):
        raise ValueError("parameter_indices contains an invalid parameter column.")
    baseline = np.asarray(baseline_scaled_jacobian, dtype=float)
    if baseline.ndim != 2:
        raise ValueError("baseline_scaled_jacobian must be two-dimensional.")
    if baseline.shape[1] == column.n_parameters:
        baseline = baseline[:, indices]
    elif baseline.shape[1] != indices.size:
        raise ValueError("baseline_scaled_jacobian has the wrong number of columns.")
    if not np.all(np.isfinite(baseline)):
        raise ValueError("baseline_scaled_jacobian must be finite.")

    prediction = sample_plane_probe_observables(
        column,
        jnp.asarray(values),
        candidates,
        source_mean_state,
        source_covariance,
        mode=observable_mode,
    )
    observable_count = prediction.shape[1]
    try:
        deviations = np.broadcast_to(
            np.asarray(observable_standard_deviations, dtype=float),
            (observable_count,),
        )
    except ValueError as error:
        raise ValueError(
            "observable_standard_deviations must broadcast to one candidate block."
        ) from error
    if not np.all(np.isfinite(deviations)) or np.any(deviations <= 0.0):
        raise ValueError("observable_standard_deviations must be finite and positive.")
    candidate_jacobians = np.asarray(
        jax.jacfwd(
            lambda p: sample_plane_probe_observables(
                column,
                p,
                candidates,
                source_mean_state,
                source_covariance,
                mode=observable_mode,
            )
        )(jnp.asarray(values))
    )
    candidate_jacobians = (
        candidate_jacobians[:, :, indices]
        * scales[indices][None, None, :]
        / deviations[None, :, None]
    )

    baseline_singular = np.linalg.svd(baseline, compute_uv=False)
    baseline_rank, _ = _rank_from_singular_values(
        baseline_singular, float(rank_rtol)
    )
    requested_rank = indices.size if target_rank is None else int(target_rank)
    if (
        requested_rank < max(1, baseline_rank)
        or requested_rank > indices.size
    ):
        raise ValueError("target_rank must lie between baseline rank and parameter count.")
    limit = candidates.shape[0] if maximum_states is None else int(maximum_states)
    if limit < 0 or limit > candidates.shape[0]:
        raise ValueError("maximum_states is outside the candidate count.")

    selected: list[int] = []
    stacked = baseline.copy()
    achieved_rank = baseline_rank

    def relative_target_singular_value(singular: np.ndarray) -> float:
        if singular.size < requested_rank or singular[0] <= 0.0:
            return 0.0
        return float(singular[requested_rank - 1] / singular[0])

    achieved_relative = relative_target_singular_value(baseline_singular)

    def target_is_met() -> bool:
        rank_met = achieved_rank >= requested_rank
        conditioning_met = (
            minimum_relative_singular_value is None
            or achieved_relative >= minimum_relative_singular_value
        )
        return rank_met and conditioning_met

    for _ in range(limit):
        if target_is_met():
            break
        best: tuple[tuple[float, ...], int, np.ndarray, np.ndarray] | None = None
        for candidate_index in range(candidates.shape[0]):
            if candidate_index in selected:
                continue
            trial = np.vstack((stacked, candidate_jacobians[candidate_index]))
            singular = np.linalg.svd(trial, compute_uv=False)
            rank, _ = _rank_from_singular_values(
                singular, float(rank_rtol)
            )
            # ``-candidate_index`` makes exact ties deterministic.
            pseudo_logdet = float(
                2.0 * np.sum(np.log(np.maximum(singular[:rank], np.finfo(float).tiny)))
            )
            if (
                minimum_relative_singular_value is None
                or achieved_rank < requested_rank
            ):
                # Preserve the historical rank-then-D-optimal ordering through
                # the state that first closes rank.
                score = (float(rank), pseudo_logdet, float(-candidate_index))
            else:
                relative = relative_target_singular_value(singular)
                # Once rank closes, explicitly improve the weakest requested
                # direction rather than letting a large leading mode dominate.
                score = (
                    float(min(rank, requested_rank)),
                    relative if rank >= requested_rank else 0.0,
                    pseudo_logdet,
                    float(-candidate_index),
                )
            if best is None or score > best[0]:
                best = (score, candidate_index, trial, singular)
        if best is None:  # pragma: no cover - guarded by candidate validation
            break
        _, candidate_index, stacked, singular_values = best
        selected.append(candidate_index)
        achieved_rank, _ = _rank_from_singular_values(
            singular_values, float(rank_rtol)
        )
        achieved_relative = relative_target_singular_value(singular_values)

    singular_values = np.linalg.svd(stacked, compute_uv=False)
    achieved_rank, _ = _rank_from_singular_values(
        singular_values, float(rank_rtol)
    )
    achieved_relative = relative_target_singular_value(singular_values)
    contracts = {
        "position": "registered perforation/sample position image",
        "angle": "calibrated diffraction-plane central-beam image",
        "complex_radial": (
            "registered position image + diffraction image + defocus/complex-probe curvature"
        ),
        "full": "complete phase-space audit including a coupled projection or entrance basis",
    }
    selected_array = np.asarray(selected, dtype=int)
    return CondenserProbeDesignSelection(
        selected_candidate_indices=selected_array,
        selected_dac_values=candidates[selected_array],
        scaled_jacobian=stacked,
        singular_values=singular_values,
        baseline_rank=baseline_rank,
        achieved_rank=achieved_rank,
        target_rank=requested_rank,
        parameter_names=tuple(column.parameter_names[index] for index in indices),
        observable_mode=observable_mode,
        measurement_contract=contracts[observable_mode],
        minimum_relative_singular_value=minimum_relative_singular_value,
        achieved_relative_singular_value=achieved_relative,
    )


__all__ = [
    "ApertureEdgeScanDesign",
    "ApertureEdgeScanPrediction",
    "ApertureSweep",
    "AxialPlaneCovarianceBlock",
    "CONDENSER_PROBE_OBSERVABLE_NAMES",
    "DEFLECTOR_SAMPLE_CENTRAL_DIFFERENCE_OBSERVABLE_NAMES",
    "DEFLECTOR_SAMPLE_POSITION_DIFFERENCE_OBSERVABLE_NAMES",
    "ENTRANCE_COVARIANCE_COORDINATE_NAMES",
    "HardEdgeInternalPlaneBounds",
    "HardEdgeInternalPlaneConstraint",
    "PHASE_SPACE_COVARIANCE_OBSERVABLE_NAMES",
    "POSITION_COVARIANCE_OBSERVABLE_NAMES",
    "CondenserIdentifiabilityAudit",
    "CondenserProbeDesignSelection",
    "CriticalLineScan",
    "DeflectorCentralDifference",
    "DeflectorCommandKickMap",
    "EntranceCovarianceChart",
    "EntranceCovarianceNuisanceAudit",
    "audit_axial_plane_covariance_nuisance_identifiability",
    "audit_condenser_identifiability",
    "audit_entrance_covariance_nuisance_identifiability",
    "axial_plane_covariance_observables",
    "axial_plane_covariance_observables_with_entrance_covariance",
    "aperture_edge_scan_from_moments",
    "central_deflector_command_difference",
    "circular_gaussian_aperture_transmission",
    "circular_perforation_signal",
    "condenser_aperture_edge_scan",
    "condenser_aperture_sweep",
    "condenser_parameter_indices",
    "condenser_reconnaissance_frame_observables",
    "condenser_reconnaissance_response",
    "critical_line_scan",
    "deflector_command_phase_space_moments",
    "detector_log_radius",
    "greedy_condenser_probe_design",
    "hard_edge_matrix_between_planes",
    "hard_edge_matrix_to_plane",
    "ideal_angular_kick_response_matrices",
    "linear_zero_crossings",
    "phase_space_covariance_observables",
    "phase_space_probe_observables",
    "position_moments",
    "propagate_phase_space_gaussian",
    "rms_beam_radius",
    "restrict_hard_edge_internal_plane_bounds",
    "restrict_hard_edge_internal_planes_bounds",
    "sample_perforation_edge_scan",
    "sample_plane_probe_observables",
    "sample_plane_probe_observables_with_entrance_covariance",
    "signed_wobble_response",
    "three_normal_aperture_edge_design",
    "validate_hard_edge_internal_plane_bounds",
    "validate_hard_edge_internal_planes_bounds",
]
