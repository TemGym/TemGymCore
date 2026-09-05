"""Payload-bound raw likelihood for the actual-geometry condenser Branch B.

Branch B replaces unavailable lens-current readback with three complementary
measurements: translated C-aperture edge counts, calibrated specimen-plane
phase-space covariance tomography, and central-difference responses from the
physical condenser double deflector.  This module turns the committed design
payload into a JAX-compatible nonlinear likelihood without inventing any
instrument observations.

The payload controls are normalized coordinates of the synthetic affine
``DAC -> q`` model.  They are not certified NEOARM hardware commands.  Loading
the experiment fails unless that warning and all bound file hashes are intact.
An instrument run must map the normalized states onto measured, safe mode
continuations before using this likelihood with real frames.

The fitted physical lens is an effective hard-edge solenoid.  The likelihood
can identify a predictive raw-control-to-integrated-strength law, but it cannot
factor that law into DAC-to-current and current-to-field gains or prove a
rectangular axial field profile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import tomllib
from types import MappingProxyType
from typing import Any, Literal, Mapping, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from .critical_line_calibration import (
    ENTRANCE_COVARIANCE_COORDINATE_NAMES,
    PHASE_SPACE_COVARIANCE_OBSERVABLE_NAMES,
    EntranceCovarianceChart,
    aperture_edge_scan_from_moments,
    axial_plane_covariance_observables_with_entrance_covariance,
    hard_edge_matrix_to_plane,
    ideal_angular_kick_response_matrices,
)
from .neoarm_calibration import (
    HardEdgeColumnParameterization,
    hard_edge_column_from_config,
)


BRANCH_B_SCHEMA = "temgym.neoarm.condenser-branch-b-design/v1"
EXPECTED_BRANCH_B_PAYLOAD_SHA256 = (
    "97bcfb0f262ea6ba3e0ec6c49d077d3777166c36bef24d6a8540cdadc843714d"
)
NORMALIZED_CONTROL_WARNING = (
    "normalized synthetic model controls; not certified hardware DAC commands"
)
DEFAULT_BRANCH_B_PAYLOAD = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "lens_inversion"
    / "neoarm_condenser_branch_b_design.json"
)

_EXPECTED_STATE_COUNTS = {"edge": 4, "tomography": 32, "deflector": 3}
_EXPECTED_RAW_FRAMES = {"edge": 22, "tomography": 5, "deflector": 16}
_DEFLECTOR_STATE_NAMES = (
    "sample_x_m",
    "sample_theta_x_rad",
    "sample_y_m",
    "sample_theta_y_rad",
)
_DEFLECTOR_NOMINAL_MAP = np.asarray(
    (
        (1.0, 0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0, 1.0),
        (1.0, 0.0, -1.0, 0.0),
        (0.0, 1.0, 0.0, -1.0),
    ),
    dtype=float,
)


class BranchBProvenanceError(ValueError):
    """Raised when the design payload no longer matches its bound sources."""


class BranchBPrediction(NamedTuple):
    """Predicted observations in their unflattened physical block shapes."""

    edge_expected_counts: jnp.ndarray
    tomography_observables: jnp.ndarray
    deflector_response_per_rad: jnp.ndarray


@dataclass(frozen=True)
class BranchBLocalRankDiagnostics:
    """Target information after profiling every nonlinear nuisance column."""

    target_rank: int
    nuisance_rank: int
    target_singular_values: np.ndarray
    scaled_parameter_standard_uncertainties: np.ndarray

    def __post_init__(self) -> None:
        singular = _immutable_array(
            self.target_singular_values,
            "target_singular_values",
            shape=(12,),
        )
        uncertainty = _immutable_array(
            self.scaled_parameter_standard_uncertainties,
            "scaled_parameter_standard_uncertainties",
            shape=(12,),
        )
        if not 0 <= self.target_rank <= 12 or self.nuisance_rank < 0:
            raise ValueError("Rank values are outside their valid ranges.")
        object.__setattr__(self, "target_singular_values", singular)
        object.__setattr__(
            self,
            "scaled_parameter_standard_uncertainties",
            uncertainty,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _actual_geometry(document: Mapping[str, Any]) -> dict[str, Any]:
    aperture = document["apertures"]["C_aperture"]
    deflector = document["deflectors"]["CL12_deflector"]
    return {
        "CL12_deflector": {
            key: deflector[key]
            for key in (
                "Dc",
                "shift_balance_x",
                "shift_balance_y",
                "spacing_m",
                "tilt_balance_x",
                "tilt_balance_y",
                "turns",
                "z_m",
            )
        },
        "C_aperture": {
            "radii_m": aperture["radii_m"],
            "z_m": aperture["z_m"],
        },
        "C_mini": document["lenses"]["C_mini"]["z_m"],
        "sample": document["sample"]["z_m"],
    }


def _immutable_array(
    value: Any,
    name: str,
    *,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    array = array.copy()
    array.setflags(write=False)
    return array


def _lower_cholesky(value: Any, name: str, size: int) -> np.ndarray:
    factor = _immutable_array(value, name, shape=(size, size))
    scale = max(float(np.max(np.abs(factor), initial=0.0)), 1.0)
    if np.max(np.abs(np.triu(factor, 1)), initial=0.0) > 1.0e-12 * scale:
        raise ValueError(f"{name} must be lower triangular.")
    if np.any(np.diag(factor) <= 0.0):
        raise ValueError(f"{name} must have a strictly positive diagonal.")
    return factor


def _unique_names(values: Sequence[str], name: str) -> tuple[str, ...]:
    names = tuple(str(value).strip() for value in values)
    if not names or any(not value for value in names):
        raise ValueError(f"{name} must contain non-empty names.")
    if len(names) != len(set(names)):
        raise ValueError(f"{name} must contain unique names.")
    return names


@dataclass(frozen=True)
class BranchBParameterLayout:
    """Exact ordering of the 56 nonlinear Branch-B fit parameters."""

    target_names: tuple[str, ...]
    upstream_names: tuple[str, ...]
    n_edge_states: int = 4
    names: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        target = _unique_names(self.target_names, "target_names")
        upstream = _unique_names(self.upstream_names, "upstream_names")
        if len(target) != 12 or len(upstream) != 4 or self.n_edge_states != 4:
            raise ValueError(
                "Branch B requires 12 targets, 4 upstream nuisances, and "
                "4 edge states."
            )
        deflector_names = tuple(
            f"nuisance.K16.{plane_axis}.command_{command}"
            for plane_axis in (
                "upper_kick_x",
                "upper_kick_y",
                "lower_kick_x",
                "lower_kick_y",
            )
            for command in range(4)
        )
        local_names = tuple(
            f"nuisance.edge_state_{state:02d}.{suffix}"
            for state in range(self.n_edge_states)
            for suffix in ("centre_x_um", "centre_y_um", "log_dose")
        )
        names = (
            tuple(f"target.scaled_delta.{value}" for value in target)
            + tuple(
                f"nuisance.entrance_covariance.{value}"
                for value in ENTRANCE_COVARIANCE_COORDINATE_NAMES
            )
            + tuple(f"nuisance.scaled_delta.{value}" for value in upstream)
            + deflector_names
            + (
                "nuisance.shared_log_hole_radius",
                "nuisance.shared_log_stage_xy_scale",
            )
            + local_names
        )
        object.__setattr__(self, "target_names", target)
        object.__setattr__(self, "upstream_names", upstream)
        object.__setattr__(self, "names", _unique_names(names, "parameter names"))

    @property
    def size(self) -> int:
        return len(self.names)

    @property
    def slices(self) -> Mapping[str, slice]:
        return MappingProxyType(
            {
                "target": slice(0, 12),
                "entrance_covariance": slice(12, 22),
                "source_and_obj_prefield": slice(22, 26),
                "K16": slice(26, 42),
                "shared_log_hole_radius": slice(42, 43),
                "shared_log_stage_xy_scale": slice(43, 44),
                "edge_local": slice(44, 56),
            }
        )


@dataclass(frozen=True)
class BranchBResidualLayout:
    """Exact ordering of the 458 likelihood and prior residual rows."""

    edge_scan_normal_angles_deg: tuple[float, ...]
    edge_scan_offsets_sigma: tuple[float, ...]
    names: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        angles = tuple(float(value) for value in self.edge_scan_normal_angles_deg)
        offsets = tuple(float(value) for value in self.edge_scan_offsets_sigma)
        if len(angles) != 3 or len(offsets) != 7:
            raise ValueError("Branch B requires three edge normals and seven offsets.")
        edge_names: list[str] = []
        for state in range(4):
            for angle in angles:
                for offset in offsets:
                    edge_names.append(
                        f"edge.state_{state:02d}.normal_{angle:g}_deg."
                        f"offset_{offset:+g}_sigma.poisson_count"
                    )
            edge_names.append(f"edge.state_{state:02d}.aperture_out.poisson_count")
        tomography_names = tuple(
            f"tomography.state_{state:02d}.{observable}"
            for state in range(32)
            for observable in PHASE_SPACE_COVARIANCE_OBSERVABLE_NAMES
        )
        deflector_names = tuple(
            f"deflector.state_{state:02d}.{observable}."
            f"response_per_rad.command_{command}"
            for state in range(3)
            for observable in _DEFLECTOR_STATE_NAMES
            for command in range(4)
        )
        names = (
            tuple(edge_names)
            + tomography_names
            + deflector_names
            + (
                "prior.shared_log_hole_radius",
                "prior.shared_log_stage_xy_scale",
            )
        )
        object.__setattr__(self, "edge_scan_normal_angles_deg", angles)
        object.__setattr__(self, "edge_scan_offsets_sigma", offsets)
        object.__setattr__(self, "names", _unique_names(names, "residual names"))

    @property
    def size(self) -> int:
        return len(self.names)

    @property
    def slices(self) -> Mapping[str, slice]:
        return MappingProxyType(
            {
                "edge_poisson": slice(0, 88),
                "tomography_whitened": slice(88, 408),
                "deflector_whitened": slice(408, 456),
                "shared_priors": slice(456, 458),
            }
        )


@dataclass(frozen=True)
class BranchBExperiment:
    """Frozen payload, optical model, states, and nonlinear ordering contract."""

    column: HardEdgeColumnParameterization
    payload_path: Path
    payload_sha256: str
    coordinate_status: str
    edge_controls: np.ndarray
    tomography_controls: np.ndarray
    deflector_controls: np.ndarray
    edge_commands_xy_m: np.ndarray
    entrance_reference_covariance: np.ndarray
    target_parameter_indices: np.ndarray
    target_parameter_scales: np.ndarray
    upstream_parameter_indices: np.ndarray
    upstream_parameter_scales: np.ndarray
    parameter_layout: BranchBParameterLayout
    residual_layout: BranchBResidualLayout
    aperture_z_m: float
    aperture_radius_m: float
    upper_deflector_z_m: float
    lower_deflector_z_m: float
    sample_z_m: float
    edge_incident_electrons_per_frame: float
    tomography_summary_sigma: float
    deflector_drive_rad: float
    deflector_position_sigma_m: float
    deflector_tilt_sigma_rad: float
    aperture_log_radius_prior_sigma: float
    aperture_stage_log_scale_prior_sigma: float
    edge_radial_order: int
    edge_azimuthal_order: int
    entrance_covariance_chart: EntranceCovarianceChart = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.column, HardEdgeColumnParameterization):
            raise TypeError("column must be a HardEdgeColumnParameterization.")
        if self.coordinate_status != NORMALIZED_CONTROL_WARNING:
            raise BranchBProvenanceError("The normalized-control warning was lost.")
        channels = len(self.column.channel_names)
        arrays = {
            "edge_controls": _immutable_array(
                self.edge_controls, "edge_controls", shape=(4, channels)
            ),
            "tomography_controls": _immutable_array(
                self.tomography_controls,
                "tomography_controls",
                shape=(32, channels),
            ),
            "deflector_controls": _immutable_array(
                self.deflector_controls,
                "deflector_controls",
                shape=(3, channels),
            ),
            "edge_commands_xy_m": _immutable_array(
                self.edge_commands_xy_m,
                "edge_commands_xy_m",
                shape=(4, 21, 2),
            ),
            "entrance_reference_covariance": _immutable_array(
                self.entrance_reference_covariance,
                "entrance_reference_covariance",
                shape=(4, 4),
            ),
            "target_parameter_indices": _immutable_array(
                self.target_parameter_indices,
                "target_parameter_indices",
                shape=(12,),
            ).astype(int),
            "target_parameter_scales": _immutable_array(
                self.target_parameter_scales,
                "target_parameter_scales",
                shape=(12,),
            ),
            "upstream_parameter_indices": _immutable_array(
                self.upstream_parameter_indices,
                "upstream_parameter_indices",
                shape=(4,),
            ).astype(int),
            "upstream_parameter_scales": _immutable_array(
                self.upstream_parameter_scales,
                "upstream_parameter_scales",
                shape=(4,),
            ),
        }
        for name, value in arrays.items():
            value = np.asarray(value).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "entrance_covariance_chart",
            EntranceCovarianceChart(arrays["entrance_reference_covariance"]),
        )
        if self.parameter_layout.size != 56 or self.residual_layout.size != 458:
            raise ValueError("Unexpected Branch-B parameter or residual layout size.")
        positive = (
            self.aperture_radius_m,
            self.edge_incident_electrons_per_frame,
            self.tomography_summary_sigma,
            self.deflector_drive_rad,
            self.deflector_position_sigma_m,
            self.deflector_tilt_sigma_rad,
            self.aperture_log_radius_prior_sigma,
            self.aperture_stage_log_scale_prior_sigma,
        )
        if not np.all(np.isfinite(positive)) or min(positive) <= 0.0:
            raise ValueError("Branch-B physical scales must be finite and positive.")
        if self.edge_radial_order < 8 or self.edge_azimuthal_order < 16:
            raise ValueError("The edge quadrature orders are too small.")

    @property
    def nominal_parameters(self) -> np.ndarray:
        values = np.zeros(self.parameter_layout.size, dtype=float)
        values[self.parameter_layout.slices["K16"]] = _DEFLECTOR_NOMINAL_MAP.reshape(-1)
        values.setflags(write=False)
        return values

    @property
    def raw_frame_count(self) -> int:
        """Number of detector frames represented by the three data blocks."""

        return 4 * 22 + 32 * 5 + 3 * 16

    @property
    def deflector_noise_cholesky(self) -> np.ndarray:
        per_observable = np.repeat(
            np.asarray(
                (
                    self.deflector_position_sigma_m,
                    self.deflector_tilt_sigma_rad,
                    self.deflector_position_sigma_m,
                    self.deflector_tilt_sigma_rad,
                )
            )
            / np.sqrt(2.0),
            4,
        ) / self.deflector_drive_rad
        result = np.diag(np.tile(per_observable, 3))
        result.setflags(write=False)
        return result

    @property
    def tomography_noise_cholesky(self) -> np.ndarray:
        result = np.eye(320) * self.tomography_summary_sigma
        result.setflags(write=False)
        return result

    def _decode_parameters(
        self, parameters: Any
    ) -> tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ]:
        values = jnp.asarray(parameters)
        if values.shape != (self.parameter_layout.size,):
            raise ValueError(
                f"parameters must have shape ({self.parameter_layout.size},)."
            )
        slices = self.parameter_layout.slices
        physical = jnp.asarray(self.column.nominal_parameters, dtype=values.dtype)
        physical = physical.at[jnp.asarray(self.target_parameter_indices)].add(
            values[slices["target"]]
            * jnp.asarray(self.target_parameter_scales, dtype=values.dtype)
        )
        physical = physical.at[jnp.asarray(self.upstream_parameter_indices)].add(
            values[slices["source_and_obj_prefield"]]
            * jnp.asarray(self.upstream_parameter_scales, dtype=values.dtype)
        )
        entrance_coordinates = values[slices["entrance_covariance"]]
        command_map = values[slices["K16"]].reshape((4, 4))
        log_radius = values[slices["shared_log_hole_radius"]][0]
        log_stage_scale = values[slices["shared_log_stage_xy_scale"]][0]
        edge_local = values[slices["edge_local"]].reshape((4, 3))
        return (
            physical,
            entrance_coordinates,
            command_map,
            log_radius,
            log_stage_scale,
            edge_local,
        )

    def predict(self, parameters: Any) -> BranchBPrediction:
        """Predict the three Branch-B observable blocks with JAX operations."""

        (
            physical,
            entrance_coordinates,
            command_map,
            log_radius,
            log_stage_scale,
            edge_local,
        ) = self._decode_parameters(parameters)
        covariance_chart = self.entrance_covariance_chart
        entrance_covariance = covariance_chart.covariance(entrance_coordinates)
        xy = jnp.asarray((0, 2))

        def edge_covariance(dacs: jnp.ndarray) -> jnp.ndarray:
            matrix = hard_edge_matrix_to_plane(
                self.column, physical, dacs, self.aperture_z_m
            )
            propagated = matrix @ entrance_covariance @ matrix.T
            return propagated[jnp.ix_(xy, xy)]

        edge_covariances = jax.vmap(edge_covariance)(
            jnp.asarray(self.edge_controls)
        )
        prediction = aperture_edge_scan_from_moments(
            jnp.zeros((4, 2), dtype=physical.dtype),
            edge_covariances,
            aperture_centre_xy_m=edge_local[:, :2] * 1.0e-6,
            aperture_radius_m=self.aperture_radius_m * jnp.exp(log_radius),
            incident_dose_electrons=(
                self.edge_incident_electrons_per_frame
                * jnp.exp(edge_local[:, 2])
            ),
            aperture_centre_offsets_xy_m=(
                jnp.exp(log_stage_scale) * jnp.asarray(self.edge_commands_xy_m)
            ),
            radial_order=self.edge_radial_order,
            azimuthal_order=self.edge_azimuthal_order,
        )
        edge_counts = jnp.concatenate(
            (
                prediction.expected_transmitted_electrons,
                prediction.aperture_out_expected_electrons[:, None],
            ),
            axis=1,
        )
        tomography = (
            axial_plane_covariance_observables_with_entrance_covariance(
                self.column,
                physical,
                entrance_coordinates,
                jnp.asarray(self.tomography_controls),
                covariance_chart,
                plane_z_m=self.sample_z_m,
                mode="full",
            )
        )

        def one_deflector_state(dacs: jnp.ndarray) -> jnp.ndarray:
            response = ideal_angular_kick_response_matrices(
                self.column,
                physical,
                dacs,
                (self.upper_deflector_z_m, self.lower_deflector_z_m),
                observation_z_m=self.sample_z_m,
            )
            transport = jnp.concatenate((response[0], response[1]), axis=1)
            return transport @ command_map

        deflector = jax.vmap(one_deflector_state)(
            jnp.asarray(self.deflector_controls)
        ).reshape((3, 16))
        return BranchBPrediction(edge_counts, tomography, deflector)

    def predict_deflector_signed_displacements(
        self, parameters: Any
    ) -> jnp.ndarray:
        """Return ideal raw-command ± displacements before modality extraction.

        The shape is ``(3 optical states, 4 commands, 2 signs, 4 phase-space
        components)`` and the sign axis is ``(-, +)``.  Each sign requires an
        image and diffraction exposure to recover all four components, hence
        ``3 * 4 * 2 * 2 = 48`` raw frames.  Taking ``(plus-minus)/2`` and
        dividing by :attr:`deflector_drive_rad` exactly recovers
        :attr:`BranchBPrediction.deflector_response_per_rad`.
        """

        response = self.predict(parameters).deflector_response_per_rad.reshape(
            (3, 4, 4)
        )
        command_first = jnp.swapaxes(response, 1, 2)
        driven = self.deflector_drive_rad * command_first
        return jnp.stack((-driven, driven), axis=2)


def load_branch_b_experiment(
    payload_path: str | Path = DEFAULT_BRANCH_B_PAYLOAD,
    *,
    repository_root: str | Path | None = None,
    design_name: str = "design_296",
) -> BranchBExperiment:
    """Load and hash-verify the actual-geometry 296-frame Branch-B design."""

    path = Path(payload_path).resolve()
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    payload_digest = _sha256(path)
    if payload_digest != EXPECTED_BRANCH_B_PAYLOAD_SHA256:
        raise BranchBProvenanceError(
            "The Branch-B payload differs from the checksum committed in "
            "the likelihood implementation."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != BRANCH_B_SCHEMA:
        raise BranchBProvenanceError("Unsupported Branch-B payload schema.")
    coordinate_status = str(payload.get("coordinate_status", ""))
    if coordinate_status != NORMALIZED_CONTROL_WARNING:
        raise BranchBProvenanceError("The normalized-control warning is missing.")
    if design_name != "design_296":
        raise ValueError("The raw actual-geometry likelihood requires design_296.")
    design = payload["designs"][design_name]
    if design["state_counts"] != _EXPECTED_STATE_COUNTS:
        raise BranchBProvenanceError("The 296-frame state contract changed.")
    if payload["measurement_contract"]["raw_frames_per_state"] != _EXPECTED_RAW_FRAMES:
        raise BranchBProvenanceError("The raw frame accounting changed.")
    expected_frame_count = sum(
        _EXPECTED_STATE_COUNTS[kind] * _EXPECTED_RAW_FRAMES[kind]
        for kind in ("edge", "tomography", "deflector")
    )
    if int(design["raw_frame_count"]) != expected_frame_count:
        raise BranchBProvenanceError("The selected raw frame count changed.")
    if payload["hardware_admission"]["ready"] is not False:
        raise BranchBProvenanceError(
            "A local Fisher design must not claim hardware admission."
        )

    provenance = payload["provenance"]
    config_path = root / provenance["config"]["path"]
    if _sha256(config_path) != provenance["config"]["sha256"]:
        raise BranchBProvenanceError("The bound inversion config hash changed.")
    geometry_binding = provenance["actual_geometry"]
    microscope_path = root / geometry_binding["path"]
    with microscope_path.open("rb") as stream:
        actual_geometry = _actual_geometry(tomllib.load(stream))
    if (
        _canonical_sha256(actual_geometry)
        != geometry_binding["canonical_sha256"]
        or actual_geometry != geometry_binding["values"]
    ):
        raise BranchBProvenanceError("The bound actual microscope geometry changed.")
    for implementation in provenance["implementation_files"]:
        implementation_path = root / implementation["path"]
        if _sha256(implementation_path) != implementation["sha256"]:
            raise BranchBProvenanceError(
                f"The bound implementation changed: {implementation['path']}."
            )

    with config_path.open("rb") as stream:
        column = hard_edge_column_from_config(tomllib.load(stream))
    channel_order = tuple(payload["model_control_contract"]["channel_order"])
    if tuple(column.channel_names) != channel_order:
        raise BranchBProvenanceError("The payload and column channel orders differ.")
    n_channels = len(channel_order)
    edge_controls = _immutable_array(
        design["edge_controls"], "edge_controls", shape=(4, n_channels)
    )
    tomography_controls = _immutable_array(
        design["tomography_controls"],
        "tomography_controls",
        shape=(32, n_channels),
    )
    deflector_controls = _immutable_array(
        design["deflector_controls"],
        "deflector_controls",
        shape=(3, n_channels),
    )
    lower, upper = np.asarray(
        payload["model_control_contract"]["safe_bounds_CL1_CL3_Cmini"],
        dtype=float,
    )
    all_controls = np.concatenate(
        (edge_controls, tomography_controls, deflector_controls), axis=0
    )
    if (
        lower.shape != (3,)
        or upper.shape != (3,)
        or np.any(all_controls[:, :3] < lower[None, :])
        or np.any(all_controls[:, :3] > upper[None, :])
        or not np.allclose(all_controls[:, 3:], 0.0, rtol=0.0, atol=0.0)
    ):
        raise BranchBProvenanceError(
            "The selected normalized controls violate their frozen safe box."
        )
    q_contract = payload["model_control_contract"]["affine_q_per_m"]
    q = np.asarray(q_contract["bias"])[None, :] + np.asarray(
        q_contract["slope"]
    )[None, :] * all_controls[:, :3]
    q_floor = float(
        payload["model_control_contract"]["candidate_q_floor_per_m"]
    )
    if np.any(q < q_floor):
        raise BranchBProvenanceError("The selected controls violate the q floor.")
    if not np.allclose(
        np.min(q, axis=0),
        design["selected_q_min_per_m"],
        rtol=2.0e-14,
        atol=0.0,
    ) or not np.allclose(
        np.max(q, axis=0),
        design["selected_q_max_per_m"],
        rtol=2.0e-14,
        atol=0.0,
    ):
        raise BranchBProvenanceError("The selected q extrema changed.")
    parameter_names = column.parameter_names
    target_indices = np.asarray(
        [
            index
            for index, name in enumerate(parameter_names)
            if name.startswith(("CL1.", "CL3.", "C-mini."))
        ],
        dtype=int,
    )
    upstream_indices = np.asarray(
        [
            index
            for index, name in enumerate(parameter_names)
            if name == "source_z_m" or name.startswith("Obj-prefield.")
        ],
        dtype=int,
    )
    target_names = tuple(parameter_names[index] for index in target_indices)
    upstream_names = tuple(parameter_names[index] for index in upstream_indices)
    if list(target_names) != payload["target_parameter_contract"]["names"]:
        raise BranchBProvenanceError("The condenser target ordering changed.")
    nominal = np.asarray(column.nominal_parameters)
    target_scales = np.asarray(
        [
            1.0
            if parameter_names[index].endswith(".log_length_m")
            else abs(nominal[index])
            for index in target_indices
        ]
    )
    upstream_scales = np.asarray(
        [
            1.0e-3
            if parameter_names[index] == "source_z_m"
            else (
                5.0e-4
                if parameter_names[index].endswith("z_center_m")
                else 0.08
            )
            for index in upstream_indices
        ]
    )
    entrance_covariance = _immutable_array(
        payload["illumination"]["entrance_covariance"],
        "entrance_covariance",
        shape=(4, 4),
    )
    xy = jnp.asarray((0, 2))

    def nominal_edge_covariance(dacs: jnp.ndarray) -> jnp.ndarray:
        matrix = hard_edge_matrix_to_plane(
            column,
            jnp.asarray(column.nominal_parameters),
            dacs,
            float(actual_geometry["C_aperture"]["z_m"]),
        )
        propagated = matrix @ jnp.asarray(entrance_covariance) @ matrix.T
        return propagated[jnp.ix_(xy, xy)]

    edge_covariances = np.asarray(
        jax.vmap(nominal_edge_covariance)(jnp.asarray(edge_controls))
    )
    edge_contract = payload["measurement_contract"]["edge_scan"]
    angles = np.deg2rad(edge_contract["normal_angles_deg"])
    normals = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    offsets = np.asarray(edge_contract["offsets_in_local_sigma"])
    sigma_normal = np.sqrt(
        np.einsum("ai,nij,aj->na", normals, edge_covariances, normals)
    )
    radius = float(edge_contract["hole_radius_m"])
    edge_commands = (
        normals[None, :, None, :]
        * (
            radius
            + sigma_normal[:, :, None, None]
            * offsets[None, None, :, None]
        )
    ).reshape((4, 21, 2))
    noise = payload["noise_and_priors"]
    deflector_geometry = actual_geometry["CL12_deflector"]
    return BranchBExperiment(
        column=column,
        payload_path=path,
        payload_sha256=payload_digest,
        coordinate_status=coordinate_status,
        edge_controls=edge_controls,
        tomography_controls=tomography_controls,
        deflector_controls=deflector_controls,
        edge_commands_xy_m=edge_commands,
        entrance_reference_covariance=entrance_covariance,
        target_parameter_indices=target_indices,
        target_parameter_scales=target_scales,
        upstream_parameter_indices=upstream_indices,
        upstream_parameter_scales=upstream_scales,
        parameter_layout=BranchBParameterLayout(target_names, upstream_names),
        residual_layout=BranchBResidualLayout(
            tuple(edge_contract["normal_angles_deg"]),
            tuple(edge_contract["offsets_in_local_sigma"]),
        ),
        aperture_z_m=float(actual_geometry["C_aperture"]["z_m"]),
        aperture_radius_m=radius,
        upper_deflector_z_m=float(deflector_geometry["z_m"]),
        lower_deflector_z_m=(
            float(deflector_geometry["z_m"])
            + float(deflector_geometry["spacing_m"])
        ),
        sample_z_m=float(actual_geometry["sample"]),
        edge_incident_electrons_per_frame=float(
            noise["edge_incident_electrons_per_frame"]
        ),
        tomography_summary_sigma=float(noise["tomography_summary_sigma"]),
        deflector_drive_rad=float(noise["deflector_drive_rad"]),
        deflector_position_sigma_m=float(
            noise["single_frame_position_sigma_m"]
        ),
        deflector_tilt_sigma_rad=float(noise["single_frame_tilt_sigma_rad"]),
        aperture_log_radius_prior_sigma=float(
            noise["aperture_log_radius_prior_sigma"]
        ),
        aperture_stage_log_scale_prior_sigma=float(
            noise["aperture_stage_log_scale_prior_sigma"]
        ),
        edge_radial_order=int(edge_contract["radial_quadrature_order"]),
        edge_azimuthal_order=int(edge_contract["azimuthal_quadrature_order"]),
    )


ObservationKind = Literal[
    "instrument",
    "synthetic_expectation",
    "synthetic_draw",
]


@dataclass(frozen=True)
class BranchBObservations:
    """Measured Branch-B values and their declared statistical factors.

    Instrument ``edge_counts`` must be non-negative integer electron counts.
    Tomography and deflector factors are arbitrary dense lower Cholesky factors
    in the exact flattened residual order; diagonal-noise assumptions are not
    hidden in the likelihood.
    """

    edge_counts: np.ndarray
    tomography_observables: np.ndarray
    tomography_noise_cholesky: np.ndarray
    deflector_response_per_rad: np.ndarray
    deflector_noise_cholesky: np.ndarray
    design_payload_sha256: str
    coordinate_status: str
    provenance: str
    kind: ObservationKind = "instrument"

    def __post_init__(self) -> None:
        edge = _immutable_array(
            self.edge_counts, "edge_counts", shape=(4, 22)
        )
        if np.any(edge < 0.0):
            raise ValueError("edge_counts must be non-negative.")
        if self.kind == "instrument" and not np.allclose(
            edge, np.rint(edge), rtol=0.0, atol=0.0
        ):
            raise ValueError("Instrument Poisson edge counts must be integers.")
        if self.kind not in {
            "instrument",
            "synthetic_expectation",
            "synthetic_draw",
        }:
            raise ValueError("Unsupported observation kind.")
        tomography = _immutable_array(
            self.tomography_observables,
            "tomography_observables",
            shape=(32, 10),
        )
        deflector = _immutable_array(
            self.deflector_response_per_rad,
            "deflector_response_per_rad",
            shape=(3, 16),
        )
        tomography_cholesky = _lower_cholesky(
            self.tomography_noise_cholesky,
            "tomography_noise_cholesky",
            320,
        )
        deflector_cholesky = _lower_cholesky(
            self.deflector_noise_cholesky,
            "deflector_noise_cholesky",
            48,
        )
        digest = str(self.design_payload_sha256).strip()
        if len(digest) != 64 or any(
            value not in "0123456789abcdef" for value in digest
        ):
            raise ValueError(
                "design_payload_sha256 must be a lowercase SHA-256 digest."
            )
        if self.coordinate_status != NORMALIZED_CONTROL_WARNING:
            raise BranchBProvenanceError("The normalized-control warning was lost.")
        provenance = str(self.provenance).strip()
        if not provenance:
            raise ValueError("provenance must be non-empty.")
        object.__setattr__(self, "edge_counts", edge)
        object.__setattr__(self, "tomography_observables", tomography)
        object.__setattr__(self, "tomography_noise_cholesky", tomography_cholesky)
        object.__setattr__(self, "deflector_response_per_rad", deflector)
        object.__setattr__(self, "deflector_noise_cholesky", deflector_cholesky)
        object.__setattr__(self, "design_payload_sha256", digest)
        object.__setattr__(self, "provenance", provenance)


def signed_poisson_deviance_residual(
    expected_counts: Any,
    observed_counts: Any,
) -> jnp.ndarray:
    r"""Return a stable signed square-root Poisson deviance residual.

    With sign ``expected - observed``, one half of the squared residual is the
    Poisson log-likelihood ratio to the saturated model.  The positive-count
    branch is evaluated as an analytic function of
    ``delta = (expected-observed)/observed``.  A local series removes the
    removable zero-over-zero singularity at a matched prediction, so
    ``jax.jacfwd`` returns the correct Fisher derivative ``1/sqrt(observed)``.
    """

    expected = jnp.asarray(expected_counts)
    observed = jnp.asarray(observed_counts)
    if expected.shape != observed.shape:
        raise ValueError("expected_counts and observed_counts must have equal shape.")
    dtype = jnp.result_type(expected, observed, jnp.asarray(1.0))
    expected = expected.astype(dtype)
    observed = observed.astype(dtype)
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    expected = jnp.maximum(expected, tiny)
    positive = observed > 0.0
    safe_observed = jnp.where(positive, observed, jnp.ones_like(observed))
    delta = (expected - safe_observed) / safe_observed
    small = jnp.abs(delta) < 1.0e-4
    safe_delta = jnp.where(small, jnp.asarray(0.5, dtype=dtype), delta)
    exact_factor_squared = 2.0 * (
        safe_delta - jnp.log1p(safe_delta)
    ) / (safe_delta * safe_delta)
    series_factor_squared = (
        1.0
        - (2.0 / 3.0) * delta
        + 0.5 * delta**2
        - 0.4 * delta**3
        + (1.0 / 3.0) * delta**4
    )
    factor = jnp.sqrt(
        jnp.maximum(
            jnp.where(small, series_factor_squared, exact_factor_squared),
            tiny,
        )
    )
    positive_residual = delta * jnp.sqrt(safe_observed) * factor
    zero_residual = jnp.sqrt(2.0 * expected)
    return jnp.where(positive, positive_residual, zero_residual)


@dataclass(frozen=True)
class BranchBLikelihood:
    """JAX residual and exact likelihood for one immutable observation set."""

    experiment: BranchBExperiment
    observations: BranchBObservations

    def __post_init__(self) -> None:
        if not isinstance(self.experiment, BranchBExperiment):
            raise TypeError("experiment must be a BranchBExperiment.")
        if not isinstance(self.observations, BranchBObservations):
            raise TypeError("observations must be BranchBObservations.")
        if self.observations.design_payload_sha256 != self.experiment.payload_sha256:
            raise BranchBProvenanceError(
                "Observations were not registered to this design payload."
            )
        if self.observations.coordinate_status != self.experiment.coordinate_status:
            raise BranchBProvenanceError("Observation coordinate status differs.")

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self.experiment.parameter_layout.names

    @property
    def parameter_slices(self) -> Mapping[str, slice]:
        return self.experiment.parameter_layout.slices

    @property
    def residual_names(self) -> tuple[str, ...]:
        return self.experiment.residual_layout.names

    @property
    def residual_slices(self) -> Mapping[str, slice]:
        return self.experiment.residual_layout.slices

    @property
    def nominal_parameters(self) -> np.ndarray:
        return self.experiment.nominal_parameters

    def predict(self, parameters: Any) -> BranchBPrediction:
        return self.experiment.predict(parameters)

    def data_residual(self, parameters: Any) -> jnp.ndarray:
        """Return 456 data rows: Poisson edge, tomography, then deflector."""

        prediction = self.predict(parameters)
        edge = signed_poisson_deviance_residual(
            prediction.edge_expected_counts,
            jnp.asarray(self.observations.edge_counts),
        ).reshape(-1)
        tomography = jsp.linalg.solve_triangular(
            jnp.asarray(self.observations.tomography_noise_cholesky),
            (
                prediction.tomography_observables
                - jnp.asarray(self.observations.tomography_observables)
            ).reshape(-1),
            lower=True,
        )
        deflector = jsp.linalg.solve_triangular(
            jnp.asarray(self.observations.deflector_noise_cholesky),
            (
                prediction.deflector_response_per_rad
                - jnp.asarray(self.observations.deflector_response_per_rad)
            ).reshape(-1),
            lower=True,
        )
        return jnp.concatenate((edge, tomography, deflector))

    def residual(self, parameters: Any) -> jnp.ndarray:
        """Return the complete 458-row nonlinear least-squares residual."""

        values = jnp.asarray(parameters)
        if values.shape != (len(self.parameter_names),):
            raise ValueError(
                f"parameters must have shape ({len(self.parameter_names)},)."
            )
        slices = self.parameter_slices
        prior = jnp.stack(
            (
                values[slices["shared_log_hole_radius"]][0]
                / self.experiment.aperture_log_radius_prior_sigma,
                values[slices["shared_log_stage_xy_scale"]][0]
                / self.experiment.aperture_stage_log_scale_prior_sigma,
            )
        )
        return jnp.concatenate((self.data_residual(values), prior))

    def objective(self, parameters: Any) -> jnp.ndarray:
        """Poisson deviance/2 plus Gaussian and declared-prior quadratics."""

        residual = self.residual(parameters)
        return 0.5 * jnp.dot(residual, residual)

    def negative_log_likelihood(self, parameters: Any) -> jnp.ndarray:
        """Return the fully normalized Poisson-plus-Gaussian negative log PDF."""

        prediction = self.predict(parameters)
        observed_edge = jnp.asarray(self.observations.edge_counts)
        tiny = jnp.asarray(
            jnp.finfo(prediction.edge_expected_counts.dtype).tiny,
            dtype=prediction.edge_expected_counts.dtype,
        )
        expected_edge = jnp.maximum(prediction.edge_expected_counts, tiny)
        poisson = jnp.sum(
            expected_edge
            - observed_edge * jnp.log(expected_edge)
            + jsp.special.gammaln(observed_edge + 1.0)
        )
        data = self.data_residual(parameters)
        gaussian = 0.5 * (
            jnp.dot(data[88:], data[88:])
            + 2.0
            * (
                jnp.sum(
                    jnp.log(
                        jnp.diag(
                            jnp.asarray(
                                self.observations.tomography_noise_cholesky
                            )
                        )
                    )
                )
                + jnp.sum(
                    jnp.log(
                        jnp.diag(
                            jnp.asarray(
                                self.observations.deflector_noise_cholesky
                            )
                        )
                    )
                )
            )
            + 368.0 * jnp.log(2.0 * jnp.pi)
        )
        slices = self.parameter_slices
        priors = 0.5 * (
            (
                jnp.asarray(parameters)[slices["shared_log_hole_radius"]][0]
                / self.experiment.aperture_log_radius_prior_sigma
            )
            ** 2
            + (
                jnp.asarray(parameters)[slices["shared_log_stage_xy_scale"]][0]
                / self.experiment.aperture_stage_log_scale_prior_sigma
            )
            ** 2
        )
        priors = priors + jnp.log(
            2.0
            * jnp.pi
            * self.experiment.aperture_log_radius_prior_sigma
            * self.experiment.aperture_stage_log_scale_prior_sigma
        )
        return poisson + gaussian + priors

    def jacobian(self, parameters: Any) -> jnp.ndarray:
        """Return the forward-mode JAX Jacobian used by rank/global-fit code."""

        return jax.jacfwd(self.residual)(jnp.asarray(parameters))

    def local_profiled_target_diagnostics(
        self,
        parameters: Any,
        *,
        jacobian: Any | None = None,
        nuisance_relative_rtol: float = 1.0e-10,
        target_relative_rtol: float = 1.0e-7,
    ) -> BranchBLocalRankDiagnostics:
        """SVD-profile all 44 nuisances and diagnose the 12 target directions."""

        if not 0.0 < nuisance_relative_rtol < 1.0:
            raise ValueError("nuisance_relative_rtol must lie in (0, 1).")
        if not 0.0 < target_relative_rtol < 1.0:
            raise ValueError("target_relative_rtol must lie in (0, 1).")
        matrix = np.asarray(
            self.jacobian(parameters) if jacobian is None else jacobian,
            dtype=float,
        )
        expected_shape = (len(self.residual_names), len(self.parameter_names))
        if matrix.shape != expected_shape or not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"jacobian must be finite with shape {expected_shape}."
            )
        target = matrix[:, self.parameter_slices["target"]]
        nuisance = matrix[:, 12:]
        norms = np.linalg.norm(nuisance, axis=0)
        active = norms > np.finfo(float).tiny
        normalized = nuisance[:, active] / norms[active][None, :]
        left, nuisance_singular, _ = np.linalg.svd(
            normalized, full_matrices=False
        )
        nuisance_rank = (
            int(
                np.sum(
                    nuisance_singular
                    > nuisance_relative_rtol * nuisance_singular[0]
                )
            )
            if len(nuisance_singular)
            else 0
        )
        projected = target - left[:, :nuisance_rank] @ (
            left[:, :nuisance_rank].T @ target
        )
        singular = np.linalg.svd(projected, compute_uv=False)
        target_rank = int(
            np.sum(singular > target_relative_rtol * singular[0])
        )
        uncertainty = (
            np.full(12, np.inf)
            if target_rank < 12
            else np.sqrt(
                np.maximum(
                    np.diag(np.linalg.inv(projected.T @ projected)),
                    0.0,
                )
            )
        )
        return BranchBLocalRankDiagnostics(
            target_rank=target_rank,
            nuisance_rank=nuisance_rank,
            target_singular_values=singular,
            scaled_parameter_standard_uncertainties=uncertainty,
        )

    def numpy_residual(self, parameters: np.ndarray) -> np.ndarray:
        """SciPy-compatible host wrapper around :meth:`residual`."""

        return np.asarray(self.residual(jnp.asarray(parameters)), dtype=float)


def make_synthetic_branch_b_observations(
    experiment: BranchBExperiment,
    parameters: Any | None = None,
    *,
    sample_noise: bool = False,
    seed: int = 260904,
    tomography_noise_cholesky: np.ndarray | None = None,
    deflector_noise_cholesky: np.ndarray | None = None,
) -> BranchBObservations:
    """Explicitly construct matched synthetic data; never labels it instrument data.

    ``sample_noise=False`` stores the exact model expectation and therefore has
    zero residual at ``parameters``.  ``sample_noise=True`` draws Poisson edge
    counts and correlated Gaussian tomography/deflector errors using ``seed``.
    """

    if not isinstance(experiment, BranchBExperiment):
        raise TypeError("experiment must be a BranchBExperiment.")
    truth_parameters = (
        experiment.nominal_parameters
        if parameters is None
        else np.asarray(parameters, dtype=float)
    )
    if truth_parameters.shape != (experiment.parameter_layout.size,):
        raise ValueError("parameters have the wrong shape.")
    prediction = experiment.predict(jnp.asarray(truth_parameters))
    edge = np.asarray(prediction.edge_expected_counts)
    tomography = np.asarray(prediction.tomography_observables)
    deflector = np.asarray(prediction.deflector_response_per_rad)
    tomography_cholesky = (
        experiment.tomography_noise_cholesky
        if tomography_noise_cholesky is None
        else tomography_noise_cholesky
    )
    deflector_cholesky = (
        experiment.deflector_noise_cholesky
        if deflector_noise_cholesky is None
        else deflector_noise_cholesky
    )
    if sample_noise:
        rng = np.random.default_rng(int(seed))
        edge = rng.poisson(edge).astype(float)
        tomography_noise = (
            np.asarray(tomography_cholesky) @ rng.normal(size=320)
        ).reshape((32, 10))
        tomography = tomography + tomography_noise
        deflector_noise = (
            np.asarray(deflector_cholesky) @ rng.normal(size=48)
        ).reshape((3, 16))
        deflector = deflector + deflector_noise
        kind: ObservationKind = "synthetic_draw"
    else:
        kind = "synthetic_expectation"
    return BranchBObservations(
        edge_counts=edge,
        tomography_observables=tomography,
        tomography_noise_cholesky=tomography_cholesky,
        deflector_response_per_rad=deflector,
        deflector_noise_cholesky=deflector_cholesky,
        design_payload_sha256=experiment.payload_sha256,
        coordinate_status=experiment.coordinate_status,
        provenance=(
            "Matched synthetic Branch-B expectation generated by TemGymCore."
            if not sample_noise
            else f"Synthetic Branch-B Poisson/Gaussian draw; RNG seed={int(seed)}."
        ),
        kind=kind,
    )


__all__ = [
    "BRANCH_B_SCHEMA",
    "DEFAULT_BRANCH_B_PAYLOAD",
    "EXPECTED_BRANCH_B_PAYLOAD_SHA256",
    "NORMALIZED_CONTROL_WARNING",
    "BranchBExperiment",
    "BranchBLikelihood",
    "BranchBLocalRankDiagnostics",
    "BranchBObservations",
    "BranchBParameterLayout",
    "BranchBPrediction",
    "BranchBProvenanceError",
    "BranchBResidualLayout",
    "load_branch_b_experiment",
    "make_synthetic_branch_b_observations",
    "signed_poisson_deviance_residual",
]
