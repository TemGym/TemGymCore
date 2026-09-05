"""Nonlinear no-commanded-tilt likelihood for the astigmatic condenser design.

This module turns the checksum-bound local design audit into an executable
Poisson-plus-Gaussian likelihood.  Its 43 parameters and 428 residuals have
exactly the ordering declared by :mod:`condenser_astigmatic_design`.  It uses
translated C-aperture counts, sample-plane spatial covariance, indexed-Si
angular covariance, and a relative deliberately astigmatic orientation tag.
There are no calibrated deflector or commanded-tilt observations.

The tag model is valid only when the tagged beam is created upstream of CL1.
The committed manifest has not verified that hardware condition, nonlinear
truth-envelope recovery, or held-out performance, so this likelihood does not
grant hardware admission.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from .condenser_astigmatic_design import (
    AstigmaticCondenserDesign,
    AstigmaticDesignProvenanceError,
    DEFAULT_ASTIGMATIC_DESIGN,
    build_astigmatic_design_matrix,
    load_astigmatic_condenser_design,
)
from .condenser_branch_b import (
    NORMALIZED_CONTROL_WARNING,
    BranchBExperiment,
    load_branch_b_experiment,
    signed_poisson_deviance_residual,
)
from .critical_line_calibration import (
    aperture_edge_scan_from_moments,
    axial_plane_covariance_observables_with_entrance_covariance,
    hard_edge_matrix_to_plane,
)


_PARAMETER_COUNT = 43
_RESIDUAL_COUNT = 428
_EDGE_SHAPE = (4, 22)
_SAMPLE_SHAPE = (48, 7)
_SAMPLE_SIZE = 48 * 7
_SPATIAL_OBSERVABLE_INDICES = (0, 2, 5)
_ANGULAR_OBSERVABLE_INDICES = (1, 3, 8)
_BRANCH_PARAMETER_INDICES = (*range(26), 42, 43, *range(44, 56))

ObservationKind = Literal[
    "instrument",
    "synthetic_expectation",
    "synthetic_draw",
]


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
    result = array.copy()
    result.setflags(write=False)
    return result


def _lower_cholesky(value: Any, name: str, size: int) -> np.ndarray:
    factor = _immutable_array(value, name, shape=(size, size))
    scale = max(float(np.max(np.abs(factor), initial=0.0)), 1.0)
    if np.max(np.abs(np.triu(factor, 1)), initial=0.0) > 1.0e-12 * scale:
        raise ValueError(f"{name} must be lower triangular.")
    if np.any(np.diag(factor) <= 0.0):
        raise ValueError(f"{name} must have a strictly positive diagonal.")
    return factor


def _verify_tag_plane_contract(design: AstigmaticCondenserDesign) -> None:
    import json

    document = json.loads(design.path.read_text(encoding="utf-8"))
    contract = document["sample_block"]["astigmatic_tag_contract"]
    gate = design.hardware_gates["astigmatic_tag_plane_and_repeatability"]
    if "upstream of CL1" not in str(contract.get("required_plane", "")):
        raise AstigmaticDesignProvenanceError(
            "The astigmatic tag must be created upstream of CL1."
        )
    if bool(contract.get("absolute_command_calibration_required")):
        raise AstigmaticDesignProvenanceError(
            "The relative astigmatic-tag route must not require command calibration."
        )
    if "upstream of CL1" not in str(gate.get("requirement", "")):
        raise AstigmaticDesignProvenanceError(
            "The hardware gate no longer enforces an upstream-of-CL1 tag."
        )
    if bool(gate.get("verified_on_instrument")):
        raise AstigmaticDesignProvenanceError(
            "The checksum-bound design cannot claim instrument tag verification."
        )


@dataclass(frozen=True)
class AstigmaticCondenserParameterLayout:
    """Exact ordering and slices of the 43 nonlinear coordinates."""

    names: tuple[str, ...]

    def __post_init__(self) -> None:
        names = tuple(str(value).strip() for value in self.names)
        if len(names) != _PARAMETER_COUNT or any(not value for value in names):
            raise ValueError("Expected 43 non-empty parameter names.")
        if len(names) != len(set(names)):
            raise ValueError("Parameter names must be unique.")
        if any("K16" in value or "deflector" in value.lower() for value in names):
            raise ValueError("The no-commanded-tilt likelihood cannot contain K16.")
        object.__setattr__(self, "names", names)

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
                "shared_log_hole_radius": slice(26, 27),
                "shared_log_stage_xy_scale": slice(27, 28),
                "edge_local": slice(28, 40),
                "shared_log_realspace_scale": slice(40, 41),
                "shared_indexed_si_log_angular_scale": slice(41, 42),
                "shared_astigmatic_tag_angle_offset": slice(42, 43),
            }
        )


@dataclass(frozen=True)
class AstigmaticCondenserResidualLayout:
    """Exact ordering of 428 edge, prior, and sample residuals."""

    names: tuple[str, ...]

    def __post_init__(self) -> None:
        names = tuple(str(value).strip() for value in self.names)
        if len(names) != _RESIDUAL_COUNT or any(not value for value in names):
            raise ValueError("Expected 428 non-empty residual names.")
        if len(names) != len(set(names)):
            raise ValueError("Residual names must be unique.")
        if any("K16" in value or "deflector" in value.lower() for value in names):
            raise ValueError("The no-commanded-tilt residuals cannot contain K16.")
        object.__setattr__(self, "names", names)

    @property
    def size(self) -> int:
        return len(self.names)

    @property
    def slices(self) -> Mapping[str, slice]:
        return MappingProxyType(
            {
                "edge_poisson": slice(0, 88),
                "shared_priors": slice(88, 92),
                "sample_whitened": slice(92, 428),
            }
        )


class AstigmaticCondenserPrediction(NamedTuple):
    """Predicted absolute edge counts and 48-by-7 sample summaries."""

    edge_expected_counts: jnp.ndarray
    sample_observables: jnp.ndarray


@dataclass(frozen=True)
class AstigmaticLikelihoodRankDiagnostics:
    """Local rank audit, including unresolved full-parameter directions.

    ``profiled_target_singular_values`` are computed after projecting the
    twelve condenser columns out of the locally active nuisance column space.
    ``full_null_directions`` contains right-singular vectors below the declared
    full-model relative threshold. These directions are evidence of local
    equivalence families; they must not be interpreted as recovered physical
    parameters merely because the residual objective is small.
    """

    full_rank: int
    nuisance_rank: int
    profiled_target_rank: int
    full_singular_values: np.ndarray
    nuisance_singular_values: np.ndarray
    profiled_target_singular_values: np.ndarray
    full_null_directions: np.ndarray
    full_relative_tolerance: float
    nuisance_relative_tolerance: float
    target_relative_tolerance: float

    def __post_init__(self) -> None:
        full = _immutable_array(
            self.full_singular_values,
            "full_singular_values",
            shape=(_PARAMETER_COUNT,),
        )
        nuisance = _immutable_array(
            self.nuisance_singular_values,
            "nuisance_singular_values",
            shape=(_PARAMETER_COUNT - 12,),
        )
        target = _immutable_array(
            self.profiled_target_singular_values,
            "profiled_target_singular_values",
            shape=(12,),
        )
        nullity = _PARAMETER_COUNT - int(self.full_rank)
        null = _immutable_array(
            self.full_null_directions,
            "full_null_directions",
            shape=(nullity, _PARAMETER_COUNT),
        )
        ranks = (self.full_rank, self.nuisance_rank, self.profiled_target_rank)
        if any(int(value) != value or int(value) < 0 for value in ranks):
            raise ValueError("Ranks must be non-negative integers.")
        tolerances = (
            self.full_relative_tolerance,
            self.nuisance_relative_tolerance,
            self.target_relative_tolerance,
        )
        if not np.all(np.isfinite(tolerances)) or min(tolerances) <= 0.0:
            raise ValueError("Rank tolerances must be finite and positive.")
        object.__setattr__(self, "full_singular_values", full)
        object.__setattr__(self, "nuisance_singular_values", nuisance)
        object.__setattr__(self, "profiled_target_singular_values", target)
        object.__setattr__(self, "full_null_directions", null)

    @property
    def full_nullity(self) -> int:
        return _PARAMETER_COUNT - self.full_rank

    @property
    def targets_locally_identifiable(self) -> bool:
        return self.profiled_target_rank == 12


@dataclass(frozen=True)
class AstigmaticCondenserObservations:
    """Payload-bound raw counts and dense sample-summary uncertainty."""

    edge_counts: np.ndarray
    sample_observables: np.ndarray
    sample_noise_cholesky: np.ndarray
    astigmatic_design_sha256: str
    branch_b_payload_sha256: str
    coordinate_status: str
    provenance: str
    kind: ObservationKind = "instrument"

    def __post_init__(self) -> None:
        edge = _immutable_array(self.edge_counts, "edge_counts", shape=_EDGE_SHAPE)
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
        sample = _immutable_array(
            self.sample_observables,
            "sample_observables",
            shape=_SAMPLE_SHAPE,
        )
        cholesky = _lower_cholesky(
            self.sample_noise_cholesky,
            "sample_noise_cholesky",
            _SAMPLE_SIZE,
        )
        for name, digest in (
            ("astigmatic_design_sha256", self.astigmatic_design_sha256),
            ("branch_b_payload_sha256", self.branch_b_payload_sha256),
        ):
            value = str(digest).strip()
            if len(value) != 64 or any(
                char not in "0123456789abcdef" for char in value
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
            object.__setattr__(self, name, value)
        if self.coordinate_status != NORMALIZED_CONTROL_WARNING:
            raise AstigmaticDesignProvenanceError(
                "The normalized-control warning changed."
            )
        provenance = str(self.provenance).strip()
        if not provenance:
            raise ValueError("provenance must be non-empty.")
        object.__setattr__(self, "edge_counts", edge)
        object.__setattr__(self, "sample_observables", sample)
        object.__setattr__(self, "sample_noise_cholesky", cholesky)
        object.__setattr__(self, "provenance", provenance)


def _layouts(
    design: AstigmaticCondenserDesign,
    branch: BranchBExperiment,
) -> tuple[AstigmaticCondenserParameterLayout, AstigmaticCondenserResidualLayout]:
    retained_names = tuple(
        branch.parameter_layout.names[index] for index in _BRANCH_PARAMETER_INDICES
    )
    parameter_names = retained_names + (
        "nuisance.shared_log_realspace_scale",
        "nuisance.shared_indexed_si_log_angular_scale",
        "nuisance.shared_astigmatic_tag_angle_offset_rad",
    )
    edge_names = tuple(branch.residual_layout.names[:88])
    residual_names = (
        edge_names
        + (
            "prior.shared_log_hole_radius",
            "prior.shared_log_stage_xy_scale",
            "prior.registered_sample_log_realspace_scale",
            "prior.indexed_si_log_angular_scale",
        )
        + tuple(
            f"sample.acquisition_{acquisition:02d}.{observable}"
            for acquisition in range(48)
            for observable in (
                "log_rms_x",
                "log_rms_y",
                "correlation_x_y",
                "log_rms_theta_x",
                "log_rms_theta_y",
                "correlation_theta_x_theta_y",
                "astigmatic_tag_orientation",
            )
        )
    )
    return (
        AstigmaticCondenserParameterLayout(parameter_names),
        AstigmaticCondenserResidualLayout(residual_names),
    )


def _decode_physical(
    branch: BranchBExperiment,
    values: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    physical = jnp.asarray(branch.column.nominal_parameters, dtype=values.dtype)
    physical = physical.at[jnp.asarray(branch.target_parameter_indices)].add(
        values[:12] * jnp.asarray(branch.target_parameter_scales, dtype=values.dtype)
    )
    physical = physical.at[jnp.asarray(branch.upstream_parameter_indices)].add(
        values[22:26]
        * jnp.asarray(branch.upstream_parameter_scales, dtype=values.dtype)
    )
    return physical, values[12:22], values[28:40].reshape((4, 3))


def _predict(
    design: AstigmaticCondenserDesign,
    branch: BranchBExperiment,
    parameters: Any,
) -> AstigmaticCondenserPrediction:
    values = jnp.asarray(parameters)
    if values.shape != (_PARAMETER_COUNT,):
        raise ValueError(f"parameters must have shape ({_PARAMETER_COUNT},).")
    physical, entrance_coordinates, edge_local = _decode_physical(branch, values)
    entrance_covariance = branch.entrance_covariance_chart.covariance(
        entrance_coordinates
    )
    xy = jnp.asarray((0, 2))

    def edge_covariance(dacs: jnp.ndarray) -> jnp.ndarray:
        matrix = hard_edge_matrix_to_plane(
            branch.column, physical, dacs, branch.aperture_z_m
        )
        propagated = matrix @ entrance_covariance @ matrix.T
        return propagated[jnp.ix_(xy, xy)]

    edge_covariances = jax.vmap(edge_covariance)(jnp.asarray(branch.edge_controls))
    edge_prediction = aperture_edge_scan_from_moments(
        jnp.zeros((4, 2), dtype=physical.dtype),
        edge_covariances,
        aperture_centre_xy_m=edge_local[:, :2] * 1.0e-6,
        aperture_radius_m=branch.aperture_radius_m * jnp.exp(values[26]),
        incident_dose_electrons=(
            branch.edge_incident_electrons_per_frame * jnp.exp(edge_local[:, 2])
        ),
        aperture_centre_offsets_xy_m=(
            jnp.exp(values[27]) * jnp.asarray(branch.edge_commands_xy_m)
        ),
        radial_order=branch.edge_radial_order,
        azimuthal_order=branch.edge_azimuthal_order,
    )
    edge_counts = jnp.concatenate(
        (
            edge_prediction.expected_transmitted_electrons,
            edge_prediction.aperture_out_expected_electrons[:, None],
        ),
        axis=1,
    )

    full_covariance = axial_plane_covariance_observables_with_entrance_covariance(
        branch.column,
        physical,
        entrance_coordinates,
        jnp.asarray(design.expanded_controls),
        branch.entrance_covariance_chart,
        plane_z_m=branch.sample_z_m,
        mode="full",
    )
    selected = jnp.asarray(
        (*_SPATIAL_OBSERVABLE_INDICES, *_ANGULAR_OBSERVABLE_INDICES)
    )
    sample_covariance = full_covariance[:, selected]
    sample_covariance = sample_covariance.at[:, 0].add(values[40])
    sample_covariance = sample_covariance.at[:, 1].add(values[40])
    sample_covariance = sample_covariance.at[:, 3].add(values[41])
    sample_covariance = sample_covariance.at[:, 4].add(values[41])

    target_indices = np.asarray(branch.target_parameter_indices)
    q_bias_indices = jnp.asarray(target_indices[[2, 6, 10]])
    q_slope_indices = jnp.asarray(target_indices[[3, 7, 11]])
    condenser_q = physical[q_bias_indices] + jnp.asarray(
        design.expanded_controls[:, :3]
    ) * physical[q_slope_indices]
    upstream_names = branch.parameter_layout.upstream_names
    objective_q_index = int(
        branch.upstream_parameter_indices[
            upstream_names.index("Obj-prefield.q_bias")
        ]
    )
    tag_orientation = (
        values[42] - jnp.sum(condenser_q, axis=1) - physical[objective_q_index]
    )
    sample = jnp.concatenate(
        (sample_covariance, tag_orientation[:, None]), axis=1
    )
    return AstigmaticCondenserPrediction(edge_counts, sample)


def default_astigmatic_sample_noise_cholesky(
    design: AstigmaticCondenserDesign,
) -> np.ndarray:
    """Return the manifest's diagonal 48-by-7 summary-noise factor."""

    per_acquisition = np.asarray(
        (
            *([design.covariance_summary_sigma] * 6),
            design.tag_orientation_sigma_rad,
        )
    )
    result = np.diag(np.tile(per_acquisition, 48))
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class AstigmaticCondenserLikelihood:
    """JAX-compatible 43-parameter, 428-residual nonlinear likelihood."""

    design: AstigmaticCondenserDesign
    branch_experiment: BranchBExperiment
    observations: AstigmaticCondenserObservations
    parameter_layout: AstigmaticCondenserParameterLayout
    residual_layout: AstigmaticCondenserResidualLayout

    def __post_init__(self) -> None:
        if not isinstance(self.design, AstigmaticCondenserDesign):
            raise TypeError("design must be an AstigmaticCondenserDesign.")
        if not isinstance(self.branch_experiment, BranchBExperiment):
            raise TypeError("branch_experiment must be a BranchBExperiment.")
        if not isinstance(self.observations, AstigmaticCondenserObservations):
            raise TypeError("observations must be AstigmaticCondenserObservations.")
        if self.branch_experiment.payload_sha256 != self.design.branch_b_payload_sha256:
            raise AstigmaticDesignProvenanceError(
                "The Branch-B experiment differs from the bound parent payload."
            )
        if self.observations.astigmatic_design_sha256 != self.design.sha256:
            raise AstigmaticDesignProvenanceError(
                "Observations were not registered to this astigmatic design."
            )
        if (
            self.observations.branch_b_payload_sha256
            != self.design.branch_b_payload_sha256
        ):
            raise AstigmaticDesignProvenanceError(
                "Observations were not registered to the bound Branch-B payload."
            )
        if self.observations.coordinate_status != self.design.coordinate_status:
            raise AstigmaticDesignProvenanceError(
                "Observation coordinate status differs from the design."
            )
        _verify_tag_plane_contract(self.design)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self.parameter_layout.names

    @property
    def parameter_slices(self) -> Mapping[str, slice]:
        return self.parameter_layout.slices

    @property
    def residual_names(self) -> tuple[str, ...]:
        return self.residual_layout.names

    @property
    def residual_slices(self) -> Mapping[str, slice]:
        return self.residual_layout.slices

    @property
    def nominal_parameters(self) -> np.ndarray:
        result = np.zeros(_PARAMETER_COUNT, dtype=float)
        result.setflags(write=False)
        return result

    @property
    def hardware_ready(self) -> bool:
        return False

    def predict(self, parameters: Any) -> AstigmaticCondenserPrediction:
        return _predict(self.design, self.branch_experiment, parameters)

    def edge_data_residual(self, parameters: Any) -> jnp.ndarray:
        prediction = self.predict(parameters)
        return signed_poisson_deviance_residual(
            prediction.edge_expected_counts,
            jnp.asarray(self.observations.edge_counts),
        ).reshape(-1)

    def sample_data_residual(self, parameters: Any) -> jnp.ndarray:
        prediction = self.predict(parameters)
        difference = (
            prediction.sample_observables
            - jnp.asarray(self.observations.sample_observables)
        )
        # An ellipse orientation is pi-periodic.  Wrap only those 48 entries
        # before applying the declared dense joint covariance factor.
        wrapped_tag = 0.5 * jnp.arctan2(
            jnp.sin(2.0 * difference[:, 6]),
            jnp.cos(2.0 * difference[:, 6]),
        )
        difference = difference.at[:, 6].set(wrapped_tag)
        return jsp.linalg.solve_triangular(
            jnp.asarray(self.observations.sample_noise_cholesky),
            difference.reshape(-1),
            lower=True,
        )

    def data_residual(self, parameters: Any) -> jnp.ndarray:
        """Return the 424 measured-data rows, excluding four priors."""

        return jnp.concatenate(
            (self.edge_data_residual(parameters), self.sample_data_residual(parameters))
        )

    def prior_residual(self, parameters: Any) -> jnp.ndarray:
        """Return aperture, stage, real-space, and Si angular scale priors."""

        values = jnp.asarray(parameters)
        if values.shape != (_PARAMETER_COUNT,):
            raise ValueError(f"parameters must have shape ({_PARAMETER_COUNT},).")
        return jnp.asarray(
            (
                values[26] / self.branch_experiment.aperture_log_radius_prior_sigma,
                values[27]
                / self.branch_experiment.aperture_stage_log_scale_prior_sigma,
                values[40] / self.design.realspace_log_scale_prior_sigma,
                values[41] / self.design.angular_log_scale_prior_sigma,
            )
        )

    def residual(self, parameters: Any) -> jnp.ndarray:
        """Return rows in the exact committed 88 + 4 + 336 order."""

        return jnp.concatenate(
            (
                self.edge_data_residual(parameters),
                self.prior_residual(parameters),
                self.sample_data_residual(parameters),
            )
        )

    def objective(self, parameters: Any) -> jnp.ndarray:
        residual = self.residual(parameters)
        return 0.5 * jnp.dot(residual, residual)

    def jacobian(self, parameters: Any) -> jnp.ndarray:
        return jax.jacfwd(self.residual)(jnp.asarray(parameters))

    def numpy_residual(self, parameters: np.ndarray) -> np.ndarray:
        return np.asarray(self.residual(jnp.asarray(parameters)), dtype=float)

    def verify_nominal_design_jacobian(
        self,
        *,
        rtol: float = 2.0e-7,
        atol: float = 2.0e-7,
    ) -> np.ndarray:
        """Recompute and verify the exact 428-by-43 local design contract."""

        expected, _ = build_astigmatic_design_matrix(
            self.design,
        )
        actual = np.asarray(self.jacobian(jnp.zeros(_PARAMETER_COUNT)), dtype=float)
        np.testing.assert_allclose(actual, expected.matrix, rtol=rtol, atol=atol)
        actual.setflags(write=False)
        return actual

    def local_rank_diagnostics(
        self,
        parameters: Any | None = None,
        *,
        full_relative_tolerance: float = 1.0e-7,
        nuisance_relative_tolerance: float = 1.0e-10,
        target_relative_tolerance: float = 1.0e-7,
    ) -> AstigmaticLikelihoodRankDiagnostics:
        """Audit the local full, nuisance, and nuisance-profiled target ranks.

        The nuisance columns are normalized before their span is estimated,
        matching the committed design audit. The returned full-model null
        vectors make the two local equivalence directions explicit.
        """

        tolerances = (
            full_relative_tolerance,
            nuisance_relative_tolerance,
            target_relative_tolerance,
        )
        if not np.all(np.isfinite(tolerances)) or min(tolerances) <= 0.0:
            raise ValueError("Rank tolerances must be finite and positive.")
        point = self.nominal_parameters if parameters is None else parameters
        jacobian = np.asarray(self.jacobian(jnp.asarray(point)), dtype=float)
        if not np.all(np.isfinite(jacobian)):
            raise FloatingPointError("The local likelihood Jacobian is not finite.")

        _, full_singular, full_right = np.linalg.svd(jacobian, full_matrices=False)
        full_rank = int(
            np.sum(full_singular > full_relative_tolerance * full_singular[0])
        )

        target = jacobian[:, :12]
        nuisance = jacobian[:, 12:]
        nuisance_norms = np.linalg.norm(nuisance, axis=0)
        active = nuisance_norms > np.finfo(float).tiny
        normalized_nuisance = nuisance[:, active] / nuisance_norms[active][None, :]
        nuisance_left, nuisance_singular, _ = np.linalg.svd(
            normalized_nuisance,
            full_matrices=False,
        )
        nuisance_rank = int(
            np.sum(
                nuisance_singular
                > nuisance_relative_tolerance * nuisance_singular[0]
            )
        )
        projected_target = target - nuisance_left[:, :nuisance_rank] @ (
            nuisance_left[:, :nuisance_rank].T @ target
        )
        target_singular = np.linalg.svd(projected_target, compute_uv=False)
        target_rank = int(
            np.sum(target_singular > target_relative_tolerance * target_singular[0])
        )
        return AstigmaticLikelihoodRankDiagnostics(
            full_rank=full_rank,
            nuisance_rank=nuisance_rank,
            profiled_target_rank=target_rank,
            full_singular_values=full_singular,
            nuisance_singular_values=nuisance_singular,
            profiled_target_singular_values=target_singular,
            full_null_directions=full_right[full_rank:],
            full_relative_tolerance=float(full_relative_tolerance),
            nuisance_relative_tolerance=float(nuisance_relative_tolerance),
            target_relative_tolerance=float(target_relative_tolerance),
        )


def make_synthetic_astigmatic_observations(
    design: AstigmaticCondenserDesign,
    parameters: Any | None = None,
    *,
    sample_noise: bool = False,
    seed: int = 26090419,
    sample_noise_cholesky: np.ndarray | None = None,
) -> AstigmaticCondenserObservations:
    """Construct a matched expectation or a reproducible synthetic draw."""

    if not isinstance(design, AstigmaticCondenserDesign):
        raise TypeError("design must be an AstigmaticCondenserDesign.")
    _verify_tag_plane_contract(design)
    branch = load_branch_b_experiment(design.branch_b_payload_path)
    if branch.payload_sha256 != design.branch_b_payload_sha256:
        raise AstigmaticDesignProvenanceError(
            "The loaded Branch-B experiment differs from the bound parent."
        )
    truth = (
        np.zeros(_PARAMETER_COUNT, dtype=float)
        if parameters is None
        else np.asarray(parameters, dtype=float)
    )
    if truth.shape != (_PARAMETER_COUNT,):
        raise ValueError(f"parameters must have shape ({_PARAMETER_COUNT},).")
    prediction = _predict(design, branch, jnp.asarray(truth))
    edge = np.asarray(prediction.edge_expected_counts)
    sample = np.asarray(prediction.sample_observables)
    cholesky = (
        default_astigmatic_sample_noise_cholesky(design)
        if sample_noise_cholesky is None
        else np.asarray(sample_noise_cholesky, dtype=float)
    )
    if sample_noise:
        rng = np.random.default_rng(int(seed))
        edge = rng.poisson(edge).astype(float)
        sample = sample + (cholesky @ rng.normal(size=_SAMPLE_SIZE)).reshape(
            _SAMPLE_SHAPE
        )
        # Store measured ellipse orientations in a canonical pi-periodic range.
        sample[:, 6] = 0.5 * np.arctan2(
            np.sin(2.0 * sample[:, 6]), np.cos(2.0 * sample[:, 6])
        )
        kind: ObservationKind = "synthetic_draw"
    else:
        kind = "synthetic_expectation"
    return AstigmaticCondenserObservations(
        edge_counts=edge,
        sample_observables=sample,
        sample_noise_cholesky=cholesky,
        astigmatic_design_sha256=design.sha256,
        branch_b_payload_sha256=design.branch_b_payload_sha256,
        coordinate_status=design.coordinate_status,
        provenance=(
            "matched nonlinear synthetic astigmatic expectation"
            if not sample_noise
            else f"nonlinear synthetic astigmatic draw; numpy PCG64 seed={int(seed)}"
        ),
        kind=kind,
    )


def make_astigmatic_condenser_likelihood(
    design_path: str | Path = DEFAULT_ASTIGMATIC_DESIGN,
    *,
    observations: AstigmaticCondenserObservations | None = None,
    repository_root: str | Path | None = None,
) -> AstigmaticCondenserLikelihood:
    """Load both bound payloads and build the executable likelihood."""

    design = load_astigmatic_condenser_design(
        design_path,
        repository_root=repository_root,
    )
    branch = load_branch_b_experiment(design.branch_b_payload_path)
    parameter_layout, residual_layout = _layouts(design, branch)
    selected_observations = (
        make_synthetic_astigmatic_observations(design)
        if observations is None
        else observations
    )
    return AstigmaticCondenserLikelihood(
        design,
        branch,
        selected_observations,
        parameter_layout,
        residual_layout,
    )


__all__ = [
    "AstigmaticCondenserLikelihood",
    "AstigmaticLikelihoodRankDiagnostics",
    "AstigmaticCondenserObservations",
    "AstigmaticCondenserParameterLayout",
    "AstigmaticCondenserPrediction",
    "AstigmaticCondenserResidualLayout",
    "ObservationKind",
    "default_astigmatic_sample_noise_cholesky",
    "make_astigmatic_condenser_likelihood",
    "make_synthetic_astigmatic_observations",
]
