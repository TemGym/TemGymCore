"""JAX predictions for physically supported NEOARM calibration observables.

This module is deliberately narrower than an image-observable extractor.  It
predicts only quantities that follow from the first-order hard-edge column and
declared specimen/input calibrations:

* certificate reciprocal vectors in detector FFT-bin coordinates;
* certified multilayer boundary positions or adjacent layer widths along a
  frozen detector registration axis;
* calibrated sample-position/tilt centroid responses relative to a same-state
  perforation exposure;
* probe covariance, but only when a physical sample-plane covariance is
  explicitly supplied; and
* reciprocal vectors of a diffraction distribution expressed in angular
  coordinates and transformed by the native sample-angle map.

Intensity, attenuation, and material-dependent image moments are not implied
by a ray-transfer matrix and are therefore intentionally unsupported here.
Arrays in specifications are copied and made read-only.  Predictions remain
JAX differentiable with respect to the physical column parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

from .calibration_covariance import (
    FourierObservableCovarianceAugmentation,
    deduplicate_calibration_priors,
    make_fourier_observable_covariance_augmentation,
)
from .layer_metrology import (
    RegisteredLayerMetrology,
    RegisteredLayerPeriodMetrology,
    predict_registered_layer_observables,
    predict_registered_layer_period_observables,
)
from .microscope_inversion import (
    FrameObservationBlock,
    HeterogeneousInversionDataset,
)
from .neoarm_calibration import (
    DetectorGeometry,
    FrameRequest,
    HardEdgeColumnParameterization,
    KnownFourierFeatureSet,
    strip_shared_fourier_calibration_covariance,
)
from .physical_nuisance import GaussianPriorFactor


_POSITION = (0, 2)
_ANGLE = (1, 3)


class UnsupportedPhysicalObservableError(ValueError):
    """Raised when a dataset asks the first-order model for unsupported data."""


class SingularObservableMapError(ValueError):
    """Raised when an observable requires inversion of a singular optical map."""


def _immutable_vectors(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if (
        array.ndim != 2
        or array.shape[1:] != (2,)
        or array.shape[0] == 0
        or not np.all(np.isfinite(array))
        or np.any(np.linalg.norm(array, axis=1) <= 0.0)
    ):
        raise ValueError(f"{name} must be a finite non-zero array with shape (n, 2).")
    array = array.copy()
    array.setflags(write=False)
    return array


def _immutable_covariance(name: str, values: np.ndarray) -> np.ndarray:
    covariance = np.asarray(values, dtype=float)
    if covariance.shape != (4, 4) or not np.all(np.isfinite(covariance)):
        raise ValueError(f"{name} must be a finite 4x4 array.")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-20):
        raise ValueError(f"{name} must be symmetric.")
    scale = max(float(np.max(np.abs(covariance))), np.finfo(float).tiny)
    if np.linalg.eigvalsh(covariance).min() < -1e-12 * scale:
        raise ValueError(f"{name} must be positive semidefinite.")
    covariance = covariance.copy()
    covariance.setflags(write=False)
    return covariance


def _immutable_psd_covariance(
    name: str,
    values: np.ndarray,
    count: int,
) -> np.ndarray:
    covariance = np.asarray(values, dtype=float)
    if covariance.shape != (count, count) or not np.all(np.isfinite(covariance)):
        raise ValueError(f"{name} must be a finite {count}x{count} array.")
    scale = max(
        float(np.max(np.abs(covariance), initial=0.0)), np.finfo(float).tiny
    )
    if np.max(np.abs(covariance - covariance.T), initial=0.0) > 1e-10 * scale:
        raise ValueError(f"{name} must be symmetric.")
    covariance = 0.5 * (covariance + covariance.T)
    if np.linalg.eigvalsh(covariance).min(initial=0.0) < -1e-10 * scale:
        raise ValueError(f"{name} must be positive semidefinite.")
    covariance = covariance.copy()
    covariance.setflags(write=False)
    return covariance


def _labels(values: Sequence[str]) -> tuple[str, ...]:
    labels = tuple(str(value).strip() for value in values)
    if (
        not labels
        or any(not label for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("Feature labels must be non-empty and unique.")
    return labels


def _provenance(value: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError("Observable provenance must be non-empty.")
    return result


def _optional_nonempty(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    return _provenance(value)


def _certificate_group_name(
    explicit: str | None,
    certificate_id: str | None,
    labels: Sequence[str],
) -> str | None:
    if explicit is not None:
        return _provenance(explicit)
    if certificate_id is None:
        return None
    return f"specimen_fourier:{certificate_id}:{','.join(labels)}"


def _layer_certificate_group_name(
    explicit: str | None,
    certificate_id: str,
) -> str:
    if explicit is not None:
        return _provenance(explicit)
    return f"specimen_layers:{_provenance(certificate_id)}"


@dataclass(frozen=True)
class ImagingReciprocalVectors:
    """Certificate vectors transformed by ``g_d = H^{-T} g_sample``.

    Input vectors use specimen-world ``(x, y)`` coordinates in cycles/metre.
    Output components are detector FFT-bin offsets in conventional ``(y, x)``
    order.  ``name_prefix`` may be ``"material_pool."`` or
    ``"material_frame_<id>."`` to match an aggregated physical dataset.
    """

    request: FrameRequest
    labels: tuple[str, ...]
    world_reciprocal_vectors_xy_per_m: np.ndarray
    name_prefix: str = ""
    provenance: str = "certificate_reciprocal_vector"
    covariance_world_per_m2: np.ndarray | None = None
    certificate_id: str | None = None
    covariance_provenance: str | None = None
    calibration_group_name: str | None = None

    def __post_init__(self) -> None:
        labels = _labels(self.labels)
        vectors = _immutable_vectors(
            "world_reciprocal_vectors_xy_per_m",
            self.world_reciprocal_vectors_xy_per_m,
        )
        if vectors.shape[0] != len(labels):
            raise ValueError("Feature labels must match the reciprocal-vector count.")
        if self.request.measurement not in {"sample", "sample_repeat"}:
            raise ValueError("Imaging reciprocal vectors require a material frame.")
        covariance = self.covariance_world_per_m2
        certificate_id = _optional_nonempty(self.certificate_id, "certificate_id")
        covariance_provenance = _optional_nonempty(
            self.covariance_provenance, "covariance_provenance"
        )
        group_name = (
            None
            if covariance is None and self.calibration_group_name is None
            else _certificate_group_name(
                self.calibration_group_name,
                certificate_id,
                labels,
            )
        )
        if covariance is not None:
            covariance = _immutable_psd_covariance(
                "covariance_world_per_m2", covariance, 2 * len(labels)
            )
            if covariance_provenance is None:
                if certificate_id is None:
                    raise ValueError(
                        "Reciprocal-vector covariance requires certificate_id or "
                        "explicit covariance_provenance."
                    )
                covariance_provenance = f"certificate:{certificate_id}"
            if group_name is None:
                raise ValueError(
                    "Reciprocal-vector covariance requires a shared calibration "
                    "group name or certificate_id."
                )
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "world_reciprocal_vectors_xy_per_m", vectors)
        object.__setattr__(self, "name_prefix", str(self.name_prefix))
        object.__setattr__(self, "provenance", _provenance(self.provenance))
        object.__setattr__(self, "covariance_world_per_m2", covariance)
        object.__setattr__(self, "certificate_id", certificate_id)
        object.__setattr__(self, "covariance_provenance", covariance_provenance)
        object.__setattr__(self, "calibration_group_name", group_name)

    @property
    def observable_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for label in self.labels
            for name in (
                f"{self.name_prefix}fourier_{label}_y_px",
                f"{self.name_prefix}fourier_{label}_x_px",
            )
        )


@dataclass(frozen=True)
class ImagingLayerBoundaries:
    """Certificate layer geometry observed along one frozen detector axis.

    The associated :class:`RegisteredLayerMetrology` contains the local
    certificate points, fixed sample pose, and detector peak association.  Its
    reference-image peak centres are used only to keep edge identity and sign;
    prediction always maps the certificate coordinates through
    ``H = detector_affine @ A``.
    """

    request: FrameRequest
    metrology: RegisteredLayerMetrology
    name_prefix: str = ""
    provenance: str = "certificate_multilayer_interfaces"
    calibration_group_name: str | None = None

    def __post_init__(self) -> None:
        if self.request.measurement not in {"sample", "sample_repeat"}:
            raise ValueError("Imaging layer boundaries require a material frame.")
        if not isinstance(self.metrology, RegisteredLayerMetrology):
            raise TypeError("metrology must be RegisteredLayerMetrology.")
        object.__setattr__(self, "name_prefix", str(self.name_prefix))
        certificate = self.metrology.geometry.certificate_id
        region = self.metrology.geometry.region_name
        provenance = f"{_provenance(self.provenance)}:{certificate}:{region}"
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(
            self,
            "calibration_group_name",
            _layer_certificate_group_name(
                self.calibration_group_name,
                certificate,
            ),
        )

    @property
    def observable_names(self) -> tuple[str, ...]:
        return tuple(
            f"{self.name_prefix}{name}" for name in self.metrology.observable_names
        )


@dataclass(frozen=True)
class ImagingLayerPeriods:
    """Certified same-transition layer periods along one frozen detector axis.

    The wrapped :class:`RegisteredLayerPeriodMetrology` retains the registered
    boundary identities, but prediction maps its certificate coordinates
    through the current physical sample-to-detector position map.  The
    accumulated two-layer spans are therefore a distinct, CTF-robust physical
    observable rather than a relabelled failed adjacent-width measurement.
    """

    request: FrameRequest
    metrology: RegisteredLayerPeriodMetrology
    name_prefix: str = ""
    provenance: str = "certificate_multilayer_same_transition_periods"
    calibration_group_name: str | None = None

    def __post_init__(self) -> None:
        if self.request.measurement not in {"sample", "sample_repeat"}:
            raise ValueError("Imaging layer periods require a material frame.")
        if not isinstance(self.metrology, RegisteredLayerPeriodMetrology):
            raise TypeError("metrology must be RegisteredLayerPeriodMetrology.")
        object.__setattr__(self, "name_prefix", str(self.name_prefix))
        certificate = self.metrology.geometry.certificate_id
        region = self.metrology.geometry.region_name
        provenance = f"{_provenance(self.provenance)}:{certificate}:{region}"
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(
            self,
            "calibration_group_name",
            _layer_certificate_group_name(
                self.calibration_group_name,
                certificate,
            ),
        )

    @property
    def observable_names(self) -> tuple[str, ...]:
        return tuple(
            f"{self.name_prefix}{name}" for name in self.metrology.observable_names
        )


LayerCertificateObservablePart: TypeAlias = (
    ImagingLayerBoundaries | ImagingLayerPeriods
)


@dataclass(frozen=True)
class LayerCertificateCalibration:
    """One physical layer-coordinate prior shared by every optical state.

    ``boundary_indices`` use the immutable full-stack certificate identities,
    while ``boundary_coordinates_local_x_m`` contain canonical physical
    specimen coordinates.  Different crops select rows of this one vector and
    registered multilayer regions add fixed origin offsets; neither operation
    creates an independent copy of the certified width increments.  An optional
    same-transition boundary pair also drives the nonlinear reciprocal-period
    observable from this exact prior.
    """

    group_name: str
    certificate_id: str
    certificate_source: str
    boundary_indices: tuple[int, ...]
    boundary_coordinates_local_x_m: np.ndarray
    boundary_covariance_local_m2: np.ndarray
    boundary_uncertainty_model: str
    region_origin_offsets_local_x_m: tuple[tuple[str, float], ...] = ()
    multilayer_period_boundary_indices: tuple[int, int] | None = None
    multilayer_fourier_group_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        group = _provenance(self.group_name)
        certificate = _provenance(self.certificate_id)
        source = _provenance(self.certificate_source)
        indices = tuple(int(value) for value in self.boundary_indices)
        if (
            not indices
            or len(indices) != len(set(indices))
            or any(right <= left for left, right in zip(indices, indices[1:]))
        ):
            raise ValueError(
                "Layer-certificate boundary indices must be non-empty, unique, "
                "and strictly increasing."
            )
        coordinates = np.asarray(
            self.boundary_coordinates_local_x_m, dtype=float
        )
        if coordinates.shape != (len(indices),) or not np.all(
            np.isfinite(coordinates)
        ):
            raise ValueError(
                "boundary_coordinates_local_x_m must contain one finite value "
                "per certificate boundary."
            )
        if np.any(np.diff(coordinates) <= 0.0):
            raise ValueError("Layer-certificate coordinates must increase strictly.")
        covariance = _immutable_psd_covariance(
            "boundary_covariance_local_m2",
            self.boundary_covariance_local_m2,
            len(indices),
        )
        model = str(self.boundary_uncertainty_model)
        if model not in {
            "cumulative_layer_widths",
            "independent_boundary_marginals",
        }:
            raise ValueError("Unknown layer-boundary uncertainty model.")
        variance = np.diag(covariance)
        expected_covariance = (
            np.minimum.outer(variance, variance)
            if model == "cumulative_layer_widths"
            else np.diag(variance)
        )
        covariance_scale = max(
            float(np.max(np.abs(covariance), initial=0.0)),
            np.finfo(float).tiny,
        )
        if (
            np.max(
                np.abs(covariance - expected_covariance), initial=0.0
            )
            > 1e-10 * covariance_scale
        ):
            raise ValueError(
                "boundary_covariance_local_m2 is inconsistent with the "
                "declared layer-boundary uncertainty model."
            )
        coordinates = coordinates.copy()
        coordinates.setflags(write=False)
        object.__setattr__(self, "group_name", group)
        object.__setattr__(self, "certificate_id", certificate)
        object.__setattr__(self, "certificate_source", source)
        object.__setattr__(self, "boundary_indices", indices)
        object.__setattr__(
            self, "boundary_coordinates_local_x_m", coordinates
        )
        object.__setattr__(self, "boundary_covariance_local_m2", covariance)
        object.__setattr__(self, "boundary_uncertainty_model", model)
        offsets: list[tuple[str, float]] = []
        seen_regions: set[str] = set()
        for raw_region, raw_offset in self.region_origin_offsets_local_x_m:
            region = _provenance(raw_region)
            offset = float(raw_offset)
            if region in seen_regions or not np.isfinite(offset):
                raise ValueError(
                    "Layer-certificate region names must be unique and their "
                    "origin offsets finite."
                )
            seen_regions.add(region)
            offsets.append((region, offset))
        period_pair = self.multilayer_period_boundary_indices
        if period_pair is not None:
            period_pair = tuple(int(value) for value in period_pair)
            if (
                len(period_pair) != 2
                or period_pair[1] != period_pair[0] + 2
                or any(value not in indices for value in period_pair)
            ):
                raise ValueError(
                    "multilayer_period_boundary_indices must identify two "
                    "retained same-transition boundaries i and i+2."
                )
        fourier_groups = tuple(
            _provenance(value) for value in self.multilayer_fourier_group_names
        )
        if len(fourier_groups) != len(set(fourier_groups)):
            raise ValueError("Multilayer Fourier calibration groups must be unique.")
        object.__setattr__(
            self, "region_origin_offsets_local_x_m", tuple(offsets)
        )
        object.__setattr__(
            self, "multilayer_period_boundary_indices", period_pair
        )
        object.__setattr__(
            self, "multilayer_fourier_group_names", fourier_groups
        )

    @property
    def value_names(self) -> tuple[str, ...]:
        return tuple(
            f"boundary_{index:03d}.local_x_m" for index in self.boundary_indices
        )

    @property
    def prior(self) -> GaussianPriorFactor:
        variance = np.diag(self.boundary_covariance_local_m2)
        if self.boundary_uncertainty_model == "cumulative_layer_widths":
            incremental_variance = np.diff(np.concatenate(((0.0,), variance)))
            tolerance = 1e-12 * max(
                float(np.max(variance, initial=0.0)), np.finfo(float).tiny
            )
            if np.any(incremental_variance < -tolerance):
                raise ValueError(
                    "Cumulative layer-boundary variance must be non-decreasing."
                )
            incremental_variance = np.maximum(incremental_variance, 0.0)
            keep = incremental_variance > 0.0
            factor = np.tril(np.ones((len(variance), len(variance))))[
                :, keep
            ] * np.sqrt(incremental_variance[keep])
            retained_indices = np.asarray(self.boundary_indices)[keep]
            latent_names = tuple(
                f"cumulative_width_increment_through_boundary_{index:03d}_z"
                for index in retained_indices
            )
        else:
            keep = variance > 0.0
            factor = np.eye(len(variance))[:, keep] * np.sqrt(variance[keep])
            retained_indices = np.asarray(self.boundary_indices)[keep]
            latent_names = tuple(
                f"boundary_{index:03d}_marginal_z" for index in retained_indices
            )
        return GaussianPriorFactor(
            group_name=self.group_name,
            value_names=self.value_names,
            mean=self.boundary_coordinates_local_x_m,
            factor=factor,
            latent_names=latent_names,
            provenance=(
                f"layer-boundary certificate {self.certificate_id}: "
                f"{self.certificate_source}; covariance_model="
                f"{self.boundary_uncertainty_model}"
            ),
        )

    def rows_for_boundary_indices(
        self, boundary_indices: Sequence[int]
    ) -> tuple[int, ...]:
        row_by_index = {
            index: row for row, index in enumerate(self.boundary_indices)
        }
        try:
            return tuple(row_by_index[int(index)] for index in boundary_indices)
        except KeyError as error:
            raise ValueError(
                f"Layer calibration group {self.group_name!r} does not contain "
                f"certificate boundary index {error.args[0]}."
            ) from error

    def region_origin_offset_m(self, region_name: str) -> float:
        """Return the fixed registered origin offset for one physical region."""

        offsets = dict(self.region_origin_offsets_local_x_m)
        if not offsets:
            return 0.0
        try:
            return offsets[_provenance(region_name)]
        except KeyError as error:
            raise ValueError(
                f"Layer calibration group {self.group_name!r} has no registered "
                f"origin for region {region_name!r}."
            ) from error

    def boundary_coordinates_for_region(
        self,
        region_name: str,
        boundary_indices: Sequence[int],
        calibrated_coordinates: object,
    ) -> jnp.ndarray:
        """Select shared certificate coordinates and restore a region origin."""

        values = jnp.asarray(calibrated_coordinates)
        if values.shape != (len(self.boundary_indices),):
            raise ValueError(
                "calibrated_coordinates must contain one value per shared "
                "certificate boundary."
            )
        rows = self.rows_for_boundary_indices(boundary_indices)
        return values[jnp.asarray(rows)] + self.region_origin_offset_m(region_name)

    def multilayer_period_m(self, calibrated_coordinates: object) -> jnp.ndarray:
        """Evaluate the certified SiGe+Si span from shared boundary values."""

        pair = self.multilayer_period_boundary_indices
        if pair is None:
            raise ValueError(
                f"Layer calibration group {self.group_name!r} has no declared "
                "multilayer-period relation."
            )
        values = jnp.asarray(calibrated_coordinates)
        if values.shape != (len(self.boundary_indices),):
            raise ValueError(
                "calibrated_coordinates must contain one value per shared "
                "certificate boundary."
            )
        left, right = self.rows_for_boundary_indices(pair)
        return values[right] - values[left]


@dataclass(frozen=True)
class LayerCertificateObservableCovarianceAugmentation:
    """State-specific push-forward of one shared layer-certificate prior."""

    frame_id: int
    optical_state_index: int
    observable_names: tuple[str, ...]
    group_name: str
    certificate_id: str
    certificate_prior: GaussianPriorFactor
    boundary_indices: tuple[int, ...]
    calibration_value_rows: tuple[int, ...]
    observable_jacobian_px_per_m: np.ndarray
    certificate_output_factor: np.ndarray

    def __post_init__(self) -> None:
        if self.frame_id < 0 or self.optical_state_index < 0:
            raise ValueError("Frame and optical-state indices must be non-negative.")
        names = _labels(self.observable_names)
        group = _provenance(self.group_name)
        certificate = _provenance(self.certificate_id)
        prior = self.certificate_prior
        if not isinstance(prior, GaussianPriorFactor):
            raise TypeError("certificate_prior must be a GaussianPriorFactor.")
        if prior.group_name != group:
            raise ValueError("certificate_prior must use the declared group name.")
        indices = tuple(int(value) for value in self.boundary_indices)
        rows = tuple(int(value) for value in self.calibration_value_rows)
        if len(indices) != len(rows) or len(set(indices)) != len(indices):
            raise ValueError(
                "Boundary identities and calibration rows must align uniquely."
            )
        jacobian = np.asarray(self.observable_jacobian_px_per_m, dtype=float)
        factor = np.asarray(self.certificate_output_factor, dtype=float)
        if (
            jacobian.ndim != 2
            or jacobian.shape[0] != len(names)
            or jacobian.shape[1] != prior.n_values
            or not np.all(np.isfinite(jacobian))
            or factor.ndim != 2
            or factor.shape[0] != len(names)
            or not np.all(np.isfinite(factor))
        ):
            raise ValueError(
                "Layer-certificate Jacobian/factor rows must match observables."
            )
        expected_factor = jacobian @ prior.factor
        if factor.shape != expected_factor.shape or not np.allclose(
            factor, expected_factor, rtol=1e-11, atol=1e-15
        ):
            raise ValueError(
                "certificate_output_factor must equal the observable Jacobian "
                "times the shared certificate-prior factor."
            )
        jacobian = jacobian.copy()
        factor = factor.copy()
        jacobian.setflags(write=False)
        factor.setflags(write=False)
        object.__setattr__(self, "observable_names", names)
        object.__setattr__(self, "group_name", group)
        object.__setattr__(self, "certificate_id", certificate)
        object.__setattr__(self, "boundary_indices", indices)
        object.__setattr__(self, "calibration_value_rows", rows)
        object.__setattr__(self, "observable_jacobian_px_per_m", jacobian)
        object.__setattr__(self, "certificate_output_factor", factor)

    @property
    def certificate_covariance_px2(self) -> np.ndarray:
        covariance = self.certificate_output_factor @ self.certificate_output_factor.T
        covariance.setflags(write=False)
        return covariance


@dataclass(frozen=True)
class CentroidDifference:
    """A calibrated input response relative to a same-state perforation frame.

    ``sample_covariance`` is a physical phase-space covariance in interleaved
    ``(x, theta_x, y, theta_y)`` order.  Calibration covariance on the input
    state is measurement uncertainty and is not substituted for it.  When no
    physical covariance is supplied, only the two centroid differences are
    predicted.

    ``finite_detector_sample_covariance`` and ``finite_detector_reference_mean``
    opt into a finite-detector Gaussian centroid model, including the detector
    affine and MTF. They describe the *physical* reference probe at the sample,
    before adding either request's calibrated state. Use this path for clipped
    probes: the unbounded linear centroid is not their measured centroid.
    Covariance observables are not yet supported by this optional path.
    """

    request: FrameRequest
    perforation_request: FrameRequest
    sample_covariance: np.ndarray | None = None
    name_prefix: str | None = None
    provenance: str = "calibrated_sample_state_minus_same_state_perforation"
    finite_detector_sample_covariance: np.ndarray | None = None
    finite_detector_reference_mean: np.ndarray | None = None

    def __post_init__(self) -> None:
        request = self.request
        baseline = self.perforation_request
        if baseline.measurement != "perforation":
            raise ValueError("perforation_request must be a perforation exposure.")
        if request.measurement not in {
            "position_basis",
            "angle_basis",
            "basis_orthogonal",
            "basis_negative",
        }:
            raise ValueError("Centroid differences require a calibrated basis frame.")
        if (
            request.optical_state_index != baseline.optical_state_index
            or request.optical_state_label != baseline.optical_state_label
            or request.raw_dac_values != baseline.raw_dac_values
        ):
            raise ValueError(
                "A centroid difference requires same-state, same-DAC requests."
            )
        covariance = self.sample_covariance
        if covariance is not None:
            covariance = _immutable_covariance("sample_covariance", covariance)
        prefix = self.name_prefix
        if prefix is None:
            prefix = f"{request.measurement}_frame_{request.frame_id}."
        object.__setattr__(self, "sample_covariance", covariance)
        object.__setattr__(self, "name_prefix", str(prefix))
        object.__setattr__(self, "provenance", _provenance(self.provenance))

        finite_covariance = self.finite_detector_sample_covariance
        finite_mean = self.finite_detector_reference_mean
        if (finite_covariance is None) != (finite_mean is None):
            raise ValueError("Finite-detector prediction requires both a physical covariance and reference mean.")
        if finite_covariance is not None:
            if covariance is not None:
                raise ValueError("Finite-detector covariance observables are not implemented; select centroid differences only.")
            finite_covariance = _immutable_covariance("finite_detector_sample_covariance", finite_covariance)
            if np.linalg.eigvalsh(finite_covariance).min() <= 0:
                raise ValueError("Finite-detector beam covariance must be positive definite.")
            finite_mean = np.asarray(finite_mean, dtype=float).copy()
            if finite_mean.shape != (4,) or not np.all(np.isfinite(finite_mean)):
                raise ValueError("Finite-detector reference mean must have shape (4,).")
            finite_mean.setflags(write=False)
            object.__setattr__(self, "finite_detector_sample_covariance", finite_covariance)
            object.__setattr__(self, "finite_detector_reference_mean", finite_mean)

    @property
    def observable_names(self) -> tuple[str, ...]:
        names = (
            f"{self.name_prefix}delta_centroid_x_px",
            f"{self.name_prefix}delta_centroid_y_px",
        )
        if self.sample_covariance is None:
            return names
        return names + (
            f"{self.name_prefix}probe_covariance_xx_px2",
            f"{self.name_prefix}probe_covariance_xy_px2",
            f"{self.name_prefix}probe_covariance_yy_px2",
        )


@dataclass(frozen=True)
class AngularDistributionReciprocalVectors:
    """Angular-distribution covectors transformed by ``(D B)^{-T}``.

    The declared vectors are covectors of the sample angular distribution in
    cycles/radian, not Bragg-spot position vectors.  The latter obey the
    different direct relation ``theta = wavelength * g`` followed by ``D B``
    and must not be silently substituted here.  Predictions use the raw DACs
    of the actual indexed-diffraction exposure.
    """

    request: FrameRequest
    labels: tuple[str, ...]
    angular_reciprocal_vectors_xy_per_rad: np.ndarray
    name_prefix: str | None = None
    provenance: str = "indexed_diffraction_angular_reciprocal_vector"

    def __post_init__(self) -> None:
        if self.request.measurement != "indexed_diffraction":
            raise ValueError("Native diffraction vectors require a diffraction frame.")
        labels = _labels(self.labels)
        vectors = _immutable_vectors(
            "angular_reciprocal_vectors_xy_per_rad",
            self.angular_reciprocal_vectors_xy_per_rad,
        )
        if vectors.shape[0] != len(labels):
            raise ValueError("Feature labels must match the reciprocal-vector count.")
        prefix = self.name_prefix
        if prefix is None:
            prefix = f"diffraction_frame_{self.request.frame_id}."
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "angular_reciprocal_vectors_xy_per_rad", vectors)
        object.__setattr__(self, "name_prefix", str(prefix))
        object.__setattr__(self, "provenance", _provenance(self.provenance))

    @property
    def observable_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for label in self.labels
            for name in (
                f"{self.name_prefix}reciprocal_{label}_y_bin",
                f"{self.name_prefix}reciprocal_{label}_x_bin",
            )
        )


@dataclass(frozen=True)
class IndexedDiffractionBraggSpots:
    """Certified reciprocal vectors mapped to indexed Bragg-spot positions.

    For a specimen reciprocal vector ``g`` in cycles/metre, the paraxial
    scattering angle is ``theta = wavelength * g``.  The detector displacement
    is consequently ``D B theta``, where ``B`` is the sample-angle to detector-
    position block of the actual diffraction exposure's transfer matrix and
    ``D`` is the calibrated detector affine.  Outputs are signed pixel offsets
    from the direct beam in conventional ``(y, x)`` order.

    ``wavelength_provenance`` must identify the voltage calibration or other
    source from which the declared wavelength was obtained.  It is retained in
    each output's provenance and prevents an implicit 200-kV assumption.
    """

    request: FrameRequest
    labels: tuple[str, ...]
    world_reciprocal_vectors_xy_per_m: np.ndarray
    wavelength_m: float
    wavelength_provenance: str
    name_prefix: str | None = None
    provenance: str = "certificate_indexed_bragg_spot"
    covariance_world_per_m2: np.ndarray | None = None
    certificate_id: str | None = None
    covariance_provenance: str | None = None
    calibration_group_name: str | None = None
    wavelength_variance_m2: float | None = None
    wavelength_calibration_group_name: str | None = None

    def __post_init__(self) -> None:
        if self.request.measurement != "indexed_diffraction":
            raise ValueError("Indexed Bragg spots require a diffraction frame.")
        labels = _labels(self.labels)
        vectors = _immutable_vectors(
            "world_reciprocal_vectors_xy_per_m",
            self.world_reciprocal_vectors_xy_per_m,
        )
        if vectors.shape[0] != len(labels):
            raise ValueError("Feature labels must match the reciprocal-vector count.")
        wavelength = float(self.wavelength_m)
        if not np.isfinite(wavelength) or wavelength <= 0.0:
            raise ValueError("wavelength_m must be finite and positive.")
        wavelength_provenance = _provenance(self.wavelength_provenance)
        covariance = self.covariance_world_per_m2
        certificate_id = _optional_nonempty(self.certificate_id, "certificate_id")
        covariance_provenance = _optional_nonempty(
            self.covariance_provenance, "covariance_provenance"
        )
        group_name = (
            None
            if covariance is None and self.calibration_group_name is None
            else _certificate_group_name(
                self.calibration_group_name,
                certificate_id,
                labels,
            )
        )
        if covariance is not None:
            covariance = _immutable_psd_covariance(
                "covariance_world_per_m2", covariance, 2 * len(labels)
            )
            if covariance_provenance is None:
                if certificate_id is None:
                    raise ValueError(
                        "Bragg-vector covariance requires certificate_id or explicit "
                        "covariance_provenance."
                    )
                covariance_provenance = f"certificate:{certificate_id}"
            if group_name is None:
                raise ValueError(
                    "Bragg-vector covariance requires a shared calibration group "
                    "name or certificate_id."
                )
        wavelength_variance = self.wavelength_variance_m2
        wavelength_group = self.wavelength_calibration_group_name
        if wavelength_variance is not None:
            wavelength_variance = float(wavelength_variance)
            if not np.isfinite(wavelength_variance) or wavelength_variance < 0.0:
                raise ValueError(
                    "wavelength_variance_m2 must be finite and non-negative."
                )
            if wavelength_group is None:
                wavelength_group = f"electron_wavelength:{wavelength_provenance}"
            wavelength_group = _provenance(wavelength_group)
        elif wavelength_group is not None:
            raise ValueError(
                "A wavelength calibration group requires explicit "
                "wavelength_variance_m2."
            )
        prefix = self.name_prefix
        if prefix is None:
            prefix = f"diffraction_frame_{self.request.frame_id}."
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "world_reciprocal_vectors_xy_per_m", vectors)
        object.__setattr__(self, "wavelength_m", wavelength)
        object.__setattr__(self, "wavelength_provenance", wavelength_provenance)
        object.__setattr__(self, "name_prefix", str(prefix))
        object.__setattr__(self, "provenance", _provenance(self.provenance))
        object.__setattr__(self, "covariance_world_per_m2", covariance)
        object.__setattr__(self, "certificate_id", certificate_id)
        object.__setattr__(self, "covariance_provenance", covariance_provenance)
        object.__setattr__(self, "calibration_group_name", group_name)
        object.__setattr__(self, "wavelength_variance_m2", wavelength_variance)
        object.__setattr__(
            self, "wavelength_calibration_group_name", wavelength_group
        )

    @property
    def observable_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for label in self.labels
            for name in (
                f"{self.name_prefix}bragg_{label}_y_px",
                f"{self.name_prefix}bragg_{label}_x_px",
            )
        )

    @property
    def output_provenance(self) -> str:
        return f"{self.provenance};wavelength={self.wavelength_provenance}"


# Compatibility name retained for the initial public draft.  The preferred
# name makes clear that inverse-transpose transformation applies to an angular
# distribution's Fourier covectors, not to Bragg-spot positions.
NativeDiffractionReciprocalVectors = AngularDistributionReciprocalVectors


def imaging_reciprocal_vectors_from_feature_set(
    request: FrameRequest,
    feature_set: KnownFourierFeatureSet,
    *,
    name_prefix: str = "",
    provenance: str = "certificate_reciprocal_vector",
) -> ImagingReciprocalVectors:
    """Construct an imaging part without dropping certificate covariance."""

    if not isinstance(feature_set, KnownFourierFeatureSet):
        raise TypeError("feature_set must be a KnownFourierFeatureSet.")
    return ImagingReciprocalVectors(
        request=request,
        labels=feature_set.labels,
        world_reciprocal_vectors_xy_per_m=(
            feature_set.world_reciprocal_vectors_xy_per_m
        ),
        name_prefix=name_prefix,
        provenance=provenance,
        covariance_world_per_m2=feature_set.covariance_world_per_m2,
        certificate_id=feature_set.certificate_id,
        covariance_provenance=feature_set.covariance_provenance,
        calibration_group_name=feature_set.calibration_group_name,
    )


def indexed_diffraction_bragg_spots_from_feature_set(
    request: FrameRequest,
    feature_set: KnownFourierFeatureSet,
    wavelength_m: float,
    wavelength_provenance: str,
    *,
    wavelength_variance_m2: float,
    name_prefix: str | None = None,
    provenance: str = "certificate_indexed_bragg_spot",
    wavelength_calibration_group_name: str | None = None,
) -> IndexedDiffractionBraggSpots:
    """Construct a Bragg part with explicit specimen and wavelength covariance."""

    if not isinstance(feature_set, KnownFourierFeatureSet):
        raise TypeError("feature_set must be a KnownFourierFeatureSet.")
    return IndexedDiffractionBraggSpots(
        request=request,
        labels=feature_set.labels,
        world_reciprocal_vectors_xy_per_m=(
            feature_set.world_reciprocal_vectors_xy_per_m
        ),
        wavelength_m=wavelength_m,
        wavelength_provenance=wavelength_provenance,
        name_prefix=name_prefix,
        provenance=provenance,
        covariance_world_per_m2=feature_set.covariance_world_per_m2,
        certificate_id=feature_set.certificate_id,
        covariance_provenance=feature_set.covariance_provenance,
        calibration_group_name=feature_set.calibration_group_name,
        wavelength_variance_m2=wavelength_variance_m2,
        wavelength_calibration_group_name=wavelength_calibration_group_name,
    )


PhysicalObservablePart: TypeAlias = (
    ImagingReciprocalVectors
    | ImagingLayerBoundaries
    | ImagingLayerPeriods
    | CentroidDifference
    | AngularDistributionReciprocalVectors
    | IndexedDiffractionBraggSpots
)


@dataclass(frozen=True)
class PhysicalObservableBlockSpecification:
    """Immutable prediction contract for one heterogeneous dataset block."""

    frame_id: int
    optical_state_index: int
    parts: tuple[PhysicalObservablePart, ...]
    provenance: str = "physical_first_order_prediction"

    def __post_init__(self) -> None:
        parts = tuple(self.parts)
        if self.frame_id < 0 or self.optical_state_index < 0 or not parts:
            raise ValueError("A block requires non-negative IDs and at least one part.")
        allowed = (
            ImagingReciprocalVectors,
            ImagingLayerBoundaries,
            ImagingLayerPeriods,
            CentroidDifference,
            AngularDistributionReciprocalVectors,
            IndexedDiffractionBraggSpots,
        )
        if any(not isinstance(part, allowed) for part in parts):
            raise UnsupportedPhysicalObservableError(
                "Only calibrated centroid, certificate imaging-vector/layer, "
                "declared probe-covariance, and native diffraction-vector parts "
                "are supported."
            )
        if any(
            _part_request(part).optical_state_index != self.optical_state_index
            for part in parts
        ):
            raise ValueError("All block parts must use the block optical state.")
        names = tuple(name for part in parts for name in part.observable_names)
        if len(set(names)) != len(names):
            raise ValueError("Observable names must be unique within a block.")
        object.__setattr__(self, "parts", parts)
        object.__setattr__(self, "provenance", _provenance(self.provenance))

    @property
    def observable_names(self) -> tuple[str, ...]:
        return tuple(name for part in self.parts for name in part.observable_names)

    @property
    def observable_provenance(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (
                name,
                part.output_provenance
                if isinstance(part, IndexedDiffractionBraggSpots)
                else part.provenance,
            )
            for part in self.parts
            for name in part.observable_names
        )


def _part_request(part: PhysicalObservablePart) -> FrameRequest:
    return part.request


def _detector_metrics(detector: DetectorGeometry) -> tuple[jnp.ndarray, jnp.ndarray]:
    ny, nx = detector.shape_yx
    pixel_xy = jnp.asarray(
        (detector.pixel_size_yx_m[1], detector.pixel_size_yx_m[0])
    )
    field_of_view_xy = jnp.asarray((nx, ny)) * pixel_xy
    return pixel_xy, field_of_view_xy


def _has_tracer(values: object) -> bool:
    return any(
        isinstance(leaf, jax.core.Tracer)
        for leaf in jax.tree_util.tree_leaves(values)
    )


def _check_invertible(
    matrix: jnp.ndarray,
    *,
    map_name: str,
    singular_rtol: float,
    parameters: object,
) -> None:
    # Host-side validation gives a useful exception in simulation/fitting setup.
    # During tracing, JAX must retain a pure array program; singular trial points
    # then naturally produce non-finite solve outputs and are rejected by a fit.
    if _has_tracer(parameters):
        return
    values = np.asarray(jax.device_get(matrix), dtype=float)
    singular_values = np.linalg.svd(values, compute_uv=False)
    practical_scale = max(float(singular_values[0]), 1.0)
    if (
        not np.all(np.isfinite(singular_values))
        or singular_values[0] <= 0.0
        or singular_values[-1] <= singular_rtol * practical_scale
    ):
        raise SingularObservableMapError(
            f"{map_name} is singular at the requested optical state."
        )


def _sample_to_detector_matrix(
    column: HardEdgeColumnParameterization,
    parameters: jnp.ndarray,
    request: FrameRequest,
) -> jnp.ndarray:
    return column.matrix(
        parameters,
        jnp.asarray(request.raw_dac_values),
        segment="sample_to_detector",
    )


def _layer_geometry(
    part: LayerCertificateObservablePart,
):
    return part.metrology.geometry


def _is_multilayer_period_fourier_part(part: object) -> bool:
    """Recognise only the certificate's explicit one-vector period schema."""

    return (
        isinstance(part, ImagingReciprocalVectors)
        and part.labels == ("multilayer_period",)
        and part.certificate_id is not None
        and part.calibration_group_name is not None
        and part.world_reciprocal_vectors_xy_per_m.shape == (1, 2)
    )


def _arrays_consistent(
    first: np.ndarray,
    second: np.ndarray,
    *,
    rtol: float = 1e-10,
    absolute_floor: float = 1e-30,
) -> bool:
    first_array = np.asarray(first, dtype=float)
    second_array = np.asarray(second, dtype=float)
    if first_array.shape != second_array.shape:
        return False
    scale = max(
        float(np.max(np.abs(first_array), initial=0.0)),
        float(np.max(np.abs(second_array), initial=0.0)),
        np.finfo(float).tiny,
    )
    return bool(
        np.max(np.abs(first_array - second_array), initial=0.0)
        <= absolute_floor + rtol * scale
    )


def layer_certificate_calibrations_for_physical_observables(
    specifications: Sequence[PhysicalObservableBlockSpecification],
) -> tuple[LayerCertificateCalibration, ...]:
    """Assemble one physical multilayer prior per certificate.

    Boundary identities, rather than raster crops, join optical states.  The
    canonical coordinate vector is taken from ``multilayer_region_4`` when it
    is present (otherwise the lexicographically first region).  Other
    registered regions retain an exact fixed origin offset and share the same
    stochastic layer-width increments.  Thus translating a region cannot be
    mistaken for a new specimen certificate.

    If the same certificate also supplies the explicit low-magnification
    ``multilayer_period`` Fourier vector, a deterministic same-transition span
    ``i -> i+2`` is attached to this prior.  Its reciprocal is evaluated
    nonlinearly by the prediction callback.  The stored Fourier covariance is
    required to equal the local push-forward of these very same width
    increments; ambiguous/nonuniform declarations fail closed.
    """

    layer_by_certificate: dict[str, list[LayerCertificateObservablePart]] = {}
    fourier_by_certificate: dict[str, list[ImagingReciprocalVectors]] = {}
    for specification in tuple(specifications):
        for part in specification.parts:
            if isinstance(part, (ImagingLayerBoundaries, ImagingLayerPeriods)):
                certificate_id = _layer_geometry(part).certificate_id
                layer_by_certificate.setdefault(certificate_id, []).append(part)
            elif _is_multilayer_period_fourier_part(part):
                assert isinstance(part, ImagingReciprocalVectors)
                assert part.certificate_id is not None
                fourier_by_certificate.setdefault(part.certificate_id, []).append(part)

    calibrations: list[LayerCertificateCalibration] = []
    for certificate_id, parts in layer_by_certificate.items():
        first_geometry = _layer_geometry(parts[0])
        certificate_source = first_geometry.certificate_source
        uncertainty_model = first_geometry.boundary_uncertainty_model
        group_names = {part.calibration_group_name for part in parts}
        if len(group_names) != 1:
            raise ValueError(
                f"Certificate {certificate_id!r} uses multiple layer calibration "
                f"groups: {sorted(group_names)}."
            )
        group_name = next(iter(group_names))

        coordinates_by_region: dict[str, dict[int, float]] = {}
        variance_by_index: dict[int, float] = {}
        for part in parts:
            geometry = _layer_geometry(part)
            if (
                geometry.certificate_id != certificate_id
                or geometry.certificate_source != certificate_source
                or geometry.boundary_uncertainty_model != uncertainty_model
            ):
                raise ValueError(
                    f"Shared layer calibration group {group_name!r} mixes "
                    "incompatible certificate identities or uncertainty models."
                )
            local_x = np.asarray(
                geometry.boundary_points_local_xy_m[:, 0], dtype=float
            )
            variance = np.asarray(
                geometry.boundary_standard_uncertainties_m, dtype=float
            ) ** 2
            expected_subset_covariance = (
                np.minimum.outer(variance, variance)
                if uncertainty_model == "cumulative_layer_widths"
                else np.diag(variance)
            )
            if not _arrays_consistent(
                geometry.boundary_certificate_covariance_local_m2,
                expected_subset_covariance,
            ):
                raise ValueError(
                    f"Layer geometry in calibration group {group_name!r} has a "
                    "certificate covariance inconsistent with its declared model."
                )
            region_coordinates = coordinates_by_region.setdefault(
                geometry.region_name, {}
            )
            for index, coordinate, marginal_variance in zip(
                geometry.boundary_indices,
                local_x,
                variance,
                strict=True,
            ):
                if index in region_coordinates and not _arrays_consistent(
                    np.asarray((region_coordinates[index],)),
                    np.asarray((coordinate,)),
                    rtol=1e-12,
                    absolute_floor=1e-18,
                ):
                    raise ValueError(
                        f"Certificate boundary {index} has inconsistent coordinates "
                        f"within region {geometry.region_name!r}."
                    )
                region_coordinates[index] = float(coordinate)
                if index in variance_by_index and not _arrays_consistent(
                    np.asarray((variance_by_index[index],)),
                    np.asarray((marginal_variance,)),
                ):
                    raise ValueError(
                        f"Certificate boundary {index} has inconsistent uncertainty "
                        f"in group {group_name!r}."
                    )
                variance_by_index[index] = float(marginal_variance)

        region_names = tuple(sorted(coordinates_by_region))
        canonical_region = (
            "multilayer_region_4"
            if "multilayer_region_4" in coordinates_by_region
            else region_names[0]
        )
        coordinate_by_index = dict(coordinates_by_region[canonical_region])
        offsets: dict[str, float] = {canonical_region: 0.0}
        pending = set(region_names) - {canonical_region}
        while pending:
            progressed = False
            for region in tuple(sorted(pending)):
                values = coordinates_by_region[region]
                overlap = tuple(sorted(set(values).intersection(coordinate_by_index)))
                if not overlap:
                    continue
                differences = np.asarray(
                    tuple(values[index] - coordinate_by_index[index] for index in overlap)
                )
                offset = float(np.mean(differences))
                if not _arrays_consistent(
                    differences,
                    np.full(differences.shape, offset),
                    rtol=1e-12,
                    absolute_floor=1e-18,
                ):
                    raise ValueError(
                        f"Region {region!r} does not differ from canonical region "
                        "by one rigid layer-stack origin offset."
                    )
                offsets[region] = offset
                for index, coordinate in values.items():
                    canonical = coordinate - offset
                    if index in coordinate_by_index and not _arrays_consistent(
                        np.asarray((coordinate_by_index[index],)),
                        np.asarray((canonical,)),
                        rtol=1e-12,
                        absolute_floor=1e-18,
                    ):
                        raise ValueError(
                            f"Region {region!r} has an incompatible width at "
                            f"certificate boundary {index}."
                        )
                    if index not in coordinate_by_index:
                        coordinate_by_index[index] = canonical
                pending.remove(region)
                progressed = True
            if not progressed:
                raise ValueError(
                    f"Certificate {certificate_id!r} has disconnected region crops; "
                    "at least one common boundary identity is required to register "
                    "their origins without inventing an offset."
                )

        indices = tuple(sorted(coordinate_by_index))
        coordinates = np.asarray(
            tuple(coordinate_by_index[index] for index in indices), dtype=float
        )
        variances = np.asarray(
            tuple(variance_by_index[index] for index in indices), dtype=float
        )
        covariance = (
            np.minimum.outer(variances, variances)
            if uncertainty_model == "cumulative_layer_widths"
            else np.diag(variances)
        )

        period_parts = fourier_by_certificate.get(certificate_id, ())
        period_pair: tuple[int, int] | None = None
        fourier_groups: tuple[str, ...] = ()
        if period_parts:
            if uncertainty_model != "cumulative_layer_widths":
                raise ValueError(
                    "A unified multilayer-period Fourier relation requires the "
                    "cumulative_layer_widths certificate model."
                )
            candidate_pairs = tuple(
                (index, index + 2)
                for index in indices
                if index + 1 in coordinate_by_index
                and index + 2 in coordinate_by_index
            )
            if not candidate_pairs:
                raise ValueError(
                    f"Certificate {certificate_id!r} has no retained SiGe+Si "
                    "boundary triplet with which to define its Fourier period."
                )
            candidate_periods = np.asarray(
                tuple(
                    coordinate_by_index[right] - coordinate_by_index[left]
                    for left, right in candidate_pairs
                )
            )
            candidate_variances = np.asarray(
                tuple(
                    covariance[indices.index(right), indices.index(right)]
                    + covariance[indices.index(left), indices.index(left)]
                    - 2.0 * covariance[indices.index(left), indices.index(right)]
                    for left, right in candidate_pairs
                )
            )
            if not _arrays_consistent(
                candidate_periods,
                np.full(candidate_periods.shape, candidate_periods[0]),
                rtol=1e-10,
                absolute_floor=1e-18,
            ) or not _arrays_consistent(
                candidate_variances,
                np.full(candidate_variances.shape, candidate_variances[0]),
                rtol=1e-10,
                absolute_floor=1e-30,
            ):
                raise ValueError(
                    "The low-magnification multilayer period is ambiguous because "
                    "the retained certificate bilayers do not share one nominal "
                    "SiGe+Si span and uncertainty."
                )
            period_pair = candidate_pairs[0]
            period = float(candidate_periods[0])
            period_variance = float(candidate_variances[0])
            for part in period_parts:
                if part.covariance_world_per_m2 is None:
                    raise ValueError(
                        "Unified multilayer-period Fourier observations require "
                        "their standalone certificate covariance declaration."
                    )
                world = np.asarray(
                    part.world_reciprocal_vectors_xy_per_m, dtype=float
                ).reshape(2)
                derivative = -world / period
                expected = np.outer(derivative, derivative) * period_variance
                if not _arrays_consistent(
                    part.covariance_world_per_m2,
                    expected,
                    rtol=2e-8,
                    absolute_floor=1e-12,
                ):
                    raise ValueError(
                        "The multilayer Fourier covariance is not the push-forward "
                        "of the registered certificate's shared SiGe+Si width "
                        "increments."
                    )
            fourier_groups = tuple(
                dict.fromkeys(
                    str(part.calibration_group_name) for part in period_parts
                )
            )

        calibrations.append(
            LayerCertificateCalibration(
                group_name=group_name,
                certificate_id=certificate_id,
                certificate_source=certificate_source,
                boundary_indices=indices,
                boundary_coordinates_local_x_m=coordinates,
                boundary_covariance_local_m2=covariance,
                boundary_uncertainty_model=uncertainty_model,
                region_origin_offsets_local_x_m=tuple(
                    (region, offsets[region]) for region in region_names
                ),
                multilayer_period_boundary_indices=period_pair,
                multilayer_fourier_group_names=fourier_groups,
            )
        )
    return tuple(calibrations)


def layer_certificate_covariance_augmentations_for_physical_observables(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    detector: DetectorGeometry,
    specifications: Sequence[PhysicalObservableBlockSpecification],
) -> tuple[LayerCertificateObservableCovarianceAugmentation, ...]:
    """Push each shared physical layer prior into its state-specific pixels."""

    if not isinstance(column, HardEdgeColumnParameterization):
        raise TypeError("column must be a HardEdgeColumnParameterization.")
    if not isinstance(detector, DetectorGeometry):
        raise TypeError("detector must be a DetectorGeometry.")
    values = np.asarray(parameters, dtype=float)
    if values.shape != (column.n_parameters,) or not np.all(np.isfinite(values)):
        raise ValueError(
            f"parameters must be finite with shape ({column.n_parameters},)."
        )
    specs = tuple(specifications)
    calibration_by_group = {
        calibration.group_name: calibration
        for calibration in layer_certificate_calibrations_for_physical_observables(
            specs
        )
    }
    position_rows = jnp.asarray(_POSITION)
    affine = jnp.asarray(detector.affine_xy)
    augmentations: list[LayerCertificateObservableCovarianceAugmentation] = []
    for specification in specs:
        for part in specification.parts:
            if not isinstance(part, (ImagingLayerBoundaries, ImagingLayerPeriods)):
                continue
            calibration = calibration_by_group[part.calibration_group_name]
            prior = calibration.prior
            geometry = _layer_geometry(part)
            rows = calibration.rows_for_boundary_indices(
                geometry.boundary_indices
            )
            matrix = _sample_to_detector_matrix(
                column, jnp.asarray(values), part.request
            )
            H = affine @ matrix[jnp.ix_(position_rows, position_rows)]

            def predict_from_certificate(boundary_values):
                selected_x = (
                    boundary_values[jnp.asarray(rows)]
                    + calibration.region_origin_offset_m(geometry.region_name)
                )
                if isinstance(part, ImagingLayerBoundaries):
                    return predict_registered_layer_observables(
                        H,
                        detector.pixel_size_yx_m,
                        part.metrology,
                        boundary_coordinates_local_x_m=selected_x,
                    )
                return predict_registered_layer_period_observables(
                    H,
                    detector.pixel_size_yx_m,
                    part.metrology,
                    boundary_coordinates_local_x_m=selected_x,
                )

            jacobian = np.asarray(
                jax.jacfwd(predict_from_certificate)(
                    jnp.asarray(calibration.boundary_coordinates_local_x_m)
                ),
                dtype=float,
            )
            augmentations.append(
                LayerCertificateObservableCovarianceAugmentation(
                    frame_id=specification.frame_id,
                    optical_state_index=specification.optical_state_index,
                    observable_names=part.observable_names,
                    group_name=calibration.group_name,
                    certificate_id=calibration.certificate_id,
                    certificate_prior=prior,
                    boundary_indices=geometry.boundary_indices,
                    calibration_value_rows=rows,
                    observable_jacobian_px_per_m=jacobian,
                    certificate_output_factor=jacobian @ prior.factor,
                )
            )
    return tuple(augmentations)


def fourier_observable_covariance_augmentation(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    detector: DetectorGeometry,
    part: ImagingReciprocalVectors | IndexedDiffractionBraggSpots,
) -> FourierObservableCovarianceAugmentation:
    """Push certificate/wavelength covariance through the physical map.

    This host-side calculation uses the same hard-edge state and detector map
    as :func:`predict_physical_observable_blocks`.  Imaging uses
    ``FOV H^-T``; indexed diffraction uses the direct ``D B lambda g`` map.
    The returned input-space priors are shared across states, whereas its
    output covariance is specific to this request and reference parameter
    vector.
    """

    if not isinstance(column, HardEdgeColumnParameterization):
        raise TypeError("column must be a HardEdgeColumnParameterization.")
    if not isinstance(detector, DetectorGeometry):
        raise TypeError("detector must be a DetectorGeometry.")
    values = np.asarray(parameters, dtype=float)
    if values.shape != (column.n_parameters,) or not np.all(np.isfinite(values)):
        raise ValueError(
            f"parameters must be finite with shape ({column.n_parameters},)."
        )
    if part.covariance_world_per_m2 is None:
        raise ValueError(
            "A Fourier covariance augmentation requires explicit "
            "covariance_world_per_m2."
        )
    if part.certificate_id is None or part.calibration_group_name is None:
        raise ValueError(
            "A Fourier covariance augmentation requires certificate identity and "
            "a shared calibration group."
        )
    if part.covariance_provenance is None:
        raise ValueError("Certificate covariance provenance is required.")

    request = part.request
    matrix = np.asarray(
        column.matrix(
            values,
            np.asarray(request.raw_dac_values, dtype=float),
            segment="sample_to_detector",
        ),
        dtype=float,
    )
    affine = np.asarray(detector.affine_xy, dtype=float)
    permutation_yx_from_xy = np.asarray(((0.0, 1.0), (1.0, 0.0)))
    n_vectors = len(part.labels)
    map_provenance = (
        "HardEdgeColumnParameterization.sample_to_detector;"
        f"state={request.optical_state_label};frame={request.frame_id};"
        f"dac={tuple(float(value) for value in request.raw_dac_values)}"
    )

    if isinstance(part, ImagingReciprocalVectors):
        H = affine @ matrix[np.ix_(_POSITION, _POSITION)]
        singular_values = np.linalg.svd(H, compute_uv=False)
        if singular_values[-1] <= 1e-12 * max(singular_values[0], 1.0):
            raise SingularObservableMapError(
                "sample-to-detector position map H is singular at the requested "
                "optical state."
            )
        _pixel_xy, field_of_view_xy = _detector_metrics(detector)
        detector_map_xy = np.diag(np.asarray(field_of_view_xy)) @ np.linalg.inv(H).T
        one_vector_yx = permutation_yx_from_xy @ detector_map_xy
        jacobian = np.kron(np.eye(n_vectors), one_vector_yx)
        return make_fourier_observable_covariance_augmentation(
            observable_names=part.observable_names,
            observable_kind="imaging_reciprocal_vectors",
            labels=part.labels,
            world_reciprocal_vectors_xy_per_m=(
                part.world_reciprocal_vectors_xy_per_m
            ),
            covariance_world_per_m2=part.covariance_world_per_m2,
            world_vector_jacobian=jacobian,
            detector_map_xy=detector_map_xy,
            certificate_id=part.certificate_id,
            certificate_group_name=part.calibration_group_name,
            certificate_provenance=part.covariance_provenance,
            map_provenance=map_provenance,
        )

    if part.wavelength_variance_m2 is None:
        raise ValueError(
            "Indexed diffraction requires explicit wavelength_variance_m2, "
            "including zero when wavelength is treated as exact."
        )
    if part.wavelength_calibration_group_name is None:
        raise ValueError("Indexed diffraction requires a wavelength calibration group.")
    pixel_xy = np.asarray(
        (detector.pixel_size_yx_m[1], detector.pixel_size_yx_m[0]), dtype=float
    )
    detector_from_angle = affine @ matrix[np.ix_(_POSITION, _ANGLE)]
    detector_map_xy = np.diag(1.0 / pixel_xy) @ detector_from_angle
    one_vector_yx = permutation_yx_from_xy @ detector_map_xy
    unit_wavelength_jacobian = np.kron(np.eye(n_vectors), one_vector_yx)
    jacobian = part.wavelength_m * unit_wavelength_jacobian
    wavelength_jacobian = (
        unit_wavelength_jacobian
        @ part.world_reciprocal_vectors_xy_per_m.reshape(-1)
    )[:, None]
    return make_fourier_observable_covariance_augmentation(
        observable_names=part.observable_names,
        observable_kind="indexed_diffraction_bragg",
        labels=part.labels,
        world_reciprocal_vectors_xy_per_m=part.world_reciprocal_vectors_xy_per_m,
        covariance_world_per_m2=part.covariance_world_per_m2,
        world_vector_jacobian=jacobian,
        detector_map_xy=detector_map_xy,
        certificate_id=part.certificate_id,
        certificate_group_name=part.calibration_group_name,
        certificate_provenance=part.covariance_provenance,
        map_provenance=map_provenance,
        wavelength_m=part.wavelength_m,
        wavelength_variance_m2=part.wavelength_variance_m2,
        wavelength_jacobian=wavelength_jacobian,
        wavelength_group_name=part.wavelength_calibration_group_name,
        wavelength_provenance=part.wavelength_provenance,
        unit_wavelength_world_jacobian=unit_wavelength_jacobian,
    )


def calibration_covariance_augmentations_for_physical_observables(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    detector: DetectorGeometry,
    specifications: Sequence[PhysicalObservableBlockSpecification],
) -> tuple[FourierObservableCovarianceAugmentation, ...]:
    """Build every declared Fourier/Bragg calibration augmentation."""

    augmentations = []
    for specification in specifications:
        for part in specification.parts:
            if isinstance(part, (ImagingReciprocalVectors, IndexedDiffractionBraggSpots)):
                if part.covariance_world_per_m2 is None:
                    continue
                augmentations.append(
                    fourier_observable_covariance_augmentation(
                        column, parameters, detector, part
                    )
                )
    return tuple(augmentations)


def calibration_priors_for_physical_observables(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    detector: DetectorGeometry,
    specifications: Sequence[PhysicalObservableBlockSpecification],
) -> tuple[GaussianPriorFactor, ...]:
    """Return one shared input calibration prior per physical quantity."""

    augmentations = calibration_covariance_augmentations_for_physical_observables(
        column, parameters, detector, specifications
    )
    layer_calibrations = layer_certificate_calibrations_for_physical_observables(
        specifications
    )
    absorbed_fourier_groups = {
        group
        for calibration in layer_calibrations
        for group in calibration.multilayer_fourier_group_names
    }
    fourier_priors = deduplicate_calibration_priors(
        tuple(
            augmentation
            for augmentation in augmentations
            if augmentation.certificate_prior.group_name
            not in absorbed_fourier_groups
        )
    )
    # A multilayer Fourier vector and registered real-space interfaces describe
    # one certificate.  Retaining its old reciprocal-vector prior alongside the
    # shared boundary prior would count the same SiGe+Si uncertainty twice.
    layer_priors = tuple(calibration.prior for calibration in layer_calibrations)
    by_group: dict[str, GaussianPriorFactor] = {}
    for prior in (*fourier_priors, *layer_priors):
        existing = by_group.get(prior.group_name)
        if existing is None:
            by_group[prior.group_name] = prior
            continue
        if not (
            existing.value_names == prior.value_names
            and existing.latent_names == prior.latent_names
            and existing.provenance == prior.provenance
            and np.array_equal(existing.mean, prior.mean)
            and np.array_equal(existing.factor, prior.factor)
        ):
            raise ValueError(
                f"Shared calibration group {prior.group_name!r} has "
                "inconsistent physical definitions."
            )
    return tuple(by_group.values())


def make_calibration_aware_physical_predictor(
    column: HardEdgeColumnParameterization,
    reference_parameters: np.ndarray,
    detector: DetectorGeometry,
    specifications: Sequence[PhysicalObservableBlockSpecification],
    *,
    dataset: HeterogeneousInversionDataset | None = None,
    singular_rtol: float = 1e-12,
):
    """Return shared priors and a ``physical_nuisance`` prediction callback.

    The callback signature is ``predict(optical_parameters,
    calibration_values)``.  ``calibration_values`` must be in the returned
    prior order.  Every optical state then evaluates the exact physical map
    using the same specimen-vector and wavelength latent values.
    """

    specs = tuple(specifications)
    layer_calibrations = layer_certificate_calibrations_for_physical_observables(
        specs
    )
    unified_multilayer = tuple(
        calibration
        for calibration in layer_calibrations
        if calibration.multilayer_fourier_group_names
    )
    if dataset is not None and unified_multilayer:
        policy = dataset.metadata.get("multilayer_certificate_covariance")
        if not (
            isinstance(policy, Mapping)
            and policy.get("stripped_for_shared_latent_likelihood") is True
        ):
            raise ValueError(
                "A dataset combining multilayer Fourier and registered layer "
                "observables must be prepared with "
                "strip_shared_multilayer_certificate_covariance before the "
                "shared-latent predictor is constructed."
            )
    priors = calibration_priors_for_physical_observables(
        column, reference_parameters, detector, specs
    )

    def predict(optical_parameters, calibration_values):
        values = tuple(calibration_values)
        if len(values) != len(priors):
            raise ValueError(
                f"calibration_values must contain {len(priors)} groups in the "
                "returned prior order."
            )
        by_group = {
            prior.group_name: value
            for prior, value in zip(priors, values, strict=True)
        }
        return predict_physical_observable_blocks(
            column,
            optical_parameters,
            detector,
            specs,
            dataset=dataset,
            singular_rtol=singular_rtol,
            calibration_values_by_group=by_group,
        )

    return priors, predict


def _calibration_value(
    calibration_values_by_group: Mapping[str, object] | None,
    group_name: str | None,
    nominal: object,
    shape: tuple[int, ...],
) -> jnp.ndarray:
    if calibration_values_by_group is None or group_name is None:
        value = jnp.asarray(nominal)
    elif group_name in calibration_values_by_group:
        value = jnp.asarray(calibration_values_by_group[group_name])
    else:
        value = jnp.asarray(nominal)
    if value.shape != shape:
        raise ValueError(
            f"Calibration group {group_name!r} must provide values with shape "
            f"{shape}; received {value.shape}."
        )
    return value


def _predict_part(
    column: HardEdgeColumnParameterization,
    parameters: jnp.ndarray,
    detector: DetectorGeometry,
    part: PhysicalObservablePart,
    *,
    singular_rtol: float,
    calibration_values_by_group: Mapping[str, object] | None = None,
    layer_calibrations_by_group: Mapping[
        str, LayerCertificateCalibration
    ] | None = None,
) -> jnp.ndarray:
    request = _part_request(part)
    matrix = _sample_to_detector_matrix(column, parameters, request)
    affine = jnp.asarray(detector.affine_xy)
    pixel_xy, field_of_view_xy = _detector_metrics(detector)
    position_rows = jnp.asarray(_POSITION)

    if isinstance(part, ImagingReciprocalVectors):
        position_columns = jnp.asarray(_POSITION)
        H = affine @ matrix[jnp.ix_(position_rows, position_columns)]
        _check_invertible(
            H,
            map_name="sample-to-detector position map H",
            singular_rtol=singular_rtol,
            parameters=parameters,
        )
        multilayer_calibration = (
            None
            if layer_calibrations_by_group is None
            else layer_calibrations_by_group.get(part.calibration_group_name)
        )
        if (
            calibration_values_by_group is not None
            and multilayer_calibration is not None
            and _is_multilayer_period_fourier_part(part)
        ):
            calibrated = _calibration_value(
                calibration_values_by_group,
                multilayer_calibration.group_name,
                multilayer_calibration.boundary_coordinates_local_x_m,
                (len(multilayer_calibration.boundary_indices),),
            )
            period = multilayer_calibration.multilayer_period_m(calibrated)
            period_rows = multilayer_calibration.rows_for_boundary_indices(
                multilayer_calibration.multilayer_period_boundary_indices
            )
            nominal_coordinates = (
                multilayer_calibration.boundary_coordinates_local_x_m
            )
            nominal_period = float(
                nominal_coordinates[period_rows[1]]
                - nominal_coordinates[period_rows[0]]
            )
            # ``world * nominal_period`` is the fixed pose-transformed local
            # layer normal.  Only its reciprocal length changes with the
            # common physical SiGe+Si span.
            world_direction = (
                jnp.asarray(part.world_reciprocal_vectors_xy_per_m)
                * nominal_period
            )
            vectors = world_direction / period
        else:
            vectors = _calibration_value(
                calibration_values_by_group,
                part.calibration_group_name,
                part.world_reciprocal_vectors_xy_per_m.reshape(-1),
                (2 * len(part.labels),),
            ).reshape((-1, 2))
        detector_per_m = jnp.linalg.solve(H.T, vectors.T).T
        detector_bins_xy = detector_per_m * field_of_view_xy
        return detector_bins_xy[:, ::-1].reshape(-1)

    if isinstance(part, ImagingLayerBoundaries):
        position_columns = jnp.asarray(_POSITION)
        H = affine @ matrix[jnp.ix_(position_rows, position_columns)]
        boundary_x = None
        if calibration_values_by_group is not None:
            if (
                layer_calibrations_by_group is None
                or part.calibration_group_name not in layer_calibrations_by_group
            ):
                raise ValueError(
                    f"Missing layer-certificate layout for group "
                    f"{part.calibration_group_name!r}."
                )
            calibration = layer_calibrations_by_group[
                part.calibration_group_name
            ]
            calibrated = _calibration_value(
                calibration_values_by_group,
                calibration.group_name,
                calibration.boundary_coordinates_local_x_m,
                (len(calibration.boundary_indices),),
            )
            boundary_x = calibration.boundary_coordinates_for_region(
                part.metrology.geometry.region_name,
                part.metrology.geometry.boundary_indices,
                calibrated,
            )
        return predict_registered_layer_observables(
            H,
            detector.pixel_size_yx_m,
            part.metrology,
            boundary_coordinates_local_x_m=boundary_x,
        )

    if isinstance(part, ImagingLayerPeriods):
        position_columns = jnp.asarray(_POSITION)
        H = affine @ matrix[jnp.ix_(position_rows, position_columns)]
        boundary_x = None
        if calibration_values_by_group is not None:
            if (
                layer_calibrations_by_group is None
                or part.calibration_group_name not in layer_calibrations_by_group
            ):
                raise ValueError(
                    f"Missing layer-certificate layout for group "
                    f"{part.calibration_group_name!r}."
                )
            calibration = layer_calibrations_by_group[
                part.calibration_group_name
            ]
            calibrated = _calibration_value(
                calibration_values_by_group,
                calibration.group_name,
                calibration.boundary_coordinates_local_x_m,
                (len(calibration.boundary_indices),),
            )
            boundary_x = calibration.boundary_coordinates_for_region(
                part.metrology.geometry.region_name,
                part.metrology.geometry.boundary_indices,
                calibrated,
            )
        return predict_registered_layer_period_observables(
            H,
            detector.pixel_size_yx_m,
            part.metrology,
            boundary_coordinates_local_x_m=boundary_x,
        )

    if isinstance(part, CentroidDifference):
        if part.finite_detector_sample_covariance is not None:
            from .detector_moments import finite_gaussian_centroid
            position_map = matrix[position_rows, :]
            covariance = position_map @ jnp.asarray(part.finite_detector_sample_covariance) @ position_map.T
            reference = jnp.asarray(part.finite_detector_reference_mean)
            beam_mean = position_map @ (reference + part.request.sample_state.as_jax())
            baseline_mean = position_map @ (reference + part.perforation_request.sample_state.as_jax())
            return (finite_gaussian_centroid(beam_mean, covariance, detector)
                    - finite_gaussian_centroid(baseline_mean, covariance, detector))
        response_xy_px = (affine / pixel_xy[:, None]) @ matrix[position_rows, :]
        delta = (
            part.request.sample_state.as_jax()
            - part.perforation_request.sample_state.as_jax()
        )
        centroid = response_xy_px @ delta
        if part.sample_covariance is None:
            return centroid
        covariance = (
            response_xy_px
            @ jnp.asarray(part.sample_covariance)
            @ response_xy_px.T
        )
        if detector.mtf_sigma_px:
            covariance = covariance + detector.mtf_sigma_px**2 * jnp.eye(2)
        covariance_components = jnp.asarray(
            (covariance[0, 0], covariance[0, 1], covariance[1, 1])
        )
        return jnp.concatenate((centroid, covariance_components))

    if isinstance(part, IndexedDiffractionBraggSpots):
        angle_columns = jnp.asarray(_ANGLE)
        detector_from_angle = affine @ matrix[jnp.ix_(position_rows, angle_columns)]
        vectors = _calibration_value(
            calibration_values_by_group,
            part.calibration_group_name,
            part.world_reciprocal_vectors_xy_per_m.reshape(-1),
            (2 * len(part.labels),),
        ).reshape((-1, 2))
        wavelength = _calibration_value(
            calibration_values_by_group,
            part.wavelength_calibration_group_name,
            (part.wavelength_m,),
            (1,),
        )[0]
        angles_xy_rad = wavelength * vectors
        detector_xy_m = (detector_from_angle @ angles_xy_rad.T).T
        detector_xy_px = detector_xy_m / pixel_xy
        return detector_xy_px[:, ::-1].reshape(-1)

    if isinstance(part, AngularDistributionReciprocalVectors):
        angle_columns = jnp.asarray(_ANGLE)
        detector_from_angle = affine @ matrix[jnp.ix_(position_rows, angle_columns)]
        _check_invertible(
            detector_from_angle,
            map_name="sample-angle-to-detector map D B",
            singular_rtol=singular_rtol,
            parameters=parameters,
        )
        vectors = jnp.asarray(part.angular_reciprocal_vectors_xy_per_rad)
        detector_per_m = jnp.linalg.solve(detector_from_angle.T, vectors.T).T
        detector_bins_xy = detector_per_m * field_of_view_xy
        return detector_bins_xy[:, ::-1].reshape(-1)

    raise UnsupportedPhysicalObservableError(
        f"Unsupported physical observable part {type(part).__name__!r}."
    )


def validate_physical_observable_schema(
    specifications: Sequence[PhysicalObservableBlockSpecification],
    dataset: HeterogeneousInversionDataset,
) -> None:
    """Require one exactly named physical prediction for every dataset block."""

    specs = tuple(specifications)
    if len(specs) != len(dataset.blocks):
        raise UnsupportedPhysicalObservableError(
            "The specification count does not match the heterogeneous dataset."
        )
    for spec, block in zip(specs, dataset.blocks, strict=True):
        if (
            spec.frame_id != block.frame_id
            or spec.optical_state_index != block.optical_state_index
        ):
            raise UnsupportedPhysicalObservableError(
                "Physical block IDs/state indices do not match the dataset order."
            )
        if spec.observable_names != block.observable_names:
            unsupported = tuple(
                name
                for name in block.observable_names
                if name not in spec.observable_names
            )
            detail = (
                f" Unsupported dataset names: {unsupported}." if unsupported else ""
            )
            raise UnsupportedPhysicalObservableError(
                f"Observable schema mismatch for frame {block.frame_id}." + detail
            )


def predict_physical_observable_blocks(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    detector: DetectorGeometry,
    specifications: Sequence[PhysicalObservableBlockSpecification],
    *,
    dataset: HeterogeneousInversionDataset | None = None,
    singular_rtol: float = 1e-12,
    calibration_values_by_group: Mapping[str, object] | None = None,
) -> tuple[jnp.ndarray, ...]:
    """Predict JAX vectors in heterogeneous-dataset block order.

    Pass ``dataset`` to enforce an exact frame/state/name schema before a fit.
    ``singular_rtol`` is an explicit practical map-invertibility threshold and
    must be fixed across compared solutions.
    """

    if not np.isfinite(singular_rtol) or not 0.0 < singular_rtol < 1.0:
        raise ValueError("singular_rtol must be finite and lie strictly in (0, 1).")
    if not isinstance(column, HardEdgeColumnParameterization):
        raise TypeError("column must be a HardEdgeColumnParameterization.")
    if not isinstance(detector, DetectorGeometry):
        raise TypeError("detector must be a DetectorGeometry.")
    specs = tuple(specifications)
    if not specs:
        raise ValueError("At least one physical block specification is required.")
    if dataset is not None:
        validate_physical_observable_schema(specs, dataset)
    values = jnp.asarray(parameters)
    if values.shape != (column.n_parameters,):
        raise ValueError(f"parameters must have shape ({column.n_parameters},).")
    layer_calibrations = layer_certificate_calibrations_for_physical_observables(
        specs
    )
    layer_calibrations_by_group: dict[str, LayerCertificateCalibration] = {}
    for calibration in layer_calibrations:
        aliases = (
            calibration.group_name,
            *calibration.multilayer_fourier_group_names,
        )
        for alias in aliases:
            existing = layer_calibrations_by_group.get(alias)
            if existing is not None and existing is not calibration:
                raise ValueError(
                    f"Calibration group alias {alias!r} refers to multiple "
                    "multilayer certificates."
                )
            layer_calibrations_by_group[alias] = calibration
    if calibration_values_by_group is not None:
        declared_groups: set[str] = set()
        for specification in specs:
            for part in specification.parts:
                calibration_group = getattr(part, "calibration_group_name", None)
                if calibration_group is not None:
                    unified = layer_calibrations_by_group.get(calibration_group)
                    declared_groups.add(
                        calibration_group
                        if unified is None
                        else unified.group_name
                    )
                wavelength_group = getattr(
                    part, "wavelength_calibration_group_name", None
                )
                if wavelength_group is not None:
                    declared_groups.add(wavelength_group)
        unknown = set(calibration_values_by_group) - declared_groups
        missing = declared_groups - set(calibration_values_by_group)
        if unknown:
            raise ValueError(
                "Calibration values reference groups absent from the physical "
                f"specifications: {sorted(unknown)}."
            )
        if missing:
            raise ValueError(
                "Calibration values omit declared shared groups: "
                f"{sorted(missing)}."
            )
    predictions = []
    for spec in specs:
        parts = tuple(
            _predict_part(
                column,
                values,
                detector,
                part,
                singular_rtol=singular_rtol,
                calibration_values_by_group=calibration_values_by_group,
                layer_calibrations_by_group=layer_calibrations_by_group,
            )
            for part in spec.parts
        )
        predictions.append(jnp.concatenate(parts))
    return tuple(predictions)


def _stored_layer_certificate_corrections(
    dataset: HeterogeneousInversionDataset,
) -> dict[tuple[int, tuple[str, ...]], np.ndarray]:
    aggregation = dataset.metadata.get("aggregation")
    state_blocks = (
        aggregation.get("state_blocks", {})
        if isinstance(aggregation, Mapping)
        else {}
    )
    frame_metadata = dataset.metadata.get("frame_metadata", {})
    corrections: dict[tuple[int, tuple[str, ...]], np.ndarray] = {}

    def add(
        block: FrameObservationBlock,
        augmentation: Mapping[str, object],
        prefix: str,
    ) -> None:
        raw_names = tuple(
            str(value) for value in augmentation.get("observable_names", ())
        )
        if not raw_names:
            raise ValueError(
                "Stored layer-certificate covariance lacks observable names."
            )
        names = tuple(f"{prefix}{name}" for name in raw_names)
        covariance = np.asarray(
            augmentation.get("certificate_covariance_px2"), dtype=float
        )
        if covariance.shape != (len(names), len(names)) or not np.all(
            np.isfinite(covariance)
        ):
            raise ValueError(
                "Stored layer-certificate covariance has an invalid shape."
            )
        key = (block.frame_id, names)
        if key in corrections:
            raise ValueError(
                f"Frame {block.frame_id} declares the same layer-certificate "
                "covariance more than once."
            )
        corrections[key] = 0.5 * (covariance + covariance.T)

    for block in dataset.blocks:
        state_metadata = (
            state_blocks.get(str(block.frame_id))
            if isinstance(state_blocks, Mapping)
            else None
        )
        if isinstance(state_metadata, Mapping):
            material = state_metadata.get("material", ())
            if not isinstance(material, Sequence):
                raise ValueError("Material aggregation metadata must be a sequence.")
            for entry in material:
                if not isinstance(entry, Mapping):
                    continue
                mode = entry.get("mode")
                member_ids = tuple(entry.get("member_frame_ids", ()))
                if mode == "precision_pool":
                    prefix = "material_pool."
                elif mode == "retained" and member_ids:
                    prefix = f"material_frame_{int(member_ids[0])}."
                else:
                    raise ValueError(
                        "Cannot align layer-certificate covariance with its "
                        "aggregated material observables."
                    )
                for metadata_key in (
                    "layer_covariance_augmentation",
                    "layer_period_covariance_augmentation",
                ):
                    augmentation = entry.get(metadata_key)
                    if isinstance(augmentation, Mapping):
                        if (
                            metadata_key == "layer_period_covariance_augmentation"
                            and augmentation.get(
                                "certificate_covariance_added_once", True
                            )
                            is not True
                        ):
                            raise ValueError(
                                "Layer-period metadata does not certify that its "
                                "certificate covariance was added exactly once."
                            )
                        add(block, augmentation, prefix)
        elif isinstance(frame_metadata, Mapping):
            metadata = frame_metadata.get(str(block.frame_id), {})
            if isinstance(metadata, Mapping):
                for metadata_key in (
                    "layer_covariance_augmentation",
                    "layer_period_covariance_augmentation",
                ):
                    augmentation = metadata.get(metadata_key)
                    if isinstance(augmentation, Mapping):
                        add(block, augmentation, "")
    return corrections


def _stored_fourier_certificate_corrections(
    dataset: HeterogeneousInversionDataset,
) -> dict[tuple[int, tuple[str, ...]], np.ndarray]:
    """Read named standalone Fourier covariance blocks from dataset metadata."""

    aggregation = dataset.metadata.get("aggregation")
    state_blocks = (
        aggregation.get("state_blocks", {})
        if isinstance(aggregation, Mapping)
        else {}
    )
    frame_metadata = dataset.metadata.get("frame_metadata", {})
    corrections: dict[tuple[int, tuple[str, ...]], np.ndarray] = {}

    def add(
        block: FrameObservationBlock,
        augmentation: Mapping[str, object],
        prefix: str,
    ) -> None:
        raw_names = tuple(
            str(value) for value in augmentation.get("observable_names", ())
        )
        if not raw_names:
            raise ValueError("Stored Fourier covariance lacks observable names.")
        names = tuple(f"{prefix}{name}" for name in raw_names)
        covariance = np.asarray(
            augmentation.get("total_covariance_observable2"), dtype=float
        )
        if covariance.shape != (len(names), len(names)) or not np.all(
            np.isfinite(covariance)
        ):
            raise ValueError("Stored Fourier calibration covariance has an invalid shape.")
        key = (block.frame_id, names)
        if key in corrections:
            raise ValueError(
                f"Frame {block.frame_id} declares the same Fourier certificate "
                "covariance more than once."
            )
        corrections[key] = 0.5 * (covariance + covariance.T)

    for block in dataset.blocks:
        state_metadata = (
            state_blocks.get(str(block.frame_id))
            if isinstance(state_blocks, Mapping)
            else None
        )
        if isinstance(state_metadata, Mapping):
            material = state_metadata.get("material", ())
            if not isinstance(material, Sequence):
                raise ValueError("Material aggregation metadata must be a sequence.")
            for entry in material:
                if not isinstance(entry, Mapping):
                    continue
                augmentation = entry.get("fourier_covariance_augmentation")
                if not isinstance(augmentation, Mapping):
                    continue
                mode = entry.get("mode")
                member_ids = tuple(entry.get("member_frame_ids", ()))
                if mode == "precision_pool":
                    prefix = "material_pool."
                elif mode == "retained" and member_ids:
                    prefix = f"material_frame_{int(member_ids[0])}."
                else:
                    raise ValueError(
                        "Cannot align Fourier covariance with its aggregated "
                        "material observables."
                    )
                add(block, augmentation, prefix)
        elif isinstance(frame_metadata, Mapping):
            metadata = frame_metadata.get(str(block.frame_id), {})
            augmentation = (
                metadata.get("fourier_covariance_augmentation")
                if isinstance(metadata, Mapping)
                else None
            )
            if isinstance(augmentation, Mapping):
                add(block, augmentation, "")
    return corrections


def strip_shared_layer_certificate_covariance(
    dataset: HeterogeneousInversionDataset,
    column: HardEdgeColumnParameterization,
    reference_parameters: np.ndarray,
    detector: DetectorGeometry,
    specifications: Sequence[PhysicalObservableBlockSpecification],
    *,
    covariance_rtol: float = 2e-7,
) -> HeterogeneousInversionDataset:
    """Remove layer-certificate block terms before using shared latents.

    Checkpoint blocks are complete standalone measurements and therefore carry
    a state-specific push-forward of the certificate covariance.  A joint
    cross-state likelihood instead evaluates one common set of physical
    boundary coordinates.  This function verifies the stored push-forward
    against that exact predictor Jacobian and subtracts *only* the certificate
    term.  Detector/bootstrap covariance and independently validated model
    discrepancy remain in each block.

    Missing, duplicated, already-stripped, or numerically inconsistent
    declarations fail closed; silently fitting both representations would
    count the same certificate twice.
    """

    if not isinstance(dataset, HeterogeneousInversionDataset):
        raise TypeError("dataset must be a HeterogeneousInversionDataset.")
    if not np.isfinite(covariance_rtol) or not 0.0 <= covariance_rtol < 1.0:
        raise ValueError("covariance_rtol must lie in [0, 1).")
    specs = tuple(specifications)
    validate_physical_observable_schema(specs, dataset)
    existing_policy = dataset.metadata.get("layer_certificate_covariance")
    if (
        isinstance(existing_policy, Mapping)
        and existing_policy.get("stripped_for_shared_latent_likelihood") is True
    ):
        raise ValueError(
            "Layer-certificate covariance has already been stripped from this dataset."
        )

    expected_augmentations = (
        layer_certificate_covariance_augmentations_for_physical_observables(
            column,
            reference_parameters,
            detector,
            specs,
        )
    )
    expected = {
        (augmentation.frame_id, augmentation.observable_names): augmentation
        for augmentation in expected_augmentations
    }
    if len(expected) != len(expected_augmentations):
        raise ValueError(
            "Layer observable names must identify unique covariance subblocks."
        )
    stored = _stored_layer_certificate_corrections(dataset)
    missing = set(expected) - set(stored)
    extra = set(stored) - set(expected)
    if missing or extra:
        raise ValueError(
            "Stored and predicted layer-certificate covariance declarations do "
            f"not match; missing={sorted(missing)}, extra={sorted(extra)}."
        )

    correction_by_frame: dict[int, list[tuple[tuple[str, ...], np.ndarray]]] = {}
    for key, augmentation in expected.items():
        covariance = stored[key]
        predicted = augmentation.certificate_covariance_px2
        if not _arrays_consistent(
            covariance,
            predicted,
            rtol=float(covariance_rtol),
            absolute_floor=1e-14,
        ):
            maximum = float(np.max(np.abs(covariance - predicted), initial=0.0))
            raise ValueError(
                f"Frame {augmentation.frame_id} stored layer-certificate "
                "covariance is inconsistent with the shared physical latent "
                f"push-forward (maximum absolute difference {maximum:.6g} px^2)."
            )
        correction_by_frame.setdefault(augmentation.frame_id, []).append(
            (augmentation.observable_names, covariance)
        )

    stripped_blocks: list[FrameObservationBlock] = []
    for block in dataset.blocks:
        statistical = np.asarray(block.covariance, dtype=float).copy()
        occupied: set[int] = set()
        for names, covariance in correction_by_frame.get(block.frame_id, ()):
            try:
                indices = tuple(block.observable_names.index(name) for name in names)
            except ValueError as error:
                raise ValueError(
                    f"Frame {block.frame_id} cannot align its layer-certificate "
                    "covariance with the observable schema."
                ) from error
            if occupied.intersection(indices):
                raise ValueError(
                    f"Frame {block.frame_id} has overlapping layer-certificate "
                    "covariance subblocks."
                )
            occupied.update(indices)
            statistical[np.ix_(indices, indices)] -= covariance
        statistical = 0.5 * (statistical + statistical.T)
        eigenvalues = np.linalg.eigvalsh(statistical)
        scale = max(
            float(np.max(np.abs(statistical), initial=0.0)),
            np.finfo(float).tiny,
        )
        if eigenvalues.min() <= 0.0:
            if eigenvalues.min() < -1e-9 * scale:
                raise ValueError(
                    f"Frame {block.frame_id} covariance does not contain its "
                    "declared layer-certificate term exactly once."
                )
            floor = max(1e-18, 1e-12 * scale)
            statistical += np.eye(statistical.shape[0]) * (
                floor - eigenvalues.min()
            )
        stripped_blocks.append(
            FrameObservationBlock(
                frame_id=block.frame_id,
                optical_state_index=block.optical_state_index,
                values=block.values,
                covariance=statistical,
                observable_names=block.observable_names,
            )
        )

    calibrations = layer_certificate_calibrations_for_physical_observables(specs)
    metadata = dict(dataset.metadata)
    metadata["layer_certificate_covariance"] = {
        "previous_declaration": (
            None
            if isinstance(existing_policy, Mapping)
            else existing_policy
        ),
        "per_frame_output_covariance_included": False,
        "stripped_for_shared_latent_likelihood": True,
        "stripped_group_names": [value.group_name for value in calibrations],
        "stripped_subblock_count": len(expected_augmentations),
        "global_fit_policy": (
            "detector/bootstrap and validated model-discrepancy covariance only; "
            "evaluate one shared physical boundary-coordinate prior per certificate"
        ),
    }
    return HeterogeneousInversionDataset(
        plan=dataset.plan,
        blocks=tuple(stripped_blocks),
        metadata=metadata,
    )


def strip_shared_multilayer_certificate_covariance(
    dataset: HeterogeneousInversionDataset,
    column: HardEdgeColumnParameterization,
    reference_parameters: np.ndarray,
    detector: DetectorGeometry,
    specifications: Sequence[PhysicalObservableBlockSpecification],
    *,
    covariance_rtol: float = 2e-7,
) -> HeterogeneousInversionDataset:
    """Strip standalone terms for one unified real/reciprocal certificate fit.

    This is the fail-closed integration entry point when a dataset contains
    both low-magnification ``multilayer_period`` Fourier vectors and registered
    layer boundaries or periods.  It first verifies that both representations
    are driven by one :class:`LayerCertificateCalibration`, then removes the
    per-frame Fourier and real-space certificate covariance terms.  The
    calibration-aware predictor subsequently restores them once, including
    their cross-state and Fourier-to-boundary cross covariance.

    Other declared Fourier calibrations (for example the Si {111} spacing and
    electron wavelength) are stripped by the same first pass and remain as
    their own shared priors.  Statistical/bootstrap and validated discrepancy
    covariance are retained.
    """

    specs = tuple(specifications)
    validate_physical_observable_schema(specs, dataset)
    calibrations = layer_certificate_calibrations_for_physical_observables(specs)
    unified = tuple(
        calibration
        for calibration in calibrations
        if calibration.multilayer_fourier_group_names
        and calibration.multilayer_period_boundary_indices is not None
    )
    if not unified:
        raise ValueError(
            "No certificate is represented by both a multilayer-period Fourier "
            "observable and registered layer geometry."
        )
    existing = dataset.metadata.get("multilayer_certificate_covariance")
    if (
        isinstance(existing, Mapping)
        and existing.get("stripped_for_shared_latent_likelihood") is True
    ):
        raise ValueError(
            "Unified multilayer certificate covariance has already been stripped."
        )
    unified_fourier_groups = {
        group
        for calibration in unified
        for group in calibration.multilayer_fourier_group_names
    }
    expected_period_covariance: dict[
        tuple[int, tuple[str, ...]], np.ndarray
    ] = {}
    for specification in specs:
        for part in specification.parts:
            if not (
                _is_multilayer_period_fourier_part(part)
                and part.calibration_group_name in unified_fourier_groups
            ):
                continue
            assert isinstance(part, ImagingReciprocalVectors)
            augmentation = fourier_observable_covariance_augmentation(
                column,
                reference_parameters,
                detector,
                part,
            )
            key = (specification.frame_id, part.observable_names)
            if key in expected_period_covariance:
                raise ValueError(
                    "Multilayer-period observable names must identify unique "
                    "Fourier covariance subblocks."
                )
            expected_period_covariance[key] = (
                augmentation.total_covariance_observable2
            )
    stored_fourier = _stored_fourier_certificate_corrections(dataset)
    missing_period = set(expected_period_covariance) - set(stored_fourier)
    if missing_period:
        raise ValueError(
            "The dataset lacks standalone Fourier covariance for unified "
            f"multilayer-period blocks: {sorted(missing_period)}."
        )
    for key, expected_covariance in expected_period_covariance.items():
        if not _arrays_consistent(
            stored_fourier[key],
            expected_covariance,
            rtol=float(covariance_rtol),
            absolute_floor=1e-12,
        ):
            maximum = float(
                np.max(
                    np.abs(stored_fourier[key] - expected_covariance),
                    initial=0.0,
                )
            )
            raise ValueError(
                f"Frame {key[0]} stored multilayer Fourier covariance is "
                "inconsistent with its physical certificate push-forward "
                f"(maximum absolute difference {maximum:.6g})."
            )
    fourier_stripped = strip_shared_fourier_calibration_covariance(dataset)
    result = strip_shared_layer_certificate_covariance(
        fourier_stripped,
        column,
        reference_parameters,
        detector,
        specs,
        covariance_rtol=covariance_rtol,
    )
    metadata = dict(result.metadata)
    metadata["multilayer_certificate_covariance"] = {
        "stripped_for_shared_latent_likelihood": True,
        "certificate_ids": [value.certificate_id for value in unified],
        "shared_layer_group_names": [value.group_name for value in unified],
        "absorbed_fourier_group_names": [
            group
            for value in unified
            for group in value.multilayer_fourier_group_names
        ],
        "period_boundary_indices": {
            value.certificate_id: list(value.multilayer_period_boundary_indices)
            for value in unified
        },
        "region_origin_offsets_local_x_m": {
            value.certificate_id: {
                region: offset
                for region, offset in value.region_origin_offsets_local_x_m
            }
            for value in unified
        },
        "global_fit_policy": (
            "one shared cumulative layer-width latent per certificate; registered "
            "real-space boundaries/periods and nonlinear reciprocal SiGe+Si period "
            "are evaluated jointly exactly once"
        ),
    }
    return HeterogeneousInversionDataset(
        plan=result.plan,
        blocks=result.blocks,
        metadata=metadata,
    )


def specifications_from_aggregated_dataset(
    dataset: HeterogeneousInversionDataset,
    frame_requests: Sequence[FrameRequest],
) -> tuple[PhysicalObservableBlockSpecification, ...]:
    """Reconstruct physical specs from aggregation metadata and frame requests.

    This is the direct integration path for
    :func:`neoarm_calibration.aggregate_checkpoint_observables_for_fit`.
    Material certificate vectors or registered interface geometries are read
    from its preserved provenance; raw detector registration peaks are never
    used as inverse truth.
    """

    aggregation = dataset.metadata.get("aggregation")
    if not isinstance(aggregation, Mapping):
        raise ValueError("dataset lacks physical aggregation metadata.")
    state_blocks = aggregation.get("state_blocks")
    if not isinstance(state_blocks, Mapping):
        raise ValueError("dataset aggregation lacks state_blocks metadata.")
    request_by_id = {request.frame_id: request for request in frame_requests}
    if len(request_by_id) != len(tuple(frame_requests)):
        raise ValueError("frame_requests must have unique frame IDs.")

    specifications = []
    for block in dataset.blocks:
        metadata = state_blocks.get(str(block.frame_id))
        if not isinstance(metadata, Mapping):
            raise ValueError(
                f"Missing aggregation metadata for block {block.frame_id}."
            )
        try:
            baseline = request_by_id[int(metadata["baseline_frame_id"])]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid baseline metadata for block {block.frame_id}."
            ) from error
        parts: list[PhysicalObservablePart] = []
        for frame_id in metadata.get("response_frame_ids", ()):
            try:
                request = request_by_id[int(frame_id)]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Missing response frame {frame_id!r} for block {block.frame_id}."
                ) from error
            parts.append(CentroidDifference(request, baseline))

        material_entries = metadata.get("material", ())
        if not isinstance(material_entries, Sequence):
            raise ValueError("Material aggregation metadata must be a sequence.")
        for entry in material_entries:
            if not isinstance(entry, Mapping):
                raise ValueError("Each material provenance entry must be a mapping.")
            member_ids = tuple(
                int(value) for value in entry.get("member_frame_ids", ())
            )
            if not member_ids or member_ids[0] not in request_by_id:
                raise ValueError("Material provenance must identify source frame(s).")
            feature = entry.get("known_fourier_features")
            layer_metrology = entry.get("known_layer_metrology")
            layer_period_metrology = entry.get("known_layer_period_metrology")
            if (
                not isinstance(feature, Mapping)
                and not isinstance(layer_metrology, Mapping)
                and not isinstance(layer_period_metrology, Mapping)
            ):
                raise UnsupportedPhysicalObservableError(
                    "A material block lacks certificate reciprocal-vector, layer-"
                    "interface, or layer-period provenance."
                )
            mode = entry.get("mode")
            if mode == "precision_pool":
                prefix = "material_pool."
            elif mode == "retained":
                prefix = f"material_frame_{member_ids[0]}."
            else:
                raise ValueError(f"Unknown material aggregation mode {mode!r}.")
            if isinstance(feature, Mapping):
                labels = tuple(feature.get("labels", ()))
                vectors = np.asarray(
                    feature.get("world_reciprocal_vectors_xy_per_m")
                )
                certificate_id = feature.get("certificate_id")
                covariance_value = feature.get("covariance_world_per_m2")
                covariance = (
                    None
                    if covariance_value is None
                    else np.asarray(covariance_value, dtype=float)
                )
                provenance = (
                    "certificate_reciprocal_vector"
                    if certificate_id is None
                    else f"certificate_reciprocal_vector:{certificate_id}"
                )
                parts.append(
                    ImagingReciprocalVectors(
                        request=request_by_id[member_ids[0]],
                        labels=labels,
                        world_reciprocal_vectors_xy_per_m=vectors,
                        name_prefix=prefix,
                        provenance=provenance,
                        covariance_world_per_m2=covariance,
                        certificate_id=(
                            None if certificate_id is None else str(certificate_id)
                        ),
                        covariance_provenance=feature.get(
                            "covariance_provenance"
                        ),
                        calibration_group_name=feature.get(
                            "calibration_group_name"
                        ),
                    )
                )
            if isinstance(layer_metrology, Mapping):
                parts.append(
                    ImagingLayerBoundaries(
                        request=request_by_id[member_ids[0]],
                        metrology=RegisteredLayerMetrology.from_metadata(
                            layer_metrology
                        ),
                        name_prefix=prefix,
                    )
                )
            if isinstance(layer_period_metrology, Mapping):
                parts.append(
                    ImagingLayerPeriods(
                        request=request_by_id[member_ids[0]],
                        metrology=RegisteredLayerPeriodMetrology.from_metadata(
                            layer_period_metrology
                        ),
                        name_prefix=prefix,
                    )
                )
        specifications.append(
            PhysicalObservableBlockSpecification(
                frame_id=block.frame_id,
                optical_state_index=block.optical_state_index,
                parts=tuple(parts),
                provenance="aggregated_checkpoint_physical_observables",
            )
        )
    result = tuple(specifications)
    validate_physical_observable_schema(result, dataset)
    return result


__all__ = (
    "AngularDistributionReciprocalVectors",
    "CentroidDifference",
    "ImagingLayerBoundaries",
    "ImagingLayerPeriods",
    "ImagingReciprocalVectors",
    "IndexedDiffractionBraggSpots",
    "LayerCertificateCalibration",
    "LayerCertificateObservableCovarianceAugmentation",
    "NativeDiffractionReciprocalVectors",
    "PhysicalObservableBlockSpecification",
    "SingularObservableMapError",
    "UnsupportedPhysicalObservableError",
    "calibration_covariance_augmentations_for_physical_observables",
    "calibration_priors_for_physical_observables",
    "fourier_observable_covariance_augmentation",
    "imaging_reciprocal_vectors_from_feature_set",
    "indexed_diffraction_bragg_spots_from_feature_set",
    "layer_certificate_calibrations_for_physical_observables",
    "layer_certificate_covariance_augmentations_for_physical_observables",
    "make_calibration_aware_physical_predictor",
    "predict_physical_observable_blocks",
    "specifications_from_aggregated_dataset",
    "strip_shared_layer_certificate_covariance",
    "strip_shared_multilayer_certificate_covariance",
    "validate_physical_observable_schema",
)
