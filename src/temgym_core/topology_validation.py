"""Fail-closed validation ledger for physical NEOARM forward renders.

The Gaussian quadrature topology is only one part of a rendered-frame
identity.  A validation obtained with the same specimen crop but a different
illumination, downstream transfer map, detector, or renderer cannot certify a
production image.  This module makes those dependencies explicit and records
the numerical evidence used to admit each complete render signature.

The ledger deliberately stores discrepancies in units of the declared
experimental uncertainty.  It does not run either forward model: callers must
produce the Gaussian candidate and an independently implemented grid/CZT or
Collins reference, then bind that comparison to an exact
:class:`PhysicalRenderSignature`.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy import ndimage


_SIGNATURE_SCHEMA = "temgym_core.physical_render_signature"
_LEDGER_SCHEMA = "temgym_core.physical_topology_validation_ledger"
_VERSION = 1


_POSITION_INDICES = np.asarray((0, 2), dtype=int)
_ANGLE_INDICES = np.asarray((1, 3), dtype=int)


def _nonempty(value: object, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


def _finite_nonnegative(value: object, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


def _finite_positive(value: object, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _json_value(value: object, *, name: str) -> object:
    """Convert a value to canonical, finite JSON data."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError(f"{name} contains a non-finite value.")
        return result
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError(f"{name} cannot contain object arrays.")
        return _json_value(value.tolist(), name=name)
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{name} mapping keys must be strings.")
            result[key] = _json_value(item, name=f"{name}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _json_value(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{name} contains unsupported type {type(value).__name__!r}.")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def stable_validation_signature(
    namespace: str,
    *,
    arrays: Mapping[str, np.ndarray] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> str:
    """Return a stable SHA-256 identity for numeric validation state.

    Floating and complex arrays are canonicalized to little-endian float64 or
    complex128, so a semantically identical NumPy/JAX value has the same
    identity on CPU and GPU.  Integer, unsigned, and Boolean arrays retain
    their value category.  Object/string arrays are intentionally rejected;
    put textual values in ``metadata`` instead.
    """

    label = _nonempty(namespace, "namespace")
    array_mapping = {} if arrays is None else dict(arrays)
    metadata_mapping = {} if metadata is None else dict(metadata)
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {
                "schema": "temgym_core.validation_numeric_state",
                "version": _VERSION,
                "namespace": label,
                "metadata": _json_value(metadata_mapping, name="metadata"),
            }
        )
    )
    for name in sorted(array_mapping):
        key = _nonempty(name, "array name")
        raw = np.asarray(array_mapping[name])
        if raw.dtype.hasobject or raw.dtype.kind in {"S", "U", "V"}:
            raise TypeError(f"Array {key!r} must be numeric, not {raw.dtype}.")
        kind = raw.dtype.kind
        if kind == "f":
            canonical = np.ascontiguousarray(raw, dtype="<f8")
        elif kind == "c":
            canonical = np.ascontiguousarray(raw, dtype="<c16")
        elif kind == "i":
            canonical = np.ascontiguousarray(raw, dtype="<i8")
        elif kind == "u":
            canonical = np.ascontiguousarray(raw, dtype="<u8")
        elif kind == "b":
            canonical = np.ascontiguousarray(raw, dtype="u1")
        else:
            raise TypeError(f"Array {key!r} has unsupported dtype {raw.dtype}.")
        if kind in {"f", "c"} and not np.all(np.isfinite(canonical)):
            raise ValueError(f"Array {key!r} contains non-finite values.")
        descriptor = {
            "name": key,
            "kind": kind,
            "shape": list(canonical.shape),
        }
        digest.update(_canonical_json(descriptor))
        digest.update(canonical.tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


def _single_gaussian_state(
    beam: object,
) -> tuple[np.ndarray, np.ndarray, complex, np.ndarray, float]:
    """Return one Gaussian's centre, slope, amplitude, curvature, and wave number.

    This deliberately consumes only the public Gaussian attributes.  The
    analytic reference below does not call the production Gaussian evaluator
    or its transfer-map implementation.
    """

    try:
        vector = beam.to_vector()  # type: ignore[attr-defined]
        centres = np.asarray(vector.r_xy, dtype=float)
        slopes = np.asarray(vector.d_xy, dtype=float)
        amplitudes = np.asarray(vector.amplitude, dtype=np.complex128).reshape(-1)
        curvatures = np.asarray(vector.Q_inv, dtype=np.complex128)
        wave_numbers = np.asarray(vector.k, dtype=float).reshape(-1)
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError(
            "beam must expose the public GaussianBeam state and to_vector()."
        ) from error
    if centres.shape != (1, 2) or slopes.shape != (1, 2):
        raise ValueError("The analytic reference requires exactly one Gaussian.")
    if curvatures.shape == (2, 2):
        curvatures = curvatures[None, ...]
    if (
        curvatures.shape != (1, 2, 2)
        or amplitudes.shape != (1,)
        or wave_numbers.shape != (1,)
    ):
        raise ValueError("The analytic reference requires exactly one Gaussian.")
    if (
        not np.all(np.isfinite(centres))
        or not np.all(np.isfinite(slopes))
        or not np.all(np.isfinite(curvatures))
        or not np.all(np.isfinite(amplitudes))
        or not np.all(np.isfinite(wave_numbers))
        or wave_numbers[0] <= 0.0
    ):
        raise ValueError("The Gaussian state must be finite with positive wave number.")
    curvature = curvatures[0]
    symmetry_scale = max(float(np.linalg.norm(curvature)), np.finfo(float).tiny)
    if np.linalg.norm(curvature - curvature.T) > 1e-10 * symmetry_scale:
        raise ValueError("The Gaussian inverse-curvature matrix must be symmetric.")
    return centres[0], slopes[0], amplitudes[0], curvature, float(wave_numbers[0])


def _gaussian_intensity_covariance(
    curvature_inv: np.ndarray,
    wave_number_per_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return intensity precision and covariance for a complex Gaussian action."""

    precision = wave_number_per_m * np.imag(curvature_inv)
    precision = 0.5 * (precision + precision.T)
    eigenvalues = np.linalg.eigvalsh(precision)
    scale = max(float(np.linalg.norm(precision)), np.finfo(float).tiny)
    if eigenvalues.min() <= 64.0 * np.finfo(float).eps * scale:
        raise ValueError(
            "The Gaussian intensity precision must be positive definite."
        )
    # I(r) is proportional to exp[-(r-r0)^T P (r-r0)], hence Sigma^{-1}=2P.
    covariance = 0.5 * np.linalg.inv(precision)
    return precision, covariance


@dataclass(frozen=True)
class AnalyticGaussianDetectorReference:
    """Closed-form detector intensity for one Gaussian on a finite pixel grid.

    ``intensity_yx`` uses the same physical units as the incident Gaussian
    field squared.  ``sampled_power_fraction`` is a midpoint-quadrature audit
    of the finite detector field of view, before the detector MTF.  It is a
    diagnostic rather than an implicit renormalization.
    """

    intensity_yx: np.ndarray
    detector_centre_column_xy_m: np.ndarray
    detector_intensity_covariance_column_m2: np.ndarray
    detector_intensity_covariance_pixel2: np.ndarray
    input_power: float
    sampled_power_fraction: float
    mtf_sampled_power_fraction: float
    detector_mtf_sigma_px: float

    def __post_init__(self) -> None:
        image = np.asarray(self.intensity_yx, dtype=float).copy()
        centre = np.asarray(self.detector_centre_column_xy_m, dtype=float).copy()
        covariance = np.asarray(
            self.detector_intensity_covariance_column_m2, dtype=float
        ).copy()
        covariance_px = np.asarray(
            self.detector_intensity_covariance_pixel2, dtype=float
        ).copy()
        if (
            image.ndim != 2
            or min(image.shape) < 2
            or not np.all(np.isfinite(image))
            or np.any(image < 0.0)
        ):
            raise ValueError("intensity_yx must be a finite non-negative 2-D image.")
        if centre.shape != (2,) or not np.all(np.isfinite(centre)):
            raise ValueError("detector_centre_column_xy_m must be a finite pair.")
        for name, value in (
            ("detector_intensity_covariance_column_m2", covariance),
            ("detector_intensity_covariance_pixel2", covariance_px),
        ):
            if (
                value.shape != (2, 2)
                or not np.all(np.isfinite(value))
                or not np.allclose(value, value.T, rtol=1e-10, atol=1e-24)
                or np.linalg.eigvalsh(value).min() <= 0.0
            ):
                raise ValueError(f"{name} must be finite, symmetric, and positive definite.")
        for name in (
            "input_power",
            "sampled_power_fraction",
            "mtf_sampled_power_fraction",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        mtf = _finite_nonnegative(self.detector_mtf_sigma_px, "detector_mtf_sigma_px")
        for value in (image, centre, covariance, covariance_px):
            value.setflags(write=False)
        object.__setattr__(self, "intensity_yx", image)
        object.__setattr__(self, "detector_centre_column_xy_m", centre)
        object.__setattr__(
            self, "detector_intensity_covariance_column_m2", covariance
        )
        object.__setattr__(
            self, "detector_intensity_covariance_pixel2", covariance_px
        )
        object.__setattr__(self, "detector_mtf_sigma_px", mtf)


def analytic_single_gaussian_detector_reference(
    beam_at_input: object,
    transfer_matrix: np.ndarray,
    detector: object,
) -> AnalyticGaussianDetectorReference:
    """Independently propagate and sample one Gaussian's detector intensity.

    The production path lifts the optical map into a Gaussian beam object and
    coherently evaluates that object pixel by pixel (normally with a GPU
    kernel).  This reference instead applies the closed-form ABCD curvature
    law in NumPy and evaluates the resulting *intensity quadratic* directly.
    It therefore shares the declared first-order optics and detector MTF, but
    shares neither the lifted propagation routine nor the field evaluator.

    The detector is finite and is never renormalized to the captured field of
    view.  The declared discrete Gaussian MTF is applied after analytic
    sampling, exactly as a detector response rather than as an optical-beam
    operation.
    """

    matrix = np.asarray(transfer_matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("transfer_matrix must be a finite 4x4 array.")
    try:
        optical_coords = np.asarray(detector.optical_coords_xy_m, dtype=float)
        shape_yx = tuple(int(value) for value in detector.shape_yx)
        pixel_yx = np.asarray(detector.pixel_size_yx_m, dtype=float)
        affine = np.asarray(detector.affine_xy, dtype=float)
        mtf_sigma_px = float(detector.mtf_sigma_px)
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError("detector must expose the DetectorGeometry interface.") from error
    if (
        optical_coords.shape != (*shape_yx, 2)
        or len(shape_yx) != 2
        or min(shape_yx) < 2
        or pixel_yx.shape != (2,)
        or np.any(pixel_yx <= 0.0)
        or affine.shape != (2, 2)
        or not np.all(np.isfinite(optical_coords))
        or not np.all(np.isfinite(pixel_yx))
        or not np.all(np.isfinite(affine))
        or not np.isfinite(mtf_sigma_px)
        or mtf_sigma_px < 0.0
    ):
        raise ValueError("detector geometry is invalid.")
    affine_determinant = float(np.linalg.det(affine))
    if abs(affine_determinant) <= np.finfo(float).eps:
        raise ValueError("detector affine must be invertible.")

    centre, slope, amplitude, curvature, wave_number = _single_gaussian_state(
        beam_at_input
    )
    A = matrix[np.ix_(_POSITION_INDICES, _POSITION_INDICES)]
    B = matrix[np.ix_(_POSITION_INDICES, _ANGLE_INDICES)]
    C = matrix[np.ix_(_ANGLE_INDICES, _POSITION_INDICES)]
    D = matrix[np.ix_(_ANGLE_INDICES, _ANGLE_INDICES)]
    denominator = A + B @ curvature
    determinant = np.linalg.det(denominator)
    determinant_scale = max(float(np.linalg.norm(denominator)) ** 2, 1.0)
    if abs(determinant) <= 64.0 * np.finfo(float).eps * determinant_scale:
        raise ValueError("The Gaussian ABCD denominator is singular.")
    output_curvature = (C + D @ curvature) @ np.linalg.inv(denominator)
    output_centre = A @ centre + B @ slope
    precision, covariance = _gaussian_intensity_covariance(
        output_curvature, wave_number
    )
    input_precision, _input_covariance = _gaussian_intensity_covariance(
        curvature, wave_number
    )
    input_power = (
        abs(amplitude) ** 2
        * np.pi
        / np.sqrt(float(np.linalg.det(input_precision)))
    )
    output_peak_intensity = abs(amplitude) ** 2 / abs(determinant)
    delta = optical_coords - output_centre
    exponent = -np.einsum("...i,ij,...j->...", delta, precision, delta)
    raw_intensity = output_peak_intensity * np.exp(exponent)
    optical_pixel_area = float(np.prod(pixel_yx) / abs(affine_determinant))
    sampled_power_fraction = float(np.sum(raw_intensity) * optical_pixel_area / input_power)
    intensity = raw_intensity
    if mtf_sigma_px > 0.0:
        intensity = ndimage.gaussian_filter(
            intensity, mtf_sigma_px, mode="nearest"
        )
    mtf_sampled_power_fraction = float(
        np.sum(intensity) * optical_pixel_area / input_power
    )
    if (
        not np.all(np.isfinite(intensity))
        or np.any(intensity < 0.0)
        or sampled_power_fraction <= 0.0
        or mtf_sampled_power_fraction <= 0.0
    ):
        raise RuntimeError("The analytic Gaussian detector reference is invalid.")

    detector_from_column_pixels = (
        np.diag(1.0 / pixel_yx[::-1]) @ affine
    )
    covariance_px = detector_from_column_pixels @ covariance @ detector_from_column_pixels.T
    if mtf_sigma_px > 0.0:
        covariance_px = covariance_px + mtf_sigma_px**2 * np.eye(2)
    return AnalyticGaussianDetectorReference(
        intensity_yx=intensity,
        detector_centre_column_xy_m=output_centre,
        detector_intensity_covariance_column_m2=covariance,
        detector_intensity_covariance_pixel2=covariance_px,
        input_power=float(input_power),
        sampled_power_fraction=sampled_power_fraction,
        mtf_sampled_power_fraction=mtf_sampled_power_fraction,
        detector_mtf_sigma_px=mtf_sigma_px,
    )


@dataclass(frozen=True)
class GaussianPerforationCertificate:
    """Conservative clipping certificate for one Gaussian in a circular hole."""

    perforation_centre_xy_m: np.ndarray
    beam_centre_xy_m: np.ndarray
    effective_radius_m: float
    centre_offset_m: float
    largest_intensity_sigma_m: float
    radial_clearance_sigma: float
    lost_power_fraction_upper_bound: float
    maximum_lost_power_fraction: float
    passed: bool

    def __post_init__(self) -> None:
        perforation_centre = np.asarray(
            self.perforation_centre_xy_m, dtype=float
        ).copy()
        beam_centre = np.asarray(self.beam_centre_xy_m, dtype=float).copy()
        if (
            perforation_centre.shape != (2,)
            or beam_centre.shape != (2,)
            or not np.all(np.isfinite(perforation_centre))
            or not np.all(np.isfinite(beam_centre))
        ):
            raise ValueError("Perforation and beam centres must be finite pairs.")
        for name in (
            "effective_radius_m",
            "centre_offset_m",
            "largest_intensity_sigma_m",
            "radial_clearance_sigma",
            "lost_power_fraction_upper_bound",
            "maximum_lost_power_fraction",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, value)
        if self.effective_radius_m <= 0.0 or self.largest_intensity_sigma_m <= 0.0:
            raise ValueError("The effective radius and Gaussian sigma must be positive.")
        if self.lost_power_fraction_upper_bound > 1.0:
            raise ValueError("lost_power_fraction_upper_bound cannot exceed one.")
        if self.maximum_lost_power_fraction >= 1.0:
            raise ValueError("maximum_lost_power_fraction must be smaller than one.")
        expected_pass = (
            self.centre_offset_m < self.effective_radius_m
            and self.lost_power_fraction_upper_bound
            <= self.maximum_lost_power_fraction
        )
        if bool(self.passed) != expected_pass:
            raise ValueError("passed is inconsistent with the clipping bound.")
        perforation_centre.setflags(write=False)
        beam_centre.setflags(write=False)
        object.__setattr__(self, "perforation_centre_xy_m", perforation_centre)
        object.__setattr__(self, "beam_centre_xy_m", beam_centre)
        object.__setattr__(self, "passed", bool(self.passed))


def certify_single_gaussian_perforation(
    beam_at_perforation: object,
    *,
    perforation_centre_xy_m: Sequence[float],
    perforation_radius_m: float,
    uncertain_edge_width_m: float = 0.0,
    maximum_lost_power_fraction: float = 1e-10,
) -> GaussianPerforationCertificate:
    """Bound power clipped by a finite circular perforation.

    The edge uncertainty is subtracted from the nominal radius.  For an
    elliptical Gaussian with intensity covariance ``Sigma``, the event that a
    ray lies outside the circle implies ``||Z|| >= clearance/sqrt(lambda_max)``
    for a standard two-dimensional normal ``Z``.  Its chi-square-2 tail is
    ``exp(-clearance_sigma**2/2)``, giving a conservative bound for arbitrary
    ellipse orientation and beam offset.
    """

    centre, _slope, _amplitude, curvature, wave_number = _single_gaussian_state(
        beam_at_perforation
    )
    perforation_centre = np.asarray(perforation_centre_xy_m, dtype=float)
    if perforation_centre.shape != (2,) or not np.all(np.isfinite(perforation_centre)):
        raise ValueError("perforation_centre_xy_m must be a finite pair.")
    radius = _finite_positive(perforation_radius_m, "perforation_radius_m")
    edge = _finite_nonnegative(uncertain_edge_width_m, "uncertain_edge_width_m")
    maximum_loss = _finite_nonnegative(
        maximum_lost_power_fraction, "maximum_lost_power_fraction"
    )
    if edge >= radius:
        raise ValueError("uncertain_edge_width_m must be smaller than the radius.")
    if maximum_loss >= 1.0:
        raise ValueError("maximum_lost_power_fraction must be smaller than one.")
    _precision, covariance = _gaussian_intensity_covariance(curvature, wave_number)
    largest_sigma = float(np.sqrt(np.linalg.eigvalsh(covariance).max()))
    effective_radius = radius - edge
    offset = float(np.linalg.norm(centre - perforation_centre))
    clearance = max(effective_radius - offset, 0.0)
    clearance_sigma = clearance / largest_sigma
    loss_bound = (
        float(np.exp(-0.5 * clearance_sigma**2))
        if offset < effective_radius
        else 1.0
    )
    return GaussianPerforationCertificate(
        perforation_centre_xy_m=perforation_centre,
        beam_centre_xy_m=centre,
        effective_radius_m=effective_radius,
        centre_offset_m=offset,
        largest_intensity_sigma_m=largest_sigma,
        radial_clearance_sigma=clearance_sigma,
        lost_power_fraction_upper_bound=loss_bound,
        maximum_lost_power_fraction=maximum_loss,
        passed=bool(offset < effective_radius and loss_bound <= maximum_loss),
    )


@dataclass(frozen=True, order=True)
class PhysicalRenderSignature:
    """Complete identity of one class of production forward render.

    ``frozen_topology_id`` identifies the specimen crop/interpolation/Gaussian
    topology.  The other signatures prevent that topology from standing in
    for a different source, transfer map, detector, or implementation.
    ``optical_scope`` must distinguish a physical NEOARM column from a declared
    ideal relay; an ideal Fourier camera is not a validation of projector-lens
    diffraction settings.
    """

    acquisition_kind: str
    frozen_topology_id: str
    interaction_route: str
    source_signature: str
    downstream_signature: str
    detector_signature: str
    render_signature: str
    renderer_family: str
    optical_scope: str

    def __post_init__(self) -> None:
        kind = _nonempty(self.acquisition_kind, "acquisition_kind").lower()
        if kind not in {"imaging", "diffraction"}:
            raise ValueError("acquisition_kind must be 'imaging' or 'diffraction'.")
        object.__setattr__(self, "acquisition_kind", kind)
        for name in (
            "frozen_topology_id",
            "interaction_route",
            "source_signature",
            "downstream_signature",
            "detector_signature",
            "render_signature",
            "renderer_family",
            "optical_scope",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": _SIGNATURE_SCHEMA,
            "version": _VERSION,
            "acquisition_kind": self.acquisition_kind,
            "frozen_topology_id": self.frozen_topology_id,
            "interaction_route": self.interaction_route,
            "source_signature": self.source_signature,
            "downstream_signature": self.downstream_signature,
            "detector_signature": self.detector_signature,
            "render_signature": self.render_signature,
            "renderer_family": self.renderer_family,
            "optical_scope": self.optical_scope,
        }

    @property
    def digest(self) -> str:
        return "sha256:" + hashlib.sha256(_canonical_json(self.to_payload())).hexdigest()


@dataclass(frozen=True)
class PhysicalValidationThresholds:
    """Noise-relative admission limits from the NEOARM calibration plan."""

    sample_field_rms_in_noise_sigma: float = 0.2
    detector_rms_in_noise_sigma: float = 0.2
    observable_residual_in_sigma: float = 0.2
    parameter_change_in_sigma: float = 0.1

    def __post_init__(self) -> None:
        for name in (
            "sample_field_rms_in_noise_sigma",
            "detector_rms_in_noise_sigma",
            "observable_residual_in_sigma",
            "parameter_change_in_sigma",
        ):
            object.__setattr__(self, name, _finite_positive(getattr(self, name), name))

    def to_payload(self) -> dict[str, float]:
        return {
            "sample_field_rms_in_noise_sigma": self.sample_field_rms_in_noise_sigma,
            "detector_rms_in_noise_sigma": self.detector_rms_in_noise_sigma,
            "observable_residual_in_sigma": self.observable_residual_in_sigma,
            "parameter_change_in_sigma": self.parameter_change_in_sigma,
        }


@dataclass(frozen=True)
class PhysicalValidationRequirements:
    """Evidence required to admit one exact physical render signature."""

    required_observables: tuple[str, ...] = (
        "scale",
        "rotation",
        "centroid",
        "fourier_vector",
        "intensity",
    )
    require_sample_field: bool = True
    require_parameter_stability: bool = True
    require_independent_reference: bool = True
    allowed_optical_scopes: tuple[str, ...] = ()
    thresholds: PhysicalValidationThresholds = PhysicalValidationThresholds()

    def __post_init__(self) -> None:
        observables = tuple(
            dict.fromkeys(_nonempty(value, "required observable") for value in self.required_observables)
        )
        scopes = tuple(
            dict.fromkeys(_nonempty(value, "allowed optical scope") for value in self.allowed_optical_scopes)
        )
        object.__setattr__(self, "required_observables", observables)
        object.__setattr__(self, "allowed_optical_scopes", scopes)
        if not isinstance(self.thresholds, PhysicalValidationThresholds):
            raise TypeError("thresholds must be PhysicalValidationThresholds.")

    def to_payload(self) -> dict[str, object]:
        return {
            "required_observables": list(self.required_observables),
            "require_sample_field": bool(self.require_sample_field),
            "require_parameter_stability": bool(self.require_parameter_stability),
            "require_independent_reference": bool(self.require_independent_reference),
            "allowed_optical_scopes": list(self.allowed_optical_scopes),
            "thresholds": self.thresholds.to_payload(),
        }


@dataclass(frozen=True)
class PhysicalValidationEvidence:
    """Independent-reference discrepancies for one production signature."""

    signature: PhysicalRenderSignature
    reference_method: str
    reference_signature: str
    reference_renderer_family: str
    reference_optical_scope: str
    detector_rms_in_noise_sigma: float
    observable_residuals_in_sigma: Mapping[str, float] | Sequence[tuple[str, float]]
    sample_field_rms_in_noise_sigma: float | None = None
    parameter_change_in_sigma: float | None = None
    provenance: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.signature, PhysicalRenderSignature):
            raise TypeError("signature must be a PhysicalRenderSignature.")
        for name in (
            "reference_method",
            "reference_signature",
            "reference_renderer_family",
            "reference_optical_scope",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "detector_rms_in_noise_sigma",
            _finite_nonnegative(
                self.detector_rms_in_noise_sigma,
                "detector_rms_in_noise_sigma",
            ),
        )
        if self.sample_field_rms_in_noise_sigma is not None:
            object.__setattr__(
                self,
                "sample_field_rms_in_noise_sigma",
                _finite_nonnegative(
                    self.sample_field_rms_in_noise_sigma,
                    "sample_field_rms_in_noise_sigma",
                ),
            )
        if self.parameter_change_in_sigma is not None:
            object.__setattr__(
                self,
                "parameter_change_in_sigma",
                _finite_nonnegative(
                    abs(float(self.parameter_change_in_sigma)),
                    "parameter_change_in_sigma",
                ),
            )
        raw_observables = (
            self.observable_residuals_in_sigma.items()
            if isinstance(self.observable_residuals_in_sigma, Mapping)
            else self.observable_residuals_in_sigma
        )
        observables: dict[str, float] = {}
        for name, value in raw_observables:
            key = _nonempty(name, "observable name")
            if key in observables:
                raise ValueError(f"Duplicate observable evidence for {key!r}.")
            observables[key] = _finite_nonnegative(
                abs(float(value)), f"observable {key!r} residual"
            )
        object.__setattr__(
            self,
            "observable_residuals_in_sigma",
            tuple(sorted(observables.items())),
        )
        provenance = {} if self.provenance is None else dict(self.provenance)
        object.__setattr__(
            self,
            "provenance",
            _json_value(provenance, name="provenance"),
        )

    @property
    def observable_residual_map(self) -> dict[str, float]:
        return dict(self.observable_residuals_in_sigma)

    @property
    def independently_implemented(self) -> bool:
        return (
            self.reference_renderer_family != self.signature.renderer_family
            and self.reference_signature != self.signature.render_signature
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "signature_digest": self.signature.digest,
            "reference_method": self.reference_method,
            "reference_signature": self.reference_signature,
            "reference_renderer_family": self.reference_renderer_family,
            "reference_optical_scope": self.reference_optical_scope,
            "detector_rms_in_noise_sigma": self.detector_rms_in_noise_sigma,
            "observable_residuals_in_sigma": self.observable_residual_map,
            "sample_field_rms_in_noise_sigma": self.sample_field_rms_in_noise_sigma,
            "parameter_change_in_sigma": self.parameter_change_in_sigma,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class PhysicalValidationDecision:
    signature_digest: str
    evidence_index: int
    failure_reasons: tuple[str, ...]
    passed: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "signature_digest": self.signature_digest,
            "evidence_index": int(self.evidence_index),
            "failure_reasons": list(self.failure_reasons),
            "passed": bool(self.passed),
        }


def evaluate_physical_validation(
    evidence: PhysicalValidationEvidence,
    requirements: PhysicalValidationRequirements = PhysicalValidationRequirements(),
    *,
    evidence_index: int = 0,
) -> PhysicalValidationDecision:
    """Evaluate one record without allowing absent evidence to pass silently."""

    if not isinstance(evidence, PhysicalValidationEvidence):
        raise TypeError("evidence must be PhysicalValidationEvidence.")
    if not isinstance(requirements, PhysicalValidationRequirements):
        raise TypeError("requirements must be PhysicalValidationRequirements.")
    threshold = requirements.thresholds
    residuals = evidence.observable_residual_map
    failures: list[str] = []
    if (
        requirements.allowed_optical_scopes
        and evidence.signature.optical_scope not in requirements.allowed_optical_scopes
    ):
        failures.append("candidate_optical_scope")
    if evidence.reference_optical_scope != evidence.signature.optical_scope:
        failures.append("reference_optical_scope")
    if requirements.require_independent_reference and not evidence.independently_implemented:
        failures.append("reference_independence")
    if evidence.detector_rms_in_noise_sigma > threshold.detector_rms_in_noise_sigma:
        failures.append("detector_rms")
    if evidence.sample_field_rms_in_noise_sigma is None:
        if requirements.require_sample_field:
            failures.append("missing:sample_field")
    elif (
        evidence.sample_field_rms_in_noise_sigma
        > threshold.sample_field_rms_in_noise_sigma
    ):
        failures.append("sample_field_rms")
    for name in requirements.required_observables:
        if name not in residuals:
            failures.append(f"missing:observable:{name}")
        elif residuals[name] > threshold.observable_residual_in_sigma:
            failures.append(f"observable:{name}")
    if evidence.parameter_change_in_sigma is None:
        if requirements.require_parameter_stability:
            failures.append("missing:parameter_stability")
    elif evidence.parameter_change_in_sigma > threshold.parameter_change_in_sigma:
        failures.append("parameter_stability")
    failures = list(dict.fromkeys(failures))
    return PhysicalValidationDecision(
        signature_digest=evidence.signature.digest,
        evidence_index=int(evidence_index),
        failure_reasons=tuple(failures),
        passed=not failures,
    )


@dataclass(frozen=True)
class PhysicalValidationCoverage:
    required_count: int
    accepted_count: int
    failed_count: int
    missing_count: int
    accepted_signature_digests: tuple[str, ...]
    failed_signature_digests: tuple[str, ...]
    missing_signature_digests: tuple[str, ...]
    complete: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "required_count": self.required_count,
            "accepted_count": self.accepted_count,
            "failed_count": self.failed_count,
            "missing_count": self.missing_count,
            "accepted_signature_digests": list(self.accepted_signature_digests),
            "failed_signature_digests": list(self.failed_signature_digests),
            "missing_signature_digests": list(self.missing_signature_digests),
            "complete": self.complete,
        }


@dataclass(frozen=True)
class FisherParameterStabilityReport:
    """Aggregate forward-model error expressed in identifiable-fit sigma units.

    The per-render ledger deliberately tests image, field, and observable
    discrepancies.  Parameter stability is a property of the *joint inverse*,
    however, and assigning the same fitted-parameter change to every frame
    would be fictitious evidence.  This report instead projects one complete
    whitened candidate-minus-reference discrepancy vector through the active
    likelihood Jacobian.

    ``fisher_mahalanobis_change_in_sigma`` is the norm of the fitted active
    displacement in the absolute-noise Fisher metric.  It bounds the change of
    every unit-variance identifiable linear combination, independently of the
    arbitrary active-coordinate basis.  The aggregate gate is deliberately
    strict (``<`` rather than ``<=``).  A separate trailing-history gate
    records actual successive-topology parameter changes, so an independent
    reference comparison cannot be mislabeled as two refinement transitions.
    """

    observation_count: int
    parameter_count: int
    active_rank: int
    active_design_rank: int
    whitened_discrepancy_norm: float
    projected_whitened_discrepancy_norm: float
    unprojected_whitened_discrepancy_norm: float
    active_parameter_shift: np.ndarray
    active_standard_errors: np.ndarray
    active_coordinate_change_in_sigma: np.ndarray
    maximum_active_coordinate_change_in_sigma: float
    fisher_mahalanobis_change_in_sigma: float
    successive_refinement_changes_in_sigma: np.ndarray
    required_successive_refinements: int
    parameter_change_threshold_sigma: float
    active_design_full_rank: bool
    aggregate_projection_passed: bool
    successive_refinements_passed: bool
    passed: bool

    def __post_init__(self) -> None:
        observations = int(self.observation_count)
        parameters = int(self.parameter_count)
        active = int(self.active_rank)
        design_rank = int(self.active_design_rank)
        required = int(self.required_successive_refinements)
        if observations < 1 or parameters < 1 or active < 1:
            raise ValueError("Observation, parameter, and active counts must be positive.")
        if not 0 <= design_rank <= active <= parameters:
            raise ValueError("Active/design ranks are inconsistent with parameter_count.")
        if required < 1:
            raise ValueError("required_successive_refinements must be positive.")
        threshold = _finite_positive(
            self.parameter_change_threshold_sigma,
            "parameter_change_threshold_sigma",
        )
        scalar_names = (
            "whitened_discrepancy_norm",
            "projected_whitened_discrepancy_norm",
            "unprojected_whitened_discrepancy_norm",
            "maximum_active_coordinate_change_in_sigma",
            "fisher_mahalanobis_change_in_sigma",
        )
        scalars = {
            name: _finite_nonnegative(getattr(self, name), name)
            for name in scalar_names
        }
        arrays = {}
        for name in (
            "active_parameter_shift",
            "active_standard_errors",
            "active_coordinate_change_in_sigma",
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (active,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite vector of length active_rank.")
            value = value.copy()
            value.setflags(write=False)
            arrays[name] = value
        if np.any(arrays["active_standard_errors"] < 0.0):
            raise ValueError("active_standard_errors must be non-negative.")
        history = np.asarray(self.successive_refinement_changes_in_sigma, dtype=float)
        if history.ndim != 1 or np.any(history < 0.0) or not np.all(np.isfinite(history)):
            raise ValueError(
                "successive_refinement_changes_in_sigma must be a finite "
                "non-negative vector."
            )
        history = history.copy()
        history.setflags(write=False)
        full_rank = design_rank == active
        if bool(self.active_design_full_rank) != full_rank:
            raise ValueError("active_design_full_rank is inconsistent with the ranks.")
        aggregate_passed = bool(full_rank and scalars["fisher_mahalanobis_change_in_sigma"] < threshold)
        history_passed = bool(
            history.size >= required
            and np.all(history[-required:] < threshold)
        )
        if bool(self.aggregate_projection_passed) != aggregate_passed:
            raise ValueError("aggregate_projection_passed is inconsistent with the evidence.")
        if bool(self.successive_refinements_passed) != history_passed:
            raise ValueError("successive_refinements_passed is inconsistent with the evidence.")
        if bool(self.passed) != (aggregate_passed and history_passed):
            raise ValueError("passed is inconsistent with the component gates.")
        object.__setattr__(self, "observation_count", observations)
        object.__setattr__(self, "parameter_count", parameters)
        object.__setattr__(self, "active_rank", active)
        object.__setattr__(self, "active_design_rank", design_rank)
        object.__setattr__(self, "required_successive_refinements", required)
        object.__setattr__(self, "parameter_change_threshold_sigma", threshold)
        for name, value in scalars.items():
            object.__setattr__(self, name, value)
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "successive_refinement_changes_in_sigma", history)
        object.__setattr__(self, "active_design_full_rank", full_rank)
        object.__setattr__(self, "aggregate_projection_passed", aggregate_passed)
        object.__setattr__(self, "successive_refinements_passed", history_passed)
        object.__setattr__(self, "passed", aggregate_passed and history_passed)

    def to_payload(self) -> dict[str, object]:
        return {
            "observation_count": self.observation_count,
            "parameter_count": self.parameter_count,
            "active_rank": self.active_rank,
            "active_design_rank": self.active_design_rank,
            "active_design_full_rank": self.active_design_full_rank,
            "whitened_discrepancy_norm": self.whitened_discrepancy_norm,
            "projected_whitened_discrepancy_norm": (
                self.projected_whitened_discrepancy_norm
            ),
            "unprojected_whitened_discrepancy_norm": (
                self.unprojected_whitened_discrepancy_norm
            ),
            "active_parameter_shift": self.active_parameter_shift.tolist(),
            "active_standard_errors": self.active_standard_errors.tolist(),
            "active_coordinate_change_in_sigma": (
                self.active_coordinate_change_in_sigma.tolist()
            ),
            "maximum_active_coordinate_change_in_sigma": (
                self.maximum_active_coordinate_change_in_sigma
            ),
            "fisher_mahalanobis_change_in_sigma": (
                self.fisher_mahalanobis_change_in_sigma
            ),
            "successive_refinement_changes_in_sigma": (
                self.successive_refinement_changes_in_sigma.tolist()
            ),
            "required_successive_refinements": self.required_successive_refinements,
            "parameter_change_threshold_sigma": self.parameter_change_threshold_sigma,
            "aggregate_projection_passed": self.aggregate_projection_passed,
            "successive_refinements_passed": self.successive_refinements_passed,
            "passed": self.passed,
        }


def assess_fisher_parameter_stability(
    whitened_jacobian: np.ndarray,
    whitened_candidate_minus_reference: np.ndarray,
    parameter_scales: np.ndarray,
    active_basis_scaled: np.ndarray,
    *,
    successive_refinement_changes_in_sigma: Sequence[float] = (),
    required_successive_refinements: int = 2,
    parameter_change_threshold_sigma: float = 0.1,
    rank_rtol: float | None = None,
) -> FisherParameterStabilityReport:
    """Project aggregate renderer error through the active Fisher inverse.

    ``whitened_jacobian`` must be the Jacobian of the exact likelihood used to
    define ``active_basis_scaled``.  Its rows and the signed discrepancy must
    have identical ordering and covariance whitening.  A scaled active
    displacement ``a`` corresponds to a physical displacement
    ``parameter_scales * (active_basis_scaled @ a)``.

    The least-squares shift is the first-order change that would result from
    replacing the Gaussian-candidate observables by the independent-reference
    observables.  The Fisher-Mahalanobis norm is used for admission because it
    bounds *all* one-standard-deviation identifiable combinations, not merely
    the coordinates of one arbitrary orthonormal basis.  Two real refinement
    transitions are still required separately.
    """

    jacobian = np.asarray(whitened_jacobian, dtype=float)
    discrepancy = np.asarray(whitened_candidate_minus_reference, dtype=float)
    scales = np.asarray(parameter_scales, dtype=float)
    basis = np.asarray(active_basis_scaled, dtype=float)
    if jacobian.ndim != 2 or min(jacobian.shape) < 1 or not np.all(np.isfinite(jacobian)):
        raise ValueError("whitened_jacobian must be a finite non-empty matrix.")
    if discrepancy.shape != (jacobian.shape[0],) or not np.all(np.isfinite(discrepancy)):
        raise ValueError(
            "whitened_candidate_minus_reference must be finite and match "
            "the Jacobian rows."
        )
    if scales.shape != (jacobian.shape[1],) or not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("parameter_scales must be finite, positive, and match the columns.")
    if (
        basis.ndim != 2
        or basis.shape[0] != jacobian.shape[1]
        or basis.shape[1] < 1
        or not np.all(np.isfinite(basis))
    ):
        raise ValueError(
            "active_basis_scaled must be finite with one row per parameter "
            "and at least one active column."
        )
    gram = basis.T @ basis
    if not np.allclose(gram, np.eye(basis.shape[1]), rtol=1e-10, atol=1e-12):
        raise ValueError("active_basis_scaled columns must be orthonormal.")
    if isinstance(required_successive_refinements, (bool, np.bool_)):
        raise ValueError("required_successive_refinements must be a positive integer.")
    required = int(required_successive_refinements)
    if required != required_successive_refinements or required < 1:
        raise ValueError("required_successive_refinements must be a positive integer.")
    threshold = _finite_positive(
        parameter_change_threshold_sigma,
        "parameter_change_threshold_sigma",
    )

    active_design = jacobian @ (scales[:, None] * basis)
    left, singular, right_t = np.linalg.svd(active_design, full_matrices=False)
    if singular.size == 0:
        raise ValueError("The active design must contain at least one singular value.")
    if rank_rtol is None:
        singular_threshold = (
            np.finfo(float).eps * max(active_design.shape) * float(singular[0])
        )
    else:
        relative = float(rank_rtol)
        if not np.isfinite(relative) or not 0.0 < relative < 1.0:
            raise ValueError("rank_rtol must lie strictly in (0, 1).")
        singular_threshold = relative * float(singular[0])
    design_rank = int(np.sum(singular > singular_threshold))
    inverse_singular = np.zeros_like(singular)
    inverse_singular[:design_rank] = 1.0 / singular[:design_rank]
    shift = -(right_t.T * inverse_singular) @ (left.T @ discrepancy)
    covariance = (right_t.T * inverse_singular**2) @ right_t
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    coordinate_change = np.divide(
        np.abs(shift),
        standard_errors,
        out=np.zeros_like(shift),
        where=standard_errors > 0.0,
    )
    projected = active_design @ shift
    unprojected = discrepancy + projected
    fisher_change = float(np.linalg.norm(projected))
    maximum_coordinate = float(np.max(coordinate_change))
    history = np.asarray(successive_refinement_changes_in_sigma, dtype=float)
    if history.ndim != 1 or not np.all(np.isfinite(history)) or np.any(history < 0.0):
        raise ValueError(
            "successive_refinement_changes_in_sigma must be a finite "
            "non-negative vector."
        )
    history_passed = bool(
        history.size >= required
        and np.all(history[-required:] < threshold)
    )
    full_rank = design_rank == basis.shape[1]
    aggregate_passed = bool(full_rank and fisher_change < threshold)
    return FisherParameterStabilityReport(
        observation_count=jacobian.shape[0],
        parameter_count=jacobian.shape[1],
        active_rank=basis.shape[1],
        active_design_rank=design_rank,
        whitened_discrepancy_norm=float(np.linalg.norm(discrepancy)),
        projected_whitened_discrepancy_norm=fisher_change,
        unprojected_whitened_discrepancy_norm=float(np.linalg.norm(unprojected)),
        active_parameter_shift=shift,
        active_standard_errors=standard_errors,
        active_coordinate_change_in_sigma=coordinate_change,
        maximum_active_coordinate_change_in_sigma=maximum_coordinate,
        fisher_mahalanobis_change_in_sigma=fisher_change,
        successive_refinement_changes_in_sigma=history,
        required_successive_refinements=required,
        parameter_change_threshold_sigma=threshold,
        active_design_full_rank=full_rank,
        aggregate_projection_passed=aggregate_passed,
        successive_refinements_passed=history_passed,
        passed=aggregate_passed and history_passed,
    )


@dataclass(frozen=True)
class FisherBiasBlockInfluence:
    """Additive influence of one signed renderer-bias block.

    The Fisher information and nuisance projector are held fixed.  Therefore
    ``leave_block_fisher_change_in_sigma`` answers the targeted numerical
    question "what would the optical displacement be if this block's signed
    forward-model bias vanished?" without pretending that the acquisition or
    its information was removed from the experiment.
    """

    name: str
    nonzero_row_count: int
    whitened_discrepancy_norm: float
    nuisance_profiled_discrepancy_norm: float
    projected_fisher_change_in_sigma: float
    signed_fisher_squared_contribution: float
    signed_fisher_squared_fraction: float
    leave_block_fisher_change_in_sigma: float
    signed_fisher_mode_changes_in_sigma: np.ndarray

    def __post_init__(self) -> None:
        name = _nonempty(self.name, "name")
        count = int(self.nonzero_row_count)
        if count < 0:
            raise ValueError("nonzero_row_count must be non-negative.")
        nonnegative = {
            key: _finite_nonnegative(getattr(self, key), key)
            for key in (
                "whitened_discrepancy_norm",
                "nuisance_profiled_discrepancy_norm",
                "projected_fisher_change_in_sigma",
                "leave_block_fisher_change_in_sigma",
            )
        }
        signed_contribution = float(self.signed_fisher_squared_contribution)
        signed_fraction = float(self.signed_fisher_squared_fraction)
        if not np.isfinite(signed_contribution) or not np.isfinite(signed_fraction):
            raise ValueError("Signed Fisher contributions must be finite.")
        modes = np.asarray(self.signed_fisher_mode_changes_in_sigma, dtype=float)
        if modes.ndim != 1 or not np.all(np.isfinite(modes)):
            raise ValueError(
                "signed_fisher_mode_changes_in_sigma must be a finite vector."
            )
        modes = modes.copy()
        modes.setflags(write=False)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "nonzero_row_count", count)
        for key, value in nonnegative.items():
            object.__setattr__(self, key, value)
        object.__setattr__(
            self, "signed_fisher_squared_contribution", signed_contribution
        )
        object.__setattr__(self, "signed_fisher_squared_fraction", signed_fraction)
        object.__setattr__(self, "signed_fisher_mode_changes_in_sigma", modes)

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "nonzero_row_count": self.nonzero_row_count,
            "whitened_discrepancy_norm": self.whitened_discrepancy_norm,
            "nuisance_profiled_discrepancy_norm": (
                self.nuisance_profiled_discrepancy_norm
            ),
            "projected_fisher_change_in_sigma": (
                self.projected_fisher_change_in_sigma
            ),
            "signed_fisher_squared_contribution": (
                self.signed_fisher_squared_contribution
            ),
            "signed_fisher_squared_fraction": (
                self.signed_fisher_squared_fraction
            ),
            "leave_block_fisher_change_in_sigma": (
                self.leave_block_fisher_change_in_sigma
            ),
            "signed_fisher_mode_changes_in_sigma": (
                self.signed_fisher_mode_changes_in_sigma.tolist()
            ),
        }


@dataclass(frozen=True)
class NuisanceProfiledOpticalStabilityReport:
    """Fisher stability of the identifiable optical quotient only.

    Numerical nuisance directions are projected out before the optical
    displacement is evaluated.  This is the relevant acceptance quantity for
    recovered microscope parameters; the corresponding joint optical-plus-
    nuisance Fisher norm remains useful as a deliberately non-gating
    diagnostic.
    """

    stability: FisherParameterStabilityReport
    optical_parameter_names: tuple[str, ...]
    nuisance_parameter_count: int
    nuisance_rank: int
    nuisance_singular_values: np.ndarray
    optical_fisher_singular_values: np.ndarray
    signed_optical_fisher_mode_changes_in_sigma: np.ndarray
    optical_fisher_mode_directions_scaled: np.ndarray
    maximum_optical_fisher_mode_change_in_sigma: float
    block_influences: tuple[FisherBiasBlockInfluence, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.stability, FisherParameterStabilityReport):
            raise TypeError("stability must be a FisherParameterStabilityReport.")
        names = tuple(
            _nonempty(value, "optical_parameter_names")
            for value in self.optical_parameter_names
        )
        if len(names) != self.stability.parameter_count or len(set(names)) != len(names):
            raise ValueError(
                "optical_parameter_names must be unique and match parameter_count."
            )
        nuisance_count = int(self.nuisance_parameter_count)
        nuisance_rank = int(self.nuisance_rank)
        if not 0 <= nuisance_rank <= nuisance_count:
            raise ValueError("nuisance_rank must lie within nuisance_parameter_count.")
        nuisance_singular = np.asarray(self.nuisance_singular_values, dtype=float)
        optical_singular = np.asarray(self.optical_fisher_singular_values, dtype=float)
        signed_modes = np.asarray(
            self.signed_optical_fisher_mode_changes_in_sigma, dtype=float
        )
        directions = np.asarray(
            self.optical_fisher_mode_directions_scaled, dtype=float
        )
        active = self.stability.active_rank
        if (
            nuisance_singular.ndim != 1
            or not np.all(np.isfinite(nuisance_singular))
            or np.any(nuisance_singular < 0.0)
        ):
            raise ValueError("nuisance_singular_values must be finite and non-negative.")
        if (
            optical_singular.shape != (active,)
            or not np.all(np.isfinite(optical_singular))
            or np.any(optical_singular <= 0.0)
        ):
            raise ValueError(
                "optical_fisher_singular_values must be a positive active-rank vector."
            )
        if signed_modes.shape != (active,) or not np.all(np.isfinite(signed_modes)):
            raise ValueError(
                "signed_optical_fisher_mode_changes_in_sigma must match active_rank."
            )
        if (
            directions.shape != (self.stability.parameter_count, active)
            or not np.all(np.isfinite(directions))
            or not np.allclose(
                directions.T @ directions,
                np.eye(active),
                rtol=1e-10,
                atol=1e-12,
            )
        ):
            raise ValueError(
                "optical_fisher_mode_directions_scaled must contain orthonormal "
                "scaled optical directions."
            )
        maximum_mode = _finite_nonnegative(
            self.maximum_optical_fisher_mode_change_in_sigma,
            "maximum_optical_fisher_mode_change_in_sigma",
        )
        expected_maximum = float(np.max(np.abs(signed_modes)))
        if not np.isclose(maximum_mode, expected_maximum, rtol=1e-12, atol=1e-14):
            raise ValueError("maximum optical Fisher-mode change is inconsistent.")
        if not np.isclose(
            np.linalg.norm(signed_modes),
            self.stability.fisher_mahalanobis_change_in_sigma,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("Fisher-mode changes do not reproduce the optical norm.")
        blocks = tuple(self.block_influences)
        if not all(isinstance(value, FisherBiasBlockInfluence) for value in blocks):
            raise TypeError("block_influences must contain FisherBiasBlockInfluence values.")
        if any(
            value.signed_fisher_mode_changes_in_sigma.shape != (active,)
            for value in blocks
        ):
            raise ValueError("Every block influence must match active_rank.")
        if blocks:
            summed_modes = np.sum(
                [value.signed_fisher_mode_changes_in_sigma for value in blocks],
                axis=0,
            )
            if not np.allclose(summed_modes, signed_modes, rtol=1e-9, atol=1e-11):
                raise ValueError("Block Fisher-mode influences do not sum to the total.")
        for array in (nuisance_singular, optical_singular, signed_modes, directions):
            array.setflags(write=False)
        object.__setattr__(self, "optical_parameter_names", names)
        object.__setattr__(self, "nuisance_parameter_count", nuisance_count)
        object.__setattr__(self, "nuisance_rank", nuisance_rank)
        object.__setattr__(self, "nuisance_singular_values", nuisance_singular)
        object.__setattr__(self, "optical_fisher_singular_values", optical_singular)
        object.__setattr__(
            self, "signed_optical_fisher_mode_changes_in_sigma", signed_modes
        )
        object.__setattr__(
            self, "optical_fisher_mode_directions_scaled", directions
        )
        object.__setattr__(
            self, "maximum_optical_fisher_mode_change_in_sigma", maximum_mode
        )
        object.__setattr__(self, "block_influences", blocks)

    @property
    def passed(self) -> bool:
        return self.stability.passed

    def to_payload(self) -> dict[str, object]:
        return {
            "scope": "nuisance_profiled_identifiable_optical_quotient",
            "stability": self.stability.to_payload(),
            "optical_parameter_names": list(self.optical_parameter_names),
            "nuisance_parameter_count": self.nuisance_parameter_count,
            "nuisance_rank": self.nuisance_rank,
            "nuisance_singular_values": self.nuisance_singular_values.tolist(),
            "optical_fisher_singular_values": (
                self.optical_fisher_singular_values.tolist()
            ),
            "signed_optical_fisher_mode_changes_in_sigma": (
                self.signed_optical_fisher_mode_changes_in_sigma.tolist()
            ),
            "maximum_optical_fisher_mode_change_in_sigma": (
                self.maximum_optical_fisher_mode_change_in_sigma
            ),
            "optical_fisher_mode_directions_scaled": (
                self.optical_fisher_mode_directions_scaled.tolist()
            ),
            "block_influences": [
                value.to_payload() for value in self.block_influences
            ],
            "passed": self.passed,
        }


def assess_nuisance_profiled_optical_stability(
    whitened_jacobian: np.ndarray,
    whitened_candidate_minus_reference: np.ndarray,
    parameter_scales: np.ndarray,
    *,
    optical_parameter_count: int,
    optical_active_basis_scaled: np.ndarray,
    nuisance_rank: int,
    optical_parameter_names: Sequence[str] = (),
    bias_blocks: Sequence[tuple[str, np.ndarray]] = (),
    successive_refinement_changes_in_sigma: Sequence[float] = (),
    required_successive_refinements: int = 2,
    parameter_change_threshold_sigma: float = 0.1,
) -> NuisanceProfiledOpticalStabilityReport:
    """Profile numerical nuisance response before testing optical stability.

    ``nuisance_rank`` must be the rank selected by the same nuisance-profiled
    design that supplied ``optical_active_basis_scaled``.  Optional
    ``bias_blocks`` are an exact additive decomposition of the full signed
    discrepancy.  Their influence diagnostics hold the experiment and Fisher
    information fixed, isolating which forward-model errors are worth
    refining.
    """

    jacobian = np.asarray(whitened_jacobian, dtype=float)
    discrepancy = np.asarray(whitened_candidate_minus_reference, dtype=float)
    scales = np.asarray(parameter_scales, dtype=float)
    if jacobian.ndim != 2 or min(jacobian.shape) < 1 or not np.all(np.isfinite(jacobian)):
        raise ValueError("whitened_jacobian must be a finite non-empty matrix.")
    if discrepancy.shape != (jacobian.shape[0],) or not np.all(np.isfinite(discrepancy)):
        raise ValueError(
            "whitened_candidate_minus_reference must match the Jacobian rows."
        )
    if scales.shape != (jacobian.shape[1],) or not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("parameter_scales must be finite, positive, and match the columns.")
    if isinstance(optical_parameter_count, (bool, np.bool_)):
        raise ValueError("optical_parameter_count must be a positive integer.")
    optical_count = int(optical_parameter_count)
    if optical_count != optical_parameter_count or not 0 < optical_count <= jacobian.shape[1]:
        raise ValueError("optical_parameter_count must be a valid split point.")
    nuisance_count = jacobian.shape[1] - optical_count
    if isinstance(nuisance_rank, (bool, np.bool_)):
        raise ValueError("nuisance_rank must be a non-negative integer.")
    selected_nuisance_rank = int(nuisance_rank)
    if selected_nuisance_rank != nuisance_rank or not 0 <= selected_nuisance_rank <= nuisance_count:
        raise ValueError("nuisance_rank must lie within the nuisance parameter count.")
    basis = np.asarray(optical_active_basis_scaled, dtype=float)
    if (
        basis.ndim != 2
        or basis.shape[0] != optical_count
        or basis.shape[1] < 1
        or not np.all(np.isfinite(basis))
        or not np.allclose(
            basis.T @ basis,
            np.eye(basis.shape[1]),
            rtol=1e-10,
            atol=1e-12,
        )
    ):
        raise ValueError(
            "optical_active_basis_scaled must contain orthonormal optical columns."
        )
    names = tuple(optical_parameter_names) or tuple(
        f"optical_parameter_{index}" for index in range(optical_count)
    )
    if len(names) != optical_count or len(set(names)) != optical_count:
        raise ValueError("optical_parameter_names must be unique and match the split.")

    scaled = jacobian * scales[None, :]
    optical = scaled[:, :optical_count]
    nuisance = scaled[:, optical_count:]
    if nuisance_count:
        nuisance_left, nuisance_singular, _ = np.linalg.svd(
            nuisance, full_matrices=False
        )
        if selected_nuisance_rank > nuisance_singular.size:
            raise ValueError("nuisance_rank exceeds the numerical nuisance design rank.")
        nuisance_left = nuisance_left[:, :selected_nuisance_rank]
    else:
        nuisance_singular = np.empty(0, dtype=float)
        nuisance_left = np.empty((jacobian.shape[0], 0), dtype=float)

    def profile(values: np.ndarray) -> np.ndarray:
        return values - nuisance_left @ (nuisance_left.T @ values)

    profiled_optical = profile(optical)
    profiled_discrepancy = profile(discrepancy)
    stability = assess_fisher_parameter_stability(
        profiled_optical,
        profiled_discrepancy,
        np.ones(optical_count, dtype=float),
        basis,
        successive_refinement_changes_in_sigma=(
            successive_refinement_changes_in_sigma
        ),
        required_successive_refinements=required_successive_refinements,
        parameter_change_threshold_sigma=parameter_change_threshold_sigma,
    )
    active_design = profiled_optical @ basis
    left, singular, right_t = np.linalg.svd(active_design, full_matrices=False)
    active = basis.shape[1]
    if singular.shape != (active,) or right_t.shape != (active, active):
        raise ValueError("The profiled optical active design has insufficient rows.")
    signed_modes = -(left.T @ profiled_discrepancy)
    directions = basis @ right_t.T
    if not np.isclose(
        np.linalg.norm(signed_modes),
        stability.fisher_mahalanobis_change_in_sigma,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise RuntimeError("Profiled optical Fisher modes disagree with the stability report.")

    blocks = tuple(bias_blocks)
    influences: list[FisherBiasBlockInfluence] = []
    if blocks:
        labels = tuple(str(name).strip() for name, _ in blocks)
        if any(not name for name in labels) or len(set(labels)) != len(labels):
            raise ValueError("bias_blocks must have unique non-empty names.")
        arrays = tuple(np.asarray(value, dtype=float) for _, value in blocks)
        if any(
            value.shape != discrepancy.shape or not np.all(np.isfinite(value))
            for value in arrays
        ):
            raise ValueError("Every bias block must be finite and match the Jacobian rows.")
        summed = np.sum(arrays, axis=0)
        if not np.allclose(summed, discrepancy, rtol=1e-10, atol=1e-12):
            raise ValueError("bias_blocks must sum to the complete signed discrepancy.")
        total_squared = stability.fisher_mahalanobis_change_in_sigma**2
        for label, block in zip(labels, arrays, strict=True):
            profiled_block = profile(block)
            block_modes = -(left.T @ profiled_block)
            contribution = float(np.dot(signed_modes, block_modes))
            fraction = contribution / total_squared if total_squared > 0.0 else 0.0
            influences.append(
                FisherBiasBlockInfluence(
                    name=label,
                    nonzero_row_count=int(np.count_nonzero(block)),
                    whitened_discrepancy_norm=float(np.linalg.norm(block)),
                    nuisance_profiled_discrepancy_norm=float(
                        np.linalg.norm(profiled_block)
                    ),
                    projected_fisher_change_in_sigma=float(
                        np.linalg.norm(block_modes)
                    ),
                    signed_fisher_squared_contribution=contribution,
                    signed_fisher_squared_fraction=fraction,
                    leave_block_fisher_change_in_sigma=float(
                        np.linalg.norm(signed_modes - block_modes)
                    ),
                    signed_fisher_mode_changes_in_sigma=block_modes,
                )
            )

    return NuisanceProfiledOpticalStabilityReport(
        stability=stability,
        optical_parameter_names=names,
        nuisance_parameter_count=nuisance_count,
        nuisance_rank=selected_nuisance_rank,
        nuisance_singular_values=nuisance_singular,
        optical_fisher_singular_values=singular,
        signed_optical_fisher_mode_changes_in_sigma=signed_modes,
        optical_fisher_mode_directions_scaled=directions,
        maximum_optical_fisher_mode_change_in_sigma=float(
            np.max(np.abs(signed_modes))
        ),
        block_influences=tuple(influences),
    )


class PhysicalValidationLedger:
    """Accumulate auditable evidence for a predeclared render-signature set.

    Repeated evidence is retained.  A required signature is accepted when at
    least one of its records passes its declared requirements.  Unknown
    signatures fail closed by default, preventing a typo or stale checkpoint
    from silently expanding the required experiment.
    """

    def __init__(
        self,
        required: (
            Mapping[PhysicalRenderSignature, PhysicalValidationRequirements]
            | Iterable[PhysicalRenderSignature]
        ),
        *,
        default_requirements: PhysicalValidationRequirements = PhysicalValidationRequirements(),
    ) -> None:
        if isinstance(required, Mapping):
            pairs = list(required.items())
        else:
            pairs = [(signature, default_requirements) for signature in required]
        requirement_map: dict[PhysicalRenderSignature, PhysicalValidationRequirements] = {}
        digest_map: dict[str, PhysicalRenderSignature] = {}
        for signature, requirements in pairs:
            if not isinstance(signature, PhysicalRenderSignature):
                raise TypeError("Every required key must be PhysicalRenderSignature.")
            if not isinstance(requirements, PhysicalValidationRequirements):
                raise TypeError(
                    "Every requirement value must be PhysicalValidationRequirements."
                )
            existing = digest_map.get(signature.digest)
            if existing is not None and existing != signature:
                raise ValueError("A render-signature digest collision was detected.")
            digest_map[signature.digest] = signature
            requirement_map[signature] = requirements
        if not requirement_map:
            raise ValueError("At least one physical render signature is required.")
        self._requirements = requirement_map
        self._by_digest = digest_map
        self._evidence: dict[str, list[PhysicalValidationEvidence]] = {
            digest: [] for digest in digest_map
        }
        self._decisions: dict[str, list[PhysicalValidationDecision]] = {
            digest: [] for digest in digest_map
        }

    @property
    def required_signatures(self) -> tuple[PhysicalRenderSignature, ...]:
        return tuple(sorted(self._requirements, key=lambda item: item.digest))

    def record(
        self,
        evidence: PhysicalValidationEvidence,
        *,
        allow_unrequired: bool = False,
        requirements: PhysicalValidationRequirements | None = None,
    ) -> PhysicalValidationDecision:
        digest = evidence.signature.digest
        if digest not in self._by_digest:
            if not allow_unrequired:
                raise KeyError(
                    "Evidence signature is not in the predeclared required set: "
                    f"{digest}."
                )
            selected_requirements = (
                PhysicalValidationRequirements()
                if requirements is None
                else requirements
            )
            if not isinstance(selected_requirements, PhysicalValidationRequirements):
                raise TypeError("requirements must be PhysicalValidationRequirements.")
            self._by_digest[digest] = evidence.signature
            self._requirements[evidence.signature] = selected_requirements
            self._evidence[digest] = []
            self._decisions[digest] = []
        selected_requirements = self._requirements[evidence.signature]
        index = len(self._evidence[digest])
        decision = evaluate_physical_validation(
            evidence,
            selected_requirements,
            evidence_index=index,
        )
        self._evidence[digest].append(evidence)
        self._decisions[digest].append(decision)
        return decision

    def evidence_for(
        self, signature: PhysicalRenderSignature
    ) -> tuple[PhysicalValidationEvidence, ...]:
        return tuple(self._evidence.get(signature.digest, ()))

    def decisions_for(
        self, signature: PhysicalRenderSignature
    ) -> tuple[PhysicalValidationDecision, ...]:
        return tuple(self._decisions.get(signature.digest, ()))

    @property
    def coverage(self) -> PhysicalValidationCoverage:
        accepted: list[str] = []
        failed: list[str] = []
        missing: list[str] = []
        for digest in sorted(self._by_digest):
            decisions = self._decisions[digest]
            if any(decision.passed for decision in decisions):
                accepted.append(digest)
            elif decisions:
                failed.append(digest)
            else:
                missing.append(digest)
        required_count = len(self._by_digest)
        return PhysicalValidationCoverage(
            required_count=required_count,
            accepted_count=len(accepted),
            failed_count=len(failed),
            missing_count=len(missing),
            accepted_signature_digests=tuple(accepted),
            failed_signature_digests=tuple(failed),
            missing_signature_digests=tuple(missing),
            complete=len(accepted) == required_count,
        )

    @property
    def complete(self) -> bool:
        return self.coverage.complete

    def to_payload(self) -> dict[str, object]:
        entries: list[dict[str, object]] = []
        for signature in self.required_signatures:
            digest = signature.digest
            entries.append(
                {
                    "signature": signature.to_payload(),
                    "signature_digest": digest,
                    "requirements": self._requirements[signature].to_payload(),
                    "evidence": [item.to_payload() for item in self._evidence[digest]],
                    "decisions": [item.to_payload() for item in self._decisions[digest]],
                }
            )
        payload: dict[str, object] = {
            "schema": _LEDGER_SCHEMA,
            "version": _VERSION,
            "coverage": self.coverage.to_payload(),
            "entries": entries,
        }
        payload["sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
        return payload

    def write_json(self, path: str | Path) -> Path:
        """Atomically write the complete auditable ledger as canonical JSON."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        contents = _canonical_json(self.to_payload()) + b"\n"
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        try:
            with open(descriptor, "wb", closefd=True) as stream:
                stream.write(contents)
                stream.flush()
            Path(temporary_name).replace(destination)
        except BaseException:
            Path(temporary_name).unlink(missing_ok=True)
            raise
        return destination


__all__ = [
    "AnalyticGaussianDetectorReference",
    "FisherBiasBlockInfluence",
    "FisherParameterStabilityReport",
    "GaussianPerforationCertificate",
    "NuisanceProfiledOpticalStabilityReport",
    "PhysicalRenderSignature",
    "PhysicalValidationThresholds",
    "PhysicalValidationRequirements",
    "PhysicalValidationEvidence",
    "PhysicalValidationDecision",
    "PhysicalValidationCoverage",
    "PhysicalValidationLedger",
    "analytic_single_gaussian_detector_reference",
    "assess_fisher_parameter_stability",
    "assess_nuisance_profiled_optical_stability",
    "certify_single_gaussian_perforation",
    "stable_validation_signature",
    "evaluate_physical_validation",
]
