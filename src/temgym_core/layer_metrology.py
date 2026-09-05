"""Certified multilayer-interface metrology for NEOARM image calibration.

The MAG*I*CAL multilayer is not represented by an arbitrary Fourier period at
the magnifications where individual Si and SiGe layers are resolved.  This
module keeps the certificate geometry, detector registration, noisy-image
measurement, and first-order prediction as four explicit pieces:

* :func:`derive_certified_layer_geometry` constructs registered boundary
  lines and layer intervals from one :class:`MAGICALSampleAtlas` and its
  frozen :class:`SampleView`;
* :func:`freeze_layer_boundary_registration` uses a reference image only to
  fix the detector normal, peak ordering, search windows, and edge polarity;
* :func:`extract_registered_layer_boundaries` locates those same interfaces in
  an arbitrary noisy image with a subpixel smooth-step fit over a local cubic
  illumination background; and
* :func:`predict_registered_layer_observables` maps the certificate points
  through a JAX first-order position map.

Images and image shapes use conventional ``(y, x)`` ordering.  Geometric
vectors, matrices, and detector pixel coordinates use ``(x, y)`` ordering.
All physical coordinates are SI metres until the final detector-pixel
conversion.  The explicit convention is important for a non-square detector
with anisotropic pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, Mapping, Sequence

import jax.numpy as jnp
import numpy as np
from scipy import ndimage


LayerObservableMode = Literal[
    "boundary_positions",
    "interval_widths",
    "boundary_positions_and_interval_widths",
]
BoundaryUncertaintyModel = Literal[
    "cumulative_layer_widths",
    "independent_boundary_marginals",
]


class UnresolvedLayerBoundaryError(ValueError):
    """Raised when a frozen reference cannot identify one certified line."""

    def __init__(self, boundary_index: int, boundary_label: str, reason: str):
        self.boundary_index = int(boundary_index)
        self.boundary_label = str(boundary_label)
        super().__init__(
            f"Layer boundary {self.boundary_label!r} is unresolved: {reason}"
        )


def _finite_array(name: str, values: object, shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite array with shape {shape}.")
    result = result.copy()
    result.setflags(write=False)
    return result


def _finite_pair(name: str, values: Sequence[float]) -> np.ndarray:
    return _finite_array(name, values, (2,))


def _immutable_array(
    name: str,
    values: object,
    *,
    ndim: int | None = None,
    trailing_shape: tuple[int, ...] = (),
) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if (
        (ndim is not None and result.ndim != ndim)
        or (trailing_shape and result.shape[-len(trailing_shape) :] != trailing_shape)
        or not np.all(np.isfinite(result))
    ):
        description = (
            f" with {ndim} dimensions" if ndim is not None else ""
        ) + (f" ending in {trailing_shape}" if trailing_shape else "")
        raise ValueError(f"{name} must be a finite array{description}.")
    result = result.copy()
    result.setflags(write=False)
    return result


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    if not result:
        raise ValueError("Layer labels must contain at least one alphanumeric character.")
    return result


def _validate_mode(value: str) -> LayerObservableMode:
    mode = str(value)
    allowed = {
        "boundary_positions",
        "interval_widths",
        "boundary_positions_and_interval_widths",
    }
    if mode not in allowed:
        raise ValueError(f"observable_mode must be one of {sorted(allowed)}.")
    return mode  # type: ignore[return-value]


@dataclass(frozen=True)
class CertifiedLayerGeometry2D:
    """Certificate-backed parallel boundary lines in one frozen sample view.

    ``boundary_indices`` retain the indices of the full stack.  Index zero is
    the registered stack origin; positive indices correspond to
    ``MAGICALCertificate.layer_boundaries``.  A view may contain only a
    consecutive subset of the full stack.  ``interval_pairs`` index rows of
    the retained boundary arrays and contain only genuinely adjacent
    certificate boundaries.
    """

    certificate_id: str
    certificate_source: str
    region_name: str
    boundary_indices: tuple[int, ...]
    boundary_labels: tuple[str, ...]
    boundary_points_local_xy_m: np.ndarray
    boundary_line_endpoints_local_xy_m: np.ndarray
    boundary_standard_uncertainties_m: np.ndarray
    interval_pairs: np.ndarray
    interval_labels: tuple[str, ...]
    interval_materials: tuple[str, ...]
    sample_translation_xy_m: np.ndarray
    sample_projection_world_from_local: np.ndarray
    boundary_uncertainty_model: BoundaryUncertaintyModel = (
        "independent_boundary_marginals"
    )

    def __post_init__(self) -> None:
        certificate_id = str(self.certificate_id).strip()
        certificate_source = str(self.certificate_source).strip()
        region_name = str(self.region_name).strip()
        if not certificate_id or not certificate_source or not region_name:
            raise ValueError("Certificate ID, source, and region name must be non-empty.")

        indices = tuple(int(value) for value in self.boundary_indices)
        labels = tuple(_slug(value) for value in self.boundary_labels)
        if (
            len(indices) < 2
            or len(indices) != len(labels)
            or len(set(indices)) != len(indices)
            or len(set(labels)) != len(labels)
            or any(right <= left for left, right in zip(indices, indices[1:]))
        ):
            raise ValueError(
                "Boundary indices and labels must be unique, ordered, and contain "
                "at least two entries."
            )
        count = len(indices)
        points = _immutable_array(
            "boundary_points_local_xy_m",
            self.boundary_points_local_xy_m,
            ndim=2,
            trailing_shape=(2,),
        )
        endpoints = _immutable_array(
            "boundary_line_endpoints_local_xy_m",
            self.boundary_line_endpoints_local_xy_m,
            ndim=3,
            trailing_shape=(2, 2),
        )
        uncertainty = _immutable_array(
            "boundary_standard_uncertainties_m",
            self.boundary_standard_uncertainties_m,
            ndim=1,
        )
        if points.shape != (count, 2) or endpoints.shape != (count, 2, 2):
            raise ValueError("Boundary point/endpoint arrays must match boundary labels.")
        if uncertainty.shape != (count,) or np.any(uncertainty < 0.0):
            raise ValueError(
                "Boundary standard uncertainties must be non-negative and match "
                "the boundary count."
            )
        if np.any(np.diff(points[:, 0]) <= 0.0):
            raise ValueError("Layer boundary local x coordinates must increase strictly.")
        if not np.allclose(endpoints[:, :, 0], points[:, None, 0], atol=1e-18):
            raise ValueError("Each certified boundary line must have constant local x.")
        if np.any(endpoints[:, 1, 1] <= endpoints[:, 0, 1]):
            raise ValueError("Boundary line endpoints must increase along local y.")

        pairs = np.asarray(self.interval_pairs, dtype=int)
        interval_labels = tuple(_slug(value) for value in self.interval_labels)
        materials = tuple(_slug(value) for value in self.interval_materials)
        if (
            pairs.ndim != 2
            or pairs.shape[1:] != (2,)
            or pairs.shape[0] != len(interval_labels)
            or pairs.shape[0] != len(materials)
            or len(set(interval_labels)) != len(interval_labels)
            or np.any(pairs < 0)
            or np.any(pairs >= count)
            or np.any(pairs[:, 1] <= pairs[:, 0])
        ):
            raise ValueError("Interval pairs, labels, and materials are inconsistent.")
        for left, right in pairs:
            if indices[right] != indices[left] + 1:
                raise ValueError("Intervals may connect only adjacent certificate boundaries.")
        if not len(pairs):
            raise ValueError("At least one visible certified layer interval is required.")
        pairs = pairs.copy()
        pairs.setflags(write=False)

        translation = _finite_pair(
            "sample_translation_xy_m", self.sample_translation_xy_m
        )
        projection = _finite_array(
            "sample_projection_world_from_local",
            self.sample_projection_world_from_local,
            (2, 2),
        )
        if abs(float(np.linalg.det(projection))) <= np.finfo(float).eps:
            raise ValueError("sample_projection_world_from_local must be nonsingular.")
        uncertainty_model = str(self.boundary_uncertainty_model)
        if uncertainty_model not in {
            "cumulative_layer_widths",
            "independent_boundary_marginals",
        }:
            raise ValueError("Unknown boundary_uncertainty_model.")

        object.__setattr__(self, "certificate_id", certificate_id)
        object.__setattr__(self, "certificate_source", certificate_source)
        object.__setattr__(self, "region_name", region_name)
        object.__setattr__(self, "boundary_indices", indices)
        object.__setattr__(self, "boundary_labels", labels)
        object.__setattr__(self, "boundary_points_local_xy_m", points)
        object.__setattr__(self, "boundary_line_endpoints_local_xy_m", endpoints)
        object.__setattr__(self, "boundary_standard_uncertainties_m", uncertainty)
        object.__setattr__(self, "interval_pairs", pairs)
        object.__setattr__(self, "interval_labels", interval_labels)
        object.__setattr__(self, "interval_materials", materials)
        object.__setattr__(self, "sample_translation_xy_m", translation)
        object.__setattr__(self, "sample_projection_world_from_local", projection)
        object.__setattr__(self, "boundary_uncertainty_model", uncertainty_model)

    @property
    def n_boundaries(self) -> int:
        return len(self.boundary_labels)

    @property
    def n_intervals(self) -> int:
        return len(self.interval_labels)

    @property
    def boundary_points_world_xy_m(self) -> np.ndarray:
        values = self.sample_translation_xy_m + np.einsum(
            "ij,nj->ni",
            self.sample_projection_world_from_local,
            self.boundary_points_local_xy_m,
        )
        values.setflags(write=False)
        return values

    @property
    def boundary_line_endpoints_world_xy_m(self) -> np.ndarray:
        values = self.sample_translation_xy_m + np.einsum(
            "ij,nkj->nki",
            self.sample_projection_world_from_local,
            self.boundary_line_endpoints_local_xy_m,
        )
        values.setflags(write=False)
        return values

    @property
    def interval_widths_local_m(self) -> np.ndarray:
        x = self.boundary_points_local_xy_m[:, 0]
        values = x[self.interval_pairs[:, 1]] - x[self.interval_pairs[:, 0]]
        values.setflags(write=False)
        return values

    @property
    def interval_standard_uncertainties_m(self) -> np.ndarray:
        covariance = self.boundary_certificate_covariance_local_m2
        incidence = np.zeros((self.n_intervals, self.n_boundaries), dtype=float)
        rows = np.arange(self.n_intervals)
        incidence[rows, self.interval_pairs[:, 0]] = -1.0
        incidence[rows, self.interval_pairs[:, 1]] = 1.0
        values = np.sqrt(
            np.maximum(np.diag(incidence @ covariance @ incidence.T), 0.0)
        )
        values.setflags(write=False)
        return values

    @property
    def boundary_certificate_covariance_local_m2(self) -> np.ndarray:
        """Cumulative certificate covariance of retained boundary coordinates.

        MAG*I*CAL boundary uncertainties are cumulative uncertainties from the
        registered stack origin.  Consequently, two boundaries share all
        preceding layer-width errors and ``Cov(b_i, b_j) = min(var_i, var_j)``.
        Third-party/manual geometry defaults explicitly to independent
        marginal boundary uncertainty unless its constructor declares the
        cumulative model.
        """

        variance = self.boundary_standard_uncertainties_m**2
        if self.boundary_uncertainty_model == "cumulative_layer_widths":
            tolerance = 100.0 * np.finfo(float).eps * max(
                float(np.max(variance, initial=0.0)), np.finfo(float).tiny
            )
            if np.any(np.diff(variance) < -tolerance):
                raise ValueError(
                    "Cumulative boundary variances must be non-decreasing."
                )
            covariance = np.minimum.outer(variance, variance)
        else:
            covariance = np.diag(variance)
        covariance = np.asarray(covariance, dtype=float)
        covariance.setflags(write=False)
        return covariance

    def to_metadata(self) -> dict[str, object]:
        return {
            "certificate_id": self.certificate_id,
            "certificate_source": self.certificate_source,
            "region_name": self.region_name,
            "boundary_indices": list(self.boundary_indices),
            "boundary_labels": list(self.boundary_labels),
            "boundary_points_local_xy_m": self.boundary_points_local_xy_m.tolist(),
            "boundary_line_endpoints_local_xy_m": (
                self.boundary_line_endpoints_local_xy_m.tolist()
            ),
            "boundary_standard_uncertainties_m": (
                self.boundary_standard_uncertainties_m.tolist()
            ),
            "interval_pairs": self.interval_pairs.tolist(),
            "interval_labels": list(self.interval_labels),
            "interval_materials": list(self.interval_materials),
            "interval_widths_local_m": self.interval_widths_local_m.tolist(),
            "interval_standard_uncertainties_m": (
                self.interval_standard_uncertainties_m.tolist()
            ),
            "boundary_uncertainty_model": self.boundary_uncertainty_model,
            "sample_translation_xy_m": self.sample_translation_xy_m.tolist(),
            "sample_projection_world_from_local": (
                self.sample_projection_world_from_local.tolist()
            ),
            "coordinate_convention": "geometry_xy_metres",
        }

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> "CertifiedLayerGeometry2D":
        return cls(
            certificate_id=str(metadata["certificate_id"]),
            certificate_source=str(metadata["certificate_source"]),
            region_name=str(metadata["region_name"]),
            boundary_indices=tuple(metadata["boundary_indices"]),  # type: ignore[arg-type]
            boundary_labels=tuple(metadata["boundary_labels"]),  # type: ignore[arg-type]
            boundary_points_local_xy_m=np.asarray(
                metadata["boundary_points_local_xy_m"]
            ),
            boundary_line_endpoints_local_xy_m=np.asarray(
                metadata["boundary_line_endpoints_local_xy_m"]
            ),
            boundary_standard_uncertainties_m=np.asarray(
                metadata["boundary_standard_uncertainties_m"]
            ),
            interval_pairs=np.asarray(metadata["interval_pairs"], dtype=int),
            interval_labels=tuple(metadata["interval_labels"]),  # type: ignore[arg-type]
            interval_materials=tuple(metadata["interval_materials"]),  # type: ignore[arg-type]
            sample_translation_xy_m=np.asarray(metadata["sample_translation_xy_m"]),
            sample_projection_world_from_local=np.asarray(
                metadata["sample_projection_world_from_local"]
            ),
            boundary_uncertainty_model=str(
                metadata.get(
                    "boundary_uncertainty_model",
                    "independent_boundary_marginals",
                )
            ),  # type: ignore[arg-type]
        )


def derive_certified_layer_geometry(
    atlas: object,
    view: object,
    *,
    region_name: str | None = None,
    clip_to_view: bool = True,
    minimum_visible_intervals: int = 2,
) -> CertifiedLayerGeometry2D:
    """Derive registered multilayer lines from an atlas and frozen view.

    The layer stack runs along local ``x`` and its interfaces run along local
    ``y`` in the current MAG*I*CAL certificate.  View clipping retains only
    interfaces inside the frozen physical footprint; interval labels always
    retain their full-stack indices, so schemas do not depend on array crops.
    """

    certificate = getattr(atlas, "certificate", None)
    selection = getattr(view, "selection", None)
    if certificate is None or selection is None:
        raise TypeError("atlas and view must expose certificate and selection metadata.")
    selected_name = (
        region_name
        if region_name is not None
        else getattr(selection, "selected_region_name", None)
    )
    if selected_name is None:
        raise ValueError("A selected multilayer region is required.")
    try:
        region = atlas.region(selected_name)
    except AttributeError as error:
        raise TypeError("atlas must provide region(name).") from error
    if getattr(region, "kind", None) != "multilayer":
        raise ValueError(f"Region {selected_name!r} is not a multilayer region.")
    if int(minimum_visible_intervals) != minimum_visible_intervals or int(
        minimum_visible_intervals
    ) < 1:
        raise ValueError("minimum_visible_intervals must be a positive integer.")

    cx, cy = (float(value) for value in region.centre_xy_m)
    size_x, size_y = (float(value) for value in region.size_xy_m)
    stack_origin_x = cx - 0.5 * size_x
    full_indices = [0]
    full_x = [stack_origin_x]
    full_sigma = [0.0]
    full_labels = ["b00_stack_origin"]
    boundaries = tuple(certificate.layer_boundaries)
    boundary_material_by_index = {
        int(boundary.index): str(boundary.left_material).lower()
        for boundary in boundaries
    }
    for boundary in boundaries:
        index = int(boundary.index)
        full_indices.append(index)
        full_x.append(stack_origin_x + float(boundary.coordinate_m))
        full_sigma.append(float(boundary.standard_uncertainty_m))
        if index == len(boundaries):
            label = f"b{index:02d}_stack_end"
        else:
            label = (
                f"b{index:02d}_{str(boundary.left_material).lower()}_to_"
                f"{str(boundary.right_material).lower()}"
            )
        full_labels.append(label)

    if clip_to_view:
        bounds = tuple(float(value) for value in selection.local_bounds_xyxy_m)
        if len(bounds) != 4 or not np.all(np.isfinite(bounds)):
            raise ValueError("view selection local bounds must be four finite values.")
        xmin, xmax, ymin, ymax = bounds
        y_lower = max(cy - 0.5 * size_y, ymin)
        y_upper = min(cy + 0.5 * size_y, ymax)
        if y_upper <= y_lower:
            raise ValueError("The frozen view does not intersect the multilayer height.")
        retained = [
            index for index, x_value in enumerate(full_x) if xmin <= x_value <= xmax
        ]
    else:
        y_lower, y_upper = cy - 0.5 * size_y, cy + 0.5 * size_y
        retained = list(range(len(full_x)))
    if len(retained) < 2:
        raise ValueError("The frozen view contains fewer than two certified boundaries.")

    indices = tuple(full_indices[index] for index in retained)
    labels = tuple(full_labels[index] for index in retained)
    points = np.asarray([(full_x[index], cy) for index in retained], dtype=float)
    endpoints = np.asarray(
        [((full_x[index], y_lower), (full_x[index], y_upper)) for index in retained],
        dtype=float,
    )
    uncertainties = np.asarray([full_sigma[index] for index in retained], dtype=float)
    interval_pairs: list[tuple[int, int]] = []
    interval_labels: list[str] = []
    interval_materials: list[str] = []
    for retained_left, (left_index, right_index) in enumerate(
        zip(indices, indices[1:])
    ):
        if right_index != left_index + 1:
            continue
        layer_number = right_index
        try:
            material = boundary_material_by_index[layer_number]
        except KeyError as error:
            raise ValueError(
                f"Certificate has no material identity for layer {layer_number}."
            ) from error
        interval_pairs.append((retained_left, retained_left + 1))
        interval_labels.append(f"l{layer_number:02d}_{material}")
        interval_materials.append(material)
    if len(interval_pairs) < int(minimum_visible_intervals):
        raise ValueError(
            "The frozen view contains fewer than the requested number of adjacent "
            "certified layer intervals."
        )

    projection = np.asarray(view.projection_world_from_local, dtype=float)
    translation = np.asarray(view.centre_xy_m, dtype=float)
    return CertifiedLayerGeometry2D(
        certificate_id=str(getattr(certificate, "version", "unspecified-certificate")),
        certificate_source=str(
            getattr(certificate, "certificate_source", "unspecified-certificate-source")
        ),
        region_name=str(selected_name),
        boundary_indices=indices,
        boundary_labels=labels,
        boundary_points_local_xy_m=points,
        boundary_line_endpoints_local_xy_m=endpoints,
        boundary_standard_uncertainties_m=uncertainties,
        interval_pairs=np.asarray(interval_pairs, dtype=int),
        interval_labels=tuple(interval_labels),
        interval_materials=tuple(interval_materials),
        sample_translation_xy_m=translation,
        sample_projection_world_from_local=projection,
        boundary_uncertainty_model="cumulative_layer_widths",
    )


def clip_certified_layer_geometry_to_detector(
    geometry: CertifiedLayerGeometry2D,
    *,
    detector_position_map_xy_m_per_m: object,
    detector_pixel_size_yx_m: Sequence[float],
    image_shape_yx: Sequence[int],
    detector_optical_axis_xy_px: Sequence[float] | None = None,
    detector_margin_px: float = 2.0,
    minimum_visible_line_length_px: float = 8.0,
    minimum_visible_intervals: int = 2,
) -> CertifiedLayerGeometry2D:
    """Restrict certified lines to the usable detector rectangle.

    ``ViewSelection.local_bounds_xyxy_m`` is an axis-aligned footprint
    envelope.  For a rotated image it can contain boundary anchors whose
    finite line segment misses the detector.  This second, optical-map-aware
    clip performs exact segment/rectangle intersection in detector pixels and
    moves each retained anchor to the midpoint of its visible segment.  The
    certificate x coordinate and full-stack index never change.
    """

    shape = tuple(int(value) for value in image_shape_yx)
    if len(shape) != 2 or min(shape) < 8:
        raise ValueError("image_shape_yx must contain two integers >= 8.")
    margin = float(detector_margin_px)
    minimum_length = float(minimum_visible_line_length_px)
    if (
        not np.isfinite(margin)
        or margin < 0.0
        or 2.0 * margin >= min(shape) - 1
    ):
        raise ValueError("detector_margin_px leaves no usable detector rectangle.")
    if not np.isfinite(minimum_length) or minimum_length <= 0.0:
        raise ValueError("minimum_visible_line_length_px must be finite and positive.")
    if int(minimum_visible_intervals) != minimum_visible_intervals or int(
        minimum_visible_intervals
    ) < 1:
        raise ValueError("minimum_visible_intervals must be a positive integer.")

    pixel_map = _detector_pixel_map(
        detector_position_map_xy_m_per_m, detector_pixel_size_yx_m
    )
    ny, nx = shape
    optical_axis = (
        np.asarray((0.5 * (nx - 1), 0.5 * (ny - 1)), dtype=float)
        if detector_optical_axis_xy_px is None
        else np.asarray(detector_optical_axis_xy_px, dtype=float)
    )
    _finite_pair("detector_optical_axis_xy_px", optical_axis)
    world_endpoints = geometry.sample_translation_xy_m + np.einsum(
        "ij,nkj->nki",
        geometry.sample_projection_world_from_local,
        geometry.boundary_line_endpoints_local_xy_m,
    )
    detector_endpoints = optical_axis + np.einsum(
        "ij,nkj->nki", pixel_map, world_endpoints
    )
    rectangle_lower = np.asarray((margin, margin))
    rectangle_upper = np.asarray((nx - 1.0 - margin, ny - 1.0 - margin))

    retained_rows: list[int] = []
    clipped_local_endpoints: list[np.ndarray] = []
    clipped_local_points: list[np.ndarray] = []
    for row, (detector_segment, local_segment) in enumerate(
        zip(detector_endpoints, geometry.boundary_line_endpoints_local_xy_m, strict=True)
    ):
        start, stop = detector_segment
        direction = stop - start
        lower_t, upper_t = 0.0, 1.0
        valid = True
        for coordinate in range(2):
            if abs(direction[coordinate]) <= 100.0 * np.finfo(float).eps:
                if not (
                    rectangle_lower[coordinate]
                    <= start[coordinate]
                    <= rectangle_upper[coordinate]
                ):
                    valid = False
                    break
                continue
            crossings = (
                (rectangle_lower[coordinate] - start[coordinate])
                / direction[coordinate],
                (rectangle_upper[coordinate] - start[coordinate])
                / direction[coordinate],
            )
            lower_t = max(lower_t, min(crossings))
            upper_t = min(upper_t, max(crossings))
            if upper_t <= lower_t:
                valid = False
                break
        visible_length = np.linalg.norm(direction) * max(0.0, upper_t - lower_t)
        if not valid or visible_length < minimum_length:
            continue
        clipped = local_segment[0] + np.asarray((lower_t, upper_t))[:, None] * (
            local_segment[1] - local_segment[0]
        )
        retained_rows.append(row)
        clipped_local_endpoints.append(clipped)
        clipped_local_points.append(np.mean(clipped, axis=0))

    if len(retained_rows) < 2:
        raise ValueError("Fewer than two certified boundary lines cross the detector.")
    original_interval = {
        (
            geometry.boundary_indices[left],
            geometry.boundary_indices[right],
        ): (label, material)
        for (left, right), label, material in zip(
            geometry.interval_pairs,
            geometry.interval_labels,
            geometry.interval_materials,
            strict=True,
        )
    }
    retained_indices = tuple(geometry.boundary_indices[row] for row in retained_rows)
    interval_pairs: list[tuple[int, int]] = []
    interval_labels: list[str] = []
    interval_materials: list[str] = []
    for left_row, (left_index, right_index) in enumerate(
        zip(retained_indices, retained_indices[1:])
    ):
        description = original_interval.get((left_index, right_index))
        if description is None:
            continue
        interval_pairs.append((left_row, left_row + 1))
        interval_labels.append(description[0])
        interval_materials.append(description[1])
    if len(interval_pairs) < int(minimum_visible_intervals):
        raise ValueError(
            "The detector contains fewer than the requested number of complete "
            "certified layer intervals."
        )
    return CertifiedLayerGeometry2D(
        certificate_id=geometry.certificate_id,
        certificate_source=geometry.certificate_source,
        region_name=geometry.region_name,
        boundary_indices=retained_indices,
        boundary_labels=tuple(geometry.boundary_labels[row] for row in retained_rows),
        boundary_points_local_xy_m=np.asarray(clipped_local_points),
        boundary_line_endpoints_local_xy_m=np.asarray(clipped_local_endpoints),
        boundary_standard_uncertainties_m=geometry.boundary_standard_uncertainties_m[
            np.asarray(retained_rows, dtype=int)
        ],
        interval_pairs=np.asarray(interval_pairs, dtype=int),
        interval_labels=tuple(interval_labels),
        interval_materials=tuple(interval_materials),
        sample_translation_xy_m=geometry.sample_translation_xy_m,
        sample_projection_world_from_local=(
            geometry.sample_projection_world_from_local
        ),
        boundary_uncertainty_model=geometry.boundary_uncertainty_model,
    )


def _subset_layer_geometry(
    geometry: CertifiedLayerGeometry2D,
    retained_rows: Sequence[int],
    *,
    minimum_visible_intervals: int,
) -> CertifiedLayerGeometry2D:
    rows = tuple(int(value) for value in retained_rows)
    if (
        len(rows) < 2
        or len(set(rows)) != len(rows)
        or any(row < 0 or row >= geometry.n_boundaries for row in rows)
        or any(right <= left for left, right in zip(rows, rows[1:]))
    ):
        raise ValueError("Retained layer rows must be unique, ordered, and in range.")
    original_intervals = {
        (left, right): (label, material)
        for (left, right), label, material in zip(
            geometry.interval_pairs,
            geometry.interval_labels,
            geometry.interval_materials,
            strict=True,
        )
    }
    old_to_new = {old: new for new, old in enumerate(rows)}
    pairs: list[tuple[int, int]] = []
    labels: list[str] = []
    materials: list[str] = []
    for (left, right), (label, material) in original_intervals.items():
        if left in old_to_new and right in old_to_new:
            pairs.append((old_to_new[left], old_to_new[right]))
            labels.append(label)
            materials.append(material)
    if len(pairs) < int(minimum_visible_intervals):
        raise ValueError(
            "Reference peak pruning leaves fewer than the requested number of "
            "certified layer intervals."
        )
    row_array = np.asarray(rows, dtype=int)
    return CertifiedLayerGeometry2D(
        certificate_id=geometry.certificate_id,
        certificate_source=geometry.certificate_source,
        region_name=geometry.region_name,
        boundary_indices=tuple(geometry.boundary_indices[row] for row in rows),
        boundary_labels=tuple(geometry.boundary_labels[row] for row in rows),
        boundary_points_local_xy_m=geometry.boundary_points_local_xy_m[row_array],
        boundary_line_endpoints_local_xy_m=(
            geometry.boundary_line_endpoints_local_xy_m[row_array]
        ),
        boundary_standard_uncertainties_m=(
            geometry.boundary_standard_uncertainties_m[row_array]
        ),
        interval_pairs=np.asarray(pairs, dtype=int),
        interval_labels=tuple(labels),
        interval_materials=tuple(materials),
        sample_translation_xy_m=geometry.sample_translation_xy_m,
        sample_projection_world_from_local=(
            geometry.sample_projection_world_from_local
        ),
        boundary_uncertainty_model=geometry.boundary_uncertainty_model,
    )


@dataclass(frozen=True)
class LayerBoundaryRegistration2D:
    """Frozen detector coordinates used to identify multilayer interfaces."""

    image_shape_yx: tuple[int, int]
    detector_normal_xy: np.ndarray
    detector_tangent_xy: np.ndarray
    coordinate_origin_xy_px: np.ndarray
    detector_optical_axis_xy_px: np.ndarray
    search_centres_axis_px: np.ndarray
    search_half_widths_px: np.ndarray
    reference_peak_centres_axis_px: np.ndarray
    peak_tracking_half_widths_px: np.ndarray
    edge_model_widths_px: np.ndarray
    tangent_centres_axis_px: np.ndarray
    tangent_half_widths_px: np.ndarray
    edge_polarities: np.ndarray
    smoothing_sigma_px: float = 0.8
    profile_step_px: float = 0.125
    tangent_step_px: float = 0.5
    minimum_peak_snr: float = 2.0
    minimum_step_improvement: float = 0.02
    extraction_rejection_minimum_peak_snr: float | None = None
    extraction_rejection_minimum_step_improvement: float | None = None

    def __post_init__(self) -> None:
        shape = tuple(int(value) for value in self.image_shape_yx)
        if len(shape) != 2 or min(shape) < 8:
            raise ValueError("image_shape_yx must contain two integers >= 8.")
        normal = _finite_pair("detector_normal_xy", self.detector_normal_xy)
        tangent = _finite_pair("detector_tangent_xy", self.detector_tangent_xy)
        if not np.isclose(np.linalg.norm(normal), 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError("detector_normal_xy must be a unit vector.")
        if not np.isclose(np.linalg.norm(tangent), 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError("detector_tangent_xy must be a unit vector.")
        if abs(float(normal @ tangent)) > 1e-10:
            raise ValueError("Detector normal and tangent must be orthogonal.")
        origin = _finite_pair("coordinate_origin_xy_px", self.coordinate_origin_xy_px)
        optical_axis = _finite_pair(
            "detector_optical_axis_xy_px", self.detector_optical_axis_xy_px
        )

        centres = _immutable_array(
            "search_centres_axis_px", self.search_centres_axis_px, ndim=1
        )
        count = centres.size
        if count < 2 or np.any(np.diff(centres) <= 0.0):
            raise ValueError("At least two ordered layer search centres are required.")
        half = _immutable_array(
            "search_half_widths_px", self.search_half_widths_px, ndim=1
        )
        reference_centres = _immutable_array(
            "reference_peak_centres_axis_px",
            self.reference_peak_centres_axis_px,
            ndim=1,
        )
        tracking_half = _immutable_array(
            "peak_tracking_half_widths_px",
            self.peak_tracking_half_widths_px,
            ndim=1,
        )
        model_widths = _immutable_array(
            "edge_model_widths_px", self.edge_model_widths_px, ndim=1
        )
        tangent_centres = _immutable_array(
            "tangent_centres_axis_px", self.tangent_centres_axis_px, ndim=1
        )
        tangent_half = _immutable_array(
            "tangent_half_widths_px", self.tangent_half_widths_px, ndim=1
        )
        polarities = _immutable_array("edge_polarities", self.edge_polarities, ndim=1)
        if any(
            array.shape != (count,)
            for array in (
                half,
                reference_centres,
                tracking_half,
                model_widths,
                tangent_centres,
                tangent_half,
                polarities,
            )
        ):
            raise ValueError("Every per-boundary registration array must share one length.")
        if (
            np.any(half <= 0.0)
            or np.any(tracking_half <= 0.0)
            or np.any(model_widths <= 0.0)
            or np.any(tangent_half <= 0.0)
        ):
            raise ValueError("Search, tracking, and tangent half-widths must be positive.")
        if np.any(np.abs(reference_centres - centres) + tracking_half > half):
            raise ValueError(
                "Every reference tracking window must lie inside its association window."
            )
        if not np.all(np.isin(polarities, (-1.0, 1.0))):
            raise ValueError("edge_polarities entries must be -1 or +1.")
        if np.any(
            centres[:-1] + half[:-1] >= centres[1:] - half[1:]
        ):
            raise ValueError("Layer-boundary search windows must not overlap.")
        for name in (
            "smoothing_sigma_px",
            "profile_step_px",
            "tangent_step_px",
            "minimum_peak_snr",
            "minimum_step_improvement",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0 or (
                name in {"profile_step_px", "tangent_step_px"} and value <= 0.0
            ):
                raise ValueError(f"{name} has an invalid value.")
            object.__setattr__(self, name, value)
        if self.minimum_step_improvement >= 1.0:
            raise ValueError("minimum_step_improvement must be smaller than one.")
        for name in (
            "extraction_rejection_minimum_peak_snr",
            "extraction_rejection_minimum_step_improvement",
        ):
            raw_value = getattr(self, name)
            if raw_value is None:
                continue
            value = float(raw_value)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite, non-negative, or None.")
            if name.endswith("step_improvement") and value >= 1.0:
                raise ValueError(f"{name} must be smaller than one.")
            object.__setattr__(self, name, value)
        if self.smoothing_sigma_px == 0.0:
            object.__setattr__(self, "smoothing_sigma_px", 0.0)

        object.__setattr__(self, "image_shape_yx", shape)
        object.__setattr__(self, "detector_normal_xy", normal)
        object.__setattr__(self, "detector_tangent_xy", tangent)
        object.__setattr__(self, "coordinate_origin_xy_px", origin)
        object.__setattr__(self, "detector_optical_axis_xy_px", optical_axis)
        object.__setattr__(self, "search_centres_axis_px", centres)
        object.__setattr__(self, "search_half_widths_px", half)
        object.__setattr__(self, "reference_peak_centres_axis_px", reference_centres)
        object.__setattr__(self, "peak_tracking_half_widths_px", tracking_half)
        object.__setattr__(self, "edge_model_widths_px", model_widths)
        object.__setattr__(self, "tangent_centres_axis_px", tangent_centres)
        object.__setattr__(self, "tangent_half_widths_px", tangent_half)
        object.__setattr__(self, "edge_polarities", polarities)

    @property
    def n_boundaries(self) -> int:
        return int(self.search_centres_axis_px.size)

    def to_metadata(self) -> dict[str, object]:
        return {
            "image_shape_yx": list(self.image_shape_yx),
            "detector_normal_xy": self.detector_normal_xy.tolist(),
            "detector_tangent_xy": self.detector_tangent_xy.tolist(),
            "coordinate_origin_xy_px": self.coordinate_origin_xy_px.tolist(),
            "detector_optical_axis_xy_px": self.detector_optical_axis_xy_px.tolist(),
            "search_centres_axis_px": self.search_centres_axis_px.tolist(),
            "search_half_widths_px": self.search_half_widths_px.tolist(),
            "reference_peak_centres_axis_px": (
                self.reference_peak_centres_axis_px.tolist()
            ),
            "peak_tracking_half_widths_px": (
                self.peak_tracking_half_widths_px.tolist()
            ),
            "edge_model_widths_px": self.edge_model_widths_px.tolist(),
            "tangent_centres_axis_px": self.tangent_centres_axis_px.tolist(),
            "tangent_half_widths_px": self.tangent_half_widths_px.tolist(),
            "edge_polarities": self.edge_polarities.tolist(),
            "smoothing_sigma_px": self.smoothing_sigma_px,
            "profile_step_px": self.profile_step_px,
            "tangent_step_px": self.tangent_step_px,
            "minimum_peak_snr": self.minimum_peak_snr,
            "minimum_step_improvement": self.minimum_step_improvement,
            "reference_discovery_gates": {
                "minimum_peak_snr": self.minimum_peak_snr,
                "minimum_step_improvement": self.minimum_step_improvement,
            },
            "extraction_rejection_policy": {
                "minimum_peak_snr": self.extraction_rejection_minimum_peak_snr,
                "minimum_step_improvement": (
                    self.extraction_rejection_minimum_step_improvement
                ),
                "require_registered_polarity": True,
                "enforcement": "explicit_opt_in_only",
            },
            "coordinate_convention": "detector_xy_pixel_centres",
            "reference_role": "peak_identity_order_sign_and_window_only",
            "edge_width_role": "reference_learned_estimator_nuisance_not_inverse_target",
        }

    @classmethod
    def from_metadata(
        cls, metadata: Mapping[str, object]
    ) -> "LayerBoundaryRegistration2D":
        reference_gates = metadata.get("reference_discovery_gates", {})
        if not isinstance(reference_gates, Mapping):
            raise ValueError("reference_discovery_gates must be a mapping.")
        extraction_policy = metadata.get("extraction_rejection_policy", {})
        if not isinstance(extraction_policy, Mapping):
            raise ValueError("extraction_rejection_policy must be a mapping.")
        return cls(
            image_shape_yx=tuple(metadata["image_shape_yx"]),  # type: ignore[arg-type]
            detector_normal_xy=np.asarray(metadata["detector_normal_xy"]),
            detector_tangent_xy=np.asarray(metadata["detector_tangent_xy"]),
            coordinate_origin_xy_px=np.asarray(metadata["coordinate_origin_xy_px"]),
            detector_optical_axis_xy_px=np.asarray(
                metadata["detector_optical_axis_xy_px"]
            ),
            search_centres_axis_px=np.asarray(metadata["search_centres_axis_px"]),
            search_half_widths_px=np.asarray(metadata["search_half_widths_px"]),
            reference_peak_centres_axis_px=np.asarray(
                metadata["reference_peak_centres_axis_px"]
            ),
            peak_tracking_half_widths_px=np.asarray(
                metadata["peak_tracking_half_widths_px"]
            ),
            edge_model_widths_px=np.asarray(metadata["edge_model_widths_px"]),
            tangent_centres_axis_px=np.asarray(metadata["tangent_centres_axis_px"]),
            tangent_half_widths_px=np.asarray(metadata["tangent_half_widths_px"]),
            edge_polarities=np.asarray(metadata["edge_polarities"]),
            smoothing_sigma_px=float(metadata["smoothing_sigma_px"]),
            profile_step_px=float(metadata["profile_step_px"]),
            tangent_step_px=float(metadata["tangent_step_px"]),
            minimum_peak_snr=float(
                reference_gates.get(
                    "minimum_peak_snr", metadata["minimum_peak_snr"]
                )
            ),
            minimum_step_improvement=float(
                reference_gates.get(
                    "minimum_step_improvement",
                    metadata.get("minimum_step_improvement", 0.02),
                )
            ),
            extraction_rejection_minimum_peak_snr=(
                None
                if extraction_policy.get("minimum_peak_snr") is None
                else float(extraction_policy["minimum_peak_snr"])
            ),
            extraction_rejection_minimum_step_improvement=(
                None
                if extraction_policy.get("minimum_step_improvement") is None
                else float(extraction_policy["minimum_step_improvement"])
            ),
        )


def _detector_pixel_map(
    detector_position_map_xy_m_per_m: object,
    detector_pixel_size_yx_m: Sequence[float],
) -> np.ndarray:
    matrix = np.asarray(detector_position_map_xy_m_per_m, dtype=float)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError("detector_position_map_xy_m_per_m must be a finite 2x2 map.")
    pixel_yx = np.asarray(detector_pixel_size_yx_m, dtype=float)
    if pixel_yx.shape != (2,) or not np.all(np.isfinite(pixel_yx)) or np.any(pixel_yx <= 0.0):
        raise ValueError("detector_pixel_size_yx_m must contain two positive values.")
    pixel_xy = pixel_yx[::-1]
    return matrix / pixel_xy[:, None]


def _profile_coordinates(
    registration: LayerBoundaryRegistration2D,
    boundary_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centre = registration.search_centres_axis_px[boundary_index]
    half = registration.search_half_widths_px[boundary_index]
    u = np.arange(
        centre - half,
        centre + half + 0.5 * registration.profile_step_px,
        registration.profile_step_px,
    )
    v_centre = registration.tangent_centres_axis_px[boundary_index]
    v_half = registration.tangent_half_widths_px[boundary_index]
    v = np.arange(
        v_centre - v_half,
        v_centre + v_half + 0.5 * registration.tangent_step_px,
        registration.tangent_step_px,
    )
    xy = (
        registration.coordinate_origin_xy_px[None, None, :]
        + u[:, None, None] * registration.detector_normal_xy[None, None, :]
        + v[None, :, None] * registration.detector_tangent_xy[None, None, :]
    )
    return u, xy[..., 0], xy[..., 1]


def _edge_profile(
    image_yx: np.ndarray,
    registration: LayerBoundaryRegistration2D,
    boundary_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, x, y = _profile_coordinates(registration, boundary_index)
    sampled = ndimage.map_coordinates(
        image_yx,
        (y, x),
        order=3,
        mode="constant",
        cval=np.nan,
        prefilter=True,
    )
    finite_count = np.sum(np.isfinite(sampled), axis=1)
    if np.any(finite_count < max(4, int(0.5 * sampled.shape[1]))):
        raise ValueError(
            "A layer integration window leaves the detector; freeze a smaller window."
        )
    profile = np.nanmean(sampled, axis=1)
    if registration.smoothing_sigma_px > 0.0:
        profile = ndimage.gaussian_filter1d(
            profile,
            registration.smoothing_sigma_px / registration.profile_step_px,
            mode="nearest",
        )
    derivative = np.gradient(profile, registration.profile_step_px)
    return u, profile, derivative


def _fit_polynomial_step(
    coordinates: np.ndarray,
    profile: np.ndarray,
    *,
    target_axis_px: float,
    tracking_half_width_px: float,
    model_widths_px: Sequence[float],
    expected_polarity: float | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Fit a cubic illumination trend plus one smooth interface step.

    Returns ``(position, width, polarity, amplitude_snr, improvement,
    amplitude)``.  The polynomial absorbs a slowly varying Gaussian envelope;
    unlike a raw derivative maximum, the fitted step remains centred on a
    shoulder superposed on that envelope.
    """

    x = np.asarray(coordinates, dtype=float)
    y = np.asarray(profile, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or x.size < 12:
        raise ValueError("A layer step fit requires at least twelve profile samples.")
    widths = np.asarray(tuple(float(value) for value in model_widths_px), dtype=float)
    if not len(widths) or np.any(~np.isfinite(widths)) or np.any(widths <= 0.0):
        raise ValueError("model_widths_px must contain positive finite values.")
    if expected_polarity is not None and expected_polarity not in (-1.0, 1.0):
        raise ValueError("expected_polarity must be -1, +1, or None.")
    scale = max(0.5 * float(np.ptp(x)), 1.0)
    normalized = (x - float(target_axis_px)) / scale
    polynomial = np.column_stack(
        (
            np.ones_like(normalized),
            normalized,
            normalized**2,
            normalized**3,
        )
    )
    baseline_coefficients = np.linalg.lstsq(polynomial, y, rcond=None)[0]
    baseline_residual = y - polynomial @ baseline_coefficients
    baseline_rss = float(baseline_residual @ baseline_residual)
    candidate_positions = np.arange(
        float(target_axis_px) - float(tracking_half_width_px),
        float(target_axis_px) + float(tracking_half_width_px) + 0.5 * (x[1] - x[0]),
        x[1] - x[0],
    )
    best: tuple[float, float, float, np.ndarray, np.ndarray, int] | None = None
    best_any_polarity: (
        tuple[float, float, float, np.ndarray, np.ndarray, int] | None
    ) = None
    for width in widths:
        rss_values: list[float] = []
        coefficient_values: list[np.ndarray] = []
        design_values: list[np.ndarray] = []
        for position in candidate_positions:
            step = np.tanh((x - position) / width)
            design = np.column_stack((polynomial, step))
            coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
            amplitude = float(coefficients[-1])
            residual = y - design @ coefficients
            raw_rss = float(residual @ residual)
            candidate = (
                raw_rss,
                float(width),
                float(position),
                coefficients,
                design,
                len(rss_values),
            )
            if best_any_polarity is None or raw_rss < best_any_polarity[0]:
                best_any_polarity = candidate
            selection_rss = raw_rss
            if expected_polarity is not None and amplitude * expected_polarity <= 0.0:
                selection_rss = np.inf
            rss_values.append(selection_rss)
            coefficient_values.append(coefficients)
            design_values.append(design)
        candidate_index = int(np.argmin(rss_values))
        candidate_rss = float(rss_values[candidate_index])
        if not np.isfinite(candidate_rss):
            continue
        if best is None or candidate_rss < best[0]:
            best = (
                candidate_rss,
                float(width),
                float(candidate_positions[candidate_index]),
                coefficient_values[candidate_index],
                design_values[candidate_index],
                candidate_index,
            )
    polarity_constraint_active = best is not None
    if best is None:
        # A very noisy draw can reverse a fitted amplitude.  Returning its
        # registered-window estimator (with a failed-polarity diagnostic) is
        # essential for an uncensored bootstrap covariance.
        best = best_any_polarity
    if best is None:
        raise ValueError("no finite interface-step model was found")
    rss, width, position, coefficients, design, position_index = best
    if 0 < position_index < candidate_positions.size - 1:
        # Re-evaluate the selected width on neighboring grid points for a
        # sub-grid quadratic minimum of the residual sum of squares.
        neighboring_rss = []
        for candidate in candidate_positions[position_index - 1 : position_index + 2]:
            candidate_design = np.column_stack(
                (polynomial, np.tanh((x - candidate) / width))
            )
            candidate_coefficients = np.linalg.lstsq(
                candidate_design, y, rcond=None
            )[0]
            amplitude = float(candidate_coefficients[-1])
            candidate_residual = y - candidate_design @ candidate_coefficients
            candidate_rss = float(candidate_residual @ candidate_residual)
            if (
                polarity_constraint_active
                and expected_polarity is not None
                and amplitude * expected_polarity <= 0.0
            ):
                candidate_rss = np.inf
            neighboring_rss.append(candidate_rss)
        left, centre, right = neighboring_rss
        denominator = left - 2.0 * centre + right
        if np.all(np.isfinite(neighboring_rss)) and denominator > 0.0:
            offset = float(np.clip(0.5 * (left - right) / denominator, -1.0, 1.0))
            position += offset * float(candidate_positions[1] - candidate_positions[0])
            design = np.column_stack(
                (polynomial, np.tanh((x - position) / width))
            )
            coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
            residual = y - design @ coefficients
            rss = float(residual @ residual)
    amplitude = float(coefficients[-1])
    degrees_of_freedom = max(1, x.size - design.shape[1])
    residual_variance = rss / degrees_of_freedom
    coefficient_covariance = residual_variance * np.linalg.pinv(design.T @ design)
    amplitude_standard_error = np.sqrt(
        max(float(coefficient_covariance[-1, -1]), np.finfo(float).tiny)
    )
    amplitude_snr = abs(amplitude) / amplitude_standard_error
    improvement = (
        1.0 - rss / baseline_rss
        if baseline_rss > np.finfo(float).tiny
        else 0.0
    )
    improvement = float(np.clip(improvement, 0.0, 1.0))
    polarity = 1.0 if amplitude >= 0.0 else -1.0
    return position, width, polarity, amplitude_snr, improvement, amplitude


@dataclass(frozen=True)
class LayerBoundaryExtraction:
    """Subpixel registered boundary positions and derived interval widths."""

    boundary_positions_axis_px: np.ndarray
    interval_widths_px: np.ndarray
    peak_snr: np.ndarray
    edge_derivatives: np.ndarray
    step_model_improvement: np.ndarray
    matches_registered_polarity: np.ndarray

    def __post_init__(self) -> None:
        positions = _immutable_array(
            "boundary_positions_axis_px", self.boundary_positions_axis_px, ndim=1
        )
        widths = _immutable_array("interval_widths_px", self.interval_widths_px, ndim=1)
        snr = _immutable_array("peak_snr", self.peak_snr, ndim=1)
        derivatives = _immutable_array("edge_derivatives", self.edge_derivatives, ndim=1)
        improvement = _immutable_array(
            "step_model_improvement", self.step_model_improvement, ndim=1
        )
        polarity = np.asarray(self.matches_registered_polarity, dtype=bool)
        if (
            snr.shape != positions.shape
            or derivatives.shape != positions.shape
            or improvement.shape != positions.shape
            or polarity.shape != positions.shape
        ):
            raise ValueError("Boundary quality arrays must match boundary positions.")
        if np.any(improvement < 0.0) or np.any(improvement > 1.0):
            raise ValueError("Step-model improvements must lie in [0, 1].")
        if np.any(np.diff(positions) <= 0.0) or np.any(widths <= 0.0):
            raise ValueError("Extracted boundaries and layer widths must increase positively.")
        polarity = polarity.copy()
        polarity.setflags(write=False)
        object.__setattr__(self, "boundary_positions_axis_px", positions)
        object.__setattr__(self, "interval_widths_px", widths)
        object.__setattr__(self, "peak_snr", snr)
        object.__setattr__(self, "edge_derivatives", derivatives)
        object.__setattr__(self, "step_model_improvement", improvement)
        object.__setattr__(self, "matches_registered_polarity", polarity)


def extract_registered_layer_boundaries(
    image_yx: object,
    geometry: CertifiedLayerGeometry2D,
    registration: LayerBoundaryRegistration2D,
    *,
    enforce_quality_policy: bool = False,
) -> LayerBoundaryExtraction:
    """Extract all frozen layer peaks from one noisy detector image.

    Discovery thresholds used to accept the deterministic reference are never
    applied to bootstrap/noisy draws.  Set ``enforce_quality_policy=True`` for
    an actual-acquisition rejection decision; only the separately configured
    extraction policy is then applied.  Diagnostics are returned either way.
    """

    image = np.asarray(image_yx, dtype=float)
    if image.shape != registration.image_shape_yx or not np.all(np.isfinite(image)):
        raise ValueError(
            "image_yx must be finite and match registration.image_shape_yx."
        )
    if geometry.n_boundaries != registration.n_boundaries:
        raise ValueError("Geometry and registration boundary counts differ.")
    positions: list[float] = []
    snrs: list[float] = []
    derivatives_at_peak: list[float] = []
    improvements: list[float] = []
    polarity_matches: list[bool] = []
    for index in range(geometry.n_boundaries):
        u, profile, _derivative = _edge_profile(image, registration, index)
        try:
            position, width, polarity, snr, improvement, amplitude = (
                _fit_polynomial_step(
                u,
                profile,
                target_axis_px=registration.reference_peak_centres_axis_px[index],
                tracking_half_width_px=(
                    registration.peak_tracking_half_widths_px[index]
                ),
                model_widths_px=(registration.edge_model_widths_px[index],),
                expected_polarity=registration.edge_polarities[index],
                )
            )
        except ValueError as error:
            raise UnresolvedLayerBoundaryError(
                index, geometry.boundary_labels[index], str(error)
            ) from error
        polarity_matches.append(
            bool(polarity == registration.edge_polarities[index])
        )
        extraction_snr_gate = registration.extraction_rejection_minimum_peak_snr
        if (
            enforce_quality_policy
            and extraction_snr_gate is not None
            and snr < extraction_snr_gate
        ):
            raise UnresolvedLayerBoundaryError(
                index,
                geometry.boundary_labels[index],
                f"peak SNR {snr:.3g} is below the extraction-rejection "
                f"{extraction_snr_gate:.3g} gate",
            )
        extraction_improvement_gate = (
            registration.extraction_rejection_minimum_step_improvement
        )
        if (
            enforce_quality_policy
            and extraction_improvement_gate is not None
            and improvement < extraction_improvement_gate
        ):
            raise UnresolvedLayerBoundaryError(
                index,
                geometry.boundary_labels[index],
                f"step-model improvement {improvement:.3g} is below the "
                f"extraction-rejection {extraction_improvement_gate:.3g} gate",
            )
        if enforce_quality_policy and not polarity_matches[-1]:
            raise UnresolvedLayerBoundaryError(
                index,
                geometry.boundary_labels[index],
                "fitted edge polarity differs from the frozen registration",
            )
        positions.append(position)
        snrs.append(snr)
        improvements.append(improvement)
        # For ``a*tanh((u-u0)/w)``, a/w is the signed derivative at u0.
        derivatives_at_peak.append(float(amplitude / width))
    position_array = np.asarray(positions)
    pairs = geometry.interval_pairs
    widths = position_array[pairs[:, 1]] - position_array[pairs[:, 0]]
    return LayerBoundaryExtraction(
        boundary_positions_axis_px=position_array,
        interval_widths_px=widths,
        peak_snr=np.asarray(snrs),
        edge_derivatives=np.asarray(derivatives_at_peak),
        step_model_improvement=np.asarray(improvements),
        matches_registered_polarity=np.asarray(polarity_matches),
    )


def freeze_layer_boundary_registration(
    reference_image_yx: object,
    geometry: CertifiedLayerGeometry2D,
    *,
    detector_position_map_xy_m_per_m: object,
    detector_pixel_size_yx_m: Sequence[float],
    detector_optical_axis_xy_px: Sequence[float] | None = None,
    coordinate_origin_xy_px: Sequence[float] | None = None,
    search_half_width_px: float | Sequence[float] | None = None,
    integration_fraction: float = 0.6,
    smoothing_sigma_px: float = 0.8,
    profile_step_px: float = 0.125,
    tangent_step_px: float = 0.5,
    minimum_peak_snr: float = 2.0,
    minimum_reference_contrast_fraction: float = 1e-5,
    minimum_step_improvement: float = 0.02,
    maximum_reference_offset_fraction: float = 0.35,
    extraction_rejection_minimum_peak_snr: float | None = None,
    extraction_rejection_minimum_step_improvement: float | None = None,
) -> LayerBoundaryRegistration2D:
    """Freeze detector peak identity/order/sign from a reference exposure.

    The nominal optical map supplies the detector axis and non-overlapping
    association windows.  The reference image supplies only the peak centre
    and derivative sign within each window.  A smooth-edge width is retained
    only as an estimator nuisance parameter.  Its peak positions are never
    substituted for certified specimen coordinates in the forward predictor.
    ``minimum_peak_snr`` and ``minimum_step_improvement`` are reference-only
    discovery gates.  Optional extraction-rejection thresholds are stored as
    a separate, explicitly enforced acquisition policy and are never applied
    implicitly to bootstrap draws.
    """

    image = np.asarray(reference_image_yx, dtype=float)
    if image.ndim != 2 or min(image.shape) < 8 or not np.all(np.isfinite(image)):
        raise ValueError("reference_image_yx must be a finite 2D image of at least 8x8.")
    if not np.isfinite(integration_fraction) or not 0.0 < integration_fraction <= 0.9:
        raise ValueError("integration_fraction must lie in (0, 0.9].")
    reference_contrast_gate = float(minimum_reference_contrast_fraction)
    if not np.isfinite(reference_contrast_gate) or reference_contrast_gate < 0.0:
        raise ValueError(
            "minimum_reference_contrast_fraction must be finite and non-negative."
        )
    improvement_gate = float(minimum_step_improvement)
    if not np.isfinite(improvement_gate) or not 0.0 <= improvement_gate < 1.0:
        raise ValueError("minimum_step_improvement must lie in [0, 1).")
    maximum_offset_fraction = float(maximum_reference_offset_fraction)
    if (
        not np.isfinite(maximum_offset_fraction)
        or not 0.0 < maximum_offset_fraction < 0.5
    ):
        raise ValueError("maximum_reference_offset_fraction must lie in (0, 0.5).")
    pixel_map = _detector_pixel_map(
        detector_position_map_xy_m_per_m, detector_pixel_size_yx_m
    )
    projection = geometry.sample_projection_world_from_local
    translation = geometry.sample_translation_xy_m
    local_to_pixel = pixel_map @ projection
    tangent_vector = local_to_pixel @ np.asarray((0.0, 1.0))
    tangent_norm = float(np.linalg.norm(tangent_vector))
    if tangent_norm <= np.finfo(float).eps:
        raise ValueError("The projected layer tangent is singular on the detector.")
    tangent = tangent_vector / tangent_norm
    normal = np.asarray((tangent[1], -tangent[0]))
    if float(normal @ (local_to_pixel @ np.asarray((1.0, 0.0)))) < 0.0:
        normal = -normal

    ny, nx = image.shape
    optical_axis = (
        np.asarray((0.5 * (nx - 1), 0.5 * (ny - 1)), dtype=float)
        if detector_optical_axis_xy_px is None
        else np.asarray(detector_optical_axis_xy_px, dtype=float)
    )
    origin = optical_axis.copy() if coordinate_origin_xy_px is None else np.asarray(
        coordinate_origin_xy_px, dtype=float
    )
    _finite_pair("detector_optical_axis_xy_px", optical_axis)
    _finite_pair("coordinate_origin_xy_px", origin)

    world_points = translation + np.einsum(
        "ij,nj->ni", projection, geometry.boundary_points_local_xy_m
    )
    detector_points = optical_axis + np.einsum("ij,nj->ni", pixel_map, world_points)
    relative = detector_points - origin
    nominal_u = relative @ normal
    nominal_v = relative @ tangent
    if np.any(np.diff(nominal_u) <= 0.0):
        raise ValueError("The nominal detector map reverses or collapses boundary order.")
    gaps = np.diff(nominal_u)
    nearest_gap = np.minimum(
        np.concatenate(([gaps[0]], gaps)),
        np.concatenate((gaps, [gaps[-1]])),
    )
    certified_interval_gaps = np.asarray(
        [
            nominal_u[right] - nominal_u[left]
            for left, right in geometry.interval_pairs
        ]
    )
    nearest_gap = np.minimum(nearest_gap, np.min(certified_interval_gaps))
    if search_half_width_px is None:
        half = 0.36 * nearest_gap
    else:
        provided = np.asarray(search_half_width_px, dtype=float)
        half = (
            np.full(geometry.n_boundaries, float(provided))
            if provided.ndim == 0
            else provided
        )
    if half.shape != (geometry.n_boundaries,) or not np.all(np.isfinite(half)):
        raise ValueError("search_half_width_px must be scalar or one value per boundary.")
    if np.any(half <= max(profile_step_px, 0.25)):
        raise ValueError("Every search half-width must exceed the profile sampling step.")
    if np.any(nominal_u[:-1] + half[:-1] >= nominal_u[1:] - half[1:]):
        raise ValueError("Requested layer search windows overlap.")

    world_endpoints = translation + np.einsum(
        "ij,nkj->nki", projection, geometry.boundary_line_endpoints_local_xy_m
    )
    detector_endpoints = optical_axis + np.einsum(
        "ij,nkj->nki", pixel_map, world_endpoints
    )
    endpoint_v = (detector_endpoints - origin) @ tangent
    tangent_half = 0.5 * np.abs(endpoint_v[:, 1] - endpoint_v[:, 0]) * float(
        integration_fraction
    )
    if np.any(tangent_half < max(2.0, 4.0 * tangent_step_px)):
        raise ValueError("Projected boundary lines are too short for stable integration.")

    identity_half = 0.45 * half
    # First pass: nominal, non-overlapping windows determine the interface
    # identity, centre, polarity, and a nuisance edge width.  A cubic
    # background absorbs the illumination envelope, which is essential for a
    # weak material shoulder on a broad Gaussian probe.
    provisional = LayerBoundaryRegistration2D(
        image_shape_yx=image.shape,
        detector_normal_xy=normal,
        detector_tangent_xy=tangent,
        coordinate_origin_xy_px=origin,
        detector_optical_axis_xy_px=optical_axis,
        search_centres_axis_px=nominal_u,
        search_half_widths_px=half,
        reference_peak_centres_axis_px=nominal_u,
        peak_tracking_half_widths_px=identity_half,
        edge_model_widths_px=np.ones(geometry.n_boundaries),
        tangent_centres_axis_px=nominal_v,
        tangent_half_widths_px=tangent_half,
        edge_polarities=np.ones(geometry.n_boundaries),
        smoothing_sigma_px=smoothing_sigma_px,
        profile_step_px=profile_step_px,
        tangent_step_px=tangent_step_px,
        minimum_peak_snr=0.0,
        minimum_step_improvement=0.0,
    )
    centres: list[float] = []
    polarities: list[float] = []
    model_widths: list[float] = []
    width_candidates = np.unique(
        np.clip(
            np.asarray(
                (
                    max(0.35, 2.0 * float(profile_step_px)),
                    0.5,
                    0.75,
                    1.0,
                    1.5,
                    2.0,
                    3.0,
                    4.0,
                    6.0,
                )
            ),
            max(float(profile_step_px), 0.1),
            max(0.5, float(np.max(identity_half))),
        )
    )
    for index in range(geometry.n_boundaries):
        u, profile, _derivative = _edge_profile(image, provisional, index)
        contrast = float(np.ptp(profile)) / max(
            float(np.median(np.abs(profile))), np.finfo(float).tiny
        )
        if contrast < reference_contrast_gate:
            raise UnresolvedLayerBoundaryError(
                index,
                geometry.boundary_labels[index],
                f"reference contrast fraction {contrast:.3g} is below "
                f"{reference_contrast_gate:.3g}",
            )
        try:
            centre, model_width, polarity, signed_snr, improvement, _amplitude = (
                _fit_polynomial_step(
                u,
                profile,
                target_axis_px=nominal_u[index],
                tracking_half_width_px=identity_half[index],
                model_widths_px=width_candidates,
                expected_polarity=None,
                )
            )
        except ValueError as error:
            raise UnresolvedLayerBoundaryError(
                index, geometry.boundary_labels[index], str(error)
            ) from error
        if signed_snr < float(minimum_peak_snr):
            raise UnresolvedLayerBoundaryError(
                index,
                geometry.boundary_labels[index],
                f"reference peak SNR {signed_snr:.3g} is below the frozen "
                f"{float(minimum_peak_snr):.3g} gate",
            )
        if improvement < improvement_gate:
            raise UnresolvedLayerBoundaryError(
                index,
                geometry.boundary_labels[index],
                f"reference step-model improvement {improvement:.3g} is below "
                f"the frozen {improvement_gate:.3g} gate",
            )
        offset = abs(centre - nominal_u[index])
        if offset > maximum_offset_fraction * half[index]:
            raise UnresolvedLayerBoundaryError(
                index,
                geometry.boundary_labels[index],
                f"reference peak offset {offset:.3g} px exceeds "
                f"{maximum_offset_fraction:.3g} of its association half-width",
            )
        centres.append(centre)
        polarities.append(polarity)
        model_widths.append(model_width)
    centres_array = np.asarray(centres)
    if np.any(np.diff(centres_array) <= 0.0):
        raise ValueError("Reference layer peaks are not ordered; registration is ambiguous.")
    tracking_half = np.minimum(
        np.maximum(1.0, 2.5 * float(smoothing_sigma_px)),
        half - np.abs(centres_array - nominal_u) - float(profile_step_px),
    )
    if np.any(tracking_half <= float(profile_step_px)):
        raise ValueError("Reference peaks leave no finite tracking window.")
    registration = LayerBoundaryRegistration2D(
        image_shape_yx=image.shape,
        detector_normal_xy=normal,
        detector_tangent_xy=tangent,
        coordinate_origin_xy_px=origin,
        detector_optical_axis_xy_px=optical_axis,
        search_centres_axis_px=nominal_u,
        search_half_widths_px=half,
        reference_peak_centres_axis_px=centres_array,
        peak_tracking_half_widths_px=tracking_half,
        edge_model_widths_px=np.asarray(model_widths),
        tangent_centres_axis_px=nominal_v,
        tangent_half_widths_px=tangent_half,
        edge_polarities=np.asarray(polarities),
        smoothing_sigma_px=smoothing_sigma_px,
        profile_step_px=profile_step_px,
        tangent_step_px=tangent_step_px,
        minimum_peak_snr=minimum_peak_snr,
        minimum_step_improvement=improvement_gate,
        extraction_rejection_minimum_peak_snr=(
            extraction_rejection_minimum_peak_snr
        ),
        extraction_rejection_minimum_step_improvement=(
            extraction_rejection_minimum_step_improvement
        ),
    )
    # Confirm that the final frozen estimator is numerically evaluable.  The
    # reference discovery gates were already applied above and are deliberately
    # not reused by this extraction call.
    extract_registered_layer_boundaries(image, geometry, registration)
    return registration


def _observable_names(
    geometry: CertifiedLayerGeometry2D,
    mode: LayerObservableMode,
) -> tuple[str, ...]:
    boundary = tuple(
        f"layer_boundary_{label}_axis_px" for label in geometry.boundary_labels
    )
    intervals = tuple(
        f"layer_interval_{label}_width_px" for label in geometry.interval_labels
    )
    if mode == "boundary_positions":
        return boundary
    if mode == "interval_widths":
        return intervals
    return boundary + intervals


@dataclass(frozen=True)
class RegisteredLayerMetrology:
    """A callable, bootstrap-safe detector observable extractor."""

    geometry: CertifiedLayerGeometry2D
    registration: LayerBoundaryRegistration2D
    observable_mode: LayerObservableMode = "interval_widths"

    def __post_init__(self) -> None:
        if self.geometry.n_boundaries != self.registration.n_boundaries:
            raise ValueError("Layer geometry and registration boundary counts differ.")
        object.__setattr__(self, "observable_mode", _validate_mode(self.observable_mode))

    @property
    def observable_names(self) -> tuple[str, ...]:
        return _observable_names(self.geometry, self.observable_mode)

    def extract(
        self,
        image_yx: object,
        *,
        enforce_quality_policy: bool = False,
    ) -> LayerBoundaryExtraction:
        return extract_registered_layer_boundaries(
            image_yx,
            self.geometry,
            self.registration,
            enforce_quality_policy=enforce_quality_policy,
        )

    def values(self, extraction: LayerBoundaryExtraction) -> np.ndarray:
        if self.observable_mode == "boundary_positions":
            return extraction.boundary_positions_axis_px
        if self.observable_mode == "interval_widths":
            return extraction.interval_widths_px
        return np.concatenate(
            (extraction.boundary_positions_axis_px, extraction.interval_widths_px)
        )

    def __call__(self, counts_yx: object, _request: object, noise_model: object):
        dark_offset = float(getattr(noise_model, "dark_offset_e", 0.0))
        corrected = np.asarray(counts_yx, dtype=float) - dark_offset
        extraction = self.extract(corrected)
        return self.values(extraction), self.observable_names

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema": "temgym_core.registered_layer_metrology",
            "version": 1,
            "geometry": self.geometry.to_metadata(),
            "registration": self.registration.to_metadata(),
            "observable_mode": self.observable_mode,
            "observable_names": list(self.observable_names),
        }

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> "RegisteredLayerMetrology":
        if metadata.get("schema") != "temgym_core.registered_layer_metrology":
            raise ValueError("Unknown registered-layer metrology schema.")
        if int(metadata.get("version", -1)) != 1:
            raise ValueError("Unsupported registered-layer metrology version.")
        geometry_metadata = metadata.get("geometry")
        registration_metadata = metadata.get("registration")
        if not isinstance(geometry_metadata, Mapping) or not isinstance(
            registration_metadata, Mapping
        ):
            raise ValueError("Layer metrology metadata lacks geometry or registration.")
        result = cls(
            CertifiedLayerGeometry2D.from_metadata(geometry_metadata),
            LayerBoundaryRegistration2D.from_metadata(registration_metadata),
            _validate_mode(str(metadata["observable_mode"])),
        )
        if tuple(metadata.get("observable_names", ())) != result.observable_names:
            raise ValueError("Layer metrology metadata observable schema is inconsistent.")
        return result


@dataclass(frozen=True)
class RegisteredLayerPeriodMetrology:
    """CTF-robust accumulated periods between like-transition interfaces.

    Defocus can move the apparent shoulder of opposite material transitions
    in opposite directions.  Individual adjacent widths then inherit that
    contrast-transfer bias even when every interface has excellent SNR.  A
    distance between interfaces two layers apart connects the same material
    transition, so a shared transition-class shoulder offset cancels to first
    order.  ``span_pairs`` index the frozen boundary registration and always
    connect certificate indices ``i`` and ``i + 2``.

    This is a distinct observable, not a fallback that silently relabels a
    failed individual-layer measurement.  Its schema and certificate
    covariance therefore remain explicit.
    """

    boundary_metrology: RegisteredLayerMetrology
    span_pairs: np.ndarray
    span_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        boundary = self.boundary_metrology
        if not isinstance(boundary, RegisteredLayerMetrology):
            raise TypeError("boundary_metrology must be RegisteredLayerMetrology.")
        if boundary.observable_mode != "boundary_positions":
            raise ValueError(
                "boundary_metrology must expose frozen boundary positions."
            )
        pairs = np.asarray(self.span_pairs, dtype=int)
        labels = tuple(_slug(value) for value in self.span_labels)
        count = boundary.geometry.n_boundaries
        if (
            pairs.ndim != 2
            or pairs.shape[1:] != (2,)
            or pairs.shape[0] != len(labels)
            or not len(labels)
            or len(set(labels)) != len(labels)
            or np.any(pairs < 0)
            or np.any(pairs >= count)
            or np.any(pairs[:, 1] <= pairs[:, 0])
        ):
            raise ValueError("span_pairs and span_labels are inconsistent.")
        indices = boundary.geometry.boundary_indices
        polarities = boundary.registration.edge_polarities
        for left, right in pairs:
            if right != left + 2 or indices[right] != indices[left] + 2:
                raise ValueError(
                    "Every accumulated period must connect certificate indices i and i+2."
                )
            if not (
                polarities[left] == polarities[right]
                and polarities[left] != polarities[left + 1]
            ):
                raise ValueError(
                    "Accumulated periods require a frozen alternating-polarity triplet."
                )
        pairs = pairs.copy()
        pairs.setflags(write=False)
        object.__setattr__(self, "span_pairs", pairs)
        object.__setattr__(self, "span_labels", labels)

    @property
    def geometry(self) -> CertifiedLayerGeometry2D:
        return self.boundary_metrology.geometry

    @property
    def registration(self) -> LayerBoundaryRegistration2D:
        return self.boundary_metrology.registration

    @property
    def observable_names(self) -> tuple[str, ...]:
        return tuple(f"layer_period_{label}_width_px" for label in self.span_labels)

    @property
    def span_widths_local_m(self) -> np.ndarray:
        x = self.geometry.boundary_points_local_xy_m[:, 0]
        values = x[self.span_pairs[:, 1]] - x[self.span_pairs[:, 0]]
        values = np.asarray(values, dtype=float)
        values.setflags(write=False)
        return values

    @property
    def span_certificate_covariance_local_m2(self) -> np.ndarray:
        incidence = np.zeros((len(self.span_pairs), self.geometry.n_boundaries))
        rows = np.arange(len(self.span_pairs))
        incidence[rows, self.span_pairs[:, 0]] = -1.0
        incidence[rows, self.span_pairs[:, 1]] = 1.0
        values = (
            incidence
            @ self.geometry.boundary_certificate_covariance_local_m2
            @ incidence.T
        )
        values = np.asarray(values, dtype=float)
        values.setflags(write=False)
        return values

    def extract(
        self,
        image_yx: object,
        *,
        enforce_quality_policy: bool = False,
    ) -> LayerBoundaryExtraction:
        return self.boundary_metrology.extract(
            image_yx,
            enforce_quality_policy=enforce_quality_policy,
        )

    def values(self, extraction: LayerBoundaryExtraction) -> np.ndarray:
        positions = extraction.boundary_positions_axis_px
        return positions[self.span_pairs[:, 1]] - positions[self.span_pairs[:, 0]]

    def __call__(self, counts_yx: object, _request: object, noise_model: object):
        dark_offset = float(getattr(noise_model, "dark_offset_e", 0.0))
        corrected = np.asarray(counts_yx, dtype=float) - dark_offset
        extraction = self.extract(corrected)
        return self.values(extraction), self.observable_names

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema": "temgym_core.registered_layer_period_metrology",
            "version": 1,
            "boundary_metrology": self.boundary_metrology.to_metadata(),
            "span_pairs": self.span_pairs.tolist(),
            "span_labels": list(self.span_labels),
            "observable_names": list(self.observable_names),
            "observable_role": (
                "same-transition two-layer spans; transition-class CTF offset "
                "cancels to first order"
            ),
        }

    @classmethod
    def from_metadata(
        cls, metadata: Mapping[str, object]
    ) -> "RegisteredLayerPeriodMetrology":
        if metadata.get("schema") != "temgym_core.registered_layer_period_metrology":
            raise ValueError("Unknown registered-layer-period metrology schema.")
        if int(metadata.get("version", -1)) != 1:
            raise ValueError("Unsupported registered-layer-period metrology version.")
        boundary_metadata = metadata.get("boundary_metrology")
        if not isinstance(boundary_metadata, Mapping):
            raise ValueError("Layer-period metadata lacks boundary_metrology.")
        result = cls(
            RegisteredLayerMetrology.from_metadata(boundary_metadata),
            np.asarray(metadata["span_pairs"], dtype=int),
            tuple(metadata["span_labels"]),  # type: ignore[arg-type]
        )
        if tuple(metadata.get("observable_names", ())) != result.observable_names:
            raise ValueError("Layer-period observable schema is inconsistent.")
        return result


def registered_layer_period_metrology_from_boundary_metrology(
    boundary_metrology: RegisteredLayerMetrology,
    *,
    minimum_period_spans: int = 2,
) -> RegisteredLayerPeriodMetrology:
    """Select certified like-transition spans from a frozen registration.

    Only consecutive registered rows whose *certificate* indices are
    ``i, i + 1, i + 2`` and whose fitted edge polarities alternate are
    admitted.  Thus a pruned or polarity-inconsistent edge can never be
    bridged accidentally.  The reference-image peak locations select the
    identities and polarities only; the predictor continues to use the
    certificate coordinates.
    """

    if not isinstance(boundary_metrology, RegisteredLayerMetrology):
        raise TypeError("boundary_metrology must be RegisteredLayerMetrology.")
    if boundary_metrology.observable_mode != "boundary_positions":
        raise ValueError(
            "boundary_metrology must expose frozen boundary positions."
        )
    required = int(minimum_period_spans)
    if required < 1 or required != minimum_period_spans:
        raise ValueError("minimum_period_spans must be a positive integer.")

    geometry = boundary_metrology.geometry
    polarities = boundary_metrology.registration.edge_polarities
    pairs: list[tuple[int, int]] = []
    labels: list[str] = []
    for left in range(geometry.n_boundaries - 2):
        middle = left + 1
        right = left + 2
        certificate_indices = geometry.boundary_indices[left : right + 1]
        if certificate_indices != tuple(
            range(certificate_indices[0], certificate_indices[0] + 3)
        ):
            continue
        if not (
            polarities[left] == polarities[right]
            and polarities[left] != polarities[middle]
        ):
            continue
        pairs.append((left, right))
        labels.append(
            f"{geometry.boundary_labels[left]}_to_"
            f"{geometry.boundary_labels[right]}"
        )

    if len(pairs) < required:
        raise UnresolvedLayerBoundaryError(
            -1,
            "same_transition_periods",
            f"only {len(pairs)} certified alternating-polarity spans remain; "
            f"at least {required} are required",
        )
    return RegisteredLayerPeriodMetrology(
        boundary_metrology,
        np.asarray(pairs, dtype=int),
        tuple(labels),
    )


def _layer_observable_incidence(
    metrology: RegisteredLayerMetrology,
) -> np.ndarray:
    count = metrology.geometry.n_boundaries
    identity = np.eye(count, dtype=float)
    pairs = metrology.geometry.interval_pairs
    interval = np.zeros((metrology.geometry.n_intervals, count), dtype=float)
    rows = np.arange(metrology.geometry.n_intervals)
    interval[rows, pairs[:, 0]] = -1.0
    interval[rows, pairs[:, 1]] = 1.0
    if metrology.observable_mode == "boundary_positions":
        return identity
    if metrology.observable_mode == "interval_widths":
        return interval
    return np.vstack((identity, interval))


def registered_layer_reference_residuals_px(
    metrology: RegisteredLayerMetrology,
) -> np.ndarray:
    """Return reference-image estimator offsets in the observable schema.

    This is a diagnostic, not an inverse target.  In a matched synthetic or
    independently validated forward model, its magnitude may be declared as a
    model-discrepancy uncertainty via
    :func:`layer_observable_covariance_augmentation`.  It must not be estimated
    against an unknown real column and then used to suppress the same optical
    residual being fitted.
    """

    boundary_residual = (
        metrology.registration.reference_peak_centres_axis_px
        - metrology.registration.search_centres_axis_px
    )
    values = _layer_observable_incidence(metrology) @ boundary_residual
    values = np.asarray(values, dtype=float)
    values.setflags(write=False)
    return values


@dataclass(frozen=True)
class LayerObservableCovarianceAugmentation:
    """Non-detector covariance carried by one layer-observable block.

    Certificate uncertainty and validated forward/estimator discrepancy are
    stored separately so repeated exposures can pool detector noise without
    spuriously reducing these shared terms.
    """

    observable_names: tuple[str, ...]
    certificate_covariance_px2: np.ndarray
    validated_model_discrepancy_covariance_px2: np.ndarray
    reference_residuals_px: np.ndarray
    model_discrepancy_provenance: str | None = None

    def __post_init__(self) -> None:
        names = tuple(str(value) for value in self.observable_names)
        if not names or len(names) != len(set(names)):
            raise ValueError("Layer covariance observable names must be unique.")
        count = len(names)
        certificate = _validated_covariance(
            "certificate_covariance_px2",
            self.certificate_covariance_px2,
            count,
        )
        discrepancy = _validated_covariance(
            "validated_model_discrepancy_covariance_px2",
            self.validated_model_discrepancy_covariance_px2,
            count,
        )
        residuals = _immutable_array(
            "reference_residuals_px", self.reference_residuals_px, ndim=1
        )
        if residuals.shape != (count,):
            raise ValueError("reference_residuals_px must match observable_names.")
        provenance = self.model_discrepancy_provenance
        if np.any(discrepancy) and not str(provenance or "").strip():
            raise ValueError(
                "A non-zero model-discrepancy covariance requires provenance."
            )
        object.__setattr__(self, "observable_names", names)
        object.__setattr__(self, "certificate_covariance_px2", certificate)
        object.__setattr__(
            self,
            "validated_model_discrepancy_covariance_px2",
            discrepancy,
        )
        object.__setattr__(self, "reference_residuals_px", residuals)
        object.__setattr__(
            self,
            "model_discrepancy_provenance",
            None if provenance is None else str(provenance).strip(),
        )

    @property
    def total_covariance_px2(self) -> np.ndarray:
        values = (
            self.certificate_covariance_px2
            + self.validated_model_discrepancy_covariance_px2
        )
        values = np.asarray(values, dtype=float)
        values.setflags(write=False)
        return values

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema": "temgym_core.layer_observable_covariance_augmentation",
            "version": 1,
            "observable_names": list(self.observable_names),
            "certificate_covariance_px2": self.certificate_covariance_px2.tolist(),
            "validated_model_discrepancy_covariance_px2": (
                self.validated_model_discrepancy_covariance_px2.tolist()
            ),
            "total_covariance_px2": self.total_covariance_px2.tolist(),
            "reference_residuals_px": self.reference_residuals_px.tolist(),
            "model_discrepancy_provenance": self.model_discrepancy_provenance,
            "shared_across_same_specimen_repeats": True,
            "cross_state_certificate_correlation": (
                "retained_as_metadata; block-diagonal fit cannot encode it"
            ),
        }


def _validated_covariance(name: str, values: object, count: int) -> np.ndarray:
    covariance = np.asarray(values, dtype=float)
    if covariance.shape != (count, count) or not np.all(np.isfinite(covariance)):
        raise ValueError(f"{name} must be a finite ({count}, {count}) matrix.")
    scale = max(float(np.max(np.abs(covariance), initial=0.0)), 1.0)
    if np.max(np.abs(covariance - covariance.T), initial=0.0) > 1e-10 * scale:
        raise ValueError(f"{name} must be symmetric.")
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if float(np.min(eigenvalues, initial=0.0)) < -1e-10 * scale:
        raise ValueError(f"{name} must be positive semidefinite.")
    covariance = covariance.copy()
    covariance.setflags(write=False)
    return covariance


def layer_observable_covariance_augmentation(
    metrology: RegisteredLayerMetrology,
    *,
    validated_model_discrepancy_standard_uncertainty_px: object | None = None,
    validated_model_discrepancy_covariance_px2: object | None = None,
    model_discrepancy_provenance: str | None = None,
) -> LayerObservableCovarianceAugmentation:
    """Propagate certificate and optional validated discrepancy uncertainty.

    The nominal detector-axis scale is inferred from the certified local
    coordinates and their frozen first-order detector positions.  Certificate
    boundary covariance is then pushed through the exact boundary/interval
    incidence matrix.  Supply at most one discrepancy representation; it must
    come from an independent forward-model validation or a matched synthetic
    truth calculation and requires a provenance string.
    """

    if (
        validated_model_discrepancy_standard_uncertainty_px is not None
        and validated_model_discrepancy_covariance_px2 is not None
    ):
        raise ValueError(
            "Supply model discrepancy as standard uncertainty or covariance, not both."
        )
    local_x = metrology.geometry.boundary_points_local_xy_m[:, 0]
    nominal_u = metrology.registration.search_centres_axis_px
    axis_scales = np.diff(nominal_u) / np.diff(local_x)
    axis_scale = float(np.median(axis_scales))
    tolerance = 1e-7 * max(abs(axis_scale), np.finfo(float).tiny)
    if (
        not np.isfinite(axis_scale)
        or axis_scale <= 0.0
        or np.max(np.abs(axis_scales - axis_scale), initial=0.0) > tolerance
    ):
        raise ValueError(
            "Frozen layer positions do not define one linear detector-axis scale."
        )
    incidence = _layer_observable_incidence(metrology)
    boundary_covariance_px2 = (
        axis_scale**2
        * metrology.geometry.boundary_certificate_covariance_local_m2
    )
    certificate = incidence @ boundary_covariance_px2 @ incidence.T
    count = len(metrology.observable_names)
    discrepancy = np.zeros((count, count), dtype=float)
    if validated_model_discrepancy_standard_uncertainty_px is not None:
        standard = np.asarray(
            validated_model_discrepancy_standard_uncertainty_px, dtype=float
        )
        if standard.ndim == 0:
            standard = np.full(count, float(standard))
        if standard.shape != (count,) or np.any(~np.isfinite(standard)) or np.any(
            standard < 0.0
        ):
            raise ValueError(
                "Validated model-discrepancy standard uncertainty must be a "
                "non-negative scalar or one value per layer observable."
            )
        discrepancy = np.diag(standard**2)
    elif validated_model_discrepancy_covariance_px2 is not None:
        discrepancy = np.asarray(
            validated_model_discrepancy_covariance_px2, dtype=float
        )
    return LayerObservableCovarianceAugmentation(
        observable_names=metrology.observable_names,
        certificate_covariance_px2=certificate,
        validated_model_discrepancy_covariance_px2=discrepancy,
        reference_residuals_px=registered_layer_reference_residuals_px(metrology),
        model_discrepancy_provenance=model_discrepancy_provenance,
    )


def registered_layer_metrology_from_sample_view(
    atlas: object,
    view: object,
    reference_image_yx: object,
    *,
    detector_position_map_xy_m_per_m: object,
    detector_pixel_size_yx_m: Sequence[float],
    observable_mode: LayerObservableMode = "interval_widths",
    region_name: str | None = None,
    minimum_visible_intervals: int = 2,
    detector_margin_px: float = 2.0,
    minimum_visible_line_length_px: float = 8.0,
    detector_optical_axis_xy_px: Sequence[float] | None = None,
    coordinate_origin_xy_px: Sequence[float] | None = None,
    search_half_width_px: float | Sequence[float] | None = None,
    integration_fraction: float = 0.6,
    smoothing_sigma_px: float = 0.8,
    profile_step_px: float = 0.125,
    tangent_step_px: float = 0.5,
    minimum_peak_snr: float = 2.0,
    minimum_reference_contrast_fraction: float = 1e-5,
    minimum_step_improvement: float = 0.02,
    maximum_reference_offset_fraction: float = 0.35,
    extraction_rejection_minimum_peak_snr: float | None = None,
    extraction_rejection_minimum_step_improvement: float | None = None,
    prune_unresolved_reference_boundaries: bool = True,
) -> RegisteredLayerMetrology:
    """Build the complete certified layer observable for one frozen frame.

    This convenience constructor performs view clipping, exact finite-line
    clipping against the detector, and reference peak registration in that
    order.  It is the recommended checkpoint/notebook entry point.
    """

    mode = _validate_mode(observable_mode)
    image = np.asarray(reference_image_yx, dtype=float)
    geometry = derive_certified_layer_geometry(
        atlas,
        view,
        region_name=region_name,
        clip_to_view=True,
        minimum_visible_intervals=minimum_visible_intervals,
    )
    # Reserve the complete peak-association excursion before deciding which
    # finite boundary lines are detector-visible.  Without this extra margin,
    # an oblique line can intersect a detector corner at its nominal position
    # but leave the chip within the frozen u search window.
    if search_half_width_px is None:
        pixel_map = _detector_pixel_map(
            detector_position_map_xy_m_per_m, detector_pixel_size_yx_m
        )
        local_to_pixel = pixel_map @ geometry.sample_projection_world_from_local
        tangent_vector = local_to_pixel @ np.asarray((0.0, 1.0))
        tangent = tangent_vector / np.linalg.norm(tangent_vector)
        normal = np.asarray((tangent[1], -tangent[0]))
        if float(normal @ (local_to_pixel @ np.asarray((1.0, 0.0)))) < 0.0:
            normal = -normal
        projected = np.einsum(
            "ij,nj->ni",
            local_to_pixel,
            geometry.boundary_points_local_xy_m,
        ) @ normal
        gaps = np.diff(projected)
        nearest_gap = np.minimum(
            np.concatenate(([gaps[0]], gaps)),
            np.concatenate((gaps, [gaps[-1]])),
        )
        association_half_width = float(np.max(0.36 * nearest_gap))
    else:
        supplied_half_width = np.asarray(search_half_width_px, dtype=float)
        association_half_width = float(np.max(supplied_half_width))
    effective_detector_margin = max(
        float(detector_margin_px), 2.0 + association_half_width
    )
    geometry_before_detector_clip = geometry
    geometry = clip_certified_layer_geometry_to_detector(
        geometry,
        detector_position_map_xy_m_per_m=detector_position_map_xy_m_per_m,
        detector_pixel_size_yx_m=detector_pixel_size_yx_m,
        image_shape_yx=image.shape,
        detector_optical_axis_xy_px=detector_optical_axis_xy_px,
        detector_margin_px=effective_detector_margin,
        minimum_visible_line_length_px=minimum_visible_line_length_px,
        minimum_visible_intervals=minimum_visible_intervals,
    )
    current_search_half_width = search_half_width_px
    if current_search_half_width is not None:
        supplied_widths = np.asarray(current_search_half_width, dtype=float)
        if supplied_widths.ndim == 1:
            if supplied_widths.shape != (
                geometry_before_detector_clip.n_boundaries,
            ):
                raise ValueError(
                    "A vector search_half_width_px must match the pre-clip "
                    "view boundary count."
                )
            width_by_boundary = dict(
                zip(
                    geometry_before_detector_clip.boundary_indices,
                    supplied_widths,
                    strict=True,
                )
            )
            current_search_half_width = np.asarray(
                [width_by_boundary[index] for index in geometry.boundary_indices]
            )
    while True:
        try:
            registration = freeze_layer_boundary_registration(
                image,
                geometry,
                detector_position_map_xy_m_per_m=detector_position_map_xy_m_per_m,
                detector_pixel_size_yx_m=detector_pixel_size_yx_m,
                detector_optical_axis_xy_px=detector_optical_axis_xy_px,
                coordinate_origin_xy_px=coordinate_origin_xy_px,
                search_half_width_px=current_search_half_width,
                integration_fraction=integration_fraction,
                smoothing_sigma_px=smoothing_sigma_px,
                profile_step_px=profile_step_px,
                tangent_step_px=tangent_step_px,
                minimum_peak_snr=minimum_peak_snr,
                minimum_reference_contrast_fraction=(
                    minimum_reference_contrast_fraction
                ),
                minimum_step_improvement=minimum_step_improvement,
                maximum_reference_offset_fraction=(
                    maximum_reference_offset_fraction
                ),
                extraction_rejection_minimum_peak_snr=(
                    extraction_rejection_minimum_peak_snr
                ),
                extraction_rejection_minimum_step_improvement=(
                    extraction_rejection_minimum_step_improvement
                ),
            )
            break
        except UnresolvedLayerBoundaryError as error:
            if not prune_unresolved_reference_boundaries:
                raise
            retained = tuple(
                row
                for row in range(geometry.n_boundaries)
                if row != error.boundary_index
            )
            if current_search_half_width is not None:
                current_widths = np.asarray(current_search_half_width, dtype=float)
                if current_widths.ndim == 1:
                    current_search_half_width = current_widths[
                        np.asarray(retained, dtype=int)
                    ]
            geometry = _subset_layer_geometry(
                geometry,
                retained,
                minimum_visible_intervals=minimum_visible_intervals,
            )
            if mode == "interval_widths":
                used_rows = tuple(
                    sorted(set(int(value) for value in geometry.interval_pairs.ravel()))
                )
                if len(used_rows) != geometry.n_boundaries:
                    if current_search_half_width is not None:
                        current_widths = np.asarray(
                            current_search_half_width, dtype=float
                        )
                        if current_widths.ndim == 1:
                            current_search_half_width = current_widths[
                                np.asarray(used_rows, dtype=int)
                            ]
                    geometry = _subset_layer_geometry(
                        geometry,
                        used_rows,
                        minimum_visible_intervals=minimum_visible_intervals,
                    )
    return RegisteredLayerMetrology(
        geometry,
        registration,
        mode,
    )


def registered_layer_period_metrology_from_sample_view(
    atlas: object,
    view: object,
    reference_image_yx: object,
    *,
    detector_position_map_xy_m_per_m: object,
    detector_pixel_size_yx_m: Sequence[float],
    minimum_period_spans: int = 2,
    maximum_reference_offset_fraction: float = 0.499,
    **registration_options: object,
) -> RegisteredLayerPeriodMetrology:
    """Build a CTF-robust, certificate-backed accumulated-period extractor.

    This route is intended for a deliberately defocused image in which
    individual opposite-polarity transition shoulders are displaced but
    remain unambiguously inside their non-overlapping identity windows.  It
    keeps the normal SNR and step-improvement discovery gates supplied through
    ``registration_options``.  The only different default is the association
    offset limit: 0.499 of the already non-overlapping half-width, rather than
    0.35.  No unresolved boundary is pruned; admissible two-layer spans are
    selected explicitly by certificate adjacency and alternating polarity.

    Remaining keyword arguments are those of
    :func:`registered_layer_metrology_from_sample_view`, except
    ``observable_mode`` and ``prune_unresolved_reference_boundaries``, which
    are fixed by this route.
    """

    forbidden = {
        "observable_mode",
        "prune_unresolved_reference_boundaries",
        "maximum_reference_offset_fraction",
    }.intersection(registration_options)
    if forbidden:
        raise TypeError(
            "Period metrology fixes these registration options: "
            + ", ".join(sorted(forbidden))
        )
    boundary_metrology = registered_layer_metrology_from_sample_view(
        atlas,
        view,
        reference_image_yx,
        detector_position_map_xy_m_per_m=detector_position_map_xy_m_per_m,
        detector_pixel_size_yx_m=detector_pixel_size_yx_m,
        observable_mode="boundary_positions",
        maximum_reference_offset_fraction=maximum_reference_offset_fraction,
        prune_unresolved_reference_boundaries=False,
        **registration_options,
    )
    return registered_layer_period_metrology_from_boundary_metrology(
        boundary_metrology,
        minimum_period_spans=minimum_period_spans,
    )


def predict_registered_layer_observables(
    detector_position_map_xy_m_per_m: object,
    detector_pixel_size_yx_m: Sequence[float],
    metrology: RegisteredLayerMetrology,
    *,
    sample_translation_xy_m: object | None = None,
    sample_projection_world_from_local: object | None = None,
    boundary_coordinates_local_x_m: object | None = None,
) -> jnp.ndarray:
    """Predict the frozen scalar layer observables with JAX operations.

    ``detector_position_map_xy_m_per_m`` is ``H = detector_affine @ A``, where
    ``A`` is the sample-position to detector-position block.  The optional
    pose arguments make the operation differentiable with respect to sample
    translation, rotation/tilt projection, and ``H`` itself.  The physical
    hard-edge fit normally fixes the certified pose and differentiates only
    through ``H``.  ``boundary_coordinates_local_x_m`` may replace the nominal
    certificate coordinates while retaining the frozen boundary identities,
    local ``y`` anchors, and specimen pose.  This is the shared-certificate
    latent path used by cross-state inversion; it is deliberately indexed in
    physical specimen coordinates rather than detector pixels.
    """

    h = jnp.asarray(detector_position_map_xy_m_per_m)
    if h.shape != (2, 2):
        raise ValueError("detector_position_map_xy_m_per_m must have shape (2, 2).")
    pixel_yx = jnp.asarray(detector_pixel_size_yx_m)
    if pixel_yx.shape != (2,):
        raise ValueError("detector_pixel_size_yx_m must have shape (2,).")
    pixel_xy = pixel_yx[::-1]
    translation = jnp.asarray(
        metrology.geometry.sample_translation_xy_m
        if sample_translation_xy_m is None
        else sample_translation_xy_m
    )
    projection = jnp.asarray(
        metrology.geometry.sample_projection_world_from_local
        if sample_projection_world_from_local is None
        else sample_projection_world_from_local
    )
    if translation.shape != (2,) or projection.shape != (2, 2):
        raise ValueError("Sample translation/projection must have shapes (2,) and (2,2).")
    local_points = jnp.asarray(metrology.geometry.boundary_points_local_xy_m)
    if boundary_coordinates_local_x_m is not None:
        boundary_x = jnp.asarray(boundary_coordinates_local_x_m)
        if boundary_x.shape != (metrology.geometry.n_boundaries,):
            raise ValueError(
                "boundary_coordinates_local_x_m must contain one coordinate "
                "per retained certificate boundary."
            )
        local_points = local_points.at[:, 0].set(boundary_x)
    world_points = translation + jnp.einsum("ij,nj->ni", projection, local_points)
    detector_displacements_px = jnp.einsum("ij,nj->ni", h, world_points) / pixel_xy
    optical_offset = jnp.asarray(
        metrology.registration.detector_optical_axis_xy_px
        - metrology.registration.coordinate_origin_xy_px
    )
    detector_points_relative = detector_displacements_px + optical_offset
    normal = jnp.asarray(metrology.registration.detector_normal_xy)
    positions = detector_points_relative @ normal
    pairs = jnp.asarray(metrology.geometry.interval_pairs)
    widths = positions[pairs[:, 1]] - positions[pairs[:, 0]]
    if metrology.observable_mode == "boundary_positions":
        return positions
    if metrology.observable_mode == "interval_widths":
        return widths
    return jnp.concatenate((positions, widths))


def predict_registered_layer_period_observables(
    detector_position_map_xy_m_per_m: object,
    detector_pixel_size_yx_m: Sequence[float],
    metrology: RegisteredLayerPeriodMetrology,
    *,
    sample_translation_xy_m: object | None = None,
    sample_projection_world_from_local: object | None = None,
    boundary_coordinates_local_x_m: object | None = None,
) -> jnp.ndarray:
    """Predict frozen same-transition two-layer spans with JAX operations."""

    if not isinstance(metrology, RegisteredLayerPeriodMetrology):
        raise TypeError("metrology must be RegisteredLayerPeriodMetrology.")
    positions = predict_registered_layer_observables(
        detector_position_map_xy_m_per_m,
        detector_pixel_size_yx_m,
        metrology.boundary_metrology,
        sample_translation_xy_m=sample_translation_xy_m,
        sample_projection_world_from_local=sample_projection_world_from_local,
        boundary_coordinates_local_x_m=boundary_coordinates_local_x_m,
    )
    pairs = jnp.asarray(metrology.span_pairs)
    return positions[pairs[:, 1]] - positions[pairs[:, 0]]


__all__ = (
    "CertifiedLayerGeometry2D",
    "BoundaryUncertaintyModel",
    "LayerBoundaryExtraction",
    "LayerBoundaryRegistration2D",
    "LayerObservableCovarianceAugmentation",
    "LayerObservableMode",
    "RegisteredLayerMetrology",
    "RegisteredLayerPeriodMetrology",
    "UnresolvedLayerBoundaryError",
    "clip_certified_layer_geometry_to_detector",
    "derive_certified_layer_geometry",
    "extract_registered_layer_boundaries",
    "freeze_layer_boundary_registration",
    "layer_observable_covariance_augmentation",
    "predict_registered_layer_observables",
    "predict_registered_layer_period_observables",
    "registered_layer_reference_residuals_px",
    "registered_layer_metrology_from_sample_view",
    "registered_layer_period_metrology_from_boundary_metrology",
    "registered_layer_period_metrology_from_sample_view",
)
