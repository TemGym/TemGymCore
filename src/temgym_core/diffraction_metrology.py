"""Registered Bragg-spot metrology for NEOARM diffraction images.

The first-order optical predictor maps a certified specimen reciprocal vector
``g`` to a signed detector displacement ``D B (lambda g)``.  This module does
the complementary image-side operation: it measures that displacement from a
pair of Friedel-related spots.  Using half the positive/negative centroid
difference cancels direct-beam drift and avoids treating the Fourier transform
of a diffraction pattern as though it were a Bragg-spot coordinate.

Reference offsets are used only to choose fixed, non-overlapping registration
windows.  The measured centroids inside those windows are the observations.
This makes the function suitable for parametric detector-noise bootstrap: the
same windows and observable schema are reused for every realization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


class DiffractionSpotRegistrationError(RuntimeError):
    """A declared Bragg spot cannot be measured in its frozen search window."""


def _readonly_vectors(name: str, values: object) -> np.ndarray:
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


@dataclass(frozen=True)
class RegisteredBraggSpotMeasurement:
    """Friedel-pair displacements in detector ``(y, x)`` pixel order."""

    labels: tuple[str, ...]
    offsets_yx_px: np.ndarray
    positive_centroids_yx_px: np.ndarray
    negative_centroids_yx_px: np.ndarray
    direct_beam_centroid_yx_px: np.ndarray
    observable_names: tuple[str, ...]
    registration_offsets_yx_px: np.ndarray
    search_radius_px: float

    def __post_init__(self) -> None:
        labels = tuple(str(value).strip() for value in self.labels)
        if not labels or any(not value for value in labels) or len(set(labels)) != len(labels):
            raise ValueError("labels must be non-empty and unique.")
        count = len(labels)
        for name in (
            "offsets_yx_px",
            "positive_centroids_yx_px",
            "negative_centroids_yx_px",
            "registration_offsets_yx_px",
        ):
            array = np.asarray(getattr(self, name), dtype=float)
            if array.shape != (count, 2) or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite with shape ({count}, 2).")
            array = array.copy()
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        direct = np.asarray(self.direct_beam_centroid_yx_px, dtype=float)
        if direct.shape != (2,) or not np.all(np.isfinite(direct)):
            raise ValueError("direct_beam_centroid_yx_px must contain two finite values.")
        direct = direct.copy()
        direct.setflags(write=False)
        names = tuple(str(value) for value in self.observable_names)
        expected = tuple(
            name
            for label in labels
            for name in (f"bragg_{label}_y_px", f"bragg_{label}_x_px")
        )
        if names != expected:
            raise ValueError("observable_names do not match the labelled Bragg schema.")
        radius = float(self.search_radius_px)
        if not np.isfinite(radius) or radius <= 1.0:
            raise ValueError("search_radius_px must be finite and greater than one.")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "direct_beam_centroid_yx_px", direct)
        object.__setattr__(self, "observable_names", names)
        object.__setattr__(self, "search_radius_px", radius)

    @property
    def values(self) -> np.ndarray:
        """Return the inverse-observable vector in declared name order."""

        return self.offsets_yx_px.reshape(-1).copy()


def _window_centroid(
    image: np.ndarray,
    centre_yx: np.ndarray,
    radius_px: float,
    *,
    background_annulus_fraction: float,
    minimum_signal: float,
) -> np.ndarray:
    ny, nx = image.shape
    radius = float(radius_px)
    y0, x0 = (float(value) for value in centre_yx)
    lower_y = max(0, int(np.floor(y0 - radius)))
    upper_y = min(ny, int(np.ceil(y0 + radius)) + 1)
    lower_x = max(0, int(np.floor(x0 - radius)))
    upper_x = min(nx, int(np.ceil(x0 + radius)) + 1)
    if lower_y == 0 or lower_x == 0 or upper_y == ny or upper_x == nx:
        raise DiffractionSpotRegistrationError(
            "A frozen Bragg registration window intersects the detector edge."
        )
    yy, xx = np.meshgrid(
        np.arange(lower_y, upper_y, dtype=float),
        np.arange(lower_x, upper_x, dtype=float),
        indexing="ij",
    )
    distance = np.hypot(yy - y0, xx - x0)
    aperture = distance <= radius
    annulus = (distance >= background_annulus_fraction * radius) & aperture
    if not np.any(annulus):
        raise DiffractionSpotRegistrationError("The background annulus is empty.")
    patch = image[lower_y:upper_y, lower_x:upper_x]
    background = float(np.median(patch[annulus]))
    weights = np.maximum(patch - background, 0.0) * aperture
    signal = float(np.sum(weights))
    if not np.isfinite(signal) or signal <= minimum_signal:
        raise DiffractionSpotRegistrationError(
            "A Bragg registration window contains insufficient positive signal."
        )
    return np.asarray((np.sum(weights * yy) / signal, np.sum(weights * xx) / signal))


def extract_registered_bragg_spot_offsets(
    image_yx: np.ndarray,
    registration_offsets_yx_px: np.ndarray,
    labels: Sequence[str],
    *,
    search_radius_px: float = 5.0,
    direct_beam_radius_px: float | None = None,
    detector_centre_yx_px: Sequence[float] | None = None,
    background_annulus_fraction: float = 0.72,
    minimum_signal: float = 1e-12,
) -> RegisteredBraggSpotMeasurement:
    """Measure indexed spot vectors with frozen positive/negative windows.

    ``registration_offsets_yx_px`` contains expected signed offsets from the
    direct beam and is never returned as measured data.  Each output is
    ``(centroid(+g) - centroid(-g)) / 2``; the separately measured direct-beam
    centroid is retained only for diagnostics.  The image may contain read
    noise and therefore need not be non-negative.
    """

    image = np.asarray(image_yx, dtype=float)
    if image.ndim != 2 or not np.all(np.isfinite(image)):
        raise ValueError("image_yx must be a finite two-dimensional array.")
    offsets = _readonly_vectors("registration_offsets_yx_px", registration_offsets_yx_px)
    labels_tuple = tuple(str(value).strip() for value in labels)
    if len(labels_tuple) != offsets.shape[0]:
        raise ValueError("labels must match registration_offsets_yx_px.")
    radius = float(search_radius_px)
    direct_radius = radius if direct_beam_radius_px is None else float(direct_beam_radius_px)
    if (
        not np.isfinite(radius)
        or radius <= 1.0
        or not np.isfinite(direct_radius)
        or direct_radius <= 1.0
        or not np.isfinite(background_annulus_fraction)
        or not 0.5 < background_annulus_fraction < 1.0
        or not np.isfinite(minimum_signal)
        or minimum_signal < 0.0
    ):
        raise ValueError("Invalid Bragg registration-window parameters.")
    if detector_centre_yx_px is None:
        centre = 0.5 * (np.asarray(image.shape, dtype=float) - 1.0)
    else:
        centre = np.asarray(detector_centre_yx_px, dtype=float)
        if centre.shape != (2,) or not np.all(np.isfinite(centre)):
            raise ValueError("detector_centre_yx_px must contain two finite values.")

    direct = _window_centroid(
        image,
        centre,
        direct_radius,
        background_annulus_fraction=background_annulus_fraction,
        minimum_signal=minimum_signal,
    )
    positive = []
    negative = []
    for offset in offsets:
        positive.append(
            _window_centroid(
                image,
                centre + offset,
                radius,
                background_annulus_fraction=background_annulus_fraction,
                minimum_signal=minimum_signal,
            )
        )
        negative.append(
            _window_centroid(
                image,
                centre - offset,
                radius,
                background_annulus_fraction=background_annulus_fraction,
                minimum_signal=minimum_signal,
            )
        )
    positive_values = np.asarray(positive)
    negative_values = np.asarray(negative)
    measured = 0.5 * (positive_values - negative_values)
    names = tuple(
        name
        for label in labels_tuple
        for name in (f"bragg_{label}_y_px", f"bragg_{label}_x_px")
    )
    return RegisteredBraggSpotMeasurement(
        labels=labels_tuple,
        offsets_yx_px=measured,
        positive_centroids_yx_px=positive_values,
        negative_centroids_yx_px=negative_values,
        direct_beam_centroid_yx_px=direct,
        observable_names=names,
        registration_offsets_yx_px=offsets,
        search_radius_px=radius,
    )


__all__ = [
    "DiffractionSpotRegistrationError",
    "RegisteredBraggSpotMeasurement",
    "extract_registered_bragg_spot_offsets",
]
