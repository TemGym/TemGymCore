"""Multiscale, physically calibrated sample actions for Gaussian propagation.

The existing :class:`~temgym_core.components.InterpolatedSample2D` is a useful
compatibility wrapper around a complex transmission image.  This module uses a
more explicit representation for calibration targets: the two real quantities
that the Gaussian action actually needs, equivalent pathlength and log *field*
amplitude.  Keeping those channels separate avoids phase wrapping and zeros of
an interpolated complex transmission.

Raster inputs use the conventional ``(y, x)`` ordering.  ``interpax`` uses
``(x, y, ...)`` internally, so :class:`SampleView` performs that transpose in a
single, tested location.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from interpax import Interpolator2D
from scipy.ndimage import gaussian_filter
from scipy.interpolate import CubicSpline

from .components import GaussianActionComponent, _regularize_L2k
from .constants import energy2wavelength
from .gaussian import GaussianBeam, _solve_2x2, apply_action_delta
from .ray import Ray


BoundaryMode = Literal["vacuum", "constant", "periodic", "clamp", "nan"]
FeatureKind = Literal["gabor", "fiducial"]
RegionKind = Literal["perforation", "multilayer", "silicon_lattice"]


def _positive_pair(name: str, values: Sequence[float]) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two entries.")
    pair = (float(values[0]), float(values[1]))
    if not all(np.isfinite(pair)) or not all(value > 0.0 for value in pair):
        raise ValueError(f"{name} must contain finite positive values.")
    return pair


def _finite_pair(name: str, values: Sequence[float]) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two entries.")
    pair = (float(values[0]), float(values[1]))
    if not all(np.isfinite(pair)):
        raise ValueError(f"{name} must contain finite values.")
    return pair


def _indexed_triplet(name: str, values: Sequence[int]) -> tuple[int, int, int]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three Miller indices.")
    indices = tuple(int(value) for value in values)
    if any(float(value) != index for value, index in zip(values, indices, strict=True)):
        raise ValueError(f"{name} must contain integer Miller indices.")
    if indices == (0, 0, 0):
        raise ValueError(f"{name} cannot be the zero index.")
    return indices


def _wavelength_from_voltage_or_value(
    *, voltage: float | None, wavelength_m: float | None
) -> float:
    if (voltage is None) == (wavelength_m is None):
        raise ValueError("Provide exactly one of `voltage` or `wavelength_m`.")
    if voltage is not None:
        voltage = float(voltage)
        if not np.isfinite(voltage) or voltage <= 0.0:
            raise ValueError("voltage must be finite and positive.")
        wavelength_m = float(np.asarray(energy2wavelength(voltage)))
    assert wavelength_m is not None
    wavelength_m = float(wavelength_m)
    if not np.isfinite(wavelength_m) or wavelength_m <= 0.0:
        raise ValueError("wavelength_m must be finite and positive.")
    return wavelength_m


@dataclass(frozen=True)
class SampleLevel2D:
    """One rectilinear level of a physical sample-action pyramid.

    Arrays are stored in image order ``(ny, nx)``.  Coordinates locate pixel
    centres in the sample's local coordinate system and are expressed in
    metres.  ``log_amplitude_yx`` is the logarithm of *field* transmission.
    """

    pathlength_yx_m: np.ndarray
    log_amplitude_yx: np.ndarray
    x_coords_m: np.ndarray
    y_coords_m: np.ndarray
    name: str = ""

    def __post_init__(self) -> None:
        pathlength = np.asarray(self.pathlength_yx_m, dtype=np.float64)
        log_amplitude = np.asarray(self.log_amplitude_yx, dtype=np.float64)
        x = np.asarray(self.x_coords_m, dtype=np.float64)
        y = np.asarray(self.y_coords_m, dtype=np.float64)

        if pathlength.ndim != 2:
            raise ValueError("pathlength_yx_m must be a two-dimensional yx array.")
        if log_amplitude.shape != pathlength.shape:
            raise ValueError("pathlength and log-amplitude arrays must have equal shape.")
        if pathlength.shape != (y.size, x.size):
            raise ValueError(
                "yx array shape must equal (len(y_coords_m), len(x_coords_m))."
            )
        if x.size < 4 or y.size < 4:
            raise ValueError("C2 cubic interpolation requires at least four knots per axis.")
        if not (
            np.all(np.isfinite(pathlength))
            and np.all(np.isfinite(log_amplitude))
            and np.all(np.isfinite(x))
            and np.all(np.isfinite(y))
        ):
            raise ValueError("sample arrays and coordinates must be finite.")
        if not np.all(np.diff(x) > 0.0) or not np.all(np.diff(y) > 0.0):
            raise ValueError("sample coordinates must be strictly increasing.")
        dx = np.diff(x)
        dy = np.diff(y)
        if not np.allclose(dx, dx[0], rtol=1e-9, atol=0.0):
            raise ValueError("x_coords_m must be uniformly spaced.")
        if not np.allclose(dy, dy[0], rtol=1e-9, atol=0.0):
            raise ValueError("y_coords_m must be uniformly spaced.")

        # Host-side levels are immutable metadata.  Defensive copies prevent a
        # caller from silently changing the calibration target after creation.
        pathlength = pathlength.copy()
        log_amplitude = log_amplitude.copy()
        x = x.copy()
        y = y.copy()
        for array in (pathlength, log_amplitude, x, y):
            array.setflags(write=False)
        object.__setattr__(self, "pathlength_yx_m", pathlength)
        object.__setattr__(self, "log_amplitude_yx", log_amplitude)
        object.__setattr__(self, "x_coords_m", x)
        object.__setattr__(self, "y_coords_m", y)
        object.__setattr__(self, "name", str(self.name))

    @property
    def log_field_amplitude_yx(self) -> np.ndarray:
        """Explicit field-amplitude spelling used by the calibration API."""

        return self.log_amplitude_yx

    @classmethod
    def from_action_channels(
        cls,
        pathlength_yx_m,
        log_amplitude_yx,
        *,
        pixel_size_yx_m: Sequence[float],
        centre_xy_m: Sequence[float] = (0.0, 0.0),
        name: str = "",
    ) -> "SampleLevel2D":
        """Build a uniformly sampled level from conventional yx arrays."""

        pathlength = np.asarray(pathlength_yx_m, dtype=np.float64)
        if pathlength.ndim != 2:
            raise ValueError("pathlength_yx_m must be a two-dimensional yx array.")
        dy, dx = _positive_pair("pixel_size_yx_m", pixel_size_yx_m)
        cx, cy = _finite_pair("centre_xy_m", centre_xy_m)
        ny, nx = pathlength.shape
        x = cx + (np.arange(nx, dtype=np.float64) - 0.5 * (nx - 1)) * dx
        y = cy + (np.arange(ny, dtype=np.float64) - 0.5 * (ny - 1)) * dy
        return cls(pathlength, log_amplitude_yx, x, y, name=name)

    @classmethod
    def from_pathlength_and_log_amplitude(
        cls,
        pathlength_yx_m,
        log_amplitude_yx,
        *,
        pixel_size_yx_m: Sequence[float],
        centre_xy_m: Sequence[float] = (0.0, 0.0),
        name: str = "",
    ) -> "SampleLevel2D":
        """Explicitly named primary constructor for Gaussian action data."""

        return cls.from_action_channels(
            pathlength_yx_m,
            log_amplitude_yx,
            pixel_size_yx_m=pixel_size_yx_m,
            centre_xy_m=centre_xy_m,
            name=name,
        )

    @classmethod
    def from_amplitude_phase(
        cls,
        amplitude_yx,
        phase_yx_rad,
        *,
        pixel_size_yx_m: Sequence[float],
        centre_xy_m: Sequence[float] = (0.0, 0.0),
        voltage: float | None = None,
        wavelength_m: float | None = None,
        amplitude_floor: float = 1e-12,
        name: str = "",
    ) -> "SampleLevel2D":
        """Build from field amplitude and an already-unwrapped phase raster."""

        amplitude = np.asarray(amplitude_yx, dtype=np.float64)
        phase = np.asarray(phase_yx_rad, dtype=np.float64)
        if amplitude.shape != phase.shape or amplitude.ndim != 2:
            raise ValueError("amplitude and phase must be equal-shape yx arrays.")
        floor = float(amplitude_floor)
        if not np.isfinite(floor) or floor <= 0.0:
            raise ValueError("amplitude_floor must be finite and positive.")
        if np.any(amplitude < 0.0) or not np.all(np.isfinite(amplitude)):
            raise ValueError("field amplitude must be finite and non-negative.")
        wavelength = _wavelength_from_voltage_or_value(
            voltage=voltage, wavelength_m=wavelength_m
        )
        pathlength = phase * wavelength / (2.0 * np.pi)
        log_amplitude = np.log(np.maximum(amplitude, floor))
        return cls.from_action_channels(
            pathlength,
            log_amplitude,
            pixel_size_yx_m=pixel_size_yx_m,
            centre_xy_m=centre_xy_m,
            name=name,
        )

    @classmethod
    def from_amplitude_and_unwrapped_phase(
        cls,
        amplitude_yx,
        phase_yx_rad,
        *,
        pixel_size_yx_m: Sequence[float],
        centre_xy_m: Sequence[float] = (0.0, 0.0),
        voltage: float | None = None,
        wavelength_m: float | None = None,
        amplitude_floor: float = 1e-12,
        name: str = "",
    ) -> "SampleLevel2D":
        """Primary constructor for explicitly unwrapped physical phase."""

        return cls.from_amplitude_phase(
            amplitude_yx,
            phase_yx_rad,
            pixel_size_yx_m=pixel_size_yx_m,
            centre_xy_m=centre_xy_m,
            voltage=voltage,
            wavelength_m=wavelength_m,
            amplitude_floor=amplitude_floor,
            name=name,
        )

    @classmethod
    def from_complex_transmission(
        cls,
        transmission_yx,
        *,
        pixel_size_yx_m: Sequence[float],
        centre_xy_m: Sequence[float] = (0.0, 0.0),
        voltage: float | None = None,
        wavelength_m: float | None = None,
        phase_unwrap: Literal["unwrap", "principal"],
        near_zero_threshold: float = 1e-8,
        amplitude_floor: float = 1e-12,
        name: str = "",
    ) -> "SampleLevel2D":
        """Build from complex transmission under an explicit phase policy.

        ``phase_unwrap='unwrap'`` applies sequential numpy unwrapping in x then
        y; ``'principal'`` deliberately retains phases in ``[-pi, pi]``.  A
        near-zero field has undefined phase and is rejected rather than being
        hidden by the logarithmic amplitude floor.
        """

        transmission = np.asarray(transmission_yx, dtype=np.complex128)
        if transmission.ndim != 2 or not np.all(np.isfinite(transmission)):
            raise ValueError("transmission_yx must be a finite two-dimensional array.")
        threshold = float(near_zero_threshold)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("near_zero_threshold must be finite and non-negative.")
        minimum = float(np.min(np.abs(transmission)))
        if minimum <= threshold:
            raise ValueError(
                "complex transmission is at or below near_zero_threshold; "
                "use an exact sample-plane multiplication/re-expansion route."
            )
        if phase_unwrap not in ("unwrap", "principal"):
            raise ValueError("phase_unwrap must be explicitly 'unwrap' or 'principal'.")
        phase = np.angle(transmission)
        if phase_unwrap == "unwrap":
            phase = np.unwrap(np.unwrap(phase, axis=1), axis=0)
        return cls.from_amplitude_phase(
            np.abs(transmission),
            phase,
            pixel_size_yx_m=pixel_size_yx_m,
            centre_xy_m=centre_xy_m,
            voltage=voltage,
            wavelength_m=wavelength_m,
            amplitude_floor=amplitude_floor,
            name=name,
        )

    @property
    def shape_yx(self) -> tuple[int, int]:
        return self.pathlength_yx_m.shape

    @property
    def pixel_size_yx_m(self) -> tuple[float, float]:
        return (
            float(self.y_coords_m[1] - self.y_coords_m[0]),
            float(self.x_coords_m[1] - self.x_coords_m[0]),
        )

    @property
    def max_spacing_m(self) -> float:
        return max(self.pixel_size_yx_m)

    @property
    def bounds_xyxy_m(self) -> tuple[float, float, float, float]:
        dy, dx = self.pixel_size_yx_m
        return (
            float(self.x_coords_m[0] - 0.5 * dx),
            float(self.x_coords_m[-1] + 0.5 * dx),
            float(self.y_coords_m[0] - 0.5 * dy),
            float(self.y_coords_m[-1] + 0.5 * dy),
        )


@dataclass(frozen=True)
class CertifiedFeature2D:
    """A known smooth feature used to certify a calibration atlas."""

    name: str
    kind: FeatureKind
    centre_xy_m: tuple[float, float]
    sigma_xy_m: tuple[float, float]
    orientation_rad: float = 0.0
    period_m: float | None = None
    carrier_phase_rad: float = 0.0
    phase_amplitude_rad: float = 0.0
    log_amplitude_depth: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("gabor", "fiducial"):
            raise ValueError("feature kind must be 'gabor' or 'fiducial'.")
        object.__setattr__(self, "centre_xy_m", _finite_pair("centre_xy_m", self.centre_xy_m))
        object.__setattr__(self, "sigma_xy_m", _positive_pair("sigma_xy_m", self.sigma_xy_m))
        for name in (
            "orientation_rad",
            "carrier_phase_rad",
            "phase_amplitude_rad",
            "log_amplitude_depth",
        ):
            if not np.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite.")
        if self.log_amplitude_depth < 0.0:
            raise ValueError("log_amplitude_depth must be non-negative.")
        if self.kind == "gabor":
            if self.period_m is None or not np.isfinite(self.period_m) or self.period_m <= 0.0:
                raise ValueError("gabor features require a finite positive period_m.")
        elif self.period_m is not None:
            raise ValueError("fiducial features do not use period_m.")

    @property
    def characteristic_scale_m(self) -> float:
        if self.period_m is not None:
            return float(self.period_m)
        return 2.0 * max(self.sigma_xy_m)


@dataclass(frozen=True)
class CertifiedRegion2D:
    """Registered region with action or lattice calibration metadata."""

    name: str
    kind: RegionKind
    centre_xy_m: tuple[float, float]
    size_xy_m: tuple[float, float]
    layer_count: int = 0
    lattice_spacing_m: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("perforation", "multilayer", "silicon_lattice"):
            raise ValueError("unknown certified region kind.")
        object.__setattr__(
            self, "centre_xy_m", _finite_pair("centre_xy_m", self.centre_xy_m)
        )
        object.__setattr__(self, "size_xy_m", _positive_pair("size_xy_m", self.size_xy_m))
        layer_count = int(self.layer_count)
        if self.kind == "multilayer" and layer_count < 1:
            raise ValueError("multilayer regions require a positive layer_count.")
        if self.kind != "multilayer" and layer_count != 0:
            raise ValueError("only multilayer regions may specify layer_count.")
        object.__setattr__(self, "layer_count", layer_count)
        if self.kind == "silicon_lattice":
            if (
                self.lattice_spacing_m is None
                or not np.isfinite(self.lattice_spacing_m)
                or self.lattice_spacing_m <= 0.0
            ):
                raise ValueError("silicon_lattice regions require lattice_spacing_m.")
        elif self.lattice_spacing_m is not None:
            raise ValueError("only silicon_lattice regions specify lattice_spacing_m.")

    @property
    def bounds_xyxy_m(self) -> tuple[float, float, float, float]:
        cx, cy = self.centre_xy_m
        sx, sy = self.size_xy_m
        return (cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2)

    @property
    def characteristic_scale_m(self) -> float:
        if self.lattice_spacing_m is not None:
            return float(self.lattice_spacing_m)
        return min(self.size_xy_m)


@dataclass(frozen=True)
class CertifiedLayerBoundary1D:
    """One indexed SiGe/Si boundary relative to the layer-stack origin."""

    index: int
    coordinate_m: float
    standard_uncertainty_m: float
    left_material: Literal["SiGe", "Si"]
    right_material: Literal["SiGe", "Si"]

    def __post_init__(self) -> None:
        index = int(self.index)
        coordinate = float(self.coordinate_m)
        uncertainty = float(self.standard_uncertainty_m)
        if index < 1:
            raise ValueError("layer boundary index must be positive.")
        if not np.isfinite(coordinate) or coordinate <= 0.0:
            raise ValueError("layer boundary coordinate_m must be finite and positive.")
        if not np.isfinite(uncertainty) or uncertainty < 0.0:
            raise ValueError(
                "layer boundary standard_uncertainty_m must be finite and non-negative."
            )
        if self.left_material == self.right_material or {
            self.left_material,
            self.right_material,
        } != {"SiGe", "Si"}:
            raise ValueError("a layer boundary must separate SiGe and Si.")
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "coordinate_m", coordinate)
        object.__setattr__(self, "standard_uncertainty_m", uncertainty)


@dataclass(frozen=True)
class MAGICALCertificate:
    """Physical recipe and metadata from which an atlas is rendered."""

    field_size_xy_m: tuple[float, float]
    reference_voltage: float
    features: tuple[CertifiedFeature2D, ...]
    regions: tuple[CertifiedRegion2D, ...] = ()
    layer_pathlength_step_m: float = 0.0
    layer_log_amplitude_step: float = 0.0
    silicon_layer_width_m: float = 13.0e-9
    silicon_germanium_layer_width_m: float = 10.0e-9
    silicon_layer_width_uncertainty_m: float = 0.5e-9
    silicon_germanium_layer_width_uncertainty_m: float = 0.5e-9
    interface_width_m: float = 1.0e-9
    silicon_lattice_spacing_uncertainty_m: float = 0.00005e-9
    perforation_diameter_m: float = 40e-9
    silicon_plane_hkl: tuple[int, int, int] = (1, 1, 1)
    silicon_zone_axis_uvw: tuple[int, int, int] = (1, -1, 0)
    multilayer_repeat_count: int = 8
    layer_boundaries: tuple[CertifiedLayerBoundary1D, ...] = ()
    certificate_source: str = "nominal synthetic certificate"
    version: str = "MAGICAL-nominal-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "field_size_xy_m", _positive_pair("field_size_xy_m", self.field_size_xy_m))
        voltage = float(self.reference_voltage)
        if not np.isfinite(voltage) or voltage <= 0.0:
            raise ValueError("reference_voltage must be finite and positive.")
        object.__setattr__(self, "reference_voltage", voltage)
        object.__setattr__(self, "features", tuple(self.features))
        object.__setattr__(self, "regions", tuple(self.regions))
        names = [feature.name for feature in self.features]
        if not names or len(names) != len(set(names)):
            raise ValueError("certificate feature names must be non-empty and unique.")
        region_names = [region.name for region in self.regions]
        if len(region_names) != len(set(region_names)):
            raise ValueError("certificate region names must be unique.")
        for name in ("layer_pathlength_step_m", "layer_log_amplitude_step"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        for name in (
            "silicon_layer_width_m",
            "silicon_germanium_layer_width_m",
            "interface_width_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in (
            "silicon_layer_width_uncertainty_m",
            "silicon_germanium_layer_width_uncertainty_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.interface_width_m >= min(
            self.silicon_layer_width_m,
            self.silicon_germanium_layer_width_m,
        ):
            raise ValueError("interface_width_m must be smaller than either layer width.")
        lattice_uncertainty = float(self.silicon_lattice_spacing_uncertainty_m)
        if not np.isfinite(lattice_uncertainty) or lattice_uncertainty < 0.0:
            raise ValueError(
                "silicon_lattice_spacing_uncertainty_m must be finite and non-negative."
            )
        diameter = float(self.perforation_diameter_m)
        if not np.isfinite(diameter) or diameter <= 0.0:
            raise ValueError("perforation_diameter_m must be finite and positive.")
        if diameter >= 0.5 * min(self.field_size_xy_m):
            raise ValueError(
                "perforation_diameter_m must be smaller than half the certified field; "
                "the value is incompatible with this nanoscale atlas."
            )
        plane = _indexed_triplet("silicon_plane_hkl", self.silicon_plane_hkl)
        zone = _indexed_triplet("silicon_zone_axis_uvw", self.silicon_zone_axis_uvw)
        if sum(h * u for h, u in zip(plane, zone, strict=True)) != 0:
            raise ValueError(
                "silicon_zone_axis_uvw must lie in silicon_plane_hkl (h*u+k*v+l*w=0)."
            )
        repeats = int(self.multilayer_repeat_count)
        if repeats < 2:
            raise ValueError("multilayer_repeat_count must be at least two.")
        source = str(self.certificate_source).strip()
        if not source:
            raise ValueError("certificate_source must be non-empty.")
        object.__setattr__(self, "silicon_plane_hkl", plane)
        object.__setattr__(self, "silicon_zone_axis_uvw", zone)
        object.__setattr__(self, "multilayer_repeat_count", repeats)
        object.__setattr__(self, "certificate_source", source)

        boundaries = tuple(self.layer_boundaries)
        if not boundaries:
            generated: list[CertifiedLayerBoundary1D] = []
            coordinate = 0.0
            variance = 0.0
            index = 1
            for _ in range(repeats):
                coordinate += self.silicon_germanium_layer_width_m
                variance += self.silicon_germanium_layer_width_uncertainty_m**2
                generated.append(
                    CertifiedLayerBoundary1D(
                        index,
                        coordinate,
                        np.sqrt(variance),
                        "SiGe",
                        "Si",
                    )
                )
                index += 1
                coordinate += self.silicon_layer_width_m
                variance += self.silicon_layer_width_uncertainty_m**2
                generated.append(
                    CertifiedLayerBoundary1D(
                        index,
                        coordinate,
                        np.sqrt(variance),
                        "Si",
                        "SiGe",
                    )
                )
                index += 1
            boundaries = tuple(generated)
        if any(
            right.coordinate_m <= left.coordinate_m
            for left, right in zip(boundaries, boundaries[1:])
        ):
            raise ValueError("layer boundary coordinates must be strictly increasing.")
        if [boundary.index for boundary in boundaries] != list(
            range(1, len(boundaries) + 1)
        ):
            raise ValueError("layer boundary indices must be consecutive from one.")
        if len(boundaries) != 2 * repeats:
            raise ValueError(
                "layer_boundaries must contain two indexed boundaries per bilayer repeat."
            )
        for boundary in boundaries:
            expected = (
                ("SiGe", "Si") if boundary.index % 2 else ("Si", "SiGe")
            )
            if (boundary.left_material, boundary.right_material) != expected:
                raise ValueError(
                    "layer boundary materials must alternate SiGe-to-Si then Si-to-SiGe."
                )
        stack_span = boundaries[-1].coordinate_m
        for region in self.regions:
            if (
                region.kind == "multilayer"
                and region.size_xy_m[0] < stack_span * (1.0 - 1e-12)
            ):
                raise ValueError(
                    f"multilayer region {region.name!r} is narrower than its "
                    "certified layer-boundary span."
                )
        object.__setattr__(self, "layer_boundaries", boundaries)

    @property
    def layers(self) -> dict[str, int]:
        return {
            region.name: region.layer_count
            for region in self.regions
            if region.kind == "multilayer"
        }

    @property
    def lattice_metadata(self) -> dict[str, float]:
        return {
            region.name: float(region.lattice_spacing_m)
            for region in self.regions
            if region.kind == "silicon_lattice"
        }

    @property
    def multilayer_period_m(self) -> float:
        """Nominal repeat distance of one SiGe/Si bilayer."""

        return self.silicon_germanium_layer_width_m + self.silicon_layer_width_m

    @property
    def layer_phase_step_rad(self) -> float:
        """Per-layer phase contrast at the certificate reference voltage."""

        wavelength = float(np.asarray(energy2wavelength(self.reference_voltage)))
        return 2.0 * np.pi * self.layer_pathlength_step_m / wavelength

    @property
    def layer_log_field_amplitude_step(self) -> float:
        """Positive per-layer attenuation depth in log field amplitude."""

        return self.layer_log_amplitude_step

    @property
    def multilayer_metadata(self) -> dict[str, float]:
        """Manufacturing values retained with the synthetic ground truth."""

        return {
            "silicon_layer_width_m": self.silicon_layer_width_m,
            "silicon_germanium_layer_width_m": self.silicon_germanium_layer_width_m,
            "silicon_layer_width_uncertainty_m": (
                self.silicon_layer_width_uncertainty_m
            ),
            "silicon_germanium_layer_width_uncertainty_m": (
                self.silicon_germanium_layer_width_uncertainty_m
            ),
            "interface_width_m": self.interface_width_m,
            "multilayer_period_m": self.multilayer_period_m,
            "layer_phase_step_rad": self.layer_phase_step_rad,
            "layer_log_field_amplitude_step": self.layer_log_field_amplitude_step,
        }

    @property
    def layer_boundary_coordinates_m(self) -> tuple[float, ...]:
        return tuple(boundary.coordinate_m for boundary in self.layer_boundaries)

    @property
    def layer_boundary_standard_uncertainties_m(self) -> tuple[float, ...]:
        return tuple(
            boundary.standard_uncertainty_m for boundary in self.layer_boundaries
        )

    @property
    def silicon_orientation_metadata(self) -> dict[str, tuple[int, int, int]]:
        """Crystallographic indices defining the simulated Si view."""

        return {
            "plane_hkl": self.silicon_plane_hkl,
            "zone_axis_uvw": self.silicon_zone_axis_uvw,
        }

    @property
    def silicon_lattice_metadata(self) -> dict[str, Any]:
        """Traceable spacing/orientation certificate for the lattice tile."""

        return {
            "spacing_m": next(iter(self.lattice_metadata.values()), None),
            "spacing_standard_uncertainty_m": (
                self.silicon_lattice_spacing_uncertainty_m
            ),
            "plane_hkl": self.silicon_plane_hkl,
            "zone_axis_uvw": self.silicon_zone_axis_uvw,
            "certificate_source": self.certificate_source,
        }


class ScalarActionJet2D(NamedTuple):
    value: jnp.ndarray
    gradient_xy: jnp.ndarray
    hessian_xy: jnp.ndarray


class SampleActionJet2D(NamedTuple):
    pathlength: ScalarActionJet2D
    log_amplitude: ScalarActionJet2D


@dataclass(frozen=True)
class ViewSelection:
    level_index: int
    feature_names: tuple[str, ...]
    region_names: tuple[str, ...]
    unresolved_region_names: tuple[str, ...]
    selected_region_name: str | None
    feature_family: str
    field_centre_local_xy_m: tuple[float, float]
    sample_translation_xy_m: tuple[float, float]
    sample_tilt_xy_rad: tuple[float, float]
    projected_path_scale: float
    local_bounds_xyxy_m: tuple[float, float, float, float]
    selected_source_bounds_xyxy_m: tuple[float, float, float, float]
    margin_components_m: tuple[tuple[str, float], ...]
    total_margin_m: float
    crop_halo_knots: int
    interpolation_halo_knots: int
    required_spacing_m: float
    selected_spacing_m: float
    specimen_pixel_min_m: float
    effective_magnification: float
    routed_feature_scale_m: float | None
    projected_feature_pixels: float | None
    visible_feature_intervals: float | None
    meets_feature_sampling: bool
    meets_resolution: bool


@jdc.pytree_dataclass(kw_only=True)
class SampleView(GaussianActionComponent):
    """A fixed-LOD C2 sample component ready for JAX/Gaussian propagation."""

    interpolator: Interpolator2D
    centre_xy_m: jnp.ndarray
    rotation_rad: jnp.ndarray
    tilt_xy_rad: jnp.ndarray
    domain_bounds_xyxy_m: jdc.Static[tuple[float, float, float, float]]
    z: float = 0.0
    boundary: jdc.Static[str] = "vacuum"
    level_index: jdc.Static[int] = 0
    level_name: jdc.Static[str] = ""
    selection: jdc.Static[ViewSelection | None] = None
    log_floor: jdc.Static[float] = -50.0
    interpolation_halo_knots: jdc.Static[int] = 0
    periodic_origin_xy_m: jdc.Static[tuple[float, float]] = (0.0, 0.0)
    vacuum_discs_local_xy_radius_m: jdc.Static[
        tuple[tuple[float, float, float, float], ...]
    ] = ()

    @classmethod
    def from_level(
        cls,
        level: SampleLevel2D,
        *,
        centre_xy_m: Sequence[float] = (0.0, 0.0),
        rotation_rad: float = 0.0,
        tilt_xy_rad: Sequence[float] = (0.0, 0.0),
        boundary: BoundaryMode = "vacuum",
        z: float = 0.0,
        level_index: int = 0,
        selection: ViewSelection | None = None,
        log_floor: float = -50.0,
        vacuum_discs_local_xy_radius_m: Sequence[Sequence[float]] = (),
    ) -> "SampleView":
        if boundary == "clamp":
            boundary = "constant"
        if boundary not in ("vacuum", "constant", "periodic", "nan"):
            raise ValueError(
                "boundary must be 'vacuum', 'constant', 'periodic', or 'nan'."
            )
        centre = np.asarray(_finite_pair("centre_xy_m", centre_xy_m), dtype=np.float64)
        if not np.isfinite(rotation_rad) or not np.isfinite(z):
            raise ValueError("rotation_rad and z must be finite.")
        tilt = np.asarray(_finite_pair("tilt_xy_rad", tilt_xy_rad), dtype=np.float64)
        if np.any(np.abs(tilt) >= 0.5 * np.pi - 1e-6):
            raise ValueError("tilt_xy_rad must keep the projected sample plane nonsingular.")
        if not np.isfinite(log_floor):
            raise ValueError("log_floor must be finite.")
        vacuum_discs: list[tuple[float, float, float, float]] = []
        for index, disc in enumerate(vacuum_discs_local_xy_radius_m):
            if len(disc) != 4:
                raise ValueError(
                    "each vacuum disc must contain (centre_x_m, centre_y_m, "
                    "radius_m, transition_width_m)."
                )
            cx, cy, radius, transition = (float(value) for value in disc)
            if not np.isfinite(cx) or not np.isfinite(cy):
                raise ValueError(f"vacuum disc {index} centre must be finite.")
            if not np.isfinite(radius) or radius <= 0.0:
                raise ValueError(f"vacuum disc {index} radius must be finite and positive.")
            if not np.isfinite(transition) or transition <= 0.0:
                raise ValueError(
                    f"vacuum disc {index} transition width must be finite and positive."
                )
            vacuum_discs.append((cx, cy, radius, transition))

        # Interpax's first two data axes are x then y.  Public storage remains
        # conventional yx and this is the only conversion point.
        action_xyc = np.stack(
            (level.pathlength_yx_m.T, level.log_amplitude_yx.T), axis=-1
        )
        x_coords = level.x_coords_m
        y_coords = level.y_coords_m
        extrap = np.zeros((2,), dtype=np.float64) if boundary == "vacuum" else False
        domain_bounds = (
            float(level.x_coords_m[0]),
            float(level.x_coords_m[-1]),
            float(level.y_coords_m[0]),
            float(level.y_coords_m[-1]),
        )
        if boundary == "constant":
            # Queries are clamped before evaluation; True prevents a spurious
            # NaN at floating-point roundoff just outside the final knot.
            extrap = True
        halo_knots = 0
        periodic_origin = (0.0, 0.0)
        derivative_kwargs: dict[str, np.ndarray] = {}
        if boundary == "periodic":
            # Append the wrapped seam knot and solve periodic cubic splines for
            # the cached first/mixed derivatives.  Interpax's default cubic2
            # derivative solve is not periodic; padding it merely makes the
            # seam error smaller, not C2.  These tensor-product derivatives
            # make the two limiting bicubic patches agree through order two.
            halo_knots = 1
            dy, dx = level.pixel_size_yx_m
            period_x = level.x_coords_m.size * dx
            period_y = level.y_coords_m.size * dy
            x_coords = np.concatenate(
                (level.x_coords_m, [level.x_coords_m[0] + period_x])
            )
            y_coords = np.concatenate(
                (level.y_coords_m, [level.y_coords_m[0] + period_y])
            )
            action_xyc = np.concatenate((action_xyc, action_xyc[:1]), axis=0)
            action_xyc = np.concatenate((action_xyc, action_xyc[:, :1]), axis=1)
            fx = CubicSpline(
                x_coords, action_xyc, axis=0, bc_type="periodic"
            )(x_coords, 1)
            fy = CubicSpline(
                y_coords, action_xyc, axis=1, bc_type="periodic"
            )(y_coords, 1)
            fxy = CubicSpline(
                y_coords, fx, axis=1, bc_type="periodic"
            )(y_coords, 1)
            derivative_kwargs = {"fx": fx, "fy": fy, "fxy": fxy}
            # Public wrapping uses the conventional pixel-edge extent.  That
            # interval has the same period as the spline and places its seam
            # halfway between the final and first stored pixel centres.
            domain_bounds = level.bounds_xyxy_m
            periodic_origin = (float(x_coords[0]), float(y_coords[0]))
            extrap = True
        interpolator = Interpolator2D(
            x=x_coords,
            y=y_coords,
            f=action_xyc,
            method="cubic2",
            extrap=extrap,
            **derivative_kwargs,
        )
        return cls(
            interpolator=interpolator,
            centre_xy_m=jnp.asarray(centre),
            rotation_rad=jnp.asarray(rotation_rad, dtype=jnp.float64),
            tilt_xy_rad=jnp.asarray(tilt, dtype=jnp.float64),
            domain_bounds_xyxy_m=domain_bounds,
            z=float(z),
            boundary=boundary,
            level_index=int(level_index),
            level_name=level.name,
            selection=selection,
            log_floor=float(log_floor),
            interpolation_halo_knots=halo_knots,
            periodic_origin_xy_m=periodic_origin,
            vacuum_discs_local_xy_radius_m=tuple(vacuum_discs),
        )

    @property
    def rotation_world_from_local(self) -> jnp.ndarray:
        c = jnp.cos(self.rotation_rad)
        s = jnp.sin(self.rotation_rad)
        return jnp.asarray(((c, -s), (s, c)), dtype=jnp.float64)

    @property
    def projection_world_from_local(self) -> jnp.ndarray:
        """Orthographic xy projection of the tilted physical sample plane."""

        tx, ty = self.tilt_xy_rad
        cx, sx = jnp.cos(tx), jnp.sin(tx)
        cy, sy = jnp.cos(ty), jnp.sin(ty)
        zero = jnp.asarray(0.0, dtype=jnp.float64)
        one = jnp.asarray(1.0, dtype=jnp.float64)
        rotation_x = jnp.asarray(
            ((one, zero, zero), (zero, cx, -sx), (zero, sx, cx))
        )
        rotation_y = jnp.asarray(
            ((cy, zero, sy), (zero, one, zero), (-sy, zero, cy))
        )
        c = jnp.cos(self.rotation_rad)
        s = jnp.sin(self.rotation_rad)
        rotation_z = jnp.asarray(
            ((c, -s, zero), (s, c, zero), (zero, zero, one))
        )
        return (rotation_z @ rotation_y @ rotation_x)[:2, :2]

    @property
    def projected_path_scale(self) -> jnp.ndarray:
        """Normal-thickness multiplier for the tilted projected action."""

        tx, ty = self.tilt_xy_rad
        return 1.0 / jnp.abs(jnp.cos(tx) * jnp.cos(ty))

    def world_to_local(self, xy_world: jnp.ndarray) -> jnp.ndarray:
        xy_world = jnp.asarray(xy_world, dtype=jnp.float64)
        if xy_world.shape[-1] != 2:
            raise ValueError("xy_world must have final dimension two.")
        local_from_world = jnp.linalg.inv(self.projection_world_from_local)
        return jnp.einsum(
            "ij,...j->...i", local_from_world, xy_world - self.centre_xy_m
        )

    def _local_action_jet(self, xy_local: jnp.ndarray):
        x = xy_local[..., 0]
        y = xy_local[..., 1]
        query_shape = x.shape
        x_min, x_max, y_min, y_max = self.domain_bounds_xyxy_m

        if self.boundary == "constant":
            inside_x = (x >= x_min) & (x <= x_max)
            inside_y = (y >= y_min) & (y <= y_max)
            xq = jnp.clip(x, x_min, x_max)
            yq = jnp.clip(y, y_min, y_max)
        elif self.boundary == "periodic":
            inside_x = jnp.ones_like(x, dtype=bool)
            inside_y = jnp.ones_like(y, dtype=bool)
            origin_x, origin_y = self.periodic_origin_xy_m
            xq = origin_x + jnp.mod(x - origin_x, x_max - x_min)
            yq = origin_y + jnp.mod(y - origin_y, y_max - y_min)
        else:
            inside_x = jnp.ones_like(x, dtype=bool)
            inside_y = jnp.ones_like(y, dtype=bool)
            xq, yq = x, y

        # Interpax 0.3.x documents arbitrary query shapes, but its bicubic
        # channel broadcasting is reliable for a one-dimensional query axis.
        # Flatten and restore here so image-shaped and batched calls have the
        # same public contract.
        xq_flat = jnp.ravel(xq)
        yq_flat = jnp.ravel(yq)

        def evaluate(dx: int, dy: int):
            values = self.interpolator(xq_flat, yq_flat, dx=dx, dy=dy)
            return values.reshape(query_shape + (2,))

        value = evaluate(0, 0)
        fx = evaluate(1, 0)
        fy = evaluate(0, 1)
        fxx = evaluate(2, 0)
        fxy = evaluate(1, 1)
        fyy = evaluate(0, 2)

        gradient = jnp.stack((fx, fy), axis=-1)
        hessian = jnp.stack(
            (
                jnp.stack((fxx, fxy), axis=-1),
                jnp.stack((fxy, fyy), axis=-1),
            ),
            axis=-2,
        )
        if self.boundary == "constant":
            derivative_mask = jnp.stack((inside_x, inside_y), axis=-1)
            gradient = gradient * derivative_mask[..., None, :]
            hessian = hessian * (
                derivative_mask[..., :, None] * derivative_mask[..., None, :]
            )[..., None, :, :]
        if self.vacuum_discs_local_xy_radius_m:
            # The raster already carries the same physical C2 aperture, while
            # this analytic guard eliminates any cubic ringing into its exact
            # vacuum interior.  Product-rule jets preserve C2 continuity at
            # both edges of the quintic transition instead of introducing a
            # hard ``where`` jump at the certified radius.
            identity = jnp.eye(2, dtype=jnp.float64)
            for cx, cy, radius, transition in self.vacuum_discs_local_xy_radius_m:
                delta = jnp.stack((x - cx, y - cy), axis=-1)
                distance = jnp.linalg.norm(delta, axis=-1)
                safe_distance = jnp.maximum(distance, jnp.finfo(jnp.float64).tiny)
                unit = delta / safe_distance[..., None]
                t_raw = (distance - radius) / transition
                t = jnp.clip(t_raw, 0.0, 1.0)
                mask = t**3 * (10.0 + t * (-15.0 + 6.0 * t))
                active = (t_raw > 0.0) & (t_raw < 1.0)
                first_radial = jnp.where(
                    active, 30.0 * t**2 * (t - 1.0) ** 2 / transition, 0.0
                )
                second_radial = jnp.where(
                    active,
                    60.0 * t * (1.0 - 3.0 * t + 2.0 * t**2) / transition**2,
                    0.0,
                )
                mask_gradient = first_radial[..., None] * unit
                unit_outer = jnp.einsum("...i,...j->...ij", unit, unit)
                mask_hessian = (
                    second_radial[..., None, None] * unit_outer
                    + (first_radial / safe_distance)[..., None, None]
                    * (identity - unit_outer)
                )
                previous_value = value
                previous_gradient = gradient
                value = mask[..., None] * previous_value
                gradient = (
                    mask[..., None, None] * previous_gradient
                    + previous_value[..., :, None] * mask_gradient[..., None, :]
                )
                hessian = (
                    mask[..., None, None, None] * hessian
                    + jnp.einsum(
                        "...i,...cj->...cij", mask_gradient, previous_gradient
                    )
                    + jnp.einsum(
                        "...ci,...j->...cij", previous_gradient, mask_gradient
                    )
                    + previous_value[..., :, None, None]
                    * mask_hessian[..., None, :, :]
                )
        return value, gradient, hessian

    def action_jet(self, xy_world: jnp.ndarray) -> SampleActionJet2D:
        """Evaluate both actions and their explicit world-coordinate jets."""

        local = self.world_to_local(xy_world)
        value, local_gradient, local_hessian = self._local_action_jet(local)
        local_from_world = jnp.linalg.inv(self.projection_world_from_local)
        action_scale = self.projected_path_scale
        value = value * action_scale
        gradient = action_scale * jnp.einsum(
            "ji,...cj->...ci", local_from_world, local_gradient
        )
        hessian = jnp.einsum(
            "ji,...cjk,kl->...cil",
            local_from_world,
            local_hessian,
            local_from_world,
        ) * action_scale
        return SampleActionJet2D(
            pathlength=ScalarActionJet2D(value[..., 0], gradient[..., 0, :], hessian[..., 0, :, :]),
            log_amplitude=ScalarActionJet2D(value[..., 1], gradient[..., 1, :], hessian[..., 1, :, :]),
        )

    def evaluate_actions(self, xy_world: jnp.ndarray) -> jnp.ndarray:
        """Return ``[..., (pathlength_m, log_field_amplitude)]``."""

        jet = self.action_jet(xy_world)
        return jnp.stack((jet.pathlength.value, jet.log_amplitude.value), axis=-1)

    def phase_shift(self, xy: jnp.ndarray):
        return self.action_jet(xy).pathlength.value

    def log_transmission(self, xy: jnp.ndarray):
        return self.action_jet(xy).log_amplitude.value

    def _call_ray(self, ray: Ray):
        return ray

    @staticmethod
    def _floored_log_amplitude_jet(jet: ScalarActionJet2D, floor: float):
        # logaddexp(L, floor) and its analytic first/second derivatives.
        weight = jax.nn.sigmoid(jet.value - floor)
        value = jnp.logaddexp(jet.value, floor)
        gradient = weight[..., None] * jet.gradient_xy
        hessian = (
            weight[..., None, None] * jet.hessian_xy
            + (weight * (1.0 - weight))[..., None, None]
            * jnp.einsum("...i,...j->...ij", jet.gradient_xy, jet.gradient_xy)
        )
        return ScalarActionJet2D(value, gradient, hessian)

    def _call_gaussian_single(self, ray: GaussianBeam) -> GaussianBeam:
        jet = self.action_jet(ray.r_xy)
        log_jet = self._floored_log_amplitude_jet(jet.log_amplitude, self.log_floor)
        k = ray.k

        # Match GaussianActionComponent's absorption recentering and Q safety
        # while using the spline's explicit derivatives rather than nested AD.
        im_q = jnp.imag(ray.Q_inv)
        dx_est = _solve_2x2(im_q, log_jet.gradient_xy) / k
        ratio = jnp.dot(log_jet.gradient_xy, dx_est)
        scale = 1.0 / (1.0 + ratio)
        log_gradient = log_jet.gradient_xy * scale
        log_hessian = log_jet.hessian_xy * scale
        log_hessian_over_k = _regularize_L2k(log_hessian / k, im_q)

        d_s0 = jet.pathlength.value - 1j * (log_jet.value / k)
        d_s1 = jet.pathlength.gradient_xy - 1j * (log_gradient / k)
        d_s2 = jet.pathlength.hessian_xy - 1j * log_hessian_over_k
        r_xy, d_xy, amplitude, pathlength, q_new = apply_action_delta(
            ray, dS0=d_s0, dS1=d_s1, dS2=d_s2
        )
        return ray.derive(
            x=r_xy[..., 0],
            y=r_xy[..., 1],
            dx=d_xy[..., 0],
            dy=d_xy[..., 1],
            z=ray.z,
            amplitude=amplitude,
            pathlength=pathlength,
            Q_inv=q_new,
        )

    def _call_gaussian(self, ray: GaussianBeam) -> GaussianBeam:
        if jnp.asarray(ray.r_xy).ndim > 1:
            return jax.vmap(self._call_gaussian_single)(ray.to_vector())
        return self._call_gaussian_single(ray)

    def sampling_diagnostic(self, ray: GaussianBeam, **kwargs):
        """Run the repository's local quadratic-action adequacy diagnostic."""

        from .gaussian_corrections import diagnose_gaussian_action_sampling

        return diagnose_gaussian_action_sampling(self, ray, **kwargs)

    def validate(self, gaussian_bundle: GaussianBeam, **kwargs):
        """Validate local action use and select local or re-expansion routing."""

        return validate_sample_level(self, gaussian_bundle, **kwargs)


# Public name used by the inversion plan; SampleView remains the concise alias
# used by the atlas/view-selection API.
CubicActionField2D = SampleView
# Long-form name retained for the implementation plan and older prototypes.
CubicInterpolatedSample2D = SampleView


@dataclass(frozen=True)
class SamplePyramid2D:
    """Host-side collection of action levels, ordered finest to coarsest."""

    levels: tuple[SampleLevel2D, ...]

    def __post_init__(self) -> None:
        levels = tuple(self.levels)
        if not levels:
            raise ValueError("a sample pyramid requires at least one level.")
        spacings = [level.max_spacing_m for level in levels]
        if any(right <= left for left, right in zip(spacings, spacings[1:])):
            raise ValueError("sample levels must be ordered finest to coarsest.")
        object.__setattr__(self, "levels", levels)

    @classmethod
    def from_finest(
        cls,
        finest: SampleLevel2D,
        *,
        max_levels: int = 6,
        antialias_sigma: float = 0.8,
        downsample_boundary: Literal["vacuum", "periodic"] = "vacuum",
    ) -> "SamplePyramid2D":
        """Build factor-two levels by filtered 2x2 area downsampling."""

        max_levels = int(max_levels)
        sigma = float(antialias_sigma)
        if max_levels < 1:
            raise ValueError("max_levels must be at least one.")
        if not np.isfinite(sigma) or sigma < 0.0:
            raise ValueError("antialias_sigma must be finite and non-negative.")
        if downsample_boundary not in ("vacuum", "periodic"):
            raise ValueError("downsample_boundary must be 'vacuum' or 'periodic'.")

        levels = [finest]
        current = finest
        while len(levels) < max_levels:
            ny, nx = current.shape_yx
            if ny < 8 or nx < 8 or ny % 2 or nx % 2:
                break

            def downsample(channel):
                filtered = gaussian_filter(
                    np.asarray(channel),
                    sigma=sigma,
                    mode=("wrap" if downsample_boundary == "periodic" else "constant"),
                    cval=0.0,
                ) if sigma else np.asarray(channel)
                return filtered.reshape(ny // 2, 2, nx // 2, 2).mean(axis=(1, 3))

            pathlength = downsample(current.pathlength_yx_m)
            log_amplitude = downsample(current.log_amplitude_yx)
            x = current.x_coords_m.reshape(nx // 2, 2).mean(axis=1)
            y = current.y_coords_m.reshape(ny // 2, 2).mean(axis=1)
            current = SampleLevel2D(
                pathlength,
                log_amplitude,
                x,
                y,
                name=f"{finest.name or 'sample'}-lod{len(levels)}",
            )
            levels.append(current)
        return cls(tuple(levels))

    def view(
        self,
        level_index: int,
        *,
        centre_xy_m: Sequence[float] = (0.0, 0.0),
        rotation_rad: float = 0.0,
        tilt_xy_rad: Sequence[float] = (0.0, 0.0),
        boundary: BoundaryMode = "vacuum",
        z: float = 0.0,
        selection: ViewSelection | None = None,
    ) -> SampleView:
        level_index = int(level_index)
        if not 0 <= level_index < len(self.levels):
            raise IndexError("level_index is outside the sample pyramid.")
        return SampleView.from_level(
            self.levels[level_index],
            centre_xy_m=centre_xy_m,
            rotation_rad=rotation_rad,
            tilt_xy_rad=tilt_xy_rad,
            boundary=boundary,
            z=z,
            level_index=level_index,
            selection=selection,
        )


def _smooth_sige_fraction(
    x_m: np.ndarray,
    *,
    origin_m: float,
    boundaries: Sequence[CertifiedLayerBoundary1D],
    interface_width_m: float,
) -> np.ndarray:
    """Render certified, possibly nonuniform SiGe intervals between boundaries.

    The stack starts in SiGe.  Odd indexed boundaries enter Si, and even
    indexed boundaries re-enter SiGe.  Consequently explicit metrology
    coordinates affect the simulated action rather than serving as metadata
    only.  Generated nominal boundaries recover the declared 10/13 nm recipe.
    """

    fraction = np.zeros_like(x_m, dtype=np.float64)
    boundary_tuple = tuple(boundaries)
    for repeat in range(len(boundary_tuple) // 2):
        start_offset = (
            0.0 if repeat == 0 else boundary_tuple[2 * repeat - 1].coordinate_m
        )
        stop_offset = boundary_tuple[2 * repeat].coordinate_m
        start = origin_m + start_offset
        stop = origin_m + stop_offset
        fraction += 0.5 * (
            np.tanh((x_m - start) / interface_width_m)
            - np.tanh((x_m - stop) / interface_width_m)
        )
    return np.clip(fraction, 0.0, 1.0)


def _c2_smoothstep01(value: np.ndarray) -> np.ndarray:
    """Quintic step with zero first/second derivatives at both endpoints."""

    t = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return t**3 * (10.0 + t * (-15.0 + 6.0 * t))


def _c2_interval_mask(
    coordinate: np.ndarray, lower: float, upper: float, transition_width: float
) -> np.ndarray:
    return _c2_smoothstep01((coordinate - lower) / transition_width) * _c2_smoothstep01(
        (upper - coordinate) / transition_width
    )


def _render_certified_scene(
    certificate: MAGICALCertificate,
    x_grid_m: np.ndarray,
    y_grid_m: np.ndarray,
    *,
    grid_spacing_xy_m: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the one certificate-defined scene on an arbitrary registered grid.

    Every global, regional, compact, and wide finest level calls this function.
    Atomic detail is included only when the requesting grid resolves it, which
    makes regional rasters refinements of one scene rather than replacement
    specimens.
    """

    phase = np.zeros_like(x_grid_m, dtype=np.float64)
    log_amplitude = np.zeros_like(x_grid_m, dtype=np.float64)
    for feature in certificate.features:
        cx, cy = feature.centre_xy_m
        sx, sy = feature.sigma_xy_m
        c, s = np.cos(feature.orientation_rad), np.sin(feature.orientation_rad)
        rx, ry = x_grid_m - cx, y_grid_m - cy
        u = c * rx + s * ry
        v = -s * rx + c * ry
        envelope = np.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2))
        if feature.kind == "gabor":
            carrier = np.cos(
                2.0 * np.pi * u / float(feature.period_m)
                + feature.carrier_phase_rad
            )
            phase += feature.phase_amplitude_rad * envelope * carrier
            log_amplitude -= (
                feature.log_amplitude_depth * envelope * 0.5 * (1.0 + carrier)
            )
        else:
            phase += feature.phase_amplitude_rad * envelope
            log_amplitude -= feature.log_amplitude_depth * envelope

    wavelength = float(np.asarray(energy2wavelength(certificate.reference_voltage)))
    pathlength = phase * wavelength / (2.0 * np.pi)
    transition = certificate.interface_width_m
    max_grid_spacing = max(grid_spacing_xy_m)
    for region in certificate.regions:
        if region.kind == "perforation":
            continue
        cx, cy = region.centre_xy_m
        size_x, size_y = region.size_xy_m
        mask = _c2_interval_mask(
            x_grid_m, cx - 0.5 * size_x, cx + 0.5 * size_x, transition
        ) * _c2_interval_mask(
            y_grid_m, cy - 0.5 * size_y, cy + 0.5 * size_y, transition
        )
        if region.kind == "multilayer":
            sige_fraction = _smooth_sige_fraction(
                x_grid_m,
                origin_m=cx - 0.5 * size_x,
                boundaries=certificate.layer_boundaries,
                interface_width_m=transition,
            )
            material_action = mask * (0.55 + 0.45 * sige_fraction)
            pathlength += (
                region.layer_count
                * certificate.layer_pathlength_step_m
                * material_action
            )
            log_amplitude -= (
                region.layer_count
                * certificate.layer_log_amplitude_step
                * material_action
            )
        else:
            pathlength += 0.5 * certificate.layer_pathlength_step_m * mask
            log_amplitude -= 0.5 * certificate.layer_log_amplitude_step * mask
            spacing = float(region.lattice_spacing_m)
            if spacing >= 4.0 * max_grid_spacing:
                lattice = (
                    np.cos(2.0 * np.pi * (x_grid_m - cx) / spacing)
                    + np.cos(2.0 * np.pi * (y_grid_m - cy) / spacing)
                )
                pathlength += (
                    0.15 * certificate.layer_pathlength_step_m * mask * lattice
                )

    # The hole acts on the complete scene.  A quintic one-sided transition is
    # identically vacuum through r=R and C2-flat at R and R+interface_width.
    for region in certificate.regions:
        if region.kind != "perforation":
            continue
        cx, cy = region.centre_xy_m
        radius = 0.5 * certificate.perforation_diameter_m
        distance = np.hypot(x_grid_m - cx, y_grid_m - cy)
        material_mask = _c2_smoothstep01((distance - radius) / transition)
        pathlength *= material_mask
        log_amplitude *= material_mask
    return pathlength, log_amplitude


@dataclass(frozen=True)
class MAGICALSampleAtlas:
    """A certificate-backed multiscale asymmetric calibration sample."""

    pyramid: SamplePyramid2D
    certificate: MAGICALCertificate
    region_pyramids: dict[str, SamplePyramid2D]
    global_support_margin_m: float = 0.0
    pose_translation_xy_m: tuple[float, float] = (0.0, 0.0)
    pose_rotation_rad: float = 0.0
    pose_tilt_xy_rad: tuple[float, float] = (0.0, 0.0)
    boundary_default: BoundaryMode = "vacuum"
    interpolation_halo_knots: int = 0

    def __post_init__(self) -> None:
        region_names = {region.name for region in self.certificate.regions}
        if set(self.region_pyramids) != region_names:
            raise ValueError(
                "region_pyramids must contain exactly the certificate's named regions."
            )
        support = float(self.global_support_margin_m)
        if not np.isfinite(support) or support < 0.0:
            raise ValueError("global_support_margin_m must be finite and non-negative.")
        translation = _finite_pair(
            "pose_translation_xy_m", self.pose_translation_xy_m
        )
        rotation = float(self.pose_rotation_rad)
        if not np.isfinite(rotation):
            raise ValueError("pose_rotation_rad must be finite.")
        tilt = _finite_pair("pose_tilt_xy_rad", self.pose_tilt_xy_rad)
        if max(abs(value) for value in tilt) >= 0.5 * np.pi - 1e-6:
            raise ValueError("pose_tilt_xy_rad makes the projected plane singular.")
        boundary = "constant" if self.boundary_default == "clamp" else self.boundary_default
        if boundary not in ("vacuum", "constant", "periodic", "nan"):
            raise ValueError(
                "boundary_default must be 'vacuum', 'constant', 'periodic', or 'nan'."
            )
        halo = int(self.interpolation_halo_knots)
        if halo < 0 or float(self.interpolation_halo_knots) != halo:
            raise ValueError("interpolation_halo_knots must be a non-negative integer.")
        object.__setattr__(self, "region_pyramids", dict(self.region_pyramids))
        object.__setattr__(self, "global_support_margin_m", support)
        object.__setattr__(self, "pose_translation_xy_m", translation)
        object.__setattr__(self, "pose_rotation_rad", rotation)
        object.__setattr__(self, "pose_tilt_xy_rad", tilt)
        object.__setattr__(self, "boundary_default", boundary)
        object.__setattr__(self, "interpolation_halo_knots", halo)

    @classmethod
    def from_certificate(
        cls,
        certificate: MAGICALCertificate,
        *,
        shape_yx: Sequence[int] = (256, 256),
        region_shape_yx: Sequence[int] = (256, 256),
        authoritative_multilayer_shape_yx: Sequence[int] = (2688, 2688),
        max_levels: int = 6,
        antialias_sigma: float = 0.8,
        global_support_margin_m: float = 0.0,
        pose_translation_xy_m: Sequence[float] = (0.0, 0.0),
        pose_rotation_rad: float = 0.0,
        pose_tilt_xy_rad: Sequence[float] = (0.0, 0.0),
        boundary_default: BoundaryMode = "vacuum",
        interpolation_halo_knots: int = 0,
    ) -> "MAGICALSampleAtlas":
        if len(shape_yx) != 2:
            raise ValueError("shape_yx must contain exactly two entries.")
        requested_ny, requested_nx = (int(shape_yx[0]), int(shape_yx[1]))
        if requested_ny < 4 or requested_nx < 4:
            raise ValueError("shape_yx must have at least four samples per axis.")
        width_x, width_y = certificate.field_size_xy_m
        support_margin = float(global_support_margin_m)
        if not np.isfinite(support_margin) or support_margin < 0.0:
            raise ValueError("global_support_margin_m must be finite and non-negative.")
        # The certified field is the usable square footprint.  Its raster
        # support includes the diagonal so the same frozen physical sample can
        # accommodate arbitrary relative image/sample rotation, plus the
        # explicitly declared host-side support margin.
        raster_width_x = np.sqrt(2.0) * width_x + 2.0 * support_margin
        raster_width_y = np.sqrt(2.0) * width_y + 2.0 * support_margin
        # ``shape_yx`` specifies sampling density over the certified field.
        # Support halos gain knots rather than silently coarsening that density.
        level_divisor = 2 ** max(0, int(max_levels) - 1)
        nx = int(np.ceil(requested_nx * raster_width_x / width_x))
        ny = int(np.ceil(requested_ny * raster_width_y / width_y))
        nx = int(np.ceil(nx / level_divisor) * level_divisor)
        ny = int(np.ceil(ny / level_divisor) * level_divisor)
        dx, dy = raster_width_x / nx, raster_width_y / ny
        x = (np.arange(nx) - 0.5 * (nx - 1)) * dx
        y = (np.arange(ny) - 0.5 * (ny - 1)) * dy
        xx, yy = np.meshgrid(x, y, indexing="xy")

        pathlength, log_amplitude = _render_certified_scene(
            certificate, xx, yy, grid_spacing_xy_m=(dx, dy)
        )
        finest = SampleLevel2D(
            pathlength,
            log_amplitude,
            x,
            y,
            name=certificate.version,
        )
        global_pyramid = SamplePyramid2D.from_finest(
            finest,
            max_levels=max_levels,
            antialias_sigma=antialias_sigma,
        )
        if len(region_shape_yx) != 2:
            raise ValueError("region_shape_yx must contain exactly two entries.")
        region_ny, region_nx = int(region_shape_yx[0]), int(region_shape_yx[1])
        if region_ny < 8 or region_nx < 8:
            raise ValueError("region_shape_yx must have at least eight samples per axis.")
        if len(authoritative_multilayer_shape_yx) != 2:
            raise ValueError(
                "authoritative_multilayer_shape_yx must contain exactly two entries."
            )
        authoritative_ny, authoritative_nx = (
            int(authoritative_multilayer_shape_yx[0]),
            int(authoritative_multilayer_shape_yx[1]),
        )
        if authoritative_ny < 8 or authoritative_nx < 8:
            raise ValueError(
                "authoritative_multilayer_shape_yx must have at least eight samples per axis."
            )

        def render_region_level(
            region: CertifiedRegion2D,
            *,
            tile_shape_yx: tuple[int, int],
            tile_size_xy_m: tuple[float, float],
            suffix: str,
        ) -> SampleLevel2D:
            cx, cy = region.centre_xy_m
            tile_ny, tile_nx = tile_shape_yx
            tile_width_x, tile_width_y = tile_size_xy_m
            tile_dx, tile_dy = tile_width_x / tile_nx, tile_width_y / tile_ny
            tile_x = cx + (np.arange(tile_nx) - 0.5 * (tile_nx - 1)) * tile_dx
            tile_y = cy + (np.arange(tile_ny) - 0.5 * (tile_ny - 1)) * tile_dy
            tile_xx, tile_yy = np.meshgrid(tile_x, tile_y, indexing="xy")

            tile_pathlength, tile_log_amplitude = _render_certified_scene(
                certificate,
                tile_xx,
                tile_yy,
                grid_spacing_xy_m=(tile_dx, tile_dy),
            )

            return SampleLevel2D(
                tile_pathlength,
                tile_log_amplitude,
                tile_x,
                tile_y,
                name=f"{certificate.version}:{region.name}:{suffix}",
            )

        def merge_pyramid_levels(*pyramids: SamplePyramid2D) -> SamplePyramid2D:
            candidates = sorted(
                (level for pyramid in pyramids for level in pyramid.levels),
                key=lambda level: level.max_spacing_m,
            )
            merged: list[SampleLevel2D] = []
            for level in candidates:
                if merged and np.isclose(
                    level.max_spacing_m,
                    merged[-1].max_spacing_m,
                    rtol=1e-10,
                    atol=0.0,
                ):
                    old_bounds = merged[-1].bounds_xyxy_m
                    new_bounds = level.bounds_xyxy_m
                    old_area = (old_bounds[1] - old_bounds[0]) * (
                        old_bounds[3] - old_bounds[2]
                    )
                    new_area = (new_bounds[1] - new_bounds[0]) * (
                        new_bounds[3] - new_bounds[2]
                    )
                    if new_area > old_area:
                        merged[-1] = level
                else:
                    merged.append(level)
            return SamplePyramid2D(tuple(merged))

        def support_scaled_shape(
            base_shape_yx: tuple[int, int],
            actual_size_xy_m: tuple[float, float],
            reference_size_xy_m: tuple[float, float],
        ) -> tuple[int, int]:
            """Add knots with support so a tile's physical pitch stays fixed."""

            base_ny, base_nx = base_shape_yx
            actual_x, actual_y = actual_size_xy_m
            reference_x, reference_y = reference_size_xy_m
            scaled_nx = int(np.ceil(base_nx * actual_x / reference_x))
            scaled_ny = int(np.ceil(base_ny * actual_y / reference_y))
            # Retain the requested number of factor-two LODs where possible.
            scaled_nx = int(np.ceil(scaled_nx / level_divisor) * level_divisor)
            scaled_ny = int(np.ceil(scaled_ny / level_divisor) * level_divisor)
            return (scaled_ny, scaled_nx)

        region_pyramids: dict[str, SamplePyramid2D] = {}
        for region in certificate.regions:
            size_x, size_y = region.size_xy_m
            if region.kind == "silicon_lattice":
                spacing = float(region.lattice_spacing_m)
                registered_periods_x = int(round(size_x / spacing))
                registered_periods_y = int(round(size_y / spacing))
                # Periodic wrapping supplies unlimited repeated support; only
                # the rotated registered diagonal plus four physical periods
                # on each side are needed to construct its C2 seam.  Do not
                # inflate this atomic raster with the unrelated global-field
                # padding budget.
                periods_x = int(np.ceil(np.sqrt(2.0) * registered_periods_x)) + 8
                periods_y = int(np.ceil(np.sqrt(2.0) * registered_periods_y)) + 8
                tile_shape = (periods_y * 8, periods_x * 8)
                tile_size = (periods_x * spacing, periods_y * spacing)
                tile_level = render_region_level(
                    region,
                    tile_shape_yx=tile_shape,
                    tile_size_xy_m=tile_size,
                    suffix="periodic",
                )
                region_pyramids[region.name] = SamplePyramid2D.from_finest(
                    tile_level,
                    max_levels=max_levels,
                    antialias_sigma=antialias_sigma,
                    downsample_boundary="periodic",
                )
                continue

            base_size = (1.4 * size_x, 1.4 * size_y)
            if region.kind == "multilayer":
                routing_core = (
                    np.sqrt(2.0)
                    * certificate.multilayer_repeat_count
                    * certificate.multilayer_period_m
                )
                routing_width = (
                    routing_core + 2.0 * support_margin
                )
                wide_size = (
                    max(base_size[0], routing_width),
                    max(base_size[1], routing_width),
                )
                density_reference = (
                    max(base_size[0], routing_core),
                    max(base_size[1], routing_core),
                )
            else:
                wide_size = (
                    max(base_size[0], size_x + 2.0 * support_margin),
                    max(base_size[1], size_y + 2.0 * support_margin),
                )
                density_reference = base_size
            wide_shape = support_scaled_shape(
                (region_ny, region_nx), wide_size, density_reference
            )
            wide_level = render_region_level(
                region,
                tile_shape_yx=wide_shape,
                tile_size_xy_m=wide_size,
                suffix="wide",
            )
            wide_pyramid = SamplePyramid2D.from_finest(
                wide_level,
                max_levels=max_levels,
                antialias_sigma=antialias_sigma,
            )
            if region.kind == "multilayer" and region.layer_count == 4:
                # A compact dense tile covers the worst-case rotated 150 kx
                # footprint; the wide tile above separately covers 80 kx.  The
                # merged host-side pyramid can therefore vary support with LOD.
                # It is a registered central sub-tile of the full eight-repeat
                # certified region, so its support is set by four complete
                # repeats plus the diagonal/margins rather than by the much
                # larger wide acquisition tile.
                detail_width = (
                    np.sqrt(2.0) * 4.0 * certificate.multilayer_period_m
                    + 2.0 * support_margin
                )
                detail_height = detail_width
                detail_level = render_region_level(
                    region,
                    tile_shape_yx=(authoritative_ny, authoritative_nx),
                    tile_size_xy_m=(detail_width, detail_height),
                    suffix="authoritative",
                )
                detail_pyramid = SamplePyramid2D.from_finest(
                    detail_level,
                    max_levels=max_levels,
                    antialias_sigma=antialias_sigma,
                )
                region_pyramids[region.name] = merge_pyramid_levels(
                    detail_pyramid, wide_pyramid
                )
            else:
                region_pyramids[region.name] = wide_pyramid
        return cls(
            global_pyramid,
            certificate,
            region_pyramids,
            support_margin,
            _finite_pair("pose_translation_xy_m", pose_translation_xy_m),
            float(pose_rotation_rad),
            _finite_pair("pose_tilt_xy_rad", pose_tilt_xy_rad),
            boundary_default,
            interpolation_halo_knots,
        )

    @classmethod
    def nominal(
        cls,
        *,
        field_width_m: float = 352e-9,
        shape_yx: Sequence[int] = (640, 640),
        region_shape_yx: Sequence[int] = (1024, 1024),
        authoritative_multilayer_shape_yx: Sequence[int] = (2688, 2688),
        voltage: float = 200e3,
        max_levels: int = 6,
        antialias_sigma: float = 0.8,
        global_support_margin_m: float = 8e-9,
        silicon_layer_width_m: float = 13e-9,
        silicon_germanium_layer_width_m: float = 10e-9,
        silicon_layer_width_uncertainty_m: float = 0.5e-9,
        silicon_germanium_layer_width_uncertainty_m: float = 0.5e-9,
        interface_width_m: float = 1e-9,
        silicon_111_spacing_m: float = 0.31356e-9,
        silicon_lattice_spacing_uncertainty_m: float = 0.00005e-9,
        silicon_plane_hkl: Sequence[int] = (1, 1, 1),
        silicon_zone_axis_uvw: Sequence[int] = (1, -1, 0),
        perforation_diameter_m: float = 40e-9,
        multilayer_repeat_count: int = 8,
        layer_phase_step_rad: float = 0.06,
        layer_log_field_amplitude_step: float = 0.035,
        layer_boundaries: Sequence[CertifiedLayerBoundary1D] = (),
        certificate_source: str = "nominal synthetic certificate",
        certificate_version: str = "MAGICAL-nominal-v1",
        pose_translation_xy_m: Sequence[float] = (0.0, 0.0),
        pose_rotation_rad: float = 0.0,
        pose_tilt_xy_rad: Sequence[float] = (0.0, 0.0),
        boundary_default: BoundaryMode = "vacuum",
        interpolation_halo_knots: int = 0,
    ) -> "MAGICALSampleAtlas":
        """Construct the deterministic nominal asymmetric multiscale atlas."""

        width = float(field_width_m)
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError("field_width_m must be finite and positive.")
        features = (
            CertifiedFeature2D(
                "coarse-x", "gabor", (0.0, 0.0), (0.42 * width, 0.38 * width),
                0.09, width / 2.4, 0.2, 0.18, 0.10,
            ),
            CertifiedFeature2D(
                "coarse-oblique", "gabor", (0.0, 0.0), (0.39 * width, 0.35 * width),
                0.61, width / 4.3, 1.1, 0.14, 0.08,
            ),
            CertifiedFeature2D(
                "medium-oblique", "gabor", (0.02 * width, -0.01 * width),
                (0.28 * width, 0.24 * width), 1.17, width / 8.1, 2.0, 0.11, 0.07,
            ),
            CertifiedFeature2D(
                "fine-core", "gabor", (0.0, 0.0), (0.16 * width, 0.14 * width),
                1.79, width / 18.0, 0.7, 0.08, 0.05,
            ),
            # A scalene, unequal triplet supplies handedness and absolute pose.
            CertifiedFeature2D(
                "fiducial-a", "fiducial", (-0.27 * width, -0.20 * width),
                (0.020 * width, 0.026 * width), phase_amplitude_rad=0.12,
                log_amplitude_depth=0.34,
            ),
            CertifiedFeature2D(
                "fiducial-b", "fiducial", (0.24 * width, -0.13 * width),
                (0.029 * width, 0.018 * width), phase_amplitude_rad=-0.09,
                log_amplitude_depth=0.24,
            ),
            CertifiedFeature2D(
                "fiducial-c", "fiducial", (-0.08 * width, 0.29 * width),
                (0.015 * width, 0.036 * width), orientation_rad=0.43,
                phase_amplitude_rad=0.16, log_amplitude_depth=0.18,
            ),
        )
        wavelength = float(np.asarray(energy2wavelength(voltage)))
        lattice_spacing = float(silicon_111_spacing_m)
        perforation_diameter = float(perforation_diameter_m)
        phase_step = float(layer_phase_step_rad)
        log_field_step = float(layer_log_field_amplitude_step)
        if not np.isfinite(phase_step) or phase_step < 0.0:
            raise ValueError("layer_phase_step_rad must be finite and non-negative.")
        if not np.isfinite(log_field_step) or log_field_step < 0.0:
            raise ValueError(
                "layer_log_field_amplitude_step must be finite and non-negative."
            )
        layer_stack_width = int(multilayer_repeat_count) * (
            float(silicon_layer_width_m)
            + float(silicon_germanium_layer_width_m)
        )
        if layer_stack_width >= 0.8 * width:
            raise ValueError(
                "the certified multilayer repeat stack is too wide for the atlas field."
            )
        layer_region_height = 0.11 * width
        regions = (
            CertifiedRegion2D(
                "perforation",
                "perforation",
                (0.0, 0.0),
                (perforation_diameter, perforation_diameter),
            ),
            CertifiedRegion2D(
                "multilayer_region_1", "multilayer", (0.0, -0.36 * width),
                (layer_stack_width, layer_region_height), layer_count=1,
            ),
            CertifiedRegion2D(
                "multilayer_region_2", "multilayer", (0.0, -0.12 * width),
                (layer_stack_width, layer_region_height), layer_count=2,
            ),
            CertifiedRegion2D(
                "multilayer_region_3", "multilayer", (0.0, 0.12 * width),
                (layer_stack_width, layer_region_height), layer_count=3,
            ),
            CertifiedRegion2D(
                "multilayer_region_4", "multilayer", (0.0, 0.36 * width),
                (layer_stack_width, layer_region_height), layer_count=4,
            ),
            CertifiedRegion2D(
                "silicon_lattice_region", "silicon_lattice", (0.36 * width, 0.0),
                (80 * lattice_spacing, 80 * lattice_spacing),
                lattice_spacing_m=lattice_spacing,
            ),
        )
        certificate = MAGICALCertificate(
            (width, width),
            voltage,
            features,
            regions=regions,
            layer_pathlength_step_m=phase_step * wavelength / (2.0 * np.pi),
            layer_log_amplitude_step=log_field_step,
            silicon_layer_width_m=float(silicon_layer_width_m),
            silicon_germanium_layer_width_m=float(
                silicon_germanium_layer_width_m
            ),
            silicon_layer_width_uncertainty_m=float(
                silicon_layer_width_uncertainty_m
            ),
            silicon_germanium_layer_width_uncertainty_m=float(
                silicon_germanium_layer_width_uncertainty_m
            ),
            interface_width_m=float(interface_width_m),
            silicon_lattice_spacing_uncertainty_m=float(
                silicon_lattice_spacing_uncertainty_m
            ),
            perforation_diameter_m=perforation_diameter,
            silicon_plane_hkl=_indexed_triplet(
                "silicon_plane_hkl", silicon_plane_hkl
            ),
            silicon_zone_axis_uvw=_indexed_triplet(
                "silicon_zone_axis_uvw", silicon_zone_axis_uvw
            ),
            multilayer_repeat_count=int(multilayer_repeat_count),
            layer_boundaries=tuple(layer_boundaries),
            certificate_source=certificate_source,
            version=certificate_version,
        )
        return cls.from_certificate(
            certificate,
            shape_yx=shape_yx,
            region_shape_yx=region_shape_yx,
            authoritative_multilayer_shape_yx=authoritative_multilayer_shape_yx,
            max_levels=max_levels,
            antialias_sigma=antialias_sigma,
            global_support_margin_m=global_support_margin_m,
            pose_translation_xy_m=pose_translation_xy_m,
            pose_rotation_rad=pose_rotation_rad,
            pose_tilt_xy_rad=pose_tilt_xy_rad,
            boundary_default=boundary_default,
            interpolation_halo_knots=interpolation_halo_knots,
        )

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        shape_yx: Sequence[int] = (640, 640),
        region_shape_yx: Sequence[int] = (1024, 1024),
        authoritative_multilayer_shape_yx: Sequence[int] = (2688, 2688),
        max_levels: int = 6,
        antialias_sigma: float = 0.8,
        global_support_margin_m: float = 8e-9,
    ) -> "MAGICALSampleAtlas":
        """Build the atlas from the NEOARM TOML certificate mapping.

        This method consumes physical certificate values rather than merely
        comparing a nominal atlas against them.  A full parsed TOML mapping is
        expected so that the beam voltage and specimen identity remain
        traceable with the certificate.
        """

        if not isinstance(config, Mapping):
            raise TypeError("config must be a parsed TOML mapping.")
        sample_table = config.get("sample")
        beam_table = config.get("beam")
        if not isinstance(sample_table, Mapping):
            raise ValueError("config is missing the [sample] table.")
        if not isinstance(beam_table, Mapping):
            raise ValueError("config is missing the [beam] table.")
        certificate_table = sample_table.get("certificate")
        if not isinstance(certificate_table, Mapping):
            raise ValueError("config is missing the [sample.certificate] table.")

        required = (
            "source",
            "si_layer_width_m",
            "sige_layer_width_m",
            "layer_width_standard_uncertainty_m",
            "interface_width_m",
            "si_111_spacing_m",
            "lattice_spacing_standard_uncertainty_m",
            "perforation_diameter_m",
            "multilayer_region_count",
            "layer_phase_step_rad",
            "layer_log_field_amplitude_step",
        )
        missing = [key for key in required if key not in certificate_table]
        if missing:
            raise ValueError(
                "sample certificate is missing required values: "
                + ", ".join(missing)
            )
        if "voltage_v" not in beam_table:
            raise ValueError("config [beam] is missing voltage_v.")
        region_count = int(certificate_table["multilayer_region_count"])
        if region_count != 4:
            raise ValueError(
                "MAGICALSampleAtlas currently requires exactly four registered "
                "multilayer regions."
            )
        common_width_uncertainty = float(
            certificate_table["layer_width_standard_uncertainty_m"]
        )
        boundary_coordinates = certificate_table.get(
            "layer_boundary_coordinates_m"
        )
        boundary_uncertainties = certificate_table.get(
            "layer_boundary_standard_uncertainties_m"
        )
        if (boundary_coordinates is None) != (boundary_uncertainties is None):
            raise ValueError(
                "layer boundary coordinates and uncertainties must be provided together."
            )
        parsed_boundaries: tuple[CertifiedLayerBoundary1D, ...] = ()
        if boundary_coordinates is not None:
            if not isinstance(boundary_coordinates, Sequence) or not isinstance(
                boundary_uncertainties, Sequence
            ):
                raise ValueError(
                    "layer boundary coordinates and uncertainties must be sequences."
                )
            if len(boundary_coordinates) != len(boundary_uncertainties):
                raise ValueError(
                    "layer boundary coordinates and uncertainties must have equal length."
                )
            parsed_boundaries = tuple(
                CertifiedLayerBoundary1D(
                    index=index,
                    coordinate_m=float(coordinate),
                    standard_uncertainty_m=float(uncertainty),
                    left_material=("SiGe" if index % 2 else "Si"),
                    right_material=("Si" if index % 2 else "SiGe"),
                )
                for index, (coordinate, uncertainty) in enumerate(
                    zip(boundary_coordinates, boundary_uncertainties, strict=True),
                    start=1,
                )
            )
        source = str(certificate_table["source"])
        identity = str(sample_table.get("identity", "MAGICAL-configured"))
        return cls.nominal(
            field_width_m=float(certificate_table.get("field_width_m", 352e-9)),
            shape_yx=shape_yx,
            region_shape_yx=region_shape_yx,
            authoritative_multilayer_shape_yx=authoritative_multilayer_shape_yx,
            voltage=float(beam_table["voltage_v"]),
            max_levels=max_levels,
            antialias_sigma=antialias_sigma,
            global_support_margin_m=global_support_margin_m,
            silicon_layer_width_m=float(certificate_table["si_layer_width_m"]),
            silicon_germanium_layer_width_m=float(
                certificate_table["sige_layer_width_m"]
            ),
            silicon_layer_width_uncertainty_m=float(
                certificate_table.get(
                    "si_layer_width_standard_uncertainty_m",
                    common_width_uncertainty,
                )
            ),
            silicon_germanium_layer_width_uncertainty_m=float(
                certificate_table.get(
                    "sige_layer_width_standard_uncertainty_m",
                    common_width_uncertainty,
                )
            ),
            interface_width_m=float(certificate_table["interface_width_m"]),
            silicon_111_spacing_m=float(certificate_table["si_111_spacing_m"]),
            silicon_lattice_spacing_uncertainty_m=float(
                certificate_table["lattice_spacing_standard_uncertainty_m"]
            ),
            silicon_plane_hkl=certificate_table.get("si_plane_hkl", (1, 1, 1)),
            silicon_zone_axis_uvw=certificate_table.get(
                "si_zone_axis_uvw", (1, -1, 0)
            ),
            perforation_diameter_m=float(
                certificate_table["perforation_diameter_m"]
            ),
            multilayer_repeat_count=int(
                certificate_table.get("multilayer_repeat_count", 8)
            ),
            layer_phase_step_rad=float(certificate_table["layer_phase_step_rad"]),
            layer_log_field_amplitude_step=float(
                certificate_table["layer_log_field_amplitude_step"]
            ),
            layer_boundaries=parsed_boundaries,
            certificate_source=source,
            certificate_version=f"{identity}:{source}",
            pose_translation_xy_m=sample_table.get(
                "pose_translation_xy_m", (0.0, 0.0)
            ),
            pose_rotation_rad=np.deg2rad(
                float(sample_table.get("pose_rotation_deg", 0.0))
            ),
            pose_tilt_xy_rad=tuple(
                float(value) * 1e-3
                for value in sample_table.get("pose_tilt_xy_mrad", (0.0, 0.0))
            ),
            boundary_default=sample_table.get("boundary_default", "vacuum"),
            interpolation_halo_knots=sample_table.get("interpolation_halo_px", 0),
        )

    @classmethod
    def from_toml(
        cls,
        path: str | Path,
        **atlas_kwargs,
    ) -> "MAGICALSampleAtlas":
        """Load a TOML file and delegate to :meth:`from_config`."""

        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise ModuleNotFoundError(
                    "No TOML library found; on Python 3.10 install the 'tomli' package."
                ) from exc

        with Path(path).open("rb") as stream:
            config = tomllib.load(stream)
        return cls.from_config(config, **atlas_kwargs)

    @property
    def regions(self) -> dict[str, CertifiedRegion2D]:
        return {region.name: region for region in self.certificate.regions}

    def region(self, name: str) -> CertifiedRegion2D:
        try:
            return self.regions[name]
        except KeyError as exc:
            raise KeyError(f"unknown MAGICAL region {name!r}") from exc

    def pyramid_for_region(self, name: str) -> SamplePyramid2D:
        self.region(name)  # provide the same clear unknown-name error
        return self.region_pyramids[name]

    @property
    def perforation(self) -> CertifiedRegion2D:
        return self.region("perforation")

    @property
    def multilayer_region_1(self) -> CertifiedRegion2D:
        return self.region("multilayer_region_1")

    @property
    def multilayer_region_2(self) -> CertifiedRegion2D:
        return self.region("multilayer_region_2")

    @property
    def multilayer_region_3(self) -> CertifiedRegion2D:
        return self.region("multilayer_region_3")

    @property
    def multilayer_region_4(self) -> CertifiedRegion2D:
        return self.region("multilayer_region_4")

    @property
    def silicon_lattice_region(self) -> CertifiedRegion2D:
        return self.region("silicon_lattice_region")


def _rotation_matrix_numpy(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.asarray(((c, -s), (s, c)), dtype=np.float64)


def _sample_projection_numpy(
    rotation_rad: float, tilt_xy_rad: Sequence[float]
) -> tuple[np.ndarray, float]:
    """Return projected local-to-world map and normal pathlength scale."""

    tx, ty = _finite_pair("sample_tilt_xy_rad", tilt_xy_rad)
    if max(abs(tx), abs(ty)) >= 0.5 * np.pi - 1e-6:
        raise ValueError("sample_tilt_xy_rad makes the projected plane singular.")
    cx, sx = np.cos(tx), np.sin(tx)
    cy, sy = np.cos(ty), np.sin(ty)
    rz = np.eye(3)
    rz[:2, :2] = _rotation_matrix_numpy(rotation_rad)
    rx = np.asarray(((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx)))
    ry = np.asarray(((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)))
    rotation = rz @ ry @ rx
    return rotation[:2, :2], float(1.0 / abs(rotation[2, 2]))


def select_sample_view(
    atlas_or_pyramid: MAGICALSampleAtlas | SamplePyramid2D,
    *,
    transfer_jacobian_xy,
    transfer_jacobian_envelope_xy=None,
    detector_shape_yx: Sequence[int],
    detector_pixel_size_yx_m: Sequence[float],
    sample_centre_xy_m: Sequence[float] | None = None,
    sample_rotation_rad: float | None = None,
    sample_tilt_xy_rad: Sequence[float] | None = None,
    region_name: str | None = None,
    gaussian_radius_min_m: float | None = None,
    margin_m: float = 0.0,
    lens_wobble_margin_m: float = 0.0,
    calibrated_input_margin_m: float = 0.0,
    gaussian_support_margin_m: float = 0.0,
    pose_uncertainty_margin_m: float = 0.0,
    interpolation_halo_margin_m: float = 0.0,
    texels_per_detector_pixel: float = 2.0,
    samples_per_gaussian_radius: float = 3.0,
    samples_per_feature_period: float = 8.0,
    min_detector_pixels_per_feature: float = 3.0,
    max_detector_pixels_per_feature: float = 96.0,
    min_visible_feature_intervals: float = 2.0,
    boundary: BoundaryMode | None = None,
    z: float = 0.0,
) -> SampleView:
    """Freeze a registered field, source tile, crop, and adequate cubic LOD.

    ``transfer_jacobian_xy`` maps small specimen-world displacements to
    detector displacements.  ``transfer_jacobian_envelope_xy`` may contain an
    ``(N, 2, 2)`` set spanning calibrated/wobble uncertainty: its inverse FOVs
    are unioned for the frozen crop and its finest back-projected detector
    sampling participates in LOD acceptance, while nominal routing remains
    determined by ``transfer_jacobian_xy``.  For a MAGICAL atlas, the projected scale selects
    the experimental field: global/multilayer period at 40 kx, a registered
    multilayer tile through 400 kx, then the Si {111} tile.  The returned view
    includes the calibrated stage translation that maps the selected local
    field centre to ``sample_centre_xy_m`` (the detector-axis world point).

    Selection is deliberately host-side and discrete.  The selected region,
    source tile, FOV crop, margins, and LOD therefore remain fixed inside an
    optimizer.  ``region_name`` overrides automatic field routing.
    """

    atlas: MAGICALSampleAtlas | None
    if isinstance(atlas_or_pyramid, MAGICALSampleAtlas):
        atlas = atlas_or_pyramid
        features = atlas.certificate.features
        regions = atlas.certificate.regions
    elif isinstance(atlas_or_pyramid, SamplePyramid2D):
        atlas = None
        features = ()
        regions = ()
        if region_name is not None:
            raise ValueError("region_name requires a MAGICALSampleAtlas.")
    else:
        raise TypeError("atlas_or_pyramid must be MAGICALSampleAtlas or SamplePyramid2D.")

    jacobian = np.asarray(transfer_jacobian_xy, dtype=np.float64)
    if jacobian.shape != (2, 2) or not np.all(np.isfinite(jacobian)):
        raise ValueError("transfer_jacobian_xy must be a finite 2x2 matrix.")
    determinant = float(np.linalg.det(jacobian))
    if abs(determinant) <= np.finfo(np.float64).eps * np.linalg.norm(jacobian) ** 2:
        raise ValueError("transfer_jacobian_xy must be nonsingular for image-FOV selection.")
    inverse = np.linalg.inv(jacobian)
    jacobian_envelope: list[np.ndarray] = []
    if transfer_jacobian_envelope_xy is not None:
        envelope = np.asarray(transfer_jacobian_envelope_xy, dtype=np.float64)
        if envelope.ndim != 3 or envelope.shape[1:] != (2, 2) or envelope.shape[0] < 1:
            raise ValueError(
                "transfer_jacobian_envelope_xy must have shape (N, 2, 2), N >= 1."
            )
        for index, candidate in enumerate(envelope):
            candidate_norm = float(np.linalg.norm(candidate))
            candidate_det = float(np.linalg.det(candidate))
            if not np.all(np.isfinite(candidate)) or (
                abs(candidate_det)
                <= np.finfo(np.float64).eps * candidate_norm**2
            ):
                raise ValueError(
                    "transfer_jacobian_envelope_xy contains a non-finite or "
                    f"singular matrix at index {index}."
                )
            jacobian_envelope.append(candidate)
    all_jacobians = [jacobian, *jacobian_envelope]
    all_inverses = [np.linalg.inv(candidate) for candidate in all_jacobians]

    if len(detector_shape_yx) != 2:
        raise ValueError("detector_shape_yx must contain exactly two entries.")
    ny, nx = (int(detector_shape_yx[0]), int(detector_shape_yx[1]))
    if ny <= 0 or nx <= 0:
        raise ValueError("detector dimensions must be positive.")
    py, px = _positive_pair("detector_pixel_size_yx_m", detector_pixel_size_yx_m)
    if sample_centre_xy_m is None:
        sample_centre_xy_m = (
            atlas.pose_translation_xy_m if atlas is not None else (0.0, 0.0)
        )
    if sample_rotation_rad is None:
        sample_rotation_rad = atlas.pose_rotation_rad if atlas is not None else 0.0
    if sample_tilt_xy_rad is None:
        sample_tilt_xy_rad = (
            atlas.pose_tilt_xy_rad if atlas is not None else (0.0, 0.0)
        )
    world_field_centre = np.asarray(
        _finite_pair("sample_centre_xy_m", sample_centre_xy_m), dtype=np.float64
    )
    theta = float(sample_rotation_rad)
    if not np.isfinite(theta):
        raise ValueError("sample_rotation_rad must be finite.")
    projection, projected_path_scale = _sample_projection_numpy(
        theta, sample_tilt_xy_rad
    )
    local_from_world = np.linalg.inv(projection)

    if texels_per_detector_pixel <= 0.0:
        raise ValueError("texels_per_detector_pixel must be positive.")
    if samples_per_gaussian_radius <= 0.0 or samples_per_feature_period <= 0.0:
        raise ValueError("sample-count requirements must be positive.")
    if min_detector_pixels_per_feature <= 0.0:
        raise ValueError("min_detector_pixels_per_feature must be positive.")
    if max_detector_pixels_per_feature <= min_detector_pixels_per_feature:
        raise ValueError(
            "max_detector_pixels_per_feature must exceed the minimum."
        )
    if min_visible_feature_intervals < 2.0 or not np.isfinite(
        min_visible_feature_intervals
    ):
        raise ValueError("min_visible_feature_intervals must be finite and at least two.")

    effective_magnification = float(np.sqrt(abs(np.linalg.det(jacobian @ projection))))
    # Numerical column solves perturb nominal magnifications at roughly the
    # 1e-10 relative level.  A 1 ppm band keeps named acquisition states on
    # their intended side of the 40/150/400 kx routing boundaries without
    # merging experimentally distinct bands.
    routing_rtol = 1e-6
    at_least_40k = effective_magnification >= 40_000.0 * (1.0 - routing_rtol)
    below_150k = effective_magnification < 150_000.0 * (1.0 - routing_rtol)
    at_most_400k = effective_magnification <= 400_000.0 * (1.0 + routing_rtol)
    selected_region: CertifiedRegion2D | None = None
    feature_family = "global_multiscale"
    routing_scale_m: float | None = None
    if atlas is not None:
        if region_name is not None:
            selected_region = atlas.region(region_name)
        elif not at_most_400k:
            selected_region = atlas.silicon_lattice_region
        elif effective_magnification > 40_000.0 * (1.0 + routing_rtol):
            multilayers = [region for region in regions if region.kind == "multilayer"]
            selected_region = max(multilayers, key=lambda region: region.layer_count)

        if selected_region is not None and selected_region.kind == "silicon_lattice":
            feature_family = "silicon_lattice_vectors"
            routing_scale_m = float(selected_region.lattice_spacing_m)
        elif selected_region is not None and selected_region.kind == "perforation":
            feature_family = "perforation_edge"
            routing_scale_m = atlas.certificate.interface_width_m
        elif at_least_40k and below_150k:
            feature_family = "multilayer_period"
            routing_scale_m = atlas.certificate.multilayer_period_m
        elif not below_150k and at_most_400k:
            feature_family = "individual_layer_widths"
            routing_scale_m = min(
                atlas.certificate.silicon_layer_width_m,
                atlas.certificate.silicon_germanium_layer_width_m,
            )

    field_centre_local = np.asarray(
        selected_region.centre_xy_m if selected_region is not None else (0.0, 0.0),
        dtype=np.float64,
    )
    # SampleView stores the world location of the atlas-local origin.  This
    # calibrated translation places the selected registered field on-axis.
    sample_translation = world_field_centre - projection @ field_centre_local

    half_x, half_y = 0.5 * nx * px, 0.5 * ny * py
    detector_corners = np.asarray(
        ((-half_x, -half_y), (-half_x, half_y), (half_x, -half_y), (half_x, half_y))
    )
    specimen_world_corners = np.concatenate(
        [detector_corners @ candidate_inverse.T for candidate_inverse in all_inverses],
        axis=0,
    )
    # Row-vector equivalent of R.T @ (world - translation).  Keeping the
    # translation here is essential when a registered off-axis atlas region is
    # moved to the detector axis.
    specimen_local_corners = (
        specimen_world_corners + world_field_centre - sample_translation
    ) @ local_from_world.T
    margin_values = (
        ("legacy", margin_m),
        ("lens_wobble", lens_wobble_margin_m),
        ("calibrated_input", calibrated_input_margin_m),
        ("gaussian_support", gaussian_support_margin_m),
        ("pose_uncertainty", pose_uncertainty_margin_m),
        ("interpolation_halo", interpolation_halo_margin_m),
    )
    checked_margins: list[tuple[str, float]] = []
    for name, value in margin_values:
        value = float(value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} margin must be finite and non-negative.")
        checked_margins.append((name, value))
    total_margin = float(sum(value for _, value in checked_margins))
    xmin = float(specimen_local_corners[:, 0].min() - total_margin)
    xmax = float(specimen_local_corners[:, 0].max() + total_margin)
    ymin = float(specimen_local_corners[:, 1].min() - total_margin)
    ymax = float(specimen_local_corners[:, 1].max() + total_margin)

    specimen_pixel_min = min(
        float(
            np.linalg.svd(
                local_from_world @ candidate_inverse @ np.diag((px, py)),
                compute_uv=False,
            ).min()
        )
        for candidate_inverse in all_inverses
    )
    # Generic rasters have no spectral certificate, so detector sampling is
    # their conservative LOD requirement.  Certified analytic tiles instead
    # use their smallest routed physical feature; oversampling the resulting
    # C2 field on the detector does not require inventing finer source texels.
    # Detector sampling is an unconditional acceptance constraint.  Certified
    # feature/beam criteria may demand a finer source raster, but may never
    # relax the explicit p_specimen / texels_per_detector_pixel bound.
    requirements: list[float] = [
        specimen_pixel_min / float(texels_per_detector_pixel)
    ]
    if gaussian_radius_min_m is not None:
        gaussian_radius = float(gaussian_radius_min_m)
        if not np.isfinite(gaussian_radius) or gaussian_radius <= 0.0:
            raise ValueError("gaussian_radius_min_m must be finite and positive.")
        requirements.append(gaussian_radius / samples_per_gaussian_radius)

    visible_features: list[CertifiedFeature2D] = []
    detector_scale = np.diag((1.0 / px, 1.0 / py))

    def projected_pixels(scale_m: float, direction_local=(1.0, 0.0)) -> float:
        vector_local = np.asarray(direction_local, dtype=np.float64) * scale_m
        vector_world = projection @ vector_local
        return float(np.linalg.norm(detector_scale @ (jacobian @ vector_world)))

    for feature in features:
        cx, cy = feature.centre_xy_m
        support_x, support_y = 3.0 * feature.sigma_xy_m[0], 3.0 * feature.sigma_xy_m[1]
        intersects = (
            cx + support_x >= xmin
            and cx - support_x <= xmax
            and cy + support_y >= ymin
            and cy - support_y <= ymax
        )
        direction_local = np.asarray(
            (np.cos(feature.orientation_rad), np.sin(feature.orientation_rad))
        )
        detector_pixels = projected_pixels(
            feature.characteristic_scale_m, direction_local
        )
        if (
            intersects
            and min_detector_pixels_per_feature
            <= detector_pixels
            <= max_detector_pixels_per_feature
        ):
            visible_features.append(feature)

    intersecting_regions: list[CertifiedRegion2D] = []
    unresolved_regions: list[CertifiedRegion2D] = []
    for region in regions:
        rxmin, rxmax, rymin, rymax = region.bounds_xyxy_m
        intersects = rxmax >= xmin and rxmin <= xmax and rymax >= ymin and rymin <= ymax
        if not intersects:
            continue
        intersecting_regions.append(region)
        if region.kind == "silicon_lattice":
            scale = float(region.lattice_spacing_m)
        elif region.kind == "multilayer":
            scale = (
                min(
                    atlas.certificate.silicon_layer_width_m,
                    atlas.certificate.silicon_germanium_layer_width_m,
                )
                if feature_family == "individual_layer_widths"
                else atlas.certificate.multilayer_period_m
            )
        else:
            scale = atlas.certificate.interface_width_m
        pixels = projected_pixels(scale)
        intervals = (xmax - xmin) / scale
        source_resolves = (
            atlas.region_pyramids[region.name].levels[0].max_spacing_m
            <= scale / samples_per_feature_period * (1.0 + 1e-12)
        )
        if not (
            min_detector_pixels_per_feature
            <= pixels
            <= max_detector_pixels_per_feature
            and intervals >= min_visible_feature_intervals
            and source_resolves
        ):
            unresolved_regions.append(region)

    if routing_scale_m is not None:
        requirements.append(routing_scale_m / samples_per_feature_period)
    elif visible_features:
        requirements.append(
            min(feature.characteristic_scale_m for feature in visible_features)
            / samples_per_feature_period
        )
    elif atlas is not None:
        requirements.append(atlas.pyramid.levels[0].max_spacing_m)
    if not requirements:
        requirements.append(specimen_pixel_min / float(texels_per_detector_pixel))
    required_spacing = min(requirements)

    if selected_region is None:
        pyramid = atlas.pyramid if atlas is not None else atlas_or_pyramid
    else:
        assert atlas is not None
        pyramid = atlas.pyramid_for_region(selected_region.name)

    # Periodic rasters are registered unit cells, not finite specimen tiles.
    # Decide this before LOD coverage is tested so a requested footprint larger
    # than the stored unit cell can still select a resolution-adequate level.
    if boundary is None:
        effective_boundary: BoundaryMode = (
            "periodic"
            if selected_region is not None
            and selected_region.kind == "silicon_lattice"
            else (atlas.boundary_default if atlas is not None else "vacuum")
        )
    else:
        effective_boundary = boundary
    periodic_source = effective_boundary == "periodic"

    adequate: list[int] = []
    for index, level in enumerate(pyramid.levels):
        lxmin, lxmax, lymin, lymax = level.bounds_xyxy_m
        tolerance = max(level.max_spacing_m, required_spacing) * 1e-9
        covers = periodic_source or (
            lxmin <= xmin + tolerance
            and lxmax >= xmax - tolerance
            and lymin <= ymin + tolerance
            and lymax >= ymax - tolerance
        )
        if covers and level.max_spacing_m <= required_spacing * (1.0 + 1e-12):
            adequate.append(index)
    if adequate:
        level_index = adequate[-1]
        meets_resolution = True
    else:
        # Always return a usable view, but make undersampling machine-visible.
        level_index = 0
        level = pyramid.levels[0]
        lxmin, lxmax, lymin, lymax = level.bounds_xyxy_m
        tolerance = max(level.max_spacing_m, required_spacing) * 1e-9
        covers = periodic_source or (
            lxmin <= xmin + tolerance
            and lxmax >= xmax - tolerance
            and lymin <= ymin + tolerance
            and lymax >= ymax - tolerance
        )
        meets_resolution = (
            covers and level.max_spacing_m <= required_spacing * (1.0 + 1e-12)
        )

    selected = pyramid.levels[level_index]
    projected_feature = (
        projected_pixels(routing_scale_m) if routing_scale_m is not None else None
    )
    if routing_scale_m is None:
        visible_intervals = None
        meets_feature_sampling = True
    else:
        visible_intervals = (
            min(xmax - xmin, ymax - ymin) / routing_scale_m
            if feature_family == "silicon_lattice_vectors"
            else (xmax - xmin) / routing_scale_m
        )
        meets_feature_sampling = bool(
            min_detector_pixels_per_feature
            <= projected_feature
            <= max_detector_pixels_per_feature
            and visible_intervals >= min_visible_feature_intervals
        )
    halo_knots = 1 if effective_boundary == "periodic" else 0

    # Construct only the fixed FOV crop's spline coefficients.  This is
    # important for the dense authoritative multilayer level: a high-
    # magnification acquisition needs only its fixed crop, not a full-tile JAX
    # field.
    crop_halo_knots = 0
    view_level = selected
    if effective_boundary != "periodic":
        crop_halo_knots = (
            atlas.interpolation_halo_knots
            if atlas is not None and atlas.interpolation_halo_knots > 0
            else 8
        )
        ix0 = max(0, int(np.searchsorted(selected.x_coords_m, xmin, side="left")) - crop_halo_knots)
        ix1 = min(
            selected.x_coords_m.size,
            int(np.searchsorted(selected.x_coords_m, xmax, side="right")) + crop_halo_knots,
        )
        iy0 = max(0, int(np.searchsorted(selected.y_coords_m, ymin, side="left")) - crop_halo_knots)
        iy1 = min(
            selected.y_coords_m.size,
            int(np.searchsorted(selected.y_coords_m, ymax, side="right")) + crop_halo_knots,
        )
        if ix1 - ix0 < 4:
            ix0 = max(0, min(ix0, selected.x_coords_m.size - 4))
            ix1 = ix0 + 4
        if iy1 - iy0 < 4:
            iy0 = max(0, min(iy0, selected.y_coords_m.size - 4))
            iy1 = iy0 + 4
        if (ix0, ix1, iy0, iy1) != (
            0,
            selected.x_coords_m.size,
            0,
            selected.y_coords_m.size,
        ):
            view_level = SampleLevel2D(
                selected.pathlength_yx_m[iy0:iy1, ix0:ix1],
                selected.log_amplitude_yx[iy0:iy1, ix0:ix1],
                selected.x_coords_m[ix0:ix1],
                selected.y_coords_m[iy0:iy1],
                name=f"{selected.name}:frozen-crop",
            )
    selection = ViewSelection(
        level_index=level_index,
        feature_names=tuple(feature.name for feature in visible_features),
        region_names=tuple(region.name for region in intersecting_regions),
        unresolved_region_names=tuple(region.name for region in unresolved_regions),
        selected_region_name=(selected_region.name if selected_region is not None else None),
        feature_family=feature_family,
        field_centre_local_xy_m=tuple(float(value) for value in field_centre_local),
        sample_translation_xy_m=tuple(float(value) for value in sample_translation),
        sample_tilt_xy_rad=tuple(float(value) for value in sample_tilt_xy_rad),
        projected_path_scale=projected_path_scale,
        local_bounds_xyxy_m=(xmin, xmax, ymin, ymax),
        selected_source_bounds_xyxy_m=view_level.bounds_xyxy_m,
        margin_components_m=tuple(checked_margins),
        total_margin_m=total_margin,
        crop_halo_knots=crop_halo_knots,
        interpolation_halo_knots=halo_knots,
        required_spacing_m=float(required_spacing),
        selected_spacing_m=selected.max_spacing_m,
        specimen_pixel_min_m=specimen_pixel_min,
        effective_magnification=effective_magnification,
        routed_feature_scale_m=routing_scale_m,
        projected_feature_pixels=projected_feature,
        visible_feature_intervals=visible_intervals,
        meets_feature_sampling=meets_feature_sampling,
        meets_resolution=bool(meets_resolution and meets_feature_sampling),
    )
    vacuum_discs = (
        tuple(
            (
                float(region.centre_xy_m[0]),
                float(region.centre_xy_m[1]),
                0.5 * float(atlas.certificate.perforation_diameter_m),
                float(atlas.certificate.interface_width_m),
            )
            for region in atlas.certificate.regions
            if region.kind == "perforation"
        )
        if atlas is not None
        else ()
    )
    return SampleView.from_level(
        view_level,
        centre_xy_m=sample_translation,
        rotation_rad=sample_rotation_rad,
        tilt_xy_rad=sample_tilt_xy_rad,
        boundary=effective_boundary,
        z=z,
        level_index=level_index,
        selection=selection,
        vacuum_discs_local_xy_radius_m=vacuum_discs,
    )


class SampleRender(NamedTuple):
    pathlength_yx_m: jnp.ndarray
    log_amplitude_yx: jnp.ndarray
    transmission_yx: jnp.ndarray


def render_sample_view(
    view: SampleView,
    x_coords_m,
    y_coords_m,
    *,
    voltage: float | None = None,
    wavelength_m: float | None = None,
) -> SampleRender:
    """Rasterize a view on world-coordinate pixel centres."""

    wavelength = _wavelength_from_voltage_or_value(
        voltage=voltage, wavelength_m=wavelength_m
    )
    x = jnp.asarray(x_coords_m, dtype=jnp.float64)
    y = jnp.asarray(y_coords_m, dtype=jnp.float64)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x_coords_m and y_coords_m must be one-dimensional.")
    xx, yy = jnp.meshgrid(x, y, indexing="xy")
    actions = view.evaluate_actions(jnp.stack((xx, yy), axis=-1))
    pathlength = actions[..., 0]
    log_amplitude = actions[..., 1]
    transmission = jnp.exp(
        log_amplitude + 1j * (2.0 * jnp.pi / wavelength) * pathlength
    )
    return SampleRender(pathlength, log_amplitude, transmission)


class DerivativeValidation(NamedTuple):
    gradient_max_abs_error: jnp.ndarray
    hessian_max_abs_error: jnp.ndarray


@dataclass(frozen=True)
class SampleLevelValidation:
    """Host-side decision certificate for a Gaussian/sample interaction."""

    diagnostic: Any
    recommended_route: Literal["local_action", "sample_plane_reexpansion"]
    weighted_refinement_fraction: float
    flagged_beams: int
    total_beams: int
    minimum_recommended_width_scale: float
    reason: str

    @property
    def use_local_action(self) -> bool:
        return self.recommended_route == "local_action"


def validate_sample_level(
    view: SampleView,
    gaussian_bundle: GaussianBeam,
    *,
    phase_tol: float = 0.1,
    amplitude_tol: float = 0.02,
    support_radius: float = 2.0,
    max_weighted_refinement_fraction: float = 0.0,
    **diagnostic_kwargs,
) -> SampleLevelValidation:
    """Validate a view and route it to local action or exact re-expansion.

    The routing decision is deliberately host-side: adaptive topology must not
    change within a compiled inverse-problem loss.  Beam importance is weighted
    by incident field magnitude squared, while any non-finite diagnostic is
    always marked for re-expansion.
    """

    allowed = float(max_weighted_refinement_fraction)
    if not 0.0 <= allowed <= 1.0 or not np.isfinite(allowed):
        raise ValueError("max_weighted_refinement_fraction must lie in [0, 1].")
    diagnostic = view.sampling_diagnostic(
        gaussian_bundle,
        phase_tol=phase_tol,
        amplitude_tol=amplitude_tol,
        support_radius=support_radius,
        **diagnostic_kwargs,
    )
    needs = np.asarray(diagnostic.needs_refinement, dtype=bool).reshape(-1).copy()
    metrics = (
        diagnostic.phase_max_abs,
        diagnostic.phase_rms,
        diagnostic.amplitude_max_abs,
        diagnostic.amplitude_rms,
        diagnostic.recommended_width_scale,
    )
    finite = np.ones_like(needs)
    for metric in metrics:
        finite &= np.isfinite(np.asarray(metric).reshape(-1))
    needs |= ~finite

    bundle = gaussian_bundle.to_vector()
    weights = np.abs(np.asarray(bundle.amplitude).reshape(-1)) ** 2
    if weights.size != needs.size:
        weights = np.broadcast_to(weights, needs.shape)
    total_weight = float(np.sum(weights))
    if not np.isfinite(total_weight) or total_weight <= 0.0:
        weights = np.ones_like(needs, dtype=np.float64)
        total_weight = float(weights.size)
    weighted_fraction = float(np.sum(weights[needs]) / total_weight)
    flagged = int(np.count_nonzero(needs))
    width_scale = np.asarray(diagnostic.recommended_width_scale, dtype=float)
    finite_width_scale = width_scale[np.isfinite(width_scale)]
    minimum_scale = (
        float(np.min(finite_width_scale)) if finite_width_scale.size else 0.0
    )

    if flagged == 0 or weighted_fraction <= allowed:
        route = "local_action"
        reason = "local quadratic phase/log-amplitude residuals pass"
    else:
        route = "sample_plane_reexpansion"
        reason = (
            f"{flagged}/{needs.size} beams ({weighted_fraction:.3%} incident weight) "
            "require refinement"
        )
    return SampleLevelValidation(
        diagnostic=diagnostic,
        recommended_route=route,
        weighted_refinement_fraction=weighted_fraction,
        flagged_beams=flagged,
        total_beams=int(needs.size),
        minimum_recommended_width_scale=minimum_scale,
        reason=reason,
    )


def validate_sample_view_derivatives(
    view: SampleView, xy_world: jnp.ndarray
) -> DerivativeValidation:
    """Compare explicit spline jets with nested JAX differentiation."""

    xy = jnp.asarray(xy_world, dtype=jnp.float64)
    if xy.shape != (2,):
        raise ValueError("derivative validation expects one xy point.")
    explicit = view.action_jet(xy)
    explicit_gradient = jnp.stack(
        (explicit.pathlength.gradient_xy, explicit.log_amplitude.gradient_xy)
    )
    explicit_hessian = jnp.stack(
        (explicit.pathlength.hessian_xy, explicit.log_amplitude.hessian_xy)
    )
    jacobian = jax.jacobian(view.evaluate_actions)(xy)
    hessian = jnp.stack(
        (
            jax.hessian(lambda point: view.evaluate_actions(point)[0])(xy),
            jax.hessian(lambda point: view.evaluate_actions(point)[1])(xy),
        )
    )
    return DerivativeValidation(
        jnp.max(jnp.abs(explicit_gradient - jacobian)),
        jnp.max(jnp.abs(explicit_hessian - hessian)),
    )


__all__ = [
    "BoundaryMode",
    "CertifiedFeature2D",
    "CertifiedLayerBoundary1D",
    "CertifiedRegion2D",
    "CubicActionField2D",
    "CubicInterpolatedSample2D",
    "DerivativeValidation",
    "MAGICALCertificate",
    "MAGICALSampleAtlas",
    "SampleActionJet2D",
    "SampleLevel2D",
    "SampleLevelValidation",
    "SamplePyramid2D",
    "SampleRender",
    "SampleView",
    "ScalarActionJet2D",
    "ViewSelection",
    "render_sample_view",
    "select_sample_view",
    "validate_sample_level",
    "validate_sample_view_derivatives",
]
