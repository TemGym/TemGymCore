"""Reusable forward-model pieces for NEOARM calibration experiments.

This module joins the calibrated sample representation to the first-order
column model without hiding either approximation.  It provides a frozen
multiresolution Gaussian topology for the fast path and an independent
Collins-integral evaluator for validation.  The inverse problem itself lives
in :mod:`temgym_core.microscope_inversion`.

The functions here deliberately work in SI units.  Optical states use the
interleaved convention ``[x, theta_x, y, theta_y]`` and images use normal
``(y, x)`` array ordering.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy import integrate, linalg, ndimage
from scipy.optimize import brentq, least_squares, linear_sum_assignment

from .constants import energy2wavelength
from .calibration_covariance import (
    FourierObservableCovarianceAugmentation,
    deduplicate_calibration_priors,
    gaussian_prior_from_metadata,
    gaussian_prior_to_metadata,
)
from .evaluate import evaluate_gaussians, gpu_kernel_available
from .gaussian import FreeSpacePropagator, GaussianBeam, make_gaussian
from .layer_metrology import (
    LayerObservableCovarianceAugmentation,
    RegisteredLayerMetrology,
    layer_observable_covariance_augmentation,
)
from .microscope_inversion import drift_matrix, hard_edge_solenoid_matrix
from .microscope_inversion import (
    AcquisitionPlan,
    FrameObservationBlock,
    HeterogeneousInversionDataset,
    SamplePlaneState,
)


_POSITION = np.asarray((0, 2), dtype=int)
_ANGLE = np.asarray((1, 3), dtype=int)


def _pair_float(name: str, value: Sequence[float]) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two entries.")
    out = (float(value[0]), float(value[1]))
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite.")
    return out


@dataclass(frozen=True)
class DetectorGeometry:
    """Physical detector definition using pixel-centre coordinates.

    ``affine_xy`` maps column-frame displacements about ``centre_xy_m`` into
    calibrated detector displacements.  Rendering therefore evaluates the
    optical field at the inverse-affine pixel coordinates.  Keeping that
    direction explicit makes the same calibration act on image centroids,
    covariance, scale, rotation, and shear.  The identity default exactly
    preserves the historical detector model.
    """

    shape_yx: tuple[int, int] = (256, 256)
    pixel_size_yx_m: tuple[float, float] = (55e-6, 55e-6)
    z_m: float = 1.04375
    centre_xy_m: tuple[float, float] = (0.0, 0.0)
    mtf_sigma_px: float = 0.0
    affine_xy: np.ndarray = field(default_factory=lambda: np.eye(2, dtype=float))

    def __post_init__(self) -> None:
        shape = tuple(int(v) for v in self.shape_yx)
        if len(shape) != 2 or min(shape) < 2:
            raise ValueError("shape_yx must contain two integers >= 2.")
        pixel = _pair_float("pixel_size_yx_m", self.pixel_size_yx_m)
        if min(pixel) <= 0.0:
            raise ValueError("pixel_size_yx_m must be positive.")
        centre = _pair_float("centre_xy_m", self.centre_xy_m)
        if not np.isfinite(self.z_m):
            raise ValueError("z_m must be finite.")
        if not np.isfinite(self.mtf_sigma_px) or self.mtf_sigma_px < 0.0:
            raise ValueError("mtf_sigma_px must be finite and non-negative.")
        affine = np.asarray(self.affine_xy, dtype=float)
        if affine.shape != (2, 2) or not np.all(np.isfinite(affine)):
            raise ValueError("affine_xy must be a finite 2x2 matrix.")
        determinant = float(np.linalg.det(affine))
        if not np.isfinite(determinant) or abs(determinant) <= np.finfo(float).eps:
            raise ValueError("affine_xy must be invertible.")
        affine = affine.copy()
        affine.setflags(write=False)
        object.__setattr__(self, "shape_yx", shape)
        object.__setattr__(self, "pixel_size_yx_m", pixel)
        object.__setattr__(self, "centre_xy_m", centre)
        object.__setattr__(self, "affine_xy", affine)

    @property
    def y_coords_m(self) -> np.ndarray:
        ny = self.shape_yx[0]
        return (
            (np.arange(ny, dtype=float) - 0.5 * (ny - 1))
            * self.pixel_size_yx_m[0]
            + self.centre_xy_m[1]
        )

    @property
    def x_coords_m(self) -> np.ndarray:
        nx = self.shape_yx[1]
        return (
            (np.arange(nx, dtype=float) - 0.5 * (nx - 1))
            * self.pixel_size_yx_m[1]
            + self.centre_xy_m[0]
        )

    @property
    def coords_xy_m(self) -> np.ndarray:
        """Calibrated detector coordinates of every pixel centre."""

        xx, yy = np.meshgrid(self.x_coords_m, self.y_coords_m, indexing="xy")
        return np.stack((xx, yy), axis=-1)

    @property
    def optical_coords_xy_m(self) -> np.ndarray:
        """Pixel centres expressed in the column's detector-plane frame."""

        return self.detector_to_column_xy(self.coords_xy_m)

    @property
    def corners_xy_m(self) -> np.ndarray:
        """Corner pixel centres in calibrated detector coordinates."""

        x = self.x_coords_m[[0, -1]]
        y = self.y_coords_m[[0, -1]]
        return np.asarray(
            ((x[0], y[0]), (x[1], y[0]), (x[1], y[1]), (x[0], y[1]))
        )

    @property
    def optical_corners_xy_m(self) -> np.ndarray:
        """Corner pixel centres expressed in the column's detector frame."""

        return self.detector_to_column_xy(self.corners_xy_m)

    def column_to_detector_xy(self, coords_xy_m: np.ndarray) -> np.ndarray:
        """Apply the calibrated column-to-detector affine to coordinates."""

        coords = np.asarray(coords_xy_m, dtype=float)
        if coords.shape[-1:] != (2,) or not np.all(np.isfinite(coords)):
            raise ValueError("coords_xy_m must be a finite array ending in dimension 2.")
        centre = np.asarray(self.centre_xy_m, dtype=float)
        return centre + np.einsum("ij,...j->...i", self.affine_xy, coords - centre)

    def detector_to_column_xy(self, coords_xy_m: np.ndarray) -> np.ndarray:
        """Map calibrated detector coordinates back to the column frame."""

        coords = np.asarray(coords_xy_m, dtype=float)
        if coords.shape[-1:] != (2,) or not np.all(np.isfinite(coords)):
            raise ValueError("coords_xy_m must be a finite array ending in dimension 2.")
        centre = np.asarray(self.centre_xy_m, dtype=float)
        return centre + np.einsum(
            "ij,...j->...i", np.linalg.inv(self.affine_xy), coords - centre
        )


@dataclass(frozen=True)
class HardEdgeLensInitial:
    """Initial geometry and raw-DAC optical law for one round lens.

    ``q_bias`` and ``q_slope_per_dac`` parameterize integrated Larmor strength,
    not electrical current.  A zero ``channel_index`` is valid; ``None`` means
    that the lens control is held fixed while its physical bias, position, and
    length may still be inferred.
    """

    name: str
    z_center_m: float
    length_m: float
    q_bias: float
    q_slope_per_dac: float = 0.0
    channel_index: int | None = None
    reference_dac: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Lens name must be non-empty.")
        values = (
            self.z_center_m,
            self.length_m,
            self.q_bias,
            self.q_slope_per_dac,
            self.reference_dac,
        )
        if not np.all(np.isfinite(values)) or self.length_m <= 0.0:
            raise ValueError("Lens initial values must be finite and length positive.")
        if self.channel_index is not None and int(self.channel_index) < 0:
            raise ValueError("channel_index must be non-negative or None.")

    @property
    def entrance_z(self) -> float:
        return self.z_center_m - 0.5 * self.length_m

    @property
    def exit_z(self) -> float:
        return self.z_center_m + 0.5 * self.length_m


@dataclass(frozen=True)
class HardEdgeColumnParameterization:
    """Differentiable full-column parameter layout with a fixed sample gauge.

    Every lens contributes centre position, log physical length, and integrated
    field bias.  The seven controllable lenses additionally contribute a raw-
    DAC slope.  Objective control values are held fixed, but objective geometry
    and field bias remain available to the inverse problem.  Static ordering is
    established from the initial geometry and is therefore safe inside JIT.
    """

    lenses: tuple[HardEdgeLensInitial, ...]
    sample_z_m: float
    source_z_m: float
    detector_z_m: float
    channel_names: tuple[str, ...]

    def __post_init__(self) -> None:
        lenses = tuple(self.lenses)
        channels = tuple(str(v) for v in self.channel_names)
        if not lenses:
            raise ValueError("At least one lens is required.")
        if len({lens.name for lens in lenses}) != len(lenses):
            raise ValueError("Lens names must be unique.")
        if not channels or len(set(channels)) != len(channels):
            raise ValueError("channel_names must be non-empty and unique.")
        if not np.all(np.isfinite((self.sample_z_m, self.source_z_m, self.detector_z_m))):
            raise ValueError("Source, sample, and detector positions must be finite.")
        if not self.source_z_m < self.sample_z_m < self.detector_z_m:
            raise ValueError("Expected source_z_m < sample_z_m < detector_z_m.")
        lenses = tuple(sorted(lenses, key=lambda lens: lens.z_center_m))
        previous_exit = self.source_z_m
        for lens in lenses:
            if not (
                self.source_z_m <= lens.entrance_z
                and lens.exit_z <= self.detector_z_m
            ):
                raise ValueError(f"Lens {lens.name!r} body lies outside the column.")
            if lens.entrance_z < previous_exit:
                raise ValueError(f"Lens {lens.name!r} overlaps the preceding lens.")
            if lens.entrance_z < self.sample_z_m < lens.exit_z:
                raise ValueError(
                    f"Lens {lens.name!r} straddles the fixed sample-plane gauge."
                )
            if lens.channel_index is not None and lens.channel_index >= len(channels):
                raise ValueError(f"Lens {lens.name!r} has an invalid channel index.")
            previous_exit = lens.exit_z
        object.__setattr__(self, "lenses", lenses)
        object.__setattr__(self, "channel_names", channels)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        names = ["source_z_m", "detector_z_m"]
        for lens in self.lenses:
            names.extend(
                (
                    f"{lens.name}.z_center_m",
                    f"{lens.name}.log_length_m",
                    f"{lens.name}.q_bias",
                )
            )
            if lens.channel_index is not None:
                names.append(f"{lens.name}.q_slope_per_dac")
        return tuple(names)

    @property
    def nominal_parameters(self) -> np.ndarray:
        values = [float(self.source_z_m), float(self.detector_z_m)]
        for lens in self.lenses:
            values.extend((lens.z_center_m, np.log(lens.length_m), lens.q_bias))
            if lens.channel_index is not None:
                values.append(lens.q_slope_per_dac)
        return np.asarray(values, dtype=float)

    @property
    def n_parameters(self) -> int:
        return len(self.parameter_names)

    def _decoded_lenses(self, parameters: jnp.ndarray):
        cursor = 2
        decoded = []
        for lens in self.lenses:
            z_center = parameters[cursor]
            length = jnp.exp(parameters[cursor + 1])
            q_bias = parameters[cursor + 2]
            cursor += 3
            if lens.channel_index is None:
                q_slope = jnp.asarray(0.0, dtype=parameters.dtype)
            else:
                q_slope = parameters[cursor]
                cursor += 1
            decoded.append((lens, z_center, length, q_bias, q_slope))
        return decoded

    def matrix(
        self,
        parameters: np.ndarray,
        dac_values: np.ndarray,
        *,
        segment: str = "full",
    ) -> jnp.ndarray:
        """Return the full or sample-to-detector exact hard-edge matrix."""

        parameters = jnp.asarray(parameters)
        dacs = jnp.asarray(dac_values)
        if parameters.shape != (self.n_parameters,):
            raise ValueError(f"parameters must have shape ({self.n_parameters},).")
        if dacs.shape != (len(self.channel_names),):
            raise ValueError(
                f"dac_values must have shape ({len(self.channel_names)},)."
            )
        if segment not in {"full", "source_to_sample", "sample_to_detector"}:
            raise ValueError(
                "segment must be 'full', 'source_to_sample', or "
                "'sample_to_detector'."
            )

        source_z = parameters[0]
        detector_z = parameters[1]
        if segment in {"full", "source_to_sample"}:
            axial_cursor = source_z
            include = tuple(
                True
                if segment == "full"
                else lens.exit_z <= self.sample_z_m
                for lens in self.lenses
            )
        else:
            axial_cursor = jnp.asarray(self.sample_z_m, dtype=parameters.dtype)
            # The discrete membership cannot switch during optimization.
            include = tuple(lens.entrance_z >= self.sample_z_m for lens in self.lenses)

        total = jnp.eye(4, dtype=parameters.dtype)
        decoded = self._decoded_lenses(parameters)
        for should_include, values in zip(include, decoded, strict=True):
            if not should_include:
                continue
            lens, z_center, length, q_bias, q_slope = values
            entrance = z_center - 0.5 * length
            total = drift_matrix(entrance - axial_cursor) @ total
            if lens.channel_index is None:
                dac_displacement = jnp.asarray(0.0, dtype=parameters.dtype)
            else:
                dac_displacement = dacs[lens.channel_index] - lens.reference_dac
            q = q_bias + q_slope * dac_displacement
            total = hard_edge_solenoid_matrix(q, length) @ total
            axial_cursor = z_center + 0.5 * length
        axial_end = (
            jnp.asarray(self.sample_z_m, dtype=parameters.dtype)
            if segment == "source_to_sample"
            else detector_z
        )
        return drift_matrix(axial_end - axial_cursor) @ total

    def source_to_sample_matrix(
        self, parameters: np.ndarray, dac_values: np.ndarray
    ) -> jnp.ndarray:
        """Return the illumination transfer map up to the specimen plane."""

        return self.matrix(parameters, dac_values, segment="source_to_sample")

    def sample_position_jacobian(
        self, parameters: np.ndarray, dac_values: np.ndarray
    ) -> jnp.ndarray:
        """Return d(x_d,y_d)/d(x_s,y_s) for view selection."""

        matrix = self.matrix(parameters, dac_values, segment="sample_to_detector")
        return matrix[jnp.ix_(jnp.asarray((0, 2)), jnp.asarray((0, 2)))]

    def geometry_clearances(self, parameters: np.ndarray) -> jnp.ndarray:
        """Return physical edge clearances, all of which must stay non-negative.

        Entries cover source/lens, adjacent-lens, specimen-side, and
        lens/detector separations.  This is JAX-compatible and can therefore be
        appended to a fit as a barrier or monitored as a hard validity gate.
        """

        values = jnp.asarray(parameters)
        if values.shape != (self.n_parameters,):
            raise ValueError(f"parameters must have shape ({self.n_parameters},).")
        decoded = self._decoded_lenses(values)
        clearances: list[jnp.ndarray] = []
        previous_exit = values[0]
        for lens, centre, length, _q_bias, _q_slope in decoded:
            entrance = centre - 0.5 * length
            exit_ = centre + 0.5 * length
            clearances.append(entrance - previous_exit)
            if lens.exit_z <= self.sample_z_m:
                clearances.append(jnp.asarray(self.sample_z_m) - exit_)
            else:
                clearances.append(entrance - jnp.asarray(self.sample_z_m))
            previous_exit = exit_
        clearances.append(values[1] - previous_exit)
        return jnp.stack(clearances)

    def default_bounds(
        self,
        *,
        position_half_width_m: float = 5e-3,
        length_ratio: tuple[float, float] = (0.25, 4.0),
        q_half_width: float = 2.0,
        slope_half_width: float = 2.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct broad finite bounds without asserting identifiability."""

        if position_half_width_m <= 0.0 or q_half_width <= 0.0 or slope_half_width <= 0.0:
            raise ValueError("Bound widths must be positive.")
        low_ratio, high_ratio = _pair_float("length_ratio", length_ratio)
        if not 0.0 < low_ratio < high_ratio:
            raise ValueError("length_ratio must satisfy 0 < low < high.")
        nominal = self.nominal_parameters
        lower = nominal.copy()
        upper = nominal.copy()
        lower[:2] -= position_half_width_m
        upper[:2] += position_half_width_m
        # Partition each side of the specimen into non-overlapping axial cells.
        # Independent box bounds can then never move/expand a hard-edge body
        # through the specimen or another lens.
        cells: list[tuple[float, float]] = []
        for index, lens in enumerate(self.lenses):
            on_source_side = lens.exit_z <= self.sample_z_m
            lower_cell = self.source_z_m if on_source_side else self.sample_z_m
            upper_cell = self.sample_z_m if on_source_side else self.detector_z_m
            if index > 0:
                previous = self.lenses[index - 1]
                previous_same_side = (previous.exit_z <= self.sample_z_m) == on_source_side
                if previous_same_side:
                    lower_cell = 0.5 * (previous.exit_z + lens.entrance_z)
            if index + 1 < len(self.lenses):
                following = self.lenses[index + 1]
                following_same_side = (following.exit_z <= self.sample_z_m) == on_source_side
                if following_same_side:
                    upper_cell = 0.5 * (lens.exit_z + following.entrance_z)
            cells.append((lower_cell, upper_cell))
        cursor = 2
        for lens, (lower_cell, upper_cell) in zip(self.lenses, cells, strict=True):
            edge_clearance = min(
                lens.entrance_z - lower_cell,
                upper_cell - lens.exit_z,
            )
            safe_position_half_width = min(
                position_half_width_m,
                max(0.0, 0.45 * edge_clearance),
            )
            lower[cursor] -= safe_position_half_width
            upper[cursor] += safe_position_half_width
            lower[cursor + 1] = np.log(lens.length_m * low_ratio)
            safe_max_length = 2.0 * min(
                lens.z_center_m - safe_position_half_width - lower_cell,
                upper_cell - lens.z_center_m - safe_position_half_width,
            )
            requested_max_length = lens.length_m * high_ratio
            maximum_length = min(requested_max_length, safe_max_length)
            if maximum_length <= lens.length_m * low_ratio:
                raise ValueError(
                    f"Requested bounds leave no physical thickness range for {lens.name!r}."
                )
            upper[cursor + 1] = np.log(maximum_length)
            lower[cursor + 2] -= q_half_width
            upper[cursor + 2] += q_half_width
            cursor += 3
            if lens.channel_index is not None:
                lower[cursor] -= slope_half_width
                upper[cursor] += slope_half_width
                cursor += 1
        return lower, upper


@dataclass(frozen=True)
class MagnificationDACDesign:
    """Raw-DAC settings and focus diagnostics for nominal imaging modes.

    The imaging column remains round for every raw-DAC combination, so a
    specimen-conjugate image state is characterized by the downstream radial
    position block ``a`` matching the requested magnification while the radial
    angle-to-position block ``b`` vanishes.  Diagnostics therefore report both
    the achieved isotropic magnification and the residual specimen-conjugacy
    ``B`` norm.
    """

    target_magnifications: np.ndarray
    dac_values: np.ndarray
    achieved_magnifications: np.ndarray
    image_rotations_rad: np.ndarray
    anisotropy_ratios: np.ndarray
    path_coordinates: np.ndarray
    specimen_conjugacy_b_values: np.ndarray
    specimen_conjugacy_b_norms: np.ndarray
    feasible_mask: np.ndarray
    residual_norms: np.ndarray

    def __post_init__(self) -> None:
        targets = np.asarray(self.target_magnifications, dtype=float)
        dacs = np.asarray(self.dac_values, dtype=float)
        achieved = np.asarray(self.achieved_magnifications, dtype=float)
        rotations = np.asarray(self.image_rotations_rad, dtype=float)
        anisotropy = np.asarray(self.anisotropy_ratios, dtype=float)
        coordinates = np.asarray(self.path_coordinates, dtype=float)
        b_values = np.asarray(self.specimen_conjugacy_b_values, dtype=float)
        b_norms = np.asarray(self.specimen_conjugacy_b_norms, dtype=float)
        feasible = np.asarray(self.feasible_mask, dtype=bool)
        residuals = np.asarray(self.residual_norms, dtype=float)
        count = targets.size
        if targets.ndim != 1 or count == 0 or np.any(targets <= 0.0):
            raise ValueError("target_magnifications must be a positive 1-D array.")
        if dacs.ndim != 2 or dacs.shape[0] != count:
            raise ValueError("dac_values must contain one row per target.")
        for name, values in (
            ("achieved_magnifications", achieved),
            ("image_rotations_rad", rotations),
            ("anisotropy_ratios", anisotropy),
            ("path_coordinates", coordinates),
            ("specimen_conjugacy_b_values", b_values),
            ("specimen_conjugacy_b_norms", b_norms),
            ("residual_norms", residuals),
        ):
            if values.shape != (count,) or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"{name} must be a finite vector with one value per target."
                )
        if feasible.shape != (count,):
            raise ValueError("feasible_mask must contain one Boolean per target.")
        if not np.all(np.isfinite(dacs)) or np.any(achieved <= 0.0):
            raise ValueError("Designed controls and magnifications must be finite.")
        if np.any(anisotropy < 1.0):
            raise ValueError("anisotropy_ratios use sigma_max/sigma_min and must be >= 1.")
        if np.any(b_norms < 0.0) or np.any(residuals < 0.0):
            raise ValueError("B norms and residual norms must be non-negative.")
        for name, values in (
            ("target_magnifications", targets),
            ("dac_values", dacs),
            ("achieved_magnifications", achieved),
            ("image_rotations_rad", rotations),
            ("anisotropy_ratios", anisotropy),
            ("path_coordinates", coordinates),
            ("specimen_conjugacy_b_values", b_values),
            ("specimen_conjugacy_b_norms", b_norms),
            ("residual_norms", residuals),
        ):
            values = values.copy()
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        feasible = feasible.copy()
        feasible.setflags(write=False)
        object.__setattr__(self, "feasible_mask", feasible)

    @property
    def maximum_relative_error(self) -> float:
        return float(
            np.max(
                np.abs(self.achieved_magnifications / self.target_magnifications - 1.0)
            )
        )

    @property
    def maximum_specimen_conjugacy_b_norm(self) -> float:
        return float(np.max(self.specimen_conjugacy_b_norms))

    @property
    def all_targets_feasible(self) -> bool:
        return bool(np.all(self.feasible_mask))

    @property
    def infeasible_targets(self) -> np.ndarray:
        return self.target_magnifications[~self.feasible_mask]


class MagnificationDACInfeasibleError(RuntimeError):
    """Requested nominal magnifications cannot be focused within the DAC bounds."""

    def __init__(self, design: MagnificationDACDesign):
        self.design = design
        lines = [
            "Requested nominal imaging states are not focus-feasible within the declared raw-DAC bounds."
        ]
        for index in np.flatnonzero(~design.feasible_mask):
            lines.append(
                "target="
                f"{design.target_magnifications[index]:g}, "
                f"achieved={design.achieved_magnifications[index]:.9g}, "
                f"B_norm={design.specimen_conjugacy_b_norms[index]:.9g}, "
                f"dacs={np.array2string(design.dac_values[index], precision=6, separator=', ')}"
            )
        super().__init__("\n".join(lines))


@dataclass(frozen=True)
class SignedObjectDefocusDesign:
    """A conjugate-magnification, signed object-defocus DAC pair.

    Rows are always ordered as negative then positive object defocus.  The
    object-defocus convention is

    ``0.5 * trace(A^{-1} B)``,

    where ``A`` and ``B`` are the sample-to-detector position and
    angle-to-position blocks.  For the round hard-edge column, preserving the
    scalar magnification preserves the radial part of ``A``.  Its cumulative
    Larmor rotation is reported separately and can optionally be constrained
    when at least three independent downstream controls are selected.
    """

    selected_channel_names: tuple[str, ...]
    selected_dac_lower_bounds: np.ndarray
    selected_dac_upper_bounds: np.ndarray
    nominal_dac_values: np.ndarray
    dac_values: np.ndarray
    electron_voltage_v: float
    wavelength_m: float
    feature_spacing_m: float
    nominal_magnification: float
    nominal_object_defocus_m: float
    target_object_defocus_m: np.ndarray
    achieved_magnifications: np.ndarray
    achieved_object_defocus_m: np.ndarray
    cumulative_rotation_changes_rad: np.ndarray
    preserve_rotation: bool
    feasible_mask: np.ndarray
    residual_norms: np.ndarray
    optimizer_success_mask: np.ndarray
    optimizer_status: np.ndarray
    optimizer_messages: tuple[str, ...]
    constraint_jacobian_singular_values: np.ndarray

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.selected_channel_names)
        lower = np.asarray(self.selected_dac_lower_bounds, dtype=float)
        upper = np.asarray(self.selected_dac_upper_bounds, dtype=float)
        nominal = np.asarray(self.nominal_dac_values, dtype=float)
        dacs = np.asarray(self.dac_values, dtype=float)
        targets = np.asarray(self.target_object_defocus_m, dtype=float)
        magnifications = np.asarray(self.achieved_magnifications, dtype=float)
        defocus = np.asarray(self.achieved_object_defocus_m, dtype=float)
        rotation = np.asarray(self.cumulative_rotation_changes_rad, dtype=float)
        feasible = np.asarray(self.feasible_mask, dtype=bool)
        residuals = np.asarray(self.residual_norms, dtype=float)
        success = np.asarray(self.optimizer_success_mask, dtype=bool)
        status = np.asarray(self.optimizer_status, dtype=int)
        singular_values = np.asarray(
            self.constraint_jacobian_singular_values,
            dtype=float,
        )
        messages = tuple(str(message) for message in self.optimizer_messages)

        if not names or len(set(names)) != len(names):
            raise ValueError("selected_channel_names must be non-empty and unique.")
        if (
            lower.shape != (len(names),)
            or upper.shape != (len(names),)
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(lower >= upper)
        ):
            raise ValueError("Selected DAC bounds must be finite increasing vectors.")
        if nominal.ndim != 1 or not np.all(np.isfinite(nominal)):
            raise ValueError("nominal_dac_values must be a finite vector.")
        if dacs.shape != (2, nominal.size) or not np.all(np.isfinite(dacs)):
            raise ValueError("dac_values must be a finite (2, n_channels) array.")
        for name, values in (
            ("target_object_defocus_m", targets),
            ("achieved_magnifications", magnifications),
            ("achieved_object_defocus_m", defocus),
            ("cumulative_rotation_changes_rad", rotation),
            ("residual_norms", residuals),
        ):
            if values.shape != (2,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be a finite two-vector.")
        if not targets[0] < 0.0 < targets[1] or not np.isclose(
            targets[0],
            -targets[1],
            rtol=1e-12,
            atol=0.0,
        ):
            raise ValueError("target_object_defocus_m must contain [-d, +d].")
        if np.any(magnifications <= 0.0) or np.any(residuals < 0.0):
            raise ValueError("Magnifications must be positive and residuals non-negative.")
        for name, values in (
            ("feasible_mask", feasible),
            ("optimizer_success_mask", success),
            ("optimizer_status", status),
        ):
            if values.shape != (2,):
                raise ValueError(f"{name} must contain one entry per signed state.")
        if len(messages) != 2:
            raise ValueError("optimizer_messages must contain one message per signed state.")
        if (
            singular_values.shape != (2 + int(bool(self.preserve_rotation)),)
            or not np.all(np.isfinite(singular_values))
            or np.any(singular_values < 0.0)
        ):
            raise ValueError(
                "constraint_jacobian_singular_values has the wrong shape or values."
            )
        scalars = (
            self.electron_voltage_v,
            self.wavelength_m,
            self.feature_spacing_m,
            self.nominal_magnification,
            self.nominal_object_defocus_m,
        )
        if not np.all(np.isfinite(scalars)):
            raise ValueError("Signed-defocus scalar diagnostics must be finite.")
        if min(
            self.electron_voltage_v,
            self.wavelength_m,
            self.feature_spacing_m,
            self.nominal_magnification,
        ) <= 0.0:
            raise ValueError("Voltage, wavelength, spacing, and magnification must be positive.")

        object.__setattr__(self, "selected_channel_names", names)
        object.__setattr__(self, "optimizer_messages", messages)
        for name, values in (
            ("selected_dac_lower_bounds", lower),
            ("selected_dac_upper_bounds", upper),
            ("nominal_dac_values", nominal),
            ("dac_values", dacs),
            ("target_object_defocus_m", targets),
            ("achieved_magnifications", magnifications),
            ("achieved_object_defocus_m", defocus),
            ("cumulative_rotation_changes_rad", rotation),
            ("feasible_mask", feasible),
            ("residual_norms", residuals),
            ("optimizer_success_mask", success),
            ("optimizer_status", status),
            ("constraint_jacobian_singular_values", singular_values),
        ):
            frozen = values.copy()
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)

    @property
    def dac_perturbations(self) -> np.ndarray:
        values = self.dac_values - self.nominal_dac_values[None, :]
        values.setflags(write=False)
        return values

    @property
    def negative_dac_values(self) -> np.ndarray:
        return self.dac_values[0]

    @property
    def positive_dac_values(self) -> np.ndarray:
        return self.dac_values[1]

    @property
    def absolute_target_object_defocus_m(self) -> float:
        return float(self.target_object_defocus_m[1])

    @property
    def all_states_feasible(self) -> bool:
        return bool(np.all(self.feasible_mask))

    @property
    def magnification_relative_errors(self) -> np.ndarray:
        values = np.abs(
            self.achieved_magnifications / self.nominal_magnification - 1.0
        )
        values.setflags(write=False)
        return values


class SignedObjectDefocusInfeasibleError(RuntimeError):
    """A requested signed defocus pair is unreachable inside its DAC box."""

    def __init__(self, design: SignedObjectDefocusDesign):
        self.design = design
        lines = [
            "Requested signed object-defocus states are not feasible within "
            "the selected raw-DAC bounds: "
            f"controls={design.selected_channel_names!r}, "
            f"lower={np.array2string(design.selected_dac_lower_bounds, precision=9)}, "
            f"upper={np.array2string(design.selected_dac_upper_bounds, precision=9)}."
        ]
        for index in np.flatnonzero(~design.feasible_mask):
            sign = "negative" if index == 0 else "positive"
            lines.append(
                f"{sign}: target={design.target_object_defocus_m[index]:.9g} m, "
                f"achieved={design.achieved_object_defocus_m[index]:.9g} m, "
                f"magnification_error="
                f"{design.magnification_relative_errors[index]:.9g}, "
                f"rotation_change="
                f"{design.cumulative_rotation_changes_rad[index]:.9g} rad, "
                f"optimizer={design.optimizer_messages[index]!r}"
            )
        super().__init__("\n".join(lines))


def _round_radial_drift_matrix(distance: float) -> np.ndarray:
    return np.asarray(((1.0, float(distance)), (0.0, 1.0)), dtype=float)


def _round_radial_solenoid_matrix(q: float, length: float) -> np.ndarray:
    q = float(q)
    length = float(length)
    cosine = np.cos(q)
    sine = np.sin(q)
    if abs(q) < 1e-12:
        sinc = 1.0
    else:
        sinc = sine / q
    return np.asarray(
        (
            (cosine, length * sinc),
            (-(q / length) * sine, cosine),
        ),
        dtype=float,
    )


def _sample_to_detector_round_components(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    dac_values: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return the downstream radial matrix and accumulated image rotation."""

    values = np.asarray(parameters, dtype=float)
    dacs = np.asarray(dac_values, dtype=float)
    if values.shape != (column.n_parameters,):
        raise ValueError(f"parameters must have shape ({column.n_parameters},).")
    if dacs.shape != (len(column.channel_names),):
        raise ValueError(
            f"dac_values must have shape ({len(column.channel_names)},)."
        )
    cursor = 2
    axial_cursor = float(column.sample_z_m)
    detector_z = float(values[1])
    total = np.eye(2, dtype=float)
    rotation = 0.0
    include = tuple(lens.entrance_z >= column.sample_z_m for lens in column.lenses)
    for should_include, lens in zip(include, column.lenses, strict=True):
        z_center = float(values[cursor])
        length = float(np.exp(values[cursor + 1]))
        q_bias = float(values[cursor + 2])
        cursor += 3
        if lens.channel_index is None:
            q_slope = 0.0
        else:
            q_slope = float(values[cursor])
            cursor += 1
        if not should_include:
            continue
        entrance = z_center - 0.5 * length
        total = _round_radial_drift_matrix(entrance - axial_cursor) @ total
        if lens.channel_index is None:
            dac_displacement = 0.0
        else:
            dac_displacement = float(dacs[lens.channel_index] - lens.reference_dac)
        q = q_bias + q_slope * dac_displacement
        total = _round_radial_solenoid_matrix(q, length) @ total
        rotation -= q
        axial_cursor = z_center + 0.5 * length
    total = _round_radial_drift_matrix(detector_z - axial_cursor) @ total
    return total, rotation


def _sample_to_detector_design_diagnostics(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    dac_values: np.ndarray,
) -> tuple[float, float, float, float]:
    radial, rotation = _sample_to_detector_round_components(column, parameters, dac_values)
    a_value = float(radial[0, 0])
    b_value = float(radial[0, 1])
    alpha = a_value * np.exp(1j * rotation)
    jacobian = np.asarray(column.sample_position_jacobian(parameters, dac_values), dtype=float)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    return (
        float(abs(alpha)),
        float(np.angle(alpha)),
        float(singular_values[0] / singular_values[-1]),
        b_value,
    )


def design_nominal_magnification_dacs(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    target_magnifications: Sequence[float],
    *,
    base_dacs: Sequence[float] | None = None,
    path_direction: Sequence[float] | None = None,
    path_bounds: tuple[float, float] = (-25.0, 25.0),
    search_points: int = 8193,
    multistart_points: int = 33,
    magnification_rtol: float = 2e-5,
    specimen_conjugacy_b_norm_atol: float = 1e-6,
    require_feasible: bool = True,
) -> MagnificationDACDesign:
    """Solve reproducible focused raw-DAC rows for nominal imaging modes.

    The downstream round-solenoid chain reduces the imaging-state constraints
    to the radial sample-to-detector map ``[[a, b], [c, d]]``.  A focused
    specimen image therefore requires ``|a|`` to match the requested isotropic
    magnification while ``b`` vanishes.  Deterministic multistart plus
    continuation selects a bounded branch without pretending that every target
    is reachable inside the declared raw-DAC box.
    """

    targets = np.asarray(target_magnifications, dtype=float)
    if targets.ndim != 1 or targets.size == 0 or not np.all(np.isfinite(targets)):
        raise ValueError("target_magnifications must be a finite non-empty vector.")
    if np.any(targets <= 0.0):
        raise ValueError("target_magnifications must be positive.")
    parameters = np.asarray(parameters, dtype=float)
    if parameters.shape != (column.n_parameters,) or not np.all(np.isfinite(parameters)):
        raise ValueError(f"parameters must have shape ({column.n_parameters},).")
    n_channels = len(column.channel_names)
    base = (
        np.zeros(n_channels, dtype=float)
        if base_dacs is None
        else np.asarray(base_dacs, dtype=float)
    )
    if base.shape != (n_channels,) or not np.all(np.isfinite(base)):
        raise ValueError("base_dacs must be finite with one value per channel.")
    if path_direction is None:
        downstream = {"IL1", "IL2", "IL3", "PL1"}
        direction = np.asarray(
            [1.0 if name in downstream else 0.0 for name in column.channel_names],
            dtype=float,
        )
    else:
        direction = np.asarray(path_direction, dtype=float)
    if direction.shape != (n_channels,) or not np.all(np.isfinite(direction)):
        raise ValueError("path_direction must be finite with one value per channel.")
    if not np.any(direction != 0.0):
        raise ValueError("path_direction must change at least one control.")
    lower, upper = float(path_bounds[0]), float(path_bounds[1])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("path_bounds must be finite and increasing.")
    if int(search_points) != search_points or search_points < 257:
        raise ValueError("search_points must be an integer of at least 257.")
    if int(multistart_points) != multistart_points or multistart_points < 1:
        raise ValueError("multistart_points must be a positive integer.")
    if magnification_rtol < 0.0 or specimen_conjugacy_b_norm_atol < 0.0:
        raise ValueError("Solver tolerances must be non-negative.")

    active = np.flatnonzero(direction != 0.0)
    if active.size == 0:
        raise ValueError("path_direction must change at least one control.")
    lower_bounds = base.copy()
    upper_bounds = base.copy()
    step_lower = lower * direction[active]
    step_upper = upper * direction[active]
    lower_bounds[active] = base[active] + np.minimum(step_lower, step_upper)
    upper_bounds[active] = base[active] + np.maximum(step_lower, step_upper)
    if np.any(lower_bounds[active] >= upper_bounds[active]):
        raise ValueError("path_bounds and path_direction leave no admissible control range.")
    focus_scale = max(abs(float(parameters[1]) - float(column.sample_z_m)), 1e-12)

    def magnitude_at_coordinate(coordinate: float) -> float:
        dacs = base + float(coordinate) * direction
        radial, _rotation = _sample_to_detector_round_components(column, parameters, dacs)
        return float(abs(radial[0, 0]))

    coordinates = np.linspace(lower, upper, int(search_points), dtype=float)
    sampled = np.asarray([magnitude_at_coordinate(value) for value in coordinates], dtype=float)
    if not np.all(np.isfinite(sampled)):
        raise RuntimeError("The magnification path produced non-finite matrices.")

    path_seeds_by_target: list[np.ndarray] = []
    for target in targets:
        residual = sampled - target
        exact = np.flatnonzero(residual == 0.0)
        brackets = np.flatnonzero(residual[:-1] * residual[1:] < 0.0)
        seeds: list[float] = []
        if exact.size:
            seeds.extend(float(value) for value in coordinates[exact])
        if brackets.size:
            for index in brackets:
                a_bound = float(coordinates[index])
                b_bound = float(coordinates[index + 1])
                seeds.append(
                    float(
                        brentq(
                            lambda value: magnitude_at_coordinate(value) - float(target),
                            a_bound,
                            b_bound,
                            xtol=1e-8,
                            rtol=1e-10,
                            maxiter=60,
                        )
                    )
                )
        if not seeds:
            nearest = np.argsort(np.abs(np.log(np.maximum(sampled, np.finfo(float).tiny) / target)))
            seeds.extend(float(coordinates[index]) for index in nearest[: max(3, multistart_points // 8)])
        path_seeds_by_target.append(np.asarray(sorted(set(seeds)), dtype=float))

    selected_coordinates = np.full(targets.shape, np.nan, dtype=float)
    designed_rows: list[np.ndarray] = [np.zeros_like(base) for _ in range(targets.size)]
    achieved: list[float] = [0.0] * targets.size
    rotations: list[float] = [0.0] * targets.size
    anisotropies: list[float] = [0.0] * targets.size
    b_values: list[float] = [0.0] * targets.size
    b_norms: list[float] = [0.0] * targets.size
    feasible_mask = np.zeros(targets.size, dtype=bool)
    residual_norms = np.zeros(targets.size, dtype=float)

    active_base = base[active]
    active_lower = lower_bounds[active]
    active_upper = upper_bounds[active]
    sort_order = np.argsort(targets, kind="stable")
    branch_reference = active_base.copy()

    def objective_values(target: float, x_active: np.ndarray) -> tuple[float, float]:
        dacs = base.copy()
        dacs[active] = x_active
        radial, _rotation = _sample_to_detector_round_components(column, parameters, dacs)
        magnitude = max(abs(float(radial[0, 0])), np.finfo(float).tiny)
        return float(np.log(magnitude / target)), float(radial[0, 1])

    def seed_candidates(path_seed_coordinates: np.ndarray) -> list[np.ndarray]:
        candidates: list[np.ndarray] = [active_base.copy(), branch_reference.copy()]
        for coordinate in path_seed_coordinates[: int(multistart_points)]:
            seed = np.clip(base + coordinate * direction, lower_bounds, upper_bounds)
            candidates.append(seed[active].copy())
        if path_seed_coordinates.size:
            centre = float(path_seed_coordinates[np.argmin(np.abs(path_seed_coordinates))])
            candidates.append(
                np.clip(base + 0.5 * centre * direction, lower_bounds, upper_bounds)[active].copy()
            )
        if candidates:
            offset = candidates[-1] - active_base
            candidates.append(np.clip(active_base - offset, active_lower, active_upper))
        out: list[np.ndarray] = []
        for candidate in candidates:
            clipped = np.clip(np.asarray(candidate, dtype=float), active_lower, active_upper)
            if any(np.allclose(clipped, existing, atol=1e-12, rtol=0.0) for existing in out):
                continue
            out.append(clipped)
        return out

    for solved_count, target_index in enumerate(sort_order):
        target = float(targets[target_index])
        reference = branch_reference.copy()
        best_key: tuple[float, float, float, float] | None = None
        best_result: tuple[np.ndarray, float, float, float, float] | None = None
        for candidate_seed in seed_candidates(path_seeds_by_target[target_index]):
            def residual(x_active: np.ndarray) -> np.ndarray:
                log_magnitude_error, b_value = objective_values(target, x_active)
                penalty_scale = 1e-6 / max(1.0, np.linalg.norm(active_upper - active_lower))
                return np.concatenate(
                    (
                        np.asarray((log_magnitude_error, b_value / focus_scale), dtype=float),
                        penalty_scale * (x_active - reference),
                    )
                )

            result = least_squares(
                residual,
                candidate_seed,
                bounds=(active_lower, active_upper),
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
                max_nfev=max(200, 250 * active.size),
            )
            x_active = np.asarray(result.x, dtype=float)
            dacs = base.copy()
            dacs[active] = x_active
            achieved_mag, rotation, anisotropy, b_value = _sample_to_detector_design_diagnostics(
                column,
                parameters,
                dacs,
            )
            b_norm = float(np.sqrt(2.0) * abs(b_value))
            relative_error = abs(achieved_mag / target - 1.0)
            feasible = (
                relative_error <= magnification_rtol
                and b_norm <= specimen_conjugacy_b_norm_atol
            )
            constraint_norm = float(
                np.hypot(np.log(max(achieved_mag, np.finfo(float).tiny) / target), b_norm / focus_scale)
            )
            key = (
                0.0 if feasible else 1.0,
                constraint_norm,
                float(np.linalg.norm(x_active - reference)),
                float(np.linalg.norm(x_active - active_base)),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_result = (dacs, achieved_mag, rotation, anisotropy, b_value)
        if best_result is None or best_key is None:
            raise RuntimeError("Unable to produce any nominal imaging-state candidate.")
        dacs, achieved_mag, rotation, anisotropy, b_value = best_result
        b_norm = float(np.sqrt(2.0) * abs(b_value))
        designed_rows[target_index] = dacs
        achieved[target_index] = achieved_mag
        rotations[target_index] = rotation
        anisotropies[target_index] = anisotropy
        b_values[target_index] = b_value
        b_norms[target_index] = b_norm
        feasible_mask[target_index] = best_key[0] == 0.0
        residual_norms[target_index] = best_key[1]
        if path_seeds_by_target[target_index].size:
            selected_coordinates[target_index] = float(
                path_seeds_by_target[target_index][
                    np.argmin(np.abs(path_seeds_by_target[target_index]))
                ]
            )
        branch_reference = dacs[active].copy()

    design = MagnificationDACDesign(
        target_magnifications=targets,
        dac_values=np.asarray(designed_rows, dtype=float),
        achieved_magnifications=np.asarray(achieved),
        image_rotations_rad=np.asarray(rotations),
        anisotropy_ratios=np.asarray(anisotropies),
        path_coordinates=selected_coordinates,
        specimen_conjugacy_b_values=np.asarray(b_values),
        specimen_conjugacy_b_norms=np.asarray(b_norms),
        feasible_mask=feasible_mask,
        residual_norms=residual_norms,
    )
    if require_feasible and not design.all_targets_feasible:
        raise MagnificationDACInfeasibleError(design)
    return design


def _sample_to_detector_defocus_metrics(
    column: HardEdgeColumnParameterization,
    parameters: jnp.ndarray,
    dac_values: jnp.ndarray,
) -> jnp.ndarray:
    """Return log magnification, effective defocus, and unwrapped rotation."""

    matrix = column.matrix(parameters, dac_values, segment="sample_to_detector")
    position = jnp.asarray((0, 2), dtype=int)
    angle = jnp.asarray((1, 3), dtype=int)
    a_block = matrix[jnp.ix_(position, position)]
    b_block = matrix[jnp.ix_(position, angle)]
    _determinant_sign, log_absolute_determinant = jnp.linalg.slogdet(a_block)
    log_magnification = 0.5 * log_absolute_determinant
    object_defocus = 0.5 * jnp.trace(jnp.linalg.solve(a_block, b_block))

    # This is the continuous Larmor angle, rather than a polar-decomposition
    # angle wrapped modulo 2*pi.  Segment membership is fixed by the initial
    # geometry, matching HardEdgeColumnParameterization.matrix.
    cumulative_rotation = jnp.asarray(0.0, dtype=parameters.dtype)
    for lens, _centre, _length, q_bias, q_slope in column._decoded_lenses(parameters):
        if lens.entrance_z < column.sample_z_m:
            continue
        if lens.channel_index is None:
            displacement = jnp.asarray(0.0, dtype=parameters.dtype)
        else:
            displacement = dac_values[lens.channel_index] - lens.reference_dac
        cumulative_rotation = cumulative_rotation - (q_bias + q_slope * displacement)
    return jnp.stack((log_magnification, object_defocus, cumulative_rotation))


def design_signed_object_defocus_dacs(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    nominal_dac_values: Sequence[float],
    *,
    feature_spacing_m: float,
    electron_voltage_v: float,
    selected_channels: Sequence[str] = ("IL2", "IL3"),
    selected_dac_bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    selected_dac_half_width: float | Sequence[float] = 0.05,
    preserve_rotation: bool = False,
    magnification_rtol: float = 1e-9,
    object_defocus_atol_m: float = 1e-13,
    rotation_atol_rad: float = 1e-9,
    max_nfev: int = 600,
    require_feasible: bool = True,
) -> SignedObjectDefocusDesign:
    """Design the two weak-phase contrast states for one known spacing.

    The requested object defocus is ``+/- d**2 / (2 * wavelength)``.  At this
    value a sinusoidal phase object with spacing ``d`` lies at the first
    quarter-period of the coherent phase-transfer function.  Selected raw-DAC
    controls are solved with SciPy least squares using an exact float64 JAX
    Jacobian of the hard-edge sample-to-detector map.

    With two downstream controls the solve preserves isotropic magnification
    and sets object defocus; the small accompanying Larmor rotation is
    measured and returned.  Set ``preserve_rotation=True`` with at least three
    controls to add the unwrapped cumulative-rotation constraint.  Bounds are
    specified only for the selected controls, in the same order as
    ``selected_channels``.  If omitted, each selected nominal DAC receives the
    declared symmetric half-width.
    """

    values = np.asarray(parameters, dtype=float)
    if values.shape != (column.n_parameters,) or not np.all(np.isfinite(values)):
        raise ValueError(f"parameters must have shape ({column.n_parameters},).")
    nominal = np.asarray(nominal_dac_values, dtype=float)
    n_channels = len(column.channel_names)
    if nominal.shape != (n_channels,) or not np.all(np.isfinite(nominal)):
        raise ValueError(
            f"nominal_dac_values must be a finite ({n_channels},) vector."
        )

    names = tuple(str(name) for name in selected_channels)
    if not names or len(set(names)) != len(names):
        raise ValueError("selected_channels must be non-empty and unique.")
    unknown = tuple(name for name in names if name not in column.channel_names)
    if unknown:
        raise ValueError(f"Unknown selected raw-DAC controls: {unknown!r}.")
    active = np.asarray(
        [column.channel_names.index(name) for name in names],
        dtype=int,
    )
    constraint_count = 2 + int(bool(preserve_rotation))
    if active.size < constraint_count:
        raise ValueError(
            f"At least {constraint_count} selected controls are required for "
            f"{constraint_count} signed-defocus constraints."
        )
    inactive_on_segment = tuple(
        name
        for name, channel_index in zip(names, active, strict=True)
        if not any(
            lens.channel_index == channel_index
            and lens.entrance_z >= column.sample_z_m
            for lens in column.lenses
        )
    )
    if inactive_on_segment:
        raise ValueError(
            "Selected controls do not act on the sample-to-detector segment: "
            f"{inactive_on_segment!r}."
        )

    spacing = float(feature_spacing_m)
    voltage = float(electron_voltage_v)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("feature_spacing_m must be finite and positive.")
    if not np.isfinite(voltage) or voltage <= 0.0:
        raise ValueError("electron_voltage_v must be finite and positive.")
    tolerances = np.asarray(
        (magnification_rtol, object_defocus_atol_m, rotation_atol_rad),
        dtype=float,
    )
    if not np.all(np.isfinite(tolerances)) or np.any(tolerances <= 0.0):
        raise ValueError("All signed-defocus acceptance tolerances must be positive.")
    try:
        finite_max_nfev = float(max_nfev)
    except (TypeError, ValueError):
        finite_max_nfev = np.nan
    if (
        isinstance(max_nfev, (bool, np.bool_))
        or not np.isfinite(finite_max_nfev)
        or int(finite_max_nfev) != finite_max_nfev
    ):
        raise ValueError("max_nfev must be a positive integer.")
    if int(finite_max_nfev) < 1:
        raise ValueError("max_nfev must be a positive integer.")
    max_nfev = int(finite_max_nfev)

    nominal_active = nominal[active]
    if selected_dac_bounds is None:
        half_width = np.asarray(selected_dac_half_width, dtype=float)
        if half_width.ndim == 0:
            half_width = np.full(active.size, float(half_width), dtype=float)
        if (
            half_width.shape != (active.size,)
            or not np.all(np.isfinite(half_width))
            or np.any(half_width <= 0.0)
        ):
            raise ValueError(
                "selected_dac_half_width must be positive and scalar or one "
                "value per selected control."
            )
        lower = nominal_active - half_width
        upper = nominal_active + half_width
    else:
        if len(selected_dac_bounds) != 2:
            raise ValueError("selected_dac_bounds must contain lower and upper vectors.")
        lower = np.asarray(selected_dac_bounds[0], dtype=float)
        upper = np.asarray(selected_dac_bounds[1], dtype=float)
        if lower.ndim == 0:
            lower = np.full(active.size, float(lower), dtype=float)
        if upper.ndim == 0:
            upper = np.full(active.size, float(upper), dtype=float)
        if (
            lower.shape != (active.size,)
            or upper.shape != (active.size,)
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(lower >= upper)
        ):
            raise ValueError(
                "selected_dac_bounds must be finite increasing vectors with "
                "one entry per selected control."
            )
        if np.any(nominal_active < lower) or np.any(nominal_active > upper):
            raise ValueError("selected_dac_bounds must contain the nominal controls.")

    # SciPy calls the compiled residual synchronously from the host.  Keeping
    # this tiny solve on CPU avoids competing with the GPU forward renderer or
    # paying device-transfer overhead at every optimizer iteration.
    with jax.enable_x64(), jax.default_device(jax.devices("cpu")[0]):
        wavelength = float(np.asarray(energy2wavelength(jnp.float64(voltage))))
        if not np.isfinite(wavelength) or wavelength <= 0.0:
            raise RuntimeError("The relativistic wavelength calculation was non-finite.")
        target_absolute = float(spacing**2 / (2.0 * wavelength))
        if not np.isfinite(target_absolute) or target_absolute <= 0.0:
            raise ValueError("The feature spacing produced an invalid target defocus.")

        parameters_jax = jnp.asarray(values, dtype=jnp.float64)
        nominal_jax = jnp.asarray(nominal, dtype=jnp.float64)
        active_jax = jnp.asarray(active, dtype=int)
        nominal_metrics = np.asarray(
            _sample_to_detector_defocus_metrics(
                column,
                parameters_jax,
                nominal_jax,
            ),
            dtype=float,
        )
        nominal_matrix = np.asarray(
            column.matrix(
                parameters_jax,
                nominal_jax,
                segment="sample_to_detector",
            ),
            dtype=float,
        )
        nominal_a = nominal_matrix[np.ix_(_POSITION, _POSITION)]
        singular_a = np.linalg.svd(nominal_a, compute_uv=False)
        if (
            not np.all(np.isfinite(nominal_metrics))
            or not np.all(np.isfinite(singular_a))
            or singular_a[-1]
            <= np.finfo(float).eps * max(1.0, float(singular_a[0]))
        ):
            raise ValueError(
                "The nominal sample-to-detector A block is singular or non-finite."
            )
        nominal_magnification = float(np.exp(nominal_metrics[0]))
        rotation_scale = max(1e-5, 10.0 * float(rotation_atol_rad))
        constraint_scales = np.asarray(
            (1.0, target_absolute, rotation_scale)[:constraint_count],
            dtype=float,
        )
        constraint_scales_jax = jnp.asarray(
            constraint_scales,
            dtype=jnp.float64,
        )

        def metrics_from_active(active_values: jnp.ndarray) -> jnp.ndarray:
            controls = nominal_jax.at[active_jax].set(active_values)
            return _sample_to_detector_defocus_metrics(
                column,
                parameters_jax,
                controls,
            )

        scaled_metric_jacobian = jax.jit(
            jax.jacfwd(
                lambda x: metrics_from_active(x)[:constraint_count]
                / constraint_scales_jax
            )
        )
        active_jacobian = np.asarray(
            scaled_metric_jacobian(
                jnp.asarray(nominal_active, dtype=jnp.float64)
            ),
            dtype=float,
        )
        dimensionless_jacobian = active_jacobian * (
            0.5 * (upper - lower)
        )[None, :]
        singular_values = np.linalg.svd(
            dimensionless_jacobian,
            compute_uv=False,
        )
        if not np.all(np.isfinite(singular_values)):
            raise ValueError("The selected-control constraint Jacobian is non-finite.")
        rank_threshold = max(
            np.finfo(float).eps,
            (float(singular_values[0]) if singular_values.size else 0.0) * 1e-12,
        )
        constraint_rank = int(np.count_nonzero(singular_values > rank_threshold))
        if constraint_rank < constraint_count:
            raise ValueError(
                "Selected controls cannot independently set the requested "
                f"constraints: scaled Jacobian rank {constraint_rank}/"
                f"{constraint_count}, singular values="
                f"{np.array2string(singular_values, precision=6)}."
            )

        solved_dacs = np.repeat(nominal[None, :], 2, axis=0)
        achieved_magnifications = np.full(2, nominal_magnification, dtype=float)
        achieved_defocus = np.full(2, nominal_metrics[1], dtype=float)
        rotation_changes = np.zeros(2, dtype=float)
        feasible_mask = np.zeros(2, dtype=bool)
        residual_norms = np.full(2, np.finfo(float).max, dtype=float)
        success_mask = np.zeros(2, dtype=bool)
        status_values = np.zeros(2, dtype=int)
        messages: list[str] = ["not attempted", "not attempted"]
        target_values = np.asarray((-target_absolute, target_absolute), dtype=float)
        active_span = upper - lower

        for index, target in enumerate(target_values):
            target_metrics = nominal_metrics[:constraint_count].copy()
            target_metrics[1] = target
            if preserve_rotation:
                target_metrics[2] = nominal_metrics[2]
            target_metrics_jax = jnp.asarray(target_metrics, dtype=jnp.float64)

            def scaled_constraints(active_values: jnp.ndarray) -> jnp.ndarray:
                return (
                    metrics_from_active(active_values)[:constraint_count]
                    - target_metrics_jax
                ) / constraint_scales_jax

            include_regularizer = active.size > constraint_count

            def scaled_residual(active_values: jnp.ndarray) -> jnp.ndarray:
                constraints = scaled_constraints(active_values)
                if not include_regularizer:
                    return constraints
                regularizer = (
                    1e-8
                    * (active_values - jnp.asarray(nominal_active))
                    / jnp.asarray(active_span)
                )
                return jnp.concatenate((constraints, regularizer))

            residual_jit = jax.jit(scaled_residual)
            jacobian_jit = jax.jit(jax.jacfwd(scaled_residual))

            def scipy_residual(active_values: np.ndarray) -> np.ndarray:
                result = np.asarray(
                    residual_jit(jnp.asarray(active_values, dtype=jnp.float64)),
                    dtype=float,
                )
                if not np.all(np.isfinite(result)):
                    return np.full(result.shape, 1e100, dtype=float)
                return result

            def scipy_jacobian(active_values: np.ndarray) -> np.ndarray:
                result = np.asarray(
                    jacobian_jit(jnp.asarray(active_values, dtype=jnp.float64)),
                    dtype=float,
                )
                if not np.all(np.isfinite(result)):
                    return np.zeros(result.shape, dtype=float)
                return result

            residual_at_nominal = scipy_residual(nominal_active)[:constraint_count]
            constraint_jacobian = scipy_jacobian(nominal_active)[:constraint_count]
            linear_step = np.linalg.lstsq(
                constraint_jacobian,
                -residual_at_nominal,
                rcond=1e-12,
            )[0]
            seeds: list[np.ndarray] = [nominal_active.copy()]
            for factor in (0.5, 1.0, 1.5):
                seeds.append(
                    np.clip(
                        nominal_active + factor * linear_step,
                        lower,
                        upper,
                    )
                )
            unique_seeds: list[np.ndarray] = []
            for seed in seeds:
                if any(
                    np.allclose(seed, previous, rtol=0.0, atol=1e-14)
                    for previous in unique_seeds
                ):
                    continue
                unique_seeds.append(seed)

            best_key: tuple[float, float, float] | None = None
            best_payload: tuple[
                np.ndarray,
                np.ndarray,
                bool,
                int,
                str,
                float,
                bool,
            ] | None = None
            failure_messages: list[str] = []
            for seed in unique_seeds:
                try:
                    result = least_squares(
                        scipy_residual,
                        seed,
                        jac=scipy_jacobian,
                        bounds=(lower, upper),
                        x_scale="jac",
                        xtol=1e-13,
                        ftol=1e-13,
                        gtol=1e-13,
                        max_nfev=int(max_nfev),
                    )
                except (FloatingPointError, ValueError, np.linalg.LinAlgError) as error:
                    failure_messages.append(f"{type(error).__name__}: {error}")
                    continue
                candidate = np.asarray(result.x, dtype=float)
                candidate_metrics = np.asarray(
                    metrics_from_active(jnp.asarray(candidate, dtype=jnp.float64)),
                    dtype=float,
                )
                if not np.all(np.isfinite(candidate_metrics)):
                    failure_messages.append("candidate metrics were non-finite")
                    continue
                candidate_magnification = float(np.exp(candidate_metrics[0]))
                magnitude_error = abs(
                    candidate_magnification / nominal_magnification - 1.0
                )
                defocus_error = abs(float(candidate_metrics[1]) - target)
                rotation_error = abs(float(candidate_metrics[2] - nominal_metrics[2]))
                standardized = [
                    magnitude_error / float(magnification_rtol),
                    defocus_error / float(object_defocus_atol_m),
                ]
                if preserve_rotation:
                    standardized.append(rotation_error / float(rotation_atol_rad))
                standardized_norm = float(np.linalg.norm(standardized))
                feasible = bool(
                    result.success
                    and magnitude_error <= magnification_rtol
                    and defocus_error <= object_defocus_atol_m
                    and (not preserve_rotation or rotation_error <= rotation_atol_rad)
                )
                key = (
                    0.0 if feasible else 1.0,
                    standardized_norm,
                    float(np.linalg.norm((candidate - nominal_active) / active_span)),
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_payload = (
                        candidate,
                        candidate_metrics,
                        bool(result.success),
                        int(result.status),
                        str(result.message),
                        standardized_norm,
                        feasible,
                    )

            if best_payload is None:
                messages[index] = (
                    "; ".join(dict.fromkeys(failure_messages))
                    if failure_messages
                    else "No finite optimization candidate was produced."
                )
                continue
            (
                candidate,
                candidate_metrics,
                optimizer_success,
                optimizer_status,
                optimizer_message,
                standardized_norm,
                feasible,
            ) = best_payload
            solved_dacs[index, active] = candidate
            achieved_magnifications[index] = float(np.exp(candidate_metrics[0]))
            achieved_defocus[index] = float(candidate_metrics[1])
            rotation_changes[index] = float(candidate_metrics[2] - nominal_metrics[2])
            feasible_mask[index] = feasible
            residual_norms[index] = standardized_norm
            success_mask[index] = optimizer_success
            status_values[index] = optimizer_status
            messages[index] = optimizer_message

    design = SignedObjectDefocusDesign(
        selected_channel_names=names,
        selected_dac_lower_bounds=lower,
        selected_dac_upper_bounds=upper,
        nominal_dac_values=nominal,
        dac_values=solved_dacs,
        electron_voltage_v=voltage,
        wavelength_m=wavelength,
        feature_spacing_m=spacing,
        nominal_magnification=nominal_magnification,
        nominal_object_defocus_m=float(nominal_metrics[1]),
        target_object_defocus_m=target_values,
        achieved_magnifications=achieved_magnifications,
        achieved_object_defocus_m=achieved_defocus,
        cumulative_rotation_changes_rad=rotation_changes,
        preserve_rotation=bool(preserve_rotation),
        feasible_mask=feasible_mask,
        residual_norms=residual_norms,
        optimizer_success_mask=success_mask,
        optimizer_status=status_values,
        optimizer_messages=tuple(messages),
        constraint_jacobian_singular_values=singular_values,
    )
    if require_feasible and not design.all_states_feasible:
        raise SignedObjectDefocusInfeasibleError(design)
    return design


def hard_edge_column_from_config(config: dict[str, object]) -> HardEdgeColumnParameterization:
    """Build the differentiable column layout from the example TOML schema."""

    try:
        beam = config["beam"]
        detector = config["detector"]
        experiment = config["experiment"]
        lens_tables = config["lenses"]
        channel_names = tuple(experiment["controllable_lenses"])
    except (KeyError, TypeError) as error:
        raise ValueError("Configuration is missing a required NEOARM table.") from error
    if not isinstance(beam, dict) or not isinstance(detector, dict):
        raise ValueError("beam and detector configuration entries must be tables.")
    if not isinstance(experiment, dict) or not isinstance(lens_tables, dict):
        raise ValueError("experiment and lenses configuration entries must be tables.")

    channel_index = {str(name): index for index, name in enumerate(channel_names)}
    lenses = []
    for name, values in lens_tables.items():
        if not isinstance(values, dict):
            raise ValueError(f"Lens {name!r} configuration must be a table.")
        lens_name = str(name)
        fixed = bool(values.get("control_fixed", False))
        lenses.append(
            HardEdgeLensInitial(
                name=lens_name,
                z_center_m=float(values["z_center_m_initial"]),
                length_m=float(values["thickness_m_initial"]),
                q_bias=float(values["q_bias_initial"]),
                q_slope_per_dac=float(values.get("q_slope_per_dac_initial", 0.0)),
                channel_index=(None if fixed else channel_index[lens_name]),
                reference_dac=float(values.get("reference_dac", 0.0)),
            )
        )
    return HardEdgeColumnParameterization(
        lenses=tuple(lenses),
        sample_z_m=float(beam["sample_z_m"]),
        source_z_m=float(beam["source_z_m"]),
        detector_z_m=float(detector["z_m"]),
        channel_names=tuple(str(name) for name in channel_names),
    )


def frozen_topology_fingerprint(
    *,
    specimen_identity: str,
    atlas_version: str,
    region_name: str,
    level_index: int,
    crop_bounds_xyxy_m: Sequence[float],
    sample_translation_xy_m: Sequence[float],
    projection_world_from_local: np.ndarray,
    route: str,
    selected_spacing_m: float,
    routed_feature_scale_m: float | None,
    centres_per_period: float,
    overlap_factor: float,
    interpolation_halo_knots: int,
    topology_version: str = "reexpand-v1",
    centres_xy_m: np.ndarray | None = None,
    waist_m: np.ndarray | None = None,
) -> str:
    """Return a stable identity for one frozen sample/Gaussian topology.

    The human-readable prefix is deliberately insufficient on its own.  The
    digest also covers specimen/certificate identity, fixed crop and pose,
    local-to-world projection, cubic spacing, feature scale, interpolation
    halo, Gaussian density/overlap, and optional accepted centre/radius arrays.
    Two rows can therefore share an ID only when their numerical topology
    contract is identical.
    """

    identity = str(specimen_identity).strip()
    version = str(atlas_version).strip()
    region = str(region_name).strip()
    route_name = str(route).strip()
    topology_name = str(topology_version).strip()
    if not all((identity, version, region, route_name, topology_name)):
        raise ValueError("Topology identity strings must be non-empty.")
    if int(level_index) != level_index or level_index < 0:
        raise ValueError("level_index must be a non-negative integer.")
    bounds = np.asarray(crop_bounds_xyxy_m, dtype=np.float64)
    translation = np.asarray(sample_translation_xy_m, dtype=np.float64)
    projection = np.asarray(projection_world_from_local, dtype=np.float64)
    if bounds.shape != (4,) or not np.all(np.isfinite(bounds)):
        raise ValueError("crop_bounds_xyxy_m must contain four finite values.")
    if not (bounds[0] < bounds[1] and bounds[2] < bounds[3]):
        raise ValueError("crop_bounds_xyxy_m must be increasing.")
    if translation.shape != (2,) or not np.all(np.isfinite(translation)):
        raise ValueError("sample_translation_xy_m must contain two finite values.")
    if projection.shape != (2, 2) or not np.all(np.isfinite(projection)):
        raise ValueError("projection_world_from_local must be a finite 2x2 matrix.")
    scalars = np.asarray(
        (selected_spacing_m, centres_per_period, overlap_factor), dtype=np.float64
    )
    if not np.all(np.isfinite(scalars)) or np.any(scalars <= 0.0):
        raise ValueError("Topology spacings, density, and overlap must be positive.")
    if routed_feature_scale_m is not None and (
        not np.isfinite(routed_feature_scale_m) or routed_feature_scale_m <= 0.0
    ):
        raise ValueError("routed_feature_scale_m must be positive or None.")
    if (
        int(interpolation_halo_knots) != interpolation_halo_knots
        or interpolation_halo_knots < 0
    ):
        raise ValueError("interpolation_halo_knots must be a non-negative integer.")

    def float_hex(values: np.ndarray) -> list[str]:
        return [float(value).hex() for value in np.asarray(values).ravel()]

    payload = {
        "specimen_identity": identity,
        "atlas_version": version,
        "region_name": region,
        "level_index": int(level_index),
        "crop_bounds_xyxy_m": float_hex(bounds),
        "sample_translation_xy_m": float_hex(translation),
        "projection_world_from_local": float_hex(projection),
        "route": route_name,
        "selected_spacing_m": float(selected_spacing_m).hex(),
        "routed_feature_scale_m": (
            None
            if routed_feature_scale_m is None
            else float(routed_feature_scale_m).hex()
        ),
        "centres_per_period": float(centres_per_period).hex(),
        "overlap_factor": float(overlap_factor).hex(),
        "interpolation_halo_knots": int(interpolation_halo_knots),
        "topology_version": topology_name,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for name, values in (("centres_xy_m", centres_xy_m), ("waist_m", waist_m)):
        if values is None:
            continue
        array = np.ascontiguousarray(values, dtype="<f8")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")
        if name == "centres_xy_m" and (
            array.ndim != 2 or array.shape[1] != 2 or not len(array)
        ):
            raise ValueError("centres_xy_m must have shape (n, 2), n > 0.")
        if name == "waist_m" and (
            array.ndim != 1 or not len(array) or np.any(array <= 0.0)
        ):
            raise ValueError("waist_m must be a positive one-dimensional array.")
        digest.update(name.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    safe_region = "".join(
        character if character.isalnum() else "-" for character in region
    )
    return (
        f"{safe_region}:L{int(level_index)}:{topology_name}:"
        f"{digest.hexdigest()[:16]}"
    )


@dataclass(frozen=True)
class FrameRequest:
    """One detector exposure in the compact calibration acquisition."""

    frame_id: int
    optical_state_index: int
    optical_state_label: str
    measurement: str
    sample_state: SamplePlaneState
    raw_dac_values: tuple[float, ...] = ()
    target_magnification: float | None = None
    reference_mode_magnification: float | None = None
    predicted_effective_magnification: float | None = None
    dac_approach: str = "forward"
    dac_approach_start_values: tuple[float, ...] = ()
    dac_ramp_steps: int = 1
    settling_time_s: float = 0.0
    feature_family: str = "auto"
    selected_region_name: str | None = None
    sample_level_index: int | None = None
    sample_translation_xy_m: tuple[float, float] = (0.0, 0.0)
    sample_crop_bounds_xyxy_m: tuple[float, float, float, float] | None = None
    selected_spacing_m: float | None = None
    repeat_index: int = 0
    calibration_covariance: np.ndarray = field(
        default_factory=lambda: np.zeros((4, 4), dtype=float)
    )
    interaction_route: str = "auto"
    frozen_topology_id: str | None = None

    def __post_init__(self) -> None:
        allowed_approaches = {"forward", "reverse"}
        allowed_routes = {
            "auto",
            "vacuum",
            "local_action",
            "sample_plane_reexpansion",
        }
        if self.dac_approach not in allowed_approaches:
            raise ValueError(f"dac_approach must be one of {sorted(allowed_approaches)}.")
        if self.interaction_route not in allowed_routes:
            raise ValueError(
                f"interaction_route must be one of {sorted(allowed_routes)}."
            )
        if self.frozen_topology_id is not None:
            topology_id = str(self.frozen_topology_id).strip()
            if not topology_id:
                raise ValueError("frozen_topology_id must be non-empty or None.")
            object.__setattr__(self, "frozen_topology_id", topology_id)
        if self.frame_id < 0 or self.optical_state_index < 0 or self.repeat_index < 0:
            raise ValueError("Frame and repeat indices must be non-negative.")
        dacs = tuple(float(value) for value in self.raw_dac_values)
        if not dacs or not np.all(np.isfinite(dacs)):
            raise ValueError("raw_dac_values must be non-empty and finite.")
        object.__setattr__(self, "raw_dac_values", dacs)
        approach_start = (
            dacs
            if not self.dac_approach_start_values
            else tuple(float(value) for value in self.dac_approach_start_values)
        )
        if len(approach_start) != len(dacs) or not np.all(np.isfinite(approach_start)):
            raise ValueError(
                "dac_approach_start_values must be empty or match raw_dac_values."
            )
        if int(self.dac_ramp_steps) != self.dac_ramp_steps or self.dac_ramp_steps < 1:
            raise ValueError("dac_ramp_steps must be a positive integer.")
        if not np.isfinite(self.settling_time_s) or self.settling_time_s < 0.0:
            raise ValueError("settling_time_s must be finite and non-negative.")
        object.__setattr__(self, "dac_approach_start_values", approach_start)
        object.__setattr__(self, "dac_ramp_steps", int(self.dac_ramp_steps))
        object.__setattr__(self, "settling_time_s", float(self.settling_time_s))
        object.__setattr__(
            self,
            "sample_translation_xy_m",
            _pair_float("sample_translation_xy_m", self.sample_translation_xy_m),
        )
        for attribute in (
            "target_magnification",
            "reference_mode_magnification",
            "predicted_effective_magnification",
        ):
            value = getattr(self, attribute)
            if value is not None:
                value = float(value)
                if not np.isfinite(value) or value <= 0.0:
                    raise ValueError(f"{attribute} must be finite and positive.")
                object.__setattr__(self, attribute, value)
        if self.sample_level_index is not None and self.sample_level_index < 0:
            raise ValueError("sample_level_index must be non-negative or None.")
        if self.sample_crop_bounds_xyxy_m is not None:
            bounds = tuple(float(value) for value in self.sample_crop_bounds_xyxy_m)
            if len(bounds) != 4 or not np.all(np.isfinite(bounds)):
                raise ValueError("sample_crop_bounds_xyxy_m must contain four finite values.")
            if not (bounds[0] < bounds[1] and bounds[2] < bounds[3]):
                raise ValueError("sample crop bounds must be increasing.")
            object.__setattr__(self, "sample_crop_bounds_xyxy_m", bounds)
        if self.selected_spacing_m is not None:
            spacing = float(self.selected_spacing_m)
            if not np.isfinite(spacing) or spacing <= 0.0:
                raise ValueError("selected_spacing_m must be finite and positive.")
            object.__setattr__(self, "selected_spacing_m", spacing)
        covariance = np.asarray(self.calibration_covariance, dtype=float)
        if covariance.shape != (4, 4) or not np.all(np.isfinite(covariance)):
            raise ValueError("calibration_covariance must be a finite 4x4 matrix.")
        if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-20):
            raise ValueError("calibration_covariance must be symmetric.")
        if np.linalg.eigvalsh(covariance).min() < -1e-15:
            raise ValueError("calibration_covariance must be positive semidefinite.")
        covariance = covariance.copy()
        covariance.setflags(write=False)
        object.__setattr__(self, "calibration_covariance", covariance)


def build_acquisition_manifest(
    plan: AcquisitionPlan,
    *,
    feature_family_by_state: dict[str, str] | None = None,
    view_selection_by_state: dict[str, object] | None = None,
    sample_state_covariance: np.ndarray | None = None,
    interaction_route_by_state: dict[str, str] | None = None,
    frozen_topology_id_by_state: dict[str, str] | None = None,
    approach_precondition_raw_dac_step: float | Sequence[float] = 0.0,
    dac_ramp_steps: int = 1,
    settling_time_s: float = 0.0,
) -> tuple[FrameRequest, ...]:
    """Expand the 30 optical settings into the default 230-frame protocol.

    Every setting receives perforation, position-basis, angle-basis, and sample
    frames.  Nominal settings repeat that core from the reverse DAC approach;
    held-out settings repeat the three vacuum/basis frames and reserve their
    eighth frame for indexed diffraction.  Five nominal states and one
    single-lens state receive orthogonal and negative basis inputs.  Three
    nominal plus all three held-out settings receive indexed diffraction.
    Remaining budget is assigned to sample repeats; it changes precision, not
    rank.  The approach start, ramp count, and settling time make the reverse
    approach an executable acquisition instruction rather than a label.
    """

    feature_map = {} if feature_family_by_state is None else dict(feature_family_by_state)
    selection_map = {} if view_selection_by_state is None else dict(view_selection_by_state)
    route_map = (
        {} if interaction_route_by_state is None else dict(interaction_route_by_state)
    )
    topology_map = (
        {} if frozen_topology_id_by_state is None else dict(frozen_topology_id_by_state)
    )
    covariance = (
        np.zeros((4, 4), dtype=float)
        if sample_state_covariance is None
        else np.asarray(sample_state_covariance, dtype=float)
    )
    approach_step = np.asarray(approach_precondition_raw_dac_step, dtype=float)
    if approach_step.ndim == 0:
        approach_step = np.full(len(plan.channel_names), float(approach_step))
    if (
        approach_step.shape != (len(plan.channel_names),)
        or not np.all(np.isfinite(approach_step))
        or np.any(approach_step < 0.0)
    ):
        raise ValueError(
            "approach_precondition_raw_dac_step must be a non-negative scalar "
            "or one value per channel."
        )
    if int(dac_ramp_steps) != dac_ramp_steps or dac_ramp_steps < 1:
        raise ValueError("dac_ramp_steps must be a positive integer.")
    if not np.isfinite(settling_time_s) or settling_time_s < 0.0:
        raise ValueError("settling_time_s must be finite and non-negative.")
    nominal_indices = [i for i, state in enumerate(plan.states) if state.role == "nominal"]
    held_out_indices = [i for i, state in enumerate(plan.states) if state.role == "held_out"]
    basis_rich_indices = list(nominal_indices)
    first_single = next(
        (i for i, state in enumerate(plan.states) if state.role == "single_lens"),
        None,
    )
    if first_single is not None:
        basis_rich_indices.append(first_single)
    basis_rich = frozenset(basis_rich_indices[:6])
    diffraction_rich = frozenset((*nominal_indices[:3], *held_out_indices))
    reference_mode_magnification = next(
        (
            state.target_magnification
            for state in plan.states
            if state.target_magnification is not None
            and np.allclose(state.dac_values, plan.reference_dacs, rtol=0.0, atol=1e-12)
        ),
        None,
    )
    requests: list[FrameRequest] = []

    def append(
        state_index: int,
        measurement: str,
        sample_state: SamplePlaneState,
        *,
        approach: str = "forward",
        repeat_index: int = 0,
    ) -> None:
        optical = plan.states[state_index]
        selection = selection_map.get(optical.label)
        endpoint = np.asarray(optical.dac_values, dtype=float)
        approach_sign = -1.0 if approach == "forward" else 1.0
        approach_start = tuple(endpoint + approach_sign * approach_step)
        uses_sample = measurement in {"sample", "sample_repeat", "indexed_diffraction"}
        if uses_sample:
            interaction_route = route_map.get(
                optical.label,
                getattr(selection, "interaction_route", "auto"),
            )
            frozen_topology_id = topology_map.get(
                optical.label,
                getattr(
                    selection,
                    "frozen_topology_id",
                    getattr(selection, "topology_id", None),
                ),
            )
        else:
            interaction_route = "vacuum"
            frozen_topology_id = None
        requests.append(
            FrameRequest(
                frame_id=len(requests),
                optical_state_index=state_index,
                optical_state_label=optical.label,
                measurement=measurement,
                sample_state=sample_state,
                raw_dac_values=optical.dac_values,
                target_magnification=optical.target_magnification,
                reference_mode_magnification=(
                    optical.target_magnification
                    if optical.target_magnification is not None
                    else reference_mode_magnification
                ),
                predicted_effective_magnification=getattr(
                    selection, "effective_magnification", None
                ),
                dac_approach=approach,
                dac_approach_start_values=approach_start,
                dac_ramp_steps=int(dac_ramp_steps),
                settling_time_s=float(settling_time_s),
                feature_family=feature_map.get(
                    optical.label,
                    getattr(selection, "feature_family", "auto"),
                ),
                selected_region_name=getattr(selection, "selected_region_name", None),
                sample_level_index=getattr(selection, "level_index", None),
                sample_translation_xy_m=getattr(
                    selection, "sample_translation_xy_m", (0.0, 0.0)
                ),
                sample_crop_bounds_xyxy_m=getattr(
                    selection, "local_bounds_xyxy_m", None
                ),
                selected_spacing_m=getattr(selection, "selected_spacing_m", None),
                interaction_route=interaction_route,
                frozen_topology_id=frozen_topology_id,
                repeat_index=repeat_index,
                calibration_covariance=covariance,
            )
        )

    core = (
        ("perforation", SamplePlaneState(label="on_axis_perforation")),
        (
            "position_basis",
            SamplePlaneState(x=plan.position_amplitude, label="position_x_plus"),
        ),
        (
            "angle_basis",
            SamplePlaneState(theta_x=plan.tilt_amplitude, label="tilt_x_plus"),
        ),
        ("sample", SamplePlaneState(label="certified_sample")),
    )
    for state_index, optical in enumerate(plan.states):
        state_start = len(requests)
        for measurement, sample_state in core:
            append(state_index, measurement, sample_state)
        if optical.role in {"nominal", "held_out"}:
            reverse_core = core if optical.role == "nominal" else core[:3]
            for measurement, sample_state in reverse_core:
                append(
                    state_index,
                    measurement,
                    sample_state,
                    approach="reverse",
                )
        if state_index in basis_rich:
            append(
                state_index,
                "basis_orthogonal",
                SamplePlaneState(y=plan.position_amplitude, label="position_y_plus"),
            )
            append(
                state_index,
                "basis_negative",
                SamplePlaneState(theta_x=-plan.tilt_amplitude, label="tilt_x_minus"),
            )
        if state_index in diffraction_rich:
            append(
                state_index,
                "indexed_diffraction",
                SamplePlaneState(label="indexed_Si_diffraction"),
            )

        used = len(requests) - state_start
        if used > optical.frame_count:
            raise ValueError(
                f"Frame budget for {optical.label!r} is too small for the protocol."
            )
        for repeat_index in range(optical.frame_count - used):
            append(
                state_index,
                "sample_repeat",
                SamplePlaneState(label="certified_sample_repeat"),
                repeat_index=repeat_index + 1,
            )

    if len(requests) != plan.total_frames:
        raise RuntimeError("Expanded manifest does not match the plan frame budget.")
    return tuple(requests)


@dataclass(frozen=True)
class RefinementRegion:
    """A rectangular area that retains a shorter physical feature scale."""

    bounds_xyxy_m: tuple[float, float, float, float]
    shortest_period_m: float
    label: str = "fine"

    def __post_init__(self) -> None:
        bounds = tuple(float(v) for v in self.bounds_xyxy_m)
        if len(bounds) != 4 or not np.all(np.isfinite(bounds)):
            raise ValueError("bounds_xyxy_m must be four finite values.")
        xmin, xmax, ymin, ymax = bounds
        if xmax <= xmin or ymax <= ymin:
            raise ValueError("Refinement-region bounds must have positive area.")
        if not np.isfinite(self.shortest_period_m) or self.shortest_period_m <= 0:
            raise ValueError("shortest_period_m must be finite and positive.")
        object.__setattr__(self, "bounds_xyxy_m", bounds)


@dataclass(frozen=True)
class GaussianTopology:
    """A frozen set of Gaussian centres, radii, and represented cell areas."""

    centres_xy_m: np.ndarray
    waist_m: np.ndarray
    cell_area_m2: np.ndarray
    bounds_xyxy_m: tuple[float, float, float, float]
    centres_per_period: float
    feature_period_m: float
    labels: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        centres = np.asarray(self.centres_xy_m, dtype=float)
        waist = np.asarray(self.waist_m, dtype=float)
        area = np.asarray(self.cell_area_m2, dtype=float)
        if centres.ndim != 2 or centres.shape[1] != 2 or centres.shape[0] == 0:
            raise ValueError("centres_xy_m must have shape (n, 2), n > 0.")
        n = centres.shape[0]
        if waist.shape != (n,) or area.shape != (n,):
            raise ValueError("waist_m and cell_area_m2 must have shape (n,).")
        if not np.all(np.isfinite(centres)):
            raise ValueError("Gaussian centres must be finite.")
        if not np.all(np.isfinite(waist)) or np.any(waist <= 0.0):
            raise ValueError("Gaussian radii must be finite and positive.")
        if not np.all(np.isfinite(area)) or np.any(area <= 0.0):
            raise ValueError("Gaussian cell areas must be finite and positive.")
        bounds = tuple(float(v) for v in self.bounds_xyxy_m)
        if len(bounds) != 4:
            raise ValueError("bounds_xyxy_m must contain four values.")
        if self.centres_per_period < 8.0:
            raise ValueError("Use at least eight Gaussian centres per retained period.")
        if self.feature_period_m <= 0.0:
            raise ValueError("feature_period_m must be positive.")
        labels = tuple(str(value) for value in self.labels)
        if not labels:
            labels = tuple("default" for _ in range(n))
        if len(labels) != n:
            raise ValueError("labels must be empty or contain one entry per beamlet.")

        # Copy once and mark read-only.  The optimizer may change optical
        # parameters, but it must not silently change the quadrature topology.
        centres = np.array(centres, copy=True)
        waist = np.array(waist, copy=True)
        area = np.array(area, copy=True)
        centres.setflags(write=False)
        waist.setflags(write=False)
        area.setflags(write=False)
        object.__setattr__(self, "centres_xy_m", centres)
        object.__setattr__(self, "waist_m", waist)
        object.__setattr__(self, "cell_area_m2", area)
        object.__setattr__(self, "bounds_xyxy_m", bounds)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def beam_count(self) -> int:
        return int(self.centres_xy_m.shape[0])


def _centred_axis(lower: float, upper: float, step: float) -> np.ndarray:
    count = max(1, int(np.ceil((upper - lower) / step)))
    actual = (upper - lower) / count
    return lower + (np.arange(count, dtype=float) + 0.5) * actual


def _grid_for_region(
    bounds: tuple[float, float, float, float],
    step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax = bounds
    x = _centred_axis(xmin, xmax, step)
    y = _centred_axis(ymin, ymax, step)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    centres = np.stack((xx.ravel(), yy.ravel()), axis=-1)
    step_x = (xmax - xmin) / x.size
    step_y = (ymax - ymin) / y.size
    area = np.full(centres.shape[0], step_x * step_y)
    return centres, area, np.full(centres.shape[0], np.sqrt(step_x * step_y))


def build_multiresolution_gaussian_topology(
    bounds_xyxy_m: Sequence[float],
    shortest_period_m: float,
    *,
    centres_per_period: float = 8.0,
    overlap_factor: float = 2.0,
    refinement_regions: Sequence[RefinementRegion] = (),
    max_beamlets: int | None = None,
) -> GaussianTopology:
    """Build and freeze a coarse/fine Gaussian quadrature topology.

    ``max_beamlets`` is an optional operational guard, not an algorithmic
    ceiling.  Leaving it as ``None`` permits arbitrarily large topologies.
    """

    bounds = tuple(float(v) for v in bounds_xyxy_m)
    if len(bounds) != 4 or not np.all(np.isfinite(bounds)):
        raise ValueError("bounds_xyxy_m must contain four finite values.")
    xmin, xmax, ymin, ymax = bounds
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("bounds_xyxy_m must have positive area.")
    period = float(shortest_period_m)
    cpp = float(centres_per_period)
    overlap = float(overlap_factor)
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("shortest_period_m must be finite and positive.")
    if not np.isfinite(cpp) or cpp < 8.0:
        raise ValueError("centres_per_period must be finite and at least 8.")
    if not np.isfinite(overlap) or overlap <= 0.0:
        raise ValueError("overlap_factor must be finite and positive.")

    centres, areas, local_steps = _grid_for_region(bounds, period / cpp)
    labels = np.full(centres.shape[0], "coarse", dtype=object)

    # Replacing covered coarse cells avoids counting the same illuminated
    # area twice.  Regions may be nested; processing coarser periods first
    # leaves the final (shortest-period) region authoritative.
    regions = sorted(refinement_regions, key=lambda r: r.shortest_period_m, reverse=True)
    for region in regions:
        rx0, rx1, ry0, ry1 = region.bounds_xyxy_m
        inside = (
            (centres[:, 0] >= rx0)
            & (centres[:, 0] <= rx1)
            & (centres[:, 1] >= ry0)
            & (centres[:, 1] <= ry1)
        )
        centres = centres[~inside]
        areas = areas[~inside]
        local_steps = local_steps[~inside]
        labels = labels[~inside]
        new_centres, new_areas, new_steps = _grid_for_region(
            region.bounds_xyxy_m, region.shortest_period_m / cpp
        )
        centres = np.concatenate((centres, new_centres), axis=0)
        areas = np.concatenate((areas, new_areas), axis=0)
        local_steps = np.concatenate((local_steps, new_steps), axis=0)
        labels = np.concatenate(
            (labels, np.full(new_centres.shape[0], region.label, dtype=object)), axis=0
        )

    n = centres.shape[0]
    if max_beamlets is not None and n > int(max_beamlets):
        raise MemoryError(
            f"Requested topology contains {n:,} beamlets, exceeding the explicit "
            f"guard of {int(max_beamlets):,}.  Inspect estimate_gaussian_workload "
            "or remove the guard after choosing a streamed render configuration."
        )

    waists = overlap * local_steps
    return GaussianTopology(
        centres_xy_m=centres,
        waist_m=waists,
        cell_area_m2=areas,
        bounds_xyxy_m=bounds,
        centres_per_period=cpp,
        feature_period_m=min(
            [period, *(r.shortest_period_m for r in refinement_regions)]
        ),
        labels=tuple(str(v) for v in labels),
        metadata={
            "overlap_factor": overlap,
            "refinement_regions": tuple(r.label for r in regions),
            "topology_frozen": True,
        },
    )


@dataclass(frozen=True)
class TopologyRefinement:
    topology: GaussianTopology
    refined_parent_mask: np.ndarray
    previous_beam_count: int
    new_beam_count: int
    maximum_normalized_error: float


def refine_gaussian_topology(
    topology: GaussianTopology,
    normalized_error_by_state: np.ndarray,
    *,
    relative_power_by_state: np.ndarray | None = None,
    error_threshold: float = 1.0,
    power_floor: float = 1.0e-6,
    factor: int = 2,
    max_beamlets: int | None = None,
) -> TopologyRefinement:
    """Subdivide beamlet cells that fail in any compared optical state.

    The union over states is taken before subdivision.  Consequently the new
    topology can be frozen and used for every truth/initial/fitted or wobble
    state without introducing a state-dependent discretization into the loss.
    ``normalized_error_by_state`` may contain one row or ``(n_states, n_beams)``.
    """

    errors = np.asarray(normalized_error_by_state, dtype=float)
    if errors.ndim == 1:
        errors = errors[None, :]
    if errors.ndim != 2 or errors.shape[1] != topology.beam_count:
        raise ValueError("normalized_error_by_state must have shape (states, beamlets).")
    if not np.all(np.isfinite(errors)):
        raise ValueError("normalized errors must be finite.")
    if error_threshold <= 0.0 or not np.isfinite(error_threshold):
        raise ValueError("error_threshold must be finite and positive.")
    if not 0.0 <= power_floor <= 1.0:
        raise ValueError("power_floor must lie in [0, 1].")
    factor = int(factor)
    if factor < 2:
        raise ValueError("factor must be an integer >= 2.")
    if relative_power_by_state is None:
        power = np.ones_like(errors)
    else:
        power = np.asarray(relative_power_by_state, dtype=float)
        try:
            power = np.broadcast_to(power, errors.shape)
        except ValueError as error:
            raise ValueError("relative_power_by_state must broadcast to the error array.") from error
        if np.any(power < 0.0) or not np.all(np.isfinite(power)):
            raise ValueError("relative powers must be finite and non-negative.")
        state_max = np.maximum(np.max(power, axis=1, keepdims=True), np.finfo(float).tiny)
        power = power / state_max
    refine_mask = np.any((errors > error_threshold) & (power >= power_floor), axis=0)
    maximum_error = float(np.max(errors)) if errors.size else 0.0
    if not np.any(refine_mask):
        return TopologyRefinement(
            topology=topology,
            refined_parent_mask=refine_mask,
            previous_beam_count=topology.beam_count,
            new_beam_count=topology.beam_count,
            maximum_normalized_error=maximum_error,
        )

    keep = ~refine_mask
    centres = [topology.centres_xy_m[keep]]
    areas = [topology.cell_area_m2[keep]]
    waists = [topology.waist_m[keep]]
    labels = [np.asarray(topology.labels, dtype=object)[keep]]
    fractions = (np.arange(factor, dtype=float) + 0.5) / factor - 0.5
    offset_x, offset_y = np.meshgrid(fractions, fractions, indexing="xy")
    unit_offsets = np.stack((offset_x.ravel(), offset_y.ravel()), axis=-1)
    for index in np.flatnonzero(refine_mask):
        step = np.sqrt(topology.cell_area_m2[index])
        child_centres = topology.centres_xy_m[index] + step * unit_offsets
        children = factor * factor
        centres.append(child_centres)
        areas.append(np.full(children, topology.cell_area_m2[index] / children))
        waists.append(np.full(children, topology.waist_m[index] / factor))
        labels.append(
            np.full(children, f"{topology.labels[index]}:refined", dtype=object)
        )
    centres_array = np.concatenate(centres)
    if max_beamlets is not None and centres_array.shape[0] > int(max_beamlets):
        raise MemoryError(
            f"Refinement would create {centres_array.shape[0]:,} beamlets, above "
            f"the explicit operational guard of {int(max_beamlets):,}."
        )
    refined = GaussianTopology(
        centres_xy_m=centres_array,
        waist_m=np.concatenate(waists),
        cell_area_m2=np.concatenate(areas),
        bounds_xyxy_m=topology.bounds_xyxy_m,
        centres_per_period=topology.centres_per_period,
        feature_period_m=topology.feature_period_m,
        labels=tuple(str(value) for value in np.concatenate(labels)),
        metadata={
            **topology.metadata,
            "parent_beam_count": topology.beam_count,
            "refined_parent_count": int(np.count_nonzero(refine_mask)),
            "topology_frozen": True,
        },
    )
    return TopologyRefinement(
        topology=refined,
        refined_parent_mask=refine_mask,
        previous_beam_count=topology.beam_count,
        new_beam_count=refined.beam_count,
        maximum_normalized_error=maximum_error,
    )


@dataclass(frozen=True)
class GaussianWorkEstimate:
    beam_count: int
    detector_pixels: int
    persistent_memory_bytes: int
    working_memory_bytes: int
    interaction_count: int
    estimated_runtime_s: float | None

    def human_summary(self) -> str:
        memory_gib = (self.persistent_memory_bytes + self.working_memory_bytes) / 2**30
        runtime = (
            "not estimated"
            if self.estimated_runtime_s is None
            else f"{self.estimated_runtime_s:.1f} s"
        )
        return (
            f"{self.beam_count:,} beamlets × {self.detector_pixels:,} pixels; "
            f"approximately {memory_gib:.2f} GiB resident+tile memory; runtime {runtime}."
        )


def estimate_gaussian_workload(
    topology: GaussianTopology,
    detector: DetectorGeometry,
    *,
    tile_shape_yx: tuple[int, int] = (32, 32),
    beam_chunk: int = 256,
    complex_bytes: int = 16,
    measured_interactions_per_second: float | None = None,
) -> GaussianWorkEstimate:
    """Estimate memory and work before launching a full detector render."""

    ty, tx = (int(tile_shape_yx[0]), int(tile_shape_yx[1]))
    if min(ty, tx, int(beam_chunk)) <= 0:
        raise ValueError("tile dimensions and beam_chunk must be positive.")
    n = topology.beam_count
    pixels = int(np.prod(detector.shape_yx))
    # centres, slopes, complex amplitude, path, Q, k, waist/cell bookkeeping
    persistent = n * (2 + 2 + 2 + 1 + 8 + 1 + 2) * 8
    working = ty * tx * int(beam_chunk) * complex_bytes * 2
    interactions = n * pixels
    runtime = None
    if measured_interactions_per_second is not None:
        rate = float(measured_interactions_per_second)
        if rate <= 0.0 or not np.isfinite(rate):
            raise ValueError("measured_interactions_per_second must be positive.")
        runtime = interactions / rate
    return GaussianWorkEstimate(
        beam_count=n,
        detector_pixels=pixels,
        persistent_memory_bytes=int(persistent),
        working_memory_bytes=int(working),
        interaction_count=int(interactions),
        estimated_runtime_s=runtime,
    )


def _split_action_channels(actions: object) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Accept the atlas' documented two-channel return and named variants."""

    if hasattr(actions, "pathlength_m"):
        path = getattr(actions, "pathlength_m")
        log_amp = getattr(actions, "log_field_amplitude")
        return jnp.asarray(path), jnp.asarray(log_amp)
    if isinstance(actions, (tuple, list)) and len(actions) == 2:
        return jnp.asarray(actions[0]), jnp.asarray(actions[1])
    arr = jnp.asarray(actions)
    if arr.shape[-1] != 2:
        raise ValueError(
            "Sample actions must be two channels [pathlength, log field amplitude]."
        )
    return arr[..., 0], arr[..., 1]


def gaussian_bundle_from_sample(
    view: object,
    topology: GaussianTopology,
    *,
    voltage: float = 200e3,
    z_m: float | None = None,
    incident_field: complex | np.ndarray = 1.0 + 0.0j,
) -> GaussianBeam:
    """Sample the exact transmission on a frozen Gaussian topology.

    This is the grid-multiplication/re-expansion path for a locally uniform
    incident field.  For an arbitrary incident grid field, use
    :func:`reexpand_transmitted_field`.
    """

    centres = jnp.asarray(topology.centres_xy_m)
    pathlength, log_amplitude = _split_action_channels(view.evaluate_actions(centres))
    incident = jnp.broadcast_to(jnp.asarray(incident_field), pathlength.shape)
    # A Gaussian exp(-r^2/w^2) integrates to pi*w^2.  Weight by represented
    # cell area so a uniform field is reproduced as the lattice is refined.
    normalisation = jnp.asarray(topology.cell_area_m2) / (
        jnp.pi * jnp.asarray(topology.waist_m) ** 2
    )
    coefficients = incident * jnp.exp(log_amplitude) * normalisation
    if z_m is None:
        z_m = float(getattr(view, "z", 0.0))
    beam = make_gaussian(
        x=centres[:, 0],
        y=centres[:, 1],
        dx=jnp.zeros(centres.shape[0]),
        dy=jnp.zeros(centres.shape[0]),
        z=jnp.full(centres.shape[0], float(z_m)),
        voltage=jnp.full(centres.shape[0], float(voltage)),
        amp=jnp.abs(coefficients),
        phase=jnp.angle(coefficients),
        waist_x=jnp.asarray(topology.waist_m),
        waist_y=jnp.asarray(topology.waist_m),
        rcurv_x=jnp.full(centres.shape[0], jnp.inf),
        rcurv_y=jnp.full(centres.shape[0], jnp.inf),
    )
    return beam.derive(pathlength=pathlength)


def _uniform_step(name: str, coords: np.ndarray) -> float:
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 1 or coords.size < 2 or not np.all(np.isfinite(coords)):
        raise ValueError(f"{name} must be a finite one-dimensional coordinate array.")
    diff = np.diff(coords)
    if np.any(diff <= 0.0) or not np.allclose(diff, diff[0], rtol=1e-8, atol=0.0):
        raise ValueError(f"{name} must be strictly increasing and uniformly sampled.")
    return float(diff[0])


def _gaussian_banded_matrix(
    axis_size: int,
    width_over_step: float,
    kernel_tolerance: float,
    ridge: float,
) -> tuple[np.ndarray, int]:
    bandwidth = min(
        axis_size - 1,
        int(np.ceil(width_over_step * np.sqrt(-np.log(kernel_tolerance)))),
    )
    kernel = np.exp(-((np.arange(bandwidth + 1) / width_over_step) ** 2))
    banded = np.zeros((2 * bandwidth + 1, axis_size), dtype=float)
    for offset in range(-bandwidth, bandwidth + 1):
        start = max(0, -offset)
        stop = min(axis_size, axis_size - offset)
        banded[bandwidth + offset, start:stop] = kernel[abs(offset)]
    banded[bandwidth] += ridge
    return banded, bandwidth


@lru_cache(maxsize=32)
def _gaussian_banded_cholesky_factors(
    axis_size: int,
    width_over_step: float,
    kernel_tolerance: float,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Factor one small Gaussian Toeplitz axis without a SciPy dependency.

    The returned arrays contain the lower Cholesky diagonal/lags and the
    corresponding upper lags.  They are host-side topology constants; the
    rectangular complex right-hand sides never enter this routine.
    """

    bandwidth = min(
        axis_size - 1,
        int(np.ceil(width_over_step * np.sqrt(-np.log(kernel_tolerance)))),
    )
    kernel = np.exp(-((np.arange(bandwidth + 1) / width_over_step) ** 2))
    lower = np.zeros((axis_size, bandwidth + 1), dtype=np.float64)
    diagonal = float(kernel[0] + ridge)
    for row in range(axis_size):
        first_column = max(0, row - bandwidth)
        diagonal_remainder = diagonal
        for column in range(first_column, row):
            diagonal_remainder -= lower[row, row - column] ** 2
        if not np.isfinite(diagonal_remainder) or diagonal_remainder <= 0.0:
            raise np.linalg.LinAlgError(
                "The Gaussian cardinal kernel is not numerically positive "
                "definite; increase ridge or reduce width_over_step_yx."
            )
        lower[row, 0] = np.sqrt(diagonal_remainder)
        last_row = min(axis_size, row + bandwidth + 1)
        for target_row in range(row + 1, last_row):
            first_shared = max(0, row - bandwidth, target_row - bandwidth)
            product_sum = 0.0
            for column in range(first_shared, row):
                product_sum += (
                    lower[target_row, target_row - column]
                    * lower[row, row - column]
                )
            lower[target_row, target_row - row] = (
                kernel[target_row - row] - product_sum
            ) / lower[row, 0]

    upper_lags = np.zeros((axis_size, bandwidth), dtype=np.float64)
    for row in range(axis_size):
        for lag in range(1, min(bandwidth, axis_size - row - 1) + 1):
            upper_lags[row, lag - 1] = lower[row + lag, lag]
    lower.setflags(write=False)
    upper_lags.setflags(write=False)
    return lower, upper_lags


@jax.jit
def _solve_gaussian_banded_cholesky_jax(
    rhs: jax.Array,
    lower_factor: jax.Array,
    upper_lags: jax.Array,
) -> jax.Array:
    """Solve a real SPD banded system against a large device-resident RHS."""

    rhs = jnp.asarray(rhs)
    lower_factor = jnp.asarray(lower_factor)
    upper_lags = jnp.asarray(upper_lags)
    bandwidth = lower_factor.shape[1] - 1
    if bandwidth == 0:
        diagonal = lower_factor[:, :1]
        return rhs / (diagonal * diagonal)
    carry_shape = (bandwidth, rhs.shape[1])

    def forward(carry, inputs):
        rhs_row, factor_row = inputs
        value = (
            rhs_row
            - jnp.sum(factor_row[1:, None] * carry, axis=0)
        ) / factor_row[0]
        next_carry = jnp.concatenate((value[None, :], carry[:-1]), axis=0)
        return next_carry, value

    initial = jnp.zeros(carry_shape, dtype=rhs.dtype)
    _, intermediate = jax.lax.scan(forward, initial, (rhs, lower_factor))

    def backward(carry, inputs):
        rhs_row, lags, diagonal = inputs
        value = (rhs_row - jnp.sum(lags[:, None] * carry, axis=0)) / diagonal
        next_carry = jnp.concatenate((value[None, :], carry[:-1]), axis=0)
        return next_carry, value

    _, reversed_solution = jax.lax.scan(
        backward,
        initial,
        (intermediate[::-1], upper_lags[::-1], lower_factor[::-1, 0]),
    )
    return reversed_solution[::-1]


def gaussian_cardinal_coefficients(
    field_yx: np.ndarray | jax.Array,
    *,
    width_over_step_yx: tuple[float, float] = (2.0, 2.0),
    kernel_tolerance: float = 1e-13,
    ridge: float = 1e-10,
    method: str = "scipy",
) -> np.ndarray | jax.Array:
    """Fit a rectangular equal-width Gaussian cardinal lattice.

    The solve is separable and banded, so it remains practical for specimen
    grids whose flattened expansions contain hundreds of thousands of terms.
    The historical ``scipy`` method and NumPy return type remain the default.
    ``jax`` keeps both large complex right-hand-side solves on the active JAX
    device.  ``gpu_kernel_required`` additionally fails unless the probed JAX
    GPU backend is available.
    """

    if method not in {"scipy", "jax", "gpu_kernel_required"}:
        raise ValueError(
            "method must be 'scipy', 'jax', or 'gpu_kernel_required'."
        )
    if method == "gpu_kernel_required" and not gpu_kernel_available():
        raise RuntimeError(
            "Gaussian cardinal method 'gpu_kernel_required' requires a "
            "working JAX GPU backend; host/SciPy fallback is disabled."
        )
    values = np.asarray(field_yx) if method == "scipy" else jnp.asarray(field_yx)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("field_yx must be a two-dimensional array at least 2x2.")
    if method == "scipy" and not np.all(np.isfinite(values)):
        raise ValueError("field_yx must be finite.")
    wy, wx = _pair_float("width_over_step_yx", width_over_step_yx)
    if min(wy, wx) <= 0.0:
        raise ValueError("width_over_step_yx must be positive.")
    if not 0.0 < kernel_tolerance < 1.0:
        raise ValueError("kernel_tolerance must lie between zero and one.")
    if ridge < 0.0 or not np.isfinite(ridge):
        raise ValueError("ridge must be finite and non-negative.")

    if method == "scipy":
        by, ly = _gaussian_banded_matrix(values.shape[0], wy, kernel_tolerance, ridge)
        bx, lx = _gaussian_banded_matrix(values.shape[1], wx, kernel_tolerance, ridge)
        solve_dtype = np.result_type(values.dtype, np.float64)
        along_y = linalg.solve_banded(
            (ly, ly), by, values.astype(solve_dtype, copy=False), check_finite=False
        )
        return linalg.solve_banded(
            (lx, lx), bx, along_y.T, check_finite=False
        ).T

    lower_y, upper_y = _gaussian_banded_cholesky_factors(
        values.shape[0], wy, kernel_tolerance, ridge
    )
    lower_x, upper_x = _gaussian_banded_cholesky_factors(
        values.shape[1], wx, kernel_tolerance, ridge
    )
    solve_dtype = jnp.result_type(values.dtype, jnp.float64)
    device_values = values.astype(solve_dtype)
    along_y = _solve_gaussian_banded_cholesky_jax(
        device_values,
        jnp.asarray(lower_y),
        jnp.asarray(upper_y),
    )
    return _solve_gaussian_banded_cholesky_jax(
        along_y.T,
        jnp.asarray(lower_x),
        jnp.asarray(upper_x),
    ).T


def reexpand_complex_field(
    field_yx: np.ndarray | jax.Array,
    x_coords_m: np.ndarray,
    y_coords_m: np.ndarray,
    *,
    voltage: float = 200e3,
    z_m: float = 0.0,
    width_over_step_yx: tuple[float, float] = (2.0, 2.0),
    kernel_tolerance: float = 1e-13,
    ridge: float = 1e-10,
    coefficient_method: str = "scipy",
) -> GaussianBeam:
    """Re-expand a sampled complex field into a rectangular Gaussian lattice."""

    x = np.asarray(x_coords_m, dtype=float)
    y = np.asarray(y_coords_m, dtype=float)
    dx = _uniform_step("x_coords_m", x)
    dy = _uniform_step("y_coords_m", y)
    field_shape = (
        np.asarray(field_yx).shape
        if coefficient_method == "scipy"
        else tuple(field_yx.shape)
    )
    if field_shape != (y.size, x.size):
        raise ValueError("field_yx shape must be (len(y_coords_m), len(x_coords_m)).")
    coeff = gaussian_cardinal_coefficients(
        field_yx,
        width_over_step_yx=width_over_step_yx,
        kernel_tolerance=kernel_tolerance,
        ridge=ridge,
        method=coefficient_method,
    )
    use_device = coefficient_method != "scipy"
    array_module = jnp if use_device else np
    xx, yy = array_module.meshgrid(
        array_module.asarray(x),
        array_module.asarray(y),
        indexing="xy",
    )
    values = coeff.ravel()
    return make_gaussian(
        x=xx.ravel(),
        y=yy.ravel(),
        dx=array_module.zeros(values.size),
        dy=array_module.zeros(values.size),
        z=array_module.full(values.size, float(z_m)),
        voltage=array_module.full(values.size, float(voltage)),
        amp=array_module.abs(values),
        phase=array_module.angle(values),
        waist_x=array_module.full(values.size, width_over_step_yx[1] * dx),
        waist_y=array_module.full(values.size, width_over_step_yx[0] * dy),
        rcurv_x=array_module.full(values.size, array_module.inf),
        rcurv_y=array_module.full(values.size, array_module.inf),
    )


def reexpand_transmitted_field(
    incident_field_yx: np.ndarray | jax.Array,
    view: object,
    x_coords_m: np.ndarray,
    y_coords_m: np.ndarray,
    *,
    voltage: float = 200e3,
    z_m: float | None = None,
    width_over_step_yx: tuple[float, float] = (2.0, 2.0),
    coefficient_method: str = "scipy",
) -> tuple[GaussianBeam, np.ndarray | jax.Array]:
    """Multiply an incident field by the cubic sample, then re-expand it."""

    x = np.asarray(x_coords_m, dtype=float)
    y = np.asarray(y_coords_m, dtype=float)
    use_device = coefficient_method != "scipy"
    incident = (
        jnp.asarray(incident_field_yx)
        if use_device
        else np.asarray(incident_field_yx)
    )
    if incident.shape != (y.size, x.size):
        raise ValueError("incident_field_yx shape must match the coordinate grid.")
    array_module = jnp if use_device else np
    xx, yy = array_module.meshgrid(
        array_module.asarray(x),
        array_module.asarray(y),
        indexing="xy",
    )
    points = jnp.stack((jnp.ravel(xx), jnp.ravel(yy)), axis=-1)
    path, log_amp = _split_action_channels(view.evaluate_actions(points))
    if use_device:
        wavelength = energy2wavelength(jnp.asarray(voltage))
        transmission = jnp.exp(log_amp + 2j * jnp.pi * path / wavelength)
    else:
        wavelength = float(np.asarray(energy2wavelength(float(voltage))))
        transmission = np.exp(
            np.asarray(log_amp) + 2j * np.pi * np.asarray(path) / wavelength
        )
    transmitted = incident * transmission.reshape(incident.shape)
    if z_m is None:
        z_m = float(getattr(view, "z", 0.0))
    beam = reexpand_complex_field(
        transmitted,
        x,
        y,
        voltage=voltage,
        z_m=z_m,
        width_over_step_yx=width_over_step_yx,
        coefficient_method=coefficient_method,
    )
    return beam, transmitted


def transfer_blocks(matrix: np.ndarray) -> tuple[jnp.ndarray, ...]:
    """Extract A, B, C, D blocks in ``(x, y)`` coordinate order."""

    matrix_array = jnp.asarray(matrix)
    if matrix_array.shape != (4, 4):
        raise ValueError("matrix must have shape (4, 4).")
    return (
        matrix_array[np.ix_(_POSITION, _POSITION)],
        matrix_array[np.ix_(_POSITION, _ANGLE)],
        matrix_array[np.ix_(_ANGLE, _POSITION)],
        matrix_array[np.ix_(_ANGLE, _ANGLE)],
    )


def apply_transfer_matrix_to_gaussians(
    beam: GaussianBeam,
    matrix: np.ndarray,
    *,
    z_out_m: float,
    axial_path_m: float | None = None,
) -> GaussianBeam:
    """Apply a coupled first-order map to a Gaussian bundle.

    The quadratic endpoint action ``(r1.p1-r0.p0)/2`` accounts for drift and
    lens phase.  ``axial_path_m`` adds the common on-axis path and defaults to
    ``z_out_m - beam.z``.
    """

    A, B, C, D = (jnp.asarray(block) for block in transfer_blocks(matrix))
    r0 = beam.r_xy
    p0 = beam.d_xy
    Q0 = jnp.asarray(beam.Q_inv)
    r1 = jnp.einsum("ij,...j->...i", A, r0) + jnp.einsum("ij,...j->...i", B, p0)
    p1 = jnp.einsum("ij,...j->...i", C, r0) + jnp.einsum("ij,...j->...i", D, p0)
    denominator = A + jnp.einsum("ij,...jk->...ik", B, Q0)
    numerator = C + jnp.einsum("ij,...jk->...ik", D, Q0)
    Q1 = jnp.swapaxes(
        jnp.linalg.solve(jnp.swapaxes(denominator, -1, -2), jnp.swapaxes(numerator, -1, -2)),
        -1,
        -2,
    )
    determinant_root = jnp.sqrt(jnp.linalg.det(denominator).astype(jnp.complex128))
    amplitude = beam.amplitude / determinant_root
    endpoint_action = 0.5 * (
        jnp.sum(r1 * p1, axis=-1) - jnp.sum(r0 * p0, axis=-1)
    )
    if axial_path_m is None:
        axial_path = jnp.asarray(z_out_m) - jnp.asarray(beam.z)
    else:
        axial_path = jnp.asarray(axial_path_m)
    return beam.derive(
        x=r1[..., 0],
        y=r1[..., 1],
        dx=p1[..., 0],
        dy=p1[..., 1],
        z=jnp.broadcast_to(jnp.asarray(z_out_m), jnp.shape(beam.z)),
        amplitude=amplitude,
        pathlength=beam.pathlength + axial_path + endpoint_action,
        Q_inv=Q1,
    )


@dataclass(frozen=True)
class GaussianSliceSafetyReport:
    """Static hard-edge subdivision audit over a frozen DAC-state envelope.

    ``q`` is both the radial oscillator phase advance and the magnitude of the
    Larmor rotation accumulated through one ideal hard-edge body.  Keeping the
    per-slice value well inside the first quadrant reduces the chance that the
    principal determinant square root in the lifted Gaussian map changes
    metaplectic sheet between adjacent physical steps.  The default audit limit
    is :math:`\\pi/4`; convergence should still be checked over progressively
    doubled counts until *two successive* complex-field comparisons agree,
    because incident Gaussian curvature also contributes to the determinant
    path and convergence need not be monotonic at a single doubled count.
    """

    lens_names: tuple[str, ...]
    integrated_strengths_by_state_lens: np.ndarray
    maximum_phase_advance_per_slice_rad: float
    recommended_slices_per_lens: int
    requested_slices_per_lens: int | None = None

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.lens_names)
        strengths = np.asarray(self.integrated_strengths_by_state_lens, dtype=float)
        limit = float(self.maximum_phase_advance_per_slice_rad)
        recommended = int(self.recommended_slices_per_lens)
        requested = self.requested_slices_per_lens
        if not names or len(set(names)) != len(names):
            raise ValueError("lens_names must be non-empty and unique.")
        if (
            strengths.ndim != 2
            or strengths.shape[0] < 1
            or strengths.shape[1] != len(names)
            or not np.all(np.isfinite(strengths))
        ):
            raise ValueError(
                "integrated_strengths_by_state_lens must be a finite "
                "(n_states, n_lenses) array."
            )
        if not np.isfinite(limit) or not 0.0 < limit < 0.5 * np.pi:
            raise ValueError(
                "maximum_phase_advance_per_slice_rad must lie strictly "
                "between zero and pi/2."
            )
        if recommended < 1:
            raise ValueError("recommended_slices_per_lens must be positive.")
        maximum = float(np.max(np.abs(strengths)))
        if not maximum / recommended < limit:
            raise ValueError(
                "recommended_slices_per_lens does not satisfy the declared "
                "strict phase-advance limit."
            )
        if requested is not None:
            if isinstance(requested, (bool, np.bool_)) or int(requested) != requested:
                raise ValueError("requested_slices_per_lens must be an integer or None.")
            requested = int(requested)
            if requested < 1:
                raise ValueError("requested_slices_per_lens must be positive.")
        strengths = strengths.copy()
        strengths.setflags(write=False)
        object.__setattr__(self, "lens_names", names)
        object.__setattr__(self, "integrated_strengths_by_state_lens", strengths)
        object.__setattr__(self, "maximum_phase_advance_per_slice_rad", limit)
        object.__setattr__(self, "recommended_slices_per_lens", recommended)
        object.__setattr__(self, "requested_slices_per_lens", requested)

    @property
    def state_count(self) -> int:
        return int(self.integrated_strengths_by_state_lens.shape[0])

    @property
    def maximum_abs_q_by_lens(self) -> np.ndarray:
        values = np.max(np.abs(self.integrated_strengths_by_state_lens), axis=0)
        values.setflags(write=False)
        return values

    @property
    def maximum_abs_q(self) -> float:
        return float(np.max(np.abs(self.integrated_strengths_by_state_lens)))

    @property
    def worst_state_index(self) -> int:
        flat_index = int(np.argmax(np.abs(self.integrated_strengths_by_state_lens)))
        indices = np.unravel_index(
            flat_index,
            self.integrated_strengths_by_state_lens.shape,
        )
        return int(indices[0])

    @property
    def worst_lens_name(self) -> str:
        flat_index = int(np.argmax(np.abs(self.integrated_strengths_by_state_lens)))
        lens_index = int(
            np.unravel_index(flat_index, self.integrated_strengths_by_state_lens.shape)[1]
        )
        return self.lens_names[lens_index]

    @property
    def requested_maximum_phase_advance_rad(self) -> float | None:
        if self.requested_slices_per_lens is None:
            return None
        return self.maximum_abs_q / self.requested_slices_per_lens

    @property
    def requested_is_safe(self) -> bool | None:
        """Whether the requested count passes the phase-advance precondition.

        This does not replace complex-field refinement because it cannot see
        the incident beam's curvature or the accumulated determinant path.
        """

        phase = self.requested_maximum_phase_advance_rad
        if phase is None:
            return None
        return bool(phase < self.maximum_phase_advance_per_slice_rad)

    def human_summary(self) -> str:
        requested = "not supplied"
        if self.requested_slices_per_lens is not None:
            status = "safe" if self.requested_is_safe else "unsafe"
            requested = (
                f"{self.requested_slices_per_lens} ({status}; "
                f"{self.requested_maximum_phase_advance_rad:.6g} rad/slice)"
            )
        return (
            f"max |q|={self.maximum_abs_q:.6g} rad at state "
            f"{self.worst_state_index}, lens {self.worst_lens_name}; requested "
            f"slices={requested}; recommended uniform slices="
            f"{self.recommended_slices_per_lens} for a strict <"
            f"{self.maximum_phase_advance_per_slice_rad:.6g} rad/slice limit"
        )


def assess_gaussian_slice_safety(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    dac_values: np.ndarray,
    *,
    slices_per_lens: int | None = None,
    maximum_phase_advance_per_slice_rad: float = np.pi / 4.0,
    minimum_slices_per_lens: int = 1,
) -> GaussianSliceSafetyReport:
    """Recommend one static lens-slice count for all supplied optical states.

    The returned count is based on the largest absolute integrated hard-edge
    strength over every lens and every row in ``dac_values``.  Supplying the
    entire nominal, wobble, D-optimal, held-out, and approach-start envelope is
    therefore important: choosing a count independently per frame would change
    numerical topology inside the dataset.

    This is a conservative phase-advance audit, not a substitute for the final
    metaplectic convergence check.  Starting at the recommended count, compare
    progressively doubled complex fields and require two successive stable
    transitions before admitting them to an inversion.
    """

    values = np.asarray(parameters, dtype=float)
    if values.shape != (column.n_parameters,) or not np.all(np.isfinite(values)):
        raise ValueError(f"parameters must have shape ({column.n_parameters},).")
    controls = np.asarray(dac_values, dtype=float)
    if controls.ndim == 1:
        controls = controls[None, :]
    if (
        controls.ndim != 2
        or controls.shape[0] < 1
        or controls.shape[1] != len(column.channel_names)
        or not np.all(np.isfinite(controls))
    ):
        raise ValueError(
            "dac_values must be a finite (n_states, n_channels) array or one "
            "n_channels vector."
        )
    limit = float(maximum_phase_advance_per_slice_rad)
    if not np.isfinite(limit) or not 0.0 < limit < 0.5 * np.pi:
        raise ValueError(
            "maximum_phase_advance_per_slice_rad must lie strictly between "
            "zero and pi/2."
        )
    if (
        isinstance(minimum_slices_per_lens, (bool, np.bool_))
        or int(minimum_slices_per_lens) != minimum_slices_per_lens
        or int(minimum_slices_per_lens) < 1
    ):
        raise ValueError("minimum_slices_per_lens must be a positive integer.")
    if slices_per_lens is not None and (
        isinstance(slices_per_lens, (bool, np.bool_))
        or int(slices_per_lens) != slices_per_lens
        or int(slices_per_lens) < 1
    ):
        raise ValueError("slices_per_lens must be a positive integer or None.")

    strengths: list[np.ndarray] = []
    cursor = 2
    for lens in column.lenses:
        q_bias = float(values[cursor + 2])
        cursor += 3
        if lens.channel_index is None:
            q = np.full(controls.shape[0], q_bias, dtype=float)
        else:
            q_slope = float(values[cursor])
            cursor += 1
            displacement = controls[:, lens.channel_index] - lens.reference_dac
            q = q_bias + q_slope * displacement
        strengths.append(q)
    strength_array = np.stack(strengths, axis=1)
    maximum = float(np.max(np.abs(strength_array)))
    recommended = max(
        int(minimum_slices_per_lens),
        int(np.floor(maximum / limit)) + 1,
    )
    return GaussianSliceSafetyReport(
        lens_names=tuple(lens.name for lens in column.lenses),
        integrated_strengths_by_state_lens=strength_array,
        maximum_phase_advance_per_slice_rad=limit,
        recommended_slices_per_lens=recommended,
        requested_slices_per_lens=(
            None if slices_per_lens is None else int(slices_per_lens)
        ),
    )


def propagate_gaussians_through_column(
    beam: GaussianBeam,
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    dac_values: np.ndarray,
    *,
    segment: str = "full",
    slices_per_lens: int = 24,
) -> GaussianBeam:
    """Propagate a coherent bundle through a lifted physical column sequence.

    A single endpoint symplectic matrix does not contain the metaplectic
    (Maslov/Gouy) branch.  This routine retains it by applying each physical
    drift and a fixed number of short hard-edge slices in axial order.  Slice
    topology is static, so the result remains JAX differentiable.
    """

    if segment not in {"full", "source_to_sample", "sample_to_detector"}:
        raise ValueError(
            "segment must be 'full', 'source_to_sample', or 'sample_to_detector'."
        )
    if int(slices_per_lens) != slices_per_lens or slices_per_lens < 1:
        raise ValueError("slices_per_lens must be a positive integer.")
    values = jnp.asarray(parameters)
    controls = jnp.asarray(dac_values)
    if values.shape != (column.n_parameters,):
        raise ValueError(f"parameters must have shape ({column.n_parameters},).")
    if controls.shape != (len(column.channel_names),):
        raise ValueError(
            f"dac_values must have shape ({len(column.channel_names)},)."
        )

    if segment in {"full", "source_to_sample"}:
        axial_cursor = values[0]
    else:
        axial_cursor = jnp.asarray(column.sample_z_m, dtype=values.dtype)
    propagated = beam
    free_space = FreeSpacePropagator()
    decoded = column._decoded_lenses(values)
    slice_count = int(slices_per_lens)
    for lens, centre, length, q_bias, q_slope in decoded:
        include = (
            segment == "full"
            or (segment == "source_to_sample" and lens.exit_z <= column.sample_z_m)
            or (segment == "sample_to_detector" and lens.entrance_z >= column.sample_z_m)
        )
        if not include:
            continue
        entrance = centre - 0.5 * length
        propagated = free_space(propagated, entrance - axial_cursor)
        if lens.channel_index is None:
            displacement = jnp.asarray(0.0, dtype=values.dtype)
        else:
            displacement = controls[lens.channel_index] - lens.reference_dac
        q = q_bias + q_slope * displacement
        slice_length = length / slice_count
        slice_q = q / slice_count
        slice_matrix = hard_edge_solenoid_matrix(slice_q, slice_length)

        # Keep high-count metaplectic refinement practical for million-term
        # bundles.  A compiled loop has the same sequential square-root sheet
        # tracking as the historical Python loop without dispatching hundreds
        # of separate device operations for every hard-edge body.
        def apply_slice(slice_index, current):
            z_out = entrance + (slice_index + 1) * slice_length
            return apply_transfer_matrix_to_gaussians(
                current,
                slice_matrix,
                z_out_m=z_out,
                axial_path_m=slice_length,
            )

        propagated = jax.lax.fori_loop(
            0,
            slice_count,
            apply_slice,
            propagated,
        )
        axial_cursor = centre + 0.5 * length
    axial_end = (
        jnp.asarray(column.sample_z_m, dtype=values.dtype)
        if segment == "source_to_sample"
        else values[1]
    )
    return free_space(propagated, axial_end - axial_cursor)


@dataclass(frozen=True)
class _ExplicitGrid:
    coords: jnp.ndarray
    shape: tuple[int, int]


@dataclass(frozen=True)
class SampleInteractionResult:
    beam: GaussianBeam
    route: str
    validation: object | None
    incident_field_yx: np.ndarray | jax.Array | None = None
    transmitted_field_yx: np.ndarray | jax.Array | None = None


def interact_gaussian_with_sample(
    view: object,
    incident_beam: GaussianBeam,
    *,
    mode: str = "auto",
    x_coords_m: np.ndarray | None = None,
    y_coords_m: np.ndarray | None = None,
    phase_tol: float = 0.1,
    amplitude_tol: float = 0.02,
    support_radius: float = 2.0,
    evaluation_method: str = "auto",
    evaluation_batch_size: int = 128,
    width_over_step_yx: tuple[float, float] = (2.0, 2.0),
) -> SampleInteractionResult:
    """Apply local cubic action or exact grid multiplication and re-expansion.

    ``auto`` makes the routing decision on the host using the frozen bundle.
    If re-expansion is selected, explicit sample-plane coordinates are required;
    this prevents an optimizer from silently inventing or switching grids.
    """

    if mode not in {"auto", "local", "reexpand"}:
        raise ValueError("mode must be 'auto', 'local', or 'reexpand'.")
    validation = None
    route = mode
    if mode == "auto":
        from .multiscale_sample import validate_sample_level

        validation = validate_sample_level(
            view,
            incident_beam,
            phase_tol=phase_tol,
            amplitude_tol=amplitude_tol,
            support_radius=support_radius,
        )
        route = "local" if validation.use_local_action else "reexpand"
    if route == "local":
        return SampleInteractionResult(
            beam=view(incident_beam),
            route="local_action",
            validation=validation,
        )

    if x_coords_m is None or y_coords_m is None:
        raise ValueError(
            "x_coords_m and y_coords_m are required when sample-plane "
            "re-expansion is selected."
        )
    x = np.asarray(x_coords_m, dtype=float)
    y = np.asarray(y_coords_m, dtype=float)
    _uniform_step("x_coords_m", x)
    _uniform_step("y_coords_m", y)
    xx, yy = jnp.meshgrid(jnp.asarray(x), jnp.asarray(y), indexing="xy")
    grid = _ExplicitGrid(
        coords=jnp.stack((xx.ravel(), yy.ravel()), axis=-1),
        shape=(y.size, x.size),
    )
    evaluated_incident_field = evaluate_gaussians(
        incident_beam,
        grid,  # type: ignore[arg-type]
        method=evaluation_method,
        batch_size=evaluation_batch_size,
    )
    device_reexpansion = evaluation_method == "gpu_kernel_required"
    incident_field = (
        evaluated_incident_field
        if device_reexpansion
        else np.asarray(evaluated_incident_field)
    )
    voltage = float(jax.device_get(jnp.ravel(incident_beam.voltage)[0]))
    z_m = float(jax.device_get(jnp.ravel(incident_beam.z)[0]))
    transmitted_beam, transmitted = reexpand_transmitted_field(
        incident_field,
        view,
        x,
        y,
        voltage=voltage,
        z_m=z_m,
        width_over_step_yx=width_over_step_yx,
        coefficient_method=(
            "gpu_kernel_required" if device_reexpansion else "scipy"
        ),
    )
    return SampleInteractionResult(
        beam=transmitted_beam,
        route="sample_plane_reexpansion",
        validation=validation,
        incident_field_yx=incident_field,
        transmitted_field_yx=transmitted,
    )


@dataclass(frozen=True)
class GaussianRenderResult:
    field_yx: np.ndarray
    intensity_yx: np.ndarray
    elapsed_s: float
    estimate: GaussianWorkEstimate | None = None


@dataclass(frozen=True)
class ConjugateGridReferenceResult:
    """Independent detector intensity in the specimen-conjugate limit.

    The reference samples the cubic specimen directly at inverse-mapped
    detector coordinates and never constructs or evaluates a Gaussian
    expansion.  ``maximum_b_displacement_px`` is a conservative upper bound on
    the detector displacement omitted by taking the ``B -> 0`` imaging limit.
    """

    intensity_yx: np.ndarray
    magnification_block_xy: np.ndarray
    angle_to_position_block_xy_m: np.ndarray
    angular_bandlimit_rad: float
    maximum_b_displacement_px: float


def conjugate_gaussian_grid_reference(
    sample_view: object,
    transfer_matrix: np.ndarray,
    detector: DetectorGeometry,
    *,
    incident_centre_xy_m: Sequence[float],
    incident_waist_xy_m: Sequence[float],
    wavelength_m: float,
    sample_bandlimit_spacing_m: float,
    maximum_b_displacement_px: float = 1e-3,
) -> ConjugateGridReferenceResult:
    """Evaluate an independent grid-wave reference for a conjugate image.

    A Collins kernel becomes a numerically singular representation as the
    sample-to-detector ``B`` block approaches zero.  In that regime the exact
    linear-canonical limit is an inverse mapping through ``A`` with the
    intensity Jacobian ``1 / |det(A)|``.  This routine applies that limit to an
    analytic unit-power elliptical Gaussian multiplied by the cubic specimen,
    then applies the detector MTF.  It is independent of Gaussian cardinal
    re-expansion and detector beamlet summation.

    The approximation is admitted only when the worst displacement generated
    by a two-dimensional Nyquist angular bandlimit is below the requested
    detector-pixel tolerance.  Deliberately defocused states should instead use
    :func:`collins_propagate_2d` (or a scaled FFT/CZT implementation).
    """

    matrix = np.asarray(transfer_matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("transfer_matrix must be a finite 4x4 array.")
    centre = np.asarray(_pair_float("incident_centre_xy_m", incident_centre_xy_m))
    waist = np.asarray(_pair_float("incident_waist_xy_m", incident_waist_xy_m))
    if np.any(waist <= 0.0):
        raise ValueError("incident_waist_xy_m must be positive.")
    wavelength = float(wavelength_m)
    spacing = float(sample_bandlimit_spacing_m)
    displacement_limit = float(maximum_b_displacement_px)
    if not np.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError("wavelength_m must be finite and positive.")
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("sample_bandlimit_spacing_m must be finite and positive.")
    if not np.isfinite(displacement_limit) or displacement_limit <= 0.0:
        raise ValueError("maximum_b_displacement_px must be finite and positive.")

    A, B, _C, _D = (np.asarray(block, dtype=float) for block in transfer_blocks(matrix))
    determinant = float(np.linalg.det(A))
    if abs(determinant) <= np.finfo(float).eps * float(np.linalg.norm(A)) ** 2:
        raise ValueError("The conjugate grid reference requires an invertible A block.")

    angular_bandlimit = wavelength / (2.0 * spacing)
    pixel_xy = np.asarray(detector.pixel_size_yx_m[::-1], dtype=float)
    optical_pixel_map = np.linalg.inv(detector.affine_xy) @ np.diag(pixel_xy)
    smallest_optical_pixel = float(
        np.min(np.linalg.svd(optical_pixel_map, compute_uv=False))
    )
    # Each angular coordinate is Nyquist limited.  sqrt(2) bounds the norm at
    # a corner of the square two-dimensional passband.
    b_displacement = (
        float(np.linalg.norm(B, ord=2))
        * np.sqrt(2.0)
        * angular_bandlimit
        / smallest_optical_pixel
    )
    if b_displacement > displacement_limit:
        raise RuntimeError(
            "The sample-to-detector B block is too large for the conjugate "
            f"grid limit: {b_displacement:.6g} px exceeds "
            f"{displacement_limit:.6g} px. Use Collins/CZT propagation."
        )

    detector_column_xy = detector.optical_coords_xy_m
    sample_xy = np.einsum(
        "ij,...j->...i", np.linalg.inv(A), detector_column_xy
    )
    actions = np.asarray(
        sample_view.evaluate_actions(jnp.asarray(sample_xy.reshape((-1, 2))))
    ).reshape((*detector.shape_yx, 2))
    if not np.all(np.isfinite(actions)):
        raise ValueError("sample_view returned non-finite action values.")
    offset = sample_xy - centre
    incident_intensity = 2.0 / (np.pi * waist[0] * waist[1]) * np.exp(
        -2.0
        * (
            (offset[..., 0] / waist[0]) ** 2
            + (offset[..., 1] / waist[1]) ** 2
        )
    )
    intensity = incident_intensity * np.exp(2.0 * actions[..., 1]) / abs(determinant)
    if detector.mtf_sigma_px > 0.0:
        intensity = ndimage.gaussian_filter(
            intensity, detector.mtf_sigma_px, mode="nearest"
        )
    if not np.all(np.isfinite(intensity)) or np.any(intensity < 0.0):
        raise RuntimeError("The conjugate grid reference produced invalid intensity.")
    return ConjugateGridReferenceResult(
        intensity_yx=intensity,
        magnification_block_xy=A,
        angle_to_position_block_xy_m=B,
        angular_bandlimit_rad=float(angular_bandlimit),
        maximum_b_displacement_px=b_displacement,
    )


def render_gaussian_detector_plane(
    beam_at_detector: GaussianBeam,
    detector: DetectorGeometry,
    *,
    tile_rows: int = 32,
    beam_batch_size: int = 128,
    method: str = "auto",
    normalise: bool = False,
    workload: GaussianWorkEstimate | None = None,
) -> GaussianRenderResult:
    """Evaluate an already-propagated coherent bundle in detector-row tiles."""

    if tile_rows <= 0:
        raise ValueError("tile_rows must be positive.")
    optical_coords = detector.optical_coords_xy_m
    field = np.empty(detector.shape_yx, dtype=np.complex128)
    started = perf_counter()
    for start in range(0, detector.shape_yx[0], tile_rows):
        stop = min(detector.shape_yx[0], start + tile_rows)
        coords = jnp.asarray(optical_coords[start:stop].reshape((-1, 2)))
        grid = _ExplicitGrid(
            coords=coords,
            shape=(stop - start, detector.shape_yx[1]),
        )
        field[start:stop] = np.asarray(
            evaluate_gaussians(
                beam_at_detector,
                grid,  # type: ignore[arg-type]
                method=method,
                batch_size=beam_batch_size,
            )
        )
    intensity = np.abs(field) ** 2
    if detector.mtf_sigma_px > 0.0:
        intensity = ndimage.gaussian_filter(
            intensity, detector.mtf_sigma_px, mode="nearest"
        )
    if normalise:
        scale = float(np.max(intensity))
        if scale > 0.0:
            intensity = intensity / scale
    return GaussianRenderResult(
        field_yx=field,
        intensity_yx=intensity,
        elapsed_s=perf_counter() - started,
        estimate=workload,
    )


def render_gaussian_detector(
    beam_at_sample: GaussianBeam,
    transfer_matrix: np.ndarray,
    detector: DetectorGeometry,
    *,
    tile_rows: int = 32,
    beam_batch_size: int = 128,
    method: str = "auto",
    normalise: bool = False,
    workload: GaussianWorkEstimate | None = None,
) -> GaussianRenderResult:
    """Propagate and evaluate a Gaussian expansion in detector-row tiles."""

    propagated = apply_transfer_matrix_to_gaussians(
        beam_at_sample, transfer_matrix, z_out_m=detector.z_m
    )
    return render_gaussian_detector_plane(
        propagated,
        detector,
        tile_rows=tile_rows,
        beam_batch_size=beam_batch_size,
        method=method,
        normalise=normalise,
        workload=workload,
    )


def simpson_weights(coords: np.ndarray) -> np.ndarray:
    """Return integration weights reproducing SciPy's Simpson convention."""

    coords = np.asarray(coords, dtype=float)
    _uniform_step("coords", coords)
    return integrate.simpson(np.eye(coords.size), x=coords, axis=1)


def _collins_kernel(
    coords_in: np.ndarray,
    coords_out: np.ndarray,
    abcd: Sequence[float],
    wavelength_m: float,
) -> np.ndarray:
    A, B, _C, D = (float(v) for v in abcd)
    if abs(B) < 1e-15:
        raise ValueError(
            "The direct Collins validation kernel requires non-zero B; "
            "use a deliberately defocused validation state."
        )
    k = 2.0 * np.pi / wavelength_m
    xin = np.asarray(coords_in, dtype=float)[None, :]
    xout = np.asarray(coords_out, dtype=float)[:, None]
    phase = (A * xin**2 - 2.0 * xin * xout + D * xout**2) * (0.5 * k / B)
    return np.exp(1j * phase) / np.sqrt(1j * wavelength_m * B + 0j)


def collins_propagate_2d(
    field_yx: np.ndarray,
    x_in_m: np.ndarray,
    y_in_m: np.ndarray,
    x_out_m: np.ndarray,
    y_out_m: np.ndarray,
    *,
    abcd_x: Sequence[float],
    abcd_y: Sequence[float],
    wavelength_m: float,
) -> np.ndarray:
    """Independent separable Collins integral with Simpson quadrature.

    This intentionally does not call the Gaussian forward model.  It is a
    validation reference for states with negligible x/y coupling.  Rotating
    solenoidal states should first be transformed into their co-rotating
    Larmor frame.
    """

    field = np.asarray(field_yx, dtype=np.complex128)
    x_in = np.asarray(x_in_m, dtype=float)
    y_in = np.asarray(y_in_m, dtype=float)
    x_out = np.asarray(x_out_m, dtype=float)
    y_out = np.asarray(y_out_m, dtype=float)
    if field.shape != (y_in.size, x_in.size):
        raise ValueError("field_yx must have shape (len(y_in_m), len(x_in_m)).")
    wavelength = float(wavelength_m)
    if wavelength <= 0.0 or not np.isfinite(wavelength):
        raise ValueError("wavelength_m must be finite and positive.")
    wx = simpson_weights(x_in)
    wy = simpson_weights(y_in)
    Kx = _collins_kernel(x_in, x_out, abcd_x, wavelength) * wx[None, :]
    Ky = _collins_kernel(y_in, y_out, abcd_y, wavelength) * wy[None, :]
    return Ky @ field @ Kx.T


@dataclass(frozen=True)
class ImageObservables:
    total_intensity: float
    centroid_xy_px: tuple[float, float]
    covariance_xy_px2: np.ndarray
    principal_sigma_px: tuple[float, float]
    principal_rotation_rad: float
    fourier_vectors_yx_px: np.ndarray


@dataclass(frozen=True)
class FourierAffineCalibration:
    """Least-squares affine map between registered vector families."""

    matrix: np.ndarray
    isotropic_scale: float
    rotation_rad: float
    singular_scales: tuple[float, float]
    rms_residual: float


def fit_fourier_affine(
    reference_vectors: np.ndarray,
    measured_vectors: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> FourierAffineCalibration:
    """Fit scale, rotation, and anisotropy from two or more known vectors.

    The returned matrix maps row vectors as ``measured ~= reference @ M.T``.
    Callers may pass real-space layer vectors, reciprocal-lattice vectors, or
    calibrated position/tilt responses as long as both arrays use the same
    ordering and units.
    """

    reference = np.asarray(reference_vectors, dtype=float)
    measured = np.asarray(measured_vectors, dtype=float)
    if reference.shape != measured.shape or reference.ndim != 2 or reference.shape[1] != 2:
        raise ValueError("reference_vectors and measured_vectors must share shape (n, 2).")
    if reference.shape[0] < 2 or np.linalg.matrix_rank(reference) < 2:
        raise ValueError("At least two non-collinear vector intervals are required.")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(measured)):
        raise ValueError("Vector arrays must be finite.")
    if weights is None:
        root_weight = np.ones(reference.shape[0])
    else:
        weight = np.asarray(weights, dtype=float)
        if weight.shape != (reference.shape[0],) or np.any(weight <= 0.0):
            raise ValueError("weights must be a positive vector of length n.")
        root_weight = np.sqrt(weight)
    design = reference * root_weight[:, None]
    target = measured * root_weight[:, None]
    transpose, *_ = np.linalg.lstsq(design, target, rcond=None)
    matrix = transpose.T
    prediction = reference @ matrix.T
    rms = float(np.sqrt(np.mean(np.sum((prediction - measured) ** 2, axis=1))))

    # Polar decomposition M = R P separates an orientation-preserving rotation
    # from the two image-scale axes.  If data contain a reflection, retain its
    # sign in the smaller scale rather than silently reporting a proper rotation.
    left, singular_values, right_t = np.linalg.svd(matrix)
    rotation = left @ right_t
    scales = singular_values.copy()
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        scales[-1] *= -1.0
        rotation = left @ right_t
    angle = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    isotropic = float(np.sqrt(abs(np.linalg.det(matrix))))
    return FourierAffineCalibration(
        matrix=matrix,
        isotropic_scale=isotropic,
        rotation_rad=angle,
        singular_scales=(float(scales[0]), float(scales[1])),
        rms_residual=rms,
    )


def extract_image_observables(
    intensity_yx: np.ndarray,
    *,
    n_fourier_vectors: int = 4,
    dc_exclusion_radius_px: float = 3.0,
    fourier_window: bool = True,
    subpixel_fourier: bool = True,
) -> ImageObservables:
    """Extract unnormalised moments and dominant subpixel Fourier vectors.

    A separable Hann window suppresses crop leakage.  Local quadratic fits to
    log magnitude provide fractional-bin vectors needed for sub-percent scale
    estimates; both choices can be disabled for diagnostic comparisons.
    """

    if int(n_fourier_vectors) != n_fourier_vectors or n_fourier_vectors < 0:
        raise ValueError("n_fourier_vectors must be a non-negative integer.")
    n_fourier_vectors = int(n_fourier_vectors)
    image = np.asarray(intensity_yx, dtype=float)
    if image.ndim != 2 or min(image.shape) < 4:
        raise ValueError("intensity_yx must be a two-dimensional image >= 4x4.")
    if not np.all(np.isfinite(image)) or np.any(image < 0.0):
        raise ValueError("intensity_yx must be finite and non-negative.")
    total = float(np.sum(image))
    if total <= 0.0:
        raise ValueError("intensity_yx must contain positive power.")
    yy, xx = np.indices(image.shape, dtype=float)
    cx = float(np.sum(image * xx) / total)
    cy = float(np.sum(image * yy) / total)
    dx = xx - cx
    dy = yy - cy
    covariance = np.asarray(
        (
            (np.sum(image * dx * dx) / total, np.sum(image * dx * dy) / total),
            (np.sum(image * dx * dy) / total, np.sum(image * dy * dy) / total),
        )
    )
    eigval, eigvec = np.linalg.eigh(covariance)
    order = np.argsort(eigval)[::-1]
    eigval = np.maximum(eigval[order], 0.0)
    eigvec = eigvec[:, order]
    rotation = float(np.arctan2(eigvec[1, 0], eigvec[0, 0]))

    if n_fourier_vectors == 0:
        return ImageObservables(
            total_intensity=total,
            centroid_xy_px=(cx, cy),
            covariance_xy_px2=covariance,
            principal_sigma_px=(float(np.sqrt(eigval[0])), float(np.sqrt(eigval[1]))),
            principal_rotation_rad=rotation,
            fourier_vectors_yx_px=np.empty((0, 2), dtype=float),
        )

    fourier_input = image - np.mean(image)
    if fourier_window:
        window_y = np.hanning(image.shape[0])
        window_x = np.hanning(image.shape[1])
        fourier_input = fourier_input * np.outer(window_y, window_x)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(fourier_input)))
    fy, fx = np.indices(image.shape)
    # ``fftshift`` places the zero-frequency bin at floor(n/2), including for
    # even-sized Medipix frames.  Using the geometric image centre would add a
    # spurious half-pixel to every reciprocal vector on a 256x256 detector.
    oy = float(image.shape[0] // 2)
    ox = float(image.shape[1] // 2)
    spectrum[np.hypot(fy - oy, fx - ox) <= dc_exclusion_radius_px] = 0.0
    maximum = ndimage.maximum_filter(spectrum, size=3, mode="nearest")
    candidates = np.argwhere((spectrum == maximum) & (spectrum > 0.0))
    if candidates.size:
        power = spectrum[candidates[:, 0], candidates[:, 1]]
        candidates = candidates[np.argsort(power)[::-1]]
    selected: list[np.ndarray] = []
    for candidate in candidates:
        refined = candidate.astype(float)
        if subpixel_fourier:
            row, column = (int(candidate[0]), int(candidate[1]))
            if 0 < row < spectrum.shape[0] - 1 and 0 < column < spectrum.shape[1] - 1:
                tiny = np.finfo(float).tiny
                log_y = np.log(
                    np.maximum(spectrum[row - 1 : row + 2, column], tiny)
                )
                log_x = np.log(
                    np.maximum(spectrum[row, column - 1 : column + 2], tiny)
                )

                def parabola_offset(values: np.ndarray) -> float:
                    denominator = values[0] - 2.0 * values[1] + values[2]
                    if abs(denominator) <= 100.0 * np.finfo(float).eps:
                        return 0.0
                    return float(
                        np.clip(0.5 * (values[0] - values[2]) / denominator, -0.5, 0.5)
                    )

                refined += np.asarray(
                    (parabola_offset(log_y), parabola_offset(log_x))
                )
        vector = refined - np.asarray((oy, ox))
        # Treat Friedel pairs as one known line family.
        if any(
            np.linalg.norm(vector + prior) < 1.5
            or np.linalg.norm(vector - prior) < 1.5
            for prior in selected
        ):
            continue
        selected.append(vector)
        if len(selected) == int(n_fourier_vectors):
            break
    vectors = np.asarray(selected, dtype=float).reshape((-1, 2))
    return ImageObservables(
        total_intensity=total,
        centroid_xy_px=(cx, cy),
        covariance_xy_px2=covariance,
        principal_sigma_px=(float(np.sqrt(eigval[0])), float(np.sqrt(eigval[1]))),
        principal_rotation_rad=rotation,
        fourier_vectors_yx_px=vectors,
    )


@dataclass(frozen=True)
class FramePrediction:
    """One coherent optical prediction tied to an executable frame request."""

    request: FrameRequest
    render: GaussianRenderResult
    observables: ImageObservables
    interaction_route: str
    incident_sample_field_yx: np.ndarray | jax.Array | None = None
    transmitted_sample_field_yx: np.ndarray | jax.Array | None = None


class CheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint belongs to a different acquisition."""


class IncompleteCheckpointError(RuntimeError):
    """Raised when a requested physical checkpoint is not wholly durable."""


@dataclass(frozen=True)
class CheckpointedFrameRecord:
    """A validated frame stored by :func:`render_frame_sequence_checkpointed`.

    The paths point only to numeric ``.npy`` arrays.  Request and detector
    metadata live in canonical JSON, so loading a checkpoint never requires
    enabling NumPy object pickles.
    """

    frame_id: int
    request_sha256: str
    actual_interaction_route: str
    frozen_topology_id: str | None
    intensity_path: Path
    field_path: Path | None
    incident_sample_field_path: Path | None
    transmitted_sample_field_path: Path | None
    reused_from_checkpoint: bool

    def load_intensity(self, *, mmap_mode: str | None = None) -> np.ndarray:
        """Load the detector intensity with object deserialization disabled."""

        return np.load(self.intensity_path, allow_pickle=False, mmap_mode=mmap_mode)

    def load_field(self, *, mmap_mode: str | None = None) -> np.ndarray:
        """Load the optional coherent detector field."""

        if self.field_path is None:
            raise ValueError("This checkpoint was created without detector fields.")
        return np.load(self.field_path, allow_pickle=False, mmap_mode=mmap_mode)


@dataclass(frozen=True)
class CheckpointedAcquisition:
    """State of a complete or deliberately chunked physical acquisition."""

    checkpoint_directory: Path
    manifest_path: Path
    manifest_sha256: str
    records: tuple[CheckpointedFrameRecord, ...]
    pending_frame_ids: tuple[int, ...]

    @property
    def complete(self) -> bool:
        return not self.pending_frame_ids

    @property
    def completed_frame_ids(self) -> tuple[int, ...]:
        return tuple(record.frame_id for record in self.records)

    @property
    def rendered_frame_ids(self) -> tuple[int, ...]:
        """Frame IDs newly rendered by this invocation."""

        return tuple(
            record.frame_id
            for record in self.records
            if not record.reused_from_checkpoint
        )

    @property
    def resumed_frame_ids(self) -> tuple[int, ...]:
        """Frame IDs validated and reused by this invocation."""

        return tuple(
            record.frame_id
            for record in self.records
            if record.reused_from_checkpoint
        )

    def record(self, frame_id: int) -> CheckpointedFrameRecord:
        """Return one completed record by its original manifest frame ID."""

        matches = [record for record in self.records if record.frame_id == frame_id]
        if not matches:
            raise KeyError(f"Frame {frame_id} is not present in this checkpoint state.")
        return matches[0]


_CHECKPOINT_SCHEMA = "temgym_core.physical_acquisition_checkpoint"
_CHECKPOINT_VERSION = 1
_CHECKPOINT_ROUTES = {
    "vacuum",
    "local_action",
    "sample_plane_reexpansion",
}


def _json_compatible(value: object, *, name: str) -> object:
    """Convert numeric metadata to strict JSON without accepting opaque objects."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        scalar = float(value)
        if not np.isfinite(scalar):
            raise ValueError(f"{name} contains a non-finite floating-point value.")
        return scalar
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError(f"{name} cannot contain object arrays.")
        return _json_compatible(value.tolist(), name=name)
    if isinstance(value, Mapping):
        converted: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{name} mapping keys must be strings.")
            converted[key] = _json_compatible(item, name=f"{name}.{key}")
        return converted
    if isinstance(value, (list, tuple)):
        return [
            _json_compatible(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{name} contains unsupported metadata type {type(value).__name__!r}; "
        "use JSON scalars, numeric arrays, lists, and string-keyed mappings."
    )


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _frame_request_payload(request: FrameRequest) -> dict[str, object]:
    state = request.sample_state
    return {
        "frame_id": int(request.frame_id),
        "optical_state_index": int(request.optical_state_index),
        "optical_state_label": request.optical_state_label,
        "measurement": request.measurement,
        "sample_state": {
            "x_m": float(state.x),
            "theta_x_rad": float(state.theta_x),
            "y_m": float(state.y),
            "theta_y_rad": float(state.theta_y),
            "label": state.label,
        },
        "raw_dac_values": list(request.raw_dac_values),
        "target_magnification": request.target_magnification,
        "reference_mode_magnification": request.reference_mode_magnification,
        "predicted_effective_magnification": (
            request.predicted_effective_magnification
        ),
        "dac_approach": request.dac_approach,
        "dac_approach_start_values": list(request.dac_approach_start_values),
        "dac_ramp_steps": int(request.dac_ramp_steps),
        "settling_time_s": float(request.settling_time_s),
        "feature_family": request.feature_family,
        "selected_region_name": request.selected_region_name,
        "sample_level_index": request.sample_level_index,
        "sample_translation_xy_m": list(request.sample_translation_xy_m),
        "sample_crop_bounds_xyxy_m": (
            None
            if request.sample_crop_bounds_xyxy_m is None
            else list(request.sample_crop_bounds_xyxy_m)
        ),
        "selected_spacing_m": request.selected_spacing_m,
        "repeat_index": int(request.repeat_index),
        "calibration_covariance": request.calibration_covariance.tolist(),
        "interaction_route": request.interaction_route,
        "frozen_topology_id": request.frozen_topology_id,
    }


def _detector_payload(detector: DetectorGeometry) -> dict[str, object]:
    return {
        "shape_yx": list(detector.shape_yx),
        "pixel_size_yx_m": list(detector.pixel_size_yx_m),
        "z_m": float(detector.z_m),
        "centre_xy_m": list(detector.centre_xy_m),
        "mtf_sigma_px": float(detector.mtf_sigma_px),
        "affine_xy": detector.affine_xy.tolist(),
        "coordinate_convention": "pixel_centres_yx_storage",
    }


def _atomic_write_bytes(path: Path, contents: bytes) -> None:
    """Replace one checkpoint file only after its complete contents are durable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    values = np.asarray(array)
    if values.dtype.hasobject:
        raise TypeError("Checkpoint arrays cannot use object dtype.")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.save(stream, values, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_record(path: Path, root: Path, array: np.ndarray) -> dict[str, object]:
    values = np.asarray(array)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _file_sha256(path),
        "shape": list(values.shape),
        "dtype": values.dtype.str,
    }


def _observables_payload(observables: ImageObservables) -> dict[str, object]:
    return {
        "total_intensity": float(observables.total_intensity),
        "centroid_xy_px": list(observables.centroid_xy_px),
        "covariance_xy_px2": np.asarray(observables.covariance_xy_px2).tolist(),
        "principal_sigma_px": list(observables.principal_sigma_px),
        "principal_rotation_rad": float(observables.principal_rotation_rad),
        "fourier_vectors_yx_px": np.asarray(
            observables.fourier_vectors_yx_px
        ).tolist(),
    }


def _strict_json_document(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")
    return document


def _validate_saved_array(
    root: Path,
    metadata: object,
    *,
    expected_path: Path,
    expected_shape: tuple[int, ...] | None,
    expected_dtype: np.dtype,
) -> Path:
    if not isinstance(metadata, dict):
        raise ValueError("Missing checkpoint array metadata.")
    relative = metadata.get("path")
    if relative != expected_path.relative_to(root).as_posix():
        raise ValueError("Checkpoint array path does not match the frame ID.")
    path = root / str(relative)
    if not path.is_file() or metadata.get("sha256") != _file_sha256(path):
        raise ValueError("Checkpoint array is missing or has failed its checksum.")
    array = np.load(path, allow_pickle=False, mmap_mode="r")
    if array.dtype.hasobject or array.dtype != expected_dtype:
        raise ValueError("Checkpoint array dtype does not match the manifest.")
    shape = tuple(int(value) for value in metadata.get("shape", ()))
    if tuple(array.shape) != shape or (
        expected_shape is not None and tuple(array.shape) != expected_shape
    ):
        raise ValueError("Checkpoint array shape does not match the manifest.")
    if metadata.get("dtype") != array.dtype.str or not np.all(np.isfinite(array)):
        raise ValueError("Checkpoint array metadata or numerical values are invalid.")
    return path


def _validated_checkpoint_record(
    root: Path,
    request: FrameRequest,
    *,
    manifest_sha256: str,
    detector: DetectorGeometry,
    real_dtype: np.dtype,
    complex_dtype: np.dtype,
    store_complex_field: bool,
    store_sample_fields: bool,
) -> CheckpointedFrameRecord | None:
    prefix = f"frame_{request.frame_id:08d}"
    record_path = root / "frames" / f"{prefix}.json"
    if not record_path.is_file():
        return None
    try:
        document = _strict_json_document(record_path)
        payload = document["payload"]
        if (
            document.get("schema") != _CHECKPOINT_SCHEMA
            or document.get("version") != _CHECKPOINT_VERSION
            or not isinstance(payload, dict)
            or document.get("sha256") != _json_sha256(payload)
            or payload.get("manifest_sha256") != manifest_sha256
            or payload.get("frame_id") != request.frame_id
            or payload.get("request_sha256")
            != _json_sha256(_frame_request_payload(request))
            or payload.get("declared_interaction_route") != request.interaction_route
            or payload.get("frozen_topology_id") != request.frozen_topology_id
        ):
            return None
        actual_route = str(payload["actual_interaction_route"])
        if actual_route not in _CHECKPOINT_ROUTES:
            return None
        if (
            request.interaction_route != "auto"
            and actual_route != request.interaction_route
        ):
            return None
        arrays = payload["arrays"]
        if not isinstance(arrays, dict):
            return None
        intensity_path = _validate_saved_array(
            root,
            arrays.get("intensity_yx"),
            expected_path=root / "frames" / f"{prefix}_intensity.npy",
            expected_shape=detector.shape_yx,
            expected_dtype=real_dtype,
        )
        optional: dict[str, Path | None] = {}
        specifications = (
            ("field_yx", store_complex_field, detector.shape_yx),
            ("incident_sample_field_yx", store_sample_fields, None),
            ("transmitted_sample_field_yx", store_sample_fields, None),
        )
        for name, enabled, shape in specifications:
            metadata = arrays.get(name)
            if metadata is None:
                if enabled and name == "field_yx":
                    return None
                optional[name] = None
                continue
            if not enabled:
                return None
            optional[name] = _validate_saved_array(
                root,
                metadata,
                expected_path=root / "frames" / f"{prefix}_{name[:-3]}.npy",
                expected_shape=shape,
                expected_dtype=complex_dtype,
            )
        if store_sample_fields and actual_route == "sample_plane_reexpansion":
            incident_path = optional["incident_sample_field_yx"]
            transmitted_path = optional["transmitted_sample_field_yx"]
            if incident_path is None or transmitted_path is None:
                return None
            incident_shape = np.load(
                incident_path, allow_pickle=False, mmap_mode="r"
            ).shape
            transmitted_shape = np.load(
                transmitted_path, allow_pickle=False, mmap_mode="r"
            ).shape
            if len(incident_shape) != 2 or incident_shape != transmitted_shape:
                return None
        return CheckpointedFrameRecord(
            frame_id=request.frame_id,
            request_sha256=str(payload["request_sha256"]),
            actual_interaction_route=actual_route,
            frozen_topology_id=request.frozen_topology_id,
            intensity_path=intensity_path,
            field_path=optional["field_yx"],
            incident_sample_field_path=optional["incident_sample_field_yx"],
            transmitted_sample_field_path=optional["transmitted_sample_field_yx"],
            reused_from_checkpoint=True,
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        # A process may have stopped between the atomic array and record writes,
        # or storage may have been damaged.  The caller safely recomputes this
        # frame while leaving every other verified checkpoint untouched.
        return None


def _checkpoint_manifest_payload(
    requests: Sequence[FrameRequest],
    detector: DetectorGeometry,
    *,
    render_signature: str,
    render_metadata: Mapping[str, object] | None,
    store_complex_field: bool,
    store_sample_fields: bool,
    storage_dtype: str | np.dtype,
) -> tuple[tuple[FrameRequest, ...], dict[str, object], str, np.dtype, np.dtype]:
    """Build the canonical checkpoint identity shared by writers and readers."""

    request_tuple = tuple(requests)
    if not request_tuple:
        raise ValueError("requests must contain at least one FrameRequest.")
    if not all(isinstance(request, FrameRequest) for request in request_tuple):
        raise TypeError("requests must contain only FrameRequest instances.")
    if not isinstance(detector, DetectorGeometry):
        raise TypeError("detector must be a DetectorGeometry instance.")
    frame_ids = tuple(request.frame_id for request in request_tuple)
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("FrameRequest frame_id values must be unique.")
    signature = str(render_signature).strip()
    if not signature:
        raise ValueError("render_signature must be a non-empty stable identifier.")
    metadata = _json_compatible(
        {} if render_metadata is None else render_metadata,
        name="render_metadata",
    )
    if not isinstance(metadata, dict):
        raise TypeError("render_metadata must be a string-keyed mapping.")
    real_dtype = np.dtype(storage_dtype)
    if real_dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("storage_dtype must be float32 or float64.")
    real_dtype = real_dtype.newbyteorder("<")
    complex_dtype = np.dtype("complex64" if real_dtype.itemsize == 4 else "complex128")
    complex_dtype = complex_dtype.newbyteorder("<")
    request_payloads = [_frame_request_payload(request) for request in request_tuple]
    payload = {
        "detector": _detector_payload(detector),
        "frame_requests": request_payloads,
        "frame_order": list(frame_ids),
        "render_signature": signature,
        "render_metadata": metadata,
        "storage": {
            "format": "npy_numeric_allow_pickle_false",
            "real_dtype": real_dtype.str,
            "complex_dtype": complex_dtype.str,
            "store_complex_field": bool(store_complex_field),
            "store_sample_fields": bool(store_sample_fields),
        },
    }
    return request_tuple, payload, _json_sha256(payload), real_dtype, complex_dtype


def render_frame_sequence_checkpointed(
    requests: Sequence[FrameRequest],
    detector: DetectorGeometry,
    render_frame: Callable[[FrameRequest], FramePrediction],
    checkpoint_directory: str | os.PathLike[str],
    *,
    render_signature: str,
    render_metadata: Mapping[str, object] | None = None,
    store_complex_field: bool = False,
    store_sample_fields: bool = False,
    storage_dtype: str | np.dtype = np.float64,
    max_new_frames: int | None = None,
) -> CheckpointedAcquisition:
    """Render a physical frame sequence into an atomic, resumable checkpoint.

    ``render_frame`` normally closes over :func:`predict_frame` inputs and must
    return one :class:`FramePrediction` for the supplied request.  The explicit
    ``render_signature`` and JSON-only ``render_metadata`` must identify every
    fixed input captured by that closure (column parameters, source, sample,
    grids, and numerical settings).  Changing either, the detector, or any
    request makes resume fail closed instead of mixing incompatible frames.

    A frame becomes complete only after its numeric arrays and checksummed JSON
    record have each been atomically replaced.  Existing records are checksum-
    verified and opened with ``allow_pickle=False`` before they are reused.
    ``max_new_frames`` permits intentionally chunked jobs; omitting it runs all
    missing or damaged frames.
    """

    request_tuple, manifest_payload, manifest_sha256, real_dtype, complex_dtype = (
        _checkpoint_manifest_payload(
            requests,
            detector,
            render_signature=render_signature,
            render_metadata=render_metadata,
            store_complex_field=store_complex_field,
            store_sample_fields=store_sample_fields,
            storage_dtype=storage_dtype,
        )
    )
    if not callable(render_frame):
        raise TypeError("render_frame must be callable.")
    frame_ids = tuple(request.frame_id for request in request_tuple)
    if max_new_frames is not None:
        if int(max_new_frames) != max_new_frames or max_new_frames < 0:
            raise ValueError("max_new_frames must be a non-negative integer or None.")
        max_new_frames = int(max_new_frames)

    root = Path(checkpoint_directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    manifest_document = {
        "schema": _CHECKPOINT_SCHEMA,
        "version": _CHECKPOINT_VERSION,
        "sha256": manifest_sha256,
        "payload": manifest_payload,
    }
    if manifest_path.exists():
        try:
            existing = _strict_json_document(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise CheckpointCompatibilityError(
                "The existing acquisition manifest is unreadable or invalid."
            ) from error
        if (
            existing.get("schema") != _CHECKPOINT_SCHEMA
            or existing.get("version") != _CHECKPOINT_VERSION
            or existing.get("sha256") != manifest_sha256
            or existing.get("payload") != manifest_payload
        ):
            raise CheckpointCompatibilityError(
                "The checkpoint detector, requests, storage settings, or render "
                "identity differ from this acquisition. Use a new directory."
            )
    else:
        if any(root.iterdir()):
            raise CheckpointCompatibilityError(
                "Refusing to initialize a checkpoint in a non-empty directory "
                "without a compatible manifest."
            )
        _atomic_write_bytes(
            manifest_path,
            _canonical_json_bytes(manifest_document) + b"\n",
        )
    (root / "frames").mkdir(exist_ok=True)

    completed: list[CheckpointedFrameRecord] = []
    newly_rendered = 0
    for request in request_tuple:
        existing_record = _validated_checkpoint_record(
            root,
            request,
            manifest_sha256=manifest_sha256,
            detector=detector,
            real_dtype=real_dtype,
            complex_dtype=complex_dtype,
            store_complex_field=store_complex_field,
            store_sample_fields=store_sample_fields,
        )
        if existing_record is not None:
            completed.append(existing_record)
            continue
        if max_new_frames is not None and newly_rendered >= max_new_frames:
            continue

        prediction = render_frame(request)
        if not isinstance(prediction, FramePrediction):
            raise TypeError("render_frame must return a FramePrediction.")
        request_sha256 = _json_sha256(_frame_request_payload(request))
        if _json_sha256(_frame_request_payload(prediction.request)) != request_sha256:
            raise ValueError(
                f"Renderer returned a prediction for the wrong request at frame "
                f"{request.frame_id}."
            )
        route = str(prediction.interaction_route)
        if route not in _CHECKPOINT_ROUTES:
            raise ValueError(
                f"Renderer returned unsupported interaction route {route!r}."
            )
        if request.interaction_route != "auto" and route != request.interaction_route:
            raise ValueError(
                f"Frame {request.frame_id} declared route "
                f"{request.interaction_route!r} but rendered {route!r}."
            )
        intensity = np.asarray(prediction.render.intensity_yx)
        if (
            intensity.shape != detector.shape_yx
            or np.iscomplexobj(intensity)
            or not np.all(np.isfinite(intensity))
            or np.any(intensity < 0.0)
        ):
            raise ValueError(
                f"Frame {request.frame_id} intensity must be a finite, non-negative "
                f"real array with detector shape {detector.shape_yx}."
            )
        prefix = f"frame_{request.frame_id:08d}"
        arrays: dict[str, object] = {}
        intensity = np.asarray(intensity, dtype=real_dtype)
        intensity_path = root / "frames" / f"{prefix}_intensity.npy"
        _atomic_save_npy(intensity_path, intensity)
        arrays["intensity_yx"] = _array_record(intensity_path, root, intensity)

        field_path: Path | None = None
        if store_complex_field:
            field = np.asarray(prediction.render.field_yx)
            if field.shape != detector.shape_yx or not np.all(np.isfinite(field)):
                raise ValueError(
                    f"Frame {request.frame_id} detector field must be finite and "
                    "match the detector shape."
                )
            field = np.asarray(field, dtype=complex_dtype)
            field_path = root / "frames" / f"{prefix}_field.npy"
            _atomic_save_npy(field_path, field)
            arrays["field_yx"] = _array_record(field_path, root, field)

        sample_paths: dict[str, Path | None] = {
            "incident_sample_field_yx": None,
            "transmitted_sample_field_yx": None,
        }
        if store_sample_fields:
            if route == "sample_plane_reexpansion":
                incident_values = prediction.incident_sample_field_yx
                transmitted_values = prediction.transmitted_sample_field_yx
                if incident_values is None or transmitted_values is None:
                    raise ValueError(
                        f"Frame {request.frame_id} re-expansion must provide both "
                        "incident and transmitted sample fields when "
                        "store_sample_fields=True."
                    )
                if (
                    np.asarray(incident_values).ndim != 2
                    or np.asarray(incident_values).shape
                    != np.asarray(transmitted_values).shape
                ):
                    raise ValueError(
                        f"Frame {request.frame_id} re-expansion sample fields must "
                        "be two-dimensional and have identical declared shapes."
                    )
            for name in sample_paths:
                values = getattr(prediction, name)
                if values is None:
                    arrays[name] = None
                    continue
                values = np.asarray(values)
                if values.ndim != 2 or not np.all(np.isfinite(values)):
                    raise ValueError(
                        f"Frame {request.frame_id} {name} must be finite 2-D."
                    )
                values = np.asarray(values, dtype=complex_dtype)
                path = root / "frames" / f"{prefix}_{name[:-3]}.npy"
                _atomic_save_npy(path, values)
                arrays[name] = _array_record(path, root, values)
                sample_paths[name] = path

        record_payload = {
            "manifest_sha256": manifest_sha256,
            "frame_id": int(request.frame_id),
            "request_sha256": request_sha256,
            "declared_interaction_route": request.interaction_route,
            "actual_interaction_route": route,
            "frozen_topology_id": request.frozen_topology_id,
            "arrays": arrays,
            "observables": _observables_payload(prediction.observables),
        }
        # Validate every JSON scalar before committing the completion record.
        record_payload = _json_compatible(record_payload, name="frame_record")
        record_document = {
            "schema": _CHECKPOINT_SCHEMA,
            "version": _CHECKPOINT_VERSION,
            "sha256": _json_sha256(record_payload),
            "payload": record_payload,
        }
        record_path = root / "frames" / f"{prefix}.json"
        _atomic_write_bytes(
            record_path,
            _canonical_json_bytes(record_document) + b"\n",
        )
        completed.append(
            CheckpointedFrameRecord(
                frame_id=request.frame_id,
                request_sha256=request_sha256,
                actual_interaction_route=route,
                frozen_topology_id=request.frozen_topology_id,
                intensity_path=intensity_path,
                field_path=field_path,
                incident_sample_field_path=sample_paths["incident_sample_field_yx"],
                transmitted_sample_field_path=sample_paths[
                    "transmitted_sample_field_yx"
                ],
                reused_from_checkpoint=False,
            )
        )
        newly_rendered += 1

    completed_ids = {record.frame_id for record in completed}
    pending = tuple(
        request.frame_id
        for request in request_tuple
        if request.frame_id not in completed_ids
    )
    return CheckpointedAcquisition(
        checkpoint_directory=root,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        records=tuple(completed),
        pending_frame_ids=pending,
    )


def load_completed_frame_sequence_checkpoint(
    requests: Sequence[FrameRequest],
    detector: DetectorGeometry,
    checkpoint_directory: str | os.PathLike[str],
    *,
    render_signature: str,
    render_metadata: Mapping[str, object] | None = None,
    store_complex_field: bool = False,
    store_sample_fields: bool = False,
    storage_dtype: str | np.dtype = np.float64,
) -> CheckpointedAcquisition:
    """Load and checksum-validate a complete deterministic acquisition.

    The caller supplies the same immutable identity used for rendering.  The
    detector, ordered frame requests, render signature and metadata, and
    storage policy must match byte-for-byte at the canonical JSON level.  No
    directories or files are created, and damaged or missing frame records
    are rejected rather than silently rerendered.
    """

    request_tuple, expected_payload, manifest_sha256, real_dtype, complex_dtype = (
        _checkpoint_manifest_payload(
            requests,
            detector,
            render_signature=render_signature,
            render_metadata=render_metadata,
            store_complex_field=store_complex_field,
            store_sample_fields=store_sample_fields,
            storage_dtype=storage_dtype,
        )
    )
    root = Path(checkpoint_directory).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise IncompleteCheckpointError(
            f"Checkpoint manifest is missing from {root}."
        )
    try:
        manifest = _strict_json_document(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise CheckpointCompatibilityError(
            "The acquisition manifest is unreadable or invalid."
        ) from error
    if (
        manifest.get("schema") != _CHECKPOINT_SCHEMA
        or manifest.get("version") != _CHECKPOINT_VERSION
        or manifest.get("sha256") != manifest_sha256
        or manifest.get("payload") != expected_payload
    ):
        raise CheckpointCompatibilityError(
            "The checkpoint detector, requests, storage settings, or render "
            "identity differ from the requested acquisition."
        )

    records: list[CheckpointedFrameRecord] = []
    invalid_ids: list[int] = []
    for request in request_tuple:
        record = _validated_checkpoint_record(
            root,
            request,
            manifest_sha256=manifest_sha256,
            detector=detector,
            real_dtype=real_dtype,
            complex_dtype=complex_dtype,
            store_complex_field=store_complex_field,
            store_sample_fields=store_sample_fields,
        )
        if record is None:
            invalid_ids.append(request.frame_id)
        else:
            records.append(record)
    if invalid_ids:
        preview = ", ".join(str(value) for value in invalid_ids[:12])
        suffix = "" if len(invalid_ids) <= 12 else ", ..."
        raise IncompleteCheckpointError(
            f"Checkpoint has {len(invalid_ids)} missing, damaged, or invalid "
            f"frame record(s): {preview}{suffix}."
        )
    return CheckpointedAcquisition(
        checkpoint_directory=root,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        records=tuple(records),
        pending_frame_ids=(),
    )


def calibrated_state_centroid_covariance_px2(
    sample_to_detector_matrix: np.ndarray,
    sample_state_covariance: np.ndarray,
    detector: DetectorGeometry,
) -> np.ndarray:
    """Propagate calibrated sample-state uncertainty into detector pixels."""

    matrix = np.asarray(sample_to_detector_matrix, dtype=float)
    covariance = np.asarray(sample_state_covariance, dtype=float)
    if matrix.shape != (4, 4) or covariance.shape != (4, 4):
        raise ValueError("Expected a 4x4 transfer matrix and 4x4 state covariance.")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(covariance)):
        raise ValueError("Transfer matrix and covariance must be finite.")
    response_xy = matrix[np.ix_(_POSITION, np.arange(4))]
    response_xy = detector.affine_xy @ response_xy
    covariance_m2 = response_xy @ covariance @ response_xy.T
    pixel_xy = np.asarray(detector.pixel_size_yx_m[::-1], dtype=float)
    return covariance_m2 / np.outer(pixel_xy, pixel_xy)


def _validate_sample_view_request_contract(
    request: FrameRequest,
    sample_view: object,
) -> None:
    """Fail closed when a selected atlas view disagrees with its frame record.

    ``SampleView.selection`` is static JAX metadata, so these comparisons run
    once on the host before expensive field evaluation.  Lightweight legacy
    views without selection metadata remain supported, but a certificate-
    backed atlas view may never silently substitute another region, level,
    crop, pose, feature family, or cubic spacing.
    """

    selection = getattr(sample_view, "selection", None)
    if selection is None:
        return

    mismatches: list[str] = []
    for selection_name, requested in (
        ("selected_region_name", request.selected_region_name),
        ("level_index", request.sample_level_index),
        ("feature_family", request.feature_family),
    ):
        selected = getattr(selection, selection_name, None)
        if requested != selected:
            mismatches.append(
                f"{selection_name}: request={requested!r}, view={selected!r}"
            )

    def compare_numeric(label: str, requested: object, selected: object) -> None:
        if requested is None or selected is None:
            mismatches.append(
                f"{label}: request={requested!r}, view={selected!r}"
            )
            return
        left = np.asarray(requested, dtype=np.float64)
        right = np.asarray(selected, dtype=np.float64)
        scale = max(
            1e-300,
            float(np.max(np.abs(left))) if left.size else 0.0,
            float(np.max(np.abs(right))) if right.size else 0.0,
        )
        if left.shape != right.shape or not np.allclose(
            left,
            right,
            rtol=64.0 * np.finfo(np.float64).eps,
            atol=64.0 * np.finfo(np.float64).eps * scale,
        ):
            mismatches.append(
                f"{label}: request={requested!r}, view={selected!r}"
            )

    compare_numeric(
        "sample_translation_xy_m",
        request.sample_translation_xy_m,
        getattr(selection, "sample_translation_xy_m", None),
    )
    compare_numeric(
        "sample_crop_bounds_xyxy_m",
        request.sample_crop_bounds_xyxy_m,
        getattr(selection, "local_bounds_xyxy_m", None),
    )
    compare_numeric(
        "selected_spacing_m",
        request.selected_spacing_m,
        getattr(selection, "selected_spacing_m", None),
    )
    if mismatches:
        raise ValueError(
            "FrameRequest/sample_view frozen selection mismatch: "
            + "; ".join(mismatches)
        )


def predict_frame(
    column: HardEdgeColumnParameterization,
    parameters: np.ndarray,
    request: FrameRequest,
    detector: DetectorGeometry,
    source_beam: GaussianBeam,
    *,
    sample_view: object | None,
    interaction_mode: str | None = None,
    sample_x_coords_m: np.ndarray | None = None,
    sample_y_coords_m: np.ndarray | None = None,
    slices_per_lens: int = 24,
    phase_tol: float = 0.1,
    amplitude_tol: float = 0.02,
    support_radius: float = 2.0,
    width_over_step_yx: tuple[float, float] = (2.0, 2.0),
    tile_rows: int = 32,
    beam_batch_size: int = 128,
    evaluation_method: str = "auto",
) -> FramePrediction:
    """Run source → specimen interaction → detector for one manifest frame.

    The calibrated sample-plane state is applied as a known displacement and
    tilt (with its covariance retained on ``request``); it is never refitted as
    a beam-command scale.  ``sample_view=None`` is an explicit vacuum path.
    When ``interaction_mode`` is omitted, the frozen route declared by the
    request is used.  An explicit mode remains available for validation runs.
    ``evaluation_method='gpu_kernel_required'`` applies to both sample-plane
    re-expansion and detector evaluation and disables silent CPU fallback.
    """

    dacs = np.asarray(request.raw_dac_values, dtype=float)
    at_sample = propagate_gaussians_through_column(
        source_beam,
        column,
        parameters,
        dacs,
        segment="source_to_sample",
        slices_per_lens=slices_per_lens,
    )
    state = request.sample_state
    at_sample = at_sample.derive(
        x=at_sample.x + state.x,
        y=at_sample.y + state.y,
        dx=at_sample.dx + state.theta_x,
        dy=at_sample.dy + state.theta_y,
    )
    if sample_view is None:
        if request.interaction_route not in {"auto", "vacuum"}:
            raise ValueError(
                "A request declaring a material interaction route requires sample_view."
            )
        interaction = SampleInteractionResult(
            beam=at_sample,
            route="vacuum",
            validation=None,
        )
    else:
        if request.interaction_route == "vacuum":
            raise ValueError("A vacuum FrameRequest cannot be rendered with sample_view.")
        if (
            request.interaction_route == "sample_plane_reexpansion"
            and request.frozen_topology_id is None
        ):
            raise ValueError(
                "A sample_plane_reexpansion FrameRequest requires a "
                "frozen_topology_id."
            )
        _validate_sample_view_request_contract(request, sample_view)
        requested_mode = {
            "auto": "auto",
            "local_action": "local",
            "sample_plane_reexpansion": "reexpand",
        }[request.interaction_route]
        interaction = interact_gaussian_with_sample(
            sample_view,
            at_sample,
            mode=requested_mode if interaction_mode is None else interaction_mode,
            x_coords_m=sample_x_coords_m,
            y_coords_m=sample_y_coords_m,
            phase_tol=phase_tol,
            amplitude_tol=amplitude_tol,
            support_radius=support_radius,
            width_over_step_yx=width_over_step_yx,
            evaluation_method=evaluation_method,
            evaluation_batch_size=beam_batch_size,
        )
    at_detector = propagate_gaussians_through_column(
        interaction.beam,
        column,
        parameters,
        dacs,
        segment="sample_to_detector",
        slices_per_lens=slices_per_lens,
    )
    render = render_gaussian_detector_plane(
        at_detector,
        detector,
        tile_rows=tile_rows,
        beam_batch_size=beam_batch_size,
        method=evaluation_method,
        normalise=False,
    )
    return FramePrediction(
        request=request,
        render=render,
        observables=extract_image_observables(render.intensity_yx),
        interaction_route=interaction.route,
        incident_sample_field_yx=interaction.incident_field_yx,
        transmitted_sample_field_yx=interaction.transmitted_field_yx,
    )


def simulate_detector_counts(
    expected_intensity_yx: np.ndarray,
    *,
    dose_electrons: float,
    incident_intensity_sum: float = 1.0,
    read_noise_e: float = 0.0,
    dark_offset_e: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Generate an absolute Poisson-plus-read-noise detector frame.

    ``expected_intensity_yx / incident_intensity_sum`` is interpreted as the
    detected probability per incident electron.  Keeping the same calibrated
    ``incident_intensity_sum`` across settings preserves transmitted-intensity
    changes.  Passing each image's own sum would deliberately normalize those
    changes away and is therefore never done implicitly.
    """

    intensity = np.asarray(expected_intensity_yx, dtype=float)
    if intensity.ndim != 2 or np.any(intensity < 0.0) or not np.all(np.isfinite(intensity)):
        raise ValueError("expected_intensity_yx must be a finite non-negative image.")
    dose = float(dose_electrons)
    if dose <= 0.0 or not np.isfinite(dose):
        raise ValueError("dose_electrons must be finite and positive.")
    incident_sum = float(incident_intensity_sum)
    if incident_sum <= 0.0 or not np.isfinite(incident_sum):
        raise ValueError("incident_intensity_sum must be finite and positive.")
    if read_noise_e < 0.0 or dark_offset_e < 0.0:
        raise ValueError("Noise and dark-offset parameters must be non-negative.")
    if float(np.sum(intensity)) <= 0.0:
        raise ValueError("expected_intensity_yx must contain positive power.")
    mean = intensity * (dose / incident_sum) + float(dark_offset_e)
    rng = np.random.default_rng(int(seed))
    counts = rng.poisson(mean).astype(float)
    if read_noise_e > 0.0:
        counts += rng.normal(0.0, float(read_noise_e), size=counts.shape)
    return counts


@dataclass(frozen=True)
class DetectorNoiseModel:
    """Absolute detector-noise parameters for one exposure family."""

    dose_electrons: float
    incident_intensity_sum: float = 1.0
    read_noise_e: float = 0.0
    dark_offset_e: float = 0.0

    def __post_init__(self) -> None:
        for name in ("dose_electrons", "incident_intensity_sum"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        for name in ("read_noise_e", "dark_offset_e"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class KnownFourierFeatureSet:
    """Certificate-defined specimen vectors used by a material frame.

    ``world_reciprocal_vectors_xy_per_m`` is the model-side quantity: it lives
    in microscope specimen coordinates and is transformed by the fitted
    sample-to-detector map.  Local vectors are retained to audit sample-pose
    transformations.  ``registration_vectors_yx_px`` is optional and is used
    only to keep measured detector peaks in a fixed order/sign; it is never an
    inverse target.
    """

    labels: tuple[str, ...]
    local_reciprocal_vectors_xy_per_m: np.ndarray
    world_reciprocal_vectors_xy_per_m: np.ndarray
    covariance_world_per_m2: np.ndarray | None = None
    registration_vectors_yx_px: np.ndarray | None = None
    certificate_id: str | None = None
    covariance_provenance: str | None = None
    calibration_group_name: str | None = None

    def __post_init__(self) -> None:
        labels = tuple(str(label) for label in self.labels)
        local = np.asarray(self.local_reciprocal_vectors_xy_per_m, dtype=float)
        world = np.asarray(self.world_reciprocal_vectors_xy_per_m, dtype=float)
        if (
            not labels
            or len(set(labels)) != len(labels)
            or local.shape != world.shape
            or local.shape != (len(labels), 2)
            or not np.all(np.isfinite(local))
            or not np.all(np.isfinite(world))
            or np.any(np.linalg.norm(world, axis=1) <= 0.0)
        ):
            raise ValueError(
                "Known Fourier labels and finite non-zero local/world vectors "
                "must share shape (n, 2)."
            )
        covariance = self.covariance_world_per_m2
        if covariance is not None:
            covariance = np.asarray(covariance, dtype=float)
            expected_shape = (2 * len(labels), 2 * len(labels))
            covariance_scale = float(np.max(np.abs(covariance), initial=0.0))
            symmetry_tolerance = max(
                1e-15,
                1e-12 * covariance_scale,
            )
            eigenvalues = (
                np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
                if covariance.shape == expected_shape
                and np.all(np.isfinite(covariance))
                else np.asarray((-np.inf,))
            )
            eigen_tolerance = 1e-10 * max(
                float(np.max(np.abs(eigenvalues), initial=0.0)),
                np.finfo(float).tiny,
            )
            if (
                covariance.shape != expected_shape
                or not np.all(np.isfinite(covariance))
                or np.max(np.abs(covariance - covariance.T), initial=0.0)
                > symmetry_tolerance
                or eigenvalues.min() < -eigen_tolerance
            ):
                raise ValueError(
                    "covariance_world_per_m2 must be finite, symmetric, positive "
                    "semidefinite, and match the flattened world vectors."
                )
            covariance = covariance.copy()
            covariance.setflags(write=False)
        registration = self.registration_vectors_yx_px
        if registration is not None:
            registration = np.asarray(registration, dtype=float)
            if registration.shape != world.shape or not np.all(np.isfinite(registration)):
                raise ValueError(
                    "registration_vectors_yx_px must be finite and match the "
                    "known-vector shape."
                )
            registration = registration.copy()
            registration.setflags(write=False)
        certificate_id = (
            None
            if self.certificate_id is None
            else str(self.certificate_id).strip()
        )
        if certificate_id == "":
            raise ValueError("certificate_id must be non-empty when supplied.")
        covariance_provenance = (
            None
            if self.covariance_provenance is None
            else str(self.covariance_provenance).strip()
        )
        calibration_group_name = (
            None
            if self.calibration_group_name is None
            else str(self.calibration_group_name).strip()
        )
        if covariance is not None:
            if covariance_provenance is None:
                if certificate_id is None:
                    raise ValueError(
                        "Fourier certificate covariance requires certificate_id or "
                        "explicit covariance_provenance."
                    )
                covariance_provenance = f"certificate:{certificate_id}"
            if calibration_group_name is None:
                if certificate_id is None:
                    raise ValueError(
                        "Fourier certificate covariance requires a shared "
                        "calibration_group_name or certificate_id."
                    )
                calibration_group_name = (
                    f"specimen_fourier:{certificate_id}:{','.join(labels)}"
                )
        if covariance_provenance == "":
            raise ValueError("covariance_provenance must be non-empty when supplied.")
        if calibration_group_name == "":
            raise ValueError("calibration_group_name must be non-empty when supplied.")
        local = local.copy()
        world = world.copy()
        local.setflags(write=False)
        world.setflags(write=False)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "local_reciprocal_vectors_xy_per_m", local)
        object.__setattr__(self, "world_reciprocal_vectors_xy_per_m", world)
        object.__setattr__(self, "covariance_world_per_m2", covariance)
        object.__setattr__(self, "registration_vectors_yx_px", registration)
        object.__setattr__(self, "certificate_id", certificate_id)
        object.__setattr__(self, "covariance_provenance", covariance_provenance)
        object.__setattr__(self, "calibration_group_name", calibration_group_name)


def known_fourier_features_from_sample_view(
    atlas: object,
    view: object,
) -> KnownFourierFeatureSet | None:
    """Derive certificate reciprocal vectors for one frozen MAG*I*CAL view.

    Multilayer-period views expose one local layer-normal vector.  Silicon
    lattice views expose two orthogonal projected Si {111} tile vectors.
    Individual-layer-width views intentionally return ``None`` because their
    primary observables are certified interface locations, not two arbitrary
    Fourier harmonics; callers should provide a boundary extractor for them.
    """

    certificate = getattr(atlas, "certificate", None)
    selection = getattr(view, "selection", None)
    if certificate is None or selection is None:
        raise TypeError("atlas and view must expose certificate and selection metadata.")
    family = str(getattr(selection, "feature_family", ""))
    projection = np.asarray(getattr(view, "projection_world_from_local"), dtype=float)
    if projection.shape != (2, 2) or not np.all(np.isfinite(projection)):
        raise ValueError("view projection_world_from_local must be a finite 2x2 map.")
    reciprocal_projection = np.linalg.inv(projection).T
    certificate_id = str(getattr(certificate, "version", "unspecified-certificate"))

    if family == "multilayer_period":
        spacing = float(certificate.multilayer_period_m)
        spacing_sigma = float(
            np.hypot(
                certificate.silicon_layer_width_uncertainty_m,
                certificate.silicon_germanium_layer_width_uncertainty_m,
            )
        )
        local = np.asarray(((1.0 / spacing, 0.0),))
        world = (reciprocal_projection @ local.T).T
        derivative = -world.reshape(-1) / spacing
        covariance = np.outer(derivative, derivative) * spacing_sigma**2
        return KnownFourierFeatureSet(
            labels=("multilayer_period",),
            local_reciprocal_vectors_xy_per_m=local,
            world_reciprocal_vectors_xy_per_m=world,
            covariance_world_per_m2=covariance,
            certificate_id=certificate_id,
            covariance_provenance=(
                f"{certificate_id}:independent Si and SiGe layer-width "
                "standard uncertainties propagated to their summed period"
            ),
            calibration_group_name=(
                f"specimen_multilayer_period:{certificate_id}"
            ),
        )

    if family == "silicon_lattice_vectors":
        spacing = float(certificate.silicon_lattice_metadata["spacing_m"])
        spacing_sigma = float(
            certificate.silicon_lattice_spacing_uncertainty_m
        )
        local = np.eye(2) / spacing
        world = (reciprocal_projection @ local.T).T
        # The certificate spacing is a shared scale uncertainty, so both
        # reciprocal axes are correlated rather than receiving independent
        # diagonal errors.
        derivative = -world.reshape(-1) / spacing
        covariance = np.outer(derivative, derivative) * spacing_sigma**2
        return KnownFourierFeatureSet(
            labels=("si_111_axis_0", "si_111_axis_1"),
            local_reciprocal_vectors_xy_per_m=local,
            world_reciprocal_vectors_xy_per_m=world,
            covariance_world_per_m2=covariance,
            certificate_id=certificate_id,
            covariance_provenance=(
                f"{certificate_id}:shared certified Si {{111}} spacing"
            ),
            calibration_group_name=f"specimen_si111_spacing:{certificate_id}",
        )

    return None


@dataclass(frozen=True)
class NoisyDetectorFrame:
    """One independent synthetic detector exposure derived from a checkpoint."""

    request: FrameRequest
    counts_yx: np.ndarray
    noise_seed: int
    noise_model: DetectorNoiseModel

    def __post_init__(self) -> None:
        counts = np.asarray(self.counts_yx, dtype=float)
        if counts.ndim != 2 or not np.all(np.isfinite(counts)):
            raise ValueError("counts_yx must be a finite two-dimensional array.")
        counts = counts.copy()
        counts.setflags(write=False)
        object.__setattr__(self, "counts_yx", counts)


@dataclass(frozen=True)
class CheckpointObservableDataset:
    """Physical noisy frames and their covariance-aware inversion dataset."""

    acquisition: CheckpointedAcquisition
    dataset: HeterogeneousInversionDataset
    noisy_frames: tuple[NoisyDetectorFrame, ...]
    bootstrap_replicates: int

    def frame(self, frame_id: int) -> NoisyDetectorFrame:
        matches = [frame for frame in self.noisy_frames if frame.request.frame_id == frame_id]
        if not matches:
            raise KeyError(f"Frame {frame_id} is not present in this dataset.")
        return matches[0]


_MOMENT_OBSERVABLE_NAMES = (
    "total_counts_dark_corrected",
    "centroid_x_px",
    "centroid_y_px",
    "covariance_xx_px2",
    "covariance_xy_px2",
    "covariance_yy_px2",
)


def _registered_fourier_vectors(
    image_yx: np.ndarray,
    reference_vectors_yx_px: np.ndarray,
) -> np.ndarray:
    """Extract noisy Fourier peaks in a fixed reference order and sign."""

    reference = np.asarray(reference_vectors_yx_px, dtype=float).reshape((-1, 2))
    if not len(reference):
        return reference.copy()
    # Ask for surplus candidates because shot noise can promote weak local
    # maxima.  Assignment then prevents two references sharing the same peak.
    extracted = extract_image_observables(
        image_yx,
        n_fourier_vectors=max(2 * len(reference) + 4, len(reference)),
    ).fourier_vectors_yx_px
    if len(extracted) < len(reference):
        raise ValueError(
            "Noisy frame contains fewer Fourier peak families than its "
            "deterministic reference. Increase dose or reduce requested peaks."
        )
    direct = np.linalg.norm(
        reference[:, None, :] - extracted[None, :, :], axis=-1
    )
    reversed_cost = np.linalg.norm(
        reference[:, None, :] + extracted[None, :, :], axis=-1
    )
    cost = np.minimum(direct, reversed_cost)
    row, column = linear_sum_assignment(cost)
    registered = np.empty_like(reference)
    for reference_index, candidate_index in zip(row, column, strict=True):
        candidate = extracted[candidate_index]
        if reversed_cost[reference_index, candidate_index] < direct[
            reference_index, candidate_index
        ]:
            candidate = -candidate
        registered[reference_index] = candidate
    return registered


def _checkpoint_observable_vector(
    counts_yx: np.ndarray,
    *,
    dark_offset_e: float,
    reference_vectors_yx_px: np.ndarray,
) -> np.ndarray:
    """Return unnormalised intensity, moments, and registered Fourier peaks."""

    counts = np.asarray(counts_yx, dtype=float)
    corrected = counts - float(dark_offset_e)
    spatial_weights = np.maximum(corrected, 0.0)
    if float(np.sum(spatial_weights)) <= 0.0:
        raise ValueError(
            "Dark-corrected frame has no positive spatial weight; increase dose."
        )
    moments = extract_image_observables(spatial_weights, n_fourier_vectors=0)
    covariance = moments.covariance_xy_px2
    values = [
        float(np.sum(corrected)),
        moments.centroid_xy_px[0],
        moments.centroid_xy_px[1],
        float(covariance[0, 0]),
        float(covariance[0, 1]),
        float(covariance[1, 1]),
    ]
    vectors = _registered_fourier_vectors(spatial_weights, reference_vectors_yx_px)
    values.extend(vectors.reshape(-1).tolist())
    return np.asarray(values, dtype=float)


def _bootstrap_observable_covariance(
    samples: np.ndarray,
    measured_values: np.ndarray,
    *,
    shrinkage: float,
) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] < 2:
        raise ValueError("At least two bootstrap observable vectors are required.")
    covariance = np.atleast_2d(np.cov(samples, rowvar=False, ddof=1))
    diagonal = np.maximum(np.diag(covariance), 0.0)
    covariance = (1.0 - shrinkage) * covariance + shrinkage * np.diag(diagonal)
    # Feature-wise jitter preserves heterogeneous units: a high-variance count
    # observable must not impose its scale on subpixel coordinates.
    floor = np.maximum.reduce(
        (
            diagonal * 1e-9,
            (np.abs(np.asarray(measured_values, dtype=float)) * 1e-12) ** 2,
            np.full(diagonal.shape, 1e-18),
        )
    )
    covariance = 0.5 * (covariance + covariance.T) + np.diag(floor)
    return covariance


def _validate_requests_against_plan(
    plan: AcquisitionPlan,
    requests: Sequence[FrameRequest],
) -> None:
    for request in requests:
        if request.optical_state_index >= plan.n_states:
            raise CheckpointCompatibilityError(
                f"Frame {request.frame_id} references optical state "
                f"{request.optical_state_index}, outside the supplied plan."
            )
        state = plan.states[request.optical_state_index]
        if request.optical_state_label != state.label or not np.allclose(
            request.raw_dac_values,
            state.dac_values,
            rtol=0.0,
            atol=1e-12,
        ):
            raise CheckpointCompatibilityError(
                f"Frame {request.frame_id} does not match optical state "
                f"{request.optical_state_index} in the supplied plan."
            )


def build_heterogeneous_dataset_from_checkpoint(
    plan: AcquisitionPlan,
    requests: Sequence[FrameRequest],
    detector: DetectorGeometry,
    checkpoint_directory: str | os.PathLike[str],
    *,
    render_signature: str,
    default_noise_model: DetectorNoiseModel,
    noise_model_by_measurement: Mapping[str, DetectorNoiseModel] | None = None,
    render_metadata: Mapping[str, object] | None = None,
    store_complex_field: bool = False,
    store_sample_fields: bool = False,
    storage_dtype: str | np.dtype = np.float64,
    seed: int = 0,
    bootstrap_replicates: int = 64,
    fourier_features_by_frame: Mapping[int, KnownFourierFeatureSet] | None = None,
    fourier_features_by_family: Mapping[str, KnownFourierFeatureSet] | None = None,
    layer_metrology_by_frame: Mapping[int, RegisteredLayerMetrology] | None = None,
    layer_metrology_by_family: Mapping[str, RegisteredLayerMetrology] | None = None,
    layer_covariance_augmentation_by_frame: Mapping[
        int, LayerObservableCovarianceAugmentation
    ]
    | None = None,
    fourier_covariance_augmentation_by_frame: Mapping[
        int, FourierObservableCovarianceAugmentation
    ]
    | None = None,
    moment_observables_by_measurement: Mapping[str, Sequence[str]] | None = None,
    observable_extractor_by_measurement: Mapping[
        str,
        Callable[
            [np.ndarray, FrameRequest, DetectorNoiseModel],
            tuple[Sequence[float], Sequence[str]],
        ],
    ]
    | None = None,
    covariance_shrinkage: float = 0.05,
) -> CheckpointObservableDataset:
    """Turn deterministic checkpoint fields into a physical inverse dataset.

    A distinct seed derived from ``(seed, frame_id)`` is used for every actual
    detector exposure, so repeated sample frames remain statistically
    independent even when their deterministic means are identical.  Each
    block covariance is a parametric Poisson/read-noise bootstrap around that
    frame's deterministic mean.  Fourier peaks are registered to the mean
    image before covariance estimation, avoiding sign and ordering jumps.  A
    Fourier block is emitted only when the caller supplies certificate-defined
    local and world specimen vectors by frame or feature family; the forward
    detector image is not mistaken for known specimen truth.

    The default observable vector contains absolute dark-corrected counts,
    centroid, and the three independent image-covariance entries.  Defaults
    are measurement-aware: illumination frames retain centroid/covariance,
    material frames retain absolute intensity plus certified Fourier vectors,
    and diffraction frames retain centroid plus certified Fourier vectors.
    ``moment_observables_by_measurement`` can select any subset of the six
    named moment observables for a particular predictor.  A measurement-keyed
    ``layer_metrology_by_frame`` or ``layer_metrology_by_family`` supplies a
    frozen, certificate-backed layer-boundary extractor for an
    ``individual_layer_widths`` frame.  Its invariant schema and prediction
    metadata are retained automatically.  Certificate uncertainty is added
    to the detector bootstrap covariance.  An independently validated
    discrepancy term can be supplied with
    ``layer_covariance_augmentation_by_frame``; its decomposition is retained
    so repeat pooling does not divide shared uncertainty by the repeat count.
    State-specific push-forwards of reciprocal-vector and wavelength
    calibration covariance can likewise be supplied with
    ``fourier_covariance_augmentation_by_frame``.  Their input-space shared
    latent factors and output decomposition are retained; callers performing
    a cross-state global fit must use those shared factors rather than count
    the same certificate independently in every block.
    A measurement-keyed custom
    extractor can instead return ``(values, names)`` for another certified
    observable; the identical callable is applied to the actual exposure and
    every bootstrap realization.
    """

    request_tuple = tuple(requests)
    _validate_requests_against_plan(plan, request_tuple)
    if not isinstance(default_noise_model, DetectorNoiseModel):
        raise TypeError("default_noise_model must be a DetectorNoiseModel.")
    overrides = dict(noise_model_by_measurement or {})
    if not all(isinstance(value, DetectorNoiseModel) for value in overrides.values()):
        raise TypeError(
            "noise_model_by_measurement values must be DetectorNoiseModel instances."
        )
    if int(bootstrap_replicates) != bootstrap_replicates or bootstrap_replicates < 2:
        raise ValueError("bootstrap_replicates must be an integer >= 2.")
    bootstrap_replicates = int(bootstrap_replicates)
    if int(seed) != seed or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    seed = int(seed)
    shrinkage = float(covariance_shrinkage)
    if not np.isfinite(shrinkage) or not 0.0 < shrinkage <= 1.0:
        raise ValueError("covariance_shrinkage must lie in (0, 1].")
    features_by_frame = dict(fourier_features_by_frame or {})
    features_by_family = dict(fourier_features_by_family or {})
    if not all(
        isinstance(value, KnownFourierFeatureSet)
        for value in (*features_by_frame.values(), *features_by_family.values())
    ):
        raise TypeError("Fourier feature mappings require KnownFourierFeatureSet values.")
    unknown_frame_features = set(features_by_frame) - {
        request.frame_id for request in request_tuple
    }
    if unknown_frame_features:
        raise ValueError(
            f"Fourier features reference unknown frame IDs: "
            f"{sorted(unknown_frame_features)}."
        )
    layer_by_frame = dict(layer_metrology_by_frame or {})
    layer_by_family = dict(layer_metrology_by_family or {})
    if not all(
        isinstance(value, RegisteredLayerMetrology)
        for value in (*layer_by_frame.values(), *layer_by_family.values())
    ):
        raise TypeError(
            "Layer metrology mappings require RegisteredLayerMetrology values."
        )
    unknown_layer_frames = set(layer_by_frame) - {
        request.frame_id for request in request_tuple
    }
    if unknown_layer_frames:
        raise ValueError(
            f"Layer metrology references unknown frame IDs: "
            f"{sorted(unknown_layer_frames)}."
        )
    layer_covariance_by_frame = dict(layer_covariance_augmentation_by_frame or {})
    if not all(
        isinstance(value, LayerObservableCovarianceAugmentation)
        for value in layer_covariance_by_frame.values()
    ):
        raise TypeError(
            "Layer covariance mappings require "
            "LayerObservableCovarianceAugmentation values."
        )
    unknown_layer_covariance_frames = set(layer_covariance_by_frame) - {
        request.frame_id for request in request_tuple
    }
    if unknown_layer_covariance_frames:
        raise ValueError(
            "Layer covariance augmentation references unknown frame IDs: "
            f"{sorted(unknown_layer_covariance_frames)}."
        )
    fourier_covariance_by_frame = dict(
        fourier_covariance_augmentation_by_frame or {}
    )
    if not all(
        isinstance(value, FourierObservableCovarianceAugmentation)
        for value in fourier_covariance_by_frame.values()
    ):
        raise TypeError(
            "Fourier covariance mappings require "
            "FourierObservableCovarianceAugmentation values."
        )
    unknown_fourier_covariance_frames = set(fourier_covariance_by_frame) - {
        request.frame_id for request in request_tuple
    }
    if unknown_fourier_covariance_frames:
        raise ValueError(
            "Fourier covariance augmentation references unknown frame IDs: "
            f"{sorted(unknown_fourier_covariance_frames)}."
        )
    default_moments = {
        "perforation": _MOMENT_OBSERVABLE_NAMES,
        "position_basis": _MOMENT_OBSERVABLE_NAMES[1:],
        "angle_basis": _MOMENT_OBSERVABLE_NAMES[1:],
        "basis_orthogonal": _MOMENT_OBSERVABLE_NAMES[1:],
        "basis_negative": _MOMENT_OBSERVABLE_NAMES[1:],
        "sample": _MOMENT_OBSERVABLE_NAMES[:1],
        "sample_repeat": _MOMENT_OBSERVABLE_NAMES[:1],
        "indexed_diffraction": _MOMENT_OBSERVABLE_NAMES[1:3],
    }
    selected_moments = dict(moment_observables_by_measurement or {})
    for measurement, names in selected_moments.items():
        names = tuple(str(name) for name in names)
        if len(names) != len(set(names)) or not set(names).issubset(
            _MOMENT_OBSERVABLE_NAMES
        ):
            raise ValueError(
                f"Moment schema for {measurement!r} must contain unique names "
                "from the documented six moment observables."
            )
        selected_moments[measurement] = names
    custom_extractors = dict(observable_extractor_by_measurement or {})
    if not all(callable(extractor) for extractor in custom_extractors.values()):
        raise TypeError("Every custom observable extractor must be callable.")

    acquisition = load_completed_frame_sequence_checkpoint(
        request_tuple,
        detector,
        checkpoint_directory,
        render_signature=render_signature,
        render_metadata=render_metadata,
        store_complex_field=store_complex_field,
        store_sample_fields=store_sample_fields,
        storage_dtype=storage_dtype,
    )
    record_by_id = {record.frame_id: record for record in acquisition.records}
    blocks: list[FrameObservationBlock] = []
    noisy_frames: list[NoisyDetectorFrame] = []
    frame_metadata: dict[str, object] = {}
    for request in request_tuple:
        record = record_by_id[request.frame_id]
        expected = np.asarray(record.load_intensity(), dtype=float)
        model = overrides.get(request.measurement, default_noise_model)
        feature_set = features_by_frame.get(
            request.frame_id,
            features_by_family.get(request.feature_family),
        )
        layer_metrology = layer_by_frame.get(
            request.frame_id,
            layer_by_family.get(request.feature_family),
        )
        if feature_set is not None and layer_metrology is not None:
            raise ValueError(
                f"Frame {request.frame_id} cannot use Fourier and layer-interface "
                "material schemas simultaneously."
            )
        fourier_covariance_augmentation = fourier_covariance_by_frame.get(
            request.frame_id
        )
        if fourier_covariance_augmentation is not None and feature_set is None:
            raise ValueError(
                f"Frame {request.frame_id} has Fourier covariance but no known "
                "Fourier feature set."
            )
        if (
            fourier_covariance_augmentation is not None
            and layer_metrology is not None
        ):
            raise ValueError(
                f"Frame {request.frame_id} cannot combine Fourier and layer "
                "covariance augmentations."
            )
        layer_covariance_augmentation = None
        if layer_metrology is not None:
            layer_covariance_augmentation = layer_covariance_by_frame.get(
                request.frame_id,
                layer_observable_covariance_augmentation(layer_metrology),
            )
            if (
                layer_covariance_augmentation.observable_names
                != layer_metrology.observable_names
            ):
                raise ValueError(
                    f"Frame {request.frame_id} layer covariance schema does not "
                    "match its registered metrology."
                )
        elif request.frame_id in layer_covariance_by_frame:
            raise ValueError(
                f"Frame {request.frame_id} has layer covariance but no layer metrology."
            )
        n_fourier = 0 if feature_set is None else len(feature_set.labels)
        reference = (
            None if feature_set is None else feature_set.registration_vectors_yx_px
        )
        if reference is None:
            reference = extract_image_observables(
                expected,
                n_fourier_vectors=n_fourier,
            ).fourier_vectors_yx_px
        if len(reference) != n_fourier:
            raise ValueError(
                f"Frame {request.frame_id} supplied only {len(reference)} stable "
                f"Fourier families; {n_fourier} were requested."
            )
        moment_names = tuple(
            selected_moments.get(
                request.measurement,
                default_moments.get(request.measurement, _MOMENT_OBSERVABLE_NAMES),
            )
        )
        moment_indices = tuple(
            _MOMENT_OBSERVABLE_NAMES.index(name) for name in moment_names
        )
        fourier_names = tuple(
            name
            for label in (() if feature_set is None else feature_set.labels)
            for name in (f"fourier_{label}_y_px", f"fourier_{label}_x_px")
        )
        built_in_names = moment_names + fourier_names
        built_in_indices = moment_indices + tuple(
            range(
                len(_MOMENT_OBSERVABLE_NAMES),
                len(_MOMENT_OBSERVABLE_NAMES) + 2 * n_fourier,
            )
        )
        custom_extractor = custom_extractors.get(request.measurement)
        if layer_metrology is not None:
            if custom_extractor is not None:
                raise ValueError(
                    f"Frame {request.frame_id} has both registered layer metrology "
                    "and a measurement-wide custom extractor."
                )
            custom_extractor = layer_metrology

        def extract_selected(count_image: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
            if custom_extractor is None:
                values = _checkpoint_observable_vector(
                    count_image,
                    dark_offset_e=model.dark_offset_e,
                    reference_vectors_yx_px=reference,
                )
                return values[np.asarray(built_in_indices, dtype=int)], built_in_names
            result = custom_extractor(count_image, request, model)
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError(
                    "A custom observable extractor must return (values, names)."
                )
            values = np.asarray(result[0], dtype=float)
            names = tuple(str(name) for name in result[1])
            if (
                values.ndim != 1
                or values.size == 0
                or not np.all(np.isfinite(values))
                or len(names) != values.size
                or len(set(names)) != len(names)
            ):
                raise ValueError(
                    "Custom observable values must be finite and non-empty, and "
                    "their names must be unique and match the vector length."
                )
            return values, names

        actual_seed = int(
            np.random.SeedSequence((seed, request.frame_id, 0)).generate_state(1)[0]
        )
        counts = simulate_detector_counts(
            expected,
            dose_electrons=model.dose_electrons,
            incident_intensity_sum=model.incident_intensity_sum,
            read_noise_e=model.read_noise_e,
            dark_offset_e=model.dark_offset_e,
            seed=actual_seed,
        )
        measured, observable_names = extract_selected(counts)
        if not observable_names:
            raise ValueError(f"Frame {request.frame_id} has an empty observable schema.")
        bootstrap_values = []
        for replicate in range(bootstrap_replicates):
            replicate_seed = int(
                np.random.SeedSequence(
                    (seed, request.frame_id, replicate + 1)
                ).generate_state(1)[0]
            )
            bootstrap_counts = simulate_detector_counts(
                expected,
                dose_electrons=model.dose_electrons,
                incident_intensity_sum=model.incident_intensity_sum,
                read_noise_e=model.read_noise_e,
                dark_offset_e=model.dark_offset_e,
                seed=replicate_seed,
            )
            values, names = extract_selected(bootstrap_counts)
            if names != observable_names or values.shape != measured.shape:
                raise ValueError(
                    "A custom observable extractor must return an invariant schema "
                    "for every detector-noise realization."
                )
            bootstrap_values.append(values)
        covariance = _bootstrap_observable_covariance(
            np.asarray(bootstrap_values),
            measured,
            shrinkage=shrinkage,
        )
        if layer_covariance_augmentation is not None:
            covariance = (
                covariance + layer_covariance_augmentation.total_covariance_px2
            )
        if fourier_covariance_augmentation is not None:
            augmentation_names = (
                fourier_covariance_augmentation.observable_names
            )
            try:
                augmentation_indices = tuple(
                    observable_names.index(name) for name in augmentation_names
                )
            except ValueError as error:
                raise ValueError(
                    f"Frame {request.frame_id} Fourier covariance schema does not "
                    "match its extracted observables."
                ) from error
            covariance = np.asarray(covariance, dtype=float).copy()
            covariance[np.ix_(augmentation_indices, augmentation_indices)] += (
                fourier_covariance_augmentation.total_covariance_observable2
            )
        blocks.append(
            FrameObservationBlock(
                frame_id=request.frame_id,
                optical_state_index=request.optical_state_index,
                values=measured,
                covariance=covariance,
                observable_names=observable_names,
            )
        )
        noisy_frames.append(
            NoisyDetectorFrame(
                request=request,
                counts_yx=counts,
                noise_seed=actual_seed,
                noise_model=model,
            )
        )
        frame_metadata[str(request.frame_id)] = {
            "measurement": request.measurement,
            "request_sha256": record.request_sha256,
            "interaction_route": record.actual_interaction_route,
            "frozen_topology_id": record.frozen_topology_id,
            "noise_seed": actual_seed,
            "dose_electrons": model.dose_electrons,
            "incident_intensity_sum": model.incident_intensity_sum,
            "read_noise_e": model.read_noise_e,
            "dark_offset_e": model.dark_offset_e,
            "reference_fourier_vectors_yx_px": reference.tolist(),
            "reference_fourier_role": "peak_registration_only_not_inverse_truth",
            "known_fourier_features": (
                None
                if feature_set is None
                else {
                    "certificate_id": feature_set.certificate_id,
                    "labels": list(feature_set.labels),
                    "local_reciprocal_vectors_xy_per_m": (
                        feature_set.local_reciprocal_vectors_xy_per_m.tolist()
                    ),
                    "world_reciprocal_vectors_xy_per_m": (
                        feature_set.world_reciprocal_vectors_xy_per_m.tolist()
                    ),
                    "covariance_world_per_m2": (
                        None
                        if feature_set.covariance_world_per_m2 is None
                        else feature_set.covariance_world_per_m2.tolist()
                    ),
                    "covariance_provenance": feature_set.covariance_provenance,
                    "calibration_group_name": feature_set.calibration_group_name,
                }
            ),
            "known_layer_metrology": (
                None if layer_metrology is None else layer_metrology.to_metadata()
            ),
            "layer_covariance_augmentation": (
                None
                if layer_covariance_augmentation is None
                else layer_covariance_augmentation.to_metadata()
            ),
            "fourier_covariance_augmentation": (
                None
                if fourier_covariance_augmentation is None
                else fourier_covariance_augmentation.to_metadata()
            ),
            "sample_state_calibration_covariance": (
                request.calibration_covariance.tolist()
            ),
            "observable_extractor": (
                "built_in_moments_and_registered_fourier"
                if custom_extractor is None
                else getattr(custom_extractor, "__qualname__", repr(custom_extractor))
            ),
        }

    supplied_fourier_augmentations = tuple(
        fourier_covariance_by_frame[request.frame_id]
        for request in request_tuple
        if request.frame_id in fourier_covariance_by_frame
    )
    shared_fourier_priors = deduplicate_calibration_priors(
        supplied_fourier_augmentations
    )
    unpropagated_fourier_covariance_frame_ids = tuple(
        request.frame_id
        for request in request_tuple
        if (
            (feature := features_by_frame.get(
                request.frame_id,
                features_by_family.get(request.feature_family),
            ))
            is not None
            and feature.covariance_world_per_m2 is not None
            and request.frame_id not in fourier_covariance_by_frame
        )
    )
    dataset = HeterogeneousInversionDataset(
        plan=plan,
        blocks=tuple(blocks),
        metadata={
            "source": "deterministic_physical_checkpoint_plus_detector_noise",
            "checkpoint_directory": str(acquisition.checkpoint_directory),
            "checkpoint_manifest_sha256": acquisition.manifest_sha256,
            "render_signature": str(render_signature),
            "bootstrap_method": "parametric_poisson_read_noise",
            "bootstrap_replicates": bootstrap_replicates,
            "covariance_shrinkage": shrinkage,
            "cross_frame_covariance_assumption": (
                "conditionally_independent_detector_exposures; shared specimen, "
                "pose, and calibration uncertainties are not independent; "
                "same-state layer repeat pooling preserves its declared shared "
                "certificate/model covariance; supplied Fourier augmentations "
                "preserve same-state covariance but cross-state inference requires "
                "the declared shared input latents"
            ),
            "fourier_calibration_covariance": {
                "per_frame_output_covariance_included": bool(
                    supplied_fourier_augmentations
                ),
                "unpropagated_frame_ids": list(
                    unpropagated_fourier_covariance_frame_ids
                ),
                "global_fit_policy": (
                    "remove each included state-specific calibration augmentation "
                    "from statistical block covariance and use one shared "
                    "GaussianPriorFactor per group; otherwise cross-state shared "
                    "systematics are counted as independent"
                ),
                "shared_input_priors": [
                    gaussian_prior_to_metadata(prior)
                    for prior in shared_fourier_priors
                ],
            },
            "base_seed": seed,
            "frame_metadata": frame_metadata,
        },
    )
    return CheckpointObservableDataset(
        acquisition=acquisition,
        dataset=dataset,
        noisy_frames=tuple(noisy_frames),
        bootstrap_replicates=bootstrap_replicates,
    )


def _named_observable_subblock(
    block: FrameObservationBlock,
    names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    try:
        indices = tuple(block.observable_names.index(name) for name in names)
    except ValueError as error:
        raise ValueError(
            f"Frame {block.frame_id} is missing required observables {tuple(names)}."
        ) from error
    return (
        np.asarray(block.values)[np.asarray(indices, dtype=int)],
        np.asarray(block.covariance)[np.ix_(indices, indices)],
    )


def aggregate_checkpoint_observables_for_fit(
    product: CheckpointObservableDataset,
    *,
    material_repeat_policy: str = "pool",
) -> HeterogeneousInversionDataset:
    """Build forward-only, same-state joint blocks for physical fitting.

    Position, angle, orthogonal, and negative-basis centroids are differenced
    against the forward-approach perforation centroid in the same optical
    state.  Because those differences share a measured baseline, their joint
    covariance contains the baseline covariance in every off-diagonal 2x2
    response block.  Reverse-approach frames are deliberately excluded rather
    than silently mixed into a history-free model.

    Material frames contribute only registered certificate Fourier-vector or
    layer-interface components; unsupported intensity and arbitrary
    material-moment entries are removed.
    Repeats may be retained independently, precision-pooled, or reduced to the
    primary ``sample`` frame via ``material_repeat_policy``.  Certificate and
    world-vector prediction metadata are preserved for every retained or
    pooled material entry.  Held-out states remain ordinary plan-indexed
    blocks, so :class:`HeterogeneousInversionDataset` continues to exclude
    them from training unless explicitly requested.
    """

    if not isinstance(product, CheckpointObservableDataset):
        raise TypeError("product must be a CheckpointObservableDataset.")
    if not product.acquisition.complete:
        raise IncompleteCheckpointError(
            "Cannot aggregate an incomplete physical checkpoint dataset."
        )
    policy = str(material_repeat_policy)
    if policy not in {"retain", "pool", "primary_only"}:
        raise ValueError(
            "material_repeat_policy must be 'retain', 'pool', or 'primary_only'."
        )

    requests = tuple(frame.request for frame in product.noisy_frames)
    request_by_id = {request.frame_id: request for request in requests}
    if len(request_by_id) != len(requests):
        raise ValueError("Noisy detector frame IDs must be unique.")
    block_by_id = {block.frame_id: block for block in product.dataset.blocks}
    if set(block_by_id) != set(request_by_id):
        raise ValueError(
            "Every noisy detector frame must have exactly one observable block."
        )
    source_frame_metadata = dict(product.dataset.metadata.get("frame_metadata", {}))
    response_measurements = {
        "position_basis",
        "angle_basis",
        "basis_orthogonal",
        "basis_negative",
    }
    forward_requests = tuple(
        request for request in requests if request.dac_approach == "forward"
    )
    excluded_reverse_ids = tuple(
        request.frame_id for request in requests if request.dac_approach == "reverse"
    )

    joint_blocks: list[FrameObservationBlock] = []
    state_metadata: dict[str, object] = {}
    for state_index, state in enumerate(product.dataset.plan.states):
        state_requests = tuple(
            request
            for request in forward_requests
            if request.optical_state_index == state_index
        )
        if not state_requests:
            continue
        baselines = tuple(
            request for request in state_requests if request.measurement == "perforation"
        )
        if len(baselines) != 1:
            raise ValueError(
                f"Optical state {state.label!r} requires exactly one forward "
                "perforation baseline; found {len(baselines)}."
            )
        baseline_request = baselines[0]
        baseline_block = block_by_id[baseline_request.frame_id]
        baseline_value, baseline_covariance = _named_observable_subblock(
            baseline_block,
            ("centroid_x_px", "centroid_y_px"),
        )

        response_requests = tuple(
            request
            for request in state_requests
            if request.measurement in response_measurements
        )
        response_values: list[np.ndarray] = []
        response_covariances: list[np.ndarray] = []
        names: list[str] = []
        for request in response_requests:
            values, covariance = _named_observable_subblock(
                block_by_id[request.frame_id],
                ("centroid_x_px", "centroid_y_px"),
            )
            response_values.append(values - baseline_value)
            response_covariances.append(covariance)
            names.extend(
                (
                    f"{request.measurement}_frame_{request.frame_id}.delta_centroid_x_px",
                    f"{request.measurement}_frame_{request.frame_id}.delta_centroid_y_px",
                )
            )
        response_dimension = 2 * len(response_requests)
        response_joint_covariance = np.zeros(
            (response_dimension, response_dimension), dtype=float
        )
        for row in range(len(response_requests)):
            for column in range(len(response_requests)):
                block = baseline_covariance.copy()
                if row == column:
                    block += response_covariances[row]
                response_joint_covariance[
                    2 * row : 2 * row + 2,
                    2 * column : 2 * column + 2,
                ] = block

        material_requests = tuple(
            request
            for request in state_requests
            if request.measurement in {"sample", "sample_repeat"}
        )
        if policy == "primary_only":
            primary = tuple(
                request for request in material_requests if request.measurement == "sample"
            )
            material_requests = primary[:1]
        material_parts: list[
            tuple[FrameRequest, np.ndarray, np.ndarray, tuple[str, ...], object]
        ] = []
        for request in material_requests:
            block = block_by_id[request.frame_id]
            indices = tuple(
                index
                for index, name in enumerate(block.observable_names)
                if name.startswith(("fourier_", "layer_"))
            )
            if not indices:
                continue
            values = np.asarray(block.values)[np.asarray(indices, dtype=int)]
            covariance = np.asarray(block.covariance)[np.ix_(indices, indices)]
            observable_names = tuple(block.observable_names[index] for index in indices)
            metadata = source_frame_metadata.get(str(request.frame_id), {})
            material_parts.append(
                (request, values, covariance, observable_names, metadata)
            )

        material_values: list[np.ndarray] = []
        material_covariances: list[np.ndarray] = []
        material_provenance: list[dict[str, object]] = []
        if policy == "pool" and material_parts:
            schemas = {part[3] for part in material_parts}
            feature_identities = {
                _json_sha256(
                    {
                        "known_fourier_features": part[4].get(
                            "known_fourier_features"
                        ),
                        "known_layer_metrology": part[4].get(
                            "known_layer_metrology"
                        ),
                        "layer_covariance_augmentation": part[4].get(
                            "layer_covariance_augmentation"
                        ),
                        "fourier_covariance_augmentation": (
                            None
                            if not isinstance(
                                part[4].get("fourier_covariance_augmentation"),
                                Mapping,
                            )
                            else {
                                key: value
                                for key, value in part[4][
                                    "fourier_covariance_augmentation"
                                ].items()
                                if key != "map_provenance"
                            }
                        ),
                    }
                )
                for part in material_parts
            }
            if len(schemas) != 1 or len(feature_identities) != 1:
                raise ValueError(
                    f"Material repeats in state {state.label!r} cannot be pooled: "
                    "their observable schemas or certificate prediction specs differ."
                )
            layer_augmentation_metadata = material_parts[0][4].get(
                "layer_covariance_augmentation"
            )
            fourier_augmentation_metadata = material_parts[0][4].get(
                "fourier_covariance_augmentation"
            )
            if isinstance(layer_augmentation_metadata, Mapping) and isinstance(
                fourier_augmentation_metadata, Mapping
            ):
                raise ValueError(
                    "A material schema cannot carry both layer and Fourier "
                    "covariance augmentations."
                )
            augmentation_metadata = (
                layer_augmentation_metadata
                if isinstance(layer_augmentation_metadata, Mapping)
                else fourier_augmentation_metadata
            )
            shared_calibration_covariance = None
            independent_covariances = [part[2] for part in material_parts]
            if isinstance(augmentation_metadata, Mapping):
                covariance_key = (
                    "total_covariance_px2"
                    if "total_covariance_px2" in augmentation_metadata
                    else "total_covariance_observable2"
                )
                candidate = np.asarray(
                    augmentation_metadata.get(covariance_key), dtype=float
                )
                expected_shape = material_parts[0][2].shape
                if candidate.shape != expected_shape or not np.all(
                    np.isfinite(candidate)
                ):
                    raise ValueError(
                        "Layer covariance augmentation has an invalid pooled shape."
                    )
                shared_calibration_covariance = 0.5 * (candidate + candidate.T)
                independent_covariances = []
                for part in material_parts:
                    independent = 0.5 * (
                        part[2] - shared_calibration_covariance
                        + (part[2] - shared_calibration_covariance).T
                    )
                    eigenvalues = np.linalg.eigvalsh(independent)
                    scale = max(
                        float(np.max(np.abs(independent), initial=0.0)),
                        1e-18,
                    )
                    floor = max(1e-18, 1e-12 * scale)
                    if eigenvalues.min() < -1e-8 * scale:
                        raise ValueError(
                            "A material block covariance is smaller than its "
                            "declared shared calibration augmentation."
                        )
                    if eigenvalues.min() < floor:
                        independent = independent + np.eye(independent.shape[0]) * (
                            floor - eigenvalues.min()
                        )
                    independent_covariances.append(independent)
            precision = sum(
                np.linalg.inv(covariance)
                for covariance in independent_covariances
            )
            pooled_covariance = np.linalg.inv(precision)
            pooled_value = pooled_covariance @ sum(
                np.linalg.solve(covariance, part[1])
                for covariance, part in zip(
                    independent_covariances, material_parts, strict=True
                )
            )
            if shared_calibration_covariance is not None:
                pooled_covariance = (
                    pooled_covariance + shared_calibration_covariance
                )
            pooled_names = tuple(
                f"material_pool.{name}" for name in material_parts[0][3]
            )
            names.extend(pooled_names)
            material_values.append(pooled_value)
            material_covariances.append(pooled_covariance)
            material_provenance.append(
                {
                    "mode": "precision_pool",
                    "member_frame_ids": [part[0].frame_id for part in material_parts],
                    "known_fourier_features": material_parts[0][4].get(
                        "known_fourier_features"
                    ),
                    "known_layer_metrology": material_parts[0][4].get(
                        "known_layer_metrology"
                    ),
                    "layer_covariance_augmentation": (
                        layer_augmentation_metadata
                        if isinstance(layer_augmentation_metadata, Mapping)
                        else None
                    ),
                    "fourier_covariance_augmentation": (
                        fourier_augmentation_metadata
                        if isinstance(fourier_augmentation_metadata, Mapping)
                        else None
                    ),
                }
            )
        else:
            for request, values, covariance, observable_names, metadata in material_parts:
                prefix = f"material_frame_{request.frame_id}"
                names.extend(tuple(f"{prefix}.{name}" for name in observable_names))
                material_values.append(values)
                material_covariances.append(covariance)
                material_provenance.append(
                    {
                        "mode": "retained",
                        "member_frame_ids": [request.frame_id],
                        "known_fourier_features": metadata.get("known_fourier_features"),
                        "known_layer_metrology": metadata.get(
                            "known_layer_metrology"
                        ),
                        "layer_covariance_augmentation": metadata.get(
                            "layer_covariance_augmentation"
                        ),
                        "fourier_covariance_augmentation": metadata.get(
                            "fourier_covariance_augmentation"
                        ),
                    }
                )

        value_parts = [*response_values, *material_values]
        covariance_parts = (
            ([response_joint_covariance] if response_dimension else [])
            + material_covariances
        )
        if not value_parts:
            continue
        values = np.concatenate(value_parts)
        covariance = linalg.block_diag(*covariance_parts)
        joint_blocks.append(
            FrameObservationBlock(
                frame_id=baseline_request.frame_id,
                optical_state_index=state_index,
                values=values,
                covariance=covariance,
                observable_names=tuple(names),
            )
        )
        state_metadata[str(baseline_request.frame_id)] = {
            "optical_state_label": state.label,
            "optical_state_role": state.role,
            "held_out": state.role == "held_out",
            "baseline_frame_id": baseline_request.frame_id,
            "response_frame_ids": [request.frame_id for request in response_requests],
            "material": material_provenance,
        }

    if not joint_blocks:
        raise ValueError("No supported forward calibration observables were available.")
    return HeterogeneousInversionDataset(
        plan=product.dataset.plan,
        blocks=tuple(joint_blocks),
        metadata={
            **dict(product.dataset.metadata),
            "aggregation": {
                "approach_policy": "forward_only",
                "excluded_reverse_frame_ids": list(excluded_reverse_ids),
                "material_repeat_policy": policy,
                "response_definition": (
                    "basis centroid minus same-state forward perforation centroid"
                ),
                "response_covariance": (
                    "full same-state joint covariance with shared baseline blocks"
                ),
                "material_observables": (
                    "certificate_fourier_vectors_or_registered_layer_interfaces_only"
                ),
                "state_blocks": state_metadata,
            },
        },
    )


def shared_fourier_calibration_priors_from_dataset(
    dataset: HeterogeneousInversionDataset,
):
    """Rebuild the declared cross-state Fourier calibration priors.

    The returned values are ``GaussianPriorFactor`` instances suitable for
    ``SharedLatentNuisanceLikelihood``.  Group-name collisions are checked for
    exact agreement so one certificate cannot silently acquire a different
    uncertainty in another optical state.
    """

    if not isinstance(dataset, HeterogeneousInversionDataset):
        raise TypeError("dataset must be a HeterogeneousInversionDataset.")
    section = dataset.metadata.get("fourier_calibration_covariance")
    if not isinstance(section, Mapping):
        return ()
    entries = section.get("shared_input_priors", ())
    if not isinstance(entries, Sequence):
        raise ValueError("shared_input_priors metadata must be a sequence.")
    priors = tuple(
        gaussian_prior_from_metadata(entry)  # type: ignore[arg-type]
        for entry in entries
    )
    by_group = {}
    for prior in priors:
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
                f"Shared calibration group {prior.group_name!r} has inconsistent "
                "dataset metadata."
            )
    return tuple(by_group.values())


def strip_shared_fourier_calibration_covariance(
    dataset: HeterogeneousInversionDataset,
) -> HeterogeneousInversionDataset:
    """Remove per-block Fourier calibration terms before a shared-latent fit.

    Checkpoint construction includes propagated calibration covariance so each
    frame is statistically complete in isolation.  A multi-state
    ``physical_nuisance`` likelihood instead represents the same certificate
    and wavelength through shared input latents.  This function removes only
    those declared Fourier/Bragg augmentation subblocks, leaving detector,
    extractor, shared-baseline, and layer covariances untouched.
    """

    if not isinstance(dataset, HeterogeneousInversionDataset):
        raise TypeError("dataset must be a HeterogeneousInversionDataset.")
    frame_metadata = dataset.metadata.get("frame_metadata", {})
    aggregation = dataset.metadata.get("aggregation")
    state_blocks = (
        aggregation.get("state_blocks", {})
        if isinstance(aggregation, Mapping)
        else {}
    )
    stripped_blocks = []
    stripped_group_names: set[str] = set()
    for block in dataset.blocks:
        corrections: list[tuple[tuple[str, ...], np.ndarray, Sequence[object]]] = []
        if isinstance(state_blocks, Mapping) and str(block.frame_id) in state_blocks:
            state_metadata = state_blocks[str(block.frame_id)]
            material_entries = (
                state_metadata.get("material", ())
                if isinstance(state_metadata, Mapping)
                else ()
            )
            for entry in material_entries:
                if not isinstance(entry, Mapping):
                    continue
                augmentation = entry.get("fourier_covariance_augmentation")
                if not isinstance(augmentation, Mapping):
                    continue
                raw_names = tuple(str(value) for value in augmentation["observable_names"])
                mode = entry.get("mode")
                member_ids = tuple(entry.get("member_frame_ids", ()))
                if mode == "precision_pool":
                    prefix = "material_pool."
                elif mode == "retained" and member_ids:
                    prefix = f"material_frame_{int(member_ids[0])}."
                else:
                    raise ValueError(
                        "Cannot align a Fourier covariance augmentation with its "
                        "aggregated material observables."
                    )
                names = tuple(f"{prefix}{name}" for name in raw_names)
                covariance = np.asarray(
                    augmentation["total_covariance_observable2"], dtype=float
                )
                corrections.append(
                    (names, covariance, augmentation.get("shared_input_priors", ()))
                )
        elif isinstance(frame_metadata, Mapping):
            metadata = frame_metadata.get(str(block.frame_id), {})
            augmentation = (
                metadata.get("fourier_covariance_augmentation")
                if isinstance(metadata, Mapping)
                else None
            )
            if isinstance(augmentation, Mapping):
                names = tuple(
                    str(value) for value in augmentation["observable_names"]
                )
                covariance = np.asarray(
                    augmentation["total_covariance_observable2"], dtype=float
                )
                corrections.append(
                    (names, covariance, augmentation.get("shared_input_priors", ()))
                )

        statistical = np.asarray(block.covariance, dtype=float).copy()
        for names, covariance, prior_entries in corrections:
            if covariance.shape != (len(names), len(names)) or not np.all(
                np.isfinite(covariance)
            ):
                raise ValueError(
                    "Stored Fourier calibration covariance has an invalid shape."
                )
            try:
                indices = tuple(block.observable_names.index(name) for name in names)
            except ValueError as error:
                raise ValueError(
                    f"Frame {block.frame_id} cannot align stored Fourier "
                    "calibration covariance with its observable schema."
                ) from error
            statistical[np.ix_(indices, indices)] -= covariance
            for entry in prior_entries:
                if isinstance(entry, Mapping):
                    stripped_group_names.add(str(entry.get("group_name", "")))
        statistical = 0.5 * (statistical + statistical.T)
        eigenvalues = np.linalg.eigvalsh(statistical)
        scale = max(
            float(np.max(np.abs(statistical), initial=0.0)),
            np.finfo(float).tiny,
        )
        if eigenvalues.min() <= 0.0:
            if eigenvalues.min() < -1e-9 * scale:
                raise ValueError(
                    f"Frame {block.frame_id} covariance does not contain the "
                    "declared Fourier calibration augmentation exactly once."
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

    metadata = dict(dataset.metadata)
    covariance_metadata = dict(
        metadata.get("fourier_calibration_covariance", {})
    )
    covariance_metadata.update(
        {
            "per_frame_output_covariance_included": False,
            "stripped_for_shared_latent_likelihood": True,
            "stripped_group_names": sorted(
                value for value in stripped_group_names if value
            ),
            "global_fit_policy": (
                "statistical covariance only; use shared_input_priors exactly "
                "once through SharedLatentNuisanceLikelihood"
            ),
        }
    )
    metadata["fourier_calibration_covariance"] = covariance_metadata
    return HeterogeneousInversionDataset(
        plan=dataset.plan,
        blocks=tuple(stripped_blocks),
        metadata=metadata,
    )


@dataclass(frozen=True)
class ForwardValidationThresholds:
    sample_field_rms_noise_fraction: float = 0.2
    detector_rms_noise_fraction: float = 0.2
    observable_uncertainty_fraction: float = 0.2
    parameter_change_sigma_fraction: float = 0.1


@dataclass(frozen=True)
class ForwardValidationReport:
    sample_field_rms_in_noise_sigma: float | None
    detector_rms_in_noise_sigma: float
    observable_residuals_in_sigma: dict[str, float]
    parameter_change_in_sigma: float | None
    missing_gates: tuple[str, ...]
    passed: bool


@dataclass(frozen=True)
class GaussianRefinementSnapshot:
    """Forward and inverse results produced by one frozen Gaussian topology.

    The detector image and sample field must be evaluated on the same fixed
    validation grids at every refinement.  ``observables`` stores physical
    estimates (rather than already-normalized residuals); their experimental
    standard errors are supplied to :func:`validate_gaussian_refinement_sequence`.

    Parameters with finite positive standard errors are treated as
    identifiable by default.  Supplying ``identifiable_parameter_mask`` is the
    explicit way to exclude fixed or profile-null parameters.  The mask must
    stop changing before a refinement transition can count toward admission.
    """

    topology_id: str
    beam_count: int
    detector_intensity_yx: np.ndarray
    sample_field_yx: np.ndarray
    observables: dict[str, np.ndarray | float]
    fit_parameters: np.ndarray
    fit_standard_errors: np.ndarray
    identifiable_parameter_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        topology_id = str(self.topology_id).strip()
        if not topology_id:
            raise ValueError("topology_id must be non-empty.")
        beam_count = int(self.beam_count)
        if beam_count < 1:
            raise ValueError("beam_count must be positive.")

        detector = np.asarray(self.detector_intensity_yx, dtype=float)
        if (
            detector.ndim != 2
            or not np.all(np.isfinite(detector))
            or np.any(detector < 0.0)
        ):
            raise ValueError(
                "detector_intensity_yx must be a finite non-negative 2D image."
            )
        sample_field = np.asarray(self.sample_field_yx)
        if sample_field.ndim != 2 or not np.all(np.isfinite(sample_field)):
            raise ValueError("sample_field_yx must be a finite 2D field.")

        parameters = np.asarray(self.fit_parameters, dtype=float)
        standard_errors = np.asarray(self.fit_standard_errors, dtype=float)
        if parameters.ndim != 1 or parameters.size == 0 or not np.all(
            np.isfinite(parameters)
        ):
            raise ValueError("fit_parameters must be a non-empty finite vector.")
        if standard_errors.shape != parameters.shape:
            raise ValueError(
                "fit_standard_errors must match the fit_parameters shape."
            )
        if np.any(np.isnan(standard_errors)) or np.any(standard_errors < 0.0):
            raise ValueError("fit_standard_errors must be non-negative and not NaN.")

        if self.identifiable_parameter_mask is None:
            mask = np.isfinite(standard_errors) & (standard_errors > 0.0)
        else:
            mask = np.asarray(self.identifiable_parameter_mask, dtype=bool)
            if mask.shape != parameters.shape:
                raise ValueError(
                    "identifiable_parameter_mask must match fit_parameters."
                )
        if not np.any(mask):
            raise ValueError("At least one fit parameter must be identifiable.")
        if np.any(~np.isfinite(standard_errors[mask])) or np.any(
            standard_errors[mask] <= 0.0
        ):
            raise ValueError(
                "Identifiable parameters require finite positive standard errors."
            )

        observable_values: dict[str, np.ndarray] = {}
        for name, value in dict(self.observables).items():
            key = str(name).strip()
            array = np.asarray(value, dtype=float)
            if not key or array.size == 0 or not np.all(np.isfinite(array)):
                raise ValueError(
                    "Observable names and values must be non-empty and finite."
                )
            array = array.copy()
            array.setflags(write=False)
            observable_values[key] = array

        detector = detector.copy()
        sample_field = sample_field.copy()
        parameters = parameters.copy()
        standard_errors = standard_errors.copy()
        mask = mask.copy()
        for array in (detector, sample_field, parameters, standard_errors, mask):
            array.setflags(write=False)
        object.__setattr__(self, "topology_id", topology_id)
        object.__setattr__(self, "beam_count", beam_count)
        object.__setattr__(self, "detector_intensity_yx", detector)
        object.__setattr__(self, "sample_field_yx", sample_field)
        object.__setattr__(self, "observables", observable_values)
        object.__setattr__(self, "fit_parameters", parameters)
        object.__setattr__(self, "fit_standard_errors", standard_errors)
        object.__setattr__(self, "identifiable_parameter_mask", mask)


@dataclass(frozen=True)
class GaussianRefinementStepReport:
    """Validation result for one adjacent topology refinement."""

    previous_topology_id: str
    current_topology_id: str
    previous_beam_count: int
    current_beam_count: int
    topology_progressed: bool
    identifiable_parameter_set_stable: bool
    parameter_stability_passed: bool
    forward_validation: ForwardValidationReport
    passed: bool


@dataclass(frozen=True)
class GaussianRefinementAdmissionReport:
    """Fail-closed admission decision over a sequence of refinements."""

    steps: tuple[GaussianRefinementStepReport, ...]
    required_successive_refinements: int
    trailing_successful_refinements: int
    failure_reasons: tuple[str, ...]
    passed: bool


@dataclass(frozen=True)
class HeldOutAcceptanceThresholds:
    magnification_relative: float = 0.005
    rotation_deg: float = 0.5
    centroid_px: float = 0.5
    beam_diameter_relative: float = 0.02
    diffraction_vector_px: float = 1.0


@dataclass(frozen=True)
class HeldOutMetrics:
    magnification_relative_error: float
    rotation_error_deg: float
    centroid_error_px: float
    beam_diameter_relative_error: float
    diffraction_vector_error_px: float
    passed: bool


def evaluate_held_out_metrics(
    *,
    predicted_magnification: float,
    reference_magnification: float,
    predicted_rotation_deg: float,
    reference_rotation_deg: float,
    predicted_centroid_xy_px: Sequence[float],
    reference_centroid_xy_px: Sequence[float],
    predicted_beam_diameter_px: float,
    reference_beam_diameter_px: float,
    predicted_diffraction_vectors_yx_px: np.ndarray,
    reference_diffraction_vectors_yx_px: np.ndarray,
    thresholds: HeldOutAcceptanceThresholds = HeldOutAcceptanceThresholds(),
    require_diffraction_vectors: bool = False,
) -> HeldOutMetrics:
    """Evaluate the quantitative held-out acceptance criteria.

    Empty diffraction-vector arrays retain their historical meaning of "gate
    not requested".  Set ``require_diffraction_vectors=True`` for diffraction
    acquisitions; a missing predicted or reference set then fails closed.
    """

    reference_magnification = float(reference_magnification)
    reference_diameter = float(reference_beam_diameter_px)
    if reference_magnification == 0.0 or reference_diameter == 0.0:
        raise ValueError("Reference magnification and beam diameter must be non-zero.")
    magnification_error = abs(
        float(predicted_magnification) / reference_magnification - 1.0
    )
    # Image rotation is periodic modulo 360 degrees.
    rotation_error = abs(
        (float(predicted_rotation_deg) - float(reference_rotation_deg) + 180.0)
        % 360.0
        - 180.0
    )
    predicted_centroid = np.asarray(predicted_centroid_xy_px, dtype=float)
    reference_centroid = np.asarray(reference_centroid_xy_px, dtype=float)
    if predicted_centroid.shape != (2,) or reference_centroid.shape != (2,):
        raise ValueError("Centroids must each contain x and y pixel coordinates.")
    centroid_error = float(np.linalg.norm(predicted_centroid - reference_centroid))
    diameter_error = abs(float(predicted_beam_diameter_px) / reference_diameter - 1.0)
    predicted_vectors = np.asarray(predicted_diffraction_vectors_yx_px, dtype=float)
    reference_vectors = np.asarray(reference_diffraction_vectors_yx_px, dtype=float)
    if (
        predicted_vectors.ndim != 2
        or predicted_vectors.shape[1:] != (2,)
        or reference_vectors.ndim != 2
        or reference_vectors.shape[1:] != (2,)
        or not np.all(np.isfinite(predicted_vectors))
        or not np.all(np.isfinite(reference_vectors))
    ):
        raise ValueError(
            "Predicted and reference diffraction vectors must be finite (n, 2) arrays."
        )
    missing_required_vectors = bool(require_diffraction_vectors) and (
        predicted_vectors.shape[0] == 0 or reference_vectors.shape[0] == 0
    )
    if missing_required_vectors:
        diffraction_error = float("inf")
    elif predicted_vectors.shape != reference_vectors.shape:
        raise ValueError(
            "Predicted and reference diffraction vectors must contain matching counts."
        )
    elif predicted_vectors.size == 0:
        diffraction_error = 0.0
    else:
        # Indexed spots may arrive in any detector order and a Friedel vector
        # may be reported with either sign.  Match the two sets globally before
        # applying the one-pixel held-out criterion.
        direct = np.linalg.norm(
            predicted_vectors[:, None, :] - reference_vectors[None, :, :], axis=-1
        )
        friedel = np.linalg.norm(
            predicted_vectors[:, None, :] + reference_vectors[None, :, :], axis=-1
        )
        costs = np.minimum(direct, friedel)
        rows, columns = linear_sum_assignment(costs)
        diffraction_error = float(np.max(costs[rows, columns]))
    passed = (
        magnification_error <= thresholds.magnification_relative
        and rotation_error <= thresholds.rotation_deg
        and centroid_error <= thresholds.centroid_px
        and diameter_error <= thresholds.beam_diameter_relative
        and diffraction_error <= thresholds.diffraction_vector_px
    )
    return HeldOutMetrics(
        magnification_relative_error=magnification_error,
        rotation_error_deg=rotation_error,
        centroid_error_px=centroid_error,
        beam_diameter_relative_error=diameter_error,
        diffraction_vector_error_px=diffraction_error,
        passed=passed,
    )


def validate_forward_models(
    candidate_intensity_yx: np.ndarray,
    reference_intensity_yx: np.ndarray,
    noise_sigma_yx: float | np.ndarray,
    *,
    candidate_sample_field_yx: np.ndarray | None = None,
    reference_sample_field_yx: np.ndarray | None = None,
    sample_field_noise_sigma: float | np.ndarray | None = None,
    observable_residuals_in_sigma: dict[str, float] | None = None,
    parameter_change_in_sigma: float | None = None,
    require_complete: bool = True,
    thresholds: ForwardValidationThresholds = ForwardValidationThresholds(),
) -> ForwardValidationReport:
    """Apply the numerical gates required before images enter inversion.

    In complete mode the sample field, all five image-observable families, and
    successive-refinement parameter stability must be supplied.  Missing
    evidence fails closed rather than silently admitting an unvalidated image.
    Set ``require_complete=False`` only for an explicitly labelled exploratory
    comparison.
    """

    candidate = np.asarray(candidate_intensity_yx, dtype=float)
    reference = np.asarray(reference_intensity_yx, dtype=float)
    noise = np.asarray(noise_sigma_yx, dtype=float)
    if candidate.shape != reference.shape:
        raise ValueError("Candidate and reference detector images must match.")
    if np.any(noise <= 0.0) or not np.all(np.isfinite(noise)):
        raise ValueError("noise_sigma_yx must be finite and positive.")
    detector_rms = float(np.sqrt(np.mean(((candidate - reference) / noise) ** 2)))
    sample_field_rms: float | None = None
    supplied_sample_values = (
        candidate_sample_field_yx,
        reference_sample_field_yx,
        sample_field_noise_sigma,
    )
    if any(value is not None for value in supplied_sample_values):
        if not all(value is not None for value in supplied_sample_values):
            raise ValueError(
                "Candidate/reference sample fields and sample noise must be supplied together."
            )
        sample_candidate = np.asarray(candidate_sample_field_yx)
        sample_reference = np.asarray(reference_sample_field_yx)
        sample_noise = np.asarray(sample_field_noise_sigma, dtype=float)
        if sample_candidate.shape != sample_reference.shape:
            raise ValueError("Candidate and reference sample fields must match.")
        if np.any(sample_noise <= 0.0) or not np.all(np.isfinite(sample_noise)):
            raise ValueError("sample_field_noise_sigma must be finite and positive.")
        sample_field_rms = float(
            np.sqrt(np.mean((np.abs(sample_candidate - sample_reference) / sample_noise) ** 2))
        )
    residuals = {} if observable_residuals_in_sigma is None else {
        str(k): float(v) for k, v in observable_residuals_in_sigma.items()
    }
    required_observables = {
        "scale",
        "rotation",
        "centroid",
        "fourier_vector",
        "intensity",
    }
    missing_gates: list[str] = []
    if require_complete and sample_field_rms is None:
        missing_gates.append("sample_field")
    if require_complete:
        missing_gates.extend(
            f"observable:{name}" for name in sorted(required_observables - residuals.keys())
        )
    if require_complete and parameter_change_in_sigma is None:
        missing_gates.append("parameter_stability")
    observables_pass = all(
        abs(value) <= thresholds.observable_uncertainty_fraction
        for value in residuals.values()
    ) and not any(name.startswith("observable:") for name in missing_gates)
    parameter_pass = (
        not require_complete
        if parameter_change_in_sigma is None
        else abs(float(parameter_change_in_sigma))
        <= thresholds.parameter_change_sigma_fraction
    )
    passed = (
        (
            sample_field_rms is None
            or sample_field_rms <= thresholds.sample_field_rms_noise_fraction
        )
        and detector_rms <= thresholds.detector_rms_noise_fraction
        and observables_pass
        and parameter_pass
        and not missing_gates
    )
    return ForwardValidationReport(
        sample_field_rms_in_noise_sigma=sample_field_rms,
        detector_rms_in_noise_sigma=detector_rms,
        observable_residuals_in_sigma=residuals,
        parameter_change_in_sigma=(
            None if parameter_change_in_sigma is None else float(parameter_change_in_sigma)
        ),
        missing_gates=tuple(missing_gates),
        passed=passed,
    )


def validate_gaussian_refinement_sequence(
    snapshots: Sequence[GaussianRefinementSnapshot],
    *,
    detector_noise_sigma_yx: float | np.ndarray,
    sample_field_noise_sigma: float | np.ndarray,
    observable_standard_errors: dict[str, float | np.ndarray],
    required_successive_refinements: int = 2,
    thresholds: ForwardValidationThresholds = ForwardValidationThresholds(),
) -> GaussianRefinementAdmissionReport:
    """Admit Gaussian images only after repeated, complete convergence.

    Every adjacent pair is compared on fixed sample and detector validation
    grids.  Observable changes are normalized by their experimental standard
    errors, while fit-parameter changes use the smaller of the two adjacent
    fit standard errors.  The latter is deliberately conservative: an
    uncertainty increase cannot make a changing solution look stable.

    A transition counts only when all existing forward-model gates pass, the
    topology fingerprint changes, the beam count increases, the identifiable
    parameter set is unchanged, and *every* identifiable parameter changes by
    strictly less than ``parameter_change_sigma_fraction``.  With the default
    requirement, at least three snapshots and two consecutive passing
    transitions are therefore needed.  Earlier failed refinements do not
    prevent admission once the final two transitions pass.

    Rotation observables are compared modulo 360 degrees.  Fourier vectors
    must already carry consistent certificate indexing and sign; all other
    observables must use a continuous common coordinate convention.
    """

    if isinstance(required_successive_refinements, (bool, np.bool_)) or not isinstance(
        required_successive_refinements, (int, np.integer)
    ):
        raise ValueError("required_successive_refinements must be a positive integer.")
    required = int(required_successive_refinements)
    if required < 1:
        raise ValueError("required_successive_refinements must be a positive integer.")
    parameter_threshold = float(thresholds.parameter_change_sigma_fraction)
    if not np.isfinite(parameter_threshold) or parameter_threshold <= 0.0:
        raise ValueError("parameter_change_sigma_fraction must be finite and positive.")

    series = tuple(snapshots)
    if not all(isinstance(item, GaussianRefinementSnapshot) for item in series):
        raise TypeError("snapshots must contain GaussianRefinementSnapshot instances.")
    uncertainty_by_name = {
        str(name): np.asarray(value, dtype=float)
        for name, value in dict(observable_standard_errors).items()
    }

    step_reports: list[GaussianRefinementStepReport] = []
    for previous, current in zip(series[:-1], series[1:]):
        if current.detector_intensity_yx.shape != previous.detector_intensity_yx.shape:
            raise ValueError(
                "Successive detector validation images must match in shape."
            )
        if current.sample_field_yx.shape != previous.sample_field_yx.shape:
            raise ValueError("Successive sample validation fields must match in shape.")
        if current.fit_parameters.shape != previous.fit_parameters.shape:
            raise ValueError("Successive fit-parameter vectors must match in shape.")

        observable_residuals: dict[str, float] = {}
        common_observables = (
            current.observables.keys()
            & previous.observables.keys()
            & uncertainty_by_name.keys()
        )
        for name in sorted(common_observables):
            candidate_value = np.asarray(current.observables[name], dtype=float)
            reference_value = np.asarray(previous.observables[name], dtype=float)
            if candidate_value.shape != reference_value.shape:
                raise ValueError(
                    f"Successive {name!r} observables must match in shape."
                )
            difference = candidate_value - reference_value
            if name == "rotation":
                difference = (difference + 180.0) % 360.0 - 180.0
            uncertainty = uncertainty_by_name[name]
            if (
                uncertainty.size == 0
                or not np.all(np.isfinite(uncertainty))
                or np.any(uncertainty <= 0.0)
            ):
                raise ValueError(
                    f"Standard errors for observable {name!r} must be finite "
                    "and positive."
                )
            try:
                standardized = np.asarray(difference / uncertainty, dtype=float)
            except ValueError as exc:
                raise ValueError(
                    f"Standard errors for observable {name!r} are not "
                    "broadcast-compatible."
                ) from exc
            observable_residuals[name] = float(np.max(np.abs(standardized)))

        masks_match = bool(
            np.array_equal(
                current.identifiable_parameter_mask,
                previous.identifiable_parameter_mask,
            )
        )
        if masks_match:
            mask = np.asarray(current.identifiable_parameter_mask, dtype=bool)
            scale = np.minimum(
                current.fit_standard_errors[mask],
                previous.fit_standard_errors[mask],
            )
            parameter_change = float(
                np.max(
                    np.abs(
                        current.fit_parameters[mask] - previous.fit_parameters[mask]
                    )
                    / scale
                )
            )
        else:
            # A changing identifiable subspace cannot establish convergence.
            parameter_change = float("inf")

        forward = validate_forward_models(
            current.detector_intensity_yx,
            previous.detector_intensity_yx,
            detector_noise_sigma_yx,
            candidate_sample_field_yx=current.sample_field_yx,
            reference_sample_field_yx=previous.sample_field_yx,
            sample_field_noise_sigma=sample_field_noise_sigma,
            observable_residuals_in_sigma=observable_residuals,
            parameter_change_in_sigma=parameter_change,
            require_complete=True,
            thresholds=thresholds,
        )
        topology_progressed = bool(
            current.topology_id != previous.topology_id
            and current.beam_count > previous.beam_count
        )
        # The convergence contract is strictly below 0.1 sigma, even though
        # the generic single-comparison gate historically accepts equality.
        parameter_stability_passed = bool(parameter_change < parameter_threshold)
        step_passed = bool(
            topology_progressed
            and masks_match
            and parameter_stability_passed
            and forward.passed
        )
        step_reports.append(
            GaussianRefinementStepReport(
                previous_topology_id=previous.topology_id,
                current_topology_id=current.topology_id,
                previous_beam_count=previous.beam_count,
                current_beam_count=current.beam_count,
                topology_progressed=topology_progressed,
                identifiable_parameter_set_stable=masks_match,
                parameter_stability_passed=parameter_stability_passed,
                forward_validation=forward,
                passed=step_passed,
            )
        )

    trailing = 0
    for step in reversed(step_reports):
        if not step.passed:
            break
        trailing += 1
    passed = bool(trailing >= required)

    failure_reasons: list[str] = []
    if not passed:
        if len(step_reports) < required:
            failure_reasons.append("insufficient_refinement_history")
        relevant = step_reports[-required:]
        if any(not step.topology_progressed for step in relevant):
            failure_reasons.append("topology_progression")
        if any(not step.identifiable_parameter_set_stable for step in relevant):
            failure_reasons.append("identifiable_parameter_set_changed")
        if any(not step.parameter_stability_passed for step in relevant):
            failure_reasons.append("parameter_stability")
        if any(not step.forward_validation.passed for step in relevant):
            failure_reasons.append("forward_validation")
        for step in relevant:
            failure_reasons.extend(
                f"missing:{gate}" for gate in step.forward_validation.missing_gates
            )
        # Keep the report deterministic without repeating a shared cause from
        # both of the final transitions.
        failure_reasons = list(dict.fromkeys(failure_reasons))

    return GaussianRefinementAdmissionReport(
        steps=tuple(step_reports),
        required_successive_refinements=required,
        trailing_successful_refinements=trailing,
        failure_reasons=tuple(failure_reasons),
        passed=passed,
    )


__all__ = [
    "DetectorGeometry",
    "HardEdgeLensInitial",
    "HardEdgeColumnParameterization",
    "MagnificationDACDesign",
    "MagnificationDACInfeasibleError",
    "SignedObjectDefocusDesign",
    "SignedObjectDefocusInfeasibleError",
    "design_nominal_magnification_dacs",
    "design_signed_object_defocus_dacs",
    "hard_edge_column_from_config",
    "frozen_topology_fingerprint",
    "FrameRequest",
    "build_acquisition_manifest",
    "RefinementRegion",
    "GaussianTopology",
    "GaussianWorkEstimate",
    "build_multiresolution_gaussian_topology",
    "TopologyRefinement",
    "refine_gaussian_topology",
    "estimate_gaussian_workload",
    "gaussian_bundle_from_sample",
    "gaussian_cardinal_coefficients",
    "reexpand_complex_field",
    "reexpand_transmitted_field",
    "SampleInteractionResult",
    "interact_gaussian_with_sample",
    "transfer_blocks",
    "apply_transfer_matrix_to_gaussians",
    "GaussianSliceSafetyReport",
    "assess_gaussian_slice_safety",
    "propagate_gaussians_through_column",
    "GaussianRenderResult",
    "ConjugateGridReferenceResult",
    "conjugate_gaussian_grid_reference",
    "render_gaussian_detector_plane",
    "render_gaussian_detector",
    "simpson_weights",
    "collins_propagate_2d",
    "ImageObservables",
    "FourierAffineCalibration",
    "fit_fourier_affine",
    "extract_image_observables",
    "FramePrediction",
    "CheckpointCompatibilityError",
    "IncompleteCheckpointError",
    "CheckpointedFrameRecord",
    "CheckpointedAcquisition",
    "render_frame_sequence_checkpointed",
    "load_completed_frame_sequence_checkpoint",
    "calibrated_state_centroid_covariance_px2",
    "predict_frame",
    "simulate_detector_counts",
    "DetectorNoiseModel",
    "FourierObservableCovarianceAugmentation",
    "KnownFourierFeatureSet",
    "known_fourier_features_from_sample_view",
    "NoisyDetectorFrame",
    "CheckpointObservableDataset",
    "build_heterogeneous_dataset_from_checkpoint",
    "aggregate_checkpoint_observables_for_fit",
    "shared_fourier_calibration_priors_from_dataset",
    "strip_shared_fourier_calibration_covariance",
    "ForwardValidationThresholds",
    "ForwardValidationReport",
    "validate_forward_models",
    "GaussianRefinementSnapshot",
    "GaussianRefinementStepReport",
    "GaussianRefinementAdmissionReport",
    "validate_gaussian_refinement_sequence",
    "HeldOutAcceptanceThresholds",
    "HeldOutMetrics",
    "evaluate_held_out_metrics",
]
