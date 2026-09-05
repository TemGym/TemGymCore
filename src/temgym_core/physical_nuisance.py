"""Shared latent-nuisance likelihoods for physical microscope inversion.

The heterogeneous image dataset stores positive-definite covariance blocks for
detector/extractor noise.  Calibration errors are different: a specimen
spacing, detector metric, voltage, or calibrated input state can affect many
frames coherently and must not be counted as fresh independent noise in every
block.  This module represents those effects by standard-normal latent
coordinates and leaves the physical transformation to a caller-supplied JAX
prediction callback.

For optical parameters ``theta``, calibration latents ``z``, and signed model
discrepancy amplitudes ``a``, the residual is

``[L_stat^-1 (prediction(theta, eta0 + L_eta z) + U a - y), z, a]``.

The implementation is intentionally generic.  It does not decide whether a
calibration value is a detector scale, a reciprocal-lattice vector, a sample
state, or a wavelength.  Typed physical transformations can therefore be
added without changing the statistical contract.

Limitations
-----------
Gaussian factors describe local calibration uncertainty; bounded or
multimodal calibrations still require explicit profiling.  The rank report is
local to one reference point and does not prove global uniqueness.  A signed
Gaussian/grid discrepancy mode covers numerical forward error only; it does
not cover hard-edge-versus-real-field model inadequacy.  Finally, callers must
keep shared systematic covariance out of ``FrameObservationBlock.covariance``
when representing the same uncertainty as a latent here, or it will be counted
twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, TypeAlias

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from .microscope_inversion import HeterogeneousInversionDataset


ArrayLike: TypeAlias = Any
CalibrationValues: TypeAlias = tuple[jnp.ndarray, ...]
PredictionCallback: TypeAlias = Callable[
    [jnp.ndarray, CalibrationValues], Sequence[ArrayLike]
]


def _nonempty(value: object, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


def _unique_names(
    values: Sequence[str], name: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    result = tuple(_nonempty(value, name) for value in values)
    if (not allow_empty and not result) or len(result) != len(set(result)):
        qualifier = "unique" if allow_empty else "non-empty and unique"
        raise ValueError(f"{name} must be {qualifier}.")
    return result


def _immutable_array(
    values: object,
    name: str,
    *,
    ndim: int | None = None,
    dtype: np.dtype | type = float,
) -> np.ndarray:
    result = np.asarray(values, dtype=dtype)
    if (ndim is not None and result.ndim != ndim) or not np.all(np.isfinite(result)):
        shape = "" if ndim is None else f" with {ndim} dimensions"
        raise ValueError(f"{name} must be a finite array{shape}.")
    result = result.copy()
    result.setflags(write=False)
    return result


def _immutable_bool_array(values: object, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=bool)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    result = result.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class GaussianPriorFactor:
    """One shared calibrated quantity expressed as ``mean + factor @ z``.

    ``z`` has an independent standard-normal prior.  A rectangular factor is
    permitted, so exactly known directions and singular calibration
    covariances do not need artificial jitter.
    """

    group_name: str
    value_names: tuple[str, ...]
    mean: np.ndarray
    factor: np.ndarray
    latent_names: tuple[str, ...]
    provenance: str

    def __post_init__(self) -> None:
        group = _nonempty(self.group_name, "group_name")
        value_names = _unique_names(self.value_names, "value_names")
        latent_names = _unique_names(
            self.latent_names, "latent_names", allow_empty=True
        )
        mean = _immutable_array(self.mean, "mean", ndim=1)
        factor = _immutable_array(self.factor, "factor", ndim=2)
        if mean.shape != (len(value_names),):
            raise ValueError("mean must contain one entry per value name.")
        if factor.shape != (len(value_names), len(latent_names)):
            raise ValueError(
                "factor must have shape (len(value_names), len(latent_names))."
            )
        if factor.shape[1] and np.any(np.linalg.norm(factor, axis=0) <= 0.0):
            raise ValueError("Every latent factor column must be non-zero.")
        object.__setattr__(self, "group_name", group)
        object.__setattr__(self, "value_names", value_names)
        object.__setattr__(self, "latent_names", latent_names)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "factor", factor)
        object.__setattr__(self, "provenance", _nonempty(self.provenance, "provenance"))

    @property
    def n_values(self) -> int:
        return len(self.value_names)

    @property
    def n_latents(self) -> int:
        return len(self.latent_names)

    @property
    def covariance(self) -> np.ndarray:
        covariance = self.factor @ self.factor.T
        covariance.setflags(write=False)
        return covariance

    def values(self, standard_coordinates: ArrayLike) -> jnp.ndarray:
        coordinates = jnp.asarray(standard_coordinates)
        if coordinates.shape != (self.n_latents,):
            raise ValueError(
                f"standard_coordinates must have shape ({self.n_latents},)."
            )
        return jnp.asarray(self.mean) + jnp.asarray(self.factor) @ coordinates

    @classmethod
    def from_covariance(
        cls,
        group_name: str,
        value_names: Sequence[str],
        mean: ArrayLike,
        covariance: ArrayLike,
        *,
        provenance: str,
        eigenvalue_rtol: float = 1e-12,
    ) -> "GaussianPriorFactor":
        """Factor a positive-semidefinite covariance without adding jitter."""

        names = tuple(value_names)
        covariance_array = np.asarray(covariance, dtype=float)
        expected = (len(names), len(names))
        if covariance_array.shape != expected or not np.all(
            np.isfinite(covariance_array)
        ):
            raise ValueError(f"covariance must be a finite array with shape {expected}.")
        if not np.isfinite(eigenvalue_rtol) or not 0.0 <= eigenvalue_rtol < 1.0:
            raise ValueError("eigenvalue_rtol must lie in [0, 1).")
        symmetric = 0.5 * (covariance_array + covariance_array.T)
        scale = max(
            float(np.max(np.abs(symmetric), initial=0.0)), np.finfo(float).tiny
        )
        if (
            np.max(np.abs(covariance_array - covariance_array.T), initial=0.0)
            > 1e-10 * scale
        ):
            raise ValueError("covariance must be symmetric.")
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        if eigenvalues.min(initial=0.0) < -1e-10 * scale:
            raise ValueError("covariance must be positive semidefinite.")
        largest = max(float(eigenvalues.max(initial=0.0)), 0.0)
        keep = (
            eigenvalues > eigenvalue_rtol * largest
            if largest
            else np.zeros_like(eigenvalues, dtype=bool)
        )
        retained_values = eigenvalues[keep]
        retained_vectors = eigenvectors[:, keep]
        # Descending modes give a stable importance order for diagnostics.
        order = np.argsort(retained_values)[::-1]
        retained_values = retained_values[order]
        retained_vectors = retained_vectors[:, order]
        factor = retained_vectors * np.sqrt(np.maximum(retained_values, 0.0))
        latent_names = tuple(
            f"{_nonempty(group_name, 'group_name')}.mode_{index}"
            for index in range(factor.shape[1])
        )
        return cls(
            group_name=group_name,
            value_names=names,
            mean=np.asarray(mean, dtype=float),
            factor=factor,
            latent_names=latent_names,
            provenance=provenance,
        )


@dataclass(frozen=True)
class SignedDiscrepancyModes:
    """Low-rank signed observable discrepancies shared across exposures.

    Columns of ``factor`` are in observable units and multiply independent
    standard-normal amplitudes.  Keeping the sign and global row alignment
    allows a coherent numerical bias to remain coherent instead of shrinking
    as if it were independent detector noise.
    """

    observation_keys: tuple[str, ...]
    mode_names: tuple[str, ...]
    factor: np.ndarray
    provenance: str

    def __post_init__(self) -> None:
        keys = _unique_names(self.observation_keys, "observation_keys")
        names = _unique_names(self.mode_names, "mode_names")
        factor = _immutable_array(self.factor, "factor", ndim=2)
        if factor.shape != (len(keys), len(names)):
            raise ValueError(
                "factor must have shape (len(observation_keys), len(mode_names))."
            )
        if np.any(np.linalg.norm(factor, axis=0) <= 0.0):
            raise ValueError("Every discrepancy mode must be non-zero.")
        object.__setattr__(self, "observation_keys", keys)
        object.__setattr__(self, "mode_names", names)
        object.__setattr__(self, "factor", factor)
        object.__setattr__(self, "provenance", _nonempty(self.provenance, "provenance"))

    @property
    def n_modes(self) -> int:
        return len(self.mode_names)

    @property
    def covariance(self) -> np.ndarray:
        covariance = self.factor @ self.factor.T
        covariance.setflags(write=False)
        return covariance

    @classmethod
    def from_columns(
        cls,
        observation_keys: Sequence[str],
        signed_modes: Mapping[str, ArrayLike],
        *,
        provenance: str,
    ) -> "SignedDiscrepancyModes":
        names = tuple(signed_modes)
        if not names:
            raise ValueError("signed_modes must contain at least one mode.")
        columns = tuple(np.asarray(signed_modes[name], dtype=float) for name in names)
        return cls(
            observation_keys=tuple(observation_keys),
            mode_names=names,
            factor=np.stack(columns, axis=1),
            provenance=provenance,
        )


@dataclass(frozen=True)
class GlobalObservationLayout:
    """Immutable global ordering and statistical whitener for frame blocks."""

    n_dataset_blocks: int
    block_sizes: tuple[int, ...]
    selected_block_indices: tuple[int, ...]
    all_observation_keys: tuple[str, ...]
    observation_keys: tuple[str, ...]
    values: np.ndarray
    statistical_covariance: np.ndarray
    statistical_cholesky: np.ndarray
    held_out_observation_mask: np.ndarray
    included_observation_mask: np.ndarray
    include_held_out: bool = False

    def __post_init__(self) -> None:
        if self.n_dataset_blocks < 1:
            raise ValueError("n_dataset_blocks must be positive.")
        sizes = tuple(int(value) for value in self.block_sizes)
        if len(sizes) != self.n_dataset_blocks or min(sizes, default=0) < 1:
            raise ValueError("block_sizes must contain one positive size per block.")
        selected = tuple(int(value) for value in self.selected_block_indices)
        if (
            not selected
            or len(selected) != len(set(selected))
            or any(value < 0 or value >= self.n_dataset_blocks for value in selected)
            or any(right <= left for left, right in zip(selected, selected[1:]))
        ):
            raise ValueError("selected_block_indices must be unique, ordered, and valid.")
        all_keys = _unique_names(self.all_observation_keys, "all_observation_keys")
        keys = _unique_names(self.observation_keys, "observation_keys")
        if len(all_keys) != sum(sizes):
            raise ValueError("all_observation_keys does not match block_sizes.")
        values = _immutable_array(self.values, "values", ndim=1)
        covariance = _immutable_array(
            self.statistical_covariance, "statistical_covariance", ndim=2
        )
        cholesky = _immutable_array(
            self.statistical_cholesky, "statistical_cholesky", ndim=2
        )
        count = len(keys)
        if values.shape != (count,) or covariance.shape != (count, count):
            raise ValueError("Selected values/covariance must match observation_keys.")
        if cholesky.shape != covariance.shape or not np.allclose(
            cholesky @ cholesky.T, covariance, rtol=1e-10, atol=1e-14
        ):
            raise ValueError("statistical_cholesky must factor statistical_covariance.")
        if not np.allclose(cholesky, np.tril(cholesky), rtol=0.0, atol=1e-14):
            raise ValueError("statistical_cholesky must be lower triangular.")
        if np.any(np.diag(cholesky) <= 0.0):
            raise ValueError("statistical_cholesky must have a positive diagonal.")
        held_out = _immutable_bool_array(
            self.held_out_observation_mask, "held_out_observation_mask"
        )
        included = _immutable_bool_array(
            self.included_observation_mask, "included_observation_mask"
        )
        if held_out.shape != (len(all_keys),) or included.shape != held_out.shape:
            raise ValueError("Observation masks must match all_observation_keys.")
        if bool(self.include_held_out):
            if not np.all(included):
                raise ValueError("include_held_out=True must include every observation.")
        elif np.any(included & held_out) or not np.array_equal(included, ~held_out):
            raise ValueError(
                "Training layout must exclude exactly the held-out observations."
            )
        object.__setattr__(self, "block_sizes", sizes)
        object.__setattr__(self, "selected_block_indices", selected)
        object.__setattr__(self, "all_observation_keys", all_keys)
        object.__setattr__(self, "observation_keys", keys)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "statistical_covariance", covariance)
        object.__setattr__(self, "statistical_cholesky", cholesky)
        object.__setattr__(self, "held_out_observation_mask", held_out)
        object.__setattr__(self, "included_observation_mask", included)
        object.__setattr__(self, "include_held_out", bool(self.include_held_out))

    @property
    def n_observations(self) -> int:
        return len(self.observation_keys)

    @classmethod
    def from_dataset(
        cls,
        dataset: HeterogeneousInversionDataset,
        *,
        include_held_out: bool = False,
    ) -> "GlobalObservationLayout":
        if not isinstance(dataset, HeterogeneousInversionDataset):
            raise TypeError("dataset must be HeterogeneousInversionDataset.")
        block_sizes = tuple(block.values.size for block in dataset.blocks)
        all_keys: list[str] = []
        all_held_out: list[bool] = []
        selected_indices: list[int] = []
        selected_values: list[np.ndarray] = []
        selected_covariances: list[np.ndarray] = []
        selected_keys: list[str] = []
        for index, block in enumerate(dataset.blocks):
            held_out = dataset.plan.states[block.optical_state_index].role == "held_out"
            block_keys = tuple(
                f"frame_{block.frame_id}.{name}" for name in block.observable_names
            )
            all_keys.extend(block_keys)
            all_held_out.extend((held_out,) * block.values.size)
            if include_held_out or not held_out:
                selected_indices.append(index)
                selected_values.append(np.asarray(block.values, dtype=float))
                selected_covariances.append(np.asarray(block.covariance, dtype=float))
                selected_keys.extend(block_keys)
        if not selected_indices:
            raise ValueError("The requested layout contains no observations.")
        count = sum(value.size for value in selected_values)
        covariance = np.zeros((count, count), dtype=float)
        cursor = 0
        for block_covariance in selected_covariances:
            stop = cursor + block_covariance.shape[0]
            covariance[cursor:stop, cursor:stop] = block_covariance
            cursor = stop
        held_out_mask = np.asarray(all_held_out, dtype=bool)
        included_mask = (
            np.ones_like(held_out_mask)
            if include_held_out
            else ~held_out_mask
        )
        return cls(
            n_dataset_blocks=len(dataset.blocks),
            block_sizes=block_sizes,
            selected_block_indices=tuple(selected_indices),
            all_observation_keys=tuple(all_keys),
            observation_keys=tuple(selected_keys),
            values=np.concatenate(selected_values),
            statistical_covariance=covariance,
            statistical_cholesky=np.linalg.cholesky(covariance),
            held_out_observation_mask=held_out_mask,
            included_observation_mask=included_mask,
            include_held_out=include_held_out,
        )

    def flatten_predictions(
        self, predicted_blocks: Sequence[ArrayLike]
    ) -> jnp.ndarray:
        """Flatten predictions in the source dataset's exact block/name order."""

        blocks = tuple(predicted_blocks)
        if len(blocks) != self.n_dataset_blocks:
            raise ValueError(
                "prediction callback must return one array per source dataset block."
            )
        selected: list[jnp.ndarray] = []
        selected_set = set(self.selected_block_indices)
        for index, (prediction, size) in enumerate(
            zip(blocks, self.block_sizes, strict=True)
        ):
            array = jnp.ravel(jnp.asarray(prediction))
            if array.shape != (size,):
                raise ValueError(
                    f"Prediction block {index} has shape {array.shape}; expected ({size},)."
                )
            if index in selected_set:
                selected.append(array)
        return jnp.concatenate(selected)


@dataclass(frozen=True)
class SharedLatentNuisanceLikelihood:
    """JAX-compatible joint likelihood with shared calibration/error modes."""

    layout: GlobalObservationLayout
    optical_parameter_count: int
    prediction_callback: PredictionCallback
    calibration_priors: tuple[GaussianPriorFactor, ...] = ()
    discrepancy_modes: SignedDiscrepancyModes | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.layout, GlobalObservationLayout):
            raise TypeError("layout must be GlobalObservationLayout.")
        count = int(self.optical_parameter_count)
        if count < 1 or count != self.optical_parameter_count:
            raise ValueError("optical_parameter_count must be a positive integer.")
        if not callable(self.prediction_callback):
            raise TypeError("prediction_callback must be callable.")
        priors = tuple(self.calibration_priors)
        if any(not isinstance(prior, GaussianPriorFactor) for prior in priors):
            raise TypeError("calibration_priors must contain GaussianPriorFactor values.")
        groups = tuple(prior.group_name for prior in priors)
        if len(groups) != len(set(groups)):
            raise ValueError("Calibration prior group names must be unique.")
        latent_names = tuple(
            f"{prior.group_name}:{name}"
            for prior in priors
            for name in prior.latent_names
        )
        if len(latent_names) != len(set(latent_names)):
            raise ValueError("Calibration latent names must be globally unique.")
        discrepancy = self.discrepancy_modes
        if discrepancy is not None:
            if not isinstance(discrepancy, SignedDiscrepancyModes):
                raise TypeError("discrepancy_modes must be SignedDiscrepancyModes.")
            if discrepancy.observation_keys != self.layout.observation_keys:
                raise ValueError(
                    "Discrepancy observation keys must exactly match the selected layout."
                )
        object.__setattr__(self, "optical_parameter_count", count)
        object.__setattr__(self, "calibration_priors", priors)

    @property
    def n_calibration_latents(self) -> int:
        return sum(prior.n_latents for prior in self.calibration_priors)

    @property
    def n_discrepancy_latents(self) -> int:
        return 0 if self.discrepancy_modes is None else self.discrepancy_modes.n_modes

    @property
    def n_nuisance_latents(self) -> int:
        return self.n_calibration_latents + self.n_discrepancy_latents

    @property
    def n_joint_parameters(self) -> int:
        return self.optical_parameter_count + self.n_nuisance_latents

    @property
    def calibration_latent_names(self) -> tuple[str, ...]:
        return tuple(
            f"{prior.group_name}:{name}"
            for prior in self.calibration_priors
            for name in prior.latent_names
        )

    @property
    def discrepancy_latent_names(self) -> tuple[str, ...]:
        return (
            ()
            if self.discrepancy_modes is None
            else self.discrepancy_modes.mode_names
        )

    def split_joint_parameters(
        self, joint_parameters: ArrayLike
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        values = jnp.asarray(joint_parameters)
        if values.shape != (self.n_joint_parameters,):
            raise ValueError(
                f"joint_parameters must have shape ({self.n_joint_parameters},)."
            )
        first = self.optical_parameter_count
        second = first + self.n_calibration_latents
        return values[:first], values[first:second], values[second:]

    def join_parameters(
        self,
        optical_parameters: ArrayLike,
        calibration_coordinates: ArrayLike | None = None,
        discrepancy_coordinates: ArrayLike | None = None,
    ) -> jnp.ndarray:
        optical = jnp.asarray(optical_parameters)
        if optical.shape != (self.optical_parameter_count,):
            raise ValueError(
                f"optical_parameters must have shape ({self.optical_parameter_count},)."
            )
        calibration = (
            jnp.zeros((self.n_calibration_latents,), dtype=optical.dtype)
            if calibration_coordinates is None
            else jnp.asarray(calibration_coordinates)
        )
        discrepancy = (
            jnp.zeros((self.n_discrepancy_latents,), dtype=optical.dtype)
            if discrepancy_coordinates is None
            else jnp.asarray(discrepancy_coordinates)
        )
        if calibration.shape != (self.n_calibration_latents,):
            raise ValueError("calibration_coordinates has the wrong shape.")
        if discrepancy.shape != (self.n_discrepancy_latents,):
            raise ValueError("discrepancy_coordinates has the wrong shape.")
        return jnp.concatenate((optical, calibration, discrepancy))

    def calibration_values(self, calibration_coordinates: ArrayLike) -> CalibrationValues:
        coordinates = jnp.asarray(calibration_coordinates)
        if coordinates.shape != (self.n_calibration_latents,):
            raise ValueError("calibration_coordinates has the wrong shape.")
        values: list[jnp.ndarray] = []
        cursor = 0
        for prior in self.calibration_priors:
            stop = cursor + prior.n_latents
            values.append(prior.values(coordinates[cursor:stop]))
            cursor = stop
        return tuple(values)

    def predicted_observations(
        self,
        optical_parameters: ArrayLike,
        calibration_coordinates: ArrayLike,
        discrepancy_coordinates: ArrayLike,
    ) -> jnp.ndarray:
        optical = jnp.asarray(optical_parameters)
        calibration = jnp.asarray(calibration_coordinates)
        discrepancy = jnp.asarray(discrepancy_coordinates)
        if optical.shape != (self.optical_parameter_count,):
            raise ValueError("optical_parameters has the wrong shape.")
        if calibration.shape != (self.n_calibration_latents,):
            raise ValueError("calibration_coordinates has the wrong shape.")
        if discrepancy.shape != (self.n_discrepancy_latents,):
            raise ValueError("discrepancy_coordinates has the wrong shape.")
        blocks = self.prediction_callback(
            optical, self.calibration_values(calibration)
        )
        prediction = self.layout.flatten_predictions(blocks)
        if self.discrepancy_modes is not None:
            prediction = (
                prediction
                + jnp.asarray(self.discrepancy_modes.factor) @ discrepancy
            )
        return prediction

    def data_residual_from_parts(
        self,
        optical_parameters: ArrayLike,
        calibration_coordinates: ArrayLike,
        discrepancy_coordinates: ArrayLike,
    ) -> jnp.ndarray:
        prediction = self.predicted_observations(
            optical_parameters,
            calibration_coordinates,
            discrepancy_coordinates,
        )
        difference = prediction - jnp.asarray(self.layout.values)
        return jsp.linalg.solve_triangular(
            jnp.asarray(self.layout.statistical_cholesky),
            difference,
            lower=True,
        )

    def joint_residual_from_parts(
        self,
        optical_parameters: ArrayLike,
        calibration_coordinates: ArrayLike,
        discrepancy_coordinates: ArrayLike,
    ) -> jnp.ndarray:
        calibration = jnp.asarray(calibration_coordinates)
        discrepancy = jnp.asarray(discrepancy_coordinates)
        data = self.data_residual_from_parts(
            optical_parameters, calibration, discrepancy
        )
        return jnp.concatenate((data, calibration, discrepancy))

    def data_residual(self, joint_parameters: ArrayLike) -> jnp.ndarray:
        optical, calibration, discrepancy = self.split_joint_parameters(joint_parameters)
        return self.data_residual_from_parts(optical, calibration, discrepancy)

    def joint_residual(self, joint_parameters: ArrayLike) -> jnp.ndarray:
        optical, calibration, discrepancy = self.split_joint_parameters(joint_parameters)
        return self.joint_residual_from_parts(optical, calibration, discrepancy)


@dataclass(frozen=True)
class HeldOutBlockEvaluation:
    """Data-only prediction diagnostics for one excluded detector frame.

    The residuals are evaluated after a training fit and are conditional on
    its fitted optical and shared-calibration coordinates.  No prior residual
    is appended, and no held-out value is used to update those coordinates.
    """

    frame_id: int
    optical_state_index: int
    optical_state_label: str
    observable_names: tuple[str, ...]
    observed: np.ndarray
    predicted: np.ndarray
    residual: np.ndarray
    whitened_residual: np.ndarray
    chi_square: float
    weighted_rms_sigma: float
    maximum_absolute_whitened_residual: float

    def __post_init__(self) -> None:
        if self.frame_id < 0 or self.optical_state_index < 0:
            raise ValueError("Frame and optical-state indices must be non-negative.")
        label = _nonempty(self.optical_state_label, "optical_state_label")
        names = _unique_names(self.observable_names, "observable_names")
        arrays = tuple(
            _immutable_array(getattr(self, name), name, ndim=1)
            for name in ("observed", "predicted", "residual", "whitened_residual")
        )
        if any(array.shape != (len(names),) for array in arrays):
            raise ValueError(
                "Held-out vectors must have one entry per observable name."
            )
        observed, predicted, residual, whitened = arrays
        if not np.allclose(residual, predicted - observed, rtol=1e-12, atol=1e-14):
            raise ValueError("residual must equal predicted minus observed.")
        chi_square = float(self.chi_square)
        weighted_rms = float(self.weighted_rms_sigma)
        maximum = float(self.maximum_absolute_whitened_residual)
        expected_chi_square = float(whitened @ whitened)
        if (
            not np.isfinite((chi_square, weighted_rms, maximum)).all()
            or chi_square < 0.0
            or weighted_rms < 0.0
            or maximum < 0.0
            or not np.isclose(chi_square, expected_chi_square, rtol=1e-12, atol=1e-14)
            or not np.isclose(
                weighted_rms,
                np.sqrt(expected_chi_square / len(names)),
                rtol=1e-12,
                atol=1e-14,
            )
            or not np.isclose(
                maximum,
                np.max(np.abs(whitened)),
                rtol=1e-12,
                atol=1e-14,
            )
        ):
            raise ValueError("Held-out scalar diagnostics are inconsistent.")
        object.__setattr__(self, "optical_state_label", label)
        object.__setattr__(self, "observable_names", names)
        for name, array in zip(
            ("observed", "predicted", "residual", "whitened_residual"),
            arrays,
            strict=True,
        ):
            object.__setattr__(self, name, array)
        object.__setattr__(self, "chi_square", chi_square)
        object.__setattr__(self, "weighted_rms_sigma", weighted_rms)
        object.__setattr__(
            self, "maximum_absolute_whitened_residual", maximum
        )


@dataclass(frozen=True)
class HeldOutLikelihoodEvaluation:
    """Covariance-aware evaluation on the exact complement of training blocks."""

    blocks: tuple[HeldOutBlockEvaluation, ...]
    training_block_indices: tuple[int, ...]
    held_out_block_indices: tuple[int, ...]
    whitened_residual: np.ndarray
    chi_square: float
    weighted_rms_sigma: float
    maximum_absolute_whitened_residual: float
    calibration_group_names: tuple[str, ...]
    provenance: str = (
        "held-out data-only evaluation conditional on training-fit optical and "
        "shared-calibration coordinates"
    )

    def __post_init__(self) -> None:
        blocks = tuple(self.blocks)
        if not blocks or any(
            not isinstance(block, HeldOutBlockEvaluation) for block in blocks
        ):
            raise TypeError(
                "blocks must contain at least one HeldOutBlockEvaluation."
            )
        training = tuple(int(index) for index in self.training_block_indices)
        held_out = tuple(int(index) for index in self.held_out_block_indices)
        if (
            not training
            or not held_out
            or len(training) != len(set(training))
            or len(held_out) != len(set(held_out))
            or set(training) & set(held_out)
            or min((*training, *held_out)) < 0
        ):
            raise ValueError(
                "Training and held-out block indices must be non-empty, unique, "
                "non-negative, and disjoint."
            )
        groups = _unique_names(
            self.calibration_group_names, "calibration_group_names"
        ) if self.calibration_group_names else ()
        whitened = _immutable_array(
            self.whitened_residual, "whitened_residual", ndim=1
        )
        expected = np.concatenate(
            tuple(block.whitened_residual for block in blocks)
        )
        if whitened.shape != expected.shape or not np.allclose(
            whitened, expected, rtol=1e-12, atol=1e-14
        ):
            raise ValueError(
                "Aggregate whitened_residual must concatenate the block residuals."
            )
        chi_square = float(self.chi_square)
        weighted_rms = float(self.weighted_rms_sigma)
        maximum = float(self.maximum_absolute_whitened_residual)
        expected_chi_square = float(whitened @ whitened)
        if (
            not np.isfinite((chi_square, weighted_rms, maximum)).all()
            or not np.isclose(chi_square, expected_chi_square, rtol=1e-12, atol=1e-14)
            or not np.isclose(
                weighted_rms,
                np.sqrt(expected_chi_square / whitened.size),
                rtol=1e-12,
                atol=1e-14,
            )
            or not np.isclose(
                maximum,
                np.max(np.abs(whitened)),
                rtol=1e-12,
                atol=1e-14,
            )
        ):
            raise ValueError("Aggregate held-out scalar diagnostics are inconsistent.")
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "training_block_indices", training)
        object.__setattr__(self, "held_out_block_indices", held_out)
        object.__setattr__(self, "whitened_residual", whitened)
        object.__setattr__(self, "chi_square", chi_square)
        object.__setattr__(self, "weighted_rms_sigma", weighted_rms)
        object.__setattr__(
            self, "maximum_absolute_whitened_residual", maximum
        )
        object.__setattr__(self, "calibration_group_names", groups)
        object.__setattr__(self, "provenance", _nonempty(self.provenance, "provenance"))

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    @property
    def n_observations(self) -> int:
        return self.whitened_residual.size


def evaluate_held_out_shared_likelihood(
    likelihood: SharedLatentNuisanceLikelihood,
    dataset: HeterogeneousInversionDataset,
    fitted_joint_parameters: ArrayLike,
) -> HeldOutLikelihoodEvaluation:
    """Evaluate a training-only shared likelihood on excluded physical blocks.

    This function is deliberately strict about leakage.  ``likelihood.layout``
    must select exactly every non-held-out block of ``dataset`` and no held-out
    block.  Its callback must still return predictions in the complete dataset
    order, allowing the held-out complement to be evaluated only after fitting.

    Fitted shared-calibration values are carried into the prediction.  Signed
    discrepancy modes are rejected because their factors are defined only on
    training observation keys; extrapolating them to held-out frames requires
    a separately declared physical callback rather than an implicit zero.
    """

    if not isinstance(likelihood, SharedLatentNuisanceLikelihood):
        raise TypeError("likelihood must be SharedLatentNuisanceLikelihood.")
    if not isinstance(dataset, HeterogeneousInversionDataset):
        raise TypeError("dataset must be HeterogeneousInversionDataset.")
    if likelihood.discrepancy_modes is not None:
        raise ValueError(
            "Training discrepancy modes cannot be extrapolated to held-out "
            "frames without an explicit held-out discrepancy model."
        )

    training_indices = tuple(
        index
        for index, block in enumerate(dataset.blocks)
        if dataset.plan.states[block.optical_state_index].role != "held_out"
    )
    held_out_indices = tuple(
        index
        for index, block in enumerate(dataset.blocks)
        if dataset.plan.states[block.optical_state_index].role == "held_out"
    )
    if not training_indices or not held_out_indices:
        raise ValueError(
            "dataset must contain both training and held-out observation blocks."
        )
    expected_layout = GlobalObservationLayout.from_dataset(dataset)
    layout = likelihood.layout
    if (
        layout.include_held_out
        or layout.n_dataset_blocks != len(dataset.blocks)
        or layout.selected_block_indices != training_indices
        or layout.all_observation_keys != expected_layout.all_observation_keys
        or layout.observation_keys != expected_layout.observation_keys
        or not np.array_equal(
            layout.included_observation_mask,
            expected_layout.included_observation_mask,
        )
        or not np.array_equal(
            layout.held_out_observation_mask,
            expected_layout.held_out_observation_mask,
        )
        or not np.array_equal(layout.values, expected_layout.values)
        or not np.array_equal(
            layout.statistical_covariance,
            expected_layout.statistical_covariance,
        )
    ):
        raise ValueError(
            "likelihood.layout must be the exact training-only layout of dataset."
        )

    optical, calibration, _discrepancy = likelihood.split_joint_parameters(
        fitted_joint_parameters
    )
    calibration_values = likelihood.calibration_values(calibration)
    predictions = tuple(
        likelihood.prediction_callback(optical, calibration_values)
    )
    if len(predictions) != len(dataset.blocks):
        raise ValueError(
            "The prediction callback must return one block in complete dataset order."
        )

    reports: list[HeldOutBlockEvaluation] = []
    for index in held_out_indices:
        block = dataset.blocks[index]
        predicted = np.asarray(jax.device_get(predictions[index]), dtype=float)
        if predicted.shape != block.values.shape or not np.all(np.isfinite(predicted)):
            raise ValueError(
                f"Prediction for held-out frame {block.frame_id} must be finite "
                f"with shape {block.values.shape}."
            )
        observed = np.asarray(block.values, dtype=float)
        residual = predicted - observed
        whitened = np.linalg.solve(np.linalg.cholesky(block.covariance), residual)
        chi_square = float(whitened @ whitened)
        reports.append(
            HeldOutBlockEvaluation(
                frame_id=block.frame_id,
                optical_state_index=block.optical_state_index,
                optical_state_label=dataset.plan.states[
                    block.optical_state_index
                ].label,
                observable_names=block.observable_names,
                observed=observed,
                predicted=predicted,
                residual=residual,
                whitened_residual=whitened,
                chi_square=chi_square,
                weighted_rms_sigma=float(
                    np.sqrt(chi_square / block.values.size)
                ),
                maximum_absolute_whitened_residual=float(
                    np.max(np.abs(whitened))
                ),
            )
        )
    aggregate = np.concatenate(tuple(report.whitened_residual for report in reports))
    chi_square = float(aggregate @ aggregate)
    return HeldOutLikelihoodEvaluation(
        blocks=tuple(reports),
        training_block_indices=training_indices,
        held_out_block_indices=held_out_indices,
        whitened_residual=aggregate,
        chi_square=chi_square,
        weighted_rms_sigma=float(np.sqrt(chi_square / aggregate.size)),
        maximum_absolute_whitened_residual=float(np.max(np.abs(aggregate))),
        calibration_group_names=tuple(
            prior.group_name for prior in likelihood.calibration_priors
        ),
    )


@dataclass(frozen=True)
class RankSpectrum:
    """SVD rank summary with an explicit relative threshold."""

    singular_values: np.ndarray
    rank: int
    nullity: int
    threshold: float
    condition_number: float

    def __post_init__(self) -> None:
        singular = _immutable_array(self.singular_values, "singular_values", ndim=1)
        if np.any(singular < 0.0):
            raise ValueError("singular_values must be non-negative.")
        object.__setattr__(self, "singular_values", singular)


def _rank_spectrum(matrix: np.ndarray, *, rtol: float) -> RankSpectrum:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("Rank diagnostics require a finite matrix.")
    singular = np.linalg.svd(values, compute_uv=False)
    largest = float(singular[0]) if singular.size else 0.0
    threshold = float(rtol) * largest
    rank = int(np.sum(singular > threshold)) if largest else 0
    nullity = max(values.shape[1] - rank, 0)
    condition = (
        float("inf")
        if rank == 0 or nullity
        else largest / float(singular[rank - 1])
    )
    return RankSpectrum(singular, rank, nullity, threshold, condition)


@dataclass(frozen=True)
class BlockRankDiagnostics:
    """Rank spectra for one whitened Jacobian row block.

    ``joint`` and ``conditional_optical`` retain the caller's declared
    parameter scaling.  Their column-equilibrated counterparts divide every
    non-zero column by its Euclidean norm before the SVD.  This makes the
    structural rank visible when, for example, a near-conjugate downstream
    response is many orders of magnitude larger than a separately measured
    condenser response.

    ``profiled_optical`` removes the data-resolved nuisance column space by an
    orthogonal projection.  This is the residual-Jacobian form of the
    generalized Schur complement and avoids squaring the condition number by
    forming a Fisher matrix before the projection.
    """

    block_name: str
    n_observations: int
    joint: RankSpectrum
    column_equilibrated_joint: RankSpectrum
    conditional_optical: RankSpectrum
    column_equilibrated_conditional_optical: RankSpectrum
    nuisance: RankSpectrum
    column_equilibrated_nuisance: RankSpectrum
    profiled_optical: RankSpectrum
    column_equilibrated_profiled_optical: RankSpectrum
    joint_column_norms: np.ndarray
    joint_active_column_mask: np.ndarray
    profiled_optical_column_norms: np.ndarray
    profiled_optical_active_column_mask: np.ndarray
    profiled_optical_precision: np.ndarray

    def __post_init__(self) -> None:
        name = _nonempty(self.block_name, "block_name")
        count = int(self.n_observations)
        if count < 1 or count != self.n_observations:
            raise ValueError("n_observations must be a positive integer.")
        spectra = (
            self.joint,
            self.column_equilibrated_joint,
            self.conditional_optical,
            self.column_equilibrated_conditional_optical,
            self.nuisance,
            self.column_equilibrated_nuisance,
            self.profiled_optical,
            self.column_equilibrated_profiled_optical,
        )
        if any(not isinstance(spectrum, RankSpectrum) for spectrum in spectra):
            raise TypeError("Rank diagnostic fields must contain RankSpectrum values.")
        joint_norms = _immutable_array(
            self.joint_column_norms, "joint_column_norms", ndim=1
        )
        joint_active = _immutable_bool_array(
            self.joint_active_column_mask, "joint_active_column_mask"
        )
        profiled_norms = _immutable_array(
            self.profiled_optical_column_norms,
            "profiled_optical_column_norms",
            ndim=1,
        )
        profiled_active = _immutable_bool_array(
            self.profiled_optical_active_column_mask,
            "profiled_optical_active_column_mask",
        )
        precision = _immutable_array(
            self.profiled_optical_precision,
            "profiled_optical_precision",
            ndim=2,
        )
        if joint_active.shape != joint_norms.shape:
            raise ValueError("Joint column norms and active mask must have equal shape.")
        if profiled_active.shape != profiled_norms.shape:
            raise ValueError(
                "Profiled optical column norms and active mask must have equal shape."
            )
        if precision.shape != (profiled_norms.size, profiled_norms.size):
            raise ValueError(
                "profiled_optical_precision must be square in the optical dimension."
            )
        object.__setattr__(self, "block_name", name)
        object.__setattr__(self, "n_observations", count)
        object.__setattr__(self, "joint_column_norms", joint_norms)
        object.__setattr__(self, "joint_active_column_mask", joint_active)
        object.__setattr__(self, "profiled_optical_column_norms", profiled_norms)
        object.__setattr__(
            self, "profiled_optical_active_column_mask", profiled_active
        )
        object.__setattr__(self, "profiled_optical_precision", precision)


@dataclass(frozen=True)
class BlockwiseIdentifiabilityDiagnostics:
    """Per-block and combined data-only joint identifiability report.

    Nuisance coordinates are shared when the blocks are stacked for
    ``combined``.  Profiling each block independently is useful for finding
    which acquisition supplies a direction, but it must not be substituted
    for the combined result: a shared nuisance can be distinguishable only
    through its different response in several blocks.
    """

    block_reports: tuple[BlockRankDiagnostics, ...]
    combined: BlockRankDiagnostics
    scaled_combined_jacobian: np.ndarray
    parameter_scales: np.ndarray
    optical_parameter_count: int
    nuisance_parameter_count: int
    rank_rtol: float
    column_norm_atol: float

    def __post_init__(self) -> None:
        reports = tuple(self.block_reports)
        if not reports or any(
            not isinstance(report, BlockRankDiagnostics) for report in reports
        ):
            raise TypeError(
                "block_reports must contain at least one BlockRankDiagnostics value."
            )
        names = tuple(report.block_name for report in reports)
        if len(names) != len(set(names)):
            raise ValueError("Block names must be unique.")
        if not isinstance(self.combined, BlockRankDiagnostics):
            raise TypeError("combined must be a BlockRankDiagnostics value.")
        combined_jacobian = _immutable_array(
            self.scaled_combined_jacobian,
            "scaled_combined_jacobian",
            ndim=2,
        )
        scales = _immutable_array(self.parameter_scales, "parameter_scales", ndim=1)
        optical_count = int(self.optical_parameter_count)
        nuisance_count = int(self.nuisance_parameter_count)
        if optical_count < 1 or optical_count != self.optical_parameter_count:
            raise ValueError("optical_parameter_count must be a positive integer.")
        if nuisance_count < 0 or nuisance_count != self.nuisance_parameter_count:
            raise ValueError("nuisance_parameter_count must be a non-negative integer.")
        if scales.shape != (optical_count + nuisance_count,):
            raise ValueError("parameter_scales must match the joint parameter count.")
        if combined_jacobian.shape != (
            sum(report.n_observations for report in reports),
            scales.size,
        ):
            raise ValueError(
                "scaled_combined_jacobian shape is inconsistent with the reports."
            )
        if self.combined.n_observations != combined_jacobian.shape[0]:
            raise ValueError("The combined report has the wrong observation count.")
        if not np.isfinite(self.rank_rtol) or not 0.0 < self.rank_rtol < 1.0:
            raise ValueError("rank_rtol must lie strictly in (0, 1).")
        if not np.isfinite(self.column_norm_atol) or self.column_norm_atol < 0.0:
            raise ValueError("column_norm_atol must be finite and non-negative.")
        object.__setattr__(self, "block_reports", reports)
        object.__setattr__(self, "scaled_combined_jacobian", combined_jacobian)
        object.__setattr__(self, "parameter_scales", scales)
        object.__setattr__(self, "optical_parameter_count", optical_count)
        object.__setattr__(self, "nuisance_parameter_count", nuisance_count)

    @property
    def block_names(self) -> tuple[str, ...]:
        return tuple(report.block_name for report in self.block_reports)

    def report_for_block(self, block_name: str) -> BlockRankDiagnostics:
        """Return one named block report without exposing a mutable mapping."""

        name = _nonempty(block_name, "block_name")
        for report in self.block_reports:
            if report.block_name == name:
                return report
        raise KeyError(name)


def _column_equilibrated_matrix(
    matrix: np.ndarray,
    *,
    column_norm_atol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize informative columns while preserving explicit zero columns."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("Column equilibration requires a finite matrix.")
    norms = np.linalg.norm(values, axis=0)
    active = norms > float(column_norm_atol)
    inverse = np.zeros_like(norms)
    inverse[active] = 1.0 / norms[active]
    return values * inverse[None, :], norms, active


def _profiled_optical_jacobian(
    optical_jacobian: np.ndarray,
    nuisance_jacobian: np.ndarray,
    *,
    rank_rtol: float,
    column_norm_atol: float,
) -> np.ndarray:
    """Project optical columns off the resolved shared-nuisance column space."""

    optical = np.asarray(optical_jacobian, dtype=float)
    nuisance = np.asarray(nuisance_jacobian, dtype=float)
    if nuisance.shape[1] == 0:
        return optical.copy()
    equilibrated, _, _ = _column_equilibrated_matrix(
        nuisance, column_norm_atol=column_norm_atol
    )
    left, singular, _ = np.linalg.svd(equilibrated, full_matrices=False)
    largest = float(singular[0]) if singular.size else 0.0
    rank = int(np.sum(singular > rank_rtol * largest)) if largest else 0
    if rank == 0:
        return optical.copy()
    resolved_left = left[:, :rank]
    return optical - resolved_left @ (resolved_left.T @ optical)


def _block_rank_diagnostics(
    block_name: str,
    scaled_jacobian: np.ndarray,
    *,
    optical_parameter_count: int,
    rank_rtol: float,
    column_norm_atol: float,
) -> BlockRankDiagnostics:
    values = np.asarray(scaled_jacobian, dtype=float)
    optical = values[:, :optical_parameter_count]
    nuisance = values[:, optical_parameter_count:]
    equilibrated_joint, joint_norms, joint_active = _column_equilibrated_matrix(
        values, column_norm_atol=column_norm_atol
    )
    equilibrated_optical, _, _ = _column_equilibrated_matrix(
        optical, column_norm_atol=column_norm_atol
    )
    equilibrated_nuisance, _, _ = _column_equilibrated_matrix(
        nuisance, column_norm_atol=column_norm_atol
    )
    profiled = _profiled_optical_jacobian(
        optical,
        nuisance,
        rank_rtol=rank_rtol,
        column_norm_atol=column_norm_atol,
    )
    equilibrated_profiled, profiled_norms, profiled_active = (
        _column_equilibrated_matrix(
            profiled, column_norm_atol=column_norm_atol
        )
    )
    precision = _numerically_psd(
        profiled.T @ profiled,
        name=f"{block_name} profiled optical precision",
    )
    return BlockRankDiagnostics(
        block_name=block_name,
        n_observations=values.shape[0],
        joint=_rank_spectrum(values, rtol=rank_rtol),
        column_equilibrated_joint=_rank_spectrum(
            equilibrated_joint, rtol=rank_rtol
        ),
        conditional_optical=_rank_spectrum(optical, rtol=rank_rtol),
        column_equilibrated_conditional_optical=_rank_spectrum(
            equilibrated_optical, rtol=rank_rtol
        ),
        nuisance=_rank_spectrum(nuisance, rtol=rank_rtol),
        column_equilibrated_nuisance=_rank_spectrum(
            equilibrated_nuisance, rtol=rank_rtol
        ),
        profiled_optical=_rank_spectrum(profiled, rtol=rank_rtol),
        column_equilibrated_profiled_optical=_rank_spectrum(
            equilibrated_profiled, rtol=rank_rtol
        ),
        joint_column_norms=joint_norms,
        joint_active_column_mask=joint_active,
        profiled_optical_column_norms=profiled_norms,
        profiled_optical_active_column_mask=profiled_active,
        profiled_optical_precision=precision,
    )


def diagnose_blockwise_identifiability(
    jacobian_blocks: Mapping[str, ArrayLike]
    | Sequence[tuple[str, ArrayLike]],
    *,
    optical_parameter_count: int,
    parameter_scales: ArrayLike | None = None,
    rank_rtol: float = 1e-8,
    column_norm_atol: float = 0.0,
) -> BlockwiseIdentifiabilityDiagnostics:
    """Diagnose separately scaled acquisition blocks and their joint rank.

    Parameters
    ----------
    jacobian_blocks:
        Ordered named residual-Jacobian blocks.  Rows should already be
        whitened by their experimental statistical covariance.  All blocks
        must use the same joint parameter order: optical columns first,
        followed by shared nuisance columns.
    optical_parameter_count:
        Split point between optical and nuisance columns.
    parameter_scales:
        Positive physical-to-dimensionless column scales.  Nuisance latents
        expressed in standard-normal coordinates normally use scale one.
    rank_rtol:
        Relative SVD threshold applied independently to every reported matrix.
    column_norm_atol:
        Absolute norm, in whitened and parameter-scaled units, below which a
        column is kept explicitly zero during equilibration.  The default
        preserves every exactly non-zero column; use a positive value to
        suppress known numerical differentiation noise.

    Notes
    -----
    Column equilibration is a rank diagnostic, not a likelihood reweighting.
    Raw singular values retain the experimental information magnitude.  The
    equilibrated spectra reveal independent directions without allowing one
    exceptionally sensitive block to set the numerical scale for all others.
    """

    if not np.isfinite(rank_rtol) or not 0.0 < rank_rtol < 1.0:
        raise ValueError("rank_rtol must lie strictly in (0, 1).")
    if not np.isfinite(column_norm_atol) or column_norm_atol < 0.0:
        raise ValueError("column_norm_atol must be finite and non-negative.")
    entries = (
        tuple(jacobian_blocks.items())
        if isinstance(jacobian_blocks, Mapping)
        else tuple(jacobian_blocks)
    )
    if not entries:
        raise ValueError("jacobian_blocks must contain at least one block.")
    names = tuple(_nonempty(entry[0], "block_name") for entry in entries)
    if len(names) != len(set(names)):
        raise ValueError("jacobian block names must be unique.")
    matrices = tuple(np.asarray(entry[1], dtype=float) for entry in entries)
    if any(
        matrix.ndim != 2
        or matrix.shape[0] == 0
        or not np.all(np.isfinite(matrix))
        for matrix in matrices
    ):
        raise ValueError("Every Jacobian block must be a non-empty finite matrix.")
    joint_count = matrices[0].shape[1]
    if joint_count == 0 or any(matrix.shape[1] != joint_count for matrix in matrices):
        raise ValueError("All Jacobian blocks must share a non-zero column count.")
    optical_count = int(optical_parameter_count)
    if (
        optical_count < 1
        or optical_count > joint_count
        or optical_count != optical_parameter_count
    ):
        raise ValueError(
            "optical_parameter_count must be an integer in the joint column range."
        )
    scales = (
        np.ones(joint_count, dtype=float)
        if parameter_scales is None
        else np.asarray(parameter_scales, dtype=float)
    )
    if (
        scales.shape != (joint_count,)
        or not np.all(np.isfinite(scales))
        or np.any(scales <= 0.0)
    ):
        raise ValueError(
            "parameter_scales must be finite, positive, and match the joint columns."
        )
    scaled_blocks = tuple(
        matrix * scales[None, :] for matrix in matrices
    )
    reports = tuple(
        _block_rank_diagnostics(
            name,
            matrix,
            optical_parameter_count=optical_count,
            rank_rtol=float(rank_rtol),
            column_norm_atol=float(column_norm_atol),
        )
        for name, matrix in zip(names, scaled_blocks, strict=True)
    )
    combined_jacobian = np.vstack(scaled_blocks)
    combined = _block_rank_diagnostics(
        "combined",
        combined_jacobian,
        optical_parameter_count=optical_count,
        rank_rtol=float(rank_rtol),
        column_norm_atol=float(column_norm_atol),
    )
    return BlockwiseIdentifiabilityDiagnostics(
        block_reports=reports,
        combined=combined,
        scaled_combined_jacobian=combined_jacobian,
        parameter_scales=scales,
        optical_parameter_count=optical_count,
        nuisance_parameter_count=joint_count - optical_count,
        rank_rtol=float(rank_rtol),
        column_norm_atol=float(column_norm_atol),
    )


def _numerically_psd(
    matrix: np.ndarray,
    *,
    name: str,
    reference_scale: float | None = None,
) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    symmetric = 0.5 * (values + values.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(
        float(np.max(np.abs(eigenvalues), initial=0.0)),
        0.0 if reference_scale is None else float(reference_scale),
        np.finfo(float).tiny,
    )
    if eigenvalues.min(initial=0.0) < -1e-8 * scale:
        raise RuntimeError(
            f"{name} is not positive semidefinite within numerical tolerance."
        )
    result = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    return 0.5 * (result + result.T)


@dataclass(frozen=True)
class FisherPartition:
    """Data, prior, posterior, and profiled optical Fisher blocks."""

    data: np.ndarray
    prior: np.ndarray
    posterior: np.ndarray
    optical_optical_data: np.ndarray
    optical_nuisance_data: np.ndarray
    nuisance_nuisance_data: np.ndarray
    nuisance_nuisance_posterior: np.ndarray
    data_profiled_optical_precision: np.ndarray
    prior_assisted_marginal_optical_precision: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "data",
            "prior",
            "posterior",
            "optical_optical_data",
            "optical_nuisance_data",
            "nuisance_nuisance_data",
            "nuisance_nuisance_posterior",
            "data_profiled_optical_precision",
            "prior_assisted_marginal_optical_precision",
        ):
            object.__setattr__(
                self, name, _immutable_array(getattr(self, name), name, ndim=2)
            )


@dataclass(frozen=True)
class NuisanceIdentifiabilityDiagnostics:
    """Local ranks that keep data information separate from calibration priors.

    The two ``identifiable_covariance`` arrays are Moore--Penrose covariances
    on the resolved marginal optical subspace.  They must be interpreted with
    ``prior_assisted_marginal_optical_null_basis_scaled``; uncertainty is
    unbounded, not zero, along every vector in that null basis.
    """

    conditional_optical: RankSpectrum
    data_only_joint: RankSpectrum
    data_profiled_optical: RankSpectrum
    prior_assisted_joint: RankSpectrum
    prior_assisted_marginal_optical: RankSpectrum
    scaled_data_jacobian: np.ndarray
    scaled_prior_augmented_jacobian: np.ndarray
    fisher: FisherPartition
    optical_parameter_scales: np.ndarray
    prior_assisted_marginal_optical_identifiable_covariance_scaled: np.ndarray
    prior_assisted_marginal_optical_identifiable_covariance_physical: np.ndarray
    prior_assisted_marginal_optical_null_basis_scaled: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "scaled_data_jacobian",
            "scaled_prior_augmented_jacobian",
            "optical_parameter_scales",
            "prior_assisted_marginal_optical_identifiable_covariance_scaled",
            "prior_assisted_marginal_optical_identifiable_covariance_physical",
            "prior_assisted_marginal_optical_null_basis_scaled",
        ):
            ndim = 1 if name == "optical_parameter_scales" else 2
            object.__setattr__(
                self, name, _immutable_array(getattr(self, name), name, ndim=ndim)
            )


def diagnose_nuisance_identifiability(
    likelihood: SharedLatentNuisanceLikelihood,
    optical_reference: ArrayLike,
    *,
    optical_parameter_scales: ArrayLike | None = None,
    calibration_reference: ArrayLike | None = None,
    discrepancy_reference: ArrayLike | None = None,
    rank_rtol: float = 1e-8,
) -> NuisanceIdentifiabilityDiagnostics:
    """Partition local data/prior information and profile shared nuisances.

    Optical columns are nondimensionalized by ``optical_parameter_scales``;
    calibration and discrepancy coordinates are already standard normal.  The
    data-only ranks never include the prior rows.  The prior-assisted marginal
    optical precision is the Schur complement of all nuisance coordinates.
    """

    if not isinstance(likelihood, SharedLatentNuisanceLikelihood):
        raise TypeError("likelihood must be SharedLatentNuisanceLikelihood.")
    if not np.isfinite(rank_rtol) or not 0.0 < rank_rtol < 1.0:
        raise ValueError("rank_rtol must lie strictly in (0, 1).")
    optical = np.asarray(optical_reference, dtype=float)
    if optical.shape != (likelihood.optical_parameter_count,) or not np.all(
        np.isfinite(optical)
    ):
        raise ValueError("optical_reference has the wrong shape or non-finite values.")
    scales = (
        np.ones_like(optical)
        if optical_parameter_scales is None
        else np.asarray(optical_parameter_scales, dtype=float)
    )
    if (
        scales.shape != optical.shape
        or not np.all(np.isfinite(scales))
        or np.any(scales <= 0.0)
    ):
        raise ValueError(
            "optical_parameter_scales must be finite, positive, and match "
            "optical_reference."
        )
    calibration = (
        np.zeros(likelihood.n_calibration_latents, dtype=float)
        if calibration_reference is None
        else np.asarray(calibration_reference, dtype=float)
    )
    discrepancy = (
        np.zeros(likelihood.n_discrepancy_latents, dtype=float)
        if discrepancy_reference is None
        else np.asarray(discrepancy_reference, dtype=float)
    )
    if calibration.shape != (likelihood.n_calibration_latents,) or not np.all(
        np.isfinite(calibration)
    ):
        raise ValueError("calibration_reference has the wrong shape or non-finite values.")
    if discrepancy.shape != (likelihood.n_discrepancy_latents,) or not np.all(
        np.isfinite(discrepancy)
    ):
        raise ValueError("discrepancy_reference has the wrong shape or non-finite values.")

    reference = likelihood.join_parameters(optical, calibration, discrepancy)
    joint_scales = jnp.asarray(
        np.concatenate((scales, np.ones(likelihood.n_nuisance_latents)))
    )

    def data_at_scaled_delta(delta: jnp.ndarray) -> jnp.ndarray:
        return likelihood.data_residual(reference + joint_scales * delta)

    data_jacobian = np.asarray(
        jax.jacfwd(data_at_scaled_delta)(jnp.zeros(likelihood.n_joint_parameters)),
        dtype=float,
    )
    n_optical = likelihood.optical_parameter_count
    n_nuisance = likelihood.n_nuisance_latents
    optical_jacobian = data_jacobian[:, :n_optical]
    prior_jacobian = np.zeros((n_nuisance, likelihood.n_joint_parameters), dtype=float)
    if n_nuisance:
        prior_jacobian[:, n_optical:] = np.eye(n_nuisance)
    augmented_jacobian = np.vstack((data_jacobian, prior_jacobian))

    data_fisher = data_jacobian.T @ data_jacobian
    prior_fisher = prior_jacobian.T @ prior_jacobian
    posterior_fisher = data_fisher + prior_fisher
    f_oo = data_fisher[:n_optical, :n_optical]
    f_on = data_fisher[:n_optical, n_optical:]
    f_nn_data = data_fisher[n_optical:, n_optical:]
    f_nn_posterior = posterior_fisher[n_optical:, n_optical:]
    if n_nuisance:
        data_profiled = f_oo - f_on @ np.linalg.pinv(
            f_nn_data, rcond=rank_rtol**2, hermitian=True
        ) @ f_on.T
        marginal = f_oo - f_on @ np.linalg.solve(f_nn_posterior, f_on.T)
    else:
        data_profiled = f_oo.copy()
        marginal = f_oo.copy()
    optical_fisher_scale = float(np.max(np.abs(f_oo), initial=0.0))
    data_profiled = _numerically_psd(
        data_profiled,
        name="data-profiled optical precision",
        reference_scale=optical_fisher_scale,
    )
    marginal = _numerically_psd(
        marginal,
        name="prior-assisted marginal optical precision",
        reference_scale=optical_fisher_scale,
    )
    identifiable_covariance_scaled = np.linalg.pinv(
        marginal, rcond=rank_rtol**2, hermitian=True
    )
    marginal_eigenvalues, marginal_eigenvectors = np.linalg.eigh(marginal)
    marginal_threshold = rank_rtol**2 * max(
        float(np.max(marginal_eigenvalues, initial=0.0)), 0.0
    )
    null_basis_scaled = marginal_eigenvectors[
        :, marginal_eigenvalues <= marginal_threshold
    ]
    identifiable_covariance_physical = (
        identifiable_covariance_scaled * np.outer(scales, scales)
    )
    fisher = FisherPartition(
        data=data_fisher,
        prior=prior_fisher,
        posterior=posterior_fisher,
        optical_optical_data=f_oo,
        optical_nuisance_data=f_on,
        nuisance_nuisance_data=f_nn_data,
        nuisance_nuisance_posterior=f_nn_posterior,
        data_profiled_optical_precision=data_profiled,
        prior_assisted_marginal_optical_precision=marginal,
    )
    return NuisanceIdentifiabilityDiagnostics(
        conditional_optical=_rank_spectrum(optical_jacobian, rtol=rank_rtol),
        data_only_joint=_rank_spectrum(data_jacobian, rtol=rank_rtol),
        data_profiled_optical=_rank_spectrum(
            data_profiled, rtol=rank_rtol**2
        ),
        prior_assisted_joint=_rank_spectrum(
            augmented_jacobian, rtol=rank_rtol
        ),
        prior_assisted_marginal_optical=_rank_spectrum(
            marginal, rtol=rank_rtol**2
        ),
        scaled_data_jacobian=data_jacobian,
        scaled_prior_augmented_jacobian=augmented_jacobian,
        fisher=fisher,
        optical_parameter_scales=scales,
        prior_assisted_marginal_optical_identifiable_covariance_scaled=(
            identifiable_covariance_scaled
        ),
        prior_assisted_marginal_optical_identifiable_covariance_physical=(
            identifiable_covariance_physical
        ),
        prior_assisted_marginal_optical_null_basis_scaled=null_basis_scaled,
    )


def diagnose_blockwise_nuisance_identifiability(
    likelihood: SharedLatentNuisanceLikelihood,
    optical_reference: ArrayLike,
    *,
    optical_parameter_scales: ArrayLike | None = None,
    calibration_reference: ArrayLike | None = None,
    discrepancy_reference: ArrayLike | None = None,
    block_names: Sequence[str] | None = None,
    rank_rtol: float = 1e-8,
    column_norm_atol: float = 0.0,
) -> BlockwiseIdentifiabilityDiagnostics:
    """Apply the blockwise data diagnostic to a shared-latent likelihood.

    The selected heterogeneous-dataset blocks retain their exact order and
    are named ``dataset_block_<source index>`` unless ``block_names`` is
    supplied.  Calibration and discrepancy coordinates remain common columns
    in the combined Jacobian; they are not duplicated independently for every
    frame block.

    This function is deliberately data-only.  Use
    :func:`diagnose_nuisance_identifiability` alongside it for posterior ranks
    and prior-assisted marginal covariance.
    """

    if not isinstance(likelihood, SharedLatentNuisanceLikelihood):
        raise TypeError("likelihood must be SharedLatentNuisanceLikelihood.")
    optical = np.asarray(optical_reference, dtype=float)
    if optical.shape != (likelihood.optical_parameter_count,) or not np.all(
        np.isfinite(optical)
    ):
        raise ValueError("optical_reference has the wrong shape or non-finite values.")
    optical_scales = (
        np.ones_like(optical)
        if optical_parameter_scales is None
        else np.asarray(optical_parameter_scales, dtype=float)
    )
    if (
        optical_scales.shape != optical.shape
        or not np.all(np.isfinite(optical_scales))
        or np.any(optical_scales <= 0.0)
    ):
        raise ValueError(
            "optical_parameter_scales must be finite, positive, and match "
            "optical_reference."
        )
    calibration = (
        np.zeros(likelihood.n_calibration_latents, dtype=float)
        if calibration_reference is None
        else np.asarray(calibration_reference, dtype=float)
    )
    discrepancy = (
        np.zeros(likelihood.n_discrepancy_latents, dtype=float)
        if discrepancy_reference is None
        else np.asarray(discrepancy_reference, dtype=float)
    )
    if calibration.shape != (likelihood.n_calibration_latents,) or not np.all(
        np.isfinite(calibration)
    ):
        raise ValueError("calibration_reference has the wrong shape or non-finite values.")
    if discrepancy.shape != (likelihood.n_discrepancy_latents,) or not np.all(
        np.isfinite(discrepancy)
    ):
        raise ValueError("discrepancy_reference has the wrong shape or non-finite values.")
    reference = likelihood.join_parameters(optical, calibration, discrepancy)
    joint_scales = np.concatenate(
        (
            optical_scales,
            np.ones(likelihood.n_nuisance_latents, dtype=float),
        )
    )

    def data_at_scaled_delta(delta: jnp.ndarray) -> jnp.ndarray:
        return likelihood.data_residual(
            reference + jnp.asarray(joint_scales) * delta
        )

    scaled = np.asarray(
        jax.jacfwd(data_at_scaled_delta)(
            jnp.zeros(likelihood.n_joint_parameters)
        ),
        dtype=float,
    )
    selected = likelihood.layout.selected_block_indices
    names = (
        tuple(f"dataset_block_{index}" for index in selected)
        if block_names is None
        else _unique_names(tuple(block_names), "block_names")
    )
    if len(names) != len(selected):
        raise ValueError("block_names must contain one name per selected data block.")
    sizes = tuple(likelihood.layout.block_sizes[index] for index in selected)
    # Recover the raw-coordinate Jacobian so the low-level routine records and
    # reapplies the declared physical scales in its public result.
    raw = scaled / joint_scales[None, :]
    blocks: list[tuple[str, np.ndarray]] = []
    cursor = 0
    for name, size in zip(names, sizes, strict=True):
        stop = cursor + size
        blocks.append((name, raw[cursor:stop]))
        cursor = stop
    if cursor != raw.shape[0]:  # pragma: no cover - guarded by layout validation
        raise RuntimeError("Selected block sizes do not cover the data Jacobian rows.")
    return diagnose_blockwise_identifiability(
        blocks,
        optical_parameter_count=likelihood.optical_parameter_count,
        parameter_scales=joint_scales,
        rank_rtol=rank_rtol,
        column_norm_atol=column_norm_atol,
    )


__all__ = (
    "BlockRankDiagnostics",
    "BlockwiseIdentifiabilityDiagnostics",
    "CalibrationValues",
    "FisherPartition",
    "GaussianPriorFactor",
    "GlobalObservationLayout",
    "HeldOutBlockEvaluation",
    "HeldOutLikelihoodEvaluation",
    "NuisanceIdentifiabilityDiagnostics",
    "PredictionCallback",
    "RankSpectrum",
    "SharedLatentNuisanceLikelihood",
    "SignedDiscrepancyModes",
    "diagnose_blockwise_identifiability",
    "diagnose_blockwise_nuisance_identifiability",
    "diagnose_nuisance_identifiability",
    "evaluate_held_out_shared_likelihood",
)
