"""Calibration-covariance bookkeeping for Fourier image observables.

Detector/extractor noise is independent between exposures.  A certified
specimen spacing and an accelerating-voltage/wavelength calibration are not:
the same uncertain physical quantities affect every image in which they are
used.  This module keeps those contributions separate, records their exact
state-specific linear response, and exposes the input-space
``GaussianPriorFactor`` objects required by :mod:`temgym_core.physical_nuisance`.

The output covariance is useful for a standalone frame and for same-state
repeat pooling.  Across different optical states, callers should use the
shared input-space priors with a calibration-aware prediction callback.  The
metadata says this explicitly; treating the per-state output blocks as
independent would count the same certificate repeatedly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .physical_nuisance import GaussianPriorFactor


def _nonempty(value: object, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


def _immutable_array(
    values: object,
    name: str,
    *,
    ndim: int,
) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != ndim or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array.")
    result = result.copy()
    result.setflags(write=False)
    return result


def _validated_covariance(values: object, name: str, count: int) -> np.ndarray:
    covariance = np.asarray(values, dtype=float)
    if covariance.shape != (count, count) or not np.all(np.isfinite(covariance)):
        raise ValueError(f"{name} must be a finite ({count}, {count}) matrix.")
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


def gaussian_prior_to_metadata(prior: GaussianPriorFactor) -> dict[str, object]:
    """Serialize a shared calibration prior without losing its factorization."""

    if not isinstance(prior, GaussianPriorFactor):
        raise TypeError("prior must be a GaussianPriorFactor.")
    return {
        "schema": "temgym_core.physical_nuisance.GaussianPriorFactor",
        "version": 1,
        "group_name": prior.group_name,
        "value_names": list(prior.value_names),
        "mean": prior.mean.tolist(),
        "factor": prior.factor.tolist(),
        "latent_names": list(prior.latent_names),
        "provenance": prior.provenance,
    }


def gaussian_prior_from_metadata(metadata: Mapping[str, object]) -> GaussianPriorFactor:
    """Rebuild a :class:`GaussianPriorFactor` from stored calibration metadata."""

    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping.")
    if metadata.get("schema") != "temgym_core.physical_nuisance.GaussianPriorFactor":
        raise ValueError("metadata does not describe a GaussianPriorFactor.")
    return GaussianPriorFactor(
        group_name=str(metadata["group_name"]),
        value_names=tuple(metadata["value_names"]),  # type: ignore[arg-type]
        mean=np.asarray(metadata["mean"], dtype=float),
        factor=np.asarray(metadata["factor"], dtype=float),
        latent_names=tuple(metadata["latent_names"]),  # type: ignore[arg-type]
        provenance=str(metadata["provenance"]),
    )


@dataclass(frozen=True)
class FourierObservableCovarianceAugmentation:
    """Decomposed calibration covariance for one Fourier/Bragg observable block.

    ``world_vector_jacobian`` maps flattened certificate vectors ordered as
    ``(g0_x, g0_y, g1_x, g1_y, ...)`` to the observable ordering declared in
    ``observable_names``.  For imaging this is the exact linear map
    ``FOV H^-T`` including the final ``xy -> yx`` permutation.  For indexed
    diffraction it is the derivative of the exact direct map ``D B lambda g``
    at the declared wavelength.  ``wavelength_jacobian`` is the derivative of
    that same direct map with respect to wavelength.

    The optional ``bilinear_product_covariance_observable2`` is the exact
    second-order covariance term for independent uncertain wavelength and
    reciprocal vectors.  It is zero for imaging.  The shared latent model
    should nevertheless evaluate the bilinear prediction itself rather than
    approximate this term as an independent output error.
    """

    observable_names: tuple[str, ...]
    observable_kind: str
    certificate_id: str
    certificate_prior: GaussianPriorFactor
    world_vector_jacobian: np.ndarray
    certificate_output_factor: np.ndarray
    wavelength_prior: GaussianPriorFactor | None
    wavelength_jacobian: np.ndarray
    wavelength_output_factor: np.ndarray
    bilinear_product_covariance_observable2: np.ndarray
    detector_map_xy: np.ndarray
    map_provenance: str

    def __post_init__(self) -> None:
        names = tuple(str(value).strip() for value in self.observable_names)
        if not names or any(not value for value in names) or len(names) != len(
            set(names)
        ):
            raise ValueError("observable_names must be non-empty and unique.")
        kind = _nonempty(self.observable_kind, "observable_kind")
        if kind not in {"imaging_reciprocal_vectors", "indexed_diffraction_bragg"}:
            raise ValueError("Unsupported Fourier calibration observable_kind.")
        if not isinstance(self.certificate_prior, GaussianPriorFactor):
            raise TypeError("certificate_prior must be a GaussianPriorFactor.")
        count = len(names)
        jacobian = _immutable_array(
            self.world_vector_jacobian, "world_vector_jacobian", ndim=2
        )
        if jacobian.shape != (count, self.certificate_prior.n_values):
            raise ValueError(
                "world_vector_jacobian must map every certificate value to every "
                "observable."
            )
        certificate_factor = _immutable_array(
            self.certificate_output_factor,
            "certificate_output_factor",
            ndim=2,
        )
        expected_certificate_factor = jacobian @ self.certificate_prior.factor
        if (
            certificate_factor.shape != expected_certificate_factor.shape
            or not np.allclose(
                certificate_factor,
                expected_certificate_factor,
                rtol=1e-11,
                atol=1e-15,
            )
        ):
            raise ValueError(
                "certificate_output_factor must equal world_vector_jacobian @ "
                "certificate_prior.factor."
            )
        wavelength_jacobian = _immutable_array(
            self.wavelength_jacobian, "wavelength_jacobian", ndim=2
        )
        wavelength_factor = _immutable_array(
            self.wavelength_output_factor, "wavelength_output_factor", ndim=2
        )
        wavelength_prior = self.wavelength_prior
        if wavelength_prior is None:
            if wavelength_jacobian.shape != (count, 0) or wavelength_factor.shape != (
                count,
                0,
            ):
                raise ValueError(
                    "Imaging/no-wavelength augmentations require empty wavelength "
                    "Jacobian and factor arrays."
                )
        else:
            if not isinstance(wavelength_prior, GaussianPriorFactor):
                raise TypeError("wavelength_prior must be a GaussianPriorFactor.")
            if wavelength_prior.n_values != 1:
                raise ValueError("A wavelength prior must contain exactly one value.")
            if wavelength_jacobian.shape != (count, 1):
                raise ValueError("wavelength_jacobian must have shape (n, 1).")
            expected = wavelength_jacobian @ wavelength_prior.factor
            if wavelength_factor.shape != expected.shape or not np.allclose(
                wavelength_factor, expected, rtol=1e-11, atol=1e-15
            ):
                raise ValueError(
                    "wavelength_output_factor must equal wavelength_jacobian @ "
                    "wavelength_prior.factor."
                )
        interaction = _validated_covariance(
            self.bilinear_product_covariance_observable2,
            "bilinear_product_covariance_observable2",
            count,
        )
        detector_map = _immutable_array(
            self.detector_map_xy, "detector_map_xy", ndim=2
        )
        if detector_map.shape != (2, 2):
            raise ValueError("detector_map_xy must be 2x2.")
        object.__setattr__(self, "observable_names", names)
        object.__setattr__(self, "observable_kind", kind)
        object.__setattr__(
            self,
            "certificate_id",
            _nonempty(self.certificate_id, "certificate_id"),
        )
        object.__setattr__(self, "world_vector_jacobian", jacobian)
        object.__setattr__(self, "certificate_output_factor", certificate_factor)
        object.__setattr__(self, "wavelength_jacobian", wavelength_jacobian)
        object.__setattr__(self, "wavelength_output_factor", wavelength_factor)
        object.__setattr__(
            self, "bilinear_product_covariance_observable2", interaction
        )
        object.__setattr__(self, "detector_map_xy", detector_map)
        object.__setattr__(
            self,
            "map_provenance",
            _nonempty(self.map_provenance, "map_provenance"),
        )

    @property
    def certificate_covariance_observable2(self) -> np.ndarray:
        result = self.certificate_output_factor @ self.certificate_output_factor.T
        result.setflags(write=False)
        return result

    @property
    def wavelength_covariance_observable2(self) -> np.ndarray:
        result = self.wavelength_output_factor @ self.wavelength_output_factor.T
        result.setflags(write=False)
        return result

    @property
    def total_covariance_observable2(self) -> np.ndarray:
        result = (
            self.certificate_covariance_observable2
            + self.wavelength_covariance_observable2
            + self.bilinear_product_covariance_observable2
        )
        result = np.asarray(result, dtype=float)
        result.setflags(write=False)
        return result

    @property
    def calibration_priors(self) -> tuple[GaussianPriorFactor, ...]:
        if self.wavelength_prior is None:
            return (self.certificate_prior,)
        return (self.certificate_prior, self.wavelength_prior)

    def to_metadata(self) -> dict[str, object]:
        output_factors = [
            {
                "group_name": self.certificate_prior.group_name,
                "latent_names": list(self.certificate_prior.latent_names),
                "factor_observable": self.certificate_output_factor.tolist(),
            }
        ]
        if self.wavelength_prior is not None:
            output_factors.append(
                {
                    "group_name": self.wavelength_prior.group_name,
                    "latent_names": list(self.wavelength_prior.latent_names),
                    "factor_observable": self.wavelength_output_factor.tolist(),
                }
            )
        return {
            "schema": "temgym_core.fourier_observable_covariance_augmentation",
            "version": 1,
            "observable_names": list(self.observable_names),
            "observable_kind": self.observable_kind,
            "certificate_id": self.certificate_id,
            "certificate_covariance_observable2": (
                self.certificate_covariance_observable2.tolist()
            ),
            "wavelength_covariance_observable2": (
                self.wavelength_covariance_observable2.tolist()
            ),
            "bilinear_product_covariance_observable2": (
                self.bilinear_product_covariance_observable2.tolist()
            ),
            "total_covariance_observable2": (
                self.total_covariance_observable2.tolist()
            ),
            "world_vector_jacobian": self.world_vector_jacobian.tolist(),
            "wavelength_jacobian": self.wavelength_jacobian.tolist(),
            "detector_map_xy": self.detector_map_xy.tolist(),
            "map_provenance": self.map_provenance,
            "shared_input_priors": [
                gaussian_prior_to_metadata(prior) for prior in self.calibration_priors
            ],
            "state_specific_output_factors": output_factors,
            "shared_across_same_specimen_repeats": True,
            "cross_state_calibration_correlation": (
                "must_use_shared_input_latents; block_diagonal_output_covariance_"
                "would_treat_one_calibration_as_independent_repeats"
            ),
            "physical_nuisance_compatibility": (
                "shared_input_priors are GaussianPriorFactor metadata; evaluate "
                "the exact observable map with their values"
            ),
            "covariance_propagation": (
                "exact linear push-forward for imaging; exact first-order "
                "Jacobians plus independent bilinear product term for diffraction"
            ),
        }


def make_fourier_observable_covariance_augmentation(
    *,
    observable_names: Sequence[str],
    observable_kind: str,
    labels: Sequence[str],
    world_reciprocal_vectors_xy_per_m: object,
    covariance_world_per_m2: object,
    world_vector_jacobian: object,
    detector_map_xy: object,
    certificate_id: str,
    certificate_group_name: str,
    certificate_provenance: str,
    map_provenance: str,
    wavelength_m: float | None = None,
    wavelength_variance_m2: float | None = None,
    wavelength_jacobian: object | None = None,
    wavelength_group_name: str | None = None,
    wavelength_provenance: str | None = None,
    unit_wavelength_world_jacobian: object | None = None,
) -> FourierObservableCovarianceAugmentation:
    """Create a decomposed calibration covariance from exact map Jacobians."""

    observable_names_tuple = tuple(observable_names)
    labels_tuple = tuple(_nonempty(value, "label") for value in labels)
    if not labels_tuple or len(labels_tuple) != len(set(labels_tuple)):
        raise ValueError("labels must be non-empty and unique.")
    vectors = np.asarray(world_reciprocal_vectors_xy_per_m, dtype=float)
    if vectors.shape != (len(labels_tuple), 2) or not np.all(np.isfinite(vectors)):
        raise ValueError("world reciprocal vectors must have shape (n_labels, 2).")
    covariance = _validated_covariance(
        covariance_world_per_m2,
        "covariance_world_per_m2",
        2 * len(labels_tuple),
    )
    value_names = tuple(
        f"{label}.g_{axis}_per_m" for label in labels_tuple for axis in ("x", "y")
    )
    certificate_prior = GaussianPriorFactor.from_covariance(
        certificate_group_name,
        value_names,
        vectors.reshape(-1),
        covariance,
        provenance=certificate_provenance,
        eigenvalue_rtol=0.0,
    )
    jacobian = np.asarray(world_vector_jacobian, dtype=float)
    certificate_output_factor = jacobian @ certificate_prior.factor
    count = len(observable_names_tuple)
    empty_jacobian = np.empty((count, 0), dtype=float)
    empty_factor = np.empty((count, 0), dtype=float)
    interaction = np.zeros((count, count), dtype=float)
    wavelength_prior = None
    wavelength_jacobian_array = empty_jacobian
    wavelength_output_factor = empty_factor

    wavelength_arguments = (
        wavelength_m,
        wavelength_variance_m2,
        wavelength_jacobian,
        wavelength_group_name,
        wavelength_provenance,
    )
    if any(value is not None for value in wavelength_arguments):
        if any(value is None for value in wavelength_arguments):
            raise ValueError(
                "Diffraction covariance requires explicit wavelength mean, variance, "
                "Jacobian, group name, and provenance."
            )
        wavelength = float(wavelength_m)  # type: ignore[arg-type]
        variance = float(wavelength_variance_m2)  # type: ignore[arg-type]
        if not np.isfinite(wavelength) or wavelength <= 0.0:
            raise ValueError("wavelength_m must be finite and positive.")
        if not np.isfinite(variance) or variance < 0.0:
            raise ValueError("wavelength_variance_m2 must be finite and non-negative.")
        wavelength_prior = GaussianPriorFactor.from_covariance(
            str(wavelength_group_name),
            ("wavelength_m",),
            np.asarray((wavelength,)),
            np.asarray(((variance,),)),
            provenance=str(wavelength_provenance),
            eigenvalue_rtol=0.0,
        )
        wavelength_jacobian_array = np.asarray(wavelength_jacobian, dtype=float)
        wavelength_output_factor = (
            wavelength_jacobian_array @ wavelength_prior.factor
        )
        if unit_wavelength_world_jacobian is None:
            raise ValueError(
                "Diffraction covariance requires unit_wavelength_world_jacobian "
                "for the bilinear certificate/wavelength covariance term."
            )
        unit = np.asarray(unit_wavelength_world_jacobian, dtype=float)
        interaction = variance * (unit @ covariance @ unit.T)
    elif unit_wavelength_world_jacobian is not None:
        raise ValueError(
            "unit_wavelength_world_jacobian is valid only with wavelength covariance."
        )

    return FourierObservableCovarianceAugmentation(
        observable_names=observable_names_tuple,
        observable_kind=observable_kind,
        certificate_id=certificate_id,
        certificate_prior=certificate_prior,
        world_vector_jacobian=jacobian,
        certificate_output_factor=certificate_output_factor,
        wavelength_prior=wavelength_prior,
        wavelength_jacobian=wavelength_jacobian_array,
        wavelength_output_factor=wavelength_output_factor,
        bilinear_product_covariance_observable2=interaction,
        detector_map_xy=np.asarray(detector_map_xy, dtype=float),
        map_provenance=map_provenance,
    )


def deduplicate_calibration_priors(
    augmentations: Sequence[FourierObservableCovarianceAugmentation],
) -> tuple[GaussianPriorFactor, ...]:
    """Return one consistent prior per shared calibration group.

    The same certificate/wavelength group may occur in many states.  A name
    collision with different values, factors, or provenance is rejected
    rather than silently turning one physical calibration into several priors.
    """

    by_group: dict[str, GaussianPriorFactor] = {}
    for augmentation in augmentations:
        if not isinstance(augmentation, FourierObservableCovarianceAugmentation):
            raise TypeError(
                "augmentations must contain FourierObservableCovarianceAugmentation "
                "values."
            )
        for prior in augmentation.calibration_priors:
            existing = by_group.get(prior.group_name)
            if existing is None:
                by_group[prior.group_name] = prior
                continue
            compatible = (
                existing.value_names == prior.value_names
                and existing.latent_names == prior.latent_names
                and existing.provenance == prior.provenance
                and np.array_equal(existing.mean, prior.mean)
                and np.array_equal(existing.factor, prior.factor)
            )
            if not compatible:
                raise ValueError(
                    f"Shared calibration group {prior.group_name!r} has "
                    "inconsistent definitions across optical states."
                )
    return tuple(by_group.values())


__all__ = (
    "FourierObservableCovarianceAugmentation",
    "deduplicate_calibration_priors",
    "gaussian_prior_from_metadata",
    "gaussian_prior_to_metadata",
    "make_fourier_observable_covariance_augmentation",
)
