"""A single shared, immersed objective field in canonical ray coordinates.

The interleaved orders are ``(x, pi_x, y, pi_y)`` for canonical coordinates
and ``(x, alpha_x, y, alpha_y)`` for mechanical slopes. Momenta are normalized
by the fixed relativistic reference momentum. With signed particle charge Q,
``k = Q * B_z / (2*p)`` and the symmetric gauge, ``pi_x = alpha_x - k*y`` and
``pi_y = alpha_y + k*x``. Thus positive k gives Larmor rotation ``-integral k dz``.
For electrons Q=-e; do not silently replace Q by the unsigned elementary charge.

One smooth profile and one normalization determine both sides of the sample.
The finite integration window is explicitly truncated, with field-free exterior
planes. Its boundaries are not measured pole faces. Window and slice refinement
are mandatory before treating an image or inverse result as converged. If another
lens field overlaps this window, its field must be added before integration;
sequential independent overlapping lenses are not represented by this class.

Canonical maps act on Gaussian phases/curvatures. Conversion of calibrated
mechanical means and covariances at an immersed plane is a separate operation.
References: Floettmann & Karlovets, arXiv:2006.12948, Eq. (2); JEOL's C-O lens
description, https://www.jeol.com/words/emterms/20121023.034459.php.
"""

from dataclasses import dataclass, field as dataclass_field
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .microscope_inversion import canonical_symplectic_form


PARAMETER_NAMES = ("q_total", "z_peak_m", "log_width_m", "asymmetry")


def mechanical_to_canonical(k_m_inv):
    """T(k): mechanical slopes to normalized canonical momenta.

    Scalar k returns (4,4); batched k returns ``k.shape + (4,4)``. T is a
    coordinate conversion, not a standard canonical symplectic transfer map.
    """
    k = jnp.asarray(k_m_inv)
    zero, one = jnp.zeros_like(k), jnp.ones_like(k)
    return jnp.stack((jnp.stack((one, zero, zero, zero), axis=-1),
                      jnp.stack((zero, one, -k, zero), axis=-1),
                      jnp.stack((zero, zero, one, zero), axis=-1),
                      jnp.stack((k, zero, zero, one), axis=-1)), axis=-2)


def canonical_to_mechanical(k_m_inv):
    """Inverse T(k), retaining the same interleaved order."""
    return mechanical_to_canonical(-jnp.asarray(k_m_inv))


def mechanical_symplectic_form(k_m_inv):
    """Magnetic two-form Omega(k)=T(k).T @ J @ T(k)."""
    transform = mechanical_to_canonical(k_m_inv)
    return jnp.swapaxes(transform, -1, -2) @ canonical_symplectic_form(transform.dtype) @ transform


def _covariance_transform(covariance, transform):
    covariance = jnp.asarray(covariance)
    if covariance.ndim < 2 or covariance.shape[-2:] != (4, 4):
        raise ValueError("Covariance must have final dimensions (4,4).")
    return transform @ covariance @ jnp.swapaxes(transform, -1, -2)


def covariance_to_canonical(covariance, k_m_inv):
    """Transform a physical mechanical covariance; no nuisance is fixed here."""
    return _covariance_transform(covariance, mechanical_to_canonical(k_m_inv))


def covariance_to_mechanical(covariance, k_m_inv):
    return _covariance_transform(covariance, canonical_to_mechanical(k_m_inv))


def uniform_field_canonical_matrix(k_m_inv, distance_m):
    """Exact constant-field canonical map, smooth at zero field and length.

    Unlike evaluating q/length at a partial-body boundary, ``C=-k*sin(k*d)``
    and ``B=d*sinc(k*d)`` have finite first and second derivatives at d=0.
    Negative distances represent inverse propagation. Inputs broadcast and the
    returned array has shape ``broadcast(k,d).shape + (4,4)``.
    """
    k, distance = jnp.broadcast_arrays(jnp.asarray(k_m_inv), jnp.asarray(distance_m))
    phase = k * distance
    cosine, sine = jnp.cos(phase), jnp.sin(phase)
    b = distance * jnp.sinc(phase / jnp.pi)
    c = -k * sine
    return jnp.stack((
        jnp.stack((cosine*cosine, cosine*b, sine*cosine, sine*b), axis=-1),
        jnp.stack((cosine*c, cosine*cosine, sine*c, sine*cosine), axis=-1),
        jnp.stack((-sine*cosine, -sine*b, cosine*cosine, cosine*b), axis=-1),
        jnp.stack((-sine*c, -sine*cosine, cosine*c, cosine*cosine), axis=-1)), axis=-2)


class ObjectiveSplitMaps(NamedTuple):
    """Canonical pre/post maps and the signed physical field at the sample."""

    pre: jax.Array
    post: jax.Array
    sample_k_m_inv: jax.Array

    @property
    def full(self):
        return self.post @ self.pre

    @property
    def pre_mechanical(self):
        """Field-free exterior input to mechanical sample coordinates."""
        return canonical_to_mechanical(self.sample_k_m_inv) @ self.pre

    @property
    def post_mechanical(self):
        """Mechanical sample input to the field-free exterior output."""
        return self.post @ mechanical_to_canonical(self.sample_k_m_inv)


def _integrate_canonical(distances, rates):
    def advance(matrix, values):
        distance, rate = values
        return uniform_field_canonical_matrix(rate, distance) @ matrix, None
    initial = jnp.eye(4, dtype=jnp.result_type(distances, rates))
    return jax.lax.scan(advance, initial, (distances, rates))[0]


@dataclass(frozen=True)
class SharedObjectiveField:
    """Fixed quadrature topology for a shared asymmetric Glaser-like field.

    Parameters supplied to methods have order :data:`PARAMETER_NAMES`. The
    profile uses t=(z-z_peak)/width, u=t*exp(-a*tanh(t)), f=1/(1+u**2).
    For |a|<=0.5 the profile is smooth and positive, with its sole maximum
    at z_peak; a=0 is symmetric. Asymmetry is a fitted model option, not a
    measured field shape or an assertion of mirror symmetry.

    ``q_total`` is the signed midpoint-quadrature integral over the declared
    window, not over infinite tails. Both halves use that same normalization.
    The sample must be an explicit interior edge; no topology changes occur
    during JIT, optimization, or differentiation. Host validation is available
    separately, while traced invalid parameters produce NaNs (never clipping).
    """

    z_edges_m: tuple[float, ...]
    sample_z_m: float
    asymmetry_limit: float = .5
    _sample_index: int = dataclass_field(init=False, repr=False)

    def __post_init__(self):
        edges = np.asarray(self.z_edges_m)
        if (edges.ndim != 1 or edges.size < 3 or np.iscomplexobj(edges)
                or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0)):
            raise ValueError("z_edges_m must be a finite, strictly increasing real vector with at least three edges.")
        if not np.isfinite(self.sample_z_m):
            raise ValueError("sample_z_m must be finite.")
        indices = np.flatnonzero(edges == self.sample_z_m)
        if indices.size != 1 or indices[0] in (0, edges.size-1):
            raise ValueError("The sample must be an exact interior quadrature edge.")
        if not np.isfinite(self.asymmetry_limit) or not 0 < self.asymmetry_limit <= .5:
            raise ValueError("asymmetry_limit must be finite and in (0,0.5].")
        object.__setattr__(self, "z_edges_m", tuple(float(value) for value in edges))
        object.__setattr__(self, "sample_z_m", float(self.sample_z_m))
        object.__setattr__(self, "_sample_index", int(indices[0]))

    @classmethod
    def from_window(cls, z_min_m, sample_z_m, z_max_m, *, n_slices_per_side=128, asymmetry_limit=.5):
        """Construct two fixed grids meeting exactly at the sample."""
        if not np.all(np.isfinite([z_min_m, sample_z_m, z_max_m])) or not z_min_m < sample_z_m < z_max_m:
            raise ValueError("Expected finite z_min_m < sample_z_m < z_max_m.")
        counts = np.asarray(n_slices_per_side)
        if counts.ndim == 0:
            counts = np.repeat(counts, 2)
        if (counts.shape != (2,) or counts.dtype.kind == "b" or not np.all(np.isfinite(counts))
                or np.any(counts < 1) or np.any(counts != np.floor(counts))):
            raise ValueError("n_slices_per_side must be a positive integer or a pair of positive integers.")
        left = np.linspace(z_min_m, sample_z_m, int(counts[0])+1)
        right = np.linspace(sample_z_m, z_max_m, int(counts[1])+1)
        return cls(tuple(np.r_[left, right[1:]]), sample_z_m, asymmetry_limit)

    @property
    def parameter_names(self):
        return PARAMETER_NAMES

    @property
    def sample_index(self):
        return self._sample_index

    def _parameters(self, parameters):
        p = jnp.asarray(parameters)
        if p.shape != (4,) or jnp.iscomplexobj(p):
            raise ValueError("Objective parameters must be one real vector of length four.")
        return p

    def parameters_valid(self, parameters):
        """JAX scalar validity flag; does not assess physical identifiability."""
        p = self._parameters(parameters)
        width = jnp.exp(p[2])
        return (jnp.all(jnp.isfinite(p)) & jnp.isfinite(width) & (width > 0)
                & (p[1] > self.z_edges_m[0]) & (p[1] < self.z_edges_m[-1])
                & (jnp.abs(p[3]) <= self.asymmetry_limit))

    def validate_parameters(self, parameters):
        """Host-side validation before entering a compiled acquisition/fit."""
        if not bool(self.parameters_valid(parameters)):
            raise ValueError("Objective parameters must be finite, with positive finite width, interior peak, and bounded asymmetry.")

    def _profile(self, z_m, parameters):
        t = (jnp.asarray(z_m)-parameters[1]) / jnp.exp(parameters[2])
        u = t * jnp.exp(-parameters[3]*jnp.tanh(t))
        return 1. / (1. + u*u)

    def _normalized_samples(self, parameters):
        p = self._parameters(parameters)
        edges = jnp.asarray(self.z_edges_m, dtype=jnp.result_type(p, jnp.float32))
        distances = jnp.diff(edges)
        profile = self._profile(.5*(edges[:-1]+edges[1:]), p)
        normalization = jnp.sum(profile*distances)
        valid = (self.parameters_valid(p) & jnp.all(distances > 0)
                 & jnp.isfinite(normalization) & (normalization > 0))
        normalization = jnp.where(valid, normalization, jnp.nan)
        return p, distances, profile, normalization

    def field_samples(self, parameters):
        """Signed k [1/m] at every midpoint, normalized once for both halves."""
        p, _, profile, normalization = self._normalized_samples(parameters)
        return p[0] * profile / normalization

    def field(self, z_query_m, parameters):
        """Evaluate analytic k(z) with the shared quadrature normalization.

        Values outside the finite window are zero. The endpoints themselves
        return the interior limiting field; the split-map exterior planes lie
        just outside those truncation fringes. No artificial fringe is inserted
        at the specimen. Invalid parameters return NaNs even outside the window.
        """
        p, _, _, normalization = self._normalized_samples(parameters)
        z = jnp.asarray(z_query_m)
        inside = (z >= self.z_edges_m[0]) & (z <= self.z_edges_m[-1])
        values = jnp.where(inside, p[0] * self._profile(z, p) / normalization, 0.)
        return jnp.where(jnp.isfinite(normalization) & jnp.isfinite(z), values, jnp.nan)

    def split_maps(self, parameters):
        """Return the two canonical maps and shared sample field (JIT-safe)."""
        p, distances, profile, normalization = self._normalized_samples(parameters)
        rates = p[0] * profile / normalization
        split = self._sample_index
        pre = _integrate_canonical(distances[:split], rates[:split])
        post = _integrate_canonical(distances[split:], rates[split:])
        sample_rate = p[0] * self._profile(self.sample_z_m, p) / normalization
        return ObjectiveSplitMaps(pre, post, sample_rate)

    def support_diagnostics(self, parameters):
        """Resolution/tail indicators, not an automatic convergence certificate."""
        p, distances, profile, normalization = self._normalized_samples(parameters)
        boundary_profile = self._profile(jnp.asarray([self.z_edges_m[0], self.z_edges_m[-1]]), p)
        fraction_pre = jnp.sum(profile[:self._sample_index]*distances[:self._sample_index]) / normalization
        return {"normalization_length_m": normalization,
                "max_slice_to_width_ratio": jnp.max(distances)/jnp.exp(p[2]),
                "boundary_to_peak_field_ratio": boundary_profile,
                "upstream_strength_fraction": fraction_pre,
                "downstream_strength_fraction": 1.-fraction_pre}
