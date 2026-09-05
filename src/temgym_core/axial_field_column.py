"""Canonical transport through a summed axial magnetic field.

Hard-edge body/gap membership belongs to explicit geometric segments, never a
sampled indicator function. Fixed fractions within each moving segment allow
JAX geometry derivatives to include motion of both fringe boundaries. Smooth
objective tails are added to the active controlled field *before squaring k*.

The fourth-order commutator-free Magnus formula uses two Gaussian nodes and
two analytic Hamiltonian exponentials per interval (Alvermann et al.,
arXiv:1205.1379, equations 11–12). Every step is canonical symplectic. The
profile and hard-edge fields remain synthetic assumptions, not measured data.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .immersed_objective import (
    SharedObjectiveField, canonical_to_mechanical, mechanical_to_canonical,
)


def _cosine_and_sinc_of_square(square):
    """Entire real functions cos(sqrt(t)), sin(sqrt(t))/sqrt(t), including t<0."""
    t = jnp.asarray(square)
    root = jnp.sqrt(jnp.maximum(jnp.abs(t), 1e-3))
    cosine = jnp.where(t >= 0, jnp.cos(root), jnp.cosh(root))
    sinc = jnp.where(t >= 0, jnp.sin(root)/root, jnp.sinh(root)/root)
    # Avoid singular AD through sqrt at zero. At this threshold the omitted
    # fifth terms are below float64 roundoff, including two derivatives.
    cosine_series = 1+t*(-.5+t*(1/24+t*(-1/720+t/40320)))
    sinc_series = 1+t*(-1/6+t*(1/120+t*(-1/5040+t/362880)))
    return (jnp.where(jnp.abs(t) < 1e-3, cosine_series, cosine),
            jnp.where(jnp.abs(t) < 1e-3, sinc_series, sinc))


def round_hamiltonian_exponential(drift, rotation, focusing, distance):
    """exp(distance*[drift*D + rotation*R + focusing*F]).

    Radial generator [[0,drift],[-focusing,0]] commutes with the transverse
    rotation generator. A CF4 combination can have negative focusing; the
    hyperbolic branch is retained, not clipped or converted to abs(focusing).
    Inputs broadcast. The matrix order is (x,pi_x,y,pi_y).
    """
    drift, rotation, focusing, distance = jnp.broadcast_arrays(
        *map(jnp.asarray, (drift, rotation, focusing, distance)))
    c, sinc = _cosine_and_sinc_of_square(drift*focusing*distance**2)
    b, f = distance*drift*sinc, -distance*focusing*sinc
    cosine, sine = jnp.cos(rotation*distance), jnp.sin(rotation*distance)
    return jnp.stack((
        jnp.stack((cosine*c, cosine*b, sine*c, sine*b), -1),
        jnp.stack((cosine*f, cosine*c, sine*f, sine*c), -1),
        jnp.stack((-sine*c, -sine*b, cosine*c, cosine*b), -1),
        jnp.stack((-sine*f, -sine*c, cosine*f, cosine*c), -1)), -2)


def canonical_cf4_step(k_early, k_late, distance):
    """Fourth-order step from total signed fields at the two Gauss nodes."""
    a, b = (3-2*jnp.sqrt(3.))/12, (3+2*jnp.sqrt(3.))/12
    early = round_hamiltonian_exponential(.5, b*k_early+a*k_late,
                                          b*k_early**2+a*k_late**2, distance)
    late = round_hamiltonian_exponential(.5, a*k_early+b*k_late,
                                         a*k_early**2+b*k_late**2, distance)
    return late @ early


def objective_tail_field(z_m, parameters, normalization_field):
    """Continue the existing analytic profile without its finite-window mask.

    q retains its *original* finite-reference-window quadrature normalization;
    integrating farther never renormalizes peak field or changes the specimen
    field. The reference normalizer is fixed across transport refinement.
    """
    p = jnp.asarray(parameters)
    normalization = normalization_field.support_diagnostics(p)["normalization_length_m"]
    t = (jnp.asarray(z_m)-p[1])/jnp.exp(p[2])
    u = t*jnp.exp(-p[3]*jnp.tanh(t))
    return p[0]/(normalization*(1+u*u))


class AxialColumnMaps(NamedTuple):
    pre: jax.Array
    post: jax.Array
    source_k_m_inv: jax.Array
    sample_k_m_inv: jax.Array
    detector_k_m_inv: jax.Array

    @property
    def full(self):
        return self.post @ self.pre

    @property
    def pre_mechanical(self):
        return canonical_to_mechanical(self.sample_k_m_inv) @ self.pre @ mechanical_to_canonical(self.source_k_m_inv)

    @property
    def post_mechanical(self):
        return canonical_to_mechanical(self.detector_k_m_inv) @ self.post @ mechanical_to_canonical(self.sample_k_m_inv)

    @property
    def full_mechanical(self):
        return canonical_to_mechanical(self.detector_k_m_inv) @ self.full @ mechanical_to_canonical(self.source_k_m_inv)


@dataclass(frozen=True)
class FixedAxialFieldMesh:
    """Fixed fractions in explicitly moving, ordered body/gap segments.

    ``segment_body_indices`` uses -1 for a gap, otherwise an index into the
    supplied controlled-field rates. Source, sample and detector must be in
    gaps, not on a controlled hard fringe. Field membership is fixed by this
    topology; crossings/reordering are invalid and return NaNs under JIT.

    The optional static objective-active flags reproduce a deliberately
    truncated comparison. Full-tail propagation uses True on every segment.
    """

    normalization_field: SharedObjectiveField
    fraction_edges: tuple
    segment_body_indices: tuple
    sample_boundary_index: int
    objective_active_segments: tuple

    def __post_init__(self):
        count = len(self.fraction_edges)
        if (len(self.segment_body_indices) != count or len(self.objective_active_segments) != count
                or not 0 < self.sample_boundary_index < count):
            raise ValueError("One field label per segment and an interior sample boundary are required.")
        for values in self.fraction_edges:
            values = np.asarray(values)
            if (values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values))
                    or values[0] != 0 or values[-1] != 1 or np.any(np.diff(values) <= 0)):
                raise ValueError("Each fixed fraction grid must increase exactly from zero to one.")
        indices = np.asarray(self.segment_body_indices)
        if np.any(indices < -1) or np.any(indices != np.floor(indices)):
            raise ValueError("Body labels must be integers >=-1.")
        sample = self.sample_boundary_index
        if any(indices[i] != -1 for i in (0, count-1, sample-1, sample)):
            raise ValueError("Source, sample and detector endpoints must be in controlled-field gaps.")
        if self.objective_active_segments[sample-1] != self.objective_active_segments[sample]:
            raise ValueError("The sample must not be an artificial objective truncation fringe.")

    @classmethod
    def from_boundaries(cls, normalization_field, nominal_boundaries_m, segment_body_indices,
                        sample_boundary_index, *, steps_per_segment=32,
                        mesh_centre_m=-.0005, mesh_scale_m=.003, truncate_objective=False):
        boundaries = np.asarray(nominal_boundaries_m, float)
        if (boundaries.ndim != 1 or len(boundaries) < 3 or not np.all(np.isfinite(boundaries))
                or np.any(np.diff(boundaries) <= 0)):
            raise ValueError("Nominal segment boundaries must be finite and strictly increasing.")
        count = len(boundaries)-1
        steps = np.asarray(steps_per_segment)
        if steps.ndim == 0:
            steps = np.repeat(steps, count)
        if (steps.shape != (count,) or steps.dtype.kind == 'b' or not np.all(np.isfinite(steps))
                or np.any(steps < 1) or np.any(steps != np.floor(steps))):
            raise ValueError("Use positive integer subdivision counts on each fixed segment.")
        if not np.isfinite(mesh_centre_m) or not np.isfinite(mesh_scale_m) or mesh_scale_m <= 0:
            raise ValueError("The fixed tangent mesh needs a finite centre and positive scale.")
        fractions, active = [], []
        low, high = normalization_field.z_edges_m[0], normalization_field.z_edges_m[-1]
        for a, b, n in zip(boundaries[:-1], boundaries[1:], steps):
            angle = np.linspace(np.arctan((a-mesh_centre_m)/mesh_scale_m),
                                np.arctan((b-mesh_centre_m)/mesh_scale_m), int(n)+1)
            points = mesh_centre_m+mesh_scale_m*np.tan(angle)
            fraction = (points-a)/(b-a)
            fraction[0], fraction[-1] = 0., 1.
            fractions.append(tuple(fraction))
            if truncate_objective and ((a < low < b) or (a < high < b)):
                raise ValueError("A truncated comparison needs explicit boundaries at both truncation planes.")
            active.append(not truncate_objective or (a >= low and b <= high))
        return cls(normalization_field, tuple(fractions), tuple(segment_body_indices),
                   sample_boundary_index, tuple(active))

    @property
    def interval_count(self):
        return sum(len(values)-1 for values in self.fraction_edges)

    @property
    def sample_interval_index(self):
        return sum(len(values)-1 for values in self.fraction_edges[:self.sample_boundary_index])

    def _intervals(self, boundaries_m, controlled_rates_m_inv):
        boundaries, rates = jnp.asarray(boundaries_m), jnp.asarray(controlled_rates_m_inv)
        count = len(self.fraction_edges)
        body_count = max(self.segment_body_indices)+1
        if boundaries.shape != (count+1,) or rates.shape != (body_count,):
            raise ValueError("Boundary/rate arrays do not match the frozen mesh topology.")
        segment = jnp.asarray(np.repeat(np.arange(count), [len(values)-1 for values in self.fraction_edges]))
        left_fraction = jnp.asarray(np.concatenate([values[:-1] for values in self.fraction_edges]))
        right_fraction = jnp.asarray(np.concatenate([values[1:] for values in self.fraction_edges]))
        length = jnp.diff(boundaries)
        left = boundaries[segment]+left_fraction*length[segment]
        right = boundaries[segment]+right_fraction*length[segment]
        indices = jnp.asarray(self.segment_body_indices)[segment]+1
        body_rates = jnp.concatenate((jnp.zeros(1, dtype=rates.dtype), rates))[indices]
        active = jnp.asarray(self.objective_active_segments)[segment]
        valid = (jnp.all(jnp.isfinite(boundaries)) & jnp.all(length > 0)
                 & jnp.all(jnp.isfinite(rates)))
        return left, right, body_rates, active, valid

    def _steps(self, objective_parameters, boundaries_m, controlled_rates_m_inv, stop_z_m=None):
        left, right, body_rates, active, valid = self._intervals(boundaries_m, controlled_rates_m_inv)
        if stop_z_m is not None:
            # Zero-length intervals after the plane are exact identity maps.
            left, right = jnp.minimum(left, stop_z_m), jnp.minimum(right, stop_z_m)
            valid &= (stop_z_m >= boundaries_m[0]) & (stop_z_m <= boundaries_m[-1])
        length = right-left
        c1, c2 = .5-jnp.sqrt(3.)/6, .5+jnp.sqrt(3.)/6
        k1 = active*objective_tail_field(left+c1*length, objective_parameters, self.normalization_field)+body_rates
        k2 = active*objective_tail_field(left+c2*length, objective_parameters, self.normalization_field)+body_rates
        matrices = canonical_cf4_step(k1, k2, length)
        return jnp.where(valid, matrices, jnp.nan)

    @staticmethod
    def _product(matrices):
        return jax.lax.scan(lambda carry, matrix: (matrix@carry, None),
                            jnp.eye(4, dtype=matrices.dtype), matrices)[0]

    def split_maps(self, objective_parameters, boundaries_m, controlled_rates_m_inv):
        matrices = self._steps(objective_parameters, boundaries_m, controlled_rates_m_inv)
        pre = self._product(matrices[:self.sample_interval_index])
        post = self._product(matrices[self.sample_interval_index:])
        indices = jnp.asarray([0, self.sample_boundary_index, len(self.fraction_edges)])
        rates = objective_tail_field(jnp.asarray(boundaries_m)[indices], objective_parameters,
                                     self.normalization_field)
        active = jnp.asarray([self.objective_active_segments[0],
                              self.objective_active_segments[self.sample_boundary_index],
                              self.objective_active_segments[-1]])
        return AxialColumnMaps(pre, post, *(rates*active))

    def canonical_map_to_plane(self, objective_parameters, boundaries_m, controlled_rates_m_inv, z_m):
        """Canonical source to arbitrary in-column plane, including partial bodies.

        Position rows are physical; momentum rows at an immersed plane remain
        canonical. Derivatives at an ideal hard fringe can be one-sided. This
        is a physical hard-edge limitation, not a sampled-membership artifact.
        """
        return self._product(self._steps(objective_parameters, boundaries_m,
                                         controlled_rates_m_inv, jnp.asarray(z_m)))
