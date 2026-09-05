"""An invertible source quotient referenced to one fixed optical acquisition.

This is a coordinate change, not a measured covariance or a new source prior.
The reference control and numerical units are frozen before optimization. The
source covariance is transported back through each trial condenser, preserving
the original likelihood and its unobserved angular-momentum covariance family.
"""

import jax.numpy as jnp
import numpy as np

from neoarm_condenser_covariance_fit import BASIS, EPSILON, SOURCE_SCALE
from neoarm_condenser_image_recovery import OMEGA


class ReferenceSourceChart:
    """Nine observable source coordinates at the objective entrance.

    Requires the existing isolated round-condenser model, not an internal
    stigmator. In that model M G M^T = G for the invisible covariance G, so
    projecting out G before/after transport induces an invertible quotient map.
    """

    def __init__(self, quotient, reference_control, reference_width_m):
        self.quotient = quotient
        control = np.asarray(reference_control, dtype=float)
        width = float(reference_width_m)
        if control.shape != (7,) or not np.isfinite(control).all():
            raise ValueError("One fixed finite seven-control acquisition is required.")
        if not np.isfinite(width) or width <= 0:
            raise ValueError("A positive finite reference width is required.")
        self.reference_control = jnp.asarray(control)
        self.reference_width_m = width
        self.reference_scale = jnp.asarray([width, EPSILON/width]*2)

    def balanced_map(self, parameters):
        optical = self.quotient.decode(parameters)[0]
        matrix = self.quotient.model.condenser_matrix(optical, self.reference_control)
        return matrix*jnp.asarray(SOURCE_SCALE)[None, :]/self.reference_scale[:, None]

    def encode(self, original_parameters):
        p = jnp.asarray(original_parameters)
        matrix = self.balanced_map(p)
        covariance = jnp.einsum("k,kij->ij", p[12:21], jnp.asarray(BASIS))
        transported = matrix@covariance@matrix.T
        coordinates = jnp.einsum("kij,ij->k", jnp.asarray(BASIS), transported)
        return p.at[12:21].set(coordinates)

    def decode(self, reference_parameters):
        p = jnp.asarray(reference_parameters)
        matrix = self.balanced_map(p)
        omega = jnp.asarray(OMEGA)
        inverse = -omega@matrix.T@omega
        covariance = jnp.einsum("k,kij->ij", p[12:21], jnp.asarray(BASIS))
        transported = inverse@covariance@inverse.T
        coordinates = jnp.einsum("kij,ij->k", jnp.asarray(BASIS), transported)
        return p.at[12:21].set(coordinates)

    def admissible(self, parameters):
        return self.quotient.admissible(np.asarray(self.decode(parameters)))

