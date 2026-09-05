"""Dense canonical ray profiles through a summed axial magnetic field.

The integration mesh and all step matrices are inherited from the column
integrator. Prefix products permit dense ray drawing and event refinement
without retracing the entire column for every queried plane. These profiles
are numerical predictions of a specified field, not measured lens geometry.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .axial_field_column import canonical_cf4_step, objective_tail_field


class AxialFieldPrefix(NamedTuple):
    z_m: jax.Array
    canonical_maps: jax.Array
    larmor_radians: jax.Array
    body_rates_m_inv: jax.Array
    objective_active: jax.Array


def axial_field_prefix(mesh, objective_parameters, boundaries_m, controlled_rates_m_inv):
    """Return source-to-edge matrices and continuous accumulated Larmor angle."""
    left, right, body, active, valid = mesh._intervals(boundaries_m, controlled_rates_m_inv)
    distance = right-left
    node_early, node_late = .5-jnp.sqrt(3.)/6, .5+jnp.sqrt(3.)/6
    k1 = body+active*objective_tail_field(left+node_early*distance, objective_parameters,
                                         mesh.normalization_field)
    k2 = body+active*objective_tail_field(left+node_late*distance, objective_parameters,
                                         mesh.normalization_field)
    steps = canonical_cf4_step(k1, k2, distance)
    increments = -.5*(k1+k2)*distance
    identity = jnp.eye(4,dtype=steps.dtype)
    def step(carry, values):
        previous, angle = carry
        matrix, increment = values
        state = (matrix@previous, angle+increment)
        return state, state
    _, (matrices, angles) = jax.lax.scan(step, (identity, jnp.asarray(0.,dtype=steps.dtype)),
                                       (steps, increments))
    return AxialFieldPrefix(jnp.r_[left[:1],right],
        jnp.where(valid,jnp.concatenate((identity[None],matrices)),jnp.nan),
        jnp.where(valid,jnp.r_[0.,angles],jnp.nan),body,active)


def prefix_map_to_plane(mesh, objective_parameters, prefix, z_m):
    """Evaluate a partial final CF4 interval after its cached prefix.

    At a hard fringe the matrix is continuous and its z derivative is
    one-sided. The reference mesh must have been constructed from the same
    parameters and geometry. This helper does not validate that provenance.
    """
    z_m = jnp.asarray(z_m)
    index = jnp.clip(jnp.searchsorted(prefix.z_m,z_m,side='right')-1,
                     0,prefix.z_m.size-2)
    left = prefix.z_m[index]
    distance = z_m-left
    c1, c2 = .5-jnp.sqrt(3.)/6, .5+jnp.sqrt(3.)/6
    body, active = prefix.body_rates_m_inv[index], prefix.objective_active[index]
    k1 = body+active*objective_tail_field(left+c1*distance,objective_parameters,
                                         mesh.normalization_field)
    k2 = body+active*objective_tail_field(left+c2*distance,objective_parameters,
                                         mesh.normalization_field)
    matrix = canonical_cf4_step(k1,k2,distance)@prefix.canonical_maps[index]
    angle = prefix.larmor_radians[index]-.5*(k1+k2)*distance
    valid = (z_m>=prefix.z_m[0]) & (z_m<=prefix.z_m[-1])
    return jnp.where(valid,matrix,jnp.nan),jnp.where(valid,angle,jnp.nan)


def sample_conditioned_basis(mesh, objective_parameters, prefix, sample_z_m,
                             sample_canonical_basis, z_m):
    """Two sample-defined rays in the local co-rotating transverse frame.

    Returns both transverse components and the full canonical states. For a
    circular pure Gaussian, a canonical waist basis produces the usual two
    real radial solutions. Mechanical height/tilt inputs must first undergo
    the sample magnetic coordinate conversion; their orthogonal response
    need not vanish and is retained here.
    """
    sample_map, sample_angle = prefix_map_to_plane(mesh,objective_parameters,prefix,sample_z_m)
    source_basis = jnp.linalg.solve(sample_map,jnp.asarray(sample_canonical_basis))
    matrix, angle = prefix_map_to_plane(mesh,objective_parameters,prefix,z_m)
    states = matrix@source_basis
    phi = angle-sample_angle
    axes = jnp.array([[jnp.cos(phi),jnp.sin(phi)],[-jnp.sin(phi),jnp.cos(phi)]])
    return axes@states[jnp.array([0,2])],states
