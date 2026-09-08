"""Finite uniform-field solenoids in ordinary TemGym component columns."""

import jax.numpy as jnp
import jax_dataclasses as jdc

from .components import ABCDTransfer, Component, Plane
from .microscope_inversion import hard_edge_solenoid_matrix
from .ray import Ray
from .run import run_to_end


@jdc.pytree_dataclass(kw_only=True)
class HardEdgeSolenoid(Component):
    """Entrance at ``z``, body length in metres, signed Larmor integral ``q``.

    Includes both edges and propagation through the body. Rotation is -q;
    q is proportional to current / relativistic momentum. This ideal uniform
    field is distinct from the principal-plane ``ThickLens`` approximation.
    Coherent transport includes the quadratic action and Gouy factor. The
    coherent branch covers the first radial half-period, abs(q) < pi;
    longer fields must be split into shorter bodies to continue the phase.
    """

    z: float
    length: float
    q: float
    include_axial_path: jdc.Static[bool] = True

    @property
    def matrix(self):
        order = jnp.array([0, 2, 1, 3])
        radial = hard_edge_solenoid_matrix(self.q, self.length)
        return radial[jnp.ix_(order, order)]

    def _call_ray(self, ray):
        m = self.matrix
        out = ABCDTransfer(A=m[:2, :2], B=m[:2, 2:],
                           C=m[2:, :2], D=m[2:, 2:])(ray)
        return out.derive(z=ray.z + self.length,
                          pathlength=ray.pathlength + self.length)

    def _call_gaussian(self, ray):
        m = self.matrix
        out = ABCDTransfer(A=m[:2, :2], B=m[:2, 2:],
                           C=m[2:, :2], D=m[2:, 2:])(ray)
        # Remove the real rotation from A+BQ before following its phase.
        # For |q|<pi, b>0 and both modal factors lie in the upper half-plane.
        b = self.length * jnp.sinc(self.q / jnp.pi)
        denominator = jnp.cos(self.q) * jnp.eye(2) + b * ray.Q_inv
        determinant = jnp.linalg.det(denominator)
        root = jnp.sqrt(determinant)
        root = jnp.where(jnp.imag(determinant) < 0., -root, root)
        # Homogeneous quadratic canonical action, including magnetic rotation.
        action = .5 * (jnp.sum(out.r_xy * out.d_xy, axis=-1)
                       - jnp.sum(ray.r_xy * ray.d_xy, axis=-1))
        valid = (jnp.abs(self.q) < jnp.pi) & (self.length > 0)
        return out.derive(
            z=ray.z + self.length,
            pathlength=ray.pathlength + (self.length if self.include_axial_path else 0.) + action,
            amplitude=jnp.where(valid, ray.amplitude / root, jnp.nan + 0j),
        )


def column_matrix(components, *, source_z=0.):
    """Affine 5×5 map in [x,y,theta_x,theta_y,1] order.

    Execute five basis rays through the same components and drift machinery
    used by normal TemGym runs. Valid for affine paraxial components only.
    Offsets are measured with the fifth (on-axis) ray.
    """
    basis = jnp.eye(5)
    ray = Ray(x=basis[0], y=basis[1], dx=basis[2], dy=basis[3],
              z=source_z, pathlength=0.)
    out = run_to_end(ray, components)
    values = jnp.stack((out.x, out.y, out.dx, out.dy))
    linear = values[:, :4] - values[:, 4, None]
    return jnp.eye(5).at[:4, :4].set(linear).at[:4, 4].set(values[:, 4])


def solenoid_column(gaps, lengths, strengths, *, source_z=0.):
    """Build lenses and a final plane; gaps run before/between/after bodies."""
    if len(gaps) != len(lengths) + 1 or len(strengths) != len(lengths):
        raise ValueError("Supply N lengths/strengths and N+1 free-space gaps.")
    entrance = source_z
    components = []
    for i in range(len(lengths)):
        entrance = entrance + gaps[i]
        components.append(HardEdgeSolenoid(z=entrance, length=lengths[i], q=strengths[i]))
        entrance = entrance + lengths[i]
    components.append(Plane(z=entrance + gaps[-1]))
    return tuple(components)


def solenoid_column_matrix(gaps, lengths, strengths):
    return column_matrix(solenoid_column(gaps, lengths, strengths))
