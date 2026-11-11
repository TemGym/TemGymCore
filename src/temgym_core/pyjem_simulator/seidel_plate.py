"""A thin plate of Seidel field aberrations that acts on rays.

TemGym's :class:`~temgym_core.components.SeidelLens` acts on Gaussian beams only; the simulated microscope draws its
frames with rays, so this plate applies the same Seidel wave aberration to rays.
"""

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from ..aberrations import SeidelCoeffs, Seidel_object_pos_aperture_pos
from ..components import Component
from ..ray import Ray


@jdc.pytree_dataclass
class SeidelPlate(Component):
    """Seidel aberrations of a lens, as a thin plate at ``z`` that adds no focusing.

    A ray crosses the plate at the aperture position ``(x, y)`` and, traced back to the object plane
    ``object_plane_dist`` before it, came from ``(x, y) - object_plane_dist * (dx, dy)``. The plate kicks its slope by
    ``-d W / d(aperture position)``, the wave aberration ``W`` of ``coeffs`` at that object point
    (:func:`~temgym_core.aberrations.Seidel_object_pos_aperture_pos`), and adds ``-W`` to its path.

    The field terms (astigmatism ``C``, field curvature ``D``, distortion ``E``, coma ``F`` and their anisotropic
    forms) vanish for a beam from the axis, so a centred beam stays round. Spherical aberration ``B`` does not: it
    acts on every ray by its aperture position alone. Every kick is cubic in the coordinates, so the paraxial optics
    (A and B) are unchanged.

    Parameters
    ----------
    z : float
        Plane of the plate (m).
    object_plane_dist : float
        Distance from the object plane (the specimen) to the plate (m).
    coeffs : SeidelCoeffs
        The Seidel coefficients; for ``E``, ``F`` and ``C`` they are a slope (rad) per m cubed.
    """

    z: float
    object_plane_dist: float
    coeffs: SeidelCoeffs

    def _call_ray(self, ray: Ray):
        object_x = ray.x - self.object_plane_dist * ray.dx
        object_y = ray.y - self.object_plane_dist * ray.dy
        x, y, object_x, object_y = jnp.broadcast_arrays(
            *(jnp.asarray(value, jnp.float64) for value in (ray.x, ray.y, object_x, object_y)))
        points = jnp.stack((x, y, object_x, object_y), axis=-1)                   # aperture x, y, object x, y

        def path(point):
            return -Seidel_object_pos_aperture_pos(point[2], point[3], point[0], point[1], self.coeffs)

        value, gradient = jax.vmap(jax.value_and_grad(path))(points.reshape(-1, 4))
        value = value.reshape(x.shape)
        gradient = gradient.reshape(points.shape)           # only the aperture position's part kicks the slope
        return ray.derive(dx=ray.dx + gradient[..., 0], dy=ray.dy + gradient[..., 1],
                          pathlength=ray.pathlength + value)
