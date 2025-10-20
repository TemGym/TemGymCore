from typing import NamedTuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .ray import Ray


class BasePropagator:
    """Abstract base for ray propagators.

    Implement `__call__(ray, distance)` in subclasses to return a new ray.
    """
    def __call__(self, ray: "Ray", distance: float) -> "Ray":
        raise NotImplementedError

    def with_distance(self, distance: float) -> "Propagator":
        return Propagator(
            distance, self
        )


class Propagator(NamedTuple):
    """Callable pair of (distance, propagator) that acts on a ray.

    Parameters
    ----------
    distance : float
        Distance to propagate, metres.
    propagator : BasePropagator
        The underlying propagation model to use.

    Notes
    -----
    Calling `Propagator(ray)` returns the propagated ray.
    """
    distance: float
    propagator: BasePropagator

    def __call__(self, ray: "Ray") -> "Ray":
        return self.propagator(ray, self.distance)


class FreeSpaceParaxial(BasePropagator):
    """Paraxial free-space propagation with constant slopes.

    Notes
    -----
    Updates: x += dx*d, y += dy*d, z += d, pathlength += d. Pure and
    differentiable.
    """
    @staticmethod
    def propagate(ray: "Ray", distance: float):
        """Propagate a ray by `distance` assuming small angles.

        Parameters
        ----------
        ray : Ray
            Input ray.
        distance : float
            Propagation distance, metres.

        Returns
        -------
        ray : Ray
            Propagated ray.
        """
        return ray.derive(
            x=ray.x + ray.dx * distance,
            y=ray.y + ray.dy * distance,
            z=ray.z + distance,
            pathlength=ray.pathlength + distance,
        )

    @classmethod
    def __call__(cls, ray: "Ray", distance: float):
        return cls.propagate(ray, distance)


class FreeSpaceDirCosine(BasePropagator):
    """Free-space propagation using direction cosines (L, M, N).

    Notes
    -----
    Handles larger angles than paraxial model. Updates: x += (L/N)*d,
    y += (M/N)*d, z += d, pathlength += N*d. Not currently integrated with
    ABCD-based matrices in this repo.
    """
    @staticmethod
    def propagate(ray: "Ray", distance: float):
        """Propagate using direction cosines.

        Parameters
        ----------
        ray : Ray
            Input ray.
        distance : float
            Propagation distance, metres.

        Returns
        -------
        ray : Ray
            Propagated ray with direction-cosine model.
        """
        N = np.sqrt(1 + ray.dx**2 + ray.dy**2)
        L = ray.dx / N
        M = ray.dy / N
        return ray.derive(
            x=ray.x + L / N * distance,
            y=ray.y + M / N * distance,
            z=ray.z * ray._one + distance,
            pathlength=ray.pathlength + distance * N,
        )

    @classmethod
    def __call__(cls, ray: "Ray", distance: float):
        return cls.propagate(ray, distance)
