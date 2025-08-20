from typing import NamedTuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .ray import Ray


class BasePropagator:
    def __call__(self, ray: "Ray", distance: float) -> "Ray":
        raise NotImplementedError

    def with_distance(self, distance: float) -> "Propagator":
        return Propagator(
            distance, self
        )


class Propagator(NamedTuple):
    distance: float
    propagator: BasePropagator

    def __call__(self, ray: "Ray") -> "Ray":
        return self.propagator(ray, self.distance)


class FreeSpaceParaxial(BasePropagator):
    @staticmethod
    def propagate(ray: "Ray", distance: float):
        return ray.derive(
            x=ray.x + ray.dx * distance,
            y=ray.y + ray.dy * distance,
            z=ray.z * ray._one + distance,
            pathlength=ray.pathlength + distance,
        )

    @classmethod
    def __call__(cls, ray: "Ray", distance: float):
        return cls.propagate(ray, distance)


class FreeSpaceDirCosine(BasePropagator):
    @staticmethod
    def propagate(ray: "Ray", distance: float):
        # This method implements propagation using direction cosines
        # and should be accurate to higher angles, but needs modification
        # to work with the rest of jaxgym transfer matrices
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
