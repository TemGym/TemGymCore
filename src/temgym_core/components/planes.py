"""Planes and detectors."""


from __future__ import annotations



import jax_dataclasses as jdc

from .. import CoordsXY, Degrees, ScaleYX, ShapeYX
from ..gaussian import (
    GaussianBeam,
)
from ..grid import Grid
from ..ray import Ray



from ._base import Component


# Planes and detectors

@jdc.pytree_dataclass
class Plane(Component):
    """A marker plane; propagation to z is performed by the runner."""

    z: float

    def _call_ray(self, ray: Ray):
        return ray

    def _call_gaussian(self, ray: GaussianBeam):
        return ray


@jdc.pytree_dataclass
class ScanGrid(Component, Grid):
    """Scan-plane geometry; calling it leaves the beam unchanged."""

    z: float
    pixel_size: ScaleYX
    shape: ShapeYX
    rotation: Degrees = 0.0
    centre: CoordsXY = (0.0, 0)
    flip_y: bool = False

    def _call_ray(self, ray: Ray):
        return ray

    def _call_gaussian(self, ray: GaussianBeam):
        return ray


@jdc.pytree_dataclass
class Detector(Component, Grid):
    """Detector geometry. Calling it marks a plane; rendering is separate."""

    z: float
    pixel_size: ScaleYX
    shape: ShapeYX
    rotation: Degrees = 0.0
    centre: CoordsXY = (0.0, 0)
    flip_y: bool = False

    def _call_ray(self, ray: Ray):
        return ray

    def _call_gaussian(self, ray: GaussianBeam):
        return ray
