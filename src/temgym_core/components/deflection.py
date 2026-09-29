"""Deflection, rotation and scanning: deflectors, rotators, biprism kicks, scanners."""


from __future__ import annotations

import warnings

from typing import NamedTuple

import jax.numpy as jnp
import jax_dataclasses as jdc

from .. import Degrees
from ..gaussian import (
    FreeSpacePropagator,
    GaussianBeam,
)
from ..ray import Ray



from ._base import Component, _matvec, _rotate_quadratic_form


# Steering, rotation and scanning

@jdc.pytree_dataclass
class Deflector(Component):
    """Add a slope kick and its linear optical path; Gaussian shape is unchanged."""

    z: float
    def_x: float
    def_y: float

    def _call_ray(self, ray: Ray):
        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy
        return ray.derive(
            dx=dx + self.def_x * ray._one,
            dy=dy + self.def_y * ray._one,
            pathlength=ray.pathlength + self.def_x * x + self.def_y * y,
        )

    def _call_gaussian(self, ray: GaussianBeam):
        return self._call_ray(ray)


@jdc.pytree_dataclass
class DoubleDeflector(Component):
    """Two deflector kicks separated by an internal free-space spacing.

    User-facing commands (shift/tilt) are combined with hardware calibration
    ratios to produce the two physical kicks in each axis.
    ``include_axial_path=False`` omits the common straight-path phase, as in
    the Gaussian free-space propagator.
    """

    z: float
    spacing: float

    shift_x: float = 0.0
    shift_y: float = 0.0
    tilt_x: float = 0.0
    tilt_y: float = 0.0

    shift_balance_x: float = 1.0
    shift_balance_y: float = 1.0
    tilt_balance_x: float = 1.0
    tilt_balance_y: float = 1.0
    include_axial_path: jdc.Static[bool] = True

    @property
    def z_second(self) -> float:
        return self.z + self.spacing

    @property
    def def1_x(self):
        return self.shift_x + self.tilt_x

    @property
    def def1_y(self):
        return self.shift_y + self.tilt_y

    @property
    def def2_x(self):
        return (
            -self.shift_balance_x * self.shift_x
            -self.tilt_balance_x * self.tilt_x
        )

    @property
    def def2_y(self):
        return (
            -self.shift_balance_y * self.shift_y
            -self.tilt_balance_y * self.tilt_y
        )

    @property
    def deflectors(self):
        """The two physical kicks, including the shift/tilt balance mixing."""
        return (
            Deflector(self.z, self.def1_x, self.def1_y),
            Deflector(self.z_second, self.def2_x, self.def2_y),
        )

    def _call_ray(self, ray: Ray):
        from ..propagator import FreeSpaceParaxial

        first, second = self.deflectors
        ray = FreeSpaceParaxial()(first(ray), self.spacing)
        if not self.include_axial_path:
            ray = ray.derive(pathlength=ray.pathlength - self.spacing)
        return second(ray)

    def _call_gaussian(self, ray: GaussianBeam):
        first, second = self.deflectors
        ray = FreeSpacePropagator(include_axial_path=self.include_axial_path)(
            first(ray), self.spacing
        )
        return second(ray)


@jdc.pytree_dataclass
class Rotator(Component):
    z: float
    angle: Degrees

    def _rotation(self):
        angle = jnp.deg2rad(self.angle)
        cos_a = jnp.cos(angle)
        sin_a = jnp.sin(angle)
        return cos_a, sin_a

    def _call_ray(self, ray: Ray):
        cos_a, sin_a = self._rotation()
        new_x = ray.x * cos_a - ray.y * sin_a
        new_y = ray.x * sin_a + ray.y * cos_a
        new_dx = ray.dx * cos_a - ray.dy * sin_a
        new_dy = ray.dx * sin_a + ray.dy * cos_a
        return ray.derive(x=new_x, y=new_y, dx=new_dx, dy=new_dy)

    def _call_gaussian(self, ray: GaussianBeam):
        cos_a, sin_a = self._rotation()
        R = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=jnp.float64)
        r_xy_rot = _matvec(R, ray.r_xy)
        d_xy_rot = _matvec(R, ray.d_xy)
        Q_rot = _rotate_quadratic_form(R, ray.Q_inv)

        return ray.derive(
            x=r_xy_rot[..., 0],
            y=r_xy_rot[..., 1],
            dx=d_xy_rot[..., 0],
            dy=d_xy_rot[..., 1],
            Q_inv=Q_rot,
        )


@jdc.pytree_dataclass
class DeflectionBiprism(Component):
    z: float
    offset: float = 0.0
    rotation: Degrees = 0.0
    def_x: float = 0.0
    side: int = 1

    def _call_ray(self, ray: Ray):
        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy
        return ray.derive(
            dx=dx + self.def_x * ray._one * jnp.sign(ray.x),
            dy=dy,
            pathlength=ray.pathlength + dx * x + dy * y,
        )

    def _call_gaussian(self, ray: GaussianBeam):
        return self._call_ray(ray)


@jdc.pytree_dataclass
class Biprism(DeflectionBiprism):
    def __post_init__(self):
        warnings.warn(
            "`components.Biprism` is deprecated and will remain as an alias for "
            "`DeflectionBiprism`. Prefer `DeflectionBiprism` for clarity.",
            DeprecationWarning,
            stacklevel=2,
        )


class DescanError(NamedTuple):
    """Affine position/slope response and offsets used by Descanner."""

    pxo_pxi: float = 0.0
    pxo_pyi: float = 0.0
    pyo_pxi: float = 0.0
    pyo_pyi: float = 0.0
    sxo_pxi: float = 0.0
    sxo_pyi: float = 0.0
    syo_pxi: float = 0.0
    syo_pyi: float = 0.0
    offpxi: float = 0.0
    offpyi: float = 0.0
    offsxi: float = 0.0
    offsyi: float = 0.0

    def as_array(self) -> jnp.ndarray:
        return jnp.array(self)

    def as_matrix(self) -> jnp.ndarray:
        return jnp.array(
            [
                [self.pxo_pxi, self.pxo_pyi, 0.0, 0.0, self.offpxi],
                [self.pyo_pxi, self.pyo_pyi, 0.0, 0.0, self.offpyi],
                [self.sxo_pxi, self.sxo_pyi, 0.0, 0.0, self.offsxi],
                [self.syo_pxi, self.syo_pyi, 0.0, 0.0, self.offsyi],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )


@jdc.pytree_dataclass
class Scanner(Component):
    """Add calibrated position and slope offsets; Gaussian shape is unchanged."""

    z: float
    scan_pos_x: float
    scan_pos_y: float
    scan_tilt_x: float = 0.0
    scan_tilt_y: float = 0.0

    def _call_ray(self, ray: Ray):
        return ray.derive(
            x=ray.x + self.scan_pos_x * ray._one,
            y=ray.y + self.scan_pos_y * ray._one,
            dx=ray.dx + self.scan_tilt_x * ray._one,
            dy=ray.dy + self.scan_tilt_y * ray._one,
        )

    def _call_gaussian(self, ray: GaussianBeam):
        return self._call_ray(ray)


@jdc.pytree_dataclass
class Descanner(Component):
    """Undo scan offsets, allowing the affine errors in descan_error."""

    z: float
    scan_pos_x: float
    scan_pos_y: float
    scan_tilt_x: float = 0.0
    scan_tilt_y: float = 0.0
    descan_error: DescanError = DescanError()

    def _offsets(self):
        de = self.descan_error
        sp_x, sp_y = self.scan_pos_x, self.scan_pos_y
        st_x, st_y = self.scan_tilt_x, self.scan_tilt_y
        return (
            sp_x * de.pxo_pxi + sp_y * de.pxo_pyi + de.offpxi - sp_x,
            sp_x * de.pyo_pxi + sp_y * de.pyo_pyi + de.offpyi - sp_y,
            sp_x * de.sxo_pxi + sp_y * de.sxo_pyi + de.offsxi - st_x,
            sp_x * de.syo_pxi + sp_y * de.syo_pyi + de.offsyi - st_y,
        )

    def _call_ray(self, ray: Ray):
        ox, oy, odx, ody = self._offsets()
        return ray.derive(
            x=ray.x + ox * ray._one,
            y=ray.y + oy * ray._one,
            dx=ray.dx + odx * ray._one,
            dy=ray.dy + ody * ray._one,
        )

    def _call_gaussian(self, ray: GaussianBeam):
        return self._call_ray(ray)
