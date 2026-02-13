from __future__ import annotations

import dataclasses
import warnings
from typing import Any, Dict, NamedTuple

import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax_dataclasses as jdc
from interpax import Interpolator2D, Interpolator3D
from jax import lax
from jax.nn import softplus

from . import CoordsXY, Degrees, ScaleYX, ShapeYX
from .gaussian import (
    FreeSpacePropagator,
    GaussianBeam,
    apply_action_delta,
    taylor_expand,
)
from .aberrations import (
    KrivanekCoeffs,
    SeidelCoeffs,
    Seidel_aperture_pos_aperture_slope,
    W_krivanek,
    grad_W_krivanek,
)
from .grid import Grid
from .potential import potential_smoothed
from .ray import Ray
from .tree_utils import HasParamsMixin


class Component(HasParamsMixin):
    """Unified base component for paraxial rays and gaussian beams."""

    def __call__(self, ray: Ray) -> Ray:
        family = getattr(ray, "ray_family", "ray")
        if family == "gaussian":
            return self._call_gaussian(ray)
        if family == "ray":
            return self._call_ray(ray)
        raise TypeError(
            f"Unsupported ray family '{family}' for {type(self).__name__}."
        )

    def _call_ray(self, ray: Ray) -> Ray:
        raise NotImplementedError(
            f"{type(self).__name__} is not implemented for ray_family='ray'."
        )

    def _call_gaussian(self, ray: GaussianBeam) -> GaussianBeam:
        raise NotImplementedError(
            f"{type(self).__name__} is not implemented for ray_family='gaussian'."
        )


class GaussianActionComponent(Component):
    """Mixin for components defined by an action (phase + log-transmission)."""

    def phase_shift(self, xy: jnp.ndarray):
        return 0.0

    def log_transmission(self, xy: jnp.ndarray):
        return 0.0

    def complex_action(self, xy: jnp.ndarray, k: float) -> complex:
        logA = self.log_transmission(xy)
        L = jnp.logaddexp(logA, -50)
        return self.phase_shift(xy) - 1j * (L / k)

    def _call_gaussian(self, ray: GaussianBeam) -> GaussianBeam:
        xy_ref = ray.r_xy
        dS0, dS1, dS2 = taylor_expand(self.complex_action, xy_ref, ray.k)
        r_xy, d_xy, amplitude, pathlength, Q_new = apply_action_delta(
            ray, dS0=dS0, dS1=dS1, dS2=dS2
        )
        return ray.derive(
            x=r_xy[0],
            y=r_xy[1],
            dx=d_xy[0],
            dy=d_xy[1],
            z=ray.z,
            amplitude=amplitude,
            pathlength=pathlength,
            Q_inv=Q_new,
        )


class DescanError(NamedTuple):
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
class Plane(Component):
    z: float

    def _call_ray(self, ray: Ray):
        return ray

    def _call_gaussian(self, ray: GaussianBeam):
        return ray


@jdc.pytree_dataclass
class Lens(GaussianActionComponent):
    z: float
    focal_length: float
    x0: float = 0.0
    y0: float = 0.0

    def _call_ray(self, ray: Ray):
        f = self.focal_length
        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy
        new_dx = -x / f + dx
        new_dy = -y / f + dy
        pathlength = ray.pathlength - (x**2 + y**2) / (2 * f)
        return ray.derive(dx=new_dx, dy=new_dy, pathlength=pathlength)

    def phase_shift(self, xy: jnp.ndarray):
        x, y = xy[0] - self.x0, xy[1] - self.y0
        rho2 = x * x + y * y
        return -0.5 * rho2 / self.focal_length


@jdc.pytree_dataclass
class KrivanekLens(Lens):
    coeffs: Dict = dataclasses.field(default_factory=dict)
    axis_eps: float = 1e-24

    def _call_ray(self, ray: Ray):
        f = self.focal_length
        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy

        ideal_dx = -x / f + dx
        ideal_dy = -y / f + dy

        alpha = jnp.hypot(ideal_dx, ideal_dy)
        phi = jnp.arctan2(ideal_dy, ideal_dx)

        dWx, dWy = grad_W_krivanek(ideal_dx, ideal_dy, self.coeffs)
        dux, duy = -dWx / f, -dWy / f

        aber_dx = ideal_dx + dux
        aber_dy = ideal_dy + duy
        pathlength = W_krivanek(alpha, phi, self.coeffs)

        return ray.derive(dx=aber_dx, dy=aber_dy, pathlength=pathlength)

    def phase_shift(self, xy: jnp.ndarray):
        x = xy[0] - self.x0
        y = xy[1] - self.y0
        f = self.focal_length
        rho2 = x * x + y * y

        def _with_aberrations(_):
            rho = jnp.sqrt(rho2)
            phi = jnp.arctan2(y, x)
            alpha = rho / f
            return -0.5 * rho2 / f - W_krivanek(alpha, phi, self.coeffs)

        def _on_axis(_):
            return -0.5 * rho2 / f

        return lax.cond(rho2 > self.axis_eps, _with_aberrations, _on_axis, operand=None)


@jdc.pytree_dataclass
class AberratedLensKrivanek(KrivanekLens):
    def __post_init__(self):
        warnings.warn(
            "`AberratedLensKrivanek` is deprecated; use `KrivanekLens`.",
            DeprecationWarning,
            stacklevel=2,
        )


@jdc.pytree_dataclass
class SeidelLens(Lens):
    object_plane_dist: float = 0.0
    coeffs: SeidelCoeffs = dataclasses.field(default_factory=SeidelCoeffs)

    def _call_ray(self, ray: Ray):
        raise NotImplementedError(
            "SeidelLens is not implemented for singular rays. Use ray_family='gaussian'."
        )

    def log_transmission(self, xy):
        return 0.0

    def phase_shift(self, xy, dxy):
        xy = jnp.asarray(xy)
        dxy = jnp.asarray(dxy)
        x_a, y_a = xy[..., 0], xy[..., 1]
        x_ap, y_ap = dxy[..., 0], dxy[..., 1]
        rho2 = x_a * x_a + y_a * y_a
        return -0.5 * rho2 / self.focal_length - Seidel_aperture_pos_aperture_slope(
            x_a,
            y_a,
            x_ap,
            y_ap,
            self.object_plane_dist,
            self.coeffs,
        )

    def complex_action(self, xy: jnp.ndarray, dxy: jnp.ndarray, k: float) -> complex:
        logA = self.log_transmission(xy)
        L = jnp.logaddexp(logA, -50)
        return self.phase_shift(xy, dxy) - 1j * (L / k)

    def _call_gaussian(self, ray: GaussianBeam) -> GaussianBeam:
        dS0, dS1, dS2 = taylor_expand(
            self.complex_action,
            ray.r_xy,
            ray.d_xy,
            ray.k,
        )
        r_xy, d_xy, amplitude, pathlength, Q_new = apply_action_delta(
            ray,
            dS0=dS0,
            dS1=dS1,
            dS2=dS2,
        )
        return ray.derive(
            x=r_xy[0],
            y=r_xy[1],
            dx=d_xy[0],
            dy=d_xy[1],
            z=ray.z,
            amplitude=amplitude,
            pathlength=pathlength,
            Q_inv=Q_new,
        )


@jdc.pytree_dataclass
class DistortedLens(SeidelLens):
    IsoDist: float = 0.0
    AnisoDist: float = 0.0

    def phase_shift(self, xy, dxy):
        xy = jnp.asarray(xy)
        dxy = jnp.asarray(dxy)
        x_a, y_a = xy[..., 0], xy[..., 1]
        x_ap, y_ap = dxy[..., 0], dxy[..., 1]

        rho2 = x_a * x_a + y_a * y_a
        coeffs = SeidelCoeffs(E=self.IsoDist, e=self.AnisoDist)
        return -0.5 * rho2 / self.focal_length - Seidel_aperture_pos_aperture_slope(
            x_a,
            y_a,
            x_ap,
            y_ap,
            self.object_plane_dist,
            coeffs,
        )


@jdc.pytree_dataclass
class ScanGrid(Component, Grid):
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
class Scanner(Component):
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
        return ray.derive(
            x=ray.x + self.scan_pos_x * ray._one,
            y=ray.y + self.scan_pos_y * ray._one,
            dx=ray.dx + self.scan_tilt_x * ray._one,
            dy=ray.dy + self.scan_tilt_y * ray._one,
        )


@jdc.pytree_dataclass
class Descanner(Component):
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
        ox, oy, odx, ody = self._offsets()
        return ray.derive(
            x=ray.x + ox * ray._one,
            y=ray.y + oy * ray._one,
            dx=ray.dx + odx * ray._one,
            dy=ray.dy + ody * ray._one,
        )


@jdc.pytree_dataclass
class Detector(Component, Grid):
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
class ThickLens(Component):
    z_po: float
    z_pi: float
    focal_length: float

    def _call_ray(self, ray: Ray):
        f = self.focal_length
        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy

        new_dx = -x / f + dx
        new_dy = -y / f + dy
        pathlength = ray.pathlength - (x**2 + y**2) / (2 * f)
        new_z = ray.z - (self.z_po - self.z_pi)

        return ray.derive(dx=new_dx, dy=new_dy, pathlength=pathlength, z=new_z)

    def _call_gaussian(self, ray: GaussianBeam):
        raise NotImplementedError(
            "ThickLens is not implemented for gaussian beams."
        )

    @property
    def z(self):
        return self.z_po


@jdc.pytree_dataclass
class Deflector(Component):
    z: float
    def_x: float
    def_y: float

    def _call_ray(self, ray: Ray):
        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy
        return ray.derive(
            dx=dx + self.def_x * ray._one,
            dy=dy + self.def_y * ray._one,
            pathlength=ray.pathlength + dx * x + dy * y,
        )

    def _call_gaussian(self, ray: GaussianBeam):
        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy
        return ray.derive(
            dx=dx + self.def_x * ray._one,
            dy=dy + self.def_y * ray._one,
            pathlength=ray.pathlength + dx * x + dy * y,
        )


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

        r_xy_rot = R @ ray.r_xy
        d_xy_rot = R @ ray.d_xy
        Q_rot = R.T @ ray.Q_inv @ R

        return ray.derive(
            x=r_xy_rot[0],
            y=r_xy_rot[1],
            dx=d_xy_rot[0],
            dy=d_xy_rot[1],
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
        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy
        return ray.derive(
            dx=dx + self.def_x * ray._one * jnp.sign(ray.x),
            dy=dy,
            pathlength=ray.pathlength + dx * x + dy * y,
        )


@jdc.pytree_dataclass
class Biprism(DeflectionBiprism):
    def __post_init__(self):
        warnings.warn(
            "`components.Biprism` is deprecated and will remain as an alias for "
            "`DeflectionBiprism`. Prefer `DeflectionBiprism` for clarity.",
            DeprecationWarning,
            stacklevel=2,
        )


@jdc.pytree_dataclass
class RotatingLens(Component):
    z: float
    focal_length: float
    rotation: Degrees
    x0: float = 0.0
    y0: float = 0.0

    def _call_ray(self, ray: Ray):
        rotated_ray = Rotator(z=self.z, angle=self.rotation)(ray)
        return Lens(z=self.z, focal_length=self.focal_length, x0=self.x0, y0=self.y0)(
            rotated_ray
        )

    def _call_gaussian(self, ray: GaussianBeam):
        rotated_ray = Rotator(z=self.z, angle=self.rotation)(ray)
        return Lens(z=self.z, focal_length=self.focal_length, x0=self.x0, y0=self.y0)(
            rotated_ray
        )


@jdc.pytree_dataclass
class ElectromagneticLens(GaussianActionComponent):
    z: float
    I0: float
    Cf: float
    Kv: float
    x0: float = 0.0
    y0: float = 0.0

    @property
    def focal_length(self) -> float:
        return 1.0 / (self.Cf * self.I0**2)

    @property
    def rotation_angle(self) -> float:
        return self.Kv * self.I0

    def phase_shift(self, xy: jnp.ndarray) -> float:
        x, y = xy[0] - self.x0, xy[1] - self.y0
        rho2 = x * x + y * y
        return -0.5 * rho2 / self.focal_length

    def _call_ray(self, ray: Ray) -> Ray:
        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy
        f = self.focal_length

        new_dx = -x / f + dx
        new_dy = -y / f + dy
        pathlength = ray.pathlength - (x**2 + y**2) / (2.0 * f)

        angle = self.rotation_angle
        cos_a = jnp.cos(angle)
        sin_a = jnp.sin(angle)

        rot_x = cos_a * x - sin_a * y
        rot_y = sin_a * x + cos_a * y
        rot_dx = cos_a * new_dx - sin_a * new_dy
        rot_dy = sin_a * new_dx + cos_a * new_dy

        return ray.derive(
            x=rot_x,
            y=rot_y,
            dx=rot_dx,
            dy=rot_dy,
            pathlength=pathlength,
        )

    def _call_gaussian(self, ray: GaussianBeam) -> GaussianBeam:
        out = GaussianActionComponent._call_gaussian(self, ray)

        angle = self.rotation_angle
        cos_a = jnp.cos(angle)
        sin_a = jnp.sin(angle)
        R = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=jnp.float64)

        r_xy_rot = R @ out.r_xy
        d_xy_rot = R @ out.d_xy
        Q_rot = R.T @ out.Q_inv @ R

        return out.derive(
            x=r_xy_rot[0],
            y=r_xy_rot[1],
            dx=d_xy_rot[0],
            dy=d_xy_rot[1],
            Q_inv=Q_rot,
        )


@jdc.pytree_dataclass(kw_only=True)
class ABCDTransfer(Component):
    A: jnp.ndarray
    B: jnp.ndarray
    C: jnp.ndarray = dataclasses.field(
        default_factory=lambda: jnp.zeros((2, 2), dtype=jnp.float64)
    )
    D: jnp.ndarray = dataclasses.field(
        default_factory=lambda: jnp.eye(2, dtype=jnp.float64)
    )
    z: float = 0.0

    def _call_ray(self, ray: Ray):
        A = jnp.asarray(self.A, dtype=jnp.float64)
        B = jnp.asarray(self.B, dtype=jnp.float64)
        C = jnp.asarray(self.C, dtype=jnp.float64)
        D = jnp.asarray(self.D, dtype=jnp.float64)

        def matvec(m, v):
            return jnp.einsum("ij,...j->...i", m, v)

        r_xy = ray.r_xy
        d_xy = ray.d_xy

        r_xy_new = matvec(A, r_xy) + matvec(B, d_xy)
        d_xy_new = matvec(C, r_xy) + matvec(D, d_xy)

        return ray.derive(
            x=r_xy_new[..., 0],
            y=r_xy_new[..., 1],
            dx=d_xy_new[..., 0],
            dy=d_xy_new[..., 1],
        )

    def _call_gaussian(self, ray: GaussianBeam) -> GaussianBeam:
        A = jnp.asarray(self.A, dtype=jnp.float64)
        B = jnp.asarray(self.B, dtype=jnp.float64)
        C = jnp.asarray(self.C, dtype=jnp.float64)
        D = jnp.asarray(self.D, dtype=jnp.float64)

        def matvec(m, v):
            return jnp.einsum("ij,...j->...i", m, v)

        def matmat(m, x):
            return jnp.einsum("ij,...jk->...ik", m, x)

        r_xy = ray.r_xy
        d_xy = ray.d_xy

        r_xy_new = matvec(A, r_xy) + matvec(B, d_xy)
        d_xy_new = matvec(C, r_xy) + matvec(D, d_xy)

        Q = ray.Q_inv
        denom = A + matmat(B, Q)
        numer = C + matmat(D, Q)
        eye = jnp.eye(2, dtype=jnp.complex128)
        inv_denom = jnp.linalg.solve(jnp.swapaxes(denom, -1, -2), eye)
        inv_denom = jnp.swapaxes(inv_denom, -1, -2)
        Q_new = jnp.einsum("...ij,...jk->...ik", numer, inv_denom)

        return ray.derive(
            x=r_xy_new[..., 0],
            y=r_xy_new[..., 1],
            dx=d_xy_new[..., 0],
            dy=d_xy_new[..., 1],
            Q_inv=Q_new,
        )


@jdc.pytree_dataclass(kw_only=True)
class SigmoidAperture(GaussianActionComponent):
    radius: float = 1.0
    edge_width: float = 0.5
    sharpness: float = 1.0
    t_low: float = 0.0
    t_high: float = 1.0
    x0: float = 0.0
    y0: float = 0.0
    eps: float = 1e-15
    z: float = 0.0

    def _call_ray(self, ray: Ray):
        raise NotImplementedError(
            "SigmoidAperture is only implemented for gaussian beams."
        )

    def phase_shift(self, xy):
        return 0.0

    def log_transmission(self, xy):
        x, y = xy[0] - self.x0, xy[1] - self.y0
        rho = jnp.sqrt(x * x + y * y + self.eps * self.eps) - self.eps
        w = jnp.maximum(jnp.abs(self.edge_width), self.eps)
        s = jnn.sigmoid(self.sharpness * (rho - self.radius) / w)
        t = self.t_high - (self.t_high - self.t_low) * s
        t_clamped = jnp.clip(t, self.eps, None)
        return jnp.log(t_clamped)


@jdc.pytree_dataclass(kw_only=True)
class PhaseBiprism(GaussianActionComponent):
    strength: float
    width: float
    length: float | None = None
    theta: float = 0.0
    x0: float = 0.0
    y0: float = 0.0
    sharpness: float = 50.0
    eps: float = 1e-15
    z: float = 0.0

    def _call_ray(self, ray: Ray):
        raise NotImplementedError(
            "PhaseBiprism is only implemented for gaussian beams."
        )

    def _uv(self, xy: jnp.ndarray):
        x, y = xy[0], xy[1]
        xr, yr = x - self.x0, y - self.y0
        c, s = jnp.cos(self.theta), jnp.sin(self.theta)
        u = c * xr + s * yr
        v = -s * xr + c * yr
        return u, v

    def phase_shift(self, xy: jnp.ndarray):
        u, _ = self._uv(xy)
        hu = 0.5 * self.width
        eps_u = self.eps * hu
        au = jnp.sqrt(u * u + eps_u * eps_u)
        return -self.strength * au

    def log_transmission(self, xy: jnp.ndarray):
        u, v = self._uv(xy)

        hu = 0.5 * self.width
        eps_u = self.eps * hu
        au = jnp.sqrt(u * u + eps_u * eps_u)
        tx = self.sharpness * (au - hu)
        logA_u = -softplus(-tx)
        if self.length is None:
            # Avoid tracing the "with_length" path when length is None.
            # lax.cond traces both branches and would attempt jnp.asarray(None).
            logA_v = jnp.asarray(0.0)
        else:
            length = jnp.asarray(self.length)
            hv = 0.5 * length
            eps_v = self.eps * hu
            av = jnp.sqrt(v * v + eps_v * eps_v)
            ty = self.sharpness * (av - hv)
            logA_v = -softplus(-ty)
        return logA_u + logA_v


@jdc.pytree_dataclass(kw_only=True)
class ConstantPhaseShift(GaussianActionComponent):
    constant_phase_shift: float
    z: float = 0.0

    def _call_ray(self, ray: Ray):
        raise NotImplementedError(
            "ConstantPhaseShift is only implemented for gaussian beams."
        )

    def phase_shift(self, xy: jnp.ndarray):
        return self.constant_phase_shift


@jdc.pytree_dataclass(kw_only=True)
class LinearPhaseShift(GaussianActionComponent):
    linear_phase_shift: jnp.ndarray
    z: float = 0.0

    def _call_ray(self, ray: Ray):
        raise NotImplementedError(
            "LinearPhaseShift is only implemented for gaussian beams."
        )

    def phase_shift(self, xy: jnp.ndarray):
        return jnp.dot(self.linear_phase_shift, xy)


@jdc.pytree_dataclass(kw_only=True)
class QuadraticPhaseShift(GaussianActionComponent):
    quadratic_phase_shift: jnp.ndarray
    z: float = 0.0

    def _call_ray(self, ray: Ray):
        raise NotImplementedError(
            "QuadraticPhaseShift is only implemented for gaussian beams."
        )

    def phase_shift(self, xy: jnp.ndarray):
        return 0.5 * xy @ self.quadratic_phase_shift @ xy


@jdc.pytree_dataclass(kw_only=True)
class ConstantAmplitudeShift(GaussianActionComponent):
    amplitude: float
    z: float = 0.0

    def _call_ray(self, ray: Ray):
        raise NotImplementedError(
            "ConstantAmplitudeShift is only implemented for gaussian beams."
        )

    def log_transmission(self, xy: jnp.ndarray):
        return jnp.log(self.amplitude)


@jdc.pytree_dataclass(kw_only=True)
class LinearAmplitudeShift(GaussianActionComponent):
    linear_log_amplitude: jnp.ndarray | None = None
    linear_amplitude: jnp.ndarray | None = None
    z: float = 0.0

    def _call_ray(self, ray: Ray):
        raise NotImplementedError(
            "LinearAmplitudeShift is only implemented for gaussian beams."
        )

    def log_transmission(self, xy: jnp.ndarray):
        coeffs = self.linear_log_amplitude
        if coeffs is None:
            coeffs = self.linear_amplitude
        if coeffs is None:
            raise ValueError("LinearAmplitudeShift requires linear_log_amplitude.")
        return jnp.dot(coeffs, xy)


@jdc.pytree_dataclass(kw_only=True)
class QuadraticAmplitudeShift(GaussianActionComponent):
    quadratic_log_amplitude: jnp.ndarray
    z: float = 0.0

    def _call_ray(self, ray: Ray):
        raise NotImplementedError(
            "QuadraticAmplitudeShift is only implemented for gaussian beams."
        )

    def log_transmission(self, xy: jnp.ndarray):
        return 0.5 * xy @ self.quadratic_log_amplitude @ xy


@jdc.pytree_dataclass(kw_only=True)
class MagneticPhaseSample(GaussianActionComponent):
    strength: float
    width: float
    height: float
    x0: float = 0.0
    y0: float = 0.0
    theta: float = 0.0
    edge_sharpness: float = 5e6
    modulation_strength: float = 0.3
    skew_strength: float = 0.2
    radial_strength: float = 0.15
    eps: float = 1e-9
    z: float = 0.0

    def _call_ray(self, ray: Ray):
        raise NotImplementedError(
            "MagneticPhaseSample is only implemented for gaussian beams."
        )

    def _local_coords(self, xy):
        x = xy[0] - self.x0
        y = xy[1] - self.y0
        c = jnp.cos(self.theta)
        s = jnp.sin(self.theta)
        u = c * x + s * y
        v = -s * x + c * y
        return u, v

    def _soft_indicator(self, coord, half_extent):
        sharp = self.edge_sharpness
        pos = jax.nn.sigmoid(sharp * (coord + half_extent))
        neg = jax.nn.sigmoid(sharp * (coord - half_extent))
        plateau = jax.nn.sigmoid(sharp * half_extent) - jax.nn.sigmoid(-sharp * half_extent)
        plateau = jnp.maximum(plateau, 1e-9)
        return (pos - neg) / plateau

    def phase_shift(self, xy):
        u, v = self._local_coords(xy)

        hx = 0.5 * self.width
        hy = 0.5 * self.height

        mask = self._soft_indicator(u, hx) * self._soft_indicator(v, hy)

        u_norm = u / (hx + self.eps)
        v_norm = v / (hy + self.eps)
        radial = jnp.sqrt(u_norm * u_norm + v_norm * v_norm + self.eps)

        texture = jnp.sin(jnp.pi * u_norm) * jnp.cos(jnp.pi * v_norm)
        skew = u_norm * v_norm
        radial_term = radial - 0.5

        profile = (
            1.0
            + self.modulation_strength * texture
            + self.skew_strength * skew
            + self.radial_strength * radial_term
        )

        return self.strength * mask * profile


@jdc.pytree_dataclass(kw_only=True)
class RandomPhaseSample(GaussianActionComponent):
    strength: float
    width: float
    height: float
    x0: float = 0.0
    y0: float = 0.0
    theta: float = 0.0
    edge_sharpness: float = 5e6
    correlation_length: float = 2e-9
    eps: float = 1e-9
    z: float = 0.0

    def _call_ray(self, ray: Ray):
        raise NotImplementedError(
            "RandomPhaseSample is only implemented for gaussian beams."
        )

    def _local_coords(self, xy):
        x = xy[0] - self.x0
        y = xy[1] - self.y0
        c = jnp.cos(self.theta)
        s = jnp.sin(self.theta)
        u = c * x + s * y
        v = -s * x + c * y
        return u, v

    def _soft_indicator(self, coord, half_extent):
        sharp = self.edge_sharpness
        pos = jax.nn.sigmoid(sharp * (coord + half_extent))
        neg = jax.nn.sigmoid(sharp * (coord - half_extent))
        plateau = jax.nn.sigmoid(sharp * half_extent) - jax.nn.sigmoid(-sharp * half_extent)
        plateau = jnp.maximum(plateau, 1e-9)
        return (pos - neg) / plateau

    def _hash(self, i, j):
        return jnp.mod(jnp.sin(127.1 * i + 311.7 * j) * 43758.5453, 1.0)

    def _value_noise(self, x, y):
        xi = jnp.floor(x)
        yi = jnp.floor(y)
        xf = x - xi
        yf = y - yi

        n00 = self._hash(xi, yi)
        n10 = self._hash(xi + 1, yi)
        n01 = self._hash(xi, yi + 1)
        n11 = self._hash(xi + 1, yi + 1)

        def fade(t):
            return t * t * (3.0 - 2.0 * t)

        u = fade(xf)
        v = fade(yf)

        nx0 = n00 + u * (n10 - n00)
        nx1 = n01 + u * (n11 - n01)
        return nx0 + v * (nx1 - nx0)

    def phase_shift(self, xy):
        u, v = self._local_coords(xy)

        hx = 0.5 * self.width
        hy = 0.5 * self.height
        mask = self._soft_indicator(u, hx) * self._soft_indicator(v, hy)

        corr = self.correlation_length + self.eps
        xn = u / corr
        yn = v / corr

        raw = self._value_noise(xn, yn)
        noise = (raw - 0.5) * 2.0

        raw2 = self._value_noise(xn * 2.0, yn * 2.0)
        noise2 = (raw2 - 0.5) * 2.0
        combined = 0.7 * noise + 0.3 * noise2

        return self.strength * mask * combined


@jdc.pytree_dataclass(kw_only=True)
class InterpolatedSample2D(GaussianActionComponent):
    interpolator: Interpolator2D
    method: jdc.Static[str] = "catmull-rom"
    z: float = 0.0

    def _call_ray(self, ray: Ray):
        raise NotImplementedError(
            "InterpolatedSample2D is only implemented for gaussian beams."
        )

    @classmethod
    def from_array(cls, sample, x_coords, y_coords, extrap=1.0, z=0.0, method="cubic"):
        interpolator = Interpolator2D(
            x=x_coords,
            y=y_coords,
            f=sample,
            method=method,
            extrap=extrap,
        )
        return cls(z=z, interpolator=interpolator, method=method)

    def evaluate_complex(self, xy):
        return self.interpolator(xy[0], xy[1])

    def phase_shift(self, xy):
        z = self.evaluate_complex(xy)
        return jnp.angle(z)

    def log_transmission(self, xy):
        z = self.evaluate_complex(xy)
        amp = jnp.abs(z)
        amp_clamped = jnp.maximum(amp, 1e-15)
        return jnp.where(amp < 1e-15, 0.0, jnp.log(amp_clamped))


@jdc.pytree_dataclass(kw_only=True)
class InterpolatedFields3D(Component):
    interpolator: Interpolator3D
    method: jdc.Static[str] = "catmull-rom"
    z: float = 0.0

    @classmethod
    def from_array(cls, fields, x_coords, y_coords, z_coords, *, method="cubic"):
        interpolator = Interpolator3D(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            f=fields,
            method=method,
            extrap=0.0,
        )
        return cls(interpolator=interpolator, method=method)

    def evaluate(self, xyz):
        x, y, z = xyz
        return self.interpolator(x, y, z)

    def _call_ray(self, ray: Ray):
        return ray

    def _call_gaussian(self, ray: GaussianBeam):
        return ray


@jdc.pytree_dataclass(kw_only=True)
class AtomicPotential(Component):
    atom_xyz: jnp.ndarray
    element_params: jnp.ndarray
    cutoff_radius: float
    z: float = 0.0

    def _call_ray(self, ray: Ray):
        raise NotImplementedError(
            "AtomicPotential is only implemented for gaussian beams."
        )

    def log_transmission(self, xy):
        return 0.0

    def phase_shift(self, xy: jnp.ndarray, z, sigma, k: float) -> complex:
        x, y = xy[0], xy[1]
        r = jnp.sqrt(
            (x - self.atom_xyz[0]) ** 2
            + (y - self.atom_xyz[1]) ** 2
            + (z - self.atom_xyz[2]) ** 2
        )
        V = potential_smoothed(r, self.element_params, self.cutoff_radius)
        interaction_constant = sigma / k
        return -interaction_constant * V

    def complex_action(self, xy: jnp.ndarray, z: jnp.ndarray, sigma, k: float) -> complex:
        logA = self.log_transmission(xy)
        L = jnp.logaddexp(logA, -50)
        return self.phase_shift(xy, z, sigma, k) - 1j * (L / k)

    def _call_gaussian(self, ray: GaussianBeam) -> GaussianBeam:
        dS0, dS1, dS2 = taylor_expand(
            self.complex_action,
            ray.r_xy,
            ray.z,
            ray.sigma,
            ray.k,
        )

        r_xy, d_xy, amplitude, pathlength, Q_new = apply_action_delta(
            ray,
            dS0=dS0,
            dS1=dS1,
            dS2=dS2,
        )

        return ray.derive(
            x=r_xy[0],
            y=r_xy[1],
            dx=d_xy[0],
            dy=d_xy[1],
            z=ray.z,
            amplitude=amplitude,
            pathlength=pathlength,
            Q_inv=Q_new,
        )


@jdc.pytree_dataclass(kw_only=True)
class FourierTransform:
    f: float | jnp.ndarray
    x0: float = 0.0
    y0: float = 0.0

    def __call__(self, ray: Ray) -> Ray:
        family = getattr(ray, "ray_family", "ray")

        if family == "gaussian":
            fs = FreeSpacePropagator()
        else:
            from .propagator import FreeSpaceParaxial

            fs = FreeSpaceParaxial()

        ray = fs(ray, self.f)
        lens = Lens(z=ray.z, focal_length=self.f, x0=self.x0, y0=self.y0)
        ray = lens(ray)
        ray = fs(ray, self.f)
        return ray


__all__ = [
    "Component",
    "DescanError",
    "Plane",
    "Lens",
    "KrivanekLens",
    "AberratedLensKrivanek",
    "SeidelLens",
    "DistortedLens",
    "ScanGrid",
    "Scanner",
    "Descanner",
    "Detector",
    "ThickLens",
    "Deflector",
    "Rotator",
    "DeflectionBiprism",
    "PhaseBiprism",
    "Biprism",
    "RotatingLens",
    "ElectromagneticLens",
    "ABCDTransfer",
    "SigmoidAperture",
    "ConstantPhaseShift",
    "LinearPhaseShift",
    "QuadraticPhaseShift",
    "ConstantAmplitudeShift",
    "LinearAmplitudeShift",
    "QuadraticAmplitudeShift",
    "MagneticPhaseSample",
    "RandomPhaseSample",
    "InterpolatedSample2D",
    "InterpolatedFields3D",
    "AtomicPotential",
    "FourierTransform",
]
