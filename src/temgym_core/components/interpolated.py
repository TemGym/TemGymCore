"""Specimens and fields defined by interpolated arrays."""


from __future__ import annotations


import numpy as np
from scipy.interpolate import CubicSpline
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from interpax import Interpolator2D, Interpolator3D

from ..gaussian import (
    LENGTH,
    GaussianBeam,
)
from ..constants import (
    energy2wavelength,
)
from ..ray import Ray



from ._base import (
    Component,
    GaussianActionComponent,
    _apply_gaussian_action,
    _value_grad_hessian,
)


class ScalarActionJet2D(NamedTuple):
    value: jnp.ndarray
    gradient_xy: jnp.ndarray
    hessian_xy: jnp.ndarray


class SampleActionJet2D(NamedTuple):
    pathlength: ScalarActionJet2D
    log_amplitude: ScalarActionJet2D


@jdc.pytree_dataclass(kw_only=True)
class InterpolatedSample2D(GaussianActionComponent):
    """A thin specimen described by interpolated amplitude and phase.

    Prefer ``from_action_maps`` for physical optical path and log field
    amplitude maps in image (y, x) order. It uses the C2 cubic interpolant
    and explicit spatial derivatives needed by Gaussian propagation.
    ``from_array`` retains the original complex-transmission API (x, y order).
    """

    interpolator: Interpolator2D
    pathlength_per_radian: float = 1.0
    method: jdc.Static[str] = "catmull-rom"
    z: float = 0.0
    action_maps: jdc.Static[bool] = False
    centre_xy_m: tuple = (0.0, 0.0)
    rotation_rad: float = 0.0
    tilt_xy_rad: tuple = (0.0, 0.0)
    domain_bounds_xyxy_m: jdc.Static[tuple | None] = None
    boundary: jdc.Static[str] = "vacuum"
    log_floor: jdc.Static[float] = -50.0
    periodic_origin_xy_m: jdc.Static[tuple] = (0.0, 0.0)

    @classmethod
    def from_action_maps(
        cls,
        *,
        pathlength_yx_m,
        log_amplitude_yx,
        x_coords_m,
        y_coords_m,
        z=0.0,
        centre_xy_m=(0.0, 0.0),
        rotation_rad=0.0,
        tilt_xy_rad=(0.0, 0.0),
        boundary="vacuum",
        log_floor=-50.0,
    ):
        """Build a fixed specimen from two real maps and pixel centres in metres.

        Both maps have shape (len(y_coords_m), len(x_coords_m)). Coordinates
        must be uniformly spaced with at least four knots per axis. Keep the
        illuminated field inside the map, or choose a suitable boundary mode.
        Construction runs on the host; evaluation and propagation support JAX.
        """
        path = np.asarray(pathlength_yx_m, dtype=np.float64)
        log_amplitude = np.asarray(log_amplitude_yx, dtype=np.float64)
        x_coords = np.asarray(x_coords_m, dtype=np.float64)
        y_coords = np.asarray(y_coords_m, dtype=np.float64)
        if path.ndim != 2 or path.shape != log_amplitude.shape:
            raise ValueError("Supply equal-shape 2D pathlength and log-amplitude maps.")
        if path.shape != (y_coords.size, x_coords.size):
            raise ValueError("Maps must use (y, x) image order.")
        for coords in (x_coords, y_coords):
            if coords.ndim != 1 or coords.size < 4:
                raise ValueError("C2 cubic interpolation needs four knots per axis.")
            spacing = np.diff(coords)
            if not np.all(spacing > 0) or not np.allclose(
                spacing, spacing[0], rtol=1e-9, atol=0
            ):
                raise ValueError("Coordinates must be increasing and uniformly spaced.")
        if not all(
            np.isfinite(a).all() for a in (path, log_amplitude, x_coords, y_coords)
        ):
            raise ValueError("Maps and coordinates must be finite.")

        def finite_pair(name, value):
            pair = np.asarray(value, dtype=np.float64)
            if pair.shape != (2,) or not np.isfinite(pair).all():
                raise ValueError(f"{name} must contain two finite values.")
            return pair

        if boundary == "clamp":
            boundary = "constant"
        if boundary not in ("vacuum", "constant", "periodic", "nan"):
            raise ValueError(
                "boundary must be 'vacuum', 'constant', 'periodic', or 'nan'."
            )
        centre = np.asarray(finite_pair("centre_xy_m", centre_xy_m), dtype=np.float64)
        if not np.isfinite(rotation_rad) or not np.isfinite(z):
            raise ValueError("rotation_rad and z must be finite.")
        tilt = np.asarray(finite_pair("tilt_xy_rad", tilt_xy_rad), dtype=np.float64)
        if np.any(np.abs(tilt) >= 0.5 * np.pi - 1e-6):
            raise ValueError("tilt_xy_rad must keep the projected sample plane nonsingular.")
        if not np.isfinite(log_floor):
            raise ValueError("log_floor must be finite.")
        # Interpax's first two data axes are x then y.  Public storage remains
        # conventional yx and this is the only conversion point.
        action_xyc = np.stack(
            (path.T, log_amplitude.T), axis=-1
        )
        extrap = np.zeros((2,), dtype=np.float64) if boundary == "vacuum" else False
        domain_bounds = (
            float(x_coords[0]),
            float(x_coords[-1]),
            float(y_coords[0]),
            float(y_coords[-1]),
        )
        if boundary == "constant":
            # Queries are clamped before evaluation; True prevents a spurious
            # NaN at floating-point roundoff just outside the final knot.
            extrap = True
        periodic_origin = (0.0, 0.0)
        derivative_kwargs: dict[str, np.ndarray] = {}
        if boundary == "periodic":
            # Append the wrapped seam knot and solve periodic cubic splines for
            # the cached first/mixed derivatives.  Interpax's default cubic2
            # derivative solve is not periodic; padding it merely makes the
            # seam error smaller, not C2.  These tensor-product derivatives
            # make the two limiting bicubic patches agree through order two.
            dy, dx = y_coords[1] - y_coords[0], x_coords[1] - x_coords[0]
            period_x = x_coords.size * dx
            period_y = y_coords.size * dy
            x_coords = np.concatenate(
                (x_coords, [x_coords[0] + period_x])
            )
            y_coords = np.concatenate(
                (y_coords, [y_coords[0] + period_y])
            )
            action_xyc = np.concatenate((action_xyc, action_xyc[:1]), axis=0)
            action_xyc = np.concatenate((action_xyc, action_xyc[:, :1]), axis=1)
            fx = CubicSpline(
                x_coords, action_xyc, axis=0, bc_type="periodic"
            )(x_coords, 1)
            fy = CubicSpline(
                y_coords, action_xyc, axis=1, bc_type="periodic"
            )(y_coords, 1)
            fxy = CubicSpline(
                y_coords, fx, axis=1, bc_type="periodic"
            )(y_coords, 1)
            derivative_kwargs = {"fx": fx, "fy": fy, "fxy": fxy}
            # Public wrapping uses the conventional pixel-edge extent.  That
            # interval has the same period as the spline and places its seam
            # halfway between the final and first stored pixel centres.
            domain_bounds = (
                float(x_coords[0] - dx / 2), float(x_coords[-2] + dx / 2),
                float(y_coords[0] - dy / 2), float(y_coords[-2] + dy / 2),
            )
            periodic_origin = (float(x_coords[0]), float(y_coords[0]))
            extrap = True
        interpolator = Interpolator2D(
            x=x_coords,
            y=y_coords,
            f=action_xyc,
            method="cubic2",
            extrap=extrap,
            **derivative_kwargs,
        )
        return cls(
            interpolator=interpolator,
            centre_xy_m=jnp.asarray(centre),
            rotation_rad=jnp.asarray(rotation_rad, dtype=jnp.float64),
            tilt_xy_rad=jnp.asarray(tilt, dtype=jnp.float64),
            domain_bounds_xyxy_m=domain_bounds,
            z=float(z),
            boundary=boundary,
            action_maps=True,
            method="cubic2",
            log_floor=float(log_floor),
            periodic_origin_xy_m=periodic_origin,
        )

    @property
    def rotation_world_from_local(self) -> jnp.ndarray:
        c = jnp.cos(self.rotation_rad)
        s = jnp.sin(self.rotation_rad)
        return jnp.asarray(((c, -s), (s, c)), dtype=jnp.float64)

    @property
    def projection_world_from_local(self) -> jnp.ndarray:
        """Orthographic xy projection of the tilted physical sample plane."""

        tx, ty = self.tilt_xy_rad
        cx, sx = jnp.cos(tx), jnp.sin(tx)
        cy, sy = jnp.cos(ty), jnp.sin(ty)
        zero = jnp.asarray(0.0, dtype=jnp.float64)
        one = jnp.asarray(1.0, dtype=jnp.float64)
        rotation_x = jnp.asarray(
            ((one, zero, zero), (zero, cx, -sx), (zero, sx, cx))
        )
        rotation_y = jnp.asarray(
            ((cy, zero, sy), (zero, one, zero), (-sy, zero, cy))
        )
        c = jnp.cos(self.rotation_rad)
        s = jnp.sin(self.rotation_rad)
        rotation_z = jnp.asarray(
            ((c, -s, zero), (s, c, zero), (zero, zero, one))
        )
        return (rotation_z @ rotation_y @ rotation_x)[:2, :2]

    @property
    def projected_path_scale(self) -> jnp.ndarray:
        """Normal-thickness multiplier for the tilted projected action."""

        tx, ty = self.tilt_xy_rad
        return 1.0 / jnp.abs(jnp.cos(tx) * jnp.cos(ty))

    def world_to_local(self, xy_world: jnp.ndarray) -> jnp.ndarray:
        xy_world = jnp.asarray(xy_world, dtype=jnp.float64)
        if xy_world.shape[-1] != 2:
            raise ValueError("xy_world must have final dimension two.")
        local_from_world = jnp.linalg.inv(self.projection_world_from_local)
        return jnp.einsum(
            "ij,...j->...i", local_from_world, xy_world - self.centre_xy_m
        )

    def _local_action_jet(self, xy_local: jnp.ndarray):
        x = xy_local[..., 0]
        y = xy_local[..., 1]
        query_shape = x.shape
        x_min, x_max, y_min, y_max = self.domain_bounds_xyxy_m

        if self.boundary == "constant":
            inside_x = (x >= x_min) & (x <= x_max)
            inside_y = (y >= y_min) & (y <= y_max)
            xq = jnp.clip(x, x_min, x_max)
            yq = jnp.clip(y, y_min, y_max)
        elif self.boundary == "periodic":
            inside_x = jnp.ones_like(x, dtype=bool)
            inside_y = jnp.ones_like(y, dtype=bool)
            origin_x, origin_y = self.periodic_origin_xy_m
            xq = origin_x + jnp.mod(x - origin_x, x_max - x_min)
            yq = origin_y + jnp.mod(y - origin_y, y_max - y_min)
        else:
            inside_x = jnp.ones_like(x, dtype=bool)
            inside_y = jnp.ones_like(y, dtype=bool)
            xq, yq = x, y

        # Interpax 0.3.x documents arbitrary query shapes, but its bicubic
        # channel broadcasting is reliable for a one-dimensional query axis.
        # Flatten and restore here so image-shaped and batched calls have the
        # same public contract.
        xq_flat = jnp.ravel(xq)
        yq_flat = jnp.ravel(yq)

        def evaluate(dx: int, dy: int):
            values = self.interpolator(xq_flat, yq_flat, dx=dx, dy=dy)
            return values.reshape(query_shape + (2,))

        value = evaluate(0, 0)
        fx = evaluate(1, 0)
        fy = evaluate(0, 1)
        fxx = evaluate(2, 0)
        fxy = evaluate(1, 1)
        fyy = evaluate(0, 2)

        gradient = jnp.stack((fx, fy), axis=-1)
        hessian = jnp.stack(
            (
                jnp.stack((fxx, fxy), axis=-1),
                jnp.stack((fxy, fyy), axis=-1),
            ),
            axis=-2,
        )
        if self.boundary == "constant":
            derivative_mask = jnp.stack((inside_x, inside_y), axis=-1)
            gradient = gradient * derivative_mask[..., None, :]
            hessian = hessian * (
                derivative_mask[..., :, None] * derivative_mask[..., None, :]
            )[..., None, :, :]
        return value, gradient, hessian

    def action_jet(self, xy_world: jnp.ndarray) -> SampleActionJet2D:
        """Evaluate both actions and their explicit world-coordinate jets."""

        if not self.action_maps:
            def jet(point):
                return SampleActionJet2D(
                    ScalarActionJet2D(*_value_grad_hessian(self.phase_shift, point)),
                    ScalarActionJet2D(*_value_grad_hessian(self.log_transmission, point)),
                )
            xy = jnp.asarray(xy_world)
            if xy.ndim == 1:
                return jet(xy)
            flat = jax.vmap(jet)(xy.reshape(-1, 2))
            return jax.tree.map(lambda a: a.reshape(xy.shape[:-1] + a.shape[1:]), flat)
        local = self.world_to_local(xy_world)
        value, local_gradient, local_hessian = self._local_action_jet(local)
        local_from_world = jnp.linalg.inv(self.projection_world_from_local)
        action_scale = self.projected_path_scale
        value = value * action_scale
        gradient = action_scale * jnp.einsum(
            "ji,...cj->...ci", local_from_world, local_gradient
        )
        hessian = jnp.einsum(
            "ji,...cjk,kl->...cil",
            local_from_world,
            local_hessian,
            local_from_world,
        ) * action_scale
        return SampleActionJet2D(
            pathlength=ScalarActionJet2D(
                value[..., 0], gradient[..., 0, :], hessian[..., 0, :, :]
            ),
            log_amplitude=ScalarActionJet2D(
                value[..., 1], gradient[..., 1, :], hessian[..., 1, :, :]
            ),
        )

    def evaluate_actions(self, xy_world: jnp.ndarray) -> jnp.ndarray:
        """Return ``[..., (pathlength_m, log_field_amplitude)]``."""

        jet = self.action_jet(xy_world)
        return jnp.stack((jet.pathlength.value, jet.log_amplitude.value), axis=-1)

    def phase_shift(self, xy):
        if self.action_maps:
            return self.action_jet(xy).pathlength.value
        return jnp.angle(self.evaluate_complex(xy)) * self.pathlength_per_radian

    def log_transmission(self, xy):
        if self.action_maps:
            return self.action_jet(xy).log_amplitude.value
        value = self.evaluate_complex(xy)
        return 0.5 * jnp.log(jnp.real(value)**2 + jnp.imag(value)**2 + 1e-30)

    def _call_ray(self, ray: Ray):
        return ray

    @staticmethod
    def _floored_log_amplitude_jet(jet: ScalarActionJet2D, floor: float):
        # logaddexp(L, floor) and its analytic first/second derivatives.
        weight = jax.nn.sigmoid(jet.value - floor)
        value = jnp.logaddexp(jet.value, floor)
        gradient = weight[..., None] * jet.gradient_xy
        hessian = (
            weight[..., None, None] * jet.hessian_xy
            + (weight * (1.0 - weight))[..., None, None]
            * jnp.einsum("...i,...j->...ij", jet.gradient_xy, jet.gradient_xy)
        )
        return ScalarActionJet2D(value, gradient, hessian)

    def _call_gaussian_single(self, ray: GaussianBeam) -> GaussianBeam:
        if not self.action_maps:
            return super()._call_gaussian_single(ray)
        jet = self.action_jet(ray.r_xy)
        log_jet = self._floored_log_amplitude_jet(jet.log_amplitude, self.log_floor)
        return _apply_gaussian_action(ray, jet.pathlength, log_jet)

    @classmethod
    def from_array(
        cls,
        sample,
        x_coords,
        y_coords,
        extrap=1.0,
        z=0.0,
        method="cubic",
        *,
        voltage: float | None = None,
        wavelength: float | None = None,
        wavelength_unit: str = "m",
    ):
        """Create a sample from a complex amplitude transmission.

        Provide ``voltage`` (electron optics) or ``wavelength`` when the
        complex argument is a physical phase in radians. With neither, the
        argument is interpreted as a pathlength for compatibility with older
        TemGymCore notebooks. An explicit ``wavelength`` is expressed in
        ``wavelength_unit``.
        """
        if voltage is not None and wavelength is not None:
            raise ValueError("Provide at most one of `voltage` or `wavelength`.")
        if wavelength_unit not in LENGTH:
            raise ValueError(
                f"Unknown wavelength unit {wavelength_unit!r}; "
                f"expected one of {tuple(LENGTH)}."
            )
        if voltage is not None:
            wavelength = energy2wavelength(voltage) / LENGTH[wavelength_unit]
        pathlength_per_radian = (
            1.0 if wavelength is None else wavelength / (2.0 * jnp.pi)
        )
        interpolator = Interpolator2D(
            x=x_coords,
            y=y_coords,
            f=sample,
            method=method,
            extrap=extrap,
        )
        return cls(
            z=z,
            interpolator=interpolator,
            method=method,
            pathlength_per_radian=pathlength_per_radian,
        )

    def evaluate_complex(self, xy, *, wavelength_m=None):
        """Field transmission; action maps require the electron wavelength."""
        if not self.action_maps:
            return self.interpolator(xy[..., 0], xy[..., 1])
        if wavelength_m is None:
            raise ValueError("Supply wavelength_m to convert optical path to phase.")
        actions = self.evaluate_actions(xy)
        return jnp.exp(actions[..., 1] + 2j * jnp.pi * actions[..., 0] / wavelength_m)


def sample_interpolant(
    sample,
    x_coords,
    y_coords,
    *,
    z: float = 0.0,
    method: str = "cubic",
    extrap: float = 1.0,
    voltage: float | None = None,
    wavelength: float | None = None,
    wavelength_unit: str = "m",
) -> InterpolatedSample2D:
    """Create an interpolated sample component from a complex sample array.

    This is a convenience wrapper around ``InterpolatedSample2D.from_array``
    for notebook and pipeline code that expects a ``sample_interpolant`` API.
    """
    return InterpolatedSample2D.from_array(
        sample=sample,
        x_coords=x_coords,
        y_coords=y_coords,
        extrap=extrap,
        z=z,
        method=method,
        voltage=voltage,
        wavelength=wavelength,
        wavelength_unit=wavelength_unit,
    )


@jdc.pytree_dataclass(kw_only=True)
class InterpolatedFields3D(Component):
    """Interpolated field container; evaluate(xyz) returns the stored fields.

    Beam interaction is not implemented: calling this component currently
    returns the beam unchanged.
    """

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
