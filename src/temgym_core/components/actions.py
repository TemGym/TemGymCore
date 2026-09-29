"""Apertures and prescribed phase/amplitude elements."""


from __future__ import annotations


from collections.abc import Mapping

import jax.numpy as jnp
import jax.nn as jnn
import jax_dataclasses as jdc
from jax.nn import softplus

from ..aberrations import (
    zernike_noll_cartesian,
)
from ..ray import Ray



from ._base import GaussianActionComponent


# Apertures and prescribed phase/amplitude

@jdc.pytree_dataclass(kw_only=True)
class SigmoidAperture(GaussianActionComponent):
    """Smooth field-amplitude transmission for Gaussians; rays pass through.

    The transmission is a sigmoid of the distance from ``(x0, y0)``. That distance is softened
    over a twentieth of ``edge_width``, so ``log T`` is smooth at the centre: a plain radius has a
    conical tip there, whose Hessian is about ``1 / eps`` and would squeeze a beamlet centred
    exactly on the axis to a fraction of its width. The softening moves the distance by less than
    ``edge_width**2 / (800 * distance)``, so the profile at the edge is unchanged.
    """

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
        # Rays carry no amplitude in this model, so aperture transmission
        # cannot attenuate them; treat as geometric pass-through.
        return ray

    def phase_shift(self, xy):
        return 0.0

    def log_transmission(self, xy):
        x, y = xy[..., 0] - self.x0, xy[..., 1] - self.y0
        w = jnp.maximum(jnp.abs(self.edge_width), self.eps)
        soft = jnp.maximum(0.05 * w, self.eps)
        rho = jnp.sqrt(x * x + y * y + soft * soft)
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

    def _uv(self, xy: jnp.ndarray):
        x, y = xy[..., 0], xy[..., 1]
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

    def phase_shift(self, xy: jnp.ndarray):
        return self.constant_phase_shift


@jdc.pytree_dataclass(kw_only=True)
class LinearPhaseShift(GaussianActionComponent):
    linear_phase_shift: jnp.ndarray
    z: float = 0.0

    def phase_shift(self, xy: jnp.ndarray):
        return jnp.dot(self.linear_phase_shift, xy)


@jdc.pytree_dataclass(kw_only=True)
class QuadraticPhaseShift(GaussianActionComponent):
    quadratic_phase_shift: jnp.ndarray
    z: float = 0.0

    def phase_shift(self, xy: jnp.ndarray):
        return 0.5 * xy @ self.quadratic_phase_shift @ xy


@jdc.pytree_dataclass(kw_only=True)
class ZernikeAberration(GaussianActionComponent):
    """Thin pathlength aberration expressed in unnormalised Noll modes."""

    coefficients: Mapping[int, float]
    radius: float
    x0: float = 0.0
    y0: float = 0.0
    outside: float = 0.0
    z: float = 0.0

    def phase_shift(self, xy: jnp.ndarray):
        x = (xy[..., 0] - self.x0) / self.radius
        y = (xy[..., 1] - self.y0) / self.radius
        out = jnp.zeros_like(x, dtype=jnp.float64)
        for j, coefficient in self.coefficients.items():
            out = out + coefficient * zernike_noll_cartesian(
                int(j),
                x,
                y,
                outside=self.outside,
            )
        return out


@jdc.pytree_dataclass(kw_only=True)
class ConstantAmplitudeShift(GaussianActionComponent):
    amplitude: float
    z: float = 0.0

    def log_transmission(self, xy: jnp.ndarray):
        return jnp.log(self.amplitude)


@jdc.pytree_dataclass(kw_only=True)
class LinearAmplitudeShift(GaussianActionComponent):
    linear_log_amplitude: jnp.ndarray | None = None
    linear_amplitude: jnp.ndarray | None = None
    z: float = 0.0

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

    def log_transmission(self, xy: jnp.ndarray):
        return 0.5 * xy @ self.quadratic_log_amplitude @ xy
