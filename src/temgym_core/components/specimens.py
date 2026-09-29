"""Analytical specimen profiles and periodic (Bragg) specimens."""


from __future__ import annotations


import numpy as np
from typing import Any

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from ..gaussian import (
    GaussianBeam,
)



from ._base import Component, GaussianActionComponent


# Analytical specimen profiles

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

    def _local_coords(self, xy):
        x = xy[..., 0] - self.x0
        y = xy[..., 1] - self.y0
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

    def _local_coords(self, xy):
        x = xy[..., 0] - self.x0
        y = xy[..., 1] - self.y0
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


# Specimens

@jdc.pytree_dataclass(kw_only=True)
class BraggSample(Component):
    """Thin periodic transmission represented by explicit Fourier orders.

    The field multiplier is sum(c_m exp(i q_m . (xy - origin_xy))). Each
    incident Gaussian produces one child per retained order, with amplitude
    c_m and slope increment q_m/k; the phase at its centre is preserved.
    Coordinates, z and periods use the beam's length unit; q_xy uses radians
    per that unit. Coefficients describe transmission at the chosen energy,
    not a potential that is automatically rescaled with beam voltage.

    Use run_to_end on a scalar beam or a directly batched bundle. This sample
    changes the beam count: N inputs and M orders produce N*M outputs. A
    separate vmap over inputs would retain an extra parent-beam axis.
    This is an infinite periodic projected specimen, not dynamical diffraction.

    angular_response optionally supplies a complex field multiplier as a
    function of grazing angle and voltage. surface_normal_xyz is the outward
    surface normal in the incoming beam frame (unit length is recommended).
    Ordinary __call__ still uses forward/folded-frame thin-grating geometry;
    it does not itself reflect a 3D surface. A reflection adapter can instead
    use coefficients_at with its physical incident angles. Angular response
    is constant within each Gaussian packet: use a narrow angular basis when
    the response varies across the illumination.
    """

    q_xy: jnp.ndarray
    coefficients: jnp.ndarray
    z: float = 0.0
    origin_xy: tuple = (0.0, 0.0)
    angular_response: Any = None
    surface_normal_xyz: tuple = (0.0, 0.0, -1.0)

    def coefficients_at(self, grazing_angle_rad, voltage):
        """Order field coefficients with an optional complex angular response.

        Returns shape (..., M), without renormalizing retained orders. Surface
        reflection adapters should call this with physical incidence angles.
        The ordinary __call__ retains the thin forward/folded-frame geometry;
        an angular response alone does not reverse the propagation direction.
        """
        angles = jnp.asarray(grazing_angle_rad)
        c = jnp.broadcast_to(jnp.asarray(self.coefficients), angles.shape + (len(self.coefficients),))
        if self.angular_response is not None:
            c = c*self.angular_response.amplitude(angles, voltage)[...,None]
        return c

    def order_action(self, grazing_angle_rad, voltage, k):
        """Complex order action: exp(i*k*action) equals coefficients_at.

        An exactly zero order has infinite attenuation; direct coefficient
        multiplication in propagation preserves exact zeros without log(0).
        """
        return -1j*jnp.log(self.coefficients_at(grazing_angle_rad, voltage).astype(complex))/jnp.asarray(k)[...,None]

    @classmethod
    def from_unit_cell(
        cls, transmission_yx, *, periods_xy, orders_hk, z=0.0,
        origin_xy=(0.0, 0.0),
    ):
        """Fourier-expand a rectangular unit cell of complex field transmission.

        T[j,i] is sampled at (i*period_x/nx, j*period_y/ny) relative to
        origin_xy, without duplicate endpoints. Supply physical transmission,
        e.g. exp(1j*phase_radians), in image (y,x) order. orders_hk is a fixed
        (M,2) list of integer reciprocal indices, including (0,0) if desired.

        Construction runs on the host; subsequent propagation supports JAX.
        Refine the unit-cell grid and retained orders independently. Truncated
        coefficients are never renormalized: omitted orders remain omitted.
        A grid's Nyquist limit alone does not establish convergence.
        """
        transmission = np.asarray(transmission_yx, dtype=np.complex128)
        periods = np.asarray(periods_xy, dtype=float)
        orders = np.asarray(orders_hk)
        origin = np.asarray(origin_xy, dtype=float)
        if transmission.ndim != 2 or min(transmission.shape) < 2:
            raise ValueError("Supply a 2D unit-cell transmission with at least two samples per axis.")
        if not np.isfinite(transmission).all():
            raise ValueError("Unit-cell transmission must be finite.")
        if periods.shape != (2,) or not np.isfinite(periods).all() or np.any(periods <= 0):
            raise ValueError("periods_xy must contain two positive finite periods.")
        if origin.shape != (2,) or not np.isfinite(origin).all() or not np.isfinite(z):
            raise ValueError("Supply a finite origin_xy pair and z position.")
        if orders.ndim != 2 or orders.shape[1] != 2 or len(orders) == 0:
            raise ValueError("orders_hk must be a nonempty (M,2) list of integer orders.")
        if not np.isfinite(orders).all() or not np.equal(orders, np.round(orders)).all():
            raise ValueError("orders_hk must contain integers.")
        ny, nx = transmission.shape
        if np.any(2 * np.abs(orders) >= np.array([nx, ny])):
            raise ValueError("Retained orders must lie strictly inside the unit-cell Nyquist limits.")
        orders = orders.astype(int)
        if len(np.unique(orders, axis=0)) != len(orders):
            raise ValueError("orders_hk must contain unique orders.")
        spectrum = np.fft.fft2(transmission) / transmission.size
        coefficients = spectrum[orders[:, 1] % ny, orders[:, 0] % nx]
        return cls(
            q_xy=jnp.asarray(2 * np.pi * orders / periods),
            coefficients=jnp.asarray(coefficients), z=z,
            origin_xy=jnp.asarray(origin),
        )

    def evaluate_complex(self, xy, *, grazing_angle_rad=None, voltage=None):
        """Evaluate the retained field multiplier at specimen coordinates.

        Angle and voltage are required when angular_response is enabled.
        """
        coefficients = jnp.asarray(self.coefficients)
        if self.angular_response is not None:
            if grazing_angle_rad is None or voltage is None:
                raise ValueError("Supply grazing_angle_rad and voltage for an angular response.")
            coefficients = self.coefficients_at(grazing_angle_rad, voltage)
        phase = (jnp.asarray(xy) - jnp.asarray(self.origin_xy)) @ jnp.asarray(self.q_xy).T
        return jnp.sum(coefficients * jnp.exp(1j * phase), axis=-1)

    def _call_gaussian(self, ray: GaussianBeam) -> GaussianBeam:
        ray = ray.to_vector()
        q = jnp.asarray(self.q_xy)
        coefficients = jnp.asarray(self.coefficients)
        if q.ndim != 2 or q.shape[1] != 2 or q.shape[0] == 0:
            raise ValueError("q_xy must have shape (M,2) with at least one order.")
        if coefficients.shape != (q.shape[0],):
            raise ValueError("Supply one complex coefficient per q_xy order.")
        phase = (ray.r_xy - jnp.asarray(self.origin_xy)) @ q.T
        if self.angular_response is not None:
            direction = jnp.column_stack((ray.dx, ray.dy, jnp.ones_like(ray.dx)))
            direction /= jnp.linalg.norm(direction, axis=-1, keepdims=True)
            normal = jnp.asarray(self.surface_normal_xyz)
            normal = normal/jnp.linalg.norm(normal)
            grazing = jnp.arcsin(jnp.clip(-direction @ normal, -1., 1.))
            coefficients = self.coefficients_at(grazing, ray.voltage)
        amplitude = ray.amplitude[:, None] * coefficients * jnp.exp(1j * phase)
        slopes = ray.d_xy[:, None, :] + q[None, :, :] / ray.k[:, None, None]
        # to_vector broadcasts shared scalar fields before repeating parents.
        children = jax.tree.map(lambda value: jnp.repeat(value, q.shape[0], axis=0), ray)
        return children.derive(
            dx=slopes[..., 0].ravel(), dy=slopes[..., 1].ravel(),
            amplitude=amplitude.ravel(),
        )
