"""Test-only checks of the local quadratic Gaussian approximation.

Sample eight points around each beam's elliptical support. These residuals
help expose unresolved structure; they are not a bound on the full image error.
Image convergence must still be checked by refining the Gaussian decomposition.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class SamplingError(NamedTuple):
    phase_max_abs: jax.Array
    phase_rms: jax.Array
    amplitude_max_abs: jax.Array
    amplitude_rms: jax.Array


def _support_offsets(beam, support_radius=1.0):
    precision = beam.k * jnp.imag(beam.Q_inv)
    eigenvalues, axes = jnp.linalg.eigh(0.5 * (precision + precision.T))
    radii = support_radius * jnp.sqrt(2.0 / eigenvalues)
    directions = jnp.array([
        [1., 0.], [-1., 0.], [0., 1.], [0., -1.],
        [1., 1.], [1., -1.], [-1., 1.], [-1., -1.],
    ])
    directions /= jnp.linalg.norm(directions, axis=1, keepdims=True)
    return (directions * radii) @ axes.T


def sampling_error(beam, *, pathlength=None, log_amplitude=None, support_radius=1.0):
    """Return per-beam maximum/RMS errors in phase radians and log amplitude.

    Supply functions ``(xy, single_beam) -> scalar``; pathlength uses metres.
    Explicit arguments keep tests clear for lenses needing voltage or slope.
    An omitted function contributes zero error. Both scalar and batched beams
    use the same calculation through ``vmap``. No beam is modified or corrected.
    """
    def single(single_beam):
        centre = single_beam.r_xy
        offsets = _support_offsets(single_beam, support_radius)

        def residual(function, *, scale=1.0, floor_amplitude=False):
            if function is None:
                return jnp.array(0.), jnp.array(0.)

            def field(xy):
                value = function(xy, single_beam)
                # Match GaussianActionComponent's log-transmission floor.
                if floor_amplitude:
                    value = jnp.logaddexp(value, -50.0)
                return jnp.real(value)

            value = field(centre)
            gradient = jax.grad(field)(centre)
            hessian = jax.hessian(field)(centre)
            quadratic = value + offsets @ gradient + 0.5 * jnp.einsum(
                "ni,ij,nj->n", offsets, hessian, offsets
            )
            error = scale * (jax.vmap(field)(centre + offsets) - quadratic)
            return jnp.max(jnp.abs(error)), jnp.sqrt(jnp.mean(error**2))

        phase = residual(pathlength, scale=single_beam.k)
        amplitude = residual(log_amplitude, floor_amplitude=True)
        return SamplingError(*phase, *amplitude)

    return jax.vmap(single)(beam.to_vector())
