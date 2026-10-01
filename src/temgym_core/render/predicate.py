"""The Gaussian pair function and the single truncation predicate.

With offsets ``u = x - m_x`` and ``v = y - m_y`` a beam contributes ``A * g`` to a pixel,

    g     = exp(-delta + i * phi)
    phi   = a_x u + a_y v + p_xx u^2 + p_xy u v + p_yy v^2
    delta = q_xx u^2 + q_xy u v + q_yy v^2,

and the *pair predicate* keeps a (pixel, beam) pair only if ``delta <= -ln(cutoff)``.  Every
kernel (forward, tangent, adjoint, dense or sparse, reference or Pallas) evaluates the pair
through :func:`pair_terms`, so the predicate is one expression and the renderer is the
truncated sum whose derivatives (mask frozen) the tangent and adjoint kernels compute exactly.
Bins, windows and tiles only enumerate supersets of the pairs the predicate keeps.
"""

from __future__ import annotations

import math

import jax.numpy as jnp


def support_limit(cutoff):
    """Threshold ``-ln(cutoff)`` of the envelope exponent; ``None`` for the exact sum.

    Parameters
    ----------
    cutoff : float or None
        Truncation level in ``(0, 1)``, or ``None`` for no truncation.

    Returns
    -------
    float or None
        Pairs with ``delta > limit`` lie outside the support.
    """
    return None if cutoff is None else -math.log(cutoff)


def pair_terms(u, v, ax, ay, pxx, pxy, pyy, qxx, qxy, qyy, limit):
    """Real and imaginary parts of ``g`` for offsets ``(u, v)`` and the squares they need.

    Parameters
    ----------
    u, v : array
        Pixel-minus-centre offsets; any broadcastable shapes.
    ax, ay, pxx, pxy, pyy, qxx, qxy, qyy : array
        Packed phase and envelope columns, broadcastable against the offsets.
    limit : float or None
        Predicate threshold (see :func:`support_limit`); ``None`` keeps every pair.

    Returns
    -------
    tuple of array
        ``(g_re, g_im, uu, uv, vv)``; pairs failing the predicate give ``g = 0`` exactly.
    """
    uu, uv, vv = u * u, u * v, v * v
    phase = u * ax + v * ay + uu * pxx + uv * pxy + vv * pyy
    decay = uu * qxx + uv * qxy + vv * qyy
    att = jnp.exp(-decay)
    if limit is not None:
        att = jnp.where(decay <= limit, att, 0.0)
    return att * jnp.cos(phase), att * jnp.sin(phase), uu, uv, vv
