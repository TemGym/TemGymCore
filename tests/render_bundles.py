"""Synthetic bundles, grids and independent references for the renderer tests and benchmarks."""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.extend import core as jex_core

from temgym_core.render.packing import (AI, AR, AX, AY, MX, MY, PXX, PXY, PYY, QXX, QXY, QYY,
                                        pack_gaussian_params)


def random_packed(rng, n, *, width=0.25, box=1.0, tilt=20.0, curv=30.0, aspect=0.3):
    """Random valid packed ``(N, 12)`` bundle (positive-definite envelope).

    ``width`` is the 1/e amplitude radius, ``tilt``/``curv`` bound the phase gradients and
    curvatures.  Everything is O(1) so derivative maps are well conditioned.
    """
    m = rng.uniform(-box, box, (n, 2))
    a = rng.uniform(-tilt, tilt, (n, 2))
    qxx = (1 + aspect * rng.uniform(-1, 1, n)) / width**2
    qyy = (1 + aspect * rng.uniform(-1, 1, n)) / width**2
    qxy = 0.2 * np.sqrt(qxx * qyy) * rng.uniform(-1, 1, n)
    p = rng.uniform(-curv, curv, (n, 3))
    amp = rng.normal(size=n) + 1j * rng.normal(size=n)
    return np.column_stack([
        m[:, 0], m[:, 1], a[:, 0], a[:, 1], p[:, 0], qxx, p[:, 1], qxy, p[:, 2], qyy,
        amp.real, amp.imag,
    ])


def unit_grid(h, w, box=1.2):
    """Regular ``(h*w, 2)`` array of (x, y) pixel centres on ``[-box, box]^2``."""
    x, y = np.meshgrid(np.linspace(-box, box, w), np.linspace(-box, box, h))
    return np.column_stack([x.ravel(), y.ravel()])


def scattered_coords(rng, n, box=1.2):
    """Irregular pixel coordinates (unsorted, non-grid)."""
    return rng.uniform(-box, box, (n, 2))


def pixel_bundle(rng, n, shape, *, waist_px, tilt=0.6, curv=0.02, amp_scale=1.0,
                 waist_jitter=0.25, margin=0.0):
    """Bundle in pixel units (pitch 1, ``k = 1``) for a raster detector.

    ``waist_px`` is the 1/e amplitude radius in pixels; beam centres are drawn from the image
    extended by ``margin`` pixels on every side.  Returns the packed array, the ``(x, y)``
    pixel-index coordinates and the unpacked ``(r, dr, amplitude, pathlength, Q_inv, k)``
    arrays that ``pack_gaussian_params`` maps to exactly the same packed array.
    """
    h, w = shape
    r = np.column_stack([rng.uniform(-margin, w + margin, n), rng.uniform(-margin, h + margin, n)])
    dr = rng.uniform(-tilt, tilt, (n, 2))
    wx = waist_px * (1 + waist_jitter * rng.uniform(-1, 1, n))
    wy = waist_px * (1 + waist_jitter * rng.uniform(-1, 1, n))
    rot = rng.uniform(0, np.pi, n)
    # 2x2 precision matrix R diag(1/wx^2, 1/wy^2) R^T in units where k = 1: Im Q = 2 * precision
    c, s = np.cos(rot), np.sin(rot)
    ixx, iyy = 1 / wx**2, 1 / wy**2
    prec = np.zeros((n, 2, 2))
    prec[:, 0, 0] = c * c * ixx + s * s * iyy
    prec[:, 1, 1] = s * s * ixx + c * c * iyy
    prec[:, 0, 1] = prec[:, 1, 0] = c * s * (ixx - iyy)
    curvature = rng.uniform(-curv, curv, (n, 2, 2))
    curvature = 0.5 * (curvature + np.swapaxes(curvature, -1, -2))
    q_inv = curvature + 2j * prec
    amplitude = amp_scale * (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2)
    pathlength = rng.uniform(0, 2 * np.pi, n)  # k = 1, so this is a plain phase
    k = np.ones(n)
    x, y = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    coords = np.column_stack([x.ravel(), y.ravel()])
    packed = np.array(pack_gaussian_params(
        jnp.asarray(r), jnp.asarray(dr), jnp.asarray(amplitude), jnp.asarray(pathlength),
        jnp.asarray(q_inv), jnp.asarray(k)))
    return packed, coords, (r, dr, amplitude, pathlength, q_inv, k)


def field_scan(packed, coords, limit=None):
    """Independent forward evaluation: ``lax.scan`` over beams (the ordinary autodiff target).

    Mirrors ``evaluate_gaussians_jax_scan`` on the packed representation and is the golden
    reference for ``jax.jvp`` / ``jax.vjp`` at identical packed inputs.  With ``limit`` it is
    the truncated sum, whose autodiff derivative freezes the (non-differentiable) pair mask,
    i.e. exactly the semantics of the kernels.
    """
    packed = jnp.asarray(packed, jnp.float64)
    coords = jnp.asarray(coords, jnp.float64)
    x, y = coords[:, 0], coords[:, 1]

    def one_beam(acc, row):
        u, v = x - row[MX], y - row[MY]
        phase = (row[AX] * u + row[AY] * v + row[PXX] * u * u
                 + row[PXY] * u * v + row[PYY] * v * v)
        decay = row[QXX] * u * u + row[QXY] * u * v + row[QYY] * v * v
        amp = row[AR] + 1j * row[AI]
        term = amp * jnp.exp(-decay + 1j * phase)
        if limit is not None:
            term = jnp.where(decay <= limit, term, 0.0)
        return acc + term, None

    init = jnp.zeros(coords.shape[0], jnp.complex128)
    return lax.scan(one_beam, init, packed)[0]


def rel(a, b):
    """Largest difference relative to the largest magnitude of ``b``."""
    return float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b)))


def _walk(jaxpr):
    for eqn in jaxpr.eqns:
        yield eqn
        for sub in jex_core.jaxprs_in_params(eqn.params):
            yield from _walk(sub)


def equations(fn, *args, name):
    """All equations (recursively) of ``make_jaxpr(fn)(*args)`` whose primitive is ``name``."""
    closed = jax.make_jaxpr(fn)(*args)
    return [e for e in _walk(closed.jaxpr) if e.primitive.name == name]
