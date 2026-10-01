"""Building blocks shared by the dense and sparse Pallas (Triton, float64) kernels."""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

try:
    from jax.experimental.pallas import triton as plgpu
except ImportError:  # pragma: no cover
    plgpu = None

from ..packing import AI, AR, AX, AY, MX, MY, PXX, PXY, PYY, QXX, QXY, QYY
from ..predicate import pair_terms

F64 = jnp.float64


def cdiv(a, b):
    """Ceiling division for Python or traced integers."""
    return -(-a // b)


def pad_rows(x, multiple):
    """Zero-pad axis 0 of ``x`` to a multiple of ``multiple``."""
    extra = -x.shape[0] % multiple
    return jnp.pad(x, ((0, extra),) + ((0, 0),) * (x.ndim - 1)) if extra else x


def chunked(fn, lin, chunk):
    """Apply ``fn`` to ``lin`` in groups of ``chunk`` along axis 0 (zero padded), concatenated."""
    nt = lin.shape[0]
    if nt <= chunk:
        return fn(lin)
    padded = jnp.pad(lin, ((0, -nt % chunk),) + ((0, 0),) * (lin.ndim - 1))
    out = lax.map(fn, padded.reshape((-1, chunk) + lin.shape[1:]))
    return out.reshape((-1,) + out.shape[2:])[:nt]


def launch(kernel, *, out_shape, grid, warps, stages=None, interpret, name):
    """``pallas_call`` with Triton compiler parameters (or the interpreter on CPU)."""
    kwargs = dict(out_shape=out_shape, grid=grid, name=name)
    if interpret:
        kwargs["interpret"] = True
    else:
        kwargs["compiler_params"] = plgpu.CompilerParams(num_warps=warps, num_stages=stages)
    return pl.pallas_call(kernel, **kwargs)


def pixel_xy(spec, coords_ref, p):
    """Pixel positions ``(x, y)`` of the flat pixel indices ``p`` (vector) in the kernel frame.

    Index-space bundles use the raster index itself; otherwise the positions are read from
    the ``coords`` array (which must be padded so that every ``p`` is a valid row).
    """
    if spec.source == "index":
        iy = p // spec.shape[1]
        return (p - iy * spec.shape[1]).astype(F64), iy.astype(F64)
    return coords_ref[p, 0], coords_ref[p, 1]


def accumulate_pixel_major(beams_ref, w_ref, x, y, ids, mask, acc, *, nt, limit):
    """Add one ``(pixels, beams)`` tile of pair terms to the pixel-major accumulators.

    ``x`` and ``y`` have shape ``(pixels, 1)``; ``ids`` and ``mask`` select ``beams`` rows of
    the (sorted) packed array.  ``nt == 0`` accumulates the field ``A g`` (two accumulators);
    otherwise ``nt`` tangent fields ``g * poly_t`` sharing one ``g`` (two accumulators each),
    with the coefficients of beam ``b`` in ``w_ref[b, t * 12 : (t + 1) * 12]``.  Masked lanes
    load zeros, whose coefficients contribute exactly nothing.
    """
    def col(c):
        return plgpu.load(beams_ref.at[ids, c], mask=mask, other=0.0)[None, :]

    u, v = x - col(MX), y - col(MY)
    gr, gi, _, _, _ = pair_terms(u, v, col(AX), col(AY), col(PXX), col(PXY), col(PYY),
                                 col(QXX), col(QXY), col(QYY), limit)
    if nt == 0:
        ar, ai = col(AR), col(AI)
        return acc[0] + gr * ar - gi * ai, acc[1] + gr * ai + gi * ar
    new = []
    for t in range(nt):
        def c(j, t=t):
            return plgpu.load(w_ref.at[ids, t * 12 + j], mask=mask, other=0.0)[None, :]

        pr = c(0) + u * (c(2) + u * c(6) + v * c(8)) + v * (c(4) + v * c(10))
        pi = c(1) + u * (c(3) + u * c(7) + v * c(9)) + v * (c(5) + v * c(11))
        new.append(acc[2 * t] + gr * pr - gi * pi)
        new.append(acc[2 * t + 1] + gr * pi + gi * pr)
    return tuple(new)


def accumulate_moments(acc, x, y, beam, ct_re, ct_im, *, limit):
    """Add the six complex moments ``ct * g * mono_j`` of one tile to twelve accumulators.

    ``beam`` is the tuple of the ten packed columns ``(mx, my, ax, ay, pxx, pxy, pyy, qxx,
    qxy, qyy)`` broadcastable against the pixel tile ``(x, y)``; ``ct_re``/``ct_im`` are the
    cotangent of the tile's pixels.  Accumulator ``2 j`` and ``2 j + 1`` hold the real and
    imaginary part of moment ``j``.
    """
    mx, my, ax, ay, pxx, pxy, pyy, qxx, qxy, qyy = beam
    u, v = x - mx, y - my
    gr, gi, uu, uv, vv = pair_terms(u, v, ax, ay, pxx, pxy, pyy, qxx, qxy, qyy, limit)
    cr = ct_re * gr - ct_im * gi
    ci = ct_re * gi + ct_im * gr
    new = []
    for j, mono in enumerate((None, u, v, uu, uv, vv)):
        new.append(acc[2 * j] + (cr if mono is None else cr * mono))
        new.append(acc[2 * j + 1] + (ci if mono is None else ci * mono))
    return tuple(new)
