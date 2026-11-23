"""Pure-JAX pair loops: the CPU implementation of the three primitives and the test reference.

Dense over all (pixel, beam) pairs in pixel chunks, with the shared pair predicate applied when
``limit`` is set.  The sparse kernels sum exactly the pairs this predicate keeps, so these
functions are also the sparse renderer's specification.

``field``
    ``E_p = sum_b A_b g_b(p)``.
``polynomial_sum`` (``K``)
    ``(nt, N, 12)`` coefficients -> ``(nt, P)`` complex fields ``sum_b g_b poly_{t,b}``.
``polynomial_moments`` (``K^T``)
    ``(nt, P)`` cotangents -> moments ``M_j = sum_p ct_p g_p mono_j`` in cotangent layout
    ``(nt, N, 12)`` (see :mod:`temgym_core.render.coefficients`).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax

from ..packing import AI, AR, AX, AY, MX, MY, PXX, PXY, PYY, QXX, QXY, QYY
from ..predicate import pair_terms

N_MONOMIALS = 6


def choose_pixel_chunk(n_beams, n_tangents=1, budget_bytes=256 * 2**20):
    """Pixel chunk keeping the largest ``(nt, chunk, N)`` complex temporary within a budget."""
    per_pixel = max(n_beams, 1) * 16 * (6 + 2 * max(n_tangents, 1))
    return int(max(16, min(4096, budget_bytes // per_pixel)))


def _pad_pixels(x, chunk):
    extra = -x.shape[0] % chunk
    if extra:
        x = jnp.pad(x, ((0, extra),) + ((0, 0),) * (x.ndim - 1))
    return x


def _pairs(packed, xy, limit):
    """``g`` (complex), the offsets and the monomials for one pixel chunk: shapes ``(P, N)``."""
    col = [packed[None, :, i] for i in range(packed.shape[1])]
    u = xy[:, 0, None] - col[MX]
    v = xy[:, 1, None] - col[MY]
    gr, gi, uu, uv, vv = pair_terms(u, v, col[AX], col[AY], col[PXX], col[PXY], col[PYY],
                                    col[QXX], col[QXY], col[QYY], limit)
    mono = jnp.stack([jnp.ones_like(u), u, v, uu, uv, vv], axis=-1)
    return gr + 1j * gi, mono


def _as_complex(w):
    """``(nt, N, 12)`` interleaved reals -> ``(nt, N, 6)`` complex."""
    w = w.reshape(w.shape[:-1] + (N_MONOMIALS, 2))
    return w[..., 0] + 1j * w[..., 1]


def field(packed, xy, *, limit=None, chunk=None):
    """Coherent field ``sum_b A_b g_b`` at ``xy``, shape ``(P,)`` complex128."""
    n_pix = xy.shape[0]
    if n_pix == 0 or packed.shape[0] == 0:
        return jnp.zeros((n_pix,), jnp.complex128)
    chunk = chunk or choose_pixel_chunk(packed.shape[0])
    amp = packed[:, AR] + 1j * packed[:, AI]

    def one(c):
        g, _ = _pairs(packed, c, limit)
        return jnp.sum(g * amp[None, :], axis=1)

    chunks = _pad_pixels(xy, chunk).reshape(-1, chunk, 2)
    return lax.map(one, chunks).reshape(-1)[:n_pix]


def polynomial_sum(packed, xy, w, *, limit=None, chunk=None):
    """``K``: ``(nt, N, 12)`` polynomial coefficients to ``(nt, P)`` complex fields."""
    n_pix, nt = xy.shape[0], w.shape[0]
    if n_pix == 0 or packed.shape[0] == 0 or nt == 0:
        return jnp.zeros((nt, n_pix), jnp.complex128)
    chunk = chunk or choose_pixel_chunk(packed.shape[0], nt)
    wc = _as_complex(w)

    def one(c):
        g, mono = _pairs(packed, c, limit)
        poly = jnp.einsum("pnj,tnj->tpn", mono, wc)
        return jnp.sum(g[None] * poly, axis=-1)

    out = lax.map(one, _pad_pixels(xy, chunk).reshape(-1, chunk, 2))
    return jnp.moveaxis(out, 0, 1).reshape(nt, -1)[:, :n_pix]


def polynomial_moments(packed, xy, ct, *, limit=None, chunk=None):
    """``K^T``: ``(nt, P)`` complex cotangents to ``(nt, N, 12)`` moments (cotangent layout)."""
    nt, n_pix = ct.shape
    n_beams = packed.shape[0]
    if n_pix == 0 or n_beams == 0 or nt == 0:
        return jnp.zeros((nt, n_beams, 12), jnp.float64)
    chunk = chunk or choose_pixel_chunk(n_beams, nt)
    ct = jnp.pad(ct, ((0, 0), (0, -n_pix % chunk)))  # zero cotangent: no contribution
    chunks = _pad_pixels(xy, chunk).reshape(-1, chunk, 2)
    ct_chunks = jnp.moveaxis(ct.reshape(nt, -1, chunk), 1, 0)

    def one(acc, xs):
        c, ctc = xs
        g, mono = _pairs(packed, c, limit)
        return acc + jnp.einsum("tp,pn,pnj->tnj", ctc, g, mono), None

    init = jnp.zeros((nt, n_beams, N_MONOMIALS), jnp.complex128)
    moments, _ = lax.scan(one, init, (chunks, ct_chunks))
    return jnp.stack([moments.real, -moments.imag], axis=-1).reshape(nt, n_beams, 12)
