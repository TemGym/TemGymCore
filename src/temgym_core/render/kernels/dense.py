"""Dense Pallas (Triton, float64) kernels: every pixel against a run of beams.

``forward_run`` / ``tangent_run``
    Pixel-major: a program owns ``tiles.pixels`` pixels and loops over the beams of one
    contiguous *run* of the beam array (``start``, ``count`` are runtime values).  The run is
    the whole bundle for ``method="dense"`` and for irregular grids, and the dense radius class
    of the sparse plan otherwise.  ``nt`` tangent fields share each ``g``.
``adjoint``
    Beam-major: a program owns ``tiles.beams`` beams and loops over all pixel tiles, so each
    beam writes its own six complex moments; no atomics, bitwise deterministic.

All three apply the shared pair predicate when ``spec.limit`` is set.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

from ..packing import AX, AY, MX, MY, PXX, PXY, PYY, QXX, QXY, QYY
from .common import (F64, accumulate_moments, accumulate_pixel_major, cdiv, chunked, launch,
                     pad_rows, pixel_xy)


def _det(spec, coords, multiple):
    """Padded coordinate operand (a dummy row when positions come from the raster index)."""
    if spec.source == "index":
        return jnp.zeros((1, 2), F64)
    return pad_rows(jnp.asarray(coords, F64), multiple)


def pixel_major_run(packed, coords, w, run, *, spec, tiles, interpret, name):
    """Sum over the run ``run = (start, count)`` of ``packed`` for every pixel.

    ``w`` is ``None`` for the field (result ``(1, P)``) or the ``(nt, N, 12)`` tangent
    coefficients (result ``(nt, P)``).
    """
    n_pix, n_beams = spec.n_pixels, packed.shape[0]
    nt = 0 if w is None else w.shape[0]
    n_out = max(nt, 1)
    t_p, t_b = tiles.pixels, tiles.beams
    limit = spec.limit
    det = _det(spec, coords, t_p)
    n_pad = cdiv(n_pix, t_p) * t_p
    inputs = [det, jnp.asarray(packed, F64)]
    if nt:
        inputs.append(jnp.moveaxis(jnp.asarray(w, F64), 0, 1).reshape(n_beams, nt * 12))
    inputs.append(jnp.asarray(run, jnp.int32))

    def kernel(*refs):
        det_ref, beams_ref, *rest = refs
        w_ref = rest[0] if nt else None
        run_ref, out_ref = rest[-2], rest[-1]
        p = pl.program_id(0) * t_p + jnp.arange(t_p, dtype=jnp.int32)
        x, y = pixel_xy(spec, det_ref, p)
        x, y = x[:, None], y[:, None]
        start, count = run_ref[0], run_ref[1]

        def body(i, acc):
            ids = start + i * t_b + jnp.arange(t_b, dtype=jnp.int32)
            return accumulate_pixel_major(beams_ref, w_ref, x, y, ids, ids < start + count, acc,
                                          nt=nt, limit=limit)

        zero = jnp.zeros((t_p, t_b), F64)
        acc = lax.fori_loop(jnp.int32(0), cdiv(count, t_b), body, (zero,) * (2 * n_out))
        for t in range(2 * n_out):
            out_ref[t, p] = jnp.sum(acc[t], axis=1)

    out = launch(
        kernel, out_shape=jax.ShapeDtypeStruct((2 * n_out, n_pad), F64), grid=(n_pad // t_p,),
        warps=tiles.warps, stages=tiles.stages, interpret=interpret, name=name,
    )(*inputs)
    return (out[0::2] + 1j * out[1::2])[:, :n_pix]


def whole_bundle(n_beams):
    """The run covering every beam."""
    return jnp.array([0, n_beams], jnp.int32)


def forward(packed, coords, *, spec, interpret=False):
    """Dense coherent field ``(P,)``."""
    if spec.n_pixels == 0 or packed.shape[0] == 0:
        return jnp.zeros((spec.n_pixels,), jnp.complex128)
    return pixel_major_run(packed, coords, None, whole_bundle(packed.shape[0]), spec=spec,
                           tiles=spec.render.forward, interpret=interpret,
                           name="gaussian_forward_dense")[0]


def tangent_run(packed, coords, w, run, *, spec, interpret=False):
    """``K`` over one run: ``(nt, N, 12)`` coefficients to ``(nt, P)`` fields."""
    def one(wc):
        return pixel_major_run(packed, coords, wc, run, spec=spec, tiles=spec.render.tangent,
                               interpret=interpret, name=f"gaussian_tangent_dense_nt{wc.shape[0]}")

    return chunked(one, jnp.asarray(w, F64), spec.render.nt_chunk)


def tangent(packed, coords, w, *, spec, interpret=False):
    """Dense ``K``: ``(nt, N, 12)`` coefficients to ``(nt, P)`` complex fields."""
    if spec.n_pixels == 0 or packed.shape[0] == 0 or w.shape[0] == 0:
        return jnp.zeros((w.shape[0], spec.n_pixels), jnp.complex128)
    return tangent_run(packed, coords, w, whole_bundle(packed.shape[0]), spec=spec,
                       interpret=interpret)


def adjoint(packed, coords, ct, *, spec, interpret=False):
    """Dense ``K^T``: ``(nt, P)`` complex cotangents to ``(nt, N, 12)`` moments."""
    n_beams, nt = packed.shape[0], ct.shape[0]
    if n_beams == 0 or spec.n_pixels == 0 or nt == 0:
        return jnp.zeros((nt, n_beams, 12), F64)
    tiles = spec.render.adjoint
    t_b, t_p = tiles.beams, tiles.pixels
    limit = spec.limit
    beams = pad_rows(jnp.asarray(packed, F64), t_b)
    det = _det(spec, coords, t_p)
    n_pad_beams = beams.shape[0]
    n_pad_pix = cdiv(spec.n_pixels, t_p) * t_p
    ct = jnp.pad(jnp.asarray(ct, jnp.complex128), ((0, 0), (0, n_pad_pix - ct.shape[1])))
    n_ptiles = n_pad_pix // t_p

    def kernel(det_ref, beams_ref, ctr_ref, cti_ref, out_ref):
        b = pl.program_id(0) * t_b + jnp.arange(t_b, dtype=jnp.int32)
        t = pl.program_id(1)

        def col(c):
            return beams_ref[b, c][:, None]

        beam = tuple(col(c) for c in (MX, MY, AX, AY, PXX, PXY, PYY, QXX, QXY, QYY))

        def body(i, acc):
            p = i * t_p + jnp.arange(t_p, dtype=jnp.int32)
            x, y = pixel_xy(spec, det_ref, p)
            return accumulate_moments(acc, x[None, :], y[None, :], beam, ctr_ref[t, p][None, :],
                                      cti_ref[t, p][None, :], limit=limit)

        zeros = tuple(jnp.zeros((t_b, t_p), F64) for _ in range(12))
        acc = lax.fori_loop(0, n_ptiles, body, zeros)
        rows = t * n_pad_beams + b
        for j in range(6):
            out_ref[rows, 2 * j] = jnp.sum(acc[2 * j], axis=1)
            out_ref[rows, 2 * j + 1] = -jnp.sum(acc[2 * j + 1], axis=1)

    def one(ctc):
        n = ctc.shape[0]
        out = launch(
            kernel, out_shape=jax.ShapeDtypeStruct((n * n_pad_beams, 12), F64),
            grid=(n_pad_beams // t_b, n), warps=tiles.warps, stages=tiles.stages,
            interpret=interpret, name="gaussian_adjoint_dense",
        )(det, beams, jnp.real(ctc), jnp.imag(ctc))
        return out.reshape(n, n_pad_beams, 12)

    return chunked(one, ct, spec.render.nt_chunk)[:, :n_beams]
