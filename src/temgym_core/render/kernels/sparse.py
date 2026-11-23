"""Sparse (cutoff-truncated) Pallas kernels driven by the plan of :mod:`temgym_core.render.plan`.

``forward`` / ``tangent``
    Pixel-major and tile-major.  One launch for all sparse radius classes: a program owns
    ``rows.pixels`` consecutive pixels of a detector row and, class by class, loops over the beams
    of the three bin rows around them (contiguous runs of the sorted beam array, so coefficient
    loads are shared by all the program's pixels).  The dense class is one run of the dense
    kernel.  The results add in a fixed order.
``adjoint``
    Beam-major over the plan's work items: a program sweeps a range of row chunks of one beam's
    window with one warp and writes six partial complex moments; a second kernel adds the
    partials of each beam in a fixed order (no atomics, bitwise reproducible).

Every kernel keeps a pair exactly when the shared predicate (``spec.limit``) does; bins and
windows only enumerate supersets.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

from ..packing import AX, AY, MX, MY, PXX, PXY, PYY, QXX, QXY, QYY
from ..plan import RenderPlan, n_classes
from . import dense
from .common import (F64, accumulate_moments, accumulate_pixel_major, cdiv, chunked, launch,
                     plgpu)

REDUCE_BEAMS = 128


def _row_padded_coords(coords, shape, width_padded):
    """Coordinates laid out with the padded row stride of the row kernels (array source)."""
    h, w = shape
    xy = jnp.asarray(coords, F64).reshape(h, w, 2)
    return jnp.pad(xy, ((0, 0), (0, width_padded - w), (0, 0))).reshape(h * width_padded, 2)


def rows_call(sorted_packed, w_flat, plan, coords_padded, *, spec, nt, interpret):
    """Sum over all sparse radius classes for every pixel, ``(max(nt, 1), P)`` complex.

    One launch: a program owns ``rows.pixels`` pixels of a row and visits the classes in
    ascending order, skipping empty ones.
    """
    h, w = spec.shape
    cfg = spec.render
    tiles = cfg.rows
    t_p, t_b = tiles.pixels, tiles.beams
    bins = cfg.bins_per_axis
    m = n_classes(spec)
    limit = spec.limit
    n_out = max(nt, 1)
    n_seg = cdiv(w, t_p)
    w_pad = n_seg * t_p
    array_source = spec.source == "array"
    meta = jnp.concatenate([plan.origin, plan.cell[:, None]], axis=1)
    inputs = [sorted_packed] + ([w_flat] if nt else []) + [plan.starts, plan.counts, meta,
                                                            plan.class_count]
    if array_source:
        inputs.append(coords_padded)

    def kernel(*refs):
        beams_ref, *rest = refs
        w_ref = rest.pop(0) if nt else None
        starts_ref, counts_ref, meta_ref, members_ref = rest[:4]
        coords_ref = rest[4] if array_source else None
        out_ref = rest[-1]
        pid = pl.program_id(0)
        y_i = pid // n_seg
        seg = pid % n_seg
        px = seg * t_p + jnp.arange(t_p, dtype=jnp.int32)
        flat = y_i * w_pad + px
        if array_source:
            x = coords_ref[flat, 0][:, None]
            y = coords_ref[flat, 1][:, None]
        else:
            x, y = px.astype(F64)[:, None], y_i.astype(F64)
        first = (seg * t_p).astype(F64)
        last = jnp.minimum(seg * t_p + t_p - 1, w - 1).astype(F64)

        def one_class(cls, acc):
            origin_x, origin_y, size = meta_ref[cls, 0], meta_ref[cls, 1], meta_ref[cls, 2]

            def to_bin(v, o):
                return jnp.clip(jnp.floor((v - o) / size), -2, bins + 1).astype(jnp.int32)

            bx_lo = jnp.clip(to_bin(first, origin_x) - 1, 0, bins - 1)
            bx_hi = jnp.clip(to_bin(last, origin_x) + 1, 0, bins - 1)
            by0 = to_bin(y_i.astype(F64), origin_y)

            def bin_row(i, acc):
                # Empty classes are masked through the count: a loop bound read from memory
                # here is miscompiled by Triton (wrong sums), a constant bound is not.
                by = by0 + i.astype(jnp.int32) - 1
                inside = (by >= 0) & (by < bins)
                base = cls * (bins * bins) + jnp.clip(by, 0, bins - 1) * bins
                start = starts_ref[base + bx_lo]
                count = jnp.where(inside & (members_ref[cls] > 0),
                                  starts_ref[base + bx_hi] + counts_ref[base + bx_hi] - start, 0)

                def tile(t, acc):
                    ids = start + t * t_b + jnp.arange(t_b, dtype=jnp.int32)
                    return accumulate_pixel_major(beams_ref, w_ref, x, y, ids,
                                                  ids < start + count, acc, nt=nt, limit=limit)

                return lax.fori_loop(jnp.int32(0), cdiv(count, t_b), tile, acc)

            return lax.fori_loop(jnp.int32(0), jnp.int32(3), bin_row, acc)

        zero = jnp.zeros((t_p, t_b), F64)
        acc = lax.fori_loop(jnp.int32(0), jnp.int32(m), one_class, (zero,) * (2 * n_out))
        for t in range(2 * n_out):
            out_ref[t, flat] = jnp.sum(acc[t], axis=1)

    out = launch(
        kernel, out_shape=jax.ShapeDtypeStruct((2 * n_out, h * w_pad), F64), grid=(h * n_seg,),
        warps=tiles.warps, stages=tiles.stages, interpret=interpret, name="gaussian_rows",
    )(*inputs)
    field = out[0::2] + 1j * out[1::2]
    return field.reshape(n_out, h, w_pad)[:, :, :w].reshape(n_out, h * w)


def _pixel_major(sorted_packed, w, plan, coords, *, spec, interpret):
    """Field (``w is None``) or tangent fields of all radius classes plus the dense class."""
    nt = 0 if w is None else w.shape[0]
    w_flat = None
    if nt:
        w_flat = jnp.moveaxis(w, 0, 1).reshape(w.shape[1], nt * 12)
    coords_padded = None
    if spec.source == "array":
        tile = spec.render.rows.pixels
        coords_padded = _row_padded_coords(coords, spec.shape, cdiv(spec.shape[1], tile) * tile)
    total = None
    if n_classes(spec):
        total = rows_call(sorted_packed, w_flat, plan, coords_padded, spec=spec, nt=nt,
                          interpret=interpret)
    tiles = spec.render.tangent if nt else spec.render.forward
    part = dense.pixel_major_run(sorted_packed, coords, w, plan.dense_run, spec=spec,
                                 tiles=tiles, interpret=interpret,
                                 name=f"gaussian_dense_class_nt{nt}")
    return part if total is None else total + part


def forward(packed, coords, *plan, spec, interpret=False):
    """Sparse coherent field ``(P,)`` complex128."""
    plan = RenderPlan(*plan)
    return _pixel_major(packed[plan.order], None, plan, coords, spec=spec, interpret=interpret)[0]


def tangent(packed, coords, *rest, spec, interpret=False):
    """Sparse ``K``: ``(nt, N, 12)`` coefficients to ``(nt, P)`` complex fields."""
    *plan, w = rest
    plan = RenderPlan(*plan)
    ps = packed[plan.order]

    def one(wc):
        return _pixel_major(ps, wc[:, plan.order], plan, coords, spec=spec, interpret=interpret)

    return chunked(one, w, spec.render.nt_chunk)


def _items_call(sorted_packed, plan, ct_re, ct_im, coords, *, spec, interpret):
    """Partial moments ``(n_items, 12)`` (cotangent layout) of one cotangent field."""
    h, w = spec.shape
    warps = spec.render.window_warps
    lanes = 32 * warps
    limit = spec.limit
    n_items = plan.item_beam.shape[0]
    array_source = spec.source == "array"
    inputs = [sorted_packed, plan.windows, plan.item_beam, plan.item_lo, plan.item_hi, ct_re,
              ct_im] + ([jnp.asarray(coords, F64)] if array_source else [])

    def kernel(*refs):
        (beams_ref, win_ref, beam_ref, lo_ref, hi_ref, ctr_ref, cti_ref), rest = refs[:7], refs[7:]
        coords_ref = rest[0] if array_source else None
        out_ref = rest[-1]
        item = pl.program_id(0)
        b, lo, hi = beam_ref[item], lo_ref[item], hi_ref[item]
        x_lo, x_hi, y_lo = win_ref[b, 0], win_ref[b, 1], win_ref[b, 2]
        chunks = jnp.maximum((x_hi - x_lo + lanes) // lanes, 1)
        beam = tuple(beams_ref[b, c] for c in (MX, MY, AX, AY, PXX, PXY, PYY, QXX, QXY, QYY))
        lane = jnp.arange(lanes, dtype=jnp.int32)
        row0 = lo // chunks

        def body(_, carry):
            acc, row, chunk = carry
            y = y_lo + row
            ix = x_lo + chunk * lanes + lane
            inside = ix <= x_hi
            idx = y * w + ix
            cr = plgpu.load(ctr_ref.at[idx], mask=inside, other=0.0)[None, :]
            ci = plgpu.load(cti_ref.at[idx], mask=inside, other=0.0)[None, :]
            if array_source:
                x = plgpu.load(coords_ref.at[idx, 0], mask=inside, other=0.0)[None, :]
                yv = plgpu.load(coords_ref.at[idx, 1], mask=inside, other=0.0)[None, :]
            else:
                x, yv = ix.astype(F64)[None, :], y.astype(F64)
            acc = accumulate_moments(acc, x, yv, beam, cr, ci, limit=limit)
            chunk = chunk + 1
            wrap = chunk == chunks
            return acc, jnp.where(wrap, row + 1, row), jnp.where(wrap, 0, chunk)

        zeros = tuple(jnp.zeros((1, lanes), F64) for _ in range(12))
        acc, _, _ = lax.fori_loop(jnp.int32(0), hi - lo, body,
                                  (zeros, row0, lo - row0 * chunks))
        for j in range(6):
            out_ref[item, 2 * j] = jnp.sum(acc[2 * j])
            out_ref[item, 2 * j + 1] = -jnp.sum(acc[2 * j + 1])

    return launch(
        kernel, out_shape=jax.ShapeDtypeStruct((n_items, 12), F64), grid=(n_items,),
        warps=warps, interpret=interpret, name="gaussian_adjoint_items",
    )(*inputs)


def _reduce_call(partial, plan, *, interpret):
    """Sum the partial moments of every beam in item order and undo the beam sorting."""
    n = plan.order.shape[0]
    tb = REDUCE_BEAMS
    n_pad = cdiv(n, tb) * tb
    extra = n_pad - n
    # Padding lanes own no items and write zeros into spare rows behind the last beam.
    first = jnp.pad(plan.item_first, (0, extra))
    count = jnp.pad(plan.item_count, (0, extra))
    dest = jnp.concatenate([plan.order, jnp.arange(n, n_pad, dtype=jnp.int32)])

    def kernel(part_ref, first_ref, count_ref, dest_ref, out_ref):
        b = pl.program_id(0) * tb + jnp.arange(tb, dtype=jnp.int32)
        first, count, dest = first_ref[b], count_ref[b], dest_ref[b]

        def body(s, acc):
            take = s < count
            return tuple(acc[j] + plgpu.load(part_ref.at[first + s, j], mask=take, other=0.0)
                         for j in range(12))

        acc = lax.fori_loop(jnp.int32(0), jnp.max(count), body,
                            tuple(jnp.zeros((tb,), F64) for _ in range(12)))
        for j in range(12):
            out_ref[dest, j] = acc[j]

    out = launch(
        kernel, out_shape=jax.ShapeDtypeStruct((n_pad, 12), F64), grid=(n_pad // tb,), warps=1,
        interpret=interpret, name="gaussian_adjoint_reduce",
    )(partial, first, count, dest)
    return out[:n]


def adjoint(packed, coords, *rest, spec, interpret=False):
    """Sparse ``K^T``: ``(nt, P)`` cotangents to ``(nt, N, 12)`` moments (cotangent layout)."""
    *plan, ct = rest
    plan = RenderPlan(*plan)
    ps = packed[plan.order]

    def one(c):
        partial = _items_call(ps, plan, jnp.real(c), jnp.imag(c), coords, spec=spec,
                              interpret=interpret)
        return _reduce_call(partial, plan, interpret=interpret)

    return lax.map(one, jnp.asarray(ct, jnp.complex128))
