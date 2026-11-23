"""Spatial plan of the sparse renderer, built on the device from the packed bundle.

The plan is a set of integer (and a few float) arrays with static shapes, computed once per
evaluation and shared, as non-differentiable residuals, by the forward, tangent and adjoint
kernels.  Everything is jittable; nothing synchronises with the host.

Radius classes
    Each beamlet is classified by the half-width of its cutoff footprint (the largest
    half-extent of the ellipse ``delta <= -ln(cutoff)`` along a detector axis, in pixels).
    Class ``j`` holds radii up to the ``j``-th edge; every class gets its own bin grid whose
    cell is the largest radius of its members, so a pixel's candidates are the beams in the
    3 x 3 bins around it whatever the spread of widths in the bundle.  Beamlets wider than the
    last edge, or with a non-positive-definite envelope, form the final *dense* class, which
    every pixel sums over.
Culling
    A beamlet whose footprint box contains no pixel centre of the detector is *inactive*: it is
    sorted behind all classes and never visited, so it costs nothing beyond the sort.
Adjoint work list
    The beam-major adjoint sweeps each active beamlet's window (footprint box clipped to the
    detector) in row chunks.  Windows are cut into work items of about equal work, at most
    ``n_beams + config.items`` in total, so a few very broad beamlets cannot leave the GPU idle;
    the items of a beam are summed in a fixed order, so results are bitwise reproducible.

All pixel-frame quantities of the plan are in the detector's pixel-index frame.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .packing import MX, MY, QXX, QXY, QYY, to_index_space

WINDOW_MARGIN = 1e-9  # relative and absolute (pixels) slack: windows and bins are supersets


class RenderPlan(NamedTuple):
    """Arrays of the sparse plan (all in the *sorted* beam order unless noted).

    Attributes
    ----------
    order : array, shape (N,)
        Sorted position to original beam index: classes ascending, bins row-major inside a
        class, inactive beams last.
    starts, counts : array, shape ((n_classes + 2) * bins ** 2,)
        First sorted position and number of beams of every (class, bin); key
        ``class * bins**2 + by * bins + bx``.  The dense class is bin 0 of class ``n_classes``,
        inactive beams are bin 0 of class ``n_classes + 1``.
    origin : array, shape (n_classes, 2)
        Bin grid origin of every sparse class (pixel-index frame).
    cell : array, shape (n_classes,)
        Bin cell size of every sparse class (pixels).
    class_count : array, shape (n_classes,)
        Beams in every sparse class.
    dense_run : array, shape (2,)
        ``(start, count)`` of the dense class in the sorted order.
    windows : array, shape (N, 4)
        ``(x_lo, x_hi, y_lo, y_hi)`` pixel window of every sorted beam (empty: ``x_lo > x_hi``).
    item_first, item_count : array, shape (N,)
        First work item and number of work items of every sorted beam.
    item_beam, item_lo, item_hi : array, shape (N + config.items,)
        Sorted beam, first step and one-past-last step of every work item; a step is one chunk
        of ``32 * window_warps`` pixels of one window row.
    """

    order: jax.Array
    starts: jax.Array
    counts: jax.Array
    origin: jax.Array
    cell: jax.Array
    class_count: jax.Array
    dense_run: jax.Array
    windows: jax.Array
    item_first: jax.Array
    item_count: jax.Array
    item_beam: jax.Array
    item_lo: jax.Array
    item_hi: jax.Array


def n_classes(spec):
    """Number of sparse radius classes of a spec."""
    return len(spec.render.edges_for(spec.shape))


def footprints(index_packed, limit, shape):
    """Footprint radius, pixel window and validity of every beam in the pixel-index frame.

    Parameters
    ----------
    index_packed : array, shape (N, 12)
        Packed bundle in pixel-index space.
    limit : float
        Pair-predicate threshold ``-ln(cutoff)``.
    shape : tuple of int
        Detector ``(H, W)``.

    Returns
    -------
    tuple
        ``(radius, window, active)``.  ``radius`` is the largest axis half-width of the
        footprint (``inf`` when the envelope is not strictly positive definite), ``window`` the
        ``(N, 4)`` integer box ``(x_lo, x_hi, y_lo, y_hi)`` clipped to the detector (the whole
        detector for non-positive-definite envelopes), ``active`` whether the box is non-empty
        and the row finite.  Boxes carry a small safety margin, so they are supersets of the
        pixels the pair predicate keeps.
    """
    h, w = shape
    qxx, qxy, qyy = index_packed[:, QXX], index_packed[:, QXY], index_packed[:, QYY]
    mx, my = index_packed[:, MX], index_packed[:, MY]
    eps = jnp.finfo(jnp.float64).eps
    # The determinant is rounded down: near-singular envelopes fail the test and go dense.
    det = qxx * qyy - 0.25 * qxy * qxy - 8 * eps * qxx * qyy
    positive = (qxx > 0) & (qyy > 0) & (det > 0)
    safe = jnp.where(positive, det, 1.0)
    hx = jnp.sqrt(limit * jnp.where(positive, qyy, 1.0) / safe) * (1 + WINDOW_MARGIN) + WINDOW_MARGIN
    hy = jnp.sqrt(limit * jnp.where(positive, qxx, 1.0) / safe) * (1 + WINDOW_MARGIN) + WINDOW_MARGIN
    x_lo = jnp.clip(jnp.ceil(mx - hx), 0, w).astype(jnp.int32)
    x_hi = jnp.clip(jnp.floor(mx + hx), -1, w - 1).astype(jnp.int32)
    y_lo = jnp.clip(jnp.ceil(my - hy), 0, h).astype(jnp.int32)
    y_hi = jnp.clip(jnp.floor(my + hy), -1, h - 1).astype(jnp.int32)
    window = jnp.stack([x_lo, x_hi, y_lo, y_hi], axis=1)
    whole = jnp.array([0, w - 1, 0, h - 1], jnp.int32)
    window = jnp.where(positive[:, None], window, whole[None, :])
    empty = (window[:, 0] > window[:, 1]) | (window[:, 2] > window[:, 3])
    active = jnp.all(jnp.isfinite(index_packed), axis=1) & ~empty
    window = jnp.where(active[:, None], window, jnp.array([w, -1, h, -1], jnp.int32)[None, :])
    radius = jnp.where(positive, jnp.maximum(hx, hy), jnp.inf)
    return radius, window, active


def _class_bins(index_packed, radius, cls, m, bins):
    """Per-class bin origin and cell size (dynamic, from the members of each class).

    Masked reductions over a ``(classes, beams)`` broadcast: unlike scatter-min/max they do not
    serialise the many beams that share a class.
    """
    member = cls[None, :] == jnp.arange(m, dtype=cls.dtype)[:, None]

    def reduce(values, fill, op):
        return op(jnp.where(member, values[None, :], fill), axis=1)

    cx, cy = index_packed[:, MX], index_packed[:, MY]
    lo_x, lo_y = reduce(cx, jnp.inf, jnp.min), reduce(cy, jnp.inf, jnp.min)
    hi_x, hi_y = reduce(cx, -jnp.inf, jnp.max), reduce(cy, -jnp.inf, jnp.max)
    reach = reduce(radius, 0.0, jnp.max)
    occupied = jnp.isfinite(lo_x)
    origin = jnp.stack([jnp.where(occupied, lo_x, 0.0), jnp.where(occupied, lo_y, 0.0)], axis=1)
    span = jnp.where(occupied, jnp.maximum(hi_x - lo_x, hi_y - lo_y), 0.0)
    cell = jnp.maximum(reach * (1 + 1e-12), span / (bins - 1))
    return origin, jnp.where(occupied & (cell > 0), cell, 1.0)


def _work_items(windows, lanes, extra, min_steps):
    """Cut every window into work items of about equal work (see the module docstring).

    An item holds at least ``min_steps`` steps (except that a beam with fewer steps is one item)
    and there are at most ``n_beams + extra`` items.
    """
    n = windows.shape[0]
    width = jnp.maximum(windows[:, 1] - windows[:, 0] + 1, 0)
    height = jnp.maximum(windows[:, 3] - windows[:, 2] + 1, 0)
    steps = (height * ((width + lanes - 1) // lanes)).astype(jnp.int64)
    total = jnp.sum(steps)
    target = jnp.maximum((total + extra - 1) // extra, min_steps)
    count = jnp.where(steps > 0, (steps + target - 1) // target, 0)
    ends = jnp.cumsum(count)
    first = ends - count
    slots = jnp.arange(n + extra, dtype=jnp.int64)
    beam = jnp.minimum(jnp.searchsorted(ends, slots, side="right"), n - 1)
    live = slots < ends[-1]
    per_item = jnp.where(count[beam] > 0, (steps[beam] + count[beam] - 1) // count[beam], 1)
    lo = jnp.minimum((slots - first[beam]) * per_item, steps[beam])
    hi = jnp.minimum(lo + per_item, steps[beam])
    lo, hi = jnp.where(live, lo, 0), jnp.where(live, hi, 0)
    i32 = jnp.int32
    return (first.astype(i32), count.astype(i32), beam.astype(i32), lo.astype(i32),
            hi.astype(i32))


def build_plan(packed, affine, spec):
    """Classify, bin and cut into work items the beams of ``packed`` for a raster detector.

    Parameters
    ----------
    packed : array, shape (N, 12)
        Packed bundle in the kernels' frame (pixel-index space for ``spec.source == "index"``,
        metre space otherwise).
    affine : array, shape (6,)
        ``[J00, J01, J10, J11, ox, oy]`` of the pixel-to-metre map; used to bring metre-space
        bundles into the pixel frame for planning only.
    spec : KernelSpec
        Static description of the call (must be sparse).

    Returns
    -------
    RenderPlan
        The plan arrays.
    """
    cfg = spec.render
    n = packed.shape[0]
    bins = cfg.bins_per_axis
    edges = cfg.edges_for(spec.shape)
    m = len(edges)
    packed = lax.stop_gradient(jnp.asarray(packed, jnp.float64))
    if spec.source == "index":
        frame = packed
    else:
        frame = to_index_space(packed, lax.stop_gradient(affine[:4].reshape(2, 2)),
                               lax.stop_gradient(affine[4:]))
    radius, window, active = footprints(frame, cfg.decay_limit, spec.shape)

    if m:
        edge_array = jnp.asarray(edges, jnp.float64)
        cls = jnp.where(radius <= edge_array[-1],
                        jnp.searchsorted(edge_array, radius, side="left"), m)
    else:
        cls = jnp.zeros((n,), jnp.int32)
    cls = jnp.where(active, cls, m + 1).astype(jnp.int32)

    stride = bins * bins
    origin, cell = _class_bins(frame, radius, cls, m, bins)
    key = cls * stride
    if m:
        cls_safe = jnp.minimum(cls, m - 1)
        cell_xy = jnp.clip(jnp.floor((frame[:, [MX, MY]] - origin[cls_safe])
                                     / cell[cls_safe][:, None]), 0, bins - 1).astype(jnp.int32)
        key = jnp.where(cls < m, key + cell_xy[:, 1] * bins + cell_xy[:, 0], key)
    order = jnp.argsort(key, stable=True).astype(jnp.int32)
    counts = jnp.bincount(key, length=(m + 2) * stride).astype(jnp.int32)
    starts = jnp.cumsum(counts, dtype=jnp.int32) - counts
    class_count = counts[:m * stride].reshape(m, stride).sum(axis=1, dtype=jnp.int32)
    dense_run = jnp.stack([starts[m * stride], counts[m * stride]])

    windows = window[order]
    lanes = 32 * cfg.window_warps
    first, count, beam, lo, hi = _work_items(windows, lanes, cfg.items, cfg.item_steps)
    return RenderPlan(order, starts, counts, origin, cell, class_count, dense_run, windows,
                      first, count, beam, lo, hi)
