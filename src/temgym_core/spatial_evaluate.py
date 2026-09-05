"""GPU spatial bins for forward coherent Gaussian evaluation.

All geometry, sorting, bin ranges and evaluation remain on the device and are
rebuilt on every call. The nine neighbouring bins conservatively cover each
Gaussian's envelope down to ``cutoff``. No fixed-capacity neighbour list is used.
Broad/invalid envelopes use the original dense kernel instead. This Pallas
backend supports JIT forward evaluation; automatic differentiation is not
implemented. The existing scan backend remains available for derivatives.
"""

from functools import partial
import math
import operator
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu


class SpatialBins(NamedTuple):
    order: jax.Array
    starts: jax.Array
    counts: jax.Array
    origin: jax.Array
    cell_size: jax.Array
    use_dense: jax.Array


def _build_bins(r, q, k, *, cutoff, bins_per_axis):
    # Stable small eigenvalue via determinant / large eigenvalue. A slightly
    # enlarged support compensates for ordinary floating-point boundary error.
    precision = k[:, None, None] * (0.5 * (q + jnp.swapaxes(q, -1, -2))).imag
    a, b, c = precision[:, 0, 0], precision[:, 0, 1], precision[:, 1, 1]
    largest = 0.5 * (a + c + jnp.hypot(a - c, 2 * b))
    smallest = (a * c - b * b) / largest
    # Round down conservatively; near-singular ellipses take the dense path.
    smallest -= 32 * jnp.finfo(jnp.float64).eps * largest
    valid = jnp.all(jnp.isfinite(smallest) & (smallest > 0)) & jnp.all(jnp.isfinite(r))
    radii = jnp.sqrt(-2 * jnp.log(cutoff) / jnp.where(smallest > 0, smallest, 1.))
    origin = jnp.min(r, axis=0)
    span = jnp.max(r, axis=0) - origin
    cell_size = jnp.maximum(jnp.max(radii), jnp.max(span) / (bins_per_axis - 1))
    cell_size *= 1 + 1e-12
    valid &= jnp.isfinite(cell_size) & (cell_size > 0)
    cell_size = jnp.where(valid, cell_size, 1.)
    safe_r = jnp.where(jnp.isfinite(r), r, 0.)
    origin = jnp.where(jnp.isfinite(origin), origin, 0.)
    xy = jnp.clip(jnp.floor((safe_r - origin) / cell_size), 0, bins_per_axis - 1).astype(jnp.int32)
    keys = xy[:, 1] * bins_per_axis + xy[:, 0]
    order = jnp.argsort(keys, stable=False)
    counts = jnp.bincount(keys, length=bins_per_axis**2).astype(jnp.int32)
    starts = jnp.cumsum(counts, dtype=jnp.int32) - counts
    # Conservative performance fallback, also covering one-bin/crossover cases.
    use_dense = ~valid | (9 * jnp.max(counts) >= r.shape[0])
    return SpatialBins(order, starts, counts, origin, cell_size, use_dense)


def _spatial_kernel(packed, coords, bins, *, bins_per_axis, tile_beams):
    n_pixels = coords.shape[0]

    def kernel(beam_ref, xy_ref, starts_ref, counts_ref, origin_ref, size_ref,
               out_re_ref, out_im_ref):
        pixel = pl.program_id(0)
        x, y = xy_ref[pixel, 0], xy_ref[pixel, 1]
        # Clip before integer conversion: far-away coordinates must not overflow.
        bx = jnp.clip(jnp.floor((x - origin_ref[0]) / size_ref[()]),
                      -2, bins_per_axis + 1).astype(jnp.int32)
        by = jnp.clip(jnp.floor((y - origin_ref[1]) / size_ref[()]),
                      -2, bins_per_axis + 1).astype(jnp.int32)

        def neighbour(i, total):
            i = i.astype(jnp.int32)
            ix, iy = bx + i % 3 - 1, by + i // 3 - 1
            valid_bin = (ix >= 0) & (ix < bins_per_axis) & (iy >= 0) & (iy < bins_per_axis)
            key = jnp.clip(iy * bins_per_axis + ix, 0, bins_per_axis**2 - 1)
            start = starts_ref[key]
            count = jnp.where(valid_bin, counts_ref[key], 0)

            def beam_tile(t, acc):
                ids = start + t * tile_beams + jnp.arange(tile_beams, dtype=jnp.int32)
                mask = ids < start + count

                def load(col):
                    return plgpu.load(beam_ref.at[ids, col], mask=mask, other=0.)

                dx, dy = x - load(0), y - load(1)
                dx2, dy2, dxdy = dx * dx, dy * dy, dx * dy
                phase = dx * load(2) + dy * load(3) + dx2 * load(4) + dxdy * load(6) + dy2 * load(8)
                decay = dx2 * load(5) + dxdy * load(7) + dy2 * load(9)
                attenuation = jnp.exp(-decay)
                ar, ai = load(10), load(11)
                sine, cosine = jnp.sin(phase), jnp.cos(phase)
                real = jnp.sum(jnp.where(mask, attenuation * (ar * cosine - ai * sine), 0.))
                imag = jnp.sum(jnp.where(mask, attenuation * (ar * sine + ai * cosine), 0.))
                return acc[0] + real, acc[1] + imag

            return lax.fori_loop(
                jnp.int32(0), (count + tile_beams - 1) // tile_beams, beam_tile, total)

        real, imag = lax.fori_loop(
            jnp.int32(0), jnp.int32(9), neighbour, (jnp.float64(0), jnp.float64(0)))
        out_re_ref[pixel], out_im_ref[pixel] = real, imag

    real, imag = pl.pallas_call(
        kernel,
        out_shape=(jax.ShapeDtypeStruct((n_pixels,), jnp.float64),) * 2,
        grid=(n_pixels,), compiler_params=plgpu.CompilerParams(num_warps=4),
        name="gaussian_spatial_bins",
    )(packed, coords, bins.starts, bins.counts, bins.origin, bins.cell_size)
    return real + 1j * imag


@partial(jax.jit, static_argnames=("cutoff", "bins_per_axis", "tile_beams"))
def _evaluate_spatial(r, dr, amplitude, pathlength, q, k, coords, *,
                      cutoff, bins_per_axis, tile_beams):
    from .evaluate import _evaluate_gaussians_gpu_kernel_impl, _pack_gaussian_params

    bins = _build_bins(r, q, k, cutoff=cutoff, bins_per_axis=bins_per_axis)

    def dense(_):
        return _evaluate_gaussians_gpu_kernel_impl(
            r, dr, amplitude, pathlength, q, k, coords,
            tile_pixels=16, tile_beams=tile_beams,
        )

    def spatial(_):
        packed = _pack_gaussian_params(r, dr, amplitude, pathlength, q, k)
        return _spatial_kernel(packed[bins.order], coords, bins,
                               bins_per_axis=bins_per_axis, tile_beams=tile_beams)

    finite_actions = (jnp.all(jnp.isfinite(dr)) & jnp.all(jnp.isfinite(amplitude))
                      & jnp.all(jnp.isfinite(pathlength)) & jnp.all(jnp.isfinite(q))
                      & jnp.all(jnp.isfinite(k)) & jnp.all(jnp.isfinite(coords)))
    use_dense = bins.use_dense | ~finite_actions
    return lax.cond(use_dense, dense, spatial, operand=None)


def evaluate_gaussians_spatial(beam, grid, *, cutoff=1e-12, bins_per_axis=64,
                               tile_beams=32, batch_size=128):
    """Rebuild spatial bins on GPU and evaluate a coherent, truncated field.

    The absolute omitted-field bound per pixel is
    ``cutoff * sum(abs(beam.amplitude))``, apart from floating-point roundoff.
    This is not a relative bound at interference zeros. Broad beams fall back
    to the dense GPU kernel; CPU execution uses the exact scan evaluator.
    No index is cached across calls. ``cutoff`` and bin/tile sizes are static
    JIT configuration. This backend is forward-only (no autodiff rule).
    """
    from .evaluate import (
        _prepare_gaussian_params, evaluate_gaussians_jax_scan, gpu_kernel_available,
    )

    cutoff = float(cutoff)
    if not math.isfinite(cutoff) or not 0 < cutoff < 1:
        raise ValueError("spatial cutoff must lie strictly between 0 and 1")
    bins_per_axis = operator.index(bins_per_axis)
    tile_beams = operator.index(tile_beams)
    if not 2 <= bins_per_axis <= 1024:
        raise ValueError("spatial bins_per_axis must be between 2 and 1024")
    if tile_beams not in (16, 32, 64):
        raise ValueError("spatial tile_beams must be 16, 32 or 64")
    if not gpu_kernel_available():
        return evaluate_gaussians_jax_scan(beam, grid, batch_size=batch_size)
    params = _prepare_gaussian_params(beam)
    if params[0].shape[0] == 0 or math.prod(grid.shape) == 0:
        return jnp.zeros(grid.shape, dtype=jnp.complex128)
    return _evaluate_spatial(
        *params, grid.coords, cutoff=cutoff, bins_per_axis=bins_per_axis,
        tile_beams=tile_beams,
    ).reshape(grid.shape)
