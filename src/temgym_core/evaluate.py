"""Coherent Gaussian fields and detector images.

Use ``evaluate_gaussians`` for complex fields and ``gaussian_image`` for pixel
intensities. ``scan`` supports JAX derivatives; ``gpu_kernel`` evaluates the
full sum on GPU; ``spatial`` sums neighbouring beams with a bounded tail cutoff.
The dense GPU kernel also handles broad beams for the spatial renderer.
"""

from functools import lru_cache, partial
from dataclasses import replace
import math
import operator
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl

try:
    from jax.experimental.pallas import triton as plgpu
except ImportError:
    plgpu = None

from .grid import Grid


def evaluate_gaussians(
    gaussian_ray,
    grid: Grid,
    *,
    method: str = "auto",
    batch_size: int | None = 128,
    tile_pixels: int = 16,
    tile_beams: int = 16,
    spatial_cutoff: float = 1e-12,
    spatial_bins: int = 64,
):
    """Evaluate a Gaussian bundle with a stable public backend dispatcher.

    ``auto`` selects the Pallas implementation on GPU and the portable JAX
    scan elsewhere. ``gpu_kernel`` retains the historical CPU fallback.
    ``gpu_kernel_required`` is fail-closed: it raises unless the probed Pallas
    kernel is available on a real GPU. ``loop`` is primarily useful as a
    reference evaluator. ``spatial`` rebuilds Gaussian support bins on GPU for
    each forward call, using an envelope cutoff and an exact dense fallback for
    broad beams. It is forward-only; use ``scan`` for automatic differentiation.
    """
    if method == "auto":
        method = "gpu_kernel" if gpu_kernel_available() else "scan"
    if method == "spatial":
        return _evaluate_gaussians_spatial(
            gaussian_ray, grid, cutoff=spatial_cutoff,
            bins_per_axis=spatial_bins, tile_beams=tile_beams,
            batch_size=batch_size,
        )
    if method == "scan":
        return evaluate_gaussians_jax_scan(
            gaussian_ray,
            grid,
            batch_size=batch_size,
        )
    if method == "gpu_kernel":
        return evaluate_gaussians_gpu(
            gaussian_ray,
            grid,
            tile_pixels=tile_pixels,
            tile_beams=tile_beams,
            fallback_batch_size=batch_size,
        )
    if method == "gpu_kernel_required":
        if not gpu_kernel_available():
            raise RuntimeError(
                "Gaussian evaluation method 'gpu_kernel_required' requires "
                "a working JAX Pallas GPU backend; CPU fallback is disabled."
            )
        return evaluate_gaussians_gpu_kernel_wrapper(
            gaussian_ray,
            grid,
            tile_pixels=tile_pixels,
            tile_beams=tile_beams,
        )
    if method == "loop":
        return evaluate_gaussians_for(gaussian_ray, grid)
    raise ValueError(f"Unknown Gaussian evaluation method {method!r}.")


def gaussian_image(
    beam, detector, *, samples_per_pixel=2, blur_sigma_m=None,
    padding_pixels=4, method="scan", tile_beams=32,
):
    """Integrate coherent Gaussian intensity over a native Detector's pixels.

    The beam must already be at the detector plane. Fields interfere before
    taking their squared magnitude. Optional isotropic Gaussian blur represents
    source angular spread and/or detector PSF; its pixel integral is analytic.
    Refine quadrature and padding separately to check convergence. The result
    retains absolute flux, including losses outside the detector.
    """
    from jax.scipy.special import ndtr

    if not isinstance(samples_per_pixel, int) or samples_per_pixel < 1:
        raise ValueError("samples_per_pixel must be a positive integer.")
    if not isinstance(padding_pixels, int) or padding_pixels < 0:
        raise ValueError("padding_pixels must be a nonnegative integer.")
    if len(detector.shape) != 2 or any(int(n) != n or n <= 0 for n in detector.shape):
        raise ValueError("Detector shape must contain two positive integers.")
    n = samples_per_pixel
    ny, nx = detector.shape
    dy, dx = detector.pixel_size
    pad = padding_pixels * n if blur_sigma_m is not None else 0
    grid = replace(
        detector,
        shape=(ny * n + 2 * pad, nx * n + 2 * pad),
        pixel_size=(dy / n, dx / n),
    )
    field = evaluate_gaussians(beam, grid, method=method, tile_beams=tile_beams)
    intensity = jnp.abs(field) ** 2
    if blur_sigma_m is None:
        return intensity.reshape(ny, n, nx, n).mean(axis=(1, 3)) * dy * dx

    # Work along the detector axes, so rectangular pixels, pose and flips use
    # the same Grid transform as ordinary TemGym field evaluation.
    sigma = jnp.maximum(blur_sigma_m, 1e-30)

    def response(count, pitch):
        positions = (
            jnp.arange(count * n + 2 * pad) + 0.5 - (count * n + 2 * pad) / 2
        ) * pitch / n
        edges = (jnp.arange(count + 1) - count / 2) * pitch
        values = ndtr((edges[:, None] - positions[None, :]) / sigma)
        return values[1:] - values[:-1]

    return response(ny, dy) @ intensity @ response(nx, dx).T * (dy * dx / n**2)


def gaussian_intensity_image(
    centres,
    variance,
    weights,
    dose=1.0,
    background=0.0,
    *,
    shape_yx,
    psf_sigma_pixels=0.0,
):
    """Fast intensity approximation: integrate independent round Gaussian spots.

    Centres, variance and PSF use pixel units. Weights are integrated spot
    intensities. Off-detector flux is lost; no field interference is included.
    """
    ny, nx = shape_yx
    xe, ye = jnp.arange(nx + 1) - nx / 2, jnp.arange(ny + 1) - ny / 2
    width = jnp.sqrt(variance + psf_sigma_pixels**2)
    px = normal_interval_probability(
        (xe[:-1, None] - centres[:, 0]) / width,
        (xe[1:, None] - centres[:, 0]) / width,
    )
    py = normal_interval_probability(
        (ye[:-1, None] - centres[:, 1]) / width,
        (ye[1:, None] - centres[:, 1]) / width,
    )
    return dose * ((py * weights) @ px.T) + background


def normal_interval_probability(lower, upper):
    """Stable standard-normal probability on an ordered interval."""
    from jax.scipy.special import ndtr

    # Positive tails use symmetry to avoid subtracting two numbers near one.
    return jnp.where(lower > 0, ndtr(-lower) - ndtr(-upper), ndtr(upper) - ndtr(lower))


def zero_phase(u, idx_x, idx_y):
    """Remove the uniform phase at array index (idx_x, idx_y)."""
    u_centre = u[idx_x, idx_y]
    phase_difference = 0 - jnp.angle(u_centre)
    u_new = u * jnp.exp(1j * phase_difference)
    return u_new


def _pack_gaussian_params(r_centre, dr, amplitude, pathlength, Q_inv, k):
    """Pack phase coefficients identically for dense and spatial GPU kernels."""
    f = jnp.float64

    # Symmetrize Q_inv: after this Qxy == Qyx
    Q = jnp.asarray(Q_inv, jnp.complex128)
    Q = 0.5 * (Q + jnp.swapaxes(Q, -1, -2))

    kf = jnp.asarray(k, f)
    k_half = 0.5 * kf

    # Fold pathlength into amplitude: amp * exp(i*k*path)
    amp = jnp.asarray(amplitude, jnp.complex128) * jnp.exp(
        1j * kf * jnp.asarray(pathlength, f)
    )

    # Pack all per-beam scalars into (N, 12), pre-multiplying k into every
    # phase coefficient. After symmetrization the cross-term is 2*Qxy;
    # combined with the 0.5 prefactor: 0.5*k*2*Qxy = k*Qxy.
    rc = jnp.asarray(r_centre, f)
    drf = jnp.asarray(dr, f)
    return jnp.column_stack([
        rc[:, 0],                           # 0  mx
        rc[:, 1],                           # 1  my
        kf * drf[:, 0],                     # 2  k·drx
        kf * drf[:, 1],                     # 3  k·dry
        k_half * jnp.real(Q[:, 0, 0]),      # 4  ½k·Re(Qxx)
        k_half * jnp.imag(Q[:, 0, 0]),      # 5  ½k·Im(Qxx)
        kf * jnp.real(Q[:, 0, 1]),          # 6  k·Re(Qxy)  (= ½k·2Re(Qxy))
        kf * jnp.imag(Q[:, 0, 1]),          # 7  k·Im(Qxy)
        k_half * jnp.real(Q[:, 1, 1]),      # 8  ½k·Re(Qyy)
        k_half * jnp.imag(Q[:, 1, 1]),      # 9  ½k·Im(Qyy)
        jnp.real(amp),                      # 10 Re(amp)
        jnp.imag(amp),                      # 11 Im(amp)
    ])


def _evaluate_gaussians_gpu_kernel_impl(
    r_centre: jnp.ndarray,      # (N,2) float64
    dr: jnp.ndarray,            # (N,2) float64
    amplitude: jnp.ndarray,     # (N,) complex128
    pathlength: jnp.ndarray,    # (N,) float64
    Q_inv: jnp.ndarray,         # (N,2,2) complex128
    k: jnp.ndarray,             # (N,) float64
    r2: jnp.ndarray,            # (P,2) float64
    *,
    tile_pixels: int = 16,
    tile_beams: int = 16,
):
    """
    Evaluate sum_b amp_b * exp(i*k_b*(path_b + d·dr_b + 0.5*d^T Q_inv_b d))
    where d = r2 - r_centre_b, using a tiled Pallas/Triton kernel.

    All per-beam quantities are packed into a single (N, 12) array on the device
    with k folded into every phase coefficient, so the inner kernel loop
    contains no per-beam k multiplications.
    """
    N, P = r_centre.shape[0], r2.shape[0]
    f = jnp.float64
    beams = _pack_gaussian_params(r_centre, dr, amplitude, pathlength, Q_inv, k)

    # Column indices
    MX, MY = 0, 1
    KDX, KDY = 2, 3
    QR_XX, QI_XX, QR_XY, QI_XY, QR_YY, QI_YY = 4, 5, 6, 7, 8, 9
    AR, AI = 10, 11

    # Tiling & padding
    T_P, T_B = int(tile_pixels), int(tile_beams)
    grid_n = -(-P // T_P)
    n_btiles = -(-N // T_B)

    def _pad(x, n):
        d = n - x.shape[0]
        return jnp.pad(x, ((0, d),) + ((0, 0),) * (x.ndim - 1)) if d > 0 else x

    det = _pad(jnp.asarray(r2, f), grid_n * T_P)
    beams = _pad(beams, n_btiles * T_B)

    out_shape = (
        jax.ShapeDtypeStruct((grid_n * T_P,), f),
        jax.ShapeDtypeStruct((grid_n * T_P,), f),
    )

    # ---- Pallas kernel ----
    def kernel(det_ref, beams_ref, out_re_ref, out_im_ref):
        pid = pl.program_id(axis=0)
        p = pid * T_P + jnp.arange(T_P, dtype=jnp.int32)
        x, y = det_ref[p, 0], det_ref[p, 1]

        def body(t, acc):
            re, im = acc
            b = t * T_B + jnp.arange(T_B, dtype=jnp.int32)

            dx = x[:, None] - beams_ref[b, MX][None, :]
            dy = y[:, None] - beams_ref[b, MY][None, :]
            dx2, dy2, dxdy = dx * dx, dy * dy, dx * dy

            # Phase — k already folded into all coefficients before the kernel
            phase = (dx   * beams_ref[b, KDX][None, :]
                   + dy   * beams_ref[b, KDY][None, :]
                   + dx2  * beams_ref[b, QR_XX][None, :]
                   + dxdy * beams_ref[b, QR_XY][None, :]
                   + dy2  * beams_ref[b, QR_YY][None, :])

            # Gaussian envelope (imaginary part of quadratic form)
            atten = jnp.exp(-(dx2  * beams_ref[b, QI_XX][None, :]
                            + dxdy * beams_ref[b, QI_XY][None, :]
                            + dy2  * beams_ref[b, QI_YY][None, :]))

            s, c = jnp.sin(phase), jnp.cos(phase)
            ar = beams_ref[b, AR][None, :]
            ai = beams_ref[b, AI][None, :]

            re = re + jnp.sum(atten * (ar * c - ai * s), axis=1)
            im = im + jnp.sum(atten * (ar * s + ai * c), axis=1)
            return re, im

        re, im = lax.fori_loop(
            0, n_btiles, body,
            (jnp.zeros((T_P,), f), jnp.zeros((T_P,), f)),
        )
        out_re_ref[p] = re
        out_im_ref[p] = im

    # ---- Launch ----
    has_gpu = any(d.platform in ("gpu", "cuda", "rocm") for d in jax.devices())
    kw = dict(out_shape=out_shape, grid=(grid_n,))
    if has_gpu:
        if plgpu is not None:
            kw["compiler_params"] = plgpu.CompilerParams()
        else:
            # Compatibility with older Pallas releases.
            kw["backend"] = "triton"
    else:
        kw["interpret"] = True

    out_re, out_im = pl.pallas_call(kernel, **kw)(det, beams)

    return out_re[:P] + 1j * out_im[:P]


def evaluate_gaussians_gpu_kernel_wrapper(
    gaussian_ray,
    grid,
    *,
    tile_pixels: int = 16,
    tile_beams: int = 16,
):
    r, dr, amp, path, Q_inv, k = _prepare_gaussian_params(gaussian_ray)
    r2 = grid.coords

    fld = _evaluate_gaussians_gpu_kernel_impl(
        r, dr, amp, path, Q_inv, k, r2,
        tile_pixels=tile_pixels,
        tile_beams=tile_beams,
    )
    return fld.reshape(grid.shape)


_evaluate_gaussians_gpu_kernel_impl = jax.jit(
    _evaluate_gaussians_gpu_kernel_impl,
    static_argnames=["tile_pixels", "tile_beams"],
)


# Public low-level entry point used for direct array evaluation.
evaluate_gaussians_gpu_kernel = _evaluate_gaussians_gpu_kernel_impl


def _prepare_gaussian_params(gaussian_ray):
    """Use the native beam's batching, then set the evaluator's float64 dtypes."""
    beam = gaussian_ray.to_vector()
    return (
        jnp.asarray(beam.r_xy, jnp.float64),
        jnp.asarray(beam.d_xy, jnp.float64),
        jnp.asarray(beam.amplitude, jnp.complex128),
        jnp.asarray(beam.pathlength, jnp.float64),
        jnp.asarray(beam.Q_inv, jnp.complex128),
        jnp.asarray(beam.k, jnp.float64),
    )


def _beam_field(r, dr, amplitude, pathlength, Q_inv, k, det_xy):
    delta = det_xy - r
    linear = jnp.sum(delta * dr, axis=-1)
    quadratic = jnp.sum((delta @ Q_inv) * delta, axis=-1)
    S_tot = pathlength + linear + 0.5 * quadratic
    return amplitude * jnp.exp(1j * k * S_tot)


def evaluate_gaussians_jax_scan(
    gaussian_ray,
    grid: Grid,
    *,
    batch_size: int | None = 128,
):
    # Extract grid properties before JIT boundary
    coords = grid.coords
    shape = grid.shape
    return _evaluate_gaussians_jax_scan_impl(
        gaussian_ray, coords, shape, batch_size=batch_size
    )


def _evaluate_gaussians_jax_scan_impl(
    gaussian_ray,
    coords: jnp.ndarray,
    shape: tuple,
    *,
    batch_size: int | None = 128,
):
    r, dr, amp, pathlength, Q_inv, k = _prepare_gaussian_params(gaussian_ray)
    r2 = coords
    P = r2.shape[0]
    init = jnp.zeros((P,), dtype=jnp.complex128)

    xs = (r, dr, amp, pathlength, Q_inv, k)

    @jax.jit
    def f_element(x):
        r_c, dr_c, amp_c, pathlength_c, Q_inv_c, k_c = x
        return _beam_field(r_c, dr_c, amp_c, pathlength_c, Q_inv_c, k_c, det_xy=r2)

    out = map_reduce(f_element, jnp.add, init, xs, batch_size=batch_size)
    return out.reshape(shape)


def evaluate_gaussians_for(
    gaussian_ray,
    grid: Grid,
):
    r, dr, amp, pathlength, Q_inv, k = _prepare_gaussian_params(gaussian_ray)
    r2 = grid.coords
    n = r.shape[0]
    total_field = jnp.zeros((r2.shape[0],), dtype=jnp.complex128)
    for i in range(n):
        field = _beam_field(
            r[i],
            dr[i],
            amp[i],
            pathlength[i],
            Q_inv[i],
            k[i],
            r2
        )
        total_field = total_field + field
    return total_field.reshape(grid.shape)


_evaluate_gaussians_jax_scan_impl = jax.jit(
    _evaluate_gaussians_jax_scan_impl,
    static_argnames=("batch_size", "shape")
)


def map_reduce(f, reducer, init, xs, *, batch_size: int | None = None):
    def scan_fn(acc_inner, x):
        return reducer(acc_inner, f(x)), None

    if batch_size is None:
        acc, _ = lax.scan(scan_fn, init, xs)
        return acc

    if batch_size <= 0:
        raise ValueError("batch_size must be positive or None.")

    leaves = jax.tree_util.tree_leaves(xs)
    if not leaves:
        raise ValueError("xs must contain at least one array leaf.")
    n = leaves[0].shape[0]
    full = (n // batch_size) * batch_size
    acc = init

    def reduce_chunk(acc_inner, chunk):
        elements = jax.vmap(f)(chunk)

        def reduce_one(accumulator, value):
            return reducer(accumulator, value), None

        return lax.scan(reduce_one, acc_inner, elements)

    if full:
        chunked = jax.tree_util.tree_map(
            lambda x: x[:full].reshape((-1, batch_size) + x.shape[1:]),
            xs,
        )
        acc, _ = lax.scan(reduce_chunk, acc, chunked)

    if full < n:
        remainder = jax.tree_util.tree_map(lambda x: x[full:], xs)
        acc, _ = lax.scan(scan_fn, acc, remainder)
    return acc


def evaluate_gaussians_gpu(
    gaussian_ray,
    grid: Grid,
    *,
    tile_pixels: int = 16,
    tile_beams: int = 16,
    fallback_batch_size: int | None = 128,
):
    """Evaluate with the Pallas GPU kernel, falling back to JAX scan on CPU."""
    if not gpu_kernel_available():
        return evaluate_gaussians_jax_scan(
            gaussian_ray,
            grid,
            batch_size=fallback_batch_size,
        )
    return evaluate_gaussians_gpu_kernel_wrapper(
        gaussian_ray,
        grid,
        tile_pixels=tile_pixels,
        tile_beams=tile_beams,
    )


class _SpatialBins(NamedTuple):
    order: jax.Array
    starts: jax.Array
    counts: jax.Array
    origin: jax.Array
    cell_size: jax.Array
    use_dense: jax.Array


def _build_spatial_bins(r, q, k, *, cutoff, bins_per_axis):
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
    return _SpatialBins(order, starts, counts, origin, cell_size, use_dense)


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
    bins = _build_spatial_bins(r, q, k, cutoff=cutoff, bins_per_axis=bins_per_axis)

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


def _evaluate_gaussians_spatial(beam, grid, *, cutoff=1e-12, bins_per_axis=64,
                               tile_beams=32, batch_size=128):
    """Rebuild spatial bins on GPU and evaluate a coherent, truncated field.

    The absolute omitted-field bound per pixel is
    ``cutoff * sum(abs(beam.amplitude))``, apart from floating-point roundoff.
    This is not a relative bound at interference zeros. Broad beams fall back
    to the dense GPU kernel; CPU execution uses the exact scan evaluator.
    No index is cached across calls. ``cutoff`` and bin/tile sizes are static
    JIT configuration. This backend is forward-only (no autodiff rule).
    """
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


@lru_cache(maxsize=1)
def gpu_kernel_available() -> bool:
    """Return whether the Pallas Gaussian kernel is usable on a real GPU.

    This is a numerical execution probe rather than a package-presence check.
    It requires JAX x64 on a real GPU, exercises the production 16x16 tiles
    with both beam and pixel padding, and compares the result with an
    independent NumPy evaluation before caching success.  Call
    ``gpu_kernel_available.cache_clear()`` after changing JAX backend
    configuration in a long-lived process.
    """
    try:
        if jax.default_backend() != "gpu" or not bool(jax.config.x64_enabled):
            return False
        gpu_devices = tuple(jax.devices("gpu"))
        if not gpu_devices or any(
            device.platform != "gpu" for device in gpu_devices
        ):
            return False

        n_beams = 17
        n_pixels = 19
        beam_index = np.arange(n_beams, dtype=np.float64)
        pixel_index = np.arange(n_pixels, dtype=np.float64)
        r_centre = np.column_stack(
            (
                0.025 * (beam_index - 8.0),
                0.018 * (np.mod(5.0 * beam_index, 17.0) - 8.0),
            )
        )
        dr = np.column_stack(
            (
                0.015 * np.sin(0.37 * beam_index),
                -0.012 * np.cos(0.29 * beam_index),
            )
        )
        amplitude = (0.4 + 0.03 * beam_index) * np.exp(
            1j * (0.17 * beam_index - 0.4)
        )
        pathlength = 0.007 * np.sin(0.23 * beam_index)
        k = 1.7 + 0.025 * beam_index
        q_inv = np.zeros((n_beams, 2, 2), dtype=np.complex128)
        q_inv[:, 0, 0] = (
            0.08 + 0.01 * np.cos(0.13 * beam_index)
            + 1j * (0.20 + 0.01 * beam_index)
        )
        q_inv[:, 1, 1] = (
            -0.05 + 0.008 * np.sin(0.19 * beam_index)
            + 1j * (0.24 + 0.006 * beam_index)
        )
        q_cross = (
            0.025 * np.sin(0.11 * beam_index)
            + 1j * 0.015 * np.cos(0.07 * beam_index)
        )
        q_inv[:, 0, 1] = q_cross
        q_inv[:, 1, 0] = q_cross
        coords = np.column_stack(
            (
                0.031 * (pixel_index - 9.0),
                0.027 * (np.mod(7.0 * pixel_index, 19.0) - 9.0),
            )
        )

        field = _evaluate_gaussians_gpu_kernel_impl(
            jnp.asarray(r_centre),
            jnp.asarray(dr),
            jnp.asarray(amplitude),
            jnp.asarray(pathlength),
            jnp.asarray(q_inv),
            jnp.asarray(k),
            jnp.asarray(coords),
            tile_pixels=16,
            tile_beams=16,
        )
        actual = np.asarray(jax.block_until_ready(field))

        # Keep the reference independent of the Pallas packing and reduction.
        delta = coords[:, None, :] - r_centre[None, :, :]
        linear = np.einsum("pbi,bi->pb", delta, dr)
        quadratic = np.einsum("pbi,bij,pbj->pb", delta, q_inv, delta)
        expected = np.sum(
            amplitude[None, :]
            * np.exp(
                1j
                * k[None, :]
                * (pathlength[None, :] + linear + 0.5 * quadratic)
            ),
            axis=1,
        )
        if (
            actual.shape != (n_pixels,)
            or actual.dtype != np.dtype(np.complex128)
            or not np.all(np.isfinite(actual))
            or not np.allclose(actual, expected, rtol=2e-11, atol=2e-12)
        ):
            return False
    except Exception:
        return False
    return True


__all__ = [
    "gaussian_image",
    "gaussian_intensity_image",
    "zero_phase",
    "evaluate_gaussians",
    "evaluate_gaussians_for",
    "evaluate_gaussians_gpu",
    "evaluate_gaussians_gpu_kernel",
    "evaluate_gaussians_gpu_kernel_wrapper",
    "evaluate_gaussians_jax_scan",
    "gpu_kernel_available",
    "map_reduce",
]
