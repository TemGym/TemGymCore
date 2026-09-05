from functools import lru_cache
import math

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

try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None

try:
    from numba import cuda
except ModuleNotFoundError:
    cuda = None

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

    All per-beam quantities are packed into a single (N, 12) array on the host
    with k folded into every phase coefficient, so the inner kernel loop
    contains no per-beam k multiplications.
    """
    N, P = r_centre.shape[0], r2.shape[0]
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
    beams = jnp.column_stack([
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

            # Phase — k already folded into all coefficients on host
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


def evaluate_gaussians_gpu_kernel(
    r_centre: jnp.ndarray,
    dr: jnp.ndarray,
    amplitude: jnp.ndarray,
    pathlength: jnp.ndarray,
    Q_inv: jnp.ndarray,
    k: jnp.ndarray,
    r2: jnp.ndarray,
    *,
    tile_pixels: int = 16,
    tile_beams: int = 16,
):
    """Run the low-level Pallas kernel using its original public API."""
    return _evaluate_gaussians_gpu_kernel_impl(
        r_centre,
        dr,
        amplitude,
        pathlength,
        Q_inv,
        k,
        r2,
        tile_pixels=tile_pixels,
        tile_beams=tile_beams,
    )


# Block dims are compile-time constants so cuda.shared.array can use them
_OPT_TPB_BEAM = 32
_OPT_TPB_PIX  = 128


def _beam_field_cuda_opt(x_det, y_det, rays, out_r, out_i):
    pix_idx, beam_block_idx = cuda.grid(2)
    # loc_beam_block = cuda.threadIdx.y  -> always zero
    # loc_pix  = cuda.threadIdx.x  # fast idx

    # for accuracy, keep these f64
    acc_r = np.float64(0.0)
    acc_i = np.float64(0.0)

    for loc_beam in range(_OPT_TPB_BEAM):
        beam_idx  = beam_block_idx * _OPT_TPB_BEAM + loc_beam
        if beam_idx < rays['r'].shape[0] and pix_idx < x_det.size:
            ray = rays[beam_idx]
            # x_det, y_det -> load coalescing on fast index
            dx = x_det[pix_idx] - ray['r'][0]
            dy = y_det[pix_idx] - ray['r'][1]

            linear = dx * ray['dr'][0] + dy * ray['dr'][1]

            quad_r = (
                ray['Qi_r'][0] * dx * dx
                + ray['Qi_r'][1] * dx * dy
                + ray['Qi_r'][2] * dy * dy
            )
            quad_i = (
                ray['Qi_i'][0] * dx * dx
                + ray['Qi_i'][1] * dx * dy
                + ray['Qi_i'][2] * dy * dy
            )

            s_real = ray['pl'] + linear + 0.5 * quad_r
            s_imag = 0.5 * quad_i

            atten = math.exp(-ray['k'] * s_imag)
            phase = ray['k'] * s_real
            c = math.cos(phase)
            s = math.sin(phase)

            ar = ray['amp'].real
            ai = ray['amp'].imag

            acc_r += atten * (ar * c - ai * s)
            acc_i += atten * (ar * s + ai * c)

    cuda.syncthreads()

    # One thread per pixel reduces the beam dimension → one global atomic per block per pixel
    if pix_idx < x_det.size:
        cuda.atomic.add(out_r, pix_idx, acc_r)
        cuda.atomic.add(out_i, pix_idx, acc_i)


if cuda is not None:
    _beam_field_cuda_opt = cuda.jit(cache=True, fastmath=True, lineinfo=True)(
        _beam_field_cuda_opt
    )


rays_type = np.dtype([
    ('Qi_r', np.float64, (3,)),  # packed: (0,0), (0,1)+(1,0), (1,1)
    ('Qi_i', np.float64, (3,)),
    ('amp', np.complex128),
    ('pl', np.float64),
    ('r', np.float64, (2,)),
    ('dr', np.float64, (2,)),
    ('k', np.float64),
    ('PADDING', np.float64),
])

def evaluate_gaussians_cuda_gpu_kernel_wrapper_opt(gaussian_ray, grid):
    if cp is None or cuda is None:
        raise ImportError(
            "evaluate_gaussians_cuda_gpu_kernel_wrapper_opt requires cupy and numba."
        )

    r, dr, amp, pl, Q_inv, k = _prepare_gaussian_params(gaussian_ray)

    x_det = np.ascontiguousarray(grid.coords[:, 0], dtype=np.float64)
    y_det = np.ascontiguousarray(grid.coords[:, 1], dtype=np.float64)

    n_beams = r.shape[0]
    n_pix   = x_det.size

    Qi = np.asarray(Q_inv, dtype=np.complex128)

    d_x    = cuda.to_device(x_det)
    d_y    = cuda.to_device(y_det)

    rays = np.zeros(dtype=rays_type, shape=(n_beams,))
    rays['Qi_r'] = np.array([Qi.real[:, 0, 0], Qi.real[:, 0, 1] + Qi.real[:, 1, 0], Qi.real[:, 1, 1]]).T
    rays['Qi_i'] = np.array([Qi.imag[:, 0, 0], Qi.imag[:, 0, 1] + Qi.imag[:, 1, 0], Qi.imag[:, 1, 1]]).T
    rays['r'] = r
    rays['dr'] = dr
    rays['amp'] = amp
    rays['pl'] = pl
    rays['k'] = k

    d_rays = cuda.to_device(rays)

    d_out_r = cp.zeros(n_pix, dtype=np.float64)
    d_out_i = cp.zeros(n_pix, dtype=np.float64)

    blocks = (math.ceil(n_pix / _OPT_TPB_PIX), math.ceil(n_beams / _OPT_TPB_BEAM),)
    _beam_field_cuda_opt[blocks, (_OPT_TPB_PIX, 1)](
        d_x, d_y, d_rays, d_out_r, d_out_i
    )
    res = (d_out_r.get() + 1j * d_out_i.get()).reshape(grid.shape)

    return res




def ensure_batch(x, sample_shape=(), dtype=None):
    """
    Ensure x has shape (B, *sample_shape), collapsing any existing leading dims into
    a batch dimension. If sample_shape == (), treat x as per-beam scalars and ensure
    shape (B,).

    Args:
        x: array-like
        sample_shape: tuple, the desired trailing shape
        dtype: optional dtype conversion

    Returns:
        JAX array with leading batch axis.
    """
    x = jnp.asarray(x, dtype=dtype)
    sample_shape = tuple(sample_shape)

    # Scalar case → produce (B,)
    if len(sample_shape) == 0:
        if x.ndim == 0:
            return x[None]           # scalar → (1,)
        if x.ndim == 1:
            return x                 # already (B,)
        return x.reshape((-1,))      # collapse all dims → (B,)

    # Non-scalar case
    s = len(sample_shape)

    # If trailing dims already match sample_shape → collapse leading dims to batch
    if x.ndim >= s and tuple(x.shape[-s:]) == sample_shape:
        return x.reshape((-1,) + sample_shape)

    # Exact match to sample_shape → add leading batch axis
    if tuple(x.shape) == sample_shape:
        return x[None, ...]

    # Fallback: add leading axis
    return x[None, ...]


def _prepare_gaussian_params(gaussian_ray):
    gaussian_ray = gaussian_ray.to_vector()
    r = ensure_batch(gaussian_ray.r_xy, (2,), jnp.float64)
    dr = ensure_batch(gaussian_ray.d_xy, (2,), jnp.float64)
    amplitude = ensure_batch(gaussian_ray.amplitude, (), jnp.complex128)
    pathlength = ensure_batch(gaussian_ray.pathlength, (), jnp.float64)
    Q_inv = ensure_batch(gaussian_ray.Q_inv,  (2, 2), jnp.complex128)
    k = ensure_batch(gaussian_ray.k, (), jnp.float64)

    return r, dr, amplitude, pathlength, Q_inv, k


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


def evaluate_gaussians(
    gaussian_ray,
    grid: Grid,
    *,
    method: str = "auto",
    batch_size: int | None = 128,
    tile_pixels: int = 16,
    tile_beams: int = 16,
):
    """Evaluate a Gaussian bundle with a stable public backend dispatcher.

    ``auto`` selects the Pallas implementation on GPU and the portable JAX
    scan elsewhere. ``gpu_kernel`` retains the historical CPU fallback.
    ``gpu_kernel_required`` is fail-closed: it raises unless the probed Pallas
    kernel is available on a real GPU. ``loop`` is primarily useful as a
    reference evaluator.
    """
    if method == "auto":
        method = "gpu_kernel" if gpu_kernel_available() else "scan"
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


__all__ = [
    "evaluate_gaussians",
    "evaluate_gaussians_for",
    "evaluate_gaussians_gpu",
    "evaluate_gaussians_gpu_kernel",
    "evaluate_gaussians_gpu_kernel_wrapper",
    "evaluate_gaussians_cuda_gpu_kernel_wrapper_opt",
    "evaluate_gaussians_jax_scan",
    "gpu_kernel_available",
    "map_reduce",
]
