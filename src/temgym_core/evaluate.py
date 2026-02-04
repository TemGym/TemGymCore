import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax import lax
from temgym_core.grid import Grid
from jax._src.lax.control_flow.loops import _batch_and_remainder


def evaluate_gaussians_gpu_kernel(
    r_centre: jnp.ndarray,      # (N,2) float64
    dr: jnp.ndarray,            # (N,2) float64
    amplitude: jnp.ndarray,     # (N,) complex128
    pathlength: jnp.ndarray,    # (N,) float64  <-- NEW
    Q_inv: jnp.ndarray,         # (N,2,2) complex128
    k: jnp.ndarray,             # (N,) float64
    r2: jnp.ndarray,            # (P,2) float64
    *,
    tile_pixels: int = 16,
    tile_beams: int = 16,
):
    """
    Computes sum_b amplitude_b * exp( i * k_b * (pathlength_b
                + delta·dr_b + 0.5 * delta^T Q_inv_b delta) )
    where delta = (r - r_b).
    """

    # Shapes
    N = r_centre.shape[0]
    P = r2.shape[0]

    # Dtypes
    r_dtype = jnp.float64
    c_dtype = jnp.complex128

    # --- Host copies ---
    r2_f = jnp.asarray(r2, r_dtype)
    r_centref = jnp.asarray(r_centre, r_dtype)
    drf = jnp.asarray(dr, r_dtype)
    Qc = jnp.asarray(Q_inv, c_dtype)
    Qc = 0.5 * (Qc + jnp.swapaxes(Qc, -1, -2))  # symmetrize

    kf = jnp.asarray(k, r_dtype)
    path = jnp.asarray(pathlength, r_dtype)

    amp = jnp.asarray(amplitude, c_dtype)
    amp_r = jnp.real(amp)
    amp_i = jnp.imag(amp)

    # Extract Q_inv real/imag
    def split_re_im(x):
        return jnp.real(x).astype(r_dtype), jnp.imag(x).astype(r_dtype)

    Qxxr, Qxxi = split_re_im(Qc[:, 0, 0])
    Qxyr, Qxyi = split_re_im(Qc[:, 0, 1])
    Qyxr, Qyxi = split_re_im(Qc[:, 1, 0])
    Qyyr, Qyyi = split_re_im(Qc[:, 1, 1])

    # Extract slopes
    drx = drf[:, 0]
    dry = drf[:, 1]

    # Tiling parameters
    T_PIX = int(tile_pixels)
    T_BEAMS = int(tile_beams)

    # Output buffers
    out_shape = (
        jax.ShapeDtypeStruct((P,), r_dtype),
        jax.ShapeDtypeStruct((P,), r_dtype),
    )

    # --- Triton kernel ---
    def kernel(r2_ref, r_m_ref,
               drx_ref, dry_ref,
               Qxxr_ref, Qxxi_ref, Qxyr_ref, Qxyi_ref,
               Qyxr_ref, Qyxi_ref, Qyyr_ref, Qyyi_ref,
               amp_r_ref, amp_i_ref,
               path_ref, k_ref,
               P_ref, N_ref,
               out_re_ref, out_im_ref):

        pid = pl.program_id(axis=0)
        pix_idx = pid * T_PIX + jnp.arange(T_PIX, dtype=jnp.int32)

        P_rt = pl.load(P_ref, ())
        N_rt = pl.load(N_ref, ())
        pix_mask = pix_idx < P_rt

        # Detector points
        x = pl.load(r2_ref, (pix_idx, 0), mask=pix_mask, other=0.0)
        y = pl.load(r2_ref, (pix_idx, 1), mask=pix_mask, other=0.0)

        acc_re = jnp.zeros((T_PIX,), r_dtype)
        acc_im = jnp.zeros((T_PIX,), r_dtype)

        def step(t, acc_re, acc_im):
            b_idx = t * T_BEAMS + jnp.arange(T_BEAMS, dtype=jnp.int32)
            b_mask = b_idx < N_rt

            # centers
            mx = pl.load(r_m_ref, (b_idx, 0), mask=b_mask, other=0.0)
            my = pl.load(r_m_ref, (b_idx, 1), mask=b_mask, other=0.0)

            # slopes
            drx_b = pl.load(drx_ref, (b_idx,), mask=b_mask, other=0.0)
            dry_b = pl.load(dry_ref, (b_idx,), mask=b_mask, other=0.0)

            # Q_inv
            Qxx_r = pl.load(Qxxr_ref, (b_idx,), mask=b_mask, other=0.0)
            Qxx_i = pl.load(Qxxi_ref, (b_idx,), mask=b_mask, other=0.0)
            Qxy_r = pl.load(Qxyr_ref, (b_idx,), mask=b_mask, other=0.0)
            Qxy_i = pl.load(Qxyi_ref, (b_idx,), mask=b_mask, other=0.0)
            Qyx_r = pl.load(Qyxr_ref, (b_idx,), mask=b_mask, other=0.0)
            Qyx_i = pl.load(Qyxi_ref, (b_idx,), mask=b_mask, other=0.0)
            Qyy_r = pl.load(Qyyr_ref, (b_idx,), mask=b_mask, other=0.0)
            Qyy_i = pl.load(Qyyi_ref, (b_idx,), mask=b_mask, other=0.0)

            # amplitude
            ar_b = pl.load(amp_r_ref, (b_idx,), mask=b_mask, other=0.0)
            ai_b = pl.load(amp_i_ref, (b_idx,), mask=b_mask, other=0.0)

            # pathlength
            path_b = pl.load(path_ref, (b_idx,), mask=b_mask, other=0.0)

            # k
            k_b = pl.load(k_ref, (b_idx,), mask=b_mask, other=0.0)

            # deltas
            dx = x[:, None] - mx[None, :]
            dy = y[:, None] - my[None, :]

            # linear term
            linear = dx * drx_b[None, :] + dy * dry_b[None, :]

            # quadratic
            cross_r = (Qxy_r + Qyx_r)[None, :] * dx * dy
            cross_i = (Qxy_i + Qyx_i)[None, :] * dx * dy

            quad_r = dx**2 * Qxx_r[None, :] + cross_r + dy**2 * Qyy_r[None, :]
            quad_i = dx**2 * Qxx_i[None, :] + cross_i + dy**2 * Qyy_i[None, :]

            # Total action: S = pathlength + linear + 0.5 * quad
            S_real = path_b[None, :] + linear + 0.5 * quad_r
            S_imag = 0.5 * quad_i

            phase = k_b[None, :] * S_real
            atten = jnp.exp(-k_b[None, :] * S_imag)

            # cos/sin
            s = jnp.sin(phase)
            c = jnp.cos(phase)

            # amplitude multiplication
            real_tb = atten * (ar_b[None, :] * c - ai_b[None, :] * s)
            imag_tb = atten * (ar_b[None, :] * s + ai_b[None, :] * c)

            real_tb = jnp.where(b_mask[None, :], real_tb, 0.0)
            imag_tb = jnp.where(b_mask[None, :], imag_tb, 0.0)

            acc_re = acc_re + jnp.sum(real_tb, axis=1)
            acc_im = acc_im + jnp.sum(imag_tb, axis=1)

            return t + 1, acc_re, acc_im

        # number of beam tiles at runtime
        num_tiles_rt = (N_rt + T_BEAMS - 1) // T_BEAMS
        state0 = (jnp.int32(0), acc_re, acc_im)

        def cond_fun(s):
            t, _, _ = s
            return t < num_tiles_rt

        def body_fun(s):
            t, acc_re, acc_im = s
            return step(t, acc_re, acc_im)

        _, acc_re, acc_im = lax.while_loop(cond_fun, body_fun, state0)

        pl.store(out_re_ref, (pix_idx,), acc_re, mask=pix_mask)
        pl.store(out_im_ref, (pix_idx,), acc_im, mask=pix_mask)

    # --- launch ---
    grid_n = (P + tile_pixels - 1) // tile_pixels

    out_re, out_im = pl.pallas_call(
        kernel, out_shape=out_shape, grid=(grid_n,)
    )(
        r2_f, r_centref,
        drx, dry,
        Qxxr, Qxxi, Qxyr, Qxyi,
        Qyxr, Qyxi, Qyyr, Qyyi,
        amp_r, amp_i,
        path, kf,
        jnp.asarray(P, jnp.int32),
        jnp.asarray(N, jnp.int32),
    )

    return out_re + 1j * out_im


def evaluate_gaussians_gpu_kernel_wrapper(
    gaussian_ray,
    grid,
    *,
    tile_pixels: int = 16,
    tile_beams: int = 16,
):
    r, dr, amp, path, Q_inv, k = _prepare_gaussian_params(gaussian_ray)
    r2 = grid.coords

    fld = evaluate_gaussians_gpu_kernel(
        r, dr, amp, path, Q_inv, k, r2,
        tile_pixels=tile_pixels,
        tile_beams=tile_beams,
    )
    return fld.reshape(grid.shape)


evaluate_gaussians_gpu_kernel = jax.jit(
    evaluate_gaussians_gpu_kernel, static_argnames=["tile_pixels", "tile_beams"]
)


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
    r, dr, amp, pathlength, Q_inv, k = _prepare_gaussian_params(gaussian_ray)
    r2 = grid.coords
    P = r2.shape[0]
    init = jnp.zeros((P,), dtype=jnp.complex128)

    xs = (r, dr, amp, pathlength, Q_inv, k)

    def f_element(x):
        r_c, dr_c, amp_c, pathlength_c, Q_inv_c, k_c = x
        return _beam_field(r_c, dr_c, amp_c, pathlength_c, Q_inv_c, k_c, det_xy=r2)

    out = map_reduce(f_element, jnp.add, init, xs, batch_size=batch_size)
    return out.reshape(grid.shape)


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


evaluate_gaussians_jax_scan = jax.jit(evaluate_gaussians_jax_scan,
                                      static_argnames=("batch_size", "grid"))


def map_reduce(f, reducer, init, xs, *, batch_size: int | None = None):
    def scan_fn(acc_inner, x):
        # combine f and reducer into function appropriate for normal lax.scan in reduce-only mode
        return reducer(acc_inner, f(x)), None

    if batch_size is not None:
        scan_xs, remainder_xs = _batch_and_remainder(xs, batch_size)

        def reduce_chunk(acc, x):
            # Reduce x into acc, assuming x have already been f'dGauss
            return reducer(acc, x), None

        def map_reduce_chunk(acc, x):
            #  Vmap apply f to a chunk of x's, then reduce them sequentially into acc
            elements = jax.vmap(f)(x)
            return lax.scan(reduce_chunk, acc, elements)

        if scan_xs is not None:
            # Map f over each chunk of xs, and reduce each sequentially into init
            acc, _ = lax.scan(map_reduce_chunk, init, scan_xs)
        else:
            acc = init

        if remainder_xs is not None:
            # normal scan-reduce the remainder chunk into acc (could also be vmapped?)
            acc, _ = lax.scan(scan_fn, acc, remainder_xs)

        return acc

    acc, _ = lax.scan(scan_fn, init, xs)
    return acc
