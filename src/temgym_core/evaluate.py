import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax import lax
from temgym_core.grid import Grid
from jax._src.lax.control_flow.loops import _batch_and_remainder


def evaluate_gaussians_gpu_kernel(
    r_centre: jnp.ndarray,   # (N,2) float64
    dr: jnp.ndarray,         # (N,2) float64
    C: jnp.ndarray,          # (N,)  complex128  (init_amp)
    S_quad: jnp.ndarray,     # (N,2,2) complex128
    k: jnp.ndarray,          # (N,)  float64
    r2: jnp.ndarray,         # (P,2) float64 (detector coords)
    *,
    tile_pixels: int = 16,
    tile_beams: int = 16,
):
    """
    Computes sum_b C_b * exp(i * k_b * (delta·dr_b + 0.5 * delta^T S_quad_b delta)),
    where delta = (r - r_b), using full complex quadratic terms.
    """

    # Shapes / sizes
    N = r_centre.shape[0]
    P = r2.shape[0]

    # Dtypes
    r_dtype = jnp.float64
    c_dtype = jnp.complex128

    # ---- Host-side casting / preparation ----
    r2_f = jnp.asarray(r2, r_dtype)
    r_centref = jnp.asarray(r_centre, r_dtype)
    drf = jnp.asarray(dr, r_dtype).reshape((N, 2))

    # Keep complex parts for S_quad
    S_quadc = jnp.asarray(S_quad, c_dtype).reshape((N, 2, 2))

    # Optional symmetrization:
    S_quadc = 0.5 * (S_quadc + jnp.swapaxes(S_quadc, -1, -2))

    kf = jnp.asarray(k, r_dtype).reshape((N,))
    Cc = jnp.asarray(C, c_dtype).reshape((N,))

    # Split helper
    def split_re_im(xc):
        return jnp.real(xc).astype(r_dtype), jnp.imag(xc).astype(r_dtype)

    # C (init amplitudes)
    Cr, Ci = split_re_im(Cc)

    # dr components (real-valued)
    drx = drf[:, 0]
    dry = drf[:, 1]

    # S_quad components (xx, xy, yx, yy)
    Qxxr, Qxxi = split_re_im(S_quadc[:, 0, 0])
    Qxyr, Qxyi = split_re_im(S_quadc[:, 0, 1])
    Qyxr, Qyxi = split_re_im(S_quadc[:, 1, 0])
    Qyyr, Qyyi = split_re_im(S_quadc[:, 1, 1])

    # Tiling params
    T_PIX = int(tile_pixels)
    T_BEAMS = int(tile_beams)

    # Outputs: real & imag tiles
    out_shape = (
        jax.ShapeDtypeStruct((P,), r_dtype),  # real
        jax.ShapeDtypeStruct((P,), r_dtype),  # imag
    )

    # ---- Kernel ----
    def kernel(r2_ref, r_m_ref,
               drx_ref, dry_ref,
               # quad (re/im)
               Qxxr_ref, Qxxi_ref, Qxyr_ref, Qxyi_ref,
               Qyxr_ref, Qyxi_ref, Qyyr_ref, Qyyi_ref,
               # k and C
               Cr_ref, Ci_ref, k_vec_ref,
               # sizes
               P_ref, N_ref,
               # outputs
               out_re_ref, out_im_ref):

        pid = pl.program_id(axis=0)

        pix_idx = pid * T_PIX + jnp.arange(T_PIX, dtype=jnp.int32)
        P_rt = pl.load(P_ref, ())  # runtime sizes
        N_rt = pl.load(N_ref, ())

        pix_mask = pix_idx < P_rt

        # load pixel coords for this tile
        x = pl.load(r2_ref, (pix_idx, 0), mask=pix_mask, other=0.0)
        y = pl.load(r2_ref, (pix_idx, 1), mask=pix_mask, other=0.0)

        acc_re0 = jnp.zeros((T_PIX,), r_dtype)
        acc_im0 = jnp.zeros((T_PIX,), r_dtype)

        def step(t, acc_re, acc_im):
            b_idx = t * T_BEAMS + jnp.arange(T_BEAMS, dtype=jnp.int32)
            b_mask = b_idx < N_rt

            # centers
            mx = pl.load(r_m_ref, (b_idx, 0), mask=b_mask, other=0.0)
            my = pl.load(r_m_ref, (b_idx, 1), mask=b_mask, other=0.0)

            # slopes (real)
            drx_b = pl.load(drx_ref, (b_idx,), mask=b_mask, other=0.0)
            dry_b = pl.load(dry_ref, (b_idx,), mask=b_mask, other=0.0)

            # quad (re/im)
            Qxx_r = pl.load(Qxxr_ref, (b_idx,), mask=b_mask, other=0.0)
            Qxx_i = pl.load(Qxxi_ref, (b_idx,), mask=b_mask, other=0.0)
            Qxy_r = pl.load(Qxyr_ref, (b_idx,), mask=b_mask, other=0.0)
            Qxy_i = pl.load(Qxyi_ref, (b_idx,), mask=b_mask, other=0.0)
            Qyx_r = pl.load(Qyxr_ref, (b_idx,), mask=b_mask, other=0.0)
            Qyx_i = pl.load(Qyxi_ref, (b_idx,), mask=b_mask, other=0.0)
            Qyy_r = pl.load(Qyyr_ref, (b_idx,), mask=b_mask, other=0.0)
            Qyy_i = pl.load(Qyyi_ref, (b_idx,), mask=b_mask, other=0.0)

            # k and C
            Crb = pl.load(Cr_ref,     (b_idx,), mask=b_mask, other=0.0)
            Cib = pl.load(Ci_ref,     (b_idx,), mask=b_mask, other=0.0)
            kv = pl.load(k_vec_ref, (b_idx,), mask=b_mask, other=0.0)

            # deltas
            dx = x[:, None] - mx[None, :]
            dy = y[:, None] - my[None, :]

            # linear term
            linear = dx * drx_b[None, :] + dy * dry_b[None, :]

            # quadratic form (re & im) with symmetric cross term
            cross_r = (Qxy_r[None, :] + Qyx_r[None, :]) * dx * dy
            cross_i = (Qxy_i[None, :] + Qyx_i[None, :]) * dx * dy

            quad_r = dx**2 * Qxx_r[None, :] + cross_r + dy**2 * Qyy_r[None, :]
            quad_i = dx**2 * Qxx_i[None, :] + cross_i + dy**2 * Qyy_i[None, :]

            # action components: S = linear + 0.5 * (quad_r + i * quad_i)
            S_real = linear + 0.5 * quad_r
            S_imag = 0.5 * quad_i

            ar = kv[None, :] * S_real
            ai = kv[None, :] * S_imag

            # exp(i(ar + i ai)) = exp(-ai) * (cos ar + i sin ar)
            atten = jnp.exp(-ai)
            s, c = jnp.sin(ar), jnp.cos(ar)

            real_tb = atten * (Crb[None, :] * c - Cib[None, :] * s)
            imag_tb = atten * (Crb[None, :] * s + Cib[None, :] * c)

            # mask beams outside range
            real_tb = jnp.where(b_mask[None, :], real_tb, 0.0)
            imag_tb = jnp.where(b_mask[None, :], imag_tb, 0.0)

            acc_re = acc_re + jnp.sum(real_tb, axis=1)
            acc_im = acc_im + jnp.sum(imag_tb, axis=1)
            return t + jnp.int32(1), acc_re, acc_im

        # number of beam tiles at runtime
        num_tiles_rt = (N_rt + T_BEAMS - 1) // T_BEAMS
        state0 = (jnp.int32(0), acc_re0, acc_im0)

        def cond_fun(s):
            t, _, _ = s
            return t < num_tiles_rt

        def body_fun(s):
            t, acc_re, acc_im = s
            return step(t, acc_re, acc_im)

        _, acc_re, acc_im = lax.while_loop(cond_fun, body_fun, state0)

        pl.store(out_re_ref, (pix_idx,), acc_re, mask=pix_mask)
        pl.store(out_im_ref, (pix_idx,), acc_im, mask=pix_mask)

    # Static grid size computed on host
    grid_n = (int(P) + int(tile_pixels) - 1) // int(tile_pixels)

    out_re, out_im = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=(grid_n,),
    )(
        # positions
        r2_f, r_centref,
        # slopes
        drx, dry,
        # S_quad (re/im)
        Qxxr, Qxxi, Qxyr, Qxyi,
        Qyxr, Qyxi, Qyyr, Qyyi,
        # C (re/im) and k
        Cr, Ci, kf,
        # sizes
        jnp.asarray(P, dtype=jnp.int32),
        jnp.asarray(N, dtype=jnp.int32),
    )

    return (out_re + 1j * out_im).astype(c_dtype)


def evaluate_gaussians_gpu_kernel_wrapper(
    gaussian_ray,
    grid,
    *,
    tile_pixels: int = 32,
    tile_beams: int = 32,
):
    r_centre, dr, C, S_quad, k = _prepare_gaussian_params(gaussian_ray)
    r2 = grid.coords

    fld = evaluate_gaussians_gpu_kernel(
        r_centre, dr, C, S_quad, k, r2,
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
    C = ensure_batch(gaussian_ray.C, (), jnp.complex128)
    S_quad = ensure_batch(gaussian_ray.S2,  (2, 2), jnp.complex128)
    k = ensure_batch(gaussian_ray.k, (), jnp.float64)

    return r, dr, C, S_quad, k


def _beam_field(r, dr, C, S_quad, k, det_xy):
    delta = det_xy - r
    linear = jnp.sum(delta * dr, axis=-1)
    quadratic = jnp.sum((delta @ S_quad) * delta, axis=-1)
    S_tot = linear + 0.5 * quadratic
    expo = 1j * k * S_tot
    return C * jnp.exp(expo)


def evaluate_gaussians_jax_scan(
    gaussian_ray,
    grid: Grid,
    *,
    batch_size: int | None = 128,
):
    r, dr, C, S_quad, k = _prepare_gaussian_params(gaussian_ray)
    r2 = grid.coords
    P = r2.shape[0]
    init = jnp.zeros((P,), dtype=jnp.complex128)

    xs = (r, dr, C, S_quad, k)

    def f_element(x):
        r_c, dr_c, Cc, Squad, kc = x
        return _beam_field(r_c, dr_c, Cc, Squad, kc, det_xy=r2)

    out = map_reduce(f_element, jnp.add, init, xs, batch_size=batch_size)
    return out.reshape(grid.shape)


def evaluate_gaussians_for(
    gaussian_ray,
    grid: Grid,
):
    r, dr, C, S_quad, k = _prepare_gaussian_params(gaussian_ray)
    C = C.T
    k = k.T
    r2 = grid.coords
    n = r.shape[0]
    total_field = jnp.zeros((r2.shape[0],), dtype=jnp.complex128)
    for i in range(n):
        field = _beam_field(
            r[i],
            dr[i],
            C[i],
            S_quad[i],
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
            acc, _ = init, None

        if remainder_xs is not None:
            # normal scan-reduce the remainder chunk into acc (could also be vmapped?)
            acc, _ = lax.scan(scan_fn, acc, remainder_xs)

            return acc
