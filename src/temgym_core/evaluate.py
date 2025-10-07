import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax import lax

from temgym_core.grid import Grid
from .gaussian import GaussianRayBeta, map_reduce


def evaluate_gaussians_gpu_kernel(
    r_centre: jnp.ndarray,   # (N,2) float64
    C: jnp.ndarray,          # (N,)  complex128  (init_amp)
    S_const: jnp.ndarray,    # (N,)  float64
    S_lin: jnp.ndarray,      # (N,2) float64
    S_quad: jnp.ndarray,     # (N,2,2) float64
    k: jnp.ndarray,          # (N,)  float64
    r2: jnp.ndarray,         # (P,2) float64 (detector coords)
    *,
    tile_pixels: int = 16,
    tile_beams: int = 16,
):  # Matches new beam_field; still no grads
    N = r_centre.shape[0]
    P = r2.shape[0]

    r_dtype = jnp.float64
    c_dtype = jnp.complex128

    # dtype sanity (and flatten shapes the way we need)
    r2_f = r2.astype(r_dtype)
    r_centref = r_centre.astype(r_dtype)
    S_constf = S_const.astype(r_dtype).reshape((N,))
    S_linf = S_lin.astype(r_dtype)
    S_quadf = S_quad.astype(r_dtype)
    kf = jnp.asarray(k, r_dtype).reshape((N,))
    Cc = C.astype(c_dtype)

    def split_re_im(xc):
        return jnp.real(xc).astype(r_dtype), jnp.imag(xc).astype(r_dtype)

    Cr, Ci = split_re_im(Cc)

    T_PIX = int(tile_pixels)
    T_BEAMS = int(tile_beams)

    out_shape = (
        jax.ShapeDtypeStruct((P,), r_dtype),  # real
        jax.ShapeDtypeStruct((P,), r_dtype),  # imag
    )

    def kernel(r2_ref, r_m_ref,
               Slin_ref, Squad_ref, Sconst_ref, k_vec_ref,
               Cr_ref, Ci_ref,
               P_ref, N_ref,
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

            mx = pl.load(r_m_ref, (b_idx, 0), mask=b_mask, other=0.0)
            my = pl.load(r_m_ref, (b_idx, 1), mask=b_mask, other=0.0)

            slx = pl.load(Slin_ref, (b_idx, 0), mask=b_mask, other=0.0)
            sly = pl.load(Slin_ref, (b_idx, 1), mask=b_mask, other=0.0)

            Qxx = pl.load(Squad_ref, (b_idx, 0, 0), mask=b_mask, other=0.0)
            Qxy = pl.load(Squad_ref, (b_idx, 0, 1), mask=b_mask, other=0.0)
            Qyx = pl.load(Squad_ref, (b_idx, 1, 0), mask=b_mask, other=0.0)
            Qyy = pl.load(Squad_ref, (b_idx, 1, 1), mask=b_mask, other=0.0)

            Sc = pl.load(Sconst_ref, (b_idx,), mask=b_mask, other=0.0)
            kv = pl.load(k_vec_ref, (b_idx,), mask=b_mask, other=0.0)

            Crb = pl.load(Cr_ref, (b_idx,), mask=b_mask, other=0.0)
            Cib = pl.load(Ci_ref, (b_idx,), mask=b_mask, other=0.0)

            dx = x[:, None] - mx[None, :]
            dy = y[:, None] - my[None, :]

            # linear term: delta · S_lin
            linear = dx * slx[None, :] + dy * sly[None, :]

            # quadratic term: delta · (S_quad @ delta)
            vx = Qxx[None, :] * dx + Qxy[None, :] * dy
            vy = Qyx[None, :] * dx + Qyy[None, :] * dy
            quad = dx * vx + dy * vy

            # phase a = k * (S_const + linear + 0.5 * quad)
            a = kv[None, :] * (Sc[None, :] + linear + 0.5 * quad)

            s, c = jnp.sin(a), jnp.cos(a)

            # (Cr + i Ci) * (c + i s) = (Cr*c - Ci*s) + i (Cr*s + Ci*c)
            real_tb = Crb[None, :] * c - Cib[None, :] * s
            imag_tb = Crb[None, :] * s + Cib[None, :] * c

            real_tb = jnp.where(b_mask[None, :], real_tb, 0.0)
            imag_tb = jnp.where(b_mask[None, :], imag_tb, 0.0)

            acc_re = acc_re + jnp.sum(real_tb, axis=1)
            acc_im = acc_im + jnp.sum(imag_tb, axis=1)
            return t + jnp.int32(1), acc_re, acc_im

        # runtime number of tiles
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
        r2_f, r_centref,
        S_linf, S_quadf, S_constf, kf,
        Cr, Ci,
        jnp.asarray(P, dtype=jnp.int32),
        jnp.asarray(N, dtype=jnp.int32),
    )

    return (out_re + 1j * out_im).astype(c_dtype)


def evaluate_gaussians_gpu_kernel_wrapper(
    gaussian_ray,
    grid,
    *,
    tile_pixels: int = 16,
    tile_beams: int = 16,
):
    r_centre, C, S_const, S_lin, S_quad, k = _prepare_gaussian_params(gaussian_ray)
    r2 = grid.coords

    fld = evaluate_gaussians_gpu_kernel(
        r_centre, C, S_const, S_lin, S_quad, k, r2,
        tile_pixels=tile_pixels,
        tile_beams=tile_beams,
    )
    return fld.reshape(grid.shape)


evaluate_gaussians_gpu_kernel = jax.jit(
    evaluate_gaussians_gpu_kernel, static_argnames=["tile_pixels", "tile_beams"]
)


def _prepare_gaussian_params(gaussian_ray: GaussianRayBeta):
    r_m = jnp.asarray(gaussian_ray.r_xy)
    if r_m.ndim == 1:
        r_m = r_m[None, :]
    if r_m.ndim != 2 or r_m.shape[1] != 2:
        raise ValueError("Expected r_xy with shape (N, 2).")
    n = r_m.shape[0]

    C = jnp.broadcast_to(jnp.atleast_1d(jnp.asarray(gaussian_ray.C)), (n,))
    S_const = jnp.broadcast_to(jnp.atleast_1d(jnp.asarray(gaussian_ray.S.const)), (n,))

    S_lin = jnp.asarray(gaussian_ray.S.lin)
    if S_lin.ndim == 1:
        S_lin = S_lin[None, :]
    if S_lin.shape[-1] != 2:
        raise ValueError("Expected S.lin with shape (..., 2).")
    S_lin = jnp.broadcast_to(S_lin, (n, 2))

    S_quad = jnp.asarray(gaussian_ray.S.quad)
    if S_quad.ndim == 2:
        S_quad = S_quad[None, :, :]
    if S_quad.shape[-2:] != (2, 2):
        raise ValueError("Expected S.quad with shape (..., 2, 2).")
    S_quad = jnp.broadcast_to(S_quad, (n, 2, 2))

    k = jnp.broadcast_to(jnp.atleast_1d(jnp.asarray(gaussian_ray.k)), (n,))

    return r_m, C, S_const, S_lin, S_quad, k


def _beam_field(r_centre, init_amp, S_const, S_lin, S_quad, k, det_xy):
    delta = det_xy - r_centre
    linear = jnp.sum(delta * S_lin, axis=-1)
    quadratic = jnp.sum((delta @ S_quad) * delta, axis=-1)
    phase = S_const + linear + 0.5 * quadratic
    return init_amp * jnp.exp(1j * k * phase)


def evaluate_gaussians_jax_scan(
    gaussian_ray: GaussianRayBeta,
    grid: Grid,
    *,
    batch_size: int | None = 128,
):
    r_centre, C, S_const, S_lin, S_quad, k = _prepare_gaussian_params(gaussian_ray)
    r2 = grid.coords
    P = r2.shape[0]
    init = jnp.zeros((P,), dtype=jnp.complex128)

    xs = (r_centre, C, S_const, S_lin, S_quad, k)

    def f_element(x):
        r_c, Cc, Sc, Slin, Squad, kc = x
        return _beam_field(r_c, Cc, Sc, Slin, Squad, kc, det_xy=r2)

    out = map_reduce(f_element, jnp.add, init, xs, batch_size=batch_size)
    return out.reshape(grid.shape)


def evaluate_gaussians_for(
    gaussian_ray: GaussianRayBeta,
    grid: Grid,
):
    r_centre, C, S_const, S_lin, S_quad, k = _prepare_gaussian_params(gaussian_ray)
    r2 = grid.coords
    n = r_centre.shape[0]
    total_field = jnp.zeros((r2.shape[0],), dtype=jnp.complex128)
    for i in range(n):
        field = _beam_field(
            r_centre[i],
            C[i],
            S_const[i],
            S_lin[i],
            S_quad[i],
            k[i],
            r2
        )
        total_field = total_field + field
    return total_field.reshape(grid.shape)


evaluate_gaussians_jax_scan = jax.jit(evaluate_gaussians_jax_scan,
                                      static_argnames=("batch_size", "grid"))
