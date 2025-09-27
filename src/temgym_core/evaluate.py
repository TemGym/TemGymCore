import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax import lax

from temgym_core.grid import Grid
from .gaussian import GaussianRayBeta, map_reduce


def eval_gaussians_gpu_kernel(
    r_m: jnp.ndarray,        # (N,2) real64
    C: jnp.ndarray,          # (N,)  complex128
    eta: jnp.ndarray,        # (N,2) complex128
    Q_inv: jnp.ndarray,      # (N,2,2) complex128
    k: jnp.ndarray,          # (N,)  real64 (vector)
    r2: jnp.ndarray,         # (P,2) real64
    *,
    tile_pixels: int = 16,
    tile_beams: int = 16,
):  # Fastest solution so far but no grads
    N = r_m.shape[0]  # Num Beams
    P = r2.shape[0]  # Num Pixels

    r_dtype = jnp.float64
    c_dtype = jnp.complex128

    k = jnp.asarray(k, r_dtype).reshape((N,))
    r2_f = r2.astype(r_dtype)
    r_m_f = r_m.astype(r_dtype)
    C = C.astype(c_dtype)
    eta = eta.astype(c_dtype)
    Q_inv = Q_inv.astype(c_dtype)

    def split_re_im(xc):
        return jnp.real(xc).astype(r_dtype), jnp.imag(xc).astype(r_dtype)

    C_re, C_im = split_re_im(C)
    eta_re, eta_im = split_re_im(eta)
    Q_re,  Q_im = split_re_im(Q_inv)

    T_PIX = int(tile_pixels)
    T_BEAMS = int(tile_beams)

    out_shape = (
        jax.ShapeDtypeStruct((P,), r_dtype),  # real
        jax.ShapeDtypeStruct((P,), r_dtype),  # imag
    )

    def kernel(r2_ref, r_m_ref,
               eta_re_ref, eta_im_ref,
               Q_re_ref, Q_im_ref,
               C_re_ref, C_im_ref,
               k_vec_ref,
               P_ref, N_ref,
               out_re_ref, out_im_ref):

        pid = pl.program_id(axis=0)

        pix_idx = pid * T_PIX + jnp.arange(T_PIX, dtype=jnp.int32)
        P_rt = pl.load(P_ref, ())  # run time initialisation
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

            etx_re = pl.load(eta_re_ref, (b_idx, 0), mask=b_mask, other=0.0)
            ety_re = pl.load(eta_re_ref, (b_idx, 1), mask=b_mask, other=0.0)
            etx_im = pl.load(eta_im_ref, (b_idx, 0), mask=b_mask, other=0.0)
            ety_im = pl.load(eta_im_ref, (b_idx, 1), mask=b_mask, other=0.0)

            Qxx_re = pl.load(Q_re_ref, (b_idx, 0, 0), mask=b_mask, other=0.0)
            Qxy_re = pl.load(Q_re_ref, (b_idx, 0, 1), mask=b_mask, other=0.0)
            Qyx_re = pl.load(Q_re_ref, (b_idx, 1, 0), mask=b_mask, other=0.0)
            Qyy_re = pl.load(Q_re_ref, (b_idx, 1, 1), mask=b_mask, other=0.0)

            Qxx_im = pl.load(Q_im_ref, (b_idx, 0, 0), mask=b_mask, other=0.0)
            Qxy_im = pl.load(Q_im_ref, (b_idx, 0, 1), mask=b_mask, other=0.0)
            Qyx_im = pl.load(Q_im_ref, (b_idx, 1, 0), mask=b_mask, other=0.0)
            Qyy_im = pl.load(Q_im_ref, (b_idx, 1, 1), mask=b_mask, other=0.0)

            Cr = pl.load(C_re_ref, (b_idx,), mask=b_mask, other=0.0)
            Ci = pl.load(C_im_ref, (b_idx,), mask=b_mask, other=0.0)
            k_tile = pl.load(k_vec_ref, (b_idx,), mask=b_mask, other=0.0)

            dx = x[:, None] - mx[None, :]
            dy = y[:, None] - my[None, :]

            lin_re = dx * etx_re[None, :] + dy * ety_re[None, :]
            lin_im = dx * etx_im[None, :] + dy * ety_im[None, :]

            vx_re = Qxx_re[None, :] * dx + Qxy_re[None, :] * dy
            vy_re = Qyx_re[None, :] * dx + Qyy_re[None, :] * dy
            vx_im = Qxx_im[None, :] * dx + Qxy_im[None, :] * dy
            vy_im = Qyx_im[None, :] * dx + Qyy_im[None, :] * dy

            quad_re = dx * vx_re + dy * vy_re
            quad_im = dx * vx_im + dy * vy_im

            a = (lin_re + 0.5 * quad_re) * k_tile[None, :]
            b = (lin_im + 0.5 * quad_im) * k_tile[None, :]
            b = jnp.clip(b, a_min=-700.0, a_max=0.0)

            s, c = jnp.sin(a), jnp.cos(a)
            eb = jnp.exp(b)

            real_tb = Cr[None, :] * eb * c + Ci[None, :] * eb * s
            imag_tb = Ci[None, :] * eb * c - Cr[None, :] * eb * s

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

    # Static grid size computed on host (Python int)
    grid_n = (int(P) + int(tile_pixels) - 1) // T_PIX

    out_re, out_im = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=(grid_n,),
    )(
        r2_f, r_m_f,
        * (eta_re, eta_im),
        * (Q_re,  Q_im),
        * (C_re,  C_im),
        k,
        jnp.asarray(P, dtype=jnp.int32),   # runtime pixels
        jnp.asarray(N, dtype=jnp.int32),   # runtime beams
    )

    return (out_re + 1j * out_im).astype(c_dtype)


def eval_gaussians_gpu_kernel_wrapper(
    gaussian_ray: GaussianRayBeta,
    grid: Grid,
    *,
    tile_pixels: int = 16,  # gpu pixel tiles to evaluate at once
    tile_beams: int = 16,  # gpu beam tiles to evaluate at once
):
    r_m = gaussian_ray.r_xy
    C = gaussian_ray.C
    eta = gaussian_ray.eta
    Q_inv = gaussian_ray.Q_inv
    r2 = grid.coords
    k = gaussian_ray.k

    fld = eval_gaussians_gpu_kernel(
        r_m, C, eta, Q_inv, k, r2,
        tile_pixels=tile_pixels,
        tile_beams=tile_beams,
    )
    return fld.reshape(grid.shape)


eval_gaussians_gpu_kernel = jax.jit(
    eval_gaussians_gpu_kernel, static_argnames=["tile_pixels", "tile_beams"]
)


@jax.custom_vjp
def eval_gaussians_differentiable(r_m, C, eta, Q_inv, k, r2):
    # Forward = fast Pallas kernel (yours)
    return eval_gaussians_gpu_kernel(
        r_m, C, eta, Q_inv, k, r2
    )


def _fwd(r_m, C, eta, Q_inv, k, r2):
    y = eval_gaussians_gpu_kernel(
        r_m, C, eta, Q_inv, k, r2
    )
    # Save primals needed by backward; small and simple is fine
    residual = (r_m, C, eta, Q_inv, k, r2)
    return y, (residual,)


def _bwd(static, cotangent):
    # cotangent is dL/dy (complex128, shape (P,))
    (r_m, C, eta, Q_inv, k, r2), = static

    def ref(r_m, C, eta, Q_inv, k, r2):
        return eval_gaussians_ref(r_m, C, eta, Q_inv, k, r2)

    # Build VJP from the pure-JAX reference
    _, vjp_fun = jax.vjp(ref, r_m, C, eta, Q_inv, k, r2)
    grads = vjp_fun(cotangent)   # tuple matching inputs

    # No gradients for static kwargs
    return (*grads,)


eval_gaussians_differentiable.defvjp(_fwd, _bwd)


def _beam_field(r_centre, init_amp, S_const, S_lin, S_quad, k, det_xy):
    """
    Field = init_amp * exp(imag(E)) * exp(i * (-real(E))) with E = k * S_total.
    For attenuation, require imag(S_total) <= 0 (so imag(E) <= 0).
    S_total = S_const + lin + 0.5 * quad, where S_* are complex lengths.
    """
    # Shifted detector to beam coordinates
    delta = det_xy - r_centre

    # Taylor expansion terms of the action added together
    delta_S_lin = jnp.sum(delta * S_lin, axis=-1)
    delta_S_quad = jnp.sum((delta @ S_quad) * delta, axis=-1)
    S_total = S_const + delta_S_lin + 0.5 * delta_S_quad

    E = k * S_total

    A_tot = jnp.imag(E)
    # Prevent overflow in exp(A_tot) by clipping to a minimum value
    # Note: exp(-700) ~ 9.859e-305, which is small but not underflowing float64
    final_amp = jnp.exp(-A_tot)  # Prevent overflow
    final_phase = -jnp.real(E)

    return init_amp * final_amp * jnp.exp(1j * k * final_phase)


def _beam_field_outer(xs, r2):
    r_m_i, C_i, S_const_i, S_lin_i, S_quad_i, k_i = xs
    return _beam_field(r_m_i, C_i, S_const_i, S_lin_i, S_quad_i, k_i, r2)


def evaluate_gaussians_jax_scan(
    gaussian_ray: GaussianRayBeta,
    grid: Grid,
    *,
    batch_size: int | None = 128,
):  # Faster solution but not fastest, with grads
    r_m = gaussian_ray.r_xy
    C = gaussian_ray.C
    S_const = gaussian_ray.S.const
    S_lin = gaussian_ray.S.lin
    S_quad = gaussian_ray.S.quad
    k = gaussian_ray.k

    r2 = grid.coords
    P = r2.shape[0]
    init = jnp.zeros((P,), dtype=jnp.complex128)

    def f_pack(xs):
        return _beam_field_outer(xs, r2)

    xs = (r_m, C, S_const, S_lin, S_quad, k)
    out = map_reduce(f_pack, jnp.add, init, xs, batch_size=batch_size)
    return out.reshape(grid.shape)


evaluate_gaussians_jax_scan = jax.jit(
    evaluate_gaussians_jax_scan, static_argnames=["batch_size", "grid"]
)


# Slow solution for testing and debugging
def evaluate_gaussians_for(
    gaussian_ray: GaussianRayBeta,
    grid: Grid,
):
    """
    A simple for-loop based evaluation for testing and debugging.
    This is not JIT-compatible and will be slow.
    """
    # Extract properties for all packets. These have a leading batch dimension.
    r_m = gaussian_ray.r_xy
    C = gaussian_ray.C
    S_const = gaussian_ray.S.const
    S_lin = gaussian_ray.S.lin
    S_quad = gaussian_ray.S.quad
    k = gaussian_ray.k

    # Grid coordinates where the field is evaluated.
    r2 = grid.coords
    P = r2.shape[0]  # Total number of grid points

    if r_m.ndim == 1:
        num_packets = 1
    else:
        num_packets = len(r_m)

    # Initialize the total field accumulator.
    total_field = jnp.zeros((P,), dtype=jnp.complex128)

    # Iterate through each Gaussian packet one by one.
    if num_packets == 1:
        return _beam_field(
            r_centre=r_m,
            init_amp=C,
            S_const=S_const,
            S_quad=S_quad,
            S_lin=S_lin,
            k=k,
            det_xy=r2,
        ).reshape(grid.shape)
    else:
        for i in range(num_packets):
            # Calculate the field for the i-th packet on the entire grid.
            field_i = _beam_field(
                r_centre=r_m[i],
                init_amp=C[i],
                S_const=S_const[i],
                S_quad=S_quad[i],
                S_lin=S_lin[i],
                k=k[i],
                det_xy=r2,
            )
            # Add its contribution to the total field.
            total_field += field_i

        # Reshape the final flat array to match the grid's 2D shape.
        return total_field.reshape(grid.shape)
