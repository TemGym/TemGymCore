"""Frozen copy of the historical dense Pallas kernel (metre space, exact sum).

This is ``evaluate._evaluate_gaussians_gpu_kernel_impl`` and ``_pack_gaussian_params`` as they were
before the ``temgym_core.render`` package replaced them.  It is kept only as an independent
oracle: the metre-space renderer must reproduce it to rounding, and the accuracy of the
pixel-index frame is measured against it.  Do not import it from the package.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

try:
    from jax.experimental.pallas import triton as plgpu
except ImportError:
    plgpu = None


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


legacy_dense_kernel = jax.jit(
    _evaluate_gaussians_gpu_kernel_impl,
    static_argnames=["tile_pixels", "tile_beams"],
)
