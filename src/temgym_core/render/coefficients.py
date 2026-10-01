"""Chain rule between packed-parameter tangents and the polynomial fields of the kernels.

The tangent of the coherent field with respect to a packed bundle is ``T = K o C``.
``C`` (:func:`tangent_coefficients`) is elementwise on ``(nt, N, 12)`` and turns a packed
tangent into the six complex polynomial coefficients ``w = (w1, wu, wv, wuu, wuv, wvv)`` of

    Edot_p = sum_b g_b(p) * (w1 + wu u + wv v + wuu u^2 + wuv u v + wvv v^2)_b ,

and ``K`` is the pair loop.  ``C`` is ordinary JAX, so JAX transposes it itself; the kernels
only ever contain ``K`` and its transpose, whose output is the six complex moments
``M_j = sum_p ct_p g_p mono_j`` per beam.  :func:`moments_to_packed_cotangent` is the
closed-form ``C^T`` (verified against the autodiff transpose in the tests).  Coefficient
arrays store ``(Re, Im)`` of each of the six coefficients interleaved (``(nt, N, 12)``); moment
arrays use the cotangent layout of the same real array, ``(Re M_j, -Im M_j)``.
"""

from __future__ import annotations

import jax.numpy as jnp

from .packing import AI, AR, AX, AY, MX, MY, N_COLUMNS, PXX, PXY, PYY, QXX, QXY, QYY

N_MONOMIALS = 6  # 1, u, v, u^2, u v, v^2


def tangent_coefficients(packed, tdot):
    """Polynomial coefficients ``w`` of the tangent field for packed tangents ``tdot``.

    ``Edot_p = sum_b g_b(p) * (Adot + A * Psidot)`` with ``Psi = i phi - delta`` a quadratic
    in ``(u, v)``; the centre tangent enters through ``du = -mdot_x``, ``dv = -mdot_y``.
    Linear in ``tdot``, nonlinear in ``packed``.

    Parameters
    ----------
    packed : array, shape (N, 12)
        Packed bundle.
    tdot : array, shape (nt, N, 12)
        Tangents of the packed columns.

    Returns
    -------
    array, shape (nt, N, 12)
        Interleaved ``(Re, Im)`` of the six polynomial coefficients.
    """
    packed = jnp.asarray(packed, jnp.float64)
    tdot = jnp.asarray(tdot, jnp.float64)
    p = [packed[:, i] for i in range(N_COLUMNS)]
    t = [tdot[..., i] for i in range(N_COLUMNS)]
    amp = p[AR] + 1j * p[AI]
    alpha_x, alpha_y = 1j * p[AX], 1j * p[AY]
    beta_xx = 1j * p[PXX] - p[QXX]
    beta_xy = 1j * p[PXY] - p[QXY]
    beta_yy = 1j * p[PYY] - p[QYY]
    amp_dot = t[AR] + 1j * t[AI]
    w1 = amp_dot - amp * (t[MX] * alpha_x + t[MY] * alpha_y)
    wu = amp * (1j * t[AX] - 2 * t[MX] * beta_xx - t[MY] * beta_xy)
    wv = amp * (1j * t[AY] - t[MX] * beta_xy - 2 * t[MY] * beta_yy)
    wuu = amp * (1j * t[PXX] - t[QXX])
    wuv = amp * (1j * t[PXY] - t[QXY])
    wvv = amp * (1j * t[PYY] - t[QYY])
    w = jnp.stack([w1, wu, wv, wuu, wuv, wvv], axis=-1)
    return jnp.stack([w.real, w.imag], axis=-1).reshape(w.shape[:-1] + (12,))


def moments_to_packed_cotangent(packed, moments):
    """Closed-form ``C^T``: moment array (cotangent layout) to the packed-parameter gradient.

    With ``M`` complex and ``A`` the complex amplitude,
    ``dL/d(A_r, A_i) = (Re M1, -Im M1)``, ``dL/da_x = -Im(A Mu)``, ``dL/dp_.. = -Im(A M_..)``,
    ``dL/dq_.. = -Re(A M_..)`` and
    ``dL/dm_x = -Re(A [i a_x M1 + (2i p_xx - 2 q_xx) Mu + (i p_xy - q_xy) Mv])``.

    Parameters
    ----------
    packed : array, shape (N, 12)
        Packed bundle.
    moments : array, shape (nt, N, 12)
        Output of the adjoint kernel for ``nt`` cotangents.

    Returns
    -------
    array, shape (nt, N, 12)
        Gradient with respect to the packed columns.
    """
    packed = jnp.asarray(packed, jnp.float64)
    m = moments.reshape(moments.shape[:-1] + (N_MONOMIALS, 2))
    m = m[..., 0] - 1j * m[..., 1]  # undo the (Re, -Im) cotangent layout
    m1, mu, mv, muu, muv, mvv = [m[..., j] for j in range(N_MONOMIALS)]
    p = [packed[:, i] for i in range(N_COLUMNS)]
    amp = p[AR] + 1j * p[AI]
    d_mx = -jnp.real(amp * (1j * p[AX] * m1 + (2j * p[PXX] - 2 * p[QXX]) * mu
                            + (1j * p[PXY] - p[QXY]) * mv))
    d_my = -jnp.real(amp * (1j * p[AY] * m1 + (1j * p[PXY] - p[QXY]) * mu
                            + (2j * p[PYY] - 2 * p[QYY]) * mv))
    cols = [None] * N_COLUMNS
    cols[MX], cols[MY] = d_mx, d_my
    cols[AX], cols[AY] = -jnp.imag(amp * mu), -jnp.imag(amp * mv)
    cols[PXX], cols[PXY], cols[PYY] = (-jnp.imag(amp * muu), -jnp.imag(amp * muv),
                                       -jnp.imag(amp * mvv))
    cols[QXX], cols[QXY], cols[QYY] = (-jnp.real(amp * muu), -jnp.real(amp * muv),
                                       -jnp.real(amp * mvv))
    cols[AR], cols[AI] = jnp.real(m1), -jnp.imag(m1)
    return jnp.stack(cols, axis=-1)
