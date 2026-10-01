"""Coefficient algebra shared by the planner (NumPy, concrete) and the plan builder (JAX, traceable).

Every function that takes ``xp`` works with ``numpy`` or ``jax.numpy``. ``A, B, C, D`` are the 2x2
blocks of the ray map ``[r; theta] -> M [r; theta] + [a; b]`` in ``(x, y)`` order; lattices are
``r = centre + P @ mu`` with centred index ``mu``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def inv2(xp, M):
    """Explicit 2x2 inverse: differentiable and valid on tracers."""
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    adj = xp.stack((xp.stack((M[1, 1], -M[0, 1])), xp.stack((-M[1, 0], M[0, 0]))))
    return adj / det


def sym(M):
    """Symmetric part of a square matrix."""
    return 0.5 * (M + M.T)


def split_matrix(matrix):
    """Split a ray matrix into its ``A, B, C, D`` blocks and offsets ``a, b``.

    Parameters
    ----------
    matrix : (4, 4) or (5, 5) array
        Ray map on ``[x, y, theta_x, theta_y, (1)]``, as returned by
        :func:`temgym_core.run.column_matrix`. The last column of a 5x5 matrix holds the
        affine offset ``t`` of ``z' = M z + t``.

    Returns
    -------
    A, B, C, D : (2, 2) arrays
        Position-to-position [1], slope-to-position [m/rad], position-to-slope [rad/m] and
        slope-to-slope [1] blocks.
    a, b : (2,) arrays
        Position offset [m] and slope offset [rad].
    """
    M = jnp.asarray(matrix, dtype=jnp.float64)
    if M.shape == (5, 5):
        abcd, t = M[:4, :4], M[:4, 4]
    elif M.shape == (4, 4):
        abcd, t = M, jnp.zeros(4)
    else:
        raise ValueError(f"matrix must be 4x4 or 5x5, got {M.shape}")
    return abcd[:2, :2], abcd[:2, 2:], abcd[2:, :2], abcd[2:, 2:], t[:2], t[2:]


def mft_terms(xp, A, B, D, lam, Pin, rc, Pout, rcp, a, b):
    """Coefficients of the lattice-adapted matrix Fourier transform kernel.

    The kernel phase with centred indices ``mu`` (input) and ``mu'`` (output) is

        k W = phi_in(mu) + phi_out(mu') - 2 pi mu'^T Gamma mu,
        phi_in(mu)   = 1/2 mu^T Qin_k mu + lin_in . mu
        phi_out(mu') = 1/2 mu'^T Qout_k mu' + lin_out . mu' + const

    including the Heisenberg-Weyl offset ``(a, b)`` (position shift ``a``, slope boost ``b``).
    All returned phases are in rad or rad per index.
    """
    k = 2.0 * np.pi / lam
    Bi = inv2(xp, B)
    G = sym(Bi @ A)
    H = sym(D @ Bi)
    Gam = (Pout.T @ Bi.T @ Pin) / lam
    rho0 = rcp - a  # position at which the offset-free operator is evaluated
    Qin_k = k * (Pin.T @ G @ Pin)
    Qout_k = k * (Pout.T @ H @ Pout)
    lin_in = k * (Pin.T @ G @ rc - Pin.T @ Bi @ rho0)
    lin_out = k * (Pout.T @ H @ rho0 - Pout.T @ Bi.T @ rc + Pout.T @ b)
    const = k * (0.5 * rc @ G @ rc - rho0 @ Bi.T @ rc + 0.5 * rho0 @ H @ rho0 + b @ rcp)
    return dict(k=k, Bi=Bi, G=G, H=H, Gam=Gam, Qin_k=Qin_k, Qout_k=Qout_k,
                lin_in=lin_in, lin_out=lin_out, const=const)


def ni_terms(xp, variant, A, B, C, D, lam, Pin, rc, a, b):
    """Coefficients of the near-image branch.

    variant ``"I"``  (drift first):  M = L(C A^-1) Lin(A) Fr(A^-1 B), output lattice ``A P``.
    variant ``"II"`` (drift last):   M = Fr(B D^-1) Lin(D^-T) L(D^-1 C), output lattice ``D^-T P``.
    """
    k = 2.0 * np.pi / lam
    if variant == "I":
        Ainv = inv2(xp, A)
        S = sym(Ainv @ B)
        T = sym(C @ Ainv)
        Al = A
    elif variant == "II":
        Dinv = inv2(xp, D)
        S = sym(B @ Dinv)
        T = sym(Dinv @ C)
        Al = Dinv.T
    else:
        raise ValueError(variant)
    Pout = Al @ Pin
    rc_out_noff = Al @ rc
    rc_out = rc_out_noff + a
    Pdrift = Pin if variant == "I" else Pout
    Pd_inv = inv2(xp, Pdrift)
    Lam = lam * (Pd_inv @ S @ Pd_inv.T)  # index-space Fresnel matrix
    if variant == "I":
        Pch, rch = Pout, rc_out_noff
    else:
        Pch, rch = Pin, rc
    Qc_k = k * (Pch.T @ T @ Pch)
    lin_c = k * (Pch.T @ T @ rch)
    const_c = 0.5 * k * (rch @ T @ rch)
    lin_b = k * (Pout.T @ b)
    const_b = k * (b @ rc_out)
    return dict(k=k, S=S, T=T, Al=Al, Pout=Pout, rc_out=rc_out, Lam=Lam,
                Qc_k=Qc_k, lin_c=lin_c, const_c=const_c, lin_b=lin_b, const_b=const_b)
