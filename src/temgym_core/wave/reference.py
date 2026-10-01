"""Analytic Gaussian references and the chief-Gaussian phase protocol.

A Gaussian field is ``amp * exp(i k [path + d.(r - rc) + 1/2 (r - rc)^T Q (r - rc)])``, the same
convention as ``gaussian.GaussianBeam`` and ``evaluate._beam_field``. :func:`collins_gaussian`
applies the *engine's* operator (lift included) to such a field in closed form, so it is an exact
reference for ``apply_lct(..., phase0=0)``. Comparing it with a chief ``GaussianBeam`` propagated
through the same components (whose amplitude follows the physical, continuous Gouy branch) gives
the constant phase ``phase0`` that turns the engine's lift into the physical field.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ._terms import inv2, split_matrix
from .lattice import is_concrete
from .lct import _reflect, lift_phase

__all__ = ["gaussian_field", "collins_gaussian", "chief_phase0", "chief_from_beam"]


def gaussian_field(positions, amp, centre, slopes, Q, wavelength, path=0.0):
    """Evaluate a Gaussian on ``positions[..., 2]`` of ``(x, y)``.

    Parameters
    ----------
    positions : array, shape (..., 2)
        Evaluation points ``(x, y)``, metres (for example ``Lattice.positions()``).
    amp : complex
        Amplitude at the centre (with ``path`` phase removed).
    centre : (2,) array
        Beam centre ``(x, y)``, metres.
    slopes : (2,) array
        Slopes ``(theta_x, theta_y)`` at the centre, rad.
    Q : (2, 2) array
        Complex curvature ``Q_inv``, 1/m; its imaginary part sets the width.
    wavelength : float
        Wavelength, metres.
    path : float, default 0
        Excess optical path at the centre, metres.

    Returns
    -------
    complex array, shape (...)
    """
    k = 2.0 * np.pi / wavelength
    delta = positions - jnp.asarray(centre)
    quad = jnp.einsum("...i,ij,...j->...", delta, jnp.asarray(Q), delta)
    return amp * jnp.exp(1j * k * (path + delta @ jnp.asarray(slopes) + 0.5 * quad))


def _sqrt_det_1_plus(SQ):
    """``det(I + S Q)^{1/2}`` on the continuous branch from ``S = 0`` for a 2x2 matrix ``S Q``.

    Each eigenvalue ``mu`` of ``S Q`` keeps ``1 + mu`` in its own half plane, so the branch is the
    product of the principal roots ``sqrt(1 + mu_1) sqrt(1 + mu_2)``. The value is the smooth
    ``sqrt(det)`` times a sign taken from that product, which is differentiable also where the
    eigenvalues coincide (a round beam through a round lens), where the eigenvalue formula is not.
    """
    det1 = (1 + SQ[0, 0]) * (1 + SQ[1, 1]) - SQ[0, 1] * SQ[1, 0]
    smooth = jnp.sqrt(det1 + 0j)
    frozen = jax.lax.stop_gradient(SQ)
    tr = frozen[0, 0] + frozen[1, 1]
    det = frozen[0, 0] * frozen[1, 1] - frozen[0, 1] * frozen[1, 0]
    root = jnp.sqrt(tr * tr / 4 - det + 0j)
    exact = jnp.sqrt(1 + tr / 2 + root + 0j) * jnp.sqrt(1 + tr / 2 - root + 0j)
    sign = jnp.where(jnp.real(exact * jnp.conj(jax.lax.stop_gradient(smooth))) < 0, -1.0, 1.0)
    return sign * smooth


def _switch(pred, when_true, when_false):
    """Branch on ``pred``: a Python ``if`` when it is concrete, ``lax.cond`` under a trace."""
    if is_concrete(pred):
        return when_true() if bool(pred) else when_false()
    return jax.lax.cond(pred, when_true, when_false)


def collins_gaussian(matrix, wavelength, amp, centre, slopes, Q, path=0.0, offset_mode="after"):
    """Apply the engine's operator to a Gaussian in closed form.

    Uses the canonical composition ``L(T) Lin(A) Fr(S)`` when ``A`` is invertible, else the kernel
    form with :func:`~temgym_core.wave.lct.lift_phase`; the two agree wherever both are defined.
    The carrier ``k * z`` is not included (it is not determined by the matrix).

    Parameters
    ----------
    matrix : (5, 5) or (4, 4) array
        Ray map on ``[x, y, theta_x, theta_y, (1)]``.
    wavelength : float
        Wavelength, metres.
    amp : complex
        Amplitude at the centre.
    centre : (2,) array
        Centre ``(x, y)``, metres.
    slopes : (2,) array
        Slopes ``(theta_x, theta_y)``, rad.
    Q : (2, 2) array
        Complex curvature, 1/m.
    path : float, default 0
        Excess optical path at the centre, metres.
    offset_mode : {'after', 'before'}, default 'after'
        Ordering of the Heisenberg-Weyl offset, as in :func:`~temgym_core.wave.lct.plan_lct`.

    Returns
    -------
    amp, centre, slopes, Q, path
        The Gaussian after the operator; ``path`` gains the Hamilton actions.
    """
    k = 2.0 * np.pi / wavelength
    A, B, C, D, a, b = split_matrix(matrix)
    amp = jnp.asarray(amp, dtype=jnp.complex128)
    centre = jnp.asarray(centre, dtype=jnp.float64)
    slopes = jnp.asarray(slopes, dtype=jnp.float64)
    Q = jnp.asarray(Q, dtype=jnp.complex128)
    path = jnp.asarray(path, dtype=jnp.float64)
    if offset_mode == "before":
        # U_M T(M^-1 t): shift and boost the input Gaussian first, then the offset-free operator
        M5 = jnp.asarray(matrix, dtype=jnp.float64)
        t_pre = jnp.linalg.inv(M5[:4, :4]) @ M5[:4, 4]
        a_pre, b_pre = t_pre[:2], t_pre[2:]
        centre_shifted = centre + a_pre
        amp = amp * jnp.exp(1j * k * (b_pre @ centre_shifted))
        centre, slopes = centre_shifted, slopes + b_pre
        a = jnp.zeros(2)
        b = jnp.zeros(2)
    detA = jnp.linalg.det(A)

    def canonical():
        Ainv = inv2(jnp, A)
        S = 0.5 * (Ainv @ B + (Ainv @ B).T)
        T = 0.5 * (C @ Ainv + (C @ Ainv).T)
        SQ = S @ Q                                        # Fr(S)
        Q1 = Q @ jnp.linalg.inv(jnp.eye(2) + SQ)
        Q1 = 0.5 * (Q1 + Q1.T)
        amp1 = amp / _sqrt_det_1_plus(SQ)
        rc1 = centre + S @ slopes
        path1 = path + 0.5 * slopes @ S @ slopes
        amp2 = amp1 * _reflect(A) / jnp.sqrt(jnp.abs(detA))   # Lin(A)
        rc2 = A @ rc1
        d2 = Ainv.T @ slopes
        Q2 = Ainv.T @ Q1 @ Ainv
        Q3 = Q2 + T                                       # L(T)
        d3 = d2 + T @ rc2
        path3 = path1 + 0.5 * rc2 @ T @ rc2
        return amp2, rc2, d3, Q3, path3

    def kernel():
        Bi = inv2(jnp, B)
        G = 0.5 * (Bi @ A + (Bi @ A).T)
        H = 0.5 * (D @ Bi + (D @ Bi).T)
        Z = Q + G
        Zi = jnp.linalg.inv(Z)
        beta = slopes - Q @ centre
        a1 = amp * jnp.exp(1j * k * (-slopes @ centre + 0.5 * centre @ Q @ centre + path))
        # principal sqrt is exact: Re(-i Z) = Im Q > 0
        c = lift_phase(A, B) / (jnp.sqrt(jnp.abs(jnp.linalg.det(B))) * jnp.sqrt(jnp.linalg.det(-1j * Z)))
        Q_out = H - Bi.T @ Zi @ Bi
        Q_out = 0.5 * (Q_out + Q_out.T)
        rc_out = A @ centre + B @ slopes
        d_out = C @ centre + D @ slopes
        # centred form of exp{ik[1/2 r'Q'r' + beta'.r' - 1/2 beta Z^-1 beta]}
        const = -0.5 * beta @ Zi @ beta - 0.5 * rc_out @ Q_out @ rc_out + d_out @ rc_out
        return a1 * c * jnp.exp(1j * k * const), rc_out, d_out, Q_out, jnp.zeros((), jnp.float64)

    amp_out, rc_out, d_out, Q_out, path_out = _switch(jnp.abs(detA) > 1e-6, canonical, kernel)
    rc_full = rc_out + a                                  # Heisenberg-Weyl offset: e^{ik b.r'} U(r' - a)
    amp_full = amp_out * jnp.exp(1j * k * (b @ rc_full))
    return amp_full, rc_full, d_out + b, Q_out, path_out


def chief_from_beam(beam):
    """Extract ``(amp, centre, slopes, Q, path)`` from a scalar ``GaussianBeam``.

    Parameters
    ----------
    beam : GaussianBeam
        A single beamlet.

    Returns
    -------
    tuple
        Amplitude, centre ``(x, y)`` [m], slopes [rad], ``Q_inv`` [1/m] and excess path [m].
    """
    return (jnp.asarray(beam.amplitude), jnp.asarray(beam.r_xy).reshape(2),
            jnp.asarray(beam.d_xy).reshape(2), jnp.asarray(beam.Q_inv).reshape(2, 2),
            jnp.asarray(beam.pathlength))


def _max_eigenvalue(M):
    """Largest eigenvalue of a real symmetric 2x2 matrix."""
    tr = M[0, 0] + M[1, 1]
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    return tr / 2 + jnp.sqrt(jnp.maximum(tr * tr / 4 - det, 0.0))


def _relative_mismatch(k, centre, slopes, Q, expected):
    """Dimensionless disagreement of a Gaussian state with the chief, in units of the chief's scales."""
    _, rc_c, d_c, Q_c, _ = expected
    covariance = jnp.linalg.inv(2 * k * jnp.imag(Q_c))
    angular = jnp.real(Q_c) @ covariance @ jnp.real(Q_c).T + jnp.linalg.inv(covariance) / (4 * k**2)
    position_scale = jnp.sqrt(_max_eigenvalue(covariance))
    slope_scale = jnp.sqrt(_max_eigenvalue(angular))
    return (jnp.max(jnp.abs(centre - rc_c)) / position_scale + jnp.max(jnp.abs(slopes - d_c)) / slope_scale
            + jnp.linalg.norm(Q - Q_c) / jnp.linalg.norm(Q_c))


def chief_phase0(matrix, wavelength, chief_in, chief_out):
    """Return the constant phase that turns the engine's lift into the physical field.

    ``chief_in`` and ``chief_out`` are Gaussians before and after the *same* span, the second
    propagated with the Gaussian engine, whose amplitude carries the continuous Gouy phase and
    whose ``pathlength`` is the excess path (use ``FreeSpacePropagator()``). The phase is the same
    for every field, because the operator of an affine ray map is fixed up to one constant. It is 0
    or pi for a column with no offsets (decentring, kicks) and the Hamilton action of the chief ray
    otherwise.

    Parameters
    ----------
    matrix : (5, 5) or (4, 4) array
        Ray map of the span.
    wavelength : float
        Wavelength, metres.
    chief_in, chief_out : GaussianBeam or tuple
        Scalar beamlets, or ``(amp, centre, slopes, Q, path)`` tuples.

    Returns
    -------
    phase0 : float
        Rad; pass it to :func:`~temgym_core.wave.lct.apply_lct`.
    mismatch : float
        Dimensionless disagreement between the closed-form action of the matrix and the chief's
        centre, slopes and curvature, in units of the chief's own position, angle and curvature
        scales. It is of the order of rounding (1e-12) when the matrix and the chief describe the
        same span, and large when a component's ray and Gaussian branches disagree.
    """
    k = 2.0 * np.pi / wavelength
    cin = chief_from_beam(chief_in) if hasattr(chief_in, "Q_inv") else chief_in
    cout = chief_from_beam(chief_out) if hasattr(chief_out, "Q_inv") else chief_out
    amp_e, rc_e, d_e, Q_e, path_e = collins_gaussian(matrix, wavelength, cin[0], cin[1], cin[2], cin[3], cin[4])
    ratio = cout[0] / amp_e * jnp.exp(1j * k * (cout[4] - path_e))
    mismatch = jax.lax.stop_gradient(jnp.abs(jnp.abs(ratio) - 1.0) + _relative_mismatch(k, rc_e, d_e, Q_e, cout))
    return jnp.angle(ratio), mismatch
