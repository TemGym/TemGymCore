import jax.numpy as jnp
from dataclasses import dataclass


@dataclass
class KrivanekCoeffs:
    C10: float = 0.0
    C12: float = 0.0
    phi12: float = 0.0
    C21: float = 0.0
    phi21: float = 0.0
    C23: float = 0.0
    phi23: float = 0.0
    C30: float = 0.0
    C32: float = 0.0
    phi32: float = 0.0
    C34: float = 0.0
    phi34: float = 0.0
    C41: float = 0.0
    phi41: float = 0.0
    C43: float = 0.0
    phi43: float = 0.0
    C45: float = 0.0
    phi45: float = 0.0
    C50: float = 0.0
    C52: float = 0.0
    phi52: float = 0.0
    C54: float = 0.0
    phi54: float = 0.0
    C56: float = 0.0
    phi56: float = 0.0


def _cos_kriv(m, ph, ph0):
    return jnp.cos(m * (ph - ph0))


def _sin_kriv(m, ph, ph0):
    return jnp.sin(m * (ph - ph0))


def krivanek_coeff_brackets(phi, p: KrivanekCoeffs):
    B2 = p.C10 + p.C12 * _cos_kriv(2, phi, p.phi12)
    B3 = p.C21 * _cos_kriv(1, phi, p.phi21) + p.C23 * _cos_kriv(3, phi, p.phi23)
    B4 = p.C30 + p.C32 * _cos_kriv(2, phi, p.phi32) + p.C34 * _cos_kriv(4, phi, p.phi34)
    B5 = p.C41 * _cos_kriv(1, phi, p.phi41) + p.C43 * _cos_kriv(3, phi, p.phi43) + p.C45 * _cos_kriv(5, phi, p.phi45)  # noqa: E501
    B6 = p.C50 + p.C52 * _cos_kriv(2, phi, p.phi52) + p.C54 * _cos_kriv(4, phi, p.phi54) + p.C56 * _cos_kriv(6, phi, p.phi56)  # noqa: E501
    return B2, B3, B4, B5, B6


def W_krivanek(alpha, phi, p):

    B2, B3, B4, B5, B6 = krivanek_coeff_brackets(phi, p)
    a = alpha
    a2 = a * a
    a3 = a2 * a
    a4 = a2 * a2
    a6 = a3 * a3

    return 0.5 * a2 * B2 + (a3 / 3.0) * B3 + 0.25 * a4 * B4 + 0.2 * a4 * a * B5 + (a6 / 6.0) * B6


def grad_W_krivanek(alpha_x, alpha_y, p):
    ax, ay = alpha_x, alpha_y
    alpha = jnp.hypot(ax, ay)
    phi = jnp.arctan2(ay, ax)

    B2, B3, B4, B5, B6 = krivanek_coeff_brackets(phi, p)

    a = alpha
    a2 = a * a
    a3 = a2 * a
    a4 = a2 * a2
    a5 = a4 * a
    a6 = a3 * a3

    dW_dalpha = a * B2 + a2 * B3 + a3 * B4 + a4 * B5 + a5 * B6

    dW_dphi = (0.5 * a2) * (
        -2.0 * p.C12 * _sin_kriv(2, phi, p.phi12)
    )
    dW_dphi += (a3 / 3.0) * (
        -1.0 * p.C21 * _sin_kriv(1, phi, p.phi21)
        - 3.0 * p.C23 * _sin_kriv(3, phi, p.phi23)
    )
    dW_dphi += (0.25 * a4) * (
        -2.0 * p.C32 * _sin_kriv(2, phi, p.phi32)
        - 4.0 * p.C34 * _sin_kriv(4, phi, p.phi34)
    )
    dW_dphi += (0.2 * a4 * a) * (
        -1.0 * p.C41 * _sin_kriv(1, phi, p.phi41)
        - 3.0 * p.C43 * _sin_kriv(3, phi, p.phi43)
        - 5.0 * p.C45 * _sin_kriv(5, phi, p.phi45)
    )
    dW_dphi += (a6 / 6.0) * (
        -2.0 * p.C52 * _sin_kriv(2, phi, p.phi52)
        - 4.0 * p.C54 * _sin_kriv(4, phi, p.phi54)
        - 6.0 * p.C56 * _sin_kriv(6, phi, p.phi56)
    )

    eps = 1e-30
    a_safe = jnp.where(a == 0, eps, a)
    inv_a = 1.0 / a_safe
    inv_a2 = inv_a * inv_a

    dWx = dW_dalpha * (ax * inv_a) + dW_dphi * (-ay * inv_a2)
    dWy = dW_dalpha * (ay * inv_a) + dW_dphi * (ax * inv_a2)
    return dWx, dWy
