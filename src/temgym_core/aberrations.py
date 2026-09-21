import math

import jax.numpy as jnp
import jax_dataclasses as jdc


@jdc.pytree_dataclass(frozen=True)
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


def _krivanek_cartesian_terms(alpha_x, alpha_y, p):
    """Wave aberration and its gradient, using only real polynomials.

    alpha**n cos(m(phi-phi0)) is
    (x*x+y*y)**((n-m)//2) Re[(x+iy)**m exp(-im phi0)].
    The recurrence avoids polar coordinates and is smooth on axis at every
    derivative order. Inputs may be scalars or broadcast-compatible arrays.
    """
    x, y = alpha_x, alpha_y
    radius2 = x*x + y*y
    real = [jnp.ones_like(radius2)]
    imag = [jnp.zeros_like(radius2)]
    for m in range(1, 7):
        previous_real, previous_imag = real[-1], imag[-1]
        real.append(x*previous_real - y*previous_imag)
        imag.append(y*previous_real + x*previous_imag)
    value = jnp.zeros_like(radius2)
    gx, gy = jnp.zeros_like(radius2), jnp.zeros_like(radius2)
    for n in range(2, 7):
        for m in range(n % 2, n + 1, 2):
            coefficient = getattr(p, f"C{n-1}{m}") / n
            angle = 0.0 if m == 0 else m * getattr(p, f"phi{n-1}{m}")
            cosine, sine = jnp.cos(angle), jnp.sin(angle)
            harmonic = real[m]*cosine + imag[m]*sine
            power = (n-m)//2
            radial = radius2**power
            value = value + coefficient * radial * harmonic
            if power:
                radial_derivative = 2*power * radius2**(power-1) * harmonic
                gx = gx + coefficient * x * radial_derivative
                gy = gy + coefficient * y * radial_derivative
            if m:
                gx = gx + coefficient * radial * m * (
                    real[m-1]*cosine + imag[m-1]*sine
                )
                gy = gy + coefficient * radial * m * (
                    -imag[m-1]*cosine + real[m-1]*sine
                )
    return value, gx, gy


def W_krivanek_cartesian(alpha_x, alpha_y, p: KrivanekCoeffs):
    """Krivanek wave aberration with a nonsingular Cartesian representation."""
    return _krivanek_cartesian_terms(alpha_x, alpha_y, p)[0]


def grad_W_krivanek_cartesian(alpha_x, alpha_y, p: KrivanekCoeffs):
    """Cartesian wave-aberration gradient, including at the optical axis."""
    _, gx, gy = _krivanek_cartesian_terms(alpha_x, alpha_y, p)
    return gx, gy


@jdc.pytree_dataclass(frozen=True)
class SeidelCoeffs:
    A: float = 0.0  # Pure Phase Shift
    B: float = 0.0  # Spherical Aberration
    C: float = 0.0  # Isotropic Astigmatism
    D: float = 0.0  # Field Curvature
    E: float = 0.0  # Isotropic Distortion
    F: float = 0.0  # Isotropic Coma
    e: float = 0.0  # Anisotropic Coma
    f: float = 0.0  # Anisotropic Astigmatism
    c: float = 0.0  # Anisotropic Distortion


# Aberration functions parametrized by different variables
def Seidel_object_pos_aperture_pos(xo, yo, xa, ya, coeffs: SeidelCoeffs):
    A, B, C, D, E, F, e, f, c = coeffs.A, coeffs.B, coeffs.C, coeffs.D, coeffs.E, coeffs.F, coeffs.e, coeffs.f, coeffs.c
    R = xo**2 + yo**2
    rho = xa**2 + ya**2
    chi = xo*xa + yo*ya
    sigma = xo*ya - yo*xa
    return A*R**2 + B*rho**2 + C*chi**2 + D*R*rho + E*R*chi + F*rho*chi + e*R*sigma + f*rho*sigma + c*chi*sigma


def Seidel_object_pos_object_slope(xo, yo, xop, yop, z1, coeffs: SeidelCoeffs):
    A, B, C, D, E, F, e, f, c = coeffs.A, coeffs.B, coeffs.C, coeffs.D, coeffs.E, coeffs.F, coeffs.e, coeffs.f, coeffs.c
    R = xo**2 + yo**2
    rho = (xo + xop*z1)**2 + (yo + yop*z1)**2
    chi = xo*(xo + xop*z1) + yo*(yo + yop*z1)
    sigma = xo*(yo + yop*z1) - yo*(xo + xop*z1)
    return A*R**2 + B*rho**2 + C*chi**2 + D*R*rho + E*R*chi + F*rho*chi + e*R*sigma + f*rho*sigma + c*chi*sigma


def Seidel_aperture_pos_aperture_slope(xa, ya, xap, yap, z1, coeffs: SeidelCoeffs):
    A, B, C, D, E, F, e, f, c = coeffs.A, coeffs.B, coeffs.C, coeffs.D, coeffs.E, coeffs.F, coeffs.e, coeffs.f, coeffs.c
    R = (xa - xap*z1)**2 + (ya - yap*z1)**2
    rho = xa**2 + ya**2
    chi = (xa - xap*z1)*xa + (ya - yap*z1)*ya
    sigma = (xa - xap*z1)*ya - (ya - yap*z1)*xa
    return A*R**2 + B*rho**2 + C*chi**2 + D*R*rho + E*R*chi + F*rho*chi + e*R*sigma + f*rho*sigma + c*chi*sigma


# Lower-case aliases follow normal Python naming while retaining the original
# public functions used by existing notebooks.
def seidel_object_pos_aperture_pos(xo, yo, xa, ya, coeffs: SeidelCoeffs):
    return Seidel_object_pos_aperture_pos(xo, yo, xa, ya, coeffs)


def seidel_object_pos_object_slope(xo, yo, xop, yop, z1, coeffs: SeidelCoeffs):
    return Seidel_object_pos_object_slope(xo, yo, xop, yop, z1, coeffs)


def seidel_aperture_pos_aperture_slope(xa, ya, xap, yap, z1, coeffs: SeidelCoeffs):
    return Seidel_aperture_pos_aperture_slope(xa, ya, xap, yap, z1, coeffs)


_NOLL_TABLE: dict[int, tuple[int, int]] = {
    1: (0, 0),
    2: (1, 1),
    3: (1, -1),
    4: (2, 0),
    5: (2, -2),
    6: (2, 2),
    7: (3, -1),
    8: (3, 1),
    9: (3, -3),
    10: (3, 3),
    11: (4, 0),
    12: (4, 2),
    13: (4, -2),
    14: (4, 4),
    15: (4, -4),
    16: (5, 1),
    17: (5, -1),
    18: (5, 3),
    19: (5, -3),
    20: (5, 5),
    21: (5, -5),
    22: (6, 0),
    23: (6, -2),
    24: (6, 2),
    25: (6, -4),
    26: (6, 4),
    27: (6, -6),
    28: (6, 6),
}


def noll_to_nm(j: int) -> tuple[int, int]:
    """Return radial and azimuthal orders for the first 28 Noll indices."""
    j = int(j)
    if j not in _NOLL_TABLE:
        raise ValueError("Noll indices above 28 are not included.")
    return _NOLL_TABLE[j]


def _zernike_radial(n: int, m_abs: int, rho):
    out = jnp.zeros_like(rho, dtype=jnp.float64)
    for k in range((n - m_abs) // 2 + 1):
        coeff = (
            (-1) ** k
            * math.factorial(n - k)
            / (
                math.factorial(k)
                * math.factorial((n + m_abs) // 2 - k)
                * math.factorial((n - m_abs) // 2 - k)
            )
        )
        out = out + coeff * rho ** (n - 2 * k)
    return out


def zernike_nm(n: int, m: int, rho, phi, *, outside: float = 0.0):
    """Evaluate an unnormalised Zernike mode in polar coordinates."""
    n = int(n)
    m = int(m)
    m_abs = abs(m)
    if n < 0 or m_abs > n or (n - m_abs) % 2:
        raise ValueError(f"Invalid Zernike order (n={n}, m={m}).")
    radial = _zernike_radial(n, m_abs, rho)
    if m == 0:
        value = radial
    elif m > 0:
        value = radial * jnp.cos(m_abs * phi)
    else:
        value = radial * jnp.sin(m_abs * phi)
    return jnp.where(rho <= 1.0, value, outside)


def zernike_noll(j: int, rho, phi, *, outside: float = 0.0):
    n, m = noll_to_nm(j)
    return zernike_nm(n, m, rho, phi, outside=outside)


def zernike_noll_cartesian(j: int, x, y, *, outside: float = 0.0):
    """Evaluate a Noll Zernike polynomial without a polar-axis singularity."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    n, m = noll_to_nm(j)
    m_abs = abs(m)
    rho2 = x * x + y * y

    radial_factor = jnp.zeros_like(rho2, dtype=jnp.float64)
    for k in range((n - m_abs) // 2 + 1):
        coeff = (
            (-1) ** k
            * math.factorial(n - k)
            / (
                math.factorial(k)
                * math.factorial((n + m_abs) // 2 - k)
                * math.factorial((n - m_abs) // 2 - k)
            )
        )
        power = (n - 2 * k - m_abs) // 2
        radial_factor = radial_factor + coeff * rho2**power

    if m_abs == 0:
        angular_polynomial = jnp.ones_like(rho2)
    else:
        z_power = (
            x.astype(jnp.complex128) + 1j * y.astype(jnp.complex128)
        ) ** m_abs
        angular_polynomial = jnp.real(z_power) if m > 0 else jnp.imag(z_power)

    value = radial_factor * angular_polynomial
    return jnp.where(rho2 <= 1.0, value, outside)
