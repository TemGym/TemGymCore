import jax.numpy as jnp
import jax
from scipy.special import j0
from jax.scipy.integrate import trapezoid
from scipy.integrate import simpson
from scipy.special import kn

# Si parameters as JAX array
Si = jnp.array([
    [2.87189142611612, -2.06173501195173,  2.17114024204478,
     -0.0663073633058801, 0.00301070709670513],
    [5.08487103642989,  0.429178185305126, 0.366485434192162,
    0.119710611296903, 0.0143994536128397],
])


def potential(r, p):
    return (
        p[0, 0] * (2.0 / (p[1, 0] * r) + 1.0) * jnp.exp(-p[1, 0] * r) +
        p[0, 1] * (2.0 / (p[1, 1] * r) + 1.0) * jnp.exp(-p[1, 1] * r) +
        p[0, 2] * (2.0 / (p[1, 2] * r) + 1.0) * jnp.exp(-p[1, 2] * r) +
        p[0, 3] * (2.0 / (p[1, 3] * r) + 1.0) * jnp.exp(-p[1, 3] * r) +
        p[0, 4] * (2.0 / (p[1, 4] * r) + 1.0) * jnp.exp(-p[1, 4] * r)
)


def projected_potential(r, p):
    v = 2 * (
        2 * p[0][:, None] / p[1][:, None] * kn(0, r[None] * p[1][:, None])
        + p[0][:, None] * r[None] * kn(1, r[None] * p[1][:, None])
    ).sum(0)
    return v.astype(jnp.float32)


def scattering_factor(k2, p):
    return (
        (p[0, 0] * (2.0 + p[1, 0] * k2) / (1.0 + p[1, 0] * k2) ** 2)
        + (p[0, 1] * (2.0 + p[1, 1] * k2) / (1.0 + p[1, 1] * k2) ** 2)
        + (p[0, 2] * (2.0 + p[1, 2] * k2) / (1.0 + p[1, 2] * k2) ** 2)
        + (p[0, 3] * (2.0 + p[1, 3] * k2) / (1.0 + p[1, 3] * k2) ** 2)
        + (p[0, 4] * (2.0 + p[1, 4] * k2) / (1.0 + p[1, 4] * k2) ** 2)
    )


def projected_scattering_factor(k2, p):
    pi = jnp.array(jnp.pi)
    pi2 = jnp.array(jnp.pi**2)
    k2 = 4 * pi2 * k2
    f = (
        8
        * pi
        * (
            (
                p[0, 0] / p[1, 0] / (k2 + p[1, 0] ** 2)
                + p[0, 0] * p[1, 0] / (k2 + p[1, 0] ** 2) ** 2
            )
            + (
                p[0, 1] / p[1, 1] / (k2 + p[1, 1] ** 2)
                + p[0, 1] * p[1, 1] / (k2 + p[1, 1] ** 2) ** 2
            )
            + (
                p[0, 2] / p[1, 2] / (k2 + p[1, 2] ** 2)
                + p[0, 2] * p[1, 2] / (k2 + p[1, 2] ** 2) ** 2
            )
            + (
                p[0, 3] / p[1, 3] / (k2 + p[1, 3] ** 2)
                + p[0, 3] * p[1, 3] / (k2 + p[1, 3] ** 2) ** 2
            )
            + (
                p[0, 4] / p[1, 4] / (k2 + p[1, 4] ** 2)
                + p[0, 4] * p[1, 4] / (k2 + p[1, 4] ** 2) ** 2
            )
        )
    )
    return f


# Derivative of potential with respect to r
dV_dr = lambda r, p: jax.grad(lambda rr: potential(rr, p))(r)


def spline_core(r, p, r_pix, V_max):
    rr = jnp.abs(r)

    r1 = 2.0 * r_pix

    V1 = potential(r1, p)
    d1 = dV_dr(r1, p)

    DeltaV = V1 - V_max
    a = 2.0 * DeltaV - 0.5 * d1 * r1
    b = 0.5 * d1 * r1 - DeltaV

    y = (rr / r1)**2
    V_core = V_max + a * y + b * y**2
    return V_core


def potential_smoothed(r, p, r_pix):
    # Smoothed atomic potential with spline core within 2 pixel radii
    # This smoothed core avoids the tophat function that would occur with a hard cutoff
    # which multislice introduces by pixelating the potential.
    # The spline core ensures a smooth first and second derivative within the core region,
    # so the gaussian beam propagation can correctly account for the potential gradients.

    # r: radial distance from atom center [m]
    # p: atomic potential parameters
    # r_pix: pixel size [m]

    rr = jnp.abs(r)

    r1 = 2.0 * r_pix
    V_max = potential(r_pix, p)
    V_vals = potential(rr, p)
    V_core_vals = spline_core(rr, p, r_pix, V_max)

    V_eff = jnp.where(rr <= r1, V_core_vals, V_vals)

    return V_eff


def chi_of_b(b, p, sigma):
    V = projected_potential(b, p)
    chi = sigma * V
    return chi


chi_of_b_vmap = jax.vmap(chi_of_b, in_axes=(0, None, None, None, None))


def glauber_amplitude_radial(q, k, b_grid, Gamma_b):
    """
    Core radial Hankel integral used in all cases:

        f(q) = i k ∫_0^∞ db  b J0(q b) Γ(b)

    q:       scalar |q|
    k:       incident wavenumber
    b_grid:  1D jnp array of b values (0 ... b_max)
    Gamma_b: 1D jnp array Γ(b) = 1 - exp(i χ(b)) on the same grid
    """
    # SciPy Bessel J0 evaluated on numpy view of b_grid*q
    J0_eval_np = j0((q * b_grid).astype(float))
    J0_eval = jnp.asarray(J0_eval_np)

    integrand = b_grid * J0_eval * Gamma_b

    db = b_grid[1] - b_grid[0]
    f_q = 1j * k * simpson(integrand, dx=db)
    return f_q


def glauber_scattering_amplitude(q,
                                 k,
                                 p,
                                 b_max,
                                 nb,
                                 r_cutoff,
                                 sigma):
    """
    q: scalar |q| - scattering angle
    k: incident wavenumber
    p: Potential parameters
    r_cutoff: cutoff region to smooth centre of atom potential with spline
    b_max: max impact parameter
    nb: number of b points
    z_max: max z for line integral
    nz: number of z points
    sigma: interaction constant
    """
    # b and z grids
    b_grid = jnp.linspace(r_cutoff, b_max, nb)

    # eikonal phase for each b
    chi_b = chi_of_b(b_grid, p, sigma)

    Gamma_b = 1.0 - jnp.exp(1j * chi_b)

    # reuse the common radial integral
    return glauber_amplitude_radial(q, k, b_grid, Gamma_b)


def glauber_scattering_amplitude_many_q(q_array,
                                        k,
                                        p,
                                        b_max,
                                        nb,
                                        r_cutoff,
                                        sigma):
    """
    Convenience wrapper for an array of q values (no vmap, just a loop).
    """
    f_list = []
    for q in q_array:
        f_list.append(glauber_scattering_amplitude(float(q), k, p, b_max, nb, r_cutoff, sigma))
    return jnp.stack(f_list)
