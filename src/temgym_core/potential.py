import jax.numpy as jnp
import jax
from jax.scipy.special import bessel_jn


def J0(x):
    # returns J_0(x)
    return bessel_jn(x, v=0)[0]

# Si parameters as JAX array
Si = jnp.array([
    [ 2.87189142611612, -2.06173501195173,  2.17114024204478,
     -0.0663073633058801, 0.00301070709670513],
    [ 5.08487103642989,  0.429178185305126, 0.366485434192162,
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


def chi_of_b(b, p, r_pix, z_grid, sigma):
    """
    b: scalar impact parameter
    z_grid: 1D array of z points (symmetric around 0)
    sigma: interaction constant (TEM σ)
    """
    # r(z) along straight line at impact parameter b
    r = jnp.sqrt(b**2 + z_grid**2)

    # V(r) along the trajectory
    V_line = potential_smoothed(r, p, r_pix)

    # z-integration (simple trapezoidal rule)
    dz = z_grid[1] - z_grid[0]
    # manual trapezoid to be 100% JAX-core friendly
    trap = dz * (0.5 * (V_line[0] + V_line[-1]) + jnp.sum(V_line[1:-1]))

    chi = sigma * trap
    return chi


chi_of_b_vmap = jax.vmap(chi_of_b, in_axes=(0, None, None, None, None))


def glauber_amplitude(q, k, p, r_pix,
                      b_max, nb,
                      z_max, nz,
                      sigma):
    """
    q: scalar |q|
    k: incident wavenumber
    p: Si parameters
    r_pix: pixel size
    b_max: max impact parameter
    nb: number of b points
    z_max: max z for line integral
    nz: number of z points
    sigma: interaction constant
    """

    # b and z grids
    b_grid = jnp.linspace(1e-5, b_max, nb)
    z_grid = jnp.linspace(-z_max, z_max, nz)

    # eikonal phase for each b
    chi_b = chi_of_b_vmap(b_grid, p, r_pix, z_grid, sigma)

    # profile function Γ(b) = 1 - exp(i χ(b))
    Gamma_b = 1.0 - jnp.exp(1j * chi_b)

    J0_eval = J0(q * b_grid)


    # radial integral: f(q) = i k ∫_0^∞ db b J0(qb) Γ(b)
    integrand = b_grid * J0_eval * Gamma_b
    db = b_grid[1] - b_grid[0]
    trap = db * (0.5 * (integrand[0] + integrand[-1]) + jnp.sum(integrand[1:-1]))

    f_q = 1j * k * trap
    return f_q


# Vectorize over multiple q if needed:
glauber_amplitude_vmap = jax.vmap(
    glauber_amplitude,
    in_axes=(0, None, None, None, None, None, None, None, None),
)
