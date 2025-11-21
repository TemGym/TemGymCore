import numpy as np
import jax.numpy as jnp
from scipy.special import j1
from temgym_core.potential import glauber_amplitude_radial


def f_black_disk_analytic(theta, k, a):
    """
    Analytic Glauber amplitude for an opaque disk of radius a:

        f(θ) = i a J1( 2 k a sin(θ/2) ) / (2 sin(θ/2))
    """
    theta = np.asarray(theta)
    s = np.sin(theta / 2.0)
    qa = 2.0 * k * a * s
    return 1j * a * j1(qa) / (2.0 * s)


def test_black_disk_glauber_converges():
    # Choose some convenient k and a (units arbitrary but consistent)
    k = 50.0   # ~Å^{-1} scale
    a = 1.0    # disk radius in Å

    # Avoid θ = 0 exactly (analytic formula has 0/0 there)
    thetas = np.linspace(0.0001, 0.4, 200)  # radians
    f_exact = f_black_disk_analytic(thetas, k, a)

    # Corresponding q values: q = 2 k sin(θ/2)
    q_vals = 2.0 * k * np.sin(thetas / 2.0)

    b_grid = jnp.linspace(0.0, a, 100_000_0)
    Gamma = jnp.where(b_grid < a, 1.0, 0.0)  # Γ(b)=1 inside disk

    f_numeric = []
    for q in q_vals:
        f_numeric.append(glauber_amplitude_radial(
            q,
            k,
            b_grid,
            Gamma,
        ))
    f_numeric = jnp.array(f_numeric)

    np.testing.assert_allclose(abs(f_numeric) ** 2, abs(f_exact) ** 2, rtol=0.01, atol=1e-2)
