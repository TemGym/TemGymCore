from ase import units
import jax.numpy as jnp


def wavelength2energy(wavelength: float) -> float:
    """
    Calculate energy from relativistic de Broglie wavelength.

    Parameters
    ----------
    wavelength: float
        Relativistic de Broglie wavelength [Å].

    Returns
    -------
    float
        Energy [eV].
    """
    # Constants
    h = units._hplanck
    c = units._c
    m_e = units._me
    e = units._e

    term1 = (h * c / wavelength) ** 2
    term2 = (m_e * c**2) ** 2

    energy_joules = jnp.sqrt(term1 + term2) - m_e * c**2

    return energy_joules / e


def energy2wavelength(energy: float) -> float:
    """
    Calculate relativistic de Broglie wavelength from energy with higher precision options.

    Parameters
    ----------
    energy : float
        Kinetic energy [eV].
    """
    h = units._hplanck
    c = units._c
    m_e = units._me
    e = units._e

    E_j = energy * e
    mc2 = m_e * c * c
    E_total = E_j + mc2
    rad = E_total * E_total - mc2 * mc2
    p = jnp.sqrt(rad) / c
    wavelength_m = h / p
    return wavelength_m


def interaction_constant_CE(energy: float) -> float:
    """
    Electrostatic phase interaction constant C_E [rad / (V·m)]
    as used in electron holography.

    Parameters
    ----------
    energy : float
        Kinetic energy [eV].
    """

    m_e = units._me
    c = units._c
    e = units._e
    hbar = units._hbar

    gamma = 1.0 + energy / (m_e * c**2 / e)
    lam = energy2wavelength(energy)

    return gamma * m_e * e * lam / (2 * jnp.pi * hbar**2)


def e_over_hbar() -> float:
    """
    Return e/ħ in SI units: rad / (T·m)
    Typically used as the magnetic prefactor in the A·dl integral.
    """
    e = units._e
    hbar = units._hbar
    return e / hbar


def relativistic_mass_correction(energy: float) -> float:
    e = units._e
    m_e = units._me
    c = units._c
    return 1 + e * energy / (m_e * c**2)


def energy2sigma(energy: float) -> float:
    lam = energy2wavelength(energy)
    mass = relativistic_mass_correction(energy) * units._me
    return (
        2 * jnp.pi * mass * units.kg * units._e * units.C * lam
        / (units._hplanck * units.s * units.J) ** 2
    )
