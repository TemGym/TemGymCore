"""How lens strengths scale with the high tension (HT)."""

import jax.numpy as jnp
from scipy.constants import c, electron_mass, elementary_charge


def magnetic_strength_ratio(voltage_v, reference_voltage_v=200000.0):
    """p(reference)/p(voltage); fixed magnetic fields have q proportional to 1/p.

    Relativistic kinetic energy gives (pc)^2 = eV(eV + 2mc^2).
    Voltages are positive kinetic energies per electron charge, in volts.
    """
    rest_voltage = electron_mass * c**2 / elementary_charge
    voltage = jnp.asarray(voltage_v)
    reference = jnp.asarray(reference_voltage_v)
    return jnp.sqrt(
        reference
        * (reference + 2 * rest_voltage)
        / (voltage * (voltage + 2 * rest_voltage))
    )
