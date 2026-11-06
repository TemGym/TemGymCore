"""Native aperture transmission, diffraction and per-hole centres."""

import jax.numpy as jnp

from .condenser_model import condenser_at_voltage


def aperture_centres(parameters, number_of_apertures):
    """Micrometres; first centre is p[20:22], additional centres follow source two."""
    p = jnp.asarray(parameters)
    if number_of_apertures == 1:
        return p[20:22][None, :]
    return jnp.concatenate(
        [p[20:22][None, :], p[29:].reshape(number_of_apertures - 1, 2)]
    )


def effective_parameters(parameters, voltage, reference, source_index):
    p = jnp.asarray(parameters)
    effective = condenser_at_voltage(
        p,
        voltage,
        reference_voltage_v=reference,
        source_parameters=p[22:29] if source_index else None,
    )
    return p.at[:22].set(effective)


