"""Physical projector parameters mapped to native finite-solenoid components."""

import jax.numpy as jnp
import numpy as np

from temgym_core.components import solenoid_column
from temgym_core.run import column_matrix


LENSES = ("Objective", "IL1", "IL2", "IL3", "PL1")


SIGNS = jnp.array([1.0, 1.0, 1.0, -1.0])

#: Objective, IL1, IL2, IL3 and PL1 strengths at the 40k row that :func:`initial_guess` was made
#: for (a fit maps a microscope's 40k-row DAC values onto these to start near its solution).
REFERENCE_40K_CONTROLS = np.array([1.0, 0.6156826056827933, 0.5675232215801136, 1.7562259961484548,
                                   0.8387606122572018])


def initial_guess():
    """Literal engineering guesses, independently specified from the generator."""
    return np.r_[
        np.log([0.0025, 0.19, 0.08, 0.05, 0.06, 0.34]),
        np.log([0.0005, 0.006, 0.004, 0.006, 0.0012]),
        -0.40,
        np.log([0.50, 0.60, 0.50, 0.70]),
    ]


def decode_geometry(p):
    gaps, lengths = jnp.exp(p[:6]), jnp.exp(p[6:11])
    entrances = jnp.cumsum(gaps[:-1]) + jnp.r_[0.0, jnp.cumsum(lengths[:-1])]
    return gaps, lengths, entrances + lengths / 2, jnp.sum(gaps) + jnp.sum(lengths)


def projector_components(p, controls):
    """Executable TemGym projector, from specimen to detector.

    Four controls mean IL1, IL2, IL3, PL1 with the objective held at one.
    Five controls mean Objective, IL1, IL2, IL3, PL1. The objective control
    multiplies its fitted reference strength, just as for the other lenses.
    """
    controls = jnp.asarray(controls)
    if controls.shape not in ((4,), (5,)):
        raise ValueError(
            "Supply four projector controls or five controls including the objective first."
        )
    gaps, lengths, _, _ = decode_geometry(p)
    objective = controls[0] if controls.shape == (5,) else 1.0
    q = jnp.r_[p[11] * objective, SIGNS * jnp.exp(p[12:16]) * controls[-4:]]
    return solenoid_column(gaps, lengths, q)


def matrix5(p, controls):
    """Grouped 5×5 ray map of the same components used for coherent imaging."""
    return column_matrix(projector_components(p, controls))


def at_voltage(parameters, voltage, reference=200000.0):
    from .high_tension import magnetic_strength_ratio

    ratio = magnetic_strength_ratio(voltage, reference)
    return jnp.asarray(parameters).at[11].multiply(ratio).at[12:16].add(jnp.log(ratio))


