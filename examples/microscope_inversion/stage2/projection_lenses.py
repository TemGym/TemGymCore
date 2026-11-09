"""Stage 2's copy of the lens model (the stage-1 notebook has the version in use). The projection system as TemGym optics: the one place this experiment builds a TemGym column.

Five lenses after the specimen, each a TemGym ``HardEdgeSolenoid`` (a uniform magnetic field of
some length), then the camera::

    specimen ─gap─ [Objective] ─gap─ [IL1] ─gap─ [IL2] ─gap─ [IL3] ─gap─ [PL1] ─gap─ camera

They are described by 16 numbers ``p`` (stated at a reference HT, 200 kV):

=============  =====================  ==========================================================
``p[GAPS]``    6 numbers              log of the 6 gaps (m)
``p[LENGTHS]`` 5 numbers              log of the 5 lens lengths (m)
``p[OBJECTIVE]`` 1 number             the objective's strength q (rad); it is never driven
``p[GAINS]``   4 numbers              log of IL1, IL2, IL3, PL1's gain: strength q per unit control
=============  =====================  ==========================================================

A lens's strength ``q`` is the angle (rad) the beam turns through in it, proportional to its
field over the beam's momentum. For a setting, ``controls`` gives each lens's relative strength
(objective first, normally 1), and the lens's strength is ``sign * gain * control``.

Uses TemGym: :func:`column` builds the lenses (``temgym_core.components.HardEdgeSolenoid``) and
:func:`at_voltage` scales their strengths with HT (``temgym_core.constants.magnetic_strength_ratio``).
What the lenses do to a ray is then TemGym's job: the fit asks
``temgym_core.run.position_and_angle_response`` for A and B, the simulator traces rays through them.

Used by the fit (``fit.py``, with fitted ``p``) and by the virtual NEOARM (``virtual_neoarm.py``,
with the hidden ``p`` of ``examples/virtual_neoarm/virtual_neoarm.json``).
"""

import jax.numpy as jnp
import numpy as np

from temgym_core.components import HardEdgeSolenoid, Plane
from temgym_core.constants import magnetic_strength_ratio

LENSES = ("Objective", "IL1", "IL2", "IL3", "PL1")

#: Where each kind of number sits in ``p``.
GAPS, LENGTHS, OBJECTIVE, GAINS = slice(0, 6), slice(6, 11), 11, slice(12, 16)
N_PARAMETERS = 16

#: Field direction of IL1, IL2, IL3, PL1 (PL1 turns the beam the other way).
GAIN_SIGNS = jnp.array([1.0, 1.0, 1.0, -1.0])

#: Objective, IL1, IL2, IL3 and PL1 controls at the 40k row that :func:`starting_guess` was made
#: for. Only a choice of unit: the fit maps a microscope's 40k-row DAC values onto these.
REFERENCE_40K_CONTROLS = np.array([1.0, 0.6156826056827933, 0.5675232215801136, 1.7562259961484548,
                                   0.8387606122572018])


def starting_guess():
    """Engineering guesses for ``p``, chosen by hand (not from the hidden optics)."""
    return np.r_[
        np.log([0.0025, 0.19, 0.08, 0.05, 0.06, 0.34]),      # gaps (m)
        np.log([0.0005, 0.006, 0.004, 0.006, 0.0012]),       # lengths (m)
        -0.40,                                               # objective q
        np.log([0.50, 0.60, 0.50, 0.70]),                    # gains
    ]


def geometry(p):
    """Gaps, lengths, lens centres (from the specimen) and the specimen-to-camera length, in m."""
    gaps, lengths = jnp.exp(p[GAPS]), jnp.exp(p[LENGTHS])
    entrances = jnp.cumsum(gaps[:-1]) + jnp.r_[0.0, jnp.cumsum(lengths[:-1])]
    return gaps, lengths, entrances + lengths / 2, jnp.sum(gaps) + jnp.sum(lengths)


def column(p, controls):
    """The TemGym components from the specimen to the camera for one setting.

    ``controls`` are the five relative strengths, objective first. Returns five
    ``HardEdgeSolenoid`` and the camera ``Plane``.
    """
    controls = jnp.asarray(controls)
    gaps, lengths, _, _ = geometry(p)
    q = jnp.r_[p[OBJECTIVE] * controls[0], GAIN_SIGNS * jnp.exp(p[GAINS]) * controls[1:]]
    entrances = jnp.cumsum(gaps[:5]) + jnp.r_[0.0, jnp.cumsum(lengths[:4])]
    return [*(HardEdgeSolenoid(z=entrances[i], length=lengths[i], q=q[i]) for i in range(5)),
            Plane(z=gaps.sum() + lengths.sum())]


def at_voltage(p, voltage, reference=200000.0):
    """``p`` at HT ``voltage``, from ``p`` stated at ``reference``: the same fields turn a slower
    beam more, so every strength is multiplied by p(reference) / p(voltage)."""
    ratio = magnetic_strength_ratio(voltage, reference)
    return jnp.asarray(p).at[OBJECTIVE].multiply(ratio).at[GAINS].add(jnp.log(ratio))
