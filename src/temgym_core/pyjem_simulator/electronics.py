"""Hidden electronics of the simulated microscope: DAC values to physical controls.

A JEOL column is set through 16-bit DAC values (0-65535): one per lens supply and
two per deflector pair. The real relation between a DAC value and the lens current or
the specimen kick is not known to the user. Here it is part of the hidden truth:

* :class:`LensDrive` maps a lens DAC value to the relative lens control used by the
  projector model (current divided by its reference current).
* :class:`DeflectorDrive` maps the CLA1 (shift) and CLA2 (tilt) DAC values to the
  calibrated specimen displacement and angle, with the magnetic 1/p scaling of
  a fixed deflector field when the high tension changes.
* :class:`LensHysteresis` makes a lens's field lag behind its DAC value, depending on
  the values the lens had before (its iron pole pieces).
"""

from dataclasses import dataclass, field

import numpy as np

from ..constants import magnetic_strength_ratio

DAC_MIN, DAC_MAX = 0, 65535
NEUTRAL_DAC = 32768


class DACRangeError(ValueError):
    """A requested DAC value lies outside the 16-bit range 0-65535."""


def check_dac(value, name="DAC value"):
    """Return ``value`` as an int after checking it is an integer DAC value in 0-65535.

    Parameters
    ----------
    value : int
        Requested DAC value.
    name : str, optional
        Name used in the error message.

    Returns
    -------
    int
        The validated DAC value.

    Raises
    ------
    DACRangeError
        If ``value`` is not integral or outside 0-65535.
    """
    if isinstance(value, (bool, np.bool_)) or int(value) != value:
        raise DACRangeError(f"{name} must be an integer DAC value, not {value!r}.")
    value = int(value)
    if not DAC_MIN <= value <= DAC_MAX:
        raise DACRangeError(f"{name}={value} is outside the 16-bit range {DAC_MIN}-{DAC_MAX}.")
    return value


@dataclass(frozen=True)
class LensDrive:
    """Linear lens supply: ``control = (DAC value - dac_offset) / dac_per_unit``.

    Parameters
    ----------
    dac_per_unit : float
        DAC values per unit of relative lens control (positive).
    dac_offset : float, optional
        DAC value at which the relative control is zero.
    """

    dac_per_unit: float
    dac_offset: float = 0.0

    def __post_init__(self):
        if not np.isfinite(self.dac_per_unit) or self.dac_per_unit <= 0:
            raise ValueError("dac_per_unit must be finite and positive.")
        if not np.isfinite(self.dac_offset):
            raise ValueError("dac_offset must be finite.")

    def control(self, dac):
        """Relative lens control for an integer DAC value.

        Parameters
        ----------
        dac : int
            Lens DAC value.

        Returns
        -------
        float
            Current divided by the reference current.
        """
        return (check_dac(dac) - self.dac_offset) / self.dac_per_unit

    def control_at(self, effective_dac):
        """Relative lens control for an effective DAC value, which need not be whole (:class:`LensHysteresis`)."""
        return (float(effective_dac) - self.dac_offset) / self.dac_per_unit

    def dac(self, control):
        """Nearest integer DAC value for a relative control.

        Parameters
        ----------
        control : float
            Current divided by the reference current.

        Returns
        -------
        int
            The rounded DAC value.

        Raises
        ------
        DACRangeError
            If the control needs a DAC value outside 0-65535.
        """
        return check_dac(int(np.rint(control * self.dac_per_unit + self.dac_offset)))


@dataclass(frozen=True)
class LensHysteresis:
    """A lens's field lagging behind its DAC value, by how the value was reached.

    The iron of a lens remembers: after the DAC value rises to ``d`` the field is that of a slightly lower value,
    after it falls, of a slightly higher one, and a large swing wipes out the memory of smaller ones. This is the
    sum of a few backlash elements (a Prandtl-Ishlinskii model): element ``i`` follows the DAC value but stays up to
    ``widths_dac[i]`` behind it, and the lens acts as if its DAC value were

        effective = DAC value + sum_i fractions[i] * (element_i - DAC value),

    so the lag is at most ``sum_i fractions[i] * widths_dac[i]`` DAC units. It depends on the order of the values,
    not on time. With no elements the lens follows its DAC value exactly.

    Parameters
    ----------
    widths_dac : tuple of float
        How far behind (DAC units) each element can stay.
    fractions : tuple of float
        How much of each element's lag the field shows.
    """

    widths_dac: tuple = ()
    fractions: tuple = ()

    def __post_init__(self):
        widths, fractions = tuple(map(float, self.widths_dac)), tuple(map(float, self.fractions))
        if len(widths) != len(fractions) or any(w < 0 for w in widths):
            raise ValueError("Give one fraction per width, and widths that are not negative.")
        object.__setattr__(self, "widths_dac", widths)
        object.__setattr__(self, "fractions", fractions)

    def at_rest(self, dac):
        """The elements of a lens that has no lag at this DAC value."""
        return (float(dac),) * len(self.widths_dac)

    def move(self, elements, dac):
        """The elements after the DAC value goes (in one direction) to ``dac``."""
        return tuple(min(max(e, dac - w), dac + w) for e, w in zip(elements, self.widths_dac))

    def effective_dac(self, elements, dac):
        """The DAC value the lens acts as if it had."""
        return float(dac) + sum(f * (e - dac) for f, e in zip(self.fractions, elements))

    def effective_sequence(self, dac_values, start=None):
        """The effective DAC value after each of a sequence of DAC values (the lens at rest at ``start``, by default
        at the first value): what a fit applies to a recording's rows.

        Parameters
        ----------
        dac_values : array_like
            The DAC values the lens took, in order.
        start : float, optional
            The DAC value the lens was at rest at before the sequence.

        Returns
        -------
        numpy.ndarray
        """
        values = np.asarray(dac_values, float)
        elements = self.at_rest(values[0] if start is None else start)
        out = np.empty(len(values))
        for i, dac in enumerate(values):
            elements = self.move(elements, dac)
            out[i] = self.effective_dac(elements, dac)
        return out


@dataclass(frozen=True)
class DeflectorDrive:
    """Linear CLA deflectors: DAC values to specimen kick ``[x, y, theta_x, theta_y]``.

    Parameters
    ----------
    kick_per_dac : array_like
        4x4 matrix from DAC value offsets ``[CLA1 x, CLA1 y, CLA2 x, CLA2 y]`` to the
        specimen displacement (m) and angle (rad) at ``reference_voltage_v``.
    zero_dac : tuple of int, optional
        DAC values giving no kick.
    reference_voltage_v : float, optional
        High tension at which ``kick_per_dac`` holds. A fixed deflector field
        deflects by ``p(reference) / p(voltage)`` times more at other voltages.
    """

    kick_per_dac: np.ndarray
    zero_dac: tuple = (NEUTRAL_DAC,) * 4
    reference_voltage_v: float = 200e3

    def __post_init__(self):
        matrix = np.array(self.kick_per_dac, dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) == 0:
            raise ValueError("kick_per_dac must be a finite, invertible 4x4 matrix.")
        object.__setattr__(self, "kick_per_dac", matrix)
        object.__setattr__(self, "zero_dac", tuple(check_dac(w) for w in self.zero_dac))

    def kick(self, dac_values, voltage_v):
        """Specimen kick produced by CLA DAC values at a high tension.

        Parameters
        ----------
        dac_values : array_like
            ``[CLA1 x, CLA1 y, CLA2 x, CLA2 y]`` integer DAC values.
        voltage_v : float
            High tension.

        Returns
        -------
        numpy.ndarray
            ``[x, y, theta_x, theta_y]`` in metres and radians.
        """
        offsets = np.array([check_dac(w) for w in dac_values], dtype=float) - np.array(self.zero_dac)
        ratio = float(magnetic_strength_ratio(voltage_v, self.reference_voltage_v))
        return ratio * self.kick_per_dac @ offsets

    def dac_values(self, kick, voltage_v):
        """Nearest integer CLA DAC values for a requested kick.

        Parameters
        ----------
        kick : array_like
            ``[x, y, theta_x, theta_y]`` in metres and radians.
        voltage_v : float
            High tension.

        Returns
        -------
        tuple of int
            ``(CLA1 x, CLA1 y, CLA2 x, CLA2 y)``.
        """
        ratio = float(magnetic_strength_ratio(voltage_v, self.reference_voltage_v))
        offsets = np.linalg.solve(self.kick_per_dac * ratio, np.asarray(kick, dtype=float))
        return tuple(check_dac(int(np.rint(o)) + z) for o, z in zip(offsets, self.zero_dac))


@dataclass(frozen=True)
class Electronics:
    """All hidden DAC-to-physics laws of the virtual column.

    Parameters
    ----------
    lenses : dict
        :class:`LensDrive` per modelled lens: ``IL1``, ``IL2``, ``IL3``, ``PL1``.
    deflector : DeflectorDrive
        Beam shift (CLA1) and tilt (CLA2).
    objective_dac : tuple of int, optional
        ``(OLc, OLf)``: the objective's DAC values. Other objective DAC values are not modelled.
    objective_control : float, optional
        The objective's relative control at those DAC values (1 at the reference HT's alignment;
        p(HT) / p(reference) for a console aligned at another HT).
    hysteresis : dict, optional
        :class:`LensHysteresis` per lens name (projection or condenser lenses); a lens not in it follows its DAC
        value exactly.
    """

    lenses: dict
    deflector: DeflectorDrive
    objective_dac: tuple = (45818, 39178)
    objective_control: float = 1.0
    hysteresis: dict = field(default_factory=dict)
    modelled: tuple = field(default=("IL1", "IL2", "IL3", "PL1"), init=False)

    def __post_init__(self):
        missing = set(self.modelled) - set(self.lenses)
        if missing:
            raise ValueError(f"Supply a LensDrive for {sorted(missing)}.")
        object.__setattr__(self, "lenses", dict(self.lenses))
        object.__setattr__(self, "hysteresis", dict(self.hysteresis))
        object.__setattr__(self, "objective_dac", tuple(check_dac(w) for w in self.objective_dac))
