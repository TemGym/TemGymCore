"""Hidden electronics of the simulated microscope: DAC values to physical controls.

A JEOL column is set through 16-bit DAC values (0-65535): one per lens supply and
two per deflector pair. The real relation between a DAC value and the lens current or
the specimen kick is not known to the user. Here it is part of the hidden truth:

* :class:`LensDrive` maps a lens DAC value to the relative lens control used by the
  projector model (current divided by its reference current).
* :class:`DeflectorDrive` maps the CLA1 (shift) and CLA2 (tilt) DAC values to the
  calibrated specimen displacement and angle, with the magnetic 1/p scaling of
  a fixed deflector field when the high tension changes.
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
    """

    lenses: dict
    deflector: DeflectorDrive
    objective_dac: tuple = (45818, 39178)
    objective_control: float = 1.0
    modelled: tuple = field(default=("IL1", "IL2", "IL3", "PL1"), init=False)

    def __post_init__(self):
        missing = set(self.modelled) - set(self.lenses)
        if missing:
            raise ValueError(f"Supply a LensDrive for {sorted(missing)}.")
        object.__setattr__(self, "lenses", dict(self.lenses))
        object.__setattr__(self, "objective_dac", tuple(check_dac(w) for w in self.objective_dac))
