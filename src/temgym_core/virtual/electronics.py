"""Hidden electronics of the virtual microscope: control words to physical controls.

A JEOL column is set through 16-bit DAC words (0-65535): one per lens supply and
two per deflector pair. The real relation between a word and the lens current or
the specimen kick is not known to the user. Here it is part of the hidden truth:

* :class:`LensDrive` maps a lens word to the relative lens control used by the
  calibration model (current divided by its reference current).
* :class:`DeflectorDrive` maps the CLA1 (shift) and CLA2 (tilt) words to the
  calibrated specimen displacement and angle, with the magnetic 1/p scaling of
  a fixed deflector field when the high tension changes.
"""

from dataclasses import dataclass, field

import numpy as np

from ..calibration.condenser_model import magnetic_strength_ratio

WORD_MIN, WORD_MAX = 0, 65535
NEUTRAL_WORD = 32768


class WordRangeError(ValueError):
    """A requested control word lies outside the 16-bit range 0-65535."""


def check_word(value, name="word"):
    """Return ``value`` as an int after checking it is an integer word in 0-65535.

    Parameters
    ----------
    value : int
        Requested word.
    name : str, optional
        Name used in the error message.

    Returns
    -------
    int
        The validated word.

    Raises
    ------
    WordRangeError
        If ``value`` is not integral or outside 0-65535.
    """
    if isinstance(value, (bool, np.bool_)) or int(value) != value:
        raise WordRangeError(f"{name} must be an integer word, not {value!r}.")
    value = int(value)
    if not WORD_MIN <= value <= WORD_MAX:
        raise WordRangeError(f"{name}={value} is outside the 16-bit range {WORD_MIN}-{WORD_MAX}.")
    return value


@dataclass(frozen=True)
class LensDrive:
    """Linear lens supply: ``control = (word - offset_word) / words_per_unit``.

    Parameters
    ----------
    words_per_unit : float
        DAC words per unit of relative lens control (positive).
    offset_word : float, optional
        Word at which the relative control is zero.
    """

    words_per_unit: float
    offset_word: float = 0.0

    def __post_init__(self):
        if not np.isfinite(self.words_per_unit) or self.words_per_unit <= 0:
            raise ValueError("words_per_unit must be finite and positive.")
        if not np.isfinite(self.offset_word):
            raise ValueError("offset_word must be finite.")

    def control(self, word):
        """Relative lens control for an integer word.

        Parameters
        ----------
        word : int
            Lens DAC word.

        Returns
        -------
        float
            Current divided by the reference current.
        """
        return (check_word(word) - self.offset_word) / self.words_per_unit

    def word(self, control):
        """Nearest integer word for a relative control.

        Parameters
        ----------
        control : float
            Current divided by the reference current.

        Returns
        -------
        int
            The rounded word.

        Raises
        ------
        WordRangeError
            If the control needs a word outside 0-65535.
        """
        return check_word(int(np.rint(control * self.words_per_unit + self.offset_word)))


@dataclass(frozen=True)
class DeflectorDrive:
    """Linear CLA deflectors: words to specimen kick ``[x, y, theta_x, theta_y]``.

    Parameters
    ----------
    kick_per_word : array_like
        4x4 matrix from word offsets ``[CLA1 x, CLA1 y, CLA2 x, CLA2 y]`` to the
        specimen displacement (m) and angle (rad) at ``reference_voltage_v``.
    zero_words : tuple of int, optional
        Words giving no kick.
    reference_voltage_v : float, optional
        High tension at which ``kick_per_word`` holds. A fixed deflector field
        deflects by ``p(reference) / p(voltage)`` times more at other voltages.
    """

    kick_per_word: np.ndarray
    zero_words: tuple = (NEUTRAL_WORD,) * 4
    reference_voltage_v: float = 200e3

    def __post_init__(self):
        matrix = np.array(self.kick_per_word, dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) == 0:
            raise ValueError("kick_per_word must be a finite, invertible 4x4 matrix.")
        object.__setattr__(self, "kick_per_word", matrix)
        object.__setattr__(self, "zero_words", tuple(check_word(w) for w in self.zero_words))

    def kick(self, words, voltage_v):
        """Specimen kick produced by CLA words at a high tension.

        Parameters
        ----------
        words : array_like
            ``[CLA1 x, CLA1 y, CLA2 x, CLA2 y]`` integer words.
        voltage_v : float
            High tension.

        Returns
        -------
        numpy.ndarray
            ``[x, y, theta_x, theta_y]`` in metres and radians.
        """
        offsets = np.array([check_word(w) for w in words], dtype=float) - np.array(self.zero_words)
        ratio = float(magnetic_strength_ratio(voltage_v, self.reference_voltage_v))
        return ratio * self.kick_per_word @ offsets

    def words(self, kick, voltage_v):
        """Nearest integer CLA words for a requested kick.

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
        offsets = np.linalg.solve(self.kick_per_word * ratio, np.asarray(kick, dtype=float))
        return tuple(check_word(int(np.rint(o)) + z) for o, z in zip(offsets, self.zero_words))


@dataclass(frozen=True)
class Electronics:
    """All hidden word-to-physics laws of the virtual column.

    Parameters
    ----------
    lenses : dict
        :class:`LensDrive` per modelled lens: ``CL1``, ``CL3``, ``CM`` (condenser
        mini lens) and ``IL1``, ``IL2``, ``IL3``, ``PL1``.
    deflector : DeflectorDrive
        Beam shift (CLA1) and tilt (CLA2).
    objective_words : tuple of int, optional
        ``(OLc, OLf)``: the objective's words. Other objective words are not modelled.
    objective_control : float, optional
        The objective's relative control at those words (1 at the reference HT's alignment;
        p(HT) / p(reference) for a console aligned at another HT).
    """

    lenses: dict
    deflector: DeflectorDrive
    objective_words: tuple = (45818, 39178)
    objective_control: float = 1.0
    modelled: tuple = field(default=("CL1", "CL3", "CM", "IL1", "IL2", "IL3", "PL1"), init=False)

    def __post_init__(self):
        missing = set(self.modelled) - set(self.lenses)
        if missing:
            raise ValueError(f"Supply a LensDrive for {sorted(missing)}.")
        object.__setattr__(self, "lenses", dict(self.lenses))
        object.__setattr__(self, "objective_words", tuple(check_word(w) for w in self.objective_words))
