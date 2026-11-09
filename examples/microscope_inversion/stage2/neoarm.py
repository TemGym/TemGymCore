"""Stage 2's copy of the NEOARM facts (the stage-1 notebook has the version in use). Facts about the NEOARM, written once: the projection lenses, the DAC range and the recorded
free-lens table (``neoarm_lens_table.json``: IL1-IL3, PL1, OLf and OLc DAC values at each nominal
magnification). Used by the fit and the virtual NEOARM; ``record.py`` names its four lenses itself, so that it
needs only ``pyjem_session.py`` beside it on the microscope PC.

Magnifications in the table are nominal labels, not calibrated measurements.

Uses TemGym: nothing.
"""

import json
from pathlib import Path

import numpy as np

#: Every lens supply is a 16-bit DAC: 0 to this.
DAC_MAX = 65535


#: The projection lenses the calibration drives, in this order everywhere.
PROJECTOR_LENSES = ("IL1", "IL2", "IL3", "PL1")
CONTROL_NAMES = PROJECTOR_LENSES + ("OLf", "OLc")

TABLE = Path(__file__).parents[2] / "virtual_neoarm" / "neoarm_lens_table.json"


def row_40k():
    """The recorded 40k row (at 200 kV): lens name -> DAC value, for IL1-IL3, PL1, OLf and OLc."""
    table = read_flc_table(minimum_mag=40_000, maximum_mag=40_000)
    return {name: int(table[name][0]) for name in CONTROL_NAMES}


def read_flc_table(path: str | Path = TABLE, *, minimum_mag=40_000, maximum_mag=600_000):
    """Return sorted nominal magnifications and raw DAC columns for one range.

    Each control column has the same row order as ``nominal_magnification``.
    """
    if not 0 < minimum_mag <= maximum_mag < float("inf"):
        raise ValueError("Magnification limits must be finite, positive, and ordered.")
    raw = json.loads(Path(path).read_text(encoding="utf-8"))["raw_dac_hex"]
    rows = []
    for label, controls in raw.items():
        mag = int(label)
        if str(mag) != str(label) or mag <= 0:
            raise ValueError("Magnification labels must be canonical positive integers.")
        if set(controls) != set(CONTROL_NAMES):
            raise ValueError("Every row must contain IL1, IL2, IL3, PL1, OLf, and OLc.")
        values = [int(controls[name], 16) for name in CONTROL_NAMES]
        if any(value < 0 or value > DAC_MAX for value in values):
            raise ValueError("Raw DAC values must fit 16 bits (0-65535).")
        if minimum_mag <= mag <= maximum_mag:
            rows.append((mag, values))
    if not rows:
        raise ValueError("No recorded settings in the requested magnification range.")
    rows.sort()
    result = {"nominal_magnification": np.array([row[0] for row in rows])}
    for column, name in enumerate(CONTROL_NAMES):
        result[name] = np.array([row[1][column] for row in rows], dtype=np.int64)
    for values in result.values():
        values.setflags(write=False)
    return result
