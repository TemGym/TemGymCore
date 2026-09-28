"""Read the recorded NEOARM free-lens table from ``neoarm_flc_table.json``.

Magnifications are nominal labels, not calibrated image measurements. Raw
16-bit control values are neither measured currents nor permission to drive
the instrument. This module performs no model fit or hardware operation.
"""

import json
from pathlib import Path

import numpy as np


PROJECTOR_LENSES = ("IL1", "IL2", "IL3", "PL1")
CONTROL_NAMES = PROJECTOR_LENSES + ("OLf", "OLc")


TABLE = Path(__file__).with_name("neoarm_flc_table.json")


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
        if any(value < 0 or value > 65535 for value in values):
            raise ValueError("Raw DAC values must fit an unsigned 16-bit word.")
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
