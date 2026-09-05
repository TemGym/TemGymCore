"""Read the recorded free-lens table without executing its research notebook.

Magnifications are nominal labels, not calibrated image measurements. Raw
16-bit control values are neither measured currents nor permission to drive
the instrument. This module performs no model fit or hardware operation.
"""

import ast
import json
from pathlib import Path

import numpy as np


PROJECTOR_LENSES = ("IL1", "IL2", "IL3", "PL1")
CONTROL_NAMES = PROJECTOR_LENSES + ("OLf", "OLc")


def read_flc_table(notebook: str | Path, *, minimum_mag=40_000, maximum_mag=600_000):
    """Return sorted nominal magnifications and raw DAC columns for one range.

    Only a single top-level literal assignment to ``RAW_DAC_HEX`` is read;
    notebook code and stored outputs are never executed. Each control column
    has the same row order as ``nominal_magnification``.
    """
    if not 0 < minimum_mag <= maximum_mag < float("inf"):
        raise ValueError("Magnification limits must be finite, positive, and ordered.")
    document = json.loads(Path(notebook).read_text(encoding="utf-8"))
    assignments = []
    for cell in document["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "RAW_DAC_HEX" not in source:
            continue
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "RAW_DAC_HEX"
                for target in node.targets
            ):
                assignments.append(ast.literal_eval(node.value))
    if len(assignments) != 1 or not isinstance(assignments[0], dict):
        raise ValueError("Expected exactly one literal RAW_DAC_HEX dictionary.")
    rows = []
    for label, controls in assignments[0].items():
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
