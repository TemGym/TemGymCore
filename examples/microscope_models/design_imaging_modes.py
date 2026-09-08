"""Find and check 40k–600k imaging settings with a fixed objective and projector.

Run from the repository root. Add --write-table to save currents to microscope.toml.
The currents describe the simulation, not calibrated microscope DAC commands.
"""
import argparse
import json
from pathlib import Path
import re

import jax
import numpy as np

from temgym_core.microscope import Microscope, find_imaging_condition

jax.config.update("jax_enable_x64", True)
CONFIG = Path(__file__).with_name("microscope.toml")
TARGETS = (40_000, 60_000, 80_000, 100_000, 150_000,
           200_000, 250_000, 300_000, 400_000, 600_000)


def design(microscope):
    """Return successful imaging solutions, continuing from each previous setting."""
    initial = microscope.with_settings(magnification=TARGETS[0]).with_currents(PL1=-3.5)
    solutions = []
    for target in TARGETS:
        solution = find_imaging_condition(
            initial, magnification=target, vary=["IL1", "IL2", "IL3"],
        )
        solutions.append(solution)
        initial = solution.microscope
    return solutions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-table", action="store_true")
    args = parser.parse_args()
    microscope = Microscope.from_toml(CONFIG)
    solutions = design(microscope)
    table = microscope.tables["magnification"]
    rows = [[target, *[float(s.microscope.components[name].current) for name in table.lens_names]]
            for target, s in zip(TARGETS, solutions)]
    for target, solution in zip(TARGETS, solutions):
        matrix = solution.matrix
        print(f"{target:6d}x  max |A/M - I| = {np.max(np.abs(matrix[:2, :2] / target - np.eye(2))):.2e}"
              f"  max |B| = {np.max(np.abs(matrix[:2, 2:4])):.2e} m")
    if args.write_table:
        block = ('[modes.imaging.magnification]\n'
                 '# Simulation currents: use find_imaging_condition after interpolation.\n'
                 f'headers = {json.dumps(["mag", *table.lens_names])}\nvalues = [\n'
                 + ''.join(f'  {json.dumps(row)},\n' for row in rows) + ']\n\n')
        source, count = re.subn(
            r'\[modes\.imaging\.magnification\]\n.*?(?=\[modes\.imaging\.diffraction\])',
            lambda _: block, CONFIG.read_text(), flags=re.S,
        )
        if count != 1:
            raise ValueError("Expected one magnification table before the diffraction table.")
        CONFIG.write_text(source)


if __name__ == "__main__":
    main()
