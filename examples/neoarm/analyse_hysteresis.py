"""The lag of each lens, from a recording of record_hysteresis.py.

For each lens, route 2 (the value plus a step, from below) gives how far one DAC unit moves the
beam. The jump from route 1 (from below) to route 3 (from above) is twice the lag. From below, a
lens acts as if its value were lower by its lag. From above, it acts as if its value were higher.
The lag is that jump along the direction of route 2, in DAC units. Routes 4 and 5 repeat routes 1
and 3. Their distance from routes 1 and 3 shows how well the microscope repeats a route.

Run it from this folder with the command below. With no argument, it reads the newest recording.
python analyse_hysteresis.py [output/record_hysteresis/<time>/table.csv]
"""

import csv
import sys
from pathlib import Path

import numpy as np

recordings = sorted((Path(__file__).parent / "output/record_hysteresis").glob("*/table.csv"))
table_path = Path(sys.argv[1]) if len(sys.argv) > 1 else recordings[-1]
print(f"recording: {table_path}")
with open(table_path, newline="") as table:
    lines = list(csv.DictReader(table))

print(f"{'lens':5s} {'px per DAC':>11s} {'jump (px)':>10s} {'lag (DAC)':>10s} {'repeat (px)':>12s}")
for name in dict.fromkeys(line["lens"] for line in lines):
    beam = {
        int(line["route"]): np.array([float(line["beam_x_px"]), float(line["beam_y_px"])])
        for line in lines
        if line["lens"] == name
    }
    dac = {int(line["route"]): int(line["dac"]) for line in lines if line["lens"] == name}
    if 2 not in beam or 3 not in beam:
        print(f"{name:5s} routes 2 or 3 missing")
        continue
    # The step of route 2, from the table.
    per_dac = (beam[2] - beam[1]) / (dac[2] - dac[1])
    jump = beam[3] - beam[1]
    # The lag is half the jump, in DAC units along the direction that the lens moves the beam.
    lag_dac = (jump @ per_dac) / (per_dac @ per_dac) / 2
    repeat_px = max(np.hypot(*(beam[4] - beam[1])), np.hypot(*(beam[5] - beam[3])))
    print(
        f"{name:5s} {np.hypot(*per_dac):11.4f} {np.hypot(*jump):10.2f} {lag_dac:10.1f} "
        f"{repeat_px:12.2f}"
    )
