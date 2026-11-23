"""Record how far each lens lags behind its DAC value, with the same value reached by five routes.

The script works at MAG X40k. It moves the beam off the middle of the SightX with CLA1. A change of
a projector lens, which changes the magnification, moves the beam. Then the script takes each of
CL1, CL3, IL1, IL2, IL3 and PL1 in turn. It brings the lens to its value at this row by five routes
and takes one frame after each route. The routes are these:

1. From APPROACH_DAC below.
2. To the value plus STEP_DAC, from below. The beam moves by the lens's px per DAC unit.
3. From APPROACH_DAC above, or from DAC_MAX.
4. From below again. The beam must come back to its position after route 1 (repeatability).
5. From above again. The beam must come back to its position after route 3.

The jump between routes 1 and 3 is twice the lens's lag, in pixels. analyse_hysteresis.py changes
it into DAC units with route 2. This script is stage 10 of the change record. It is also the
from-below, from-above test of design/lens_hysteresis.md. It uses only the approaches and does not
relax the lenses. The principal decided this on 2026-11-21.

The script changes the function mode, the selector, CL1, CL3, IL1, IL2, IL3, PL1 and CLA1. It
saves each frame to its own .npy file and one line for each frame in table.csv, in
output/record_hysteresis/<time>/. At the end, also after an error or Ctrl-C, it sets every value
that a script may change back to its value at the start.

A rehearsal on the simulated NEOARM runs on the CPU. From this folder:
JAX_PLATFORMS=cpu python record_hysteresis.py
"""

import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np

SIMULATED = True  # False at the microscope

if SIMULATED:
    from simulated_neoarm import install_simulated_neoarm

    install_simulated_neoarm()
from PyJEM import TEM3, detector, sightx  # noqa: E402

from jemwrap import MicroscopeMonitor, PyJEMAdapter  # noqa: E402
from jemwrap.control import (  # noqa: E402
    read_value,
    restore,
    set_value,
    settable_snapshot,
    take_frame,
)
from microscope_functions import (  # noqa: E402
    APPROACH_DAC,
    centre_with,
    check_binning,
    enter_mode,
    find_beam,
    leave_mode,
)  # noqa: E402

adapter = PyJEMAdapter(
    connect=not SIMULATED, tem3_module=TEM3, detector_module=detector, sightx_module=sightx
)
# jemwrap spaces the calls to the real NEOARM by 0.1 s (requirement R16). The simulated NEOARM
# needs no spacing, so a rehearsal spaces its calls by 1 ms.
monitor = MicroscopeMonitor(adapter=adapter, rate_hz=1000.0 if SIMULATED else 10.0)

MAG, ROW = 0, 13  # MAG X40k
LENSES = ("CL1", "CL3", "IL1", "IL2", "IL3", "PL1")
DAC_MAX = 65535
# The step of route 2. On the simulated NEOARM with the beam in the middle, 300 units of CL3 moved
# the beam 25 px. 300 units of IL1 moved it 0.7 px (2026-11-22). The value is a choice.
STEP_DAC = 300
# The beam sits this far from the middle along x and y, so that a change of magnification moves
# it. The value is a choice. It keeps the beam and its moves on the SightX.
OFFSET_PX = (100, 60)
# The trial kick of CLA1 for the centring and the offset (centre_with). The value is a choice.
CLA1_TRIAL_DAC = 20
# The script waits for the column to settle. The values are choices, and no measurement gives them.
# R15 gives 2 s after a change of mode.
SETTLE_MODE_S = 2.0
SETTLE_ROW_S = 1.0
SETTLE_LENS_S = 0.5
SETTLE_DEFLECTOR_S = 0.2
# The simulated NEOARM has no time to settle, so a rehearsal does not wait.
if SIMULATED:
    SETTLE_MODE_S = SETTLE_ROW_S = SETTLE_LENS_S = SETTLE_DEFLECTOR_S = 0.0

OUTPUT = (
    Path(__file__).parent / "output/record_hysteresis" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
)
FIELDS = ["frame", "lens", "route", "from_dac", "dac", "beam_x_px", "beam_y_px", "beam_rms_px"]


def set_lens_by_route(monitor, name, dac, start_dac):
    """Set one lens under free lens control to ``dac``, coming from ``start_dac``."""
    print(f"{name} -> {start_dac} -> {dac}")
    set_value(monitor, f"lens.{name}", start_dac)
    set_value(monitor, f"lens.{name}", dac)
    time.sleep(SETTLE_LENS_S)


def record_frame(monitor, name, route, start_dac, dac):
    """Take one frame, save it, and write one line of the table."""
    frame = take_frame(monitor)
    file_name = f"{len(saved_frames):04d}.npy"
    np.save(OUTPUT / file_name, frame)
    saved_frames.append(file_name)
    x, y, rms = find_beam(frame)
    table.writerow(
        dict(
            frame=file_name,
            lens=name,
            route=route,
            from_dac=start_dac,
            dac=dac,
            beam_x_px=round(float(x), 2),
            beam_y_px=round(float(y), 2),
            beam_rms_px=round(float(rms), 2),
        )
    )
    table_file.flush()
    print(f"{name} route {route}: beam at ({x:.2f}, {y:.2f}) px from the middle")


# CHANGES holds the values that this script changes, for the restore at the end (Q20 of
# design/requirements.md). The script does not read or set back a value that it does not touch.
CHANGES = (
    "eos.function_mode",
    "eos.current_mag_selector_id",
    "lens.CL1",
    "lens.CL3",
    "lens.IL1",
    "lens.IL2",
    "lens.IL3",
    "lens.PL1",
    "deflectors.CLA1",
)
check_binning(monitor)
start = settable_snapshot(monitor, CHANGES)
OUTPUT.mkdir(parents=True)
saved_frames = []
table_file = open(OUTPUT / "table.csv", "w", newline="")
table = csv.DictWriter(table_file, FIELDS)
table.writeheader()
mode_values = None
try:
    print("function mode -> MAG")
    mode_values = enter_mode(monitor, MAG, SETTLE_MODE_S)
    # The lenses leave free lens control first, so that they take the console's values of the row.
    print(", ".join(LENSES) + " -> the console's values")
    restore(monitor, {f"lens.{name}": (False, start[f"lens.{name}"][1]) for name in LENSES})
    print(f"selector -> {ROW}")
    set_value(monitor, "eos.current_mag_selector_id", ROW)
    time.sleep(SETTLE_ROW_S)
    # Every lens of the script first comes from below, so that each starts with the same history.
    row_dac = {name: int(read_value(monitor, f"lens.{name}")) for name in LENSES}
    for name in LENSES:
        set_lens_by_route(monitor, name, row_dac[name], row_dac[name] - APPROACH_DAC)
    centre_with(monitor, "CLA1", CLA1_TRIAL_DAC, SETTLE_DEFLECTOR_S)
    # CLA1 moves the beam from the middle by OFFSET_PX, from its measured response.
    cla1 = np.array(read_value(monitor, "deflectors.CLA1"), dtype=int)
    middle = np.array(find_beam(take_frame(monitor))[:2])
    response = []
    for step in ((CLA1_TRIAL_DAC, 0), (0, CLA1_TRIAL_DAC)):
        print(f"CLA1 -> {tuple(int(v) for v in cla1 + step)} (trial)")
        set_value(monitor, "deflectors.CLA1", tuple(int(v) for v in cla1 + step))
        time.sleep(SETTLE_DEFLECTOR_S)
        moved = np.array(find_beam(take_frame(monitor))[:2])
        if np.isnan(moved).any():
            raise RuntimeError("a trial kick of CLA1 took the beam off the SightX")
        response.append((moved - middle) / CLA1_TRIAL_DAC)
    offset = np.linalg.solve(np.array(response).T, np.array(OFFSET_PX) - middle)
    print(f"CLA1 -> {tuple(int(round(v)) for v in cla1 + offset)}")
    set_value(monitor, "deflectors.CLA1", tuple(int(round(v)) for v in cla1 + offset))
    time.sleep(SETTLE_DEFLECTOR_S)

    for name in LENSES:
        dac = row_dac[name]
        above = min(dac + APPROACH_DAC, DAC_MAX)
        for route, (target, start_dac) in enumerate(
            (
                (dac, dac - APPROACH_DAC),
                (dac + STEP_DAC, dac + STEP_DAC - APPROACH_DAC),
                (dac, above),
                (dac, dac - APPROACH_DAC),
                (dac, above),
            ),
            start=1,
        ):
            if target > DAC_MAX:
                print(f"{name} route {route}: {target} is above {DAC_MAX}, left out")
                continue
            set_lens_by_route(monitor, name, target, start_dac)
            record_frame(monitor, name, route, start_dac, target)
        set_lens_by_route(monitor, name, dac, dac - APPROACH_DAC)
finally:
    print("every value that a script may change -> its value at the start")
    try:
        leave_mode(monitor, MAG, mode_values)
    finally:
        restore(monitor, start)
        table_file.close()

print(f"saved {len(saved_frames)} frames and table.csv to {OUTPUT}")
