"""Measure how far CLA1 moves the beam at MAG 40k.

The script changes CLA1 only. It saves five SightX frames and the CLA1 values to cla1_kicks.npz.
At the end, also after an error or Ctrl-C, it sets every value that a script may change back to
its value at the start.

All calls to the microscope go through jemwrap, which spaces them by at least 0.1 s and refuses
values that a script must not change.
"""

import time
from pathlib import Path

import numpy as np

SIMULATED = True  # False at the microscope

if SIMULATED:
    from simulated_neoarm import install_simulated_neoarm

    install_simulated_neoarm()
from PyJEM import TEM3, detector, sightx  # noqa: E402

from jemwrap import MicroscopeMonitor, PyJEMAdapter  # noqa: E402
from microscope_functions import check_binning  # noqa: E402
from jemwrap.control import (  # noqa: E402
    read_value,
    restore,
    set_value,
    settable_snapshot,
    take_frame,
)

adapter = PyJEMAdapter(
    connect=not SIMULATED, tem3_module=TEM3, detector_module=detector, sightx_module=sightx
)
# jemwrap spaces the calls to the real NEOARM by 0.1 s (requirement R16). The simulated NEOARM
# needs no spacing, so a rehearsal spaces its calls by 1 ms.
monitor = MicroscopeMonitor(adapter=adapter, rate_hz=1000.0 if SIMULATED else 10.0)

# A kick of 200 units moves the beam about 40 px at MAG 40k (recording of 2026-11-12).
KICKS_DAC = (-400, -200, 0, 200, 400)
# The wait after each change of CLA1, before a frame. This time is a choice. No measurement gives
# it.
SETTLE_S = 0.2
# The simulated NEOARM has no time to settle, so a rehearsal does not wait.
if SIMULATED:
    SETTLE_S = 0.0

# CHANGES holds the values that this script changes, for the restore at the end (Q20 of
# design/requirements.md). The script does not read or set back a value that it does not touch.
CHANGES = ("deflectors.CLA1",)
check_binning(monitor)
start = settable_snapshot(monitor, CHANGES)
# jemwrap gives DAC values as unsigned 16-bit numbers. Plain integers can go below zero in a sum.
cla1_x_start, cla1_y_start = (int(dac) for dac in read_value(monitor, "deflectors.CLA1"))
frames = []
try:
    for kick_dac in KICKS_DAC:
        cla1 = (cla1_x_start + kick_dac, cla1_y_start)
        print(f"CLA1 -> {cla1}")
        set_value(monitor, "deflectors.CLA1", cla1)
        time.sleep(SETTLE_S)
        frames.append(take_frame(monitor))
finally:
    print("every value that a script may change -> its value at the start")
    restore(monitor, start)

np.savez(
    Path(__file__).parent / "cla1_kicks.npz",
    frames=np.array(frames),
    kicks_dac=KICKS_DAC,
    cla1_start=(cla1_x_start, cla1_y_start),
)
print(f"saved {len(frames)} frames to cla1_kicks.npz")
