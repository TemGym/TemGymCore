"""Set the condenser's tilt pivot point with the tilt balance, from the alignment guide (S8, M8).

The guide has this goal. The beam tilts on the specimen without a shift. The script works in MAG
from X15k up, row by row. At each magnification, it centres the beam with CLA1. Then it wobbles the
beam tilt CLA2 along x, as the console's Tilt X wobbler does. It sets TILT BALANCE x where the beam
swings least. Then it does the same along y. At the highest magnification, it wobbles once more and
reports the swing. Example E10 asks for a swing under 10 px.

The script changes the function mode, the selector, CLA1, CLA2 and TiltBal. It keeps the new TiltBal
at the end, because that is its purpose. Every other value goes back to its value at the start.
After an error or Ctrl-C, every value goes back, TiltBal too. It saves nothing. A new TiltBal also
moves the beam. At the start row, with CLA1 and CLA2 back at their start values, the beam can be off
the SightX. Centre it again, as after any balance change.

The guide also asks for a beam converged with CL3. The script leaves CL3 at the console's value. On
the simulated NEOARM, this value gives a beam of rms radius about 7 px.

A rehearsal on the simulated NEOARM runs on the CPU. From this folder:
JAX_PLATFORMS=cpu python tilt_balance.py
"""

import time

import numpy as np

SIMULATED = True  # False at the microscope

if SIMULATED:
    from simulated_neoarm import install_simulated_neoarm

    install_simulated_neoarm()
from PyJEM import TEM3, detector, sightx  # noqa: E402

from jemwrap import MicroscopeMonitor, PyJEMAdapter  # noqa: E402
from jemwrap.control import read_value, restore, set_value, settable_snapshot  # noqa: E402
from microscope_functions import (  # noqa: E402
    CENTRING_TRIAL_DAC,
    centre_with,
    check_binning,
    enter_mode,
    leave_mode,
    set_balance,
    wobble,
)

adapter = PyJEMAdapter(
    connect=not SIMULATED, tem3_module=TEM3, detector_module=detector, sightx_module=sightx
)
# jemwrap spaces the calls to the real NEOARM by 0.1 s (requirement R16). The simulated NEOARM
# needs no spacing, so a rehearsal spaces its calls by 1 ms.
monitor = MicroscopeMonitor(adapter=adapter, rate_hz=1000.0 if SIMULATED else 10.0)

MAG = 0
# ROWS holds every MAG row from X15k to X500k. The guide starts at 200K and increases the
# magnification. At the lower rows, a change of the balance moves the beam less. There the balance
# can go most of the way in fewer, larger steps while it is far out. On the simulated NEOARM,
# TiltBal y needed more than 150 steps from a start at X60k. A step of one row (about 1.3 times)
# keeps the centred beam on the SightX. A step from X100k to X200k at once took the beam off the
# SightX.
ROWS = range(9, 25)
# The wobble of CLA2, in DAC units each way. The value is a choice. In the rehearsal on the
# simulated NEOARM, this wobble kept the swinging beam on the SightX at every row.
WOBBLE_DAC = 200
# The script waits after each change before a frame, after a change of the selector, and after a
# change between MAG and DIFF. The first two waits are choices. No measurement gives the first wait.
# Requirement R15 gives the third wait.
SETTLE_S = 0.2
SETTLE_ROW_S = 1.0
SETTLE_MODE_S = 2.0
# The simulated NEOARM has no time to settle, so a rehearsal does not wait.
if SIMULATED:
    SETTLE_S = SETTLE_ROW_S = SETTLE_MODE_S = 0.0

# Example E10 asks that the wobbled beam swings less than this at the last row.
E10_PX = 10

# CHANGES holds the values that this script changes, for the restore at the end (Q20 of
# design/requirements.md). The script does not read or set back a value that it does not touch.
CHANGES = (
    "eos.function_mode",
    "eos.current_mag_selector_id",
    "deflectors.CLA1",
    "deflectors.CLA2",
    "deflectors.TiltBal",
)
check_binning(monitor)
start = settable_snapshot(monitor, CHANGES)
mode_values = None
try:
    print("function mode -> MAG")
    mode_values = enter_mode(monitor, MAG, SETTLE_MODE_S)
    for selector in ROWS:
        print(f"selector -> {selector}")
        set_value(monitor, "eos.current_mag_selector_id", selector)
        time.sleep(SETTLE_ROW_S)
        print(f"MAG row {selector}: {read_value(monitor, 'eos.mag')}")
        centre_with(monitor, "CLA1", CENTRING_TRIAL_DAC, SETTLE_S)
        for axis in "xy":
            swing = set_balance(
                monitor,
                "TiltBal",
                axis,
                "CLA2",
                "CLA1",
                WOBBLE_DAC,
                SETTLE_S,
            )
            print(f"tilt {axis} wobble after the balance: swing {np.round(swing, 1)} px")

    # For example E10, the script does both wobbles with the new balance at the highest
    # magnification.
    for axis in "xy":
        swing = np.hypot(*wobble(monitor, "CLA2", axis, WOBBLE_DAC, SETTLE_S))
        verdict = "under" if swing < E10_PX else "not under"
        print(
            f"E10: tilt {axis} wobbler at the last row swings the beam {swing:.1f} px, "
            f"{verdict} {E10_PX} px"
        )
    print(f"TiltBal now {tuple(int(v) for v in read_value(monitor, 'deflectors.TiltBal'))}")
except BaseException:
    print("every value that a script may change -> its value at the start")
    try:
        leave_mode(monitor, MAG, mode_values)
    finally:
        restore(monitor, start)
    raise
print("every value but TiltBal -> its value at the start")
# The restore runs even if a step before it fails.
try:
    leave_mode(monitor, MAG, mode_values)
finally:
    restore(monitor, {key: value for key, value in start.items() if key != "deflectors.TiltBal"})
