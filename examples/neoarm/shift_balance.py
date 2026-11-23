"""Set the condenser's shift pivot point with the shift balance, from the alignment guide (S8, M8).

The guide has this goal. The beam shifts on the specimen without a tilt. In DIFF, the SightX shows
the beam's angle. A pure shift does not move the diffraction spot. The script works in DIFF from 30
cm up, row by row. At each camera length, it centres the spot with the projector deflector PLA
(diffraction shift). Then it wobbles the beam shift CLA1 along x, as the console's Shift X wobbler
does. It sets SHIFT BALANCE x where the spot swings least. Then it does the same along y. At the
longest camera length, it wobbles once more and reports the swing. Example E10 asks for a swing
under 10 px. Run the script after tilt_balance.py, as the guide does.

The script changes the function mode, the selector, CLA1, PLA and ShiftBal. It keeps the new
ShiftBal at the end, because that is its purpose. Every other value goes back to its value at the
start. After an error or Ctrl-C, every value goes back, ShiftBal too. It saves nothing. A new
ShiftBal also moves the beam. At the start row, with CLA1 and PLA back at their start values, the
beam can be off the SightX. Centre it again, as after any balance change.

The guide also asks for the smallest diffraction spot with CL3 and a focused caustic with DIFF
FOCUS (IL1). It also asks for ANGLE BALANCE where the swing is oval. The script leaves CL3 and IL1
at the console's values. It does not touch AngBal, because jemwrap does not let a script set it.

A rehearsal on the simulated NEOARM runs on the CPU. From this folder:
JAX_PLATFORMS=cpu python shift_balance.py
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

DIFF = 4
# ROWS holds every DIFF row from 30 cm to 150 cm. The guide increases the camera length. A step of
# one row keeps the centred spot on the SightX.
ROWS = range(6, 14)
# The wobble of CLA1, in DAC units each way. The value is a choice. On the simulated NEOARM, CLA1
# gives about 1.5 nm of pure shift per unit (2026-11-22). In the rehearsal, this wobble kept the
# swinging beam on the SightX at every row.
WOBBLE_DAC = 7
# The projector deflector PLA (diffraction shift) keeps the pattern in the middle. PLA does not
# change the illumination. Beam tilt (CLA2) would change it. After the tilt balance, CLA2 moves the
# pattern only about 0.01 px per 20 units. PLA_TRIAL_DAC is the first trial kick of PLA. It moves
# the pattern about 15 px on the simulated NEOARM. The value is a choice.
PLA_TRIAL_DAC = 1000
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
    "deflectors.PLA",
    "deflectors.ShiftBal",
)
check_binning(monitor)
start = settable_snapshot(monitor, CHANGES)
mode_values = None
try:
    print("function mode -> DIFF")
    mode_values = enter_mode(monitor, DIFF, SETTLE_MODE_S)
    for selector in ROWS:
        print(f"selector -> {selector}")
        set_value(monitor, "eos.current_mag_selector_id", selector)
        time.sleep(SETTLE_ROW_S)
        print(f"DIFF row {selector}: camera length {read_value(monitor, 'eos.mag')} cm")
        centre_with(monitor, "PLA", PLA_TRIAL_DAC, SETTLE_S)
        for axis in "xy":
            swing = set_balance(
                monitor,
                "ShiftBal",
                axis,
                "CLA1",
                "PLA",
                WOBBLE_DAC,
                SETTLE_S,
                centring_trial_dac=PLA_TRIAL_DAC,
            )
            print(f"shift {axis} wobble after the balance: swing {np.round(swing, 1)} px")

    # For example E10, the script does both wobbles with the new balance at the longest camera
    # length.
    for axis in "xy":
        swing = np.hypot(*wobble(monitor, "CLA1", axis, WOBBLE_DAC, SETTLE_S))
        verdict = "under" if swing < E10_PX else "not under"
        print(
            f"E10: shift {axis} wobbler at the last row swings the spot {swing:.1f} px, "
            f"{verdict} {E10_PX} px"
        )
    print(f"ShiftBal now {tuple(int(v) for v in read_value(monitor, 'deflectors.ShiftBal'))}")
except BaseException:
    print("every value that a script may change -> its value at the start")
    try:
        leave_mode(monitor, DIFF, mode_values)
    finally:
        restore(monitor, start)
    raise
print("every value but ShiftBal -> its value at the start")
# The restore runs even if a step before it fails.
try:
    leave_mode(monitor, DIFF, mode_values)
finally:
    restore(monitor, {key: value for key, value in start.items() if key != "deflectors.ShiftBal"})
