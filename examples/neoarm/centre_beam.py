"""Move the beam back to the middle of the SightX with the fitted model (scenario S3).

When the model's prediction leaves the beam off centre, this script brings it back in one step. It
takes one frame and finds the beam. It reads from neoarm_response.json how far CLA1 (MAG) or the
projector deflector PLA (DIFF) moves the beam at the row of the microscope. The fitted model of
notebook 2 gives these values. The script changes that deflector by the amount that, by the model,
brings the beam to the middle. It takes one more frame to show where the beam landed.

The script changes only CLA1 or PLA, and keeps the new value at the end, because that is its
purpose. It saves nothing. After an error or Ctrl-C, it sets every value that a script may change
back to its value at the start. It needs the beam on the SightX. For a beam off the camera, use the
search of record_lens_changes.py.

The table holds for the balances of the recording that notebook 2 fitted. When tilt_balance.py or
shift_balance.py changes TiltBal or ShiftBal, CLA1 acts differently. Then the table needs a new
recording and fit.

On the simulated NEOARM, the script first does what an operator might do. It selects another row,
so that the beam lands off centre. At the microscope, that part does not run.

A rehearsal on the simulated NEOARM runs on the CPU. From this folder:
JAX_PLATFORMS=cpu python centre_beam.py
"""

import json
import time
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
from microscope_functions import check_binning, enter_mode, find_beam, leave_mode  # noqa: E402

adapter = PyJEMAdapter(
    connect=not SIMULATED, tem3_module=TEM3, detector_module=detector, sightx_module=sightx
)
# jemwrap spaces the calls to the real NEOARM by 0.1 s (requirement R16). The simulated NEOARM
# needs no spacing, so a rehearsal spaces its calls by 1 ms.
monitor = MicroscopeMonitor(adapter=adapter, rate_hz=1000.0 if SIMULATED else 10.0)

# PyJEM's numbers for the two function modes.
MAG, DIFF = 0, 4
MODE_NAMES = {MAG: "MAG", DIFF: "DIFF"}
# The fitted model's px per DAC unit of the centring deflector at each row (notebook 2).
RESPONSE = json.loads((Path(__file__).parent / "neoarm_response.json").read_text())
# The beam centre must be within this many pixels of the middle (requirement R20, example E4).
CENTRED_PX = 10
# The wait after the change of the deflector, before the frame. This time is a choice. No
# measurement gives it.
SETTLE_DEFLECTOR_S = 0.2
# The row that the rehearsal selects first. MAG X60k is a row that the fit never saw.
REHEARSAL_MODE, REHEARSAL_SELECTOR = MAG, 15
# The simulated NEOARM has no time to settle, so a rehearsal does not wait.
if SIMULATED:
    SETTLE_DEFLECTOR_S = 0.0

# CHANGES holds the values that this script changes, for the restore after an error (Q20 of
# design/requirements.md). The script does not read or set back a value that it does not touch.
CHANGES = (
    "eos.function_mode",
    "eos.current_mag_selector_id",
    "deflectors.CLA1",
    "deflectors.CLA2",
    "deflectors.PLA",
)
check_binning(monitor)
start = settable_snapshot(monitor, CHANGES)
mode_values = None
try:
    if SIMULATED:
        print(f"rehearsal: function mode -> {MODE_NAMES[REHEARSAL_MODE]}")
        mode_values = enter_mode(monitor, REHEARSAL_MODE, 0.0)
        print(f"rehearsal: selector -> {REHEARSAL_SELECTOR}")
        set_value(monitor, "eos.current_mag_selector_id", REHEARSAL_SELECTOR)

    mode = MODE_NAMES[int(read_value(monitor, "eos.function_mode"))]
    selector = str(int(read_value(monitor, "eos.current_mag_selector_id")))
    row = RESPONSE[mode][selector]
    deflector = row["deflector"]
    print(f"{mode} row {selector} ({row['label']}): centring with {deflector}")

    x, y, _ = find_beam(take_frame(monitor))
    if np.isnan(x):
        raise SystemExit("no whole beam on the SightX: find the beam first")
    print(f"beam centre before: ({x:.1f}, {y:.1f}) px from the middle")

    # The model moves the beam by px_per_dac times the change of the deflector. The change that
    # moves the beam from (x, y) to the middle solves that for (-x, -y).
    change = np.linalg.solve(np.array(row["px_per_dac"]), [-x, -y])
    deflector_x, deflector_y = read_value(monitor, f"deflectors.{deflector}")
    new_value = (int(round(deflector_x + change[0])), int(round(deflector_y + change[1])))
    print(f"{deflector} -> {new_value}")
    set_value(monitor, f"deflectors.{deflector}", new_value)
    time.sleep(SETTLE_DEFLECTOR_S)

    x, y, _ = find_beam(take_frame(monitor))
    if np.isnan(x):
        raise SystemExit("the beam left the SightX: the table does not fit this state")
    miss = np.hypot(x, y)
    print(f"beam centre after: ({x:.1f}, {y:.1f}) px from the middle, {miss:.1f} px off")
    print("centred (E4)" if miss <= CENTRED_PX else f"more than {CENTRED_PX} px off: run again")
except BaseException:
    print("every value that a script may change -> its value at the start")
    try:
        leave_mode(monitor, REHEARSAL_MODE, mode_values)
    finally:
        restore(monitor, start)
    raise
