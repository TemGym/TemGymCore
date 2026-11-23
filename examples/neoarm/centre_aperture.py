"""Centre the condenser aperture, from the alignment guide (S8, example E21).

The guide has this goal. When CL3 takes the beam through its crossover, the beam does not move.
An aperture off the beam's axis lets through a beam that leans. On one side of the crossover, the
beam lands on one side. On the other side of the crossover, it lands on the other side. The script
works at MAG X8000. It centres the beam with CLA1. Then it takes the beam centre with CL3 a step
below and a step above its value. The difference of the two centres is how far the beam moves
through the crossover. Two trial moves of the aperture give how that difference changes with the
aperture's position. Then the aperture goes, in limited steps, to where the difference vanishes.
The script repeats this until the beam moves less than SHIFT_PX through the crossover.

The script centres the aperture that the console selects. Select the condenser aperture to centre
before the script starts. jemwrap does not let a script change the selection. On the simulated
NEOARM, the script selects the 10 um hole. This hole limits the beam as an alignment aperture does.

The script changes the function mode, the selector, CL3, CLA1 and the aperture position. It keeps
the new aperture position at the end, because that is its purpose. Every other value goes back to
its value at the start. After an error or Ctrl-C, every value goes back, the aperture too. It saves
nothing.

A rehearsal on the simulated NEOARM runs on the CPU. From this folder:
JAX_PLATFORMS=cpu python centre_aperture.py
"""

import time

import numpy as np

SIMULATED = True  # False at the microscope

if SIMULATED:
    from simulated_neoarm import install_simulated_neoarm

    # The operator chooses the 10 um hole of the simulated condenser aperture.
    install_simulated_neoarm(condenser_hole=4)
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
    CENTRING_TRIAL_DAC,
    centre_with,
    check_binning,
    enter_mode,
    find_beam,
    leave_mode,
    set_lens_from_below,
)

adapter = PyJEMAdapter(
    connect=not SIMULATED, tem3_module=TEM3, detector_module=detector, sightx_module=sightx
)
# jemwrap spaces the calls to the real NEOARM by 0.1 s (requirement R16). The simulated NEOARM
# needs no spacing, so a rehearsal spaces its calls by 1 ms.
monitor = MicroscopeMonitor(adapter=adapter, rate_hz=1000.0 if SIMULATED else 10.0)

MAG, ROW = 0, 6  # At MAG X8000, the beam stays on the SightX on both sides of the crossover.
# CL3 goes this far below and above its value, from below each time. On the simulated NEOARM at
# X8000, 800 units moved the beam by 3 to 40 px through the crossover, depending on the aperture
# (2026-11-22). The value is a choice.
CL3_STEP_DAC = 800
# The trial move of the aperture and its largest move at once, in Apt3 units. On the simulated
# NEOARM, 200 units changed the shift through the crossover by up to 40 px. Both values are
# choices.
TRIAL_APERTURE_DAC = 50
LARGEST_APERTURE_DAC = 150
# The aperture counts as centred when the beam moves less than this through the crossover (E21
# asks for under 10 px). The value is a choice.
SHIFT_PX = 1.0
MOST_STEPS = 20
# The script waits for the column to settle. The values are choices. R15 gives 2 s after a change of
# mode.
SETTLE_S = 0.2
SETTLE_LENS_S = 0.5
SETTLE_MODE_S = 2.0
SETTLE_ROW_S = 1.0
# The simulated NEOARM has no time to settle, so a rehearsal does not wait.
if SIMULATED:
    SETTLE_S = SETTLE_LENS_S = SETTLE_MODE_S = SETTLE_ROW_S = 0.0

# Example E21 asks that the beam moves less than this through the crossover after the script.
E21_PX = 10


def shift_through_crossover(monitor, cl3):
    """Return the beam centre with CL3 above its value minus the centre below it, in px."""
    centres = []
    for step in (-CL3_STEP_DAC, CL3_STEP_DAC):
        set_lens_from_below(monitor, "CL3", cl3 + step, SETTLE_LENS_S)
        x, y, _ = find_beam(take_frame(monitor))
        if np.isnan(x):
            raise RuntimeError("the beam left the SightX with CL3 away from its value")
        centres.append((x, y))
    set_lens_from_below(monitor, "CL3", cl3, SETTLE_LENS_S)
    return np.subtract(centres[1], centres[0])


def move_aperture(monitor, position):
    """Move the selected aperture to ``position`` (Apt3 units) and centre the beam again."""
    print(f"aperture -> {position}")
    set_value(monitor, "apertures.position", position)
    time.sleep(SETTLE_S)
    centre_with(monitor, "CLA1", CENTRING_TRIAL_DAC, SETTLE_S)


# CHANGES holds the values that this script changes, for the restore at the end (Q20 of
# design/requirements.md). The script does not read or set back a value that it does not touch.
CHANGES = (
    "eos.function_mode",
    "eos.current_mag_selector_id",
    "lens.CL3",
    "deflectors.CLA1",
    "apertures.position",
)
check_binning(monitor)
start = settable_snapshot(monitor, CHANGES)
mode_values = None
try:
    print("function mode -> MAG")
    mode_values = enter_mode(monitor, MAG, SETTLE_MODE_S)
    print(f"selector -> {ROW}")
    set_value(monitor, "eos.current_mag_selector_id", ROW)
    time.sleep(SETTLE_ROW_S)
    cl3 = int(read_value(monitor, "lens.CL3"))
    set_lens_from_below(monitor, "CL3", cl3, SETTLE_LENS_S)
    centre_with(monitor, "CLA1", CENTRING_TRIAL_DAC, SETTLE_S)
    print(f"aperture selected: {read_value(monitor, 'apertures.selected')}")

    for _ in range(MOST_STEPS):
        position = np.array(read_value(monitor, "apertures.position"), dtype=int)
        shift = shift_through_crossover(monitor, cl3)
        print(
            f"aperture at {tuple(int(v) for v in position)}: the beam moves "
            f"{np.round(shift, 1)} px through the crossover"
        )
        if np.hypot(*shift) < SHIFT_PX:
            break
        # The trial moves measure how the shift changes per unit of aperture move, along x and y.
        response = []
        for direction in ((1, 0), (0, 1)):
            move_aperture(
                monitor, tuple(int(v) for v in position + TRIAL_APERTURE_DAC * np.array(direction))
            )
            response.append((shift_through_crossover(monitor, cl3) - shift) / TRIAL_APERTURE_DAC)
        step = np.linalg.solve(np.array(response).T, -shift)
        step *= min(1.0, LARGEST_APERTURE_DAC / np.max(np.abs(step)))
        move_aperture(monitor, tuple(int(round(v)) for v in position + step))
    else:
        raise RuntimeError(f"the aperture was not centred after {MOST_STEPS} steps")

    shift = np.hypot(*shift_through_crossover(monitor, cl3))
    verdict = "under" if shift < E21_PX else "not under"
    print(f"E21: the beam moves {shift:.1f} px through the crossover, {verdict} {E21_PX} px")
    print(f"aperture now {tuple(int(v) for v in read_value(monitor, 'apertures.position'))}")
except BaseException:
    print("every value that a script may change -> its value at the start")
    try:
        leave_mode(monitor, MAG, mode_values)
    finally:
        restore(monitor, start)
    raise
print("every value but the aperture position -> its value at the start")
# The restore runs even if a step before it fails.
try:
    leave_mode(monitor, MAG, mode_values)
finally:
    restore(monitor, {key: value for key, value in start.items() if key != "apertures.position"})
