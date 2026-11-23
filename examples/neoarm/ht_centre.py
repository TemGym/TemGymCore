"""Set the HT centre with the beam tilt CLA2, from the alignment guide (S8, Q19, example E24).

The guide has this goal. With the HT wobbler on, the image breathes about the middle of the screen
and does not swing to the side. When the high tension changes, every lens changes its strength. A
beam with a tilt against the objective's axis then lands elsewhere. The image swings from side to
side with the wobbler. The beam tilt CLA2 sets where it swings least.

The script works at MAG X30k. CL3 spreads the beam over a specimen with features. On the simulated
NEOARM, the specimen is a cross grating in focus. The script switches the HT wobbler on and takes
FRAMES frames. The movement of the specimen's features between them (cross-correlation) gives the
size of the swing. A frame does not record on which side of the wobble the camera took it. For this
reason, the script uses only the size of the swing and does not use its direction. The swing squared
is a quadratic function of CLA2. Six trial values of CLA2 give this function. Then CLA2 goes, in
limited steps, to where the function is least. The script repeats this until the image swings less
than SWING_PX. It also stops when a step does not lower the swing. In that case, the rest of the
swing comes from something the beam tilt cannot change, and the script says so.

The script changes the function mode, the selector, CL3, CLA2 and the HT wobbler. It keeps the new
CLA2 at the end, because that is its purpose. Every other value goes back to its value at the
start (the HT wobbler off, CL3 from below). After an error or Ctrl-C, every value goes back, CLA2
too. It saves nothing.

A rehearsal on the simulated NEOARM runs on the CPU. From this folder:
JAX_PLATFORMS=cpu python ht_centre.py
"""

import time

import numpy as np

SIMULATED = True  # False at the microscope

if SIMULATED:
    from simulated_neoarm import CROSS_GRATING, install_simulated_neoarm

    # The operator's specimen is a cross grating. The operator focuses it with stage Z.
    install_simulated_neoarm(specimen=CROSS_GRATING)
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
    check_binning,
    enter_mode,
    feature_shift,
    leave_mode,
)

adapter = PyJEMAdapter(
    connect=not SIMULATED, tem3_module=TEM3, detector_module=detector, sightx_module=sightx
)
# jemwrap spaces the calls to the real NEOARM by 0.1 s (requirement R16). The simulated NEOARM
# needs no spacing, so a rehearsal spaces its calls by 1 ms.
monitor = MicroscopeMonitor(adapter=adapter, rate_hz=1000.0 if SIMULATED else 10.0)

# At MAG X30k, the grating's period is 35 px on the SightX. feature_shift mistakes a swing of more
# than half of the period (17 px) for a smaller one. On the simulated NEOARM, the swing starts at
# 7 px.
MAG, ROW = 0, 12
# CL3 spreads the beam over the SightX this far above its value, approached from below. On the
# simulated NEOARM 3000 units spread it over the whole frame (2026-11-22). The value is a choice.
CL3_SPREAD_DAC = 3000
# The number of frames for one measurement of the swing. The simulated wobbler changes side at
# every frame. The real wobbler swings continuously, and more frames catch more of its swing. The
# value is a choice.
FRAMES = 6
# The trial move of CLA2 and its largest move at once. On the simulated NEOARM with the wobbler at
# 100 V, 1000 units of CLA2 changed the swing by about 12 px (2026-11-22). Both are choices.
TRIAL_CLA2_DAC = 500
LARGEST_CLA2_DAC = 1000
# The HT centre is right when the image swings less than this (E24 asks for under 2 px). The value
# is a choice.
SWING_PX = 1.0
MOST_STEPS = 10
# A step of CLA2 that lowers the swing by less than this fraction ends the search. In that case,
# the rest of the swing is beyond the beam tilt. The principal said on 2026-11-22: "Maybe 6 pixels
# is an ok error". The value is a choice.
SMALLEST_GAIN = 0.1
# The script waits for the column to settle. The values are choices. R15 gives 2 s after a change of
# mode.
SETTLE_S = 0.2
SETTLE_LENS_S = 0.5
SETTLE_MODE_S = 2.0
SETTLE_ROW_S = 1.0
# The simulated NEOARM has no time to settle, so a rehearsal does not wait.
if SIMULATED:
    SETTLE_S = SETTLE_LENS_S = SETTLE_MODE_S = SETTLE_ROW_S = 0.0

# The trial moves of CLA2, in units of TRIAL_CLA2_DAC. Six values fix a quadratic function of two
# variables.
TRIALS = np.array([(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1)])


def swing(monitor):
    """Return how far the image swings with the HT wobbler, in px.

    The features' positions in FRAMES frames, relative to the first frame, lie along one line.
    The swing is twice their standard deviation along that line. For frames that alternate
    between the two sides of the wobble, this value is the distance between the sides.
    """
    frames = [take_frame(monitor) for _ in range(FRAMES)]
    positions = np.array([feature_shift(frames[0], frame) for frame in frames])
    largest_variance = np.linalg.eigvalsh(np.cov(positions.T, bias=True))[-1]
    return 2 * np.sqrt(largest_variance)


def set_cla2(monitor, value):
    """Set CLA2 to ``value`` (x, y)."""
    print(f"CLA2 -> {tuple(int(v) for v in value)}")
    set_value(monitor, "deflectors.CLA2", tuple(int(v) for v in value))
    time.sleep(SETTLE_S)


def cl3_from_below(monitor, cl3_dac):
    """Take CL3 to APPROACH_DAC below ``cl3_dac``, its lens value at the start.

    The restore then brings CL3 back from below. The predictor assumes this history (Q22). A
    direct drop from the spread beam would bring CL3 from above.
    """
    below = int(cl3_dac) - APPROACH_DAC
    print(f"CL3 -> {below}")
    set_value(monitor, "lens.CL3", below)


def least_swing(cla2, swings):
    """Return the CLA2 where the quadratic through the six trial swings squared is least.

    The swing is the length of a vector that changes linearly with CLA2. Its square is
    d^T M d + g^T d + c in the trial move d. A quadratic without a least value (M not positive)
    shows that the measurements have too much noise. In that case, the function returns the best
    trial value.
    """
    d = TRIALS * TRIAL_CLA2_DAC
    columns = np.column_stack(
        [d[:, 0] ** 2, d[:, 0] * d[:, 1], d[:, 1] ** 2, d[:, 0], d[:, 1], np.ones(len(d))]
    )
    a, b, c, gx, gy, _ = np.linalg.solve(columns, np.square(swings))
    curvature = np.array([[2 * a, b], [b, 2 * c]])
    if np.any(np.linalg.eigvalsh(curvature) <= 0):
        return cla2 + d[np.argmin(swings)]
    step = np.linalg.solve(curvature, -np.array([gx, gy]))
    step *= min(1.0, LARGEST_CLA2_DAC / np.max(np.abs(step)))
    return cla2 + step


# Example E24 asks for a swing under this, or for the rest to be named as beyond CLA2.
E24_PX = 2

# CHANGES holds the values that this script changes, for the restore at the end (Q20 of
# design/requirements.md). The script does not read or set back a value that it does not touch.
CHANGES = (
    "eos.function_mode",
    "eos.current_mag_selector_id",
    "lens.CL3",
    "deflectors.CLA2",
    "gun.ht_wobbler",
)
check_binning(monitor)
start = settable_snapshot(monitor, CHANGES)
# The lens value of CL3 at the start. The snapshot holds the free lens control value, which is not
# the lens value while free lens control is off.
cl3_at_start = int(read_value(monitor, "lens.CL3"))
mode_values = None
try:
    print("function mode -> MAG")
    mode_values = enter_mode(monitor, MAG, SETTLE_MODE_S)
    print(f"selector -> {ROW}")
    set_value(monitor, "eos.current_mag_selector_id", ROW)
    time.sleep(SETTLE_ROW_S)
    spread = int(read_value(monitor, "lens.CL3")) + CL3_SPREAD_DAC
    print(f"CL3 -> {spread - APPROACH_DAC} -> {spread}")
    set_value(monitor, "lens.CL3", spread - APPROACH_DAC)
    set_value(monitor, "lens.CL3", spread)
    time.sleep(SETTLE_LENS_S)
    print("HT wobbler -> on")
    set_value(monitor, "gun.ht_wobbler", True)

    cla2 = np.array(read_value(monitor, "deflectors.CLA2"), dtype=float)
    best = (np.inf, cla2)
    for _ in range(MOST_STEPS):
        swings = []
        for trial in TRIALS:
            set_cla2(monitor, cla2 + trial * TRIAL_CLA2_DAC)
            swings.append(swing(monitor))
        print(f"CLA2 at {tuple(int(v) for v in cla2)}: the image swings {swings[0]:.1f} px")
        if swings[0] < SWING_PX:
            best = (swings[0], cla2)
            break
        if swings[0] > (1 - SMALLEST_GAIN) * best[0]:
            # CLA2 does not lower the swing. The rest of the swing is beyond the beam tilt.
            break
        best = (swings[0], cla2)
        cla2 = np.round(least_swing(cla2, np.array(swings)))
    else:
        raise RuntimeError(f"the swing still fell after {MOST_STEPS} steps")

    cla2 = best[1]
    set_cla2(monitor, cla2)
    final = swing(monitor)
    if final < E24_PX:
        print(f"E24: the HT wobbler swings the image {final:.1f} px, under {E24_PX} px")
    else:
        print(
            f"E24: CLA2 at its least swing; the HT wobbler still swings the image {final:.1f} px, "
            "which the beam tilt cannot take away (on the simulated NEOARM: CL1 and CL3 off the "
            "axis, for the gun and spot alignments)"
        )
except BaseException:
    print("every value that the script changed -> its value at the start")
    try:
        cl3_from_below(monitor, cl3_at_start)
        leave_mode(monitor, MAG, mode_values)
    finally:
        restore(monitor, start)
    raise
print("every value but CLA2 -> its value at the start")
# The restore runs even if a step before it fails.
try:
    cl3_from_below(monitor, cl3_at_start)
    leave_mode(monitor, MAG, mode_values)
finally:
    restore(monitor, {key: value for key, value in start.items() if key != "deflectors.CLA2"})
