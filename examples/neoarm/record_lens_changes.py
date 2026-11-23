"""Record how the projector lenses, the objective focus and the deflectors move the beam.

The script needs no specimen. It goes through every MAG row from X8000 to X500k and every DIFF
camera length from 8 cm to 2 m. At each row it brings IL1, IL2, IL3 and PL1 to the row's values from
below. It centres the beam with CLA1 (MAG) or the projector deflector PLA (DIFF), and it keeps that
value for the next row.
In MAG it then sets CL3 so that the beam has an rms radius of 20 to 60 px. It kicks the beam with
CLA1 (MAG) or CLA2 (DIFF) along +x, -x, +y and -y. Before each set of kicks it centres the beam
again. The beam can be off the SightX at a new row or before a centring. Then the script looks for
the beam, as an operator would. It walks the centring deflector out on a spiral. When a trial kick
loses the beam, the script tries the kick the other way. For a y trial, it then also tries a smaller
kick. If none of this finds the beam, the script stops and sets the microscope back. The table keeps
a kicked or sizing frame that loses the beam, with no centre. At some rows, the script does the same
with one projector lens moved up or down. It also does the same with several lenses moved at once,
and with the objective's fine focus OLf moved up or down. At MAG 40k, it does the same at each spot
size. There it also kicks SpotA where SpotA moves the beam. At a few rows, it brings each of CL1,
CL3, IL1, IL2, IL3 and PL1 to the row's value from above, one at a time. Then it kicks the beam
again. The jump between the two approaches shows each lens's lag.

The script changes the function mode (MAG and DIFF only), the selector and the spot size. It also
changes CL1, CL3, OLf, IL1, IL2, IL3, PL1, CLA1, CLA2, PLA and SpotA. It saves each frame to its own
.npy file when it takes the frame. It also writes one line for each frame in table.csv. Both go to
output/record_lens_changes/<time>/. A line holds what the script did and every lens and deflector
value that the microscope had. For a kicked frame, the columns of the kicked deflector hold its
value before the kick, and kick_x and kick_y hold the kick. At the end, also after an error or
Ctrl-C, it sets every value that a script may change back to its value at the start.

Each function mode recalls its own last row. The snapshot for the restore holds the row of the
start mode only. So the script records the other mode first, reads that mode's row when it
switches to it, and selects that row again before it leaves. After an interrupt in the other
mode, that mode keeps the row of the interrupt.

The SightX must be at binning 4 before the start, as in the recordings of 2026-11-10. The sizes in
pixels below are for binning 4, and jemwrap cannot change the binning. The script stops before
its first change if the frames have another size.

All calls to the microscope go through jemwrap, which spaces them by at least 0.1 s and refuses
values that a script must not change.

A rehearsal on the simulated NEOARM runs on the CPU, because the GPU is slower for these small
frames. From this folder: JAX_PLATFORMS=cpu python record_lens_changes.py
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
from microscope_functions import (  # noqa: E402
    APPROACH_DAC,
    check_binning,
    enter_mode,
    find_beam,
    leave_mode,
)
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

# PyJEM's numbers for the two function modes.
MAG, DIFF = 0, 4
# The largest DAC value of a lens or a deflector.
DAC_MAX = 65535
MODE_NAMES = {MAG: "MAG", DIFF: "DIFF"}
# Selector rows, from the console tables of 2026-11-17. MAG rows 6 to 24 are X8000 to X500k. DIFF
# rows 0 to 14 are 8 cm to 2 m. Together they are the range of requirement R22.
ROWS = {MAG: range(6, 25), DIFF: range(0, 15)}
# The rows where each projector lens moves are X8000, X15k, X40k, X100k, X200k and X500k. In DIFF
# they are 8 cm, 20 cm, 40 cm, 1 m and 1.5 m. They include both ends of each range. DIFF 2 m is not
# one of them, because IL2 is at 65520 there and cannot go 1000 units up.
LENS_CHANGE_ROWS = {MAG: (6, 9, 13, 17, 20, 24), DIFF: (0, 4, 7, 11, 13)}
PROJECTOR_LENSES = ("IL1", "IL2", "IL3", "PL1")
PROJECTOR_KEYS = tuple(f"lens.{name}" for name in PROJECTOR_LENSES)
# Each projector lens moves by these DAC values. 1000 units changed the magnification at MAG 40k
# by 2 to 20 % (recording of 2026-11-10).
LENS_STEPS_DAC = (-1000, 1000)
# Settings with several lenses moved at once, at X40k and at 40 cm, for acceptance example E3.
COMBINED_ROWS = {MAG: 13, DIFF: 7}
COMBINED_CHANGES_DAC = ({"IL1": 1000, "IL3": -1000}, {"IL2": -1000, "IL3": 1000, "PL1": -1000})
# The rows of the focus series are X15k, X40k and X100k and 20 cm, 40 cm and 1 m. FOCUS_STEPS_DAC
# gives the steps of the objective's fine focus OLf. At about 1.7 nm per unit, 3000 units are about
# 5 um of focus. At X200k the rehearsal moved the beam 135 px for 3000 units.
FOCUS_ROWS = {MAG: (9, 13, 17), DIFF: (4, 7, 11)}
FOCUS_STEPS_DAC = (-3000, -1000, 1000, 3000)
# The spot sizes recorded at X40k, as PyJEM's indices (spot sizes 1 to 5).
SPOT_ROW = 13
SPOTS = (0, 1, 2, 3, 4)

# The rows where each lens also comes to its value from above: X8000, X40k and X500k, and 8 cm,
# 40 cm and 1.5 m. The jump between the two approaches is twice the lens's lag (stage 10's way of
# measuring the lag, design/lens_hysteresis.md). A lens near DAC_MAX comes from DAC_MAX.
ABOVE_ROWS = {MAG: (6, 13, 24), DIFF: (0, 7, 13)}
ABOVE_LENSES = ("CL1", "CL3", *PROJECTOR_LENSES)
# A kick moves the beam about this far. 60 px keeps all four kicked beams on the SightX.
KICK_PX = 60
# The trial kicks, smallest first. At X500k a kick of 1 DAC unit moves the beam about 2.5 px on the
# real NEOARM (recording of 2026-11-10). Near the condenser focus, 500 units of SpotA moved the beam
# of the simulated NEOARM less than 5 px (rehearsal of 2026-11-20). The script uses the first trial
# that moves the beam at least SMALLEST_TRIAL_PX. A kick is never smaller than the first trial.
# At DIFF 8 cm on the simulated NEOARM, 5000 units of CLA2 moved the spot about 2 px (2026-11-22).
TRIAL_KICKS_DAC = (5, 50, 500, 5000, 20000)
SMALLEST_TRIAL_PX = 5
LARGEST_KICK_DAC = 12000
# The script looks for a beam off the SightX as an operator would. The centring deflector walks
# out from its value on a square spiral of SEARCH_RINGS rings, with a frame at each point. It stops
# when a frame shows the whole beam. Each step moves the beam about SEARCH_STEP_PX. The script
# calculates the step from the deflector's strength at the row before. It scales this strength by
# the change of the magnification or camera length. With steps of a
# third of the frame, a beam of rms radius 60 px is wholly in the frame at one of the points. Six
# rings reach about 1000 px, two frames, in at most 168 frames. The step and the rings are choices.
SEARCH_STEP_PX = 160
SEARCH_RINGS = 6
# The script centres the beam with CLA1 (MAG) or PLA (DIFF) where it is farther from the middle
# than this. The beam then stays on the SightX when the next row magnifies its offset. The value
# carries over from row to row, as on the recording of 2026-11-10.
CENTRED_PX = 20
# In MAG, CL3 brings the beam's rms radius into this range. The range is a choice. A beam much
# smaller saturates the camera. A beam much larger touches the edge. The real NEOARM needed CL3
# up to 8600 units above the console's value at X8000 (recording of 2026-11-10). Doubling steps
# from 500 reach that in five steps.
BEAM_RMS_PX = (20, 60)
FIRST_CL3_STEP_DAC = 500
SIZING_STEPS = 10
# Waits for the column to settle. The wait after a change between MAG and DIFF comes from the
# principal (requirement R15). The other waits are choices. No measurement gives them.
SETTLE_MODE_S = 2.0
SETTLE_ROW_S = 1.0
SETTLE_LENS_S = 0.5
SETTLE_DEFLECTOR_S = 0.2
# The simulated NEOARM has no time to settle, so a rehearsal does not wait.
if SIMULATED:
    SETTLE_MODE_S = SETTLE_ROW_S = SETTLE_LENS_S = SETTLE_DEFLECTOR_S = 0.0

# The values read for each frame. The script cannot change CM or the objective, but the fit needs
# them. For each lens, the table has the lens value and the free lens control value. What
# the lens value reports under free lens control on the real NEOARM is not known.
EOS_KEYS = ("function_mode", "current_mag_selector_id", "mag", "spot_size", "alpha_selector")
LENSES_READ = ("CL1", "CL3", "CM", "OLc", "OLf", "IL1", "IL2", "IL3", "PL1")
FREE_LENSES = ("CL1", "CL3", "OLf", "IL1", "IL2", "IL3", "PL1")
DEFLECTORS_READ = ("SpotA", "CLA1", "CLA2", "ShiftBal", "TiltBal", "PLA")
FRAME_FIELDS = ["frame", "setting", "role", "deflector", "kick_x", "kick_y", "from_above"]
BEAM_FIELDS = ["beam_x_px", "beam_y_px", "beam_rms_px"]

OUTPUT = (
    Path(__file__).parent
    / "output/record_lens_changes"
    / datetime.now().strftime("%Y-%m-%d_%H%M%S")
)


# The functions below write to the table and the output folder that the steps open. go_to_row
# uses the start values.


def read_state(monitor):
    """Read every value of the table now, and return them as plain integers by column name."""
    state = {key: int(read_value(monitor, f"eos.{key}")) for key in EOS_KEYS}
    for name in LENSES_READ:
        state[name] = int(read_value(monitor, f"lens.{name}"))
    for name in FREE_LENSES:
        on, dac = read_value(monitor, f"lens.free_lens_control.{name}")
        state[f"{name}_flc_on"], state[f"{name}_flc_dac"] = int(on), int(dac)
    for name in DEFLECTORS_READ:
        x, y = read_value(monitor, f"deflectors.{name}")
        state[f"{name}_x"], state[f"{name}_y"] = int(x), int(y)
    return state


def record_frame(monitor, state, setting, role, deflector="", kick=(0, 0)):
    """Take one frame, save it with one line of the table, and return the beam as find_beam does.

    ``kick`` is the DAC change of ``deflector`` from its value in ``state``.
    """
    frame = take_frame(monitor)
    name = f"{len(saved_frames):04d}.npy"
    np.save(OUTPUT / name, frame)
    saved_frames.append(name)
    beam = find_beam(frame)
    line = dict(
        frame=name, setting=setting, role=role, deflector=deflector, kick_x=kick[0], kick_y=kick[1]
    )
    # The lenses that came to their values from above, as "CL3" or "", for the fit's lag.
    line["from_above"] = " ".join(sorted(lenses_from_above))
    # The table holds NaN for a beam that is too large to measure, as for no beam.
    line.update(zip(BEAM_FIELDS, np.round(np.where(np.isinf(beam), np.nan, beam), 2)))
    table.writerow(line | state)
    table_file.flush()
    return beam


def set_lens_from_below(monitor, name, dac):
    """Set one lens under free lens control to ``dac``, from APPROACH_DAC units below."""
    print(f"{name} -> {dac - APPROACH_DAC} -> {dac}")
    set_value(monitor, f"lens.{name}", dac - APPROACH_DAC)
    set_value(monitor, f"lens.{name}", dac)
    lenses_from_above.discard(name)
    time.sleep(SETTLE_LENS_S)


def set_lens_from_above(monitor, name, dac):
    """Set one lens under free lens control to ``dac``, from APPROACH_DAC units above or DAC_MAX."""
    above = min(dac + APPROACH_DAC, DAC_MAX)
    print(f"{name} -> {above} -> {dac}")
    set_value(monitor, f"lens.{name}", above)
    set_value(monitor, f"lens.{name}", dac)
    lenses_from_above.add(name)
    time.sleep(SETTLE_LENS_S)


def go_to_row(monitor, mode, selector, spot):
    """Bring the microscope to one console row and spot size, with every free lens from below.

    Before the change, CL1, CL3, OLf and the projector lenses leave free lens control, so that they
    take the console's values for the new row.
    """
    print("CL1, CL3, OLf, IL1, IL2, IL3, PL1 -> the console's values")
    # restore switches off free lens control for a lens whose snapshot says "off".
    released = ("lens.CL1", "lens.CL3", "lens.OLf") + PROJECTOR_KEYS
    restore(monitor, {key: (False, start[key][1]) for key in released})
    if read_value(monitor, "eos.function_mode") != mode:
        print(f"function mode -> {MODE_NAMES[mode]}")
        set_value(monitor, "eos.function_mode", mode)
        time.sleep(SETTLE_MODE_S)
    print(f"spot size -> {spot}, selector -> {selector}")
    set_value(monitor, "eos.spot_size", spot)
    set_value(monitor, "eos.current_mag_selector_id", selector)
    time.sleep(SETTLE_ROW_S)
    # Each lens that a row or a spot size changes comes to its value from below. Its iron has the
    # same history at every row (H3). Then its lag is the same each time.
    for name in ("CL1", "CL3", "OLf", *PROJECTOR_LENSES):
        set_lens_from_below(monitor, name, int(read_value(monitor, f"lens.{name}")))


def size_beam(monitor, setting, deflector, px_per_dac):
    """Raise CL3 from the console's value until the beam's rms radius is in BEAM_RMS_PX.

    The console's CL3 is near the condenser focus. Above the focus the beam grows with CL3, so the
    script stays above it, as the recording of 2026-11-10 did. A beam too large to measure counts
    as too large. Where a change of CL3 moves the beam off the SightX, ``deflector`` looks for it,
    with ``px_per_dac`` as in search_for_beam. The steps double, up from a beam that is too small
    and down from a beam that is too large. When the script has one value of each, each step goes
    halfway between the last two. CL3 first comes to the console's value from below, as every
    change of CL3 does, and it never goes below that value. A beam too large there stays as it is.
    """
    cl3 = lowest_cl3 = int(read_value(monitor, "lens.CL3"))
    set_lens_from_below(monitor, "CL3", cl3)
    *_, rms = search_for_beam(monitor, setting, "sizing", deflector, px_per_dac)
    too_small_cl3 = too_large_cl3 = None
    step = FIRST_CL3_STEP_DAC
    for _ in range(SIZING_STEPS):
        if BEAM_RMS_PX[0] <= rms <= BEAM_RMS_PX[1]:
            return
        if rms < BEAM_RMS_PX[0]:
            too_small_cl3 = cl3
        elif cl3 == lowest_cl3:
            print(f"{setting}: the beam is too large at the console's CL3, {cl3}")
            return
        else:
            too_large_cl3 = cl3
        if too_small_cl3 is not None and too_large_cl3 is not None:
            cl3 = (too_small_cl3 + too_large_cl3) // 2
        else:
            cl3 = cl3 + step if too_large_cl3 is None else cl3 - step
            step *= 2
        cl3 = int(np.clip(cl3, lowest_cl3, DAC_MAX))
        set_lens_from_below(monitor, "CL3", cl3)
        *_, rms = search_for_beam(monitor, setting, "sizing", deflector, px_per_dac)
    print(f"the beam's rms radius is {rms:.0f} px after {SIZING_STEPS} steps of CL3")


def search_for_beam(monitor, setting, role, deflector, px_per_dac):
    """Take a frame and, if it has no light, walk ``deflector`` on a spiral until one has a beam.

    ``px_per_dac`` is an estimate of how far one DAC unit of ``deflector`` moves the beam. The first
    frame has ``role``, and the frames of the spiral have the role "search". The deflector stays
    where the function found the beam. The function returns the beam as find_beam does. The result
    is a whole beam, or a beam too large to measure, which no search would help. If no frame has
    light, it sets the deflector back and stops the script.
    """
    state = read_state(monitor)
    x0, y0 = state[f"{deflector}_x"], state[f"{deflector}_y"]
    beam = record_frame(monitor, state, setting, role)
    if not np.isnan(beam[2]):
        return beam
    step_dac = SEARCH_STEP_PX / px_per_dac
    print(f"{setting}: the beam is off the SightX, {deflector} looks for it")
    for ring in range(1, SEARCH_RINGS + 1):
        for i in range(-ring, ring + 1):
            for j in range(-ring, ring + 1):
                if max(abs(i), abs(j)) != ring:
                    continue
                # The search moves the deflector by no more than a centring does.
                dx, dy = np.clip(
                    np.round([i * step_dac, j * step_dac]), -LARGEST_KICK_DAC, LARGEST_KICK_DAC
                )
                x = int(np.clip(x0 + dx, 0, DAC_MAX))
                y = int(np.clip(y0 + dy, 0, DAC_MAX))
                print(f"{deflector} -> {(x, y)}")
                set_value(monitor, f"deflectors.{deflector}", (x, y))
                time.sleep(SETTLE_DEFLECTOR_S)
                beam = record_frame(monitor, read_state(monitor), setting, "search")
                if not np.isnan(beam[2]):
                    return beam
    print(f"{deflector} -> {(x0, y0)}")
    set_value(monitor, f"deflectors.{deflector}", (x0, y0))
    raise RuntimeError(f"{setting}: the search did not find the beam")


def trial_kick(monitor, state, setting, deflector, kick):
    """Kick ``deflector`` by ``kick`` DAC units, and by the opposite kick if the beam leaves.

    The kick is from the value in ``state``, and the deflector goes back to that value after each
    frame. The function returns the kick that kept the whole beam on the SightX and the beam's
    centre. If both kicks lost the beam, it returns the last kick and NaN.
    """
    x0, y0 = state[f"{deflector}_x"], state[f"{deflector}_y"]
    for sign in (1, -1):
        signed = (sign * kick[0], sign * kick[1])
        print(f"{deflector} -> {(x0 + signed[0], y0 + signed[1])}")
        set_value(monitor, f"deflectors.{deflector}", (x0 + signed[0], y0 + signed[1]))
        time.sleep(SETTLE_DEFLECTOR_S)
        moved = np.array(record_frame(monitor, state, setting, "trial", deflector, signed)[:2])
        set_value(monitor, f"deflectors.{deflector}", (x0, y0))
        time.sleep(SETTLE_DEFLECTOR_S)
        if np.all(np.isfinite(moved)):
            break
    return signed, moved


def measure_response(monitor, state, setting, deflector):
    """Return how far the beam moves, in px per DAC unit, for a change of ``deflector``.

    The result is a 2 x 2 matrix. Its first column is the move (x, y) for a change along x, and its
    second column is the move for a change along y. Trial kicks give it. The x trial is the first
    that moves the beam SMALLEST_TRIAL_PX. The y trial starts at the same size. Where it loses the
    beam both ways, y is the stronger direction, and the next smaller trial follows.
    """
    start_px = np.array(record_frame(monitor, state, setting, "trial", deflector)[:2])
    if not np.all(np.isfinite(start_px)):
        raise RuntimeError(f"{setting}: the beam is not wholly on the SightX")
    # Only the trials that keep the deflector inside its DAC range.
    x0, y0 = state[f"{deflector}_x"], state[f"{deflector}_y"]
    room = min(x0, y0, DAC_MAX - x0, DAC_MAX - y0)
    trials = [t for t in TRIAL_KICKS_DAC if t <= room] or [TRIAL_KICKS_DAC[0]]
    for trial in trials:
        kick_x, moved_x = trial_kick(monitor, state, setting, deflector, (trial, 0))
        if not np.all(np.isfinite(moved_x)):
            raise RuntimeError(f"{setting}: {trial} DAC units of {deflector} lose the beam")
        if np.hypot(*(moved_x - start_px)) >= SMALLEST_TRIAL_PX:
            break
    for trial_y in [t for t in trials if t <= trial][::-1]:
        kick_y, moved_y = trial_kick(monitor, state, setting, deflector, (0, trial_y))
        if np.all(np.isfinite(moved_y)):
            break
    else:
        raise RuntimeError(f"{setting}: every y trial of {deflector} loses the beam")
    response = np.column_stack([(moved_x - start_px) / kick_x[0], (moved_y - start_px) / kick_y[1]])
    # The largest trial did not move the beam SMALLEST_TRIAL_PX. The response is mostly noise.
    if np.hypot(*response[:, 0]) * trial < SMALLEST_TRIAL_PX:
        raise RuntimeError(f"{setting}: {trial} DAC units of {deflector} hardly move the beam")
    return response


def centre_beam(monitor, setting, deflector, response):
    """Move ``deflector`` in one step, from ``response``, so that the beam goes to the middle.

    A beam off the SightX is first looked for with search_for_beam. A beam too large to measure is
    left where it is.
    """
    px_per_dac = np.mean(np.hypot(*response))
    centre_px = np.array(search_for_beam(monitor, setting, "centring", deflector, px_per_dac)[:2])
    if not np.all(np.isfinite(centre_px)):
        print(f"{setting}: the beam is too large to centre")
        return
    if np.hypot(*centre_px) <= CENTRED_PX:
        return
    x0, y0 = (int(v) for v in read_value(monitor, f"deflectors.{deflector}"))
    change = np.clip(np.linalg.solve(response, -centre_px), -LARGEST_KICK_DAC, LARGEST_KICK_DAC)
    change_x, change_y = np.rint(change).astype(int)
    centred = (int(np.clip(x0 + change_x, 0, DAC_MAX)), int(np.clip(y0 + change_y, 0, DAC_MAX)))
    print(f"{deflector} -> {centred}")
    set_value(monitor, f"deflectors.{deflector}", centred)
    time.sleep(SETTLE_DEFLECTOR_S)


def kick_for(state, deflector, response):
    """Return the kick in DAC units that moves the beam about KICK_PX, from ``response``."""
    x0, y0 = state[f"{deflector}_x"], state[f"{deflector}_y"]
    px_per_dac = np.mean(np.hypot(*response))
    # The four kicked values must stay inside the DAC range.
    room = min(x0, y0, DAC_MAX - x0, DAC_MAX - y0, LARGEST_KICK_DAC)
    return int(np.clip(KICK_PX / px_per_dac, TRIAL_KICKS_DAC[0], room))


def record_block(monitor, state, setting, deflector, kick):
    """Take five frames, one with no kick and four with ``kick`` DAC units along +x, -x, +y, -y."""
    x0, y0 = state[f"{deflector}_x"], state[f"{deflector}_y"]
    for role, change in (
        ("not deflected", (0, 0)),
        ("deflected +x", (kick, 0)),
        ("deflected -x", (-kick, 0)),
        ("deflected +y", (0, kick)),
        ("deflected -y", (0, -kick)),
    ):
        print(f"{deflector} -> {(x0 + change[0], y0 + change[1])}")
        set_value(monitor, f"deflectors.{deflector}", (x0 + change[0], y0 + change[1]))
        time.sleep(SETTLE_DEFLECTOR_S)
        record_frame(monitor, state, setting, role, deflector, change)
    set_value(monitor, f"deflectors.{deflector}", (x0, y0))
    time.sleep(SETTLE_DEFLECTOR_S)


check_binning(monitor)

# CHANGES holds the values that this script changes, for the restore at the end (Q20 of
# design/requirements.md). The script does not read or set back a value that it does not touch.
CHANGES = (
    "eos.function_mode",
    "eos.current_mag_selector_id",
    "eos.spot_size",
    "lens.CL1",
    "lens.CL3",
    "lens.OLf",
    "lens.IL1",
    "lens.IL2",
    "lens.IL3",
    "lens.PL1",
    "deflectors.SpotA",
    "deflectors.CLA1",
    "deflectors.CLA2",
    "deflectors.PLA",
)
start = settable_snapshot(monitor, CHANGES)
start_spot = int(start["eos.spot_size"])
start_mode = int(start["eos.function_mode"])
other_mode = DIFF if start_mode == MAG else MAG
OUTPUT.mkdir(parents=True)
saved_frames = []
lenses_from_above = set()
table_file = open(OUTPUT / "table.csv", "w", newline="")
table = csv.DictWriter(table_file, FRAME_FIELDS + BEAM_FIELDS + list(read_state(monitor)))
table.writeheader()
other_mode_values = None
try:
    for mode in (other_mode, start_mode):
        # The kicked deflector: the beam shift in MAG, the beam tilt in DIFF. The centring
        # deflector: the beam shift in MAG, the projector deflector PLA in DIFF, which moves a
        # diffraction pattern without changing the illumination.
        deflector = "CLA1" if mode == MAG else "CLA2"
        centring = "CLA1" if mode == MAG else "PLA"
        if mode == other_mode:
            print(f"function mode -> {MODE_NAMES[mode]}")
            other_mode_values = enter_mode(monitor, mode, SETTLE_MODE_S)
        # The strength of the deflector at the row before, in px per DAC unit, and that row's
        # magnification or camera length. The first row of each mode has none.
        row_before = None
        for selector in ROWS[mode]:
            go_to_row(monitor, mode, selector, start_spot)
            row = f"{MODE_NAMES[mode]} row {selector}"
            print(f"{row}: {read_value(monitor, 'eos.mag')}")
            if row_before is not None:
                px_per_dac, mag_before = row_before
                px_per_dac *= int(read_value(monitor, "eos.mag")) / mag_before
                *_, rms = search_for_beam(monitor, row, "check", centring, px_per_dac)
                # The script sizes a beam that is too large to measure before the trial kicks.
                if mode == MAG and np.isinf(rms):
                    size_beam(monitor, row, deflector, px_per_dac)
            state = read_state(monitor)
            response = measure_response(monitor, state, row, deflector)
            if mode == MAG:
                centring_response = response
            else:
                centring_response = measure_response(monitor, state, row, centring)
            row_before = (np.mean(np.hypot(*centring_response)), state["mag"])
            centre_beam(monitor, row, centring, centring_response)
            # CL3 is off the axis, so a change of CL3 moves the beam. The script centres it again.
            if mode == MAG:
                size_beam(monitor, row, deflector, row_before[0])
                centre_beam(monitor, row, centring, centring_response)
            state = read_state(monitor)
            kick = kick_for(state, deflector, response)
            record_block(monitor, state, row, deflector, kick)
            row_dac = {name: state[f"{name}_flc_dac"] for name in PROJECTOR_LENSES}

            if selector in LENS_CHANGE_ROWS[mode]:
                for name in PROJECTOR_LENSES:
                    for step in LENS_STEPS_DAC:
                        set_lens_from_below(monitor, name, row_dac[name] + step)
                        setting = f"{row}, {name} {step:+d}"
                        centre_beam(monitor, setting, centring, centring_response)
                        record_block(monitor, read_state(monitor), setting, deflector, kick)
                    set_lens_from_below(monitor, name, row_dac[name])

            if selector in ABOVE_ROWS[mode]:
                # The beam stays where it is, so that the not-deflected frame shows the jump.
                for name in ABOVE_LENSES:
                    set_lens_from_above(monitor, name, int(state[f"{name}_flc_dac"]))
                    setting = f"{row}, {name} from above"
                    record_block(monitor, read_state(monitor), setting, deflector, kick)
                    set_lens_from_below(monitor, name, int(state[f"{name}_flc_dac"]))

            if selector in FOCUS_ROWS[mode]:
                # Free lens control may be off, and then the lens reading holds the row's value.
                row_olf = state["OLf_flc_dac"] if state["OLf_flc_on"] else state["OLf"]
                row_deflector = (state[f"{deflector}_x"], state[f"{deflector}_y"])
                for step in FOCUS_STEPS_DAC:
                    # Each step starts from the row's deflector value. It does not start from the
                    # centring of the step before.
                    set_value(monitor, f"deflectors.{deflector}", row_deflector)
                    set_lens_from_below(monitor, "OLf", row_olf + step)
                    setting = f"{row}, OLf {step:+d}"
                    centre_beam(monitor, setting, centring, centring_response)
                    record_block(monitor, read_state(monitor), setting, deflector, kick)
                print("OLf -> the console's value")
                restore(monitor, {"lens.OLf": (False, start["lens.OLf"][1])})
                set_value(monitor, f"deflectors.{deflector}", row_deflector)

            if selector == COMBINED_ROWS[mode]:
                for number, changes in enumerate(COMBINED_CHANGES_DAC):
                    for name, step in changes.items():
                        set_lens_from_below(monitor, name, row_dac[name] + step)
                    setting = f"{row}, " + ", ".join(f"{n} {s:+d}" for n, s in changes.items())
                    centre_beam(monitor, setting, centring, centring_response)
                    record_block(monitor, read_state(monitor), setting, deflector, kick)
                    for name in changes:
                        set_lens_from_below(monitor, name, row_dac[name])

            if mode == MAG and selector == SPOT_ROW:
                for spot in SPOTS:
                    go_to_row(monitor, MAG, SPOT_ROW, spot)
                    setting = f"{row}, spot {spot}"
                    state = read_state(monitor)
                    response = measure_response(monitor, state, setting, "CLA1")
                    centre_beam(monitor, setting, "CLA1", response)
                    size_beam(monitor, setting, "CLA1", np.mean(np.hypot(*response)))
                    centre_beam(monitor, setting, "CLA1", response)
                    state = read_state(monitor)
                    record_block(monitor, state, setting, "CLA1", kick_for(state, "CLA1", response))
                    # At some spots, SpotA tilts the beam about a point near the specimen. Then it
                    # hardly moves the beam on the SightX. Where SpotA hardly moves the beam, or its
                    # trials lose it, the script leaves out its block at that spot. measure_response
                    # sets SpotA back first.
                    try:
                        response = measure_response(monitor, state, setting, "SpotA")
                    except RuntimeError as problem:
                        print(f"{problem}: no SpotA block at this spot")
                        continue
                    record_block(
                        monitor, state, setting, "SpotA", kick_for(state, "SpotA", response)
                    )
        if mode == other_mode:
            leave_mode(monitor, other_mode, other_mode_values)
finally:
    # After an error in the other mode, its selector and PLA go back too. The restore runs even
    # if that fails.
    try:
        leave_mode(monitor, other_mode, other_mode_values)
    finally:
        print("every value that a script may change -> its value at the start")
        restore(monitor, start)
        table_file.close()

print(f"saved {len(saved_frames)} frames and table.csv to {OUTPUT}")
