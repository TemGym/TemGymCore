"""Record how the projector lenses, CLA1, CLA2 and SpotA move the beam on the SightX.

The script needs no specimen. It goes through every MAG row from X8000 to X500k and every DIFF
camera length from 8 cm to 2 m. At each row it brings IL1, IL2, IL3 and PL1 to the row's values
from below. In MAG it then sets CL3 so that the beam has an rms radius of 20 to 60 px. It kicks
the beam with CLA1 (MAG) or CLA2 (DIFF) along +x, -x, +y and -y. At some rows it does the same with
one projector lens moved up or down, or with several lenses moved at once. At MAG 40k it does the
same at each spot size, and also kicks SpotA.

The script changes the function mode (MAG and DIFF only), the selector, the spot size, CL3,
IL1, IL2, IL3, PL1, CLA1, CLA2 and SpotA. It saves each frame to its own .npy file as it takes the
frame, and one line for each frame in table.csv, in output/record_lens_changes/<time>/. A line
holds what the script did and every lens and deflector value that the microscope had. For a
kicked frame, the columns of the kicked deflector hold its value before the kick, and kick_x and
kick_y hold the kick. At the end,
also after an error or Ctrl-C, it sets every value that a script may change back to its value at
the start.

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
# The spot sizes recorded at X40k, as PyJEM's indices (spot sizes 1 to 5).
SPOT_ROW = 13
SPOTS = (0, 1, 2, 3, 4)

# A lens comes to each value from this many DAC units below, so that it has the same magnetic
# history each time (the principal's rule).
APPROACH_DAC = 2000
# A kick moves the beam about this far. 60 px keeps all four kicked beams on the SightX.
KICK_PX = 60
# The trial kicks, smallest first. At X500k a kick of 1 DAC unit moves the beam about 2.5 px on the
# real NEOARM (recording of 2026-11-10). It moves the beam about 8 px on the simulated NEOARM
# (rehearsal of 2026-11-19). The script uses the first trial that moves the beam at least
# SMALLEST_TRIAL_PX. A kick is never smaller than the first trial.
TRIAL_KICKS_DAC = (5, 50, 500)
SMALLEST_TRIAL_PX = 5
LARGEST_KICK_DAC = 12000
# In MAG, CL3 brings the beam's rms radius into this range, aiming at BEAM_AIM_PX. The range is a
# choice. A beam much smaller saturates the camera. A beam much larger touches the edge.
BEAM_RMS_PX = (20, 60)
BEAM_AIM_PX = 40
FIRST_CL3_STEP_DAC = 500
LARGEST_CL3_STEP_DAC = 3000
SIZING_STEPS = 8
# Waits for the column to settle. The wait after a change between MAG and DIFF comes from the
# principal (requirement R15). The other waits are choices. No measurement gives them.
SETTLE_MODE_S = 2.0
SETTLE_ROW_S = 1.0
SETTLE_LENS_S = 0.5
SETTLE_DEFLECTOR_S = 0.2
# The simulated NEOARM has no time to settle, so a rehearsal does not wait.
if SIMULATED:
    SETTLE_MODE_S = SETTLE_ROW_S = SETTLE_LENS_S = SETTLE_DEFLECTOR_S = 0.0

# The values read for each frame. The script cannot change CM, the objective or PLA, but the fit
# needs them. For each lens, the table has the lens value and the free lens control value. What
# the lens value reports under free lens control on the real NEOARM is not known.
EOS_KEYS = ("function_mode", "current_mag_selector_id", "mag", "spot_size", "alpha_selector")
LENSES_READ = ("CL1", "CL3", "CM", "OLc", "OLf", "IL1", "IL2", "IL3", "PL1")
FREE_LENSES = ("CL1", "CL3", "IL1", "IL2", "IL3", "PL1")
DEFLECTORS_READ = ("SpotA", "CLA1", "CLA2", "ShiftBal", "TiltBal", "PLA")
# The frame size at SightX binning 4: 1916 px / 4.
FRAME_SHAPE = (479, 479)
FRAME_FIELDS = ["frame", "setting", "role", "deflector", "kick_x", "kick_y"]
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


def find_beam(frame):
    """Return the beam's centre (x, y) from the middle of the frame and its rms radius, in px.

    The rms radius is along one axis, as in the recordings of 2026-11-10. The beam is every pixel
    above the background by more than a fifth of the brightest pixel. A lower limit lets the noise
    of the background into the sums. The median of the frame is the background, because the beam
    covers less than half of the frame.
    """
    signal = frame.astype(float) - np.median(frame)
    beam = np.where(signal > 0.2 * signal.max(), signal, 0.0)
    rows, columns = np.indices(frame.shape)
    total = beam.sum()
    x = (columns * beam).sum() / total
    y = (rows * beam).sum() / total
    rms = np.sqrt((((columns - x) ** 2 + (rows - y) ** 2) * beam).sum() / total / 2)
    return x - frame.shape[1] / 2, y - frame.shape[0] / 2, rms


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
    line.update(zip(BEAM_FIELDS, np.round(beam, 2)))
    table.writerow(line | state)
    table_file.flush()
    return beam


def set_lens_from_below(monitor, name, dac):
    """Set one lens under free lens control to ``dac``, from APPROACH_DAC units below."""
    print(f"{name} -> {dac - APPROACH_DAC} -> {dac}")
    set_value(monitor, f"lens.{name}", dac - APPROACH_DAC)
    set_value(monitor, f"lens.{name}", dac)
    time.sleep(SETTLE_LENS_S)


def go_to_row(monitor, mode, selector, spot):
    """Bring the microscope to one console row and spot size, with IL1 to PL1 from below.

    Before the change, CL3 and the projector lenses leave free lens control, so that they take the
    console's values for the new row.
    """
    print("CL3, IL1, IL2, IL3, PL1 -> the console's values")
    # restore switches off free lens control for a lens whose snapshot says "off".
    restore(monitor, {key: (False, start[key][1]) for key in ("lens.CL3",) + PROJECTOR_KEYS})
    if read_value(monitor, "eos.function_mode") != mode:
        print(f"function mode -> {MODE_NAMES[mode]}")
        set_value(monitor, "eos.function_mode", mode)
        time.sleep(SETTLE_MODE_S)
    print(f"spot size -> {spot}, selector -> {selector}")
    set_value(monitor, "eos.spot_size", spot)
    set_value(monitor, "eos.current_mag_selector_id", selector)
    time.sleep(SETTLE_ROW_S)
    for name in PROJECTOR_LENSES:
        set_lens_from_below(monitor, name, int(read_value(monitor, f"lens.{name}")))


def size_beam(monitor, setting):
    """Change CL3 until the beam's rms radius is in BEAM_RMS_PX, in at most SIZING_STEPS steps.

    Each step assumes that the radius changes in proportion to CL3, with the slope of the last two
    frames. Where the slope is too small to use, the step doubles.
    """
    cl3 = int(read_value(monitor, "lens.CL3"))
    *_, rms = record_frame(monitor, read_state(monitor), setting, "sizing")
    step = FIRST_CL3_STEP_DAC
    for _ in range(SIZING_STEPS):
        if BEAM_RMS_PX[0] <= rms <= BEAM_RMS_PX[1]:
            return
        step = int(np.clip(cl3 + step, APPROACH_DAC, DAC_MAX)) - cl3
        set_lens_from_below(monitor, "CL3", cl3 + step)
        *_, new_rms = record_frame(monitor, read_state(monitor), setting, "sizing")
        slope = (new_rms - rms) / step
        cl3, rms = cl3 + step, new_rms
        # A slope below 0.001 px per DAC unit, or a beam off the camera, gives no usable step.
        if np.isfinite(slope) and abs(slope) > 1e-3:
            step = (BEAM_AIM_PX - rms) / slope
        else:
            step = 2 * step
        step = int(np.clip(step, -LARGEST_CL3_STEP_DAC, LARGEST_CL3_STEP_DAC)) or FIRST_CL3_STEP_DAC
    print(f"the beam's rms radius is {rms:.0f} px after {SIZING_STEPS} steps of CL3")


def choose_kick(monitor, state, setting, deflector):
    """Return the kick in DAC units that moves the beam about KICK_PX, from trial kicks along x."""
    x0, y0 = state[f"{deflector}_x"], state[f"{deflector}_y"]
    start_x, start_y, _ = record_frame(monitor, state, setting, "trial", deflector)
    for trial in TRIAL_KICKS_DAC:
        print(f"{deflector} -> {(x0 + trial, y0)}")
        set_value(monitor, f"deflectors.{deflector}", (x0 + trial, y0))
        time.sleep(SETTLE_DEFLECTOR_S)
        x, y, _ = record_frame(monitor, state, setting, "trial", deflector, (trial, 0))
        moved_px = np.hypot(x - start_x, y - start_y)
        if moved_px >= SMALLEST_TRIAL_PX:
            break
    set_value(monitor, f"deflectors.{deflector}", (x0, y0))
    # The four kicked values must stay inside the DAC range.
    room = min(x0, y0, DAC_MAX - x0, DAC_MAX - y0, LARGEST_KICK_DAC)
    return (
        int(np.clip(KICK_PX * trial / moved_px, TRIAL_KICKS_DAC[0], room))
        if np.isfinite(moved_px)
        else trial
    )


def record_block(monitor, state, setting, deflector, kick):
    """Take five frames. The first has no kick. The others have ``kick`` DAC units along +x, -x, +y, -y."""
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


frame_shape = take_frame(monitor).shape
if frame_shape != FRAME_SHAPE:
    raise SystemExit(
        f"set the SightX binning to 4: the frames are {frame_shape}, not {FRAME_SHAPE}"
    )

# Every value that a script may change, for the restore at the end.
start = settable_snapshot(monitor)
start_spot = int(start["eos.spot_size"])
start_mode = int(start["eos.function_mode"])
other_mode = DIFF if start_mode == MAG else MAG
OUTPUT.mkdir(parents=True)
saved_frames = []
table_file = open(OUTPUT / "table.csv", "w", newline="")
table = csv.DictWriter(table_file, FRAME_FIELDS + BEAM_FIELDS + list(read_state(monitor)))
table.writeheader()
try:
    for mode in (other_mode, start_mode):
        deflector = "CLA1" if mode == MAG else "CLA2"
        if mode == other_mode:
            print(f"function mode -> {MODE_NAMES[mode]}")
            set_value(monitor, "eos.function_mode", mode)
            time.sleep(SETTLE_MODE_S)
            other_mode_selector = int(read_value(monitor, "eos.current_mag_selector_id"))
        for selector in ROWS[mode]:
            go_to_row(monitor, mode, selector, start_spot)
            row = f"{MODE_NAMES[mode]} row {selector}"
            print(f"{row}: {read_value(monitor, 'eos.mag')}")
            if mode == MAG:
                size_beam(monitor, row)
            state = read_state(monitor)
            kick = choose_kick(monitor, state, row, deflector)
            record_block(monitor, state, row, deflector, kick)
            row_dac = {name: state[f"{name}_flc_dac"] for name in PROJECTOR_LENSES}

            if selector in LENS_CHANGE_ROWS[mode]:
                for name in PROJECTOR_LENSES:
                    for step in LENS_STEPS_DAC:
                        set_lens_from_below(monitor, name, row_dac[name] + step)
                        setting = f"{row}, {name} {step:+d}"
                        record_block(monitor, read_state(monitor), setting, deflector, kick)
                    set_lens_from_below(monitor, name, row_dac[name])

            if selector == COMBINED_ROWS[mode]:
                for number, changes in enumerate(COMBINED_CHANGES_DAC):
                    for name, step in changes.items():
                        set_lens_from_below(monitor, name, row_dac[name] + step)
                    setting = f"{row}, " + ", ".join(f"{n} {s:+d}" for n, s in changes.items())
                    record_block(monitor, read_state(monitor), setting, deflector, kick)
                    for name in changes:
                        set_lens_from_below(monitor, name, row_dac[name])

            if mode == MAG and selector == SPOT_ROW:
                for spot in SPOTS:
                    go_to_row(monitor, MAG, SPOT_ROW, spot)
                    setting = f"{row}, spot {spot}"
                    size_beam(monitor, setting)
                    state = read_state(monitor)
                    for kicked in ("CLA1", "SpotA"):
                        spot_kick = choose_kick(monitor, state, setting, kicked)
                        record_block(monitor, state, setting, kicked, spot_kick)
        if mode == other_mode:
            print(f"selector -> {other_mode_selector}, the {MODE_NAMES[mode]} row at the start")
            set_value(monitor, "eos.current_mag_selector_id", other_mode_selector)
finally:
    print("every value that a script may change -> its value at the start")
    restore(monitor, start)
    table_file.close()

print(f"saved {len(saved_frames)} frames and table.csv to {OUTPUT}")
