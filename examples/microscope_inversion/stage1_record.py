# %% [markdown]
# # Stage 1 at the NEOARM: how far a beam shift (CLA1) or tilt (CLA2) moves the beam on the camera, for different projection-lens settings (IL1, IL2, IL3, PL1)
#
# In MAG (40k) a beam shift (CLA1) moves the beam on the camera and a tilt hardly does; in DIFF (40 cm) it is the
# other way round. So MAG records CLA1, and DIFF records CLA2. Both modes go through the same steps:
#
# 1. read what the console has, and take over IL1-PL1 at those values (free-lens control);
# 2. find a deflection size that moves the beam a useful distance, in a straight line;
# 3. find how far to change each lens so that the deflection moves the beam noticeably differently;
# 4. record: at the console's lenses and at each lens up and down, the beam deflected + and - along x and y;
# 5. restore the microscope.
#
# Every saved frame gives one row of a table (`table.csv`): what the frame is for, the lens and deflector values read
# at that moment, and where the beam is. The last part of the notebook fits a lens model to that table.
#
# Run the cells one by one and check what each prints before running the next. The microscope is only touched
# through `ask` (read), `change` (set; in a dry run it is only printed) and `take_frame`, each followed by a short
# pause, because too many calls at once can crash the microscope. If a cell fails or is interrupted, the microscope
# is restored: free-lens control off, and CLA1, CLA2 and PLA back where the console had them.

# %% [markdown]
# ## Setup

# %%
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append("..")
from pyjem_session import find_beam, show_frame          # noqa: E402  -- image analysis only

FOR_REAL = False     # False: a dry run (reads the microscope and takes frames, changes nothing, prints each change)
SIMULATED = True     # True: the simulated NEOARM (needs TemGym); False: the real microscope
SCREEN = True        # show every frame with the beam marked (needs matplotlib)
DRY_RUN = not FOR_REAL

LENSES = ["IL1", "IL2", "IL3", "PL1"]                    # the lenses whose settings we vary
FREE_LENS_NUMBER = {"IL1": 10, "IL2": 11, "IL3": 12, "PL1": 14}   # PyJEM's numbers for free-lens control
ALL_LENSES = ["CL1", "CL2", "CL3", "CM", "OLc", "OLf", "OM", "OM2", "IL1", "IL2", "IL3", "IL4", "PL1", "PL2", "PL3"]
PAUSE_S = 0.1                                            # after every call to the microscope
DAC_MAX = 65535                                          # lenses and deflectors take values from 0 to 65535
MIN_ROUNDNESS = 0.9                                      # a deflected beam must stay this fraction as round as before

print("DRY RUN: nothing will be changed" if DRY_RUN else "FOR REAL: the microscope will be changed")

# %% [markdown]
# ## Connecting to the microscope

# %%
if SIMULATED:                                            # the simulated NEOARM, at 200 kV like the real one
    os.environ["JAX_PLATFORMS"] = "cpu"
    import temgym_core
    from temgym_core.pyjem_simulator import SimulatedMicroscope
    temgym_core.enable_x64()
    microscope = SimulatedMicroscope.from_file("../virtual_neoarm/virtual_neoarm.json", ht_v=200e3,
                                               diffraction_camera_length_m=0.4)     # the console is left on 40 cm in DIFF
    TEM3, detector = microscope.pyjem()
    camera = detector.Detector(detector.get_attached_detector()[0])
    lens_settle_s, mode_settle_s = 0, 0                  # the simulated microscope settles instantly
    folder = Path("output/stage1_simulated")
else:                                                    # the real NEOARM
    from PyJEM import TEM3
    from PyJEM import sightx                             # the SightX camera
    sightx.set_ip("172.17.41.1")                         # the camera PC
    camera = sightx.Detector("sightx")
    lens_settle_s, mode_settle_s = 2, 5                  # seconds to wait after changing a lens, and after switching mode
    folder = Path("output/stage1_neoarm")

lens, deflectors, eos = TEM3.Lens3(), TEM3.Def3(), TEM3.EOS3()

if SIMULATED:                                            # the simulated console starts at 8000x: set it to the 40k row
    while eos.GetMagValue()[0] < 40000:
        eos.UpSelector()

# %% [markdown]
# ## Talking to the microscope: `ask`, `change`, `take_frame`
#
# The state kept while recording: the table rows, what the console had in each mode, and the dark level of the camera.

# %%
rows = []                     # the table: one row per saved frame
modes = {}                    # what the console had in each mode: saved to modes.json
mode = ""                     # "MAG 40k" or "DIFF 40 cm", for the table
start_value = {}              # CLA1, CLA2 and PLA as the console had them in this mode: what restore_microscope() sets
dark_level = None             # the camera's value with the beam blanked (until measured, find_beam estimates it)
last_change = "(nothing yet)"     # for the caption of the next frame on the screen
frames_shown = 0


def ask(call, *arguments):
    """Read something from the microscope, then pause."""
    answer = call(*arguments)
    time.sleep(PAUSE_S)
    return answer


def change(call, *arguments):
    """Set something on the microscope, then pause. In a dry run, only print it."""
    global last_change
    last_change = f"{call.__name__}{arguments}" + (" (dry run: not done)" if DRY_RUN else "")
    if DRY_RUN:
        print(f"   would call {call.__name__}{arguments}")
        return
    call(*arguments)
    time.sleep(PAUSE_S)


def take_frame():
    """One camera frame, as an array of integers. With SCREEN on, it is shown with the beam marked."""
    global frames_shown
    frame = np.frombuffer(ask(camera.snapshot_rawdata), dtype=frame_dtype).reshape(frame_shape)
    if SCREEN:
        frames_shown += 1
        beam = find_beam(frame=frame, background=dark_level)
        show_frame(frame, beams=[("beam", beam)],
                   title=f"{frames_shown}. {mode or 'before the recording'}  |  after {last_change}\n"
                         f"beam at ({beam.centre.real:+.1f}, {beam.centre.imag:+.1f}) px, rms {beam.rms:.1f} px, "
                         f"roundness {beam.roundness:.2f}")
    return frame


def read_values():
    """The condenser lenses, the projection lenses, CLA1, CLA2 and PLA, as the microscope has them now."""
    cla1, cla2, pla = ask(deflectors.GetCLA1), ask(deflectors.GetCLA2), ask(deflectors.GetPLA)
    return {"CL1": ask(lens.GetCL1), "CL2": ask(lens.GetCL2), "CL3": ask(lens.GetCL3),
            "IL1": ask(lens.GetIL1), "IL2": ask(lens.GetIL2), "IL3": ask(lens.GetIL3), "PL1": ask(lens.GetPL1),
            "CLA1_x": cla1[0], "CLA1_y": cla1[1], "CLA2_x": cla2[0], "CLA2_y": cla2[1], "PLA_x": pla[0], "PLA_y": pla[1]}


def take_and_save_frame(**about):
    """Take a frame, find the beam, save the frame, and add a row to the table: `about` (what the frame is for),
    the values read now, and the beam's position, radius, roundness and signal. Returns the beam."""
    row = dict(mode=mode, **about)
    row.update(read_values())
    frame = take_frame()
    beam = find_beam(frame=frame, background=dark_level)
    row.update(beam_x_px=beam.centre.real, beam_y_px=beam.centre.imag, beam_rms_px=beam.rms,
               beam_roundness=beam.roundness, beam_signal=beam.signal, beam_usable=beam.usable,
               time_s=round(time.time(), 2))
    if not DRY_RUN:
        row["frame"] = f"frames/{len(rows):04d}.npy"
        np.save(folder / row["frame"], frame)
    rows.append(row)
    return beam

# %% [markdown]
# ## Lenses, deflectors, restoring and saving

# %%
def set_lenses(values):
    """IL1, IL2, IL3 and PL1 to these values (free-lens control), then wait for them to settle."""
    for name, value in zip(LENSES, values):
        if not 0 <= value <= DAC_MAX:
            raise ValueError(f"{name} = {value} is outside 0 to {DAC_MAX}")
        change(lens.SetFLCAbs, FREE_LENS_NUMBER[name], int(value))
    if not DRY_RUN:
        time.sleep(lens_settle_s)


def set_deflector(deflector, x, y):
    """CLA1 or CLA2 to (x, y)."""
    change(getattr(deflectors, "Set" + deflector), x, y)


def deflect_and_measure(deflector, units, **about):
    """A frame with the deflector where the console had it, one with it `units` further along x, then the deflector
    back. Returns how far the beam moved (px, x + iy), and whether that can be trusted: the beam found in both
    frames, and still round. If not, it says why."""
    x, y = start_value[deflector]
    if not 0 <= x + units <= DAC_MAX:
        raise ValueError(f"{deflector} x = {x + units} is outside 0 to {DAC_MAX}")
    before = take_and_save_frame(role="not deflected", **about)
    set_deflector(deflector, x + units, y)
    after = take_and_save_frame(role="deflected", deflector=deflector, deflection_x=units, deflection_y=0, **about)
    set_deflector(deflector, x, y)
    still_round = after.roundness >= MIN_ROUNDNESS * before.roundness
    usable = before.usable and after.usable and still_round
    if not usable:
        reasons = before.problems + after.problems
        if not still_round:
            reasons.append(f"the deflected beam's roundness is {after.roundness:.2f}, not deflected {before.roundness:.2f}")
        print("      not usable: " + ("; ".join(reasons) or "the beam cannot be found"))
    return after.centre - before.centre, usable


def restore_microscope():
    """Free-lens control off (the console's own IL1-PL1 again), CLA1, CLA2 and PLA back where the console had them,
    and the beam unblanked. Safe to run at any time."""
    for number in FREE_LENS_NUMBER.values():
        change(lens.SetFLCSw, number, 0)
    if start_value:
        for deflector in ["CLA1", "CLA2", "PLA"]:
            change(getattr(deflectors, "Set" + deflector), *start_value[deflector])
    change(deflectors.SetBeamBlank, 0)


def save_table():
    """Write every row to table.csv, and what the console had in each mode to modes.json."""
    if DRY_RUN or not rows:
        print(f"{len(rows)} rows; not saved ({'a dry run' if DRY_RUN else 'nothing recorded'})")
        return
    columns = []
    for row in rows:
        columns += [key for key in row if key not in columns]
    with open(folder / "table.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (folder / "modes.json").write_text(json.dumps(modes, indent=1, default=str))
    print(f"{len(rows)} rows in {folder / 'table.csv'}")

# %% [markdown]
# ## The steps of a recording, done the same way in MAG and in DIFF

# %%
def read_console():
    """Step 1: IL1-PL1, CLA1, CLA2 and PLA as the console has them in this mode (every lens is noted in `modes`),
    then free-lens control of IL1-PL1 at those values. Returns IL1-PL1."""
    global start_value
    here = [ask(getattr(lens, "Get" + name)) for name in LENSES]
    start_value = {name: ask(getattr(deflectors, "Get" + name)) for name in ["CLA1", "CLA2", "PLA"]}
    every_lens = {name: ask(getattr(lens, "Get" + name)) if hasattr(lens, "Get" + name) else None
                  for name in ALL_LENSES}
    modes[mode] = dict(function_mode=ask(eos.GetFunctionMode), magnification=ask(eos.GetMagValue),
                       lenses=every_lens, deflectors=start_value)
    print("IL1, IL2, IL3, PL1:", here)
    print("CLA1, CLA2, PLA:", start_value)
    print("every lens:", every_lens)
    for name, value in zip(LENSES, here):
        change(lens.SetFLCAbs, FREE_LENS_NUMBER[name], value)
        change(lens.SetFLCSw, FREE_LENS_NUMBER[name], 1)
    return here


def find_biggest_deflection(deflector, beam_at_start):
    """Step 2a: deflect by 100, 200, 400, ... units until the beam moves 10 px. From that, the biggest deflection
    that keeps the beam (its centre and 3 radii) inside the middle two-thirds of the camera; never bigger than one
    where the beam was lost or not round."""
    size, px_per_unit, last_good = 100, None, None
    while True:
        moved, usable = deflect_and_measure(deflector, size, setting="size search")
        print(f"   {deflector} by {size:5d} units: the beam moves {abs(moved):5.1f} px")
        if not usable:
            break
        last_good, px_per_unit = size, max(abs(moved) / size, 1e-6)
        if abs(moved) >= 10 or size >= 20000:
            break
        size = min(2 * size, 20000)
    if px_per_unit is None:
        raise RuntimeError(f"{deflector}: the beam cannot be used even at 100 units: check the alignment")
    biggest = int(min(20000, (frame_shape[1] / 3 - 3 * beam_at_start.rms) / px_per_unit))
    if not usable:
        biggest = min(biggest, last_good)
    print(f"{deflector}: about {px_per_unit:.5f} px per unit; the biggest deflection to try: {biggest} units")
    return biggest


def choose_deflection(deflector, biggest):
    """Step 2b: six equal steps up to `biggest`. A line through zero is fitted to the three smallest; the deflection
    to record with is the biggest step still on the line (within 0.3 px or 2 %), with the beam found and round.
    Returns that deflection (units) and how far it moves the beam (px, x + iy)."""
    steps = [biggest * k // 6 for k in range(1, 7)]
    moved_by = {}
    for step in steps:
        moved, usable = deflect_and_measure(deflector, step, setting="size scan")
        moved_by[step] = moved if usable else None
    small = [step for step in steps[:3] if moved_by[step] is not None]
    if len(small) < 2:
        raise RuntimeError(f"{deflector}: the beam cannot be used at the smallest steps: check the alignment")
    slope = sum(moved_by[step] * step for step in small) / sum(step * step for step in small)    # px per unit
    size, moved_at_size = None, None
    for step in steps:
        if moved_by[step] is None:
            print(f"   {step:6d} units: the beam is lost or not round: stop here")
            break
        off_the_line = abs(moved_by[step] - slope * step)
        on_the_line = off_the_line <= max(0.3, 0.02 * abs(moved_by[step]))
        print(f"   {step:6d} units: the beam moves {abs(moved_by[step]):6.1f} px, {off_the_line:5.2f} px off the straight line"
              + ("" if on_the_line else "   <- not on the line: stop here"))
        if not on_the_line:
            break
        size, moved_at_size = step, moved_by[step]
    if size is None:
        raise RuntimeError(f"{deflector}: no step was usable: check the alignment")
    print(f"{deflector}: record with {size} units, which move the beam {abs(moved_at_size):.1f} px")
    return size, moved_at_size


def find_lens_steps(deflector, size, moved_at_size, here, largest_lens_step):
    """Step 3: change each lens by 250, 500, 1000, ... units (at most `largest_lens_step`) until the deflection moves
    the beam more than 10 px differently. Each lens is changed the way it has more room (staying between 2000 and
    65535); the step found is then used up and down, each way no more than the room on that side."""
    step_up, step_down = {}, {}
    for i, name in enumerate(LENSES):
        room_up, room_down = DAC_MAX - here[i], here[i] - 2000
        direction = +1 if room_up >= room_down else -1
        largest_step = int(min(largest_lens_step, max(room_up, room_down)))
        print(f"{name} is at {here[i]}. {'Raise' if direction > 0 else 'Lower'} it by 250, 500, 1000, ... "
              f"(at most {largest_step}) until {deflector} moves the beam more than 10 px differently:")
        step = 250
        while True:
            trial = list(here)
            trial[i] = here[i] + direction * step
            set_lenses(trial)
            moved, usable = deflect_and_measure(deflector, size, setting="lens step search")
            difference = abs(moved - moved_at_size)
            print(f"   {name} at {trial[i]} ({direction * step:+d}): {deflector} moves the beam {abs(moved):5.1f} px "
                  f"(at the console's {name}: {abs(moved_at_size):5.1f} px); difference {difference:4.1f} px")
            if not usable:                               # the beam left the camera or lost its shape: back to half
                step = step // 2
                break
            if difference > 10 or step >= largest_step:
                break
            step = min(2 * step, largest_step)
        step_up[name], step_down[name] = int(min(step, room_up)), int(min(step, room_down))
    set_lenses(here)
    print("steps up:", step_up, "  down:", step_down)
    return step_up, step_down


def make_plan(here, step_up, step_down):
    """The lens settings to record: the console's, each lens up and down, and the console's again (the repeat
    shows how much the measurements wobble)."""
    plan = [(f"{mode} row", list(here))]
    for i, name in enumerate(LENSES):
        up, down = list(here), list(here)
        up[i] = here[i] + step_up[name]
        down[i] = here[i] - step_down[name]
        plan.append((f"{mode}: {name} +{step_up[name]}", up))
        plan.append((f"{mode}: {name} -{step_down[name]}", down))
    plan.append((f"{mode} row", list(here)))
    for number, (name, values) in enumerate(plan, 1):
        print(f"{number:2d}. {name:28s} IL1, IL2, IL3, PL1 = {[int(v) for v in values]}")
    return plan


def record(deflector, size, plan):
    """Step 4: at each lens setting, a frame not deflected, then the deflector + and - along x, and + and - along y."""
    x, y = start_value[deflector]
    for number, (name, values) in enumerate(plan, 1):
        print(f"{number}/{len(plan)}. {name}")
        set_lenses(values)
        take_and_save_frame(role="not deflected", number=number, setting=name)
        for direction, dx, dy in [("x", size, 0), ("y", 0, size)]:
            for role, sign in [("deflected +", +1), ("deflected -", -1)]:
                set_deflector(deflector, x + sign * dx, y + sign * dy)
                take_and_save_frame(role=role, number=number, setting=name, deflector=deflector, direction=direction,
                                    deflection_x=sign * dx, deflection_y=sign * dy)
        set_deflector(deflector, x, y)


def print_beam(beam):
    print(f"beam at ({beam.centre.real:+.1f}, {beam.centre.imag:+.1f}) px from the centre, rms {beam.rms:.1f} px, "
          f"roundness {beam.roundness:.2f}; problems:", "; ".join(beam.problems) or "none")

# %% [markdown]
# ## If a cell fails or is interrupted, restore the microscope
#
# Run this cell once: after it, every cell that fails (or is stopped) calls `restore_microscope()`.

# %%
def restore_if_a_cell_failed(result):
    if result.error_in_exec is not None:
        print("A cell failed or was interrupted: restoring the microscope.")
        restore_microscope()


get_ipython().events.register("post_run_cell", restore_if_a_cell_failed)  # noqa: F821

# %% [markdown]
# ## The camera: frame size and pixel type, and a first look at the beam

# %%
settings = ask(camera.get_detectorsetting)
print("camera settings:", settings)
frame_shape = (settings["ImagingArea"]["Height"], settings["ImagingArea"]["Width"])        # or set it by hand
frame_bytes = ask(camera.snapshot_rawdata)
frame_dtype = {2: "<i2", 4: "<i4"}[len(frame_bytes) // (frame_shape[0] * frame_shape[1])]   # signed; "<u2" if bright
frame = take_frame()                                                                        # pixels come out negative
print(f"one frame: {frame_shape[1]} x {frame_shape[0]} px, {frame_dtype}, values {frame.min()} to {frame.max()}")
print_beam(find_beam(frame=frame))

# %% [markdown]
# ## The microscope as found, and the folder for this recording

# %%
mode_at_start = ask(eos.GetFunctionMode)[0]              # switched back to at the end
print("the microscope says:", ask(eos.GetFunctionMode), ask(eos.GetMagValue))
folder = folder / time.strftime("projection_response_%Y-%m-%d_%H%M%S")      # a new folder each time
if not DRY_RUN:
    (folder / "frames").mkdir(parents=True)
    print("recording into", folder)

# %% [markdown]
# ## MAG at 40k: CLA1 (beam shift)
#
# ### Check that the microscope is in MAG at the 40k row (set at the console before starting)

# %%
mode = "MAG 40k"
print("the microscope says:", ask(eos.GetFunctionMode), ask(eos.GetMagValue))       # expect [0, 'MAG'] [40000, ...]
if (ask(eos.GetFunctionMode)[0], ask(eos.GetMagValue)[0]) != (0, 40000):
    raise RuntimeError("set MAG at the 40k row at the console, then run this cell again")

# %% [markdown]
# ### 1. What the console has, and take over IL1-PL1

# %%
here = read_console()

# %% [markdown]
# ### The dark level: a frame with the beam blanked (it holds for DIFF too)

# %%
change(deflectors.SetBeamBlank, 1)
time.sleep(0.5)
dark_frame = take_frame()
change(deflectors.SetBeamBlank, 0)
dark_level = None if DRY_RUN else float(np.median(dark_frame))       # in a dry run the beam was not blanked
print("dark level:", dark_level)

# %% [markdown]
# ### 2. The deflection size for CLA1

# %%
beam_at_start = take_and_save_frame(role="not deflected", setting="start")
print_beam(beam_at_start)
biggest = find_biggest_deflection("CLA1", beam_at_start)

# %%
size, moved_at_size = choose_deflection("CLA1", biggest)

# %% [markdown]
# ### 3. How far to change each lens (at most 8000 units in MAG)

# %%
step_up, step_down = find_lens_steps("CLA1", size, moved_at_size, here, largest_lens_step=8000)

# %% [markdown]
# ### 4. Record

# %%
plan = make_plan(here, step_up, step_down)

# %%
record("CLA1", size, plan)

# %% [markdown]
# ### 5. Restore the microscope

# %%
restore_microscope()

# %% [markdown]
# ## DIFF at 40 cm: CLA2 (beam tilt)
#
# ### Switch to DIFF, and check that the camera length is 40 cm
#
# Free-lens control was switched off above, so the console loads its own DIFF lens values. The camera length is the
# one the console has in DIFF: set it to 40 cm at the console before starting.

# %%
mode = "DIFF 40 cm"
print(f"switching to DIFF (SelectFunctionMode 4), then waiting {mode_settle_s} s")
change(eos.SelectFunctionMode, 4)                        # 4: DIFF
if not DRY_RUN:
    time.sleep(mode_settle_s)
print("the microscope says:", ask(eos.GetFunctionMode), ask(eos.GetMagValue))       # expect [4, 'DIFF'] [40, 'cm', ...]
if not DRY_RUN and (ask(eos.GetFunctionMode)[0], ask(eos.GetMagValue)[0]) != (4, 40):
    raise RuntimeError("expected DIFF at 40 cm: set the camera length at the console, then run this cell again")

# %% [markdown]
# ### 1. What the console has, and take over IL1-PL1

# %%
here = read_console()

# %% [markdown]
# ### Centre the diffraction pattern with PLA
#
# PLA is moved by a trial step along x (100, 200, 400, ... units until the pattern moves 5 px) and along y; from how
# far the pattern moved, the move that centres it is worked out and made. At most three tries; PLA is never moved more
# than 15000 units from where the console had it. `restore_microscope()` puts it back when DIFF is finished.

# %%
pla = list(start_value["PLA"])
for attempt in range(4):
    change(deflectors.SetPLA, *pla)
    pattern = find_beam(frame=take_frame(), background=dark_level)
    print(f"   pattern at ({pattern.centre.real:+.1f}, {pattern.centre.imag:+.1f}) px from the middle, PLA {pla}")
    if not pattern.usable:
        raise RuntimeError("the whole pattern cannot be found: centre it with PLA by hand")
    if abs(pattern.centre) < 2 or DRY_RUN:
        break
    if attempt == 3:
        raise RuntimeError("the pattern is not centred after three tries: centre it with PLA by hand")
    step = 100
    while True:
        change(deflectors.SetPLA, pla[0] + step, pla[1])
        moved_x = find_beam(frame=take_frame(), background=dark_level).centre - pattern.centre
        if abs(moved_x) > 5 or step >= 4000:
            break
        step *= 2
    change(deflectors.SetPLA, pla[0], pla[1] + step)
    moved_y = find_beam(frame=take_frame(), background=dark_level).centre - pattern.centre
    px_per_pla_unit = np.array([[moved_x.real, moved_y.real], [moved_x.imag, moved_y.imag]]) / step
    if abs(np.linalg.det(px_per_pla_unit)) < 1e-12:
        raise RuntimeError("PLA does not move the pattern: centre it by hand")
    move_x, move_y = np.linalg.solve(px_per_pla_unit, [-pattern.centre.real, -pattern.centre.imag])
    pla = [int(round(pla[0] + move_x)), int(round(pla[1] + move_y))]
    too_far = max(abs(pla[0] - start_value["PLA"][0]), abs(pla[1] - start_value["PLA"][1])) > 15000
    if too_far or not (0 <= pla[0] <= DAC_MAX and 0 <= pla[1] <= DAC_MAX):
        raise RuntimeError(f"centring would need PLA at {pla}, too far from {start_value['PLA']}: centre it by hand")

# %% [markdown]
# ### 2. The deflection size for CLA2

# %%
beam_at_start = take_and_save_frame(role="not deflected", setting="start")
print_beam(beam_at_start)
biggest = find_biggest_deflection("CLA2", beam_at_start)

# %%
size, moved_at_size = choose_deflection("CLA2", biggest)

# %% [markdown]
# ### 3. How far to change each lens (at most 2000 units in DIFF)

# %%
step_up, step_down = find_lens_steps("CLA2", size, moved_at_size, here, largest_lens_step=2000)

# %% [markdown]
# ### 4. Record

# %%
plan = make_plan(here, step_up, step_down)

# %%
record("CLA2", size, plan)

# %% [markdown]
# ### 5. Restore the microscope

# %%
restore_microscope()

# %% [markdown]
# ## The end: back to the mode the microscope started in, and the table saved

# %%
if ask(eos.GetFunctionMode)[0] != mode_at_start:
    print(f"switching back to function mode {mode_at_start}, then waiting {mode_settle_s} s")
    change(eos.SelectFunctionMode, mode_at_start)
    if not DRY_RUN:
        time.sleep(mode_settle_s)
print("the microscope says:", ask(eos.GetFunctionMode), ask(eos.GetMagValue))
save_table()

# %% [markdown]
# # Fitting the lens model to the recording (needs TemGym and JAX)
#
# This part runs on a computer with TemGym. To fit a recording made earlier (for example one copied from the
# microscope), run the setup cell, then set `recording` below to its folder and run the cells from there.

# %%
recording = folder                 # the recording just made; or one made earlier, for example
# recording = Path("output/stage1_neoarm/projection_response_2026-11-10_150000")

os.environ["JAX_PLATFORMS"] = "cpu"                      # JAX on the CPU (before it is imported)
import jax                                               # noqa: E402
import jax.numpy as jnp                                  # noqa: E402
from jax.flatten_util import ravel_pytree                # noqa: E402
from scipy.optimize import least_squares                 # noqa: E402
from temgym_core.components import HardEdgeSolenoid, Plane      # noqa: E402
from temgym_core.run import position_and_angle_response          # noqa: E402
jax.config.update("jax_enable_x64", True)

# %% [markdown]
# ### 1. The displacements, from the table
#
# For each lens setting and direction, the beam centres in the + and the - frame; the displacement is half their
# difference, so where the beam started cancels. Pairs where the beam was not usable are left out.

# %%
table = list(csv.DictReader(open(recording / "table.csv")))
plus, minus = {}, {}
for row in table:
    key = (row["mode"], row["number"], row["direction"])
    if row["role"] == "deflected +":
        plus[key] = row
    if row["role"] == "deflected -":
        minus[key] = row
measurements = []
for key in plus:
    if key not in minus:
        continue
    p, m = plus[key], minus[key]
    if p["beam_usable"] != "True" or m["beam_usable"] != "True":
        print("left out, the beam was not usable:", key)
        continue
    centre_plus = complex(float(p["beam_x_px"]), float(p["beam_y_px"]))
    centre_minus = complex(float(m["beam_x_px"]), float(m["beam_y_px"]))
    measurements.append(dict(mode=p["mode"], number=int(p["number"]), setting=p["setting"], deflector=p["deflector"],
                             direction=p["direction"], lenses=[float(p[name]) for name in LENSES],
                             deflection=complex(float(p["deflection_x"]), float(p["deflection_y"])),
                             displacement=(centre_plus - centre_minus) / 2))
print(len(measurements), "displacements")
for m in measurements:
    print(f"   {m['setting']:24s} {m['deflector']} {m['direction']} by {m['deflection']:.0f} units: "
          f"({m['displacement'].real:+7.2f}, {m['displacement'].imag:+7.2f}) px")

# %% [markdown]
# ### 2. The wobble: the console's lenses are recorded twice in each mode, so those displacements should be the same

# %%
differences, averages = [], 0
for mode_name in ["MAG 40k", "DIFF 40 cm"]:
    for direction in ["x", "y"]:
        repeats = np.array([m["displacement"] for m in measurements
                            if m["setting"] == f"{mode_name} row" and m["direction"] == direction])
        if len(repeats) > 1:
            differences.extend(repeats - repeats.mean())
            averages += 1
wobble = float(np.sqrt(np.sum(np.abs(differences) ** 2) / (2 * (len(differences) - averages))))    # px, in x or y
print(f"wobble: {wobble:.4f} px")

# %% [markdown]
# ### 3. The model of the projection system: the objective, IL1, IL2, IL3 and PL1 as five TemGym solenoids
#
# A lens's strength is the angle the beam turns through in it. The fit knows each lens by its gap from the one before,
# its length, and its strength at the 40k row; at another setting IL1-PL1 scale with their values relative to the 40k
# row. The fitted numbers are logarithms, so that gaps, lengths and strengths stay positive. For each lens setting
# TemGym gives A and B: how far the beam moves on the camera per metre of shift and per radian of tilt at the specimen.

# %%
SIGNS = jnp.array([1.0, 1.0, 1.0, -1.0])                 # PL1's field points the other way
PIXEL = 110e-6                                           # camera pixel (m); its value only scales the deflector sizes

START = dict(                                            # a rough projection system to start from
    log_gaps=np.log([0.0025, 0.19, 0.08, 0.05, 0.06, 0.34]),    # m: specimen to objective, objective to IL1, ..., PL1 to camera
    log_lengths=np.log([0.0005, 0.006, 0.004, 0.006, 0.0012]),   # m: objective, IL1, IL2, IL3, PL1
    objective=-0.40,                                             # strength (rad)
    log_strengths_40k=np.log([0.31, 0.34, 0.88, 0.59]),          # strengths (rad) of IL1-PL1 at the 40k row
)


def lens_column(lenses, strengths):
    """TemGym components from the specimen to the camera, for IL1-PL1 at `strengths` (1: the 40k row)."""
    gaps, lengths = jnp.exp(lenses["log_gaps"]), jnp.exp(lenses["log_lengths"])
    q = jnp.r_[lenses["objective"], SIGNS * jnp.exp(lenses["log_strengths_40k"]) * jnp.asarray(strengths)]
    entrances = jnp.cumsum(gaps[:5]) + jnp.r_[0.0, jnp.cumsum(lengths[:4])]
    return [*(HardEdgeSolenoid(z=entrances[i], length=lengths[i], q=q[i]) for i in range(5)),
            Plane(z=gaps.sum() + lengths.sum())]


def a_and_b_one_setting(lenses, strengths):
    return position_and_angle_response(lens_column(lenses, strengths))


a_and_b = jax.jit(jax.vmap(a_and_b_one_setting, in_axes=(None, 0)))    # A and B at many lens settings at once

# %% [markdown]
# ### 4. What a deflection does, and the displacement the model predicts
#
# A deflection along a deflector's x and y axes becomes a shift (CLA1) or tilt (CLA2) at the specimen; each deflector
# has its own axes (size, direction, and the y axis relative to the x axis). CLA1's x axis is the specimen's x. A turns
# the shift, and B the tilt, into a displacement on the camera, which the camera turns by its rotation.

# %%
def at_specimen(axes, deflection):
    """A deflection (units, x + iy) as what it does at the specimen (x + iy)."""
    x_axis = jnp.exp(axes["log_size"] + 1j * axes.get("x_angle", 0.0))
    y_axis = x_axis * (axes["y_axis"][0] + 1j * axes["y_axis"][1])
    return x_axis * deflection.real + y_axis * deflection.imag


def predicted_displacements(params, data):
    """How far each deflection moves the beam on the camera (px, x + iy)."""
    a, b = a_and_b(params["lenses"], data["strengths"])
    shift_at_specimen = at_specimen(params["CLA1"], data["deflection"])                     # m, for CLA1
    tilt_at_specimen = at_specimen(params["CLA2"], data["deflection"])                      # rad, for CLA2
    on_camera = jnp.where(data["is_tilt"], b * tilt_at_specimen, a * shift_at_specimen)     # m
    return jnp.exp(1j * params["camera_rotation"]) * on_camera / PIXEL


def least_squares_fit(start, data):
    """The parameters that best predict the measured displacements, each misfit counted in wobbles."""
    start_as_list, as_params = ravel_pytree(start)

    def misfits(parameter_list):
        misfit = (predicted_displacements(as_params(parameter_list), data) - data["displacement"]) / data["wobble"]
        return jnp.concatenate([misfit.real, misfit.imag])

    result = least_squares(jax.jit(misfits), np.asarray(start_as_list), jac=jax.jit(jax.jacfwd(misfits)),
                           method="lm", xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=2000)
    return dict(params=as_params(result.x), chi2=float(result.fun @ result.fun), dof=len(result.fun) - len(result.x))

# %% [markdown]
# ### 5. The fit
#
# It starts from the rough projection system, with square deflector axes sized to the measured displacements. The
# camera image may be mirrored, so it is fitted both ways, and each again half a turn round (a fit can get stuck the
# wrong way round). The best of the four is kept.

# %%
lenses_40k = np.array(next(m["lenses"] for m in measurements if m["setting"] == "MAG 40k row"))
data = dict(
    strengths=np.array([m["lenses"] for m in measurements]) / lenses_40k,   # IL1-PL1, relative to the 40k row
    is_tilt=np.array([m["deflector"] == "CLA2" for m in measurements]),     # CLA1 shifts, CLA2 tilts
    deflection=np.array([m["deflection"] for m in measurements]),           # units, x + iy
    displacement=np.array([m["displacement"] for m in measurements]),       # px, x + iy
    wobble=wobble,                                                          # px
)
a, b = a_and_b(START, data["strengths"])
size_per_unit = np.abs(data["displacement"]) * PIXEL / (np.abs(np.where(data["is_tilt"], b, a)) * np.abs(data["deflection"]))
start = dict(lenses=START, camera_rotation=0.0,
             CLA1=dict(log_size=np.log(np.median(size_per_unit[~data["is_tilt"]])), y_axis=np.array([0.0, 1.0])),
             CLA2=dict(log_size=np.log(np.median(size_per_unit[data["is_tilt"]])), x_angle=0.0, y_axis=np.array([0.0, 1.0])))
fits = []
for mirrored in [False, True]:
    data_as_seen = dict(data, displacement=np.conj(data["displacement"]) if mirrored else data["displacement"])
    shift, tilt = ~data["is_tilt"], data["is_tilt"]
    start_seen = predicted_displacements(start, data_as_seen)
    rotation = np.angle(np.vdot(start_seen[shift], data_as_seen["displacement"][shift]))
    tilt_angle = np.angle(np.vdot(start_seen[tilt], data_as_seen["displacement"][tilt])) - rotation
    for half_turns in [0, 1]:
        first = dict(start, camera_rotation=rotation + half_turns * np.pi,
                     CLA2=dict(start["CLA2"], x_angle=tilt_angle - half_turns * np.pi))
        fit = least_squares_fit(first, data_as_seen)
        print(f"   mirrored {mirrored!s:5s}, {half_turns} half turns: chi2/dof {fit['chi2'] / fit['dof']:.2f}")
        fits.append(dict(fit, mirrored=mirrored, data=data_as_seen))
best = fits[int(np.argmin([fit["chi2"] for fit in fits]))]
params = best["params"]
print(f"\nbest: camera image mirrored {best['mirrored']}; chi2/dof {best['chi2'] / best['dof']:.2f} (about 1: it fits to the wobble)")
for deflector_name in ["CLA1", "CLA2"]:
    y_axis = complex(*params[deflector_name]["y_axis"])
    print(f"{deflector_name} y axis, relative to its x axis: turned {np.degrees(np.angle(y_axis)):.1f} degrees, "
          f"{abs(y_axis):.3f} times as long")
print(f"CLA2 x axis, relative to CLA1's: turned {np.degrees(np.angle(np.exp(1j * float(params['CLA2']['x_angle'])))):.1f} degrees")

# %% [markdown]
# ### 6. How well it fits: each displacement measured and predicted, the worst first

# %%
predicted = np.asarray(predicted_displacements(params, best["data"]))
if best["mirrored"]:
    predicted = np.conj(predicted)                       # back to the camera's own x and y
misfit = np.abs(predicted - data["displacement"])
for i in np.argsort(misfit)[::-1]:
    m = measurements[i]
    print(f"   {m['setting']:24s} {m['deflector']} {m['direction']}: measured ({m['displacement'].real:+7.2f}, "
          f"{m['displacement'].imag:+7.2f}) px, predicted ({predicted[i].real:+7.2f}, {predicted[i].imag:+7.2f}) px, "
          f"missed by {misfit[i]:.3f} px")

(recording / "model.json").write_text(json.dumps(dict(params=params, mirrored=best["mirrored"], lenses_40k=lenses_40k,
                                                      wobble_px=wobble, chi2=best["chi2"], dof=best["dof"]),
                                                 indent=1, default=lambda array: np.asarray(array).tolist()))
print("the model:", recording / "model.json")
