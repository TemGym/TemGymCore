# %% [markdown]
# # How far a beam shift (CLA1) or tilt (CLA2) displaces the beam on the camera, for different IL1, IL2, IL3, PL1
#
# In MAG (40k) a beam shift (CLA1) displaces the beam on the camera and a tilt hardly does; in DIFF (40 cm) it is the
# other way round. So MAG records CLA1, and DIFF records CLA2. Both modes go through the same steps:
#
# 1. note what the console has, and take over IL1-PL1 at those values (free-lens control);
# 2. find a deflection that displaces the beam a useful distance, in a straight line;
# 3. find how far to change each lens so that the deflection displaces the beam noticeably differently;
# 4. record: at the console's lenses and at each lens up and down, the beam deflected + and - along x and y;
# 5. restore the microscope.
#
# Every saved frame is one row of `table.csv` (see `neoarm.py`). `fit_projection_response.ipynb` fits TemGym's lenses
# to it.
#
# Run it as a script from this folder:
#     python record_projection_response.py --simulated    the virtual NEOARM (needs TemGym)
#     python record_projection_response.py                the real NEOARM, a dry run: reads it, changes nothing
#     python record_projection_response.py --for-real     the real NEOARM
# or cell by cell (set the switches below), checking what each cell prints before running the next.

# %%
import sys

import numpy as np

from neoarm import DAC_MAX, Neoarm, describe

SIMULATED = "--simulated" in sys.argv      # or set True here when running cell by cell
FOR_REAL = "--for-real" in sys.argv
LENSES = ["IL1", "IL2", "IL3", "PL1"]      # the lenses whose settings we vary
MIN_ROUNDNESS = 0.9                        # a deflected beam must stay this fraction as round as before

scope = Neoarm("projection_response", for_real=FOR_REAL, simulated=SIMULATED, magnification=40000, camera_length_m=0.4)
scope.restore_if_anything_fails()
eos = scope.eos
label = ""                                  # "MAG 40k" or "DIFF 40 cm": the mode column of the table


# %% [markdown]
# ## The steps, done the same way in MAG and in DIFF

# %%
def deflect_and_measure(deflector, units, **about):
    """A frame with the deflector where the console had it, one with it `units` further along x, then the deflector
    back. Returns the displacement (px, x + iy), and whether it can be trusted: the beam found in both frames, and
    still round. If not, it says why."""
    x, y = scope.deflectors_at_start[deflector]
    before = scope.record(mode=label, role="not deflected", **about)
    scope.set_deflector(deflector, x + units, y)
    after = scope.record(mode=label, role="deflected", deflector=deflector, deflection_x=units, deflection_y=0, **about)
    scope.set_deflector(deflector, x, y)
    still_round = after.roundness >= MIN_ROUNDNESS * before.roundness
    usable = before.usable and after.usable and still_round
    if not usable:
        reasons = before.problems + after.problems
        if not still_round:
            reasons.append(f"the deflected beam's roundness is {after.roundness:.2f}, not deflected {before.roundness:.2f}")
        print("      not usable: " + ("; ".join(reasons) or "the beam cannot be found"))
    return after.centre - before.centre, usable


def start_mode(label):
    """Step 1: note what the console has, and take over IL1-PL1 at those values. Returns IL1-PL1."""
    state = scope.remember_console(label)
    here = [state[name] for name in LENSES]
    print("IL1, IL2, IL3, PL1:", here)
    print("CLA1, CLA2, PLA:", scope.deflectors_at_start)
    scope.take_over(*LENSES)
    return here


def find_biggest_deflection(deflector, beam_at_start):
    """Step 2a: deflect by 100, 200, 400, ... units until the beam is displaced 10 px. From that, the biggest
    deflection that keeps the beam (its centre and 3 radii) inside the middle two-thirds of the camera; never bigger
    than one where the beam was lost or not round."""
    size, px_per_unit, last_good = 100, None, None
    while True:
        displacement, usable = deflect_and_measure(deflector, size, setting="size search")
        print(f"   {deflector} by {size:5d} units: the beam is displaced {abs(displacement):5.1f} px")
        if not usable:
            break
        last_good, px_per_unit = size, max(abs(displacement) / size, 1e-6)
        if abs(displacement) >= 10 or size >= 20000:
            break
        size = min(2 * size, 20000)
    if px_per_unit is None:
        raise RuntimeError(f"{deflector}: the beam cannot be used even at 100 units: check the alignment")
    biggest = int(min(20000, (scope.frame_shape[1] / 3 - 3 * beam_at_start.rms) / px_per_unit))
    if not usable:
        biggest = min(biggest, last_good)
    print(f"{deflector}: about {px_per_unit:.5f} px per unit; the biggest deflection to try: {biggest} units")
    return biggest


def choose_deflection(deflector, biggest):
    """Step 2b: six equal steps up to `biggest`. A line through zero is fitted to the three smallest; the deflection
    to record with is the biggest step still on the line (within 0.3 px or 2 %), with the beam found and round.
    Returns that deflection (units) and its displacement (px, x + iy)."""
    steps = [biggest * k // 6 for k in range(1, 7)]
    displaced_by = {}
    for step in steps:
        displacement, usable = deflect_and_measure(deflector, step, setting="size scan")
        displaced_by[step] = displacement if usable else None
    small = [step for step in steps[:3] if displaced_by[step] is not None]
    if len(small) < 2:
        raise RuntimeError(f"{deflector}: the beam cannot be used at the smallest steps: check the alignment")
    slope = sum(displaced_by[step] * step for step in small) / sum(step * step for step in small)    # px per unit
    size, displacement_at_size = None, None
    for step in steps:
        if displaced_by[step] is None:
            print(f"   {step:6d} units: the beam is lost or not round: stop here")
            break
        off_the_line = abs(displaced_by[step] - slope * step)
        on_the_line = off_the_line <= max(0.3, 0.02 * abs(displaced_by[step]))
        print(f"   {step:6d} units: displaced {abs(displaced_by[step]):6.1f} px, {off_the_line:5.2f} px off the straight line"
              + ("" if on_the_line else "   <- not on the line: stop here"))
        if not on_the_line:
            break
        size, displacement_at_size = step, displaced_by[step]
    if size is None:
        raise RuntimeError(f"{deflector}: no step was usable: check the alignment")
    print(f"{deflector}: record with {size} units, which displace the beam {abs(displacement_at_size):.1f} px")
    return size, displacement_at_size


def find_lens_steps(deflector, size, displacement_at_size, here, largest_lens_step):
    """Step 3: change each lens by 250, 500, 1000, ... units (at most `largest_lens_step`) until the deflection
    displaces the beam more than 10 px differently. Each lens is changed the way it has more room (staying between
    2000 and 65535); the step found is then used up and down, each way no more than the room on that side."""
    step_up, step_down = {}, {}
    for i, name in enumerate(LENSES):
        room_up, room_down = DAC_MAX - here[i], here[i] - 2000
        direction = +1 if room_up >= room_down else -1
        largest_step = int(min(largest_lens_step, max(room_up, room_down)))
        print(f"{name} is at {here[i]}. {'Raise' if direction > 0 else 'Lower'} it by 250, 500, 1000, ... "
              f"(at most {largest_step}) until {deflector} displaces the beam more than 10 px differently:")
        step = 250
        while True:
            trial = list(here)
            trial[i] = here[i] + direction * step
            scope.set_lenses(**dict(zip(LENSES, trial)))
            displacement, usable = deflect_and_measure(deflector, size, setting="lens step search")
            difference = abs(displacement - displacement_at_size)
            print(f"   {name} at {trial[i]} ({direction * step:+d}): displaced {abs(displacement):5.1f} px "
                  f"(at the console's {name}: {abs(displacement_at_size):5.1f} px); difference {difference:4.1f} px")
            if not usable:                               # the beam left the camera or lost its shape: back to half
                step = step // 2
                break
            if difference > 10 or step >= largest_step:
                break
            step = min(2 * step, largest_step)
        step_up[name], step_down[name] = int(min(step, room_up)), int(min(step, room_down))
    scope.set_lenses(**dict(zip(LENSES, here)))
    print("steps up:", step_up, "  down:", step_down)
    return step_up, step_down


def make_plan(label, here, step_up, step_down):
    """The lens settings to record: the console's, each lens up and down, and the console's again (the repeat shows
    how well a displacement repeats)."""
    plan = [(f"{label} row", list(here))]
    for i, name in enumerate(LENSES):
        up, down = list(here), list(here)
        up[i] = here[i] + step_up[name]
        down[i] = here[i] - step_down[name]
        plan.append((f"{label}: {name} +{step_up[name]}", up))
        plan.append((f"{label}: {name} -{step_down[name]}", down))
    plan.append((f"{label} row", list(here)))
    for number, (name, values) in enumerate(plan, 1):
        print(f"{number:2d}. {name:28s} IL1, IL2, IL3, PL1 = {[int(v) for v in values]}")
    return plan


def record_plan(label, deflector, size, plan):
    """Step 4: at each lens setting, a frame not deflected, then the deflector + and - along x, and + and - along y."""
    x, y = scope.deflectors_at_start[deflector]
    for number, (name, values) in enumerate(plan, 1):
        print(f"{number}/{len(plan)}. {name}")
        scope.set_lenses(**dict(zip(LENSES, values)))
        scope.record(mode=label, role="not deflected", number=number, setting=name)
        for direction, dx, dy in [("x", size, 0), ("y", 0, size)]:
            for role, sign in [("deflected +", +1), ("deflected -", -1)]:
                scope.set_deflector(deflector, x + sign * dx, y + sign * dy)
                scope.record(mode=label, role=role, number=number, setting=name, deflector=deflector,
                             direction=direction, deflection_x=sign * dx, deflection_y=sign * dy)
        scope.set_deflector(deflector, x, y)


# %% [markdown]
# ## The camera and the microscope as found

# %%
mode_at_start = scope.ask(eos.GetFunctionMode)[0]          # switched back to at the end
print("the microscope says:", scope.ask(eos.GetFunctionMode), scope.ask(eos.GetMagValue))
frame = scope.take_frame()
print(f"one frame: {frame.shape[1]} x {frame.shape[0]} px, {frame.dtype}, values {frame.min()} to {frame.max()}")
print(describe(scope.measure()))

# %% [markdown]
# ## MAG at 40k: CLA1 (beam shift)
#
# Set MAG at the 40k row at the console before starting.

# %%
label = "MAG 40k"
if (scope.ask(eos.GetFunctionMode)[0], scope.ask(eos.GetMagValue)[0]) != (0, 40000):
    raise RuntimeError("set MAG at the 40k row at the console, then run this again")
here = start_mode(label)
scope.measure_dark_level()                                   # it holds for DIFF too

# %%
beam_at_start = scope.record(mode=label, role="not deflected", setting="start")
print(describe(beam_at_start))
biggest = find_biggest_deflection("CLA1", beam_at_start)
size, displacement_at_size = choose_deflection("CLA1", biggest)

# %%
step_up, step_down = find_lens_steps("CLA1", size, displacement_at_size, here, largest_lens_step=8000)
plan = make_plan(label, here, step_up, step_down)

# %%
record_plan(label, "CLA1", size, plan)
scope.restore()

# %% [markdown]
# ## DIFF at 40 cm: CLA2 (beam tilt)
#
# Free-lens control was switched off above, so the console loads its own DIFF lens values. Set the camera length to
# 40 cm at the console before starting.

# %%
label = "DIFF 40 cm"
print(f"switching to DIFF (SelectFunctionMode 4), then waiting {scope.mode_settle_s} s")
scope.change(eos.SelectFunctionMode, 4)
scope.wait(scope.mode_settle_s)
print("the microscope says:", scope.ask(eos.GetFunctionMode), scope.ask(eos.GetMagValue))   # expect [4, 'DIFF'] [40, 'cm', ...]
if not scope.dry_run and (scope.ask(eos.GetFunctionMode)[0], scope.ask(eos.GetMagValue)[0]) != (4, 40):
    raise RuntimeError("expected DIFF at 40 cm: set the camera length at the console, then run this again")
here = start_mode(label)

# %% [markdown]
# ### Centre the diffraction pattern with PLA
#
# PLA is changed by a trial step along x (100, 200, 400, ... units until the pattern is displaced 5 px) and along y;
# from those displacements the setting that centres it is worked out and made. At most three tries; PLA is never more
# than 15000 units from where the console had it. `restore()` puts it back.

# %%
pla = list(scope.deflectors_at_start["PLA"])
for attempt in range(4):
    scope.set_deflector("PLA", *pla)
    pattern = scope.measure("centring with PLA")
    print(f"   pattern at ({pattern.centre.real:+.1f}, {pattern.centre.imag:+.1f}) px from the middle, PLA {pla}")
    if not pattern.usable:
        raise RuntimeError("the whole pattern cannot be found: centre it with PLA by hand")
    if abs(pattern.centre) < 2 or scope.dry_run:
        break
    if attempt == 3:
        raise RuntimeError("the pattern is not centred after three tries: centre it with PLA by hand")
    step = 100
    while True:
        scope.set_deflector("PLA", pla[0] + step, pla[1])
        along_x = scope.measure("PLA trial along x").centre - pattern.centre
        if abs(along_x) > 5 or step >= 4000:
            break
        step *= 2
    scope.set_deflector("PLA", pla[0], pla[1] + step)
    along_y = scope.measure("PLA trial along y").centre - pattern.centre
    px_per_pla_unit = np.array([[along_x.real, along_y.real], [along_x.imag, along_y.imag]]) / step
    if abs(np.linalg.det(px_per_pla_unit)) < 1e-12:
        raise RuntimeError("PLA does not displace the pattern: centre it by hand")
    change_x, change_y = np.linalg.solve(px_per_pla_unit, [-pattern.centre.real, -pattern.centre.imag])
    pla = [int(round(pla[0] + change_x)), int(round(pla[1] + change_y))]
    start = scope.deflectors_at_start["PLA"]
    if max(abs(pla[0] - start[0]), abs(pla[1] - start[1])) > 15000 or not all(0 <= v <= DAC_MAX for v in pla):
        raise RuntimeError(f"centring would need PLA at {pla}, too far from {start}: centre it by hand")

# %%
beam_at_start = scope.record(mode=label, role="not deflected", setting="start")
print(describe(beam_at_start))
biggest = find_biggest_deflection("CLA2", beam_at_start)
size, displacement_at_size = choose_deflection("CLA2", biggest)

# %%
step_up, step_down = find_lens_steps("CLA2", size, displacement_at_size, here, largest_lens_step=2000)
plan = make_plan(label, here, step_up, step_down)

# %%
record_plan(label, "CLA2", size, plan)
scope.restore()

# %% [markdown]
# ## The end: back to the mode the microscope started in, and the recording saved

# %%
if scope.ask(eos.GetFunctionMode)[0] != mode_at_start:
    print(f"switching back to function mode {mode_at_start}, then waiting {scope.mode_settle_s} s")
    scope.change(eos.SelectFunctionMode, mode_at_start)
    scope.wait(scope.mode_settle_s)
scope.save()
