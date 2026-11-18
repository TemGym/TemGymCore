# %% [markdown]
# # The projection lenses in metres: one recording on the screen camera and the Merlin below it
#
# The two cameras sit at different heights, so a beam seen on both gives its slope after the last lens as well as where
# it lands. That slope only says something new when PL1 is not at its usual strong value. All four lenses weak showed
# no beam on the real NEOARM (2026-11-16), so this recording never leaves a normal row by more than PL1:
#
# 1. reads the microscope as found, and takes a frame from each camera;
# 2. six normal rows (screen camera; MAG 8000 and DIFF 40 cm on the Merlin too);
# 3. at MAG 8000 and at DIFF 40 cm: only PL1 turned down, step by step (40000, 24000, 12000), IL1-IL3 left at the
#    row's values, CLA1 and CLA2 kicks on BOTH cameras at each step. If the beam is lost at a step, that row's
#    steps end there and the script goes on (it does not stop and ask);
# 4. the rulers: gold rings and a grating image, with the cross-grating with gold in the beam;
# 5. with --kept: eight more rows, never fitted, as a test of the model.
#
# At every setting, on each camera: the beam is sized with CL3 and centred (CLA1 in MAG, CLA2 in DIFF); then one frame
# not deflected, and CLA1 and CLA2 each + and - along x and along y. Every lens comes to its value from APPROACH units
# below. PLA stays neutral. Every saved frame is one row of `table.csv` (see `neoarm.py`), with every lens and
# deflector value. `fit_two_cameras.ipynb` fits TemGym's lenses to it.
#
# The camera is changed by the script: it moves the screen with PyJEM (getScreen / setScreen: 1 the SightX view camera,
# 2 the Merlin) and waits 4 s each time (SCREEN_SETTLE_S in neoarm.py).
#
# Run it as a script from this folder:
#     python record_two_cameras.py --simulated    the virtual NEOARM (needs TemGym)
#     python record_two_cameras.py                the real NEOARM, a dry run: reads it, prints every change, changes nothing
#     python record_two_cameras.py --for-real     the real NEOARM
# or cell by cell (set the switches below), checking what each cell prints before running the next.
# Also: --merlin=NAME (PyJEM's name for the Merlin; if PyJEM does not know it, merlin_camera.py reads it),
#       --approach=N (default 8000), --quick (a short rehearsal of the script, too short to fit), --kept (the eight test rows too).
#       --by-hand: do not move the screen, stop and ask you to change the camera instead.

# %%
import sys

import numpy as np

from neoarm import DAC_MAX, DAC_MIN, SCREEN_BOTTOM, SCREEN_VIEW, Neoarm, describe, find_pitch, find_rings

SIMULATED = "--simulated" in sys.argv      # or set True here when running cell by cell
FOR_REAL = "--for-real" in sys.argv
QUICK = "--quick" in sys.argv
WITH_KEPT = "--kept" in sys.argv
BY_HAND = "--by-hand" in sys.argv
MERLIN = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--merlin=")), "merlin")
APPROACH = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--approach=")), 8000))

# ---- the cameras (the real NEOARM) ----
# screen camera (SightX), PyJEM's name "screenCamera": a 1916 x 1916 px sensor read at binning 4, so frames of
#     479 x 479 px, unsigned 16 bit (uint16); one binned pixel is 110 um (27.5 um sensor pixels x 4).
#     neoarm.py holds these numbers (FRAME_SHAPE, FRAME_DTYPE). The virtual NEOARM gives the same size, signed 32 bit.
# Merlin: 512 x 512 px, unsigned 16 bit, 55 um pixels (merlin_camera.py holds the shape for the lines you fill in).
SCREEN = "screenCamera"
SCREEN_FRAME = ((1916 // 4, 1916 // 4), "uint16")      # (rows, columns), data type: checked against what is read below
MERLIN_FRAME = ((512, 512), "uint16")
SCREEN_AT = {SCREEN: SCREEN_VIEW, MERLIN: SCREEN_BOTTOM}   # PyJEM's screen position that brings the beam to each camera

LENSES = ["IL1", "IL2", "IL3", "PL1"]
MAG, DIFF = 0, 4                           # PyJEM's function modes

# ---- the recipe: six rows, and PL1 alone turned down at two of them ----
PL1_STEPS = [40000, 24000, 12000]          # PL1 at the rows below (the console has it near 64000)
PL1_ROWS = [(MAG, 8000), (DIFF, 40)]       # where PL1 is turned down; these rows are also taken on the Merlin
ROWS = [(MAG, 40000), (DIFF, 40), (MAG, 8000), (DIFF, 15), (MAG, 200000), (DIFF, 100)]   # DIFF: camera length in cm
ON_THE_MERLIN_TOO = PL1_ROWS
KEPT = [(MAG, 12000), (MAG, 20000), (MAG, 100000), (MAG, 300000), (DIFF, 25), (DIFF, 60), (DIFF, 80), (DIFF, 150)]
GOLD = [(40, SCREEN), (80, SCREEN), (15, MERLIN), (20, MERLIN)]   # camera lengths (cm) where the rings fit the camera
GRATING = [8000, 12000]                    # magnifications, screen camera
if QUICK:
    PL1_STEPS, ROWS, KEPT, GOLD, GRATING = PL1_STEPS[1:2], ROWS[1:3], KEPT[:1], GOLD[::2], GRATING[:1]

LARGEST_KICK = {"CLA1": 12000, "CLA2": 1500}   # a larger tilt would bring in the objective's aberrations
STEER = {MAG: "CLA1", DIFF: "CLA2"}            # centres the beam
NEUTRAL = (DAC_MIN + DAC_MAX + 1) // 2         # PLA

scope = Neoarm("two_cameras", for_real=FOR_REAL, simulated=SIMULATED)
scope.restore_if_anything_fails()
eos, lens = scope.eos, scope.lens
stage = ""                                     # the stage column of the table
cl3_that_worked = {}                           # (camera, kind of setting) -> CL3: where the next search starts
brightest = {}                                 # camera -> the most signal seen: a real beam has a good part of it
dry_run_camera = SCREEN                        # a dry run only pretends to change the camera


# %% [markdown]
# ## The steps, done the same way at every setting

# %%
def set_from_below(**values):
    """Lenses to these values, each first to APPROACH below its value: every value is reached the same way."""
    scope.set_lenses(**{name: max(DAC_MIN, value - APPROACH) for name, value in values.items()})
    scope.set_lenses(**values)


def pla_neutral():
    """PLA to neutral, if the console put it elsewhere (it may reload its own values with the mode)."""
    if tuple(scope.ask(scope.deflectors.GetPLA)) != (NEUTRAL, NEUTRAL):
        scope.set_deflector("PLA", NEUTRAL, NEUTRAL)


def row_label(mode):
    """PyJEM reports real DIFF in mm or m, and simulated DIFF in cm. Keep whole-cm labels as integers."""
    value, unit, _ = scope.ask(eos.GetMagValue)
    if mode == MAG:
        return value
    cm_per_unit = {"mm": 0.1, "cm": 1, "m": 100}
    if unit not in cm_per_unit:
        raise ValueError(f"Unknown diffraction selector unit {unit!r}")
    cm = value * cm_per_unit[unit]
    return int(cm) if cm == int(cm) else cm


def go_to_row(mode, label, *, exact=False):
    """Reach a magnification or camera length (cm), verify its readback and approach IL1–PL1 from below."""
    scope.release(*LENSES)
    if scope.ask(eos.GetFunctionMode)[0] != mode:
        scope.change(eos.SelectFunctionMode, mode)
        scope.wait(scope.mode_settle_s)
    if not scope.dry_run:
        if scope.ask(eos.GetFunctionMode)[0] != mode:
            raise RuntimeError(f"Mode {mode} was not selected")
        for _ in range(60):
            if any(scope.ask(getattr(lens, "Get" + key)) != console[key] for key in ("OLc", "OLf", "OM")):
                raise RuntimeError("Objective changed on row selection. Start with MAG and DIFF in the objective-on range.")
            at = row_label(mode)
            if at == label:
                break
            up = at < label
            scope.change(eos.UpSelector if up else eos.DownSelector)
            scope.wait(scope.mode_settle_s)
            now = row_label(mode)
            if now == at:
                break
            if (at < label < now) or (now < label < at):
                if not exact and abs(at - label) < abs(now - label):
                    scope.change(eos.DownSelector if up else eos.UpSelector)
                    scope.wait(scope.mode_settle_s)
                break
        else:
            raise RuntimeError(f"Selector did not converge on {label}")
        selected = row_label(mode)
        if exact and selected != label:
            raise RuntimeError(f"Requested row {label}, reached {selected}")
        if selected != label:
            print(f"Using nearest row {selected} for requested {label}")
    values = {name: scope.ask(getattr(lens, "Get" + name)) for name in LENSES}
    scope.take_over(*LENSES)
    set_from_below(**values)
    pla_neutral()
    return label if scope.dry_run else selected


def use(camera):
    """Frames from now on come from this camera: the screen is moved to it (PyJEM setScreen), then 4 s wait.
    With --by-hand the script asks you to change the camera instead. In a dry run frames stay on the screen camera."""
    global dry_run_camera
    if scope.dry_run:
        if camera != dry_run_camera:
            if BY_HAND:
                print(f"   (here you would be asked to put the {camera} in the beam)")
            else:
                scope.set_screen(SCREEN_AT[camera])        # prints the call it would make
        dry_run_camera = camera
        return
    scope.use_camera(camera, by_hand=BY_HAND and not SIMULATED)


def on_camera(beam):
    """The whole beam on the camera, and a real beam: it has at least 30 % of the most signal seen on this camera
    (not a speck of noise)."""
    brightest[scope.camera_name] = max(brightest.get(scope.camera_name, 0.0), beam.signal)
    return beam.usable and beam.signal > 0.3 * brightest[scope.camera_name]


def size_beam(smallest, largest):
    """Turn CL3 until the beam's rms radius is between `smallest` and `largest` px. CL3 moves in steps, starting at
    250 units: a step that brings the beam nearer that size is kept and the next one is twice as big (at most 3000); a
    step that does not is not kept, and the next one goes the other way, half as big. Returns the beam."""
    def how_far_off(beam):                     # 0: the right size. A beam not wholly on the camera: 9 to 10
        if not on_camera(beam):
            return 10 - beam.signal / max(brightest[scope.camera_name], 1e-9)
        return max(0.0, np.log(smallest / beam.rms), np.log(beam.rms / largest))

    value, beam, step = scope.ask(lens.GetCL3), scope.measure(), 250
    for _ in range(30):
        if how_far_off(beam) == 0 or abs(step) < 20:
            break
        trial = int(np.clip(value + step, DAC_MIN, DAC_MAX))
        scope.set_lenses(CL3=trial)
        new = scope.measure()
        if how_far_off(new) < how_far_off(beam):
            value, beam, step = trial, new, int(np.clip(2 * step, -3000, 3000))
        else:
            step = -step // 2
    scope.set_lenses(CL3=value)
    return beam


def cla(deflector):
    return tuple(scope.ask(getattr(scope.deflectors, "Get" + deflector)))


def response(deflector):
    """How far one unit of the deflector moves the beam along its x and its y (px, x + iy), from trial steps that
    are not kept: along x 100, 200, 400, ... units until the beam moves 10 px (or the largest kick), then the same
    along y (+, or - if + takes the beam off the camera)."""
    x, y = cla(deflector)
    start = scope.measure().centre
    along_x, along_y, used = 0j, 0j, 0
    for kick in [k for k in (100, 200, 400, 800, 1500, 3000, 6000, 12000) if k <= LARGEST_KICK[deflector]]:
        scope.set_deflector(deflector, x + kick, y)
        beam = scope.measure()
        if not on_camera(beam):
            break
        along_x, used = (beam.centre - start) / kick, kick
        if abs(beam.centre - start) >= 10:
            break
    for sign in (+1, -1):
        if used:
            scope.set_deflector(deflector, x, y + sign * used)
            beam = scope.measure()
            if on_camera(beam):
                along_y = sign * (beam.centre - start) / used
                break
    scope.set_deflector(deflector, x, y)
    return along_x, along_y or -1j * along_x                  # y unmeasurable: taken at right angles to x


def centre_beam(deflector):
    """Bring the beam near the middle (within 6 % of the camera width) with this deflector: from its response, the
    change that would put the beam in the middle; a change that makes it worse is undone. Three tries at most."""
    for _ in range(3):
        beam = scope.measure()
        if abs(beam.centre) <= 0.06 * scope.frame_shape[1]:
            break
        along_x, along_y = response(deflector)
        if along_x == 0:
            break
        change = np.linalg.solve([[along_x.real, along_y.real], [along_x.imag, along_y.imag]],
                                 [-beam.centre.real, -beam.centre.imag])
        change *= min(1.0, LARGEST_KICK[deflector] / max(np.abs(change).max(), 1))
        x, y = cla(deflector)
        scope.set_deflector(deflector, np.clip(x + change[0], DAC_MIN, DAC_MAX), np.clip(y + change[1], DAC_MIN, DAC_MAX))
        moved = scope.measure()
        if not on_camera(moved) or abs(moved.centre) > abs(beam.centre):
            scope.set_deflector(deflector, x, y)
            break


def record_kicks(deflector, **about):
    """Four frames kept: the deflector + and - along x, + and - along y, by a kick that moves the beam about a fifth
    of the camera (at most the largest kick). Returns the kick and how far it moved the beam (px)."""
    along_x, along_y = response(deflector)
    per_unit = max(abs(along_x), abs(along_y), 1e-9)
    kick = int(np.clip(min(60, 0.2 * scope.frame_shape[1]) / per_unit, 20, LARGEST_KICK[deflector]))
    x, y = cla(deflector)
    centres = {}
    for direction, dx, dy in [("x", kick, 0), ("y", 0, kick)]:
        for sign in (+1, -1):
            scope.set_deflector(deflector, x + sign * dx, y + sign * dy)
            beam = scope.record(stage=stage, role="deflected", deflector=deflector, direction=direction, sign=sign,
                                kick=kick, **about)
            centres[direction, sign] = beam.centre if beam.usable else np.nan
    scope.set_deflector(deflector, x, y)
    return kick, max(abs(centres["x", 1] - centres["x", -1]), abs(centres["y", 1] - centres["y", -1])) / 2


def kicks_here(setting, camera, kind, mode=MAG, ask=True, **about):
    """At the lens setting the microscope is in now, on this camera: the beam sized and centred, one frame not
    deflected, then the CLA1 and CLA2 kicks: 9 frames kept. `kind` (a label such as "MAG" or "DIFF"): CL3 starts where it last
    worked for that kind of setting on this camera. With no whole beam: the real run stops and asks you to bring it
    on (not with ask=False). Returns True if the 9 frames were taken."""
    use(camera)
    if scope.dry_run:
        print(f"   {setting} on the {camera}: 9 frames (not taken: dry run)")
        return True
    if (camera, kind) in cl3_that_worked:
        scope.set_lenses(CL3=cl3_that_worked[camera, kind])
    largest_rms = 0.06 * scope.frame_shape[1]
    beam = size_beam(3, largest_rms)
    if not on_camera(beam) and FOR_REAL and ask:
        print(f"\n   what the {camera} shows: {describe(beam)}")
        input(f">>> No whole beam on the {camera} at {setting}: bring it on with CL3, CLA1 or CLA2, then press Enter ")
        beam = size_beam(3, largest_rms)
    if not on_camera(beam):
        print(f"   {setting} on the {camera}: NOT MEASURED: {describe(beam)}")
        return False
    centre_beam(STEER[mode])
    beam = size_beam(3, largest_rms)
    cl3_that_worked[camera, kind] = scope.ask(lens.GetCL3)
    about = dict(setting=setting, **about)
    scope.record(stage=stage, role="not deflected", **about)
    moved = {name: record_kicks(name, **about) for name in ("CLA1", "CLA2")}
    print(f"   {setting} on the {camera}: {describe(beam)}; CL3 {cl3_that_worked[camera, kind]}; "
          + "; ".join(f"{name} kick {kick} moved it {px:.0f} px" for name, (kick, px) in moved.items()))
    return True


# %% [markdown]
# ## The microscope as found, and both cameras
#
# Leave the stored DIFF selector on an objective-on row (8–200 cm on the NEOARM), then start in MAG
# (8k–500k) with the beam on the screen camera and an empty hole of the specimen in the beam.

# %%
if scope.ask(eos.GetFunctionMode)[0] != MAG:
    raise RuntimeError("put the microscope in MAG mode at the console, then run this again")
console = scope.remember_console("at the start")
mag_at_start, spot_at_start = scope.ask(eos.GetMagValue)[0], console["spot"]
print(f"MAG {mag_at_start}x, spot index {spot_at_start}, CL3 {console['CL3']}, IL1-PL1 {[console[n] for n in LENSES]}")
print("CLA1, CLA2, PLA:", scope.deflectors_at_start)
try:
    print("cameras PyJEM can read:", scope.attached_cameras())
except Exception as error:
    print("this PyJEM cannot list its cameras:", error)
screen_at_start = scope.get_screen()
print(f"the screen is at {screen_at_start} (0 lower screen, 1 SightX view camera, 2 Merlin); it is put back there at the end")
scope.add_camera(SCREEN)
scope.cameras[SCREEN]["screen"] = None if BY_HAND else SCREEN_AT[SCREEN]
try:
    scope.add_camera(MERLIN, above=[SCREEN], screen=None if BY_HAND else SCREEN_AT[MERLIN])                  # the Merlin, if PyJEM knows it by that name
except Exception:
    import merlin_camera
    print(f"PyJEM has no camera called '{MERLIN}': merlin_camera.py reads it")
    scope.add_camera(MERLIN, reader=merlin_camera.Merlin(), above=[SCREEN], screen=None if BY_HAND else SCREEN_AT[MERLIN])
    scope.cameras[MERLIN].update(frame_shape=merlin_camera.FRAME_SHAPE, frame_dtype=np.dtype("<u2"))
scope.console["about this recording"] = dict(simulated=SIMULATED, cameras=[SCREEN, MERLIN], pl1_steps=PL1_STEPS,
                                             approach=APPROACH, largest_kick=LARGEST_KICK, quick=QUICK)
scope.measure_dark_level()
frame = scope.take_frame()
print(f"the {SCREEN}: frames of {frame.shape[1]} x {frame.shape[0]} px, {frame.dtype}, values {frame.min()} to {frame.max()}")
if scope.simulated:
    print("   (simulated: the virtual NEOARM gives signed 32 bit, the real one unsigned 16 bit)")
elif (frame.shape, frame.dtype) != (SCREEN_FRAME[0], np.dtype(SCREEN_FRAME[1])):
    raise RuntimeError(f"the {SCREEN} gives {frame.shape} {frame.dtype}, not {SCREEN_FRAME[0]} {SCREEN_FRAME[1]}: "
                       "change FRAME_SHAPE and FRAME_DTYPE in neoarm.py and SCREEN_FRAME above, or the binning")
print(describe(scope.measure()))

# %%
use(MERLIN)
if not scope.dry_run:
    scope.measure_dark_level()
    frame = scope.take_frame()
    print(f"the {MERLIN}: frames of {frame.shape[1]} x {frame.shape[0]} px, {frame.dtype}, values {frame.min()} to {frame.max()}")
    if not SIMULATED and (frame.shape, frame.dtype) != (MERLIN_FRAME[0], np.dtype(MERLIN_FRAME[1])):
        print(f"   WARNING: expected {MERLIN_FRAME[0]} {MERLIN_FRAME[1]}: check merlin_camera.py and MERLIN_FRAME above")
    beam = scope.measure()
    print(describe(beam))
    if not beam.found and not BY_HAND:
        raise RuntimeError(f"no beam on the {MERLIN} with the screen at {scope.screen_now}: is the screen really up, "
                           "and is the beam where the Merlin can see it? Stopped before recording anything.")
use(SCREEN)

# %%
scope.take_over("CL3")
pla_neutral()
for camera in (SCREEN, MERLIN):                # a normal row's beam is looked for first at the console's brightness
    cl3_that_worked[camera, "MAG"] = cl3_that_worked[camera, "DIFF"] = console["CL3"]

# %% [markdown]
# ## 1. Normal rows (screen camera; MAG 8000 and DIFF 40 cm on the Merlin too)
#
# A two-camera anchor requires its exact row. For an ordinary row the closest available label goes in the table.

# %%
stage = "rows"
for mode, label in ROWS:
    label = go_to_row(mode, label, exact=(mode, label) in ON_THE_MERLIN_TOO)
    kind = "MAG" if mode == MAG else "DIFF"
    if (mode, label) in ON_THE_MERLIN_TOO:
        kicks_here(f"{kind} {label}", MERLIN, kind, mode, mode_row=label)
    kicks_here(f"{kind} {label}", SCREEN, kind, mode, mode_row=label)

# %% [markdown]
# ## 2. PL1 alone turned down, at MAG 8000 and at DIFF 40 cm, on both cameras
#
# IL1-IL3 stay at the row's values. PL1 comes to each step from below. With PL1 lower the picture is smaller, so the
# beam should stay on the cameras. If there is no whole beam on a camera at a step, the script prints NOT MEASURED,
# leaves the lower steps of that row out and goes on: it does not wait for you.

# %%
stage = "pl1"
for mode, label in PL1_ROWS:
    label = go_to_row(mode, label, exact=True)
    kind = "MAG" if mode == MAG else "DIFF"
    for value in PL1_STEPS:
        set_from_below(PL1=value)
        setting = f"{kind} {label}, PL1 {value}"
        seen = [kicks_here(setting, camera, f"{kind} PL1 {value}", mode, ask=False, mode_row=label, lens="PL1", value=value)
                for camera in (MERLIN, SCREEN)]
        if not any(seen):
            print(f"   no beam on either camera at {setting}: the lower PL1 steps of this row are left out")
            break

# %% [markdown]
# ## 3. Kept-back rows: never fitted, the model has to predict them (only with --kept)

# %%
stage = "kept"
for mode, label in (KEPT if WITH_KEPT else []):
    label = go_to_row(mode, label, exact=True)
    kind = "MAG" if mode == MAG else "DIFF"
    kicks_here(f"{kind} {label}", SCREEN, kind, mode, mode_row=label)

# %% [markdown]
# ## 4. The rulers: gold rings (real radians) and a grating image (real metres)
#
# Move the cross-grating with gold into the beam first. For the rings the beam is made small (a central disc of 1-4
# px rms), so the rings are sharp. For the grating the beam is spread over the whole camera, because the period is
# measured over the whole frame.

# %%
stage = "rulers"
if SIMULATED:
    from temgym_core.components import CrossGrating, PolycrystallineFilm
elif FOR_REAL:
    input("\n>>> Move the cross-grating with gold into the beam, then press Enter ")
for label, camera in GOLD:
    label = go_to_row(DIFF, label, exact=True)
    use(camera)
    if SIMULATED:
        scope.microscope.specimen = PolycrystallineFilm.gold()
    if scope.dry_run:
        print(f"   gold rings at DIFF {label} cm on the {camera}: 1 frame (not taken: dry run)")
        continue
    scope.set_lenses(CL3=cl3_that_worked[camera, "DIFF"])
    size_beam(1, 4)
    centre_beam("CLA2")
    disc = scope.record(stage=stage, role="gold rings", setting=f"DIFF {label}", mode_row=label)
    radius, width = find_rings(np.load(scope.folder / scope.rows[-1]["frame"]), disc.centre, disc.rms)
    scope.rows[-1].update(ring_radius_px=round(radius, 3), ring_width_px=round(width, 2))
    print(f"   gold at DIFF {label} cm on the {camera}: the 111 ring is {radius:.2f} px in radius (disc rms {disc.rms:.1f} px)")

# %%
use(SCREEN)
for label in GRATING:
    label = go_to_row(MAG, label, exact=True)
    if SIMULATED:
        scope.microscope.specimen = CrossGrating(period=1e-3 / 2160)
    if scope.dry_run:
        print(f"   grating at MAG {label}x: 1 frame (not taken: dry run)")
        continue
    value = console["CL3"]
    for _ in range(14):                        # CL3 up 1000 at a time until the beam is wider than the camera
        scope.set_lenses(CL3=value)
        beam = scope.measure()
        if beam.found and beam.touches_edge and beam.rms > 0.25 * scope.frame_shape[1]:
            break
        value = min(value + 1000, DAC_MAX)
    scope.record(stage=stage, role="grating", setting=f"MAG {label}", mode_row=label)
    pitch, angle = find_pitch(np.load(scope.folder / scope.rows[-1]["frame"]))
    scope.rows[-1].update(grating_pitch_px=round(pitch, 3), grating_angle_rad=round(angle, 4))
    print(f"   grating at MAG {label}x: one period is {pitch:.2f} px")
if SIMULATED:
    scope.microscope.specimen = None

# %% [markdown]
# ## The end: the microscope as it was at the start, and the recording saved

# %%
scope.restore()                                # free-lens control off, CLA1, CLA2, PLA back
scope.change(eos.SelectSpotSize, spot_at_start)
if scope.ask(eos.GetFunctionMode)[0] != MAG:
    scope.change(eos.SelectFunctionMode, MAG)
    scope.wait(scope.mode_settle_s)
for _ in range(60):
    at = scope.ask(eos.GetMagValue)[0]
    if at == mag_at_start or scope.dry_run:
        break
    scope.change(eos.UpSelector if at < mag_at_start else eos.DownSelector)
    scope.wait(scope.mode_settle_s)
scope.save()
for name in dict.fromkeys(row["stage"] for row in scope.rows):
    mine = [row for row in scope.rows if row["stage"] == name]
    print(f"   {name:<8}{len(mine):>4} frames: " + ", ".join(f"{sum(r['camera'] == c for r in mine)} on the {c}"
                                                       for c in (SCREEN, MERLIN)))
