"""Record what a predictor of the microscope needs: one recording, five stages.

    survey      every lens value at every magnification row, camera length and spot size (no frames)
    history     the same lens values reached from below and from above, after small and large swings
    kicks       how far CLA1 and CLA2 displace the beam, at every magnification row and camera length, and with
                IL1-PL1 changed at a few rows
    deflectors  is the displacement proportional to the kick? does it change with CL3 or the spot size?
    condenser   the beam itself, in MAG and in DIFF, over spot size x CL3 (and the aperture holes, with --holes)

No specimen is needed. Everything is in DAC units and camera pixels: nothing here is in metres or radians.

Every frame is a row of table.csv with every lens and deflector value (see neoarm.py), so the order in which the
lenses took their values is in the table. A kick is always + and - along x and along y, so where the beam started
cancels. Before the frames of a setting, IL1-PL1 are brought to their values from APPROACH DAC units below, so
every setting is reached the same way.

Run it from this folder:
    python record_predictor.py --simulated                  the virtual NEOARM (needs TemGym), every stage
    python record_predictor.py --simulated pilot            fixed-objective acquisition for the blind predictor notebook
    python record_predictor.py --simulated pilot --compact  two lens-wobble anchors; eight nominal rows on the simulator
    python record_predictor.py --for-real pilot             same recipe on the NEOARM, starting at MAG 40k, alpha 0
        First park DIFF at 40 cm, then return to MAG 40k. Switching modes recalls each mode's last row.
    python record_predictor.py                              the real NEOARM, a dry run: reads it, changes nothing
    python record_predictor.py --for-real kicks condenser   the real NEOARM, these stages only
    add --holes to also change the condenser aperture's hole (a motor: check its alignment afterwards)
    add --quick for a short version of every stage (a rehearsal of the script itself)
"""

import sys

import numpy as np

from neoarm import DAC_MAX, DAC_MIN, FREE_LENS, Neoarm, describe

STAGES = ["survey", "history", "kicks", "deflectors", "condenser"]
stages = ["pilot"] if "pilot" in sys.argv else ([s for s in STAGES if s in sys.argv] or STAGES)
COMPACT = "--compact" in sys.argv
if COMPACT and stages != ["pilot"]:
    raise SystemExit("--compact is a version of the pilot stage; include pilot in the command.")
QUICK = "--quick" in sys.argv
PROJECTION_LENSES = ["IL1", "IL2", "IL3", "PL1"]
MAG, DIFF = 0, 4                               # PyJEM's function modes

APPROACH = 2000                                # a lens comes to its value from this many DAC units below
DISPLACEMENT_PX = 60                           # a kick should displace the beam about this far ...
DISPLACEMENT_RANGE_PX = (20, 150)              # ... and is done again if it lands outside this
LARGEST_KICK = 12000
WEAK_KICK = 1500                               # for the deflector that hardly displaces the beam in a mode
BEAM_RMS_PX = (8, 45)                          # the beam's rms radius is kept inside this with CL3 ...
BEAM_AIM_PX = 20                               # ... aiming here
LARGEST_CL3_STEP = 3000
LENS_STEPS = [-1000, 1000]                     # IL1-PL1 changed by these at the rows below
ROWS_WITH_LENS_STEPS = {MAG: [8000, 40000, 200000], DIFF: [40, 100]}
SWINGS = [300, 3000, 12000]                    # history: how far a lens goes away before it comes back
CL3_OFFSETS = [-400, -200, -100, 0, 100, 200, 400, 800, 1600, 3200, 6400, 10000]      # condenser: from the spot's CL3
SPOTS = [0, 1, 2, 3, 4]                        # PyJEM spot-size indices (spot sizes 1-5)
HOLES = [2, 4]                                 # with --holes: these too, at the console's spot
if QUICK:
    LENS_STEPS, SWINGS, CL3_OFFSETS, SPOTS = [1000], [3000], [-200, 0, 400, 3200], [1, 2]
    ROWS_WITH_LENS_STEPS = {MAG: [40000], DIFF: [40]}

scope = Neoarm("predictor", for_real="--for-real" in sys.argv, simulated="--simulated" in sys.argv,
               full_readback=stages == ["pilot"])
eos, lens = scope.eos, scope.lens
neutral = (DAC_MIN + DAC_MAX + 1) // 2
response = {}                                  # (mode, row label, deflector) -> px per DAC unit along its x and its y (each x + iy)
stage = "start"


# ---------------------------------------------------------------------------------------------------------------------
# Rows of the selector
# ---------------------------------------------------------------------------------------------------------------------
def row():
    """The selected row's label: the magnification, or the camera length in cm."""
    value, unit, _ = scope.ask(eos.GetMagValue)
    if unit == "mm":
        return value / 10
    if unit == "m":
        return value * 100
    return int(value)


def select_mode(mode):
    if scope.ask(eos.GetFunctionMode)[0] != mode:
        scope.release(*PROJECTION_LENSES)
        scope.change(eos.SelectFunctionMode, mode)
        scope.wait(scope.mode_settle_s)


def check_objective():
    if stages == ["pilot"]:
        for name in ("OLc", "OLf", "OM"):
            if scope.ask(getattr(lens, "Get" + name)) != console[name]:
                raise RuntimeError(f"{name} changed at row {row()}; return to the fixed-objective range.")


def all_rows(mode):
    """Every row's label in a mode, lowest first (the selector is stepped down to its end, then up to its end)."""
    select_mode(mode)
    scope.release(*PROJECTION_LENSES)

    def step_to_the_end(step):
        seen = [row()]
        while not scope.dry_run:
            scope.change(step)
            scope.wait(scope.mode_settle_s)
            if row() == seen[-1]:
                break
            seen.append(row())
        return seen

    step_to_the_end(eos.DownSelector)
    return step_to_the_end(eos.UpSelector)


def go_to_row(mode, label):
    """Select a row and bring IL1-PL1 to its values from below."""
    select_mode(mode)
    check_objective()
    scope.release(*PROJECTION_LENSES)
    for _ in range(60):
        at = row()
        if at == label or scope.dry_run:
            break
        scope.change(eos.UpSelector if at < label else eos.DownSelector)
        scope.wait(scope.mode_settle_s)
        check_objective()
        if row() == at:
            raise RuntimeError(f"the selector does not reach {label}: it stops at {at}")
    if not scope.dry_run and (row() != label or scope.ask(eos.GetFunctionMode)[0] != mode):
        raise RuntimeError(f"Requested mode {mode}, row {label}; reached {scope.ask(eos.GetFunctionMode)}, row {row()}")
    values = {name: scope.ask(getattr(lens, "Get" + name)) for name in PROJECTION_LENSES}
    scope.take_over(*PROJECTION_LENSES)
    set_from_below(**values)
    return values


def set_from_below(**values):
    """Lenses to these values, each coming up from APPROACH below its value."""
    scope.set_lenses(**{name: max(DAC_MIN, value - APPROACH) for name, value in values.items()})
    scope.set_lenses(**values)


# ---------------------------------------------------------------------------------------------------------------------
# The beam: on the camera, a sensible size, near the middle
# ---------------------------------------------------------------------------------------------------------------------
def cl3():
    return scope.ask(lens.GetCL3)


def cla(name):
    return tuple(scope.ask(getattr(scope.deflectors, "Get" + name)))


def fits(beam):
    return beam.usable and BEAM_RMS_PX[0] <= beam.rms <= BEAM_RMS_PX[1]


def size_beam(most_steps=14):
    """Turn CL3 until the beam's rms radius is inside BEAM_RMS_PX.

    While the beam is bigger than the camera its radius cannot be measured, but the signal on the camera can: it
    grows as the beam shrinks. So CL3 is first stepped the way the signal grows (steps doubling) until the beam is
    wholly on the camera. Then each step assumes the radius changes in proportion to CL3, with the slope of the last
    two frames.
    """
    beam = scope.measure()
    if fits(beam):
        return beam
    value, slope, step = cl3(), None, 100

    def go(change):
        nonlocal value
        value = int(np.clip(value + change, DAC_MIN, DAC_MAX))
        scope.set_lenses(CL3=value)
        new = scope.measure()
        print(f"      CL3 {value}: {describe(new)}")
        return new

    for _ in range(most_steps):
        if fits(beam):
            break
        if beam.touches_edge or not beam.found:                     # bigger than the camera: follow the signal
            new = go(step)
            if new.signal < beam.signal and (new.touches_edge or not new.found):
                step = -step                                         # the wrong way: turn round
            else:
                step = int(np.clip(2 * step, -LARGEST_CL3_STEP, LARGEST_CL3_STEP))
            beam, slope = new, None
            continue
        if slope is None:                                            # on the camera: a small trial for the slope
            new = go(50)
            slope = (new.rms - beam.rms) / 50 if abs(new.rms - beam.rms) > 0.2 else None
            beam = new
            if slope is None:
                continue
        aim = max(BEAM_AIM_PX, beam.rms / 3)                         # never shrink it more than three times in one step
        change = int(np.clip((aim - beam.rms) / slope, -LARGEST_CL3_STEP, LARGEST_CL3_STEP)) or 20
        new = go(change)
        if abs(new.rms - beam.rms) > 0.5 and not new.touches_edge:
            slope = (new.rms - beam.rms) / change
        beam = new
    return beam


def expected_response(mode, label, name):
    """px per DAC unit of a deflector at a row: as measured there, else from the nearest row measured in this mode,
    scaled by the labels (the response grows with magnification and camera length); None if none is measured."""
    if (mode, label, name) in response:
        return response[(mode, label, name)]
    measured = [l for (m, l, n) in response if m == mode and n == name]
    if not measured:
        return None
    nearest = min(measured, key=lambda l: abs(np.log(l / label)))
    along_x, along_y = response[(mode, nearest, name)]
    return along_x * label / nearest, along_y * label / nearest


def units_for(displacement, per_unit):
    """The deflector change (x, y in DAC units) that displaces the beam by ``displacement`` (px, x + iy)."""
    along_x, along_y = per_unit
    return np.linalg.solve([[along_x.real, along_y.real], [along_x.imag, along_y.imag]],
                           [displacement.real, displacement.imag])


def search_for_beam(mode, label):
    """No beam on the camera: step the beam over a grid, 150 px apart, until it is (CLA1 in MAG, CLA2 in DIFF)."""
    name = "CLA2" if mode == DIFF else "CLA1"
    per_unit = expected_response(mode, label, name)
    if per_unit is None:
        raise RuntimeError("no beam on the camera, and no row measured yet to search from")
    x, y = cla(name)
    for reach in (1, 2, 3):
        for i in range(-reach, reach + 1):
            for j in range(-reach, reach + 1):
                if max(abs(i), abs(j)) != reach:
                    continue
                step = units_for(150 * (i + 1j * j), per_unit)
                if not (DAC_MIN <= x + step[0] <= DAC_MAX and DAC_MIN <= y + step[1] <= DAC_MAX):
                    continue
                scope.set_deflector(name, x + step[0], y + step[1])
                beam = scope.measure()
                if beam.found:
                    print(f"      the beam was off the camera: found {150 * abs(i + 1j * j):.0f} px away")
                    return beam
    scope.set_deflector(name, x, y)
    raise RuntimeError("no beam on the camera within 450 px")


def centre_beam(mode, label, most_steps=4):
    """Bring the beam within 15 px of the middle with CLA1 (in DIFF: with CLA2), using this row's response."""
    name = "CLA2" if mode == DIFF else "CLA1"
    for _ in range(most_steps):
        beam = scope.measure()
        if not beam.found:
            beam = search_for_beam(mode, label)
        if abs(beam.centre) <= 15:
            return beam
        # A distant row can have a different image rotation. Measure the local x,y directions
        # on the first usable off-centre frame before inverting the response.
        per_unit = response.get((mode, label, name))
        if per_unit is None and beam.usable and not beam.saturated:
            per_unit = measure_response(mode, label, name)
        if per_unit is None:
            per_unit = expected_response(mode, label, name)
        if per_unit is None:
            raise RuntimeError(f"No usable local response to centre the beam at {label}")
        step = units_for(-beam.centre, per_unit)
        step = step * min(1.0, LARGEST_KICK / max(np.abs(step).max(), 1))
        x, y = cla(name)
        scope.set_deflector(name, np.clip(x + step[0], DAC_MIN, DAC_MAX), np.clip(y + step[1], DAC_MIN, DAC_MAX))
    return scope.measure()


def measure_response(mode, label, name):
    """px per DAC unit of a deflector along its x and its y, found with trial kicks that grow until the beam is
    displaced 10 px (not recorded). Used only where no response is known yet."""
    x, y = cla(name)
    limit = min(x, y, DAC_MAX - x, DAC_MAX - y, LARGEST_KICK)
    start = scope.measure()
    if limit < 1 or not start.usable or start.saturated:
        raise RuntimeError(f"{name} needs a beam entirely on camera and room for a DAC perturbation")
    kick = min(limit, 50)
    try:
        for _ in range(12):
            scope.set_deflector(name, x + kick, y)
            bx = scope.measure()
            scope.set_deflector(name, x, y + kick)
            by = scope.measure()
            if not (bx.usable and by.usable) or bx.saturated or by.saturated:
                limit, kick = kick - 1, kick // 2
                if kick == 0:
                    raise RuntimeError(f"{name}: no usable trial kick")
                continue
            along_x, along_y = bx.centre - start.centre, by.centre - start.centre
            displacement = min(abs(along_x), abs(along_y))
            if displacement >= 10 or (kick == limit and displacement >= 1):
                break
            if kick == limit:
                raise RuntimeError(f"{name}: response below one pixel throughout the available DAC range")
            kick = int(min(limit, kick * 30 / max(displacement, 1)))
        else:
            raise RuntimeError(f"{name}: no usable response after 12 trials")
    finally:
        scope.set_deflector(name, x, y)
    response[(mode, label, name)] = (along_x / kick, along_y / kick)
    return response[(mode, label, name)]


def prepare(mode, label):
    """The beam on the camera, sized (MAG only: in DIFF its size is its spread of angles) and centred."""
    if not scope.measure().found:
        search_for_beam(mode, label)
    if mode == MAG:
        size_beam()                 # first: a beam bigger than the camera cannot be centred
        centre_beam(mode, label)
        size_beam()
    return centre_beam(mode, label)


# ---------------------------------------------------------------------------------------------------------------------
# A kick: + and - along x, + and - along y
# ---------------------------------------------------------------------------------------------------------------------
def kick_size(mode, label, name):
    """A kick that should displace the beam DISPLACEMENT_PX, from the response expected at this row (trial kicks
    only if no row has been measured yet)."""
    known = expected_response(mode, label, name)
    if known is None:
        known = measure_response(mode, label, name)
    return int(np.clip(DISPLACEMENT_PX / max(abs(known[0]), 1e-9), 20, LARGEST_KICK))


def record_kick(name, kick, axes="xy", **about):
    """Record a signed pair per requested axis. Return pixel displacement per DAC and the actual kick."""
    x, y = cla(name)
    for axis in axes:
        value = {"x": x, "y": y}[axis]
        kick = int(min(kick, value - DAC_MIN, DAC_MAX - value))
    if kick <= 0:
        raise ValueError(f"{name} is on a DAC limit at ({x}, {y}); a signed pair cannot be recorded")
    centres = {}
    for direction in axes:
        dx, dy = (kick, 0) if direction == "x" else (0, kick)
        for sign in (+1, -1):
            scope.set_deflector(name, x + sign * dx, y + sign * dy)
            beam = scope.record(stage=stage, role="deflected", deflector=name, direction=direction, sign=sign,
                                kick=kick, **about)
            centres[direction, sign] = beam.centre if beam.usable and not beam.saturated else np.nan
    scope.set_deflector(name, x, y)
    return tuple((centres[d, +1] - centres[d, -1]) / (2 * kick) for d in axes), kick


def kick_both(mode, label, **about):
    """CLA1 and CLA2 kicks at the setting the microscope is in. The deflector that displaces the beam in this mode
    (CLA1 in MAG, CLA2 in DIFF) gets a kick sized for DISPLACEMENT_PX, done again once if it lands outside
    DISPLACEMENT_RANGE_PX; the other gets WEAK_KICK."""
    strong, weak = ("CLA2", "CLA1") if mode == DIFF else ("CLA1", "CLA2")
    kick = kick_size(mode, label, strong)
    for attempt in range(2):
        per_unit, used = record_kick(strong, kick, mode_row=label, **about)
        displaced = abs(per_unit[0]) * used
        if np.isfinite(displaced) and np.isfinite(abs(per_unit[1])):
            response[(mode, label, strong)] = per_unit
        if DISPLACEMENT_RANGE_PX[0] <= displaced <= DISPLACEMENT_RANGE_PX[1] or attempt:
            break
        kick = int(np.clip(used * DISPLACEMENT_PX / displaced, 20, LARGEST_KICK)) if displaced > 0 else used // 3
        print(f"      {strong} kick {used} displaced the beam {displaced:.0f} px: again with {kick}")
    record_kick(weak, WEAK_KICK, mode_row=label, **about)
    measured = response.get((mode, label, strong))
    print(f"   {'DIFF' if mode == DIFF else 'MAG'} {label}: {strong} "
          + (f"{abs(measured[0]):.4f} px per unit" if measured is not None else "NOT MEASURED: the beam was not usable"))


# ---------------------------------------------------------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------------------------------------------------------
if scope.ask(eos.GetFunctionMode)[0] != MAG:
    raise SystemExit("put the microscope in MAG mode at the console, then start again")
console = scope.remember_console("at the start")
mag_at_start, spot_at_start = row(), console["spot"]
if stages == ["pilot"]:
    if (mag_at_start, console["alpha"]) != (40000, 0) or not 0 <= spot_at_start <= 4:
        raise SystemExit("Start the pilot at MAG 40k, alpha 0, a spot index from 0 to 4.")
    for name in ("CL3", *PROJECTION_LENSES):
        if scope.ask(lens.GetFLCInfo, FREE_LENS[name])[0]:
            raise SystemExit(f"Release {name} free-lens control before starting the pilot.")
    if scope.ask(scope.deflectors.GetBeamBlank):
        raise SystemExit("Start the pilot with the beam visible on the SightX.")
print(f"at the start: MAG {mag_at_start}x, spot index {spot_at_start}, CL3 {console['CL3']}")
scope.measure_dark_level()
print("the beam now:", describe(scope.measure()))
if scope.dry_run:
    raise SystemExit("\nDry run: the connection and the camera work. Run with --for-real to record.")


def select_spot(spot):
    """A spot size, with CL3 following the console's table for it (then under free-lens control again)."""
    scope.release("CL3")
    scope.change(eos.SelectSpotSize, spot)
    scope.wait(scope.lens_settle_s)
    scope.take_over("CL3")


def survey():
    """Every lens value at every row of MAG and DIFF and at every spot size. No frames."""
    for mode, name in [(MAG, "MAG"), (DIFF, "DIFF")]:
        for label in all_rows(mode):
            go_to_row(mode, label)
            scope.release(*PROJECTION_LENSES)
            scope.remember_console(f"{name} {label}")
    select_mode(MAG)
    go_to_row(MAG, mag_at_start)
    scope.release("CL3")
    for spot in range(8):
        scope.change(eos.SelectSpotSize, spot)
        scope.wait(scope.lens_settle_s)
        scope.remember_console(f"spot index {spot}")
    scope.change(eos.SelectSpotSize, spot_at_start)
    scope.take_over("CL3")
    print(f"survey: {len(scope.console)} console states read")


def history():
    """At one row: each lens goes away by a swing and comes back, from below and from above; after each, the beam
    not deflected and a CLA1 kick. Then the magnification is changed and changed back."""
    values = go_to_row(MAG, mag_at_start)
    prepare(MAG, mag_at_start)
    kick = kick_size(MAG, mag_at_start, "CLA1")

    def frames(**about):
        scope.record(stage=stage, role="not deflected", mode_row=mag_at_start, **about)
        record_kick("CLA1", kick, mode_row=mag_at_start, **about)

    frames(reached="start")
    for name in PROJECTION_LENSES:
        for swing in SWINGS:
            for side, sign in [("from below", -1), ("from above", +1)]:
                away = int(np.clip(values[name] + sign * swing, DAC_MIN, DAC_MAX))
                scope.set_lenses(**{name: away})
                scope.set_lenses(**{name: values[name]})
                frames(reached=side, lens=name, swing=swing)
        set_from_below(**values)
    rows = all_rows(MAG)
    for other in (rows[0], rows[-1]):
        go_to_row(MAG, other)
        go_to_row(MAG, mag_at_start)
        frames(reached="after a magnification change", swing=other)
    print("history: done")


def kicks():
    """CLA1 and CLA2 kicks at every magnification row and camera length; at a few rows also with IL1-PL1 changed."""
    for mode in (MAG, DIFF):
        labels = all_rows(mode)
        if QUICK:
            labels = [l for l in labels if (20000 <= l <= 60000 if mode == MAG else 25 <= l <= 50)]
        if mode == DIFF:
            scope.set_lenses(CL3=console["CL3"])                     # the console's brightness: a small disc
        for label in labels:
            values = go_to_row(mode, label)
            beam = prepare(mode, label)
            if not beam.usable:
                print(f"   {'DIFF' if mode == DIFF else 'MAG'} {label}: NOT MEASURED: {describe(beam)}")
                continue
            kick_both(mode, label, setting="row")
            if label in ROWS_WITH_LENS_STEPS[mode]:
                for name in PROJECTION_LENSES:
                    for change in LENS_STEPS:
                        if not DAC_MIN <= values[name] + change <= DAC_MAX:
                            continue
                        set_from_below(**dict(values, **{name: values[name] + change}))
                        if scope.measure().usable:
                            kick_both(mode, label, setting=f"{name} {change:+d}", lens=name, change=change)
                        else:
                            print(f"      {name} {change:+d}: the beam is not usable; left out")
                set_from_below(**values)
    go_to_row(MAG, mag_at_start)


def deflectors():
    """Is the displacement proportional to the kick, and does it change with CL3 or the spot size?"""
    for mode, label, name in [(MAG, mag_at_start, "CLA1"), (DIFF, 40, "CLA2")]:
        go_to_row(mode, label)
        prepare(mode, label)
        full = kick_size(mode, label, name) * DISPLACEMENT_RANGE_PX[1] // DISPLACEMENT_PX
        for fraction in (0.05, 0.1, 0.25, 0.5, 1.0):
            record_kick(name, max(10, int(full * fraction)), mode_row=label, test="kick size")
        base = cl3()
        for offset in ([-150, 150] if QUICK else [-300, -150, 150, 300, 600]):
            scope.set_lenses(CL3=base + offset)
            if centre_beam(mode, label).usable:
                record_kick(name, kick_size(mode, label, name), mode_row=label, test="CL3", change=offset)
        scope.set_lenses(CL3=base)
        for spot in SPOTS:
            select_spot(spot)
            if prepare(mode, label).usable:
                record_kick(name, kick_size(mode, label, name), mode_row=label, test="spot")
        select_spot(spot_at_start)
        scope.set_lenses(CL3=base)
    go_to_row(MAG, mag_at_start)


def fit_on_camera(mode, labels, at):
    """Step the selector until the beam fits the camera (rms 6 to 90 px, not touching the edge). Returns the row's
    index and the beam."""
    for _ in range(len(labels)):
        go_to_row(mode, labels[at])
        beam = scope.measure()
        if beam.found and abs(beam.centre) > 40 and not beam.touches_edge:
            beam = centre_beam(mode, labels[at])
        too_big = (not beam.found) or beam.touches_edge or beam.rms > 90
        if too_big and at > 0:
            at -= 1
        elif beam.found and beam.rms < 6 and at < len(labels) - 1:
            at += 1
        else:
            break
    return at, beam


def condenser():
    """The beam in MAG and in DIFF, at each spot size and CL3 (and hole): what the condenser sends to the specimen."""
    rows = {MAG: all_rows(MAG), DIFF: all_rows(DIFF)}
    at = {MAG: rows[MAG].index(mag_at_start) if mag_at_start in rows[MAG] else 0, DIFF: len(rows[DIFF]) // 2}
    hole_at_start = scope.read_state().get("condenser_hole")
    settings = [(spot, hole_at_start) for spot in SPOTS]
    if "--holes" in sys.argv:
        settings += [(spot_at_start, hole) for hole in HOLES if hole != hole_at_start]
    for spot, hole in settings:
        if hole != scope.read_state().get("condenser_hole"):
            scope.change(scope.apertures.SelectKind, 1)
            scope.change(scope.apertures.SetSize, hole)
            scope.wait(scope.mode_settle_s)
        select_spot(spot)
        table_cl3 = cl3()
        for offset in CL3_OFFSETS:
            value = table_cl3 + offset
            if not DAC_MIN <= value <= DAC_MAX:
                continue
            scope.set_lenses(CL3=max(DAC_MIN, value - APPROACH))
            scope.set_lenses(CL3=value)
            for mode in (MAG, DIFF):
                at[mode], beam = fit_on_camera(mode, rows[mode], at[mode])
                scope.record(stage=stage, role="beam", mode_row=rows[mode][at[mode]], hole=hole, cl3_offset=offset)
            print(f"   spot index {spot}, hole {hole}, CL3 {value}: MAG {rows[MAG][at[MAG]]}x and DIFF {rows[DIFF][at[DIFF]]} cm")
    if hole_at_start != scope.read_state().get("condenser_hole"):
        scope.change(scope.apertures.SelectKind, 1)
        scope.change(scope.apertures.SetSize, hole_at_start)
    scope.change(eos.SelectSpotSize, spot_at_start)


def go_back():
    """Free-lens control off, the deflectors, spot size, mode and magnification as at the start."""
    print("\nputting the microscope back")
    scope.restore()
    scope.change(eos.SelectSpotSize, spot_at_start)
    select_mode(MAG)
    for _ in range(60):
        at = row()
        if at == mag_at_start:
            break
        scope.change(eos.UpSelector if at < mag_at_start else eos.DownSelector)
        scope.wait(scope.mode_settle_s)


def pilot():
    """A fixed-objective calibration with entire rows and compound commands reserved for testing."""
    simulated = scope.simulated
    source = "simulated" if simulated else "hardware"
    scope.console["pilot_design"] = "compact" if COMPACT else "dense"
    check_state = None
    camera_settings = getattr(scope.camera, "get_detectorsetting", None)
    scope.console["SightX"] = dict(shape=scope.frame_shape, dtype=str(scope.frame_dtype), dark=scope.dark_level,
                                   settings=scope.ask(camera_settings) if camera_settings else None)

    def checks():
        nonlocal check_state
        go_to_row(MAG, 40000)
        if check_state is None:
            if not prepare(MAG, 40000).usable:
                raise RuntimeError("No usable beam at the 40k reference setting")
            check_state = scope.read_state()
        set_from_below(CL3=check_state["CL3"])
        for name in ("CLA1", "CLA2", "IS1", "IS2", "PLA"):
            scope.set_deflector(name, check_state[name + "_x"], check_state[name + "_y"])
        current = scope.read_state()
        mismatch = [name for name in check_state if name != "stage_position" and current[name] != check_state[name]]
        if mismatch:
            raise RuntimeError(f"The repeated 40k setting changed: {mismatch}")
        for _ in range(2):
            scope.record(stage="pilot", role="reference", source=source, mode_row=40000, split="check", setting="row")

    # Keep every image from a held row together. These settings are identified before acquisition.
    groups = [(MAG, [30000, 40000, 80000, 150000, 300000, 500000], "train"),
              (MAG, [8000, 12000, 20000], "train"),
              (DIFF, ([15] if simulated else [8, 15]) + [40, 80, 150, 200], "train"),
              (MAG, [10000, 15000, 60000, 200000, 400000], "held_row"),
              (DIFF, [25, 100], "held_row")]
    nominal_rows = {MAG: {8000, 20000, 30000, 40000, 500000}, DIFF: {8, 15, 40, 200}}
    branch_kicks = {(MAG, 8000), (MAG, 20000), (MAG, 40000), (MAG, 300000), (DIFF, 15), (DIFF, 40)}
    checks()
    for mode, labels, split in groups:
        for label in labels:
            nominal = not COMPACT or split != "train" or label in nominal_rows[mode]
            branch = not simulated and (mode, label) in branch_kicks
            if not nominal and not branch:
                continue
            values = go_to_row(mode, label)
            if mode == DIFF:
                set_from_below(CL3=console["CL3"])
            prepare(mode, label)
            about = dict(source=source, mode_row=label, split=split)
            beam = scope.record(stage=stage, role="reference", setting="row", **about)
            if not beam.usable or beam.saturated:
                print(f"Row {label} excluded: {describe(beam)}")
                continue
            if branch:
                for name in ("IS1", "IS2", "PLA"):
                    for trial in (1000, 500, 250):
                        slopes, used = record_kick(name, trial, setting="row", **about)
                        if np.all(np.isfinite(slopes)):
                            print(f"{name}: signed responses {np.abs(slopes) * used} px; compare with repeats.")
                            break
            if not nominal:
                continue
            strong, weak = ("CLA1", "CLA2") if mode == MAG else ("CLA2", "CLA1")
            amplitude = kick_size(mode, label, strong)
            measured, amplitude = record_kick(strong, amplitude, setting="row", **about)
            if np.all(np.isfinite(measured)):
                response[(mode, label, strong)] = measured
            record_kick(weak, WEAK_KICK, setting="row", **about)
            origin, other = cla(strong), cla(weak)
            if split == "train" and (not COMPACT or (mode, label) in {(MAG, 40000), (DIFF, 40)}):
                # Measure both CLA slopes at both signed lens endpoints; each pair's midpoint also measures offset.
                for name in PROJECTION_LENSES:
                    wobble = 500 if name == "IL3" and mode == MAG and label >= 300000 else 1000
                    for sign in (-1, 1):
                        delta = sign * wobble
                        if not DAC_MIN <= values[name] + delta <= DAC_MAX:
                            continue
                        for offset in (delta, delta // 2, delta // 4):
                            set_from_below(**{name: values[name] + offset})
                            trial = scope.measure()
                            if trial.usable and not trial.saturated:
                                break
                        else:
                            print(f"Skipped {name} {delta:+d} at {label}: beam clipped even at a quarter step")
                            continue
                        for coil, probe in ((strong, max(1, amplitude // 3)), (weak, WEAK_KICK // 2)):
                            while probe >= 1:
                                slopes, used = record_kick(coil, probe, axes="x", lens=name, change=offset,
                                                           setting=f"{name} {offset:+d}", **about)
                                if np.isfinite(slopes[0]):
                                    break
                                probe = used // 2
                    set_from_below(**{name: values[name]})
            # Reserve both signs of a two-lens, two-deflector move. Use interior amplitudes.
            margin = min(*other, DAC_MAX - other[0], DAC_MAX - other[1], WEAK_KICK // 4)
            for sign in (-1, 1):
                mixed = {"IL1": values["IL1"] + sign * 300, "IL3": values["IL3"] - sign * 200}
                if not all(DAC_MIN <= v <= DAC_MAX for v in mixed.values()):
                    continue
                set_from_below(**mixed)
                scope.set_deflector(strong, origin[0] + sign * (amplitude // 4), origin[1] - sign * (amplitude // 5))
                scope.set_deflector(weak, other[0] - sign * margin, other[1] + sign * margin)
                scope.record(stage=stage, role="compound", setting="IL1 IL3 CLA1 CLA2",
                             **dict(about, split="held_move" if split == "train" else split))
            # A fresh challenge after adding weak-coil probes. Keep its labels out of fit and model choice.
            set_from_below(**values)
            sign = 1 if mode == MAG else -1
            pl_delta = -350 if values["PL1"] > DAC_MAX - 350 else 350
            if DAC_MIN <= values["IL2"] + sign * 280 <= DAC_MAX and DAC_MIN <= values["PL1"] + pl_delta <= DAC_MAX:
                set_from_below(IL2=values["IL2"] + sign * 280, PL1=values["PL1"] + pl_delta)
                scope.set_deflector(strong, origin[0] - amplitude // 6, origin[1] + amplitude // 3)
                scope.set_deflector(weak, other[0] + margin // 2, other[1] - margin)
                scope.record(stage=stage, role="compound", setting="IL2 PL1 CLA1 CLA2", **dict(about, split="prospective"))
            scope.set_deflector(strong, *origin)
            scope.set_deflector(weak, *other)
            set_from_below(**values)
            print(f"Completed {source} {mode=} {label=} {split=}", flush=True)
        checks()
        scope.save()

    # Spot changes load CL1, CL3 and SpotA together; local CL3 steps distinguish brightness at fixed spot.
    spot_state = {}
    for mode, label in ((MAG, 40000), (DIFF, 40)):
        go_to_row(mode, label)
        for spot in (0, 1, 2, 3, 4):
            select_spot(spot)
            if mode == MAG and not prepare(mode, label).usable:
                continue
            if mode == MAG:
                spot_state[spot] = scope.read_state()
            if spot not in spot_state:
                continue
            matched = spot_state[spot]
            if mode == DIFF:
                for name in ("CLA1", "CLA2"):
                    scope.set_deflector(name, matched[name + "_x"], matched[name + "_y"])
            table_cl3 = matched["CL3"]
            for change in (-100, 0, 100):
                target = table_cl3 + change
                if DAC_MIN <= target <= DAC_MAX:
                    set_from_below(CL3=target)
                    scope.record(stage=stage, role="reference", source=source, mode_row=label,
                                 split="held_spot" if spot == (1 if spot_at_start == 3 else 3) else "train",
                                 setting="spot CL3", change=change)
        select_spot(spot_at_start)
    checks()


acquired = False
try:
    scope.take_over("CL3")
    for stage in stages:
        print(f"\n===== {stage} =====")
        {"survey": survey, "history": history, "kicks": kicks, "deflectors": deflectors, "condenser": condenser,
         "pilot": pilot}[stage]()
    acquired = True
finally:
    try:
        go_back()
        if stages == ["pilot"]:
            brightness_change = console["CL3"] - scope.ask(lens.GetCL3)
            if brightness_change:
                scope.change(eos.SetBrightness, brightness_change)
                scope.wait(scope.lens_settle_s)
            for name in ("CLA1", "CLA2", "IS1", "IS2", "PLA"):
                scope.set_deflector(name, console[name + "_x"], console[name + "_y"])
            final = scope.read_state()
            scope.console["after restore"] = final
            scope.console["frame_counts"] = dict(saved=len(scope.rows), total_exposures=scope.exposure_count)
            mismatch = [k for k in console if k != "stage_position" and final[k] != console[k]]
            mismatch += [f"{name} FLC enabled" for name in ("CL3", *PROJECTION_LENSES)
                         if scope.ask(lens.GetFLCInfo, FREE_LENS[name])[0]]
            if mismatch:
                raise RuntimeError(f"Check the console before leaving; original readbacks differ: {mismatch}")
            scope.console["pilot_complete"] = acquired
    finally:
        scope.save()
