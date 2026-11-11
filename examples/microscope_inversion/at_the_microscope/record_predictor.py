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
    python record_predictor.py                              the real NEOARM, a dry run: reads it, changes nothing
    python record_predictor.py --for-real kicks condenser   the real NEOARM, these stages only
    add --holes to also change the condenser aperture's hole (a motor: check its alignment afterwards)
    add --quick for a short version of every stage (a rehearsal of the script itself)
"""

import sys

import numpy as np

from neoarm import DAC_MAX, DAC_MIN, Neoarm, describe

STAGES = ["survey", "history", "kicks", "deflectors", "condenser"]
stages = [s for s in STAGES if s in sys.argv] or STAGES
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

scope = Neoarm("predictor", for_real="--for-real" in sys.argv, simulated="--simulated" in sys.argv)
eos, lens = scope.eos, scope.lens
neutral = (DAC_MIN + DAC_MAX + 1) // 2
response = {}                                  # (mode, row label, deflector) -> px per DAC unit along its x and its y (each x + iy)
stage = "start"


# ---------------------------------------------------------------------------------------------------------------------
# Rows of the selector
# ---------------------------------------------------------------------------------------------------------------------
def row():
    """The selected row's label: the magnification, or the camera length in cm."""
    return int(scope.ask(eos.GetMagValue)[0])


def select_mode(mode):
    if scope.ask(eos.GetFunctionMode)[0] != mode:
        scope.release(*PROJECTION_LENSES)
        scope.change(eos.SelectFunctionMode, mode)
        scope.wait(scope.mode_settle_s)


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
    scope.release(*PROJECTION_LENSES)
    for _ in range(60):
        at = row()
        if at == label or scope.dry_run:
            break
        scope.change(eos.UpSelector if at < label else eos.DownSelector)
        scope.wait(scope.mode_settle_s)
        if row() == at:
            raise RuntimeError(f"the selector does not reach {label}: it stops at {at}")
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
        per_unit = expected_response(mode, label, name) or measure_response(mode, label, name)
        step = units_for(-beam.centre, per_unit)
        step = step * min(1.0, LARGEST_KICK / max(np.abs(step).max(), 1))
        x, y = cla(name)
        scope.set_deflector(name, np.clip(x + step[0], DAC_MIN, DAC_MAX), np.clip(y + step[1], DAC_MIN, DAC_MAX))
    return scope.measure()


def measure_response(mode, label, name):
    """px per DAC unit of a deflector along its x and its y, found with trial kicks that grow until the beam is
    displaced 10 px (not recorded). Used only where no response is known yet."""
    x, y = cla(name)
    start, kick = scope.measure().centre, 50
    while True:
        scope.set_deflector(name, x + kick, y)
        along_x = scope.measure().centre - start
        scope.set_deflector(name, x, y + kick)
        along_y = scope.measure().centre - start
        scope.set_deflector(name, x, y)
        if min(abs(along_x), abs(along_y)) >= 10 or kick >= LARGEST_KICK:
            break
        kick = int(min(LARGEST_KICK, kick * 30 / max(min(abs(along_x), abs(along_y)), 1)))
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


def record_kick(name, kick, **about):
    """The four frames of a kick. Returns the displacement per DAC unit along the deflector's x and its y (px,
    x + iy), and the kick used."""
    x, y = cla(name)
    kick = int(min(kick, x - DAC_MIN, DAC_MAX - x, y - DAC_MIN, DAC_MAX - y))
    centres = {}
    for direction, dx, dy in [("x", kick, 0), ("y", 0, kick)]:
        for sign in (+1, -1):
            scope.set_deflector(name, x + sign * dx, y + sign * dy)
            beam = scope.record(stage=stage, role="deflected", deflector=name, direction=direction, sign=sign,
                                kick=kick, **about)
            centres[direction, sign] = beam.centre if beam.usable else np.nan
    scope.set_deflector(name, x, y)
    return tuple((centres[d, +1] - centres[d, -1]) / (2 * kick) for d in "xy"), kick


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
print(f"at the start: MAG {mag_at_start}x, spot index {spot_at_start}, CL3 {console['CL3']}")
scope.measure_dark_level()
print("the beam now:", describe(scope.measure()))
if scope.dry_run:
    raise SystemExit("\nDry run: the connection and the camera work. Run with --for-real to record.")
scope.take_over("CL3")


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


try:
    for stage in stages:
        print(f"\n===== {stage} =====")
        {"survey": survey, "history": history, "kicks": kicks, "deflectors": deflectors, "condenser": condenser}[stage]()
finally:
    go_back()
    scope.save()
