"""Centre the beam and make it fill 20 % of the camera, at every MAG row from 8k to 500k.

At each magnification:
    CL3 (free-lens control) sets the beam's size;
    CLA1 (beam shift) centres it;
    how far CLA1 displaces the beam (px per CLA1 unit) is measured and kept.
Every frame taken is a row of table.csv (see neoarm.py), with every lens value. At the end the microscope is put back:
CL3 free-lens control off, CLA1 where the console had it, and the magnification it started at.

"Fills 20 %" means: the beam, seen as a disc of radius 2 x its rms radius (exact for a uniform disc), covers 20 % of
the frame's area.

Run it from this folder:
    python beam_per_magnification.py --simulated     the virtual NEOARM (needs TemGym)
    python beam_per_magnification.py                 the real NEOARM, a dry run: reads it, takes a frame, changes nothing
    python beam_per_magnification.py --for-real      the real NEOARM
"""

import sys

import numpy as np

from neoarm import Neoarm, describe

FIRST_MAG, LAST_MAG = 8_000, 500_000
FILL = 0.20                    # fraction of the frame the beam's disc should cover
FILL_TOLERANCE = 0.01          # 19 % to 21 % is good enough
CENTRE_TOLERANCE_PX = 2.0      # the beam's centre this close to the frame's centre is good enough
LARGEST_CL3_STEP = 4000        # never change CL3 by more than this in one go
LARGEST_CLA1_STEP = 8000       # never change CLA1 by more than this in one go

scope = Neoarm("beam_per_magnification", for_real="--for-real" in sys.argv, simulated="--simulated" in sys.argv)
eos = scope.eos
TARGET_RMS_PX = np.sqrt(FILL * scope.frame_shape[0] * scope.frame_shape[1] / np.pi) / 2     # disc radius / 2


def fill(beam):
    """The fraction of the frame the beam's disc (radius 2 x rms) covers."""
    return np.pi * (2 * beam.rms) ** 2 / (scope.frame_shape[0] * scope.frame_shape[1])


def measure():
    beam = scope.measure()
    if not beam.found:
        raise RuntimeError("no beam on the camera")
    return beam


def magnification():
    return int(scope.ask(eos.GetMagValue)[0])


def set_cl3(value):
    value = int(round(value))
    scope.set_lenses(CL3=value)
    return value


def set_cla1(x, y):
    x, y = int(round(x)), int(round(y))
    scope.set_deflector("CLA1", x, y)
    return x, y


# ---------------------------------------------------------------------------------------------------------------------
# Sizing with CL3. The beam's rms radius changes nearly in proportion to CL3 (away from the crossover). Each step
# assumes a straight line, with the slope from the last two frames; a step is never more than LARGEST_CL3_STEP, and
# never aims below half the size wanted, so the beam is not focused to a point.
# ---------------------------------------------------------------------------------------------------------------------
def cl3_slope(cl3, trial_step=200):
    """How much the beam's rms radius changes per CL3 unit (px per unit): CL3 is raised by 200, 400, ... units until
    the radius changes by 2 px, then put back. The sign says which side of the crossover the beam is on."""
    before = measure()
    while True:
        set_cl3(cl3 + trial_step)
        after = measure()
        if abs(after.rms - before.rms) >= 2 or trial_step >= LARGEST_CL3_STEP:
            break
        trial_step *= 2
    set_cl3(cl3)
    return (after.rms - before.rms) / trial_step


def size_beam(cl3, slope, most_steps=8):
    """Change CL3 until the beam fills FILL of the frame. Returns CL3 and the slope last measured."""
    beam = measure()
    for _ in range(most_steps):
        if abs(fill(beam) - FILL) <= FILL_TOLERANCE:
            break
        aim = max(TARGET_RMS_PX, beam.rms / 2)                      # never more than halve it in one step
        step = np.clip((aim - beam.rms) / slope, -LARGEST_CL3_STEP, LARGEST_CL3_STEP)
        cl3 = set_cl3(cl3 + step)
        new = measure()
        if abs(new.rms - beam.rms) > 0.5:                           # a change big enough to measure the slope
            slope = (new.rms - beam.rms) / step
        beam = new
        print(f"      CL3 {cl3:5d}: {describe(beam)}, fills {fill(beam):.1%}")
    return cl3, slope


# ---------------------------------------------------------------------------------------------------------------------
# Centring with CLA1. How CLA1 displaces the beam on the camera is a 2 x 2 matrix (it turns and scales with the
# magnification), so it is measured at each magnification: CLA1 is changed along x and along y, and put back.
# Centring is then one solve. The trial frames are recorded: they are the beam-shift response at this magnification.
# ---------------------------------------------------------------------------------------------------------------------
def cla1_response(cla1, trial_step, mag):
    """The 2 x 2 matrix (px per CLA1 unit). The trial step grows until the beam is displaced 10 px."""
    while True:
        start = scope.record(magnification_row=mag, role="not deflected", deflector="CLA1").centre
        displaced = []
        for direction, dx, dy in [("x", trial_step, 0), ("y", 0, trial_step)]:
            set_cla1(cla1[0] + dx, cla1[1] + dy)
            beam = scope.record(magnification_row=mag, role="deflected +", deflector="CLA1", direction=direction,
                                deflection_x=dx, deflection_y=dy)
            displaced.append(beam.centre - start)
        set_cla1(*cla1)
        if min(abs(d) for d in displaced) >= 10 or trial_step >= LARGEST_CLA1_STEP:
            break
        trial_step = int(min(LARGEST_CLA1_STEP, trial_step * 30 / max(min(abs(d) for d in displaced), 1)))
    response = np.array([[displaced[0].real, displaced[1].real], [displaced[0].imag, displaced[1].imag]]) / trial_step
    return response, trial_step


def centre_beam(cla1, response, most_steps=3):
    """Change CLA1 until the beam is within CENTRE_TOLERANCE_PX of the frame's centre. Returns CLA1."""
    beam = measure()
    for _ in range(most_steps):
        if abs(beam.centre) <= CENTRE_TOLERANCE_PX:
            break
        step = np.linalg.solve(response, [-beam.centre.real, -beam.centre.imag])
        step = step * min(1.0, LARGEST_CLA1_STEP / np.abs(step).max())
        cla1 = set_cla1(cla1[0] + step[0], cla1[1] + step[1])
        beam = measure()
        print(f"      CLA1 ({cla1[0]:5d}, {cla1[1]:5d}): {describe(beam)}")
    return cla1


# ---------------------------------------------------------------------------------------------------------------------
# Start: check MAG mode, note what the console has, measure the dark level
# ---------------------------------------------------------------------------------------------------------------------
if scope.ask(eos.GetFunctionMode)[0] != 0:
    raise SystemExit("put the microscope in MAG mode at the console, then start again")
mag_at_start = magnification()
console = scope.remember_console("at the start")
print(f"at the start: MAG {mag_at_start}x, CL1 {console['CL1']}, CL3 {console['CL3']}, "
      f"CLA1 ({console['CLA1_x']}, {console['CLA1_y']})")
print(f"frame {scope.frame_shape[1]} x {scope.frame_shape[0]} px: 20 % fill is a beam of rms radius {TARGET_RMS_PX:.1f} px")
scope.measure_dark_level()
print("the beam now:", describe(measure()))
if scope.dry_run:
    raise SystemExit("\nDry run: the connection and the camera work. Run with --for-real to set up the beam at each magnification.")


def go_back():
    """CL3 free-lens control off, CLA1 as at the start, the starting magnification."""
    print("\nputting the microscope back: CL3 free-lens control off, CLA1 as at the start, the starting magnification")
    scope.restore()
    for _ in range(40):
        if magnification() == mag_at_start:
            break
        scope.change(eos.DownSelector if magnification() > mag_at_start else eos.UpSelector)
        scope.wait(scope.mode_settle_s)


# ---------------------------------------------------------------------------------------------------------------------
# Go through the magnifications
# ---------------------------------------------------------------------------------------------------------------------
summary = []
scope.take_over("CL3")
try:
    while magnification() > FIRST_MAG:                             # down to the first magnification
        scope.change(eos.DownSelector)
        scope.wait(scope.mode_settle_s)
    if magnification() != FIRST_MAG:
        raise RuntimeError(f"there is no {FIRST_MAG}x row: the selector stopped at {magnification()}x")

    cl3, cla1 = console["CL3"], (console["CLA1_x"], console["CLA1_y"])
    slope, cla1_trial, previous_mag = None, 500, None
    while True:
        mag = magnification()
        print(f"\n{mag}x")
        if previous_mag:                                            # sizes on the camera scale with the magnification
            slope *= mag / previous_mag
            cla1_trial = int(min(LARGEST_CLA1_STEP, cla1_trial * previous_mag / mag))
        print("   now:", describe(measure()))

        if slope is None:                                           # size first: then the beam can be centred well
            slope = cl3_slope(cl3)
        cl3, slope = size_beam(cl3, slope)
        response, cla1_trial = cla1_response(cla1, cla1_trial, mag)
        cla1 = centre_beam(cla1, response)
        for _ in range(3):                                          # CL3 also displaces the beam a little
            beam = measure()
            if abs(fill(beam) - FILL) <= FILL_TOLERANCE and abs(beam.centre) <= CENTRE_TOLERANCE_PX:
                break
            cl3, slope = size_beam(cl3, slope)
            cla1 = centre_beam(cla1, response)

        beam = scope.record(magnification_row=mag, role="sized and centred",
                            px_per_cla1_unit=np.round(response, 5).tolist())
        good = abs(fill(beam) - FILL) <= FILL_TOLERANCE and abs(beam.centre) <= CENTRE_TOLERANCE_PX
        summary.append((mag, cl3, cla1, fill(beam), beam.centre, good))
        print(f"   {'done' if good else 'NOT DONE'}: CL3 {cl3}, CLA1 {cla1}; {describe(beam)}, fills {fill(beam):.1%}")

        if mag >= LAST_MAG:
            break
        previous_mag = mag
        scope.change(eos.UpSelector)
        scope.wait(scope.mode_settle_s)
        if magnification() == mag:
            raise RuntimeError(f"the selector does not go above {mag}x")
finally:
    go_back()
    scope.save()
    print(f"\n{'mag':>8} {'CL3':>6} {'CLA1 x':>7} {'CLA1 y':>7} {'fill':>6} {'centre (px)':>16}")
    for mag, cl3, cla1, filled, centre, good in summary:
        print(f"{mag:>8} {cl3:>6} {cla1[0]:>7} {cla1[1]:>7} {filled:>6.1%} ({centre.real:+6.1f}, {centre.imag:+6.1f})"
              + ("" if good else "  NOT DONE"))
