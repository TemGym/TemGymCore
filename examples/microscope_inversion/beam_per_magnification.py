"""Centre the beam and make it fill 20 % of the camera, at every MAG row from 8k to 500k.

At each magnification:
    CL3 (free-lens control) sets the beam's size;
    CLA1 (beam shift) centres it.
The values found are written to a table, one row per magnification. At the end the microscope is put back as it
was: CL3 free-lens control off, CLA1 where the console had it, and the magnification it started at.

"Fills 20 %" means: the beam, seen as a disc of radius 2 x its rms radius (exact for a uniform disc), covers 20 % of
the frame's area.

Run it from this folder:
    python beam_per_magnification.py --simulated     the virtual NEOARM (needs TemGym)
    python beam_per_magnification.py                 the real NEOARM, a dry run: reads it, takes a frame, changes nothing
    python beam_per_magnification.py --for-real      the real NEOARM

Needs numpy, and pyjem_session.py (for find_beam) in this folder or the one above.
"""

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from pyjem_session import find_beam  # noqa: E402


# ---------------------------------------------------------------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------------------------------------------------------------
SIMULATED = "--simulated" in sys.argv
FOR_REAL = "--for-real" in sys.argv or SIMULATED          # the simulated microscope is always changed for real

FIRST_MAG, LAST_MAG = 8_000, 500_000
FILL = 0.20                    # fraction of the frame the beam's disc should cover
FILL_TOLERANCE = 0.01          # 19 % to 21 % is good enough
CENTRE_TOLERANCE_PX = 2.0      # the beam's centre this close to the frame's centre is good enough
DAC_MIN, DAC_MAX = 0, 65535    # lenses and deflectors
CL3_FLC = 2                    # PyJEM's free-lens number of CL3
LARGEST_CL3_STEP = 4000        # never change CL3 by more than this in one go
LARGEST_CLA1_STEP = 8000       # never move CLA1 by more than this in one go
PAUSE_S = 0.1                  # after every call to the microscope


# ---------------------------------------------------------------------------------------------------------------------
# 2. Connect to the microscope and the camera
# ---------------------------------------------------------------------------------------------------------------------
if SIMULATED:
    os.environ["JAX_PLATFORMS"] = "cpu"
    import temgym_core
    from temgym_core.pyjem_simulator import SimulatedMicroscope
    temgym_core.enable_x64()
    microscope = SimulatedMicroscope.from_file(HERE.parent / "virtual_neoarm/virtual_neoarm.json", ht_v=200e3,
                                               magnification=40000, ray_sample_power=20)    # left on 40k
    TEM3, detector = microscope.pyjem()
    camera = detector.Detector(detector.get_attached_detector()[0])
    area = camera.get_detectorsetting()["ImagingArea"]
    bits = camera.get_detectorsetting()["OutputImageInformation"]["DataBits"]
    FRAME_SHAPE, FRAME_DTYPE = (area["Height"], area["Width"]), np.dtype("<i4" if bits == 32 else "<i2")
    LENS_SETTLE_S, MAG_SETTLE_S = 0, 0
    output = HERE / "output/beam_per_magnification_simulated"
else:
    from PyJEM import TEM3
    from PyJEM import sightx
    sightx.set_ip("172.17.41.1")
    camera = sightx.Detector("sightx")
    FRAME_SHAPE, FRAME_DTYPE = (1916 // 4, 1916 // 4), np.dtype(np.uint16)       # as in stage1_record.py
    LENS_SETTLE_S, MAG_SETTLE_S = 2, 3
    output = HERE / "output/beam_per_magnification_neoarm"

lens, deflectors, eos = TEM3.Lens3(), TEM3.Def3(), TEM3.EOS3()
TARGET_RMS_PX = np.sqrt(FILL * FRAME_SHAPE[0] * FRAME_SHAPE[1] / np.pi) / 2     # disc radius / 2


# ---------------------------------------------------------------------------------------------------------------------
# 3. Talking to the microscope: read, set, take a frame, measure the beam
# ---------------------------------------------------------------------------------------------------------------------
def ask(call, *arguments):
    """Read something from the microscope, then pause."""
    answer = call(*arguments)
    time.sleep(PAUSE_S)
    return answer


def change(call, *arguments):
    """Set something on the microscope, then pause. In a dry run, only print it."""
    if not FOR_REAL:
        print(f"      would call {call.__name__}{arguments}")
        return
    call(*arguments)
    time.sleep(PAUSE_S)


def set_cl3(value):
    value = int(round(value))
    if not DAC_MIN <= value <= DAC_MAX:
        raise RuntimeError(f"CL3 would be {value}, outside {DAC_MIN} to {DAC_MAX}")
    change(lens.SetFLCAbs, CL3_FLC, value)
    time.sleep(LENS_SETTLE_S)
    return value


def set_cla1(x, y):
    x, y = int(round(x)), int(round(y))
    if not (DAC_MIN <= x <= DAC_MAX and DAC_MIN <= y <= DAC_MAX):
        raise RuntimeError(f"CLA1 would be ({x}, {y}), outside {DAC_MIN} to {DAC_MAX}")
    change(deflectors.SetCLA1, x, y)
    return x, y


def take_frame():
    return np.frombuffer(ask(camera.snapshot_rawdata), dtype=FRAME_DTYPE).reshape(FRAME_SHAPE)


def measure():
    """Take a frame and find the beam: its centre (px from the frame's centre, x + iy) and rms radius (px)."""
    beam = find_beam(frame=take_frame(), background=dark_level)
    if not beam.found:
        raise RuntimeError("no beam on the camera")
    return beam


def fill(beam):
    """The fraction of the frame the beam's disc (radius 2 x rms) covers."""
    return np.pi * (2 * beam.rms) ** 2 / (FRAME_SHAPE[0] * FRAME_SHAPE[1])


def describe(beam):
    return (f"centre ({beam.centre.real:+6.1f}, {beam.centre.imag:+6.1f}) px, rms {beam.rms:5.1f} px, "
            f"fills {fill(beam):5.1%}" + ("  [" + "; ".join(beam.problems) + "]" if beam.problems else ""))


def magnification():
    return int(ask(eos.GetMagValue)[0])


# ---------------------------------------------------------------------------------------------------------------------
# 4. Sizing with CL3
#
# The beam's rms radius on the camera changes nearly in proportion to CL3 (away from the crossover). Each step
# assumes a straight line, with the slope measured by the last two frames; a step is never more than
# LARGEST_CL3_STEP, and never aims below half the size wanted, so the beam is not focused to a point.
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
        print(f"      CL3 {cl3:5d}: {describe(beam)}")
    return cl3, slope


# ---------------------------------------------------------------------------------------------------------------------
# 5. Centring with CLA1
#
# How the beam moves on the camera per CLA1 unit is a 2 x 2 matrix (it turns and scales with the magnification), so it
# is measured at each magnification: CLA1 is moved along x and along y, and put back. Centring is then one solve.
# ---------------------------------------------------------------------------------------------------------------------
def cla1_response(cla1, trial_step):
    """The 2 x 2 matrix (px per CLA1 unit) at this magnification. The trial move grows until the beam moves 10 px."""
    while True:
        start = measure().centre
        moved = []
        for dx, dy in [(trial_step, 0), (0, trial_step)]:
            set_cla1(cla1[0] + dx, cla1[1] + dy)
            moved.append(measure().centre - start)
        set_cla1(*cla1)
        if min(abs(m) for m in moved) >= 10 or trial_step >= LARGEST_CLA1_STEP:
            break
        trial_step = int(min(LARGEST_CLA1_STEP, trial_step * 30 / max(min(abs(m) for m in moved), 1)))
    response = np.array([[moved[0].real, moved[1].real], [moved[0].imag, moved[1].imag]]) / trial_step
    return response, trial_step


def centre_beam(cla1, response, most_steps=3):
    """Move CLA1 until the beam is within CENTRE_TOLERANCE_PX of the frame's centre. Returns CLA1."""
    beam = measure()
    for _ in range(most_steps):
        if abs(beam.centre) <= CENTRE_TOLERANCE_PX:
            break
        move = np.linalg.solve(response, [-beam.centre.real, -beam.centre.imag])
        move = move * min(1.0, LARGEST_CLA1_STEP / np.abs(move).max())
        cla1 = set_cla1(cla1[0] + move[0], cla1[1] + move[1])
        beam = measure()
        print(f"      CLA1 ({cla1[0]:5d}, {cla1[1]:5d}): {describe(beam)}")
    return cla1


# ---------------------------------------------------------------------------------------------------------------------
# 6. Start: check MAG mode, read what the console has, measure the dark level
# ---------------------------------------------------------------------------------------------------------------------
print("SIMULATED NEOARM" if SIMULATED else ("REAL NEOARM, FOR REAL" if FOR_REAL else "REAL NEOARM, DRY RUN"))
if ask(eos.GetFunctionMode)[0] != 0:
    raise SystemExit("put the microscope in MAG mode at the console, then start again")
mag_at_start = magnification()
cla1_at_start = tuple(ask(deflectors.GetCLA1))
cl1, cl3_at_start = ask(lens.GetCL1), ask(lens.GetCL3)
print(f"at the start: MAG {mag_at_start}x, CL1 {cl1}, CL3 {cl3_at_start}, CLA1 {cla1_at_start}")
print(f"frame {FRAME_SHAPE[1]} x {FRAME_SHAPE[0]} px: 20 % fill is a beam of rms radius {TARGET_RMS_PX:.1f} px")

dark_level = None
change(deflectors.SetBeamBlank, 1)
dark_level = float(np.median(take_frame())) if FOR_REAL else None
change(deflectors.SetBeamBlank, 0)
print("dark level:", dark_level)
print("the beam now:", describe(measure()))

if not FOR_REAL:
    print("\nDry run: the connection and the camera work. Run with --for-real to set up the beam at each magnification.")
    raise SystemExit


# ---------------------------------------------------------------------------------------------------------------------
# 7. Go through the magnifications
# ---------------------------------------------------------------------------------------------------------------------
folder = output / time.strftime("%Y-%m-%d_%H%M%S")
(folder / "frames").mkdir(parents=True)
rows = []

change(lens.SetFLCAbs, CL3_FLC, cl3_at_start)                       # take over CL3 where the console has it
change(lens.SetFLCSw, CL3_FLC, 1)
try:
    # a. down to the first magnification
    while magnification() > FIRST_MAG:
        change(eos.DownSelector)
        time.sleep(MAG_SETTLE_S)
    if magnification() != FIRST_MAG:
        raise RuntimeError(f"there is no {FIRST_MAG}x row: the selector stopped at {magnification()}x")

    cl3, cla1 = cl3_at_start, cla1_at_start
    slope, cla1_trial = None, 500
    previous_mag = None
    while True:
        mag = magnification()
        print(f"\n{mag}x")
        if previous_mag:                                            # sizes on the camera scale with the magnification
            slope *= mag / previous_mag
            cla1_trial = int(min(LARGEST_CLA1_STEP, cla1_trial * previous_mag / mag))
        print("   now:", describe(measure()))

        # b. size first: after a change of magnification the beam can be far too big or small to centre well
        if slope is None:
            slope = cl3_slope(cl3)
        cl3, slope = size_beam(cl3, slope)

        # c. how CLA1 moves the beam here, then centre it
        response, cla1_trial = cla1_response(cla1, cla1_trial)
        cla1 = centre_beam(cla1, response)

        # d. CL3 also moves the beam a little: size and centre again until both are good
        for _ in range(3):
            beam = measure()
            if abs(fill(beam) - FILL) <= FILL_TOLERANCE and abs(beam.centre) <= CENTRE_TOLERANCE_PX:
                break
            cl3, slope = size_beam(cl3, slope)
            cla1 = centre_beam(cla1, response)

        # e. one frame to keep, and a row of the table
        frame = take_frame()
        beam = find_beam(frame=frame, background=dark_level)
        good = abs(fill(beam) - FILL) <= FILL_TOLERANCE and abs(beam.centre) <= CENTRE_TOLERANCE_PX
        print(f"   {'done' if good else 'NOT DONE'}: CL1 {ask(lens.GetCL1)}, CL3 {cl3}, CLA1 {cla1}; {describe(beam)}")
        np.save(folder / f"frames/{mag}.npy", frame)
        rows.append(dict(magnification=mag, CL1=ask(lens.GetCL1), CL3=cl3, CLA1_x=cla1[0], CLA1_y=cla1[1],
                         beam_x_px=round(beam.centre.real, 2), beam_y_px=round(beam.centre.imag, 2),
                         beam_rms_px=round(beam.rms, 2), fill=round(fill(beam), 4), good=good,
                         px_per_cla1_unit=np.round(response, 5).tolist(), frame=f"frames/{mag}.npy"))

        # f. the next magnification
        if mag >= LAST_MAG:
            break
        previous_mag = mag
        change(eos.UpSelector)
        time.sleep(MAG_SETTLE_S)
        if magnification() == mag:
            raise RuntimeError(f"the selector does not go above {mag}x")

finally:
    # 8. Put the microscope back, and save the table
    print("\nputting the microscope back: CL3 free-lens control off, CLA1 as at the start, the starting magnification")
    change(lens.SetFLCSw, CL3_FLC, 0)
    change(deflectors.SetCLA1, *cla1_at_start)
    for _ in range(40):
        if magnification() == mag_at_start:
            break
        change(eos.DownSelector if magnification() > mag_at_start else eos.UpSelector)
        time.sleep(MAG_SETTLE_S)
    if rows:
        with open(folder / "table.csv", "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{len(rows)} magnifications in {folder / 'table.csv'}")
        print(f"\n{'mag':>8} {'CL3':>6} {'CLA1 x':>7} {'CLA1 y':>7} {'fill':>6} {'centre (px)':>16}")
        for r in rows:
            print(f"{r['magnification']:>8} {r['CL3']:>6} {r['CLA1_x']:>7} {r['CLA1_y']:>7} {r['fill']:>6.1%} "
                  f"({r['beam_x_px']:+6.1f}, {r['beam_y_px']:+6.1f})" + ("" if r["good"] else "  NOT DONE"))
