"""Stage 1 at the NEOARM: recording how beam shift (CLA1) and tilt (CLA2) displace the beam on the camera
across projection-lens settings. Sections 1-7 of stage1_projection_response.ipynb, as a plain script for a
microscope PC without Jupyter (made from the notebook's own cells; change the notebook, not this file).

    python stage1_record.py                  dry run: reads the microscope, takes frames, changes nothing,
                                             and prints every change it would make
    python stage1_record.py --for-real       records; asks before each section
    python stage1_record.py --simulated      the same on the simulated NEOARM (needs TemGym)

Needs numpy and PyJEM, and ../pyjem_session.py. Recordings go to output/stage1_neoarm/ next to this file.
If anything stops it (an error or Ctrl-C) once the microscope is being changed, it is put back as it was.
Afterwards, fit the recording with section 8 of the notebook, on a computer with TemGym.
"""
import argparse
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):                   # a Windows console that cannot show a character: replace it
    sys.stdout.reconfigure(errors="replace")

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--for-real", action="store_true", help="change the microscope (default: a dry run)")
parser.add_argument("--simulated", action="store_true", help="the simulated NEOARM instead of the real one")
arguments = parser.parse_args()
DRY_RUN = not arguments.for_real
os.chdir(Path(__file__).resolve().parent)                # paths as in the notebook: output/, ../pyjem_session.py


def section(heading):
    print(f"\n=== {heading} ===", flush=True)
    if not DRY_RUN:
        input("Enter to go on, Ctrl-C to stop: ")


print("DRY RUN: nothing will be changed" if DRY_RUN else "FOR REAL: the microscope will be changed")
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append("..")
from pyjem_session import Session, camera_among, check_pyjem, find_beam, newest_file_in, read_frame, show_frame   # numpy and PyJEM only


section("1. Connecting to the microscope")

on_microscope = not arguments.simulated   # the real NEOARM through PyJEM, or the simulated one (needs TemGym)

if on_microscope:
    from PyJEM import TEM3
    from PyJEM import sightx as detector                 # the SightX camera: the same calls as PyJEM's detector package
    detector.set_ip("172.17.41.1")                       # the camera PC
    camera_name = "sightx"
    frame_shape = None                                   # None: from the camera's settings; or (height, width)
    frame_dtype = None                                   # None: 2 or 4 bytes per pixel, signed; or e.g. "<u2"
    settle_s = 2                                         # s for the lenses to settle after a change
    output = "output/stage1_neoarm"
else:
    import os
    os.environ["JAX_PLATFORMS"] = "cpu"                  # TemGym's JAX on the CPU (before it is imported): no GPU needed
    import temgym_core
    from temgym_core.pyjem_simulator import SimulatedMicroscope
    temgym_core.enable_x64()
    microscope = SimulatedMicroscope.from_file("../virtual_neoarm/virtual_neoarm.json", ht_v=200e3)
    TEM3, detector = microscope.pyjem()
    camera_name = None                                   # None: the attached detector that is not a STEM detector
    frame_shape = None
    frame_dtype = None
    settle_s = 0                                         # the simulated lenses settle instantly
    output = "output/stage1_simulated"

take_frame = None                                        # None: frames from PyJEM's camera
show_frames = False                                      # no figures in the script


section("2. Checking the camera and the beam before changing anything")

missing = check_pyjem(TEM3=TEM3, detector=detector, camera_name=camera_name)

if take_frame is None:
    camera = detector.Detector(camera_name or camera_among(detector.get_attached_detector()))
    try:
        print("camera settings:", camera.get_detectorsetting())
    except Exception as error:
        print("no camera settings from PyJEM:", error, "(set frame_shape in section 1)")
    frame, dtype = read_frame(camera, frame_dtype, frame_shape)
    print(f"one frame: {frame.shape[1]} x {frame.shape[0]} px, read as {dtype}, values {frame.min()} to {frame.max()}")
else:
    print("frames come from take_frame, not through PyJEM")

LENSES = ("IL1", "IL2", "IL3", "PL1")

session = Session(TEM3=TEM3, detector=detector, take_frame=take_frame, lenses=LENSES, settle_s=settle_s,
                  camera_name=camera_name, frame_shape=frame_shape, dtype=frame_dtype,
                  folder=f"{output}/projection_response_{time.strftime('%Y-%m-%d_%H%M%S')}")    # a new folder each time

frame = session.frame()
beam = find_beam(frame=frame, background=session.dark_level)
print(f"beam at ({beam.centre.real:+.1f}, {beam.centre.imag:+.1f}) px from the centre, rms {beam.rms:.1f} px")
print("problems:", "; ".join(beam.problems) or "none")

if show_frames:
    show_frame(frame, beams=[("beam", beam)], title="the beam as found")

try:

    section("3. Reading the lens settings of the 40k MAG row and the 40 cm DIFF row")

    dry_run = DRY_RUN

    lenses_at = {}                                   # IL1-PL1 DAC values at each starting point

    with session.dry_run(dry_run):
        lenses_at["40k"] = session.lenses_now()
        session.diffraction(camera_length_cm=40)
        lenses_at["DIFF"] = session.lenses_now()
        session.imaging()

        session.take_over_lenses()
        dark = session.dark_frame()

    if show_frames:
        show_frame(dark, title="dark frame (beam blanked)")

    print(lenses_at)


    section("4. Choosing the deflection sizes for CLA1 (shift) and CLA2 (tilt)")

    dry_run = DRY_RUN

    deflection_units = {}          # deflection size, per starting point and deflector
    displacement_px = {}           # how far that deflection displaces the beam on the camera (x + iy)

    with session.dry_run(dry_run):
        for start in ["40k", "DIFF"]:
            session.set_lenses(lenses_at[start])
            frame = session.frame(role="trying a deflection")
            beam_not_deflected = find_beam(frame=frame, background=session.dark_level)

            for deflector in ["CLA1", "CLA2"]:
                print(f"{start}, {deflector}: deflections of 100, 200, 400, ... units (at most 20000) until the beam is displaced by > 25 px")
                deflection = 100
                while True:
                    session.deflect(deflector, deflection)
                    frame = session.frame(role="trying a deflection")
                    beam_deflected = find_beam(frame=frame, background=session.dark_level)
                    session.deflect(deflector, 0)
                    displacement = beam_deflected.centre - beam_not_deflected.centre
                    print(f"   {deflector} {deflection:5d} units: beam displaced {abs(displacement):5.1f} px")
                    if start == "40k" and deflector == "CLA1":      # the sample frame shown at the end: the last try here
                        sample = dict(frame=frame, beams=[("not deflected", beam_not_deflected), ("deflected", beam_deflected)],
                                      title=f"40k row, CLA1 deflected by {deflection} units")

                    if not beam_deflected.usable:           # off the camera: back to half
                        deflection = deflection // 2
                        break
                    if abs(displacement) > 25 or deflection == 20000:
                        break
                    deflection = min(2 * deflection, 20000)

                deflection_units[start, deflector] = deflection
                displacement_px[start, deflector] = displacement

    if show_frames:
        show_frame(**sample)

    print(deflection_units)


    section("5. Choosing the step sizes for IL1–IL3 and PL1")

    dry_run = DRY_RUN

    lens_step_units = {}                             # lens step, per starting point, lens and direction (+1 up, -1 down)

    with session.dry_run(dry_run):
        for start in ["40k", "DIFF"]:
            # the deflector whose deflection displaces the beam more here is the yardstick
            if abs(displacement_px[start, "CLA1"]) > abs(displacement_px[start, "CLA2"]):
                deflector = "CLA1"
            else:
                deflector = "CLA2"
            deflection = deflection_units[start, deflector]
            displacement_at_start = displacement_px[start, deflector]

            for i, lens in enumerate(LENSES):
                room_up = 65535 - lenses_at[start][i]           # DAC units the lens can go up
                room_down = lenses_at[start][i] - 2000          # and down
                if room_up >= room_down:
                    sign, direction, room = +1, "up", room_up
                else:
                    sign, direction, room = -1, "down", room_down
                largest_step = int(min(8000, room))
                print(f"{start}, {lens}: steps {direction} of 250, 500, 1000, ... DAC units (at most {largest_step}) "
                      f"until the {deflector} deflection's displacement changes by > 10 px")
                lens_step = 250
                while True:
                    lenses = lenses_at[start].copy()
                    lenses[i] = lenses[i] + sign * lens_step
                    session.set_lenses(lenses)
                    frame = session.frame(role="trying a lens step")
                    beam_not_deflected = find_beam(frame=frame, background=session.dark_level)
                    session.deflect(deflector, deflection)
                    frame = session.frame(role="trying a lens step")
                    beam_deflected = find_beam(frame=frame, background=session.dark_level)
                    session.deflect(deflector, 0)
                    displacement = beam_deflected.centre - beam_not_deflected.centre
                    change = abs(displacement - displacement_at_start)
                    print(f"   {lens} {sign * lens_step:+5d}: the {deflector} deflection's displacement changed by {change:5.1f} px")
                    if start == "40k" and lens == "IL1":            # the sample frame shown at the end: the last try here
                        sample = dict(frame=frame, beams=[("not deflected", beam_not_deflected), ("deflected", beam_deflected)],
                                      title=f"40k row, IL1 {sign * lens_step:+d}, {deflector} deflected by {deflection} units")

                    if not (beam_not_deflected.usable and beam_deflected.usable):     # beam lost: back to half
                        lens_step = lens_step // 2
                        break
                    if change > 10 or lens_step >= largest_step:
                        break
                    lens_step = min(2 * lens_step, largest_step)

                # the step found, used up and down: each way at most the room on that side
                lens_step_units[start, lens, +1] = int(min(lens_step, room_up))
                lens_step_units[start, lens, -1] = int(min(lens_step, room_down))

            session.set_lenses(lenses_at[start])

    if show_frames:
        show_frame(**sample)

    print(lens_step_units)


    section("6. Measurement plan: lens settings and deflections")

    measurement_plan = []

    for start in ["40k", "DIFF"]:
        deflections = {"CLA1": deflection_units[start, "CLA1"], "CLA2": deflection_units[start, "CLA2"]}

        measurement_plan.append(dict(name=f"{start} row", lenses=lenses_at[start], deflections=deflections))
        for i, lens in enumerate(LENSES):
            for sign in [+1, -1]:
                lenses = lenses_at[start].copy()
                lenses[i] = lenses[i] + sign * lens_step_units[start, lens, sign]
                name = f"{start} row: {lens} {sign * lens_step_units[start, lens, sign]:+d}"
                measurement_plan.append(dict(name=name, lenses=lenses, deflections=deflections))
        measurement_plan.append(dict(name=f"{start} row", lenses=lenses_at[start], deflections=deflections))

    measurement_plan.append(dict(name="40k row", lenses=lenses_at["40k"],
                     deflections={"CLA1": deflection_units["40k", "CLA1"], "CLA2": deflection_units["40k", "CLA2"]}))

    for number, setting in enumerate(measurement_plan, 1):
        lenses = ", ".join(f"{lens} {value:.0f}" for lens, value in zip(LENSES, setting["lenses"]))
        deflections = f"CLA1 ±{setting['deflections']['CLA1']}, CLA2 ±{setting['deflections']['CLA2']} (x and y)"
        print(f"{number:2d}. {setting['name']:22s} {lenses}    {deflections}")


    section("7. Recording the beam displacements")

    dry_run = DRY_RUN

    with session.dry_run(dry_run):
        for number, setting in enumerate(measurement_plan, 1):
            print(f"{number}. {setting['name']}")
            about_this_setting = dict(number=number, setting=setting["name"], lenses=[float(v) for v in setting["lenses"]])

            session.set_lenses(setting["lenses"])
            frame_not_deflected = session.frame(role="not deflected", **about_this_setting)

            for deflector in ["CLA1", "CLA2"]:                  # CLA1 shifts the beam, CLA2 tilts it
                size = setting["deflections"][deflector]
                for direction, deflection in [("x", size), ("y", 1j * size)]:      # deflector units, x + iy
                    session.deflect(deflector, +deflection)
                    frame_plus = session.frame(role="deflected +", deflector=deflector, direction=direction,
                                               deflection_units=[deflection.real, deflection.imag], **about_this_setting)
                    session.deflect(deflector, -deflection)
                    frame_minus = session.frame(role="deflected -", deflector=deflector, direction=direction,
                                                deflection_units=[-deflection.real, -deflection.imag], **about_this_setting)
                    if number == 2 and deflector == "CLA1" and direction == "x":     # the sample frames shown at the end
                        sample = dict(setting=setting["name"], not_deflected=frame_not_deflected, plus=frame_plus, minus=frame_minus)
                session.deflect(deflector, 0)

        session.restore()

    if show_frames:
        beams = [("not deflected", find_beam(frame=sample["not_deflected"], background=session.dark_level)),
                 ("deflected +", find_beam(frame=sample["plus"], background=session.dark_level)),
                 ("deflected −", find_beam(frame=sample["minus"], background=session.dark_level))]
        show_frame(sample["plus"], beams=beams, title=f"{sample['setting']}: CLA1 + along x")

finally:                                     # whatever happened, the microscope as it was (restoring twice is harmless)
    with session.dry_run(DRY_RUN):
        session.restore()
