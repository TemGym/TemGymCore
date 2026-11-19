---
name: pyjem
description: How to write PyJEM code for the JEOL microscope in this repository. Use it before writing or editing any script or notebook cell that calls PyJEM, on the real microscope or on the simulated microscope in TemGym.
---

# Writing PyJEM scripts

DRAFT for David to edit.

**This skill changes at stage 4 of `design/changes/neoarm-first-version.md`.** The principal
decided on 2026-11-19 that scripts handle the microscope through jemwrap (`jemwrap/`), once
jemwrap has control calls. Until that stage is done, do not write a new microscope script.
The rules below on the order of a script, visible loops, printed steps, raw data and safety
stay. The rules on calling PyJEM by its own names are replaced.

A PyJEM script is a list of the things an operator does at the microscope, in order. A
person who knows the microscope and has not seen the script before can read it once and
say what it does to the column. The same script runs on the simulated microscope and on
the real microscope.

## Read first

1. The PyJEM calls that the simulated microscope has: `src/temgym_core/pyjem_simulator/pyjem.py`.
   The simulated microscope raises `NotEmulatedError` for a call it does not have.
2. The `readable-python` skill. Its rules for code and comments apply here.

Do not use `examples/microscope_inversion/at_the_microscope/neoarm.py` or the scripts
that use its `Neoarm` object as a pattern. They have the wrappers that this skill removes.

## Rules

1. **Call PyJEM by its own names.** Write `lens.SetCL3(31214)` and `deflector.GetCLA1()`.
   The reader can find each call in the PyJEM manual.
2. **No microscope object, no classes, no layers.** Do not write a `scope` object, a class
   that holds the microscope, or a function that only renames a PyJEM call. Do not select
   a PyJEM call with `getattr` and a string. Write the call.
3. **A script has this order:**
   1. A docstring that says what the script does to the microscope and what it saves.
   2. The imports and the PyJEM objects (`TEM3.Lens3()`, `TEM3.Def3()`, `TEM3.EOS3()`,
      the camera).
   3. The settings, as named constants with units and sources.
   4. At most a few functions, each for an action that the script repeats.
   5. The steps, as plain lines from top to bottom.
4. **A function is for a repeated action on the microscope**, such as "set a lens from
   below" or "take a frame and find the beam". A function has a name that an operator
   would say. A function does one action. A step that happens once is plain lines.
5. **A shared file needs a reason.** When two scripts need the same function, put the
   function in one file of plain functions, `microscope_functions.py`, next to the scripts.
   Each function takes the PyJEM objects it uses as arguments. The file has no classes and
   no state. Say that you made this decision and why.
6. **Loops are visible.** A loop over magnifications or over deflector values is a `for`
   statement in the steps. The reader sees what changes on each pass.
7. **Print what the script does.** Before each change to the microscope, print one line:
   what changes and to which value. A person at the console can follow the script.
8. **Save the raw data.** Save each frame and the lens and deflector values that the
   microscope had for that frame. Analysis is a different script or notebook.
9. **Do not call PyJEM too often.** Many calls in a short time can crash PyJEM. Each script
   has two constants at the top:

   ```python
   PAUSE_S = 0.1      # the shortest time between two PyJEM calls
   SETTLE_S = 2.0     # the wait after a big change, so that the lenses settle
   ```

   Write `time.sleep(PAUSE_S)` after each PyJEM call that is followed by another PyJEM
   call. Write `time.sleep(SETTLE_S)` after each big change. A big change is a change
   between MAG mode and DIFF mode. The sleeps are visible lines in the steps. No loop reads the
   microscope without a sleep in the loop.
10. **Rehearse on the simulated microscope first.** The script gets its two PyJEM modules
   from one switch at the top. Everything below the switch is the same for both.

   ```python
   SIMULATED = True      # False at the microscope
   if SIMULATED:
       from simulated_neoarm import TEM3, detector
   else:
       from PyJEM import TEM3, detector
   ```

   Run the script on the simulated microscope and report the output before you hand the
   script over.

## Safety

Add a check only where a wrong value or a stopped script can leave the microscope in a bad
state. The checks that apply:

1. **Read before you write.** At the start, read each value that the script will change
   with its PyJEM call. Keep each value in a plain variable with a plain name, such as
   `cla1_x_start`. Do not collect the values in a dictionary or an object.
2. **Put the microscope back.** The steps are inside `try:` and the restore is inside
   `finally:`. The restore sets each changed value to its start value and sets
   free lens control to off for each lens the script changed. A script that stops on an
   error or on Ctrl-C leaves the microscope as it found it.
3. **Check each value before you send it.** A lens or deflector value is an integer from
   0 to 65535. Use one `assert` with a message next to the call.
4. **Set a lens from below.** Set the lens to 2000 units below the value, then to the
   value. The lens then has the same magnetic history each time.
5. **Do not touch what the task does not need.** Do not move the stage, change the high
   voltage, change an aperture, or insert or retract a detector unless David asked for
   that in this task.
6. **Ask about anything new.** Before the first use of a PyJEM call that changes the
   microscope and that no earlier script used, tell David which call and what it does.

Do not add checks for cases that cannot occur. Do not add retries, timeouts or logging
frameworks.

## The pattern

```python
"""Measure how far CLA1 moves the beam at MAG 40k.

The script changes CLA1 only. It saves five frames and the CLA1 values to cla1_kicks.npz.
At the end the script sets CLA1 to its value at the start.
"""
import time

import numpy as np

SIMULATED = True      # False at the microscope
if SIMULATED:
    from simulated_neoarm import TEM3, detector
else:
    from PyJEM import TEM3, detector

deflector = TEM3.Def3()
camera = detector.Detector("screenCamera")
frame_width_px = camera.get_detectorsetting()["ImagingArea"]["Width"]

# A kick of 200 units moves the beam about 40 px at MAG 40k (recording of 2026-11-12).
KICKS_DAC = (-400, -200, 0, 200, 400)
PAUSE_S = 0.1      # the shortest time between two PyJEM calls


def take_frame():
    """Return one camera frame as a 2D array of counts."""
    raw = camera.snapshot_rawdata()
    return np.frombuffer(raw, dtype=np.int32).reshape(-1, frame_width_px)


cla1_x_start, cla1_y_start = deflector.GetCLA1()
frames = []
try:
    for kick_dac in KICKS_DAC:
        cla1_x = cla1_x_start + kick_dac
        assert 0 <= cla1_x <= 65535, f"CLA1 x = {cla1_x} is outside 0 to 65535"
        print(f"CLA1 x -> {cla1_x}")
        deflector.SetCLA1(cla1_x, cla1_y_start)
        time.sleep(PAUSE_S)
        frames.append(take_frame())
        time.sleep(PAUSE_S)
finally:
    print(f"CLA1 -> {cla1_x_start}, {cla1_y_start} (the values at the start)")
    deflector.SetCLA1(cla1_x_start, cla1_y_start)

np.savez("cla1_kicks.npz", frames=np.array(frames), kicks_dac=KICKS_DAC,
         cla1_start=(cla1_x_start, cla1_y_start))
```

## Before handing over

- Run the script on the simulated microscope. Report the command and the output.
- List every PyJEM call that changes the microscope, and say where the script restores it.
- List every function in the script with the number of places that call it.
- Say which of the safety checks the script has and which do not apply.
