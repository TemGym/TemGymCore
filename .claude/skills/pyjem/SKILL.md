---
name: pyjem
description: How to write microscope scripts for the JEOL NEOARM in this repository. Scripts handle the microscope through jemwrap, which wraps PyJEM. Use it before writing or editing any script or notebook cell that reads or changes the microscope, on the real microscope or on the simulated microscope in TemGym.
---

# Writing microscope scripts

DRAFT for David to edit.

A microscope script is a list of the things an operator does at the microscope, in order. A
person who knows the microscope and has not seen the script before can read it once and say
what it does to the column. The same script runs on the simulated microscope and on the real
microscope.

Scripts handle the microscope through jemwrap (`jemwrap/`, a separate repository with its own
copy of PyJEM). jemwrap spaces every call to the microscope by at least 0.1 s, refuses values
that a script must not change, and restores the start values. The principal decided this on
2026-11-19 (requirements J7 to J12).

## Read first

1. The "Control" section of `jemwrap/requirements.md`: the five functions and the values a
   script may set (`SETTABLE` in `jemwrap/src/jemwrap/control.py`).
2. The pattern below and `examples/neoarm/cla1_kicks.py`, the same script as a file that runs.
3. `examples/neoarm/simulated_neoarm.py`: the simulated NEOARM behind PyJEM's names.
4. The `readable-python` skill. Its rules for code and comments apply here.

Do not use `examples/microscope_inversion/at_the_microscope/neoarm.py` or the scripts that use
its `Neoarm` object as a pattern. They have the wrappers that this skill removes.

## Rules

1. **Handle the microscope with jemwrap's five functions.** `read_value`, `set_value`,
   `settable_snapshot`, `restore` and `take_frame` from `jemwrap.control`. Each takes the
   monitor and a key that names the value as PyJEM does: `"deflectors.CLA1"` is
   `Def3.GetCLA1` and `Def3.SetCLA1`, and `"lens.CL3"` is CL3 under free lens control. Do not
   call a PyJEM setter in a script.
2. **No microscope object, no classes, no layers.** The adapter and the monitor are the only
   objects, and the script makes them in two plain lines. Do not wrap them, and do not write a
   function that only renames a jemwrap call.
3. **A script has this order:**
   1. A docstring that says what the script does to the microscope and what it saves.
   2. The imports, the switch to the simulated microscope, the adapter and the monitor.
   3. The settings, as named constants with units and sources.
   4. At most a few functions, each for an action that the script repeats.
   5. The steps, as plain lines from top to bottom.
4. **A function is for a repeated action on the microscope**, such as "set a lens from
   below" or "take a frame and find the beam". A function has a name that an operator would
   say. A function does one action. A step that happens once is plain lines.
5. **A shared file needs a reason.** When two scripts need the same function, put the
   function in one file of plain functions, `microscope_functions.py`, next to the scripts.
   Each function takes the monitor as its first argument. The file has no classes and no
   state. Say that you made this decision and why.
6. **Loops are visible.** A loop over magnifications or over deflector values is a `for`
   statement in the steps. The reader sees what changes on each pass.
7. **Print what the script does.** Before each change to the microscope, print one line:
   what changes and to which value. A person at the console can follow the script.
8. **Save the raw data.** Save each frame and the lens and deflector values that the
   microscope had for that frame. Analysis is a different script or notebook.
9. **Wait where the microscope needs time.** jemwrap spaces the calls, so a script has no
   pause between calls. A script waits only where the column needs time to settle, with a
   named constant and a visible `time.sleep`:

   ```python
   SETTLE_S = 2.0     # after a change between MAG and DIFF, so that the lenses settle
   ```

   Give the source of each wait, or say that it is a choice.
10. **Rehearse on the simulated microscope first.** One switch at the top decides which
    microscope `import PyJEM` finds. Everything below the switch is the same for both.

    ```python
    SIMULATED = True      # False at the microscope
    if SIMULATED:
        from simulated_neoarm import install_simulated_neoarm
        install_simulated_neoarm()
    from PyJEM import TEM3, detector, sightx
    ```

    The simulated microscope needs no spacing and no time to settle. The switch also sets the
    spacing to 1 ms and every wait to zero, so that a rehearsal takes minutes:

    ```python
    monitor = MicroscopeMonitor(adapter=adapter, rate_hz=1000.0 if SIMULATED else 10.0)
    if SIMULATED:
        SETTLE_S = 0.0
    ```

    Run the script on the simulated microscope and report the output before you hand the
    script over.

## Safety

jemwrap does the checks that every script needs. It refuses a key that is not settable and a
value outside the type and range of the endpoint table, before any call to the microscope. Do
not repeat these checks in the script. The checks that the script does:

1. **Keep the start.** At the start, take `start = settable_snapshot(monitor)` for the
   restore. Read each start value that the steps need with `read_value` into a plain variable
   with a plain name, such as `cla1_x_start`.
2. **Put the microscope back.** The steps are inside `try:` and `restore(monitor, start)` is
   inside `finally:`. `restore` sets back each value that changed, and switches off free lens
   control for each lens that was not under it at the start. A script that stops on an error
   or on Ctrl-C leaves the microscope as it found it.
3. **Set a lens from below.** Set the lens to 2000 units below the value, then to the value,
   with two `set_value` lines. The lens then has the same magnetic history each time.
4. **Do not touch what the task does not need.** jemwrap does not let a script move the
   stage, change the high voltage, the gun or an aperture, or insert or retract a detector.
   Do not ask for such a change unless David asked for it in this task.
5. **Ask about anything new.** Before a value goes onto jemwrap's settable list, tell David
   which value and why. A change to jemwrap follows the style of its author (requirement J12).

Do not add checks for cases that cannot occur. Do not add retries, timeouts or logging
frameworks.

## The pattern

```python
"""Measure how far CLA1 moves the beam at MAG 40k.

The script changes CLA1 only. It saves five SightX frames and the CLA1 values to cla1_kicks.npz.
At the end, also after an error or Ctrl-C, it sets every value that a script may change back to
its value at the start.
"""
import time

import numpy as np

SIMULATED = True      # False at the microscope
if SIMULATED:
    from simulated_neoarm import install_simulated_neoarm
    install_simulated_neoarm()
from PyJEM import TEM3, detector, sightx

from jemwrap import MicroscopeMonitor, PyJEMAdapter
from jemwrap.control import read_value, restore, set_value, settable_snapshot, take_frame

adapter = PyJEMAdapter(connect=not SIMULATED, tem3_module=TEM3, detector_module=detector, sightx_module=sightx)
# jemwrap spaces the calls to the real NEOARM by 0.1 s. The simulated NEOARM needs no spacing.
monitor = MicroscopeMonitor(adapter=adapter, rate_hz=1000.0 if SIMULATED else 10.0)

# A kick of 200 units moves the beam about 40 px at MAG 40k (recording of 2026-11-12).
KICKS_DAC = (-400, -200, 0, 200, 400)
# The wait after each change of CLA1, before a frame. This time is a choice. No measurement gives it.
SETTLE_S = 0.2
# The simulated NEOARM has no time to settle, so a rehearsal does not wait.
if SIMULATED:
    SETTLE_S = 0.0

# Every value that a script may change, for the restore at the end.
start = settable_snapshot(monitor)
# jemwrap gives DAC values as unsigned 16-bit numbers. Plain integers can go below zero in a sum.
cla1_x_start, cla1_y_start = (int(dac) for dac in read_value(monitor, "deflectors.CLA1"))
frames = []
try:
    for kick_dac in KICKS_DAC:
        cla1 = (cla1_x_start + kick_dac, cla1_y_start)
        print(f"CLA1 -> {cla1}")
        set_value(monitor, "deflectors.CLA1", cla1)
        time.sleep(SETTLE_S)
        frames.append(take_frame(monitor))
finally:
    print("every value that a script may change -> its value at the start")
    restore(monitor, start)

np.savez("cla1_kicks.npz", frames=np.array(frames), kicks_dac=KICKS_DAC,
         cla1_start=(cla1_x_start, cla1_y_start))
```

jemwrap is installed in the project's environment with `pip install -e jemwrap`. At the
microscope, install it the same way.

## Before handing over

- Run the script on the simulated microscope. Report the command and the output.
- List every `set_value` call, and say that `restore` in `finally:` covers it.
- List every function in the script with the number of places that call it.
- Say which of the safety checks the script has and which do not apply.
