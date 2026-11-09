# Stage 1 at the microscope

**What stage 1 gives you:** a model that predicts where the beam lands on the camera for any
lens setting and any beam shift or tilt. No specimen is needed. Everything is done at **one HT**.
60 kV is the plan, but stage 1 works the same at 200 kV. Use whichever HT the microscope is at
today, and don't change it during stage 1.

Everything below is by hand: at the console, with free-lens control (FLC). PyJEM is optional
(see the end).

---

## 0. Set up (about 15 min)

- TEM mode, MAG, the **40k** magnification row.
- **Specimen out**, or the beam through a large hole.
- **Objective aperture out, SA aperture out.** Condenser aperture in.
- Normal alignment, including the shift and tilt **pivot points**. A beam shift must not tilt the
  beam, and a beam tilt must not move it at the specimen.
- **Condense the beam to a small round spot**, centred on the camera, about 10–20 px across at 40k.
  It has to stay on the camera at every setting below.
- **Camera:** pick one binning and one exposure and keep them fixed. The spot must not saturate at
  any setting. Write both down.
- Take one **dark frame**, with the beam blanked.

## 0b. The deflectors

With the pivot points aligned, CLA1 is a pure beam shift and CLA2 a pure tilt. No test is needed:
the analysis checks this from the stage-1 data and uses a mix of shift and tilt only if the data
need it.

## 1. Read the two rows

**The 40k row.** Switch free-lens control on for **IL1, IL2, IL3 and PL1** and write down their
four DAC values (0–65535):

> IL1₀ = ______  IL2₀ = ______  IL3₀ = ______  PL1₀ = ______

**The DIFF row.** Switch free-lens control off, go to **DIFF** at about 40 cm (60 kV), write down the
same four values, and go back to MAG at the 40k row:

> IL1ᴅ = ______  IL2ᴅ = ______  IL3ᴅ = ______  PL1ᴅ = ______

Everything below is set with free-lens control **in MAG**.

## 2. The 20 measurements

Set the four lens values, wait a few seconds for the lenses to settle, then take **5 frames**:

| frame | deflector |
|---|---|
| 1 | no kick (shift and tilt at their centre values) |
| 2 | shift **+** |
| 3 | shift **−** |
| 4 | tilt **+** |
| 5 | tilt **−** |

**How big a kick:**
- Choose a change in the shift (or tilt) deflector that moves the beam about **50 px**, roughly a
  fifth of the frame.
- Use the **same size** for + and −, and go back to the centre value afterwards.
- If the beam would leave the camera, use a smaller kick.
- If the beam hardly moves even for a big change, **that is expected**: a tilt where the image is
  in focus, a shift near the DIFF row. Use a large change and record it anyway.

**The lens settings.** The same factors are applied to each row (rounded to a whole number):

| factors | IL1 | IL2 | IL3 | PL1 |
|---|---|---|---|---|
| a (the row) | 1 | 1 | 1 | 1 |
| b | **0.598** | 1 | 1 | 1 |
| c | 1 | 1 | **0.703** | 1 |
| d | 1 | 1 | 1 | **0.600** |
| e | 1 | 1 | **1.013** | **0.500** |
| f | 1 | **1.609** | 1 | **0.600** |
| g | 1 | **1.609** | **1.013** | 1 |
| h | 1 | **1.609** | 1 | **0.500** |

If a value would exceed 65535, use 1/factor for it instead.

**The order of the measurements:**

| measurement | setting | kicks |
|---|---|---|
| 1 | 40k row | x |
| 2–5 | 40k row × b, c, d, e | x |
| 6 | **40k row again** | x |
| 7–9 | 40k row × f, g, h | x |
| 10 | DIFF row | x |
| 11–14 | DIFF row × b, c, d, e | x |
| 15 | **DIFF row again** | x |
| 16–18 | DIFF row × f, g, h | x |
| 19 | **40k row again** | x |
| 20 | **40k row** | **y** instead of x |

The two rows cover image-like and diffraction-like settings evenly. The 40k row three times shows how
much the beam position wobbles; measurement 20 checks that a y kick acts like an x kick turned by 90°.

In total: 20 measurements × 5 frames = **100 frames**, plus the dark frame. Allow about two hours by
hand.

## 3. What to write down for every frame

One row per frame, in a spreadsheet or text file:

| file name | time | HT | IL1 | IL2 | IL3 | PL1 | shift x, y | tilt x, y | exposure | binning | note |
|---|---|---|---|---|---|---|---|---|---|---|---|

- Lens values: **read back** after setting them, not just typed in.
- Shift and tilt: the deflector values exactly as the console or PyJEM shows them (CLA1 and CLA2).
- Note anything odd: the beam cut by the frame edge, saturation, the spot not returning to the
  same place.

## 4. Checks while you go

- **The whole spot is on the camera** in every frame. Nothing may be cut off by the edge.
- **Not saturated.**
- **The spot is several pixels across** at every setting, also near the DIFF row, where it is
  smallest. A spot
  of a pixel or two "locks" to pixel centres and its measured position jumps instead of moving
  smoothly. Spread the beam a little if needed.
- **The 40k row comes back** to the same place in measurements 6 and 19, within about a pixel. If not,
  note how far it moved.
- **The deflectors return** to the same centre values.

Stage 1 is complete on its own: it gives the map from lens DAC values and deflector units to where
the beam goes on the camera. Physical units and lens positions come in stage 2 (a specimen and a
second HT).

---

## With PyJEM: the script does steps 1–5 for you, one step at a time

Copy **`acquire.py`** to the microscope PC. It needs only numpy and PyJEM. Do step 0 by hand
(setup, ending in MAG at the 40k row, free-lens control off). Then, in Python on the microscope PC,
one line at a time. Each step does one thing, prints what it did and refuses to run before the
steps it needs; you can stop between any two.

```python
from PyJEM import TEM3, detector
import acquire

acquire.check_connection(TEM3=TEM3, detector=detector)     # reads, one frame, small kicks; harmless

recording = acquire.Recording(TEM3=TEM3, detector=detector,
                              folder="D:/projection_response/2026-11-06")   # a new, empty folder
recording.read_diff_row(camera_length_cm=40)               # DIFF, read IL1–PL1, back to MAG
recording.take_over_lenses()                               # the 40k row into free-lens control
recording.plan_settings(beam_move_px=50,                   # each kick sized to move the beam about 50 px,
                        test_kick_units=300,                # from a test kick of 300 deflector units,
                        max_kick_units=5000)                # start cautious: at most 5000 units
recording.dark_frame()
recording.measure(setting="40k row", kicks_along="x")      # one measurement first, to see that it works
recording.measure_remaining()                              # the other 19
recording.restore()                                        # the microscope as it was
```

**What each step does:**
- `check_connection`: reads the HT, mode, lenses, free-lens control, deflectors and camera, takes
  one frame, moves CLA1 and CLA2 a little and puts them back. The lenses are not touched and
  nothing is saved. Its last line lists any problems. Run it first, and after any change to
  `acquire.py`.
- `Recording(...)`: creates the folder and saves the microscope state. Nothing changes.
- `read_diff_row`: switches to DIFF at the camera length, reads IL1–PL1, and goes back to MAG.
- `take_over_lenses`: reads the 40k row from the lenses as they are, writes it into free-lens
  control for IL1–PL1 and switches that on (a value left in free-lens control from earlier cannot
  creep in).
- `plan_settings`: prints the 20 measurements with their DAC values and the kick sizes, and saves
  them. Nothing moves.
- `dark_frame`: a frame with the beam blanked.
- `measure(setting=..., kicks_along=...)` / `measure_remaining()`: sets IL1–PL1 through free-lens
  control, takes a frame without a kick, then for CLA1 (shift) and CLA2 (tilt) a test kick, a
  +/- pair sized to move the beam about `beam_move_px` (at most `max_kick_units`), and the
  deflector back to its centre. A kick that takes the beam to the frame edge is halved and taken
  again. Every frame and a log line are saved as they are taken. After an error the deflectors go
  back to their centre.
- `restore`: puts the deflectors, free-lens control, blanking, function mode and HT back as they
  were at `Recording(...)`. Safe at any point, also after an error.

**What it prints:**
- the first frame's size and value range: check these look sensible;
- for every measurement, each action: the lens values written, the beam position without a kick,
  each test kick and the kick size it led to (in deflector units), how far each kick of the pair
  moved the beam (px), and the deflector back at its centre.

**Before relying on it, check on the real system:**
- the free-lens ids (IL1 10, IL2 11, IL3 12, PL1 14);
- that CLA1 is the beam shift and CLA2 the tilt (the analysis copes with any mix of the two);
- the camera's pixel type. Pass `dtype="<u2"` to `Recording` if the first-frame values look wrong.

**Afterwards:** copy the folder back. `model = analyse.fit_projection_response(folder=...)` fits the
model; `analyse.predict(model=model, dac=dac, shift=(dx, dy))` predicts beam moves.

**If there is time, test the model at the microscope** at new lens settings within the range of the
9 (for example from `analyse.new_settings`):

```python
new = acquire.Recording(TEM3=TEM3, detector=detector, folder="D:/projection_response/new_settings")
new.take_over_lenses()
new.plan_new_settings(settings=settings, beam_move_px=50, test_kick_units=300, max_kick_units=5000)
new.dark_frame()
new.measure_remaining()
new.restore()
```

`analyse.compare(model=model, folder=new.folder)` then shows measured against predicted.

Dry run on the virtual NEOARM at 60 kV (`stage1_rehearsal.ipynb`): χ²/dof 0.7. The fit found
the virtual camera's mirror on its own. At 8 lens settings stage 1 never used, predictions matched
the measurements to within 0.002 px.
