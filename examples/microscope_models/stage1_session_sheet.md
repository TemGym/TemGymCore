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

## 1. Read the 40k-row lens values

Switch free-lens control on for **IL1, IL2, IL3 and PL1** and write down their four DAC values
(0–65535):

> IL1₀ = ______  IL2₀ = ______  IL3₀ = ______  PL1₀ = ______

Every other setting is a fixed multiple of these.

## 2. Read the diffraction setting

1. Switch to **DIFF** mode. Choose a camera length that would put gold rings well inside the
   camera (for stage 2 later). The camera length that matters is at *our* camera, which may differ
   from the console's number. Roughly: about 0.4 m at 60 kV, about 0.8 m at 200 kV.
2. **The diffraction setting = the four DIFF-mode values of IL1, IL2, IL3, PL1.** Write them down
   and go back to MAG. Stage 1 sets these four values with free-lens control **in MAG mode**.

## 3. The 12 visits

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
- If the beam hardly moves even for a big change, **that is expected** at some settings:
  - **tilt at setting 0**, where the image is in focus;
  - **shift at the diffraction setting**.

  Use a large change and record it anyway.

**The lens settings.** Each value is the multiple × the 40k-row value, rounded to a whole
number:

| setting | IL1 | IL2 | IL3 | PL1 |
|---|---|---|---|---|
| 0 | 1 × IL1₀ | 1 × IL2₀ | 1 × IL3₀ | 1 × PL1₀ |
| 1 | **0.598** × IL1₀ | 1 | 1 | 1 |
| 2 | 1 | 1 | **0.703** × IL3₀ | 1 |
| 3 | 1 | 1 | 1 | **0.600** × PL1₀ |
| 4 | 1 | 1 | **1.013** × IL3₀ | **0.500** × PL1₀ |
| 5 | 1 | **1.609** × IL2₀ | 1 | **0.600** × PL1₀ |
| 6 | 1 | **1.609** × IL2₀ | **1.013** × IL3₀ | 1 |
| 7 | 1 | **1.609** × IL2₀ | 1 | **0.500** × PL1₀ |
| diff. | the four DIFF-mode values from step 2 | | | |

("1" means unchanged from the 40k row.) Check that nothing exceeds 65535. At 200 kV the largest is
IL2 at 1.609 × 24 862 ≈ 40 000, so it fits; at 60 kV every value is about half as large.

**The order of visits:**

| visit | setting | kicks |
|---|---|---|
| 1 | 0 | x |
| 2–5 | 1, 2, 3, 4 | x |
| 6 | **0 again** | x |
| 7–9 | 5, 6, 7 | x |
| 10 | diffraction | x |
| 11 | **0 again** | x |
| 12 | **0** | **y** instead of x |

Repeating setting 0 three times shows how much the beam position wobbles when nothing should have
changed, including how well the lenses return. Visit 12 checks that a y kick moves the beam exactly
like an x kick turned by 90°.

In total: 12 visits × 5 frames = **60 frames**, plus the dark frame. Allow about an hour by hand.

## 4. What to write down for every frame

One row per frame, in a spreadsheet or text file:

| file name | time | HT | IL1 | IL2 | IL3 | PL1 | shift x, y | tilt x, y | exposure | binning | note |
|---|---|---|---|---|---|---|---|---|---|---|---|

- Lens values: **read back** after setting them, not just typed in.
- Shift and tilt: the deflector values exactly as the console or PyJEM shows them (CLA1 and CLA2).
- Note anything odd: the beam cut by the frame edge, saturation, the spot not returning to the
  same place.

## 5. Checks while you go

- **The whole spot is on the camera** in every frame. Nothing may be cut off by the edge.
- **Not saturated.**
- **The spot is several pixels across** at every setting, including the diffraction setting. A spot
  of a pixel or two "locks" to pixel centres and its measured position jumps instead of moving
  smoothly. Spread the beam a little if needed.
- **Setting 0 comes back** to the same place on visits 6 and 11, within about a pixel. If not,
  note how far it moved.
- **The deflectors return** to the same centre values.

Stage 1 is complete on its own: it gives the predictor. Turning it into physical units and lens
positions (a specimen for the units, a second HT for the positions) comes later.

---

## With PyJEM: the script does steps 1–5 for you

Copy **`acquire.py`** to the microscope PC. It needs only numpy and PyJEM. Do step 0 by hand
(setup, ending in MAG at the 40k row, free-lens control off). Then, in Python on the microscope PC:

```python
from PyJEM import TEM3, detector
import acquire

folder = acquire.stage1(TEM3, detector, "D:/stage1/2026-11-06",   # a new, empty folder
                        camera_length_cm=40,              # DIFF row of the diffraction setting (60 kV)
                        max_units=(5000, 5000))           # start cautious: largest deflector change
```

**What the script does:**
- reads the diffraction setting: switches to DIFF at the camera length, reads IL1–PL1, and goes back
  to MAG;
- reads the 40k row from the lenses as they are, writes it into free-lens control for IL1–PL1 and
  switches that on (a value left in free-lens control from earlier cannot creep in);
- takes a dark frame;
- does the 12 visits, auto-scaling each kick to about 50 px and halving any kick that cuts the
  beam at the frame edge;
- saves every frame and a log line after each one;
- puts the lenses, deflectors and blanking back at the end, even after an error.

**What it prints:**
- the first frame's size and value range: check these look sensible;
- one line per visit.

**Before relying on it, check on the real system:**
- the free-lens ids (IL1 10, IL2 11, IL3 12, PL1 14);
- that CLA1 is the beam shift and CLA2 the tilt (the analysis copes with any mix of the two);
- the camera's pixel type. Pass `dtype="<u2"` if the first-frame values look wrong.

**Afterwards:** copy the folder back. `model = analyse.stage1(folder)` fits the model;
`analyse.predict(model, dac, shift=(dx, dy))` predicts beam moves.

**If there is time, check the model at the microscope.** Pick some new lens settings within the
range of the 9, and run:

```python
check = acquire.check(TEM3, detector, "D:/stage1/check", settings, offsets=(2000, 2000))
```

`analyse.compare(model, check)` then shows measured against predicted.

Dry run on the virtual NEOARM at 60 kV (`stage1_rehearsal.ipynb`): χ²/dof 0.6. The fit found
the virtual camera's mirror on its own. At 8 lens settings stage 1 never used, predictions matched
the measurements to within 0.009 px.
