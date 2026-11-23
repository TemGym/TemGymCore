# Lens hysteresis: what is known, and what we do about it

Written 2026-11-21 after the stage 6b fit of the NEOARM model found a second minimum held up by
implausible lens lags (thousands of DAC units). Sources are listed at the end.

## Why it matters here

The iron of a lens remembers: after a lens's DAC value rises, its field is a little weaker than
the value should give; after it falls, a little stronger. The model of notebook 2 brings every
lens to its value from one direction and fits one constant lag per lens (`lens_dac - lag`). That
holds only if the approach wipes the lens's memory. The recording never sees the other branch,
so the lag trades against the lens's coil strength, and the fit can find a second minimum with
lags of thousands of DAC (CL3 6890, IL2 2320 in the harder simulated test).

## What the literature says

- Uncontrolled, it is large. JEOL's glossary: hysteresis of the intermediate lens gives a
  magnification error of 5 to 10 %, depending on the operation mode. Objective and projector
  lenses: "not a problem" because their excitations are almost fixed. Condenser: it affects the
  reproducibility of the probe's position (no number). Calibration guides: manufacturer
  magnifications are off by up to 10 %; approach a setting from the same direction.
- JEOL condenser, in practice (de la Cruz et al. 2017, JEM-3200FSC): switching low-dose modes moved
  the beam off the detector at 20 000x. The fix: through Free Lens Control, ramp CL1, then CL3,
  to its maximum and back, twice in quick succession. No size in nm given. The intermediate and
  projector lenses were not affected for recall.
- Controlled, it is tiny (Czigany and Kovacs Kis 2022): fixed specimen height, illumination and
  lens currents gave a camera length reproducible to better than 0.03 % between sessions; lens
  currents equal to the last displayed digit.
- The iron is never fully reset (TU Eindhoven SEM study, 2010): the pole piece is not saturated,
  so "the major loop can not be reached"; the field depends on history, and a large pulse left an
  error larger than a small focus sweep could correct. Numbers not extracted.
- None of the sources gives the lag in DAC units or a settling time.

## The published ways to handle it

1. Approach every setting from the same direction (textbook practice; the cheapest).
2. Normalise: ramp the lens to its maximum and back, once or twice (JEOL users, vendor
   "normalise"/"standard focus" buttons).
3. Demagnetise with a decaying oscillation (SEM "DeGauss", the NEOARM's lens relaxation, about
   15 s): the cleanest reset, the slowest.
4. Model the hysteresis (Preisach or Prandtl-Ishlinskii models, as in the TU Eindhoven control
   work) and correct for it from the history of DAC values: research grade, needs the loop measured.
5. Close the loop on the image: measure the beam and correct (autofocus, beam centring).

No source compares them. A plan that uses several: a deterministic history (2 or 1, with 3 only
when a lens must go back the other way), one constant lag per lens in the model, and the image as
the final check (stage 9's live window).

## What the first real session should measure

- Repeatability: set the same row twice, away and back; the beam should land within a fraction
  of a pixel.
- Approach size: approaches of 500, 2000 and 5000 DAC, and to maximum; the frame stops changing
  once the approach is big enough.
- The lag: at a low and a high DAC value of each lens, approach from below and from above; the
  jump is twice the lag, and shows whether the lag changes with the setting.
- Settling: frames 1, 5 and 30 s after a change.

## Test with wider loops (2026-11-21, simulated)

The hidden microscope's loops were replaced, before any fit, by six play elements per lens
(a Prandtl-Ishlinskii model, the symmetric case of Preisach) with widths 30 to 10 000 DAC and the
largest lag 210 to 390 DAC by lens. Two elements are wider than the 2000-unit approach, so the
approach does not wipe the memory: for IL2 at 44 000 it lands at a lag of 123 to 390 DAC,
depending on what came before. Same recording script, same fit (one constant lag per lens).

- Every acceptance check passes. Kept-back MAG rows: 0.72 px rms, 1.5 px at most (0.06 px with
  the narrow loops); DIFF rows 0.35 px rms; size 2.0 %; spot 0 1.0 px; MAG focus series 1.61 px
  rms, 3.4 px at most. E9 passes. Misfit 1581 against 112.
- The fitted lags are not the lenses' (IL3 2371, IL1 1302, IL2 -765 DAC against true largest lags
  of 270 to 390). They are numbers of the model, as the gauge numbers are; the predictions hold.
- Again only 1 of 6 objective starts reaches the best misfit.
- Not tested: a lens history unlike the recording's (a person turning knobs, another order of
  rows). The kept-back frames were taken with the same procedure as the fitted ones. With
  another history the spread above (about 270 DAC for IL2) is what the model could miss by.

## Sources

- JEOL glossary, hysteresis: https://www.jeol.com/words/emterms/20121023.084957.php
- de la Cruz et al., condenser hysteresis in a JEOL JEM-3200FSC: https://www.biorxiv.org/content/10.1101/153395v1.full
- Czigany and Kovacs Kis, accuracy of SAED: https://pmc.ncbi.nlm.nih.gov/articles/PMC10087671/
- Electron microscopy experiments concerning hysteresis in the magnetic lens system (TU Eindhoven, 2010): https://pure.tue.nl/ws/files/3180256/Metis242256.pdf
- Control of dynamics and hysteresis in electromagnetic lenses (TU Eindhoven; not read): https://pure.tue.nl/ws/files/2934922/711031.pdf
- Williams and Carter, TEM chapter 9: https://iubemcenter.indiana.edu/doc/williams-and-carter/transmission-electron-microscopy-chapter-09.pdf
- Globalsino, hysteresis removal: https://www.globalsino.com/EM/page4355.html
- McCaffrey, Calibrating the TEM: https://cambridge.org/core/journals/microscopy-today/article/calibrating-the-tem/BB8F3C36FD418D78E3A807C8007601DE
