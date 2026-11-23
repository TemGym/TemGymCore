# A model of the JEOL NEOARM, and scripts that use it

This folder holds the first version of a TemGym model of the JEOL NEOARM at ER-C, fitted to what
the beam does on the SightX camera, and the PyJEM scripts that record, correct and align the
microscope. Every script runs on a simulated NEOARM first; the same text runs at the microscope
with one switch changed. The requirements are in `design/requirements.md`, the acceptance examples
in `design/examples.md`, and the record of how this version was built in
`design/changes/neoarm-first-version.md`.

## Before you start

- Run everything from this folder, with the repository's Python environment (`../../.venv`).
- The scripts talk to the microscope through jemwrap (`../../jemwrap/src`), which spaces every
  call to the microscope and refuses values that a script must not change.
- Each script has `SIMULATED = True` at the top. Then `import PyJEM` finds the simulated NEOARM of
  `simulated_neoarm.py`, a microscope that hides its own numbers, and nothing touches a real
  instrument. At the microscope the switch is `False`; nothing else changes.
- A rehearsal (a run on the simulated NEOARM) runs on the CPU: put `JAX_PLATFORMS=cpu` in front
  of `python`, for example `JAX_PLATFORMS=cpu ../../.venv/bin/python centre_beam.py`. JAX uses
  every core it finds. On the 2026-11-22 machine `centre_beam.py` and the hysteresis scripts took
  seconds, `shift_balance.py` under a minute and `tilt_balance.py` about 10 minutes. The fit of
  notebook 2 runs best on a GPU (about 16 minutes there).
- The notebooks open in Jupyter with the kernel of `../../.venv`; the `.ipynb` files hold the
  outputs of their last run, so they can be read without running them.
- The simulated NEOARM can hold a specimen: `install_simulated_neoarm(specimen=CROSS_GRATING)`
  shows its shadow (rays), and `wave_beamlets=127` images it with one coherent Gaussian beam
  through the whole column (about 0.08 s a frame on the GPU after a first compilation of about
  9 s). `condenser_hole=4` selects the 10 um condenser aperture, which limits the beam.
- Each rehearsal starts a new simulated NEOARM, in the state of the console's MAG 40k row. Nothing
  carries over from one script to the next; at the microscope it does.

Words used below:

- A **row** is one line of the console's magnification table, chosen by its selector number: MAG
  rows 6 to 24 are X8000 to X500k, DIFF rows 0 to 14 are camera lengths 8 cm to 2 m.
- **Kept-back** settings are frames that the fit never sees; the fitted model predicts them, and
  the checks compare.
- E1, E4 and so on are the acceptance examples of `design/examples.md`; R, S, D and M numbers are
  requirements in `design/requirements.md`. Lengths in the model are in L, the distance from the
  specimen to the SightX.
- A **lag** is how far a lens's field trails its DAC value, because its iron remembers where it
  came from (hysteresis).
- Comments that name a stage (stage 8) or a decision refer to the build record,
  `design/changes/neoarm-first-version.md`; "the principal" is the scientist the work is for.

## The order

The work has two parts. First the alignment scripts bring the microscope into a known state,
with no model. Then the recording and the fit give a predictor of the microscope in that state.

### Part 1: alignment by recipe

1. **`alignment.ipynb`: the alignment guide, step by step.** It quotes each step of
   `design/alignment_guide.md`, shows what the step looks at, runs the step's script on one
   simulated NEOARM and checks the step's end point. It runs every step twice, as the guide
   advises. Steps without a script say why.
2. **The scripts it runs**, the same files that run at the microscope:
   - `centre_aperture.py`: the condenser aperture selected on the console, centred so that CL3
     through the crossover does not move the beam (E21).
   - `tilt_balance.py` and `shift_balance.py`: the condenser pivot points (E10). The tilt balance
     comes first, as in the guide. The shift balance works in DIFF and keeps the pattern in the
     middle with the projector deflector PLA. A balance whose wobble moves the beam less than half
     a pixel stays as it is.
   - `ht_centre.py`: the condenser alignment tilt (CLA2) with the HT wobbler, on a specimen with
     features in focus (E24). It stops where CLA2 cannot lower the swing and reports the rest.

### Part 2: the predictor

3. **`record_lens_changes.py`: the recording.** It goes through the MAG rows X8000 to X500k and
   the DIFF camera lengths 8 cm to 2 m. It centres the beam (CLA1 in MAG, PLA in DIFF), kicks it
   (CLA1 in MAG, CLA2 in DIFF), moves each projector lens and the objective's fine focus, changes
   the spot size, and brings lenses to their values from below and, at a few rows, from above. It
   saves every frame and a table in `output/record_lens_changes/<time>/`. The recording that
   notebook 2 uses is `recordings/lens_changes_simulated_corrector_off_2026-11-22.csv`.
4. **`neoarm_fit.ipynb` (notebook 2): the fit.** It fits the model to the recording and predicts
   settings that the fit never saw (E1, E2, E3, E9, E14, E15, E17, E19). It writes the fitted
   numbers to `neoarm_fit.json` and the deflector response of each row to `neoarm_response.json`.
   It tries the corrector off, imaging with -1, and (by reversed kicks) with +1. At its end it
   turns the model's pixels into nm and mrad at the specimen with the rulers (E23).
5. **`record_rulers.py`: the rulers**, one run each. RULER = "grating" records the cross grating's
   463 nm period at MAG X40k. RULER = "gold" records the gold 111 ring at DIFF row 10.
6. **`centre_beam.py`: the correction.** When a prediction leaves the beam off the middle, it
   moves the beam back in one step with CLA1 (MAG) or PLA (DIFF), using `neoarm_response.json`
   (E4).

### Other tools

- `neoarm_model.ipynb` (notebook 1): the column as TemGym components, with the start values of
  the fit.
- `alignment_window.ipynb`: runs one script and shows, while it runs, the fitted column, the
  SightX frame with the model's beam, and what the script prints (E11).
- `record_hysteresis.py` and `analyse_hysteresis.py`: the lens lags, measured directly.
- `microscope_functions.py`: the functions that more than one script needs. They find the beam,
  centre it, find a lost beam, wobble a deflector, set a balance, find the smallest spot, bring a
  lens from below, and measure a grating's period, a powder ring's radius and a shift of features.

## At the NEOARM for the first time

The scripts have run only on the simulated NEOARM. Before the first real run:

1. In each script you run, set `SIMULATED = False`. Nothing else changes.
2. Load a recent alignment on the console, TEM mode, alpha as in the recording (alpha 1). Insert
   the SightX at binning 4. Check that the beam is on the SightX at MAG X40k.
3. Run `cla1_kicks.py` first. It kicks CLA1 by small amounts and sets it back, and it shows that
   jemwrap reaches the microscope and that a restore works.
4. Run `record_hysteresis.py` and `analyse_hysteresis.py`. The lags tell whether the fitted
   model's history (every lens from below) holds on the real lenses.
5. Before `centre_aperture.py`, select on the console the condenser aperture to centre. Check that
   the aperture the console shows as selected is the one Apt3 moves (jemwrap guards
   `GetExpKind`; Apt3 moves the kind set by `SelectKind`). If they differ, do not run it.
6. Then the alignment scripts in the order of the alignment notebook, then
   `record_lens_changes.py` (it takes longer than on the simulator: every call to the microscope
   waits 0.1 s), then notebook 2 on a GPU, then `record_rulers.py`.
7. Every script restores the values it changed, also after Ctrl-C. A script keeps only the value
   that is its result (the aperture position, a balance, CLA2, CLA1 or PLA).

What may differ at the real NEOARM:

- The HT wobbler's amplitude and frequency are not known. The simulated wobbler changes side at
  each frame. A real one swings continuously, so a frame catches some point of the swing, or a
  blur of it. `ht_centre.py` measures the spread of the features' positions over six frames.
  Check by eye that the swing it reports matches what the screen shows.
- The wobble amplitudes of the balance scripts (`WOBBLE_DAC`) were chosen on the simulator. A
  balance whose wobble moves the beam less than 0.5 px is left as it is, and E10 then passes.
  If the real wobble is too small to see, raise `WOBBLE_DAC` before trusting that pass.
- The scripts that converge a spot in DIFF (`shift_balance.py`, the gold ruler) do not limit the
  dose on the SightX. Check the exposure first.
- `centre_aperture.py` assumes that CL3 at X8000 puts the beam near its crossover, and it waits
  0.2 s after each aperture move. The aperture motor may need longer.
- The beam-shift and beam-tilt ranges of CLA1 and CLA2 are assumptions (about 50 um and 38 mrad
  over half the range). The real recording of 2026-11-10 agrees for CLA1.
- In DIFF at the shortest camera lengths the recording kicks CLA2 by up to 20000 units, about
  20 mrad of beam tilt on the simulator.

## What to expect on the simulated NEOARM

The simulated NEOARM of 2026-11-22 has the corrector off and a condenser matched to the console's
spot table. It starts from a recent alignment that is nearly right, with every lens from below.

- `alignment.ipynb`: the aperture 0.6 px through the crossover before and after (the simulated
  aperture hardly sets the beam's axis); the tilt wobbler 0.0 px (TiltBal stays, its error does
  not show); the shift wobbler 1.6 px before and 0.2 px after; the condenser tilt with the HT
  wobbler 5.9 px before and after (CL1 and CL3 sit off the axis, which the gun and spot alignments
  would set). The second pass changes nothing.
- Notebook 2: the best of 72 starts reaches a misfit of 457 (corrector -1, mirror -1). Kept-back
  MAG rows within 0.18 px rms, DIFF rows within 0.05 px, beam sizes within 0.5 % (E1, E2); several
  lenses moved at once within 0.3 px (E3); spot 0 within 1.1 px (E14); the focus series within
  5.4 px (E15). E9 fails: the refits with CL1 moved 5 mm and with the objective's focal length
  moved 20 % end at a misfit of about 23 900 and do not find the best fit's valley again. The
  linear estimate of the same check moves the predictions by 0.03 and 0.58 px. Open. The grating
  ruler gives 7.93 nm per px at MAG X40k. The whole notebook takes about 40 minutes on the GPU.
- `record_rulers.py`: the grating's period within 1.2 % of the hidden column's. The gold ruler
  does not find rings on this simulator.

## What this version does not do

- The model does not follow a change of TiltBal or ShiftBal, and `neoarm_response.json` holds
  only for the balances of the fitted recording.
- The model's beam assumes that every lens came to its value from below, as in the recording.
- The gun, spot alignment, stigmator, diff focus and image focus steps of the alignment guide have
  no script.
- The gold ruler (angles in mrad) does not work on the simulated NEOARM yet.
- `find_lost_beam` (in `microscope_functions.py`) has no caller yet. It is the routine for a beam
  lost off the SightX.
- Nothing here has run on the real NEOARM yet.
