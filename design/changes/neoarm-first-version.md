# Build plan: the first version of the NEOARM predictor

Date: 2026-11-19. Status: stages 1, 2, 2b, 3 and 4 are built (2026-11-19). The build stops
after stage 4 for the principal's review of the jemwrap interface (B3).

This plan says what the first version contains, in which order the agent builds it, and
how each stage is checked. Requirement and example numbers refer to `../requirements.md`
and `../examples.md`.

## What the first version is

A person can open two notebooks and a folder of PyJEM scripts, read each from top to
bottom, and see a non-dimensional TemGym model of the NEOARM that is fitted through PyJEM
calls and that follows the microscope through the first alignments. Everything runs on
the simulated microscope. The same scripts are ready to run on the real microscope.

## Stages

The agent stops after each stage and shows the result. The principal reviews stage 2 in
full, because it is the first test of whether the three skills give code the principal
likes.

| Stage | What is built | Checked by |
|---|---|---|
| 1 | **Checks.** The comment check and a check for scipy imports run on the new folder. `ruff` runs on the new folder. | The checks fail on a bad sample file and pass on a good one. |
| 2 | **Notebook 1, the model** (D1). The column as plain TemGym components: one list of components that change (M1, C6) and one list of static components (M2, M9). Direct `run_to_end` calls. The column drawn with the legacy plot. Start values from the console data (R30) and the drawing (R31). | E8, E12. The agent opens each figure and checks the text. |
| 2b | **A condenser that follows the real microscope** (C1). `fit_condenser.py` fits CL1, CL3 and CM to the real CL3 data (C8). Spot alignment is a deflector in notebook 1 (C6). The realistic lens offsets of the hidden microscope move to stage 3, where the simulated microscope is built from notebook 1's model. | E13. |
| 3 | **jemwrap on the simulated microscope** (J6). A small file builds the simulated microscope from notebook 1's model. TemGym's `pyjem_simulator` gets what jemwrap reads and it lacks, such as a `sightx` module. jemwrap's monitor runs on it. | jemwrap's state table shows the simulated lens and deflector values with no error on the values this project uses. |
| 4 | **Control in jemwrap** (J9, J12). On a branch of the jemwrap repository, in its author's style: a call that sets one value, the list of values that may be set, a call that reads one value at once, a call that restores from a snapshot. Tests in jemwrap's `tests/`. jemwrap's `requirements.md` updated. Then the PyJEM skill is rewritten around jemwrap, with an example script that runs. | jemwrap's tests pass. A set call outside the list is refused. E6, E7. |
| 5 | **The recording script.** One script of plain steps that uses jemwrap to size and centre the beam, kick the deflectors (SpotA too) and change the projection lenses. Each projector lens moves up and down at several MAG rows and several camera lengths, so that the recording decides the lens law (M10) and R22 in DIFF. | E5, E6, E7. |
| 6 | **Notebook 2, the fit** (D2, D3). The non-dimensional model fitted with JAX to the recording of stage 5. Tested on rows it never saw. The fit run twice with two static values. | E1, E2, E3, E9, E14. |
| 7 | **The correction routine** (S3). A short script that moves the beam back to the centre with the fitted model. | E4. |
| 8 | **Pivot points** (M8). CLA1 and CLA2 written as TemGym's `DoubleDeflector`, which already has tilt balance and shift balance. Two scripts from the alignment guide: tilt balance and shift balance. | E10. |
| 9 | **The live window** (D6). One notebook for each alignment script. The column view and the detector view are drawn from jemwrap's state, so the same window works on the simulated and on the real microscope. | E11. |
| 10 | **The hysteresis recording** (H6). One script that reaches the same lens values by different routes, and a short analysis of the lag in pixels. | It runs on the simulated microscope and recovers the lag that the simulator has. |
| 11 | **Review and hallway test.** An agent that did not write the code reviews it against the skills. A second fresh agent works through the notebooks from the README alone. | Their reports, with every finding answered. |

After stage 11 the first version goes to the real microscope: the recording of stage 5 and
the recording of stage 10. Before that visit, the author of jemwrap reviews the branch of
stage 4. That visit resolves U1 and H5.

## Decisions

| Decision | Who decided | Can it change later |
|---|---|---|
| Static components are ordinary TemGym components in a named list (M9). | The principal | Yes |
| Centre within 10 px, size within 20 % (R20, R21). | The principal | Yes |
| Pivot points are the first alignment. Gun, stigmators, corrector tilt and aperture are left out (M8). | The principal | Yes |
| Spot alignment is one fitted deflector, read from the console (C6). | The principal | Yes |
| Hysteresis is avoided and measured. It is fitted only above 10 px (H6). | The principal | Yes |
| Scripts handle the microscope through jemwrap. jemwrap gets control calls in its author's style (J7, J9, J12). | The principal | Yes |
| The live view is drawn from jemwrap's state, not by the script (D6). | The agent proposed, the principal has not confirmed | Yes |
| The sample-plane scale is the nominal magnification (M7). | The principal | Yes |
| The lenses are `PolePieceLens` components, Lenz's field with a gap and a bore (M10). | The principal, 2026-11-19 | Yes |
| The console table sets the MAG focus and the mean MAG and DIFF scale only (M11). | The principal, 2026-11-19 | Yes |

## Must, should, will not

- **Must:** stages 1 to 6 (with 2b) and 8. Examples E1, E2, E3, E5, E6, E7, E8, E9, E13.
- **Should:** stages 7, 9, 10 and 11. Examples E4, E10, E11, E12, E14.
- **Will not, in this version:**
  - micrometres on the sample from a ruler (U2);
  - a fitted hysteresis model;
  - the gun, the stigmators, the corrector and the aperture;
  - a second camera;
  - a rewrite of the existing recording scripts and `neoarm.py`;
  - any change to the files of the other agent (`neoarm_forward*`, `neoarm_predictor.ipynb`).

## Answers to the questions that blocked the start (2026-11-19)

- **B1** The new work lives in a new folder, `examples/neoarm/`.
  `examples/microscope_inversion/` stays as it is until the new version passes its
  examples.
- **B2** The agent may add the checks of stage 1 to `.pre-commit-config.yaml` and the new
  folder to `ruff` in `pyproject.toml`.
- **B3** After the principal accepts stage 2, the agent may run the later stages without a
  stop between them, with a ceiling of $20 at list price. The agent also stops after
  stage 4, so that the principal sees the jemwrap interface before scripts use it.

## Assumptions the build starts on

- "Rough behaviour" of the condenser means what C2 and example E13 say. The principal has
  not confirmed the numbers in E13.

- Nine items in the requirements are taken from earlier sessions and are not yet confirmed
  by the principal: R7, R8, S1, R13, R17, R18, R23, R27, M4.
