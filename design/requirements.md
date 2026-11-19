# Demand and requirements: a predictive model of the NEOARM

Author: Claude. Principal: the project lead. Date: 2026-11-19.
Interviewed: the principal in person, and from the record (the working sessions of
2026-11-16 to 2026-11-19, the notebooks in `examples/microscope_inversion/`).

Markers per item:
- `[answered]` said or confirmed by the principal;
- `[record]` taken from earlier work, not yet confirmed;
- `[assumed]` proposed by the author;
- `[unknown]` listed in 2.2 with what would resolve it;
- `[deferred]` listed under open questions.

To change this document, edit it or tell the agent which item is wrong. Acceptance examples
are in `examples.md`. Users are in `stakeholders.md`.

## 1. Demand

### 1.1 Problem and context

- R1 `[answered]` The principal wants to keep track of the beam on the sample plane and
  decide where the beam should go, in absolute units.
- R2 `[answered]` The principal wants to do some of the alignments automatically.
- R3 `[answered]` The work must verify the whole concept: that a model of the microscope
  can predict what the beam will be at various stages of the column, reliably enough to
  build on.
- R4 `[answered]` With a reliable, predictable model, a layer of control scripts is
  written around it.
- R5 `[answered]` The model lets the principal design new experiments and their
  calibration routines. Example: near-field ptychography, where the beam is scanned across
  the sample in micrometre or nanometre steps and IL1 is changed to defocus the beam. That
  experiment needs a calibration routine and a prediction model that can be relied on.
- R6 `[answered]` Offline tools are used to design scripts that then go to the microscope
  and control it.

### 1.2 Current solutions

- R7 `[record]` Today the beam is positioned by hand at the console. The console values
  (DAC values) have no stated relation to position on the sample or on the camera.
- R8 `[record]` `examples/microscope_inversion/` holds a model notebook
  (`neoarm_model.ipynb`), an offline rehearsal (`neoarm_predictor_offline.ipynb`) and
  recording scripts. The principal finds the code hard to read (wrappers, a `scope`
  object). Source: sessions of 2026-11-18 and 2026-11-19.
- R9 `[unknown]` Prior art outside this project has not been searched.

### 1.3 Benefits

- R10 `[answered]` Automatic alignment, beam tracking in absolute units, and experiments
  that cannot be set up by hand (R5).

### 1.4 Users

See `stakeholders.md`.

- R11 `[answered]` For now the principal and one colleague. Later, other people.
- R12 `[answered]` The present goal is to prove that the approach works.

## 2. Requirements

### 2.1 Use scenarios

- S1 `[record]` **Calibrate.** At the microscope, a plain PyJEM script records frames while
  it changes lenses and deflectors. The model is fitted to the recording. Examples: E1, E2.
- S2 `[answered]` **Predict.** From console values, the model gives the beam's centre and
  size on the SightX camera. Examples: E1, E2, E3.
- S3 `[answered]` **Recover.** When a prediction is off, a small calibration routine moves
  the beam back to the centre. The beam does not have to land perfectly the first time.
  Example: E4.
- S4 `[answered]` **Rehearse offline.** Every script that will run at the microscope is
  designed and tried on the simulated microscope in TemGym first. Example: E5.
- S5 `[deferred]` **Track on the sample plane in absolute units** (R1).
- S6 `[answered]` **Automatic alignments** (R2). The alignments are the ones in the
  principal's guide, `alignment_guide.md`. The fitted model follows each alignment, and a
  PyJEM script demonstrates each alignment on the model microscope. Which alignments come
  first is open (Q2).
- S7 `[deferred]` **Experiment design**, such as near-field ptychography (R5).

### 2.2 Demand uncertainties

- U1 `[unknown]` Whether a model fitted in the rehearsal reaches the same accuracy on the
  real NEOARM. Resolved by the first real recording.
- U2 `[unknown]` How absolute units on the sample plane are obtained. Frames from one
  camera cannot give them without a ruler (a grating image, diffraction rings or a second
  camera). Source: `neoarm_model.ipynb`, section 7.
- U5 `[unknown]` Whether the choice of a static value changes any prediction. Resolved by
  example E9: fit twice with two different static values and compare the predictions.
- U4 `[answered]` The model must be clear about which components are static and which
  change. The model must not depend on things that cannot be known at the real microscope.
  The principal wants a round of questions on how the TemGym model is represented before
  S5 is settled (Q6).
- U3 `[unknown]` The condenser model follows the measured focus curve to 3 % but the
  rehearsal cannot use it yet. Source: session of 2026-11-18.

### 2.3 Interfaces with existing systems

- R13 `[record]` The microscope is a JEOL NEOARM, controlled through PyJEM (`TEM3`,
  `detector`). The camera is the SightX ("screenCamera").
- R14 `[answered]` PyJEM must not be called too often. Too many calls in a short time can
  crash it. A crash is not serious, but scripts must pace their calls.
- R15 `[answered]` After a big change the lenses need time to settle, for example 1 or 2
  seconds. The wait is an option at the top of the script. A big change is a change
  between MAG mode and DIFF mode.
- R16 `[answered]` The shortest time between two PyJEM calls is 0.1 s, for now.
- R28 `[answered]` The SightX camera is 12 cm wide. This is one known length in real
  units. `[record]` The simulated camera has 1916 pixels of 62.6 µm, which is 12 cm.
- R17 `[record]` The model is built from TemGym components. Optics that TemGym lacks are
  added to TemGym (`.claude/skills/temgym/SKILL.md`).

- R30 `[answered]` Data from the real microscope exists: the console's DAC values at
  various settings (the console JSON, read with `record_console_tables.py`). The fitted
  model uses this data for the MAG focus and the mean scale of MAG and DIFF (M11).
- R31 `[answered]` The engineering drawing of the column is a loose guide to start from.
  `[record]` It is `examples/virtual_neoarm/neoarm_column_diagram.png`, with positions read
  by hand in `neoarm_column_diagram.json`.

### Watching the microscope with jemwrap

- J1 `[answered]` The principal has a package, jemwrap, that reads the state of the
  microscope through PyJEM in a clearer way. It is read-only: it does not control the
  microscope. It makes it much easier to see what a PyJEM script does.
- J2 `[answered]` Monitoring in this project may need to go through jemwrap. The PyJEM
  skill may need a revision for this. The build does not start before this is settled.
- J3 `[record]` From jemwrap's own requirements document: a background monitor polls the
  microscope and keeps a typed state (lenses, deflectors, apertures, stage, EOS, SightX and
  more). All its reads go through one rate limiter, 10 Hz by default. A value is polled
  every 15 s, and every 1 s while it changes. PyJEM modules can be injected, so it can run
  on an offline or simulated microscope. It can serve the state as a web page.
- J7 `[answered]` jemwrap is in the folder `jemwrap/` of this checkout. The principal wants
  jemwrap changed so that it has all the calls this project needs and is a clean, safe
  interface to the microscope. Scripts use jemwrap to handle the microscope. This replaces
  the rule of plain PyJEM calls in the PyJEM skill (R27), once the design below is agreed.
- J8 `[record]` jemwrap today has no call that changes the microscope. Its endpoint table
  already records, for each value, the PyJEM setter name, the type and the allowed range.
  Its rate limiter serialises calls and keeps a minimum time between them. Source:
  `jemwrap/src/jemwrap/endpoints.py`, `rate_limit.py`, `requirements.md`.
- J9 `[answered]` The design for control in jemwrap, accepted on the condition of J12:
  1. One call sets one value. The call checks the type and range from the endpoint table,
     goes through the same rate limiter as the reads, reads the value back and returns it.
  2. Only values on an explicit list can be set: the lenses, CLA1, CLA2, SpotA, the tilt
     and shift balances, the magnification selector, the mode, the spot size and alpha.
     jemwrap refuses the stage, the high voltage, the gun, the apertures, the vacuum and
     the detector insertion (R25).
  3. One call reads one chosen value at once, so that a script does not use a value that
     is 15 s old.
  4. One call restores the settable values from a snapshot taken at the start (E6).
  5. The web page stays read-only. No call that changes the microscope is reachable from
     the network.
- J13 `[record]` What stage 4 built (2026-11-19), on branch `control` of the jemwrap
  repository: `jemwrap.control` with `read_value`, `set_value`, `settable_snapshot`,
  `restore` and `take_frame`; a write path in the endpoint table; `MicroscopeMonitor.poll_now`.
  Three choices are the agent's and wait for the principal: `[assumed]`
  - plain functions that take the monitor, in place of a control object;
  - the narrowest settable list (M1's lenses, SpotA, CLA1, CLA2, the shift and tilt balances,
    the selector, MAG or DIFF, the spot size, alpha);
  - every lens through free lens control, CL3 too, although the table has `SetCL3`.
  Every settable key ran on the simulated NEOARM and on the offline PyJEM. No setter has
  run on a real microscope.
- J12 `[answered]` Every change to jemwrap conforms logically to the style that its author
  set. `[record]` That style is: a declarative table of endpoints, typed dataclasses for
  the state, one small module for each responsibility, a test for each behaviour in
  `tests/`, its own `requirements.md` kept up to date as the description of the package,
  and `flake8` with a line length of 110.
- J10 `[record]` jemwrap is a separate git repository with its own remote, and it holds a
  bundled copy of PyJEM. It is not part of the public TemGymCore repository and must not
  be committed into it.
- J4 `[answered]` A script's calls and jemwrap's polling share one rate limiter (J9, item 1).
  This resolves R14 in one place.
- J5 `[answered]` A script reads the values it needs at once with the call of J9, item 3.
- J6 `[record]` jemwrap runs on TemGym's simulated microscope (2026-11-19, stage 3). The
  simulator gained a `sightx` module, `GetAlphaSelectorEx` and `GetScreen`, and now returns
  PyJEM's types for `GetCurrentMagSelectorID` and the camera settings. Every value this project
  uses reads without an error. The gun, vacuum, stage, STEM detectors and EDS stay unemulated,
  and jemwrap shows errors for them. A frame through jemwrap at the recorded MAG 40k state has
  an rms beam size of 58 px. The real NEOARM gave 60 px.
- J11 `[answered]` The width of the SightX stays at 12 cm (R28). jemwrap's notes give
  10636 pixels per metre for the camera, which gives 9 cm or 18 cm depending on the
  binning. The principal decided on 2026-11-19 not to use that value.

### The model

- M1 `[answered]` **Components that change**, set from the console: the lenses CL1, CL3,
  IL1, IL2, IL3 and PL1, and the deflectors CLA1 and CLA2. Spot alignment (SpotA) is
  one more (C6). This split is for now, to keep
  things simple.
- M2 `[answered]` **Components that are static**: CM, the objective pre-field, the
  objective post-field and the corrector. They do not change much. The fit never changes
  them.
- M3 `[answered]` A static component is a box that we do not know and do not change. It
  holds one non-dimensional value. In a non-dimensional microscope the choice of that value
  should not matter: another value would still fit the same microscope. The principal
  wants a clear representation of this in TemGym and care in how it is done.
- M9 `[answered]` A static component is an ordinary TemGym component, in a list that is
  named as static, with one stated value. This holds for the model notebook and for the
  fitted model. The form must be extendable: a static component can later become a
  component that changes or that the fit adjusts, without a rewrite of the model. One
  `ABCDTransfer` box for each static section is rejected.
- M4 `[record]` M3 holds within a range. In the condenser fit, any CM field length from
  2 mm to 10 mm fitted the data equally well, and 30 mm or more did not. Source: session of
  2026-11-18.
- M5 `[answered]` The model is non-dimensional. All lengths are in units of the column
  length L. Lens positions in metres cannot be found for now, unless a special fitting
  routine is made. The goal is a non-dimensional microscope that predicts.
- M6 `[answered]` The 12 cm width of the SightX (R28) is the only real length the model
  uses. It converts pixels to metres at the camera.
- M7 `[answered]` The first scale on the sample plane is the console's nominal
  magnification with the camera width. A grating or diffraction rings is a later check.
- M10 `[answered]` The lenses are TemGym `PolePieceLens` components: Lenz's field of two
  pole pieces, set by a gap and a bore radius, with the excitation from the DAC value. The
  principal chose this law for the projector lenses on 2026-11-19 (the condenser lenses use it
  too, `[assumed]`), and rejected a fitted strength factor
  exp(a x + b x^2) as not physical.
  `[record]` Evidence of 2026-11-19 (session scratch fits, then
  `examples/neoarm/fit_projector_lenses.py`):
  - The projector lenses turn the image by up to 2 rad, so a thin lens is not a good model.
  - On the lens changes recorded on 2026-11-10 at MAG 40k and DIFF 40 cm, the misfit was 88
    for the thin lens, 26 for Glaser's bell field, 6.7 for a generic 3-number law and 2.7
    for Lenz's field. These scratch fits used the sizes of the 16 ratios only, with an
    allowed error of 1 %. `fit_projector_lenses.py` also fits the turn of each ratio, with
    an allowed error of 0.5 %, so its misfit is larger and not comparable.
  - A Lenz fit to the MAG changes alone predicts the DIFF changes best (median miss 37 % of
    the effect, against 80 % to 99 % for the others). The recording has too few numbers to
    confirm the law.
  - Fits to every row of the console table, with each label held to a few per cent and each
    row in focus, failed for every law. This held for a thicker objective, for free lens
    positions and for free label factors.
  - The Lenz fit to the lens changes alone, never fitted to the table, kept every MAG row in
    focus. Its magnifications and camera lengths were a near-constant multiple of the labels
    (about 1.75 in MAG and 0.54 in DIFF).
  - With the labels setting only the mean MAG and mean DIFF scale, the fit gives MAG within
    8 % of the labels from 20k to 500k and every DIFF camera length within 4 %. The lens
    changes still fit to 0.5 %. MAG 8k to 15k comes out 7 to 16 % low. The objective focal
    length comes out at 2.1 mm with the drawing's L. The manual gives 1.9 mm.
- M11 `[answered]` The console table enters the fit in two ways only. Every MAG row keeps
  the image in focus, and the labels set one mean scale for MAG and one for DIFF. Recordings
  of lens and deflector changes give the rest. The principal called this result very good on
  2026-11-19 and asked to continue with it.
- M12 `[record]` One factor along the column stays free. Every axial length can stretch by
  one factor with no change to the lens changes, the MAG magnifications or the MAG focus. The
  DIFF labels, in metres, set that factor through the drawing's L. The manual's objective
  focal length (1.9 mm) and the fitted 2.1 mm agree to about 10 %. A known diffraction ring
  or a grating sets it independently (U2).
- M8 `[answered]` Beyond M1, the first controls added from the alignment guide are the
  condenser pivot points (tilt balance and shift balance). The gun shift and tilt, the
  stigmators, the corrector beam tilt and the aperture are not in the first version.

### The condenser and spot alignment

- C1 `[answered]` The model does not have to reproduce the real CL3 data exactly. The
  model must have the degrees of freedom to capture the real microscope, and its condenser
  must follow the rough behaviour of the real microscope.
- C2 `[assumed]` "Rough behaviour" means three things. The beam goes through a focus as
  CL3 changes. The CL3 value of that focus moves with the spot size in the same direction
  as on the real microscope. The beam size changes with CL3 at about the measured rate.
  Example E13 gives the numbers.
- C3 `[record]` The condenser that the rehearsal runs today is a designed one. It does not
  follow the real CL3 data. The condenser that does follow the data (focus curve to 3 %)
  cannot run in the rehearsal yet, because the hidden microscope's lens offsets are too
  large for it (U3). So C1 is not met today.
- C4 `[answered]` Spot alignment (SpotA) is a concern. Its value moves in the console data,
  and the model must deal with it.
- C5 `[record]` In the console data of 2026-11-17, SpotA changes with the spot size (about
  570 units in x and 420 in y from spot 0 to spot 4 at alpha 0) and with alpha. It does
  not change across the MAG rows at one spot. The condenser stigmator (CLs) also changes
  with the spot size. The present notebooks do not have SpotA.
- C6 `[answered]` SpotA is one more component that changes (M1): a TemGym `Deflector` in the
  condenser, with its value read from the console for each frame. The recording kicks
  SpotA like CLA1, so that the fit finds how far a SpotA unit moves the beam. The model
  then follows the beam when the spot size changes.
- C7 `[answered]` Another condenser deflector, at the bottom of the condenser, is sometimes
  used and cannot be controlled. It is ignored for now.
- C8 `[record]` On 2026-11-19 `examples/neoarm/fit_condenser.py` fitted CL1, CL3 and CM as
  `PolePieceLens` components (M10) to the focus curve and the parallel settings at alpha 1.
  - The beam size follows the focus curve within 2 % from MAG 30k to 500k. At MAG 40k the
    model gives 60.8 px against the measured 59.8 px.
  - E13 passes. At alpha 1 the model's focus is within 61 to 524 CL3 units of the console's
    value at every spot, and it falls from spot 0 to spot 4 as the console's value does.
  - The fit does not reach the operator's parallel settings. The model's parallel beam is
    600 to 3000 CL3 units away, more at the larger spot sizes. A free source position or a
    factor on the condenser coils does not remove this. The cause is not known.
  - The fit cannot find the source size, which runs to zero. Only the spread of angles
    matters for these data.
  - No rotation measurement gives the ampere-turns per DAC unit of the condenser. It uses
    the projector's mean value, `[assumed]`.

### Hysteresis

- H1 `[answered]` Hysteresis of the lenses is a known problem. The principal wants a good
  way to include it: either the model fits it, or the work finds a way to deal with the
  errors that it gives.
- H2 `[record]` The simulated microscope has a hysteresis model for each lens
  (`LensHysteresis` in `pyjem_simulator`). In the rehearsal, the lag of CL1 and CL3 moved
  the beam by up to 7 px at MAG 500k near the condenser focus. Source: session of
  2026-11-18.
- H3 `[record]` Scripts already set a lens from 2000 units below its value, so that the
  lens has the same history each time (`.claude/skills/pyjem/SKILL.md`).
- H4 `[record]` `record_predictor.py` has a stage that reaches the same lens values from
  below and from above. No such recording from the real microscope has been analysed.
- H6 `[answered]` The first version avoids hysteresis (H3) and measures it on the real
  microscope (H5). The model fits hysteresis only if the measured error is larger than the
  10 px budget (R20).
- H5 `[unknown]` How large the hysteresis error is on the real NEOARM, for each lens, in
  pixels. Resolved by a recording that reaches the same values by different routes.

### 2.4 Operating conditions

- R18 `[record]` Recording scripts run on the microscope PC with numpy, PyJEM and jemwrap
  (which needs `rich`).
  Fitting and rehearsal run offline with TemGym and JAX.

### 2.5 Availability and reliability

- R19 `[answered]` A wrong prediction is acceptable as long as the beam generally stays on
  the screen and a small correction can bring it back (S3).

### 2.6 Performance

- R20 `[answered]` Beam centre on the SightX: within 10 px. 5 px is too tight for now.
- R21 `[answered]` Beam size on the SightX: within 20 %.
- R22 `[answered]` Range: MAG 8k to 500k and the DIFF camera lengths. Source: plan of
  2026-11-17.
  `[record]` On 2026-11-19 no model followed the console table in MAG and DIFF together
  (M10). Whether one model meets R22 in DIFF as well as in MAG is open until the recording of
  stage 5.
- R29 `[answered]` A calibration recording can take any length of time that is not hours
  of waiting.

### 2.7 Testing, installation and service

- R23 `[record]` The fit is tested on rows it never saw (kept-back rows), a kept-back spot
  size, and moves of several controls at once.

### 2.8 Portability and distribution

- R24 `[answered]` The scripts folder is copied to the microscope PC by hand.

### Deliverables

- D1 `[answered]` **Notebook 1: the model.** A working TemGym model that is the best
  representation of the microscope that we can make.
- D2 `[answered]` **Notebook 2: the fit.** It shows how to fit a non-dimensional model,
  with all the right moving parts, to an unknown microscope so that the model captures the
  behaviour of the microscope. The fit uses PyJEM calls that perturb the lenses and
  deflectors in a planned way.
- D3 `[answered]` **One fitted model** that predicts the beam.
- D4 `[answered]` **Alignment scripts.** A set of PyJEM scripts that demonstrate the
  alignments of the guide on the model microscope, based on the fitted model. It must be
  very clear what each one is.
- D6 `[answered]` Each alignment is a PyJEM script. A notebook calls the script. The
  notebook is a window into what the microscope does during the alignment. It shows three
  things together, live or near live while the script runs: a cross-section view of the
  column with the beam, the beam on the detector, and the script that is called.
- D7 `[answered]` The live view does not add code to the script. The script stays plain
  jemwrap calls (J7). The simulated microscope in TemGym redraws the view when the script changes the
  microscope or takes a frame. A small addition to `pyjem_simulator` for this is approved.
  A small delay in the view is acceptable.
- D8 `[unknown]` Which plotting library draws the live view (matplotlib, Plotly or another)
  is not decided. The agent compares what exists and proposes one.

### 2.9 Jurisdictions

Not asked, because the work is research code in a public repository with no personal data.

### 2.10 Lifetime and decommissioning

Not asked yet.

### 2.11 Safety and risks

- R25 `[answered]` A script leaves the microscope as it found it, also after an error or
  Ctrl-C. A script does not move the stage, change the high voltage, change an aperture or
  insert a detector unless the task asks for that (`.claude/skills/pyjem/SKILL.md`).
- R26 `[answered]` The principal handles who may run scripts on the instrument. The agent
  does not need to record an owner.

### 2.12 Applicable standards

- R27 `[record]` Code, comments and notebooks follow the three project skills in
  `.claude/skills/` (readable-python, temgym, pyjem).

## Release

Decides: the principal. Builds: the agent, reviewed by the principal. Signs off: the
principal. Owns the instrument: handled by the principal (R26).

## Open questions

None. Closed on 2026-11-19: sample-plane units (M7) and the model representation (M1 to M9).
