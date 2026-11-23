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
- S8 `[answered]` **Alignment by recipe.** An alignment routine for TEM that follows the
  principal's guide (`alignment_guide.md`), written as fixed recipes and tested on the NEOARM.
  The principal (2026-11-22): "can it all be done with a hardcoded recipe we test on our
  microscope?" `[assumed]` Each step is a closed loop on the SightX: wobble or change one value,
  find the beam (or a feature), adjust the step's control, repeat until the guide's endpoint. A
  recipe needs no fitted model; the model helps it choose step sizes and keep the beam on the
  SightX (S2, S3), and the simulated microscope rehearses it (S4). Which steps the jemwrap
  controls allow now, and which need new controls (rule 5 of the pyjem skill) or a specimen, is in
  the change record (2026-11-22). Examples: E20, E21 (aperture centring), E22 (one pass with the
  predictor). How S8 relates to S6 is Q15.
  `[answered]` The principal (2026-11-22, on Q15): "I think recipe is simpler, but I still need
  the predictor model too, I am not sure." Then: "we would need to run the predictor samples
  first, then run the align recipe? I think that is so - so two run side by side then." So S8
  sits beside S6: the recording for the predictor comes first, then the recipe aligns.

### 2.2 Demand uncertainties

- U1 `[unknown]` Whether a model fitted in the rehearsal reaches the same accuracy on the
  real NEOARM. Resolved by the first real recording.
- U2 `[unknown]` How absolute units on the sample plane are obtained. Frames from one
  camera cannot give them without a ruler (a grating image, diffraction rings or a second
  camera). Source: `neoarm_model.ipynb`, section 7.
- U5 `[record]` Whether the choice of a fixed value changes any prediction. With only the gauge
  freedoms fixed (M14), no: E9 moves each of the six fixed numbers by a plausible error, and the
  kept-back predictions move by 0.08 px at most (2026-11-20, M13). Before that, the fixed
  objective moved them by up to 17 px, and CL1's position, the other condenser positions and the
  corrector were not settled.
- U4 `[answered]` The model must be clear about which components are static and which
  change. The model must not depend on things that cannot be known at the real microscope.
  The principal wants a round of questions on how the TemGym model is represented before
  S5 is settled (Q6).
- U3 `[unknown]` The condenser model follows the measured focus curve to 3 % but the
  rehearsal cannot use it yet. Source: session of 2026-11-18.
- U6 `[unknown]` Whether the controls added to jemwrap on 2026-11-22 work on the real NEOARM:
  the getters of CLs, OLs, GunA1, GunA2 and the aperture position (every script's snapshot now
  reads them, J16), `Apt3.SetPosition` moving the condenser aperture, and `GUN3.SetHtWobbler`
  (marked "Ver.1.2" in the bundled PyJEM). None has run there. Resolved by a first session that
  reads and sets each once, with the console watching.
- U7 `[unknown]` How far the predictor holds outside the recorded procedure. Two limits are
  known (2026-11-22, simulated): the model assumes every lens came to its value from below (with
  an unknown history its beam was 26 px off; with every lens brought from below, 0.3 px); and it
  does not follow a change of ShiftBal or TiltBal, so after the pivot-point scripts its deflector
  response no longer holds. Resolved by Q22 and by the real session's repeatability checks.
- U8 `[unknown]` Notebook 2's model has no condenser aperture. With the large hole the simulated
  beam does not touch it (2026-11-22); with an aperture that limits the beam, as for aperture
  centring, the model's predictions of beam size and centre are not checked. Resolved by a
  recording with a limiting aperture and, if it misses, an aperture in the model.

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
  Every settable key ran on the simulated NEOARM and on the offline PyJEM. No setter has
  run on a real microscope.
- J14 `[answered]` The principal accepted on 2026-11-19:
  - plain functions that take the monitor, in place of a control object;
  - the narrowest settable list (M1's lenses, SpotA, CLA1, CLA2, the shift and tilt balances,
    the selector, MAG or DIFF, the spot size, alpha). CM and the objective go on the list
    when a recording needs them, each with its reason (the pyjem skill, safety rule 5).
    On 2026-11-20 the principal added the objective's fine focus OLf, for the focus series of
    M15; the coarse focus OLc stays off the list;
  - every lens through free lens control, CL3 too, although the table has `SetCL3`, so that
    the restore of every lens is its switch;
  - the selector must be a row of its function mode: 31 MAG rows and 30 DIFF rows, from the
    console tables of 2026-11-17 (`SELECTOR_ROWS` in `jemwrap.control`);
  - the `control` branch goes to jemwrap's author as a pull request if the author reviews on
    GitHub, otherwise as a patch file. The agent does not push it (J10).
- J15 `[answered]` The fits read small tables committed in `examples/neoarm/recordings/`,
  copied unchanged out of the recording zips (2026-11-19). The zips stay outside git as the
  raw source. A fresh checkout reruns `fit_projector_lenses.py` and `fit_condenser.py`, and
  both give the same JSON as from the zips.
- J16 `[answered]` Added at the principal's request on 2026-11-22, for the steps of the guide:
  the condenser and objective stigmators (CLs, OLs), the gun shift and tilt (GunA1, GunA2) and the
  position of the aperture selected on the console, each "full range, no window" (the
  principal's choice of limits), and the HT wobbler switch, which PyJEM can set but not read
  (`WRITE_ONLY`: restore switches it off first, only if the script switched it on). Stage Z
  `[deferred]`: "there is no stage z getter in pyjem, but there is a setter - we can deal with
  that later" (Q21). `[record]` A restore moves the aperture back only while the aperture the
  script moved is still selected (review, 2026-11-22).
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
- M2 `[answered]` **Components that are static**: CM and the corrector. No console value of the
  recorded settings changes them. Their positions and strengths are still numbers of the model,
  and the fit adjusts every one that is not a gauge choice (M14). The objective's pre-field and
  post-field are not set by the projector's console values. The fit adjusts their focal length and
  their distance from the specimen, as M9 allows, and their focus follows OLf (M15). The principal
  chose to fit the objective on 2026-11-20 after E9 failed with a fixed objective (M13). Moving CM
  instead was rejected: in practice CM seldom changes.
  - `[answered]` (Q13, 2026-11-20) The corrector is turned off on the NEOARM. There are no lenses
    there, only a deflector that is sometimes used (the corrector beam tilt of the alignment guide,
    M8). The principal: "There are no hidden lenses." The unequal corrector lenses of the harder
    test of 2026-11-20 are not a form the real column can have.
  - `[answered]` The corrector is still a transfer block: a long region between CLA2 and CM. The
    model draws it as a rectangle about the corrector's length in the column diagram, as a
    component. The principal is not sure that the corrector is off, so the model has two
    versions of the block: free space (corrector off), and a transfer with a magnification of -1
    and nothing else (corrector on). The fit runs with each, and the misfit shows which one the
    microscope has.
- M15 `[answered]` The objective's focus is a component that changes. On 2026-11-20 the
  principal saw that the model holds only for the objective at the standard focus of the console
  rows, in normal work at eucentric height, and chose to model a focus change. The objective's
  excitation follows its fine focus OLf, with a fitted change per DAC unit; its focal length goes
  as the inverse square of its excitation. The recording script records a focus series (OLf
  -3000, -1000, +1000, +3000) at X15k, X40k, X100k, 20 cm, 40 cm and 1 m. The coarse focus OLc
  and a refocus outside that range are not modelled. Changing the specimen's height alone changes
  no lens and does not affect the predictions.
- M14 `[answered]` The fitted microscope will always be wrong in its numbers. It must be a
  physics-constrained predictor whose predictions do not depend on the values chosen for the
  numbers held fixed, as long as the parts they describe do not change. The principal set this on
  2026-11-20, and pointed out the gauge freedom: only the relations between the numbers can
  matter. So the fit holds fixed exactly one number for each gauge freedom (a change of the
  numbers that no recording can see) and fits every other number.
  - The gauge freedoms come from the derivatives of everything the recording measures with
    respect to all the numbers: the combinations with a zero singular value. In notebook 2 there
    are five: a turn of the whole column about its axis, a move of CL1 to CLA2 together along the
    axis (the corrector images them one to one), and three trade-offs among CM's gap and bore and
    the corrector's two positions. The fit holds the camera's turn, CL1's position, CM's gap and
    bore and the corrector's two positions; no zero is left. These six numbers hold one combination
    more than the five freedoms: a change among CM and the corrector that the recording sees only
    at about a millionth of its strongest change. Notebook 2 computes these singular values and
    asserts the count; E9 shows that the sixth choice does not move a prediction either. The review
    of 2026-11-20 found this.
  - Notebook 2 checks every fixed number after the fit, first to first order from the fit's
    derivatives, then in full (E9). The plausible errors (20 mrad, 5 mm, 10 %) and the 1 px of E9
    are `[assumed]`.
- M16 `[answered]` (Q7) The goal is that the model's form does not differ from the NEOARM's. The
  principal, 2026-11-20: "to have the generalised model with enough gauge freedoms where we will
  effectively have the recipe to one shot the scale invariant predictor microscope of the
  neoarm". So the model is general enough that the form of the real column is absorbed by
  fitted numbers, and one recording and one fit at the microscope give the predictor. Freedoms
  that no recording can see are gauge freedoms and are held as M14 says. `[assumed]` A misfit
  far above the scatter of the frames means the form is still wrong; the answer is to change
  the model, not the pass rules. Source of the question: the harder test of 2026-11-20, whose
  hidden microscope has lens hysteresis, unequal corrector lenses and a shorter objective field;
  its fit misses the beam centres by 6 px rms against a scatter of 1 px.
  - `[answered]` (Q12) "One shot": one recording session and one fit at the real NEOARM, with
    no return visit to fix the model. The recovery routine of S3 stays allowed. The physics of
    the form is understood already (the principal: "we have the physics already understood -
    hysteresis etc."), so the hidden microscope of a rehearsal differs from the model only in
    physics the real column has, not in invented components.
  - `[answered]` (Q14) A freedom goes into the model when the model cannot otherwise follow a
    part of the frames, and only if its physics is real for a column. After each addition the
    gauge count (M14) is computed again.
- M17 `[answered]` (Q8) The condenser is measured for better spot sizes on the SightX now
  (R21; E2 and E14 fail in the harder test), "creeping toward a better representation" of it.
  A prediction at the specimen (probe size, convergence angle) is not required yet; it is a
  direction, not a target.
- M18 `[answered]` Specimen images come from the Gaussian beam model, for the guide's steps that
  need a specimen (HT wobbler, objective stigmator, image focus). The principal (2026-11-22): "I
  do also need the gaussian beam model back anyway (made exceptionally fast to generate images on
  a small downsample detector - 256 x 256)"; "We can use gold particles or the cross grating
  (modelled after real images) as the sample"; "the gaussian beam model will play nicely with the
  ray object already... it's an extended ray object"; plain wave optics "needs padding, does not
  go through the optics automatically - this gaussian model will without too much bother -
  provided we have sampled enough"; "keep on sampling gaussians until convergence"; a thin phase
  object is "a harder problem... but still". Local refinement of the sampling was tried and set
  aside by the principal (2026-11-22).
- M19 `[record]` Where the Gaussian beam model stands (2026-11-22, change record): the sparse
  Pallas renderer is back, 256 x 256 in about 2 ms; `converged_gaussian_image` refines the
  beamlets until the image changes by less than a tolerance; against exact images a cross
  grating at focus is 0.36 % off at 261 121 beamlets, and a 10 nm aperture at 10 um defocus
  0.056 % at 65 025; defocus matches exact Fresnel diffraction. Since Q18 the simulated NEOARM
  images a cross grating with one coherent Gaussian through its whole column, 0.08 s a frame
  (PolePieceLens now carries Gaussian beams). Not yet: partial coherence, a thin phase object,
  gold particles.
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
- M13 `[record]` The fit of 2026-11-20 (`examples/neoarm/neoarm_fit.ipynb`) starts from notebook 1
  and fits the recording of the simulated NEOARM. It never reads the hidden numbers.
  - The fitted microscope will always be wrong in its numbers. It must predict, and its
    predictions must not depend on the values chosen for the parts that do not change (M14). A
    fit that returns the hidden numbers is not the aim.
  - The hidden microscope also differs in CM's gap (10 %), which the fit holds fixed as a gauge
    choice, and in CLA1's position (3 mm), which it now fits. Its SightX is off the axis and
    turned, which the fit's camera shift and its fitted deflector directions take up.
  - The objective sits only 0.03 mm further from the specimen than its focal length. 1 % of the
    focal length is 75 % of that distance. A hidden objective 8 % shorter made a column that the
    recording script could not follow. Hidden differences are now chosen with a trace without
    frames first.
  - With the objective held fixed: a 10 % longer objective moved kept-back predictions by up to
    3.7 px (E9 failed), and a hidden objective at another distance (2.20 mm against 2.10 mm, focal
    length 2.15 mm against 2.07 mm) moved them by up to 17 px (E14 failed).
  - The objective's focal length, distance and focus law are now fitted (M2, M15). From notebook
    1's objective the fit gets lost. The notebook starts the focal length at 0.5 to 3 times
    notebook 1's gap to the specimen, with both mirror choices, ten fits at once on the GPU. The
    starts further from it end at the same fit (misfit 138) and predict the same. This is a check
    of the fit's start, not E9.
  - The fit ends at a distance of 1.84 mm, a focal length of 1.81 mm and 3.2e-7 of excitation per
    OLf unit, not the hidden 2.20 mm, 2.15 mm and 4e-7. It still predicts every kept-back frame
    within 1.2 px. The recording fixes what the objective does, not these numbers one by one.
  - Kept-back rows (MAG and DIFF), settings with several lenses moved and spot 0 are predicted
    within 0.1 px, the beam sizes within 0.3 % (median). The focus series of X100k and 1 m, which
    the fit never saw, is predicted within 0.5 px and 0.1 px rms (E15; 2 of its 48 frames lost the
    beam). These misses show that the fit recovers a column of this kind. They do not show the
    accuracy at the real NEOARM (U1).
  - The hidden microscope has the model's form: the same lens laws and focus law, and many of
    notebook 1's numbers. It differs in its source size and in numbers the fit can follow. So the
    test shows that a model of the right form predicts, not how a wrong form fails (review of
    2026-11-20).
  - With twelve numbers fixed, E9 failed or was not settled for seven of them. The fit now holds
    only the six gauge-fixing numbers (M14) and fits the rest, 63 numbers. E9 moves each of the
    six in twenty steps with a refit at each: every refit reaches the best misfit (138), and the
    kept-back predictions move by 0.08 px at most. E9 passes. Moved by 5 mm, CL1 needs these small
    steps: half a millimetre at once leaves a misfit of 270 000, and the fit follows by moving
    CL3, SpotA, CLA1 and CLA2 and lengthening the corrector's focal length by 2 %.
  - The fit needs the camera's mirror image. The other choice ends with a misfit of 20 000 against
    138.
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
- R32 `[answered]` A focused DIFF spot on the SightX, with no specimen and CL3 at the console
  value, is safe for the camera (the principal, 2026-11-19). The settle waits of
  `record_lens_changes.py` are accepted: 2 s after a change of mode, 1 s after a row, 0.5 s after
  a lens and 0.2 s after a deflector.
- R29 `[answered]` A calibration recording can take any length of time that is not hours
  of waiting.

- R33 `[answered]` (Q10) One simulated round takes under 30 minutes: the simulated recording
  within 15 minutes, and the fit with all its checks within 10 minutes on the GPU. On
  2026-11-20 a round took more than two hours.

### 2.7 Testing, installation and service

- R23 `[record]` The fit is tested on rows it never saw (kept-back rows), a kept-back spot
  size, and moves of several controls at once.

- R34 `[answered]` (Q9) A fit is finished when its misfit falls by less than 0.1 % over 50 steps,
  not after a fixed number of steps. A fit that reaches its step limit first says so, and its
  predictions are not reported as final. Source: in the harder test of 2026-11-20 the refits of
  E9 lowered the best fit's misfit and moved its predictions by 12 px.
- R35 `[answered]` (Q11) Every group of settings in a recording earns its place, shown with the
  fit's derivatives: for each group, the notebook shows how much the uncertainty of each kept-back
  prediction grows when the group is left out. A group that changes no prediction's uncertainty by
  more than 10 % is a candidate to drop. A new measurement (a condenser focus series, a pivot
  point) goes in only if it lowers the uncertainty of a spot-size prediction by at least 20 %.

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

Closed on 2026-11-19: sample-plane units (M7) and the model representation (M1 to M9).

Opened on 2026-11-20 by the harder test (the hidden microscope differs from the model in form)
and its fit; in interview with the principal:

Q7 to Q14 answered on 2026-11-20: M16, M17, M2 (corrector off), R33, R34, R35.

Opened on 2026-11-22:

- Q15 `[answered]` 2026-11-22: beside, not instead (S8). The predictor's recording runs first,
  then the recipe. How the predictor then follows the alignment is Q22.
- Q16 `[answered]` 2026-11-22: diff focus is the next recipe after the pivot points ("Q16 yes").
  The rest stays open below. Original question: which steps of the guide are recipes in the first
  version beyond the pivot
  points: diff focus (controls allowed now); condenser stigmator, gun tilt and shift, aperture
  centring (each needs a control on jemwrap's list, so David decides); HT wobbler, objective
  stigmator and image focus (need a specimen and controls outside the list). For the principal,
  resolves S8 (and Q2).
- Q17 `[answered]` 2026-11-22: "Yes please". Aperture centring: the simulated NEOARM has no condenser aperture in its
  column, so `Apt3.SetPosition` changes nothing there. Put one in (its height, hole, metres per
  DAC unit and a hidden off-centre position are choices with no source), then a recipe through the
  crossover? For the principal, resolves S8.
- Q18 `[answered]` 2026-11-22: "Yes" (the cross grating first, then gold particles, as proposed);
  one coherent beam first (the principal). Built 2026-11-22 for the cross grating.
  Specimen frames in the simulated NEOARM: render its SightX frames with the
  Gaussian beam model through the whole column with a specimen, and which specimen first (the
  cross grating, gold particles)? For the principal, resolves M18, unblocks Q19.
- Q19 `[answered]` 2026-11-22: the principal: "Ok review, Q 19, then Q 23". Built: every lens of
  the simulated NEOARM follows the high tension, the HT wobbler moves it by HT_WOBBLER_V each way
  between frames, and `ht_centre.py` sets CLA2 where the specimen's features swing least (E24).
  Open: the real wobbler's amplitude and frequency are not known; 100 V is a choice. Original
  question: the HT wobbler: let every lens depend on the high tension and wobble it between
  frames, then a recipe that finds the shift of the specimen's features (cross-correlation) and
  sets CLA2?
- Q20 `[answered]` 2026-11-22: "Q20 yes": each script names the values it changes, and only
  those are snapshotted and restored. Built 2026-11-22 (`settable_snapshot(monitor, keys)`). Original question: should a script snapshot and restore
  only the values it changes? Today every
  snapshot reads all settable values, so a getter that fails on the NEOARM stops every script
  (U6), and a restore undoes what an alignment set unless the script leaves it out. For the
  principal, resolves J16 and R25.
- Q23 `[answered]` 2026-11-22: yes, built after the HT wobbler (Q19). The illumination at the
  specimen from the predictor (the principal,
  2026-11-22: "cant I get size from the specimen plane if the lens system below predicts mag....
  so in a way we can predict that? We just view it through the projector."): the fitted model
  traces to the specimen plane and gives, for any spot size and CL3, the illuminated patch's rms
  size (seen in MAG, in focus, through the projector) and the convergence angle (seen in DIFF,
  through the camera length), in nanometres once a ruler fixes the scale (U2: the cross grating's
  463 nm period). `[record]` The principal on checking it: the microscope does not give the
  true illumination; only the simulated NEOARM's hidden column can, in rehearsal. Example: E23.
  Resolves M17's direction.
- Q24 `[answered]` 2026-11-22: "ignore aberrations for now. yes to everything else": the rings
  first (the coherent wave from just above the condenser aperture), in the simulator only; the
  predictor stays paraxial and without an aperture until a real recording asks for more.
  Aberrations and rings from the condenser aperture (the principal,
  2026-11-22: "what about getting aberrations and defocus rings from the condenser aperture?"):
  start the coherent wave just above the condenser aperture instead of at the specimen, so that
  defocused and DIFF images show the aperture edge's Fresnel rings; and give the hidden column
  condenser (and objective pre-field) aberrations with TemGym's Seidel terms, whose coefficients
  are choices with no NEOARM source. The fitted model is paraxial and has no aperture (U8). For
  the principal: which first, the rings or the aberrations; and does the predictor need to fit
  them, or only the simulator show them? (Answered above.)
- Q21 `[deferred]` Stage Z: settable as a value of `WRITE_ONLY`, and within which limits? Safety
  limit: the principal or the instrument's owner decides. Deferred by the principal to later.
- Q22 `[answered]` 2026-11-22. The principal: "so here I guess its iterative so... I know that
  is something we have to worry about." Then "Q22a yes, Q22b yes": the loop is record for the
  predictor and fit, run the recipe, re-record only what the alignment changed and refit those
  parts, until neither changes (22a); the first version does one pass (record, align, short
  refit), and the full loop waits until the real NEOARM's drift is seen (22b). Original question:
  how the predictor follows an alignment: refit the parts it changed from a short
  recording after the alignment, model the balances, or neither for now? And should scripts that
  predict bring the lenses from below first (U7)? For the principal, with Q15.

Status of 2026-11-22, evening (built without further questions, at the principal's request:
"run to satisfy the requirements as best you can ... make sure that what we have has a good
chance of working on the neoarm first time"):

- Q13 (corrector) `[answered]` 2026-11-22: "its a probe corrector, but in TEM mode the software
  shows that all currents are zero - even to the transfer lens. Except one deflector." The
  simulated NEOARM's corrector is free space. The fit tries off, -1 and +1 (the last two by
  starts with the deflectors' kicks reversed); a +1 test recording fitted as well as -1 (misfit
  138 against 139).
- PLA `[answered]` 2026-11-22: "Yes do add PLA for sure". On jemwrap's settable list; DIFF
  patterns are centred with PLA in shift_balance.py, record_lens_changes.py and centre_beam.py,
  and the fitted model has PLA.
- Near-aligned start `[answered]` 2026-11-22: the simulated NEOARM starts from an alignment that
  is nearly right, lenses from below. Its hidden condenser matches the console's whole spot table
  (focus and SpotA) with the corrector off.
- Q19 status: the condenser alignment tilt (HT wobbler) stops where CLA2 lowers the swing by less
  than 10 % and reports the rest (E24 reworded, the principal: "Maybe 6 pixels is an ok error").
  On the simulated NEOARM about 6 px stay, from CL1 and CL3 off the axis; the gun and spot
  alignments, which have no script, would remove them.
- Q23 status (superseded below): the grating ruler works (1.2 % on the simulator); the gold
  ruler does not work on
  the final simulator (no rings at the smallest DIFF spot). The predictor gives sizes in nm and
  angles in px of DIFF row 10 until it does. Open, for the next session.
- Q24 `[deferred]`: the aperture rings are not built.
- New, for the principal: the beam-shift range of the NEOARM. No source was found; the simulator
  assumes +-50 um over half the range, which agrees with the real recording of 2026-11-10 (one unit
  of CLA1 about 1.25 nm at the specimen; the simulator 1.5 nm).
- E9 status (2026-11-23, superseded below): fails on the final simulator. The refits with CL1
  moved 5 mm and the
  objective's focal length moved 20 % end at a misfit of about 23 900 against the best 457; the
  linear estimate moves the predictions by 0.03 and 0.58 px. Open, for the next session.

Status of 2026-11-23 (the principal's review):

- The model with positions `[answered]` 2026-11-23. The principal asked whether a predictive
  model that is "very sensitive to values I can't know" forces a framework without positions,
  which could not be extended. Agreed ("Yes, agree with 1 to 5 - fix E9 that way"):
  1. Keep the physical model with positions. The held numbers stay unknowable; the requirement
     is that the predictions do not depend on them.
  2. E9 refits by continuation: each held number moves in 20 steps, and before each refit the
     fitted numbers take the first-order change that follows the step (a predictor-corrector
     continuation). The old refits in ten plain steps got lost in the misfit's curved valley; a
     test of 2026-11-23 with continuation: CL1 at +-5 mm moves the predictions by 0.03 px
     (misfit 445 and 469 against 457), the objective's focal length at +-20 % by 0.70 and
     0.69 px (misfit 531 and 427).
  3. A second check outside the recorded range: example E25.
  4. For each extension, E9 and E25 run again. A held number that then matters is pinned by a
     measurement that sees it, not by a change of framework.
  5. A model without positions (the A and B elements as functions of the DAC values) and the
     manifold boundary approximation method (Transtrum and Sethna) are analysis tools only, to
     show which combinations of numbers the data fix.

- Results of 2026-11-23 (neoarm_fit.ipynb re-executed): E1, E2, E3, E14, E15 pass as before. E9
  passes: the refits by continuation move the kept-back predictions by at most 0.70 px (the
  objective's focal length, misfit 531 against 457; CL1 0.03 px, misfit 445). E25 fails: at the
  settings outside the recorded range, the objective's focal length moves a prediction by up to
  28 px and CL1's position by 1.3 px, both most with OLf 9000 units off (MAG row 23). By
  decision 4, the remedy is a measurement that sees the focal length, such as a focus series
  over a wider OLf range; for the principal.
- Q23 status, 2026-11-23: the gold ruler works on the simulator. Its fault was the centre: in
  DIFF the direct beam and many diffracted spots saturate the SightX, and the brightest pixel was
  often a diffracted spot. record_rulers.py now centres on the largest bright patch and picks CL3
  where the three rings match best (the smallest spot by area failed: above the console's CL3,
  the grains' shadow image has patches as small). Rehearsal: CL3 30693, the 111 ring 46.00 px
  against 46.62 (1.3 %); 0.2315 mrad per px at DIFF row 10. The blotchy rings are physical for a
  narrow beam: each lit grain diffracts into one spot.
- Wide focus series `[answered]` 2026-11-23 (the principal: "Yes, add it"). record_lens_changes.py
  moves OLf by +-6000 and +-9000 units at DIFF rows 4, 7 and 11 (in MAG the beam left the SightX
  or grew too large to centre from 3000 to 6000 units). Recording:
  `recordings/lens_changes_simulated_wide_focus_2026-11-23.csv`, 1729 frames; its other frames
  agree with the corrector-off recording to the frame noise (median 0.02 px). Refit: misfit 547;
  E1, E2, E3, E14 pass as before; E15 1.1 px rms at X100k (was 5.3) and 0.6 px at 1 m out to
  9000 units. E9 passes, the focal length now 0.19 px (was 0.70). E25 still fails narrowly: CL1's
  position 2.2 px, the focal length 1.3 px (was 28), most in MAG with OLf 9000 units off, where no
  frame can follow the beam. Open, for the principal: whether E25 should count only settings
  whose beam a frame could measure.
