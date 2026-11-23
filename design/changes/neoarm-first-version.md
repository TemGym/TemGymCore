# Build plan: the first version of the NEOARM predictor

Date: 2026-11-20. Status: stages 1 to 6 are built.

- Stage 3 did not give the simulated NEOARM the hidden differences that stage 2b moved to it. Stage
  6 added them (`HIDDEN` in `simulated_neoarm.py`), with the operator's alignment of CLA1 and CLA2
  and SpotA from the console's spot table (a small change to `pyjem_simulator`).
- Stage 5, `record_lens_changes.py`, has run on the simulated NEOARM only: 1239 frames, with the
  focus series, in 6 minutes with no settle waits. The hidden microscope showed that the script
  must centre the beam at each row, size it from above the condenser focus, and stop when it loses
  the beam. On the real NEOARM it should take about 45 minutes.
- Stage 6, `neoarm_fit.ipynb`, fits the recording of the hidden microscope from the start of
  notebook 1. The hidden objective sits at another distance with another focal length, and its
  focus follows OLf (M15). The fit fits all three, from several starts at once on the GPU (M2,
  M13). E1, E2, E3, E14 and E15 pass with misses of 1.2 px at most. This shows that the fit
  recovers a column of this kind, not the accuracy at the real NEOARM. The fit holds only one
  number per gauge freedom and fits the rest (M14). E9 moves each fixed number and refits; the
  predictions move by 0.08 px at most, so E9 passes (M13, U5).
- jemwrap's settable list now holds OLf (J14), uncommitted in jemwrap's `control` branch.
- A fresh agent reviewed stages 5 and 6. Its findings are fixed, except E14's spot order, which
  the notebook explains. A second review, of the objective and focus work, found that "E9 passes"
  tested the objective's start, not a fixed value; E9 is now a refit with each fixed number moved.
  Its other findings are fixed: OLf leaves free lens control after the focus series, a lost beam
  stops the trial kicks, and the texts say what the code does.
- A third review, of the gauge rule, found that six numbers are held for five exact gauge
  freedoms; the sixth is a change among CM and the corrector that the recording barely sees. The
  notebook now computes the singular values and asserts the count, E9 and the first-order check
  assert, and the texts say that the hidden microscope has the model's form (M13, M14). Left as
  they are: E15 does not check that the model also loses the two lost frames, and the start grid
  of the objective was chosen after the first start failed. A lost beam in the y trial of SpotA
  no longer stops the recording (see below).
- The recording script looks for a beam that is off the SightX, as an operator would: it walks
  the centring deflector out on a spiral. A trial kick that loses the beam is tried the other way,
  a y trial then smaller, and a SpotA block whose trials fail is left out. On the simulated
  NEOARM, with the beam pushed 350 to 780 px off at MAG 40k, MAG 200k and DIFF 40 cm and a
  strength estimate 1.5 times wrong, the search found it within 2 to 58 frames.
- A harder test (2026-11-20, asked for by the principal after the review). The hidden microscope
  of the first test had the model's form. The second hidden microscope differs in form and moves
  the numbers that equalled the fit's start or its fixed values. Written down before any fit:
  - Form: CL1, CL3 and the projector lenses lag behind their DAC values (the simulator's
    `LensHysteresis`, up to 150 units, about 0.5 %); the model has no lag. The corrector's two
    lenses differ (focal lengths x1.03 and x0.98), where the model has one relay. The objective's
    post-field is 0.5 % shorter than its pre-field, where the model has one focal length.
  - Tried and dropped before any recording: projector lenses as uniform fields (`HardEdgeSolenoid`).
    With the console's tables, a uniform field of the same weak-lens strength put the MAG rows at
    half their labels, and one matched at MAG 40k at up to 200 times. A real microscope's labels
    are not that far off, so that microscope would not be a fair test. A 2 % shorter post-field
    moved the DIFF camera lengths by 20 %; 0.5 % moves them by 4 %.
  - Numbers: CL1 +0.004 L (2.9 mm), CL3 -0.005 L, CM +0.003 L, SpotA +0.004 L, CLA2 -0.003 L,
    corrector top +0.003 L and bottom -0.002 L, condenser coils x1.011, CM bore x0.95. The coils
    were first x1.02. That put the condenser focus about 500 units of CL3 below the console's
    value, against the console's own spot table, and the first recording lost the beam from MAG
    150k up. 1.1 % puts the focus back at the console's CL3 (MAG 40k), before any fit.
  - The pass rules stay: E1, E3, E14 and E15 within 10 px, E2 as before, E9 within 1 px and
    within 1 % of the best misfit. The hidden values are not changed after the fit is seen. If an
    example fails, the form differences are tried one at a time to find the one that breaks it.

This plan says what the first version contains, in which order the agent builds it, and
how each stage is checked. Requirement and example numbers refer to `../requirements.md`
and `../examples.md`.

- Stage 6b (2026-11-20 to 2026-11-22, done 2026-11-22; E18 over its 10 minutes, a note). The harder test failed by lens hysteresis alone: with the
  hysteresis of the hidden microscope removed, every check passed to 0.1 px; with only the
  objective difference removed, the misses stayed at 7 px. Built so far:
  - `ABCDTransfer` in TemGym takes a `length` and is drawn as a box (approved by the principal).
    Notebook 1's corrector is that box with a magnification of -1; its results did not change.
  - The hidden microscope's corrector has no lenses: a -1 transfer. Its condenser coils were
    chosen again (x1.017) so that the focus sits at the console's CL3 at MAG 100k, by a trace
    before any fit of this microscope.
  - The recording script brings CL1, CL3 and OLf to each row from below as well, so that every
    lens's lag is the same at every setting (H3). A recording takes 93 s on the CPU.
  - Notebook 2 fits one lag per lens (CL1, CL3 and the projector lenses, in DAC units): under a
    fixed approach the lag of a lens is a constant (H6 allows a fit of hysteresis above 10 px;
    the harder test missed E14 by 14 px). It fits with the corrector on and off, and a fit stops
    on its misfit (R34).
  - Results on the new recording (`harder2`, 1253 frames, 2026-11-21): best misfit 112 against
    11 444 before; frames in the fit 0.05 px rms; E1 0.06 px rms (MAG), E2 0.4 %, E14 0.0 px,
    E15 0.29 px rms; E9 passes (0.05 px at most); the fit picks the corrector on and the mirror
    image. 1000 more steps move no prediction.
  - Speed (R33): TemGym's pole-piece integration was the cost: 100 steps instead of 200 halve it
    and move predictions by 0.005 px. Levenberg-Marquardt crawled along a curved valley (condition
    number 1e19) and stopped 3.5 px of prediction short; geodesic acceleration (Transtrum and Sethna)
    reaches the minimum in about 100 steps. The fit cell takes 9.7 min, E9 3.5 min: 15 min in all,
    over R33's 10.
  - The gauge count is now 4 (the relay's focal length is gone). Fitting CM's bore and the
    corrector's bottom as well sent the fit to a misfit of 3673; held, they move no prediction
    (E9, 0.05 px). Six numbers stay held.
  - E3 asserted that the predicted beam lies inside the frame to two rms radii. At MAG row 13,
    IL1 +1000, IL3 -1000 the beam is 68 px rms and only 2 of 6 frames meet that margin, while all
    6 measured frames hold a whole beam. The assert now follows E3's text (the measured beam).
  - Open: only one of five objective starts reaches the best misfit; two stop at a second minimum
    (11 264). The check that two starts agree fails in two runs. Asked the principal.
  - 2026-11-21, the principal chose to widen the start grid. Starts at 1.5, 2, 2.5, 3, 3.5 and 4
    times the gap (24 fits; 0.5 and 1 dropped, lost in both runs). Still 1 of 6 reaches 112
    (the 3 times start); 1.5, 2 and 2.5 stop at 11 264, 3.5 at 42 568, 4 at 1 230 036. Every
    other check passes with the same numbers as above (E1 0.06 px, E9 0.05 px).
  - 11 264 is a real minimum, not a stall: no damped step and no Gauss-Newton step lowers it, and
    1000 more steps end at 11 260. Its frames miss by 1.1 px rms in MAG and 0.2 px in DIFF (the
    best fit: 0.06 and 0.03). It holds lens lags of 6890 DAC for CL3, 2320 for IL2, -1153 for IL3
    and -884 for CL1, where the best fit's are below 370. The hidden lag is a few hundred DAC.
    The start at 3.5 times stays lost after 1400 steps. Third failure; asked the principal.
  - 2026-11-21, the principal's answer: fit the objective as its gap, with its focal length held
    (see Decisions). Lens hysteresis findings are in `design/lens_hysteresis.md`. Fits with the
    focal length held at 2.07, 1.7 and 1.9 mm are running.
  - Gap fits (harder recording, gap started at 1, 1.5, 2, 3 and 4 times notebook 1's 0.028 mm):
    the focal length is not a free choice. Held at 1.7 mm: best misfit 108, E1 to E15 pass, 1 of 5
    starts. At 2.07 mm: best 132, E1 to E15 pass, 1 of 5 starts, lags of IL1 to IL3 500 to 700 DAC.
    At 1.9 mm: no start reaches a good fit; the best is the second minimum (11 262, CL3 lag 6992).
    The second minimum (about 11 260) appears at every focal length, so it is not the objective's
    trade. The gap fit fails the start check; by the decision, the lags are now fixed at zero
    (fits with f 2.07 and 1.7 mm running).
  - E9 on the 2.07 mm gap fit: the six old fixed numbers move predictions by 0.07 px at most. The
    focal length 20 % longer moves them by 0.38 px but raises the misfit from 132 to 189: the
    recording sees the focal length weakly, it is not a gauge freedom. The predictions barely
    depend on it, which is the honest statement: the focal length is not known to 20 %, and need
    not be for predictions.
  - Lags fixed at zero (the decision's fallback), focal length held at 1.7 and at 2.07 mm: 3 of 5
    gap starts reach the same misfit (11 488 and 11 525) and predict within 0.04 px of each other;
    the starts at 3 and 4 times, and the corrector off, are lost. Every check passes: kept-back
    MAG 1.24 px rms (2.7 px at most), DIFF 0.13 px, size 8.7 %, spot 0 1.9 px, MAG focus series
    2.2 px rms (3.9 at most). The two focal lengths predict within 0.47 px of each other. Against
    the free-lag fit at 1.7 mm (misfit 108), the predictions differ by 1.3 px rms, 4.2 px at most:
    that is the cost of not knowing the lags. The second minimum near 11 260 with free lags is this
    no-lag column held up by lags. The lags are now out of the model; the full fit with E9 runs
    alone on the GPU for R33's timing.
  - Full no-lag fit alone on the GPU: fit cell 10.4 min (the lost start at 1 times runs 1300
    steps), E9 5.8 min. E9 fails on the focal length: held 20 % longer, the refit reaches a misfit
    of 212 against 11 525. The focal length is not a free choice once the lags are gone; the
    11 525 fit sits at a wrong focal length that the lags used to make up for. (The CL1 refit got
    lost, 31 px.) As agreed with the principal on 2026-11-21 for this case, the focal length is
    fitted again: held for the first 300 steps, then free, with the gap as its own number. Gap
    starts at 1.5, 2, 2.5 and 3 times (1 and 4 dropped: lost). Running.
  - With the focal length free from notebook 1's 2.07 mm, all four starts end at 1.61 mm and a
    misfit of 11 486 (fit cell 9.4 min); the misfit-212 fit near 2.5 mm is not found. Now the
    focal length starts at 0.8, 1 and 1.2 times notebook 1's, the gap at 1.5 and 2.5 times
    (24 fits). If the best of these still fails a check, the loop stops and asks the principal.
  - 2026-11-22, the principal chose to pin the lags (option C). The recording now also brings
    CL1, CL3 and the projector lenses to their values from above, one at a time, at MAG X8000,
    X40k, X500k and DIFF 8 cm, 40 cm, 1.5 m (`ABOVE_ROWS`; PL1 at 64000 comes from 65535; jemwrap
    unchanged). New recording `lens_changes_simulated_harder_2026-11-22.csv`: 1433 frames, 180
    from above (5 lost, among them CL3 at X500k). Its other blocks match the 11-20 recording
    within 0.7 px, so the hidden microscope is unchanged. Jumps between the approaches: CL3 up
    to 43 px, CL1 up to 88 px, IL1 up to 43 px in DIFF.
  - Fit with pinned lags, focal length held at 2.07 mm: lags 135 to 201 DAC (the hidden
    microscope's are at most 150). Misfit 18 938, 3 of 5 starts agree within 0.04 px. E1 1.22
    px rms MAG, 0.20 DIFF; E2 8.4 %; E3 0.6 px; E14 4.6 px; E15 2.25 and 0.72 px rms; kept-back
    from-above frames at 1.5 m 0.70 px rms. E9 fails as before: the focal length held 20 %
    longer refits to a misfit of 215 (7.8 px at spot 0), and CL1's position moved refits to
    16 947 (6.4 px at spot 0).
  - Freed from 2.07 mm the focal length again slides to 1.61 mm (18 903); starts at 1.0, 1.2 and
    1.4 times do not find the 215 fit either. So the lags were not the cause of the objective's
    trouble: pinning them fixed the lags and the second minimum, not the focal length. Stopped
    to ask the principal (2026-11-22).
  - Diagnosis (2026-11-22, the principal leaning to a split objective): from the 18 938 fit, f
    stepped up by 4 % per step refits to 156, and walked back down to 2.07 mm it stays at 145.
    So f held at 2.07 mm has a fit with misfit 145; the 18 938 fit is a trap of the other
    numbers, not of the objective. Nine tenths of the trap's misfit is beam sizes; its condenser
    coils are 35 % weak (0.052 against 0.083), with large CL1 and CL3 bores. The 145 fit: lags
    146 to 150 DAC (hidden 150); E1 0.05 px rms MAG, 0.03 DIFF; E2 0.3 %; E14 0.1 px; E15 0.37
    and 0.28 px; from above at 1.5 m 0.18 px. Over f from 0.82 to 1.22 times the misfit runs 116
    to 215 while predictions move at most 0.41 px: the recording barely sees f, and predictions
    do not care. E9 there: CM, corrector top and camera turn pass; f fails only the 1 % misfit
    rule (215, 0.41 px); CL1 position and corrector bottom refits fall into the trap (16 950,
    8.9 px) where first order says 0.02 to 0.04 px, so the search is fragile, not the model.
    The split objective is not needed. Fitting the centres first, or the sizes at a tenth for
    the first 300 steps, did not reach the 145 fit (all starts above 70 000). Next is a decision:
    how the fit finds the narrow valley (stopped to ask).
  - Cause of the trap: a beam has the same size before and past the condenser's focus. In the
    trap about 190 of 719 sized frames sit on the other side of the focus than in the 145 fit
    (beam grows with stronger CL3 in 697 frames against 509). The size misfit was a logarithm,
    which costs without bound at the focus, so a frame could not cross. The principal chose
    (2026-11-22) to score the size as (model - measured) / measured instead. Result: 2 of 5
    starts reach misfit 139 and agree (the others stop at 32 026 or more); condenser coils 0.082.
    E1 0.05 px rms MAG, 0.03 DIFF; E2 0.3 %; E14 0.1 px; E15 0.37 and 0.28 px rms; from above at
    1.5 m 0.18 px. E9: every fixed number refits to 139-140 and moves predictions at most
    0.04 px, except the focal length: misfit 209, predictions 0.41 px. It fails only the 1 %
    misfit rule. Open for the principal: judge f by its predictions (pass) or by the misfit.
  - The principal chose (2026-11-22) to judge f by its predictions: E9 in design/examples.md now
    asks only that predictions stay within 1 px, and the notebook names a number whose refit
    raises the misfit but moves no prediction as one the recording sees weakly; the objective's
    focal length is reported as not known from the recording. The notebook text is updated
    (lags, held f, size misfit, E9) and `neoarm_fit.ipynb` is rebuilt and runs top to bottom.
  - R33/E18 timing: the simulated recording takes 1.7 min on the CPU (limit 15). The fit cell
    takes 11.3 min and E9 3.9 min, about 16 min with the other checks (limit 10): over, a note
    by the decision of 2026-11-21.
  - R35/E19: the notebook leaves out each group of settings (its centres and sizes), with the
    singular values of J at the best fit (through JᵀJ the condition, about 1e18, is beyond double
    precision; a first version with `pinv(JᵀJ)` was wrong, found by the review). Uncertainty of the
    kept-back centres grows, at most, by 36 % without the plain rows, 318 % without the one-lens
    moves, 27 500 % without the focus series and 58 500 % without the from-above settings; without
    the spot sizes or the beam sizes some predictions are not determined. Every group earns its
    place. (Test on the saved fit; the rerun notebook holds the final numbers.)
  - Fresh-agent review (2026-11-22) of the notebook, the recording change and E9: fixed the E19
    numerics and grouping, the held-number count (seven), the E3 and E15 headings, dated misfits,
    CL1 in the recording's docstring, two long lines, a stale comment in `simulated_neoarm.py`
    (comment only, the hidden numbers unchanged), and the meaning of the extra columns of
    `lenses`. Open for the real NEOARM: PL1 sits at 64000, so its from-above approach starts at
    65535, 1535 units above instead of 2000 (H3's same distance); harmless in the simulation,
    where the widest loop element is 400 units.
- Stage 7 (2026-11-22, done). `centre_beam.py` finds the beam in one frame, reads the fitted
  model's px per DAC unit of CLA1 (MAG) or CLA2 (DIFF) at the microscope's row from
  `neoarm_response.json`, and changes the deflector once. Notebook 2 writes that table from its
  derivatives, at the lens values the recording read at each row (the console tables give CM as
  58624 where the microscope reads 26624; with them the table was wrong by a factor of 3 and
  45 degrees, found in the first rehearsal). No TemGym or JAX at the microscope. `find_beam` moved
  to `microscope_functions.py`, since two scripts need it (pyjem rule 5). The script keeps the new
  deflector value and restores the start only after an error. E4 on the simulated NEOARM, the row
  selected first so that the beam lands off centre: MAG X60k 84 px to 2.1 px; DIFF 120 cm 265 px to
  10.0 px; DIFF 12 cm 32 px to 4.8 px. At MAG X300k the beam was off the SightX and the script
  stopped, as it should: finding a lost beam is the recording script's spiral search.
- Stage 8 (2026-11-22, done; the principal chose "simulator + scripts": the model does not yet
  follow a balance change, deferred). The simulated NEOARM ignored ShiftBal and TiltBal; now CLA1
  (shift) and CLA2 (tilt) are commands to one TemGym `DoubleDeflector` in
  `simulated_neoarm.py` (no change to TemGym itself). At the console's balances each command acts
  as its single kick did before, so the recording is unchanged: a re-recording matched all 973
  block frames to 0.0 px. The balance scale (2e-3 per DAC unit) is a choice, with no JEOL source;
  with it the pure tilt sits 2260 units of TiltBal and the pure shift 751 units of ShiftBal from
  the console's values (computed from the hidden optics before any script ran). Two model-free
  recipes, `tilt_balance.py` (MAG, X15k to X500k row by row, wobble CLA2, centre with CLA1) and
  `shift_balance.py` (DIFF, 30 cm to 150 cm, wobble CLA1, centre with CLA2), share `wobble`,
  `set_balance` and `centre_with` in `microscope_functions.py`. A balance change also moves the
  beam, so each step is sized by a probe to move the beam at most 30 px and the beam is recentred
  after it; probes and centring kicks halve while they lose the beam. Rehearsal problems found and
  fixed on the way: the beam left the SightX on a 500-unit trial, on a jump from X100k to X200k,
  on the centring kick at X400k, and at 60 px moves at X500k. E10: after the scripts the tilt
  wobbler at X500k swings the beam 0.1 px (x and y) and the shift wobbler at 150 cm 0.1 px; TiltBal
  ended at +2264 and ShiftBal at +751 from the console's values. `centre_beam.py` now says its
  table holds only for the balances of the fitted recording.
- Stage 9 (2026-11-22, done). `alignment_window.ipynb` runs one alignment script (`SCRIPT` at the
  top: tilt_balance.py, shift_balance.py or centre_beam.py) in a thread of its own, with no code
  added to the script (D7), and redraws every 3 s, from the script's own jemwrap monitor: the
  fitted column at the microscope's present values (TemGym's legacy plot), the SightX frame with
  the model's beam as a circle, and the script's last printed lines. Plotting library (D8, the
  agent's proposal): matplotlib with IPython's display handle, already installed; no Plotly. One
  notebook for every script instead of one per script, to keep one copy of the column (stage
  table said one per script; for the principal's review). The column and `beam_on_sightx` are
  the same text as notebook 2's (a copy, so the window is readable alone). E11 on the simulated
  NEOARM with tilt_balance.py: the saved views (figures/alignment_window_*.png) change row, beam
  and printed lines as the script runs. Found on the way: the model's beam assumes every lens
  came from below; after centre_beam.py's rehearsal it sat 26 px from the real beam, and 0.3 px
  once every lens was brought from below. Near the condenser focus the model's beam is smaller
  (6.5 against 10 px rms), since the model leaves out the source's size. The window's model has
  no balances, so after a balance change its beam no longer follows. After the pivot scripts
  restore the start row the beam can be off the SightX (the new balance moves it): their
  docstrings say to centre it again.
- Stage 10 (2026-11-22, done). `record_hysteresis.py` works at MAG X40k with the beam moved
  (100, 60) px off the middle by CLA1, so that a change of magnification moves it. For each of
  CL1, CL3, IL1, IL2, IL3 and PL1 it takes a frame after five routes to the row's value: from
  below, 300 units higher from below, from above (PL1 from 65535), from below again, from above
  again. No relaxation (decision of 2026-11-21); jemwrap unchanged. `analyse_hysteresis.py` turns
  the jump between below and above into DAC units along the 300-unit step. On the simulated
  NEOARM: lag 150.0 DAC units for every lens (150.1 for CL1), the simulator's 150; jumps 1.2 px
  (PL1) to 25.0 px (CL3); routes repeat within 0.01 px. 30 frames.
- Stage 11 (2026-11-22). New `examples/neoarm/README.md`: the order of the files, how to rehearse,
  the words used, what to expect, what this version does not do.
  - Hallway test (a fresh agent with the README alone): the scripts' numbers matched the README
    (centre_beam 84 to 2.1 px, lags 150.0, shift balance 0.1 px). Answered: notebook 2 now says
    for each fit start whether it stopped on its misfit, reached its step limit (E17) or was left
    with the other mirror choice, and why most starts end far above the best; the E3 line says the
    two-radii count is stricter than E3; the README explains rows, kept-back settings, the E/R
    numbers, L, lags and stage references, run times, how to open the notebooks, that each
    rehearsal starts a new simulated NEOARM, and where hysteresis recordings go; prints name
    their units and origin.
  - Code review (a fresh agent): the balance algebra checked (the new coils agree with the old
    single kicks to 1e-21 m). Fixed: the window ran the script in a thread, where an interrupt
    did not reach it and its restore never ran (R25); now the script runs in the notebook's own
    thread and the view in a second one. `centre_with` and the balance probe stop with a plain
    message when even one unit loses the beam; a balance step is at most 2000 units; a balance
    that does not change the swing stops the script; every change prints a line (probes and
    trials too); the pivot scripts wait 2 s after a change of mode (R15); record_hysteresis.py
    takes the lenses off free lens control before it reads the row's values and waits 1 s after
    the row change; analyse_hysteresis.py reads the step from the table. After the fixes:
    centre_beam 2.1 px, lags 150.0, shift balance 0.1 px. Not changed, for the principal: the
    pivot scripts and centre_beam.py save no frames (pyjem rule 8; their docstrings say so), and
    four setting-back calls in record_lens_changes.py (stage 5) have no print line.
- After stage 11 (2026-11-22, the principal's request: controls for the guide's steps, and the
  Gaussian image renderer back and faster).
  - jemwrap (its `control` branch, uncommitted): `SETTABLE` gains the condenser and objective
    stigmators (CLs, OLs), the gun shift and tilt (GunA1, GunA2) and `apertures.position` (the
    aperture selected on the console; the selection stays off the list so a restore goes back to
    the same aperture), all over their full range (the principal's choice). The HT wobbler, which
    PyJEM can set but not read, is a new kind, `WRITE_ONLY`: set without a read-back, held as
    "off" in the snapshot, and always switched off by `restore`. Stage Z waits (no getter on the
    NEOARM; the principal: later). jemwrap's requirements.md updated; 202 tests pass (194 before).
    TemGym's simulated PyJEM gains a `GUN3` that keeps the wobbler switch, so every restore runs
    on the simulated NEOARM; the beam does not wobble there.
  - The sparse, differentiable Pallas renderer is back from `park/gpu-gaussian-renderer`
    (`src/temgym_core/render/`, `evaluate.py`, its tests and benchmark, as before commit ff666cc).
    With `JAX_PLATFORMS=cuda`: 285 renderer tests pass. Its benchmark reproduces its September
    numbers on the GH200 (20 000 narrow beamlets on 256 x 256: 0.8 ms).
  - Faster: on a realistic bundle (a wide coherent beam in 20 000 or 65 000 beamlets, a cross
    grating, a lens, 256 x 256) a call took 46 ms whatever the size, while its kernel took 1.1 ms:
    the packing and the grid's pixel map ran as about 120 one-at-a-time GPU operations, and a
    presence check rebuilt the map each call. Both now run compiled: 2.0 to 2.4 ms per image, same
    result (1e-12 against the dense kernel); the dense kernel went from 61 to 13 ms.
  - Tried and set aside: a separable matrix-product evaluator (0.35 ms) is exact only when no
    beamlet has its own xy curvature; a specimen gives each beamlet one (|delta| w^2 about 35),
    and the error stayed 3 to 20 %.
  - The first grating images were wrong (bright edges): the beam was decomposed over 26 um for a
    frame of 8.5 um, so the beamlets inside (about 116 nm) were far wider than the 5 nm edges.
    Resampled for the frame (beam rms one fifth of it, decomposed over +-3 rms, 9.2 nm per pixel,
    230 nm period = 25 px, edges 1.5 px): at focus the image differs from the exact one
    (|illumination x transmission|^2, magnified) by 3.9 % of its peak with 16 129 beamlets and
    0.8 % with 65 025: it converges as the sampling grows. 1.9 to 2.2 ms per 256 x 256 image.
    Defocus up to 4 um changes little, as expected against a Talbot length of about 42 um.
    Figure: examples/neoarm/figures/cross_grating_gaussian_beams.png.
  - Restoring the parked evaluate.py had dropped `ray_image`, which the simulated microscope
    imports; put back. 443 tests that touch the renderer, Gaussian evaluation and the simulator
    pass on the GPU.
  - Defocus checked (scratchpad only, at the principal's request): in free space a 10 nm square
    aperture's Gaussian-beam image matches the exact angular-spectrum diffraction within 2.1 to
    3.7 % at Fresnel numbers 20, 4 and 1, the same as at focus (so the error is the sampling of the
    edges, not the propagation). Defocus looked weak because 200 kV fringes are sqrt(lambda dz),
    2 to 5 nm for micrometres, under a 9.2 nm pixel; and a detector moved by M^2 dz is not a pure
    defocus when the specimen sits 33 um outside a 2 mm focal length (it changed the
    magnification by about 30 %): defocus through a lens belongs in the lens's focus.
  - `converged_gaussian_image` (evaluate.py, with `ConvergedImage`): decompose the beam, trace,
    render, double the beamlets across (2n + 1), until the intensity changes by at most
    `tolerance` of its peak, or `max_across`; it says whether it converged. Three CPU tests
    (tests/test_converged_image.py). On the GPU grating case at 256 x 256: changes 197 %, 27 %,
    4.1 %, 0.62 % at 63, 127, 255 and 511 across; with a 1 % tolerance it stops at 261 121
    beamlets, true error 0.36 % against the exact image, 0.73 s for the whole loop. Uniform
    refinement only; refining only where the specimen curves over a beamlet is not built.
  - Local refinement, `adaptive_decompose_gaussian` (source.py), at the principal's request.
    Beamlets sit on nested lattices (half the step per level); a beamlet is split where the
    specimen's action departs from its quadratic expansion across it, weighted by its amplitude
    and the brightest transmission it reaches. A split is the exact decomposition of the beamlet
    into the next level's; children at one site add, so a level costs about four beamlets per
    split one. Exactness: no split gives decompose_gaussian to 1e-16; splitting everything
    over 1, 2, 3 levels changes the field by 5e-5, 1e-4, 1.5e-4. Two CPU tests.
  - Found on the way: the "2 to 4 %" errors of the defocus check were a half-pixel offset in my
    exact reference, not the renderer. Corrected, uniform refinement of a 10 nm aperture at 10 um
    defocus converges fast: 0.82 %, 0.056 %, 0.0047 % at 127, 255, 511 across.
  - Gain of local refinement (against exact images): the grating, with edges every 10 to 15 px,
    about 1.4x fewer beamlets at 0.35 % (192 525 against 261 121), none at looser tolerances.
    The aperture (sparse): 4 769 beamlets for 1.2 % and 8 437 for 0.47 %, against 16 129 for
    0.82 % uniform, so 2 to 3x; but it stalls near 0.47 %, where uniform 255 reaches 0.056 %:
    some error is not yet caught by the split rule. The principal set it aside (2026-11-22): the
    uniform loop is used (65 025 beamlets give about 0.1 to 1 % in about 2 ms per image). The code
    and its two tests are parked in design/parked/, to revisit for gold particles.
  - Fresh-agent review (2026-11-22) of the renderer restore, its speed-ups, the convergence loop,
    GUN3 and jemwrap's controls. Confirmed: the compiled pixel map equals the old one bit for bit
    (rotations, flips, a traced flip_y under jit), gradients agree, the rest of render/ and its
    tests equal the parked version, everything importing temgym_core.evaluate resolves. Fixed:
    restore switched the HT wobbler off last and on every run (now first, only if the script set
    it, and never on from a snapshot); restoring the aperture position could move whichever
    aperture is selected now (now only the one the script moved, else a failure);
    converged_gaussian_image crashed with num_across above max_across and looped on a dark
    detector (both refused now); the simulator's wobbler state starts off. jemwrap 205 tests, the
    TemGym files touched 40 tests, centre_beam still 2.1 px. For the principal: every script's
    snapshot now also reads CLs, OLs, GunA1, GunA2 and the aperture position, so a getter that
    fails on the NEOARM stops every script, and a restore undoes what an alignment set on them
    unless the script keeps it out of the restore (as tilt_balance.py does for TiltBal); and
    evaluate_gaussians's default is now "auto" (the sparse renderer on a GPU, a sum cut at 1e-12)
    where it was the exact "scan", and method="gpu" is gone (no callers).
  - Next (the principal, 2026-11-22): a thin phase object, then gold particles.
- Aperture centring (Q17, the principal's "Yes please", 2026-11-22). TemGym's simulator now passes
  the condenser aperture's hole and Apt3 position into the whole column (`CondenserAperture`), and
  `simulated_neoarm.py` puts a `SigmoidAperture` at the drawing's CLAperture: holes 150, 70, 40 and
  10 um, 0.1 um per Apt3 unit about 2048, 0.5 um edges (the virtual NEOARM's numbers, no source
  for the real one), centre off the axis by (5, -3) um. The simulated beam is about 10 um rms at
  the aperture, so holes 1 to 3 hardly touch it: with hole 1, the simulator's default, a
  re-recording matched the fitted one to 0.03 px and 0.0 % in size, so notebook 2 stands. The
  10 um hole limits the beam and then sets its axis; `install_simulated_neoarm(condenser_hole=4)`
  selects it for the rehearsal, as an operator selects the aperture to centre. Notebook 2's model
  has no aperture, so with a limiting aperture its predictions are not checked (open).
  `centre_aperture.py` (MAG X8000): centre the beam with CLA1, take the beam with CL3 800 units
  below and above its value, learn from two trial moves how the difference changes with the
  aperture, move the aperture (at most 150 units at once) to where it vanishes, repeat. Rehearsal:
  the beam moved 11.6 px through the crossover; after one step 0.2 px (E21 passes). The aperture
  ends at (1971, 2052), not at the geometric centre (1998, 2078): the recipe nulls the shift, as
  the guide asks, which also absorbs CL3 being off the axis in the hidden microscope.
- HT centre (Q19, the principal: "Ok review, Q 19, then Q 23", 2026-11-22).
  - TemGym's simulator: `ht_wobbler_v` and `beam_voltage_v()`. While GUN3.SetHtWobbler is on, the
    high tension of each frame is ht_v + or - ht_wobbler_v, alternating; the voltage reaches the
    rays (and, in the older table-driven columns, the deflector kicks and PLA; the whole-column
    NEOARM's deflector kicks do not follow it). The pole-piece lenses follow the rays' voltage.
  - `simulated_neoarm.py`: the objective's focal length now grows with the momentum squared, as a
    magnetic lens does. HT_WOBBLER_V = 100 V, a choice (the NEOARM's value is not known).
  - Found: the simulated specimen was not in focus. The hidden objective is 4 % stronger than the
    drawing's, so the plane imaged onto the SightX lies 48.4 um below z = 0, and OLf (8e-7 of the
    focal length per unit) cannot move it there. An operator brings the specimen there with stage
    Z. `SPECIMEN_Z` computes that plane (the same within 0.1 um at every row from X30k up) and
    CROSS_GRATING sits there. Before this, the HT wobble moved the grating's out-of-focus shadow
    by about 8 px at 10 V, whatever CLA2 did; in focus it moves the image by about 1 px at 10 V and
    CLA2 changes that. Not a retune of HIDDEN: the operator's focus, as ALIGNING_KICK is the
    operator's alignment. For the principal to check.
  - `microscope_functions.feature_shift`: cross-correlation of two frames after removing the
    illumination (a 20 px blur subtracted) and tapering the edges, to a fraction of a pixel.
  - `ht_centre.py` (MAG X30k, CL3 3000 above its value): the HT wobbler on, FRAMES = 6 frames, the
    swing is twice the features' standard deviation along their line. It uses only the size of
    the swing, since a frame on the real NEOARM does not know which side of the wobble it caught:
    six trial CLA2 values fix the quadratic swing^2(CLA2), and CLA2 moves (at most 1000 units) to
    its least. Rehearsal: 6.8 px, after one step 0.8 px (E24 passes). CLA2 kept, the wobbler off.
- Review of today's code (fresh agent, 2026-11-22) and fixes.
  - B1 fixed: a restore left the other function mode's selector at the script's last row (the
    console keeps one selector per mode; a snapshot holds only the current one).
    `microscope_functions.enter_mode` returns the mode's selector and `leave_mode` sets it back,
    while in that mode, before every restore; in tilt_balance, shift_balance, centre_aperture,
    ht_centre, record_hysteresis, centre_beam (rehearsal) and record_lens_changes (now also after
    an error). Checked: shift_balance (MAG -> DIFF) leaves MAG row 13 and DIFF row 0 as found.
  - B2 fixed: the wave frame ignored the condenser aperture. The rays to the specimen are now
    weighted by the apertures' transmission, as in ray_image, and the Gaussian carries the
    transmitted fraction (a fully blocked beam gives a dark frame). Hole 4 at three positions:
    ray 0.0462 / 0.1175 / 0.0876, wave 0.0460 / 0.1168 / 0.0872. Also the curvature's ddof mix.
  - B3 open, for the real NEOARM: jemwrap's aperture guard reads GetExpKind, while Apt3's
    Get/SetPosition act on the kind set by SelectKind. Check at the microscope which one the
    console's selection sets before centre_aperture.py runs there.
  - Smaller: jemwrap forgets the aperture selection after restoring the position (36 control tests
    pass). Not fixed: centre_aperture.py has no guard against a response matrix of noise (a hole
    that does not limit the beam); the coherent factor's sign in PolePieceLens can be off by pi
    (the same for every beamlet, so images are unchanged); the wave path has no test.
  - Lens history: ht_centre.py took CL3 straight back down from the spread beam (from above). It
    now goes 2000 below first; checked, CL3 ends at the from-below value (DAC - lag).
- Q23 check (2026-11-22), not built. With the saved fit and the derivative method of E9 (each held
  number moved by its plausible error, the fit following):
  - Predictable: the patch as seen on the SightX in px (0.6 % for 20 % of the objective's focal
    length) and the magnification ratio between rows (0.0 %).
  - Not predictable: the convergence angle at the specimen (9 % for the focal length, 6 % for CM's
    bore) and lengths at the specimen in metres (the fit's lengths are in its own unit). Against
    the hidden column at MAG 40k, spot 2: fit 22 nm and 0.83 mrad, hidden 62 nm and 1.65 mrad.
  - The fit also misses the focused spot itself on the SightX: MAG row 13, fitted frame, model
    3.8 px against 7.3 px measured (row 20: 24.0 against 24.0). The fit has no camera blur (the
    simulated SightX blurs by 2 px rms), which matters most for small beams. Open.
  - So: sizes need a ruler in MAG (the grating's period in one in-focus frame); angles a ruler in
    DIFF (gold's 111 ring, about 10.7 mrad at 200 kV; the grating's spots, about 5 urad apart,
    are too close). For the principal.
- Rulers for Q23 (the principal: "Yes, both rulers", 2026-11-22). Started, not built.
  - TemGym has `PolycrystallineFilm.gold()` (111, 200, 220 rings; rays only): no new component.
  - Found: at the simulated NEOARM's MAG 40k state the beam at the specimen is a probe (62 nm,
    1.65 mrad rms). In DIFF the central disc reaches about 4 mrad, so the three rings (10.7, 12.4,
    17.5 mrad) smear into one plateau. CL3 cannot make it parallel (from -20000 to +3000 units the
    crossover stays within tens of um of the specimen). CM at 20000 gives 0.32 mrad, enough to
    separate the rings, but CM is not on jemwrap's settable list (pyjem rule 5): the operator would
    set the condenser for the ruler on the console, and the rehearsal sets it in the simulator.
  - The principal: "don't stop it - we need the ruler too" (2026-11-22). Built:
    - `simulated_neoarm.py`: `GOLD_FILM` (TemGym's gold film at the operator's focus) and
      `cm_dac`, a CM the operator sets on the console (20000 for the gold ruler, 0.3 mrad).
    - `microscope_functions.grating_period` (Fourier peaks of the cross grating, padded to 4096,
      refined by a parabola) and `ring_radius` (radial profile about the direct beam against all
      three gold rings at once at their known ratios; circles at least a quarter on the frame; it
      refuses when the rings do not fit; bins taken at their middles).
    - `record_rulers.py` (RULER = "grating" at MAG X40k with CL3 spread, or "gold" at DIFF row
      10): it brings the row's lenses from below first, the history the model assumes. Without
      that the grating read 3 % off (57.9 against 56.1 px): the lags move the magnification.
    - Rehearsal against the hidden column in the state measured: grating 57.53 against 57.65 px
      (0.2 %), gold 111 ring 46.65 against 46.62 px (0.1 %). Directly at four DIFF rows: within
      0.1 %; at three MAG rows: within 0.6 %.
  - The conversion, checked in scratch (not yet in the notebook): the fitted model's patch on the
    SightX and its DIFF disc in px (both safe from the held numbers: at most 0.7 % and 1 % for
    spread beams), times the rulers. At MAG 40k with CL3 brought from below:
    - CL3 +3000: 1.26 um and 29.9 mrad against the hidden 1.35 um and 32.0 mrad (-6.5 %, -6.4 %);
      -3000: -6.7 %, -6.9 %; +6000: -4.0 %, -3.9 %. The ratio angle / size (where the crossover
      is) agrees to 0.1 %; the fit's source spread is about 6 % small. E23 (5 %) not met yet.
    - At CL3's own value the crossover lies on the specimen (hidden: 4.8 nm, 0.09 mrad); the fit
      gives 38 nm and 0.64 mrad. A crossover is not predictable to 5 %: its size is set by what
      the paraxial model leaves out (and the camera blur the fit leaves out).
  - Paused for the principal's new order (2026-11-22): gold particles, rounding the code, then an
    alignment notebook and a predictor notebook (which takes the rulers in).
- Alignment notebook, started (the principal: one section per guide step, the scripts run on one
  shared simulated NEOARM, 2026-11-22).
  - `install_simulated_neoarm` returns the microscope, and does nothing when one is installed, so
    scripts run from a notebook share it. The simulator starts out of alignment by itself (the
    console's settings: pivot points, aperture and CLA2 out).
  - Running the sequence on one microscope found a fault in stage 8's balance law: a balance was
    the lower coils' ratio r to the upper, so it changed the kick by (1 - r); the tilt balance
    (about 2300 units) took r past 1 and left the beam 0.4 rad tilted at the specimen, so the
    shift balance then found no beam in DIFF. Now (2026-11-22) a balance sets the share s of
    each command's kick in the upper coils: upper s k, lower (1 - s) k, the kick kept, only the
    pivot moving, no singularity. Identical at the console's balances (the aligned beam and
    ALIGNING_KICK unchanged), so the recordings and the fit stand. The 2300 and 750 unit figures
    above are of the old law. A second law (balances acting only on changes from the console's
    values) was tried and dropped: it hid the recipe's gap instead of fixing it.
  - Still open: with the new law the tilt balance ends at TiltBal (22955, 22907) and the DIFF
    spot is then about 400 px off the SightX: the scripts need a routine that finds a lost beam.
  - Correction (2026-11-22): CL3 alone does make the beam nearly parallel in the simulator. At
    CL3's own value, from below, the beam is 4.8 nm and 0.09 mrad; earlier scans were too coarse
    and filtered by size. It is too narrow for the gold rings (few grains); a beam a few tens of
    CL3 units off parallel (about 0.5 to 1 mrad, 20 to 45 nm) is the likely ruler state, to be
    found by the script (the guide's "minimise the diffraction spot with CL3"). record_rulers.py
    uses CM 20000 as a stopgap until then.
- Basic routines (the principal: "build the basic routines first - but not with CM being movable",
  2026-11-22), in `microscope_functions.py`:
  - `set_lens_from_below` and APPROACH_DAC, shared.
  - `smallest_spot`: the lens value of the smallest spot, from 1 / peak brightness at offsets on
    both sides (a V blurred by the camera: 1 / peak = a^2 (dac - c)^2 + b^2). The peak, not a
    size from the beam's pixels: the gold's ring speckles upset find_beam's size.
  - `find_lost_beam`: square rings of trial values of a deflector until a whole beam shows, then
    centre_with. Test (MAG X40k): CLA1 900 units off found at the first trial, (-700, +800) at the
    16th; both centred to 1 px.
  - `set_balance`: the trial change now halves while its wobble loses the beam.
  - The gold ruler without CM (the CM option is gone from the simulator): record_rulers.py finds
    the smallest DIFF spot with CL3 and goes 40 units below. Rehearsal: smallest at CL3 33477
    (about 33483 expected), 111 ring 47.00 px against the hidden 46.62 (0.8 %). Grating
    unchanged: 57.53 against 57.65 px.
  - Balance law: the second law is back (a balance shares only the console's change of CLA1 or
    CLA2 between the coils; the operator's aligning kick keeps the console's share). Under the
    first law the tilt balance "passed" E10 in a broken state: the beam 16 um off and 385 mrad
    tilted at the specimen, CLA2 acting on nothing. That pass was false.
  - Blocker, for the principal: under the second law the tilt balance again ends at TiltBal
    (22955, 22907), and there CLA2 tilts the beam at the specimen by 0.018 mrad per 100 units
    against 4 mrad at the console's balance: a CLA2 kick that pivots on the specimen hardly tilts
    it, so in DIFF the shift balance cannot centre the spot with CLA2. In this hidden condenser
    the specimen's conjugate lies far from the coils (about 9 cm below the lower pair), so the
    angular magnification there is tiny. The real NEOARM's tilt does not behave so. A change of
    the hidden column's geometry is a decision for the principal (it means a new recording and
    refit).
  - Investigated (the principal: "Yes, investigate", 2026-11-22). The map from the lower CLA
    coils to the specimen runs through the corrector, CM and the objective's pre-field. A kick
    pivoting at the specimen's conjugate (about 85 mm below the lower coils) tilts the beam at the
    specimen by 1/A, about 5 times the coil kick; at the console's pivot (4 mm above the lower
    coils) the same kick tilts it about 460 times and shifts it 20 um per mrad. Notebook 1's own
    column, every hidden number reset, gives the same (6 times, 78 mm below): it is the drawing's
    geometry, not a hidden number. 5 times means about 0.036 mrad per 100 units of CLA2, about
    +-10 mrad over its range: a plausible beam tilt. So the column is not at fault; the earlier
    "blocker" reading above is withdrawn.
  - The fault is the shift-balance recipe: it centres the DIFF spot with CLA2 (20-unit trial). With
    the tilt balanced, 20 units move the spot about 0.01 px and the solve runs away; and moving the
    spot a few hundred px at long camera lengths would need tens of mrad of tilt, outside CLA2's
    range and a change of the illumination. An operator centres a diffraction pattern with the
    projector deflector PLA (diffraction shift). PLA is not on jemwrap's settable list (pyjem rule
    5: for the principal) and not in the simulated NEOARM's whole column.
  - PLA added (the principal: "Yes do add PLA for sure", 2026-11-22): jemwrap's settable list
    (36 control tests pass, requirements.md updated); the simulated NEOARM's column, a Deflector
    midway between IL3 and PL1, 1.2e-6 rad per unit about 32768 (a choice; about 0.015 px per
    unit in DIFF; none at 32768, so the recordings stand). shift_balance.py centres with PLA.
    `centre_with` doubles a trial kick while it moves the beam less than 2 px.
  - The sequence on one microscope now: aperture 0.3 px (E21), tilt balance 0.1 px, shift balance
    0.0 px (E10), each in turn. ht_centre.py fails: after the tilt balance CLA2 tilts the beam
    about 0.036 mrad per 100 units, and the swing (about 6 px) hardly changes over CLA2's range.
  - Why: with CL1 and CL3 put on the axis the swing falls from about 6 px to 1 px, and with every
    lens and the SightX on the axis to 0. CL1 and CL3 sit 2 to 3 um off the axis (HIDDEN); their
    strength follows the high tension, so the illumination's tilt changes between the wobbler's
    frames, which CLA2's fixed tilt cannot cancel. The earlier pass of E24 (6.8 to 0.8 px) used
    about 1000 units of the unbalanced CLA2, about 30 mrad of beam tilt: a false pass. The guide
    centres the gun and condenser before this step (gun tilt and shift, spot alignment), steps
    that have no script yet.
- Alignment notebook `examples/neoarm/alignment.ipynb` (source built from scratch
  `alignment.py`; the guide's sections quoted at build time). First pass ran: aperture, tilt and
  shift balance, HT centre shown failing with the reason. The second pass stopped:
  centre_aperture.py's CLA1 centring asked for a value of 203853.
  - Why: after both balances the coils are weak. From the coils-to-specimen map (A -0.22, B
    -0.0187 m, C -5150 /m, D -441): a pure shift is 1 nm per 100 units, +-0.33 um over CLA1's
    range; a pure tilt 0.023 mrad per 100 units, +-7.6 mrad. At the console's pivots the same
    100 units shift 141 nm and tilt 3.3 mrad together. A beam shift of +-0.3 um cannot centre the
    beam at X8000; a real beam shift moves it over micrometres. The coils' strength
    (kick_per_dac, 1e-7 L rad per unit) was a choice with no source, only ever checked at the
    console's pivots. The earlier note that the tilt range is plausible stands for the tilt
    alone; the shift range is not. For the principal: the NEOARM's beam shift and tilt ranges.
  - HT and the gun: the guide's gun shift (no beam movement between spot 1 and 5) was tried with
    an ideal source shift (16.9, -18.3 um) and the beam re-centred: the HT swing grew from about
    6 to 45 px. Bringing the beam through CL1 and CL3 on their axes needs the gun tilt too (the
    guide's anode wobble; no anode in the simulator). Not solved.
  - Coils (the principal: "you'd have to search the internet or read it in a patent or paper",
    2026-11-22). No published beam shift or tilt range for a JEOL condenser deflector was found.
    Beam tilt: the coils of modern TEMs reach precession angles of 0 to 3 degrees, 1 degree (17.5
    mrad) common on a JEOL 2100F (Wikipedia, Precession electron diffraction). CLA1 30 and CLA2 5
    times stronger: +-10 um of pure shift (a choice) and +-38 mrad of pure tilt over half the
    range. shift_balance.py wobbles CLA1 by 7 units. Both passes then run on one microscope:
    aperture 0.2 px, tilt 0.4 px, shift 0.0 px, the second pass changing nothing; HT fails as
    above.
  - But the recording fails with these coils: at the console's pivots CLA1 moves the beam about
    42 nm per unit (MAG row 20: 5 units lose the beam). The simulated console's balances put the
    pivots at the drawing's deflector planes, far from right; a real NEOARM starts from a loaded,
    recent alignment whose pivots are nearly right. For the principal.
- Near-aligned start (the principal: "Yes, near-aligned start", 2026-11-22). The console's
  ShifBal and TiltBal now put the pivots nearly right: CONSOLE_SHARE from the coils-to-specimen
  map (pure tilt at L = -B/A, pure shift at L = -D/C, about 85 mm below the coils), off by
  HIDDEN["pivot_error_dac"] (ShifBal (-300, 200), TiltBal (600, -400), choices); the balance knob
  10 times finer (2e-4 of the share per unit, 3.5 um of pivot per unit; a choice).
  - alignment.ipynb runs both passes: aperture 11.6 -> 0.3 -> 0.4 px; Tilt X wobbler 7.9 -> 0.0
    -> 0.0 px; Shift X wobbler 5.0 -> 0.0 -> 0.1 px; TiltBal moved (599, -401), ShiftBal
    (-282, 217) against the hidden errors. Condenser alignment tilt (HT wobbler): 6.8 px
    throughout, shown failing with the reason.
  - The fit (scratch neoarm_fit.py) now draws CLA1 and CLA2 as the coils are built: one kick
    each, shared between the two coil pairs so that it acts at its fitted pivot plane, which may
    lie beyond the coils (prior 200 mm). The alignment window's copy of `column` too. A fit of a
    new recording (1467 frames) on the near-aligned simulator is running.
  - The principal on the HT step: "6 pixels does not sound so bad either? Maybe 6 pixels is an
    ok error." Proposed: ht_centre.py stops where CLA2 cannot lower the swing and reports the
    rest as beyond CLA2 (the condenser's misalignment), E24 reworded to match. 6.8 px is at X30k
    with the assumed +-100 V; at the guide's X500k it would be about 110 px.
  - Corrector variants on the near-aligned simulator: +1 and -1 with 3 % lost the beam at X400k
    (the condenser re-matched at X40k, the usual one at X100k: to be matched relative to it);
    free space is not a believable column: without the -1 imaging the hidden condenser needs a
    19 mrad aligning kick at the coils and pivots 24 coil spacings away. The principal said the
    NEOARM's corrector is off (Q13): the honest test is a hidden column built with the corrector
    off from the start. For the principal.
- Corrector off (the principal: "Yes, corrector off", 2026-11-22). `HIDDEN["corrector"]` =
  "free space"; condenser_turns_factor 1.017 x 0.9162 (the ratio of the smallest beam at the
  specimen, free space over -1); CLA1 and CLA2 strengths now follow from target ranges
  (COIL_RANGES: +-10 um pure shift, +-38 mrad pure tilt over half the range, from the map A, B, C,
  D). Pivots about 0.42 m below the coils (shares about -24: the pairs push nearly equal and
  opposite). The aligning kick is 19 mrad at the coils.
  - ht_centre.py stops where a CLA2 step lowers the swing by under 10 % and reports the rest as
    beyond the beam tilt (the principal: "Maybe 6 pixels is an ok error").
  - alignment.ipynb on this microscope: aperture 28.1 -> 0.3 -> 0.2 px; Tilt X wobbler 4.1 ->
    0.0 -> 0.0 px; Shift X wobbler 2.7 -> 0.0 -> 0.0 px; TiltBal moved (599, -401), ShiftBal
    (-277, 225); condenser tilt 5.7 -> 5.6 -> 5.7 px, "CLA2 at its least swing".
  - The predictor's recording fails at MAG X40k: the spot changes. With the corrector off the
    hidden column's spot changes, with the console's SpotA, tilt the beam at the specimen by up
    to 58 mrad (spot 0) and shift it 2.5 um; centring then drives CLA1 out of its range. The
    console's spot table was made for the real column; the hidden condenser (drawing positions,
    off-axis CL1 and CL3) does not match it with the corrector off. For the principal.
  - Spot alignment re-solved (the principal: "Re-solve SpotA per spot", 2026-11-22):
    `spot_alignment()` sets each spot's SpotA so the beam stays where it is at spot 2 (tilt then
    within 0.06 mrad too), missed by HIDDEN["spot_error_dac"]; spot changes now move the beam at
    most 34 nm and tilt it 0.75 mrad. The recording's failure was elsewhere: its per-row centring
    needed about 2800 units of CLA1 at X8000 at 0.3 nm per unit; COIL_RANGES CLA1 now 50 um
    (a choice: the beam onto the SightX at X2000).
  - Then the recording stopped at MAG X40k, spot 0: with the corrector off only spot 2 focuses
    at its console CL3 (beam rms on the SightX: spot 0 332 px, 1 136, 2 19, 3 159, 4 291).
    Matching the hidden condenser to the console's spot table (every spot in focus at its CL3)
    needs the condenser coils at 0.39 of the drawing's and CL1 moved 79 mm: not plausible. With
    the -1 corrector the same table needed 1.017. The console's spot table, set on the real
    NEOARM, fits a corrector region that images the beam (-1 or +1), not free space: evidence
    that the real corrector region is not empty optically (transfer lenses on while the
    corrector is off?). For the principal (Q13).
  - The principal (2026-11-22): "not an imaging corrector - its a probe corrector, but in TEM
    mode the software shows that all currents are zero - even to the transfer lens. Except one
    deflector." So free space stands, and the hidden condenser was wrong elsewhere: the drawing
    gives CL1, CL3 and CM one ampere-turns per DAC unit. With separate strengths
    (`condenser_lens_factor`: CL1 1.1712, CL3 0.8814, CM 1.0; CL1 and CL3 moved -0.92 and +1.24
    mm; a least-squares match to the console's spot table, held near the drawing) every spot
    focuses at its console CL3 within 1 nm per urad (beams 0.6 to 3.2 px rms on the SightX). The
    earlier note that the spot table favours an imaging corrector is withdrawn: it came from the
    shared strength. The fitted model also shares one condenser strength among CL1, CL3 and CM;
    whether its fitted gaps and bores make up for it, the refit shows.
  - State at the end of 2026-11-22 (not consistent yet; HIDDEN not frozen):
    - The condenser match now puts each spot's focus 300 units below its console CL3 (factors
      CL1 1.1686, CL3 0.8902; CL1 -0.93 mm, CL3 +1.24 mm), so that the console's CL3 from below
      lies just above focus, as record_lens_changes.py expects.
    - The recording still fails at spot 0: SpotA's position response at the specimen nearly
      vanishes there (determinant 69 against 14 000 (nm per unit)^2 at spot 2, the sign flipping
      between spots 1 and 0): at spot 0 CL3 images SpotA's plane onto the specimen. The solved
      spot alignment needs about -29 700 units at spot 0, against a smooth -352 .. +232 on the
      real console. So the matched condenser is not yet consistent with the console's SpotA
      column. A 59 times stronger SpotA, calibrated on that blow-up, was reverted.
    - The operator solves (ALIGNING_KICK, the focus match, spot_alignment) are made at nominal
      DACs; the scripts work with lenses from below (lags about 150 units). To be solved in the
      from-below state, from the scope's own hysteresis.
    - Not current: neoarm_fit.json, neoarm_response.json and alignment_window.ipynb belong to the
      -1 simulator; alignment.ipynb's last good run predates the 50 um shift range and the
      condenser match (the last run stopped at the tilt balance); the ruler, HT, lost-beam and
      E23 figures were measured on earlier simulators. No fit with the shared-coil model has
      finished.
- Final simulator of 2026-11-22 (the principal: "run to satisfy the requirements ... don't stop
  to ask questions ... make sure that what we have has a good chance of working on the neoarm
  first time"):
  - Corrector off (free space). The hidden condenser matched to the console's whole spot table
    of alpha 1 with lenses from below (least squares, held near the drawing): CL1 x1.0617, CL3
    x1.0276, CM x0.3833 (the mini lens), CL1 -0.50 mm, CL3 +0.63 mm, CM +2.68 mm, SpotA -0.08 mm
    and x1.043. Each spot focuses 300 units below its console CL3 within 0.4 nm per urad, and the
    console's own SpotA keeps the beam within 34 nm from spot to spot. The re-solved spot table
    is gone; the console's SpotA is used as it is.
  - The operator's aligning kick is solved with CL1 and CL3 from below (FROM_BELOW_LAG_DAC = 150,
    from the hidden hysteresis), and a new simulated NEOARM starts with every lens in that state.
    The aligning kick is now 0.2 mrad (it was 19 mrad).
  - A check of the real recording of 2026-11-10: one unit of CLA1 moved the beam about 2.5 px at
    X500k, about 1.25 nm at the specimen. The simulator's 50 um beam shift over half the range
    gives 1.5 nm per unit.
  - record_lens_changes.py: trial kicks up to 20000 units within the DAC range (realistic beam
    tilt moves a DIFF spot little at 8 cm); in DIFF it centres with PLA (its own trial kicks)
    and kicks CLA2 for the data; PLA goes back to its start value before MAG. Recording: 1659
    frames, `recordings/lens_changes_simulated_corrector_off_2026-11-22.csv`.
  - The fitted model gains PLA (a kick between IL3 and PL1 with its own strength, offset and
    place); the reversed-kick starts turn the deflectors above the corrector only.
    neoarm_response.json now names PLA for DIFF, and centre_beam.py centres DIFF with PLA.
  - Rulers: the grating reads 58.36 px against the hidden 57.65 (1.2 %). The gold ruler does not
    work on this simulator: the smallest DIFF spot lies about 3400 units below the console's CL3,
    and there the pattern shows a filled cloud, not rings, with a direct beam far wider than the
    angle at the specimen. Not resolved; the predictor notebook gives sizes in nm from the grating
    and angles in px of DIFF row 10 until a gold ruler works. record_rulers.py searches the
    smallest spot coarse then fine (smallest_spot: the windowed rms of the spot, robust to
    saturation and rings).
- Second review (fresh agent, 2026-11-22 night) and fixes: the PLA response was taken against
  three deflectors (fixed); DIFF's own PLA and selector now go back through enter_mode/leave_mode
  in every script; every clean-up step runs inside try/finally so the restore always runs; CL3
  comes back from its lens value at the start (the snapshot holds the free lens control value);
  find_beam ignores single hot pixels and takes one count as the least noise; centre_with cannot
  go to and fro; set_balance stops after 8000 units in all; every script checks binning 4; the
  recording centres DIFF with PLA in every branch; the ruler functions moved into
  record_rulers.py; one APPROACH_DAC; the E thresholds named; every set-back printed. Kept: the
  three copies of the fitted column (fit, window) and find_lost_beam without a caller.
- The fit gains separate coil strengths for CL1, CL3 and CM (priors 0.2, 0.2, 0.6 in log) and
  starts CM at 1.0, 0.5 and 0.3 of the drawing's (72 starts). Results on the final simulator:
  misfit 457 (corrector -1 chosen, the hidden one is off; the predictions do not depend on it,
  M2); E1 0.18 / 0.05 px rms; E2 0.5 %; E3 0.3 px; E14 1.1 px; E15 5.4 px. The gauge freedoms are
  five (CM's gap, bore and strength trade). The linear screen of the fixed numbers names the
  objective's focal length (0.58 px against 0.5); E9 decides. E9 fails: with CL1 moved 5 mm or the
  focal length 20 %, the refits (ten steps) end near a misfit of 23 900 and their predictions
  differ by 39 and 44 px. Open: the refits do not find the valley again.
- Corrector test (the principal: "we also need to be careful that our assumptions about the
  corrector have zero effect on our ability to model the real microscope, if the corrector is on
  or off", 2026-11-22). `HIDDEN["corrector"]` ("minus one" as before, identical). Running: three
  hidden correctors (free space; -1 with a 3 % magnification and a 50 mrad turn; +1), each
  recorded on the simulator and fitted with the fit's two forms (off, -1), checked by the
  kept-back predictions.
  - +1 (recorded with the old coils): E1 3.3 px rms MAG, 1.4 px DIFF, E2 0.5 %, but the best
    misfit 22 534 (139 for -1) and E14 (spot 0) 20.9 px: fail. A +1 corrector is the -1 column
    with everything above it turned by half a turn, which the fit can represent exactly; its
    starts all had kicks pointing one way. The fit now also starts with the kicks reversed
    (scratch neoarm_fit.py). Refit of the same +1 recording (the earlier simulator, single-kick
    deflectors): best misfit 138 (the -1 microscope: 139), from a reversed-kick start; E1 0.06 px
    rms MAG, 0.03 px DIFF; E2 0.3 %; E14 0.1 px; E1, E2, E3, E14, E15 and E9 pass. The corrector's
    sign no longer matters to the predictions.
  - Free space and -1 with 3 % and a turn could not be recorded: the console's CL3 was made for
    the -1 corrector, and the beam left the SightX. They now re-match the condenser's strength to
    the console's CL3 first (free space needs 0.93 of the -1 corrector's; -1 with 3 %, 1.018).
- Specimen frames (Q18, the principal: "Yes", one coherent beam first, 2026-11-22).
  - TemGym's simulator puts its `specimen` into the whole column at its plane (ray frames show its
    shadow), and with `wave_beamlets` images it with one coherent Gaussian: the rays are traced to
    the specimen, a round Gaussian takes their centre, tilt, rms size and curvature there (the rest
    of their spread, the incoherent part, is left out), it is decomposed into wave_beamlets^2
    beamlets over +-3 rms, run through the specimen and the column below, and rendered.
  - TemGym gap found and closed with the principal's OK: `PolePieceLens` refused Gaussian beams.
    Now the centre, slope and path follow its ray map, the curvature the ABCD law with the Larmor
    rotation, the decay by congruence, the amplitude 1/sqrt(det(A + BQ)). Three tests: the ray map
    for the centre and path, a positive decay, and a 961-beamlet decomposition through a rotating,
    off-axis lens giving the whole beam's field to 2e-5. The old test that Gaussians are refused
    became a test that they pass.
  - `simulated_neoarm.py`: `CROSS_GRATING` (2160 lines/mm, a 463 nm period; 60 % holes, bars 5 %,
    20 nm edges, turned 0.3 rad: choices) and `specimen`, `wave_beamlets` options of
    `simulated_neoarm` and `install_simulated_neoarm`. Frames at X8000 with the beam spread show
    the grating's bright holes and dark bars in the illumination disk, rays and waves alike; at
    X40k the ray frame is speckled by the finite rays, the wave frame smooth.
  - Speed: a wave frame took 3.6 s (127 across); PolePieceLens's field integration ran op by op
    (0.18 s a call) and is now one compiled function; the trace below the specimen is compiled once
    per column structure; the image uses the GPU renderer. Now 0.08 s a frame at 127 or 255 across,
    after a first compilation of about 9 s. The ray frames are unchanged: a re-recording matches
    the fitted one to 0.014 px. 51 tests of the simulator, lenses and convergence pass.
  - Not yet: partial coherence (the principal: later), gold particles, the HT dependence (Q19).
- Per-script restore (Q20, the principal: "Q20 yes", 2026-11-22). jemwrap's
  `settable_snapshot(monitor, keys)` reads only the named values (and the TEM-mode and
  function-mode checks), so `restore` sets back only those; without `keys` it reads every
  settable value, as before; an unknown name is refused before any call. Three new jemwrap
  tests, 208 in all. Every script in examples/neoarm names what it changes in `CHANGES`, so a
  getter that fails on the NEOARM stops only a script that needs it, and a restore no longer
  undoes an alignment's result on a value the script did not touch.
- Alignment by recipe (S8, added 2026-11-22 at the principal's request). The guide's steps, what
  each needs, and whether jemwrap's list allows it today:

  | Step | Recipe on the SightX | Controls | Allowed now |
  |---|---|---|---|
  | Aperture centring | beam centre either side of crossover, aperture to the mean | CL3, aperture position | no (apertures) |
  | Gun tilt, gun shift | anode wobble / spot 1 vs 5, beam centre | GunA1, GunA2, anode wobbler | no (gun) |
  | Spot alignment | none in the guide | SpotA | yes, no recipe |
  | Corrector beam tilt | beam intensity spread | ASCOR, outside PyJEM | no |
  | Condenser astigmatism | beam roundness either side of crossover | CL3, condenser stigmator | no (stigmator) |
  | Pivot points, tilt and shift | wobble CLA2 / CLA1, beam motion, linear solve | CLA1, CLA2, TiltBal, ShiftBal, CL3, IL1 | yes (stage 8) |
  | CLA2 by HT wobbler | feature motion under HT wobble | HT, a specimen | no |
  | Objective stigmator | FFT of an amorphous image | objective stigmator, a specimen | no |
  | Diff focus | transmitted beam motion under shift wobble | IL1, CLA1 | yes |
  | Image focus | contrast under image wobble | stage Z, a specimen | no |

  None of the recipes needs the fitted model to work: each closes the loop on the SightX. The
  model sets the wobble and step sizes, keeps the beam on the SightX through changes of row
  (S2, S3), and the simulated microscope rehearses each recipe. Open: Q15 (S8 beside or instead of
  S6's model part) and Q16 (which steps next).

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
| 6b | **The interview of 2026-11-20** (M2, M16, M17, R33, R34, R35). The corrector becomes a transfer block drawn as a rectangle, in two versions: free space and a transfer of magnification -1; the fit runs with each. The hidden microscope loses its corrector lenses. A fit stops on its misfit, not on a step count. The fit with its checks within 10 minutes on the GPU, the simulated recording within 15. The notebook shows what each group of settings adds, and a condenser measurement goes in only where the derivatives show it improves a spot-size prediction. | E16, E17, E18, E19, then E1 to E15 again. |
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
| The live view is drawn from jemwrap's state, not by the script (D6). | The agent proposed, the principal confirmed 2026-11-21 | Yes |
| The sample-plane scale is the nominal magnification (M7). | The principal | Yes |
| The lenses are `PolePieceLens` components, Lenz's field with a gap and a bore (M10). | The principal, 2026-11-19 | Yes |
| The console table sets the MAG focus and the mean MAG and DIFF scale only (M11). | The principal, 2026-11-19 | Yes |
| jemwrap control: plain functions, the narrow settable list, every lens through free lens control, the selector limited to its mode's rows (J14). | The principal, 2026-11-19 | Yes |
| The fits read small tables committed next to them; the recording zips stay outside git (J15). | The principal, 2026-11-19 | Yes |
| The objective is fitted as its gap (its distance beyond its focal plane); its focal length is a held number, checked by E9. If that fit passes, the lens lags stay fitted as one constant per lens; if it fails, the lags are fixed at zero; the loop stops only if that fails too. | The principal, 2026-11-21 | Yes |
| R33's time limit: the fit and E9 are timed separately (about 12 and 4 min); an overrun is recorded as a note, not a reason to stop. | The principal, 2026-11-21 | Yes |
| Stage 10 reaches lens values from below and from above only, with no lens relaxation, so jemwrap's settable list does not grow. | The principal, 2026-11-21 | Yes |
| The $20 ceiling (B3) does not apply: the principal is on a subscription. | The principal, 2026-11-21 | Yes |

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
