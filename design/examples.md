# Acceptance examples: a predictive model of the NEOARM

One line each: an ID, the situation, the outcome that can be observed, a priority, and the
requirement it comes from. Tests and notebooks cite the IDs. An example marked `[assumed]`
has numbers that the principal has not confirmed.

| ID | When | Then | Priority | From |
|---|---|---|---|---|
| E1 | The model is fitted to a rehearsal recording and predicts rows it never saw, from MAG 8k to 500k. | The predicted beam centre is within 10 px rms of the measured centre. | must | R20, S2 |
| E2 | The same fit predicts the beam size on rows it never saw. | The median error of the predicted size is at most 20 %. | must | R21, S2 |
| E3 | The model predicts a setting where several controls change at once. | The measured beam is fully on the SightX. `[assumed]` | must | R19 |
| E4 | A prediction leaves the beam off centre, and the correction routine runs once. | The beam centre is within 10 px of the camera centre. `[assumed]` | should | S3 |
| E5 | A recording script runs on the simulated microscope and then on the real one. | The text of the script is the same apart from the one switch at the top. | must | S4 |
| E6 | A script stops on an error or Ctrl-C part of the way through. | Every lens and deflector the script changed has its start value, and free lens control is off. | must | R25 |
| E7 | A script runs at the microscope. | No two PyJEM calls are closer than the pacing time, and the script waits the settle time after each big change. | must | R14, R15 |
| E8 | A person who knows the microscope reads a script or notebook for the first time. | They can say what each step does to the column without opening another file. | must | R8, R27 |
| E9 | The fit runs twice on the same recording, with two different values for a static component. | The two fitted models predict the kept-back rows to the same accuracy, within 1 px. `[assumed]` | must | M3, U5 |
| E10 | The tilt wobbler and the shift wobbler of the alignment guide run on the simulated microscope after the pivot point script. | The beam moves less than 10 px at the highest magnification and camera length used. `[assumed]` | should | M8, S6, D4 |
| E11 | An alignment notebook runs its script on the simulated microscope. | While the script runs, the notebook shows the column with the beam, the beam on the detector and the text of the script, and the two views change as the script changes the microscope. | must | D6 |
| E12 | A static component is made a component that the fit adjusts. | The change is in the list of static components and the list of fitted values only. No other code changes. `[assumed]` | should | M9 |
| E13 | The model's beam size is computed against CL3 at the five spot sizes of the console data, alpha 1. | The beam goes through a focus at each spot. The CL3 value of the focus is within 2500 units of the operator's value and moves with spot size in the same direction. `[assumed]` | must | C1, C2 |
| E14 | The spot size changes from 4 to 0 on the simulated microscope, and SpotA takes its console value. | The fitted model predicts the new beam centre within 10 px. `[assumed]` | should | C4, C6 |
