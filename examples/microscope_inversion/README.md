# The NEOARM through PyJEM: recording, and fitting TemGym to the recordings

| Where | What |
|---|---|
| [`at_the_microscope/`](at_the_microscope/) | Copy this folder to the microscope PC (needs numpy and PyJEM only). |
| [`at_the_microscope/neoarm.py`](at_the_microscope/neoarm.py) | Shared by every script: connect (real or simulated), `ask` (read), `change` (set; only printed in a dry run), `set_lenses`, `set_deflector`, `take_frame`, `find_beam`, `record` (a frame and every lens and deflector value: one row), `restore`, `save`. |
| [`at_the_microscope/record_projection_response.py`](at_the_microscope/record_projection_response.py) | How far CLA1 (MAG 40k) and CLA2 (DIFF 40 cm) displace the beam, with IL1, IL2, IL3, PL1 each changed up and down. |
| [`at_the_microscope/beam_per_magnification.py`](at_the_microscope/beam_per_magnification.py) | At every MAG row from 8k to 500k: size the beam with CL3, centre it with CLA1, and measure px per CLA1 unit. |
| [`at_the_microscope/record_predictor.py`](at_the_microscope/record_predictor.py) | One recording, five stages, no specimen: every lens value at every row and spot (`survey`); the same lens values reached from below and above (`history`); CLA1 and CLA2 kicks at every magnification row and camera length, and with IL1-PL1 changed at a few rows (`kicks`); tests of the deflectors (`deflectors`); the beam in MAG and DIFF over spot size and CL3 (`condenser`). Name stages to run only those; `--quick` is a short version; `--holes` also changes the condenser aperture. |
| [`recordings/`](recordings/) | Recordings as copied from the microscope PC, one folder per day (zips, not in git). |
| [`fit_projection_response.ipynb`](fit_projection_response.ipynb) | Fits TemGym `ElectromagneticLens`es (IL1-PL1) to a projection-response recording, tests the fit on settings it never saw and on the magnification recording, shows the frames, and writes a virtual NEOARM from the fit to `results/` (the one used for rehearsals is designed: see `../virtual_neoarm/`). |
| [`fit_predictor.ipynb`](fit_predictor.ipynb) | Builds a predictor from the latest `record_predictor.py` recording: how well a displacement repeats, the lag of each lens by the route to its value, the deflector tests, a TemGym column fitted to half the rows and tested on the other half and on settings recorded afterwards, and the beam's size and spread of angles against spot and CL3. All in DAC units and pixels. |
| [`results/`](results/) | What the notebook saves: the fitted model. |

Each script runs three ways, from `at_the_microscope/`:

    python record_projection_response.py --simulated     # the virtual NEOARM (needs TemGym)
    python record_projection_response.py                 # the real NEOARM, a dry run: reads it, changes nothing
    python record_projection_response.py --for-real      # the real NEOARM

A recording goes to `at_the_microscope/output/<name>/<date_time>/`: `table.csv` (one row per frame, the same columns
from every script), `console.json` (what the console had), `frames/`, and a copy of the scripts used.

A new script: start from `Neoarm("<name>", for_real=..., simulated=...)`, keep every change to the microscope inside
`scope.change`, `scope.set_lenses` and `scope.set_deflector`, and save every frame worth keeping with `scope.record(...)`.
