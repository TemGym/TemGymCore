# %% [markdown]
# # Beam tilt calibration: one image and the CLA2 values at each of four tilts
#
# You move the beam to a new tilt yourself (with the CLA2 knobs at the console). At each of the four positions this
# script takes one image from the screen camera and writes down the CLA2 values PyJEM reports. Nothing is changed on
# the microscope: it only reads CLA2, takes images and saves them.
#
# Run it from this folder, as a script or cell by cell:
#     python record_tilt_calibration.py
#
# Each image and the table of CLA2 values go in output/tilt_calibration/<date_time>/.

# %%
import csv
import time
from pathlib import Path

import numpy as np
from PyJEM import TEM3, sightx

POSITIONS = 4                                  # how many tilts you will set
CAMERA_IP = "172.17.41.1"                      # the camera PC
CAMERA_NAME = "screenCamera"                   # the SightX camera
FRAME_SHAPE = (1916 // 4, 1916 // 4)           # rows, columns: a 1916 x 1916 sensor at binning 4
FRAME_DTYPE = np.uint16

sightx.set_ip(CAMERA_IP)
camera = sightx.Detector(CAMERA_NAME)
deflectors = TEM3.Def3()

folder = Path("output/tilt_calibration") / time.strftime("%Y-%m-%d_%H%M%S")
folder.mkdir(parents=True)
rows = []

# %% [markdown]
# ## Check: the camera and CLA2 can be read
# CLA2 comes back as two numbers (x, y). The image should show the beam.

# %%
print("CLA2 now:", deflectors.GetCLA2())
frame = np.frombuffer(camera.snapshot_rawdata(), dtype=FRAME_DTYPE).reshape(FRAME_SHAPE)
print(f"an image of {frame.shape[1]} x {frame.shape[0]} px, {frame.dtype}, values {frame.min()} to {frame.max()}")

# %% [markdown]
# ## The four tilts
# For each: set the tilt at the console, press Enter, and the script reads CLA2, takes the image and saves it.

# %%
for number in range(1, POSITIONS + 1):
    input(f"\n>>> Tilt {number} of {POSITIONS}: move the beam to the tilt you want, then press Enter ")
    cla2_before = deflectors.GetCLA2()
    frame = np.frombuffer(camera.snapshot_rawdata(), dtype=FRAME_DTYPE).reshape(FRAME_SHAPE)
    cla2_after = deflectors.GetCLA2()
    if tuple(cla2_before) != tuple(cla2_after):
        print("   WARNING: CLA2 changed while the image was taken: this tilt is not clean, do it again")
    name = f"tilt_{number}.npy"
    np.save(folder / name, frame)
    rows.append(dict(tilt=number, CLA2_x=cla2_before[0], CLA2_y=cla2_before[1], image=name,
                     time=time.strftime("%H:%M:%S")))
    print(f"   tilt {number}: CLA2 = {tuple(cla2_before)}, image saved as {name} (values {frame.min()} to {frame.max()})")

# %% [markdown]
# ## Save the table

# %%
with open(folder / "table.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"\n{len(rows)} tilts saved in {folder}")
