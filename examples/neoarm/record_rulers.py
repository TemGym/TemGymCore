"""Record the predictor's two rulers, a cross grating in MAG and gold rings in DIFF (Q23, E23).

The fitted model predicts what the SightX shows, in pixels. A ruler turns pixels into lengths and
angles at the specimen. In MAG, the ruler is a cross grating of 2160 lines/mm. Its period is
463 nm. The period in px at one row gives nm per px at that row. In DIFF, the ruler is a gold
film. Its 111 ring lies at the angle wavelength / d(111), which is 10.65 mrad at 200 kV. The ring's
radius in px at one row gives mrad per px. The fitted model's ratios between rows carry each ruler
to every other row. `neoarm_fit.ipynb` reads the two frames.

One run records one ruler. Set RULER, put that specimen in, focus it, and start the script:

- "grating": the script works at MAG X40k and spreads the beam with CL3 over many periods.
- "gold": the script works at DIFF row 10. The rings separate only with a nearly parallel beam.
  The guide says "minimise the diffraction spot with CL3". The script finds that CL3 by where
  the rings match best, and measures them there. CM is not touched.

The script changes the function mode, the selector, the lenses of the row and CL3. Each lens of
the row comes to its value from below, because the model assumes this history. Every value goes
back to its value at the start, CL3 from below. It saves the frame, with the row and the measured
period or ring radius, to a new folder in `output/record_rulers/`.

A rehearsal on the simulated NEOARM runs on the CPU. From this folder:
JAX_PLATFORMS=cpu python record_rulers.py
"""

import time
from datetime import datetime
from pathlib import Path

import numpy as np

SIMULATED = True  # False at the microscope
RULER = "grating"  # or "gold"

if SIMULATED:
    from simulated_neoarm import CROSS_GRATING, GOLD_FILM, install_simulated_neoarm

    # The operator's specimen, in focus. The gold's rings need more rays for a smooth profile.
    if RULER == "grating":
        install_simulated_neoarm(specimen=CROSS_GRATING)
    else:
        install_simulated_neoarm(specimen=GOLD_FILM, ray_sample_power=18)
from PyJEM import TEM3, detector, sightx  # noqa: E402

from jemwrap import MicroscopeMonitor, PyJEMAdapter  # noqa: E402
from jemwrap.control import (  # noqa: E402
    read_value,
    restore,
    set_value,
    settable_snapshot,
    take_frame,
)
from microscope_functions import (  # noqa: E402
    APPROACH_DAC,
    ILLUMINATION_BLUR_PX,
    check_binning,
    enter_mode,
    find_beam,
    leave_mode,
    set_lens_from_below,
)

# ring_radius leaves out DIRECT_BEAM_PX pixels about the direct beam. It tries rings of the widths
# RING_WIDTHS_PX (rms). On the simulated NEOARM at DIFF row 10, the smallest direct beam is about
# 15 px wide. The rings there are about 5 px wide (2026-11-23). The values are choices.
DIRECT_BEAM_PX = 20
# A best match within EDGE_RADII steps of a quarter pixel from DIRECT_BEAM_PX is no ring. A wide
# direct beam falls off there and matches a ring at the inner edge best (2026-11-23). A choice.
EDGE_RADII = 8
RING_WIDTHS_PX = (2.0, 4.0, 6.0, 9.0)

# grating_period pads the frame to this many pixels before its Fourier transform. The peak of a
# period of 50 px falls on a grid of about 0.1 % of its frequency. The value is a choice.
PADDED_PX = 4096


def direct_beam(frame):
    """Return the centre (row, column) of the direct beam in a DIFF frame and its area in px.

    The direct beam is the largest patch of touching pixels at half the brightest pixel or more.
    The brightest pixel alone does not do. In DIFF the direct beam saturates the SightX, and so do
    many of the gold's diffracted spots. Of the pixels at the highest count, the first one in the
    frame is then often a diffracted spot (2026-11-23). The centre is the patch's centre of counts.
    Its area shrinks as the beam at the specimen becomes parallel.
    """
    bright = frame >= 0.5 * frame.max()
    seen = np.zeros(frame.shape, bool)
    largest = []
    for start in zip(*np.nonzero(bright)):
        if seen[start]:
            continue
        # Walk through the pixels that touch the patch, side by side, until none is new.
        seen[start] = True
        patch, to_visit = [], [start]
        while to_visit:
            row, column = to_visit.pop()
            patch.append((row, column))
            sides = ((row + 1, column), (row - 1, column), (row, column + 1), (row, column - 1))
            for near in sides:
                inside = 0 <= near[0] < frame.shape[0] and 0 <= near[1] < frame.shape[1]
                if inside and bright[near] and not seen[near]:
                    seen[near] = True
                    to_visit.append(near)
        if len(patch) > len(largest):
            largest = patch
    rows, columns = np.array(largest).T
    counts = frame[rows, columns]
    centre = ((counts * rows).sum() / counts.sum(), (counts * columns).sum() / counts.sum())
    return centre, len(largest)


def grating_period(frame):
    """Return the period of a cross grating in a frame, in px.

    The grating's two lines give the two strongest peaks of the frame's Fourier transform, at right
    angles. The function first removes the light's slow variation and tapers the frame at its edges,
    as in feature_shift. The period is the mean of the two peaks' periods, each placed between grid
    points by a parabola.
    """
    rows, columns = frame.shape
    ky = np.fft.fftfreq(rows)[:, None]
    kx = np.fft.rfftfreq(columns)[None, :]
    blur = np.exp(-2 * (np.pi * ILLUMINATION_BLUR_PX) ** 2 * (kx**2 + ky**2))
    edges = frame.astype(float) - np.fft.irfft2(np.fft.rfft2(frame) * blur, frame.shape)
    edges *= np.outer(np.hanning(rows), np.hanning(columns))
    power = np.abs(np.fft.fftshift(np.fft.fft2(edges, (PADDED_PX, PADDED_PX)))) ** 2
    middle = PADDED_PX // 2
    fy, fx = np.indices(power.shape) - middle
    # Frequencies below one period across the blur come from the light. They do not come from the
    # grating.
    power[np.hypot(fx, fy) < PADDED_PX / (2 * np.pi * ILLUMINATION_BLUR_PX)] = 0
    periods = []
    for _ in range(2):
        row, column = np.unravel_index(np.argmax(power), power.shape)
        offsets = []
        for before, peak, after in (
            power[row - 1 : row + 2, column],
            power[row, column - 1 : column + 2],
        ):
            offsets.append(0.5 * (before - after) / (before - 2 * peak + after))
        frequency = np.hypot(row + offsets[0] - middle, column + offsets[1] - middle) / PADDED_PX
        periods.append(1 / frequency)
        # The function removes the peak, its mirror image and their surroundings before the next
        # search.
        for r, c in ((row, column), (2 * middle - row, 2 * middle - column)):
            power[np.hypot(fy + middle - r, fx + middle - c) < 0.2 * frequency * PADDED_PX] = 0
    return float(np.mean(periods))


def ring_radius(frame, ratios, weights):
    """Return the radius of the first ring of a powder pattern in a DIFF frame (px), and the match.

    ``ratios`` are the radii of the rings over the radius of the first ring. ``weights`` are their
    strengths. For gold, these are the spacings and weights of TemGym's PolycrystallineFilm.gold.
    The centre is the centre of the direct beam, from ``direct_beam``. The frame's mean
    over each circle about the centre is the radial profile. The function compares the radial
    profile with Gaussian rings at those ratios. It does this for every first radius and ring width
    on a grid. It returns the radius of the best match, refined by a parabola. It also returns the
    match, the correlation of the profile with the rings, 1 for a perfect match. A fit of all
    rings at once also uses rings that lie too close to separate by eye. A best match at the edge
    of the radii tried means no rings to measure, and the function raises RuntimeError. At the
    outer edge, the outer ring falls off the frame. At the inner edge, the profile only falls off
    from the direct beam.
    """
    frame = frame.astype(float)
    rows, columns = np.indices(frame.shape)
    (centre_row, centre_column), _ = direct_beam(frame)
    radius = np.hypot(rows - centre_row, columns - centre_column)
    counts = np.bincount(radius.astype(int).ravel())
    profile = np.bincount(radius.astype(int).ravel(), frame.ravel()) / np.maximum(counts, 1)
    # Circles of which at least a quarter lies on the frame, and none inside the direct beam.
    on_frame = counts / np.maximum(2 * np.pi * np.arange(len(counts)), 1)
    largest = int(np.flatnonzero(on_frame >= 0.25)[-1])
    # Bin r holds the pixels from r to r + 1 px from the centre. Its middle is r + 0.5.
    r = np.arange(DIRECT_BEAM_PX, largest) + 0.5
    observed = profile[DIRECT_BEAM_PX:largest]
    observed = observed - observed.mean()
    first_radii = np.arange(DIRECT_BEAM_PX, largest / max(ratios), 0.25)
    best = (-np.inf, None)
    match = []
    for width in RING_WIDTHS_PX:
        scores = []
        for first in first_radii:
            model = sum(
                w * np.exp(-0.5 * ((r - first * q) / width) ** 2) for q, w in zip(ratios, weights)
            )
            model -= model.mean()
            scores.append(model @ observed / np.linalg.norm(model) / np.linalg.norm(observed))
        scores = np.array(scores)
        if scores.max() > best[0]:
            best = (scores.max(), width)
            match = scores
    k = int(np.argmax(match))
    if k > len(match) - 4:
        raise RuntimeError(
            "the rings do not fit on the frame: centre the pattern or use a lower row"
        )
    if k < EDGE_RADII:
        raise RuntimeError("no rings: the profile falls off from the direct beam only")
    before, peak, after = match[k - 1 : k + 2]
    radius = first_radii[k] + 0.25 * 0.5 * (before - after) / (before - 2 * peak + after)
    return float(radius), float(best[0])


def sharpest_rings(monitor, name, offsets_dac, settle_s):
    """Return the value of lens ``name`` (DAC) where the gold's rings match best on the SightX.

    The function takes a frame at the lens's value plus each of ``offsets_dac``, each approached
    from below, and measures the rings with ``ring_radius``. It returns the value of the best
    match. A frame without rings counts as no match. When the best value is the first or the last
    one tried, the function says so. A better one may lie beyond. The lens ends at its value at
    the start, from below.
    """
    start = int(read_value(monitor, f"lens.{name}"))
    values = start + np.asarray(offsets_dac)
    matches = []
    for dac in values:
        set_lens_from_below(monitor, name, dac, settle_s)
        frame = take_frame(monitor)
        if np.isnan(find_beam(frame.astype(float))[2]):
            set_lens_from_below(monitor, name, start, settle_s)
            raise RuntimeError(f"no beam on the SightX with {name} at {dac}")
        try:
            radius, match = ring_radius(frame, GOLD_RATIOS, GOLD_WEIGHTS)
            print(f"{name} {dac}: rings match {match:.3f}, 111 ring {radius:.2f} px")
        except RuntimeError as why:
            match = -1.0
            print(f"{name} {dac}: {why}")
        matches.append(match)
    set_lens_from_below(monitor, name, start, settle_s)
    best = int(values[np.argmax(matches)])
    if max(matches) < 0:
        raise RuntimeError(f"no rings at any value of {name} tried")
    if best in (values.min(), values.max()):
        print(f"the best rings lie at the edge of the values tried, {best}: try others")
    return best


adapter = PyJEMAdapter(
    connect=not SIMULATED, tem3_module=TEM3, detector_module=detector, sightx_module=sightx
)
# jemwrap spaces the calls to the real NEOARM by 0.1 s (requirement R16). The simulated NEOARM
# needs no spacing, so a rehearsal spaces its calls by 1 ms.
monitor = MicroscopeMonitor(adapter=adapter, rate_hz=1000.0 if SIMULATED else 10.0)

MAG, DIFF = 0, 4
# The rows of the rulers. At MAG X40k, the grating's period is 56 px on the simulated SightX. At
# DIFF row 10, the gold's 111 ring has a radius of 45 px, and its 220 ring stays on the SightX
# (2026-11-22). Both rows are choices.
MAG_ROW = 13
DIFF_ROW = 10
# CL3 spreads the beam over the grating this far above its value, approached from below, as in
# ht_centre.py. The value is a choice.
CL3_SPREAD_DAC = 3000
# For the gold, the script measures the rings at CL3 values about its value, first coarsely and
# then finely about the best coarse value. On the simulated NEOARM the rings match best about 2600
# units below the console's CL3, at the smallest spot. The 111 ring measures within 1 % of its
# true radius from 2800 to 1800 units below (2026-11-23). The fine steps are well inside that
# span. The real NEOARM's best value is not known. Both sets are choices.
COARSE_RING_OFFSETS_DAC = tuple(range(-6000, 3001, 1000))
FINE_RING_OFFSETS_DAC = tuple(range(-1000, 1001, 200))
# The lenses of the row. Each lens comes to the row's value from below, as in the recording that
# the model fits. A lens's lag moves the magnification by a few percent (2026-11-22, simulated).
PROJECTOR_LENSES = ("IL1", "IL2", "IL3", "PL1")
# The ratios of the gold rings' radii (111, 200, 220) and their strengths, as in TemGym's
# PolycrystallineFilm.gold. A ratio is d(111) / d(hkl) for a cubic lattice. A strength is the
# multiplicity times a guess of the structure factor.
GOLD_RATIOS = (1.0, 2 / np.sqrt(3), np.sqrt(8 / 3))
GOLD_WEIGHTS = (8.0, 4.8, 6.0)
# The script waits for the column to settle. The values are choices. R15 gives 2 s after a change of
# mode.
SETTLE_LENS_S = 0.5
SETTLE_MODE_S = 2.0
SETTLE_ROW_S = 1.0
# The simulated NEOARM has no time to settle, so a rehearsal does not wait.
if SIMULATED:
    SETTLE_LENS_S = SETTLE_MODE_S = SETTLE_ROW_S = 0.0

OUTPUT = Path(__file__).parent / "output/record_rulers" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
mode, row = (MAG, MAG_ROW) if RULER == "grating" else (DIFF, DIFF_ROW)

# CHANGES holds the values that this script changes, for the restore at the end (Q20 of
# design/requirements.md). The script does not read or set back a value that it does not touch.
CHANGES = (
    "eos.function_mode",
    "eos.current_mag_selector_id",
    "lens.CL3",
    *(f"lens.{name}" for name in PROJECTOR_LENSES),
)
check_binning(monitor)
start = settable_snapshot(monitor, CHANGES)
# The lens value of CL3 at the start. The snapshot holds the free lens control value, which is not
# the lens value while free lens control is off.
cl3_at_start = int(read_value(monitor, "lens.CL3"))
mode_values = None
try:
    print(f"function mode -> {'MAG' if mode == MAG else 'DIFF'}")
    mode_values = enter_mode(monitor, mode, SETTLE_MODE_S)
    print(f"selector -> {row}")
    set_value(monitor, "eos.current_mag_selector_id", row)
    time.sleep(SETTLE_ROW_S)
    for name in PROJECTOR_LENSES:
        set_lens_from_below(monitor, name, int(read_value(monitor, f"lens.{name}")), SETTLE_LENS_S)
    cl3 = int(read_value(monitor, "lens.CL3"))
    if RULER == "grating":
        set_lens_from_below(monitor, "CL3", cl3 + CL3_SPREAD_DAC, SETTLE_LENS_S)
    else:
        # The guide says "minimise the diffraction spot with CL3". The smallest spot is where the
        # rings are sharpest. The script finds it by the rings themselves. Above the console's
        # CL3, the pattern turns into a shadow image of the grains. Its bright patches can be as
        # small as the spot (2026-11-23).
        coarse = sharpest_rings(monitor, "CL3", COARSE_RING_OFFSETS_DAC, SETTLE_LENS_S)
        set_lens_from_below(monitor, "CL3", coarse, SETTLE_LENS_S)
        sharpest = sharpest_rings(monitor, "CL3", FINE_RING_OFFSETS_DAC, SETTLE_LENS_S)
        print(f"the sharpest rings: CL3 {sharpest}")
        set_lens_from_below(monitor, "CL3", sharpest, SETTLE_LENS_S)
    frame = take_frame(monitor)
    if RULER == "grating":
        measured = grating_period(frame)
        print(f"the grating's period at MAG row {row}: {measured:.2f} px")
    else:
        measured, match = ring_radius(frame, GOLD_RATIOS, GOLD_WEIGHTS)
        print(
            f"the gold 111 ring's radius at DIFF row {row}: {measured:.2f} px (match {match:.3f})"
        )
    OUTPUT.mkdir(parents=True)
    np.savez(OUTPUT / f"ruler_{RULER}.npz", frame=frame, mode=mode, row=row, measured_px=measured)
    print(f"saved the frame to {OUTPUT}")
    # CL3 goes below its value at the start, so that the restore brings it back from below.
finally:
    print("every value that the script changed -> its value at the start")
    try:
        # CL3 goes below its value at the start, so that the restore brings it back from below.
        print(f"CL3 -> {cl3_at_start - APPROACH_DAC}")
        set_value(monitor, "lens.CL3", cl3_at_start - APPROACH_DAC)
        leave_mode(monitor, mode, mode_values)
    finally:
        restore(monitor, start)
