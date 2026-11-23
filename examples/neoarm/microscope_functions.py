"""Functions that more than one microscope script needs.

`record_lens_changes.py` and `centre_beam.py` both find the beam in a SightX frame. The two
pivot-point scripts, `tilt_balance.py` and `shift_balance.py`, both wobble a deflector, set a
balance and centre the beam. These functions live here once (rule 5 of the pyjem skill). The file
has plain functions and no state. Each function that handles the microscope takes the monitor first.
"""

import time

import numpy as np
from jemwrap.control import read_value, set_value, take_frame

# The trial kicks of the centring deflector that measure how it moves the beam (centre_with). On
# the fitted model of notebook 2, 20 units of CLA1 move the beam about 20 px at X120k and 90 px at
# X500k (neoarm_response.json). The value is a choice.
CENTRING_TRIAL_DAC = 20
# A trial kick doubles while it moves the beam less than this, up to LARGEST_TRIAL_DAC. The values
# are choices.
SMALLEST_TRIAL_MOVE_PX = 2.0
LARGEST_TRIAL_DAC = 8192
# The most steps that set_balance takes toward one balance before it stops. On the simulated
# NEOARM, TiltBal y moves the beam about 2 px per unit at X60k. A step there is about 15 units.
# tilt_balance.py starts at X15k, where the steps are larger. The value is a choice.
BALANCE_STEPS = 150
# set_balance probes a balance with a change of BALANCE_PROBE_DAC. One change of a balance moves the
# beam by at most LARGEST_MOVE_PX. This limit keeps the whole beam on the SightX (half its width is
# 240 px). A change of a balance is also never more than LARGEST_BALANCE_DAC. At X500k the beam is
# wide, and a move of 60 px took it off the SightX. The values are choices.
BALANCE_PROBE_DAC = 10
LARGEST_MOVE_PX = 30
LARGEST_BALANCE_DAC = 2000
# set_balance stops, and sets the balance back, if it moves a balance further than this in all.
# The hidden pivot errors of the simulated NEOARM are at most 600 units. The value is a choice.
LARGEST_BALANCE_TOTAL_DAC = 8000
# set_balance leaves a balance alone where the wobble swings the beam less than this. find_beam
# places a beam to about 0.1 px on the simulated SightX (2026-11-22). The value is a choice.
SETTLED_SWING_PX = 0.5

# A frame holds a beam where its brightest pixel is this many noise widths above the median. A frame
# of the simulated SightX with no beam peaks 5 noise widths above the median. The cut for the beam
# pixels is also at least BEAM_CUT_NOISE noise widths. This cut keeps noise at the edge out of the
# beam.
SMALLEST_BEAM_NOISE = 15
BEAM_CUT_NOISE = 5
# The SightX frame at binning 4, in rows and columns, as the recordings of 2026-11-10 had it.
FRAME_SHAPE = (479, 479)
# The border of the frame where find_beam measures the background and the noise.
BORDER_PX = 10
# A whole beam has at least this many beam pixels per square pixel of its rms radius. A round beam
# has about 10 (a Gaussian, cut at a fifth of its peak) to 12. A beam that spreads over the whole
# frame gives scattered bright pixels, and these have far fewer (the rehearsal of 2026-11-20).
BEAM_PIXELS_PER_RMS2 = 3


def check_binning(monitor):
    """Stop the script unless the SightX gives frames of FRAME_SHAPE, its binning 4.

    Every pixel number of these scripts and of neoarm_response.json holds at binning 4.
    """
    shape = take_frame(monitor).shape
    if shape != FRAME_SHAPE:
        raise SystemExit(f"set the SightX binning to 4: the frames are {shape}, not {FRAME_SHAPE}")


def find_beam(frame):
    """Return the beam's centre (x, y) from the middle of the frame and its rms radius, in px.

    The rms radius is along one axis, as in the recordings of 2026-11-10. The beam is every pixel
    above the background by more than a fifth of the brightest pixel. A lower limit lets the noise
    of the background into the sums. A border of BORDER_PX pixels gives the background (its median)
    and the noise width.

    A frame can show a whole beam, light but no whole beam, or no light. With light but no whole
    beam, the beam touches the edge of the frame and the edge cuts it. Or the beam spreads so wide
    that only scattered pixels are bright. Then the centre is NaN and the rms radius is infinite.
    This result means that the beam is too large to measure. With no light, all three values are
    NaN. This result means that the beam is off the SightX.
    """
    border = np.concatenate(
        [
            frame[:BORDER_PX].ravel(),
            frame[-BORDER_PX:].ravel(),
            frame[:, :BORDER_PX].ravel(),
            frame[:, -BORDER_PX:].ravel(),
        ]
    )
    background = np.median(border)
    # 1.4826 times the median distance from the median is the standard deviation of normal noise.
    # A frame of whole counts with a flat border gives no width, so one count is the least.
    noise = max(1.4826 * np.median(np.abs(border - background)), 1.0)
    signal = frame.astype(float) - background
    # A hot pixel or an X-ray spike is one pixel, brighter than all four of its neighbours. It takes
    # the value of its brightest neighbour, so that it neither sets the cut nor counts as a beam.
    padded = np.pad(signal, 1, mode="edge")
    brightest_neighbour = np.maximum.reduce(
        [padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:]]
    )
    signal = np.minimum(signal, brightest_neighbour)
    cut = max(0.2 * signal.max(), BEAM_CUT_NOISE * noise)
    beam = np.where(signal > cut, signal, 0.0)
    if signal.max() < SMALLEST_BEAM_NOISE * noise:
        return np.nan, np.nan, np.nan
    rows, columns = np.indices(frame.shape)
    total = beam.sum()
    x = (columns * beam).sum() / total
    y = (rows * beam).sum() / total
    rms = np.sqrt((((columns - x) ** 2 + (rows - y) ** 2) * beam).sum() / total / 2)
    touches_edge = beam[[0, -1], :].any() or beam[:, [0, -1]].any()
    if touches_edge or np.count_nonzero(beam) < BEAM_PIXELS_PER_RMS2 * rms**2:
        return np.nan, np.nan, np.inf
    # The middle of a frame of 479 pixels is pixel 239.
    return x - (frame.shape[1] - 1) / 2, y - (frame.shape[0] - 1) / 2, rms


def wobble(monitor, deflector, axis, amplitude_dac, settle_s):
    """Wobble one axis of a deflector by +-amplitude_dac and return how far the beam swings (px).

    The function does the work of the console's wobbler step by step. The deflector goes to its
    value plus the amplitude, and the function takes a frame. Then the deflector goes to its value
    minus the amplitude, and the function takes a frame. Then the deflector goes back. The function
    returns the beam centre at plus minus the centre at minus, as (x, y). A frame without a whole
    beam stops the script.
    """
    # PyJEM gives the values as 16-bit integers. Python integers do not wrap below zero.
    value = [int(v) for v in read_value(monitor, f"deflectors.{deflector}")]
    centres = []
    for sign in (+1, -1):
        moved = list(value)
        moved["xy".index(axis)] += sign * amplitude_dac
        print(f"{deflector} -> {tuple(moved)}")
        set_value(monitor, f"deflectors.{deflector}", tuple(moved))
        time.sleep(settle_s)
        x, y, _ = find_beam(take_frame(monitor))
        if np.isnan(x):
            print(f"{deflector} -> {tuple(int(v) for v in value)}")
            set_value(monitor, f"deflectors.{deflector}", tuple(value))
            raise RuntimeError(f"the wobble of {deflector} {axis} took the beam off the SightX")
        centres.append((x, y))
    print(f"{deflector} -> {tuple(value)}")
    set_value(monitor, f"deflectors.{deflector}", tuple(value))
    time.sleep(settle_s)
    return np.subtract(centres[0], centres[1])


def set_balance(
    monitor, balance, axis, wobbled, centring, amplitude_dac, settle_s, centring_trial_dac=None
):
    """Set one axis of a balance where a wobble of ``wobbled`` moves the beam least.

    The swing of the beam is linear in the balance. The function wobbles at the balance's value and
    at a trial value. It finds where the straight line through the two swings comes closest to no
    swing. A change of a balance also moves the beam. The amount differs from axis to axis and from
    row to row. A small probe first measures this movement. Every change of the balance moves the
    beam by at most LARGEST_MOVE_PX. After each change, the deflector ``centring`` brings the beam
    back to the middle. Its first trial kick is ``centring_trial_dac``, with CENTRING_TRIAL_DAC as
    the default. In the same way, an operator turns a knob slowly and follows the beam. A balance
    whose wobble swings the beam less than SETTLED_SWING_PX stays as it is. The function returns
    the swing at the end, in px as (x, y).
    """
    index = "xy".index(axis)
    first = [int(v) for v in read_value(monitor, f"deflectors.{balance}")]
    for _ in range(BALANCE_STEPS):
        value = [int(v) for v in read_value(monitor, f"deflectors.{balance}")]
        if abs(value[index] - first[index]) > LARGEST_BALANCE_TOTAL_DAC:
            print(f"{balance} -> {tuple(first)}")
            set_value(monitor, f"deflectors.{balance}", tuple(first))
            raise RuntimeError(
                f"{balance} {axis} moved more than {LARGEST_BALANCE_TOTAL_DAC} units "
                "without settling"
            )
        # A probe of BALANCE_PROBE_DAC measures how far one unit of the balance moves the beam. The
        # probe halves while it takes the beam off the SightX.
        before = np.array(find_beam(take_frame(monitor))[:2])
        if np.isnan(before).any():
            raise RuntimeError(f"no whole beam on the SightX before the probe of {balance}")
        probe_dac = BALANCE_PROBE_DAC
        while True:
            probe = list(value)
            probe[index] += probe_dac
            print(f"{balance} -> {tuple(probe)} (probe)")
            set_value(monitor, f"deflectors.{balance}", tuple(probe))
            time.sleep(settle_s)
            moved = np.hypot(*(np.array(find_beam(take_frame(monitor))[:2]) - before))
            if not np.isnan(moved) or probe_dac == 1:
                break
            probe_dac //= 2
        print(f"{balance} -> {tuple(value)}")
        set_value(monitor, f"deflectors.{balance}", tuple(value))
        time.sleep(settle_s)
        if np.isnan(moved):
            raise RuntimeError(f"one unit of {balance} {axis} takes the beam off the SightX")
        largest_dac = max(
            1, min(LARGEST_BALANCE_DAC, int(LARGEST_MOVE_PX * probe_dac / max(moved, 1e-3)))
        )

        swing = wobble(monitor, wobbled, axis, amplitude_dac, settle_s)
        if np.hypot(*swing) < SETTLED_SWING_PX:
            # The wobble does not move the beam measurably. The balance stays as it is, since a
            # step from noise would only wander.
            return swing
        # The trial change halves while the wobble at it takes the beam off the SightX. A balance
        # that hardly moves the beam can still change the swing a lot.
        while True:
            trial = list(value)
            trial[index] += largest_dac
            print(f"{balance} -> {tuple(trial)}")
            set_value(monitor, f"deflectors.{balance}", tuple(trial))
            centre_with(monitor, centring, centring_trial_dac or CENTRING_TRIAL_DAC, settle_s)
            try:
                trial_swing = wobble(monitor, wobbled, axis, amplitude_dac, settle_s)
                break
            except RuntimeError:
                if largest_dac == 1:
                    raise
                largest_dac //= 2
                print(f"{balance} -> {tuple(value)}")
                set_value(monitor, f"deflectors.{balance}", tuple(value))
                centre_with(monitor, centring, centring_trial_dac or CENTRING_TRIAL_DAC, settle_s)
        swing_per_dac = (trial_swing - swing) / largest_dac
        if swing_per_dac @ swing_per_dac == 0:
            raise RuntimeError(f"{balance} {axis} does not change the swing of {wobbled}")
        step = -(swing @ swing_per_dac) / (swing_per_dac @ swing_per_dac)
        best = list(value)
        best[index] += int(round(np.clip(step, -largest_dac, largest_dac)))
        print(f"{balance} -> {tuple(best)}")
        set_value(monitor, f"deflectors.{balance}", tuple(best))
        centre_with(monitor, centring, centring_trial_dac or CENTRING_TRIAL_DAC, settle_s)
        if abs(step) <= largest_dac:
            return wobble(monitor, wobbled, axis, amplitude_dac, settle_s)
    raise RuntimeError(f"{balance} {axis} still moving after {BALANCE_STEPS} steps")


def centre_with(monitor, deflector, trial_dac, settle_s):
    """Bring the beam to the middle of the SightX with ``deflector``, from a measured response.

    Two trial kicks, along x and along y, give how far the deflector moves the beam. A trial kick
    starts at trial_dac. It halves while it takes the beam off the SightX, because the same kick
    moves the beam further at a higher magnification. It doubles while it moves the beam less than
    SMALLEST_TRIAL_MOVE_PX. This doubling keeps the measurement of a weak deflector above the noise.
    The deflector then changes by the amount that moves the beam to the middle. Because the function
    needs no model, it also works after a balance changes how the deflector acts.
    """
    value = np.array(read_value(monitor, f"deflectors.{deflector}"), dtype=int)
    start = np.array(find_beam(take_frame(monitor))[:2])
    if np.isnan(start).any():
        raise RuntimeError(f"no whole beam on the SightX before centring with {deflector}")
    response = []
    for direction in ((1, 0), (0, 1)):
        kick = trial_dac
        # A kick that once lost the beam does not grow again, so the search cannot go to and fro.
        lost_once = False
        while True:
            kicked = tuple(int(v) for v in value + kick * np.array(direction))
            print(f"{deflector} -> {kicked} (trial)")
            set_value(monitor, f"deflectors.{deflector}", kicked)
            time.sleep(settle_s)
            moved = np.array(find_beam(take_frame(monitor))[:2])
            if np.isnan(moved).any() and kick > 1:
                kick //= 2
                lost_once = True
            elif (
                not np.isnan(moved).any()
                and not lost_once
                and np.hypot(*(moved - start)) < SMALLEST_TRIAL_MOVE_PX
                and kick < LARGEST_TRIAL_DAC
            ):
                kick *= 2
            else:
                break
        if np.isnan(moved).any():
            print(f"{deflector} -> {tuple(int(v) for v in value)}")
            set_value(monitor, f"deflectors.{deflector}", tuple(int(v) for v in value))
            raise RuntimeError(f"one unit of {deflector} takes the beam off the SightX")
        response.append((moved - start) / kick)
    change = np.linalg.solve(np.array(response).T, -start)
    centred = tuple(int(round(v)) for v in value + change)
    print(f"{deflector} -> {centred}")
    set_value(monitor, f"deflectors.{deflector}", centred)
    time.sleep(settle_s)


# feature_shift removes the light's slow variation across the frame before it compares two frames.
# It subtracts the frame blurred over this many pixels (rms). Only the specimen's edges remain. The
# grating's period is 35 px at MAG X30k on the simulated NEOARM. The value is a choice.
ILLUMINATION_BLUR_PX = 20.0


def feature_shift(first, second):
    """Return how far the specimen's features moved from ``first`` to ``second`` (x, y), in px.

    The shift is the peak of the two frames' cross-correlation, to a fraction of a pixel. Before
    that, the function removes each frame's slow variation (the illumination). It also tapers each
    frame to zero at its edges. Neither the beam's own movement nor the frame's border sets the
    peak. The function mistakes a shift of more than half the specimen's period for a smaller one.
    """
    rows, columns = first.shape
    # A multiplication of the frame's Fourier transform blurs the frame with a Gaussian.
    ky = np.fft.fftfreq(rows)[:, None]
    kx = np.fft.rfftfreq(columns)[None, :]
    blur = np.exp(-2 * (np.pi * ILLUMINATION_BLUR_PX) ** 2 * (kx**2 + ky**2))
    taper = np.outer(np.hanning(rows), np.hanning(columns))
    spectra = []
    for frame in (first, second):
        frame = frame.astype(float)
        edges = frame - np.fft.irfft2(np.fft.rfft2(frame) * blur, frame.shape)
        spectra.append(np.fft.rfft2(edges * taper))
    correlation = np.fft.fftshift(np.fft.irfft2(np.conj(spectra[0]) * spectra[1], first.shape))
    row, column = np.unravel_index(np.argmax(correlation), correlation.shape)
    # A parabola through the peak and its two neighbours, along each axis, places the peak between
    # pixels.
    offsets = []
    for before, peak, after in (
        correlation[row - 1 : row + 2, column],
        correlation[row, column - 1 : column + 2],
    ):
        offsets.append(0.5 * (before - after) / (before - 2 * peak + after))
    return column + offsets[1] - columns // 2, row + offsets[0] - rows // 2


def enter_mode(monitor, mode, settle_s):
    """Switch to function mode ``mode`` (0 MAG, 4 DIFF) and return its values at entry.

    The console keeps a selector and a PLA value for MAG and others for DIFF. A snapshot holds
    only those of the mode at the start. The returned dict holds this mode's selector and PLA, and
    leave_mode sets them back with it.
    """
    set_value(monitor, "eos.function_mode", mode)
    time.sleep(settle_s)
    return {
        "eos.current_mag_selector_id": int(read_value(monitor, "eos.current_mag_selector_id")),
        "deflectors.PLA": tuple(int(v) for v in read_value(monitor, "deflectors.PLA")),
    }


def leave_mode(monitor, mode, at_entry):
    """Set the selector and PLA of function mode ``mode`` back to ``at_entry``, before a restore.

    The function acts only while the microscope is in that mode, since these values belong to the
    mode the microscope is in. ``at_entry`` None means that the script had not entered the mode, and
    then the function does nothing. A failure here must not stop the restore that follows, so the
    callers call restore in a finally clause.
    """
    if at_entry is None or int(read_value(monitor, "eos.function_mode")) != mode:
        return
    for key, value in at_entry.items():
        print(f"{key} -> {value}, this mode's value at the start")
        set_value(monitor, key, value)


# Every lens comes to a value from this far below, as in record_lens_changes.py. The fitted model
# assumes this history.
APPROACH_DAC = 2000


def set_lens_from_below(monitor, name, dac, settle_s):
    """Set lens ``name`` under free lens control to ``dac``, from APPROACH_DAC units below."""
    print(f"{name} -> {dac - APPROACH_DAC} -> {dac}")
    set_value(monitor, f"lens.{name}", int(dac) - APPROACH_DAC)
    set_value(monitor, f"lens.{name}", int(dac))
    time.sleep(settle_s)


def find_lost_beam(monitor, deflector, step_dac, most_rings, settle_s):
    """Bring a beam that is off the SightX back to its middle with ``deflector``.

    The deflector walks out from its value on square rings of trial values, ``step_dac`` apart.
    It goes ring after ring, up to ``most_rings`` rings. In the same way, an operator turns the
    knobs in widening circles.
    At the first trial value that shows a whole beam, centre_with takes the beam to the middle.
    The step must be smaller than the SightX in deflector units, so that no beam slips between
    two trial values. If no ring shows a beam, the deflector goes back and the function raises.
    """
    value = np.array(read_value(monitor, f"deflectors.{deflector}"), dtype=int)
    if not np.isnan(find_beam(take_frame(monitor))[0]):
        return
    for ring in range(1, most_rings + 1):
        for i in range(-ring, ring + 1):
            for j in range(-ring, ring + 1):
                if max(abs(i), abs(j)) != ring:
                    continue  # The search uses only the trial values on this ring.
                trial = tuple(int(v) for v in value + step_dac * np.array([i, j]))
                print(f"{deflector} -> {trial} (search)")
                set_value(monitor, f"deflectors.{deflector}", trial)
                time.sleep(settle_s)
                if not np.isnan(find_beam(take_frame(monitor))[0]):
                    centre_with(monitor, deflector, CENTRING_TRIAL_DAC, settle_s)
                    return
    print(f"{deflector} -> {tuple(int(v) for v in value)}")
    set_value(monitor, f"deflectors.{deflector}", tuple(int(v) for v in value))
    raise RuntimeError(f"no beam within {most_rings} rings of {deflector} trial values")
