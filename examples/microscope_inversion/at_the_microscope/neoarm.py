"""The NEOARM from a Python script: connect, read and set values, take frames, find the beam, keep a recording.

Copy the folder ``at_the_microscope/`` to the microscope PC. This file needs numpy and PyJEM only (TemGym only for
the simulated NEOARM, matplotlib only to show frames)::

    scope = Neoarm("my_experiment", simulated=True)     # or for_real=True at the microscope; neither: a dry run
    scope.ask(scope.eos.GetMagValue)                      # read: the call, then a short pause
    scope.change(scope.deflectors.SetCLA1, x, y)          # set: in a dry run only printed
    beam = scope.record(role="not deflected")             # a frame, and every lens and deflector value: one row
    scope.restore()                                       # free-lens control off, deflectors back
    scope.save()                                          # table.csv, console.json, frames/ and a copy of the scripts

Every row of a recording has the same columns, whatever script made it: what the frame is for, the function mode,
magnification or camera length, high tension, spot size, every lens, CLA1, CLA2 and PLA, the condenser aperture's
hole, the time, and where the beam is in the frame (found by :func:`find_beam`).
"""

import atexit
import csv
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# The real NEOARM, as used on 2026-11-10
CAMERA_IP = "172.17.41.1"                        # the camera PC
CAMERA_NAME = "screenCamera"                     # the SightX camera, by PyJEM's name for it
FRAME_SHAPE = (1916 // 4, 1916 // 4)             # pixels (rows, columns) at binning 4
FRAME_DTYPE = np.uint16
LENS_SETTLE_S = 2                                # wait after changing a lens
MODE_SETTLE_S = 5                                # wait after changing the function mode or the magnification
PAUSE_S = 0.1                                    # after every call: too many calls at once can crash the microscope

DAC_MIN, DAC_MAX = 0, 65535                      # lenses and deflectors
FREE_LENS = {"CL1": 0, "CL2": 1, "CL3": 2, "CM": 3, "OLc": 6, "OLf": 7, "OM": 8, "OM2": 9,
             "IL1": 10, "IL2": 11, "IL3": 12, "IL4": 13, "PL1": 14, "PL2": 15, "PL3": 16}   # PyJEM's free-lens numbers
DEFLECTORS = ["CLA1", "CLA2", "PLA"]


class Neoarm:
    """One session at the NEOARM (or the simulated one), and the recording it makes.

    Parameters
    ----------
    name : str
        The recording's name: frames and table go to ``output/<name>/<date_time>/`` next to the script.
    for_real : bool
        False (the default): a dry run. The microscope is read and frames are taken, but nothing is changed (each
        change is printed) and nothing is saved.
    simulated : bool
        The virtual NEOARM (TemGym) instead of the real one; always changed for real.
    magnification, camera_length_m : float
        Simulated only: the MAG row and the DIFF camera length the console starts on.
    show_frames : bool
        Show every frame, with the beam marked (needs matplotlib).
    """

    def __init__(self, name, *, for_real=False, simulated=False, magnification=40000, camera_length_m=0.4,
                 show_frames=False):
        self.simulated, self.show_frames = simulated, show_frames
        self.dry_run = not (for_real or simulated)
        if simulated:
            import os
            os.environ.setdefault("JAX_PLATFORMS", "cpu")
            import temgym_core
            from temgym_core.pyjem_simulator import SimulatedMicroscope
            temgym_core.enable_x64()
            self.microscope = SimulatedMicroscope.from_file(
                HERE.parents[1] / "virtual_neoarm/virtual_neoarm.json", ht_v=200e3, magnification=magnification,
                diffraction_camera_length_m=camera_length_m,
                condenser_path=HERE.parents[1] / "virtual_neoarm/condenser.json",
                ray_sample_power=20, detector_samples_per_pixel=2)
            TEM3, detector = self.microscope.pyjem()
            self.camera = detector.Detector(detector.get_attached_detector()[0])
            settings = self.camera.get_detectorsetting()
            self.frame_shape = (settings["ImagingArea"]["Height"], settings["ImagingArea"]["Width"])
            self.frame_dtype = np.dtype("<i4" if settings["OutputImageInformation"]["DataBits"] == 32 else "<i2")
            self.lens_settle_s = self.mode_settle_s = self.pause_s = 0
        else:
            from PyJEM import TEM3, sightx
            sightx.set_ip(CAMERA_IP)
            self.camera = sightx.Detector(CAMERA_NAME)
            self.frame_shape, self.frame_dtype = FRAME_SHAPE, np.dtype(FRAME_DTYPE)
            self.lens_settle_s, self.mode_settle_s, self.pause_s = LENS_SETTLE_S, MODE_SETTLE_S, PAUSE_S
        self.lens, self.deflectors, self.eos, self.ht = TEM3.Lens3(), TEM3.Def3(), TEM3.EOS3(), TEM3.HT3()
        self.apertures = TEM3.Apt3()
        self.name = name
        self.folder = None                       # made at the first saved frame
        self.rows = []                           # the table: one row per saved frame
        self.console = {}                        # what the console had, by label: saved to console.json
        self.deflectors_at_start = {}            # what restore() puts back
        self.taken_over = []                     # lenses under free-lens control
        self.dark_level = None                   # the camera's value with the beam blanked (until measured: estimated)
        self.last_change = "(nothing yet)"
        print({True: "DRY RUN: nothing will be changed or saved", False: "FOR REAL: the microscope will be changed"}
              [self.dry_run], "(simulated NEOARM)" if simulated else "(the NEOARM)")

    # ------------------------------------------------------------------------------------- reading and setting
    def ask(self, call, *arguments):
        """Read something: ``call(*arguments)``, then a short pause."""
        answer = call(*arguments)
        time.sleep(self.pause_s)
        return answer

    def change(self, call, *arguments):
        """Set something: ``call(*arguments)``, then a short pause. In a dry run it is only printed."""
        self.last_change = f"{call.__name__}{arguments}"
        if self.dry_run:
            print(f"   would call {call.__name__}{arguments}")
            return
        call(*arguments)
        time.sleep(self.pause_s)

    def wait(self, seconds):
        """Wait for the microscope to settle (not in a dry run, nor on the simulated NEOARM)."""
        if not self.dry_run and not self.simulated:
            time.sleep(seconds)

    def set_lenses(self, **values):
        """Lenses to DAC values by free-lens control, e.g. ``set_lenses(IL1=41782, PL1=60000)``, then wait."""
        for name, value in values.items():
            if not DAC_MIN <= value <= DAC_MAX:
                raise ValueError(f"{name} = {value} is outside {DAC_MIN} to {DAC_MAX}")
            self.change(self.lens.SetFLCAbs, FREE_LENS[name], int(value))
        self.wait(self.lens_settle_s)

    def take_over(self, *names):
        """Free-lens control of these lenses, at the values the console has now."""
        for name in names:
            value = self.ask(getattr(self.lens, "Get" + name))
            self.change(self.lens.SetFLCAbs, FREE_LENS[name], value)
            self.change(self.lens.SetFLCSw, FREE_LENS[name], 1)
            if name not in self.taken_over:
                self.taken_over.append(name)

    def release(self, *names):
        """Free-lens control off for these lenses: they go back to the console's values."""
        for name in names:
            self.change(self.lens.SetFLCSw, FREE_LENS[name], 0)
            if name in self.taken_over:
                self.taken_over.remove(name)

    def set_deflector(self, name, x, y):
        """CLA1, CLA2 or PLA to (x, y)."""
        x, y = int(round(x)), int(round(y))
        if not (DAC_MIN <= x <= DAC_MAX and DAC_MIN <= y <= DAC_MAX):
            raise ValueError(f"{name} = ({x}, {y}) is outside {DAC_MIN} to {DAC_MAX}")
        self.change(getattr(self.deflectors, "Set" + name), x, y)

    def read_state(self):
        """Function mode, magnification or camera length, high tension, spot size, every lens and CLA1, CLA2, PLA,
        as the microscope has them now (a lens this PyJEM cannot read is left out)."""
        mode, magnification = self.ask(self.eos.GetFunctionMode), self.ask(self.eos.GetMagValue)
        state = dict(function_mode=mode[1], magnification=magnification[0], magnification_unit=magnification[1],
                     ht_v=self.ask(self.ht.GetHtValue), spot=self.ask(self.eos.GetSpotSize))
        for name in FREE_LENS:
            if hasattr(self.lens, "Get" + name):
                state[name] = self.ask(getattr(self.lens, "Get" + name))
        for name in DEFLECTORS:
            state[name + "_x"], state[name + "_y"] = self.ask(getattr(self.deflectors, "Get" + name))
        try:
            state["condenser_hole"] = self.ask(self.apertures.GetSize, 1)       # 1: the condenser aperture; 0: none in
        except Exception:                                                        # this PyJEM cannot read it
            pass
        return state

    def remember_console(self, label):
        """Note what the console has now under ``label`` (saved to console.json); CLA1, CLA2 and PLA as they are now
        become what :meth:`restore` puts back. Returns the state."""
        state = self.read_state()
        self.console[label] = state
        self.deflectors_at_start = {name: (state[name + "_x"], state[name + "_y"]) for name in DEFLECTORS}
        return state

    # ------------------------------------------------------------------------------------------------- frames
    def take_frame(self):
        """One camera frame, as an array of integers."""
        raw = self.ask(self.camera.snapshot_rawdata)
        return np.frombuffer(raw, dtype=self.frame_dtype).reshape(self.frame_shape)

    def measure(self, title=""):
        """Take a frame and find the beam in it (nothing saved). Returns the beam (:class:`Beam`)."""
        frame = self.take_frame()
        beam = find_beam(frame=frame, background=self.dark_level)
        if self.show_frames:
            show_frame(frame, beams=[("beam", beam)], title=f"{title}  |  after {self.last_change}\n{describe(beam)}")
        return beam

    def measure_dark_level(self):
        """The camera's value with the beam blanked: the median of one blanked frame."""
        self.change(self.deflectors.SetBeamBlank, 1)
        self.wait(0.5)
        frame = self.take_frame()
        self.change(self.deflectors.SetBeamBlank, 0)
        self.dark_level = None if self.dry_run else float(np.median(frame))      # a dry run did not blank
        print("dark level:", self.dark_level)

    def record(self, **about):
        """Take a frame and keep it: the table gets a row of ``about`` (what the frame is for), the state of the
        microscope now, and where the beam is. Returns the beam."""
        row = dict(about, **self.read_state())
        frame = self.take_frame()
        beam = find_beam(frame=frame, background=self.dark_level)
        row.update(beam_x_px=round(beam.centre.real, 3), beam_y_px=round(beam.centre.imag, 3), beam_rms_px=round(beam.rms, 3),
                   beam_roundness=round(beam.roundness, 4), beam_signal=round(beam.signal, 1), beam_usable=beam.usable,
                   time_s=round(time.time(), 2))
        if not self.dry_run:
            if self.folder is None:
                script = Path(sys.argv[0]).resolve()
                base = script.parent if script.suffix == ".py" else Path.cwd()    # next to the script; in a notebook here
                self.folder = base / "output" / self.name / time.strftime("%Y-%m-%d_%H%M%S")
                (self.folder / "frames").mkdir(parents=True)
                print("recording into", self.folder)
            row["frame"] = f"frames/{len(self.rows):04d}.npy"
            np.save(self.folder / row["frame"], frame)
        self.rows.append(row)
        if self.show_frames:
            show_frame(frame, beams=[("beam", beam)], title=f"{len(self.rows)}. {about}\n{describe(beam)}")
        return beam

    # ------------------------------------------------------------------------------------ restoring and saving
    def restore(self):
        """Free-lens control off for the lenses taken over, CLA1, CLA2 and PLA back where the console had them, and
        the beam unblanked. Safe to run at any time."""
        for name in self.taken_over:
            self.change(self.lens.SetFLCSw, FREE_LENS[name], 0)
        self.taken_over = []
        for name, (x, y) in self.deflectors_at_start.items():
            self.change(getattr(self.deflectors, "Set" + name), x, y)
        self.change(self.deflectors.SetBeamBlank, 0)

    def restore_if_anything_fails(self):
        """In a notebook: restore after any cell that fails or is stopped. As a script: restore and save however
        the script ends."""
        try:
            shell = get_ipython()                                               # noqa: F821
        except NameError:
            atexit.register(lambda: (self.restore(), self.save()))
            return
        shell.events.register("post_run_cell", lambda result: result.error_in_exec is None or self.restore())

    def save(self):
        """table.csv (one row per frame), console.json (what the console had), and a copy of the scripts used."""
        if self.dry_run or not self.rows or self.folder is None:
            print(f"{len(self.rows)} rows; not saved ({'a dry run' if self.dry_run else 'nothing recorded'})")
            return
        columns = []
        for row in self.rows:
            columns += [key for key in row if key not in columns]
        with open(self.folder / "table.csv", "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.rows)
        (self.folder / "console.json").write_text(json.dumps(self.console, indent=1, default=str))
        for script in {Path(__file__).resolve(), Path(sys.argv[0]).resolve()}:
            if script.suffix == ".py" and script.exists():
                shutil.copy(script, self.folder / script.name)
        print(f"{len(self.rows)} rows in {self.folder / 'table.csv'}")


def describe(beam):
    """One line about a beam: centre, size, roundness and any problem."""
    return (f"beam at ({beam.centre.real:+.1f}, {beam.centre.imag:+.1f}) px, rms {beam.rms:.1f} px, "
            f"roundness {beam.roundness:.2f}" + ("  [" + "; ".join(beam.problems) + "]" if beam.problems else ""))


# ----------------------------------------------------------------------------------------------- finding the beam



@dataclass
class Beam:
    """The beam found in a frame by :func:`find_beam`. Positions in px from the frame centre, x + iy."""

    found: bool                   # a region clearly above the background
    centre: complex               # intensity-weighted centre of the beam's region
    rms: float                    # rms radius (px)
    axes: tuple                   # rms along its long and short axes (px), and the long axis' angle (rad)
    signal: float                 # total signal above the background in the region
    area: int                     # pixels in the region
    touches_edge: bool            # the region reaches the frame edge: the beam is (partly) off the camera
    saturated: int                # pixels at the camera's highest value
    others: float                 # signal of the next-brightest region, as a fraction of the beam's
    problems: list = field(default_factory=list)

    @property
    def roundness(self):
        """Short axis / long axis of the beam's region (the same ellipse ``regionprops`` reports): 1 for a round beam."""
        return self.axes[1] / self.axes[0] if self.axes[0] > 0 else 0.0

    @property
    def usable(self):
        """Fit for measuring its centre: found, wholly on the camera, and not one of two similar regions."""
        return self.found and not self.touches_edge and self.others < 0.5


def _box_smooth(image, half):
    """Mean over a (2 half + 1)^2 box, by cumulative sums (edges: the box shrinks)."""
    out = image
    for axis in (0, 1):
        c = np.cumsum(np.insert(out, 0, 0.0, axis=axis), axis=axis)
        n = out.shape[axis]
        lo = np.clip(np.arange(n) - half, 0, n)
        hi = np.clip(np.arange(n) + half + 1, 0, n)
        out = (np.take(c, hi, axis=axis) - np.take(c, lo, axis=axis)) / np.expand_dims(hi - lo, 1 - axis)
    return out


def _grow(mask, steps):
    """The mask grown by ``steps`` pixels (8-connected)."""
    for _ in range(steps):
        m = mask.copy()
        m[1:] |= mask[:-1]; m[:-1] |= mask[1:]; m[:, 1:] |= mask[:, :-1]; m[:, :-1] |= mask[:, 1:]
        m[1:, 1:] |= mask[:-1, :-1]; m[:-1, :-1] |= mask[1:, 1:]; m[1:, :-1] |= mask[:-1, 1:]; m[:-1, 1:] |= mask[1:, :-1]
        mask = m
    return mask


def _label(mask):
    """Connected regions of a mask (8-connected): an integer image, 0 outside, and their number."""
    try:
        from scipy.ndimage import label
        return label(mask, structure=np.ones((3, 3)))
    except ImportError:                                  # numpy only: spread the largest label until stable
        labels = np.where(mask, np.arange(1, mask.size + 1).reshape(mask.shape), 0)
        while True:
            spread = labels.copy()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    spread = np.maximum(spread, np.roll(np.roll(labels, dy, 0), dx, 1) * mask)
            spread *= mask
            if np.array_equal(spread, labels):
                break
            labels = spread
        values, labels = np.unique(labels, return_inverse=True)
        return labels.reshape(mask.shape), len(values) - 1


_image_slot = {"cell": None, "handle": None}      # the image of the running notebook cell, replaced by the next frame


def show_frame(frame, *, beams=(), title=""):
    """Draw one whole frame (square root of the signal above its median, to show faint parts) and mark
    the beam centres in ``beams``, a list of (label, Beam from :func:`find_beam`). In a notebook, a new frame
    replaces the previous one of the same cell. Needs matplotlib (imported only here)."""
    import matplotlib.pyplot as plt

    if frame is None:
        return print(f"{title}: no frame (dry run)")
    frame = np.asarray(frame, dtype=float)
    centre_of_frame = (frame.shape[1] - 1) / 2 + 1j * (frame.shape[0] - 1) / 2
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.imshow(np.sqrt(np.clip(frame - np.median(frame), 0, None)), origin="lower", cmap="gray")
    found = []
    colours = ["#F0E442", "#D55E00", "#56B4E9", "#009E73"]
    for (label, beam), colour in zip(beams, colours):
        if beam.found:
            position = beam.centre + centre_of_frame
            ax.plot(position.real, position.imag, "+", color=colour, ms=14, mew=2, label=label)
            found.append(label)
    if found:
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="normal")
    ax.set(xlabel="x (px)", ylabel="y (px)")
    try:
        from IPython import get_ipython
        from IPython.display import display
        cell = get_ipython().execution_count             # None outside a notebook or IPython
    except (ImportError, AttributeError):
        cell = None
    if cell is None:
        plt.show()
        return
    plt.close(fig)                                       # shown by hand below, not by the inline backend
    if _image_slot["cell"] == cell:
        _image_slot["handle"].update(fig)                # the same cell: replace its image
    else:
        _image_slot.update(cell=cell, handle=display(fig, display_id=True))     # a new cell: a new image


def find_beam(*, frame, background=None, saturation=None):
    """Find the beam in a camera frame by segmentation, and measure it.

    The frame is smoothed a little and thresholded just above the background (the larger of six
    times the smoothed noise and 2 % of the brightest point); bright pixels a few px apart are
    joined, so a caustic or a broken-up beam stays one region. The beam is the region with the most
    signal; it is grown by 3 px to take in its faint edge, and its centre and size are measured
    inside it only, so stray light elsewhere does not pull it. Frames where that is not reliable say
    why (:attr:`Beam.problems`): no beam, the beam reaching the frame edge, a second region nearly
    as bright, saturated pixels.

    Parameters
    ----------
    frame : array_like
        The camera frame.
    background : float or array_like, optional
        The dark level: a number (e.g. the median of a frame with the beam blanked) or a dark
        frame. Without it, the median of the dimmest quarter of the frame.
    saturation : float, optional
        The camera's highest value; by default the largest value of an integer frame's type.

    Returns
    -------
    Beam
    """
    raw = np.asarray(frame)
    f = raw.astype(float)
    if background is None:
        background = float(np.median(np.sort(f, axis=None)[: f.size // 4]))
    f = f - background
    step = (f[:, 1:] - f[:, :-1]).ravel()                    # neighbour differences: the beam is smooth, noise is not
    noise = 1.4826 * float(np.median(np.abs(step - np.median(step)))) / np.sqrt(2) or 1.0
    smooth = _box_smooth(f, 2)                               # 5 x 5 px: noise down 5 times
    threshold = max(6 * noise / 5, 0.02 * smooth.max())
    if saturation is None and np.issubdtype(raw.dtype, np.integer):
        saturation = np.iinfo(raw.dtype).max
    saturated = int(np.sum(raw >= saturation)) if saturation is not None else 0
    nothing = Beam(False, 0j, 0.0, (0.0, 0.0, 0.0), 0.0, 0, False, saturated, 0.0)
    bright = smooth > threshold
    if smooth.max() <= 6 * noise / 5 or not bright.any():
        nothing.problems.append("no beam on the camera")
        return nothing
    labels, n = _label(_grow(bright, 3))                     # fragments within ~6 px join
    signals = np.bincount(labels.ravel(), weights=np.clip(f, 0, None).ravel(), minlength=n + 1)[1:]
    order = np.argsort(signals)[::-1]
    region = _grow(bright & (labels == order[0] + 1), 3)
    others = float(signals[order[1]] / signals[order[0]]) if n > 1 else 0.0
    w = np.where(region, np.clip(f, 0, None), 0.0)
    total = w.sum()
    ny, nx = f.shape
    y, x = np.mgrid[0:ny, 0:nx]
    y, x = y - (ny - 1) / 2, x - (nx - 1) / 2
    cx, cy = (w * x).sum() / total, (w * y).sum() / total
    cov = np.array([[(w * (x - cx) ** 2).sum(), (w * (x - cx) * (y - cy)).sum()],
                    [(w * (x - cx) * (y - cy)).sum(), (w * (y - cy) ** 2).sum()]]) / total
    values, vectors = np.linalg.eigh(cov)
    long_axis = float(np.arctan2(vectors[1, 1], vectors[0, 1]))
    touches = bool(region[0].any() or region[-1].any() or region[:, 0].any() or region[:, -1].any())
    beam = Beam(True, complex(cx, cy), float(np.sqrt(np.trace(cov) / 2)),
                (float(np.sqrt(values[1])), float(np.sqrt(max(values[0], 0.0))), long_axis),
                float(total), int(region.sum()), touches, saturated, others)
    if touches:
        beam.problems.append("the beam reaches the frame edge: part of it is off the camera")
    if others >= 0.05:
        beam.problems.append(f"another bright region, {others:.0%} of the beam's signal")
    if saturated:
        beam.problems.append(f"{saturated} saturated pixels: shorten the exposure or spread the beam")
    return beam
