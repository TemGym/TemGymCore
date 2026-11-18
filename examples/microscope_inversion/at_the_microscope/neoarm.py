"""The NEOARM from a Python script: connect, read and set values, take frames, find the beam, keep a recording.

Copy the folder ``at_the_microscope/`` to the microscope PC. This file needs numpy and PyJEM only (TemGym only for
the simulated NEOARM, matplotlib only to show frames)::

    scope = Neoarm("my_experiment", simulated=True)     # or for_real=True at the microscope; neither: a dry run
    scope.ask(scope.eos.GetMagValue)                      # read: the call, then a short pause
    scope.change(scope.deflectors.SetCLA1, x, y)          # set: in a dry run only printed
    beam = scope.record(role="not deflected")             # a frame, and every lens and deflector value: one row
    scope.add_camera("merlin"); scope.use_camera("merlin")   # a second camera: frames now come from it
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

# The screen: which camera the beam reaches is a PyJEM call too (as JEOL gave it: detect3 getScreen / setScreen).
SCREEN_LOWER, SCREEN_VIEW, SCREEN_BOTTOM = 0, 1, 2   # the lower screen; the view camera (the SightX, "sight cross"); the bottom camera (the Merlin)
SCREEN_SETTLE_S = 4                              # wait after moving the screen
BLANK_SETTLE_S = 4                               # wait after blanking or unblanking the beam (dark level)
SCREEN_CLASS, SCREEN_GET, SCREEN_SET = "Detector3", "getScreen", "setScreen"   # PyJEM's names: if they differ, change them here

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
    full_readback : bool
        Also record the alignment pairs, alpha selector and stage used by the predictor calibration.
    """

    def __init__(self, name, *, for_real=False, simulated=False, magnification=40000, camera_length_m=0.4,
                 show_frames=False, full_readback=False):
        self.simulated, self.show_frames = simulated, show_frames
        self.full_readback = full_readback
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
            self.detector_module = detector
            self.camera = detector.Detector(detector.get_attached_detector()[0])
            settings = self.camera.get_detectorsetting()
            self.frame_shape = (settings["ImagingArea"]["Height"], settings["ImagingArea"]["Width"])
            self.frame_dtype = np.dtype("<i4" if settings["OutputImageInformation"]["DataBits"] == 32 else "<i2")
            self.lens_settle_s = self.mode_settle_s = self.pause_s = 0
        else:
            from PyJEM import TEM3, sightx
            sightx.set_ip(CAMERA_IP)
            self.detector_module = sightx
            self.camera = sightx.Detector(CAMERA_NAME)
            self.frame_shape, self.frame_dtype = FRAME_SHAPE, np.dtype(FRAME_DTYPE)
            self.lens_settle_s, self.mode_settle_s, self.pause_s = LENS_SETTLE_S, MODE_SETTLE_S, PAUSE_S
        self.tem3, self.screen_unit, self.screen_now, self.screen_at_start = TEM3, None, None, None
        self.lens, self.deflectors, self.eos, self.ht = TEM3.Lens3(), TEM3.Def3(), TEM3.EOS3(), TEM3.HT3()
        self.apertures = TEM3.Apt3()
        self.name = name
        self.folder = None                       # made at the first saved frame
        self.rows = []                           # the table: one row per saved frame
        self.console = {}                        # what the console had, by label: saved to console.json
        self.deflectors_at_start = {}            # what restore() puts back
        self.taken_over = []                     # lenses under free-lens control
        self.dark_level = None                   # the camera's value with the beam blanked (until measured: estimated)
        self.exposure_count = 0                  # includes dark, sizing, centring and trial pulses
        self.last_change = "(nothing yet)"
        self.camera_name = None                  # set by use_camera(); None: the one camera this session started with
        self.cameras = {}                        # more cameras, by name: see add_camera()
        if simulated:
            print("Simulated NEOARM")
        else:
            print("DRY RUN: nothing will be changed or saved" if self.dry_run else "FOR REAL: the microscope will be changed",
                  "(the NEOARM)")

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
        pairs = list(DEFLECTORS)
        if self.full_readback:
            pairs.extend(("IS1", "IS2", "SpotA", "OLs", "ILs", "CLs", "GunA1", "GunA2", "ShifBal", "TiltBal", "AngBal"))
        for name in pairs:
            getter = getattr(self.deflectors, "Get" + name, None)
            if getter is not None:
                state[name + "_x"], state[name + "_y"] = self.ask(getter)
        if self.full_readback:
            state["alpha"] = self.ask(self.eos.GetAlpha)
            state["stage_position"] = self.ask(self.tem3.Stage3().GetPos)
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
        """One camera frame, as an array of integers. Checks the screen first (see :meth:`check_screen`)."""
        self.check_screen()
        reader = self.cameras[self.camera_name]["reader"] if self.cameras else None
        if reader is not None:                   # a camera PyJEM does not know
            frame = np.asarray(self.ask(reader.take_frame))
        else:
            raw = self.ask(self.camera.snapshot_rawdata)
            frame = np.frombuffer(raw, dtype=self.frame_dtype).reshape(self.frame_shape)
        self.exposure_count += 1
        return frame

    def measure(self, title=""):
        """Take a frame and find the beam in it (nothing saved). Returns the beam (:class:`Beam`)."""
        frame = self.take_frame()
        beam = find_beam(frame=frame, background=self.dark_level)
        if self.show_frames:
            show_frame(frame, beams=[("beam", beam)], title=f"{title}  |  after {self.last_change}\n{describe(beam)}")
        return beam

    def measure_dark_level(self):
        """The camera's value with the beam blanked: the median of one frame taken BLANK_SETTLE_S after blanking
        (then BLANK_SETTLE_S again after unblanking, before anything else is read)."""
        original_blank = self.ask(self.deflectors.GetBeamBlank)
        try:
            self.change(self.deflectors.SetBeamBlank, 1)
            self.wait(BLANK_SETTLE_S)
            frame = self.take_frame()
        finally:
            self.change(self.deflectors.SetBeamBlank, original_blank)
            self.wait(BLANK_SETTLE_S)
        self.dark_level = None if self.dry_run else float(np.median(frame))      # a dry run did not blank
        print("dark level:", self.dark_level)

    # ------------------------------------------------------------------------------------------- more cameras
    # ------------------------------------------------------------------------------------- the screen
    def _screen(self):
        """PyJEM's object with getScreen / setScreen (made at the first use; the simulated NEOARM has one too)."""
        if self.screen_unit is None:
            try:
                self.screen_unit = getattr(self.tem3, SCREEN_CLASS)()
                getattr(self.screen_unit, SCREEN_GET), getattr(self.screen_unit, SCREEN_SET)
            except AttributeError as error:
                raise RuntimeError(f"PyJEM's TEM3 has no {SCREEN_CLASS}().{SCREEN_GET} / {SCREEN_SET} ({error}): "
                                   "put the right names in SCREEN_CLASS, SCREEN_GET, SCREEN_SET at the top of neoarm.py")
        return self.screen_unit

    def get_screen(self):
        """Where the screen is now: 0 lower screen, 1 view camera (SightX), 2 bottom camera (Merlin)."""
        self.screen_now = int(self.ask(getattr(self._screen(), SCREEN_GET)))
        if self.screen_at_start is None:
            self.screen_at_start = self.screen_now
        return self.screen_now

    def set_screen(self, position):
        """Move the screen to 0, 1 or 2, wait SCREEN_SETTLE_S, then read it back: stops if it is not there.
        Always asks the microscope where the screen is first (never trusts what it was last time)."""
        if position not in (SCREEN_LOWER, SCREEN_VIEW, SCREEN_BOTTOM):
            raise ValueError(f"screen position {position}: it must be 0, 1 or 2")
        if self.dry_run and self.screen_now is not None:
            if position != self.screen_now:          # a dry run moved nothing: remember what it would have done
                self.change(getattr(self._screen(), SCREEN_SET), position)
                self.screen_now = position
            return
        if self.get_screen() == position:
            return
        self.change(getattr(self._screen(), SCREEN_SET), position)
        self.wait(SCREEN_SETTLE_S)
        for _ in range(10):                          # still moving: up to 10 s more, asking once a second
            if self.dry_run or self.get_screen() == position:
                break
            self.wait(1)
        else:
            raise RuntimeError(f"asked for screen {position}, but {SCREEN_GET} still says {self.screen_now} after "
                               f"{SCREEN_SETTLE_S + 10} s: the beam is not on the camera frames would be read from. Stopped.")
        if not self.dry_run:
            print(f"   screen now at {position} ({SCREEN_GET} says so)")
        self.screen_now = position

    def check_screen(self):
        """Before a frame from a camera that has a screen position: ask the microscope where the screen is. If it is
        somewhere else (someone moved it, or a call did nothing), stop rather than save a frame of nothing."""
        entry = self.cameras.get(self.camera_name) if self.cameras else None
        if self.dry_run or not entry or entry.get("screen") is None:
            return
        if self.get_screen() != entry["screen"]:
            raise RuntimeError(f"a frame from the {self.camera_name} needs screen {entry['screen']}, but "
                               f"{SCREEN_GET} says {self.screen_now}. Stopped: no frame taken.")

    # ------------------------------------------------------------------------------------- cameras
    def attached_cameras(self):
        """The names of the cameras PyJEM can read."""
        return list(self.ask(self.detector_module.get_attached_detector))

    def add_camera(self, name, reader=None, above=(), screen=None):
        """Make a camera known by ``name``, so that :meth:`use_camera` can switch to it.

        Without ``reader`` it is the camera PyJEM calls ``name``. Otherwise ``reader`` is any object with
        ``take_frame()`` (a 2-D array of integers) and, if it can, ``insert()`` and ``retract()``: a camera PyJEM
        does not know. ``above``: the names of the cameras that sit above it and block it while they are in.
        ``screen``: the screen position (0, 1, 2) that brings the beam to this camera: :meth:`use_camera` then moves
        the screen by itself instead of asking you to change the camera by hand.
        """
        if not self.cameras:                     # the camera the session started with keeps its place
            first = self.camera_name or getattr(self.camera, "name", CAMERA_NAME)
            self.cameras[first] = dict(detector=self.camera, reader=None, frame_shape=self.frame_shape,
                                       frame_dtype=self.frame_dtype, dark_level=self.dark_level, above=(), screen=None)
            self.camera_name = first
        if name in self.cameras:
            return
        entry = dict(detector=None, reader=reader, frame_shape=None, frame_dtype=None, dark_level=None, above=tuple(above), screen=screen)
        if reader is None:
            entry["detector"] = self.detector_module.Detector(name)
            settings = self.ask(entry["detector"].get_detectorsetting)
            entry["frame_shape"] = (settings["ImagingArea"]["Height"], settings["ImagingArea"]["Width"])
            entry["frame_dtype"] = np.dtype("<i4" if settings["OutputImageInformation"]["DataBits"] == 32 else "<u2")
        self.cameras[name] = entry

    def use_camera(self, name, by_hand=False):
        """Frames now come from this camera. The cameras above it are taken out of the beam and it is put in
        (``by_hand``: the script waits while you do that at the microscope). Nothing else is changed."""
        if name == self.camera_name:
            return
        self.cameras[self.camera_name]["dark_level"] = self.dark_level
        entry = self.cameras[name]
        if entry.get("screen") is not None:          # the screen brings the beam to this camera: nothing by hand
            self.set_screen(entry["screen"])
        elif by_hand:
            if not self.dry_run:
                input(f"\n>>> Make the beam reach the {name} (take out: {', '.join(entry['above']) or 'nothing'}), "
                      "then press Enter ")
        else:
            for other in entry["above"]:
                mover = self.cameras[other]["reader"] or self.cameras[other]["detector"]
                if hasattr(mover, "retract"):
                    self.change(mover.retract)
                elif not self.dry_run:                       # this camera cannot be moved from here
                    input(f"\n>>> Take the {other} out of the beam, then press Enter ")
            mover = entry["reader"] or entry["detector"]
            if hasattr(mover, "insert"):
                self.change(mover.insert)
            self.wait(self.mode_settle_s)
        self.camera_name, self.camera = name, entry["detector"]
        self.frame_shape, self.frame_dtype, self.dark_level = entry["frame_shape"], entry["frame_dtype"], entry["dark_level"]

    def record(self, **about):
        """Take a frame and keep it: the table gets a row of ``about`` (what the frame is for), the state of the
        microscope now, and where the beam is. Returns the beam."""
        row = dict(about, **self.read_state())
        if self.cameras:
            row["camera"] = self.camera_name
        frame = self.take_frame()
        beam = find_beam(frame=frame, background=self.dark_level)
        row.update(beam_x_px=round(beam.centre.real, 3), beam_y_px=round(beam.centre.imag, 3), beam_rms_px=round(beam.rms, 3),
                   beam_roundness=round(beam.roundness, 4), beam_signal=round(beam.signal, 1), beam_usable=beam.usable,
                   beam_saturated=beam.saturated, beam_touches_edge=beam.touches_edge,
                   dark_level=self.dark_level, time_s=round(time.time(), 2))
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
        if self.screen_at_start is not None and self.screen_now != self.screen_at_start:
            self.set_screen(self.screen_at_start)    # the screen where it was found

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
    """Number 8-connected foreground regions in reading order, reserving zero for background."""
    mask = np.asarray(mask, dtype=bool)
    starts = mask.copy()
    starts[:, 1:] &= ~mask[:, :-1]
    runs = np.cumsum(starts, dtype=np.int32).reshape(mask.shape) * mask
    parent = np.arange(starts.sum() + 1)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # Join horizontal runs that touch directly or diagonally in adjacent rows.
    for upper, lower in ((runs[:-1], runs[1:]), (runs[:-1, :-1], runs[1:, 1:]),
                         (runs[:-1, 1:], runs[1:, :-1])):
        contact = (upper > 0) & (lower > 0)
        contact[:, 1:] &= ~contact[:, :-1]  # one union per contiguous overlap
        for a, b in zip(upper[contact], lower[contact]):
            a, b = root(a), root(b)
            parent[max(a, b)] = min(a, b)

    roots = [root(i) for i in range(len(parent))]  # includes 0 even for a full-field beam
    regions, numbers = np.unique(roots, return_inverse=True)
    return numbers[runs], len(regions) - 1


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


def show_kept(scope, **which):
    """Draw frames a recording has kept, a row of pictures for each camera, with a cross where the beam was found.

    ``scope``: the session (a :class:`Neoarm`). ``which``: columns of the table and the value wanted, for example
    ``show_kept(scope, setting="all weak")`` or ``show_kept(scope, stage="rulers")``. Needs matplotlib."""
    import matplotlib.pyplot as plt

    kept = [row for row in scope.rows if "frame" in row and all(row.get(key) == value for key, value in which.items())]
    if not kept:
        return print("no frames to show (a dry run, or this has not been recorded)")
    by_camera = {}
    for row in kept:
        by_camera.setdefault(row.get("camera", CAMERA_NAME), []).append(row)
    for camera, rows in by_camera.items():
        fig, axes = plt.subplots(1, len(rows), figsize=(1.75 * len(rows), 2.3), squeeze=False)
        for ax, row in zip(axes[0], rows):
            frame = np.load(scope.folder / row["frame"]).astype(float)
            image = np.sqrt(np.clip(frame - np.median(frame), 0, None))
            beam_only = row["role"] in ("not deflected", "deflected")
            ax.imshow(image, origin="lower", cmap="gray", vmax=None if beam_only else max(np.percentile(image, 99.5), 1e-9))
            if beam_only:
                ax.plot((frame.shape[1] - 1) / 2 + row["beam_x_px"], (frame.shape[0] - 1) / 2 + row["beam_y_px"], "+",
                        color="#D55E00", ms=9, mew=1.2)
            what = (f"{row['deflector']} {row['direction']} {'+' if row['sign'] > 0 else '-'}{row['kick']}"
                    if row["role"] == "deflected" else row["role"])
            ax.set_title(f"{row.get('setting', '')}\n{what}", fontsize=7.5, fontweight="normal")
            ax.set(xticks=[], yticks=[])
        fig.suptitle(f"the {camera}: frames of {frame.shape[1]} x {frame.shape[0]} px", fontsize=9, x=0.01, ha="left")
        fig.tight_layout()
        plt.show()


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


# ------------------------------------------------------------------------------------- rulers: rings and a grating
GOLD_SPACINGS_M = (0.40782e-9 / np.sqrt(3), 0.40782e-9 / 2, 0.40782e-9 / np.sqrt(8))      # the 111, 200 and 220 rings


def radial_profile(frame, centre, step=0.5):
    """The mean of a frame in thin rings about ``centre`` (px from the frame centre, x + iy), out to the nearest
    frame edge. Returns the radii (px) and the means."""
    f = np.asarray(frame, dtype=float)
    ny, nx = f.shape
    y, x = np.mgrid[0:ny, 0:nx]
    cx, cy = (nx - 1) / 2 + centre.real, (ny - 1) / 2 + centre.imag
    r = np.hypot(x - cx, y - cy)
    whole = int(min(cx, cy, nx - 1 - cx, ny - 1 - cy) / step)                 # only rings wholly on the camera
    index = (r / step).astype(int)
    keep = index < whole
    sums = np.bincount(index[keep], weights=f[keep], minlength=whole)
    counts = np.bincount(index[keep], minlength=whole)
    return (np.arange(whole) + 0.5) * step, sums / np.maximum(counts, 1)


def find_rings(frame, centre, disc_rms, spacings=GOLD_SPACINGS_M):
    """The radius (px) of the innermost ring of a ring pattern (gold: the 111 ring), and how wide the rings are.

    The rings' radii are in known proportion (1 / spacing), so one radius places them all. For each trial radius
    the profile outside the central disc is fitted by a smooth background plus one bump per ring; the radius whose
    fit is best is returned, with the bumps' rms width (px). ``centre`` and ``disc_rms`` are the central disc's, from
    :func:`find_beam`. Returns (nan, nan) if no rings are found.
    """
    r, profile = radial_profile(frame, centre)
    start = 4 * disc_rms + 3
    use = r > start
    if use.sum() < 40:
        return float("nan"), float("nan")
    r, profile = r[use], profile[use]
    ratios = spacings[0] / np.asarray(spacings)                              # 1, 1.155, 1.633 for gold
    t = (r - r.mean()) / (np.ptp(r) / 2)
    background = np.stack([t ** k for k in range(5)], axis=1)
    best = (np.inf, np.nan, np.nan)
    scores = {}
    for width in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        for radius in np.arange(start + 3 * width, r[-1] / ratios[1] - 3 * width, 0.25):
            inside = [k for k in ratios if radius * k + 3 * width < r[-1]]
            bumps = np.stack([np.exp(-0.5 * ((r - radius * k) / width) ** 2) for k in inside], axis=1)
            amplitudes, *_ = np.linalg.lstsq(np.concatenate([background, bumps], axis=1), profile, rcond=None)
            if np.any(amplitudes[5:] <= 0):
                continue
            left = profile - background @ amplitudes[:5] - bumps @ amplitudes[5:]
            scores[width, radius] = float(left @ left)
            if scores[width, radius] < best[0]:
                best = (scores[width, radius], radius, width)
    _, radius, width = best
    if not np.isfinite(radius):
        return float("nan"), float("nan")
    lower, upper = scores.get((width, radius - 0.25)), scores.get((width, radius + 0.25))
    if lower is not None and upper is not None and lower + upper - 2 * best[0] > 0:      # the bottom of the parabola
        radius += 0.125 * (lower - upper) / (lower + upper - 2 * best[0])
    return float(radius), float(width)


def find_pitch(frame, zoom=4):
    """The period (px) of a cross-grating in a frame, and the angle (rad) of its lines. Returns (nan, nan) if none.

    First the strongest repeating pattern is found roughly, as the peak of the frame's Fourier transform nearest the
    middle. Then it is measured exactly: the frame is multiplied by a wave of that period, which leaves a phase that
    drifts across the frame by however much the rough period was off; a plane fitted to that phase gives the
    correction. The same is done for the lines at right angles, and the two periods are averaged.
    """
    f = np.asarray(frame, dtype=float)
    ny, nx = f.shape
    f = f - _box_smooth(f, max(nx, ny) // 8)
    power = np.abs(np.fft.fftshift(np.fft.fft2(f * np.outer(np.hanning(ny), np.hanning(nx)), s=(zoom * ny, zoom * nx)))) ** 2
    ky, kx = np.mgrid[0:zoom * ny, 0:zoom * nx]
    ky, kx = (ky - zoom * ny // 2) / (zoom * ny), (kx - zoom * nx // 2) / (zoom * nx)      # cycles per px
    k = np.hypot(kx, ky)
    power = np.where((k > 6 / min(nx, ny)) & (k < 0.25) & (kx >= 0), power, 0.0)   # 4 px to a sixth of the frame
    if power.max() <= 0:
        return float("nan"), float("nan")
    strong = power > 0.3 * power.max()
    iy, ix = np.unravel_index(np.argmin(np.where(strong, k, np.inf)), power.shape)       # the innermost strong peak
    for _ in range(50):                                                                    # climb to its top
        window = power[iy - 1:iy + 2, ix - 1:ix + 2]
        if window.shape != (3, 3):
            return float("nan"), float("nan")
        dy, dx = np.unravel_index(np.argmax(window), (3, 3))
        if (dy, dx) == (1, 1):
            break
        iy, ix = iy + dy - 1, ix + dx - 1
    rough = complex(kx[iy, ix], ky[iy, ix])
    y, x = np.mgrid[0:ny, 0:nx]
    x, y = x - (nx - 1) / 2, y - (ny - 1) / 2

    def exact(wave):
        """The wave (cycles per px, x + iy) that the pattern near ``wave`` really has."""
        for _ in range(3):
            turned = f * np.exp(-2j * np.pi * (wave.real * x + wave.imag * y))
            half = int(round(1 / abs(wave)))                                               # average over two periods
            local = _box_smooth(turned.real, half) + 1j * _box_smooth(turned.imag, half)
            weight = np.abs(local) ** 2
            weight = np.where(weight > 0.1 * weight.max(), weight, 0.0)
            phase = np.angle(local * np.exp(-1j * np.angle(np.sum(weight * local))))        # about the mean phase: no wrap
            design = np.stack([np.ones_like(x), x, y], axis=-1)[weight > 0]
            w = np.sqrt(weight[weight > 0])
            plane, *_ = np.linalg.lstsq(design * w[:, None], phase[weight > 0] * w, rcond=None)
            wave = wave + complex(plane[1], plane[2]) / (2 * np.pi)
        return wave

    first, second = exact(rough), exact(1j * rough)
    if not 0.8 < abs(second) / abs(first) < 1.25:                                          # no lines at right angles: one only
        second = first
    return float(2 / (abs(first) + abs(second))), float(np.angle(first))
