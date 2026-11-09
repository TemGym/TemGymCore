"""Drive the microscope for the projection-system calibration, through PyJEM only (numpy + PyJEM;
nothing from TemGym). Every function and step takes named arguments only.

The general session tools (frames, beam centre, free-lens control, saving and restoring) are the
shared ``examples/pyjem_session.py``; this file adds only this calibration's plan and
measurement. On the microscope PC, copy both files into one folder: this one and
``examples/pyjem_session.py``.

Start in MAG at the 40k row, free-lens control off, at the HT of the session (60 kV planned).
A :class:`Recording` is made one step at a time; each step does one thing, prints what it did and
refuses to run before the steps it needs, so the session can be run cell by cell and stopped
between any two steps:

    from PyJEM import TEM3, detector
    import record

    record.check_connection(TEM3=TEM3, detector=detector)    # reads, one frame, small kicks; harmless
    recording = record.Recording(TEM3=TEM3, detector=detector, folder="D:/projection_response/2026-11-06")
    recording.read_diff_row(camera_length_cm=40)              # DIFF, read IL1-PL1, back to MAG
    recording.take_over_lenses()                              # the 40k row into free-lens control
    recording.plan_settings()                                 # nothing moves
    recording.dark_frame()                                    # beam blanked
    recording.measure(setting="40k row", kicks_along="x")     # one measurement at a time, or
    recording.measure_remaining()                             # every planned one not yet made
    recording.restore()                                       # the microscope as it was

The 22 measurements cover image-like and diffraction-like lens settings evenly: around the 40k
row and around the DIFF row (IL1-PL1 as the console sets them in DIFF), each lens stepped up and
down on its own by a fixed number of DAC units, all set through free-lens control in MAG. The 40k
row is measured four times (the wobble), once with y kicks. At each: a frame without a kick, then
a +/- beam shift (CLA1) and a +/- beam tilt (CLA2).

Steps are sized at the microscope (:func:`pyjem_session.size_step`): a kick starts at 100 units and
grows until the beam moves about 50 px; a lens step starts at 250 DAC units and doubles until it
clearly changes how the beam moves under the row's own kick (by about 10 of its 50 px), at most
8000 units and never below 2000 (no lens switched off); a step that loses the beam (off the camera,
at its edge) is backed off. Every frame is saved (``frames/NNNN.npy``) with a log
line (``log.json``, rewritten after every frame).

To test the model at new lens settings, open a second recording and use
``plan_new_settings(settings=...)`` instead of ``plan_settings()``. Stage 2 (draft) is in
``stage2/record.py``.

Check on the real system before relying on it: that CLA1 is the beam shift and CLA2 the beam tilt
(and :mod:`pyjem_session`'s own checks: the free-lens ids and the camera data type).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

# The shared session tools: next to this file on the microscope PC, in examples/ in the repository.
sys.path.append(str(Path(__file__).resolve().parents[2]))
from pyjem_session import DAC_MAX, LENS_IDS, find_beam, json_value, read_frame, select_camera_length  # noqa: E402
from stage2.stage1_session import Session, size_step  # noqa: E402

#: The lenses this calibration drives, through free-lens control.
LENSES = ("IL1", "IL2", "IL3", "PL1")

#: Around each row: the row itself, each lens stepped up and down on its own, and the row again.
STEPS = [(lens, sign) for lens in LENSES for sign in (+1, -1)]
#: The measurements in order: row, lens step (None: the row itself) and kick direction. The 40k row
#: at the start and end of its block and twice more at the end gives the wobble; the last has y
#: kicks (roundness check).
PLAN = ([(row, step, "x") for row in ("40k", "DIFF") for step in (None, *STEPS, None)]
        + [("40k", None, "x"), ("40k", None, "y")])


def _format(dac):
    """IL1-PL1 DAC values as text."""
    return ", ".join(f"{name} {int(round(v))}" for name, v in zip(LENSES, dac))


def _label(row, step):
    """A setting's name, e.g. ``"DIFF row"`` or ``"DIFF row: IL1 down"``."""
    return f"{row} row" + (f": {step[0]} {'up' if step[1] > 0 else 'down'}" if step else "")


def _xy(z):
    """A complex camera position as text, px."""
    return f"({z.real:+.1f}, {z.imag:+.1f})"


def check_connection(*, TEM3, detector, camera_name=None, dtype=None, kick_units=300, settle_s=0.5):
    """A quick, harmless check before a session, and after any change to this file. Reads and
    prints the HT, function mode and MAG row, IL1-PL1 and their free-lens control, the deflectors
    and the camera; takes one frame and finds the beam; moves CLA1 and then CLA2 by ``kick_units``
    in x, a frame each, and puts them back (in MAG with the image in focus, the shift moves the
    beam and the tilt hardly does). The lenses are not touched and nothing is saved.

    Prints one line per item and a list of problems (none, if all is well). Returns everything as
    a dict. Raises if a deflector does not read back its starting value afterwards."""
    lens, defl, ht, eos = TEM3.Lens3(), TEM3.Def3(), TEM3.HT3(), TEM3.EOS3()
    camera = detector.Detector(camera_name or detector.get_attached_detector()[0])
    problems = []
    mode, mag = eos.GetFunctionMode(), eos.GetMagValue()
    flc = {name: lens.GetFLCInfo(LENS_IDS[name]) for name in LENSES}
    console = np.array([getattr(lens, f"Get{name}")() for name in LENSES], float)
    centre = {name: list(getattr(defl, f"Get{name}")()) for name in ("CLA1", "CLA2")}
    print(f"HT {ht.GetHtValue() / 1e3:g} kV; {mode[1]} at {mag[2]}; lenses {_format(console)}", flush=True)
    print(f"free-lens control (on, value): {flc}; CLA1 {centre['CLA1']}, CLA2 {centre['CLA2']}", flush=True)
    if mode[0] != 0:
        problems.append(f"function mode is {mode[1]}, not MAG")
    if any(on for on, _ in flc.values()):
        problems.append("free-lens control is on for " + ", ".join(n for n, (on, _) in flc.items() if on))
    frame, dtype = read_frame(camera, dtype)
    beam = find_beam(frame=frame)
    centre_px, rms = beam.centre, beam.rms
    print(f"camera: {frame.shape[1]}x{frame.shape[0]}, read as {dtype}, values {frame.min()} to {frame.max()}; "
          f"beam at {centre_px.real:+.1f}, {centre_px.imag:+.1f} px from the centre, rms {rms:.1f} px", flush=True)
    problems += beam.problems
    moves = {}
    try:
        for name, kind in (("CLA1", "shift"), ("CLA2", "tilt")):
            x, y = centre[name]
            getattr(defl, f"Set{name}")(int(x + kick_units), int(y))
            time.sleep(settle_s)
            moved = find_beam(frame=read_frame(camera, dtype)[0]).centre
            getattr(defl, f"Set{name}")(int(x), int(y))
            moves[kind] = abs(moved - centre_px) / kick_units
            note = " (expected: in focus, a tilt hardly moves the beam)" if kind == "tilt" else ""
            print(f"{name} ({kind}) +{kick_units} units: the beam moved {abs(moved - centre_px):.1f} px{note}", flush=True)
            if kind == "shift" and abs(moved - centre_px) < 0.5:
                problems.append("CLA1 moved the beam less than 0.5 px: a larger kick_units, or check the deflector")
    finally:
        for name in ("CLA1", "CLA2"):
            getattr(defl, f"Set{name}")(*(int(v) for v in centre[name]))
    after = {name: list(getattr(defl, f"Get{name}")()) for name in ("CLA1", "CLA2")}
    if after != centre:
        raise RuntimeError(f"the deflectors read {after} after the check, {centre} before")
    print("problems: " + ("none" if not problems else "; ".join(problems)), flush=True)
    return dict(ht_v=ht.GetHtValue(), function_mode=mode, mag=mag, lenses=console.tolist(), flc=flc,
                deflectors=centre, frame_shape=frame.shape, dtype=dtype, beam_px=centre_px, beam_rms_px=rms,
                px_per_unit=moves, problems=problems)


# ------------------------------------------------------------------------- the recording
class Recording(Session):
    """A recording of how the beam moves on the camera, made one step at a time (see the module
    docstring for the order): a :class:`pyjem_session.Session` driving IL1-PL1 and
    moving CLA1 and CLA2. Opening it creates ``folder`` (new, empty) and saves the microscope
    state (``state_before.json``); nothing on the microscope changes. :meth:`restore` puts the
    microscope back as it was then, at any point, also after an error.

    Parameters
    ----------
    TEM3, detector
        PyJEM's (or the simulated microscope's) ``TEM3`` and ``detector``.
    folder : str or Path
        Output folder; must not exist yet.
    camera_name : str, optional
        Camera to use (default: the first attached).
    dtype : str, optional
        Pixel type of the camera's raw data (default: signed integers of the raw size).
    settle_s : float
        Wait after every lens change (s).
    """

    def __init__(self, *, TEM3, detector, folder, camera_name=None, dtype=None, settle_s=2.0):
        super().__init__(TEM3=TEM3, detector=detector, folder=folder, lenses=LENSES, deflectors=("CLA1", "CLA2"),
                         camera_name=camera_name, dtype=dtype, settle_s=settle_s)
        self.diffraction = self.settings = None
        self.planned, self.done, self.kicks = [], set(), {}
        self.row_kick = {}                  # per row: the kick that moves the beam most there, and its move

    # -- the steps ------------------------------------------------------------------------
    def read_diff_row(self, *, camera_length_cm=40):
        """Switch to DIFF at the first camera length >= ``camera_length_cm``, read IL1, IL2, IL3
        and PL1 there (the DIFF row) and go back to MAG. Run it before :meth:`take_over_lenses`,
        so the console's own values are read. Returns the four values."""
        self._require(self.base is None, "read_diff_row() before take_over_lenses()")
        select_camera_length(eos=self.eos, camera_length_cm=camera_length_cm)
        row = self.eos.GetMagValue()[2]
        self.diffraction = self.console_lenses()
        self.eos.SelectFunctionMode(0)
        if self.eos.GetMagValue() != self.state["mag"]:
            raise RuntimeError(f"back in MAG at {self.eos.GetMagValue()[2]}, but the recording started at "
                               f"{self.state['mag'][2]}: select that row by hand, then go on")
        print(f"DIFF row ({row}): {_format(self.diffraction)}; back in MAG at {self.state['mag'][2]}", flush=True)
        return self.diffraction

    def take_over_lenses(self):
        """Put IL1, IL2, IL3 and PL1 under free-lens control at the values they run at now (the
        40k row) and switch it on; see :meth:`Session.take_over_lenses`. Returns the four values."""
        return super().take_over_lenses()

    def plan_settings(self, *, beam_move_px=50.0, lens_change_px=20.0, first_kick_units=100, first_lens_step=250,
                      max_kick_units=20000, max_lens_step=8000, lowest_lens_dac=2000):
        """Plan the 22 measurements (module docstring) around the 40k row and the DIFF row. Each kick
        (CLA1 and CLA2) is sized at the microscope from ``first_kick_units`` up to
        ``max_kick_units``, to move the beam about ``beam_move_px``. Each lens step is sized from
        ``first_lens_step`` DAC units up to ``max_lens_step`` (never taking a lens below
        ``lowest_lens_dac``), to change how the beam moves under the row's own kick by about
        ``lens_change_px`` / 2 or more.
        Writes ``plan.json`` and prints the measurements; nothing on the microscope changes."""
        self._require(self.diffraction is not None, "read_diff_row() first")
        self._require(self.base is not None, "take_over_lenses() first")
        self._require(not self.planned, "the measurements are already planned")
        rows = {"40k": self.base, "DIFF": self.diffraction}
        self.settings, self.steps, self.planned = {}, {}, []
        for row, step, direction in PLAN:
            label = _label(row, step)
            self.settings[label] = None if step else rows[row]
            if step:
                self.steps[label] = (rows[row], LENSES.index(step[0]), step[1])
            self.planned.append((label, direction))
        self._set_steps(beam_move_px=beam_move_px, lens_change_px=lens_change_px, first_kick_units=first_kick_units,
                        first_lens_step=first_lens_step, max_kick_units=max_kick_units, max_lens_step=max_lens_step,
                        lowest_lens_dac=lowest_lens_dac)

    def plan_new_settings(self, *, settings, beam_move_px=50.0, first_kick_units=100, max_kick_units=20000):
        """Plan one measurement at each of ``settings`` (rows of IL1, IL2, IL3, PL1 DAC values, e.g.
        from ``new_settings`` in the notebook), kicks sized as in :meth:`plan_settings` -- to compare
        with the model's predictions. Writes ``plan.json`` and prints the measurements; nothing on
        the microscope changes."""
        self._require(self.base is not None, "take_over_lenses() first")
        self._require(not self.planned, "the measurements are already planned")
        self.settings = {f"new {k}": row for k, row in enumerate(np.atleast_2d(np.asarray(settings, float)))}
        self.steps = {}
        self._check_range(self.settings)
        self.planned = [(label, "x") for label in self.settings]
        self._set_steps(beam_move_px=beam_move_px, first_kick_units=first_kick_units, max_kick_units=max_kick_units)

    def dark_frame(self):
        """A frame with the beam blanked; its median is the background of every later frame."""
        self._require(self.base is not None, "take_over_lenses() first")
        with self.blanked():
            dark = self.snap(role="dark", kind="none", offset=[0, 0], measurement=0, setting="dark", direction="x",
                             dac=list(self.base))
        self.background = float(np.median(dark))
        print(f"dark frame: background {self.background:.1f}", flush=True)

    def measure(self, *, setting, kicks_along):
        """Measure at one planned lens ``setting`` (as :meth:`plan_settings` prints it, e.g.
        "40k row" or "DIFF row: IL1 x0.60"; "new 0", ... after :meth:`plan_new_settings`), with the kicks along
        ``kicks_along`` ("x" or "y"). Sets IL1-PL1 through free-lens control, takes a frame
        without a kick, then a +/- kick pair on CLA1 (shift) and on CLA2 (tilt), each deflector back
        at its centre afterwards; every action is printed. It makes the first planned measurement
        of that setting and direction not yet made; once all are made, it adds a repeat (the
        analysis uses the last pair). On an error the deflectors go back to their centre. Returns
        the kick sizes used (deflector units)."""
        self._require(self.planned, "plan_settings() or plan_new_settings() first")
        self._require(setting in self.settings, f"setting {setting!r} is not planned; planned: {list(self.settings)}")
        self._require(kicks_along in ("x", "y"), 'kicks_along is "x" or "y"')
        todo = [n for n, entry in enumerate(self.planned, 1) if entry == (setting, kicks_along) and n not in self.done]
        if todo:
            return self._measure_planned(todo[0])
        self.planned.append((setting, kicks_along))
        print(f"(a repeat: every planned measurement of setting {setting} along {kicks_along} is made)", flush=True)
        return self._measure_planned(len(self.planned))

    def measure_remaining(self):
        """Every planned measurement not yet made, in the planned order."""
        for number in range(1, len(self.planned) + 1):
            if number not in self.done:
                self._measure_planned(number)

    def restore(self):
        """Put the microscope back as it was when the recording was opened
        (:meth:`Session.restore`). Safe at any point."""
        super().restore()
        print(f"{len(self.done)} of {len(self.planned)} measurements made, in {self.folder}", flush=True)

    # -- helpers ----------------------------------------------------------------------------
    def _measure_planned(self, number):
        """Planned measurement ``number`` (from 1), with the checks and the error handling."""
        self._require(self.base is not None, "take_over_lenses() first")
        self._require(self.background is not None, "dark_frame() first")
        label, direction = self.planned[number - 1]
        print(f"measurement {number} of {len(self.planned)}", flush=True)
        try:
            if self.settings[label] is None:                        # a lens step not yet sized
                self.settings[label] = self._size_lens_step(number, label, direction)
                self._write_plan(show=False)
            used = self._measure(number, label, self.settings[label], direction, centre=self.centre,
                                 beam_move_px=self.kicks["beam_move_px"], first_kick_units=self.kicks["first_kick_units"],
                                 max_kick_units=self.kicks["max_kick_units"])
        except BaseException:
            self.recentre_deflectors()
            print(f"measurement {number} stopped; the deflectors are back at their centre. "
                  "restore() puts the rest back.", flush=True)
            raise
        self.done.add(number)
        return used

    def _check_range(self, settings):
        bad = {k: v for k, v in settings.items() if v is not None and (np.any(np.round(v) > DAC_MAX) or np.any(v < 0))}
        if bad:
            raise ValueError(f"settings outside 0-{DAC_MAX}: {bad}")

    def _set_steps(self, **sizes):
        self.kicks = sizes
        self._write_plan()
        print(f"kicks: from {sizes['first_kick_units']} units, grown to move the beam about {sizes['beam_move_px']:g} px, "
              f"at most {sizes['max_kick_units']} units (deflector centres CLA1 {self.centre['CLA1']}, "
              f"CLA2 {self.centre['CLA2']}, range 0-{DAC_MAX})"
              + (f"; lens steps: from {sizes['first_lens_step']} DAC units, grown until the beam clearly changes, at "
                 f"most {sizes['max_lens_step']}" if "first_lens_step" in sizes else ""), flush=True)

    def _size_lens_step(self, number, label, direction):
        """Size a planned lens step (:func:`pyjem_session.size_step`) and return its DAC values. The
        step's effect is measured as the change of the beam's move under the row's own kick."""
        row, i, sign = self.steps[label]
        row_label = label.split(":")[0]
        self._require(row_label in self.row_kick, f"measure the {row_label} first (its kick is the yardstick)")
        name, kick, move = self.row_kick[row_label]
        lowest, room_up = self.kicks["lowest_lens_dac"], DAC_MAX - row[i]
        room = room_up if sign > 0 else row[i] - lowest
        if room < self.kicks["first_lens_step"]:
            sign, room = -sign, (row[i] - lowest if sign > 0 else room_up)
            print(f"  {LENSES[i]} has no room that way; stepping it the other way", flush=True)
        meta = dict(measurement=number, setting=label, direction=direction, kind="lens")
        home = np.array(self.centre[name], float)

        def try_step(step):
            dac = row.copy()
            dac[i] += sign * round(step)
            readback = [self.set_lenses(dac)[n] for n in LENSES]
            still = find_beam(frame=self.snap(role="lens test", offset=[0, 0], dac=readback, **meta),
                              background=self.background)
            self.set_deflector(name, home + kick)
            kicked = find_beam(frame=self.snap(role="lens test", offset=list(kick), dac=readback, **meta),
                               background=self.background)
            self.set_deflector(name, home)
            return abs((kicked.centre - still.centre) - move), still.usable and kicked.usable

        print(f"{label}: sizing the {LENSES[i]} step by its effect on the {name} kick's {abs(move):.0f} px move", flush=True)
        step, change = size_step(try_step=try_step, start=self.kicks["first_lens_step"],
                                 largest=min(self.kicks["max_lens_step"], room), target_px=self.kicks["lens_change_px"],
                                 name=f"{LENSES[i]} {'+' if sign > 0 else '-'}")
        dac = row.copy()
        dac[i] += sign * round(step)
        print(f"  {LENSES[i]} step {'+' if sign > 0 else '-'}{round(step)} DAC units: the {name} move changes by "
              f"{change:.1f} px", flush=True)
        return dac

    def _write_plan(self, show=True):
        (self.folder / "plan.json").write_text(json.dumps(dict(
            base_dac=self.base.tolist(), diffraction=None if self.diffraction is None else self.diffraction.tolist(),
            settings={k: (None if v is None else v.tolist()) for k, v in self.settings.items()}, centre=self.centre,
            plan=self.planned, steps=self.kicks), indent=1, default=json_value))
        if not show:
            return
        print("planned measurements:", flush=True)
        width = max(len(label) for label, _ in self.planned)
        for number, (label, direction) in enumerate(self.planned, 1):
            values = self.settings[label]
            print(f"  {number:2d}  {label:{width}s}  kicks along {direction}   "
                  f"{'step sized at the microscope' if values is None else _format(values)}", flush=True)

    # -- one lens setting --------------------------------------------------------------------
    def _measure(self, number, label, dac, direction, *, centre, beam_move_px=50.0, first_kick_units=100,
                 max_kick_units=20000, skip_off_camera=False):
        """Set the lenses, take a frame without a kick, then for CLA1 (shift) and CLA2 (tilt) a +/-
        pair of kicks along ``direction``, each deflector back at its centre afterwards. Every
        action is printed. Each kick is sized (:func:`pyjem_session.size_step`) from
        ``first_kick_units`` to move the beam about ``beam_move_px``, at most ``max_kick_units``. If the beam reaches the frame edge
        the kick is halved and the pair taken again. With ``skip_off_camera``, a setting whose beam
        is cut by the frame edge (or off the camera) without any kick is skipped. Returns the kick
        sizes used (deflector units), or ``None`` if skipped."""
        readback = self.set_lenses(dac)
        print(f"{label}: {_format(dac)} through free-lens control; waited {self.settle_s:g} s", flush=True)
        meta = dict(measurement=number, setting=label, direction=direction, dac=[readback[n] for n in LENSES])
        base = self.snap(role="base", kind="none", offset=[0, 0], **meta)
        beam = find_beam(frame=base, background=self.background)
        c0, rms = beam.centre, beam.rms
        print(f"  frame without a kick: beam at {_xy(c0)} px from the centre, rms {rms:.1f} px"
              + "".join(f"; {p}" for p in beam.problems), flush=True)
        if not beam.usable and skip_off_camera:
            print(f"  skipped setting {label}: {'; '.join(beam.problems)}", flush=True)
            return None
        if not beam.usable:
            print(f"  WARNING setting {label}: {'; '.join(beam.problems)}. Condense or centre the beam, or these "
                  "frames will be dropped.", flush=True)
        unit = np.array([1, 0]) if direction == "x" else np.array([0, 1])
        used = {}
        for kind, name in (("shift", "CLA1"), ("tilt", "CLA2")):
            home = np.array(centre[name], float)

            def try_kick(step):
                kick = round(step) * unit
                self.set_deflector(name, home + kick)
                found = find_beam(frame=self.snap(role="test", kind=kind, offset=list(kick), **meta),
                                  background=self.background)
                self.set_deflector(name, home)
                return abs(found.centre - c0), found.usable

            limit = int(min(*(DAC_MAX - home), *home))
            print(f"  sizing the {name} ({kind}) kick along {direction}", flush=True)
            size, change = size_step(try_step=try_kick, start=first_kick_units, largest=min(max_kick_units, limit),
                                     target_px=beam_move_px, proportional=True, name=f"{name} kick")
            size = int(round(size))
            if size == 0:
                print(f"  {name}: no kick keeps the beam usable; this pair is skipped", flush=True)
                continue
            for _ in range(6):
                frames = []
                for sign, role in ((+1, "plus"), (-1, "minus")):
                    self.set_deflector(name, home + sign * size * unit)
                    frames.append(self.snap(role=role, kind=kind, offset=list(sign * size * unit), **meta))
                self.set_deflector(name, home)
                spots = [find_beam(frame=f, background=self.background) for f in frames]
                if all(not spot.touches_edge for spot in spots):
                    break
                print(f"  {name} +/-{size}: the beam reached the frame edge; halving the kick and taking the pair again",
                      flush=True)
                size = int(round(size / 2))
                for entry in self.log[-2:]:
                    entry["role"] += " (clipped, retaken)"
            print(f"  {name} ({kind}) kicks +{size} and -{size} along {direction}: beam moved {_xy(spots[0].centre - c0)} "
                  f"and {_xy(spots[1].centre - c0)} px; {name} back to {[int(v) for v in home]}", flush=True)
            used[kind] = float(size)
            move = (spots[0].centre - spots[1].centre) / 2
            if ":" not in label and abs(move) > abs(self.row_kick.get(label, (None, None, 0))[2]):
                self.row_kick[label] = (name, size * unit, move)       # the row's own kick: the lens steps' yardstick
        return used


