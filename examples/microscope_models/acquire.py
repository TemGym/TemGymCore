"""Drive the microscope for the projection-system calibration, through PyJEM only (numpy + PyJEM;
nothing from TemGym). This is the one file to copy to the microscope PC. Every function and step
takes named arguments only.

Start in MAG at the 40k row, free-lens control off, at the HT of the session (60 kV planned).
A :class:`Recording` is made one step at a time; each step does one thing, prints what it did and
refuses to run before the steps it needs, so the session can be run cell by cell and stopped
between any two steps:

    from PyJEM import TEM3, detector
    import acquire

    acquire.check_connection(TEM3=TEM3, detector=detector)    # reads, one frame, small kicks; harmless
    recording = acquire.Recording(TEM3=TEM3, detector=detector, folder="D:/projection_response/2026-11-06")
    recording.read_diff_row(camera_length_cm=40)              # DIFF, read IL1-PL1, back to MAG
    recording.take_over_lenses()                              # the 40k row into free-lens control
    recording.plan_settings(beam_move_px=50, test_kick_units=300, max_kick_units=20000)   # nothing moves
    recording.dark_frame()                                    # beam blanked
    recording.measure(setting="40k row", kicks_along="x")     # one measurement at a time, or
    recording.measure_remaining()                             # every planned one not yet made
    recording.restore()                                       # the microscope as it was

The 20 measurements cover image-like and diffraction-like lens settings evenly: the 40k row and
the DIFF row (IL1-PL1 as the console sets them in DIFF), each alone and times the same 7 sets of
factors, all set through free-lens control in MAG. The 40k row is measured three times (the
wobble), once with y kicks. At each: a frame without a kick, a small test kick, then a +/- beam
shift (CLA1) and a +/- beam tilt (CLA2), each scaled to move the beam about 50 px, kept on the
camera. Every frame is saved (``frames/NNNN.npy``) with a log line (``log.json``,
rewritten after every frame).

To test the model at new lens settings, open a second recording and use
``plan_new_settings(settings=...)`` instead of ``plan_settings()``. Stage 2 (draft) is in
``stage2/record.py``.

Check on the real system before relying on it: the free-lens ids (IL1 10, IL2 11, IL3 12, PL1 14,
CL3 2), that CLA1 is the beam shift and CLA2 the beam tilt, and the camera data type (``dtype``; by
default 2 or 4 bytes per pixel are read as signed integers).
"""

import json
import time
from pathlib import Path

import numpy as np

FLC_IDS = {"IL1": 10, "IL2": 11, "IL3": 12, "PL1": 14}
ALL_LENSES = ("CL1", "CL2", "CL3", "CM", "OLc", "OLf", "OM", "OM2", "IL1", "IL2", "IL3", "IL4", "PL1", "PL2", "PL3")
DAC_MAX = 65535


def _plain(value):
    """JSON fallback for numpy numbers and arrays."""
    return value.tolist() if hasattr(value, "tolist") else str(value)

#: The factors applied to a row's DAC values (IL1, IL2, IL3, PL1); the first set is the row itself.
MULTIPLIERS = np.array([
    [41782, 24862, 35544, 64000],
    [25000, 24862, 35544, 64000],
    [41782, 24862, 25000, 64000],
    [41782, 24862, 35544, 38400],
    [41782, 24862, 36000, 32000],
    [41782, 40000, 35544, 38400],
    [41782, 40000, 36000, 64000],
    [41782, 40000, 35544, 32000],
]) / np.array([41782, 24862, 35544, 64000])

#: The measurements in order: row, factor set (0: the row itself) and kick direction. The 40k row
#: at the start, middle and end gives the wobble; the last has y kicks (roundness check).
PLAN = [(row, k, "x") for row in ("40k", "DIFF") for k in (0, 1, 2, 3, 4, 0, 5, 6, 7)] + [("40k", 0, "x"), ("40k", 0, "y")]


# ------------------------------------------------------------------------- frames
def measure_spot(*, frame, background=0.0, window=4.0):
    """Beam centre (x, y in px, from the frame centre), rms radius (px) and the fraction of the
    signal within 3 px of the frame edge. The centre is the intensity-weighted mean inside a circle
    of ``window`` x rms around the spot, iterated; the background is subtracted first."""
    f = np.asarray(frame, float) - background
    ny, nx = f.shape
    y, x = np.mgrid[0:ny, 0:nx]
    y, x = y - (ny - 1) / 2, x - (nx - 1) / 2
    w = np.clip(f, 0, None)
    cx, cy = (w * x).sum() / w.sum(), (w * y).sum() / w.sum()
    rms = np.sqrt(((w * ((x - cx) ** 2 + (y - cy) ** 2)).sum() / w.sum()) / 2)
    for _ in range(5):
        inside = np.hypot(x - cx, y - cy) < max(window * rms, 6.0)
        wi = f * inside
        if wi.sum() <= 0:                                  # nothing above background here: keep the last estimate
            break
        cx, cy = (wi * x).sum() / wi.sum(), (wi * y).sum() / wi.sum()
        rms = np.sqrt(max((wi * ((x - cx) ** 2 + (y - cy) ** 2)).sum() / wi.sum(), 0.25) / 2)
    edge = np.zeros_like(f, bool)
    edge[:3], edge[-3:], edge[:, :3], edge[:, -3:] = True, True, True, True
    return complex(cx, cy), float(rms), float(w[edge].sum() / w.sum())


def _format(dac):
    """IL1-PL1 DAC values as text."""
    return ", ".join(f"{name} {int(round(v))}" for name, v in zip(FLC_IDS, dac))


def _label(row, factors):
    """A setting's name: its row and the factors that differ from 1, e.g. ``"DIFF row: IL1 x0.60"``."""
    changed = [f"{name} x{f:.2f}" for name, f in zip(FLC_IDS, factors) if abs(f - 1) > 1e-9]
    return f"{row} row" + (": " + ", ".join(changed) if changed else "")


def _xy(z):
    """A complex camera position as text, px."""
    return f"({z.real:+.1f}, {z.imag:+.1f})"


def _read_frame(camera, dtype=None):
    """One raw frame from a PyJEM camera, as integers, and the pixel type used."""
    area = camera.get_detectorsetting()["ImagingArea"]
    raw = camera.snapshot_rawdata()
    dtype = dtype or {2: "<i2", 4: "<i4"}[len(raw) // (area["Height"] * area["Width"])]
    return np.frombuffer(raw, dtype=dtype).reshape(area["Height"], area["Width"]), dtype


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
    flc = {name: lens.GetFLCInfo(i) for name, i in FLC_IDS.items()}
    console = np.array([getattr(lens, f"Get{name}")() for name in FLC_IDS], float)
    centre = {name: list(getattr(defl, f"Get{name}")()) for name in ("CLA1", "CLA2")}
    print(f"HT {ht.GetHtValue() / 1e3:g} kV; {mode[1]} at {mag[2]}; lenses {_format(console)}", flush=True)
    print(f"free-lens control (on, value): {flc}; CLA1 {centre['CLA1']}, CLA2 {centre['CLA2']}", flush=True)
    if mode[0] != 0:
        problems.append(f"function mode is {mode[1]}, not MAG")
    if any(on for on, _ in flc.values()):
        problems.append("free-lens control is on for " + ", ".join(n for n, (on, _) in flc.items() if on))
    frame, dtype = _read_frame(camera, dtype)
    background = float(np.median(frame))
    centre_px, rms, edge = measure_spot(frame=frame, background=background)
    print(f"camera: {frame.shape[1]}x{frame.shape[0]}, read as {dtype}, values {frame.min()} to {frame.max()}; "
          f"beam at {centre_px.real:+.1f}, {centre_px.imag:+.1f} px from the centre, rms {rms:.1f} px", flush=True)
    if frame.max() - background < 10 * max(np.std(frame[frame <= np.percentile(frame, 50)]), 1.0):
        problems.append("no clear beam on the camera")
    if edge > 1e-3:
        problems.append("the beam touches the frame edge: condense it or centre it")
    moves = {}
    try:
        for name, kind in (("CLA1", "shift"), ("CLA2", "tilt")):
            x, y = centre[name]
            getattr(defl, f"Set{name}")(int(x + kick_units), int(y))
            time.sleep(settle_s)
            moved = measure_spot(frame=_read_frame(camera, dtype)[0], background=background)[0]
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


def select_magnification(*, eos, magnification):
    """Step the TEM MAG selector (``TEM3.EOS3()``) to a nominal magnification; returns its index."""
    previous = None
    while eos.GetCurrentMagSelectorID()[0] != previous:          # to the lowest row
        previous = eos.GetCurrentMagSelectorID()[0]
        eos.DownSelector()
    previous = None
    while True:
        index, value = eos.GetCurrentMagSelectorID()
        if value == magnification:
            return index
        if index == previous:
            raise ValueError(f"the magnification selector has no {magnification:g}x row")
        previous = index
        eos.UpSelector()


def select_camera_length(*, eos, camera_length_cm):
    """Switch to DIFF and step the selector up to the first camera length >= ``camera_length_cm``."""
    eos.SelectFunctionMode(4)
    for _ in range(30):
        if eos.GetMagValue()[0] >= camera_length_cm:
            break
        eos.UpSelector()


def read_all_lenses(*, TEM3):
    """Every lens value PyJEM can read (saved with each session). Missing getters are skipped."""
    lens, eos = TEM3.Lens3(), TEM3.EOS3()
    values = {}
    for name in ALL_LENSES:
        try:
            values[name] = getattr(lens, f"Get{name}")()
        except Exception as error:                                   # noqa: BLE001 -- report and continue
            values[name] = f"unavailable ({type(error).__name__})"
    try:
        values["function_mode"] = eos.GetFunctionMode()
    except Exception:                                                # noqa: BLE001
        pass
    return values


# ------------------------------------------------------------------------- the recording
class Recording:
    """A recording of how the beam moves on the camera, made one step at a time (see the module
    docstring for the order). Opening it creates ``folder`` (new, empty) and saves the microscope
    state (``state_before.json``); nothing on the microscope changes. :meth:`restore` puts the
    microscope back as it was then, at any point, also after an error.

    Parameters
    ----------
    TEM3, detector
        PyJEM's (or the virtual microscope's) ``TEM3`` and ``detector``.
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
        self.lens, self.defl, self.ht, self.eos = TEM3.Lens3(), TEM3.Def3(), TEM3.HT3(), TEM3.EOS3()
        self.camera = detector.Detector(camera_name or detector.get_attached_detector()[0])
        self.folder = Path(folder)
        (self.folder / "frames").mkdir(parents=True, exist_ok=False)
        self.dtype, self.settle_s, self.log = dtype, settle_s, []
        self.base = self.diffraction = self.settings = self.background = None
        self.planned, self.done, self.kicks = [], set(), {}
        self.centre = self._deflectors()
        self.state = dict(time=time.time(), ht_v=self.ht.GetHtValue(), mag=self.eos.GetMagValue(),
                          function_mode=self.eos.GetFunctionMode(), lenses=read_all_lenses(TEM3=TEM3),
                          flc={n: self.lens.GetFLCInfo(i) for n, i in FLC_IDS.items()}, **self.centre,
                          beam_blank=self.defl.GetBeamBlank(), camera=self.camera.get_detectorsetting())
        (self.folder / "state_before.json").write_text(json.dumps(self.state, indent=1, default=_plain))
        print(f"opened {self.folder}: microscope state saved ({self.state['ht_v'] / 1e3:g} kV, "
              f"{self.state['mag'][2]}, CLA1 {self.centre['CLA1']}, CLA2 {self.centre['CLA2']}); nothing changed",
              flush=True)

    # -- the steps ------------------------------------------------------------------------
    def read_diff_row(self, *, camera_length_cm=40):
        """Switch to DIFF at the first camera length >= ``camera_length_cm``, read IL1, IL2, IL3
        and PL1 there (the DIFF row) and go back to MAG. Run it before :meth:`take_over_lenses`,
        so the console's own values are read. Returns the four values."""
        self._require(self.base is None, "read_diff_row() before take_over_lenses()")
        select_camera_length(eos=self.eos, camera_length_cm=camera_length_cm)
        row = self.eos.GetMagValue()[2]
        self.diffraction = self._console_lenses()
        self.eos.SelectFunctionMode(0)
        if self.eos.GetMagValue() != self.state["mag"]:
            raise RuntimeError(f"back in MAG at {self.eos.GetMagValue()[2]}, but the recording started at "
                               f"{self.state['mag'][2]}: select that row by hand, then go on")
        print(f"DIFF row ({row}): {_format(self.diffraction)}; back in MAG at {self.state['mag'][2]}", flush=True)
        return self.diffraction

    def take_over_lenses(self):
        """Put IL1, IL2, IL3 and PL1 under free-lens control at the values they run at now (the
        40k row) and switch it on. The values are read before free-lens control is touched and
        written into it before switching on, so a value left over from earlier cannot replace
        them. Returns the four values."""
        self._require(self.base is None, "the lenses are already under free-lens control")
        base = self._console_lenses()
        if any(self.lens.GetFLCInfo(i)[0] for i in FLC_IDS.values()):
            print("  note: free-lens control was already on; its values are taken as the 40k row", flush=True)
        for value, i in zip(base, FLC_IDS.values()):
            self.lens.SetFLCAbs(i, int(value))
            self.lens.SetFLCSw(i, 1)
        readback = self._flc_values()
        if np.any(readback != base):
            raise RuntimeError(f"free-lens control reads {readback}, expected the 40k row {base}")
        self.base = base
        print(f"free-lens control on at the 40k row: {_format(base)}", flush=True)
        return base

    def plan_settings(self, *, beam_move_px=50.0, test_kick_units=300, max_kick_units=20000):
        """Plan the 20 measurements (module docstring) from the 40k row and the DIFF row. A factor
        that would take a lens above the DAC range is replaced by its reciprocal. Each kick (CLA1
        and CLA2, in deflector units) will be sized from a test kick of ``test_kick_units`` to move
        the beam about ``beam_move_px`` on the camera, at most ``max_kick_units``. Writes
        ``plan.json`` and prints the measurements; nothing on the microscope changes."""
        self._require(self.diffraction is not None, "read_diff_row() first")
        self._require(self.base is not None, "take_over_lenses() first")
        self._require(not self.planned, "the measurements are already planned")
        rows = {"40k": self.base, "DIFF": self.diffraction}
        settings, self.planned = {}, []
        for row, k, direction in PLAN:
            factors = MULTIPLIERS[k].copy()
            over = np.round(rows[row] * factors) > DAC_MAX
            factors[over] = 1 / factors[over]                        # too high: use the reciprocal factor
            label = _label(row, factors)
            settings[label] = rows[row] * factors
            self.planned.append((label, direction))
        self._check_range(settings)
        self.settings = settings
        self._set_kicks(beam_move_px, test_kick_units, max_kick_units)

    def plan_new_settings(self, *, settings, beam_move_px=50.0, test_kick_units=300, max_kick_units=20000):
        """Plan one measurement at each of ``settings`` (rows of IL1, IL2, IL3, PL1 DAC values, e.g.
        from ``analyse.new_settings``), kicks sized as in :meth:`plan_settings` -- to compare with
        the model's predictions. Writes ``plan.json`` and prints the measurements; nothing on the
        microscope changes."""
        self._require(self.base is not None, "take_over_lenses() first")
        self._require(not self.planned, "the measurements are already planned")
        self.settings = {f"new {k}": row for k, row in enumerate(np.atleast_2d(np.asarray(settings, float)))}
        self._check_range(self.settings)
        self.planned = [(label, "x") for label in self.settings]
        self._set_kicks(beam_move_px, test_kick_units, max_kick_units)

    def dark_frame(self):
        """A frame with the beam blanked; its median is the background of every later frame."""
        self._require(self.base is not None, "take_over_lenses() first")
        self.defl.SetBeamBlank(1)
        time.sleep(0.5)
        try:
            dark = self._snap(role="dark", kind="none", offset=[0, 0], measurement=0, setting="dark", direction="x",
                              dac=list(self.base))
        finally:
            self.defl.SetBeamBlank(0)
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
        """Put the deflectors, free-lens control (IL1-PL1), beam blanking, function mode and HT back
        as they were when the recording was opened. Safe at any point."""
        for name in ("CLA1", "CLA2"):
            self._set_deflector(name, self.centre[name])
        for name, i in FLC_IDS.items():
            self.lens.SetFLCAbs(i, self.state["flc"][name][1])
            self.lens.SetFLCSw(i, self.state["flc"][name][0])
        self.defl.SetBeamBlank(self.state["beam_blank"])
        if self.eos.GetFunctionMode()[0] != self.state["function_mode"][0]:
            self.eos.SelectFunctionMode(self.state["function_mode"][0])
        if self.ht.GetHtValue() != self.state["ht_v"]:
            self.ht.SetHtValue(self.state["ht_v"])
        self.base = None
        print(f"restored: deflectors, free-lens control, blanking, function mode and HT as at opening; "
              f"{len(self.done)} of {len(self.planned)} measurements made, in {self.folder}", flush=True)

    # -- helpers ----------------------------------------------------------------------------
    def _measure_planned(self, number):
        """Planned measurement ``number`` (from 1), with the checks and the error handling."""
        self._require(self.base is not None, "take_over_lenses() first")
        self._require(self.background is not None, "dark_frame() first")
        label, direction = self.planned[number - 1]
        print(f"measurement {number} of {len(self.planned)}", flush=True)
        try:
            used = self._measure(number, label, self.settings[label], direction, centre=self.centre, **self.kicks)
        except BaseException:
            for name in ("CLA1", "CLA2"):
                self._set_deflector(name, self.centre[name])
            print(f"measurement {number} stopped; the deflectors are back at their centre. "
                  "restore() puts the rest back.", flush=True)
            raise
        self.done.add(number)
        return used

    @staticmethod
    def _require(condition, message):
        if not condition:
            raise RuntimeError(message)

    def _console_lenses(self):
        return np.array([getattr(self.lens, f"Get{name}")() for name in FLC_IDS], float)

    def _flc_values(self):
        return np.array([self.lens.GetFLCInfo(i)[1] for i in FLC_IDS.values()], float)

    def _check_range(self, settings):
        bad = {k: v for k, v in settings.items() if np.any(np.round(v) > DAC_MAX) or np.any(v < 0)}
        if bad:
            raise ValueError(f"settings outside 0-{DAC_MAX}: {bad}")

    def _set_kicks(self, beam_move_px, test_kick_units, max_kick_units):
        self.kicks = dict(beam_move_px=beam_move_px, test_kick_units=test_kick_units, max_kick_units=max_kick_units)
        self._write_plan(plan=self.planned, kicks=self.kicks)
        print(f"kicks: a test kick of {test_kick_units} units, then each +/- pair sized to move the beam about "
              f"{beam_move_px:g} px, at most {max_kick_units} units. The deflectors run 0-{DAC_MAX}; centres "
              f"CLA1 {self.centre['CLA1']}, CLA2 {self.centre['CLA2']}.", flush=True)

    def _write_plan(self, **extra):
        (self.folder / "plan.json").write_text(json.dumps(dict(
            base_dac=self.base.tolist(), diffraction=None if self.diffraction is None else self.diffraction.tolist(),
            settings={k: v.tolist() for k, v in self.settings.items()}, centre=self.centre, **extra),
            indent=1, default=_plain))
        print("planned measurements:", flush=True)
        width = max(len(label) for label, _ in self.planned)
        for number, (label, direction) in enumerate(self.planned, 1):
            print(f"  {number:2d}  {label:{width}s}  kicks along {direction}   {_format(self.settings[label])}",
                  flush=True)

    def _flc(self):
        return {n: self.lens.GetFLCInfo(i)[1] for n, i in FLC_IDS.items()}

    def _set_lenses(self, dac):
        for (name, i), value in zip(FLC_IDS.items(), dac):
            value = int(round(value))
            if not 0 <= value <= DAC_MAX:
                raise ValueError(f"{name} = {value} is outside 0-{DAC_MAX}")
            self.lens.SetFLCAbs(i, value)
        time.sleep(self.settle_s)
        return self._flc()

    def _deflectors(self):
        return {"CLA1": list(self.defl.GetCLA1()), "CLA2": list(self.defl.GetCLA2())}

    def _set_deflector(self, name, xy):
        x, y = (int(round(v)) for v in xy)
        getattr(self.defl, f"Set{name}")(x, y)

    def _snap(self, **meta):
        """Take a frame, save it, log it with the read-back state. Returns the frame."""
        frame, dtype = _read_frame(self.camera, self.dtype)
        if not self.log:
            print(f"first frame: {frame.shape[1]}x{frame.shape[0]}, read as {dtype}, values {frame.min()} to {frame.max()}"
                  " (if these look wrong, pass dtype='<u2' or similar)", flush=True)
        k = len(self.log)
        np.save(self.folder / "frames" / f"{k:04d}.npy", frame)
        self.log.append(dict(index=k, file=f"frames/{k:04d}.npy", time=time.time(), ht_v=self.ht.GetHtValue(),
                             flc=self._flc(), **self._deflectors(), **meta))
        (self.folder / "log.json").write_text(json.dumps(self.log, indent=1, default=_plain))
        return frame

    # -- one lens setting --------------------------------------------------------------------
    def _measure(self, number, label, dac, direction, *, centre, beam_move_px=50.0, test_kick_units=300,
                 max_kick_units=20000, skip_off_camera=False):
        """Set the lenses, take a frame without a kick, then for CLA1 (shift) and CLA2 (tilt) a +/-
        pair of kicks along ``direction``, each deflector back at its centre afterwards. Every
        action is printed. A test kick of ``test_kick_units`` measures how far the beam moves per
        unit, and the pair is sized to move it about ``beam_move_px``, at most ``max_kick_units``. If the beam reaches the frame edge
        the kick is halved and the pair taken again. With ``skip_off_camera``, a setting whose beam
        is cut by the frame edge (or off the camera) without any kick is skipped. Returns the kick
        sizes used (deflector units), or ``None`` if skipped."""
        readback = self._set_lenses(dac)
        print(f"{label}: {_format(dac)} through free-lens control; waited {self.settle_s:g} s", flush=True)
        meta = dict(measurement=number, setting=label, direction=direction, dac=[readback[n] for n in FLC_IDS])
        base = self._snap(role="base", kind="none", offset=[0, 0], **meta)
        c0, rms, edge0 = measure_spot(frame=base, background=self.background)
        print(f"  frame without a kick: beam at {_xy(c0)} px from the centre, rms {rms:.1f} px", flush=True)
        clipped = 1e-3                              # more signal near the frame edge than this: the beam is cut
        signal = float(np.sum(base - self.background))
        self.reference_signal = getattr(self, "reference_signal", None) or signal      # the first measurement's
        if skip_off_camera and signal < 0.05 * self.reference_signal:
            edge0 = 1.0                                              # (almost) no beam on the camera
        if edge0 > clipped and skip_off_camera:
            print(f"  skipped setting {label}: the beam is cut by the frame edge or off the camera", flush=True)
            return None
        if edge0 > clipped:
            print(f"  WARNING setting {label}: the beam touches the frame edge even without a kick "
                  f"(rms {rms:.0f} px). Condense the beam, or these frames will be dropped.", flush=True)
        unit = np.array([1, 0]) if direction == "x" else np.array([0, 1])
        used = {}
        for kind, name in (("shift", "CLA1"), ("tilt", "CLA2")):
            home = np.array(centre[name], float)
            test = int(test_kick_units)
            self._set_deflector(name, home + test * unit)
            moved = self._snap(role="test", kind=kind, offset=list(test * unit), **meta)
            self._set_deflector(name, home)
            move = measure_spot(frame=moved, background=self.background)[0] - c0
            wanted = beam_move_px * test / max(abs(move), 1e-9)
            size = int(round(min(max_kick_units, wanted)))
            why = (f"{size} units to move it about {beam_move_px:g} px" if wanted <= max_kick_units else
                   f"{size} units, the largest allowed ({beam_move_px:g} px would need {wanted:.0f})")
            print(f"  {name} ({kind}) test kick +{test} along {direction}: beam moved {abs(move):.2f} px "
                  f"-> {why}", flush=True)
            limit = int(min(*(DAC_MAX - home), *home))
            if size > limit:
                size = limit
                print(f"  {name}: kick limited to {limit} units by the deflector range (0-{DAC_MAX})", flush=True)
            for _ in range(6):
                frames = []
                for sign, role in ((+1, "plus"), (-1, "minus")):
                    self._set_deflector(name, home + sign * size * unit)
                    frames.append(self._snap(role=role, kind=kind, offset=list(sign * size * unit), **meta))
                self._set_deflector(name, home)
                spots = [measure_spot(frame=f, background=self.background) for f in frames]
                if all(spot[2] < clipped for spot in spots):
                    break
                print(f"  {name} +/-{size}: the beam reached the frame edge; halving the kick and taking the pair again",
                      flush=True)
                size = int(round(size / 2))
                for entry in self.log[-2:]:
                    entry["role"] += " (clipped, retaken)"
            print(f"  {name} ({kind}) kicks +{size} and -{size} along {direction}: beam moved {_xy(spots[0][0] - c0)} "
                  f"and {_xy(spots[1][0] - c0)} px; {name} back to {[int(v) for v in home]}", flush=True)
            used[kind] = float(size)
        return used


