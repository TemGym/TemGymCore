"""Tools for a microscope session through PyJEM, for any experiment: read frames, find the beam,
step the selector, set lenses through free-lens control, move deflectors, and save every frame
with a log line, with the microscope put back as it was afterwards.

Shared by the experiments in ``examples/``; not part of TemGym. numpy and PyJEM only, so it can
be copied to the microscope PC next to an experiment's recording script. It works the same on the
simulated microscope (:mod:`temgym_core.pyjem_simulator`).

    session = Session(TEM3=TEM3, detector=detector, lenses=("IL1", "PL1"), folder="D:/session")
    session.set_lenses([30000, 64000])        # prints: would set IL1 41782 -> 30000; nothing changes

    with session.dry_run(False):              # only inside such a block does the microscope change
        session.take_over_lenses()            # those lenses under free-lens control, at their present values
        session.set_lenses([30000, 64000])
        beam = session.beam(note="first")     # frame saved as frames/0000.npy, logged; the beam found in it
        session.restore()                     # everything as it was when the session was opened

Check on the real system before relying on it: the free-lens ids and the camera data type
(``dtype``; by default 2 or 4 bytes per pixel are read as signed integers).
"""

import contextlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: PyJEM free-lens-control ids of the lenses (``TEM3.Lens3().SetFLCAbs(id, value)``).
LENS_IDS = {"CL1": 0, "CL2": 1, "CL3": 2, "CM": 3, "OLc": 6, "OLf": 7, "OM": 8, "OM2": 9,
            "IL1": 10, "IL2": 11, "IL3": 12, "IL4": 13, "PL1": 14, "PL2": 15, "PL3": 16}
#: Lens and deflector supplies are 16-bit DACs; a deflector's neutral value is the middle.
DAC_MIN, DAC_MAX = 0, 65535
NEUTRAL_DAC = 32768


def json_value(value):
    """JSON fallback for numpy numbers and arrays."""
    return value.tolist() if hasattr(value, "tolist") else str(value)


# ------------------------------------------------------------------------------ frames
def read_frame(camera, dtype=None):
    """One raw frame from a PyJEM camera, as integers, and the pixel type used."""
    area = camera.get_detectorsetting()["ImagingArea"]
    raw = camera.snapshot_rawdata()
    dtype = dtype or {2: "<i2", 4: "<i4"}[len(raw) // (area["Height"] * area["Width"])]
    return np.frombuffer(raw, dtype=dtype).reshape(area["Height"], area["Width"]), dtype


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


# ------------------------------------------------------------------------------ the console
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
    """Every lens value PyJEM can read, and the function mode. Missing getters are skipped."""
    lens, eos = TEM3.Lens3(), TEM3.EOS3()
    values = {}
    for name in LENS_IDS:
        try:
            values[name] = getattr(lens, f"Get{name}")()
        except Exception as error:                                   # noqa: BLE001 -- report and continue
            values[name] = f"unavailable ({type(error).__name__})"
    try:
        values["function_mode"] = eos.GetFunctionMode()
    except Exception:                                                # noqa: BLE001
        pass
    return values


# ------------------------------------------------------------------------------ a session
#: The PyJEM calls a session uses, per part of TEM3.
PYJEM_CALLS = {
    "Lens3": ["GetIL1", "GetIL2", "GetIL3", "GetPL1", "GetFLCInfo", "SetFLCAbs", "SetFLCSw"],
    "Def3": ["GetCLA1", "GetCLA2", "SetCLA1", "SetCLA2", "GetBeamBlank", "SetBeamBlank"],
    "EOS3": ["GetFunctionMode", "SelectFunctionMode", "GetMagValue", "UpSelector"],
    "HT3": ["GetHtValue"],
}


def check_pyjem(*, TEM3, detector=None):
    """Print which of the PyJEM calls a session uses this PyJEM has, and return the missing ones.
    Changes nothing. Run it first on an unfamiliar (for example older) PyJEM."""
    missing = []
    for part in PYJEM_CALLS:
        try:
            available = dir(getattr(TEM3, part)())
        except Exception as error:                                   # noqa: BLE001 -- report and continue
            print(f"TEM3.{part}: cannot be opened ({type(error).__name__}: {error})")
            missing += [f"{part}.{call}" for call in PYJEM_CALLS[part]]
            continue
        absent = [call for call in PYJEM_CALLS[part] if call not in available]
        missing += [f"{part}.{call}" for call in absent]
        print(f"TEM3.{part}: " + ("all there" if not absent else "missing " + ", ".join(absent)))
    if detector is not None:
        absent = [call for call in ["get_attached_detector", "Detector"] if not hasattr(detector, call)]
        missing += [f"detector.{call}" for call in absent]
        print("detector: " + ("all there" if not absent else "missing " + ", ".join(absent)
                              + " (use take_frame = newest_file_in(...) instead)"))
    return missing


def newest_file_in(folder, *, pattern="*", timeout_s=30.0):
    """A ``take_frame`` for a camera whose own software saves every frame into ``folder``: each call
    waits for a file newer than the call (up to ``timeout_s``) and returns it as an array.
    Reads ``.npy``, and ``.tif`` / ``.tiff`` (with the ``tifffile`` package, or Pillow)."""
    folder = Path(folder)

    def load(path):
        if path.suffix == ".npy":
            return np.load(path)
        try:
            import tifffile
            return tifffile.imread(path)
        except ImportError:
            from PIL import Image
            return np.asarray(Image.open(path))

    def take_frame():
        asked = time.time()
        while time.time() - asked < timeout_s:
            files = [f for f in folder.glob(pattern) if f.is_file() and f.stat().st_mtime > asked]
            if files:
                newest = max(files, key=lambda f: f.stat().st_mtime)
                time.sleep(0.2)                   # let the camera software finish writing it
                return load(newest)
            time.sleep(0.1)
        raise TimeoutError(f"no new frame in {folder} within {timeout_s:g} s")

    return take_frame


def _read(get, *args):
    """``get(*args)``, or a note saying it could not be read (so one missing reading does not stop a session)."""
    try:
        return get(*args)
    except Exception as error:                                       # noqa: BLE001 -- report and continue
        return f"unavailable ({type(error).__name__}: {error})"


class Session:
    """A session at the microscope: a few plain tools, each doing one thing. It is a dry run unless told otherwise.

    The session only changes the microscope inside a ``with session.dry_run(False):`` block; anywhere
    else, and inside ``with session.dry_run(True):``, it is a dry run: everything is read from the
    microscope, nothing on it changes and nothing is saved; each change the code would make is
    printed instead ("would set IL1 21534 -> 23534"). Frames are still taken (harmless), so the
    code runs through, with the beam where it is now.

    When it does change the microscope, every frame is saved (``folder/frames/NNNN.npy``) with a
    line in ``folder/log.json``, and :meth:`restore` puts the microscope back as it was at opening.
    Opening a session changes nothing: it notes the microscope's state (``state_before.json``).

    Parameters
    ----------
    TEM3 : PyJEM's (or the simulated microscope's) ``TEM3``.
    lenses : sequence of str
        The lenses driven through free-lens control (keys of :data:`LENS_IDS`).
    detector : PyJEM's ``detector``, for frames from PyJEM's camera.
    take_frame : callable, optional
        For any other camera: a function returning the current frame as an array, e.g.
        :func:`newest_file_in`. Used instead of ``detector``.
    folder : str or Path
        Where frames and the log go (a new folder).
    settle_s : float
        Wait after every lens change (s).
    """

    def __init__(self, *, TEM3, lenses, folder, detector=None, take_frame=None, settle_s=2.0, dtype=None):
        self.lens, self.defl, self.ht, self.eos = TEM3.Lens3(), TEM3.Def3(), TEM3.HT3(), TEM3.EOS3()
        if take_frame is None:                    # PyJEM's camera
            self.camera = detector.Detector(detector.get_attached_detector()[0])
            take_frame = lambda: read_frame(self.camera, self.dtype)       # noqa: E731
        else:                                     # any other camera: a function returning the frame
            self.camera = None
        self.take_frame = take_frame
        self.lenses = tuple(lenses)
        self.ids = [LENS_IDS[name] for name in self.lenses]
        self.dry, self.settle_s, self.dtype = True, settle_s, dtype     # a dry run until told otherwise
        self.dark = None                          # until dark_frame(): find_beam estimates it from each frame
        self.log = []

        # the microscope as found: what restore() puts back
        self.centre = {name: list(getattr(self.defl, f"Get{name}")()) for name in ("CLA1", "CLA2")}
        self.state = dict(
            ht_v=_read(self.ht.GetHtValue),
            mag=_read(self.eos.GetMagValue),
            function_mode=_read(self.eos.GetFunctionMode),
            lenses=read_all_lenses(TEM3=TEM3),
            flc={name: _read(self.lens.GetFLCInfo, i) for name, i in zip(self.lenses, self.ids)},
            beam_blank=_read(self.defl.GetBeamBlank),
            camera=_read(self.camera.get_detectorsetting) if self.camera else "not through PyJEM",
            **self.centre,
        )
        # what the code has set so far (a dry run only pretends, so it keeps track here too)
        self.now = dict(lenses=self.lenses_now(), CLA1=list(self.centre["CLA1"]), CLA2=list(self.centre["CLA2"]))

        ht, mag = self.state["ht_v"], self.state["mag"]
        print((f"{ht / 1e3:g} kV, " if isinstance(ht, (int, float)) else f"HT {ht}, ")
              + (f"{mag[2]}; " if isinstance(mag, (list, tuple)) else f"MAG row {mag}; ")
              + f"{self._format(self.now['lenses'])}; CLA1 {self.centre['CLA1']}, CLA2 {self.centre['CLA2']}",
              flush=True)
        self.folder = Path(folder)
        (self.folder / "frames").mkdir(parents=True, exist_ok=False)
        (self.folder / "state_before.json").write_text(json.dumps(self.state, indent=1, default=json_value))

    @contextlib.contextmanager
    def dry_run(self, dry):
        """``with session.dry_run(False):`` lets the block change the microscope; ``dry_run(True)``
        keeps it a dry run. After the block (also after an error) it is a dry run again."""
        print("DRY RUN: nothing will change" if dry else "FOR REAL: the microscope will change", flush=True)
        self.dry = bool(dry)
        try:
            yield self
        finally:
            self.dry = True

    def _format(self, dac):
        return ", ".join(f"{name} {int(round(v))}" for name, v in zip(self.lenses, dac))

    def _would(self, text):
        print(f"  would {text}", flush=True)

    # -- reading -------------------------------------------------------------------------------
    def lenses_now(self):
        """The driven lenses' values as the console runs them now. A dry run that would be in DIFF
        cannot know them (the console loads them only when it switches): blanks (NaN)."""
        if self.dry and getattr(self, "now", {}).get("mode") == "DIFF":
            print("  (dry run: the DIFF values are known only after switching to DIFF)", flush=True)
            return np.full(len(self.lenses), np.nan)
        return np.array([getattr(self.lens, f"Get{name}")() for name in self.lenses], float)

    def frame(self, **info):
        """Take a frame. Unless dry, save it with a log line holding the HT, the lenses' free-lens
        values, the deflectors and ``info``."""
        frame = self.take_frame()
        frame, dtype = (frame if isinstance(frame, tuple) else (np.asarray(frame), np.asarray(frame).dtype))
        if not getattr(self, "_seen_a_frame", False):
            self._seen_a_frame = True
            print(f"first frame: {frame.shape[1]} x {frame.shape[0]} px, read as {dtype}, values {frame.min()} to "
                  f"{frame.max()} (if bright pixels show as large negative numbers, open the session with dtype='<u2')",
                  flush=True)
        if not self.dry:
            k = len(self.log)
            np.save(self.folder / "frames" / f"{k:04d}.npy", frame)
            self.log.append(dict(index=k, file=f"frames/{k:04d}.npy", time=time.time(), ht_v=_read(self.ht.GetHtValue),
                                 flc=[_read(self.lens.GetFLCInfo, i) for i in self.ids],
                                 CLA1=_read(self.defl.GetCLA1), CLA2=_read(self.defl.GetCLA2), **info))
            (self.folder / "log.json").write_text(json.dumps(self.log, indent=1, default=json_value))
        return frame

    def beam(self, **info):
        """Take a frame (as :meth:`frame`) and find the beam in it (:func:`find_beam`, above the dark level)."""
        return find_beam(frame=self.frame(**info), background=self.dark)

    # -- changing things ------------------------------------------------------------------------
    def take_over_lenses(self):
        """Put the driven lenses under free-lens control at the values they run at now, and switch it on."""
        values = self.lenses_now()
        if self.dry:
            return self._would(f"switch free-lens control on for {', '.join(self.lenses)} at {self._format(values)}")
        for value, i in zip(values, self.ids):
            self.lens.SetFLCAbs(i, int(value))
            self.lens.SetFLCSw(i, 1)

    def set_lenses(self, dac):
        """Set the driven lenses (free-lens control), then wait ``settle_s``."""
        if self.dry and np.isnan(dac).any():      # DIFF values: unknown until switched
            self._would("set " + ", ".join(f"{n} to {'its DIFF value (unknown in a dry run)' if np.isnan(v) else int(v)}"
                                           for n, v in zip(self.lenses, dac)))
            self.now["lenses"] = np.asarray(dac, float)
            return
        dac = [int(round(v)) for v in dac]
        if any(not DAC_MIN <= v <= DAC_MAX for v in dac):
            raise ValueError(f"lens values outside {DAC_MIN}-{DAC_MAX}: {self._format(dac)}")
        if self.dry:
            changes = [f"{n} {a:.0f} -> {b}" for n, a, b in zip(self.lenses, self.now["lenses"], dac) if not a == b]
            self.now["lenses"] = np.array(dac, float)
            return self._would("set " + ", ".join(changes)) if changes else None
        for i, value in zip(self.ids, dac):
            self.lens.SetFLCAbs(i, value)
        self.now["lenses"] = np.array(dac, float)
        time.sleep(self.settle_s)

    def deflect(self, deflector, units):
        """Move ``deflector`` ("CLA1" or "CLA2") to its centre plus ``units`` (x + iy, DAC units); 0 puts it back."""
        x, y = (int(round(c + d)) for c, d in zip(self.centre[deflector], (np.real(units), np.imag(units))))
        if not (DAC_MIN <= x <= DAC_MAX and DAC_MIN <= y <= DAC_MAX):
            raise ValueError(f"{deflector} ({x}, {y}) is outside {DAC_MIN}-{DAC_MAX}")
        if self.dry:
            if units != 0:                                   # (putting it back is not worth a line)
                self._would(f"deflect {deflector} by {np.real(units):+.0f}, {np.imag(units):+.0f} units (to {[x, y]})")
            self.now[deflector] = [x, y]
            return
        getattr(self.defl, f"Set{deflector}")(x, y)
        self.now[deflector] = [x, y]

    def dark_frame(self):
        """Blank the beam, take a frame, unblank; its median is the dark level from then on."""
        if self.dry:
            return self._would("blank the beam, take a dark frame, unblank")
        self.defl.SetBeamBlank(1)
        time.sleep(0.5)
        try:
            self.dark = float(np.median(self.frame(role="dark")))
        finally:
            self.defl.SetBeamBlank(0)

    def diffraction(self, *, camera_length_cm):
        """Switch to DIFF at the first camera length of at least ``camera_length_cm``."""
        if self.dry:
            self.now["mode"] = "DIFF"
            return self._would(f"switch to DIFF at {camera_length_cm} cm")
        select_camera_length(eos=self.eos, camera_length_cm=camera_length_cm)

    def imaging(self):
        """Switch back to MAG (the row the session started at)."""
        if self.dry:
            self.now["mode"] = "MAG"
            return self._would("switch back to MAG")
        self.eos.SelectFunctionMode(0)

    def restore(self):
        """Put the deflectors, free-lens control, blanking and function mode back as at opening. (The HT
        is never changed.) Anything that could not be read at opening is left alone, except free-lens
        control, which is then switched off: the console's own lens values come back."""
        if self.dry:
            return self._would("put everything back as at opening")
        for name in ("CLA1", "CLA2"):
            getattr(self.defl, f"Set{name}")(*self.centre[name])
        for name, i in zip(self.lenses, self.ids):
            switch_and_value = self.state["flc"][name]
            if isinstance(switch_and_value, str):            # not read at opening
                self.lens.SetFLCSw(i, 0)
            else:
                self.lens.SetFLCAbs(i, switch_and_value[1])
                self.lens.SetFLCSw(i, switch_and_value[0])
        if not isinstance(self.state["beam_blank"], str):
            self.defl.SetBeamBlank(self.state["beam_blank"])
        else:
            self.defl.SetBeamBlank(0)
        function_mode = self.state["function_mode"]
        if not isinstance(function_mode, str) and self.eos.GetFunctionMode()[0] != function_mode[0]:
            self.eos.SelectFunctionMode(function_mode[0])
        print(f"restored; {len(self.log)} frames in {self.folder}", flush=True)
