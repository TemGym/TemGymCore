"""Drive the microscope for the projector calibration, through PyJEM only (numpy + PyJEM;
nothing from TemGym). This is the one file to copy to the microscope PC.

Start in MAG at the 40k row, free-lens control off, at the HT of the session (60 kV planned).

    from PyJEM import TEM3, detector
    import acquire

    folder = acquire.stage1(TEM3, detector, "D:/stage1/2026-11-06", camera_length_cm=40)

:func:`stage1` (no specimen, one HT):

1. records the full microscope state;
2. reads the diffraction setting: switches to DIFF at ``camera_length_cm``, reads IL1, IL2, IL3
   and PL1, and goes back to MAG;
3. reads the same four lenses at the 40k row, writes them into free-lens control (FLC) and
   switches it on;
4. takes a dark frame (beam blanked);
5. visits 12 lens settings (8 multiples of the 40k row, the diffraction setting, setting 0 three
   times in all, and setting 0 once with y kicks). At each: a frame without a kick, a small test
   kick, then a +/- beam shift (CLA1) and a +/- beam tilt (CLA2), each scaled to move the beam
   about ``target_px``, kept on the camera;
6. saves every frame (``frames/NNNN.npy``) and a log line per frame (``log.json``, rewritten after
   every frame), and restores the lenses, deflectors and blanking at the end, even after an error.

:func:`check` measures beam moves at new lens settings, to test the model's predictions.
:func:`stage2` (draft) is the second-HT part of stage 2; :func:`take_frame` takes the specimen
frames.

``camera_length_cm`` is the DIFF row of the diffraction setting (about 40 cm at 60 kV, 80 cm at
200 kV, so the gold rings fit on the camera for stage 2). To use values read by hand instead, pass
``diffraction=[IL1, IL2, IL3, PL1]``.

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

#: The 8 stage-1 settings as multiples of the 40k-row DAC values (IL1, IL2, IL3, PL1).
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

#: Visit order: setting (0-7 or "diffraction") and kick direction. Setting 0 at the start, middle
#: and end gives the wobble; the last visit repeats setting 0 with y kicks (roundness check).
PLAN = [(0, "x"), (1, "x"), (2, "x"), (3, "x"), (4, "x"), (0, "x"), (5, "x"), (6, "x"), (7, "x"),
        ("diffraction", "x"), (0, "x"), (0, "y")]


# ------------------------------------------------------------------------- frames
def measure_spot(frame, background=0.0, window=4.0):
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


# ------------------------------------------------------------------------- the session
class Session:
    """PyJEM handles, the output folder and the log."""

    def __init__(self, TEM3, detector, out, *, camera_name=None, dtype=None, settle_s=2.0):
        self.lens, self.defl, self.ht, self.eos = TEM3.Lens3(), TEM3.Def3(), TEM3.HT3(), TEM3.EOS3()
        name = camera_name or detector.get_attached_detector()[0]
        self.camera = detector.Detector(name)
        self.out = Path(out)
        (self.out / "frames").mkdir(parents=True, exist_ok=False)
        self.dtype, self.settle_s, self.log = dtype, settle_s, []
        self.background = 0.0

    # -- reading and setting ------------------------------------------------------------
    def flc(self):
        return {n: self.lens.GetFLCInfo(i)[1] for n, i in FLC_IDS.items()}

    def set_lenses(self, dac):
        for (name, i), value in zip(FLC_IDS.items(), dac):
            value = int(round(value))
            if not 0 <= value <= DAC_MAX:
                raise ValueError(f"{name} = {value} is outside 0-{DAC_MAX}")
            self.lens.SetFLCAbs(i, value)
        time.sleep(self.settle_s)
        return self.flc()

    def deflectors(self):
        return {"CLA1": list(self.defl.GetCLA1()), "CLA2": list(self.defl.GetCLA2())}

    def set_deflector(self, name, xy):
        x, y = (int(round(v)) for v in xy)
        getattr(self.defl, f"Set{name}")(x, y)

    def snap(self, **meta):
        """Take a frame, save it, log it with the read-back state. Returns the frame."""
        area = self.camera.get_detectorsetting()["ImagingArea"]
        raw = self.camera.snapshot_rawdata()
        dtype = self.dtype or {2: "<i2", 4: "<i4"}[len(raw) // (area["Height"] * area["Width"])]
        frame = np.frombuffer(raw, dtype=dtype).reshape(area["Height"], area["Width"])
        if not self.log:
            print(f"first frame: {area['Width']}x{area['Height']}, read as {dtype}, values {frame.min()} to {frame.max()}"
                  " (if these look wrong, pass dtype='<u2' or similar)", flush=True)
        k = len(self.log)
        np.save(self.out / "frames" / f"{k:04d}.npy", frame)
        self.log.append(dict(index=k, file=f"frames/{k:04d}.npy", time=time.time(), ht_v=self.ht.GetHtValue(),
                             flc=self.flc(), **self.deflectors(), **meta))
        (self.out / "log.json").write_text(json.dumps(self.log, indent=1, default=_plain))
        return frame

    # -- session bookkeeping (shared by stage 1 and stage 2) ---------------------------------------
    def save_state(self, TEM3):
        """Record the whole microscope state (``state_before.json``) and return it."""
        state = dict(time=time.time(), ht_v=self.ht.GetHtValue(), mag=self.eos.GetMagValue(),
                     function_mode=self.eos.GetFunctionMode(), lenses=read_all_lenses(TEM3),
                     flc={n: self.lens.GetFLCInfo(i) for n, i in FLC_IDS.items()}, **self.deflectors(),
                     beam_blank=self.defl.GetBeamBlank(), camera=self.camera.get_detectorsetting())
        (self.out / "state_before.json").write_text(json.dumps(state, indent=1, default=_plain))
        return state

    def take_over_lenses(self, state):
        """Put IL1-PL1 under free-lens control at the values they run at now (the 40k row), and
        return those values. They are read before free-lens control is touched and written into it
        before switching on, so a value left over from earlier cannot replace them."""
        base = np.array([state["lenses"][name] for name in FLC_IDS], float)
        if any(state["flc"][name][0] for name in FLC_IDS):
            print("  note: free-lens control was already on; its values are taken as the 40k row", flush=True)
        for k, (name, i) in enumerate(FLC_IDS.items()):
            self.lens.SetFLCAbs(i, int(base[k]))
            self.lens.SetFLCSw(i, 1)
        readback = np.array([self.lens.GetFLCInfo(i)[1] for i in FLC_IDS.values()], float)
        if np.any(readback != base):
            raise RuntimeError(f"free-lens control reads {readback}, expected the 40k row {base}")
        return base

    def dark_frame(self, dac):
        """A frame with the beam blanked; its median is the background of every later frame."""
        self.defl.SetBeamBlank(1)
        time.sleep(0.5)
        dark = self.snap(role="dark", kind="none", offset=[0, 0], visit=-1, setting="dark", direction="x",
                         dac=list(dac))
        self.defl.SetBeamBlank(0)
        self.background = float(np.median(dark))

    def restore(self, state, centre):
        """Put the deflectors, free-lens control, blanking and HT back as they were."""
        for name in ("CLA1", "CLA2"):
            self.set_deflector(name, centre[name])
        for (name, i) in FLC_IDS.items():
            self.lens.SetFLCAbs(i, state["flc"][name][1])
            self.lens.SetFLCSw(i, state["flc"][name][0])
        self.defl.SetBeamBlank(state["beam_blank"])
        if self.ht.GetHtValue() != state["ht_v"]:
            self.ht.SetHtValue(state["ht_v"])

    # -- one lens setting --------------------------------------------------------------------
    def visit(self, visit, label, dac, direction, *, centre, target_px, test_units, max_units, offsets=None,
              skip_off_camera=False):
        """Base frame, then for shift (CLA1) and tilt (CLA2): a test kick (unless ``offsets`` are
        given), and the +/- pair, scaled to move the beam ``target_px``. If either frame of the
        pair shows the beam cut by the frame edge, the kick is halved and the pair retaken.
        With ``skip_off_camera``, a setting whose beam is cut by the frame edge (or off the camera)
        without any kick is skipped. Returns the offsets used, in deflector units, or ``None`` if
        skipped."""
        readback = self.set_lenses(dac)
        meta = dict(visit=visit, setting=label, direction=direction, dac=[readback[n] for n in FLC_IDS])
        base = self.snap(role="base", kind="none", offset=[0, 0], **meta)
        c0, rms, edge0 = measure_spot(base, self.background)
        clipped = 1e-3                              # more signal near the frame edge than this: the beam is cut
        signal = float(np.sum(base - self.background))
        self.reference_signal = getattr(self, "reference_signal", None) or signal      # the first visit's
        if skip_off_camera and signal < 0.05 * self.reference_signal:
            edge0 = 1.0                                              # (almost) no beam on the camera
        if edge0 > clipped and skip_off_camera:
            print(f"  skipped setting {label}: the beam is cut by the frame edge or off the camera", flush=True)
            return None
        if edge0 > clipped:
            print(f"  WARNING visit {visit + 1} (setting {label}): the beam touches the frame edge even without a "
                  f"kick (spot rms {rms:.0f} px). Condense the beam, or these frames will be dropped.", flush=True)
        unit = np.array([1, 0]) if direction == "x" else np.array([0, 1])
        used = {}
        for kind, name, test, cap in (("shift", "CLA1", test_units[0], max_units[0]),
                                      ("tilt", "CLA2", test_units[1], max_units[1])):
            home = np.array(centre[name], float)
            if offsets is None:
                self.set_deflector(name, home + test * unit)
                moved = self.snap(role="test", kind=kind, offset=list(test * unit), **meta)
                self.set_deflector(name, home)
                per_unit = abs(measure_spot(moved, self.background)[0] - c0) / test
                size = min(cap, target_px / max(per_unit, 1e-12))
            else:
                size = offsets[kind]
            size = min(size, *(DAC_MAX - home), *home)               # stay inside the deflector range
            for _ in range(6):
                frames = []
                for sign, role in ((+1, "plus"), (-1, "minus")):
                    self.set_deflector(name, home + sign * size * unit)
                    frames.append(self.snap(role=role, kind=kind, offset=list(sign * size * unit), **meta))
                self.set_deflector(name, home)
                if offsets is not None or all(measure_spot(f, self.background)[2] < clipped for f in frames):
                    break
                size /= 2                                            # the beam reached the edge: halve, retake
                for entry in self.log[-2:]:
                    entry["role"] += " (clipped, retaken)"
            used[kind] = float(size)
        return used


def take_frame(detector, out, name, *, camera_name=None, dtype=None):
    """One camera frame with the present settings, saved as ``out/name.npy`` and returned (for the
    specimen frames of stage 2: the grating image and the gold pattern)."""
    camera = detector.Detector(camera_name or detector.get_attached_detector()[0])
    area = camera.get_detectorsetting()["ImagingArea"]
    raw = camera.snapshot_rawdata()
    dtype = dtype or {2: "<i2", 4: "<i4"}[len(raw) // (area["Height"] * area["Width"])]
    frame = np.frombuffer(raw, dtype=dtype).reshape(area["Height"], area["Width"]).astype(float)
    Path(out).mkdir(parents=True, exist_ok=True)
    np.save(Path(out) / f"{name}.npy", frame)
    return frame


def select_magnification(eos, magnification):
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


def select_camera_length(eos, camera_length_cm):
    """Switch to DIFF and step the selector up to the first camera length >= ``camera_length_cm``."""
    eos.SelectFunctionMode(4)
    for _ in range(30):
        if eos.GetMagValue()[0] >= camera_length_cm:
            break
        eos.UpSelector()


def read_diffraction_setting(TEM3, camera_length_cm=40):
    """Read IL1, IL2, IL3, PL1 at the 40k row (MAG), then in DIFF at ``camera_length_cm`` (the
    diffraction setting), and go back to MAG. Start in MAG at the 40k row, with free-lens control
    off, so the console's own values are read. Returns both as arrays of DAC values."""
    lens, eos = TEM3.Lens3(), TEM3.EOS3()
    four = lambda: np.array([getattr(lens, f"Get{name}")() for name in FLC_IDS], float)
    base = four()
    select_camera_length(eos, camera_length_cm)
    print(f"DIFF at {eos.GetMagValue()[2]}", flush=True)
    diffraction = four()
    eos.SelectFunctionMode(0)                                     # back to MAG
    return base, diffraction


def read_all_lenses(TEM3):
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


def stage1(TEM3, detector, out, *, camera_length_cm=40, diffraction=None, camera_name=None, dtype=None,
        settle_s=2.0, target_px=50.0, test_units=(300, 300), max_units=(20000, 20000)):
    """Stage 1 (see the module docstring). The diffraction setting is read in DIFF at
    ``camera_length_cm``, unless given as ``diffraction`` (IL1, IL2, IL3, PL1 DAC values).
    Returns the output folder."""
    s = Session(TEM3, detector, out, camera_name=camera_name, dtype=dtype, settle_s=settle_s)
    state = s.save_state(TEM3)
    centre = s.deflectors()
    if diffraction is None:
        _, diffraction = read_diffraction_setting(TEM3, camera_length_cm)
    try:
        base = s.take_over_lenses(state)
        settings, flipped = {}, []
        for k, m in enumerate(MULTIPLIERS):
            m = m.copy()
            over = np.round(base * m) > DAC_MAX
            if over.any():                                           # too high: use the reciprocal factor
                m[over] = 1 / m[over]
                flipped.append(k)
            settings[k] = base * m
        settings["diffraction"] = np.asarray(diffraction, float)
        bad = {k: v for k, v in settings.items() if np.any(np.round(v) > DAC_MAX) or np.any(v < 0)}
        if bad:
            raise ValueError(f"settings outside 0-{DAC_MAX}: {bad}")
        if flipped:
            print(f"settings {flipped}: a factor above the DAC range was replaced by its reciprocal", flush=True)
        (s.out / "plan.json").write_text(json.dumps(dict(base_dac=base.tolist(), diffraction=list(map(float, diffraction)),
                                                         settings={str(k): v.tolist() for k, v in settings.items()},
                                                         plan=[[str(a), b] for a, b in PLAN], centre=centre, flipped=flipped,
                                                         target_px=target_px), indent=1, default=_plain))
        s.dark_frame(base)
        for visit, (label, direction) in enumerate(PLAN):
            s.visit(visit, str(label), settings[label], direction, centre=centre, target_px=target_px,
                    test_units=test_units, max_units=max_units)
            print(f"visit {visit + 1}/{len(PLAN)} (setting {label}, {direction} kicks) done", flush=True)
    finally:                                                         # always put the microscope back
        s.restore(state, centre)
    return s.out


def check(TEM3, detector, out, settings, *, offsets=(3000, 3000), camera_name=None, dtype=None, settle_s=2.0):
    """Measure fixed +/- shift and tilt kicks (``offsets``, deflector units) at lens ``settings``
    (rows of IL1, IL2, IL3, PL1 DAC values) -- for comparing with the model's predictions."""
    s = Session(TEM3, detector, out, camera_name=camera_name, dtype=dtype, settle_s=settle_s)
    state = dict(flc={n: s.lens.GetFLCInfo(i) for n, i in FLC_IDS.items()}, beam_blank=s.defl.GetBeamBlank(),
                 camera=s.camera.get_detectorsetting(), ht_v=s.ht.GetHtValue())
    (s.out / "state_before.json").write_text(json.dumps(state, indent=1, default=_plain))
    centre = s.deflectors()
    settings = np.atleast_2d(np.asarray(settings, float))
    (s.out / "plan.json").write_text(json.dumps(dict(settings=settings.tolist(), offsets=list(offsets), centre=centre,
                                                     base_dac=settings[0].tolist(), diffraction=settings[0].tolist()), default=_plain))
    try:
        for i in FLC_IDS.values():
            s.lens.SetFLCSw(i, 1)
        s.defl.SetBeamBlank(1)
        dark = s.snap(role="dark", kind="none", offset=[0, 0], visit=-1, setting="dark", direction="x",
                      dac=list(settings[0]))
        s.defl.SetBeamBlank(0)
        s.background = float(np.median(dark))
        for k, dac in enumerate(settings):
            s.visit(k, f"check {k}", dac, "x", centre=centre, target_px=0, test_units=(0, 0), max_units=(0, 0),
                    offsets={"shift": offsets[0], "tilt": offsets[1]})
    finally:
        for name in ("CLA1", "CLA2"):
            s.set_deflector(name, centre[name])
        for (name, i) in FLC_IDS.items():
            s.lens.SetFLCAbs(i, state["flc"][name][1])
            s.lens.SetFLCSw(i, state["flc"][name][0])
        s.defl.SetBeamBlank(state["beam_blank"])
    return s.out


# ---------------------------------------------------------------------------- stage 2 (draft)
#: Free-lens id of CL3 (the brightness): re-condensing after an HT change.
CL3_ID = 2


def fixed_settings(base_dac, n=30):
    """The fixed stage-2 list: ``n`` lens settings spread evenly (a Halton sequence) over 0.1-2x
    the 40k-row DAC values, within the DAC range. Known before the session."""
    base = np.asarray(base_dac, float)
    lo, hi = 0.1 * base, np.minimum(2 * base, DAC_MAX)
    return np.round(lo + (hi - lo) * _halton(n, (2, 3, 5, 7)))


def _halton(n, bases):
    """Points 1..n of the Halton sequence (radical inverses in the given prime bases)."""
    out = np.zeros((n, len(bases)))
    for j, b in enumerate(bases):
        for i in range(1, n + 1):
            f, k, x = 1.0, i, 0.0
            while k:
                f /= b
                x += f * (k % b)
                k //= b
            out[i - 1, j] = x
    return out


def recondense(s, dac, *, span=0.06, steps=25):
    """At lens setting ``dac``, step CL3 (free-lens control) over +/- ``span`` of its present value
    and keep the value that gives the smallest spot with the beam fully on the camera. Returns it."""
    s.set_lenses(dac)
    start = s.lens.GetFLCInfo(CL3_ID)[1]
    s.lens.SetFLCAbs(CL3_ID, start)
    s.lens.SetFLCSw(CL3_ID, 1)
    trials = []
    for value in np.round(start * np.linspace(1 - span, 1 + span, steps)).astype(int):
        s.lens.SetFLCAbs(CL3_ID, int(value))
        frame = s.snap(role="condense", kind="none", offset=[0, 0], visit=-1, setting="condense", direction="x",
                       dac=list(dac), cl3=int(value))
        _, rms, edge = measure_spot(frame, s.background)
        trials.append((int(value), rms, edge, float(np.sum(frame - s.background))))
    most = max(t[3] for t in trials)
    whole = [t for t in trials if t[2] < 1e-3 and t[3] > 0.9 * most]     # the whole beam on the camera
    if not whole:
        raise RuntimeError("no CL3 value in the range puts the whole beam on the camera; condense by hand")
    value, rms = min(whole, key=lambda t: t[1])[:2]
    s.lens.SetFLCAbs(CL3_ID, value)
    print(f"  re-condensed: CL3 {start} -> {value} (spot rms {rms:.1f} px)", flush=True)
    return value


def stage2(TEM3, detector, out, *, stage1_folder, ht_v1, n_settings=30, ht_settle_s=60.0, camera_name=None, dtype=None,
        settle_s=2.0, target_px=50.0, test_units=(300, 300), max_units=(20000, 20000)):
    """**Draft.** The second-HT part of stage 2, specimen out: at HT ``ht_v1`` the 9 stage-1
    settings and a fixed list of ``n_settings`` more, then the fixed list again back at the stage-1
    HT. The projector DAC values stay fixed; after each HT change the beam is re-condensed with CL3
    (:func:`recondense`). Settings whose beam is off the camera are skipped. Everything, including
    the HT and CL3, is restored at the end. Returns the output folder."""
    plan1 = json.loads((Path(stage1_folder) / "plan.json").read_text())
    stage1 = {k: np.asarray(v, float) for k, v in plan1["settings"].items()}
    s = Session(TEM3, detector, out, camera_name=camera_name, dtype=dtype, settle_s=settle_s)
    state = s.save_state(TEM3)
    ht_v0, centre = state["ht_v"], s.deflectors()
    cl3 = s.lens.GetFLCInfo(CL3_ID)                                  # [switch, value], restored at the end
    try:
        base = s.take_over_lenses(state)
        if np.any(base != np.asarray(plan1["base_dac"])):
            print(f"  note: the 40k row reads {base}, stage 1 had {plan1['base_dac']}", flush=True)
        fixed = {f"f{k}": dac for k, dac in enumerate(fixed_settings(base, n_settings))}
        visits = [(label, ht_v1) for label in [*stage1, *fixed]] + [(label, ht_v0) for label in fixed]
        settings = {**stage1, **fixed}
        (s.out / "plan.json").write_text(json.dumps(dict(
            base_dac=base.tolist(), diffraction=plan1["diffraction"], ht_v0=ht_v0, ht_v1=ht_v1,
            settings={k: v.tolist() for k, v in settings.items()}, visits=visits, centre=centre,
            stage1_folder=str(stage1_folder), target_px=target_px), indent=1, default=_plain))
        s.dark_frame(base)
        skipped = []
        for visit, (label, ht) in enumerate(visits):
            if s.ht.GetHtValue() != ht:
                print(f"HT -> {ht / 1e3:g} kV, waiting {ht_settle_s:g} s", flush=True)
                s.ht.SetHtValue(ht)
                time.sleep(ht_settle_s)
                recondense(s, base)
            used = s.visit(visit, label, settings[label], "x", centre=centre, target_px=target_px,
                           test_units=test_units, max_units=max_units, skip_off_camera=True)
            if used is None:
                skipped.append(f"{label} at {ht / 1e3:g} kV")
        print(f"{len(visits) - len(skipped)} of {len(visits)} visits measured"
              + (f"; skipped (beam off the camera): {', '.join(skipped)}" if skipped else ""), flush=True)
    finally:                                                         # always put the microscope back
        s.lens.SetFLCAbs(CL3_ID, cl3[1])
        s.lens.SetFLCSw(CL3_ID, cl3[0])
        s.restore(state, centre)
    return s.out
