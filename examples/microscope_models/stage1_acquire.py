"""Stage 1 at the microscope, through PyJEM only (needs numpy and PyJEM; nothing from TemGym).

What it does, at the HT and 40k row the microscope is at:

1. records the full microscope state;
2. switches free-lens control (FLC) on for IL1, IL2, IL3 and PL1 and reads their DAC values (the 40k row);
3. takes a dark frame (beam blanked);
4. visits 12 lens settings (8 multiples of the 40k row, the diffraction setting, setting 0 three
   times in all, and setting 0 once with y kicks). At each: a frame without a kick, a small test
   kick, then a +/- beam shift (CLA1) and a +/- beam tilt (CLA2), each scaled to move the beam
   about ``target_px``, kept on the camera;
5. saves every frame (``frames/NNNN.npy``) and a log line per frame (``log.json``, rewritten after
   every frame), and restores the lenses, deflectors and blanking at the end, even after an error.

Use at the microscope::

    from PyJEM import TEM3, detector
    import stage1_acquire as S
    S.run(TEM3, detector, "D:/stage1/2026-11-06", diffraction=[IL1, IL2, IL3, PL1])

``diffraction`` are the four DAC values read in DIFF mode (see ``stage1_at_the_microscope.md``).
:func:`read_all_lenses` reads every lens value, for the MAG / DIFF comparison.

Check on the real system before relying on it: the free-lens ids (IL1 10, IL2 11, IL3 12, PL1 14),
that CLA1 is the beam shift and CLA2 the beam tilt, and the camera data type (``dtype``; by default 2 or 4 bytes per pixel are read as signed integers).
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

    # -- one lens setting --------------------------------------------------------------------
    def visit(self, visit, label, dac, direction, *, centre, target_px, test_units, max_units, offsets=None):
        """Base frame, then for shift (CLA1) and tilt (CLA2): a test kick (unless ``offsets`` are
        given), and the +/- pair, scaled to move the beam ``target_px``. If either frame of the
        pair shows the beam cut by the frame edge, the kick is halved and the pair retaken.
        Returns the offsets used, in deflector units."""
        readback = self.set_lenses(dac)
        meta = dict(visit=visit, setting=label, direction=direction, dac=[readback[n] for n in FLC_IDS])
        base = self.snap(role="base", kind="none", offset=[0, 0], **meta)
        c0, rms, edge0 = measure_spot(base, self.background)
        clipped = 1e-3                              # more signal near the frame edge than this: the beam is cut
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


def read_all_lenses(TEM3):
    """Every lens value PyJEM can read (for the MAG / DIFF comparison). Missing getters are skipped."""
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


def run(TEM3, detector, out, *, diffraction, camera_name=None, dtype=None, settle_s=2.0, target_px=50.0,
        test_units=(300, 300), max_units=(20000, 20000)):
    """Run stage 1 (see the module docstring). ``diffraction`` = the four DIFF-mode DAC values
    (IL1, IL2, IL3, PL1). Returns the output folder."""
    s = Session(TEM3, detector, out, camera_name=camera_name, dtype=dtype, settle_s=settle_s)
    state = dict(time=time.time(), ht_v=s.ht.GetHtValue(), mag=s.eos.GetMagValue(),
                 function_mode=s.eos.GetFunctionMode(), lenses=read_all_lenses(TEM3),
                 flc={n: s.lens.GetFLCInfo(i) for n, i in FLC_IDS.items()}, **s.deflectors(),
                 beam_blank=s.defl.GetBeamBlank(), camera=s.camera.get_detectorsetting())
    (s.out / "state_before.json").write_text(json.dumps(state, indent=1, default=_plain))
    centre = s.deflectors()
    try:
        # The 40k row = what the lenses run at now, read before free-lens control is touched. It is
        # written into free-lens control before switching on, so a value left over from earlier
        # (free-lens control may keep it while switched off) cannot replace it.
        base = np.array([state["lenses"][name] for name in FLC_IDS], float)
        if any(state["flc"][name][0] for name in FLC_IDS):
            print("  note: free-lens control was already on; its values are taken as the 40k row", flush=True)
        for name, i in FLC_IDS.items():
            s.lens.SetFLCAbs(i, int(base[list(FLC_IDS).index(name)]))
            s.lens.SetFLCSw(i, 1)
        readback = np.array([s.lens.GetFLCInfo(i)[1] for i in FLC_IDS.values()], float)
        if np.any(readback != base):
            raise RuntimeError(f"free-lens control reads {readback}, expected the 40k row {base}")
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
        s.defl.SetBeamBlank(1)
        time.sleep(0.5)
        dark = s.snap(role="dark", kind="none", offset=[0, 0], visit=-1, setting="dark", direction="x",
                      dac=list(base))
        s.defl.SetBeamBlank(0)
        s.background = float(np.median(dark))
        for visit, (label, direction) in enumerate(PLAN):
            s.visit(visit, str(label), settings[label], direction, centre=centre, target_px=target_px,
                    test_units=test_units, max_units=max_units)
            print(f"visit {visit + 1}/{len(PLAN)} (setting {label}, {direction} kicks) done", flush=True)
    finally:                                                         # always put the microscope back
        for name in ("CLA1", "CLA2"):
            s.set_deflector(name, centre[name])
        for (name, i) in FLC_IDS.items():
            s.lens.SetFLCAbs(i, state["flc"][name][1])
            s.lens.SetFLCSw(i, state["flc"][name][0])
        s.defl.SetBeamBlank(state["beam_blank"])
    return s.out


def check_run(TEM3, detector, out, settings, *, offsets=(3000, 3000), camera_name=None, dtype=None, settle_s=2.0):
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


def _half_differences(session, entries):
    """Half the centroid difference of the last plus/minus frames of each kind in ``entries``."""
    out = {}
    for kind in ("shift", "tilt"):
        plus = [e for e in entries if e["kind"] == kind and e["role"] == "plus"]
        minus = [e for e in entries if e["kind"] == kind and e["role"] == "minus"]
        if plus and minus:
            frames = [np.load(session.out / e["file"]) for e in (plus[-1], minus[-1])]
            centres = [measure_spot(f, session.background)[0] for f in frames]
            out[kind] = (centres[0] - centres[1]) / 2
    return out


def deflector_test(TEM3, detector, out, focus_dac, diffraction_dac, *, units=10000, camera_name=None,
                   dtype=None, settle_s=2.0):
    """The two-minute deflector test. At the in-focus setting a pure tilt does not move the spot;
    at the diffraction setting a pure shift does not. Returns the beam movement (px) for a change
    of ``units`` in CLA1 (shift) and CLA2 (tilt) at both settings."""
    s = Session(TEM3, detector, out, camera_name=camera_name, dtype=dtype, settle_s=settle_s)
    flc = {n: s.lens.GetFLCInfo(i) for n, i in FLC_IDS.items()}
    centre = s.deflectors()
    results = {}
    try:
        for i in FLC_IDS.values():
            s.lens.SetFLCSw(i, 1)
        for k, (label, dac) in enumerate((("in focus", focus_dac), ("diffraction", diffraction_dac))):
            start = len(s.log)
            s.visit(k, label, dac, "x", centre=centre, target_px=0, test_units=(0, 0), max_units=(0, 0),
                    offsets={"shift": units, "tilt": units})
            moves = _half_differences(s, s.log[start:])
            results[label] = {"CLA1 (shift)": abs(moves["shift"]), "CLA2 (tilt)": abs(moves["tilt"])}
    finally:
        for name in ("CLA1", "CLA2"):
            s.set_deflector(name, centre[name])
        for (name, i) in FLC_IDS.items():
            s.lens.SetFLCAbs(i, flc[name][1])
            s.lens.SetFLCSw(i, flc[name][0])
    return results


def find_diffraction_il3(TEM3, detector, out, start_dac, *, span=0.3, steps=11, units=3000, camera_name=None,
                         dtype=None, settle_s=2.0, max_moves=6):
    """Fallback when DIFF mode cannot be read: from ``start_dac`` (IL1, IL2, IL3, PL1), step IL3 and
    measure how far a CLA1 (shift) change of ``units`` moves the spot. Where it does not move, A = 0:
    that is the diffraction setting.

    The window (+/- ``span`` around the start) moves on while the smallest movement sits at its edge,
    then is refined twice around the smallest. Returns the four DAC values and every (IL3, movement)."""
    s = Session(TEM3, detector, out, camera_name=camera_name, dtype=dtype, settle_s=settle_s)
    flc = {n: s.lens.GetFLCInfo(i) for n, i in FLC_IDS.items()}
    centre = s.deflectors()
    start_dac = np.asarray(start_dac, float)
    scan = []

    def measure(il3):
        dac = np.r_[start_dac[:2], round(il3), start_dac[3]]
        start = len(s.log)
        s.visit(len(scan), "IL3 scan", dac, "x", centre=centre, target_px=0, test_units=(0, 0),
                max_units=(0, 0), offsets={"shift": units, "tilt": 0})
        scan.append((float(dac[2]), abs(_half_differences(s, s.log[start:])["shift"])))

    try:
        for i in FLC_IDS.values():
            s.lens.SetFLCSw(i, 1)
        width = 2 * span * start_dac[2]
        lo = start_dac[2] - width / 2
        for move in range(max_moves + 1):                          # coarse: move the window downhill
            grid = np.clip(np.linspace(lo, lo + width, steps), 1, DAC_MAX)
            for il3 in grid:
                measure(il3)
            values = [m for _, m in scan[-steps:]]
            k = int(np.argmin(values))
            if 0 < k < steps - 1 or move == max_moves:
                break
            lo = lo - width * 0.8 if k == 0 else lo + width * 0.8
            if lo + width < 1 or lo > DAC_MAX:
                break
        for _ in range(2):                                         # refine around the smallest
            best = min(scan, key=lambda t: t[1])[0]
            step = width / (steps - 1)
            width = 2 * step
            for il3 in np.clip(np.linspace(best - step, best + step, steps), 1, DAC_MAX):
                measure(il3)
    finally:
        for name in ("CLA1", "CLA2"):
            s.set_deflector(name, centre[name])
        for (name, i) in FLC_IDS.items():
            s.lens.SetFLCAbs(i, flc[name][1])
            s.lens.SetFLCSw(i, flc[name][0])
    best = min(scan, key=lambda t: t[1])
    return np.r_[start_dac[:2], round(best[0]), start_dac[3]], scan
