"""Stage 2 at the microscope, through PyJEM only (needs numpy, PyJEM and ``stage1_acquire.py``).

**Draft** (2026-11-06): written for 60 -> 80 kV, not yet run there (the virtual NEOARM covers
200 and 195 kV only).

Stage 1 gave a predictor at one HT, in deflector units. Stage 2 makes it physical: real units
(metres, radians) and the lens and camera positions. It needs two kinds of data, fitted together:

1. **Specimen frames**, at the stage-1 HT, specimen in (by hand, then :func:`stage1_acquire.take_frame`):
   a cross-grating image at the 40k row with the beam spread over the frame (gives the
   magnification |A|), and a gold ring pattern at the diffraction setting (gives the camera
   length |B|).
2. **A second HT** (:func:`run`), specimen out: at HT ``ht_v1`` the 9 stage-1 settings and a fixed
   list of 30 more, then the 30 again back at the stage-1 HT. At each, as in stage 1, a +/- beam
   shift and a +/- beam tilt of about ``target_px``. Settings whose beam is off the camera are
   skipped.

Use at the microscope, after stage 1, in MAG at the 40k row and the stage-1 HT::

    from PyJEM import TEM3, detector
    import stage2_acquire as S2
    S2.run(TEM3, detector, "D:/stage2/2026-11-10", stage1_folder="D:/stage1/2026-11-06", ht_v1=195e3)

The projector DAC values stay fixed while the HT changes (free-lens control). The condenser does
not stay condensed, so after each HT change the script re-condenses the beam with CL3 (the
brightness), as you would at the console; the beam centres do not depend on the condenser.
Everything, including the HT and CL3, is restored at the end.
"""

import json
import time
from pathlib import Path

import numpy as np

from stage1_acquire import DAC_MAX, Session, _plain, measure_spot

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


def run(TEM3, detector, out, *, stage1_folder, ht_v1, n_settings=30, ht_settle_s=60.0, camera_name=None, dtype=None,
        settle_s=2.0, target_px=50.0, test_units=(300, 300), max_units=(20000, 20000)):
    """Run the second-HT part of stage 2 (see the module docstring). Returns the output folder."""
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
