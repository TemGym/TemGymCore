"""Stage 2 at the microscope (draft), through PyJEM only: the specimen frames and the recording
at a second HT. Uses :class:`record.Recording`. Uses TemGym: nothing."""

import json
import time
from pathlib import Path

import numpy as np

from stage2.stage1_record import DAC_MAX, Recording, json_value, read_frame
from stage2.stage1_fit import measure_spot


def take_frame(*, detector, folder, name, camera_name=None, dtype=None):
    """One camera frame with the present settings, saved as ``folder/name.npy`` and returned (for the
    specimen frames of stage 2: the grating image and the gold pattern)."""
    camera = detector.Detector(camera_name or detector.get_attached_detector()[0])
    frame = read_frame(camera, dtype)[0].astype(float)
    Path(folder).mkdir(parents=True, exist_ok=True)
    np.save(Path(folder) / f"{name}.npy", frame)
    return frame


#: Free-lens id of CL3 (the brightness): re-condensing after an HT change.
CL3_ID = 2


def fixed_settings(*, base_dac, n=30):
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


def _recondense(r, dac, *, span=0.06, steps=25):
    """At lens setting ``dac``, step CL3 (free-lens control) over +/- ``span`` of its present value
    and keep the value that gives the smallest spot with the beam fully on the camera. Returns it."""
    r._set_lenses(dac)
    start = r.lens.GetFLCInfo(CL3_ID)[1]
    r.lens.SetFLCAbs(CL3_ID, start)
    r.lens.SetFLCSw(CL3_ID, 1)
    trials = []
    for value in np.round(start * np.linspace(1 - span, 1 + span, steps)).astype(int):
        r.lens.SetFLCAbs(CL3_ID, int(value))
        frame = r._snap(role="condense", kind="none", offset=[0, 0], measurement=0, setting="condense", direction="x",
                       dac=list(dac), cl3=int(value))
        _, rms, edge = measure_spot(frame=frame, background=r.background)
        trials.append((int(value), rms, edge, float(np.sum(frame - r.background))))
    most = max(t[3] for t in trials)
    whole = [t for t in trials if t[2] < 1e-3 and t[3] > 0.9 * most]     # the whole beam on the camera
    if not whole:
        raise RuntimeError("no CL3 value in the range puts the whole beam on the camera; condense by hand")
    value, rms = min(whole, key=lambda t: t[1])[:2]
    r.lens.SetFLCAbs(CL3_ID, value)
    print(f"  re-condensed: CL3 {start} -> {value} (spot rms {rms:.1f} px)", flush=True)
    return value


def record_second_ht(*, TEM3, detector, folder, first_ht_folder, second_ht_v, n_settings=30, ht_settle_s=60.0,
                     camera_name=None, dtype=None, settle_s=2.0, beam_move_px=50.0, test_kick_units=300,
                     max_kick_units=20000):
    """**Draft** (to be split into steps like :class:`Recording` when stage 2 resumes). The
    second-HT part of stage 2, specimen out: at ``second_ht_v`` the settings of the first-HT
    recording (``first_ht_folder``) and a fixed list of ``n_settings`` more; then the fixed list
    and the DIFF row again back at the first HT. The projection-system DAC values stay fixed; after each HT change
    the beam is re-condensed with CL3 (:func:`_recondense`). Settings whose beam is off the camera
    are skipped. Everything, including the HT and CL3, is restored at the end. Returns the folder."""
    plan1 = json.loads((Path(first_ht_folder) / "plan.json").read_text())
    first = {k: np.asarray(v, float) for k, v in plan1["settings"].items()}
    r = Recording(TEM3=TEM3, detector=detector, folder=folder, camera_name=camera_name, dtype=dtype, settle_s=settle_s)
    ht_v0 = r.state["ht_v"]
    cl3 = r.lens.GetFLCInfo(CL3_ID)                                  # [switch, value], restored at the end
    try:
        base = r.take_over_lenses()
        if np.any(base != np.asarray(plan1["base_dac"])):
            print(f"  note: the 40k row reads {base}, the first HT had {plan1['base_dac']}", flush=True)
        fixed = {f"f{k}": dac for k, dac in enumerate(fixed_settings(base_dac=base, n=n_settings))}
        planned = [(label, second_ht_v) for label in [*first, *fixed]] + [(label, ht_v0) for label in [*fixed, "DIFF row"]]
        settings = {**first, **fixed}
        (r.folder / "plan.json").write_text(json.dumps(dict(
            base_dac=base.tolist(), diffraction=plan1["diffraction"], ht_v0=ht_v0, ht_v1=second_ht_v,
            settings={k: v.tolist() for k, v in settings.items()}, plan=planned, centre=r.centre,
            first_ht_folder=str(first_ht_folder), beam_move_px=beam_move_px), indent=1, default=json_value))
        r.dark_frame()
        skipped = []
        for number, (label, ht) in enumerate(planned, 1):
            if r.ht.GetHtValue() != ht:
                print(f"HT -> {ht / 1e3:g} kV, waiting {ht_settle_s:g} s", flush=True)
                r.ht.SetHtValue(ht)
                time.sleep(ht_settle_s)
                _recondense(r, base)
            used = r._measure(number, label, settings[label], "x", centre=r.centre, beam_move_px=beam_move_px,
                             test_kick_units=test_kick_units, max_kick_units=max_kick_units, skip_off_camera=True)
            if used is None:
                skipped.append(f"{label} at {ht / 1e3:g} kV")
        print(f"{len(planned) - len(skipped)} of {len(planned)} measurements made"
              + (f"; skipped (beam off the camera): {', '.join(skipped)}" if skipped else ""), flush=True)
    finally:                                                         # always put the microscope back
        r.lens.SetFLCAbs(CL3_ID, cl3[1])
        r.lens.SetFLCSw(CL3_ID, cl3[0])
        r.restore()
    return r.folder
