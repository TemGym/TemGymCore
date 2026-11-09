"""Stage 2's copy of the stage-1 session tools as they were (the old Session and size_step);
the stage-1 notebook uses the simpler tools of ``examples/pyjem_session.py``."""

import contextlib
import json
import time
from pathlib import Path

import numpy as np

from pyjem_session import DAC_MAX, DAC_MIN, LENS_IDS, json_value, read_all_lenses, read_frame


def size_step(*, try_step, start, largest, target_px, proportional=False, name="step"):
    """Start small, grow until the change is clear, back off if the beam is lost; returns (step, change)."""
    good, lost, step = None, None, float(min(start, largest))
    for _ in range(16):
        change, usable = try_step(step)
        print(f"    {name} {step:.0f}: beam changed {change:.2f} px" + ("" if usable else ", beam lost"), flush=True)
        if usable and (change >= 0.5 * target_px or step >= largest):
            return step, change
        if usable:
            good = (step, change)
        else:
            lost = step
        if lost is not None:
            low = good[0] if good else 0.0
            if lost - low < 0.05 * lost or lost < 1:
                break
            step = (low + lost) / 2
        elif proportional and change >= 2.0:
            step = min(step * target_px / change, largest)
        else:
            step = min(step * 2, largest)
    return good if good else (0.0, 0.0)


class Session:
    """The old session class (stage 2's copy): saves every frame and restores the microscope."""

    def __init__(self, *, TEM3, detector, folder, lenses, deflectors=("CLA1", "CLA2"), camera_name=None,
                 dtype=None, settle_s=2.0):
        self.lens, self.defl, self.ht, self.eos = TEM3.Lens3(), TEM3.Def3(), TEM3.HT3(), TEM3.EOS3()
        self.camera = detector.Detector(camera_name or detector.get_attached_detector()[0])
        self.lenses, self.deflector_names = tuple(lenses), tuple(deflectors)
        self.lens_ids = {name: LENS_IDS[name] for name in self.lenses}
        self.folder = Path(folder)
        (self.folder / "frames").mkdir(parents=True, exist_ok=False)
        self.dtype, self.settle_s, self.log = dtype, settle_s, []
        self.base = self.background = None
        self.centre = self.deflectors()
        self.state = dict(time=time.time(), ht_v=self.ht.GetHtValue(), mag=self.eos.GetMagValue(),
                          function_mode=self.eos.GetFunctionMode(), lenses=read_all_lenses(TEM3=TEM3),
                          flc={n: self.lens.GetFLCInfo(i) for n, i in self.lens_ids.items()}, **self.centre,
                          beam_blank=self.defl.GetBeamBlank(), camera=self.camera.get_detectorsetting())
        (self.folder / "state_before.json").write_text(json.dumps(self.state, indent=1, default=json_value))

    def format_dac(self, dac):
        return ", ".join(f"{name} {int(round(v))}" for name, v in zip(self.lenses, dac))

    def console_lenses(self):
        return np.array([getattr(self.lens, f"Get{name}")() for name in self.lenses], float)

    def free_lens_values(self):
        return {n: self.lens.GetFLCInfo(i)[1] for n, i in self.lens_ids.items()}

    def take_over_lenses(self):
        self._require(self.base is None, "the lenses are already under free-lens control")
        base = self.console_lenses()
        for value, i in zip(base, self.lens_ids.values()):
            self.lens.SetFLCAbs(i, int(value))
            self.lens.SetFLCSw(i, 1)
        self.base = base
        return base

    def set_lenses(self, dac):
        for (name, i), value in zip(self.lens_ids.items(), dac):
            value = int(round(value))
            if not DAC_MIN <= value <= DAC_MAX:
                raise ValueError(f"{name} = {value} is outside {DAC_MIN}-{DAC_MAX}")
            self.lens.SetFLCAbs(i, value)
        time.sleep(self.settle_s)
        return self.free_lens_values()

    def deflectors(self):
        return {name: list(getattr(self.defl, f"Get{name}")()) for name in self.deflector_names}

    def set_deflector(self, name, xy):
        x, y = (int(round(v)) for v in xy)
        getattr(self.defl, f"Set{name}")(x, y)

    def recentre_deflectors(self):
        for name in self.deflector_names:
            self.set_deflector(name, self.centre[name])

    @contextlib.contextmanager
    def blanked(self, settle_s=0.5):
        self.defl.SetBeamBlank(1)
        time.sleep(settle_s)
        try:
            yield
        finally:
            self.defl.SetBeamBlank(0)

    def snap(self, **meta):
        frame, dtype = read_frame(self.camera, self.dtype)
        k = len(self.log)
        np.save(self.folder / "frames" / f"{k:04d}.npy", frame)
        self.log.append(dict(index=k, file=f"frames/{k:04d}.npy", time=time.time(), ht_v=self.ht.GetHtValue(),
                             flc=self.free_lens_values(), **self.deflectors(), **meta))
        (self.folder / "log.json").write_text(json.dumps(self.log, indent=1, default=json_value))
        return frame

    def restore(self):
        self.recentre_deflectors()
        for name, i in self.lens_ids.items():
            self.lens.SetFLCAbs(i, self.state["flc"][name][1])
            self.lens.SetFLCSw(i, self.state["flc"][name][0])
        self.defl.SetBeamBlank(self.state["beam_blank"])
        if self.eos.GetFunctionMode()[0] != self.state["function_mode"][0]:
            self.eos.SelectFunctionMode(self.state["function_mode"][0])
        if self.ht.GetHtValue() != self.state["ht_v"]:
            self.ht.SetHtValue(self.state["ht_v"])
        self.base = None

    @staticmethod
    def _require(condition, message):
        if not condition:
            raise RuntimeError(message)
