from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .components import ElectromagneticLens
from .constants import compute_Rc_from_voltage, voltage_scaling_ratio


@dataclass(frozen=True)
class LensConfig:
    name: str
    z_position: float
    turns: float
    Gc: float
    Rc: float
    Tc: float = 0.0


@dataclass
class OperatingMode:
    control_values: np.ndarray
    normalized_currents: np.ndarray
    full_scale_current: float
    allow_signed_currents: bool = False
    gc_scales: np.ndarray | None = None

    def __post_init__(self):
        control_values = np.asarray(self.control_values, dtype=float)
        normalized_currents = np.asarray(self.normalized_currents, dtype=float)
        full_scale_current = float(self.full_scale_current)
        allow_signed_currents = bool(self.allow_signed_currents)

        if control_values.ndim != 1:
            raise ValueError("control_values must be a 1D array.")
        if control_values.size == 0:
            raise ValueError("control_values must not be empty.")
        if np.any(np.diff(control_values) <= 0.0):
            raise ValueError("control_values must be strictly increasing.")

        if normalized_currents.ndim != 2:
            raise ValueError("normalized_currents must be a 2D array.")
        if normalized_currents.shape[0] != control_values.size:
            raise ValueError(
                "normalized_currents shape[0] must match len(control_values)."
            )
        if allow_signed_currents:
            if np.any((normalized_currents < -1.0) | (normalized_currents > 1.0)):
                raise ValueError(
                    "normalized_currents entries must be within [-1, 1] when "
                    "allow_signed_currents=True."
                )
        else:
            if np.any((normalized_currents < 0.0) | (normalized_currents > 1.0)):
                raise ValueError("normalized_currents entries must be within [0, 1].")

        if full_scale_current <= 0.0:
            raise ValueError("full_scale_current must be > 0.")

        gc_scales = self.gc_scales
        if gc_scales is not None:
            gc_scales = np.asarray(gc_scales, dtype=float)
            if gc_scales.ndim != 2:
                raise ValueError("gc_scales must be a 2D array when provided.")
            if gc_scales.shape != normalized_currents.shape:
                raise ValueError(
                    "gc_scales shape must match normalized_currents shape."
                )
            if np.any(~np.isfinite(gc_scales)):
                raise ValueError("gc_scales entries must be finite.")
            if np.any(gc_scales < 0.0):
                raise ValueError("gc_scales entries must be >= 0.")

        self.control_values = control_values
        self.normalized_currents = normalized_currents
        self.full_scale_current = full_scale_current
        self.allow_signed_currents = allow_signed_currents
        self.gc_scales = gc_scales

    @property
    def n_lenses(self) -> int:
        return int(self.normalized_currents.shape[1])

    def interpolate_normalized_currents(self, control_value: float) -> np.ndarray:
        value = float(control_value)
        out = np.empty(self.n_lenses, dtype=float)
        for i in range(self.n_lenses):
            out[i] = np.interp(
                value,
                self.control_values,
                self.normalized_currents[:, i],
            )
        return out

    def interpolate_currents(self, control_value: float) -> np.ndarray:
        return self.full_scale_current * self.interpolate_normalized_currents(
            control_value
        )

    def interpolate_gc_scales(self, control_value: float) -> np.ndarray:
        if self.gc_scales is None:
            return np.ones(self.n_lenses, dtype=float)

        value = float(control_value)
        out = np.empty(self.n_lenses, dtype=float)
        for i in range(self.n_lenses):
            out[i] = np.interp(
                value,
                self.control_values,
                self.gc_scales[:, i],
            )
        return out


@dataclass
class MicroscopeModel:
    voltage: float
    lenses: tuple[LensConfig, ...]
    modes: Mapping[str, OperatingMode]
    reference_voltage: float | None = None
    tc_voltage_exponent: float = 0.0

    def __post_init__(self):
        voltage = float(self.voltage)
        if voltage <= 0.0:
            raise ValueError("voltage must be > 0.")
        self.voltage = voltage

        if self.reference_voltage is None:
            self.reference_voltage = voltage
        else:
            self.reference_voltage = float(self.reference_voltage)
            if self.reference_voltage <= 0.0:
                raise ValueError("reference_voltage must be > 0.")

        self.lenses = tuple(self.lenses)
        if len(self.lenses) == 0:
            raise ValueError("lenses must not be empty.")

        lens_names = [lens.name for lens in self.lenses]
        if len(set(lens_names)) != len(lens_names):
            raise ValueError("Lens names must be unique.")

        self.modes = dict(self.modes)
        if len(self.modes) == 0:
            raise ValueError("modes must not be empty.")

        n_lenses = len(self.lenses)
        for mode_name, mode in self.modes.items():
            if mode.n_lenses != n_lenses:
                raise ValueError(
                    f"Mode '{mode_name}' has {mode.n_lenses} lenses, expected {n_lenses}."
                )

    def build_components(
        self,
        mode_name: str,
        control_value: float,
    ) -> tuple[ElectromagneticLens, ...]:
        if mode_name not in self.modes:
            available = ", ".join(sorted(self.modes.keys()))
            raise KeyError(
                f"Unknown mode '{mode_name}'. Available modes: {available}."
            )

        mode = self.modes[mode_name]
        currents = mode.interpolate_currents(control_value)
        gc_scale_mode = mode.interpolate_gc_scales(control_value)

        voltage = float(self.voltage)
        reference_voltage = float(self.reference_voltage)

        gc_scale = float(voltage_scaling_ratio(voltage, reference_voltage))
        tc_scale = gc_scale ** float(self.tc_voltage_exponent)

        rc_ref_base = float(compute_Rc_from_voltage(reference_voltage))
        rc_voltage_base = float(compute_Rc_from_voltage(voltage))
        if rc_ref_base == 0.0:
            raise ValueError("reference_voltage produced Rc=0; cannot calibrate Rc.")

        components: list[ElectromagneticLens] = []
        for i, lens in enumerate(self.lenses):
            rc_calibration = float(lens.Rc) / rc_ref_base
            components.append(
                ElectromagneticLens(
                    z=float(lens.z_position),
                    turns=float(lens.turns),
                    current=float(currents[i]),
                    Gc=float(lens.Gc) * gc_scale * float(gc_scale_mode[i]),
                    Rc=rc_calibration * rc_voltage_base,
                    Tc=float(lens.Tc) * tc_scale,
                )
            )

        return tuple(components)
