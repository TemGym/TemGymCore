from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Mapping

import numpy as np

from .components import DoubleDeflector, ElectromagneticLens
from .constants import compute_Rc_from_voltage, effective_accelerating_potential, voltage_scaling_ratio


@dataclass(frozen=True)
class LensConfig:
    name: str
    z_position: float
    turns: float
    Gc: float
    Tc: float = 0.0


@dataclass(frozen=True)
class DeflectorConfig:
    """Static description of a double-deflector pair.

    Parameters
    ----------
    name : str
        Unique identifier (matches the TOML key).
    z_position : float
        Z-position of the upper (1st) deflector [m].
    spacing : float
        Gap between the two deflectors [m].
    turns : float
        Coil turns per deflector.
    Dc : float
        Deflection constant (geometry): alpha = Dc * NI / sqrt(V*) [rad].
    shift_balance_x, shift_balance_y : float
        Hardware calibration ratio for shift (2nd/1st kick). Default 1.0.
    tilt_balance_x, tilt_balance_y : float
        Hardware calibration ratio for tilt. Default 1.0.
    """
    name: str
    z_position: float
    spacing: float
    turns: float
    Dc: float
    shift_balance_x: float = 1.0
    shift_balance_y: float = 1.0
    tilt_balance_x: float = 1.0
    tilt_balance_y: float = 1.0


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


def _mode_from_toml_table(
    table: dict,
    lens_names: list[str],
    default_currents: dict[str, float] | None = None,
) -> OperatingMode:
    """Parse a TOML mode table into an :class:`OperatingMode`.

    Parameters
    ----------
    table : dict
        Must contain ``"headers"`` (list of str) and ``"values"``
        (list of lists of float).  The first header is the control
        variable (e.g. ``"spotsize"``); the remaining headers are
        lens names whose currents are given in the value columns.
    lens_names : list[str]
        Ordered list of *all* lens names in the model.  Lenses not
        mentioned in *headers* receive *default_currents* or zero.
    default_currents : dict, optional
        ``{lens_name: current}`` applied to lenses absent from the
        table headers (e.g. ``{"Obj_prefield": 1.0}``).
    """
    headers = list(table["headers"])
    values = np.asarray(table["values"], dtype=float)

    if len(headers) < 2 or values.ndim != 2 or values.shape[1] != len(headers):
        raise ValueError(
            "Invalid mode table: need >= 2 headers and matching value columns."
        )

    controls = values[:, 0]

    lens_index = {name: i for i, name in enumerate(lens_names)}
    currents = np.zeros((values.shape[0], len(lens_names)), dtype=float)

    if default_currents is not None:
        for name, current in default_currents.items():
            if name in lens_index:
                currents[:, lens_index[name]] = float(current)

    for col, name in enumerate(headers[1:], start=1):
        if name not in lens_index:
            continue
        currents[:, lens_index[name]] = values[:, col]

    max_abs = float(np.max(np.abs(currents))) if currents.size else 1.0
    full_scale = max(1.0, max_abs)
    normalized = currents / full_scale
    allow_signed = bool(np.any(currents < 0.0))

    return OperatingMode(
        control_values=controls,
        normalized_currents=normalized,
        full_scale_current=full_scale,
        allow_signed_currents=allow_signed,
    )


@dataclass
class MicroscopeModel:
    voltage: float
    lenses: tuple[LensConfig, ...]
    modes: Mapping[str, OperatingMode]
    deflectors: tuple[DeflectorConfig, ...] = ()
    reference_voltage: float | None = None
    tc_voltage_exponent: float = 0.0
    auxiliary: dict[str, Any] | None = None

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

        self.deflectors = tuple(self.deflectors)
        deflector_names = [d.name for d in self.deflectors]
        if len(set(deflector_names)) != len(deflector_names):
            raise ValueError("Deflector names must be unique.")

        self.modes = dict(self.modes)
        if len(self.modes) == 0:
            raise ValueError("modes must not be empty.")

        if self.auxiliary is None:
            self.auxiliary = {}
        else:
            self.auxiliary = dict(self.auxiliary)

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
        deflector_drives: Mapping[str, tuple[float, float, float, float]] | None = None,
    ) -> tuple[ElectromagneticLens | DoubleDeflector, ...]:
        """Build lens and deflector component instances for a given mode.

        Parameters
        ----------
        mode_name : str
            Name of the operating mode (must be a key in ``self.modes``).
        control_value : float
            Control parameter for the mode (e.g. magnification, spot size).
        deflector_drives : dict, optional
            Mapping of deflector name to ``(shift_x, shift_y, tilt_x, tilt_y)``
            drive currents [A].  Deflection angle is computed as
            ``Dc * turns * I_drive / sqrt(V*)``.  Deflectors not listed here
            are included with zero drive.

        Returns
        -------
        tuple
            Components (lenses and deflectors) sorted by z-position.
        """
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

        # Convert TOML Gc (calibrated at reference voltage) to pure geometry Gc.
        # TOML stores Gc_toml such that f = 1/(Gc_toml * NI^2) at reference voltage,
        # so Gc_geom = Gc_toml * V*_ref.
        V_star_ref = float(effective_accelerating_potential(reference_voltage))
        tc_scale = float(voltage_scaling_ratio(voltage, reference_voltage)) ** float(
            self.tc_voltage_exponent
        )

        components: list[ElectromagneticLens | DoubleDeflector] = []
        for i, lens in enumerate(self.lenses):
            Gc_geom = float(lens.Gc) * V_star_ref * float(gc_scale_mode[i])
            components.append(
                ElectromagneticLens(
                    z=float(lens.z_position),
                    turns=float(lens.turns),
                    current=float(currents[i]),
                    Gc=Gc_geom,
                    Tc=float(lens.Tc) * tc_scale,
                )
            )

        # Build deflectors with voltage-scaled angular kicks.
        # alpha = Dc * turns * I_drive / sqrt(V*)
        if deflector_drives is None:
            deflector_drives = {}
        V_star = float(effective_accelerating_potential(voltage))
        sqrt_V_star = float(V_star ** 0.5)

        for defl in self.deflectors:
            drives = deflector_drives.get(defl.name, (0.0, 0.0, 0.0, 0.0))
            scale = float(defl.Dc) * float(defl.turns) / sqrt_V_star
            components.append(
                DoubleDeflector(
                    z=float(defl.z_position),
                    spacing=float(defl.spacing),
                    shift_x=scale * float(drives[0]),
                    shift_y=scale * float(drives[1]),
                    tilt_x=scale * float(drives[2]),
                    tilt_y=scale * float(drives[3]),
                    shift_balance_x=float(defl.shift_balance_x),
                    shift_balance_y=float(defl.shift_balance_y),
                    tilt_balance_x=float(defl.tilt_balance_x),
                    tilt_balance_y=float(defl.tilt_balance_y),
                )
            )

        components.sort(key=lambda c: float(c.z))
        return tuple(components)

    def to_npz(self, filepath: str) -> None:
        import json

        data = {}

        metadata = {
            "voltage": float(self.voltage),
            "reference_voltage": float(self.reference_voltage),
            "tc_voltage_exponent": float(self.tc_voltage_exponent),
            "auxiliary": self.auxiliary,
            "lenses": [
                {
                    "name": l.name,
                    "z_position": l.z_position,
                    "turns": l.turns,
                    "Gc": l.Gc,
                    "Tc": l.Tc,
                }
                for l in self.lenses
            ],
            "deflectors": [
                {
                    "name": d.name,
                    "z_position": d.z_position,
                    "spacing": d.spacing,
                    "turns": d.turns,
                    "Dc": d.Dc,
                    "shift_balance_x": d.shift_balance_x,
                    "shift_balance_y": d.shift_balance_y,
                    "tilt_balance_x": d.tilt_balance_x,
                    "tilt_balance_y": d.tilt_balance_y,
                }
                for d in self.deflectors
            ],
            "modes": list(self.modes.keys()),
        }
        data["metadata.json"] = np.array([json.dumps(metadata)])

        for mode_name, mode in self.modes.items():
            mode_meta = {
                "full_scale_current": float(mode.full_scale_current),
                "allow_signed_currents": bool(mode.allow_signed_currents),
                "has_gc_scales": mode.gc_scales is not None,
            }
            data[f"mode_{mode_name}_meta.json"] = np.array([json.dumps(mode_meta)])
            data[f"mode_{mode_name}_control_values"] = mode.control_values
            data[f"mode_{mode_name}_normalized_currents"] = mode.normalized_currents
            if mode.gc_scales is not None:
                data[f"mode_{mode_name}_gc_scales"] = mode.gc_scales

        np.savez(filepath, **data)

    @classmethod
    def from_npz(cls, filepath: str) -> MicroscopeModel:
        import json

        with np.load(filepath) as f:
            meta = json.loads(str(f["metadata.json"][0]))

            lenses = tuple(LensConfig(**lc) for lc in meta["lenses"])
            deflectors = tuple(
                DeflectorConfig(**dc) for dc in meta.get("deflectors", [])
            )

            modes = {}
            for mode_name in meta["modes"]:
                mode_meta = json.loads(str(f[f"mode_{mode_name}_meta.json"][0]))
                control_values = f[f"mode_{mode_name}_control_values"]
                normalized_currents = f[f"mode_{mode_name}_normalized_currents"]
                gc_scales = (
                    f[f"mode_{mode_name}_gc_scales"]
                    if mode_meta["has_gc_scales"]
                    else None
                )

                modes[mode_name] = OperatingMode(
                    control_values=control_values,
                    normalized_currents=normalized_currents,
                    full_scale_current=mode_meta["full_scale_current"],
                    allow_signed_currents=mode_meta["allow_signed_currents"],
                    gc_scales=gc_scales,
                )

            return cls(
                voltage=meta["voltage"],
                lenses=lenses,
                modes=modes,
                deflectors=deflectors,
                reference_voltage=meta["reference_voltage"],
                tc_voltage_exponent=meta["tc_voltage_exponent"],
                auxiliary=meta.get("auxiliary", {}),
            )

    @classmethod
    def from_toml(
        cls,
        path: str,
        *,
        mode_names: Mapping[str, str] | None = None,
        mode_defaults: Mapping[str, dict[str, float]] | None = None,
        reference_voltage: float | None = None,
        tc_voltage_exponent: float = 0.0,
    ) -> MicroscopeModel:
        """Load a :class:`MicroscopeModel` from a TOML file.

        Parameters
        ----------
        path : str
            Path to the TOML configuration file.
        mode_names : dict, optional
            Rename modes from their dotted TOML path to a short name,
            e.g. ``{"illumination.parallel": "spot"}``.  Modes not
            listed keep their leaf name (e.g. ``"parallel"``).
        mode_defaults : dict, optional
            Per-mode default currents for lenses absent from the table
            headers.  Keyed by *final* mode name (after renaming).
            E.g. ``{"spot": {"Obj_prefield": 1.0}}``.
        reference_voltage : float, optional
            Reference voltage [V] for Gc calibration.  Defaults to the
            beam voltage read from the TOML.
        tc_voltage_exponent : float
            Tc voltage-scaling exponent (default 0).
        """
        try:
            import tomllib
        except ModuleNotFoundError:  # Python < 3.11
            import tomli as tomllib  # type: ignore[no-redef]

        with open(path, "rb") as fh:
            cfg = tomllib.load(fh)

        # -- Beam / voltage ---------------------------------------------------
        beam = cfg["beam"]
        voltage = float(beam["voltage_kV"]) * 1e3
        if reference_voltage is None:
            reference_voltage = voltage

        # -- Lenses -----------------------------------------------------------
        lenses_cfg = cfg.get("lenses", {})
        lens_names = list(lenses_cfg.keys())
        lenses = tuple(
            LensConfig(
                name=name,
                z_position=float(spec["z_m"]),
                turns=float(spec["turns"]),
                Gc=float(spec["Gc"]),
                Tc=float(spec.get("Tc", 0.0)),
            )
            for name, spec in lenses_cfg.items()
        )

        # -- Deflectors -------------------------------------------------------
        deflectors_cfg = cfg.get("deflectors", {})
        deflectors = tuple(
            DeflectorConfig(
                name=name,
                z_position=float(spec["z_m"]),
                spacing=float(spec["spacing_m"]),
                turns=float(spec["turns"]),
                Dc=float(spec["Dc"]),
                shift_balance_x=float(spec.get("shift_balance_x", 1.0)),
                shift_balance_y=float(spec.get("shift_balance_y", 1.0)),
                tilt_balance_x=float(spec.get("tilt_balance_x", 1.0)),
                tilt_balance_y=float(spec.get("tilt_balance_y", 1.0)),
            )
            for name, spec in deflectors_cfg.items()
        )

        # -- Modes ------------------------------------------------------------
        # Flatten nested tables: modes.illumination.parallel -> "illumination.parallel"
        raw_modes = cfg.get("modes", {})
        flat_modes: dict[str, dict] = {}
        for group_name, group in raw_modes.items():
            if isinstance(group, dict) and "headers" in group:
                flat_modes[group_name] = group
            else:
                for sub_name, table in group.items():
                    flat_modes[f"{group_name}.{sub_name}"] = table

        if not flat_modes:
            raise ValueError("No mode tables found in TOML [modes] section.")

        # Determine final mode names.
        if mode_names is None:
            leaves = [key.rsplit(".", 1)[-1] for key in flat_modes]
            if len(set(leaves)) == len(leaves):
                name_map = {k: k.rsplit(".", 1)[-1] for k in flat_modes}
            else:
                name_map = {k: k for k in flat_modes}
        else:
            name_map = dict(mode_names)
            for key in flat_modes:
                if key not in name_map:
                    name_map[key] = key.rsplit(".", 1)[-1]

        if mode_defaults is None:
            mode_defaults = {}

        modes: dict[str, OperatingMode] = {}
        for toml_key, table in flat_modes.items():
            final_name = name_map[toml_key]
            defaults = mode_defaults.get(final_name)
            modes[final_name] = _mode_from_toml_table(
                table, lens_names, defaults,
            )

        # -- Auxiliary (source, apertures, detector, beam params) --------------
        auxiliary: dict[str, Any] = {}

        if "z_source_m" in beam:
            auxiliary["z_source"] = float(beam["z_source_m"])

        for key in ("virtual_source_diameter_nm", "source_half_angle_mrad"):
            if key in beam:
                auxiliary[key] = float(beam[key])

        apertures_cfg = cfg.get("apertures", {})
        if apertures_cfg:
            auxiliary["apertures"] = {
                name: {
                    "z": float(spec["z_m"]),
                    "radii": [float(r) for r in spec.get("radii_m", [])],
                }
                for name, spec in apertures_cfg.items()
            }

        detector_cfg = cfg.get("detector", {})
        if detector_cfg:
            auxiliary["detector"] = {
                "z": float(detector_cfg["z_m"]),
                "pixel_size": float(detector_cfg["pixel_size_m"]),
                "shape": list(detector_cfg["shape"]),
            }

        return cls(
            voltage=voltage,
            lenses=lenses,
            modes=modes,
            deflectors=deflectors,
            reference_voltage=reference_voltage,
            tc_voltage_exponent=tc_voltage_exponent,
            auxiliary=auxiliary,
        )
