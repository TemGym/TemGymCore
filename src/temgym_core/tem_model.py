"""
Unified TEM column model.

Provides a :class:`TEMModel` that bundles an illumination (condenser)
and projection (intermediate-lens / projector) sub-system into a single
serialisable object.  Both sub-systems are represented by :class:`LensSystem`,
which stores named lenses, inter-lens geometry, per-lens model coefficients
(``Gc``, ``alpha_nl`` for the non-linear focal-length model), and one or more
:class:`DialCurve` look-up tables that map a user-facing control parameter
(FWHM, magnification step, spot-size step …) to DAC codes for each lens.

Typical workflow
----------------
1. Run the export notebook to solve condenser / IL parameters.
2. ``export_tem_model_json(model, path)`` → JSON.
3. ``model = load_tem_model_json(path)`` in your interactive notebook.
4. ``model.build_full_column(brightness_nm=250, magnification=100_000)``
   returns TemGym components ready for ``plot_model`` or ``run_to_end``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import differential_evolution, minimize

from .components import Lens, NonLinearElectromagneticLens, Plane
from .transfer_matrices import propagation_matrix, lens_matrix


# ---------------------------------------------------------------------------
# DAC ↔ hex helpers
# ---------------------------------------------------------------------------

def dac_hex_to_int(hex_str: str) -> int:
    """Parse a hex DAC string like ``'0x990a'`` to an integer."""
    return int(hex_str.strip(), 16)


def dac_int_to_hex(val: int) -> str:
    """Convert an integer DAC value back to ``'0xhhhh'`` string."""
    return f"0x{int(val):04x}"


def dac_to_normalised_current(dac_int: int | float) -> float:
    """Normalise a 16-bit DAC code to [0, 1] by dividing by 2^16."""
    return float(dac_int) / 65536.0


# ---------------------------------------------------------------------------
# Focal-length ↔ synthetic-DAC conversion
# ---------------------------------------------------------------------------

def focal_from_current(I: float, gc: float, alpha_nl: float = 0.0) -> float:
    """Focal length from the non-linear lens model.

    .. math::
        f = 1 / (G_c I^2 + \\alpha I^4)
    """
    power = gc * I**2 + alpha_nl * I**4
    if power <= 0:
        return 1e9
    return 1.0 / power


def current_from_focal(f: float, gc: float, alpha_nl: float = 0.0) -> float:
    """Invert the focal model to find current ``I`` given ``f``.

    Solves  Gc·I² + α·I⁴ = 1/f  for I > 0.
    When ``alpha_nl == 0`` this is just ``I = 1/sqrt(Gc·f)``.
    """
    target_power = 1.0 / f
    if alpha_nl == 0.0:
        return np.sqrt(target_power / gc)
    # Solve the bi-quadratic  α u² + Gc u - (1/f) = 0  where u = I²
    disc = gc**2 + 4.0 * alpha_nl * target_power
    if disc < 0:
        return np.sqrt(target_power / gc)  # fallback
    u = (-gc + np.sqrt(disc)) / (2.0 * alpha_nl)
    if u < 0:
        u = target_power / gc
    return np.sqrt(u)


def focal_to_synthetic_dac(
    f: float,
    gc: float,
    alpha_nl: float = 0.0,
    *,
    i_min: float = 0.0,
    i_max: float = 1.0,
) -> int:
    """Convert a focal length to a synthetic DAC code in [0, 65535].

    The current ``I`` is found from the focal model, then linearly mapped
    from ``[i_min, i_max]`` to ``[0, 65535]``.
    """
    I = current_from_focal(f, gc, alpha_nl)
    span = max(i_max - i_min, 1e-30)
    frac = np.clip((I - i_min) / span, 0.0, 1.0)
    return int(round(frac * 65535))


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DialCurve:
    """A look-up table that maps a user control parameter to DAC codes.

    Parameters
    ----------
    name : str
        A human-readable name (``"brightness"``, ``"magnification"``…).
    control_values : np.ndarray
        Sorted 1-D array of control-parameter values (e.g. FWHM in nm, or
        magnification in ×).
    control_unit : str
        Unit label for the control parameter (``"nm"``, ``"x"``, ``"step"``).
    dac_codes : np.ndarray
        Shape ``(n_steps, n_lenses)``.  16-bit-range DAC integers.
    """

    name: str
    control_values: np.ndarray
    control_unit: str
    dac_codes: np.ndarray  # (n_steps, n_lenses)

    # -- query helpers -----------------------------------------------------

    def interpolate_dac(self, control_value: float) -> np.ndarray:
        """Log-interpolate DAC codes at an arbitrary control value."""
        log_cv = np.log(np.maximum(self.control_values, 1e-30))
        q = np.log(max(float(control_value), 1e-30))
        out = np.array([
            np.interp(q, log_cv, self.dac_codes[:, j].astype(float))
            for j in range(self.dac_codes.shape[1])
        ])
        return np.round(out).astype(int)

    def interpolate_focal_lengths(
        self,
        control_value: float,
        gc: np.ndarray,
        alpha_nl: np.ndarray,
    ) -> np.ndarray:
        """Interpolate DAC codes, then convert to focal lengths."""
        dacs = self.interpolate_dac(control_value)
        n = len(gc)
        focals = np.empty(n, dtype=float)
        for j in range(n):
            I = dac_to_normalised_current(int(dacs[j]))
            focals[j] = focal_from_current(I, float(gc[j]), float(alpha_nl[j]))
        return focals

    def dac_to_hex_str(self, lens_idx: int, step_idx: int) -> str:
        return dac_int_to_hex(int(self.dac_codes[step_idx, lens_idx]))

    # -- serialisation -----------------------------------------------------

    def as_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "control_values": self.control_values.tolist(),
            "control_unit": self.control_unit,
            "dac_codes": self.dac_codes.tolist(),
        }

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> "DialCurve":
        return cls(
            name=str(d["name"]),
            control_values=np.asarray(d["control_values"], dtype=float),
            control_unit=str(d["control_unit"]),
            dac_codes=np.asarray(d["dac_codes"], dtype=int),
        )


@dataclass
class LensSystemGeometry:
    """Ordered lens names + inter-element drifts."""

    lens_names: tuple[str, ...]
    drift_distances_m: np.ndarray  # length = n_lenses + 1

    def z_positions(self, z0: float = 0.0) -> dict[str, float]:
        """Compute the z-position of every lens and the final plane."""
        zpos: dict[str, float] = {}
        z = z0
        for i, name in enumerate(self.lens_names):
            z += float(self.drift_distances_m[i])
            zpos[name] = z
        z += float(self.drift_distances_m[-1])
        zpos["_end"] = z
        return zpos

    def as_json_dict(self) -> dict[str, Any]:
        return {
            "lens_names": list(self.lens_names),
            "drift_distances_m": self.drift_distances_m.tolist(),
        }

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> "LensSystemGeometry":
        return cls(
            lens_names=tuple(d["lens_names"]),
            drift_distances_m=np.asarray(d["drift_distances_m"], dtype=float),
        )


@dataclass
class LensModel:
    """Per-lens model coefficients and fixed/variable partitioning."""

    gc: np.ndarray          # (n_lenses,) — geometry constants
    alpha_nl: np.ndarray    # (n_lenses,) — quartic coefficients
    fixed_mask: np.ndarray  # bool (n_lenses,) — True ⇒ lens current is fixed
    fixed_currents: np.ndarray  # (n_lenses,) — current for each lens (used where fixed)

    def focal_from_current(self, I: float, lens_idx: int) -> float:
        return focal_from_current(I, float(self.gc[lens_idx]), float(self.alpha_nl[lens_idx]))

    def focal_from_dac(self, dac_int: int, lens_idx: int) -> float:
        I = dac_to_normalised_current(dac_int)
        return self.focal_from_current(I, lens_idx)

    def all_focals_from_dac(self, dac_codes: np.ndarray) -> np.ndarray:
        """Return focal lengths for all lenses given a 1-D DAC array."""
        n = len(self.gc)
        out = np.empty(n, dtype=float)
        for j in range(n):
            if self.fixed_mask[j]:
                I = float(self.fixed_currents[j])
            else:
                I = dac_to_normalised_current(int(dac_codes[j]))
            out[j] = focal_from_current(I, float(self.gc[j]), float(self.alpha_nl[j]))
        return out

    def as_json_dict(self) -> dict[str, Any]:
        return {
            "gc": self.gc.tolist(),
            "alpha_nl": self.alpha_nl.tolist(),
            "fixed_mask": self.fixed_mask.tolist(),
            "fixed_currents": self.fixed_currents.tolist(),
        }

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> "LensModel":
        return cls(
            gc=np.asarray(d["gc"], dtype=float),
            alpha_nl=np.asarray(d["alpha_nl"], dtype=float),
            fixed_mask=np.asarray(d["fixed_mask"], dtype=bool),
            fixed_currents=np.asarray(d["fixed_currents"], dtype=float),
        )


@dataclass
class LensSystem:
    """A group of lenses with geometry, model, and one or more dial curves.

    Unified representation used by both the illumination (condenser) and
    projection (intermediate-lens / projector) halves of the microscope.
    """

    name: str
    geometry: LensSystemGeometry
    model: LensModel
    dial_curves: dict[str, DialCurve] = field(default_factory=dict)

    # -- realisation -------------------------------------------------------

    def realize(
        self,
        dial_name: str,
        control_value: float,
    ) -> dict[str, Any]:
        """Realize a lens-system setting at a given dial value.

        Returns a dict with keys:
        - ``focal_lengths_m`` — per-lens focal lengths
        - ``dac_codes`` — interpolated DAC codes (1-D int array)
        - ``dac_hex`` — dict mapping lens name → hex string
        - ``control_value`` — echoed back
        """
        curve = self.dial_curves[dial_name]
        dac = curve.interpolate_dac(control_value)
        focals = self.model.all_focals_from_dac(dac)
        names = self.geometry.lens_names
        return {
            "focal_lengths_m": focals,
            "dac_codes": dac,
            "dac_hex": {names[j]: dac_int_to_hex(int(dac[j])) for j in range(len(names))},
            "control_value": float(control_value),
        }

    def build_components(
        self,
        dial_name: str,
        control_value: float,
        z0: float = 0.0,
        *,
        use_electromagnetic: bool = False,
        end_plane: bool = True,
    ) -> tuple:
        """Build a tuple of TemGym ``Component`` objects for this system.

        Parameters
        ----------
        dial_name : str
            Which dial curve to use.
        control_value : float
            Control-parameter value.
        z0 : float
            z-offset for the first drift.
        use_electromagnetic : bool
            If *True*, build :class:`NonLinearElectromagneticLens` components
            (requires ``Gc``/``alpha_nl`` and assigns ``Rc=0``).  Otherwise
            build ideal thin :class:`Lens` objects.
        end_plane : bool
            Append a :class:`Plane` at the exit z.
        """
        real = self.realize(dial_name, control_value)
        focals = real["focal_lengths_m"]
        dacs = real["dac_codes"]
        zpos = self.geometry.z_positions(z0)
        names = self.geometry.lens_names

        comps: list = []
        for j, name in enumerate(names):
            z = zpos[name]
            if use_electromagnetic:
                if self.model.fixed_mask[j]:
                    I0 = float(self.model.fixed_currents[j])
                else:
                    I0 = dac_to_normalised_current(int(dacs[j]))
                comps.append(NonLinearElectromagneticLens(
                    z=float(z),
                    I0=I0,
                    Gc=float(self.model.gc[j]),
                    Rc=0.0,
                    alpha_nl=float(self.model.alpha_nl[j]),
                ))
            else:
                comps.append(Lens(z=float(z), focal_length=float(focals[j])))
        if end_plane:
            comps.append(Plane(z=zpos["_end"]))
        return tuple(comps)

    # -- serialisation -----------------------------------------------------

    def as_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "geometry": self.geometry.as_json_dict(),
            "model": self.model.as_json_dict(),
            "dial_curves": {k: v.as_json_dict() for k, v in self.dial_curves.items()},
        }

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> "LensSystem":
        return cls(
            name=str(d["name"]),
            geometry=LensSystemGeometry.from_json_dict(d["geometry"]),
            model=LensModel.from_json_dict(d["model"]),
            dial_curves={
                k: DialCurve.from_json_dict(v)
                for k, v in d.get("dial_curves", {}).items()
            },
        )


# ---------------------------------------------------------------------------
# TEMModel — top-level bundle
# ---------------------------------------------------------------------------

@dataclass
class TEMModel:
    """Full TEM column model (illumination + projection).

    Attributes
    ----------
    model_type : str
        Descriptive tag, e.g. ``"tem_column_v1"``.
    schema_version : int
        For forward-compatible JSON evolution.
    voltage_v : float
        Accelerating voltage in volts.
    illumination : LensSystem
        Condenser sub-system (source → sample).
    projection : LensSystem
        Projection sub-system (sample → screen).
    sample_z_m : float
        z-position of the sample plane used to join the two systems.
    """

    model_type: str
    schema_version: int
    voltage_v: float
    illumination: LensSystem
    projection: LensSystem
    sample_z_m: float = 0.0

    # -- high-level builders -----------------------------------------------

    def build_full_column(
        self,
        brightness_nm: float = 250.0,
        magnification: float = 100_000.0,
        z0: float = 0.0,
    ) -> tuple:
        """Build the full component chain from source to screen.

        Parameters
        ----------
        brightness_nm : float
            Brightness dial value (FWHM in nm for the condenser).
        magnification : float
            Magnification dial value for the projection system.
        z0 : float
            Starting z-coordinate.

        Returns
        -------
        tuple[Component, ...]
        """
        # Illumination: source → sample (no end plane, we connect at sample)
        illum = self.illumination.build_components(
            "brightness", brightness_nm, z0=z0, end_plane=False,
        )
        # Projection: sample → screen
        proj = self.projection.build_components(
            "magnification", magnification, z0=self.sample_z_m,
            use_electromagnetic=True, end_plane=True,
        )
        return illum + proj

    def realize_full(
        self,
        brightness_nm: float = 250.0,
        magnification: float = 100_000.0,
    ) -> dict[str, Any]:
        """Realize both sub-systems and return a merged state dict."""
        illum = self.illumination.realize("brightness", brightness_nm)
        proj = self.projection.realize("magnification", magnification)
        return {
            "illumination": illum,
            "projection": proj,
            "brightness_nm": brightness_nm,
            "magnification": magnification,
        }

    # -- serialisation -----------------------------------------------------

    def as_json_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "schema_version": self.schema_version,
            "voltage_v": self.voltage_v,
            "illumination": self.illumination.as_json_dict(),
            "projection": self.projection.as_json_dict(),
            "sample_z_m": self.sample_z_m,
        }

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> "TEMModel":
        return cls(
            model_type=str(d["model_type"]),
            schema_version=int(d["schema_version"]),
            voltage_v=float(d["voltage_v"]),
            illumination=LensSystem.from_json_dict(d["illumination"]),
            projection=LensSystem.from_json_dict(d["projection"]),
            sample_z_m=float(d.get("sample_z_m", 0.0)),
        )


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def export_tem_model_json(model: TEMModel, path: str | Path) -> Path:
    """Serialise a :class:`TEMModel` to JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model.as_json_dict(), indent=2))
    return out


def load_tem_model_json(path: str | Path) -> TEMModel:
    """Load a :class:`TEMModel` from a JSON file."""
    payload = json.loads(Path(path).read_text())
    return TEMModel.from_json_dict(payload)


# ---------------------------------------------------------------------------
# Intermediate-lens solver (factored from il123_solution.ipynb)
# ---------------------------------------------------------------------------

# JEOL ARM-200F magnification look-up table (Mode A–D)
RAW_DAC_HEX: dict[str, dict[str, str]] = {
    "2000":    {"IL1": "0x4b4b", "IL2": "0x4b2b", "IL3": "0xaf98", "OLf": "0x625a", "OLc": "0xb2fa", "PL1": "0xfa00"},
    "2500":    {"IL1": "0x4cd7", "IL2": "0x458f", "IL3": "0xb471", "OLf": "0x562a", "OLc": "0xb2fa", "PL1": "0xfa00"},
    "3000":    {"IL1": "0x4e0b", "IL2": "0x4046", "IL3": "0xb8fd", "OLf": "0x582a", "OLc": "0xb2fa", "PL1": "0xfa00"},
    "4000":    {"IL1": "0x4fcf", "IL2": "0x3657", "IL3": "0xc0d7", "OLf": "0x656a", "OLc": "0xb2fa", "PL1": "0xfa00"},
    "5000":    {"IL1": "0x512f", "IL2": "0x2d1d", "IL3": "0xc719", "OLf": "0x6d6a", "OLc": "0xb2fa", "PL1": "0xfa00"},
    "6000":    {"IL1": "0x526e", "IL2": "0x2515", "IL3": "0xcccc", "OLf": "0x684a", "OLc": "0xb2fa", "PL1": "0xfa00"},
    "8000":    {"IL1": "0x6a31", "IL2": "0x607f", "IL3": "0xadd0", "OLf": "0x990a", "OLc": "0xb2fa", "PL1": "0xe2cb"},
    "10000":   {"IL1": "0x6fe4", "IL2": "0x5c11", "IL3": "0xb1c7", "OLf": "0x990a", "OLc": "0xb2fa", "PL1": "0xe5a6"},
    "12000":   {"IL1": "0x73f5", "IL2": "0x5925", "IL3": "0xb6fe", "OLf": "0x990a", "OLc": "0xb2fa", "PL1": "0xeb2f"},
    "15000":   {"IL1": "0x7830", "IL2": "0x56ad", "IL3": "0xc026", "OLf": "0x990a", "OLc": "0xb2fa", "PL1": "0xf548"},
    "20000":   {"IL1": "0x8057", "IL2": "0x52d6", "IL3": "0xbeb9", "OLf": "0x990a", "OLc": "0xb2fa", "PL1": "0xf9bb"},
    "25000":   {"IL1": "0x8692", "IL2": "0x50e2", "IL3": "0x9b35", "OLf": "0x990a", "OLc": "0xb2fa", "PL1": "0xffff"},
    "30000":   {"IL1": "0x9f22", "IL2": "0x611e", "IL3": "0x9b35", "OLf": "0x990a", "OLc": "0xb2fa", "PL1": "0xfa00"},
    "40000":   {"IL1": "0xa336", "IL2": "0x611e", "IL3": "0x8ad8", "OLf": "0x990a", "OLc": "0xb2fa", "PL1": "0xfa00"},
    "50000":   {"IL1": "0xa674", "IL2": "0x6498", "IL3": "0x85b7", "OLf": "0x990a", "OLc": "0xb2fa", "PL1": "0xfa00"},
    "60000":   {"IL1": "0xa810", "IL2": "0x6a48", "IL3": "0x7f59", "OLf": "0x990a", "OLc": "0xb2fa", "PL1": "0xfa00"},
    "80000":   {"IL1": "0xa8ea", "IL2": "0x7182", "IL3": "0x77d2", "OLf": "0x990a", "OLc": "0xb2fa", "PL1": "0xfa00"},
    "100000":  {"IL1": "0xa800", "IL2": "0x7d46", "IL3": "0x7205", "OLf": "0x990a", "OLc": "0xb2fa", "PL1": "0xfa00"},
    "120000":  {"IL1": "0xa7d4", "IL2": "0x7d46", "IL3": "0x6c7f", "OLf": "0x990a", "OLc": "0xb2fa", "PL1": "0xfa00"},
    "150000":  {"IL1": "0xa761", "IL2": "0x8431", "IL3": "0x65e6", "OLf": "0x990a", "OLc": "0xb2fa", "PL1": "0xfa00"},
    "200000":  {"IL1": "0xa6c5", "IL2": "0x957b", "IL3": "0x5c4a", "OLf": "0x990a", "OLc": "0xb2fa", "PL1": "0xfa00"},
    "250000":  {"IL1": "0xa6a8", "IL2": "0x957b", "IL3": "0x552b", "OLf": "0x990a", "OLc": "0xb2fa", "PL1": "0xfa00"},
    "300000":  {"IL1": "0xa608", "IL2": "0x9ec6", "IL3": "0x4ba8", "OLf": "0x990a", "OLc": "0xb2fa", "PL1": "0xfa00"},
    "400000":  {"IL1": "0xa584", "IL2": "0xad0c", "IL3": "0x3e16", "OLf": "0x990a", "OLc": "0xb2fa", "PL1": "0xfa00"},
    "500000":  {"IL1": "0xa51d", "IL2": "0xb944", "IL3": "0x3268", "OLf": "0x990a", "OLc": "0xb2fa", "PL1": "0xfa00"},
    "600000":  {"IL1": "0xa4d0", "IL2": "0xc3c7", "IL3": "0x2771", "OLf": "0x990a", "OLc": "0xb2fa", "PL1": "0xfa00"},
    "800000":  {"IL1": "0xe300", "IL2": "0xe54f", "IL3": "0x478",  "OLf": "0x990a", "OLc": "0xb2fa", "PL1": "0xfa00"},
    "1000000": {"IL1": "0xe300", "IL2": "0xbd3a", "IL3": "0x1ac9", "OLf": "0x93ca", "OLc": "0xb2fa", "PL1": "0xfa00"},
    "1200000": {"IL1": "0xe300", "IL2": "0xcd09", "IL3": "0x1ac9", "OLf": "0x93ca", "OLc": "0xb2fa", "PL1": "0xfa00"},
    "1500000": {"IL1": "0xe300", "IL2": "0xec00", "IL3": "0x0",   "OLf": "0x93ca", "OLc": "0xb2fa", "PL1": "0xfa00"},
    "2000000": {"IL1": "0xffff", "IL2": "0xf800", "IL3": "0x0",   "OLf": "0x93ca", "OLc": "0xb2fa", "PL1": "0xfa00"},
}

IL_LENS_NAMES = ("IL1", "IL2", "IL3")
AUX_LENS_NAMES = ("OLf", "OLc", "PL1")


def parse_dac_table(
    raw: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, np.ndarray]:
    """Parse the JEOL DAC hex table into structured arrays.

    Returns a dict with ``"mag"``, ``"IL1"`` … ``"PL1"`` (float arrays),
    and ``"mode"`` (string array: ``A`` / ``B`` / ``C`` / ``D``).
    """
    if raw is None:
        raw = RAW_DAC_HEX
    mags = sorted(int(k) for k in raw.keys())
    data: dict[str, Any] = {"mag": np.array(mags, dtype=float)}
    for lens in list(IL_LENS_NAMES) + list(AUX_LENS_NAMES):
        data[lens] = np.array(
            [dac_hex_to_int(raw[str(m)][lens]) for m in mags], dtype=float,
        )
    modes = []
    for m in mags:
        if m <= 6000:
            modes.append("A")
        elif m <= 25000:
            modes.append("B")
        elif m <= 600000:
            modes.append("C")
        else:
            modes.append("D")
    data["mode"] = np.array(modes)
    return data


def solve_il_system(
    raw_dac_hex: Mapping[str, Mapping[str, str]] | None = None,
    *,
    M_obj: float = 60.0,
    M_proj: float = 100.0,
    mode_filter: str = "C",
    d_il1_to_il2: float = 52.083e-3,
    d_il2_to_il3: float = 62.500e-3,
    d_il3_to_im_nominal: float | None = None,
    de_seed: int = 42,
) -> dict[str, Any]:
    """Solve the IL1/IL2/IL3 system for Mode C DAC data.

    Fits per-lens ``(G_c, alpha)`` and inter-lens distances using
    ``differential_evolution`` + ``L-BFGS-B``.

    Returns a dict with:
    - ``d_obj_to_il1``, ``d_il1_to_il2``, ``d_il2_to_il3``, ``d_il3_to_im``
    - ``gc`` — array of 3 G_c values  (IL1, IL2, IL3)
    - ``alpha_nl`` — array of 3 alpha values
    - ``dac_data`` — the parsed DAC table dict
    - ``mode_c_idx`` — indices of Mode C rows
    - ``mode_c_mag`` — magnification values for Mode C
    - ``il_mag_targets`` — M_total / (M_obj * M_proj) for Mode C
    - ``success`` — solver success flag
    """
    dac = parse_dac_table(raw_dac_hex)
    mode_c_idx = np.where(dac["mode"] == mode_filter)[0]
    mode_c_mag = dac["mag"][mode_c_idx]
    il_mag_targets = mode_c_mag / (M_obj * M_proj)

    # Projector magnification: z2/z1 where z2+z1 = 325 mm total throw
    z2_proj = 325e-3
    z1_proj = z2_proj / M_proj
    if d_il3_to_im_nominal is None:
        d_il3_to_im_nominal = 52.083e-3 - z1_proj

    def _f_nonlinear(I, gc, alpha):
        power = gc * I**2 + alpha * I**4
        return 1.0 / power if power > 0 else 1e9

    def _loss(params):
        d_obj, d12, d23, d3im, gc1, a1, gc2, a2, gc3, a3 = params
        total = 0.0
        calc_mags = []
        for i, mtgt in zip(mode_c_idx, il_mag_targets):
            f1 = _f_nonlinear(dac["IL1"][i] / 65536, gc1, a1)
            f2 = _f_nonlinear(dac["IL2"][i] / 65536, gc2, a2)
            f3 = _f_nonlinear(dac["IL3"][i] / 65536, gc3, a3)

            p1 = np.array(propagation_matrix(d_obj, xp=np), dtype=float)
            p2 = np.array(propagation_matrix(d12, xp=np), dtype=float)
            p3 = np.array(propagation_matrix(d23, xp=np), dtype=float)
            p4 = np.array(propagation_matrix(d3im, xp=np), dtype=float)

            l1 = np.array(lens_matrix(f1, xp=np), dtype=float)
            l2 = np.array(lens_matrix(f2, xp=np), dtype=float)
            l3 = np.array(lens_matrix(f3, xp=np), dtype=float)

            M = p4 @ l3 @ p3 @ l2 @ p2 @ l1 @ p1
            A = M[0, 0]
            B = M[0, 1]
            mag = abs(A)
            calc_mags.append(mag)
            total += (B * 1000)**2 * 10 + ((mag - mtgt) / mtgt)**2 * 10000

        calc_mags = np.array(calc_mags)
        diffs = np.diff(calc_mags)
        violations = diffs[diffs <= 0]
        if len(violations) > 0:
            total += np.sum(violations**2) * 1e6
        return total

    bounds = [
        (8e-3, 20e-3),
        (d_il1_to_il2 - 2e-3, d_il1_to_il2 + 2e-3),
        (d_il2_to_il3 - 2e-3, d_il2_to_il3 + 2e-3),
        (d_il3_to_im_nominal - 2e-3, d_il3_to_im_nominal + 2e-3),
        (10, 5000), (-10000, 10000),
        (10, 5000), (-10000, 10000),
        (10, 5000), (-10000, 10000),
    ]

    result_de = differential_evolution(
        _loss, bounds, maxiter=500, popsize=15,
        tol=1e-4, mutation=(0.5, 1.5), recombination=0.7, seed=de_seed,
    )
    result = minimize(_loss, result_de.x, bounds=bounds, method="L-BFGS-B")
    x = result.x

    return {
        "d_obj_to_il1": float(x[0]),
        "d_il1_to_il2": float(x[1]),
        "d_il2_to_il3": float(x[2]),
        "d_il3_to_im": float(x[3]),
        "gc": np.array([x[4], x[6], x[8]], dtype=float),
        "alpha_nl": np.array([x[5], x[7], x[9]], dtype=float),
        "dac_data": dac,
        "mode_c_idx": mode_c_idx,
        "mode_c_mag": mode_c_mag,
        "il_mag_targets": il_mag_targets,
        "M_obj": M_obj,
        "M_proj": M_proj,
        "success": bool(result.success),
        "loss": float(result.fun),
    }


# ---------------------------------------------------------------------------
# Builder helpers for the export notebook
# ---------------------------------------------------------------------------

def build_illumination_system(
    condenser_json_path: str | Path,
) -> tuple[LensSystem, float]:
    """Build the illumination :class:`LensSystem` from a condenser fit JSON.

    Returns ``(lens_system, sample_z)`` where ``sample_z`` is the absolute
    z-position of the sample plane (end of the illumination stack).
    """
    payload = json.loads(Path(condenser_json_path).read_text())
    geo = payload["geometry_m"]
    fixed = payload["fixed_focals_m"]
    fit_table = payload["fit_table"]

    # Lens stack: CL1 → CL3 → Cmini → OPL
    # The aperture plane between CL3 and Cmini is a drift only — merged into
    # the two drifts d_C3_A and d_A_Cmini.
    lens_names = ("CL1", "CL3", "Cmini", "OPL")
    # Drifts: [before CL1, CL1→CL3, CL3→Cmini, Cmini→OPL, OPL→Sample]
    drifts = np.array([
        geo["d_S_C1"],
        geo["d_C1_C3"],
        geo["d_C3_A"] + geo["d_A_Cmini"],
        geo["d_Cmini_OPL"],
        geo["d_OPL_Sp"],
    ], dtype=float)

    geometry = LensSystemGeometry(lens_names=lens_names, drift_distances_m=drifts)

    # For the condenser lenses we use simple thin-lens model.
    # Assign a G_c so that the solved focal lengths can be stored as DAC
    # codes.  We pick a reference current of 0.5 (mid-range normalised)
    # and derive G_c = 1 / (f_ref * I_ref^2) for each lens.
    I_ref = 0.5
    f_cl1 = fixed["f_CL1_m"]
    f_cmini = fixed["f_Cmini_m"]
    f_opl = fixed["f_OPL_m"]

    # For CL3, use the mid-range focal length from the fit table
    fwhm_vals = np.array([r["target_fwhm_nm"] for r in fit_table], dtype=float)
    f_cl3_vals = np.array([r["f_CL3_m"] for r in fit_table], dtype=float)
    f_cl3_mid = float(np.median(f_cl3_vals))

    gc_cl1 = 1.0 / (f_cl1 * I_ref**2)
    gc_cl3 = 1.0 / (f_cl3_mid * I_ref**2)
    gc_cmini = 1.0 / (f_cmini * I_ref**2)
    gc_opl = 1.0 / (f_opl * I_ref**2)

    gc = np.array([gc_cl1, gc_cl3, gc_cmini, gc_opl], dtype=float)
    alpha_nl = np.zeros(4, dtype=float)
    fixed_mask = np.array([True, False, True, True], dtype=bool)

    # Fixed-lens currents: chosen so f = 1/(Gc·I²) yields the right focal
    fixed_currents = np.array([I_ref, 0.0, I_ref, I_ref], dtype=float)

    model_obj = LensModel(
        gc=gc, alpha_nl=alpha_nl,
        fixed_mask=fixed_mask, fixed_currents=fixed_currents,
    )

    # Build brightness dial curve: convert f_CL3 → synthetic DAC for CL3
    # For CL3, the current range spans the solved focal lengths
    i_cl3_vals = np.array([
        current_from_focal(f, gc_cl3, 0.0) for f in f_cl3_vals
    ])
    i_min_cl3 = float(np.min(i_cl3_vals)) * 0.95
    i_max_cl3 = float(np.max(i_cl3_vals)) * 1.05

    n_steps = len(fit_table)
    n_lenses = 4
    dac_codes = np.zeros((n_steps, n_lenses), dtype=int)
    for row_idx, row in enumerate(fit_table):
        f_cl3 = row["f_CL3_m"]
        dac_codes[row_idx, 0] = focal_to_synthetic_dac(f_cl1, gc_cl1, 0.0, i_min=0.0, i_max=1.0)
        dac_codes[row_idx, 1] = focal_to_synthetic_dac(f_cl3, gc_cl3, 0.0, i_min=i_min_cl3, i_max=i_max_cl3)
        dac_codes[row_idx, 2] = focal_to_synthetic_dac(f_cmini, gc_cmini, 0.0, i_min=0.0, i_max=1.0)
        dac_codes[row_idx, 3] = focal_to_synthetic_dac(f_opl, gc_opl, 0.0, i_min=0.0, i_max=1.0)

    brightness_curve = DialCurve(
        name="brightness",
        control_values=fwhm_vals,
        control_unit="nm",
        dac_codes=dac_codes,
    )

    illum = LensSystem(
        name="illumination",
        geometry=geometry,
        model=model_obj,
        dial_curves={"brightness": brightness_curve},
    )
    sample_z = float(np.sum(drifts))
    return illum, sample_z


def build_projection_system(
    il_solution: dict[str, Any],
    *,
    d_sample_to_opl_post: float = 2.083e-3,
    d_opl_post_to_saa: float = 0.0,
    f_opl_post: float = 5.0e-3,
    f_pl1: float = 3.25e-3,
    sample_z: float = 0.0,
) -> LensSystem:
    """Build the projection :class:`LensSystem` from an IL solver result.

    Parameters
    ----------
    il_solution : dict
        Output of :func:`solve_il_system`.
    d_sample_to_opl_post : float
        Drift from sample plane to objective post-field lens.
    d_opl_post_to_saa : float
        Extra drift from OPL_post to selected-area aperture (if modelled).
    f_opl_post : float
        Fixed focal length of the objective post-field lens.
    f_pl1 : float
        Fixed focal length of the projector lens (PL1).
    sample_z : float
        z-position of the sample plane (sets the origin of this sub-system).
    """
    dac = il_solution["dac_data"]
    mode_c_idx = il_solution["mode_c_idx"]
    mode_c_mag = il_solution["mode_c_mag"]

    # Lens stack: OPL_post → IL1 → IL2 → IL3 → PL1
    lens_names = ("OPL_post", "IL1", "IL2", "IL3", "PL1")

    # Distance from SAA to IL1 is absorbed into d_opl_post → IL1
    d_opl_post_to_il1 = il_solution["d_obj_to_il1"] - d_sample_to_opl_post
    if d_opl_post_to_saa > 0:
        d_opl_post_to_il1 -= d_opl_post_to_saa

    # Projector throw: z2_proj is the PL1→screen distance
    z2_proj = 325e-3
    d_pl1_to_screen = z2_proj

    drifts = np.array([
        d_sample_to_opl_post,                          # sample → OPL_post
        max(d_opl_post_to_il1 + d_opl_post_to_saa, 1e-6),  # OPL_post → IL1
        il_solution["d_il1_to_il2"],                   # IL1 → IL2
        il_solution["d_il2_to_il3"],                   # IL2 → IL3
        il_solution["d_il3_to_im"],                    # IL3 → PL1
        d_pl1_to_screen,                               # PL1 → screen
    ], dtype=float)

    geometry = LensSystemGeometry(lens_names=lens_names, drift_distances_m=drifts)

    # G_c / alpha_nl for IL lenses come from the solver
    # For OPL_post and PL1: derive Gc from their fixed focal lengths
    # using the same I_ref = normalised-current convention as the ILs
    # For OPL_post, we use an assumed fixed current
    I_opl = dac_to_normalised_current(int(dac["OLf"][mode_c_idx[0]]))
    gc_opl = 1.0 / (f_opl_post * I_opl**2)

    I_pl1 = dac_to_normalised_current(int(dac["PL1"][mode_c_idx[0]]))
    gc_pl1 = 1.0 / (f_pl1 * I_pl1**2)

    gc = np.array([
        gc_opl,
        il_solution["gc"][0],
        il_solution["gc"][1],
        il_solution["gc"][2],
        gc_pl1,
    ], dtype=float)

    alpha_nl = np.array([
        0.0,
        il_solution["alpha_nl"][0],
        il_solution["alpha_nl"][1],
        il_solution["alpha_nl"][2],
        0.0,
    ], dtype=float)

    # OPL_post and PL1 are fixed in Mode C
    fixed_mask = np.array([True, False, False, False, True], dtype=bool)
    fixed_currents = np.array([I_opl, 0.0, 0.0, 0.0, I_pl1], dtype=float)

    model_obj = LensModel(
        gc=gc, alpha_nl=alpha_nl,
        fixed_mask=fixed_mask, fixed_currents=fixed_currents,
    )

    # Magnification dial curve — Mode C DAC codes
    n_steps = len(mode_c_idx)
    n_lenses = 5
    dac_codes = np.zeros((n_steps, n_lenses), dtype=int)
    for row_i, idx in enumerate(mode_c_idx):
        dac_codes[row_i, 0] = int(dac["OLf"][idx])
        dac_codes[row_i, 1] = int(dac["IL1"][idx])
        dac_codes[row_i, 2] = int(dac["IL2"][idx])
        dac_codes[row_i, 3] = int(dac["IL3"][idx])
        dac_codes[row_i, 4] = int(dac["PL1"][idx])

    mag_curve = DialCurve(
        name="magnification",
        control_values=mode_c_mag,
        control_unit="x",
        dac_codes=dac_codes,
    )

    proj = LensSystem(
        name="projection",
        geometry=geometry,
        model=model_obj,
        dial_curves={"magnification": mag_curve},
    )
    return proj


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "dac_hex_to_int",
    "dac_int_to_hex",
    "dac_to_normalised_current",
    "focal_from_current",
    "current_from_focal",
    "focal_to_synthetic_dac",
    "DialCurve",
    "LensSystemGeometry",
    "LensModel",
    "LensSystem",
    "TEMModel",
    "export_tem_model_json",
    "load_tem_model_json",
    "RAW_DAC_HEX",
    "parse_dac_table",
    "solve_il_system",
    "build_illumination_system",
    "build_projection_system",
]
