"""PyJEM-1.4-compatible control interface for :class:`~.scope.SimulatedMicroscope`.

``pyjem_interface(scope)`` returns two module objects, ``TEM3`` and ``detector``, that
mirror ``from PyJEM import TEM3, detector`` (PyJEM 1.4.0.3782): the same class
and method names, bare return values (``int``, ``float`` or ``list``),
setters returning ``None``, and :class:`TEMError` with
``code = ErrorTypeKind * 1024 + ErrorCodeKind`` on failure. Every ``TEM3``
class shares the one simulated microscope, as the real classes share one column.

``install(scope)`` additionally registers ``PyJEM``, ``PyJEM.TEM3``,
``PyJEM.TEM3.exception``, ``PyJEM.detector`` (and ``PyJEM.offline``) in
``sys.modules`` for the duration of a ``with`` block, so unmodified PyJEM
scripts drive the simulated microscope.

Only the calls listed in this module are emulated. Any other PyJEM name raises
:class:`NotEmulatedError`, never a silent no-op. Behaviour the PyJEM
documentation leaves open (for example free-lens-control switch-on values) is
documented where it is implemented and in this package's ``README.md``.
"""

import contextlib
import functools
import enum
import sys
import types

import numpy as np

from .electronics import NEUTRAL_DAC, DACRangeError, check_dac
from .microscope import (
    APERTURE_KINDS,
    CONDENSER_APERTURE,
    DEFLECTORS,
    LENS_IDS,
    STEM_FUNCTION_MODES,
    TEM_FUNCTION_MODES,
)


class ErrorTypeKind(enum.IntEnum):
    """Subsystem part of a PyJEM error code (``code // 1024``)."""

    TEM = 1
    EOS = 2
    Lens = 3
    Def = 4
    Screen = 5
    Camera = 6
    Gonio = 7
    PASYS = 10
    HT = 16
    GUN = 17
    FEG = 18
    Filter = 19
    Stage = 20
    Detector = 21
    Apt = 22
    Scan = 23
    MDS = 24
    VACUUM = 25
    NITROGEN = 26
    Notify = 30
    General = 31


class ErrorCodeKind(enum.IntEnum):
    """Cause part of a PyJEM error code (``code % 1024``)."""

    ARG = 1
    BUSY = 3
    TIMEOUT = 10
    ALLOC = 11
    SOCKET = 12
    TEMERR = 13
    NETDATA = 14
    GUNTYPE = 16
    I_ARG = 100


class TEMError(Exception):
    """PyJEM-style error: ``code = kind * 1024 + cause``.

    Parameters
    ----------
    kind : ErrorTypeKind
        Subsystem.
    cause : ErrorCodeKind
        Cause.
    message : str
        Human-readable detail.
    """

    def __init__(self, kind, cause, message):
        self.code = int(kind) * 1024 + int(cause)
        self.kind = ErrorTypeKind(kind)
        self.cause = ErrorCodeKind(cause)
        self.label = f"[{self.kind.name}] {message}"
        super().__init__(self.label)


class NotEmulatedError(NotImplementedError, AttributeError):
    """A PyJEM name the simulated microscope does not emulate."""


class _Accessor:
    """Common behaviour of the emulated TEM3 classes."""

    _scope = None
    _kind = ErrorTypeKind.TEM

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        raise NotEmulatedError(f"PyJEM TEM3.{type(self).__name__}.{name} is not emulated by the simulated microscope.")

    def _error(self, message, cause=ErrorCodeKind.ARG):
        return TEMError(self._kind, cause, message)

    def _dac(self, value, name):
        try:
            return check_dac(value, name)
        except (DACRangeError, TypeError, ValueError) as error:
            raise self._error(str(error)) from None

    def _choice(self, value, allowed, name):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) not in allowed:
            raise self._error(f"{name}={value!r} is not one of {sorted(allowed)}.")
        return int(value)


# --------------------------------------------------------------------- lenses
class Lens3(_Accessor):
    """Lens DAC values, direct focus setters and free lens control (FLC).

    Getters return the effective DAC value: the FLC value when FLC is on for that
    lens, else a direct setting, else the selector/spot table value. Switching
    FLC on keeps the lens where it is unless ``SetFLCAbs`` has stored a value;
    a value stored while FLC is off takes effect when it is switched on.
    """

    _kind = ErrorTypeKind.Lens

    def _lens(self, lens_id):
        names = {v: k for k, v in LENS_IDS.items()}
        return names[self._choice(lens_id, names, "lens_id")]

    def GetFLCInfo(self, lens_id):
        """Return ``[switch, value]`` of free lens control for ``lens_id``."""
        name = self._lens(lens_id)
        s = self._scope
        value = s.flc_value[name] if s.flc_value[name] is not None else s.lens_dac(name)
        return [int(s.flc_switch[name]), int(value if value is not None else NEUTRAL_DAC)]

    def SetFLCAbs(self, lens_id, value):
        """Store an absolute free-lens DAC value for ``lens_id`` (0-65535)."""
        name = self._lens(lens_id)
        self._scope.flc_value[name] = self._dac(value, f"FLC {name}")

    def SetFLCRel(self, lens_id, value):
        """Change the free-lens DAC value of ``lens_id`` by a signed amount."""
        name = self._lens(lens_id)
        s = self._scope
        start = s.flc_value[name] if s.flc_value[name] is not None else s.lens_dac(name)
        if isinstance(value, bool) or int(value) != value:
            raise self._error(f"FLC step must be an integer, not {value!r}.")
        s.flc_value[name] = self._dac(int(start if start is not None else NEUTRAL_DAC) + int(value),
                                       f"FLC {name}")

    def SetFLCSw(self, lens_id, sw):
        """Switch free lens control on (1) or off (0) for ``lens_id``."""
        name = self._lens(lens_id)
        on = self._choice(sw, {0, 1}, "sw")
        s = self._scope
        if on and s.flc_value[name] is None:
            s.flc_value[name] = s.lens_dac(name)
        s.flc_switch[name] = on

    def SetFLCSwAllLens(self, sw):
        """Switch free lens control on (1) or off (0) for every lens."""
        on = self._choice(sw, {0, 1}, "sw")
        for lens_id in LENS_IDS.values():
            self.SetFLCSw(lens_id, on)

    def _direct(self, name, value):
        self._scope.direct[name] = self._dac(value, name)

    def SetCL3(self, value):
        """Set the CL3 (brightness) DAC value directly, without the MAG link."""
        self._direct("CL3", value)

    def SetOLc(self, value):
        """Set the objective coarse DAC value directly."""
        self._direct("OLc", value)

    def SetOLf(self, value):
        """Set the objective fine DAC value directly."""
        self._direct("OLf", value)

    def SetILFocus(self, value):
        """Set the IL1 (intermediate focus) DAC value directly."""
        self._direct("IL1", value)

    def SetPLFocus(self, value):
        """Set the PL1 (projector focus) DAC value directly."""
        self._direct("PL1", value)

    def SetNtrl(self, kind):
        """Return a focus to its table value: 0 brightness, 1 objective, 3 IL, 4 PL (2, 5: no-op)."""
        kind = self._choice(kind, {0, 1, 2, 3, 4, 5}, "kind")
        names = {0: ("CL3",), 1: ("OLc", "OLf"), 3: ("IL1",), 4: ("PL1",)}.get(kind, ())
        for name in names:
            self._scope.direct.pop(name, None)


def _lens_getter(name):
    def getter(self):
        dac = self._scope.lens_dac(name)
        return int(dac if dac is not None else NEUTRAL_DAC)

    getter.__name__ = f"Get{name}"
    getter.__doc__ = f"Return the effective {name} DAC value (0-65535)."
    return getter


for _name in LENS_IDS:
    setattr(Lens3, f"Get{_name}", _lens_getter(_name))


# ----------------------------------------------------------------- deflectors
class Def3(_Accessor):
    """Deflector DAC value pairs ``[x, y]``, beam blanking and neutral resets.

    Only CLA1 (beam shift) and CLA2 (beam tilt) act on the virtual beam; every
    other pair must stay neutral (32768) for a frame to be rendered.
    """

    _kind = ErrorTypeKind.Def
    _NEUTRAL_INDEX = {0: "GunA1", 1: "GunA2", 2: "CLA1", 3: "CLA2", 7: "CLs", 8: "IS1", 9: "IS2",
                      10: "SpotA", 11: "PLA", 12: "OLs", 13: "ILs"}

    def GetBeamBlank(self):
        """Return 1 if the beam is blanked, else 0."""
        return int(self._scope.beam_blank)

    def SetBeamBlank(self, sw):
        """Blank (1) or unblank (0) the beam."""
        self._scope.beam_blank = self._choice(sw, {0, 1}, "sw")

    def SetNtrl(self, index):
        """Set the deflector with this ``GetDefInfo`` index to neutral DAC values."""
        name = self._NEUTRAL_INDEX[self._choice(index, self._NEUTRAL_INDEX, "index")]
        self._scope.deflectors[name] = [NEUTRAL_DAC, NEUTRAL_DAC]


def _deflector_pair(name):
    def getter(self):
        return list(self._scope.deflectors[name])

    def setter(self, x, y):
        self._scope.deflectors[name] = [self._dac(x, f"{name} x"), self._dac(y, f"{name} y")]

    getter.__name__, setter.__name__ = f"Get{name}", f"Set{name}"
    getter.__doc__ = f"Return the {name} DAC values ``[x, y]``."
    setter.__doc__ = f"Set the {name} DAC values ``x, y`` (0-65535)."
    return getter, setter


for _name in DEFLECTORS:
    _get, _set = _deflector_pair(_name)
    setattr(Def3, f"Get{_name}", _get)
    setattr(Def3, f"Set{_name}", _set)


# ---------------------------------------------------------- electron optics
class EOS3(_Accessor):
    """Function mode, magnification selector, spot size and related selectors."""

    _kind = ErrorTypeKind.EOS

    def GetTemStemMode(self):
        """Return 0 for TEM, 1 for STEM."""
        return int(self._scope.tem_stem)

    def SelectTemStem(self, mode):
        """Select TEM (0) or STEM (1)."""
        self._scope.tem_stem = self._choice(mode, {0, 1}, "mode")
        self._scope.function_mode = 0

    def GetFunctionMode(self):
        """Return ``[mode, name]`` of the function mode."""
        modes = STEM_FUNCTION_MODES if self._scope.tem_stem else TEM_FUNCTION_MODES
        return [int(self._scope.function_mode), modes[self._scope.function_mode]]

    def SelectFunctionMode(self, mode):
        """Select a function mode (TEM: 0 MAG, 1 MAG2, 2 LowMAG, 3 SAMAG, 4 DIFF); reloads the lens tables."""
        modes = STEM_FUNCTION_MODES if self._scope.tem_stem else TEM_FUNCTION_MODES
        self._scope.function_mode = self._choice(mode, modes, "mode")
        self._scope.reload_tables()

    def _rows(self):
        """The current mode's selector table and the name of its selector attribute."""
        s = self._scope
        if s.diffraction:
            if not s.diffraction_table:
                raise self._error("this simulated microscope has no DIFF table", ErrorCodeKind.TEMERR)
            return s.diffraction_table, "camera_length_selector"
        return s.magnification_table, "selector"

    def GetMagValue(self):
        """Return ``[value, unit, label]`` of the selected row.

        MAG: ``[40000, 'X', 'X40000']``. DIFF: the camera length, ``[80, 'cm', '80cm']``
        (the unit string the real PyJEM uses in DIFF is not yet verified).
        """
        if self._scope.diffraction:
            value = int(round(self._scope.camera_length * 100))
            return [value, "cm", f"{value}cm"]
        value = int(round(self._scope.magnification))
        return [value, "X", f"X{value}"]

    def GetCurrentMagSelectorID(self):
        """Return ``[index, value]`` of the selected row (magnification, or camera length in cm in DIFF)."""
        rows, attribute = self._rows()
        return [int(getattr(self._scope, attribute)), self.GetMagValue()[0]]

    def SetSelector(self, index):
        """Select a row of the current mode's table by index (reloads the lens tables)."""
        rows, attribute = self._rows()
        setattr(self._scope, attribute, self._choice(index, range(len(rows)), "index"))
        self._scope.reload_tables()

    def UpSelector(self):
        """Select the next row (higher magnification, or longer camera length), if any."""
        rows, attribute = self._rows()
        self.SetSelector(min(getattr(self._scope, attribute) + 1, len(rows) - 1))

    def DownSelector(self):
        """Select the previous row, if any."""
        rows, attribute = self._rows()
        self.SetSelector(max(getattr(self._scope, attribute) - 1, 0))

    def GetSpotSize(self):
        """Return the spot-size index (0-7)."""
        return int(self._scope.spot)

    def SelectSpotSize(self, size):
        """Select a spot-size index (0-7; accepted, not modelled)."""
        self._scope.spot = self._choice(size, range(8), "size")
        self._scope.reload_tables()

    def GetAlpha(self):
        """Return the alpha-selector index."""
        return int(self._scope.alpha)

    def SetAlphaSelector(self, value):
        """Select an alpha index (0-8); only 0 is modelled."""
        self._scope.alpha = self._choice(value, range(9), "alpha")

    def GetProbeMode(self):
        """Return ``[mode, name]``: 0 TEM, 1 EDS, 2 NBD, 3 CBD."""
        names = {0: "TEM", 1: "EDS", 2: "NBD", 3: "CBD"}
        return [int(self._scope.probe_mode), names[self._scope.probe_mode]]

    def SelectProbMode(self, mode):
        """Select a probe mode (0 TEM, 1 EDS, 2 NBD, 3 CBD); only TEM is modelled."""
        self._scope.probe_mode = self._choice(mode, {0, 1, 2, 3}, "mode")

    def SetBrightness(self, value):
        """Turn the brightness knob: change the CL3 DAC value by ``value`` (relative)."""
        s = self._scope
        if isinstance(value, bool) or int(value) != value:
            raise self._error(f"Brightness step must be an integer, not {value!r}.")
        current = s.lens_dac("CL3")
        s.direct["CL3"] = self._dac(int(current if current is not None else NEUTRAL_DAC) + int(value), "CL3")


# ------------------------------------------------------------- high tension
class HT3(_Accessor):
    """High tension in volts; frames render only at the modelled values."""

    _kind = ErrorTypeKind.HT
    _RANGE = (200_000.0, 60_000.0)

    def GetHtValue(self):
        """Return the high tension in volts."""
        return float(self._scope.ht_v)

    def SetHtValue(self, value):
        """Set the high tension in volts."""
        top, bottom = self._RANGE
        if isinstance(value, bool) or not np.isfinite(float(value)) or not bottom <= float(value) <= top:
            raise self._error(f"High tension {value!r} V is outside {bottom:g}-{top:g} V.")
        self._scope.ht_v = float(value)

    def GetHtRange(self):
        """Return ``[maximum, minimum]`` high tension in volts."""
        return list(self._RANGE)


# ----------------------------------------------------------------- apertures
class Apt3(_Accessor):
    """Movable apertures. The condenser aperture (kind 1, ARM extension kind 0) is accepted but not
    modelled; the others must stay open."""

    _kind = ErrorTypeKind.Apt
    _EXP_TO_KIND = {0: CONDENSER_APERTURE}

    def SelectKind(self, kind):
        """Select the aperture kind for ``SetSize``/``SetPosition`` (1 CLA, 2 OLA, 3 HCA, 4 SAA, 5 ENTA, 6 EDS)."""
        self._scope.aperture_kind = self._choice(kind, APERTURE_KINDS, "kind")

    def GetKind(self):
        """Return the selected aperture kind."""
        return int(self._scope.aperture_kind)

    def SetSize(self, hole):
        """Insert hole 1-4 of the selected aperture, or 0 to open it."""
        self._scope.apertures[self._scope.aperture_kind]["size"] = self._choice(hole, range(5), "hole")

    def GetSize(self, kind):
        """Return the hole number of an aperture kind (0 = open)."""
        return int(self._scope.apertures[self._choice(kind, APERTURE_KINDS, "kind")]["size"])

    def GetPosition(self):
        """Return ``[x, y]`` of the selected aperture (0-4095)."""
        return list(self._scope.apertures[self._scope.aperture_kind]["position"])

    def SetPosition(self, x, y):
        """Move the selected aperture to ``x, y`` (0-4095)."""
        position = []
        for value, name in ((x, "x"), (y, "y")):
            if isinstance(value, bool) or int(value) != value or not 0 <= int(value) <= 4095:
                raise self._error(f"Aperture {name}={value!r} is outside 0-4095.")
            position.append(int(value))
        self._scope.apertures[self._scope.aperture_kind]["position"] = position

    def SelectExpKind(self, kind):
        """ARM extension: select an aperture kind (0 CL1 ... 11 AUX4); only 0 is accepted."""
        kind = self._choice(kind, range(12), "kind")
        self._scope.exp_kind = kind
        if kind in self._EXP_TO_KIND:
            self._scope.aperture_kind = self._EXP_TO_KIND[kind]

    def GetExpKind(self):
        """ARM extension: return the selected extension kind."""
        return int(self._scope.exp_kind)

    def SetExpSize(self, kind, hole):
        """ARM extension: insert ``hole`` (0 = open) of extension kind ``kind``."""
        kind = self._choice(kind, range(12), "kind")
        if kind not in self._EXP_TO_KIND:
            raise NotEmulatedError(f"PyJEM Apt3 extension kind {kind} is not modelled.")
        self._scope.apertures[self._EXP_TO_KIND[kind]]["size"] = self._choice(hole, range(5), "hole")

    def GetExpSize(self, kind):
        """ARM extension: return the hole of extension kind ``kind``."""
        kind = self._choice(kind, range(12), "kind")
        if kind not in self._EXP_TO_KIND:
            raise NotEmulatedError(f"PyJEM Apt3 extension kind {kind} is not modelled.")
        return int(self._scope.apertures[self._EXP_TO_KIND[kind]]["size"])


# --------------------------------------------------------------------- stage
class Stage3(_Accessor):
    """Specimen stage in nm and degrees; it does not affect vacuum-hole frames."""

    _kind = ErrorTypeKind.Stage
    _LIMITS = (100_000.0, 100_000.0, 100_000.0, 90.0, 90.0)

    def GetPos(self):
        """Return ``[x, y, z, tilt_x, tilt_y]`` in nm and degrees."""
        return [float(v) for v in self._scope.stage]

    def GetStatus(self):
        """Return the five axis states (0 rest, 1 moving, 2 limit); moves are instant."""
        return [0, 0, 0, 0, 0]

    def Stop(self):
        """Stop all axes (moves are instant)."""

    def _move(self, axis, value, relative):
        if isinstance(value, bool) or not np.isfinite(float(value)):
            raise self._error(f"Stage value {value!r} is not a finite number.")
        target = self._scope.stage[axis] * relative + float(value)
        if abs(target) > self._LIMITS[axis]:
            raise self._error(f"Stage axis {axis} target {target:g} exceeds ±{self._LIMITS[axis]:g}.")
        self._scope.stage[axis] = target

    def SetX(self, value):
        """Move x to ``value`` nm."""
        self._move(0, value, False)

    def SetY(self, value):
        """Move y to ``value`` nm."""
        self._move(1, value, False)

    def SetZ(self, value):
        """Move z to ``value`` nm."""
        self._move(2, value, False)

    def SetTiltXAngle(self, value):
        """Tilt about x to ``value`` degrees."""
        self._move(3, value, False)

    def SetTiltYAngle(self, value):
        """Tilt about y to ``value`` degrees."""
        self._move(4, value, False)

    def SetXRel(self, value):
        """Move x by ``value`` nm."""
        self._move(0, value, True)

    def SetYRel(self, value):
        """Move y by ``value`` nm."""
        self._move(1, value, True)

    def SetZRel(self, value):
        """Move z by ``value`` nm."""
        self._move(2, value, True)

    def SetTXRel(self, value):
        """Tilt about x by ``value`` degrees."""
        self._move(3, value, True)

    def SetTYRel(self, value):
        """Tilt about y by ``value`` degrees."""
        self._move(4, value, True)


# ------------------------------------------------------------------ detector
class Detector:
    """PyJEM ``detector.Detector(name)`` for the virtual camera.

    Parameters
    ----------
    name : str
        Detector name, as returned by ``get_attached_detector()``.
    """

    _scope = None

    def __init__(self, name):
        if name != self._scope.camera.name:
            raise ValueError(f"No detector named {name!r}; attached: [{self._scope.camera.name!r}].")
        self.name = name

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        raise NotEmulatedError(f"PyJEM detector.Detector.{name} is not emulated by the simulated microscope.")

    def get_detectorsetting(self):
        """Return the detector settings dictionary (PyJEM keys)."""
        s, camera = self._scope, self._scope.camera
        x, y, width, height = s.detector["area"]
        _, pitch, _ = camera.geometry(s.detector["binning"])
        return {
            "ImagingArea": {"X": x, "Y": y, "Width": width, "Height": height},
            "BinningSize": s.detector["binning"],
            "ExposureTimeValue": s.detector["exposure_us"],
            "GainIndex": s.detector["gain_index"],
            "OffsetIndex": s.detector["offset_index"],
            "frameIntegration": s.detector["frames"],
            "scanMode": 0,
            "OutputImageInformation": {"DataBits": camera.data_bits,
                                       "ImageSize": width * height * camera.dtype.itemsize,
                                       "PixelsPerMeter": 1.0 / pitch},
        }

    def set_detectorsetting(self, setting):
        """Apply the given keys of a settings dictionary; return the new settings."""
        if "BinningSize" in setting:
            self._set_binning(setting["BinningSize"])
        if "ImagingArea" in setting:
            area = setting["ImagingArea"]
            self.set_imaging_area(area["Width"], area["Height"], area["X"], area["Y"])
        if "ExposureTimeValue" in setting:
            self.set_exposuretime_value(setting["ExposureTimeValue"])
        if "frameIntegration" in setting:
            frames = setting["frameIntegration"]
            if isinstance(frames, bool) or int(frames) != frames or int(frames) < 1:
                raise ValueError(f"frameIntegration must be a positive integer, not {frames!r}.")
            self._scope.detector["frames"] = int(frames)
        return self.get_detectorsetting()

    def _set_binning(self, binning):
        shape, _, _ = self._scope.camera.geometry(int(binning))
        self._scope.detector.update(binning=int(binning), area=[0, 0, shape[1], shape[0]])

    def set_binningindex(self, index):
        """Select binning ``2**index`` (an assumption: PyJEM does not document the index)."""
        self._set_binning(2 ** int(index))
        return self.get_detectorsetting()

    def set_exposuretime_value(self, value):
        """Set the exposure time in microseconds."""
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("The exposure time must be positive (µs).")
        self._scope.detector["exposure_us"] = value
        return self.get_detectorsetting()

    def set_imaging_area(self, width, height, x=0, y=0):
        """Read out a ``width`` x ``height`` window at ``x, y`` (binned pixels)."""
        shape, _, _ = self._scope.camera.geometry(self._scope.detector["binning"])
        if not (0 <= x < shape[1] and 0 <= y < shape[0] and 0 < width <= shape[1] - x
                and 0 < height <= shape[0] - y):
            raise ValueError(f"Imaging area {width}x{height}+{x}+{y} does not fit the {shape} frame.")
        self._scope.detector["area"] = [int(x), int(y), int(width), int(height)]
        return self.get_detectorsetting()

    def snapshot_rawdata(self):
        """Acquire one frame and return its raw little-endian pixel bytes (rows of width ``Width``)."""
        return self._scope.acquire().tobytes()

    def livestart(self):
        """Start live view (no effect)."""
        return {"status": "OK"}

    def livestop(self):
        """Stop live view (no effect)."""
        return {"status": "OK"}

    def insert(self):
        """Insert the camera."""
        self._scope.detector["inserted"] = True
        return {"status": "OK"}

    def retract(self):
        """Retract the camera."""
        self._scope.detector["inserted"] = False
        return {"status": "OK"}

    def get_insert_state(self):
        """Return ``True`` when the camera is inserted."""
        return bool(self._scope.detector["inserted"])


# -------------------------------------------------------------- module glue
def _then_settle(method):
    """The same call, after which every lens goes to the DAC value it now has (so lens hysteresis sees each change,
    in order)."""
    @functools.wraps(method)
    def call(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        finally:
            self._scope.settle()

    return call


def pyjem_interface(scope):
    """Build ``TEM3`` and ``detector`` module objects bound to one simulated microscope.

    Parameters
    ----------
    scope : SimulatedMicroscope
        The microscope every returned class drives.

    Returns
    -------
    TEM3, detector : types.ModuleType
        Use as ``TEM3.Lens3().GetIL1()``, ``detector.Detector("camera")``.
    """
    exception = types.ModuleType("PyJEM.TEM3.exception")
    exception.TEMError, exception.ErrorTypeKind, exception.ErrorCodeKind = TEMError, ErrorTypeKind, ErrorCodeKind
    tem3 = types.ModuleType("PyJEM.TEM3", "Virtual-microscope emulation of PyJEM.TEM3 (PyJEM 1.4).")
    for cls in (Lens3, Def3, EOS3, HT3, Apt3, Stage3):
        members = {"_scope": scope, "__doc__": cls.__doc__}
        if cls in (Lens3, EOS3):             # calls that can change a lens: let the lenses settle after each
            members.update({name: _then_settle(getattr(cls, name)) for name in dir(cls)
                            if name.startswith(("Set", "Select", "Up", "Down"))})
        setattr(tem3, cls.__name__, type(cls.__name__, (cls,), members))
    state = {"connected": True}
    tem3.connect = lambda: state.update(connected=True) or True
    tem3.isconnect = lambda: state["connected"]
    tem3.SetRecvTimeout = lambda milliseconds: None
    tem3.exception, tem3.TEMError = exception, TEMError
    det = types.ModuleType("PyJEM.detector", "Virtual-microscope emulation of PyJEM.detector (PyJEM 1.4).")
    det.Detector = type("Detector", (Detector,), {"_scope": scope, "__doc__": Detector.__doc__})
    det.get_attached_detector = lambda: [scope.camera.name]
    det.check_connection = lambda timeout=3.0: True
    return tem3, det


@contextlib.contextmanager
def install(scope, *, include_offline=True, force=False):
    """Make ``import PyJEM`` resolve to the simulated microscope inside a ``with`` block.

    Parameters
    ----------
    scope : SimulatedMicroscope
        The microscope to drive.
    include_offline : bool, optional
        Also register ``PyJEM.offline`` (scripts often fall back to it).
    force : bool, optional
        Replace an already imported (real) PyJEM for the block.

    Yields
    ------
    tuple
        ``(TEM3, detector)`` module objects.
    """
    if "PyJEM" in sys.modules and not getattr(sys.modules["PyJEM"], "_temgym_virtual", False) and not force:
        raise RuntimeError("A real PyJEM is already imported; pass force=True to shadow it.")
    tem3, det = pyjem_interface(scope)
    package = types.ModuleType("PyJEM", "Virtual-microscope emulation of PyJEM 1.4.")
    package.__path__ = []
    package._temgym_virtual = True
    package.TEM3, package.detector = tem3, det
    modules = {"PyJEM": package, "PyJEM.TEM3": tem3, "PyJEM.TEM3.exception": tem3.exception,
               "PyJEM.detector": det}
    if include_offline:
        offline = types.ModuleType("PyJEM.offline")
        offline.__path__ = []
        offline.TEM3, offline.detector = tem3, det
        package.offline = offline
        modules.update({"PyJEM.offline": offline, "PyJEM.offline.TEM3": tem3, "PyJEM.offline.detector": det})
    saved = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        yield tem3, det
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


__all__ = ["TEMError", "ErrorTypeKind", "ErrorCodeKind", "NotEmulatedError", "pyjem_interface", "install"]
