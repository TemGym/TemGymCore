"""A virtual JEOL-like TEM behind a PyJEM-compatible interface.

The hidden optics are TemGym components (:mod:`temgym_core.calibration`); the
controls are 16-bit DAC values, volts and indices; frames come from a camera model.
Scripts written against PyJEM run on it unchanged. See
``README.md`` in this package for what is modelled and the realism still missing.
"""

from .camera import CameraModel
from .electronics import DeflectorDrive, Electronics, LensDrive, WordRangeError
from .pyjem import ErrorCodeKind, ErrorTypeKind, NotEmulatedError, TEMError, pyjem_interface, install
from .specimens import CrossGrating, PolycrystallineGold
from .scope import (
    OpticalTruth,
    PhysicalSetting,
    SpecimenBeam,
    UnmodelledStateError,
    VirtualMicroscope,
)

__all__ = [
    "CameraModel",
    "CrossGrating",
    "DeflectorDrive",
    "Electronics",
    "ErrorCodeKind",
    "ErrorTypeKind",
    "LensDrive",
    "NotEmulatedError",
    "OpticalTruth",
    "PolycrystallineGold",
    "PhysicalSetting",
    "SpecimenBeam",
    "TEMError",
    "UnmodelledStateError",
    "VirtualMicroscope",
    "WordRangeError",
    "install",
    "pyjem_interface",
]
