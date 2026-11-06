"""A virtual JEOL-like TEM behind a PyJEM-compatible interface.

The hidden truth is the calibration model of :mod:`temgym_core.calibration`;
the controls are 16-bit words, volts and indices; frames come from a camera
model. Calibration protocols written against PyJEM run on it unchanged. See
``README.md`` in this package for what is modelled and the realism still missing.
"""

from .camera import CameraModel
from .electronics import DeflectorDrive, Electronics, LensDrive, WordRangeError
from .pyjem import ErrorCodeKind, ErrorTypeKind, NotEmulatedError, TEMError, facade, install
from .scope import (
    OpticalTruth,
    PhysicalSetting,
    UnmodelledStateError,
    VirtualMicroscope,
    vendor_calibration,
)

__all__ = [
    "CameraModel",
    "DeflectorDrive",
    "Electronics",
    "ErrorCodeKind",
    "ErrorTypeKind",
    "LensDrive",
    "NotEmulatedError",
    "OpticalTruth",
    "PhysicalSetting",
    "TEMError",
    "UnmodelledStateError",
    "VirtualMicroscope",
    "WordRangeError",
    "facade",
    "install",
    "vendor_calibration",
]
