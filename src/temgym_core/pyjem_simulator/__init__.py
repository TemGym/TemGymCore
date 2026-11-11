"""A virtual JEOL-like TEM behind a PyJEM-compatible interface.

The hidden optics are TemGym components (any TemGym column, :class:`OpticalTruth`); the
controls are 16-bit DAC values, volts and indices; frames come from a camera model.
Scripts written against PyJEM run on it unchanged. See
``README.md`` in this package for what is modelled and the realism still missing.
"""

from .camera import CameraModel
from .condenser import Condenser
from .electronics import DeflectorDrive, Electronics, LensDrive, LensHysteresis, DACRangeError
from .pyjem import ErrorCodeKind, ErrorTypeKind, NotEmulatedError, TEMError, pyjem_interface, install
from .seidel_plate import SeidelPlate
from .microscope import (
    OpticalTruth,
    PhysicalSetting,
    SpecimenBeam,
    UnmodelledStateError,
    SimulatedMicroscope,
)

__all__ = [
    "CameraModel",
    "Condenser",
    "DeflectorDrive",
    "Electronics",
    "ErrorCodeKind",
    "ErrorTypeKind",
    "LensDrive",
    "LensHysteresis",
    "NotEmulatedError",
    "OpticalTruth",
    "SeidelPlate",
    "PhysicalSetting",
    "SpecimenBeam",
    "TEMError",
    "UnmodelledStateError",
    "SimulatedMicroscope",
    "DACRangeError",
    "install",
    "pyjem_interface",
]
