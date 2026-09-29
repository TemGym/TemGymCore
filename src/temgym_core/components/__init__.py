"""Optical components for rays and coherent Gaussian beams.

Read Component for dispatch, then Lens for a simple physical example.
The package groups planes, lenses, deflection, phase/amplitude elements and specimens.
GaussianActionComponent shares the local optical-path/transmission update.

Use component(beam) at its plane or run_to_end(beam, components) for a column.
Positions use the beam length unit; slopes are radians. Parameters annotated
Degrees use degrees; parameters ending in _rad use radians. See each component
for finite-body propagation and any ray/Gaussian support limitations.
"""

from ._base import (
    Component,
    GaussianActionComponent,
    _call_single_or_batched_gaussian,  # noqa: F401 (used by experimental/potential.py)
    _regularize_L2k,  # noqa: F401 (used by tests)
)
from .planes import (
    Plane,
    ScanGrid,
    Detector,
)
from .lenses import (
    Lens,
    Stigmator,
    KrivanekLens,
    AberratedLensKrivanek,
    SeidelLens,
    DistortedLens,
    ABCDTransfer,
    ThickLens,
    HardEdgeSolenoid,
    solenoid_column,
    RotatingLens,
    ElectromagneticLens,
    FourierTransform,
)
from .deflection import (
    Deflector,
    DoubleDeflector,
    Rotator,
    DeflectionBiprism,
    Biprism,
    DescanError,
    Scanner,
    Descanner,
)
from .actions import (
    SigmoidAperture,
    PhaseBiprism,
    ConstantPhaseShift,
    LinearPhaseShift,
    QuadraticPhaseShift,
    ZernikeAberration,
    ConstantAmplitudeShift,
    LinearAmplitudeShift,
    QuadraticAmplitudeShift,
)
from .specimens import (
    MagneticPhaseSample,
    RandomPhaseSample,
    BraggSample,
)
from .interpolated import (
    InterpolatedSample2D,
    sample_interpolant,
    InterpolatedFields3D,
)

__all__ = [
    "Component",
    "GaussianActionComponent",
    "DescanError",
    "Plane",
    "Lens",
    "Stigmator",
    "KrivanekLens",
    "AberratedLensKrivanek",
    "SeidelLens",
    "DistortedLens",
    "ScanGrid",
    "Scanner",
    "Descanner",
    "Detector",
    "ThickLens",
    "HardEdgeSolenoid",
    "solenoid_column",
    "Deflector",
    "DoubleDeflector",
    "Rotator",
    "DeflectionBiprism",
    "PhaseBiprism",
    "Biprism",
    "RotatingLens",
    "ElectromagneticLens",
    "ABCDTransfer",
    "SigmoidAperture",
    "ConstantPhaseShift",
    "LinearPhaseShift",
    "QuadraticPhaseShift",
    "ZernikeAberration",
    "ConstantAmplitudeShift",
    "LinearAmplitudeShift",
    "QuadraticAmplitudeShift",
    "MagneticPhaseSample",
    "RandomPhaseSample",
    "BraggSample",
    "InterpolatedSample2D",
    "sample_interpolant",
    "InterpolatedFields3D",
    "FourierTransform",
]
