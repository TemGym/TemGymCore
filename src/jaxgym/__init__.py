from typing_extensions import TypeAlias
from typing import NamedTuple
import numpy as np
from numpy.typing import NDArray


PositiveFloat: TypeAlias = float
NonNegativeFloat: TypeAlias = float
Radians: TypeAlias = float
Degrees: TypeAlias = float


class ShapeYX(NamedTuple):
    y: int
    x: int


class ScaleYX(NamedTuple):
    y: float
    x: float


class CoordsXY(NamedTuple):
    x: NDArray[np.floating]
    y: NDArray[np.floating]


class PixelsYX(NamedTuple):
    y: NDArray[np.integer]
    x: NDArray[np.integer]
