from typing_extensions import TypeAlias, Literal
import numpy as np
from numpy.typing import NDArray


PositiveFloat: TypeAlias = float
NonNegativeFloat: TypeAlias = float
Radians: TypeAlias = float
Degrees: TypeAlias = float
Shape_YX: TypeAlias = tuple[int, int]
Scale_YX: TypeAlias = tuple[float, float]
Coords_XY: TypeAlias = tuple[NDArray[np.floating], NDArray[np.floating]]
Coords_YX: TypeAlias = tuple[NDArray[np.floating], NDArray[np.floating]]
Pixels_YX: TypeAlias = tuple[NDArray[np.integer], NDArray[np.integer]]


BackendT = Literal["cpu", "gpu"]


class UsageError(Exception): ...


class InvalidModelError(Exception): ...
