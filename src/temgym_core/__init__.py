from typing_extensions import TypeAlias
from typing import NamedTuple
import numpy as np
from numpy.typing import NDArray


PositiveFloat: TypeAlias = float
NonNegativeFloat: TypeAlias = float
Radians: TypeAlias = float
Degrees: TypeAlias = float


class ShapeYX(NamedTuple):
    """Describe image shape as (height, width) in pixels.

    Parameters
    ----------
    y : int
        Image height in pixels.
    x : int
        Image width in pixels.

    Notes
    -----
    Conforms to numpy/jax indexing order (y first, then x).
    """
    y: int
    x: int


class ScaleYX(NamedTuple):
    """Describe pixel scale in metres per pixel along y and x.

    Parameters
    ----------
    y : float
        Pixel size along y, metres per pixel.
    x : float
        Pixel size along x, metres per pixel.

    Notes
    -----
    Values can be different if pixels are anisotropic.
    """
    y: float
    x: float


class CoordsXY(NamedTuple):
    """Continuous coordinates in the optical frame.

    Parameters
    ----------
    x : numpy.ndarray
        X position(s), metres. Shape broadcastable to context.
    y : numpy.ndarray
        Y position(s), metres. Shape broadcastable to context.
    """
    x: NDArray[np.floating]
    y: NDArray[np.floating]


class PixelsYX(NamedTuple):
    """Discrete pixel coordinates for images.

    Parameters
    ----------
    y : numpy.ndarray
        Pixel row indices. Integer dtype.
    x : numpy.ndarray
        Pixel column indices. Integer dtype.

    Notes
    -----
    Pixel indices are 0-based.
    """
    y: NDArray[np.integer]
    x: NDArray[np.integer]


# Convenience re-exports
try:
    from .plotting import plot_model, PlotParams  # noqa: F401
except Exception:
    # Plotting has optional dependencies (matplotlib); ignore import errors at package import time
    pass
