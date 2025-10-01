from typing_extensions import TypeAlias
from typing import NamedTuple, Union

import jax; jax.config.update("jax_enable_x64", True)  # noqa: E702
import jax.numpy as jnp
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


class CoordXY(NamedTuple):
    """Continuous coordinates in the optical frame.

    Parameters
    ----------
    x : float
        X position, metres.
    y : float
        Y position, metres.
    """
    x: float
    y: float

    def to_coords(self) -> 'CoordsXY':
        return CoordsXY(
            x=jnp.array((self.x,)),
            y=jnp.array((self.y,))
        )


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


class PixelYX(NamedTuple):
    """Pixel coordinates for images.

    Parameters
    ----------
    y : Union[int, float]
        Pixel row indices
    x : Union[int, float]
        Pixel column indices

    Notes
    -----
    Pixel indices are 0-based.
    """
    y: Union[int, float]
    x: Union[int, float]

    def to_pixels(self) -> 'PixelsYX':
        return PixelsYX(
            x=jnp.array((self.x,)),
            y=jnp.array((self.y,))
        )


class PixelsYX(NamedTuple):
    """Pixel coordinates for images.

    Parameters
    ----------
    y : numpy.ndarray
        Pixel row indices. Integer or floating dtype.
    x : numpy.ndarray
        Pixel column indices. Integer or floating dtype.

    Notes
    -----
    Pixel indices are 0-based.
    """
    y: NDArray[Union[np.integer, np.floating]]
    x: NDArray[Union[np.integer, np.floating]]


# Convenience re-exports
try:
    from .plotting import plot_model, PlotParams  # noqa: F401
except Exception:
    # Plotting has optional dependencies (matplotlib); ignore import errors at package import time
    pass
