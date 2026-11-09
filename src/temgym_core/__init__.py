"""Differentiable electron optics in JAX: rays, Gaussian beamlets and wavefront rendering."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _package_version
from typing import NamedTuple, TypeAlias

import numpy as np
from numpy.typing import NDArray

try:
    __version__ = _package_version("temgym-core")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0+unknown"


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


# Convenience names are lazy so importing rays does not import plotting or samples.

_EXPORTS = {
    **dict.fromkeys(
        (
            "plot_model",
            "plot_model_plotly",
            "plot_abcd_profile",
            "PlotParams",
            "legacy_beam_plot_params",
        ),
        "plotting",
    ),
    "InterpolatedSample2D": "components",
    "BraggSample": "components",
    "enable_x64": "config",
    "require_x64": "config",
    "x64_enabled": "config",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
