from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def P2R(radii: NDArray[np.float64], angles: NDArray[np.floating]) -> NDArray[np.complex128]:
    return radii * np.exp(1j*angles)


def R2P(x: NDArray[np.complex128]) -> Tuple[NDArray[np.float64], NDArray[np.floating]]:
    return np.abs(x), np.angle(x)


def as_gl_lines(coords, zvals, z_mult: int = 1):
    num_rays, num_planes, _ = coords.shape
    assert zvals.size == num_planes, f"{zvals.size} != {num_planes}"

    xy_coords = coords[..., :2]
    assert xy_coords.shape == (num_rays, num_planes, 2)

    zvals = np.repeat(zvals * z_mult, num_rays).reshape(-1, num_rays).T
    zvals = zvals[..., np.newaxis]
    assert zvals.shape == (num_rays, num_planes, 1)

    xyz_coords = np.concatenate((xy_coords, zvals), axis=-1)
    assert xyz_coords.shape == (num_rays, num_planes, 3)
    xyz_view = np.lib.stride_tricks.sliding_window_view(xyz_coords, 2, axis=1)
    assert xyz_view.shape == (num_rays, num_planes - 1, 3, 2), f"{xyz_view.shape} != {(num_rays, num_planes - 1, 3, 2)}"
    xyz_view = np.moveaxis(xyz_view, (0, 1, 2, 3), (0, 1, 3, 2))
    vertices = xyz_view.reshape(-1, 3)

    return vertices
