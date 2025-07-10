from typing import Tuple
from typing_extensions import TypeAlias

import numpy as np
from numpy.typing import NDArray

RadiansNP: TypeAlias = np.float64


def P2R(radii: NDArray[np.float64], angles: NDArray[RadiansNP]) -> NDArray[np.complex128]:
    return radii * np.exp(1j*angles)


def R2P(x: NDArray[np.complex128]) -> Tuple[NDArray[np.float64], NDArray[RadiansNP]]:
    return np.abs(x), np.angle(x)


def as_gl_lines(all_rays, z_mult: int = 1):
    num_vertices = 0
    for r in all_rays[:-1]:
        num_vertices += r.num_display
    num_vertices *= 2

    xp = all_rays[0].xp
    vertices = xp.empty(
        (num_vertices, 3),
        dtype=np.float32,
    )
    idx = 0

    def _add_vertices(r1, r0):
        nonlocal idx, vertices

        num_endpoints = r1.num_display
        sl = slice(idx, idx + num_endpoints * 2, 2)
        vertices[sl, 0] = r1.x_central
        vertices[sl, 1] = r1.y_central
        vertices[sl, 2] = r1.z * z_mult
        sl = slice(1 + idx, 1 + idx + num_endpoints * 2, 2)
        # Relies on the fact that indexing
        # with None creates a new axis, only
        vertices[sl, 0] = r0.x_central[r1.mask_display].ravel()
        vertices[sl, 1] = r0.y_central[r1.mask_display].ravel()
        vertices[sl, 2] = r0.z * z_mult
        idx += num_endpoints * 2
        return idx

    r1 = all_rays[-1]
    for r0 in reversed(all_rays[:-1]):
        _add_vertices(r1, r0)
        if (r1b := r1.blocked_rays()) is not None:
            _add_vertices(r1b, r0)
        r1 = r0

    return vertices
