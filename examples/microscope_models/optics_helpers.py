"""Shared optics utilities for microscope design notebooks."""

import numpy as np
from temgym_core.transfer_matrices import propagation_matrix, lens_matrix


def build_chain(elements, xp=np):
    """Compose ('P', d) / ('L', f) elements into a 3x3 transfer matrix."""
    M = xp.eye(3)
    for kind, value in elements:
        op = propagation_matrix(value, xp=xp) if kind == 'P' else lens_matrix(value, xp=xp)
        M = op @ M
    return M


def extract_abcd(M):
    """Extract (A, B, C, D) from a 3x3 transfer matrix."""
    return M[0, 0], M[0, 1], M[1, 0], M[1, 1]


def electron_wavelength(V):
    """Relativistic de Broglie wavelength [m] for voltage V [volts]."""
    return 12.264306e-10 / np.sqrt(V * (1.0 + 0.978466e-6 * V))


def q_inv_from_source(sigma_x, sigma_theta, wavelength):
    """Complex 1/q from source size [m] and divergence [rad]."""
    w = 2.0 * sigma_x
    q_inv_im = wavelength / (np.pi * w**2)
    sigma_diff = wavelength / (2.0 * np.pi * w)
    curv_sq = max(sigma_theta**2 - sigma_diff**2, 0.0)
    q_inv_re = np.sqrt(curv_sq) / sigma_x if curv_sq > 0 else 0.0
    return complex(q_inv_re, q_inv_im)


def propagate_q_inv(q_inv, M):
    """ABCD-law: q_out^-1 = (C + D*q^-1) / (A + B*q^-1)."""
    A, B, C, D = float(M[0, 0]), float(M[0, 1]), float(M[1, 0]), float(M[1, 1])
    return (C + D * q_inv) / (A + B * q_inv)


def fwhm_from_q_inv(q_inv, wavelength):
    """Beam FWHM from Im(1/q)."""
    w = np.sqrt(wavelength / (np.pi * np.imag(q_inv)))
    return w * np.sqrt(2.0 * np.log(2.0))


def current_from_focal(f, Gc, turns):
    """Coil current [A] from focal length via 1/f = Gc*(NI)^2."""
    return np.sqrt(1.0 / (Gc * f)) / turns


def focal_bounds_from_current_bounds(Gc, turns, I_min, I_max):
    """Map current limits to (f_min, f_max)."""
    return 1.0 / (Gc * (turns * I_max)**2), 1.0 / (Gc * (turns * I_min)**2)


# CAD pixel calibration (shared reference span from measure_microscope_geometry)
PX_TO_M = 0.2 / (624 - 528)
