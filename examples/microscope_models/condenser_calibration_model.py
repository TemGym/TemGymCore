"""Finite condenser lenses and an unknown round virtual source.

Ray coordinates are [x, y, theta_x, theta_y, 1], with the specimen at z=0.
The source plane is defined as its effective waist: position/angle covariance
is zero there. Its z, x, y, mean angles, size and divergence are all unknown.
"""

import jax.numpy as jnp
import numpy as np

from temgym_core.solenoid import solenoid_column_matrix

LENS_NAMES = ("CL1", "CL3", "Mini", "Objective pre-field")


def initial_parameters():
    return np.r_[
        np.log([0.195, 0.04, 0.05, 0.025, 0.0025]),
        np.log([0.005, 0.005, 0.003, 0.001]),
        np.log([0.18, 0.35, 0.38, 0.45]),
        [0.1, -0.1],
        [0.01, -0.01],
        np.log(10e-9),
        np.log(60e-6),
    ]


def generating_parameters():
    return np.r_[
        np.log([0.1975, 0.035, 0.056, 0.0205, 0.002]),
        np.log([0.005, 0.005, 0.003, 0.001]),
        np.log([0.20, 0.37, 0.40, 0.48]),
        [0.2, -0.3],
        [0.02, -0.01],
        np.log(8e-9),
        np.log(80e-6),
    ]


def matrix5(parameters, controls):
    """Map an unknown source waist to the specimen through four finite lenses."""
    p = jnp.asarray(parameters)
    gaps, lengths = jnp.exp(p[:5]), jnp.exp(p[5:9])
    strengths = jnp.exp(p[9:13]) * jnp.r_[jnp.asarray(controls), -1.0]
    return solenoid_column_matrix(gaps, lengths, strengths)


def source_moments(parameters):
    p = jnp.asarray(parameters)
    mean = jnp.r_[p[13:15] * 1e-8, p[15:17] * 1e-4]
    covariance = jnp.diag(
        jnp.r_[jnp.repeat(jnp.exp(2 * p[17]), 2), jnp.repeat(jnp.exp(2 * p[18]), 2)]
    )
    return mean, covariance


def specimen_moments(parameters, controls):
    transfer = matrix5(parameters, controls)[:4, :4]
    mean, covariance = source_moments(parameters)
    return transfer @ mean, transfer @ covariance @ transfer.T


def moment_observables(parameters, controls):
    """Seven round-beam quantities: mean position/angle and the three covariance entries."""
    mean, covariance = specimen_moments(parameters, controls)
    sx, st = jnp.sqrt(covariance[0, 0]), jnp.sqrt(covariance[2, 2])
    return jnp.r_[
        mean[:2] / 1e-7,
        mean[2:] / 1e-3,
        jnp.log(sx),
        jnp.log(st),
        covariance[0, 2] / (sx * st),
    ]


def named_parameters(p):
    p = np.asarray(p, dtype=float)
    gaps, lengths = np.exp(p[:5]), np.exp(p[5:9])
    source_z = -(gaps.sum() + lengths.sum())
    centres = (
        source_z
        + np.cumsum(gaps[:-1])
        + np.r_[0.0, np.cumsum(lengths[:-1])]
        + lengths / 2
    )
    return {
        "source_z_mm": source_z * 1000,
        **{f"{name}.z_mm": float(z * 1000) for name, z in zip(LENS_NAMES, centres)},
        **{
            f"{name}.length_mm": float(v * 1000) for name, v in zip(LENS_NAMES, lengths)
        },
        **{
            f"{name}.strength": float(v)
            for name, v in zip(LENS_NAMES, np.exp(p[9:13]) * [1, 1, 1, -1])
        },
        "source_x_nm": p[13] * 10,
        "source_y_nm": p[14] * 10,
        "source_theta_x_mrad": p[15] * 0.1,
        "source_theta_y_mrad": p[16] * 0.1,
        "source_rms_size_nm": np.exp(p[17]) * 1e9,
        "source_rms_angle_mrad": np.exp(p[18]) * 1000,
    }


def equivalent_parameters(parameters, *, upstream_scale=1.0, objective_strength=None):
    """An exact two-parameter family with identical specimen means/covariances.

    Scale upstream axial distances, rescale the source's size and divergence
    inversely, and adjust the static final lens plus its two surrounding gaps.
    A change of its constant Larmor rotation is absorbed into the unknown
    source mean orientation. This is an identifiability diagnostic, not data.
    """
    p = np.asarray(parameters, dtype=float)
    scale = float(upstream_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Upstream scale must be positive.")
    old_q = -np.exp(p[12])
    new_q = old_q if objective_strength is None else float(objective_strength)
    if not np.isfinite(new_q) or not -1.2 < new_q < 0:
        raise ValueError("Use the declared negative objective-strength branch.")
    gaps, lengths = np.exp(p[:5]), np.exp(p[5:9])
    c = -old_q * np.sin(old_q) / lengths[3]
    a = np.cos(old_q) + gaps[4] * c
    d = np.cos(old_q) + gaps[3] * c
    root = np.sqrt(scale)
    new_c = c / root
    new_length = -new_q * np.sin(new_q) / new_c
    new_gap_after = (a / root - np.cos(new_q)) / new_c
    new_gap_before = (d * root - np.cos(new_q)) / new_c
    if min(new_length, new_gap_after, new_gap_before) <= 0:
        raise ValueError("This family member has a nonpositive length or gap.")
    result = p.copy()
    result[:3] += np.log(scale)
    result[5:8] += np.log(scale)
    result[3:5] = np.log([new_gap_before, new_gap_after])
    result[8] = np.log(new_length)
    result[12] = np.log(-new_q)
    angle = new_q - old_q
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    result[13:15] = root * (rotation @ p[13:15])
    result[15:17] = (rotation @ p[15:17]) / root
    result[17] += np.log(root)
    result[18] -= np.log(root)
    return result
