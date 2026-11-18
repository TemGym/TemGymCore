"""Differentiable first-order NEOARM in units of specimen-to-SightX distance L.

The two adjacent notebooks share these definitions. Enable JAX 64-bit mode in the
calling notebook before constructing parameters or tracing rays.
"""
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from temgym_core.components import Deflector, ElectromagneticLens, Lens, Plane
from temgym_core.ray import Ray
from temgym_core.run import run_to_end

LENSES = ("CL1", "CL3", "CM", "IL1", "IL2", "IL3", "PL1")
STEER = ("CLA1", "CLA2", "IS1", "IS2", "PLA", "SpotA", "OLs")
HERE = Path(__file__).resolve().parent
VOLTAGE = 200_000.0
SPECIMEN_INDEX = 9


def readbacks(row, *, reference=None):
    """Decode one console snapshot; optionally require the same fixed optics as a reference."""
    if reference is not None:
        for name in ("ht", "OLf", "OLc", "OM", "alpha", "GunA1", "GunA2"):
            if not np.array_equal(row[name], reference[name]):
                raise ValueError(f"{name} differs from the fixed session configuration")
    result = {}
    for name in (*LENSES, *STEER):
        value = np.asarray(row[name], dtype=float)
        shape = () if name in LENSES else (2,)
        if value.shape != shape or not np.all((0 <= value) & (value <= 65535)):
            raise ValueError(f"Invalid {name} readback: {value}")
        result[name] = jnp.asarray(value)
    return result


def parameters(reference):
    """Start with the console fit and drawing. Kick gains and source sizes are guesses."""
    drawing = json.loads((HERE.parent / "virtual_neoarm/neoarm_column_diagram.json").read_text())["z_m"]
    z = {name: value / drawing["Screen"] for name, value in drawing.items()}
    u = np.array(json.loads((HERE / "results/neoarm_nondimensional_start.json").read_text())["u"])
    gaps = np.diff([0, *(z[name] for name in LENSES[3:]), 1]) * np.exp(np.r_[u[2:6], 0.])
    projection_z = np.cumsum(gaps / gaps.sum())[:4]
    f = np.exp(u[0])
    top, bottom = z["Corrector Top"], z["Corrector Bottom"]
    # Spot alignment and objective-stigmator feed-through use effective kick planes.
    planes = [z["CL2"] + t * (top - z["CL2"]) for t in (0.4, 0.75)]
    planes += [0.50 * projection_z[0], 0.75 * projection_z[0], projection_z[-1] - 0.005, z["CL1"], 0.]
    return {key: jnp.asarray(value, dtype=float) for key, value in dict(
        z_lens=[z["CL1"], z["CL2"], z["CLMini"], *projection_z],
        source_z=z["Source"], objective_f=f, objective_z=f * (1 + np.exp(-u[1])),
        corrector_limits=[top, bottom], z_steer=planes,
        Gc=np.r_[0.6233, 1.5797, 0.2384, np.exp(u[6:10])],
        Tc=np.r_[np.zeros(3), np.exp(u[10:14])],
        turns=0.080 * np.array([1, 1, 1, -1, -1, -1, 1]),
        phi_a=u[16:20], phi_b=u[20:24], middle_dac=[31593., 43363., 31047., 61029.5],
        reference_dac=np.stack([reference[name] for name in STEER]),
        kick_matrix=np.array([g * np.eye(2) for g in (1e-7, 1e-7, 1e-6, 1e-6, 1e-6, 0., 0.)]),
        source_mean=np.zeros(4), source_sigma=[1e-8, 1e-8, 2e-5, 2e-5],
        pixel_scale=5_000., pixel_origin=[0., 0.], camera_axes=np.eye(2),
    ).items()}


def column(controls, p):
    """TemGym components, in propagation order. The specimen is at z=0, SightX at z=1."""
    dac = jnp.stack([controls[name] for name in LENSES])
    x = (dac[3:] - p["middle_dac"]) / 20_000
    phi = jnp.exp(p["phi_a"] * x + p["phi_b"] * x**2)
    strength = p["Gc"] * jnp.concatenate([jnp.ones(3), phi])
    cl1, cl3, cm, il1, il2, il3, pl1 = [
        ElectromagneticLens(z=p["z_lens"][i], turns=p["turns"][i], current=dac[i],
                            Gc=strength[i], Tc=p["Tc"][i]) for i in range(len(LENSES))
    ]
    delta = jnp.stack([controls[name] for name in STEER]) - p["reference_dac"]
    kick = jnp.einsum("nij,nj->ni", p["kick_matrix"], delta)
    cla1, cla2, is1, is2, pla, spot, ols = [
        Deflector(z=p["z_steer"][i], def_x=kick[i, 0], def_y=kick[i, 1]) for i in range(len(STEER))
    ]
    top, bottom = p["corrector_limits"]
    relay_f = (bottom - top) / 4
    objective_f = p["objective_f"]
    return [
        cl1, spot, cl3, cla1, cla2,
        Lens(z=top + relay_f, focal_length=relay_f), Lens(z=bottom - relay_f, focal_length=relay_f),
        cm, Lens(z=-objective_f, focal_length=objective_f), Plane(z=0.), ols,
        Lens(z=p["objective_z"], focal_length=objective_f), is1, is2, il1, il2, il3, pla, pl1, Plane(z=1.),
    ]


def predict(controls, p):
    """Detector column, row and one-axis RMS width (pixels relative to frame centre)."""
    # For affine optics four one-sigma offsets propagate the full diagonal input covariance exactly.
    q = p["source_mean"] + jnp.vstack([jnp.zeros(4), jnp.diag(p["source_sigma"])])
    ray = Ray(x=q[:, 0], y=q[:, 1], dx=q[:, 2], dy=q[:, 3],
              z=p["source_z"], pathlength=0., voltage=VOLTAGE)
    out = run_to_end(ray, column(controls, p))
    xy = jnp.stack([out.x, out.y], axis=1) @ p["camera_axes"].T * p["pixel_scale"]
    width = jnp.sqrt(jnp.sum((xy[1:] - xy[0])**2) / 2)
    return jnp.r_[xy[0] + p["pixel_origin"], width]
