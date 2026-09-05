"""Design and audit ten focused, rotation-free microscope imaging settings.

Run from the repository root with ``.venv/bin/python
examples/microscope_models/design_imaging_modes.py --write-table``. Omit
``--write-table`` to reproduce the figures/report without changing the TOML.
All currents are simulation parameters, not NEOARM hardware commands.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import re
import sys
import tomllib

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

from temgym_core.components import Plane
from temgym_core.imaging_modes import (
    ConjugateImagingConstraint, imaging_geometry, radial_chain,
    refine_conjugate_imaging_currents,
)
from temgym_core.microscope_model import MicroscopeModel
from temgym_core.plotting import export_ray_figure
from temgym_core.ray import Ray
from temgym_core.run import run_to_end


jax.config.update("jax_enable_x64", True)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TARGETS = np.array((40_000, 60_000, 80_000, 100_000, 150_000,
                    200_000, 250_000, 300_000, 400_000, 600_000))
CONSTRAINT = ConjugateImagingConstraint(("IL1", "IL2", "IL3"))
PL1_CURRENT_A = -3.5


def design(model):
    """Continue one smooth branch with the original objective and fixed PL1."""
    names = [lens.name for lens in model.lenses]
    seed = model.modes["magnification"].interpolate_currents(40_000)
    seed[names.index("PL1")] = PL1_CURRENT_A
    rows = []
    for target in TARGETS:
        seed = refine_conjugate_imaging_currents(model, seed, target, CONSTRAINT)
        rows.append(seed.copy())
    return np.array(rows)


def write_table(path, model, rows):
    config = tomllib.loads(path.read_text())
    headers = config["modes"]["imaging"]["magnification"]["headers"]
    indices = [[l.name for l in model.lenses].index(name) for name in headers[1:]]
    values = [[int(target), *map(float, row[indices])] for target, row in zip(TARGETS, rows)]
    block = (
        "[modes.imaging.magnification]\n"
        "# Designed by design_imaging_modes.py at 200 kV: A = mag * identity, B = 0.\n"
        "# Simulation currents; no calibrated relationship to NEOARM DAC words.\n"
        "# Refine interpolated settings to preserve conjugacy and zero rotation.\n"
        'conjugate_imaging = { relay_lenses = ["IL1", "IL2", "IL3"], current_bounds_a = [0.01, 5.0], rotation_rad = 0.0 }\n'
        f"headers = {json.dumps(headers)}\nvalues = [\n"
        + "".join(f"  {json.dumps(row)},\n" for row in values) + "]\n\n"
    )
    source, count = re.subn(
        r"\[modes\.imaging\.magnification\]\n.*?(?=\[modes\.imaging\.diffraction\])",
        lambda _: block, path.read_text(), flags=re.S,
    )
    if count != 1:
        raise ValueError("Expected one magnification table before the diffraction table.")
    path.write_text(source)


def ray_matrix(model, target, *, refine=True):
    """Differentiate the actual component propagation, independently of the solve."""
    if not refine:
        model = replace(model, modes={**model.modes, "magnification": replace(
            model.modes["magnification"], conjugate_imaging=None,
        )})
    start = model.auxiliary["sample"]["z"]
    end = model.auxiliary["detector"]["z"]
    components = [c for c in model.build_components("magnification", float(target))
                  if start < float(c.z) < end] + [Plane(z=end)]

    def trace(state):
        ray = Ray(x=state[0], y=state[1], dx=state[2], dy=state[3],
                  z=start, pathlength=0.0, voltage=model.voltage)
        out = run_to_end(ray, components)
        return jnp.array((out.x, out.y, out.dx, out.dy))

    return np.asarray(jax.jacfwd(trace)(jnp.zeros(4)))


def diagnostics(matrix, target):
    a, b = matrix[:2, :2], matrix[:2, 2:]
    return {
        "magnification": float(target),
        "max_A_relative_error": float(np.max(np.abs(a / target - np.eye(2)))),
        "B_operator_norm_m": float(np.linalg.norm(b, 2)),
        "rotation_deg": float(np.degrees(np.arctan2(a[1, 0], a[0, 0]))),
        "matrix_grouped_xy_angles": matrix.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-table", action="store_true")
    args = parser.parse_args()
    path = HERE / "microscope.toml"
    model = MicroscopeModel.from_toml(str(path))
    rows = design(model)
    if args.write_table:
        write_table(path, model, rows)
    model = MicroscopeModel.from_toml(str(path))
    mode = model.modes["magnification"]
    if not np.array_equal(mode.control_values, TARGETS) or mode.conjugate_imaging is None:
        raise ValueError("Generate the focused table first with --write-table.")
    stored = np.array([mode.interpolate_currents(m) for m in TARGETS])
    np.testing.assert_allclose(stored, rows, rtol=1e-10, atol=1e-12)
    names = [l.name for l in model.lenses]
    geometry = imaging_geometry(model, rows[0])
    downstream_names = [l.name for l in geometry["lenses"]]
    focal_mm = 1000 / (geometry["gains"] * rows[:, geometry["indices"]]**2)

    # Audit stored rows without allowing refinement to hide incorrect values.
    records = [diagnostics(ray_matrix(model, m, refine=False), m) for m in TARGETS]
    midpoints = (TARGETS[:-1] + TARGETS[1:]) / 2
    between = [diagnostics(ray_matrix(model, m), m) for m in midpoints]
    plain = [diagnostics(ray_matrix(model, m, refine=False), m) for m in midpoints]
    for record in records + between:
        if record["max_A_relative_error"] > 1e-9 or record["B_operator_norm_m"] > 1e-8:
            raise RuntimeError(f"Imaging validation failed: {record}")
    for record, current, focal in zip(records, rows, focal_mm):
        record["model_currents_a"] = dict(zip(names, map(float, current)))
        record["focal_lengths_mm"] = dict(zip(downstream_names, map(float, focal)))

    sys.path.insert(0, str(HERE.parent / "lens_inversion"))
    from neoarm_flc import read_flc_table
    flc = read_flc_table(HERE.parent / "lens_inversion/dac_lens_inversion.ipynb")
    dense_mags = np.geomspace(TARGETS[0], TARGETS[-1], 201)
    dense = np.array([refine_conjugate_imaging_currents(
        model, mode.interpolate_currents(m), m, CONSTRAINT,
    ) for m in dense_mags])
    comparison = {}
    for name in CONSTRAINT.relay_lenses + ("PL1",):
        i = names.index(name)
        predicted = np.array([refine_conjugate_imaging_currents(
            model, mode.interpolate_currents(float(m)), float(m), CONSTRAINT,
        )[i] for m in flc["nominal_magnification"]])
        observed = flc[name] / flc[name][0]
        predicted = predicted / predicted[0]
        comparison[name] = {
            "normalization": "Each series divided by its own 40k value; no fitted DAC-to-current conversion",
            "nominal_magnifications": flc["nominal_magnification"].tolist(),
            "recorded_relative_DAC": observed.tolist(),
            "designed_relative_current": predicted.tolist(),
            "relative_curve_rms_difference": float(np.sqrt(np.mean((predicted - observed)**2))),
        }

    output = ROOT / "examples/generated/microscope_imaging_modes"
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "Self-consistent 200 kV thin-round-lens simulation; not a physical NEOARM calibration",
        "coordinate_order": ["x", "y", "theta_x", "theta_y"],
        "geometry_m": {key: float(geometry[key]) for key in (
            "sample_z", "detector_z", "objective_image_z", "projector_object_z")},
        "lens_z_m": dict(zip(downstream_names, map(float, geometry["z"]))),
        "objective_magnification": geometry["objective_magnification"],
        "projector_magnification": geometry["projector_magnification"],
        "stored_settings": records, "refined_midpoints": between,
        "unrefined_midpoints": plain, "neoarm_curve_comparison": comparison,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for ax, name in zip(axes.flat, CONSTRAINT.relay_lenses + ("PL1",)):
        i = names.index(name)
        ax.semilogx(dense_mags / 1000, dense[:, i] / dense[0, i], label="Designed current / 40k current")
        ax.semilogx(flc["nominal_magnification"] / 1000, flc[name] / flc[name][0], "o--", label="Recorded DAC / 40k DAC")
        ax.set(title=name, xlabel="Magnification (thousands ×)", ylabel="Relative control")
        ax.grid(alpha=.2)
        ax.set_xticks([40, 100, 200, 400, 600], ["40", "100", "200", "400", "600"])
        ax.xaxis.set_minor_formatter(NullFormatter())
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Focused model versus recorded NEOARM control curves\nSeparate 40k normalization; no current calibration")
    export_ray_figure(fig, output / "neoarm_curve_comparison")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for name in CONSTRAINT.relay_lenses:
        i = names.index(name)
        axes[0].plot(dense_mags / 1000, dense[:, i], label=name)
    axes[0].set(xlabel="Magnification (thousands ×)", ylabel="Model current (A)", title="Smooth relay-lens settings")
    for i, name in enumerate(downstream_names):
        axes[1].semilogy(TARGETS / 1000, focal_mm[:, i], "o-", label=name)
    axes[1].set(xlabel="Magnification (thousands ×)", ylabel="Effective focal length (mm)", title="Thin-lens focal lengths")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    export_ray_figure(fig, output / "lens_parameters")
    plt.close(fig)

    # True-z basis rays in the rotating frame, with each conjugate stage visible.
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), layout="constrained")
    stages = [(geometry["sample_z"], geometry["objective_image_z"], "Objective"),
              (geometry["objective_image_z"], geometry["projector_object_z"], "IL1–IL3 relay"),
              (geometry["projector_object_z"], geometry["detector_z"], "PL1")]
    for row_index, target in enumerate((40_000, 200_000, 600_000)):
        current = rows[np.flatnonzero(TARGETS == target)[0]]
        powers = geometry["gains"] * current[geometry["indices"]]**2
        for ax, (start, end, title) in zip(axes[row_index], stages):
            zs = np.unique(np.r_[np.linspace(start, end, 300), geometry["z"][(geometry["z"] > start) & (geometry["z"] < end)]])
            matrices = []
            for z in zs:
                include = geometry["z"] <= z
                matrices.append(radial_chain(geometry["z"][include], powers[include], geometry["sample_z"], z))
            matrices = np.array(matrices)
            for x, color in [(-1e-9, "#0072B2"), (0., "#777777"), (1e-9, "#D55E00")]:
                for angle, style in [(-1e-4, "--"), (0., "-"), (1e-4, ":")]:
                    ax.plot(zs * 1000, (matrices[:, 0, 0] * x + matrices[:, 0, 1] * angle) * 1e6,
                            style, color=color, lw=.9)
            for z, name in zip(geometry["z"], downstream_names):
                if start < z < end:
                    ax.axvline(z * 1000, color="black", alpha=.15)
                    ax.text(z * 1000, .98, name, transform=ax.get_xaxis_transform(), rotation=90, va="top", fontsize=8)
            ax.set(xlabel="Column z (mm)", ylabel="Ray height (µm)", title=f"{target // 1000}k — {title}")
            ax.grid(alpha=.15)
    fig.suptitle("Conjugate stages, co-rotating coordinates\nSpecimen points −1, 0, +1 nm; launch angles −0.1, 0, +0.1 mrad")
    export_ray_figure(fig, output / "conjugate_rays")
    plt.close(fig)
    print("Verified stored rows and refined midpoints through run_to_end:")
    print("max relative A error", max(r["max_A_relative_error"] for r in records + between))
    print("max B operator norm (m)", max(r["B_operator_norm_m"] for r in records + between))
    print("Report and SVG/PDF/PNG figures:", output)
    for record in records:
        print(int(record["magnification"]), record["focal_lengths_mm"])


if __name__ == "__main__":
    main()
