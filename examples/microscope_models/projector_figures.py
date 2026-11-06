"""Figures and tables in the style of the paper figure: the column drawn vertically with lenses
as rings and the beam as a coloured band (``column_panel``), Okabe-Ito colours, bold panel
titles, no grid. ``session_frames`` and ``prediction_check_figure`` serve the stage-1 session.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display

import projector_calibration as P

# Paper colours: ink, then blue / vermilion / bluish green lines, with the matching beam fills.
INK, INK2, LIGHT = "#222222", "#595959", "#ececec"
BLUE, ORANGE, GREEN, YELLOW = "#0072B2", "#D55E00", "#009E73", "#F0E442"
FILL = {"truth": "#43C78E", "orange": "#F5A35C", "blue": "#6BAED6"}
COLUMN_LABELS = ("Objective", "IL1", "IL2", "IL3", "PL1", "Camera")
SET_LABELS = [str(i) for i in range(8)] + ["diff."]


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 100, "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelsize": 11, "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "lines.linewidth": 1.8, "image.cmap": "gray",
        "legend.frameon": False, "legend.fontsize": 10})


def table(header, rows):
    """Show a Markdown table. ``rows`` are lists of already formatted cells."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    display(Markdown("\n".join(lines)))


def _length(value_m):
    for scale, unit in ((1e-3, "mm"), (1e-6, "µm"), (1e-9, "nm")):
        if value_m >= scale:
            return f"{float(f'{value_m / scale:.2g}'):g} {unit}"
    return f"{float(f'{value_m * 1e9:.2g}'):g} nm"


def _opaque(colour, alpha=0.6):
    """The colour of ``colour`` at ``alpha`` over white, as an opaque colour."""
    from matplotlib.colors import to_rgb
    return tuple(alpha * np.array(to_rgb(colour)) + (1 - alpha))


# ------------------------------------------------------------------------- the column
def column_panel(ax, theta, dac, voltage, reference, *, xmax, bottom, camera_half_width, title,
                 colour=INK, fill=FILL["truth"], truth_planes=(), beam_half_width=20e-9):
    """One column, specimen at the top: a parallel beam (2 x ``beam_half_width`` wide) through the
    lenses to the camera, traced by TemGym's ``plot_model`` in the frame that rotates with the
    lenses.

    Lenses are rings the beam passes through: the back of each ring is hidden by the beam, the
    front is drawn over it. The camera is drawn at its true width; the transverse axis is linear
    within the beam's starting width and logarithmic beyond, so the beam is visible at the
    specimen and at the camera. ``truth_planes`` (m) are drawn dashed.
    """
    from matplotlib.patches import Arc
    from matplotlib.transforms import blended_transform_factory
    from temgym_core.calibration.projector_model import at_voltage, projector_components
    from temgym_core.plotting import legacy_beam_plot_params, plot_model, trace_ray_paths
    from temgym_core.ray import Ray

    column = projector_components(at_voltage(jnp.asarray(theta[:16]), voltage, reference), jnp.asarray(P.controls(dac)))
    x = np.linspace(-beam_half_width, beam_half_width, 21)
    zero = np.zeros_like(x)
    rays = Ray(x=x, y=zero, dx=zero, dy=zero, z=0.0, pathlength=0.0, voltage=voltage)
    style = legacy_beam_plot_params(fixed_xmax=xmax, fill_color=_opaque(fill), fill_alpha=1.0,
                                    interior_ray_alpha=0.0, center_ray_alpha=0.0, ray_alpha=0.6,
                                    edge_lw=1.0, show_side_guides=False)
    plot_model(column, rays=rays, ax=ax, ray_coordinate="x_rot", plot_params=style)
    for artist in [*ax.texts, *ax.patches]:                     # keep only the beam
        artist.remove()
    for artist in [*ax.collections, *ax.lines]:
        artist.set_zorder(5)
    ax.set_xscale("asinh", linear_width=beam_half_width)
    ax.set(xlim=(-xmax, xmax), ylim=(bottom, -0.03 * bottom), xticks=[], yticks=[])
    ax.set_xticks([], minor=True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    across = blended_transform_factory(ax.transAxes, ax.transData)
    lens_z = [float(c.z + c.length / 2) for c in column[:-1]]
    camera_z = float(column[-1].z)
    for z in lens_z:
        for theta1, theta2, z_order in ((180, 360, 2), (0, 180, 10)):   # back (hidden), front
            ax.add_patch(Arc((0.5, z), 0.96, 0.018 * bottom, theta1=theta1, theta2=theta2, transform=across,
                             lw=1.4, color="0.15", zorder=z_order))
    ax.plot([-camera_half_width, camera_half_width], [camera_z] * 2, color="0.35", lw=5,
            solid_capstyle="butt", zorder=10)
    for z in truth_planes:
        ax.plot([0.0, 1.0], [z] * 2, transform=across, color="0.35", lw=1.2, ls=(0, (4, 3)), zorder=9)
    positions = [*lens_z, camera_z]
    for i in range(1, len(positions)):                         # keep labels apart
        positions[i] = max(positions[i], positions[i - 1] + 0.05 * bottom)
    for z, name in zip(positions, COLUMN_LABELS):
        ax.text(1.02, z, name, transform=across, fontsize=12, va="center")
    ax.plot(0, 0, "o", color="#795548", ms=7, zorder=11, clip_on=False)
    _, traced = trace_ray_paths(column, rays, ray_coordinate="x_rot")
    for z, text in ((0.0, _length(2 * beam_half_width)), (camera_z, _length(np.ptp(traced[-1])))):
        ax.text(-0.02, z, text, transform=across, fontsize=10, color=INK2, va="center", ha="right")
    ax.set_title(title, fontsize=14, color=colour, pad=12)


def column_panels(panels, reference, camera, *, truth=None, height=7.5):
    """Several :func:`column_panel` side by side on one scale. ``panels`` holds
    (title, theta, DAC values, HT, title colour, beam colour); with ``truth`` the true lens and
    camera planes are dashed in every panel that is not the truth itself."""
    fig = plt.figure(figsize=(4.4 * len(panels), height))
    bottom = 1.03 * max(P.positions_mm(theta)[-1] / 1e3 for _, theta, *_ in panels)
    half_width = camera.n * camera.pitch / 2
    planes = () if truth is None else list(P.positions_mm(truth) / 1e3)
    width = 1.0 / len(panels)
    for k, (title, theta, dac, voltage, colour, fill) in enumerate(panels):
        ax = fig.add_axes([0.08 + k * width, 0.04, 0.55 * width, 0.86])
        column_panel(ax, theta, dac, voltage, reference, xmax=1.06 * half_width, bottom=bottom,
                     camera_half_width=half_width, title=title, colour=colour, fill=fill,
                     truth_planes=planes if theta is not truth else ())
    return fig


def session_frames(folder, visits=(0, 4, 9)):
    """Frames of a saved stage-1 session: no kick, -/+ shift, -/+ tilt at the given visits;
    + marks the measured centre."""
    import json
    from pathlib import Path

    from stage1_acquire import measure_spot
    folder = Path(folder)
    log = json.loads((folder / "log.json").read_text())
    dark = [e for e in log if e["role"] == "dark"]
    background = float(np.median(np.load(folder / dark[0]["file"]))) if dark else 0.0
    fig, axes = plt.subplots(len(visits), 5, figsize=(12, 2.5 * len(visits)), squeeze=False)
    for row, v in enumerate(visits):
        entries = [e for e in log if e["visit"] == v]
        pick = [next(e for e in entries if e["role"] == "base")]
        for kind in ("shift", "tilt"):
            pick += [[e for e in entries if e["kind"] == kind and e["role"] == role][-1] for role in ("minus", "plus")]
        for col, (e, name) in enumerate(zip(pick, ("No kick", "− shift", "+ shift", "− tilt", "+ tilt"))):
            img = np.load(folder / e["file"]).astype(float) - background
            ax = axes[row, col]
            ax.imshow(np.sqrt(np.clip(img, 0, None)), origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            c = measure_spot(img)[0] + (np.array(img.shape[::-1]) - 1) @ np.array([0.5, 0.5j])
            ax.plot(c.real, c.imag, "+", color=YELLOW, ms=11, mew=1.6)
            if row == 0:
                ax.set_title(name, loc="center", fontweight="normal")
            if col == 0:
                ax.set_ylabel(f"visit {v + 1}: setting {entries[0]['setting']}", fontsize=11)
    fig.tight_layout()


def prediction_check_figure(rows):
    """Measured and predicted beam moves at unseen lens settings (arrows from the origin), and the
    error of each prediction."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, kind, tag in ((axes[0], "shift", "(a)"), (axes[1], "tilt", "(b)")):
        sel = [r for r in rows if r["kind"] == kind]
        for r in sel:
            for z, colour, lw in ((r["measured"], INK, 2.2), (r["predicted"], ORANGE, 1.2)):
                ax.annotate("", xy=(z.real, z.imag), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=colour, lw=lw))
        span = 1.15 * max(max(abs(r["measured"]), abs(r["predicted"])) for r in sel)
        ax.set(xlim=(-span, span), ylim=(-span, span), aspect="equal", xlabel="x on the camera (px)",
               ylabel="y on the camera (px)")
        ax.set_title(f"{tag}  {kind.capitalize()}: measured (black), predicted (orange)")
    settings = sorted({r["setting"] for r in rows}, key=lambda name: int(name.split()[-1]))
    x = {name: i for i, name in enumerate(settings)}
    for kind, colour, offset in (("shift", BLUE, -0.2), ("tilt", ORANGE, 0.2)):
        sel = [r for r in rows if r["kind"] == kind]
        axes[2].bar([x[r["setting"]] + offset for r in sel], [r["error"] for r in sel], 0.4, color=colour, label=kind)
    axes[2].set_xticks(range(len(settings)), [name.split()[-1] for name in settings])
    axes[2].set_xlabel("unseen lens setting")
    axes[2].set_ylabel("|measured − predicted| (px)"); axes[2].legend(); axes[2].set_title("(c)  Prediction error")
    fig.tight_layout()
