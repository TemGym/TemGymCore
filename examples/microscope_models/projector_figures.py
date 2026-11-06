"""Figures and tables for ``projector_calibration.ipynb``, in the style of the paper figure
(``figures/projector_inversion.png``): the column drawn vertically with lenses as rings and the
beam as a coloured band, Okabe-Ito colours, bold panel titles, no grid.
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


def responses_figure(theta, setup, camera_rotation):
    """|A|, |B| and the image rotation on the camera at the 9 stage-1 settings, at both HTs."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    x = np.arange(9)
    for k, (v, colour) in enumerate(((setup.v0, BLUE), (setup.v1, ORANGE))):
        a, b = P.responses(theta[:16], setup.sets, np.full(9, v), setup.v0)
        w = 0.38
        axes[0].bar(x + (k - 0.5) * w, np.abs(a), w, color=colour, label=f"{v / 1e3:.0f} kV")
        axes[1].bar(x + (k - 0.5) * w, np.abs(b), w, color=colour)
        axes[2].bar(x + (k - 0.5) * w, np.degrees(np.angle(a * np.exp(1j * camera_rotation))), w, color=colour)
    axes[0].set_yscale("log"); axes[0].set_title("(a)  |A|, magnification")
    axes[1].set_yscale("log"); axes[1].set_ylim(1e-4, 30); axes[1].set_title("(b)  |B|, camera length (m/rad)")
    axes[2].set_title("(c)  Image rotation on the camera (°)")
    for ax in axes:
        ax.set_xticks(x, SET_LABELS); ax.set_xlabel("lens setting")
    axes[0].legend()
    fig.tight_layout()


# ------------------------------------------------------------------------- frames and data
def frames_figure(frames, setup, settings=(0, 4, 8), voltage=None):
    """Frames at several settings: no kick, -/+ shift, -/+ tilt; + marks the measured centroid."""
    voltage = setup.v0 if voltage is None else voltage
    fig, axes = plt.subplots(len(settings), 5, figsize=(12, 2.5 * len(settings)), squeeze=False)
    off = (setup.camera.n - 1) / 2 * (1 + 1j)
    for row, i in enumerate(settings):
        f = frames[(i, voltage)]
        images = [f["base"], f["shift"][0], f["shift"][1], f["tilt"][0], f["tilt"][1]]
        for col, (img, name) in enumerate(zip(images, ("No kick", "− shift", "+ shift", "− tilt", "+ tilt"))):
            ax = axes[row, col]
            ax.imshow(np.sqrt(img), origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            c = P.centroid(img, setup.camera) + off
            ax.plot(c.real, c.imag, "+", color=YELLOW, ms=11, mew=1.6)
            if row == 0:
                ax.set_title(name, loc="center", fontweight="normal")
            if col == 0:
                ax.set_ylabel(f"setting {SET_LABELS[i]}", fontsize=12)
    fig.tight_layout()


def displacements_figure(data, setup):
    """Measured half-differences (beam displacement per kick) as arrows, one per setting."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, kind, name, tag in ((axes[0], 0, "shift", "(a)"), (axes[1], 1, "tilt", "(b)")):
        sel = (data["kind"] == kind) & (data["voltage"] == setup.v0)
        short = []
        for m, dac in zip(data["measured"][sel], data["dac"][sel]):
            label = SET_LABELS[int(np.flatnonzero((setup.sets == dac).all(1))[0])]
            ax.annotate("", xy=(m.real, m.imag), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.6))
            if abs(m) > 8:
                ax.text(m.real + 4.5 * m.real / abs(m), m.imag + 4.5 * m.imag / abs(m), label, ha="center", va="center")
            else:
                short.append(f"{label}: {abs(m):.2f}")
        ax.text(0.02, 0.02, "short arrows (px): " + ", ".join(short), transform=ax.transAxes, fontsize=9, color=INK2)
        ax.axhline(0, color=LIGHT, lw=1, zorder=0); ax.axvline(0, color=LIGHT, lw=1, zorder=0)
        ax.set_aspect("equal"); ax.set_xlim(-60, 60); ax.set_ylim(-60, 60)
        ax.set_xlabel("x on the camera (px)"); ax.set_ylabel("y on the camera (px)")
        ax.set_title(f"{tag}  Beam displacement per {name}")
    fig.tight_layout()


def stage1_fit_figure(result, data, shift_errors, tilt_errors):
    """Normalised residuals of a fit, and its beam-position errors at unseen settings."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    n = len(data["kind"])
    axes[0].axhspan(-1, 1, color=LIGHT, lw=0)
    axes[0].plot(result["residuals"][:n], "o", color=BLUE, ms=6, label="x")
    axes[0].plot(result["residuals"][n:2 * n], "s", color=ORANGE, ms=6, label="y")
    axes[0].set_xlabel("measurement"); axes[0].set_ylabel("residual / σ"); axes[0].legend()
    axes[0].set_title("(a)  Residuals (grey: ±1σ)")
    bins = np.linspace(0, max(np.max(shift_errors), np.max(tilt_errors)) * 1.05, 20)
    axes[1].hist(shift_errors, bins=bins, color=BLUE, label="shift")
    axes[1].hist(tilt_errors, bins=bins, color=ORANGE, alpha=0.75, label="tilt")
    axes[1].set_xlabel("beam position error for a 50 px kick (px)"); axes[1].set_ylabel("lens settings")
    axes[1].legend(); axes[1].set_title("(b)  Prediction at 60 unseen lens settings")
    fig.tight_layout()


def anchors_figure(grating, centres, lattice, rings, radii, ring_info, camera):
    """The grating image with square centres and lattice vectors; the gold rings and profile."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    off = (camera.n - 1) / 2 * (1 + 1j)
    axes[0].imshow(grating, origin="lower")
    c = centres + off
    axes[0].plot(c.real, c.imag, "+", color=YELLOW, ms=14, mew=2)
    for v, colour in zip(lattice, (BLUE, ORANGE)):
        axes[0].annotate("", xy=(c[0].real + v[0], c[0].imag + v[1]), xytext=(c[0].real, c[0].imag),
                         arrowprops=dict(arrowstyle="->", color=colour, lw=2.2))
    axes[0].set_title("(a)  Cross grating, setting 0")
    axes[1].imshow(np.sqrt(rings), origin="lower")
    t = np.linspace(0, 2 * np.pi, 400)
    cc = ring_info["centre"] + off
    for k, colour in zip(radii, (BLUE, ORANGE)):
        axes[1].plot(cc.real + radii[k] * np.cos(t), cc.imag + radii[k] * np.sin(t), color=colour, lw=1.2, label=f"Au {k}")
    axes[1].legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")
    axes[1].set_title("(b)  Gold, diffraction setting")
    for ax in axes[:2]:
        ax.set_xticks([]); ax.set_yticks([])
    axes[2].plot(ring_info["bins"], ring_info["profile"] / ring_info["profile"].max(), color=INK2, lw=1.5)
    for k, colour in zip(radii, (BLUE, ORANGE)):
        axes[2].axvline(radii[k], color=colour, lw=1.6, label=f"Au {k}: {radii[k]:.2f} px")
    axes[2].set_xlim(40, camera.n / 2); axes[2].set_xlabel("radius (px)"); axes[2].set_ylabel("intensity (norm.)")
    axes[2].legend(); axes[2].set_title("(c)  Radial profile of (b)")
    fig.tight_layout()


def design_figure(setup, designed):
    """Designed lens settings against the stage-1 settings, in DAC values."""
    lo, hi = setup.design_box
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (i, j), tag in zip(axes, ((0, 1), (2, 3)), ("(a)", "(b)")):
        ax.add_patch(plt.Rectangle((lo[i], lo[j]), hi[i] - lo[i], hi[j] - lo[j], fill=False, ec=INK2, ls="--", lw=1))
        ax.plot(setup.sets[:, i], setup.sets[:, j], "o", color=BLUE, ms=9, label="stage 1")
        ax.plot(designed[:, i], designed[:, j], "D", color=GREEN, ms=7, mec="white", mew=1, label="designed (stage 2b)")
        ax.set_xlabel(f"{P.LENSES[i]} DAC value"); ax.set_ylabel(f"{P.LENSES[j]} DAC value")
        ax.set_title(f"{tag}  {P.LENSES[i]} and {P.LENSES[j]}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles + [plt.Line2D([], [], color=INK2, ls="--", lw=1)], labels + ["DAC-feasible range"],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))


def positions_figure(fits, truth_mm):
    """Fitted minus true positions with ±1σ, for each named fit."""
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(6)
    for k, ((name, (pos, sig)), colour) in enumerate(zip(fits.items(), (ORANGE, BLUE))):
        ax.errorbar(x + (k - 0.5) * 0.25, pos - truth_mm, yerr=sig, fmt="o", color=colour, ms=7, capsize=3,
                    lw=1.5, label=f"{name} (±1σ)")
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(x, COLUMN_LABELS); ax.set_ylabel("fitted − true (mm)")
    ax.set_yscale("symlog", linthresh=1); ax.legend()
    ax.set_title("Lens and camera positions")
    fig.tight_layout()


def zscore_figure(z):
    """(fitted - true) / sigma for each position, one dot per run."""
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.axhspan(-1, 1, color=LIGHT, lw=0); ax.axhline(0, color=INK, lw=1)
    for k in range(z.shape[1]):
        ax.plot(np.full(len(z), k) + np.linspace(-0.15, 0.15, len(z)), z[:, k], "o", color=BLUE, ms=6)
    ax.set_xticks(range(z.shape[1]), COLUMN_LABELS); ax.set_ylabel("(fitted − true) / σ")
    ax.set_title(f"z-scores of the final positions, {len(z)} runs (grey: ±1)")
    fig.tight_layout()


# ------------------------------------------------------------------------- a stage-1 session
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
