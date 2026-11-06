"""Figures and tables for the projector calibration, in the style of the paper figure: the column
drawn vertically with lenses as rings and the beam as a coloured band (:func:`column_panels`),
Okabe-Ito colours, bold panel titles, no grid. Every figure applies that style itself.

    import figures

    figures.frames(folder)                          # frames of a session, centres marked
    figures.prediction_check(analyse.compare(...))  # measured against predicted beam moves
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import functools

import numpy as np
from IPython.display import Markdown, display

import model as M

# Paper colours: ink, then blue / vermilion / bluish green lines, with the matching beam fills.
INK, INK2, LIGHT = "#222222", "#595959", "#ececec"
BLUE, ORANGE, GREEN, YELLOW = "#0072B2", "#D55E00", "#009E73", "#F0E442"
FILL = {"truth": "#43C78E", "orange": "#F5A35C", "blue": "#6BAED6"}
COLUMN_LABELS = ("Objective", "IL1", "IL2", "IL3", "PL1", "Camera")
SET_LABELS = [str(i) for i in range(8)] + ["diff."]


#: The paper style, applied by every figure function while it draws (nothing global changes).
STYLE = {"figure.dpi": 100, "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
         "axes.titlelocation": "left", "axes.labelsize": 11, "axes.grid": False,
         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
         "xtick.color": INK2, "ytick.color": INK2, "lines.linewidth": 1.8, "image.cmap": "gray",
         "legend.frameon": False, "legend.fontsize": 10}


def _styled(draw):
    """Draw ``draw``'s figure in the paper style and return it, closed in pyplot (a notebook shows
    the returned figure once; nothing is left open)."""
    @functools.wraps(draw)
    def wrapper(*args, **kwargs):
        with plt.rc_context(STYLE):
            fig = draw(*args, **kwargs)
        plt.close(fig)
        return fig
    return wrapper


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
                 colour=INK, fill=FILL["truth"], truth_planes=(), beam_half_width=20e-9, sizes=True):
    """One column, specimen at the top: a parallel beam (2 x ``beam_half_width`` wide) through the
    lenses to the camera, traced by TemGym's ``plot_model`` in the frame that rotates with the
    lenses.

    Lenses are rings the beam passes through: the back of each ring is hidden by the beam, the
    front is drawn over it. The camera is drawn at its true width; the transverse axis is linear
    within the beam's starting width and logarithmic beyond, so the beam is visible at the
    specimen and at the camera. ``truth_planes`` (m) are drawn dashed. ``sizes`` labels the beam
    width at the specimen and at the camera.
    """
    from matplotlib.patches import Arc
    from matplotlib.transforms import blended_transform_factory
    from temgym_core.calibration.projector_model import at_voltage, projector_components
    from temgym_core.plotting import legacy_beam_plot_params, plot_model, trace_ray_paths
    from temgym_core.ray import Ray

    column = projector_components(at_voltage(jnp.asarray(theta[:16]), voltage, reference), jnp.asarray(M.controls(dac)))
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
    for z, true_z, name in zip(positions, [*lens_z, camera_z], COLUMN_LABELS):
        ax.text(1.04, z, name, transform=across, fontsize=12, va="center")
        if abs(z - true_z) > 1e-3 * bottom:                    # label moved apart: a leader to its plane
            ax.plot([0.985, 1.03], [true_z, z], transform=across, color=INK2, lw=0.8, clip_on=False, zorder=10)
    ax.plot(0, 0, "o", color="#795548", ms=7, zorder=11, clip_on=False)
    if sizes:
        _, traced = trace_ray_paths(column, rays, ray_coordinate="x_rot")
        for z, text in ((0.0, _length(2 * beam_half_width)), (camera_z, _length(np.ptp(traced[-1])))):
            ax.text(-0.02, z, text, transform=across, fontsize=10, color=INK2, va="center", ha="right")
    ax.set_title(title, fontsize=14, color=colour, pad=12)


@_styled
def column_panels(panels, reference, camera, *, truth=None, height=7.5, sizes=True):
    """Several :func:`column_panel` side by side on one scale. ``panels`` holds
    (title, theta, DAC values, HT, title colour, beam colour); with ``truth`` the true lens and
    camera planes are dashed in every panel that is not the truth itself."""
    fig = plt.figure(figsize=(4.4 * len(panels), height))
    bottom = 1.03 * max(M.positions_mm(theta)[-1] / 1e3 for _, theta, *_ in panels)
    half_width = camera.n * camera.pitch / 2
    planes = () if truth is None else list(M.positions_mm(truth) / 1e3)
    width = 1.0 / len(panels)
    for k, (title, theta, dac, voltage, colour, fill) in enumerate(panels):
        ax = fig.add_axes([0.08 + k * width, 0.04, 0.55 * width, 0.86])
        column_panel(ax, theta, dac, voltage, reference, xmax=1.06 * half_width, bottom=bottom,
                     camera_half_width=half_width, title=title, colour=colour, fill=fill,
                     truth_planes=planes if theta is not truth else (), sizes=sizes)
    return fig


@_styled
def frames(folder, visits=(0, 4, 9)):
    """Frames of a saved stage-1 session: no kick, -/+ shift, -/+ tilt at the given visits;
    + marks the measured centre."""
    import json
    from pathlib import Path

    from acquire import measure_spot
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
    return fig


@_styled
def prediction_check(rows):
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
    axes[2].set_ylabel("|measured − predicted| (px)"); axes[2].legend()
    axes[2].set_title(f"(c)  Prediction error: largest {max(r['error'] for r in rows):.3f} px")
    fig.tight_layout()
    return fig


@_styled
def specimen_frames(grating, gold, result):
    """The two specimen frames side by side, with the measured magnification and camera length."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    axes[0].imshow(grating, cmap="gray")
    axes[0].set_title(f"(a)  cross grating, 40k row: |A| = {result['magnification']:.0f}")
    axes[1].imshow(np.arcsinh(np.clip(gold, 0, None) / max(np.percentile(gold, 50), 1.0)), cmap="magma")
    axes[1].set_title(f"(b)  gold, diffraction: |B| = {result['camera_length_m']:.4f} m")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    return fig


@_styled
def model_response(model, folder, units=1000):
    """The stage-1 model as it is known, unscaled: at each of the session's lens settings, the beam
    move on the camera (px, camera axes) for ``units`` deflector units of CLA1 (shift, top row) and
    of CLA2 (tilt, bottom row). Arrows show the direction and the number the size; arrow lengths
    are on a log scale over three decades within a row, and a dot marks a move below that."""
    import json
    from pathlib import Path

    import analyse

    settings = json.loads((Path(folder) / "plan.json").read_text())["settings"]
    labels = list(settings)
    moves = {kind: [complex(analyse.predict(model, settings[k], **{kind: (units, 0)})[0]) for k in labels]
             for kind in ("shift", "tilt")}
    fig, axes = plt.subplots(2, len(labels), figsize=(1.6 * len(labels), 4.4))
    for row, (kind, name, colour) in enumerate((("shift", "CLA1 (shift)", BLUE), ("tilt", "CLA2 (tilt)", ORANGE))):
        largest = max(abs(m) for m in moves[kind])
        for col, (label, move) in enumerate(zip(labels, moves[kind])):
            ax = axes[row, col]
            ax.set(xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), aspect="equal", xticks=[], yticks=[])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color=LIGHT, lw=1.0))
            ratio = abs(move) / largest
            if ratio > 1e-3:                                     # log length: 1 at the largest, 0 at 1/1000
                end = (1 + np.log10(ratio) / 3) * move / abs(move)
                ax.annotate("", xy=(end.real, end.imag), xytext=(0, 0),
                            arrowprops=dict(arrowstyle="-|>", color=colour, lw=2.0, mutation_scale=12))
            else:
                ax.plot(0, 0, "o", color=colour, ms=4)
            ax.text(0, -1.45, f"{abs(move):.2g} px", ha="center", va="top", fontsize=10, color=INK)
            if row == 0:
                ax.set_title("diff." if label == "diffraction" else f"setting {label}", loc="center",
                             fontsize=11, fontweight="normal")
        axes[row, 0].text(-1.6, 0, name, rotation=90, ha="center", va="center", fontsize=11, color=colour,
                          fontweight="bold")
    fig.suptitle(f"Stage-1 model, unscaled: beam move on the camera per {units} deflector units "
                 "(arrow length on a log scale)",
                 x=0.02, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


@_styled
def model_family(model, columns):
    """Equally good stage-1 columns (from :func:`analyse.equivalent_columns`) side by side at the
    40k row: each assumes different metres per shift unit and radians per tilt unit, and all
    predict the same beam moves. None is physical; stage 2 picks the physical one."""
    units = np.asarray(model["units"])
    dac, ht = np.asarray(model["base_dac"]) * units, model["ht_v"]
    camera = M.Camera(n=model["frame_width"], pitch=model["pitch_m"])
    panels = [(f"({chr(97 + k)})  χ²/dof {c['chi2_dof']:.2f}", np.asarray(c["theta"]), dac, ht, INK2, LIGHT)
              for k, c in enumerate(columns)]
    fig = column_panels(panels, ht, camera, sizes=False)
    fig.suptitle("Equally good stage-1 columns: unscaled, not physical", x=0.04, ha="left", y=1.0,
                 fontsize=14, fontweight="bold", color=INK)
    fig.text(0.04, -0.03, "Each column assumes different metres per shift unit and radians per tilt unit; all "
             "predict the same beam\nmoves, so stage 1 cannot tell them apart. Stage 2 (specimen frames and a "
             "second HT) picks the physical one.", ha="left", va="top", fontsize=11, color=INK2)
    return fig

