"""Figures and tables for the projector calibration, in the style of the paper figure: the column
drawn vertically with lenses as rings and the beam as a coloured band (:func:`column_panels`),
Okabe-Ito colours, bold panel titles, no grid. Every figure applies that style itself. Every
function takes named arguments only.

    import figures

    figures.frames(folder=recording.folder)               # frames of a recording, centres marked
    figures.prediction_test(model=model, folder=new.folder)   # the beam at new settings, predicted and measured
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


def table(*, header, rows):
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
def column_panel(*, ax, theta, dac, voltage, reference, xmax, bottom, camera_half_width, title,
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
def column_panels(*, panels, reference, camera, truth=None, height=7.5, sizes=True):
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
        column_panel(ax=ax, theta=theta, dac=dac, voltage=voltage, reference=reference, xmax=1.06 * half_width,
                     bottom=bottom, camera_half_width=half_width, title=title, colour=colour, fill=fill,
                     truth_planes=planes if theta is not truth else (), sizes=sizes)
    return fig


@_styled
def frames(*, folder, measurements=(1, 5, 10)):
    """Frames of a recording: no kick, -/+ shift, -/+ tilt at the given measurements (numbered from 1);
    + marks the measured centre. Each row shows the part of the frame that holds all five beams."""
    import json
    from pathlib import Path

    from acquire import measure_spot
    folder = Path(folder)
    log = json.loads((folder / "log.json").read_text())
    dark = [e for e in log if e["role"] == "dark"]
    background = float(np.median(np.load(folder / dark[0]["file"]))) if dark else 0.0
    fig, axes = plt.subplots(len(measurements), 5, figsize=(12, 2.5 * len(measurements)), squeeze=False)
    for row, number in enumerate(measurements):
        entries = [e for e in log if e["measurement"] == number]
        pick = [next(e for e in entries if e["role"] == "base")]
        for kind in ("shift", "tilt"):
            pick += [[e for e in entries if e["kind"] == kind and e["role"] == role][-1] for role in ("minus", "plus")]
        images = [np.load(folder / e["file"]).astype(float) - background for e in pick]
        spots = [measure_spot(frame=img) for img in images]
        offset = (np.array(images[0].shape[::-1]) - 1) @ np.array([0.5, 0.5j])
        centres = np.array([spot[0] for spot in spots]) + offset
        margin = max(10.0, 4 * max(spot[1] for spot in spots))
        middle = (centres.real.min() + centres.real.max()) / 2 + 1j * (centres.imag.min() + centres.imag.max()) / 2
        half = max(np.ptp(centres.real), np.ptp(centres.imag)) / 2 + margin
        for col, (img, c, name) in enumerate(zip(images, centres, ("No kick", "− shift", "+ shift", "− tilt", "+ tilt"))):
            ax = axes[row, col]
            ax.imshow(np.sqrt(np.clip(img, 0, None)), origin="lower")
            ax.set(xticks=[], yticks=[], xlim=(middle.real - half, middle.real + half),
                   ylim=(middle.imag - half, middle.imag + half))
            ax.plot(c.real, c.imag, "+", color=YELLOW, ms=9, mew=1.0)
            if row == 0:
                ax.set_title(name, loc="center", fontweight="normal")
            if col == 0:
                ax.set_ylabel(f"{number}. {pick[0]['setting']}", fontsize=9)
    fig.tight_layout()
    return fig


@_styled
def prediction_test(*, model, folder):
    """The beam at new lens settings: for each setting (rows), the frames with the + and - shift
    and tilt kicks. Orange: the predicted beam (the frame without a kick, moved by the model's
    predicted beam move, outlined at half its peak, + at its centre); yellow +: the measured centre.
    Each panel shows the region around the beam; its title is the distance between the two (px).
    Frames whose beam is cut by the frame edge are not used (as in the analysis)."""
    import json
    from pathlib import Path

    from scipy.ndimage import shift as move_image

    import analyse
    from acquire import measure_spot
    from analyse import EDGE_LIMIT
    folder = Path(folder)
    log = json.loads((folder / "log.json").read_text())
    dark = [e for e in log if e["role"] == "dark"]
    background = float(np.median(np.load(folder / dark[0]["file"]))) if dark else 0.0
    numbers = sorted({e["measurement"] for e in log if e["measurement"] >= 1})
    columns = [("shift", "plus", "+ shift"), ("shift", "minus", "− shift"), ("tilt", "plus", "+ tilt"),
               ("tilt", "minus", "− tilt")]
    fig, axes = plt.subplots(len(numbers), 4, figsize=(9.5, 2.45 * len(numbers)), squeeze=False)
    largest = 0.0
    for row, number in enumerate(numbers):
        entries = [e for e in log if e["measurement"] == number]
        base_entry = next(e for e in entries if e["role"] == "base")
        base = np.load(folder / base_entry["file"]).astype(float) - background
        c0, rms, _ = measure_spot(frame=base)
        half_width = max(10.0, 5 * rms)                                # the region shown around the beam
        centre = (np.array(base.shape[::-1]) - 1) @ np.array([0.5, 0.5j])
        for col, (kind, role, name) in enumerate(columns):
            ax = axes[row, col]
            ax.set_xticks([]); ax.set_yticks([])
            pick = [e for e in entries if e["kind"] == kind and e["role"] == role]
            if not pick:
                ax.set_title("cut by the edge, not used", loc="center", fontsize=9, fontweight="normal")
                ax.axis("off")
                continue
            e = pick[-1]
            img = np.load(folder / e["file"]).astype(float) - background
            ax.imshow(np.sqrt(np.clip(img, 0, None)), origin="lower")
            measured, _, edge = measure_spot(frame=img)
            if edge > EDGE_LIMIT:
                ax.set_title("cut by the edge, not used", loc="center", fontsize=9, fontweight="normal")
                continue
            kick = tuple(e["offset"])
            predicted = c0 + complex(analyse.predict(model=model, dac=e["dac"],
                                                     **{kind: kick, ("tilt" if kind == "shift" else "shift"): (0, 0)})[0])
            outline = move_image(base, ((predicted - c0).imag, (predicted - c0).real), order=1)
            ax.contour(outline, levels=[0.5 * outline.max()], colors=ORANGE, linewidths=1.2)
            for z, colour, size in ((predicted, ORANGE, 12), (measured, YELLOW, 7)):
                z = z + centre
                ax.plot(z.real, z.imag, "+", color=colour, ms=size, mew=1.0)
            view = predicted + centre
            ax.set(xlim=(view.real - half_width, view.real + half_width),
                   ylim=(view.imag - half_width, view.imag + half_width))
            difference = abs(measured - predicted)
            largest = max(largest, difference)
            ax.set_title(f"{difference:.3f} px", loc="center", fontsize=10, fontweight="normal")
            if row == 0:
                ax.text(0.5, 1.22, name, transform=ax.transAxes, ha="center", fontsize=12, fontweight="bold")
        axes[row, 0].set_ylabel(base_entry["setting"], fontsize=11)
    fig.suptitle(f"The beam at new lens settings: predicted (orange) and measured (yellow); "
                 f"largest difference {largest:.3f} px", x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


@_styled
def specimen_frames(*, grating, gold, result):
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
