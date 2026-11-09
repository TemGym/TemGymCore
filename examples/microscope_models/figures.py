"""Stage-1 figures in the style of the paper figure: Okabe-Ito colours, bold panel titles, no grid. Every figure applies that style itself. Every
function takes named arguments only.

    import figures

    figures.frames(folder=recording.folder)               # frames of a recording, centres marked
    figures.prediction_test(model=model, folder=new.folder)   # the beam at new settings, predicted and measured
"""

import matplotlib.pyplot as plt
import functools

import numpy as np


# Paper colours: ink, then blue / vermilion / bluish green lines, with the matching beam fills.
INK, INK2, LIGHT = "#222222", "#595959", "#ececec"
BLUE, ORANGE, GREEN, YELLOW = "#0072B2", "#D55E00", "#009E73", "#F0E442"


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


# ------------------------------------------------------------------------- the column


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


