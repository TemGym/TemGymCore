"""Reflection SEM figure for the ERC Part B1: explore the model, then fit it.

(a) The live notebook view: sliders above the native column for Cu(001) 2×1
at 3°, beside the diffraction pattern it produces on the detector. (b) A hidden
ground-truth microscope makes the measured patterns; the fit starts from a
guess with every element misplaced (dotted) and recovers the true microscope
(solid). 170 mm wide, all text 8 pt or larger. Reads saved outputs; nothing is
re-simulated or refitted.
"""
from make_calibration_workflow_figure import DATA, OUT, INK, MUTED, TEAL, OCHRE, image_stack, arrow
from make_microscope_ptychography_figure import use_arial, notebook_gui, mini_microscope
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse, Rectangle
from scipy.ndimage import gaussian_filter, maximum_filter
from temgym_core import reflection_calibration as calibration
from temgym_core.plotting import export_ray_figure

WIDTH_MM, HEIGHT_MM = 170, 125
MIN_PT = 8
ROW = 52        # comparison acquisition, as in the earlier figures
# Initial guess with every element clearly misplaced and its Bragg spots still
# on the detector; refine_b1_guess.py fits from it and records that it reaches
# the saved best fit.
START = DATA / "b1_figure_start.json"
# One restrained palette: neutral hardware, warm guess, teal model and fit,
# and a single colormap for every detector image.
HARDWARE = "#8C99A1"
GUESS_RING = "#F5C451"
FIT_RING = "#8FF0DE"
CMAP = "magma"
TITLE = ("Using my electron microscope simulation software to prototype\n"
         "the reflection electron microscope and infer its hidden optical state")
# Saved live-view state, rendered by render_b1_detector_view.py: it drives the
# column, the sliders and the detector image.
VIEW = DATA.parent / "copper_b1" / "incidence_3"


def heading(fig, y, letter, title, rest):
    fig.text(.012, y, letter, fontsize=10.5, weight="bold", color=INK)
    label = fig.text(.036, y, title, fontsize=10.5, weight="bold", color=INK)
    # Continue the sentence in regular weight right after the bold verb.
    end = label.get_window_extent(fig.canvas.get_renderer()).x1
    fig.text(fig.transFigure.inverted().transform((end, 0))[0] + .006, y, rest,
             fontsize=10.5, color=INK)


def calm(ax):
    """Neutral lens coils and deflectors; the beam keeps the colour."""
    for patch in ax.patches:
        if patch.get_gid() == "reflection_lens_coil":
            patch.set_facecolor(HARDWARE); patch.set_edgecolor(HARDWARE); patch.set_hatch(None)
    for line in ax.lines:
        if line.get_gid() == "reflection_deflector":
            line.set_color(HARDWARE)


def explore(fig):
    """Live view (a): sliders and column, beside the detector at three angles."""
    gui = notebook_gui(fig, left=.0, bottom=.49, width=.6, text_size=8.5,
                       slider_y=.81, height=.25, view_dir=VIEW,
                       offsets=[(-4, 30), (-12, -27), (-30, 24), (20, 34), (14, -26)])
    column = fig.axes[-1]
    # Heights are exaggerated in this view; a horizontal scale bar replaces the axes.
    column.set_axis_off()
    for text in list(column.texts):
        if text.get_text() == "Cu(001) specimen":
            text.set_text("Sample")
            x, y = text.get_position()
            text.set_position((x + 11, y + 1.2))
        elif text.get_text() == "Detector":
            # Beside the detector bar rather than across its top.
            x, y = text.get_position()
            text.set_position((x + 7, y)); text.set_ha("left")
        elif text.get_text().startswith("Deflector"):
            text.remove()
    column.plot([-130, -30], [-16.5, -16.5], color=INK, lw=1, solid_capstyle="butt")
    column.text(-80, -15.8, "100 mm", ha="center", va="bottom", fontsize=8, color=INK)
    calm(column)

    with np.load(VIEW / "camera_view.npz") as saved:
        intensity = saved["intensity"].T   # (lateral, height) -> (height, lateral)
    pitch = json.loads((VIEW / "camera_view.json").read_text())["parameters"]["pixel_pitch"] * 1e3
    rows, cols = intensity.shape
    extent = np.array([-cols, cols, -rows, rows]) * pitch / 2.
    # Crop to the beams on the Laue arc.
    lit = intensity > 1e-6 * intensity.max()
    lit_rows, lit_cols = np.flatnonzero(lit.any(axis=1)), np.flatnonzero(lit.any(axis=0))
    band = (np.array([lit_rows[0], lit_rows[-1] + 1]) - rows / 2) * pitch + [-.5, .5]
    span = (np.array([lit_cols[0], lit_cols[-1] + 1]) - cols / 2) * pitch + [-.6, .6]
    # One large detector image for the state shown by the column and sliders.
    ax = fig.add_axes([.64, .5, .345, .3])
    # Logarithmic contrast down to 1e-7 of the peak shows each beam's extent.
    ax.imshow(np.maximum(intensity / intensity.max(), 1e-7), origin="lower", extent=extent,
              cmap=CMAP, norm=LogNorm(1e-7, 1), interpolation="bilinear", aspect="equal")
    ax.set(xlim=span, ylim=band)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(.5); spine.set_color("#A5B2BB")
    ax.set_title("Reflected diffraction pattern on detector", fontsize=9, color=INK, pad=3)
    bar = span[1] - .3 - np.array([1., 0.])
    ax.plot(bar, [band[0] + .45] * 2, color="white", lw=1.2, solid_capstyle="butt")
    ax.text(bar.mean(), band[0] + .6, "1 mm", ha="center", va="bottom", fontsize=8, color="white")
    arrow(fig, (.53, .652), (.632, .652), color=TEAL, scale=9)
    return gui


def rings(ax, mean, covariance, pitch, color, *, dashed=False):
    """2-sigma detector contours of each predicted order, in mm."""
    for centre, matrix in zip(mean, covariance):
        eigenvalues, vectors = np.linalg.eigh(matrix[np.ix_([1, 0], [1, 0])])
        angle = np.rad2deg(np.arctan2(vectors[1, 1], vectors[0, 1]))
        # A minimum drawn size keeps a sharp predicted spot visible as a ring.
        width, height = (max(4 * np.sqrt(e) * pitch, .55) for e in eigenvalues[::-1])
        ax.add_patch(Ellipse(centre[::-1] * pitch, width, height, angle=angle, fill=False,
                             edgecolor=color, lw=1.3, ls=(0, (2, 1.2)) if dashed else "-",
                             zorder=3))


def build():
    spec = json.loads((DATA / "experiment.json").read_text())
    fit = json.loads((DATA / "fit.json").read_text())
    truth = np.asarray(json.loads((DATA / "scoring_truth.json").read_text())["parameters"])
    fitted = np.asarray(fit["best"]["parameters"])
    # A real multistart guess; refined on its own, it reaches the same answer.
    start = json.loads(START.read_text())
    initial = np.asarray(start["initial"])
    assert start["reaches_saved_best_fit"] and np.allclose(start["parameters"], fitted, rtol=1e-4)
    deviation = (fitted - truth) / np.asarray(fit["standard_errors"])
    with np.load(DATA / "images.npz") as data:
        images = data["images"]
    with np.load(DATA / "settings.npz") as data:
        controls, detector_dz = data["controls"][ROW:ROW + 1], data["detector_dz_m"][ROW:ROW + 1]
        n_fit = int((data["purpose"] == "fit").sum())  # the rest are held out

    use_arial()
    plt.rcParams.update({"font.family": "Arial", "font.size": MIN_PT,
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": .5, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), facecolor="white")
    fig.text(.012, .99, TITLE, fontsize=11.5, weight="bold", color=INK, va="top", linespacing=1.15)

    heading(fig, .87, "a", "Explore", "possible microscope arrangements with real-time diffraction simulation")
    gui = explore(fig)

    heading(fig, .425, "b", "Recover", "hidden lens, source and detector settings from diffraction patterns")
    alpha = np.deg2rad(spec["incidence_deg"])
    elements = [0, 1, 2, 3, 8]   # source, C1, C2, objective, detector distance
    true_x = truth[elements] * np.cos(alpha)
    _, predict = calibration.make_predictor(spec)
    pitch = spec["pixel_pitch_m"] * 1000
    image = images[ROW]
    nr, nc = image.shape
    flux = calibration.order_flux(spec) * spec["incident_electrons"]
    norm = LogNorm(1, image.max())

    predictions = {}
    for name, parameters in (("initial", initial), ("fitted", fitted)):
        mean, covariance = map(np.asarray, predict(parameters, controls, detector_dz))
        predictions[name] = calibration.integrated_image(mean[0], covariance[0], flux, spec)
    # One shared crop: the smallest box holding every beam, measured or predicted.
    lit = np.any([p > .02 * p.max() for p in (image, *predictions.values())], axis=0)
    lit_rows, lit_cols = np.flatnonzero(lit.any(axis=1)), np.flatnonzero(lit.any(axis=0))
    crop_x = (np.array([lit_cols[0], lit_cols[-1] + 1]) - nc / 2) * pitch + [-.5, .5]
    crop_y = (nr / 2 - np.array([lit_rows[-1] + 1, lit_rows[0]])) * pitch + [-.5, .5]
    tile_height = .24 * WIDTH_MM / HEIGHT_MM * np.ptp(crop_y) / np.ptp(crop_x)

    # Outline of each measured beam at 6% of its own peak, from the recorded
    # image of the same frame (lightly smoothed against shot noise).
    smooth = gaussian_filter(image.astype(float), 1.5)
    local_peak = maximum_filter(smooth, size=31)
    relative = np.where(local_peak > .05 * smooth.max(), smooth / local_peak, 0.)

    def tile(centre, pixels, title, color):
        """One detector image; all three share crop, size and contrast."""
        ax = fig.add_axes([centre - .12, .012, .24, tile_height])
        extent = np.array([-nc, nc, -nr, nr]) * pitch / 2
        ax.imshow(np.maximum(pixels, 1), origin="upper", cmap=CMAP, norm=norm,
                  extent=extent, interpolation="bilinear")
        ax.contour(relative, levels=[.06], origin="upper", extent=extent, colors="white",
                   linewidths=1., linestyles=[(0, (2.5, 1.5))])
        ax.set(xlim=crop_x, ylim=crop_y, aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(color); spine.set_linewidth(1.)
        ax.set_title(title, fontsize=8.5, pad=3, color=color, weight="bold" if color != INK else "normal")
        return ax

    for centre, parameters, color, coil, title, caption in [
            (.21, initial, OCHRE, "#D9A95B", "Initial microscope model (guess)", "Predicted diffraction pattern"),
            (.79, fitted, TEAL, TEAL, "Fitted microscope model", "Predicted diffraction pattern")]:
        guess = color == OCHRE
        fig.text(centre, .385, title, ha="center", fontsize=9.5, weight="bold", color=color)
        # The guess sits a little lower, clear of its title and dotted outlines.
        ax = mini_microscope(fig, [centre - .19, .232 - (.014 if guess else 0.), .38, .135],
                             parameters, spec,
                             gui["fixed_deflector_positions_m"], muted=guess, symbol_scale=.65)
        ax.set_ylim(-19, 9)   # headroom so no lens symbol is clipped
        for patch in ax.patches:
            patch.set_clip_on(False)
        calm(ax)
        for patch in ax.patches:
            if patch.get_gid() == "reflection_lens_coil":
                patch.set_facecolor(coil); patch.set_edgecolor(coil)
        if guess:
            # Guess drawn tentatively: dotted gold lens outlines, hollow gold
            # source, dashed gold detector. Its beam stays, since the guess
            # predicts the pattern below.
            dotted = (0, (1.2, 1.2))
            for patch in ax.patches:
                if patch.get_gid() == "reflection_lens_yoke":
                    patch.set_facecolor("none"); patch.set_edgecolor(OCHRE)
                    patch.set_linewidth(1.3); patch.set_linestyle(dotted); patch.set_zorder(1005)
                elif patch.get_gid() == "reflection_lens_coil":
                    patch.set_visible(False)
            for line in ax.lines:
                if line.get_marker() == "o":
                    line.set_markerfacecolor("white"); line.set_markeredgecolor(OCHRE)
                    line.set_markersize(4.5); line.set_markeredgewidth(1.2)
                elif line.get_linewidth() >= 2 and line.get_gid() != "reflection_deflector":
                    line.set_color(OCHRE); line.set_linewidth(1.6); line.set_linestyle(dotted)
            # The true microscope underneath in solid neutral grey, as in (a).
            truth_ax = mini_microscope(fig, ax.get_position().bounds, truth, spec,
                                       gui["fixed_deflector_positions_m"], symbol_scale=.65)
            truth_ax.set_ylim(ax.get_ylim()); truth_ax.patch.set_alpha(0)
            for patch in truth_ax.patches:
                patch.set_clip_on(False)
            truth_ax.set_zorder(ax.get_zorder() - 1); ax.patch.set_alpha(0)
            calm(truth_ax)
            for artist in truth_ax.collections:
                artist.set_visible(False)
            for line in truth_ax.lines:
                if line.get_marker() == "o" or (line.get_linewidth() >= 2
                                                and line.get_gid() != "reflection_deflector"):
                    continue   # true source and detector stay solid
                line.set_visible(False)
            for patch in truth_ax.patches:
                if patch.get_gid() not in ("reflection_lens_yoke", "reflection_lens_coil"):
                    patch.set_visible(False)
            # Symbol key in the empty top-left of the guess panel, drawn with the
            # same square symbols as the lenses.
            side = .011
            for row, (label, style) in enumerate([
                    ("guess", dict(facecolor="none", edgecolor=OCHRE, lw=1.3, ls=dotted)),
                    ("true", dict(facecolor="#C9D1D6", edgecolor=INK, lw=.9))]):
                y = .352 - row * .03
                fig.add_artist(Rectangle((.035, y), side, side * WIDTH_MM / HEIGHT_MM,
                                         transform=fig.transFigure, **style))
                fig.text(.035 + side + .008, y + .007, label, va="center", fontsize=8, color=INK)
        pattern = tile(centre, predictions["initial" if guess else "fitted"], caption, color)
        if guess:
            pattern.text(.03, .06, "- - measured", transform=pattern.transAxes, va="bottom",
                         fontsize=8, color="white")
    stack = image_stack(fig, images, [1, 7, 16, 28, 52])
    stack.set_position([.405, .0, .19, .14])
    for layer in stack.images:
        layer.set_cmap(CMAP)
    fig.text(.5, .178, "Measured diffraction patterns", ha="center", fontsize=8.5, color=INK)
    # The recorded frames vary the relative lens powers (and detector distance).
    fig.text(.5, .151, "from varying lens settings", ha="center", fontsize=8.5, color=MUTED)

    arrow(fig, (.42, .31), (.58, .31), color=TEAL, scale=11)
    fig.text(.5, .325, "fit", ha="center", va="bottom", fontsize=9.5, weight="bold", color=TEAL)

    for axis in fig.axes:
        for label in (*axis.get_xticklabels(), *axis.get_yticklabels()):
            label.set_fontfamily("Arial")
    sizes = [t.get_fontsize() for t in fig.findobj(plt.Text) if t.get_text().strip()]
    assert min(sizes) >= MIN_PT, min(sizes)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "reflection_calibration_erc_b1"
    files = export_ray_figure(fig, stem, dpi=600, bbox_inches=None, metadata=dict(
        title=TITLE, source=str(DATA), width_mm=WIDTH_MM, height_mm=HEIGHT_MM, min_font_pt=MIN_PT,
        exploration=gui,
        comparison_record_zero_based=ROW, fitting_images=n_fit,
        initial_parameters=initial.tolist(),
        initial_source=f"{START.name}: fitted from this guess, reaches the saved best fit",
        fitted=fitted.tolist(), truth=truth.tolist(),
        deviation_in_standard_errors=deviation.tolist(),
        envelope_display="Fit views: transverse envelope enlarged 40x upstream, 1x at specimen",
        rings="2-sigma predicted contours, drawn at least 0.55 mm across"))
    fig.savefig(stem.with_name(stem.name + "_preview.png"), dpi=200, facecolor="white")
    plt.close(fig)
    return files


if __name__ == "__main__":
    for kind, path in build().items():
        print(f"{kind}: {path}")
