"""Publication workflow figure from the saved reflection calibration.

Run from any directory with the TemGymCore environment. The SVG retains editable
text and vector microscope/arrow artwork; only measured diffraction images are
embedded rasters. No fitting or acquisition is rerun.
"""
from pathlib import Path
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import json
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon, FancyArrowPatch
from matplotlib.transforms import Affine2D
from temgym_core import reflection_calibration as calibration
from temgym_core.plotting import export_ray_figure

HERE = Path(__file__).resolve().parent
DATA = HERE / "output/calibration_gaussian"
OUT = HERE / "output/publication"
INK = "#243440"
MUTED = "#62727B"
TEAL = "#168879"
OCHRE = "#B58B48"
BLUE = "#427A9D"


def microscope(ax, parameters, spec, *, fitted=False):
    calibration._draw_column(ax, parameters, np.ones(3), 0., spec, "")
    # Retain native traced envelopes, reflected paths, crystal and coil glyphs;
    # replace notebook annotations with typography sized for a 180 mm figure.
    for text in list(ax.texts):
        text.remove()
    ax.set_title("")
    for patch in ax.patches:
        if patch.get_gid() in ("reflection_lens_yoke", "reflection_lens_coil"):
            patch.set_transform(Affine2D().scale(.46) + patch.get_data_transform())
            patch.set_linewidth(.55)
            if patch.get_gid() == "reflection_lens_coil":
                patch.set_facecolor(TEAL if fitted else OCHRE)
                patch.set_edgecolor(TEAL if fitted else OCHRE)
                patch.set_hatch(None)
    ax.set(xlim=(-430, 185), ylim=(-23, 14))
    ax.set_axis_off()
    angle = np.deg2rad(spec["incidence_deg"])
    for z, label in zip(parameters[1:4], ["C1", "C2", "Objective"]):
        x, y = z * np.cos(angle), z * np.sin(angle)
        ax.annotate(label, (x, y - 2.3), xytext=(x, y - 8.), ha="center", va="top",
                    fontsize=6.5, color=INK,
                    arrowprops=dict(arrowstyle="-", lw=.45, color="#A6B1B7", shrinkA=2, shrinkB=2))
    ax.annotate("Source", (parameters[0] * np.cos(angle), parameters[0] * np.sin(angle)),
                xytext=(-1, -11), textcoords="offset points", fontsize=6.5, ha="center", color=INK)
    ax.text(0, 4., "Cu(001)", ha="center", fontsize=6.5, color=INK)
    ax.text(parameters[8] * np.cos(angle), 8., "Detector", ha="center", fontsize=6.5, color=INK)


def label(fig, letter, title, x, y):
    fig.text(x, y, letter, fontsize=10, fontweight="bold", color=INK, va="baseline")
    fig.text(x + .028, y, title, fontsize=9, fontweight="normal", color=INK, va="baseline")


def arrow(fig, start, end, *, color=INK, connectionstyle="arc3,rad=0", scale=8):
    item = FancyArrowPatch(start, end, transform=fig.transFigure, arrowstyle="-|>",
                           mutation_scale=scale, lw=.85, color=color,
                           connectionstyle=connectionstyle, clip_on=False)
    fig.add_artist(item)
    return item


def image_stack(fig, images, records):
    ax = fig.add_axes([.664, .591, .306, .288])
    ax.set(xlim=(-.15, 1.65), ylim=(-.12, 1.57), aspect="equal")
    ax.axis("off")
    # An affine oblique view keeps every pixel traceable to its recorded frame.
    # A shared crop and contrast are applied to all layers.
    extent = (0, 1, 0, 1)
    shared_max = max(images[r].max() for r in records)
    for layer, row in enumerate(records):
        offset = len(records) - 1 - layer
        transform = Affine2D.from_values(.96, .24, -.23, .74,
                                         .23 + offset * .102, .025 + offset * .136)
        corners = transform.transform([[0, 0], [1, 0], [1, 1], [0, 1]])
        ax.add_patch(Polygon(corners, facecolor="white", edgecolor="#B3BFC7", lw=.6,
                             zorder=layer * 3))
        ax.imshow(np.maximum(images[row, 63:178, 29:227], 1), origin="upper", extent=extent,
                  cmap="gray", norm=LogNorm(1, shared_max), interpolation="bilinear",
                  transform=transform + ax.transData, zorder=layer * 3 + 1)
        ax.add_patch(Polygon(corners, facecolor="none", edgecolor="#91A2AC", lw=.55,
                             zorder=layer * 3 + 2))
    return ax


def build():
    spec = json.loads((DATA / "experiment.json").read_text())
    fit = json.loads((DATA / "fit.json").read_text())
    with np.load(DATA / "images.npz") as file:
        images = file["images"]
    with np.load(DATA / "settings.npz") as file:
        settings = dict(file)
    final = np.asarray(fit["best"]["parameters"])
    records = [1, 7, 16, 28, 52]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": .5, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(180 / 25.4, 113 / 25.4), facecolor="white")
    label(fig, "a", "Initial microscope guess", .035, .943)
    label(fig, "b", "Record diffraction patterns", .635, .943)
    fig.text(.063, .901, "Unknown source, lens positions and focal lengths", fontsize=6.8, color=MUTED)
    fig.text(.663, .900, "Vary lens powers and translate the detector", fontsize=6.8, color=MUTED)
    initial_ax = fig.add_axes([.026, .608, .563, .263])
    microscope(initial_ax, calibration.INITIAL, spec)
    image_stack(fig, images, records)
    fig.text(.805, .570, "Known relative settings · unknown absolute geometry",
             ha="center", fontsize=6.2, color=MUTED)

    # Independent model and data feed the same inverse problem; there is no
    # arrow suggesting that the initial guess generated the recorded patterns.
    arrow(fig, (.367, .594), (.710, .460), color=OCHRE,
          connectionstyle="angle3,angleA=-90,angleB=180")
    fig.text(.425, .530, "Predict patterns", fontsize=7, color=OCHRE)
    arrow(fig, (.809, .542), (.809, .470), color=BLUE)
    fig.text(.831, .503, "Measured data", fontsize=6.7, color=BLUE, va="center")

    label(fig, "c", "Fit the microscope", .635, .417)
    fig.text(.663, .375, "Compare measurements with model predictions", fontsize=6.7, color=MUTED)
    # Actual saved measurement and fitted-model prediction at the same setting.
    row = 52
    _, predict = calibration.make_predictor(spec)
    mean, covariance = map(np.asarray, predict(final, settings["controls"][row:row + 1],
                                               settings["detector_dz_m"][row:row + 1]))
    prediction = calibration.integrated_image(mean[0], covariance[0],
        calibration.order_flux(spec) * spec["incident_electrons"], spec)
    for left, image, title in ((.675, images[row], "Recorded"), (.824, prediction, "Predicted")):
        ax = fig.add_axes([left, .219, .124, .115])
        ax.imshow(np.maximum(image[63:178, 29:227], 1), cmap="gray", origin="upper",
                  norm=LogNorm(1, images[row].max()), aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#A5B2BB"); spine.set_linewidth(.45)
        ax.set_title(title, fontsize=6.6, pad=4, color=INK)
    fig.text(.811, .263, "↔", fontsize=11, ha="center", color=MUTED)
    fig.text(.811, .157, "Update positions and lens strengths", fontsize=7, ha="center", color=INK)
    fig.text(.811, .119, "Joint fit across the recorded image series", fontsize=6.4, ha="center", color=MUTED)

    label(fig, "d", "Recovered microscope", .035, .417)
    fig.text(.063, .375, "Fitted geometry with uncertainty estimates", fontsize=6.8, color=MUTED)
    final_ax = fig.add_axes([.026, .083, .563, .263])
    microscope(final_ax, final, spec, fitted=True)
    arrow(fig, (.641, .250), (.585, .250), color=TEAL, scale=9)
    fig.text(.615, .278, "Fit", fontsize=6.8, color=TEAL, ha="center")

    # A discreet footer distinguishes the illustrated synthetic experiment from
    # a hardware acquisition without cluttering the scientific workflow.
    fig.text(.035, .023, "Reflection calibration", fontsize=6.4, color=INK)
    fig.text(.965, .023, "Illustrated with simulated Cu(001) diffraction data; beam envelopes enlarged",
             fontsize=5.9, color=MUTED, ha="right")
    OUT.mkdir(parents=True, exist_ok=True)
    files = export_ray_figure(fig, OUT / "reflection_calibration_workflow", dpi=600,
        metadata=dict(title="Reflection microscope calibration workflow",
                      width_mm=180, height_mm=113, image_records_zero_based=records,
                      comparison_record_zero_based=row, source=str(DATA),
                      image_treatment="Common crop and logarithmic contrast; affine stack projection",
                      model="Local Gaussian reflection pilot; native traced column geometry",
                      svg_text="Editable text; diffraction panels embedded as raster images"))
    fig.savefig(OUT / "reflection_calibration_workflow_preview.png", dpi=180, facecolor="white")
    plt.close(fig)
    return files


if __name__ == "__main__":
    for kind, path in build().items():
        print(f"{kind}: {path}")
