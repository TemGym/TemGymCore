"""Overview centred on the preliminary fitted SEM; prospective ptychography interface."""
from make_calibration_workflow_figure import (
    DATA, OUT, INK, MUTED, TEAL, OCHRE, BLUE, microscope, image_stack, arrow,
)
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from temgym_core import reflection_calibration as calibration
from temgym_core.source import gaussian_waist
from temgym_core.run import run_to_end
from temgym_core.plotting import export_ray_figure

PURPLE = "#776294"


def heading(fig, letter, title, x, y):
    fig.text(x, y, letter, fontsize=9, weight="bold", color=INK)
    fig.text(x + .026, y, title, fontsize=8.5, color=INK)


def small_axes(fig, box):
    ax = fig.add_axes(box)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    return ax


def build():
    spec = json.loads((DATA / "experiment.json").read_text())
    fit = json.loads((DATA / "fit.json").read_text())
    checks = json.loads((DATA / "checks.json").read_text())
    p = np.asarray(fit["best"]["parameters"])
    with np.load(DATA / "images.npz") as data:
        images = data["images"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": .5, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(180 / 25.4, 145 / 25.4), facecolor="white")

    heading(fig, "a", "Explore configurations", .035, .954)
    fig.text(.061, .923, "Vary lens settings and inspect the beam", fontsize=6.8, color=MUTED)
    controls = small_axes(fig, [.062, .796, .245, .104])
    for index, (name, value) in enumerate(zip(["C1", "C2", "Objective"], [.30, .66, .45])):
        y = .83 - index * .33
        controls.text(0, y, name, fontsize=6.4, va="center", color=INK)
        controls.plot([.36, .95], [y, y], color="#D4DDE0", lw=2, solid_capstyle="round")
        controls.plot([.36, .36 + .59 * value], [y, y], color=OCHRE, lw=2, solid_capstyle="round")
        controls.plot(.36 + .59 * value, y, "o", ms=4, color=OCHRE, markeredgecolor="white", markeredgewidth=.6)
    fig.text(.348, .862, "Interactive\nforward model", fontsize=7, color=OCHRE, ha="center", va="center")
    arrow(fig, (.43, .85), (.48, .744), color=OCHRE, connectionstyle="arc3,rad=-.18")

    heading(fig, "b", "Calibrate an unknown microscope", .560, .954)
    fig.text(.586, .923, "Initial guess + recorded diffraction patterns", fontsize=6.8, color=MUTED)
    stack = image_stack(fig, images, [1, 7, 16, 28, 52])
    stack.set_position([.74, .752, .23, .157])
    fig.text(.655, .849, "Fit positions\nand focal lengths", fontsize=7, ha="center", va="center", color=BLUE)
    arrow(fig, (.745, .832), (.72, .832), color=BLUE)
    arrow(fig, (.645, .806), (.59, .744), color=BLUE, connectionstyle="arc3,rad=.12")

    # The central drawing is the actual fitted result, not a new illustrative column.
    fig.text(.50, .710, "Preliminary fitted microscope", fontsize=10, ha="center", color=INK)
    fig.text(.50, .680, "A shared physical model for exploration, calibration and state queries",
             fontsize=7, color=MUTED, ha="center")
    ax = fig.add_axes([.032, .435, .734, .227])
    microscope(ax, p, spec, fitted=True)

    # Compact, data-backed evidence beside the central model.
    fig.text(.785, .619, "SYNTHETIC CALIBRATION", fontsize=5.8, color=TEAL, weight="bold")
    fig.text(.785, .581, f"{checks['objective_working_distance_mm']:.2f} ± {fit['standard_errors'][3]:.2f} mm",
             fontsize=8, color=INK)
    fig.text(.785, .559, "Objective working distance", fontsize=6.2, color=MUTED)
    fig.text(.785, .518, f"{checks['prediction_rms_noise_units']:.2f} noise units", fontsize=8, color=INK)
    fig.text(.785, .496, "Held-out prediction RMS", fontsize=6.2, color=MUTED)
    fig.text(.785, .457, "Nine fitted parameters", fontsize=6.5, color=TEAL)

    # A representation inset: the component profiles are schematic, not
    # independent ray intensities. Their coherent sum defines the field.
    packets = small_axes(fig, [.075, .613, .225, .043])
    x = np.linspace(0, 1, 250)
    component_fields = np.exp(-((x[None, :] - np.array([.32, .50, .68])[:, None]) / .13) ** 2)
    for field in component_fields:
        packets.plot(x, .1 + .55 * field, color=TEAL, lw=.65, alpha=.65)
    packets.plot(x, .1 + .55 * component_fields.sum(axis=0), color=INK, lw=.9)
    fig.text(.315, .638, "Gaussian field representation", fontsize=6.8, color=INK, va="center")
    fig.text(.315, .618, "Coherent propagation through the column", fontsize=6.1, color=MUTED)

    # Three state queries are connected to their physical planes. Dashed
    # connectors distinguish the proposed reconstruction interface from the fit.
    surface_x = .032 + .734 * (430 / 615)
    detector_x = .032 + .734 * ((p[8] * np.cos(np.deg2rad(2)) + 430) / 615)
    for start, end, color in [((surface_x, .450), (.18, .310), TEAL),
                               ((surface_x + .015, .450), (.50, .310), OCHRE),
                               ((detector_x, .450), (.815, .310), BLUE)]:
        # Route through the clear gap to the right of the Gaussian inset.
        item = arrow(fig, start, end, color=color, connectionstyle="arc3,rad=.07", scale=6)
        item.set_linestyle((0, (3, 3)))
    heading(fig, "c", "Query the microscope state", .035, .405)
    # White heading background maintains separation from the dashed connectors.
    fig.text(.964, .096, "Proposed ptychography interface", fontsize=6.2,
             color=PURPLE, ha="right", bbox=dict(facecolor="white", edgecolor="none", pad=1))

    fig.text(.18, .294, "Probe state", ha="center", fontsize=8, color=TEAL)
    fig.text(.50, .294, "Scan coordinates", ha="center", fontsize=8, color=OCHRE)
    fig.text(.815, .294, "Detector coordinates", ha="center", fontsize=8, color=BLUE)

    # Calculate the input wave at the specimen from the fitted native Gaussian.
    source = gaussian_waist(z=p[0] * 1e-3, voltage=spec['voltage_v'], rms_size=p[7] * 1e-9)
    beam = run_to_end(source, calibration.column(p, np.ones(3)), propagator=calibration.FS)
    k = float(beam.k)
    q = np.asarray(beam.Q_inv)
    sigma = np.sqrt(np.linalg.inv(2 * k * q.imag)[0, 0])
    xx, yy = np.meshgrid(np.linspace(-3 * sigma, 3 * sigma, 100), np.linspace(-3 * sigma, 3 * sigma, 100))
    points = np.stack((xx, yy), axis=-1)
    argument = k / 2 * np.einsum('...i,ij,...j->...', points, q, points)
    intensity, phase = np.exp(-2 * argument.imag), np.angle(np.exp(1j * argument.real))
    for left, data, cmap, title, limits in [(.077, intensity, 'magma', 'Intensity', (0, 1)),
                                           (.188, phase, 'twilight', 'Phase', (-np.pi, np.pi))]:
        image_ax = fig.add_axes([left, .183, .095, .091])
        image_ax.imshow(data, cmap=cmap, vmin=limits[0], vmax=limits[1], origin='lower')
        image_ax.axis('off')
        image_ax.text(.5, -.12, title, ha='center', transform=image_ax.transAxes, fontsize=6.2, color=MUTED)
    fig.text(.18, .139, "Incident complex beam state", fontsize=6.2, color=MUTED, ha="center")

    scan = small_axes(fig, [.425, .179, .15, .096])
    scan.set_aspect('equal')
    # Schematic overlapping scan positions, not an acquired ptychography scan.
    for y in np.linspace(.2, .8, 4):
        for x0 in np.linspace(.2, .8, 4):
            scan.add_patch(Circle((x0, y), .135, fc=OCHRE, ec='none', alpha=.075))
            scan.plot(x0, y, '.', ms=2.4, color=OCHRE)
    scan.arrow(.07, .07, .85, 0, head_width=.035, lw=.5, color=MUTED, length_includes_head=True)
    scan.arrow(.07, .07, 0, .85, head_width=.035, lw=.5, color=MUTED, length_includes_head=True)
    fig.text(.50, .139, "Scan commands → specimen positions", fontsize=6.2, color=MUTED, ha="center")

    detector = small_axes(fig, [.746, .178, .145, .096])
    transform = np.array([[.75, -.19], [.20, .65]])
    origin = np.array([.24, .08])
    for value in np.linspace(0, 1, 7):
        for endpoints in (np.array([[value, 0], [value, 1]]), np.array([[0, value], [1, value]])):
            line = endpoints @ transform.T + origin
            detector.plot(line[:, 0], line[:, 1], color=BLUE, lw=.5, alpha=.8)
    detector.text(.91, .22, 'u', fontsize=6.3, color=BLUE)
    detector.text(.04, .72, 'v', fontsize=6.3, color=BLUE)
    fig.text(.815, .139, "Pixel indices → physical coordinates", fontsize=6.2, color=MUTED, ha="center")

    # Joint state enters the reconstruction alongside a distinct ptychographic
    # dataset. No reconstructed object is drawn because none has been run here.
    for x0 in (.18, .50, .815):
        fig.add_artist(plt.Line2D([x0, x0], [.127, .111], transform=fig.transFigure,
                                 color=PURPLE, lw=.6, linestyle=(0, (3, 3))))
    fig.add_artist(plt.Line2D([.18, .815], [.111, .111], transform=fig.transFigure, color=PURPLE, lw=.6))
    arrow(fig, (.68, .111), (.68, .085), color=PURPLE, scale=6)
    box = FancyBboxPatch((.544, .037), .271, .049, boxstyle="round,pad=.006,rounding_size=.008",
                         transform=fig.transFigure, fc="#F3F0F7", ec=PURPLE, lw=.6)
    fig.add_artist(box)
    fig.text(.679, .061, "Ptychographic reconstruction", ha="center", va="center", fontsize=7.3, color=INK)
    fig.text(.25, .061, "Scan-resolved diffraction data", ha="center", va="center", fontsize=7, color=BLUE)
    arrow(fig, (.426, .061), (.531, .061), color=BLUE, scale=7)
    arrow(fig, (.827, .061), (.859, .061), color=PURPLE, scale=7)
    fig.text(.877, .061, "Specimen\namplitude + phase", va="center", fontsize=6.5, color=PURPLE)
    fig.text(.035, .012, "Local Gaussian reflection pilot · enlarged beam envelopes · scan and detector-query icons are schematic",
             fontsize=5.6, color=MUTED)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "microscope_model_to_ptychography"
    files = export_ray_figure(fig, stem, dpi=600, metadata=dict(
        title="A fitted microscope model for exploration and ptychographic reconstruction",
        source=str(DATA), fit_parameters=p.tolist(),
        demonstrated="Synthetic local-Gaussian reflection calibration and fitted specimen-plane Gaussian",
        proposed="Scan/detector coordinate queries and the ptychography connection; no reconstruction shown",
        schematic="Gaussian representation inset, interactive sliders, scan and detector icons",
        probe_extent_nm=float(6 * sigma * 1e9), probe_plane="Column-transverse specimen plane; not surface footprint",
        probe_phase="Wrapped quadratic phase, arbitrary global phase omitted",
        image_records_zero_based=[1, 7, 16, 28, 52], width_mm=180, height_mm=145))
    fig.savefig(stem.with_name(stem.name + "_preview.png"), dpi=180, facecolor="white")
    plt.close(fig)
    return files


if __name__ == '__main__':
    for kind, path in build().items():
        print(f'{kind}: {path}')
