"""Explore saved images or geometric ray previews beside native ray diagrams."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .acquisition import _column_source, render_ray_image


def _ray_diagram(spec, settings, model, row, aperture):
    from temgym_core.plotting import trace_ray_envelope
    from temgym_core.ray import Ray

    column, mean, covariance, source_z, voltage = _column_source(
        spec, settings, model, row, aperture
    )
    radius, angle = np.meshgrid(
        np.linspace(0, 2, 33), np.linspace(0, 2 * np.pi, 128, endpoint=False)
    )
    rays = Ray(
        x=mean[0] + np.sqrt(covariance[0, 0]) * (radius * np.cos(angle)).ravel(),
        y=mean[1],
        dx=mean[2] + np.sqrt(covariance[2, 2]) * (radius * np.sin(angle)).ravel(),
        dy=mean[3],
        z=source_z,
        pathlength=0.0,
        voltage=voltage,
    )
    return column, trace_ray_envelope(column, rays)


def _draw_ray_section(
    ax, column, segments, section, *, section_bounds=None,
    component_labels=None, source_label="Source waist", mark_source_image=True,
    scale_segments=None,
):
    """Draw a calibration-style linear zoom of a traced geometric bundle.

    Optional bounds and labels support other native columns; defaults retain
    the calibration browser's specimen-centred geometry and section choices.
    Set ``mark_source_image=False`` for a source plane that is not a waist.
    Supply fixed ``scale_segments`` to keep the transverse scale and symbols
    stable while the displayed trajectories change.
    """
    from matplotlib.patches import Arc

    from temgym_core.components import (
        Deflector,
        DoubleDeflector,
        ElectromagneticLens,
        HardEdgeSolenoid,
        Lens,
        SigmoidAperture,
    )
    from temgym_core.plotting import (
        _spread_label_positions,
        legacy_beam_plot_params,
        plot_ray_bundle,
    )

    column = tuple(
        piece
        for c in column
        for piece in (c.deflectors if isinstance(c, DoubleDeflector) else (c,))
    )
    source_z, detector_z = segments[0][0][0], float(column[-1].z)
    if section_bounds is None:
        cl3_z = float(column[1].z)
        mini_exit = float(column[5].z + column[5].length)
        projector_exit = float(column[-2].z + column[-2].length)
        section_bounds = [
            (source_z, mini_exit),
            (cl3_z - 0.005, 0.005),
            (0.005, projector_exit),
            (projector_exit, detector_z),
        ]
    z0, z1 = section_bounds[section]
    factor, unit = (1e3, "mm") if section == 3 else (1e6, "µm")

    def clipped_segments(source_segments):
        visible = []
        for z, bounds in source_segments:
            lo, hi = max(z0, z[0]), min(z1, z[-1])
            if hi > lo:
                zz = np.unique(np.r_[lo, z[(z > lo) & (z < hi)], hi])
                visible.append(
                    (zz, np.column_stack([np.interp(zz, z, b) for b in bounds.T]))
                )
        return visible

    visible = clipped_segments(segments)
    scale = visible if scale_segments is None else clipped_segments(scale_segments)
    width = 1.15 * max((np.abs(x).max() for _, x in scale), default=1e-9)
    style = legacy_beam_plot_params(fill_color="#98E0C1", fill_alpha=1.0, edge_lw=0.65)
    for z, bounds in visible:
        plot_ray_bundle(ax, bounds * factor, z * 1000, style)
    ax.set(
        xlim=(-width * factor, 2.8 * width * factor),
        ylim=((z1 + 0.025 * (z1 - z0)) * 1000, (z0 - 0.025 * (z1 - z0)) * 1000),
        xlabel=f"x ({unit})",
        ylabel="z from specimen (mm)",
    )
    ax.set_xticks(np.array([-1, 0, 1]) * width * factor / 1.15)
    ax.xaxis.set_major_formatter("{x:.2g}")
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(0, color="0.8", lw=0.5, zorder=-1)
    labels = component_labels if component_labels is not None else [
        "CL1",
        "CL3",
        "Aperture",
        "Deflector 1*",
        "Deflector 2*",
        "Mini lens",
        "Objective pre",
        "Specimen",
        "Objective post",
        "IL1",
        "IL2",
        "IL3",
        "PL1",
        "Detector",
    ]
    annotations, rotation = [], 0.0
    if section == 0:
        ax.plot(
            segments[0][1][0].mean() * factor,
            source_z * 1000,
            "o",
            color="#795548",
            ms=4,
        )
        annotations.append((source_z, source_label))
    if section == 0 and mark_source_image:
        # The first-lens source image is a conjugate, not a beam-radius minimum.
        first = column[0]
        matrix = np.asarray(first.matrix)
        gap = float(first.z) - source_z
        bb = matrix[:2, :2] * gap + matrix[:2, 2:]
        dd = matrix[2:, :2] * gap + matrix[2:, 2:]
        distance = -np.sum(bb * dd) / max(np.sum(dd * dd), 1e-24)
        crossover = float(first.z + first.length) + distance
        if distance > 0 and crossover < float(column[1].z):
            ax.hlines(
                crossover * 1000,
                -width * factor,
                width * factor,
                color="#7453a4",
                ls=":",
                lw=0.8,
            )
            annotations.append((crossover, "CL1 source image"))
    for component, label in zip(column, labels):
        z = float(component.z + getattr(component, "length", 0) / 2)
        if (
            label
            and z0 <= z <= z1
            and not (section == 3 and isinstance(component, HardEdgeSolenoid))
        ):
            annotations.append((z, label))
            if isinstance(component, (HardEdgeSolenoid, ElectromagneticLens, Lens)):
                height = max(float(getattr(component, "length", 0.)),
                             0.012 * (z1 - z0)) * 1000
                for a, b, depth in [(180, 360, -0.5), (0, 180, 4)]:
                    ax.add_patch(
                        Arc(
                            (0, z * 1000),
                            2 * width * factor,
                            height,
                            theta1=a,
                            theta2=b,
                            color="#557080",
                            lw=1.1,
                            zorder=depth,
                        )
                    )
            elif isinstance(component, SigmoidAperture):
                centre = float(component.x0) * np.cos(rotation) + float(
                    component.y0
                ) * np.sin(rotation)
                for a, b in [
                    (-width, centre - component.radius),
                    (centre + component.radius, width),
                ]:
                    if b > a:
                        ax.plot(
                            np.array([a, b]) * factor,
                            [z * 1000] * 2,
                            color="#555555",
                            lw=3,
                        )
            elif isinstance(component, Deflector):
                for a, b, color in [
                    (-width, -0.25 * width, "#D55E00"),
                    (0.25 * width, width, "#0072B2"),
                ]:
                    ax.plot(
                        np.array([a, b]) * factor, [z * 1000] * 2, color=color, lw=2
                    )
            else:
                ax.plot(
                    np.array([-width, width]) * factor,
                    [z * 1000] * 2,
                    color="#557080",
                    lw=1,
                )
        rotation -= float(getattr(component, "q", 0))
    positions = _spread_label_positions([z for z, _ in annotations], z0, z1)
    for (z, label), label_z in zip(annotations, positions):
        ax.annotate(
            label,
            (width * factor, z * 1000),
            xytext=(1.25 * width * factor, label_z * 1000),
            va="center",
            fontsize=9,
            arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "0.7"},
        )


def browse_images(*, experiment_directory):
    """Return notebook controls for saved images and matching native ray diagrams.

    Select recorded settings, not new lens currents. Saved images are loaded
    once. Without images, render geometric previews from the generating model
    on demand; these previews are not the measurements used by the fitter.
    The initial guess can be compared with truth while the image stays fixed.
    Requires the optional ``inversion`` dependencies and a live notebook kernel.
    """
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    from temgym_core.components import DoubleDeflector

    from .condenser_model import condenser_starting_guesses
    from .inference import PRESETS
    from .projector_model import model_to_parameters

    with plt.ioff():
        fig, (image_ax, ray_ax) = plt.subplots(
            1, 2, figsize=(10, 5.4), layout="constrained"
        )
    if not isinstance(fig.canvas, widgets.Widget):
        plt.close(fig)
        raise RuntimeError(  # noqa: TRY004 — the active backend needs changing
            "Run %matplotlib widget before opening the browser. "
            "Install ipympl in the notebook kernel environment if needed."
        )

    root = Path(experiment_directory)
    spec = json.loads((root / "experiment.json").read_text())
    with np.load(root / "settings.npz") as data:
        settings = dict(data)
    images = None
    if (root / "images.npz").exists():
        with np.load(root / "images.npz") as data:
            images = data["images"]
    guesses, _, _ = condenser_starting_guesses(
        starting_guess_path=PRESETS / "microscope_condenser_starting_guess.json",
        number_of_broad_starts=0,
        number_of_apertures=len(spec["aperture_diameters_um"]),
    )
    models = {}
    if (root / "scoring_truth.json").exists():
        models["Ground truth"] = json.loads((root / "scoring_truth.json").read_text())
    models.update({
        "Initial guess": {
            "condenser_parameters": guesses[0],
            "projector_parameters": model_to_parameters(
                json.loads((PRESETS / "projector_starting_guess.json").read_text())
            ),
        },
    })
    fitted = root / "fit.json"
    if fitted.exists():
        result = json.loads(fitted.read_text())
        if result.get("best", {}).get("acceptable", False):
            models["Fitted model"] = result
    diagram = lru_cache(maxsize=12)(
        lambda row, aperture, model: _ray_diagram(
            spec, settings, models[model], row, aperture
        )
    )
    preview = lru_cache(maxsize=12)(
        lambda row, aperture: render_ray_image(
            spec, settings, models["Ground truth"], row, aperture
        )
    )
    count = len(settings["voltage_index"])
    nominal = np.asarray(
        settings.get(
            "nominal_magnification",
            np.full(count, np.nan),
        )
    )
    kicks = settings["specimen_kick"]
    beam_labels = np.array(
        [
            "No steering"
            if not np.any(kick)
            else ("Shift X", "Shift Y", "Tilt X", "Tilt Y")[
                int(np.argmax(np.abs(kick)))
            ]
            for kick in kicks
        ]
    )

    def dropdown(description, options):
        return widgets.Dropdown(
            description=description,
            options=options,
            style={"description_width": "90px"},
            layout=widgets.Layout(width="100%"),
        )

    purpose = dropdown(
        "Experiment",
        [
            ("All records", "all"),
            ("Projector calibration", "projector"),
            ("Condenser calibration", "condenser"),
            ("Prediction checks", "check"),
        ],
    )
    voltage = dropdown(
        "Voltage", [(f"{v / 1000:g} kV", i) for i, v in enumerate(spec["voltages_v"])]
    )
    magnification = dropdown(
        "Nominal mag",
        [("Both / all", -1)]
        + [(f"{m:,.0f}×", m) for m in np.unique(nominal[np.isfinite(nominal)])],
    )
    beam = dropdown(
        "Beam input",
        [("All inputs", "all")] + [(x, x) for x in dict.fromkeys(beam_labels)],
    )
    aperture = dropdown(
        "Aperture",
        [(f"{d:g} µm", i) for i, d in enumerate(spec["aperture_diameters_um"])],
    )
    row = widgets.SelectionSlider(
        description="Image",
        options=[("001", 0)],
        continuous_update=False,
        readout=False,
        layout=widgets.Layout(flex="1 1 auto", width="auto"),
    )
    previous = widgets.Button(
        icon="chevron-left",
        tooltip="Previous matching image",
        layout=widgets.Layout(width="40px"),
    )
    following = widgets.Button(
        icon="chevron-right",
        tooltip="Next matching image",
        layout=widgets.Layout(width="40px"),
    )
    progress = widgets.HTML(layout=widgets.Layout(width="150px"))
    section = dropdown(
        "Ray view",
        [
            ("Source → mini", 0),
            ("Aperture → specimen", 1),
            ("Projector lenses", 2),
            ("Detector", 3),
        ],
    )
    model = dropdown("Ray model", list(models))
    contrast = widgets.Checkbox(description="Log contrast", value=True, indent=False)
    summary, details = widgets.HTML(), widgets.HTML()
    errors = widgets.Output()
    shape = images.shape[-2:] if images is not None else spec["shape_yx"]
    height, width = np.array(shape) * spec["pixel_pitch_m"] * 0.5e3
    recorded = np.isfinite(images).all(axis=(-2, -1)) if images is not None else np.ones((count, 2), bool)
    first_row, first_aperture = np.argwhere(recorded)[0]
    initial_image = images[first_row, first_aperture] if images is not None else preview(0, 0)
    maximum = max(
        float(np.nanmax(images) if images is not None else initial_image.max()), 2.0
    )
    artist = image_ax.imshow(
        initial_image,
        origin="upper",
        extent=(-width, width, -height, height),
        cmap="gray",
        norm=LogNorm(1, maximum),
    )
    fig.colorbar(
        artist,
        ax=image_ax,
        shrink=0.65,
        label="Expected counts / pixel" if images is None else "Counts / pixel",
    )
    image_ax.set(
        title="Recorded beam"
        if images is not None
        else "Ray image · geometric preview",
        xlabel="Detector x (mm)",
        ylabel="Detector y (mm)",
    )
    fig.canvas.layout = widgets.Layout(width="100%", height="570px", min_height="570px")
    fig.canvas.toolbar_position = "bottom"
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    last_ray_view, filtering = None, False

    @errors.capture(clear_output=True, wait=True)
    def update(event=None):
        nonlocal last_ray_view
        if filtering:
            return
        ri, ai = row.value, aperture.value
        valid = ri >= 0
        artist.set_visible(valid)
        ray_ax.set_visible(valid)
        previous.disabled = not valid or row.index == 0
        following.disabled = not valid or row.index == len(row.options) - 1
        progress.value = (
            f"<b>{row.index + 1} / {len(row.options)}</b> matching"
            if valid
            else "No matches"
        )
        if not valid:
            summary.value = "<p>No records match these filters. Choose ‘All records’ or ‘All inputs’.</p>"
            details.value = ""
            fig.canvas.draw_idle()
            return
        shift, tilt = kicks[ri, :2] * 1e9, kicks[ri, 2:] * 1e3
        purpose_text = {
            "projector": "Projector calibration",
            "condenser": "Condenser calibration",
            "check": "Prediction check",
        }[str(settings["purpose"][ri])]
        mag_label = (
            f"{nominal[ri]:,.0f}× nominal"
            if np.isfinite(nominal[ri])
            else "Nominal magnification not recorded"
        )
        summary.value = (
            '<div style="padding:14px 18px;background:#f0f5f7;border-radius:10px;color:#243746;">'
            f'<div style="font-size:17px;font-weight:600;">{purpose_text}</div>'
            '<div style="margin-top:6px;line-height:1.8;">'
            f"Frame {ri + 1:03d} · {mag_label} · {spec['voltages_v'][settings['voltage_index'][ri]] / 1000:g} kV · "
            f"{spec['aperture_diameters_um'][ai]:g} µm aperture<br>"
            f"Specimen shift <b>({shift[0]:g}, {shift[1]:g}) nm</b> &nbsp; "
            f"Tilt <b>({tilt[0]:g}, {tilt[1]:g}) mrad</b></div></div>"
        )
        frame = images[ri, ai] if images is not None else preview(ri, ai)
        artist.set_data(np.maximum(frame, 1 if contrast.value else 0))
        upper = maximum if images is not None else max(float(frame.max()), 2.0)
        artist.set_norm(LogNorm(1, upper) if contrast.value else Normalize(0, upper))
        fig.canvas.draw_idle()
        column, segments = diagram(ri, ai, model.value)
        ray_view = (ri, ai, section.value, model.value)
        if ray_view != last_ray_view:
            ray_ax.clear()
            _draw_ray_section(ray_ax, column, segments, section.value)
            ray_ax.set_title(f"{model.value} · rays", loc="left")
            last_ray_view = ray_view
        names = ["CL1", "CL3", "Mini", "Objective", "IL1", "IL2", "IL3", "PL1"]
        currents = np.r_[
            settings["condenser_controls"][ri], settings["projector_controls"][ri]
        ]
        table = (
            '<table style="width:100%;text-align:left;"><tr>'
            + "".join(f"<th>{n}</th>" for n in names)
            + "</tr><tr>"
        )
        table += "".join(f"<td>{v:.5g}</td>" for v in currents) + "</tr></table>"
        balances = ""
        for c in column:
            if isinstance(c, DoubleDeflector):
                balances = f"<p>Model balance ratios (X, Y): shift ({c.shift_balance_x:.4g}, {c.shift_balance_y:.4g}); tilt ({c.tilt_balance_x:.4g}, {c.tilt_balance_y:.4g}).</p>"
        details.value = (
            "<p>Relative lens currents for this acquisition setting</p>"
            + table
            + balances
            + "<p>Deflector positions are assumed, not fitted. Ray widths are geometric. Nominal magnification labels identify the baseline, not the exact magnification after a wobble.</p>"
            + (
                "<p>The preview bins 262,144 native rays, weighted by aperture intensity transmission, then applies detector blur. No diffraction or measurement noise is included. Contrast adjusts per frame. The fit uses separate, directly calculated moments that include diffraction; it does not analyse these preview pixels.</p>"
                if images is None
                else ""
            )
        )
        fig.canvas.draw_idle()

    def filter_rows(event=None):
        nonlocal filtering
        if filtering:
            return
        matching = (
            (settings["voltage_index"] == voltage.value)
            & recorded[:, aperture.value]
            & ((purpose.value == "all") | (settings["purpose"] == purpose.value))
            & ((magnification.value == -1) | (nominal == magnification.value))
        )
        filtering = True
        previous_input = beam.value
        available_inputs = list(dict.fromkeys(beam_labels[matching]))
        beam.options = [("All inputs", "all")] + [(x, x) for x in available_inputs]
        beam.value = previous_input if previous_input in available_inputs else "all"
        selected = np.flatnonzero(
            matching & ((beam.value == "all") | (beam_labels == beam.value))
        )
        row.options = [(f"{i + 1:03d}", int(i)) for i in selected] or [
            ("No records", -1)
        ]
        row.index = 0
        row.disabled = not len(selected)
        filtering = False
        update()

    for control in (purpose, voltage, magnification, beam, aperture):
        control.observe(filter_rows, names="value")
    for control in (row, section, model, contrast):
        control.observe(update, names="value")
    previous.on_click(lambda _: setattr(row, "index", max(0, row.index - 1)))
    following.on_click(
        lambda _: setattr(row, "index", min(len(row.options) - 1, row.index + 1))
    )
    filter_rows()
    # Initialise the embedded canvas before a frontend exists (also saves its PNG).
    fig.canvas.manager.resize(*fig.canvas.get_width_height())
    fig.canvas.draw()
    grid = widgets.GridBox(
        [purpose, voltage, magnification, beam, aperture],
        layout=widgets.Layout(
            grid_template_columns="repeat(auto-fit, minmax(260px, 1fr))",
            grid_gap="10px 18px",
        ),
    )
    advanced = widgets.Accordion(children=[details], selected_index=None)
    advanced.set_title(0, "Lens currents and model details")
    return widgets.VBox(
        [
            widgets.HTML(
                '<h3 style="margin:0 0 6px;">Explore the experiment</h3><p style="margin:0;color:#60717d;">Filter a kind of change, then step through its records. The beam and ray diagram stay together.</p>'
                + (
                    "<p><b>Geometric ray preview only — the current fit uses directly calculated moments, not these pixels. Diffraction rings are not simulated.</b></p>"
                    if images is None
                    else ""
                )
            ),
            grid,
            widgets.HBox(
                [previous, row, following, progress],
                layout=widgets.Layout(align_items="center", grid_gap="8px"),
            ),
            summary,
            widgets.HBox(
                [section, model, contrast],
                layout=widgets.Layout(grid_gap="18px", align_items="center"),
            ),
            fig.canvas,
            advanced,
            errors,
        ],
        layout=widgets.Layout(
            width="100%",
            max_width="1200px",
            padding="20px",
            grid_gap="16px",
            border="1px solid #dce3e8",
        ),
    )


def show_result(*, checks):
    from IPython.display import Markdown

    print(
        f"Fit RMS: {checks['fit_residual_rms']:.3f}; prediction RMS: {checks['prediction_residual_rms']:.3f}"
    )
    table = [
        "| Parameter | Fitted | Truth | Error | Standard deviation |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in checks["geometry"]:
        values = " | ".join(
            "—" if row[key] is None else f"{row[key]:.6g}"
            for key in ("fitted", "truth", "error", "standard_deviation")
        )
        table.append(f"| {row['parameter']} | {values} |")
    return Markdown("\n".join(table))
