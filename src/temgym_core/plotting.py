from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from .ray import Ray
from .source import make_waist_divergence_rays  # noqa: F401
from .components import (
    Component,
    DeflectionBiprism,
    Deflector,
    DoubleDeflector,
    Detector,
    Lens,
    PhaseBiprism,
)
from .run import run_iter_vmapped


@dataclass
class PlotParams:
    figsize: Tuple[float, float] = (9.0, 5.0)
    extent_scale: float = 0.7
    label_fontsize: int = 11
    font_family: str = "DejaVu Sans"
    text_color: str = "black"
    tick_color: str = "black"
    figure_facecolor: str = "white"
    axes_facecolor: str = "white"
    grid_major_color: str = "lightgrey"
    grid_minor_color: str = "#EEEEEE"
    grid_major_ls: str = "--"
    grid_minor_ls: str = ":"
    grid_major_lw: float = 0.5
    grid_minor_lw: float = 0.5
    ray_color: str = "tab:blue"
    ray_lw: float = 1.2
    ray_alpha: float = 0.8
    interior_ray_lw: float = 0.7
    interior_ray_alpha: float = 0.25
    center_ray_lw: float = 1.0
    center_ray_alpha: float = 0.45
    fill_color: str = "#87cefa"  # light sky blue
    fill_alpha: float = 0.20
    edge_lw: float = 1.8
    solid_beam: bool = False
    component_lw: float = 3.0
    x_padding_frac: float = 0.0
    fixed_xmax: float | None = None
    component_half_width_frac: float = 0.02
    label_gap_frac: float = 0.04
    label_right_pad_frac: float = 0.35
    show_side_guides: bool = False
    side_guide_color: str = "#D9D9D9"
    side_guide_lw: float = 1.0
    side_guide_ls: str = "--"
    side_guide_alpha: float = 0.9
    lens_height: float = 1e-5  # relative to figure height
    auto_lens_height: bool = False
    lens_height_frac: float = 0.03
    biprism_radius: float = 1e-7  # radius of circle to draw biprism
    solution_waist_color: str = "tab:red"
    solution_divergence_color: str = "tab:green"
    solution_ray_lw: float = 2.0
    solution_ray_alpha: float = 0.95
    solution_ray_ls: str = "-"


def legacy_beam_plot_params(**overrides) -> PlotParams:
    """Return a pre-tuned dark plotting style with solid beam rendering."""
    params = PlotParams(
        figsize=(6.0, 10.0),
        extent_scale=0.80,
        label_fontsize=12,
        font_family="DejaVu Sans",
        text_color="black",
        tick_color="black",
        figure_facecolor="white",
        axes_facecolor="white",
        grid_major_color="black",
        grid_minor_color="black",
        ray_color="black",
        ray_lw=0.65,
        ray_alpha=0.28,
        interior_ray_lw=0.65,
        interior_ray_alpha=0.30,
        center_ray_lw=1.0,
        center_ray_alpha=0.50,
        fill_color="#43C78E",
        fill_alpha=0.50,
        edge_lw=1.5,
        solid_beam=True,
        component_lw=6.0,
        x_padding_frac=0.20,
        component_half_width_frac=0.92,
        label_gap_frac=0.04,
        label_right_pad_frac=0.42,
        show_side_guides=True,
        side_guide_color="#DFE4EA",
        side_guide_lw=1.1,
        side_guide_ls="--",
        side_guide_alpha=0.85,
        lens_height=1e-5,
        auto_lens_height=True,
        lens_height_frac=0.03,
    )
    unknown = set(overrides) - set(PlotParams.__dataclass_fields__)
    if unknown:
        unknown_fmt = ", ".join(sorted(unknown))
        raise TypeError(f"Unknown PlotParams override(s): {unknown_fmt}")
    return replace(params, **overrides)


def _as_name(obj: object) -> str:
    return type(obj).__name__


def _detector_half_width_x(detector: Detector) -> float:
    # Detector is ShapeYX / ScaleYX, i.e. (y, x). Width in x uses index 1.
    try:
        return float(detector.pixel_size[1] * detector.shape[1] / 2.0)
    except Exception:
        return float(detector.pixel_size[0] * detector.shape[0] / 2.0)


def _style_axes(ax: mpl.axes.Axes, p: PlotParams) -> None:
    ax.figure.patch.set_facecolor(p.figure_facecolor)
    ax.set_facecolor(p.axes_facecolor)

    ax.tick_params(axis="both", which="major", labelsize=12, colors=p.tick_color)
    ax.tick_params(axis="both", which="minor", labelsize=10, colors=p.tick_color)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)

    ax.grid(color=p.grid_major_color, linestyle=p.grid_major_ls, linewidth=p.grid_major_lw)
    ax.grid(which="minor", color=p.grid_minor_color, linestyle=p.grid_minor_ls, linewidth=p.grid_minor_lw)

    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_fontfamily(p.font_family)
        label.set_color(p.tick_color)


def _label_component(
    ax: mpl.axes.Axes,
    x: float,
    z: float,
    name: str,
    p: PlotParams,
    *,
    zorder: int = 1000,
) -> None:
    ax.text(
        x,
        z,
        name,
        fontsize=p.label_fontsize,
        va="center",
        ha="left",
        zorder=zorder,
        color=p.text_color,
        fontfamily=p.font_family,
        clip_on=False,
    )


def plot_model(
    components: Sequence[Component],
    *,
    rays: Ray | None = None,
    solution_rays: Ray | None = None,
    plot_params: PlotParams = legacy_beam_plot_params(),
    ax: mpl.axes.Axes | None = None,
    band_mode: str = "fill",  # "fill" (envelope fill) or "lines" (draw lines between rays)
    yscale: str = "linear",   # "linear", "log", or "symlog"
    y_linthresh: float = 1e-6,  # linthresh for symlog
    include_input_rays: bool = True,
):
    """Plot a  schematic of a model (components vs z) with ray bundle.

    Parameters
    ----------
    components : sequence of Component
        Model elements ordered by increasing z.
    rays : Ray, optional
        A Ray or a Ray bundle to use as the starting input.
    solution_rays : Ray, optional
        Optional 2-ray bundle overlaid as (waist ray, divergence ray).
        See `make_waist_divergence_rays` to construct this directly from
        an input waist and voltage/wavelength.
    plot_params : PlotParams, optional
        Style parameters for the plot. Set `fixed_xmax` to keep horizontal
        scene geometry fixed across frames (useful for animations).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, a new figure and axes are created.
    band_mode : {"fill", "lines"}, default "fill"
        "fill": fill between the two edge rays (envelope).
        "lines": draw horizontal line segments between adjacent rays at each z.
    yscale : {"linear", "log", "symlog"}, default "linear"
        Set y-axis (z) scaling. "log" requires all z>0; otherwise falls back to "symlog".
    y_linthresh : float, default 1e-6
        Linear range around zero used when yscale="symlog".
    include_input_rays : bool, default True
        If True, prepend the input ray state so plotting starts at the
        initial ray z-position instead of the first component step.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    p = plot_params
    if rays is None:
        raise ValueError("plot_model requires `rays` to be provided.")

    # Accumulate rays after each step (including propagations)
    steps = tuple(run_iter_vmapped(rays, components))
    if include_input_rays:
        steps = (rays,) + steps

    X, Z = _stack_ray_positions(steps)

    # Optional solution-ray overlay (waist/divergence basis)
    X_solution = None
    Z_solution = None
    if solution_rays is not None:
        solution_steps = tuple(run_iter_vmapped(solution_rays, components))
        if include_input_rays:
            solution_steps = (solution_rays,) + solution_steps
        X_solution, Z_solution = _stack_ray_positions(solution_steps)
        if X_solution.size > 0 and X_solution.shape[1] != 2:
            raise ValueError(
                "`solution_rays` must contain exactly 2 rays "
                "(waist ray, divergence ray)."
            )

    if X.size == 0:
        # Nothing to plot
        if ax is None:
            fig, ax = plt.subplots(figsize=p.figsize)
        else:
            fig = ax.figure
        _style_axes(ax, p)
        return fig, ax

    # Determine x extent using both beam and detector width if present.
    # When fixed_xmax is provided, keep horizontal scene scaling stable.
    detector_range_x = 0.0
    for c in components:
        if isinstance(c, Detector):
            # Width in x direction
            detector_range_x = max(
                detector_range_x,
                _detector_half_width_x(c),
            )

    if p.fixed_xmax is None:
        max_beam_x = float(np.max(np.abs(X)))
        if X_solution is not None and X_solution.size > 0:
            max_beam_x = max(max_beam_x, float(np.max(np.abs(X_solution))))
        base_max_x = max(max_beam_x, detector_range_x, np.finfo(float).eps)
    else:
        fixed_xmax = float(p.fixed_xmax)
        if not np.isfinite(fixed_xmax):
            raise ValueError("plot_params.fixed_xmax must be finite when provided.")
        base_max_x = max(abs(fixed_xmax), detector_range_x, np.finfo(float).eps)

    max_x = base_max_x * (1.0 + max(0.0, p.x_padding_frac))
    component_x = max_x * float(np.clip(p.component_half_width_frac, 0.0, 1.0))

    # z limits and ticks
    comp_zs = [float(getattr(c, "z")) for c in components if hasattr(c, "z")]
    min_z_candidates = [float(np.min(Z))]
    max_z_candidates = [float(np.max(Z))]
    if X_solution is not None and X_solution.size > 0:
        min_z_candidates.append(float(np.min(Z_solution)))
        max_z_candidates.append(float(np.max(Z_solution)))
    min_z = min(min_z_candidates + comp_zs) if comp_zs else min(min_z_candidates)
    max_z = max(max_z_candidates + comp_zs) if comp_zs else max(max_z_candidates)
    z_span = max(np.finfo(float).eps, max_z - min_z)
    lens_height = p.lens_height
    if p.auto_lens_height:
        lens_height = max(lens_height, p.lens_height_frac * z_span)

    label_gap_x = max(np.finfo(float).eps, p.label_gap_frac * max_x)
    label_x = max(p.extent_scale * max_x, component_x + label_gap_x)
    x_max_plot = max(max_x, label_x + p.label_right_pad_frac * max_x)

    if ax is None:
        fig, ax = plt.subplots(figsize=p.figsize)
    else:
        fig = ax.figure

    # Style
    _style_axes(ax, p)

    # Y-axis scaling (z-axis)
    scale = str(yscale).lower().strip()
    if scale == "log":
        # Require strictly positive z for log
        if min_z <= 0:
            ax.set_yscale("symlog", linthresh=y_linthresh)
            scale = "symlog"
        else:
            ax.set_yscale("log")
    elif scale == "symlog":
        ax.set_yscale("symlog", linthresh=y_linthresh)
    else:
        ax.set_yscale("linear")

    # Ticks and limits
    ytick_values = [float(np.min(Z)), float(np.max(Z))]
    if X_solution is not None and X_solution.size > 0:
        ytick_values.extend([float(np.min(Z_solution)), float(np.max(Z_solution))])
    yticks = sorted(set(ytick_values + comp_zs))
    if scale == "log":
        yticks = [t for t in yticks if t > 0]
    ax.set_yticks(yticks)
    ax.set_xlim([-x_max_plot, x_max_plot])
    ax.set_ylim([max_z, min_z])  # invert z-axis (optical drawings convention)

    if p.show_side_guides and component_x > 0:
        for xg in (-component_x, component_x):
            ax.plot(
                [xg, xg],
                [min_z, max_z],
                color=p.side_guide_color,
                linewidth=p.side_guide_lw,
                linestyle=p.side_guide_ls,
                alpha=p.side_guide_alpha,
                zorder=2,
            )

    # Rays
    _ = plot_ray_bundle(ax, X, Z, p, band_mode=band_mode)
    if X_solution is not None and X_solution.size > 0:
        _ = plot_solution_rays(ax, X_solution, Z_solution, p)

    # Components
    aspect = p.figsize[1] / p.figsize[0]
    left_x = -component_x
    right_x = component_x

    def _draw_deflector(z_pos: float) -> None:
        ax.plot(
            [left_x, 0], [z_pos, z_pos], color="lightcoral",
            linewidth=p.component_lw, zorder=999,
        )
        ax.plot(
            [0, right_x], [z_pos, z_pos], color="lightblue",
            linewidth=p.component_lw, zorder=999,
        )
        ax.plot(
            [left_x, right_x], [z_pos, z_pos], color="k", alpha=0.8,
            linewidth=p.component_lw + 2, zorder=998,
        )

    for c in components:
        name = _as_name(c)
        if isinstance(c, DoubleDeflector):
            _label_component(ax, label_x, c.z, name, p)
            _draw_deflector(c.z)
            _draw_deflector(c.z_second)
        elif isinstance(c, Deflector):
            _label_component(ax, label_x, c.z, name, p)
            _draw_deflector(c.z)
        elif isinstance(c, Lens):
            lens_width = max(np.finfo(float).eps, 2.0 * component_x)
            _label_component(ax, label_x, c.z, name, p)
            ax.add_patch(
                mpl.patches.Arc(
                    (0, c.z), lens_width, height=lens_height / aspect,
                    theta1=0, theta2=180, linewidth=1,
                    fill=False, zorder=999, edgecolor="k",
                )
            )
            ax.add_patch(
                mpl.patches.Arc(
                    (0, c.z), lens_width, height=lens_height / aspect,
                    theta1=180, theta2=0, linewidth=1,
                    fill=False, zorder=-1, edgecolor="k",
                )
            )
        elif isinstance(c, Detector):
            _label_component(ax, label_x, c.z, name, p)
            det_rx = _detector_half_width_x(c)
            ax.plot([-det_rx, det_rx], [c.z, c.z], color="dimgrey", zorder=1000, linewidth=5)
        elif isinstance(c, (DeflectionBiprism, PhaseBiprism)):
            ax.add_patch(plt.Circle((0, c.z), p.biprism_radius, edgecolor="k", facecolor="w", zorder=1000))
        else:
            # Generic annotation at z
            if hasattr(c, "z"):
                _label_component(ax, label_x, float(getattr(c, "z")), name, p, zorder=500)

    return fig, ax


def plot_ray_bundle(
    ax: mpl.axes.Axes,
    X: np.ndarray,
    Z: np.ndarray,
    p: PlotParams,
    *,
    band_mode: str = "fill",
):
    """
    Plot a ray bundle and optional band/envelope.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    X : ndarray, shape (nsteps, nrays)
        Ray x positions along propagation.
    Z : ndarray, shape (nsteps,)
        z positions corresponding to rows in X.
    p : PlotParams
        Plot styling parameters.
    band_mode : {"fill", "lines"}
        "fill": fill between edge rays; "lines": horizontal segments between rays.
    """
    nrays = X.shape[1] if X.ndim == 2 else 1
    mode = str(band_mode).lower().strip()

    # Draw rays (optionally as "solid beam" with faint interior lines)
    ray_lines = []
    center_lines = []
    if p.solid_beam and nrays >= 2:
        order0 = np.argsort(X[0, :])
        interior_idx = order0[1:-1]
        if interior_idx.size > 0:
            ray_lines += ax.plot(
                X[:, interior_idx],
                Z[:, None],
                color=p.ray_color,
                linewidth=p.interior_ray_lw,
                alpha=p.interior_ray_alpha,
                zorder=1,
            )
        if nrays >= 3:
            center_idx = int(order0[nrays // 2])
            center_lines += ax.plot(
                X[:, center_idx],
                Z,
                color=p.ray_color,
                linewidth=p.center_ray_lw,
                alpha=p.center_ray_alpha,
                zorder=2,
            )
    else:
        ray_lines = ax.plot(
            X, Z[:, None],
            color=p.ray_color,
            linewidth=p.ray_lw,
            alpha=p.ray_alpha,
            zorder=1,
        )

    # Band/envelope rendering
    band_artists = []
    edge_lines = []

    if nrays >= 2:
        min_x_idx = int(np.argmin(X[0, :]))
        max_x_idx = int(np.argmax(X[0, :]))

        if mode == "fill":
            band_artists.append(
                ax.fill_betweenx(
                    Z,
                    X[:, min_x_idx],
                    X[:, max_x_idx],
                    color=p.fill_color,
                    edgecolor=p.fill_color,
                    zorder=0,
                    alpha=p.fill_alpha,
                    linewidth=0.0,
                )
            )
            edge_lines += ax.plot(
                X[:, min_x_idx], Z,
                color=p.ray_color,
                linewidth=p.edge_lw,
                alpha=p.ray_alpha,
                zorder=1,
            )
            edge_lines += ax.plot(
                X[:, max_x_idx], Z,
                color=p.ray_color,
                linewidth=p.edge_lw,
                alpha=p.ray_alpha,
                zorder=1,
            )
        elif mode == "lines":
            for k, z in enumerate(Z):
                xrow = np.sort(X[k, :])
                for i in range(nrays - 1):
                    edge_lines += ax.plot(
                        [xrow[i], xrow[i + 1]],
                        [z, z],
                        color=p.ray_color,
                        linewidth=max(0.6, p.ray_lw * 0.7),
                        alpha=min(0.9, p.ray_alpha),
                        zorder=0,
                    )

    return {
        "ray_lines": ray_lines,
        "center_lines": center_lines,
        "band_artists": band_artists,
        "edge_lines": edge_lines,
    }


def plot_solution_rays(
    ax: mpl.axes.Axes,
    X_solution: np.ndarray,
    Z_solution: np.ndarray,
    p: PlotParams,
):
    """Plot the two solution rays (waist and divergence)."""
    if X_solution.ndim != 2 or X_solution.shape[1] != 2:
        raise ValueError(
            "Expected `X_solution` shape (nsteps, 2) for "
            "(waist ray, divergence ray)."
        )

    waist_line = ax.plot(
        X_solution[:, 0],
        Z_solution,
        color=p.solution_waist_color,
        linewidth=p.solution_ray_lw,
        alpha=p.solution_ray_alpha,
        linestyle=p.solution_ray_ls,
        zorder=3,
    )
    divergence_line = ax.plot(
        X_solution[:, 1],
        Z_solution,
        color=p.solution_divergence_color,
        linewidth=p.solution_ray_lw,
        alpha=p.solution_ray_alpha,
        linestyle=p.solution_ray_ls,
        zorder=3,
    )
    return {
        "waist_line": waist_line,
        "divergence_line": divergence_line,
    }


# Functionalized: build X (positions) and Z (z positions) from simulation steps
def _stack_ray_positions(
    steps_seq: Sequence[Tuple[object, Ray]]
) -> Tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    zs: list[float] = []
    for r in steps_seq:
        x = np.atleast_1d(np.asarray(r.x))
        z_arr = np.asarray(r.z)
        z_val = float(np.mean(z_arr))  # z identical across bundle; use scalar mean
        xs.append(x)
        zs.append(z_val)

    if not xs:
        return np.empty((0, 0), dtype=float), np.empty((0,), dtype=float)

    X_out = np.stack(xs, axis=0)  # (nsteps, nrays)
    Z_out = np.asarray(zs, dtype=float)  # (nsteps,)
    return X_out, Z_out
