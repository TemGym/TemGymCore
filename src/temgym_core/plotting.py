from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .components import (
    Component,
    DeflectionBiprism,
    Deflector,
    Detector,
    DoubleDeflector,
    ElectromagneticLens,
    HardEdgeSolenoid,
    InterpolatedSample2D,
    Lens,
    MagneticPhaseSample,
    PhaseBiprism,
    Plane,
    RandomPhaseSample,
    SigmoidAperture,
    ThickLens,
)
from .ray import Ray
from .run import run_iter_vmapped
from .source import make_waist_divergence_rays


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


def _detector_half_width_y(detector: Detector) -> float:
    # Detector is ShapeYX / ScaleYX, i.e. (y, x). Height in y uses index 0.
    try:
        return float(detector.pixel_size[0] * detector.shape[0] / 2.0)
    except Exception:
        return float(detector.pixel_size[1] * detector.shape[1] / 2.0)


def _detector_radius(detector: Detector) -> float:
    return float(np.hypot(_detector_half_width_x(detector), _detector_half_width_y(detector)))


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


def _half_plot_metric(rays: Ray) -> np.ndarray:
    x0 = np.ravel(np.atleast_1d(np.asarray(rays.x, dtype=float)))
    dx0 = np.ravel(np.atleast_1d(np.asarray(rays.dx, dtype=float)))
    n = x0.size

    if dx0.size not in (1, n):
        raise ValueError(
            "Cannot infer half-ray split: rays.dx must be scalar or match rays.x size."
        )
    if dx0.size == 1 and n > 1:
        dx0 = np.broadcast_to(dx0, (n,))

    x_scale = max(1.0, float(np.max(np.abs(x0))))
    x_tol = 10.0 * np.finfo(float).eps * x_scale
    if float(np.max(x0) - np.min(x0)) > x_tol:
        return x0

    dx_scale = max(1.0, float(np.max(np.abs(dx0))))
    dx_tol = 10.0 * np.finfo(float).eps * dx_scale
    if float(np.max(dx0) - np.min(dx0)) > dx_tol:
        return dx0

    # Final fallback: deterministic split by index order.
    return np.arange(n, dtype=float) - 0.5 * (n - 1)


def _subset_rays_by_half_sign(
    rays: Ray,
    ray_half_sign: int | None,
) -> Ray:
    if ray_half_sign is None:
        return rays

    sign = int(ray_half_sign)
    if sign not in (-1, 1):
        raise ValueError("ray_half_sign must be one of {None, -1, +1}.")

    metric = _half_plot_metric(rays)
    if metric.size <= 1:
        return rays

    metric_scale = max(1.0, float(np.max(np.abs(metric))))
    metric_tol = 10.0 * np.finfo(float).eps * metric_scale

    if sign > 0:
        keep = metric >= -metric_tol
    else:
        keep = metric <= metric_tol

    idx = np.flatnonzero(keep)
    if idx.size == 0:
        idx = np.asarray([int(np.argmax(sign * metric))], dtype=int)
    return rays[idx]


def _ray_side_sign_from_metric(rays: Ray) -> np.ndarray:
    metric = np.ravel(np.atleast_1d(np.asarray(_half_plot_metric(rays), dtype=float)))
    if metric.size == 0:
        return metric
    scale = max(1.0, float(np.max(np.abs(metric))))
    tol = 10.0 * np.finfo(float).eps * scale
    sign = np.zeros_like(metric)
    sign[metric > tol] = 1.0
    sign[metric < -tol] = -1.0
    return sign


# Valid ray_coordinate modes
_RAY_COORD_MODES = ("x", "y", "x_rot", "y_rot", "r")
_RAY_COORD_ALIASES = {
    "x_corot": "x_rot",
    "aligned": "x_rot",
    "co_rot": "x_rot",
    "co-rot": "x_rot",
}


def _plotting_voltage_from_rays(rays: Ray, default: float = 200e3) -> float | np.ndarray:
    """Return scalar or per-ray voltage values for plotting helpers."""
    voltage = getattr(rays, "voltage", None)
    if voltage is None:
        return float(default)

    voltage_arr = np.asarray(voltage, dtype=float)
    if voltage_arr.size == 0:
        return float(default)
    if voltage_arr.ndim == 0 or voltage_arr.size == 1:
        return float(np.ravel(voltage_arr)[0])
    return np.ravel(voltage_arr)


def _normalize_ray_coordinate_mode(ray_coordinate: str) -> str:
    mode = str(ray_coordinate).lower().strip()
    mode = _RAY_COORD_ALIASES.get(mode, mode)
    if mode not in _RAY_COORD_MODES:
        raise ValueError(
            f"ray_coordinate must be one of {set(_RAY_COORD_MODES) | set(_RAY_COORD_ALIASES)}, "
            f"got {ray_coordinate!r}."
        )
    return mode


def _compute_cumulative_rotation(
    components: Sequence[Component],
    *,
    voltage: float | np.ndarray = 200e3,
    include_input_rays: bool = True,
) -> np.ndarray:
    """Compute the cumulative Larmor rotation angle at each ray step.

    `run_iter` yields 2 steps per component: (propagation, application).
    Rotation is applied during electromagnetic-lens and solenoid steps.

    Returns an array of shape ``(n_steps,)`` for scalar voltage, or
    ``(n_steps, n_rays)`` for per-ray voltage, with cumulative angles
    in radians.
    """
    from .components import ElectromagneticLens as _EMLens

    voltage_arr = np.asarray(voltage, dtype=float)
    vector_voltage = voltage_arr.ndim > 0 and voltage_arr.size > 1
    voltage_use: float | np.ndarray
    if vector_voltage:
        voltage_use = np.ravel(voltage_arr)
        zero_angle: float | np.ndarray = np.zeros_like(voltage_use, dtype=float)
        cum: float | np.ndarray = zero_angle.copy()
    else:
        voltage_use = float(np.ravel(voltage_arr)[0]) if voltage_arr.size else 200e3
        zero_angle = 0.0
        cum = 0.0

    angles: list[float | np.ndarray] = []
    if include_input_rays:
        angles.append(np.asarray(zero_angle, dtype=float).copy())  # prepended input-ray step

    for c in components:
        # Propagation step – no rotation change
        angles.append(np.asarray(cum, dtype=float).copy())
        # Component application step
        if isinstance(c, _EMLens):
            cum = np.asarray(cum, dtype=float) + np.asarray(
                c.rotation_angle(voltage_use), dtype=float
            )
        elif isinstance(c, HardEdgeSolenoid):
            cum = np.asarray(cum, dtype=float) - np.asarray(c.q, dtype=float)
        angles.append(np.asarray(cum, dtype=float).copy())

    return np.asarray(angles, dtype=float)


def plot_model(
    components: Sequence[Component],
    *,
    rays: Ray | None = None,
    solution_rays: Ray | None = None,
    plot_params: PlotParams = legacy_beam_plot_params(),
    ax: mpl.axes.Axes | None = None,
    band_mode: str = "fill",  # "fill", "gradient", or "lines"
    ray_coordinate: str = "x",  # "x", "y", "x_rot", "y_rot", or "r"
    ray_half_sign: int | None = None,  # None=all rays, +1=positive half, -1=negative half
    r_side_by_sign: bool = False,
    break_same_z_jumps: bool = True,
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
    band_mode : {"fill", "gradient", "lines"}, default "fill"
        "fill": fill between the two edge rays (envelope).
        "lines": draw horizontal line segments between adjacent rays at each z.
    ray_coordinate : {"x", "y", "x_rot", "y_rot", "r"}, default "x"
        Horizontal coordinate used for plotting rays.

        * ``"x"`` / ``"y"`` – lab-frame position.
        * ``"x_rot"`` / ``"y_rot"`` – position in the co-rotating frame
          obtained by undoing the cumulative Larmor rotation of every
          `ElectromagneticLens` in *components*.  Rays stay "aligned"
          through rotations so the diagram is easier to read.
        * ``"r"`` – radial magnitude ``sqrt(x**2 + y**2)``.

        Legacy aliases ``"x_corot"`` and ``"aligned"`` map to ``"x_rot"``.
    ray_half_sign : {None, -1, +1}, default None
        Optionally plot only one half of the input `rays` bundle.
        The split is inferred from the initial bundle sign (x if spread exists,
        otherwise dx). Use +1 for the non-negative half and -1 for the
        non-positive half. `solution_rays` are not filtered.
    r_side_by_sign : bool, default False
        If True and `ray_coordinate="r"`, apply a per-ray sign inferred from
        the input bundle (x if spread exists, otherwise dx) and plot signed
        radius (±sqrt(x^2 + y^2)). This places rays on left/right without
        duplicating mirrored copies.
    break_same_z_jumps : bool, default True
        If True, split polylines where successive samples have the same z but
        different x (avoids horizontal jump segments at component planes). Set
        False to keep those same-z jumps visually connected.
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
    coord_mode = _normalize_ray_coordinate_mode(ray_coordinate)
    side_sign_in_r = bool(r_side_by_sign and coord_mode == "r")
    two_sided_r = bool(side_sign_in_r)
    rays_for_plot = _subset_rays_by_half_sign(rays, ray_half_sign)
    radial_sign = _ray_side_sign_from_metric(rays_for_plot) if side_sign_in_r else None

    # Compute cumulative rotation from EM lenses for _rot modes
    _voltage = _plotting_voltage_from_rays(rays_for_plot)
    cum_rotation = (
        _compute_cumulative_rotation(components, voltage=_voltage, include_input_rays=include_input_rays)
        if coord_mode in ("x_rot", "y_rot")
        else None
    )

    # Accumulate rays after each step (including propagations)
    steps = tuple(run_iter_vmapped(rays_for_plot, components))
    if include_input_rays:
        steps = (rays_for_plot,) + steps

    X, Z = _stack_ray_positions(
        steps,
        ray_coordinate=coord_mode,
        radial_sign=radial_sign,
        cumulative_rotation=cum_rotation,
    )

    # Optional solution-ray overlay (waist/divergence basis)
    X_solution = None
    Z_solution = None
    if solution_rays is not None:
        solution_steps = tuple(run_iter_vmapped(solution_rays, components))
        if include_input_rays:
            solution_steps = (solution_rays,) + solution_steps
        solution_radial_sign = _ray_side_sign_from_metric(solution_rays) if side_sign_in_r else None
        solution_cum_rotation = (
            _compute_cumulative_rotation(
                components,
                voltage=_plotting_voltage_from_rays(solution_rays),
                include_input_rays=include_input_rays,
            )
            if coord_mode in ("x_rot", "y_rot")
            else None
        )
        X_solution, Z_solution = _stack_ray_positions(
            solution_steps,
            ray_coordinate=coord_mode,
            radial_sign=solution_radial_sign,
            cumulative_rotation=solution_cum_rotation,
        )
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
            # Detector horizontal range in current plotting coordinate.
            if coord_mode == "r":
                detector_extent = _detector_radius(c)
            elif coord_mode in ("y", "y_rot"):
                detector_extent = _detector_half_width_y(c)
            else:
                detector_extent = _detector_half_width_x(c)
            detector_range_x = max(
                detector_range_x,
                detector_extent,
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
    if coord_mode == "r" and not two_sided_r:
        ax.set_xlim([0.0, x_max_plot])
    else:
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
    _ = plot_ray_bundle(
        ax,
        X,
        Z,
        p,
        band_mode=band_mode,
        break_same_z_jumps=break_same_z_jumps,
    )
    if X_solution is not None and X_solution.size > 0:
        _ = plot_solution_rays(
            ax,
            X_solution,
            Z_solution,
            p,
            break_same_z_jumps=break_same_z_jumps,
        )

    # Components
    aspect = p.figsize[1] / p.figsize[0]
    left_x = 0.0 if (coord_mode == "r" and not two_sided_r) else -component_x
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
            if coord_mode == "r":
                det_rx = _detector_radius(c)
                if two_sided_r:
                    ax.plot([-det_rx, det_rx], [c.z, c.z], color="dimgrey", zorder=1000, linewidth=5)
                else:
                    ax.plot([0.0, det_rx], [c.z, c.z], color="dimgrey", zorder=1000, linewidth=5)
            else:
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
    break_same_z_jumps: bool = True,
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
    band_mode : {"fill", "gradient", "lines"}
        "fill": fill between edge rays; "gradient": soft transverse shading
        within the pointwise envelope; "lines": horizontal segments between
        rays. Gradient shading is a display style, not calibrated intensity.
    break_same_z_jumps : bool, default True
        If True, insert NaN separators where equal-z samples would otherwise
        draw horizontal jump segments.
    """
    nrays = X.shape[1] if X.ndim == 2 else 1
    mode = str(band_mode).lower().strip()
    if break_same_z_jumps:
        X_plot, Z_plot = _break_same_z_jumps(X, Z)
    else:
        X_plot, Z_plot = _coerce_plot_arrays(X, Z)

    # Draw rays (optionally as "solid beam" with faint interior lines)
    ray_lines = []
    center_lines = []
    if p.solid_beam and nrays >= 2:
        order0 = np.argsort(X[0, :])
        interior_idx = order0[1:-1]
        if interior_idx.size > 0 and p.interior_ray_alpha > 0:
            ray_lines += ax.plot(
                X_plot[:, interior_idx],
                Z_plot[:, None],
                color=p.ray_color,
                linewidth=p.interior_ray_lw,
                alpha=p.interior_ray_alpha,
                zorder=1,
            )
        if nrays >= 3 and p.center_ray_alpha > 0:
            center_idx = int(order0[nrays // 2])
            center_lines += ax.plot(
                X_plot[:, center_idx],
                Z_plot,
                color=p.ray_color,
                linewidth=p.center_ray_lw,
                alpha=p.center_ray_alpha,
                zorder=2,
            )
    else:
        ray_lines = ax.plot(
            X_plot, Z_plot[:, None],
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

        if mode == "gradient":
            # Nested vector fills avoid a bitmap and remain smooth in SVG/PDF.
            # Min/max at each plane preserves the envelope through crossings.
            lower, upper = np.min(X_plot,axis=1), np.max(X_plot,axis=1)
            centre = .5*(lower+upper)
            radius = .5*(upper-lower)
            previous_opacity = 0.
            for fraction in np.linspace(1.,0.,40,endpoint=False):
                profile = (np.exp(-2*fraction**2)-np.exp(-2))/(1-np.exp(-2))
                opacity = np.clip(p.fill_alpha,0.,1.)*(.025+.975*profile)
                layer_alpha = (opacity-previous_opacity)/max(1-previous_opacity,np.finfo(float).eps)
                band_artists.append(ax.fill_betweenx(
                    Z_plot,centre-fraction*radius,centre+fraction*radius,
                    facecolor=p.fill_color,edgecolor="none",linewidth=0.,
                    antialiased=False,alpha=layer_alpha,zorder=0,
                ))
                previous_opacity = opacity
            if p.edge_lw > 0:
                for edge in (lower,upper):
                    edge_lines += ax.plot(edge,Z_plot,color=p.ray_color,
                        linewidth=p.edge_lw,alpha=p.ray_alpha,zorder=1,
                        solid_joinstyle="round",solid_capstyle="round")
        elif mode == "fill":
            band_artists.append(
                ax.fill_betweenx(
                    Z_plot,
                    X_plot[:, min_x_idx],
                    X_plot[:, max_x_idx],
                    color=p.fill_color,
                    edgecolor=p.fill_color,
                    zorder=0,
                    alpha=p.fill_alpha,
                    linewidth=0.0,
                )
            )
            edge_lines += ax.plot(
                X_plot[:, min_x_idx], Z_plot,
                color=p.ray_color,
                linewidth=p.edge_lw,
                alpha=p.ray_alpha,
                zorder=1,
            )
            edge_lines += ax.plot(
                X_plot[:, max_x_idx], Z_plot,
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
    *,
    break_same_z_jumps: bool = True,
):
    """Plot the two solution rays (waist and divergence)."""
    if X_solution.ndim != 2 or X_solution.shape[1] != 2:
        raise ValueError(
            "Expected `X_solution` shape (nsteps, 2) for "
            "(waist ray, divergence ray)."
        )
    if break_same_z_jumps:
        X_plot, Z_plot = _break_same_z_jumps(X_solution, Z_solution)
    else:
        X_plot, Z_plot = _coerce_plot_arrays(X_solution, Z_solution)

    waist_line = ax.plot(
        X_plot[:, 0],
        Z_plot,
        color=p.solution_waist_color,
        linewidth=p.solution_ray_lw,
        alpha=p.solution_ray_alpha,
        linestyle=p.solution_ray_ls,
        zorder=3,
    )
    divergence_line = ax.plot(
        X_plot[:, 1],
        Z_plot,
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


def _coerce_plot_arrays(
    X: np.ndarray,
    Z: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    X_arr = np.asarray(X, dtype=float)
    Z_arr = np.asarray(Z, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr[:, None]
    if X_arr.ndim != 2 or Z_arr.ndim != 1 or X_arr.shape[0] != Z_arr.shape[0]:
        raise ValueError("Expected X shape (nsteps, nrays) and Z shape (nsteps,).")
    return X_arr, Z_arr


def _break_same_z_jumps(
    X: np.ndarray,
    Z: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Insert NaN separators where equal-z samples would draw x jumps."""
    X_arr, Z_arr = _coerce_plot_arrays(X, Z)
    if X_arr.shape[0] < 2:
        return X_arr, Z_arr

    z_scale = max(1.0, float(np.nanmax(np.abs(Z_arr))))
    x_scale = max(1.0, float(np.nanmax(np.abs(X_arr))))
    z_atol = 10.0 * np.finfo(float).eps * z_scale
    x_atol = 10.0 * np.finfo(float).eps * x_scale

    same_z = np.isclose(np.diff(Z_arr), 0.0, rtol=0.0, atol=z_atol)
    same_x = np.all(
        np.isclose(np.diff(X_arr, axis=0), 0.0, rtol=0.0, atol=x_atol),
        axis=1,
    )
    break_after = np.flatnonzero(same_z & ~same_x)
    if break_after.size == 0:
        return X_arr, Z_arr

    n_steps, n_rays = X_arr.shape
    X_out = np.empty((n_steps + break_after.size, n_rays), dtype=float)
    Z_out = np.empty((n_steps + break_after.size,), dtype=float)

    out_idx = 0
    break_mask = np.zeros((n_steps - 1,), dtype=bool)
    break_mask[break_after] = True
    for step_idx in range(n_steps - 1):
        X_out[out_idx] = X_arr[step_idx]
        Z_out[out_idx] = Z_arr[step_idx]
        out_idx += 1
        if break_mask[step_idx]:
            X_out[out_idx] = np.nan
            Z_out[out_idx] = np.nan
            out_idx += 1

    X_out[out_idx] = X_arr[-1]
    Z_out[out_idx] = Z_arr[-1]
    return X_out, Z_out


# Functionalized: build X (positions) and Z (z positions) from simulation steps
def _stack_ray_positions(
    steps_seq: Sequence[Ray],
    *,
    ray_coordinate: str = "x",
    radial_sign: np.ndarray | None = None,
    cumulative_rotation: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract plotting coordinates from a sequence of Ray steps.

    Parameters
    ----------
    steps_seq : sequence of Ray
        One Ray per simulation step.
    ray_coordinate : str
        One of ``"x"``, ``"y"``, ``"x_rot"``, ``"y_rot"``, ``"r"``.
    radial_sign : ndarray, optional
        Per-ray sign array for ``"r"`` mode with side-by-sign.
    cumulative_rotation : ndarray, optional
        Per-step cumulative Larmor rotation (radians) for ``"x_rot"`` /
        ``"y_rot"`` modes.  Length must match ``len(steps_seq)``.
    """
    coord_mode = _normalize_ray_coordinate_mode(ray_coordinate)
    if radial_sign is not None and coord_mode != "r":
        raise ValueError("`radial_sign` is only valid when ray_coordinate='r'.")
    if cumulative_rotation is not None and coord_mode not in ("x_rot", "y_rot"):
        raise ValueError("`cumulative_rotation` is only valid for 'x_rot' / 'y_rot' modes.")

    sign_arr: np.ndarray | None = None
    if radial_sign is not None:
        sign_arr = np.ravel(np.atleast_1d(np.asarray(radial_sign, dtype=float)))

    cum_rot: np.ndarray | None = None
    if cumulative_rotation is not None:
        cum_rot = np.asarray(cumulative_rotation, dtype=float)

    xs: list[np.ndarray] = []
    zs: list[float] = []
    for step_idx, r in enumerate(steps_seq):
        x_arr = np.asarray(r.x, dtype=float)
        y_arr = np.asarray(r.y, dtype=float)

        if coord_mode == "r":
            x_b, y_b = np.broadcast_arrays(x_arr, y_arr)
            rmag = np.sqrt(x_b * x_b + y_b * y_b)
            if sign_arr is not None:
                sign_use = sign_arr
                r_flat = np.ravel(np.atleast_1d(rmag))
                if sign_use.size not in (1, r_flat.size):
                    raise ValueError(
                        "radial_sign size must be scalar or match rays per step."
                    )
                if sign_use.size == 1 and r_flat.size > 1:
                    sign_use = np.broadcast_to(sign_use, (r_flat.size,))
                val = r_flat * sign_use
            else:
                val = np.atleast_1d(rmag)

        elif coord_mode in ("x_rot", "y_rot"):
            # De-rotate by cumulative Larmor angle.  With per-ray voltages,
            # `angle` is a vector and NumPy broadcasts it across the bundle.
            angle = 0.0 if cum_rot is None else np.asarray(cum_rot[step_idx], dtype=float)
            cos_a = np.cos(-angle)
            sin_a = np.sin(-angle)
            x_b, y_b = np.broadcast_arrays(x_arr, y_arr)
            x_derot = np.ravel(np.atleast_1d(x_b * cos_a - y_b * sin_a))
            y_derot = np.ravel(np.atleast_1d(x_b * sin_a + y_b * cos_a))
            val = x_derot if coord_mode == "x_rot" else y_derot

        elif coord_mode == "y":
            val = np.atleast_1d(y_arr)

        else:  # "x"
            val = np.atleast_1d(x_arr)

        z_arr = np.asarray(r.z)
        z_val = float(np.mean(z_arr))  # z identical across bundle; use scalar mean
        xs.append(val)
        zs.append(z_val)

    if not xs:
        return np.empty((0, 0), dtype=float), np.empty((0,), dtype=float)

    X_out = np.stack(xs, axis=0)  # (nsteps, nrays)
    Z_out = np.asarray(zs, dtype=float)  # (nsteps,)
    return X_out, Z_out


def plot_model_plotly(
    components: Sequence[Component],
    *,
    rays: Ray | None = None,
    solution_rays: Ray | None = None,
    component_labels: Sequence[str] | None = None,
    band_mode: str = "lines",
    ray_coordinate: str = "x",
    ray_half_sign: int | None = None,
    r_side_by_sign: bool = False,
    break_same_z_jumps: bool = True,
    yscale: str = "linear",
    y_linthresh: float = 1e-6,
    include_input_rays: bool = True,
    width: int = 900,
    height: int = 700,
    label_stagger: float = 0.12,
    show_component_labels: bool = True,
):
    """Plot a model and ray bundle using Plotly.

    This is a Plotly equivalent of :func:`plot_model` and uses the same
    ray propagation and coordinate extraction pipeline.
    """
    try:
        import plotly.graph_objects as go
    except Exception as exc:
        raise ImportError(
            "plot_model_plotly requires plotly. Install with `pip install plotly`."
        ) from exc

    if rays is None:
        raise ValueError("plot_model_plotly requires `rays` to be provided.")

    labels: Sequence[str] | None = None
    if component_labels is not None:
        labels = tuple(component_labels)

    coord_mode = _normalize_ray_coordinate_mode(ray_coordinate)
    side_sign_in_r = bool(r_side_by_sign and coord_mode == "r")
    two_sided_r = bool(side_sign_in_r)
    rays_for_plot = _subset_rays_by_half_sign(rays, ray_half_sign)
    radial_sign = _ray_side_sign_from_metric(rays_for_plot) if side_sign_in_r else None

    _voltage = _plotting_voltage_from_rays(rays_for_plot)
    cum_rotation = (
        _compute_cumulative_rotation(components, voltage=_voltage, include_input_rays=include_input_rays)
        if coord_mode in ("x_rot", "y_rot")
        else None
    )

    steps = tuple(run_iter_vmapped(rays_for_plot, components))
    if include_input_rays:
        steps = (rays_for_plot,) + steps

    X, Z = _stack_ray_positions(
        steps,
        ray_coordinate=coord_mode,
        radial_sign=radial_sign,
        cumulative_rotation=cum_rotation,
    )

    X_solution = None
    Z_solution = None
    if solution_rays is not None:
        solution_steps = tuple(run_iter_vmapped(solution_rays, components))
        if include_input_rays:
            solution_steps = (solution_rays,) + solution_steps
        solution_radial_sign = _ray_side_sign_from_metric(solution_rays) if side_sign_in_r else None
        solution_cum_rotation = (
            _compute_cumulative_rotation(
                components,
                voltage=_plotting_voltage_from_rays(solution_rays),
                include_input_rays=include_input_rays,
            )
            if coord_mode in ("x_rot", "y_rot")
            else None
        )
        X_solution, Z_solution = _stack_ray_positions(
            solution_steps,
            ray_coordinate=coord_mode,
            radial_sign=solution_radial_sign,
            cumulative_rotation=solution_cum_rotation,
        )
        if X_solution.size > 0 and X_solution.shape[1] != 2:
            raise ValueError(
                "`solution_rays` must contain exactly 2 rays "
                "(waist ray, divergence ray)."
            )

    if X.size == 0:
        return go.Figure()

    detector_range_x = 0.0
    for c in components:
        if isinstance(c, Detector):
            if coord_mode == "r":
                detector_extent = _detector_radius(c)
            elif coord_mode in ("y", "y_rot"):
                detector_extent = _detector_half_width_y(c)
            else:
                detector_extent = _detector_half_width_x(c)
            detector_range_x = max(detector_range_x, detector_extent)

    max_beam_x = float(np.max(np.abs(X)))
    if X_solution is not None and X_solution.size > 0:
        max_beam_x = max(max_beam_x, float(np.max(np.abs(X_solution))))
    max_x = max(max_beam_x, detector_range_x, np.finfo(float).eps)

    comp_zs = [float(getattr(c, "z")) for c in components if hasattr(c, "z")]
    min_z_candidates = [float(np.min(Z))]
    max_z_candidates = [float(np.max(Z))]
    if X_solution is not None and X_solution.size > 0:
        min_z_candidates.append(float(np.min(Z_solution)))
        max_z_candidates.append(float(np.max(Z_solution)))
    min_z = min(min_z_candidates + comp_zs) if comp_zs else min(min_z_candidates)
    max_z = max(max_z_candidates + comp_zs) if comp_zs else max(max_z_candidates)

    if break_same_z_jumps:
        X_plot, Z_plot = _break_same_z_jumps(X, Z)
    else:
        X_plot, Z_plot = _coerce_plot_arrays(X, Z)

    fig = go.Figure()

    nrays = X_plot.shape[1]
    for idx in range(nrays):
        fig.add_trace(
            go.Scattergl(
                x=np.asarray(X_plot[:, idx], dtype=float).tolist(),
                y=np.asarray(Z_plot, dtype=float).tolist(),
                mode="lines",
                line=dict(color="rgba(60,60,60,0.35)", width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    mode = str(band_mode).lower().strip()
    if nrays >= 2 and mode == "fill":
        min_x_idx = int(np.argmin(X[0, :]))
        max_x_idx = int(np.argmax(X[0, :]))
        fig.add_trace(
            go.Scatter(
                x=np.asarray(X_plot[:, min_x_idx], dtype=float).tolist(),
                y=np.asarray(Z_plot, dtype=float).tolist(),
                mode="lines",
                line=dict(color="rgba(80,120,200,0.0)", width=0),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.asarray(X_plot[:, max_x_idx], dtype=float).tolist(),
                y=np.asarray(Z_plot, dtype=float).tolist(),
                mode="lines",
                fill="tonextx",
                fillcolor="rgba(80,200,140,0.2)",
                line=dict(color="rgba(80,120,200,0.0)", width=0),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    if X_solution is not None and X_solution.size > 0:
        if break_same_z_jumps:
            X_sol_plot, Z_sol_plot = _break_same_z_jumps(X_solution, Z_solution)
        else:
            X_sol_plot, Z_sol_plot = _coerce_plot_arrays(X_solution, Z_solution)

        fig.add_trace(
            go.Scattergl(
                x=np.asarray(X_sol_plot[:, 0], dtype=float).tolist(),
                y=np.asarray(Z_sol_plot, dtype=float).tolist(),
                mode="lines",
                line=dict(color="red", width=2),
                name="waist",
            )
        )
        fig.add_trace(
            go.Scattergl(
                x=np.asarray(X_sol_plot[:, 1], dtype=float).tolist(),
                y=np.asarray(Z_sol_plot, dtype=float).tolist(),
                mode="lines",
                line=dict(color="green", width=2),
                name="divergence",
            )
        )

    component_half_width = 0.92 * max_x
    left_x = 0.0 if (coord_mode == "r" and not two_sided_r) else -component_half_width
    right_x = component_half_width

    component_tick_zs: list[float] = []
    for comp_idx, c in enumerate(components):
        if not hasattr(c, "z"):
            continue
        zc = float(getattr(c, "z"))
        if labels is not None and comp_idx < len(labels):
            cname = str(labels[comp_idx])
        else:
            cname = _as_name(c)
        component_tick_zs.append(zc)

        if isinstance(c, Detector):
            if coord_mode == "r":
                det_rx = _detector_radius(c)
                x0 = 0.0 if not two_sided_r else -det_rx
                x1 = det_rx
            elif coord_mode in ("y", "y_rot"):
                det_rx = _detector_half_width_y(c)
                x0, x1 = -det_rx, det_rx
            else:
                det_rx = _detector_half_width_x(c)
                x0, x1 = -det_rx, det_rx
            fig.add_trace(
                go.Scatter(
                    x=[float(x0), float(x1)],
                    y=[zc, zc],
                    mode="lines",
                    line=dict(color="dimgray", width=4),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[float(left_x), float(right_x)],
                    y=[zc, zc],
                    mode="lines",
                    line=dict(color="black", width=2),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        if show_component_labels:
            fig.add_trace(
                go.Scatter(
                    x=[float(right_x + (0.08 + label_stagger * (comp_idx % 3)) * max_x)],
                    y=[zc],
                    mode="text",
                    text=[f"{cname}  (z={zc:.4f} m)"],
                    textposition="middle left",
                    textfont=dict(size=11, color="black"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    fig.update_layout(
        width=int(width),
        height=int(height),
        template="plotly_white",
        margin=dict(l=40, r=280, t=30, b=40),
        showlegend=bool(solution_rays is not None),
    )

    if coord_mode == "r" and not two_sided_r:
        fig.update_xaxes(range=[0.0, max_x])
    else:
        fig.update_xaxes(range=[-max_x, max_x])

    scale = str(yscale).lower().strip()
    if scale == "log" and min_z > 0:
        fig.update_yaxes(type="log")
    elif scale == "symlog":
        fig.update_yaxes(type="linear")
        _ = y_linthresh
    else:
        fig.update_yaxes(type="linear")

    if component_tick_zs:
        tickvals = sorted(set([float(np.min(Z)), float(np.max(Z))] + component_tick_zs))
        fig.update_yaxes(tickmode="array", tickvals=tickvals)

    fig.update_yaxes(autorange="reversed", range=[max_z, min_z])
    fig.update_xaxes(title_text=coord_mode)
    fig.update_yaxes(title_text="z [m]")
    return fig


def _rotation_matrix_5x5(theta):
    """Build a 5×5 rotation matrix for angle *theta* (radians).

    Rotates (x, y) and (dx, dy) simultaneously::

        [[c, -s, 0, 0, 0],
         [s,  c, 0, 0, 0],
         [0,  0, c,-s, 0],
         [0,  0, s, c, 0],
         [0,  0, 0, 0, 1]]
    """
    c, s = np.cos(theta), np.sin(theta)
    R = np.eye(5)
    R[0, 0] = c;  R[0, 1] = -s
    R[1, 0] = s;  R[1, 1] = c
    R[2, 2] = c;  R[2, 3] = -s
    R[3, 2] = s;  R[3, 3] = c
    return R


def plot_abcd_profile(
    z,
    cumulative,
    labels=None,
    axis="x",
    elements=("A", "B", "C", "D"),
    rotation=None,
    reference=None,
    axes=None,
    **fig_kw,
):
    """Plot ABCD transfer-matrix elements as a function of z.

    Parameters
    ----------
    z : array-like, shape (N,)
        z-positions along the column (from :func:`solve_model_with_z`).
    cumulative : array-like, shape (N, 5, 5)
        Cumulative ABCD matrices at each step.
    labels : list of str, optional
        Step labels from :func:`solve_model_with_z`.  Steps whose label
        is *not* ``'drift'`` are drawn as vertical component markers.
    axis : {'x', 'y'}
        Transverse axis to plot.  ``'x'`` uses matrix indices (0, 2);
        ``'y'`` uses (1, 3).
    elements : tuple of str
        Which of ``('A', 'B', 'C', 'D')`` to include as sub-plots.
    rotation : array-like, shape (N,), optional
        Cumulative Larmor rotation angles (radians) at each step,
        as returned by :func:`solve_model_with_z`.  When provided the
        inverse rotation is applied to each cumulative matrix before
        extracting ABCD elements (co-rotating frame).
    reference : int or str, optional
        Reference plane for the transfer matrices.  If an *int*, it is
        the index into the step arrays.  If a *str*, it is matched
        against *labels* (uses the last match).  The ABCD elements are
        then computed relative to this plane:
        ``M_ref_to_k = M_cumulative[k] @ inv(M_cumulative[ref])``.
        Useful for checking imaging conditions (e.g. B = 0 from the
        sample to the detector).
    axes : array of Axes, optional
        Pre-existing matplotlib axes (one per element).  Created if *None*.
    **fig_kw
        Forwarded to :func:`matplotlib.pyplot.subplots`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : numpy.ndarray of matplotlib.axes.Axes
    """
    z = np.asarray(z)
    cumulative = np.asarray(cumulative)

    # Apply inverse Larmor rotation (co-rotating frame)
    if rotation is not None:
        rotation = np.asarray(rotation)
        cumulative = np.stack([
            _rotation_matrix_5x5(-theta) @ M
            for theta, M in zip(rotation, cumulative)
        ])

    # Re-reference cumulative matrices to a chosen plane
    if reference is not None:
        if isinstance(reference, str):
            # Find the last label match
            ref_idx = None
            if labels is not None:
                for i, lbl in enumerate(labels):
                    if lbl == reference:
                        ref_idx = i
            if ref_idx is None:
                raise ValueError(
                    f"reference label {reference!r} not found in labels"
                )
        else:
            ref_idx = int(reference)
        M_ref_inv = np.linalg.inv(cumulative[ref_idx])
        cumulative = np.stack([M @ M_ref_inv for M in cumulative])

    pos_idx, slope_idx = (0, 2) if axis == "x" else (1, 3)
    element_map = {
        "A": (pos_idx, pos_idx),
        "B": (pos_idx, slope_idx),
        "C": (slope_idx, pos_idx),
        "D": (slope_idx, slope_idx),
    }

    n_panels = len(elements)
    if axes is None:
        fig_kw.setdefault("figsize", (10, 2.5 * n_panels))
        fig, axes = plt.subplots(n_panels, 1, sharex=True, **fig_kw)
        if n_panels == 1:
            axes = np.array([axes])
    else:
        axes = np.atleast_1d(axes)
        fig = axes.flat[0].figure

    # Identify component (non-drift) positions for vertical markers
    comp_z, comp_names = [], []
    if labels is not None:
        for zi, lbl in zip(z, labels):
            if lbl != "drift":
                comp_z.append(float(zi))
                comp_names.append(lbl)

    for ax, elem in zip(axes.flat, elements):
        r, c = element_map[elem]
        ax.plot(z, cumulative[:, r, c], linewidth=1.5)
        ax.set_ylabel(f"{elem}  ({axis})")
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
        for cz in comp_z:
            ax.axvline(cz, color="grey", linewidth=0.5, linestyle=":")

    axes.flat[-1].set_xlabel("z (m)")

    # Component-name ticks along the top of the first panel
    if comp_z:
        ax_top = axes.flat[0].twiny()
        ax_top.set_xlim(axes.flat[0].get_xlim())
        ax_top.set_xticks(comp_z)
        ax_top.set_xticklabels(comp_names, rotation=45, ha="left", fontsize=8)

    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Publication-quality column ray diagrams
# ---------------------------------------------------------------------------

# Okabe-Ito-derived colours.  The public mapping makes it straightforward for
# notebooks to use exactly the same visual language in adjacent diagnostics.
NEOARM_RAY_COLORS = {
    "position": "#0072B2",       # blue
    "angle": "#D55E00",          # vermillion
    "envelope": "#56B4E9",       # sky blue
    "centroid": "#222222",
    "condenser_lens": "#0072B2",
    "objective_lens": "#6F4C9B",
    "intermediate_lens": "#E69F00",
    "projector_lens": "#009E73",
    "lens": "#4B5563",
    "sample": "#CC79A7",
    "aperture": "#666666",
    "detector": "#009E73",
    "deflector": "#D55E00",
    "corrector": "#6F4C9B",
    "corrector_off": "#9CA3AF",
    "plane": "#6B7280",
    "source": "#111827",
}


@dataclass(frozen=True)
class PositionAngleTrace:
    """A calibrated two-basis paraxial trace about a reference ray.

    ``position`` and ``angle`` are responses *about* ``centroid``.  For a
    Gaussian launched at a waist, their root-sum-square is the physical
    1/e-field (1/e^2-intensity) radius.  Storing the affine centroid
    separately keeps this relation correct in the presence of deflectors.
    ``voltage`` records the accelerating voltage used for the trace so that
    electromagnetic-lens rotation annotations remain calibrated.  All lengths
    are in metres and angles are in radians.
    """

    z: np.ndarray
    centroid: np.ndarray
    position: np.ndarray
    angle: np.ndarray
    radius: np.ndarray
    coordinate: str
    position_scale: float
    angle_scale: float
    label: str = "nominal"
    voltage: float = 200e3

    @property
    def position_path(self) -> np.ndarray:
        return np.asarray(self.centroid) + np.asarray(self.position)

    @property
    def angle_path(self) -> np.ndarray:
        return np.asarray(self.centroid) + np.asarray(self.angle)

    @property
    def lower_envelope(self) -> np.ndarray:
        return np.asarray(self.centroid) - np.asarray(self.radius)

    @property
    def upper_envelope(self) -> np.ndarray:
        return np.asarray(self.centroid) + np.asarray(self.radius)


@dataclass(frozen=True)
class AxialRayEvent:
    """A derived event on a paraxial trace."""

    kind: str
    z: float
    x: float
    radius: float | None
    segment_index: int


@dataclass(frozen=True)
class ColumnElement:
    """Display metadata for one active or passive column element."""

    z: float
    label: str
    role: str = "auto"
    z_end: float | None = None


@dataclass(frozen=True)
class ColumnSection:
    """A linearly scaled detail region in a column ray diagram.

    ``x_limit`` is a symmetric half-width in metres.  If omitted, the limit
    is derived from all traces visible in the section.
    """

    title: str
    z_min: float
    z_max: float
    x_limit: float | None = None


def make_position_angle_basis_rays(
    position_scale: float,
    angle_scale: float,
    *,
    z: float = 0.0,
    voltage: float = 200e3,
    x0: float = 0.0,
    y0: float = 0.0,
    axis: str = "x",
) -> Ray:
    """Construct calibrated position- and angle-basis rays.

    The first ray has a pure ``position_scale`` displacement and the second a
    pure ``angle_scale`` slope, both relative to ``(x0, y0)``.  This is the
    general calibrated counterpart of :func:`make_waist_divergence_rays`.
    """

    position_scale = float(position_scale)
    angle_scale = float(angle_scale)
    z = float(z)
    voltage = float(voltage)
    axis = str(axis).lower().strip()
    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'.")
    if not np.isfinite(position_scale) or position_scale == 0.0:
        raise ValueError("position_scale must be finite and non-zero.")
    if not np.isfinite(angle_scale) or angle_scale == 0.0:
        raise ValueError("angle_scale must be finite and non-zero.")
    if not np.isfinite(voltage) or voltage <= 0.0:
        raise ValueError("voltage must be finite and positive.")

    x = np.full(2, float(x0), dtype=float)
    y = np.full(2, float(y0), dtype=float)
    dx = np.zeros(2, dtype=float)
    dy = np.zeros(2, dtype=float)
    if axis == "x":
        x[0] += position_scale
        dx[1] = angle_scale
    else:
        y[0] += position_scale
        dy[1] = angle_scale

    return Ray(
        x=x,
        y=y,
        dx=dx,
        dy=dy,
        z=np.full(2, z, dtype=float),
        pathlength=np.zeros(2, dtype=float),
        voltage=np.full(2, voltage, dtype=float),
    )


def trace_ray_paths(
    components: Sequence[Component],
    rays: Ray,
    *,
    ray_coordinate: str = "x",
    include_input_rays: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Trace a ray bundle and return ``(z, transverse_position)`` arrays.

    The returned transverse array has shape ``(n_steps, n_rays)``.  The helper
    uses the same propagation and Larmor de-rotation path as :func:`plot_model`
    without coupling numerical tracing to a particular renderer.
    """

    mode = _normalize_ray_coordinate_mode(ray_coordinate)
    steps = tuple(run_iter_vmapped(rays, components))
    if include_input_rays:
        steps = (rays,) + steps
    rotation = (
        _compute_cumulative_rotation(
            components,
            voltage=_plotting_voltage_from_rays(rays),
            include_input_rays=include_input_rays,
        )
        if mode in ("x_rot", "y_rot")
        else None
    )
    X, Z = _stack_ray_positions(
        steps,
        ray_coordinate=mode,
        cumulative_rotation=rotation,
    )
    return Z, X


def trace_ray_envelope(components, rays, *, ray_coordinate="x_rot"):
    """Return ``(z, lower/upper bounds)`` segments of a geometric ray bundle.

    Trace native components, resolving solenoid bodies with 16 slices and
    free-space crossovers with 65 samples. At a sigmoid aperture, rays outside
    its nominal circular radius stop. This diagram is a geometric illustration;
    it does not represent the smooth transmission or diffracted wave intensity.
    Separate segments preserve aperture clipping and scanner position jumps.
    """
    mode = _normalize_ray_coordinate_mode(ray_coordinate)
    rotation, segments = 0.0, []
    voltage = _plotting_voltage_from_rays(rays)

    def coordinate(ray, angle):
        x, y = np.broadcast_arrays(np.asarray(ray.x), np.asarray(ray.y))
        if mode == "x_rot":
            return x * np.cos(angle) + y * np.sin(angle)
        if mode == "y_rot":
            return -x * np.sin(angle) + y * np.cos(angle)
        return {"x": x, "y": y, "r": np.hypot(x, y)}[mode]

    for component in components:
        pieces = (component,)
        if isinstance(component, DoubleDeflector):
            pieces = component.deflectors
        elif isinstance(component, HardEdgeSolenoid):
            pieces = tuple(
                replace(component, z=component.z + i * component.length / 16,
                        length=component.length / 16, q=component.q / 16)
                for i in range(16)
            )
        for piece in pieces:
            entrance, exit_ray = run_iter_vmapped(rays, (piece,))
            delta = _compute_cumulative_rotation(
                (piece,), voltage=voltage, include_input_rays=False
            )[-1]
            for end, angle in ((entrance, rotation), (exit_ray, rotation + delta)):
                start_x, end_x = coordinate(rays, rotation), coordinate(end, angle)
                t = np.linspace(0, 1, 65)[:, None]
                x = (1 - t) * start_x + t * end_x
                z = np.linspace(float(np.mean(rays.z)), float(np.mean(end.z)), len(t))
                segments.append((z, np.column_stack((x.min(axis=1), x.max(axis=1)))))
                rays, rotation = end, angle
            if isinstance(piece, SigmoidAperture):
                keep = np.hypot(np.asarray(rays.x) - piece.x0,
                                np.asarray(rays.y) - piece.y0) <= piece.radius
                if not np.any(keep):
                    return segments
                rays = rays[np.flatnonzero(keep)]
    return segments


def _reference_ray(
    *,
    z: float,
    voltage: float,
    x0: float,
    y0: float,
) -> Ray:
    return Ray(
        x=np.asarray([x0], dtype=float),
        y=np.asarray([y0], dtype=float),
        dx=np.asarray([0.0], dtype=float),
        dy=np.asarray([0.0], dtype=float),
        z=np.asarray([z], dtype=float),
        pathlength=np.asarray([0.0], dtype=float),
        voltage=np.asarray([voltage], dtype=float),
    )


def _trace_basis_and_reference(
    components: Sequence[Component],
    basis_rays: Ray,
    reference_ray: Ray,
    *,
    ray_coordinate: str,
    position_scale: float,
    angle_scale: float,
    label: str,
) -> PositionAngleTrace:
    def _joined_field(name: str) -> np.ndarray:
        return np.concatenate(
            (
                np.ravel(np.asarray(getattr(basis_rays, name), dtype=float)),
                np.ravel(np.asarray(getattr(reference_ray, name), dtype=float)),
            )
        )

    # Trace the two basis rays and affine reference in one batch.  Apart from
    # being faster for wobble sweeps, this guarantees identical axial samples.
    ray_bundle = Ray(
        x=_joined_field("x"),
        y=_joined_field("y"),
        dx=_joined_field("dx"),
        dy=_joined_field("dy"),
        z=_joined_field("z"),
        pathlength=_joined_field("pathlength"),
        voltage=_joined_field("voltage"),
    )
    z_basis, X_bundle = trace_ray_paths(
        components,
        ray_bundle,
        ray_coordinate=ray_coordinate,
        include_input_rays=True,
    )
    if X_bundle.ndim != 2 or X_bundle.shape[1] != 3:
        raise ValueError("Expected two basis rays and one reference ray.")

    centroid = np.asarray(X_bundle[:, 2], dtype=float)
    position = np.asarray(X_bundle[:, 0], dtype=float) - centroid
    angle = np.asarray(X_bundle[:, 1], dtype=float) - centroid
    radius = gaussian_basis_envelope(position, angle)
    bundle_voltages = np.ravel(np.asarray(ray_bundle.voltage, dtype=float))
    if bundle_voltages.size == 0 or not np.all(np.isfinite(bundle_voltages)):
        raise ValueError("Basis-ray voltage must be finite.")
    if not np.allclose(bundle_voltages, bundle_voltages[0], rtol=1e-12, atol=0.0):
        raise ValueError("All calibrated basis and reference rays must share one voltage.")
    return PositionAngleTrace(
        z=np.asarray(z_basis, dtype=float),
        centroid=centroid,
        position=position,
        angle=angle,
        radius=radius,
        coordinate=_normalize_ray_coordinate_mode(ray_coordinate),
        position_scale=float(position_scale),
        angle_scale=float(angle_scale),
        label=str(label),
        voltage=float(bundle_voltages[0]),
    )


def trace_position_angle_basis(
    components: Sequence[Component],
    position_scale: float,
    angle_scale: float,
    *,
    z: float = 0.0,
    voltage: float = 200e3,
    x0: float = 0.0,
    y0: float = 0.0,
    axis: str = "x",
    ray_coordinate: str | None = None,
    label: str = "nominal",
) -> PositionAngleTrace:
    """Trace a calibrated position/angle basis and its affine centroid."""

    basis = make_position_angle_basis_rays(
        position_scale,
        angle_scale,
        z=z,
        voltage=voltage,
        x0=x0,
        y0=y0,
        axis=axis,
    )
    reference = _reference_ray(
        z=float(z), voltage=float(voltage), x0=float(x0), y0=float(y0)
    )
    coordinate = axis if ray_coordinate is None else ray_coordinate
    coordinate_norm = _normalize_ray_coordinate_mode(coordinate)
    if coordinate_norm == "r":
        raise ValueError(
            "A signed linear basis cannot be recovered from ray_coordinate='r'; "
            "use x/y or x_rot/y_rot."
        )
    return _trace_basis_and_reference(
        components,
        basis,
        reference,
        ray_coordinate=coordinate_norm,
        position_scale=position_scale,
        angle_scale=angle_scale,
        label=label,
    )


def trace_gaussian_basis(
    components: Sequence[Component],
    waist: float,
    *,
    voltage: float | None = None,
    wavelength: float | None = None,
    z: float = 0.0,
    x0: float = 0.0,
    y0: float = 0.0,
    ray_coordinate: str = "x",
    label: str = "nominal",
) -> PositionAngleTrace:
    """Trace the two real basis rays associated with an input Gaussian waist.

    Pass ``voltage`` for an electromagnetic column.  Supplying only a
    wavelength retains the historical 200 kV voltage carried by
    :func:`make_waist_divergence_rays`, which is appropriate only for
    voltage-independent components or an actual 200 kV column.
    """

    coordinate = _normalize_ray_coordinate_mode(ray_coordinate)
    if coordinate == "r":
        raise ValueError(
            "A signed Gaussian basis cannot be recovered from "
            "ray_coordinate='r'; use x/y or x_rot/y_rot."
        )
    basis = make_waist_divergence_rays(
        waist,
        voltage=voltage,
        wavelength=wavelength,
        z=z,
        x0=x0,
        y0=y0,
    )
    basis_voltage = float(np.ravel(np.asarray(basis.voltage, dtype=float))[0])
    theta = float(np.ravel(np.asarray(basis.dx, dtype=float))[1])
    if coordinate in ("y", "y_rot"):
        # The historical constructor is x-oriented.  Retain it as the source
        # of the wavelength-calibrated divergence, then rotate that basis into
        # y for the corresponding signed coordinate.
        basis = make_position_angle_basis_rays(
            waist,
            theta,
            z=z,
            voltage=basis_voltage,
            x0=x0,
            y0=y0,
            axis="y",
        )
    reference = _reference_ray(
        z=float(z), voltage=basis_voltage, x0=float(x0), y0=float(y0)
    )
    return _trace_basis_and_reference(
        components,
        basis,
        reference,
        ray_coordinate=coordinate,
        position_scale=float(waist),
        angle_scale=theta,
        label=label,
    )


def gaussian_basis_envelope(
    position_response: np.ndarray,
    angle_response: np.ndarray,
) -> np.ndarray:
    """Return the Gaussian radius from its orthogonal real ray responses."""

    position = np.asarray(position_response, dtype=float)
    angle = np.asarray(angle_response, dtype=float)
    if position.shape != angle.shape:
        raise ValueError("position_response and angle_response must have the same shape.")
    return np.hypot(position, angle)


def _event_tolerance(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)[np.isfinite(values)]
    # A response may legitimately be picometres, so an absolute floor of one
    # metre would classify the complete trace as zero.  Keep this tolerance
    # relative to the response magnitude instead.
    scale = float(np.max(np.abs(finite))) if finite.size else 1.0
    scale = max(scale, np.finfo(float).tiny)
    return 64.0 * np.finfo(float).eps * scale


def _axial_tolerance(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)[np.isfinite(values)]
    scale = max(1.0, float(np.max(np.abs(finite))) if finite.size else 1.0)
    return 64.0 * np.finfo(float).eps * scale


def _deduplicate_events(events: Sequence[AxialRayEvent]) -> tuple[AxialRayEvent, ...]:
    ordered = sorted(events, key=lambda event: (event.z, event.segment_index))
    if not ordered:
        return ()
    z_scale = max(1.0, max(abs(event.z) for event in ordered))
    z_tol = 128.0 * np.finfo(float).eps * z_scale
    unique = [ordered[0]]
    for event in ordered[1:]:
        if abs(event.z - unique[-1].z) <= z_tol and event.kind == unique[-1].kind:
            # Prefer an event with a radius payload (waists) and otherwise the
            # first deterministic segment occurrence.
            if unique[-1].radius is None and event.radius is not None:
                unique[-1] = event
            continue
        unique.append(event)
    return tuple(unique)


def find_ray_crossings(
    z: np.ndarray,
    first: np.ndarray,
    second: np.ndarray | None = None,
    *,
    kind: str = "crossover",
    include_endpoints: bool = False,
) -> tuple[AxialRayEvent, ...]:
    """Find zeros or pairwise intersections on piecewise-linear ray paths.

    Equal-z component-application jumps are never interpolated, preventing a
    thin lens or lab-frame Larmor rotation from creating a false crossing.
    """

    z_arr = np.ravel(np.asarray(z, dtype=float))
    first_arr = np.ravel(np.asarray(first, dtype=float))
    if z_arr.shape != first_arr.shape:
        raise ValueError("z and first must have the same one-dimensional shape.")
    if second is None:
        second_arr = None
        delta = first_arr
    else:
        second_arr = np.ravel(np.asarray(second, dtype=float))
        if second_arr.shape != first_arr.shape:
            raise ValueError("first and second must have the same shape.")
        delta = first_arr - second_arr
    if z_arr.size < 2:
        return ()

    tol = _event_tolerance(delta)
    finite_delta = delta[np.isfinite(delta)]
    if finite_delta.size and np.all(np.abs(finite_delta) <= tol):
        # Coincident paths (or a response that vanishes everywhere) do not
        # define an isolated axial event.
        return ()
    events: list[AxialRayEvent] = []

    # Exact sampled roots.  Exclude only the global input/output when asked;
    # roots at component planes are physically meaningful and are retained.
    for idx, value in enumerate(delta):
        if not np.isfinite(value) or abs(value) > tol:
            continue
        if not include_endpoints and idx in (0, z_arr.size - 1):
            continue
        x_value = 0.0 if second_arr is None else float(
            0.5 * (first_arr[idx] + second_arr[idx])
        )
        events.append(
            AxialRayEvent(str(kind), float(z_arr[idx]), x_value, None, max(0, idx - 1))
        )

    for idx in range(z_arr.size - 1):
        z0, z1 = z_arr[idx], z_arr[idx + 1]
        v0, v1 = delta[idx], delta[idx + 1]
        if not np.all(np.isfinite([z0, z1, v0, v1])) or z1 == z0:
            continue
        # Exact endpoint roots were handled above; only strict sign changes
        # need interpolation here.
        if not ((v0 < -tol and v1 > tol) or (v0 > tol and v1 < -tol)):
            continue
        t = -v0 / (v1 - v0)
        z_cross = z0 + t * (z1 - z0)
        if second_arr is None:
            x_cross = 0.0
        else:
            x_cross = first_arr[idx] + t * (first_arr[idx + 1] - first_arr[idx])
        events.append(
            AxialRayEvent(str(kind), float(z_cross), float(x_cross), None, idx)
        )

    return _deduplicate_events(events)


def find_gaussian_waists(
    z: np.ndarray,
    position_response: np.ndarray,
    angle_response: np.ndarray,
    *,
    centroid: np.ndarray | None = None,
    include_endpoints: bool = False,
) -> tuple[AxialRayEvent, ...]:
    """Find local minima of the Gaussian basis radius along linear drifts."""

    z_arr = np.ravel(np.asarray(z, dtype=float))
    pos = np.ravel(np.asarray(position_response, dtype=float))
    ang = np.ravel(np.asarray(angle_response, dtype=float))
    if z_arr.shape != pos.shape or pos.shape != ang.shape:
        raise ValueError("z, position_response, and angle_response must share shape.")
    if centroid is None:
        centre = np.zeros_like(z_arr)
    else:
        centre = np.ravel(np.asarray(centroid, dtype=float))
        if centre.shape != z_arr.shape:
            raise ValueError("centroid must have the same shape as z.")
    if z_arr.size == 0:
        return ()

    events: list[AxialRayEvent] = []
    radius = np.hypot(pos, ang)
    response_scale = max(
        float(np.max(np.abs(pos))) if pos.size else 0.0,
        float(np.max(np.abs(ang))) if ang.size else 0.0,
        np.finfo(float).tiny,
    )
    denom_tol = 64.0 * np.finfo(float).eps * response_scale**2

    # Analytic minimum of ||v0 + t*dv|| on each non-zero-length drift.
    for idx in range(z_arr.size - 1):
        z0, z1 = z_arr[idx], z_arr[idx + 1]
        if not np.all(np.isfinite([z0, z1, pos[idx], pos[idx + 1], ang[idx], ang[idx + 1]])):
            continue
        if z1 == z0:
            continue
        dp = pos[idx + 1] - pos[idx]
        da = ang[idx + 1] - ang[idx]
        denom = dp * dp + da * da
        if denom <= denom_tol:
            continue
        t = -(pos[idx] * dp + ang[idx] * da) / denom
        if 0.0 < t < 1.0:
            p_star = pos[idx] + t * dp
            a_star = ang[idx] + t * da
            c_star = centre[idx] + t * (centre[idx + 1] - centre[idx])
            events.append(
                AxialRayEvent(
                    "waist",
                    float(z0 + t * (z1 - z0)),
                    float(c_star),
                    float(np.hypot(p_star, a_star)),
                    idx,
                )
            )

    # A thin component can put a minimum exactly at its plane.  Compare each
    # unique-z sample with its nearest samples at different z.
    unique_indices: list[int] = []
    for idx, z_value in enumerate(z_arr):
        if unique_indices and z_value == z_arr[unique_indices[-1]]:
            unique_indices[-1] = idx
        else:
            unique_indices.append(idx)
    for order_idx, idx in enumerate(unique_indices):
        if order_idx in (0, len(unique_indices) - 1) and not include_endpoints:
            continue
        left = unique_indices[order_idx - 1] if order_idx > 0 else None
        right = unique_indices[order_idx + 1] if order_idx + 1 < len(unique_indices) else None
        left_ok = left is None or radius[idx] <= radius[left]
        right_ok = right is None or radius[idx] <= radius[right]
        strict = (
            (left is not None and radius[idx] < radius[left])
            or (right is not None and radius[idx] < radius[right])
        )
        if left_ok and right_ok and strict:
            events.append(
                AxialRayEvent(
                    "waist",
                    float(z_arr[idx]),
                    float(centre[idx]),
                    float(radius[idx]),
                    max(0, idx - 1),
                )
            )

    return _deduplicate_events(events)


def _interpolate_trace_value(z: np.ndarray, values: np.ndarray, at_z: float) -> float:
    z_arr = np.asarray(z, dtype=float)
    values_arr = np.asarray(values, dtype=float)
    exact = np.flatnonzero(np.isclose(z_arr, at_z, rtol=0.0, atol=_axial_tolerance(z_arr)))
    if exact.size:
        return float(values_arr[exact[-1]])
    for idx in range(z_arr.size - 1):
        z0, z1 = z_arr[idx], z_arr[idx + 1]
        if z1 == z0 or not (min(z0, z1) <= at_z <= max(z0, z1)):
            continue
        t = (at_z - z0) / (z1 - z0)
        return float(values_arr[idx] + t * (values_arr[idx + 1] - values_arr[idx]))
    return float("nan")


def basis_trace_events(
    trace: PositionAngleTrace,
) -> dict[str, tuple[AxialRayEvent, ...]]:
    """Return A-zero crossovers, B-zero conjugates, and Gaussian waists."""

    z = np.asarray(trace.z, dtype=float)
    z0 = float(z[0]) if z.size else 0.0

    def _without_input(events: tuple[AxialRayEvent, ...]) -> tuple[AxialRayEvent, ...]:
        tol = _axial_tolerance(z)
        return tuple(event for event in events if abs(event.z - z0) > tol)

    crossovers = _without_input(
        find_ray_crossings(
            z,
            trace.position,
            kind="crossover",
            include_endpoints=True,
        )
    )
    conjugates = _without_input(
        find_ray_crossings(
            z,
            trace.angle,
            kind="conjugate",
            include_endpoints=True,
        )
    )
    waists = _without_input(
        find_gaussian_waists(
            z,
            trace.position,
            trace.angle,
            centroid=trace.centroid,
            include_endpoints=True,
        )
    )

    def _put_on_centroid(events: tuple[AxialRayEvent, ...]) -> tuple[AxialRayEvent, ...]:
        return tuple(
            replace(
                event,
                x=_interpolate_trace_value(z, trace.centroid, event.z),
            )
            for event in events
        )

    return {
        "crossovers": _put_on_centroid(crossovers),
        "conjugates": _put_on_centroid(conjugates),
        "waists": waists,
    }


def _infer_element_role(component: object, label: str) -> str:
    text = str(label).lower().replace("-", "_").replace(" ", "_")
    if "corrector" in text:
        return "corrector_off" if "off" in text else "corrector"
    if isinstance(component, Detector) or "detector" in text or "screen" in text:
        return "detector"
    if isinstance(component, (DoubleDeflector, Deflector)) or "deflector" in text:
        return "deflector"
    if (
        isinstance(component, (MagneticPhaseSample, RandomPhaseSample, InterpolatedSample2D))
        or "sample" in text
        or "specimen" in text
        or "grating" in text
    ):
        return "sample"
    if isinstance(component, SigmoidAperture) or "aperture" in text:
        return "aperture"
    # Instrument-native names such as CL1, C Mini, OL, IL1, and PL1 carry
    # useful semantics even when a notebook represents a fixed lens plane by
    # a passive ``Plane`` or adds it through ``extra_elements``.
    if (
        text.startswith("cl")
        or text in ("cm", "cmin")
        or "c_mini" in text
        or "cmini" in text
        or "condenser" in text
    ):
        return "condenser_lens"
    if text.startswith("ol") or "obj" in text or "objective" in text:
        return "objective_lens"
    if text.startswith("il") or "intermediate" in text:
        return "intermediate_lens"
    if text.startswith("pl") or "projector" in text:
        return "projector_lens"
    if isinstance(component, (Lens, ElectromagneticLens, ThickLens)) or "lens" in text:
        return "lens"
    if "source" in text or "gun" in text:
        return "source"
    if isinstance(component, Plane):
        return "plane"
    return "plane"


def column_elements_from_components(
    components: Sequence[Component],
    *,
    labels: Sequence[str] | None = None,
    roles: Sequence[str] | None = None,
    extra_elements: Sequence[ColumnElement] = (),
) -> tuple[ColumnElement, ...]:
    """Attach stable names and semantic roles to simulation components."""

    components = tuple(components)
    if labels is not None and len(labels) != len(components):
        raise ValueError("labels must have one entry per component.")
    if roles is not None and len(roles) != len(components):
        raise ValueError("roles must have one entry per component.")

    elements: list[ColumnElement] = []
    for idx, component in enumerate(components):
        z_value = float(component.z)
        label = str(labels[idx]) if labels is not None else _as_name(component)
        role = str(roles[idx]) if roles is not None else _infer_element_role(component, label)
        z_end: float | None = None
        if isinstance(component, DoubleDeflector):
            z_end = float(component.z_second)
        elif isinstance(component, ElectromagneticLens):
            thickness = float(np.asarray(component.thickness))
            if np.isfinite(thickness) and thickness > 0.0:
                z_end = z_value + thickness
        elif isinstance(component, ThickLens):
            z_other = float(component.z_pi)
            z_value, z_end = min(z_value, z_other), max(z_value, z_other)
        elements.append(ColumnElement(z=z_value, label=label, role=role, z_end=z_end))

    elements.extend(extra_elements)
    normalized = [
        ColumnElement(
            z=float(element.z),
            label=str(element.label),
            role=(
                _infer_element_role(object(), element.label)
                if str(element.role).lower() == "auto"
                else str(element.role)
            ),
            z_end=None if element.z_end is None else float(element.z_end),
        )
        for element in elements
    ]
    return tuple(sorted(normalized, key=lambda element: (element.z, element.label)))


def _find_element_z(elements: Sequence[ColumnElement], *patterns: str) -> float | None:
    patterns_lower = tuple(pattern.lower() for pattern in patterns)
    for element in elements:
        label = element.label.lower().replace(" ", "_")
        if any(pattern in label for pattern in patterns_lower):
            return float(element.z)
    return None


def infer_neoarm_sections(
    elements: Sequence[ColumnElement],
    z_min: float,
    z_max: float,
) -> tuple[ColumnSection, ColumnSection, ColumnSection]:
    """Infer three honest linear-z detail regions for a NEOARM-like column."""

    z_min, z_max = float(z_min), float(z_max)
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_max <= z_min:
        raise ValueError("z_max must be finite and greater than z_min.")
    span = z_max - z_min
    aperture_z = _find_element_z(elements, "c_aperture", "condenser_aperture")
    sample_z = _find_element_z(elements, "sample", "specimen", "grating")
    obj_post_z = _find_element_z(elements, "obj_post", "objective_post")
    il1_z = _find_element_z(elements, "il1", "intermediate_1")

    if aperture_z is None or not z_min < aperture_z < z_max:
        aperture_z = z_min + span / 3.0
    if sample_z is None or not aperture_z < sample_z < z_max:
        sample_z = aperture_z + 0.45 * (z_max - aperture_z)

    if obj_post_z is not None and il1_z is not None and obj_post_z < il1_z:
        objective_end = 0.5 * (obj_post_z + il1_z)
    else:
        objective_end = min(z_max, sample_z + 0.12 * span)
    objective_end = max(objective_end, sample_z + 0.01 * span)
    objective_end = min(objective_end, z_max)

    return (
        ColumnSection("Illumination: source to condenser aperture", z_min, aperture_z),
        ColumnSection("Corrector and objective region", aperture_z, objective_end),
        ColumnSection("Imaging: sample to detector", sample_z, z_max),
    )


def cumulative_larmor_profile(
    components: Sequence[Component],
    *,
    voltage: float = 200e3,
    z_min: float | None = None,
    z_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an axial cumulative Larmor-rotation profile in radians.

    Finite-thickness :class:`ElectromagneticLens` rotations are displayed as a
    linear accumulation between their entrance and exit planes.  Thin lenses
    produce an explicit equal-z step.  This is a display interpolation of the
    component's calibrated net rotation; the traced rays remain authoritative.
    """

    components = tuple(components)
    voltage = float(voltage)
    if not np.isfinite(voltage) or voltage <= 0.0:
        raise ValueError("voltage must be finite and positive.")

    records: list[tuple[float, float, float]] = []
    component_positions: list[float] = []
    for component in components:
        if not hasattr(component, "z"):
            continue
        entry = float(component.z)
        component_positions.append(entry)
        if not isinstance(component, ElectromagneticLens):
            continue
        thickness = max(0.0, float(np.asarray(component.thickness)))
        exit_z = entry + thickness
        rotation = float(np.asarray(component.rotation_angle(voltage)))
        if not np.all(np.isfinite([entry, exit_z, rotation])):
            raise ValueError("Electromagnetic-lens positions and rotations must be finite.")
        component_positions.append(exit_z)
        records.append((entry, exit_z, rotation))

    if z_min is None:
        if not component_positions:
            raise ValueError("z_min is required when components have no axial positions.")
        z_min = min(component_positions)
    if z_max is None:
        if not component_positions:
            raise ValueError("z_max is required when components have no axial positions.")
        z_max = max(component_positions)
    z_min, z_max = float(z_min), float(z_max)
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_max <= z_min:
        raise ValueError("z_max must be finite and greater than z_min.")

    breakpoints = {z_min, z_max}
    for entry, exit_z, _ in records:
        if z_min <= entry <= z_max:
            breakpoints.add(entry)
        if z_min <= exit_z <= z_max:
            breakpoints.add(exit_z)

    profile_z: list[float] = []
    profile_theta: list[float] = []
    axial_tol = _axial_tolerance(np.asarray([z_min, z_max], dtype=float))
    for z_value in sorted(breakpoints):
        theta_before = 0.0
        thin_step = 0.0
        for entry, exit_z, rotation in records:
            if exit_z > entry:
                fraction = np.clip((z_value - entry) / (exit_z - entry), 0.0, 1.0)
                theta_before += rotation * float(fraction)
            elif z_value > entry + axial_tol:
                theta_before += rotation
            elif abs(z_value - entry) <= axial_tol:
                thin_step += rotation
        profile_z.append(z_value)
        profile_theta.append(theta_before)
        if abs(thin_step) > 0.0:
            profile_z.append(z_value)
            profile_theta.append(theta_before + thin_step)

    return np.asarray(profile_z, dtype=float), np.asarray(profile_theta, dtype=float)


def _length_display_unit(max_abs_m: float) -> tuple[float, str]:
    value = abs(float(max_abs_m))
    if value < 1e-9:
        return 1e12, "pm"
    if value < 1e-6:
        return 1e9, "nm"
    if value < 1e-3:
        return 1e6, "µm"
    if value < 1.0:
        return 1e3, "mm"
    return 1.0, "m"


def _format_engineering_length(value_m: float) -> str:
    value = float(value_m)
    if not np.isfinite(value):
        return "not finite"
    if value == 0.0:
        return "0 m"
    factor, unit = _length_display_unit(abs(value))
    return f"{value * factor:.3g} {unit}"


def _trace_values_in_section(
    trace: PositionAngleTrace,
    z_min: float,
    z_max: float,
) -> np.ndarray:
    z = np.asarray(trace.z, dtype=float)
    mask = (z >= z_min) & (z <= z_max)
    if not np.any(mask):
        return np.empty((0,), dtype=float)
    return np.concatenate(
        (
            trace.position_path[mask],
            trace.angle_path[mask],
            trace.lower_envelope[mask],
            trace.upper_envelope[mask],
        )
    )


def _nice_symmetric_limit(value: float) -> float:
    value = max(abs(float(value)), np.finfo(float).eps)
    exponent = np.floor(np.log10(value))
    fraction = value / (10.0**exponent)
    nice_fraction = next(candidate for candidate in (1.0, 1.5, 2.0, 3.0, 5.0, 10.0) if fraction <= candidate)
    return float(nice_fraction * 10.0**exponent)


def _section_scale(
    traces: Sequence[PositionAngleTrace],
    section: ColumnSection,
) -> tuple[float, float, str]:
    if section.x_limit is not None:
        max_abs = abs(float(section.x_limit))
        if not np.isfinite(max_abs) or max_abs <= 0.0:
            raise ValueError("ColumnSection.x_limit must be finite and positive.")
    else:
        values = [
            _trace_values_in_section(trace, section.z_min, section.z_max)
            for trace in traces
        ]
        finite = np.concatenate([value for value in values if value.size]) if any(value.size for value in values) else np.asarray([1.0])
        finite = finite[np.isfinite(finite)]
        max_abs = float(np.max(np.abs(finite))) if finite.size else 1.0
    limit_m = _nice_symmetric_limit(1.08 * max_abs)
    factor, unit = _length_display_unit(limit_m)
    return limit_m, factor, unit


def _spread_label_positions(
    values: Sequence[float],
    z_min: float,
    z_max: float,
    *,
    minimum_fraction: float = 0.035,
) -> np.ndarray:
    if not values:
        return np.empty((0,), dtype=float)
    order = np.argsort(np.asarray(values, dtype=float))
    sorted_values = np.asarray(values, dtype=float)[order]
    min_sep = max(0.0, minimum_fraction * (z_max - z_min))
    placed = sorted_values.copy()
    for idx in range(1, placed.size):
        placed[idx] = max(placed[idx], placed[idx - 1] + min_sep)
    if placed[-1] > z_max:
        placed -= placed[-1] - z_max
        for idx in range(placed.size - 2, -1, -1):
            placed[idx] = min(placed[idx], placed[idx + 1] - min_sep)
    if placed[0] < z_min:
        placed += z_min - placed[0]
    out = np.empty_like(placed)
    out[order] = placed
    return out


def _spread_label_boxes(
    values: Sequence[float],
    line_counts: Sequence[int],
    z_min: float,
    z_max: float,
    *,
    line_height_fraction: float,
    gap_fraction: float,
) -> np.ndarray:
    """Place variable-height text boxes without axial overlap."""

    if not values:
        return np.empty((0,), dtype=float)
    values_array = np.asarray(values, dtype=float)
    lines_array = np.maximum(1, np.asarray(line_counts, dtype=float))
    if values_array.shape != lines_array.shape:
        raise ValueError("values and line_counts must have the same shape.")
    span = max(float(z_max - z_min), np.finfo(float).eps)
    order = np.argsort(values_array)
    centres = values_array[order].copy()
    half_heights = 0.5 * lines_array[order] * line_height_fraction * span
    gap = gap_fraction * span

    total_height = float(2.0 * np.sum(half_heights) + gap * (centres.size - 1))
    available = span
    if total_height > available:
        compression = available / total_height
        half_heights *= compression
        gap *= compression

    # Compute the latest feasible centre for every box by packing backwards
    # from the lower endpoint.  A forward placement can then retain each
    # requested anchor whenever possible while reserving guaranteed room for
    # every later label.  This matters for a microscope column: widely spaced
    # CL1/CL3 labels followed by a dense C-mini/corrector/objective stack would
    # defeat a simple global overflow shift and put the final labels outside
    # the panel even though ample space existed above the stack.
    earliest_first = z_min + half_heights[0]
    latest = np.empty_like(centres)
    latest[-1] = z_max - half_heights[-1]
    for idx in range(centres.size - 2, -1, -1):
        separation = half_heights[idx] + gap + half_heights[idx + 1]
        latest[idx] = latest[idx + 1] - separation

    centres[0] = np.clip(centres[0], earliest_first, latest[0])
    for idx in range(1, centres.size):
        separation = half_heights[idx - 1] + gap + half_heights[idx]
        earliest = centres[idx - 1] + separation
        centres[idx] = np.clip(centres[idx], earliest, latest[idx])

    output = np.empty_like(centres)
    output[order] = centres
    return output


def _wrapped_label(label: str, max_characters: int) -> str:
    if max_characters <= 0:
        return str(label)
    return "\n".join(
        textwrap.wrap(
            str(label),
            width=max_characters,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def _element_mid_z(element: ColumnElement) -> float:
    return float(element.z) if element.z_end is None else 0.5 * (float(element.z) + float(element.z_end))


def _draw_column_element(
    ax: mpl.axes.Axes,
    element: ColumnElement,
    *,
    z_span: float,
) -> None:
    """Draw a fixed-width semantic glyph using axes-x/data-z coordinates."""

    transform = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    role = element.role if element.role in NEOARM_RAY_COLORS else "plane"
    color = NEOARM_RAY_COLORS[role]
    z_mid = _element_mid_z(element)
    half_height = max(0.004 * z_span, np.finfo(float).eps)

    if role.endswith("lens") or role == "lens":
        if element.z_end is not None and element.z_end != element.z:
            z0, z1 = sorted((element.z, element.z_end))
            ax.add_patch(
                mpl.patches.Rectangle(
                    (0.435, z0),
                    0.13,
                    z1 - z0,
                    transform=transform,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=1.0,
                    alpha=0.10,
                    zorder=4,
                )
            )
        left = mpl.path.Path(
            [(0.435, z_mid - half_height), (0.49, z_mid), (0.435, z_mid + half_height)],
            [mpl.path.Path.MOVETO, mpl.path.Path.CURVE3, mpl.path.Path.CURVE3],
        )
        right = mpl.path.Path(
            [(0.565, z_mid - half_height), (0.51, z_mid), (0.565, z_mid + half_height)],
            [mpl.path.Path.MOVETO, mpl.path.Path.CURVE3, mpl.path.Path.CURVE3],
        )
        for path in (left, right):
            ax.add_patch(
                mpl.patches.PathPatch(
                    path,
                    transform=transform,
                    fill=False,
                    edgecolor=color,
                    linewidth=2.0,
                    zorder=7,
                )
            )
    elif role == "aperture":
        ax.plot([0.42, 0.485], [z_mid, z_mid], transform=transform, color=color, lw=4.5, zorder=7)
        ax.plot([0.515, 0.58], [z_mid, z_mid], transform=transform, color=color, lw=4.5, zorder=7)
    elif role == "sample":
        ax.plot([0.42, 0.58], [z_mid, z_mid], transform=transform, color=color, lw=2.4, zorder=7)
        for x_value in np.linspace(0.435, 0.565, 7):
            ax.plot(
                [x_value - 0.008, x_value + 0.008],
                [z_mid - 0.65 * half_height, z_mid + 0.65 * half_height],
                transform=transform,
                color=color,
                lw=0.8,
                zorder=7,
            )
    elif role == "detector":
        ax.add_patch(
            mpl.patches.Rectangle(
                (0.42, z_mid - half_height),
                0.16,
                2.0 * half_height,
                transform=transform,
                facecolor=color,
                edgecolor="#1F2937",
                linewidth=1.0,
                alpha=0.65,
                zorder=7,
            )
        )
        for x_value in np.linspace(0.44, 0.56, 5):
            ax.plot(
                [x_value, x_value],
                [z_mid - half_height, z_mid + half_height],
                transform=transform,
                color="white",
                lw=0.5,
                alpha=0.8,
                zorder=8,
            )
    elif role == "deflector":
        z_positions = [element.z]
        if element.z_end is not None:
            z_positions.append(element.z_end)
        for z_value in z_positions:
            ax.plot([0.43, 0.49], [z_value, z_value], transform=transform, color="#D55E00", lw=3.0, zorder=7)
            ax.plot([0.51, 0.57], [z_value, z_value], transform=transform, color="#0072B2", lw=3.0, zorder=7)
    elif role in ("corrector", "corrector_off"):
        z0 = element.z
        z1 = element.z_end if element.z_end is not None else element.z + 2.0 * half_height
        z0, z1 = sorted((z0, z1))
        ax.add_patch(
            mpl.patches.Rectangle(
                (0.43, z0),
                0.14,
                z1 - z0,
                transform=transform,
                facecolor="#F3F4F6",
                edgecolor=color,
                hatch="///" if role == "corrector_off" else None,
                linewidth=1.0,
                zorder=5,
            )
        )
    elif role == "source":
        ax.plot(
            [0.5],
            [z_mid],
            transform=transform,
            marker="*",
            markersize=8,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=8,
        )
    else:
        ax.plot([0.445, 0.555], [z_mid, z_mid], transform=transform, color=color, lw=1.2, zorder=6)


def _draw_elements(
    ax: mpl.axes.Axes,
    elements: Sequence[ColumnElement],
    *,
    z_min: float,
    z_max: float,
    label_fontsize: float,
    label_x: float = 0.72,
    max_label_characters: int = 28,
    label_line_height_fraction: float = 0.042,
    label_horizontal_alignment: str = "left",
) -> None:
    if label_horizontal_alignment not in {"left", "right"}:
        raise ValueError("label_horizontal_alignment must be 'left' or 'right'.")
    visible = [
        element
        for element in elements
        if (element.z_end if element.z_end is not None else element.z) >= z_min
        and element.z <= z_max
    ]
    z_span = max(z_max - z_min, np.finfo(float).eps)
    for element in visible:
        _draw_column_element(ax, element, z_span=z_span)

    actual_positions = [_element_mid_z(element) for element in visible]
    display_labels = []
    for element in visible:
        suffix = (
            " (off)"
            if element.role == "corrector_off" and "off" not in element.label.lower()
            else ""
        )
        display_labels.append(
            _wrapped_label(f"{element.label}{suffix}", max_label_characters)
        )
    line_counts = [label.count("\n") + 1 for label in display_labels]
    # ``label_line_height_fraction`` is a useful physical-z design target, but
    # it can be smaller than the rendered font in short detail panels.  Respect
    # the target while enforcing enough display-space for the glyphs and the
    # small white text-box pad.  This remains deterministic for raster and
    # vector backends because it depends only on figure geometry and points.
    axes_height_px = max(float(ax.get_window_extent().height), 1.0)
    minimum_line_height_px = (
        1.20 * float(label_fontsize) + 1.0
    ) * float(ax.figure.dpi) / 72.0
    effective_line_height_fraction = max(
        float(label_line_height_fraction),
        minimum_line_height_px / axes_height_px,
    )
    label_positions = _spread_label_boxes(
        actual_positions,
        line_counts,
        z_min,
        z_max,
        line_height_fraction=effective_line_height_fraction,
        gap_fraction=0.007,
    )
    transform = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    if label_horizontal_alignment == "right":
        # Keep leaders in the small neutral gutter between the fixed-width
        # semantic glyph and the label band.  Text is right-anchored separately
        # so even the final column of a small-multiple figure remains in-axis.
        leader_end_x = min(label_x - 0.012, 0.655)
    else:
        leader_end_x = label_x - 0.012
    for element, actual_z, label_z, display_label in zip(
        visible,
        actual_positions,
        label_positions,
        display_labels,
    ):
        ax.plot(
            [0.585, leader_end_x],
            [actual_z, label_z],
            transform=transform,
            color="#9CA3AF",
            linewidth=0.55,
            zorder=8,
        )
        ax.text(
            label_x,
            label_z,
            display_label,
            transform=transform,
            ha=label_horizontal_alignment,
            va="center",
            linespacing=1.05,
            fontsize=label_fontsize,
            color="#374151",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.5},
            clip_on=True,
            zorder=9,
        )


def _masked_trace_arrays(
    trace: PositionAngleTrace,
    z_min: float,
    z_max: float,
) -> tuple[np.ndarray, ...]:
    z = np.asarray(trace.z, dtype=float)
    mask = (z >= z_min) & (z <= z_max)
    return (
        z[mask],
        trace.position_path[mask],
        trace.angle_path[mask],
        trace.centroid[mask],
        trace.lower_envelope[mask],
        trace.upper_envelope[mask],
    )


def _plot_trace_layer(
    ax: mpl.axes.Axes,
    trace: PositionAngleTrace,
    *,
    z_min: float,
    z_max: float,
    factor: float,
    linestyle: str,
    alpha: float,
    linewidth: float,
    fill_envelope: bool,
    scenario_label: str,
    show_envelope: bool = True,
) -> None:
    z, position, angle, centroid, lower, upper = _masked_trace_arrays(trace, z_min, z_max)
    if z.size == 0:
        return
    if show_envelope and fill_envelope:
        ax.fill_betweenx(
            z,
            lower * factor,
            upper * factor,
            color=NEOARM_RAY_COLORS["envelope"],
            alpha=0.14,
            linewidth=0.0,
            zorder=1,
            label=f"{scenario_label}: Gaussian radius",
        )
    elif show_envelope:
        ax.plot(
            lower * factor,
            z,
            color=NEOARM_RAY_COLORS["envelope"],
            linestyle=linestyle,
            linewidth=max(0.8, linewidth - 0.3),
            alpha=0.65 * alpha,
            zorder=1,
        )
        ax.plot(
            upper * factor,
            z,
            color=NEOARM_RAY_COLORS["envelope"],
            linestyle=linestyle,
            linewidth=max(0.8, linewidth - 0.3),
            alpha=0.65 * alpha,
            zorder=1,
        )

    position_plot, z_position = _break_same_z_jumps(position[:, None] * factor, z)
    angle_plot, z_angle = _break_same_z_jumps(angle[:, None] * factor, z)
    centroid_plot, z_centroid = _break_same_z_jumps(centroid[:, None] * factor, z)
    ax.plot(
        position_plot[:, 0],
        z_position,
        color=NEOARM_RAY_COLORS["position"],
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        zorder=3,
        label=f"{scenario_label}: height basis",
    )
    ax.plot(
        angle_plot[:, 0],
        z_angle,
        color=NEOARM_RAY_COLORS["angle"],
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        zorder=3,
        label=f"{scenario_label}: angle basis",
    )
    ax.plot(
        centroid_plot[:, 0],
        z_centroid,
        color=NEOARM_RAY_COLORS["centroid"],
        linestyle=linestyle,
        linewidth=max(0.7, linewidth - 0.8),
        alpha=0.45 * alpha,
        zorder=2,
    )


def _annotate_trace_events(
    ax: mpl.axes.Axes,
    trace: PositionAngleTrace,
    *,
    z_min: float,
    z_max: float,
    factor: float,
    unit: str,
    max_per_kind: int = 4,
    include_waists: bool = True,
) -> None:
    _ = unit  # Event radii intentionally choose their own engineering unit.
    events = basis_trace_events(trace)
    event_styles = {
        "crossovers": ("A=0 crossover", "o", NEOARM_RAY_COLORS["position"]),
        "conjugates": ("B=0 conjugate", "s", NEOARM_RAY_COLORS["angle"]),
        "waists": ("Gaussian waist", "D", NEOARM_RAY_COLORS["objective_lens"]),
    }
    axial_span = z_max - z_min
    transverse_span = abs(float(np.diff(ax.get_xlim())[0]))
    candidates: list[tuple[str, AxialRayEvent]] = []
    for group_name, group_events in events.items():
        if group_name == "waists" and not include_waists:
            continue
        visible_all = [event for event in group_events if z_min <= event.z <= z_max]
        visible: list[AxialRayEvent] = []
        annotation_z_tolerance = 0.045 * axial_span
        for event in visible_all:
            same_visual_location = (
                visible
                and abs(event.z - visible[-1].z) < annotation_z_tolerance
                and abs((event.x - visible[-1].x) * factor)
                < 0.04 * transverse_span
            )
            if same_visual_location:
                # Event detection remains exact and complete; the figure only
                # collapses visually indistinguishable annotations.  For two
                # nearby waists, retain the tighter one.
                old_radius = visible[-1].radius
                if (
                    event.radius is not None
                    and old_radius is not None
                    and event.radius < old_radius
                ):
                    visible[-1] = event
                continue
            visible.append(event)
        candidates.extend((group_name, event) for event in visible[:max_per_kind])

    candidates.sort(key=lambda item: (item[1].z, item[0]))
    clusters: list[list[tuple[str, AxialRayEvent]]] = []
    for candidate in candidates:
        event = candidate[1]
        if clusters:
            previous_events = [item[1] for item in clusters[-1]]
            cluster_z = float(np.mean([item.z for item in previous_events]))
            cluster_x = float(np.mean([item.x for item in previous_events]))
            near_cluster = (
                abs(event.z - cluster_z) < 0.028 * axial_span
                and abs((event.x - cluster_x) * factor) < 0.045 * transverse_span
            )
        else:
            near_cluster = False
        if near_cluster:
            clusters[-1].append(candidate)
        else:
            clusters.append([candidate])

    cluster_labels: list[str] = []
    cluster_zs: list[float] = []
    for cluster in clusters:
        label_lines = []
        for group_name, event in cluster:
            text, _, _ = event_styles[group_name]
            if event.radius is None:
                label_lines.append(text)
            else:
                label_lines.append(
                    f"{text}, w={_format_engineering_length(event.radius)}"
                )
        # Events at a conjugate often coincide (or nearly coincide) with a
        # Gaussian waist.  Keep that information in one compact horizontal
        # callout; stacking one line per event makes the projector-panel
        # callouts collide even after their anchor positions are spread.
        cluster_labels.append(" · ".join(dict.fromkeys(label_lines)))
        cluster_zs.append(float(np.mean([event.z for _, event in cluster])))

    label_zs = _spread_label_boxes(
        cluster_zs,
        [label.count("\n") + 1 for label in cluster_labels],
        z_min,
        z_max,
        line_height_fraction=0.047,
        gap_fraction=0.012,
    )
    label_transform = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for cluster, label, label_z in zip(clusters, cluster_labels, label_zs):
        for group_name, event in cluster:
            _, marker, color = event_styles[group_name]
            x_value = event.x * factor
            ax.plot(
                [x_value],
                [event.z],
                marker=marker,
                markersize=4.5,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.0,
                linestyle="none",
                zorder=12,
            )
        anchor_x = float(np.mean([event.x for _, event in cluster])) * factor
        anchor_z = float(np.mean([event.z for _, event in cluster]))
        ax.annotate(
            label,
            xy=(anchor_x, anchor_z),
            xycoords="data",
            xytext=(0.405, label_z),
            textcoords=label_transform,
            ha="right",
            va="center",
            linespacing=1.06,
            fontsize=6.6,
            color="#374151",
            bbox={
                "facecolor": "white",
                "edgecolor": "#E5E7EB",
                "linewidth": 0.35,
                "alpha": 0.88,
                "pad": 0.7,
            },
            arrowprops={"arrowstyle": "-", "color": "#9CA3AF", "lw": 0.55},
            annotation_clip=True,
            zorder=13,
        )


def _configure_column_axis(
    ax: mpl.axes.Axes,
    *,
    section: ColumnSection,
    limit_m: float,
    factor: float,
    unit: str,
    coordinate: str,
    title: str,
) -> None:
    # Leave a small *physical-z* gutter beyond the declared section.  Endpoint
    # glyphs (notably the source and detector) and their text otherwise sit on
    # the clipping boundary even though their numerical z positions are
    # correct.  The scale remains linear and the same gutter is used for every
    # member of a comparison row.
    z_gutter = 0.018 * (section.z_max - section.z_min)
    ax.set_facecolor("white")
    ax.axvline(0.0, color="#CBD5E1", linewidth=0.8, zorder=0)
    ax.set_xlim(-limit_m * factor, limit_m * factor)
    ax.set_ylim(section.z_max + z_gutter, section.z_min - z_gutter)
    ax.set_yscale("linear")
    ax.set_xlabel(f"{coordinate} [{unit}]")
    ax.set_ylabel("z [m]")
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", color="#E5E7EB", linestyle="-", linewidth=0.45, zorder=0)
    ax.grid(axis="x", color="#F3F4F6", linestyle=":", linewidth=0.4, zorder=0)
    ax.tick_params(labelsize=8, colors="#4B5563")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("#D1D5DB")


def _plot_larmor_strip(
    ax: mpl.axes.Axes,
    components: Sequence[Component],
    trace: PositionAngleTrace,
    *,
    component_labels: Sequence[str] | None,
    z_min: float,
    z_max: float,
) -> None:
    profile_z, profile_radians = cumulative_larmor_profile(
        components,
        voltage=trace.voltage,
        z_min=z_min,
        z_max=z_max,
    )
    profile_degrees = np.rad2deg(profile_radians)

    ax.set_facecolor("white")
    ax.plot(
        profile_z,
        profile_degrees,
        color=NEOARM_RAY_COLORS["objective_lens"],
        linewidth=1.55,
        zorder=3,
    )
    ax.fill_between(
        profile_z,
        0.0,
        profile_degrees,
        color=NEOARM_RAY_COLORS["objective_lens"],
        alpha=0.10,
        linewidth=0.0,
        zorder=1,
    )

    labels = tuple(component_labels) if component_labels is not None else None
    transform = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    em_index = 0
    for component_index, component in enumerate(components):
        if not isinstance(component, ElectromagneticLens):
            continue
        entry = float(component.z)
        thickness = max(0.0, float(np.asarray(component.thickness)))
        exit_z = entry + thickness
        if exit_z < z_min or entry > z_max:
            continue
        label = labels[component_index] if labels is not None else _as_name(component)
        role = _infer_element_role(component, label)
        color = NEOARM_RAY_COLORS.get(role, NEOARM_RAY_COLORS["lens"])
        if thickness > 0.0:
            ax.axvspan(entry, exit_z, color=color, alpha=0.07, linewidth=0.0, zorder=0)
        else:
            ax.axvline(entry, color=color, alpha=0.28, linewidth=0.7, zorder=0)
        marker_z = min(max(exit_z, z_min), z_max)
        marker_theta = _interpolate_trace_value(profile_z, profile_degrees, marker_z)
        ax.plot(
            [marker_z],
            [marker_theta],
            marker="o",
            markersize=3.0,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.8,
            zorder=4,
        )
        # Four deterministic lanes keep neighbouring lens names legible in
        # dense objective stacks.  Alternating only two lanes still lets, for
        # example, C-mini and Obj-post collide at publication size.
        label_lanes = (0.94, 0.73, 0.52, 0.31)
        ax.text(
            0.5 * (max(entry, z_min) + min(exit_z, z_max)),
            label_lanes[em_index % len(label_lanes)],
            str(label),
            transform=transform,
            ha="center",
            va="top",
            fontsize=5.8,
            color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.3},
            clip_on=True,
            zorder=5,
        )
        em_index += 1

    if em_index == 0:
        ax.text(
            0.5,
            0.54,
            "No electromagnetic-lens rotation in this model",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=6.5,
            color="#6B7280",
        )

    ax.set_xlim(z_min, z_max)
    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.set_xlabel("z [m]  (source → detector)", fontsize=7.5)
    ax.set_ylabel("θ [deg]", fontsize=7.5)
    ax.set_title(
        f"Cumulative Larmor rotation at {trace.voltage / 1e3:g} kV",
        loc="left",
        fontsize=8.2,
        fontweight="bold",
    )
    ax.axhline(0.0, color="#CBD5E1", linewidth=0.65, zorder=0)
    ax.grid(color="#E5E7EB", linestyle=":", linewidth=0.45, zorder=0)
    ax.tick_params(labelsize=6.8, colors="#4B5563", pad=1.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("#D1D5DB")

    radians_axis = ax.secondary_yaxis(
        "right",
        functions=(np.deg2rad, np.rad2deg),
    )
    radians_axis.set_ylabel("θ [rad]", fontsize=7.5)
    radians_axis.tick_params(labelsize=6.8, colors="#4B5563", pad=1.5)
    radians_axis.spines["right"].set_color("#D1D5DB")


def plot_neoarm_ray_diagram(
    components: Sequence[Component],
    trace: PositionAngleTrace,
    *,
    component_labels: Sequence[str] | None = None,
    component_roles: Sequence[str] | None = None,
    extra_elements: Sequence[ColumnElement] = (),
    sections: Sequence[ColumnSection] | None = None,
    comparison_traces: Mapping[str, PositionAngleTrace] | None = None,
    title: str | None = None,
    annotate_events: bool = True,
    trace_semantics: str = "gaussian",
    figsize: tuple[float, float] = (14.0, 10.0),
) -> tuple[mpl.figure.Figure, dict[str, mpl.axes.Axes]]:
    """Render a full-column overview, three linear details, and Larmor strip.

    Every panel uses a linear physical z-axis.  Detail panels deliberately use
    independent, explicitly labelled transverse scales; this avoids hiding a
    nanometre source/sample beam when the detector footprint is millimetres.
    ``comparison_traces`` overlays fitted or +/-wobble solutions with common
    limits and distinct line styles.  When fitted source or detector positions
    differ, the overview uses the union of every trace's axial extent so no
    comparison ray is silently clipped.

    ``trace_semantics='calibrated_response'`` draws the independently
    calibrated position and tilt response rays without treating their
    root-sum-square as a Gaussian envelope or labelling its minima as waists.
    The default ``'gaussian'`` preserves the physical Gaussian envelope.

    The returned axes mapping contains ``overview``, ``detail_1`` through
    ``detail_3``, and ``larmor``.  The compact Larmor panel shows cumulative
    electromagnetic-lens rotation in degrees with a secondary radian scale.
    """

    if not isinstance(trace, PositionAngleTrace):
        raise TypeError("trace must be a PositionAngleTrace.")
    trace_semantics = str(trace_semantics).strip().lower()
    if trace_semantics not in {"gaussian", "calibrated_response"}:
        raise ValueError(
            "trace_semantics must be 'gaussian' or 'calibrated_response'."
        )
    gaussian_semantics = trace_semantics == "gaussian"
    z = np.asarray(trace.z, dtype=float)
    if z.ndim != 1 or z.size < 2 or not np.all(np.isfinite(z)):
        raise ValueError("trace.z must be a finite one-dimensional array with at least two values.")
    primary_z_min, primary_z_max = float(np.min(z)), float(np.max(z))
    if primary_z_max <= primary_z_min:
        raise ValueError("trace must span a positive axial distance.")

    comparisons = dict(comparison_traces or {})
    all_traces = [trace, *comparisons.values()]
    trace_extents = [(primary_z_min, primary_z_max)]
    for comparison in comparisons.values():
        if not isinstance(comparison, PositionAngleTrace):
            raise TypeError("Every comparison trace must be a PositionAngleTrace.")
        if comparison.coordinate != trace.coordinate:
            raise ValueError("All traces must use the same ray coordinate.")
        comparison_z = np.asarray(comparison.z, dtype=float)
        if (
            comparison_z.ndim != 1
            or comparison_z.size < 2
            or not np.all(np.isfinite(comparison_z))
        ):
            raise ValueError(
                "Every comparison trace.z must be a finite one-dimensional "
                "array with at least two values."
            )
        comparison_z_min = float(np.min(comparison_z))
        comparison_z_max = float(np.max(comparison_z))
        if comparison_z_max <= comparison_z_min:
            raise ValueError("Every comparison trace must span a positive axial distance.")
        trace_extents.append((comparison_z_min, comparison_z_max))
    z_min = min(extent[0] for extent in trace_extents)
    z_max = max(extent[1] for extent in trace_extents)

    elements = list(
        column_elements_from_components(
            components,
            labels=component_labels,
            roles=component_roles,
            extra_elements=extra_elements,
        )
    )
    if not any(element.role == "source" for element in elements):
        elements.append(
            ColumnElement(z=primary_z_min, label="Source", role="source")
        )
        elements.sort(key=lambda element: (element.z, element.label))

    if sections is None:
        sections_use = infer_neoarm_sections(elements, z_min, z_max)
    else:
        sections_use = tuple(sections)
        if len(sections_use) != 3:
            raise ValueError("sections must contain exactly three ColumnSection objects.")
        for section in sections_use:
            if section.z_max <= section.z_min:
                raise ValueError("Each section must have z_max > z_min.")

    with mpl.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 10.5,
            "axes.labelsize": 9,
            "savefig.facecolor": "white",
        }
    ):
        fig = plt.figure(figsize=figsize, facecolor="white", constrained_layout=False)
        grid = fig.add_gridspec(
            4,
            2,
            left=0.055,
            right=0.985,
            bottom=0.065,
            top=0.865,
            width_ratios=(1.04, 2.35),
            height_ratios=(1.0, 1.0, 1.0, 0.30),
            hspace=0.46,
            wspace=0.10,
        )
        axes: dict[str, mpl.axes.Axes] = {
            "overview": fig.add_subplot(grid[:, 0]),
            "detail_1": fig.add_subplot(grid[0, 1]),
            "detail_2": fig.add_subplot(grid[1, 1]),
            "detail_3": fig.add_subplot(grid[2, 1]),
            "larmor": fig.add_subplot(grid[3, 1]),
        }

        overview_section = ColumnSection("Full column", z_min, z_max)
        overview_limit, overview_factor, overview_unit = _section_scale(
            all_traces, overview_section
        )
        _configure_column_axis(
            axes["overview"],
            section=overview_section,
            limit_m=overview_limit,
            factor=overview_factor,
            unit=overview_unit,
            coordinate=trace.coordinate,
            title="Full column — true linear z",
        )
        _plot_trace_layer(
            axes["overview"],
            trace,
            z_min=z_min,
            z_max=z_max,
            factor=overview_factor,
            linestyle="-",
            alpha=0.95,
            linewidth=1.35,
            fill_envelope=True,
            scenario_label=trace.label,
            show_envelope=gaussian_semantics,
        )
        comparison_linestyles = ("--", ":", "-.")
        for idx, (comparison_label, comparison) in enumerate(comparisons.items()):
            _plot_trace_layer(
                axes["overview"],
                comparison,
                z_min=z_min,
                z_max=z_max,
                factor=overview_factor,
                linestyle=comparison_linestyles[idx % len(comparison_linestyles)],
                alpha=0.75,
                linewidth=1.1,
                fill_envelope=False,
                scenario_label=str(comparison_label),
                show_envelope=gaussian_semantics,
            )
        _draw_elements(
            axes["overview"],
            elements,
            z_min=z_min,
            z_max=z_max,
            label_fontsize=6.6,
            label_x=0.975,
            max_label_characters=22,
            label_line_height_fraction=0.015,
            label_horizontal_alignment="right",
        )

        for panel_idx, section in enumerate(sections_use, start=1):
            ax = axes[f"detail_{panel_idx}"]
            limit_m, factor, unit = _section_scale(all_traces, section)
            _configure_column_axis(
                ax,
                section=section,
                limit_m=limit_m,
                factor=factor,
                unit=unit,
                coordinate=trace.coordinate,
                title=section.title,
            )
            _plot_trace_layer(
                ax,
                trace,
                z_min=section.z_min,
                z_max=section.z_max,
                factor=factor,
                linestyle="-",
                alpha=0.98,
                linewidth=1.7,
                fill_envelope=True,
                scenario_label=trace.label,
                show_envelope=gaussian_semantics,
            )
            for idx, (comparison_label, comparison) in enumerate(comparisons.items()):
                _plot_trace_layer(
                    ax,
                    comparison,
                    z_min=section.z_min,
                    z_max=section.z_max,
                    factor=factor,
                    linestyle=comparison_linestyles[idx % len(comparison_linestyles)],
                    alpha=0.78,
                    linewidth=1.25,
                    fill_envelope=False,
                    scenario_label=str(comparison_label),
                    show_envelope=gaussian_semantics,
                )
            _draw_elements(
                ax,
                elements,
                z_min=section.z_min,
                z_max=section.z_max,
                label_fontsize=7.3,
                label_x=0.975,
                max_label_characters=27,
                label_line_height_fraction=0.050,
                label_horizontal_alignment="right",
            )
            if annotate_events:
                _annotate_trace_events(
                    ax,
                    trace,
                    z_min=section.z_min,
                    z_max=section.z_max,
                    factor=factor,
                    unit=unit,
                    include_waists=gaussian_semantics,
                )

        _plot_larmor_strip(
            axes["larmor"],
            components,
            trace,
            component_labels=component_labels,
            z_min=z_min,
            z_max=z_max,
        )

        # A compact figure-level legend keeps labels consistent across panels.
        if gaussian_semantics:
            handles = [
                mpl.lines.Line2D([], [], color=NEOARM_RAY_COLORS["position"], lw=1.8, label="height basis (A·h)"),
                mpl.lines.Line2D([], [], color=NEOARM_RAY_COLORS["angle"], lw=1.8, label="angle basis (B·α)"),
                mpl.patches.Patch(facecolor=NEOARM_RAY_COLORS["envelope"], alpha=0.14, label="Gaussian ±w"),
                mpl.lines.Line2D([], [], color="#374151", lw=1.5, linestyle="-", label=str(trace.label)),
            ]
        else:
            handles = [
                mpl.lines.Line2D([], [], color=NEOARM_RAY_COLORS["position"], lw=1.8, label="calibrated sample-position response"),
                mpl.lines.Line2D([], [], color=NEOARM_RAY_COLORS["angle"], lw=1.8, label="calibrated sample-tilt response"),
                mpl.lines.Line2D([], [], color="#374151", lw=1.5, linestyle="-", label=str(trace.label)),
            ]
        for idx, comparison_label in enumerate(comparisons):
            handles.append(
                mpl.lines.Line2D(
                    [],
                    [],
                    color="#374151",
                    lw=1.3,
                    linestyle=comparison_linestyles[idx % len(comparison_linestyles)],
                    label=str(comparison_label),
                )
            )
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.925),
            ncols=min(len(handles), 5),
            frameon=False,
            fontsize=8,
        )
        fig.suptitle(
            title or "NEOARM paraxial ray solution",
            x=0.52,
            y=0.975,
            fontsize=13,
            fontweight="bold",
        )
        fig.text(
            0.995,
            0.004,
            "All z axes are linear; detail panels use independent labelled transverse scales.",
            ha="right",
            va="bottom",
            fontsize=7,
            color="#6B7280",
        )
    return fig, axes


def _scenario_setting(setting, scenario: str, default=None):
    if isinstance(setting, Mapping):
        return setting.get(scenario, default)
    return setting


def plot_neoarm_ray_small_multiples(
    components_by_label: Mapping[str, Sequence[Component]] | Sequence[Component],
    traces_by_label: Mapping[str, PositionAngleTrace],
    *,
    component_labels: Mapping[str, Sequence[str]] | Sequence[str] | None = None,
    component_roles: Mapping[str, Sequence[str]] | Sequence[str] | None = None,
    extra_elements: (
        Mapping[str, Sequence[ColumnElement]] | Sequence[ColumnElement]
    ) = (),
    sections: Sequence[ColumnSection] | None = None,
    title: str | None = None,
    annotate_events: bool = False,
    figsize: tuple[float, float] | None = None,
) -> tuple[mpl.figure.Figure, dict[str, dict[str, mpl.axes.Axes]]]:
    """Plot scenarios as shared-scale full-column and detail small multiples.

    The mapping order defines the columns, making the helper suitable for
    ``truth / initial / fitted`` and ``nominal / -wobble / +wobble`` figures.
    Every row has one shared physical z extent, transverse unit, and transverse
    limit across scenarios.  The full-column row uses the union of all trace
    extents, while an automatically supplied source glyph remains at each
    scenario's own trace start.  Unlike an overlay, this keeps each solution
    fully visible while preserving direct geometric comparability.
    """

    scenarios = tuple(str(key) for key in traces_by_label)
    if not scenarios:
        raise ValueError("traces_by_label must contain at least one scenario.")
    traces = {str(key): value for key, value in traces_by_label.items()}
    if len(traces) != len(traces_by_label):
        raise ValueError("Scenario labels must remain unique when converted to strings.")
    for scenario, trace in traces.items():
        if not isinstance(trace, PositionAngleTrace):
            raise TypeError(f"Trace for {scenario!r} must be a PositionAngleTrace.")

    if isinstance(components_by_label, Mapping):
        missing = [scenario for scenario in scenarios if scenario not in components_by_label]
        if missing:
            raise ValueError(f"components_by_label is missing scenarios: {missing}")
        components_map = {
            scenario: tuple(components_by_label[scenario]) for scenario in scenarios
        }
    else:
        common_components = tuple(components_by_label)
        components_map = {scenario: common_components for scenario in scenarios}

    reference = traces[scenarios[0]]
    trace_extents: dict[str, tuple[float, float]] = {}
    for scenario in scenarios:
        trace = traces[scenario]
        trace_z = np.asarray(trace.z, dtype=float)
        if trace_z.ndim != 1 or trace_z.size < 2 or not np.all(np.isfinite(trace_z)):
            raise ValueError("Each trace.z must be finite, one-dimensional, and non-empty.")
        if trace.coordinate != reference.coordinate:
            raise ValueError("All scenario traces must use the same ray coordinate.")
        scenario_z_min = float(np.min(trace_z))
        scenario_z_max = float(np.max(trace_z))
        if scenario_z_max <= scenario_z_min:
            raise ValueError("Traces must span a positive axial distance.")
        trace_extents[scenario] = (scenario_z_min, scenario_z_max)
    z_min = min(extent[0] for extent in trace_extents.values())
    z_max = max(extent[1] for extent in trace_extents.values())

    elements_map: dict[str, tuple[ColumnElement, ...]] = {}
    for scenario in scenarios:
        labels_use = _scenario_setting(component_labels, scenario)
        roles_use = _scenario_setting(component_roles, scenario)
        extras_use = _scenario_setting(extra_elements, scenario, ())
        elements = list(
            column_elements_from_components(
                components_map[scenario],
                labels=labels_use,
                roles=roles_use,
                extra_elements=tuple(extras_use or ()),
            )
        )
        if not any(element.role == "source" for element in elements):
            elements.append(
                ColumnElement(trace_extents[scenario][0], "Source", "source")
            )
            elements.sort(key=lambda element: (element.z, element.label))
        elements_map[scenario] = tuple(elements)

    if sections is None:
        detail_sections = infer_neoarm_sections(
            elements_map[scenarios[0]], z_min, z_max
        )
    else:
        detail_sections = tuple(sections)
        if len(detail_sections) != 3:
            raise ValueError("sections must contain exactly three ColumnSection objects.")
        if any(section.z_max <= section.z_min for section in detail_sections):
            raise ValueError("Each section must have z_max > z_min.")
    rows = (
        ColumnSection("Full column — true linear z", z_min, z_max),
        *detail_sections,
    )
    row_keys = ("overview", "detail_1", "detail_2", "detail_3")

    n_scenarios = len(scenarios)
    if figsize is None:
        figsize = (max(5.0, 4.15 * n_scenarios), 10.0)
    with mpl.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "savefig.facecolor": "white",
        }
    ):
        fig, axes_array = plt.subplots(
            4,
            n_scenarios,
            figsize=figsize,
            squeeze=False,
            facecolor="white",
        )
        fig.subplots_adjust(
            left=0.065,
            right=0.985,
            bottom=0.075,
            top=0.855,
            wspace=0.12,
            hspace=0.38,
        )
        axes: dict[str, dict[str, mpl.axes.Axes]] = {
            scenario: {} for scenario in scenarios
        }
        all_traces = [traces[scenario] for scenario in scenarios]

        for row_index, (row_key, section) in enumerate(zip(row_keys, rows)):
            limit_m, factor, unit = _section_scale(all_traces, section)
            for column_index, scenario in enumerate(scenarios):
                ax = axes_array[row_index, column_index]
                axes[scenario][row_key] = ax
                _configure_column_axis(
                    ax,
                    section=section,
                    limit_m=limit_m,
                    factor=factor,
                    unit=unit,
                    coordinate=reference.coordinate,
                    title=section.title,
                )
                if column_index > 0:
                    ax.set_ylabel("")
                    ax.tick_params(labelleft=False)
                _plot_trace_layer(
                    ax,
                    traces[scenario],
                    z_min=section.z_min,
                    z_max=section.z_max,
                    factor=factor,
                    linestyle="-",
                    alpha=0.98,
                    linewidth=1.45 if row_index == 0 else 1.65,
                    fill_envelope=True,
                    scenario_label=scenario,
                )
                _draw_elements(
                    ax,
                    elements_map[scenario],
                    z_min=section.z_min,
                    z_max=section.z_max,
                    label_fontsize=5.9 if row_index == 0 else 6.2,
                    label_x=0.975,
                    max_label_characters=20,
                    label_line_height_fraction=0.050,
                    label_horizontal_alignment="right",
                )
                if annotate_events:
                    _annotate_trace_events(
                        ax,
                        traces[scenario],
                        z_min=section.z_min,
                        z_max=section.z_max,
                        factor=factor,
                        unit=unit,
                        max_per_kind=2,
                    )

        for column_index, scenario in enumerate(scenarios):
            box = axes_array[0, column_index].get_position()
            fig.text(
                0.5 * (box.x0 + box.x1),
                0.885,
                scenario,
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color="#1F2937",
            )
        handles = [
            mpl.lines.Line2D(
                [],
                [],
                color=NEOARM_RAY_COLORS["position"],
                lw=1.8,
                label="height basis (A·h)",
            ),
            mpl.lines.Line2D(
                [],
                [],
                color=NEOARM_RAY_COLORS["angle"],
                lw=1.8,
                label="angle basis (B·α)",
            ),
            mpl.patches.Patch(
                facecolor=NEOARM_RAY_COLORS["envelope"],
                alpha=0.14,
                label="Gaussian ±w",
            ),
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.945),
            ncols=3,
            frameon=False,
            fontsize=8,
        )
        fig.suptitle(
            title or "NEOARM ray-solution comparison",
            y=0.985,
            fontsize=13,
            fontweight="bold",
        )
        fig.text(
            0.995,
            0.006,
            "Each row uses identical linear z and transverse scales across scenarios.",
            ha="right",
            va="bottom",
            fontsize=7,
            color="#6B7280",
        )
    return fig, axes


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def export_ray_figure(
    fig: mpl.figure.Figure,
    path_stem: str | Path,
    *,
    formats: Sequence[str] = ("png", "svg", "pdf"),
    dpi: int = 300,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, Path]:
    """Export a ray figure to vector/raster formats plus JSON metadata.

    A conservative subset of metadata is embedded in each image.  The complete
    JSON-safe mapping is written to ``<stem>.metadata.json`` for lossless,
    format-independent provenance.
    """

    if not isinstance(fig, mpl.figure.Figure):
        raise TypeError("fig must be a matplotlib Figure.")
    dpi = int(dpi)
    if dpi <= 0:
        raise ValueError("dpi must be positive.")
    formats_use = tuple(str(fmt).lower().lstrip(".") for fmt in formats)
    if not formats_use:
        raise ValueError("formats must not be empty.")
    supported = {"png", "svg", "pdf"}
    unknown = set(formats_use) - supported
    if unknown:
        raise ValueError(f"Unsupported export format(s): {sorted(unknown)}")

    stem = Path(path_stem)
    if stem.suffix.lower().lstrip(".") in supported:
        stem = stem.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)

    metadata_safe = _json_safe(dict(metadata or {}))
    payload = {
        "renderer": "temgym_core.plotting.export_ray_figure",
        "formats": list(formats_use),
        "dpi": dpi,
        "figure_size_inches": [float(v) for v in fig.get_size_inches()],
        "metadata": metadata_safe,
    }
    title = str(metadata_safe.get("title", "NEOARM paraxial ray diagram"))
    author = str(metadata_safe.get("author", "TemGymCore"))
    description = json.dumps(metadata_safe, sort_keys=True, ensure_ascii=False)

    outputs: dict[str, Path] = {}
    for fmt in formats_use:
        output = stem.with_suffix(f".{fmt}")
        if fmt == "png":
            embedded = {
                "Title": title,
                "Author": author,
                "Description": description,
                "Software": "TemGymCore",
            }
        elif fmt == "svg":
            embedded = {
                "Title": title,
                "Creator": author,
                "Description": description,
            }
        else:  # PDF
            embedded = {
                "Title": title,
                "Author": author,
                "Subject": description,
                "Creator": "TemGymCore",
            }
        fig.savefig(
            output,
            format=fmt,
            dpi=dpi,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            transparent=False,
            metadata=embedded,
        )
        outputs[fmt] = output

    metadata_path = stem.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    outputs["metadata"] = metadata_path
    return outputs


class _ColumnSymbolTransform(mpl.transforms.Affine2DBase):
    """Rigid point-sized symbol anchored to a posed column in display space.

    Input coordinates are (transverse, axial) offsets in typographic points.
    Recompute the displayed column direction after axis limits, layout or DPI
    change. Transverse and axial directions stay perpendicular on the page.
    """

    def __init__(self, ax, pose, z):
        super().__init__()
        self._axes, self._pose, self._z = ax, pose, float(z)
        self.set_children(ax.transData, ax.figure.dpi_scale_trans, pose)

    def get_matrix(self):
        positions = self._axes.transData.transform(
            self._pose.transform(np.array([[0., self._z], [0., self._z + .001]]))
        )
        axial = positions[1] - positions[0]
        axial /= np.linalg.norm(axial)
        transverse = np.array([-axial[1], axial[0]])
        scale = self._axes.figure.dpi / 72.
        return np.array([[transverse[0]*scale, axial[0]*scale, positions[0,0]],
                         [transverse[1]*scale, axial[1]*scale, positions[0,1]],
                         [0.,0.,1.]])


def plot_reflection_column(
    components: Sequence[Component],
    trace: PositionAngleTrace,
    *,
    sample_z: float,
    outgoing_paths_xz: np.ndarray,
    detector_segment_xz: np.ndarray,
    surface_angle_rad: float = 0.0,
    column_frame=None,
    component_labels: Sequence[str] | None = None,
    order_weights: np.ndarray | None = None,
    envelope_scale: float | None = None,
    incident_envelope: np.ndarray | None = None,
    plot_params: PlotParams | None = None,
    title: str = "SEM reflection geometry",
    specimen_label: str = "Si(001)",
    show_section_labels: bool = True,
    ax: mpl.axes.Axes | None = None,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """Pose the existing plot_model drawing in a horizontal laboratory view.

    column_frame maps incident-column (x,y,z) into laboratory XYZ. Supplied
    reflected paths and detector endpoints are laboratory (X,Z), in metres.
    The sample is at local (0,0,sample_z). surface_angle_rad is its laboratory
    tilt, zero for a flat crystal. The column must lie in the laboratory XZ
    plane. Only the incident Gaussian envelope is magnified; axes show physical
    laboratory positions with independent horizontal/vertical display scales.

    incident_envelope optionally supplies (z, centroid, radius) rows in metres
    from Gaussian propagation, including an aberrated objective. Otherwise the
    envelope is reconstructed from the supplied linear basis trace.

    Uses plot_model and its PlotParams/deflector symbols, together with
    plot_ray_bundle for the smooth Gaussian envelope. Paired coil-and-yoke
    glyphs retain the reflection example's lens appearance. Component symbols
    use point-sized rigid display transforms, perpendicular to the displayed
    column axis, so unequal data scales cannot shear or stretch them.
    """
    from .coordinate_transforms import PlaneFrame

    paths = np.asarray(outgoing_paths_xz, dtype=float)
    screen = np.asarray(detector_segment_xz, dtype=float)
    if paths.ndim != 3 or paths.shape[-1] != 2 or paths.shape[0] == 0 or paths.shape[1] < 2:
        raise ValueError("outgoing_paths_xz must have shape (orders, points>=2, 2).")
    if screen.shape != (2, 2) or not np.isfinite(screen).all() or not np.isfinite(paths).all():
        raise ValueError("Supply finite reflected paths and two detector (X,Z) endpoints.")
    if not np.isfinite(surface_angle_rad) or not np.isfinite(sample_z):
        raise ValueError("Supply finite specimen geometry.")
    labels = tuple(component_labels) if component_labels is not None else tuple(_as_name(c) for c in components)
    if len(labels) != len(components):
        raise ValueError("Provide one label per incident-column component.")
    weights = np.ones(paths.shape[0]) if order_weights is None else np.asarray(order_weights, dtype=float)
    if weights.shape != (paths.shape[0],) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Provide a finite nonnegative weight per reflected path.")
    z = np.asarray(trace.z, dtype=float)
    if z.size < 2 or np.any(np.diff(z) < 0) or not np.isfinite(z).all():
        raise ValueError("The incident trace must contain finite ordered axial positions.")
    if not np.isclose(z[-1], sample_z, rtol=0, atol=1e-12):
        raise ValueError("The incident trace must end at sample_z.")
    if column_frame is None:
        column_frame = PlaneFrame(origin_xyz=np.zeros(3), u_xyz=np.array([1.,0.,0.]),
                                  v_xyz=np.array([0.,1.,0.]))
    basis = np.asarray(column_frame.basis_xyz)
    origin = np.asarray(column_frame.origin_xyz)
    if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-12):
        raise ValueError("column_frame must be orthonormal.")
    if not np.allclose(basis[1,[0,2]], 0., atol=1e-12):
        raise ValueError("This side view requires a column in the laboratory XZ plane.")
    # Native plot_model coordinates are (transverse x, axial z). Reuse its
    # artists, applying the physical pose and a metres-to-mm display transform.
    matrix = np.array([[basis[2,0]*1e3, basis[2,2]*1e3, origin[2]*1e3],
                       [basis[0,0]*1e3, basis[0,2]*1e3, origin[0]*1e3], [0.,0.,1.]])
    pose = mpl.transforms.Affine2D(matrix)
    p = plot_params if plot_params is not None else legacy_beam_plot_params(
        figsize=(14., 5.), fill_color="#269F83", fill_alpha=.55,
        ray_color="#207E69", edge_lw=0., interior_ray_alpha=0.,
        center_ray_lw=.55, center_ray_alpha=.4,
        fixed_xmax=.004, component_half_width_frac=.92,
        auto_lens_height=False, lens_height=.0015, show_side_guides=False,
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=p.figsize)
    else:
        fig = ax.figure
    before = set(ax.get_children())
    basis_rays = make_waist_divergence_rays(trace.position_scale, voltage=trace.voltage, z=z[0])
    plot_model(components, rays=basis_rays, plot_params=p, ax=ax)
    # Keep plot_model's actual optical symbols. Replace only its vertical
    # labels and sparse two-ray display with the supplied Gaussian trace.
    for artist in tuple(ax.get_children()):
        if artist in before:
            continue
        if isinstance(artist, mpl.text.Text) or (
            isinstance(artist, (mpl.lines.Line2D, mpl.collections.Collection)) and artist.get_zorder() < 100
        ):
            artist.remove()
        elif isinstance(artist, (mpl.lines.Line2D, mpl.patches.Patch)):
            artist.set_transform(pose + ax.transData)

    # Deflectors keep plot_model's red/blue style, but their transverse
    # direction must be perpendicular to the *displayed* microscope axis.
    deflector_positions = []
    for component in components:
        if isinstance(component, (Deflector, DoubleDeflector)):
            deflector_positions.append(float(component.z))
            if isinstance(component, DoubleDeflector):
                deflector_positions.append(float(component.z_second))
    for deflector_z in deflector_positions:
        lines = [line for line in ax.lines if line not in before
                 and line.get_zorder() >= 100
                 and np.allclose(line.get_ydata(), deflector_z, rtol=0, atol=1e-12)]
        half_width = max((float(np.max(np.abs(line.get_xdata()))) for line in lines), default=0.)
        if half_width > 0:
            transform = _ColumnSymbolTransform(ax, pose, deflector_z)
            for line in lines:
                line.set_xdata(np.asarray(line.get_xdata()) * 20. / half_width)
                line.set_ydata(np.zeros_like(line.get_ydata()))
                line.set_transform(transform)
                line.set_gid("reflection_deflector")

    # Retain the paired coil-and-yoke models as rigid square symbols. Their
    # dimensions are in points, not physical manufactured polepiece sizes.
    for patch in tuple(ax.patches):
        if isinstance(patch, mpl.patches.Arc) and patch not in before:
            patch.remove()
    for component in components:
        if not isinstance(component, (Lens, ElectromagneticLens, HardEdgeSolenoid, ThickLens)):
            continue
        transform = _ColumnSymbolTransform(ax, pose, float(component.z))
        for sign in (-1, 1):
            left = 15. if sign > 0 else -27.
            ax.add_patch(mpl.patches.FancyBboxPatch(
                (left, -6.), 12., 12.,
                boxstyle="round,pad=0,rounding_size=0.8",
                transform=transform, facecolor="#DCE4EA", edgecolor="#223447",
                linewidth=1.1, zorder=999, gid="reflection_lens_yoke",
            ))
            ax.add_patch(mpl.patches.Rectangle(
                (left+2., -4.), 8., 8., transform=transform,
                facecolor="#C49657", edgecolor="#8B673D", linewidth=.7,
                hatch="|||", zorder=1000, gid="reflection_lens_coil",
            ))

    dense_z = np.unique(np.concatenate((z, np.linspace(z[0], sample_z, 2000))))
    position = np.interp(dense_z, z, trace.position)
    angle = np.interp(dense_z, z, trace.angle)
    centre = np.interp(dense_z, z, trace.centroid)
    radius = gaussian_basis_envelope(position, angle)
    if incident_envelope is not None:
        envelope = np.asarray(incident_envelope, dtype=float)
        if (envelope.ndim != 2 or envelope.shape[1] != 3 or envelope.shape[0] < 2
                or not np.isfinite(envelope).all() or np.any(np.diff(envelope[:, 0]) < 0)
                or np.any(envelope[:, 2] < 0)
                or not np.isclose(envelope[0, 0], z[0], rtol=0, atol=1e-12)
                or not np.isclose(envelope[-1, 0], sample_z, rtol=0, atol=1e-12)):
            raise ValueError("incident_envelope must contain ordered finite (z, centroid, radius) rows spanning the column.")
        dense_z = np.unique(np.concatenate((dense_z, envelope[:, 0])))
        centre = np.interp(dense_z, envelope[:, 0], envelope[:, 1])
        radius = np.interp(dense_z, envelope[:, 0], envelope[:, 2])
    if envelope_scale is None:
        envelope_scale = .0015 / max(float(radius.max()), np.finfo(float).eps)
    if not np.isfinite(envelope_scale) or envelope_scale <= 0:
        raise ValueError("envelope_scale must be finite and positive.")
    rays = np.column_stack((centre-envelope_scale*radius,centre,centre+envelope_scale*radius))
    artists = plot_ray_bundle(ax,rays,dense_z,p,band_mode="gradient")
    for group in artists.values():
        for artist in group:
            artist.set_transform(pose + ax.transData)

    def world_xz(local_x, local_z):
        return pose.transform(np.column_stack((np.atleast_1d(local_x), np.atleast_1d(local_z))))

    sample = world_xz(0.,sample_z)[0]
    source = world_xz(trace.centroid[0],z[0])[0]
    ax.plot(*source, marker="o", ms=4, color=p.text_color, zorder=1001)
    ax.annotate("Source", source, xytext=(-4,-24), textcoords="offset points", ha="center", fontsize=11)
    # Stagger close objective/deflector labels without changing their positions.
    for component, label in zip(components,labels):
        if isinstance(component, Plane):
            continue
        location = world_xz(0.,float(component.z))[0]
        offset = 39 if isinstance(component, Deflector) else -69
        ax.annotate(label, location, xytext=(0,offset),textcoords="offset points",
                    ha="center", va="bottom" if offset > 0 else "top", fontsize=10,
                    arrowprops=dict(arrowstyle="-",lw=.45,color=".55",shrinkA=5,shrinkB=18))
    # Layered lattice schematic; the beam-facing edge stays on the physical plane.
    length = 48.  # schematic crystal length in mm
    tangent = np.array([np.cos(surface_angle_rad),np.sin(surface_angle_rad)])
    up = np.array([-tangent[1],tangent[0]])
    ends = sample + np.array([-.5,.5])[:,None]*length*tangent
    crystal = np.array([ends[0],ends[1],ends[1]+2*up,ends[0]+2*up])
    ax.add_patch(mpl.patches.Polygon(crystal, facecolor="#E5EDF6",
        edgecolor="#536C86", lw=.8, zorder=1000))
    # Atom rows illustrate crystallinity, not a literal atomic cross section.
    for row in range(4):
        lateral = np.linspace(-.45, .45, 13) * length
        if row % 2:
            lateral = lateral[:-1] + length*.0375
        atoms = sample + lateral[:,None]*tangent + (.25 + row*.45)*up
        ax.plot(atoms[:,0], atoms[:,1], color="#ABC0D4", lw=.45, zorder=1001)
        ax.scatter(atoms[:,0], atoms[:,1], s=5 if row else 8,
            color="#688BAA" if row else "#C29452", edgecolors="white",
            linewidths=.25, zorder=1002)
    ax.plot(ends[:,0],ends[:,1],color="#314E69",lw=1.4,zorder=1001)
    ax.text(sample[0],sample[1]+4.5,specimen_label,ha="center",fontsize=10)
    ax.scatter(*sample,s=15,color="black",zorder=1002)
    # Opposite lateral orders can coincide in an XZ side projection. Draw
    # that geometric path once and combine its display weight, rather than
    # accumulating dark strokes on top of each other.
    projected = {}
    for path,weight in zip(paths,weights):
        key = tuple(np.round(path,12).ravel())
        if key in projected:
            projected[key][1] += weight
        else:
            projected[key] = [path,float(weight)]
    maximum = max(max(weight for _,weight in projected.values()),np.finfo(float).eps)
    for path,weight in projected.values():
        if weight <= 0:
            continue
        fraction = np.sqrt(weight/maximum)
        ax.plot(path[:,1]*1e3,path[:,0]*1e3,color=p.ray_color,lw=.55+.55*fraction,
                alpha=.3+.45*fraction,zorder=3,solid_capstyle="round",solid_joinstyle="round")
    ax.plot(screen[:,1]*1e3,screen[:,0]*1e3,color="dimgrey",lw=5,solid_capstyle="butt",zorder=1000)
    endpoint = screen[np.argmax(screen[:,0]),[1,0]]*1e3
    ax.annotate("Medipix3",endpoint,xytext=(0,15),textcoords="offset points",ha="center",fontsize=11)
    # Context labels occupy clear space between the column and reflected fan.
    if show_section_labels:
        ax.text(.23,1.01,"SEM column",transform=ax.transAxes,ha="center",fontsize=12)
        ax.text(.76,1.01,"Reflected diffraction",transform=ax.transAxes,ha="center",fontsize=12)
    ax.set_title(title,loc="left",fontsize=15,pad=39 if show_section_labels else 12)
    all_points = np.concatenate((paths.reshape(-1,2)[:,::-1]*1e3,screen[:,::-1]*1e3,source[None,:],crystal))
    lo,hi = all_points.min(axis=0),all_points.max(axis=0)
    ax.set(xlim=(lo[0]-25,hi[0]+30),ylim=(lo[1]-8,hi[1]+9),
           xlabel="Laboratory horizontal position (mm)",ylabel="Height (mm)")
    ax.set_yticks(np.arange(np.ceil((lo[1]-8)/5)*5,hi[1]+9,5))
    ax.grid(False)
    ax.tick_params(labelsize=9,length=0,colors=".4")
    ax.xaxis.label.set_color(".4"); ax.yaxis.label.set_color(".4")
    return fig,ax
