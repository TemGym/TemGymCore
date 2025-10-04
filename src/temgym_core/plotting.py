from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from .ray import Ray
from .run import run_iter
from .components import (
    Component,
    Detector,
    Lens,
    Deflector,
    Biprism,
)
from .source import Source


@dataclass
class PlotParams:
    figsize: Tuple[float, float] = (9.0, 5.0)
    extent_scale: float = 0.7
    label_fontsize: int = 11
    ray_color: str = "tab:blue"
    ray_lw: float = 1.2
    ray_alpha: float = 0.8
    fill_color: str = "#7fffd4"  # aqua green (aquamarine)
    fill_alpha: float = 0.20
    edge_lw: float = 1.8
    component_lw: float = 3.0
    lens_height: float = 0.03  # relative to figure height
    lens_radius: float = 0.002  # radius of circle to draw lens
    biprism_radius: float = 0.001  # radius of circle to draw biprism
    label_offset: float = 0.0  # additional x offset for component labels
    label_side: str = "right"  # "left" or "right" side
    min_z: float | None = None  # optional override for min z limit
    max_z: float | None = None  # optional override for max z limit
    component_x: float = 0.002  # x extent for components (e.g. deflector half-width)
    # Plot behavior parameters (moved from plot_model signature)
    band_mode: str = "fill"  # "fill" (envelope fill) or "lines" (draw line segments)
    yscale: str = "linear"   # "linear", "log", or "symlog"
    y_linthresh: float = 1e-6  # linthresh for symlog
    # Sample overlay configuration
    sample_show: bool = True
    sample_z: float = 0.0
    sample_x: float = 0.0
    sample_color: str = "dodgerblue"
    sample_label: str | None = None  # default is generated when None
    sample_label_anchor: str = "NE"  # one of: N,S,E,W,NE,NW,SE,SW,C
    sample_zorder: float = 1000.0
    sample_lw: float = 1.6
    sample_arrow_lw: float = 1.0
    sample_show_center: bool = True
    sample_center_ms: float = 4.0
    sample_show_label: bool = True
    sample_side: float = 0.0
    sample_side_frac: float = 0.08
    sample_label_offset: Tuple[float, float] | None = None  # (dx, dy) in data units
    sample_label_offset_frac: Tuple[float, float] = (0.2, 0.2)  # fractions of side
    label_side: str = "right"  # "left" or "right" side
    label_offset: float = 0.0  # additional x offset for component labels


def _as_name(obj: object) -> str:
    return type(obj).__name__


def _ensure_initial_ray(
    components: Sequence[Source | Component],
    initial_ray: Ray | None,
    num_rays: int,
    random: bool,
) -> Ray:
    # Prefer a provided ray bundle
    if initial_ray is not None:
        return initial_ray

    # Try to generate from the first Source found
    for c in components:
        if isinstance(c, Source):
            return c.make_rays(num_rays, random=random)

    # Fall back to a single on-axis ray at the first element's z
    z0 = 0.0
    if len(components) > 0 and hasattr(components[0], "z"):
        z0 = float(getattr(components[0], "z"))
    return Ray(x=0.0, y=0.0, dx=0.0, dy=0.0, z=z0, pathlength=0.0)


def plot_model(
    components: Sequence[Source | Component],
    *,
    rays: Ray | None = None,
    initial_ray: Ray | None = None,
    num_rays: int = 101,
    random: bool = False,
    plot_params: PlotParams = PlotParams(),
    ax: mpl.axes.Axes | None = None,
):
    """Plot a 2D schematic (x vs z) of a model with a ray bundle and components.

    Parameters
    ----------
    components : sequence of Source or Component
        Model elements ordered by increasing z.
    rays : Ray, optional
        A Ray or Ray bundle to use as input. Overrides initial_ray and Source-based generation.
    initial_ray : Ray, optional
        Back-compat alias for the starting ray/bundle. Ignored if `rays` is provided.
    num_rays : int, default 101
        Number of rays if generating from a Source.
    random : bool, default False
        Randomize sampling when using a Source.
    plot_params : PlotParams, optional
        Style parameters for the plot.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, a new figure and axes are created.
    Additional styling via PlotParams
    ---------------------------------
    - band_mode: {"fill", "lines"}
        "fill": fill envelope between edge rays; "lines": segments between adjacent rays per z.
    - yscale: {"linear", "log", "symlog"}
        Set z-axis scaling. "log" requires all z>0; otherwise falls back to "symlog".
    - y_linthresh: float
        Linear range around zero used when yscale="symlog".

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    p = plot_params

    # Prefer explicitly provided rays/bundle; else initial_ray; else Source/auto.
    ray0 = rays if rays is not None else _ensure_initial_ray(
        components, initial_ray, num_rays, random
    )

    # Simulate and stack positions
    steps: list[Tuple[object, Ray]] = list(run_iter(ray0, components))
    X, Z = _stack_ray_positions(steps)

    # Create axes if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=p.figsize)
    else:
        fig = ax.figure

    if X.size == 0:
        return fig, ax

    # Determine x half-extent from components, detectors, and rays
    comp_half_x = float(p.component_x)
    det_half_x = 0.0
    for c in components:
        if isinstance(c, Detector):
            det_half_x = max(det_half_x, float(c.pixel_size[0] * c.shape[0] / 2.0))

    x_half = max(comp_half_x, det_half_x)
    x_half *= 1.02  # small margin

    # z limits
    comp_zs = [float(getattr(c, "z")) for c in components if hasattr(c, "z")]
    p_min_z = getattr(p, "min_z", None)
    p_max_z = getattr(p, "max_z", None)
    min_z = float(p_min_z) if p_min_z is not None else (
        min([float(np.min(Z))] + comp_zs) if comp_zs else float(np.min(Z))
    )
    max_z = float(p_max_z) if p_max_z is not None else (
        max([float(np.max(Z))] + comp_zs) if comp_zs else float(np.max(Z))
    )

    # Axes style
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=10)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(color="lightgrey", linestyle="--", linewidth=0.5)
    ax.grid(which="minor", color="#EEEEEE", linestyle=":", linewidth=0.5)

    # Y-axis scaling (z-axis)
    scale = str(p.yscale).lower().strip()
    if scale == "log":
        if min_z <= 0:
            ax.set_yscale("symlog", linthresh=p.y_linthresh)
            scale = "symlog"
        else:
            ax.set_yscale("log")
    elif scale == "symlog":
        ax.set_yscale("symlog", linthresh=p.y_linthresh)
    else:
        ax.set_yscale("linear")

    # Limits and ticks (invert z-axis for optical drawings)
    ax.set_xlim([-x_half, x_half])
    ax.set_ylim([max_z, min_z])

    yticks = sorted(comp_zs)
    if scale == "log":
        yticks = [t for t in yticks if t > 0]
    if yticks:
        ax.set_yticks(yticks)

    # Rays and envelope
    _ = plot_ray_bundle(ax, X, Z, p)

    # Configurable square "sample" overlay with movable label
    sample_show = bool(p.sample_show)
    sample_z = float(p.sample_z)
    sample_x = float(p.sample_x)
    sample_color = str(p.sample_color)
    sample_label = (
        p.sample_label if p.sample_label is not None else f"Sample — z = {p.sample_z:g}"
    )
    # one of: N,S,E,W,NE,NW,SE,SW,C
    sample_anchor = str(p.sample_label_anchor).upper()
    sample_zorder = float(p.sample_zorder)
    sample_lw = float(p.sample_lw)
    sample_arrow_lw = float(p.sample_arrow_lw)
    sample_show_label = bool(p.sample_show_label)
    abs_off = p.sample_label_offset
    off_frac = p.sample_label_offset_frac

    # Only draw if visible on current scale
    visible_in_range = (min_z <= sample_z <= max_z)
    allowed_on_scale = (scale != "log") or (sample_z > 0.0)
    if sample_show and visible_in_range and allowed_on_scale:
        # Determine side length (data units).
        # Use explicit size if provided, else a fraction of plot extents.
        side = float(p.sample_side)
        if not (side > 0.0):
            # of the smaller of x or z spans
            frac = float(p.sample_side_frac)
            side = max(1e-12, frac * min(2.0 * x_half, abs(max_z - min_z)))

        # Draw visually square patch: adjust height in data units to match x width on screen
        trans = ax.transData
        dx0 = max(1e-12, 1e-6 * (2.0 * x_half))
        dy0 = max(1e-12, 1e-6 * max(abs(max_z - min_z), 1.0))
        x1p = trans.transform((sample_x - dx0 / 2.0, sample_z))[0]
        x2p = trans.transform((sample_x + dx0 / 2.0, sample_z))[0]
        y1p = trans.transform((sample_x, sample_z - dy0 / 2.0))[1]
        y2p = trans.transform((sample_x, sample_z + dy0 / 2.0))[1]
        pix_per_data_x = abs(x2p - x1p) / dx0
        pix_per_data_y = max(1e-12, abs(y2p - y1p) / dy0)

        width_data = side
        height_data = width_data * (pix_per_data_x / pix_per_data_y)

        rect = mpl.patches.Rectangle(
            (sample_x - width_data / 2.0, sample_z - height_data / 2.0),
            width_data,
            height_data,
            facecolor="none",
            edgecolor=sample_color,
            lw=sample_lw,
            zorder=sample_zorder,
        )
        ax.add_patch(rect)

        # Movable label with simple anchor and offset controls
        if sample_show_label:
            anchor_map = {
                "N": (0.0, +0.5), "S": (0.0, -0.5), "E": (+0.5, 0.0), "W": (-0.5, 0.0),
                "NE": (+0.5, +0.5), "NW": (-0.5, +0.5), "SE": (+0.5, -0.5), "SW": (-0.5, -0.5),
                "C": (0.0, 0.0),
            }
            axo, ayo = anchor_map.get(sample_anchor, anchor_map["NE"])
            anchor_x = sample_x + axo * side
            anchor_y = sample_z + ayo * side

            # Offset can be absolute (data units) or fractional of side
            if abs_off is not None:
                dx, dy = float(abs_off[0]), float(abs_off[1])
            else:
                sign_x = 1 if axo >= 0 else -1
                sign_y = 1 if ayo >= 0 else -1
                # If anchored at center, push to the NE by default
                if axo == 0 and ayo == 0:
                    sign_x = sign_y = 1
                dx = float(off_frac[0]) * side * sign_x
                dy = float(off_frac[1]) * side * sign_y

            label_x = anchor_x + dx
            label_y = anchor_y + dy
            text_ha = "left" if dx >= 0 else "right"
            text_va = "bottom" if dy >= 0 else "top"

            ax.annotate(
                str(sample_label),
                xy=(anchor_x, anchor_y),
                xytext=(label_x, label_y),
                textcoords="data",
                arrowprops=dict(
                    arrowstyle="-",
                    connectionstyle="arc3,rad=0.0",
                    color=sample_color,
                    lw=sample_arrow_lw,
                ),
                ha=text_ha,
                va=text_va,
                color=sample_color,
                fontsize=p.label_fontsize,
                zorder=sample_zorder + 2,
            )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.yaxis.set_label_coords(-0.05, 0.25)

    # Components
    aspect = p.figsize[1] / p.figsize[0]
    label_side = p.label_side
    label_offset = p.label_offset
    label_x_base = p.extent_scale * x_half
    label_x = (label_x_base if label_side != "left" else -label_x_base) + label_offset
    ha = "left" if label_side != "left" else "right"

    for c in components:
        name = _as_name(c)
        if isinstance(c, Deflector):
            hw = float(p.component_x)
            ax.text(label_x, c.z, name, fontsize=p.label_fontsize, va="center", ha=ha, zorder=1000)
            ax.plot([-hw, 0], [c.z, c.z], color="lightcoral", linewidth=p.component_lw, zorder=999)
            ax.plot([0, hw], [c.z, c.z], color="lightblue", linewidth=p.component_lw, zorder=999)
            ax.plot(
                [-hw, hw], [c.z, c.z],
                color="k", alpha=0.8,
                linewidth=p.component_lw + 2, zorder=998,
            )
        elif isinstance(c, Lens):
            ax.text(label_x, c.z, name, fontsize=p.label_fontsize, va="center", ha=ha, zorder=1000)
            lens_w = max(1e-12, 2.0 * float(p.component_x) * float(p.lens_radius))
            lens_h = max(1e-12, float(p.lens_height) / float(aspect))
            ax.add_patch(
                mpl.patches.Arc(
                    (0, c.z),
                    lens_w,
                    lens_h,
                    theta1=0,
                    theta2=180,
                    linewidth=1,
                    fill=False,
                    zorder=999,
                    edgecolor="k",
                )
            )
            ax.add_patch(
                mpl.patches.Arc(
                    (0, c.z),
                    lens_w,
                    lens_h,
                    theta1=180,
                    theta2=360,
                    linewidth=1,
                    fill=False,
                    zorder=998,
                    edgecolor="k",
                )
            )
        elif isinstance(c, Detector):
            ax.text(label_x, c.z, name, fontsize=p.label_fontsize, va="center", ha=ha, zorder=1000)
            det_rx = float(c.pixel_size[0] * c.shape[0] / 2.0)
            ax.plot(
                [-det_rx, det_rx], [c.z, c.z],
                color="dimgrey", zorder=1000,
                linewidth=max(3.0, p.component_lw),
            )
        elif isinstance(c, Biprism):
            ax.add_patch(
                plt.Circle(
                    (0, c.z), float(p.biprism_radius),
                    edgecolor="k", facecolor="w", zorder=1000,
                )
            )
            ax.text(label_x, c.z, name, fontsize=p.label_fontsize, va="center", ha=ha, zorder=1000)
        else:
            if hasattr(c, "z"):
                ax.text(
                    label_x,
                    float(getattr(c, "z")),
                    name,
                    fontsize=p.label_fontsize,
                    va="center",
                    ha=ha,
                    zorder=500,
                )

    # Leave room on the right for labels
    fig.subplots_adjust(right=0.7)

    return fig, ax


def plot_ray_bundle(
    ax: mpl.axes.Axes,
    X: np.ndarray,
    Z: np.ndarray,
    p: PlotParams,
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
        Plot styling parameters, including p.band_mode = {"fill", "lines"}.
    """
    # Draw all rays
    ray_lines = ax.plot(
        X, Z[:, None],
        color=p.ray_color,
        linewidth=p.ray_lw,
        alpha=p.ray_alpha,
        zorder=1,
    )

    # Band/envelope rendering
    nrays = X.shape[1] if X.ndim == 2 else 1
    band_artists = []
    edge_lines = []

    if nrays >= 2:
        min_x_idx = int(np.argmin(X[0, :]))
        max_x_idx = int(np.argmax(X[0, :]))
        mode = str(getattr(p, "band_mode", "fill")).lower().strip()

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
        "band_artists": band_artists,
        "edge_lines": edge_lines,
    }


# Functionalized: build X (positions) and Z (z positions) from simulation steps
def _stack_ray_positions(
    steps_seq: Sequence[Tuple[object, Ray]]
) -> Tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    zs: list[float] = []
    for _, r in steps_seq:
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
