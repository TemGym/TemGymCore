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
    fill_color: str = "#87cefa"  # light sky blue
    fill_alpha: float = 0.20
    edge_lw: float = 1.8
    component_lw: float = 3.0
    lens_height: float = 0.03  # relative to figure height
    biprism_radius: float = 0.001  # radius of circle to draw biprism


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
    band_mode: str = "fill",  # "fill" (envelope fill) or "lines" (draw lines between rays)
    yscale: str = "linear",   # "linear", "log", or "symlog"
    y_linthresh: float = 1e-6,  # linthresh for symlog
):
    """Plot a 2D schematic of a model (components vs z) with ray bundle.

    Parameters
    ----------
    components : sequence of Source or Component
        Model elements ordered by increasing z.
    rays : Ray, optional
        A Ray or a Ray bundle to use as the starting input. If provided,
        this overrides `initial_ray` and any Source-based generation.
    initial_ray : Ray, optional
        Back-compat alias for explicitly providing the starting ray/bundle.
        Ignored if `rays` is provided.
    num_rays : int, default 101
        Number of rays to generate if using a Source.
    random : bool, default False
        Whether to randomize sampling when using a Source.
    plot_params : PlotParams, optional
        Style parameters for the plot.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, a new figure and axes are created.
    band_mode : {"fill", "lines"}, default "fill"
        "fill": fill between the two edge rays (envelope).
        "lines": draw horizontal line segments between adjacent rays at each z.
    yscale : {"linear", "log", "symlog"}, default "linear"
        Set y-axis (z) scaling. "log" requires all z>0; otherwise falls back to "symlog".
    y_linthresh : float, default 1e-6
        Linear range around zero used when yscale="symlog".

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    p = plot_params

    # Prefer explicitly provided rays/bundle; then legacy initial_ray; else Source/auto
    ray0 = rays if rays is not None else _ensure_initial_ray(
        components, initial_ray, num_rays, random
    )

    # Accumulate rays after each step (including propagations)
    steps: list[Tuple[object, Ray]] = list(run_iter(ray0, components))

    X, Z = _stack_ray_positions(steps)

    if X.size == 0:
        # Nothing to plot
        if ax is None:
            fig, ax = plt.subplots(figsize=p.figsize)
        else:
            fig = ax.figure
        return fig, ax

    # Determine x extent using both beam and detector width if present
    max_beam_x = float(np.max(np.abs(X)))
    component_x = max_beam_x * 1.3
    detector_range_x = 0.0
    for c in components:
        if isinstance(c, Detector):
            # Width in x direction
            detector_range_x = max(
                detector_range_x,
                float(c.pixel_size[0] * c.shape[0] / 2.0),
            )
    max_x = max(component_x, detector_range_x)

    # z limits and ticks
    comp_zs = [float(getattr(c, "z")) for c in components if hasattr(c, "z")]
    min_z = min([float(np.min(Z))] + comp_zs) if comp_zs else float(np.min(Z))
    max_z = max([float(np.max(Z))] + comp_zs) if comp_zs else float(np.max(Z))

    extent = p.extent_scale * max_x

    if ax is None:
        fig, ax = plt.subplots(figsize=p.figsize)
    else:
        fig = ax.figure

    # Style
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=10)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(color="lightgrey", linestyle="--", linewidth=0.5)
    ax.grid(which="minor", color="#EEEEEE", linestyle=":", linewidth=0.5)

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
    yticks = sorted(set([float(np.min(Z)), float(np.max(Z))] + comp_zs))
    if scale == "log":
        yticks = [t for t in yticks if t > 0]
    ax.set_yticks(yticks)
    ax.set_xlim([-max_x, max_x])
    ax.set_ylim([max_z, min_z])  # invert z-axis (optical drawings convention)

    # Rays
    if len(components) > 0:
        ax.text(
            extent,
            comp_zs[0] if comp_zs else float(Z[0]),
            _as_name(components[0]),
            fontsize=p.label_fontsize,
            va="center",
            zorder=1000,
        )
    _ = plot_ray_bundle(ax, X, Z, p, band_mode=band_mode)

    # Components
    aspect = p.figsize[1] / p.figsize[0]
    for c in components:
        name = _as_name(c)
        if isinstance(c, Deflector):
            radius = -component_x
            ax.text(extent, c.z, name, fontsize=p.label_fontsize, va="center", zorder=1000)
            ax.plot(
                [-radius, 0], [c.z, c.z], color="lightcoral",
                linewidth=p.component_lw, zorder=999,
            )
            ax.plot(
                [0, radius], [c.z, c.z], color="lightblue",
                linewidth=p.component_lw, zorder=999,
            )
            ax.plot(
                [-radius, radius], [c.z, c.z], color="k", alpha=0.8,
                linewidth=p.component_lw + 2, zorder=998,
            )
        elif isinstance(c, Lens):
            radius = -component_x * 2
            ax.text(extent, c.z, name, fontsize=p.label_fontsize, va="center", zorder=1000)
            ax.add_patch(
                mpl.patches.Arc(
                    (0, c.z), radius, height=p.lens_height / aspect,
                    theta1=0, theta2=180, linewidth=1,
                    fill=False, zorder=999, edgecolor="k",
                )
            )
            ax.add_patch(
                mpl.patches.Arc(
                    (0, c.z), radius, height=p.lens_height / aspect,
                    theta1=180, theta2=0, linewidth=1,
                    fill=False, zorder=-1, edgecolor="k",
                )
            )
        elif isinstance(c, Detector):
            ax.text(extent, c.z, name, fontsize=p.label_fontsize, va="center", zorder=1000)
            det_rx = float(c.pixel_size[0] * c.shape[0] / 2.0)
            ax.plot([-det_rx, det_rx], [c.z, c.z], color="dimgrey", zorder=1000, linewidth=5)
        elif isinstance(c, Biprism):
            ax.add_patch(plt.Circle((0, c.z), p.biprism_radius, edgecolor="k", facecolor="w", zorder=1000))
        else:
            # Generic annotation at z
            if hasattr(c, "z"):
                ax.text(
                    extent, float(getattr(c, "z")), name,
                    fontsize=p.label_fontsize, va="center", zorder=500,
                )

    plt.subplots_adjust(right=0.7)
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
        mode = str(band_mode).lower().strip()

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
