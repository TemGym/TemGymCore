"""Stage 2 (draft) figures: the column drawn vertically with lenses as rings and the beam as a
coloured band, and the two specimen frames. In the style of :mod:`figures`.
Uses TemGym: :func:`projection_lenses.column` and ``temgym_core.plotting`` to trace and draw rays."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display

from stage2 import stage1_fit as F
from stage2.stage1_figures import INK, INK2, styled
from stage2.joint_fit import positions_mm


FILL = {"truth": "#43C78E", "orange": "#F5A35C", "blue": "#6BAED6"}


COLUMN_LABELS = ("Objective", "IL1", "IL2", "IL3", "PL1", "Camera")


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
    from stage2 import projection_lenses as lenses
    from temgym_core.plotting import legacy_beam_plot_params, plot_model, trace_ray_paths
    from temgym_core.ray import Ray

    column = lenses.column(lenses.at_voltage(jnp.asarray(theta[F.LENS]), voltage, reference), jnp.asarray(F.controls(dac)))
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


@styled
def column_panels(*, panels, reference, camera, truth=None, height=7.5, sizes=True):
    """Several :func:`column_panel` side by side on one scale. ``panels`` holds
    (title, theta, DAC values, HT, title colour, beam colour); with ``truth`` the true lens and
    camera planes are dashed in every panel that is not the truth itself."""
    fig = plt.figure(figsize=(4.4 * len(panels), height))
    bottom = 1.03 * max(positions_mm(theta)[-1] / 1e3 for _, theta, *_ in panels)
    half_width = camera.n * camera.pitch / 2
    planes = () if truth is None else list(positions_mm(truth) / 1e3)
    width = 1.0 / len(panels)
    for k, (title, theta, dac, voltage, colour, fill) in enumerate(panels):
        ax = fig.add_axes([0.08 + k * width, 0.04, 0.55 * width, 0.86])
        column_panel(ax=ax, theta=theta, dac=dac, voltage=voltage, reference=reference, xmax=1.06 * half_width,
                     bottom=bottom, camera_half_width=half_width, title=title, colour=colour, fill=fill,
                     truth_planes=planes if theta is not truth else (), sizes=sizes)
    return fig


@styled
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
