import matplotlib.pyplot as plt
import jax.numpy as jnp


def plot_model(model, ray_histories, ax=None):
    """Plot the optical system and rays passing through it.

    Parameters:
        model: list of components with .z attribute.
        ray_histories: list of ray history lists. Each history is a list of Ray objects.
        ax: matplotlib Axes to plot on. If None, creates a new figure and axes.

    Returns:
        ax: matplotlib Axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
    # Compute x-limits from all ray histories
    xs_all = [ray.x for ray in ray_histories]
    xmin, xmax = jnp.min(jnp.array(xs_all)), jnp.max(jnp.array(xs_all))
    x_margin = (xmax - xmin) * 0.2 if xmax != xmin else 1.0
    xmin -= x_margin
    xmax += x_margin

    # Plot each component as a horizontal line with label
    z_vals = [float(c.z) for c in model if hasattr(c, "z")]
    z_min, z_max = min(z_vals), max(z_vals)
    z_margin = (z_max - z_min) * 0.02 if z_max != z_min else 1.0
    x_offset = (xmax - xmin) * 0.03 if xmax != xmin else 1.0
    line_length = xmax - xmin
    max_label_width = line_length * 0.25  # 25% of the line length for label text

    for idx, comp in enumerate(model):
        try:
            z_val = float(comp.z)
        except Exception:
            continue
        ax.hlines(y=z_val, xmin=xmin, xmax=xmax, colors="black", linewidth=2)
        label = comp.__class__.__name__
        # All labels on the right, with extra offset
        x_text = xmax + x_offset * 2
        ha = "left"
        ax.text(x_text, z_val, label, va="center", ha=ha, color="black")

        # Plot back focal plane for lenses
        if hasattr(comp, "focal_length"):
            try:
                z_bfp = float(comp.z) + float(comp.focal_length)
                # Only draw the BFP line from xmin to xmax
                ax.hlines(
                    y=z_bfp,
                    xmin=xmin,
                    xmax=xmax,
                    colors="blue",
                    linestyle="dashed",
                    linewidth=1,
                )
                # Move BFP label further right to avoid overlap
                ax.text(
                    xmax + x_offset * 2,
                    z_bfp,
                    "Back Focal Plane",
                    va="center",
                    ha="left",
                    color="blue",
                    fontsize=8,
                    alpha=0.7,
                )
                z_ffp = float(comp.z) - float(comp.focal_length)
                # Only draw the BFP line from xmin to xmax
                ax.hlines(
                    y=z_ffp,
                    xmin=xmin,
                    xmax=xmax,
                    colors="blue",
                    linestyle="dashed",
                    linewidth=1,
                )
                # Move BFP label further right to avoid overlap
                ax.text(
                    xmax + x_offset * 2,
                    z_ffp,
                    "Back Focal Plane",
                    va="center",
                    ha="left",
                    color="blue",
                    fontsize=8,
                    alpha=0.7,
                )
            except Exception:
                pass

    # Plot ray paths
    for history in ray_histories:
        xs = [ray.x for ray in ray_histories]
        zs = [ray.z for ray in ray_histories]
        ax.plot(xs, zs, color="green")

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("Ray Paths Through Optical System")
    ax.invert_yaxis()
    # Remove top, left, and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax
