"""Replay saved gradient-descent steps with native TemGym ray plotting."""

from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from temgym_core.calibration.apertures import aperture_centres, make_aperture_moments
from temgym_core.calibration.condenser_model import condenser_with_aperture, source_moments
from temgym_core.plotting import (
    ColumnElement, _draw_column_element, legacy_beam_plot_params,
    plot_ray_bundle, trace_ray_envelope,
)
from temgym_core.ray import Ray


class ConvergenceRayViewer:
    """A cached ray diagram for the actual accepted optimization iterates.

    Native envelopes use nominal-radius clipping, as in the calibration image
    browser. The specimen zoom uses the optimizer's smooth-aperture covariance.
    """

    def __init__(self, *, results, experiment_directory):
        self.results = results
        self.spec = json.loads((Path(experiment_directory) / "experiment.json").read_text())
        self.parameters = jnp.asarray(results["condenser_parameters"])
        self.voltage = results["voltage_v"]
        self.cl1 = results["CL1_fixed"]
        index = list(self.spec["aperture_diameters_um"]).index(results["aperture_diameter_um"])
        self.centre = aperture_centres(
            self.parameters, len(self.spec["aperture_diameters_um"])
        )[index] * 1e-6
        self.predict = make_aperture_moments(
            radius_m=results["aperture_diameter_um"] * 0.5e-6,
            edge_width_m=self.spec["aperture_edge_width_m"],
            voltage_v=self.voltage, diffraction=results["aperture_diffraction"],
        )
        mean, covariance = map(np.asarray, source_moments(self.parameters, self.voltage))
        self.source_z = -float(jnp.exp(self.parameters[:9]).sum())
        # Same two-RMS meridional source bundle as the calibration browser.
        radius, phase = np.meshgrid(np.linspace(0, 2, 17), np.linspace(0, 2*np.pi, 96, endpoint=False))
        self.rays = Ray(
            x=mean[0] + np.sqrt(covariance[0, 0]) * (radius*np.cos(phase)).ravel(),
            y=mean[1], dx=mean[2] + np.sqrt(covariance[2, 2]) * (radius*np.sin(phase)).ravel(),
            dy=mean[3], z=self.source_z, pathlength=0., voltage=self.voltage,
        )
        self.target_values = [float(run["target_mrad"]) for run in results["runs"]]
        self.zoom_z = np.linspace(-25e-6, 25e-6, 401)
        # Determine common limits from all recorded states. They stay fixed
        # across targets and iterations, including transient overshoots.
        states = np.vstack([run["history"] for run in results["runs"]])
        self.zoom_width = 1.15 * max(
            (abs(states[:, 3])*1e-6 + abs(self.zoom_z).max()) * states[:, 2]*1e-3
            + states[:, 4]*1e-9
        )
        self._widgets = None

    @lru_cache(maxsize=256)
    def trace(self, cl3, mini):
        controls = jnp.array([self.cl1, cl3, mini])
        column = condenser_with_aperture(
            self.parameters, controls,
            radius_m=self.results["aperture_diameter_um"] * 0.5e-6,
            edge_width_m=self.spec["aperture_edge_width_m"], centre_m=self.centre,
        )
        segments = trace_ray_envelope(column, self.rays)
        _, covariance, _ = self.predict(self.parameters, controls, self.centre)
        return column, segments, np.asarray(covariance)

    @staticmethod
    def _visible_segments(segments, lower, upper):
        """Combine continuous native segments, preserving jumps at clipping."""
        groups = []
        for z, bounds in segments:
            lo, hi = max(lower, z[0]), min(upper, z[-1])
            if hi <= lo:
                continue
            zz = np.unique(np.r_[lo, z[(z > lo) & (z < hi)], hi])
            xx = np.column_stack([np.interp(zz, z, b) for b in bounds.T])
            if groups and np.isclose(groups[-1][0][-1], zz[0], atol=1e-14, rtol=0) and np.allclose(groups[-1][1][-1], xx[0], atol=1e-14, rtol=0):
                groups[-1] = (np.r_[groups[-1][0], zz[1:]], np.vstack([groups[-1][1], xx[1:]]))
            else:
                groups.append((zz, xx))
        return groups

    def draw(self, axes, run_index, iteration):
        run = self.results["runs"][run_index]
        history = np.asarray(run["history"])
        state = history[iteration]
        cl3, mini, angle, waist = state[:4]
        target = run["target_mrad"]
        column, segments, covariance = self.trace(float(cl3), float(mini))
        style = legacy_beam_plot_params(
            fill_color="#238779", fill_alpha=.30, ray_color="#238779", edge_lw=.9,
        )
        labels = ["CL1 · fixed", f"CL3 · {cl3:.4f}", "50 µm aperture", f"Mini · {mini:.4f}", "Objective · fixed", "Specimen"]
        roles = ["condenser_lens", "condenser_lens", "aperture", "condenser_lens", "objective_lens", "sample"]
        # Fixed physical ranges, wide enough for every recorded descent state.
        views = [(self.source_z, 0., 220., "Full condenser"),
                 (float(column[3].z) - .002, 0., 220., "Mini to specimen")]
        for ax, (lo, hi, width, title) in zip(axes[:2], views):
            for z, bounds in self._visible_segments(segments, lo, hi):
                plot_ray_bundle(ax, bounds*1e6, z*1e3, style)
            ax.axvline(0, color=".8", lw=.6, zorder=-1)
            for component, label, role in zip(column, labels, roles):
                mid = float(component.z + getattr(component, 'length', 0.)/2)
                if lo-1e-12 <= mid <= hi+1e-12:
                    end = getattr(component, 'length', 0.)
                    element = ColumnElement(
                        z=float(component.z)*1e3, label=label, role=role,
                        z_end=float(component.z + end)*1e3 if end else None,
                    )
                    _draw_column_element(ax, element, z_span=(hi-lo)*1e3)
                    # In the overview CL3/aperture and mini/objective are close;
                    # horizontal labels fit in reserved right-hand space.
            visible = [(float(c.z + getattr(c, 'length', 0.)/2)*1e3, label)
                       for c, label in zip(column, labels)
                       if lo-1e-12 <= float(c.z + getattr(c, 'length', 0.)/2) <= hi+1e-12]
            from temgym_core.plotting import _spread_label_positions
            positions = _spread_label_positions([z for z, _ in visible], lo*1e3, hi*1e3)
            for (z, label), label_z in zip(visible, positions):
                ax.annotate(label, (width*.30, z), xytext=(width*.65, label_z),
                            fontsize=8, va="center", bbox=dict(facecolor="white", edgecolor="none", alpha=.8, pad=1), arrowprops=dict(arrowstyle="-", lw=.5, color=".6"))
            ax.set(title=title, xlabel="x (µm)", ylabel="z from specimen (mm)",
                   xlim=(-width, width), ylim=((hi+.015*(hi-lo))*1e3, lo*1e3))
            ax.spines[["top", "right"]].set_visible(False)

        ax = axes[2]
        angular_variance = np.trace(covariance[2:, 2:])/2
        cross = np.trace(covariance[:2, 2:])/2
        spatial_variance = np.trace(covariance[:2, :2])/2
        # Complete the square to resolve a small waist without subtracting
        # nearly equal large spatial terms at every display point.
        waist_m = -cross/angular_variance
        minimum_variance = max(spatial_variance-cross**2/angular_variance, 0.)
        z = np.unique(np.r_[self.zoom_z, waist_m])
        radius = np.sqrt(minimum_variance + angular_variance*(z-waist_m)**2)
        target_radius = abs(z) * target * 1e-3
        plot_ray_bundle(ax, np.column_stack([-radius, radius])*1e6, z*1e6, style)
        ax.plot(-target_radius*1e6, z*1e6, color=".4", ls="--", lw=1)
        ax.plot(target_radius*1e6, z*1e6, color=".4", ls="--", lw=1, label="Requested RMS cone")
        ax.axhline(0, color="#a35b91", lw=1.4, label="Specimen")
        ax.axhline(waist_m*1e6, color="#238779", ls=":", lw=1.2, label="Current waist")
        ax.set(title="Specimen zoom · ±1 RMS radius", xlabel="Radius about beam centre (µm)",
               ylabel="z from specimen (µm)", xlim=(-self.zoom_width*1e6, self.zoom_width*1e6),
               ylim=(25., -25.))
        ax.legend(loc="lower right", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        return f"Target {target:g} mrad  |  Step {iteration}/{len(history)-1}  |  Angle {angle:.5f} mrad  |  Waist offset {waist:+.5f} µm"

    def figure(self, run_index, iteration):
        with plt.ioff():
            fig, axes = plt.subplots(1, 3, figsize=(13, 5.7), layout="constrained")
            summary = self.draw(axes, run_index, iteration)
            fig.suptitle(summary, fontsize=12)
        return fig

    @lru_cache(maxsize=256)
    def png(self, run_index, iteration):
        fig = self.figure(run_index, iteration)
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=105, facecolor="white")
        plt.close(fig)
        return buffer.getvalue()

    def widget(self, initial_target=24.):
        import ipywidgets as widgets

        index = self.target_values.index(initial_target)
        target = widgets.Dropdown(options=[(f"{value:g} mrad", i) for i, value in enumerate(self.target_values)], value=index, description="Target")
        last = len(self.results["runs"][index]["history"])-1
        step = widgets.IntSlider(value=0, min=0, max=last, description="Iteration", continuous_update=False, layout=widgets.Layout(width="430px"))
        play = widgets.Play(value=0, min=0, max=last, interval=1000, description="Play")
        link = widgets.jslink((play, "value"), (step, "value"))
        finish = widgets.Button(description="Final setting", icon="step-forward")
        restart = widgets.Button(description="Restart", icon="undo")
        image = widgets.Image(value=self.png(index, 0), format="png", layout=widgets.Layout(width="100%"))
        status = widgets.HTML("Choose a target and press Play, or move the iteration slider.")
        busy = False

        def update(change=None):
            if busy:
                return
            status.value = f"Tracing step {step.value}…"
            try:
                image.value = self.png(target.value, step.value)
                run = self.results["runs"][target.value]
                row = run["history"][step.value]
                status.value = (f"<b>CL3 {row[0]:.5f} · Mini {row[1]:.5f}</b> &nbsp; "
                                f"Angle error {abs(row[2]-run['target_mrad']):.5g} mrad · "
                                f"Waist {row[3]:+.5g} µm from specimen")
            except Exception:
                status.value = "Ray update failed; rerun the cell to see the traceback."
                play.playing = False
                raise

        def select_target(change):
            nonlocal busy
            busy = True
            play.playing = False
            play.value = step.value = 0
            play.max = step.max = len(self.results["runs"][target.value]["history"])-1
            busy = False
            update()

        target.observe(select_target, names="value")
        step.observe(update, names="value")
        def jump(value):
            play.playing = False
            play.value = step.value = value

        finish.on_click(lambda button: jump(step.max))
        restart.on_click(lambda button: jump(0))
        panel = widgets.VBox([
            widgets.HBox([target, play, step]), widgets.HBox([restart, finish]), status, image,
        ])
        self._widgets = dict(target=target, step=step, play=play, image=image, status=status, link=link)
        return panel

    def save_comparison(self, output_directory, target=24.):
        index = self.target_values.index(target)
        last = len(self.results["runs"][index]["history"])-1
        with plt.ioff():
            fig, axes = plt.subplots(2, 3, figsize=(13, 10), layout="constrained")
            for row, iteration in enumerate((0, last)):
                self.draw(axes[row], index, iteration)
                for ax in axes[row]:
                    ax.set_title(("Initial · " if row == 0 else "Optimized · ") + ax.get_title(), fontsize=10)
            fig.suptitle(f"Gradient descent reshapes the beam to {target:g} mrad RMS at the specimen", fontsize=14)
            first = self.results["runs"][index]["history"][0]
            fig.supxlabel(f"Start: {first[2]:.3f} mrad, waist {first[3]:+.3f} µm  →  Final: {target:g} mrad, waist at specimen\n"
                          "Left: native clipped ray bundle · right: smooth-aperture RMS envelope · fixed axes for comparison", fontsize=10)
            stem = Path(output_directory) / "condenser_ray_evolution"
            for suffix in ('png', 'pdf'):
                fig.savefig(stem.with_suffix('.'+suffix), dpi=160, facecolor='white')
            plt.close(fig)
        return stem.with_suffix('.png')

    def save_animation(self, output_directory, target=24.):
        """Export every accepted iterate as a GIF, viewable without a kernel."""
        from PIL import Image

        index = self.target_values.index(target)
        count = len(self.results["runs"][index]["history"])
        frames = [Image.open(BytesIO(self.png(index, step))).convert("RGB")
                  for step in range(count)]
        path = Path(output_directory) / "condenser_ray_evolution.gif"
        frames[0].save(path, save_all=True, append_images=frames[1:],
                       duration=[800] + [500]*(count-2) + [2500], loop=0)
        return path
