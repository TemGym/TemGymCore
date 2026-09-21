"""Figure: select a focused convergence angle using the calibrated condenser.

Run directly to use the notebook's saved fit, or call build_figure with the
microscope returned by calibration.fit_microscope. No acquisition or fit is rerun.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from temgym_core.calibration.apertures import (
    aperture_centres,
    effective_parameters,
    make_aperture_moments,
)
from temgym_core.calibration.condenser_model import first_crossover_fraction

HERE = Path(__file__).resolve().parent
COLORS = ["#238779", "#427aaf", "#a86ba5", "#d28a36"]


def build_figure(*, microscope, experiment_directory, output_directory):
    """Return a Matplotlib figure and results; export PNG, PDF, SVG, and JSON.

    The control box is an illustrative operating range, not measured hardware
    limits. Geometry/source/aperture parameters come only from the supplied fit.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable jax_enable_x64 before building the figure.")
    spec = json.loads((Path(experiment_directory) / "experiment.json").read_text())
    voltage = float(spec["voltages_v"][0])
    aperture_index = list(spec["aperture_diameters_um"]).index(50.0)
    parameters = effective_parameters(
        jnp.asarray(microscope["condenser_parameters"]), voltage, voltage, 0
    )
    centre = aperture_centres(parameters, len(spec["aperture_diameters_um"]))[
        aperture_index
    ] * 1e-6
    cl1 = float(spec["condenser_reference_controls"][0])
    lower = jnp.array([1.64, 2.40])
    upper = jnp.array([1.80, 2.65])
    span = upper - lower
    targets = np.array([8., 12., 18., 24.])
    initial = np.array([0.45, 0.70])  # Same normalized currents for every target.
    diffraction = bool(spec.get("aperture_diffraction", False))

    def predictor(radial_order=192, angular_order=64):
        return make_aperture_moments(
            radius_m=25e-6,
            edge_width_m=spec["aperture_edge_width_m"],
            voltage_v=voltage,
            radial_order=radial_order,
            angular_order=angular_order,
            diffraction=diffraction,
        )

    predict = predictor()

    def measures(covariance, flux):
        position_variance = jnp.trace(covariance[:2, :2]) / 2
        angular_variance = jnp.trace(covariance[2:, 2:]) / 2
        cross = jnp.trace(covariance[:2, 2:]) / 2
        # The trace beam radius in a drift is s_x^2 + 2*z*cross + z^2*s_theta^2.
        # Its minimum is at z_waist = -cross / s_theta^2, relative to specimen.
        return jnp.array([
            jnp.sqrt(angular_variance) * 1e3,
            -cross / angular_variance * 1e6,
            jnp.sqrt(position_variance) * 1e9,
            flux,
        ])

    @jax.jit
    def stats(controls):
        _, covariance, flux = predict(parameters, jnp.r_[cl1, controls], centre)
        return measures(covariance, flux)

    @jax.jit
    def loss(normalized, target):
        angle, waist = stats(lower + span * normalized)[:2]
        return 0.5 * (((angle - target) / 10.)**2 + (waist / 10.)**2)

    value_gradient = jax.jit(jax.value_and_grad(loss))
    # Check the autodiff gradient against an independent central difference.
    _, gradient = value_gradient(jnp.asarray(initial), targets[1])
    delta = np.eye(2) * 1e-5
    finite_difference = np.array([
        (float(loss(initial + d, targets[1])) - float(loss(initial - d, targets[1])))
        / 2e-5 for d in delta
    ])
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-5, atol=1e-7)

    runs = []
    for target in targets:
        normalized = initial.copy()
        history = []
        for iteration in range(1001):
            value, gradient = value_gradient(jnp.asarray(normalized), target)
            value, gradient = float(value), np.asarray(gradient)
            controls = np.asarray(lower + span * normalized)
            current_stats = np.asarray(stats(controls))
            history.append(np.r_[controls, current_stats, value])
            if abs(current_stats[0] - target) < 2e-5 and abs(current_stats[1]) < 2e-5:
                break
            if iteration == 1000:
                raise RuntimeError(f"Gradient descent did not reach {target:g} mrad.")
            # Projected steepest descent, with Armijo backtracking. No lens
            # solver or requested-angle lookup is used to initialize these runs.
            step = 1.0
            for _ in range(40):
                trial = np.clip(normalized - step * gradient, 0., 1.)
                trial_value = float(loss(trial, target))
                if trial_value <= value + 1e-4 * np.dot(gradient, trial - normalized):
                    break
                step *= 0.5
            else:
                raise RuntimeError("Gradient-descent line search failed.")
            if np.max(np.abs(trial - normalized)) < 1e-14:
                raise RuntimeError(f"Target {target:g} mrad stalled at a control limit.")
            normalized = trial
        history = np.array(history)
        assert np.all(np.diff(history[:, -1]) <= 1e-12)
        assert np.all(history[:, :2] >= np.asarray(lower))
        assert np.all(history[:, :2] <= np.asarray(upper))
        runs.append(history)

    # Independently trace the waist-at-specimen branch to delimit the angle
    # range. This curve is used for the figure only, never by gradient descent.
    focus_controls = []
    for cl3 in np.linspace(float(lower[0]), float(upper[0]), 121):
        mini = brentq(
            lambda mini: float(stats(jnp.array([cl3, mini]))[1]),
            float(lower[1]), float(upper[1]), xtol=1e-13,
        )
        focus_controls.append([cl3, mini])
    focus_controls = np.array(focus_controls)
    focus_stats = np.array([stats(u) for u in focus_controls])
    assert np.all(np.diff(focus_stats[:, 0]) > 0)
    achievable = focus_stats[[0, -1], 0]
    assert np.all((targets > achievable[0]) & (targets < achievable[1]))
    crossover = float(first_crossover_fraction(parameters, jnp.r_[cl1, lower]))
    assert spec["crossover_fraction_bounds"][0] < crossover < spec["crossover_fraction_bounds"][1]

    # Refine aperture quadrature at all endpoints, including range boundaries.
    fine_predict = predictor(384, 128)
    final_controls = np.array([run[-1, :2] for run in runs])
    check_controls = np.vstack([final_controls, focus_controls[[0, -1]]])
    coarse_stats = np.array([stats(u) for u in check_controls])
    fine_stats = []
    for controls in check_controls:
        _, covariance, flux = fine_predict(parameters, jnp.r_[cl1, controls], centre)
        fine_stats.append(np.asarray(measures(covariance, flux)))
    fine_stats = np.array(fine_stats)
    np.testing.assert_allclose(fine_stats[:, 0], coarse_stats[:, 0], rtol=1e-5)
    np.testing.assert_allclose(fine_stats[:, 1], coarse_stats[:, 1], atol=2e-5, rtol=0)

    x = np.linspace(float(lower[0]), float(upper[0]), 61)
    y = np.linspace(float(lower[1]), float(upper[1]), 61)
    xx, yy = np.meshgrid(x, y)
    grid_controls = jnp.asarray(np.column_stack([xx.ravel(), yy.ravel()]))
    grid = np.asarray(jax.jit(lambda u: jax.lax.map(stats, u, batch_size=16))(
        grid_controls
    )).reshape(len(y), len(x), 4)

    with plt.rc_context({
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlelocation": "left", "svg.fonttype": "none", "pdf.fonttype": 42,
    }):
        fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.0), layout="constrained")
        fig.suptitle("Choose a convergence angle with the calibrated condenser", fontsize=16)
        ax = axes[0, 0]
        field = ax.pcolormesh(xx, yy, grid[:, :, 0], cmap="Greys", shading="auto",
                             vmin=0, vmax=36, rasterized=True)
        fig.colorbar(field, ax=ax, label=r"RMS angle $\alpha$ (mrad)", pad=.025)
        ax.plot(*focus_controls.T, color="white", lw=3.5)
        ax.plot(*focus_controls.T, color="0.25", ls="--", lw=1.2,
                label="Waist at specimen")
        for target, history, color in zip(targets, runs, COLORS):
            ax.plot(*history[:, :2].T, color=color, lw=1.8)
            ax.scatter(*history[-1, :2], color=color, edgecolor="white", s=65, zorder=5)
            ax.annotate(f"{target:g}", history[-1, :2], xytext=(6, 7),
                        textcoords="offset points", color=color, fontweight="bold")
        start = np.asarray(lower + span * initial)
        ax.scatter(*start, marker="*", color="#b74447", edgecolor="white", s=140, zorder=6,
                   label="Common start")
        ax.set(title="a   Lens settings and descent paths", xlabel="CL3 relative current",
               ylabel="Mini condenser relative current", xlim=(x[0], x[-1]), ylim=(y[0], y[-1]))
        ax.legend(loc="upper right", fontsize=8, framealpha=.95)

        ax = axes[0, 1]
        for target, history, color in zip(targets, runs, COLORS):
            ax.axhline(target, color=color, lw=.9, ls="--", alpha=.65)
            ax.plot(history[:, 2], color=color, lw=1.8, label=f"{target:g} mrad")
            ax.scatter(len(history) - 1, history[-1, 2], color=color, s=25)
        ax.set(title="b   Gradient descent selects the angle", xlabel="Gradient-descent iteration",
               ylabel="RMS angle at specimen (mrad)", ylim=(5, 27))
        ax.legend(ncols=2, fontsize=9, frameon=False)
        ax.text(.97, .06, "Dashed lines: requested angles", transform=ax.transAxes,
                ha="right", color="0.4", fontsize=9)

        ax = axes[1, 0]
        for target, history, color in zip(targets, runs, COLORS):
            angle_error = np.abs(history[:, 2] - target)
            ax.semilogy(np.maximum(angle_error, 1e-12), color=color, lw=1.8)
        ax.axhline(2e-5, color="0.5", ls=":", lw=1)
        ax.text(.97, .06, "Angle tolerance: 0.00002 mrad\nWaist tolerance: 0.00002 µm",
                transform=ax.transAxes, ha="right", fontsize=9, color="0.4")
        ax.set(title="c   Gradient descent reduces the angle error",
               xlabel="Gradient-descent iteration", ylabel="Absolute angle error (mrad)")

        ax = axes[1, 1]
        ax.axhspan(*achievable, color="#238779", alpha=.10)
        ax.plot([0, 32], [0, 32], color="0.55", ls="--", lw=1, label="Requested = achieved")
        ax.hlines(achievable, 0, 32, color="#238779", ls=":", lw=1)
        for target, history, color in zip(targets, runs, COLORS):
            ax.scatter(target, history[-1, 2], color=color, s=65, edgecolor="white", zorder=4)
        ax.text(1, achievable[1] + 1., f"Upper limit: {achievable[1]:.2f} mrad", color="#238779", fontsize=9)
        ax.text(1, achievable[0] - 2., f"Lower limit: {achievable[0]:.2f} mrad", color="#238779", fontsize=9)
        ax.text(1, 17, "Focused range within\nthe chosen control limits", color="#238779", fontsize=9)
        ax.set(title="d   Requested and achieved convergence", xlabel="Requested RMS angle (mrad)",
               ylabel="Achieved RMS angle (mrad)", xlim=(0, 32), ylim=(0, 32))
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        for ax in axes.flat:
            ax.tick_params(direction="out")
        fig.supxlabel(
            f"{voltage / 1000:g} kV  ·  50 µm aperture  ·  CL1 fixed at {cl1:.3f}  ·  "
            "objective fixed  ·  " + ("aperture diffraction included" if diffraction else "geometric aperture model"),
            fontsize=10, color="0.35",
        )

        out = Path(output_directory)
        out.mkdir(parents=True, exist_ok=True)
        stem = out / "condenser_convergence_control"
        for extension in ("png", "pdf", "svg"):
            fig.savefig(stem.with_suffix(f".{extension}"), dpi=220, facecolor="white")

    result = {
        "angle_definition": "sqrt((Var(theta_x) + Var(theta_y)) / 2), transmitted beam",
        "focus_definition": "minimum trace spatial covariance in a local drift at specimen",
        "scope": "Fitted paraxial model; chosen control box, not measured hardware limits.",
        "voltage_v": voltage, "aperture_diameter_um": 50., "aperture_diffraction": diffraction,
        "CL1_fixed": cl1, "objective_current_fixed": spec["objective_current"],
        "controls_order": ["CL3", "mini"], "lower": np.asarray(lower).tolist(),
        "upper": np.asarray(upper).tolist(), "initial_controls": start.tolist(),
        "focused_angle_range_mrad": achievable.tolist(),
        "first_crossover_fraction": crossover,
        "loss": "0.5 * (((alpha_mrad - target_mrad) / 10)^2 + (waist_um / 10)^2)",
        "optimizer": "Projected gradient descent in normalized controls; Armijo backtracking",
        "quadrature_max_angle_difference_mrad": float(np.max(np.abs(fine_stats[:, 0] - coarse_stats[:, 0]))),
        "quadrature_max_waist_difference_um": float(np.max(np.abs(fine_stats[:, 1] - coarse_stats[:, 1]))),
        "history_columns": ["CL3", "mini", "angle_mrad", "waist_offset_um", "radius_rms_nm", "transmission", "loss"],
        "runs": [dict(target_mrad=float(target), iterations=len(history)-1,
                      final_controls=history[-1, :2].tolist(),
                      achieved_mrad=float(history[-1, 2]), waist_offset_um=float(history[-1, 3]),
                      transmission=float(history[-1, 5]), history=history.tolist())
                 for target, history in zip(targets, runs)],
        "condenser_parameters": np.asarray(parameters).tolist(),
        "focus_curve_controls": focus_controls.tolist(), "focus_curve_stats": focus_stats.tolist(),
    }
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Focused RMS-angle range in this control box: {achievable[0]:.3f}–{achievable[1]:.3f} mrad")
    for run in result["runs"]:
        print(f"Target {run['target_mrad']:5.1f} → {run['achieved_mrad']:.6f} mrad; "
              f"{run['iterations']:3d} steps; waist offset {run['waist_offset_um']:.2g} µm")
    print(f"Saved figure and numerical results: {stem}")
    return fig, result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path,
                        default=HERE.parents[1] / "generated/microscope_calibration_tilt_y_notebook")
    parser.add_argument("--output", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    plt.switch_backend("Agg")
    fitted = json.loads((args.experiment / "fit.json").read_text())
    build_figure(microscope=fitted, experiment_directory=args.experiment, output_directory=args.output)
