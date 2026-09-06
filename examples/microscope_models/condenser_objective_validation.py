"""Frozen microscope predictions for new images, with fitted uncertainty."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from condenser_objective_experiment import make_pattern_image_operator, predict_pattern


def check_new_objective_images(*, fitted, image_directory):
    """Predict new pixels and pattern measurements without refitting the microscope.

    Per-image exposure/background estimates come from the independent image
    analysis. They are the only measured image parameters used in rendering;
    the microscope predicts all fourteen pattern quantities. No truth is read.
    """
    folder = Path(image_directory)
    data = np.load(folder / "prediction_images.npz")
    measured = np.load(folder / "prediction_measurements.npz")
    detector = json.loads((folder / "detector.json").read_text())
    views = json.loads((folder / "reference_projector.json").read_text())
    projector = jnp.asarray(views["calibrated_projector_parameters"])
    controls, currents = (
        jnp.asarray(data["controls"]),
        jnp.asarray(data["objective_commands"]),
    )
    whitening = jnp.asarray(np.linalg.inv(np.linalg.cholesky(measured["covariance"])))

    def patterns(p):
        response = p[22:] if fitted["fit_post_response"] else jnp.array([1.0, 0.0, 0.0])
        return jax.vmap(
            lambda c, u: predict_pattern(
                p[:22], c, u, projector, detector["sample_kernel_sigma_m"], response
            )
        )(controls, currents)

    def residual(p):
        return jnp.einsum(
            "nij,nj->ni", whitening, patterns(p) - measured["values"]
        ).ravel()

    p = jnp.asarray(fitted["best_fit"]["parameters"])
    errors = np.asarray(residual(p))
    derivative = np.asarray(jax.jacfwd(residual)(p))
    covariance = np.asarray(fitted["covariance"])
    # Full covariance retains shared prediction errors between different states.
    total_covariance = np.eye(len(errors)) + derivative @ covariance @ derivative.T
    combined_errors = np.linalg.solve(np.linalg.cholesky(total_covariance), errors)
    renderer = make_pattern_image_operator(
        projector_matrices=data["projector_matrices"][0],
        template=dict(np.load(folder / "template.npz")),
        detector=detector,
    )
    predicted_patterns = patterns(p)
    predictions, pixel_rms = [], []
    for i, pattern in enumerate(predicted_patterns):
        exposures = measured["image_parameters"][i, 14:]
        predicted = np.asarray(
            renderer(
                jnp.r_[pattern, exposures], jnp.asarray(data["projector_matrices"][i])
            )
        )
        noise = np.sqrt(
            np.maximum(data["images"][i], 1.0) + detector["read_noise_e"] ** 2
        )
        pixel_rms.append(
            float(np.sqrt(np.mean(((predicted - data["images"][i]) / noise) ** 2)))
        )
        predictions.append(predicted)
    np.savez_compressed(
        folder / "frozen_prediction_images.npz",
        predicted_images=predictions,
        objective_commands=data["objective_commands"],
        controls=data["controls"],
    )
    return {
        "pattern_rms_measurement_noise_only": float(np.sqrt(np.mean(errors**2))),
        "pattern_rms_including_fitted_uncertainty": float(
            np.sqrt(np.mean(combined_errors**2))
        ),
        "pixel_rms": float(np.sqrt(np.mean(np.square(pixel_rms)))),
        "pixel_rms_per_state": pixel_rms,
        "aperture_centre_standard_errors_um": np.sqrt(
            np.diag(covariance)[20:22]
        ).tolist(),
        "scope": "Frozen geometry and post-field response; exposure/background taken from independent image analysis. Prediction uncertainty is a local linear approximation, including correlations between states.",
    }


def show_prediction_check(*, validation, fitted, checks, image_directory):
    """Display the frozen-model check and the remaining practical limitations."""
    import matplotlib.pyplot as plt
    from IPython.display import Markdown, display

    folder = Path(image_directory)
    data = np.load(folder / "prediction_images.npz")
    predictions = np.load(folder / "frozen_prediction_images.npz")["predicted_images"]
    detector = json.loads((folder / "detector.json").read_text())
    figure, axes = plt.subplots(2, 3, figsize=(10, 5.5), constrained_layout=True)
    for row, index in enumerate([0, 1]):
        measured, predicted = data["images"][index, 0], predictions[index, 0]
        vmax = np.percentile(measured, 99.5)
        axes[row, 0].imshow(measured, cmap="gray", vmin=0, vmax=vmax)
        axes[row, 1].imshow(predicted, cmap="gray", vmin=0, vmax=vmax)
        noise = np.sqrt(np.maximum(measured, 1.0) + detector["read_noise_e"] ** 2)
        axes[row, 2].imshow(
            (measured - predicted) / noise, cmap="coolwarm", vmin=-3, vmax=3
        )
        for ax, title in zip(
            axes[row], ["New image", "Frozen prediction", "Residual / noise"]
        ):
            ax.set_title(
                f"{100 * data['objective_commands'][index]:.0f}% objective\n{title}",
                fontsize=10,
            )
            ax.set_axis_off()
    figure.savefig(folder / "objective_prediction_check.png", dpi=160)
    display(figure)
    plt.close(figure)
    centre = np.asarray(fitted["best_fit"]["parameters"])[20:22]
    uncertainty = validation["aperture_centre_standard_errors_um"]
    truth = {row["parameter"]: row["truth"] for row in checks["parameter_comparison"]}
    converged = sum(row["success"] for row in fitted["starts"])
    display(
        Markdown(
            f"**New-image pixel RMS: {validation['pixel_rms']:.3f}.** "
            f"The measured-pattern RMS is {validation['pattern_rms_measurement_noise_only']:.3f}; "
            f"including the fit's uncertainty gives {validation['pattern_rms_including_fitted_uncertainty']:.3f}.\n\n"
            "**The lateral aperture centre remains weakly measured:** "
            f"$x={centre[0]:.3f}\\pm{uncertainty[0]:.3f}$ µm, "
            f"$y={centre[1]:.3f}\\pm{uncertainty[1]:.3f}$ µm (local one-standard-error estimates). "
            f"The hidden centre was $({truth['aperture_x_um']:.3f},{truth['aperture_y_um']:.3f})$ µm. "
            "The axial lens geometry is much better recovered.\n\n"
            f"{converged}/{len(fitted['starts'])} starts converged in this run. "
            "This is evidence that the objective sweep removes the old ambiguity within the model, not a completed global search."
        )
    )
