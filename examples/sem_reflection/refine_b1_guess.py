"""Fit the saved reflection calibration from the ERC B1 figure's initial guess.

The figure's guess is chosen to show large lens errors while its Bragg spots
stay on the detector; none of the saved multistarts does both. This script
runs the same two stages as reflection_calibration.fit_microscope (camera
initialisation from the data, coarse shape fit, full refinement) from that one
start, and records whether it reaches the saved best fit. Truth is not read
during fitting; it is only compared afterwards.
"""
from pathlib import Path
import json
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from temgym_core import reflection_calibration as calibration
from temgym_core.optimization import least_squares

DATA = Path(__file__).resolve().parent / "output/calibration_gaussian"
GUESS = np.array([-385., -235., -105., -25., 45., 30., 20., 2.5, 140.])
OUT = DATA / "b1_figure_start.json"


def main():
    spec = json.loads((DATA / "experiment.json").read_text())
    settings = dict(np.load(DATA / "settings.npz"))
    with np.load(DATA / "measurements.npz") as data:
        observed, covariance, strips = (data[k] for k in ("observations", "covariance", "strips"))
    use = settings["purpose"] == "fit"
    predict, _ = calibration.make_predictor(spec)
    controls, dz, strips_fit = (jnp.asarray(a[use]) for a in
                                (settings["controls"], settings["detector_dz_m"], strips))
    whiten = jnp.asarray(np.linalg.inv(np.linalg.cholesky(covariance[use])))
    target = jnp.asarray(observed[use])

    def residual(p):
        difference = predict(p, controls, dz, strips_fit) - target
        return jnp.einsum("...ij,...j->...i", whiten, difference).ravel()

    def coarse_residual(p):
        predicted = predict(p, controls, dz, strips_fit)
        return jnp.concatenate((.1 * (predicted[..., :2] - target[..., :2]).ravel(),
            (jnp.log(jnp.maximum(predicted[..., 2:], .1)) -
             jnp.log(jnp.maximum(target[..., 2:], .1))).ravel()))

    # Camera initialisation from measured order positions, as in fit_microscope.
    _, _, slopes = calibration.order_geometry(spec)
    response = slopes[:, 1] * 1e-3 / spec["pixel_pitch_m"]
    motion = (settings["detector_dz_m"][use, None] * np.cos(np.deg2rad(spec["incidence_deg"]))
              * slopes[:, 1] / spec["pixel_pitch_m"])
    weights = 1 / covariance[use, :, 1, 1]
    camera = np.sum(weights * response * (observed[use, :, 1] - motion)) / np.sum(weights * response ** 2)
    start = GUESS.copy()
    start[8] = float(np.clip(camera, calibration.LOWER[8], calibration.UPPER[8]))

    bounds = (calibration.LOWER, calibration.UPPER)
    coarse = least_squares(coarse_residual, start, bounds=bounds, x_scale=calibration.SCALE,
                           max_iter=250, max_nfev=500, ftol=1e-8, geodesic=True)
    result = least_squares(residual, np.asarray(coarse.x), bounds=bounds, x_scale=calibration.SCALE,
                           max_iter=250, max_nfev=500, ftol=1e-9, xtol=1e-9, gtol=1e-6,
                           geodesic=True)
    fitted = np.asarray(result.x)
    saved = json.loads((DATA / "fit.json").read_text())["best"]
    same = bool(np.allclose(fitted, saved["parameters"], rtol=1e-4))
    record = dict(initial=GUESS.tolist(), refinement_start=np.asarray(coarse.x).tolist(),
                  parameters=fitted.tolist(), chi_squared=2 * float(result.cost),
                  saved_best_chi_squared=saved["chi_squared"], success=bool(result.success),
                  status=result.status, reaches_saved_best_fit=same,
                  note="Same stages and tolerances as reflection_calibration.fit_microscope")
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print(f"chi2 {record['chi_squared']:.3f} (saved {saved['chi_squared']:.3f}); "
          f"reaches saved best fit: {same}; {result.status}")


if __name__ == "__main__":
    main()
