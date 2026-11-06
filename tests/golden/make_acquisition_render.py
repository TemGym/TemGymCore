"""Regenerate ``acquisition_render.npz`` next to this file.

It freezes ``calibration.acquisition.render_ray_image`` on a few rows of the
selected ray acquisition at low fidelity (64x64 pixels, 2**10 rays), so that
refactors of the ray-image path (and the virtual microscope that reuses it)
cannot silently change the images. Run from the repository root, on CPU, and
only for an intended, reviewed numerical change::

    JAX_PLATFORMS=cpu PYTHONPATH=src:tests python tests/golden/make_acquisition_render.py
"""
import json
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from temgym_core.calibration.acquisition import render_ray_image  # noqa: E402

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
ACQUISITION = ROOT / "examples/microscope_models/calibration_results/tilt_y_30_ray_acquisition"
# Projector rows with a shift (200 kV) and a tilt (195 kV); a condenser row per voltage; a check row.
ROWS = (0, 144, 108, 250, 283)


def scenario():
    """Low-fidelity specification, settings and truth used by the golden images."""
    spec = json.loads((ACQUISITION / "experiment.json").read_text())
    spec.update(shape_yx=[64, 64], pixel_pitch_m=4 * spec["pixel_pitch_m"],
                psf_sigma_pixels=spec["psf_sigma_pixels"] / 4, ray_sample_power=10,
                detector_samples_per_pixel=2)
    settings = dict(np.load(ACQUISITION / "settings.npz"))
    truth = json.loads((ACQUISITION / "scoring_truth.json").read_text())
    return spec, settings, truth


def render(spec, settings, truth):
    return np.array([[render_ray_image(spec, settings, truth, row, aperture)
                      for aperture in range(2)] for row in ROWS])


if __name__ == "__main__":
    images = render(*scenario())
    np.savez_compressed(HERE / "acquisition_render.npz", rows=np.array(ROWS), images=images)
    print("saved", images.shape, "total electrons per frame:", images.sum(axis=(-2, -1)).round(1).tolist())
