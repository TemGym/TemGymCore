"""Regenerate ``ray_images.npz`` next to this file.

It freezes the ray-traced images of ``tem_column.render`` (``setting_column``
then ``render_rays``) on a few rows of the selected ray acquisition at low fidelity
(64x64 pixels, 2**10 rays), so that refactors of the ray-image path (and the virtual
microscope that reuses it) cannot silently change the images. Run from the repository root, on CPU, and
only for an intended, reviewed numerical change::

    JAX_PLATFORMS=cpu PYTHONPATH=src:tests python tests/reference/make_ray_images.py
"""
import json
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from temgym_core.tem_column.render import render_rays, setting_column  # noqa: E402

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
HIDDEN_OPTICS = ROOT / "examples/microscope_models/hidden_optics"
# Projector rows with a shift (200 kV) and a tilt (195 kV); a condenser row per voltage; a check row.
ROWS = (0, 144, 108, 250, 283)


def scenario():
    """Low-fidelity specification, settings and truth used by the saved reference images."""
    spec = json.loads((HIDDEN_OPTICS / "experiment.json").read_text())
    spec.update(shape_yx=[64, 64], pixel_pitch_m=4 * spec["pixel_pitch_m"],
                psf_sigma_pixels=spec["psf_sigma_pixels"] / 4, ray_sample_power=10,
                detector_samples_per_pixel=2)
    settings = dict(np.load(Path(__file__).with_name("ray_image_settings.npz")))
    truth = json.loads((HIDDEN_OPTICS / "optics.json").read_text())
    return spec, settings, truth


def render_row(spec, settings, truth, row, aperture):
    """Expected electrons per pixel for one recorded setting and aperture (no noise)."""
    vi = int(settings["voltage_index"][row])
    column, mean, covariance, source_z = setting_column(
        truth["condenser_parameters"], truth["projector_parameters"],
        voltage_v=spec["voltages_v"][vi], reference_voltage_v=spec["voltages_v"][0], source_index=vi,
        condenser_controls=settings["condenser_controls"][row], projector_controls=settings["projector_controls"][row],
        specimen_kick=settings["specimen_kick"][row], aperture_index=aperture,
        aperture_diameter_um=spec["aperture_diameters_um"][aperture], aperture_edge_width_m=spec["aperture_edge_width_m"],
        gap_fractions=spec.get("deflector_gap_fractions", (0.3, 0.7)))
    image = render_rays(column, mean, covariance, source_z, spec["voltages_v"][vi], shape_yx=spec["shape_yx"],
                        pixel_pitch_m=spec["pixel_pitch_m"], psf_sigma_pixels=spec["psf_sigma_pixels"],
                        ray_sample_power=spec["ray_sample_power"],
                        detector_samples_per_pixel=spec["detector_samples_per_pixel"])
    return image * spec["nominal_incident_electrons_per_image"]


def render(spec, settings, truth):
    return np.array([[render_row(spec, settings, truth, row, aperture) for aperture in range(2)] for row in ROWS])


if __name__ == "__main__":
    images = render(*scenario())
    np.savez_compressed(HERE / "ray_images.npz", rows=np.array(ROWS), images=images)
    print("saved", images.shape, "total electrons per frame:", images.sum(axis=(-2, -1)).round(1).tolist())
