"""evaluate.ray_image: rays traced, weighted by transmission and counted in pixels."""
import jax
import numpy as np

from temgym_core.components import Detector, Plane, SigmoidAperture
from temgym_core.evaluate import ray_image
from temgym_core.source import sample_rays

jax.config.update("jax_enable_x64", True)


def test_ray_image_counts_every_ray_and_matches_the_simple_counter():
    rays = sample_rays(np.zeros(4), np.diag([4e-12, 4e-12, 1e-10, 1e-10]), z=0.0, voltage=200e3, sample_power=12)
    detector = Detector(z=0.1, shape=(32, 32), pixel_size=(1e-6, 1e-6))
    image = ray_image(rays, (Plane(z=0.1),), detector)
    assert image.sum() == np.float64(1.0)                       # every ray lands, each counts 1/N
    at = jax.tree_util.tree_map(np.asarray, rays)
    moved = at.__class__(*(np.asarray(v) for v in (at.x + 0.1 * at.dx, at.y + 0.1 * at.dy, at.dx, at.dy)),
                         z=0.1, pathlength=0.0, voltage=200e3)
    counts = detector.into_image(moved)
    np.testing.assert_allclose(image * 2 ** 12, counts)


def test_ray_image_weights_rays_by_aperture_transmission_and_conserves_flux_under_blur():
    rays = sample_rays(np.zeros(4), np.diag([4e-12, 4e-12, 0.0, 0.0]) + 1e-30 * np.eye(4), z=0.0,
                         voltage=200e3, sample_power=12)
    detector = Detector(z=0.1, shape=(32, 32), pixel_size=(1e-6, 1e-6))
    open_ = ray_image(rays, (Plane(z=0.1),), detector, samples_per_pixel=4, psf_sigma_pixels=1.0)
    clipped = ray_image(rays, (SigmoidAperture(z=0.05, radius=1e-6, edge_width=1e-8), Plane(z=0.1)), detector)
    assert abs(open_.sum() - 1) < 1e-12                         # the blur keeps every ray's share
    assert 0.05 < clipped.sum() < 0.2                           # about the share inside one micron


def test_position_and_angle_response_of_a_thin_lens_imaging_one_to_one():
    """A thin lens of focal length f, object and image at 2f: A = -1 (inverted image), B = 0."""
    import pytest

    from temgym_core.components import Lens
    from temgym_core.run import position_and_angle_response

    f = 0.01
    a, b = position_and_angle_response((Lens(z=2 * f, focal_length=f), Plane(z=4 * f)))
    assert complex(a) == pytest.approx(-1.0, abs=1e-12)
    assert abs(complex(b)) < 1e-12
