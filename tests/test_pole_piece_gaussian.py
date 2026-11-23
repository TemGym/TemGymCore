"""PolePieceLens with Gaussian beams: the ray map for the centre, the ABCD law for the shape."""

import jax.numpy as jnp
import numpy as np

from temgym_core.components import Detector, PolePieceLens
from temgym_core.evaluate import evaluate_gaussians
from temgym_core.ray import Ray
from temgym_core.run import run_to_end
from temgym_core.source import decompose_gaussian, gaussian_waist

LENS = PolePieceLens(z=0.05, turns=500.0, current=2.0, gap=0.008, bore_radius=0.003,
                     x0=2e-6, y0=-1e-6)


def test_a_beamlet_centre_and_path_follow_the_ray_map():
    beam = gaussian_waist(z=0.0, voltage=200e3, rms_size=1e-6, x=3e-6, y=-2e-6,
                          theta_x=1e-5, theta_y=2e-5)
    out = LENS(run_to_end(beam, []).derive(z=0.05))
    ray = LENS(Ray(x=3e-6, y=-2e-6, dx=1e-5, dy=2e-5, z=0.05, pathlength=0.0, voltage=200e3))
    for name in ("x", "y", "dx", "dy", "pathlength"):
        assert np.allclose(np.asarray(getattr(out, name)), np.asarray(getattr(ray, name)),
                           rtol=0, atol=1e-18)


def test_the_beam_keeps_a_positive_decay():
    beam = gaussian_waist(z=0.05, voltage=200e3, rms_size=1e-6)
    out = LENS(beam)
    assert np.all(np.linalg.eigvalsh(np.asarray(jnp.imag(out.Q_inv))) > 0)


def test_a_decomposed_beam_through_the_lens_gives_the_whole_beam():
    """The beamlets of a decomposition, each through the lens, must add up to the beam through it:
    this checks every beamlet's curvature, amplitude and path together."""
    beam = gaussian_waist(z=0.0, voltage=200e3, rms_size=0.2e-6)
    bundle = decompose_gaussian(beam, num_across=31, extent_rms=6.0)
    detector = Detector(z=0.06, pixel_size=(40e-9, 40e-9), shape=(32, 32))
    whole = evaluate_gaussians(run_to_end(beam, [LENS, detector]), detector, method="scan")
    parts = evaluate_gaussians(run_to_end(bundle, [LENS, detector]), detector, method="scan")
    difference = float(jnp.max(jnp.abs(whole - parts)) / jnp.max(jnp.abs(whole)))
    assert difference < 1e-3
