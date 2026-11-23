"""converged_gaussian_image: refine a beam's decomposition until its image stops changing."""

import jax.numpy as jnp
import numpy as np

from temgym_core.components import CrossGrating, Detector
from temgym_core.evaluate import converged_gaussian_image
from temgym_core.source import gaussian_waist

PX = 9.2e-9
N = 32
RMS = N * PX / 5


def test_a_plain_beam_converges_at_the_first_refinement():
    beam = gaussian_waist(z=0.0, voltage=200e3, rms_size=RMS)
    detector = Detector(z=1e-6, pixel_size=(PX, PX), shape=(N, N))
    result = converged_gaussian_image(beam, [], detector, num_across=15, tolerance=1e-3,
                                      extent_rms=3.0, method="scan")
    assert result.converged
    assert [level for level, _ in result.history] == [15, 31]
    assert result.change < 1e-3


def test_a_grating_converges_to_its_exact_image():
    grating = CrossGrating(z=0.0, period=230e-9, edge_width=1.5 * PX, rotation=0.3)
    beam = gaussian_waist(z=-1e-12, voltage=200e3, rms_size=RMS)
    detector = Detector(z=1e-12, pixel_size=(PX, PX), shape=(N, N))
    result = converged_gaussian_image(beam, [grating], detector, num_across=15, tolerance=2e-2,
                                      extent_rms=3.0, method="scan")
    assert result.converged
    xy = jnp.asarray(detector.coords).reshape(N, N, 2)
    exact = np.asarray(jnp.exp(-(xy[..., 0] ** 2 + xy[..., 1] ** 2) / (2 * RMS**2))
                       * jnp.exp(2 * grating.log_transmission(xy)))
    image = np.abs(np.asarray(result.field)) ** 2
    image *= (image * exact).sum() / (image * image).sum()
    # The image is within a few times the last change of the exact one.
    assert np.abs(image - exact).max() / exact.max() < 3 * result.change + 1e-3


def test_it_says_so_when_it_stops_at_the_largest_decomposition():
    grating = CrossGrating(z=0.0, period=230e-9, edge_width=1.5 * PX, rotation=0.3)
    beam = gaussian_waist(z=-1e-12, voltage=200e3, rms_size=RMS)
    detector = Detector(z=1e-12, pixel_size=(PX, PX), shape=(N, N))
    result = converged_gaussian_image(beam, [grating], detector, num_across=8, tolerance=1e-9,
                                      max_across=40, extent_rms=3.0, method="scan")
    assert not result.converged
    assert result.num_across == 35
    assert [level for level, _ in result.history] == [8, 17, 35]



def test_it_refuses_a_start_above_its_limit_and_a_dark_detector():
    import pytest

    beam = gaussian_waist(z=0.0, voltage=200e3, rms_size=RMS)
    detector = Detector(z=1e-6, pixel_size=(PX, PX), shape=(N, N))
    with pytest.raises(ValueError):
        converged_gaussian_image(beam, [], detector, num_across=31, max_across=20, method="scan")
    far = Detector(z=1e-6, pixel_size=(PX, PX), shape=(N, N), centre=(1e-3, 0.0))
    with pytest.raises(ValueError):
        converged_gaussian_image(beam, [], far, num_across=15, extent_rms=3.0, method="scan")
