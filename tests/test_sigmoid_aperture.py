"""SigmoidAperture is smooth at its exact centre and unchanged at its edge."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.components import SigmoidAperture
from temgym_core.gaussian import make_gaussian

jax.config.update("jax_enable_x64", True)

RADIUS, EDGE = 1.6e-6, 0.1e-6


def aperture(**changes):
    return SigmoidAperture(z=0.0, radius=RADIUS, edge_width=EDGE, **changes)


def hessian(x, y=0.0, **changes):
    return np.asarray(jax.hessian(aperture(**changes).log_transmission)(jnp.array([x, y])))


def test_hessian_at_the_exact_centre_is_finite_and_small():
    # A plain radius has a conical tip: about -1e15 1/m^2 here, against the 6e12 1/m^2 curvature
    # of a 0.6 um beamlet's own field. The softened radius stays six orders below that.
    h = hessian(0.0)
    assert np.all(np.isfinite(h))
    assert np.max(np.abs(h)) < 1e10


def test_hessian_is_continuous_through_the_centre():
    at = [np.max(np.abs(hessian(x))) for x in (0.0, 1e-12, 1e-10, 1e-9, 1e-8)]
    assert max(at) / min(at) < 10


@pytest.mark.parametrize("waist", [0.3e-6, 0.6e-6, 1.0e-6])
def test_beamlet_centred_on_the_aperture_keeps_its_width(waist):
    beam = make_gaussian(x=0.0, y=0.0, voltage=200e3, waist_x=waist, waist_y=waist)
    out = aperture()(beam)
    np.testing.assert_allclose(np.imag(out.Q_inv), np.imag(beam.Q_inv), rtol=2e-3)
    np.testing.assert_allclose([out.x, out.y], [0.0, 0.0], atol=1e-15)


def test_off_centre_beamlets_are_unchanged_by_the_softening():
    for offset in (1e-8, 1e-7, 4e-7):
        beam = make_gaussian(x=offset, y=-offset / 2, voltage=200e3, waist_x=0.6e-6,
                             waist_y=0.6e-6)
        out = aperture()(beam)
        h = hessian(offset, -offset / 2)
        # Gaussian action: Im Q gains -H / k.
        expected = np.imag(beam.Q_inv) - h / float(beam.k)
        np.testing.assert_allclose(np.imag(out.Q_inv), expected, rtol=0.05, atol=1e-6)


def test_edge_profile_is_the_plain_sigmoid():
    """Transmission across the edge equals 1 - sigmoid((rho - R) / w) to well under a percent."""
    rho = np.array([0.9, 0.98, 1.0, 1.02, 1.1]) * RADIUS
    xy = jnp.stack([jnp.asarray(rho), jnp.zeros_like(rho)], axis=-1)
    actual = np.exp(np.asarray(aperture().log_transmission(xy)))
    expected = 1.0 - jax.nn.sigmoid((rho - RADIUS) / EDGE)
    np.testing.assert_allclose(actual, expected, rtol=1e-3)
    assert abs(actual[2] - 0.5) < 1e-3


def test_sharp_apertures_and_axis_offsets_still_work():
    sharp = SigmoidAperture(z=0.0, radius=1e-6, edge_width=1e-9, x0=2e-7, y0=-1e-7)
    xy = jnp.array([[2e-7, -1e-7], [1.5e-6, -1e-7], [-8e-7, -1e-7]])
    t = np.exp(np.asarray(sharp.log_transmission(xy)))
    np.testing.assert_allclose(t, [1.0, 0.0, 0.5], atol=1e-6)
