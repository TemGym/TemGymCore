import numpy as np
import jax
import jax.numpy as jnp
import pytest

from temgym_core.gaussian_action2D import (
    ThinLens2D,
    AberratedLens2D,
    GaussianBeamFactory,
    GaussianBeamBundle,
    GaussianBeam,
)

jax.config.update("jax_enable_x64", True)


def _expected_phase(focal_length, cubic_coeff, quartic_coeff, xy, center, eps=0.0):
    x = xy[0] - center[0]
    y = xy[1] - center[1]
    rho2 = x * x + y * y
    rho3 = rho2 * jnp.sqrt(rho2 + eps)
    rho4 = rho2 * rho2
    return (
        -0.5 * rho2 / focal_length
        + cubic_coeff * rho3
        + quartic_coeff * rho4
    )


def test_aberrated_lens_phase_matches_expected_profile():
    focal_length = 3.0
    cubic = 1e-2
    quartic = -4e-3
    center = (0.1, -0.15)
    xy = jnp.array([0.35, -0.05], dtype=jnp.float64)

    lens = AberratedLens2D(
        focal_length=focal_length,
        cubic_coeff=cubic,
        quartic_coeff=quartic,
        center=center,
    )
    phase = lens.phase_shift(xy)
    expected = _expected_phase(focal_length, cubic, quartic, xy, center, lens.eps)

    np.testing.assert_allclose(
        float(phase), float(expected), rtol=1e-12, atol=1e-12
    )


def test_aberrated_lens_differs_from_thin_lens_when_aberration_nonzero():
    xy = jnp.array([0.25, -0.2], dtype=jnp.float64)
    center = (0.0, 0.0)
    focal_length = 4.0

    thin = ThinLens2D(focal_length=focal_length, center=center)
    aberrated = AberratedLens2D(
        focal_length=focal_length,
        cubic_coeff=5e-3,
        quartic_coeff=1e-3,
        center=center,
    )

    phase_thin = thin.phase_shift(xy)
    phase_aberrated = aberrated.phase_shift(xy)

    assert not np.isclose(float(phase_thin), float(phase_aberrated))


def test_gaussian_beam_factory_round_aperture_outputs_gaussian_rays():
    factory = GaussianBeamFactory(
        voltage=200e3,
        waist_radius=5e-9,
        normalization="unit",
    )
    beam = factory.round_aperture(aperture_radius=1e-8, num_rays=4)

    assert isinstance(beam, GaussianBeamBundle)
    assert len(beam) == 4
    assert beam.rays and all(isinstance(ray, GaussianBeam) for ray in beam.rays)

    stacked = beam.stack_parameters()
    assert stacked["C"].shape == (4,)
    assert stacked["S1"].shape == (4, 2)
    assert stacked["S2"].shape == (4, 2, 2)
    assert stacked["voltage"].shape == (4,)

    first_ray = beam.rays[0]
    assert first_ray.voltage == pytest.approx(factory.voltage)
    assert first_ray.wavelength > 0.0
    assert first_ray.mass > 0.0
    assert first_ray.sigma > 0.0
