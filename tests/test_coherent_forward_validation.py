"""Independent reference checks, including TemGym's optical-path phase units."""
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'examples/microscope_models'))
import coherent_forward_validation as validation

from temgym_core.coherent_image import CoherentDetector, microscope_image
from temgym_core.components import Lens
from temgym_core.gaussian import make_gaussian


@pytest.mark.parametrize('distance', [-12e-6, 0., 12e-6])
def test_fft_reference_agrees_with_analytic_quadratic_phase_gaussian(monkeypatch, distance):
    waist, focal = 8e-9, 50e-6
    monkeypatch.setattr(validation, 'illumination_amplitude',
                        lambda xy: np.exp(-np.sum(np.asarray(xy)**2, axis=-1) / waist**2))

    class QuadraticSample:
        def evaluate_actions(self, xy):
            path_m = -jnp.sum(xy**2, axis=-1) / (2 * focal)
            return jnp.stack((path_m, jnp.zeros_like(path_m)), axis=-1)

    matrix = np.eye(5)
    matrix[:2, 2:4] = np.eye(2) * distance
    detector = CoherentDetector((16, 16), 2e-9, distance, 2)
    reference = validation.fft_reference(QuadraticSample(), [matrix], [detector],
                                          step_nm=.125, width_nm=128.)[0]
    source = make_gaussian(waist_x=waist, waist_y=waist, voltage=validation.VOLTAGE)
    actual = microscope_image(source, (Lens(z=0., focal_length=focal),), detector)
    actual = np.asarray(actual) / validation.INCIDENT_FLUX
    assert validation.relative_error(actual, reference) < 1e-9


def test_refinement_does_not_change_the_prescribed_illumination():
    from types import SimpleNamespace

    from temgym_core.evaluate import evaluate_gaussians
    axis = np.linspace(-36e-9, 36e-9, 29)
    xx, yy = np.meshgrid(axis, axis)
    xy = np.stack((xx.ravel(), yy.ravel()), axis=-1)
    grid = SimpleNamespace(shape=xx.shape, coords=jnp.asarray(xy))
    reference = validation.illumination_amplitude(xy).reshape(xx.shape)
    counts = []
    for width in (2., 1.):
        beam, omitted = validation.source_wave(width)
        counts.append(beam.x.size)
        field = np.asarray(evaluate_gaussians(beam, grid, method='scan'))
        assert validation.relative_error(field, reference) < 5e-4
        assert omitted < 1e-9
    assert counts[1] > counts[0]


def test_magical_image_derivative_with_round_beam_and_moving_gaussian_grid():
    import jax

    from temgym_core.coherent_microscope import GaussianSourceState, WaveNumerics
    sample,column,detector,_,_=validation.make_scene((6,6),(-4.,),2)
    numerics=WaveNumerics(specimen_half_width=80e-9,beamlet_width=8e-9,follow_source=True)
    def prediction(size):
        beam=GaussianSourceState(.2,rms_size=size*1e-9).beam(0.,200000.,jnp.zeros(2))
        return microscope_image(numerics.resolve(beam),(sample,*column(-4.)),detector(-4.))*1e5
    prediction=jax.jit(prediction)
    exact=jax.jacfwd(prediction)(10.)
    assert np.isfinite(exact).all()
    finite=(prediction(10.00001)-prediction(9.99999))/.00002
    np.testing.assert_allclose(exact,finite,rtol=3e-4,atol=1e-4)
