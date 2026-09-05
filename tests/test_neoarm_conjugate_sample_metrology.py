"""Unit checks; actual archive-vs-template closure is a separate saved audit."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/lens_inversion"))
from neoarm_conjugate_sample_metrology import make_conjugate_count_predictor
from temgym_core.multiscale_sample import SampleLevel2D, SampleView
from temgym_core.neoarm_calibration import DetectorGeometry

jax.config.update("jax_enable_x64", True)


def fixture(mtf=.65):
    coordinates = np.linspace(-100e-9, 100e-9, 24)
    x, y = np.meshgrid(coordinates, coordinates)
    log_amplitude = -.1 + 2e5*x - 3e5*y + 1e12*x*y
    level = SampleLevel2D(np.zeros_like(x), log_amplitude, coordinates, coordinates)
    view = SampleView.from_level(level, centre_xy_m=(1e-9, -2e-9), rotation_rad=.1)
    detector = DetectorGeometry(shape_yx=(25, 31), pixel_size_yx_m=(20e-6, 15e-6),
                                affine_xy=np.array([[1.03, .02], [-.01, .98]]),
                                z_m=1., mtf_sigma_px=mtf)
    angle, mag = .4, 2e4
    a = mag * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    matrix = np.zeros((4, 4))
    matrix[np.ix_([0, 2], [0, 2])] = a
    matrix[np.ix_([1, 3], [1, 3])] = np.linalg.inv(a.T)
    parameters = np.array([np.log(mag), angle, 6., -3., np.log(1e6), np.log(15e-9), np.log(1.1)])
    return view, detector, matrix, parameters


@pytest.mark.parametrize("mtf", [0., .65, 1.2])
def test_template_matches_independent_polynomial_cubic_object_and_finite_detector(mtf):
    view, detector, matrix, p = fixture(mtf)
    source = np.array([2e-9, -1e-9])
    predict, phase = make_conjugate_count_predictor(view, detector, source,
        reference_matrix=matrix, wavelength_m=2.5e-12, shortest_scale_m=1e-9)
    assert phase == 0
    a = matrix[np.ix_([0, 2], [0, 2])]
    specimen = np.einsum("ij,...j->...i", np.linalg.inv(a), detector.optical_coords_xy_m)
    local = np.einsum("ij,...j->...i", np.linalg.inv(np.asarray(view.projection_world_from_local)),
                      specimen - p[2:4]*1e-9)
    x, y = local[..., 0], local[..., 1]
    attenuation = -.1 + 2e5*x - 3e5*y + 1e12*x*y
    waist = np.exp(p[5])
    expected = 2/(np.pi*waist**2) * np.exp(-2*np.sum((specimen-source)**2, axis=-1)/waist**2)
    expected *= np.exp(2*attenuation*np.exp(p[6])) * np.exp(p[4]-2*p[0])
    expected *= np.prod(detector.pixel_size_yx_m)/abs(np.linalg.det(detector.affine_xy))
    expected = gaussian_filter(expected, mtf, mode="nearest") if mtf else expected
    np.testing.assert_allclose(predict(p), expected, rtol=2e-12, atol=1e-9)


def test_template_jacobian_vmap_and_translation_units():
    view, detector, matrix, p = fixture()
    predict, _ = make_conjugate_count_predictor(view, detector, (2e-9, -1e-9),
        reference_matrix=matrix, wavelength_m=2.5e-12, shortest_scale_m=1e-9)
    derivative = np.asarray(jax.jacfwd(predict)(p))
    step = 1e-5
    numerical = np.stack([(np.asarray(predict(p+step*d))-np.asarray(predict(p-step*d)))/(2*step)
                          for d in np.eye(7)], axis=-1)
    np.testing.assert_allclose(derivative, numerical, atol=1e-5, rtol=2e-6)
    assert jax.jit(jax.vmap(predict))(jnp.stack((p, p+.001))).shape == (2, 25, 31)


@pytest.mark.parametrize("invalid", ["defocus", "reflection", "anisotropic", "nan_wavelength"])
def test_nonconjugate_or_non_similarity_states_are_not_silently_used(invalid):
    view, detector, matrix, _ = fixture()
    wavelength = 2.5e-12
    if invalid == "defocus":
        matrix[0, 1] = 1.
    elif invalid == "reflection":
        matrix[:, 0] *= -1
    elif invalid == "anisotropic":
        matrix[0] *= 1.01
    else:
        wavelength = np.nan
    with pytest.raises(ValueError):
        make_conjugate_count_predictor(view, detector, (0., 0.), reference_matrix=matrix,
                                       wavelength_m=wavelength, shortest_scale_m=1e-9)
