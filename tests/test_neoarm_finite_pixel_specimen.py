from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_finite_pixel_specimen import (
    RegularGaussianLattice, evaluate_regular_lattice, make_finite_pixel_conjugate_predictor,
)
from neoarm_mixed_gaussian_validation import imaging_map
from temgym_core.evaluate import _beam_field
from temgym_core.gaussian import make_gaussian
from temgym_core.gaussian_image_likelihood import make_gaussian_detector_operator
from temgym_core.multiscale_sample import SampleLevel2D, SampleView
from temgym_core.neoarm_calibration import DetectorGeometry, apply_transfer_matrix_to_gaussians

jax.config.update('jax_enable_x64', True)


def lattice_fixture():
    x, y = np.linspace(-18e-9, 18e-9, 19), np.linspace(-22e-9, 22e-9, 23)
    xx, yy = np.meshgrid(x, y, indexing='xy')
    beam = make_gaussian(x=xx.ravel(), y=yy.ravel(), dx=xx.ravel()*120,
        dy=yy.ravel()*-200, amp=np.exp(-(xx.ravel()**2+yy.ravel()**2)/(20e-9)**2),
        phase=.12*np.sin(xx.ravel()/3e-9), waist_x=2.5e-9, waist_y=2.5e-9, voltage=200e3)
    matrix = imaging_map(1200., .7, 0.)
    return apply_transfer_matrix_to_gaussians(beam, matrix, z_out_m=.8, axial_path_m=0.), xx.shape


def exhaustive(parameters, coords):
    values = jax.vmap(_beam_field, in_axes=(0, 0, 0, 0, 0, 0, None))(*parameters, coords)
    return jnp.sum(values, axis=0)


def test_local_coherent_fields_match_exhaustive_with_phase_and_edge_points():
    beam, shape = lattice_fixture()
    lattice = RegularGaussianLattice.from_beam(beam, shape, require_gpu=False)
    coords = jnp.asarray(np.random.default_rng(91).uniform(-30e-6, 30e-6, (49, 2)))
    expected = exhaustive(lattice.parameters, coords)
    actual = lattice(coords)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-17)
    assert lattice.validation['maximum_q_difference'] == 0


def test_spatial_gradients_and_hessians_retain_complex_beam_phases():
    beam, shape = lattice_fixture()
    lattice = RegularGaussianLattice.from_beam(beam, shape, require_gpu=False)
    point = jnp.array([1.17e-6, -2.46e-6])
    fast = lambda p: jnp.abs(lattice(p[None, :])[0])**2
    exact = lambda p: jnp.abs(exhaustive(lattice.parameters, p))**2
    np.testing.assert_allclose(jax.grad(fast)(point), jax.grad(exact)(point), rtol=1e-9, atol=1e-15)
    np.testing.assert_allclose(jax.hessian(fast)(point), jax.hessian(exact)(point), rtol=1e-8, atol=1e-9)


def test_amplitude_parameter_jvp_matches_exhaustive_coherent_sum():
    beam, shape = lattice_fixture()
    lattice = RegularGaussianLattice.from_beam(beam, shape, require_gpu=False)
    coords = jnp.array([[1.17e-6, -2.46e-6], [-3.71e-6, 4.12e-6]])
    parameter_direction = jnp.sin(jnp.arange(lattice.parameters[2].size))
    def parameters(t):
        values = list(lattice.parameters)
        values[2] = values[2]*(1+t*parameter_direction)
        return tuple(values)
    fast = lambda t: evaluate_regular_lattice(parameters(t), coords, shape_yx=shape, radius=lattice.radius)
    exact = lambda t: exhaustive(parameters(t), coords)
    np.testing.assert_allclose(jax.jvp(fast, (0.,), (1.,))[1], jax.jvp(exact, (0.,), (1.,))[1],
                               rtol=1e-11, atol=1e-17)


def test_irregular_or_incompatible_gaussians_fail_the_fast_adapter():
    beam, shape = lattice_fixture()
    with pytest.raises(ValueError, match='Regular positions'):
        RegularGaussianLattice.from_beam(beam.derive(x=beam.x.at[17].add(1e-6)), shape, require_gpu=False)
    with pytest.raises(ValueError, match='identical Q'):
        RegularGaussianLattice.from_beam(beam.derive(Q_inv=beam.Q_inv.at[17, 0, 0].add(1.)), shape, require_gpu=False)
    with pytest.raises(ValueError, match='too broad'):
        RegularGaussianLattice.from_beam(beam, shape, maximum_radius=2, require_gpu=False)


def test_finite_pixel_cubic_reference_reduces_to_physical_vacuum_gaussian_integral():
    x, y = np.linspace(-.1e-6, .1e-6, 17), np.linspace(-.08e-6, .08e-6, 15)
    view = SampleView.from_level(SampleLevel2D(np.zeros((15, 17)), np.zeros((15, 17)), x, y))
    matrix = imaging_map(1200., .7, 0.)
    detector = DetectorGeometry(shape_yx=(19, 27), pixel_size_yx_m=(9e-6, 7e-6),
        affine_xy=np.array([[1.02, .1], [-.06, .98]]), centre_xy_m=(3e-6, -2e-6), mtf_sigma_px=.65)
    centre, waist, dose = np.array([2e-9, -1e-9]), 18e-9, 2e7
    predictor, _ = make_finite_pixel_conjugate_predictor(view, detector, centre,
        reference_matrix=matrix, wavelength_m=2.5e-12, shortest_scale_m=1e-9, quadrature_order=16)
    truth = jnp.array([np.log(1200.), .7, 0., 0., np.log(dose), np.log(waist), 0.])
    a = matrix[np.ix_([0, 2], [0, 2])]
    expected = make_gaussian_detector_operator(detector, 16)(a@centre, a@(np.eye(2)*waist**2/4)@a.T, dose)
    np.testing.assert_allclose(predictor(truth), expected, rtol=1e-12, atol=1e-7)
    assert np.all(np.isfinite(jax.jacfwd(predictor)(truth)))


def test_finite_pixel_intensity_template_rejects_nonconjugate_map():
    x = np.linspace(-10e-9, 10e-9, 9)
    view = SampleView.from_level(SampleLevel2D(np.zeros((9, 9)), np.zeros((9, 9)), x, x))
    with pytest.raises(ValueError, match='requires conjugacy'):
        make_finite_pixel_conjugate_predictor(view, DetectorGeometry(), [0., 0.],
            reference_matrix=imaging_map(4e4, .2, 1e-4), wavelength_m=2.5e-12, shortest_scale_m=1e-9)
