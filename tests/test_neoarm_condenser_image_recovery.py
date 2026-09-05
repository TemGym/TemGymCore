"""Small matrix, gauge and initialization checks; no CPU detector rendering."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

EXAMPLES = Path(__file__).resolve().parents[1]/"examples/lens_inversion"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from neoarm_condenser_image_recovery import (
    CondenserModel, SOURCE_REFERENCE_SCALES, covariance_observable,
    covariance_from_observable, covariance_chart_coordinates,
    linear_source_initializer, profile_optical_information, finite_json,
)
from temgym_core.critical_line_calibration import EntranceCovarianceChart
from temgym_core.microscope_inversion import drift_matrix

jax.config.update("jax_enable_x64", True)


def matrix_fixture():
    model = CondenserModel.__new__(CondenserModel)
    model.source_z_m = -.7; model.aperture_z_m = -.35; model.objective_entrance_m = -.05
    model.channels = (0, 1, 2); model.reference_controls = jnp.zeros(3)
    model.prefield = drift_matrix(.05)
    model.downstream = jnp.stack([drift_matrix(.1), drift_matrix(.4)])
    model.base = np.array([-.6, np.log(.015), .2, .55,
                           -.45, np.log(.02), .15, .65,
                           -.2, np.log(.012), .1, .45])
    model.scales = np.tile([.01, .1, .1, .1], 3)
    model.chart = EntranceCovarianceChart(np.diag(SOURCE_REFERENCE_SCALES**2))
    return model


def test_covariance_observables_roundtrip():
    covariance = np.array([[9., -1.2], [-1.2, 4.]])
    np.testing.assert_allclose(covariance_from_observable(covariance_observable(covariance)), covariance, rtol=1e-14)


def test_source_chart_roundtrip():
    model = matrix_fixture()
    covariance = np.diag(np.array([8e-9, 40e-6, 10e-9, 35e-6])**2)
    coordinates = covariance_chart_coordinates(model, covariance)
    np.testing.assert_allclose(model.chart.covariance(coordinates), covariance, rtol=1e-13, atol=1e-30)


def test_source_reference_plane_covariance_gauge():
    model = matrix_fixture(); control = jnp.array([.2, -.1, .4, 0, 0, 0, 0])
    covariance = np.diag(SOURCE_REFERENCE_SCALES**2)
    first = model.maps(model.base, control)
    shift = .013
    second = model.maps(model.base, control, source_z_m=model.source_z_m+shift)
    moved_covariance = drift_matrix(shift)@covariance@drift_matrix(shift).T
    np.testing.assert_allclose(first@covariance@first.transpose(0, 2, 1),
        second@moved_covariance@second.transpose(0, 2, 1), rtol=1e-12, atol=1e-23)


def test_round_spatial_images_cannot_measure_antisymmetric_position_angle_covariance():
    model = matrix_fixture()
    covariance = np.diag(SOURCE_REFERENCE_SCALES**2)
    null = np.zeros((4, 4)); null[0, 3]=null[3, 0]=1e-14; null[1, 2]=null[2, 1]=-1e-14
    for control in (jnp.zeros(7), jnp.array([.2, -.1, .4, 0, 0, 0, 0])):
        maps = np.asarray(model.maps(model.base, control))[:, [0, 2], :]
        np.testing.assert_allclose(maps@null@maps.transpose(0, 2, 1), 0., atol=1e-25)
        np.testing.assert_allclose(maps@(covariance+null)@maps.transpose(0, 2, 1),
            maps@covariance@maps.transpose(0, 2, 1), rtol=1e-12, atol=1e-25)


def test_covariance_initializer_uses_observations_and_diagram():
    model = matrix_fixture()
    rng = np.random.default_rng(82)
    controls = np.column_stack([rng.uniform(-.2, .3, (10, 3)), np.zeros((10, 4))])
    aperture_controls = controls[:4]
    physical_covariance = np.diag(np.array([8e-9, 40e-6, 10e-9, 35e-6])**2)
    angle = np.linspace(-np.pi, np.pi, 49, endpoint=False)[31]
    p = np.r_[np.zeros(12), covariance_chart_coordinates(model, physical_covariance), angle]
    beam, aperture = model.observables(p, controls, aperture_controls)
    noise = np.broadcast_to(np.eye(3)*1e-6, beam.shape[:-1]+(3, 3))
    anoise = np.broadcast_to(np.eye(3)*1e-6, aperture.shape[:-1]+(3, 3))
    initial, evidence = linear_source_initializer(model, controls, beam, noise, aperture_controls, aperture, anoise)
    assert evidence["initial_quantum_cone_diagonal_shift"] == 0
    predicted_beam, predicted_aperture = model.observables(initial, controls, aperture_controls)
    np.testing.assert_allclose(predicted_beam, beam, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(predicted_aperture, aperture, rtol=1e-10, atol=1e-11)


def test_optical_information_profiles_nuisance_column():
    matrix = np.column_stack([np.eye(5)[:, 0], np.eye(5)[:, 1], np.eye(5)[:, 1]])
    result, covariance = profile_optical_information(matrix, count=2)
    assert result["optical_rank"] == 1
    assert np.isnan(covariance).all()
    assert finite_json(result)["normalized_parameter_standard_errors"] == [None, None]


def test_observable_jacobian_matches_central_difference():
    model = matrix_fixture()
    controls = jnp.array([[.2, -.1, .4, 0., 0., 0., 0.], [-.1, .3, .1, 0., 0., 0., 0.]])
    p = np.r_[np.zeros(12), np.linspace(-.2, .15, 10), .31]
    def observable(value):
        beam, aperture = model.observables(value, controls, controls)
        return jnp.r_[beam.reshape(-1), aperture.reshape(-1)]
    actual = np.asarray(jax.jit(jax.jacfwd(observable))(p))
    step = 1e-5
    expected = np.column_stack([(np.asarray(observable(p+step*unit))-
        np.asarray(observable(p-step*unit)))/(2*step) for unit in np.eye(len(p))])
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-7)
