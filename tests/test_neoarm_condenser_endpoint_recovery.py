import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"examples/lens_inversion"))
from neoarm_condenser_endpoint_recovery import (
    calibration_training, ORDER, INVERSE, JointEndpointModel, joint_training_residual,
)
from neoarm_condenser_endpoint_calibration import EndpointRows, GROUPED
from neoarm_condenser_image_recovery import covariance_observable
from temgym_core.microscope_inversion import drift_matrix


class ToyCondenser:
    def condenser_matrix(self, optical, control):
        return drift_matrix(.1+.01*optical[0]+.001*control[0])


class ToyQuotient:
    model = ToyCondenser()

    def decode(self, p):
        return p[:12], jnp.diag(jnp.array([1e-14, 1e-6, 2e-14, 2e-6])), p[21], p[22]

    def observables(self, p, beam, aperture):
        ap = jnp.tile(jnp.array([-p[22], -p[22], 0.]), (len(aperture), 1))
        return jnp.zeros((len(beam), 2, 3)), ap

    def admissible(self, p):
        return bool(np.isfinite(p).all())


def test_training_selection_does_not_access_checks_or_truth():
    class OnlyTraining(dict):
        def __getitem__(self, key):
            assert key in ("response", "response_covariance")
            return super().__getitem__(key)
    data = OnlyTraining(response=np.zeros((2, 2, 4)),
                        response_covariance=np.tile(np.eye(8), (2, 1, 1)))
    selected, covariance = calibration_training(data)
    assert selected.shape == (2, 2, 4)
    assert covariance.shape == (2, 8, 8)


def test_joint_geometry_order_releases_every_parameter():
    p = np.arange(29)
    np.testing.assert_array_equal(p[ORDER][INVERSE], p)
    np.testing.assert_array_equal(ORDER[:6], [0, 1, 4, 5, 8, 9])
    assert len(ORDER[6:]) == 23


def test_canonical_beam_composition_and_derivatives():
    endpoints = EndpointRows(-873.)
    quotient = ToyQuotient()
    prefield = drift_matrix(.003)
    joint = JointEndpointModel(quotient, endpoints, prefield)
    p = jnp.r_[jnp.zeros(23), jnp.array([3., .12, .2, 3.1, -.14, -.4])]
    controls = jnp.zeros((2, 7)).at[1, 0].set(.3)
    actual = jax.jit(joint.beam_observables)(p, controls)
    rows = np.asarray(endpoints.canonical_rows(p[23:]))[:, :, GROUPED]
    cov = np.asarray(quotient.decode(p[:23])[1])
    expected = []
    for u in controls:
        matrix = np.asarray(prefield)@np.asarray(quotient.model.condenser_matrix(p[:12], u))
        propagated = rows@matrix@cov@matrix.T@rows.swapaxes(-1, -2)
        expected.append(np.asarray(jax.vmap(covariance_observable)(propagated/endpoints.pixel_m**2)))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    derivative = np.asarray(jax.jacfwd(joint.beam_observables)(p, controls))
    delta = jnp.zeros(29).at[23].set(1e-5)
    numerical = (joint.beam_observables(p+delta, controls)-joint.beam_observables(p-delta, controls))/2e-5
    np.testing.assert_allclose(derivative[..., 23], numerical, rtol=1e-6, atol=1e-8)


def test_likelihood_has_118_data_coordinates_and_one_radius_prior():
    endpoints = EndpointRows(-873.)
    joint = JointEndpointModel(ToyQuotient(), endpoints, drift_matrix(.003))
    p = jnp.r_[jnp.zeros(23), jnp.array([3., .12, .2, 3.1, -.14, -.4])]
    controls = np.zeros((8, 7))
    beam = np.asarray(joint.beam_observables(p, controls))
    covariance = np.tile(.01*np.eye(3), (8, 2, 1, 1))
    old = dict(controls=controls, beam_observables=beam, beam_observable_covariances=covariance,
        aperture_observables=np.zeros((2, 3)), aperture_observable_covariances=np.tile(.01*np.eye(3), (2, 1, 1)))
    aperture = dict(records=[dict(controls=np.zeros(7)) for _ in range(2)])
    calibration = (np.asarray(endpoints.response(p[23:])), np.tile(.01*np.eye(8), (2, 1, 1)))
    residual = jax.jit(joint_training_residual(joint, old, aperture,
        (controls, beam, covariance), calibration))
    np.testing.assert_allclose(residual(p), np.zeros(119), atol=1e-12)
    shifted = np.asarray(residual(p.at[22].set(.005)))
    assert shifted[-1] == 1.
    np.testing.assert_allclose(shifted[:96], 0., atol=1e-12)
    np.testing.assert_allclose(shifted[102:118], 0., atol=1e-12)


@pytest.mark.parametrize("bad", ["shape", "nan", "asymmetric", "indefinite"])
def test_calibration_covariance_validation(bad):
    data = dict(response=np.zeros((2, 2, 4)), response_covariance=np.tile(np.eye(8), (2, 1, 1)))
    if bad == "shape": data["response"] = np.zeros((2, 4))
    if bad == "nan": data["response"][0, 0, 0] = np.nan
    if bad == "asymmetric": data["response_covariance"][0, 0, 1] = .1
    if bad == "indefinite": data["response_covariance"][0, 0, 0] = -1.
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        calibration_training(data)
