import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_condenser_cold_start import displaced_diagram_optics, cold_initializers, ORDER, INVERSE


def base():
    return np.array([[-.5, np.log(.02), .2, .55], [-.4, np.log(.015), .15, .65],
                     [-.1, np.log(.008), .1, .45]]).ravel()


def test_prescribed_displacements_hold_q_over_sqrt_length_without_fitted_inputs():
    original = base().reshape(3, 4)
    for sign in (-1, 0, 1):
        value = displaced_diagram_optics(original.ravel(), sign).reshape(3, 4)
        np.testing.assert_allclose(value[:, 0]-original[:, 0], sign*np.array([1e-3, -.3e-3, .15e-3]), atol=1e-16)
        np.testing.assert_allclose(value[:, 1]-original[:, 1], sign*np.array([.08, -.06, .10]), atol=1e-15)
        np.testing.assert_allclose(value[:, 2:4]/np.exp(value[:, 1, None]/2),
                                   original[:, 2:4]/np.exp(original[:, 1, None]/2), rtol=1e-14)


def test_cold_initializers_use_training_inputs_and_restore_provisional_model():
    original = base()
    model = SimpleNamespace(base=original.copy())
    model.decode = lambda p: (model.base.copy(), np.eye(4)*1e-12, .3)
    quotient = SimpleNamespace(model=model, optical_coordinates=lambda p: np.asarray(p))
    calls = []
    controls, data, cov = np.zeros((3, 7)), np.ones((3, 2, 3)), np.tile(np.eye(3), (3, 2, 1, 1))
    ac, ad, av = np.zeros((2, 7)), np.zeros((2, 3)), np.tile(np.eye(3), (2, 1, 1))
    def initializer(m, bc, bd, bv, apc, apd, apv):
        assert bc is controls and bd is data and bv is cov
        assert apc is ac and apd is ad and apv is av
        calls.append(m.base.copy())
        return np.zeros(1), dict(data_only=True)
    seeds = cold_initializers(quotient, controls, data, cov, ac, ad, av, initializer=initializer)
    assert [s['sign'] for s in seeds] == [-1, 1, 0]
    assert [s['geometry_genuinely_displaced_from_diagram'] for s in seeds] == [True, True, False]
    for value, sign in zip(calls, (-1, 1, 0)):
        np.testing.assert_array_equal(value, displaced_diagram_optics(original, sign))
    np.testing.assert_array_equal(model.base, original)


def test_failed_tomography_cannot_leave_model_at_a_temporary_seed():
    original = base(); model = SimpleNamespace(base=original.copy())
    quotient = SimpleNamespace(model=model)
    def fail(*args):
        raise RuntimeError('tomography failure')
    with pytest.raises(RuntimeError):
        cold_initializers(quotient, None, None, None, None, None, None, initializer=fail)
    np.testing.assert_array_equal(model.base, original)


def test_all23_coordinates_are_retained_by_geometry_first_order():
    np.testing.assert_array_equal(ORDER[INVERSE], np.arange(23))
    assert list(ORDER[:6]) == [0, 1, 4, 5, 8, 9]


def test_nonprescribed_geometry_displacement_rejected():
    with pytest.raises(ValueError):
        displaced_diagram_optics(base(), 2)
