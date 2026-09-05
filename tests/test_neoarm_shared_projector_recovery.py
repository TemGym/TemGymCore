"""Training-role and non-generating seed contracts for shared projectors."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1]/"examples/lens_inversion"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
from neoarm_shared_projector_recovery import selected_data, diagram_geometry_seed, GEOMETRY, EXCITATION, ORDER, INVERSE


def archive():
    return dict(training_mask=np.array([True, False]), heldout_mask=np.array([False, True]),
        native_controls=np.zeros((2, 7)), response=np.arange(16.).reshape(2, 2, 4),
        response_covariance=np.tile(np.eye(8), (2, 1, 1)))


def test_selected_data_does_not_mix_heldout_response():
    data = archive(); data["response"][1] = np.nan
    assert np.isfinite(selected_data(data, True)[1]).all()
    with pytest.raises(ValueError):
        selected_data(data, False)


def test_roles_and_complete_covariance_are_required():
    data = archive(); data["training_mask"][1] = True
    with pytest.raises(ValueError): selected_data(data)
    data = archive(); data["response_covariance"] = np.ones((2, 8))
    with pytest.raises(ValueError): selected_data(data)


def diagram():
    return SimpleNamespace(detector_z_m=.8, lenses=tuple(SimpleNamespace(name=name,
        z_center_m=.1+.15*i, length_m=.02, q_bias=900+i, q_slope_per_dac=-700-i)
        for i, name in enumerate(("IL1", "IL2", "IL3", "PL1"))))


def test_generic_q_seed_ignores_lens_excitation_attributes():
    source = diagram(); p = diagram_geometry_seed(source, 1)
    for lens in source.lenses:
        lens.q_bias = np.nan; lens.q_slope_per_dac = np.nan
    np.testing.assert_array_equal(diagram_geometry_seed(source, 1), p)
    np.testing.assert_array_equal(p[EXCITATION], [-1.5, -.5, 1.5, .5, 1.5, .5, 1.5, .5])


def test_two_geometry_starts_are_nontrivially_displaced():
    minus, plus = [diagram_geometry_seed(diagram(), sign) for sign in (-1, 1)]
    assert np.min(abs(plus[GEOMETRY]-minus[GEOMETRY])) >= .05999999
    np.testing.assert_array_equal(minus[EXCITATION], plus[EXCITATION])
    np.testing.assert_array_equal(plus[ORDER][INVERSE], plus)
    with pytest.raises(ValueError): diagram_geometry_seed(diagram(), 0)
