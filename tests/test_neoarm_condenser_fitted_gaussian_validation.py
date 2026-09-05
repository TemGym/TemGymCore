"""Frozen selection, no-observation access and strongly chirped source checks."""
from pathlib import Path
import sys

import jax
import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1]/"examples/lens_inversion"
if str(EXAMPLES) not in sys.path: sys.path.insert(0, str(EXAMPLES))

from neoarm_condenser_fitted_gaussian_validation import selected_parameters, heldout_controls
from neoarm_condenser_gaussian_validation import covariance_to_mixed_source, rank_adapted_source_modes

jax.config.update("jax_enable_x64", True)


def selection_pair():
    p = list(np.linspace(-.2, .3, 23))
    report = dict(continuation=dict(original_stationary_success=True, original_parameters=p), heldout_observations_used=False)
    selection = dict(no_heldout_or_generating_values_used=True, fit_report_sha256="fixed", parameters=p.copy())
    return report, selection


def test_exact_stationary_training_selection_required():
    report, selection = selection_pair()
    np.testing.assert_array_equal(selected_parameters(report, selection, "fixed"), selection["parameters"])
    report["continuation"]["original_stationary_success"] = False
    with pytest.raises(ValueError, match="stationary"): selected_parameters(report, selection, "fixed")
    report, selection = selection_pair(); report["heldout_observations_used"] = True
    with pytest.raises(ValueError, match="stationary"): selected_parameters(report, selection, "fixed")


def test_changed_fit_hash_or_parameters_rejected():
    report, selection = selection_pair()
    with pytest.raises(ValueError, match="previously frozen"): selected_parameters(report, selection, "other")
    selection["parameters"][0] += 1e-9
    with pytest.raises(ValueError, match="parameters changed"): selected_parameters(report, selection, "fixed")


def test_only_roles_and_controls_are_read_not_heldout_observations():
    class NoObservationArchive(dict):
        def __getitem__(self, key):
            assert key not in ("beam_observables", "noisy_beam_images", "source_covariance", "optical_parameters")
            return super().__getitem__(key)
    archive = NoObservationArchive(controls=np.arange(35).reshape(5, 7),
        training_mask=np.array([True, True, False, False, False]),
        heldout_mask=np.array([False, False, True, True, True]), analyser_controls=np.zeros((2, 7)))
    controls, analysers = heldout_controls(archive)
    np.testing.assert_array_equal(controls, archive["controls"][2:])
    assert analysers.shape == (2, 7)


def test_displaced_few_nm_waist_has_physical_rank_two_source_representation():
    waist = np.diag(np.array([5e-9, 50e-6, 6e-9, 45e-6])**2)
    drift = np.eye(4); drift[0, 1] = drift[2, 3] = .00641
    chirped = drift@waist@drift.T
    assert np.sqrt(chirped[0, 0]) > 300e-9
    source, metadata = covariance_to_mixed_source(chirped)
    _, _, modes = rank_adapted_source_modes(source, 7)
    assert modes["centre_distribution_rank"] == 2
    assert modes["coherent_mode_count"] == 49
    assert metadata["pure_plus_centres_relative_covariance_error"] < 1e-10
    np.testing.assert_allclose(source.total_covariance, chirped, rtol=1e-9, atol=1e-27)
