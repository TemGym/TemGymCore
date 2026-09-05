import sys
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"examples/lens_inversion"))
import neoarm_condenser_reference_recovery as recovery


def install_result(monkeypatch, parameters, residual, success=True):
    chart = SimpleNamespace(encode=jnp.asarray, decode=jnp.asarray,
        reference_control=np.zeros(7), reference_width_m=1e-6)
    monkeypatch.setattr(recovery, "fixed_reference_chart", lambda: chart)
    monkeypatch.setattr(recovery, "refine_training", lambda *args, **kwargs: dict(
        parameters=np.asarray(parameters), residual=np.asarray(residual),
        success=success, reason="mock diagnostic", history=[]))


def test_original_stationarity_required_even_if_transformed_solver_claims_success(monkeypatch):
    install_result(monkeypatch, np.ones(23), np.ones(23))
    result = recovery.source_reference_solver(lambda p: p, np.ones(23), lambda p: True)
    assert not result["success"]
    assert result["normalized_gradient"] == 1.


def test_original_likelihood_closure_required(monkeypatch):
    install_result(monkeypatch, np.zeros(23), np.ones(23))
    result = recovery.source_reference_solver(lambda p: p, np.ones(23), lambda p: True)
    assert not result["success"]
    assert result["history"][0]["final_residual_closure_sigma"] == 1.


def test_stationary_coordinate_equivalent_fit_accepted(monkeypatch):
    install_result(monkeypatch, np.zeros(23), np.zeros(23))
    result = recovery.source_reference_solver(lambda p: p, np.ones(23), lambda p: True)
    assert result["success"]
    np.testing.assert_array_equal(result["jacobian"], np.eye(23))
    assert result["history"][0]["coordinate_change_only"]


def test_unrecognized_solver_does_not_silently_fall_back(monkeypatch):
    install_result(monkeypatch, np.zeros(23), np.zeros(23))
    with pytest.raises(ValueError, match="Select"):
        recovery.source_reference_solver(lambda p: p, np.zeros(23), lambda p: True,
                                         method="unknown")
