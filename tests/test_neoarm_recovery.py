"""Scaled optimizer and physical branch seeds on small deterministic problems."""

from pathlib import Path
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import json

sys.path.insert(0, str(Path(__file__).parents[1] / "examples/lens_inversion"))
import neoarm_recovery as recovery
from neoarm_recovery import trust_region_fit, polish_in_likelihood_coordinates, adjacent_excitation_starts

jax.config.update("jax_enable_x64", True)


def test_raw_least_squares_and_sensitivity_polish_keep_bounds_and_all_directions():
    scales = np.array([1e-6, 1., 1.])
    target = np.array([.4e-6, .3, .0])
    matrix = jnp.array([[2e7, .2, 0.], [1e6, 3., 0.], [0., 1., 0.]])
    residual = jax.jit(lambda p:matrix@(p-target))
    bounds = (np.array([-1e-6, -.2, -.5]), np.array([1e-6, .2, .5]))
    problem = SimpleNamespace(scales=scales, residual=residual,
                              jacobian=jax.jit(jax.jacfwd(residual)), bounds=lambda:bounds)
    initial = np.array([-.4e-6, -.1, .15])
    fitted, fit = trust_region_fit(problem, initial)
    polished, result = polish_in_likelihood_coordinates(problem, fitted)
    assert fit["success"] and result["success"]
    assert result["directions_removed"] == 0
    assert np.all(polished >= bounds[0]-1e-12) and np.all(polished <= bounds[1]+1e-12)
    np.testing.assert_allclose(polished[1], .2, atol=1e-8)
    np.testing.assert_allclose(polished[2], .15, atol=1e-8)
    assert result["cost"] < float(.5*jnp.sum(residual(initial)**2))


def test_branch_seeds_preserve_anchor_body_alias_without_clipping():
    lens = SimpleNamespace(name="IL1", channel_index=0, reference_dac=0.)
    column = SimpleNamespace(lenses=[lens], parameter_names=[
        "IL1.q_bias", "IL1.q_slope_per_dac", "IL1.log_length_m"])
    bounds = (np.array([-1., .3, np.log(.005)]), np.array([1., .7, np.log(.08)]))
    problem = SimpleNamespace(column=column, bounds=lambda:bounds,
        image=SimpleNamespace(dacs=np.array([[-59.], [-61.]]), train_indices=np.array([0, 1])))
    reference = np.array([.2, .5, np.log(.02)])
    candidates = adjacent_excitation_starts(problem, reference)
    assert len(candidates) == 2
    for label, candidate in candidates:
        q = candidate[0]-60*candidate[1]
        np.testing.assert_allclose(q-label["anchor_q"], label["order"]*np.pi, atol=1e-13)
        np.testing.assert_allclose(np.exp(candidate[2]-reference[2]), q/label["anchor_q"])
        assert np.all(candidate >= bounds[0]) and np.all(candidate <= bounds[1])
    np.testing.assert_array_equal(reference, [.2, .5, np.log(.02)])


def test_failed_truth_gate_never_starts_optimizer(tmp_path, monkeypatch):
    problem_path, report_path = tmp_path / "problem.json", tmp_path / "report.json"
    problem_path.write_text(json.dumps({"sha256":"test"}))
    fake = SimpleNamespace(truth=np.zeros(2), diagnostics=lambda _: {
        "image_training_rms_sigma":100., "condenser_training_rms_sigma":1.})
    monkeypatch.setattr(recovery, "load_problem", lambda _: {})
    monkeypatch.setattr(recovery, "CoupledProblem", lambda *args: fake)
    def forbidden(*args, **kwargs):
        pytest.fail("An inconsistent synthetic likelihood must never enter the optimizer.")
    monkeypatch.setattr(recovery, "trust_region_fit", forbidden)
    monkeypatch.setattr(sys, "argv", ["neoarm_recovery", "--problem", str(problem_path),
                                    "--output", str(report_path), "--fit"])
    with pytest.raises(RuntimeError, match="truth fails"):
        recovery.main()
    assert json.loads(report_path.read_text())["truth"]["image_training_rms_sigma"] == 100.
