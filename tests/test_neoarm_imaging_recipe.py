"""Single-ray and actual-control checks for the compact imaging design."""

import importlib.util
import json
from pathlib import Path
import sys
import tomllib
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.neoarm_calibration import hard_edge_column_from_config

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "neoarm_imaging_recipe", ROOT / "examples/lens_inversion/neoarm_imaging_recipe.py")
RECIPE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RECIPE
SPEC.loader.exec_module(RECIPE)
jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def column():
    return hard_edge_column_from_config(tomllib.loads(
        (ROOT / "examples/lens_inversion/neoarm_inversion_config.toml").read_text()))


@pytest.mark.parametrize("segment", ["full", "source_to_sample", "sample_to_detector"])
def test_single_ray_jacobian_matches_independent_matrix_composition(column, segment):
    p, d = column.nominal_parameters, np.linspace(-.2, .2, 7)
    expected = column.matrix(p, d, segment=segment)
    actual = RECIPE.single_ray_transfer(column, p, d, segment=segment)
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=1e-10)
    ray = np.array([3e-9, 2e-6, -1e-8, 4e-6])
    np.testing.assert_allclose(RECIPE.trace_column_ray(column, p, d, ray, segment=segment),
                               expected @ ray, rtol=3e-12, atol=1e-15)


def test_control_gradient_through_ray_jacobian_is_correct(column):
    p, d = column.nominal_parameters, np.zeros(7)
    from_ray = jax.jacfwd(lambda u: RECIPE.single_ray_transfer(column, p, u))(d)
    from_matrix = jax.jacfwd(lambda u: column.matrix(p, u, segment="sample_to_detector"))(d)
    np.testing.assert_allclose(from_ray, from_matrix, rtol=3e-12, atol=1e-9)
    np.testing.assert_array_equal(np.asarray(from_ray)[..., :3], 0)


def test_actual_controls_reproduce_requested_diversity_and_fixed_magnification(column):
    p, d = column.nominal_parameters, np.zeros(7)
    reference = np.asarray(RECIPE.imaging_metrics(column.matrix(p, d, segment="sample_to_detector")))
    targets = reference[1] + np.array([-1e-6, 0., 1e-6])
    result = RECIPE.design_imaging_diversity(column, p, d, defocus_m=targets, continuation_steps=1)
    np.testing.assert_allclose(result.achieved_metrics[:, 0], reference[0], rtol=1e-7)
    np.testing.assert_allclose(result.achieved_metrics[:, 1], targets, atol=1e-10)
    np.testing.assert_array_equal(result.controls[:, [0, 1, 2, 3, 6]], np.zeros((3, 5)))
    assert np.max(np.abs(result.controls)) <= .05
    assert all(row["constraint_satisfied"] for row in result.continuation)
    assert not result.parameter_fit_admitted


def test_calibration_sample_is_resolved_at_both_proposed_magnifications():
    rows = RECIPE.sample_scale_table((40000, 80000))
    np.testing.assert_allclose([row["sample_fov_nm"] for row in rows], [352., 176.])
    np.testing.assert_allclose([row["sample_pixel_nm"] for row in rows], [1.375, .6875])
    assert all(3 < row["period_detector_pixels"] < 96 for row in rows)
    assert all(row["periods_across_field"] > 2 for row in rows)


def test_geometry_notebook_is_read_as_data_and_discrepancy_retained(column):
    path = ROOT / "examples/microscope_models/measure_microscope_geometry.ipynb"
    anchors = RECIPE.geometry_anchors_from_notebook(path)
    assert anchors["CL1"] == pytest.approx(-.5104166666666666)
    assert anchors["CL3"] == pytest.approx(-.42916666666666664)
    assert anchors["C-aperture"] == pytest.approx(-.42083333333333334)
    comparison = {row["component"]: row for row in RECIPE.compare_geometry(column, path)}
    assert abs(comparison["CL1"]["difference_m"]) > .38
    assert "stigmator" not in anchors
    rebuilt = RECIPE.column_with_geometry_anchors(column, path)
    assert rebuilt.sample_z_m == 0.
    assert rebuilt.source_z_m == pytest.approx(-.6770833333333334)
    assert rebuilt.detector_z_m == pytest.approx(.71875)
    assert np.min(rebuilt.geometry_clearances(rebuilt.nominal_parameters)) >= 0
    assert column.sample_z_m == .325  # no mutation or checkpoint relabelling
    for before, after in zip(column.lenses, rebuilt.lenses):
        assert after.length_m == before.length_m
        assert after.q_bias == before.q_bias
        assert after.z_center_m == anchors[after.name]


@pytest.mark.parametrize("kwargs", [dict(selected_channels=("CL1", "IL2")),
                                    dict(control_half_width=-1), dict(defocus_m=[np.nan]),
                                    dict(continuation_steps=0)])
def test_invalid_or_upstream_control_design_is_rejected(column, kwargs):
    with pytest.raises(ValueError):
        RECIPE.design_imaging_diversity(column, column.nominal_parameters, np.zeros(7), **kwargs)


def test_invalid_upstream_inputs_and_overlaps_fail_before_downstream_solve(column):
    p = column.nominal_parameters.copy()
    bad = p.copy(); bad[2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        RECIPE.design_imaging_diversity(column, bad, np.zeros(7))
    p[2] = .4  # CL1 now overlaps its neighbour/the specimen in fixed topology.
    with pytest.raises(ValueError, match="geometry"):
        RECIPE.design_imaging_diversity(column, p, np.zeros(7))
    d = np.zeros(7); d[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        RECIPE.design_imaging_diversity(column, column.nominal_parameters, d)


def test_control_root_admission_uses_forward_constraints_not_optimizer_label(column, monkeypatch):
    import temgym_core.jax_least_squares as solver
    p, d = column.nominal_parameters, np.zeros(7)
    delta = float(RECIPE.imaging_metrics(RECIPE.single_ray_transfer(column, p, d))[1])
    monkeypatch.setattr(solver, "least_squares", lambda f, x, **kw:
                        SimpleNamespace(x=x, success=False, status="stalled"))
    result = RECIPE.design_imaging_diversity(column, p, d, defocus_m=[delta], continuation_steps=1)
    assert result.continuation[0]["constraint_satisfied"]
    assert not result.continuation[0]["optimizer_success"]
    monkeypatch.setattr(solver, "least_squares", lambda f, x, **kw:
                        SimpleNamespace(x=x, success=True, status="xtol"))
    with pytest.raises(RuntimeError, match="local failure is not proof"):
        RECIPE.design_imaging_diversity(column, p, d, defocus_m=[delta+1e-3], continuation_steps=1)


@pytest.mark.parametrize("source", [
    "label_y = __import__('builtins').print('must not execute')",
    "length_scale_px=96\nlength_scale_px=100",
    "length_scale_px=96\nlength_scale_m=__import__('builtins').print('must not execute')",
    "length_scale_px=96",
])
def test_notebook_expressions_are_never_executed(tmp_path, capsys, source):
    path = tmp_path / "geometry.ipynb"
    path.write_text(json.dumps({"cells": [{"cell_type": "code", "source": source}]}))
    with pytest.raises(ValueError):
        RECIPE.geometry_anchors_from_notebook(path)
    assert capsys.readouterr().out == ""
