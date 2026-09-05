import importlib.util
from pathlib import Path
import sys
import tomllib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.neoarm_calibration import hard_edge_column_from_config

ROOT = Path(__file__).parents[1]
EXAMPLES = ROOT / "examples/lens_inversion"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
from neoarm_measured_initialization import (attempt_bounded_diversity, conjugacy_b,
    first_conjugate_strength, initialize_measured_imaging, scan_bounded_imaging_diversity)
from neoarm_imaging_recipe import column_with_geometry_anchors, imaging_metrics


@pytest.fixture(scope="module")
def measured_column():
    original = hard_edge_column_from_config(tomllib.loads(
        (EXAMPLES / "neoarm_inversion_config.toml").read_text()))
    return column_with_geometry_anchors(original,
        ROOT / "examples/microscope_models/measure_microscope_geometry.ipynb")


@pytest.fixture(scope="module")
def initialized(measured_column):
    return initialize_measured_imaging(measured_column)


def test_first_conjugate_root_and_implicit_derivative():
    inputs = jnp.array([.002, .05, .018])
    f = lambda x: first_conjugate_strength(x[0], x[1], x[2])
    q = float(f(inputs))
    assert 0 < q < np.pi
    assert abs(float(conjugacy_b(q, *inputs))) < 1e-14
    # Independently differentiate the root with finite perturbations of each
    # physical drift/body length; do not compare AD with the same AD formula.
    eps = 1e-7
    numeric = [(float(f(inputs + eps*d)) - float(f(inputs - eps*d))) / (2*eps)
               for d in np.eye(3)]
    np.testing.assert_allclose(jax.jacfwd(f)(inputs), numeric, rtol=2e-8)
    np.testing.assert_allclose(jax.jacrev(f)(inputs), numeric, rtol=2e-8)


def test_measured_geometry_first_branch_connected_40k_80k(initialized, measured_column):
    r = initialized
    np.testing.assert_allclose(r.nominal_metrics[:, 0], [40000., 80000.], rtol=1e-9)
    np.testing.assert_allclose(r.nominal_metrics[:, 1], 0., atol=1e-12)
    assert np.all(np.diff(r.connected_metrics[:, 0]) > 0)
    np.testing.assert_allclose(r.connected_metrics[:, 1], 0., atol=1e-12)
    np.testing.assert_array_equal(r.nominal_controls[0], 0.)
    np.testing.assert_array_equal(r.connected_controls[:, :3], 0.)
    for lens_before, lens_after in zip(measured_column.lenses, r.column.lenses):
        assert lens_before.z_center_m == lens_after.z_center_m
        assert lens_before.length_m == lens_after.length_m
        assert lens_before.q_slope_per_dac == lens_after.q_slope_per_dac
        if lens_before.name not in {"Obj-post", "IL1", "IL2", "IL3", "PL1"}:
            assert lens_before == lens_after
        else:
            controls = 0. if lens_after.channel_index is None else r.connected_controls[:, lens_after.channel_index]
            q = lens_after.q_bias + lens_after.q_slope_per_dac*(controls-lens_after.reference_dac)
            assert np.all(np.abs(q) < np.pi)
    objective_before = next(l for l in measured_column.lenses if l.name == "Obj-post")
    objective_after = next(l for l in r.column.lenses if l.name == "Obj-post")
    assert objective_before.q_bias == -.75
    assert objective_after.q_bias == pytest.approx(-1.55224846, abs=1e-8)
    assert objective_after.channel_index is None
    # Verify against direct matrix multiplication, not the stage magnification.
    actual = np.stack([np.asarray(imaging_metrics(r.column.matrix(r.parameters, u,
                      segment="sample_to_detector"))) for u in r.nominal_controls])
    np.testing.assert_allclose(actual, r.nominal_metrics, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("kwargs", [dict(magnifications=(80000., 40000.)),
    dict(magnifications=(1e9,)), dict(relay_fraction_bounds=(0., .99)),
    dict(objective_image_gap_m=1.), dict(continuation_points=1)])
def test_invalid_or_unestablished_relay_is_not_accepted(measured_column, kwargs):
    with pytest.raises(ValueError):
        initialize_measured_imaging(measured_column, **kwargs)


def test_zero_diversity_root_and_failed_limited_search_remain_distinct(initialized):
    zero = attempt_bounded_diversity(initialized, target_defocus_m=[0.], continuation_steps=1)
    assert zero[0]["path_established"]
    # No optimization budget: a nominal B/A=0 state cannot prove ±100 µm.
    fail = attempt_bounded_diversity(initialized, target_defocus_m=[100e-6],
                                      continuation_steps=1, max_nfev=1)
    assert not fail[0]["path_established"]
    assert fail[0]["continuation"][0]["optimizer_status"] == "evaluation_limit"


def test_bounded_imaging_path_is_not_an_arbitrary_specimen_defocus(initialized):
    result = scan_bounded_imaging_diversity(initialized, control_half_width=.2, steps_per_side=2)
    assert all(path["endpoint_established"] for path in result["paths"])
    lo, hi = result["verified_path_defocus_range_m"]
    assert -1e-6 < lo < 0 < hi < 1e-6  # Not the old surrogate's ±100 µm.
    for path in result["paths"]:
        for point in path["points"]:
            u = np.asarray(point["controls"])
            np.testing.assert_array_equal(u[[0, 1, 2, 4, 5]], 0.)
            assert np.max(np.abs(u)) <= .2
            assert point["achieved_metrics"][0] == pytest.approx(40000., rel=1e-9)
    assert not result["global_maximum_proved"]
    assert not result["hardware_commands_admitted"]
