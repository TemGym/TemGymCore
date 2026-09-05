from dataclasses import replace
from pathlib import Path
import sys
import tomllib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.immersed_objective import SharedObjectiveField, mechanical_to_canonical
from temgym_core.microscope_inversion import drift_matrix, hard_edge_solenoid_matrix
from temgym_core.neoarm_calibration import hard_edge_column_from_config

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples/lens_inversion"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
from neoarm_imaging_recipe import column_with_geometry_anchors
from neoarm_shared_objective import (
    SharedObjectiveColumn, initialize_shared_imaging, shared_imaging_metrics,
    shared_projection_states,
)


@pytest.fixture(scope="module")
def base():
    config = tomllib.loads((EXAMPLES/"neoarm_inversion_config.toml").read_text())
    return column_with_geometry_anchors(hard_edge_column_from_config(config),
        ROOT/"examples/microscope_models/measure_microscope_geometry.ipynb")


@pytest.fixture(scope="module")
def column(base):
    field = SharedObjectiveField.from_window(-.05, 0., .05, n_slices_per_side=64)
    return SharedObjectiveColumn.from_hard_edge(base, objective=field)


@pytest.fixture(scope="module")
def initialized(column):
    return initialize_shared_imaging(column)


def test_no_independent_objective_parameters_and_no_input_mutation(base, column):
    assert len(base.lenses) == 9 and len(column.lenses) == 7
    assert column.n_parameters == 34
    assert not any("Obj-" in name for name in column.parameter_names)
    assert column.parameter_names[-4:] == ("objective.q_total", "objective.z_peak_m",
                                          "objective.log_width_m", "objective.asymmetry")
    assert all(lens.channel_index is not None for lens in column.lenses)
    np.testing.assert_array_equal(column.nominal_parameters[-4:], column.objective_initial)


def test_complete_column_sample_conversions_cancel(column):
    p, u = jnp.asarray(column.nominal_parameters), jnp.linspace(-.01, .01, 7)
    full = column.matrix(p, u)
    for convention in ("mechanical", "canonical"):
        pre = column.matrix(p, u, segment="source_to_sample", coordinates=convention)
        post = column.matrix(p, u, segment="sample_to_detector", coordinates=convention)
        np.testing.assert_allclose(post@pre, full, rtol=2e-10, atol=1e-9)
    ray = jnp.array([1e-8, 20e-6, -2e-8, 15e-6])
    np.testing.assert_allclose(column.trace(p, u, ray), full@ray, rtol=2e-11, atol=1e-14)


def test_mechanical_basis_has_current_field_information(column):
    p, u = jnp.asarray(column.nominal_parameters), jnp.zeros(7)
    canonical = column.matrix(p, u, segment="sample_to_detector", coordinates="canonical")
    mechanical = column.matrix(p, u, segment="sample_to_detector")
    k = column.objective.field(column.sample_z_m, p[-4:])
    np.testing.assert_allclose(mechanical, canonical@mechanical_to_canonical(k), rtol=2e-12, atol=1e-10)
    np.testing.assert_allclose(mechanical[:, [1, 3]], canonical[:, [1, 3]], rtol=2e-12)
    assert np.max(np.abs(np.asarray(mechanical[:, [0, 2]]-canonical[:, [0, 2]]))) > 1.
    equivalent = np.asarray(shared_imaging_metrics(mechanical))[2:].reshape(2, 2)
    assert abs(equivalent[1, 0]-equivalent[0, 1]) > 1e-7


def test_single_ray_matrix_and_nested_parameter_derivatives(column):
    p, u = jnp.asarray(column.nominal_parameters), jnp.zeros(7)
    ray = jnp.array([1e-8, 20e-6, -2e-8, 15e-6])
    transfer = lambda v: column.matrix(v, u, segment="sample_to_detector")@ray
    exact = np.asarray(jax.jit(jax.jacfwd(transfer))(p))
    for index, delta in ((14, 1e-7), (16, 1e-5), (30, 1e-5), (31, 1e-8), (32, 1e-5), (33, 1e-5)):
        shifted = np.eye(column.n_parameters)[index]*delta
        numeric = (np.asarray(transfer(p+shifted))-np.asarray(transfer(p-shifted)))/(2*delta)
        np.testing.assert_allclose(exact[:, index], numeric, rtol=3e-6, atol=1e-7)
    values = jax.jit(jax.vmap(lambda state: column.trace(p, u, state)))(jnp.stack((ray, -ray)))
    np.testing.assert_allclose(values[0], -values[1], rtol=2e-12, atol=1e-14)


def test_aperture_transfer_in_drift_and_partial_cl3_body(column):
    p, controls = column.nominal_parameters, np.zeros(7)
    for fraction in (.25, .75, 1.):
        first, second = column.lenses[:2]
        target = second.entrance_z+fraction*second.length_m
        expected = (hard_edge_solenoid_matrix(second.q_bias*fraction, second.length_m*fraction)
            @ drift_matrix(second.entrance_z-first.exit_z)
            @ hard_edge_solenoid_matrix(first.q_bias, first.length_m)
            @ drift_matrix(first.entrance_z-column.source_z_m))
        actual = column.matrix_to_plane(p, controls, target)
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
    target = second.exit_z+.0005
    expected = drift_matrix(.0005)@expected
    np.testing.assert_allclose(column.matrix_to_plane(p, controls, target), expected, rtol=1e-11, atol=1e-12)
    derivative = jax.jit(jax.jacfwd(lambda z: column.matrix_to_plane(p, controls, z)))(second.entrance_z)
    assert np.all(np.isfinite(derivative))
    assert np.all(np.isnan(column.matrix_to_plane(p, controls, 0.)))


def test_bound_boxes_keep_lenses_out_of_fixed_objective_window(column):
    lower, upper = column.default_bounds()
    assert np.all(lower < column.nominal_parameters) and np.all(column.nominal_parameters < upper)
    rng = np.random.default_rng(31)
    points = lower+(upper-lower)*rng.random((512, column.n_parameters))
    clearances = np.asarray(jax.jit(jax.vmap(column.geometry_clearances))(jnp.asarray(points)))
    assert np.all(clearances >= 0.)
    for bad in (dict(length_ratio=(1., 2.)), dict(position_half_width_m=0.), dict(q_half_width=-1.)):
        with pytest.raises(ValueError):
            column.default_bounds(**bad)


def test_connected_40k_80k_shared_field_initialization(initialized, column):
    result = initialized
    np.testing.assert_allclose(result.nominal_metrics[:, 0], [40000., 80000.], rtol=1e-9)
    np.testing.assert_allclose(result.nominal_metrics[:, 2:], 0., atol=1e-12)
    np.testing.assert_allclose(result.connected_metrics[:, 2:], 0., atol=1e-12)
    assert np.all(np.diff(result.connected_metrics[:, 0]) > 0)
    np.testing.assert_array_equal(result.nominal_controls[0], 0.)
    np.testing.assert_array_equal(result.connected_controls[:, :3], 0.)
    np.testing.assert_array_equal(result.parameters[-3:], column.nominal_parameters[-3:])
    assert result.parameters[-4] < 0
    assert result.objective_root_bracket[0] < abs(result.parameters[-4]) < result.objective_root_bracket[1]
    for lens in result.column.lenses[3:]:
        phases = lens.q_bias + lens.q_slope_per_dac*(result.connected_controls[:, lens.channel_index]-lens.reference_dac)
        assert np.max(np.abs(phases)) < np.pi


def test_connected_projection_states_retain_non_scalar_response(initialized):
    result = shared_projection_states(initialized, steps_per_side=2)
    assert result["all_paths_established"]
    controls = np.asarray(result["controls"])
    np.testing.assert_array_equal(controls[:, [0, 1, 2, 4, 5]], 0.)
    assert np.max(np.abs(controls)) <= .2
    metrics = np.asarray(result["metrics"])
    np.testing.assert_allclose(metrics[:, 0], 40000., rtol=1e-9)
    # The complete off-focus response is not a scalar specimen drift.
    assert np.max(np.abs(metrics[[0, 2], 3:5])) > 1e-14
    assert not result["hardware_commands_admitted"]


@pytest.mark.parametrize("kwargs", [dict(magnifications=(80000., 40000.)),
    dict(magnifications=(1e10,)), dict(objective_sign=0.), dict(continuation_points=1)])
def test_invalid_or_unestablished_initialization_is_not_accepted(column, kwargs):
    with pytest.raises(ValueError):
        initialize_shared_imaging(column, **kwargs)


def test_overlap_of_shared_support_is_rejected(column):
    with pytest.raises(ValueError, match="overlap"):
        replace(column, objective=SharedObjectiveField.from_window(-.15, 0., .05, n_slices_per_side=16))

