from pathlib import Path
import sys
import tomllib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"examples/lens_inversion"))

from neoarm_imaging_recipe import column_with_geometry_anchors
from neoarm_objective_tail_overlap import TailOverlapColumn, BODY_INDICES
from neoarm_shared_objective import SharedObjectiveColumn
from temgym_core.immersed_objective import SharedObjectiveField
from temgym_core.neoarm_calibration import hard_edge_column_from_config


@pytest.fixture
def tail_column():
    config = tomllib.loads((ROOT/"examples/lens_inversion/neoarm_inversion_config.toml").read_text())
    measured = column_with_geometry_anchors(hard_edge_column_from_config(config),
        ROOT/"examples/microscope_models/measure_microscope_geometry.ipynb")
    field = SharedObjectiveField.from_window(-.05, 0., .05, n_slices_per_side=32)
    shared = SharedObjectiveColumn.from_hard_edge(measured, objective=field)
    p = jnp.asarray(shared.nominal_parameters)
    return TailOverlapColumn.from_shared(shared, p, steps_per_segment=4), p


def test_explicit_boundaries_single_ray_and_endpoint_composition(tail_column):
    column, p = tail_column
    u = jnp.zeros(7)
    edges, rates = column.geometry_and_rates(p, u)
    assert edges.shape == (19,)
    assert rates.shape == (7,)
    assert column.mesh.segment_body_indices == BODY_INDICES
    assert edges[8] == column.base.sample_z_m
    assert np.all(np.diff(edges) > 0)
    maps = jax.jit(column.split_maps)(p, u)
    downstream = jax.jit(lambda v: column.matrix(p, v, segment='sample_to_detector'))(u)
    np.testing.assert_allclose(downstream, maps.post_mechanical, rtol=2e-13, atol=1e-12)
    np.testing.assert_allclose(column.matrix(p, u), maps.post_mechanical@maps.pre_mechanical,
                               rtol=2e-12, atol=3e-10)


def test_zero_tail_recovers_existing_lenses_and_partial_aperture_transport(tail_column):
    column, p = tail_column
    p = p.at[-4].set(0.)
    u = jnp.linspace(-.02, .02, 7)
    np.testing.assert_allclose(column.matrix(p, u), column.base.matrix(p, u), rtol=2e-13, atol=1e-11)
    z = column.base.lenses[1].exit_z+1e-6
    np.testing.assert_allclose(column.matrix_to_plane(p, u, z), column.base.matrix_to_plane(p, u, z),
                               rtol=2e-13, atol=1e-12)
    assert np.all(np.isfinite(jax.jacfwd(lambda values: column.matrix_to_plane(values, u, z))(p)))
