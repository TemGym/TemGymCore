import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_condenser_geometry_profile_diagnostic import (
    GEOMETRY_INDICES, NUISANCE_INDICES, geometry_first, original_order,
)


def test_geometry_strength_partition_preserves_every_coordinate_once():
    assert GEOMETRY_INDICES == (0, 1, 4, 5, 8, 9)
    assert len(NUISANCE_INDICES) == 17
    assert sorted(GEOMETRY_INDICES+NUISANCE_INDICES) == list(range(23))
    p = jnp.arange(23.)
    np.testing.assert_array_equal(original_order(geometry_first(p)), p)
    np.testing.assert_array_equal(geometry_first(p)[:6], np.array(GEOMETRY_INDICES))


def test_reorder_is_jittable_and_has_exact_inverse_derivative():
    transform = jax.jit(lambda p: original_order(geometry_first(p)))
    p = jnp.arange(23.)/7
    np.testing.assert_array_equal(transform(p), p)
    np.testing.assert_array_equal(jax.jacfwd(transform)(p), np.eye(23))
