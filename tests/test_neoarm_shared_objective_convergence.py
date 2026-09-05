from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"examples/lens_inversion"))

from neoarm_shared_objective_convergence import (
    compare_images, preserve_profile_amplitude, tangent_objective,
)
from temgym_core.immersed_objective import SharedObjectiveField


def test_fixed_tangent_mesh_and_amplitude_preservation():
    reference = SharedObjectiveField.from_window(-.05, 0., .05, n_slices_per_side=128)
    candidate = tangent_objective(64, bounds=(-.075, .15))
    assert candidate.sample_index == 64
    assert len(candidate.z_edges_m) == 129
    assert candidate.z_edges_m[0] == -.075
    assert candidate.z_edges_m[-1] == .15
    assert candidate.z_edges_m[64] == 0.
    assert np.all(np.diff(candidate.z_edges_m) > 0)
    p = jnp.array([42., -5.7, -.0005, np.log(.002), .15])
    transformed = preserve_profile_amplitude(p, candidate, reference)
    np.testing.assert_array_equal(transformed[jnp.array([0, 2, 3, 4])], p[jnp.array([0, 2, 3, 4])])
    z = jnp.linspace(-.049, .049, 31)
    np.testing.assert_allclose(candidate.field(z, transformed[-4:]), reference.field(z, p[-4:]), rtol=3e-15)
    assert candidate.field(.1, transformed[-4:]) != 0
    assert reference.field(.1, p[-4:]) == 0


@pytest.mark.parametrize("n", [0, -1, True, 2.5])
def test_invalid_tangent_mesh(n):
    with pytest.raises(ValueError):
        tangent_objective(n)


def test_observable_gate_cannot_be_replaced_by_pixel_rms_gate():
    reference = dict(image_parameters=np.zeros((1, 6)), images=np.ones((1, 4, 4)))
    candidate = dict(image_parameters=np.array([[.3, 0, 0, 0, 0, 0]]), images=np.ones((1, 4, 4)))
    result = compare_images(candidate, reference, np.eye(6)[None])
    assert result["pixel_gate_passed"]
    assert not result["joint_observable_gate_passed"]
    assert result["maximum_centroid_shift_sigma"] == pytest.approx(.3)
    identical = compare_images(reference, reference, np.eye(6)[None])
    assert identical["joint_observable_gate_passed"]
