"""Small-matrix contracts for the same-column16-coordinate adapter."""
from pathlib import Path
from types import SimpleNamespace
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1]/"examples/lens_inversion"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from neoarm_shared_projector_calibration import SharedProjectorModel, PARAMETER_NAMES, GROUPED
from temgym_core.microscope_inversion import drift_matrix
from temgym_core.immersed_objective import uniform_field_canonical_matrix, mechanical_to_canonical

jax.config.update("jax_enable_x64", True)


def model_and_stages():
    names = ("CL1", "CL3", "C-mini", "IL1", "IL2", "IL3", "PL1")
    objective = SimpleNamespace(z_edges_m=np.array([-.02, 0., .02]),
        split_maps=lambda p: SimpleNamespace(post=drift_matrix(.003), sample_k_m_inv=.4))
    column = SimpleNamespace(channel_names=names,
        lenses=tuple(SimpleNamespace(reference_dac=.03*i) for i in range(7)), objective=objective)
    p = np.zeros(34); p[:2] = [-.4, .8]
    for i in range(7):
        p[2+4*i:6+4*i] = [-.25+.1*i, np.log(.01), -1.2 if i == 6 else .6+.1*i, .5]
    model = SharedProjectorModel(column, p)
    return model, model.stage_coordinates(p)


def independent_native(model, stages, controls):
    p = model.native_parameters(stages)
    result = model.post
    cursor = model.objective_exit_m
    for i in range(3, 7):
        z, log_length, bias, slope = p[2+4*i:6+4*i]
        length = jnp.exp(log_length)
        q = bias+slope*(controls[i]-model.column.lenses[i].reference_dac)
        result = uniform_field_canonical_matrix(q/length, length)@drift_matrix(z-length/2-cursor)@result
        cursor = z+length/2
    return drift_matrix(model.detector_z_m-cursor)@result


def test_native_roundtrip_preserves_geometry_and_affine_laws():
    model, stages = model_and_stages()
    assert len(PARAMETER_NAMES) == 16
    np.testing.assert_allclose(model.native_parameters(stages), model.fixed, atol=1e-15)
    np.testing.assert_allclose(model.stage_coordinates(model.native_parameters(stages)), stages, atol=1e-14)


@pytest.mark.parametrize("offset", [0., .03, -.025])
def test_moving_geometry_closes_with_independent_native_sequence(offset):
    model, stages = model_and_stages()
    moving = stages+np.tile([offset, -offset, .7*offset, -.3*offset], 4)
    control = jnp.arange(7)*.04
    actual = model.matrix(moving, control)
    expected = independent_native(model, moving, control)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=2e-12)
    np.testing.assert_allclose(model.matrix(moving, control, coordinates="mechanical"),
        expected@mechanical_to_canonical(model.kappa_per_m), rtol=1e-12, atol=2e-12)


def test_geometry_derivatives_include_the_moving_objective_to_il1_drift():
    model, stages = model_and_stages()
    control = jnp.zeros(7)
    auto = jax.jacfwd(model.matrix)(jnp.asarray(stages), control)
    independent = jax.jacfwd(lambda p: independent_native(model, p, control))(jnp.asarray(stages))
    np.testing.assert_allclose(auto, independent, rtol=1e-11, atol=2e-10)
    assert np.linalg.norm(np.asarray(auto[:, :, 1])) > 0


def test_upstream_condenser_controls_cannot_affect_sample_response():
    model, stages = model_and_stages()
    control = jnp.zeros((2, 7)).at[1, :3].set(jnp.array([10., -20., 30.]))
    response = model.predict(stages, control)
    np.testing.assert_array_equal(response[0], response[1])


def test_geometry_overlap_and_coordinate_shape_rejected():
    model, stages = model_and_stages()
    assert model.admissible(stages)
    assert not model.admissible(stages+np.tile([2., 2., 0., 0.], 4))
    with pytest.raises(ValueError):
        model.matrix(np.zeros(17), jnp.zeros(7))
    with pytest.raises(ValueError):
        model.matrix(stages, jnp.zeros(7), coordinates="unspecified")
