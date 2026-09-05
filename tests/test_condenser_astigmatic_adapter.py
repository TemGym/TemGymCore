from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.condenser_astigmatic_adapter import (
    AstigmaticFullColumnAdapter,
    default_astigmatic_outer_parameter_contract,
    make_astigmatic_full_column_adapter,
)
from temgym_core.condenser_astigmatic_likelihood import (
    make_astigmatic_condenser_likelihood,
)
from temgym_core.physical_nuisance import (
    GlobalObservationLayout,
    SharedLatentNuisanceLikelihood,
)


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).parents[1]
MANIFEST = (
    ROOT
    / "examples"
    / "lens_inversion"
    / "neoarm_condenser_astigmatic_design.json"
)


@pytest.fixture(scope="module")
def adapter() -> AstigmaticFullColumnAdapter:
    likelihood = make_astigmatic_condenser_likelihood(
        MANIFEST,
        repository_root=ROOT,
    )
    return make_astigmatic_full_column_adapter(
        likelihood,
        likelihood.branch_experiment.column,
    )


def test_adapter_has_exact_shared_and_outer_coordinate_contract(adapter):
    likelihood = adapter.likelihood
    column = adapter.shared_column

    assert adapter.optical_parameter_count == 36
    assert adapter.outer_parameter_count == 27
    assert adapter.joint_parameter_count == 63
    assert adapter.target_physical_indices == tuple(
        likelihood.branch_experiment.target_parameter_indices
    )
    assert adapter.upstream_physical_indices == tuple(
        likelihood.branch_experiment.upstream_parameter_indices
    )
    assert len(adapter.shared_physical_indices) == 16
    assert len(adapter.unobserved_physical_indices) == 20
    assert adapter.joint_parameter_names[:36] == tuple(column.parameter_names)
    assert adapter.outer_contract.native_parameter_indices == (
        *range(12, 22),
        *range(26, 43),
    )
    assert adapter.outer_contract.block_slices == {
        "entrance_covariance": slice(0, 10),
        "aperture_and_stage": slice(10, 12),
        "edge_local": slice(12, 24),
        "sample_scales": slice(24, 26),
        "astigmatic_tag_orientation": slice(26, 27),
    }
    forbidden = ("k16", "deflector", "tilt_command")
    assert not any(
        token in name.lower()
        for name in adapter.joint_parameter_names
        for token in forbidden
    )
    np.testing.assert_array_equal(
        adapter.joint_reference[:36], column.nominal_parameters
    )
    np.testing.assert_array_equal(adapter.joint_reference[36:], 0.0)
    assert np.all(adapter.outer_contract.lower < adapter.outer_contract.reference)
    assert np.all(adapter.outer_contract.reference < adapter.outer_contract.upper)
    assert np.all(adapter.outer_contract.scales > 0.0)


def test_mapping_is_exact_and_downstream_physical_coordinates_are_not_duplicated(
    adapter,
):
    branch = adapter.likelihood.branch_experiment
    physical = np.asarray(adapter.shared_column.nominal_parameters).copy()
    physical += 1.0e-4 * np.sin(np.arange(physical.size) + 1.0)
    outer = 0.03 * np.cos(np.arange(adapter.outer_parameter_count) + 1.0)

    native = np.asarray(adapter.likelihood_parameters_from_parts(physical, outer))
    np.testing.assert_allclose(
        native[:12],
        (
            physical[np.asarray(adapter.target_physical_indices)]
            - branch.column.nominal_parameters[
                np.asarray(adapter.target_physical_indices)
            ]
        )
        / branch.target_parameter_scales,
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        native[22:26],
        (
            physical[np.asarray(adapter.upstream_physical_indices)]
            - branch.column.nominal_parameters[
                np.asarray(adapter.upstream_physical_indices)
            ]
        )
        / branch.upstream_parameter_scales,
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_array_equal(
        native[np.asarray(adapter.outer_contract.native_parameter_indices)], outer
    )

    changed_downstream = physical.copy()
    changed_downstream[np.asarray(adapter.unobserved_physical_indices)] += 0.17
    np.testing.assert_array_equal(
        adapter.likelihood_parameters_from_parts(changed_downstream, outer), native
    )
    np.testing.assert_allclose(
        adapter.residual_from_parts(physical, outer),
        adapter.likelihood.residual(native),
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_coordinate_and_residual_jacobians_obey_exact_jax_chain_rule(adapter):
    joint = jnp.asarray(adapter.joint_reference)
    analytic_mapping = np.asarray(adapter.coordinate_jacobian())
    automatic_mapping = np.asarray(jax.jacfwd(adapter.likelihood_parameters)(joint))
    np.testing.assert_array_equal(automatic_mapping, analytic_mapping)

    native = adapter.likelihood_parameters(joint)
    native_jacobian = np.asarray(adapter.likelihood.jacobian(native))
    adapter_jacobian = np.asarray(adapter.jacobian(joint))
    np.testing.assert_allclose(
        adapter_jacobian,
        native_jacobian @ analytic_mapping,
        rtol=5.0e-12,
        atol=5.0e-9,
    )
    assert adapter_jacobian.shape == (428, 63)
    np.testing.assert_array_equal(
        adapter_jacobian[:, np.asarray(adapter.unobserved_physical_indices)], 0.0
    )

    jit_residual = jax.jit(adapter.residual)(joint)
    np.testing.assert_allclose(
        np.asarray(jit_residual),
        np.asarray(adapter.residual(joint)),
        rtol=0.0,
        atol=2.0e-10,
    )


def test_rank_audit_preserves_identifiable_condenser_directions_and_nulls(adapter):
    diagnostics = adapter.local_rank_diagnostics()

    assert diagnostics.joint_rank == 41
    assert diagnostics.joint_nullity == 22
    assert diagnostics.outer_rank == 26
    assert diagnostics.profiled_shared_physical_rank == 15
    assert diagnostics.profiled_condenser_target_rank == 12
    assert len(diagnostics.shared_physical_parameter_names) == 16
    assert len(diagnostics.unobserved_physical_parameter_names) == 20
    assert diagnostics.joint_singular_values.shape == (43,)
    assert diagnostics.outer_singular_values.shape == (27,)
    assert diagnostics.profiled_shared_physical_singular_values.shape == (15,)
    assert diagnostics.profiled_condenser_target_singular_values.shape == (12,)
    assert set(diagnostics.unobserved_physical_parameter_names) == {
        "detector_z_m",
        "Obj-post.z_center_m",
        "Obj-post.log_length_m",
        "Obj-post.q_bias",
        "IL1.z_center_m",
        "IL1.log_length_m",
        "IL1.q_bias",
        "IL1.q_slope_per_dac",
        "IL2.z_center_m",
        "IL2.log_length_m",
        "IL2.q_bias",
        "IL2.q_slope_per_dac",
        "IL3.z_center_m",
        "IL3.log_length_m",
        "IL3.q_bias",
        "IL3.q_slope_per_dac",
        "PL1.z_center_m",
        "PL1.log_length_m",
        "PL1.q_bias",
        "PL1.q_slope_per_dac",
    }


def test_shared_latent_downstream_likelihood_uses_one_physical_vector(adapter):
    """Exercise the same concatenation seam used by the notebook global fit."""

    count = adapter.optical_parameter_count
    keys = tuple(f"downstream.{name}" for name in adapter.shared_column.parameter_names)
    identity = np.eye(count)
    layout = GlobalObservationLayout(
        n_dataset_blocks=1,
        block_sizes=(count,),
        selected_block_indices=(0,),
        all_observation_keys=keys,
        observation_keys=keys,
        values=np.asarray(adapter.shared_column.nominal_parameters),
        statistical_covariance=identity,
        statistical_cholesky=identity,
        held_out_observation_mask=np.zeros(count, dtype=bool),
        included_observation_mask=np.ones(count, dtype=bool),
    )

    def downstream_prediction(physical_parameters, _calibration_values):
        return (physical_parameters,)

    downstream = SharedLatentNuisanceLikelihood(
        layout,
        count,
        downstream_prediction,
    )

    def combined_residual(joint_parameters):
        physical, outer = adapter.split_joint_parameters(joint_parameters)
        downstream_residual = downstream.joint_residual_from_parts(
            physical,
            jnp.empty((0,)),
            jnp.empty((0,)),
        )
        return jnp.concatenate(
            (adapter.residual_from_parts(physical, outer), downstream_residual)
        )

    joint = jnp.asarray(adapter.joint_reference)
    jacobian = np.asarray(jax.jacfwd(combined_residual)(joint))
    assert jacobian.shape == (428 + count, count + 27)
    np.testing.assert_allclose(jacobian[-count:, :count], identity)
    np.testing.assert_array_equal(jacobian[-count:, count:], 0.0)
    assert adapter.joint_parameter_names.count("CL1.z_center_m") == 1


def test_joint_scales_bounds_and_shape_failures_are_explicit(adapter):
    optical_scales = np.linspace(0.01, 0.36, adapter.optical_parameter_count)
    joint_scales = adapter.joint_scales(optical_scales)
    np.testing.assert_array_equal(joint_scales[:36], optical_scales)
    np.testing.assert_array_equal(joint_scales[36:], adapter.outer_contract.scales)

    optical_lower = adapter.shared_column.nominal_parameters - 0.1
    optical_upper = adapter.shared_column.nominal_parameters + 0.1
    lower, upper = adapter.joint_bounds((optical_lower, optical_upper))
    np.testing.assert_array_equal(lower[:36], optical_lower)
    np.testing.assert_array_equal(upper[:36], optical_upper)
    np.testing.assert_array_equal(lower[36:], adapter.outer_contract.lower)
    np.testing.assert_array_equal(upper[36:], adapter.outer_contract.upper)

    with pytest.raises(ValueError, match="physical_parameters must have shape"):
        adapter.join_parameters(np.zeros(35))
    with pytest.raises(ValueError, match="outer_parameters must have shape"):
        adapter.join_parameters(np.zeros(36), np.zeros(26))
    with pytest.raises(ValueError, match="joint_parameters must have shape"):
        adapter.residual(np.zeros(62))
    with pytest.raises(ValueError, match="positive"):
        adapter.joint_scales(np.zeros(36))


def test_column_identity_and_custom_outer_contract_fail_closed(adapter):
    wrong_control_order = replace(
        adapter.shared_column,
        channel_names=tuple(reversed(adapter.shared_column.channel_names)),
    )
    with pytest.raises(ValueError, match="raw-control order differs"):
        AstigmaticFullColumnAdapter.from_likelihood(
            adapter.likelihood,
            wrong_control_order,
        )

    contract = default_astigmatic_outer_parameter_contract(adapter.likelihood)
    with pytest.raises(ValueError, match="cannot contain K16"):
        replace(
            contract,
            names=("nuisance.deflector.K16", *contract.names[1:]),
        )
