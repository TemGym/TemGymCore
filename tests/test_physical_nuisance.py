import jax
import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.microscope_inversion import (
    AcquisitionPlan,
    AcquisitionState,
    FrameObservationBlock,
    HeterogeneousInversionDataset,
)
from temgym_core.physical_nuisance import (
    BlockwiseIdentifiabilityDiagnostics,
    GaussianPriorFactor,
    GlobalObservationLayout,
    SharedLatentNuisanceLikelihood,
    SignedDiscrepancyModes,
    diagnose_blockwise_identifiability,
    diagnose_blockwise_nuisance_identifiability,
    diagnose_nuisance_identifiability,
    evaluate_held_out_shared_likelihood,
)


jax.config.update("jax_enable_x64", True)


def _dataset(
    count: int,
    *,
    variance: float = 1.0,
    held_out_last: bool = False,
) -> HeterogeneousInversionDataset:
    states = tuple(
        AcquisitionState(
            label=f"state_{index}",
            dac_values=(float(index),),
            role="held_out" if held_out_last and index == count - 1 else "nominal",
            frame_count=1,
        )
        for index in range(count)
    )
    plan = AcquisitionPlan(
        channel_names=("lens",),
        states=states,
        reference_dacs=(0.0,),
    )
    blocks = tuple(
        FrameObservationBlock(
            frame_id=10 + index,
            optical_state_index=index,
            values=np.asarray((0.0,)),
            covariance=np.asarray(((variance,),)),
            observable_names=("centroid_x_px",),
        )
        for index in range(count)
    )
    return HeterogeneousInversionDataset(plan, blocks)


def _constant_optical_callback(count: int):
    def predict(optical_parameters, _calibration_values):
        return tuple(jnp.asarray((optical_parameters[0],)) for _ in range(count))

    return predict


def test_gaussian_prior_factor_preserves_singular_covariance_without_jitter():
    covariance = np.asarray(((4.0, 0.0), (0.0, 0.0)))
    prior = GaussianPriorFactor.from_covariance(
        "detector_metric",
        ("log_scale", "rotation_rad"),
        (0.3, -0.2),
        covariance,
        provenance="traceable detector calibration",
    )

    assert prior.n_values == 2
    assert prior.n_latents == 1
    np.testing.assert_allclose(prior.covariance, covariance, atol=1e-14)
    np.testing.assert_allclose(prior.values(jnp.asarray((0.5,))), (1.3, -0.2))
    assert not prior.mean.flags.writeable
    assert not prior.factor.flags.writeable
    assert not prior.covariance.flags.writeable


def test_global_layout_preserves_block_name_order_and_excludes_heldout():
    dataset = _dataset(2, held_out_last=True)
    training = GlobalObservationLayout.from_dataset(dataset)
    complete = GlobalObservationLayout.from_dataset(
        dataset, include_held_out=True
    )

    assert training.observation_keys == ("frame_10.centroid_x_px",)
    assert complete.observation_keys == (
        "frame_10.centroid_x_px",
        "frame_11.centroid_x_px",
    )
    np.testing.assert_array_equal(training.held_out_observation_mask, (False, True))
    np.testing.assert_array_equal(training.included_observation_mask, (True, False))
    np.testing.assert_array_equal(complete.included_observation_mask, (True, True))

    def predict(optical_parameters, _calibration_values):
        return (
            jnp.asarray((optical_parameters[0],)),
            jnp.asarray((10.0 * optical_parameters[0],)),
        )

    training_likelihood = SharedLatentNuisanceLikelihood(training, 1, predict)
    complete_likelihood = SharedLatentNuisanceLikelihood(complete, 1, predict)
    assert training_likelihood.data_residual(jnp.asarray((2.0,))).shape == (1,)
    np.testing.assert_allclose(
        complete_likelihood.data_residual(jnp.asarray((2.0,))), (2.0, 20.0)
    )


def _training_and_two_held_out_dataset(
    held_out_values: tuple[float, float] = (3.0, 8.0),
) -> HeterogeneousInversionDataset:
    states = (
        AcquisitionState("train", (0.0,), "nominal", 1),
        AcquisitionState("held_a", (1.0,), "held_out", 1),
        AcquisitionState("held_b", (2.0,), "held_out", 1),
    )
    plan = AcquisitionPlan(
        channel_names=("lens",), states=states, reference_dacs=(0.0,)
    )
    blocks = (
        FrameObservationBlock(20, 0, (4.0,), ((1.0,),), ("x_px",)),
        FrameObservationBlock(
            21, 1, (held_out_values[0],), ((4.0,),), ("x_px",)
        ),
        FrameObservationBlock(
            22, 2, (held_out_values[1],), ((1.0,),), ("x_px",)
        ),
    )
    return HeterogeneousInversionDataset(plan, blocks)


def _held_out_test_likelihood(dataset: HeterogeneousInversionDataset):
    prior = GaussianPriorFactor(
        group_name="shared_scale",
        value_names=("offset",),
        mean=np.asarray((0.5,)),
        factor=np.asarray(((0.25,),)),
        latent_names=("offset_z",),
        provenance="independent calibration",
    )

    def predict(optical_parameters, calibration_values):
        base = optical_parameters[0] + calibration_values[0][0]
        return tuple(jnp.asarray((base + index,)) for index in range(3))

    return SharedLatentNuisanceLikelihood(
        GlobalObservationLayout.from_dataset(dataset),
        1,
        predict,
        calibration_priors=(prior,),
    )


def test_true_held_out_evaluation_uses_complement_and_fitted_calibration():
    dataset = _training_and_two_held_out_dataset()
    likelihood = _held_out_test_likelihood(dataset)
    # theta=3 and z=2 gives the fitted shared offset 0.5 + 0.25*2 = 1.
    fitted = likelihood.join_parameters((3.0,), (2.0,))
    report = evaluate_held_out_shared_likelihood(likelihood, dataset, fitted)

    assert report.training_block_indices == (0,)
    assert report.held_out_block_indices == (1, 2)
    assert tuple(block.optical_state_label for block in report.blocks) == (
        "held_a",
        "held_b",
    )
    np.testing.assert_allclose(report.blocks[0].predicted, (5.0,))
    np.testing.assert_allclose(report.blocks[1].predicted, (6.0,))
    np.testing.assert_allclose(report.whitened_residual, (1.0, -2.0))
    assert report.chi_square == pytest.approx(5.0)
    assert report.weighted_rms_sigma == pytest.approx(np.sqrt(2.5))
    assert report.maximum_absolute_whitened_residual == pytest.approx(2.0)
    assert report.calibration_group_names == ("shared_scale",)
    assert not report.whitened_residual.flags.writeable


def test_held_out_values_cannot_change_training_residual_but_change_evaluation():
    first_dataset = _training_and_two_held_out_dataset((3.0, 8.0))
    second_dataset = _training_and_two_held_out_dataset((103.0, -92.0))
    first = _held_out_test_likelihood(first_dataset)
    second = _held_out_test_likelihood(second_dataset)
    fitted = first.join_parameters((3.0,), (2.0,))

    np.testing.assert_array_equal(first.joint_residual(fitted), second.joint_residual(fitted))
    first_report = evaluate_held_out_shared_likelihood(first, first_dataset, fitted)
    second_report = evaluate_held_out_shared_likelihood(second, second_dataset, fitted)
    assert first_report.chi_square != second_report.chi_square


def test_held_out_evaluation_rejects_leaky_layout_and_unmapped_discrepancy():
    dataset = _training_and_two_held_out_dataset()
    training = _held_out_test_likelihood(dataset)
    fitted = training.join_parameters((3.0,), (2.0,))
    leaky = SharedLatentNuisanceLikelihood(
        GlobalObservationLayout.from_dataset(dataset, include_held_out=True),
        1,
        training.prediction_callback,
        calibration_priors=training.calibration_priors,
    )
    with pytest.raises(ValueError, match="exact training-only"):
        evaluate_held_out_shared_likelihood(
            leaky, dataset, leaky.join_parameters((3.0,), (2.0,))
        )

    discrepancy = SignedDiscrepancyModes.from_columns(
        training.layout.observation_keys,
        {"training_only_bias": np.asarray((0.1,))},
        provenance="training grid discrepancy",
    )
    unmapped = SharedLatentNuisanceLikelihood(
        training.layout,
        1,
        training.prediction_callback,
        calibration_priors=training.calibration_priors,
        discrepancy_modes=discrepancy,
    )
    with pytest.raises(ValueError, match="cannot be extrapolated"):
        evaluate_held_out_shared_likelihood(
            unmapped, dataset, unmapped.join_parameters((3.0,), (2.0,), (0.0,))
        )


def test_shared_signed_systematic_does_not_average_down_like_detector_noise():
    systematic_sigma = 0.7
    variances = []
    for count in (1, 64):
        layout = GlobalObservationLayout.from_dataset(_dataset(count))
        discrepancy = SignedDiscrepancyModes.from_columns(
            layout.observation_keys,
            {"shared_grid_bias": np.full(count, systematic_sigma)},
            provenance="signed Gaussian-minus-grid observable residual",
        )
        likelihood = SharedLatentNuisanceLikelihood(
            layout,
            1,
            _constant_optical_callback(count),
            discrepancy_modes=discrepancy,
        )
        diagnostics = diagnose_nuisance_identifiability(
            likelihood, np.asarray((0.0,)), rank_rtol=1e-10
        )
        variances.append(
            diagnostics.prior_assisted_marginal_optical_identifiable_covariance_physical[
                0, 0
            ]
        )
        assert diagnostics.data_only_joint.rank == 1
        assert diagnostics.data_only_joint.nullity == 1

    assert variances[0] == pytest.approx(1.0 + systematic_sigma**2)
    assert variances[1] == pytest.approx(1.0 / 64.0 + systematic_sigma**2)
    assert variances[1] > 0.99 * systematic_sigma**2


def test_data_only_scale_ambiguity_is_not_hidden_by_calibration_prior():
    count = 8
    layout = GlobalObservationLayout.from_dataset(_dataset(count))
    calibration_sigma = 2.0
    prior = GaussianPriorFactor(
        group_name="specimen_scale",
        value_names=("scale_offset",),
        mean=np.asarray((0.0,)),
        factor=np.asarray(((calibration_sigma,),)),
        latent_names=("scale_standard_normal",),
        provenance="specimen certificate",
    )

    def predict(optical_parameters, calibration_values):
        scale_offset = calibration_values[0][0]
        value = optical_parameters[0] + scale_offset
        return tuple(jnp.asarray((value,)) for _ in range(count))

    likelihood = SharedLatentNuisanceLikelihood(
        layout, 1, predict, calibration_priors=(prior,)
    )
    diagnostics = diagnose_nuisance_identifiability(
        likelihood, np.asarray((0.0,)), rank_rtol=1e-10
    )

    assert diagnostics.conditional_optical.rank == 1
    assert diagnostics.data_only_joint.rank == 1
    assert diagnostics.data_only_joint.nullity == 1
    assert diagnostics.data_profiled_optical.rank == 0
    assert diagnostics.prior_assisted_joint.rank == 2
    assert diagnostics.prior_assisted_marginal_optical.rank == 1
    assert (
        diagnostics.prior_assisted_marginal_optical_identifiable_covariance_physical[
            0, 0
        ]
        == pytest.approx(1.0 / count + calibration_sigma**2)
    )


def test_joint_residual_has_data_then_calibration_then_discrepancy_and_jits():
    layout = GlobalObservationLayout.from_dataset(_dataset(2, variance=4.0))
    prior = GaussianPriorFactor(
        group_name="calibration",
        value_names=("offset",),
        mean=np.asarray((0.5,)),
        factor=np.asarray(((0.25,),)),
        latent_names=("offset_z",),
        provenance="test calibration",
    )
    discrepancy = SignedDiscrepancyModes.from_columns(
        layout.observation_keys,
        {"signed_reference_difference": np.asarray((0.2, -0.1))},
        provenance="independent reference",
    )

    def predict(optical_parameters, calibration_values):
        offset = calibration_values[0][0]
        return (
            jnp.asarray((optical_parameters[0] + offset,)),
            jnp.asarray((optical_parameters[0] - offset,)),
        )

    likelihood = SharedLatentNuisanceLikelihood(
        layout,
        1,
        predict,
        calibration_priors=(prior,),
        discrepancy_modes=discrepancy,
    )
    joint = likelihood.join_parameters(
        jnp.asarray((1.0,)),
        jnp.asarray((0.4,)),
        jnp.asarray((-0.3,)),
    )
    expected_prediction = np.asarray((1.54, 0.43))
    expected = np.concatenate((expected_prediction / 2.0, (0.4,), (-0.3,)))

    compiled = jax.jit(likelihood.joint_residual)
    residual = np.asarray(compiled(joint))
    jacobian = np.asarray(jax.jacfwd(likelihood.joint_residual)(joint))
    np.testing.assert_allclose(residual, expected, rtol=1e-12, atol=1e-12)
    assert jacobian.shape == (4, 3)
    assert np.all(np.isfinite(jacobian))


def test_discrepancy_requires_exact_global_observation_alignment():
    layout = GlobalObservationLayout.from_dataset(_dataset(2))
    discrepancy = SignedDiscrepancyModes.from_columns(
        tuple(reversed(layout.observation_keys)),
        {"mode": np.ones(2)},
        provenance="misordered test mode",
    )
    with pytest.raises(ValueError, match="exactly match"):
        SharedLatentNuisanceLikelihood(
            layout,
            1,
            _constant_optical_callback(2),
            discrepancy_modes=discrepancy,
        )


def test_blockwise_equilibration_prevents_one_large_block_masking_rank():
    upstream = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )
    )
    downstream = np.asarray(((0.0, 0.0, 1.0e12),))

    diagnostics = diagnose_blockwise_identifiability(
        {"upstream": upstream, "downstream": downstream},
        optical_parameter_count=3,
        rank_rtol=1.0e-7,
    )

    assert isinstance(diagnostics, BlockwiseIdentifiabilityDiagnostics)
    assert diagnostics.block_names == ("upstream", "downstream")
    assert diagnostics.report_for_block("upstream").joint.rank == 2
    assert diagnostics.report_for_block("downstream").joint.rank == 1
    # The ordinary combined relative threshold is set by 1e12 and hides the
    # two independent upstream columns.
    assert diagnostics.combined.joint.rank == 1
    assert diagnostics.combined.column_equilibrated_joint.rank == 3
    np.testing.assert_allclose(
        diagnostics.combined.joint_column_norms,
        (1.0, 1.0, 1.0e12),
    )
    assert not diagnostics.scaled_combined_jacobian.flags.writeable
    assert not diagnostics.combined.joint_column_norms.flags.writeable
    with pytest.raises(KeyError):
        diagnostics.report_for_block("missing")


def test_blockwise_profile_keeps_a_nuisance_shared_across_blocks():
    # Separately, each block can trade optical parameter zero against the
    # nuisance.  Their opposite nuisance signs make both optical directions
    # identifiable when the shared coordinate is profiled jointly.
    first = np.asarray(
        (
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 0.0),
        )
    )
    second = np.asarray(((1.0, 0.0, -1.0),))

    diagnostics = diagnose_blockwise_identifiability(
        (("positive", first), ("negative", second)),
        optical_parameter_count=2,
        rank_rtol=1.0e-10,
    )

    assert diagnostics.report_for_block("positive").profiled_optical.rank == 1
    assert diagnostics.report_for_block("negative").profiled_optical.rank == 0
    assert diagnostics.combined.joint.rank == 3
    assert diagnostics.combined.profiled_optical.rank == 2
    assert diagnostics.combined.column_equilibrated_profiled_optical.rank == 2
    np.testing.assert_allclose(
        diagnostics.combined.profiled_optical_precision,
        np.diag((2.0, 1.0)),
        rtol=1e-12,
        atol=1e-12,
    )


def test_likelihood_blockwise_wrapper_preserves_selected_block_order_and_scales():
    layout = GlobalObservationLayout.from_dataset(_dataset(2))
    prior = GaussianPriorFactor(
        group_name="shared_offset",
        value_names=("offset",),
        mean=np.asarray((0.0,)),
        factor=np.asarray(((1.0,),)),
        latent_names=("offset_z",),
        provenance="test calibration",
    )

    def predict(optical_parameters, calibration_values):
        offset = calibration_values[0][0]
        return (
            jnp.asarray((optical_parameters[0] + offset,)),
            jnp.asarray((optical_parameters[0] - offset,)),
        )

    likelihood = SharedLatentNuisanceLikelihood(
        layout,
        1,
        predict,
        calibration_priors=(prior,),
    )
    diagnostics = diagnose_blockwise_nuisance_identifiability(
        likelihood,
        np.asarray((0.0,)),
        optical_parameter_scales=np.asarray((7.0,)),
        block_names=("plus", "minus"),
        rank_rtol=1e-10,
    )

    assert diagnostics.block_names == ("plus", "minus")
    np.testing.assert_allclose(diagnostics.parameter_scales, (7.0, 1.0))
    assert diagnostics.report_for_block("plus").profiled_optical.rank == 0
    assert diagnostics.report_for_block("minus").profiled_optical.rank == 0
    assert diagnostics.combined.profiled_optical.rank == 1
    assert diagnostics.combined.column_equilibrated_joint.rank == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"optical_parameter_count": 0}, "optical_parameter_count"),
        (
            {"optical_parameter_count": 1, "parameter_scales": (1.0, -1.0)},
            "parameter_scales",
        ),
        (
            {"optical_parameter_count": 1, "column_norm_atol": -1.0},
            "column_norm_atol",
        ),
    ),
)
def test_blockwise_identifiability_rejects_invalid_contract(kwargs, message):
    with pytest.raises(ValueError, match=message):
        diagnose_blockwise_identifiability(
            {"block": np.eye(2)},
            **kwargs,
        )
