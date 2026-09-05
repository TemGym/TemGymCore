from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from temgym_core.condenser_astigmatic_design import (
    AstigmaticDesignAdmissionError,
    AstigmaticDesignProvenanceError,
    build_astigmatic_design_matrix,
    load_astigmatic_condenser_design,
    recompute_astigmatic_design_audit,
    require_astigmatic_hardware_readiness,
)
from temgym_core.condenser_branch_b import NORMALIZED_CONTROL_WARNING


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).parents[1]
MANIFEST = (
    ROOT
    / "examples"
    / "lens_inversion"
    / "neoarm_condenser_astigmatic_design.json"
)


@pytest.fixture(scope="module")
def design():
    return load_astigmatic_condenser_design(
        MANIFEST,
        repository_root=ROOT,
    )


@pytest.fixture(scope="module")
def matrix_and_summaries(design):
    return build_astigmatic_design_matrix(design)


@pytest.fixture(scope="module")
def audit(design):
    return recompute_astigmatic_design_audit(design)


def test_candidate_api_is_exported_from_package_root():
    import temgym_core

    assert (
        temgym_core.load_astigmatic_condenser_design
        is load_astigmatic_condenser_design
    )
    assert (
        temgym_core.recompute_astigmatic_design_audit
        is recompute_astigmatic_design_audit
    )


def test_manifest_freezes_strict_candidate_and_frame_accounting(design):
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert design.coordinate_status == NORMALIZED_CONTROL_WARNING
    assert "conditional" in design.scope
    assert "selected_q_range_per_m" not in document["sample_block"]
    assert (
        "selected_integrated_larmor_q_range_dimensionless"
        in document["sample_block"]
    )
    assert "legacy naming" in document["provenance"]["candidate_search"][
        "integrated_larmor_naming"
    ]
    assert design.controls.shape == (13, 7)
    assert design.repeat_counts.tolist() == [2, 2, 1, 1, 3, 8, 2, 1, 2, 2, 19, 3, 2]
    assert int(np.sum(design.repeat_counts)) == 48
    assert np.unique(design.controls[:, :3], axis=0).shape[0] == 13
    np.testing.assert_array_equal(design.controls[:, 3:], 0.0)

    assert design.conservative_raw_frame_count == 232
    assert design.joint_covariance_raw_frame_count == 184
    assert design.frame_accounting["edge_frames"] == 88
    assert design.frame_accounting["conservative_frames_per_sample_acquisition"] == 3
    assert design.frame_accounting["joint_dense_covariance_frames_per_sample_acquisition"] == 2
    assert "complete joint bootstrap covariance" in design.frame_accounting["rule"]


def test_matrix_has_no_calibrated_deflector_or_commanded_tilt_rows(
    matrix_and_summaries,
):
    matrix, summaries = matrix_and_summaries

    assert matrix.matrix.shape == (428, 43)
    assert summaries.shape == (48, 10)
    assert not any("deflector" in name.lower() for name in matrix.row_names)
    assert not any("K16" in name for name in matrix.column_names)
    assert any("correlation_theta_x_theta_y" in name for name in matrix.row_names)
    assert any("astigmatic_tag_orientation" in name for name in matrix.row_names)
    assert any("shared_astigmatic_tag_angle_offset" in name for name in matrix.column_names)


def test_recomputed_local_audit_is_ranked_but_has_almost_no_margin(
    audit,
    matrix_and_summaries,
):
    assert audit.target_rank == 12
    assert audit.nuisance_rank == 29
    assert audit.minimum_singular_value_ratio == pytest.approx(
        0.0013263222440776795,
        rel=3.0e-8,
    )
    assert audit.maximum_scaled_standard_uncertainty == pytest.approx(
        0.009971049145141537,
        rel=3.0e-8,
    )
    assert audit.conditional_precision_pass
    assert audit.maximum_scaled_standard_uncertainty > 0.0099

    _matrix, summaries = matrix_and_summaries
    all_expanded_equivalent_rms = np.sqrt(
        0.5 * (np.exp(2.0 * summaries[:, 0]) + np.exp(2.0 * summaries[:, 2]))
    )
    assert all_expanded_equivalent_rms.shape == (48,)
    assert np.all(all_expanded_equivalent_rms >= 3.0e-8)
    assert np.all(all_expanded_equivalent_rms <= 2.0e-7)
    np.testing.assert_allclose(
        audit.sample_equivalent_rms_range_m,
        (np.min(all_expanded_equivalent_rms), np.max(all_expanded_equivalent_rms)),
        rtol=3.0e-8,
    )
    assert np.all(audit.sample_equivalent_rms_range_m >= 3.0e-8)
    assert np.all(audit.sample_equivalent_rms_range_m <= 2.0e-7)


def test_nominal_long_camera_geometry_is_only_a_prediction(audit, design):
    assert design.planned_camera_length_m == pytest.approx(0.4)
    assert audit.si_111_radius_px == pytest.approx(58.16915496486905)
    np.testing.assert_allclose(
        audit.angular_eigen_rms_range_px,
        ((1.20646629, 1.80919153), (2.20468574, 3.30261849)),
        rtol=3.0e-8,
    )
    assert audit.nominal_detector_geometry_pass

    nominal_camera_scale = 0.1 / design.planned_camera_length_m
    nominal_minor_axis_px = (
        nominal_camera_scale * audit.angular_eigen_rms_range_px[0, 0]
    )
    assert nominal_minor_axis_px < design.detector_mtf_sigma_px

    assert not design.angular_observability_verified
    assert not design.nonlinear_recovery_verified
    assert not design.truth_envelope_verified
    assert not design.held_out_verified
    assert not design.hardware_ready
    assert not audit.hardware_ready
    assert "native_diffraction_camera_length" in design.hardware_gates
    assert all(
        not bool(gate["verified_on_instrument"])
        for gate in design.hardware_gates.values()
    )


def test_astigmatic_tag_plane_is_an_explicit_hardware_gate(design):
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    contract = document["sample_block"]["astigmatic_tag_contract"]
    gate = design.hardware_gates["astigmatic_tag_plane_and_repeatability"]

    assert "upstream of CL1" in contract["required_plane"]
    assert contract["absolute_command_calibration_required"] is False
    assert "unknown common orientation offset" in contract["fitted_shared_quantities"]
    assert "upstream of CL1" in gate["requirement"]
    assert gate["verified_on_instrument"] is False


def test_outside_paper_routes_remain_fail_closed_reconnaissance():
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    reconnaissance = document["outside_paper_reconnaissance"]
    aperture = reconnaissance["centered_c_aperture_images"]
    phase = reconnaissance["signed_phase_diversity"]

    assert "fail_closed" in aperture["status"]
    assert aperture["raw_CL1_CL3_target_rank"] == 8
    assert aperture["profiled_CL1_CL3_target_rank"] == 7
    assert aperture["maximum_absolute_Cmini_jacobian_entry"] == 0.0
    assert aperture["combined_rank_with_selected_sample_block"] == 12
    assert aperture[
        "combined_maximum_scaled_sigma_with_0p1_percent_scale_priors"
    ] > 0.025

    assert "fail_closed" in phase["status"]
    assert phase["plus_minus_100_and_200_um_summary_sigma_0p002"] > 0.01
    assert phase["best_tested_without_indexed_angular_ellipse"] > 0.014


def test_hardware_admission_fails_closed(design, audit):
    with pytest.raises(
        AstigmaticDesignAdmissionError,
        match="not hardware ready; unverified gates",
    ):
        require_astigmatic_hardware_readiness(design, audit)


def test_manifest_checksum_is_enforced(tmp_path):
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    document["hardware_admission"]["ready"] = True
    changed = tmp_path / MANIFEST.name
    changed.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(
        AstigmaticDesignProvenanceError,
        match="manifest checksum changed",
    ):
        load_astigmatic_condenser_design(changed, repository_root=ROOT)


def test_parent_payload_provenance_is_enforced(tmp_path):
    relative_parent = Path(
        "examples/lens_inversion/neoarm_condenser_branch_b_design.json"
    )
    changed_parent = tmp_path / relative_parent
    changed_parent.parent.mkdir(parents=True)
    parent_document = json.loads(
        (ROOT / relative_parent).read_text(encoding="utf-8")
    )
    parent_document["model_control_contract"]["affine_q_per_m"]["bias"][0] += 1.0e-9
    changed_parent.write_text(json.dumps(parent_document), encoding="utf-8")

    with pytest.raises(
        AstigmaticDesignProvenanceError,
        match="Branch-B design payload changed",
    ):
        load_astigmatic_condenser_design(
            MANIFEST,
            repository_root=tmp_path,
        )
