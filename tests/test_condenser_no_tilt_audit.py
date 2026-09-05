from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

from temgym_core.condenser_branch_b import (
    BranchBLikelihood,
    load_branch_b_experiment,
    make_synthetic_branch_b_observations,
)
from temgym_core.condenser_no_tilt_audit import (
    NoTiltRouteAdmissionError,
    audit_no_tilt_route,
    build_no_tilt_audit_matrix,
    require_practical_no_tilt_admission,
)


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).parents[1]
PAYLOAD = (
    ROOT
    / "examples"
    / "lens_inversion"
    / "neoarm_condenser_branch_b_design.json"
)


def test_no_tilt_api_is_exported_from_package_root():
    import temgym_core

    assert temgym_core.audit_no_tilt_route is audit_no_tilt_route
    assert temgym_core.build_no_tilt_audit_matrix is build_no_tilt_audit_matrix


@pytest.fixture(scope="module")
def likelihood() -> BranchBLikelihood:
    experiment = load_branch_b_experiment(PAYLOAD, repository_root=ROOT)
    return BranchBLikelihood(
        experiment,
        make_synthetic_branch_b_observations(experiment),
    )


def test_no_angle_scale_anchor_retains_the_axial_dilation_null(likelihood):
    matrix = build_no_tilt_audit_matrix(
        likelihood,
        angular_scale_mode="none",
    )
    audit = audit_no_tilt_route(likelihood, angular_scale_mode="none")

    assert matrix.matrix.shape == (217, 57)
    assert not any("deflector." in name for name in matrix.row_names)
    assert any(
        "astigmatic_illumination.relative_orientation" in name
        for name in matrix.row_names
    )
    assert audit.target_rank == 11
    assert not audit.structural_pass
    assert not audit.practical_pass
    assert audit.minimum_singular_value_ratio < 1.0e-12
    assert np.all(np.isinf(audit.scaled_standard_uncertainties))

    null = audit.null_family
    log_lengths = np.asarray(
        (
            null["CL1.log_length_m"],
            null["CL3.log_length_m"],
            null["C-mini.log_length_m"],
        )
    )
    np.testing.assert_allclose(log_lengths, log_lengths[0], rtol=1.0e-7)
    q_values = np.asarray(
        [value for name, value in null.items() if ".q_" in name]
    )
    assert np.max(np.abs(q_values)) < 1.0e-8
    with pytest.raises(NoTiltRouteAdmissionError, match="rank 11/12"):
        require_practical_no_tilt_admission(audit)


def test_unanchored_diffraction_scale_does_not_create_rank(likelihood):
    matrix = build_no_tilt_audit_matrix(
        likelihood,
        angular_scale_mode="unanchored",
    )
    audit = audit_no_tilt_route(
        likelihood,
        angular_scale_mode="unanchored",
    )

    assert matrix.matrix.shape == (249, 58)
    assert "nuisance.indexed_si_log_angular_scale" in matrix.column_names
    assert not any("prior.indexed_si" in name for name in matrix.row_names)
    assert audit.target_rank == 11
    assert not audit.structural_pass


def test_si_ratio_prior_alone_is_formally_ranked_but_not_practical(likelihood):
    audit = audit_no_tilt_route(
        likelihood,
        angular_scale_mode="si_ratio_prior",
        si_log_angular_scale_prior_sigma=0.002,
        rotation_sigma_rad=0.02,
    )

    assert audit.target_rank == 12
    assert audit.structural_pass
    assert not audit.practical_pass
    assert audit.minimum_singular_value_ratio == pytest.approx(
        6.184081716083459e-7, rel=2.0e-6
    )
    assert audit.maximum_scaled_standard_uncertainty == pytest.approx(
        29.428523026867687, rel=2.0e-6
    )
    with pytest.raises(NoTiltRouteAdmissionError, match="precision gate"):
        require_practical_no_tilt_admission(audit)


def test_fixed_scale_synthetic_upper_bound_still_misses_one_percent(likelihood):
    matrix = build_no_tilt_audit_matrix(
        likelihood,
        angular_scale_mode="fixed",
        profile_realspace_scale=False,
        rotation_sigma_rad=0.02,
    )
    audit = audit_no_tilt_route(
        likelihood,
        angular_scale_mode="fixed",
        profile_realspace_scale=False,
        rotation_sigma_rad=0.02,
    )

    assert matrix.matrix.shape == (249, 56)
    assert audit.target_rank == 12
    assert audit.maximum_scaled_standard_uncertainty == pytest.approx(
        0.027916282679431678, rel=2.0e-6
    )
    assert not audit.practical_pass


def test_strong_dual_scale_priors_are_labelled_conditional(likelihood):
    audit = audit_no_tilt_route(
        likelihood,
        angular_scale_mode="si_ratio_prior",
        si_log_angular_scale_prior_sigma=5.0e-4,
        realspace_log_scale_prior_sigma=5.0e-4,
        rotation_sigma_rad=1.0e-3,
    )

    assert audit.target_rank == 12
    assert audit.minimum_singular_value_ratio > 1.0e-3
    assert audit.maximum_scaled_standard_uncertainty == pytest.approx(
        0.00964063536468679, rel=2.0e-6
    )
    assert audit.practical_pass
    assert "local matched-model" in audit.scope
    require_practical_no_tilt_admission(audit)


def test_realspace_prior_cannot_be_attached_to_a_fixed_scale(likelihood):
    with pytest.raises(ValueError, match="requires profile_realspace_scale"):
        build_no_tilt_audit_matrix(
            likelihood,
            angular_scale_mode="fixed",
            profile_realspace_scale=False,
            realspace_log_scale_prior_sigma=0.001,
        )
