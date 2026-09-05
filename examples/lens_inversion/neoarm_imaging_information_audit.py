"""Audit image-derived condenser information without an independent rotation tag.

The archived design grants a free ten-coordinate entrance covariance but also
models its measured ellipse angle as ``offset - sum(q)``.  That angle law is
not valid for a general covariance.  Moreover, angle extracted from the same
image is a deterministic function of the three spatial covariance summaries.

This audit removes that extra datum.  Optional signed diversity means an
independently calibrated effective sample-side B/A, *not* a claimed stage
movement, accessible stigmator, or safe NEOARM control.  The information
matrices still grant ideal Gaussian covariance summaries, so their local
precision is an optimistic design diagnostic, not an image recovery proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from temgym_core.condenser_astigmatic_likelihood import (
    make_astigmatic_condenser_likelihood,
)
from temgym_core.constants import energy2wavelength
from temgym_core.critical_line_calibration import (
    aperture_edge_scan_from_moments,
    hard_edge_matrix_to_plane,
)


def ellipse_angle(summary):
    """Major-axis orientation from (log RMS x, log RMS y, correlation)."""
    sx, sy = jnp.exp(jnp.asarray(summary)[:2])
    rho = jnp.asarray(summary)[2]
    return 0.5 * jnp.arctan2(2.0 * rho * sx * sy, sx * sx - sy * sy)


def spatial_summary(covariance):
    """Registered Gaussian ellipse, without a redundant angle coordinate."""
    covariance = jnp.asarray(covariance)
    sigma = jnp.sqrt(jnp.diag(covariance))
    return jnp.array((jnp.log(sigma[0]), jnp.log(sigma[1]),
                      covariance[0, 1] / jnp.prod(sigma)))


def profile_targets(matrix, target_count=12):
    """Profile nuisance directions before reporting local target information.

    Singular values use the archived scaled optical coordinates.  A reported
    scaled uncertainty is not an unqualified relative error for a lens centre.
    Rank uses the same tolerances as the archived design for comparability.
    """
    matrix = np.asarray(matrix, dtype=float)
    target, nuisance = matrix[:, :target_count], matrix[:, target_count:]
    norms = np.linalg.norm(nuisance, axis=0)
    active = norms > np.finfo(float).tiny
    if np.any(active):
        left, singular, _ = np.linalg.svd(
            nuisance[:, active] / norms[active], full_matrices=False
        )
        nuisance_rank = int(np.sum(singular > 1e-10 * singular[0]))
        target = target - left[:, :nuisance_rank] @ (
            left[:, :nuisance_rank].T @ target
        )
    else:
        nuisance_rank = 0
    _, singular, right = np.linalg.svd(target, full_matrices=False)
    rank = int(np.sum(singular > 1e-7 * singular[0])) if singular[0] else 0
    uncertainties = None
    if rank == target_count:
        uncertainties = np.sqrt(np.sum(right.T**2 / singular**2, axis=1))
    return {
        "rows": int(matrix.shape[0]),
        "nuisance_rank": nuisance_rank,
        "target_rank": rank,
        "target_singular_values": singular.tolist(),
        "weakest_target_direction": right[-1].tolist(),
        "scaled_standard_uncertainties": (
            None if uncertainties is None else uncertainties.tolist()
        ),
        "maximum_scaled_standard_uncertainty": (
            None if uncertainties is None else float(np.max(uncertainties))
        ),
    }


def diversity_matrix(likelihood, diversities_m, *, summary_sigma=0.002):
    """Jacobian of genuine transported spatial ellipses, with a shared B/A scale.

    The last added coordinate is a log calibration error in effective B/A.
    Scale priors are separate rows and are inserted by ``run_audit``.
    """
    branch = likelihood.branch_experiment
    controls = jnp.asarray(likelihood.design.expanded_controls)
    diversities = jnp.asarray(diversities_m)
    xy = jnp.asarray((0, 2))

    def summaries(values):
        physical = jnp.asarray(branch.column.nominal_parameters)
        physical = physical.at[jnp.asarray(branch.target_parameter_indices)].add(
            values[:12] * jnp.asarray(branch.target_parameter_scales)
        )
        physical = physical.at[jnp.asarray(branch.upstream_parameter_indices)].add(
            values[22:26] * jnp.asarray(branch.upstream_parameter_scales)
        )
        entrance = branch.entrance_covariance_chart.covariance(values[12:22])

        def for_state(dacs):
            matrix = hard_edge_matrix_to_plane(
                branch.column, physical, dacs, branch.sample_z_m
            )
            covariance = matrix @ entrance @ matrix.T

            def at_diversity(diversity):
                drift = jnp.eye(4).at[0, 1].set(diversity * jnp.exp(values[43]))
                drift = drift.at[2, 3].set(diversity * jnp.exp(values[43]))
                projected = drift @ covariance @ drift.T
                result = spatial_summary(projected[jnp.ix_(xy, xy)])
                return result.at[:2].add(values[40])

            return jax.vmap(at_diversity)(diversities)

        return jax.vmap(for_state)(controls).reshape(-1) / summary_sigma

    return np.asarray(jax.jacfwd(summaries)(jnp.zeros(44)))


def centered_throughput_matrix(likelihood, radii_m=(10e-6, 20e-6, 30e-6)):
    """Optimistic centered-aperture counts and independent dose references.

    Uses four aperture-reconnaissance states and thirteen sample states. Each
    radius has its own calibration nuisance. The final coordinate translates
    the aperture plane by one mm. Centering is assumed verified, and ALL
    transmitted electrons are assumed counted. Neither assumption is proven
    by a finite field-of-view image. No aperture-plane shape datum is granted.
    """
    branch = likelihood.branch_experiment
    controls = jnp.asarray(np.unique(np.concatenate((
        branch.edge_controls, likelihood.design.controls
    )), axis=0))
    radii = jnp.asarray(radii_m)
    xy = jnp.asarray((0, 2))
    dose = branch.edge_incident_electrons_per_frame
    n_parameters = 44 + len(radii_m) + 1

    def log_transmission(values):
        physical = jnp.asarray(branch.column.nominal_parameters)
        physical = physical.at[jnp.asarray(branch.target_parameter_indices)].add(
            values[:12] * jnp.asarray(branch.target_parameter_scales)
        )
        physical = physical.at[jnp.asarray(branch.upstream_parameter_indices)].add(
            values[22:26] * jnp.asarray(branch.upstream_parameter_scales)
        )
        entrance = branch.entrance_covariance_chart.covariance(values[12:22])

        def at_state(dacs):
            matrix = hard_edge_matrix_to_plane(
                branch.column, physical, dacs, branch.aperture_z_m
            )
            drift = jnp.eye(4).at[0, 1].set(0.001 * values[-1])
            drift = drift.at[2, 3].set(0.001 * values[-1])
            matrix = drift @ matrix
            covariance = matrix @ entrance @ matrix.T
            covariance = covariance[jnp.ix_(xy, xy)]
            prediction = aperture_edge_scan_from_moments(
                jnp.zeros((len(radii_m), 2)), covariance,
                aperture_centre_xy_m=jnp.zeros(2),
                aperture_radius_m=radii * jnp.exp(values[44:-1]),
                incident_dose_electrons=dose,
                radial_order=48, azimuthal_order=128,
            )
            return jnp.log(prediction.expected_transmitted_electrons[:, 0] / dose)

        return jax.vmap(at_state)(controls).reshape(-1)

    nominal = jnp.zeros(n_parameters)
    transmission = np.exp(np.asarray(log_transmission(nominal)))
    jacobian = np.asarray(jax.jacfwd(log_transmission)(nominal))
    # Profiling an independent unknown incident dose against a Poisson
    # aperture-out reference gives this Fisher log-ratio variance. It is not
    # normalization that discards flux: both counts contribute their noise.
    sigma = np.sqrt(1.0 / (dose * transmission) + 1.0 / dose)
    return jacobian / sigma[:, None], {
        "radii_m": list(radii_m), "unique_lens_states": len(controls),
        "inserted_frames": len(controls) * len(radii_m),
        "independent_dose_reference_frames": len(controls) * len(radii_m),
        "transmission_range": [float(transmission.min()), float(transmission.max())],
        "incident_electrons_per_frame": float(dose),
        "assumptions": "known centered beam, total transmitted counts captured, independent 0.5 percent radius priors; approximate branch geometry only",
    }


def run_audit(*, include_diversity=True, include_centered=False):
    jax.config.update("jax_enable_x64", True)
    likelihood = make_astigmatic_condenser_likelihood()
    parameters = np.asarray(likelihood.nominal_parameters)
    jacobian = np.asarray(likelihood.jacobian(parameters))
    prediction = np.asarray(likelihood.predict(parameters).sample_observables)
    base = list(range(92))
    spatial = [92 + 7 * i + k for i in range(48) for k in (0, 1, 2)]
    angular = [92 + 7 * i + k for i in range(48) for k in (3, 4, 5)]
    tag = [92 + 7 * i + 6 for i in range(48)]
    image_rows = [row for row in base if row != 91] + spatial
    cases = {
        "legacy_independent_tag": profile_targets(jacobian),
        "spatial_and_angular_no_independent_tag": profile_targets(
            jacobian[base + spatial + angular]
        ),
        "spatial_only_no_independent_tag": profile_targets(jacobian[image_rows]),
        "spatial_and_angular_without_translated_aperture": profile_targets(
            jacobian[[90, 91] + spatial + angular]
        ),
        "spatial_angular_translated_aperture_unanchored_stage_scale": profile_targets(
            jacobian[[row for row in base if row != 89] + spatial + angular]
        ),
    }
    # A genuinely separate exposure can improve noise precision, but its angle
    # derivative must come from the same transported covariance as the image.
    corrected = jacobian.copy()
    angle_gradients = np.asarray(jax.vmap(jax.grad(ellipse_angle))(prediction[:, :3]))
    corrected[tag] = np.einsum(
        "ni,nij->nj", angle_gradients, jacobian[spatial].reshape(48, 3, 43)
    ) * likelihood.design.covariance_summary_sigma / likelihood.design.tag_orientation_sigma_rad
    corrected[tag, 42] = 1.0 / likelihood.design.tag_orientation_sigma_rad
    cases["spatial_plus_separate_physical_ellipse_angle"] = profile_targets(
        corrected[image_rows + tag]
    )
    cases["spatial_angular_plus_separate_physical_ellipse_angle"] = profile_targets(
        corrected[base + spatial + angular + tag]
    )
    if include_diversity:
        # The existing in-focus observations remain, while each additional
        # diversity grants a separate image per repeated condenser acquisition.
        for diversities in ((-100e-6, 100e-6), (-200e-6, -100e-6, 100e-6, 200e-6)):
            retained = np.pad(jacobian[image_rows], ((0, 0), (0, 1)))
            retained[90] *= likelihood.design.realspace_log_scale_prior_sigma / 0.002
            scale_prior = np.zeros((1, 44))
            scale_prior[0, 43] = 1.0 / 0.002
            diversity = diversity_matrix(likelihood, diversities)
            matrix = np.vstack((retained, diversity, scale_prior))
            key = "imaging_diversity_" + "_".join(f"{value * 1e6:+g}um" for value in diversities)
            cases[key] = profile_targets(matrix)
            cases[key]["additional_image_count"] = 48 * len(diversities)
            cases[key]["shared_effective_B_over_A_log_scale_sigma"] = 0.002
            cases[key]["shared_realspace_log_scale_sigma"] = 0.002
            no_aperture = np.vstack((
                retained[90:], diversity, scale_prior
            ))
            cases[key + "_without_translated_aperture"] = profile_targets(no_aperture)
            cases[key + "_unanchored_stage_scale"] = profile_targets(
                np.delete(matrix, 89, axis=0)
            )
    centered_contract = None
    if include_centered:
        centered, centered_contract = centered_throughput_matrix(likelihood)
        column_count = centered.shape[1]
        radius_priors = np.zeros((3, column_count))
        radius_priors[np.arange(3), 44 + np.arange(3)] = 1.0 / 0.005
        modes = {
            "spatial_angular": np.pad(jacobian[[90, 91] + spatial + angular],
                                       ((0, 0), (0, column_count - 43))),
        }
        if include_diversity:
            # The same +/-100 um contract as above, excluding translated-edge
            # measurements.  The effective B/A scale is independently measured.
            retained = np.pad(jacobian[[90] + spatial], ((0, 0), (0, column_count - 43)))
            retained[0] *= likelihood.design.realspace_log_scale_prior_sigma / 0.002
            diversity = np.pad(diversity_matrix(likelihood, (-100e-6, 100e-6)),
                               ((0, 0), (0, column_count - 44)))
            prior = np.zeros((1, column_count)); prior[0, 43] = 1.0 / 0.002
            modes["imaging_diversity"] = np.vstack((retained, diversity, prior))
        for mode, data in modes.items():
            for plane_sigma_mm in (0.0, 1.0, 5.0, None):
                matrix = np.vstack((data, centered, radius_priors))
                if plane_sigma_mm == 0.0:
                    matrix = matrix[:, :-1]
                elif plane_sigma_mm is not None:
                    prior = np.zeros((1, column_count)); prior[0, -1] = 1.0 / plane_sigma_mm
                    matrix = np.vstack((matrix, prior))
                label = "free" if plane_sigma_mm is None else f"{plane_sigma_mm:g}mm"
                cases[f"centered_total_counts_{mode}_aperture_plane_{label}"] = profile_targets(matrix)
            one_radius = np.vstack((data, centered[::3], radius_priors))[:, :-1]
            cases[f"centered_10um_radius_only_{mode}_fixed_aperture_plane"] = profile_targets(one_radius)
    mismatches = []
    for amplitude in (0.0, 0.001, 0.01, 0.03):
        changed = parameters.copy()
        changed[12:22] = amplitude * np.sin(np.arange(10) + 1)
        summaries = np.asarray(likelihood.predict(changed).sample_observables)
        angle = np.asarray(jax.vmap(ellipse_angle)(summaries[:, :3]))
        difference = angle - summaries[:, 6]
        # Even generously allowing major/minor-axis swaps (pi/2), a generic
        # covariance does not obey state-independent offset minus sum(q).
        delta = 0.25 * np.arctan2(
            np.sin(4 * (difference - difference[0])),
            np.cos(4 * (difference - difference[0])),
        )
        mismatches.append({"entrance_eta_amplitude": amplitude,
                           "angle_residual_rms_rad": float(np.sqrt(np.mean(delta**2))),
                           "angle_residual_max_rad": float(np.max(np.abs(delta)))})
    covariance = likelihood.branch_experiment.entrance_reference_covariance
    omega = np.kron(np.eye(2), np.array(((0, 1), (-1, 0))))
    emittances = np.sort(np.abs(np.linalg.eigvals(omega @ covariance)))[::2]
    return {
        "schema": "temgym.neoarm.imaging-information-audit/v1",
        "scope": "local optimistic covariance-summary information audit; not raw-image recovery or hardware admission",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "design_provenance": {
            "astigmatic_design_path": str(likelihood.design.path),
            "astigmatic_design_sha256": likelihood.design.sha256,
            "branch_b_design_path": str(likelihood.design.branch_b_payload_path),
            "branch_b_design_sha256": likelihood.design.branch_b_payload_sha256,
        },
        "audit_options": {
            "include_diversity": bool(include_diversity),
            "include_centered_throughput": bool(include_centered),
        },
        "astigmatic_design_sha256": likelihood.design.sha256,
        "same_image_ellipse_angle_is_redundant": True,
        "hardware_ready": False,
        "cases": cases,
        "tag_law_mismatch_under_released_entrance_covariance": mismatches,
        "entrance_symplectic_emittances_m_rad": emittances.tolist(),
        "quantum_minimum_emittance_m_rad": float(energy2wavelength(200000.0)) / (4 * np.pi),
        "centered_throughput_contract": centered_contract,
        "geometry_warning": "branch aperture is 75 mm upstream of sample; user measure_microscope_geometry notebook places it about 420.8 mm upstream, with correspondingly different lens positions. This audit does not transfer that geometry or prescribe hardware commands.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-diversity", action="store_true")
    parser.add_argument("--centered-throughput", action="store_true")
    parser.add_argument("--output", type=Path, help="Write JSON here; default prints JSON to stdout.")
    arguments = parser.parse_args(argv)
    report = run_audit(include_diversity=not arguments.skip_diversity,
                       include_centered=arguments.centered_throughput)
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if arguments.output is None:
        print(serialized, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized, encoding="utf-8")
        print(f"Saved information audit: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
