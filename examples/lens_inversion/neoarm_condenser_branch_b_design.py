"""Verify and independently re-audit the NEOARM Branch-B condenser design.

This file is the executable provenance companion to
``neoarm_condenser_branch_b_design.json``.  The normal invocation is a fast,
stdlib/NumPy-only integrity check::

    python examples/lens_inversion/neoarm_condenser_branch_b_design.py

Use ``--audit`` to rebuild the selected-design Fisher matrices with the
current x64 JAX hard-edge, aperture-integral, covariance-chart, and deflector
implementations.  That is intentionally slower than the integrity check, but
does not repeat the original 500,000-candidate stochastic search::

    python examples/lens_inversion/neoarm_condenser_branch_b_design.py \
        --audit design_248 --audit design_296

The coordinates in the JSON file are *normalized synthetic model controls*.
They are not certified NEOARM DAC commands and must not be sent to hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import tomllib
from typing import Any

import numpy as np


SCHEMA = "temgym.neoarm.condenser-branch-b-design/v1"
DEFAULT_PAYLOAD = Path(__file__).with_suffix(".json")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class DesignVerificationError(ValueError):
    """Raised when a committed design no longer matches its provenance."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _actual_geometry(document: dict[str, Any]) -> dict[str, Any]:
    aperture = document["apertures"]["C_aperture"]
    deflector = document["deflectors"]["CL12_deflector"]
    return {
        "CL12_deflector": {
            key: deflector[key]
            for key in (
                "Dc",
                "shift_balance_x",
                "shift_balance_y",
                "spacing_m",
                "tilt_balance_x",
                "tilt_balance_y",
                "turns",
                "z_m",
            )
        },
        "C_aperture": {
            "radii_m": aperture["radii_m"],
            "z_m": aperture["z_m"],
        },
        "C_mini": document["lenses"]["C_mini"]["z_m"],
        "sample": document["sample"]["z_m"],
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DesignVerificationError(message)


def _as_finite_array(value: Any, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    _require(
        array.shape == shape,
        f"{name} must have shape {shape}, got {array.shape}.",
    )
    _require(np.all(np.isfinite(array)), f"{name} contains non-finite values.")
    return array


def load_and_verify(
    payload_path: str | Path = DEFAULT_PAYLOAD,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Load the frozen payload and verify identity, schema, and safe controls."""

    started = time.perf_counter()
    root = Path(repository_root).resolve()
    payload_file = Path(payload_path).resolve()
    payload = json.loads(payload_file.read_text(encoding="utf-8"))
    _require(payload.get("schema") == SCHEMA, "Unsupported design schema.")
    _require(
        payload.get("coordinate_status")
        == "normalized synthetic model controls; not certified hardware DAC commands",
        "The payload must retain its normalized-coordinate hardware warning.",
    )
    _require(
        payload.get("scope")
        == "local matched-model Fisher design; not nonlinear or held-out certification",
        "The local-Fisher scope warning is missing or changed.",
    )

    provenance = payload["provenance"]
    config_binding = provenance["config"]
    config_path = root / config_binding["path"]
    _require(config_path.is_file(), f"Missing bound config {config_path}.")
    _require(
        _sha256(config_path) == config_binding["sha256"],
        "neoarm_inversion_config.toml hash differs from the design payload.",
    )

    geometry_binding = provenance["actual_geometry"]
    microscope_path = root / geometry_binding["path"]
    _require(microscope_path.is_file(), f"Missing geometry file {microscope_path}.")
    with microscope_path.open("rb") as stream:
        geometry = _actual_geometry(tomllib.load(stream))
    _require(
        _canonical_sha256(geometry) == geometry_binding["canonical_sha256"],
        "The actual C-aperture/deflector/C-mini/sample geometry changed.",
    )
    _require(
        geometry == geometry_binding["values"],
        "The canonical geometry values differ from the recorded values.",
    )

    for implementation in provenance["implementation_files"]:
        path = root / implementation["path"]
        _require(path.is_file(), f"Missing bound implementation {path}.")
        _require(
            _sha256(path) == implementation["sha256"],
            f"Implementation hash changed for {implementation['path']}.",
        )

    channels = tuple(payload["model_control_contract"]["channel_order"])
    _require(
        channels == ("CL1", "CL3", "C-mini", "IL1", "IL2", "IL3", "PL1"),
        "Unexpected normalized-control channel order.",
    )
    lower, upper = np.asarray(
        payload["model_control_contract"]["safe_bounds_CL1_CL3_Cmini"],
        dtype=float,
    )
    _require(lower.shape == upper.shape == (3,), "Safe bounds must be two 3-vectors.")
    q_contract = payload["model_control_contract"]["affine_q_per_m"]
    q_bias = _as_finite_array(q_contract["bias"], name="q bias", shape=(3,))
    q_slope = _as_finite_array(q_contract["slope"], name="q slope", shape=(3,))
    q_floor = float(payload["model_control_contract"]["candidate_q_floor_per_m"])
    expected_lower = np.maximum(-0.35, (q_floor - q_bias) / q_slope)
    np.testing.assert_allclose(lower, expected_lower, rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(upper, 0.35, rtol=0.0, atol=0.0)

    frames_per = payload["measurement_contract"]["raw_frames_per_state"]
    verified_designs: dict[str, dict[str, Any]] = {}
    for design_name, design in payload["designs"].items():
        counts = design["state_counts"]
        def controls(kind: str) -> np.ndarray:
            count = int(counts[kind])
            values = np.asarray(design[f"{kind}_controls"], dtype=float)
            if count == 0 and values.size == 0:
                values = values.reshape((0, len(channels)))
            return _as_finite_array(
                values,
                name=f"{design_name}.{kind}_controls",
                shape=(count, len(channels)),
            )

        edge = controls("edge")
        tomography = controls("tomography")
        deflector = controls("deflector")
        expected_frames = sum(
            int(counts[kind]) * int(frames_per[kind])
            for kind in ("edge", "tomography", "deflector")
        )
        _require(
            int(design["raw_frame_count"]) == expected_frames,
            f"{design_name} frame accounting is inconsistent.",
        )
        all_controls = np.concatenate((edge, tomography, deflector), axis=0)
        _require(
            np.allclose(all_controls[:, 3:], 0.0, rtol=0.0, atol=0.0),
            f"{design_name} unexpectedly changes a downstream model control.",
        )
        _require(
            np.all(all_controls[:, :3] >= lower[None, :])
            and np.all(all_controls[:, :3] <= upper[None, :]),
            f"{design_name} leaves the declared normalized-control bounds.",
        )
        q = q_bias[None, :] + q_slope[None, :] * all_controls[:, :3]
        _require(np.all(q >= q_floor), f"{design_name} violates the q floor.")
        np.testing.assert_allclose(
            np.min(q, axis=0), design["selected_q_min_per_m"], rtol=2e-14
        )
        np.testing.assert_allclose(
            np.max(q, axis=0), design["selected_q_max_per_m"], rtol=2e-14
        )
        filter_audit = design["filter_audit"]
        edge_rms = _as_finite_array(
            filter_audit["edge_aperture_equivalent_rms_m"],
            name=f"{design_name}.edge_aperture_equivalent_rms_m",
            shape=(int(counts["edge"]),),
        )
        tomography_rms = _as_finite_array(
            filter_audit["tomography_sample_equivalent_rms_m"],
            name=f"{design_name}.tomography_sample_equivalent_rms_m",
            shape=(int(counts["tomography"]),),
        )
        deflector_rms = _as_finite_array(
            filter_audit["deflector_sample_equivalent_rms_m"],
            name=f"{design_name}.deflector_sample_equivalent_rms_m",
            shape=(int(counts["deflector"]),),
        )
        filters = payload["search_contract"]["candidate_filtering"]
        edge_low, edge_high = filters["edge_aperture_equivalent_rms_m"]
        sample_low, sample_high = filters["tomography_sample_equivalent_rms_m"]
        _require(
            np.all((edge_rms >= edge_low) & (edge_rms <= edge_high)),
            f"{design_name} records an edge state outside its candidate filter.",
        )
        _require(
            np.all((tomography_rms >= sample_low) & (tomography_rms <= sample_high)),
            f"{design_name} records a tomography state outside its candidate filter.",
        )
        _require(
            np.all((deflector_rms >= sample_low) & (deflector_rms <= sample_high)),
            f"{design_name} records a deflector state outside its candidate filter.",
        )
        fisher = design["fisher_audit"]
        _require(
            int(fisher["profiled_target_rank"]) == 12,
            f"{design_name} is rank deficient.",
        )
        singular = _as_finite_array(
            fisher["profiled_target_singular_values"],
            name=f"{design_name}.singular_values",
            shape=(12,),
        )
        _require(
            np.all(np.diff(singular) <= 0.0),
            f"{design_name} singular values are not descending.",
        )
        metrics = _as_finite_array(
            fisher["scaled_parameter_standard_uncertainties"],
            name=f"{design_name}.metrics",
            shape=(12,),
        )
        _require(
            np.isclose(np.max(metrics), fisher["maximum_scaled_standard_uncertainty"]),
            f"{design_name} maximum metric is inconsistent.",
        )
        verified_designs[design_name] = {
            "frames": expected_frames,
            "states": int(sum(int(v) for v in counts.values())),
            "rank": 12,
            "max_scaled_standard_uncertainty": float(np.max(metrics)),
        }

    _require(
        verified_designs.get("design_248", {}).get("frames") == 248,
        "The required 248-frame selection is missing.",
    )
    _require(
        verified_designs.get("design_296", {}).get("frames") == 296,
        "The required 296-frame selection is missing.",
    )
    return {
        "schema": SCHEMA,
        "payload": str(payload_file),
        "payload_sha256": _sha256(payload_file),
        "actual_geometry_sha256": geometry_binding["canonical_sha256"],
        "designs": verified_designs,
        "elapsed_s": time.perf_counter() - started,
    }


def _direct_audit(blocks: list[np.ndarray]) -> dict[str, Any]:
    jacobian = np.concatenate(blocks)
    target = jacobian[:, :12]
    nuisance = jacobian[:, 12:]
    norms = np.linalg.norm(nuisance, axis=0)
    active = norms > np.finfo(float).tiny
    nuisance = nuisance[:, active] / norms[active][None, :]
    left, nuisance_singular, _ = np.linalg.svd(nuisance, full_matrices=False)
    nuisance_rank = (
        int(np.sum(nuisance_singular > 1.0e-10 * nuisance_singular[0]))
        if len(nuisance_singular)
        else 0
    )
    projected = target - left[:, :nuisance_rank] @ (
        left[:, :nuisance_rank].T @ target
    )
    singular = np.linalg.svd(projected, compute_uv=False)
    rank = int(np.sum(singular > 1.0e-7 * singular[0]))
    information = projected.T @ projected
    uncertainty = (
        np.full(12, np.inf)
        if rank < 12
        else np.sqrt(np.maximum(np.diag(np.linalg.inv(information)), 0.0))
    )
    return {
        "profiled_target_rank": rank,
        "profiled_nuisance_rank": nuisance_rank,
        "profiled_target_singular_values": singular,
        "scaled_parameter_standard_uncertainties": uncertainty,
    }


def recompute_fisher_audit(
    design_name: str,
    payload_path: str | Path = DEFAULT_PAYLOAD,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Rebuild one selected Fisher audit without repeating candidate search."""

    # Platform selection must occur before the lazy JAX import.  The audit is
    # deliberately CPU-portable; the million-Gaussian imaging path is separate.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    root = Path(repository_root).resolve()
    source_root = str(root / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    import jax
    import jax.numpy as jnp

    from temgym_core.critical_line_calibration import (
        EntranceCovarianceChart,
        aperture_edge_scan_from_moments,
        axial_plane_covariance_observables_with_entrance_covariance,
        hard_edge_matrix_to_plane,
        ideal_angular_kick_response_matrices,
    )
    from temgym_core.neoarm_calibration import hard_edge_column_from_config

    jax.config.update("jax_enable_x64", True)
    started = time.perf_counter()
    payload_file = Path(payload_path).resolve()
    payload = json.loads(payload_file.read_text(encoding="utf-8"))
    if design_name not in payload["designs"]:
        raise DesignVerificationError(f"Unknown design {design_name!r}.")
    design = payload["designs"][design_name]
    with (root / payload["provenance"]["config"]["path"]).open("rb") as stream:
        config = tomllib.load(stream)
    column = hard_edge_column_from_config(config)
    parameters0 = jnp.asarray(column.nominal_parameters)
    parameter_names = column.parameter_names
    n_channels = len(column.channel_names)
    target_indices = np.asarray(
        [
            i
            for i, name in enumerate(parameter_names)
            if name.startswith(("CL1.", "CL3.", "C-mini."))
        ]
    )
    upstream_indices = np.asarray(
        [
            i
            for i, name in enumerate(parameter_names)
            if name == "source_z_m" or name.startswith("Obj-prefield.")
        ]
    )
    target_scales = np.asarray(
        [
            1.0
            if parameter_names[i].endswith(".log_length_m")
            else abs(float(parameters0[i]))
            for i in target_indices
        ]
    )
    upstream_scales = np.asarray(
        [
            1.0e-3
            if parameter_names[i] == "source_z_m"
            else (5.0e-4 if parameter_names[i].endswith("z_center_m") else 0.08)
            for i in upstream_indices
        ]
    )
    edge_states = jnp.asarray(design["edge_controls"])
    tomography_states = jnp.asarray(design["tomography_controls"])
    deflector_states = jnp.asarray(design["deflector_controls"])
    entrance_covariance = jnp.asarray(
        payload["illumination"]["entrance_covariance"]
    )
    entrance_chart = EntranceCovarianceChart(np.asarray(entrance_covariance))
    eta0 = jnp.zeros(10)
    xy = jnp.asarray((0, 2))
    geometry = payload["provenance"]["actual_geometry"]["values"]
    z_aperture = float(geometry["C_aperture"]["z_m"])
    z_upper = float(geometry["CL12_deflector"]["z_m"])
    z_lower = z_upper + float(geometry["CL12_deflector"]["spacing_m"])
    z_sample = float(geometry["sample"])
    radius = float(payload["measurement_contract"]["edge_scan"]["hole_radius_m"])

    def covariance_xy(parameters: Any, eta: Any, states: Any) -> Any:
        covariance = entrance_chart.covariance(eta)

        def one(dacs: Any) -> Any:
            matrix = hard_edge_matrix_to_plane(column, parameters, dacs, z_aperture)
            propagated = matrix @ covariance @ matrix.T
            return propagated[jnp.ix_(xy, xy)]

        return jax.vmap(one)(states)

    normal_angles = jnp.deg2rad(
        jnp.asarray(payload["measurement_contract"]["edge_scan"]["normal_angles_deg"])
    )
    normals = jnp.stack((jnp.cos(normal_angles), jnp.sin(normal_angles)), axis=-1)
    sigma_offsets = jnp.asarray(
        payload["measurement_contract"]["edge_scan"]["offsets_in_local_sigma"]
    )
    truth_covariances = covariance_xy(parameters0, eta0, edge_states)
    expected_filters = design["filter_audit"]
    edge_equivalent_rms = np.sqrt(
        0.5
        * np.asarray(
            truth_covariances[:, 0, 0] + truth_covariances[:, 1, 1]
        )
    )
    np.testing.assert_allclose(
        edge_equivalent_rms,
        expected_filters["edge_aperture_equivalent_rms_m"],
        rtol=2.0e-12,
        atol=1.0e-18,
    )

    def equivalent_sample_radii(states: Any) -> Any:
        def one(dacs: Any) -> Any:
            matrix = hard_edge_matrix_to_plane(
                column, parameters0, dacs, z_sample
            )
            propagated = matrix @ entrance_covariance @ matrix.T
            return jnp.sqrt(0.5 * (propagated[0, 0] + propagated[2, 2]))

        return jax.vmap(one)(states)

    np.testing.assert_allclose(
        np.asarray(equivalent_sample_radii(tomography_states)),
        expected_filters["tomography_sample_equivalent_rms_m"],
        rtol=2.0e-12,
        atol=1.0e-18,
    )
    if len(deflector_states):
        np.testing.assert_allclose(
            np.asarray(equivalent_sample_radii(deflector_states)),
            expected_filters["deflector_sample_equivalent_rms_m"],
            rtol=2.0e-12,
            atol=1.0e-18,
        )
    sigma_normal = jnp.sqrt(
        jnp.einsum("ai,nij,aj->na", normals, truth_covariances, normals)
    )
    edge_commands = (
        normals[None, :, None, :]
        * (
            radius
            + sigma_normal[:, :, None, None]
            * sigma_offsets[None, None, :, None]
        )
    ).reshape((len(edge_states), -1, 2))

    def edge_response(
        parameters: Any,
        eta: Any,
        log_radius: Any,
        log_scale: Any,
        centres_um: Any,
    ) -> Any:
        covariances = covariance_xy(parameters, eta, edge_states)
        prediction = aperture_edge_scan_from_moments(
            jnp.zeros((len(edge_states), 2)),
            covariances,
            aperture_centre_xy_m=centres_um * 1.0e-6,
            aperture_radius_m=radius * jnp.exp(log_radius),
            incident_dose_electrons=1.0,
            aperture_centre_offsets_xy_m=jnp.exp(log_scale) * edge_commands,
            radial_order=int(
                payload["measurement_contract"]["edge_scan"][
                    "radial_quadrature_order"
                ]
            ),
            azimuthal_order=int(
                payload["measurement_contract"]["edge_scan"][
                    "azimuthal_quadrature_order"
                ]
            ),
        )
        return jnp.concatenate(
            (
                prediction.transmission,
                prediction.aperture_out_expected_electrons[:, None],
            ),
            axis=1,
        )

    zero = jnp.asarray(0.0)
    zero_centres = jnp.zeros((len(edge_states), 2))
    edge_truth = np.asarray(
        edge_response(parameters0, eta0, zero, zero, zero_centres)
    )
    edge_jp = np.asarray(
        jax.jacfwd(
            lambda p: edge_response(p, eta0, zero, zero, zero_centres)
        )(parameters0)
    )
    edge_je = np.asarray(
        jax.jacfwd(
            lambda e: edge_response(parameters0, e, zero, zero, zero_centres)
        )(eta0)
    )
    edge_jr = np.asarray(
        jax.jacfwd(
            lambda r: edge_response(parameters0, eta0, r, zero, zero_centres)
        )(zero)
    )
    edge_js = np.asarray(
        jax.jacfwd(
            lambda s: edge_response(parameters0, eta0, zero, s, zero_centres)
        )(zero)
    )
    edge_jc = np.asarray(
        jax.jacfwd(
            lambda c: edge_response(parameters0, eta0, zero, zero, c)
        )(zero_centres)
    )

    slices = {
        "target": slice(0, 12),
        "eta": slice(12, 22),
        "upstream": slice(22, 26),
        "deflector": slice(26, 42),
        "radius": slice(42, 43),
        "stage_scale": slice(43, 44),
    }
    n_global = 44
    edge_blocks: list[np.ndarray] = []
    incident = float(
        payload["noise_and_priors"]["edge_incident_electrons_per_frame"]
    )
    for state_index in range(len(edge_states)):
        probability = np.maximum(edge_truth[state_index], 1.0e-12)
        whiten = np.sqrt(incident / probability)
        block = np.zeros((22, n_global))
        block[:, slices["target"]] = (
            edge_jp[state_index][:, target_indices] * target_scales[None, :]
        )
        block[:, slices["eta"]] = edge_je[state_index]
        block[:, slices["upstream"]] = (
            edge_jp[state_index][:, upstream_indices] * upstream_scales[None, :]
        )
        block[:, slices["radius"]] = edge_jr[state_index, :, None]
        block[:, slices["stage_scale"]] = edge_js[state_index, :, None]
        block *= whiten[:, None]
        local = np.column_stack(
            (
                edge_jc[state_index, :, state_index, 0],
                edge_jc[state_index, :, state_index, 1],
                probability,
            )
        )
        local[:, :2] *= 1.0e-6
        local *= whiten[:, None]
        left, singular, _ = np.linalg.svd(local, full_matrices=False)
        rank = int(np.sum(singular > 1.0e-10 * singular[0]))
        block -= left[:, :rank] @ (left[:, :rank].T @ block)
        edge_blocks.append(block)

    def tomography_response(parameters: Any, eta: Any) -> Any:
        return axial_plane_covariance_observables_with_entrance_covariance(
            column,
            parameters,
            eta,
            tomography_states,
            entrance_chart,
            plane_z_m=z_sample,
            mode="full",
        )

    tomography_jp = np.asarray(
        jax.jacfwd(lambda p: tomography_response(p, eta0))(parameters0)
    )
    tomography_je = np.asarray(
        jax.jacfwd(lambda e: tomography_response(parameters0, e))(eta0)
    )
    tomography_blocks: list[np.ndarray] = []
    tomography_sigma = float(payload["noise_and_priors"]["tomography_summary_sigma"])
    for state_index in range(len(tomography_states)):
        block = np.zeros((10, n_global))
        block[:, slices["target"]] = (
            tomography_jp[state_index][:, target_indices]
            * target_scales[None, :]
        )
        block[:, slices["eta"]] = tomography_je[state_index]
        block[:, slices["upstream"]] = (
            tomography_jp[state_index][:, upstream_indices]
            * upstream_scales[None, :]
        )
        tomography_blocks.append(block / tomography_sigma)

    def deflector_response(parameters: Any, map_flat: Any) -> Any:
        command_map = map_flat.reshape((4, 4))

        def one(dacs: Any) -> Any:
            response = ideal_angular_kick_response_matrices(
                column,
                parameters,
                dacs,
                (z_upper, z_lower),
                observation_z_m=z_sample,
            )
            transport = jnp.concatenate((response[0], response[1]), axis=1)
            return transport @ command_map

        return jax.vmap(one)(deflector_states)

    deflector_blocks: list[np.ndarray] = []
    if len(deflector_states):
        command_map0 = jnp.asarray(
            (
                (1.0, 0.0, 1.0, 0.0),
                (0.0, 1.0, 0.0, 1.0),
                (1.0, 0.0, -1.0, 0.0),
                (0.0, 1.0, 0.0, -1.0),
            )
        ).reshape(-1)
        deflector_jp = np.asarray(
            jax.jacfwd(lambda p: deflector_response(p, command_map0))(
                parameters0
            )
        )
        deflector_jg = np.asarray(
            jax.jacfwd(lambda g: deflector_response(parameters0, g))(
                command_map0
            )
        )
        drive = float(payload["noise_and_priors"]["deflector_drive_rad"])
        position_sigma = float(
            payload["noise_and_priors"]["single_frame_position_sigma_m"]
        )
        tilt_sigma = float(payload["noise_and_priors"]["single_frame_tilt_sigma_rad"])
        sigma = np.repeat(
            np.asarray((position_sigma, tilt_sigma, position_sigma, tilt_sigma))
            / np.sqrt(2.0),
            4,
        ) / drive
        for state_index in range(len(deflector_states)):
            block = np.zeros((16, n_global))
            flat_jp = deflector_jp[state_index].reshape(16, -1)
            block[:, slices["target"]] = (
                flat_jp[:, target_indices] * target_scales[None, :]
            )
            block[:, slices["upstream"]] = (
                flat_jp[:, upstream_indices] * upstream_scales[None, :]
            )
            block[:, slices["deflector"]] = deflector_jg[
                state_index
            ].reshape(16, 16)
            deflector_blocks.append(block / sigma[:, None])

    prior = np.zeros((2, n_global))
    prior[0, slices["radius"]] = 1.0 / float(
        payload["noise_and_priors"]["aperture_log_radius_prior_sigma"]
    )
    prior[1, slices["stage_scale"]] = 1.0 / float(
        payload["noise_and_priors"]["aperture_stage_log_scale_prior_sigma"]
    )
    audit = _direct_audit(
        [prior, *edge_blocks, *tomography_blocks, *deflector_blocks]
    )
    expected = design["fisher_audit"]
    _require(
        audit["profiled_target_rank"]
        == int(expected["profiled_target_rank"]),
        "Recomputed target rank differs.",
    )
    _require(
        audit["profiled_nuisance_rank"]
        == int(expected["profiled_nuisance_rank"]),
        "Recomputed nuisance rank differs.",
    )
    np.testing.assert_allclose(
        audit["profiled_target_singular_values"],
        expected["profiled_target_singular_values"],
        rtol=2.0e-8,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        audit["scaled_parameter_standard_uncertainties"],
        expected["scaled_parameter_standard_uncertainties"],
        rtol=2.0e-8,
        atol=1.0e-10,
    )
    return {
        "design": design_name,
        "profiled_target_rank": audit["profiled_target_rank"],
        "profiled_nuisance_rank": audit["profiled_nuisance_rank"],
        "minimum_profiled_target_singular_value": float(
            audit["profiled_target_singular_values"][-1]
        ),
        "maximum_scaled_standard_uncertainty": float(
            np.max(audit["scaled_parameter_standard_uncertainties"])
        ),
        "elapsed_s": time.perf_counter() - started,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, default=DEFAULT_PAYLOAD)
    parser.add_argument(
        "--audit",
        action="append",
        choices=("design_248", "design_296"),
        default=[],
        help="Rebuild this selected Fisher audit; may be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result: dict[str, Any] = {"verification": load_and_verify(args.payload)}
    if args.audit:
        result["audits"] = [
            recompute_fisher_audit(name, args.payload) for name in args.audit
        ]
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
