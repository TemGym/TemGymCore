"""Check whether saved projector and condenser calibrations share an instrument.

Generating values are used only to test synthetic-model compatibility, never
to initialize a fit. Matching controlled-lens excitation through an affine
control bridge cannot repair a different objective/sample-prefix map.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from neoarm_aperture_capture import ROOT, load_cases
from neoarm_coupled_projector_fit import grouped_stage
from neoarm_last_lens_tomography import J4, mechanical_shear
from neoarm_shared_objective_audit import provenance_snapshot, verify_provenance_snapshot

FOLDER = ROOT/"examples/generated/neoarm_inversion/imaging_recipe"
GROUPED = np.array([0, 2, 1, 3])
NAMES = ("PL1", "IL3", "IL2", "IL1")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_verified(path):
    report = json.loads(Path(path).read_text())
    for name, expected in report["source_hashes"].items():
        if digest(ROOT/name) != expected: raise ValueError(f"Stale model-compatibility input: {name}")
    return report


def excitation_bridge(old_bias, old_slope, new_bias, new_slope, new_reference=0.):
    """new_u=offset+gain*old_t, equating q without identifying hardware units."""
    values = np.asarray([old_bias, old_slope, new_bias, new_slope, new_reference], float)
    if not np.all(np.isfinite(values)) or new_slope == 0:
        raise ValueError("Finite affine laws and nonzero target slope are required")
    return float(new_reference+(old_bias-new_bias)/new_slope), float(old_slope/new_slope)


def compose_projectors(stages, controls):
    """Grouped canonical IL1-entrance to detector map, rightmost IL1 first."""
    matrix = jnp.eye(4)
    for i in (3, 2, 1, 0): matrix = grouped_stage(jnp.asarray(stages)[4*i:4*i+4], controls[i])@matrix
    return matrix


def sample_kappa_invariant(mechanical_map):
    """For F=S T_kappa, F.T J F=T_kappa.T J T_kappa for any canonical S.

    The grouped position-position block is 2*kappa*J2. This concerns a full
    reconstructed 4D map, not an unqualified one-image centroid measurement.
    """
    matrix = np.asarray(mechanical_map)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("Expected a finite grouped 4x4 mechanical map")
    pullback = matrix.T@np.asarray(J4)@matrix
    return float((pullback[1, 0]-pullback[0, 1])/4), pullback


def run(output):
    if jax.default_backend() != "gpu" or not jax.config.x64_enabled:
        raise RuntimeError("Use GPU float64 for the native optical-map comparison")
    started = perf_counter()
    initial_sources, computational_inputs = provenance_snapshot()
    output = Path(output).resolve()
    projector_path = FOLDER/"il1_extension_images_truth.json"
    shared_path = FOLDER/"shared_objective_information_narrow_probe.json"
    old = read_verified(projector_path); shared = read_verified(shared_path)
    _, _, column, parameters = load_cases(shared_path)
    stages = np.asarray([[np.log(old[name]["length_m"]), np.log(old[name]["gap_m"]),
        old[name]["q_bias"], old[name]["q_slope"]] for name in NAMES])
    bridge = []; new_stages = []; channels = []
    for index, name in enumerate(NAMES):
        lens_index = 6-index; lens = column.lenses[lens_index]
        if lens.name != name: raise ValueError("Unexpected native projector order")
        offset = 2+4*lens_index; centre, log_length, bias, slope = parameters[offset:offset+4]
        length = np.exp(log_length)
        next_entrance = parameters[1] if index == 0 else parameters[offset+4]-np.exp(parameters[offset+5])/2
        gap = next_entrance-centre-length/2
        u0, gain = excitation_bridge(stages[index, 2], stages[index, 3], bias, slope, lens.reference_dac)
        bridge.append(dict(lens=name, shared_channel_index=lens.channel_index,
            offset=u0, gain=gain, formula="shared_dac=offset+gain*old_normalized_control"))
        channels.append(lens.channel_index); new_stages.append([log_length, np.log(gap), bias, slope])
    old_controls = np.array([[0., 0., 0., 0.], [.3, -.1, .2, .1], [-.4, .3, -.2, -.1]])
    new_controls = np.zeros((len(old_controls), 7))
    for index, record in enumerate(bridge):
        new_controls[:, channels[index]] = record["offset"]+record["gain"]*old_controls[:, index]
    old_q = stages[:, 2]+old_controls*stages[:, 3]
    new_q = np.asarray(new_stages)[:, 2]+new_controls[:, channels]*np.asarray(new_stages)[:, 3]
    # All four synthetic native controls currently have zero reference DAC.
    new_q -= np.array([column.lenses[6-i].reference_dac for i in range(4)])*np.asarray(new_stages)[:, 3]
    q_closure = float(np.max(abs(old_q-new_q)))
    if q_closure > 1e-13:
        raise ValueError("Affine excitation bridge does not equate the four q laws")
    post = np.asarray(jax.jit(jax.vmap(lambda c: compose_projectors(stages.ravel(), c)))(old_controls))
    new_canonical = np.asarray(jax.jit(jax.vmap(lambda c:
        column.matrix(parameters, c, segment="sample_to_detector", coordinates="canonical")))(new_controls))
    new_mechanical = np.asarray(jax.jit(jax.vmap(lambda c:
        column.matrix(parameters, c, segment="sample_to_detector", coordinates="mechanical")))(new_controls))
    new_canonical = new_canonical[:, GROUPED][:, :, GROUPED]
    new_mechanical = new_mechanical[:, GROUPED][:, :, GROUPED]
    # The old truth file explicitly stores canonical prefix interleaved and
    # mechanical prefix grouped; they must not receive the same permutation.
    old_prefix_canonical = np.asarray(old["upstream_to_IL1_canonical"])[np.ix_(GROUPED, GROUPED)]
    old_prefix_mechanical = np.asarray(old["upstream_to_IL1_mechanical"])
    old_canonical = post@old_prefix_canonical; old_mechanical = post@old_prefix_mechanical
    new_prefix_canonical = np.linalg.solve(post[0], new_canonical[0])
    new_prefix_mechanical = np.linalg.solve(post[0], new_mechanical[0])
    split = column.objective.split_maps(jnp.asarray(parameters[-4:]))
    kappa_new = float(split.sample_k_m_inv); kappa_old = float(old["sample_kappa_per_m"])
    old_invariant, old_pullback = sample_kappa_invariant(old_prefix_mechanical)
    new_invariant, new_pullback = sample_kappa_invariant(new_prefix_mechanical)
    closure = float(np.max(abs(post@new_prefix_mechanical-new_mechanical))/np.max(abs(new_mechanical)))
    if closure > 1e-10: raise ValueError("The common controlled-lens composition did not reproduce the new native prefix")
    steps = np.array([5e-9, 5e-9, 1e-3, 1e-3]); pixel_m = 55e-6
    rows = []
    for i in range(len(old_controls)):
        rows.append(dict(old_projector_controls=old_controls[i].tolist(), q_matched_shared_controls=new_controls[i].tolist(),
            canonical_relative_matrix_difference=float(np.linalg.norm(old_canonical[i]-new_canonical[i])/np.linalg.norm(old_canonical[i])),
            mechanical_relative_matrix_difference=float(np.linalg.norm(old_mechanical[i]-new_mechanical[i])/np.linalg.norm(old_mechanical[i])),
            maximum_calibrated_basis_position_difference_pixels=float(np.max(abs((old_mechanical[i, :2]-new_mechanical[i, :2])*steps/pixel_m))),
            old_canonical_sample_to_detector=old_canonical[i].tolist(), shared_canonical_sample_to_detector=new_canonical[i].tolist(),
            old_mechanical_sample_to_detector=old_mechanical[i].tolist(), shared_mechanical_sample_to_detector=new_mechanical[i].tolist()))
    sources, _ = provenance_snapshot()
    verify_provenance_snapshot(initial_sources)
    sources.update(old["source_hashes"]); sources.update(shared["source_hashes"])
    for path in (projector_path, shared_path): sources[str(path.relative_to(ROOT))] = digest(path)
    verify_provenance_snapshot(sources)
    report = dict(schema="neoarm-projector-condenser-column-compatibility-v1", source_hashes=sources,
        computational_input_hashes=computational_inputs, backend=jax.default_backend(), x64_enabled=True,
        elapsed_seconds=perf_counter()-started,
        projector_truth_path=str(projector_path.relative_to(ROOT)), projector_truth_sha256=digest(projector_path),
        condenser_truth_path=str(shared_path.relative_to(ROOT)), condenser_truth_sha256=digest(shared_path),
        generating_values_used_only_for_model_compatibility=True, fit_seed_or_noise_selection_performed=False,
        matrices_grouped_order=["x", "y", "canonical_or_mechanical_angle_x", "canonical_or_mechanical_angle_y"],
        projector_stage_order=list(NAMES), geometry_lengths_and_gaps_match=bool(np.allclose(stages[:, :2], np.asarray(new_stages)[:, :2], rtol=0, atol=1e-13)),
        old_stage_logL_loggap_bias_slope=stages.tolist(), shared_stage_logL_loggap_bias_slope=np.asarray(new_stages).tolist(),
        affine_excitation_bridge=bridge, old_sample_kappa_per_m=kappa_old, shared_sample_kappa_per_m=kappa_new,
        maximum_excitation_bridge_error_radians=q_closure,
        old_prefix_pullback_kappa_per_m=old_invariant, shared_prefix_pullback_kappa_per_m=new_invariant,
        old_prefix_symplectic_pullback=old_pullback.tolist(), shared_prefix_symplectic_pullback=new_pullback.tolist(),
        old_canonical_prefix=old_prefix_canonical.tolist(), shared_canonical_prefix=new_prefix_canonical.tolist(),
        old_mechanical_prefix=old_prefix_mechanical.tolist(), shared_mechanical_prefix=new_prefix_mechanical.tolist(),
        q_matched_controlled_section_native_closure_relative=closure, comparisons=rows,
        existing_projector_posterior_is_a_valid_shared_column_prior=False,
        same_synthetic_instrument_under_affine_DAC_bridge=False,
        correct_next_conditional_adapter="Acquire signed calibrated sample inputs through the same condenser-column analyser controls; fit their endpoint response maps jointly with original condenser observations. Retain fixed prefield/sample-kappa as explicit conditional knowledge for that first adapter.",
        likelihood_assembly="Use each original response/image covariance block once; no posterior prior from the same responses. Shared endpoint-map parameters propagate calibration uncertainty into all condenser frames through the joint likelihood.",
        canonical_composition="H_j=R(phi_j)[a_j I,b_j I] T(kappa); F_pre_mechanical=T(kappa)^-1 F_pre_canonical. Their T factors cancel for field-free source-to-detector transport.",
        limitations=["Three q-matched control examples certify this incompatibility, not an exhaustive fit or new physical-instrument experiment.",
            "The bridge equates synthetic excitation laws; it grants neither known currents nor safe hardware command ranges.",
            "The two datasets retain useful separate demonstrations, but share different fixed objective/sample-prefix hypotheses.",
            "A measured sample-to-IL1 prefix does not determine the objective prefield without a physical field model tying both sides.",
            "Allowing generic pre/post maps can produce a predictive section fit; it does not identify objective field shape or absolute section location."])
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(geometry_matches=report["geometry_lengths_and_gaps_match"],
        sample_kappa=[kappa_old, kappa_new], max_basis_difference_pixels=max(r["maximum_calibrated_basis_position_difference_pixels"] for r in rows),
        compatible=report["same_synthetic_instrument_under_affine_DAC_bridge"])), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); jax.config.update("jax_enable_x64", True); run(args.output)
