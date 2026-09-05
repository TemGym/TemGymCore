"""Ideal internal-stigmator audit with ten unknown source moments.

No images, calibrated kicks, current readback or recovery fits are used.  Exact
source-congruence families distinguish missing excitation from local rank.
"""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from neoarm_condenser_stigmator_contract import (
    InternalStigmatorModel, ROOT, SOURCE_SCALES, OPTICAL_SCALES,
    PROJECTION_B_OVER_A_M, drift, quadrupole, round_source,
)
from temgym_core.constants import energy2wavelength

LOWER = np.tril_indices(4, -1)
OMEGA = np.block([[np.zeros((2, 2)), np.eye(2)],
                  [-np.eye(2), np.zeros((2, 2))]])
J2 = np.array([[0., -1.], [1., 0.]])
INVISIBLE = np.block([[np.zeros((2, 2)), J2], [-J2, np.zeros((2, 2))]])


def source_covariance(coordinates):
    """Full-rank ten-coordinate SPD chart; quantum feasibility is checked below."""
    p = jnp.asarray(coordinates)
    if p.shape != (10,):
        raise ValueError('Ten source covariance coordinates are required.')
    lower = jnp.diag(jnp.exp(p[:4])).at[LOWER].set(p[4:])
    return (lower@lower.T)*jnp.outer(SOURCE_SCALES, SOURCE_SCALES)


def source_coordinates(covariance):
    c = np.asarray(covariance, float)
    if c.shape != (4, 4) or not np.isfinite(c).all() or not np.allclose(c, c.T, atol=0, rtol=1e-12):
        raise ValueError('A finite symmetric4x4 source covariance is required.')
    lower = np.linalg.cholesky(c/np.outer(SOURCE_SCALES, SOURCE_SCALES))
    return np.r_[np.log(np.diag(lower)), lower[LOWER]]


def quantum_margin(covariance):
    epsilon = float(energy2wavelength(200e3))/(4*np.pi)
    scales = np.array([5e-9, 5e-9, epsilon/5e-9, epsilon/5e-9])
    return float(np.linalg.eigvalsh(np.asarray(covariance)/np.outer(scales, scales)+1j*OMEGA).min())


class GeneralSourceStigmator(InternalStigmatorModel):
    def decode(self, parameters):
        p = jnp.asarray(parameters)
        if p.shape != (25,):
            raise ValueError('All25 optical/source/quadrupole coordinates are required.')
        optical = (self.base+OPTICAL_SCALES*p[:12]).reshape(3, 4)
        return optical, source_covariance(p[12:22]), self.quadrupole_nominal_z_m+.01*p[22], quadrupole(30*p[23:25])


def initial_parameters(*, rotationally_invariant=False):
    covariance = (np.asarray(round_source(jnp.array([0., 0., .03, .04])))
                  if rotationally_invariant else np.asarray(source_covariance(
                      jnp.r_[jnp.log(jnp.array([1., 1.2, 2., 1.8])),
                             jnp.array([.15, .08, .12, -.17, .04, .13])])) )
    return np.r_[np.zeros(12), source_coordinates(covariance), [0., .7, .4]]


def information(jacobian):
    """Column-scaled SVD and nuisance projection; not a noise-weighted Fisher."""
    j = np.asarray(jacobian)
    scale = np.linalg.norm(j, axis=0)
    normalized = j/np.where(scale > 1e-25, scale, 1.)
    singular = np.linalg.svd(normalized, compute_uv=False)
    nuisance = normalized[:, 12:]
    u, ns, _ = np.linalg.svd(nuisance, full_matrices=False)
    nrank = int(np.sum(ns > ns[0]*1e-10))
    optical = normalized[:, :12]
    profiled = optical-u[:, :nrank]@(u[:, :nrank].T@optical)
    ps = np.linalg.svd(profiled, compute_uv=False)
    return dict(full_rank=int(np.sum(singular > singular[0]*1e-10)),
        nuisance_rank=nrank, profiled_optical_rank=int(np.sum(ps > max(ps[0], 1.)*1e-10)),
        normalized_singular_values=singular.tolist(), profiled_column_scaled_singular_values=ps.tolist(),
        column_derivative_norms=scale.tolist(), relative_rank_tolerance=1e-10,
        optical_coordinates=12, source_coordinates=10, quadrupole_coordinates=3,
        measurement_noise_used=False)


def fixed_upstream_source_family(model, parameters, changed_q, *, reference_control=None,
                                 optical_offsets=None):
    """Absorb an arbitrary new Q in one source when pre-Q controls are fixed.

    If F and F' map the same source plane through different Q choices at one
    reference control, Sigma'=F'^{-1} F Sigma F^T F'^{-T}.  Downstream variable
    optics cancel in this congruence; independently varying pre-Q optics do not.
    """
    p = np.asarray(parameters).copy()
    control = np.zeros(3) if reference_control is None else np.asarray(reference_control)
    candidate = p.copy(); candidate[22:] = changed_q
    if optical_offsets is not None:
        offset = np.asarray(optical_offsets)
        if offset.shape != (12,) or np.any(offset[4*model.insert_before:] != 0):
            raise ValueError('Only fixed pre-Q optical parameters may enter this family.')
        candidate[:12] += offset
    original_map = np.asarray(model.maps(p, control)[0])
    changed_map = np.asarray(model.maps(candidate, control)[0])
    congruence = np.linalg.solve(changed_map, original_map)
    covariance = congruence@np.asarray(model.decode(p)[1])@congruence.T
    candidate[12:22] = source_coordinates((covariance+covariance.T)/2)
    return candidate, congruence


def audit(output):
    if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
        raise RuntimeError('Use GPU float64 for the ideal forward/Jacobian audit.')
    p = initial_parameters(); covariance = np.asarray(source_covariance(p[12:22]))
    if quantum_margin(covariance) <= 0:
        raise RuntimeError('The full source must lie strictly inside the quantum cone.')
    rows = []
    for location in ('after_cl1', 'after_cl3'):
        model = GeneralSourceStigmator(location)
        function = jax.jit(model.observables)
        nominal = np.asarray(function(p))
        jacobian = jax.jit(jax.jacfwd(model.observables))
        complete = information(jacobian(p))
        restricted = GeneralSourceStigmator(location)
        controls = np.asarray(restricted.controls).copy()
        controls[:, :restricted.insert_before] = 0.
        restricted.controls = jnp.asarray(np.unique(controls, axis=0))
        fixed_function = jax.jit(restricted.observables)
        fixed_nominal = np.asarray(fixed_function(p))
        fixed_info = information(jax.jit(jax.jacfwd(restricted.observables))(p))
        families = []
        variations = [((-.2, -.12, .08), None), ((.15, .1, -.05), None), ((.4, -.05, .1), None)]
        optical_offsets = np.zeros(12)
        optical_offsets[:4*model.insert_before] = np.tile([.02, .05, .1, -.15], model.insert_before)
        variations.append(((.1, .08, -.04), optical_offsets))
        aperture_map = np.asarray(model.maps(p, np.zeros(3))[1])
        aperture_covariance = (aperture_map@covariance@aperture_map.T)[:2, :2]
        for (dz, dg, dh), optical_offsets in variations:
            candidate, congruence = fixed_upstream_source_family(model, p, p[22:]+[dz, dg, dh], optical_offsets=optical_offsets)
            changed = np.asarray(fixed_function(candidate))
            candidate_covariance = np.asarray(model.decode(candidate)[1])
            changed_aperture_map = np.asarray(model.maps(candidate, np.zeros(3))[1])
            changed_aperture = (changed_aperture_map@candidate_covariance@changed_aperture_map.T)[:2, :2]
            families.append(dict(quadrupole_parameter_change=[dz, dg, dh],
                pre_Q_optical_coordinate_change=(np.zeros(12) if optical_offsets is None else optical_offsets).tolist(),
                parameters=candidate.tolist(), quadrupole_position_m=float(model.decode(candidate)[2]),
                fixed_upstream_maximum_observable_change=float(np.max(abs(changed-fixed_nominal))),
                full_scan_maximum_observable_change=float(np.max(abs(np.asarray(function(candidate))-nominal))),
                congruence_symplectic_residual=float(np.max(abs(congruence.T@OMEGA@congruence-OMEGA))),
                source_quantum_margin=quantum_margin(candidate_covariance),
                aperture_spatial_covariance_relative_change=float(np.linalg.norm(changed_aperture-aperture_covariance)/np.linalg.norm(aperture_covariance))))
        # Source reference is a coordinate gauge even with every lens excited.
        source_plane_families = []
        for shift in (-.002, .003):
            moved = GeneralSourceStigmator(location); moved.source_z_m += shift
            d = np.asarray(drift(shift)); c = d@covariance@d.T
            pp = p.copy(); pp[12:22] = source_coordinates((c+c.T)/2)
            difference = np.asarray(jax.jit(moved.observables)(pp))-nominal
            source_plane_families.append(dict(source_plane_shift_m=shift,
                maximum_observable_change=float(np.max(abs(difference))),
                source_quantum_margin=quantum_margin(c)))
        # With Q=0 its location is arbitrary; the round-system angular-momentum
        # covariance is unobservable again, including for an astigmatic source.
        zero = p.copy(); zero[23:] = 0.
        zero_nominal = np.asarray(function(zero)); zero_families = []
        for angular in (-.05, .05):
            pp = zero.copy(); pp[22] += .3
            c = covariance+angular*INVISIBLE*np.outer(SOURCE_SCALES, SOURCE_SCALES)
            pp[12:22] = source_coordinates(c)
            zero_families.append(dict(invisible_balanced_covariance_change=angular,
                quadrupole_position_change_m=.003,
                maximum_observable_change=float(np.max(abs(np.asarray(function(pp))-zero_nominal))),
                source_quantum_margin=quantum_margin(c)))
        polarity = []
        for index, name in enumerate(('CL1', 'CL3', 'C-mini')):
            changed = np.asarray(function(model.polarity_reversal(p, index)))
            polarity.append(dict(lens=name, before_Q=index < model.insert_before,
                source_and_Q_unchanged=True, alternative_nuisances_refitted=False,
                maximum_observable_change=float(np.max(abs(changed-nominal)))))
        round_parameters = initial_parameters(rotationally_invariant=True)
        round_observables = np.asarray(function(round_parameters))
        round_polarity = [dict(lens=('CL1', 'CL3', 'C-mini')[index],
            maximum_observable_change=float(np.max(abs(np.asarray(function(
                model.polarity_reversal(round_parameters, index)))-round_observables))))
            for index in range(model.insert_before)]
        rows.append(dict(location=location, parameters=p.tolist(), controls=np.asarray(model.controls).tolist(),
            source_reference_z_m=model.source_z_m, quadrupole_nominal_z_m=model.quadrupole_nominal_z_m,
            full_scan_information=complete, fixed_upstream_controls=np.asarray(restricted.controls).tolist(),
            fixed_upstream_information=fixed_info, exact_source_Q_families=families,
            exact_source_reference_families=source_plane_families, zero_Q_exact_families=zero_families,
            general_source_polarity_reversals=polarity,
            rotationally_invariant_source_exact_pre_Q_polarity_alternatives=round_polarity,
            round_point_with_ten_unknown_source_coordinates=information(jacobian(round_parameters))))
    old = ROOT/'examples/generated/neoarm_inversion/imaging_recipe/condenser_stigmator_contract.json'
    inherited = json.loads(old.read_text())
    sources = dict(inherited['source_hashes'])
    for path in (Path(__file__), ROOT/'tests/test_neoarm_condenser_general_source_contract.py', old):
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path, checksum in sources.items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != checksum:
            raise RuntimeError(f'A source changed: {path}')
    family_margins = [member['source_quantum_margin'] for row in rows
        for key in ('exact_source_Q_families', 'exact_source_reference_families', 'zero_Q_exact_families')
        for member in row[key]]
    if min(family_margins) <= 0:
        raise RuntimeError('A claimed exact source family left the physical quantum cone.')
    report = dict(schema='neoarm-general-source-internal-stigmator-contract-v1',
        backend=jax.default_backend(), x64_enabled=True, source_hashes=sources,
        source_covariance=covariance.tolist(), source_quantum_margin=quantum_margin(covariance),
        source_scale_vector=SOURCE_SCALES.tolist(),
        all_exact_family_members_quantum_physical=True,
        minimum_exact_family_quantum_margin=min(family_margins),
        source_chart='Full10-dimensional Cholesky SPD chart; quantum cone validated at every finite family member, not enforced for arbitrary coordinates.',
        exact_family_equations=dict(
            fixed_upstream='Sigma_new=F_new^{-1} F_old Sigma_old F_old^T F_new^{-T}; F_old/new use the same fixed pre-Q lens controls. Canonical congruence is symplectic.',
            source_reference='z_source_new=z_source+delta; Sigma_new=D(delta) Sigma_old D(delta)^T. This leaves every acquisition unchanged.',
            zero_Q='Q=I: z_Q arbitrary and Sigma_new=Sigma_old+ell [[0,J],[-J,0]] in matched transverse scales, within the quantum cone.'),
        canonical_coordinate_order=['x', 'y', 'pi_x', 'pi_y'],
        projection_B_over_A_m=PROJECTION_B_OVER_A_M.tolist(),
        ideal_observables='243 log-width/correlation coordinates from27 Cartesian control states and3 known covariance analysers; no experimental noise.',
        cases=rows, rank_is_not_recovery_or_global_uniqueness=True,
        finite_pixel_images_generated=False, hardware_feasibility_tested=False,
        aperture_counts_used=False, upstream_kicks_or_current_readback_used=False,
        limitations=[
            'The25-coordinate audit fixes source-reference gauge, known analyser maps, longitudinal reference, energy, topology and isolated hard-edge fields.',
            'A general unchanged source is unknown; a fixed thin quadrupole has unknown position and two strengths. Its real instrument location and preparation stability are not supplied.',
            'Exact source/Q congruences show why downstream-only excitation cannot locate or calibrate an unknown internal stigmator.',
            'A calibrated covariance measurement before Q can break these particular downstream-only families. The recorded aperture-covariance change is a geometric diagnostic, not a rendered aperture/count measurement; no aperture data enter rank.',
            'A source-plane drift gauge survives all controls unless the source reference is fixed; this is not recovery of an absolute physical gun/source location.',
            'Zero stigmator strength restores exact location and angular-momentum-source ambiguities. Nonzero generic local rank cannot exclude these special cases.',
            'Known anisotropic source covariance is not granted. A generic anisotropic unknown source can carry upstream polarity information, but a rotationally invariant source retains the old discrete ambiguity.',
            'The generic-source finite polarity tests hold source and Q fixed; they do not profile alternative nuisance values or prove global branch uniqueness.',
            'Any full local rank here is an ideal capability ceiling, without finite-pixel capture, covariance uncertainty, heldout images, global branch search or practical precision.',
            'Continuous source/Q families are tested at finite shifts within each declared gap; arbitrary field overlap and differing component orders require a separate model.'])
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); jax.config.update('jax_enable_x64', True)
    result = audit(args.output)
    print([(r['location'], r['full_scan_information']['full_rank'],
            r['full_scan_information']['profiled_optical_rank']) for r in result['cases']], flush=True)
