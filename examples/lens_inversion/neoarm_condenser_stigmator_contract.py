"""Ideal local condenser information with an internal, uncalibrated stigmator.

This is a preparation-contract counterexample, not an image/recovery audit.
A later quadrupole can retain upstream focusing information without creating
an upstream Larmor tag. Independent pre-quadrupole polarity reversals remain
exactly invisible for the rotationally invariant source and non-overlapping
hard-edge lenses tested here. Real overlapping fields require a separate test.
"""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from neoarm_measured_information_audit import measured_column, CONFIG, DIAGRAM
from temgym_core.constants import energy2wavelength
from temgym_core.immersed_objective import uniform_field_canonical_matrix


ROOT = Path(__file__).resolve().parents[2]
# Keep host constants in float64 even when a CLI enables JAX x64 only later.
GROUPED = np.array([0, 2, 1, 3])
I2 = np.eye(2)
J2 = np.array([[0., -1.], [1., 0.]])
SOURCE_SCALES = np.array([5e-9, 5e-9, 50e-6, 50e-6])
OPTICAL_SCALES = np.tile(np.array([.01, .1, .1, .1]), 3)
PROJECTION_B_OVER_A_M = np.array([-.03, 0., .03])
APERTURE_RADII_M = np.array([3e-6, 10e-6, 30e-6])
CONTROL_VALUES = (-.3, .2, .7)
PARAMETER_NAMES = tuple(f'{lens}.{name}_offset' for lens in ('CL1', 'CL3', 'C-mini')
    for name in ('centre', 'log_length', 'q_bias', 'q_slope')) + (
    'source.log_position_variance', 'source.log_angle_variance',
    'source.symmetric_cross_covariance', 'source.angular_momentum_covariance',
    'quadrupole.z_offset', 'quadrupole.normal_strength', 'quadrupole.skew_strength')


def drift(distance):
    return jnp.eye(4).at[:2, 2:].set(distance*I2)


def quadrupole(strengths):
    """Canonical thin quadrupole; strengths in inverse metres, both unknown."""
    normal, skew = jnp.asarray(strengths)
    gradient = jnp.array([[normal, skew], [skew, -normal]])
    return jnp.eye(4).at[2:, :2].set(-gradient)


def round_source(coordinates):
    """Four rotationally invariant moments, including angular momentum.

    This local chart does not enforce the quantum cone for arbitrary values;
    the audit validates its chosen interior point before differentiating.
    """
    log_a, log_b, symmetric, angular = jnp.asarray(coordinates)
    balanced = jnp.block([[jnp.exp(log_a)*I2, symmetric*I2+angular*J2],
                         [symmetric*I2-angular*J2, jnp.exp(log_b)*I2]])
    return balanced*jnp.outer(SOURCE_SCALES, SOURCE_SCALES)


class InternalStigmatorModel:
    """Fixed order with unknown lens geometry and unknown internal Q plane.

    ``after_cl3`` places Q after the known C-aperture as well as after CL3.
    Geometry perturbations may not change this declared topological ordering.
    The final known projection plane is at -50 mm, not the detector/sample.
    """

    def __init__(self, location):
        if location not in ('after_cl1', 'after_cl3'):
            raise ValueError('Select one explicitly supported stigmator gap.')
        column = measured_column()
        self.location = location
        self.base = jnp.asarray(column.nominal_parameters[2:14])
        self.source_z_m = float(column.nominal_parameters[0])
        self.aperture_z_m = -.42083333333333334
        self.reference_z_m = -.05
        self.quadrupole_nominal_z_m = -.47 if location == 'after_cl1' else -.3
        self.insert_before = 1 if location == 'after_cl1' else 2
        self.controls = jnp.array([[a, b, c] for a in CONTROL_VALUES
                                  for b in CONTROL_VALUES for c in CONTROL_VALUES])

    def decode(self, parameters):
        p = jnp.asarray(parameters)
        if p.shape != (19,):
            raise ValueError('All 19 coordinates are required.')
        optical = (self.base+OPTICAL_SCALES*p[:12]).reshape(3, 4)
        return optical, round_source(p[12:16]), self.quadrupole_nominal_z_m+.01*p[16], quadrupole(30*p[17:19])

    def maps(self, parameters, control):
        optical, _, z_q, q_matrix = self.decode(parameters)
        matrix, cursor = jnp.eye(4), self.source_z_m
        aperture_map = jnp.eye(4)
        for index in range(3):
            centre, log_length, bias, slope = optical[index]
            length = jnp.exp(log_length)
            if index == self.insert_before:
                matrix = q_matrix@drift(z_q-cursor)@matrix
                cursor = z_q
            matrix = drift(centre-length/2-cursor)@matrix
            native = uniform_field_canonical_matrix((bias+slope*control[index])/length, length)
            matrix = native[jnp.ix_(GROUPED, GROUPED)]@matrix
            cursor = centre+length/2
            if index == 1:
                aperture_map = drift(self.aperture_z_m-cursor)@matrix
        return drift(self.reference_z_m-cursor)@matrix, aperture_map

    def observables(self, parameters, *, angular_order=256, include_aperture=False):
        """Known covariance projections; experimental aperture option is off.

        Direct angular aperture quadrature is deliberately excluded from the
        accepted rank contract: the audit records its failure for one case.
        """
        _, covariance, _, _ = self.decode(parameters)
        angles = jnp.linspace(0., 2*jnp.pi, angular_order, endpoint=False)
        circle = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)

        def one(control):
            matrix, aperture_map = self.maps(parameters, control)

            def projection(b):
                row = jnp.concatenate((I2, b*I2), axis=1)@matrix
                cov = row@covariance@row.T
                return jnp.array([.5*jnp.log(cov[0, 0]), .5*jnp.log(cov[1, 1]),
                                  cov[0, 1]/jnp.sqrt(cov[0, 0]*cov[1, 1])])

            output = jax.vmap(projection)(PROJECTION_B_OVER_A_M).reshape(-1)
            if not include_aperture:
                return output
            cov = (aperture_map@covariance@aperture_map.T)[:2, :2]
            precision = jnp.einsum('ai,ij,aj->a', circle, jnp.linalg.inv(cov), circle)

            def count(radius):
                probability = jnp.mean(-jnp.expm1(-.5*radius**2*precision)/precision)/jnp.sqrt(jnp.linalg.det(cov))
                return jnp.log(probability)

            return jnp.r_[output, jax.vmap(count)(APERTURE_RADII_M)]

        return jax.vmap(one)(self.controls).reshape(-1)

    def polarity_reversal(self, parameters, lens_index):
        p = jnp.asarray(parameters)
        optical = self.decode(p)[0]
        indices = jnp.array([4*lens_index+2, 4*lens_index+3])
        return p.at[indices].set((-optical[lens_index, 2:]-self.base[indices])/OPTICAL_SCALES[indices])


def local_information(jacobian):
    jac = np.asarray(jacobian)
    norms = np.linalg.norm(jac, axis=0)
    singular = np.linalg.svd(jac/np.where(norms > 0, norms, 1.), compute_uv=False)
    left, nuisance_singular, _ = np.linalg.svd(jac[:, 12:], full_matrices=False)
    nuisance_rank = int(np.sum(nuisance_singular > nuisance_singular[0]*1e-10))
    profile = jac[:, :12]-left[:, :nuisance_rank]@(left[:, :nuisance_rank].T@jac[:, :12])
    profiled = np.linalg.svd(profile, compute_uv=False)
    return dict(full_rank=int(np.sum(singular > singular[0]*1e-10)), nuisance_rank=nuisance_rank,
        profiled_optical_rank=int(np.sum(profiled > profiled[0]*1e-10)),
        normalized_full_singular_values=singular.tolist(), profiled_optical_singular_values=profiled.tolist(),
        rank_relative_tolerance=1e-10, source_angular_momentum_derivative_norm=float(norms[15]),
        measurement_noise_or_uncertainty_used=False)


def audit(output):
    if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
        raise RuntimeError('Use GPU float64 for this ideal forward/Jacobian audit.')
    parameters = jnp.r_[jnp.zeros(12), jnp.array([0., 0., .03, .04]), jnp.array([0., .7, .4])]
    epsilon = float(energy2wavelength(200e3))/(4*np.pi)
    quantum_scale = np.array([5e-9, 5e-9, epsilon/5e-9, epsilon/5e-9])
    omega = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]])
    covariance = np.asarray(round_source(parameters[12:16]))
    quantum_margin = float(np.linalg.eigvalsh(covariance/np.outer(quantum_scale, quantum_scale)+1j*omega).min())
    if quantum_margin <= 0:
        raise RuntimeError('The synthetic source must be strictly quantum admissible.')
    rows = []
    for location in ('after_cl1', 'after_cl3'):
        model = InternalStigmatorModel(location)
        function = jax.jit(model.observables)
        nominal = np.asarray(function(parameters))
        jac = np.asarray(jax.jit(jax.jacfwd(model.observables))(parameters))
        information = local_information(jac)
        reversals = []
        for index, lens in enumerate(('CL1', 'CL3', 'C-mini')):
            changed = np.asarray(function(model.polarity_reversal(parameters, index)))
            reversals.append(dict(lens=lens, before_stigmator=index < model.insert_before,
                maximum_observable_change=float(np.max(np.abs(changed-nominal))),
                q_bias_and_slope_both_reversed=True))
        comparisons = []
        previous = np.asarray(jax.jit(lambda p: model.observables(
            p, include_aperture=True))(parameters)).reshape(27, 12)[:, 9:]
        for order in (512, 1024):
            current = np.asarray(jax.jit(lambda p: model.observables(
                p, angular_order=order, include_aperture=True))(parameters)).reshape(27, 12)[:, 9:]
            comparisons.append(dict(order=order, maximum_change=float(np.max(abs(current-previous)))))
            previous = current
        rows.append(dict(location=location, stigmator_nominal_z_m=model.quadrupole_nominal_z_m,
            component_order=('Source,CL1,Q,CL3,C-aperture,C-mini,reference' if location == 'after_cl1'
                             else 'Source,CL1,CL3,C-aperture,Q,C-mini,reference'),
            base_optical_parameters=np.asarray(model.base).reshape(3, 4).tolist(), parameters=parameters.tolist(),
            controls=np.asarray(model.controls).tolist(), source_z_m=model.source_z_m,
            aperture_z_m=model.aperture_z_m, reference_z_m=model.reference_z_m,
            information=information, information_uses_covariance_only=True,
            polarity_reversals=reversals,
            optional_aperture_diagnostic=dict(
                aperture_quadrature_refinement=comparisons,
                numerical_refinement_pass=all(c['maximum_change'] < 1e-8 for c in comparisons),
                used_for_any_rank_or_polarity_claim=False,
                interpretation='Direct angular quadrature is not admitted for strongly anisotropic beams; failed refinements are retained, not hidden.')))
    dependencies = [Path(__file__), CONFIG, DIAGRAM,
        ROOT/'tests/test_neoarm_condenser_stigmator_contract.py',
        ROOT/'examples/lens_inversion/neoarm_measured_information_audit.py',
        ROOT/'examples/lens_inversion/neoarm_imaging_recipe.py',
        ROOT/'src/temgym_core/immersed_objective.py', ROOT/'src/temgym_core/neoarm_calibration.py',
        ROOT/'src/temgym_core/microscope_inversion.py',
        ROOT/'src/temgym_core/constants.py', ROOT/'examples/lens_inversion/references/arxiv_2607.29411v1.pdf']
    report = dict(schema='neoarm-internal-stigmator-contract-v1', backend=jax.default_backend(), x64_enabled=True,
        source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies},
        provenance_scope='Direct model/equation sources, geometry/config, cached paper, and focused tests; not an environment lock.',
        jax_version=jax.__version__, numpy_version=np.__version__,
        parameter_names=PARAMETER_NAMES, parameter_count=19,
        optical_coordinates=12, source_coordinates=4, quadrupole_coordinates=3,
        canonical_coordinate_order=['x', 'y', 'pi_x', 'pi_y'],
        covariance_equation='Sigma_out=M_down Q M_up Sigma_source M_up^T Q^T M_down^T',
        quadrupole_equation='Q=[[I,0],[-G,I]], G=[[gx,gy],[gy,-gx]]',
        source_equation='Sigma=diag(scales) [[a I,c I+ell J],[c I-ell J,b I]] diag(scales)',
        source_scale_vector=np.asarray(SOURCE_SCALES).tolist(), source_quantum_margin=quantum_margin,
        projection_B_over_A_m=np.asarray(PROJECTION_B_OVER_A_M).tolist(), aperture_radii_m=np.asarray(APERTURE_RADII_M).tolist(),
        observable_layout='Per control: three (log_sigma_x,log_sigma_y,correlation) projections; 243 covariance coordinates total.',
        aperture_counts_used_for_main_contract=False,
        actual_detector_capture_or_real_control_feasibility_tested=False,
        nonlinear_optical_recovery_performed=False, instrument_stigmator_location_known=False,
        independent_upstream_kicks_or_current_readback_used=False,
        rank_is_not_global_uniqueness_or_precision=True, cases=rows,
        primary_paper=dict(url='https://arxiv.org/html/2607.29411v1', sections=['II.2', 'II.2.1', 'II.2.2', 'Supplement I.4'],
            scope='Examples emphasize projector calibration; upstream VOA ellipticity is a downstream rotation tag. Nion aperture order C1/VOA/C2/C3 differs from this diagram. No NEOARM stigmator location is given.'),
        limitations=[
            'The source is unknown within the four-coordinate rotationally invariant family, not within the full ten-coordinate covariance family.',
            'This ideal rank test has neither finite-pixel images, measured summary covariance, capture gates, nor branch-recovery fits.',
            'The optional direct-angular aperture integral fails refinement for the after-CL1 quadrupole. No aperture count enters any reported information rank or polarity comparison.',
            'Projection B/A values and aperture radii are illustrative measurement capabilities, not validated NEOARM settings.',
            'Both quadrupole strengths and its z position are free local nuisances; the component order is fixed, and actual hardware location is unconfirmed.',
            'A fixed internal quadrupole cannot generally be absorbed into one constant source covariance while upstream lenses vary.',
            'Internal quadrupoles can expose antisymmetric position-angle covariance: the all-round nine-visible-source quotient cannot be reused unchanged.',
            'Independent pre-Q polarity reversals are exact only for non-overlapping isolated round fields and rotationally invariant incident distributions.',
            'Overlapping signed fields can provide polarity-sensitive cross terms in (sum k_i)^2; real-field overlap requires a separate audit.',
            'Near the thin-lens limit, q^2/L focusing can leave length/strength practically confounded despite local hard-edge rank.',
            'Known aperture radii and dose references anchor transverse scale/throughput, but a centred circular aperture is not an orientation reference.'])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    jax.config.update('jax_enable_x64', True)
    result = audit(args.output)
    print([(r['location'], r['information']) for r in result['cases']], flush=True)
