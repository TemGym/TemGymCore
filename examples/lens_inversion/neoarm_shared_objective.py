"""Measured-geometry NEOARM variant with one immersed objective field.

Seven controllable hard-edge lenses surround one fixed-support smooth field.
The field is fitted at one objective control setting; its pre/post sections
share strength, peak position, width, and asymmetry. Mechanical sample inputs
are converted to canonical coordinates inside the current fitted field.

This is a new synthetic model, not a reinterpretation of the saved split-field
truth. Its finite objective support and provisional lens/DAC laws require
model-error and hardware checks before an instrument claim.
"""

from dataclasses import dataclass, replace
import argparse
import hashlib
import json
from pathlib import Path
import tomllib

import jax
import jax.numpy as jnp
import numpy as np

from temgym_core.immersed_objective import (
    SharedObjectiveField, canonical_to_mechanical, mechanical_to_canonical,
    uniform_field_canonical_matrix,
)
from temgym_core.jax_least_squares import least_squares
from temgym_core.microscope_inversion import drift_matrix, larmor_rotation_4x4
from temgym_core.neoarm_calibration import hard_edge_column_from_config
from neoarm_imaging_recipe import column_with_geometry_anchors
from neoarm_measured_initialization import first_conjugate_strength


OBJECTIVE_PARAMETER_NAMES = (
    "objective.q_total", "objective.z_peak_m", "objective.log_width_m",
    "objective.asymmetry",
)
CONDENSER_NAMES = ("CL1", "CL3", "C-mini")
PROJECTOR_NAMES = ("IL1", "IL2", "IL3", "PL1")


@dataclass(frozen=True, eq=False)
class SharedObjectiveColumn:
    lenses: tuple
    objective: SharedObjectiveField
    objective_initial: tuple
    sample_z_m: float
    source_z_m: float
    detector_z_m: float
    channel_names: tuple

    def __post_init__(self):
        names = tuple(lens.name for lens in self.lenses)
        if names != CONDENSER_NAMES + PROJECTOR_NAMES:
            raise ValueError("Expected CL1/CL3/C-mini then IL1/IL2/IL3/PL1.")
        if any(lens.channel_index is None for lens in self.lenses):
            raise ValueError("Only the seven controllable lenses belong outside the objective.")
        if len(self.channel_names) != 7 or len(set(self.channel_names)) != 7:
            raise ValueError("Seven unique free-lens channel names are required.")
        if {lens.channel_index for lens in self.lenses} != set(range(7)):
            raise ValueError("Each free-lens channel must occur exactly once.")
        if float(self.objective.sample_z_m) != float(self.sample_z_m):
            raise ValueError("Objective and column sample planes must agree.")
        if np.asarray(self.objective_initial).shape != (4,):
            raise ValueError("The objective requires four physical profile parameters.")
        self.objective.validate_parameters(self.objective_initial)
        clearances = np.asarray(self.geometry_clearances(self.nominal_parameters))
        if not np.all(np.isfinite(clearances)) or np.any(clearances < 0):
            raise ValueError("Lens bodies, source, detector, and objective support must not overlap.")

    @classmethod
    def from_hard_edge(cls, column, *, objective=None, objective_initial=None):
        sample = float(column.sample_z_m)
        if objective is None:
            objective = SharedObjectiveField.from_window(sample-.05, sample, sample+.05,
                                                        n_slices_per_side=256)
        if objective_initial is None:
            objective_initial = (-3., sample-.0005, np.log(.002), .15)
        lenses = tuple(lens for lens in column.lenses
                       if lens.name not in {"Obj-prefield", "Obj-post"})
        return cls(lenses, objective, tuple(objective_initial), sample,
                   column.source_z_m, column.detector_z_m, column.channel_names)

    @property
    def parameter_names(self):
        return ("source_z_m", "detector_z_m") + tuple(
            f"{lens.name}.{field}" for lens in self.lenses
            for field in ("z_center_m", "log_length_m", "q_bias", "q_slope_per_dac")
        ) + OBJECTIVE_PARAMETER_NAMES

    @property
    def n_parameters(self):
        return len(self.parameter_names)

    @property
    def nominal_parameters(self):
        return np.array([self.source_z_m, self.detector_z_m] + [
            value for lens in self.lenses for value in
            (lens.z_center_m, np.log(lens.length_m), lens.q_bias, lens.q_slope_per_dac)
        ] + list(self.objective_initial), dtype=float)

    def objective_parameters(self, parameters):
        return jnp.asarray(parameters)[-4:]

    def _decoded_lenses(self, parameters):
        parameters = jnp.asarray(parameters)
        if parameters.shape != (self.n_parameters,):
            raise ValueError(f"Expected {self.n_parameters} optical parameters.")
        return tuple((lens, parameters[2+4*i], jnp.exp(parameters[3+4*i]),
                      parameters[4+4*i], parameters[5+4*i])
                     for i, lens in enumerate(self.lenses))

    def trace(self, parameters, dac_values, ray, *, segment="full", coordinates="mechanical"):
        """Trace one ray; sample endpoints can be mechanical or canonical.

        Full-column endpoints are field-free, so either convention gives the
        same full map. Gaussian propagation must request canonical coordinates
        when its input or output lies at the immersed sample.
        """
        if segment not in {"full", "source_to_sample", "sample_to_detector"}:
            raise ValueError("Unknown column segment.")
        if coordinates not in {"mechanical", "canonical"}:
            raise ValueError("Declare mechanical or canonical sample coordinates.")
        p, controls, state = map(jnp.asarray, (parameters, dac_values, ray))
        if controls.shape != (7,) or state.shape != (4,):
            raise ValueError("Expected seven controls and one four-coordinate ray.")
        decoded = self._decoded_lenses(p)
        split = self.objective.split_maps(self.objective_parameters(p))
        z_min, z_max = self.objective.z_edges_m[0], self.objective.z_edges_m[-1]

        def traverse(state, values, cursor):
            for lens, centre, length, bias, slope in values:
                state = drift_matrix(centre-length/2-cursor) @ state
                q = bias + slope*(controls[lens.channel_index]-lens.reference_dac)
                state = uniform_field_canonical_matrix(q/length, length) @ state
                cursor = centre + length/2
            return state, cursor

        if segment in {"full", "source_to_sample"}:
            state, cursor = traverse(state, decoded[:3], p[0])
            state = split.pre @ (drift_matrix(z_min-cursor) @ state)
            if segment == "source_to_sample":
                return (canonical_to_mechanical(split.sample_k_m_inv) @ state
                        if coordinates == "mechanical" else state)
        elif coordinates == "mechanical":
            state = mechanical_to_canonical(split.sample_k_m_inv) @ state
        state = split.post @ state
        state, cursor = traverse(state, decoded[3:], z_max)
        return drift_matrix(p[1]-cursor) @ state

    def matrix(self, parameters, dac_values, *, segment="full", coordinates="mechanical"):
        """Transfer matrix from differentiation of one traced input ray."""
        return jax.jacfwd(lambda ray: self.trace(parameters, dac_values, ray,
            segment=segment, coordinates=coordinates))(jnp.zeros(4, dtype=jnp.asarray(parameters).dtype))

    def source_to_sample_matrix(self, parameters, dac_values, *, coordinates="mechanical"):
        return self.matrix(parameters, dac_values, segment="source_to_sample", coordinates=coordinates)

    def sample_position_jacobian(self, parameters, dac_values):
        matrix = self.matrix(parameters, dac_values, segment="sample_to_detector")
        xy = jnp.array([0, 2])
        return matrix[jnp.ix_(xy, xy)]

    def matrix_to_plane(self, parameters, dac_values, z_m):
        """Source to an upstream aperture plane, including partial lens bodies.

        Valid only between source and the fixed objective entrance. The caller
        must enforce that support for uncertain aperture position. Position
        rows are physical at every plane, including inside a hard-edge field;
        momentum rows inside a lens are canonical. This method is intended for
        aperture throughput, not a calibrated mechanical-angle measurement.
        """
        p, controls, target = map(jnp.asarray, (parameters, dac_values, z_m))
        if controls.shape != (7,) or target.ndim != 0:
            raise ValueError("Expected seven controls and a scalar aperture plane.")
        total, cursor = jnp.eye(4, dtype=p.dtype), p[0]
        for lens, centre, length, bias, slope in self._decoded_lenses(p)[:3]:
            entrance = centre-length/2
            distance = jnp.clip(target-cursor, 0., entrance-cursor)
            total = drift_matrix(distance) @ total
            body_length = jnp.clip(target-entrance, 0., length)
            q = bias + slope*(controls[lens.channel_index]-lens.reference_dac)
            total = uniform_field_canonical_matrix(q/length, body_length) @ total
            cursor = centre+length/2
        total = drift_matrix(jnp.maximum(target-cursor, 0.)) @ total
        valid = (target >= p[0]) & (target <= self.objective.z_edges_m[0])
        return jnp.where(valid, total, jnp.nan)

    def geometry_clearances(self, parameters):
        p = jnp.asarray(parameters)
        decoded = self._decoded_lenses(p)
        clearances, cursor = [], p[0]
        for i, (_, centre, length, _, _) in enumerate(decoded):
            if i == 3:
                clearances.append(self.objective.z_edges_m[0]-cursor)
                cursor = self.objective.z_edges_m[-1]
            clearances.append(centre-length/2-cursor)
            cursor = centre+length/2
        clearances.append(p[1]-cursor)
        return jnp.stack(clearances)

    def default_bounds(self, *, position_half_width_m=.01, length_ratio=(.5, 2.),
                       q_half_width=.75, slope_half_width=.5):
        """Independent optical boxes that preserve non-overlapping field support.

        These are synthetic inference bounds, not hardware safety limits or
        evidence that the excitation branch is unique.
        """
        if (not np.all(np.isfinite([position_half_width_m, *length_ratio, q_half_width, slope_half_width]))
                or min(position_half_width_m, q_half_width, slope_half_width) <= 0
                or not 0 < length_ratio[0] < 1 < length_ratio[1]):
            raise ValueError("Bounds require positive finite widths and a length ratio bracketing one.")
        lower, upper = self.nominal_parameters.copy(), self.nominal_parameters.copy()
        lower[:2] -= position_half_width_m
        upper[:2] += position_half_width_m
        for i, lens in enumerate(self.lenses):
            if i in (0, 3):
                left = upper[0] if i == 0 else self.objective.z_edges_m[-1]
            else:
                left = .5*(self.lenses[i-1].exit_z+lens.entrance_z)
            if i in (2, 6):
                right = self.objective.z_edges_m[0] if i == 2 else lower[1]
            else:
                right = .5*(lens.exit_z+self.lenses[i+1].entrance_z)
            move = min(position_half_width_m, .4*(lens.entrance_z-left), .4*(right-lens.exit_z))
            maximum_length = min(lens.length_m*length_ratio[1],
                                 2*(lens.z_center_m-move-left), 2*(right-lens.z_center_m-move))
            if move <= 0 or maximum_length <= lens.length_m:
                raise ValueError("Requested source/detector bounds leave no valid lens geometry box.")
            index = 2+4*i
            lower[index:index+4] = (lens.z_center_m-move, np.log(lens.length_m*length_ratio[0]),
                                    lens.q_bias-q_half_width, lens.q_slope_per_dac-slope_half_width)
            upper[index:index+4] = (lens.z_center_m+move, np.log(maximum_length),
                                    lens.q_bias+q_half_width, lens.q_slope_per_dac+slope_half_width)
        q, peak, log_width, _ = self.objective_initial
        lower[-4:] = (q-q_half_width, max(peak-.005, self.objective.z_edges_m[0]+.001),
                       log_width+np.log(.25), -.5)
        upper[-4:] = (q+q_half_width, min(peak+.005, self.objective.z_edges_m[-1]-.001),
                       log_width+np.log(4.), .5)
        return lower, upper


def shared_imaging_metrics(matrix):
    """Magnification/rotation plus the full mechanical equivalent-drift block."""
    xy, momentum = jnp.array([0, 2]), jnp.array([1, 3])
    a, b = matrix[jnp.ix_(xy, xy)], matrix[jnp.ix_(xy, momentum)]
    equivalent = jnp.linalg.solve(a, b)
    return jnp.concatenate((jnp.array([jnp.sqrt(jnp.sum(a*a)/2),
        jnp.arctan2(a[1, 0], a[0, 0])]), equivalent.reshape(-1)))


@dataclass(frozen=True)
class SharedImagingInitialization:
    column: SharedObjectiveColumn
    parameters: np.ndarray
    magnifications: np.ndarray
    nominal_controls: np.ndarray
    nominal_matrices: np.ndarray
    nominal_metrics: np.ndarray
    relay_fractions: np.ndarray
    relay_image_planes_m: np.ndarray
    connected_controls: np.ndarray
    connected_metrics: np.ndarray
    objective_root_bracket: tuple


def initialize_shared_imaging(base_column, *, magnifications=(40000., 80000.),
                              objective=None, objective_initial=None,
                              objective_image_gap_m=.001, relay_fraction_bounds=(.5, .995),
                              continuation_points=9, objective_sign=-1.):
    """First objective conjugacy crossing followed by four finite-lens relays.

    This initializes the common field once, then keeps it fixed across both
    magnifications. A first sampled root crossing is a local branch convention;
    it is not evidence identifying the instrument's excitation branch.
    """
    column = (base_column if isinstance(base_column, SharedObjectiveColumn) else
              SharedObjectiveColumn.from_hard_edge(base_column, objective=objective,
                                                   objective_initial=objective_initial))
    targets, bounds = np.asarray(magnifications, float), np.asarray(relay_fraction_bounds, float)
    if (targets.ndim != 1 or not 1 <= targets.size <= 2 or not np.all(np.isfinite(targets))
            or np.any(targets <= 0) or np.any(np.diff(targets) <= 0)):
        raise ValueError("Choose one or two positive increasing magnifications.")
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or not 0 < bounds[0] < bounds[1] < 1:
        raise ValueError("Relay-fraction bounds must lie strictly inside (0,1).")
    if (not np.isfinite(objective_image_gap_m) or objective_image_gap_m <= 0
            or objective_sign not in (-1., 1.)
            or isinstance(continuation_points, bool) or int(continuation_points) != continuation_points
            or continuation_points < 2):
        raise ValueError("Invalid image gap, field sign, or continuation count.")
    projectors = column.lenses[3:]
    first_image = projectors[0].entrance_z-objective_image_gap_m
    if not column.objective.z_edges_m[-1] < first_image < projectors[0].entrance_z:
        raise ValueError("Objective image must lie between its support and IL1.")
    initial_objective = jnp.asarray(column.objective_initial)
    edges = jnp.asarray(column.objective.z_edges_m)
    midpoint = .5*(edges[1:]+edges[:-1])
    post_weights = jnp.where(midpoint > column.sample_z_m, jnp.diff(edges), 0.)

    def objective_image(q):
        p = initial_objective.at[0].set(objective_sign*q)
        split = column.objective.split_maps(p)
        phase = jnp.sum(column.objective.field_samples(p)*post_weights)
        return larmor_rotation_4x4(phase) @ drift_matrix(first_image-edges[-1]) @ split.post

    b = jax.jit(lambda q: objective_image(q)[0, 1])
    q_grid = np.linspace(0., 4*np.pi, 257)
    b_grid = np.asarray(jax.jit(jax.vmap(b))(jnp.asarray(q_grid)))
    crossings = np.flatnonzero((b_grid[:-1] >= 0) & (b_grid[1:] < 0))
    if len(crossings) == 0:
        raise RuntimeError("No first objective conjugacy crossing in the declared q search.")
    index = int(crossings[0]); bracket = (q_grid[index], q_grid[index+1])

    @jax.jit
    def refine_root(low, high):
        def step(_, endpoints):
            lo, hi = endpoints
            mid = .5*(lo+hi)
            return jnp.where(b(mid) > 0, mid, lo), jnp.where(b(mid) > 0, hi, mid)
        lo, hi = jax.lax.fori_loop(0, 56, step, (low, high))
        return .5*(lo+hi)

    q_objective = float(refine_root(*bracket))
    objective_matrix = np.asarray(objective_image(q_objective))
    if abs(objective_matrix[0, 1]) > 1e-12:
        raise RuntimeError("Objective conjugacy failed its forward check.")
    objective_mag = abs(objective_matrix[0, 0])
    length = jnp.array([lens.length_m for lens in projectors])
    entrance = jnp.array([lens.entrance_z for lens in projectors])
    exit_ = jnp.array([lens.exit_z for lens in projectors])
    signs = jnp.array([np.sign(lens.q_bias) or 1. for lens in projectors])

    @jax.jit
    def relay(fraction):
        planes = jnp.concatenate((jnp.array([first_image]),
            exit_[:-1]+fraction*(entrance[1:]-exit_[:-1]), jnp.array([column.detector_z_m])))
        a, b_gap = entrance-planes[:-1], planes[1:]-exit_
        q = first_conjugate_strength(a, b_gap, length)
        stage_a = jnp.cos(q)-b_gap*q/length*jnp.sin(q)
        return objective_mag*jnp.prod(jnp.abs(stage_a)), signs*q, planes

    achievable = np.array([float(relay(t)[0]) for t in bounds])
    if np.any(targets < min(achievable)) or np.any(targets > max(achievable)):
        raise ValueError(f"Targets outside this relay family's range {achievable}.")
    fractions, phases, planes = [], [], []
    current = np.array([np.mean(bounds)])
    for target in targets:
        residual = lambda value: jnp.array([jnp.log(relay(value[0])[0]/target)])
        fit = least_squares(residual, current, bounds=(bounds[:1], bounds[1:]), x_scale=[.1],
                            ftol=0., xtol=0., gtol=1e-12, max_nfev=100)
        if np.max(np.abs(np.asarray(residual(fit.x)))) > 1e-9:
            raise RuntimeError("Shared-objective projector-relay initialization failed.")
        current = np.asarray(fit.x)
        _, phase, image_planes = relay(current[0])
        fractions.append(current[0]); phases.append(np.asarray(phase)); planes.append(np.asarray(image_planes))
    phase_biases = dict(zip(PROJECTOR_NAMES, phases[0]))
    objective_p = np.asarray(initial_objective).copy(); objective_p[0] = objective_sign*q_objective
    column = replace(column, objective_initial=tuple(objective_p), lenses=tuple(
        replace(lens, q_bias=phase_biases.get(lens.name, lens.q_bias)) for lens in column.lenses))

    def controls_for_phases(q):
        controls = np.array([next(lens.reference_dac for lens in column.lenses
                                  if lens.channel_index == i) for i in range(7)])
        for lens, value in zip(column.lenses[3:], q):
            if lens.q_slope_per_dac == 0:
                raise ValueError("Each projector needs a nonzero provisional DAC slope.")
            controls[lens.channel_index] += (value-lens.q_bias)/lens.q_slope_per_dac
        return controls

    controls = np.stack([controls_for_phases(q) for q in phases])
    connection = np.linspace(fractions[0], fractions[-1], int(continuation_points))
    connected = np.stack([controls_for_phases(np.asarray(relay(t)[1])) for t in connection])
    p = column.nominal_parameters
    matrix_fun = jax.jit(jax.vmap(lambda u: column.matrix(p, u, segment="sample_to_detector")))
    matrices = np.asarray(matrix_fun(controls))
    metrics = np.asarray(jax.vmap(shared_imaging_metrics)(matrices))
    connected_metrics = np.asarray(jax.vmap(shared_imaging_metrics)(matrix_fun(connected)))
    if (not np.all(np.isfinite(metrics)) or np.max(np.abs(metrics[:, 0]/targets-1)) > 1e-9
            or np.max(np.abs(metrics[:, 2:])) > 1e-12):
        raise RuntimeError("Single-ray shared-field map fails magnification or full B-block conjugacy.")
    return SharedImagingInitialization(column, p, targets, controls, matrices, metrics,
        np.asarray(fractions), np.asarray(planes), connected, connected_metrics, bracket)


def shared_projection_states(initialization, *, nominal_index=0, control_half_width=.2,
                             steps_per_side=4):
    """Connected IL1/PL1 projection changes; preserve all mechanical A/B data."""
    if not np.isfinite(control_half_width) or control_half_width <= 0:
        raise ValueError("A positive finite synthetic control half-width is required.")
    if isinstance(steps_per_side, bool) or int(steps_per_side) != steps_per_side or steps_per_side < 1:
        raise ValueError("steps_per_side must be a positive integer.")
    column, parameters = initialization.column, initialization.parameters
    nominal = initialization.nominal_controls[nominal_index]
    i1, ip = column.channel_names.index("IL1"), column.channel_names.index("PL1")
    target_mag = initialization.magnifications[nominal_index]
    evaluate = jax.jit(lambda u: shared_imaging_metrics(column.matrix(parameters, u, segment="sample_to_detector")))
    paths, endpoints = [], []
    for sign in (-1., 1.):
        current, path = np.array([nominal[ip]]), []
        for delta in np.linspace(0., sign*control_half_width, int(steps_per_side)+1):
            base = jnp.asarray(nominal).at[i1].add(delta)
            residual = lambda v: jnp.array([jnp.log(evaluate(base.at[ip].set(v[0]))[0]/target_mag)])
            fit = least_squares(residual, current, x_scale=[control_half_width],
                bounds=([nominal[ip]-control_half_width], [nominal[ip]+control_half_width]),
                ftol=0., xtol=0., gtol=1e-12, max_nfev=100)
            current = np.asarray(fit.x)
            control = np.asarray(base.at[ip].set(current[0]))
            metrics = np.asarray(evaluate(control))
            valid = bool(np.all(np.isfinite(metrics)) and abs(np.log(metrics[0]/target_mag)) < 1e-9)
            path.append(dict(controls=control.tolist(), metrics=metrics.tolist(),
                             constraint_satisfied=valid, optimizer_success=bool(fit.success)))
            if not valid:
                break
        paths.append(path)
        endpoints.append(np.asarray(path[-1]["controls"]))
    controls = np.stack((endpoints[0], nominal, endpoints[1]))
    matrices = np.asarray(jax.jit(jax.vmap(lambda u: column.matrix(parameters, u,
                                         segment="sample_to_detector")))(controls))
    return dict(controls=controls.tolist(), matrices=matrices.tolist(),
        metrics=np.asarray(jax.vmap(shared_imaging_metrics)(matrices)).tolist(), paths=paths,
        all_paths_established=all(path[-1]["constraint_satisfied"] for path in paths),
        global_diversity_bound_proved=False, hardware_commands_admitted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slices-per-side", type=int, default=256)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "gpu":
        raise RuntimeError("The shared-objective forward demonstration requires a GPU.")
    root = Path(__file__).resolve().parents[2]
    config = root/"examples/lens_inversion/neoarm_inversion_config.toml"
    diagram = root/"examples/microscope_models/measure_microscope_geometry.ipynb"
    base = column_with_geometry_anchors(hard_edge_column_from_config(tomllib.loads(config.read_text())), diagram)
    field = SharedObjectiveField.from_window(-.05, 0., .05, n_slices_per_side=args.slices_per_side)
    result = initialize_shared_imaging(base, objective=field)
    sources = [Path(__file__), root/"src/temgym_core/immersed_objective.py", config, diagram,
               root/"src/temgym_core/microscope_inversion.py", root/"src/temgym_core/jax_least_squares.py"]
    report = dict(schema="neoarm-shared-objective-initialization-v1", backend=jax.default_backend(),
        parameter_names=result.column.parameter_names, parameters=result.parameters.tolist(),
        magnifications=result.magnifications.tolist(), nominal_controls=result.nominal_controls.tolist(),
        nominal_matrices=result.nominal_matrices.tolist(), nominal_metrics=result.nominal_metrics.tolist(),
        connected_controls=result.connected_controls.tolist(), connected_metrics=result.connected_metrics.tolist(),
        objective_first_crossing_bracket=result.objective_root_bracket,
        objective_window_m=[-.05, .05], slices_per_side=args.slices_per_side,
        projection_states=[shared_projection_states(result, nominal_index=i) for i in range(len(result.magnifications))],
        source_hashes={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        objective_controls_fixed=True, objective_field_measured=False, physical_lens_recovery_proved=False,
        branch_uniqueness_proved=False, hardware_commands_admitted=False,
        assumptions=["One globally normalized asymmetric objective field on finite support.",
            "Sample input rays are mechanical; Gaussian phase transport is canonical.",
            "Approximate measured centres with provisional hard-edge thicknesses and synthetic DAC laws.",
            "Field-window and slice refinement are separate from conjugate initialization."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print("Shared objective:", result.parameters[-4:], flush=True)
    print("40k/80k mechanical metrics:", result.nominal_metrics, flush=True)


if __name__ == "__main__":
    main()
