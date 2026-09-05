"""Low-excitation synthetic initialization on the microscope diagram geometry.

The diagram supplies approximate *positions*, not measured fields or DAC laws.
This construction puts each finite lens on its first conjugate root, adjusts
the previously provisional objective postfield once, and then holds it fixed.
It is an optical feasibility initializer, never a calibrated hardware recipe.
"""

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from temgym_core.jax_least_squares import least_squares
from temgym_core.neoarm_calibration import hard_edge_column_from_config
from neoarm_imaging_recipe import column_with_geometry_anchors, imaging_metrics, single_ray_transfer


def conjugacy_b(q, object_gap, image_gap, length):
    """B of drift(image_gap) @ finite_lens(q, length) @ drift(object_gap)."""
    return ((object_gap + image_gap) * jnp.cos(q)
            + length * jnp.sinc(q / jnp.pi)
            - object_gap * image_gap * q / length * jnp.sin(q))


@jax.custom_jvp
def first_conjugate_strength(object_gap, image_gap, length):
    """First positive B=0 root, for nonnegative gaps and positive length.

    Bracketed evaluation stays in (0, pi). Its JVP uses implicit derivatives,
    rather than differentiating binary branch decisions in the root search.
    """
    a, b, length = jnp.broadcast_arrays(object_gap, image_gap, length)
    def bisect(_, bounds):
        lo, hi = bounds
        mid = .5 * (lo + hi)
        positive = conjugacy_b(mid, a, b, length) > 0
        return jnp.where(positive, mid, lo), jnp.where(positive, hi, mid)
    lo, hi = jax.lax.fori_loop(0, 60, bisect, (jnp.zeros_like(a), jnp.full_like(a, jnp.pi)))
    return .5 * (lo + hi)


@first_conjugate_strength.defjvp
def _first_conjugate_strength_jvp(primals, tangents):
    a, b, length = primals
    da, db, dl = tangents
    q = first_conjugate_strength(a, b, length)
    sine, cosine = jnp.sin(q), jnp.cos(q)
    f_q = (-(a + b) * sine + length * (q * cosine - sine) / q**2
           - a * b / length * (sine + q * cosine))
    change = ((cosine - b * q / length * sine) * da
              + (cosine - a * q / length * sine) * db
              + (sine / q + a * b * q / length**2 * sine) * dl)
    return q, -change / f_q


@dataclass(frozen=True)
class MeasuredImagingInitialization:
    column: object
    parameters: np.ndarray
    magnifications: np.ndarray
    nominal_controls: np.ndarray
    relay_fractions: np.ndarray
    relay_image_planes_m: np.ndarray
    nominal_matrices: np.ndarray
    nominal_metrics: np.ndarray
    connected_controls: np.ndarray
    connected_metrics: np.ndarray
    initialization_attempts: tuple[dict, ...]
    original_downstream_biases: dict


def _relay_function(column, objective_image_gap_m):
    lenses = tuple(lens for lens in column.lenses if lens.entrance_z >= column.sample_z_m)
    if tuple(lens.name for lens in lenses) != ("Obj-post", "IL1", "IL2", "IL3", "PL1"):
        raise ValueError("Expected fixed Obj-post followed by IL1, IL2, IL3, PL1.")
    length = jnp.array([lens.length_m for lens in lenses])
    entrance = jnp.array([lens.entrance_z for lens in lenses])
    exit_ = jnp.array([lens.exit_z for lens in lenses])
    first_image = float(entrance[1]) - objective_image_gap_m
    if not float(exit_[0]) < first_image < float(entrance[1]):
        raise ValueError("The objective image must lie in the Obj-post/IL1 free drift.")
    signs = jnp.array([-1. if lens.q_bias < 0 else 1. for lens in lenses])

    @jax.jit
    def relay(fraction):
        planes = jnp.concatenate((jnp.array([column.sample_z_m, first_image]),
            exit_[1:-1] + fraction * (entrance[2:] - exit_[1:-1]),
            jnp.array([column.detector_z_m])))
        a, b = entrance - planes[:-1], planes[1:] - exit_
        q = first_conjugate_strength(a, b, length)
        stage_a = jnp.cos(q) - b * q / length * jnp.sin(q)
        return jnp.prod(jnp.abs(stage_a)), q * signs, planes
    return lenses, relay


def initialize_measured_imaging(column, *, magnifications=(40000., 80000.),
                                objective_image_gap_m=1e-3,
                                relay_fraction_bounds=(.5, .995),
                                continuation_points=9):
    """Construct connected conjugate states, rebasing synthetic controls at 40k.

    Three intermediate projector images share a fraction of their individual
    free gaps. Tuning that fraction changes magnification while every section
    remains conjugate. The objective image plane is invariant. Existing lens
    thicknesses remain provisional; no real field/current measurement is used.
    """
    targets = np.asarray(magnifications, float)
    bounds = np.asarray(relay_fraction_bounds, float)
    if (targets.ndim != 1 or not 1 <= targets.size <= 2
            or not np.all(np.isfinite(targets)) or np.any(targets <= 0)
            or np.any(np.diff(targets) <= 0)):
        raise ValueError("Choose one or two positive finite increasing magnifications.")
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or not 0 < bounds[0] < bounds[1] < 1:
        raise ValueError("Relay fractions must have finite bounds strictly inside (0, 1).")
    if not np.isfinite(objective_image_gap_m) or objective_image_gap_m <= 0:
        raise ValueError("objective_image_gap_m must be finite and positive.")
    if isinstance(continuation_points, bool) or int(continuation_points) != continuation_points or continuation_points < 2:
        raise ValueError("continuation_points must be an integer >= 2.")
    if np.any(np.asarray(column.geometry_clearances(column.nominal_parameters)) < 0):
        raise ValueError("Column geometry must be physically ordered.")
    lenses, relay = _relay_function(column, objective_image_gap_m)
    achievable = np.array([float(relay(value)[0]) for value in bounds])
    if np.any(targets < achievable[0]) or np.any(targets > achievable[1]):
        raise ValueError(f"Target outside this relay family's endpoint range {achievable}; not an impossibility proof.")
    fractions, qs, planes, attempts = [], [], [], []
    current = np.array([float(np.mean(bounds))])
    for target in targets:
        residual = lambda t: jnp.array([jnp.log(relay(t[0])[0] / target)])
        fit = least_squares(residual, current, bounds=(bounds[:1], bounds[1:]),
            x_scale=[.1], ftol=0., xtol=0., gtol=1e-12, max_nfev=120)
        error = float(np.asarray(residual(fit.x))[0])
        valid = np.isfinite(error) and abs(error) <= 1e-9
        attempts.append({"target_magnification": float(target), "optimizer_status": fit.status,
                         "optimizer_success": fit.success, "log_magnification_error": error,
                         "constraint_satisfied": bool(valid), "nfev": fit.nfev})
        if not valid:
            raise RuntimeError(f"Relay initialization failed: {attempts[-1]}")
        current = np.asarray(fit.x)
        _, q, image_planes = relay(current[0])
        fractions.append(current[0]); qs.append(np.asarray(q)); planes.append(np.asarray(image_planes))
    qs = np.asarray(qs)
    initial_biases = {lens.name: lens.q_bias for lens in lenses}
    updated_biases = dict(zip((lens.name for lens in lenses), qs[0]))
    initialized = replace(column, lenses=tuple(replace(lens, q_bias=updated_biases.get(lens.name, lens.q_bias))
                                               for lens in column.lenses))
    p = initialized.nominal_parameters
    def controls_from_q(values):
        controls = np.array([next(lens.reference_dac for lens in initialized.lenses
                                  if lens.channel_index == i) for i in range(len(initialized.channel_names))])
        for lens, q in zip(lenses, values):
            if lens.channel_index is not None:
                if lens.q_slope_per_dac == 0:
                    raise ValueError(f"{lens.name} requires a nonzero provisional control slope.")
                controls[lens.channel_index] = lens.reference_dac + (q - updated_biases[lens.name]) / lens.q_slope_per_dac
        return controls
    controls = np.stack([controls_from_q(q) for q in qs])
    connection = np.linspace(fractions[0], fractions[-1], int(continuation_points))
    connected = np.stack([controls_from_q(np.asarray(relay(t)[1])) for t in connection])
    matrix_fun = jax.jit(jax.vmap(lambda u: single_ray_transfer(initialized, p, u)))
    matrices = np.asarray(matrix_fun(controls))
    metrics = np.asarray(jax.vmap(imaging_metrics)(matrices))
    connected_metrics = np.asarray(jax.vmap(imaging_metrics)(matrix_fun(connected)))
    if not (np.all(np.isfinite(metrics)) and np.max(np.abs(metrics[:, 0] / targets - 1.)) < 1e-9
            and np.max(np.abs(metrics[:, 1])) < 1e-12):
        raise RuntimeError("Full single-ray model failed independent relay constraints.")
    return MeasuredImagingInitialization(initialized, p, targets, controls, np.asarray(fractions),
        np.asarray(planes), matrices, metrics, connected, connected_metrics, tuple(attempts), initial_biases)


def attempt_bounded_diversity(initialization, nominal_index=0, *, target_defocus_m=(-100e-6, 100e-6),
                              max_q=np.pi - .05, continuation_steps=8, max_nfev=300):
    """Try all four projectors, retaining every failed local continuation.

    Bounds constrain absolute excitation phases, not hardware DAC safety.
    Failure means this local path was not established; it excludes no other
    excitation branch, objective setting, thickness model, or control route.
    """
    c, p = initialization.column, initialization.parameters
    nominal = initialization.nominal_controls[nominal_index]
    target_mag = initialization.magnifications[nominal_index]
    lenses = tuple(lens for lens in c.lenses if lens.name in {"IL1", "IL2", "IL3", "PL1"})
    indices = jnp.array([lens.channel_index for lens in lenses])
    slopes = jnp.array([lens.q_slope_per_dac for lens in lenses])
    biases = jnp.array([lens.q_bias for lens in lenses])
    references = jnp.array([lens.reference_dac for lens in lenses])
    signs = jnp.sign(biases)
    initial_q = np.asarray(signs * (biases + slopes * (nominal[np.asarray(indices)] - references)))
    if not np.isfinite(max_q) or max_q <= np.max(initial_q):
        raise ValueError("max_q must exceed all nominal absolute projector strengths.")
    targets = np.asarray(target_defocus_m, float)
    if targets.ndim != 1 or targets.size == 0 or not np.all(np.isfinite(targets)):
        raise ValueError("target_defocus_m must be a nonempty finite vector.")
    if isinstance(continuation_steps, bool) or int(continuation_steps) != continuation_steps or continuation_steps < 1:
        raise ValueError("continuation_steps must be a positive integer.")
    def control(q):
        return jnp.asarray(nominal).at[indices].set(references + (signs*q - biases) / slopes)
    metrics = jax.jit(lambda q: imaging_metrics(single_ray_transfer(c, p, control(q))))
    records = []
    for target in targets:
        current = initial_q.copy()
        scale = max(abs(target), 1e-7)
        path = []
        for step_target in np.linspace(0., target, int(continuation_steps)+1)[1:]:
            residual = lambda q: jnp.array([jnp.log(metrics(q)[0]/target_mag),
                                             (metrics(q)[1]-step_target)/scale])
            fit = least_squares(residual, current, bounds=(.05, max_q), x_scale=.5,
                                ftol=0., xtol=0., gtol=1e-12, max_nfev=max_nfev)
            current = np.asarray(fit.x)
            error = np.asarray(residual(current))
            valid = bool(np.all(np.isfinite(error)) and np.max(np.abs(error)) < 1e-7)
            path.append({"target_defocus_m": float(step_target), "achieved_metrics": np.asarray(metrics(current)).tolist(),
                         "controls": np.asarray(control(current)).tolist(), "absolute_projector_q": current.tolist(),
                         "constraint_error": error.tolist(), "constraint_satisfied": valid,
                         "optimizer_status": fit.status, "optimizer_success": fit.success, "nfev": fit.nfev})
            if not valid:
                break
        records.append({"requested_defocus_m": float(target), "nominal_magnification": float(target_mag),
                        "absolute_q_bounds": [.05, float(max_q)], "all_projector_controls": [lens.name for lens in lenses],
                        "path_established": bool(path[-1]["constraint_satisfied"]), "continuation": path})
    return records


def scan_bounded_imaging_diversity(initialization, nominal_index=0, *, control_half_width=.05,
                                   primary="IL1", compensator="PL1", steps_per_side=4):
    """Measure available signed B/A along a bounded constant-magnification path.

    Sweep one synthetic control and solve one compensating projector control.
    The returned span is verified for this path, not a global maximum over
    controls. Objective fields and both remaining projector controls are fixed.
    """
    allowed = {"IL1", "IL2", "IL3", "PL1"}
    if primary == compensator or primary not in allowed or compensator not in allowed:
        raise ValueError("Two distinct downstream projector channels are required.")
    if not np.isfinite(control_half_width) or control_half_width <= 0:
        raise ValueError("control_half_width must be finite and positive.")
    if isinstance(steps_per_side, bool) or int(steps_per_side) != steps_per_side or steps_per_side < 1:
        raise ValueError("steps_per_side must be a positive integer.")
    c, p = initialization.column, initialization.parameters
    nominal = initialization.nominal_controls[nominal_index]
    target_mag = initialization.magnifications[nominal_index]
    primary_index, compensator_index = c.channel_names.index(primary), c.channel_names.index(compensator)
    evaluate = jax.jit(lambda u: imaging_metrics(single_ray_transfer(c, p, u)))
    paths = []
    for sign in (-1., 1.):
        current = np.array([nominal[compensator_index]])
        points = []
        for delta in np.linspace(0., sign*control_half_width, int(steps_per_side)+1):
            base = jnp.asarray(nominal).at[primary_index].add(delta)
            residual = lambda value: jnp.array([jnp.log(evaluate(base.at[compensator_index].set(value[0]))[0]/target_mag)])
            fit = least_squares(residual, current,
                bounds=([nominal[compensator_index]-control_half_width], [nominal[compensator_index]+control_half_width]),
                x_scale=[control_half_width], ftol=0., xtol=0., gtol=1e-12, max_nfev=100)
            current = np.asarray(fit.x)
            control = np.asarray(base.at[compensator_index].set(current[0]))
            achieved = np.asarray(evaluate(control))
            error = float(np.log(achieved[0]/target_mag))
            valid = bool(np.all(np.isfinite(achieved)) and abs(error) <= 1e-9)
            points.append({"primary_delta": float(delta), "controls": control.tolist(),
                "achieved_metrics": achieved.tolist(), "constraint_satisfied": valid,
                "optimizer_status": fit.status, "optimizer_success": fit.success})
            if not valid:
                break
        paths.append({"direction": int(sign), "endpoint_established": bool(points[-1]["constraint_satisfied"]),
                      "points": points})
    established = [point["achieved_metrics"][1] for path in paths for point in path["points"]
                   if point["constraint_satisfied"]]
    derivative = np.asarray(jax.jacfwd(evaluate)(jnp.asarray(nominal)))
    return {"nominal_magnification": float(target_mag), "primary": primary, "compensator": compensator,
            "control_half_width": float(control_half_width), "paths": paths,
            "verified_path_defocus_range_m": [float(min(established)), float(max(established))],
            "nominal_metric_control_jacobian": derivative.tolist(), "global_maximum_proved": False,
            "hardware_commands_admitted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempt-diversity", action="store_true")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "gpu":
        raise RuntimeError("Accepted measured-geometry forward initialization requires a GPU.")
    import tomllib
    root = Path(__file__).resolve().parents[2]
    config = root / "examples/lens_inversion/neoarm_inversion_config.toml"
    notebook = root / "examples/microscope_models/measure_microscope_geometry.ipynb"
    column = column_with_geometry_anchors(hard_edge_column_from_config(tomllib.loads(config.read_text())), notebook)
    result = initialize_measured_imaging(column)
    report = {"schema": "neoarm-measured-geometry-initialization-v1", "backend": jax.default_backend(),
        "scope": "matched synthetic optical feasibility on approximate measured positions; no inversion or hardware proof",
        "hardware_commands_admitted": False, "recovery_proved": False, "branch_uniqueness_proved": False,
        "lengths_measured": False, "objective_field_measured": False,
        "parameters": result.parameters.tolist(), "parameter_names": result.column.parameter_names,
        "magnifications": result.magnifications.tolist(), "nominal_controls": result.nominal_controls.tolist(),
        "nominal_metrics": result.nominal_metrics.tolist(), "nominal_matrices": result.nominal_matrices.tolist(),
        "relay_fractions": result.relay_fractions.tolist(), "relay_image_planes_m": result.relay_image_planes_m.tolist(),
        "connected_controls": result.connected_controls.tolist(), "connected_metrics": result.connected_metrics.tolist(),
        "initialization_attempts": result.initialization_attempts,
        "original_downstream_biases": result.original_downstream_biases,
        "provisional_lengths_m": {lens.name: lens.length_m for lens in result.column.lenses},
        "assumptions": ["diagram landmarks are approximate physical centres, not measured principal planes",
            "lens lengths and DAC slopes remain provisional synthetic values",
            "objective postfield initialized consistently once, then fixed for every control state",
            "downstream control zero rebased at the first synthetic conjugate state; not an instrument command",
            "earlier failed fixed-old-objective searches and their artifacts are not overwritten"],
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "geometry_notebook_sha256": hashlib.sha256(notebook.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    print("Nominal measured-geometry initialization:", result.nominal_metrics, flush=True)
    if args.attempt_diversity:
        report["diversity_attempts"] = [record for i in range(len(result.magnifications))
            for record in attempt_bounded_diversity(result, i)]
        print("Signed ±100 µm paths established:", [r["path_established"] for r in report["diversity_attempts"]], flush=True)
    report["bounded_imaging_paths"] = [scan_bounded_imaging_diversity(result, i, control_half_width=width)
        for i in range(len(result.magnifications)) for width in (.05, .2)]
    print("Bounded IL1/PL1 B/A spans [m]:", [r["verified_path_defocus_range_m"] for r in report["bounded_imaging_paths"]], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
