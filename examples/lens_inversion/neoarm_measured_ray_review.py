"""Publication ray diagnostics for the diagram-geometry synthetic column.

The illustrated Gaussian is defined at the sample and back-propagated. It is
not a fit or independent calibration of the mixed condenser source beam.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from temgym_core.components import ElectromagneticLens
from temgym_core.constants import compute_Rc_from_voltage, energy2wavelength
from temgym_core.microscope_inversion import hard_edge_solenoid_matrix
from temgym_core.neoarm_calibration import hard_edge_column_from_config
from temgym_core.plotting import (ColumnElement, ColumnSection, basis_trace_events,
    export_ray_figure, plot_neoarm_ray_diagram, plot_neoarm_ray_small_multiples)
from neoarm_imaging_recipe import column_with_geometry_anchors, single_ray_transfer
from neoarm_measured_initialization import initialize_measured_imaging
from neoarm_workflow import column_basis_trace, sample_basis_at_source


def exact_corotating_basis(column, parameters, controls, source_basis, frame_rotation_offset):
    """Return an exact scalar-z basis evaluator for event verification."""
    p, u, source_basis = map(jnp.asarray, (parameters, controls, source_basis))
    decoded = column._decoded_lenses(p)
    def evaluate(z):
        state, rotation, cursor = source_basis, jnp.asarray(0.), p[0]
        for lens, centre, length, bias, slope in decoded:
            entrance, exit_ = centre-length/2, centre+length/2
            distance = jnp.clip(z-cursor, 0., entrance-cursor)
            state = state.at[0].add(distance*state[1]).at[2].add(distance*state[3])
            fraction = jnp.clip((z-entrance)/length, 0., 1.)
            q = bias if lens.channel_index is None else bias+slope*(u[lens.channel_index]-lens.reference_dac)
            # Keep both branches finite when z precedes a lens. Multiplication
            # by an explicitly selected identity gives exactly zero body drift.
            matrix = hard_edge_solenoid_matrix(q*fraction, length*jnp.where(fraction > 0, fraction, 1.))
            state = jnp.where(fraction > 0, matrix, jnp.eye(4)) @ state
            rotation -= q*fraction
            cursor = exit_
        distance = jnp.clip(z-cursor, 0., p[1]-cursor)
        state = state.at[0].add(distance*state[1]).at[2].add(distance*state[3])
        axis = jnp.array([jnp.cos(rotation+frame_rotation_offset), jnp.sin(rotation+frame_rotation_offset)])
        return axis @ state[jnp.array([0, 2])]
    return jax.jit(evaluate)


def verified_basis_trace(column, parameters, controls, position_scale, angle_scale, *, label):
    """Use the existing tracer and verify/refine its interior A/B roots."""
    source, offset = sample_basis_at_source(column, parameters, controls, position_scale, angle_scale)
    trace = column_basis_trace(column, parameters, controls, position_scale, angle_scale,
                              source_basis=source, frame_rotation_offset=offset, label=label)
    exact = exact_corotating_basis(column, parameters, controls, source, offset)
    all_values = np.asarray(jax.jit(jax.vmap(exact))(jnp.asarray(trace.z)))
    error = float(np.max(np.abs(all_values-np.column_stack((trace.position, trace.angle)))))
    if error > 2e-11*max(float(np.max(trace.radius)), 1e-9):
        raise RuntimeError("Dense ray tracer disagrees with exact finite-body evaluation.")
    event_records, roots = [], []
    for family, channel in (("crossovers", 0), ("conjugates", 1)):
        @jax.jit
        def refine(bounds):
            def step(_, interval):
                left, right = interval
                middle = .5*(left+right)
                same_sign = exact(left)[channel]*exact(middle)[channel] > 0
                return jnp.where(same_sign, middle, left), jnp.where(same_sign, right, middle)
            a, b = jax.lax.fori_loop(0, 50, step, bounds)
            return .5*(a+b)
        events = basis_trace_events(trace)[family]
        for event in events:
            index = np.searchsorted(trace.z, event.z)
            lo = trace.z[max(0, index-1)]
            hi = trace.z[min(len(trace.z)-1, index+1)]
            endpoint = event.z in (trace.z[0], trace.z[-1])
            bracketed = float(exact(lo)[channel])*float(exact(hi)[channel]) <= 0
            root = float(refine((jnp.asarray(lo), jnp.asarray(hi)))) if bracketed and not endpoint else float(event.z)
            value = float(exact(root)[channel])
            roots.append(root)
            event_records.append({"kind": family, "z_m": root, "original_z_m": float(event.z),
                                  "position_residual_m": value, "bracketed": bracketed})
    # Adding exact event samples lets the existing renderer place its root
    # markers accurately without any change to public plotting APIs.
    z = np.unique(np.r_[trace.z, roots])
    values = np.asarray(jax.jit(jax.vmap(exact))(jnp.asarray(z)))
    trace = replace(trace, z=z, centroid=np.zeros(z.size), position=values[:, 0],
                    angle=values[:, 1], radius=np.hypot(values[:, 0], values[:, 1]))
    sample = np.asarray(exact(column.sample_z_m))
    expected_sample = np.array([position_scale, 0.])
    if not np.allclose(sample, expected_sample, atol=1e-14, rtol=1e-9):
        raise RuntimeError("Back-propagated sample basis is inconsistent.")
    matrix = np.asarray(single_ray_transfer(column, parameters, controls))
    basis = np.zeros((4, 2)); basis[0, 0], basis[1, 1] = position_scale, angle_scale
    rotation = -sum(float(bias + (0. if lens.channel_index is None else slope*(controls[lens.channel_index]-lens.reference_dac)))
                    for lens, _, _, bias, slope in column._decoded_lenses(jnp.asarray(parameters)))
    axis = np.array([np.cos(rotation+offset), np.sin(rotation+offset)])
    expected_detector = axis @ (matrix @ basis)[[0, 2]]
    detector_error = np.asarray([trace.position[-1], trace.angle[-1]])-expected_detector
    if np.max(np.abs(detector_error)) > 1e-11*max(np.max(np.abs(expected_detector)), 1e-9):
        raise RuntimeError("Detector basis disagrees with the differentiated single-ray map.")
    derivative = jax.jit(lambda value: jnp.dot(exact(value), jax.jacfwd(exact)(value)))
    @jax.jit
    def refine_waist(bounds):
        def step(_, interval):
            lo, hi = interval
            middle = .5*(lo+hi)
            descending = derivative(middle) < 0
            return jnp.where(descending, middle, lo), jnp.where(descending, hi, middle)
        lo, hi = jax.lax.fori_loop(0, 50, step, bounds)
        return .5*(lo+hi)
    waists = []
    for event in basis_trace_events(trace)["waists"]:
        exact_radius = float(jnp.linalg.norm(exact(event.z)))
        index = np.searchsorted(z, event.z)
        lo, hi = z[max(0, index-2)], z[min(z.size-1, index+2)]
        # A refined A/B root can sit almost exactly on an existing sample and
        # create several neighbouring floating-point z values. Do not let
        # that collapse the physical stationary-point bracket to roundoff.
        lo, hi = max(z[0], min(lo, event.z-1e-6)), min(z[-1], max(hi, event.z+1e-6))
        stationary_bracketed = float(derivative(lo)) <= 0 <= float(derivative(hi))
        exact_z = float(refine_waist((jnp.asarray(lo), jnp.asarray(hi)))) if stationary_bracketed else None
        waists.append({"z_m": float(event.z), "rendered_radius_m": float(event.radius),
                       "exact_radius_m": exact_radius, "radius_error_m": exact_radius-float(event.radius),
                       "stationary_bracketed": stationary_bracketed, "exact_stationary_z_m": exact_z,
                       "annotation_radius_m": None if exact_z is None else float(jnp.linalg.norm(exact(exact_z))),
                       "stationary_derivative_m": None if exact_z is None else float(derivative(exact_z)),
                       "position_error_m": None if exact_z is None else float(event.z)-exact_z})
    return trace, {"dense_trace_max_error_m": error, "source_basis": source.tolist(),
                   "sample_basis_error_m": (sample-expected_sample).tolist(),
                   "detector_basis_error_m": detector_error.tolist(), "events": event_records,
                   "waists": waists, "frame_rotation_offset_rad": float(offset)}


def annotate_verified_events(axes, sections, evidence, *, gaussian):
    """Place exact finite-body event roots using the established label layout.

    The general plotting helper's piecewise-linear waist finder is appropriate
    to drifts. Here we annotate the independently verified stationary roots
    inside finite lens bodies as well, without changing that public API.
    """
    from matplotlib.transforms import blended_transform_factory
    from temgym_core.plotting import _spread_label_boxes, _format_engineering_length
    candidates = [{"z": row["z_m"], "text": "A=0 crossover" if row["kind"] == "crossovers" else "B=0 conjugate",
                   "marker": "o" if row["kind"] == "crossovers" else "s",
                   "color": "#0072B2" if row["kind"] == "crossovers" else "#D55E00"}
                  for row in evidence["events"]]
    if gaussian:
        seen = []
        for row in evidence["waists"]:
            if not row["stationary_bracketed"]:
                continue
            z = row["exact_stationary_z_m"]
            if any(abs(z-other) < 1e-9 for other in seen):
                continue
            seen.append(z)
            candidates.append({"z": z, "text": f"waist w={_format_engineering_length(row['annotation_radius_m'])}",
                               "marker": "D", "color": "#7B52AB"})
    for index, section in enumerate(sections, 1):
        axis = axes[f"detail_{index}"]
        visible = sorted((row for row in candidates if section.z_min <= row["z"] <= section.z_max), key=lambda row: row["z"])
        groups = []
        for row in visible:
            if groups and abs(row["z"]-np.mean([entry["z"] for entry in groups[-1]])) < .012*(section.z_max-section.z_min):
                groups[-1].append(row)
            else:
                groups.append([row])
        labels = [" · ".join(dict.fromkeys(row["text"] for row in group)) for group in groups]
        positions = [float(np.mean([row["z"] for row in group])) for group in groups]
        label_z = _spread_label_boxes(positions, [1]*len(groups), section.z_min, section.z_max,
                                      line_height_fraction=.047, gap_fraction=.012)
        transform = blended_transform_factory(axis.transAxes, axis.transData)
        for group, label, at in zip(groups, labels, label_z):
            for row in group:
                axis.plot([0.], [row["z"]], marker=row["marker"], markersize=4.5,
                          markerfacecolor="white", markeredgecolor=row["color"], linestyle="none", zorder=12)
            axis.annotate(label, xy=(0., float(np.mean([row["z"] for row in group]))),
                xytext=(.405, at), textcoords=transform, ha="right", va="center", fontsize=6.6,
                color="#374151", bbox={"facecolor": "white", "edgecolor": "#E5E7EB", "lw": .35, "alpha": .9, "pad": .7},
                arrowprops={"arrowstyle": "-", "color": "#9CA3AF", "lw": .55}, zorder=13)


def label_overlap_audit(figure):
    """Check actual text boxes, excluding annotation leader-line extents."""
    from matplotlib.text import Text
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    collisions = []
    for index, axis in enumerate(figure.axes):
        labels = [text for text in axis.texts if text.get_visible() and text.get_text()]
        boxes = [Text.get_window_extent(text, renderer) for text in labels]
        for i in range(len(labels)):
            for j in range(i+1, len(labels)):
                if boxes[i].overlaps(boxes[j]):
                    collisions.append({"axis": index, "first": labels[i].get_text(), "second": labels[j].get_text()})
    return collisions


def physical_plot_geometry(column, parameters, controls):
    """Reuse the finite-body glyph convention of workflow.draw_column_rays."""
    rc = float(compute_Rc_from_voltage(200000.))
    components, labels = [], []
    for lens, centre, length, bias, slope in column._decoded_lenses(jnp.asarray(parameters)):
        q = float(bias if lens.channel_index is None else bias+slope*(controls[lens.channel_index]-lens.reference_dac))
        excitation = -q/rc
        components.append(ElectromagneticLens(z=float(centre-length/2), turns=1., current=excitation,
            Gc=1., Tc=float(length)/abs(excitation)))
        labels.append(lens.name)
    extras = (ColumnElement(column.sample_z_m, "MAGICAL · 23 nm multilayer", "sample"),
              ColumnElement(float(parameters[1]), "Medipix 256 × 256", "detector"))
    sections = (ColumnSection("Condenser · sample-conditioned basis", float(parameters[0]), -.03),
                ColumnSection("Objective / sample · approximate physical positions", -.025, .025),
                ColumnSection("Projector · first excitation branch", .15, float(parameters[1])))
    return components, labels, extras, sections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "gpu":
        raise RuntimeError("Accepted ray forward verification requires the GPU.")
    import tomllib
    root = Path(__file__).resolve().parents[2]
    config_path = root / "examples/lens_inversion/neoarm_inversion_config.toml"
    notebook_path = root / "examples/microscope_models/measure_microscope_geometry.ipynb"
    column = column_with_geometry_anchors(hard_edge_column_from_config(tomllib.loads(config_path.read_text())), notebook_path)
    initialization = initialize_measured_imaging(column)
    column, p = initialization.column, initialization.parameters
    # Read the independently selected candidate as data, preserving its
    # provenance rather than silently inventing a source or condenser setting.
    audit_path = root / "examples/generated/neoarm_inversion/imaging_recipe/measured_information_audit.json"
    audit_bytes = audit_path.read_bytes()
    audit = json.loads(audit_bytes)
    initializer_hash = hashlib.sha256((Path(__file__).parent / "neoarm_measured_initialization.py").read_bytes()).hexdigest()
    diagram_hash = hashlib.sha256(notebook_path.read_bytes()).hexdigest()
    initializer_report_path = root / "examples/generated/neoarm_inversion/imaging_recipe/measured_initialization.json"
    initializer_report_hash = hashlib.sha256(initializer_report_path.read_bytes()).hexdigest()
    if (audit["initializer_source_sha256"] != initializer_hash or audit["diagram_sha256"] != diagram_hash
            or audit["initializer_report_sha256"] != initializer_report_hash):
        raise ValueError("Condenser candidate provenance does not match this initializer/diagram.")
    condenser_q = np.asarray(audit["selected_q"][0], float)
    condenser_controls = np.asarray(audit["selected_controls"][0][:3], float)
    if condenser_q.shape != (3,) or condenser_controls.shape != (3,) or not np.all(np.isfinite(condenser_controls)):
        raise ValueError("Condenser audit must provide three finite candidate controls.")
    metadata = {"title": "NEOARM measured-diagram geometry · conditional two-ray solution",
        "backend": jax.default_backend(), "parameters": p.tolist(), "parameter_names": column.parameter_names,
        "geometry_notebook_sha256": diagram_hash,
        "initialization_source_sha256": initializer_hash,
        "initialization_report_sha256": initializer_report_hash,
        "illumination_audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
        "illumination_candidate_index": int(audit["selected_indices"][0]),
        "ray_review_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "condenser_q": condenser_q.tolist(), "sample_waist_m": 60e-9, "voltage_v": 200000.,
        "calibrated_position_scale_m": 5e-9, "calibrated_angle_scale_rad": 1e-6,
        "conditioning": "Gaussian waist defined at sample, back-propagated; not the mixed source calibration",
        "positions_approximate": True, "lengths_and_fields_provisional": True, "hardware_commands_admitted": False,
        "objective_model": "Independent hard-edge pre/post surrogate; a shared physical objective field is not enforced",
        "physical_neoarm_objective_validated": False,
        "corrector_disabled": True, "objective_controls_fixed": True}
    traces, geometries, records = {}, {}, {}
    for index, mag in enumerate(initialization.magnifications):
        control = initialization.nominal_controls[index].copy(); control[:3] = condenser_controls
        label = f"{mag/1000:g}k · sample-defined 60 nm waist"
        wavelength = float(energy2wavelength(200000.))
        gaussian, gaussian_checks = verified_basis_trace(column, p, control, 60e-9, wavelength/(np.pi*60e-9), label=label)
        calibrated, calibrated_checks = verified_basis_trace(column, p, control, 5e-9, 1e-6,
                                                             label=f"{mag/1000:g}k · 5 nm / 1 µrad inputs")
        components, labels, extras, sections = physical_plot_geometry(column, p, control)
        traces[f"{mag/1000:g}k"] = gaussian; geometries[f"{mag/1000:g}k"] = components
        records[f"{mag/1000:g}k"] = {"controls": control.tolist(), "gaussian": gaussian_checks,
                                     "calibrated": calibrated_checks}
        for kind, trace, semantics in (("gaussian", gaussian, "gaussian"), ("calibrated", calibrated, "calibrated_response")):
            figure, axes = plot_neoarm_ray_diagram(components, trace, component_labels=labels,
                component_roles=["lens"]*len(labels), extra_elements=extras, sections=sections,
                trace_semantics=semantics, annotate_events=False,
                title=f"NEOARM {mag/1000:g}k · measured-diagram geometry", figsize=(15., 11.))
            annotate_verified_events(axes, sections, gaussian_checks if kind == "gaussian" else calibrated_checks,
                                     gaussian=kind == "gaussian")
            figure.text(.52, .945, "Approximate geometry; independent pre/post-field surrogate. Shared objective field is not enforced.",
                        ha="center", fontsize=8, color="#59616D")
            overlaps = label_overlap_audit(figure)
            if overlaps:
                raise RuntimeError(f"Publication labels overlap: {overlaps}")
            export_ray_figure(figure, args.output_dir / f"measured_{int(mag/1000)}k_{kind}",
                             metadata={**metadata, "state": records[f"{mag/1000:g}k"], "basis_semantics": semantics,
                                       "label_overlap_audit": overlaps})
            plt.close(figure)
        print(f"Exported verified {mag/1000:g}k Gaussian and calibrated two-ray diagrams", flush=True)
    figure, _ = plot_neoarm_ray_small_multiples(geometries, traces, component_labels=labels,
        component_roles=["lens"]*len(labels), extra_elements=extras, sections=sections,
        title="NEOARM 40k / 80k · measured geometry, identical panel scales", figsize=(13., 13.))
    figure.text(.5, .96, "60 nm sample-conditioned waist · independent pre/post-field surrogate · shared objective field not enforced", ha="center", fontsize=8)
    overlaps = label_overlap_audit(figure)
    if overlaps:
        raise RuntimeError(f"Comparison labels overlap: {overlaps}")
    export_ray_figure(figure, args.output_dir / "measured_nominal_comparison", metadata={**metadata, "states": records,
                                                                                   "label_overlap_audit": overlaps})
    plt.close(figure)


if __name__ == "__main__":
    main()
