"""Design a small imaging-only diversity experiment using the actual column.

This is a computational design, not a NEOARM command generator.  The archived
nominal free-lens coordinates are synthetic and retain their original units.
Only IL2/IL3 change during each local continuation; objective controls stay
fixed.  Finite-defocus states are solved through a differentiated single ray,
not by pretending that a projector-control increment is a specimen z shift.
"""

from dataclasses import dataclass, replace
import argparse
import ast
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from temgym_core.microscope_inversion import hard_edge_solenoid_matrix
from temgym_core.neoarm_calibration import hard_edge_column_from_config


def trace_column_ray(column, parameters, dacs, ray, *, segment="sample_to_detector"):
    """Trace one [x, theta_x, y, theta_y] ray through finite lens bodies."""
    if segment not in {"sample_to_detector", "source_to_sample", "full"}:
        raise ValueError("Unknown column segment.")
    parameters, dacs, ray = map(jnp.asarray, (parameters, dacs, ray))
    if parameters.shape != (column.n_parameters,) or dacs.shape != (len(column.channel_names),):
        raise ValueError("Parameter/control dimensions do not match the column.")
    if ray.shape != (4,):
        raise ValueError("A ray has four coordinates in interleaved order.")
    cursor = column.sample_z_m if segment == "sample_to_detector" else parameters[0]

    def drift(state, distance):
        return state.at[0].add(distance*state[1]).at[2].add(distance*state[3])

    for lens, centre, length, bias, slope in column._decoded_lenses(parameters):
        if segment == "sample_to_detector" and lens.entrance_z < column.sample_z_m:
            continue
        if segment == "source_to_sample" and lens.exit_z > column.sample_z_m:
            continue
        ray = drift(ray, centre-length/2-cursor)
        q = bias if lens.channel_index is None else bias+slope*(dacs[lens.channel_index]-lens.reference_dac)
        ray = hard_edge_solenoid_matrix(q, length) @ ray
        cursor = centre+length/2
    endpoint = column.sample_z_m if segment == "source_to_sample" else parameters[1]
    return drift(ray, endpoint-cursor)


def single_ray_transfer(column, parameters, dacs, *, segment="sample_to_detector"):
    """Differentiate the final single-ray state with respect to its input."""
    return jax.jacfwd(lambda ray: trace_column_ray(
        column, parameters, dacs, ray, segment=segment))(jnp.zeros(4))


def imaging_metrics(matrix):
    """Return positive magnification, B/A [m], and signed A-vector angle.

    B/A is an equivalent object-plane drift for a round first-order map; it
    is neither a physical specimen displacement nor a raw-DAC calibration.
    """
    matrix = jnp.asarray(matrix)
    xy, slopes = jnp.array([0, 2]), jnp.array([1, 3])
    a, b = matrix[jnp.ix_(xy, xy)], matrix[jnp.ix_(xy, slopes)]
    mag = jnp.sqrt(jnp.sum(a*a)/2)
    return jnp.array([mag, jnp.trace(jnp.linalg.solve(a, b))/2,
                      jnp.arctan2(a[1, 0], a[0, 0])])


@dataclass(frozen=True)
class ImagingDiversityDesign:
    controls: np.ndarray
    matrices: np.ndarray
    target_defocus_m: np.ndarray
    achieved_metrics: np.ndarray
    nominal_controls: np.ndarray
    selected_channels: tuple[str, ...]
    continuation: tuple[dict, ...]

    @property
    def parameter_fit_admitted(self):
        # Control feasibility alone supplies no measurement or inversion proof.
        return False


def design_imaging_diversity(column, parameters, nominal_controls, *,
                             defocus_m=(-100e-6, 0., 100e-6),
                             selected_channels=("IL2", "IL3"),
                             control_half_width=.05, continuation_steps=4):
    """Solve bounded connected control paths for fixed-M, signed-B/A images.

    The solver is JAX-native; magnification/defocus derivatives are obtained
    by differentiating the single-ray Jacobian again with respect to DACs.
    Returned continuation points need actual acquired measurements before
    they can support any assertion about the instrument's excitation branch.
    """
    from temgym_core.jax_least_squares import least_squares

    p, nominal = np.asarray(parameters, float), np.asarray(nominal_controls, float)
    targets = np.asarray(defocus_m, float)
    if p.shape != (column.n_parameters,) or nominal.shape != (len(column.channel_names),):
        raise ValueError("Parameter/control dimensions do not match the column.")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(nominal)):
        raise ValueError("Every physical parameter and control must be finite.")
    clearances = np.asarray(column.geometry_clearances(p))
    if not np.all(np.isfinite(clearances)) or np.any(clearances < 0):
        raise ValueError("Column geometry overlaps a lens, boundary, or the sample.")
    if targets.ndim != 1 or targets.size == 0 or not np.all(np.isfinite(targets)):
        raise ValueError("defocus_m must be a nonempty finite vector in metres.")
    if not np.isfinite(control_half_width) or control_half_width <= 0:
        raise ValueError("control_half_width must be finite and positive.")
    if int(continuation_steps) != continuation_steps or continuation_steps < 1:
        raise ValueError("continuation_steps must be a positive integer.")
    if len(selected_channels) != 2 or len(set(selected_channels)) != 2:
        raise ValueError("Exactly two distinct downstream channels are required.")
    if any(name not in {"IL1", "IL2", "IL3", "PL1"} for name in selected_channels):
        raise ValueError("Only downstream free-lens channels may change.")
    active = np.asarray([column.channel_names.index(name) for name in selected_channels])
    reference = np.asarray(imaging_metrics(single_ray_transfer(column, p, nominal)))
    if reference[0] <= 0 or not np.all(np.isfinite(reference)):
        raise ValueError("Nominal state is not a finite imaging map.")
    scale = max(float(np.max(np.abs(targets))), 1e-6)

    def metrics(values):
        control = jnp.asarray(nominal).at[active].set(values)
        return imaging_metrics(single_ray_transfer(column, p, control))

    # Share a JIT residual with target-as-data across continuation points.
    residual = jax.jit(lambda values, target: jnp.array([
        jnp.log(metrics(values)[0]/reference[0]), (metrics(values)[1]-target)/scale]))
    controls, matrices, achieved, path = [], [], [], []
    for target in targets:
        current = nominal[active].copy()
        for intermediate in np.linspace(reference[1], target, int(continuation_steps)+1)[1:]:
            result = least_squares(lambda values: residual(values, intermediate), current,
                bounds=(nominal[active]-control_half_width, nominal[active]+control_half_width),
                x_scale=np.full(active.size, control_half_width), max_nfev=220, max_iter=200,
                ftol=0., xtol=0., gtol=1e-14)
            current = np.asarray(result.x)
            error = np.asarray(residual(current, intermediate))
            constraint_satisfied = bool(np.all(np.isfinite(error)) and np.max(np.abs(error)) <= 1e-7)
            path.append({"target_defocus_m": float(intermediate),
                         "selected_control_values": current.tolist(),
                         "scaled_constraint_error": error.tolist(),
                         "optimizer_success": bool(result.success),
                         "optimizer_status": result.status,
                         "constraint_satisfied": constraint_satisfied})
            # This is a deterministic two-equation root design. Directly
            # verified forward constraints certify it even if the optimizer
            # stalls at roundoff; preserve that status instead of relabelling it.
            if not constraint_satisfied:
                raise RuntimeError(
                    "Signed imaging diversity solve failed within the declared local bounds: "
                    f"target={intermediate:g} m, status={result.status}, residual={error}. "
                    "This local failure is not proof that no alternative control path exists.")
        control = nominal.copy()
        control[active] = current
        matrix = np.asarray(single_ray_transfer(column, p, control))
        controls.append(control)
        matrices.append(matrix)
        achieved.append(np.asarray(imaging_metrics(matrix)))
    return ImagingDiversityDesign(np.asarray(controls), np.asarray(matrices), targets,
        np.asarray(achieved), nominal, tuple(selected_channels), tuple(path))


def sample_scale_table(magnifications, *, period_m=23e-9, pixel_m=55e-6, detector_n=256):
    """Nominal physical sampling for choosing a certified multilayer region."""
    return [{"magnification": float(m), "sample_pixel_nm": pixel_m/m*1e9,
             "sample_fov_nm": detector_n*pixel_m/m*1e9,
             "period_detector_pixels": period_m*m/pixel_m,
             "periods_across_field": detector_n*pixel_m/m/period_m}
            for m in magnifications]


def geometry_anchors_from_notebook(path):
    """Read manual diagram annotations as data, without executing notebook code.

    Only literal assignments and +/-/*// arithmetic for the three calibration
    constants are admitted. Other code, magic commands, and document text are
    never executed. The result is an approximate geometry, not a metrology
    uncertainty or a stigmator-location measurement.
    """
    notebook = json.loads(Path(path).read_text())
    wanted = {"label_y", "length_scale_px", "length_scale_m"}
    constants = {}

    def numeric(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            a, b = numeric(node.left), numeric(node.right)
            if isinstance(node.op, ast.Add): return a+b
            if isinstance(node.op, ast.Sub): return a-b
            if isinstance(node.op, ast.Mult): return a*b
            return a/b
        raise ValueError("Geometry scale must be literal numeric arithmetic.")

    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        source = source if isinstance(source, str) else "".join(source)
        # Cell magics cannot provide calibration constants in this contract.
        if source.lstrip().startswith("%%"):
            continue
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        for node in ast.parse(source).body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name) or target.id not in wanted:
                continue
            if target.id in constants:
                raise ValueError("Ambiguous repeated geometry calibration assignment.")
            constants[target.id] = ast.literal_eval(node.value) if target.id == "label_y" else numeric(node.value)
    if set(constants) != wanted:
        raise ValueError("Notebook must define label_y, length_scale_px, and length_scale_m.")
    labels = constants["label_y"]
    pixel_span, physical_span = constants["length_scale_px"], constants["length_scale_m"]
    if not np.all(np.isfinite([pixel_span, physical_span])) or min(pixel_span, physical_span) <= 0:
        raise ValueError("Geometry scale must be finite and positive.")
    origin = labels["Sample"]
    mapping = {"Source": "source_z_m", "CL1": "CL1", "CL2": "CL3",
               "CLAperture": "C-aperture", "CLMini": "C-mini",
               "Objective lens Pre": "Obj-prefield", "Objective lens Post": "Obj-post",
               "SAAperture": "SA-aperture", "IL1": "IL1", "IL2": "IL2",
               "IL3": "IL3", "PL1": "PL1", "Screen": "detector_z_m"}
    result = {name: (float(labels[key])-float(origin))*physical_span/pixel_span
              for key, name in mapping.items()}
    if not np.all(np.isfinite(list(result.values()))):
        raise ValueError("Every requested geometry landmark must be finite.")
    return result


def compare_geometry(column, notebook_path):
    anchors = geometry_anchors_from_notebook(notebook_path)
    initial = {lens.name: lens.z_center_m-column.sample_z_m for lens in column.lenses}
    initial.update(source_z_m=column.source_z_m-column.sample_z_m,
                   detector_z_m=column.detector_z_m-column.sample_z_m)
    return [{"component": name, "notebook_sample_relative_z_m": value,
             "synthetic_sample_relative_z_m": initial.get(name),
             "difference_m": None if name not in initial else value-initial[name]}
            for name, value in anchors.items()]


def column_with_geometry_anchors(column, notebook_path):
    """Build a new initial column in the diagram's sample-at-zero coordinates.

    Lens lengths and DAC laws remain provisional values from the caller.
    They need new initialization; old fitted controls, entrance covariances,
    image checkpoints, and parameter bounds are not transferable to this
    geometry. The input column and both user files remain unchanged.
    """
    anchors = geometry_anchors_from_notebook(notebook_path)
    return replace(column, lenses=tuple(replace(lens, z_center_m=anchors[lens.name])
                                        for lens in column.lenses),
                   sample_z_m=0., source_z_m=anchors["source_z_m"],
                   detector_z_m=anchors["detector_z_m"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--magnifications", type=int, nargs="+", default=[40000, 80000])
    parser.add_argument("--defocus-um", type=float, default=100.)
    args = parser.parse_args()
    if len(args.magnifications) not in (1, 2) or args.defocus_um <= 0:
        parser.error("Choose one or two magnifications and a positive defocus extent.")
    if jax.default_backend() != "gpu":
        raise RuntimeError("The imaging-design forward model requires the GPU.")
    jax.config.update("jax_enable_x64", True)
    root = Path(__file__).resolve().parents[2]
    import tomllib
    config_path = root / "examples/lens_inversion/neoarm_inversion_config.toml"
    column = hard_edge_column_from_config(tomllib.loads(config_path.read_text()))
    from neoarm_workflow import load_saved_acquisition, nominal_sample_requests
    archive = load_saved_acquisition(root / "examples/generated/neoarm_inversion/physical_checkpoint_native_imaging_v5_gpu", config_path)
    nominal = {int(request.target_magnification): request for request in nominal_sample_requests(archive)}
    parameters = np.asarray(archive.metadata["parameters"])
    records = []
    for mag in args.magnifications:
        request = nominal[mag]
        design = design_imaging_diversity(column, parameters, request.raw_dac_values,
                                          defocus_m=np.array([-1, 0, 1])*args.defocus_um*1e-6)
        records.append({"nominal_magnification": mag, "nominal_controls": design.nominal_controls.tolist(),
            "controls": design.controls.tolist(), "sample_to_detector_matrices": design.matrices.tolist(),
            "target_defocus_m": design.target_defocus_m.tolist(),
            "achieved_magnification_defocus_angle": design.achieved_metrics.tolist(),
            "selected_channels": design.selected_channels, "continuation": design.continuation})
        print(f"{mag/1000:g}k: signed imaging states solved on {jax.devices()[0]}", flush=True)
    report = {"schema": "neoarm-imaging-diversity-design-v1", "backend": jax.default_backend(),
        "scope": "actual synthetic column control feasibility; not acquired images or lens recovery",
        "hardware_commands_admitted": False, "branch_uniqueness_proved": False,
        "calibration_uncertainty_propagated": False, "condenser_recovery_proved": False,
        "requires": ["calibrated sample-position/angle responses including covariance", "verified control units and safe bounds"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "archive_manifest_sha256": archive.checkpoint.manifest_sha256,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scale_table": sample_scale_table(args.magnifications), "designs": records,
        "geometry_comparison": compare_geometry(column, root / "examples/microscope_models/measure_microscope_geometry.ipynb"),
        "geometry_notebook_sha256": hashlib.sha256((root / "examples/microscope_models/measure_microscope_geometry.ipynb").read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
