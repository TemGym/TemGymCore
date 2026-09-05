"""Small, explicit helpers for the NEOARM review notebook.

This module reads existing acquisitions, makes figures, and reproduces a
sample-plane Gaussian render. It does not run the experimental global fitter.
The physical model and checkpoint validation remain in ``temgym_core``.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SavedAcquisition:
    directory: Path
    payload: dict
    detector: object
    requests: tuple
    checkpoint: object

    def request(self, frame_id):
        return next(r for r in self.requests if r.frame_id == frame_id)

    def image(self, frame_id):
        return self.checkpoint.record(frame_id).load_intensity()

    @property
    def metadata(self):
        return self.payload["render_metadata"]


def load_saved_acquisition(directory, config_path):
    """Validate every frame/array checksum and the configuration before use.

    This checks archive integrity, not agreement with the current source code
    or experimental accuracy. The old checkpoints do not hash their source.
    """
    from temgym_core.microscope_inversion import SamplePlaneState
    from temgym_core.neoarm_calibration import (
        DetectorGeometry,
        FrameRequest,
        load_completed_frame_sequence_checkpoint,
    )

    directory = Path(directory)
    with (directory / "manifest.json").open() as stream:
        payload = json.load(stream)["payload"]
    config_digest = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
    if payload["render_metadata"].get("config_sha256") != config_digest:
        raise ValueError(
            "Saved acquisition uses a different configuration; do not relabel it."
        )
    geometry = dict(payload["detector"])
    if geometry.pop("coordinate_convention") != "pixel_centres_yx_storage":
        raise ValueError("Unsupported detector coordinate convention.")
    detector = DetectorGeometry(**geometry)
    requests = []
    for raw in payload["frame_requests"]:
        values = dict(raw)
        state = values.pop("sample_state")
        values["sample_state"] = SamplePlaneState(
            x=state["x_m"],
            y=state["y_m"],
            theta_x=state["theta_x_rad"],
            theta_y=state["theta_y_rad"],
            label=state["label"],
        )
        requests.append(FrameRequest(**values))
    storage = payload["storage"]
    checkpoint = load_completed_frame_sequence_checkpoint(
        requests,
        detector,
        directory,
        render_signature=payload["render_signature"],
        render_metadata=payload["render_metadata"],
        store_complex_field=storage["store_complex_field"],
        store_sample_fields=storage["store_sample_fields"],
        storage_dtype=storage["real_dtype"],
    )
    return SavedAcquisition(directory, payload, detector, tuple(requests), checkpoint)


def read_figure_metadata(path):
    """Read a historical figure's report; absence is not a successful result."""
    with Path(path).open() as stream:
        return json.load(stream)["metadata"]


def fit_status(metadata):
    """Keep optimizer convergence, synthetic acceptance and instrument use separate."""
    return {
        "optimizer_success": metadata.get("fit_success") is True,
        "accepted_synthetic_fit": metadata.get("selected_route_fit_accepted") is True,
        "instrument_accepted": metadata.get("final_instrument_acceptance") is True,
        "converged_starts": metadata.get("converged_multistarts"),
        "training_rms_sigma": metadata.get("data_only_training_weighted_rms_sigma"),
        "held_out_rms_sigma": metadata.get("held_out_weighted_rms_sigma"),
    }


def nominal_sample_requests(run):
    return tuple(
        sorted(
            (
                r
                for r in run.requests
                if r.measurement == "sample"
                and r.optical_state_label.startswith("nominal_")
                and r.dac_approach == "forward"
                and r.repeat_index == 0
            ),
            key=lambda r: r.target_magnification,
        )
    )


def expected_detector_counts(intensity, detector, dose_electrons):
    """Unit-integral incident beam → counts, preserving transmission losses.

    These checkpoints store intensity density at pixel centres. The optical
    area includes the detector affine calibration. This is the original
    centre-sampling detector approximation, not pixel-area quadrature.
    """
    intensity = np.asarray(intensity, dtype=float)
    if not np.isfinite(dose_electrons) or dose_electrons <= 0:
        raise ValueError("dose_electrons must be finite and positive.")
    if (
        intensity.shape != tuple(detector.shape_yx)
        or not np.all(np.isfinite(intensity))
        or np.any(intensity < 0)
    ):
        raise ValueError("Expected a finite nonnegative detector image.")
    optical_pixel_area = np.prod(detector.pixel_size_yx_m) / abs(
        np.linalg.det(detector.affine_xy)
    )
    return intensity * optical_pixel_area * dose_electrons


def draw_detector_images(
    run, requests, column, *, dose_electrons=2e7, read_noise_e=0.6
):
    """Show reproducible synthetic exposures, with an absolute count scale."""
    import matplotlib.pyplot as plt

    if not np.isfinite(read_noise_e) or read_noise_e < 0:
        raise ValueError("read_noise_e must be finite and nonnegative.")
    requests = tuple(requests)
    if not requests:
        raise ValueError("Select at least one frame.")
    means = [
        expected_detector_counts(run.image(r.frame_id), run.detector, dose_electrons)
        for r in requests
    ]
    counts = []
    for request, mean in zip(requests, means):
        rng = np.random.default_rng(51000 + request.frame_id)
        counts.append(
            rng.poisson(mean).astype(float) + rng.normal(0, read_noise_e, mean.shape)
        )
    width = run.detector.shape_yx[1] * run.detector.pixel_size_yx_m[1] * 1e3
    height = run.detector.shape_yx[0] * run.detector.pixel_size_yx_m[0] * 1e3
    fig, axes = plt.subplots(
        1,
        len(requests),
        figsize=(3.1 * len(requests), 3.9),
        squeeze=False,
        layout="constrained",
    )
    vmax = max(float(np.percentile(mean, 99.5)) for mean in means)
    for ax, request, frame in zip(axes[0], requests, counts):
        artist = ax.imshow(
            frame,
            origin="lower",
            cmap="gray",
            vmin=0,
            vmax=vmax,
            extent=(-width / 2, width / 2, -height / 2, height / 2),
        )
        jacobian = run.detector.affine_xy @ np.asarray(
            column.sample_position_jacobian(
                np.asarray(run.metadata["parameters"]), request.raw_dac_values
            )
        )
        bar_mm = width / 4
        specimen_bar_nm = (
            np.linalg.norm(np.linalg.solve(jacobian, [bar_mm * 1e-3, 0])) * 1e9
        )
        x0, y0 = -0.40 * width, -0.40 * height
        ax.plot([x0, x0 + bar_mm], [y0, y0], color="#E69F00", lw=3)
        ax.text(
            x0,
            y0 + 0.04 * height,
            f"{specimen_bar_nm:.1f} nm",
            color="#E69F00",
            fontsize=9,
        )
        mag = request.predicted_effective_magnification or request.target_magnification
        label = request.measurement.replace("_", " ")
        ax.set(
            title=f"{mag / 1000:.0f}k · {label}\nframe {request.frame_id}",
            xlabel="detector x [mm]",
        )
    axes[0, 0].set_ylabel("detector y [mm]")
    fig.suptitle("Synthetic specimen exposures · Gaussian forward model", fontsize=12)
    fig.colorbar(artist, ax=axes[0].tolist(), shrink=0.72, label="electrons / pixel")
    return fig, tuple(counts)


def frozen_sample_view(atlas, run, request, *, z_m):
    """Reconstruct a saved imaging crop, with the original interpolation halo.

    This does not reselect an LOD during reproduction. Fresh acquisitions use
    ``select_sample_view`` and must pass their own convergence checks.
    """
    from temgym_core.multiscale_sample import SampleLevel2D, SampleView

    if request.sample_level_index is None or request.sample_crop_bounds_xyxy_m is None:
        raise ValueError("The selected frame has no frozen sample crop.")
    pyramid = (
        atlas.pyramid
        if request.selected_region_name is None
        else atlas.pyramid_for_region(request.selected_region_name)
    )
    level = pyramid.levels[request.sample_level_index]
    if not np.isclose(
        level.max_spacing_m, request.selected_spacing_m, rtol=1e-9, atol=0
    ):
        raise ValueError("The current atlas level differs from the saved level.")
    periodic = request.feature_family == "silicon_lattice_vectors"
    if not periodic:
        xmin, xmax, ymin, ymax = request.sample_crop_bounds_xyxy_m
        halo = atlas.interpolation_halo_knots or 8
        ix0 = max(0, np.searchsorted(level.x_coords_m, xmin) - halo)
        ix1 = min(
            level.x_coords_m.size,
            np.searchsorted(level.x_coords_m, xmax, side="right") + halo,
        )
        iy0 = max(0, np.searchsorted(level.y_coords_m, ymin) - halo)
        iy1 = min(
            level.y_coords_m.size,
            np.searchsorted(level.y_coords_m, ymax, side="right") + halo,
        )
        level = SampleLevel2D(
            level.pathlength_yx_m[iy0:iy1, ix0:ix1],
            level.log_amplitude_yx[iy0:iy1, ix0:ix1],
            level.x_coords_m[ix0:ix1],
            level.y_coords_m[iy0:iy1],
            name=level.name + ":saved-crop",
        )
    holes = tuple(
        (
            r.centre_xy_m[0],
            r.centre_xy_m[1],
            atlas.certificate.perforation_diameter_m / 2,
            atlas.certificate.interface_width_m,
        )
        for r in atlas.regions.values()
        if r.kind == "perforation"
    )
    return SampleView.from_level(
        level,
        centre_xy_m=request.sample_translation_xy_m,
        rotation_rad=atlas.pose_rotation_rad,
        tilt_xy_rad=atlas.pose_tilt_xy_rad,
        boundary="periodic" if periodic else atlas.boundary_default,
        z=z_m,
        level_index=request.sample_level_index,
        vacuum_discs_local_xy_radius_m=holes,
    )


def reproduce_sample_image_gpu(run, frame_id, atlas, column, *, voltage_v=200000.0):
    """Recompute a chosen-mode image from the cubic specimen on the GPU.

    Uses a calibrated, unit-current Gaussian at the sample; this experiment
    therefore tests the sample/projector, not the condenser. The chosen-mode
    checkpoint supplies the fixed grid, waist, DACs and slice count. There is
    no CPU fallback and no beam-count cap. Returned images need independent
    convergence checks before they are admitted to an inversion.
    """
    import jax
    import jax.numpy as jnp
    from temgym_core.evaluate import gpu_kernel_available
    from temgym_core.neoarm_calibration import (
        reexpand_transmitted_field,
        propagate_gaussians_through_column,
        render_gaussian_detector_plane,
    )

    if not gpu_kernel_available() or jax.default_backend() != "gpu":
        raise RuntimeError("This forward simulation requires a working GPU kernel.")
    request = run.request(frame_id)
    if request.measurement != "sample":
        raise ValueError("Choose an on-axis sample frame.")
    metadata = run.metadata
    if "grid_origin_xy_m" not in metadata:
        raise ValueError(
            "Use the chosen-mode checkpoint with its explicit specimen grid."
        )
    ny, nx = metadata["grid_shape_yx"]
    dx, dy = metadata["grid_spacing_xy_m"]
    x = metadata["grid_origin_xy_m"][0] + np.arange(nx) * dx
    y = metadata["grid_origin_xy_m"][1] + np.arange(ny) * dy
    view = frozen_sample_view(atlas, run, request, z_m=column.sample_z_m)
    corners = np.array([[x[0], y[0]], [x[0], y[-1]], [x[-1], y[0]], [x[-1], y[-1]]])
    local = np.asarray(view.world_to_local(jnp.asarray(corners)))
    xmin, xmax, ymin, ymax = view.domain_bounds_xyxy_m
    if view.boundary != "periodic" and not (
        np.all((local[:, 0] >= xmin) & (local[:, 0] <= xmax))
        and np.all((local[:, 1] >= ymin) & (local[:, 1] <= ymax))
    ):
        raise ValueError("Saved sample crop does not cover the concrete beam grid.")
    print(
        f"GPU {jax.devices()[0]}: {nx * ny:,} beamlets; "
        f"{nx * ny * np.prod(run.detector.shape_yx):,} beamlet–pixel pairs; "
        f"~{nx * ny * 160 / 2**20:.0f} MiB for primary beam arrays (workspace additional).",
        flush=True,
    )
    print(
        "Runtime depends on GPU and compilation; detector evaluation is streamed in 16-row tiles.",
        flush=True,
    )
    with jax.default_device(jax.devices("gpu")[0]):
        xx, yy = jnp.meshgrid(jnp.asarray(x), jnp.asarray(y), indexing="xy")
        cx, cy = atlas.pose_translation_xy_m
        waist = metadata["sample_illumination_radius_m"]
        incident = (
            np.sqrt(2 / np.pi)
            / waist
            * jnp.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / waist**2)
        )
        beam, transmitted = reexpand_transmitted_field(
            incident,
            view,
            x,
            y,
            voltage=voltage_v,
            z_m=column.sample_z_m,
            width_over_step_yx=tuple(metadata["width_over_step_yx"]),
            coefficient_method="jax",
        )
        downstream = propagate_gaussians_through_column(
            beam,
            column,
            np.asarray(metadata["parameters"]),
            np.asarray(request.raw_dac_values),
            segment="sample_to_detector",
            slices_per_lens=metadata["slices_per_lens"],
        )
        rendered = render_gaussian_detector_plane(
            downstream,
            run.detector,
            method="gpu_kernel_required",
            tile_rows=16,
        )
    return rendered, np.asarray(transmitted)


def sample_basis_at_source(column, parameters, dacs, position_scale, angle_scale):
    """Back-propagate two lab-x specimen inputs and return their frame offset.

    Sum the physical Larmor phase instead of taking the angle of A: a signed
    focusing coefficient can change that angle by pi without rotating the field.
    """
    upstream = np.asarray(column.matrix(parameters, dacs, segment="source_to_sample"))
    basis = np.zeros((4, 2))
    basis[0, 0], basis[1, 1] = position_scale, angle_scale
    rotation = 0.0
    for lens, _, _, bias, slope in column._decoded_lenses(np.asarray(parameters)):
        if lens.exit_z <= column.sample_z_m:
            q = float(bias)
            if lens.channel_index is not None:
                q += float(slope) * (dacs[lens.channel_index] - lens.reference_dac)
            rotation -= q
    return np.linalg.solve(upstream, basis), -rotation


def column_basis_trace(
    column,
    parameters,
    dacs,
    position_scale,
    angle_scale,
    *,
    voltage_v=200000.0,
    source_basis=None,
    frame_rotation_offset=0.0,
    label="synthetic nominal",
):
    """Dense two-ray trace in the co-rotating frame, with exact drift extrema.

    Phase sampling resolves high-order hard-edge lenses; drift A/B roots and
    envelope minima are inserted analytically rather than missed by plotting.
    """
    import jax
    import jax.numpy as jnp
    from temgym_core.microscope_inversion import hard_edge_solenoid_matrix
    from temgym_core.plotting import PositionAngleTrace

    parameters = np.asarray(parameters, dtype=float)
    states = (
        np.zeros((4, 2))
        if source_basis is None
        else np.array(source_basis, dtype=float, copy=True)
    )
    if source_basis is None:
        states[0, 0], states[1, 1] = position_scale, angle_scale
    if states.shape != (4, 2) or not np.all(np.isfinite(states)):
        raise ValueError(
            "source_basis must have shape (4,2) in x,theta_x,y,theta_y order."
        )
    z, rotation, rows = float(parameters[0]), 0.0, []

    def record():
        axis = np.array(
            [
                np.cos(rotation + frame_rotation_offset),
                np.sin(rotation + frame_rotation_offset),
            ]
        )
        heights = axis @ states[[0, 2]]
        rows.append((z, *heights))

    def drift_to(target):
        nonlocal z, states
        length = float(target) - z
        if length < -1e-12:
            raise ValueError("Lens bodies overlap or are out of order.")
        if length <= 0:
            return
        axis = np.array(
            [
                np.cos(rotation + frame_rotation_offset),
                np.sin(rotation + frame_rotation_offset),
            ]
        )
        h, slope = axis @ states[[0, 2]], axis @ states[[1, 3]]
        candidates = [*np.linspace(0, length, 65)[1:]]
        candidates.extend(-h[i] / slope[i] for i in range(2) if slope[i] != 0)
        if slope @ slope > 0:
            candidates.append(-(h @ slope) / (slope @ slope))
        if z < column.sample_z_m < target:
            candidates.append(column.sample_z_m - z)
        start, entrance = z, states.copy()
        for distance in sorted(set(t for t in candidates if 0 < t <= length)):
            states = entrance.copy()
            states[[0, 2]] += distance * entrance[[1, 3]]
            z = start + distance
            record()

    record()
    for lens, centre, length, bias, slope in column._decoded_lenses(
        jnp.asarray(parameters)
    ):
        centre, length, bias, slope = map(float, (centre, length, bias, slope))
        q = (
            bias
            if lens.channel_index is None
            else bias + slope * (dacs[lens.channel_index] - lens.reference_dac)
        )
        drift_to(centre - length / 2)
        entrance, start, initial_rotation = states.copy(), z, rotation
        fractions = np.linspace(0, 1, max(128, int(np.ceil(abs(q) * 64))) + 1)[1:]
        matrices = np.asarray(
            jax.vmap(lambda f: hard_edge_solenoid_matrix(q * f, length * f))(
                jnp.asarray(fractions)
            )
        )
        for fraction, matrix in zip(fractions, matrices):
            states = matrix @ entrance
            z, rotation = start + length * fraction, initial_rotation - q * fraction
            record()
    drift_to(parameters[1])
    data = np.asarray(rows)
    return PositionAngleTrace(
        z=data[:, 0],
        centroid=np.zeros(len(data)),
        position=data[:, 1],
        angle=data[:, 2],
        radius=np.hypot(data[:, 1], data[:, 2]),
        coordinate="x_rot",
        position_scale=position_scale,
        angle_scale=angle_scale,
        label=label,
        voltage=voltage_v,
    )


def draw_column_rays(
    column,
    parameters,
    dacs,
    trace,
    *,
    sample_label,
    voltage_v=200000.0,
    semantics="gaussian",
):
    """Use the existing publication renderer with physical finite-lens glyphs."""
    from temgym_core.components import ElectromagneticLens
    from temgym_core.constants import compute_Rc_from_voltage
    from temgym_core.plotting import (
        ColumnElement,
        ColumnSection,
        plot_neoarm_ray_diagram,
    )

    rc = float(compute_Rc_from_voltage(voltage_v))
    components, labels, roles = [], [], []
    for lens, centre, length, bias, slope in column._decoded_lenses(
        np.asarray(parameters)
    ):
        q = (
            float(bias)
            if lens.channel_index is None
            else float(bias + slope * (dacs[lens.channel_index] - lens.reference_dac))
        )
        excitation = -q / rc
        if excitation == 0:
            raise ValueError(
                "Zero-strength glyph needs an explicit passive finite lens."
            )
        components.append(
            ElectromagneticLens(
                z=float(centre - length / 2),
                turns=1.0,
                current=excitation,
                Gc=1.0,
                Tc=float(length) / abs(excitation),
            )
        )
        labels.append(lens.name)
        # Reserve color for the basis rays. Names and section titles carry
        # lens semantics; all finite lens bodies use the neutral glyph style.
        roles.append("lens")
    sample_z = column.sample_z_m
    elements = (
        ColumnElement(sample_z, sample_label, "sample"),
        ColumnElement(
            sample_z - 0.018, "Corrector (off)", "corrector_off", sample_z - 0.008
        ),
        ColumnElement(float(parameters[1]), "Medipix 256 × 256", "detector"),
    )
    sections = (
        ColumnSection("Condenser", float(parameters[0]), sample_z - 0.004),
        ColumnSection("Objective / sample", sample_z - 0.032, sample_z + 0.036),
        ColumnSection("Intermediate / projector", sample_z, float(parameters[1])),
    )
    return plot_neoarm_ray_diagram(
        components,
        trace,
        component_labels=labels,
        component_roles=roles,
        extra_elements=elements,
        sections=sections,
        trace_semantics=semantics,
        title="Synthetic nominal column · physical z · co-rotating ray coordinate",
    )
