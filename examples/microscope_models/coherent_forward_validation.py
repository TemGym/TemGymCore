"""Forward-only checks of coherent Gaussian microscope images against Fresnel FFT.

No optical fitting, noise, intensity mixtures, or fitted image normalizations.
The illumination is fixed while its Gaussian basis is refined.
"""

import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from coherent_microscope_workflow import load_model, make_magical_sample
from scipy.ndimage import map_coordinates
from scipy.special import gamma

from temgym_core.coherent_image import CoherentDetector, microscope_image
from temgym_core.constants import energy2wavelength
from temgym_core.evaluate import evaluate_gaussians, gpu_kernel_available
from temgym_core.gaussian import FreeSpacePropagator, make_gaussian
from temgym_core.solenoid import column_matrix
from temgym_core.source import _gaussian_cardinal_coefficients

RADIUS_M = 30e-9
VOLTAGE = 200000.0
SOURCE_HALF_WIDTH_M = 52e-9
INCIDENT_FLUX = np.pi * RADIUS_M**2 * gamma(1.25)


def illumination_amplitude(xy):
    """Fixed smooth circular field: intensity exp[-(r/R)^8]."""
    return np.exp(-0.5 * (np.sum(np.asarray(xy) ** 2, axis=-1) / RADIUS_M**2) ** 4)


def source_wave(waist_nm):
    """Represent the SAME illumination on a refined Gaussian lattice.

    Existing TemGym cardinal coefficients reconstruct the sampled field; the
    source envelope is not defined by the numerical width of its beamlets.
    """
    width = float(waist_nm) * 1e-9
    step = width / 1.1
    half = int(np.ceil(SOURCE_HALF_WIDTH_M / step))
    axis = np.arange(-half, half + 1) * step
    xx, yy = np.meshgrid(axis, axis)
    xy = np.stack((xx, yy), axis=-1)
    coefficients = _gaussian_cardinal_coefficients(
        illumination_amplitude(xy), width_over_step=width / step
    )
    # Bound omitted field contributions by the sum of removed coefficients.
    keep = np.abs(coefficients) > 1e-14
    omitted_bound = float(np.abs(coefficients[~keep]).sum())
    return make_gaussian(
        x=xx[keep],
        y=yy[keep],
        amp=coefficients[keep],
        waist_x=width,
        waist_y=width,
        voltage=VOLTAGE,
    ), omitted_bound


def make_scene(shape_yx, defocus_um, samples_per_pixel):
    sample, _, roi = make_magical_sample(VOLTAGE)
    sample = replace(sample, centre_xy_m=jnp.asarray(-roi))
    model = load_model(Path(__file__).with_name("coherent_generating_model.json"))
    controls = jnp.asarray(
        [
            1.0,
            0.7878671830443049,
            0.8960533754728769,
            0.7326077442697242,
            1.0895397789606636,
        ]
    )
    train = model.projector

    def components(t):
        return t.components(controls, VOLTAGE, VOLTAGE, 0.0)

    m = np.asarray(column_matrix(components(train)))
    correction = -np.trace(np.linalg.solve(m[:2, :2], m[:2, 2:4])) / 2
    train = replace(train, gaps=train.gaps.at[0].add(correction))
    focused = np.asarray(column_matrix(components(train)))
    magnification = np.linalg.norm(focused[:2, 0])
    pitch = magnification * 2e-9

    def column(offset_um):
        return components(replace(train, gaps=train.gaps.at[0].add(offset_um * 1e-6)))

    def detector(offset_um, samples=samples_per_pixel):
        return CoherentDetector(
            tuple(shape_yx), pitch, column(offset_um)[-1].z, samples
        )

    matrices = [np.asarray(column_matrix(column(offset))) for offset in defocus_um]
    return sample, column, detector, matrices, float(magnification)


def fft_reference(sample, matrices, detectors, *, step_nm, width_nm):
    """Independent scalar Fresnel propagation, then mapped detector integrals.

    For round A and B, intensity at detector r equals the Fresnel intensity
    at A^-1 r, propagated by z=tr(A^-1 B)/2, divided by abs(det(A)).
    This factors out the output quadratic phase, which cannot affect intensity.
    Complex specimen transmission is sampled directly, without beamlets or
    the sample component's local quadratic action approximation.
    """
    step = step_nm * 1e-9
    size = round(width_nm / step_nm)
    if size % 2:
        raise ValueError("Use an even reference FFT size.")
    half = int(np.ceil(SOURCE_HALF_WIDTH_M / step))
    axis = np.arange(-half, half) * step
    xx, yy = np.meshgrid(axis, axis)
    xy = np.stack((xx, yy), axis=-1)
    actions = np.asarray(sample.evaluate_actions(jnp.asarray(xy)))
    # SampleView stores optical path in metres, not phase in radians.
    wavelength = float(energy2wavelength(VOLTAGE))
    exit_wave = illumination_amplitude(xy) * np.exp(
        actions[..., 1] + 1j * (2 * np.pi / wavelength) * actions[..., 0]
    )
    padded = np.zeros((size, size), dtype=complex)
    padded[size // 2 - half : size // 2 + half, size // 2 - half : size // 2 + half] = (
        exit_wave
    )
    spectrum = np.fft.fft2(np.fft.ifftshift(padded))
    frequency = np.fft.fftfreq(size, step)
    frequency2 = frequency[:, None] ** 2 + frequency[None, :] ** 2
    wavelength = float(energy2wavelength(VOLTAGE))
    images = []
    for matrix, detector in zip(matrices, detectors):
        a, b = matrix[:2, :2], matrix[:2, 2:4]
        effective_drift = np.linalg.solve(a, b)
        distance = np.trace(effective_drift) / 2
        np.testing.assert_allclose(effective_drift, np.eye(2) * distance, atol=1e-12)
        wave = np.fft.fftshift(
            np.fft.ifft2(
                spectrum * np.exp(-1j * np.pi * wavelength * distance * frequency2)
            )
        )
        xy_object = np.asarray(detector.grid.coords) @ np.linalg.inv(a).T
        indices = (xy_object / step + size / 2)[:, ::-1].T
        values = map_coordinates(np.abs(wave) ** 2, indices, order=3, mode="constant")
        n = detector.samples_per_pixel
        ny, nx = detector.shape_yx
        images.append(
            values.reshape(ny, n, nx, n).mean(axis=(1, 3))
            * detector.pixel_pitch_m**2
            / abs(np.linalg.det(a))
            / INCIDENT_FLUX
        )
    return np.asarray(images)


def relative_error(actual, reference):
    return float(
        np.linalg.norm(np.asarray(actual) - reference) / np.linalg.norm(reference)
    )


def check_coherent_forward(
    *,
    output_directory,
    waists_nm=(4.0, 2.0, 1.0, 0.5),
    defocus_um=(-12.0, 0.0, 12.0),
    shape_yx=(40, 40),
    samples_per_pixel=2,
    tolerance=0.005,
):
    """Supply resolution levels; get forward convergence, images and diagnostics.

    The default target is 0.5% relative image L2 error across all three planes.
    Independent FFT sampling and padding must each agree much more closely.
    A failure is saved and reported as failure, never hidden by normalization.
    """
    widths = tuple(float(w) for w in waists_nm)
    if (
        len(widths) < 2
        or not all(np.isfinite(widths))
        or min(widths) <= 0
        or np.any(np.diff(widths) >= 0)
    ):
        raise ValueError("Supply at least two decreasing positive beamlet widths.")
    if not 0 < tolerance < 1:
        raise ValueError("Supply a relative error tolerance between zero and one.")
    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    sample, column, detector, matrices, magnification = make_scene(
        shape_yx, defocus_um, samples_per_pixel
    )
    detectors = [detector(offset) for offset in defocus_um]
    print("Checking the independent FFT reference...", flush=True)
    coarse_reference = fft_reference(
        sample, matrices, detectors, step_nm=0.25, width_nm=256.0
    )
    padded_reference = fft_reference(
        sample, matrices, detectors, step_nm=0.25, width_nm=512.0
    )
    reference = fft_reference(
        sample, matrices, detectors, step_nm=0.125, width_nm=512.0
    )
    padding_errors = [
        relative_error(a, b) for a, b in zip(coarse_reference, padded_reference)
    ]
    fft_errors = [relative_error(a, b) for a, b in zip(padded_reference, reference)]
    refined_detectors = [
        detector(offset, samples=2 * samples_per_pixel) for offset in defocus_um
    ]
    refined_reference = fft_reference(
        sample, matrices, refined_detectors, step_nm=0.125, width_nm=512.0
    )
    pixel_errors = [relative_error(a, b) for a, b in zip(reference, refined_reference)]
    print(
        f"FFT padding/sampling errors: {max(padding_errors):.3g}, {max(fft_errors):.3g}",
        flush=True,
    )
    apply_sample = jax.jit(lambda beam: sample(beam))
    render = jax.jit(
        lambda beam, offset: (
            microscope_image(beam, column(offset), detector(offset), method="scan")
            / INCIDENT_FLUX
        )
    )
    rows, all_images = [], []
    for width in widths:
        source, discarded = source_wave(width)
        exited = jax.block_until_ready(apply_sample(source))
        images, timings = [], []
        for offset in defocus_um:
            jax.block_until_ready(render(exited, jnp.asarray(offset)))
            start = perf_counter()
            images.append(
                np.asarray(jax.block_until_ready(render(exited, jnp.asarray(offset))))
            )
            timings.append(perf_counter() - start)
        images = np.asarray(images)
        row = {
            "waist_nm": width,
            "gaussians": int(source.x.size),
            "image_errors": [relative_error(a, b) for a, b in zip(images, reference)],
            "change_from_previous": None
            if not all_images
            else [relative_error(a, b) for a, b in zip(all_images[-1], images)],
            "discarded_source_field_bound": discarded,
            "warm_seconds": timings,
        }
        rows.append(row)
        all_images.append(images)
        print(
            f"{width:g} nm: {row['gaussians']:,} Gaussians; "
            f"errors {[round(100 * e, 4) for e in row['image_errors']]}%",
            flush=True,
        )
        (root / "progress.json").write_text(json.dumps(rows, indent=2))
    # Isolate source reconstruction and full-column transport from sample error.
    points = np.asarray(detectors[0].grid.coords) @ np.linalg.inv(matrices[0][:2, :2]).T
    grid = SimpleNamespace(shape=detectors[0].grid.shape, coords=jnp.asarray(points))
    incoming = np.asarray(evaluate_gaussians(source, grid, method="scan"))
    source_error = relative_error(
        incoming, illumination_amplitude(points).reshape(grid.shape)
    )
    optics_errors = []
    for i, offset in enumerate(defocus_um):
        a, b = matrices[i][:2, :2], matrices[i][:2, 2:4]
        distance = np.trace(np.linalg.solve(a, b)) / 2
        points = np.asarray(detectors[i].grid.coords) @ np.linalg.inv(a).T
        grid = SimpleNamespace(
            shape=detectors[i].grid.shape, coords=jnp.asarray(points)
        )
        out = FreeSpacePropagator(include_axial_path=False)(exited, distance)
        field = np.asarray(evaluate_gaussians(out, grid, method="scan"))
        ny, nx = shape_yx
        n = samples_per_pixel
        equivalent = (np.abs(field) ** 2).reshape(ny, n, nx, n).mean(axis=(1, 3))
        equivalent *= (
            detectors[i].pixel_pitch_m ** 2 / abs(np.linalg.det(a)) / INCIDENT_FLUX
        )
        optics_errors.append(relative_error(all_images[-1][i], equivalent))
    backend = "GPU spatial" if gpu_kernel_available() else "CPU scan fallback"
    spatial = (
        np.asarray(
            microscope_image(
                exited, column(defocus_um[-1]), detectors[-1], method="spatial"
            )
        )
        / INCIDENT_FLUX
    )
    spatial_error = relative_error(spatial, all_images[-1][-1])
    passed = bool(
        max(rows[-1]["image_errors"]) < tolerance
        and max(rows[-1]["change_from_previous"]) < tolerance
        and max(fft_errors + padding_errors) < tolerance / 10
        and max(pixel_errors) < tolerance
        and max(optics_errors) < 1e-4
        and source_error < tolerance / 10
        and spatial_error < 1e-8
    )
    report = {
        "passed": passed,
        "image_tolerance": tolerance,
        "voltage_v": VOLTAGE,
        "magnification": magnification,
        "defocus_um": list(defocus_um),
        "shape_yx": list(shape_yx),
        "samples_per_pixel": samples_per_pixel,
        "illumination": "Fixed coherent circular super-Gaussian: intensity exp[-(r/30nm)^8].",
        "rows": rows,
        "fft_padding_errors": padding_errors,
        "fft_sampling_errors": fft_errors,
        "pixel_quadrature_errors": pixel_errors,
        "source_field_error": source_error,
        "full_column_transport_errors": optics_errors,
        "spatial_backend": backend,
        "spatial_image_error": spatial_error,
        "scope": "Noise-free coherent forward validation only; no fitting or intensity mixtures.",
    }
    np.savez_compressed(
        root / "coherent_forward.npz",
        images=np.asarray(all_images),
        reference=reference,
        refined_pixel_reference=refined_reference,
    )
    (root / "coherent_forward.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def show_coherent_forward(*, output_directory, report):
    import matplotlib.pyplot as plt

    data = np.load(Path(output_directory) / "coherent_forward.npz")
    fig, axes = plt.subplots(
        2,
        len(report["defocus_um"]),
        figsize=(10, 6),
        constrained_layout=True,
        squeeze=False,
    )
    for i, offset in enumerate(report["defocus_um"]):
        upper = max(data["reference"][i].max(), data["images"][-1, i].max())
        for row, (name, values) in enumerate(
            (
                ("Gaussian beams", data["images"][-1]),
                ("Fresnel FFT reference", data["reference"]),
            )
        ):
            axes[row, i].imshow(
                values[i], origin="lower", cmap="gray", vmin=0, vmax=upper
            )
            axes[row, i].set(title=f"{name}: {offset:+g} µm")
            axes[row, i].set_axis_off()
    fig.savefig(Path(output_directory) / "coherent_forward.png", dpi=160)
    plt.show()
    fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
    for i, offset in enumerate(report["defocus_um"]):
        ax.loglog(
            [r["gaussians"] for r in report["rows"]],
            [100 * r["image_errors"][i] for r in report["rows"]],
            "o-",
            label=f"{offset:+g} µm",
        )
    ax.axhline(
        100 * report["image_tolerance"], color="black", linestyle="--", label="Target"
    )
    ax.set(xlabel="Number of coherent Gaussians", ylabel="Image error against FFT (%)")
    ax.legend()
    fig.savefig(Path(output_directory) / "coherent_convergence.png", dpi=160)
    plt.show()
    row = report["rows"][-1]
    print(
        f"{row['gaussians']:,} Gaussians: worst image error {100 * max(row['image_errors']):.4f}%."
    )
    print(
        f"Detector quadrature refinement: {100 * max(report['pixel_quadrature_errors']):.4f}%"
    )
    print(
        f"Full column versus equivalent Fresnel transport: {100 * max(report['full_column_transport_errors']):.6f}%"
    )
    print(
        "PASS"
        if report["passed"]
        else "NOT CONVERGED: inspect the saved diagnostics before fitting."
    )


def check_full_microscope_sampling(
    *, model_path, measurement_directory, refinements=(1, 2, 4, 8, 16, 32)
):
    """Refine Gaussian counts for every recorded image of a supplied microscope.

    This is a basis-refinement check, not an independent FFT reference and not
    a detector-quadrature check. It may also be used on a fitted model. Data and
    currents saved here are noiseless predictions, never substituted for the
    measured arrays used by the fit.
    """
    from coherent_microscope_workflow import (
        acquisition_state,
        experiment_numerics,
        load_measurements,
    )

    from temgym_core.coherent_calibration import make_predictor

    root = Path(measurement_directory)
    settings, _ = load_measurements(root)
    sample, _, roi = make_magical_sample()
    sample = replace(sample, centre_xy_m=jnp.asarray(-roi))
    state = acquisition_state(load_model(model_path), len(settings.images))
    previous, rows = None, []
    for refinement in refinements:
        numerical = experiment_numerics(refinement=refinement)
        start = perf_counter()
        pixels, currents = make_predictor(settings, sample, numerical)(state)
        pixels = np.asarray(pixels)
        difference = None
        if previous is not None:
            numerator = np.linalg.norm(
                (pixels - previous).reshape(len(settings.images), -1), axis=1
            )
            denominator = np.linalg.norm(
                (pixels - state.image_background[:, None, None]).reshape(
                    len(settings.images), -1
                ),
                axis=1,
            )
            difference = float(np.max(numerator / denominator))
        row = {
            "refinement": refinement,
            "gaussians": len(numerical.centres),
            "seconds": perf_counter() - start,
            "worst_image_change": difference,
        }
        rows.append(row)
        print(row, flush=True)
        (root / "full_forward_refinement.json").write_text(json.dumps(rows, indent=2))
        np.savez_compressed(
            root / f"full_forward_refinement_{refinement}.npz",
            images=pixels,
            currents=currents,
        )
        previous = pixels
    return rows


def matched_exit_gaussians(
    *, template, detector, beam_coordinates, step_nm, half_width_nm=192.0
):
    """Known specimen exit wave as actual TemGym GaussianBeam objects.

    Preserve the earlier experiment's smoothed intensity specimen and Gaussian
    illumination, add the known MAGICAL phase, and represent that complex field
    with TemGym's banded cardinal-Gaussian solve. Curvature is carried by each
    Gaussian's Q matrix, slope and action. No Fourier propagation is used.
    """
    p = np.asarray(beam_coordinates)
    step = step_nm * 1e-9
    half = int(np.ceil(half_width_nm / step_nm))
    axis = np.arange(-half, half + 1) * step
    old = np.asarray(template["axis_m"])
    xx, yy = np.meshgrid(old, old)
    mean = p[:2] * 1e-7
    variance = np.exp(2 * p[4])
    kernel = detector["sample_kernel_sigma_m"] ** 2
    total = variance + kernel
    weights = np.asarray(template["intensity"]) * np.exp(
        -((xx - mean[0]) ** 2 + (yy - mean[1]) ** 2) / (2 * total)
    )
    weights /= weights.sum()
    product_variance = variance * kernel / total

    def gaussian(centre):
        locations = (variance * old + kernel * centre) / total
        return np.exp(
            -0.5 * (axis[:, None] - locations[None, :]) ** 2 / product_variance
        ) / np.sqrt(2 * np.pi * product_variance)

    intensity = gaussian(mean[1]) @ weights @ gaussian(mean[0]).T
    xx, yy = np.meshgrid(axis, axis)
    xy = np.stack((xx, yy), axis=-1)
    voltage = detector["voltage_v"]
    sample, _, roi = make_magical_sample(voltage)
    optical_path = np.asarray(
        sample.evaluate_actions(jnp.asarray(xy) + jnp.asarray(roi))
    )[..., 0]
    k = 2 * np.pi / float(energy2wavelength(voltage))
    target = np.sqrt(np.maximum(intensity, 0.0)) * np.exp(1j * k * optical_path)
    coefficients = _gaussian_cardinal_coefficients(target, width_over_step=1.1)
    keep = np.abs(coefficients) > np.max(np.abs(coefficients)) * 1e-12
    width = 1.1 * step
    curvature = p[5] * 1e4
    theta = p[2:4] * 1e-3 - curvature * mean
    position = xy[keep]
    slopes = curvature * position + theta
    phase = k * (0.5 * curvature * np.sum(position**2, axis=1) + position @ theta)
    source = make_gaussian(
        x=position[:, 0],
        y=position[:, 1],
        dx=slopes[:, 0],
        dy=slopes[:, 1],
        amp=coefficients[keep],
        phase=phase,
        waist_x=width,
        waist_y=width,
        rcurv_x=np.inf if curvature == 0 else 1 / curvature,
        rcurv_y=np.inf if curvature == 0 else 1 / curvature,
        voltage=voltage,
    )
    bound = float(np.abs(coefficients[~keep]).sum() * width * np.sqrt(np.pi / 2))
    return source, bound


def _recorded_gaussian_case(
    original_experiment_directory, voltage_index, condenser_setting,
    projector_view, detector_shape_yx, pixel_samples,
):
    """Shared physical setup for convergence checks and device benchmarks."""
    from condenser_image_pipeline import predict_beam_coordinates
    from condenser_voltage_experiment import (
        condenser_at_voltage,
        magnetic_strength_ratio,
    )
    from finite_projector_pipeline import projector_components

    original = Path(original_experiment_directory)
    folder = original / "condenser" / f"voltage_{voltage_index}"
    detector = json.loads((folder / "detector.json").read_text())
    template = dict(np.load(folder / "template.npz"))
    data = dict(np.load(folder / "images.npz"))
    voltage = detector["voltage_v"]
    # Generator/validation only; these values never seed a calibration fit.
    truth = np.asarray(
        json.loads((original / "condenser/scoring_truth.json").read_text())[
            "parameters"
        ]
    )
    effective = condenser_at_voltage(
        truth, voltage, source_parameters=truth[22:29] if voltage_index else None
    )
    beam = np.asarray(
        predict_beam_coordinates(effective, data["controls"][condenser_setting])
    )
    pp = np.asarray(
        json.loads((original / "projector/scoring_truth.json").read_text())[
            "parameters"
        ]
    )
    ratio = float(magnetic_strength_ratio(voltage, 200000.0))
    pp[11] *= ratio
    pp[12:16] += np.log(ratio)
    column = tuple(
        replace(c, include_axial_path=False) if hasattr(c, "include_axial_path") else c
        for c in projector_components(pp, data["projector_controls"][projector_view])
    )
    matrix = np.asarray(column_matrix(column))
    sensor = CoherentDetector(
        tuple(detector_shape_yx),
        detector["pixel_pitch_m"],
        float(column[-1].z),
        pixel_samples,
        0.0,
    )
    floor = float(energy2wavelength(voltage)) / (4 * np.pi * np.exp(beam[4]))
    extra2 = np.exp(2 * beam[6]) - floor**2
    if extra2 < 0:
        raise ValueError("The input covariance is below the coherent emittance limit.")
    blur = np.sqrt(
        extra2 * np.sum(matrix[:2, 2] ** 2)
        + (detector["psf_sigma_pixels"] * detector["pixel_pitch_m"]) ** 2
    )
    return template, detector, beam, column, sensor, blur


def check_recorded_gaussian_forward(
    *,
    original_experiment_directory,
    output_directory,
    voltage_index=0,
    condenser_setting=3,
    projector_view=2,
    steps_nm=(2.0, 1.0, 0.5),
    method="scan",
    detector_shape_yx=(8, 8),
    pixel_samples=8,
):
    """Direct TemGym propagation at the recorded difficult near-crossover view.

    Check Gaussian count, detector quadrature and source/detector padding with
    physical geometry, source moments, specimen and detector unchanged. The
    Gaussian angular ensemble is a real-space blur of the coherent sum.
    This is one forward check, not a full coherent microscope fit.
    """
    from temgym_core.coherent_image import blurred_gaussian_image
    from temgym_core.run import run_to_end

    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    template, detector, beam, column, sensor, blur = _recorded_gaussian_case(
        original_experiment_directory, voltage_index, condenser_setting,
        projector_view, detector_shape_yx, pixel_samples,
    )
    voltage = detector["voltage_v"]
    records = []
    images = []
    transport = jax.jit(
        lambda source: run_to_end(
            source, column, propagator=FreeSpacePropagator(include_axial_path=False)
        )
    )

    def render(source, samples=pixel_samples, padding=4):
        output = transport(source)
        return np.asarray(
            blurred_gaussian_image(
                output,
                replace(sensor, samples_per_pixel=samples),
                blur_sigma_m=blur,
                padding_pixels=padding,
                method=method,
            )
        )

    for step in steps_nm:
        start = perf_counter()
        source, bound = matched_exit_gaussians(
            template=template, detector=detector, beam_coordinates=beam, step_nm=step
        )
        image = render(source)
        images.append(image)
        record = {
            "step_nm": step,
            "gaussians": int(np.size(source.x)),
            "seconds": perf_counter() - start,
            "omitted_field_l2_bound": bound,
        }
        records.append(record)
        print(
            f"Direct Gaussian check: {record['gaussians']:,} Gaussians, {record['seconds']:.1f} s",
            flush=True,
        )
    # Dose and noise units match the original condenser acquisition.
    noise = np.sqrt(4e7 * images[-1] + 3.0 + detector["read_noise_e"] ** 2)

    def error(a, b):
        return float(np.sqrt(np.mean((4e7 * (a - b) / noise) ** 2)))

    for i in range(1, len(images)):
        records[i]["change_rms_noise"] = error(images[i], images[i - 1])
    fine_pixels = render(source, samples=2 * pixel_samples)
    padded_detector = render(source, padding=6)
    source_wide, _ = matched_exit_gaussians(
        template=template,
        detector=detector,
        beam_coordinates=beam,
        step_nm=steps_nm[-1],
        half_width_nm=256.0,
    )
    identical_source = all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(jax.tree.leaves(source), jax.tree.leaves(source_wide))
    )
    padded_source = images[-1] if identical_source else render(source_wide)
    checks = {
        "gaussian_change_rms_noise": records[-1].get("change_rms_noise", float("inf")),
        "pixel_change_rms_noise": error(fine_pixels, images[-1]),
        "detector_padding_change_rms_noise": error(padded_detector, images[-1]),
        "source_padding_change_rms_noise": error(padded_source, images[-1]),
    }
    report = {
        "backend": method,
        "device": jax.devices()[0].device_kind,
        "jax_backend": jax.default_backend(),
        "uses_fft_propagation": False,
        "voltage_v": voltage,
        "condenser_setting": condenser_setting,
        "projector_view": projector_view,
        "refinements": records,
        "checks": checks,
        "passed": bool(max(checks.values()) < 0.2),
        "detector_shape": list(sensor.shape_yx),
        "pixel_pitch_m": sensor.pixel_pitch_m,
        "pixel_samples_per_axis": pixel_samples,
        "refined_pixel_samples_per_axis": 2 * pixel_samples,
        "source_padding_produced_identical_gaussians": identical_source,
        "angular_ensemble": "Real-space Gaussian average after coherent field summation.",
        "full_detector_shape": detector["shape_yx"],
        "scope": "Central detector region at one recorded near-crossover setting; no full-image or coherent geometry recovery claimed.",
    }
    np.savez_compressed(
        root / "direct_images.npz",
        images=images,
        fine_pixels=fine_pixels,
        padded_source=padded_source,
        padded_detector=padded_detector,
    )
    (root / "direct_gaussian_check.json").write_text(json.dumps(report, indent=2))
    return report


def benchmark_recorded_gaussian_forward(
    *, original_experiment_directory, output_directory,
    methods=("gpu_kernel_required", "spatial", "scan"),
    step_nm=0.5, detector_shape_yx=(8, 8), pixel_samples=8,
    repeats=3, check_tilt_derivative=False, tile_beams=32,
    detector_pixel_pitch_m=None, padding_pixels=4,
    projector_view=2,
):
    """Time current TemGym renderers with identical optics and sampling.

    Separates compilation, source construction and warmed evaluations. Every
    timed call waits for device completion. The optional derivative is one
    specimen-tilt JVP through propagation and the scan image renderer, not a
    full microscope Jacobian. Source discretization stays fixed.
    Override detector pitch to compare different pixel counts over the same
    area; the physical source, optics and detector blur remain unchanged.
    """
    import hashlib
    import inspect

    import temgym_core.evaluate as evaluator
    import temgym_core.spatial_evaluate as spatial
    from temgym_core.coherent_image import blurred_gaussian_image
    from temgym_core.run import run_to_end

    if repeats < 1 or not methods:
        raise ValueError("Supply render methods and at least one repeat.")
    # The probe executes and checks a real kernel; run it outside a JIT trace.
    available = gpu_kernel_available()
    if "gpu_kernel_required" in methods and not available:
        raise RuntimeError("The verified GPU Gaussian kernel is unavailable.")
    template, detector, beam, column, sensor, blur = _recorded_gaussian_case(
        original_experiment_directory, 0, 3, projector_view, detector_shape_yx, pixel_samples,
    )
    if detector_pixel_pitch_m is not None:
        sensor = replace(sensor, pixel_pitch_m=detector_pixel_pitch_m)
    start = perf_counter()
    source, bound = matched_exit_gaussians(
        template=template, detector=detector, beam_coordinates=beam, step_nm=step_nm,
    )
    jax.block_until_ready(source)
    construction_seconds = perf_counter() - start
    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for obj in (evaluator, spatial, blurred_gaussian_image, matched_exit_gaussians):
        path = Path(inspect.getfile(obj)).resolve()
        files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "device": jax.devices()[0].device_kind,
        "jax_backend": jax.default_backend(), "jax_version": jax.__version__,
        "verified_gpu_kernel": available,
        "x64": jax.config.x64_enabled, "source_files_sha256": files,
        "gaussians": int(source.x.size), "step_nm": step_nm,
        "projector_view": projector_view,
        "detector_shape": list(detector_shape_yx), "pixel_samples": pixel_samples,
        "tile_beams": tile_beams,
        "pixel_pitch_m": sensor.pixel_pitch_m,
        "physical_blur_sigma_m": float(blur),
        "padding_pixels": padding_pixels,
        "source_construction_seconds": construction_seconds,
        "omitted_field_l2_bound": bound, "repeats": repeats, "timings": {},
        "scope": "One prescribed specimen exit wave; no full microscope fit.",
    }
    images = {}

    def save():
        (root / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
        if images:
            np.savez_compressed(root / "images.npz", **images)

    def timed(name, function, argument):
        start = perf_counter()
        value = jax.block_until_ready(function(argument))
        first = perf_counter() - start
        times = []
        for _ in range(repeats):
            start = perf_counter()
            value = jax.block_until_ready(function(argument))
            times.append(perf_counter() - start)
        report["timings"][name] = {
            "first_call_seconds": first, "warm_seconds": times,
            "warm_median_seconds": float(np.median(times)),
        }
        print(name, json.dumps(report["timings"][name]), flush=True)
        save()
        return value

    transport = jax.jit(lambda b: run_to_end(
        b, column, propagator=FreeSpacePropagator(include_axial_path=False),
    ))
    output = timed("transport", transport, source)
    bins = spatial._build_bins(
        output.r_xy, output.Q_inv, output.k, cutoff=1e-12, bins_per_axis=64,
    )
    report["spatial_uses_dense_fallback"] = bool(bins.use_dense)
    reference = None
    for method in methods:
        render = jax.jit(lambda b, selected_method=method: blurred_gaussian_image(
            b, sensor, blur_sigma_m=blur, method=selected_method, tile_beams=tile_beams,
            padding_pixels=padding_pixels,
        ))
        image = np.asarray(timed(method, render, output))
        images[method] = image
        if reference is None:
            reference = image
        sigma = np.sqrt(4e7 * reference + 3 + detector["read_noise_e"] ** 2)
        report["timings"][method]["difference_rms_noise"] = float(
            np.sqrt(np.mean((4e7 * (image - reference) / sigma) ** 2))
        )
        save()

    if check_tilt_derivative:
        def predict(tilt_mrad, input_beam):
            angle = tilt_mrad * 1e-3
            tilted = input_beam.derive(
                dx=input_beam.dx + angle,
                amplitude=input_beam.amplitude * jnp.exp(
                    1j * input_beam.k * angle * input_beam.x
                ),
            )
            return blurred_gaussian_image(
                transport(tilted), sensor, blur_sigma_m=blur, method="scan",
                padding_pixels=padding_pixels,
            )

        value = jnp.asarray(0.0)
        derivative = jax.jit(lambda args: jax.jvp(
            lambda t: predict(t, args[1]), (args[0],), (jnp.ones_like(args[0]),),
        )[1])
        actual = np.asarray(timed("tilt_jvp_scan", derivative, (value, source)))
        predict = jax.jit(predict)
        step = 1e-5
        finite = np.asarray(
            (predict(value + step, source) - predict(value - step, source)) / (2 * step)
        )
        relative = float(np.linalg.norm(actual - finite) / np.linalg.norm(finite))
        report["tilt_derivative"] = {
            "finite_difference_step_mrad": step,
            "relative_l2_error": relative,
            "passed": bool(np.isfinite(relative) and relative < 1e-3),
        }
        save()
    return report


def show_recorded_gaussian_check(*, report):
    """Small notebook tables; full numerical details remain in the report."""
    from IPython.display import Markdown, display

    rows = []
    for row in report["refinements"]:
        change = row.get("change_rms_noise")
        change = "—" if change is None else f"{change:.4g}"
        rows.append(f"| {row['step_nm']:g} | {row['gaussians']:,} | {change} |")
    display(
        Markdown(
            "| Spacing (nm) | Gaussians | Change in noise units |\n|---:|---:|---:|\n"
            + "\n".join(rows)
        )
    )
    rows = [
        f"| {name.replace('_', ' ')} | {value:.4g} |"
        for name, value in report["checks"].items()
    ]
    display(
        Markdown("| Check | Change in noise units |\n|---|---:|\n" + "\n".join(rows))
    )
    print("Forward check passed:", report["passed"])
