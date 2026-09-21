"""Cache integrity and the recorded-image fitting boundary."""

import json

import numpy as np
import pytest
from scipy.special import ndtr

from temgym_core.calibration.inference import KICK_UNITS, measure_beam_images
from temgym_core.calibration.workflow import _record_step, input_signature


def test_two_magnification_acquisition_keeps_calibrated_pairs_and_fixed_illumination(
    tmp_path, monkeypatch
):
    from temgym_core.calibration import acquisition

    def fixed_objective_views(plan, objective):
        commands = np.asarray([view["controls"] for view in plan["views"]]).copy()
        commands[:, 0] = objective
        return None, commands

    monkeypatch.setattr(
        acquisition, "choose_objective_projector_views", fixed_objective_views
    )
    acquisition.prepare_experiment(
        directory=tmp_path,
        voltages_v=(200000.0, 195000.0),
        aperture_diameters_um=(4.0, 10.0),
        random_seed=24001,
    )
    settings = np.load(tmp_path / "settings.npz")
    assert sum(settings["purpose"] == "projector") * 2 == 288
    assert sum(settings["purpose"] == "condenser") * 2 == 128
    assert sum(settings["purpose"] == "check") * 2 == 24
    assert set(settings["nominal_magnification"]) == {40000.0, 45000.0}
    for voltage in range(2):
        selected = (settings["voltage_index"] == voltage) & (
            settings["purpose"] == "projector"
        )
        assert len(np.unique(settings["condenser_controls"][selected], axis=0)) == 1
    for group in np.unique(settings["response_group"]):
        if group < 0:
            continue
        rows = np.flatnonzero(settings["response_group"] == group)
        assert len(rows) == 2
        plus, minus = settings["specimen_kick"][rows]
        np.testing.assert_array_equal(plus, -minus)
        np.testing.assert_array_equal(
            settings["projector_controls"][rows[0]],
            settings["projector_controls"][rows[1]],
        )
        assert len(np.unique(settings["voltage_index"][rows])) == 1
        assert len(np.unique(settings["lens_change"][rows])) == 1
    import jax
    import jax.numpy as jnp

    from temgym_core.calibration.apertures import effective_parameters
    from temgym_core.calibration.condenser_model import first_crossover_fraction

    truth = json.loads((tmp_path / "scoring_truth.json").read_text())
    for vi, voltage in enumerate([200000.0, 195000.0]):
        parameters = effective_parameters(
            jnp.asarray(truth["condenser_parameters"]), voltage, 200000.0, vi
        )
        fractions = jax.vmap(first_crossover_fraction, in_axes=(None, 0))(
            parameters, settings["condenser_controls"][settings["voltage_index"] == vi]
        )
        assert np.all((fractions >= 0.01) & (fractions <= 0.99))
    assert np.all(settings["projector_controls"][:, 0] == 1.0)
    spec = json.loads((tmp_path / "experiment.json").read_text())
    assert spec["nominal_incident_electrons_per_image"] == 1e8
    assert spec["aperture_inserted_for_every_image"]


def test_stage_reuse_checks_inputs_and_required_outputs(tmp_path):
    output = tmp_path / "result.txt"
    calls = []

    def run():
        calls.append(1)
        output.write_text("result")

    def stage(signature):
        _record_step(
            directory=tmp_path,
            name="fitting",
            signature=signature,
            required=[output],
            function=run,
            reuse_completed=True,
        )

    stage("first")
    stage("first")
    assert len(calls) == 1
    stage("changed-input")
    assert len(calls) == 2
    output.unlink()
    stage("changed-input")
    assert len(calls) == 3
    output.unlink()
    with pytest.raises(RuntimeError, match="required outputs"):
        _record_step(
            directory=tmp_path,
            name="fitting",
            signature="failed",
            required=[output],
            function=lambda: None,
            reuse_completed=False,
        )
    assert not (tmp_path / "fitting_manifest.json").exists()


def test_cache_fingerprint_uses_array_contents():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        a, b = Path(directory) / "a.npz", Path(directory) / "b.npz"
        np.savez(a, x=[1, 2])
        np.savez_compressed(b, x=[1, 2])
        assert input_signature(paths=[a], settings={}) == input_signature(
            paths=[b], settings={}
        )
        np.savez(b, x=[1, 3])
        assert input_signature(paths=[a], settings={}) != input_signature(
            paths=[b], settings={}
        )


@pytest.mark.parametrize(
    "axes, rotated", [([0, 1, 2, 3], False), ([0, 2], False), ([0, 2], True)]
)
def test_measurements_recover_signed_responses_without_reading_truth(
    tmp_path, axes, rotated
):
    spec = {"background_counts": 3.0, "read_noise_counts": 2.0}
    (tmp_path / "experiment.json").write_text(json.dumps(spec))
    (tmp_path / "scoring_truth.json").write_text(
        "Do not read generating parameters during image measurement."
    )
    kicks = np.r_[np.eye(4)[axes], -np.eye(4)[axes]]
    if rotated:
        rotation = np.array([[0.8, -0.6], [0.6, 0.8]])
        kicks[:, :2] = kicks[:, :2] @ rotation
        kicks[:, 2:] = kicks[:, 2:] @ rotation
    response = np.array([[2.0, 0.2, 3.0, -0.5], [-0.3, 2.2, 0.8, 2.1]])
    images = []
    edges = np.arange(-32, 33)
    for kick in kicks:
        centre = response @ kick + np.array([0.5, -0.2])
        pair = []
        for width, dose in [(2.8, 3e5), (4.2, 7e5)]:
            px = np.diff(ndtr((edges - centre[0]) / width))
            py = np.diff(ndtr((edges - centre[1]) / width))[::-1]
            pair.append(dose * py[:, None] * px[None, :] + 3.0)
        images.append(pair)
    np.savez_compressed(tmp_path / "images.npz", images=images)
    np.savez(
        tmp_path / "settings.npz",
        specimen_kick=kicks * KICK_UNITS,
        response_group=np.zeros(len(kicks), dtype=int),
        projector_controls=np.ones((len(kicks), 5)),
        voltage_index=np.zeros(len(kicks), dtype=int),
    )
    measure_beam_images(directory_path=tmp_path)
    result = np.load(tmp_path / "projector_responses.npz")
    np.testing.assert_allclose(
        result["values"],
        (response @ (result["specimen_kicks"] / KICK_UNITS).T).T,
        atol=1e-8,
    )
    assert np.linalg.eigvalsh(result["covariance"]).min() > 0
    measured = np.load(tmp_path / "measurements.npz")
    assert measured["values"].shape == (len(kicks), 2, 6)
    assert measured["covariance"].shape == (len(kicks), 2, 6, 6)


def test_numerical_cache_ignores_presentation_changes(tmp_path, monkeypatch):
    from temgym_core.calibration import workflow

    package = tmp_path / "calibration"
    package.mkdir()
    monkeypatch.setattr(workflow, "__file__", str(package / "workflow.py"))
    numerical = package / "gaussian.py"
    numerical.write_text("physical model")
    before = input_signature(paths=[], settings={})
    for path in [
        package / "viewer.py",
        package / "workflow.py",
        tmp_path / "plotting.py",
    ]:
        path.write_text("changed presentation")
    assert input_signature(paths=[], settings={}) == before
    numerical.write_text("different physical model")
    assert input_signature(paths=[], settings={}) != before


@pytest.mark.parametrize("data_kind", ["images", "moments", "recorded"])
def test_browser_keeps_saved_images_and_selected_settings_together(
    tmp_path, monkeypatch, data_kind
):
    pytest.importorskip("ipywidgets")
    pytest.importorskip("ipympl")
    import matplotlib.pyplot as plt

    previous_backend = plt.get_backend()
    plt.switch_backend("module://ipympl.backend_nbagg")

    from temgym_core.calibration import viewer

    spec = {
        "voltages_v": [200000.0, 195000.0],
        "aperture_diameters_um": [4.0, 10.0],
        "pixel_pitch_m": 1e-4,
        "shape_yx": [8, 8],
    }
    (tmp_path / "experiment.json").write_text(json.dumps(spec))
    if data_kind != "recorded":
        (tmp_path / "scoring_truth.json").write_text("{}")
    frames = np.arange(4).reshape(2, 2, 1, 1) * np.ones((2, 2, 8, 8)) + 10
    moments = np.broadcast_to([1.0, 2.0, 3.0, 0.5, 4.0, 10000.0], (2, 2, 6)).copy()
    moments[..., 0] += np.arange(4).reshape(2, 2)
    if data_kind == "recorded":
        frames[0, 0] = np.nan  # Only the larger aperture was collected here.
    if data_kind != "moments":
        np.savez(tmp_path / "images.npz", images=frames)
    else:
        np.savez(tmp_path / "measurements.npz", values=moments)
    np.savez(
        tmp_path / "settings.npz",
        voltage_index=[0, 1],
        condenser_controls=np.ones((2, 3)),
        projector_controls=np.ones((2, 5)),
        purpose=["projector", "condenser"],
        specimen_kick=np.array([[20e-9, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
        nominal_magnification=[40000.0, 60000.0],
        lens_change=["Baseline", "CL1 +0.10%"],
    )
    traces = []

    def diagram(spec, settings, model, row, aperture):
        traces.append((row, aperture))
        return (), []

    monkeypatch.setattr(viewer, "_ray_diagram", diagram)
    monkeypatch.setattr(viewer, "_draw_ray_section", lambda *args: None)
    previews = []

    def preview(spec, settings, model, row, aperture):
        previews.append((row, aperture))
        return frames[row, aperture]

    monkeypatch.setattr(viewer, "render_ray_image", preview)
    browser = viewer.browse_images(experiment_directory=tmp_path)
    if data_kind == "recorded":
        assert list(browser.children[4].children[1].options) == ["Initial guess"]
    canvas = browser.children[5]
    fig = canvas.figure

    def assert_displayed(ri, ai):
        np.testing.assert_array_equal(fig.axes[0].images[0].get_array(), frames[ri, ai])
        if data_kind == "moments":
            assert "preview" in fig.axes[0].get_title()
            assert "not these pixels" in browser.children[0].value

    purpose, voltage, magnification, change, beam, aperture = browser.children[
        1
    ].children
    contrast = browser.children[4].children[2]
    voltage.value = 1
    aperture.value = 1
    assert_displayed(1, 1)
    assert traces[-1] == (1, 1)
    purpose.value = "projector"
    assert browser.children[2].children[1].disabled
    voltage.value = 0
    assert_displayed(0, 1)
    assert traces[-1] == (0, 1)
    fig.axes[0].set_xlim(-0.1, 0.1)
    fig.axes[1].set_xlim(-0.2, 0.2)
    trace_count = len(traces)
    contrast.value = False
    assert browser.children[5] is canvas
    assert len(traces) == trace_count
    assert fig.axes[0].get_xlim() == (-0.1, 0.1)
    assert fig.axes[1].get_xlim() == (-0.2, 0.2)
    assert len(fig.axes[0].images) == 1
    assert len(previews) == (4 if data_kind == "moments" else 0)
    purpose.value = "all"
    voltage.value = 1
    magnification.value = 60000.0
    change.value = "CL1 +0.10%"
    beam.value = "No steering"
    assert_displayed(1, 1)
    assert [value for _, value in beam.options] == ["all", "No steering"]
    change.value = "Baseline"
    assert browser.children[2].children[1].disabled
    assert beam.value == "all"
    assert list(beam.options) == [("All inputs", "all")]
    magnification.value = -1
    voltage.value = 0
    assert [value for _, value in beam.options] == ["all", "Shift X"]
    assert_displayed(0, 1)
    plt.close(fig)
    browser.close()
    plt.switch_backend(previous_backend)


def test_ray_preview_weights_aperture_and_bins_native_detector_coordinates():
    from temgym_core.calibration.acquisition import _ray_renderer
    from temgym_core.components import Plane, SigmoidAperture
    from temgym_core.ray import Ray

    rays = Ray(
        x=np.array([0.0, 1.0, 20.0]),
        y=np.array([0.0, 1.0, 0.0]),
        dx=0.0,
        dy=0.0,
        z=0.0,
        pathlength=0.0,
    )
    aperture = SigmoidAperture(z=0.0, radius=2.0, edge_width=0.2)
    image = np.asarray(_ray_renderer((9, 9))((aperture, Plane(z=1.0)), rays, 1.0))
    weights = np.exp(2 * np.asarray(aperture.log_transmission(rays.r_xy))) / 3
    expected = np.zeros((9, 9))
    expected[4, 4], expected[3, 5] = weights[:2]
    np.testing.assert_allclose(image, expected, atol=1e-14)
    # Aperture attenuation is squared amplitude; outside-camera rays are lost.
    np.testing.assert_allclose(image.sum(), weights[:2].sum())


def test_geometric_aperture_option_preserves_flux_and_mean():
    from temgym_core.calibration.apertures import make_aperture_moments
    from temgym_core.calibration.condenser_model import generating_parameters

    parameters = np.r_[
        generating_parameters(), np.log(0.0025 / (0.056 - 0.0025)), 0.2, -0.15
    ]
    outputs = [
        make_aperture_moments(
            radius_m=2e-6,
            edge_width_m=0.15e-6,
            voltage_v=200000.0,
            radial_order=96,
            angular_order=32,
            diffraction=diffraction,
        )(parameters, np.array([1.0, 1.0, 1.0]), np.array([0.2e-6, -0.15e-6]))
        for diffraction in (False, True)
    ]
    np.testing.assert_allclose(outputs[0][0], outputs[1][0], atol=1e-15)
    np.testing.assert_allclose(outputs[0][2], outputs[1][2], rtol=1e-12)
    assert np.linalg.norm(np.asarray(outputs[1][1] - outputs[0][1])) > 0


def test_detector_subpixels_reduce_ray_rounding_without_changing_exposure(monkeypatch):
    from temgym_core.calibration import acquisition
    from temgym_core.components import Plane
    from temgym_core.moments import measure_detector_moments

    centre = np.array([0.31, 0.17])
    monkeypatch.setattr(
        acquisition,
        "_column_source",
        lambda *args: (
            (Plane(z=1.0),),
            np.r_[centre, 0.0, 0.0],
            np.eye(4) * 1e-16,
            0.0,
            200000.0,
        ),
    )
    spec = {
        "shape_yx": [32, 32],
        "pixel_pitch_m": 1.0,
        "psf_sigma_pixels": 1.0,
        "nominal_incident_electrons_per_image": 1000.0,
        "ray_sample_power": 3,
    }
    errors = []
    for sampling in (1, 8):
        frame = acquisition.render_ray_image(
            dict(spec, detector_samples_per_pixel=sampling), {}, {}, 0, 0
        )
        measured = measure_detector_moments(
            images=frame[None], background_counts=0.0, read_noise_counts=0.0
        )
        np.testing.assert_allclose(frame.sum(), 1000.0, rtol=1e-12)
        assert frame.shape == (32, 32)
        errors.append(np.linalg.norm(measured["values"][0, :2] - centre))
    assert errors[1] < errors[0] / 5


def test_double_deflector_alignment_preserves_the_calibrated_specimen_response():
    from dataclasses import replace

    import jax
    import jax.numpy as jnp

    from temgym_core.calibration.condenser_model import (
        condenser_with_aperture,
        condenser_with_steering,
        generating_parameters,
    )
    from temgym_core.components import DoubleDeflector
    from temgym_core.ray import Ray
    from temgym_core.run import column_matrix, run_to_end

    cp = np.r_[generating_parameters(), -3.0, 0.2, -0.15]
    column = condenser_with_aperture(
        cp, np.ones(3), radius_m=2e-6, edge_width_m=150e-9, centre_m=cp[20:22] * 1e-6
    )
    source_z = -np.exp(cp[:9]).sum()
    base = np.asarray(column_matrix(column, source_z=source_z))
    kick = np.array([20e-9, -10e-9, 3e-3, -1e-3])
    physical_kicks = []
    for spacing_mm in (10.0, 22.0, 40.0):
        half_spacing = spacing_mm * 0.5e-3 / float(column[3].z - column[2].z)
        fractions = (0.5 - half_spacing, 0.5 + half_spacing)
        steered = condenser_with_steering(
            column, kick, voltage_v=200000.0, gap_fractions=fractions
        )
        # Source-to-aperture optics are identical; both steerers are downstream.
        assert steered[:3] == column[:3]
        pair = steered[3]
        assert isinstance(pair, DoubleDeflector)
        assert column[2].z < pair.z < pair.z_second < column[3].z
        result = np.asarray(column_matrix(steered, source_z=source_z))
        np.testing.assert_allclose(result[:4, :4], base[:4, :4], rtol=2e-5, atol=1e-9)
        np.testing.assert_allclose(
            result[:4, 4] - base[:4, 4], kick, rtol=2e-4, atol=2e-9
        )
        physical_kicks.append(
            [(float(c.def_x), float(c.def_y)) for c in pair.deflectors]
        )

        def response(commands, pair=pair, downstream=steered[4:]):
            driven = replace(
                pair,
                shift_x=commands[0],
                shift_y=commands[1],
                tilt_x=commands[2],
                tilt_y=commands[3],
            )
            ray = run_to_end(Ray.origin().derive(z=pair.z), (driven, *downstream))
            return jnp.stack((ray.x, ray.y, ray.dx, ray.dy))

        matrix = np.asarray(jax.jacfwd(response)(jnp.zeros(4)))
        # Both transverse axes must be decoupled despite solenoid rotation.
        np.testing.assert_allclose(matrix[2:, :2], 0, atol=2e-7)
        np.testing.assert_allclose(matrix[:2, 2:], 0, atol=2e-9)
        assert np.linalg.matrix_rank(matrix) == 4
    # Location cannot be identified when only the final calibrated kick is kept.
    assert not np.allclose(physical_kicks[0], physical_kicks[-1])


def test_feasibility_constraint_is_not_counted_as_measured_information(tmp_path):
    import jax.numpy as jnp

    from temgym_core.calibration.inference import run_starts

    result = run_starts(
        lambda x: jnp.array([x[0] - 0.5, 2 * (x[0] - 0.5)]),
        [[1.8]],
        ([0.0], [2.0]),
        output_path=tmp_path / "starts.json",
        constraints=lambda x: jnp.maximum(x - 1.0, 0.0),
    )
    assert result["best"]["acceptable"]
    assert result["best"]["degrees_of_freedom"] == 1
    np.testing.assert_allclose(result["covariance"], [[0.2]], rtol=1e-10)
    rejected = run_starts(
        lambda x: jnp.array([x[0] - 0.5, 2 * (x[0] - 0.5)]),
        [[0.5]],
        ([0.0], [2.0]),
        output_path=tmp_path / "rejected.json",
        constraints=lambda x: jnp.ones(1),
    )
    assert not rejected["best"]["acceptable"]
    assert rejected["best"]["constraint_max_violation"] == 1.0


def test_direct_measurements_and_checks_need_no_images(tmp_path):
    import jax.numpy as jnp

    from temgym_core.calibration import check_microscope, simulate_measurements
    from temgym_core.calibration.condenser_model import generating_parameters
    from temgym_core.calibration.inference import PRESETS
    from temgym_core.calibration.projector_model import (
        generating_parameters as projector_parameters,
    )
    from temgym_core.calibration.projector_model import (
        projector_coordinates,
    )

    reference, output = tmp_path / "reference", tmp_path / "moments"
    reference.mkdir()
    cp = np.r_[generating_parameters(), -3.0, 0.2, -0.15]
    cp = np.r_[cp, cp[0], cp[13:19], 0.12, -0.08]
    pp = projector_parameters()
    spec = {
        "voltages_v": [200000.0, 195000.0],
        "aperture_diameters_um": [4.0, 10.0],
        "aperture_edge_width_m": 150e-9,
        "pixel_pitch_m": 110e-6,
        "psf_sigma_pixels": 1.0,
        "nominal_incident_electrons_per_image": 1e8,
    }
    (reference / "experiment.json").write_text(json.dumps(spec))
    (reference / "scoring_truth.json").write_text(
        json.dumps(
            {
                "projector_parameters": pp.tolist(),
                "condenser_parameters": cp.tolist(),
            }
        )
    )
    np.savez(
        reference / "settings.npz",
        voltage_index=[0, 0, 1, 1],
        condenser_controls=np.ones((4, 3)),
        projector_controls=np.ones((4, 5)),
        specimen_kick=np.array(
            [[20e-9, 0, 0, 0], [-20e-9, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        ),
        response_group=[0, 0, -1, -1],
        purpose=["projector", "projector", "check", "check"],
    )
    covariance = np.broadcast_to(np.eye(6), (4, 2, 6, 6)).copy()
    np.savez(reference / "measurements.npz", covariance=covariance)
    # No reference pixels are required, nor is an image renderer called.
    simulate_measurements(reference_directory=reference, output_directory=output)
    exact = np.load(output / "measurements.npz")["values"].copy()
    assert not (output / "images.npz").exists()
    np.testing.assert_array_equal(
        np.load(output / "measurements.npz")["covariance"], covariance
    )
    assert np.all(exact[..., 5] > 0)
    encode, _, _, _ = projector_coordinates(PRESETS / "projector_position_priors.json")
    parameters = np.r_[encode(jnp.asarray(pp))[:16], cp]
    checks = check_microscope(
        experiment_directory=output,
        render_images=False,
        microscope={
            "solver_parameters": parameters.tolist(),
            "covariance": np.eye(47).tolist(),
            "best": {"rank": 47, "residual_rms": 0.0},
        },
    )
    assert not checks["image_predictions_checked"]
    assert checks["prediction_pixel_rms"] == []
    assert checks["prediction_noise_only_rms"] < 1e-8
    assert all(abs(row["error"]) < 1e-10 for row in checks["geometry"])
    simulate_measurements(
        reference_directory=reference,
        output_directory=output,
        add_noise=True,
        random_seed=7,
    )
    noisy = np.load(output / "measurements.npz")["values"].copy()
    np.testing.assert_allclose(
        noisy - exact, np.random.default_rng(7).normal(size=exact.shape), atol=1e-6
    )
    with pytest.raises(ValueError, match="separate"):
        simulate_measurements(reference_directory=reference, output_directory=reference)


def test_ray_detector_blur_preserves_calibrated_variance(monkeypatch):
    """A narrow, resolved beam exposes PSF truncation that more rays cannot fix."""
    from temgym_core.calibration import acquisition
    from temgym_core.components import Plane
    from temgym_core.moments import measure_detector_moments

    monkeypatch.setattr(
        acquisition,
        "_column_source",
        lambda *args: (
            (Plane(z=1.0),),
            np.array([0.31, 0.17, 0.0, 0.0]),
            np.diag([0.2**2, 0.2**2, 1e-16, 1e-16]),
            0.0,
            200000.0,
        ),
    )
    spec = {
        "shape_yx": [32, 32],
        "pixel_pitch_m": 1.0,
        "psf_sigma_pixels": 1.0,
        "nominal_incident_electrons_per_image": 1e8,
        "ray_sample_power": 18,
        "detector_samples_per_pixel": 8,
    }
    image = acquisition.render_ray_image(spec, {}, {}, 0, 0)
    measured = measure_detector_moments(
        images=image[None], background_counts=0.0, read_noise_counts=0.0
    )
    expected_variance = 0.2**2 + 1.0 + 1 / 12
    np.testing.assert_allclose(
        measured["values"][0, [2, 4]], expected_variance, atol=8e-5, rtol=0
    )
    np.testing.assert_allclose(image.sum(), 1e8, rtol=1e-12)


@pytest.mark.parametrize('location', [(80, 80), (0, 0), None])
def test_ray_blur_crop_matches_full_detector(monkeypatch, location):
    from scipy.ndimage import gaussian_filter
    from temgym_core.calibration import acquisition
    from temgym_core.components import Plane

    sampling = 4
    histogram = np.zeros((256, 256))
    if location is not None:
        histogram[location] = 1.0
    monkeypatch.setattr(acquisition, '_ray_renderer', lambda shape: lambda *args: histogram)
    monkeypatch.setattr(acquisition, '_column_source', lambda *args: (
        (Plane(z=1.0),), np.zeros(4), np.eye(4), 0.0, 200000.0
    ))
    spec = dict(shape_yx=[64, 64], pixel_pitch_m=1.0, psf_sigma_pixels=1.0,
                nominal_incident_electrons_per_image=1000.0, ray_sample_power=3,
                detector_samples_per_pixel=sampling)
    actual = acquisition.render_ray_image(spec, {}, {}, 0, 0)
    expected = gaussian_filter(histogram, sampling, mode='constant', truncate=8.0)
    expected = expected.reshape(64, sampling, 64, sampling).sum(axis=(1, 3)) * 1000
    # FFT and direct summation differ only at floating-point round-off.
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-13)


@pytest.mark.parametrize('aperture_um', [20, 50])
@pytest.mark.parametrize('sparse', [False, True])
def test_single_projector_aperture_excludes_other_pixels_and_response(
    tmp_path, monkeypatch, aperture_um, sparse
):
    import jax.numpy as jnp
    from temgym_core.calibration import inference

    ai = [20, 50].index(aperture_um)
    spec = dict(voltages_v=[200000, 195000], aperture_diameters_um=[20, 50],
                pixel_pitch_m=110e-6, measurement_noise_added=True)
    (tmp_path / 'experiment.json').write_text(json.dumps(spec))
    np.savez(tmp_path / 'settings.npz', purpose=['projector', 'projector', 'condenser', 'check'],
             voltage_index=[0, 0, 1, 1], condenser_controls=np.ones((4, 3)),
             projector_controls=np.ones((4, 5)), specimen_kick=np.zeros((4, 4)))
    values = np.zeros((4, 2, 6))
    values[..., 5] = 1
    values[:2, 1 - ai] = 1e9  # Deliberately unusable observations from the other aperture.
    mask = np.ones((4, 2), bool)
    if sparse:
        mask[:2, 1 - ai] = False
    np.savez(tmp_path / 'measurements.npz', values=values, recorded=mask,
             covariance=np.broadcast_to(np.eye(6), (4, 2, 6, 6)))
    responses = np.zeros((2, 2))
    responses[1 - ai] = 1e9
    np.savez(tmp_path / 'projector_responses.npz', values=responses,
             covariance=np.broadcast_to(np.eye(2), (2, 2, 2)),
             controls=np.ones((2, 5)), voltage_index=[0, 0], specimen_kicks=np.zeros((2, 4)))
    if sparse:
        data = dict(np.load(tmp_path / 'projector_responses.npz'))
        np.savez(tmp_path / 'projector_responses.npz', aperture_index=[ai],
                 **{k: v[ai:ai + 1] for k, v in data.items()})
    prediction = jnp.zeros((3, 2, 6)).at[..., 5].set(1)
    monkeypatch.setattr(inference, 'make_predictions', lambda *args: lambda p: prediction)
    lengths = []

    def inspect_loss(residual, starts, bounds, **kwargs):
        error = np.asarray(residual(jnp.asarray(starts[0])))
        np.testing.assert_allclose(error, 0, atol=1e-12)
        lengths.append(error.size)
        record = dict(parameters=np.asarray(starts[0]).tolist(), success=True,
                      acceptable=True, chi_squared=0, rank=len(starts[0]))
        return dict(best=record, starts=[record], covariance=np.eye(len(starts[0])).tolist())

    monkeypatch.setattr(inference, 'run_starts', inspect_loss)
    result = inference.fit_beam_microscope(
        directory_path=tmp_path, number_of_starts=1,
        projector_aperture_um=None if sparse else aperture_um
    )
    assert lengths == [2, 24, 24]  # One response; four used images, six values each.
    assert result['fitted_image_count'] == 4
    assert result['profiled_dose_parameters'] == 3


def test_recorded_import_omits_unrecorded_apertures(tmp_path):
    from temgym_core.calibration import import_images
    from temgym_core.calibration.inference import measured_images

    spec = dict(voltages_v=[200000, 195000], aperture_diameters_um=[20, 50],
                pixel_pitch_m=110e-6, psf_sigma_pixels=1.0, background_counts=3.0,
                read_noise_counts=2.0, aperture_edge_width_m=0.15e-6,
                aperture_diffraction=False, image_units='electrons',
                image_axes='row_down_column_right', control_units='current/reference_current')
    records = []
    for i, sign in enumerate([1, -1]):
        records.append(dict(measurement_id=str(i), aperture_um=50, voltage_v=200000,
                            exposure_s=1.0, purpose='projector', response_group=0,
                            projector_controls=[1] * 5, condenser_controls=[1] * 3,
                            specimen_kick=[sign * 20e-9, 0, 0, 0]))
    yy, xx = np.mgrid[:64, :64]
    frames = np.array([3 + 1000 * np.exp(-((xx - cx)**2 + (yy - 31)**2) / 18)
                       for cx in [33, 29]])
    root = import_images(images=frames, records=records, specification=spec,
                         output_directory=tmp_path / 'recorded')
    assert not (root / 'scoring_truth.json').exists()
    packed = np.load(root / 'images.npz')['images']
    assert np.isnan(packed[:, 0]).all()
    measure_beam_images(directory_path=root)
    measured = np.load(root / 'measurements.npz')
    np.testing.assert_array_equal(measured['recorded'], [[False, True], [False, True]])
    dense = np.stack([frames, frames], axis=1)
    expected, covariance, _ = measured_images(dense, spec)
    np.testing.assert_array_equal(measured['values'][:, 1], expected[:, 1])
    np.testing.assert_array_equal(measured['covariance'][:, 1], covariance[:, 1])
    responses = np.load(root / 'projector_responses.npz')
    np.testing.assert_array_equal(responses['aperture_index'], [1])
    assert responses['values'].shape == (1, 2)
    with pytest.raises(FileExistsError):
        import_images(images=frames, records=records, specification=spec, output_directory=root)
    bad = [dict(r) for r in records]
    bad[1]['specimen_kick'] = [20e-9, 0, 0, 0]
    with pytest.raises(ValueError, match='opposite'):
        import_images(images=frames, records=bad, specification=spec, output_directory=tmp_path / 'bad')
    with pytest.raises(ValueError, match='conversion is not inferred'):
        import_images(images=frames, records=records, specification={**spec, 'image_units': 'ADU'},
                      output_directory=tmp_path / 'adu')
    packed[0, 1, 0, 0] = np.nan
    with pytest.raises(ValueError, match='entirely NaN'):
        measured_images(packed, spec)
