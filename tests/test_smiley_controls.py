"""Interactive controls must update the native optics and their Gaussian preview."""
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.components import DistortedLens, SigmoidAperture
from temgym_core.run import column_matrix, run_to_end


@pytest.fixture(scope="module")
def demo():
    path = Path(__file__).parents[1] / "examples/microscope_models/smiley_defocus.py"
    import sys
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("smiley_controls", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SmileyDefocus((32, 32))


def test_zero_aberrations_preserve_native_magnetic_column(demo):
    incident = demo.incident_beamlets(0., demo.diagram_indices)
    sample = demo.sample
    expected = jax.jit(lambda b: run_to_end(b, (sample, *demo.downstream),
                                          propagator=demo.propagator))(incident)
    actual = jax.jit(lambda b: run_to_end(b, (sample, *demo.objective_column()),
                                        propagator=demo.propagator))(incident)
    for name in ("r_xy", "d_xy", "Q_inv", "amplitude", "pathlength"):
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name),
                                   rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize("magnification", [60000., 600000.])
def test_magnification_and_spot_use_native_tables(demo, magnification):
    changed = demo.with_settings(spot_size=4., magnification=magnification)
    matrix = np.asarray(column_matrix(changed.downstream, source_z=changed.z_sample,
                                      voltage=changed.model.voltage))
    np.testing.assert_allclose(np.linalg.svd(matrix[:2, :2], compute_uv=False), magnification, rtol=1e-8)
    np.testing.assert_allclose(matrix[:2, 2:4], 0., atol=1e-8)
    from temgym_core.components import HardEdgeSolenoid
    assert isinstance(changed.model.components["CL1"], HardEdgeSolenoid)
    assert changed.model.components["CL1"].q != demo.model.components["CL1"].q
    assert changed.source is demo.source
    assert changed.objective is changed.model.components["Obj_post"]
    from temgym_core.components import KrivanekLens
    assert isinstance(changed.objective_column(astigmatism_um=10.)[0], KrivanekLens)
    assert changed.objective_column(astigmatism_um=10.)[0].coeffs.C12 == pytest.approx(10e-6)
    assert not np.allclose(changed.nominal_illumination.Q_inv, demo.nominal_illumination.Q_inv)
    assert changed.cache is not demo.cache
    assert demo.magnification == 40000.
    assert demo.spot_size == 3.


def test_aperture_uses_gaussians_and_recovers_open_field(demo):
    opened = demo.image(0.)[0]
    changed = demo.with_settings(aperture_radius_m=10*1e-6)
    assert isinstance(changed.aperture, SigmoidAperture)
    assert changed.aperture_beams.ray_family == "gaussian"
    assert changed.aperture.radius == 10e-6
    assert changed.aperture.z == demo.model.config["apertures"]["C_aperture"]["z_m"]
    image, _, cached = changed.image(0.)
    assert not cached
    # The calibrated source actually illuminates the condenser aperture.
    assert np.linalg.norm(image-opened)/np.linalg.norm(opened) > 1e-3
    assert changed.image(0.)[2]


def test_aberrated_preview_is_the_rendered_gaussian_subset(demo):
    controls = (125., 4., 8., 12000., 10.)
    output = demo.propagate(*map(jnp.asarray, controls))
    _, coordinates, _ = demo.trace_preview(*map(jnp.asarray, controls))
    np.testing.assert_allclose(coordinates[-1], output.r_xy[demo.diagram_indices],
                               rtol=1e-9, atol=1e-12)
    ideal = demo.propagate(jnp.asarray(controls[0]))
    assert np.linalg.norm(np.asarray(output.r_xy-ideal.r_xy)) > 1e-6
    projector = next(c for c in demo.objective_column(projector_distortion=10.)
                     if isinstance(c, DistortedLens))
    assert projector.IsoDist == 1e10
    assert projector.object_plane_dist > 0
    ideal_image = demo.image(0.)[0]
    distorted, _, cached = demo.image(0., projector_distortion=10.)
    assert not cached
    assert np.linalg.norm(distorted-ideal_image)/np.linalg.norm(ideal_image) > 1e-3
    assert demo.image(0., projector_distortion=10.)[2]


def test_invalid_controls_and_separate_sampling_cache(demo):
    with pytest.raises(ValueError):
        demo.with_settings(spot_size=9.)
    with pytest.raises(ValueError):
        demo.with_settings(aperture_radius_m=1e-6)
    with pytest.raises(ValueError):
        demo.image(0., projector_distortion=np.nan)
    other = demo.with_lattice_shape((16, 16))
    assert other.cache == {}
    assert other.num_beams == 256
    assert demo.num_beams == 1024


def test_broad_spot_retains_specimen_sampling_and_incident_field(demo):
    from types import SimpleNamespace
    from temgym_core.evaluate import evaluate_gaussians

    broad = demo.with_settings(spot_size=1.)
    beamlets = broad.incident_beamlets(0.)
    # Cover the full illumination instead of silently truncating at 500 nm.
    assert np.max(np.abs(np.asarray(beamlets.x))) > 500e-9
    axis = jnp.linspace(-80e-9, 80e-9, 33)
    grid = SimpleNamespace(coords=jnp.stack((axis, jnp.zeros_like(axis)), axis=-1),
                           shape=(33,))
    actual = np.asarray(evaluate_gaussians(beamlets, grid, method="scan"))
    expected = np.asarray(evaluate_gaussians(broad.nominal_illumination, grid, method="scan"))
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=1e-10)


def test_source_bundle_round_trip_and_recomposition_choice(demo):
    assert demo.propagation_mode == "source"
    for height in (0., 10000.):
        source = demo.source_beamlets(height, demo.diagram_indices)
        assert source.ray_family == "gaussian"
        np.testing.assert_allclose(source.z, demo.source.z)
        incident = demo.continuous_incident_beamlets(height, demo.diagram_indices)
        target = demo._specimen_basis(height, demo.diagram_indices, apply_aperture=False)
        for name in ("r_xy", "d_xy", "amplitude"):
            np.testing.assert_allclose(getattr(incident, name), getattr(target, name),
                                       rtol=1e-7, atol=1e-10)
        scale = np.max(np.abs(np.asarray(target.Q_inv)))
        np.testing.assert_allclose(incident.Q_inv/scale, target.Q_inv/scale,
                                   rtol=1e-9, atol=1e-10)
    alternate = demo.with_propagation_mode("recompose")
    assert alternate.cache == {}
    recomposed, continuous = alternate.image(0.)[0], demo.image(0.)[0]
    # Near interference minima a pixelwise relative error is ill-conditioned.
    assert np.linalg.norm(recomposed-continuous)/np.linalg.norm(recomposed) < 1e-5
    assert np.max(np.abs(recomposed-continuous)) < 1e-5*np.max(recomposed)
    assert demo.propagation_mode == "source"
    with pytest.raises(ValueError):
        demo.with_propagation_mode("unknown")


@pytest.mark.parametrize('magnification', [40000., 600000.])
def test_il1_screen_is_native_fourier_plane(demo, magnification):
    view = demo.with_view_plane('il1_crossover')
    if magnification != demo.magnification:
        view = view.with_settings(magnification=magnification)
    assert view.model.components['IL1'].z < view.view_detector.z < view.model.components['IL2'].z
    native = tuple(c for c in view.downstream if c.z < view.view_detector.z)+(view.view_detector,)
    matrix = np.asarray(column_matrix(native, source_z=view.z_sample,
                                      voltage=view.model.voltage))
    np.testing.assert_allclose(matrix[:2, :2], 0., atol=1e-8)
    assert view.viewing_column()[-1] is view.view_detector
    assert not any(isinstance(c, DistortedLens) for c in view.viewing_column())
    assert demo.view_plane == 'detector'
    assert demo.view_detector is demo.detector
    assert view.cache is not demo.cache


def test_diffraction_is_coherent_field_before_projector(demo):
    from copy import copy
    from temgym_core.components import Plane
    from temgym_core.evaluate import evaluate_gaussians

    view = demo.with_view_plane('il1_crossover')
    ideal = view.image(0.)[0]
    distorted = view.image(0., projector_distortion=-100.)[0]
    np.testing.assert_allclose(distorted, ideal, rtol=1e-12)
    aberrated = view.image(0., astigmatism_um=10.)[0]
    assert np.linalg.norm(aberrated-ideal)/np.linalg.norm(ideal) > 1e-3

    # Independent vacuum oracle: one Gaussian propagated through native optics
    # must agree with the coherent sum of the entire source decomposition.
    vacuum = copy(view)
    vacuum.sample = Plane(z=view.z_sample)
    vacuum._configure_sampling(view.lattice_shape)
    bundle_image = vacuum.image(0.)[0]
    column = tuple(c for c in view.downstream if c.z < view.view_detector.z)+(view.view_detector,)
    beam = run_to_end(view.nominal_illumination, column, propagator=view.propagator)
    expected = np.abs(np.asarray(evaluate_gaussians(beam, view.view_detector, method='scan')))**2
    assert np.linalg.norm(bundle_image-expected)/np.linalg.norm(expected) < .01


def test_source_is_fitted_virtual_waist_and_crosses_after_cl1(demo):
    from temgym_core.calibration.condenser_model import first_crossover_fraction
    params = jnp.asarray(demo.model.config['calibration']['condenser_parameters'])
    np.testing.assert_allclose(demo.source.z, -jnp.exp(params[:9]).sum())
    np.testing.assert_allclose(demo.source.Q_inv.real, 0., atol=1e-12)
    rms = 1/jnp.sqrt(2*demo.source.k*jnp.imag(demo.source.Q_inv[0, 0]))
    np.testing.assert_allclose(rms, jnp.exp(params[17]))
    controls = jnp.array([demo.model.components[n].q/jnp.exp(params[9+i])
                         for i, n in enumerate(('CL1', 'CL3', 'C_mini'))])
    assert .01 < float(first_crossover_fraction(params, controls)) < .99


def test_stale_transport_requires_kernel_restart(demo, monkeypatch):
    from temgym_core.components import HardEdgeSolenoid
    check = type(demo).__init__.__globals__["assert_current_transport"]
    check()
    monkeypatch.setattr(HardEdgeSolenoid, "_call_gaussian", lambda self, ray: ray)
    with pytest.raises(RuntimeError, match=r"Stale TemGymCore.*HardEdgeSolenoid.*Restart"):
        check()


@pytest.mark.parametrize("pixel", [np.nan, 0.])
def test_render_failure_reports_reproduction_settings(demo, monkeypatch, pixel):
    monkeypatch.setattr(demo, "render", lambda *args: np.full((2, 2), pixel))
    with pytest.raises(RuntimeError) as error:
        demo.image(123., projector_distortion=-42., use_cache=False)
    message = str(error.value)
    for setting in ("PL1=-42", "magnification=40000", "spot=3", "aperture=out",
                    "beams=1024", "height_nm=123", "view=detector", "propagation=source"):
        assert setting in message
    assert ("4 non-finite pixels" if np.isnan(pixel) else "zero intensity") in message
