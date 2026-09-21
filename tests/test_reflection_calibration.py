"""Pixel-based recovery, truth isolation and native reflection conventions."""
import ast
from dataclasses import replace
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core import reflection_calibration as rc
from temgym_core.coordinate_transforms import PlaneFrame
from temgym_core.run import run_to_end
from temgym_core.source import gaussian_waist


def test_local_orders_match_existing_reflection_adapter():
    """Independent notebook adapter must give the same propagated local envelope."""
    spec = rc.default_specification()
    p = rc.demonstration_truth()
    source = gaussian_waist(z=p[0] * 1e-3, voltage=spec['voltage_v'], rms_size=p[7] * 1e-9)
    incident = run_to_end(source, rc.column(p, np.ones(3)), propagator=rc.FS)
    incoming, outgoing = rc.frames(spec)
    orders = np.column_stack((np.zeros(3, dtype=int), spec['orders']))
    period = spec['surface_period_m']
    notebook = Path(__file__).resolve().parents[1] / 'examples/sem_reflection/sem_reflection.ipynb'
    cells = json.loads(notebook.read_text())['cells']
    cell = next(''.join(c['source']) for c in cells if 'def reflect_packets(' in ''.join(c['source']))
    function = next(node for node in ast.parse(cell).body if isinstance(node, ast.FunctionDef) and node.name == 'reflect_packets')
    env = dict(np=np, jnp=jnp, jax=jax, replace=replace, orders=orders,
               coefficients=np.ones(3, dtype=complex), column_frame=incoming, out_frame=outgoing,
               surface=PlaneFrame(origin_xyz=jnp.zeros(3), u_xyz=jnp.array([0., 0., 1.]), v_xyz=jnp.array([0., 1., 0.])),
               surface_period=period, lateral_period=period, fs=rc.FS, pivot=jnp.zeros(3),
               P={'sample_z': 0., 'min_exit_deg': .25})
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<existing reflection adapter>', 'exec'), env)
    reflected, _ = env['reflect_packets'](incident, apply_response=False)
    _, predict = rc.make_predictor(spec)
    for dz in (-.025, 0., .025):
        beam = rc.FS(reflected, p[8] * 1e-3 + dz * np.cos(np.deg2rad(spec['incidence_deg'])))
        cov = np.linalg.inv(2 * np.asarray(beam.k)[..., None, None] * np.asarray(beam.Q_inv.imag)) / spec['pixel_pitch_m'] ** 2
        cov += np.eye(2) * spec['psf_sigma_pixels'] ** 2
        mu, actual = map(np.asarray, predict(p, np.ones((1, 3)), np.array([dz])))
        np.testing.assert_allclose(actual[0], cov, rtol=2e-10, atol=1e-10)
        np.testing.assert_allclose(mu[0, :, 1], beam.y / spec['pixel_pitch_m'], atol=1e-10)


def test_camera_translation_and_parameter_derivatives():
    spec = rc.default_specification()
    p = rc.INITIAL.copy()
    _, predict = rc.make_predictor(spec)
    controls = np.array([[.85, 1.12, .9], [1.1, .9, 1.15]])
    dz = np.array([-.025, .025])
    fn = lambda x: jnp.concatenate([v.ravel() for v in predict(x, controls, dz)])
    jac = np.asarray(jax.jacfwd(fn)(p))
    for index in range(len(p)):
        step = rc.SCALE[index] * 1e-5
        delta = np.eye(len(p))[index] * step
        finite = np.asarray((fn(p + delta) - fn(p - delta)) / (2 * step))
        np.testing.assert_allclose(jac[:, index], finite, rtol=2e-5, atol=2e-6)
    mu, _ = map(np.asarray, predict(p, np.ones((2, 3)), dz))
    # Specular movement comes from the known laboratory translation, not a
    # camera that follows specular or moves only along its own normal.
    expected = -(dz[1] - dz[0]) * np.sin(np.deg2rad(spec['incidence_deg'])) / spec['pixel_pitch_m']
    np.testing.assert_allclose(mu[1, 1, 0] - mu[0, 1, 0], expected, atol=1e-12)


def test_pixel_integration_matches_marginal_moments():
    spec = rc.default_specification()
    p = rc.demonstration_truth()
    moments, predict = rc.make_predictor(spec)
    controls = np.array([[1., 1., 1.], [.8, 1.2, .85]])
    dz = np.array([0., .025])
    mu, cov = map(np.asarray, predict(p, controls, dz))
    # Fixed strips contain well separated orders; the narrow baseline tests
    # explicit pixel integration rather than blindly adding pixel_pitch²/12.
    strips = np.tile([[-127.5, -25.5], [-25.5, 25.5], [25.5, 128.5]], (2, 1, 1))
    expected = np.asarray(moments(p, controls, dz, strips))
    u, v = np.meshgrid(127.5 - np.arange(256), np.arange(256) - 127.5, indexing='ij')
    for i in range(2):
        image = rc.integrated_image(mu[i], cov[i], np.ones(3), spec)
        refined = rc.integrated_image(mu[i], cov[i], np.ones(3), spec, quadrature=24)
        np.testing.assert_allclose(image, refined, atol=1e-13)
        for order, (lo, hi) in enumerate(strips[i]):
            mask = (v >= lo) & (v < hi)
            weights = image * mask
            total = weights.sum()
            a, b = (np.sum(weights * grid) / total for grid in (u, v))
            actual = [a, b, np.sum(weights * (u - a) ** 2) / total, np.sum(weights * (v - b) ** 2) / total]
            np.testing.assert_allclose(actual, expected[i, order], atol=2e-6)


def test_saved_pixel_recovery_without_truth(tmp_path, monkeypatch):
    root = rc.simulate_experiment(tmp_path / 'experiment')
    truth = json.loads((root / 'scoring_truth.json').read_text())['parameters']
    (root / 'scoring_truth.json').unlink()
    # The fitter may not quietly obtain the generator's default parameters.
    monkeypatch.setattr(rc, 'demonstration_truth', lambda: pytest.fail('truth accessed'))
    rc.measure_images(root)
    fit = rc.fit_microscope(root, number_of_starts=8)
    checks = rc.check_microscope(root)
    assert fit['accepted'] and fit['rank'] == 9
    assert checks['validated']
    assert 'truth' not in checks and 'errors' not in checks
    assert .8 < checks['prediction_rms_noise_units'] < 1.2
    recovered, std = np.asarray(fit['best']['parameters']), np.asarray(fit['standard_errors'])
    assert np.all(abs(recovered - truth) < 4 * std)
    assert abs(recovered[3] + 40) < 1.
    assert abs(recovered[8] - 120) < .001
    assert any(r['cost'] > 20 for r in fit['initialization'])  # wrong branches retained
    assert all(np.linalg.norm(np.asarray(r['parameters']) - recovered) < .02 for r in fit['starts'] if r['chi_squared'] < 2000)
    # Changed acquisition metadata must not reuse an old fit.
    spec = json.loads((root / 'experiment.json').read_text())
    spec['pixel_pitch_m'] *= 1.01
    (root / 'experiment.json').write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='stale'):
        rc.check_microscope(root)
