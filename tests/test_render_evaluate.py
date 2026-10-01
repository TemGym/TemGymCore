"""``evaluate_gaussians`` on the differentiable GPU renderers.

Beam-level derivatives (through the packing chain rule and the kernels), the two coordinate
modes and the detector pose, invalid beamlets, aliases and configuration.  The renderers run
their pure-JAX implementation on any platform (``impl="reference"``) and the Pallas kernels on
a GPU (``impl="pallas"``, marked ``gpu``).
"""

import dataclasses
import importlib
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from render_bundles import rel

from temgym_core.components import Detector
from temgym_core.evaluate import (
    _render_config,
    describe_plan,
    evaluate_gaussians,
    evaluate_gaussians_jax_scan,
    gpu_kernel_available,
)
from temgym_core.gaussian import make_gaussian
from temgym_core.render import RenderConfig, Tiles

evaluate_module = importlib.import_module("temgym_core.evaluate")
jax.config.update("jax_enable_x64", True)

N_BEAMS = 24
SHAPE = (9, 7)
CONFIGS = {
    "reference": RenderConfig(impl="reference", pixel_chunk=16, nt_chunk=4),
    "pallas": RenderConfig(impl="pallas", nt_chunk=4, items=256).with_tiles(16, 8, 4),
}


@pytest.fixture
def gpu_backend():
    if not gpu_kernel_available():
        pytest.skip("requires a verified Pallas GPU backend (JAX_PLATFORMS=cuda)")


@pytest.fixture(
    params=[pytest.param("reference", id="reference"),
            pytest.param("pallas", id="pallas", marks=pytest.mark.gpu)]
)
def cfg(request):
    if request.param == "pallas" and not gpu_kernel_available():
        pytest.skip("requires a verified Pallas GPU backend (JAX_PLATFORMS=cuda)")
    return CONFIGS[request.param]


@pytest.fixture(params=["index", "metre"])
def mode(request, cfg):
    return dataclasses.replace(cfg, coordinates=request.param)


def detector(rotation=23.0, centre=(0.3e-9, -0.2e-9), pixel_size=(1.3e-9, 1.7e-9), flip_y=False):
    return Detector(z=0.0, pixel_size=pixel_size, shape=SHAPE, rotation=rotation, centre=centre,
                    flip_y=flip_y)


def plain_grid():
    x, y = np.meshgrid(np.linspace(-6e-9, 6e-9, SHAPE[1]), np.linspace(-5e-9, 5e-9, SHAPE[0]))
    return SimpleNamespace(shape=SHAPE, coords=jnp.asarray(np.column_stack((x.ravel(), y.ravel()))))


def theta0():
    rng = np.random.default_rng(0)
    return dict(
        x=jnp.asarray(rng.uniform(-4e-9, 4e-9, N_BEAMS)),
        y=jnp.asarray(rng.uniform(-3e-9, 3e-9, N_BEAMS)),
        dx=jnp.asarray(rng.uniform(-8e-4, 8e-4, N_BEAMS)),
        dy=jnp.asarray(rng.uniform(-8e-4, 8e-4, N_BEAMS)),
        amp=jnp.asarray(rng.uniform(0.5, 1.5, N_BEAMS)),
        phase=jnp.asarray(rng.uniform(0, 6, N_BEAMS)),
        waist_x=jnp.asarray(rng.uniform(1.5e-9, 3e-9, N_BEAMS)),
        waist_y=jnp.asarray(rng.uniform(1.5e-9, 3e-9, N_BEAMS)),
        rcurv_x=jnp.asarray(rng.uniform(2e-6, 8e-6, N_BEAMS)),
        rcurv_y=jnp.asarray(rng.uniform(-8e-6, -2e-6, N_BEAMS)),
        pathlength=jnp.asarray(rng.uniform(0, 2e-9, N_BEAMS)),
        voltage=jnp.asarray(200e3 + 500 * rng.uniform(-1, 1, N_BEAMS)),
    )


def beam_of(theta):
    beam = make_gaussian(
        x=theta["x"], y=theta["y"], dx=theta["dx"], dy=theta["dy"], voltage=theta["voltage"],
        amp=theta["amp"], phase=theta["phase"], waist_x=theta["waist_x"],
        waist_y=theta["waist_y"], rcurv_x=theta["rcurv_x"], rcurv_y=theta["rcurv_y"],
    )
    return beam.derive(pathlength=theta["pathlength"])


def flatten(theta):
    keys = sorted(theta)
    return keys, jnp.concatenate([theta[k] for k in keys])


def unflatten(keys, vec):
    return {k: vec[i * N_BEAMS:(i + 1) * N_BEAMS] for i, k in enumerate(keys)}


def render(grid, config, method="sparse"):
    return lambda b: evaluate_gaussians(b, grid, method=method, config=config)


def scan(grid):
    return lambda b: evaluate_gaussians_jax_scan(b, grid, batch_size=8)


@pytest.mark.parametrize("method", ["sparse", "dense"])
@pytest.mark.parametrize("flip", [False, True])
def test_field_matches_scan_renderer(mode, method, flip):
    grid, theta = detector(flip_y=flip), theta0()
    mine = render(grid, mode, method)(beam_of(theta))
    assert mine.shape == SHAPE and mine.dtype == jnp.complex128
    assert rel(mine, scan(grid)(beam_of(theta))) <= 1e-11


def test_field_on_grids_without_a_pixel_map(cfg):
    """Only pixel positions: the dense kernels on metre coordinates."""
    grid, theta = plain_grid(), theta0()
    for method in ("sparse", "dense"):
        assert rel(render(grid, cfg, method)(beam_of(theta)), scan(grid)(beam_of(theta))) <= 1e-11


@pytest.mark.parametrize("method", ["sparse", "dense"])
def test_jacobian_wrt_all_beam_fields_matches_scan_autodiff(mode, method):
    grid, theta = detector(), theta0()
    keys, vec = flatten(theta)

    def field(renderer):
        return lambda v: renderer(beam_of(unflatten(keys, v))).ravel()

    mine = jax.jacfwd(field(render(grid, mode, method)))(vec)
    expected = jax.jacfwd(field(scan(grid)))(vec)
    assert mine.shape == expected.shape == (SHAPE[0] * SHAPE[1], len(vec))
    assert rel(mine, expected) <= 1e-9
    per_key = jnp.max(jnp.abs(expected), axis=0).reshape(len(keys), N_BEAMS).max(axis=1)
    assert bool(jnp.all(per_key > 0)), dict(zip(keys, per_key))  # every field family matters


def test_gradient_of_an_intensity_loss_matches_scan_autodiff(mode):
    grid, theta = detector(), theta0()
    keys, vec = flatten(theta)
    target = jnp.asarray(np.random.default_rng(3).uniform(0, 4, SHAPE))

    def loss(renderer):
        def fn(v):
            intensity = jnp.abs(renderer(beam_of(unflatten(keys, v)))) ** 2
            return jnp.sum((intensity - target) ** 2)
        return fn

    mine = jax.jit(jax.grad(loss(render(grid, mode))))(vec)
    assert rel(mine, jax.grad(loss(scan(grid)))(vec)) <= 1e-9


def test_index_and_metre_modes_agree_to_rounding(cfg):
    grid, beam = detector(rotation=-41.0, flip_y=True), beam_of(theta0())
    index = render(grid, dataclasses.replace(cfg, coordinates="index"))(beam)
    metre = render(grid, dataclasses.replace(cfg, coordinates="metre"))(beam)
    assert rel(index, metre) <= 1e-11


def pose_loss(renderer, beam, weights):
    def loss(pose):
        grid = detector(rotation=pose[0], centre=(pose[1], pose[2]), pixel_size=(pose[3], pose[4]))
        return jnp.sum(weights * jnp.abs(renderer(grid)(beam)) ** 2)
    return loss


POSE = jnp.array([23.0, 0.3e-9, -0.2e-9, 1.3e-9, 1.7e-9])


def test_detector_pose_gradients_are_ordinary_autodiff_in_index_mode(cfg):
    beam = beam_of(theta0())
    weights = jnp.asarray(np.random.default_rng(4).normal(size=SHAPE))
    mine = jax.grad(pose_loss(lambda g: render(g, cfg), beam, weights))(POSE)
    expected = jax.grad(pose_loss(scan, beam, weights))(POSE)
    scale = jnp.max(jnp.abs(expected), initial=0.0)
    assert float(jnp.max(jnp.abs(mine - expected) / jnp.maximum(jnp.abs(expected), 1e-3 * scale))) <= 1e-8
    # jvp along the pose, and jit
    tangent = jnp.array([1.0, 1e-10, -1e-10, 1e-10, -1e-10])
    _, t_mine = jax.jvp(pose_loss(lambda g: render(g, cfg), beam, weights), (POSE,), (tangent,))
    _, t_ref = jax.jvp(pose_loss(scan, beam, weights), (POSE,), (tangent,))
    assert abs(float(t_mine - t_ref)) <= 1e-8 * abs(float(t_ref))


def test_metre_mode_rejects_pose_derivatives_and_names_the_alternative(cfg):
    metre = dataclasses.replace(cfg, coordinates="metre")
    beam = beam_of(theta0())
    weights = jnp.ones(SHAPE)
    with pytest.raises(NotImplementedError, match="coordinates='index'"):
        jax.grad(pose_loss(lambda g: render(g, metre), beam, weights))(POSE)
    # differentiating the beamlets only is fine in metre mode
    theta = theta0()
    grad = jax.grad(lambda amp: jnp.sum(jnp.abs(render(detector(), metre)(
        beam_of({**theta, "amp": amp}))) ** 2))(theta["amp"])
    assert np.isfinite(np.asarray(grad)).all()


def test_second_derivatives_raise_through_the_dispatcher(cfg):
    grid, theta = detector(), theta0()
    loss = lambda amp: jnp.sum(jnp.abs(render(grid, cfg)(beam_of({**theta, "amp": amp}))) ** 2)
    with pytest.raises(NotImplementedError, match="Second and higher derivatives"):
        jax.hessian(loss)(theta["amp"])


# ---------------------------------------------------------------------------
# Invalid beamlets, scalar and empty inputs
# ---------------------------------------------------------------------------
FIELDS = ("x", "y", "dx", "dy", "amplitude", "pathlength", "Q_inv")


@pytest.mark.parametrize("field", ["amplitude", "x", "Q_inv", "pathlength"])
def test_nonfinite_beamlets_contribute_zero_with_zero_gradient(mode, field):
    """A beamlet with a non-finite field is dropped: zero field, zero gradient, no poisoning."""
    grid, beam = detector(), beam_of(theta0())
    renderer = render(grid, mode)
    bad = beam.derive(**{field: getattr(beam, field).at[5].set(jnp.nan)})
    healthy = beam_of({k: jnp.delete(v, 5) for k, v in theta0().items()})
    assert np.isfinite(np.asarray(renderer(bad))).all()
    assert rel(renderer(bad), renderer(healthy)) <= 1e-12

    def loss(fields, template):
        return jnp.sum(jnp.abs(renderer(template.derive(**fields))) ** 2)

    def grads(template):
        return jax.grad(loss)({k: getattr(template, k) for k in FIELDS}, template)

    grad_bad, grad_healthy = grads(bad), grads(healthy)
    for key, g in grad_bad.items():
        assert np.isfinite(np.asarray(g)).all(), key
        assert float(jnp.max(jnp.abs(g[5]))) == 0.0, key
        assert rel(jnp.delete(g, 5, axis=0) + 1e-300, grad_healthy[key] + 1e-300) <= 1e-9, key


def test_nonfinite_beamlet_does_not_poison_the_pose_gradient(cfg):
    theta = theta0()
    theta["amp"] = theta["amp"].at[3].set(jnp.nan)
    beam = beam_of(theta)
    weights = jnp.asarray(np.random.default_rng(4).normal(size=SHAPE))
    grad = jax.grad(pose_loss(lambda g: render(g, cfg), beam, weights))(POSE)
    assert np.isfinite(np.asarray(grad)).all()


def test_single_beamlet_and_beamlet_outside_the_detector(cfg):
    grid = detector()
    one = make_gaussian(waist_x=2e-9, waist_y=3e-9, voltage=200e3)
    assert rel(render(grid, cfg)(one), scan(grid)(one)) <= 1e-11
    far = make_gaussian(x=1.0, waist_x=2e-9, waist_y=3e-9, voltage=200e3)
    assert float(jnp.max(jnp.abs(render(grid, cfg)(far)))) == 0.0


# ---------------------------------------------------------------------------
# Dispatcher: aliases, legacy keywords, configuration, diagnostics
# ---------------------------------------------------------------------------
def test_aliases_select_the_same_renderers():
    grid, beam = detector(), beam_of(theta0())
    reference = RenderConfig(impl="reference", pixel_chunk=16)
    sparse = evaluate_gaussians(beam, grid, method="sparse", config=reference)
    dense = evaluate_gaussians(beam, grid, method="dense", config=reference)
    np.testing.assert_array_equal(evaluate_gaussians(beam, grid, method="spatial", config=reference),
                                  sparse)
    np.testing.assert_array_equal(evaluate_gaussians(beam, grid, method="gpu_kernel",
                                                     config=reference), dense)
    assert rel(sparse, dense) <= 1e-11
    for method in ("scan", "loop"):
        assert rel(evaluate_gaussians(beam, grid, method=method), dense) <= 1e-11


def test_auto_is_scan_without_a_gpu(monkeypatch):
    monkeypatch.setattr(evaluate_module, "gpu_kernel_available", lambda: False)
    grid, beam = detector(), beam_of(theta0())
    np.testing.assert_array_equal(evaluate_gaussians(beam, grid, method="auto"),
                                  evaluate_gaussians(beam, grid, method="scan"))
    for method in ("sparse", "dense"):
        np.testing.assert_array_equal(evaluate_gaussians(beam, grid, method=method),
                                      evaluate_gaussians(beam, grid, method="scan"))


def test_scan_stays_differentiable_by_jax_itself():
    grid, theta = detector(), theta0()
    grad = jax.grad(lambda amp: jnp.sum(jnp.abs(evaluate_gaussians(
        beam_of({**theta, "amp": amp}), grid, method="scan")) ** 2))(theta["amp"])
    assert np.isfinite(np.asarray(grad)).all() and float(jnp.max(jnp.abs(grad))) > 0


def test_legacy_keywords_map_onto_the_config():
    config = _render_config(None, 32, 8, 1e-9, 128)
    assert config.cutoff == 1e-9 and config.bins_per_axis == 128
    assert config.rows == Tiles(32, 8, 4) and config.forward == Tiles(32, 8, 4)
    assert _render_config(None, None, None, None, None) == RenderConfig()
    kept = RenderConfig(cutoff=1e-6, coordinates="metre")
    assert _render_config(kept, None, 32, None, None).coordinates == "metre"
    assert _render_config(kept, None, 32, None, None).rows == Tiles(16, 32, 8)
    with pytest.raises(ValueError):
        _render_config(None, None, 24, None, None)


@pytest.mark.parametrize("changes", [
    {"cutoff": 0.0}, {"cutoff": 1.0}, {"cutoff": float("nan")}, {"coordinates": "pixel"},
    {"impl": "cuda"}, {"bins_per_axis": 1}, {"bins_per_axis": 4096}, {"class_edges": (5.0, 5.0)},
    {"class_edges": (-1.0,)}, {"n_classes": 99}, {"class_ratio": 1.0}, {"items": 0},
    {"window_warps": 3}, {"nt_chunk": 0},
])
def test_invalid_render_configurations_are_rejected(changes):
    with pytest.raises(ValueError):
        RenderConfig(**changes)


def test_class_edges_are_a_geometric_ladder_up_to_half_the_detector():
    edges = RenderConfig().edges_for((1024, 800))
    assert len(edges) == 8 and edges[-1] == pytest.approx(512.0)
    assert all(b / a == pytest.approx(2 ** 0.5) for a, b in zip(edges, edges[1:]))
    assert RenderConfig(class_edges=(3.0, 9.0)).edges_for((10, 10)) == (3.0, 9.0)
    assert RenderConfig(n_classes=0).edges_for((10, 10)) == ()


def test_describe_plan_counts_classes():
    grid = Detector(z=0.0, pixel_size=(1e-9, 1e-9), shape=(64, 64))
    narrow = make_gaussian(x=jnp.linspace(-20e-9, 20e-9, 30), waist_x=3e-9, waist_y=3e-9,
                           voltage=200e3)
    summary = describe_plan(narrow, grid, RenderConfig(n_classes=4))
    assert summary.dense == 0 and sum(summary.class_counts) + summary.inactive == 30
    assert len(summary.edges) == len(summary.class_counts) == len(summary.cell_sizes) == 4
    far = make_gaussian(x=jnp.array([1.0, -1.0]), waist_x=3e-9, waist_y=3e-9, voltage=200e3)
    assert describe_plan(far, grid).inactive == 2
    with pytest.raises(ValueError, match="pixel-to-metre"):
        describe_plan(narrow, plain_grid())


def test_empty_bundle_and_bitwise_repeatability(cfg):
    grid, beam = detector(), beam_of(theta0())
    empty = jax.tree.map(lambda a: a[:0] if a.ndim and a.shape[0] == N_BEAMS else a, beam)
    for method in ("sparse", "dense"):
        out = render(grid, cfg, method)(empty)
        assert out.shape == SHAPE and float(jnp.max(jnp.abs(out))) == 0.0
    renderer = render(grid, cfg)
    weights = jnp.asarray(np.random.default_rng(5).normal(size=SHAPE))

    def gradient():
        return jax.grad(lambda amp: jnp.sum(weights * jnp.abs(renderer(
            beam.derive(amplitude=amp)) ** 2)))(beam.amplitude)

    runs = [(np.asarray(renderer(beam)), np.asarray(gradient())) for _ in range(3)]
    for field, grad in runs[1:]:
        np.testing.assert_array_equal(field, runs[0][0])
        np.testing.assert_array_equal(grad, runs[0][1])


@pytest.mark.gpu
def test_gpu_sparse_equals_dense_with_predicate_on_random_detector_poses(gpu_backend):
    """Bins and windows are supersets in both coordinate modes: rotated, flipped, anisotropic."""
    master = np.random.default_rng(7)
    for case in range(6):
        rng = np.random.default_rng(master.integers(1 << 30))
        n = 150
        shape = (int(rng.choice([7, 20, 45])), int(rng.choice([9, 31, 60])))
        pixel = (float(rng.uniform(0.6e-6, 1.6e-6)), float(rng.uniform(0.6e-6, 1.6e-6)))
        grid = Detector(z=0.0, pixel_size=pixel, shape=shape, rotation=float(rng.uniform(-180, 180)),
                        centre=(float(rng.uniform(-3e-6, 3e-6)), float(rng.uniform(-3e-6, 3e-6))),
                        flip_y=bool(case % 2))
        spread = 1.2 * pixel[0] * max(shape)
        waists = rng.uniform(0.8e-6, 4e-6, (2, n)) * (10.0 ** rng.uniform(-0.3, 0.5, n)) ** (rng.random(n) < 0.1)
        beam = make_gaussian(
            x=rng.uniform(-spread, spread, n), y=rng.uniform(-spread, spread, n),
            dx=rng.uniform(-2e-4, 2e-4, n), dy=rng.uniform(-2e-4, 2e-4, n), voltage=200e3,
            waist_x=waists[0], waist_y=waists[1], amp=rng.normal(size=n))
        plain = SimpleNamespace(shape=shape, coords=grid.coords)        # dense kernels, same predicate
        # Same arithmetic frame (metre): a pair the plan missed would differ by ~cutoff, far above
        # rounding.  The index frame re-rounds the packed parameters (~1e-13), so it is checked
        # at cutoff 1e-8 only.
        for coordinates, cutoff, tolerance in (("metre", float(rng.choice([1e-8, 1e-12])), 2e-13),
                                               ("index", 1e-8, 2e-12)):
            config = RenderConfig(cutoff=cutoff, coordinates=coordinates, bins_per_axis=16,
                                  items=256)
            got = evaluate_gaussians(beam, grid, method="sparse", config=config)
            want = evaluate_gaussians(beam, plain, method="sparse", config=config)
            assert rel(got, want) <= tolerance, (case, coordinates)


def test_gpu_availability_probe_is_eager_when_first_called_while_tracing(monkeypatch):
    """First use inside ``jit`` used to fail on a tracer and cache "no GPU" for the process."""
    monkeypatch.setattr(evaluate_module.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(evaluate_module.jax, "devices",
                        lambda backend=None: [type("FakeGpu", (), {"platform": "gpu"})()])

    def reference_kernel(r, dr, amplitude, pathlength, q_inv, k, coords, **_kwargs):
        delta = coords[:, None, :] - r[None, :, :]
        linear = jnp.einsum("pbi,bi->pb", delta, dr)
        quadratic = jnp.einsum("pbi,bij,pbj->pb", delta, q_inv, delta)
        return jnp.sum(amplitude[None, :] * jnp.exp(
            1j * k[None, :] * (pathlength[None, :] + linear + 0.5 * quadratic)), axis=1)

    monkeypatch.setattr(evaluate_module, "_evaluate_gaussians_gpu_kernel_impl", reference_kernel)
    gpu_kernel_available.cache_clear()
    try:
        seen = []
        jax.jit(lambda x: seen.append(gpu_kernel_available()) or x)(jnp.zeros(1))
        assert seen == [True]
    finally:
        gpu_kernel_available.cache_clear()


@pytest.mark.gpu
def test_gpu_first_use_inside_jit_selects_the_kernels(gpu_backend):
    """In a fresh process the first ``jit`` must run the probe eagerly and pick the GPU renderer."""
    import subprocess
    import sys

    code = (
        "import jax, jax.numpy as jnp\n"
        "jax.config.update('jax_enable_x64', True)\n"
        "from temgym_core.evaluate import gpu_kernel_available\n"
        "seen = []\n"
        "jax.jit(lambda x: seen.append(gpu_kernel_available()) or x)(jnp.zeros(1))\n"
        "print('AVAILABLE', seen[0], gpu_kernel_available())\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            timeout=600)
    assert "AVAILABLE True True" in result.stdout, result.stderr[-2000:]
