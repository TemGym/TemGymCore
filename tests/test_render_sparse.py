"""The sparse renderer's plan and its end-to-end behaviour on awkward bundles.

The plan (radius classes, bins, windows, adjoint work items) is plain JAX, so its invariants
are checked against brute-force geometry on the CPU.  Bundles with widths spanning orders of
magnitude, beamlets off the detector, invalid beamlets and empty inputs then run through
``evaluate_packed`` (pure JAX on CPU, the Pallas kernels on the GPU) against the dense sum with
the same pair predicate.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from render_bundles import field_scan, pixel_bundle, rel

from temgym_core.evaluate import gpu_kernel_available
from temgym_core.render import RenderConfig, evaluate_packed
from temgym_core.render.config import MAX_INDEX, KernelSpec
from temgym_core.render.packing import NULL_ROW
from temgym_core.render.plan import build_plan, footprints
from temgym_core.render.primitives import MAX_BEAMS

jax.config.update("jax_enable_x64", True)

QCOLS = [5, 7, 9]
CUTOFF = 1e-8


@pytest.fixture
def gpu():
    if not gpu_kernel_available():
        pytest.skip("requires a verified Pallas GPU backend (JAX_PLATFORMS=cuda)")


def mixed_bundle(seed, n, shape, margin, log_spread=(-3.5, 0.5), waist=2.5):
    """Bundle whose footprint radii span two orders of magnitude (pixel-index frame)."""
    rng = np.random.default_rng(seed)
    packed, coords, _ = pixel_bundle(rng, n, shape, waist_px=waist, margin=margin)
    packed[:, QCOLS] *= 10 ** rng.uniform(*log_spread, (n, 1))
    return rng, jnp.asarray(packed), jnp.asarray(coords)


def config_for(**changes):
    return RenderConfig(cutoff=CUTOFF, bins_per_axis=16, items=96, n_classes=4, **changes)


def sparse_spec(shape, config):
    return KernelSpec(config, shape[0] * shape[1], shape, "index", sparse=True)


def predicate_mask(packed, coords, limit):
    """Brute-force ``(P, N)`` predicate in float64 numpy (the definition of the support)."""
    pk = np.asarray(packed)
    u = np.asarray(coords)[:, 0, None] - pk[None, :, 0]
    v = np.asarray(coords)[:, 1, None] - pk[None, :, 1]
    decay = u * u * pk[None, :, 5] + u * v * pk[None, :, 7] + v * v * pk[None, :, 9]
    return decay <= limit


# ---------------------------------------------------------------------------
# Plan invariants (CPU)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed,shape,margin", [(0, (37, 53), 30.0), (1, (16, 16), 80.0),
                                               (2, (1, 90), 10.0)])
def test_plan_windows_bins_and_items_cover_every_pair_the_predicate_keeps(seed, shape, margin):
    n = 300
    _, packed, coords = mixed_bundle(seed, n, shape, margin)
    config = config_for()
    spec = sparse_spec(shape, config)
    plan = build_plan(packed, jnp.zeros(6), spec)
    limit = config.decay_limit
    keep = predicate_mask(packed, coords, limit)
    order = np.asarray(plan.order)
    assert sorted(order) == list(range(n))

    # windows: every kept pixel of every beam lies inside the beam's window
    windows = np.asarray(plan.windows)
    ix, iy = np.asarray(coords)[:, 0], np.asarray(coords)[:, 1]
    for pos, b in enumerate(order):
        x_lo, x_hi, y_lo, y_hi = windows[pos]
        pixels = np.flatnonzero(keep[:, b])
        assert np.all((ix[pixels] >= x_lo) & (ix[pixels] <= x_hi)
                      & (iy[pixels] >= y_lo) & (iy[pixels] <= y_hi))

    # classes and bins: rebuild each sorted beam's (class, bin) from starts/counts
    bins, m = config.bins_per_axis, len(config.edges_for(shape))
    starts, counts = np.asarray(plan.starts), np.asarray(plan.counts)
    key = np.repeat(np.arange(len(counts)), counts)
    assert len(key) == n and np.all(np.diff(key) >= 0)
    assert np.array_equal(starts, np.cumsum(counts) - counts)
    cls, cell_index = key // bins**2, key % bins**2
    origin, cell = np.asarray(plan.origin), np.asarray(plan.cell)
    active = np.asarray(footprints(packed, limit, shape)[2])[order]
    assert np.array_equal(cls >= m + 1, ~active)          # inactive beams are behind everything
    for pos in np.flatnonzero(cls < m):                   # sparse classes: candidate coverage
        b, j = order[pos], cls[pos]
        by, bx = divmod(cell_index[pos], bins)
        pixels = np.flatnonzero(keep[:, b])
        pixel_bx = np.clip(np.floor((ix[pixels] - origin[j, 0]) / cell[j]), -2, bins + 1)
        pixel_by = np.clip(np.floor((iy[pixels] - origin[j, 1]) / cell[j]), -2, bins + 1)
        assert np.all(np.abs(pixel_bx - bx) <= 1) and np.all(np.abs(pixel_by - by) <= 1)
    # inactive beams keep no pixel at all
    assert not keep[:, order[cls == m + 1]].any()
    assert np.array_equal(np.asarray(plan.dense_run), [starts[m * bins**2], counts[m * bins**2]])

    # work items partition every window's steps
    lanes = 32 * config.window_warps
    width = np.maximum(windows[:, 1] - windows[:, 0] + 1, 0)
    height = np.maximum(windows[:, 3] - windows[:, 2] + 1, 0)
    steps = height * ((width + lanes - 1) // lanes)
    first, count = np.asarray(plan.item_first), np.asarray(plan.item_count)
    lo, hi = np.asarray(plan.item_lo), np.asarray(plan.item_hi)
    assert plan.item_beam.shape[0] == n + config.items and count.sum() <= n + config.items
    for pos in range(n):
        items = slice(first[pos], first[pos] + count[pos])
        assert np.all(np.asarray(plan.item_beam)[items] == pos)
        assert np.all(lo[items][1:] == hi[items][:-1]) or count[pos] <= 1
        assert (hi[items] - lo[items]).sum() == steps[pos]
        if count[pos]:
            assert lo[items][0] == 0 and hi[items][-1] == steps[pos]
    assert np.all(hi[count.sum():] == lo[count.sum():])   # unused items are empty


def test_plan_is_deterministic_and_jittable():
    shape = (30, 40)
    _, packed, _ = mixed_bundle(3, 200, shape, 20.0)
    spec = sparse_spec(shape, config_for())
    plans = [jax.jit(lambda p: build_plan(p, jnp.zeros(6), spec))(packed) for _ in range(2)]
    for a, b in zip(*plans):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_broad_and_degenerate_envelopes_form_the_dense_class():
    shape = (20, 30)
    _, packed, _ = mixed_bundle(4, 50, shape, 5.0, log_spread=(0, 0))
    config = config_for()
    spec = sparse_spec(shape, config)
    n_sparse = int(np.asarray(build_plan(packed, jnp.zeros(6), spec).dense_run)[1])
    broad = packed.at[0, QCOLS].mul(1e-6)                                # radius >> detector
    flat = packed.at[1, 5].set(0.0).at[1, 7].set(0.0)                    # not positive definite
    singular = packed.at[2, 5].set(1.0).at[2, 9].set(1.0).at[2, 7].set(2.0)  # det == 0
    negative = packed.at[3, 5].set(-1.0)
    for bundle, expected in ((packed, n_sparse), (broad, n_sparse + 1), (flat, n_sparse + 1),
                             (singular, n_sparse + 1), (negative, n_sparse + 1)):
        plan = build_plan(bundle, jnp.zeros(6), spec)
        assert int(np.asarray(plan.dense_run)[1]) == expected


def test_no_classes_means_everything_dense():
    shape = (12, 12)
    _, packed, _ = mixed_bundle(5, 20, shape, 0.0, log_spread=(0, 0), waist=4.0)
    spec = sparse_spec(shape, config_for(class_edges=()))
    plan = build_plan(packed, jnp.zeros(6), spec)
    assert int(plan.dense_run[1]) == int(jnp.sum(footprints(packed, spec.limit, shape)[2]))


# ---------------------------------------------------------------------------
# Awkward bundles through the renderer
# ---------------------------------------------------------------------------
def renderers(shape, impl):
    config = config_for(impl=impl, nt_chunk=4)

    def render(pk):
        return evaluate_packed(pk, shape=shape, config=config)

    def scan(pk):
        h, w = shape
        x, y = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
        return field_scan(pk, jnp.asarray(np.column_stack([x.ravel(), y.ravel()])),
                          config.decay_limit)

    return render, scan


IMPLS = [pytest.param("reference", id="reference"),
         pytest.param("pallas", id="pallas", marks=pytest.mark.gpu)]


@pytest.fixture(params=IMPLS)
def impl(request):
    if request.param == "pallas" and not gpu_kernel_available():
        pytest.skip("requires a verified Pallas GPU backend (JAX_PLATFORMS=cuda)")
    return request.param


def loss_of(field_fn, seed=0):
    rng = np.random.default_rng(seed)

    def loss(pk):
        e = field_fn(pk)
        w = jnp.asarray(rng.normal(size=e.shape))
        return jnp.sum(w * jnp.abs(e) ** 2)
    return loss


def test_heterogeneous_radii_match_the_truncated_scan(impl):
    shape = (61, 83)
    n = 1500 if impl == "pallas" else 80
    _, packed, _ = mixed_bundle(6, n, shape, 30.0)
    render, scan = renderers(shape, impl)
    assert rel(render(packed), scan(packed)) <= 1e-10
    assert rel(jax.grad(loss_of(render))(packed), jax.grad(loss_of(scan))(packed)) <= 1e-9
    t = jnp.asarray(np.random.default_rng(7).normal(size=packed.shape))
    assert rel(jax.jvp(render, (packed,), (t,))[1], jax.jvp(scan, (packed,), (t,))[1]) <= 1e-9


def test_beamlets_off_the_detector_cost_nothing_and_get_zero_gradient(impl):
    shape = (30, 40)
    rng, packed, _ = mixed_bundle(8, 60, shape, 0.0, log_spread=(0, 0), waist=2.0)
    far = jnp.asarray(pixel_bundle(rng, 40, shape, waist_px=2.0, margin=0.0)[0])
    far = far.at[:, 0].add(5000.0).at[:20, 1].add(-7000.0)             # nowhere near the detector
    bundle = jnp.concatenate([packed, far])
    render, scan = renderers(shape, impl)
    assert rel(render(bundle), render(packed)) <= 1e-14
    gradient = jax.grad(loss_of(render))(bundle)
    assert float(jnp.max(jnp.abs(gradient[60:]))) == 0.0
    assert rel(gradient[:60], jax.grad(loss_of(render))(packed)) <= 1e-13
    spec = sparse_spec(shape, config_for())
    plan = build_plan(bundle, jnp.zeros(6), spec)
    assert int(np.asarray(plan.item_count).sum()) <= 60 + 96      # the far beams own no items


def test_every_beamlet_off_the_detector(impl):
    shape = (12, 15)
    _, packed, _ = mixed_bundle(9, 30, shape, 0.0, log_spread=(0, 0))
    packed = packed.at[:, 0].add(1e4)
    render, _ = renderers(shape, impl)
    assert float(jnp.max(jnp.abs(render(packed)))) == 0.0
    assert float(jnp.max(jnp.abs(jax.grad(lambda p: jnp.sum(jnp.abs(render(p))))(packed)))) == 0.0
    assert float(jnp.max(jnp.abs(jax.jvp(render, (packed,), (jnp.ones_like(packed),))[1]))) == 0.0


@pytest.mark.parametrize("poison", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("column", [0, 3, 5, 10, 11])
def test_invalid_beamlets_contribute_zero_and_never_poison_the_others(impl, poison, column):
    shape = (14, 17)
    _, packed, _ = mixed_bundle(10, 24, shape, 3.0, log_spread=(-1, 0), waist=1.6)
    bad = packed.at[7, column].set(poison).at[11, (column + 1) % 12].set(np.nan)
    healthy = jnp.delete(packed, jnp.array([7, 11]), axis=0)
    render, _ = renderers(shape, impl)
    assert np.isfinite(np.asarray(render(bad))).all()
    assert rel(render(bad), render(healthy)) <= 1e-12
    gradient = jax.grad(loss_of(render))(bad)
    assert np.isfinite(np.asarray(gradient)).all()
    assert float(jnp.max(jnp.abs(gradient[jnp.array([7, 11])]))) == 0.0
    expected = jax.grad(loss_of(render))(healthy)
    rest = jnp.delete(gradient, jnp.array([7, 11]), axis=0)
    # the loss weights are drawn from the field size: identical for both bundles
    assert rel(rest, expected) <= 1e-10


def test_all_invalid_bundle_renders_zero(impl):
    shape = (6, 7)
    render, _ = renderers(shape, impl)
    bundle = jnp.full((5, 12), jnp.nan)
    assert float(jnp.max(jnp.abs(render(bundle)))) == 0.0
    assert float(jnp.max(jnp.abs(jax.jacfwd(lambda p: render(p).real)(bundle)))) == 0.0


def test_null_beamlet_is_off_every_detector_and_silent():
    assert NULL_ROW[10] == 0.0 and NULL_ROW[11] == 0.0 and NULL_ROW[0] > 1e20


def test_rectangular_and_degenerate_rasters(impl):
    for shape in [(1, 41), (43, 1), (2, 2), (1, 1), (7, 130)]:
        _, packed, _ = mixed_bundle(11, 30, shape, 6.0, log_spread=(-1, 0), waist=1.5)
        render, scan = renderers(shape, impl)
        assert rel(render(packed) + 1e-300, scan(packed) + 1e-300) <= 1e-10, shape
        assert rel(jax.grad(loss_of(render))(packed), jax.grad(loss_of(scan))(packed)) <= 1e-9, shape


def test_a_single_broad_beamlet_among_many_narrow_ones_is_exact(impl):
    """The mixed-radius case that broke a single global bin size (timing: benchmarks/render.py)."""
    shape = (48, 64)
    n = 600 if impl == "pallas" else 60
    rng = np.random.default_rng(12)
    packed = jnp.asarray(pixel_bundle(rng, n, shape, waist_px=1.5, margin=4.0)[0])
    packed = packed.at[0, QCOLS].mul((1.5 / 30.0) ** 2)                # waist 30 px
    render, scan = renderers(shape, impl)
    assert rel(render(packed), scan(packed)) <= 1e-10


@pytest.mark.parametrize("cutoff", [1e-3, 1e-8, 1e-12])
def test_truncation_error_respects_the_cutoff_bound(impl, cutoff):
    shape = (33, 41)
    _, packed, coords = mixed_bundle(13, 90, shape, 8.0, log_spread=(-1, 0), waist=2.0)
    config = RenderConfig(cutoff=cutoff, impl=impl, bins_per_axis=16, n_classes=4, items=64)
    exact = evaluate_packed(packed, shape=shape, config=config, exact=True)
    truncated = evaluate_packed(packed, shape=shape, config=config)
    bound = cutoff * float(jnp.sum(jnp.hypot(packed[:, 10], packed[:, 11])))
    assert float(jnp.max(jnp.abs(truncated - exact))) <= bound * (1 + 1e-9) + 1e-13


def test_bin_boundaries_are_covered():
    """Pixels exactly on bin edges: integer centres and an integer cell size."""
    shape, bins = (40, 256), 32
    rng = np.random.default_rng(14)
    packed = pixel_bundle(rng, 200, shape, waist_px=0.8, margin=0.0)[0]
    packed[:, 0] = rng.integers(0, 32, 200) * 8.0                      # x span 248 = 31 * 8
    packed[:, 1] = rng.integers(0, 40, 200).astype(float)
    packed[0, 0], packed[1, 0] = 0.0, 248.0
    packed = jnp.asarray(packed)
    config = RenderConfig(cutoff=CUTOFF, bins_per_axis=bins, n_classes=1, class_edges=(60.0,))
    spec = sparse_spec(shape, config)
    plan = build_plan(packed, jnp.zeros(6), spec)
    assert float(plan.cell[0]) == pytest.approx(8.0, rel=1e-9)
    x, y = np.meshgrid(np.arange(256.0), np.arange(40.0))
    xy = jnp.asarray(np.column_stack([x.ravel(), y.ravel()]))
    from temgym_core.render.kernels import reference as ref
    keep = predicate_mask(packed, xy, config.decay_limit)
    # each kept pair lies within one bin of each other, also on the exact boundaries
    origin, cell = np.asarray(plan.origin[0]), float(plan.cell[0])
    px = np.floor((np.asarray(xy)[:, 0, None] - origin[0]) / cell)
    bx = np.floor((np.asarray(packed)[None, :, 0] - origin[0]) / cell)
    assert np.all(np.abs(px - bx)[keep] <= 1)
    assert float(jnp.max(jnp.abs(ref.field(packed, xy, limit=config.decay_limit)))) > 0


@pytest.mark.gpu
def test_gpu_fuzz_sparse_kernels_against_dense_with_predicate(gpu):
    """Random shapes (incl. 1xW, Hx1, 2x2), beams off the image, cutoffs, bins, classes and nt."""
    from temgym_core.render.primitives import adjoint_map, tangent_map

    master = np.random.default_rng(2026)
    for _ in range(8):
        rng = np.random.default_rng(master.integers(1 << 30))
        h = int(rng.choice([1, 2, 3, 5, 13, 31, 64, 97]))
        w = int(rng.choice([1, 2, 7, 16, 33, 100, 129]))
        n = int(rng.choice([300, 800, 1500]))
        packed = pixel_bundle(rng, n, (h, w), waist_px=float(rng.choice([0.5, 1.5, 4.0])),
                              margin=float(rng.choice([0.0, 5.0, 30.0])))[0]
        packed[:, QCOLS] *= 10 ** rng.uniform(-2, 0, (n, 1)) * (rng.random((n, 1)) < 0.05) \
            + (rng.random((n, 1)) >= 0.05)                      # a few broad beams
        packed = jnp.asarray(packed)
        x, y = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
        xy = jnp.asarray(np.column_stack([x.ravel(), y.ravel()]))
        config = RenderConfig(cutoff=float(rng.choice([1e-6, 1e-10, 1e-12])), nt_chunk=4,
                              bins_per_axis=int(rng.choice([4, 16, 64])),
                              n_classes=int(rng.choice([1, 3, 6])), items=int(rng.choice([32, 512])))
        nt = int(rng.integers(1, 4))
        tdot = jnp.asarray(rng.normal(size=(nt,) + packed.shape))
        ct = jnp.asarray(rng.normal(size=(nt, h * w)) + 1j * rng.normal(size=(nt, h * w)))
        got = (evaluate_packed(packed, shape=(h, w), config=config),
               tangent_map(packed, shape=(h, w), tdot=tdot, config=config),
               adjoint_map(packed, shape=(h, w), ct=ct, config=config))
        want = (evaluate_packed(packed, xy, config=config),
                tangent_map(packed, xy, tdot, config=config),
                adjoint_map(packed, xy, ct, config=config))
        for a, b in zip(got, want):
            scale = max(float(jnp.max(jnp.abs(b))), 1e-300)
            assert float(jnp.max(jnp.abs(a - b))) <= 1e-11 * scale + 1e-14, (h, w, n, config)


# ---------------------------------------------------------------------------
# Index-range guards (no silent int32 overflow)
# ---------------------------------------------------------------------------
def test_too_many_pixels_or_beamlets_raise_a_clear_error():
    config = RenderConfig()
    with pytest.raises(ValueError, match="pixels exceed the int32 index range"):
        jax.eval_shape(lambda p: evaluate_packed(p, shape=(50000, 50000), config=config),
                       jax.ShapeDtypeStruct((4, 12), jnp.float64))
    with pytest.raises(ValueError, match="beamlets exceed the int32 index range"):
        jax.eval_shape(lambda p: evaluate_packed(p, shape=(4, 4), config=config),
                       jax.ShapeDtypeStruct((MAX_BEAMS, 12), jnp.float64))
    assert MAX_INDEX < 2**31


@pytest.mark.gpu
@pytest.mark.slow
def test_more_than_two_to_the_24_pixels_are_indexed_exactly(gpu):
    """4104 x 4104 = 16.8M pixels (above 2**24): field and gradient agree with a probe of it."""
    shape = (4104, 4104)
    assert shape[0] * shape[1] > 2**24
    rng = np.random.default_rng(15)
    packed = pixel_bundle(rng, 40, shape, waist_px=6.0, margin=-5.0)[0]
    packed[:, 0] = rng.uniform(4000, 4103, 40)     # the last rows and columns: largest indices
    packed[:, 1] = rng.uniform(4000, 4103, 40)
    packed = jnp.asarray(packed)
    config = RenderConfig(cutoff=CUTOFF, items=512)
    field = evaluate_packed(packed, shape=shape, config=config)
    lo = 3900                                      # beyond every footprint (radius < 100 px)
    window = np.s_[lo:4104, lo:4104]
    rows, cols = np.meshgrid(np.arange(lo, 4104), np.arange(lo, 4104), indexing="ij")
    xy = jnp.asarray(np.column_stack([cols.ravel(), rows.ravel()]).astype(float))
    expected = field_scan(packed, xy, config.decay_limit)
    assert rel(field.reshape(shape)[window].ravel(), expected) <= 1e-10
    assert float(jnp.max(jnp.abs(field.reshape(shape)[:lo]))) == 0.0
    gradient = jax.grad(lambda p: jnp.sum(jnp.abs(evaluate_packed(p, shape=shape, config=config)) ** 2))(packed)
    reference = jax.grad(lambda p: jnp.sum(jnp.abs(field_scan(p, xy, config.decay_limit)) ** 2))(packed)
    assert rel(gradient, reference) <= 1e-9


@pytest.mark.gpu
@pytest.mark.slow
def test_more_than_two_to_the_24_beamlets_are_planned_exactly(gpu):
    """16.8M beamlets, almost all far away: the plan's counters must not wrap around."""
    n = 2**24 + 5
    shape = (48, 64)
    rng = np.random.default_rng(16)
    real = pixel_bundle(rng, 300, shape, waist_px=2.0, margin=2.0)[0]
    base = np.tile(NULL_ROW, (n - 300, 1))
    base[:, 0] = 1e6 + np.arange(n - 300) % 1000.0
    base[:, 1] = 5.0
    packed = jnp.concatenate([jnp.asarray(real), jnp.asarray(base)])
    config = dataclasses.replace(RenderConfig(cutoff=CUTOFF), items=1024)
    field = evaluate_packed(packed, shape=shape, config=config)
    x, y = np.meshgrid(np.arange(64.0), np.arange(48.0))
    xy = jnp.asarray(np.column_stack([x.ravel(), y.ravel()]))
    assert rel(field, field_scan(jnp.asarray(real), xy, config.decay_limit)) <= 1e-10
    plan = build_plan(packed, jnp.zeros(6), sparse_spec(shape, config))
    assert int(plan.counts.sum()) == n
