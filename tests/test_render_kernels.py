"""The Pallas kernels against the pure-JAX reference.

CPU tests run the kernel bodies in Pallas interpret mode on tiny problems (kernel logic,
padding, tangent-count chunking, the pair predicate, plan-driven traversals).  Tests marked
``gpu`` compile the Triton kernels in float64: run with ``JAX_PLATFORMS=cuda``.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legacy_dense_kernel import legacy_dense_kernel
from render_bundles import pixel_bundle, random_packed, rel, scattered_coords, unit_grid

from temgym_core.evaluate import gpu_kernel_available
from temgym_core.render import RenderConfig, Tiles
from temgym_core.render.config import KernelSpec
from temgym_core.render.kernels import dense, sparse
from temgym_core.render.kernels.common import plgpu
from temgym_core.render.kernels import reference as ref
from temgym_core.render.plan import build_plan
from temgym_core.render.predicate import support_limit

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def gpu():
    if not gpu_kernel_available():
        pytest.skip("requires a verified Pallas GPU backend (JAX_PLATFORMS=cuda)")


def dense_spec(coords, config, exact=True):
    return KernelSpec(config, coords.shape[0], None, "array", sparse=False, truncate=not exact)


def sparse_spec(shape, config):
    return KernelSpec(config, shape[0] * shape[1], shape, "index", sparse=True)


def problem(seed, n, n_pix, width=0.3, grid=False):
    rng = np.random.default_rng(seed)
    packed = jnp.asarray(random_packed(rng, n, width=width))
    coords = jnp.asarray(unit_grid(*n_pix) if grid else scattered_coords(rng, n_pix))
    return rng, packed, coords


# ---------------------------------------------------------------------------
# CPU: interpret-mode dense kernels on tiny problems (odd sizes force padding)
# ---------------------------------------------------------------------------
INTERPRET = RenderConfig(impl="pallas", nt_chunk=4).with_tiles(8, 4, 1)
needs_triton = pytest.mark.skipif(plgpu is None, reason="jax.experimental.pallas.triton is missing")


@needs_triton
@pytest.mark.parametrize("cutoff", [None, 1e-2])
def test_interpret_dense_kernels_match_reference(cutoff):
    config = INTERPRET if cutoff is None else dataclasses.replace(INTERPRET, cutoff=cutoff)
    spec = dense_spec(problem(0, 13, 21)[2], config, exact=cutoff is None)
    limit = spec.limit
    rng, packed, coords = problem(0, 13, 21)
    assert rel(dense.forward(packed, coords, spec=spec, interpret=True),
               ref.field(packed, coords, limit=limit)) <= 1e-12
    for nt in (1, 3, 9):  # 9 > nt_chunk: exercises chunking with zero padding
        w = jnp.asarray(rng.normal(size=(nt, 13, 12)))
        ct = jnp.asarray(rng.normal(size=(nt, 21)) + 1j * rng.normal(size=(nt, 21)))
        assert rel(dense.tangent(packed, coords, w, spec=spec, interpret=True),
                   ref.polynomial_sum(packed, coords, w, limit=limit)) <= 1e-12
        assert rel(dense.adjoint(packed, coords, ct, spec=spec, interpret=True),
                   ref.polynomial_moments(packed, coords, ct, limit=limit)) <= 1e-12


@needs_triton
def test_interpret_kernel_shapes_and_dtypes():
    rng, packed, coords = problem(1, 5, 7)
    spec = dense_spec(coords, INTERPRET)
    w = jnp.asarray(rng.normal(size=(2, 5, 12)))
    assert dense.forward(packed, coords, spec=spec, interpret=True).shape == (7,)
    out = dense.tangent(packed, coords, w, spec=spec, interpret=True)
    assert out.shape == (2, 7) and out.dtype == jnp.complex128
    ct = jnp.ones((2, 7), jnp.complex128)
    moments = dense.adjoint(packed, coords, ct, spec=spec, interpret=True)
    assert moments.shape == (2, 5, 12) and moments.dtype == jnp.float64


@needs_triton
def test_interpret_dense_run_visits_only_its_beams():
    rng, packed, coords = problem(2, 12, 10)
    spec = dense_spec(coords, INTERPRET)
    run = jnp.array([3, 5], jnp.int32)
    got = dense.pixel_major_run(packed, coords, None, run, spec=spec, tiles=INTERPRET.forward,
                                interpret=True, name="test_run")
    assert rel(got[0], ref.field(packed[3:8], coords)) <= 1e-12
    empty = dense.pixel_major_run(packed, coords, None, jnp.array([4, 0], jnp.int32), spec=spec,
                                  tiles=INTERPRET.forward, interpret=True, name="test_run")
    assert float(jnp.max(jnp.abs(empty))) == 0.0


@needs_triton
def test_interpret_sparse_kernels_match_reference():
    """Forward, tangent and adjoint through the plan (two classes, off-image beams)."""
    shape = (9, 13)
    config = RenderConfig(impl="pallas", cutoff=1e-8, nt_chunk=2, bins_per_axis=4, n_classes=2,
                          items=6, rows=Tiles(4, 2, 1))
    rng = np.random.default_rng(3)
    packed, coords, _ = pixel_bundle(rng, 30, shape, waist_px=1.2, margin=4.0)
    packed = jnp.asarray(packed).at[:3, 5].mul(0.05).at[:3, 9].mul(0.05)  # three broad beams
    spec = sparse_spec(shape, config)
    plan = tuple(build_plan(packed, jnp.zeros(6), spec))
    limit, xy = spec.limit, jnp.asarray(coords)
    assert rel(sparse.forward(packed, None, *plan, spec=spec, interpret=True),
               ref.field(packed, xy, limit=limit)) <= 1e-11
    w = jnp.asarray(rng.normal(size=(3, 30, 12)))
    assert rel(sparse.tangent(packed, None, *plan, w, spec=spec, interpret=True),
               ref.polynomial_sum(packed, xy, w, limit=limit)) <= 1e-11
    ct = jnp.asarray(rng.normal(size=(2, 117)) + 1j * rng.normal(size=(2, 117)))
    assert rel(sparse.adjoint(packed, None, *plan, ct, spec=spec, interpret=True),
               ref.polynomial_moments(packed, xy, ct, limit=limit)) <= 1e-11


# ---------------------------------------------------------------------------
# GPU: compiled Triton kernels
# ---------------------------------------------------------------------------
TILES_FORWARD_TANGENT = [(32, 8, 4), (16, 8, 2), (64, 4, 4), (16, 16, 4)]
TILES_ADJOINT = [(4, 32, 4), (8, 16, 4), (16, 8, 4), (4, 16, 2)]


@pytest.mark.gpu
@pytest.mark.parametrize("tile_pixels,tile_beams,num_warps", TILES_FORWARD_TANGENT)
def test_gpu_dense_forward_and_tangent_match_reference(gpu, tile_pixels, tile_beams, num_warps):
    tiles = Tiles(tile_pixels, tile_beams, num_warps)
    config = RenderConfig(forward=tiles, tangent=tiles)
    rng, packed, coords = problem(3, 203, (61, 59), width=0.2, grid=True)  # odd sizes
    spec = dense_spec(coords, config)
    fwd = jax.jit(lambda a, b: dense.forward(a, b, spec=spec))(packed, coords)
    assert rel(fwd, ref.field(packed, coords)) <= 1e-12
    for nt in (1, 7, 11):
        w = jnp.asarray(rng.normal(size=(nt, 203, 12)))
        out = jax.jit(lambda a, b, c: dense.tangent(a, b, c, spec=spec))(packed, coords, w)
        assert rel(out, ref.polynomial_sum(packed, coords, w)) <= 1e-10


@pytest.mark.gpu
@pytest.mark.parametrize("tile_beams,tile_pixels,num_warps", TILES_ADJOINT)
def test_gpu_dense_adjoint_matches_reference(gpu, tile_beams, tile_pixels, num_warps):
    config = RenderConfig(adjoint=Tiles(tile_pixels, tile_beams, num_warps))
    rng, packed, coords = problem(4, 203, (61, 59), width=0.2, grid=True)
    spec = dense_spec(coords, config)
    for nt in (1, 3):
        ct = jnp.asarray(rng.normal(size=(nt, coords.shape[0]))
                         + 1j * rng.normal(size=(nt, coords.shape[0])))
        out = jax.jit(lambda a, b, c: dense.adjoint(a, b, c, spec=spec))(packed, coords, ct)
        assert rel(out, ref.polynomial_moments(packed, coords, ct)) <= 1e-10


def sparse_problem(seed=5, n=3000, shape=(70, 90), waist=2.5, margin=20.0, **config):
    rng = np.random.default_rng(seed)
    packed, coords, _ = pixel_bundle(rng, n, shape, waist_px=waist, margin=margin)
    packed = jnp.asarray(packed)
    # a few broad beams, so several radius classes and the dense class are occupied
    packed = packed.at[:4, [5, 7, 9]].mul(0.02).at[4:20, [5, 7, 9]].mul(0.3)
    config = RenderConfig(cutoff=1e-10, bins_per_axis=32, items=512, **config)
    spec = sparse_spec(shape, config)
    return rng, packed, jnp.asarray(coords), spec, tuple(build_plan(packed, jnp.zeros(6), spec))


@pytest.mark.gpu
@pytest.mark.parametrize("rows,warps,edges", [
    (Tiles(16, 4, 1), 1, None),
    (Tiles(32, 4, 2), 1, None),
    (Tiles(16, 8, 1), 2, (10.0, 30.0)),
    (Tiles(64, 8, 4), 1, (4.0, 8.0, 16.0, 32.0)),
    (Tiles(16, 4, 1), 1, ()),
])
def test_gpu_sparse_kernels_match_dense_with_predicate(gpu, rows, warps, edges):
    rng, packed, xy, spec, plan = sparse_problem(rows=rows, window_warps=warps, class_edges=edges)
    limit = spec.limit
    assert rel(jax.jit(lambda p, *pl: sparse.forward(p, None, *pl, spec=spec))(packed, *plan),
               ref.field(packed, xy, limit=limit)) <= 1e-11
    for nt in (1, 3, 9):
        w = jnp.asarray(rng.normal(size=(nt, packed.shape[0], 12)))
        assert rel(sparse.tangent(packed, None, *plan, w, spec=spec),
                   ref.polynomial_sum(packed, xy, w, limit=limit)) <= 1e-11
    ct = jnp.asarray(rng.normal(size=(2, xy.shape[0])) + 1j * rng.normal(size=(2, xy.shape[0])))
    assert rel(sparse.adjoint(packed, None, *plan, ct, spec=spec),
               ref.polynomial_moments(packed, xy, ct, limit=limit)) <= 1e-11


@pytest.mark.gpu
def test_gpu_dot_product_of_the_dense_kernel_pair(gpu):
    """K and K^T as compiled kernels are exact transposes (real pairing), <= 1e-13."""
    rng, packed, coords = problem(5, 500, (48, 48), width=0.15, grid=True)
    spec = dense_spec(coords, RenderConfig())
    w = jnp.asarray(rng.normal(size=(2, 500, 12)))
    ct = jnp.asarray(rng.normal(size=(2, coords.shape[0])) + 1j * rng.normal(size=(2, coords.shape[0])))
    tangent = dense.tangent(packed, coords, w, spec=spec)
    moments = dense.adjoint(packed, coords, ct, spec=spec)
    lhs = jnp.real(jnp.sum(ct * tangent, axis=-1))
    rhs = jnp.sum(w * moments, axis=(1, 2))
    scale = jnp.linalg.norm(tangent) * jnp.linalg.norm(ct)
    assert float(jnp.max(jnp.abs(lhs - rhs)) / scale) <= 1e-13


@pytest.mark.gpu
def test_gpu_dot_product_between_the_two_sparse_traversals(gpu):
    """Pixel-major tangent and beam-major adjoint keep the same pairs: |<Kw, ct> - <w, K^T ct>|."""
    rng, packed, xy, spec, plan = sparse_problem()
    w = jnp.asarray(rng.normal(size=(2, packed.shape[0], 12)))
    ct = jnp.asarray(rng.normal(size=(2, xy.shape[0])) + 1j * rng.normal(size=(2, xy.shape[0])))
    tangent = sparse.tangent(packed, None, *plan, w, spec=spec)
    moments = sparse.adjoint(packed, None, *plan, ct, spec=spec)
    lhs = jnp.real(jnp.sum(ct * tangent, axis=-1))
    rhs = jnp.sum(w * moments, axis=(1, 2))
    scale = jnp.linalg.norm(tangent) * jnp.linalg.norm(ct)
    assert float(jnp.max(jnp.abs(lhs - rhs)) / scale) <= 1e-13


@pytest.mark.gpu
def test_gpu_dot_product_breaks_without_the_shared_predicate(gpu):
    """Evidence that the shared predicate matters: a different cutoff in one kernel is visible."""
    rng, packed, xy, spec, plan = sparse_problem()
    other = dataclasses.replace(spec, render=dataclasses.replace(spec.render, cutoff=1e-4))
    w = jnp.asarray(rng.normal(size=(1, packed.shape[0], 12)))
    ct = jnp.asarray(rng.normal(size=(1, xy.shape[0])) + 1j * rng.normal(size=(1, xy.shape[0])))
    tangent = sparse.tangent(packed, None, *plan, w, spec=spec)
    moments = sparse.adjoint(packed, None, *plan, ct, spec=other)
    lhs, rhs = jnp.real(jnp.sum(ct * tangent)), jnp.sum(w * moments)
    assert float(jnp.abs(lhs - rhs) / (jnp.linalg.norm(tangent) * jnp.linalg.norm(ct))) > 1e-8


@pytest.mark.gpu
def test_gpu_every_kernel_is_bitwise_repeatable(gpu):
    rng, packed, coords = problem(7, 400, (40, 40), width=0.2, grid=True)
    dspec = dense_spec(coords, RenderConfig())
    w = jnp.asarray(rng.normal(size=(3, 400, 12)))
    ct = jnp.asarray(rng.normal(size=(3, coords.shape[0])) + 0j)
    runs = [
        [np.asarray(dense.forward(packed, coords, spec=dspec)) for _ in range(3)],
        [np.asarray(dense.tangent(packed, coords, w, spec=dspec)) for _ in range(3)],
        [np.asarray(dense.adjoint(packed, coords, ct, spec=dspec)) for _ in range(3)],
    ]
    rng, sp, xy, spec, plan = sparse_problem(n=1500)
    ws = jnp.asarray(rng.normal(size=(2, sp.shape[0], 12)))
    cts = jnp.asarray(rng.normal(size=(2, xy.shape[0])) + 1j * rng.normal(size=(2, xy.shape[0])))
    runs += [
        [np.asarray(sparse.forward(sp, None, *plan, spec=spec)) for _ in range(3)],
        [np.asarray(sparse.tangent(sp, None, *plan, ws, spec=spec)) for _ in range(3)],
        [np.asarray(sparse.adjoint(sp, None, *plan, cts, spec=spec)) for _ in range(3)],
        [np.asarray(build_plan(sp, jnp.zeros(6), spec).order) for _ in range(3)],
    ]
    for group in runs:
        assert all(np.array_equal(group[0], r) for r in group[1:])


@pytest.mark.gpu
def test_gpu_predicate_truncation_bound_and_tangent_adjoint(gpu):
    rng = np.random.default_rng(8)
    packed, coords, _ = pixel_bundle(rng, 2000, (64, 64), waist_px=2.5)
    packed, coords_j = jnp.asarray(packed), jnp.asarray(coords)
    cutoff = 1e-10
    truncated_spec = dense_spec(coords_j, RenderConfig(cutoff=cutoff), exact=False)
    limit = support_limit(cutoff)
    field = dense.forward(packed, coords_j, spec=truncated_spec)
    assert rel(field, ref.field(packed, coords_j, limit=limit)) <= 1e-11
    exact = dense.forward(packed, coords_j, spec=dense_spec(coords_j, RenderConfig()))
    bound = cutoff * float(jnp.sum(jnp.hypot(packed[:, 10], packed[:, 11])))
    assert float(jnp.max(jnp.abs(field - exact))) <= bound
    w = jnp.asarray(rng.normal(size=(2, 2000, 12)))
    assert rel(dense.tangent(packed, coords_j, w, spec=truncated_spec),
               ref.polynomial_sum(packed, coords_j, w, limit=limit)) <= 1e-10


@pytest.mark.gpu
def test_gpu_dense_forward_agrees_with_the_frozen_legacy_kernel(gpu):
    rng = np.random.default_rng(9)
    packed, coords, (r, dr, amp, path, q, k) = pixel_bundle(rng, 3000, (96, 80), waist_px=2.5)
    args = tuple(jnp.asarray(a) for a in (r, dr, amp, path, q, k, coords))
    legacy = legacy_dense_kernel(*args, tile_pixels=16, tile_beams=16)
    coords_j = jnp.asarray(coords)
    mine = dense.forward(jnp.asarray(packed), coords_j, spec=dense_spec(coords_j, RenderConfig()))
    assert rel(mine, legacy) <= 1e-12
