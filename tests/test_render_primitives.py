"""The forward / tangent / adjoint primitives and their JAX transform rules.

Every check compares against ordinary autodiff of an independent scan renderer at the same
packed inputs.  The transform matrix runs for the dense kernels (``exact=True``, arbitrary
pixel positions) and for the sparse kernels (pixel-index frame, plan operands), each with the
pure-JAX implementation (``impl="reference"``, any platform) and with the Pallas kernels
(``impl="pallas"``: run with ``JAX_PLATFORMS=cuda``, marked ``gpu``; on CPU the sparse
kernels run in interpret mode on a tiny grid).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from render_bundles import (equations, field_scan, pixel_bundle, random_packed, rel, unit_grid)

from temgym_core.evaluate import gpu_kernel_available
from temgym_core.render import RenderConfig, Tiles, evaluate_packed
from temgym_core.render.kernels.common import plgpu
from temgym_core.render.plan import RenderPlan

jax.config.update("jax_enable_x64", True)

N_PLAN = len(RenderPlan._fields)


class Problem:
    """One bundle, one set of pixels, a renderer under test and its scan reference."""

    def __init__(self, kind, config, n, shape, seed=1, **bundle):
        rng = np.random.default_rng(seed)
        self.kind, self.config, self.rng, self.n, self.shape = kind, config, rng, n, shape
        if kind == "dense":
            self.packed = jnp.asarray(random_packed(rng, n, width=0.4))
            self.coords = jnp.asarray(unit_grid(*shape))
            self.limit = None
        else:
            packed, coords, _ = pixel_bundle(rng, n, shape, **bundle)
            self.packed, self.coords = jnp.asarray(packed), jnp.asarray(coords)
            self.limit = config.decay_limit
        self.n_pix = self.coords.shape[0]
        self.weights = jnp.asarray(rng.normal(size=self.n_pix) + 1j * rng.normal(size=self.n_pix))
        self.basis = jnp.asarray(rng.normal(size=(7, n, 12)))
        if kind != "dense":  # keep the envelopes positive definite under perturbation
            self.basis = self.basis.at[:, :, [5, 7, 9]].mul(0.01)

    def render(self, pk, coords=None):
        if self.kind == "dense":
            return evaluate_packed(pk, self.coords if coords is None else coords,
                                   config=self.config, exact=True)
        return evaluate_packed(pk, shape=self.shape, config=self.config)

    def scan(self, pk, coords=None):
        return field_scan(pk, self.coords if coords is None else coords, self.limit)

    def loss(self, field_fn):
        """Real loss built from |E|^2 and Im(E): exercises both cotangent components."""
        w = self.weights

        def loss(pk):
            e = field_fn(pk)
            return jnp.sum(jnp.abs(e) ** 2 * w.real) + jnp.sum(e.imag * w.imag)
        return loss


DENSE_REF = RenderConfig(impl="reference", pixel_chunk=16)
DENSE_PALLAS = RenderConfig(impl="pallas", nt_chunk=4).with_tiles(16, 8, 4)
SPARSE_REF = RenderConfig(impl="reference", cutoff=1e-10, nt_chunk=4, pixel_chunk=64)
SPARSE_PALLAS = RenderConfig(impl="pallas", cutoff=1e-10, nt_chunk=4, bins_per_axis=32,
                             items=256)
SPARSE_INTERPRET = RenderConfig(impl="pallas", cutoff=1e-10, nt_chunk=2, bins_per_axis=8,
                                n_classes=2, items=8, rows=Tiles(4, 2, 1))

PROBLEMS = {
    "dense-reference": lambda: Problem("dense", DENSE_REF, 12, (6, 6)),
    "dense-pallas": lambda: Problem("dense", DENSE_PALLAS, 12, (6, 6)),
    "sparse-reference": lambda: Problem("sparse", SPARSE_REF, 60, (13, 17), waist_px=1.6,
                                        margin=4.0),
    "sparse-pallas": lambda: Problem("sparse", SPARSE_PALLAS, 1500, (61, 83), waist_px=2.5,
                                     margin=25.0),
}


@pytest.fixture(
    params=[
        pytest.param("dense-reference", id="dense-reference"),
        pytest.param("dense-pallas", id="dense-pallas", marks=pytest.mark.gpu),
        pytest.param("sparse-reference", id="sparse-reference"),
        pytest.param("sparse-pallas", id="sparse-pallas", marks=pytest.mark.gpu),
    ]
)
def problem(request):
    if request.param.endswith("pallas") and not gpu_kernel_available():
        pytest.skip("requires a verified Pallas GPU backend (JAX_PLATFORMS=cuda)")
    return PROBLEMS[request.param]()


def real_parts(fn):
    return lambda x: (lambda e: jnp.concatenate([e.real, e.imag]))(fn(x))


def test_primal_matches_scan(problem):
    pk = problem.packed
    assert rel(problem.render(pk), problem.scan(pk)) <= 1e-11


def test_jvp(problem):
    pk = problem.packed
    t = jnp.asarray(problem.rng.normal(size=pk.shape))
    out, tangent = jax.jvp(problem.render, (pk,), (t,))
    out_ref, tangent_ref = jax.jvp(problem.scan, (pk,), (t,))
    assert rel(out, out_ref) <= 1e-11
    assert rel(tangent, tangent_ref) <= 1e-10


def test_grad_and_vjp(problem):
    pk, w = problem.packed, problem.weights
    assert rel(jax.grad(problem.loss(problem.render))(pk),
               jax.grad(problem.loss(problem.scan))(pk)) <= 1e-10
    ct = w * (1 + 0.5j)
    _, pull = jax.vjp(problem.render, pk)
    _, pull_ref = jax.vjp(problem.scan, pk)
    assert rel(pull(ct)[0], pull_ref(ct)[0]) <= 1e-10


def with_theta(problem, fn):
    return lambda th: fn(problem.packed + jnp.einsum("c,cnj->nj", th, problem.basis))


def test_jacfwd_is_one_fused_tangent_equation(problem):
    theta = jnp.asarray(problem.rng.normal(size=7) * 0.05)
    jac = jax.jacfwd(with_theta(problem, problem.render))(theta)
    assert rel(jac, jax.jacfwd(with_theta(problem, problem.scan))(theta)) <= 1e-10
    fn = jax.jacfwd(with_theta(problem, problem.render))
    tangents = equations(fn, theta, name="gaussian_tangent")
    assert len(tangents) == 1, "jacfwd must issue exactly one fused tangent call"
    assert tangents[0].outvars[0].aval.shape == (7, problem.n_pix)
    assert len(equations(fn, theta, name="gaussian_forward")) == 1


def test_jacrev_is_one_fused_adjoint_equation(problem):
    theta = jnp.asarray(problem.rng.normal(size=7) * 0.05)
    fn = jax.jacrev(real_parts(with_theta(problem, problem.render)))
    assert rel(fn(theta), jax.jacrev(real_parts(with_theta(problem, problem.scan)))(theta)) <= 1e-10
    adjoints = equations(fn, theta, name="gaussian_adjoint")
    assert len(adjoints) == 1
    assert adjoints[0].outvars[0].aval.shape == (2 * problem.n_pix, problem.n, 12)


def test_jit_grad(problem):
    pk = problem.packed
    got = jax.jit(jax.grad(problem.loss(problem.render)))(pk)
    assert rel(got, jax.grad(problem.loss(problem.scan))(pk)) <= 1e-10


def bundles(problem, count):
    rng = problem.rng
    if problem.kind == "dense":
        return jnp.asarray(np.stack([random_packed(rng, problem.n, width=0.4)
                                     for _ in range(count)]))
    return jnp.asarray(np.stack([
        pixel_bundle(rng, problem.n, problem.shape, waist_px=2.0, margin=3.0)[0]
        for _ in range(count)]))


def test_vmap_grad_over_bundles(problem):
    batch = bundles(problem, 3)
    loss_r, loss_s = problem.loss(problem.render), problem.loss(problem.scan)
    expected = jax.vmap(jax.grad(loss_s))(batch)
    assert rel(jax.vmap(jax.grad(loss_r))(batch), expected) <= 1e-10
    assert rel(jax.jit(jax.vmap(jax.grad(loss_r)))(batch), expected) <= 1e-10
    assert rel(jax.vmap(problem.render)(batch), jax.vmap(problem.scan)(batch)) <= 1e-11
    # the primal is mapped sequentially: no kernel primitive ever sees a batched bundle
    for name in ("gaussian_forward", "gaussian_adjoint"):
        calls = equations(jax.vmap(jax.grad(loss_r)), batch, name=name)
        assert calls, name
        assert all(e.invars[0].aval.shape == (problem.n, 12) for e in calls), name
    tangents = jnp.asarray(problem.rng.normal(size=batch.shape))
    for name in ("gaussian_forward", "gaussian_tangent"):
        calls = equations(lambda b, t: jax.vmap(
            lambda x, y: jax.jvp(problem.render, (x,), (y,)))(b, t), batch, tangents, name=name)
        assert calls, name
        assert all(e.invars[0].aval.shape == (problem.n, 12) for e in calls), name


def test_vmap_over_tangents_folds_into_tangent_count(problem):
    """vmap of jvp over tangent directions (batched linear operand only) is one fused call."""
    pk = problem.packed
    tangents = jnp.asarray(problem.rng.normal(size=(5,) + pk.shape))

    def many(fn):
        return lambda ts: jax.vmap(lambda t: jax.jvp(fn, (pk,), (t,))[1])(ts)

    assert rel(many(problem.render)(tangents), many(problem.scan)(tangents)) <= 1e-10
    calls = equations(many(problem.render), tangents, name="gaussian_tangent")
    assert len(calls) == 1 and calls[0].outvars[0].aval.shape == (5, problem.n_pix)
    nested = jnp.asarray(problem.rng.normal(size=(2, 3) + pk.shape))
    assert rel(jax.vmap(many(problem.render))(nested), jax.vmap(many(problem.scan))(nested)) <= 1e-10


def test_vmap_over_cotangents_folds_into_one_adjoint(problem):
    pk, rng = problem.packed, problem.rng
    cts = jnp.asarray(rng.normal(size=(4, problem.n_pix)) + 1j * rng.normal(size=(4, problem.n_pix)))

    def many(fn):
        return lambda cs: jax.vmap(lambda ct: jax.vjp(fn, pk)[1](ct)[0])(cs)

    assert rel(many(problem.render)(cts), many(problem.scan)(cts)) <= 1e-10
    calls = equations(many(problem.render), cts, name="gaussian_adjoint")
    assert len(calls) == 1 and calls[0].outvars[0].aval.shape == (4, problem.n, 12)


def test_vmap_over_coordinate_sets_runs_sequentially():
    """Batched residuals (here: pixel coordinates) cannot be fused; results still agree."""
    p = Problem("dense", DENSE_REF, 12, (6, 6))
    grids = jnp.stack([jnp.asarray(unit_grid(6, 6, box=b)) for b in (1.0, 1.2, 1.5)])

    def grads(fn_of_coords):
        return lambda cs: jax.vmap(
            lambda cc: jax.grad(p.loss(lambda pk: fn_of_coords(pk, cc)))(p.packed))(cs)

    assert rel(grads(p.render)(grids), grads(p.scan)(grids)) <= 1e-10
    for name in ("gaussian_forward", "gaussian_adjoint"):
        calls = equations(grads(p.render), grids, name=name)
        assert calls, name
        assert all(e.invars[1].aval.shape == (p.n_pix, 2) for e in calls), name


def test_checkpoint_grad(problem):
    pk = problem.packed

    def block(fn, remat):
        def inner(q):
            return fn(q) * jnp.exp(1j * jnp.sum(q[:, 0]))
        body = jax.checkpoint(inner) if remat else inner
        return lambda q: jnp.sum(jnp.abs(body(q)) ** 2)

    expected = jax.grad(block(problem.scan, False))(pk)
    assert rel(jax.grad(block(problem.render, True))(pk), expected) <= 1e-10
    saved = jax.jit(jax.grad(jax.checkpoint(block(problem.render, False),
                                            policy=jax.checkpoint_policies.nothing_saveable)))(pk)
    assert rel(saved, expected) <= 1e-10


def test_linearize_then_transpose(problem):
    pk, w = problem.packed, problem.weights
    _, linear = jax.linearize(problem.render, pk)
    t = jnp.asarray(problem.rng.normal(size=pk.shape))
    assert rel(linear(t), jax.jvp(problem.scan, (pk,), (t,))[1]) <= 1e-10
    transpose = jax.linear_transpose(linear, pk)
    _, pull_ref = jax.vjp(problem.scan, pk)
    assert rel(transpose(w)[0], pull_ref(w)[0]) <= 1e-10


def test_second_derivatives_raise_deliberately(problem):
    pk = problem.packed
    loss = problem.loss(problem.render)
    with pytest.raises(NotImplementedError, match="Second and higher derivatives"):
        jax.hessian(loss)(pk)
    with pytest.raises(NotImplementedError, match="Second and higher derivatives"):
        jax.jvp(jax.grad(loss), (pk,), (pk,))
    with pytest.raises(NotImplementedError, match="Second and higher derivatives"):
        jax.grad(lambda x: jnp.sum(jax.jvp(problem.render, (x,), (x,))[1].real))(pk)


def test_first_order_derivative_of_a_linear_function_of_the_tangent_is_allowed(problem):
    """d/dt of (T t) is T again: linear in the tangent operand, so this is not second order."""
    pk, rng = problem.packed, problem.rng
    t0, t1 = (jnp.asarray(rng.normal(size=pk.shape)) for _ in range(2))
    f = lambda t: jax.jvp(problem.render, (pk,), (t,))[1]
    _, dout = jax.jvp(f, (t0,), (t1,))
    assert rel(dout, jax.jvp(problem.scan, (pk,), (t1,))[1]) <= 1e-10


def test_no_tangent_work_when_packed_is_not_differentiated(problem):
    pk = problem.packed
    grad = jax.grad(lambda s: jnp.sum(jnp.abs(problem.render(pk) * s) ** 2))
    assert not equations(grad, 1.3, name="gaussian_tangent")
    assert not equations(grad, 1.3, name="gaussian_adjoint")


def test_check_grads_against_finite_differences(problem):
    """jax.test_util.check_grads (forward and reverse mode) to 1e-8 in float64."""
    rng = np.random.default_rng(1)
    if problem.kind == "dense":
        smooth = jnp.asarray(random_packed(rng, 6, width=0.5, tilt=6.0, curv=8.0))
        p = Problem("dense", problem.config, 6, (5, 5))
    else:
        smooth, _, _ = pixel_bundle(rng, 6, (7, 9), waist_px=1.8, tilt=0.4, curv=0.02, margin=1.0)
        smooth = jnp.asarray(smooth)
        p = Problem("sparse", problem.config, 6, (7, 9), waist_px=1.8)
    check_grads(p.render, (smooth,), order=1, modes=("fwd", "rev"), atol=1e-8, rtol=1e-8,
                eps=1e-6)


def test_shape_validation(problem):
    with pytest.raises(ValueError, match="packed must have shape"):
        evaluate_packed(jnp.zeros((3, 11)), jnp.zeros((4, 2)), config=problem.config)
    with pytest.raises(ValueError, match="coords must have shape"):
        evaluate_packed(jnp.zeros((3, 12)), jnp.zeros((4, 3)), config=problem.config)
    with pytest.raises(ValueError, match="raster shape"):
        evaluate_packed(jnp.zeros((3, 12)), config=problem.config)
    with pytest.raises(ValueError, match="does not match"):
        evaluate_packed(jnp.zeros((3, 12)), jnp.zeros((4, 2)), shape=(3, 3), config=problem.config)


def test_empty_and_single_beam_edge_cases(problem):
    one = problem.packed[:1]
    assert rel(problem.render(one), problem.scan(one)) <= 1e-11
    assert rel(jax.jacfwd(real_parts(problem.render))(one),
               jax.jacfwd(real_parts(problem.scan))(one)) <= 1e-10


def test_empty_bundles_and_grids(problem):
    empty = jnp.zeros((0, 12))
    assert problem.render(empty).shape == (problem.n_pix,)
    assert float(jnp.max(jnp.abs(problem.render(empty)))) == 0.0
    assert jax.jacfwd(real_parts(problem.render))(empty).shape == (2 * problem.n_pix, 0, 12)
    no_pixels = evaluate_packed(problem.packed, jnp.zeros((0, 2)), config=problem.config)
    assert no_pixels.shape == (0,)
    grad = jax.grad(lambda pk: jnp.sum(jnp.abs(
        evaluate_packed(pk, jnp.zeros((0, 2)), config=problem.config))))(problem.packed)
    assert float(jnp.max(jnp.abs(grad))) == 0.0


def test_tile_validation():
    with pytest.raises(ValueError, match="power of two"):
        Tiles(24, 8)
    with pytest.raises(ValueError, match="power of two"):
        Tiles(32, 8, warps=3)
    assert Tiles.sized(16, 4) == Tiles(16, 4, 1) and Tiles.sized(32, 8) == Tiles(32, 8, 4)


@pytest.mark.parametrize("impl", ["reference", "pallas"])
def test_sparse_transform_matrix_on_a_tiny_grid(impl):
    """The sparse kernels in Pallas interpret mode (CPU) or on the GPU with the same problem."""
    if impl == "pallas" and jax.default_backend() != "cpu":
        pytest.skip("interpret mode runs on the CPU platform")
    if impl == "pallas" and plgpu is None:
        pytest.skip("jax.experimental.pallas.triton is missing")
    config = SPARSE_INTERPRET if impl == "pallas" else SPARSE_REF
    p = Problem("sparse", config, 40, (10, 12), waist_px=1.3, margin=2.0)
    pk, rng = p.packed, p.rng
    assert rel(p.render(pk), p.scan(pk)) <= 1e-10
    t = jnp.asarray(rng.normal(size=pk.shape))
    out, tangent = jax.jvp(p.render, (pk,), (t,))
    out_ref, tangent_ref = jax.jvp(p.scan, (pk,), (t,))
    assert rel(out, out_ref) <= 1e-10 and rel(tangent, tangent_ref) <= 1e-10
    assert rel(jax.jit(jax.grad(p.loss(p.render)))(pk), jax.grad(p.loss(p.scan))(pk)) <= 1e-10
    cts = jnp.asarray(rng.normal(size=(2, p.n_pix)) + 1j * rng.normal(size=(2, p.n_pix)))
    many = lambda fn: (lambda cs: jax.vmap(lambda ct: jax.vjp(fn, pk)[1](ct)[0])(cs))
    assert rel(many(p.render)(cts), many(p.scan)(cts)) <= 1e-10
    calls = equations(many(p.render), cts, name="gaussian_adjoint")
    assert len(calls) == 1 and len(calls[0].invars) == 2 + N_PLAN + 1
