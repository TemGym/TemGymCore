"""Three JAX primitives with hand-written derivative rules for the packed Gaussian renderer.

``gaussian_forward_p``  ``(packed, coords, *plan) -> E``                 nonlinear
``gaussian_tangent_p``  ``(packed, coords, *plan, w) -> Edot``           linear in ``w``  (``K``)
``gaussian_adjoint_p``  ``(packed, coords, *plan, ct) -> M``             linear in ``ct`` (``K^T``)

``w`` are the polynomial coefficients of the tangent field, ``(nt, N, 12)``, and ``M`` the
cotangent-layout moments, ``(nt, N, 12)`` (see :mod:`temgym_core.render.coefficients`).  The
elementwise maps between packed tangents and ``w`` (``C`` and its transpose) are ordinary JAX,
so JAX transposes them itself; the two heavy pair loops are the only things with kernels.  Each
of ``K`` and ``K^T`` is the transpose of the other, so ``jvp``, ``vjp``, ``jacfwd``, ``jacrev``,
``vmap`` and ``checkpoint`` compose through JAX's own machinery.

``plan`` is a (possibly empty) tuple of arrays describing the spatial structure of the sparse
kernels (:class:`temgym_core.render.plan.RenderPlan`).  It is built once per evaluation from
``packed`` and threaded through all three primitives as non-differentiable residuals, so
forward, tangent and adjoint traverse the same structure and apply the same pair predicate.

Batching folds a batched *linear* operand into the tangent count ``nt`` (one fused kernel
launch); anything else (batched ``packed``, ``coords`` or ``plan``) runs the kernel sequentially
with ``lax.map``.  Second derivatives raise ``NotImplementedError`` by design.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.core import ShapedArray
from jax.custom_derivatives import SymbolicZero, zero_from_primal
from jax.extend import core as jex_core
from jax.interpreters import ad, batching, mlir, xla

from temgym_core.config import require_x64

from .coefficients import moments_to_packed_cotangent, tangent_coefficients
from .config import KernelSpec, RenderConfig
from .kernels import reference as ref
from .packing import index_coords, sanitise_packed
from .plan import build_plan

MAX_BEAMS = 2**27  # int32 row indices of the kernels stay below 2**31 (nt_chunk * N * 12 words)

SECOND_ORDER_MESSAGE = (
    "Second and higher derivatives of the Gaussian renderer are not implemented: "
    "the hand-written tangent/adjoint kernels are first-order only. Differentiate "
    "once (grad/jvp/jacfwd/jacrev), or use method='scan' for Hessians."
)
COORDINATE_MESSAGE = (
    "Derivatives with respect to the pixel coordinates are not implemented in metre-space "
    "rendering. Render with coordinates='index', which maps the bundle into the detector's "
    "pixel frame so that detector pose derivatives are ordinary autodiff, or differentiate "
    "the beamlets only."
)

forward_p = jex_core.Primitive("gaussian_forward")
tangent_p = jex_core.Primitive("gaussian_tangent")
adjoint_p = jex_core.Primitive("gaussian_adjoint")
for _p in (forward_p, tangent_p, adjoint_p):
    _p.def_impl(functools.partial(xla.apply_primitive, _p))


def _check_operands(packed, coords):
    if packed.ndim != 2 or packed.shape[1] != 12:
        raise ValueError(f"packed must have shape (N, 12), got {packed.shape}")
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape (P, 2), got {coords.shape}")


def _forward_abstract(packed, coords, *plan, spec):
    _check_operands(packed, coords)
    return ShapedArray((spec.n_pixels,), jnp.complex128)


def _tangent_abstract(packed, coords, *rest, spec):
    _check_operands(packed, coords)
    w = rest[-1]
    if w.ndim != 3 or w.shape[1:] != (packed.shape[0], 12):
        raise ValueError(f"w must have shape (nt, N, 12), got {w.shape}")
    return ShapedArray((w.shape[0], spec.n_pixels), jnp.complex128)


def _adjoint_abstract(packed, coords, *rest, spec):
    _check_operands(packed, coords)
    ct = rest[-1]
    if ct.ndim != 2 or ct.shape[1] != spec.n_pixels:
        raise ValueError(f"ct must have shape (nt, P), got {ct.shape}")
    return ShapedArray((ct.shape[0], packed.shape[0], 12), jnp.float64)


forward_p.def_abstract_eval(_forward_abstract)
tangent_p.def_abstract_eval(_tangent_abstract)
adjoint_p.def_abstract_eval(_adjoint_abstract)


# ---------------------------------------------------------------------------
# Derivative rules
# ---------------------------------------------------------------------------
def _is_zero(t):
    return type(t) is ad.Zero


def _forward_jvp(primals, tangents, *, spec):
    # First-order derivatives never reach here: the custom_jvp rule of ``evaluate_packed``
    # replaces the primal's derivative. Only higher-order differentiation traces the rule's
    # own forward call.
    raise NotImplementedError(SECOND_ORDER_MESSAGE)


def _linear_jvp(prim, primals, tangents, *, spec):
    """JVP of a primitive linear in its last operand and nonlinear in all the others."""
    *fixed, lin = primals
    *t_fixed, t_lin = tangents
    if not all(_is_zero(t) for t in t_fixed):
        # integer plan operands carry float0 tangents, which are Zero here as well
        raise NotImplementedError(SECOND_ORDER_MESSAGE)
    out = prim.bind(*fixed, lin, spec=spec)
    if _is_zero(t_lin):
        return out, ad.Zero.from_primal_value(out)
    return out, prim.bind(*fixed, t_lin, spec=spec)


def _transpose_in_last_operand(other, ct, *operands, spec):
    *fixed, lin = operands
    if any(ad.is_undefined_primal(x) for x in fixed):
        raise NotImplementedError(SECOND_ORDER_MESSAGE)
    assert ad.is_undefined_primal(lin)
    nones = [None] * len(fixed)
    if _is_zero(ct):
        return nones + [ad.Zero(lin.aval)]
    return nones + [other.bind(*fixed, ct, spec=spec)]


def _tangent_transpose(ct, *operands, spec):
    return _transpose_in_last_operand(adjoint_p, ct, *operands, spec=spec)


def _adjoint_transpose(ct, *operands, spec):
    return _transpose_in_last_operand(tangent_p, ct, *operands, spec=spec)


ad.primitive_jvps[forward_p] = _forward_jvp
ad.primitive_jvps[tangent_p] = functools.partial(_linear_jvp, tangent_p)
ad.primitive_jvps[adjoint_p] = functools.partial(_linear_jvp, adjoint_p)
ad.primitive_transposes[tangent_p] = _tangent_transpose
ad.primitive_transposes[adjoint_p] = _adjoint_transpose


# ---------------------------------------------------------------------------
# Batching rules
# ---------------------------------------------------------------------------
def _sequential(prim, args, dims, spec):
    """Run ``prim`` once per batch element with ``lax.map`` (one compiled kernel)."""
    moved = [a if d is None else jnp.moveaxis(a, d, 0) for a, d in zip(args, dims)]
    mapped = tuple(a for a, d in zip(moved, dims) if d is not None)
    fixed = tuple(a for a, d in zip(moved, dims) if d is None)

    def one(xs):
        it_mapped, it_fixed = iter(xs), iter(fixed)
        full = [next(it_mapped) if d is not None else next(it_fixed) for d in dims]
        return prim.bind(*full, spec=spec)

    return lax.map(one, mapped)


def _forward_batch(args, dims, *, spec):
    return _sequential(forward_p, args, dims, spec), 0


def _linear_batch(prim, args, dims, *, spec):
    *fixed, lin = args
    *fixed_dims, d_lin = dims
    if all(d is None for d in fixed_dims):
        # Only the linear operand is batched: fold the batch into the tangent count.
        lin = jnp.moveaxis(lin, d_lin, 0)
        batch, nt = lin.shape[:2]
        out = prim.bind(*fixed, lin.reshape((batch * nt,) + lin.shape[2:]), spec=spec)
        return out.reshape((batch, nt) + out.shape[1:]), 0
    return _sequential(prim, args, dims, spec), 0


batching.primitive_batchers[forward_p] = _forward_batch
batching.primitive_batchers[tangent_p] = functools.partial(_linear_batch, tangent_p)
batching.primitive_batchers[adjoint_p] = functools.partial(_linear_batch, adjoint_p)


# ---------------------------------------------------------------------------
# Lowering: pure JAX on CPU, Pallas on CUDA
# ---------------------------------------------------------------------------
def _use_pallas(spec, on_cuda):
    impl = spec.render.impl
    return impl == "pallas" or (impl == "auto" and on_cuda)


def _positions(spec, coords):
    return index_coords(spec.shape) if spec.source == "index" else coords


# The pure-JAX implementations ignore the plan: the sparse kernels sum exactly the pairs
# that satisfy the shared predicate, which is what the dense reference computes with the same
# limit.
def _ref_forward(packed, coords, *plan, spec):
    return ref.field(packed, _positions(spec, coords), limit=spec.limit,
                     chunk=spec.render.pixel_chunk)


def _ref_tangent(packed, coords, *rest, spec):
    return ref.polynomial_sum(packed, _positions(spec, coords), rest[-1], limit=spec.limit,
                              chunk=spec.render.pixel_chunk)


def _ref_adjoint(packed, coords, *rest, spec):
    return ref.polynomial_moments(packed, _positions(spec, coords), rest[-1], limit=spec.limit,
                                  chunk=spec.render.pixel_chunk)


def _make_lowerings(reference_fn, kind):
    def impl(on_cuda):
        def run(packed, coords, *rest, spec):
            if _use_pallas(spec, on_cuda):
                from .kernels import dense, sparse

                module = sparse if spec.sparse else dense
                return getattr(module, kind)(packed, coords, *rest, spec=spec,
                                             interpret=not on_cuda)
            return reference_fn(packed, coords, *rest, spec=spec)
        return run
    return impl(False), impl(True)


for _prim, _ref, _kind in (
    (forward_p, _ref_forward, "forward"),
    (tangent_p, _ref_tangent, "tangent"),
    (adjoint_p, _ref_adjoint, "adjoint"),
):
    _cpu, _cuda = _make_lowerings(_ref, _kind)
    mlir.register_lowering(_prim, mlir.lower_fun(_cpu, multiple_results=False), platform="cpu")
    mlir.register_lowering(_prim, mlir.lower_fun(_cuda, multiple_results=False), platform="cuda")


# ---------------------------------------------------------------------------
# Differentiable entry point on the packed array
# ---------------------------------------------------------------------------
def _plan(packed, affine, spec):
    """Spatial plan of the sparse path (empty tuple for the dense path)."""
    return tuple(build_plan(packed, affine, spec)) if spec.sparse else ()


@functools.lru_cache(maxsize=None)
def _sequential_forward(spec):
    # Under vmap the primal is evaluated once per batch element, plan included, so the
    # kernels never see a batched bundle and no batched control flow is turned into a select.
    @jax.custom_batching.sequential_vmap
    def forward(packed, coords, affine):
        return forward_p.bind(packed, coords, *_plan(packed, affine, spec), spec=spec)

    return forward


@functools.partial(jax.custom_jvp, nondiff_argnums=(3,))
def _render(packed, coords, affine, spec):
    return _sequential_forward(spec)(packed, coords, affine)


def _render_jvp(spec, primals, tangents):
    packed, coords, affine = primals
    t_packed, t_coords, t_affine = tangents
    if not (isinstance(t_coords, SymbolicZero) and isinstance(t_affine, SymbolicZero)):
        raise NotImplementedError(COORDINATE_MESSAGE)
    plan = _plan(packed, affine, spec)  # built once, shared by the primal and the tangent
    out = forward_p.bind(packed, coords, *plan, spec=spec)
    if isinstance(t_packed, SymbolicZero):
        return out, zero_from_primal(out, symbolic_zeros=True)
    w = tangent_coefficients(packed, t_packed[None])
    return out, tangent_p.bind(packed, coords, *plan, w, spec=spec)[0]


_render.defjvp(_render_jvp, symbolic_zeros=True)


@functools.partial(jax.jit, static_argnums=3)
def _render_jit(packed, coords, affine, spec):
    """Sanitise and render under one ``jit`` so eager calls compile once per shape."""
    packed, _ = sanitise_packed(packed)
    return _render(packed, coords, affine, spec)


def make_spec(n_pixels, config, *, shape=None, coords=None, affine=None, exact=False):
    """Choose the kernel family for a call and describe it statically.

    Parameters
    ----------
    n_pixels : int
        Number of pixels.
    config : RenderConfig
        User options.
    shape : tuple of int, optional
        ``(H, W)`` if the pixels form a row-major raster.
    coords : array, optional
        Metre-space pixel positions; ``None`` selects the pixel-index frame.
    affine : array, optional
        Pixel-to-metre map, needed by the sparse renderer for metre-space bundles.
    exact : bool
        Sum every pair (no truncation, dense kernels).

    Returns
    -------
    KernelSpec
        The hashable static parameter of the primitives.
    """
    index = coords is None
    sparse = not exact and shape is not None and (index or affine is not None)
    return KernelSpec(render=config, n_pixels=n_pixels, shape=shape,
                      source="index" if index else "array", sparse=sparse, truncate=not exact)


def evaluate_packed(packed, coords=None, *, shape=None, affine=None, config=None, exact=False):
    """Coherent field ``sum_b A_b g_b`` of a packed ``(N, 12)`` bundle on a set of pixels.

    First-order differentiable with respect to ``packed`` (``jvp``, ``grad``, ``jacfwd``,
    ``jacrev``, ``vmap``, ``jit``, ``checkpoint``), not with respect to ``coords`` or ``affine``.
    Rows with any non-finite entry contribute zero and receive a zero gradient.

    Parameters
    ----------
    packed : array, shape (N, 12)
        Packed bundle (see :mod:`temgym_core.render.packing`).
    coords : array, shape (P, 2), optional
        Metre-space pixel positions ``(x, y)``.  ``None`` renders in the pixel-index frame:
        ``packed`` must already be in pixel-index space (``to_index_space``) and ``shape``
        given.
    shape : tuple of int, optional
        ``(H, W)`` when the pixels are the row-major raster of an ``H x W`` detector.  Enables
        the sparse renderer.
    affine : array, shape (6,), optional
        ``[J00, J01, J10, J11, ox, oy]`` with ``(x, y) = J (ix, iy) + o``; lets the sparse
        renderer plan metre-space bundles.  Without it, metre-space pixels use the dense
        kernels.
    config : RenderConfig, optional
        Options; the default is ``RenderConfig()``.
    exact : bool
        Sum every pair with the dense kernels instead of the cutoff-truncated sum.

    Returns
    -------
    array, shape (P,)
        Complex128 field.
    """
    require_x64("evaluate_packed")
    config = RenderConfig() if config is None else config
    packed = jnp.asarray(packed, jnp.float64)
    if packed.ndim != 2 or packed.shape[1] != 12:
        raise ValueError(f"packed must have shape (N, 12), got {packed.shape}")
    if packed.shape[0] >= MAX_BEAMS:
        raise ValueError(f"{packed.shape[0]} beamlets exceed the int32 index range of the GPU "
                         f"kernels ({MAX_BEAMS}); render the bundle in parts")
    if coords is None:
        if shape is None:
            raise ValueError("rendering in the pixel-index frame needs the raster shape")
        n_pixels = shape[0] * shape[1]
        coords_array = jnp.zeros((0, 2), jnp.float64)
    else:
        coords_array = jnp.asarray(coords, jnp.float64)
        if coords_array.ndim != 2 or coords_array.shape[1] != 2:
            raise ValueError(f"coords must have shape (P, 2), got {coords_array.shape}")
        n_pixels = coords_array.shape[0]
    shape = None if shape is None else (int(shape[0]), int(shape[1]))
    if shape is not None and shape[0] * shape[1] != n_pixels:
        raise ValueError(f"shape {shape} does not match {n_pixels} pixels")
    spec = make_spec(n_pixels, config, shape=shape, coords=coords, affine=affine, exact=exact)
    if packed.shape[0] == 0 or n_pixels == 0:
        return jnp.zeros((n_pixels,), jnp.complex128)
    affine_array = (jnp.zeros((6,), jnp.float64) if affine is None
                    else jnp.asarray(affine, jnp.float64).reshape(6))
    return _render_jit(packed, coords_array, affine_array, spec)


def tangent_map(packed, coords=None, tdot=None, *, shape=None, affine=None, config=None,
                exact=False):
    """Forward tangent map ``T tdot`` for ``tdot`` of shape ``(nt, N, 12)`` (one fused call).

    A test and benchmark entry point below the ``custom_jvp``: the packed bundle must already
    be sanitised.  Arguments as :func:`evaluate_packed`.

    Returns
    -------
    array, shape (nt, P)
        Complex128 tangent fields.
    """
    config, packed, coords_array, affine_array, spec = _prepare(
        packed, coords, shape, affine, config, exact)
    w = tangent_coefficients(packed, tdot)
    return tangent_p.bind(packed, coords_array, *_plan(packed, affine_array, spec), w, spec=spec)


def adjoint_map(packed, coords=None, ct=None, *, shape=None, affine=None, config=None,
                exact=False):
    """Adjoint map ``T^T ct`` for ``ct`` of shape ``(nt, P)``, result ``(nt, N, 12)``.

    A test and benchmark entry point below the ``custom_jvp``; arguments as
    :func:`evaluate_packed`.

    Returns
    -------
    array, shape (nt, N, 12)
        Gradients with respect to the packed columns.
    """
    config, packed, coords_array, affine_array, spec = _prepare(
        packed, coords, shape, affine, config, exact)
    moments = adjoint_p.bind(packed, coords_array, *_plan(packed, affine_array, spec), ct,
                             spec=spec)
    return moments_to_packed_cotangent(packed, moments)


def _prepare(packed, coords, shape, affine, config, exact):
    config = RenderConfig() if config is None else config
    packed = jnp.asarray(packed, jnp.float64)
    if coords is None:
        n_pixels, coords_array = shape[0] * shape[1], jnp.zeros((0, 2), jnp.float64)
    else:
        coords_array = jnp.asarray(coords, jnp.float64)
        n_pixels = coords_array.shape[0]
    spec = make_spec(n_pixels, config, shape=shape, coords=coords, affine=affine, exact=exact)
    affine_array = (jnp.zeros((6,), jnp.float64) if affine is None
                    else jnp.asarray(affine, jnp.float64).reshape(6))
    return config, packed, coords_array, affine_array, spec
