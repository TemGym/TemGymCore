"""Differentiable GPU renderer of coherent Gaussian bundles.

``temgym_core.evaluate.evaluate_gaussians`` is the public entry point and dispatcher; this
package holds what it dispatches to.  A bundle is packed into an ``(N, 12)`` float64 array
(:mod:`~temgym_core.render.packing`), classified and binned on the device
(:mod:`~temgym_core.render.plan`) and summed by three JAX primitives with hand-written
derivative rules (:mod:`~temgym_core.render.primitives`): the forward field, its tangent map
``K`` and the transposed adjoint map ``K^T``, with Pallas/Triton kernels on CUDA
(:mod:`~temgym_core.render.kernels`) and pure JAX elsewhere.  All of them apply one pair
predicate (:mod:`~temgym_core.render.predicate`), so the renderer is the cutoff-truncated sum and
its first derivatives are exact.
"""

from .config import RenderConfig, Tiles
from .packing import affine_from_grid, index_coords, pack_gaussian_params, to_index_space
from .primitives import evaluate_packed

__all__ = [
    "RenderConfig",
    "Tiles",
    "affine_from_grid",
    "evaluate_packed",
    "index_coords",
    "pack_gaussian_params",
    "to_index_space",
]
