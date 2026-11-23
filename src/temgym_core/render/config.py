"""Configuration of the differentiable Gaussian renderer.

``RenderConfig`` is the one user-facing object: it fixes the truncation ``cutoff``, the
coordinate ``coordinates`` mode, the radius classes of the sparse plan and the Triton launch
shapes.  ``KernelSpec`` (internal) adds the static description of the pixel grid; it is the
hashable ``config`` parameter of the three renderer primitives.
"""

from __future__ import annotations

import dataclasses
import math

COORDINATES = ("index", "metre")
IMPLEMENTATIONS = ("auto", "reference", "pallas")
MAX_INDEX = 2**31 - 2**20


def _power_of_two(name, value):
    if not isinstance(value, int) or value < 1 or value & (value - 1):
        raise ValueError(f"{name} must be a power of two, got {value!r}")


@dataclasses.dataclass(frozen=True)
class Tiles:
    """Triton launch shape of one pair-loop kernel.

    Parameters
    ----------
    pixels : int
        Pixels per program (pixel-major kernels) or pixels per inner-loop tile
        (beam-major kernels).  A power of two.
    beams : int
        Beams per inner-loop tile (pixel-major) or beams per program (beam-major).
        A power of two.
    warps : int
        Warps per program.  A power of two.
    stages : int, optional
        Software pipelining depth handed to Triton (``None``: Triton's default).

    Notes
    -----
    The fp64 pair tile must stay small: keep ``pixels * beams / (32 * warps)`` near 2,
    larger tiles spill registers and can be 30-60x slower.
    """

    pixels: int
    beams: int
    warps: int = 4
    stages: int | None = None

    def __post_init__(self):
        for name in ("pixels", "beams", "warps"):
            _power_of_two(f"Tiles.{name}", getattr(self, name))

    @classmethod
    def sized(cls, pixels, beams):
        """Tile of the given extents with the warp count that keeps two elements per thread."""
        _power_of_two("Tiles.pixels", pixels)
        _power_of_two("Tiles.beams", beams)
        warps = 1
        while pixels * beams > 64 * warps and warps < 8:
            warps *= 2
        return cls(pixels, beams, warps)


@dataclasses.dataclass(frozen=True)
class RenderConfig:
    """Options of ``evaluate_gaussians`` for the differentiable GPU renderers.

    Parameters
    ----------
    cutoff : float
        Truncation level of the sparse renderer, strictly between 0 and 1.  A beam
        contributes to a pixel only where its envelope satisfies
        ``exp(-delta) >= cutoff`` (``delta`` is the envelope exponent, the quadratic form of
        ``Im Q_inv``); the omitted field is bounded by ``cutoff * sum(abs(amplitude))``.
        The same test gates the forward field and both derivative kernels, so gradients are
        exactly those of the truncated sum.  ``method="dense"`` ignores it and sums every pair.
    coordinates : {"index", "metre"}
        ``"index"`` maps the bundle into the detector's pixel-index frame before rendering:
        detector rotation, flip, pixel size and centre are ordinary differentiable inputs.
        The map re-rounds the packed parameters, which matters only for extreme phases
        (about ``1e-16`` relative to the phase, i.e. ``1e-7`` of the peak at ``1e10`` rad).
        ``"metre"`` renders with the metre-space parameters and the metre pixel coordinates,
        accurate to the inputs as the historical dense kernel; differentiating with respect to
        the detector pose then raises ``NotImplementedError``.
    class_edges : tuple of float, optional
        Upper edges (in detector pixels) of the radius classes.  A beam is assigned by the
        half-width of its cutoff footprint; class ``j`` holds ``edges[j-1] < radius <=
        edges[j]``, each with its own bin size, and everything above the last edge (or
        with a non-positive-definite envelope) forms the final dense class.  ``None`` builds
        ``n_classes`` geometric edges ending at half the larger detector side.
    n_classes : int
        Number of sparse radius classes when ``class_edges`` is ``None``.
    class_ratio : float
        Ratio of consecutive automatic edges (greater than one).
    bins_per_axis : int
        Maximum bins per axis of each class's bin grid (2 to 1024).  A class's cell is its
        largest footprint radius, but never below the extent of its beamlet centres divided by
        ``bins_per_axis - 1``: beamlets narrower than that (about 4 px on a 1024 px detector at
        the default) share cells larger than their footprint.
    forward, tangent, adjoint : Tiles
        Launch shapes of the dense pixel-major forward and tangent kernels and the dense
        beam-major adjoint (irregular grids, ``method="dense"`` and the dense class).
    rows : Tiles
        Launch shape of the sparse forward and tangent kernels: ``pixels`` consecutive pixels
        of a row per program, ``beams`` beams per inner tile.
    window_warps : int
        Warps of the sparse adjoint program; it sweeps ``32 * window_warps`` pixels of a
        window row at a time.
    items : int
        Work items of the sparse adjoint beyond one per beam.  Windows are cut into at most
        ``n_beams + items`` pieces of equal work and reduced in a fixed order.
    item_steps : int
        Smallest work item of the sparse adjoint, in row chunks.
    nt_chunk : int
        Tangent directions or cotangents processed per kernel launch.
    impl : {"auto", "reference", "pallas"}
        ``"auto"`` uses the Pallas kernels on CUDA and pure JAX elsewhere, ``"reference"`` the
        pure-JAX pair loops on every platform, ``"pallas"`` the kernels (interpreted on CPU,
        for testing).  With anything but ``"auto"`` the primitives run instead of the
        portable ``scan`` fallback on machines without a GPU.
    pixel_chunk : int, optional
        Pixels per chunk of the pure-JAX implementation (``None``: chosen from the bundle size).

    Notes
    -----
    The defaults come from sweeps of the whole real hologram bundle (160,801 beamlets, 1024 x 1024
    camera) and of synthetic bundles (waists 1 to 15 px, up to 594,000 beamlets) on a GH200,
    always timed against each other in one process:

    * The geometric ladder of eight classes (ratio 1.41, ending at half the detector) is within 1%
      of the hand-picked edges (190, 260, 512 px) on the hologram bundle; the bulk of the
      beamlets, which share one width, gets a class of its own.  A coarser ladder (six classes,
      ratio 1.5) put the bulk with wider beamlets and cost 18% in the forward render.  Fewer
      classes cost at most 5% when the widths are similar.
    * ``bins_per_axis=256`` is 4x faster than 64 for beamlets of waist 1 px and equal otherwise;
      32 is 20% slower for waist 4.5 px.
    * The pixel-major tile ``rows=Tiles(16, 4, 1)`` is the best or within 2% of the best of six to
      nine shapes on every bundle; the dense tiles have a flat optimum too (a badly chosen tile
      spills registers and can be an order of magnitude slower).
    * ``item_steps`` and ``window_warps`` change nothing measurable; ``items=32768`` or more
      gives the best gradient (16,384 costs 5% on the hologram bundle).
    """

    cutoff: float = 1e-12
    coordinates: str = "index"
    class_edges: tuple | None = None
    n_classes: int = 8
    class_ratio: float = math.sqrt(2.0)
    bins_per_axis: int = 256
    forward: Tiles = Tiles(32, 8, 4)
    tangent: Tiles = Tiles(32, 4, 2)
    adjoint: Tiles = Tiles(32, 4, 4)
    rows: Tiles = Tiles(16, 4, 1)
    window_warps: int = 1
    items: int = 32768
    item_steps: int = 256
    nt_chunk: int = 8
    impl: str = "auto"
    pixel_chunk: int | None = None

    def __post_init__(self):
        cutoff = self.cutoff
        if not (isinstance(cutoff, (int, float)) and math.isfinite(cutoff) and 0 < cutoff < 1):
            raise ValueError("cutoff must lie strictly between 0 and 1")
        if self.coordinates not in COORDINATES:
            raise ValueError(f"coordinates must be one of {COORDINATES}, got {self.coordinates!r}")
        if self.impl not in IMPLEMENTATIONS:
            raise ValueError(f"impl must be one of {IMPLEMENTATIONS}, got {self.impl!r}")
        if not (isinstance(self.bins_per_axis, int) and 2 <= self.bins_per_axis <= 1024):
            raise ValueError("bins_per_axis must be an integer between 2 and 1024")
        if self.class_edges is not None:
            edges = tuple(float(e) for e in self.class_edges)
            if any(not math.isfinite(e) or e <= 0 for e in edges) or list(edges) != sorted(set(edges)):
                raise ValueError("class_edges must be positive, finite and strictly increasing")
            object.__setattr__(self, "class_edges", edges)
        if not (isinstance(self.n_classes, int) and 0 <= self.n_classes <= 32):
            raise ValueError("n_classes must be an integer between 0 and 32")
        if not (math.isfinite(self.class_ratio) and self.class_ratio > 1):
            raise ValueError("class_ratio must be greater than one")
        _power_of_two("window_warps", self.window_warps)
        for name in ("items", "item_steps", "nt_chunk"):
            if not (isinstance(getattr(self, name), int) and getattr(self, name) >= 1):
                raise ValueError(f"{name} must be a positive integer")

    @property
    def decay_limit(self):
        """``-ln(cutoff)``: pairs with envelope exponent above it lie outside the support."""
        return -math.log(self.cutoff)

    def with_tiles(self, pixels, beams, warps=4, **changes):
        """Copy with the same tile shape for every dense and sparse pair kernel (sweeps, tests)."""
        tiles = Tiles(pixels, beams, warps)
        return dataclasses.replace(
            self, forward=tiles, tangent=tiles, adjoint=tiles, rows=tiles, **changes)

    def edges_for(self, shape):
        """Class edges (pixels) for a detector of ``shape``: explicit or the geometric ladder."""
        if self.class_edges is not None:
            return self.class_edges
        if self.n_classes == 0:
            return ()
        top = 0.5 * max(shape)
        return tuple(top / self.class_ratio ** (self.n_classes - 1 - j)
                     for j in range(self.n_classes))


@dataclasses.dataclass(frozen=True)
class KernelSpec:
    """Static description of one renderer call (hashable primitive parameter).

    Parameters
    ----------
    render : RenderConfig
        The user options; ``render.cutoff`` is ignored when ``truncate`` is false.
    n_pixels : int
        Number of pixels.
    shape : tuple of int, optional
        ``(H, W)`` when the pixels form a row-major raster (``H * W == n_pixels``).
    source : {"index", "array"}
        Where kernels take pixel positions from: computed from the raster index (the packed
        bundle is in pixel-index space) or read from the ``coords`` array.
    sparse : bool
        Use the plan-based sparse kernels (needs a raster and, for ``"array"``, the affine map).
    truncate : bool
        Apply the pair predicate (``False``: the exact dense sum).
    """

    render: RenderConfig
    n_pixels: int
    shape: tuple | None = None
    source: str = "array"
    sparse: bool = False
    truncate: bool = True

    def __post_init__(self):
        if self.n_pixels >= MAX_INDEX:
            raise ValueError(
                f"{self.n_pixels} pixels exceed the int32 index range of the GPU kernels "
                f"({MAX_INDEX}); render the detector in tiles")
        if self.shape is not None and self.shape[0] * self.shape[1] != self.n_pixels:
            raise ValueError(f"shape {self.shape} does not match {self.n_pixels} pixels")
        if self.sparse and self.shape is None:
            raise ValueError("the sparse renderer needs a raster shape")
        if self.sparse and not self.truncate:
            raise ValueError("the sparse renderer is the truncated sum: it cannot be exact")
        if self.source == "index" and self.shape is None:
            raise ValueError("index-space rendering needs a raster shape")

    @property
    def limit(self):
        """Pair-predicate threshold ``-ln(cutoff)`` or ``None`` for the exact sum."""
        return self.render.decay_limit if self.truncate else None
