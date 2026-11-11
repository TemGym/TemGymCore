"""Finding the beam in a camera frame (:func:`find_beam`) and drawing a frame (:func:`show_frame`).

Image analysis only: numpy, nothing that talks to the microscope (the recording script does that
itself with direct PyJEM calls). Shared by the experiments in ``examples/``; not part of TemGym.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Beam:
    """The beam found in a frame by :func:`find_beam`. Positions in px from the frame centre, x + iy."""

    found: bool                   # a region clearly above the background
    centre: complex               # intensity-weighted centre of the beam's region
    rms: float                    # rms radius (px)
    axes: tuple                   # rms along its long and short axes (px), and the long axis' angle (rad)
    signal: float                 # total signal above the background in the region
    area: int                     # pixels in the region
    touches_edge: bool            # the region reaches the frame edge: the beam is (partly) off the camera
    saturated: int                # pixels at the camera's highest value
    others: float                 # signal of the next-brightest region, as a fraction of the beam's
    problems: list = field(default_factory=list)

    @property
    def roundness(self):
        """Short axis / long axis of the beam's region (the same ellipse ``regionprops`` reports): 1 for a round beam."""
        return self.axes[1] / self.axes[0] if self.axes[0] > 0 else 0.0

    @property
    def usable(self):
        """Fit for measuring its centre: found, wholly on the camera, and not one of two similar regions."""
        return self.found and not self.touches_edge and self.others < 0.5


def _box_smooth(image, half):
    """Mean over a (2 half + 1)^2 box, by cumulative sums (edges: the box shrinks)."""
    out = image
    for axis in (0, 1):
        c = np.cumsum(np.insert(out, 0, 0.0, axis=axis), axis=axis)
        n = out.shape[axis]
        lo = np.clip(np.arange(n) - half, 0, n)
        hi = np.clip(np.arange(n) + half + 1, 0, n)
        out = (np.take(c, hi, axis=axis) - np.take(c, lo, axis=axis)) / np.expand_dims(hi - lo, 1 - axis)
    return out


def _grow(mask, steps):
    """The mask grown by ``steps`` pixels (8-connected)."""
    for _ in range(steps):
        m = mask.copy()
        m[1:] |= mask[:-1]; m[:-1] |= mask[1:]; m[:, 1:] |= mask[:, :-1]; m[:, :-1] |= mask[:, 1:]
        m[1:, 1:] |= mask[:-1, :-1]; m[:-1, :-1] |= mask[1:, 1:]; m[1:, :-1] |= mask[:-1, 1:]; m[:-1, 1:] |= mask[1:, :-1]
        mask = m
    return mask


def _label(mask):
    """Connected regions of a mask (8-connected): an integer image, 0 outside, and their number."""
    try:
        from scipy.ndimage import label
        return label(mask, structure=np.ones((3, 3)))
    except ImportError:                                  # numpy only: spread the largest label until stable
        labels = np.where(mask, np.arange(1, mask.size + 1).reshape(mask.shape), 0)
        while True:
            spread = labels.copy()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    spread = np.maximum(spread, np.roll(np.roll(labels, dy, 0), dx, 1) * mask)
            spread *= mask
            if np.array_equal(spread, labels):
                break
            labels = spread
        values, labels = np.unique(labels, return_inverse=True)
        return labels.reshape(mask.shape), len(values) - 1


_image_slot = {"cell": None, "handle": None}      # the image of the running notebook cell, replaced by the next frame


def show_frame(frame, *, beams=(), title=""):
    """Draw one whole frame (square root of the signal above its median, to show faint parts) and mark
    the beam centres in ``beams``, a list of (label, Beam from :func:`find_beam`). In a notebook, a new frame
    replaces the previous one of the same cell. Needs matplotlib (imported only here)."""
    import matplotlib.pyplot as plt

    if frame is None:
        return print(f"{title}: no frame (dry run)")
    frame = np.asarray(frame, dtype=float)
    centre_of_frame = (frame.shape[1] - 1) / 2 + 1j * (frame.shape[0] - 1) / 2
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.imshow(np.sqrt(np.clip(frame - np.median(frame), 0, None)), origin="lower", cmap="gray")
    found = []
    colours = ["#F0E442", "#D55E00", "#56B4E9", "#009E73"]
    for (label, beam), colour in zip(beams, colours):
        if beam.found:
            position = beam.centre + centre_of_frame
            ax.plot(position.real, position.imag, "+", color=colour, ms=14, mew=2, label=label)
            found.append(label)
    if found:
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="normal")
    ax.set(xlabel="x (px)", ylabel="y (px)")
    try:
        from IPython import get_ipython
        from IPython.display import display
        cell = get_ipython().execution_count             # None outside a notebook or IPython
    except (ImportError, AttributeError):
        cell = None
    if cell is None:
        plt.show()
        return
    plt.close(fig)                                       # shown by hand below, not by the inline backend
    if _image_slot["cell"] == cell:
        _image_slot["handle"].update(fig)                # the same cell: replace its image
    else:
        _image_slot.update(cell=cell, handle=display(fig, display_id=True))     # a new cell: a new image


def find_beam(*, frame, background=None, saturation=None):
    """Find the beam in a camera frame by segmentation, and measure it.

    The frame is smoothed a little and thresholded just above the background (the larger of six
    times the smoothed noise and 2 % of the brightest point); bright pixels a few px apart are
    joined, so a caustic or a broken-up beam stays one region. The beam is the region with the most
    signal; it is grown by 3 px to take in its faint edge, and its centre and size are measured
    inside it only, so stray light elsewhere does not pull it. Frames where that is not reliable say
    why (:attr:`Beam.problems`): no beam, the beam reaching the frame edge, a second region nearly
    as bright, saturated pixels.

    Parameters
    ----------
    frame : array_like
        The camera frame.
    background : float or array_like, optional
        The dark level: a number (e.g. the median of a frame with the beam blanked) or a dark
        frame. Without it, the median of the dimmest quarter of the frame.
    saturation : float, optional
        The camera's highest value; by default the largest value of an integer frame's type.

    Returns
    -------
    Beam
    """
    raw = np.asarray(frame)
    f = raw.astype(float)
    if background is None:
        background = float(np.median(np.sort(f, axis=None)[: f.size // 4]))
    f = f - background
    step = (f[:, 1:] - f[:, :-1]).ravel()                    # neighbour differences: the beam is smooth, noise is not
    noise = 1.4826 * float(np.median(np.abs(step - np.median(step)))) / np.sqrt(2) or 1.0
    smooth = _box_smooth(f, 2)                               # 5 x 5 px: noise down 5 times
    threshold = max(6 * noise / 5, 0.02 * smooth.max())
    if saturation is None and np.issubdtype(raw.dtype, np.integer):
        saturation = np.iinfo(raw.dtype).max
    saturated = int(np.sum(raw >= saturation)) if saturation is not None else 0
    nothing = Beam(False, 0j, 0.0, (0.0, 0.0, 0.0), 0.0, 0, False, saturated, 0.0)
    bright = smooth > threshold
    if smooth.max() <= 6 * noise / 5 or not bright.any():
        nothing.problems.append("no beam on the camera")
        return nothing
    labels, n = _label(_grow(bright, 3))                     # fragments within ~6 px join
    signals = np.bincount(labels.ravel(), weights=np.clip(f, 0, None).ravel(), minlength=n + 1)[1:]
    order = np.argsort(signals)[::-1]
    region = _grow(bright & (labels == order[0] + 1), 3)
    others = float(signals[order[1]] / signals[order[0]]) if n > 1 else 0.0
    w = np.where(region, np.clip(f, 0, None), 0.0)
    total = w.sum()
    ny, nx = f.shape
    y, x = np.mgrid[0:ny, 0:nx]
    y, x = y - (ny - 1) / 2, x - (nx - 1) / 2
    cx, cy = (w * x).sum() / total, (w * y).sum() / total
    cov = np.array([[(w * (x - cx) ** 2).sum(), (w * (x - cx) * (y - cy)).sum()],
                    [(w * (x - cx) * (y - cy)).sum(), (w * (y - cy) ** 2).sum()]]) / total
    values, vectors = np.linalg.eigh(cov)
    long_axis = float(np.arctan2(vectors[1, 1], vectors[0, 1]))
    touches = bool(region[0].any() or region[-1].any() or region[:, 0].any() or region[:, -1].any())
    beam = Beam(True, complex(cx, cy), float(np.sqrt(np.trace(cov) / 2)),
                (float(np.sqrt(values[1])), float(np.sqrt(max(values[0], 0.0))), long_axis),
                float(total), int(region.sum()), touches, saturated, others)
    if touches:
        beam.problems.append("the beam reaches the frame edge: part of it is off the camera")
    if others >= 0.05:
        beam.problems.append(f"another bright region, {others:.0%} of the beam's signal")
    if saturated:
        beam.problems.append(f"{saturated} saturated pixels: shorten the exposure or spread the beam")
    return beam
