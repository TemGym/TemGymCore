"""Vectorised replacements for the two former numba helpers match the original loops."""
import numpy as np
import pytest

from temgym_core.components import Detector
from temgym_core.source import _ring_angles, concentric_rings


def historic_multi_cumsum(values, partitions, start):
    """The original loop, unchanged, without the numba decorator."""
    values = values.copy()
    part_idx = 0
    current_part_len = partitions[part_idx]
    part_count = 0
    values[0] = start
    for v_idx in range(1, values.size):
        if current_part_len == part_count:
            part_count = 0
            part_idx += 1
            current_part_len = partitions[part_idx]
            values[v_idx] = start
        else:
            values[v_idx] += values[v_idx - 1]
            part_count += 1
    return values


@pytest.mark.parametrize("approx", [1, 2, 3, 5, 10, 17, 33, 60, 100, 257, 1000, 4096, 20000])
def test_ring_angles_match_the_original_accumulation(approx):
    num_rings = max(1, int(np.floor((-1 + np.sqrt(1 + 4 * approx / np.pi)) / 2)))
    per_kth = np.round(2 * np.pi * np.arange(1, num_rings + 1)).astype(int)
    points = np.round(per_kth * (approx / per_kth.sum())).astype(int)
    div = 2 * np.pi / points
    repeated = np.repeat(div, points)
    expected = historic_multi_cumsum(repeated, points, 0.0)
    np.testing.assert_array_equal(_ring_angles(div, points), expected)


def test_concentric_rings_is_deterministic_and_inside_the_disk():
    pts = concentric_rings(500, 2.5)
    assert pts.shape == (500, 2)
    assert np.hypot(*pts.T).max() <= 2.5 + 1e-12
    np.testing.assert_array_equal(pts, concentric_rings(500, 2.5))


def test_into_image_counts_in_bounds_pixels_and_ignores_the_rest():
    detector = Detector(z=0.0, pixel_size=(1e-3, 1e-3), shape=(5, 7))
    rng = np.random.default_rng(3)
    yy = rng.integers(-3, 9, size=400)
    xx = rng.integers(-3, 11, size=400)
    image = detector.into_image((yy, xx))
    expected = np.zeros((5, 7), dtype=int)
    for y, x in zip(yy, xx):
        if 0 <= y < 5 and 0 <= x < 7:
            expected[y, x] += 1
    np.testing.assert_array_equal(image, expected)
    # Accumulating into an existing image adds to it.
    again = detector.into_image((yy, xx), acc=image.copy())
    np.testing.assert_array_equal(again, 2 * expected)
