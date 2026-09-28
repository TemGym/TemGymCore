"""Smiley recipe copied unchanged from ../gaussians/microscope_sample_objects.ipynb."""
import numpy as np

def smiley(size):
    '''
    Smiley face test object from https://doi.org/10.1093/micmic/ozad021
    '''
    obj = np.ones((size, size), dtype=np.complex64)
    y, x = np.ogrid[-size // 2:size // 2, -size // 2:size // 2]

    outline = (((y * 1.2) ** 2 + x**2) > (110 / 256 * size) ** 2) & (
        ((y * 1.2) ** 2 + x**2) < (120 / 256 * size) ** 2
    )
    obj[outline] = 0.0

    left_eye = ((y + 40 / 256 * size) ** 2 + (x + 40 / 256 * size) ** 2) < (20 / 256 * size) ** 2
    obj[left_eye] = 0
    right_eye = (np.abs(y + 40 / 256 * size) < 15 / 256 * size) & (
        np.abs(x - 40 / 256 * size) < 30 / 256 * size
    )
    obj[right_eye] = 0

    nose = (y + 20 / 256 * size + x > 0) & (x < 0) & (y < 10 / 256 * size)
    obj[nose] = (0.05j * x + 0.05j * y)[nose]

    mouth = (((y * 1) ** 2 + x**2) > (50 / 256 * size) ** 2) & (
        ((y * 1) ** 2 + x**2) < (70 / 256 * size) ** 2
    ) & (y > 20 / 256 * size)
    obj[mouth] = 0

    tongue = (((y - 50 / 256 * size) ** 2 + (x - 50 / 256 * size) ** 2) < (20 / 256 * size) ** 2) & (
        (y**2 + x**2) > (70 / 256 * size) ** 2
    )
    obj[tongue] = 0

    signature_wave = np.exp(1j * (3 * y + 7 * x) * 2 * np.pi / size)
    obj += 0.3 * signature_wave - 0.3
    return obj
