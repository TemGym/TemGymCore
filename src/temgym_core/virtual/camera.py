"""Camera of the virtual microscope, read out through PyJEM-style detector settings.

The default camera is ideal: a 32-bit signed output that never saturates in
practice, unit gain and no offset, with the same Poisson and read-noise model as
the synthetic calibration images. A realistic JEOL camera (16-bit or 12-bit
output, gain, offset, saturation) is the same class with other numbers; see
``README.md`` in this package for the realism still missing.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraModel:
    """Pixelated detector: sensor geometry, binning, PSF, noise and digitisation.

    Parameters
    ----------
    name : str, optional
        Detector name reported by ``detector.get_attached_detector()``.
    sensor_shape : tuple of int, optional
        Physical pixels (rows, columns).
    sensor_pixel_m : float, optional
        Physical pixel pitch.
    psf_sigma_sensor_pixels : float, optional
        Gaussian point-spread function, in physical pixels.
    binnings : tuple of int, optional
        Allowed binning sizes; each must divide both sensor dimensions.
    gain_adu_per_electron, offset_adu : float, optional
        Digitisation: ``ADU = gain * electrons + offset``.
    read_noise_electrons : float, optional
        Gaussian read noise per output pixel and frame.
    background_electrons : float, optional
        Mean stray electrons per output pixel and frame (Poisson).
    data_bits : {16, 32}, optional
        Signed output word size; values are clipped to its range.
    """

    name: str = "camera"
    sensor_shape: tuple = (2048, 2048)
    sensor_pixel_m: float = 13.75e-6
    psf_sigma_sensor_pixels: float = 8.0
    binnings: tuple = (1, 2, 4, 8)
    gain_adu_per_electron: float = 1.0
    offset_adu: float = 0.0
    read_noise_electrons: float = 2.0
    background_electrons: float = 3.0
    data_bits: int = 32

    def __post_init__(self):
        if self.data_bits not in (16, 32):
            raise ValueError("data_bits must be 16 or 32.")
        if any(s % b for b in self.binnings for s in self.sensor_shape):
            raise ValueError("Every binning must divide both sensor dimensions.")
        if self.gain_adu_per_electron <= 0 or self.read_noise_electrons < 0 or self.background_electrons < 0:
            raise ValueError("Use a positive gain and non-negative noise.")

    @property
    def dtype(self):
        """Little-endian signed output type (``'<i2'`` or ``'<i4'``)."""
        return np.dtype("<i2" if self.data_bits == 16 else "<i4")

    def geometry(self, binning):
        """Binned frame geometry.

        Parameters
        ----------
        binning : int
            One of :attr:`binnings`.

        Returns
        -------
        shape_yx : tuple of int
            Output rows and columns.
        pixel_pitch_m : float
            Output pixel pitch.
        psf_sigma_pixels : float
            PSF in output pixels.
        """
        if binning not in self.binnings:
            raise ValueError(f"Binning {binning} is not one of {self.binnings}.")
        shape = tuple(s // binning for s in self.sensor_shape)
        return shape, self.sensor_pixel_m * binning, self.psf_sigma_sensor_pixels / binning

    def expose(self, expected_electrons, rng, frames=1):
        """Digitise one exposure, summing ``frames`` independent read-outs.

        Parameters
        ----------
        expected_electrons : numpy.ndarray
            Mean signal electrons per output pixel for the whole exposure.
        rng : numpy.random.Generator
            Random source.
        frames : int, optional
            Frames integrated; the signal is shared between them and each frame
            adds its own background and read noise.

        Returns
        -------
        numpy.ndarray
            Integer ADU in :attr:`dtype`, clipped to its range.
        """
        info = np.iinfo(self.dtype)
        total = np.zeros(expected_electrons.shape, dtype=np.int64)
        for _ in range(int(frames)):
            electrons = rng.poisson(expected_electrons / frames + self.background_electrons)
            noisy = electrons + rng.normal(0.0, self.read_noise_electrons, electrons.shape)
            adu = np.rint(self.gain_adu_per_electron * noisy + self.offset_adu)
            total += np.clip(adu, info.min, info.max).astype(np.int64)
        return np.clip(total, info.min, info.max).astype(self.dtype)
