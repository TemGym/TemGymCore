"""The hologram notebook's beamlet bundle at the camera: the workload the GPU renderer is sized for.

Notebook illumination (3 nm waist, 1.5 nm lattice, 600 nm aperture = 401 x 401 = 160,801 beamlets)
through the frozen 150,000x biprism column onto a 1024 x 1024 camera with 14 um pixels. At the
camera about 90% of the beamlets that reach the sensor share one width (waist 32.1 px, 1e-12
footprint radius 169 px); about 7% are intermediate (footprints of 190-512 px) and 401 (3%) are
extremely broad (waist up to 85,000 px): they are the lattice line that sits exactly on the biprism
wire's axis. Only about 12,500 of the 160,801 beamlets reach the sensor.

Run ``python benchmarks/hologram_bundle.py`` to print these statistics.
"""
import sys
from pathlib import Path

import jax
import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tests"))

jax.config.update("jax_enable_x64", True)

PIXEL = 14e-6
SHAPE = (1024, 1024)


def hologram_bundle(aperture=600e-9, waist=3e-9, lattice=1.5e-9):
    """Return ``(beam, detector)``: the propagated bundle at the camera and the camera grid.

    Parameters
    ----------
    aperture, waist, lattice : float
        Illuminated width, beamlet waist and lattice pitch at the specimen (metres).

    Returns
    -------
    tuple
        The ``GaussianBeam`` bundle at the detector plane and the ``Detector``.
    """
    import reference_scenario as G
    from temgym_core.components import Detector, Plane
    from temgym_core.run import run_to_end
    from temgym_core.source import rectangular_input_wave

    microscope, column, planes = G.biprism_column(G.focused_microscope())
    detector_z = float(microscope.components["detector"].z)
    beam = rectangular_input_wave(
        aperture_width=aperture, aperture_height=aperture, waist=waist,
        voltage=microscope.voltage, overlap_factor=waist / lattice, z0=planes["z_sample"],
    )
    at_camera = jax.jit(run_to_end)(beam, column + (Plane(z=detector_z),))
    jax.block_until_ready(at_camera.x)
    return at_camera, Detector(z=detector_z, pixel_size=(PIXEL, PIXEL), shape=SHAPE)


def footprint_half_width(beam, cutoff=1e-12):
    """Half-width (px) of each beamlet's cutoff footprint along its broad axis."""
    k = np.asarray(beam.k)
    smallest = np.linalg.eigvalsh(np.asarray(beam.Q_inv).imag)[:, 0]
    return np.sqrt(-2 * np.log(cutoff) / (k * smallest)) / PIXEL


if __name__ == "__main__":
    beam, detector = hologram_bundle()
    foot = footprint_half_width(beam)
    x, y = np.asarray(beam.x) / PIXEL, np.asarray(beam.y) / PIXEL
    half = np.array(SHAPE) / 2
    touching = (np.abs(x) < half[1] + foot) & (np.abs(y) < half[0] + foot)
    print(f"{beam.x.shape[0]:,} beamlets, {int(touching.sum()):,} touch the sensor")
    for lo, hi in ((0, 190), (190, 260), (260, 512), (512, np.inf)):
        n = int(((foot > lo) & (foot <= hi) & touching).sum())
        print(f"  footprint half-width ({lo}, {hi}] px: {n:,} touching beamlets")
