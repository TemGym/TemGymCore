"""The frozen 150,000x biprism column: the sampled wave against the coherent Gaussian beamlet sum.

Both answers run through the same components. The wave is the reference: its linear spans are exact
Collins transforms and the biprism is an exact mask on a fine lattice. The beamlet sum expands the
biprism's action to second order around each beamlet's centre and uses a finite beamlet basis, so it
differs from the wave by a few parts in 1e3, and the difference shrinks with the beamlet waist.

The reduced problem keeps the column and the camera pixel of the reference scenario and shrinks the
arrays: a 120 nm beam (the two half beams must overlap on the camera, which needs 2 x 6.5 mm of beam
there), 96 x 96 camera pixels, and a biprism plane sampled at 8 nm on 1024 x 1024 samples.
"""

import types

import jax.numpy as jnp
import numpy as np
import pytest

import reference_scenario as scenario
from temgym_core.components import Detector
from temgym_core.constants import energy2wavelength
from temgym_core.evaluate import evaluate_gaussians
from temgym_core.run import run_to_end
from temgym_core.source import rectangular_input_wave
from temgym_core.wave import Lattice, Wavefront, column_transform, input_lattice, propagate
from wave_validation import rel_err, rel_err_no_phase

pytestmark = pytest.mark.slow

CAMERA_PIXELS = 96
MID_SAMPLES, MID_PITCH = 1024, 8e-9
WAIST, WIDTH = 15e-9, 120e-9
TOL = 1e-10


@pytest.fixture(scope="module")
def column_and_planes():
    solution = scenario.focused_microscope()
    microscope, column, planes = scenario.biprism_column(solution)
    camera = Detector(z=float(microscope.components["detector"].z), pixel_size=(14e-6, 14e-6),
                      shape=(CAMERA_PIXELS, CAMERA_PIXELS))
    return microscope.voltage, column, planes, camera


def run_wave(voltage, column, planes, camera, mid_pitch, mid_samples, tol):
    """Objective, biprism plane sampled at ``mid_pitch``, projector and camera: the wave and the beamlet sum."""
    wavelength = float(energy2wavelength(voltage))
    z_sample, z_biprism = planes["z_sample"], planes["z_biprism"]
    to_biprism, from_biprism = column[:2], column[2:] + (camera,)      # objective | biprism, projector, camera
    grid = Lattice.from_grid(camera)
    # Sample the biprism plane at mid_pitch: for a matrix Fourier transform that is the cross-term
    # parameter gamma = p_mid * p_camera / (lambda |B|) of the span behind the biprism.
    B = column_transform(column[3:] + (camera,), z_biprism, voltage).matrix[:2, 2:4]
    gamma = mid_pitch * 14e-6 / (wavelength * np.sqrt(abs(np.linalg.det(B))))
    mid = input_lattice(from_biprism, grid, source_z=z_biprism, voltage=voltage,
                        in_shape=(mid_samples, mid_samples), gamma=(gamma, gamma))
    source = input_lattice(to_biprism, mid, source_z=z_sample, voltage=voltage, method="near_image")
    beam = rectangular_input_wave(aperture_width=WIDTH, aperture_height=WIDTH, waist=WAIST, voltage=voltage,
                                  overlap_factor=2.0, z0=z_sample)
    samples = types.SimpleNamespace(coords=source.positions().reshape(-1, 2), shape=source.shape)
    start = Wavefront(evaluate_gaussians(beam, samples, method="scan"), source, z_sample, voltage)
    at_biprism = propagate(start, to_biprism, mid, method="near_image", tol=tol)
    wave = propagate(at_biprism, from_biprism, grid, tol=tol)
    beamlets = evaluate_gaussians(run_to_end(beam, column + (camera,)), camera, method="scan")
    return dict(wave=wave, beamlets=beamlets, source=source, mid=mid, planes=planes, at_biprism=at_biprism)


@pytest.fixture(scope="module")
def hologram(column_and_planes):
    return run_wave(*column_and_planes, MID_PITCH, MID_SAMPLES, TOL)


def test_wave_and_beamlet_sum_agree_on_the_biprism_hologram(hologram):
    wave, beamlets = hologram["wave"].data, hologram["beamlets"]
    assert wave.shape == beamlets.shape == (CAMERA_PIXELS, CAMERA_PIXELS)
    assert abs(float(jnp.abs(wave).max() / jnp.abs(beamlets).max()) - 1) < 1e-3
    assert rel_err(jnp.abs(wave) ** 2, jnp.abs(beamlets) ** 2) < 1e-2          # measured 3.8e-3
    assert rel_err_no_phase(wave, beamlets) < 1e-2                             # measured 2.6e-3


def test_the_hologram_fringes_have_the_frozen_period(hologram):
    """Two half beams cross on the camera: fringes of the period the saved baseline froze for the full beam
    (146.3 um from a 1024-pixel line; a 96-pixel row resolves it to one bin, 149.3 um)."""
    baseline = 0.00014628571428571428      # tests/reference/baseline.json, fringes.period_m
    for field in (hologram["wave"].data, hologram["beamlets"]):
        row = np.abs(np.asarray(field)[CAMERA_PIXELS // 2]) ** 2
        spectrum = np.abs(np.fft.rfft(row - row.mean()))
        peak = int(np.argmax(spectrum[2:]) + 2)
        assert abs(CAMERA_PIXELS * 14e-6 / peak / baseline - 1) < 0.03
        assert 2 * spectrum[peak] / (len(row) * row.mean()) > 0.9              # high-contrast fringes


def test_the_chain_of_lattices_ends_on_the_camera_plane(hologram):
    """The field at the biprism plane lives on the lattice chosen for it, at the biprism plane."""
    assert hologram["at_biprism"].z == pytest.approx(hologram["planes"]["z_biprism"])
    assert hologram["at_biprism"].lattice.shape == (MID_SAMPLES, MID_SAMPLES)
    assert float(jnp.abs(hologram["at_biprism"].data).max()) > 0
    assert hologram["source"].shape == (MID_SAMPLES, MID_SAMPLES)


def test_the_wave_is_converged_in_its_sampling(hologram, column_and_planes):
    """The wave is the reference because it converges: sampling the biprism plane at 6 nm on 1536 x 1536
    samples with a tighter tolerance changes it by 2e-7, three orders below its distance to the beamlet
    sum. (Halving the beamlets again does not close that distance: it is set by the beamlets' second-order
    expansion of a 25 nm biprism edge, not by their number.)"""
    finer = run_wave(*column_and_planes, 6e-9, 1536, 1e-11)["wave"].data
    wave = hologram["wave"].data
    assert rel_err(wave, finer) < 1e-6                                          # measured 1.9e-7
    assert rel_err(jnp.abs(wave) ** 2, jnp.abs(finer) ** 2) < 1e-6
    assert rel_err(wave, hologram["beamlets"]) > 100 * rel_err(wave, finer)
