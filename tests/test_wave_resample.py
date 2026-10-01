"""Band-limited resampling between lattices, and the imported-array workflow around it."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from temgym_core.components import Detector, ElectromagneticLens, Plane
from temgym_core.constants import (
    compute_Rc_from_voltage,
    effective_accelerating_potential,
    energy2wavelength,
)
from temgym_core.gaussian import make_gaussian
from temgym_core.run import run_to_end
from temgym_core.wave import (
    Lattice,
    SamplingError,
    Wavefront,
    gaussian_field,
    input_lattice,
    propagate,
    resample_field,
)
from temgym_core.wave.reference import chief_from_beam
from wave_validation import gaussian_Q, rel_err

VOLTAGE = 200e3
LAM = float(energy2wavelength(VOLTAGE))
Q = np.array([[0.0 + 0.9j, 0.1j], [0.1j, 0.0 + 1.2j]])


def field(lattice, centre=(0.2, -0.1), slopes=(0.01, 0.02)):
    """A smooth, compact, band-limited test field in units where the wavelength is 1."""
    return gaussian_field(lattice.positions(), 1.0, np.array(centre), np.array(slopes), Q, 1.0)


SOURCE = Lattice.regular((96, 80), (0.12, 0.1), rotation=20.0, centre=(0.05, 0.02))
TARGETS = {
    "rotated and finer": Lattice.regular((88, 100), (0.09, 0.09), rotation=-7.0),
    "rotated by 50 degrees at a similar pitch": Lattice.regular((96, 96), (0.11, 0.11), rotation=50.0),
    "sheared": Lattice(pitch=jnp.array([[0.1, 0.04], [0.0, 0.11]]), centre=jnp.array([0.03, 0.0]), shape=(90, 90)),
    "mirrored": Lattice(pitch=jnp.array([[-0.1, 0.0], [0.0, 0.1]]), centre=jnp.zeros(2), shape=(90, 90)),
    "rotated by 90 degrees": Lattice.regular((80, 96), (0.1, 0.1), rotation=90.0),
    "sub-pixel shift only": Lattice.regular((96, 80), (0.12, 0.1), rotation=20.0, centre=(0.061, 0.0237)),
    "different shape, same pitch": Lattice.regular((120, 110), (0.12, 0.1), rotation=20.0, centre=(0.05, 0.02)),
}


@pytest.mark.parametrize("name", TARGETS)
def test_resample_reproduces_the_analytic_field_on_another_lattice(name):
    target = TARGETS[name]
    got = resample_field(field(SOURCE), SOURCE, target)
    assert got.shape == target.shape
    assert rel_err(got, field(target)) < 1e-12


def test_identical_lattices_return_the_input_untouched():
    u = field(SOURCE)
    assert resample_field(u, SOURCE, SOURCE) is not None
    assert np.array_equal(np.asarray(resample_field(u, SOURCE, SOURCE)), np.asarray(u))


def test_round_trip_through_a_rotated_lattice_and_back():
    there = Lattice.regular((104, 104), (0.1, 0.1), rotation=33.0, centre=(0.0, 0.03))
    u = field(SOURCE)
    back = resample_field(resample_field(u, SOURCE, there), there, SOURCE)
    assert rel_err(back, u) < 1e-12


@pytest.mark.parametrize("seed", range(12))
def test_random_lattice_pairs(seed):
    """Random rotation, anisotropic scale, shear, mirror, offset and shape: the pivoted two-pass algorithm
    reproduces the analytic field."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 2 * np.pi)
    src = Lattice(pitch=jnp.asarray(np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ np.diag(rng.uniform(0.09, 0.13, 2))),
                  centre=jnp.asarray(rng.normal(size=2) * 0.05), shape=(int(rng.integers(90, 110)), int(rng.integers(90, 110))))
    b = rng.uniform(0, 2 * np.pi)
    P = np.array([[np.cos(b), -np.sin(b)], [np.sin(b), np.cos(b)]]) @ np.diag(rng.uniform(0.09, 0.12, 2))
    P = P @ np.array([[1.0, rng.uniform(-0.25, 0.25)], [0.0, 1.0]])
    if seed % 3 == 0:
        P = P @ np.diag([1.0, -1.0])
    dst = Lattice(pitch=jnp.asarray(P), centre=jnp.asarray(rng.normal(size=2) * 0.1),
                  shape=(int(rng.integers(70, 100)), int(rng.integers(70, 100))))
    u = field(src, centre=(0.0, 0.0), slopes=(0.0, 0.0))
    got = resample_field(u, src, dst, tol=1e-12)
    assert rel_err(got, field(dst, centre=(0.0, 0.0), slopes=(0.0, 0.0))) < 1e-8, seed


def test_batch_axes_and_flux():
    target = TARGETS["rotated and finer"]
    u = field(SOURCE)
    batch = jnp.stack([u, 2j * u])
    got = resample_field(batch, SOURCE, target)
    assert got.shape == (2,) + target.shape
    assert rel_err(got[1], 2j * resample_field(u, SOURCE, target)) < 1e-14
    flux_in = float(jnp.sum(jnp.abs(u) ** 2) * SOURCE.cell_area)
    flux_out = float(jnp.sum(jnp.abs(got[0]) ** 2) * target.cell_area)
    assert abs(flux_out / flux_in - 1) < 1e-9


def test_a_detector_stands_for_its_lattice():
    detector = Detector(z=0.0, pixel_size=(0.1, 0.1), shape=(64, 64), rotation=10.0)
    grid = Lattice.from_grid(detector)
    u = field(SOURCE)
    assert np.allclose(np.asarray(resample_field(u, SOURCE, detector)), np.asarray(resample_field(u, SOURCE, grid)), atol=1e-14)


def test_resample_is_linear_and_differentiable_in_the_field_and_jit_compatible():
    target = TARGETS["rotated and finer"]
    u = field(SOURCE)
    weights = jnp.asarray(np.random.default_rng(2).normal(size=target.shape))

    def loss(x):
        return jnp.sum(weights * jnp.abs(resample_field(x, SOURCE, target, check="off")) ** 2)

    check_grads(loss, (u,), order=1, modes=("rev",), atol=1e-6, rtol=1e-6)
    jitted = jax.jit(lambda x: resample_field(x, SOURCE, target))
    assert rel_err(jitted(u), resample_field(u, SOURCE, target)) < 1e-14


# ------------------------------------------------------------------------------------------------
# the checks: nothing aliased is returned silently
# ------------------------------------------------------------------------------------------------
def test_an_output_lattice_that_does_not_resolve_the_field_is_refused():
    coarse = Lattice.regular((40, 40), (0.3, 0.3))
    with pytest.raises(SamplingError, match="does not resolve"):
        resample_field(field(SOURCE), SOURCE, coarse)
    assert resample_field(field(SOURCE), SOURCE, coarse, check="off").shape == (40, 40)


def test_a_field_touching_the_array_edge_or_at_nyquist_is_refused():
    lattice = Lattice.regular((64, 64), 0.1)
    filled = jnp.ones((64, 64), dtype=complex)                       # a hard edge: fills the window
    with pytest.raises(SamplingError, match="edge"):
        resample_field(filled, lattice, Lattice.regular((64, 64), 0.1, rotation=10.0))
    noise = jnp.asarray(np.random.default_rng(0).normal(size=(64, 64)))
    with pytest.raises(SamplingError, match="Nyquist"):
        resample_field(noise, lattice, Lattice.regular((64, 64), 0.1, rotation=10.0))


def test_the_shear_pass_limit_is_reported():
    """The first pass raises the row band by |shear| times the column band; a narrow-band field passes a
    large rotation, a broad-band one is refused before any interpolation."""
    lattice = Lattice.regular((96, 96), 0.1)
    rotated = Lattice.regular((96, 96), 0.1, rotation=40.0)
    broad = gaussian_field(lattice.positions(), 1.0, np.zeros(2), np.zeros(2), gaussian_Q(0.25, 0.25, 1.0), 1.0)
    with pytest.raises(SamplingError, match="first pass|does not resolve|Nyquist"):
        resample_field(broad, lattice, rotated)
    smooth = gaussian_field(lattice.positions(), 1.0, np.zeros(2), np.zeros(2), gaussian_Q(0.7, 0.7, 1.0), 1.0)
    assert rel_err(resample_field(smooth, lattice, rotated),
                   gaussian_field(rotated.positions(), 1.0, np.zeros(2), np.zeros(2), gaussian_Q(0.7, 0.7, 1.0), 1.0)) < 1e-9


def test_invalid_arguments():
    lattice = Lattice.regular((16, 16), 0.1)
    with pytest.raises(ValueError, match="check"):
        resample_field(jnp.zeros((16, 16)), lattice, lattice, check="maybe")
    with pytest.raises(ValueError, match="shape"):
        resample_field(jnp.zeros((8, 8)), lattice, lattice)
    with pytest.raises(ValueError, match="invertible"):
        resample_field(jnp.zeros((16, 16)), lattice, Lattice(pitch=jnp.zeros((2, 2)), centre=jnp.zeros(2), shape=(16, 16)))
    with pytest.raises(TypeError):
        resample_field(jnp.zeros((16, 16)), lattice, object())

    def traced(pitch):
        return resample_field(jnp.zeros((16, 16)), lattice, Lattice.regular((16, 16), pitch)).sum()

    with pytest.raises(ValueError, match="concrete"):
        jax.jit(traced)(0.1)


# ------------------------------------------------------------------------------------------------
# an imported Cartesian array through a magnetic lens: resample, propagate, resample back
# ------------------------------------------------------------------------------------------------
def _magnetic_lens(z, rotation_rad=0.5, focal=2e-3):
    Rc = float(compute_Rc_from_voltage(VOLTAGE))
    NI = rotation_rad / Rc
    return ElectromagneticLens(z=z, turns=100.0, current=NI / 100.0,
                               Gc=float(effective_accelerating_potential(VOLTAGE)) / (focal * NI**2))


@pytest.fixture(scope="module")
def imported():
    """A Cartesian array (as if read from a file), the column, the Cartesian detector and the physical answer."""
    column = [_magnetic_lens(3e-3), Plane(z=3e-3 + 1.03 * 2e-3)]
    w0 = 1e-7
    chief = make_gaussian(z=0.0, voltage=VOLTAGE, waist_x=w0, waist_y=w0)
    amp, centre, slopes, Q_out, path = chief_from_beam(run_to_end(chief, column))
    detector = Detector(z=0.0, pixel_size=(3e-9, 3e-9), shape=(128, 128))
    reference = gaussian_field(Lattice.from_grid(detector).positions(), amp, centre, slopes, Q_out, LAM, path)
    cartesian = Lattice.regular((192, 192), 10e-9)
    array = gaussian_field(cartesian.positions(), 1.0, np.zeros(2), np.zeros(2), gaussian_Q(w0, w0, LAM), LAM)
    return column, detector, reference, cartesian, array


def test_imported_array_propagated_on_the_engine_lattice_and_resampled_back(imported):
    """Forward mode: the engine chooses its output lattice (rotated by the Larmor angle, scaled by the
    Fresnel geometry); the result is interpolated onto the Cartesian detector."""
    column, detector, reference, cartesian, array = imported
    wave = propagate(Wavefront(array, cartesian, 0.0, VOLTAGE), column, tol=1e-14)
    assert abs(np.arctan2(float(wave.lattice.pitch[1, 0]), float(wave.lattice.pitch[0, 0]))) > 0.3   # a rotated lattice
    assert rel_err(resample_field(wave.data, wave.lattice, detector), reference) < 1e-10


def test_imported_array_resampled_onto_the_adapted_source_lattice(imported):
    """Backward mode: the array is interpolated onto the lattice that lands exactly on the Cartesian detector."""
    column, detector, reference, cartesian, array = imported
    source = input_lattice(column, detector, source_z=0.0, voltage=VOLTAGE, method="mft")
    assert abs(float(source.pitch[0, 1])) > 1e-9                                  # rotated: not Cartesian
    on_source = resample_field(array, cartesian, source)
    wave = propagate(Wavefront(on_source, source, 0.0, VOLTAGE), column, detector, method="mft", tol=1e-14)
    assert rel_err(wave.data, reference) < 1e-10
    assert np.allclose(np.asarray(wave.lattice.pitch), np.asarray(Lattice.from_grid(detector).pitch))
