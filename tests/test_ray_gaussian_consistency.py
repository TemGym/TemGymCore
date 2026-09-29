"""A Gaussian beamlet's centre follows the ray it is centred on, through every component.

``pathlength`` is the excess optical path: what a ray accumulates beyond its straight-line
axial travel. Free space adds ``d * (dx**2 + dy**2) / 2`` to rays and to Gaussian beamlets alike,
and every thin element adds its optical path while kicking the slope by the path's gradient.
Under those two rules ``component(ray)`` and ``component(beamlet)`` give one chief-ray state,
and a whole column does too.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.aberrations import KrivanekCoeffs
from temgym_core.components import (
    ConstantPhaseShift,
    DeflectionBiprism,
    Deflector,
    Descanner,
    Detector,
    DoubleDeflector,
    ElectromagneticLens,
    GaussianActionComponent,
    HardEdgeSolenoid,
    KrivanekLens,
    Lens,
    LinearPhaseShift,
    MagneticPhaseSample,
    PhaseBiprism,
    Plane,
    QuadraticPhaseShift,
    Rotator,
    RotatingLens,
    Scanner,
    SigmoidAperture,
    Stigmator,
    ZernikeAberration,
)
from temgym_core.gaussian import FreeSpacePropagator, make_gaussian
from temgym_core.propagator import FreeSpaceParaxial
from temgym_core.ray import Ray
from temgym_core.run import run_to_end

jax.config.update("jax_enable_x64", True)

VOLTAGE = 200e3
START = dict(x=3e-4, y=-2e-4, dx=4e-3, dy=-3e-3)


def pair(z=0.0, **overrides):
    """The same chief ray as a ``Ray`` and as the centre of a narrow Gaussian beamlet."""
    state = {**START, **overrides}
    ray = Ray(z=z, pathlength=0.0, voltage=VOLTAGE, **state)
    beam = make_gaussian(z=z, voltage=VOLTAGE, waist_x=2e-5, waist_y=2e-5, **state)
    return ray, beam


def chief(out):
    return np.array([float(np.squeeze(v)) for v in (out.x, out.y, out.dx, out.dy, out.pathlength)])


def assert_same_chief(ray_out, beam_out):
    expected, actual = chief(ray_out), chief(beam_out)
    np.testing.assert_allclose(actual[:4], expected[:4], rtol=1e-10, atol=1e-16)
    np.testing.assert_allclose(actual[4], expected[4], rtol=1e-8, atol=1e-16)


COEFFS = KrivanekCoeffs(C10=2e-6, C12=1e-6, phi12=0.4, C21=1e-5, phi21=0.2, C30=1e-3)
THIN = {
    "plane": Plane(z=0.0),
    "detector": Detector(z=0.0, pixel_size=(1e-5, 1e-5), shape=(8, 8)),
    "lens": Lens(z=0.0, focal_length=0.7),
    "lens_off_axis": Lens(z=0.0, focal_length=0.7, x0=1e-4, y0=-5e-5),
    "stigmator": Stigmator(z=0.0, focal_length_x=0.6, focal_length_y=0.9),
    "krivanek_lens": KrivanekLens(z=0.0, focal_length=0.5, coeffs=COEFFS),
    "deflector": Deflector(z=0.0, def_x=2e-4, def_y=-1e-4),
    "scanner": Scanner(z=0.0, scan_pos_x=1e-6, scan_pos_y=-2e-6, scan_tilt_x=1e-5),
    "descanner": Descanner(z=0.0, scan_pos_x=1e-6, scan_pos_y=-2e-6, scan_tilt_x=1e-5),
    "rotator": Rotator(z=0.0, angle=12.0),
    "rotating_lens": RotatingLens(z=0.0, focal_length=0.4, rotation=20.0),
    "deflection_biprism": DeflectionBiprism(z=0.0, def_x=3e-5),
    "phase_biprism": PhaseBiprism(z=0.0, strength=3.5e-5, width=0.6e-6, theta=0.3,
                                  sharpness=4 / 100e-9, eps=0.02),
    "constant_phase": ConstantPhaseShift(z=0.0, constant_phase_shift=3e-9),
    "linear_phase": LinearPhaseShift(z=0.0, linear_phase_shift=jnp.array([2e-4, -1e-4])),
    "quadratic_phase": QuadraticPhaseShift(
        z=0.0, quadratic_phase_shift=jnp.array([[1.2, 0.3], [0.3, -0.7]])),
    "zernike": ZernikeAberration(z=0.0, coefficients={4: 2e-9, 5: 1e-9, 7: 5e-10}, radius=1e-3),
    "magnetic_phase_sample": MagneticPhaseSample(z=0.0, strength=2e-9, width=8e-4, height=6e-4,
                                                 edge_sharpness=5e3),
    "solenoid": HardEdgeSolenoid(z=0.0, length=0.02, q=0.3),
    "electromagnetic_lens": ElectromagneticLens(z=0.0, turns=1000.0, current=1.0, Gc=0.7),
    "electromagnetic_lens_body": ElectromagneticLens(z=0.0, turns=1000.0, current=1.0, Gc=0.7,
                                                     Tc=5e-6),
    "electromagnetic_lens_stigmated": ElectromagneticLens(
        z=0.0, turns=1000.0, current=1.0, Gc=0.7, Tc=5e-6, stigmator_strength_x=0.05),
    "double_deflector": DoubleDeflector(z=0.0, spacing=0.05, shift_x=1e-6, tilt_x=2e-5,
                                        tilt_y=-1e-5),
}


@pytest.mark.parametrize("name", THIN)
def test_component_chief_ray(name):
    component = THIN[name]
    ray, beam = pair()
    assert_same_chief(component(ray), component(beam))


@pytest.mark.parametrize("name", THIN)
def test_component_chief_ray_with_off_axis_start(name):
    component = THIN[name]
    ray, beam = pair(x=-1.5e-4, y=4e-4, dx=-2e-3, dy=6e-3)
    assert_same_chief(component(ray), component(beam))


def thin_action_components():
    return [(n, c) for n, c in THIN.items() if isinstance(c, GaussianActionComponent)
            and type(c).phase_shift is not GaussianActionComponent.phase_shift
            and n not in {"aperture"}]


@pytest.mark.parametrize("name,component", thin_action_components())
def test_thin_element_kick_is_the_gradient_of_its_path(name, component):
    """Slope kick = grad of the optical path the element adds (Hamilton)."""
    ray, _ = pair()
    out = component(ray)
    xy = jnp.array([ray.x, ray.y])
    path_gradient = jax.grad(lambda p: jnp.real(component.phase_shift(p)))(xy)
    np.testing.assert_allclose([out.dx - ray.dx, out.dy - ray.dy], path_gradient,
                               rtol=1e-8, atol=1e-16)
    np.testing.assert_allclose(out.pathlength - ray.pathlength,
                               jnp.real(component.phase_shift(xy)), rtol=1e-8, atol=1e-16)


@pytest.mark.parametrize("distance", [0.3, -0.2, 1e-4])
def test_free_space_is_hamilton_consistent(distance):
    ray, beam = pair()
    ray_out = FreeSpaceParaxial()(ray, distance)
    beam_out = FreeSpacePropagator()(beam, distance)
    assert_same_chief(ray_out, beam_out)
    slope_sq = START["dx"] ** 2 + START["dy"] ** 2
    np.testing.assert_allclose(float(np.squeeze(ray_out.pathlength)), 0.5 * distance * slope_sq,
                               rtol=1e-12)


def test_axial_ray_accumulates_no_excess_path():
    ray = Ray(x=1e-4, y=0.0, dx=0.0, dy=0.0, z=0.0, pathlength=0.0, voltage=VOLTAGE)
    out = FreeSpaceParaxial()(ray, 0.9)
    assert float(out.pathlength) == 0.0


def test_column_chief_ray_matches_gaussian_centre():
    column = (
        Lens(z=0.2, focal_length=0.15),
        Deflector(z=0.3, def_x=1e-4, def_y=-5e-5),
        HardEdgeSolenoid(z=0.4, length=0.03, q=0.5),
        PhaseBiprism(z=0.55, strength=2e-5, width=1e-4, theta=0.4, x0=5e-3, sharpness=4e4,
                     eps=0.02),
        ElectromagneticLens(z=0.6, turns=1000.0, current=1.0, Gc=0.7, Tc=5e-6),
        Detector(z=0.9, pixel_size=(1e-5, 1e-5), shape=(8, 8)),
    )
    ray, beam = pair()
    assert_same_chief(run_to_end(ray, column), run_to_end(beam, column))


def test_ray_pathlength_is_excess_over_axial_distance():
    """A column of pure drifts: the ray's path is the sum of d * slope**2 / 2."""
    ray, _ = pair()
    out = run_to_end(ray, (Plane(z=0.1), Plane(z=0.5), Plane(z=0.8)))
    slope_sq = START["dx"] ** 2 + START["dy"] ** 2
    np.testing.assert_allclose(float(out.pathlength), 0.5 * 0.8 * slope_sq, rtol=1e-12)


def test_absorbing_aperture_passes_rays_through():
    """Rays carry no amplitude, so an aperture leaves them alone (a beamlet recentres instead)."""
    aperture = SigmoidAperture(z=0.0, radius=1e-3, edge_width=1e-4)
    ray, _ = pair()
    assert chief(aperture(ray)).tolist() == chief(ray).tolist()


def test_deflection_biprism_kick_is_the_gradient_of_its_path():
    biprism = DeflectionBiprism(z=0.0, def_x=3e-5)
    for x in (2e-4, -2e-4):
        ray, _ = pair(x=x)
        out = biprism(ray)
        np.testing.assert_allclose(out.dx - ray.dx, 3e-5 * np.sign(x), rtol=1e-12)
        np.testing.assert_allclose(out.pathlength, 3e-5 * abs(x), rtol=1e-12)
