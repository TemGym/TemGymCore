from numpy.testing import assert_allclose

from temgym_core.components import Scanner, Plane, Descanner
from temgym_core.source import PointSource
from temgym_core.ray import Ray
from temgym_core.run import run_iter
from temgym_core.propagator import Propagator, FreeSpaceParaxial


def test_run_iter():
    # These components shouldn't change the ray as it passes through
    components = (
        PointSource(z=0., semi_conv=0.023),
        Scanner(z=1.2, scan_pos_x=0., scan_pos_y=0.),
        Plane(z=1.2),
        Descanner(z=1.2, scan_pos_x=0., scan_pos_y=0.),
        Plane(z=3.1)
    )
    ray = Ray(
        x=0.12,
        y=0.23,
        dx=0.34,
        dy=0.45,
        z=3.14,
        pathlength=0.34
    )
    res = list(run_iter(ray=ray, components=components))

    prev_ray = ray

    for i, component in enumerate(components):
        prop_index = 2*i
        comp_index = 2*i + 1
        prop, prop_r = res[prop_index]
        comp, comp_r = res[comp_index]
        assert isinstance(prop, Propagator)
        assert isinstance(prop.propagator, FreeSpaceParaxial)
        assert prop.distance == component.z - prev_ray.z
        assert_allclose(prop_r.z, comp.z)
        assert_allclose(comp_r.z, comp.z)
        assert prev_ray.dx == prop_r.dx
        assert prev_ray.dy == prop_r.dy
        assert_allclose(prop_r.x, prev_ray.x + prev_ray.dx*prop.distance)
        assert_allclose(prop_r.y, prev_ray.y + prev_ray.dy*prop.distance)
        # FIXME add test for correct path length

        prev_ray = comp_r


def test_run_iter_noprop():
    # everything at the same z level
    z = 1.2
    # These components do change the ray, i.e. we
    # test that run_iter() actually passes the ray through the components
    components = (
        PointSource(z=z, semi_conv=0.023),
        Scanner(z=z, scan_pos_x=23., scan_pos_y=42.),
        Plane(z=z),
        Descanner(z=z, scan_pos_x=13., scan_pos_y=11.),
        Plane(z=z)
    )
    ray = Ray(
        x=0.12,
        y=0.23,
        dx=0.34,
        dy=0.45,
        z=z,
        pathlength=0.34
    )
    res = list(run_iter(ray=ray, components=components))

    # Reference result: Compose the components without propagation
    res_ref = ray
    for comp in components:
        res_ref = comp(res_ref)

    prev_ray = ray
    for i, component in enumerate(components):
        prop_index = 2*i
        comp_index = 2*i + 1
        prop, prop_r = res[prop_index]
        comp, comp_r = res[comp_index]
        assert isinstance(prop, Propagator)
        assert isinstance(prop.propagator, FreeSpaceParaxial)
        assert prop.distance == component.z - prev_ray.z
        assert_allclose(prop_r.z, comp.z)
        assert_allclose(comp_r.z, comp.z)
        prev_ray = comp_r

    final_ray = res[-1][1]
    for attr in ('x', 'y', 'dx', 'dy', '_one', 'z', 'pathlength'):
        assert_allclose(getattr(final_ray, attr), getattr(res_ref, attr))
