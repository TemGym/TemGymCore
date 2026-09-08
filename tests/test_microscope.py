"""Public microscope API: native components, explicit settings, physical units."""
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.components import ElectromagneticLens, HardEdgeSolenoid, Plane
from temgym_core.constants import effective_accelerating_potential
from temgym_core.microscope import CurrentTable, Microscope, find_imaging_condition
from temgym_core.run import column_matrix, run_to_end
from temgym_core.ray import Ray

jax.config.update("jax_enable_x64", True)
CONFIG = Path(__file__).parents[1] / "examples/microscope_models/microscope.toml"


def test_loads_native_components_and_preserves_physical_units():
    model = Microscope.from_toml(CONFIG)
    assert isinstance(model.components["sample"], Plane)
    assert model.components["detector"].pixel_size == (55e-6, 55e-6)
    assert model.components["CL12_deflector"].shift_balance_x == -1.
    assert [c.z for c in model.column] == sorted(c.z for c in model.column)
    powered = model.with_settings(parallel=3, magnification=40_000)
    for name, spec in model.config["lenses"].items():
        lens = powered.components[name]
        assert isinstance(lens, ElectromagneticLens)
        expected = spec["Gc"] * (spec["turns"] * lens.current)**2
        np.testing.assert_allclose(1 / lens.focal_length(model.voltage), expected)
        ratio = effective_accelerating_potential(201_000) / effective_accelerating_potential(200_000)
        np.testing.assert_allclose(lens.focal_length(201_000) / lens.focal_length(200_000), ratio)


def test_updates_preserve_original_and_accept_native_nuisance_parameters():
    original = Microscope.from_toml(CONFIG).with_settings(magnification=40_000)
    lens = original.components["IL1"]
    changed = original.with_currents(IL1=lens.current * 1.05)
    changed = changed.with_components(IL1=replace(changed.components["IL1"], x0=1e-6))
    assert original.components["IL1"] is lens
    assert changed.components["IL1"].x0 == 1e-6
    assert lens.x0 == 0
    assert changed.components["PL1"] is original.components["PL1"]
    with pytest.raises(KeyError):
        original.with_currents(misspelled_lens=1.)


def test_tables_interpolate_amperes_and_only_touch_their_named_lenses():
    table = CurrentTable(("L",), [1, 3], [[-2], [4]])
    assert table.at(2) == {"L": 1.}
    lens = ElectromagneticLens(z=.1, turns=1, current=0, Gc=1)
    model = Microscope({"L": lens, "other": replace(lens, z=.2, current=7)},
                       tables={"first": table, "second": CurrentTable(("L",), [1], [[3]])})
    assert model.with_settings(first=2).components["other"].current == 7
    with pytest.raises(ValueError, match="conflicting"):
        model.with_settings(first=2, second=1)


@pytest.mark.parametrize("value", [0, 4, np.nan, np.inf])
def test_tables_reject_outside_range(value):
    with pytest.raises(ValueError, match="table range"):
        CurrentTable(("L",), [1, 3], [[1], [2]]).at(value)


@pytest.mark.parametrize("names,controls,currents", [
    (("L",), [2, 1], [[1], [2]]),
    (("L",), [1, 2], [[1]]),
    (("L", "L"), [1], [[1, 2]]),
    (("L",), [1], [[np.nan]]),
])
def test_malformed_tables_are_rejected(names, controls, currents):
    with pytest.raises(ValueError):
        CurrentTable(names, controls, currents)


def test_unknown_table_lenses_are_rejected():
    with pytest.raises(ValueError, match="existing"):
        Microscope({}, tables={"bad": CurrentTable(("missing",), [1], [[1]])})


def test_no_operating_tables_required_and_other_lens_types_are_native():
    solenoid = HardEdgeSolenoid(z=.1, length=.01, q=.02)
    model = Microscope({"solenoid": solenoid, "detector": Plane(z=.3)})
    np.testing.assert_array_equal(column_matrix(model.column), column_matrix((solenoid, Plane(z=.3))))


def test_current_updates_remain_differentiable_through_native_components():
    model = Microscope({"L": ElectromagneticLens(z=.1, turns=100, current=1., Gc=1.)})
    def focal_power(current):
        lens = model.with_currents(L=current).components["L"]
        return 1 / lens.focal_length(200_000.)
    derivative = jax.jit(jax.grad(focal_power))(jnp.array(1.))
    np.testing.assert_allclose(derivative, 2 * focal_power(1.))


TARGETS = (40_000, 60_000, 80_000, 100_000, 150_000,
           200_000, 250_000, 300_000, 400_000, 600_000)
RELAY = ("IL1", "IL2", "IL3")


@pytest.fixture(scope="module")
def model():
    return Microscope.from_toml(CONFIG)


def trace_basis(model, start=None, end=None):
    start = model.components["sample"].z if start is None else start
    end = model.components["detector"].z if end is None else end
    components = [c for c in model.column if start < float(c.z) < end] + [Plane(z=end)]
    scales = np.array((1e-9, 1e-9, 1e-4, 1e-4))
    states = jnp.asarray(np.diag(scales))
    ray = Ray(x=states[0], y=states[1], dx=states[2], dy=states[3],
              z=start, pathlength=0.0, voltage=model.voltage)
    out = run_to_end(ray, components)
    return np.asarray((out.x, out.y, out.dx, out.dy)) / scales[None, :]


@pytest.mark.parametrize("target", TARGETS)
def test_stored_currents_form_rotation_free_specimen_image(model, target):
    matrix = trace_basis(model.with_settings(magnification=target, parallel=3.))
    np.testing.assert_allclose(matrix[:2, :2] / target, np.eye(2), rtol=0, atol=1e-10)
    np.testing.assert_allclose(matrix[:2, 2:], 0, rtol=0, atol=1e-10)


@pytest.mark.parametrize("target", (50_000, 70_000, 90_000, 125_000, 175_000,
                                    225_000, 275_000, 350_000, 500_000))
def test_explicit_solve_refocuses_intermediate_magnifications(model, target):
    initial = model.with_settings(magnification=target, parallel=3.)
    # Selecting a table is interpolation only; it must not silently refocus.
    assert np.max(np.abs(trace_basis(initial)[:2, 2:])) > 1e-8
    result = find_imaging_condition(initial, magnification=target, vary=RELAY)
    matrix = trace_basis(result.microscope)
    np.testing.assert_allclose(matrix[:2, :2] / target, np.eye(2), rtol=0, atol=1e-10)
    np.testing.assert_allclose(matrix[:2, 2:], 0, rtol=0, atol=1e-10)
    np.testing.assert_allclose(result.matrix[:4, :4], matrix, rtol=1e-10, atol=1e-8)
    assert np.max(np.abs(result.residual)) < 1e-10
    for name in initial.components.keys() - set(RELAY):
        assert result.microscope.components[name] is initial.components[name]


@pytest.mark.parametrize("target", (40_000, 200_000, 600_000))
def test_three_defined_stages_are_conjugate_and_compose(model, target):
    focused = model.with_settings(magnification=target)
    objective, projector = (focused.components[name] for name in ("Obj_post", "PL1"))
    start, end = (focused.components[name].z for name in ("sample", "detector"))
    u, v = objective.z - start, end - projector.z
    objective_v = 1 / (1 / float(objective.focal_length(focused.voltage)) - 1 / u)
    projector_u = 1 / (1 / float(projector.focal_length(focused.voltage)) - 1 / v)
    planes = (start, objective.z + objective_v, projector.z - projector_u, end)
    magnifications = (objective_v / u, target / (objective_v / u * v / projector_u), v / projector_u)
    total = np.eye(4)
    for start, end, mag in zip(planes[:-1], planes[1:], magnifications):
        matrix = trace_basis(focused, start, end)
        np.testing.assert_allclose(matrix[:2, 2:], 0, atol=1e-10)
        np.testing.assert_allclose(np.linalg.svd(matrix[:2, :2], compute_uv=False), mag, rtol=1e-10)
        total = matrix @ total
    np.testing.assert_allclose(total, trace_basis(focused), rtol=1e-10, atol=1e-8)


def test_finite_electromagnetic_lenses_are_focused(model):
    initial = model.with_settings(magnification=40_000)
    initial = initial.with_components(**{
        name: replace(initial.components[name], Tc=1e-8) for name in RELAY
    })
    result = find_imaging_condition(initial, magnification=40_000, vary=RELAY)
    matrix = trace_basis(result.microscope)
    np.testing.assert_allclose(matrix[:2, :2] / 40_000, np.eye(2), atol=1e-9)
    np.testing.assert_allclose(matrix[:2, 2:], 0, atol=1e-8)


def test_infeasible_bounds_fail_instead_of_returning_unfocused_microscope(model):
    with pytest.raises(ValueError, match="No converged"):
        find_imaging_condition(model.with_settings(magnification=40_000),
                               magnification=40_000, vary=RELAY, bounds=(.1, .2))


def test_nonzero_rotation_and_second_voltage(model):
    result = find_imaging_condition(model.with_settings(magnification=60_000),
                                    magnification=60_000, rotation=.02, voltage=200_100, vary=RELAY)
    matrix = trace_basis(result.microscope)
    rotation = np.array(((np.cos(.02), -np.sin(.02)), (np.sin(.02), np.cos(.02))))
    np.testing.assert_allclose(matrix[:2, :2] / 60_000, rotation, atol=1e-10)
    np.testing.assert_allclose(matrix[:2, 2:], 0, atol=1e-10)
    assert result.microscope.voltage == 200_100


def test_two_stigmators_and_steering_with_an_off_axis_reference():
    from temgym_core.components import Stigmator, Deflector
    microscope = Microscope({
        "sample": Plane(z=0.),
        "first": Stigmator(z=.1, focal_length_x=.09, focal_length_y=.11, x0=.001),
        "second": Stigmator(z=.3, focal_length_x=.11, focal_length_y=.09),
        "steer": Deflector(z=.35, def_x=0., def_y=0.),
        "detector": Plane(z=.4),
    })
    reference = (.0002, -.0001, 0., 0.)
    result = find_imaging_condition(
        microscope, magnification=1., rotation=np.pi,
        vary=["first.focal_length_x", "first.focal_length_y",
              "second.focal_length_x", "second.focal_length_y", "steer.def_x", "steer.def_y"],
        bounds=([.05]*4 + [-.1]*2, [.15]*4 + [.1]*2),
        reference=reference, image_position=(0., 0.),
    )
    np.testing.assert_allclose(result.matrix[:2, :2], -np.eye(2), atol=1e-9)
    np.testing.assert_allclose(result.matrix[:2, 2:4], 0, atol=1e-8)
    ray = Ray(x=reference[0], y=reference[1], dx=0., dy=0., z=0., pathlength=0.)
    out = run_to_end(ray, result.microscope.column)
    np.testing.assert_allclose((out.x, out.y), (0., 0.), atol=1e-9)
    assert result.microscope.components["first"].x0 == .001


def test_one_finite_solenoid_with_adjustable_q():
    lens = HardEdgeSolenoid(z=.1, length=.02, q=.6)
    # Locate the conjugate plane for this known finite lens using basis rays.
    basis = column_matrix((lens,))
    detector_z = lens.z + lens.length - float(basis[0, 2] / basis[2, 2])
    truth = Microscope({"sample": Plane(z=0.), "lens": lens, "detector": Plane(z=detector_z)})
    target_matrix = column_matrix(truth.column)
    target = float(np.linalg.norm(target_matrix[:2, 0]))
    rotation = float(np.arctan2(target_matrix[1, 0], target_matrix[0, 0]))
    initial = truth.with_components(lens=replace(lens, q=.55))
    result = find_imaging_condition(initial, magnification=target, rotation=rotation,
                                    vary=["lens.q"], bounds=(.3, .9))
    np.testing.assert_allclose(result.microscope.components["lens"].q, .6, atol=1e-10)
    np.testing.assert_allclose(result.matrix, target_matrix, atol=1e-8)


def test_local_matrix_and_parameter_derivatives_for_a_nonlinear_component():
    import jax_dataclasses as jdc
    from temgym_core.components import Component

    @jdc.pytree_dataclass
    class CubicKick(Component):
        z: float
        strength: float

        def _call_ray(self, ray):
            return ray.derive(dx=ray.dx + self.strength * ray.x**3)

    reference = jnp.array((.2, .1, .03, .04))
    def matrix(strength):
        return column_matrix((CubicKick(z=0., strength=strength),), reference=reference)
    actual = matrix(.4)
    expected = np.eye(5)
    expected[2, 0] = 3 * .4 * .2**2
    expected[2, 4] = -2 * .4 * .2**3
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    derivative = jax.jit(jax.jacfwd(matrix))(.4)
    np.testing.assert_allclose(derivative[2, 0], 3 * .2**2, atol=1e-14)
    np.testing.assert_allclose(derivative[2, 4], -2 * .2**3, atol=1e-14)
