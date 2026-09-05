"""Physical imaging checks through the public component propagation API."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.components import Plane
from temgym_core.imaging_modes import imaging_geometry
from temgym_core.microscope_model import MicroscopeModel
from temgym_core.ray import Ray
from temgym_core.run import run_to_end


jax.config.update("jax_enable_x64", True)
CONFIG = Path(__file__).parents[1] / "examples/microscope_models/microscope.toml"
TARGETS = (40_000, 60_000, 80_000, 100_000, 150_000,
           200_000, 250_000, 300_000, 400_000, 600_000)


@pytest.fixture(scope="module")
def model():
    return MicroscopeModel.from_toml(str(CONFIG))


def trace_basis(model, target, *, start=None, end=None, combined=False):
    """Propagate physical height/angle basis rays; no solver matrix is reused."""
    start = model.auxiliary["sample"]["z"] if start is None else start
    end = model.auxiliary["detector"]["z"] if end is None else end
    modes = {"magnification": target, "parallel": 3.0} if combined else {"magnification": target}
    components = [c for c in model.build_components(modes) if start < float(c.z) < end]
    components.append(Plane(z=end))
    scales = np.array((1e-9, 1e-9, 1e-4, 1e-4))
    states = jnp.asarray(np.diag(scales))
    ray = Ray(x=states[0], y=states[1], dx=states[2], dy=states[3],
              z=start, pathlength=0.0, voltage=model.voltage)
    out = run_to_end(ray, components)
    return np.asarray((out.x, out.y, out.dx, out.dy)) / scales[None, :]


@pytest.mark.parametrize("target", TARGETS)
def test_stored_currents_form_rotation_free_specimen_image(model, target):
    # Disable the runtime solve so it cannot repair a defective stored row.
    stored_model = replace(model, modes={**model.modes, "magnification": replace(
        model.modes["magnification"], conjugate_imaging=None,
    )})
    matrix = trace_basis(stored_model, target, combined=True)
    np.testing.assert_allclose(matrix[:2, :2] / target, np.eye(2), rtol=0, atol=1e-10)
    np.testing.assert_allclose(matrix[:2, 2:], 0, rtol=0, atol=1e-10)


@pytest.mark.parametrize("target", (50_000, 70_000, 90_000, 125_000, 175_000,
                                    225_000, 275_000, 350_000, 500_000))
def test_intermediate_magnifications_are_refocused(model, target):
    matrix = trace_basis(model, target)
    np.testing.assert_allclose(matrix[:2, :2] / target, np.eye(2), rtol=0, atol=1e-10)
    np.testing.assert_allclose(matrix[:2, 2:], 0, rtol=0, atol=1e-10)


@pytest.mark.parametrize("target", (40_000, 200_000, 600_000))
def test_three_defined_stages_are_conjugate_and_compose(model, target):
    currents = model.modes["magnification"].interpolate_currents(target)
    g = imaging_geometry(model, currents)
    planes = (g["sample_z"], g["objective_image_z"], g["projector_object_z"], g["detector_z"])
    magnifications = (abs(g["objective_magnification"]),
                      target / (g["objective_magnification"] * g["projector_magnification"]),
                      abs(g["projector_magnification"]))
    total = np.eye(4)
    for start, end, mag in zip(planes[:-1], planes[1:], magnifications):
        matrix = trace_basis(model, target, start=start, end=end)
        np.testing.assert_allclose(matrix[:2, 2:], 0, atol=1e-10)
        np.testing.assert_allclose(np.linalg.svd(matrix[:2, :2], compute_uv=False), mag, rtol=1e-10)
        total = matrix @ total
    np.testing.assert_allclose(total, trace_basis(model, target), rtol=1e-10, atol=1e-8)


@pytest.mark.parametrize("target", (39_999, 600_001, float("nan")))
def test_focused_modes_reject_out_of_range_requests(model, target):
    with pytest.raises(ValueError, match="configured range"):
        model.build_components("magnification", target)


def test_opted_in_solver_rejects_unsupported_thick_lens(model):
    changed = replace(model, lenses=tuple(
        replace(lens, Tc=1e-8) if lens.name == "IL1" else lens for lens in model.lenses
    ))
    with pytest.raises(ValueError, match="thin, round"):
        changed.build_components("magnification", 40_000)


def test_infeasible_current_bounds_do_not_return_an_unfocused_mode(model):
    mode = model.modes["magnification"]
    changed = replace(model, modes={**model.modes, "magnification": replace(
        mode, conjugate_imaging=replace(mode.conjugate_imaging, current_bounds_a=(0.1, 0.2)),
    )})
    with pytest.raises(ValueError, match="No converged"):
        changed.build_components("magnification", 40_000)
