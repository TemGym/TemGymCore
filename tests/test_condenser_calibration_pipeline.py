"""Check information content, physical transport, and the truth-free fitting boundary."""

import json
from pathlib import Path

import jax
import numpy as np

EXAMPLES = Path(__file__).parents[1] / "src/temgym_core/calibration/presets"
jax.config.update("jax_enable_x64", True)
from temgym_core.calibration.projector_model import matrix5 as projector_matrix5


def test_calibrated_views_are_real_projector_maps_with_fixed_objective():
    record = json.loads((EXAMPLES / "condenser_projector_views.json").read_text())
    for view in record["views"]:
        assert view["controls"][0] == 1.0
        expected = projector_matrix5(
            np.asarray(record["calibrated_projector_parameters"]),
            np.asarray(view["controls"]),
        )
        np.testing.assert_allclose(view["matrix"], expected, rtol=2e-12, atol=1e-08)


def test_condenser_design_changes_size_at_one_real_plane():
    import jax
    import numpy as np

    from temgym_core.calibration.condenser_model import (
        condenser_components,
        design_condenser_curve,
        generating_parameters,
    )
    from temgym_core.components import Plane
    from temgym_core.ray import Ray
    from temgym_core.run import column_matrix, run_to_end

    jax.config.update("jax_enable_x64", True)
    parameters = generating_parameters()
    curve = design_condenser_curve(
        parameters, image_magnifications=[0.05, 0.1, 0.2, 0.3]
    )
    source_z = -np.exp(parameters[:9]).sum()
    for target, controls, matrix in zip(
        curve["image_magnifications"], curve["controls"], curve["image_matrices"]
    ):
        column = condenser_components(parameters, controls)
        assert column[1].z + column[1].length < curve["image_plane_z_m"] < column[2].z
        np.testing.assert_allclose(np.linalg.svd(matrix[:2, :2])[1], target, atol=1e-10)
        np.testing.assert_allclose(matrix[:2, 2:4], 0, atol=1e-10)
        # Trace a finite off-axis source point at different angles: it must
        # arrive at the same image position, including native lens rotation.
        for angle in [-1e-4, 1e-4]:
            ray = Ray(
                x=8e-9,
                y=-3e-9,
                dx=angle,
                dy=angle / 2,
                z=source_z,
                pathlength=0.0,
                voltage=200e3,
            )
            image = run_to_end(ray, (*column[:2], Plane(z=curve["image_plane_z_m"])))
            np.testing.assert_allclose(
                [image.x, image.y],
                matrix[:2, :2] @ [8e-9, -3e-9],
                atol=1e-13,
            )
        full = np.asarray(column_matrix(column, source_z=source_z))
        np.testing.assert_allclose(full[2:4, 2:4], 0, atol=1e-8)
    np.testing.assert_array_equal(curve["controls"][:, 2], 1.5)
    assert np.all(np.diff(curve["controls"][:, 0]) < 0)
    assert np.all(np.diff(curve["controls"][:, 1]) > 0)


def test_condenser_design_rejects_virtual_image_plane():
    import pytest

    from temgym_core.calibration.condenser_model import (
        design_condenser_curve,
        generating_parameters,
    )

    with pytest.raises(ValueError, match="real front focal plane"):
        design_condenser_curve(
            generating_parameters(), image_magnifications=[0.1], mini_control=1.0
        )


def test_condenser_design_rejects_unreachable_size():
    import pytest

    from temgym_core.calibration.condenser_model import (
        design_condenser_curve,
        generating_parameters,
    )

    with pytest.raises(ValueError, match="No accurate condenser design"):
        design_condenser_curve(
            generating_parameters(),
            image_magnifications=[0.1],
            initial_controls=(0.15, 0.15),
            control_bounds=(0.1, 0.2),
        )
