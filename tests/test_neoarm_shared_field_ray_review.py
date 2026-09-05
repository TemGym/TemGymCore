from pathlib import Path
import sys
import tomllib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"examples/lens_inversion"))

from neoarm_imaging_recipe import column_with_geometry_anchors
from neoarm_objective_tail_overlap import TailOverlapColumn
from neoarm_shared_objective import SharedObjectiveColumn
from neoarm_shared_field_ray_review import physical_elements,verified_trace
from temgym_core.axial_field_column import objective_tail_field
from temgym_core.immersed_objective import SharedObjectiveField
from temgym_core.neoarm_calibration import hard_edge_column_from_config

jax.config.update("jax_enable_x64",True)


@pytest.fixture(scope="module")
def column():
    config = tomllib.loads((ROOT/"examples/lens_inversion/neoarm_inversion_config.toml").read_text())
    measured = column_with_geometry_anchors(hard_edge_column_from_config(config),
        ROOT/"examples/microscope_models/measure_microscope_geometry.ipynb")
    field = SharedObjectiveField.from_window(-.05,0.,.05,n_slices_per_side=64)
    shared = SharedObjectiveColumn.from_hard_edge(measured,objective=field)
    p = jnp.asarray(shared.nominal_parameters)
    return TailOverlapColumn.from_shared(shared,p,steps_per_segment=16),p


def test_smooth_objective_glyph_marks_fwhm_not_artificial_field_stop(column):
    model,p = column
    elements,sections = physical_elements(model,p)
    objective = next(e for e in elements if "FWHM" in e.label)
    field = lambda z:float(objective_tail_field(z,p[-4:],model.mesh.normalization_field))
    assert field(objective.z)/field(p[-3]) == pytest.approx(.5,abs=2e-14)
    assert field(objective.z_end)/field(p[-3]) == pytest.approx(.5,abs=2e-14)
    assert objective.z>-.05 and objective.z_end<.05
    assert len([e for e in elements if e.role=="lens"])==8
    assert len(sections)==3


@pytest.mark.parametrize("gaussian",[True,False])
def test_shared_field_events_and_both_basis_semantics(column,gaussian):
    model,p = column
    trace,evidence,orthogonal,prefix = verified_trace(model,p,jnp.zeros(7),gaussian=gaussian)
    assert evidence["detector_basis_relative_error"]<1e-10
    assert np.all(np.diff(trace.z)>0)
    assert orthogonal.shape==(len(trace.z),2)
    sample_index = np.searchsorted(trace.z,model.base.sample_z_m)
    assert trace.position[sample_index] == pytest.approx(trace.position_scale,rel=1e-9)
    for row in evidence["events"]:
        assert abs(row["position_residual_m"])<1e-12
    if gaussian:
        assert evidence["gaussian_covariance_radius_relative_error"]<1e-8
        np.testing.assert_allclose(trace.radius,np.hypot(trace.position,trace.angle))
        assert evidence["waists"]
    else:
        assert evidence["waists"]==[]
        assert evidence["gaussian_covariance_radius_relative_error"] is None
        expected = evidence["sample_k_m_inv"]*trace.position_scale/trace.angle_scale*trace.angle
        np.testing.assert_allclose(orthogonal[:,0],expected,rtol=1e-9,atol=1e-14)
