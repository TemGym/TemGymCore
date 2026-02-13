import warnings

import jax
import jax.numpy as jnp
import numpy as np

from temgym_core import components as C
from temgym_core.gaussian import (
    make_gaussian,
    run_to_end as gaussian_run_to_end,
    run_to_end_vmapped,
)
from temgym_core.run import run_to_end as unified_run_to_end


def test_component_dispatch_preserves_gaussian_type():
    beam = make_gaussian(
        x=0.0,
        y=0.0,
        dx=0.0,
        dy=0.0,
        z=0.0,
        voltage=200e3,
        waist_x=1e-6,
        waist_y=1e-6,
    )
    lens = C.Lens(z=0.0, focal_length=0.5)

    out = lens(beam)

    assert getattr(out, "ray_family", None) == "gaussian"
    assert hasattr(out, "Q_inv")
    assert hasattr(out, "amplitude")


def test_unified_runner_auto_selects_gaussian_propagator():
    beam = make_gaussian(
        x=0.0,
        y=0.0,
        dx=1e-3,
        dy=-2e-3,
        z=0.0,
        voltage=200e3,
        waist_x=2e-6,
        waist_y=2e-6,
    )
    detector = C.Detector(z=0.1, pixel_size=(1e-6, 1e-6), shape=(8, 8))

    out_auto = unified_run_to_end(beam, (detector,))
    out_ref = gaussian_run_to_end(beam, (detector,))

    np.testing.assert_allclose(np.asarray(out_auto.x), np.asarray(out_ref.x), atol=1e-12)
    np.testing.assert_allclose(np.asarray(out_auto.y), np.asarray(out_ref.y), atol=1e-12)
    np.testing.assert_allclose(np.asarray(out_auto.pathlength), np.asarray(out_ref.pathlength), atol=1e-12)
    np.testing.assert_allclose(np.asarray(out_auto.Q_inv), np.asarray(out_ref.Q_inv), atol=1e-12)


def test_biprism_aliases_emit_deprecation_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = C.Biprism(z=0.0, def_x=1e-3)
    assert any("deprecated" in str(w.message).lower() for w in caught)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = C.AberratedLensKrivanek(z=0.0, focal_length=1.0, coeffs={})
    assert any("deprecated" in str(w.message).lower() for w in caught)


def test_phase_biprism_and_deflection_biprism_are_distinct():
    p = C.PhaseBiprism(z=0.0, strength=1.0, width=1e-6)
    d = C.DeflectionBiprism(z=0.0, def_x=1e-3)

    assert type(p) is C.PhaseBiprism
    assert type(d) is C.DeflectionBiprism
    assert p is not d


def test_phase_biprism_none_length_runs_in_vmapped_gaussian_runner():
    beam = make_gaussian(
        x=0.0,
        y=0.0,
        dx=0.0,
        dy=0.0,
        z=0.0,
        voltage=200e3,
        waist_x=1e-6,
        waist_y=1e-6,
    )
    biprism = C.PhaseBiprism(z=0.01, strength=1.0, width=1e-6, length=None)

    beam_batched = jax.tree_util.tree_map(lambda a: jnp.expand_dims(a, axis=0), beam)
    out = run_to_end_vmapped(beam_batched, (biprism,))

    assert np.isfinite(np.asarray(out.pathlength)).all()
