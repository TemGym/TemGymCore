"""Independent aperture centres, alignment accuracy and shared current references."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from temgym_core.calibration.apertures import (
    alignment_residual,
    aperture_centres,
    aperture_moments,
    aperture_specification,
    current_fractions,
    make_aperture_transmission,
    make_focus_transmission,
    simulate_aligned_apertures,
)
from temgym_core.calibration.inference import condenser_starting_guesses


def initial(number_of_apertures):
    return jnp.asarray(
        condenser_starting_guesses(
            starting_guess_path=Path(__file__).parents[1]
            / "src/temgym_core/calibration/presets/microscope_condenser_starting_guess.json",
            number_of_apertures=number_of_apertures,
            number_of_broad_starts=0,
        )[0][0]
    )


def test_independent_centres_reduce_to_common_centre_and_have_local_derivatives():
    p = initial(3)
    spec = aperture_specification()
    old = make_focus_transmission(specification=spec)
    new = make_aperture_transmission(
        specification={**spec, "independent_centres": True}
    )
    command = jnp.array([1.75, 0.75, 1.0])
    np.testing.assert_allclose(new(p, command), old(p, command), rtol=1e-11, atol=1e-13)
    jac = np.asarray(jax.jacfwd(lambda x: new(x, command))(p))
    np.testing.assert_array_equal(jac[0, 29:], 0.0)
    np.testing.assert_array_equal(jac[1, 31:], 0.0)
    np.testing.assert_array_equal(jac[2, 29:31], 0.0)
    assert np.linalg.norm(jac[1, 29:31]) > 0


def test_alignment_is_finite_accuracy_for_each_hole_and_can_be_removed():
    p = initial(3)
    controls = jnp.array([1.75, 0.75, 1.0])
    mean, _ = aperture_moments(p, controls, "after")
    centre = mean * 1000000.0 + p[20:22]
    p = p.at[20:22].set(centre).at[29:].set(jnp.tile(centre, 2))
    constraint = {
        "lens_controls": controls.tolist(),
        "mean_minus_aperture_centre_m": [0.0, 0.0],
        "isotropic_sigma_m": 2e-07,
        "number_of_apertures": 3,
    }
    np.testing.assert_allclose(alignment_residual(p, constraint), 0.0, atol=1e-14)
    p = p.at[29].add(0.2)
    np.testing.assert_allclose(
        alignment_residual(p, constraint), [0, 0, -1, 0, 0, 0], atol=1e-14
    )
    assert alignment_residual(p, {**constraint, "enabled": False}).shape == (0,)


def test_current_ratios_preserve_shared_reference_covariance():
    values, cov = current_fractions([[20, 60]], [100])
    np.testing.assert_allclose(values, [[0.2, 0.6]])
    np.testing.assert_allclose(cov, [[[0.0024, 0.0012], [0.0012, 0.0096]]])
    one, one_cov = current_fractions([20], [100])
    assert one.shape == (1, 1) and one_cov.shape == (1, 1, 1)


def test_two_and_three_holes_share_data_but_have_distinct_alignment_errors(tmp_path):
    for count in [2, 3]:
        root = tmp_path / str(count)
        root.mkdir()
        (root / "experiment.json").write_text(
            json.dumps(
                {"voltages_v": [200000.0, 195000.0], "reference_voltage_v": 200000.0}
            )
        )
        (root / "scoring_truth.json").write_text(
            json.dumps({"parameters": np.asarray(initial(1)).tolist()})
        )
        for index in [0, 1]:
            folder = root / f"voltage_{index}"
            (folder / "prediction").mkdir(parents=True)
            (folder / "aperture.json").write_text(json.dumps(aperture_specification()))
            for name in ["counts.npz", "prediction/current_counts.npz"]:
                np.savez(folder / name, controls=[[1.75, 0.75, 1.0], [1.8, 1.0, 1.0]])
        simulate_aligned_apertures(
            experiment_directory=root,
            diameters_um=[4.0, 10.0, 20.0][:count],
            centering_sigma_um=0.2,
        )
    for index in [0, 1]:
        two = np.load(tmp_path / f"2/voltage_{index}/counts.npz")
        three = np.load(tmp_path / f"3/voltage_{index}/counts.npz")
        np.testing.assert_array_equal(
            two["transmitted_electrons"], three["transmitted_electrons"][:, :2]
        )
        np.testing.assert_array_equal(
            two["reference_electrons"], three["reference_electrons"]
        )
    truth = json.loads((tmp_path / "3/scoring_truth.json").read_text())
    centres = np.asarray(aperture_centres(truth["parameters"], 3))
    assert not np.allclose(centres[0], centres[1])
    constraint = json.loads(
        (tmp_path / "3/alignment/alignment_constraint.json").read_text()
    )
    assert constraint["isotropic_sigma_m"] == 2e-07
    assert constraint["current_readings"] == 0
    assert not (tmp_path / "3/alignment/scan.npz").exists()
