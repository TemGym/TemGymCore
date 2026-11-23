"""The two tests of the parked adaptive_decompose_gaussian (from tests/test_converged_image.py)."""
def test_an_adaptive_decomposition_without_splits_is_the_plain_one():
    from temgym_core.components._base import GaussianActionComponent
    from temgym_core.evaluate import evaluate_gaussians
    from temgym_core.run import run_to_end
    from temgym_core.source import adaptive_decompose_gaussian, decompose_gaussian

    beam = gaussian_waist(z=0.0, voltage=200e3, rms_size=RMS)
    detector = Detector(z=2e-7, pixel_size=(PX, PX), shape=(N, N))
    nothing = GaussianActionComponent()
    plain = decompose_gaussian(beam, num_across=15, extent_rms=3.0)
    same = adaptive_decompose_gaussian(beam, nothing, num_across=15, tolerance=1.0, extent_rms=3.0)
    assert same.x.shape == plain.x.shape
    a = evaluate_gaussians(run_to_end(plain, [detector]), detector, method="scan")
    b = evaluate_gaussians(run_to_end(same, [detector]), detector, method="scan")
    assert float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(a))) < 1e-12


def test_splitting_every_beamlet_keeps_the_field():
    from temgym_core.components._base import GaussianActionComponent
    from temgym_core.evaluate import evaluate_gaussians
    from temgym_core.run import run_to_end
    from temgym_core.source import adaptive_decompose_gaussian, decompose_gaussian

    beam = gaussian_waist(z=0.0, voltage=200e3, rms_size=RMS)
    detector = Detector(z=2e-7, pixel_size=(PX, PX), shape=(N, N))
    nothing = GaussianActionComponent()
    plain = decompose_gaussian(beam, num_across=15, extent_rms=3.0)
    split = adaptive_decompose_gaussian(beam, nothing, num_across=15, tolerance=-1.0, levels=2,
                                        extent_rms=3.0)
    assert split.x.shape[0] > 10 * plain.x.shape[0]
    a = evaluate_gaussians(run_to_end(plain, [detector]), detector, method="scan")
    b = evaluate_gaussians(run_to_end(split, [detector]), detector, method="scan")
    assert float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(a))) < 1e-3
