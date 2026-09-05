"""Signed-input calibration images in the SAME column as the condenser data.

Two near-40k analyser states are calibrated independently of the older
four-projector experiment, which uses a different objective. No imported
projector posterior enters this experiment. The objective prefield and sample
magnetic shear remain conditional; the six canonical position-row coordinates
are inferred from detector centroids with calibrated sample-plane inputs.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jnp
import numpy as np

from neoarm_aperture_capture import ROOT, XY
from neoarm_condenser_extension_design import load_context, FOLDER
from neoarm_condenser_extension_images import fit_data_only_image
from neoarm_last_lens_images import render_pure_counts, signed_response_from_image_fits
from neoarm_last_lens_tomography import mechanical_shear, rotation
from neoarm_mixed_gaussian_validation import MixedGaussianSource
from neoarm_projector_branch_experiment import pure_source, signed_inputs
from neoarm_shared_objective_audit import provenance_snapshot
from temgym_core.gaussian_image_likelihood import make_gaussian_detector_operator
from temgym_core.immersed_objective import canonical_to_mechanical
from temgym_core.neoarm_calibration import DetectorGeometry

GROUPED = np.array([0, 2, 1, 3])
STEPS = np.array([5e-9, 5e-9, 1e-3, 1e-3])
PIXEL_M = 55e-6


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class EndpointRows:
    """Two round canonical first rows, with fixed known sample shear.

    Per analyser: scaled radial A, scaled radial B, rotation. A/B are scaled
    into detector pixels for the declared basis inputs. This chart does not
    assert individual downstream lens geometry or a physical objective fit.
    """

    def __init__(self, kappa_per_m, steps=STEPS, pixel_m=PIXEL_M):
        self.steps = np.asarray(steps, float)
        if (self.steps.shape != (4,) or not np.isfinite(self.steps).all()
                or np.any(self.steps <= 0) or not np.isfinite(pixel_m) or pixel_m <= 0
                or not np.isfinite(kappa_per_m)):
            raise ValueError("Finite positive input scales/pitch and finite sample shear required.")
        if self.steps[0] != self.steps[1] or self.steps[2] != self.steps[3]:
            raise ValueError("This round calibration uses equal x/y position and tilt steps.")
        self.pixel_m = float(pixel_m)
        self.shear = mechanical_shear(float(kappa_per_m))

    def canonical_rows(self, parameters):
        p = jnp.asarray(parameters).reshape(2, 3)
        a = p[:, 0]*self.pixel_m/self.steps[0]
        b = p[:, 1]*self.pixel_m/self.steps[2]
        return rotation(p[:, 2])@jnp.concatenate(
            (a[:, None, None]*jnp.eye(2), b[:, None, None]*jnp.eye(2)), axis=-1)

    def response(self, parameters):
        return (self.canonical_rows(parameters)@self.shear)*self.steps/self.pixel_m

    def encode_response(self, response):
        """Projection for initialization only; final loss uses all response data."""
        response = np.asarray(response)
        if response.shape != (2, 2, 4) or not np.isfinite(response).all():
            raise ValueError("Two finite 2x4 signed detector responses required.")
        canonical = (response*self.pixel_m/self.steps)@np.linalg.inv(np.asarray(self.shear))
        result = []
        for row in canonical:
            z = [(.5*np.trace(block)+.5j*(block[1, 0]-block[0, 1]))
                 for block in (row[:, :2], row[:, 2:])]
            phase = np.angle(z[0] if abs(z[0])*self.steps[0] >= abs(z[1])*self.steps[2] else z[1])
            a, b = np.real(np.asarray(z)*np.exp(-1j*phase))
            result.extend((a*self.steps[0]/self.pixel_m, b*self.steps[2]/self.pixel_m, phase))
        return np.asarray(result)


def acquisition_inputs():
    """Nine training inputs, then two independent diagonal +/- check pairs."""
    training = np.asarray(signed_inputs(STEPS))
    directions = np.array([[STEPS[0], STEPS[1], 0., 0.],
                           [0., 0., STEPS[2], -STEPS[3]]])/np.sqrt(2)
    checks = np.stack((directions, -directions), axis=1).reshape(4, 4)
    return np.concatenate((training, checks)), directions


def check_responses(parameters, covariances):
    """Two signed responses; independent frames, no reused reference noise."""
    p, c = np.asarray(parameters), np.asarray(covariances)
    if p.shape != (4, 6) or c.shape != (4, 6, 6):
        raise ValueError("Two independent positive/negative six-coordinate image pairs required.")
    means = .5*(p[::2, :2]-p[1::2, :2])
    covariance = .25*(c[::2, :2, :2]+c[1::2, :2, :2])
    return means, covariance


def acquire(output, *, seed=1953701):
    if jax.default_backend() != "gpu" or not jax.config.x64_enabled:
        raise RuntimeError("Forward images and inference require GPU float64.")
    started = perf_counter()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _, _, _, aperture, model = load_context()
    split = model.column.objective.split_maps(jnp.asarray(model.fixed[-4:]))
    matrices = np.asarray(model.downstream@canonical_to_mechanical(split.sample_k_m_inv))
    kappa = -float(split.sample_k_m_inv)
    # The grouped shear helper uses canonical=mechanical+kappa*J2*x;
    # determine its sign by direct conversion rather than naming convention.
    inverse_conversion = np.linalg.inv(np.asarray(canonical_to_mechanical(split.sample_k_m_inv)))
    if not np.allclose(np.asarray(mechanical_shear(kappa))[np.ix_(GROUPED, GROUPED)],
                       inverse_conversion, rtol=1e-12, atol=1e-12):
        kappa = -kappa
    shear = np.asarray(mechanical_shear(kappa))[np.ix_(GROUPED, GROUPED)]
    if not np.allclose(shear, inverse_conversion, rtol=1e-12, atol=1e-12):
        raise RuntimeError("Canonical/mechanical sample convention did not close.")
    endpoints = EndpointRows(kappa)
    inputs, directions = acquisition_inputs()
    canonical_means = inputs[:, GROUPED]@shear.T
    source = pure_source()
    detector = DetectorGeometry(z_m=float(model.fixed[1]), mtf_sigma_px=.65)
    operator = jax.jit(make_gaussian_detector_operator(detector, 16))
    dose, read_noise = 2e6, .6
    rng = np.random.default_rng(seed)
    arrays = {k: [] for k in ("expected_counts", "noisy_counts", "fitted_counts",
        "fitted_variance", "image_parameters", "image_parameter_covariances")}
    rows, refinements = [], []
    sources, _ = provenance_snapshot()
    for path in (Path(__file__), FOLDER/"condenser_extension_images.json",
                 FOLDER/"aperture_image_fit.json", *[ROOT/p for p in aperture["report_input_hashes"]]):
        sources[str(path.relative_to(ROOT))] = digest(path)
    for analyser, matrix in enumerate(matrices):
        covariance = matrix[np.ix_(XY, np.arange(4))]@source.total_covariance@matrix[np.ix_(XY, np.arange(4))].T
        for frame, mean in enumerate(canonical_means):
            beam = MixedGaussianSource(source.position_covariance_xy_m2,
                source.curvature_xy_per_m, source.centre_covariance, mean)
            image, symplectic_error = render_pure_counts(beam, matrix, detector,
                pixel_order=16, incident_electrons=dose)
            image = np.asarray(image)
            reference = np.asarray(operator(matrix[XY]@mean, covariance, dose))
            noise_sigma = np.sqrt(reference+read_noise**2)
            error = float(np.linalg.norm((image-reference)/noise_sigma))
            if error >= .2:
                raise RuntimeError("Gaussian versus finite-pixel covariance reference failed.")
            observed = rng.poisson(image)+rng.normal(0., read_noise, image.shape)
            fit, variance, metadata = fit_data_only_image(observed, detector)
            if not metadata["accepted"] or image.sum()/dose < .99:
                raise RuntimeError("Calibration image fitting or detector capture failed.")
            for key, value in zip(arrays, (image, observed, fit.fitted_counts_yx,
                    variance, fit.parameters, fit.parameter_covariance)):
                arrays[key].append(np.asarray(value))
            rows.append(dict(analyser=analyser, frame=frame, training=frame < 9,
                capture=float(image.sum()/dose), gaussian_reference_error_l2_sigma=error,
                balanced_symplectic_error=symplectic_error, image_fit=metadata))
            if frame == 7:
                images = {16: image}
                for order in (8, 32):
                    images[order] = np.asarray(render_pure_counts(beam, matrix, detector,
                        pixel_order=order, incident_electrons=dose)[0])
                changes = [float(np.linalg.norm((images[b]-images[a])/noise_sigma))
                           for a, b in ((8, 16), (16, 32))]
                if max(changes) >= .2:
                    raise RuntimeError("Two successive calibration pixel refinements failed.")
                refinements.append(dict(analyser=analyser, frame=frame,
                    orders=[8, 16, 32], transition_l2_sigma=changes))
            print("endpoint calibration", analyser, frame, metadata["accepted"], flush=True)
    arrays = {key: np.asarray(value).reshape((2, 13)+np.asarray(value).shape[1:])
              for key, value in arrays.items()}
    pairs = [signed_response_from_image_fits(p[:9], c[:9]) for p, c in
             zip(arrays["image_parameters"], arrays["image_parameter_covariances"])]
    checks = [check_responses(p[9:], c[9:]) for p, c in
              zip(arrays["image_parameters"], arrays["image_parameter_covariances"])]
    response, response_covariance = map(np.asarray, zip(*pairs))
    truth_response = np.asarray(model.downstream)[:, XY][:, :, GROUPED]*STEPS/PIXEL_M
    truth_parameters = endpoints.encode_response(truth_response)
    closure = float(np.max(abs(np.asarray(endpoints.response(truth_parameters))-truth_response)))
    if closure > 1e-10:
        raise RuntimeError("The six-coordinate endpoint model does not close.")
    arrays.update(response=response, response_covariance=response_covariance,
        heldout_response=np.asarray([c[0] for c in checks]),
        heldout_covariance=np.asarray([c[1] for c in checks]),
        input_states_grouped=inputs, heldout_directions_grouped=directions,
        analyser_controls=np.asarray(model.analyser_controls))
    path = output.with_suffix(".npz")
    np.savez_compressed(path, **arrays)
    truth_path = output.with_name(output.stem+"_truth.json")
    truth_path.write_text(json.dumps(dict(endpoint_parameters=truth_parameters.tolist(),
        expected_response=truth_response.tolist(), canonical_maps=matrices.tolist(),
        conditional_sample_kappa_per_m=kappa), indent=2)+"\n")
    if any(digest(ROOT/name) != checksum for name, checksum in sources.items()):
        raise RuntimeError("A frozen acquisition dependency changed.")
    report = dict(schema="neoarm-same-column-endpoint-calibration-v1", backend=jax.default_backend(),
        source_hashes=sources, arrays_path=str(path.relative_to(ROOT)), arrays_sha256=digest(path),
        truth_path=str(truth_path.relative_to(ROOT)), truth_sha256=digest(truth_path),
        seed=seed, input_steps_grouped=STEPS.tolist(), detector_pixel_m=PIXEL_M,
        conditional_sample_kappa_per_m=kappa, analyser_controls=np.asarray(model.analyser_controls).tolist(),
        generated_frame_count=26, training_frame_count=18, heldout_frame_count=8,
        training_response_coordinate_count=16, heldout_response_coordinate_count=8,
        incident_electrons=dose, records=rows, refinements=refinements,
        six_coordinate_response_closure_pixels=closure, all_images_accepted=True,
        heldout_roles_fixed_before_noise=True, old_projector_posterior_used=False,
        image_weights_use_generating_expectation=False, elapsed_seconds=perf_counter()-started,
        limitations=["Two endpoint position rows, not individual projector recovery.",
            "Conditional shared objective prefield and sample shear; exact calibrated input/pixel/PSF scales.",
            "Clear-path calibrated sample-plane probe; internal source preparation is a separate measurement.",
            "Same synthetic column as condenser images, not the older independent-half objective.",
            "Hardware ranges, drift, hysteresis, clearance and dose-rate limits remain unverified."])
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    acquire(args.output)
