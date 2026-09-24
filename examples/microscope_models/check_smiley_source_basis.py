"""Check continuous source bundles against independent aperture/focus fields.

Run from the repository environment, preferably on GPU:
    python examples/microscope_models/check_smiley_source_basis.py --counts 128 1000

All candidate bundles start at the source and traverse each native component.
Choosing a sampling plane only constructs the source basis; no candidate is
re-decomposed during propagation. References multiply the exact incident field
by the sigmoid at the aperture and use a radial Fresnel integral at the specimen.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import j0, roots_legendre

from smiley_defocus import SmileyDefocus
from temgym_core.components import Plane, SigmoidAperture
from temgym_core.evaluate import evaluate_gaussians
from temgym_core.run import column_matrix, run_to_end
from temgym_core.source import decompose_gaussian


def relative_error(actual, expected, weights=None):
    if weights is None:
        weights = 1.
    value = np.sqrt(np.sum(weights*np.abs(actual-expected)**2)
                    / np.sum(weights*np.abs(expected)**2))
    return float(value) if np.isfinite(value) else None


def focus_reference(demo, aperture, incident_at_aperture, after_aperture):
    """Independent radial Fresnel quadrature; no beamlet/aperture Taylor action."""
    matrix = np.asarray(column_matrix(demo.downstream, source_z=demo.z_sample,
                                      voltage=demo.model.voltage))
    coords = np.asarray(demo.detector.coords) @ np.linalg.inv(matrix[:2, :2]).T
    grid = SimpleNamespace(coords=jnp.asarray(coords), shape=demo.detector.shape)
    field = np.asarray(evaluate_gaussians(demo.nominal_illumination, grid, method="scan"))
    if aperture is None:
        ratio = np.ones(coords.shape[0], dtype=complex)
        quadrature_error = 0.
    else:
        transfer = np.asarray(column_matrix(after_aperture, source_z=aperture.z,
                                           voltage=demo.model.voltage))
        # Round, centred condenser: A=aR and B=bR. Absorb the sign into R.
        b = np.linalg.norm(transfer[:2, 2])
        rotation = transfer[:2, 2:4]/b
        a = np.trace(rotation.T @ transfer[:2, :2])/2
        np.testing.assert_allclose(transfer[:2, :2], a*rotation, atol=1e-12)
        q = complex(np.asarray(incident_at_aperture.Q_inv)[0, 0])
        k = float(incident_at_aperture.k)
        width = np.sqrt(2/(k*q.imag))
        radius = np.linspace(0., np.linalg.norm(coords, axis=1).max(), 1024)
        p = -.5j*k*(q+a/b)
        u = k*radius/b
        # Integral of rho*exp(-p*rho^2)*J0(u*rho) on [0, infinity).
        unmasked_integral = np.exp(-u*u/(4*p))/(2*p)
        ratios = []
        for order in (1024, 2048, 4096, 8192, 16384, 32768):
            nodes, weights = roots_legendre(order)
            rho = (nodes+1)*2.5*width
            weights = weights*2.5*width
            points = jnp.asarray(np.column_stack((rho, np.zeros_like(rho))))
            transmission = np.exp(np.asarray(aperture.log_transmission(points)))
            integrand = weights*rho*np.exp(-p*rho*rho)
            kernel = j0(u[:, None]*rho[None, :])
            # Also verify that this quadrature resolves the unmasked Gaussian.
            if not np.allclose(kernel @ integrand, unmasked_integral,
                               rtol=2e-7, atol=abs(unmasked_integral[0])*1e-9):
                continue
            ratios.append((kernel @ (integrand*transmission))/unmasked_integral)
            if len(ratios) >= 2 and relative_error(ratios[-2], ratios[-1]) < 1e-6:
                break
        if len(ratios) < 2 or relative_error(ratios[-2], ratios[-1]) >= 1e-6:
            raise RuntimeError("Fresnel reference has not converged; increase quadrature order.")
        quadrature_error = relative_error(ratios[-2], ratios[-1])
        ratio = np.interp(np.linalg.norm(coords, axis=1), radius, ratios[-1])
    actions = np.asarray(demo.sample.evaluate_actions(jnp.asarray(coords)))
    intensity = np.abs(field)**2 * (np.abs(ratio)**2*np.exp(2*actions[:, 1])).reshape(grid.shape)
    return intensity, quadrature_error


def check(*, counts=(128, 1000), bases=("source", "aperture", "specimen"),
          aperture_um=10., spot=3.):
    demo = SmileyDefocus((1000, 1000))
    if spot != 3.:
        demo = demo.with_settings(spot_size=spot)
    z_aperture = demo.model.config["apertures"]["C_aperture"]["z_m"]
    before = tuple(c for c in demo.upstream if float(c.z) < z_aperture) + (Plane(z=z_aperture),)
    after = tuple(c for c in demo.upstream if float(c.z) > z_aperture) + (Plane(z=demo.z_sample),)
    to_sample = demo.upstream + (Plane(z=demo.z_sample),)
    aperture = (SigmoidAperture(z=z_aperture, radius=aperture_um*1e-6,
                               edge_width=aperture_um*1e-6*.05)
                if aperture_um else None)
    at_aperture = run_to_end(demo.source, before, propagator=demo.propagator)
    width_ap = np.sqrt(2/(float(at_aperture.k)*complex(at_aperture.Q_inv[0, 0]).imag))
    radius = np.linspace(0., 4*width_ap, 256)
    points = np.concatenate([np.column_stack((radius*np.cos(theta), radius*np.sin(theta)))
                             for theta in (0., np.pi/4)])
    grid_ap = SimpleNamespace(coords=jnp.asarray(points), shape=(len(points),))
    aperture_reference = np.asarray(evaluate_gaussians(at_aperture, grid_ap, method="scan"))
    transmitted_reference = aperture_reference.copy()
    if aperture is not None:
        transmitted_reference *= np.exp(np.asarray(aperture.log_transmission(grid_ap.coords)))
    reference, reference_error = focus_reference(demo, aperture, at_aperture, after)
    weights = np.tile(radius, 2)
    result = dict(spot=spot, aperture_radius_um=aperture_um or None,
                  magnification=demo.magnification, device=str(jax.devices()[0]),
                  reference_quadrature_relative_error=reference_error, cases=[])

    @jax.jit
    def propagate(source):
        incoming = run_to_end(source, before, propagator=demo.propagator)
        transmitted = incoming if aperture is None else aperture(incoming)
        specimen = run_to_end(transmitted, after, propagator=demo.propagator)
        detector = run_to_end(specimen, (demo.sample, *demo.objective_column()),
                              propagator=demo.propagator)
        intensity = jnp.abs(evaluate_gaussians(detector, demo.detector,
                            method=demo.method, tile_beams=32)*demo.magnification)**2
        field_before = evaluate_gaussians(incoming, grid_ap, method="scan", tile_beams=32)
        field_after = evaluate_gaussians(transmitted, grid_ap, method="scan", tile_beams=32)
        widths = [jnp.median(jnp.sqrt(2/(b.k*jnp.imag(b.Q_inv[:, 0, 0]))))
                  for b in (incoming, specimen)]
        return intensity, field_before, field_after, jnp.array(widths)

    for count in counts:
        for basis in bases:
            through = {"source": (), "aperture": before, "specimen": to_sample,
                       "fixed_aperture": before, "fixed_specimen": to_sample}[basis]
            overlap = 1.1
            if basis.startswith("fixed_"):
                # Refine the lattice without shrinking the physical packets.
                target = run_to_end(demo.source, through, propagator=demo.propagator)
                width = np.sqrt(2/(float(target.k)*complex(target.Q_inv[0, 0]).imag))
                packet_radius = 250e-9 if basis == "fixed_aperture" else 1e-9
                overlap = packet_radius/(5*width/count)
            print(f"Starting {basis}, {count**2:,} beams, aperture {aperture_um:g} µm", flush=True)
            start = perf_counter()
            source = decompose_gaussian(demo.source, num_across=count, through=through,
                                        extent_rms=5., overlap=overlap)
            image, f_before, f_after, widths = map(np.asarray, propagate(source))
            case = dict(basis=basis, count=count**2, overlap=overlap,
                        aperture_packet_radius_um=float(widths[0]*1e6),
                        specimen_packet_radius_nm=float(widths[1]*1e9),
                        incident_field_error=relative_error(f_before, aperture_reference, weights),
                        transmitted_field_error=relative_error(f_after, transmitted_reference, weights),
                        focused_intensity_error=relative_error(image, reference),
                        seconds=perf_counter()-start)
            result["cases"].append(case)
            print(json.dumps(case, allow_nan=False), flush=True)
    # Current fallback, compared to the same independent focus reference.
    fallback = demo.with_settings(aperture_radius_m=aperture.radius if aperture else None).with_propagation_mode("recompose")
    fallback_image = fallback.image(0.)[0]
    result["recomposed_focused_intensity_error"] = relative_error(fallback_image, reference)
    print("Recomposed focus error:", result["recomposed_focused_intensity_error"], flush=True)
    return result


def check_window(*, spots=(1., 3., 8.), count=1000):
    """Validate the actual finite-window source basis and moving-specimen UI."""
    base = SmileyDefocus((count, count))
    records = []
    for spot in spots:
        source = base.with_settings(spot_size=spot, aperture_radius_m=base.aperture_options[0])
        recomposed = source.with_propagation_mode("recompose")
        z = source.aperture.z
        before = tuple(c for c in source.upstream if float(c.z) < z) + (Plane(z=z),)
        after = tuple(c for c in source.upstream if float(c.z) > z) + (Plane(z=source.z_sample),)
        at_aperture = run_to_end(source.source, before, propagator=source.propagator)
        reference, quad_error = focus_reference(source, source.aperture, at_aperture, after)
        for height in (0., -10000., 10000.):
            image = source.image(height)[0]
            fallback = recomposed.image(height)[0]
            row = dict(spot=spot, height_nm=height, count=source.num_beams,
                       continuous_vs_recomposed_error=relative_error(image, fallback))
            if height == 0.:
                row.update(continuous_focus_error=relative_error(image, reference),
                           recomposed_focus_error=relative_error(fallback, reference),
                           reference_quadrature_error=quad_error)
            print(json.dumps(row, allow_nan=False), flush=True)
            records.append(row)
    return dict(aperture_radius_um=10., magnification=base.magnification,
                source_basis_window_nm=500., cases=records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-check", action="store_true", help="Check the actual source pipeline across spots and heights")
    parser.add_argument("--spots", nargs="+", type=float, default=[1., 3., 8.])
    parser.add_argument("--counts", nargs="+", type=int, default=[128, 1000],
                        help="Number across each side; 1000 means one million beams")
    parser.add_argument("--bases", nargs="+", default=["source", "aperture", "specimen"],
                        choices=["source", "aperture", "specimen", "fixed_aperture", "fixed_specimen"])
    parser.add_argument("--aperture-um", type=float, default=10., help="Radius; zero withdraws the aperture")
    parser.add_argument("--spot", type=float, default=3.)
    parser.add_argument("--output", type=Path, default=Path("/tmp/smiley_source_basis_check.json"))
    args = parser.parse_args()
    data = (check_window(spots=args.spots, count=max(args.counts)) if args.window_check else
            check(counts=args.counts, bases=args.bases, aperture_um=args.aperture_um, spot=args.spot))
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
