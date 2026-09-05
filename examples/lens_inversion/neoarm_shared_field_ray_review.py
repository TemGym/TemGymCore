"""Verified two-ray figures for the full, overlapping objective-field model.

The saved optical states are model hypotheses with unchanged controls. A
sample-conditioned circular Gaussian illustrates propagation; it is distinct
from the mixed condenser illumination used in the image experiments.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tomllib

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from temgym_core.axial_field_column import objective_tail_field
from temgym_core.axial_field_profile import (
    axial_field_prefix, prefix_map_to_plane, sample_conditioned_basis,
)
from temgym_core.constants import energy2wavelength
from temgym_core.immersed_objective import SharedObjectiveField, mechanical_to_canonical
from temgym_core.neoarm_calibration import hard_edge_column_from_config
from temgym_core.plotting import (
    ColumnElement, ColumnSection, PositionAngleTrace, export_ray_figure,
    plot_neoarm_ray_diagram, plot_neoarm_ray_small_multiples,
)
from neoarm_imaging_recipe import column_with_geometry_anchors
from neoarm_measured_ray_review import annotate_verified_events, label_overlap_audit
from neoarm_objective_tail_overlap import TailOverlapColumn
from neoarm_shared_objective import SharedObjectiveColumn, shared_imaging_metrics


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT/"examples/generated/neoarm_inversion/imaging_recipe/objective_tail_overlap.json"


def verified_trace(model, parameters, controls, *, gaussian=True, label="", steps_note=None):
    """Check rays against single-ray Jacobians and the circular covariance."""
    p, u = jnp.asarray(parameters), jnp.asarray(controls)
    edges, rates = model.geometry_and_rates(p, u)
    prefix = jax.jit(lambda: axial_field_prefix(model.mesh,p[-4:],edges,rates))()
    sample_z = model.base.sample_z_m
    waist = 60e-9 if gaussian else 5e-9
    divergence = float(energy2wavelength(200000.))/(np.pi*waist) if gaussian else 1e-3
    basis = jnp.zeros((4,2)).at[0,0].set(waist).at[1,1].set(divergence)
    sample_k = objective_tail_field(sample_z,p[-4:],model.mesh.normalization_field)
    canonical_basis = basis if gaussian else mechanical_to_canonical(sample_k)@basis
    exact = jax.jit(lambda z: sample_conditioned_basis(model.mesh,p[-4:],prefix,
        sample_z,canonical_basis,z)[0])
    z = np.asarray(prefix.z_m)
    values = np.asarray(jax.jit(jax.vmap(exact))(jnp.asarray(z)))
    radial = jax.jit(lambda z: exact(z)[0])
    events, roots = [], []
    for channel, kind in enumerate(("crossovers","conjugates")):
        @jax.jit
        def refine(left,right):
            def step(_, bounds):
                a,b = bounds
                m = .5*(a+b)
                same = radial(a)[channel]*radial(m)[channel]>0
                return jnp.where(same,m,a),jnp.where(same,b,m)
            a,b = jax.lax.fori_loop(0,48,step,(left,right))
            return .5*(a+b)
        sign_changes = np.flatnonzero(values[:-1,0,channel]*values[1:,0,channel]<0)
        for i in sign_changes:
            root = float(refine(z[i],z[i+1]))
            roots.append(root)
            events.append(dict(kind=kind,z_m=root,position_residual_m=float(radial(root)[channel])))
        # The sample-defined angle solution vanishes exactly at the specimen.
        if channel == 1 and not any(abs(sample_z-r)<1e-9 for r in roots):
            roots.append(sample_z)
            events.append(dict(kind=kind,z_m=sample_z,position_residual_m=float(radial(sample_z)[channel])))
    waists = []
    if gaussian:
        derivative = jax.jit(lambda value: jnp.dot(radial(value),jax.jacfwd(radial)(value)))
        derivatives = np.asarray(jax.jit(jax.vmap(derivative))(jnp.asarray(z)))
        @jax.jit
        def refine_waist(left,right):
            def step(_,bounds):
                a,b = bounds
                m = .5*(a+b)
                negative = derivative(m)<0
                return jnp.where(negative,m,a),jnp.where(negative,b,m)
            a,b = jax.lax.fori_loop(0,48,step,(left,right))
            return .5*(a+b)
        for i in np.flatnonzero((derivatives[:-1]<0)&(derivatives[1:]>=0)):
            root = float(refine_waist(z[i],z[i+1]))
            if any(abs(root-row["exact_stationary_z_m"])<1e-9 for row in waists):
                continue
            roots.append(root)
            waists.append(dict(stationary_bracketed=True,exact_stationary_z_m=root,
                annotation_radius_m=float(jnp.linalg.norm(radial(root))),
                stationary_derivative_m=float(derivative(root))))
    z = np.unique(np.r_[z,roots])
    values = np.asarray(jax.jit(jax.vmap(exact))(jnp.asarray(z)))
    radius = np.linalg.norm(values[:,0],axis=1)
    trace = PositionAngleTrace(z=z,centroid=np.zeros(z.size),position=values[:,0,0],
        angle=values[:,0,1],radius=radius,coordinate="x_rot",position_scale=waist,
        angle_scale=divergence,label=label,voltage=200000.)
    # A separate single-ray derivative supplies the endpoint check. Its sample
    # and detector planes use mechanical coordinates only for calibrated kicks.
    coordinates = "canonical" if gaussian else "mechanical"
    downstream = jax.jit(lambda: model.matrix(p,u,segment="sample_to_detector",coordinates=coordinates))()
    predicted = np.asarray(downstream@basis)[[0,2]]
    sample_map, sample_angle = prefix_map_to_plane(model.mesh,p[-4:],prefix,sample_z)
    _, detector_angle = prefix_map_to_plane(model.mesh,p[-4:],prefix,p[1])
    phi = float(detector_angle-sample_angle)
    axes = np.array([[np.cos(phi),np.sin(phi)],[-np.sin(phi),np.cos(phi)]])
    detector_error = values[-1]-axes@predicted
    relative_endpoint_error = float(np.linalg.norm(detector_error)/np.linalg.norm(predicted))
    if relative_endpoint_error>2e-8:
        raise RuntimeError(f"Prefix rays disagree with the single-ray Jacobian: {relative_endpoint_error}")
    covariance_error = None
    if gaussian:
        # For intensity radius w, each circular spatial variance is w²/4.
        covariance = jnp.diag(jnp.array([waist**2,divergence**2,waist**2,divergence**2])/4)
        inv_sample = jnp.linalg.inv(sample_map)
        def covariance_radius(z):
            matrix,_ = prefix_map_to_plane(model.mesh,p[-4:],prefix,z)
            transport = matrix@inv_sample
            spatial = transport@covariance@transport.T
            return jnp.sqrt(2*(spatial[0,0]+spatial[2,2]))
        checked = np.asarray(jax.jit(jax.vmap(covariance_radius))(jnp.asarray(z)))
        covariance_error = float(np.max(np.abs(checked-radius)/np.maximum(radius,1e-20)))
        if covariance_error>2e-8:
            raise RuntimeError("Two-ray envelope disagrees with propagated covariance.")
    evidence = dict(label=label,gaussian=gaussian,steps_per_segment=steps_note,
        position_scale_m=waist,angle_scale_rad=divergence,events=events,waists=waists,
        detector_basis_relative_error=relative_endpoint_error,
        gaussian_covariance_radius_relative_error=covariance_error,
        orthogonal_basis_max_m=np.max(np.abs(values[:,1]),axis=0).tolist(),
        sample_k_m_inv=float(sample_k),source_to_detector_larmor_rad=float(prefix.larmor_radians[-1]),
        sample_canonical_basis=np.asarray(canonical_basis).tolist(),
        downstream_mechanical_metrics=np.asarray(shared_imaging_metrics(
            model.matrix(p,u,segment="sample_to_detector"))).tolist())
    return trace,evidence,values[:,1],prefix


def physical_elements(model, parameters):
    """Finite controlled bodies and a labelled FWHM guide for the smooth field."""
    p = jnp.asarray(parameters)
    elements = [ColumnElement(float(p[0]),"Source reference plane","source")]
    for lens, centre, length, *_ in model.base._decoded_lenses(p):
        elements.append(ColumnElement(float(centre-length/2),lens.name,"lens",float(centre+length/2)))
    peak,width,asymmetry = map(float,(p[-3],jnp.exp(p[-2]),p[-1]))
    def half_width(sign):
        lo,hi = 0.,5.
        for _ in range(60):
            middle = .5*(lo+hi)
            if middle*np.exp(-asymmetry*np.tanh(sign*middle))<1:
                lo = middle
            else:
                hi = middle
        return sign*.5*(lo+hi)*width+peak
    elements.extend((ColumnElement(half_width(-1),"Objective · FWHM guide","lens",half_width(1)),
        ColumnElement(model.base.sample_z_m,"MAGICAL · 23 nm multilayer","sample"),
        ColumnElement(float(p[1]),"Medipix · 256 × 256","detector")))
    sections = (ColumnSection("Condenser · sample-conditioned basis",float(p[0]),-.03),
        ColumnSection("Objective / sample · continuous field tails",-.025,.025),
        ColumnSection("Projector · unchanged model controls",.15,float(p[1])))
    return tuple(elements),sections


def draw_larmor(axis,prefix):
    for child in list(axis.child_axes):
        child.remove()
    axis.clear()
    axis.plot(np.asarray(prefix.z_m),np.rad2deg(np.asarray(prefix.larmor_radians)),color="#7B52AB",lw=1.6)
    axis.set(xlabel="z (m)",ylabel="Rotation (°)",xlim=(float(prefix.z_m[0]),float(prefix.z_m[-1])))
    axis.set_title("Cumulative Larmor rotation · summed continuous field",fontsize=9,loc="left")
    axis.grid(alpha=.2)
    axis.tick_params(labelsize=7)
    axis.secondary_yaxis("right",functions=(np.deg2rad,np.rad2deg)).set_ylabel("rad",fontsize=8)


def run(output_dir, *, steps=512):
    if jax.default_backend()!="gpu" or not jax.config.x64_enabled:
        raise RuntimeError("GPU float64 is required for this forward verification.")
    saved = json.loads(INPUT.read_text())
    hashes = dict(saved["source_hashes"])
    for name,digest in hashes.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f"Stale field-model input: {name}")
    for path in (INPUT,Path(__file__),ROOT/"src/temgym_core/axial_field_profile.py",
                 ROOT/"src/temgym_core/plotting.py",ROOT/"examples/lens_inversion/neoarm_measured_ray_review.py"):
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    p = np.asarray(saved["fixed_parameters"])
    field = SharedObjectiveField.from_window(-.05,0.,.05,n_slices_per_side=8192)
    config = tomllib.loads((ROOT/"examples/lens_inversion/neoarm_inversion_config.toml").read_text())
    measured = column_with_geometry_anchors(hard_edge_column_from_config(config),
        ROOT/"examples/microscope_models/measure_microscope_geometry.ipynb")
    base = SharedObjectiveColumn.from_hard_edge(measured,objective=field,objective_initial=p[-4:])
    model = TailOverlapColumn.from_shared(base,p,steps_per_segment=steps)
    nominal = np.asarray(saved["fixed_controls"][:2])
    minus,plus = nominal[0].copy(),nominal[0].copy()
    minus[5]-=.02/p[25];plus[5]+=.02/p[25]
    states = {"nominal_40k":nominal[0],"nominal_80k":nominal[1],"IL3_minus":minus,"IL3_plus":plus}
    traces,records,prefixes,arrays = {},{},{},{}
    for name,controls in states.items():
        trace,evidence,orthogonal,prefix = verified_trace(model,p,controls,label=name,steps_note=steps)
        traces[name],records[name],prefixes[name] = trace,evidence,prefix
        arrays.update({f"{name}_z_m":trace.z,f"{name}_height_m":trace.position,
            f"{name}_angle_m":trace.angle,f"{name}_radius_m":trace.radius,
            f"{name}_orthogonal_m":orthogonal,f"{name}_larmor_z_m":np.asarray(prefix.z_m),
            f"{name}_larmor_rad":np.asarray(prefix.larmor_radians)})
        print(name,"endpoint error",evidence["detector_basis_relative_error"],flush=True)
    refinements = []
    # The same nominal controls and canonical sample radius are retained while
    # refining transport. Comparison uses both rays, not interpolated envelopes.
    previous = None
    query = jnp.asarray(traces["nominal_40k"].z)
    for n in (steps//4,steps//2,steps):
        refined = TailOverlapColumn.from_shared(base,p,steps_per_segment=n)
        edges,rates = refined.geometry_and_rates(p,nominal[0])
        prefix = jax.jit(lambda: axial_field_prefix(refined.mesh,jnp.asarray(p[-4:]),edges,rates))()
        basis = jnp.asarray(records["nominal_40k"]["sample_canonical_basis"])
        values = np.asarray(jax.jit(jax.vmap(lambda z:sample_conditioned_basis(
            refined.mesh,jnp.asarray(p[-4:]),prefix,base.sample_z_m,basis,z)[0]))(query))
        if previous is not None:
            scale = np.maximum(np.linalg.norm(values[:,0],axis=1),1e-20)
            error = float(np.max(np.linalg.norm(values[:,0]-previous[:,0],axis=1)/scale))
            refinements.append(dict(from_steps=n//2,to_steps=n,max_local_radius_relative_ray_change=error))
        previous = values
    elements,sections = physical_elements(model,p)
    output_dir.mkdir(parents=True,exist_ok=True)
    metadata = dict(schema="neoarm-shared-tail-ray-review-v1",backend=jax.default_backend(),
        source_hashes=hashes,parameters=p.tolist(),controls={k:v.tolist() for k,v in states.items()},
        field_profile_measured=False,optical_parameters_fitted=False,
        objective_glyph="FWHM interval, not a physical polepiece or field boundary",
        gaussian_definition="Circular canonical sample reference radius w=60 nm with flat radial wavefront; w²=x_h²+x_angle². In-field reference plane need not be a radius minimum.",
        calibrated_definition="Mechanical sample inputs of 5 nm and 1 mrad, converted at nonzero Bz",
        scope="Unrefocused full-tail model predictions at previously nominal 40k/80k controls",
        true_z=True,records=records,successive_refinements=refinements,figures={})
    for name,trace in traces.items():
        actual_m = records[name]["downstream_mechanical_metrics"][0]
        fig,axes = plot_neoarm_ray_diagram((),trace,extra_elements=elements,sections=sections,annotate_events=False,
            title=f"NEOARM shared-field hypothesis · {actual_m/1000:.2f}k × · {name.replace('_',' ')}")
        draw_larmor(axes["larmor"],prefixes[name])
        annotate_verified_events(axes,sections,records[name],gaussian=True)
        collisions = label_overlap_audit(fig)
        export_ray_figure(fig,output_dir/name,metadata={**metadata,"state":name,"label_collisions":collisions})
        metadata["figures"][name] = dict(stem=str((output_dir/name).relative_to(ROOT)),label_collisions=collisions)
        plt.close(fig)
    comparison = {"− IL3 wobble":traces["IL3_minus"],"Nominal":traces["nominal_40k"],"+ IL3 wobble":traces["IL3_plus"]}
    fig,axes = plot_neoarm_ray_small_multiples((),comparison,extra_elements=elements,sections=sections,
        title="Shared-field model · IL3 ±0.02 rad · common physical extents")
    collisions = label_overlap_audit(fig)
    export_ray_figure(fig,output_dir/"IL3_comparison",metadata={**metadata,"label_collisions":collisions})
    metadata["figures"]["IL3_comparison"] = dict(stem=str((output_dir/"IL3_comparison").relative_to(ROOT)),label_collisions=collisions)
    plt.close(fig)
    trace,evidence,orthogonal,prefix = verified_trace(model,p,nominal[0],gaussian=False,
        label="5 nm position / 1 mrad tilt",steps_note=steps)
    records["calibrated_40k"] = evidence
    arrays.update(calibrated_z_m=trace.z,calibrated_height_x_m=trace.position,
        calibrated_angle_x_m=trace.angle,calibrated_height_angle_y_m=orthogonal)
    fig,axes = plot_neoarm_ray_diagram((),trace,extra_elements=elements,sections=sections,annotate_events=False,
        trace_semantics="calibrated_response",title="Calibrated mechanical inputs · co-rotating x response")
    draw_larmor(axes["larmor"],prefix)
    annotate_verified_events(axes,sections,evidence,gaussian=False)
    collisions = label_overlap_audit(fig)
    export_ray_figure(fig,output_dir/"calibrated_40k",metadata={**metadata,"label_collisions":collisions})
    metadata["figures"]["calibrated_40k"] = dict(stem=str((output_dir/"calibrated_40k").relative_to(ROOT)),label_collisions=collisions)
    plt.close(fig)
    np.savez_compressed(output_dir/"ray_profiles.npz",**arrays)
    metadata["array_file"] = str((output_dir/"ray_profiles.npz").relative_to(ROOT))
    metadata["array_sha256"] = hashlib.sha256((output_dir/"ray_profiles.npz").read_bytes()).hexdigest()
    for record in metadata["figures"].values():
        stem = ROOT/record["stem"]
        record["file_hashes"] = {str(stem.with_suffix(suffix).relative_to(ROOT)):
            hashlib.sha256(stem.with_suffix(suffix).read_bytes()).hexdigest()
            for suffix in (".png",".svg",".pdf",".metadata.json")}
    for name,digest in hashes.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f"Source changed during figure generation: {name}")
    (output_dir/"ray_review.json").write_text(json.dumps(metadata,indent=2,allow_nan=False)+"\n")
    print("Saved",output_dir/"ray_review.json",flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--steps",type=int,default=512)
    args = parser.parse_args()
    if args.steps<16 or args.steps%4:
        parser.error("--steps must be a multiple of four, at least16")
    jax.config.update("jax_enable_x64",True)
    run(args.output_dir.resolve(),steps=args.steps)
