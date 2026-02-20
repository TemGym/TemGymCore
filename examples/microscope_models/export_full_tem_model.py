"""
Export a full TEM column model (JEOL ARM-200F).

This script:
1. Runs the IL system solver (~ 1-2 min with differential_evolution).
2. Loads the condenser CL3-only fit JSON.
3. Assembles both halves into a TEMModel.
4. Exports to ``full_tem_model.json``.

Run this once, then load the JSON in the interactive notebook.
"""
# %%
from pathlib import Path
import numpy as np

from temgym_core.tem_model import (
    build_illumination_system,
    build_projection_system,
    solve_il_system,
    TEMModel,
    export_tem_model_json,
)

# %% [markdown]
# ## 1. Solve the IL system (Mode C, 30k–600k)

# %%
print("Solving IL system (differential_evolution + L-BFGS-B) …")
il = solve_il_system(
    M_obj=60.0,
    M_proj=100.0,
    mode_filter="C",
    d_il1_to_il2=52.083e-3,
    d_il2_to_il3=62.500e-3,
    de_seed=42,
)
print(f"  Success: {il['success']}, Loss: {il['loss']:.6e}")
print(f"  d_obj→IL1 = {il['d_obj_to_il1']*1e3:.3f} mm")
print(f"  d_IL1→IL2 = {il['d_il1_to_il2']*1e3:.3f} mm")
print(f"  d_IL2→IL3 = {il['d_il2_to_il3']*1e3:.3f} mm")
print(f"  d_IL3→IM  = {il['d_il3_to_im']*1e3:.3f} mm")
for i, name in enumerate(("IL1", "IL2", "IL3")):
    print(f"  {name}: Gc={il['gc'][i]:.3f}, α={il['alpha_nl'][i]:.3f}")

# %% [markdown]
# ## 2. Build the illumination (condenser) system

# %%
condenser_json = Path(__file__).parent / "examples" / "condenser_cl3_only_fit_model.json"
if not condenser_json.exists():
    # Try an alternate location
    condenser_json = Path(__file__).parent.parent / "microscope_models" / "examples" / "condenser_cl3_only_fit_model.json"
print(f"Loading condenser from: {condenser_json}")

illum, sample_z = build_illumination_system(condenser_json)
print(f"  Illumination lenses: {illum.geometry.lens_names}")
print(f"  Brightness steps: {illum.dial_curves['brightness'].control_values}")
print(f"  Sample z = {sample_z*1e3:.3f} mm")

# %% [markdown]
# ## 3. Build the projection (IL + projector) system

# %%
proj = build_projection_system(
    il,
    d_sample_to_opl_post=2.083e-3,
    f_opl_post=5.0e-3,
    f_pl1=3.25e-3,
    sample_z=sample_z,
)
print(f"  Projection lenses: {proj.geometry.lens_names}")
mag_curve = proj.dial_curves["magnification"]
print(f"  Magnification steps: {mag_curve.control_values}")

# %% [markdown]
# ## 4. Assemble and export the TEMModel

# %%
model = TEMModel(
    model_type="jeol_arm200f_mode_c",
    schema_version=1,
    voltage_v=200e3,
    illumination=illum,
    projection=proj,
    sample_z_m=sample_z,
)

out_path = Path(__file__).parent / "full_tem_model.json"
export_tem_model_json(model, out_path)
print(f"\nExported → {out_path}  ({out_path.stat().st_size:,} bytes)")

# %% [markdown]
# ## 5. Quick sanity check — round-trip load

# %%
from temgym_core.tem_model import load_tem_model_json

loaded = load_tem_model_json(out_path)
state = loaded.realize_full(brightness_nm=250, magnification=100_000)
print("\nRound-trip verification:")
print(f"  Illumination DAC hex: {state['illumination']['dac_hex']}")
print(f"  Projection  DAC hex: {state['projection']['dac_hex']}")
print("Done ✓")
