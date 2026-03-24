from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "No TOML library found. On Python < 3.11, install tomli: "
            "pip install tomli"
        ) from exc


def load_microscope_config(path: str | Path) -> dict:
    """Load a microscope configuration from a TOML file.

    Parameters
    ----------
    path : str or Path
        Path to the ``.toml`` configuration file.

    Returns
    -------
    dict
        Parsed configuration with top-level keys ``beam``, ``microscope``,
        typed component tables (for example ``lenses``/``deflectors``/``apertures``),
        and ``modes``.

        Schema conventions:
        - ``microscope.components`` provides the global ``order`` and typed groups.
        - Each lens entry in ``lenses`` contains absolute ``z_m`` plus optics.
        - Each mode defines ``current_lenses`` ordering for current vectors.
        - Each mode point defines only ``control_value`` and ``currents_A``.
    """
    with open(Path(path), "rb") as f:
        cfg = tomllib.load(f)

    components = cfg.get("microscope", {}).get("components", {})
    component_order = components.get("order", [])
    lens_names = components.get("lenses", [])
    deflector_names = components.get("deflectors", [])
    aperture_names = components.get("apertures", [])

    if not component_order:
        raise ValueError("Config is missing microscope.components.order.")
    if not lens_names:
        raise ValueError("Config is missing microscope.components.lenses.")

    tables = {
        "lenses": cfg.get("lenses", {}),
        "deflectors": cfg.get("deflectors", {}),
        "apertures": cfg.get("apertures", {}),
    }

    missing_lenses = [name for name in lens_names if name not in tables["lenses"]]
    if missing_lenses:
        raise ValueError(f"Lens definitions missing for: {missing_lenses}")

    missing_deflectors = [name for name in deflector_names if name not in tables["deflectors"]]
    if missing_deflectors:
        raise ValueError(f"Deflector definitions missing for: {missing_deflectors}")

    missing_apertures = [name for name in aperture_names if name not in tables["apertures"]]
    if missing_apertures:
        raise ValueError(f"Aperture definitions missing for: {missing_apertures}")

    defined_components = set(lens_names) | set(deflector_names) | set(aperture_names)
    missing_in_order = [name for name in defined_components if name not in component_order]
    if missing_in_order:
        raise ValueError(
            f"Components listed under microscope.components are missing from order: {missing_in_order}"
        )

    modes = cfg.get("modes", {})
    for mode_name, mode_data in modes.items():
        current_lenses = mode_data.get("current_lenses", lens_names)
        unknown = [name for name in current_lenses if name not in lens_names]
        if unknown:
            raise ValueError(
                f"modes.{mode_name}.current_lenses contains non-lens names: {unknown}"
            )

        n_lenses = len(current_lenses)
        points = mode_data.get("points", [])
        for i, point in enumerate(points):
            currents = point.get("currents_A", [])
            if len(currents) != n_lenses:
                raise ValueError(
                    f"modes.{mode_name}.points[{i}] has {len(currents)} currents; "
                    f"expected {n_lenses} from modes.{mode_name}.current_lenses."
                )

    return cfg
