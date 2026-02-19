from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

from .components import ElectromagneticLens, Plane
from .constants import compute_Rc_from_voltage
from .transfer_matrices import lens_matrix, propagation_matrix


DEFAULT_PROJECTOR_LENS_NAMES = ("IL1", "IL2", "IL3", "PL1")


@dataclass
class ProjectorGeometry:
    """
    Fixed projector stack geometry from object plane to detector plane.

    Distances are drifts between consecutive planes/lenses:

    object -> OPL -> IL1 -> IL2 -> IL3 -> PL1 -> detector
    """

    d_object_to_opl_m: float = 0.010
    d_opl_to_il1_m: float = 0.050
    d_il1_to_il2_m: float = 0.030
    d_il2_to_il3_m: float = 0.025
    d_il3_to_pl1_m: float = 0.040
    d_pl1_to_detector_m: float = 0.200
    f_opl_m: float = 5.0e-3
    opl_i0_at: float = 3000.0

    def drifts_m(self) -> np.ndarray:
        return np.array(
            [
                self.d_object_to_opl_m,
                self.d_opl_to_il1_m,
                self.d_il1_to_il2_m,
                self.d_il2_to_il3_m,
                self.d_il3_to_pl1_m,
                self.d_pl1_to_detector_m,
            ],
            dtype=float,
        )

    def total_length_m(self) -> float:
        return float(np.sum(self.drifts_m()))


@dataclass
class HexProjectorDataset:
    magnification: np.ndarray
    log_magnification: np.ndarray
    lens_names: tuple[str, ...]
    codes: np.ndarray  # shape (n_settings, n_lenses)
    branches: np.ndarray  # shape (n_settings,)
    branch_breaks: tuple[float, float]
    normalized_codes: np.ndarray  # shape (n_settings, n_lenses), branch-local in [-1, 1]

    def as_json_dict(self) -> dict[str, Any]:
        return {
            "magnification": self.magnification.tolist(),
            "log_magnification": self.log_magnification.tolist(),
            "lens_names": list(self.lens_names),
            "codes": self.codes.tolist(),
            "branches": self.branches.tolist(),
            "branch_breaks": list(self.branch_breaks),
            "normalized_codes": self.normalized_codes.tolist(),
        }


@dataclass
class ProjectorFitParams:
    gc_var_lenses: np.ndarray  # shape (n_lenses,)
    beta: np.ndarray  # shape (n_lenses,)
    alpha: np.ndarray  # shape (n_lenses, n_branches)
    rotation_signs: np.ndarray  # shape (n_lenses,)
    k_tilde_ratios: np.ndarray  # shape (n_lenses,), normalized to k[0]=1

    def as_json_dict(self) -> dict[str, Any]:
        return {
            "gc_var_lenses": self.gc_var_lenses.tolist(),
            "beta": self.beta.tolist(),
            "alpha": self.alpha.tolist(),
            "rotation_signs": self.rotation_signs.tolist(),
            "k_tilde_ratios": self.k_tilde_ratios.tolist(),
        }


@dataclass
class ProjectorModel:
    model_type: str
    schema_version: int
    voltage_v: float
    rc_rad_per_at: float
    lens_names: tuple[str, ...]
    geometry: ProjectorGeometry
    dataset: HexProjectorDataset
    fit_params: ProjectorFitParams
    fit_table: list[dict[str, Any]]
    solver_info: dict[str, Any]

    def as_json_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "schema_version": self.schema_version,
            "voltage_v": self.voltage_v,
            "rc_rad_per_at": self.rc_rad_per_at,
            "lens_names": list(self.lens_names),
            "geometry": asdict(self.geometry),
            "dataset": self.dataset.as_json_dict(),
            "fit_params": self.fit_params.as_json_dict(),
            "fit_table": self.fit_table,
            "solver_info": self.solver_info,
        }


def _hex_to_int(value: str | int | float) -> float:
    if isinstance(value, str):
        text = value.strip().lower()
        if text.startswith("0x"):
            return float(int(text, 16))
        return float(text)
    return float(value)


def _branch_ids_from_mag(
    magnification: np.ndarray, branch_breaks: tuple[float, float]
) -> np.ndarray:
    b0, b1 = branch_breaks
    out = np.zeros_like(magnification, dtype=int)
    out[magnification > b0] = 1
    out[magnification > b1] = 2
    return out


def parse_projector_hex_table(
    raw_hex_table: Mapping[str, Mapping[str, Any]],
    lens_names: Sequence[str] = DEFAULT_PROJECTOR_LENS_NAMES,
    branch_breaks: tuple[float, float] = (6000.0, 25000.0),
) -> HexProjectorDataset:
    key_by_mag = {float(k): k for k in raw_hex_table.keys()}
    mags = np.array(sorted(key_by_mag.keys()), dtype=float)
    lens_names = tuple(str(name) for name in lens_names)
    n = len(mags)
    l = len(lens_names)

    codes = np.zeros((n, l), dtype=float)
    for i, mag in enumerate(mags):
        row = raw_hex_table[key_by_mag[float(mag)]]
        for j, lens in enumerate(lens_names):
            if lens not in row:
                raise KeyError(f"Missing lens '{lens}' at magnification {int(mag)}.")
            codes[i, j] = _hex_to_int(row[lens])

    branches = _branch_ids_from_mag(mags, branch_breaks)
    n_branches = 3
    u = np.zeros_like(codes, dtype=float)

    for b in range(n_branches):
        mask = branches == b
        if not np.any(mask):
            continue
        for j in range(l):
            vals = codes[mask, j]
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
            if np.isclose(vmin, vmax):
                u[mask, j] = 0.0
            else:
                u[mask, j] = 2.0 * (vals - vmin) / (vmax - vmin) - 1.0

    return HexProjectorDataset(
        magnification=mags,
        log_magnification=np.log10(mags),
        lens_names=lens_names,
        codes=codes,
        branches=branches,
        branch_breaks=branch_breaks,
        normalized_codes=u,
    )


def _build_abcd_np(dists: np.ndarray, focals: np.ndarray) -> np.ndarray:
    m = np.array(propagation_matrix(float(dists[-1]), xp=np), dtype=float)
    for i in reversed(range(len(focals))):
        m = m @ np.array(lens_matrix(float(focals[i]), xp=np), dtype=float)
        m = m @ np.array(propagation_matrix(float(dists[i]), xp=np), dtype=float)
    return m


def _currents_from_piecewise(
    beta: np.ndarray, alpha: np.ndarray, branches: np.ndarray, u: np.ndarray
) -> np.ndarray:
    n_settings, n_lenses = u.shape
    currents = np.zeros((n_settings, n_lenses), dtype=float)
    for i in range(n_settings):
        b = int(branches[i])
        currents[i, :] = beta + alpha[:, b] * u[i, :]
    return currents


def simulate_projector_response(
    dataset: HexProjectorDataset,
    geometry: ProjectorGeometry,
    gc_var_lenses: np.ndarray,
    beta: np.ndarray,
    alpha: np.ndarray,
    rotation_signs: np.ndarray,
) -> dict[str, np.ndarray]:
    gc_var_lenses = np.asarray(gc_var_lenses, dtype=float)
    beta = np.asarray(beta, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    rotation_signs = np.asarray(rotation_signs, dtype=float)

    currents = _currents_from_piecewise(
        beta=beta,
        alpha=alpha,
        branches=dataset.branches,
        u=dataset.normalized_codes,
    )

    i_sq = np.maximum(currents**2, 1e-30)
    f_var = 1.0 / (gc_var_lenses[None, :] * i_sq)
    f_opl = np.full((len(dataset.magnification), 1), float(geometry.f_opl_m), dtype=float)
    focals_all = np.concatenate([f_opl, f_var], axis=1)

    dists = geometry.drifts_m()
    A_abs = np.zeros(len(dataset.magnification), dtype=float)
    B = np.zeros(len(dataset.magnification), dtype=float)
    for i in range(len(dataset.magnification)):
        m = _build_abcd_np(dists=dists, focals=focals_all[i, :])
        A_abs[i] = abs(float(m[0, 0]))
        B[i] = float(m[0, 1])

    psi_total = currents @ rotation_signs
    return {
        "currents": currents,
        "focals_var": f_var,
        "focals_all": focals_all,
        "magnification_abs": A_abs,
        "B": B,
        "psi_total": psi_total,
    }


def _solve_k_tilde(currents: np.ndarray) -> np.ndarray:
    # currents @ k ~= 0, with k0 = 1
    a = currents[:, 1:]
    b = -currents[:, 0]
    k_rest, *_ = np.linalg.lstsq(a, b, rcond=None)
    return np.array([1.0, *k_rest.tolist()], dtype=float)


def fit_projector_model(
    raw_hex_table: Mapping[str, Mapping[str, Any]],
    geometry: ProjectorGeometry | None = None,
    *,
    voltage_v: float = 200e3,
    lens_names: Sequence[str] = DEFAULT_PROJECTOR_LENS_NAMES,
    branch_breaks: tuple[float, float] = (6000.0, 25000.0),
    target_magnification: np.ndarray | None = None,
    absolute_labels: bool = True,
    n_starts: int = 20,
    random_seed: int = 0,
    max_nfev: int = 4000,
    w_b: float = 0.5,
    w_rotation: float = 1.0,
    w_smooth: float = 0.05,
    w_cont: float = 0.2,
) -> ProjectorModel:
    geometry = geometry or ProjectorGeometry()
    dataset = parse_projector_hex_table(
        raw_hex_table=raw_hex_table,
        lens_names=lens_names,
        branch_breaks=branch_breaks,
    )
    target = (
        np.asarray(target_magnification, dtype=float)
        if target_magnification is not None
        else np.asarray(dataset.magnification, dtype=float)
    )
    if target.shape != dataset.magnification.shape:
        raise ValueError("target_magnification must match number of settings.")

    n_lenses = len(dataset.lens_names)
    n_branches = 3
    rotation_signs = np.resize(np.array([1.0, -1.0], dtype=float), n_lenses)
    dists = geometry.drifts_m()
    total_length = max(float(np.sum(dists)), 1e-12)
    i_ref = 5000.0

    def decode(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = 0
        log_gc = x[idx : idx + n_lenses]
        idx += n_lenses
        beta = x[idx : idx + n_lenses]
        idx += n_lenses
        alpha = x[idx : idx + n_lenses * n_branches].reshape(n_lenses, n_branches)
        return np.exp(log_gc), beta, alpha

    def residual(x: np.ndarray) -> np.ndarray:
        gc, beta, alpha = decode(x)
        sim = simulate_projector_response(
            dataset=dataset,
            geometry=geometry,
            gc_var_lenses=gc,
            beta=beta,
            alpha=alpha,
            rotation_signs=rotation_signs,
        )
        a = np.clip(sim["magnification_abs"], 1e-20, np.inf)
        t = np.clip(target, 1e-20, np.inf)
        if absolute_labels:
            r_mag = np.log(a) - np.log(t)
        else:
            r_mag = np.log(a / a[0]) - np.log(t / t[0])

        r_b = w_b * sim["B"] / total_length
        r_rot = w_rotation * sim["psi_total"] / i_ref

        extra: list[np.ndarray] = []
        # Smoothness over log-magnification.
        order = np.argsort(dataset.log_magnification)
        cur_sorted = sim["currents"][order, :]
        for j in range(n_lenses):
            col = cur_sorted[:, j]
            if len(col) > 2:
                d2 = col[2:] - 2.0 * col[1:-1] + col[:-2]
                extra.append(w_smooth * d2 / i_ref)

        # Branch boundary continuity in piecewise affine current model.
        for j in range(n_lenses):
            for b in range(n_branches - 1):
                i_end = beta[j] + alpha[j, b] * 1.0
                i_start = beta[j] + alpha[j, b + 1] * (-1.0)
                extra.append(np.array([w_cont * (i_end - i_start) / i_ref], dtype=float))

        if extra:
            return np.concatenate([r_mag, r_b, r_rot, *extra], axis=0)
        return np.concatenate([r_mag, r_b, r_rot], axis=0)

    x0 = np.concatenate(
        [
            np.log(np.full(n_lenses, 5.0e-6, dtype=float)),
            np.full(n_lenses, 5000.0, dtype=float),
            np.full(n_lenses * n_branches, 2000.0, dtype=float),
        ]
    )
    lower = np.concatenate(
        [
            np.log(np.full(n_lenses, 1.0e-12, dtype=float)),
            np.full(n_lenses, 200.0, dtype=float),
            np.full(n_lenses * n_branches, -2.0e4, dtype=float),
        ]
    )
    upper = np.concatenate(
        [
            np.log(np.full(n_lenses, 1.0e-4, dtype=float)),
            np.full(n_lenses, 2.0e4, dtype=float),
            np.full(n_lenses * n_branches, 2.0e4, dtype=float),
        ]
    )

    rng = np.random.default_rng(random_seed)
    best_sol = None
    best_loss = np.inf
    for start in range(max(1, int(n_starts))):
        if start == 0:
            x_init = x0.copy()
        else:
            noise = rng.normal(size=x0.shape)
            x_init = np.clip(x0 + 0.25 * noise, lower, upper)
        sol = least_squares(
            residual,
            x_init,
            bounds=(lower, upper),
            method="trf",
            loss="soft_l1",
            max_nfev=int(max_nfev),
        )
        loss = float(np.sum(residual(sol.x) ** 2))
        if loss < best_loss:
            best_loss = loss
            best_sol = sol

    assert best_sol is not None
    gc, beta, alpha = decode(best_sol.x)
    sim = simulate_projector_response(
        dataset=dataset,
        geometry=geometry,
        gc_var_lenses=gc,
        beta=beta,
        alpha=alpha,
        rotation_signs=rotation_signs,
    )
    k_tilde = _solve_k_tilde(sim["currents"])

    fit_table: list[dict[str, Any]] = []
    for i, mag in enumerate(dataset.magnification):
        row = {
            "magnification_label": float(mag),
            "magnification_target": float(target[i]),
            "magnification_predicted": float(sim["magnification_abs"][i]),
            "magnification_error_pct": float(
                100.0 * (sim["magnification_abs"][i] - target[i]) / max(target[i], 1e-30)
            ),
            "B": float(sim["B"][i]),
            "psi_total": float(sim["psi_total"][i]),
            "branch": int(dataset.branches[i]),
        }
        for j, name in enumerate(dataset.lens_names):
            row[f"I0_{name}_at"] = float(sim["currents"][i, j])
            row[f"f_{name}_m"] = float(sim["focals_var"][i, j])
            row[f"dac_{name}"] = float(dataset.codes[i, j])
        fit_table.append(row)

    fit_params = ProjectorFitParams(
        gc_var_lenses=gc,
        beta=beta,
        alpha=alpha,
        rotation_signs=rotation_signs,
        k_tilde_ratios=k_tilde,
    )

    rc = float(compute_Rc_from_voltage(voltage_v))
    solver_info = {
        "best_loss": best_loss,
        "n_starts": int(n_starts),
        "success": bool(best_sol.success),
        "message": str(best_sol.message),
        "nfev": int(best_sol.nfev),
    }
    return ProjectorModel(
        model_type="tem_projector_em_piecewise_fit",
        schema_version=1,
        voltage_v=float(voltage_v),
        rc_rad_per_at=rc,
        lens_names=tuple(dataset.lens_names),
        geometry=geometry,
        dataset=dataset,
        fit_params=fit_params,
        fit_table=fit_table,
        solver_info=solver_info,
    )


def _evaluate_magnification_from_currents(
    currents: np.ndarray,
    gc_var_lenses: np.ndarray,
    geometry: ProjectorGeometry,
) -> tuple[float, float]:
    focals_var = 1.0 / (np.maximum(gc_var_lenses * np.maximum(currents**2, 1e-30), 1e-30))
    focals = np.concatenate([[geometry.f_opl_m], focals_var], axis=0)
    m = _build_abcd_np(dists=geometry.drifts_m(), focals=focals)
    return abs(float(m[0, 0])), float(m[0, 1])


def realize_projector_setting(
    model: ProjectorModel,
    magnification: float,
    *,
    mode: str = "interpolate",
    correct_zero_rotation: bool = True,
) -> dict[str, Any]:
    mags = np.asarray(model.dataset.magnification, dtype=float)
    log_mags = np.log10(mags)
    x = float(magnification)
    x_clip = float(np.clip(x, float(np.min(mags)), float(np.max(mags))))

    currents_table = np.array(
        [
            [row[f"I0_{name}_at"] for name in model.lens_names]
            for row in model.fit_table
        ],
        dtype=float,
    )

    if mode == "nearest":
        idx = int(np.argmin(np.abs(mags - x)))
        currents = currents_table[idx].copy()
    elif mode == "interpolate":
        q = np.log10(x_clip)
        currents = np.array(
            [np.interp(q, log_mags, currents_table[:, j]) for j in range(currents_table.shape[1])],
            dtype=float,
        )
    else:
        raise ValueError("mode must be 'interpolate' or 'nearest'.")

    gc = np.asarray(model.fit_params.gc_var_lenses, dtype=float)
    s = np.asarray(model.fit_params.rotation_signs, dtype=float)
    psi_total = float(np.dot(s, currents))
    a0, b0 = _evaluate_magnification_from_currents(currents, gc, model.geometry)

    # Runtime correction for image-plane zero rotation.
    if correct_zero_rotation and len(currents) >= 2:
        # Adjust one lens while balancing another so sum(s_i * I_i) == 0 is preserved.
        i_tune = len(currents) - 1
        i_balance = 0
        if i_balance == i_tune:
            i_balance = max(0, i_tune - 1)

        if abs(s[i_balance]) > 1e-14:
            target = max(x_clip, 1e-20)
            lo = float(np.min(currents_table[:, i_tune]) * 0.98)
            hi = float(np.max(currents_table[:, i_tune]) * 1.02)

            def currents_with_zero_rotation(i_tune_value: float) -> np.ndarray:
                c = currents.copy()
                c[i_tune] = float(i_tune_value)
                residual = float(np.dot(s, c))
                c[i_balance] = c[i_balance] - residual / s[i_balance]
                return c

            def objective(i_tune_value: float) -> float:
                c = currents_with_zero_rotation(i_tune_value)
                a, _ = _evaluate_magnification_from_currents(c, gc, model.geometry)
                return np.log(max(a, 1e-20)) - np.log(target)

            # Start from interpolation and project onto zero-rotation manifold.
            currents = currents_with_zero_rotation(currents[i_tune])
            f_lo = objective(lo)
            f_hi = objective(hi)

            if np.isfinite(f_lo) and np.isfinite(f_hi) and (f_lo == 0.0 or f_hi == 0.0 or f_lo * f_hi < 0.0):
                a_bracket = lo
                b_bracket = hi
                fa = f_lo
                fb = f_hi
                if fa == 0.0:
                    currents = currents_with_zero_rotation(a_bracket)
                elif fb == 0.0:
                    currents = currents_with_zero_rotation(b_bracket)
                else:
                    # Bisection on bounded interval for robust convergence.
                    for _ in range(40):
                        mid = 0.5 * (a_bracket + b_bracket)
                        fm = objective(mid)
                        if not np.isfinite(fm) or abs(fm) < 1e-10:
                            currents = currents_with_zero_rotation(mid)
                            break
                        if fa * fm < 0.0:
                            b_bracket, fb = mid, fm
                        else:
                            a_bracket, fa = mid, fm
                    else:
                        currents = currents_with_zero_rotation(0.5 * (a_bracket + b_bracket))
            else:
                # Fallback: minimize |objective| over a coarse grid while preserving zero rotation.
                grid = np.linspace(lo, hi, 64, dtype=float)
                values = np.array([abs(objective(g)) for g in grid], dtype=float)
                best = float(grid[int(np.nanargmin(values))])
                currents = currents_with_zero_rotation(best)

    a, b = _evaluate_magnification_from_currents(currents, gc, model.geometry)
    psi = float(np.dot(s, currents))
    focals_var = 1.0 / (np.maximum(gc * np.maximum(currents**2, 1e-30), 1e-30))
    return {
        "magnification_request": float(magnification),
        "magnification_clipped": x_clip,
        "mode": mode,
        "correct_zero_rotation": bool(correct_zero_rotation),
        "currents_at": currents,
        "focals_var_m": focals_var,
        "magnification_predicted": float(a),
        "B": float(b),
        "psi_total": psi,
        "magnification_predicted_before_correction": float(a0),
        "B_before_correction": float(b0),
    }


def build_projector_components(
    model: ProjectorModel,
    magnification: float,
    *,
    z0: float = 0.0,
    mode: str = "interpolate",
    correct_zero_rotation: bool = True,
    include_detector_plane: bool = True,
) -> tuple[Any, ...]:
    realized = realize_projector_setting(
        model=model,
        magnification=magnification,
        mode=mode,
        correct_zero_rotation=correct_zero_rotation,
    )
    currents = np.asarray(realized["currents_at"], dtype=float)
    gc_var = np.asarray(model.fit_params.gc_var_lenses, dtype=float)
    s = np.asarray(model.fit_params.rotation_signs, dtype=float)
    rc = float(model.rc_rad_per_at)

    g = model.geometry
    d = g.drifts_m()
    z_opl = z0 + d[0]
    z_il1 = z_opl + d[1]
    z_il2 = z_il1 + d[2]
    z_il3 = z_il2 + d[3]
    z_pl1 = z_il3 + d[4]
    z_det = z_pl1 + d[5]

    opl_gc = 1.0 / (g.f_opl_m * g.opl_i0_at**2)
    comps: list[Any] = [
        ElectromagneticLens(
            z=float(z_opl),
            I0=float(g.opl_i0_at),
            Gc=float(opl_gc),
            Rc=0.0,
        ),
        ElectromagneticLens(z=float(z_il1), I0=float(currents[0]), Gc=float(gc_var[0]), Rc=float(rc * s[0])),
        ElectromagneticLens(z=float(z_il2), I0=float(currents[1]), Gc=float(gc_var[1]), Rc=float(rc * s[1])),
        ElectromagneticLens(z=float(z_il3), I0=float(currents[2]), Gc=float(gc_var[2]), Rc=float(rc * s[2])),
        ElectromagneticLens(z=float(z_pl1), I0=float(currents[3]), Gc=float(gc_var[3]), Rc=float(rc * s[3])),
    ]
    if include_detector_plane:
        comps.append(Plane(z=float(z_det)))
    return tuple(comps)


def projector_metrics(
    model: ProjectorModel,
    magnification: float,
    *,
    mode: str = "interpolate",
    correct_zero_rotation: bool = True,
) -> dict[str, Any]:
    return realize_projector_setting(
        model=model,
        magnification=magnification,
        mode=mode,
        correct_zero_rotation=correct_zero_rotation,
    )


def export_projector_model_json(model: ProjectorModel, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model.as_json_dict(), indent=2))
    return out


def load_projector_model_json(path: str | Path) -> ProjectorModel:
    payload = json.loads(Path(path).read_text())
    geometry = ProjectorGeometry(**payload["geometry"])
    dataset_p = payload["dataset"]
    dataset = HexProjectorDataset(
        magnification=np.asarray(dataset_p["magnification"], dtype=float),
        log_magnification=np.asarray(dataset_p["log_magnification"], dtype=float),
        lens_names=tuple(dataset_p["lens_names"]),
        codes=np.asarray(dataset_p["codes"], dtype=float),
        branches=np.asarray(dataset_p["branches"], dtype=int),
        branch_breaks=tuple(float(x) for x in dataset_p["branch_breaks"]),
        normalized_codes=np.asarray(dataset_p["normalized_codes"], dtype=float),
    )
    fit_p = payload["fit_params"]
    fit_params = ProjectorFitParams(
        gc_var_lenses=np.asarray(fit_p["gc_var_lenses"], dtype=float),
        beta=np.asarray(fit_p["beta"], dtype=float),
        alpha=np.asarray(fit_p["alpha"], dtype=float),
        rotation_signs=np.asarray(fit_p["rotation_signs"], dtype=float),
        k_tilde_ratios=np.asarray(fit_p["k_tilde_ratios"], dtype=float),
    )
    return ProjectorModel(
        model_type=str(payload["model_type"]),
        schema_version=int(payload["schema_version"]),
        voltage_v=float(payload["voltage_v"]),
        rc_rad_per_at=float(payload["rc_rad_per_at"]),
        lens_names=tuple(payload["lens_names"]),
        geometry=geometry,
        dataset=dataset,
        fit_params=fit_params,
        fit_table=list(payload["fit_table"]),
        solver_info=dict(payload["solver_info"]),
    )


@dataclass
class SimplifiedProjectorModel:
    """
    DAC-free projector zoom model solved from target magnifications.

    The model assumes:
    - fixed strong OPL focal length
    - three IL lenses (IL1/IL2/IL3) with fixed Gc and variable current
    - optional variable PL1 current (fixed PL1 Gc)
    - zero total image rotation enforced through current balancing
    """

    model_type: str
    schema_version: int
    voltage_v: float
    rc_rad_per_at: float
    geometry: ProjectorGeometry
    objective_focal_length_m: float
    projector_focal_length_m: float
    objective_current_at: float
    projector_current_at: float
    il_gc: np.ndarray  # shape (3,)
    il_rotation_signs: np.ndarray  # shape (3,)
    il_current_bounds_at: tuple[float, float]
    weak_min_focal_m: float
    fit_table: list[dict[str, Any]]
    solver_info: dict[str, Any]
    variable_projector_current: bool = False
    projector_current_bounds_at: tuple[float, float] | None = None

    def as_json_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "schema_version": self.schema_version,
            "voltage_v": self.voltage_v,
            "rc_rad_per_at": self.rc_rad_per_at,
            "geometry": asdict(self.geometry),
            "objective_focal_length_m": self.objective_focal_length_m,
            "projector_focal_length_m": self.projector_focal_length_m,
            "objective_current_at": self.objective_current_at,
            "projector_current_at": self.projector_current_at,
            "il_gc": self.il_gc.tolist(),
            "il_rotation_signs": self.il_rotation_signs.tolist(),
            "il_current_bounds_at": list(self.il_current_bounds_at),
            "weak_min_focal_m": self.weak_min_focal_m,
            "fit_table": self.fit_table,
            "solver_info": self.solver_info,
            "variable_projector_current": bool(self.variable_projector_current),
            "projector_current_bounds_at": (
                list(self.projector_current_bounds_at)
                if self.projector_current_bounds_at is not None
                else None
            ),
        }


def _focals_with_fixed_strong_lenses(
    objective_focal_length_m: float,
    il_focals_m: np.ndarray,
    projector_focal_length_m: float,
) -> np.ndarray:
    return np.array(
        [
            float(objective_focal_length_m),
            float(il_focals_m[0]),
            float(il_focals_m[1]),
            float(il_focals_m[2]),
            float(projector_focal_length_m),
        ],
        dtype=float,
    )


def _balanced_il_currents_from_free(
    free_currents: np.ndarray,
    il_rotation_signs: np.ndarray,
) -> np.ndarray:
    i1 = float(free_currents[0])
    i2 = float(free_currents[1])
    s = np.asarray(il_rotation_signs, dtype=float)
    if np.isclose(s[2], 0.0):
        raise ValueError("il_rotation_signs[2] must be non-zero for balance solve.")
    i3 = -(s[0] * i1 + s[1] * i2) / s[2]
    return np.array([i1, i2, i3], dtype=float)


def _bounded_to_unconstrained(currents: np.ndarray, low: float, high: float) -> np.ndarray:
    span = max(float(high - low), 1e-12)
    clipped = np.clip(np.asarray(currents, dtype=float), low + 1e-9, high - 1e-9)
    q = np.clip((clipped - low) / span, 1e-9, 1.0 - 1e-9)
    return np.log(q) - np.log1p(-q)


def _balanced_il_currents_from_unconstrained(
    unconstrained_free: np.ndarray,
    il_rotation_signs: np.ndarray,
    il_current_bounds_at: tuple[float, float],
) -> np.ndarray:
    low, high = il_current_bounds_at
    span = max(float(high - low), 1e-12)
    z = np.clip(np.asarray(unconstrained_free, dtype=float), -60.0, 60.0)
    free = low + span / (1.0 + np.exp(-z))
    return _balanced_il_currents_from_free(free_currents=free, il_rotation_signs=il_rotation_signs)


def _independent_il_currents_from_unconstrained(
    unconstrained_il: np.ndarray,
    il_current_bounds_at: tuple[float, float],
) -> np.ndarray:
    low, high = il_current_bounds_at
    span = max(float(high - low), 1e-12)
    z = np.clip(np.asarray(unconstrained_il, dtype=float), -60.0, 60.0)
    return low + span / (1.0 + np.exp(-z))


def _evaluate_simplified_setting(
    unconstrained_free: np.ndarray,
    *,
    il_rotation_signs: np.ndarray,
    exact_il_rotation_balance: bool,
    il_current_bounds_at: tuple[float, float],
    il_gc: np.ndarray,
    objective_focal_length_m: float,
    projector_focal_length_m: float,
    projector_gc: float,
    variable_projector_current: bool,
    projector_current_bounds_at: tuple[float, float],
    projector_current_at_fixed: float,
    geometry: ProjectorGeometry,
    rc_rad_per_at: float,
) -> dict[str, Any]:
    u = np.asarray(unconstrained_free, dtype=float).reshape(-1)
    n_il = 2 if exact_il_rotation_balance else 3
    n_expected = n_il + (1 if variable_projector_current else 0)
    if u.size < n_expected:
        raise ValueError(
            f"unconstrained_free must have length >= {n_expected} "
            f"(exact_il_rotation_balance={exact_il_rotation_balance}, "
            f"variable_projector_current={variable_projector_current})."
        )

    if exact_il_rotation_balance:
        il_currents = _balanced_il_currents_from_unconstrained(
            unconstrained_free=u[:2],
            il_rotation_signs=il_rotation_signs,
            il_current_bounds_at=il_current_bounds_at,
        )
    else:
        il_currents = _independent_il_currents_from_unconstrained(
            unconstrained_il=u[:3],
            il_current_bounds_at=il_current_bounds_at,
        )
    if variable_projector_current:
        pl1_lo, pl1_hi = projector_current_bounds_at
        z_pl1 = float(np.clip(u[n_il], -60.0, 60.0))
        projector_current_at = float(pl1_lo + (pl1_hi - pl1_lo) / (1.0 + np.exp(-z_pl1)))
    else:
        projector_current_at = float(projector_current_at_fixed)

    il_currents_safe = np.maximum(il_currents, 1e-12)
    il_focals = 1.0 / (np.asarray(il_gc, dtype=float) * il_currents_safe**2)
    projector_focal_effective = float(
        1.0 / max(float(projector_gc) * max(projector_current_at**2, 1e-30), 1e-30)
    )
    focals = _focals_with_fixed_strong_lenses(
        objective_focal_length_m=objective_focal_length_m,
        il_focals_m=il_focals,
        projector_focal_length_m=projector_focal_effective,
    )
    mtx = _build_abcd_np(dists=geometry.drifts_m(), focals=focals)
    mtx_pre = _build_abcd_np(
        dists=geometry.drifts_m()[:-1],
        focals=np.array(
            [
                float(objective_focal_length_m),
                float(il_focals[0]),
                float(il_focals[1]),
                float(il_focals[2]),
            ],
            dtype=float,
        ),
    )
    a_val = abs(float(mtx[0, 0]))
    b_val = float(mtx[0, 1])
    a_pre = float(mtx_pre[0, 0])
    b_pre = float(mtx_pre[0, 1])
    psi_val = float(rc_rad_per_at * np.dot(np.asarray(il_rotation_signs, dtype=float), il_currents))
    ffp_error = float(b_pre - a_pre * projector_focal_effective)
    return {
        "currents": il_currents,
        "currents_safe": il_currents_safe,
        "il_focals": il_focals,
        "magnification_abs": a_val,
        "B": b_val,
        "psi_total": psi_val,
        "projector_current_at": projector_current_at,
        "projector_focal_effective_m": projector_focal_effective,
        "A_pre": a_pre,
        "B_pre": b_pre,
        "ffp_error_m": ffp_error,
    }


def _simplified_setting_residual(
    unconstrained_free: np.ndarray,
    *,
    target_magnification: float,
    il_rotation_signs: np.ndarray,
    exact_il_rotation_balance: bool,
    il_current_bounds_at: tuple[float, float],
    il_gc: np.ndarray,
    objective_focal_length_m: float,
    projector_focal_length_m: float,
    projector_gc: float,
    variable_projector_current: bool,
    projector_current_bounds_at: tuple[float, float],
    projector_current_at_fixed: float,
    geometry: ProjectorGeometry,
    rc_rad_per_at: float,
    weak_min_focal_m: float,
    total_length_m: float,
    prev_currents: np.ndarray | None,
    prev_target_magnification: float | None,
    smoothness_weight: float,
    trend_band_c_magnification: tuple[float, float] | None,
    trend_band_c_weight: float,
    psi_weight: float,
    projector_focal_bounds_m: tuple[float, float] | None,
    projector_focal_bounds_weight: float,
    b_weight: float,
    ffp_weight: float,
) -> np.ndarray:
    state = _evaluate_simplified_setting(
        unconstrained_free=unconstrained_free,
        il_rotation_signs=il_rotation_signs,
        exact_il_rotation_balance=exact_il_rotation_balance,
        il_current_bounds_at=il_current_bounds_at,
        il_gc=il_gc,
        objective_focal_length_m=objective_focal_length_m,
        projector_focal_length_m=projector_focal_length_m,
        projector_gc=projector_gc,
        variable_projector_current=variable_projector_current,
        projector_current_bounds_at=projector_current_bounds_at,
        projector_current_at_fixed=projector_current_at_fixed,
        geometry=geometry,
        rc_rad_per_at=rc_rad_per_at,
    )

    low, high = il_current_bounds_at
    psi_scale = max(abs(float(rc_rad_per_at)) * float(high), 1.0)
    r_mag = np.log(max(state["magnification_abs"], 1e-20)) - np.log(max(float(target_magnification), 1e-20))
    r_b = float(b_weight) * state["B"] / max(float(total_length_m), 1e-12)
    r_psi = float(psi_weight) * state["psi_total"] / psi_scale
    r_ffp = float(ffp_weight) * state["ffp_error_m"] / max(float(projector_focal_length_m), 1e-12)

    weak_pen = 0.2 * np.clip(weak_min_focal_m - state["il_focals"], 0.0, np.inf) / weak_min_focal_m
    i3_low = max(low - state["currents"][2], 0.0) / max(low, 1e-12)
    i3_high = max(state["currents"][2] - high, 0.0) / max(high, 1e-12)
    pl1_low = 0.0
    pl1_high = 0.0
    if variable_projector_current:
        pl1_lo, pl1_hi = projector_current_bounds_at
        pl1_low = max(pl1_lo - state["projector_current_at"], 0.0) / max(pl1_lo, 1e-12)
        pl1_high = max(state["projector_current_at"] - pl1_hi, 0.0) / max(pl1_hi, 1e-12)
    f_pl1_low = 0.0
    f_pl1_high = 0.0
    if projector_focal_bounds_m is not None:
        f_lo, f_hi = projector_focal_bounds_m
        f_eff = float(state["projector_focal_effective_m"])
        f_scale = float(projector_focal_bounds_weight)
        f_pl1_low = f_scale * max(f_lo - f_eff, 0.0) / max(f_lo, 1e-12)
        f_pl1_high = f_scale * max(f_eff - f_hi, 0.0) / max(f_hi, 1e-12)

    pieces: list[np.ndarray] = [
        np.array([r_mag, r_b, r_psi, r_ffp], dtype=float),
        weak_pen,
        np.array([i3_low, i3_high, pl1_low, pl1_high, f_pl1_low, f_pl1_high], dtype=float),
    ]
    if (
        trend_band_c_magnification is not None
        and trend_band_c_weight > 0.0
        and prev_currents is not None
        and prev_target_magnification is not None
    ):
        band_lo, band_hi = trend_band_c_magnification
        in_band_now = band_lo <= float(target_magnification) <= band_hi
        in_band_prev = band_lo <= float(prev_target_magnification) <= band_hi
        if in_band_now and in_band_prev:
            curr_safe = np.maximum(state["currents_safe"], low)
            prev_safe = np.maximum(prev_currents, low)
            dlog = np.log(curr_safe) - np.log(prev_safe)
            trend_pen = float(trend_band_c_weight) * np.array(
                [
                    dlog[0],            # IL1 should stay approximately flat.
                    min(dlog[1], 0.0), # IL2 should increase across band C.
                    max(dlog[2], 0.0), # IL3 should decrease across band C.
                ],
                dtype=float,
            )
            pieces.append(trend_pen)
    if prev_currents is not None and smoothness_weight > 0.0:
        smooth_pen = smoothness_weight * (
            np.log(np.maximum(state["currents_safe"], low)) - np.log(np.maximum(prev_currents, low))
        )
        pieces.append(smooth_pen)
    return np.concatenate(pieces, axis=0)


def _solve_setting_with_restarts(
    residual_fn,
    x0: np.ndarray,
    *,
    method: str,
    max_nfev: int,
    n_starts: int,
) -> tuple[Any, float]:
    x_base = np.asarray(x0, dtype=float)
    n_dim = int(x_base.size)
    rng = np.random.default_rng(0)
    best_sol = None
    best_loss = np.inf
    for k in range(max(1, int(n_starts))):
        if k == 0:
            offset = np.zeros(n_dim, dtype=float)
        else:
            scale = 0.8 * (1.0 + 0.15 * (k // 8))
            offset = rng.normal(loc=0.0, scale=scale, size=n_dim)
        x_init = x_base + offset
        sol = least_squares(
            residual_fn,
            x_init,
            method=method,
            max_nfev=int(max_nfev),
        )
        r = np.asarray(residual_fn(sol.x), dtype=float)
        loss = float(np.dot(r, r))
        if loss < best_loss:
            best_loss = loss
            best_sol = sol
    assert best_sol is not None
    return best_sol, float(best_loss)


def solve_simplified_projector_zoom(
    target_magnifications: Sequence[float],
    geometry: ProjectorGeometry | None = None,
    *,
    objective_focal_length_m: float = 2.0e-3,
    projector_focal_length_m: float = 2.0e-3,
    objective_current_at: float = 5000.0,
    projector_current_at: float = 5000.0,
    il_reference_focal_lengths_m: Sequence[float] = (15.0e-3, 18.0e-3, 22.0e-3),
    il_reference_currents_at: Sequence[float] = (3000.0, 3000.0, 3000.0),
    il_rotation_signs: Sequence[float] = (1.0, -1.0, 1.0),
    voltage_v: float = 200e3,
    rc_rad_per_at: float | None = None,
    exact_il_rotation_balance: bool = True,
    il_current_bounds_at: tuple[float, float] = (200.0, 40000.0),
    variable_projector_current: bool = False,
    projector_current_bounds_at: tuple[float, float] | None = None,
    projector_focal_bounds_m: tuple[float, float] | None = None,
    projector_focal_bounds_weight: float = 1.0,
    weak_min_ratio: float = 1.10,
    smoothness_weight: float = 0.02,
    trend_band_c_magnification: tuple[float, float] | None = None,
    trend_band_c_weight: float = 0.0,
    psi_weight: float = 0.1,
    psi_tolerance_rad: float = 1e-9,
    b_weight: float = 1.0,
    b_weight_relaxed: float = 0.05,
    ffp_weight: float = 0.0,
    ffp_tolerance_m: float = 5.0e-5,
    solver_method: str = "lm",
    n_starts_per_setting: int = 6,
    max_nfev: int = 3000,
    mag_tolerance_pct: float = 2.0,
    b_tolerance_m: float = 5.0e-5,
) -> SimplifiedProjectorModel:
    """
    Solve a projector model from target magnifications with configurable
    rotation balancing.

    By default, the solver uses two free IL currents per setting and computes
    the third current from the zero-rotation constraint (exact balance mode).
    When `exact_il_rotation_balance=False`, all three IL currents are free and
    `psi_total` is controlled through a weighted residual term.
    """

    geometry = geometry or ProjectorGeometry()
    mags = np.asarray(target_magnifications, dtype=float).reshape(-1)
    if mags.size == 0:
        raise ValueError("target_magnifications must contain at least one value.")
    if np.any(mags <= 0):
        raise ValueError("target_magnifications must be > 0.")
    mags = np.sort(mags)

    if objective_focal_length_m <= 0 or projector_focal_length_m <= 0:
        raise ValueError("Strong lens focal lengths must be > 0.")
    if objective_current_at <= 0 or projector_current_at <= 0:
        raise ValueError("Strong lens currents must be > 0.")

    il_f_ref = np.asarray(il_reference_focal_lengths_m, dtype=float)
    il_i_ref = np.asarray(il_reference_currents_at, dtype=float)
    il_s = np.asarray(il_rotation_signs, dtype=float)
    if il_f_ref.shape != (3,) or il_i_ref.shape != (3,) or il_s.shape != (3,):
        raise ValueError("IL reference focal/current arrays and rotation signs must have length 3.")
    if np.any(il_f_ref <= 0) or np.any(il_i_ref <= 0):
        raise ValueError("IL reference focal lengths and currents must be > 0.")
    if np.any(np.isclose(il_s, 0.0)):
        raise ValueError("IL rotation signs must be non-zero.")

    i_lo = float(il_current_bounds_at[0])
    i_hi = float(il_current_bounds_at[1])
    if i_lo <= 0 or i_hi <= i_lo:
        raise ValueError("il_current_bounds_at must satisfy 0 < low < high.")
    pl1_bounds = projector_current_bounds_at or il_current_bounds_at
    pl1_lo = float(pl1_bounds[0])
    pl1_hi = float(pl1_bounds[1])
    if pl1_lo <= 0 or pl1_hi <= pl1_lo:
        raise ValueError("projector_current_bounds_at must satisfy 0 < low < high.")
    if projector_focal_bounds_m is not None:
        f_pl1_lo = float(projector_focal_bounds_m[0])
        f_pl1_hi = float(projector_focal_bounds_m[1])
        if f_pl1_lo <= 0 or f_pl1_hi <= f_pl1_lo:
            raise ValueError("projector_focal_bounds_m must satisfy 0 < low < high.")
    else:
        f_pl1_lo = 0.0
        f_pl1_hi = 0.0
    if projector_focal_bounds_weight < 0:
        raise ValueError("projector_focal_bounds_weight must be >= 0.")
    if trend_band_c_magnification is not None:
        band_lo = float(trend_band_c_magnification[0])
        band_hi = float(trend_band_c_magnification[1])
        if band_lo <= 0 or band_hi <= band_lo:
            raise ValueError("trend_band_c_magnification must satisfy 0 < low < high.")
    else:
        band_lo = 0.0
        band_hi = 0.0

    rc = float(rc_rad_per_at) if rc_rad_per_at is not None else float(compute_Rc_from_voltage(voltage_v))
    il_gc = 1.0 / (il_f_ref * il_i_ref**2)
    projector_gc = 1.0 / max(float(projector_focal_length_m) * float(projector_current_at) ** 2, 1e-30)
    weak_min_focal_m = float(
        weak_min_ratio * max(float(objective_focal_length_m), float(projector_focal_length_m))
    )
    total_length = max(float(geometry.total_length_m()), 1e-12)
    mag_tol_log = np.log(1.0 + max(float(mag_tolerance_pct), 0.0) / 100.0)

    allowed_methods = {"lm", "trf", "dogbox"}
    method = str(solver_method).lower().strip()
    if method not in allowed_methods:
        raise ValueError(f"solver_method must be one of {sorted(allowed_methods)}.")

    fit_table: list[dict[str, Any]] = []
    n_il_free = 2 if exact_il_rotation_balance else 3
    x_prev_il = _bounded_to_unconstrained(np.clip(il_i_ref[:n_il_free], i_lo, i_hi), i_lo, i_hi)
    if variable_projector_current:
        x_prev_pl1 = _bounded_to_unconstrained(np.array([projector_current_at], dtype=float), pl1_lo, pl1_hi)
        x_prev = np.concatenate([x_prev_il, x_prev_pl1], axis=0)
    else:
        x_prev = x_prev_il
    prev_currents = np.asarray(il_i_ref, dtype=float)
    prev_target = None
    success_count = 0
    relaxed_count = 0

    for mag in mags:
        target = float(mag)

        def make_residual(weight_b: float):
            return lambda x: _simplified_setting_residual(
                unconstrained_free=x,
                target_magnification=target,
                il_rotation_signs=il_s,
                exact_il_rotation_balance=bool(exact_il_rotation_balance),
                il_current_bounds_at=(i_lo, i_hi),
                il_gc=il_gc,
                objective_focal_length_m=objective_focal_length_m,
                projector_focal_length_m=projector_focal_length_m,
                projector_gc=projector_gc,
                variable_projector_current=bool(variable_projector_current),
                projector_current_bounds_at=(pl1_lo, pl1_hi),
                projector_current_at_fixed=float(projector_current_at),
                geometry=geometry,
                rc_rad_per_at=rc,
                weak_min_focal_m=weak_min_focal_m,
                total_length_m=total_length,
                prev_currents=prev_currents,
                prev_target_magnification=prev_target,
                smoothness_weight=smoothness_weight,
                trend_band_c_magnification=(
                    None if trend_band_c_magnification is None else (band_lo, band_hi)
                ),
                trend_band_c_weight=float(trend_band_c_weight),
                psi_weight=float(psi_weight),
                projector_focal_bounds_m=(
                    None if projector_focal_bounds_m is None else (f_pl1_lo, f_pl1_hi)
                ),
                projector_focal_bounds_weight=float(projector_focal_bounds_weight),
                b_weight=weight_b,
                ffp_weight=ffp_weight,
            )

        residual_primary = make_residual(float(b_weight))
        sol, _ = _solve_setting_with_restarts(
            residual_primary,
            x_prev,
            method=method,
            max_nfev=int(max_nfev),
            n_starts=int(n_starts_per_setting),
        )
        state = _evaluate_simplified_setting(
            unconstrained_free=sol.x,
            il_rotation_signs=il_s,
            exact_il_rotation_balance=bool(exact_il_rotation_balance),
            il_current_bounds_at=(i_lo, i_hi),
            il_gc=il_gc,
            objective_focal_length_m=objective_focal_length_m,
            projector_focal_length_m=projector_focal_length_m,
            projector_gc=projector_gc,
            variable_projector_current=bool(variable_projector_current),
            projector_current_bounds_at=(pl1_lo, pl1_hi),
            projector_current_at_fixed=float(projector_current_at),
            geometry=geometry,
            rc_rad_per_at=rc,
        )
        used_b_weight = float(b_weight)
        used_relaxed_b = False
        mag_err_log = float(np.log(max(state["magnification_abs"], 1e-20)) - np.log(target))

        if abs(mag_err_log) > mag_tol_log and float(b_weight_relaxed) < float(b_weight):
            residual_relaxed = make_residual(float(b_weight_relaxed))
            sol_relaxed, _ = _solve_setting_with_restarts(
                residual_relaxed,
                x_prev,
                method=method,
                max_nfev=int(max_nfev),
                n_starts=int(n_starts_per_setting),
            )
            state_relaxed = _evaluate_simplified_setting(
                unconstrained_free=sol_relaxed.x,
                il_rotation_signs=il_s,
                exact_il_rotation_balance=bool(exact_il_rotation_balance),
                il_current_bounds_at=(i_lo, i_hi),
                il_gc=il_gc,
                objective_focal_length_m=objective_focal_length_m,
                projector_focal_length_m=projector_focal_length_m,
                projector_gc=projector_gc,
                variable_projector_current=bool(variable_projector_current),
                projector_current_bounds_at=(pl1_lo, pl1_hi),
                projector_current_at_fixed=float(projector_current_at),
                geometry=geometry,
                rc_rad_per_at=rc,
            )
            mag_err_log_relaxed = float(
                np.log(max(state_relaxed["magnification_abs"], 1e-20)) - np.log(target)
            )
            if abs(mag_err_log_relaxed) < abs(mag_err_log):
                sol = sol_relaxed
                state = state_relaxed
                mag_err_log = mag_err_log_relaxed
                used_b_weight = float(b_weight_relaxed)
                used_relaxed_b = True
                relaxed_count += 1

        x_prev = np.asarray(sol.x, dtype=float)
        il_currents = np.asarray(state["currents"], dtype=float)
        il_currents_safe = np.asarray(state["currents_safe"], dtype=float)
        il_focals = np.asarray(state["il_focals"], dtype=float)
        a_pred = float(state["magnification_abs"])
        b_pred = float(state["B"])
        psi_pred = float(state["psi_total"])
        i_pl1 = float(state["projector_current_at"])
        f_pl1 = float(state["projector_focal_effective_m"])
        ffp_err = float(state["ffp_error_m"])
        mag_err_log = float(np.log(max(a_pred, 1e-20)) - np.log(target))
        mag_err_pct = float(100.0 * (a_pred - target) / max(target, 1e-30))
        weak_ok = bool(np.all(il_focals >= weak_min_focal_m))
        i3_ok = bool(i_lo <= il_currents[2] <= i_hi)
        pl1_ok = bool((not variable_projector_current) or (pl1_lo <= i_pl1 <= pl1_hi))
        f_pl1_ok = bool(
            (projector_focal_bounds_m is None) or (f_pl1_lo <= f_pl1 <= f_pl1_hi)
        )
        ffp_ok = bool((ffp_weight <= 0.0) or (abs(ffp_err) <= float(ffp_tolerance_m)))
        psi_tol = 1e-9 if exact_il_rotation_balance else float(psi_tolerance_rad)
        row_success = bool(
            abs(mag_err_log) <= mag_tol_log
            and abs(b_pred) <= b_tolerance_m
            and abs(psi_pred) <= psi_tol
            and weak_ok
            and i3_ok
            and pl1_ok
            and f_pl1_ok
            and ffp_ok
        )
        if row_success:
            success_count += 1

        fit_table.append(
            {
                "magnification_target": target,
                "magnification_predicted": float(a_pred),
                "magnification_error_pct": mag_err_pct,
                "B": float(b_pred),
                "psi_total": float(psi_pred),
                "I0_IL1_at": float(il_currents[0]),
                "I0_IL2_at": float(il_currents[1]),
                "I0_IL3_at": float(il_currents[2]),
                "I0_PL1_at": float(i_pl1),
                "f_IL1_m": float(il_focals[0]),
                "f_IL2_m": float(il_focals[1]),
                "f_IL3_m": float(il_focals[2]),
                "f_PL1_m": float(f_pl1),
                "success": row_success,
                "weak_constraint_ok": weak_ok,
                "i3_bounds_ok": i3_ok,
                "pl1_bounds_ok": pl1_ok,
                "projector_focal_bounds_ok": f_pl1_ok,
                "ffp_error_m": float(ffp_err),
                "ffp_constraint_ok": ffp_ok,
                "residual_norm": float(np.linalg.norm(make_residual(used_b_weight)(sol.x))),
                "used_b_weight": float(used_b_weight),
                "used_relaxed_b_weight": bool(used_relaxed_b),
                "solver_nfev": int(sol.nfev),
                "solver_success": bool(sol.success),
                "solver_message": str(sol.message),
            }
        )
        prev_currents = il_currents_safe
        prev_target = target

    solver_info = {
        "n_settings": int(len(mags)),
        "n_success": int(success_count),
        "success_fraction": float(success_count / len(mags)),
        "mag_tolerance_pct": float(mag_tolerance_pct),
        "b_tolerance_m": float(b_tolerance_m),
        "weak_min_focal_m": float(weak_min_focal_m),
        "variable_projector_current": bool(variable_projector_current),
        "projector_current_bounds_at": [float(pl1_lo), float(pl1_hi)],
        "projector_focal_bounds_m": (
            None if projector_focal_bounds_m is None else [float(f_pl1_lo), float(f_pl1_hi)]
        ),
        "projector_focal_bounds_weight": float(projector_focal_bounds_weight),
        "exact_il_rotation_balance": bool(exact_il_rotation_balance),
        "psi_weight": float(psi_weight),
        "psi_tolerance_rad": float(psi_tolerance_rad),
        "trend_band_c_magnification": (
            None if trend_band_c_magnification is None else [float(band_lo), float(band_hi)]
        ),
        "trend_band_c_weight": float(trend_band_c_weight),
        "ffp_weight": float(ffp_weight),
        "ffp_tolerance_m": float(ffp_tolerance_m),
        "solver_method": method,
        "n_starts_per_setting": int(n_starts_per_setting),
        "b_weight": float(b_weight),
        "b_weight_relaxed": float(b_weight_relaxed),
        "n_relaxed_rows": int(relaxed_count),
    }

    return SimplifiedProjectorModel(
        model_type="tem_projector_simplified_zoom_v1",
        schema_version=1,
        voltage_v=float(voltage_v),
        rc_rad_per_at=float(rc),
        geometry=geometry,
        objective_focal_length_m=float(objective_focal_length_m),
        projector_focal_length_m=float(projector_focal_length_m),
        objective_current_at=float(objective_current_at),
        projector_current_at=float(projector_current_at),
        il_gc=np.asarray(il_gc, dtype=float),
        il_rotation_signs=np.asarray(il_s, dtype=float),
        il_current_bounds_at=(i_lo, i_hi),
        weak_min_focal_m=float(weak_min_focal_m),
        fit_table=fit_table,
        solver_info=solver_info,
        variable_projector_current=bool(variable_projector_current),
        projector_current_bounds_at=(pl1_lo, pl1_hi),
    )


def realize_simplified_projector_setting(
    model: SimplifiedProjectorModel,
    magnification: float,
    *,
    mode: str = "interpolate",
) -> dict[str, Any]:
    mags = np.asarray([row["magnification_target"] for row in model.fit_table], dtype=float)
    log_mags = np.log10(mags)
    request = float(magnification)
    mag_clip = float(np.clip(request, float(np.min(mags)), float(np.max(mags))))
    s = np.asarray(model.il_rotation_signs, dtype=float)
    exact_il_balance = bool(model.solver_info.get("exact_il_rotation_balance", True))

    i1_table = np.asarray([row["I0_IL1_at"] for row in model.fit_table], dtype=float)
    i2_table = np.asarray([row["I0_IL2_at"] for row in model.fit_table], dtype=float)
    i3_table = np.asarray([row["I0_IL3_at"] for row in model.fit_table], dtype=float)
    i_pl1_table = np.asarray(
        [float(row.get("I0_PL1_at", model.projector_current_at)) for row in model.fit_table],
        dtype=float,
    )

    if mode == "nearest":
        idx = int(np.argmin(np.abs(mags - request)))
        i1 = float(i1_table[idx])
        i2 = float(i2_table[idx])
        i3 = float(i3_table[idx])
        i_pl1 = float(i_pl1_table[idx])
    elif mode in {"interpolate", "solve"}:
        q = np.log10(mag_clip)
        i1 = float(np.interp(q, log_mags, i1_table))
        i2 = float(np.interp(q, log_mags, i2_table))
        i3 = float(np.interp(q, log_mags, i3_table))
        i_pl1 = float(np.interp(q, log_mags, i_pl1_table))
    else:
        raise ValueError("mode must be 'interpolate', 'nearest', or 'solve'.")

    i_lo, i_hi = model.il_current_bounds_at
    variable_pl1 = bool(
        getattr(model, "variable_projector_current", False)
        or model.solver_info.get("variable_projector_current", False)
    )
    if model.projector_current_bounds_at is not None:
        pl1_lo, pl1_hi = model.projector_current_bounds_at
    else:
        pl1_b = model.solver_info.get("projector_current_bounds_at", [i_lo, i_hi])
        pl1_lo, pl1_hi = float(pl1_b[0]), float(pl1_b[1])

    x0_il_values = np.array([i1, i2] if exact_il_balance else [i1, i2, i3], dtype=float)
    x0_il = _bounded_to_unconstrained(x0_il_values, i_lo, i_hi)
    if variable_pl1:
        x0_pl1 = _bounded_to_unconstrained(np.array([i_pl1], dtype=float), pl1_lo, pl1_hi)
        x0 = np.concatenate([x0_il, x0_pl1], axis=0)
    else:
        x0 = x0_il

    b_weight = float(model.solver_info.get("b_weight", 1.0))
    b_weight_relaxed = float(model.solver_info.get("b_weight_relaxed", 0.05))
    ffp_weight = float(model.solver_info.get("ffp_weight", 0.0))
    psi_weight = float(model.solver_info.get("psi_weight", 0.1))
    psi_tolerance_rad = float(model.solver_info.get("psi_tolerance_rad", 1e-9))
    trend_band = model.solver_info.get("trend_band_c_magnification", None)
    trend_band_tuple = (
        None
        if trend_band is None
        else (float(trend_band[0]), float(trend_band[1]))
    )
    trend_weight = float(model.solver_info.get("trend_band_c_weight", 0.0))
    pl1_focal_bounds = model.solver_info.get("projector_focal_bounds_m", None)
    pl1_focal_bounds_tuple = (
        None
        if pl1_focal_bounds is None
        else (float(pl1_focal_bounds[0]), float(pl1_focal_bounds[1]))
    )
    pl1_focal_bounds_weight = float(model.solver_info.get("projector_focal_bounds_weight", 1.0))
    solve_method = str(model.solver_info.get("solver_method", "lm")).lower().strip()
    n_starts = int(model.solver_info.get("n_starts_per_setting", 6))
    mag_tol_pct = float(model.solver_info.get("mag_tolerance_pct", 2.0))
    mag_tol_log = np.log(1.0 + max(mag_tol_pct, 0.0) / 100.0)
    projector_gc = 1.0 / max(
        float(model.projector_focal_length_m) * float(model.projector_current_at) ** 2,
        1e-30,
    )

    def make_residual(weight_b: float):
        return lambda x: _simplified_setting_residual(
            unconstrained_free=x,
            target_magnification=mag_clip,
            il_rotation_signs=s,
            exact_il_rotation_balance=exact_il_balance,
            il_current_bounds_at=(i_lo, i_hi),
            il_gc=np.asarray(model.il_gc, dtype=float),
            objective_focal_length_m=model.objective_focal_length_m,
            projector_focal_length_m=model.projector_focal_length_m,
            projector_gc=projector_gc,
            variable_projector_current=variable_pl1,
            projector_current_bounds_at=(pl1_lo, pl1_hi),
            projector_current_at_fixed=float(model.projector_current_at),
            geometry=model.geometry,
            rc_rad_per_at=model.rc_rad_per_at,
            weak_min_focal_m=model.weak_min_focal_m,
            total_length_m=max(model.geometry.total_length_m(), 1e-12),
            prev_currents=None,
            prev_target_magnification=None,
            smoothness_weight=0.0,
            trend_band_c_magnification=trend_band_tuple,
            trend_band_c_weight=trend_weight,
            psi_weight=psi_weight,
            projector_focal_bounds_m=pl1_focal_bounds_tuple,
            projector_focal_bounds_weight=pl1_focal_bounds_weight,
            b_weight=weight_b,
            ffp_weight=ffp_weight,
        )

    # Keep realization physically consistent with target magnification and B~0.
    solve_enabled = mode in {"interpolate", "solve"}
    if solve_enabled:
        residual_primary = make_residual(b_weight)
        sol, _ = _solve_setting_with_restarts(
            residual_primary,
            x0,
            method=solve_method,
            max_nfev=800,
            n_starts=max(1, n_starts // 2),
        )
        state = _evaluate_simplified_setting(
            unconstrained_free=sol.x,
            il_rotation_signs=s,
            exact_il_rotation_balance=exact_il_balance,
            il_current_bounds_at=(i_lo, i_hi),
            il_gc=np.asarray(model.il_gc, dtype=float),
            objective_focal_length_m=model.objective_focal_length_m,
            projector_focal_length_m=model.projector_focal_length_m,
            projector_gc=projector_gc,
            variable_projector_current=variable_pl1,
            projector_current_bounds_at=(pl1_lo, pl1_hi),
            projector_current_at_fixed=float(model.projector_current_at),
            geometry=model.geometry,
            rc_rad_per_at=model.rc_rad_per_at,
        )
        mag_err_log = float(np.log(max(state["magnification_abs"], 1e-20)) - np.log(max(mag_clip, 1e-20)))
        if abs(mag_err_log) > mag_tol_log and b_weight_relaxed < b_weight:
            residual_relaxed = make_residual(b_weight_relaxed)
            sol_relaxed, _ = _solve_setting_with_restarts(
                residual_relaxed,
                x0,
                method=solve_method,
                max_nfev=800,
                n_starts=max(1, n_starts // 2),
            )
            state_relaxed = _evaluate_simplified_setting(
                unconstrained_free=sol_relaxed.x,
                il_rotation_signs=s,
                exact_il_rotation_balance=exact_il_balance,
                il_current_bounds_at=(i_lo, i_hi),
                il_gc=np.asarray(model.il_gc, dtype=float),
                objective_focal_length_m=model.objective_focal_length_m,
                projector_focal_length_m=model.projector_focal_length_m,
                projector_gc=projector_gc,
                variable_projector_current=variable_pl1,
                projector_current_bounds_at=(pl1_lo, pl1_hi),
                projector_current_at_fixed=float(model.projector_current_at),
                geometry=model.geometry,
                rc_rad_per_at=model.rc_rad_per_at,
            )
            mag_err_relaxed = float(
                np.log(max(state_relaxed["magnification_abs"], 1e-20)) - np.log(max(mag_clip, 1e-20))
            )
            if abs(mag_err_relaxed) < abs(mag_err_log):
                state = state_relaxed
        il_currents = np.asarray(state["currents"], dtype=float)
    else:
        state = _evaluate_simplified_setting(
            unconstrained_free=x0,
            il_rotation_signs=s,
            exact_il_rotation_balance=exact_il_balance,
            il_current_bounds_at=(i_lo, i_hi),
            il_gc=np.asarray(model.il_gc, dtype=float),
            objective_focal_length_m=model.objective_focal_length_m,
            projector_focal_length_m=model.projector_focal_length_m,
            projector_gc=projector_gc,
            variable_projector_current=variable_pl1,
            projector_current_bounds_at=(pl1_lo, pl1_hi),
            projector_current_at_fixed=float(model.projector_current_at),
            geometry=model.geometry,
            rc_rad_per_at=model.rc_rad_per_at,
        )
        il_currents = np.asarray(state["currents"], dtype=float)

    il_currents_safe = np.asarray(state["currents_safe"], dtype=float)
    il_focals = np.asarray(state["il_focals"], dtype=float)
    i_pl1_out = float(state["projector_current_at"])
    f_pl1_out = float(state["projector_focal_effective_m"])
    focals = _focals_with_fixed_strong_lenses(
        objective_focal_length_m=model.objective_focal_length_m,
        il_focals_m=il_focals,
        projector_focal_length_m=f_pl1_out,
    )
    mtx = _build_abcd_np(dists=model.geometry.drifts_m(), focals=focals)
    a_pred = abs(float(mtx[0, 0]))
    b_pred = float(mtx[0, 1])
    psi_pred = float(model.rc_rad_per_at * np.dot(s, il_currents))

    return {
        "magnification_request": request,
        "magnification_clipped": mag_clip,
        "mode": mode,
        "currents_il_at": il_currents,
        "current_pl1_at": float(i_pl1_out),
        "focals_il_m": il_focals,
        "focal_pl1_m": float(f_pl1_out),
        "magnification_predicted": float(a_pred),
        "B": float(b_pred),
        "psi_total": float(psi_pred),
        "ffp_error_m": float(state["ffp_error_m"]),
        "i3_bounds_ok": bool(
            model.il_current_bounds_at[0] <= il_currents[2] <= model.il_current_bounds_at[1]
        ),
        "pl1_bounds_ok": bool((not variable_pl1) or (pl1_lo <= i_pl1_out <= pl1_hi)),
        "projector_focal_bounds_ok": bool(
            (pl1_focal_bounds_tuple is None)
            or (pl1_focal_bounds_tuple[0] <= f_pl1_out <= pl1_focal_bounds_tuple[1])
        ),
        "psi_tolerance_ok": bool(
            abs(psi_pred) <= (1e-9 if exact_il_balance else psi_tolerance_rad)
        ),
        "weak_constraint_ok": bool(np.all(il_focals >= model.weak_min_focal_m)),
    }


def build_simplified_projector_components(
    model: SimplifiedProjectorModel,
    magnification: float,
    *,
    z0: float = 0.0,
    mode: str = "interpolate",
    include_detector_plane: bool = True,
) -> tuple[Any, ...]:
    realized = realize_simplified_projector_setting(model=model, magnification=magnification, mode=mode)
    il_currents = np.asarray(realized["currents_il_at"], dtype=float)
    pl1_current = float(realized.get("current_pl1_at", model.projector_current_at))
    il_gc = np.asarray(model.il_gc, dtype=float)
    il_signs = np.asarray(model.il_rotation_signs, dtype=float)

    g = model.geometry
    d = g.drifts_m()
    z_opl = z0 + d[0]
    z_il1 = z_opl + d[1]
    z_il2 = z_il1 + d[2]
    z_il3 = z_il2 + d[3]
    z_pl1 = z_il3 + d[4]
    z_det = z_pl1 + d[5]

    gc_opl = 1.0 / (model.objective_focal_length_m * model.objective_current_at**2)
    gc_pl1 = 1.0 / (model.projector_focal_length_m * model.projector_current_at**2)
    comps: list[Any] = [
        ElectromagneticLens(
            z=float(z_opl),
            I0=float(model.objective_current_at),
            Gc=float(gc_opl),
            Rc=0.0,
        ),
        ElectromagneticLens(
            z=float(z_il1),
            I0=float(il_currents[0]),
            Gc=float(il_gc[0]),
            Rc=float(model.rc_rad_per_at * il_signs[0]),
        ),
        ElectromagneticLens(
            z=float(z_il2),
            I0=float(il_currents[1]),
            Gc=float(il_gc[1]),
            Rc=float(model.rc_rad_per_at * il_signs[1]),
        ),
        ElectromagneticLens(
            z=float(z_il3),
            I0=float(il_currents[2]),
            Gc=float(il_gc[2]),
            Rc=float(model.rc_rad_per_at * il_signs[2]),
        ),
        ElectromagneticLens(
            z=float(z_pl1),
            I0=pl1_current,
            Gc=float(gc_pl1),
            Rc=0.0,
        ),
    ]
    if include_detector_plane:
        comps.append(Plane(z=float(z_det)))
    return tuple(comps)


def export_simplified_projector_model_json(
    model: SimplifiedProjectorModel,
    path: str | Path,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model.as_json_dict(), indent=2))
    return out


def load_simplified_projector_model_json(path: str | Path) -> SimplifiedProjectorModel:
    payload = json.loads(Path(path).read_text())
    geometry = ProjectorGeometry(**payload["geometry"])
    return SimplifiedProjectorModel(
        model_type=str(payload["model_type"]),
        schema_version=int(payload["schema_version"]),
        voltage_v=float(payload["voltage_v"]),
        rc_rad_per_at=float(payload["rc_rad_per_at"]),
        geometry=geometry,
        objective_focal_length_m=float(payload["objective_focal_length_m"]),
        projector_focal_length_m=float(payload["projector_focal_length_m"]),
        objective_current_at=float(payload["objective_current_at"]),
        projector_current_at=float(payload["projector_current_at"]),
        il_gc=np.asarray(payload["il_gc"], dtype=float),
        il_rotation_signs=np.asarray(payload["il_rotation_signs"], dtype=float),
        il_current_bounds_at=(
            float(payload["il_current_bounds_at"][0]),
            float(payload["il_current_bounds_at"][1]),
        ),
        weak_min_focal_m=float(payload["weak_min_focal_m"]),
        fit_table=list(payload["fit_table"]),
        solver_info=dict(payload["solver_info"]),
        variable_projector_current=bool(payload.get("variable_projector_current", False)),
        projector_current_bounds_at=(
            None
            if payload.get("projector_current_bounds_at", None) is None
            else (
                float(payload["projector_current_bounds_at"][0]),
                float(payload["projector_current_bounds_at"][1]),
            )
        ),
    )


__all__ = [
    "DEFAULT_PROJECTOR_LENS_NAMES",
    "ProjectorGeometry",
    "HexProjectorDataset",
    "ProjectorFitParams",
    "ProjectorModel",
    "parse_projector_hex_table",
    "simulate_projector_response",
    "fit_projector_model",
    "realize_projector_setting",
    "projector_metrics",
    "build_projector_components",
    "export_projector_model_json",
    "load_projector_model_json",
    "SimplifiedProjectorModel",
    "solve_simplified_projector_zoom",
    "realize_simplified_projector_setting",
    "build_simplified_projector_components",
    "export_simplified_projector_model_json",
    "load_simplified_projector_model_json",
]
