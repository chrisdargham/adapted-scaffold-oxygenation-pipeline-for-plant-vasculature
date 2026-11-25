#!/usr/bin/env python3
"""
oxy_sim_netflow.py
==================
Run oxy_sim.py using a NetFlow-derived hemodynamic cache produced by
netflow_to_oxysim_bridge_final.py.

This wrapper deliberately bypasses:
  - oxy_sim.py image segmentation
  - oxy_sim.py internal graph_poiseuille_flows()

Instead it injects the NetFlow cache into oxy_sim.run_simulation().
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from skimage.morphology import binary_dilation, disk


def load_oxy_module(path: Path) -> types.ModuleType:
    """
    Load oxy_sim.py as a module even if the file ends with a bare main() call.
    The uploaded version calls main() unconditionally, so normal import is unsafe.
    """
    src = path.read_text(encoding="utf-8", errors="replace")
    src = re.sub(r"\n\s*main\(\)\s*$", "\n# main() suppressed by oxy_sim_netflow.py\n", src)
    module = types.ModuleType("oxy_sim_imported")
    module.__file__ = str(path)
    module.__name__ = "oxy_sim_imported"
    code = compile(src, str(path), "exec")
    exec(code, module.__dict__)
    return module


def neighbors8(x: int, y: int, nx: int, ny: int) -> Iterable[Tuple[int, int]]:
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            xx, yy = x + dx, y + dy
            if 0 <= xx < nx and 0 <= yy < ny:
                yield xx, yy


def required_param_dict(cfg: Dict[str, Any]) -> Dict[str, Any]:
    required = [
        "Lx", "Ly", "nx", "ny", "T", "nt", "step",
        "D", "Vmax", "Km", "enable_pasteur", "K_pasteur", "beta_pasteur",
        "P_wall", "min_radius_px", "exchange_clamp", "radius_scale",
        "alpha", "mu_blood", "P_in", "P_out", "single_io", "u_min",
        "P50", "hill_n", "Hb_capacity", "pO2_max",
        "rebuild_every", "rebuild_tol", "band_px", "check_every", "tol",
        "stop_on_converge", "tol_abs", "min_checks", "max_minutes",
        "stop_band_only", "enable_death_model", "Ccrit",
    ]
    out = {k: cfg[k] for k in required}
    if cfg.get("inlet_concentration", None) is not None:
        out["inlet_concentration"] = cfg["inlet_concentration"]
        out["inlet_mmHg"] = cfg.get("inlet_mmHg", None)
    elif cfg.get("inlet_mmHg", None) is not None:
        out["inlet_mmHg"] = cfg["inlet_mmHg"]
        out["inlet_concentration"] = None
    else:
        raise ValueError("Config must define inlet_concentration or inlet_mmHg.")
    return out


def load_hemo_cache(path: Path) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    required = [
        "vessel_mask", "skel", "speed_map", "pressures", "radius_m", "inlet_mask",
        "reachable", "perfused_vessel", "transit_time", "lift_ix", "lift_iy",
    ]
    missing = [k for k in required if k not in z]
    if missing:
        raise RuntimeError(f"Hemo cache is missing required arrays: {missing}")
    return {
        "vessel_mask": z["vessel_mask"].astype(bool),
        "skel": z["skel"].astype(bool),
        "speed_map": z["speed_map"].astype(np.float32),
        "pressures": z["pressures"].astype(np.float32),
        "radius_m": z["radius_m"].astype(np.float32),
        "inlet_mask": z["inlet_mask"].astype(bool),
        "outlet_mask": z["outlet_mask"].astype(bool) if "outlet_mask" in z else np.zeros_like(z["skel"], dtype=bool),
        "reachable": z["reachable"].astype(bool),
        "perfused_vessel": z["perfused_vessel"].astype(bool),
        "transit_time": z["transit_time"].astype(np.float64),
        "lift_indices": (z["lift_ix"].astype(np.int32), z["lift_iy"].astype(np.int32)),
        "lift_ix": z["lift_ix"].astype(np.int32),
        "lift_iy": z["lift_iy"].astype(np.int32),
        "Lx_m": float(z["Lx_m"]) if "Lx_m" in z else None,
        "Ly_m": float(z["Ly_m"]) if "Ly_m" in z else None,
    }


def make_initializer(params, vessel_mask):
    C = np.zeros((params.nx, params.ny), dtype=np.float32)
    D = np.full((params.nx, params.ny), params.D, dtype=np.float32)
    alphax = D * params.dt / (params.dx ** 2)
    alphay = D * params.dt / (params.dy ** 2)
    betax = np.zeros_like(D, dtype=np.float32)
    betay = np.zeros_like(D, dtype=np.float32)
    return C, D, alphax, alphay, betax, betay


def compute_metrics(oxy, C_final, C_lumen, vessel_mask, Vmax_map, k_map, params, hemo_cache, cfg):
    tissue_mask = ~vessel_mask
    tissue = C_final[tissue_mask]
    alpha = params.alpha_mM_per_mmHg
    pO2 = C_final / max(alpha, 1e-30)
    tissue_pO2 = pO2[tissue_mask]

    rngs = oxy.field_ranges(C_final, C_lumen, vessel_mask, params)
    afs = oxy.area_fractions(C_final, params, vessel_mask, thresholds_mmHg=(15.0, 20.0))
    vis = oxy.viability_summary(
        C_final, params, vessel_mask,
        threshold_mM=float(cfg.get("viability_mM", 0.0013)),
        human_cell_area_um2=float(cfg.get("human_cell_area_um2", 250.0)),
    )
    tdiff_map, tdeath_map = oxy.oxygen_timing_diagnostics(vessel_mask, params, Dmap=None, Vmax_map=Vmax_map)

    dx, dy = params.dx, params.dy
    area = dx * dy
    uptake = Vmax_map * C_final / (params.Km + C_final + 1e-12)
    exchange = k_map * (C_lumen - C_final)
    total_uptake = float(np.sum(uptake[tissue_mask]) * area)
    total_exchange = float(np.sum(exchange[vessel_mask]) * area)
    mass_residual_pct = 100.0 * abs(total_exchange - total_uptake) / max(abs(total_uptake), 1e-30)

    out = {
        "tissue_min_mM": rngs["tissue_min"],
        "tissue_median_mM": rngs["tissue_median"],
        "tissue_max_mM": rngs["tissue_max"],
        "lumen_min_mM": rngs["lumen_min"],
        "lumen_median_mM": rngs["lumen_median"],
        "lumen_max_mM": rngs["lumen_max"],
        "PbtO2_ge_15_mmHg_pct": afs[15.0],
        "PbtO2_ge_20_mmHg_pct": afs[20.0],
        "viable_tissue_pct": vis["Percent of viable tissue (%)"],
        "viable_tissue_area_mm2": vis["Viable tissue area (mm^2)"],
        "total_tissue_area_mm2": vis["Total tissue area (mm^2)"],
        "tdiff_max_s": float(np.nanmax(tdiff_map[tissue_mask])) if np.any(tissue_mask) else 0.0,
        "tdeath_max_s": float(np.nanmax(tdeath_map[tissue_mask])) if np.any(tissue_mask) else float("inf"),
        "pO2_p5_mmHg": float(np.percentile(tissue_pO2, 5)) if tissue_pO2.size else 0.0,
        "pO2_p10_mmHg": float(np.percentile(tissue_pO2, 10)) if tissue_pO2.size else 0.0,
        "pO2_p25_mmHg": float(np.percentile(tissue_pO2, 25)) if tissue_pO2.size else 0.0,
        "pO2_p50_mmHg": float(np.percentile(tissue_pO2, 50)) if tissue_pO2.size else 0.0,
        "pO2_p75_mmHg": float(np.percentile(tissue_pO2, 75)) if tissue_pO2.size else 0.0,
        "pO2_p90_mmHg": float(np.percentile(tissue_pO2, 90)) if tissue_pO2.size else 0.0,
        "pO2_p95_mmHg": float(np.percentile(tissue_pO2, 95)) if tissue_pO2.size else 0.0,
        "anoxic_lt1_pct": float(np.mean(tissue_pO2 < 1.0) * 100.0) if tissue_pO2.size else 0.0,
        "severe_hypoxic_1_5_pct": float(np.mean((tissue_pO2 >= 1.0) & (tissue_pO2 < 5.0)) * 100.0) if tissue_pO2.size else 0.0,
        "mild_hypoxic_5_30_pct": float(np.mean((tissue_pO2 >= 5.0) & (tissue_pO2 < 30.0)) * 100.0) if tissue_pO2.size else 0.0,
        "normoxic_ge30_pct": float(np.mean(tissue_pO2 >= 30.0) * 100.0) if tissue_pO2.size else 0.0,
        "exchange_integral_mM_m2_s": total_exchange,
        "consumption_integral_mM_m2_s": total_uptake,
        "mass_balance_residual_pct": mass_residual_pct,
        "cv_pO2": float(np.std(tissue_pO2) / max(np.mean(tissue_pO2), 1e-30)) if tissue_pO2.size else 0.0,
        "skeleton_pixels": int(np.sum(hemo_cache["skel"])),
        "reachable_skeleton_pixels": int(np.sum(hemo_cache["reachable"])),
        "reachable_skeleton_pct": float(np.sum(hemo_cache["reachable"]) / max(1, np.sum(hemo_cache["skel"])) * 100.0),
        "perfused_vessel_pct": float(np.sum(hemo_cache["perfused_vessel"]) / max(1, np.sum(vessel_mask)) * 100.0),
        "inlet_pixels": int(np.sum(hemo_cache["inlet_mask"])),
        "outlet_pixels": int(np.sum(hemo_cache.get("outlet_mask", np.zeros_like(vessel_mask)))),
        "P_wall": float(params.P_wall),
        "Vmax": float(params.Vmax),
        "alpha": float(alpha),
        "Lx_m": float(params.Lx),
        "Ly_m": float(params.Ly),
        "nx": int(params.nx),
        "ny": int(params.ny),
    }
    return out


def write_summary(path: Path, metrics: Dict[str, Any]) -> None:
    lines = [
        "=== OxySim-NetFlow Summary ===",
        f"PbtO2 ≥ 15 mmHg (%): {metrics['PbtO2_ge_15_mmHg_pct']:.3f}",
        f"PbtO2 ≥ 20 mmHg (%): {metrics['PbtO2_ge_20_mmHg_pct']:.3f}",
        f"Median tissue pO2 (mmHg): {metrics['pO2_p50_mmHg']:.3f}",
        f"Viable tissue (%): {metrics['viable_tissue_pct']:.3f}",
        f"Perfused vessel (%): {metrics['perfused_vessel_pct']:.3f}",
        f"Reachable skeleton (%): {metrics['reachable_skeleton_pct']:.3f}",
        f"Mass-balance residual (%): {metrics['mass_balance_residual_pct']:.3f}",
        "",
        "Interpretation rule: if reachable skeleton is low or mass-balance residual is high, this run is a technical failure, not a biological result.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    outdir = Path(args.outdir or cfg.get("outdir", "oxy_result"))
    outdir.mkdir(parents=True, exist_ok=True)

    oxy = load_oxy_module(Path(args.oxy_sim))
    # Critical: OxySim lumen traversal must use 8-connectivity because bridge skeletons are Bresenham-rasterized.
    oxy._neighbors_4 = neighbors8

    hemo = load_hemo_cache(Path(args.hemo))
    cfg["nx"] = int(cfg.get("nx") or hemo["vessel_mask"].shape[0])
    cfg["ny"] = int(cfg.get("ny") or hemo["vessel_mask"].shape[1])
    cfg["Lx"] = float(cfg.get("Lx") or hemo["Lx_m"])
    cfg["Ly"] = float(cfg.get("Ly") or hemo["Ly_m"])
    cfg["outdir"] = str(outdir)

    params = oxy.SimulationParams(**required_param_dict(cfg))
    if hemo["vessel_mask"].shape != (params.nx, params.ny):
        raise RuntimeError(
            f"Hemo grid {hemo['vessel_mask'].shape} does not match config grid {(params.nx, params.ny)}. "
            "Regenerate bridge with the same --grid-nx/--grid-ny or update config."
        )

    # CFL must be enforced before coefficients are constructed.
    D_pre = np.full((params.nx, params.ny), params.D, dtype=np.float32)
    oxy.enforce_cfl(params, D_pre)
    initializer = make_initializer(params, hemo["vessel_mask"])

    vessel_mask = hemo["vessel_mask"].astype(bool)
    r_map_m = hemo["radius_m"].astype(np.float32) * float(params.radius_scale)
    r_map_m[vessel_mask & (r_map_m <= 0)] = np.nanmedian(r_map_m[vessel_mask & (r_map_m > 0)]) if np.any(vessel_mask & (r_map_m > 0)) else params.dx
    k_map, r_map_m = oxy.estimate_exchange_rate_radius_aware(
        vessel_mask, params, dt=params.dt, safety=params.exchange_clamp, r_map_m=r_map_m,
    )
    Vmax_map = oxy.build_Vmax_map(params, vessel_mask, tissue_type_img_path=cfg.get("tissue_type_img"), type_to_multiplier=None)

    hemo_cache = {
        "speed_map": hemo["speed_map"],
        "inlet_mask": hemo["inlet_mask"],
        "reachable": hemo["reachable"],
        "perfused_vessel": hemo["perfused_vessel"],
        "pressures": hemo["pressures"],
        "transit_time": hemo["transit_time"],
        "skel": hemo["skel"],
        "lift_indices": hemo["lift_indices"],
    }

    # Expose plot dimensions to oxy_sim plotting helpers.
    class _P: pass
    oxy._GLOBAL_PARAMS_FOR_PLOTTING = _P()
    oxy._GLOBAL_PARAMS_FOR_PLOTTING.Lx = params.Lx
    oxy._GLOBAL_PARAMS_FOR_PLOTTING.Ly = params.Ly
    oxy._GLOBAL_PARAMS_FOR_PLOTTING.nx = params.nx
    oxy._GLOBAL_PARAMS_FOR_PLOTTING.ny = params.ny
    oxy._GLOBAL_CONFIG_FOR_PLOTTING = cfg

    oxy._test_hb_roundtrip(params)
    results, C_lumen = oxy.run_simulation(
        params, initializer, vessel_mask, Vmax_map, k_map, r_map_m, hemo_cache,
        check_every=params.check_every, tol=params.tol,
    )
    C_final = results[-1]

    oxy.plot_heatmap_readable(C_final, vessel_mask, params, "Final Oxygen Concentration (Tissue)", str(outdir / "fig_tissue_O2.png"))
    oxy.plot_heatmap_readable(C_lumen, vessel_mask, params, "Final Oxygen Concentration (Lumen)", str(outdir / "fig_lumen_O2.png"))
    oxy.plot_transit_time(hemo_cache["transit_time"], hemo_cache["skel"], str(outdir / "fig_transit_time.png"))
    cdf = oxy.plot_distance_cdf(vessel_mask, params, savepath=str(outdir / "fig_distance_cdf.png"))
    range_info, _ = oxy.compute_distance_hist_ranges(vessel_mask, params, bins_um=cfg.get("distance_bins_um", [0, 50, 100, 200, 1000]))
    oxy.plot_distance_ranges_bar(range_info, savepath=str(outdir / "fig_distance_ranges.png"))

    metrics = compute_metrics(oxy, C_final, C_lumen, vessel_mask, Vmax_map, k_map, params, hemo_cache, cfg)
    metrics["distance_ranges_bins_um"] = range_info["bins"]
    metrics["distance_ranges_labels"] = range_info["labels"]
    metrics["distance_ranges_percentages"] = range_info["percentages"]

    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, allow_nan=True)
    pd.DataFrame([metrics]).to_csv(outdir / "metrics.csv", index=False)
    write_summary(outdir / "summary.txt", metrics)

    np.savez_compressed(
        outdir / "fields.npz",
        C_tissue=C_final.astype(np.float32),
        C_lumen=C_lumen.astype(np.float32),
        vessel_mask=vessel_mask.astype(bool),
        k_map=k_map.astype(np.float32),
        Vmax_map=Vmax_map.astype(np.float32),
        radius_m=r_map_m.astype(np.float32),
        speed_map=hemo["speed_map"].astype(np.float32),
        pressures=hemo["pressures"].astype(np.float32),
        skel=hemo["skel"].astype(bool),
        reachable=hemo["reachable"].astype(bool),
        perfused_vessel=hemo["perfused_vessel"].astype(bool),
        transit_time=hemo["transit_time"].astype(np.float64),
        inlet_mask=hemo["inlet_mask"].astype(bool),
        outlet_mask=hemo["outlet_mask"].astype(bool),
        alpha=np.array(float(params.alpha_mM_per_mmHg)),
        Lx_m=np.array(float(params.Lx)),
        Ly_m=np.array(float(params.Ly)),
        dx_m=np.array(float(params.dx)),
        dy_m=np.array(float(params.dy)),
        P_wall=np.array(float(params.P_wall)),
        Vmax=np.array(float(params.Vmax)),
        Km=np.array(float(params.Km)),
    )

    # Save the actual config used, including bridge-derived dimensions.
    (outdir / "sim_config.used.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"[OXY] saved: {outdir}")
    print(f"[OXY] PbtO2>=15={metrics['PbtO2_ge_15_mmHg_pct']:.2f}% | median pO2={metrics['pO2_p50_mmHg']:.2f} mmHg | mass residual={metrics['mass_balance_residual_pct']:.2f}%")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--oxy-sim", type=Path, required=True, help="Path to oxy_sim.py")
    p.add_argument("--hemo", type=Path, required=True, help="Bridge hemo NPZ")
    p.add_argument("--config", type=Path, required=True, help="Simulation config JSON")
    p.add_argument("--outdir", type=Path, default=None)
    return p


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
