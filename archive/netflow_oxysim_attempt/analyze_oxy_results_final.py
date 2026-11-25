#!/usr/bin/env python3
"""
analyze_oxy_results_final.py
============================
Aggregate OxySim-NetFlow sweep outputs into:
  - aggregate_metrics.csv
  - summary_by_condition.csv

It reads each sweeps/*/fields.npz plus sim_config.json / metrics.json and computes
additional spatial oxygenation metrics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, label
from scipy.spatial import cKDTree


def read_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def parse_condition(run_dir: Path, cfg: dict, metrics: dict) -> dict:
    return {
        "image": run_dir.parents[1].name if len(run_dir.parents) >= 2 else "unknown",
        "condition": run_dir.name,
        "P_wall": float(cfg.get("P_wall", metrics.get("P_wall", np.nan))),
        "Vmax": float(cfg.get("Vmax", metrics.get("Vmax", np.nan))),
        "run_dir": str(run_dir),
    }


def patch_geometry(pO2: np.ndarray, tissue_mask: np.ndarray, threshold: float, pixel_area_um2: float) -> dict:
    hyp = (pO2 < threshold) & tissue_mask
    struct = np.ones((3, 3), dtype=np.uint8)
    lab, n = label(hyp.astype(np.uint8), structure=struct)
    if n == 0:
        return {
            f"patch_count_lt{threshold:g}": 0,
            f"largest_patch_area_um2_lt{threshold:g}": 0.0,
            f"mean_patch_area_um2_lt{threshold:g}": 0.0,
        }
    counts = np.bincount(lab.ravel())[1:]
    areas = counts.astype(float) * pixel_area_um2
    return {
        f"patch_count_lt{threshold:g}": int(n),
        f"largest_patch_area_um2_lt{threshold:g}": float(np.max(areas)),
        f"mean_patch_area_um2_lt{threshold:g}": float(np.mean(areas)),
    }


def morans_i(values: np.ndarray, coords: np.ndarray, max_points: int = 5000, k: int = 8) -> float:
    if values.size < k + 2:
        return float("nan")
    rng = np.random.default_rng(12345)
    if values.size > max_points:
        idx = rng.choice(values.size, size=max_points, replace=False)
        values = values[idx]
        coords = coords[idx]
    x = values.astype(float)
    xbar = float(np.mean(x))
    z = x - xbar
    denom = float(np.sum(z * z))
    if denom <= 0:
        return float("nan")
    tree = cKDTree(coords.astype(float))
    _, inds = tree.query(coords, k=min(k + 1, len(coords)))
    if inds.ndim == 1:
        return float("nan")
    num = 0.0
    wsum = 0.0
    for i in range(inds.shape[0]):
        for j in inds[i, 1:]:
            num += z[i] * z[j]
            wsum += 1.0
    if wsum == 0:
        return float("nan")
    return float((len(x) / wsum) * (num / denom))


def analyze_run(run_dir: Path) -> dict:
    fields_path = run_dir / "fields.npz"
    if not fields_path.exists():
        raise FileNotFoundError(fields_path)
    cfg = read_json(run_dir / "sim_config.used.json") or read_json(run_dir / "sim_config.json")
    metrics0 = read_json(run_dir / "metrics.json")
    z = np.load(fields_path, allow_pickle=False)

    C = z["C_tissue"].astype(float)
    C_lumen = z["C_lumen"].astype(float)
    vessel_mask = z["vessel_mask"].astype(bool)
    k_map = z["k_map"].astype(float)
    Vmax_map = z["Vmax_map"].astype(float)
    alpha = float(z["alpha"]) if "alpha" in z else float(cfg.get("alpha", 1.34e-3))
    Lx = float(z["Lx_m"]) if "Lx_m" in z else float(cfg["Lx"])
    Ly = float(z["Ly_m"]) if "Ly_m" in z else float(cfg["Ly"])
    Km = float(z["Km"]) if "Km" in z else float(cfg.get("Km", 1e-3))

    nx, ny = C.shape
    dx_m = Lx / max(nx - 1, 1)
    dy_m = Ly / max(ny - 1, 1)
    px_um = 0.5 * (dx_m + dy_m) * 1e6
    pixel_area_um2 = dx_m * dy_m * 1e12
    pixel_area_m2 = dx_m * dy_m

    tissue_mask = ~vessel_mask
    pO2 = C / max(alpha, 1e-30)
    vals = pO2[tissue_mask]

    out = parse_condition(run_dir, cfg, metrics0)
    for p in [5, 10, 25, 50, 75, 90, 95]:
        out[f"pO2_p{p}_mmHg"] = float(np.percentile(vals, p)) if vals.size else 0.0

    out.update({
        "anoxic_lt1_pct": float(np.mean(vals < 1.0) * 100.0) if vals.size else 0.0,
        "severe_hypoxic_1_5_pct": float(np.mean((vals >= 1.0) & (vals < 5.0)) * 100.0) if vals.size else 0.0,
        "mild_hypoxic_5_30_pct": float(np.mean((vals >= 5.0) & (vals < 30.0)) * 100.0) if vals.size else 0.0,
        "normoxic_ge30_pct": float(np.mean(vals >= 30.0) * 100.0) if vals.size else 0.0,
        "PbtO2_ge_15_mmHg_pct": float(np.mean(vals >= 15.0) * 100.0) if vals.size else 0.0,
        "PbtO2_ge_20_mmHg_pct": float(np.mean(vals >= 20.0) * 100.0) if vals.size else 0.0,
        "cv_pO2": float(np.std(vals) / max(np.mean(vals), 1e-30)) if vals.size else 0.0,
    })

    for thr in [1, 5, 15, 30]:
        out.update(patch_geometry(pO2, tissue_mask, float(thr), pixel_area_um2))

    dist_um = distance_transform_edt(tissue_mask) * px_um
    bins = [(0, 25), (25, 50), (50, 100), (100, 200), (200, np.inf)]
    for lo, hi in bins:
        m = tissue_mask & (dist_um >= lo) & (dist_um < hi)
        key = f"mean_pO2_dist_{int(lo)}_{'inf' if np.isinf(hi) else int(hi)}_um"
        out[key] = float(np.mean(pO2[m])) if np.any(m) else float("nan")

    uptake = Vmax_map * C / (Km + C + 1e-12)
    exchange = k_map * (C_lumen - C)
    total_uptake = float(np.sum(uptake[tissue_mask]) * pixel_area_m2)
    total_exchange = float(np.sum(exchange[vessel_mask]) * pixel_area_m2)
    out["exchange_integral_mM_m2_s"] = total_exchange
    out["consumption_integral_mM_m2_s"] = total_uptake
    out["mass_balance_residual_pct"] = 100.0 * abs(total_exchange - total_uptake) / max(abs(total_uptake), 1e-30)

    xs, ys = np.where(tissue_mask)
    coords = np.column_stack([xs * dx_m, ys * dy_m])
    out["morans_I_pO2"] = morans_i(vals, coords)

    # Pass through scalar metrics generated by oxy_sim_netflow.py.
    for k, v in metrics0.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            out.setdefault(k, v)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--summary", type=Path, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    run_dirs = sorted(p for p in args.results_dir.glob("*/sweeps/*") if (p / "fields.npz").exists())
    if not run_dirs:
        raise SystemExit(f"No fields.npz files found under {args.results_dir}/<image>/sweeps/<condition>/")
    rows = []
    failed = []
    for rd in run_dirs:
        try:
            rows.append(analyze_run(rd))
        except Exception as e:
            failed.append((str(rd), str(e)))
    df = pd.DataFrame(rows)
    out_path = args.output or (args.results_dir / "aggregate_metrics.csv")
    df.to_csv(out_path, index=False)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    summary_path = args.summary or (args.results_dir / "summary_by_condition.csv")
    if {"P_wall", "Vmax"}.issubset(df.columns):
        summ = df.groupby(["P_wall", "Vmax"])[numeric_cols].agg(["mean", "std"])
        summ.columns = ["_".join(c).strip("_") for c in summ.columns.to_flat_index()]
        summ.reset_index().to_csv(summary_path, index=False)
    else:
        pd.DataFrame().to_csv(summary_path, index=False)

    print(f"[OK] {out_path}")
    print(f"[OK] {summary_path}")
    if failed:
        print("Failures:")
        for rd, err in failed:
            print(f"  {rd}: {err}")


if __name__ == "__main__":
    main()
