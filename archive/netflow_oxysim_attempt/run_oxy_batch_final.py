#!/usr/bin/env python3
"""
run_oxy_batch_final.py
======================
End-to-end batch pipeline:

LeafVeinCNN .mat
  -> leafveincnn_to_netflow.py
  -> prepare_network.py
  -> NetFlowV2.exe
  -> netflow_to_oxysim_bridge_final.py
  -> oxy_sim_netflow.py
  -> aggregate_metrics.csv + summary_by_condition.csv

This replaces GreensV4 with OxySim while preserving NetFlowV2 hemodynamics.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


BASE_CONFIG: Dict[str, object] = {
    "label": "pbs_leaf_scaffold",
    "image": None,
    "outdir": None,
    "nx": 384,
    "ny": 384,
    "Lx": None,
    "Ly": None,
    "T": 0.5,
    "nt": 2000,
    "step": 100,
    "D": 1.5e-9,
    "Vmax": 1.0e-4,
    "Km": 1.0e-3,
    "enable_pasteur": True,
    "K_pasteur": 0.013,
    "beta_pasteur": 1.0,
    "P_wall": 3.0e-5,
    "min_radius_px": 0.5,
    "radius_scale": 1.0,
    "exchange_clamp": 0.9,
    "alpha": 1.34e-3,
    "inlet_concentration": 0.20,
    "inlet_mmHg": None,
    "mu_blood": 7.0e-4,
    "P_in": 40.0,
    "P_out": 35.0,
    "single_io": False,
    "u_min": 1.0e-4,
    "P50": 26.0,
    "hill_n": 2.7,
    "Hb_capacity": 0.0,
    "pO2_max": 500.0,
    "rebuild_every": 5,
    "rebuild_tol": 1e-3,
    "band_px": 2,
    "check_every": 50,
    "tol": 1e-6,
    "stop_on_converge": True,
    "tol_abs": None,
    "min_checks": 3,
    "max_minutes": 30,
    "stop_band_only": False,
    "tissue_type_img": None,
    "viability_mM": 0.0013,
    "human_cell_area_um2": 250.0,
    "enable_death_model": False,
    "Ccrit": 0.001,
    "distance_bins_um": [0, 25, 50, 100, 200, 1000],
    "verify": False,
}

P_WALL_SWEEP = [1.0e-6, 3.0e-5, 1.0e-4]
VMAX_SWEEP = [1.0e-5, 1.0e-4, 1.0e-3]


def q(s) -> str:
    s = str(s)
    return f'"{s}"' if (" " in s or "(" in s or ")" in s) else s


def run_cmd(cmd: List[object], cwd: Optional[Path] = None, dry_run: bool = False,
            log_path: Optional[Path] = None) -> int:
    cmd_str = " ".join(q(c) for c in cmd)
    print(f"  >> {cmd_str}")
    if dry_run:
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(f"DRY RUN: {cmd_str}\n", encoding="utf-8")
        return 0
    r = subprocess.run(cmd_str, cwd=str(cwd) if cwd else None, text=True,
                       shell=True, capture_output=True)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"COMMAND:\n{cmd_str}\n\nRETURN CODE: {r.returncode}\n\nSTDOUT:\n{r.stdout}\n\nSTDERR:\n{r.stderr}\n",
            encoding="utf-8", errors="replace",
        )
    if r.stdout:
        print(r.stdout[-4000:])
    if r.returncode != 0 and r.stderr:
        print(r.stderr[-4000:])
    return int(r.returncode)


def write_pbs_rheolparams(path: Path, viscosity_cp: float = 0.7) -> None:
    content = f"""0.964    6.98  -13.29            bifpar   (ignored when phaseseparation=0)
0.80  -0.075   -11.0   12.0       cpar     (ignored when varyviscosity=0)
6.0 -0.085 3.2 -2.44 -0.06 0.645 viscpar  (ignored when varyviscosity=0)
35000 1.e-10 1.95                 nitmax, tol, omega - linear solver
100 1.e-3 1.e-3                   nitmax1, qtol, hdtol - nonlinear solver
1.1   0.5                         optw, optlam
{viscosity_cp:.6f} {viscosity_cp:.6f} 55.0      constvisc,vplas,mcv - PBS/culture medium approximation
0.0                                consthd - no RBC hematocrit
0                                  varyviscosity OFF for PBS
0                                  phaseseparation OFF for PBS
"""
    path.write_text(content, encoding="utf-8")


def write_contourparams_from_network(network_dat: Path, dst: Path) -> None:
    lines = network_dat.read_text(encoding="utf-8", errors="replace").splitlines()
    alx = aly = 1000.0
    if len(lines) > 1:
        try:
            alx, aly = map(float, lines[1].split()[:2])
        except Exception:
            pass
    content = (
        f"0.0 0.0 0.0 xsl0\n"
        f"{alx:.6f} 0.0 0.0 xsl1\n"
        f"0.0 {aly:.6f} 0.0 xsl2\n"
    )
    dst.write_text(content, encoding="utf-8")


def parse_sweep(text: Optional[str], default: List[float]) -> List[float]:
    if not text:
        return default
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def condition_label(p_wall: float, vmax: float) -> str:
    return f"Pw{p_wall:.0e}_V{vmax:.0e}".replace("+0", "").replace("+", "")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mat-dir", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--netflow-dir", type=Path, required=True)
    p.add_argument("--netflow-exe", type=str, default="NetFlowV2.exe")
    p.add_argument("--exporter", type=Path, required=True)
    p.add_argument("--prepare", type=Path, required=True)
    p.add_argument("--bridge", type=Path, required=True)
    p.add_argument("--oxy-wrapper", type=Path, required=True)
    p.add_argument("--oxy-sim", type=Path, required=True)
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--grid-nx", type=int, default=384)
    p.add_argument("--grid-ny", type=int, default=384)
    p.add_argument("--resample-step", type=float, default=75.0)
    p.add_argument("--snap-px", type=int, default=15)
    p.add_argument("--inlet-rc", type=str, default=None)
    p.add_argument("--auto-inlet", action="store_true", default=True)
    p.add_argument("--no-auto-inlet", action="store_false", dest="auto_inlet")
    p.add_argument("--inlet-pressure", type=float, default=40.0)
    p.add_argument("--outlet-pressure", type=float, default=35.0)
    p.add_argument("--boundary-margin", type=float, default=250.0)
    p.add_argument("--lumen-fraction", type=float, default=0.5)
    p.add_argument("--diameter-cap", type=float, default=150.0)
    p.add_argument("--min-length", type=float, default=0.5)
    p.add_argument("--pbs-viscosity-cp", type=float, default=0.7)
    p.add_argument("--write-pbs-rheology", action="store_true", default=True)
    p.add_argument("--p-wall-sweep", type=str, default=None)
    p.add_argument("--vmax-sweep", type=str, default=None)
    p.add_argument("--single-condition", action="store_true")
    p.add_argument("--T", type=float, default=0.5)
    p.add_argument("--nt", type=int, default=2000)
    p.add_argument("--max-minutes", type=float, default=30.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--continue-on-error", action="store_true", default=True)
    return p


def main() -> None:
    args = build_parser().parse_args()
    mat_files = sorted(args.mat_dir.glob("*.mat"))
    if not mat_files:
        raise SystemExit(f"No .mat files found in {args.mat_dir}")

    p_wall_list = [3.0e-5] if args.single_condition else parse_sweep(args.p_wall_sweep, P_WALL_SWEEP)
    vmax_list = [1.0e-4] if args.single_condition else parse_sweep(args.vmax_sweep, VMAX_SWEEP)
    n_conditions = len(p_wall_list) * len(vmax_list)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    if args.write_pbs_rheology and not args.dry_run:
        write_pbs_rheolparams(args.netflow_dir / "RheolParams.dat", viscosity_cp=args.pbs_viscosity_cp)

    print("=" * 72)
    print(f"LeafCNN → NetFlowV2 → OxySim batch: {len(mat_files)} images × {n_conditions} conditions")
    print(f"Results: {args.results_dir}")
    print("=" * 72)

    failures = []
    aggregate_rows = []
    netflow_exe = (args.netflow_dir / args.netflow_exe).resolve()

    for i, mat_path in enumerate(mat_files, 1):
        image = mat_path.stem
        print(f"\n{'─' * 72}\n[{i}/{len(mat_files)}] {image}\n{'─' * 72}")
        img_dir = args.results_dir / image
        prep_dir = img_dir / "netflow_input"
        netflow_results_dir = img_dir / "netflow_results"
        bridge_dir = img_dir / "bridge"
        logs_dir = img_dir / "logs"
        for d in [prep_dir, netflow_results_dir, bridge_dir, logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        prep_cmd = [
            args.python, args.prepare,
            "--mat", mat_path,
            "--exporter", args.exporter,
            "--output-dir", prep_dir,
            "--inlet-pressure", args.inlet_pressure,
            "--outlet-pressure", args.outlet_pressure,
            "--lumen-fraction", args.lumen_fraction,
            "--diameter-cap", args.diameter_cap,
            "--min-length", args.min_length,
            "--boundary-outlets-only",
            "--boundary-margin-um", args.boundary_margin,
            "--resample-step-um", args.resample_step,
            "--snap-px", args.snap_px,
        ]
        if args.auto_inlet:
            prep_cmd.append("--auto-inlet")
        elif args.inlet_rc:
            prep_cmd += ["--inlet-rc", args.inlet_rc]
        else:
            failures.append((image, "no_inlet"))
            if not args.continue_on_error:
                break
            continue

        print("[1/4] prepare_network.py")
        rc = run_cmd(prep_cmd, dry_run=args.dry_run, log_path=logs_dir / "01_prepare_network.log")
        if rc != 0:
            failures.append((image, "prepare_network"))
            if not args.continue_on_error:
                break
            continue

        network_ready = prep_dir / "network_ready.dat"
        if not args.dry_run and not network_ready.exists():
            failures.append((image, "network_ready_missing"))
            continue

        print("[2/4] NetFlowV2")
        if not args.dry_run:
            shutil.copy2(network_ready, args.netflow_dir / "Network.dat")
            write_contourparams_from_network(network_ready, args.netflow_dir / "ContourParams.dat")
        rc = run_cmd([netflow_exe], cwd=args.netflow_dir, dry_run=args.dry_run, log_path=logs_dir / "02_netflow.log")
        if rc != 0:
            failures.append((image, "NetFlowV2"))
            if not args.continue_on_error:
                break
            continue

        current_dir = args.netflow_dir / "Current"
        networknew = current_dir / "NetworkNew.dat"
        if not args.dry_run:
            if not networknew.exists():
                failures.append((image, "NetworkNew_missing"))
                continue
            shutil.copy2(networknew, netflow_results_dir / "NetworkNew.dat")
            for fname in ["Results.out", "histogram-logflows.out", "histogram-logvelocities.out", "histogram-pressures.out"]:
                src = current_dir / fname
                if src.exists():
                    shutil.copy2(src, netflow_results_dir / fname)

        print("[3/4] NetFlow → OxySim bridge")
        bridge_input = netflow_results_dir / "NetworkNew.dat"
        bridge_cmd = [
            args.python, args.bridge,
            "--input", bridge_input,
            "--output-dir", bridge_dir,
            "--grid-nx", args.grid_nx,
            "--grid-ny", args.grid_ny,
            "--stem", image,
            "--u-min", BASE_CONFIG["u_min"],
            "--viscosity-cp", args.pbs_viscosity_cp,
        ]
        rc = run_cmd(bridge_cmd, dry_run=args.dry_run, log_path=logs_dir / "03_bridge.log")
        if rc != 0:
            failures.append((image, "bridge"))
            if not args.continue_on_error:
                break
            continue

        hemo_npz = bridge_dir / f"{image}_hemo.npz"
        meta_path = bridge_dir / f"{image}_meta.json"
        if not args.dry_run and (not hemo_npz.exists() or not meta_path.exists()):
            failures.append((image, "bridge_outputs_missing"))
            continue
        meta = {} if args.dry_run else read_json(meta_path)

        print(f"[4/4] OxySim sweep ({n_conditions} conditions)")
        for p_wall in p_wall_list:
            for vmax in vmax_list:
                label = condition_label(p_wall, vmax)
                cond_dir = img_dir / "sweeps" / label
                cond_dir.mkdir(parents=True, exist_ok=True)
                cfg = copy.deepcopy(BASE_CONFIG)
                cfg["label"] = f"{image}_{label}"
                cfg["outdir"] = str(cond_dir.resolve())
                cfg["nx"] = int(args.grid_nx)
                cfg["ny"] = int(args.grid_ny)
                cfg["Lx"] = float(meta.get("Lx_m", 0.0)) if meta else None
                cfg["Ly"] = float(meta.get("Ly_m", 0.0)) if meta else None
                cfg["P_wall"] = float(p_wall)
                cfg["Vmax"] = float(vmax)
                cfg["P_in"] = float(args.inlet_pressure)
                cfg["P_out"] = float(args.outlet_pressure)
                cfg["T"] = float(args.T)
                cfg["nt"] = int(args.nt)
                cfg["max_minutes"] = float(args.max_minutes)
                cfg_path = cond_dir / "sim_config.json"
                cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

                oxy_cmd = [
                    args.python, args.oxy_wrapper,
                    "--oxy-sim", args.oxy_sim,
                    "--hemo", hemo_npz,
                    "--config", cfg_path,
                    "--outdir", cond_dir,
                ]
                rc = run_cmd(oxy_cmd, dry_run=args.dry_run, log_path=logs_dir / f"04_oxy_{label}.log")
                if rc != 0:
                    failures.append((image, f"oxy_{label}"))
                    if not args.continue_on_error:
                        break
                    continue

                metrics_path = cond_dir / "metrics.json"
                if not args.dry_run and metrics_path.exists():
                    m = read_json(metrics_path)
                    row = {
                        "image": image,
                        "condition": label,
                        "P_wall": p_wall,
                        "Vmax": vmax,
                        "result_dir": str(cond_dir),
                        **{k: v for k, v in m.items() if isinstance(v, (int, float, str, bool)) or v is None},
                    }
                    row.update({
                        "n_active_segments": meta.get("n_active_segments"),
                        "mean_diameter_um_bridge": meta.get("mean_diameter_um"),
                        "mean_speed_mm_s_bridge": meta.get("mean_speed_mm_s"),
                    })
                    aggregate_rows.append(row)

    if aggregate_rows:
        df = pd.DataFrame(aggregate_rows)
        agg_path = args.results_dir / "aggregate_metrics.csv"
        df.to_csv(agg_path, index=False)
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        group_cols = ["P_wall", "Vmax"]
        summary = df.groupby(group_cols)[numeric_cols].agg(["mean", "std"])
        summary.columns = ["_".join(c).strip("_") for c in summary.columns.to_flat_index()]
        summary = summary.reset_index()
        summary.to_csv(args.results_dir / "summary_by_condition.csv", index=False)
        print(f"\n[AGG] {agg_path}")
        print(f"[AGG] {args.results_dir / 'summary_by_condition.csv'}")

    print("\n" + "=" * 72)
    print("BATCH COMPLETE")
    print(f"Images: {len(mat_files)} | conditions/image: {n_conditions}")
    if failures:
        print(f"Failures ({len(failures)}):")
        for img, stage in failures:
            print(f"  {img}: {stage}")
    print(f"Results: {args.results_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
