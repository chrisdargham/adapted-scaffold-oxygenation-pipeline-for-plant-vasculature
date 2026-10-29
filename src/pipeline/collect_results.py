#!/usr/bin/env python3
"""
collect_results.py  — Parse all pipeline outputs into a single CSV
===================================================================
Reads from --results-dir (output of run_pipeline.py) and produces:
  results_summary.csv   — one row per image × Keff combination
  
COLUMNS
-------
image_name          str   — MAT file stem
keff_um_s           float — wall permeability (1, 10, 100)
lumen_fraction      float — from run parameters
delta_p_mmhg        float — inlet - outlet pressure
n_segments          int   — total type-4/5 segments in network
n_bc_nodes          int   — boundary condition nodes
n_outlets           int   — outlet BC nodes
--- NetFlowV2 outputs ---
flow_median_nlmin   float — median segment flow
flow_mean_nlmin     float — mean segment flow
flow_max_nlmin      float — maximum segment flow
pressure_mean_mmhg  float — mean nodal pressure
pressure_sd_mmhg    float — pressure standard deviation
velocity_median_mms float — median flow velocity (mm/s)
velocity_p95_mms    float — 95th percentile velocity
n_zero_flow_segs    int   — segments with Q=0 (type 0 after solver)
--- GreensV4 outputs ---
tissue_po2_mean     float — mean tissue PO2 (mmHg)
tissue_po2_sd       float — std dev of tissue PO2
tissue_po2_min      float — minimum tissue PO2
tissue_po2_max      float — maximum tissue PO2
frac_hypoxic_10     float — fraction of tissue points with PO2 < 10 mmHg
frac_hypoxic_5      float — fraction of tissue points with PO2 < 5 mmHg
vessel_po2_mean     float — mean intravascular PO2
vessel_po2_min      float — minimum intravascular PO2
greens_converged    int   — 1 if TissueLevels exists and valid, 0 otherwise

USAGE
-----
python collect_results.py --results-dir C:\\path\\to\\results --output results_summary.csv
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, required=True,
                   help="Output of run_pipeline.py")
    p.add_argument("--output",      type=Path, default=Path("results_summary.csv"))
    p.add_argument("--hypoxia-thresholds", type=str, default="10,5",
                   help="PO2 thresholds (mmHg) for hypoxia fraction columns")
    p.add_argument("--lumen-fraction",  type=float, default=0.5)
    p.add_argument("--inlet-pressure",  type=float, default=40.0)
    p.add_argument("--outlet-pressure", type=float, default=35.0)
    return p


# ── Parsers ───────────────────────────────────────────────────────────────

def parse_network_dat(path: Path) -> dict:
    """Extract segment count, BC node count from network_ready.dat."""
    result = {"n_segments": 0, "n_bc_nodes": 0, "n_outlets": 0}
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        in_seg = in_bc = False
        for line in lines:
            l = line.strip()
            if l.startswith("SegName"):
                in_seg = True; continue
            if "number of nodes" in l.lower():
                in_seg = False
            if "boundary nodes" in l.lower():
                in_bc = True; continue
            if "bctype" in l.lower():
                in_bc = False; continue
            if in_seg:
                p = l.split()
                if len(p) >= 5:
                    try:
                        if int(p[1]) in [4, 5]:
                            result["n_segments"] += 1
                    except: pass
            if in_bc:
                p = l.split()
                if p:
                    try:
                        nid = int(p[0])
                        press = float(p[2])
                        result["n_bc_nodes"] += 1
                        if abs(press - 35.0) < 2.0:   # outlet range
                            result["n_outlets"] += 1
                    except: pass
    except Exception as e:
        print(f"  [WARN] parse_network_dat: {e}")
    return result


def parse_results_out(path: Path) -> dict:
    """Extract flow/pressure/velocity stats from NetFlowV2 Results.out."""
    result = {
        "flow_median_nlmin": None, "flow_mean_nlmin": None, "flow_max_nlmin": None,
        "pressure_mean_mmhg": None, "pressure_sd_mmhg": None,
        "velocity_median_mms": None, "velocity_p95_mms": None,
        "n_zero_flow_segs": 0,
    }
    flows, pressures, velocities = [], [], []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        # Parse header for mean/sd
        for line in lines:
            l = line.strip()
            if "Nodal pressure mean" in l:
                parts = l.replace("+-", " ").split()
                try:
                    result["pressure_mean_mmhg"] = float(parts[4])
                    result["pressure_sd_mmhg"]   = float(parts[5])
                except: pass
            if "Maximum flow:" in l:
                try:
                    result["flow_max_nlmin"] = float(l.split()[2])
                except: pass
            if "Segment flow mean" in l:
                try:
                    result["flow_mean_nlmin"] = float(l.split()[-1])
                except: pass

        # Parse segment data
        in_data = False
        for line in lines:
            if "segment name" in line.lower():
                in_data = True; continue
            if not in_data: continue
            p = line.strip().split()
            if len(p) >= 9:
                try:
                    q    = abs(float(p[2]))
                    lseg = float(p[3])
                    d    = float(p[8]) if len(p) > 8 else 0  # viscosity col
                    pres = float(p[5])
                    diam_col = None  # diameter not in Results.out directly
                    flows.append(q)
                    pressures.append(pres)
                    if q == 0.0:
                        result["n_zero_flow_segs"] += 1
                except: pass

        if flows:
            import statistics
            flows_sorted = sorted(flows)
            n = len(flows_sorted)
            result["flow_median_nlmin"] = flows_sorted[n // 2]
            if result["flow_mean_nlmin"] is None:
                result["flow_mean_nlmin"] = sum(flows) / n

    except Exception as e:
        print(f"  [WARN] parse_results_out: {e}")
    return result


def parse_networknew_velocities(path: Path) -> dict:
    """Compute velocities from NetworkNew.dat (has segment diameters) + Results.out flows."""
    # NetworkNew.dat has flows in col 6 after NetFlowV2
    result = {"velocity_median_mms": None, "velocity_p95_mms": None}
    try:
        import math
        vels = []
        with open(path, encoding="utf-8", errors="replace") as f:
            in_seg = False
            for line in f:
                l = line.strip()
                if l.startswith("SegName"): in_seg = True; continue
                if "number of nodes" in l: in_seg = False
                if in_seg:
                    p = l.split()
                    if len(p) >= 7:
                        try:
                            if int(p[1]) in [4, 5]:
                                d = float(p[4])   # diameter µm
                                q = abs(float(p[5]))  # flow nl/min
                                if d > 0:
                                    v = 4000.0 / 60.0 * q / math.pi / (d ** 2)  # mm/s
                                    vels.append(v)
                        except: pass

        if vels:
            vels.sort()
            n = len(vels)
            result["velocity_median_mms"] = vels[n // 2]
            result["velocity_p95_mms"]    = vels[int(0.95 * n)]
    except Exception as e:
        print(f"  [WARN] parse_networknew_velocities: {e}")
    return result


def parse_tissue_levels(path: Path, thresholds: list) -> dict:
    """Parse TissueLevels001.out for PO2 statistics."""
    result = {
        "tissue_po2_mean": None, "tissue_po2_sd": None,
        "tissue_po2_min": None,  "tissue_po2_max": None,
        "greens_converged": 0,
    }
    for t in thresholds:
        result[f"frac_hypoxic_{t}"] = None

    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        values = []
        in_values = False
        for i, line in enumerate(lines):
            l = line.strip()
            if l == "pmeant":
                try:
                    result["tissue_po2_mean"] = float(lines[i + 1].strip().split()[0])
                except: pass
            elif l == "psdt":
                try:
                    result["tissue_po2_sd"] = float(lines[i + 1].strip().split()[0])
                except: pass
            elif l == "pmin":
                try:
                    result["tissue_po2_min"] = float(lines[i + 1].strip().split()[0])
                except: pass
            elif l == "pmax":
                try:
                    result["tissue_po2_max"] = float(lines[i + 1].strip().split()[0])
                except: pass
            elif l == "values":
                in_values = True
                continue
            elif in_values and l:
                try:
                    values.append(float(l.split()[0]))
                except: pass

        if values:
            result["greens_converged"] = 1
            n = len(values)
            for t in thresholds:
                result[f"frac_hypoxic_{t}"] = sum(1 for v in values if v < t) / n

    except Exception as e:
        print(f"  [WARN] parse_tissue_levels: {e}")
    return result


def parse_vessel_levels(path: Path) -> dict:
    """Parse VesselLevels001.out for vessel PO2 stats."""
    result = {"vessel_po2_mean": None, "vessel_po2_min": None}
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            l = line.strip()
            if l == "pmeanv":
                try:
                    result["vessel_po2_mean"] = float(lines[i + 1].strip().split()[0])
                except: pass
            elif l == "pmin":
                try:
                    result["vessel_po2_min"] = float(lines[i + 1].strip().split()[0])
                except: pass
    except Exception as e:
        print(f"  [WARN] parse_vessel_levels: {e}")
    return result


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    args = build_parser().parse_args()
    thresholds = [int(t.strip()) for t in args.hypoxia_thresholds.split(",")]

    results_dir = args.results_dir
    if not results_dir.exists():
        print(f"[ERROR] results-dir not found: {results_dir}")
        sys.exit(1)

    rows = []

    image_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    print(f"Found {len(image_dirs)} image directories")

    for img_dir in image_dirs:
        image_name = img_dir.name
        print(f"\nProcessing: {image_name}")

        # Network stats (from network_ready.dat)
        network_ready = img_dir / "netflow_input" / "network_ready.dat"
        network_stats = parse_network_dat(network_ready) if network_ready.exists() else {}

        # NetFlowV2 results
        results_out = img_dir / "netflow_results" / "Results.out"
        networknew  = img_dir / "netflow_results" / "NetworkNew.dat"
        flow_stats  = parse_results_out(results_out) if results_out.exists() else {}
        vel_stats   = parse_networknew_velocities(networknew) if networknew.exists() else {}

        # Merge velocity into flow_stats
        flow_stats.update(vel_stats)

        # GreensV4 results for each Keff
        greens_dir = img_dir / "greens_results"
        if greens_dir.exists():
            keff_dirs = sorted([d for d in greens_dir.iterdir() if d.is_dir()])
        else:
            keff_dirs = []

        if not keff_dirs:
            # No GreensV4 results — write one row with netflow data only
            row = {
                "image_name":         image_name,
                "keff_um_s":          None,
                "lumen_fraction":     args.lumen_fraction,
                "delta_p_mmhg":       args.inlet_pressure - args.outlet_pressure,
            }
            row.update(network_stats)
            row.update(flow_stats)
            row.update({
                "tissue_po2_mean": None, "tissue_po2_sd": None,
                "tissue_po2_min": None,  "tissue_po2_max": None,
                "vessel_po2_mean": None, "vessel_po2_min": None,
                "greens_converged": 0,
            })
            for t in thresholds:
                row[f"frac_hypoxic_{t}"] = None
            rows.append(row)
        else:
            for keff_dir in keff_dirs:
                keff_str = keff_dir.name  # e.g. "Keff10"
                try:
                    keff_val = float(keff_str.replace("Keff", ""))
                except ValueError:
                    keff_val = None

                tissue_stats = parse_tissue_levels(
                    keff_dir / "TissueLevels001.out", thresholds
                )
                vessel_stats = parse_vessel_levels(keff_dir / "VesselLevels001.out")

                row = {
                    "image_name":         image_name,
                    "keff_um_s":          keff_val,
                    "lumen_fraction":     args.lumen_fraction,
                    "delta_p_mmhg":       args.inlet_pressure - args.outlet_pressure,
                }
                row.update(network_stats)
                row.update(flow_stats)
                row.update(tissue_stats)
                row.update(vessel_stats)
                rows.append(row)
                print(f"  Keff={keff_val}: tissue_PO2_mean={tissue_stats.get('tissue_po2_mean'):.1f} "
                      f"frac_hypoxic_10={tissue_stats.get('frac_hypoxic_10')}"
                      if tissue_stats.get("tissue_po2_mean") is not None else
                      f"  Keff={keff_val}: GreensV4 output missing")

    if not rows:
        print("[ERROR] No results found.")
        sys.exit(1)

    # Write CSV
    all_keys = list(rows[0].keys())
    # Ensure consistent column order
    col_order = [
        "image_name", "keff_um_s", "lumen_fraction", "delta_p_mmhg",
        "n_segments", "n_bc_nodes", "n_outlets",
        "flow_median_nlmin", "flow_mean_nlmin", "flow_max_nlmin",
        "pressure_mean_mmhg", "pressure_sd_mmhg",
        "velocity_median_mms", "velocity_p95_mms", "n_zero_flow_segs",
        "tissue_po2_mean", "tissue_po2_sd", "tissue_po2_min", "tissue_po2_max",
    ] + [f"frac_hypoxic_{t}" for t in thresholds] + [
        "vessel_po2_mean", "vessel_po2_min", "greens_converged",
    ]
    # Add any extra keys not in the fixed order
    extra = [k for k in all_keys if k not in col_order]
    col_order += extra

    output_path = args.output
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=col_order, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*55}")
    print(f" DONE → {output_path}")
    print(f" Rows: {len(rows)}  Columns: {len(col_order)}")
    print(f" Images: {len(image_dirs)}  Keff values: up to {len(keff_dirs) if keff_dirs else '?'}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
