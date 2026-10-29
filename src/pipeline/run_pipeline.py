#!/usr/bin/env python3
"""
run_pipeline.py  — Full automated pipeline for 30 leaf scaffold images
=======================================================================
WHAT IT DOES
------------
For each .mat file in --mat-dir:
  1. Runs prepare_network.py → Network.dat + network_ready.dat
  2. Copies network_ready.dat into NetFlowV2 folder → runs NetFlowV2
  3. Copies NetworkNew.dat into GreensV4 folder, sets PO2 column
  4. Runs GreensV4 THREE times (Keff = 1, 10, 100 µm/s)
  5. Saves all outputs into --results-dir/<image_name>/<keff>/

USAGE
-----
python run_pipeline.py
    --mat-dir       MAT_FOLDER
    --results-dir   RESULTS_FOLDER
    --netflow-dir   NETFLOW_FOLDER
    --greens-dir    GREENS_FOLDER
    --netflow-exe   NetFlowV2.exe
    --greens-exe    GreensV4.exe
    --inlet-rc      1400,5
    --exporter      leafveincnn_to_netflow.py
    --prepare       prepare_network.py

PARAMETERS (match your biological choices)
-------------------------------------------
inlet-pressure  = 40.0 mmHg
outlet-pressure = 35.0 mmHg  (ΔP = 5 mmHg)
lumen-fraction  = 0.5         (sensitivity baseline)
boundary-margin = 250 µm      (~45 outlets)
resample-step   = 25 µm
inlet-po2       = 150 mmHg   (air-equilibrated PBS, Gershlak 2017)
outlet-po2      = 40 mmHg    (venous lower bound)
keff-values     = 1, 10, 100 µm/s (cellulose wall sweep)
"""

import argparse
import shutil
import subprocess
import sys
import os
import time
from pathlib import Path


# ── CLI ───────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="Full automated leaf scaffold pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mat-dir",         type=Path, required=True,
                   help="Folder containing all .mat files")
    p.add_argument("--results-dir",     type=Path, required=True,
                   help="Where to save all outputs")
    p.add_argument("--netflow-dir",     type=Path, required=True,
                   help="NetFlowV2 working directory (where Network.dat goes)")
    p.add_argument("--greens-dir",      type=Path, required=True,
                   help="GreensV4 working directory")
    p.add_argument("--netflow-exe",     type=str,  default="NetFlowV2.exe",
                   help="NetFlowV2 executable name")
    p.add_argument("--greens-exe",      type=str,  default="GreensV4.exe",
                   help="GreensV4 executable name")
    p.add_argument("--exporter",        type=Path, required=True,
                   help="Path to leafveincnn_to_netflow.py")
    p.add_argument("--prepare",         type=Path, required=True,
                   help="Path to prepare_network.py")
    p.add_argument("--inlet-rc",        type=str,  default="1400,5")
    p.add_argument("--inlet-pressure",  type=float, default=40.0)
    p.add_argument("--outlet-pressure", type=float, default=35.0)
    p.add_argument("--lumen-fraction",  type=float, default=0.5)
    p.add_argument("--boundary-margin", type=float, default=250.0)
    p.add_argument("--resample-step",   type=float, default=25.0)
    p.add_argument("--inlet-po2",       type=float, default=150.0)
    p.add_argument("--outlet-po2",      type=float, default=40.0)
    p.add_argument("--keff-values",     type=str,  default="1,10,100",
                   help="Comma-separated Keff values in µm/s")
    p.add_argument("--python",          type=str,  default=sys.executable)
    p.add_argument("--dry-run",         action="store_true",
                   help="Print commands without executing")
    return p


# ── Helpers ───────────────────────────────────────────────────────────────

def run(cmd, cwd=None, dry_run=False):
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"\n  >> {cmd_str}")
    if dry_run:
        return 0
    r = subprocess.run(cmd, cwd=str(cwd) if cwd else None,
                       capture_output=False, text=True)
    if r.returncode != 0:
        print(f"  [ERROR] exit code {r.returncode}")
    return r.returncode


def set_po2_in_network(src: Path, dst: Path, inlet_press: float,
                        inlet_po2: float, outlet_po2: float):
    """Read NetworkNew.dat, set the PO2 column of every boundary node, write Network.dat for GreensV4.

    Boundary rows are: node  bctype  press/flow  HD  PO2  [*]. The node whose pressure
    matches inlet_press gets inlet_po2; all others get outlet_po2.

    Bugs fixed (Oct 2026):
    1. The first version switched its "inside boundary section" flag off when it reached
       the header line ("Node Bctype ..."), which comes right before the boundary rows.
       No row was ever changed, so GreensV4 received PO2 = 0 at the inlet.
    2. A node counted as the inlet if its pressure was within 0.5 mmHg of inlet_press.
       With a small pressure drop (the May run used 40 -> 39.9955 mmHg) every outlet
       would also have been given the inlet PO2. The tolerance is now 1e-4 mmHg.
    """
    lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
    j = next(i for i, l in enumerate(lines) if "boundary nodes" in l.lower())
    nbc = int(lines[j].split()[0])
    for i in range(j + 2, j + 2 + nbc):          # skip the count line and the header line
        p = lines[i].split()
        star = p[-1] == "*"
        vals = p[:-1] if star else p
        while len(vals) < 5:
            vals.append("0.000000")
        press = float(vals[2])
        vals[4] = f"{inlet_po2 if abs(press - inlet_press) < 1e-4 else outlet_po2:.6f}"
        lines[i] = " ".join(vals + (["*"] if star else []))
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_intravascres(keff_um_s: float, dst: Path):
    """Write IntravascRes.dat for a given Keff [µm/s]."""
    alpha_PBS = 1.3e-3  # µm³ O2 / µm³ medium / mmHg
    gamma1 = 1.0 / (keff_um_s * alpha_PBS)
    diams = [4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 40, 50, 75, 100, 150]
    lines = [
        f" {len(diams)} Solute 1",
        f"D (um)  K (mmHg.um.s/um3O2)  Keff={keff_um_s} um/s plant cellulose wall",
    ]
    for d in diams:
        lines.append(f"  {d:<6}  {gamma1:.4f}")
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_greens_outputs(greens_current: Path, dest: Path, image_name: str, keff: float):
    """Copy GreensV4 outputs to results folder."""
    dest.mkdir(parents=True, exist_ok=True)
    files_to_save = [
        "TissueLevels001.out",
        "VesselLevels001.out",
        "TissueSources001.out",
        "VesselSources001.out",
        "GreensRes001.out",
        "summary.out",
    ]
    for fname in files_to_save:
        src = greens_current / fname
        if src.exists():
            shutil.copy2(src, dest / fname)
        else:
            # Try without zero-padded index
            alt = greens_current / fname.replace("001", "")
            if alt.exists():
                shutil.copy2(alt, dest / fname)


def save_netflow_outputs(netflow_current: Path, dest: Path):
    """Copy NetFlowV2 outputs to results folder."""
    dest.mkdir(parents=True, exist_ok=True)
    for fname in ["NetworkNew.dat", "Results.out"]:
        src = netflow_current / fname
        if src.exists():
            shutil.copy2(src, dest / fname)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    args = build_parser().parse_args()

    keff_list = [float(k.strip()) for k in args.keff_values.split(",")]

    mat_files = sorted(args.mat_dir.glob("*.mat"))
    if not mat_files:
        print(f"[ERROR] No .mat files found in {args.mat_dir}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f" PIPELINE START — {len(mat_files)} images, {len(keff_list)} Keff values")
    print(f" Total runs: {len(mat_files) * len(keff_list)} GreensV4 runs")
    print(f"{'='*60}")

    args.results_dir.mkdir(parents=True, exist_ok=True)

    failed = []

    for mat_idx, mat_path in enumerate(mat_files, 1):
        image_name = mat_path.stem
        print(f"\n{'─'*60}")
        print(f" [{mat_idx}/{len(mat_files)}] IMAGE: {image_name}")
        print(f"{'─'*60}")

        img_dir = args.results_dir / image_name
        img_dir.mkdir(parents=True, exist_ok=True)

        # ── STEP 1: prepare_network.py → Network.dat ──────────────────
        print(f"\n[1/4] Running prepare_network.py …")
        prep_out = img_dir / "netflow_input"
        prep_cmd = [
            args.python, str(args.prepare),
            "--mat",                str(mat_path),
            "--exporter",          str(args.exporter),
            "--output-dir",        str(prep_out),
            "--inlet-rc",          args.inlet_rc,
            "--inlet-pressure",    str(args.inlet_pressure),
            "--outlet-pressure",   str(args.outlet_pressure),
            "--lumen-fraction",    str(args.lumen_fraction),
            "--boundary-outlets-only",
            "--boundary-margin-um", str(args.boundary_margin),
            "--resample-step-um",  str(args.resample_step),
        ]
        rc = run(prep_cmd, dry_run=args.dry_run)
        if rc != 0:
            print(f"  [SKIP] prepare_network.py failed for {image_name}")
            failed.append((image_name, "prepare_network"))
            continue

        # ── STEP 2: NetFlowV2 ─────────────────────────────────────────
        print(f"\n[2/4] Running NetFlowV2 …")
        network_ready = prep_out / "network_ready.dat"
        netflow_network = args.netflow_dir / "Network.dat"

        if not args.dry_run:
            shutil.copy2(network_ready, netflow_network)
        print(f"  Copied network_ready.dat → {netflow_network}")

        rc = run([args.netflow_exe], cwd=args.netflow_dir, dry_run=args.dry_run)
        if rc != 0:
            print(f"  [SKIP] NetFlowV2 failed for {image_name}")
            failed.append((image_name, "NetFlowV2"))
            continue

        # Save NetFlowV2 outputs
        netflow_current = args.netflow_dir / "Current"
        netflow_results_dir = img_dir / "netflow_results"
        if not args.dry_run:
            save_netflow_outputs(netflow_current, netflow_results_dir)
        print(f"  Outputs saved → {netflow_results_dir}")

        # ── STEP 3+4: GreensV4 × 3 Keff values ───────────────────────
        print(f"\n[3/4] Setting up and running GreensV4 ({len(keff_list)} Keff values) …")

        networknew = netflow_current / "NetworkNew.dat"
        if not networknew.exists() and not args.dry_run:
            print(f"  [SKIP] NetworkNew.dat not found after NetFlowV2")
            failed.append((image_name, "NetworkNew_missing"))
            continue

        for keff in keff_list:
            print(f"\n  Keff = {keff} µm/s …")

            # Write Network.dat with PO2 set
            greens_network = args.greens_dir / "Network.dat"
            if not args.dry_run:
                set_po2_in_network(
                    networknew, greens_network,
                    args.inlet_pressure, args.inlet_po2, args.outlet_po2
                )

            # Write IntravascRes.dat for this Keff
            intravascres_dst = args.greens_dir / "IntravascRes.dat"
            if not args.dry_run:
                make_intravascres(keff, intravascres_dst)

            # Run GreensV4
            rc = run([args.greens_exe], cwd=args.greens_dir, dry_run=args.dry_run)

            # Save outputs
            keff_str = f"Keff{int(keff)}"
            greens_out_dir = img_dir / "greens_results" / keff_str
            if not args.dry_run:
                greens_current = args.greens_dir / "Current"
                save_greens_outputs(greens_current, greens_out_dir, image_name, keff)
            print(f"    Outputs saved → {greens_out_dir}")

            if rc != 0:
                print(f"    [WARNING] GreensV4 returned non-zero for {image_name} Keff={keff}")
                failed.append((image_name, f"GreensV4_Keff{keff}"))

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" PIPELINE COMPLETE")
    print(f" Images processed: {len(mat_files) - len([f for f in failed if 'prepare' in f[1]])}")
    if failed:
        print(f" Failures ({len(failed)}):")
        for img, stage in failed:
            print(f"   {img} — {stage}")
    print(f" Results directory: {args.results_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
