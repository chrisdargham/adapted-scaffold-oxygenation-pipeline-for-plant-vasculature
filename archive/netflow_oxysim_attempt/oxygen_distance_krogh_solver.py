#!/usr/bin/env python3
r"""
oxygen_distance_krogh_solver.py

Fast oxygenation screening model for plant-scaffold NetFlow outputs.

Model:
  Uses the nearest perfused vessel as an oxygen source and estimates tissue oxygen
  by a steady zero-order diffusion-consumption approximation:

      C(d) = max(C_vessel - Vmax * d^2 / (2D), 0)

  This is a conservative slab-style distance approximation. It is not a full
  convective Green's-function network solver. It is intended as a robust,
  auditable fallback when the custom OxySim PDE solver fails verification.

Inputs:
  --hemo: NetFlow-to-OxySim bridge .npz file containing speed_map or perfused_vessel.
  --outdir: output folder.

Outputs:
  metrics.json
  oxygen_map_mmHg.npy
  distance_to_perfused_vessel_um.npy
  oxygen_map_mmHg.png
  distance_to_perfused_vessel_um.png

PowerShell example:
  python scripts\oxygen_distance_krogh_solver.py `
    --hemo results_oxy\ACJ01-T343-B1S-CLAHE\bridge_fixed\ACJ01-T343-B1S-CLAHE_hemo.npz `
    --outdir results_oxy\ACJ01-T343-B1S-CLAHE\sweeps\krogh_distance `
    --D 1.5e-9 `
    --Vmax 1e-4 `
    --alpha 0.00134 `
    --Cvessel 0.2
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def save_png(arr, path: Path, title: str, label: str):
    if plt is None:
        return
    plt.figure(figsize=(7, 7))
    im = plt.imshow(arr)
    plt.title(title)
    plt.axis("off")
    cb = plt.colorbar(im, fraction=0.046, pad=0.04)
    cb.set_label(label)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hemo", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--D", type=float, default=1.5e-9, help="oxygen diffusivity, m^2/s")
    ap.add_argument("--Vmax", type=float, default=1e-4, help="zero-order oxygen consumption, mM/s")
    ap.add_argument("--alpha", type=float, default=0.00134, help="oxygen solubility, mM/mmHg")
    ap.add_argument("--Cvessel", type=float, default=0.2, help="source dissolved oxygen concentration, mM")
    ap.add_argument("--thresholds", type=float, nargs="*", default=[5, 10, 15, 20, 40])
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    z = np.load(args.hemo, allow_pickle=True)

    if "speed_map" in z.files:
        source = z["speed_map"] > 0
    elif "perfused_vessel" in z.files:
        source = z["perfused_vessel"].astype(bool)
    elif "vessel_mask" in z.files:
        source = z["vessel_mask"].astype(bool)
    else:
        raise RuntimeError("Hemo file lacks speed_map, perfused_vessel, and vessel_mask.")

    if not np.any(source):
        raise RuntimeError("No perfused source pixels found.")

    nx = int(z["nx"]) if "nx" in z.files else source.shape[1]
    ny = int(z["ny"]) if "ny" in z.files else source.shape[0]
    Lx = float(z["Lx_m"]) if "Lx_m" in z.files else 1.0
    Ly = float(z["Ly_m"]) if "Ly_m" in z.files else 1.0

    dx = Lx / max(nx - 1, 1)
    dy = Ly / max(ny - 1, 1)

    # distance to nearest perfused source pixel, in meters
    dist_m = distance_transform_edt(~source, sampling=(dy, dx))

    # steady slab/Krogh-style zero-order approximation
    C = args.Cvessel - args.Vmax * dist_m**2 / (2.0 * args.D)
    C = np.maximum(C, 0.0)
    pO2 = C / args.alpha

    tissue = ~source
    tissue_p = pO2[tissue]

    metrics = {
        "algorithm": "distance_to_perfused_vessel_krogh_slab_approximation",
        "interpretation": "fast upper-bound/screening model; not a full network Green's-function solver",
        "D_m2_s": args.D,
        "Vmax_mM_s": args.Vmax,
        "alpha_mM_per_mmHg": args.alpha,
        "Cvessel_mM": args.Cvessel,
        "Cvessel_mmHg": args.Cvessel / args.alpha,
        "nx": nx,
        "ny": ny,
        "Lx_m": Lx,
        "Ly_m": Ly,
        "dx_um": dx * 1e6,
        "dy_um": dy * 1e6,
        "source_pixels": int(source.sum()),
        "tissue_pixels": int(tissue.sum()),
        "distance_median_um": float(np.median(dist_m[tissue]) * 1e6),
        "distance_p95_um": float(np.percentile(dist_m[tissue], 95) * 1e6),
        "distance_max_um": float(np.max(dist_m[tissue]) * 1e6),
        "pO2_median_mmHg": float(np.median(tissue_p)),
        "pO2_p05_mmHg": float(np.percentile(tissue_p, 5)),
        "pO2_p95_mmHg": float(np.percentile(tissue_p, 95)),
        "pO2_min_mmHg": float(np.min(tissue_p)),
        "pO2_max_mmHg": float(np.max(tissue_p)),
    }

    for th in args.thresholds:
        metrics[f"tissue_pct_ge_{th:g}_mmHg"] = 100.0 * float(np.mean(tissue_p >= th))

    # Useful analytical critical distances
    for th in args.thresholds:
        Cth = th * args.alpha
        if args.Cvessel > Cth:
            dcrit = np.sqrt(2.0 * args.D * (args.Cvessel - Cth) / args.Vmax)
            metrics[f"critical_distance_for_{th:g}_mmHg_um"] = float(dcrit * 1e6)
        else:
            metrics[f"critical_distance_for_{th:g}_mmHg_um"] = 0.0

    (args.outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.save(args.outdir / "oxygen_map_mmHg.npy", pO2)
    np.save(args.outdir / "distance_to_perfused_vessel_um.npy", dist_m * 1e6)

    save_png(pO2, args.outdir / "oxygen_map_mmHg.png", "Krogh-distance oxygen estimate", "pO2 (mmHg)")
    save_png(dist_m * 1e6, args.outdir / "distance_to_perfused_vessel_um.png", "Distance to nearest perfused vessel", "distance (µm)")

    print("=== Krogh-distance oxygen solver ===")
    print(f"source pixels                 : {metrics['source_pixels']}")
    print(f"median distance               : {metrics['distance_median_um']:.2f} um")
    print(f"p95 distance                  : {metrics['distance_p95_um']:.2f} um")
    print(f"max distance                  : {metrics['distance_max_um']:.2f} um")
    print(f"median tissue pO2             : {metrics['pO2_median_mmHg']:.2f} mmHg")
    for th in args.thresholds:
        print(f"tissue >= {th:g} mmHg              : {metrics[f'tissue_pct_ge_{th:g}_mmHg']:.3f}%")
    print(f"wrote                         : {args.outdir}")


if __name__ == "__main__":
    main()
