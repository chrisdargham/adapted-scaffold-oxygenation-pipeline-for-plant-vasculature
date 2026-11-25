#!/usr/bin/env python3
r"""
keff_o2_concentration_screen.py

Fast k_eff oxygen-concentration screen for the plant scaffold project.

This replaces the unstable transient OxySim stage with a transparent steady
diffusion-consumption approximation.

Model:
  1. Use NetFlow-derived perfused scaffold/vessel pixels as O2 source locations.
  2. Compute distance d from every tissue pixel to nearest perfused source.
  3. Convert vessel oxygen concentration Cvessel to an effective scaffold/tissue
     source concentration Csource using k_eff:

        Csource = Cvessel * k_eff / (k_eff + k_cons)

     where k_cons = Vmax/(Km + Cvessel).

  4. Estimate tissue concentration with a zero-order diffusion-consumption
     approximation:

        C(d) = max(Csource - Vmax*d^2/(2D), 0)

This is not a full Green's-function vascular solver. It is a fast, conservative,
auditable fallback that gives O2 concentration with k_eff and consumption.

Outputs:
  concentration_mM.npy
  pO2_mmHg.npy
  distance_to_perfused_source_um.npy
  metrics.json
  PNG figures

PowerShell example:
  python scripts\keff_o2_concentration_screen.py `
    --hemo results_oxy\ACJ01-T343-B1S-CLAHE\bridge_fixed\ACJ01-T343-B1S-CLAHE_hemo.npz `
    --outdir results_oxy\ACJ01-T343-B1S-CLAHE\sweeps\keff_screen `
    --D 1.5e-9 --Vmax 1e-4 --Km 0.001 --keff 3e-5 --Cvessel 0.2 --alpha 0.00134
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation

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
    ap.add_argument("--D", type=float, default=1.5e-9, help="O2 diffusivity [m^2/s]")
    ap.add_argument("--Vmax", type=float, default=1e-4, help="max O2 consumption [mM/s]")
    ap.add_argument("--Km", type=float, default=0.001, help="MM half-saturation [mM]")
    ap.add_argument("--keff", type=float, default=3e-5, help="effective exchange [1/s]")
    ap.add_argument("--Cvessel", type=float, default=0.2, help="vessel/source O2 [mM]")
    ap.add_argument("--alpha", type=float, default=0.00134, help="mM/mmHg")
    ap.add_argument("--source-band-px", type=int, default=1)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    z = np.load(args.hemo, allow_pickle=True)
    keys = set(z.files)

    if "speed_map" in keys:
        source = z["speed_map"].astype(float) > 0
    elif "perfused_vessel" in keys:
        source = z["perfused_vessel"].astype(bool)
    elif "vessel_mask" in keys:
        source = z["vessel_mask"].astype(bool)
    else:
        raise RuntimeError("Hemo file must contain speed_map, perfused_vessel, or vessel_mask.")

    if args.source_band_px > 0:
        source = binary_dilation(source, iterations=args.source_band_px)

    if not np.any(source):
        raise RuntimeError("No perfused scaffold/source pixels found.")

    ny, nx = source.shape

    Lx = float(z["Lx_m"]) if "Lx_m" in keys else (float(z["Lx"]) if "Lx" in keys else 0.01281384)
    Ly = float(z["Ly_m"]) if "Ly_m" in keys else (float(z["Ly"]) if "Ly" in keys else 0.01923038)

    dx = Lx / max(nx - 1, 1)
    dy = Ly / max(ny - 1, 1)

    dist_m = distance_transform_edt(~source, sampling=(dy, dx))

    k_cons_source = args.Vmax / max(args.Km + args.Cvessel, 1e-30)
    Csource_eff = args.Cvessel * args.keff / (args.keff + k_cons_source)

    C = Csource_eff - args.Vmax * dist_m**2 / (2.0 * args.D)
    C = np.maximum(C, 0.0)
    pO2 = C / args.alpha

    tissue = ~source
    Ct = C[tissue]
    Pt = pO2[tissue]
    dist_um = dist_m * 1e6

    metrics = {
        "algorithm": "keff_distance_diffusion_consumption_screen",
        "equation": "C(d)=max(Csource_eff - Vmax*d^2/(2D),0), Csource_eff=Cvessel*keff/(keff+Vmax/(Km+Cvessel))",
        "interpretation": "fast k_eff-controlled O2 concentration screen; no axial depletion; not full PDE",
        "D_m2_s": args.D,
        "Vmax_mM_s": args.Vmax,
        "Km_mM": args.Km,
        "keff_1_s": args.keff,
        "Cvessel_mM": args.Cvessel,
        "Cvessel_mmHg": args.Cvessel / args.alpha,
        "alpha_mM_per_mmHg": args.alpha,
        "k_cons_source_1_s": k_cons_source,
        "Csource_eff_mM": Csource_eff,
        "Csource_eff_mmHg": Csource_eff / args.alpha,
        "nx": nx,
        "ny": ny,
        "Lx_m": Lx,
        "Ly_m": Ly,
        "dx_um": dx * 1e6,
        "dy_um": dy * 1e6,
        "source_band_px": args.source_band_px,
        "source_pixels": int(source.sum()),
        "tissue_pixels": int(tissue.sum()),
        "distance_median_um": float(np.median(dist_um[tissue])),
        "distance_p95_um": float(np.percentile(dist_um[tissue], 95)),
        "distance_max_um": float(np.max(dist_um[tissue])),
        "C_median_mM": float(np.median(Ct)),
        "C_p05_mM": float(np.percentile(Ct, 5)),
        "C_p95_mM": float(np.percentile(Ct, 95)),
        "C_min_mM": float(np.min(Ct)),
        "C_max_mM": float(np.max(Ct)),
        "pO2_median_mmHg": float(np.median(Pt)),
        "pO2_p05_mmHg": float(np.percentile(Pt, 5)),
        "pO2_p95_mmHg": float(np.percentile(Pt, 95)),
        "pO2_min_mmHg": float(np.min(Pt)),
        "pO2_max_mmHg": float(np.max(Pt)),
    }

    for th in [5, 10, 15, 20, 40]:
        metrics[f"tissue_pct_ge_{th}_mmHg"] = 100.0 * float(np.mean(Pt >= th))

    np.save(args.outdir / "concentration_mM.npy", C)
    np.save(args.outdir / "pO2_mmHg.npy", pO2)
    np.save(args.outdir / "distance_to_perfused_source_um.npy", dist_um)
    np.save(args.outdir / "source_mask.npy", source)
    (args.outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    save_png(C, args.outdir / "concentration_mM.png", "O2 concentration with k_eff", "C (mM)")
    save_png(pO2, args.outdir / "pO2_mmHg.png", "pO2 with k_eff", "pO2 (mmHg)")
    save_png(dist_um, args.outdir / "distance_to_perfused_source_um.png", "Distance to perfused source", "µm")
    save_png(source.astype(float), args.outdir / "source_mask.png", "Perfused source mask", "source")

    print("=== k_eff O2 concentration screen ===")
    print(f"k_eff                 : {args.keff:.3e} 1/s")
    print(f"Cvessel               : {args.Cvessel:.6g} mM = {args.Cvessel/args.alpha:.2f} mmHg")
    print(f"Csource_eff           : {Csource_eff:.6g} mM = {Csource_eff/args.alpha:.2f} mmHg")
    print(f"source pixels         : {metrics['source_pixels']}")
    print(f"median distance       : {metrics['distance_median_um']:.2f} µm")
    print(f"median C              : {metrics['C_median_mM']:.6g} mM")
    print(f"median pO2            : {metrics['pO2_median_mmHg']:.6g} mmHg")
    print(f"tissue >=15 mmHg      : {metrics['tissue_pct_ge_15_mmHg']:.3f}%")
    print(f"tissue >=20 mmHg      : {metrics['tissue_pct_ge_20_mmHg']:.3f}%")
    print(f"wrote                 : {args.outdir}")


if __name__ == "__main__":
    main()
