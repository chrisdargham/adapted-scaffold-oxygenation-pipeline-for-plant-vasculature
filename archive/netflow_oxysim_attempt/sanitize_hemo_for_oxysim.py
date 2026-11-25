#!/usr/bin/env python3
r"""
sanitize_hemo_for_oxysim.py

Purpose:
  Make the NetFlow->OxySim hemodynamic cache safe for MySim/oxy_sim.py.

Problem fixed:
  MySim/oxy_sim.py divides by speed_map/u_loc.
  The bridge contains zero-speed pixels inside vessel_mask because the mask includes
  zero-flow or dilated vessel pixels. This causes ZeroDivisionError.

Biological decision:
  Zero-flow/stagnant segments are excluded from the convective lumen-transport mask.
  They are not treated as oxygen-supplying perfused vessels.

Usage:
  python scripts\sanitize_hemo_for_oxysim.py ^
    --input results_oxy\ACJ01-T343-B1S-CLAHE\bridge_fixed\ACJ01-T343-B1S-CLAHE_hemo.npz ^
    --output results_oxy\ACJ01-T343-B1S-CLAHE\bridge_fixed\ACJ01-T343-B1S-CLAHE_hemo_safe.npz ^
    --speed-floor 1e-6
"""

import argparse
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--speed-floor", type=float, default=1e-6,
                    help="m/s floor applied only inside already perfused pixels.")
    args = ap.parse_args()

    z = np.load(args.input, allow_pickle=True)
    arrays = {k: z[k] for k in z.files}

    if "speed_map" not in arrays:
        raise RuntimeError("Missing speed_map in hemo npz.")
    if "speed_raw_m_s" in arrays:
        raw = arrays["speed_raw_m_s"]
    else:
        raw = arrays["speed_map"]

    speed = arrays["speed_map"].astype(np.float32).copy()

    # True perfusion from raw NetFlow-derived speed, not from the floor.
    perfused = raw > 0

    # Remove zero-speed pixels from masks used by oxy_sim.py for lumen transport.
    for key in ["vessel_mask", "skel", "skel_mask", "skeleton_mask", "centerline_mask", "reachable", "perfused_vessel"]:
        if key in arrays:
            arrays[key] = np.logical_and(arrays[key].astype(bool), perfused)

    # Ensure any pixel still marked perfused has strictly positive speed.
    speed_safe = speed.copy()
    speed_safe[perfused & (speed_safe <= 0)] = args.speed_floor
    speed_safe[perfused & (speed_safe < args.speed_floor)] = args.speed_floor
    speed_safe[~perfused] = 0.0

    arrays["speed_map"] = speed_safe.astype(np.float32)
    arrays["speed_m_s"] = speed_safe.astype(np.float32)
    arrays["speed"] = speed_safe.astype(np.float32)
    arrays["u"] = speed_safe.astype(np.float32)

    # Make velocity components safe if present; preserve direction where available.
    for ux_key, uy_key in [("ux_m_s", "uy_m_s"), ("Ux", "Uy"), ("vx", "vy")]:
        if ux_key in arrays and uy_key in arrays:
            ux = arrays[ux_key].astype(np.float32).copy()
            uy = arrays[uy_key].astype(np.float32).copy()
            mag = np.sqrt(ux * ux + uy * uy)
            zero_dir = perfused & (mag <= 0)
            ux[zero_dir] = speed_safe[zero_dir]
            uy[zero_dir] = 0.0
            scale_mask = perfused & (mag > 0) & (mag < args.speed_floor)
            scale = np.ones_like(speed_safe, dtype=np.float32)
            scale[scale_mask] = args.speed_floor / np.maximum(mag[scale_mask], 1e-30)
            ux[scale_mask] *= scale[scale_mask]
            uy[scale_mask] *= scale[scale_mask]
            ux[~perfused] = 0.0
            uy[~perfused] = 0.0
            arrays[ux_key] = ux.astype(np.float32)
            arrays[uy_key] = uy.astype(np.float32)

    # Radius and pressure should not be active outside perfused lumen mask for this solver.
    if "radius_m" in arrays:
        r = arrays["radius_m"].astype(np.float32).copy()
        r[~perfused] = 0.0
        arrays["radius_m"] = r
    if "pressures" in arrays:
        p = arrays["pressures"].astype(np.float32).copy()
        p[~perfused] = 0.0
        arrays["pressures"] = p
    if "pressure_map" in arrays:
        p = arrays["pressure_map"].astype(np.float32).copy()
        p[~perfused] = 0.0
        arrays["pressure_map"] = p

    # Recompute transit time safely.
    if "transit_time" in arrays:
        tt = np.zeros_like(speed_safe, dtype=np.float32)
        tt[perfused] = 1.0 / np.maximum(speed_safe[perfused], args.speed_floor)
        arrays["transit_time"] = tt

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)

    # Report exact sanity checks.
    out = np.load(args.output)
    vessel = out["vessel_mask"].astype(bool) if "vessel_mask" in out.files else out["skel"].astype(bool)
    zero_inside = int(np.logical_and(vessel, out["speed_map"] <= 0).sum())
    print(f"[OK] wrote: {args.output}")
    print(f"[OK] perfused pixels kept: {int(perfused.sum())}")
    print(f"[OK] vessel pixels in safe cache: {int(vessel.sum())}")
    print(f"[OK] zero-speed pixels inside vessel_mask: {zero_inside}")
    print(f"[OK] speed floor used only in perfused pixels: {args.speed_floor:g} m/s")
    if zero_inside != 0:
        raise RuntimeError("Still has zero-speed pixels inside vessel_mask.")


if __name__ == "__main__":
    main()
