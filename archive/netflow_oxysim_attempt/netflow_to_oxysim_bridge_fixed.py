#!/usr/bin/env python3
r"""
netflow_to_oxysim_bridge_fixed.py

Drop-in replacement for the NetFlow -> OxySim bridge.

Critical fixes:
  1. Boundary roles are inferred from actual NetFlow flow sign, not PO2 labels,
     pressure labels, or StartNode/EndNode order.
  2. Outlet mask is explicitly rasterized from boundary nodes with outward flux.
  3. True perfusion is reported from raw NetFlow q, separately from any u_min floor.
  4. NetworkNew.dat files that lost the extra PO2 columns are still handled correctly.

Example:
  python scripts/netflow_to_oxysim_bridge_fixed.py ^
    --input results_oxy\ACJ01-T343-B1S-CLAHE\netflow_results\NetworkNew.dat ^
    --output-dir results_oxy\ACJ01-T343-B1S-CLAHE\bridge_fixed ^
    --grid-nx 384 --grid-ny 384 --stem ACJ01-T343-B1S-CLAHE ^
    --u-min 0.0001 --viscosity-cp 0.7
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def _first_int(line: str) -> int | None:
    parts = line.replace("\t", " ").split()
    if not parts:
        return None
    try:
        return int(float(parts[0]))
    except ValueError:
        return None


def _find_line(lines: List[str], required: List[str], start: int = 0) -> int | None:
    required = [r.lower() for r in required]
    for i in range(start, len(lines)):
        low = lines[i].lower()
        if all(r in low for r in required):
            return i
    return None


def parse_netflow_network(path: Path):
    lines = path.read_text(errors="ignore").splitlines()

    if len(lines) < 10:
        raise ValueError(f"File too short to be a NetFlow network: {path}")

    # Global dimensions. NetFlow uses microns.
    box_parts = lines[1].split()
    try:
        Lx_um, Ly_um, Lz_um = map(float, box_parts[:3])
    except Exception as e:
        raise ValueError("Could not parse box dimensions from line 2 of Network.dat/NetworkNew.dat") from e

    seg_count_idx = _find_line(lines, ["total number of segments"])
    if seg_count_idx is None:
        raise ValueError("Could not find 'total number of segments'.")
    nseg = _first_int(lines[seg_count_idx])
    if nseg is None:
        raise ValueError("Malformed segment-count line.")

    seg_header_idx = _find_line(lines, ["segname", "startnode", "endnode"], start=seg_count_idx)
    if seg_header_idx is None:
        raise ValueError("Could not find segment table header: SegName Type StartNode EndNode ...")

    segments = []
    for raw in lines[seg_header_idx + 1 : seg_header_idx + 1 + nseg]:
        parts = raw.replace("*", " ").split()
        if len(parts) < 7:
            continue
        segments.append(
            {
                "name": int(parts[0]),
                "type": int(parts[1]),
                "start": int(parts[2]),
                "end": int(parts[3]),
                "diam_um": float(parts[4]),
                "q_nl_min": float(parts[5]),
                "hd": float(parts[6]),
            }
        )

    node_count_idx = _find_line(lines, ["number of nodes"], start=seg_header_idx + 1 + nseg)
    if node_count_idx is None:
        raise ValueError("Could not find 'number of nodes'.")
    nnod = _first_int(lines[node_count_idx])
    if nnod is None:
        raise ValueError("Malformed node-count line.")

    node_header_idx = _find_line(lines, ["name", "x", "y", "z"], start=node_count_idx)
    if node_header_idx is None:
        raise ValueError("Could not find node table header: Name x y z.")

    nodes: Dict[int, Tuple[float, float, float]] = {}
    for raw in lines[node_header_idx + 1 : node_header_idx + 1 + nnod]:
        parts = raw.replace("*", " ").split()
        if len(parts) >= 4:
            nodes[int(parts[0])] = (float(parts[1]), float(parts[2]), float(parts[3]))

    bc_count_idx = _find_line(lines, ["boundary nodes"], start=node_header_idx + 1 + nnod)
    if bc_count_idx is None:
        raise ValueError("Could not find boundary-node count.")
    nbc = _first_int(lines[bc_count_idx])
    if nbc is None:
        raise ValueError("Malformed boundary-node-count line.")

    bc_header_idx = None
    for i in range(bc_count_idx + 1, min(len(lines), bc_count_idx + 15)):
        low = lines[i].lower()
        if "node" in low and ("bctype" in low or "bctyp" in low):
            bc_header_idx = i
            break
    if bc_header_idx is None:
        raise ValueError("Could not find boundary-node table header.")

    bcs = []
    for raw in lines[bc_header_idx + 1 : bc_header_idx + 1 + nbc]:
        parts = raw.replace("*", " ").split()
        if len(parts) >= 4:
            bcs.append(
                {
                    "node": int(parts[0]),
                    "bctype": int(parts[1]),
                    "press_or_flow": float(parts[2]),
                    "hd": float(parts[3]),
                    # If NetworkNew.dat retained PO2, keep it; otherwise None/0 is irrelevant.
                    "po2": float(parts[4]) if len(parts) >= 5 else None,
                }
            )

    if len(segments) != nseg:
        raise ValueError(f"Parsed {len(segments)} segments but declared {nseg}.")
    if len(nodes) != nnod:
        raise ValueError(f"Parsed {len(nodes)} nodes but declared {nnod}.")
    if len(bcs) != nbc:
        raise ValueError(f"Parsed {len(bcs)} boundary nodes but declared {nbc}.")

    return {
        "Lx_um": Lx_um,
        "Ly_um": Ly_um,
        "Lz_um": Lz_um,
        "segments": segments,
        "nodes": nodes,
        "bcs": bcs,
    }


def node_to_grid(x_um: float, y_um: float, Lx_um: float, Ly_um: float, nx: int, ny: int) -> Tuple[int, int]:
    ix = int(round((x_um / Lx_um) * (nx - 1))) if Lx_um > 0 else 0
    iy = int(round((y_um / Ly_um) * (ny - 1))) if Ly_um > 0 else 0
    ix = max(0, min(nx - 1, ix))
    iy = max(0, min(ny - 1, iy))
    return ix, iy


def bresenham(x0: int, y0: int, x1: int, y1: int):
    """Integer line rasterization, returns list of (x,y)."""
    points = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return points


def dilate_binary(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = mask.astype(bool).copy()
    for _ in range(iterations):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        new = out.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                new |= padded[1 + dy : 1 + dy + out.shape[0], 1 + dx : 1 + dx + out.shape[1]]
        out = new
    return out


def q_to_speed_m_s(q_nl_min: float, diam_um: float) -> float:
    # 1 nl = 1e-12 m^3. q in nl/min -> m^3/s.
    q_m3_s = abs(q_nl_min) * 1e-12 / 60.0
    radius_m = (diam_um * 1e-6) / 2.0
    area_m2 = math.pi * radius_m * radius_m
    if area_m2 <= 0:
        return 0.0
    return q_m3_s / area_m2


def build_bridge(network, nx: int, ny: int, u_min: float, boundary_dilate: int, vessel_dilate: int):
    Lx_um = network["Lx_um"]
    Ly_um = network["Ly_um"]
    nodes = network["nodes"]
    segments = network["segments"]
    bcs = network["bcs"]

    vessel_mask = np.zeros((ny, nx), dtype=bool)
    centerline_mask = np.zeros((ny, nx), dtype=bool)
    inlet_mask = np.zeros((ny, nx), dtype=bool)
    outlet_mask = np.zeros((ny, nx), dtype=bool)
    zero_bc_mask = np.zeros((ny, nx), dtype=bool)

    speed_m_s = np.zeros((ny, nx), dtype=np.float32)
    speed_raw_m_s = np.zeros((ny, nx), dtype=np.float32)
    ux_m_s = np.zeros((ny, nx), dtype=np.float32)
    uy_m_s = np.zeros((ny, nx), dtype=np.float32)
    q_nl_min_map = np.zeros((ny, nx), dtype=np.float32)
    diam_um_map = np.zeros((ny, nx), dtype=np.float32)
    segment_id_map = np.zeros((ny, nx), dtype=np.int32)

    active_segments = [s for s in segments if s["type"] in (4, 5)]
    active_nonzero = [s for s in active_segments if abs(s["q_nl_min"]) > 1e-12]

    # adjacency for boundary flux classification
    adj: Dict[int, List[dict]] = {}
    for s in active_segments:
        adj.setdefault(s["start"], []).append(s)
        adj.setdefault(s["end"], []).append(s)

    # Rasterize active segments. Direction follows actual flow.
    segment_rows = []
    for s in active_segments:
        if s["start"] not in nodes or s["end"] not in nodes:
            continue
        x0_um, y0_um, _ = nodes[s["start"]]
        x1_um, y1_um, _ = nodes[s["end"]]
        x0, y0 = node_to_grid(x0_um, y0_um, Lx_um, Ly_um, nx, ny)
        x1, y1 = node_to_grid(x1_um, y1_um, Lx_um, Ly_um, nx, ny)

        pts = bresenham(x0, y0, x1, y1)
        raw_speed = q_to_speed_m_s(s["q_nl_min"], s["diam_um"])
        solver_speed = raw_speed if raw_speed > 0 else 0.0
        if solver_speed > 0 and solver_speed < u_min:
            solver_speed = u_min

        # Actual flow direction: q>0 start->end, q<0 end->start.
        if s["q_nl_min"] >= 0:
            dx_um = x1_um - x0_um
            dy_um = y1_um - y0_um
        else:
            dx_um = x0_um - x1_um
            dy_um = y0_um - y1_um
        norm = math.hypot(dx_um, dy_um)
        ex = dx_um / norm if norm > 0 else 0.0
        ey = dy_um / norm if norm > 0 else 0.0

        for x, y in pts:
            centerline_mask[y, x] = True
            # Keep the largest-speed segment where multiple centerlines overlap.
            if solver_speed >= speed_m_s[y, x]:
                speed_m_s[y, x] = solver_speed
                speed_raw_m_s[y, x] = raw_speed
                ux_m_s[y, x] = solver_speed * ex
                uy_m_s[y, x] = solver_speed * ey
                q_nl_min_map[y, x] = s["q_nl_min"]
                diam_um_map[y, x] = s["diam_um"]
                segment_id_map[y, x] = s["name"]

        segment_rows.append({
            "segname": s["name"],
            "type": s["type"],
            "start": s["start"],
            "end": s["end"],
            "diam_um": s["diam_um"],
            "q_nl_min": s["q_nl_min"],
            "raw_speed_m_s": raw_speed,
            "raw_speed_mm_s": raw_speed * 1000.0,
        })

    vessel_mask = dilate_binary(centerline_mask, iterations=vessel_dilate) if vessel_dilate > 0 else centerline_mask.copy()

    bc_rows = []
    inlets = []
    outlets = []
    unused = []
    for bc in bcs:
        node = bc["node"]
        flux_into_network = 0.0
        for s in adj.get(node, []):
            # q>0 means start -> end. Positive boundary flux means enters network from boundary.
            if s["start"] == node:
                flux_into_network += s["q_nl_min"]
            elif s["end"] == node:
                flux_into_network += -s["q_nl_min"]

        if node not in nodes:
            continue
        x_um, y_um, z_um = nodes[node]
        ix, iy = node_to_grid(x_um, y_um, Lx_um, Ly_um, nx, ny)

        if flux_into_network > 1e-6:
            role = "INLET"
            inlet_mask[iy, ix] = True
            inlets.append(node)
        elif flux_into_network < -1e-6:
            role = "OUTLET"
            outlet_mask[iy, ix] = True
            outlets.append(node)
        else:
            role = "ZERO_OR_UNUSED"
            zero_bc_mask[iy, ix] = True
            unused.append(node)

        bc_rows.append({
            "node": node,
            "role_from_q": role,
            "flux_into_network_nl_min": flux_into_network,
            "pressure_or_flow": bc["press_or_flow"],
            "po2_in_file": bc["po2"],
            "x_um": x_um,
            "y_um": y_um,
            "grid_x": ix,
            "grid_y": iy,
            "degree_active": len(adj.get(node, [])),
        })

    if boundary_dilate > 0:
        inlet_mask = dilate_binary(inlet_mask, iterations=boundary_dilate)
        outlet_mask = dilate_binary(outlet_mask, iterations=boundary_dilate)
        zero_bc_mask = dilate_binary(zero_bc_mask, iterations=boundary_dilate)

    q_in = sum(r["flux_into_network_nl_min"] for r in bc_rows if r["flux_into_network_nl_min"] > 0)
    q_out = -sum(r["flux_into_network_nl_min"] for r in bc_rows if r["flux_into_network_nl_min"] < 0)

    # Internal mass conservation.
    node_balance: Dict[int, float] = {}
    for s in active_segments:
        node_balance[s["start"]] = node_balance.get(s["start"], 0.0) + s["q_nl_min"]
        node_balance[s["end"]] = node_balance.get(s["end"], 0.0) - s["q_nl_min"]
    bc_nodes = {bc["node"] for bc in bcs}
    internal_residuals = [abs(v) for n, v in node_balance.items() if n not in bc_nodes and len(adj.get(n, [])) > 1]
    worst_internal_residual = max(internal_residuals) if internal_residuals else 0.0

    true_perfused_centerline = np.logical_and(centerline_mask, speed_raw_m_s > 0)
    true_perfused_pct = 100.0 * true_perfused_centerline.sum() / max(centerline_mask.sum(), 1)
    solver_active_pct = 100.0 * np.logical_and(centerline_mask, speed_m_s > 0).sum() / max(centerline_mask.sum(), 1)

    raw_speeds = np.array([q_to_speed_m_s(s["q_nl_min"], s["diam_um"]) for s in active_segments], dtype=float)
    diameters = np.array([s["diam_um"] for s in active_segments], dtype=float)

    meta = {
        "source": "netflow_to_oxysim_bridge_fixed.py",
        "Lx_um": Lx_um,
        "Ly_um": Ly_um,
        "Lz_um": network["Lz_um"],
        "Lx_m": Lx_um * 1e-6,
        "Ly_m": Ly_um * 1e-6,
        "nx": nx,
        "ny": ny,
        "pixel_size_x_um": Lx_um / nx,
        "pixel_size_y_um": Ly_um / ny,
        "n_segments": len(segments),
        "n_active_segments": len(active_segments),
        "n_active_nonzero_segments": len(active_nonzero),
        "n_boundary_nodes": len(bcs),
        "n_inlet_nodes_from_q": len(inlets),
        "n_outlet_nodes_from_q": len(outlets),
        "n_zero_or_unused_boundary_nodes": len(unused),
        "inlet_nodes": inlets,
        "outlet_nodes": outlets,
        "zero_or_unused_boundary_nodes": unused,
        "inlet_pixels": int(inlet_mask.sum()),
        "outlet_pixels": int(outlet_mask.sum()),
        "zero_or_unused_boundary_pixels": int(zero_bc_mask.sum()),
        "skeleton_pixels": int(centerline_mask.sum()),
        "vessel_pixels": int(vessel_mask.sum()),
        "true_perfused_vessel_pct_raw_q": true_perfused_pct,
        "solver_active_vessel_pct_after_u_min": solver_active_pct,
        "boundary_inflow_nl_min": q_in,
        "boundary_outflow_nl_min": q_out,
        "boundary_balance_nl_min": q_in - q_out,
        "worst_internal_mass_residual_nl_min": worst_internal_residual,
        "mean_diameter_um_bridge": float(np.mean(diameters)) if len(diameters) else 0.0,
        "mean_speed_mm_s_bridge_raw": float(np.mean(raw_speeds) * 1000.0) if len(raw_speeds) else 0.0,
        "median_speed_mm_s_bridge_raw": float(np.median(raw_speeds) * 1000.0) if len(raw_speeds) else 0.0,
        "max_speed_mm_s_bridge_raw": float(np.max(raw_speeds) * 1000.0) if len(raw_speeds) else 0.0,
        "u_min_m_s": u_min,
        "boundary_dilate_px": boundary_dilate,
        "vessel_dilate_px": vessel_dilate,
    }

    arrays = {
        # Canonical masks
        "vessel_mask": vessel_mask,
        "centerline_mask": centerline_mask,
        "skeleton_mask": centerline_mask,
        "skel_mask": centerline_mask,
        "inlet_mask": inlet_mask,
        "outlet_mask": outlet_mask,
        "zero_bc_mask": zero_bc_mask,
        # Velocity fields, several aliases for compatibility.
        "speed_m_s": speed_m_s,
        "speed": speed_m_s,
        "u": speed_m_s,
        "speed_raw_m_s": speed_raw_m_s,
        "ux_m_s": ux_m_s,
        "uy_m_s": uy_m_s,
        "Ux": ux_m_s,
        "Uy": uy_m_s,
        "vx": ux_m_s,
        "vy": uy_m_s,
        # Segment scalar maps
        "q_nl_min_map": q_nl_min_map,
        "diam_um_map": diam_um_map,
        "diameter_um": diam_um_map,
        "radius_um_map": diam_um_map / 2.0,
        "segment_id_map": segment_id_map,
        # Metadata as arrays too, for npz-only readers.
        "Lx_m": np.array(meta["Lx_m"], dtype=np.float64),
        "Ly_m": np.array(meta["Ly_m"], dtype=np.float64),
        "nx": np.array(nx, dtype=np.int32),
        "ny": np.array(ny, dtype=np.int32),
    }

    return arrays, meta, bc_rows, segment_rows


def save_png(mask_or_float: np.ndarray, path: Path):
    arr = mask_or_float
    if arr.dtype == bool:
        out = (arr.astype(np.uint8) * 255)
    else:
        finite = np.isfinite(arr)
        out = np.zeros(arr.shape, dtype=np.uint8)
        if finite.any() and float(np.nanmax(arr)) > float(np.nanmin(arr)):
            scaled = (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr))
            out = np.clip(scaled * 255, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path, help="NetFlowV2 Current/NetworkNew.dat")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--grid-nx", type=int, default=384)
    ap.add_argument("--grid-ny", type=int, default=384)
    ap.add_argument("--stem", default="netflow")
    ap.add_argument("--u-min", type=float, default=0.0001, help="Minimum nonzero solver speed in m/s; raw perfusion is still reported separately.")
    ap.add_argument("--viscosity-cp", type=float, default=0.7)
    ap.add_argument("--boundary-dilate", type=int, default=2)
    ap.add_argument("--vessel-dilate", type=int, default=1)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    net = parse_netflow_network(args.input)
    arrays, meta, bc_rows, segment_rows = build_bridge(
        net,
        nx=args.grid_nx,
        ny=args.grid_ny,
        u_min=args.u_min,
        boundary_dilate=args.boundary_dilate,
        vessel_dilate=args.vessel_dilate,
    )
    meta["input"] = str(args.input)
    meta["viscosity_cp"] = args.viscosity_cp

    hemo_path = args.output_dir / f"{args.stem}_hemo.npz"
    np.savez_compressed(hemo_path, **arrays)

    meta_path = args.output_dir / f"{args.stem}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Compatibility mask name used by the previous bridge.
    save_png(arrays["vessel_mask"], args.output_dir / f"{args.stem}_mask.png")
    save_png(arrays["centerline_mask"], args.output_dir / f"{args.stem}_centerline_mask.png")
    save_png(arrays["inlet_mask"], args.output_dir / f"{args.stem}_inlet_mask.png")
    save_png(arrays["outlet_mask"], args.output_dir / f"{args.stem}_outlet_mask.png")
    save_png(arrays["speed_m_s"], args.output_dir / f"{args.stem}_speed.png")

    with (args.output_dir / f"{args.stem}_boundary_table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(bc_rows[0].keys()))
        writer.writeheader()
        writer.writerows(bc_rows)

    with (args.output_dir / f"{args.stem}_segment_table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(segment_rows[0].keys()))
        writer.writeheader()
        writer.writerows(segment_rows)

    print("[BRIDGE-FIXED] input      :", args.input)
    print("[BRIDGE-FIXED] hemo npz   :", hemo_path)
    print("[BRIDGE-FIXED] metadata   :", meta_path)
    print(f"[BRIDGE-FIXED] active segments = {meta['n_active_segments']} | nonzero = {meta['n_active_nonzero_segments']}")
    print(f"[BRIDGE-FIXED] inlet nodes = {meta['n_inlet_nodes_from_q']} | outlet nodes = {meta['n_outlet_nodes_from_q']} | unused BC = {meta['n_zero_or_unused_boundary_nodes']}")
    print(f"[BRIDGE-FIXED] inlet pixels = {meta['inlet_pixels']} | outlet pixels = {meta['outlet_pixels']}")
    print(f"[BRIDGE-FIXED] boundary balance = {meta['boundary_balance_nl_min']:.6g} nl/min")
    print(f"[BRIDGE-FIXED] worst internal residual = {meta['worst_internal_mass_residual_nl_min']:.6g} nl/min")
    print(f"[BRIDGE-FIXED] true perfused centerline from raw q = {meta['true_perfused_vessel_pct_raw_q']:.3f}%")
    print(f"[BRIDGE-FIXED] solver-active centerline after u_min = {meta['solver_active_vessel_pct_after_u_min']:.3f}%")
    print(f"[BRIDGE-FIXED] max speed raw = {meta['max_speed_mm_s_bridge_raw']:.3f} mm/s")

    if meta["outlet_pixels"] <= 0:
        raise SystemExit("[BRIDGE-FIXED:FAIL] outlet_pixels is still zero. Do not run OxySim.")
    if meta["n_outlet_nodes_from_q"] <= 0:
        raise SystemExit("[BRIDGE-FIXED:FAIL] no outlet nodes from q sign. Do not run OxySim.")


if __name__ == "__main__":
    main()
