#!/usr/bin/env python3
"""
netflow_to_oxysim_bridge_final.py
=================================
Convert NetFlowV2 Current/NetworkNew.dat into the hemodynamic cache expected
by oxy_sim.py, without using GreensV4 and without recomputing image-based
Poiseuille flow inside oxy_sim.

Outputs
-------
<stem>_hemo.npz
    vessel_mask, skel, speed_map, pressures, radius_m, inlet_mask,
    outlet_mask, reachable, perfused_vessel, transit_time, lift_ix, lift_iy,
    plus metadata scalars.
<stem>_mask.png
    vessel mask for visual checking only.
<stem>_meta.json
    dimensions, counts, velocity statistics, and bridge diagnostics.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
from scipy import sparse
from scipy.ndimage import distance_transform_edt, binary_dilation
from scipy.sparse.linalg import spsolve

NodeId = int


def parse_network(path: Path) -> dict:
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 10:
        raise RuntimeError(f"{path} is too short to be a NetFlow network file.")

    title = lines[0].strip()
    alx, aly, alz = map(float, lines[1].split()[:3])
    mxx, myy, mzz = map(int, lines[2].split()[:3])
    lb = float(lines[3].split()[0])
    maxl = float(lines[4].split()[0])
    nodsegm = int(lines[5].split()[0])
    nseg = int(lines[6].split()[0])

    segs = []
    for i in range(8, 8 + nseg):
        parts = lines[i].replace("*", "").split()
        if len(parts) < 7:
            continue
        try:
            segs.append({
                "name": int(parts[0]),
                "type": int(parts[1]),
                "n1": int(parts[2]),
                "n2": int(parts[3]),
                "diam_um": float(parts[4]),
                "flow_nl_min": float(parts[5]),
                "hd": float(parts[6]),
            })
        except ValueError:
            continue

    node_count_idx = 8 + nseg
    nnod = int(lines[node_count_idx].split()[0])
    nodes: Dict[int, Tuple[float, float, float]] = {}
    for i in range(node_count_idx + 2, node_count_idx + 2 + nnod):
        parts = lines[i].replace("*", "").split()
        if len(parts) < 4:
            continue
        try:
            nodes[int(parts[0])] = (float(parts[1]), float(parts[2]), float(parts[3]))
        except ValueError:
            continue

    bc_count_idx = node_count_idx + 2 + nnod
    bcs = []
    if bc_count_idx < len(lines):
        try:
            nbc = int(lines[bc_count_idx].split()[0])
        except Exception:
            nbc = 0
        for i in range(bc_count_idx + 2, bc_count_idx + 2 + nbc):
            if i >= len(lines):
                break
            parts = lines[i].replace("*", "").split()
            if len(parts) < 4:
                continue
            try:
                bcs.append({
                    "node": int(parts[0]),
                    "bctype": int(parts[1]),
                    "value": float(parts[2]),
                    "hd": float(parts[3]),
                    "po2": float(parts[4]) if len(parts) >= 5 else 0.0,
                })
            except ValueError:
                continue

    return {
        "title": title,
        "alx_um": alx,
        "aly_um": aly,
        "alz_um": alz,
        "mxx": mxx,
        "myy": myy,
        "mzz": mzz,
        "lb": lb,
        "maxl": maxl,
        "nodsegm": nodsegm,
        "nseg_header": nseg,
        "segs": segs,
        "nodes": nodes,
        "bcs": bcs,
    }


def active_segments(net: dict) -> List[dict]:
    return [s for s in net["segs"] if s["type"] in (4, 5)
            and s["n1"] in net["nodes"] and s["n2"] in net["nodes"]]


def segment_length_um(seg: dict, nodes: Dict[int, Tuple[float, float, float]]) -> float:
    x1, y1, z1 = nodes[seg["n1"]]
    x2, y2, z2 = nodes[seg["n2"]]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def build_components(segs: List[dict]) -> List[set]:
    adj: Dict[int, set] = defaultdict(set)
    for s in segs:
        adj[s["n1"]].add(s["n2"])
        adj[s["n2"]].add(s["n1"])
    seen = set()
    comps = []
    for start in adj:
        if start in seen:
            continue
        q = deque([start])
        comp = set()
        seen.add(start)
        while q:
            n = q.popleft()
            comp.add(n)
            for nb in adj[n]:
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        comps.append(comp)
    return comps


def reconstruct_pressures(net: dict, viscosity_cp: float = 0.7) -> Dict[int, float]:
    """
    Reconstruct nodal pressures by solving Kirchhoff pressure equations using
    fixed-pressure boundary nodes from NetworkNew.dat.

    The absolute NetFlow conversion factor cancels for pressure-only boundary
    conditions. A constant viscosity factor is retained only for numerical
    consistency if mixed conductance scaling is later required.
    """
    nodes = net["nodes"]
    segs = [s for s in active_segments(net) if segment_length_um(s, nodes) > 1e-9]
    fixed = {bc["node"]: bc["value"] for bc in net["bcs"] if bc.get("bctype", 0) == 0}
    pressure = {nid: float("nan") for nid in nodes}

    if not segs:
        return {nid: 0.0 for nid in nodes}

    adjacency: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for s in segs:
        L = max(segment_length_um(s, nodes), 1e-12)
        d = max(float(s["diam_um"]), 1e-6)
        # Relative conductance. NetFlow uses d^4/(L*visc); global constants are unnecessary.
        g = (d ** 4) / (L * max(viscosity_cp, 1e-12))
        adjacency[s["n1"]].append((s["n2"], g))
        adjacency[s["n2"]].append((s["n1"], g))

    for comp in build_components(segs):
        comp_fixed = {n: fixed[n] for n in comp if n in fixed}
        if not comp_fixed:
            for n in comp:
                pressure[n] = 0.0
            continue
        comp_nodes = sorted(comp)
        idx = {n: i for i, n in enumerate(comp_nodes)}
        rows, cols, vals = [], [], []
        b = np.zeros(len(comp_nodes), dtype=np.float64)
        for n in comp_nodes:
            i = idx[n]
            if n in comp_fixed:
                rows.append(i); cols.append(i); vals.append(1.0)
                b[i] = comp_fixed[n]
                continue
            gsum = 0.0
            for nb, g in adjacency.get(n, []):
                if nb not in idx:
                    continue
                rows.append(i); cols.append(idx[nb]); vals.append(-g)
                gsum += g
            rows.append(i); cols.append(i); vals.append(gsum + 1e-30)
        A = sparse.csr_matrix((vals, (rows, cols)), shape=(len(comp_nodes), len(comp_nodes)))
        sol = spsolve(A, b)
        for n, p in zip(comp_nodes, sol):
            pressure[n] = float(p) if np.isfinite(p) else 0.0

    for n in list(pressure):
        if not np.isfinite(pressure[n]):
            pressure[n] = fixed.get(n, 0.0)
    return pressure


def xy_to_grid(x_um: float, y_um: float, alx_um: float, aly_um: float,
               nx: int, ny: int) -> Tuple[int, int]:
    ix = int(round((x_um / max(alx_um, 1e-12)) * (nx - 1)))
    iy = int(round((y_um / max(aly_um, 1e-12)) * (ny - 1)))
    return int(np.clip(ix, 0, nx - 1)), int(np.clip(iy, 0, ny - 1))


def bresenham(ix0: int, iy0: int, ix1: int, iy1: int) -> Iterable[Tuple[int, int]]:
    dx = abs(ix1 - ix0)
    dy = abs(iy1 - iy0)
    sx = 1 if ix0 < ix1 else -1
    sy = 1 if iy0 < iy1 else -1
    err = dx - dy
    x, y = ix0, iy0
    while True:
        yield x, y
        if x == ix1 and y == iy1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def disk_offsets(r_px: int) -> List[Tuple[int, int]]:
    r_px = max(1, int(r_px))
    out = []
    for dx in range(-r_px, r_px + 1):
        for dy in range(-r_px, r_px + 1):
            if dx * dx + dy * dy <= r_px * r_px:
                out.append((dx, dy))
    return out


def rasterize(net: dict, pressure: Dict[int, float], nx: int, ny: int,
              min_radius_um: float = 1.0) -> dict:
    alx, aly = float(net["alx_um"]), float(net["aly_um"])
    px_um_x = alx / max(nx - 1, 1)
    px_um_y = aly / max(ny - 1, 1)
    px_um = 0.5 * (px_um_x + px_um_y)

    vessel_mask = np.zeros((nx, ny), dtype=bool)
    skel = np.zeros((nx, ny), dtype=bool)
    radius_m = np.zeros((nx, ny), dtype=np.float32)
    speed_map = np.zeros((nx, ny), dtype=np.float32)
    pressures = np.zeros((nx, ny), dtype=np.float32)
    pressure_weight = np.zeros((nx, ny), dtype=np.float32)

    for seg in active_segments(net):
        L_um = segment_length_um(seg, net["nodes"])
        if L_um <= 1e-9:
            continue
        x1, y1, _ = net["nodes"][seg["n1"]]
        x2, y2, _ = net["nodes"][seg["n2"]]
        ix1, iy1 = xy_to_grid(x1, y1, alx, aly, nx, ny)
        ix2, iy2 = xy_to_grid(x2, y2, alx, aly, nx, ny)
        pts = list(bresenham(ix1, iy1, ix2, iy2))
        if not pts:
            continue

        r_um = max(float(seg["diam_um"]) / 2.0, min_radius_um)
        r_m = r_um * 1e-6
        r_px = max(1, int(round(r_um / max(px_um, 1e-12))))
        offsets = disk_offsets(r_px)
        q_m3_s = abs(float(seg["flow_nl_min"])) * 1e-12 / 60.0
        area_m2 = math.pi * r_m * r_m
        speed = q_m3_s / max(area_m2, 1e-20)
        p1 = pressure.get(seg["n1"], 0.0)
        p2 = pressure.get(seg["n2"], 0.0)

        for k, (ix, iy) in enumerate(pts):
            t = k / max(len(pts) - 1, 1)
            p_here = (1.0 - t) * p1 + t * p2
            skel[ix, iy] = True
            if r_m > radius_m[ix, iy]:
                radius_m[ix, iy] = r_m
            if speed > speed_map[ix, iy]:
                speed_map[ix, iy] = speed
            pressures[ix, iy] += p_here
            pressure_weight[ix, iy] += 1.0
            for ox, oy in offsets:
                jx, jy = ix + ox, iy + oy
                if 0 <= jx < nx and 0 <= jy < ny:
                    vessel_mask[jx, jy] = True
                    if r_m > radius_m[jx, jy]:
                        radius_m[jx, jy] = r_m

    nonzero_p = pressure_weight > 0
    pressures[nonzero_p] = pressures[nonzero_p] / pressure_weight[nonzero_p]
    radius_m[vessel_mask & (radius_m <= 0)] = min_radius_um * 1e-6

    # Light dilation prevents isolated holes from thick line rasterization.
    vessel_mask = binary_dilation(vessel_mask, structure=np.ones((3, 3), dtype=bool))
    radius_m[vessel_mask & (radius_m <= 0)] = min_radius_um * 1e-6

    return {
        "vessel_mask": vessel_mask,
        "skel": skel,
        "radius_m": radius_m,
        "speed_map": speed_map,
        "pressures": pressures,
    }


def neighbors8(x: int, y: int, nx: int, ny: int) -> Iterable[Tuple[int, int, float]]:
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            xx, yy = x + dx, y + dy
            if 0 <= xx < nx and 0 <= yy < ny:
                yield xx, yy, math.sqrt(dx * dx + dy * dy)


def snap_bc_masks(net: dict, skel: np.ndarray, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    inlet = np.zeros_like(skel, dtype=bool)
    outlet = np.zeros_like(skel, dtype=bool)
    if not np.any(skel):
        return inlet, outlet
    _, (nearest_x, nearest_y) = distance_transform_edt(~skel, return_indices=True)
    pressures = [bc["value"] for bc in net["bcs"] if bc.get("bctype", 0) == 0]
    pmax = max(pressures) if pressures else None
    alx, aly = float(net["alx_um"]), float(net["aly_um"])
    for bc in net["bcs"]:
        if bc["node"] not in net["nodes"]:
            continue
        x, y, _ = net["nodes"][bc["node"]]
        ix, iy = xy_to_grid(x, y, alx, aly, nx, ny)
        sx, sy = int(nearest_x[ix, iy]), int(nearest_y[ix, iy])
        if bc.get("bctype", 0) == 0 and pmax is not None and abs(bc["value"] - pmax) < 1e-6:
            inlet[sx, sy] = True
        else:
            outlet[sx, sy] = True
    outlet &= ~inlet
    return inlet, outlet


def reachable_bfs(skel: np.ndarray, inlet: np.ndarray) -> np.ndarray:
    reach = np.zeros_like(skel, dtype=bool)
    q = deque()
    for x, y in zip(*np.where(inlet & skel)):
        reach[x, y] = True
        q.append((int(x), int(y)))
    nx, ny = skel.shape
    while q:
        x, y = q.popleft()
        for xx, yy, _ in neighbors8(x, y, nx, ny):
            if skel[xx, yy] and not reach[xx, yy]:
                reach[xx, yy] = True
                q.append((xx, yy))
    return reach


def dijkstra_transit(skel: np.ndarray, inlet: np.ndarray, speed_map: np.ndarray,
                     dx_m: float, dy_m: float, u_min: float) -> np.ndarray:
    nx, ny = skel.shape
    inf = np.float64(1e30)
    T = np.full((nx, ny), inf, dtype=np.float64)
    pq = []
    for x, y in zip(*np.where(inlet & skel)):
        T[x, y] = 0.0
        heapq.heappush(pq, (0.0, int(x), int(y)))
    while pq:
        t, x, y = heapq.heappop(pq)
        if t > T[x, y]:
            continue
        for xx, yy, _ in neighbors8(x, y, nx, ny):
            if not skel[xx, yy]:
                continue
            ds = math.hypot((xx - x) * dx_m, (yy - y) * dy_m)
            u = max(float(speed_map[xx, yy]), float(u_min))
            nt = t + ds / u
            if nt < T[xx, yy]:
                T[xx, yy] = nt
                heapq.heappush(pq, (nt, xx, yy))
    return T


def build_cache(net: dict, ras: dict, u_min: float) -> dict:
    vessel_mask = ras["vessel_mask"]
    skel = ras["skel"]
    speed_map = ras["speed_map"]
    nx, ny = skel.shape
    inlet, outlet = snap_bc_masks(net, skel, nx, ny)
    if not np.any(inlet & skel) and np.any(skel):
        # deterministic fallback: choose skeleton point with maximum pressure
        p = ras["pressures"].copy()
        p[~skel] = -1e9
        ix, iy = np.unravel_index(int(np.argmax(p)), p.shape)
        inlet[ix, iy] = True
    reachable = reachable_bfs(skel, inlet)

    if np.any(reachable):
        _, (lift_ix, lift_iy) = distance_transform_edt(~reachable, return_indices=True)
        perfused_vessel = vessel_mask.copy()
        # A vessel pixel is perfused if it can be lifted to a reachable skeleton pixel.
        perfused_vessel[vessel_mask] = reachable[lift_ix[vessel_mask], lift_iy[vessel_mask]]
    else:
        lift_ix = np.zeros_like(skel, dtype=np.int32)
        lift_iy = np.zeros_like(skel, dtype=np.int32)
        perfused_vessel = np.zeros_like(vessel_mask, dtype=bool)

    dx_m = float(net["alx_um"]) * 1e-6 / max(nx - 1, 1)
    dy_m = float(net["aly_um"]) * 1e-6 / max(ny - 1, 1)
    transit = dijkstra_transit(skel, inlet, speed_map, dx_m, dy_m, u_min)

    return {
        **ras,
        "inlet_mask": inlet,
        "outlet_mask": outlet,
        "reachable": reachable,
        "perfused_vessel": perfused_vessel,
        "transit_time": transit,
        "lift_ix": lift_ix.astype(np.int32),
        "lift_iy": lift_iy.astype(np.int32),
    }


def save_outputs(args: argparse.Namespace, net: dict, cache: dict) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or Path(args.input).stem.replace("NetworkNew", Path(args.input).parent.parent.name)

    mask_path = args.output_dir / f"{stem}_mask.png"
    # Transpose only for visual display; the NPZ keeps oxy_sim convention (nx, ny).
    Image.fromarray((cache["vessel_mask"].T.astype(np.uint8) * 255)).save(mask_path)

    npz_path = args.output_dir / f"{stem}_hemo.npz"
    np.savez_compressed(
        npz_path,
        vessel_mask=cache["vessel_mask"].astype(bool),
        skel=cache["skel"].astype(bool),
        speed_map=cache["speed_map"].astype(np.float32),
        pressures=cache["pressures"].astype(np.float32),
        radius_m=cache["radius_m"].astype(np.float32),
        inlet_mask=cache["inlet_mask"].astype(bool),
        outlet_mask=cache["outlet_mask"].astype(bool),
        reachable=cache["reachable"].astype(bool),
        perfused_vessel=cache["perfused_vessel"].astype(bool),
        transit_time=cache["transit_time"].astype(np.float64),
        lift_ix=cache["lift_ix"].astype(np.int32),
        lift_iy=cache["lift_iy"].astype(np.int32),
        Lx_m=np.array(float(net["alx_um"]) * 1e-6),
        Ly_m=np.array(float(net["aly_um"]) * 1e-6),
        alx_um=np.array(float(net["alx_um"])),
        aly_um=np.array(float(net["aly_um"])),
        grid_nx=np.array(int(args.grid_nx)),
        grid_ny=np.array(int(args.grid_ny)),
    )

    segs = active_segments(net)
    speeds = cache["speed_map"][cache["skel"] & (cache["speed_map"] > 0)]
    radii = cache["radius_m"][cache["vessel_mask"]]
    meta = {
        "image_stem": stem,
        "input": str(args.input),
        "mask_png": str(mask_path),
        "hemo_npz": str(npz_path),
        "alx_um": float(net["alx_um"]),
        "aly_um": float(net["aly_um"]),
        "Lx_m": float(net["alx_um"]) * 1e-6,
        "Ly_m": float(net["aly_um"]) * 1e-6,
        "grid_nx": int(args.grid_nx),
        "grid_ny": int(args.grid_ny),
        "n_segments_header": int(net["nseg_header"]),
        "n_active_segments": int(len(segs)),
        "n_nodes": int(len(net["nodes"])),
        "n_bcs": int(len(net["bcs"])),
        "n_skel_pixels": int(np.sum(cache["skel"])),
        "n_reachable_skel_pixels": int(np.sum(cache["reachable"])),
        "reachable_fraction_skel": float(np.sum(cache["reachable"]) / max(1, np.sum(cache["skel"]))),
        "perfused_vessel_fraction": float(np.sum(cache["perfused_vessel"]) / max(1, np.sum(cache["vessel_mask"]))),
        "n_inlet_pixels": int(np.sum(cache["inlet_mask"])),
        "n_outlet_pixels": int(np.sum(cache["outlet_mask"])),
        "mean_radius_um": float(np.mean(radii) * 1e6) if radii.size else 0.0,
        "mean_diameter_um": float(2.0 * np.mean(radii) * 1e6) if radii.size else 0.0,
        "mean_speed_mm_s": float(np.mean(speeds) * 1e3) if speeds.size else 0.0,
        "max_speed_mm_s": float(np.max(speeds) * 1e3) if speeds.size else 0.0,
        "warning": "If reachable_fraction_skel is much below 0.90, inspect BCs and raster connectivity before trusting oxygen results.",
    }
    meta_path = args.output_dir / f"{stem}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[BRIDGE] mask     : {mask_path}")
    print(f"[BRIDGE] hemo npz : {npz_path}")
    print(f"[BRIDGE] metadata : {meta_path}")
    print(f"[BRIDGE] skel reachable = {100.0 * meta['reachable_fraction_skel']:.1f}% | perfused vessel = {100.0 * meta['perfused_vessel_fraction']:.1f}%")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True, help="NetFlowV2 Current/NetworkNew.dat")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--grid-nx", type=int, default=384)
    p.add_argument("--grid-ny", type=int, default=384)
    p.add_argument("--stem", type=str, default=None)
    p.add_argument("--u-min", type=float, default=1e-4)
    p.add_argument("--viscosity-cp", type=float, default=0.7, help="constant viscosity used only for pressure reconstruction weights")
    p.add_argument("--min-radius-um", type=float, default=1.0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    net = parse_network(args.input)
    pressure = reconstruct_pressures(net, viscosity_cp=args.viscosity_cp)
    ras = rasterize(net, pressure, args.grid_nx, args.grid_ny, min_radius_um=args.min_radius_um)
    cache = build_cache(net, ras, u_min=args.u_min)
    save_outputs(args, net, cache)


if __name__ == "__main__":
    main()
