#!/usr/bin/env python3
"""
graph_oxygen_netflow_solver.py
================================

NetFlowV2 -> graph-based intravascular O2 solver -> rasterized tissue O2 screen.

Purpose
-------
This script replaces the fragile raster/skeleton lumen-transport step with a
true NetFlow-segment graph oxygen solve. It reads a NetFlowV2 NetworkNew.dat
(or Network.dat with solved flows), uses the actual segment table and actual
flow signs, solves intraluminal dissolved oxygen on the graph by a sparse
linear system, then rasterizes segment oxygen to a grid and computes a stable
Krogh-distance tissue oxygen screening map.

Claim discipline
----------------
This is a steady dissolved-O2 prototype. It is suitable for capstone-level
method development and screening. It is not a validated replacement for a full
3D Green's-function or FEM oxygen solver unless validated further.

Model
-----
For each perfused segment e with volumetric flow Q_e, radius r_e, length L_e,
and local perivascular tissue concentration C_t,e:

    C_out,e = C_t,e + (C_in,e - C_t,e) * exp(- 2*pi*r_e*P_wall*L_e / Q_e)

At each non-inlet node, incoming segment outlet concentrations are mixed by
flow-weighted averaging:

    C_node = sum(Q_in,e * C_out,e) / sum(Q_in,e)

This creates a sparse linear system A*C_nodes = b. Unlike GreensV4-style
upstream ordering, it can solve directed loops simultaneously.

Tissue screen
-------------
After vessel oxygen is solved, segment oxygen is rasterized to a 2D grid.
Tissue oxygen is estimated as an optimistic/conservative screening field:

    C_tissue(d) = max(C_nearest_vessel - Vmax*d^2/(2D), 0)

This does not solve a full reaction-diffusion PDE; it is a stable distance-
to-perfused-vessel oxygen accessibility model.

Example
-------
python graph_oxygen_netflow_solver.py \
  --network results_oxy/ACJ01-T343-B1S-CLAHE/netflow_results/NetworkNew.dat \
  --outdir results_oxy/ACJ01-T343-B1S-CLAHE/graph_oxygen \
  --grid-nx 384 --grid-ny 384 \
  --inlet-mmHg 150 --alpha 0.00134 \
  --D 1.5e-9 --Vmax 1e-4 --P-wall 3e-5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy import sparse
from scipy.ndimage import distance_transform_edt
from scipy.sparse.linalg import spsolve


@dataclass
class Segment:
    name: int
    typ: int
    n1: int
    n2: int
    diam_um: float
    q_nl_min: float
    hd: float


@dataclass
class BoundaryCondition:
    node: int
    bctype: int
    value: float
    hd: float
    po2: Optional[float]


@dataclass
class Network:
    title: str
    alx_um: float
    aly_um: float
    alz_um: float
    mxx: int
    myy: int
    mzz: int
    lb: float
    maxl: float
    nodsegm: int
    segments: List[Segment]
    nodes: Dict[int, Tuple[float, float, float]]
    bcs: List[BoundaryCondition]


def _first_number(line: str, cast=float):
    parts = line.replace("\t", " ").replace("*", " ").split()
    if not parts:
        return None
    try:
        return cast(float(parts[0]))
    except Exception:
        return None


def _find_line(lines: Sequence[str], required: Sequence[str], start: int = 0) -> Optional[int]:
    required = [r.lower() for r in required]
    for i in range(start, len(lines)):
        low = lines[i].lower()
        if all(r in low for r in required):
            return i
    return None


def parse_netflow(path: Path) -> Network:
    """Parse NetFlowV2 Network.dat / NetworkNew.dat robustly."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 10:
        raise RuntimeError(f"{path} is too short to be a NetFlow network file.")

    title = lines[0].strip()
    try:
        alx, aly, alz = map(float, lines[1].replace("*", " ").split()[:3])
        mxx, myy, mzz = map(lambda x: int(float(x)), lines[2].replace("*", " ").split()[:3])
        lb = float(lines[3].replace("*", " ").split()[0])
        maxl = float(lines[4].replace("*", " ").split()[0])
        nodsegm = int(float(lines[5].replace("*", " ").split()[0]))
    except Exception as e:
        raise RuntimeError("Could not parse NetFlow header lines 1-6.") from e

    seg_count_idx = _find_line(lines, ["total", "segments"], start=0)
    if seg_count_idx is None:
        # canonical fixed line fallback
        seg_count_idx = 6
    nseg = _first_number(lines[seg_count_idx], int)
    if nseg is None:
        raise RuntimeError("Could not parse number of segments.")

    seg_header_idx = _find_line(lines, ["segname", "startnode", "endnode"], start=seg_count_idx)
    if seg_header_idx is None:
        seg_header_idx = seg_count_idx + 1

    segments: List[Segment] = []
    i = seg_header_idx + 1
    while i < len(lines) and len(segments) < nseg:
        parts = lines[i].replace("*", " ").split()
        if len(parts) >= 7:
            try:
                segments.append(Segment(
                    name=int(float(parts[0])),
                    typ=int(float(parts[1])),
                    n1=int(float(parts[2])),
                    n2=int(float(parts[3])),
                    diam_um=float(parts[4]),
                    q_nl_min=float(parts[5]),
                    hd=float(parts[6]),
                ))
            except ValueError:
                pass
        i += 1
    if len(segments) != nseg:
        raise RuntimeError(f"Parsed {len(segments)} segments, expected {nseg}.")

    node_count_idx = _find_line(lines, ["number", "nodes"], start=i)
    if node_count_idx is None:
        # canonical fallback: first line after segment table
        node_count_idx = i
    nnod = _first_number(lines[node_count_idx], int)
    if nnod is None:
        raise RuntimeError("Could not parse number of nodes.")

    node_header_idx = _find_line(lines, ["name", "x", "y", "z"], start=node_count_idx)
    if node_header_idx is None:
        node_header_idx = node_count_idx + 1

    nodes: Dict[int, Tuple[float, float, float]] = {}
    i = node_header_idx + 1
    while i < len(lines) and len(nodes) < nnod:
        parts = lines[i].replace("*", " ").split()
        if len(parts) >= 4:
            try:
                nodes[int(float(parts[0]))] = (float(parts[1]), float(parts[2]), float(parts[3]))
            except ValueError:
                pass
        i += 1
    if len(nodes) != nnod:
        raise RuntimeError(f"Parsed {len(nodes)} nodes, expected {nnod}.")

    bc_count_idx = _find_line(lines, ["boundary", "nodes"], start=i)
    bcs: List[BoundaryCondition] = []
    if bc_count_idx is not None:
        nbc = _first_number(lines[bc_count_idx], int)
        if nbc is not None and nbc > 0:
            bc_header_idx = None
            for j in range(bc_count_idx + 1, min(len(lines), bc_count_idx + 20)):
                low = lines[j].lower()
                if "node" in low and ("bctype" in low or "bctyp" in low):
                    bc_header_idx = j
                    break
            if bc_header_idx is None:
                bc_header_idx = bc_count_idx + 1
            i = bc_header_idx + 1
            while i < len(lines) and len(bcs) < nbc:
                parts = lines[i].replace("*", " ").split()
                if len(parts) >= 4:
                    try:
                        bcs.append(BoundaryCondition(
                            node=int(float(parts[0])),
                            bctype=int(float(parts[1])),
                            value=float(parts[2]),
                            hd=float(parts[3]),
                            po2=float(parts[4]) if len(parts) >= 5 else None,
                        ))
                    except ValueError:
                        pass
                i += 1

    return Network(title, alx, aly, alz, mxx, myy, mzz, lb, maxl, nodsegm, segments, nodes, bcs)


def active_segments(net: Network, q_min_nl_min: float) -> List[Segment]:
    return [s for s in net.segments if s.typ in (4, 5) and s.n1 in net.nodes and s.n2 in net.nodes and abs(s.q_nl_min) > q_min_nl_min]


def all_active_segments(net: Network) -> List[Segment]:
    return [s for s in net.segments if s.typ in (4, 5) and s.n1 in net.nodes and s.n2 in net.nodes]


def segment_length_m(seg: Segment, net: Network) -> float:
    x1, y1, z1 = net.nodes[seg.n1]
    x2, y2, z2 = net.nodes[seg.n2]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2) * 1e-6


def q_nl_min_to_m3_s(q_nl_min: float) -> float:
    return abs(q_nl_min) * 1e-12 / 60.0


def classify_boundary_flux(net: Network, segs: Sequence[Segment], flux_tol_nl_min: float) -> Tuple[List[int], List[int], Dict[int, float]]:
    """Classify boundary nodes by actual flow sign. Positive means flux into network."""
    adj: Dict[int, List[Segment]] = defaultdict(list)
    for s in segs:
        adj[s.n1].append(s)
        adj[s.n2].append(s)

    bc_nodes = [bc.node for bc in net.bcs if bc.node in net.nodes]
    flux_into: Dict[int, float] = {}
    inlets, outlets = [], []
    for node in bc_nodes:
        val = 0.0
        for s in adj.get(node, []):
            # q > 0 means n1 -> n2.
            if s.n1 == node:
                val += s.q_nl_min
            elif s.n2 == node:
                val += -s.q_nl_min
        flux_into[node] = val
        if val > flux_tol_nl_min:
            inlets.append(node)
        elif val < -flux_tol_nl_min:
            outlets.append(node)

    return sorted(set(inlets)), sorted(set(outlets)), flux_into


def parse_int_list(text: Optional[str]) -> List[int]:
    if text is None or str(text).strip() == "":
        return []
    out = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def components_from_segments(segs: Sequence[Segment]) -> List[set[int]]:
    adj: Dict[int, set[int]] = defaultdict(set)
    for s in segs:
        adj[s.n1].add(s.n2)
        adj[s.n2].add(s.n1)
    seen = set()
    comps = []
    for start in list(adj):
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        comp = set()
        while q:
            n = q.popleft()
            comp.add(n)
            for nb in adj[n]:
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        comps.append(comp)
    return comps


def directed_edge(seg: Segment) -> Tuple[int, int]:
    """Return upstream, downstream node by actual q sign."""
    if seg.q_nl_min >= 0:
        return seg.n1, seg.n2
    return seg.n2, seg.n1


def attenuation_gamma(seg: Segment, net: Network, p_wall_m_s: float, q_floor_m3_s: float) -> Tuple[float, float]:
    L = max(segment_length_m(seg, net), 1e-12)
    r = max(seg.diam_um * 0.5e-6, 1e-12)
    Q = max(q_nl_min_to_m3_s(seg.q_nl_min), q_floor_m3_s)
    beta = 2.0 * math.pi * r * p_wall_m_s * L / Q
    # Avoid underflow warnings but preserve physical full depletion.
    gamma = math.exp(-min(beta, 700.0))
    return gamma, beta


def solve_graph_oxygen(
    net: Network,
    segs: Sequence[Segment],
    inlet_nodes: Sequence[int],
    inlet_C_mM: float,
    p_wall_m_s: float,
    segment_tissue_C: Dict[int, float],
    q_floor_m3_s: float,
) -> Tuple[Dict[int, float], Dict[int, dict]]:
    """Sparse simultaneous graph oxygen solve for dissolved oxygen concentration at nodes."""
    # Nodes participating in nonzero-flow active graph.
    node_set = sorted({s.n1 for s in segs} | {s.n2 for s in segs})
    idx = {n: i for i, n in enumerate(node_set)}
    N = len(node_set)
    inlet_set = set(inlet_nodes) & set(node_set)

    # Fix components with no inlet to zero concentration; they are unsupported.
    comp_no_source_nodes = set()
    for comp in components_from_segments(segs):
        if not (comp & inlet_set):
            comp_no_source_nodes |= comp

    incoming: Dict[int, List[Segment]] = defaultdict(list)
    outgoing: Dict[int, List[Segment]] = defaultdict(list)
    seg_info: Dict[int, dict] = {}
    for s in segs:
        u, v = directed_edge(s)
        incoming[v].append(s)
        outgoing[u].append(s)
        gamma, beta = attenuation_gamma(s, net, p_wall_m_s, q_floor_m3_s)
        seg_info[s.name] = {
            "upstream": u,
            "downstream": v,
            "gamma": gamma,
            "beta": beta,
            "Ctissue_mM": float(segment_tissue_C.get(s.name, 0.0)),
            "Q_m3_s": q_nl_min_to_m3_s(s.q_nl_min),
            "length_m": segment_length_m(s, net),
            "radius_m": max(s.diam_um * 0.5e-6, 1e-12),
        }

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    b = np.zeros(N, dtype=np.float64)

    for n in node_set:
        i = idx[n]
        if n in inlet_set:
            rows.append(i); cols.append(i); vals.append(1.0)
            b[i] = inlet_C_mM
            continue
        if n in comp_no_source_nodes:
            rows.append(i); cols.append(i); vals.append(1.0)
            b[i] = 0.0
            continue
        inc = incoming.get(n, [])
        if not inc:
            # Unclassified source-like node. Fix to zero instead of inventing oxygen.
            rows.append(i); cols.append(i); vals.append(1.0)
            b[i] = 0.0
            continue

        qsum = sum(q_nl_min_to_m3_s(s.q_nl_min) for s in inc)
        if qsum <= 0:
            rows.append(i); cols.append(i); vals.append(1.0)
            b[i] = 0.0
            continue

        rows.append(i); cols.append(i); vals.append(1.0)
        rhs = 0.0
        for s in inc:
            info = seg_info[s.name]
            u = info["upstream"]
            if u not in idx:
                continue
            w = q_nl_min_to_m3_s(s.q_nl_min) / qsum
            gamma = info["gamma"]
            Ct = info["Ctissue_mM"]
            # C_node - sum(w*gamma*C_upstream) = sum(w*(1-gamma)*Ct)
            rows.append(i); cols.append(idx[u]); vals.append(-w * gamma)
            rhs += w * (1.0 - gamma) * Ct
        b[i] = rhs

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))
    sol = spsolve(A, b)
    sol = np.asarray(sol, dtype=np.float64)
    sol[~np.isfinite(sol)] = 0.0
    sol = np.clip(sol, 0.0, max(inlet_C_mM * 1.5, inlet_C_mM + 1e-9))
    node_C = {n: float(sol[idx[n]]) for n in node_set}

    # Compute segment in/out concentrations for reporting and rasterization.
    for s in segs:
        info = seg_info[s.name]
        u = info["upstream"]
        Cin = float(node_C.get(u, 0.0))
        Ct = info["Ctissue_mM"]
        gamma = info["gamma"]
        Cout = Ct + (Cin - Ct) * gamma
        info["Cin_mM"] = Cin
        info["Cout_mM"] = float(np.clip(Cout, 0.0, max(inlet_C_mM * 1.5, inlet_C_mM + 1e-9)))
        info["Cmean_mM"] = 0.5 * (info["Cin_mM"] + info["Cout_mM"])

    return node_C, seg_info


def xy_to_grid(x_um: float, y_um: float, alx_um: float, aly_um: float, nx: int, ny: int) -> Tuple[int, int]:
    ix = int(round((x_um / max(alx_um, 1e-12)) * (nx - 1)))
    iy = int(round((y_um / max(aly_um, 1e-12)) * (ny - 1)))
    return int(np.clip(ix, 0, nx - 1)), int(np.clip(iy, 0, ny - 1))


def bresenham(ix0: int, iy0: int, ix1: int, iy1: int) -> List[Tuple[int, int]]:
    pts: List[Tuple[int, int]] = []
    dx = abs(ix1 - ix0)
    dy = abs(iy1 - iy0)
    sx = 1 if ix0 < ix1 else -1
    sy = 1 if iy0 < iy1 else -1
    err = dx - dy
    x, y = ix0, iy0
    while True:
        pts.append((x, y))
        if x == ix1 and y == iy1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return pts


def disk_offsets(radius_px: int) -> List[Tuple[int, int]]:
    r = max(0, int(radius_px))
    if r <= 0:
        return [(0, 0)]
    out = []
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                out.append((dx, dy))
    return out


def rasterize_vessel_oxygen(
    net: Network,
    segs: Sequence[Segment],
    seg_info: Dict[int, dict],
    nx: int,
    ny: int,
    min_radius_um: float,
    vessel_dilate_px: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, List[Tuple[int, int]]]]:
    """Rasterize vessel mask and dissolved vessel oxygen concentration map."""
    px_um_x = net.alx_um / max(nx - 1, 1)
    px_um_y = net.aly_um / max(ny - 1, 1)
    px_um = 0.5 * (px_um_x + px_um_y)

    vessel_mask = np.zeros((nx, ny), dtype=bool)
    centerline_mask = np.zeros((nx, ny), dtype=bool)
    Csum = np.zeros((nx, ny), dtype=np.float64)
    Wsum = np.zeros((nx, ny), dtype=np.float64)
    segment_pixels: Dict[int, List[Tuple[int, int]]] = {}

    for s in segs:
        x1, y1, _ = net.nodes[s.n1]
        x2, y2, _ = net.nodes[s.n2]
        ix1, iy1 = xy_to_grid(x1, y1, net.alx_um, net.aly_um, nx, ny)
        ix2, iy2 = xy_to_grid(x2, y2, net.alx_um, net.aly_um, nx, ny)
        pts = bresenham(ix1, iy1, ix2, iy2)
        segment_pixels[s.name] = pts
        r_um = max(s.diam_um * 0.5, min_radius_um)
        r_px = max(vessel_dilate_px, int(round(r_um / max(px_um, 1e-12))))
        offsets = disk_offsets(r_px)

        info = seg_info[s.name]
        u, v = info["upstream"], info["downstream"]
        # Need coordinates consistent with flow direction.
        if u == s.n1 and v == s.n2:
            flow_pts = pts
        else:
            flow_pts = list(reversed(pts))
        beta_total = info["beta"]
        Ct = info["Ctissue_mM"]
        Cin = info["Cin_mM"]
        npts = max(len(flow_pts) - 1, 1)

        for k, (ix, iy) in enumerate(flow_pts):
            frac = k / npts
            C_here = Ct + (Cin - Ct) * math.exp(-min(beta_total * frac, 700.0))
            centerline_mask[ix, iy] = True
            for ox, oy in offsets:
                jx, jy = ix + ox, iy + oy
                if 0 <= jx < nx and 0 <= jy < ny:
                    vessel_mask[jx, jy] = True
                    Csum[jx, jy] += C_here
                    Wsum[jx, jy] += 1.0

    vessel_C = np.zeros((nx, ny), dtype=np.float32)
    has = Wsum > 0
    vessel_C[has] = (Csum[has] / Wsum[has]).astype(np.float32)
    return vessel_mask, centerline_mask, vessel_C, segment_pixels


def tissue_screen_from_vessels(
    vessel_mask: np.ndarray,
    vessel_C: np.ndarray,
    dx_m: float,
    dy_m: float,
    D_m2_s: float,
    Vmax_mM_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if not np.any(vessel_mask):
        raise RuntimeError("Cannot compute tissue screen: vessel_mask is empty.")
    dist_m, (ix, iy) = distance_transform_edt(~vessel_mask, sampling=(dx_m, dy_m), return_indices=True)
    Cv_near = vessel_C[ix, iy]
    tissue_C = Cv_near - Vmax_mM_s * (dist_m ** 2) / (2.0 * max(D_m2_s, 1e-30))
    tissue_C = np.maximum(tissue_C, 0.0).astype(np.float32)
    # Preserve vessel values on vessel pixels for plotting; metrics exclude vessel pixels.
    tissue_C[vessel_mask] = vessel_C[vessel_mask]
    return tissue_C, dist_m.astype(np.float32)


def update_segment_tissue_C(
    segment_pixels: Dict[int, List[Tuple[int, int]]],
    vessel_mask: np.ndarray,
    tissue_C: np.ndarray,
    radius_px: int = 2,
) -> Dict[int, float]:
    nx, ny = vessel_mask.shape
    offsets = disk_offsets(max(1, radius_px))
    out: Dict[int, float] = {}
    global_tissue = float(np.median(tissue_C[~vessel_mask])) if np.any(~vessel_mask) else 0.0
    for sid, pts in segment_pixels.items():
        vals = []
        for ix, iy in pts:
            for ox, oy in offsets:
                jx, jy = ix + ox, iy + oy
                if 0 <= jx < nx and 0 <= jy < ny and not vessel_mask[jx, jy]:
                    vals.append(float(tissue_C[jx, jy]))
        out[sid] = float(np.mean(vals)) if vals else global_tissue
    return out


def summarize_metrics(
    net: Network,
    segs: Sequence[Segment],
    inlet_nodes: Sequence[int],
    outlet_nodes: Sequence[int],
    flux_into: Dict[int, float],
    node_C: Dict[int, float],
    seg_info: Dict[int, dict],
    vessel_mask: np.ndarray,
    centerline_mask: np.ndarray,
    tissue_C: np.ndarray,
    distance_m: np.ndarray,
    alpha: float,
    inlet_C_mM: float,
    D: float,
    Vmax: float,
    P_wall: float,
    iterations_done: int,
    convergence_rel: float,
) -> Dict[str, float | int | str | list]:
    tissue_mask = ~vessel_mask
    tissue_vals = tissue_C[tissue_mask]
    pO2 = tissue_vals / max(alpha, 1e-30)
    dist_um = distance_m[tissue_mask] * 1e6
    seg_Cout = np.array([info["Cout_mM"] for info in seg_info.values()], dtype=float)
    seg_Cmean = np.array([info["Cmean_mM"] for info in seg_info.values()], dtype=float)
    betas = np.array([info["beta"] for info in seg_info.values()], dtype=float)
    flows = np.array([info["Q_m3_s"] for info in seg_info.values()], dtype=float)

    total_in = sum(max(v, 0.0) for v in flux_into.values())
    total_out = -sum(min(v, 0.0) for v in flux_into.values())

    def pct_ge(thr):
        return float(np.mean(pO2 >= thr) * 100.0) if pO2.size else 0.0

    def pct_lt(thr):
        return float(np.mean(pO2 < thr) * 100.0) if pO2.size else 0.0

    return {
        "title": net.title,
        "n_segments_total": int(len(net.segments)),
        "n_segments_active_nonzero": int(len(segs)),
        "n_nodes_total": int(len(net.nodes)),
        "n_bcs": int(len(net.bcs)),
        "n_inlet_nodes": int(len(inlet_nodes)),
        "n_outlet_nodes": int(len(outlet_nodes)),
        "inlet_nodes": list(map(int, inlet_nodes)),
        "outlet_nodes": list(map(int, outlet_nodes)),
        "boundary_inflow_nl_min": float(total_in),
        "boundary_outflow_nl_min": float(total_out),
        "boundary_balance_nl_min": float(total_in - total_out),
        "grid_shape": [int(vessel_mask.shape[0]), int(vessel_mask.shape[1])],
        "vessel_pixels": int(vessel_mask.sum()),
        "centerline_pixels": int(centerline_mask.sum()),
        "inlet_C_mM": float(inlet_C_mM),
        "inlet_pO2_mmHg": float(inlet_C_mM / max(alpha, 1e-30)),
        "D_m2_s": float(D),
        "Vmax_mM_s": float(Vmax),
        "P_wall_m_s": float(P_wall),
        "iterations_done": int(iterations_done),
        "convergence_rel_node_C": float(convergence_rel),
        "vessel_Cout_min_mM": float(np.min(seg_Cout)) if seg_Cout.size else 0.0,
        "vessel_Cout_median_mM": float(np.median(seg_Cout)) if seg_Cout.size else 0.0,
        "vessel_Cout_max_mM": float(np.max(seg_Cout)) if seg_Cout.size else 0.0,
        "vessel_Cmean_median_mM": float(np.median(seg_Cmean)) if seg_Cmean.size else 0.0,
        "vessel_pO2_out_median_mmHg": float(np.median(seg_Cout) / max(alpha, 1e-30)) if seg_Cout.size else 0.0,
        "tissue_C_min_mM": float(np.min(tissue_vals)) if tissue_vals.size else 0.0,
        "tissue_C_median_mM": float(np.median(tissue_vals)) if tissue_vals.size else 0.0,
        "tissue_C_max_mM": float(np.max(tissue_vals)) if tissue_vals.size else 0.0,
        "tissue_pO2_p5_mmHg": float(np.percentile(pO2, 5)) if pO2.size else 0.0,
        "tissue_pO2_p10_mmHg": float(np.percentile(pO2, 10)) if pO2.size else 0.0,
        "tissue_pO2_median_mmHg": float(np.median(pO2)) if pO2.size else 0.0,
        "tissue_pO2_p90_mmHg": float(np.percentile(pO2, 90)) if pO2.size else 0.0,
        "tissue_pO2_p95_mmHg": float(np.percentile(pO2, 95)) if pO2.size else 0.0,
        "tissue_ge_15_mmHg_pct": pct_ge(15.0),
        "tissue_ge_20_mmHg_pct": pct_ge(20.0),
        "tissue_ge_30_mmHg_pct": pct_ge(30.0),
        "tissue_lt_1_mmHg_pct": pct_lt(1.0),
        "tissue_1_to_5_mmHg_pct": float(np.mean((pO2 >= 1.0) & (pO2 < 5.0)) * 100.0) if pO2.size else 0.0,
        "tissue_5_to_30_mmHg_pct": float(np.mean((pO2 >= 5.0) & (pO2 < 30.0)) * 100.0) if pO2.size else 0.0,
        "distance_median_um": float(np.median(dist_um)) if dist_um.size else 0.0,
        "distance_p95_um": float(np.percentile(dist_um, 95)) if dist_um.size else 0.0,
        "distance_max_um": float(np.max(dist_um)) if dist_um.size else 0.0,
        "tissue_within_50um_pct": float(np.mean(dist_um <= 50.0) * 100.0) if dist_um.size else 0.0,
        "tissue_within_100um_pct": float(np.mean(dist_um <= 100.0) * 100.0) if dist_um.size else 0.0,
        "tissue_within_200um_pct": float(np.mean(dist_um <= 200.0) * 100.0) if dist_um.size else 0.0,
        "segment_beta_median": float(np.median(betas)) if betas.size else 0.0,
        "segment_beta_p95": float(np.percentile(betas, 95)) if betas.size else 0.0,
        "segment_beta_max": float(np.max(betas)) if betas.size else 0.0,
        "flow_m3_s_median": float(np.median(flows)) if flows.size else 0.0,
        "flow_m3_s_min": float(np.min(flows)) if flows.size else 0.0,
        "flow_m3_s_max": float(np.max(flows)) if flows.size else 0.0,
    }


def save_array_png(arr: np.ndarray, path: Path, title: str, cbar_label: str, vessel_mask: Optional[np.ndarray] = None, cmap: str = "viridis") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(arr.T, origin="lower", cmap=cmap)
    if vessel_mask is not None and np.any(vessel_mask):
        ax.contour(vessel_mask.T.astype(float), levels=[0.5], colors="k", linewidths=0.4)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(cbar_label)
    ax.set_title(title)
    ax.set_xlabel("x grid")
    ax.set_ylabel("y grid")
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def save_outputs(
    outdir: Path,
    metrics: Dict,
    node_C: Dict[int, float],
    seg_info: Dict[int, dict],
    tissue_C: np.ndarray,
    vessel_C: np.ndarray,
    vessel_mask: np.ndarray,
    centerline_mask: np.ndarray,
    distance_m: np.ndarray,
    alpha: float,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(outdir / "metrics.csv", index=False)

    with (outdir / "node_oxygen.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["node", "C_mM", "pO2_mmHg"])
        w.writeheader()
        for n, c in sorted(node_C.items()):
            w.writerow({"node": n, "C_mM": c, "pO2_mmHg": c / max(alpha, 1e-30)})

    seg_rows = []
    for sid, info in sorted(seg_info.items()):
        row = {"segment": sid}
        row.update({k: v for k, v in info.items() if isinstance(v, (int, float, np.floating))})
        row["Cin_pO2_mmHg"] = row.get("Cin_mM", 0.0) / max(alpha, 1e-30)
        row["Cout_pO2_mmHg"] = row.get("Cout_mM", 0.0) / max(alpha, 1e-30)
        row["Cmean_pO2_mmHg"] = row.get("Cmean_mM", 0.0) / max(alpha, 1e-30)
        seg_rows.append(row)
    pd.DataFrame(seg_rows).to_csv(outdir / "segment_oxygen.csv", index=False)

    np.save(outdir / "tissue_C_mM.npy", tissue_C.astype(np.float32))
    np.save(outdir / "tissue_pO2_mmHg.npy", (tissue_C / max(alpha, 1e-30)).astype(np.float32))
    np.save(outdir / "vessel_C_mM.npy", vessel_C.astype(np.float32))
    np.save(outdir / "vessel_mask.npy", vessel_mask.astype(bool))
    np.save(outdir / "centerline_mask.npy", centerline_mask.astype(bool))
    np.save(outdir / "distance_to_vessel_um.npy", (distance_m * 1e6).astype(np.float32))

    Image.fromarray((vessel_mask.T.astype(np.uint8) * 255)).save(outdir / "vessel_mask.png")
    Image.fromarray((centerline_mask.T.astype(np.uint8) * 255)).save(outdir / "centerline_mask.png")

    save_array_png(tissue_C / max(alpha, 1e-30), outdir / "tissue_pO2_mmHg.png", "Tissue oxygen screen", "pO2 (mmHg)", vessel_mask, cmap="hot")
    save_array_png(vessel_C / max(alpha, 1e-30), outdir / "vessel_pO2_mmHg.png", "Rasterized vessel oxygen", "pO2 (mmHg)", vessel_mask, cmap="viridis")
    save_array_png(distance_m * 1e6, outdir / "distance_to_vessel_um.png", "Distance to nearest perfused vessel", "distance (µm)", vessel_mask, cmap="magma")

    lines = [
        "=== Graph-based NetFlow oxygen solver summary ===",
        f"segments solved: {metrics['n_segments_active_nonzero']}",
        f"inlet nodes: {metrics['n_inlet_nodes']} | outlet nodes: {metrics['n_outlet_nodes']}",
        f"median vessel outlet pO2: {metrics['vessel_pO2_out_median_mmHg']:.3f} mmHg",
        f"median tissue pO2: {metrics['tissue_pO2_median_mmHg']:.3f} mmHg",
        f"tissue >=15 mmHg: {metrics['tissue_ge_15_mmHg_pct']:.3f}%",
        f"tissue >=20 mmHg: {metrics['tissue_ge_20_mmHg_pct']:.3f}%",
        f"median distance: {metrics['distance_median_um']:.3f} µm",
        f"p95 distance: {metrics['distance_p95_um']:.3f} µm",
        f"convergence rel node C: {metrics['convergence_rel_node_C']:.3e}",
        "",
        "Interpretation: this is a graph-based steady dissolved-O2 prototype plus Krogh-distance tissue screen. It is stronger than the raster OxySim lumen traversal, but still requires validation before biological claims.",
    ]
    (outdir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_solver(args: argparse.Namespace) -> Path:
    net = parse_netflow(args.network)
    segs = active_segments(net, q_min_nl_min=args.q_min_nl_min)
    if not segs:
        raise RuntimeError("No active nonzero-flow segments found. Use a solved NetworkNew.dat, not an unsolved network.")

    detected_inlets, detected_outlets, flux_into = classify_boundary_flux(net, segs, flux_tol_nl_min=args.flux_tol_nl_min)
    user_inlets = parse_int_list(args.inlet_nodes)
    inlet_nodes = user_inlets if user_inlets else detected_inlets
    if not inlet_nodes:
        raise RuntimeError("No inlet nodes detected. Provide --inlet-nodes node1,node2 or check NetFlow boundary flows.")

    inlet_C_mM = args.inlet_C_mM if args.inlet_C_mM is not None else args.alpha * args.inlet_mmHg
    dx_m = net.alx_um * 1e-6 / max(args.grid_nx - 1, 1)
    dy_m = net.aly_um * 1e-6 / max(args.grid_ny - 1, 1)
    q_floor_m3_s = q_nl_min_to_m3_s(args.q_floor_nl_min)

    segment_tissue_C = {s.name: float(args.initial_tissue_C_mM) for s in segs}
    prev_node_vec = None
    conv_rel = float("inf")
    final = None

    for it in range(1, args.iterations + 1):
        node_C, seg_info = solve_graph_oxygen(
            net, segs, inlet_nodes, inlet_C_mM, args.P_wall, segment_tissue_C, q_floor_m3_s
        )
        vessel_mask, centerline_mask, vessel_C, segment_pixels = rasterize_vessel_oxygen(
            net, segs, seg_info, args.grid_nx, args.grid_ny, args.min_radius_um, args.vessel_dilate_px
        )
        tissue_C, distance_m = tissue_screen_from_vessels(vessel_mask, vessel_C, dx_m, dy_m, args.D, args.Vmax)
        if args.feedback:
            segment_tissue_C = update_segment_tissue_C(segment_pixels, vessel_mask, tissue_C, radius_px=args.feedback_radius_px)
        node_ids = sorted(node_C)
        node_vec = np.array([node_C[n] for n in node_ids], dtype=float)
        if prev_node_vec is not None:
            conv_rel = float(np.linalg.norm(node_vec - prev_node_vec) / max(np.linalg.norm(node_vec), 1e-30))
            if conv_rel < args.tol:
                final = (it, node_C, seg_info, vessel_mask, centerline_mask, vessel_C, tissue_C, distance_m)
                break
        prev_node_vec = node_vec
        final = (it, node_C, seg_info, vessel_mask, centerline_mask, vessel_C, tissue_C, distance_m)

    assert final is not None
    it, node_C, seg_info, vessel_mask, centerline_mask, vessel_C, tissue_C, distance_m = final
    metrics = summarize_metrics(
        net, segs, inlet_nodes, detected_outlets, flux_into, node_C, seg_info,
        vessel_mask, centerline_mask, tissue_C, distance_m, args.alpha, inlet_C_mM,
        args.D, args.Vmax, args.P_wall, it, conv_rel if np.isfinite(conv_rel) else -1.0
    )

    save_outputs(args.outdir, metrics, node_C, seg_info, tissue_C, vessel_C, vessel_mask, centerline_mask, distance_m, args.alpha)
    return args.outdir


# ----------------------------- self tests -----------------------------

def _toy_network_loop() -> Tuple[Network, List[Segment], List[int]]:
    # Construct a flow-balanced directed loop with one inlet node.
    # Node 1 fixed inlet. Segments:
    # 1->2 q=1.3, 2->3 q=1.0, 3->2 q=-? Instead use sign convention to create 3->2 q=0.3 by n1=3,n2=2,q=0.3, and 2->4 outlet q=0.3.
    # Node 2 incoming = 1.3 + 0.3, outgoing = 1.0 + 0.3? This is not exactly mass-conserved, but it forces a directed cycle in oxygen dependencies.
    nodes = {1:(0,0,0), 2:(100,0,0), 3:(100,100,0), 4:(200,0,0)}
    segs = [
        Segment(1,5,1,2,20,1.3,0),
        Segment(2,5,2,3,20,1.0,0),
        Segment(3,5,3,2,20,0.3,0),
        Segment(4,5,2,4,20,0.3,0),
    ]
    bcs = [BoundaryCondition(1,0,40,0,None), BoundaryCondition(4,0,35,0,None)]
    net = Network("toy_loop", 250, 150, 20, 10, 10, 1, 100, 150, 4, segs, nodes, bcs)
    return net, segs, [1]


def run_self_tests() -> None:
    net, segs, inlet = _toy_network_loop()
    C0 = 0.2
    # No exchange: all concentrations should equal inlet concentration in supplied component.
    node_C, seg_info = solve_graph_oxygen(net, segs, inlet, C0, 0.0, {s.name:0.0 for s in segs}, q_nl_min_to_m3_s(1e-12))
    err = max(abs(c - C0) for c in node_C.values())
    if err > 1e-8:
        raise AssertionError(f"self-test no-exchange failed: max error {err}")
    # With high exchange and zero tissue, vessel outlet should be lower than inlet.
    node_C, seg_info = solve_graph_oxygen(net, segs, inlet, C0, 1e-3, {s.name:0.0 for s in segs}, q_nl_min_to_m3_s(1e-12))
    if not all(0.0 <= info["Cout_mM"] <= C0 + 1e-9 for info in seg_info.values()):
        raise AssertionError("self-test high-exchange bounds failed")
    print("[SELF-TEST] graph oxygen loop tests passed")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Graph-based NetFlow oxygen solver prototype.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--network", type=Path, required=False, help="NetFlowV2 NetworkNew.dat or solved Network.dat")
    p.add_argument("--outdir", type=Path, default=Path("graph_oxygen_out"))
    p.add_argument("--grid-nx", type=int, default=384)
    p.add_argument("--grid-ny", type=int, default=384)
    p.add_argument("--alpha", type=float, default=0.00134, help="O2 solubility in mM/mmHg")
    p.add_argument("--inlet-mmHg", type=float, default=150.0)
    p.add_argument("--inlet-C-mM", type=float, default=None)
    p.add_argument("--D", type=float, default=1.5e-9, help="O2 diffusivity, m^2/s")
    p.add_argument("--Vmax", type=float, default=1e-4, help="zero-order screening uptake, mM/s")
    p.add_argument("--P-wall", dest="P_wall", type=float, default=3e-5, help="wall mass transfer/permeability, m/s")
    p.add_argument("--initial-tissue-C-mM", type=float, default=0.0, help="initial local tissue concentration used in segment oxygen equation")
    p.add_argument("--feedback", action="store_true", help="iterate segment tissue concentration from rasterized tissue screen")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--feedback-radius-px", type=int, default=2)
    p.add_argument("--inlet-nodes", type=str, default=None, help="optional comma-separated inlet node IDs overriding automatic q-sign classification")
    p.add_argument("--q-min-nl-min", type=float, default=1e-12, help="segments with |q| below this are excluded")
    p.add_argument("--q-floor-nl-min", type=float, default=1e-9, help="floor only inside attenuation formula to avoid division by zero")
    p.add_argument("--flux-tol-nl-min", type=float, default=1e-9, help="boundary flux tolerance for inlet/outlet classification")
    p.add_argument("--min-radius-um", type=float, default=1.0)
    p.add_argument("--vessel-dilate-px", type=int, default=1)
    p.add_argument("--self-test", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        run_self_tests()
        if args.network is None:
            return
    if args.network is None:
        raise SystemExit("Provide --network or run only --self-test.")
    out = run_solver(args)
    print(f"[DONE] graph oxygen outputs written to: {out}")


if __name__ == "__main__":
    main()
