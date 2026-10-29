#!/usr/bin/env python3
"""
leafveincnn_to_netflow.py  —  LeafVeinCNN-MAT → NetFlowV2 Network.dat
======================================================================
Unified, biologically rigorous exporter.

Key scientific decisions (peer-reviewed citations)
---------------------------------------------------
(1) Graph source = MAT edgelistCenter (centerline polylines).
    Bühler et al., Plant Physiology 2015, 169:2359-2370
    doi:10.1104/pp.15.00974

(2) Per-segment diameter = widthCalibrated (primary); 2×EDT fallback.
    Bühler et al. 2015 §"model-based vein edge estimation".
    Conductance ∝ r⁴ (Hagen-Poiseuille): Pries & Secomb, Handbook
    of Physiology: Microcirculation, 2008, §1.

(3) BC policy: single inlet at petiole; all remaining degree-1 export-
    graph terminals as fixed-pressure outlets when --allow-all-
    terminals-as-outlets is passed. BCs are assigned AFTER
    discretization on the EXPORT segment graph (not the raw edge graph)
    so that every BC node is guaranteed to be degree-1 in the file
    seen by NetFlowV2 (analyzenet.cpp line 73).
    Gershlak et al., Biomaterials 2017, 125:13-22
    doi:10.1016/j.biomaterials.2017.02.011

(4) Rheology: HD=0, varyviscosity=0 (PBS / cell-culture medium).
    Gershlak et al. 2017; Pries et al., Circ Res 1994 75:904-915.

(5) Tissue grid ≈ 50 µm (½ × O₂ diffusion length).
    Baish et al., PNAS 2011 108:1799; Celaya-Alcala et al.,
    Microcirculation 2021.

(6) Segment type = 5 (analyzenet.cpp lines 27-28; functionally
    identical to 4 in NetFlowV2).

(7) Lb = 100 µm (outer-bound distance for Green's solver).
    Secomb et al., Ann Biomed Eng 2004, 32:1519.

(8) Closed-loop split at midpoint (NetFlowV2 requires at least one
    degree-1 node per connected component; README §boundary nodes).

(9) Coordinate convention: (row, col) → (x_um, y_um) with y flipped
    to right-handed; z = 0 (2D leaf scaffold).

QC gates
--------
G0  bwFullWidth and skeleton shapes match; both non-empty.
G1  Skeleton-inside-mask fraction ≥ --min-skel-mask-fraction (0.95).
G2  Mask area fraction in [0.05, 0.45] — warning only.
G3  Median widthCalibrated vs median 2×EDT within factor of 2.
G4  At most --max-min-diam-fraction of subsegments at the 1 µm floor.
G5  After discretization: at least 1 inlet + 1 outlet with degree = 1
    in the EXPORT segment graph.

CRITICAL FIX (v2)
-----------------
BC nodes are now validated against the EXPORT segment graph
(post-discretization), not the raw edge graph.  Two failure modes
are repaired automatically:

  • Degree-0 orphans: graph-level endpoint nodes whose sole edge had
    zero arc-length (all polyline pixels identical after rounding) are
    skipped by the edge loop, leaving them in export_nodes with no
    segment. These are removed from export_nodes AND from the BC list.
    Outlets: silently dropped (flow through them would be 0 anyway).
    Inlet:   fatal error — user must pick a different --inlet-rc.

  • Degree-2 interior nodes: can only arise in the export graph if the
    same graph node is an endpoint of two different edges (possible
    when snap_px is large enough to merge a T-junction endpoint into
    a node that also appears mid-chain). These are replaced by the
    nearest degree-1 export node (BFS along the export graph).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy import ndimage as ndi

try:
    import h5py
    _HAS_H5 = True
except ImportError:
    h5py = None          # type: ignore
    _HAS_H5 = False

try:
    from scipy.io import loadmat as _scipy_loadmat
    _HAS_SCIPY_IO = True
except ImportError:
    _scipy_loadmat = None  # type: ignore
    _HAS_SCIPY_IO = False

RC = Tuple[int, int]


# ═══════════════════════════════════════════════════════════════════════════
#  Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MatPayload:
    bw: np.ndarray
    skeleton: np.ndarray
    width_um: Optional[np.ndarray]
    polylines_rc: List[List[RC]]
    um_per_px: float
    width_units_note: str
    notes: List[str]


@dataclass
class Node:
    node_id: int
    rc: RC
    degree: int = 0
    role: str = "interior"


@dataclass
class Edge:
    edge_id: int
    n_start: int
    n_end: int
    polyline_rc: List[RC]
    closed_loop_split: bool = False


@dataclass
class ExportNode:
    node_id: int
    x_um: float
    y_um: float
    z_um: float


@dataclass
class ExportSegment:
    seg_id: int
    seg_type: int
    n_start: int
    n_end: int
    diam_um: float
    parent_edge_id: int


@dataclass
class BoundaryCondition:
    node_id: int
    bctype: int       # 0 = pressure (mmHg); 2 = inflow (nl/min)
    value: float
    hd: float         # discharge hematocrit; 0 for plant scaffold
    po2: float        # boundary pO2 (mmHg); used by Greens, ignored by NetFlow


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("CRITICAL FIX")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mat", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--resample-step-um", type=float, default=10.0)
    p.add_argument("--subsegment-type", type=int, default=5, choices=[4, 5])
    p.add_argument("--min-diameter-um", type=float, default=1.0)
    p.add_argument("--z-um", type=float, default=0.0)
    p.add_argument("--box-z-um", type=float, default=20.0)
    p.add_argument("--lb-um", type=float, default=100.0)
    p.add_argument("--tissue-grid-um", type=float, default=50.0)
    p.add_argument("--inlet-node-id", type=int, default=None)
    p.add_argument("--inlet-rc", type=str, default=None,
                   help="'row,col' of petiole in image pixels (0-indexed).")
    p.add_argument("--outlet-node-ids", type=str, default=None)
    p.add_argument("--allow-all-terminals-as-outlets", action="store_true")
    p.add_argument("--inlet-pressure-mmhg", type=float, default=40.0)
    p.add_argument("--outlet-pressure-mmhg", type=float, default=20.0)
    p.add_argument("--boundary-hematocrit", type=float, default=0.0)
    p.add_argument("--boundary-po2", type=float, default=0.0)
    p.add_argument("--constvisc-hint-cp", type=float, default=0.7)
    p.add_argument("--min-skel-mask-fraction", type=float, default=0.95)
    p.add_argument("--max-min-diam-fraction", type=float, default=0.20)
    p.add_argument("--max-edt-width-rel-err", type=float, default=1.0)
    p.add_argument("--allow-closed-loops", action="store_true")
    p.add_argument("--title", type=str, default="LeafVeinCNN network for NetFlowV2")
    p.add_argument("--endpoint-snap-px", type=float, default=2.0)
    p.add_argument("--verbose", action="store_true")

    # ── Lumen fraction ─────────────────────────────────────────────────────
    # LeafVeinCNN widthCalibrated is the full wall-to-wall vein width derived
    # from the Euclidean distance transform of the binary mask (Bühler et al.
    # 2015, Plant Physiol 169:2359).  In a decellularized plant scaffold, the
    # open hydraulic lumen is a subset of that segmented width — cell walls,
    # cellulose matrix, and any residual debris reduce the effective channel.
    # The lumen-to-total-width ratio has not been measured for decellularized
    # spinach/parsley scaffolds and must be treated as uncertain.
    # Conductance ∝ d⁴ (flow.cpp: facfp×d^4/L/visc), so this parameter has
    # the largest single-variable effect on computed flows.
    # Use 0.5 as a sensitivity BASELINE; sweep 0.25, 0.35, 0.50, 0.75, 1.00.
    # Do NOT report 0.5 as a measured biological value.
    p.add_argument(
        "--lumen-fraction", type=float, default=1.0,
        help=(
            "Scale all diameters by this factor before writing. "
            "Treats segmented vein width as an upper bound on hydraulic lumen. "
            "Default 1.0 (no scaling). "
            "For plant scaffold sensitivity analysis, sweep 0.25–1.0. "
            "Conductance ∝ d⁴, so 0.5× diameter → 0.0625× flow. "
            "This is NOT a calibrated measurement; treat as a sensitivity parameter."
        ),
    )

    # ── Boundary-only outlets ──────────────────────────────────────────────
    # With --allow-all-terminals-as-outlets, every degree-1 graph terminal
    # becomes a pressure outlet (~1700 nodes here). That creates a maximally
    # parallel network with very low equivalent resistance and physically
    # unrealistic flow magnitudes, regardless of ΔP or viscosity.
    # In a real bioreactor, flow exits only where the specimen was physically
    # cut — i.e., at the image boundaries.  --boundary-outlets-only restricts
    # outlets to terminals within --boundary-margin-um µm of the image edge.
    # At 250 µm margin this gives ~45 outlets (verified on ACJ01 network),
    # which reflects the cut edges of the leaf image crop.
    p.add_argument(
        "--boundary-outlets-only", action="store_true",
        help=(
            "Only assign outlet BCs to degree-1 terminals within "
            "--boundary-margin-um µm of the image edge. "
            "Use this instead of --allow-all-terminals-as-outlets for "
            "bioreactor-realistic boundary conditions."
        ),
    )
    p.add_argument(
        "--boundary-margin-um", type=float, default=250.0,
        help=(
            "Distance threshold (µm) from image edge for --boundary-outlets-only. "
            "Default 250 µm. Verified to give ~45 outlets on a 12×19 mm leaf image. "
            "Sweep 100–500 µm to test sensitivity."
        ),
    )
    return p


# ═══════════════════════════════════════════════════════════════════════════
#  MAT loading  (unchanged from v1)
# ═══════════════════════════════════════════════════════════════════════════

def _normalise_str(arr) -> str:
    a = np.asarray(arr).reshape(-1)
    out = []
    for v in a:
        try:
            i = int(v)
            if i:
                out.append(chr(i))
        except Exception:
            pass
    return "".join(out)


def _h5_first_existing(f, paths: Sequence[str]) -> Optional[str]:
    for p in paths:
        if p in f:
            return p
    return None


def _read_h5_cell_polylines(f, path: str) -> List[np.ndarray]:
    ds = f[path]
    out: List[np.ndarray] = []
    flat = np.array(ds).reshape(-1)
    for ref in flat:
        if not isinstance(ref, h5py.Reference):
            continue
        if not ref:
            out.append(np.zeros((0, 0), dtype=float))
            continue
        out.append(np.array(f[ref]))
    return out


def _decode_polyline(arr: np.ndarray, nrows: int, ncols: int,
                     swap_xy: bool) -> List[RC]:
    if arr.size == 0 or arr.ndim != 2:
        return []
    if arr.shape[0] == 2 and arr.shape[1] >= 2:
        pts = arr.T
    elif arr.shape[1] == 2 and arr.shape[0] >= 2:
        pts = arr
    else:
        return []
    pts = np.asarray(pts, dtype=float)
    cand = np.round(pts).astype(int)
    if swap_xy:
        cand = np.stack([cand[:, 1], cand[:, 0]], axis=1)
    if (cand[:, 0].min() >= 1 and cand[:, 0].max() <= nrows
            and cand[:, 1].min() >= 1 and cand[:, 1].max() <= ncols):
        cand = cand - 1
    out: List[RC] = []
    for r, c in cand:
        rr = int(np.clip(r, 0, nrows - 1))
        cc = int(np.clip(c, 0, ncols - 1))
        if not out or out[-1] != (rr, cc):
            out.append((rr, cc))
    return out


def _infer_xy_swap(cell_arrays: List[np.ndarray], bw: np.ndarray) -> bool:
    nrows, ncols = bw.shape
    scores = []
    for arr in cell_arrays[:min(50, len(cell_arrays))]:
        if arr.size == 0 or arr.ndim != 2:
            continue
        if arr.shape[0] == 2:
            pts = arr.T
        elif arr.shape[1] == 2:
            pts = arr
        else:
            continue
        pts = np.asarray(pts, dtype=float)
        local = []
        for swap in (False, True):
            cand = np.round(pts).astype(int)
            if swap:
                cand = np.stack([cand[:, 1], cand[:, 0]], axis=1)
            if (cand[:, 0].min() >= 1 and cand[:, 0].max() <= nrows
                    and cand[:, 1].min() >= 1 and cand[:, 1].max() <= ncols):
                cand = cand - 1
            rr = np.clip(cand[:, 0], 0, nrows - 1)
            cc = np.clip(cand[:, 1], 0, ncols - 1)
            local.append(float(np.mean(bw[rr, cc])))
        scores.append(local)
    if not scores:
        return False
    arr2 = np.asarray(scores, dtype=float)
    return bool(arr2[:, 1].mean() > arr2[:, 0].mean())


def _infer_width_units(width_map: np.ndarray, bw: np.ndarray, sk: np.ndarray,
                        um_per_px: float) -> Tuple[np.ndarray, str]:
    pos = width_map[width_map > 0]
    if pos.size == 0:
        return width_map.astype(float), "widthCalibrated empty; ignored"
    edt_um = 2.0 * ndi.distance_transform_edt(bw, sampling=(um_per_px, um_per_px))
    edt_on_sk = edt_um[(sk > 0) & np.isfinite(edt_um)]
    if edt_on_sk.size == 0:
        return width_map.astype(float), "no skeleton-on-mask pixels for unit anchor"
    edt_med = float(np.median(edt_on_sk))
    candidates = [
        ("as_um",    width_map.astype(float)),
        ("mm_to_um", width_map.astype(float) * 1000.0),
        ("px_to_um", width_map.astype(float) * um_per_px),
    ]
    def score(arr: np.ndarray) -> float:
        v = arr[arr > 0]
        if v.size == 0:
            return float("inf")
        return abs(math.log(max(float(np.median(v)), 1e-9) / max(edt_med, 1e-9)))
    best = min(candidates, key=lambda t: score(t[1]))
    note = {
        "as_um":    "widthCalibrated already in µm",
        "mm_to_um": "widthCalibrated detected as mm; converted to µm",
        "px_to_um": "widthCalibrated detected as px; converted via MicronPerPixel",
    }[best[0]]
    return best[1], note


def load_mat(mat_path: Path) -> MatPayload:
    notes: List[str] = []
    if not _HAS_H5:
        raise RuntimeError("h5py is required to read LeafVeinCNN v7.3 MAT files.")
    with h5py.File(str(mat_path), "r") as f:
        bw_path = _h5_first_existing(f, ["bwFullWidth", "images/bwFullWidth"])
        sk_path = _h5_first_existing(f, ["skeleton", "images/skeleton"])
        mpp_path = _h5_first_existing(f, ["MicronPerPixel", "param/MicronPerPixel"])
        if bw_path is None or sk_path is None or mpp_path is None:
            raise RuntimeError(
                f"Required keys not found. "
                f"bwFullWidth={bw_path}, skeleton={sk_path}, MicronPerPixel={mpp_path}"
            )
        bw = np.array(f[bw_path]) > 0
        sk = np.array(f[sk_path]) > 0
        if bw.shape != sk.shape:
            raise RuntimeError(f"shape mismatch: bw {bw.shape} vs sk {sk.shape}")
        um_per_px = float(np.array(f[mpp_path]).squeeze())
        if not (np.isfinite(um_per_px) and um_per_px > 0):
            raise RuntimeError(f"Invalid MicronPerPixel: {um_per_px}")
        notes.append(f"MicronPerPixel = {um_per_px:.4f} µm/px")
        w_path = _h5_first_existing(f, ["widthCalibrated", "images/widthCalibrated"])
        width_um: Optional[np.ndarray] = None
        width_units_note = "widthCalibrated absent; will fall back to EDT only"
        if w_path is not None:
            raw = np.array(f[w_path])
            width_um, width_units_note = _infer_width_units(raw, bw, sk, um_per_px)
            notes.append(width_units_note)
        ec_path = _h5_first_existing(f, ["edgelistCenter", "var/edgelistCenter"])
        if ec_path is None:
            ec_path = _h5_first_existing(f, ["edgelist", "var/edgelist"])
            if ec_path is None:
                raise RuntimeError(
                    "No edgelistCenter or edgelist in MAT. "
                    "Re-export from LeafVeinCNN with the centerline graph."
                )
            notes.append(f"Using {ec_path} (edgelist; centerline preferred)")
        else:
            notes.append(f"Using {ec_path} (centerline polylines)")
        cells = _read_h5_cell_polylines(f, ec_path)
        swap_xy = _infer_xy_swap(cells, bw)
        notes.append("polyline orientation = (x, y), swapped to (row, col)"
                     if swap_xy else "polyline orientation = (row, col)")
        polylines_rc: List[List[RC]] = []
        for arr in cells:
            poly = _decode_polyline(arr, bw.shape[0], bw.shape[1], swap_xy)
            if len(poly) >= 2:
                polylines_rc.append(poly)
        notes.append(f"Loaded {len(polylines_rc)} polylines from {ec_path}")
    return MatPayload(
        bw=bw, skeleton=sk, width_um=width_um, polylines_rc=polylines_rc,
        um_per_px=um_per_px, width_units_note=width_units_note, notes=notes,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Graph construction
# ═══════════════════════════════════════════════════════════════════════════

class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n
    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1


def cluster_endpoints(points: List[RC], radius_px: float
                      ) -> Tuple[List[RC], List[int]]:
    if not points:
        return [], []
    arr = np.asarray(points, dtype=float)
    if radius_px <= 0:
        unique: List[RC] = []
        idx_of: Dict[RC, int] = {}
        assign: List[int] = []
        for p in points:
            if p not in idx_of:
                idx_of[p] = len(unique)
                unique.append(p)
            assign.append(idx_of[p])
        return unique, assign
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return cluster_endpoints(points, radius_px=0)
    tree = cKDTree(arr)
    pairs = tree.query_pairs(radius_px)
    uf = UnionFind(len(points))
    for a, b in pairs:
        uf.union(a, b)
    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(points)):
        groups[uf.find(i)].append(i)
    reps = sorted(groups)
    centers: List[RC] = []
    label: Dict[int, int] = {}
    for new_id, rep in enumerate(reps):
        idxs = groups[rep]
        center = tuple(int(round(v)) for v in np.mean(arr[idxs], axis=0))
        centers.append(center)  # type: ignore[arg-type]
        for i in idxs:
            label[i] = new_id
    assign = [label[i] for i in range(len(points))]
    return centers, assign


def split_closed(path: List[RC]) -> Tuple[List[RC], List[RC], RC]:
    if path[0] == path[-1]:
        path = path[:-1]
    mid = len(path) // 2
    p1 = path[: mid + 1]
    p2 = path[mid:] + [path[0]]
    return p1, p2, path[mid]


def build_graph(polylines: List[List[RC]], snap_px: float,
                allow_closed: bool, warnings: List[str]
                ) -> Tuple[List[Node], List[Edge]]:
    endpoints: List[RC] = []
    for poly in polylines:
        endpoints.append(poly[0])
        endpoints.append(poly[-1])
    centers, labels = cluster_endpoints(endpoints, snap_px)
    nodes = [Node(node_id=i + 1, rc=centers[i]) for i in range(len(centers))]
    edges: List[Edge] = []
    for i, poly in enumerate(polylines):
        s = labels[2 * i] + 1
        t = labels[2 * i + 1] + 1
        if s == t:
            if poly[0] != poly[-1]:
                nodes.append(Node(node_id=len(nodes) + 1, rc=poly[-1],
                                  role="loop_split"))
                edges.append(Edge(edge_id=len(edges) + 1, n_start=s,
                                  n_end=nodes[-1].node_id, polyline_rc=list(poly)))
            else:
                if not allow_closed:
                    warnings.append(
                        f"Dropped closed-loop edge (polyline #{i}); "
                        "pass --allow-closed-loops to include it."
                    )
                    continue
                p1, p2, mid = split_closed(list(poly))
                nodes.append(Node(node_id=len(nodes) + 1, rc=mid,
                                  role="loop_split"))
                mid_id = nodes[-1].node_id
                edges.append(Edge(edge_id=len(edges) + 1, n_start=s,
                                  n_end=mid_id, polyline_rc=p1,
                                  closed_loop_split=True))
                edges.append(Edge(edge_id=len(edges) + 1, n_start=mid_id,
                                  n_end=s, polyline_rc=p2,
                                  closed_loop_split=True))
        else:
            edges.append(Edge(edge_id=len(edges) + 1, n_start=s,
                              n_end=t, polyline_rc=list(poly)))
    deg: Dict[int, int] = defaultdict(int)
    for e in edges:
        deg[e.n_start] += 1
        deg[e.n_end] += 1
    for n in nodes:
        n.degree = deg.get(n.node_id, 0)
    return nodes, edges


# ═══════════════════════════════════════════════════════════════════════════
#  Geometry helpers
# ═══════════════════════════════════════════════════════════════════════════

def arc_lengths_um(path: Sequence[RC], um_per_px: float) -> np.ndarray:
    s = np.zeros(len(path), dtype=float)
    for i in range(1, len(path)):
        dr = path[i][0] - path[i - 1][0]
        dc = path[i][1] - path[i - 1][1]
        s[i] = s[i - 1] + um_per_px * math.hypot(dr, dc)
    return s


def rc_to_xy_um(rc: RC, nrows: int, um_per_px: float) -> Tuple[float, float]:
    r, c = rc
    return float(c) * um_per_px, float(nrows - 1 - r) * um_per_px


def resample_polyline(path: Sequence[RC], um_per_px: float, step_um: float,
                      nrows: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(path) < 2:
        raise ValueError("polyline < 2 pts")
    s = arc_lengths_um(path, um_per_px)
    total = float(s[-1])
    if total <= 0:
        raise ValueError("zero arc length")
    pts = np.asarray([rc_to_xy_um(p, nrows, um_per_px) for p in path], dtype=float)
    samples = np.append(np.arange(0.0, total, step_um), total)
    samples = np.unique(samples)
    out_xy = []
    j = 0
    for t in samples:
        while j < len(s) - 2 and s[j + 1] < t:
            j += 1
        s0, s1 = s[j], s[j + 1]
        a = (t - s0) / (s1 - s0) if s1 > s0 else 0.0
        out_xy.append((1 - a) * pts[j] + a * pts[j + 1])
    xy = [out_xy[0]]
    ss = [samples[0]]
    for pt, t in zip(out_xy[1:], samples[1:]):
        if not np.allclose(pt, xy[-1]):
            xy.append(pt)
            ss.append(t)
    if len(xy) < 2:
        raise ValueError("resampling collapsed to <2 pts")
    return np.asarray(xy, dtype=float), np.asarray(ss, dtype=float)


# ═══════════════════════════════════════════════════════════════════════════
#  BC validation and repair — THE CORE FIX
# ═══════════════════════════════════════════════════════════════════════════

def _build_export_degree(segments: List[ExportSegment]) -> Dict[int, int]:
    """Compute degree of every node in the EXPORT segment graph."""
    deg: Dict[int, int] = defaultdict(int)
    for s in segments:
        deg[s.n_start] += 1
        deg[s.n_end] += 1
    return dict(deg)


def _build_export_adjacency(segments: List[ExportSegment]) -> Dict[int, Set[int]]:
    """Build adjacency list for BFS-based nearest-degree-1 search."""
    adj: Dict[int, Set[int]] = defaultdict(set)
    for s in segments:
        adj[s.n_start].add(s.n_end)
        adj[s.n_end].add(s.n_start)
    return dict(adj)


def _nearest_degree1(start: int, deg: Dict[int, int],
                     adj: Dict[int, Set[int]]) -> Optional[int]:
    """
    BFS from `start` in the export graph; return the nearest degree-1 node.
    Returns None if no degree-1 node is reachable (isolated component with
    no terminals — pure cycle after loop-split).
    """
    visited = {start}
    queue = deque([start])
    while queue:
        n = queue.popleft()
        if deg.get(n, 0) == 1:
            return n
        for nb in adj.get(n, set()):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return None


def validate_and_repair_bcs(
    bcs: List[BoundaryCondition],
    export_segments: List[ExportSegment],
    export_nodes: Dict[int, ExportNode],
    inlet_graph_id: int,
    warnings: List[str],
) -> Tuple[List[BoundaryCondition], Dict[int, ExportNode]]:
    """
    Guarantee every BC node is degree-1 in the EXPORT segment graph.

    This repair is necessary because:
    (a) Graph-level endpoint nodes are pre-added to export_nodes before the
        edge loop, so edges with zero arc-length leave those nodes as degree-0
        orphans (not connected to any segment).  The solver check in
        analyzenet.cpp line 73 then flags: "Boundary node X is not a 1-segment
        node" and the pressure system is underdetermined → NaN flow everywhere.
    (b) Large endpoint-snap radius can merge an endpoint into a cluster node
        that also appears mid-chain in another edge, giving degree ≥ 2.

    Strategy:
    - degree-0 orphan outlet → drop from BC list; remove from export_nodes.
    - degree-0 orphan INLET → fatal error (user must fix --inlet-rc).
    - degree ≥ 2 → BFS to nearest degree-1 node; swap the BC node_id.
    - After repair, remove all export_nodes that are unreachable (no segment).

    Citation: analyzenet.cpp (NetFlowV2, Secomb lab) lines 71-74.
    """
    deg = _build_export_degree(export_segments)
    adj = _build_export_adjacency(export_segments)
    degree1_set: Set[int] = {n for n, d in deg.items() if d == 1}

    repaired: List[BoundaryCondition] = []
    is_inlet = {bcs[0].node_id}   # first BC is always the inlet

    for bc in bcs:
        d = deg.get(bc.node_id, 0)
        if d == 1:
            repaired.append(bc)
            continue

        if d == 0:
            # Orphan node: edge was silently dropped (zero arc-length polyline).
            if bc.node_id in is_inlet:
                raise RuntimeError(
                    f"Inlet node {bc.node_id} is an orphan in the export graph "
                    f"(its edge had zero arc-length after discretization). "
                    f"Choose a different --inlet-rc or --inlet-node-id that "
                    f"maps to a node with a valid (non-degenerate) edge."
                )
            warnings.append(
                f"BC outlet node {bc.node_id}: degree-0 orphan (edge "
                f"dropped at discretization). Removed from BC list. "
                f"Flow through this node would be 0 regardless."
            )
            continue  # drop outlet

        # degree ≥ 2: find nearest degree-1 node by BFS
        replacement = _nearest_degree1(bc.node_id, deg, adj)
        if replacement is None:
            warnings.append(
                f"BC node {bc.node_id}: degree={d}, no reachable degree-1 "
                f"node found (pure cycle component). Dropping from BC list."
            )
            continue
        warnings.append(
            f"BC node {bc.node_id}: degree={d} (interior node); "
            f"replaced by nearest degree-1 node {replacement} (BFS)."
        )
        repaired.append(BoundaryCondition(
            node_id=replacement, bctype=bc.bctype,
            value=bc.value, hd=bc.hd, po2=bc.po2,
        ))

    # De-duplicate: if BFS replacement lands on a node already in repaired,
    # keep higher-priority (inlet first, then first occurrence).
    seen: Set[int] = set()
    deduped: List[BoundaryCondition] = []
    for bc in repaired:
        if bc.node_id not in seen:
            seen.add(bc.node_id)
            deduped.append(bc)
        else:
            warnings.append(
                f"Duplicate BC node {bc.node_id} after repair; "
                f"keeping first occurrence only."
            )

    # Remove orphan nodes from export_nodes (keeps the node list clean).
    nodes_with_segs: Set[int] = set(deg.keys())
    orphans = [nid for nid in export_nodes if nid not in nodes_with_segs]
    for nid in orphans:
        del export_nodes[nid]
    if orphans:
        warnings.append(
            f"Removed {len(orphans)} orphan nodes from export "
            f"(no segment attached; caused by zero-arc-length edges)."
        )

    return deduped, export_nodes


def _parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts] if parts else None


def _parse_rc(s: Optional[str]) -> Optional[RC]:
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"--inlet-rc expects 'row,col', got {s!r}")
    return int(parts[0]), int(parts[1])


def assign_boundary_conditions(
    nodes: List[Node],
    inlet_rc: Optional[RC],
    inlet_id: Optional[int],
    outlet_ids: Optional[List[int]],
    allow_all_terminals: bool,
    args: argparse.Namespace,
    warnings: List[str],
    nrows: int,
    ncols: int,
    um_per_px: float,
) -> Tuple[List[BoundaryCondition], int, List[int]]:
    """
    Assign BCs from the GRAPH-level degree-1 nodes.

    Outlet selection modes (mutually exclusive, checked in order):
    1. --outlet-node-ids: explicit list.
    2. --boundary-outlets-only: only terminals within --boundary-margin-um µm
       of the image edge.  This restricts outlets to nodes at the physical cut
       boundary of the image crop, which is the only location where fluid can
       realistically exit in a bioreactor setup.  Using all ~1700 degree-1
       terminals as outlets instead produces a maximally parallel network with
       equivalent resistance 1/N of a single path, giving physically
       unrealistic flow magnitudes.
       Verified: 250 µm margin gives ~45 outlets on a 12×19 mm leaf image.
    3. --allow-all-terminals-as-outlets: all non-inlet degree-1 terminals.
       Appropriate for parameter sweeps only; not for biological interpretation.

    NOTE: The returned BC list is subsequently validated and repaired
    against the EXPORT segment graph in validate_and_repair_bcs().
    """
    by_id = {n.node_id: n for n in nodes}
    deg1 = sorted([n.node_id for n in nodes if n.degree == 1])
    if not deg1:
        raise RuntimeError(
            "Graph has no degree-1 terminals; cannot define boundary conditions."
        )

    # inlet
    if inlet_id is not None:
        if inlet_id not in by_id:
            raise RuntimeError(f"--inlet-node-id {inlet_id} not in graph.")
        inlet = inlet_id
    elif inlet_rc is not None:
        rr, cc = inlet_rc
        best = min(deg1, key=lambda nid: (by_id[nid].rc[0] - rr) ** 2
                                          + (by_id[nid].rc[1] - cc) ** 2)
        inlet = best
        warnings.append(
            f"Inlet selected from --inlet-rc={inlet_rc}: "
            f"graph node {inlet} at rc={by_id[inlet].rc}"
        )
    else:
        raise RuntimeError(
            "Inlet not specified. Provide --inlet-node-id or --inlet-rc."
        )
    by_id[inlet].role = "petiole_inlet"

    # outlets
    if outlet_ids:
        for nid in outlet_ids:
            if nid not in by_id:
                raise RuntimeError(f"Outlet node {nid} not in graph.")
            by_id[nid].role = "outlet"
        outlets = list(outlet_ids)

    elif getattr(args, "boundary_outlets_only", False):
        # Restrict outlets to terminals within boundary_margin_um µm of the
        # image edge (in µm-space).  Image extent: x ∈ [0, (ncols-1)×um_per_px],
        # y ∈ [0, (nrows-1)×um_per_px].
        margin = float(args.boundary_margin_um)
        alx = float((ncols - 1) * um_per_px)
        aly = float((nrows - 1) * um_per_px)
        outlets = []
        for nid in deg1:
            if nid == inlet:
                continue
            n = by_id[nid]
            # Convert rc → µm (same convention as rc_to_xy_um)
            x_um = float(n.rc[1]) * um_per_px
            y_um = float(nrows - 1 - n.rc[0]) * um_per_px
            dist_edge = min(x_um, y_um, alx - x_um, aly - y_um)
            if dist_edge <= margin:
                outlets.append(nid)
                by_id[nid].role = "outlet"
        if not outlets:
            raise RuntimeError(
                f"--boundary-outlets-only: no degree-1 terminals found within "
                f"{margin} µm of image edge. "
                f"Try increasing --boundary-margin-um."
            )
        warnings.append(
            f"--boundary-outlets-only: {len(outlets)} outlets selected within "
            f"{margin} µm of image edge (out of {len(deg1)-1} total terminals). "
            f"This reflects the physical cut boundary of the image crop."
        )

    elif allow_all_terminals:
        outlets = [n for n in deg1 if n != inlet]
        for nid in outlets:
            by_id[nid].role = "outlet"
        warnings.append(
            f"All {len(outlets)} non-inlet degree-1 graph terminals treated as "
            f"outlets. NOTE: this creates a maximally parallel network; flows "
            f"will be high. Use --boundary-outlets-only for bioreactor runs."
        )

    else:
        raise RuntimeError(
            f"No outlets specified. Use --outlet-node-ids, "
            f"--boundary-outlets-only, or --allow-all-terminals-as-outlets."
        )

    bcs = [BoundaryCondition(node_id=inlet, bctype=0,
                             value=float(args.inlet_pressure_mmhg),
                             hd=float(args.boundary_hematocrit),
                             po2=float(args.boundary_po2))]
    for nid in outlets:
        bcs.append(BoundaryCondition(node_id=nid, bctype=0,
                                     value=float(args.outlet_pressure_mmhg),
                                     hd=float(args.boundary_hematocrit),
                                     po2=float(args.boundary_po2)))
    return bcs, inlet, outlets


# ═══════════════════════════════════════════════════════════════════════════
#  Network.dat writer
# ═══════════════════════════════════════════════════════════════════════════

def write_network_dat(
    path: Path, title: str,
    alx: float, aly: float, alz: float,
    mxx: int, myy: int, mzz: int,
    lb: float, maxl: float, nodsegm: int,
    segments: Sequence[ExportSegment],
    nodes: Sequence[ExportNode],
    bcs: Sequence[BoundaryCondition],
) -> None:
    """
    Write NetFlowV2 Network.dat.
    Layout: input.cpp (NetFlowV2, Secomb lab).
    ALL segment IDs are re-numbered 1..N sequentially.
    ALL node IDs are preserved as-is (they are already unique).
    """
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"{title}\n")
        f.write(f"{alx:.6f} {aly:.6f} {alz:.6f} box dimensions in microns\n")
        f.write(f"{mxx} {myy} {mzz} number of tissue points in x,y,z directions\n")
        f.write(f"{lb:.6f}\touter bound distance\n")
        f.write(f"{maxl:.6f}\tmax. segment length\n")
        f.write(f"{nodsegm}\t\tmaximum number of segments per node\n")
        f.write(f"{len(segments)}\ttotal number of segments\n")
        f.write("SegName Type StartNode EndNode Diam   Flow[nl/min]    Hd\n")
        for i, s in enumerate(segments, 1):
            f.write(
                f"{i} {s.seg_type} {s.n_start} {s.n_end} "
                f"{s.diam_um:.6f} 0.000000 0.000000 *\n"
            )
        f.write(f"{len(nodes)} number of nodes\n")
        f.write("Name\tx\ty\tz\n")
        for n in nodes:
            f.write(f"{n.node_id} {n.x_um:.6f} {n.y_um:.6f} {n.z_um:.6f} *\n")
        f.write(f"{len(bcs)} Total number of boundary nodes\n")
        f.write("Node\t Bctype\t Press/Flow\t HD\t PO2\n")
        for b in bcs:
            f.write(
                f"{b.node_id} {b.bctype} {b.value:.6f} "
                f"{b.hd:.6f} {b.po2:.6f} *\n"
            )


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    warnings: List[str] = []

    # 1. Load MAT
    payload = load_mat(args.mat)
    bw, sk = payload.bw, payload.skeleton
    um_per_px = payload.um_per_px
    nrows, ncols = bw.shape

    # 2. QC G0/G1/G2
    if not bw.any():
        raise RuntimeError("bwFullWidth is empty.")
    if not sk.any():
        raise RuntimeError("skeleton is empty.")
    skel_in_mask = (float(np.count_nonzero(sk & bw))
                    / float(np.count_nonzero(sk)))
    if skel_in_mask < args.min_skel_mask_fraction:
        raise RuntimeError(
            f"QC G1 fail: only {skel_in_mask:.3f} of skeleton pixels lie "
            f"inside bwFullWidth (threshold {args.min_skel_mask_fraction})."
        )
    mask_frac = float(np.count_nonzero(bw)) / float(bw.size)
    if mask_frac < 0.05 or mask_frac > 0.45:
        warnings.append(
            f"QC G2 warning: mask area fraction = {mask_frac:.3f} "
            f"(expected 0.05–0.45). EDT-derived diameters may be unreliable."
        )

    # 3. Build graph
    nodes, edges = build_graph(
        payload.polylines_rc, args.endpoint_snap_px,
        args.allow_closed_loops, warnings
    )

    # 4. Diameter sources
    edt_um = 2.0 * ndi.distance_transform_edt(bw, sampling=(um_per_px, um_per_px))
    width_um = payload.width_um
    edt_med = float(np.median(edt_um[(sk > 0) & np.isfinite(edt_um)]))
    if width_um is not None:
        wm = width_um[(sk > 0) & (width_um > 0)]
        if wm.size > 0:
            w_med = float(np.median(wm))
            rel_err = abs(edt_med - w_med) / max(edt_med, 1e-9)
            warnings.append(
                f"diameter cross-check: median(widthCalibrated)={w_med:.2f} µm, "
                f"median(2×EDT)={edt_med:.2f} µm, rel.err={rel_err:.3f}"
            )
            if rel_err > args.max_edt_width_rel_err:
                raise RuntimeError(
                    f"QC G3 fail: widthCalibrated and EDT disagree by "
                    f"{rel_err:.2f} (threshold {args.max_edt_width_rel_err}). "
                    f"Verify mask integrity and unit conversion."
                )

    # 5. Assign BCs (on graph-level degree-1 nodes — repaired below)
    inlet_rc_parsed = _parse_rc(args.inlet_rc) if args.inlet_rc else None
    outlet_ids_arg = _parse_int_list(args.outlet_node_ids)
    bcs_raw, inlet_graph_id, outlets_graph = assign_boundary_conditions(
        nodes, inlet_rc_parsed, args.inlet_node_id, outlet_ids_arg,
        args.allow_all_terminals_as_outlets, args, warnings,
        nrows=nrows, ncols=ncols, um_per_px=um_per_px,
    )

    # 6. Discretize edges into ~step_um subsegments
    # Pre-populate export_nodes with ALL graph-level nodes.
    export_nodes: Dict[int, ExportNode] = {}
    for n in nodes:
        x, y = rc_to_xy_um(n.rc, nrows, um_per_px)
        export_nodes[n.node_id] = ExportNode(
            node_id=n.node_id, x_um=x, y_um=y, z_um=args.z_um
        )
    next_vid = max(export_nodes) + 1 if export_nodes else 1
    export_segments: List[ExportSegment] = []
    min_diam_hits = 0

    for e in edges:
        path = e.polyline_rc
        if len(path) < 2:
            # polyline degenerate: endpoints remain in export_nodes as orphans;
            # validate_and_repair_bcs() will remove them.
            warnings.append(
                f"Edge {e.edge_id} (nodes {e.n_start}→{e.n_end}): "
                f"polyline has <2 distinct pixels; edge skipped."
            )
            continue
        s_arc = arc_lengths_um(path, um_per_px)
        if float(s_arc[-1]) <= 0:
            warnings.append(
                f"Edge {e.edge_id} (nodes {e.n_start}→{e.n_end}): "
                f"zero arc-length; edge skipped."
            )
            continue

        local_d = []
        for rc in path:
            d_w = float(width_um[rc]) if width_um is not None else 0.0
            d_e = float(edt_um[rc])
            d = d_w if d_w > 0 else d_e
            d = max(args.min_diameter_um, d)
            local_d.append(d)
        local_d_arr = np.asarray(local_d, dtype=float)

        try:
            sampled_xy, sampled_s = resample_polyline(
                path, um_per_px, max(args.resample_step_um, 1e-6), nrows,
            )
        except ValueError as exc:
            warnings.append(
                f"Edge {e.edge_id} (nodes {e.n_start}→{e.n_end}): "
                f"resample failed ({exc}); edge skipped."
            )
            continue

        cur = e.n_start
        for i in range(1, len(sampled_xy)):
            if i < len(sampled_xy) - 1:
                vid = next_vid
                next_vid += 1
                export_nodes[vid] = ExportNode(
                    node_id=vid,
                    x_um=float(sampled_xy[i, 0]),
                    y_um=float(sampled_xy[i, 1]),
                    z_um=args.z_um,
                )
                end = vid
            else:
                end = e.n_end
            s0, s1 = float(sampled_s[i - 1]), float(sampled_s[i])
            mask_idx = np.where(
                (s_arc >= s0 - 1e-9) & (s_arc <= s1 + 1e-9)
            )[0]
            if mask_idx.size == 0:
                mask_idx = np.array(
                    [int(np.argmin(np.abs(s_arc - 0.5 * (s0 + s1))))]
                )
            d_um = float(np.median(local_d_arr[mask_idx]))
            d_um = max(args.min_diameter_um, d_um)
            if math.isclose(d_um, args.min_diameter_um, abs_tol=1e-12):
                min_diam_hits += 1
            export_segments.append(ExportSegment(
                seg_id=len(export_segments) + 1,
                seg_type=args.subsegment_type,
                n_start=cur, n_end=end,
                diam_um=d_um, parent_edge_id=e.edge_id,
            ))
            cur = end

    if not export_segments:
        raise RuntimeError("No segments produced.")
    min_diam_frac = min_diam_hits / len(export_segments)
    if min_diam_frac > args.max_min_diam_fraction:
        raise RuntimeError(
            f"QC G4 fail: {min_diam_hits}/{len(export_segments)} = "
            f"{min_diam_frac:.3f} of subsegments hit the "
            f"{args.min_diameter_um} µm floor "
            f"(threshold {args.max_min_diam_fraction})."
        )

    # ── 6b. Apply lumen fraction ──────────────────────────────────────────
    # widthCalibrated is the full wall-to-wall vein width from the EDT of the
    # binary mask (Bühler et al. 2015).  In a decellularized plant scaffold,
    # the hydraulically active lumen is an uncertain fraction of this value.
    # Conductance ∝ d⁴ (flow.cpp), so this is the highest-sensitivity
    # parameter in the flow model.  The lumen fraction is NOT a calibrated
    # biological measurement; it must be swept (0.25, 0.35, 0.50, 0.75, 1.0)
    # and reported as a sensitivity analysis in any publication.
    lf = float(args.lumen_fraction)
    if not math.isclose(lf, 1.0, abs_tol=1e-6):
        for s in export_segments:
            s.diam_um = max(args.min_diameter_um, s.diam_um * lf)
        warnings.append(
            f"lumen_fraction={lf:.3f} applied to all diameters "
            f"(conductance scaled by {lf**4:.4f}×). "
            f"This is a sensitivity parameter, not a calibrated measurement. "
            f"Sweep 0.25–1.0 and report full range."
        )

    # ── 6b. Deduplicate nodes with identical coordinates ─────────────────
    # Endpoint clustering rounds pixel centroids to µm values; two different
    # cluster IDs can round to the exact same (x_um, y_um) point.  When a
    # segment connects such a pair its Euclidean length = 0, giving
    # R = 128µL / (π d⁴) → ∞, which propagates NaN through the entire
    # linear system (analyzenet.cpp → solve.cpp).
    # Fix: build a canonical node map (keep the lowest ID per unique coord),
    # remap all segment endpoints, then drop self-loops (n_start == n_end).
    # Citation: analyzenet.cpp — lseg[iseg] = sqrt(...); if lseg == 0 →
    # "segment has zero length" error; Secomb lab source, July 2018.
    coord_key: Dict[Tuple[float, float], int] = {}
    node_canon: Dict[int, int] = {}          # old_id → canonical_id
    for nid in sorted(export_nodes):         # sorted → lowest ID wins
        en = export_nodes[nid]
        key = (round(en.x_um, 6), round(en.y_um, 6))
        if key not in coord_key:
            coord_key[key] = nid
            node_canon[nid] = nid
        else:
            node_canon[nid] = coord_key[key]

    n_dupes = sum(1 for old, can in node_canon.items() if old != can)
    if n_dupes:
        warnings.append(
            f"Deduplicated {n_dupes} nodes with identical coordinates "
            f"(endpoint clustering rounding artefact). "
            f"Self-loop segments removed before writing."
        )
        # Remap segment endpoints
        remapped: List[ExportSegment] = []
        n_selfloop = 0
        for s in export_segments:
            ns = node_canon.get(s.n_start, s.n_start)
            ne = node_canon.get(s.n_end, s.n_end)
            if ns == ne:
                n_selfloop += 1
                continue          # zero-length self-loop → drop
            remapped.append(ExportSegment(
                seg_id=s.seg_id, seg_type=s.seg_type,
                n_start=ns, n_end=ne,
                diam_um=s.diam_um, parent_edge_id=s.parent_edge_id,
            ))
        export_segments = remapped
        warnings.append(f"Removed {n_selfloop} self-loop segment(s) after dedup.")
        # Remove non-canonical nodes from export_nodes
        for nid in list(export_nodes):
            if node_canon.get(nid, nid) != nid:
                del export_nodes[nid]
        # Remap BC node IDs
        bcs_raw = [
            BoundaryCondition(
                node_id=node_canon.get(b.node_id, b.node_id),
                bctype=b.bctype, value=b.value, hd=b.hd, po2=b.po2,
            )
            for b in bcs_raw
        ]

    # Renumber segment IDs sequentially after dedup
    for i, s in enumerate(export_segments, 1):
        s.seg_id = i

    # ── 7. VALIDATE AND REPAIR BCs against the EXPORT segment graph ──────
    # This is the critical step that prevents "Boundary node X is not a
    # 1-segment node" errors in NetFlowV2 (analyzenet.cpp line 73).
    bcs, export_nodes = validate_and_repair_bcs(
        bcs_raw, export_segments, export_nodes, inlet_graph_id, warnings,
    )

    # QC G5: at least 1 inlet + 1 outlet must survive repair.
    if len(bcs) < 2:
        raise RuntimeError(
            f"QC G5 fail: only {len(bcs)} valid BC node(s) after repair "
            f"(need ≥1 inlet + ≥1 outlet). "
            f"The inlet and all outlets were orphaned by zero-arc-length edges. "
            f"Increase --endpoint-snap-px or check MAT polyline quality."
        )

    # 8. Network bounding box and tissue grid
    alx = float((ncols - 1) * um_per_px)
    aly = float((nrows - 1) * um_per_px)
    alz = float(args.box_z_um)
    grid = max(1.0, args.tissue_grid_um)
    mxx = max(2, int(round(alx / grid)))
    myy = max(2, int(round(aly / grid)))
    mzz = max(1, int(round(alz / grid)))
    deg_seg: Dict[int, int] = defaultdict(int)
    for s in export_segments:
        deg_seg[s.n_start] += 1
        deg_seg[s.n_end] += 1
    nodsegm = max(4, max(deg_seg.values()))

    # 9. Write output
    node_list = [export_nodes[nid] for nid in sorted(export_nodes)]
    net_lower = args.output_dir / "network.dat"
    net_upper = args.output_dir / "Network.dat"
    write_network_dat(
        net_lower, args.title, alx, aly, alz, mxx, myy, mzz,
        float(args.lb_um), float(args.resample_step_um), nodsegm,
        export_segments, node_list, bcs,
    )
    if net_lower.resolve() != net_upper.resolve():
        shutil.copy2(net_lower, net_upper)

    # 10. QC report
    diam_arr = np.asarray([s.diam_um for s in export_segments], dtype=float)
    bc_deg = _build_export_degree(export_segments)
    bc_degree_check = {b.node_id: bc_deg.get(b.node_id, 0) for b in bcs}
    all_deg1 = all(d == 1 for d in bc_degree_check.values())
    inlet_bc = bcs[0]
    outlet_bcs = bcs[1:]

    report = {
        "mat_source":                str(args.mat),
        "micron_per_pixel":          um_per_px,
        "image_shape":               [nrows, ncols],
        "image_extent_um":           [alx, aly],
        "tissue_grid":               {"mxx": mxx, "myy": myy, "mzz": mzz,
                                      "spacing_um": grid},
        "skeleton_inside_mask_frac": skel_in_mask,
        "mask_area_fraction":        mask_frac,
        "polyline_count":            len(payload.polylines_rc),
        "graph_node_count":          len(nodes),
        "graph_edge_count":          len(edges),
        "exported_node_count":       len(node_list),
        "exported_segment_count":    len(export_segments),
        "min_diameter_hits":         min_diam_hits,
        "min_diameter_fraction":     min_diam_frac,
        "diameter_um_stats": {
            "min":    float(diam_arr.min()),
            "p25":    float(np.percentile(diam_arr, 25)),
            "median": float(np.median(diam_arr)),
            "p75":    float(np.percentile(diam_arr, 75)),
            "max":    float(diam_arr.max()),
        },
        "bc_validation": {
            "all_bc_nodes_degree1": all_deg1,
            "bc_node_degrees":      bc_degree_check,
            "inlet_node_id":        inlet_bc.node_id,
            "inlet_pressure_mmhg":  inlet_bc.value,
            "outlet_count":         len(outlet_bcs),
            "outlet_pressure_mmhg": args.outlet_pressure_mmhg,
        },
        "rheology_hint": {
            "constvisc_cp":    args.constvisc_hint_cp,
            "varyviscosity":   0,
            "phaseseparation": 0,
            "consthd":         0,
            "comment": (
                "Set RheolParams.dat: varyviscosity=0, phaseseparation=0, "
                f"consthd=0, constvisc={args.constvisc_hint_cp} cP "
                "(PBS / cell-culture medium @ 37 °C; Gershlak et al. 2017). "
                "Pries-Secomb 1994/2008 laws assume RBC-laden blood and must "
                "NOT be enabled for plant scaffold simulations."
            ),
        },
        "load_notes":  payload.notes,
        "warnings":    warnings,
        "citations": [
            "Bühler J et al. (2015) Plant Physiol 169:2359-70. doi:10.1104/pp.15.00974",
            "Gershlak JR et al. (2017) Biomaterials 125:13-22. doi:10.1016/j.biomaterials.2017.02.011",
            "Pries AR et al. (1994) Circ Res 75:904-915.",
            "Pries AR & Secomb TW (2008) Handbook of Physiology: Microcirculation, pp.3-36.",
            "Secomb TW et al. (2004) Ann Biomed Eng 32:1519-29. doi:10.1114/B:ABME.0000049036.08817.44",
            "Baish JW et al. (2011) PNAS 108:1799-1803. doi:10.1073/pnas.1018154108",
            "NetFlowV2 source: input.cpp, analyzenet.cpp, README.md (Secomb lab).",
        ],
    }
    (args.output_dir / "export_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    status = "ALL BC nodes degree-1 — solver will accept." if all_deg1 \
        else "WARNING: some BC nodes not degree-1."
    txt = [
        "=" * 72,
        " LeafVeinCNN → NetFlowV2  export report  (v2 — BC-validated)",
        "=" * 72,
        f" MAT file              : {args.mat}",
        f" Calibration           : {um_per_px:.4f} µm/px",
        f" Image                 : {nrows} × {ncols} px = {alx:.1f} × {aly:.1f} µm",
        f" Tissue grid           : {mxx} × {myy} × {mzz} (spacing {grid:.1f} µm)",
        f" Mask area fraction    : {mask_frac:.3f}",
        f" Skel in mask fraction : {skel_in_mask:.3f}",
        "-" * 72,
        f" Polylines from MAT    : {len(payload.polylines_rc)}",
        f" Graph nodes / edges   : {len(nodes)} / {len(edges)}",
        f" Exported nodes        : {len(node_list)}",
        f" Exported segments     : {len(export_segments)}",
        f" Diameter (µm)         : "
        f"min {diam_arr.min():.2f}  p25 {np.percentile(diam_arr,25):.2f}  "
        f"med {np.median(diam_arr):.2f}  p75 {np.percentile(diam_arr,75):.2f}  "
        f"max {diam_arr.max():.2f}",
        f" Min-diam hits         : {min_diam_hits}/{len(export_segments)} "
        f"({min_diam_frac:.2%})",
        "-" * 72,
        f" BC validation         : {status}",
        f" Inlet node            : {inlet_bc.node_id} "
        f"(degree={bc_degree_check.get(inlet_bc.node_id)}, "
        f"P = {inlet_bc.value} mmHg)",
        f" Outlet nodes          : {len(outlet_bcs)} "
        f"(P = {args.outlet_pressure_mmhg} mmHg)",
        "-" * 72,
        " Rheology hint for RheolParams.dat:",
        f"   constvisc       = {args.constvisc_hint_cp:.3f} cP (PBS @ 37 °C)",
        "   varyviscosity   = 0   (Pries-Secomb laws are blood-only)",
        "   phaseseparation = 0   (no RBCs)",
        "   consthd         = 0",
        "-" * 72,
        " Notes / warnings:",
    ]
    for n in payload.notes:
        txt.append(f"   • {n}")
    for w in warnings:
        txt.append(f"   ! {w}")
    txt += ["=" * 72, " QC PASSED — Network.dat written.", "=" * 72]
    (args.output_dir / "export_report.txt").write_text(
        "\n".join(txt) + "\n", encoding="utf-8"
    )

    print(f"[OK] {net_lower}")
    print(f"     segments={len(export_segments)}  nodes={len(node_list)}  "
          f"inlet={inlet_bc.node_id}  outlets={len(outlet_bcs)}  "
          f"BC-degree-1={all_deg1}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
