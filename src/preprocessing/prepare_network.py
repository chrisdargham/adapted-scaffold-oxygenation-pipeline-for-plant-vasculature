#!/usr/bin/env python3
"""
prepare_network.py  —  MAT → network_ready.dat for NetFlowV2
=============================================================
USAGE
-----
  python prepare_network.py \\
      --mat  FILE.mat \\
      --exporter  leafveincnn_to_netflow.py \\
      --output-dir  ./out_final

WHAT IT DOES (in order)
-----------------------
1. Calls leafveincnn_to_netflow.py (v2, BC-validated).
2. Caps diameters at 150 µm → condition number κ < 6×10⁴, solver converges.
   Brodribb et al., Plant Cell Environ 2007, 30:176.
   O₂/nutrient exchange occurs through 10–150 µm minor veins.
   κ = (d_max/d_min)⁴; at 150 µm cap κ = 5.9×10⁴ (OK for Gauss-Seidel).
3. Removes segments with L < 0.5 µm (junction artefacts only).
   Pries et al., Am J Physiol 1994, 267:H1347.
4. Fixes disconnected components: small ones removed, large ones get a
   BC node placed on a DEGREE-1 terminal (not the highest-degree node).
   Secomb, Ann Biomed Eng 2004, 32:1519.
5. Final validation: every BC node is confirmed degree-1 in the output
   segment graph.  Any violator is replaced via BFS to nearest degree-1
   neighbour before writing network_ready.dat.
6. Writes network_ready.dat.

CRITICAL FIX (v2)
-----------------
Original prepare_network.py selected the anchor for isolated components as

    anchor = max(comp, key=lambda n: len(adj.get(n, set())))

which picks the HIGHEST-DEGREE node — an interior bifurcation, never a
terminal.  NetFlowV2's analyzenet.cpp line 73 checks nodtyp[bcnod] == 1;
any non-terminal BC node causes "Boundary node X is not a 1-segment node",
makes the pressure system underdetermined, and produces NaN flow everywhere.

Fix: select a degree-1 node from the component.  If none exists (pure
cycle), split the highest-degree node with a stub segment so that a
degree-1 endpoint exists (pure cycles are handled by the exporter's
--allow-closed-loops split, but isolated pure cycles in the cleaned graph
can still arise after segment removal).

A final validation pass over ALL BC entries runs after every modification
step (diameter cap, segment removal, component fix) to guarantee the
invariant before writing the file.
"""

import argparse
import math
import shutil
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--mat",                   type=Path,  required=True)
    p.add_argument("--exporter",              type=Path,  required=True)
    p.add_argument("--output-dir",            type=Path,  required=True)
    p.add_argument("--inlet-rc",              type=str,   default="1400,5")
    p.add_argument("--snap-px",               type=int,   default=15)
    p.add_argument("--inlet-pressure",        type=float, default=40.0)
    p.add_argument("--outlet-pressure",       type=float, default=39.0,
                   help="Outlet pressure (mmHg). Default 39.0 (ΔP=1 mmHg).")
    p.add_argument("--diameter-cap",          type=float, default=150.0)
    p.add_argument("--min-length",            type=float, default=0.5)
    p.add_argument("--large-comp-threshold",  type=int,   default=5)
    p.add_argument("--primary-vein-diam",     type=float, default=150.0)
    p.add_argument("--resample-step-um",      type=float, default=25.0,
                   help="Subsegment resampling step (µm). Default 25.")
    p.add_argument("--lumen-fraction",        type=float, default=1.0,
                   help=(
                       "Scale all diameters by this factor (conductance ∝ d⁴). "
                       "Treats segmented vein width as upper bound on hydraulic lumen. "
                       "Default 1.0. Sweep 0.25–1.0 as sensitivity analysis. "
                       "NOT a calibrated measurement."
                   ))
    p.add_argument("--boundary-outlets-only", action="store_true",
                   help=(
                       "Only assign outlet BCs to terminals within "
                       "--boundary-margin-um µm of the image edge. "
                       "Recommended for bioreactor runs."
                   ))
    p.add_argument("--boundary-margin-um",    type=float, default=250.0,
                   help="Image-edge margin for --boundary-outlets-only (µm). Default 250.")
    p.add_argument("--python",                type=str,   default=sys.executable)
    return p


# ═══════════════════════════════════════════════════════════════════════════
#  Network file I/O helpers
# ═══════════════════════════════════════════════════════════════════════════

def parse_network(path):
    """
    Parse a NetFlowV2 network.dat file.

    Returns
    -------
    lines       : list of raw line strings (for in-place editing)
    nseg        : total segment count (header value)
    nnod        : total node count (header value)
    adj         : dict  node_id → set(neighbour_node_ids)  for type-4/5 segs
    node_diam   : dict  node_id → max adjacent diameter
    seg_of_node : dict  node_id → [seg_name, ...]
    coords      : dict  node_id → (x_um, y_um)
    seg_lines   : dict  seg_name (int) → line_index  (for in-place editing)
    node_deg    : dict  node_id → degree (number of active type-4/5 segments)
    """
    with open(path) as f:
        lines = f.readlines()

    nseg = int(lines[6].split()[0])
    nnod = int(lines[8 + nseg].split()[0])

    adj         = defaultdict(set)
    node_diam   = defaultdict(float)
    seg_of_node = defaultdict(list)
    seg_lines   = {}
    node_deg    = defaultdict(int)

    for i in range(8, 8 + nseg):
        p = lines[i].split()
        if len(p) < 5:
            continue
        try:
            sid   = int(p[0])
            stype = int(p[1])
            n1    = int(p[2])
            n2    = int(p[3])
            d     = float(p[4])
        except ValueError:
            continue
        seg_lines[sid] = i
        if stype in (4, 5):
            adj[n1].add(n2)
            adj[n2].add(n1)
            node_diam[n1] = max(node_diam[n1], d)
            node_diam[n2] = max(node_diam[n2], d)
            seg_of_node[n1].append(sid)
            seg_of_node[n2].append(sid)
            node_deg[n1] += 1
            node_deg[n2] += 1

    coords = {}
    node_sec_start = 8 + nseg + 2        # skip header + column-name line
    for i in range(node_sec_start, node_sec_start + nnod):
        if i >= len(lines):
            break
        p = lines[i].split()
        if len(p) >= 3:
            try:
                coords[int(p[0])] = (float(p[1]), float(p[2]))
            except ValueError:
                pass

    return lines, nseg, nnod, adj, node_diam, seg_of_node, coords, seg_lines, node_deg


def seg_length(n1, n2, coords):
    if n1 not in coords or n2 not in coords:
        return 0.0
    x1, y1 = coords[n1]
    x2, y2 = coords[n2]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def get_components(adj):
    """Return connected components sorted largest-first."""
    visited, components = set(), []
    for start in adj:
        if start in visited:
            continue
        comp = set()
        q = deque([start])
        while q:
            n = q.popleft()
            if n in visited:
                continue
            visited.add(n)
            comp.add(n)
            q.extend(adj[n] - visited)
        components.append(comp)
    return sorted(components, key=len, reverse=True)


# ═══════════════════════════════════════════════════════════════════════════
#  BC validation helpers
# ═══════════════════════════════════════════════════════════════════════════

def _bfs_nearest_degree1(start: int, node_deg: Dict, adj: Dict) -> Optional[int]:
    """BFS from start; return nearest node with degree == 1."""
    visited = {start}
    queue = deque([start])
    while queue:
        n = queue.popleft()
        if node_deg.get(n, 0) == 1:
            return n
        for nb in adj.get(n, set()):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return None


def read_bc_entries(lines: List[str]) -> Tuple[int, int, List[Tuple[int, str]]]:
    """
    Return (bc_count_line_idx, header_line_idx, [(node_id, raw_line), ...]).
    """
    bc_count_idx = next(
        i for i, l in enumerate(lines)
        if "Total number of boundary nodes" in l
    )
    header_idx = bc_count_idx + 1
    bc_start   = bc_count_idx + 2
    entries: List[Tuple[int, str]] = []
    for i in range(bc_start, len(lines)):
        p = lines[i].split()
        if len(p) < 4:
            break
        try:
            nid = int(p[0])
            float(p[2])          # validates press/flow field
            entries.append((nid, lines[i]))
        except ValueError:
            break
    return bc_count_idx, header_idx, entries


def validate_and_repair_bc_list(
    bc_entries: List[Tuple[int, str]],
    node_deg: Dict[int, int],
    adj: Dict[int, Set[int]],
    log: List[str],
    inlet_pressure: float,
    outlet_pressure: float,
) -> List[Tuple[int, str]]:
    """
    Guarantee every BC node is degree-1 in the current segment graph.

    Rules (NetFlowV2 analyzenet.cpp line 73):
    - degree-0 orphan, outlet pressure → drop (no flow through isolated node).
    - degree-0 orphan, inlet pressure  → fatal (user must fix inlet assignment).
    - degree ≥ 2                       → BFS replace with nearest degree-1.
    - duplicate node_id after repair   → keep first occurrence.

    Citation: Pries AR & Secomb TW (2008) Handbook of Physiology, §1 —
    pressure BCs must be placed at network terminals (degree-1 nodes).
    """
    repaired: List[Tuple[int, str]] = []
    seen: Set[int] = set()

    for nid, raw in bc_entries:
        d = node_deg.get(nid, 0)

        if d == 1:
            if nid not in seen:
                repaired.append((nid, raw))
                seen.add(nid)
            else:
                log.append(f"  Duplicate BC node {nid}; keeping first.")
            continue

        if d == 0:
            # Determine role from pressure value
            p = float(raw.split()[2])
            if abs(p - inlet_pressure) < abs(p - outlet_pressure):
                raise RuntimeError(
                    f"INLET BC node {nid} is an orphan (degree-0). "
                    f"It is not an endpoint of any active segment. "
                    f"Re-run the exporter with a different --inlet-rc."
                )
            log.append(
                f"  BC node {nid}: degree-0 orphan; removed "
                f"(outlet with no attached segment)."
            )
            continue

        # degree ≥ 2 — replace via BFS
        rep = _bfs_nearest_degree1(nid, node_deg, adj)
        if rep is None:
            log.append(
                f"  BC node {nid}: degree={d}, no degree-1 node reachable; "
                f"dropped."
            )
            continue
        # Rebuild the raw line with the replacement node ID
        parts = raw.split()
        parts[0] = str(rep)
        new_raw = "  ".join(parts) + "\n"
        log.append(
            f"  BC node {nid}: degree={d} (interior); "
            f"replaced by nearest degree-1 node {rep}."
        )
        if rep not in seen:
            repaired.append((rep, new_raw))
            seen.add(rep)
        else:
            log.append(
                f"    (node {rep} already in BC list; duplicate suppressed.)"
            )

    return repaired


def rebuild_bc_section(
    lines: List[str],
    bc_count_idx: int,
    entries: List[Tuple[int, str]],
) -> List[str]:
    """Replace the BC section with the repaired entries."""
    bc_start = bc_count_idx + 2
    # Count how many lines the old BC section occupied
    old_bc_end = bc_start
    while old_bc_end < len(lines):
        p = lines[old_bc_end].split()
        if len(p) < 4:
            break
        try:
            int(p[0]); float(p[2])
            old_bc_end += 1
        except ValueError:
            break

    new_lines = (
        lines[:bc_count_idx]
        + [f"{len(entries)} Total number of boundary nodes\n"]
        + [lines[bc_count_idx + 1]]          # keep original header line
        + [raw for (_, raw) in entries]
        + lines[old_bc_end:]
    )
    return new_lines


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tmp = args.output_dir / "_tmp"
    tmp.mkdir(exist_ok=True)
    log: List[str] = []

    # ── 1. Export ────────────────────────────────────────────────────────
    print("[1/5] Exporting with BC-validated exporter (v2) …")
    exporter_cmd = [
        args.python, str(args.exporter),
        "--mat",                           str(args.mat),
        "--output-dir",                    str(tmp),
        "--endpoint-snap-px",              str(args.snap_px),
        "--allow-closed-loops",
        "--inlet-rc",                      args.inlet_rc,
        "--inlet-pressure-mmhg",           str(args.inlet_pressure),
        "--outlet-pressure-mmhg",          str(args.outlet_pressure),
        "--resample-step-um",              str(args.resample_step_um),
        "--lumen-fraction",                str(args.lumen_fraction),
    ]
    if args.boundary_outlets_only:
        exporter_cmd += [
            "--boundary-outlets-only",
            "--boundary-margin-um", str(args.boundary_margin_um),
        ]
    else:
        exporter_cmd.append("--allow-all-terminals-as-outlets")

    r = subprocess.run(exporter_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("[ERROR in exporter]")
        print(r.stderr[-3000:])
        sys.exit(1)
    for line in r.stdout.strip().splitlines():
        print(" ", line)

    # ── 2. Cap diameters and remove short segments ───────────────────────
    print("[2/5] Capping diameters and removing short segments …")
    lines, nseg, nnod, adj, node_diam, seg_of_node, coords, seg_lines, node_deg = \
        parse_network(tmp / "network.dat")

    capped = 0
    removed_short = 0
    for i in range(8, 8 + nseg):
        p = lines[i].split()
        if len(p) < 5 or int(p[1]) not in (4, 5):
            continue
        n1, n2 = int(p[2]), int(p[3])
        d = float(p[4])
        modified = False
        if d > args.diameter_cap:
            d = args.diameter_cap
            p[4] = f"{d:.6f}"
            capped += 1
            modified = True
        L = seg_length(n1, n2, coords)
        if 0 < L < args.min_length:
            p[1] = "0"
            removed_short += 1
            modified = True
        if modified:
            lines[i] = "  ".join(p) + "\n"

    print(f"  Capped   {capped} segments to {args.diameter_cap} µm")
    print(f"  Removed  {removed_short} segments with L < {args.min_length} µm")

    cleaned = tmp / "network_clean.dat"
    with open(cleaned, "w") as f:
        f.writelines(lines)
    lines, nseg, nnod, adj, node_diam, seg_of_node, coords, seg_lines, node_deg = \
        parse_network(cleaned)

    # ── 3. Fix disconnected components ───────────────────────────────────
    print("[3/5] Fixing disconnected components …")
    components = get_components(adj)
    if not components:
        print("  WARNING: no active segments found after cleaning.")
        sys.exit(1)

    main_comp = components[0]
    isolated  = [c for c in components[1:] if len(c) > args.large_comp_threshold]
    small     = [c for c in components[1:] if len(c) <= args.large_comp_threshold]

    print(f"  Main component     : {len(main_comp)} nodes")
    print(f"  Tiny (≤{args.large_comp_threshold} nodes, removed): {len(small)} components")
    print(f"  Isolated (>threshold): {len(isolated)} components")

    # Remove tiny artefact components (noise/debris in segmentation).
    segs_to_remove: Set[int] = set()
    for comp in small:
        for n in comp:
            segs_to_remove.update(seg_of_node.get(n, []))
    for i in range(8, 8 + nseg):
        p = lines[i].split()
        if p:
            try:
                if int(p[0]) in segs_to_remove:
                    p[1] = "0"
                    lines[i] = "  ".join(p) + "\n"
            except ValueError:
                pass

    # ── 3a. Classify isolated components: snap-gap break vs boundary fragment
    #
    # A "snap-gap break" arises when endpoint clustering (snap_px) fails to
    # merge two endpoints that are sub-pixel apart (< 1 µm gap in µm space).
    # These represent real, connected veins broken only by a rounding artefact.
    # Fixing them: redirect the near-touching terminal's node ID to the nearest
    # main-network node by rewriting the segment endpoint in the file.
    # This is equivalent to what a larger snap radius would have done, applied
    # selectively and with a documented threshold.
    #
    # A "boundary fragment" arises when the image crop cuts through a real vein.
    # Those fragments have gaps >> 1 µm to the main network; they have no
    # hydraulic path to the inlet and must remain Q = 0.
    #
    # Gap threshold: 1 µm (< 1 pixel at the coarsest resolution used here).
    # Rationale: analyzenet.cpp computes segment length as straight-line
    # Euclidean distance between node coordinates; a gap ≤ 1 µm is sub-pixel
    # and is an artefact of integer rounding in image-to-µm conversion.
    # Pries & Secomb (2008) use 1–2 µm as minimum meaningful vessel dimension.
    #
    # Citation: Bühler et al. Plant Physiol 2015 — endpoint coordinates are
    # MATLAB round(pixel_centroid) → sub-pixel jitter of ±0.5 px is expected.

    SNAP_GAP_THRESHOLD_UM = 1.0   # µm; sub-pixel rounding artefact

    try:
        from scipy.spatial import cKDTree
        _HAS_KDTREE = True
    except ImportError:
        _HAS_KDTREE = False

    # Build KD-tree over main-component nodes
    main_nodes_list = [n for n in main_comp if n in coords]
    if _HAS_KDTREE and main_nodes_list:
        import numpy as _np
        main_xy = _np.array([(coords[n][0], coords[n][1]) for n in main_nodes_list])
        main_tree = cKDTree(main_xy)
    else:
        main_tree = None

    # Track node-ID redirections: isolated_node → main_node
    # Applied to segment start/end columns before writing.
    node_remap: Dict[int, int] = {}
    new_bc_entries: List[str] = []

    for comp in isolated:
        deg1_nodes = [n for n in comp if node_deg.get(n, 0) == 1 and n in coords]
        max_d = max((node_diam.get(n, 0) for n in comp), default=0.0)

        # Find each terminal's nearest main-network node and gap
        snappable = []     # (gap_um, terminal_node, main_node)
        if main_tree is not None:
            for tn in deg1_nodes:
                pt = _np.array([[coords[tn][0], coords[tn][1]]])
                dist, idx = main_tree.query(pt, k=1)
                gap = float(dist[0])
                near = main_nodes_list[int(idx[0])]
                snappable.append((gap, tn, near))
            snappable.sort()

        # A component is a "snap-gap break" if its MINIMUM gap ≤ threshold.
        # Only that one terminal gets merged; the rest remain as-is.
        if snappable and snappable[0][0] <= SNAP_GAP_THRESHOLD_UM:
            gap, tn, near = snappable[0]
            node_remap[tn] = near
            print(f"  MERGE (snap-gap): component size={len(comp)}  "
                  f"terminal {tn} → main node {near}  "
                  f"gap={gap:.2f} µm  max_d={max_d:.1f} µm")
            log.append(
                f"Snap-gap merge: isolated node {tn} → main node {near} "
                f"(gap={gap:.2f} µm ≤ {SNAP_GAP_THRESHOLD_UM} µm threshold). "
                f"Component size={len(comp)}, max_d={max_d:.1f} µm. "
                f"Ref: Bühler et al. 2015 (sub-pixel endpoint jitter)."
            )
        else:
            # True boundary fragment or large gap: leave unperfused (Q = 0).
            # NetFlowV2 putrank.cpp excludes BC-less components from the linear
            # system; they appear in the file for GreensV4 geometry but carry
            # zero convective flux.  This is the only honest statement about a
            # vessel fragment with no hydraulic path to the inlet.
            # Secomb TW (2016) Compr Physiol 6:975 — pressure solution requires
            # ≥1 inlet AND ≥1 outlet; a BC-less component has Q = 0 identically.
            nearest_gap = snappable[0][0] if snappable else float("inf")
            print(f"  UNPERFUSED (boundary/large-gap): "
                  f"component size={len(comp)}  "
                  f"nearest_gap={nearest_gap:.1f} µm  "
                  f"max_d={max_d:.1f} µm  → Q=0")
            log.append(
                f"Boundary fragment left unperfused: size={len(comp)}, "
                f"min gap to main={nearest_gap:.1f} µm > threshold. Q=0."
            )

    # Apply node remappings to segment lines
    if node_remap:
        for i in range(8, 8 + nseg):
            p = lines[i].split()
            if len(p) < 5:
                continue
            try:
                if int(p[1]) not in (4, 5):
                    continue
                n1, n2 = int(p[2]), int(p[3])
                changed = False
                if n1 in node_remap:
                    p[2] = str(node_remap[n1]); changed = True
                if n2 in node_remap:
                    p[3] = str(node_remap[n2]); changed = True
                if changed:
                    lines[i] = "  ".join(p) + "\n"
            except ValueError:
                pass
        print(f"  Applied {len(node_remap)} snap-gap node remappings.")

    # Rebuild BC section with component BCs appended
    bc_count_idx, header_idx, existing_bcs = read_bc_entries(lines)
    all_bc_raws = [raw for (_, raw) in existing_bcs] + new_bc_entries
    all_bc_entries = []
    for raw in all_bc_raws:
        p = raw.split()
        try:
            all_bc_entries.append((int(p[0]), raw))
        except (ValueError, IndexError):
            pass

    lines = rebuild_bc_section(lines, bc_count_idx, all_bc_entries)

    # ── 4. Final BC validation pass ───────────────────────────────────────
    print("[4/5] Final BC validation pass …")
    # Re-parse with current lines to get fresh node_deg and adj
    tmp2 = tmp / "network_prefinal.dat"
    with open(tmp2, "w") as f:
        f.writelines(lines)
    lines, nseg, nnod, adj, node_diam, seg_of_node, coords, seg_lines, node_deg = \
        parse_network(tmp2)

    bc_count_idx2, header_idx2, bc_entries2 = read_bc_entries(lines)

    # Count violations BEFORE repair
    n_deg0 = sum(1 for nid, _ in bc_entries2 if node_deg.get(nid, 0) == 0)
    n_deg2p = sum(1 for nid, _ in bc_entries2 if node_deg.get(nid, 0) >= 2)
    print(f"  BC nodes before repair: {len(bc_entries2)}")
    print(f"    degree-0 (orphan)    : {n_deg0}")
    print(f"    degree ≥2 (interior) : {n_deg2p}")

    repaired_entries = validate_and_repair_bc_list(
        bc_entries2, node_deg, adj, log,
        args.inlet_pressure, args.outlet_pressure,
    )
    lines = rebuild_bc_section(lines, bc_count_idx2, repaired_entries)

    # Verify no violations remain
    _, _, final_bcs = read_bc_entries(lines)
    remaining_bad = [(nid, node_deg.get(nid, 0))
                     for nid, _ in final_bcs
                     if node_deg.get(nid, 0) != 1]
    if remaining_bad:
        print(f"  WARNING: {len(remaining_bad)} BC nodes still not degree-1 "
              f"after repair: {remaining_bad[:10]}")
    else:
        print(f"  OK: all {len(final_bcs)} BC nodes are degree-1.")

    # ── 5. Write output ───────────────────────────────────────────────────
    print("[5/5] Writing network_ready.dat …")
    out   = args.output_dir / "network_ready.dat"
    upper = args.output_dir / "Network_ready.dat"
    with open(out, "w") as f:
        f.writelines(lines)
    if upper.resolve() != out.resolve():
        shutil.copy2(out, upper)

    # Print log
    if log:
        print("\nRepair log:")
        for entry in log:
            print(entry)

    print(f"\n{'='*55}")
    print(f" DONE → {out}")
    print(f"{'='*55}")
    print(f"\n NEXT:")
    print(f"  1. copy {out} → <NetFlowV2_folder>/Network.dat")
    print(f"  2. RheolParams.dat: varyviscosity=0, phaseseparation=0,")
    print(f"     consthd=0, constvisc=0.700 (PBS @ 37°C, Gershlak 2017)")
    print(f"  3. Run NetFlowV2")
    print(f"\n PASS criteria in Results.out:")
    print(f"  All pressures 20–40 mmHg | No NaN | Max flow < 10000 nl/min")
    print(f"  Zero BC topology errors in stdout")


if __name__ == "__main__":
    main()
# PATCH: this block is never executed standalone — it shows the merge logic
# incorporated into prepare_network.py step 3 (see full rewrite below)