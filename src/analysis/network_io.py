"""Read NetFlowV2 / GreensV4 network files and run graph checks on them.

Used by the analysis scripts in this folder. Every function works on the plain
text Network.dat format shared by NetFlowV2 and GreensV4.
"""
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------- reading / writing

def read_network(path):
    """Return (segments DataFrame, {node: (x, y)}, boundary DataFrame, raw parts).

    segments: seg, type, start, end, diam, flow, length_um
    boundary: node, bctype, value (pressure or flow), and the raw text line
    raw parts are kept so a modified file can be written back in the same format.
    """
    lines = Path(path).read_text().splitlines()
    nseg = int(lines[6].split()[0])
    seg_lines = lines[8:8 + nseg]
    i = 8 + nseg
    nnod = int(lines[i].split()[0])
    node_lines = lines[i + 2:i + 2 + nnod]
    j = i + 2 + nnod
    nbc = int(lines[j].split()[0])
    bc_lines = lines[j + 2:j + 2 + nbc]

    seg = pd.DataFrame([dict(seg=int(p[0]), type=int(p[1]), start=int(p[2]), end=int(p[3]),
                             diam=float(p[4]), flow=float(p[5]))
                        for p in (ln.split() for ln in seg_lines)])
    xy = {int(p[0]): (float(p[1]), float(p[2])) for p in (ln.split() for ln in node_lines)}
    bc = pd.DataFrame([dict(node=int(p[0]), bctype=int(p[1]), value=float(p[2]))
                       for p in (ln.split() for ln in bc_lines)])
    seg["length_um"] = [math.dist(xy[a], xy[b]) for a, b in zip(seg.start, seg.end)]
    raw = dict(header=lines[:6], seg_header=lines[7], node_header=lines[i + 1],
               bc_header=lines[j + 1], seg_lines=seg_lines, node_lines=node_lines, bc_lines=bc_lines)
    return seg, xy, bc, raw


def write_network(path, raw, keep_segs=None, bc_lines=None):
    """Write a Network.dat, optionally keeping only some segment names and replacing BC rows.

    Nodes and boundary nodes no longer used by any kept segment are dropped.
    """
    segs = [s for s in raw["seg_lines"] if keep_segs is None or int(s.split()[0]) in keep_segs]
    used = {int(s.split()[k]) for s in segs for k in (2, 3)}
    nodes = [n for n in raw["node_lines"] if int(n.split()[0]) in used]
    bcs = [b for b in (bc_lines if bc_lines is not None else raw["bc_lines"]) if int(b.split()[0]) in used]
    out = (raw["header"] + [f"{len(segs)}\t\ttotal number of segments", raw["seg_header"]] + segs
           + [f"{len(nodes)} number of nodes", raw["node_header"]] + nodes
           + [f"{len(bcs)} Total number of boundary nodes", raw["bc_header"]] + bcs)
    Path(path).write_text("\n".join(out) + "\n")


# ---------------------------------------------------------------- graph checks

def active(seg):
    """Segments that NetFlowV2 / GreensV4 actually use (type 4 or 5)."""
    return seg[seg.type.isin([4, 5])].reset_index(drop=True)


def components(seg, nodes=None):
    """Number of connected pieces (union-find)."""
    nodes = set(seg.start) | set(seg.end) if nodes is None else nodes
    parent = {n: n for n in nodes}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(seg.start, seg.end):
        parent[find(a)] = find(b)
    return len({find(n) for n in nodes})


def independent_loops(seg):
    """Cycle rank of the undirected graph: edges - nodes + components."""
    nodes = set(seg.start) | set(seg.end)
    return len(seg) - len(nodes) + components(seg, nodes)


def has_directed_cycle(seg):
    """True if the flow-directed graph contains a cycle (Kahn's algorithm).

    Flow driven by a pressure field always runs from higher to lower pressure,
    so a solved network should never contain one.
    """
    out, indeg = defaultdict(list), defaultdict(int)
    for a, b, q in zip(seg.start, seg.end, seg.flow):
        u, v = (a, b) if q > 0 else (b, a)
        out[u].append(v)
        indeg[v] += 1
        indeg.setdefault(u, 0)
    queue = [n for n, d in indeg.items() if d == 0]
    seen = 0
    while queue:
        u = queue.pop()
        seen += 1
        for v in out[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    return seen < len(indeg)


def blind_branches(seg, boundary):
    """Index labels of segments in blind-ending branches.

    Repeatedly strips dead-end nodes that are not boundary nodes. These
    branches cannot carry through-flow, so any flow NetFlowV2 reports in them
    is solver round-off.
    """
    adj = defaultdict(set)
    for i, a, b in zip(seg.index, seg.start, seg.end):
        adj[a].add(i)
        adj[b].add(i)
    deg = {n: len(s) for n, s in adj.items()}
    removed = set()
    stack = [n for n, d in deg.items() if d == 1 and n not in boundary]
    while stack:
        n = stack.pop()
        if deg[n] != 1 or n in boundary:
            continue
        i = next(i for i in adj[n] if i not in removed)
        removed.add(i)
        deg[n] = 0
        other = seg.end[i] if seg.start[i] == n else seg.start[i]
        deg[other] -= 1
        if deg[other] == 1 and other not in boundary:
            stack.append(other)
    return removed


def greens_convect_check(seg, boundary):
    """Count the two GreensV4 convect.cpp error sources on a network.

    fake_inlets: degree-1 nodes that are not boundary nodes but have outflow.
        putrank.cpp ranks them as inflow nodes, but convect.cpp only assigns
        oxygen to real boundary inlets, so their segments trigger
        "wrong segment sequence in convect".
    balance_failures: nodes where |inflow - outflow| / (inflow + outflow) > 1%,
        which trigger "Flow conservation violation".
    """
    out_n, in_n = defaultdict(int), defaultdict(int)
    qin, qout = defaultdict(float), defaultdict(float)
    for a, b, q in zip(seg.start, seg.end, seg.flow):
        u, v = (a, b) if q >= 0 else (b, a)
        out_n[u] += 1
        in_n[v] += 1
        qout[u] += abs(q)
        qin[v] += abs(q)
    nodes = set(out_n) | set(in_n)
    fake = sum(1 for n in nodes if out_n[n] == 1 and in_n[n] == 0 and n not in boundary)
    bal = sum(1 for n in nodes if in_n[n] + out_n[n] > 1 and qin[n] + qout[n] > 0
              and abs(qin[n] - qout[n]) / (qin[n] + qout[n]) > 0.01)
    return fake, bal


def putrank_fraction(seg):
    """Fraction of nodes GreensV4's putrank.cpp can rank (line-by-line port of its rule).

    A node is ranked once every inflow segment comes from a ranked node with
    strictly positive flow in that direction. A segment with flow exactly 0
    can never satisfy this, so nothing downstream of it is ranked.
    """
    nodes = set(seg.start) | set(seg.end)
    out_n, inflow = defaultdict(int), defaultdict(list)
    for a, b, q in zip(seg.start, seg.end, seg.flow):
        n1, n2 = (a, b) if q >= 0 else (b, a)
        out_n[n1] += 1
        inflow[n2].append((a, b, q))
    ranked = {n for n in nodes if out_n[n] == 1 and not inflow[n]}
    changed = True
    while changed:
        changed = False
        for n in nodes - ranked:
            ok = all(not ((n == b and (a not in ranked or q <= 0)) or
                          (n == a and (b not in ranked or q >= 0)))
                     for a, b, q in inflow[n])
            if ok:
                ranked.add(n)
                changed = True
    return len(ranked) / len(nodes)


def node_net_flow(seg):
    """Net flow into each node (positive = more in than out)."""
    net = defaultdict(float)
    for a, b, q in zip(seg.start, seg.end, seg.flow):
        net[a] -= q
        net[b] += q
    return net
