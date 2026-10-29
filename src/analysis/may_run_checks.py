"""Check the NetFlowV2 and GreensV4 files kept from the May 2026 runs.

No solver is needed. Writes results/leaf_ACJ01/may_run_checks.csv (every
number the README quotes about the May runs) and a flow map figure.

Run from the repository root:

    python src/analysis/may_run_checks.py
"""
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

sys.path.insert(0, str(Path(__file__).parent))
import network_io as nio

ROOT = Path(__file__).resolve().parents[2]
NF = ROOT / "data" / "leaf_ACJ01" / "netflow_may"
GV = ROOT / "data" / "leaf_ACJ01" / "greensv4_may"
OUT = ROOT / "results" / "leaf_ACJ01"
OUT.mkdir(parents=True, exist_ok=True)

rows = []


def add(stage, check, value, note=""):
    rows.append(dict(stage=stage, check=check, value=value, note=note))


# ---- NetFlowV2 output (May) -----------------------------------------------------
seg, xy, bc, _ = nio.read_network(NF / "NetworkNew.dat")
flowing = seg[seg.flow.abs() > 1e-6]
inlet = int(bc.node[bc.value.idxmax()])
outlets = bc[bc.node != inlet]
net = nio.node_net_flow(seg)
bset = set(bc.node)
q_in = abs(net[inlet])

add("NetFlowV2", "segments", len(seg))
add("NetFlowV2", "nodes", len(xy))
add("NetFlowV2", "boundary nodes", len(bc), "1 inlet + outlets")
add("NetFlowV2", "inlet pressure (mmHg)", bc.value.max())
add("NetFlowV2", "outlet pressure (mmHg)", round(outlets.value.median(), 6))
add("NetFlowV2", "inlet flow (nl/min)", round(q_in, 3))
add("NetFlowV2", "outlet flow (nl/min)", round(sum(abs(net[n]) for n in outlets.node), 3),
    "should equal inlet flow")
add("NetFlowV2", "max interior node imbalance (nl/min)",
    f"{max(abs(v) for n, v in net.items() if n not in bset):.3g}", "solver residual")
add("NetFlowV2", "segments with non-zero flow (%)", round(100 * len(flowing) / len(seg), 1))

out_q = sorted(((abs(net[n]), math.dist(xy[n], xy[inlet])) for n in outlets.node), reverse=True)
add("NetFlowV2", "inflow leaving through the top 2 outlets (%)",
    round(100 * (out_q[0][0] + out_q[1][0]) / sum(q for q, _ in out_q), 1),
    f"outlets {out_q[0][1]:.0f} and {out_q[1][1]:.0f} um from the inlet")
add("NetFlowV2", "median |flow| / inlet flow", f"{(flowing.flow.abs() / q_in).median():.2e}")

blind = seg.loc[sorted(nio.blind_branches(seg, bset))]
zero = set(seg.index[seg.flow.abs() <= 1e-6])
add("NetFlowV2", "zero-flow segments in blind-ending branches (%)",
    round(100 * len(zero & set(blind.index)) / len(zero), 1))
add("NetFlowV2", "independent loops (whole network)", nio.independent_loops(nio.active(seg)))
add("NetFlowV2", "directed cycle in flow graph", nio.has_directed_cycle(flowing),
    "flow follows pressure, so there should be none")

# ---- GreensV4 input and log (May) -----------------------------------------------
gseg, gxy, gbc, graw = nio.read_network(GV / "Network.dat")
g = nio.active(gseg)
gb = set(gbc.node)
lines = (GV / "Network.dat").read_text().splitlines()
nx, ny, nz = (int(v) for v in lines[2].split()[:3])
maxl = float(lines[4].split()[0])
inlet_row = [l for l in graw["bc_lines"] if float(l.split()[2]) == gbc.value.max()][0]

add("GreensV4", "inlet / outlet pressure (mmHg)", f"{gbc.value.max():g} / {gbc.value.median():g}",
    "a different NetFlowV2 run from the one above")
add("GreensV4", "inlet PO2 in input file (mmHg)", float(inlet_row.split()[4]),
    "should be 150; set_po2_in_network() bug")
add("GreensV4", "active segments", len(g))
add("GreensV4", "active segments with flow exactly 0", int((g.flow == 0).sum()))
add("GreensV4", "nodes putrank.cpp can rank (%)", round(100 * nio.putrank_fraction(g), 1))
fake, bal = nio.greens_convect_check(g, gb)
add("GreensV4", "degree-1 non-boundary nodes with outflow (fake inlets)", fake,
    "-> 'wrong segment sequence in convect'")
add("GreensV4", "nodes failing the 1% flow balance in convect.cpp", bal,
    "-> 'Flow conservation violation'")
add("GreensV4", "directed cycle in flow graph", nio.has_directed_cycle(g[g.flow != 0]))
add("GreensV4", "independent loops", nio.independent_loops(g))

nnv = int(sum(max(1, math.ceil(L / maxl)) for L in g.length_um))
add("GreensV4", "vessel points nnv (est.)", nnv, f"segments split at {maxl:g} um")
add("GreensV4", "tissue points nnt", nx * ny * nz, f"{nx} x {ny} x {nz}")
add("GreensV4", "memory for gvv + al + mat (GB, est.)", round(16 * nnv ** 2 / 1e9, 2),
    "float + float + double nnv x nnv (setuparrays2.cpp)")
add("GreensV4", "same for the 61,021-point run in the manuscript (GB)", round(16 * 61021 ** 2 / 1e9, 1))

log = (GV / "GreensLog.txt").read_text()
add("GreensV4", "qvsum is NaN in GreensLog.txt", "qvsum = nan" in log)

table = pd.DataFrame(rows)
table.to_csv(OUT / "may_run_checks.csv", index=False)
print(table.to_string(index=False))

# ---- Figure: where the May NetFlowV2 flow goes ----------------------------------
(OUT / "figures").mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(6, 8.5))
q = seg.flow.abs().to_numpy()
segs_xy = [[xy[a], xy[b]] for a, b in zip(seg.start, seg.end)]
z = q <= 1e-6
ax.add_collection(LineCollection([s for s, k in zip(segs_xy, z) if k], colors="0.8", linewidths=0.5))
lc = LineCollection([s for s, k in zip(segs_xy, z) if not k], array=np.log10(q[~z] / q_in),
                    cmap="viridis", linewidths=0.8)
ax.add_collection(lc)
ax.plot(*xy[inlet], "r^", ms=10, label="inlet")
ax.plot([xy[n][0] for n in outlets.node], [xy[n][1] for n in outlets.node], "ko", ms=3, label="outlets")
ax.autoscale(); ax.set_aspect("equal"); ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
fig.colorbar(lc, ax=ax, shrink=0.6, label="log10(|flow| / inlet flow)")
ax.legend(loc="lower right", fontsize=8)
ax.set_title("May NetFlowV2 run: flow relative to inlet\n(grey = zero flow)")
fig.tight_layout()
fig.savefig(OUT / "figures" / "may_flow_map.png", dpi=120)
print("\nSaved results/leaf_ACJ01/may_run_checks.csv and figures/may_flow_map.png")
