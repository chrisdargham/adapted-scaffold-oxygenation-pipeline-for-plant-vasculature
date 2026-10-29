"""Rerun NetFlowV2 on the real leaf: reproduce the May run, then test outlet choices.

Needs the compiled solver (bash solvers/build_solvers.sh). Takes about 15 s.

1. Reproduction: runs NetFlowV2 on the May input file and compares every segment
   flow with the May output.
2. Outlet study: the May run used all 44 edge terminals as outlets, at
   0.0045 mmHg below the inlet. Three scenarios are compared:
     A  May boundary conditions (44 outlets, 40 -> 39.9955 mmHg)
     B  same, but without the outlets within 1 mm of the inlet (40 -> 35 mmHg)
     C  only outlets on the far (bottom) edge of the leaf (40 -> 35 mmHg)
   Flow scales linearly with the pressure drop here (constant viscosity), so
   the comparison uses flows relative to the inlet flow.

Run from the repository root:

    python src/analysis/netflow_rerun.py
"""
import math
import shutil
import subprocess
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
NETFLOW = ROOT / "solvers" / "netflow"
MAY = ROOT / "data" / "leaf_ACJ01" / "netflow_may"
OUT = ROOT / "results" / "leaf_ACJ01" / "netflow"
RUNS = OUT / "runs"
if not NETFLOW.exists():
    sys.exit("solvers/netflow not found. Build it first: bash solvers/build_solvers.sh")


def run_netflow(folder, network_text):
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    (folder / "Network.dat").write_text(network_text)
    for f in ("RheolParams.dat", "ContourParams.dat"):
        shutil.copy(MAY / f, folder / f)
    with open(folder / "netflow_stdout.txt", "w") as log:
        subprocess.run([str(NETFLOW.resolve())], cwd=folder, stdout=log, stderr=subprocess.STDOUT, check=True)
    shutil.copy(folder / "Current" / "NetworkNew.dat", folder / "NetworkNew.dat")
    return folder / "NetworkNew.dat"


# ---- 1. Reproduce the May run ---------------------------------------------------
new = run_netflow(RUNS / "may_reproduction", (MAY / "Network.dat").read_text())
a, _, _, _ = nio.read_network(MAY / "NetworkNew.dat")
b, _, _, _ = nio.read_network(new)
m = a.merge(b, on="seg", suffixes=("_may", "_now"))
diff = (m.flow_now - m.flow_may).abs()
repro = pd.DataFrame([dict(
    segments_compared=len(m),
    segments_with_flow=int((m.flow_may.abs() > 1e-6).sum()),
    max_abs_flow_difference_nl_min=float(diff.max()),
    inlet_flow_may=float(m.flow_may.abs().max()),
    inlet_flow_now=float(m.flow_now.abs().max()),
    segment_types_identical=bool((m.type_may == m.type_now).all()),
)])
OUT.mkdir(parents=True, exist_ok=True)
repro.to_csv(OUT / "may_reproduction.csv", index=False)
print("Reproduction of the May NetFlowV2 run:")
print(repro.T.to_string(header=False), "\n")

# ---- 2. Outlet study ------------------------------------------------------------
seg0, xy, bc0, raw = nio.read_network(MAY / "Network.dat")
inlet = int(bc0.node[bc0.value.idxmax()])
others = [int(n) for n in bc0.node if n != inlet]
dist = {n: math.dist(xy[n], xy[inlet]) for n in others}

scenarios = {
    "A_may_all_outlets": ("44 edge terminals (May)", others, float(bc0.value.min())),
    "B_no_outlets_near_inlet": ("outlets > 1 mm from inlet", [n for n in others if dist[n] > 1000], 35.0),
    "C_far_edge_outlets": ("bottom-edge outlets only", [n for n in others if xy[n][1] < 1000], 35.0),
}

rows, maps = [], {}
for name, (label, outs, p_out) in scenarios.items():
    bcs = [f"{inlet} 0 40.000000 0.000000 150.000000 *"] + \
          [f"{n} 0 {p_out:.6f} 0.000000 0.000000 *" for n in outs]
    text = "\n".join(raw["header"] + [f"{len(raw['seg_lines'])}\t\ttotal number of segments", raw["seg_header"]]
                     + raw["seg_lines"] + [f"{len(raw['node_lines'])} number of nodes", raw["node_header"]]
                     + raw["node_lines"] + [f"{len(bcs)} Total number of boundary nodes", raw["bc_header"]]
                     + bcs) + "\n"
    res = run_netflow(RUNS / name, text)
    seg, _, _, _ = nio.read_network(res)
    net = nio.node_net_flow(seg)
    q_in = abs(net[inlet])
    out_q = sorted((abs(net[n]) for n in outs), reverse=True)
    q, L = seg.flow.abs(), seg.length_um
    rows.append(dict(
        scenario=name, description=label, outlets=len(outs),
        pressure_drop_mmHg=round(40.0 - p_out, 4),
        inlet_flow_nl_min=round(q_in, 3),
        top2_outlet_share_pct=round(100 * sum(out_q[:2]) / sum(out_q), 1),
        length_q_ge_1pct_of_inflow_pct=round(100 * L[q >= 0.01 * q_in].sum() / L.sum(), 1),
        length_q_ge_0p1pct_of_inflow_pct=round(100 * L[q >= 0.001 * q_in].sum() / L.sum(), 1),
        median_flow_over_inflow=float((q[q > 1e-9] / q_in).median()),
        directed_cycle=nio.has_directed_cycle(seg[q > 1e-6]),
    ))
    maps[name] = (label, seg, q_in, outs)

table = pd.DataFrame(rows)
table.to_csv(OUT / "outlet_study.csv", index=False)
print("Outlet study:")
print(table.to_string(index=False))

# ---- Figure: relative flow under each scenario ----------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 6.5), sharey=True)
for ax, (name, (label, seg, q_in, outs)) in zip(axes, maps.items()):
    rel = np.log10(np.maximum(seg.flow.abs().to_numpy() / q_in, 1e-8))
    lc = LineCollection([[xy[a], xy[b]] for a, b in zip(seg.start, seg.end)], array=rel,
                        cmap="viridis", linewidths=0.7, norm=plt.Normalize(-6, 0))
    ax.add_collection(lc)
    ax.plot(*xy[inlet], "r^", ms=10)
    ax.plot([xy[n][0] for n in outs], [xy[n][1] for n in outs], "ko", ms=3)
    share = table.set_index("scenario").loc[name, "length_q_ge_0p1pct_of_inflow_pct"]
    ax.set_title(f"{name[0]}: {label}\n{share}% of vein length carries >= 0.1% of inflow", fontsize=9)
    ax.autoscale(); ax.set_aspect("equal"); ax.set_xticks([])
axes[0].set_ylabel("y (um)")
fig.colorbar(lc, ax=axes, shrink=0.7, label="log10(|flow| / inlet flow)")
fig.suptitle("Same leaf, same inlet: outlet choice decides where the flow goes "
             "(red = inlet, black = outlets)", fontsize=10, y=0.86)
(OUT.parent / "figures").mkdir(exist_ok=True)
fig.savefig(OUT.parent / "figures" / "outlet_study_flow_maps.png", dpi=120, bbox_inches="tight")
print("\nSaved results/leaf_ACJ01/netflow/*.csv and figures/outlet_study_flow_maps.png")
