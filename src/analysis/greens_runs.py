"""Diagnose why GreensV4 failed on the leaf, and run it on a cleaned network.

Needs the compiled solver (bash solvers/build_solvers.sh). The diagnosis step
takes seconds. A GreensV4 run uses about 2.5 GB of memory and takes about 75 s
per main iteration on the coarse tissue grid (64 x 96) or about 6 min on the
May grid (256 x 385), for up to 100 iterations.

Result (October 2026): after cleaning, and with the cell-free perfusate patch,
GreensV4 runs with no errors and no NaN, but does not converge: the vessel PO2
error levels off 3,000 to 8,000 times above tolerance. See run_summary.csv and
figures/greensv4_convergence.png.

What GreensV4 needs (from its source code):
  * putrank.cpp orders nodes upstream to downstream. A node is ranked only when
    every inflow segment comes from a ranked node with strictly positive flow.
    A segment with flow exactly 0 blocks everything downstream of it.
  * convect.cpp treats any degree-1 node with outflow as an inlet, but only
    assigns oxygen at real boundary inlets ("wrong segment sequence"), and it
    flags any node where inflow and outflow differ by more than 1%
    ("Flow conservation violation").

Blind-ending veins carry no through-flow. NetFlowV2 gives them either 0 or a
tiny round-off flow in a random direction, which triggers all three problems.
Cleaning = repeatedly remove segments with flow exactly 0, then blind-ending
branches, until nothing changes. Loops are kept.

Steps (run from the repository root):

    python src/analysis/greens_runs.py diagnose     # writes greensv4/diagnosis.csv
    python src/analysis/greens_runs.py prepare      # builds the run folders
    python src/analysis/greens_runs.py run NAME     # runs GreensV4 in one folder
    python src/analysis/greens_runs.py summarise    # writes greensv4/run_summary.csv
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import network_io as nio

ROOT = Path(__file__).resolve().parents[2]
MAY = ROOT / "data" / "leaf_ACJ01" / "greensv4_may"
CFG = ROOT / "configs" / "greensv4"
OUT = ROOT / "results" / "leaf_ACJ01" / "greensv4"
RUNS = OUT / "runs"
GREENS = ROOT / "solvers" / "greens"
FAR_EDGE = ROOT / "results" / "leaf_ACJ01" / "netflow" / "runs" / "C_far_edge_outlets" / "NetworkNew.dat"
INLET_PO2 = 150.0


def clean(seg, boundary):
    """Remove zero-flow segments and blind-ending branches until nothing changes."""
    s = nio.active(seg)
    while True:
        n0 = len(s)
        s = s[s.flow != 0].reset_index(drop=True)
        s = s.drop(index=sorted(nio.blind_branches(s, boundary))).reset_index(drop=True)
        if len(s) == n0:
            return s


def describe(label, s, boundary):
    fake, bal = nio.greens_convect_check(s, boundary)
    return dict(network=label, active_segments=len(s),
                segments_flow_exactly_0=int((s.flow == 0).sum()),
                independent_loops=nio.independent_loops(s),
                connected_pieces=nio.components(s),
                nodes_putrank_can_rank_pct=round(100 * nio.putrank_fraction(s), 1),
                fake_inlets=fake, nodes_failing_1pct_balance=bal,
                directed_cycle=nio.has_directed_cycle(s[s.flow != 0]))


def inputs():
    """(label, network file, short name) for each network studied."""
    if not FAR_EDGE.exists():
        sys.exit("Far-edge network not found. Run python src/analysis/netflow_rerun.py first.")
    return [("May GreensV4 input (May outlets, 40 -> 35 mmHg)", MAY / "Network.dat", "may_outlets"),
            ("Far-edge outlets (outlet study C)", FAR_EDGE, "far_edge_outlets")]


def diagnose():
    rows = []
    for label, path, _ in inputs():
        seg, _, bc, _ = nio.read_network(path)
        b = set(bc.node)
        rows.append(describe(label + ": as given", nio.active(seg), b))
        rows.append(describe(label + ": cleaned", clean(seg, b), b))
    t = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "diagnosis.csv", index=False)
    print(t.to_string(index=False))


def make_folder(name, path, cleaned, grid=None):
    """grid = (nx, ny) overrides the tissue grid (the May file uses 256 x 385, about 50 um)."""
    seg, _, bc, raw = nio.read_network(path)
    if grid:
        raw = dict(raw)
        raw["header"] = raw["header"][:2] + [f"{grid[0]} {grid[1]} 1 number of tissue points in x,y,z directions"] + raw["header"][3:]
    keep = set(clean(seg, set(bc.node)).seg) if cleaned else None
    inlet_p = bc.value.max()
    bcs = []
    for line in raw["bc_lines"]:
        p = line.split()
        p[4] = f"{INLET_PO2 if float(p[2]) == inlet_p else 0.0:.6f}"
        bcs.append(" ".join(p))
    d = RUNS / name
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    nio.write_network(d / "Network.dat", raw, keep_segs=keep, bc_lines=bcs)
    shutil.copy(MAY / "IntravascRes.dat", d / "IntravascRes.dat")       # K_eff = 10 um/s (May)
    shutil.copy(MAY / "Varyparams.dat", d / "VaryParams.dat")
    for f in ("SoluteParams.dat", "ContourParams.dat", "tissrate.cpp.dat"):
        shutil.copy(CFG / f, d / f)
    print(f"prepared {d.relative_to(ROOT)}")


def prepare():
    for _, path, short in inputs():
        make_folder(f"{short}_as_given", path, cleaned=False)
        make_folder(f"{short}_cleaned", path, cleaned=True)
        # Coarse tissue grid (about 200 um): converges in minutes instead of hours.
        make_folder(f"{short}_cleaned_coarse", path, cleaned=True, grid=(64, 96))


def run(name, limit_s=6 * 3600):
    d = RUNS / name
    if not GREENS.exists():
        sys.exit("solvers/greens not found. Build it first: bash solvers/build_solvers.sh")
    t0 = time.time()
    with open(d / "greens_stdout.txt", "w") as log:
        try:
            # stdbuf: line-buffered output, so the log survives if the time limit stops the run
            cmd = ([shutil.which("stdbuf"), "-oL"] if shutil.which("stdbuf") else []) + [str(GREENS.resolve())]
            rc = subprocess.run(cmd, cwd=d, stdout=log, stderr=subprocess.STDOUT, timeout=limit_s).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    (d / "run_info.txt").write_text(f"return code {rc}\nseconds {time.time() - t0:.0f}\n")
    print(f"{name}: return code {rc}, {time.time() - t0:.0f} s")


def parse_log(path):
    """One row per main iteration of a GreensLog.txt: qtsum, qvsum, g0 and the four errors."""
    rows, cur = [], None
    for line in Path(path).read_text().splitlines():
        if "kmain =" in line:
            cur = dict(iteration=int(line.split("kmain =")[1].split(",")[0]))
            rows.append(cur)
        elif cur is not None and "qvsum" in line:
            p = line.replace(",", "").split()
            cur.update(qtsum=float(p[4]), qvsum=float(p[7]), g0=float(p[10]))
        elif cur is not None and "err" in line and "=" in line:
            p = line.replace(",", "").split()
            cur[p[2]] = float(p[4])
    return pd.DataFrame(rows)


def summarise():
    """Summary table, logs and a convergence figure for every run folder."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, logs = [], {}
    folders = sorted(p for p in RUNS.iterdir() if p.is_dir()) if RUNS.exists() else []
    if not any((d / "Current" / "GreensLog.txt").exists() for d in folders):
        sys.exit("No finished GreensV4 runs in results/leaf_ACJ01/greensv4/runs/ "
                 "(run folders are not tracked in git). Kept the existing run_summary.csv.")
    (OUT / "run_logs").mkdir(parents=True, exist_ok=True)
    for d in folders:
        log = d / "Current" / "GreensLog.txt"
        if not log.exists():
            continue
        it = parse_log(log)
        logs[d.name] = it
        stdout = (d / "greens_stdout.txt").read_text(errors="replace") if (d / "greens_stdout.txt").exists() else ""
        info = (d / "run_info.txt").read_text().strip().replace("\n", "; ") if (d / "run_info.txt").exists() else ""
        last = it.iloc[-1]
        nx, ny = (d / "Network.dat").read_text().splitlines()[2].split()[:2]
        rows.append(dict(
            run=d.name, tissue_grid=f"{nx} x {ny}", status=info,
            main_iterations=len(it),
            nan_in_log="nan" in log.read_text().lower(),
            wrong_segment_sequence_errors=stdout.count("wrong segment sequence"),
            flow_conservation_errors=stdout.count("Flow conservation violation"),
            last_qtsum=last.get("qtsum"), last_qvsum=last.get("qvsum"), last_g0=last.get("g0"),
            last_errvessel_p=last.get("errvessel_p"),
            min_errvessel_p=it["errvessel_p"].min() if "errvessel_p" in it else None,
            converged=(d / "Current" / "TissueLevels001.out").exists()))
        shutil.copy(log, OUT / "run_logs" / f"{d.name}_GreensLog.txt")
        if stdout:
            (OUT / "run_logs" / f"{d.name}_stdout_last200.txt").write_text(
                "\n".join(stdout.splitlines()[-200:]) + "\n")
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "run_summary.csv", index=False)
    print(t.T.to_string(header=False))

    # Convergence figure. GreensV4 stops when every error is below 1 (errors are
    # changes per iteration divided by errfac x pref).
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, (name, it) in enumerate(logs.items()):
        style = dict(lw=5 - 3.5 * (k > 0), alpha=0.45 if k == 0 else 1,
                     label=f"{name} ({len(it)} iterations)")
        ax[0].semilogy(it.iteration, it.errvessel_p, **style)
        ax[1].plot(it.iteration, -it.qtsum / it.qvsum, **style)
    ax[0].axhline(1, ls="--", c="k", lw=0.8)
    ax[0].set(xlabel="main iteration", ylabel="errvessel_p (converged below 1)",
              title="Vessel PO2 change per iteration")
    ax[1].axhline(1, ls="--", c="k", lw=0.8)
    ax[1].set(xlabel="main iteration", ylabel="O2 consumed / O2 supplied",
              title="Tissue consumption vs vessel supply", ylim=(0, 2))
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "leaf_ACJ01" / "figures" / "greensv4_convergence.png", dpi=120)
    print("\nSaved results/leaf_ACJ01/greensv4/run_summary.csv and figures/greensv4_convergence.png")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "diagnose"
    if cmd == "diagnose":
        diagnose()
    elif cmd == "prepare":
        prepare()
    elif cmd == "run":
        run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6 * 3600)
    elif cmd == "summarise":
        summarise()
    else:
        sys.exit(__doc__)
