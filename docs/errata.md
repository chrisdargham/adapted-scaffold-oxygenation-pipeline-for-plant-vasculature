# Errata to the May 2026 manuscript

The manuscript (`capstone_manuscript_May2026.pdf`) and slides are kept as
submitted. In October 2026 the surviving files were re-analysed, the two
Secomb-lab solvers were rebuilt from source, and every claim was checked
against a file. This page lists what changed. Each point names the file or
script that supports it.

## 1. GreensV4 did not fail because of leaf loops

**Manuscript (§2.7, §2.8, slides 9–11):** GreensV4's convective pass needs an
upstream-to-downstream order, and the loops of the reticulated leaf make the
flow graph impossible to rank ("Loop B↔C means neither can be processed
first"). Pruning low-flow segments to fix this would destroy the leaf
topology.

**What the files show:**

- Neither flow graph contains a directed cycle. They can't: NetFlowV2 computes
  each flow from a pressure difference, so flow always runs from higher to
  lower pressure, and a loop of decreasing pressures cannot close.
  (`results/leaf_ACJ01/may_run_checks.csv`)
- GreensV4's ranking routine (`putrank.cpp`) only ranks a node when every
  inflow segment carries strictly positive flow. In the May input, 3,614
  active segments had flow exactly 0, so only 8.3% of nodes could be ranked.
- `convect.cpp` treats any degree-1 node with outflow as an inlet, but only
  supplies oxygen at real boundary inlets. Blind-ending veins carry tiny
  round-off flows in random directions, which created 725 fake inlets
  ("wrong segment sequence") and 908 nodes failing GreensV4's 1% flow-balance
  check ("Flow conservation violation").
- Removing segments with zero flow and blind-ending branches (which carry no
  through-flow) keeps **468 of 476 loops**, one connected network, and clears
  every ranking and balance error.
  (`results/leaf_ACJ01/greensv4/diagnosis.csv`, `src/analysis/greens_runs.py`)

So the cleanup the manuscript rejected keeps the reticulated topology. The
failure was caused by dead-end veins and zero flows, not by loops.

With the cleanup and the patch in point 2, GreensV4 runs on the leaf with no
errors or NaN, but it still does not converge: the vessel PO2 error levels off
3,000 to 8,000 times above tolerance (`results/leaf_ACJ01/greensv4/run_summary.csv`,
`results/leaf_ACJ01/figures/greensv4_convergence.png`). The manuscript's
practical conclusion stands: GreensV4 gave no usable oxygen map for this leaf.
The reason it gave is what changes.

## 2. GreensV4 returns NaN for any cell-free perfusate

**Manuscript (§2.7, Table):** NaN outputs were taken as a consequence of the
topology failure.

**What the files show:** GreensV4's own sample problem (a rat tumour network)
also returns NaN as soon as blood is replaced by a perfusate with haematocrit
0. The first guess for vessel oxygen fluxes (`initgreens.cpp`, line 43) divides
by the haematocrit-weighted vessel length, which is 0 for PBS. A one-line patch
uses the flow-weighted guess instead
(`solvers/patches/greensv4_cell_free_perfusate.patch`). With the patch, the
blood sample gives an identical answer, and the cell-free sample converges.

## 3. The May GreensV4 input had no oxygen at the inlet

The kept GreensV4 input file has inlet PO2 = 0 mmHg. The cause is a bug in
`set_po2_in_network()` (`run_pipeline.py`): it switched itself off on the
header line just before the boundary rows, so no PO2 was ever written. A second
problem: it labelled any node within 0.5 mmHg of the inlet pressure as an
inlet, which with the May pressure drop (0.0045 mmHg) would have included every
outlet. Both are fixed and tested (`tests/test_set_po2.py`).

## 4. Memory: the mechanism is right, the scale depends on the run

**Manuscript (§2.6):** 61,021 vessel points; a single dense float matrix of
14.9 GB.

GreensV4 allocates three dense nnv × nnv matrices (`gvv` and `al` as float,
`mat` as double; `setuparrays2.cpp`), about 16 × nnv² bytes. That is about
60 GB at 61,021 points, so the conclusion holds for that run. The run that was
kept had about 18,000 vessel points (5.4 GB), and about 12,000 after cleanup
(2.4 GB), which fits a normal workstation. The 61,021-point log was not kept.

## 5. The May NetFlowV2 run barely perfused the leaf

The May boundary conditions put 44 outlets around the leaf edge at
39.9955 mmHg, 0.0045 mmHg below the inlet. 99.5% of the inflow left through two
outlets 125 and 177 µm from the inlet. With outlets only on the far edge,
56.5% of vein length carries at least 0.1% of the inflow, against 0.5% before.
(`results/leaf_ACJ01/netflow/outlet_study.csv`)

The manuscript's results table (§4.1, slide 8) reports a 1,130-segment
demonstration network (31 inlets, 5 outlets), not the leaf. Its files were not
kept.

## 6. Oxygen solver: overshoot at high wall permeability

The explicit wall-exchange update overshoots when `dt × k` is large. At
`P_wall` = 1e-4 m/s, tissue pO2 went far above the inlet value, which is
impossible. The exchange term is now semi-implicit
(`src/solver/patches/wall_exchange_fix.patch`). Runs at `P_wall` ≤ 1e-6, which
include all runs in the May sweep report, are unchanged
(`analysis/oxygen_sweep_analysis.md`, section 5).

## 7. Sweep design is stated three different ways

§3.8 lists `P_wall` = 1e-8, 1e-6, 3e-5, 1e-4 (24 runs). §3.11 and slide 16 list
1e-6, 3e-5, 1e-4. The May sweep report used 1e-8 and 1e-6 (12 runs). This
repository runs all four values (24 runs).

## 8. Smaller points

- §2.1 says "five stages" and lists four.
- The May sweep report PDF is not included here. Its numbers are recorded in
  `data/leaf_ACJ01/may_sweep_report_values.csv` and are reproduced by
  `analysis/oxygen_sweep_analysis.Rmd`, which replaces it.
