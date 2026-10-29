#!/usr/bin/env bash
# Rerun part of the real-leaf sweep with the May (pre-fix) solver, for comparison.
# Rebuilds the May solver by reverse-applying patches/wall_exchange_fix.patch to a
# temporary copy. 12 runs (P_wall 1e-6 and 1e-4), about 15 minutes.
# Run from the repository root.
set -euo pipefail
TMP=$(mktemp -d)
cp src/solver/oxy_sim_mat.py "$TMP/oxy_sim_mat.py"
patch -s -R "$TMP/oxy_sim_mat.py" src/solver/patches/wall_exchange_fix.patch
python src/solver/o2_sweep.py --solver "$TMP/oxy_sim_mat.py" \
  --base-config configs/leaf_ACJ01_may_settings.json \
  --output-root results/leaf_ACJ01/oxygen_sweep_before_fix \
  --mode full --pwall-values 1e-6,1e-4
rm -rf "$TMP"
