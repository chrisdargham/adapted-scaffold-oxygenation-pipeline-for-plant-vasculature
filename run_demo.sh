#!/usr/bin/env bash
# Run every stage of the pipeline on the synthetic leaf (about 1 minute).
# Run from the repository root after: pip install -r requirements.txt
#                                     bash solvers/build_solvers.sh
set -euo pipefail
OUT=results/demo

echo "== 1. Synthetic LeafVeinCNN file"
python src/demo/make_synthetic_mat.py

echo "== 2. Preprocessing: .mat -> NetFlowV2 network"
python src/preprocessing/prepare_network.py \
  --mat data/demo/synthetic_leaf.mat \
  --exporter src/preprocessing/leafveincnn_to_netflow.py \
  --output-dir $OUT/netflow_ready \
  --inlet-rc 20,5 --inlet-pressure 40 --outlet-pressure 35 \
  --boundary-outlets-only --boundary-margin-um 250 > $OUT.prepare.log
tail -n 12 $OUT.prepare.log | head -n 3

echo "== 3. NetFlowV2 flow solve"
rm -rf $OUT/netflow && mkdir -p $OUT/netflow
cp $OUT/netflow_ready/network_ready.dat $OUT/netflow/Network.dat
cp data/leaf_ACJ01/netflow_may/RheolParams.dat data/leaf_ACJ01/netflow_may/ContourParams.dat $OUT/netflow/
(cd $OUT/netflow && ../../../solvers/netflow > netflow_stdout.txt)
sed -n 2,3p $OUT/netflow/Current/Results.out

echo "== 4. Oxygen screening solver"
python src/solver/oxy_sim_mat.py --config configs/demo_configuration.json > $OUT.oxygen.log
python - <<'PY'
import pandas as pd
s = pd.read_csv("results/demo/oxygen_single_run/oxygenation_summary.csv").iloc[0]
print(f"median tissue pO2 {s.tissue_pO2_median_mmHg:.1f} mmHg, "
      f"tissue >= 20 mmHg {s.frac_tissue_ge_20mmHg_pct:.1f}%, "
      f"above inlet: {s.nonphysical_tissue_gt_inlet}")
PY
echo "== Done. Outputs in $OUT/"
