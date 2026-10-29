"""Make a small synthetic leaf-vein network in LeafVeinCNN .mat format.

The real LeafVeinCNN export was lost. This file lets anyone run every stage of
the pipeline (preprocessing, NetFlowV2, oxygen solver) without it.

The network is a 300 x 300 px grid of veins with one diagonal vein, so it has
loops like a real reticulated leaf. The file is written as HDF5 (the MATLAB
v7.3 layout LeafVeinCNN uses) with the fields the pipeline reads:
bwFullWidth, skeleton, MicronPerPixel, widthCalibrated and edgelistCenter
(one centreline polyline per vein between junctions, 1-based row/col).

Run from the repository root:

    python src/demo/make_synthetic_mat.py
"""
from pathlib import Path

import h5py
import numpy as np
from skimage.draw import line
from skimage.morphology import dilation, disk

OUT = Path("data/demo/synthetic_leaf.mat")
H = W = 300
UM_PER_PX = 9.62          # same calibration as the real images
SPACING = 40              # px between parallel veins
HALF_WIDTH_PX = 2         # veins are 5 px (about 48 um) wide

rows = list(range(20, H, SPACING))
cols = list(range(20, W, SPACING))

# Centreline polylines between neighbouring junctions (graph edges).
polylines = []
for r in rows:                                   # horizontal veins, split at each crossing
    stops = [5] + cols + [W - 6]
    for c0, c1 in zip(stops[:-1], stops[1:]):
        rr, cc = line(r, c0, r, c1)
        polylines.append(np.stack([rr, cc], axis=1))
for c in cols:                                   # vertical veins
    stops = [5] + rows + [H - 6]
    for r0, r1 in zip(stops[:-1], stops[1:]):
        rr, cc = line(r0, c, r1, c)
        polylines.append(np.stack([rr, cc], axis=1))
rr, cc = line(5, 5, H - 6, W - 6)                # one diagonal vein
polylines.append(np.stack([rr, cc], axis=1))

skeleton = np.zeros((H, W), bool)
for p in polylines:
    skeleton[p[:, 0], p[:, 1]] = True
bw = dilation(skeleton, disk(HALF_WIDTH_PX))
width_px = np.where(bw, 2 * HALF_WIDTH_PX + 1, 0).astype(float)

OUT.parent.mkdir(parents=True, exist_ok=True)
with h5py.File(OUT, "w") as f:
    f["bwFullWidth"] = bw.astype(np.uint8)
    f["skeleton"] = skeleton.astype(np.uint8)
    f["MicronPerPixel"] = np.array([[UM_PER_PX]])
    f["widthCalibrated"] = width_px
    refs_group = f.create_group("#refs#")
    refs = []
    for i, p in enumerate(polylines):
        ds = refs_group.create_dataset(f"edge{i}", data=(p + 1).astype(float))  # 1-based like MATLAB
        refs.append(ds.ref)
    f.create_dataset("edgelistCenter", data=np.array(refs, dtype=h5py.ref_dtype).reshape(1, -1))

print(f"Saved {OUT}: {int(bw.sum())} vein pixels, {len(polylines)} centreline edges")
